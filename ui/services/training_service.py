"""
Training Service

Manages training job execution via subprocess, log streaming, and job control.
This is the bridge between the UI and actual training processes.
"""

import asyncio
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from collections import deque

from .metrics_parser import MetricsParser, ParsedMetrics

# Import notification helpers (only used when UI is running)
try:
    from ui.components.notifications import notify_checkpoint_saved
    HAS_UI_NOTIFICATIONS = True
except ImportError:
    HAS_UI_NOTIFICATIONS = False


@dataclass
class TrainingMetrics:
    """Current training metrics."""
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: float = 0.0
    step: int = 0
    total_steps: int = 0
    cycle: int = 0  # RAFT
    total_cycles: int = 0  # RAFT
    compile_rate: Optional[float] = None  # RAFT
    grad_norm: Optional[float] = None


class TrainingService:
    """
    Service for launching and managing training processes.
    
    This service:
    - Spawns training as subprocess with proper env vars
    - Streams stdout/stderr and parses metrics
    - Updates job state in real-time
    - Handles graceful and forced termination
    
    Usage:
        from ui.state import state
        service = TrainingService(state)
        
        job_id = await service.launch_sft(
            model="Qwen/Qwen2.5-Coder-3B-Instruct",
            dataset="humaneval",
            output_dir="./outputs/sft",
            epochs=3,
        )
        
        # Later...
        await service.stop_job(job_id)
    """
    
    def __init__(self, state):
        """
        Initialize training service.
        
        Args:
            state: AppState instance for job tracking
        """
        self.state = state
        self._parsers: dict[str, MetricsParser] = {}
        self._log_buffers: dict[str, deque] = {}
        self._callbacks: dict[str, list[Callable]] = {}
    
    def _get_strix_halo_env(self) -> dict[str, str]:
        """Get environment variables optimized for AMD Strix Halo."""
        env = os.environ.copy()
        
        # GPU architecture
        env.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.5.1')
        env.setdefault('PYTORCH_ROCM_ARCH', 'gfx1151')
        env.setdefault('HIP_VISIBLE_DEVICES', '0')
        
        # Memory management for unified memory
        env.setdefault(
            'PYTORCH_HIP_ALLOC_CONF',
            'backend:native,expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512'
        )
        
        # Stability settings
        env.setdefault('HSA_ENABLE_SDMA', '0')
        
        # Dataloader settings (critical for unified memory)
        env.setdefault('OMP_NUM_THREADS', '1')
        
        return env
    
    async def launch_sft(
        self,
        model: str,
        dataset: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        use_lora: bool = True,
        lora_rank: int = 32,
        on_log: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> str:
        """
        Launch SFT training as subprocess.
        
        Args:
            model: Model name or path
            dataset: Dataset name or path
            output_dir: Output directory for checkpoints
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            use_lora: Whether to use LoRA
            lora_rank: LoRA rank if using LoRA
            on_log: Optional callback for log lines
            **kwargs: Additional CLI arguments
            
        Returns:
            Job ID
        """
        # Create job in state
        job = self.state.create_job(
            job_type="sft",
            name=f"SFT: {Path(model).name} on {dataset}",
            output_dir=Path(output_dir),
        )
        job.total_epochs = epochs
        
        # Build command
        cmd = [
            "python", "-m", "halo_forge.cli", "sft", "train",
            "--model", model,
            "--dataset", dataset,
            "--output", output_dir,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
        ]
        
        if use_lora:
            cmd.extend(["--lora-rank", str(lora_rank)])
        
        # Add any extra arguments
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        # Launch subprocess
        await self._launch_process(job.id, cmd, on_log)
        
        return job.id
    
    async def launch_raft(
        self,
        model: str,
        prompts: str,
        output_dir: str,
        verifier: str = "execution",
        cycles: int = 5,
        samples_per_prompt: int = 4,
        temperature: float = 0.7,
        keep_percent: float = 0.25,
        reward_threshold: float = 0.5,
        on_log: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> str:
        """
        Launch RAFT training as subprocess.
        
        Args:
            model: Model name or path
            prompts: Path to prompts file
            output_dir: Output directory
            verifier: Verifier type
            cycles: Number of RAFT cycles
            samples_per_prompt: Samples per prompt
            temperature: Sampling temperature
            keep_percent: Percentage of samples to keep
            reward_threshold: Minimum reward threshold
            on_log: Optional callback for log lines
            **kwargs: Additional CLI arguments
            
        Returns:
            Job ID
        """
        # Create job in state
        job = self.state.create_job(
            job_type="raft",
            name=f"RAFT: {Path(model).name}",
            output_dir=Path(output_dir),
        )
        job.total_cycles = cycles
        
        # Build command
        cmd = [
            "python", "-m", "halo_forge.cli", "raft", "train",
            "--model", model,
            "--prompts", prompts,
            "--output", output_dir,
            "--verifier", verifier,
            "--cycles", str(cycles),
            "--samples-per-prompt", str(samples_per_prompt),
            "--temperature", str(temperature),
            "--keep-percent", str(keep_percent),
            "--reward-threshold", str(reward_threshold),
        ]
        
        # Add any extra arguments
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        # Launch subprocess
        await self._launch_process(job.id, cmd, on_log)
        
        return job.id
    
    async def _launch_process(
        self,
        job_id: str,
        cmd: list[str],
        on_log: Optional[Callable[[str], None]] = None,
    ):
        """Launch subprocess and start log streaming."""
        job = self.state.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Set up parser and log buffer
        self._parsers[job_id] = MetricsParser()
        self._log_buffers[job_id] = deque(maxlen=1000)
        
        if on_log:
            if job_id not in self._callbacks:
                self._callbacks[job_id] = []
            self._callbacks[job_id].append(on_log)
        
        # Get optimized environment
        env = self._get_strix_halo_env()
        
        # Launch subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=Path.cwd(),
        )
        
        job.process = process
        job.started_at = datetime.now()
        self.state.update_job_status(job_id, "running")
        
        # Start log streaming task
        asyncio.create_task(self._stream_logs(job_id))
    
    async def _stream_logs(self, job_id: str):
        """Stream subprocess output and parse metrics."""
        job = self.state.get_job(job_id)
        if not job or not job.process:
            return
        
        parser = self._parsers.get(job_id)
        log_buffer = self._log_buffers.get(job_id, deque(maxlen=1000))
        callbacks = self._callbacks.get(job_id, [])
        
        try:
            async for line_bytes in job.process.stdout:
                line = line_bytes.decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                
                # Store log line
                log_buffer.append({
                    'timestamp': datetime.now().isoformat(),
                    'line': line,
                })
                
                # Call callbacks
                for callback in callbacks:
                    try:
                        callback(line)
                    except Exception:
                        pass
                
                # Parse metrics
                if parser:
                    metrics = parser.parse_line(line)
                    if metrics:
                        self._update_job_metrics(job_id, metrics)
                
                # Detect checkpoint saves and notify
                line_lower = line.lower()
                if ('checkpoint' in line_lower and 'saved' in line_lower) or \
                   ('saving' in line_lower and 'checkpoint' in line_lower):
                    if HAS_UI_NOTIFICATIONS:
                        # Extract checkpoint path if available, else use output dir
                        checkpoint_path = job.output_dir if job.output_dir else "checkpoint"
                        notify_checkpoint_saved(str(checkpoint_path))
        
        except Exception as e:
            job.error_message = str(e)
        
        # Process completed
        return_code = await job.process.wait()
        
        if return_code == 0:
            self.state.update_job_status(job_id, "completed")
        elif return_code == -signal.SIGTERM or return_code == -signal.SIGKILL:
            self.state.update_job_status(job_id, "stopped")
        else:
            self.state.update_job_status(job_id, "failed")
            job.error_message = f"Process exited with code {return_code}"
    
    def _update_job_metrics(self, job_id: str, metrics: ParsedMetrics):
        """Update job state with parsed metrics."""
        job = self.state.get_job(job_id)
        if not job:
            return
        
        if metrics.loss is not None:
            job.latest_loss = metrics.loss
            step = metrics.step or job.current_step
            self.state.add_metric(job_id, 'loss', step, metrics.loss)
        
        if metrics.learning_rate is not None:
            job.latest_lr = metrics.learning_rate
        
        if metrics.epoch is not None:
            job.current_epoch = int(metrics.epoch)
        
        if metrics.step is not None:
            job.current_step = metrics.step
        
        if metrics.total_steps is not None:
            job.total_steps = metrics.total_steps
        
        if metrics.cycle is not None:
            job.current_cycle = metrics.cycle
        
        if metrics.total_cycles is not None:
            job.total_cycles = metrics.total_cycles
        
        if metrics.compile_rate is not None:
            job.verification_rate = metrics.compile_rate
        
        if metrics.grad_norm is not None:
            job.latest_grad_norm = metrics.grad_norm
            self.state.add_metric(job_id, 'grad_norm', job.current_step, metrics.grad_norm)
    
    async def stop_job(self, job_id: str, timeout: float = 30.0) -> bool:
        """
        Stop a running training job.
        
        Sends SIGTERM first to allow graceful shutdown (checkpoint saving),
        then SIGKILL if the process doesn't exit within timeout.
        
        Args:
            job_id: Job ID to stop
            timeout: Seconds to wait for graceful shutdown
            
        Returns:
            True if job was stopped, False if not found or not running
        """
        job = self.state.get_job(job_id)
        if not job or not job.process:
            return False
        
        if job.status != "running":
            return False
        
        # Send SIGTERM first (allows checkpoint saving)
        try:
            job.process.terminate()
        except ProcessLookupError:
            # Process already dead
            self.state.update_job_status(job_id, "stopped")
            return True
        
        try:
            # Wait for graceful shutdown
            await asyncio.wait_for(job.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Force kill if still running
            try:
                job.process.kill()
                await job.process.wait()
            except ProcessLookupError:
                pass
        
        self.state.update_job_status(job_id, "stopped")
        return True
    
    def get_logs(self, job_id: str, last_n: Optional[int] = None) -> list[dict]:
        """
        Get log entries for a job.
        
        Args:
            job_id: Job ID
            last_n: Only return last N entries
            
        Returns:
            List of log entries with timestamp and line
        """
        buffer = self._log_buffers.get(job_id, deque())
        logs = list(buffer)
        
        if last_n is not None:
            logs = logs[-last_n:]
        
        return logs
    
    def get_metrics(self, job_id: str) -> TrainingMetrics:
        """
        Get current metrics for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            TrainingMetrics with current values
        """
        job = self.state.get_job(job_id)
        if not job:
            return TrainingMetrics()
        
        return TrainingMetrics(
            loss=job.latest_loss,
            learning_rate=job.latest_lr,
            epoch=float(job.current_epoch),
            step=job.current_step,
            total_steps=job.total_steps,
            cycle=job.current_cycle,
            total_cycles=job.total_cycles,
            compile_rate=job.verification_rate,
            grad_norm=job.latest_grad_norm,
        )
    
    def add_log_callback(self, job_id: str, callback: Callable[[str], None]):
        """Add a callback for log lines."""
        if job_id not in self._callbacks:
            self._callbacks[job_id] = []
        self._callbacks[job_id].append(callback)
    
    def remove_log_callback(self, job_id: str, callback: Callable[[str], None]):
        """Remove a log callback."""
        if job_id in self._callbacks:
            try:
                self._callbacks[job_id].remove(callback)
            except ValueError:
                pass
