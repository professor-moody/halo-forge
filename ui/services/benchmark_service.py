"""
Benchmark Service - For Reporting

Launches benchmark jobs for **model evaluation** (not training).
This is for comparing trained models to published results.

Uses community tools where available:
- VLMEvalKit for VLM benchmarks
- Native pass@k for code benchmarks

For training verification (RAFT loop), see halo_forge.rlvr.verifiers.

Supports Code, VLM, Audio, and Agentic benchmark types.
"""

import asyncio
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
from collections import deque
from enum import Enum

from .event_bus import (
    get_event_bus,
    Event,
    EventType,
    build_transition_payload,
)


class BenchmarkType(Enum):
    """Available benchmark types."""
    CODE = "code"
    VLM = "vlm"
    AUDIO = "audio"
    AGENTIC = "agentic"


@dataclass
class BenchmarkPreset:
    """Preset configuration for a benchmark."""
    name: str
    type: BenchmarkType
    dataset: str
    description: str
    default_limit: int = 500
    cli_args: dict = field(default_factory=dict)


# Preset benchmarks for each type
CODE_PRESETS = [
    # Python benchmarks
    BenchmarkPreset(
        name="HumanEval",
        type=BenchmarkType.CODE,
        dataset="humaneval",
        description="Python code generation benchmark (164 problems)",
        default_limit=164,
    ),
    BenchmarkPreset(
        name="MBPP",
        type=BenchmarkType.CODE,
        dataset="mbpp",
        description="Mostly Basic Python Problems (974 problems)",
        default_limit=500,
    ),
    BenchmarkPreset(
        name="LiveCodeBench",
        type=BenchmarkType.CODE,
        dataset="livecodebench",
        description="Competitive programming from recent contests",
        default_limit=200,
    ),
    # C++ benchmarks (using internal verifiers)
    BenchmarkPreset(
        name="C++ (Native)",
        type=BenchmarkType.CODE,
        dataset="cpp",
        description="C++ code generation with GCC verification",
        default_limit=16,
        cli_args={"language": "cpp", "verifier": "gcc"},
    ),
    BenchmarkPreset(
        name="C++ (Windows)",
        type=BenchmarkType.CODE,
        dataset="cpp",
        description="C++ code generation with MinGW (Windows cross-compile)",
        default_limit=16,
        cli_args={"language": "cpp", "verifier": "mingw"},
    ),
    # Rust benchmarks
    BenchmarkPreset(
        name="Rust (Native)",
        type=BenchmarkType.CODE,
        dataset="rust",
        description="Rust code generation with cargo verification",
        default_limit=10,
        cli_args={"language": "rust", "verifier": "rust"},
    ),
    # Go benchmarks
    BenchmarkPreset(
        name="Go (Native)",
        type=BenchmarkType.CODE,
        dataset="go",
        description="Go code generation with go build verification",
        default_limit=10,
        cli_args={"language": "go", "verifier": "go"},
    ),
]

VLM_PRESETS = [
    BenchmarkPreset(
        name="TextVQA",
        type=BenchmarkType.VLM,
        dataset="textvqa",
        description="Visual question answering with text in images",
        default_limit=500,
    ),
    BenchmarkPreset(
        name="DocVQA",
        type=BenchmarkType.VLM,
        dataset="docvqa",
        description="Document understanding and QA",
        default_limit=500,
    ),
    BenchmarkPreset(
        name="ChartQA",
        type=BenchmarkType.VLM,
        dataset="chartqa",
        description="Chart understanding and reasoning",
        default_limit=500,
    ),
]

AUDIO_PRESETS = [
    BenchmarkPreset(
        name="LibriSpeech",
        type=BenchmarkType.AUDIO,
        dataset="librispeech",
        description="Speech recognition benchmark",
        default_limit=500,
        cli_args={"task": "asr"},
    ),
    BenchmarkPreset(
        name="CommonVoice",
        type=BenchmarkType.AUDIO,
        dataset="common_voice",
        description="Multi-language speech recognition",
        default_limit=500,
        cli_args={"task": "asr"},
    ),
]

AGENTIC_PRESETS = [
    BenchmarkPreset(
        name="xLAM Function Calling",
        type=BenchmarkType.AGENTIC,
        dataset="xlam",
        description="Function/tool calling benchmark",
        default_limit=500,
    ),
]

ALL_PRESETS = CODE_PRESETS + VLM_PRESETS + AUDIO_PRESETS + AGENTIC_PRESETS

BENCHMARK_DATASET_ALIASES: dict[BenchmarkType, dict[str, str]] = {
    BenchmarkType.AUDIO: {
        "commonvoice": "common_voice",
    },
}


def get_presets_for_type(benchmark_type: BenchmarkType) -> list[BenchmarkPreset]:
    """Get all presets for a benchmark type."""
    return [p for p in ALL_PRESETS if p.type == benchmark_type]


class BenchmarkService:
    """
    Service for launching and managing benchmark processes.
    
    This service:
    - Spawns benchmark as subprocess with proper env vars
    - Streams stdout/stderr
    - Updates job state in real-time
    - Handles graceful termination
    
    Usage:
        from ui.state import state
        service = BenchmarkService(state)
        
        job_id = await service.launch_benchmark(
            model="Qwen/Qwen2.5-Coder-3B-Instruct",
            benchmark_type=BenchmarkType.CODE,
            benchmark_name="humaneval",
            limit=164,
        )
    """
    
    def __init__(self, state):
        """
        Initialize benchmark service.
        
        Args:
            state: AppState instance for job tracking
        """
        self.state = state
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

    @staticmethod
    def _validate_required_text(value: Optional[str], field_name: str) -> str:
        """Validate a required string-like benchmark launch field."""
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{field_name} is required")
        return text

    def _validate_launch_payload(
        self,
        model: str,
        benchmark_name: str,
        limit: Optional[int],
        samples_per_prompt: int,
    ) -> tuple[str, str]:
        """Validate benchmark launch inputs before creating a job."""
        model = self._validate_required_text(model, "model")
        benchmark_name = self._validate_required_text(benchmark_name, "benchmark_name")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0 when provided")
        if samples_per_prompt <= 0:
            raise ValueError("samples_per_prompt must be greater than 0")
        return model, benchmark_name

    def _canonicalize_benchmark_name(
        self,
        benchmark_type: BenchmarkType,
        benchmark_name: str,
    ) -> str:
        """Normalize benchmark aliases and validate dataset-backed benchmark names."""
        normalized = (benchmark_name or "").strip().lower()
        alias_map = BENCHMARK_DATASET_ALIASES.get(benchmark_type, {})
        canonical = alias_map.get(normalized, normalized)

        if canonical != normalized:
            print(f"Normalized benchmark alias '{benchmark_name}' -> '{canonical}'")

        if benchmark_type == BenchmarkType.VLM:
            try:
                from halo_forge.vlm.data import list_vlm_datasets
                supported = set(list_vlm_datasets())
            except Exception as e:
                supported = None
                print(f"Warning: skipping VLM dataset validation ({e})")
            if supported is not None and canonical not in supported:
                raise ValueError(
                    f"Unsupported VLM benchmark dataset '{benchmark_name}' "
                    f"(canonical '{canonical}'). Available: {sorted(supported)}"
                )
        elif benchmark_type == BenchmarkType.AUDIO:
            try:
                from halo_forge.audio.data import list_audio_datasets
                supported = set(list_audio_datasets())
            except Exception as e:
                supported = None
                print(f"Warning: skipping audio dataset validation ({e})")
            if supported is not None and canonical not in supported:
                raise ValueError(
                    f"Unsupported audio benchmark dataset '{benchmark_name}' "
                    f"(canonical '{canonical}'). Available: {sorted(supported)}"
                )
        elif benchmark_type == BenchmarkType.AGENTIC:
            try:
                from halo_forge.agentic.data import list_agentic_datasets
                supported = set(list_agentic_datasets().keys())
            except Exception as e:
                supported = None
                print(f"Warning: skipping agentic dataset validation ({e})")
            if supported is not None and canonical not in supported:
                raise ValueError(
                    f"Unsupported agentic benchmark dataset '{benchmark_name}' "
                    f"(canonical '{canonical}'). Available: {sorted(supported)}"
                )

        return canonical
    
    async def launch_benchmark(
        self,
        model: str,
        benchmark_type: BenchmarkType,
        benchmark_name: str,
        limit: Optional[int] = None,
        output_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        samples_per_prompt: int = 5,
        verifier: Optional[str] = None,
        run_after_compile: bool = True,
        on_log: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> str:
        """
        Launch benchmark as subprocess.
        
        Args:
            model: Model name or path
            benchmark_type: Type of benchmark (CODE, VLM, AUDIO, AGENTIC)
            benchmark_name: Benchmark/dataset name
            limit: Max samples to evaluate
            output_path: Output JSON path for results
            output_dir: Legacy output directory (will append benchmark.json)
            samples_per_prompt: Samples per prompt for pass@k (code benchmarks)
            verifier: Verifier type for code benchmarks
            run_after_compile: MVR mode (run after compile) vs MVP (compile-only)
            on_log: Optional callback for log lines
            **kwargs: Additional CLI arguments
            
        Returns:
            Job ID
        """
        model, benchmark_name = self._validate_launch_payload(
            model=model,
            benchmark_name=benchmark_name,
            limit=limit,
            samples_per_prompt=samples_per_prompt,
        )
        benchmark_name = self._canonicalize_benchmark_name(benchmark_type, benchmark_name)

        # Backward compatibility: older callers may pass output_dir positionally,
        # which now lands in output_path.
        if output_path and output_dir is None and not str(output_path).lower().endswith(".json"):
            output_dir = output_path
            output_path = None

        # Normalize to a concrete output file path.
        if output_path is None:
            if output_dir is not None:
                output_root = Path(output_dir)
            else:
                output_root = Path("results/benchmarks") / f"{Path(model).name}-{benchmark_name}"
            output_path = str(output_root / "benchmark.json")
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create job in state
        job = self.state.create_job(
            job_type="benchmark",
            name=f"Benchmark: {benchmark_name} ({Path(model).name})",
            output_dir=output_file.parent,
        )
        
        # Emit job created event
        created_transition = {
            "from_status": None,
            "to_status": "pending",
            "applied": True,
            "source": "benchmark_service.launch_benchmark",
            "reason": "job_created",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"job_type": "benchmark"},
        }
        get_event_bus().emit_sync(Event(
            type=EventType.JOB_CREATED,
            job_id=job.id,
            data=build_transition_payload(
                created_transition,
                name=job.name,
                type="benchmark",
                benchmark_type=benchmark_type.value,
            ),
        ))
        
        # Build command based on benchmark type
        cmd = self._build_command(
            model=model,
            benchmark_type=benchmark_type,
            benchmark_name=benchmark_name,
            limit=limit,
            output_path=str(output_file),
            samples_per_prompt=samples_per_prompt,
            verifier=verifier,
            run_after_compile=run_after_compile,
            **kwargs
        )
        
        # Launch subprocess
        await self._launch_process(job.id, cmd, on_log)
        
        return job.id
    
    def _build_command(
        self,
        model: str,
        benchmark_type: BenchmarkType,
        benchmark_name: str,
        limit: Optional[int],
        output_path: str,
        samples_per_prompt: int = 5,
        verifier: Optional[str] = None,
        run_after_compile: bool = True,
        **kwargs
    ) -> list[str]:
        """Build CLI command for benchmark type."""
        benchmark_name = self._canonicalize_benchmark_name(benchmark_type, benchmark_name)
        
        if benchmark_type == BenchmarkType.CODE:
            cmd = [
                sys.executable, "-m", "halo_forge.cli", "benchmark", "eval",
                "--model", model,
                "--benchmark", benchmark_name,
                "--output", output_path,
                "--samples-per-prompt", str(samples_per_prompt),
            ]
            if limit:
                cmd.extend(["--limit", str(limit)])
            # Add verification mode (MVR = run after compile, MVP = compile-only)
            if run_after_compile:
                cmd.append("--run-after-compile")
            # Add verifier if specified for compiled languages
            if verifier and verifier not in ('humaneval', 'mbpp'):
                cmd.extend(["--verifier", verifier])
        
        elif benchmark_type == BenchmarkType.VLM:
            cmd = [
                sys.executable, "-m", "halo_forge.cli", "vlm", "benchmark",
                "--model", model,
                "--dataset", benchmark_name,
                "--output", output_path,
            ]
            if limit:
                cmd.extend(["--limit", str(limit)])
        
        elif benchmark_type == BenchmarkType.AUDIO:
            cmd = [
                sys.executable, "-m", "halo_forge.cli", "audio", "benchmark",
                "--model", model,
                "--dataset", benchmark_name,
                "--output", output_path,
            ]
            if limit:
                cmd.extend(["--limit", str(limit)])
            # Add task from kwargs or default
            task = kwargs.pop("task", "asr")
            cmd.extend(["--task", task])
        
        elif benchmark_type == BenchmarkType.AGENTIC:
            cmd = [
                sys.executable, "-m", "halo_forge.cli", "agentic", "benchmark",
                "--model", model,
                "--dataset", benchmark_name,
                "--output", output_path,
            ]
            if limit:
                cmd.extend(["--limit", str(limit)])
        
        else:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
        # Add any extra arguments
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return cmd
    
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
        
        # Set up log buffer
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
        transitioned = self.state.update_job_status(
            job_id,
            "running",
            source="benchmark_service._launch_process",
            reason="process_started",
            metadata={"command": cmd},
        )

        # Emit job started event
        if transitioned:
            transition = self.state.get_last_transition(job_id)
            await get_event_bus().emit(Event(
                type=EventType.JOB_STARTED,
                job_id=job_id,
                data=build_transition_payload(
                    transition,
                    name=job.name,
                    type="benchmark",
                ),
            ))
        
        # Start log streaming task
        asyncio.create_task(self._stream_logs(job_id))
    
    async def _stream_logs(self, job_id: str):
        """Stream subprocess output."""
        job = self.state.get_job(job_id)
        if not job or not job.process:
            return
        
        log_buffer = self._log_buffers.get(job_id, deque(maxlen=1000))
        callbacks = self._callbacks.get(job_id, [])
        event_bus = get_event_bus()
        
        try:
            async for line_bytes in job.process.stdout:
                line = line_bytes.decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                
                timestamp = datetime.now().isoformat()
                
                # Store log line
                log_buffer.append({
                    'timestamp': timestamp,
                    'line': line,
                })
                
                # Emit log line event
                await event_bus.emit(Event(
                    type=EventType.LOG_LINE,
                    job_id=job_id,
                    data={'line': line, 'timestamp': timestamp}
                ))
                
                # Call legacy callbacks
                for callback in callbacks:
                    try:
                        callback(line)
                    except Exception:
                        pass
        
        except Exception as e:
            job.error_message = str(e)
        
        # Process completed
        return_code = await job.process.wait()

        if job.stop_requested:
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="benchmark_service._stream_logs",
                reason="stop_requested",
                metadata={"return_code": return_code},
            )
            if transitioned:
                transition = self.state.get_last_transition(job_id)
                await event_bus.emit(Event(
                    type=EventType.JOB_STOPPED,
                    job_id=job_id,
                    data=build_transition_payload(
                        transition,
                        return_code=return_code,
                    ),
                ))
        elif return_code == 0:
            transitioned = self.state.update_job_status(
                job_id,
                "completed",
                source="benchmark_service._stream_logs",
                reason="process_exit_ok",
                metadata={"return_code": return_code},
            )
            if not transitioned:
                return
            transition = self.state.get_last_transition(job_id)
            await event_bus.emit(Event(
                type=EventType.JOB_COMPLETED,
                job_id=job_id,
                data=build_transition_payload(
                    transition,
                    return_code=return_code,
                ),
            ))
        elif return_code == -signal.SIGTERM or return_code == -signal.SIGKILL:
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="benchmark_service._stream_logs",
                reason="terminated_signal",
                metadata={"return_code": return_code},
            )
            if not transitioned:
                return
            transition = self.state.get_last_transition(job_id)
            await event_bus.emit(Event(
                type=EventType.JOB_STOPPED,
                job_id=job_id,
                data=build_transition_payload(
                    transition,
                    return_code=return_code,
                ),
            ))
        else:
            job.error_message = f"Process exited with code {return_code}"
            transitioned = self.state.update_job_status(
                job_id,
                "failed",
                source="benchmark_service._stream_logs",
                reason="process_exit_error",
                metadata={"return_code": return_code, "error": job.error_message},
            )
            if not transitioned:
                return
            transition = self.state.get_last_transition(job_id)
            await event_bus.emit(Event(
                type=EventType.JOB_FAILED,
                job_id=job_id,
                data=build_transition_payload(
                    transition,
                    return_code=return_code,
                    error=job.error_message,
                ),
            ))
    
    async def stop_job(self, job_id: str, timeout: float = 30.0) -> bool:
        """
        Stop a running benchmark job.
        
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

        job.stop_requested = True
        
        # Send SIGTERM first
        try:
            job.process.terminate()
        except ProcessLookupError:
            self.state.update_job_status(
                job_id,
                "stopped",
                source="benchmark_service.stop_job",
                reason="process_missing",
            )
            return True
        
        try:
            await asyncio.wait_for(job.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                job.process.kill()
                await job.process.wait()
            except ProcessLookupError:
                pass
        
        self.state.update_job_status(
            job_id,
            "stopped",
            source="benchmark_service.stop_job",
            reason="stop_completed",
        )
        return True
    
    def get_logs(self, job_id: str, last_n: Optional[int] = None) -> list[dict]:
        """Get log entries for a job."""
        buffer = self._log_buffers.get(job_id, deque())
        logs = list(buffer)
        
        if last_n is not None:
            logs = logs[-last_n:]
        
        return logs


# Singleton instance
_benchmark_service: Optional[BenchmarkService] = None


def get_benchmark_service(state=None) -> BenchmarkService:
    """Get or create the benchmark service singleton."""
    global _benchmark_service
    if _benchmark_service is None:
        if state is None:
            from ui.state import state as app_state
            state = app_state
        _benchmark_service = BenchmarkService(state)
    return _benchmark_service
