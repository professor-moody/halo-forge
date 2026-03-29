"""
Benchmark Service - For Reporting

Launches benchmark jobs for **model evaluation** (not training).
This is for comparing trained models to published results.

Uses community tools where available:
- VLMEvalKit for VLM benchmarks
- Native pass@k for code benchmarks

For training verification (RAFT loop), see halo_forge.rlvr.verifiers.

Supports Code, VLM, Audio, Reasoning, and Agentic benchmark types.
"""

import asyncio
import os
import signal
import sys
import re
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
from .launch_contracts import (
    BENCHMARK_LAUNCH_CONTRACT,
    validate_launch_payload,
)
from .launch_context import (
    persist_launch_context,
    read_launch_context,
)

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class BenchmarkType(Enum):
    """Available benchmark types."""
    CODE = "code"
    VLM = "vlm"
    AUDIO = "audio"
    REASONING = "reasoning"
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

REASONING_PRESETS = [
    BenchmarkPreset(
        name="GSM8K",
        type=BenchmarkType.REASONING,
        dataset="gsm8k",
        description="Grade-school math reasoning benchmark",
        default_limit=200,
        cli_args={"split": "test"},
    ),
    BenchmarkPreset(
        name="MATH",
        type=BenchmarkType.REASONING,
        dataset="math",
        description="Competition math reasoning benchmark",
        default_limit=200,
        cli_args={"split": "test"},
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

ALL_PRESETS = CODE_PRESETS + VLM_PRESETS + AUDIO_PRESETS + REASONING_PRESETS + AGENTIC_PRESETS

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

    def _validate_launch_payload(
        self,
        model: str,
        benchmark_name: str,
        limit: Optional[int],
        samples_per_prompt: int,
    ) -> tuple[str, str]:
        """Validate benchmark launch inputs before creating a job."""
        normalized = validate_launch_payload(
            {
                "model": model,
                "benchmark_name": benchmark_name,
                "samples_per_prompt": samples_per_prompt,
            },
            BENCHMARK_LAUNCH_CONTRACT,
        )
        model = normalized["model"]
        benchmark_name = normalized["benchmark_name"]
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0 when provided")
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
        elif benchmark_type == BenchmarkType.REASONING:
            try:
                from halo_forge.reasoning.data import list_math_datasets

                supported = set(list_math_datasets())
            except Exception as e:
                supported = None
                print(f"Warning: skipping reasoning dataset validation ({e})")
            if supported is not None and canonical not in supported:
                raise ValueError(
                    f"Unsupported reasoning benchmark dataset '{benchmark_name}' "
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

    def _build_lifecycle_metadata(
        self,
        *,
        origin_job_id: Optional[str],
        relaunch: bool,
        launch_context_file: Optional[Path],
        resume_strategy: Optional[str],
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if origin_job_id:
            metadata["origin_job_id"] = origin_job_id
        if relaunch:
            metadata["relaunch"] = True
        if launch_context_file:
            metadata["launch_context_file"] = str(launch_context_file)
        if resume_strategy:
            metadata["resume_strategy"] = resume_strategy
        return metadata

    def _event_extra_fields(self, job) -> dict[str, Any]:
        if not job or not job.lifecycle_metadata:
            return {}
        return {
            "origin_job_id": job.lifecycle_metadata.get("origin_job_id"),
            "relaunch": job.lifecycle_metadata.get("relaunch"),
            "launch_context_file": job.lifecycle_metadata.get("launch_context_file"),
            "resume_strategy": job.lifecycle_metadata.get("resume_strategy"),
        }

    def _merge_transition_metadata(self, job, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if job and job.lifecycle_metadata:
            metadata.update(job.lifecycle_metadata)
        if extra:
            metadata.update(extra)
        return metadata
    
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
        source_ui_page: str = "/benchmark",
        origin_job_id: Optional[str] = None,
        relaunch: bool = False,
        resume_strategy: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Launch benchmark as subprocess.
        
        Args:
            model: Model name or path
            benchmark_type: Type of benchmark (CODE, VLM, AUDIO, REASONING, AGENTIC)
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
        if limit is not None and int(limit) > 0:
            job.total_steps = int(limit)
        
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

        launch_args = {
            "model": model,
            "benchmark_type": benchmark_type.value,
            "benchmark_name": benchmark_name,
            "limit": limit,
            "output_path": str(output_file),
            "output_dir": str(output_file.parent),
            "samples_per_prompt": samples_per_prompt,
            "verifier": verifier,
            "run_after_compile": run_after_compile,
        }
        launch_args.update({k: v for k, v in kwargs.items() if v is not None})
        launch_context_file = None
        try:
            launch_context_file = persist_launch_context(
                output_dir=output_file.parent,
                job_type="benchmark",
                service="benchmark",
                source_ui_page=source_ui_page,
                command=cmd,
                args=launch_args,
                relaunch_capabilities={
                    "can_relaunch": True,
                    "can_clone": True,
                    "can_resume_latest": False,
                },
            )
        except Exception as e:
            print(f"[BenchmarkService] Failed to persist launch context: {e}")
        lifecycle_metadata = self._build_lifecycle_metadata(
            origin_job_id=origin_job_id,
            relaunch=relaunch,
            launch_context_file=launch_context_file,
            resume_strategy=resume_strategy,
        )
        job.launch_context_file = launch_context_file
        job.launch_args = launch_args
        job.lifecycle_metadata = lifecycle_metadata

        # Emit job created event
        created_transition = {
            "from_status": None,
            "to_status": "pending",
            "applied": True,
            "source": "benchmark_service.launch_benchmark",
            "reason": "job_created",
            "timestamp": datetime.now().isoformat(),
            "metadata": self._merge_transition_metadata(
                job,
                {"job_type": "benchmark"},
            ),
        }
        get_event_bus().emit_sync(Event(
            type=EventType.JOB_CREATED,
            job_id=job.id,
            data=build_transition_payload(
                created_transition,
                name=job.name,
                type="benchmark",
                benchmark_type=benchmark_type.value,
                **self._event_extra_fields(job),
            ),
        ))
        
        # Launch subprocess
        await self._launch_process(job.id, cmd, on_log)
        
        return job.id

    async def relaunch_from_context(
        self,
        launch_context_file: str | Path,
        *,
        origin_job_id: Optional[str] = None,
        source_ui_page: str = "/monitor",
        on_log: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Relaunch a benchmark job from persisted launch context."""
        context = read_launch_context(launch_context_file)
        if context.service != "benchmark":
            raise ValueError("launch context does not belong to benchmark service")
        if context.job_type != "benchmark":
            raise ValueError(f"Unsupported benchmark relaunch job_type: {context.job_type}")

        args = dict(context.args)
        benchmark_type_value = str(args.pop("benchmark_type", "")).strip().lower()
        try:
            benchmark_type = BenchmarkType(benchmark_type_value)
        except ValueError as e:
            raise ValueError(f"Invalid benchmark_type in launch context: {benchmark_type_value}") from e

        return await self.launch_benchmark(
            benchmark_type=benchmark_type,
            on_log=on_log,
            source_ui_page=source_ui_page,
            origin_job_id=origin_job_id,
            relaunch=True,
            resume_strategy="relaunch",
            **args,
        )
    
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
        
        elif benchmark_type == BenchmarkType.REASONING:
            split = kwargs.pop("split", "test")
            cmd = [
                sys.executable, "-m", "halo_forge.cli", "reasoning", "benchmark",
                "--model", model,
                "--dataset", benchmark_name,
                "--split", str(split),
                "--output", output_path,
            ]
            if limit:
                cmd.extend(["--limit", str(limit)])

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
            metadata=self._merge_transition_metadata(
                job,
                {"command": cmd},
            ),
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
                    **self._event_extra_fields(job),
                ),
            ))

        # Set up persistent log file path for monitor reload continuity.
        if job.output_dir:
            try:
                log_path = Path(job.output_dir) / f"{job_id}_benchmark.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                job.log_file_path = log_path
            except Exception:
                pass
        
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
        log_file = None
        if job.log_file_path:
            try:
                log_file = open(job.log_file_path, "a", encoding="utf-8")
            except Exception:
                log_file = None
        
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

                if log_file:
                    try:
                        log_file.write(f"[{timestamp}] {line}\n")
                        log_file.flush()
                    except Exception:
                        pass
                
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

                # Parse lightweight benchmark progress/metrics from log lines.
                normalized_line = ANSI_ESCAPE_RE.sub("", line)
                line_lower = normalized_line.lower()
                metrics_updated = False
                if "samples evaluated" in line_lower:
                    match = re.search(r"samples evaluated:\s*(\d+)", normalized_line, re.IGNORECASE)
                    if match:
                        job.current_step = int(match.group(1))
                        job.lifecycle_metadata["benchmark_samples_evaluated"] = job.current_step
                        if job.total_steps == 0 and job.current_step > 0:
                            job.total_steps = job.current_step
                        metrics_updated = True
                elif "total_prompts" in line_lower:
                    match = re.search(r"total_prompts:\s*(\d+)", normalized_line, re.IGNORECASE)
                    if match:
                        parsed_total = int(match.group(1))
                        if parsed_total > 0:
                            job.total_steps = parsed_total
                            job.lifecycle_metadata["benchmark_total_prompts"] = parsed_total
                            metrics_updated = True
                elif "pass_at_1" in line_lower:
                    match = re.search(r"pass_at_1:\s*([0-9]*\.?[0-9]+)", normalized_line, re.IGNORECASE)
                    if match:
                        job.verification_rate = float(match.group(1))
                        job.lifecycle_metadata["benchmark_pass_at_1"] = job.verification_rate
                        if "benchmark_pass_rate" not in job.lifecycle_metadata:
                            job.lifecycle_metadata["benchmark_pass_rate"] = job.verification_rate
                        metrics_updated = True
                elif "pass@1" in line_lower:
                    match = re.search(r"pass@1:\s*([0-9]*\.?[0-9]+)", normalized_line, re.IGNORECASE)
                    if match:
                        job.verification_rate = float(match.group(1))
                        job.lifecycle_metadata["benchmark_pass_at_1"] = job.verification_rate
                        if "benchmark_pass_rate" not in job.lifecycle_metadata:
                            job.lifecycle_metadata["benchmark_pass_rate"] = job.verification_rate
                        metrics_updated = True
                elif "pass_at_5" in line_lower:
                    match = re.search(r"pass_at_5:\s*([0-9]*\.?[0-9]+)", normalized_line, re.IGNORECASE)
                    if match:
                        job.lifecycle_metadata["benchmark_pass_at_5"] = float(match.group(1))
                        metrics_updated = True
                elif "pass_at_10" in line_lower:
                    match = re.search(r"pass_at_10:\s*([0-9]*\.?[0-9]+)", normalized_line, re.IGNORECASE)
                    if match:
                        job.lifecycle_metadata["benchmark_pass_at_10"] = float(match.group(1))
                        metrics_updated = True
                elif "pass_rate" in line_lower:
                    match = re.search(r"pass_rate:\s*([0-9]*\.?[0-9]+)", normalized_line, re.IGNORECASE)
                    if match:
                        job.lifecycle_metadata["benchmark_pass_rate"] = float(match.group(1))
                        metrics_updated = True
                elif "passed:" in line_lower:
                    match = re.search(r"passed:\s*(\d+)", normalized_line, re.IGNORECASE)
                    if match:
                        parsed_passed = int(match.group(1))
                        job.lifecycle_metadata["benchmark_passed"] = parsed_passed
                        if job.current_step == 0:
                            job.current_step = parsed_passed
                        metrics_updated = True

                if metrics_updated:
                    await event_bus.emit(Event(
                        type=EventType.METRICS_UPDATE,
                        job_id=job_id,
                        data={
                            "step": job.current_step,
                            "total_steps": job.total_steps,
                            "compile_rate": job.verification_rate,
                        },
                    ))
        
        except Exception as e:
            job.error_message = str(e)
        finally:
            if log_file:
                try:
                    log_file.close()
                except Exception:
                    pass
        
        # Process completed
        return_code = await job.process.wait()

        if job.stop_requested:
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="benchmark_service._stream_logs",
                reason="stop_requested",
                metadata=self._merge_transition_metadata(
                    job,
                    {"return_code": return_code},
                ),
            )
            if transitioned:
                transition = self.state.get_last_transition(job_id)
                await event_bus.emit(Event(
                    type=EventType.JOB_STOPPED,
                    job_id=job_id,
                    data=build_transition_payload(
                        transition,
                        return_code=return_code,
                        **self._event_extra_fields(job),
                    ),
                ))
        elif return_code == 0:
            transitioned = self.state.update_job_status(
                job_id,
                "completed",
                source="benchmark_service._stream_logs",
                reason="process_exit_ok",
                metadata=self._merge_transition_metadata(
                    job,
                    {"return_code": return_code},
                ),
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
                    **self._event_extra_fields(job),
                ),
            ))
        elif return_code == -signal.SIGTERM or return_code == -signal.SIGKILL:
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="benchmark_service._stream_logs",
                reason="terminated_signal",
                metadata=self._merge_transition_metadata(
                    job,
                    {"return_code": return_code},
                ),
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
                    **self._event_extra_fields(job),
                ),
            ))
        else:
            job.error_message = f"Process exited with code {return_code}"
            transitioned = self.state.update_job_status(
                job_id,
                "failed",
                source="benchmark_service._stream_logs",
                reason="process_exit_error",
                metadata=self._merge_transition_metadata(
                    job,
                    {"return_code": return_code, "error": job.error_message},
                ),
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
                    **self._event_extra_fields(job),
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
        if not job:
            return False
        if job.status in {"stopped", "completed", "failed"}:
            return True
        if not job.process:
            return False
        
        if job.status != "running":
            return job.status in {"stopped", "completed", "failed"}

        job.stop_requested = True
        
        # Send SIGTERM first
        try:
            job.process.terminate()
        except ProcessLookupError:
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="benchmark_service.stop_job",
                reason="process_missing",
                metadata=self._merge_transition_metadata(job),
            )
            if transitioned:
                transition = self.state.get_last_transition(job_id)
                await get_event_bus().emit(Event(
                    type=EventType.JOB_STOPPED,
                    job_id=job_id,
                    data=build_transition_payload(
                        transition,
                        **self._event_extra_fields(job),
                    ),
                ))
            return True
        
        try:
            await asyncio.wait_for(job.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                job.process.kill()
                await job.process.wait()
            except ProcessLookupError:
                pass
        
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
