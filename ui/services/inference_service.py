"""
Inference Service

Manages inference optimize/benchmark job execution via subprocess, log streaming,
and durable launch context for relaunch operations.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .event_bus import Event, EventType, build_transition_payload, get_event_bus
from .launch_context import persist_launch_context, read_launch_context
from .launch_contracts import (
    INFERENCE_BENCHMARK_LAUNCH_CONTRACT,
    INFERENCE_OPTIMIZE_LAUNCH_CONTRACT,
    validate_launch_payload,
)


class InferenceService:
    """Launch and manage inference optimize/benchmark commands."""

    def __init__(self, state):
        self.state = state
        self._log_buffers: dict[str, deque] = {}
        self._callbacks: dict[str, list[Callable]] = {}

    def _get_strix_halo_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
        env.setdefault("PYTORCH_ROCM_ARCH", "gfx1151")
        env.setdefault("HIP_VISIBLE_DEVICES", "0")
        env.setdefault(
            "PYTORCH_HIP_ALLOC_CONF",
            "backend:native,expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512",
        )
        env.setdefault("HSA_ENABLE_SDMA", "0")
        env.setdefault("OMP_NUM_THREADS", "1")
        return env

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

    def _merge_transition_metadata(
        self,
        job,
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if job and job.lifecycle_metadata:
            metadata.update(job.lifecycle_metadata)
        if extra:
            metadata.update(extra)
        return metadata

    def _validate_optimize_payload(
        self,
        *,
        model: str,
        output_dir: str,
        target_precision: str,
        target_latency: float,
    ) -> tuple[str, str, str, float]:
        normalized = validate_launch_payload(
            {
                "model": model,
                "output_dir": output_dir,
                "target_precision": target_precision,
                "target_latency": target_latency,
            },
            INFERENCE_OPTIMIZE_LAUNCH_CONTRACT,
        )
        parsed_target_latency = float(normalized["target_latency"])
        if parsed_target_latency <= 0:
            raise ValueError("target_latency must be > 0")
        precision = str(normalized["target_precision"]).strip().lower()
        if precision not in {"int4", "int8", "fp16"}:
            raise ValueError("target_precision must be one of: int4, int8, fp16")
        return (
            normalized["model"],
            normalized["output_dir"],
            precision,
            parsed_target_latency,
        )

    def _validate_benchmark_payload(
        self,
        *,
        model: str,
        output_dir: str,
        num_prompts: int,
        max_tokens: int,
        warmup: int,
    ) -> tuple[str, str, int, int, int]:
        normalized = validate_launch_payload(
            {
                "model": model,
                "output_dir": output_dir,
                "num_prompts": num_prompts,
                "max_tokens": max_tokens,
                "warmup": warmup,
            },
            INFERENCE_BENCHMARK_LAUNCH_CONTRACT,
        )
        return (
            normalized["model"],
            normalized["output_dir"],
            int(normalized["num_prompts"]),
            int(normalized["max_tokens"]),
            int(normalized["warmup"]),
        )

    def build_optimize_command(
        self,
        *,
        model: str,
        target_precision: str,
        target_latency: float,
        calibration_data: Optional[str],
        output_dir: str,
        dry_run: bool,
    ) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "halo_forge.cli",
            "inference",
            "optimize",
            "--model",
            model,
            "--target-precision",
            target_precision,
            "--target-latency",
            str(target_latency),
            "--output",
            output_dir,
        ]
        if calibration_data:
            cmd.extend(["--calibration-data", calibration_data])
        if dry_run:
            cmd.append("--dry-run")
        return cmd

    def build_benchmark_command(
        self,
        *,
        model: str,
        prompts: Optional[str],
        num_prompts: int,
        max_tokens: int,
        warmup: int,
        measure_memory: bool,
    ) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "halo_forge.cli",
            "inference",
            "benchmark",
            "--model",
            model,
            "--num-prompts",
            str(num_prompts),
            "--max-tokens",
            str(max_tokens),
            "--warmup",
            str(warmup),
        ]
        if prompts:
            cmd.extend(["--prompts", prompts])
        if measure_memory:
            cmd.append("--measure-memory")
        return cmd

    async def launch_optimize(
        self,
        *,
        model: str,
        output_dir: str = "models/optimized",
        target_precision: str = "int4",
        target_latency: float = 50.0,
        calibration_data: Optional[str] = None,
        dry_run: bool = False,
        on_log: Optional[Callable[[str], None]] = None,
        source_ui_page: str = "/inference",
        origin_job_id: Optional[str] = None,
        relaunch: bool = False,
        resume_strategy: Optional[str] = None,
    ) -> str:
        model, output_dir, target_precision, target_latency = self._validate_optimize_payload(
            model=model,
            output_dir=output_dir,
            target_precision=target_precision,
            target_latency=target_latency,
        )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cmd = self.build_optimize_command(
            model=model,
            target_precision=target_precision,
            target_latency=target_latency,
            calibration_data=calibration_data,
            output_dir=output_dir,
            dry_run=dry_run,
        )

        job = self.state.create_job(
            job_type="inference",
            name=f"Inference Optimize: {Path(model).name}",
            output_dir=output_path,
        )

        launch_args = {
            "mode": "optimize",
            "model": model,
            "output_dir": output_dir,
            "target_precision": target_precision,
            "target_latency": target_latency,
            "calibration_data": calibration_data,
            "dry_run": dry_run,
        }
        launch_context_file = None
        try:
            launch_context_file = persist_launch_context(
                output_dir=output_path,
                job_type="inference",
                service="inference",
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
            print(f"[InferenceService] Failed to persist launch context: {e}")

        lifecycle_metadata = self._build_lifecycle_metadata(
            origin_job_id=origin_job_id,
            relaunch=relaunch,
            launch_context_file=launch_context_file,
            resume_strategy=resume_strategy,
        )
        job.launch_context_file = launch_context_file
        job.launch_args = launch_args
        job.lifecycle_metadata = lifecycle_metadata

        created_transition = {
            "from_status": None,
            "to_status": "pending",
            "applied": True,
            "source": "inference_service.launch_optimize",
            "reason": "job_created",
            "timestamp": datetime.now().isoformat(),
            "metadata": self._merge_transition_metadata(
                job,
                {"job_type": "inference", "mode": "optimize"},
            ),
        }
        get_event_bus().emit_sync(
            Event(
                type=EventType.JOB_CREATED,
                job_id=job.id,
                data=build_transition_payload(
                    created_transition,
                    name=job.name,
                    type="inference",
                    mode="optimize",
                    **self._event_extra_fields(job),
                ),
            )
        )

        await self._launch_process(job.id, cmd, on_log=on_log)
        return job.id

    async def launch_benchmark(
        self,
        *,
        model: str,
        output_dir: str = "results/inference_benchmarks",
        prompts: Optional[str] = None,
        num_prompts: int = 10,
        max_tokens: int = 100,
        warmup: int = 3,
        measure_memory: bool = False,
        on_log: Optional[Callable[[str], None]] = None,
        source_ui_page: str = "/inference",
        origin_job_id: Optional[str] = None,
        relaunch: bool = False,
        resume_strategy: Optional[str] = None,
    ) -> str:
        model, output_dir, num_prompts, max_tokens, warmup = self._validate_benchmark_payload(
            model=model,
            output_dir=output_dir,
            num_prompts=num_prompts,
            max_tokens=max_tokens,
            warmup=warmup,
        )

        output_path = Path(output_dir) / Path(model).name
        output_path.mkdir(parents=True, exist_ok=True)
        cmd = self.build_benchmark_command(
            model=model,
            prompts=prompts,
            num_prompts=num_prompts,
            max_tokens=max_tokens,
            warmup=warmup,
            measure_memory=measure_memory,
        )

        job = self.state.create_job(
            job_type="inference",
            name=f"Inference Benchmark: {Path(model).name}",
            output_dir=output_path,
        )
        job.total_steps = num_prompts

        launch_args = {
            "mode": "benchmark",
            "model": model,
            "output_dir": str(output_path),
            "prompts": prompts,
            "num_prompts": num_prompts,
            "max_tokens": max_tokens,
            "warmup": warmup,
            "measure_memory": measure_memory,
        }
        launch_context_file = None
        try:
            launch_context_file = persist_launch_context(
                output_dir=output_path,
                job_type="inference",
                service="inference",
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
            print(f"[InferenceService] Failed to persist launch context: {e}")

        lifecycle_metadata = self._build_lifecycle_metadata(
            origin_job_id=origin_job_id,
            relaunch=relaunch,
            launch_context_file=launch_context_file,
            resume_strategy=resume_strategy,
        )
        job.launch_context_file = launch_context_file
        job.launch_args = launch_args
        job.lifecycle_metadata = lifecycle_metadata

        created_transition = {
            "from_status": None,
            "to_status": "pending",
            "applied": True,
            "source": "inference_service.launch_benchmark",
            "reason": "job_created",
            "timestamp": datetime.now().isoformat(),
            "metadata": self._merge_transition_metadata(
                job,
                {"job_type": "inference", "mode": "benchmark"},
            ),
        }
        get_event_bus().emit_sync(
            Event(
                type=EventType.JOB_CREATED,
                job_id=job.id,
                data=build_transition_payload(
                    created_transition,
                    name=job.name,
                    type="inference",
                    mode="benchmark",
                    **self._event_extra_fields(job),
                ),
            )
        )

        await self._launch_process(job.id, cmd, on_log=on_log)
        return job.id

    async def relaunch_from_context(
        self,
        launch_context_file: str | Path,
        *,
        origin_job_id: Optional[str] = None,
        source_ui_page: str = "/monitor",
        on_log: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Relaunch inference command from persisted launch context."""
        context = read_launch_context(launch_context_file)
        if context.service != "inference":
            raise ValueError("launch context does not belong to inference service")
        if context.job_type != "inference":
            raise ValueError(f"Unsupported inference relaunch job_type: {context.job_type}")

        args = dict(context.args)
        mode = str(args.pop("mode", "")).strip().lower()
        if mode == "optimize":
            return await self.launch_optimize(
                on_log=on_log,
                source_ui_page=source_ui_page,
                origin_job_id=origin_job_id,
                relaunch=True,
                resume_strategy="relaunch",
                **args,
            )
        if mode == "benchmark":
            return await self.launch_benchmark(
                on_log=on_log,
                source_ui_page=source_ui_page,
                origin_job_id=origin_job_id,
                relaunch=True,
                resume_strategy="relaunch",
                **args,
            )
        raise ValueError(f"Unsupported inference launch mode: {mode}")

    async def _launch_process(
        self,
        job_id: str,
        cmd: list[str],
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        job = self.state.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        self._log_buffers[job_id] = deque(maxlen=1000)
        if on_log:
            self._callbacks.setdefault(job_id, []).append(on_log)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self._get_strix_halo_env(),
            cwd=Path.cwd(),
        )

        job.process = process
        job.started_at = datetime.now()
        transitioned = self.state.update_job_status(
            job_id,
            "running",
            source="inference_service._launch_process",
            reason="process_started",
            metadata=self._merge_transition_metadata(job, {"command": cmd}),
        )
        if transitioned:
            transition = self.state.get_last_transition(job_id)
            await get_event_bus().emit(
                Event(
                    type=EventType.JOB_STARTED,
                    job_id=job_id,
                    data=build_transition_payload(
                        transition,
                        name=job.name,
                        type="inference",
                        **self._event_extra_fields(job),
                    ),
                )
            )

        asyncio.create_task(self._stream_logs(job_id))

    async def _stream_logs(self, job_id: str) -> None:
        job = self.state.get_job(job_id)
        if not job or not job.process:
            return

        log_buffer = self._log_buffers.get(job_id, deque(maxlen=1000))
        callbacks = self._callbacks.get(job_id, [])
        event_bus = get_event_bus()

        try:
            async for line_bytes in job.process.stdout:
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                timestamp = datetime.now().isoformat()
                log_buffer.append({"timestamp": timestamp, "line": line})
                await event_bus.emit(
                    Event(
                        type=EventType.LOG_LINE,
                        job_id=job_id,
                        data={"line": line, "timestamp": timestamp},
                    )
                )
                for callback in callbacks:
                    try:
                        callback(line)
                    except Exception:
                        pass
        except Exception as e:
            job.error_message = str(e)

        return_code = await job.process.wait()
        if job.stop_requested:
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="inference_service._stream_logs",
                reason="stop_requested",
                metadata=self._merge_transition_metadata(job, {"return_code": return_code}),
            )
            if transitioned:
                transition = self.state.get_last_transition(job_id)
                await event_bus.emit(
                    Event(
                        type=EventType.JOB_STOPPED,
                        job_id=job_id,
                        data=build_transition_payload(
                            transition,
                            return_code=return_code,
                            **self._event_extra_fields(job),
                        ),
                    )
                )
            return

        if return_code == 0:
            transitioned = self.state.update_job_status(
                job_id,
                "completed",
                source="inference_service._stream_logs",
                reason="process_exit_ok",
                metadata=self._merge_transition_metadata(job, {"return_code": return_code}),
            )
            if transitioned:
                transition = self.state.get_last_transition(job_id)
                await event_bus.emit(
                    Event(
                        type=EventType.JOB_COMPLETED,
                        job_id=job_id,
                        data=build_transition_payload(
                            transition,
                            return_code=return_code,
                            **self._event_extra_fields(job),
                        ),
                    )
                )
            return

        if return_code in (-signal.SIGTERM, -signal.SIGKILL):
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="inference_service._stream_logs",
                reason="terminated_signal",
                metadata=self._merge_transition_metadata(job, {"return_code": return_code}),
            )
            if transitioned:
                transition = self.state.get_last_transition(job_id)
                await event_bus.emit(
                    Event(
                        type=EventType.JOB_STOPPED,
                        job_id=job_id,
                        data=build_transition_payload(
                            transition,
                            return_code=return_code,
                            **self._event_extra_fields(job),
                        ),
                    )
                )
            return

        job.error_message = f"Process exited with code {return_code}"
        transitioned = self.state.update_job_status(
            job_id,
            "failed",
            source="inference_service._stream_logs",
            reason="process_exit_error",
            metadata=self._merge_transition_metadata(
                job,
                {"return_code": return_code, "error": job.error_message},
            ),
        )
        if transitioned:
            transition = self.state.get_last_transition(job_id)
            await event_bus.emit(
                Event(
                    type=EventType.JOB_FAILED,
                    job_id=job_id,
                    data=build_transition_payload(
                        transition,
                        return_code=return_code,
                        error=job.error_message,
                        **self._event_extra_fields(job),
                    ),
                )
            )

    async def stop_job(self, job_id: str, timeout: float = 30.0) -> bool:
        """Stop a running inference job."""
        job = self.state.get_job(job_id)
        if not job or not job.process:
            return False
        if job.status != "running":
            return False

        job.stop_requested = True
        try:
            job.process.terminate()
        except ProcessLookupError:
            self.state.update_job_status(
                job_id,
                "stopped",
                source="inference_service.stop_job",
                reason="process_missing",
                metadata=self._merge_transition_metadata(job),
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
            source="inference_service.stop_job",
            reason="stop_completed",
            metadata=self._merge_transition_metadata(job),
        )
        return True

    def get_logs(self, job_id: str, last_n: Optional[int] = None) -> list[dict]:
        """Fetch buffered logs for an inference job."""
        buffer = self._log_buffers.get(job_id, deque())
        logs = list(buffer)
        if last_n is not None:
            logs = logs[-last_n:]
        return logs


_inference_service: Optional[InferenceService] = None


def get_inference_service(state=None) -> InferenceService:
    """Get or create inference service singleton."""
    global _inference_service
    if _inference_service is None:
        if state is None:
            from ui.state import state as app_state

            state = app_state
        _inference_service = InferenceService(state)
    return _inference_service
