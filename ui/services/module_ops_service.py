"""
Module Ops Service

Tracked async execution for utility CLI modules (config/data/info/plot).
"""

from __future__ import annotations

import asyncio
import json
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
    MODULE_OPS_LAUNCH_CONTRACT,
    UTILITY_EXECUTION_MODES,
    UTILITY_MODULE_TYPES,
    validate_launch_payload,
)

DEFAULT_MODULE_OPS_OUTPUT_ROOT = "results/ops"


class ModuleOpsService:
    """Launch and manage utility module operations as tracked jobs."""

    def __init__(self, state):
        self.state = state
        self._log_buffers: dict[str, deque] = {}
        self._callbacks: dict[str, list[Callable]] = {}
        self._commands: dict[str, list[str]] = {}

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

    def _validate_module_ops_launch(
        self,
        *,
        module: str,
        execution_mode: str,
        output_root: str,
    ) -> tuple[str, str, str]:
        normalized = validate_launch_payload(
            {
                "module": module,
                "execution_mode": execution_mode,
                "output_root": output_root,
            },
            MODULE_OPS_LAUNCH_CONTRACT,
        )

        normalized_module = str(normalized["module"]).strip().lower()
        if normalized_module not in UTILITY_MODULE_TYPES:
            raise ValueError(
                f"module must be one of: {', '.join(UTILITY_MODULE_TYPES)}"
            )

        normalized_mode = str(normalized["execution_mode"]).strip().lower()
        if normalized_mode not in UTILITY_EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of: {', '.join(UTILITY_EXECUTION_MODES)}"
            )

        normalized_output_root = str(normalized["output_root"]).strip()
        return normalized_module, normalized_mode, normalized_output_root

    def _sanitize_launch_args(self, args: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in args.items():
            if value is None:
                continue
            if isinstance(value, Path):
                payload[key] = str(value)
            else:
                payload[key] = value
        return payload

    def _build_command(
        self,
        *,
        module: str,
        execution_mode: str,
        output_dir: Path,
        args: dict[str, Any],
    ) -> list[str]:
        if execution_mode == "contract":
            return self._build_contract_command(module=module, output_dir=output_dir, args=args)
        return self._build_live_command(module=module, output_dir=output_dir, args=args)

    def _build_contract_command(
        self,
        *,
        module: str,
        output_dir: Path,
        args: dict[str, Any],
    ) -> list[str]:
        if module == "config":
            config_path = Path(str(args.get("config_path") or "configs/sft_example.yaml"))
            config_type = str(args.get("config_type") or "auto")
            if config_path.exists():
                return [
                    sys.executable,
                    "-m",
                    "halo_forge.cli",
                    "config",
                    "validate",
                    str(config_path),
                    "--type",
                    config_type,
                ]
            return [
                sys.executable,
                "-m",
                "halo_forge.cli",
                "config",
                "validate",
                "--help",
            ]

        if module == "data":
            data_file = Path(str(args.get("data_file") or "data/rlvr/humaneval_prompts.jsonl"))
            if data_file.exists():
                return [
                    sys.executable,
                    "-m",
                    "halo_forge.cli",
                    "data",
                    "validate",
                    str(data_file),
                ]
            return [
                sys.executable,
                "-m",
                "halo_forge.cli",
                "data",
                "prepare",
                "--list",
            ]

        if module == "info":
            return [
                sys.executable,
                "-m",
                "halo_forge.cli",
                "info",
            ]

        if module == "plot":
            return [
                sys.executable,
                "-m",
                "halo_forge.cli",
                "plot",
                "benchmarks",
                "--help",
            ]

        raise ValueError(f"Unsupported module: {module}")

    def _build_live_command(
        self,
        *,
        module: str,
        output_dir: Path,
        args: dict[str, Any],
    ) -> list[str]:
        if module == "config":
            config_path = str(args.get("config_path") or "").strip()
            if not config_path:
                raise ValueError("config_path is required for live config runs")
            config_type = str(args.get("config_type") or "auto")
            cmd = [
                sys.executable,
                "-m",
                "halo_forge.cli",
                "config",
                "validate",
                config_path,
                "--type",
                config_type,
            ]
            if bool(args.get("verbose")):
                cmd.append("--verbose")
            return cmd

        if module == "data":
            action = str(args.get("data_action") or "validate").strip().lower()
            if action == "validate":
                data_file = str(args.get("data_file") or "").strip()
                if not data_file:
                    raise ValueError("data_file is required for data validate")
                return [
                    sys.executable,
                    "-m",
                    "halo_forge.cli",
                    "data",
                    "validate",
                    data_file,
                ]

            if action == "prepare":
                dataset = str(args.get("dataset") or "").strip()
                if not dataset:
                    raise ValueError("dataset is required for data prepare")
                output_file = str(args.get("data_output") or (output_dir / "prepared.jsonl"))
                cmd = [
                    sys.executable,
                    "-m",
                    "halo_forge.cli",
                    "data",
                    "prepare",
                    "--dataset",
                    dataset,
                    "--output",
                    output_file,
                ]
                template = str(args.get("template") or "").strip()
                if template:
                    cmd.extend(["--template", template])
                system_prompt = str(args.get("system_prompt") or "").strip()
                if system_prompt:
                    cmd.extend(["--system-prompt", system_prompt])
                return cmd

            if action == "generate":
                topic = str(args.get("topic") or "").strip()
                if not topic:
                    raise ValueError("topic is required for data generate")
                output_file = str(args.get("data_output") or (output_dir / "generated.jsonl"))
                backend = str(args.get("backend") or "deepseek").strip()
                cmd = [
                    sys.executable,
                    "-m",
                    "halo_forge.cli",
                    "data",
                    "generate",
                    "--topic",
                    topic,
                    "--backend",
                    backend,
                    "--output",
                    output_file,
                ]
                model = str(args.get("backend_model") or "").strip()
                if model:
                    cmd.extend(["--model", model])
                template = str(args.get("template") or "").strip()
                if template:
                    cmd.extend(["--template", template])
                return cmd

            if action == "list":
                return [
                    sys.executable,
                    "-m",
                    "halo_forge.cli",
                    "data",
                    "prepare",
                    "--list",
                ]

            raise ValueError("data_action must be one of: validate, prepare, generate, list")

        if module == "info":
            return [
                sys.executable,
                "-m",
                "halo_forge.cli",
                "info",
            ]

        if module == "plot":
            action = str(args.get("plot_action") or "benchmarks").strip().lower()
            if action == "training":
                log_dir = str(args.get("plot_input") or "").strip() or "logs"
                cmd = [
                    sys.executable,
                    "-m",
                    "halo_forge.cli",
                    "plot",
                    "training",
                    log_dir,
                ]
                output_path = str(args.get("plot_output") or "").strip()
                if output_path:
                    cmd.extend(["--output", output_path])
                if bool(args.get("plot_compare")):
                    cmd.append("--compare")
                return cmd

            if action == "benchmarks":
                results_dir = str(args.get("plot_input") or "").strip() or "results/benchmarks"
                cmd = [
                    sys.executable,
                    "-m",
                    "halo_forge.cli",
                    "plot",
                    "benchmarks",
                    results_dir,
                ]
                output_path = str(args.get("plot_output") or "").strip()
                if output_path:
                    cmd.extend(["--output", output_path])
                return cmd

            raise ValueError("plot_action must be one of: training, benchmarks")

        raise ValueError(f"Unsupported module: {module}")

    async def launch_module_op(
        self,
        *,
        module: str,
        execution_mode: str = "contract",
        output_root: str = DEFAULT_MODULE_OPS_OUTPUT_ROOT,
        on_log: Optional[Callable[[str], None]] = None,
        source_ui_page: str = "/ops-console",
        origin_job_id: Optional[str] = None,
        relaunch: bool = False,
        resume_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        module, execution_mode, output_root = self._validate_module_ops_launch(
            module=module,
            execution_mode=execution_mode,
            output_root=output_root,
        )

        output_root_path = Path(output_root)
        output_root_path.mkdir(parents=True, exist_ok=True)

        launch_args = self._sanitize_launch_args(
            {
                "module": module,
                "execution_mode": execution_mode,
                "output_root": str(output_root_path),
                "source_ui_page": source_ui_page,
                **kwargs,
            }
        )

        job = self.state.create_job(
            job_type=module,
            name=f"{module.upper()} Ops ({execution_mode})",
            output_dir=output_root_path / module,
        )

        job_output_dir = output_root_path / module / job.id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir = job_output_dir

        cmd = self._build_command(
            module=module,
            execution_mode=execution_mode,
            output_dir=job_output_dir,
            args=dict(launch_args),
        )

        launch_context_file = None
        try:
            launch_context_file = persist_launch_context(
                output_dir=job_output_dir,
                job_type=module,
                service="module_ops",
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
            print(f"[ModuleOpsService] Failed to persist launch context: {e}")

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
            "source": "module_ops_service.launch_module_op",
            "reason": "job_created",
            "timestamp": datetime.now().isoformat(),
            "metadata": self._merge_transition_metadata(
                job,
                {
                    "module": module,
                    "execution_mode": execution_mode,
                },
            ),
        }
        get_event_bus().emit_sync(
            Event(
                type=EventType.JOB_CREATED,
                job_id=job.id,
                data=build_transition_payload(
                    created_transition,
                    name=job.name,
                    type=module,
                    module=module,
                    execution_mode=execution_mode,
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
        """Relaunch utility command from persisted launch context."""
        context = read_launch_context(launch_context_file)
        if context.service != "module_ops":
            raise ValueError("launch context does not belong to module ops service")
        if context.job_type not in UTILITY_MODULE_TYPES:
            raise ValueError(f"Unsupported module relaunch job_type: {context.job_type}")

        args = dict(context.args)
        module = str(args.pop("module", context.job_type)).strip().lower() or context.job_type
        execution_mode = str(args.pop("execution_mode", "contract")).strip().lower()
        output_root = str(args.pop("output_root", DEFAULT_MODULE_OPS_OUTPUT_ROOT))

        return await self.launch_module_op(
            module=module,
            execution_mode=execution_mode,
            output_root=output_root,
            on_log=on_log,
            source_ui_page=source_ui_page,
            origin_job_id=origin_job_id,
            relaunch=True,
            resume_strategy="relaunch",
            **args,
        )

    def _get_env(self) -> dict[str, str]:
        return os.environ.copy()

    async def _launch_process(
        self,
        job_id: str,
        cmd: list[str],
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        job = self.state.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        self._log_buffers[job_id] = deque(maxlen=5000)
        self._commands[job_id] = list(cmd)
        if on_log:
            self._callbacks.setdefault(job_id, []).append(on_log)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self._get_env(),
            cwd=Path.cwd(),
        )

        job.process = process
        job.started_at = datetime.now()
        transitioned = self.state.update_job_status(
            job_id,
            "running",
            source="module_ops_service._launch_process",
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
                        type=job.type,
                        module=job.type,
                        **self._event_extra_fields(job),
                    ),
                )
            )

        asyncio.create_task(self._stream_logs(job_id))

    def _write_run_summary(self, job_id: str, return_code: int) -> None:
        job = self.state.get_job(job_id)
        if not job or not job.output_dir:
            return

        output_dir = Path(job.output_dir)
        summary_path = output_dir / "run_summary.json"
        stdout_log = output_dir / "stdout.log"
        launch_context = output_dir / "launch_context.json"

        payload = {
            "contract_version": 1,
            "job_id": job.id,
            "module": job.type,
            "name": job.name,
            "status": job.status,
            "return_code": int(return_code),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "duration_seconds": job.duration,
            "execution_mode": str(job.launch_args.get("execution_mode") or "contract"),
            "source_ui_page": str(job.launch_args.get("source_ui_page") or ""),
            "command": self._commands.get(job_id, []),
            "output_dir": str(output_dir),
            "stdout_log": str(stdout_log),
            "stderr_log": None,
            "launch_context": str(launch_context),
            "artifact_pointers": {
                "run_summary": str(summary_path),
                "stdout_log": str(stdout_log),
                "launch_context": str(launch_context),
            },
            "error_message": job.error_message,
            "metadata": dict(job.lifecycle_metadata or {}),
        }

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = summary_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temp_path.replace(summary_path)

    async def _stream_logs(self, job_id: str) -> None:
        job = self.state.get_job(job_id)
        if not job or not job.process:
            return

        log_buffer = self._log_buffers.get(job_id, deque(maxlen=5000))
        callbacks = self._callbacks.get(job_id, [])
        event_bus = get_event_bus()
        output_dir = Path(job.output_dir) if job.output_dir else None
        stdout_log_path = output_dir / "stdout.log" if output_dir else None
        stdout_file = None
        if stdout_log_path:
            stdout_log_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_file = stdout_log_path.open("a", encoding="utf-8")

        try:
            async for line_bytes in job.process.stdout:
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                if not line:
                    continue

                timestamp = datetime.now().isoformat()
                log_buffer.append({"timestamp": timestamp, "line": line})
                if stdout_file:
                    stdout_file.write(f"[{timestamp}] {line}\n")
                    stdout_file.flush()

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
        finally:
            if stdout_file:
                stdout_file.close()

        return_code = await job.process.wait()

        if job.stop_requested:
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="module_ops_service._stream_logs",
                reason="stop_requested",
                metadata=self._merge_transition_metadata(job, {"return_code": return_code}),
            )
            event_type = EventType.JOB_STOPPED
            transition_reason = "stopped"
        elif return_code == 0:
            transitioned = self.state.update_job_status(
                job_id,
                "completed",
                source="module_ops_service._stream_logs",
                reason="process_exit_ok",
                metadata=self._merge_transition_metadata(job, {"return_code": return_code}),
            )
            event_type = EventType.JOB_COMPLETED
            transition_reason = "completed"
        elif return_code in (-signal.SIGTERM, -signal.SIGKILL):
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="module_ops_service._stream_logs",
                reason="terminated_signal",
                metadata=self._merge_transition_metadata(job, {"return_code": return_code}),
            )
            event_type = EventType.JOB_STOPPED
            transition_reason = "stopped"
        else:
            job.error_message = f"Process exited with code {return_code}"
            transitioned = self.state.update_job_status(
                job_id,
                "failed",
                source="module_ops_service._stream_logs",
                reason="process_exit_error",
                metadata=self._merge_transition_metadata(
                    job,
                    {"return_code": return_code, "error": job.error_message},
                ),
            )
            event_type = EventType.JOB_FAILED
            transition_reason = "failed"

        self._write_run_summary(job_id, return_code)

        if transitioned:
            transition = self.state.get_last_transition(job_id)
            await event_bus.emit(
                Event(
                    type=event_type,
                    job_id=job_id,
                    data=build_transition_payload(
                        transition,
                        return_code=return_code,
                        status=transition_reason,
                        error=job.error_message,
                        module=job.type,
                        **self._event_extra_fields(job),
                    ),
                )
            )

    async def stop_job(self, job_id: str, timeout: float = 30.0) -> bool:
        """Stop a running module ops job."""
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
                source="module_ops_service.stop_job",
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
            source="module_ops_service.stop_job",
            reason="stop_completed",
            metadata=self._merge_transition_metadata(job),
        )
        return True

    def get_logs(self, job_id: str, last_n: Optional[int] = None) -> list[dict]:
        """Fetch buffered logs for a module ops job."""
        buffer = self._log_buffers.get(job_id, deque())
        logs = list(buffer)
        if last_n is not None:
            logs = logs[-last_n:]
        return logs


_module_ops_service: Optional[ModuleOpsService] = None


def get_module_ops_service(state=None) -> ModuleOpsService:
    """Get or create module-ops service singleton."""
    global _module_ops_service
    if _module_ops_service is None:
        if state is None:
            from ui.state import state as app_state

            state = app_state
        _module_ops_service = ModuleOpsService(state)
    return _module_ops_service
