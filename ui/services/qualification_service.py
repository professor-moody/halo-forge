"""Qualification Service

Tracked async execution for all-module qualification orchestration runs.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .event_bus import Event, EventType, build_transition_payload, get_event_bus
from .launch_context import persist_launch_context, read_launch_context

DEFAULT_QUALIFICATION_OUTPUT_ROOT = "results/readiness/qualification"
DEFAULT_CANONICAL_REPORT_PATH = Path("results/readiness/all_module_qualification.v1.json")
DEFAULT_BASELINE_PATH = Path("tests/baselines/all_module_qualification_baseline.v1.json")


class QualificationService:
    """Launch and manage all-module qualification jobs as tracked runs."""

    def __init__(self, state):
        self.state = state
        self._log_buffers: dict[str, deque] = {}
        self._callbacks: dict[str, list[Callable[[str], None]]] = {}
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

    def _sanitize_launch_args(self, args: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in args.items():
            if value is None:
                continue
            if isinstance(value, Path):
                payload[key] = str(value)
            elif isinstance(value, list):
                payload[key] = [str(item) for item in value]
            else:
                payload[key] = value
        return payload

    def _build_command(
        self,
        *,
        qualification_profile: str,
        report_file: Path,
        baseline_file: Optional[Path],
        compare_baseline: bool,
        write_baseline: bool,
        strict: bool,
        module_filters: list[str],
        fixture_pack: str,
    ) -> list[str]:
        script_path = Path.cwd() / "scripts" / "run_all_module_qualification.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--qualification-profile",
            str(qualification_profile),
            "--write-report",
            "--report-file",
            str(report_file),
        ]

        if baseline_file:
            cmd.extend(["--baseline-file", str(baseline_file)])
        if compare_baseline:
            cmd.append("--compare-baseline")
        if write_baseline:
            cmd.append("--write-baseline")
        if strict:
            cmd.append("--strict")
        if fixture_pack:
            cmd.extend(["--fixture-pack", str(fixture_pack)])
        for module in module_filters:
            cmd.extend(["--module", str(module)])
        return cmd

    async def launch_qualification(
        self,
        *,
        qualification_profile: str = "contract-v1",
        output_root: str = DEFAULT_QUALIFICATION_OUTPUT_ROOT,
        baseline_file: Optional[str] = None,
        compare_baseline: bool = False,
        write_baseline: bool = False,
        strict: bool = False,
        module_filters: Optional[list[str]] = None,
        fixture_pack: str = "",
        source_ui_page: str = "/research-hub",
        origin_job_id: Optional[str] = None,
        relaunch: bool = False,
        resume_strategy: Optional[str] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> str:
        profile = str(qualification_profile or "contract-v1").strip()
        if profile not in {"contract-v1", "fixture-v1", "live-local"}:
            raise ValueError("qualification_profile must be one of: contract-v1, fixture-v1, live-local")

        selected_modules = []
        for module in module_filters or []:
            key = str(module or "").strip().lower()
            if not key:
                continue
            if key not in {
                "config",
                "data",
                "info",
                "plot",
                "sft",
                "raft",
                "benchmark_code",
                "benchmark_non_code",
                "inference",
                "vlm",
                "audio",
                "reasoning",
                "agentic",
                "ui_ops",
            }:
                raise ValueError(f"Unsupported module filter: {key}")
            if key not in selected_modules:
                selected_modules.append(key)

        output_root_path = Path(output_root)
        output_root_path.mkdir(parents=True, exist_ok=True)

        launch_args = self._sanitize_launch_args(
            {
                "qualification_profile": profile,
                "output_root": str(output_root_path),
                "baseline_file": baseline_file or str(DEFAULT_BASELINE_PATH),
                "compare_baseline": bool(compare_baseline),
                "write_baseline": bool(write_baseline),
                "strict": bool(strict),
                "module_filters": list(selected_modules),
                "fixture_pack": str(fixture_pack or ""),
                "source_ui_page": source_ui_page,
            }
        )

        job = self.state.create_job(
            job_type="qualification",
            name=f"All-Module Qualification ({profile})",
            output_dir=output_root_path,
        )

        job_output_dir = output_root_path / job.id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        job.output_dir = job_output_dir

        report_file = job_output_dir / "all_module_qualification.v1.json"
        baseline_path = Path(str(launch_args.get("baseline_file") or DEFAULT_BASELINE_PATH))

        cmd = self._build_command(
            qualification_profile=profile,
            report_file=report_file,
            baseline_file=baseline_path,
            compare_baseline=bool(compare_baseline),
            write_baseline=bool(write_baseline),
            strict=bool(strict),
            module_filters=selected_modules,
            fixture_pack=str(fixture_pack or ""),
        )

        launch_context_file = None
        try:
            launch_context_file = persist_launch_context(
                output_dir=job_output_dir,
                job_type="qualification",
                service="qualification",
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
            print(f"[QualificationService] Failed to persist launch context: {e}")

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
            "source": "qualification_service.launch_qualification",
            "reason": "job_created",
            "timestamp": datetime.now().isoformat(),
            "metadata": self._merge_transition_metadata(
                job,
                {
                    "qualification_profile": profile,
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
                    type=job.type,
                    qualification_profile=profile,
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
        """Relaunch qualification command from persisted launch context."""
        context = read_launch_context(launch_context_file)
        if context.service != "qualification":
            raise ValueError("launch context does not belong to qualification service")
        if context.job_type != "qualification":
            raise ValueError("launch context job_type is not qualification")

        args = dict(context.args)
        return await self.launch_qualification(
            qualification_profile=str(args.get("qualification_profile") or "contract-v1"),
            output_root=str(args.get("output_root") or DEFAULT_QUALIFICATION_OUTPUT_ROOT),
            baseline_file=str(args.get("baseline_file") or DEFAULT_BASELINE_PATH),
            compare_baseline=bool(args.get("compare_baseline", False)),
            write_baseline=bool(args.get("write_baseline", False)),
            strict=bool(args.get("strict", False)),
            module_filters=[str(v) for v in args.get("module_filters", []) if v is not None],
            fixture_pack=str(args.get("fixture_pack") or ""),
            source_ui_page=source_ui_page,
            origin_job_id=origin_job_id,
            relaunch=True,
            resume_strategy="relaunch",
            on_log=on_log,
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
            source="qualification_service._launch_process",
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
        qualification_report = output_dir / "all_module_qualification.v1.json"

        # Keep canonical latest report updated for UI read surfaces.
        if qualification_report.exists():
            try:
                DEFAULT_CANONICAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(qualification_report, DEFAULT_CANONICAL_REPORT_PATH)
            except Exception as e:
                print(f"[QualificationService] Failed to copy canonical report: {e}")

        payload = {
            "contract_version": 1,
            "job_id": job.id,
            "module": "qualification",
            "name": job.name,
            "status": job.status,
            "return_code": int(return_code),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "duration_seconds": job.duration,
            "execution_mode": "contract",
            "source_ui_page": str(job.launch_args.get("source_ui_page") or ""),
            "command": self._commands.get(job_id, []),
            "output_dir": str(output_dir),
            "stdout_log": str(stdout_log),
            "stderr_log": None,
            "launch_context": str(launch_context),
            "qualification_report": str(qualification_report),
            "artifact_pointers": {
                "run_summary": str(summary_path),
                "stdout_log": str(stdout_log),
                "launch_context": str(launch_context),
                "qualification_report": str(qualification_report),
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
            job.log_file_path = stdout_log_path
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
                source="qualification_service._stream_logs",
                reason="stop_requested",
                metadata=self._merge_transition_metadata(job, {"return_code": return_code}),
            )
            event_type = EventType.JOB_STOPPED
            transition_reason = "stopped"
        elif return_code == 0:
            transitioned = self.state.update_job_status(
                job_id,
                "completed",
                source="qualification_service._stream_logs",
                reason="process_exit_ok",
                metadata=self._merge_transition_metadata(job, {"return_code": return_code}),
            )
            event_type = EventType.JOB_COMPLETED
            transition_reason = "completed"
        elif return_code in (-signal.SIGTERM, -signal.SIGKILL):
            transitioned = self.state.update_job_status(
                job_id,
                "stopped",
                source="qualification_service._stream_logs",
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
                source="qualification_service._stream_logs",
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
                        module="qualification",
                        **self._event_extra_fields(job),
                    ),
                )
            )

    async def stop_job(self, job_id: str, timeout: float = 30.0) -> bool:
        """Stop a running qualification job."""
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
        try:
            job.process.terminate()
        except ProcessLookupError:
            self.state.update_job_status(
                job_id,
                "stopped",
                source="qualification_service.stop_job",
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
            source="qualification_service.stop_job",
            reason="stop_completed",
            metadata=self._merge_transition_metadata(job),
        )
        return True

    def get_logs(self, job_id: str, last_n: Optional[int] = None) -> list[dict]:
        """Fetch buffered logs for a qualification job."""
        buffer = self._log_buffers.get(job_id, deque())
        logs = list(buffer)
        if last_n is not None:
            logs = logs[-last_n:]
        return logs


_qualification_service: Optional[QualificationService] = None


def get_qualification_service(state=None) -> QualificationService:
    """Get or create qualification service singleton."""
    global _qualification_service
    if _qualification_service is None:
        if state is None:
            from ui.state import state as app_state

            state = app_state
        _qualification_service = QualificationService(state)
    return _qualification_service
