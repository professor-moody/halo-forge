"""Executable worker for the durable single-workstation queue.

Most work items provide a concrete argv list. Checkpoint-gated training and
evaluation items instead carry a deliberately incomplete command template; the
worker resolves it only after the dependency has published a verified model
artifact. This prevents a resumed segment from silently restarting at zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, BinaryIO, Callable, Mapping, Optional, Protocol, Sequence

from halo_forge.run_db import LabV4Catalog, WorkItemRecord, get_database
from halo_forge.workstation_jobs.resources import (
    TELEMETRY_SAMPLE_INTERVAL_SECONDS,
    WorkstationCapacity,
    WorkstationTelemetrySample,
    sample_workstation_capacity,
)
from halo_forge.workstation_jobs.scheduler import WorkstationScheduler, process_start_time


class ProcessHandle(Protocol):
    """The process operations needed by :class:`WorkstationWorker`."""

    pid: int

    def poll(self) -> Optional[int]: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class ProcessRunner(Protocol):
    """Injectable subprocess factory used by the worker tests and runtime."""

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Optional[str],
        env: Mapping[str, str],
        log_file: BinaryIO,
    ) -> ProcessHandle: ...


class _SubprocessHandle:
    """Popen wrapper that terminates the complete process group on POSIX."""

    def __init__(self, process: subprocess.Popen[bytes]):
        self._process = process
        self.pid = int(process.pid)

    def poll(self) -> Optional[int]:
        return self._process.poll()

    def terminate(self) -> None:
        if os.name == "posix":
            try:
                os.killpg(self.pid, signal.SIGTERM)
                return
            except ProcessLookupError:
                return
        self._process.terminate()

    def kill(self) -> None:
        if os.name == "posix":
            try:
                os.killpg(self.pid, signal.SIGKILL)
                return
            except ProcessLookupError:
                return
        self._process.kill()


class SubprocessRunner:
    """Start argv directly, never through a shell."""

    def start(
        self,
        command: Sequence[str],
        *,
        cwd: Optional[str],
        env: Mapping[str, str],
        log_file: BinaryIO,
    ) -> ProcessHandle:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            shell=False,
            start_new_session=os.name == "posix",
        )
        return _SubprocessHandle(process)


class LostClaimError(RuntimeError):
    """Raised when another owner has recovered or interrupted this claim."""


class WorkstationWorker:
    """Claim and execute durable work items one at a time."""

    def __init__(
        self,
        scheduler: WorkstationScheduler,
        *,
        runner: Optional[ProcessRunner] = None,
        poll_interval: float = 0.25,
        heartbeat_interval: Optional[float] = None,
        terminate_timeout: float = 10.0,
        process_identity: Callable[[int], Optional[float]] = process_start_time,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        telemetry_sampler: Optional[Callable[..., WorkstationCapacity]] = None,
    ):
        self.scheduler = scheduler
        self.runner = runner or SubprocessRunner()
        self.poll_interval = max(0.01, float(poll_interval))
        default_heartbeat = max(0.25, scheduler.lease_ttl_seconds / 3)
        self.heartbeat_interval = max(
            0.05,
            float(default_heartbeat if heartbeat_interval is None else heartbeat_interval),
        )
        # The heartbeat must happen before the scheduler lease can expire.
        self.heartbeat_interval = min(
            self.heartbeat_interval,
            max(0.05, scheduler.lease_ttl_seconds * 0.8),
        )
        self.terminate_timeout = max(0.0, float(terminate_timeout))
        self.process_identity = process_identity
        self.sleep = sleep
        self.monotonic = monotonic
        self.telemetry_sampler = telemetry_sampler or sample_workstation_capacity
        self.stop_event = threading.Event()
        self._orchestration_recovery_complete = False

    def stop(self) -> None:
        """Ask a watching worker to stop after safely ending current work."""

        self.stop_event.set()

    def run_once(self, *, work_item_id: Optional[str] = None) -> Optional[WorkItemRecord]:
        """Claim at most one ready item and return its terminal record.

        ``None`` means the queue has no dependency- and resource-ready item.
        This is also how retained serving leases and unfinished dependencies
        block execution: the scheduler simply yields no claim.  ``work_item_id``
        lets synchronous transports such as the CLI execute the exact durable
        item they just created without jumping ahead of unrelated queued work.
        """

        if not self._orchestration_recovery_complete:
            self._orchestration_recovery_complete = self._recover_checkpoint_orchestration()
        item = self.scheduler.claim(work_item_id=work_item_id)
        if item is None:
            return None
        allowed, _availability = self.scheduler.confirm_pre_spawn(item)
        if not allowed:
            return self.scheduler.database.get_work_item(item.id)
        if item.launch_spec.get("handler") == "adaptive_lab.execute_work_item":
            terminal = self._execute_adaptive_lab(item)
        elif item.launch_spec.get("handler") == "future_lab.execute_work_item":
            terminal = self._execute_future_lab(item)
        elif item.launch_spec.get("handler") == "product_v17.execute_work_item":
            terminal = self._execute_product_v17(item)
        elif item.launch_spec.get("handler") == "training_plan.execute_work_item":
            terminal = self._execute_training_plan(item)
        elif item.launch_spec.get("handler") == "managed_runtime.execute_work_item":
            terminal = self._execute_managed_runtime(item)
        elif item.launch_spec.get("handler") == "training_path_certification.execute_work_item":
            terminal = self._execute_training_path_certification(item)
        elif item.launch_spec.get("handler") == "review_lab.generate_suggestion":
            terminal = self._execute_review_suggestion(item)
        elif item.launch_spec.get("handler") == "review_lab.resolve_acquisition":
            terminal = self._execute_review_acquisition(item)
        elif item.launch_spec.get("handler") == "review_lab.execute_work_item":
            terminal = self._execute_review_lab(item)
        elif item.launch_spec.get("handler") == "artifact_studio.execute_work_item":
            terminal = self._execute_artifact_studio(item)
        elif item.launch_spec.get("handler") == "dataset_lab.run_queued":
            terminal = self._execute_dataset_lab(item)
        elif item.launch_spec.get("handler") == "own_data.inspect":
            terminal = self._execute_own_data_inspection(item)
        elif item.launch_spec.get("handler") == "corpus_lab.extract_source":
            terminal = self._execute_corpus_extraction(item)
        elif item.launch_spec.get("handler") == "own_data.register":
            terminal = self._execute_own_data_registration(item)
        elif item.launch_spec.get("handler") == "own_data.refresh_source":
            terminal = self._execute_own_data_source_refresh(item)
        elif item.launch_spec.get("handler") == "verifier_lab.run_calibration":
            terminal = self._execute_verifier_lab(item)
        elif item.launch_spec.get("handler") == "reward_integrity.execute_audit":
            terminal = self._execute_reward_integrity(item)
        elif item.launch_spec.get("handler") == "evaluation_lab.run_queued":
            terminal = self._execute_evaluation_lab(item)
        else:
            terminal = self._execute(item)
        self._after_terminal_event(item, terminal)
        return terminal

    def _execute_managed_runtime(self, item: WorkItemRecord) -> WorkItemRecord:
        """Prepare or qualify a pinned accelerator runtime durably."""

        from halo_forge.managed_runtime import ManagedRuntimeService

        root = Path(str(item.launch_spec.get("runtime_root") or Path.home() / ".halo-forge" / "runtimes")).expanduser()
        service = ManagedRuntimeService(
            self.scheduler.database,
            root=root,
            source_root=item.launch_spec.get("source_root"),
            scheduler=self.scheduler,
        )
        table = "runtime_preparations" if item.domain_kind == "runtime_preparation" else "runtime_qualifications"

        def snapshot() -> tuple[str, Mapping[str, Any]]:
            row = self.scheduler.database._conn.execute(
                f"SELECT stage,progress_json FROM {table} WHERE id=?", (item.domain_id,)
            ).fetchone()
            return (
                (str(row["stage"]), json.loads(row["progress_json"] or "{}"))
                if row
                else ("running", {})
            )

        def cancel_domain() -> None:
            self.scheduler.database._conn.execute(
                f"UPDATE {table} SET cancel_requested=1 WHERE id=?", (item.domain_id,)
            )
            self.scheduler.database._conn.commit()

        stop = None
        thread = None
        heartbeat_errors: list[str] = []
        try:
            stop, thread, heartbeat_errors = self._start_domain_heartbeat(
                item, snapshot=snapshot, cancel_domain=cancel_domain
            )
            result = service.execute_work_item(item)
            if (
                item.domain_kind == "runtime_preparation"
                and str(result.get("status") or "") == "completed"
            ):
                qualification = service.qualify(
                    str(result["runtime_revision_id"]), enqueue=True
                )
                result = {**dict(result), "qualification": qualification.to_dict()}
            elif (
                item.domain_kind == "runtime_qualification"
                and str(result.get("status") or "") in {"vendor_supported", "local_verified"}
            ):
                # Progressive V21 setup: once core operations pass, enqueue the
                # recommended real SFT path. This does not mark any trainer
                # available; only the certification's published evidence can.
                from halo_forge.training_path_certification import (
                    TrainingPathCertificationService,
                )

                path_service = TrainingPathCertificationService(
                    self.scheduler.database,
                    runtime_service=service,
                    scheduler=self.scheduler,
                    source_root=item.launch_spec.get("source_root"),
                )
                family = service.get_profile(
                    service.get_revision(str(result["runtime_revision_id"])).profile_id
                ).accelerator_family
                matrix = path_service.capabilities(family)
                path = matrix.recommended_path_revision_id
                if path:
                    certification = path_service.certify(
                        path,
                        str(result["runtime_revision_id"]),
                        enqueue=True,
                    )
                    result = {
                        **dict(result),
                        "recommended_path_certification": certification.to_dict(),
                    }
            stop.set()
            thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            stop = None
            thread = None
            if heartbeat_errors:
                raise LostClaimError(heartbeat_errors[-1])
            if str(result.get("status") or "") == "blocked" and str(result.get("stage") or "") == "waiting_for_accelerator":
                if not item.claim_token:
                    raise LostClaimError("runtime qualification lost its accelerator claim")
                queued = self.scheduler.database.defer_claimed_work_item_for_accelerator(
                    item.id,
                    claim_token=item.claim_token,
                    reason=str(result.get("error") or "waiting for accelerator"),
                    details={"qualification": result},
                    not_before=(datetime.now(timezone.utc) + timedelta(seconds=15)).isoformat(),
                )
                if queued is None:
                    raise LostClaimError("could not return runtime qualification to the queue")
                return queued
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                raise LostClaimError(f"could not complete managed runtime work {item.id}")
            return finished
        except Exception as exc:
            current = self.scheduler.database.get_work_item(item.id)
            # Waiting is a non-failure state and has already released its claim.
            if current is not None and current.status == "queued" and current.stage == "waiting_for_accelerator":
                return current
            def prepare_retry() -> None:
                if item.domain_kind == "runtime_preparation":
                    self.scheduler.database._conn.execute(
                        "UPDATE runtime_preparations SET status='queued',stage='queued',error=NULL,cancel_requested=0 WHERE id=?",
                        (item.domain_id,),
                    )
                else:
                    self.scheduler.database._conn.execute(
                        "UPDATE runtime_qualifications SET status='queued',stage='queued',error=NULL,cancel_requested=0 WHERE id=?",
                        (item.domain_id,),
                    )
                self.scheduler.database._conn.commit()

            return self._finish_domain_failure(
                item,
                result={"domain_kind": item.domain_kind, "domain_id": item.domain_id},
                error=f"managed runtime work failed: {exc}",
                prepare_retry=prepare_retry,
            )
        finally:
            if stop is not None:
                stop.set()
            if thread is not None:
                thread.join(timeout=max(1.0, self.heartbeat_interval * 2))

    def _execute_training_path_certification(self, item: WorkItemRecord) -> WorkItemRecord:
        """Run or resume V21 real-trainer certification with safe waiting."""

        from halo_forge.managed_runtime import ManagedRuntimeService
        from halo_forge.training_path_certification import TrainingPathCertificationService

        runtime = ManagedRuntimeService(
            self.scheduler.database,
            root=Path(str(item.launch_spec.get("runtime_root") or Path.home() / ".halo-forge" / "runtimes")),
            source_root=item.launch_spec.get("source_root"),
            scheduler=self.scheduler,
        )
        service = TrainingPathCertificationService(
            self.scheduler.database,
            root=Path(str(item.launch_spec.get("certification_root") or Path.home() / ".halo-forge" / "certifications")),
            runtime_service=runtime,
            scheduler=self.scheduler,
            source_root=item.launch_spec.get("source_root"),
        )

        workstation = item.domain_kind == "workstation_certification"
        table = (
            "workstation_certifications"
            if workstation
            else "training_path_certifications"
        )

        def snapshot() -> tuple[str, Mapping[str, Any]]:
            row = self.scheduler.database._conn.execute(
                f"SELECT stage,progress_json FROM {table} WHERE id=?",
                (item.domain_id,),
            ).fetchone()
            return (
                (str(row["stage"]), json.loads(row["progress_json"] or "{}"))
                if row
                else ("running", {})
            )

        def cancel_domain() -> None:
            self.scheduler.database._conn.execute(
                f"UPDATE {table} SET cancel_requested=1 WHERE id=?",
                (item.domain_id,),
            )
            self.scheduler.database._conn.commit()

        stop = None
        thread = None
        heartbeat_errors: list[str] = []
        try:
            stop, thread, heartbeat_errors = self._start_domain_heartbeat(
                item, snapshot=snapshot, cancel_domain=cancel_domain
            )
            result = service.execute_work_item(item)
            stop.set()
            thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            stop = None
            thread = None
            if heartbeat_errors:
                raise LostClaimError(heartbeat_errors[-1])
            if not workstation and str(result.get("status") or "") == "waiting_for_accelerator":
                if not item.claim_token:
                    raise LostClaimError("training-path certification lost its accelerator claim")
                queued = self.scheduler.database.defer_claimed_work_item_for_accelerator(
                    item.id,
                    claim_token=item.claim_token,
                    reason=str(result.get("error") or "waiting for accelerator"),
                    details={"certification": result},
                    not_before=(datetime.now(timezone.utc) + timedelta(seconds=15)).isoformat(),
                )
                if queued is None:
                    raise LostClaimError("could not return certification to the queue")
                return queued
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                raise LostClaimError(f"could not complete training-path certification {item.id}")
            return finished
        except Exception as exc:
            current = self.scheduler.database.get_work_item(item.id)
            if current is not None and current.status == "queued" and current.stage == "waiting_for_accelerator":
                return current

            def prepare_retry() -> None:
                self.scheduler.database._conn.execute(
                    f"UPDATE {table} SET status='queued',stage='queued',error=NULL,cancel_requested=0 WHERE id=?",
                    (item.domain_id,),
                )
                self.scheduler.database._conn.commit()

            return self._finish_domain_failure(
                item,
                result={"domain_kind": item.domain_kind, "domain_id": item.domain_id},
                error=f"training-path certification failed: {exc}",
                prepare_retry=prepare_retry,
            )
        finally:
            if stop is not None:
                stop.set()
            if thread is not None:
                thread.join(timeout=max(1.0, self.heartbeat_interval * 2))

    def _execute_training_plan(self, item: WorkItemRecord) -> WorkItemRecord:
        """Run V18 model preparation and disposable capacity checks."""

        from halo_forge.run_db import RunDatabase
        from halo_forge.training_plan import TrainingPlanService

        root = Path(
            str(item.launch_spec.get("training_plan_root") or Path.home() / ".halo-forge")
        ).expanduser()
        domain_database = self.scheduler.database
        owns_domain_database = False
        if domain_database.path != ":memory:":
            # Domain work can hold a transaction while an accelerator-backed
            # scratch step runs.  Keep heartbeat writes on the scheduler
            # connection and domain publication on its own SQLite connection
            # so neither thread can roll back the other's transaction.
            domain_database = RunDatabase(domain_database.path)
            owns_domain_database = True
        service = TrainingPlanService(
            domain_database,
            root=root,
            scheduler=self.scheduler,
        )
        domain_kind = str(item.domain_kind or "")
        domain_id = str(item.domain_id or "")

        def snapshot() -> tuple[str, Mapping[str, Any]]:
            table = {
                "model_preparation": "model_preparations",
                "training_capacity_check": "training_capacity_checks",
            }.get(domain_kind)
            if not table:
                return "running", {}
            stage_column = "stage" if domain_kind == "training_capacity_check" else "status"
            row = self.scheduler.database._conn.execute(
                f"SELECT status,{stage_column} AS stage,progress_json FROM {table} WHERE id=?",
                (domain_id,),
            ).fetchone()
            if not row:
                return "running", {}
            return str(row["stage"] or row["status"]), dict(
                json.loads(row["progress_json"] or "{}")
            )

        def cancel_domain() -> None:
            table = {
                "model_preparation": "model_preparations",
                "training_capacity_check": "training_capacity_checks",
            }.get(domain_kind)
            if table:
                self.scheduler.database._conn.execute(
                    f"UPDATE {table} SET cancel_requested=1 WHERE id=?", (domain_id,)
                )
                self.scheduler.database._conn.commit()

        stop = None
        thread = None
        heartbeat_errors: list[str] = []
        try:
            stop, thread, heartbeat_errors = self._start_domain_heartbeat(
                item, snapshot=snapshot, cancel_domain=cancel_domain
            )
            result = service.execute_work_item(item)
            stop.set()
            thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            stop = None
            thread = None
            if heartbeat_errors:
                raise LostClaimError(heartbeat_errors[-1])
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                raise LostClaimError(
                    f"could not complete V18 work item {item.id}: claim was lost"
                )
            return finished
        except Exception as exc:
            current = self.scheduler.database.get_work_item(item.id)
            cancelled = bool(current and current.cancel_requested)
            state = "cancelled" if cancelled else "failed"
            table = {
                "model_preparation": "model_preparations",
                "training_capacity_check": "training_capacity_checks",
            }.get(domain_kind)
            if table:
                stage_sql = ",stage=?" if domain_kind == "training_capacity_check" else ""
                values: tuple[Any, ...] = (
                    (state, state, str(exc), domain_id)
                    if stage_sql
                    else (state, str(exc), domain_id)
                )
                self.scheduler.database._conn.execute(
                    f"UPDATE {table} SET status=?{stage_sql},error=? WHERE id=?", values
                )
                self.scheduler.database._conn.commit()

            def prepare_retry() -> None:
                if table:
                    if domain_kind == "training_capacity_check":
                        self.scheduler.database._conn.execute(
                            f"UPDATE {table} SET status='queued',stage='queued',error=NULL,cancel_requested=0 WHERE id=?",
                            (domain_id,),
                        )
                    else:
                        self.scheduler.database._conn.execute(
                            f"UPDATE {table} SET status='queued',error=NULL,cancel_requested=0 WHERE id=?",
                            (domain_id,),
                        )
                    self.scheduler.database._conn.commit()

            return self._finish_domain_failure(
                item,
                result={"domain_kind": domain_kind, "domain_id": domain_id},
                error=f"V18 worker failed: {exc}",
                prepare_retry=prepare_retry,
            )
        finally:
            if stop is not None:
                stop.set()
            if thread is not None:
                thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            if owns_domain_database:
                domain_database.close()

    def _execute_product_v17(self, item: WorkItemRecord) -> WorkItemRecord:
        """Execute repair scans/previews and support bundles durably."""

        from halo_forge.product_lab import ProductLabService

        root = Path(
            str(item.launch_spec.get("product_root") or Path.home() / ".halo-forge")
        ).expanduser()
        service = ProductLabService(
            self.scheduler.database,
            root=root,
            scheduler=self.scheduler,
        )
        domain_kind = str(item.domain_kind or "")
        domain_id = str(item.domain_id or "")

        def snapshot() -> tuple[str, Mapping[str, Any]]:
            db = self.scheduler.database._conn
            if domain_kind == "dataset_repair_session":
                row = db.execute(
                    "SELECT status,stage,progress_json FROM dataset_repair_sessions WHERE id=?",
                    (domain_id,),
                ).fetchone()
            elif domain_kind == "dataset_repair_preview":
                row = db.execute(
                    "SELECT status,status AS stage,counts_json AS progress_json FROM dataset_repair_previews WHERE id=?",
                    (domain_id,),
                ).fetchone()
            elif domain_kind == "support_bundle":
                row = db.execute(
                    "SELECT status,status AS stage,'{}' AS progress_json FROM support_bundles WHERE id=?",
                    (domain_id,),
                ).fetchone()
            elif domain_kind == "release_qualification":
                row = db.execute(
                    "SELECT status,status AS stage,progress_json FROM release_qualifications WHERE id=?",
                    (domain_id,),
                ).fetchone()
            else:
                row = None
            if row is None:
                return "running", {}
            try:
                progress = json.loads(row["progress_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                progress = {}
            return str(row["stage"] or row["status"] or "running"), progress

        def cancel_domain() -> None:
            if domain_kind == "support_bundle":
                self.scheduler.database._conn.execute(
                    "UPDATE support_bundles SET cancel_requested=1 WHERE id=?",
                    (domain_id,),
                )
                self.scheduler.database._conn.commit()
                return
            if domain_kind == "release_qualification":
                self.scheduler.database._conn.execute(
                    "UPDATE release_qualifications SET cancel_requested=1 WHERE id=?",
                    (domain_id,),
                )
                self.scheduler.database._conn.commit()
                return
            if domain_kind in {"dataset_repair_session", "dataset_repair_preview"}:
                session_id = domain_id
                if domain_kind == "dataset_repair_preview":
                    row = self.scheduler.database._conn.execute(
                        "SELECT session_id FROM dataset_repair_previews WHERE id=?",
                        (domain_id,),
                    ).fetchone()
                    session_id = str(row["session_id"]) if row else ""
                if session_id:
                    self.scheduler.database._conn.execute(
                        "UPDATE dataset_repair_sessions SET cancel_requested=1 WHERE id=?",
                        (session_id,),
                    )
                    self.scheduler.database._conn.commit()

        stop = None
        thread = None
        heartbeat_errors: list[str] = []
        try:
            stop, thread, heartbeat_errors = self._start_domain_heartbeat(
                item,
                snapshot=snapshot,
                cancel_domain=cancel_domain,
            )
            result = service.execute_work_item(item)
            if heartbeat_errors:
                raise LostClaimError(heartbeat_errors[-1])
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                raise LostClaimError(
                    f"could not complete V17 work item {item.id}: claim was lost"
                )
            return finished
        except Exception as exc:
            current = self.scheduler.database.get_work_item(item.id)
            cancelled = bool(current and current.cancel_requested)
            db = self.scheduler.database._conn
            state = "cancelled" if cancelled else "failed"
            message = "Operation cancelled by the operator" if cancelled else str(exc)
            if domain_kind == "dataset_repair_session":
                db.execute(
                    "UPDATE dataset_repair_sessions SET status=?,stage=?,error=? WHERE id=?",
                    (state, state, message, domain_id),
                )
            elif domain_kind == "dataset_repair_preview":
                db.execute(
                    "UPDATE dataset_repair_previews SET status=?,error=? WHERE id=?",
                    (state, message, domain_id),
                )
            elif domain_kind == "support_bundle":
                db.execute(
                    "UPDATE support_bundles SET status=?,error=? WHERE id=?",
                    (state, message, domain_id),
                )
            elif domain_kind == "release_qualification":
                db.execute(
                    "UPDATE release_qualifications SET status=?,error=? WHERE id=?",
                    (state, message, domain_id),
                )
            db.commit()

            def prepare_retry() -> None:
                if domain_kind == "dataset_repair_session":
                    db.execute(
                        "UPDATE dataset_repair_sessions SET status='scanning',stage='scanning',error=NULL,cancel_requested=0 WHERE id=?",
                        (domain_id,),
                    )
                elif domain_kind == "dataset_repair_preview":
                    db.execute(
                        "UPDATE dataset_repair_previews SET status='queued',error=NULL WHERE id=?",
                        (domain_id,),
                    )
                elif domain_kind == "support_bundle":
                    db.execute(
                        "UPDATE support_bundles SET status='queued',error=NULL,cancel_requested=0 WHERE id=?",
                        (domain_id,),
                    )
                elif domain_kind == "release_qualification":
                    db.execute(
                        "UPDATE release_qualifications SET status='queued',error=NULL,cancel_requested=0 WHERE id=?",
                        (domain_id,),
                    )
                db.commit()

            return self._finish_domain_failure(
                item,
                result={"domain_kind": domain_kind, "domain_id": domain_id},
                error=f"V17 worker failed: {exc}",
                prepare_retry=prepare_retry,
            )
        finally:
            if stop is not None:
                stop.set()
            if thread is not None:
                thread.join(timeout=max(1.0, self.heartbeat_interval * 2))

    def _execute_future_lab(self, item: WorkItemRecord) -> WorkItemRecord:
        """Run V16 operational work through one supervised durable handler."""

        from halo_forge.public_api.service import PublicApiService

        root = Path(
            str(item.launch_spec.get("future_lab_root") or Path.home() / ".halo-forge")
        ).expanduser()
        service = PublicApiService(
            database=self.scheduler.database,
            workstation_scheduler=self.scheduler,
            dataset_storage_root=Path(
                str(
                    item.launch_spec.get("dataset_root")
                    or root / "datasets"
                )
            ).expanduser(),
            evaluation_storage_root=Path(
                str(
                    item.launch_spec.get("evaluation_root")
                    or root / "evaluations"
                )
            ).expanduser(),
            future_lab_storage_root=root,
        )

        def snapshot() -> tuple[str, Mapping[str, Any]]:
            domain_kind = str(item.domain_kind or "")
            domain_id = str(item.domain_id or "")
            db = self.scheduler.database._conn
            table = {
                "training_outcome_assessment": "training_outcome_assessments",
                "adaptation_study_analysis": "adaptation_study_analyses",
                "grounded_generation_batch": "grounded_generation_batches",
                "agent_episode": "agent_episodes",
            }.get(domain_kind)
            if table:
                row = db.execute(
                    f"SELECT status,stage,progress_json FROM {table} WHERE id=?",
                    (domain_id,),
                ).fetchone()
                if row is not None:
                    try:
                        progress = json.loads(row["progress_json"] or "{}")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        progress = {}
                    return str(row["stage"] or row["status"] or "running"), progress
            if domain_kind == "adaptation_study_protocol_revision":
                row = db.execute(
                    """SELECT launch_status,launch_progress_json
                       FROM adaptation_study_protocol_revisions WHERE id=?""",
                    (domain_id,),
                ).fetchone()
                if row is not None:
                    try:
                        progress = json.loads(row["launch_progress_json"] or "{}")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        progress = {}
                    return str(row["launch_status"] or "running"), progress
            return "running", {}

        def cancel_domain() -> None:
            domain_kind = str(item.domain_kind or "")
            domain_id = str(item.domain_id or "")
            db = self.scheduler.database._conn
            table = {
                "training_outcome_assessment": "training_outcome_assessments",
                "adaptation_study_analysis": "adaptation_study_analyses",
                "grounded_generation_batch": "grounded_generation_batches",
                "agent_episode": "agent_episodes",
            }.get(domain_kind)
            if table:
                db.execute(
                    f"UPDATE {table} SET cancel_requested=1 WHERE id=?",
                    (domain_id,),
                )
            elif domain_kind == "adaptation_study_protocol_revision":
                db.execute(
                    """UPDATE adaptation_study_protocol_revisions
                       SET launch_status='cancelled',
                           launch_error='Launch cancelled by the operator'
                       WHERE id=?""",
                    (domain_id,),
                )
            db.commit()

        stop = None
        thread = None
        heartbeat_errors: list[str] = []
        try:
            stop, thread, heartbeat_errors = self._start_domain_heartbeat(
                item,
                snapshot=snapshot,
                cancel_domain=cancel_domain,
            )
            result = service.execute_future_lab_work_item(item)
            if heartbeat_errors:
                raise LostClaimError(heartbeat_errors[-1])
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                raise LostClaimError(
                    f"could not complete future lab work item {item.id}: claim was lost"
                )
            return finished
        except Exception as exc:
            domain_kind = str(item.domain_kind or "")
            domain_id = str(item.domain_id or "")
            current_item = self.scheduler.database.get_work_item(item.id)
            cancellation_requested = bool(
                current_item and current_item.cancel_requested
            )

            def prepare_retry() -> None:
                db = self.scheduler.database._conn
                table = {
                    "training_outcome_assessment": "training_outcome_assessments",
                    "adaptation_study_analysis": "adaptation_study_analyses",
                    "grounded_generation_batch": "grounded_generation_batches",
                    "agent_episode": "agent_episodes",
                }.get(domain_kind)
                if table:
                    db.execute(
                        f"""UPDATE {table}
                            SET status='queued',stage='waiting',error=NULL,
                                cancel_requested=0 WHERE id=?""",
                        (domain_id,),
                    )
                elif domain_kind == "adaptation_study_protocol_revision":
                    db.execute(
                        """UPDATE adaptation_study_protocol_revisions
                           SET launch_status='queued',launch_error=NULL WHERE id=?""",
                        (domain_id,),
                    )
                db.commit()

            db = self.scheduler.database._conn
            if domain_kind == "adaptation_study_protocol_revision":
                db.execute(
                    """UPDATE adaptation_study_protocol_revisions
                       SET launch_status=?,launch_error=? WHERE id=?""",
                    (
                        "cancelled" if cancellation_requested else "failed",
                        (
                            "Launch cancelled by the operator"
                            if cancellation_requested
                            else str(exc)
                        ),
                        domain_id,
                    ),
                )
            else:
                table = {
                    "training_outcome_assessment": "training_outcome_assessments",
                    "adaptation_study_analysis": "adaptation_study_analyses",
                    "grounded_generation_batch": "grounded_generation_batches",
                    "agent_episode": "agent_episodes",
                }.get(domain_kind)
                if table:
                    db.execute(
                        f"""UPDATE {table}
                            SET status=?,stage=?,error=? WHERE id=?""",
                        (
                            "cancelled" if cancellation_requested else "failed",
                            "cancelled" if cancellation_requested else "failed",
                            (
                                "Operation cancelled by the operator"
                                if cancellation_requested
                                else str(exc)
                            ),
                            domain_id,
                        ),
                    )
            db.commit()
            return self._finish_domain_failure(
                item,
                result={"domain_kind": domain_kind, "domain_id": domain_id},
                error=f"Future Lab worker failed: {exc}",
                prepare_retry=prepare_retry,
            )
        finally:
            if stop is not None:
                stop.set()
            if thread is not None:
                thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self._finalize_telemetry(item)

    def _recover_checkpoint_orchestration(self) -> bool:
        """Resume gate publication missed between a terminal event and callback."""

        recovered = True
        try:
            from halo_forge.orchestration import ExperimentOrchestrationService

            service = ExperimentOrchestrationService(
                self.scheduler.database,
                scheduler=self.scheduler,
            )
            for group in self.scheduler.database.list_run_groups(limit=10000):
                if group.status in {"completed", "cancelled", "failed"}:
                    continue
                plan = getattr(group, "resolved_checkpoint_plan", {}) or {}
                if not plan:
                    continue
                try:
                    service.advance_ready_checkpoint_policy(group.id)
                    service.advance_ready_successive_halving(group.id)
                except Exception:
                    recovered = False
                    continue
        except Exception:
            # Recovery is best effort and must never prevent unrelated ready
            # work from being claimed. A later terminal event or explicit
            # reconciliation repeats the idempotent gate check.
            return False
        return recovered

    def _after_terminal_event(self, item: WorkItemRecord, terminal: WorkItemRecord) -> None:
        """Apply safe event-driven orchestration after durable completion."""

        if item.launch_spec.get("operation") == "managed_training":
            self._sync_managed_run_record(item, terminal)
        elif item.launch_spec.get("operation") == "managed_training_segment" and terminal.status in {
            "failed",
            "cancelled",
            "interrupted",
            "needs_reconciliation",
        }:
            segment_id = str(item.launch_spec.get("direct_run_segment_id") or "")
            if segment_id:
                self.scheduler.database.update_direct_run_segment(
                    segment_id,
                    status=terminal.status,
                    decision_reason=terminal.error or "managed segment did not complete",
                )
            self._sync_managed_run_record(item, terminal)

        if item.launch_spec.get("operation") != "evaluate_trial_segment":
            return
        if terminal.status not in {"completed", "failed", "cancelled", "interrupted"}:
            return
        run_group_id = str(item.launch_spec.get("run_group_id") or "").strip()
        if not run_group_id:
            return
        catalog = LabV4Catalog(self.scheduler.database)
        try:
            from halo_forge.orchestration import ExperimentOrchestrationService

            service = ExperimentOrchestrationService(
                self.scheduler.database,
                scheduler=self.scheduler,
            )
            adaptive_outcome = service.advance_ready_checkpoint_policy(
                run_group_id,
                trial_segment_id=str(item.launch_spec.get("trial_segment_id") or "") or None,
            )
            catalog.add_event(
                item.id,
                "checkpoint_policy_checked",
                payload=adaptive_outcome,
            )
        except Exception as exc:
            self._orchestration_recovery_complete = False
            catalog.add_event(
                item.id,
                "checkpoint_policy_auto_advance_failed",
                payload={"run_group_id": run_group_id, "error": str(exc)},
            )
            self._append_log(
                self._resolve_log_path(item),
                f"automatic checkpoint-policy advance failed: {exc}\n",
            )
            return
        if terminal.status != "completed":
            return
        try:
            outcome = service.advance_ready_successive_halving(run_group_id)
            catalog.add_event(
                item.id,
                "successive_halving_checked",
                payload=outcome,
            )
        except Exception as exc:
            self._orchestration_recovery_complete = False
            # Evaluation publication is already complete. Auto-advance is a
            # follow-up decision and must never corrupt that terminal state;
            # the durable event and log make a manual advance straightforward.
            catalog.add_event(
                item.id,
                "successive_halving_auto_advance_failed",
                payload={"run_group_id": run_group_id, "error": str(exc)},
            )
            self._append_log(
                self._resolve_log_path(item),
                f"automatic successive-halving advance failed: {exc}\n",
            )

    def _execute_adaptive_lab(self, item: WorkItemRecord) -> WorkItemRecord:
        """Run adaptive evidence publication under the claimed durable lease."""

        from halo_forge.adaptive_lab import AdaptiveLabService

        stop, thread, heartbeat_errors = self._start_publication_heartbeat(item)
        try:
            result = AdaptiveLabService(self.scheduler.database).execute_work_item(item)
            if heartbeat_errors:
                raise LostClaimError(heartbeat_errors[-1])
            current = self.scheduler.database.get_work_item(item.id)
            # Compatibility for custom handlers that already finalize the
            # claimed record. The built-in adaptive service intentionally
            # leaves finalization to this worker.
            if current is not None and current.status in {
                "completed",
                "failed",
                "cancelled",
                "interrupted",
            }:
                return current
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                raise LostClaimError(
                    f"could not complete adaptive work item {item.id}: claim was lost"
                )
            return finished
        except Exception as exc:
            bundle_id = str(item.launch_spec.get("evidence_bundle_id") or item.domain_id or "")

            def prepare_retry() -> None:
                if bundle_id:
                    self.scheduler.database.update_evidence_bundle(
                        bundle_id,
                        status="queued",
                        error="",
                    )

            return self._finish_domain_failure(
                item,
                result={"evidence_bundle_id": bundle_id},
                error=f"Adaptive evidence worker failed: {exc}",
                prepare_retry=prepare_retry,
            )
        finally:
            stop.set()
            thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self._finalize_telemetry(item)

    def _execute_review_lab(self, item: WorkItemRecord) -> WorkItemRecord:
        """Run an acquisition, suggestion, or publication under a durable lease."""

        from halo_forge.review_lab import ReviewLabService

        log_path = self._resolve_log_path(item)
        stop, thread, heartbeat_errors = self._start_publication_heartbeat(item)
        try:
            service = ReviewLabService(
                self.scheduler.database,
                root=item.launch_spec.get("review_root"),
            )

            def ensure_not_cancelled() -> None:
                current = self.scheduler.database.get_work_item(item.id)
                if current is None or current.cancel_requested:
                    raise RuntimeError("review publication cancellation requested")

            action = str(item.launch_spec.get("action") or "").strip().lower()
            self._append_log(
                log_path,
                f"Starting Review Studio operation {action or item.kind}.\n",
            )
            if action == "publish_label_set":
                ensure_not_cancelled()
                result = service.publish_label_set(
                    str(item.launch_spec.get("queue_id") or item.domain_id or ""),
                    name=item.launch_spec.get("name"),
                    output_adapter_id=item.launch_spec.get("output_adapter_id"),
                    build_mode=item.launch_spec.get("build_mode"),
                    check_cancelled=ensure_not_cancelled,
                ).to_dict()
            else:
                result = service.execute_work_item(item)
            if heartbeat_errors:
                raise LostClaimError(heartbeat_errors[-1])
            self._append_log(
                log_path,
                f"Completed Review Studio operation {action or item.kind}.\n",
            )
            current = self.scheduler.database.get_work_item(item.id)
            if current is not None and current.status in {
                "completed",
                "failed",
                "cancelled",
                "interrupted",
            }:
                return current
            finished = self.scheduler.complete(
                item,
                result=result,
                ignore_late_cancel=action == "publish_label_set",
            )
            if finished is None:
                raise LostClaimError(
                    f"could not complete review work item {item.id}: claim was lost"
                )
            return finished
        except Exception as exc:
            self._append_log(log_path, f"Review Studio operation failed: {exc}\n")
            return self._finish_domain_failure(
                item,
                result={"review_operation": item.launch_spec.get("action")},
                error=f"Review Studio worker failed: {exc}",
                prepare_retry=lambda: None,
            )
        finally:
            stop.set()
            thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self._finalize_telemetry(item)

    def _execute_review_suggestion(self, item: WorkItemRecord) -> WorkItemRecord:
        """Generate one requested suggestion without persisting provider secrets."""

        from halo_forge.public_api.service import PublicApiService

        log_path = self._resolve_log_path(item)
        stop, thread, heartbeat_errors = self._start_publication_heartbeat(item)
        try:
            self._append_log(
                log_path,
                f"Generating review suggestion for item {item.launch_spec.get('item_id')}.\n",
            )
            service = PublicApiService(
                database=self.scheduler.database,
                review_storage_root=Path(str(item.launch_spec.get("review_root") or "")),
                dataset_storage_root=Path(str(item.launch_spec.get("dataset_root") or "")),
                evaluation_storage_root=Path(str(item.launch_spec.get("evaluation_root") or "")),
                artifact_storage_root=Path(str(item.launch_spec.get("artifact_root") or "")),
                workstation_scheduler=self.scheduler,
            )

            def ensure_not_cancelled() -> None:
                current = self.scheduler.database.get_work_item(item.id)
                if current is None or current.cancel_requested:
                    raise RuntimeError("review suggestion cancellation requested")

            ensure_not_cancelled()
            result = service.generate_review_suggestions(
                str(item.launch_spec.get("item_id") or ""),
                dict(item.launch_spec.get("payload") or {}),
                cancellation_check=ensure_not_cancelled,
            )
            if heartbeat_errors:
                raise LostClaimError(heartbeat_errors[-1])
            self._append_log(
                log_path,
                f"Published review suggestion {result.get('id', item.domain_id)}.\n",
            )
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                raise LostClaimError(
                    f"could not complete review suggestion {item.id}: claim was lost"
                )
            return finished
        except Exception as exc:
            self._append_log(log_path, f"Review suggestion failed: {exc}\n")
            return self._finish_domain_failure(
                item,
                result={"suggestion_id": item.domain_id},
                error=f"Review suggestion worker failed: {exc}",
                prepare_retry=lambda: None,
            )
        finally:
            stop.set()
            thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self._finalize_telemetry(item)

    def _execute_review_acquisition(self, item: WorkItemRecord) -> WorkItemRecord:
        """Resolve paged evidence sources and atomically publish a review batch."""

        from halo_forge.public_api.service import PublicApiService
        from halo_forge.review_lab import ReviewLabService

        batch_id = str(item.launch_spec.get("batch_id") or item.domain_id or "")
        review_root = Path(str(item.launch_spec.get("review_root") or ""))
        log_path = self._resolve_log_path(item)
        stop, thread, heartbeat_errors = self._start_publication_heartbeat(item)
        try:
            self._append_log(
                log_path,
                f"Resolving review acquisition {batch_id} from pinned sources.\n",
            )
            service = PublicApiService(
                database=self.scheduler.database,
                review_storage_root=review_root,
                dataset_storage_root=Path(str(item.launch_spec.get("dataset_root") or "")),
                evaluation_storage_root=Path(str(item.launch_spec.get("evaluation_root") or "")),
                artifact_storage_root=Path(str(item.launch_spec.get("artifact_root") or "")),
                workstation_scheduler=self.scheduler,
            )
            result = service.resolve_acquisition_batch(
                batch_id,
                dict(item.launch_spec.get("payload") or {}),
                work_item_id=item.id,
            )
            if heartbeat_errors:
                raise LostClaimError(heartbeat_errors[-1])
            self._append_log(
                log_path,
                f"Published review acquisition {result.get('id', batch_id)} with "
                f"{result.get('row_count', 0)} selected records.\n",
            )
            finished = self.scheduler.complete(
                item,
                result=result,
                # A ready acquisition has crossed the atomic manifest
                # publication boundary and completed catalog reconciliation.
                # Cancellation racing after that boundary is late and must not
                # create a cancelled work item pointing at a ready domain.
                ignore_late_cancel=str(result.get("status") or "") == "ready",
            )
            if finished is None:
                raise LostClaimError(
                    f"could not complete review acquisition {item.id}: claim was lost"
                )
            return finished
        except Exception as exc:
            self._append_log(log_path, f"Review acquisition failed: {exc}\n")
            review = ReviewLabService(self.scheduler.database, root=review_root)
            current_work = self.scheduler.database.get_work_item(item.id)
            if current_work is not None and current_work.cancel_requested:
                try:
                    review.cancel_acquisition(batch_id)
                except Exception:
                    pass
                return self._finish_domain_failure(
                    item,
                    result={"acquisition_batch_id": batch_id},
                    error="Review acquisition cancelled before publication",
                    prepare_retry=lambda: None,
                )
            try:
                review.mark_acquisition_failed(batch_id, str(exc))
            except Exception:
                pass

            def prepare_retry() -> None:
                review.retry_acquisition(batch_id)

            return self._finish_domain_failure(
                item,
                result={"acquisition_batch_id": batch_id},
                error=f"Review acquisition worker failed: {exc}",
                prepare_retry=prepare_retry,
            )
        finally:
            stop.set()
            thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self._finalize_telemetry(item)

    def _sync_managed_run_record(self, item: WorkItemRecord, terminal: WorkItemRecord) -> None:
        """Mirror the durable terminal event into the canonical run index."""

        from halo_forge.run_db import RunRecord

        run_id = str(item.canonical_run_id or terminal.canonical_run_id or "").strip()
        if not run_id:
            return
        config = dict(item.launch_spec.get("resolved_launch_config") or {})
        record = self.scheduler.database.get_run(run_id) or RunRecord(run_id=run_id)
        result = dict(terminal.result or {})
        summary = dict(result.get("training_summary") or {})
        output_dir = str(
            result.get("output_dir") or item.launch_spec.get("output_dir") or record.output_dir
        )
        record.fs_id = run_id
        record.modality = str(config.get("mode") or summary.get("modality") or record.modality)
        record.model_name = str(
            config.get("training_plan_model_id")
            or config.get("model")
            or summary.get("model_name")
            or record.model_name
        )
        record.status = terminal.status
        record.output_dir = output_dir
        record.timestamp = terminal.completed_at or terminal.updated_at or record.timestamp
        record.seed = int(config.get("seed") or summary.get("seed") or record.seed or 42)
        record.failure_reason = terminal.error if terminal.status != "completed" else None
        record.cycles_executed = int(
            summary.get("cycles_executed")
            or len(summary.get("cycles") or [])
            or record.cycles_executed
        )
        record.total_train_steps = int(
            summary.get("total_train_steps_executed")
            or summary.get("training_steps")
            or record.total_train_steps
        )
        record.final_train_loss = summary.get(
            "final_train_loss", summary.get("training_loss", record.final_train_loss)
        )
        record.weights_updated = bool(
            summary.get("weights_updated")
            or record.total_train_steps > 0
            or result.get("artifact_occurrence_ids")
        )
        artifacts = self.scheduler.database.list_model_artifacts(run_id=run_id)
        preferred = next(
            (
                artifact
                for artifact in reversed(artifacts)
                if artifact.artifact_kind in {"final_model", "adapter"}
            ),
            artifacts[-1] if artifacts else None,
        )
        record.final_model_path = preferred.path if preferred is not None else None
        record.raw_json = json.dumps(
            {
                **record.raw,
                **summary,
                "run_id": run_id,
                "status": terminal.status,
                "work_item_id": item.id,
                "output_dir": output_dir,
                "artifact_occurrence_ids": result.get("artifact_occurrence_ids", []),
                "model_artifact_ids": result.get("model_artifact_ids", []),
            },
            sort_keys=True,
            default=str,
        )
        output_path = Path(output_dir) if output_dir else None
        if output_path is not None and output_path.exists():
            record.source_mtime = output_path.stat().st_mtime
        self.scheduler.database.upsert_run(record)

    def _start_domain_heartbeat(
        self,
        item: WorkItemRecord,
        *,
        snapshot: Callable[[], tuple[str, Mapping[str, Any]]],
        cancel_domain: Callable[[], None],
    ) -> tuple[threading.Event, threading.Thread, list[str]]:
        """Mirror an in-process domain job into its durable work-item lease."""

        stop = threading.Event()
        errors: list[str] = []

        def keep_alive() -> None:
            next_telemetry = self.monotonic()
            while not stop.wait(min(2.0, self.heartbeat_interval)):
                current = self.scheduler.database.get_work_item(item.id)
                if current is None or current.status != "running":
                    if current is None or current.status not in {
                        "completed",
                        "failed",
                        "cancelled",
                    }:
                        errors.append("domain work-item claim is no longer active")
                    return
                if current.cancel_requested or self.stop_event.is_set():
                    try:
                        cancel_domain()
                    except Exception as exc:  # cancellation remains best effort
                        errors.append(f"could not propagate domain cancellation: {exc}")
                    if self.stop_event.is_set() and not current.cancel_requested:
                        self.scheduler.cancel(item.id)
                try:
                    stage, progress = snapshot()
                except Exception as exc:
                    errors.append(f"could not read domain progress: {exc}")
                    stage, progress = "running", {}
                try:
                    refreshed = self.scheduler.heartbeat(
                        item,
                        stage=stage or "running",
                        progress=dict(progress),
                    )
                except Exception as exc:
                    # A domain publication may briefly own SQLite's write
                    # lock.  Missing one heartbeat inside the existing lease
                    # is safe; sharing or rolling back the domain transaction
                    # is not.  A sustained lock still becomes a rejected or
                    # expired claim on a later iteration.
                    if "database is locked" in str(exc).lower():
                        continue
                    errors.append(f"domain work-item heartbeat failed: {exc}")
                    return
                if refreshed is None:
                    errors.append("domain work-item heartbeat was rejected")
                    return
                if self.monotonic() >= next_telemetry:
                    self._record_telemetry(item, process_pid=os.getpid())
                    next_telemetry = self.monotonic() + TELEMETRY_SAMPLE_INTERVAL_SECONDS

        initial_stage, initial_progress = snapshot()
        if (
            self.scheduler.heartbeat(
                item,
                stage=initial_stage or "starting",
                progress=dict(initial_progress),
            )
            is None
        ):
            raise LostClaimError("domain work-item claim was lost before execution")
        thread = threading.Thread(
            target=keep_alive,
            name=f"halo-forge-domain-heartbeat-{item.id}",
            daemon=True,
        )
        thread.start()
        return stop, thread, errors

    def _finish_domain_failure(
        self,
        item: WorkItemRecord,
        *,
        result: Mapping[str, Any],
        error: str,
        prepare_retry: Callable[[], None],
    ) -> WorkItemRecord:
        if not item.claim_token:
            raise LostClaimError(f"cannot fail work item {item.id}: claim token is missing")
        finished = self.scheduler.database.finish_work_item(
            item.id,
            claim_token=item.claim_token,
            result=result,
            error=error,
        )
        if finished is None:
            raise LostClaimError(f"could not fail work item {item.id}: claim was lost")
        if finished.status == "cancelled":
            return finished
        retried = self.scheduler.retry(
            finished.id,
            force=False,
            reason=f"automatic retry after domain failure: {error}",
            sync_domain=False,
        )
        if retried is not None:
            try:
                prepare_retry()
            except Exception as exc:
                LabV4Catalog(self.scheduler.database).add_event(
                    item.id,
                    "domain_retry_sync_failed",
                    payload={"error": str(exc)},
                )
                self.scheduler.cancel(item.id)
                return self.scheduler.database.get_work_item(item.id) or retried
        return retried or finished

    def _execute_dataset_lab(self, item: WorkItemRecord) -> WorkItemRecord:
        from halo_forge.data_lab import DatasetLab

        root_value = str(item.launch_spec.get("dataset_root") or "").strip()
        root = Path(root_value).expanduser()
        job_id = str(item.launch_spec.get("dataset_job_id") or item.domain_id or "")
        if not job_id or not root_value:
            return self._finish_failed(item, {}, "dataset work item is missing its job identity")
        database_path = str(item.launch_spec.get("database_path") or "").strip()
        database = self.scheduler.database
        if database_path and str(getattr(database, "path", "")) != database_path:
            from halo_forge.run_db import get_database

            database = get_database(database_path)
        lab = DatasetLab(root, database=database)

        def snapshot() -> tuple[str, Mapping[str, Any]]:
            job = lab.get_job(job_id)
            return job.stage, {
                "domain_status": job.status,
                "processed": job.processed,
                "total": job.total,
                "accepted": job.accepted,
                "rejected": job.rejected,
                "output_size_bytes": job.output_size_bytes,
                "latest_log": job.logs[-1] if job.logs else None,
            }

        stop = None
        thread = None
        errors: list[str] = []
        try:
            stop, thread, errors = self._start_domain_heartbeat(
                item,
                snapshot=snapshot,
                cancel_domain=lambda: lab.cancel(job_id),
            )
            job = lab.run_queued(job_id)
        except Exception as exc:
            result = {"dataset_job_id": job_id}
            return self._finish_domain_failure(
                item,
                result=result,
                error=f"Dataset Lab worker failed: {exc}",
                prepare_retry=lambda: lab.job_manager.reset_for_retry(job_id),
            )
        finally:
            if stop is not None:
                stop.set()
            if thread is not None:
                thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self._finalize_telemetry(item)
            lab.close()

        result = {
            "dataset_job_id": job.id,
            "dataset_job": job.to_dict(),
        }
        if errors and job.status not in {"succeeded", "failed", "cancelled"}:
            raise LostClaimError(errors[-1])
        if job.status == "succeeded":
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                raise LostClaimError(f"could not complete dataset work item {item.id}")
            return finished
        if job.status == "cancelled":
            self.scheduler.cancel(item.id)
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                current = self.scheduler.database.get_work_item(item.id)
                if current is not None and current.status == "cancelled":
                    return current
                raise LostClaimError(f"could not cancel dataset work item {item.id}")
            return finished
        return self._finish_domain_failure(
            item,
            result=result,
            error=job.error or "Dataset Lab job failed",
            prepare_retry=lambda: lab.job_manager.reset_for_retry(job_id),
        )

    def _execute_own_data_inspection(self, item: WorkItemRecord) -> WorkItemRecord:
        from halo_forge.own_data import GuidedOwnDataService

        inspection_id = str(item.launch_spec.get("inspection_id") or item.domain_id or "")
        dataset_root = str(item.launch_spec.get("dataset_root") or "").strip()
        imports_root = str(item.launch_spec.get("imports_root") or "").strip()
        if not inspection_id or not dataset_root or not imports_root:
            return self._finish_failed(
                item, {}, "guided dataset inspection is missing its durable identity or roots"
            )
        service = GuidedOwnDataService(
            self.scheduler.database,
            datasets_root=dataset_root,
            imports_root=imports_root,
            scheduler=self.scheduler,
        )

        def prepare_retry() -> None:
            inspection = self.scheduler.database.get_dataset_source_inspection(
                inspection_id
            )
            self.scheduler.database.update_dataset_source_inspection(
                inspection_id,
                status="queued",
                error=None,
                completed_at=None,
            )
            if inspection is not None and inspection.import_id:
                self.scheduler.database.update_dataset_import(
                    inspection.import_id,
                    status="inspecting",
                    work_item_id=item.id,
                    error=None,
                    completed_at=None,
                )

        try:
            inspection = service.execute_inspection(inspection_id)
        except Exception as exc:
            current = self.scheduler.database.get_work_item(item.id)
            if current is not None and current.status == "cancelled":
                return current
            return self._finish_domain_failure(
                item,
                result={"inspection_id": inspection_id},
                error=f"Dataset inspection failed: {exc}",
                prepare_retry=prepare_retry,
            )
        finished = self.scheduler.complete(
            item,
            result={"inspection_id": inspection_id, "inspection": inspection},
        )
        if finished is None:
            raise LostClaimError(f"could not complete dataset inspection work item {item.id}")
        return finished

    def _execute_corpus_extraction(self, item: WorkItemRecord) -> WorkItemRecord:
        from halo_forge.corpus_lab import CorpusExtractionService

        extraction_id = str(
            item.launch_spec.get("extraction_id") or item.domain_id or ""
        ).strip()
        corpus_root = str(item.launch_spec.get("corpus_root") or "").strip()
        if not extraction_id or not corpus_root:
            return self._finish_failed(
                item,
                {},
                "corpus extraction is missing its durable identity or storage root",
            )
        service = CorpusExtractionService(
            self.scheduler.database,
            root=corpus_root,
            scheduler=self.scheduler,
        )

        def prepare_retry() -> None:
            self.scheduler.database.update_document_extraction(
                extraction_id,
                status="queued",
                error=None,
                completed_at=None,
                work_item_id=item.id,
            )

        try:
            result = service.execute(extraction_id)
        except Exception as exc:
            current = self.scheduler.database.get_work_item(item.id)
            if current is not None and current.status == "cancelled":
                return current
            return self._finish_domain_failure(
                item,
                result={"extraction_id": extraction_id},
                error=f"Corpus extraction failed: {exc}",
                prepare_retry=prepare_retry,
            )
        finished = self.scheduler.complete(
            item,
            result={
                "extraction_id": extraction_id,
                "extraction": result.extraction,
            },
            ignore_late_cancel=True,
        )
        if finished is None:
            raise LostClaimError(
                f"could not complete corpus extraction work item {item.id}"
            )
        return finished

    def _own_data_public_service(self, item: WorkItemRecord) -> Any:
        from halo_forge.public_api.service import PublicApiService

        dataset_root = str(item.launch_spec.get("dataset_root") or "").strip()
        imports_root = str(item.launch_spec.get("imports_root") or "").strip()
        if not dataset_root or not imports_root:
            raise ValueError("guided dataset work is missing its storage roots")
        return PublicApiService(
            base_path=Path(dataset_root).expanduser().resolve().parent,
            database=self.scheduler.database,
            dataset_storage_root=Path(dataset_root),
            dataset_import_root=Path(imports_root),
            workstation_scheduler=self.scheduler,
        )

    def _execute_own_data_registration(self, item: WorkItemRecord) -> WorkItemRecord:
        inspection_id = str(item.launch_spec.get("inspection_id") or "").strip()
        import_id = str(item.launch_spec.get("import_id") or item.domain_id or "").strip()
        dataset_id = str(item.launch_spec.get("dataset_id") or "").strip()
        source_id = str(item.launch_spec.get("source_id") or "").strip()
        payload = item.launch_spec.get("registration_payload")
        if (
            not inspection_id
            or not import_id
            or not dataset_id
            or not source_id
            or not isinstance(payload, Mapping)
        ):
            return self._finish_failed(
                item, {}, "guided dataset registration is missing its durable identity"
            )

        service = self._own_data_public_service(item)
        log_path = self._resolve_log_path(item)
        stop, thread, heartbeat_errors = self._start_publication_heartbeat(item)

        def prepare_retry() -> None:
            self.scheduler.database.update_dataset_import(
                import_id,
                status="completed",
                work_item_id=item.id,
                error=None,
            )

        try:
            self._append_log(log_path, "Publishing guided dataset registration.\n")
            current = self.scheduler.database.get_work_item(item.id)
            if current is None or current.cancel_requested:
                raise RuntimeError("guided dataset registration cancellation requested")
            result = service.execute_inspected_dataset_registration(
                inspection_id,
                payload,
                dataset_id=dataset_id,
                source_id=source_id,
            )
            if heartbeat_errors:
                raise LostClaimError(heartbeat_errors[-1])
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                current = self.scheduler.database.get_work_item(item.id)
                if current is not None and current.status == "cancelled":
                    return current
                raise LostClaimError(
                    f"could not complete guided dataset registration {item.id}"
                )
            return finished
        except Exception as exc:
            current = self.scheduler.database.get_work_item(item.id)
            if current is not None and current.status == "cancelled":
                return current
            self.scheduler.database.update_dataset_import(
                import_id,
                status="failed",
                work_item_id=item.id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return self._finish_domain_failure(
                item,
                result={
                    "inspection_id": inspection_id,
                    "import_id": import_id,
                    "dataset_id": dataset_id,
                    "source_id": source_id,
                },
                error=f"Guided dataset registration failed: {exc}",
                prepare_retry=prepare_retry,
            )
        finally:
            stop.set()
            thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self._finalize_telemetry(item)
            lab = getattr(service, "_dataset_lab", None)
            if lab is not None:
                lab.close()

    def _execute_own_data_source_refresh(self, item: WorkItemRecord) -> WorkItemRecord:
        source_id = str(item.launch_spec.get("source_id") or item.domain_id or "").strip()
        if not source_id:
            return self._finish_failed(
                item, {}, "dataset source refresh is missing its source identity"
            )
        service = self._own_data_public_service(item)
        log_path = self._resolve_log_path(item)
        stop, thread, heartbeat_errors = self._start_publication_heartbeat(item)
        try:
            self._append_log(log_path, "Checking referenced source for a new revision.\n")
            current = self.scheduler.database.get_work_item(item.id)
            if current is None or current.cancel_requested:
                raise RuntimeError("dataset source refresh cancellation requested")
            source = service.execute_dataset_source_refresh(source_id)
            result = {"source_id": source_id, "source": source}
            if heartbeat_errors:
                raise LostClaimError(heartbeat_errors[-1])
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                current = self.scheduler.database.get_work_item(item.id)
                if current is not None and current.status == "cancelled":
                    return current
                raise LostClaimError(f"could not complete source refresh {item.id}")
            return finished
        except Exception as exc:
            current = self.scheduler.database.get_work_item(item.id)
            if current is not None and current.status == "cancelled":
                return current
            return self._finish_domain_failure(
                item,
                result={"source_id": source_id},
                error=f"Dataset source refresh failed: {exc}",
                prepare_retry=lambda: None,
            )
        finally:
            stop.set()
            thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self._finalize_telemetry(item)
            lab = getattr(service, "_dataset_lab", None)
            if lab is not None:
                lab.close()

    def _execute_evaluation_lab(self, item: WorkItemRecord) -> WorkItemRecord:
        from halo_forge.evaluation_lab import EvaluationLabService

        evaluation_id = str(item.launch_spec.get("evaluation_id") or item.domain_id or "")
        artifact_root = item.launch_spec.get("artifact_root")
        if not evaluation_id:
            return self._finish_failed(item, {}, "evaluation work item has no evaluation ID")
        service = EvaluationLabService(
            self.scheduler.database,
            artifact_root=artifact_root,
        )

        def snapshot() -> tuple[str, Mapping[str, Any]]:
            evaluation = service.jobs.get(evaluation_id)
            return evaluation.stage, {
                "domain_status": evaluation.status,
                "processed": evaluation.processed_samples,
                "total": evaluation.total_samples,
                "latest_log": evaluation.logs[-1] if evaluation.logs else None,
            }

        stop = None
        thread = None
        errors: list[str] = []
        try:
            stop, thread, errors = self._start_domain_heartbeat(
                item,
                snapshot=snapshot,
                cancel_domain=lambda: service.jobs.cancel(evaluation_id),
            )
            evaluation = service.jobs.run_queued(evaluation_id)
        except Exception as exc:
            result = {"evaluation_id": evaluation_id}
            return self._finish_domain_failure(
                item,
                result=result,
                error=f"Evaluation Lab worker failed: {exc}",
                prepare_retry=lambda: service.jobs.retry(evaluation_id, submit=False),
            )
        finally:
            if stop is not None:
                stop.set()
            if thread is not None:
                thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self._finalize_telemetry(item)
            service.shutdown(wait=False, cancel_futures=False)

        result: dict[str, Any] = {"evaluation_id": evaluation.id}
        if errors and evaluation.status not in {"completed", "failed", "cancelled"}:
            raise LostClaimError(errors[-1])
        if evaluation.status == "completed":
            verification_error: Optional[str] = None
            if evaluation.artifact_path:
                try:
                    service.jobs._verify_published(
                        evaluation,
                        Path(evaluation.artifact_path),
                    )
                except Exception as exc:
                    verification_error = str(exc)
            revision = self.scheduler.database.get_benchmark_suite_revision(
                evaluation.suite_revision_id
            )
            metrics = {
                metric.name: float(metric.value)
                for metric in self.scheduler.database.list_evaluation_metrics(evaluation.id)
                if not metric.suite_item_id
            }
            if verification_error:
                evaluation = (
                    self.scheduler.database.update_evaluation(
                        evaluation.id,
                        status="failed",
                        stage="verification_failed",
                        error=f"evaluation evidence verification failed: {verification_error}",
                    )
                    or evaluation
                )
            elif revision is None or revision.primary_metric not in metrics:
                evaluation = (
                    self.scheduler.database.update_evaluation(
                        evaluation.id,
                        status="failed",
                        stage="verification_failed",
                        error="completed evaluation has no primary metric",
                    )
                    or evaluation
                )
            elif self.scheduler.database.count_evaluation_samples(evaluation.id) <= 0:
                evaluation = (
                    self.scheduler.database.update_evaluation(
                        evaluation.id,
                        status="failed",
                        stage="verification_failed",
                        error="completed evaluation has no sample evidence",
                    )
                    or evaluation
                )
            elif not evaluation.artifact_path or not Path(evaluation.artifact_path).is_dir():
                evaluation = (
                    self.scheduler.database.update_evaluation(
                        evaluation.id,
                        status="failed",
                        stage="verification_failed",
                        error="completed evaluation evidence bundle is missing",
                    )
                    or evaluation
                )
            else:
                result.update(
                    metrics=metrics,
                    sample_count=self.scheduler.database.count_evaluation_samples(evaluation.id),
                    artifact_path=evaluation.artifact_path,
                )
                finished = self.scheduler.complete(item, result=result)
                if finished is None:
                    raise LostClaimError(f"could not complete evaluation work item {item.id}")
                return finished
        if evaluation.status == "cancelled":
            self.scheduler.cancel(item.id)
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                current = self.scheduler.database.get_work_item(item.id)
                if current is not None and current.status == "cancelled":
                    return current
                raise LostClaimError(f"could not cancel evaluation work item {item.id}")
            return finished
        return self._finish_domain_failure(
            item,
            result=result,
            error=evaluation.error or "Evaluation Lab job failed verification",
            prepare_retry=lambda: service.jobs.retry(evaluation_id, submit=False),
        )

    def _execute_reward_integrity(self, item: WorkItemRecord) -> WorkItemRecord:
        """Rescore a sealed exact-output shard through its pinned auditors."""

        from halo_forge.reward_integrity import RewardIntegrityService
        from halo_forge.reward_integrity.runtime import execute_pinned_audit

        audit_id = str(item.launch_spec.get("audit_id") or item.domain_id or "").strip()
        if not audit_id:
            return self._finish_failed(item, {}, "reward audit work item has no audit ID")
        root = item.launch_spec.get("reward_integrity_root")
        service = RewardIntegrityService(self.scheduler.database, root=root)

        def snapshot() -> tuple[str, Mapping[str, Any]]:
            audit = service.get_audit(audit_id)
            return audit.stage, {
                "domain_status": audit.status,
                "processed": audit.processed_samples,
                "total": audit.total_samples,
                "coverage": (
                    audit.processed_samples / audit.total_samples
                    if audit.total_samples
                    else 0.0
                ),
            }

        stop = None
        thread = None
        errors: list[str] = []
        try:
            audit = service.get_audit(audit_id)
            if audit.status != "completed":
                stop, thread, errors = self._start_domain_heartbeat(
                    item,
                    snapshot=snapshot,
                    cancel_domain=lambda: service.cancel_audit(audit_id),
                )
                audit = execute_pinned_audit(
                    self.scheduler.database,
                    audit_id,
                    root=str(root) if root else None,
                )
        except Exception as exc:
            try:
                service.store.update_audit(
                    audit_id,
                    status="failed",
                    stage="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                pass
            return self._finish_domain_failure(
                item,
                result={"audit_id": audit_id},
                error=f"Reward Integrity audit failed: {exc}",
                prepare_retry=lambda: service.retry_audit(
                    audit_id,
                    reason="automatic retry after reward-audit execution failure",
                ),
            )
        finally:
            if stop is not None:
                stop.set()
            if thread is not None:
                thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self._finalize_telemetry(item)

        if errors and audit.status not in {"completed", "cancelled"}:
            raise LostClaimError(errors[-1])
        result = {
            "audit_id": audit.id,
            "status": audit.status,
            "artifact_path": audit.artifact_path,
            "manifest_hash": audit.manifest_hash,
            "processed_samples": audit.processed_samples,
        }
        if audit.status == "completed":
            decisions = service.store.list_decisions(audit.id, limit=1000)
            latest_decision = decisions.items[-1] if decisions.items else None
            if latest_decision is not None:
                result["decision"] = latest_decision.to_dict()
                result["requires_review"] = latest_decision.action == "pause"
            run = self.scheduler.database.get_run(audit.run_id)
            if run is not None:
                result["run_status"] = run.status
            verification = service.verify_audit_bundle(audit.id)
            result["verification"] = verification
            try:
                result["replay_sync"] = service.sync_audit_replay(
                    audit.id, decision=latest_decision
                )
            except Exception as exc:
                return self._finish_domain_failure(
                    item,
                    result=result,
                    error=f"Reward Integrity replay synchronization failed: {exc}",
                    # The immutable audit is already valid. A retry only needs
                    # to revisit replay publication, and the completed-audit
                    # fast path above makes that idempotent.
                    prepare_retry=lambda: None,
                )
            if (
                latest_decision is not None
                and latest_decision.action == "continue"
                and audit.direct_run_segment_id
            ):
                try:
                    from halo_forge.reward_integrity.direct_segments import (
                        enqueue_next_direct_segment,
                    )

                    next_work = enqueue_next_direct_segment(
                        self.scheduler.database,
                        self.scheduler,
                        current_segment_id=audit.direct_run_segment_id,
                        dependency_work_item_id=item.id,
                    )
                    if next_work is not None:
                        result["next_work_item_id"] = next_work.id
                except Exception as exc:
                    return self._finish_domain_failure(
                        item,
                        result=result,
                        error=f"Reward Integrity next-segment enqueue failed: {exc}",
                        prepare_retry=lambda: None,
                    )
            finished = self.scheduler.complete(item, result=result)
            if finished is None:
                raise LostClaimError(f"could not complete reward audit work item {item.id}")
            return finished
        if audit.status == "cancelled":
            self.scheduler.cancel(item.id)
            finished = self.scheduler.complete(item, result=result)
            if finished is not None:
                return finished
            current = self.scheduler.database.get_work_item(item.id)
            if current is not None and current.status == "cancelled":
                return current
            raise LostClaimError(f"could not cancel reward audit work item {item.id}")
        return self._finish_domain_failure(
            item,
            result=result,
            error=audit.error or "Reward Integrity audit did not publish",
            prepare_retry=lambda: service.retry_audit(
                audit_id,
                reason="automatic retry after unpublished reward-audit result",
            ),
        )

    def _execute_verifier_lab(self, item: WorkItemRecord) -> WorkItemRecord:
        """Run one immutable verifier calibration under a durable work lease."""

        from halo_forge.verifier_lab import VerifierLabService

        calibration_id = str(
            item.launch_spec.get("calibration_id") or item.domain_id or ""
        ).strip()
        if not calibration_id:
            return self._finish_failed(
                item, {}, "verifier calibration work item has no calibration ID"
            )
        service = VerifierLabService(
            self.scheduler.database,
            root=item.launch_spec.get("calibration_root"),
            scheduler=self.scheduler,
        )

        def snapshot() -> tuple[str, Mapping[str, Any]]:
            value = service.get_calibration(calibration_id)
            if value is None:
                return "missing", {"domain_status": "missing"}
            payload = value.to_dict() if hasattr(value, "to_dict") else dict(value)
            return str(payload.get("stage") or "running"), {
                "domain_status": payload.get("status"),
                "processed": payload.get("processed_records", 0),
                "total": payload.get("total_records"),
            }

        stop = None
        thread = None
        heartbeat_errors: list[str] = []
        try:
            stop, thread, heartbeat_errors = self._start_domain_heartbeat(
                item,
                snapshot=snapshot,
                cancel_domain=lambda: service.cancel_calibration(calibration_id),
            )
            result = service.run_calibration(
                calibration_id,
                work_item_id=item.id,
            )
            if heartbeat_errors:
                raise LostClaimError(heartbeat_errors[-1])
            payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            verification = service.verify_calibration(calibration_id)
            verified = (
                bool(verification.get("valid"))
                if isinstance(verification, Mapping)
                else bool(getattr(verification, "valid", False))
            )
            if not verified:
                raise RuntimeError("published verifier calibration bundle failed verification")
            finished = self.scheduler.complete(
                item,
                result={
                    "calibration_id": calibration_id,
                    "artifact_path": payload.get("artifact_path"),
                    "manifest_hash": payload.get("manifest_hash"),
                },
            )
            if finished is None:
                raise LostClaimError(
                    f"could not complete verifier calibration work item {item.id}"
                )
            return finished
        except Exception as exc:
            current_calibration = service.get_calibration(calibration_id)
            if (
                current_calibration is not None
                and getattr(current_calibration, "status", None) == "cancelled"
            ):
                self.scheduler.cancel(item.id)
                finished = self.scheduler.complete(
                    item,
                    result={"calibration_id": calibration_id, "status": "cancelled"},
                )
                if finished is not None:
                    return finished
                current_item = self.scheduler.database.get_work_item(item.id)
                if current_item is not None:
                    return current_item
            return self._finish_domain_failure(
                item,
                result={"calibration_id": calibration_id},
                error=f"Verifier calibration worker failed: {exc}",
                prepare_retry=lambda: service.prepare_retry(calibration_id),
            )
        finally:
            if stop is not None:
                stop.set()
            if thread is not None:
                thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self._finalize_telemetry(item)

    def _execute_artifact_studio(self, item: WorkItemRecord) -> WorkItemRecord:
        from halo_forge.artifact_studio import (
            ArtifactStudioService,
            SubprocessServingStarter,
        )
        from halo_forge.evaluation_lab import EvaluationLabService
        from halo_forge.qualification_lab import EvaluationQualificationExecutor

        stop, thread, heartbeat_errors = self._start_publication_heartbeat(item)
        evaluations = None
        try:
            catalog = LabV4Catalog(self.scheduler.database)
            evaluations = EvaluationLabService(
                self.scheduler.database,
                artifact_root=item.launch_spec.get("evaluation_artifact_root"),
            )
            executor = EvaluationQualificationExecutor(
                self.scheduler.database, catalog, evaluations
            )
            ArtifactStudioService(
                self.scheduler.database,
                catalog=catalog,
                scheduler=self.scheduler,
                qualification_executor=executor,
                serving_starter=SubprocessServingStarter(catalog),
                artifact_root=item.launch_spec.get("artifact_root"),
            ).execute_work_item(item.id)
        except Exception as exc:
            current = self.scheduler.database.get_work_item(item.id)
            if current is not None and current.status == "failed":
                retried = self.scheduler.retry(
                    current.id,
                    force=False,
                    reason=f"automatic retry after artifact operation failure: {exc}",
                )
                current = retried or current
            if current is None:
                raise
            return current
        finally:
            stop.set()
            thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
            self._finalize_telemetry(item)
            if evaluations is not None:
                evaluations.shutdown(wait=False, cancel_futures=False)
        current = self.scheduler.database.get_work_item(item.id)
        if heartbeat_errors and (current is None or current.status == "running"):
            raise LostClaimError(heartbeat_errors[-1])
        if current is None:
            raise LostClaimError(f"artifact work item {item.id} disappeared")
        return current

    def watch(self, *, once: bool = False) -> int:
        """Process one ready item or continuously watch the queue."""

        if once:
            self.run_once()
            return 0
        while not self.stop_event.is_set():
            completed = self.run_once()
            if completed is None and not self.stop_event.is_set():
                self.sleep(self.poll_interval)
        return 0

    def _execute(self, item: WorkItemRecord) -> WorkItemRecord:
        started = self.monotonic()
        command: list[str] = []
        log_path = self._resolve_log_path(item)
        base_result: dict[str, Any] = {
            "work_item_id": item.id,
            "log_path": str(log_path),
        }
        process: Optional[ProcessHandle] = None
        cancellation_seen = False
        shutdown_seen = False

        try:
            launch_spec = self._prepare_launch_spec(item)
            attempts = LabV4Catalog(self.scheduler.database).list_attempts(item.id)
            attempt = attempts[-1] if attempts else None
            if launch_spec.get("operation") in {
                "managed_training",
                "managed_training_segment",
            }:
                launch_spec, attempt = self._prepare_managed_training_attempt(
                    item, launch_spec, attempt
                )
            command, cwd, env = self._resolve_launch(launch_spec)
            runtime_revision_id = str(
                launch_spec.get("runtime_profile_revision_id")
                or item.resource_requirements.get("runtime_profile_revision_id")
                or ""
            ).strip()
            if runtime_revision_id:
                from halo_forge.managed_runtime import ManagedRuntimeService

                runtime_root = item.resource_requirements.get("runtime_root") or Path.home() / ".halo-forge" / "runtimes"
                runtime = ManagedRuntimeService(
                    self.scheduler.database,
                    root=runtime_root,
                    scheduler=self.scheduler,
                )
                command, cwd, runtime_env, qualification = runtime.wrap_execution(
                    runtime_revision_id,
                    command,
                    cwd=cwd,
                    launch_spec=launch_spec,
                )
                env.update(runtime_env)
                runtime.bind(
                    revision_id=runtime_revision_id,
                    qualification_id=qualification.id,
                    domain_kind=str(item.domain_kind or "work_item"),
                    domain_id=str(item.domain_id or item.id),
                    details={"work_item_id": item.id},
                )
            env["HALOFORGE_WORK_ITEM_ID"] = item.id
            if attempt is not None:
                env["HALOFORGE_ATTEMPT_ID"] = attempt.id
                if attempt.output_dir:
                    Path(attempt.output_dir).mkdir(parents=True, exist_ok=True)
                    env["HALOFORGE_ATTEMPT_DIR"] = attempt.output_dir
                    base_result["attempt_id"] = attempt.id
                    base_result["attempt_output_dir"] = attempt.output_dir
            base_result["command"] = command
            self.scheduler.database.update_work_item(item.id, log_path=str(log_path))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab", buffering=0) as log_file:
                allowed, availability = self.scheduler.confirm_pre_spawn(item)
                if not allowed:
                    current = self.scheduler.database.get_work_item(item.id)
                    if current is None:
                        raise LostClaimError("work item disappeared while waiting for accelerator")
                    return current
                process = self.runner.start(
                    command,
                    cwd=cwd,
                    env=env,
                    log_file=log_file,
                )
                child_started_at = self.process_identity(int(process.pid))
                if child_started_at is None:
                    raise RuntimeError(
                        f"could not resolve start identity for child process {process.pid}"
                    )
                bound = self.scheduler.bind_process(
                    item,
                    child_pid=int(process.pid),
                    child_pid_started_at=child_started_at,
                )
                if bound is None:
                    raise LostClaimError("work-item claim was lost before process binding")
                item = bound
                base_result.update(
                    pid=int(process.pid),
                    process_started_at=float(child_started_at),
                )
                return_code, cancellation_seen, shutdown_seen = self._monitor(item, process)
        except Exception as exc:
            if process is not None and process.poll() is None:
                self._stop_process(process)
            base_result["duration_seconds"] = max(0.0, self.monotonic() - started)
            self._append_log(log_path, f"worker error: {exc}\n")
            finished = self._finish_failed(item, base_result, str(exc))
            self._finalize_telemetry(item)
            return finished

        base_result.update(
            return_code=int(return_code),
            duration_seconds=max(0.0, self.monotonic() - started),
        )
        self._capture_result_file(launch_spec, base_result, cwd=cwd)

        if cancellation_seen:
            base_result["cancelled"] = True
            finished = self.scheduler.complete(item, result=base_result)
        elif shutdown_seen:
            finished = self._finish_failed(
                item,
                base_result,
                "worker shutdown requested while command was running",
            )
        elif return_code == 0:
            heartbeat_stop, heartbeat_thread, heartbeat_errors = self._start_publication_heartbeat(
                item
            )
            try:
                self._publish_training_artifacts(item, launch_spec, base_result)
                self._publish_managed_training(item, launch_spec, base_result)
                self._capture_evaluation_result(item, launch_spec, base_result)
            except Exception as exc:
                self._append_log(log_path, f"publication error: {exc}\n")
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
                finished = self._finish_failed(item, base_result, str(exc))
            else:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=max(1.0, self.heartbeat_interval * 2))
                if heartbeat_errors:
                    raise LostClaimError(heartbeat_errors[-1])
                finished = self.scheduler.complete(item, result=base_result)
        else:
            finished = self._finish_failed(
                item,
                base_result,
                f"command exited with status {return_code}",
            )
        if finished is None:
            raise LostClaimError(f"could not finalize work item {item.id}: claim was lost")
        self._finalize_telemetry(item)
        return finished

    def _monitor(self, item: WorkItemRecord, process: ProcessHandle) -> tuple[int, bool, bool]:
        next_heartbeat = self.monotonic()
        cancellation_seen = False
        shutdown_seen = False
        termination_started: Optional[float] = None
        killed = False
        next_telemetry = self.monotonic()
        next_contention_check = self.monotonic() + TELEMETRY_SAMPLE_INTERVAL_SECONDS
        contention_recorded = False

        while True:
            return_code = process.poll()
            if return_code is not None:
                return int(return_code), cancellation_seen, shutdown_seen

            now = self.monotonic()
            current = self.scheduler.database.get_work_item(item.id)
            if current is None or current.status != "running":
                self._stop_process(process)
                raise LostClaimError("work-item claim is no longer active")

            cancellation_seen = cancellation_seen or bool(current.cancel_requested)
            shutdown_seen = shutdown_seen or self.stop_event.is_set()
            should_stop = cancellation_seen or shutdown_seen
            if should_stop and termination_started is None:
                process.terminate()
                termination_started = now
            elif (
                should_stop
                and not killed
                and termination_started is not None
                and now - termination_started >= self.terminate_timeout
            ):
                process.kill()
                killed = True

            if now >= next_heartbeat:
                refreshed = self.scheduler.heartbeat(
                    item,
                    stage="cancelling" if should_stop else "running",
                    progress={
                        "pid": int(process.pid),
                        "cancel_requested": cancellation_seen,
                    },
                )
                if refreshed is None:
                    self._stop_process(process)
                    raise LostClaimError("work-item heartbeat was rejected")
                item = refreshed
                next_heartbeat = now + self.heartbeat_interval

            if now >= next_telemetry:
                self._record_telemetry(item, process_pid=int(process.pid))
                next_telemetry = now + TELEMETRY_SAMPLE_INTERVAL_SECONDS

            if not contention_recorded and now >= next_contention_check:
                contention = self.scheduler.late_contention(item)
                next_contention_check = now + TELEMETRY_SAMPLE_INTERVAL_SECONDS
                if contention is not None:
                    contention_recorded = True
                    resumable = bool(
                        item.resource_requirements.get("resumable")
                        or item.launch_spec.get("resumable")
                        or item.launch_spec.get("operation") in {"managed_training_segment", "train_trial_segment"}
                    )
                    progress = {
                        "pid": int(process.pid),
                        "contention_warning": True,
                        "performance_evidence_reusable": False,
                        "pause_at_checkpoint_boundary": resumable,
                        "external_owners": [
                            {
                                "pid": owner.pid,
                                "executable": owner.executable,
                                "elapsed_seconds": owner.elapsed_seconds,
                            }
                            for owner in contention.owners
                        ],
                    }
                    self.scheduler.heartbeat(item, stage="contention_warning", progress=progress)
                    self.scheduler.database._conn.execute(
                        "INSERT INTO accelerator_preflight_decisions "
                        "(id,work_item_id,runtime_revision_id,accelerator_family,decision,sample_count,evidence_hash,evidence_json,created_at) "
                        "VALUES (?,?,?,?, 'contention',1,?,?,?)",
                        (
                            f"accelerator-preflight-{uuid.uuid4().hex}",
                            item.id,
                            item.resource_requirements.get("runtime_profile_revision_id"),
                            str(item.resource_requirements.get("accelerator_family") or "unknown"),
                            hashlib.sha256(json.dumps(contention.to_dict(), sort_keys=True).encode()).hexdigest(),
                            json.dumps({"late_contention": contention.to_dict(), "resumable": resumable}, sort_keys=True),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    self.scheduler.database._conn.commit()

            self.sleep(self.poll_interval)

    def _start_publication_heartbeat(
        self, item: WorkItemRecord
    ) -> tuple[threading.Event, threading.Thread, list[str]]:
        """Keep ownership alive while hashing and atomically publishing outputs."""

        stop = threading.Event()
        errors: list[str] = []
        initial = self.scheduler.heartbeat(
            item,
            stage="publishing",
            progress={"phase": "hashing_and_publication"},
        )
        if initial is None:
            raise LostClaimError("work-item claim was lost before artifact publication")

        def keep_alive() -> None:
            interval = max(0.1, min(self.heartbeat_interval, 2.0))
            next_telemetry = time.monotonic()
            while not stop.wait(interval):
                refreshed = self.scheduler.heartbeat(
                    item,
                    stage="publishing",
                    progress={"phase": "hashing_and_publication"},
                )
                if refreshed is None:
                    current = self.scheduler.database.get_work_item(item.id)
                    if current is not None and current.status in {
                        "completed",
                        "failed",
                        "cancelled",
                    }:
                        return
                    errors.append("work-item claim was lost during artifact publication")
                    return
                if time.monotonic() >= next_telemetry:
                    self._record_telemetry(item, process_pid=os.getpid())
                    next_telemetry = time.monotonic() + TELEMETRY_SAMPLE_INTERVAL_SECONDS

        thread = threading.Thread(
            target=keep_alive,
            name=f"halo-forge-publish-heartbeat-{item.id}",
            daemon=True,
        )
        thread.start()
        return stop, thread, errors

    def _record_telemetry(self, item: WorkItemRecord, *, process_pid: int) -> None:
        try:
            attempts = LabV4Catalog(self.scheduler.database).list_attempts(item.id)
            attempt_id = attempts[-1].id if attempts else None
            output = (
                item.resource_requirements.get("output_path")
                or item.launch_spec.get("output_dir")
                or item.launch_spec.get("output_root")
            )
            if output:
                capacity_path = Path(str(output)).expanduser()
            elif self.scheduler.database.path != ":memory:":
                capacity_path = Path(self.scheduler.database.path).expanduser().parent
            else:
                capacity_path = Path.home() / ".halo-forge"
            capacity = self.telemetry_sampler(
                capacity_path,
                pid=process_pid,
                include_accelerator=item.resource_class == "accelerator",
            )
            sample = WorkstationTelemetrySample.from_capacity(
                item.id,
                capacity,
                attempt_id=attempt_id,
            )
            LabV4Catalog(self.scheduler.database).record_telemetry(sample.to_dict())
        except Exception:
            # Telemetry is observability, never a reason to fail the work. The
            # capacity sampler itself preserves missing metrics as nulls.
            return

    def _finalize_telemetry(self, item: WorkItemRecord) -> None:
        try:
            catalog = LabV4Catalog(self.scheduler.database)
            attempts = catalog.list_attempts(item.id)
            attempt_id = attempts[-1].id if attempts else None
            catalog.finalize_telemetry_rollup(item.id, attempt_id=attempt_id)
        except Exception:
            return

    def _stop_process(self, process: ProcessHandle) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        deadline = self.monotonic() + self.terminate_timeout
        while process.poll() is None and self.monotonic() < deadline:
            self.sleep(min(self.poll_interval, max(0.01, deadline - self.monotonic())))
        if process.poll() is None:
            process.kill()

    def _finish_failed(
        self,
        item: WorkItemRecord,
        result: Mapping[str, Any],
        error: str,
    ) -> WorkItemRecord:
        if not item.claim_token:
            raise LostClaimError(f"cannot fail work item {item.id}: claim token is missing")
        finished = self.scheduler.database.finish_work_item(
            item.id,
            claim_token=item.claim_token,
            result=result,
            error=error,
        )
        if finished is None:
            raise LostClaimError(f"could not fail work item {item.id}: claim was lost")
        retried = self.scheduler.retry(
            finished.id,
            force=False,
            reason=f"automatic retry after failure: {error}",
        )
        return retried or finished

    def _resolve_log_path(self, item: WorkItemRecord) -> Path:
        configured = item.log_path or item.launch_spec.get("log_path")
        if configured:
            return Path(str(configured)).expanduser()
        database_path = str(getattr(self.scheduler.database, "path", ""))
        if database_path and database_path != ":memory:":
            root = Path(database_path).expanduser().parent
        else:
            root = Path.home() / ".halo-forge"
        return root / "work-items" / f"{item.id}.log"

    def _prepare_launch_spec(self, item: WorkItemRecord) -> dict[str, Any]:
        """Resolve a dependency-produced checkpoint into executable argv."""
        launch_spec = dict(item.launch_spec)
        if launch_spec.get("command"):
            return launch_spec
        transport = launch_spec.get("command_transport")
        if not isinstance(transport, Mapping):
            return launch_spec
        if str(transport.get("status") or "") != "requires_checkpoint_resolution":
            return launch_spec

        segment_id = str(launch_spec.get("trial_segment_id") or "")
        segment = self.scheduler.database.get_trial_segment(segment_id) if segment_id else None
        operation = str(launch_spec.get("operation") or "")
        artifact = None
        if operation == "train_trial_segment":
            if segment is None:
                raise ValueError("checkpoint-gated training references a missing trial segment")
            previous = next(
                (
                    value
                    for value in self.scheduler.database.list_trial_segments(segment.trial_run_id)
                    if value.ordinal == segment.ordinal - 1
                ),
                None,
            )
            if previous is None or not previous.checkpoint_artifact_id:
                raise ValueError("previous trial segment has no published checkpoint artifact")
            artifact = self.scheduler.database.get_model_artifact(previous.checkpoint_artifact_id)
            resolved_path, resolved_occurrence_id = self._segment_artifact_path(
                previous.id,
                fallback_path=artifact.path if artifact is not None else None,
            )
            command = list(transport.get("command_template") or ())
            resolution = transport.get("checkpoint_resolution") or {}
            resume_flag = str(resolution.get("append_flag") or "")
            if not command or not resume_flag:
                raise ValueError("checkpoint-gated training has no executable command template")
            if artifact is None or resolved_path is None or not Path(resolved_path).exists():
                raise ValueError("previous checkpoint artifact is missing from local storage")
            command.extend((resume_flag, resolved_path))
        elif operation == "evaluate_trial_segment":
            if segment is None or not segment.checkpoint_artifact_id:
                raise ValueError("trial segment has no checkpoint artifact to evaluate")
            artifact = self.scheduler.database.get_model_artifact(segment.checkpoint_artifact_id)
            resolved_path, resolved_occurrence_id = self._segment_artifact_path(
                segment.id,
                fallback_path=artifact.path if artifact is not None else None,
            )
            if artifact is None or resolved_path is None or not Path(resolved_path).exists():
                raise ValueError("checkpoint artifact is missing from local storage")
            command = list(transport.get("command_prefix") or ())
            if not command:
                raise ValueError("checkpoint evaluation has no executable command prefix")
            command.extend(
                (
                    "--subject",
                    resolved_path,
                    "--request",
                    json.dumps(
                        {"trial_segment_id": segment.id},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "--wait",
                )
            )
        else:
            raise ValueError(f"unsupported checkpoint-resolution operation: {operation}")

        launch_spec["command"] = command
        launch_spec["resolved_checkpoint_artifact_id"] = artifact.id
        launch_spec["resolved_checkpoint_path"] = resolved_path
        if resolved_occurrence_id:
            launch_spec["resolved_checkpoint_occurrence_id"] = resolved_occurrence_id
        return launch_spec

    def _segment_artifact_path(
        self, segment_id: str, *, fallback_path: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        """Resolve the occurrence-local path before the legacy hash index."""

        row = self.scheduler.database._conn.execute(
            """
            SELECT occurrence.id, location.path
            FROM artifact_occurrences occurrence
            JOIN artifact_locations location ON location.blob_id = occurrence.blob_id
            WHERE occurrence.trial_segment_id = ?
              AND location.state = 'available'
            ORDER BY occurrence.created_at DESC,
                     CASE location.storage_mode WHEN 'managed' THEN 0 ELSE 1 END,
                     location.created_at DESC
            LIMIT 1
            """,
            (segment_id,),
        ).fetchone()
        if row is not None:
            return str(row["path"]), str(row["id"])
        return fallback_path, None

    def _prepare_managed_training_attempt(
        self,
        item: WorkItemRecord,
        launch_spec: Mapping[str, Any],
        attempt: Any,
    ) -> tuple[dict[str, Any], Any]:
        """Redirect a managed trainer into an attempt-local sibling directory."""

        if attempt is None:
            raise RuntimeError("managed training has no durable work attempt")
        canonical_output = Path(str(launch_spec.get("output_dir") or "")).expanduser()
        if not canonical_output.name:
            raise ValueError("managed training requires a canonical output directory")
        canonical_output = canonical_output.resolve()
        if canonical_output.exists():
            raise RuntimeError(
                f"managed run destination already exists and will not be reused: {canonical_output}"
            )
        staging_output = (
            canonical_output.parent / ".halo-forge-attempts" / item.id / str(attempt.id)
        )
        if staging_output.exists():
            raise RuntimeError(f"attempt staging directory already exists: {staging_output}")
        previous_output_value = launch_spec.get(
            "previous_segment_output_dir"
        ) or launch_spec.get("fork_checkpoint_snapshot_path")
        if previous_output_value:
            previous_output = Path(str(previous_output_value)).expanduser().resolve()
            if not previous_output.is_dir():
                raise RuntimeError(
                    "previous managed segment snapshot is missing: "
                    f"{previous_output}"
                )
            # Every attempt remains isolated.  Resumable state is copied from
            # the prior atomically published snapshot rather than mutated in
            # place, so a failed retry cannot corrupt the accepted boundary.
            shutil.copytree(previous_output, staging_output, symlinks=True)
        updated_attempt = self.scheduler.database.update_work_attempt_output_dir(
            str(attempt.id), str(staging_output)
        )
        if updated_attempt is None:
            raise LostClaimError("managed training attempt is no longer active")

        resolved = dict(launch_spec)
        command = list(resolved.get("command") or ())
        try:
            output_index = command.index("--output") + 1
        except ValueError as exc:
            raise ValueError("managed training command has no --output argument") from exc
        if output_index >= len(command):
            raise ValueError("managed training command has no --output value")
        command[output_index] = str(staging_output)
        checkpoint_pattern = str(
            resolved.get("resume_checkpoint_pattern") or ""
        ).strip()
        if checkpoint_pattern:
            checkpoint = staging_output / checkpoint_pattern
            if not checkpoint.is_dir():
                matches = sorted(staging_output.glob("checkpoint-*"))
                checkpoint = matches[-1] if matches else checkpoint
            if not checkpoint.is_dir():
                raise RuntimeError(
                    "previous managed GRPO segment has no resumable checkpoint "
                    f"matching {checkpoint_pattern}"
                )
            while "--resume" in command:
                index = command.index("--resume")
                del command[index : min(len(command), index + 2)]
            command.extend(["--resume", str(checkpoint)])
        resolved.update(
            command=command,
            execution_output_dir=str(staging_output),
            canonical_output_dir=str(canonical_output),
        )
        return resolved, updated_attempt

    def _resolve_launch(
        self,
        launch_spec: Mapping[str, Any],
    ) -> tuple[list[str], Optional[str], dict[str, str]]:
        command_value = launch_spec.get("command")
        if (
            not isinstance(command_value, (list, tuple))
            or not command_value
            or any(not isinstance(value, str) or not value for value in command_value)
        ):
            raise ValueError("launch_spec.command must be a non-empty argv list of strings")
        command = list(command_value)

        cwd_value = launch_spec.get("cwd")
        if cwd_value is not None and not isinstance(cwd_value, (str, os.PathLike)):
            raise ValueError("launch_spec.cwd must be a path string")
        cwd = str(Path(cwd_value).expanduser()) if cwd_value is not None else None

        env_value = launch_spec.get("env", launch_spec.get("environment", {}))
        if not isinstance(env_value, Mapping):
            raise ValueError("launch_spec.env must be an object")
        env = dict(os.environ)
        env.update({str(key): str(value) for key, value in env_value.items()})
        if launch_spec.get("operation") in {
            "managed_training",
            "managed_training_segment",
        }:
            from halo_forge.huggingface_access import inject_huggingface_token
            from halo_forge.runtime_determinism import RUN_ID_ENV

            canonical_run_id = str(
                (launch_spec.get("resolved_launch_config") or {}).get("run_id") or ""
            ).strip()
            if not canonical_run_id:
                raise ValueError("managed training launch has no canonical run ID")
            env[RUN_ID_ENV] = canonical_run_id
            direct_segment_id = str(
                launch_spec.get("direct_run_segment_id") or ""
            ).strip()
            if direct_segment_id:
                env["HALOFORGE_DIRECT_RUN_SEGMENT_ID"] = direct_segment_id
                env["HALOFORGE_DIRECT_RUN_SEGMENT_FINAL"] = (
                    "1" if bool(launch_spec.get("final_segment")) else "0"
                )
            inject_huggingface_token(env)
        database_path = str(getattr(self.scheduler.database, "path", "") or "")
        if database_path and database_path != ":memory:":
            env.setdefault("HALOFORGE_RUN_DB_PATH", database_path)
        return command, cwd, env

    def _publish_managed_training(
        self,
        item: WorkItemRecord,
        launch_spec: Mapping[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Verify, atomically publish, and catalog a dashboard-managed run."""

        if launch_spec.get("operation") not in {
            "managed_training",
            "managed_training_segment",
        }:
            return
        config = dict(launch_spec.get("resolved_launch_config") or {})
        run_id = str(item.canonical_run_id or config.get("run_id") or "").strip()
        if not run_id:
            raise RuntimeError("managed training completed without a canonical run ID")
        staging_output = Path(str(launch_spec.get("execution_output_dir") or "")).expanduser()
        canonical_output = Path(
            str(launch_spec.get("canonical_output_dir") or launch_spec.get("output_dir") or "")
        ).expanduser()
        if not staging_output.is_dir():
            raise RuntimeError(
                f"training exited successfully but its attempt output is missing: {staging_output}"
            )
        if canonical_output.exists():
            raise RuntimeError(f"canonical run destination already exists: {canonical_output}")
        segmented = launch_spec.get("operation") == "managed_training_segment"
        final_segment = not segmented or bool(launch_spec.get("final_segment"))
        publication_output = canonical_output
        if segmented and not final_segment:
            publication_output = Path(
                str(launch_spec.get("segment_output_dir") or "")
            ).expanduser()
            if not publication_output.name:
                raise RuntimeError("managed training segment has no snapshot destination")
            if publication_output.exists():
                raise RuntimeError(
                    "managed segment snapshot already exists and will not be reused: "
                    f"{publication_output}"
                )

        summary_path = staging_output / "training_summary.json"
        if not summary_path.is_file():
            raise RuntimeError("training exited successfully without training_summary.json")
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"training summary is unreadable: {exc}") from exc
        summary_run_id = str(summary.get("run_id") or "").strip()
        if summary_run_id and summary_run_id != run_id:
            raise RuntimeError(
                "trainer reported a different run ID: " f"{summary_run_id!r} != {run_id!r}"
            )
        summary = self._rewrite_managed_paths(
            summary,
            old_root=str(staging_output),
            new_root=str(publication_output),
        )
        summary["run_id"] = run_id
        summary["output_dir"] = str(publication_output)
        self._write_json_atomic(summary_path, summary)
        self._rewrite_managed_json_files(
            staging_output,
            old_root=str(staging_output),
            new_root=str(publication_output),
        )

        candidates = self._managed_training_artifact_candidates(staging_output)
        if not candidates:
            raise RuntimeError("training completed without a checkpoint, adapter, or final model")
        from halo_forge.artifact_lab import hash_path

        verified_candidates = []
        for candidate in candidates:
            digest = hash_path(candidate)
            verified_candidates.append(
                (
                    candidate.relative_to(staging_output),
                    digest.content_hash,
                    digest.size_bytes,
                )
            )

        if segmented and not final_segment:
            publication_output.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging_output, publication_output)
            segment_id = str(launch_spec.get("direct_run_segment_id") or "")
            if segment_id:
                self.scheduler.database.update_direct_run_segment(
                    segment_id,
                    status="audit_pending",
                    decision="complete",
                    decision_reason="checkpoint and exact-output trace published",
                )
            with self.scheduler.database._lock:
                self.scheduler.database._conn.execute(
                    "UPDATE runs SET status='audit_pending' WHERE run_id=?",
                    (run_id,),
                )
                self.scheduler.database._conn.commit()
            result.update(
                direct_run_segment_id=segment_id,
                segment_output_dir=str(publication_output),
                segment_checkpoint_count=len(verified_candidates),
                awaiting_reward_audit=True,
            )
            self._advance_reused_direct_audit(item, segment_id, result)
            return

        # Replay and relaunch evidence describe the final canonical location,
        # even though the bytes are still isolated in the attempt directory.
        from ui.services.launch_context import persist_launch_context
        from ui.services.training_service import TrainingService

        mode = str(config.get("mode") or summary.get("modality") or "").strip().lower()
        config.update(run_id=run_id, output_dir=str(canonical_output))
        dataset_value = str(
            config.get("prompts") if mode == "raft" else config.get("dataset") or ""
        )
        TrainingService._persist_replay_manifest(
            output_dir=str(staging_output),
            run_id=run_id,
            modality=mode,
            model=str(config.get("model") or ""),
            seed=int(config.get("seed") or 42),
            launch_args=config,
            dataset_value=dataset_value,
            dataset_version_id=(
                str(config["dataset_version_id"]) if config.get("dataset_version_id") else None
            ),
            dataset_split=str(config.get("dataset_split") or "train"),
            dataset_version_metadata=dict(launch_spec.get("dataset_version_metadata") or {}),
            dataset_bindings=list(config.get("dataset_bindings") or []),
            training_artifact_metadata=dict(
                config.get("training_artifact_metadata") or config.get("training_artifact") or {}
            ),
            parent_run_id=(str(config["parent_run_id"]) if config.get("parent_run_id") else None),
        )
        persist_launch_context(
            output_dir=staging_output,
            job_type=mode,
            service="training",
            source_ui_page=str(launch_spec.get("source_ui_page") or "/public/train"),
            command=list(launch_spec.get("canonical_command") or ()),
            args=config,
            relaunch_capabilities={
                "can_relaunch": True,
                "can_clone": True,
                "can_resume_latest": mode in {"raft", "vlm", "audio", "reasoning", "agentic"},
            },
            metadata={
                "managed": True,
                "canonical_run_id": run_id,
                "work_item_id": item.id,
            },
        )
        log_path = self._resolve_log_path(item)
        if log_path.is_file():
            shutil.copy2(log_path, staging_output / f"{run_id}_training.log")
        self._write_json_atomic(
            staging_output / ".halo_forge_managed_run.json",
            {
                "run_id": run_id,
                "work_item_id": item.id,
                "attempt_output_dir": str(staging_output),
                "canonical_output_dir": str(canonical_output),
            },
        )

        canonical_output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging_output, canonical_output)

        from halo_forge.artifact_studio import ArtifactStudioService

        studio = ArtifactStudioService(
            self.scheduler.database,
            artifact_root=launch_spec.get("artifact_root"),
        )
        occurrence_ids: list[str] = []
        legacy_ids: list[str] = []
        previous_occurrence_id: Optional[str] = None
        model_id = str(config.get("model") or summary.get("model_name") or "unknown")
        backend = str(config.get("accelerator") or summary.get("backend") or "local")
        for ordinal, (relative, content_hash, size_bytes) in enumerate(verified_candidates):
            candidate = canonical_output / relative
            has_adapter_config = (candidate / "adapter_config.json").is_file()
            if has_adapter_config:
                artifact_kind = "adapter"
                filesystem_kind = "adapter"
                artifact_format = "peft_adapter"
            elif relative.name == "final_model" or any(
                part.endswith("_final") for part in relative.parts
            ):
                artifact_kind = "final_model"
                filesystem_kind = "final"
                artifact_format = "huggingface"
            else:
                artifact_kind = "checkpoint"
                filesystem_kind = "checkpoint"
                artifact_format = "huggingface"
            identity = hashlib.sha256(
                f"{run_id}:{relative.as_posix()}:{content_hash}".encode("utf-8")
            ).hexdigest()[:24]
            legacy_id = f"model-artifact-{identity}"
            legacy = self.scheduler.database.get_model_artifact(legacy_id)
            if legacy is None:
                legacy = self.scheduler.database.create_model_artifact(
                    artifact_id=legacy_id,
                    artifact_hash=content_hash,
                    artifact_kind=artifact_kind,
                    run_id=run_id,
                    model_id=model_id,
                    backend=backend,
                    format=artifact_format,
                    path=str(candidate),
                    size_bytes=size_bytes,
                    verification_status="verified",
                    metadata={
                        "work_item_id": item.id,
                        "managed_run": True,
                        "relative_path": relative.as_posix(),
                    },
                )
            occurrence_id = f"artifact-{identity}"
            artifact_view = studio.import_artifact(
                candidate,
                artifact_kind=filesystem_kind,
                artifact_format=artifact_format,
                model_id=model_id,
                backend=backend,
                managed=False,
                occurrence_id=occurrence_id,
                run_id=run_id,
                tokenizer_revision=(
                    str(config["tokenizer_revision"]) if config.get("tokenizer_revision") else None
                ),
                metadata={
                    "work_item_id": item.id,
                    "managed_run": True,
                    "legacy_model_artifact_id": legacy.id,
                    "relative_path": relative.as_posix(),
                },
            )
            occurrence = artifact_view["occurrence"]
            if str(artifact_view["blob"]["content_hash"]) != content_hash:
                raise RuntimeError(f"artifact changed during publication: {relative.as_posix()}")
            occurrence_ids.append(str(occurrence["id"]))
            legacy_ids.append(legacy.id)
            if mode in {"classify", "embed", "rerank"}:
                task_contract = dict(summary.get("task_contract") or {})
                config_path = candidate / "config.json"
                model_head_hash = (
                    hash_path(config_path).content_hash
                    if config_path.is_file()
                    else content_hash
                )
                processor_candidates = [
                    candidate / name
                    for name in (
                        "tokenizer_config.json",
                        "preprocessor_config.json",
                        "processor_config.json",
                        "label_map.json",
                        "task_config.json",
                    )
                    if (candidate / name).is_file()
                ]
                processor_hash = hashlib.sha256(
                    ":".join(
                        hash_path(path).content_hash
                        for path in processor_candidates
                    ).encode("utf-8")
                ).hexdigest()
                loss_adapter = str(
                    task_contract.get("loss_adapter")
                    or {
                        "classify": "cross_entropy_or_bce",
                        "embed": "multiple_negative_ranking",
                        "rerank": "binary_scalar_cross_encoder",
                    }[mode]
                )
                retrieval_ref = str(
                    config.get("retrieval_corpus_id") or ""
                ).strip()
                with self.scheduler.database._lock:
                    self.scheduler.database._conn.execute(
                        """INSERT OR REPLACE INTO specialized_artifact_metadata
                           (artifact_occurrence_id,task_kind,modality,
                            label_schema_revision_id,model_head_hash,
                            processor_hash,loss_adapter,loss_adapter_version,
                            retrieval_corpus_hash,metadata_json,created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            str(occurrence["id"]),
                            mode,
                            str(task_contract.get("task_modality") or "text"),
                            config.get("label_schema_revision_id"),
                            model_head_hash,
                            processor_hash,
                            loss_adapter,
                            "1",
                            (
                                hashlib.sha256(retrieval_ref.encode("utf-8")).hexdigest()
                                if retrieval_ref
                                else None
                            ),
                            json.dumps(
                                {
                                    "task_contract": task_contract,
                                    "proof_run": bool(config.get("proof_run")),
                                },
                                sort_keys=True,
                            ),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    self.scheduler.database._conn.commit()
            if previous_occurrence_id:
                studio.catalog.add_edge(
                    child_occurrence_id=str(occurrence["id"]),
                    parent_occurrence_id=previous_occurrence_id,
                    relation="continued_from",
                    ordinal=ordinal - 1,
                )
            previous_occurrence_id = str(occurrence["id"])

        result.update(
            canonical_run_id=run_id,
            output_dir=str(canonical_output),
            training_summary=summary,
            artifact_occurrence_ids=occurrence_ids,
            model_artifact_ids=legacy_ids,
            published=True,
        )
        if segmented:
            segment_id = str(launch_spec.get("direct_run_segment_id") or "")
            if segment_id:
                self.scheduler.database.update_direct_run_segment(
                    segment_id,
                    status="audit_pending",
                    decision="complete",
                    decision_reason="final checkpoint and exact-output trace published",
                    checkpoint_occurrence_id=(
                        occurrence_ids[-1] if occurrence_ids else None
                    ),
                )
            with self.scheduler.database._lock:
                self.scheduler.database._conn.execute(
                    "UPDATE runs SET status='audit_pending' WHERE run_id=?",
                    (run_id,),
                )
                self.scheduler.database._conn.commit()
            result.update(
                direct_run_segment_id=segment_id,
                awaiting_reward_audit=True,
            )
            self._advance_reused_direct_audit(item, segment_id, result)

    def _advance_reused_direct_audit(
        self,
        item: WorkItemRecord,
        segment_id: str,
        result: dict[str, Any],
    ) -> None:
        """Reapply an exact reused audit without opening a second process."""

        if not segment_id:
            return
        row = self.scheduler.database._conn.execute(
            "SELECT id FROM reward_integrity_audits "
            "WHERE direct_run_segment_id=? AND status='completed' "
            "ORDER BY completed_at DESC,created_at DESC,id DESC LIMIT 1",
            (segment_id,),
        ).fetchone()
        if row is None:
            return
        from halo_forge.reward_integrity import RewardIntegrityService

        service = RewardIntegrityService(self.scheduler.database)
        audit = service.get_audit(str(row["id"]))
        decisions = service.store.list_decisions(audit.id, limit=1000)
        decision = decisions.items[-1] if decisions.items else None
        if decision is None:
            return
        service._apply_gate_state(audit, decision)
        result.update(
            reused_reward_audit_id=audit.id,
            reused_reward_audit_decision=decision.to_dict(),
            awaiting_reward_audit=False,
        )
        if decision.action == "continue":
            from halo_forge.reward_integrity.direct_segments import (
                enqueue_next_direct_segment,
            )

            next_work = enqueue_next_direct_segment(
                self.scheduler.database,
                self.scheduler,
                current_segment_id=segment_id,
                dependency_work_item_id=item.id,
            )
            if next_work is not None:
                result["next_work_item_id"] = next_work.id

    @staticmethod
    def _managed_training_artifact_candidates(output_dir: Path) -> list[Path]:
        values: list[Path] = []
        direct = [output_dir / "final_model", output_dir / "best_checkpoint"]
        direct.extend(sorted(output_dir.glob("checkpoint-*")))
        direct.extend(sorted(output_dir.glob("cycle_*_final")))
        direct.extend(sorted(output_dir.glob("cycle_*")))
        for candidate in direct:
            if not candidate.is_dir():
                continue
            model_child = candidate / "model"
            resolved = model_child if model_child.is_dir() else candidate
            if resolved not in values:
                values.append(resolved)
        if not values and any(
            (output_dir / name).is_file()
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "model.safetensors",
                "pytorch_model.bin",
            )
        ):
            values.append(output_dir)
        # Preserve training order while ensuring the final model is the last
        # lineage child even when its name sorts before checkpoints.
        values.sort(
            key=lambda value: (
                value.name == "final_model" or any(part.endswith("_final") for part in value.parts),
                value.as_posix(),
            )
        )
        return values

    @staticmethod
    def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @classmethod
    def _rewrite_managed_paths(
        cls,
        value: Any,
        *,
        old_root: str,
        new_root: str,
    ) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): cls._rewrite_managed_paths(child, old_root=old_root, new_root=new_root)
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [
                cls._rewrite_managed_paths(child, old_root=old_root, new_root=new_root)
                for child in value
            ]
        if isinstance(value, str) and value.startswith(old_root):
            return new_root + value[len(old_root) :]
        return value

    @classmethod
    def _rewrite_managed_json_files(
        cls,
        root: Path,
        *,
        old_root: str,
        new_root: str,
    ) -> None:
        """Keep checkpoint/history sidecar paths valid after atomic publish."""

        for path in root.rglob("*.json"):
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rewritten = cls._rewrite_managed_paths(
                payload, old_root=old_root, new_root=new_root
            )
            if rewritten != payload:
                temporary = path.with_name(f".{path.name}.tmp")
                temporary.write_text(
                    json.dumps(rewritten, indent=2, sort_keys=True, default=str),
                    encoding="utf-8",
                )
                os.replace(temporary, path)

    def _publish_training_artifacts(
        self,
        item: WorkItemRecord,
        launch_spec: Mapping[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Index the checkpoint/final model emitted by a managed trial command."""
        if launch_spec.get("operation") != "train_trial_segment":
            return
        output_dir = Path(str(launch_spec.get("output_dir") or "")).expanduser()
        segment = launch_spec.get("segment") or {}
        unit = str(segment.get("unit") or "full_trial")
        end_value = int(segment.get("end") or 0)
        capability = launch_spec.get("capability") or {}
        checkpoint_pattern = str(capability.get("checkpoint_pattern") or "")

        candidate: Optional[Path] = None
        artifact_kind = "checkpoint" if unit != "full_trial" else "final_model"
        if unit == "full_trial" and (output_dir / "final_model").is_dir():
            candidate = output_dir / "final_model"
        if candidate is None and checkpoint_pattern and output_dir.is_dir():
            matches = [path for path in output_dir.glob(checkpoint_pattern) if path.is_dir()]
            if matches:
                candidate = max(matches, key=self._checkpoint_sort_key)
        if candidate is None and output_dir.is_dir():
            fallbacks = [
                path
                for pattern in ("checkpoint-*", "cycle_*_final", "cycle_*")
                for path in output_dir.glob(pattern)
                if path.is_dir()
            ]
            if fallbacks:
                candidate = max(fallbacks, key=self._checkpoint_sort_key)
        if candidate is None:
            raise RuntimeError(
                f"training completed but no model artifact was published under {output_dir}"
            )
        if unit != "full_trial":
            candidate_numbers = [int(value) for value in re.findall(r"\d+", candidate.name)]
            trainer_mode = str(launch_spec.get("trainer_mode") or "")
            expected_index = (
                end_value - 1
                if unit == "cycle" and trainer_mode in {"vlm", "audio", "reasoning", "agentic"}
                else end_value
            )
            if not candidate_numbers or candidate_numbers[-1] != expected_index:
                raise RuntimeError(
                    "training completed without publishing the exact checkpoint boundary "
                    f"{unit}={end_value}"
                )
            if unit == "step":
                trainer_state = candidate / "trainer_state.json"
                if not trainer_state.is_file():
                    raise RuntimeError(
                        "step checkpoint is missing trainer_state.json; exact resume state "
                        "cannot be verified"
                    )
                try:
                    global_step = int(
                        json.loads(trainer_state.read_text(encoding="utf-8"))["global_step"]
                    )
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        "step checkpoint trainer_state.json is not verifiable"
                    ) from exc
                if global_step != end_value:
                    raise RuntimeError(
                        "step checkpoint state does not match the requested boundary: "
                        f"{global_step} != {end_value}"
                    )

        checkpoint_root = candidate
        if unit == "cycle" and (candidate / "model").is_dir():
            candidate = candidate / "model"
        artifact_hash, size_bytes = self._hash_path(candidate)
        database = self.scheduler.database
        catalog = LabV4Catalog(database)
        config = launch_spec.get("resolved_launch_config") or {}
        backend = str(
            config.get("accelerator")
            or config.get("backend")
            or capability.get("backend_family")
            or "unknown"
        )
        model_id = str(
            config.get("model") or config.get("model_name") or config.get("base_model") or "unknown"
        )
        artifact_format = (
            "peft_adapter" if (candidate / "adapter_config.json").is_file() else "huggingface"
        )
        previous_artifact_id = None
        segment_id = str(launch_spec.get("trial_segment_id") or "")
        trial_segment = database.get_trial_segment(segment_id) if segment_id else None
        if trial_segment is not None and trial_segment.ordinal > 0:
            previous = next(
                (
                    value
                    for value in database.list_trial_segments(trial_segment.trial_run_id)
                    if value.ordinal == trial_segment.ordinal - 1
                ),
                None,
            )
            previous_artifact_id = previous.checkpoint_artifact_id if previous else None

        # A checkpoint that participates in a gate is decision evidence. Copy
        # it into the content-addressed library before exposing the segment to
        # evaluators so trainer checkpoint rotation cannot invalidate resume,
        # comparison, or evidence publication later.
        from halo_forge.artifact_studio import ArtifactStudioService

        configured_root = str(launch_spec.get("artifact_root") or "").strip()
        if configured_root:
            artifact_root = Path(configured_root).expanduser()
        elif database.path and database.path != ":memory:":
            artifact_root = Path(database.path).expanduser().parent / "artifacts"
        else:
            artifact_root = output_dir.parent / ".halo-forge-artifacts"
        studio = ArtifactStudioService(database, artifact_root=artifact_root)
        occurrence_identity = hashlib.sha256(
            f"{segment_id}:{artifact_hash}".encode("utf-8")
        ).hexdigest()[:24]
        occurrence_id = f"artifact-{occurrence_identity}"
        filesystem_kind = "checkpoint" if artifact_kind == "checkpoint" else "final"
        registration = studio.store.import_artifact(
            candidate,
            artifact_kind=filesystem_kind,
            artifact_format=artifact_format,
            managed=True,
            occurrence_id=occurrence_id,
            run_id=str(item.canonical_run_id or config.get("run_id") or "unknown"),
            run_group_id=launch_spec.get("run_group_id"),
            trial_id=launch_spec.get("trial_id"),
            segment_id=segment_id or None,
            step=end_value if unit == "step" else None,
            cycle=end_value if unit == "cycle" else None,
            metadata={
                "work_item_id": item.id,
                "checkpoint_root": str(checkpoint_root),
                "segment_unit": unit,
                "segment_end": end_value,
            },
        )
        if registration.blob.content_hash != artifact_hash:
            raise RuntimeError("managed checkpoint identity changed during atomic publication")
        managed_path = str(registration.location.path)

        artifact = database.find_model_artifact(artifact_hash)
        if artifact is None:
            artifact = database.create_model_artifact(
                artifact_hash=artifact_hash,
                artifact_kind=artifact_kind,
                run_id=str(item.canonical_run_id or config.get("run_id") or "unknown"),
                run_group_id=launch_spec.get("run_group_id"),
                trial_id=launch_spec.get("trial_id"),
                trial_segment_id=segment_id or None,
                parent_artifact_id=previous_artifact_id,
                model_id=model_id,
                tokenizer_revision=config.get("tokenizer_revision"),
                chat_template_hash=config.get("chat_template_hash"),
                backend=backend,
                format=artifact_format,
                path=managed_path,
                size_bytes=size_bytes,
                step=end_value if unit == "step" else None,
                cycle=end_value if unit == "cycle" else None,
                verification_status="verified",
                metadata={
                    "work_item_id": item.id,
                    "checkpoint_root": str(checkpoint_root),
                    "segment_unit": unit,
                    "segment_end": end_value,
                    "managed": True,
                },
            )
        blob = catalog.upsert_blob(
            blob_id=registration.blob.id,
            content_hash=artifact_hash,
            artifact_type=artifact_kind,
            format=artifact_format,
            dtype=registration.blob.dtype,
            quantization=registration.blob.quantization,
            size_bytes=size_bytes,
            integrity_state="verified",
            manifest=registration.blob.to_dict(),
        )
        catalog.add_location(
            location_id=registration.location.id,
            blob_id=blob.id,
            path=managed_path,
            storage_mode="managed",
            state="available",
            size_bytes=size_bytes,
            metadata={
                "run_id": item.canonical_run_id,
                "work_item_id": item.id,
                "source_path": str(candidate),
                "filesystem_location": registration.location.to_dict(),
            },
        )
        catalog.add_location(
            blob_id=blob.id,
            path=str(candidate),
            storage_mode="referenced",
            state="available",
            size_bytes=size_bytes,
            metadata={"run_id": item.canonical_run_id, "work_item_id": item.id},
        )
        occurrence = catalog.get_occurrence(occurrence_id)
        if occurrence is None:
            occurrence = catalog.create_occurrence(
                occurrence_id=occurrence_id,
                blob_id=blob.id,
                artifact_kind=artifact_kind,
                legacy_model_artifact_id=artifact.id,
                run_id=str(item.canonical_run_id or config.get("run_id") or "unknown"),
                run_group_id=launch_spec.get("run_group_id"),
                trial_id=launch_spec.get("trial_id"),
                trial_segment_id=segment_id or None,
                model_id=model_id,
                tokenizer_revision=config.get("tokenizer_revision"),
                chat_template_hash=config.get("chat_template_hash"),
                backend=backend,
                metadata={
                    "work_item_id": item.id,
                    "checkpoint_root": str(checkpoint_root),
                    "segment_unit": unit,
                    "segment_end": end_value,
                    "managed_location_id": registration.location.id,
                },
            )
        elif occurrence.blob_id != blob.id or occurrence.trial_segment_id != (segment_id or None):
            raise RuntimeError(f"artifact occurrence identity collision for {occurrence_id}")
        if previous_artifact_id:
            parent_row = database._conn.execute(
                "SELECT id FROM artifact_occurrences "
                "WHERE trial_segment_id = ? ORDER BY created_at DESC LIMIT 1",
                (previous.id,),
            ).fetchone()
            if parent_row is not None:
                catalog.add_edge(
                    child_occurrence_id=occurrence.id,
                    parent_occurrence_id=str(parent_row["id"]),
                    relation="continued_from",
                )
        if segment_id:
            database.update_trial_segment(
                segment_id,
                checkpoint_artifact_id=artifact.id,
            )
        summary_path = output_dir / "training_summary.json"
        if summary_path.is_file():
            try:
                result["training_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                result["training_summary_error"] = str(exc)
        result.update(
            model_artifact_id=artifact.id,
            model_artifact_hash=artifact.artifact_hash,
            model_artifact_path=managed_path,
            artifact_blob_id=blob.id,
            artifact_occurrence_id=occurrence.id,
        )

    def _capture_evaluation_result(
        self,
        item: WorkItemRecord,
        launch_spec: Mapping[str, Any],
        result: dict[str, Any],
    ) -> None:
        if launch_spec.get("operation") != "evaluate_trial_segment":
            return
        suite_revision_id = str(launch_spec.get("suite_revision_id") or "")
        segment_id = str(launch_spec.get("trial_segment_id") or "")
        evaluations = self.scheduler.database.list_evaluations(
            suite_revision_id=suite_revision_id or None,
            status="completed",
            limit=100,
        )
        evaluation = next(
            (
                value
                for value in evaluations
                if str(value.request.get("trial_segment_id") or "") == segment_id
            ),
            None,
        )
        if evaluation is None:
            expected_ref = str(
                launch_spec.get("resolved_checkpoint_path")
                or (launch_spec.get("subject") or {}).get("ref")
                or item.canonical_run_id
                or ""
            )
            evaluation = next(
                (value for value in evaluations if value.subject_ref == expected_ref),
                None,
            )
        if evaluation is None:
            raise RuntimeError("evaluation command completed without a matching persistent result")
        revision = self.scheduler.database.get_benchmark_suite_revision(
            evaluation.suite_revision_id
        )
        if revision is None:
            raise RuntimeError("completed evaluation references a missing suite revision")
        metrics: dict[str, float] = {}
        for metric in self.scheduler.database.list_evaluation_metrics(evaluation.id):
            if not metric.suite_item_id:
                metrics[metric.name] = float(metric.value)
        if revision.primary_metric not in metrics:
            raise RuntimeError(
                "completed evaluation did not publish its primary metric "
                f"{revision.primary_metric!r}"
            )
        sample_count = self.scheduler.database.count_evaluation_samples(evaluation.id)
        if sample_count <= 0:
            raise RuntimeError("completed evaluation did not publish sample evidence")
        if not evaluation.artifact_path or not Path(evaluation.artifact_path).exists():
            raise RuntimeError("completed evaluation evidence bundle is missing")
        result["evaluation_id"] = evaluation.id
        result["metrics"] = metrics
        result["evaluation_evidence_path"] = evaluation.artifact_path
        result["evaluation_sample_count"] = sample_count

    @staticmethod
    def _checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
        numbers = [int(value) for value in re.findall(r"\d+", path.name)]
        try:
            modified = int(path.stat().st_mtime_ns)
        except OSError:
            modified = 0
        return (numbers[-1] if numbers else -1, modified, path.name)

    @staticmethod
    def _hash_path(path: Path) -> tuple[str, int]:
        from halo_forge.artifact_lab import hash_path

        identity = hash_path(path)
        return identity.content_hash, identity.size_bytes

    @staticmethod
    def _capture_result_file(
        launch_spec: Mapping[str, Any],
        result: dict[str, Any],
        *,
        cwd: Optional[str] = None,
    ) -> None:
        configured = launch_spec.get("result_path")
        if not configured:
            return
        path = Path(str(configured)).expanduser()
        if not path.is_absolute() and cwd is not None:
            path = Path(cwd) / path
        try:
            with path.open("r", encoding="utf-8") as handle:
                result["command_result"] = json.load(handle)
            result["result_path"] = str(path)
        except Exception as exc:
            result["result_path"] = str(path)
            result["result_capture_error"] = str(exc)

    @staticmethod
    def _append_log(path: Path, message: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("ab") as handle:
                handle.write(message.encode("utf-8", errors="replace"))
        except OSError:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Halo Forge workstation queue worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run at most one ready work item and exit",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.25,
        help="Seconds between queue/process checks (default: 0.25)",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=None,
        help="Seconds between durable lease heartbeats",
    )
    parser.add_argument(
        "--terminate-timeout",
        type=float,
        default=10.0,
        help="Seconds after TERM before KILL (default: 10)",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Override the run database path",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    scheduler = WorkstationScheduler(get_database(args.database))
    worker = WorkstationWorker(
        scheduler,
        poll_interval=args.poll_interval,
        heartbeat_interval=args.heartbeat_interval,
        terminate_timeout=args.terminate_timeout,
    )

    received_signal: list[int] = []

    def request_stop(signum: int, _frame: Any) -> None:
        received_signal.append(signum)
        worker.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        status = worker.watch(once=args.once)
        return 128 + received_signal[0] if received_signal else status
    except KeyboardInterrupt:
        worker.stop()
        return 130


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``
    raise SystemExit(main())
