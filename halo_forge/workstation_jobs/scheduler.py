"""Persistent priority/FIFO scheduling for one local workstation."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from halo_forge.run_db import ResourceLeaseRecord, RunDatabase, WorkItemRecord
from halo_forge.workstation_jobs.resources import (
    WorkstationCapacity,
    evaluate_capacity_preflight,
    sample_workstation_capacity,
)


def process_start_time(pid: int) -> Optional[float]:
    """Return a stable process start identity without requiring psutil.

    PID alone is unsafe after restart because operating systems reuse it. A
    missing probe is treated conservatively by recovery: the process cannot be
    adopted and its item is interrupted for an explicit retry.
    """

    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        pass

    # Linux exposes the process start tick in field 22 of /proc/<pid>/stat.
    # It is host-local rather than wall-clock time, which is sufficient because
    # this value is only compared with a later probe on the same workstation.
    try:
        with open(f"/proc/{int(pid)}/stat", "r", encoding="utf-8") as handle:
            fields_after_name = handle.read().rsplit(")", 1)[1].split()
        return float(fields_after_name[19])
    except Exception:
        pass

    # macOS and other BSD-like hosts have no /proc by default. ``ps lstart`` is
    # available in the base system and remains stable for the process lifetime.
    try:
        completed = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(int(pid))],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        value = " ".join(completed.stdout.split())
        if completed.returncode == 0 and value:
            return datetime.strptime(value, "%a %b %d %H:%M:%S %Y").timestamp()
    except Exception:
        pass
    return None


def process_matches(pid: Optional[int], expected_start_time: Optional[float]) -> bool:
    if pid is None or expected_start_time is None:
        return False
    actual = process_start_time(pid)
    return actual is not None and abs(actual - float(expected_start_time)) < 0.01


@dataclass(frozen=True)
class RecoveryResult:
    adopted: tuple[WorkItemRecord, ...]
    interrupted: tuple[WorkItemRecord, ...]


class WorkstationScheduler:
    """Thin orchestration layer over :class:`RunDatabase` queue primitives.

    The service does not start subprocesses itself. This separation lets every
    existing trainer keep its launch code while sharing one durable claim and
    accelerator lease protocol.
    """

    def __init__(
        self,
        database: RunDatabase,
        *,
        worker_id: Optional[str] = None,
        lease_ttl_seconds: int = 30,
        process_probe: Callable[[Optional[int], Optional[float]], bool] = process_matches,
        capacity_probe: Optional[Callable[[Path], WorkstationCapacity]] = None,
        accelerator_probe: Optional[Callable[[str], Any]] = None,
        idle_sample_count: int = 3,
        idle_sample_interval_seconds: float = 2.0,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.database = database
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.lease_ttl_seconds = max(1, int(lease_ttl_seconds))
        self.process_probe = process_probe
        self.capacity_probe = capacity_probe or (
            lambda path: sample_workstation_capacity(
                path, pid=os.getpid(), include_accelerator=False
            )
        )
        self.accelerator_probe = accelerator_probe
        self.idle_sample_count = max(1, int(idle_sample_count))
        self.idle_sample_interval_seconds = max(0.0, float(idle_sample_interval_seconds))
        self.sleeper = sleeper

    @staticmethod
    def _managed_accelerator_family(item: WorkItemRecord) -> Optional[str]:
        requirements = dict(item.resource_requirements)
        value = str(requirements.get("accelerator_family") or "").strip().lower()
        if value in {"rocm", "cuda"}:
            return value
        if item.resource_class != "accelerator":
            return None
        try:
            from halo_forge.backend import get_backend

            backend = str(get_backend().name).lower()
        except Exception:
            backend = str(os.environ.get("HALOFORGE_BACKEND") or "").lower()
        if backend.startswith("rocm"):
            return "rocm"
        if backend == "cuda":
            return "cuda"
        return None

    def _stable_external_idle(self, family: str) -> tuple[bool, tuple[Any, ...]]:
        from halo_forge.managed_runtime.occupancy import probe_accelerator

        probe = self.accelerator_probe or probe_accelerator
        evidence: list[Any] = []
        for index in range(self.idle_sample_count):
            current = probe(family)
            evidence.append(current)
            if not current.idle:
                return False, tuple(evidence)
            if index + 1 < self.idle_sample_count:
                self.sleeper(self.idle_sample_interval_seconds)
        return True, tuple(evidence)

    def confirm_pre_spawn(self, item: WorkItemRecord) -> tuple[bool, Optional[Any]]:
        """Close the sampling race immediately before a child process starts."""

        family = self._managed_accelerator_family(item)
        if not family:
            return True, None
        from halo_forge.managed_runtime.occupancy import probe_accelerator
        from halo_forge.managed_runtime.service import ManagedRuntimeService

        probe = self.accelerator_probe or probe_accelerator
        current = probe(family)
        requirements = dict(item.resource_requirements)
        runtime = ManagedRuntimeService(
            self.database,
            root=(requirements.get("runtime_root") or Path.home() / ".halo-forge" / "runtimes"),
            scheduler=self,
            occupancy_probe=probe,
        )
        runtime.record_preflight_decision(
            accelerator_family=family,
            decision=("idle" if current.idle else ("unknown" if current.state == "unknown" else "contention")),
            evidence={"pre_spawn": current.to_dict()},
            work_item_id=item.id,
            runtime_revision_id=str(requirements.get("runtime_profile_revision_id") or "") or None,
        )
        if current.idle:
            return True, current
        if item.claim_token:
            self.database.defer_claimed_work_item_for_accelerator(
                item.id,
                claim_token=item.claim_token,
                reason=current.reason or "accelerator became unavailable before launch",
                details={"availability": current.to_dict(), "phase": "pre_spawn"},
                not_before=(datetime.now(timezone.utc) + timedelta(seconds=15)).isoformat(),
            )
        return False, current

    def late_contention(self, item: WorkItemRecord) -> Optional[Any]:
        """Return new external occupancy while a Halo Forge child is active."""

        family = self._managed_accelerator_family(item)
        if not family:
            return None
        from halo_forge.managed_runtime.occupancy import probe_accelerator

        current = (self.accelerator_probe or probe_accelerator)(family)
        return current if current.state in {"busy", "unknown"} else None

    def enqueue(
        self,
        *,
        kind: str,
        launch_spec: Optional[Mapping[str, Any]] = None,
        resource_class: str = "accelerator",
        resource_requirements: Optional[Mapping[str, Any]] = None,
        priority: int = 0,
        domain_kind: Optional[str] = None,
        domain_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        canonical_run_id: Optional[str] = None,
        log_path: Optional[str] = None,
        dependencies: Sequence[str] = (),
        max_retries: int = 0,
        not_before: Optional[str] = None,
        work_item_id: Optional[str] = None,
    ) -> WorkItemRecord:
        return self.database.create_work_item(
            kind=kind,
            launch_spec=launch_spec,
            resource_class=resource_class,
            resource_requirements=resource_requirements,
            priority=priority,
            domain_kind=domain_kind,
            domain_id=domain_id,
            run_group_id=run_group_id,
            canonical_run_id=canonical_run_id,
            log_path=log_path,
            dependencies=dependencies,
            max_retries=max_retries,
            not_before=not_before,
            work_item_id=work_item_id,
        )

    def claim(
        self,
        *,
        work_item_id: Optional[str] = None,
        child_pid: Optional[int] = None,
        child_pid_started_at: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> Optional[WorkItemRecord]:
        pid = int(child_pid) if child_pid is not None else os.getpid()
        started = (
            float(child_pid_started_at)
            if child_pid_started_at is not None
            else process_start_time(pid)
        )
        item = self.database.claim_next_work_item(
            worker_id=self.worker_id,
            worker_pid=pid,
            worker_pid_started_at=started,
            lease_ttl_seconds=self.lease_ttl_seconds,
            work_item_id=work_item_id,
            now=now,
        )
        if item is None:
            return item
        requirements = dict(item.resource_requirements)
        requires_capacity_preflight = (
            item.resource_class == "accelerator"
            or str(requirements.get("lease_type") or "").lower() == "serving"
            or bool(requirements.get("capacity_preflight"))
            or "projected_disk_bytes" in requirements
            or "projected_output_bytes" in requirements
            or "projected_ram_bytes" in requirements
        )
        if not requires_capacity_preflight:
            return item
        output_value = (
            requirements.get("output_path")
            or item.launch_spec.get("output_dir")
            or item.launch_spec.get("output_root")
        )
        if output_value:
            capacity_path = Path(str(output_value)).expanduser()
        elif self.database.path and self.database.path != ":memory:":
            capacity_path = Path(self.database.path).expanduser().parent
        else:
            capacity_path = Path.home() / ".halo-forge"
        capacity = self.capacity_probe(capacity_path)
        preflight = evaluate_capacity_preflight(
            capacity,
            projected_disk_bytes=int(
                requirements.get(
                    "projected_disk_bytes",
                    requirements.get("projected_output_bytes", 0),
                )
                or 0
            ),
            projected_ram_bytes=int(
                requirements.get("projected_ram_bytes", requirements.get("ram_bytes", 0)) or 0
            ),
            override_reason=(
                str(
                    requirements.get("capacity_override_reason")
                    or requirements.get("override_reason")
                    or ""
                ).strip()
                or None
            ),
        )
        if not preflight.allowed and item.claim_token:
            self.database.block_claimed_work_item(
                item.id,
                claim_token=item.claim_token,
                reason="workstation capacity preflight failed",
                details={
                    "preflight": preflight.to_dict(),
                    "capacity": capacity.to_dict(),
                },
            )
            return None
        family = self._managed_accelerator_family(item)
        if family and item.claim_token:
            idle, availability = self._stable_external_idle(family)
            evidence = {"samples": [value.to_dict() for value in availability]}
            latest = availability[-1]
            from halo_forge.managed_runtime.service import ManagedRuntimeService

            runtime = ManagedRuntimeService(
                self.database,
                root=(requirements.get("runtime_root") or Path.home() / ".halo-forge" / "runtimes"),
                scheduler=self,
                occupancy_probe=self.accelerator_probe or __import__(
                    "halo_forge.managed_runtime.occupancy", fromlist=["probe_accelerator"]
                ).probe_accelerator,
            )
            runtime.record_preflight_decision(
                accelerator_family=family,
                decision=("idle" if idle else ("unknown" if latest.state == "unknown" else "waiting")),
                evidence=evidence,
                work_item_id=item.id,
                runtime_revision_id=str(requirements.get("runtime_profile_revision_id") or "") or None,
            )
            if not idle:
                from datetime import timedelta

                owners = [
                    {
                        "pid": owner.pid,
                        "executable": owner.executable,
                        "elapsed_seconds": owner.elapsed_seconds,
                    }
                    for owner in latest.owners
                ]
                self.database.defer_claimed_work_item_for_accelerator(
                    item.id,
                    claim_token=item.claim_token,
                    reason=latest.reason or "accelerator occupancy could not be verified idle",
                    details={"availability": latest.to_dict(), "owners": owners},
                    not_before=((now or datetime.now(timezone.utc)) + timedelta(seconds=15)).isoformat(),
                )
                return None
        if item.claim_token:
            refreshed = self.database.heartbeat_work_item(
                item.id,
                claim_token=item.claim_token,
                stage="starting",
                progress={
                    "capacity_preflight": preflight.to_dict(),
                    "capacity_override": preflight.overridden,
                },
                lease_ttl_seconds=self.lease_ttl_seconds,
                now=now,
            )
            return refreshed or item
        return item

    def heartbeat(
        self,
        item: WorkItemRecord,
        *,
        stage: Optional[str] = None,
        progress: Optional[Mapping[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> Optional[WorkItemRecord]:
        if not item.claim_token:
            return None
        return self.database.heartbeat_work_item(
            item.id,
            claim_token=item.claim_token,
            stage=stage,
            progress=progress,
            lease_ttl_seconds=self.lease_ttl_seconds,
            now=now,
        )

    def bind_process(
        self,
        item: WorkItemRecord,
        *,
        child_pid: int,
        child_pid_started_at: Optional[float] = None,
    ) -> Optional[WorkItemRecord]:
        if not item.claim_token:
            return None
        started = (
            float(child_pid_started_at)
            if child_pid_started_at is not None
            else process_start_time(child_pid)
        )
        if started is None:
            raise ValueError("could not resolve child process start identity")
        return self.database.bind_work_item_process(
            item.id,
            claim_token=item.claim_token,
            worker_pid=int(child_pid),
            worker_pid_started_at=started,
        )

    def complete(
        self,
        item: WorkItemRecord,
        *,
        result: Optional[Mapping[str, Any]] = None,
        ignore_late_cancel: bool = False,
    ) -> Optional[WorkItemRecord]:
        if not item.claim_token:
            return None
        return self.database.finish_work_item(
            item.id,
            claim_token=item.claim_token,
            result=result,
            ignore_late_cancel=ignore_late_cancel,
        )

    def fail(self, item: WorkItemRecord, *, error: str) -> Optional[WorkItemRecord]:
        if not item.claim_token:
            return None
        return self.database.finish_work_item(item.id, claim_token=item.claim_token, error=error)

    def cancel(self, work_item_id: str) -> Optional[WorkItemRecord]:
        item = self.database.request_cancel_work_item(work_item_id)
        if item is None:
            return None
        # Keep linked domain jobs truthful when cancellation originates in the
        # global Activity Center rather than their dedicated screen.
        if item.domain_kind == "evaluation" and item.domain_id:
            changes: dict[str, Any] = {"cancel_requested": True}
            if item.status == "cancelled":
                changes.update(
                    status="cancelled",
                    stage="cancelled",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
            self.database.update_evaluation(item.domain_id, **changes)
        elif (
            item.domain_kind == "dataset_job"
            and item.domain_id
            and item.launch_spec.get("dataset_root")
        ):
            try:
                from halo_forge.data_lab.jobs import SerialJobManager

                manager = SerialJobManager(
                    Path(str(item.launch_spec["dataset_root"])) / ".jobs.json",
                    recover=False,
                )
                try:
                    manager.cancel(item.domain_id)
                finally:
                    manager.shutdown(wait=False, cancel_futures=False)
            except Exception:
                # The durable work item remains cancelled even if a damaged
                # legacy JSON job file needs later reconciliation.
                pass
        elif item.domain_kind == "dataset_inspection" and item.domain_id:
            inspection = self.database.get_dataset_source_inspection(item.domain_id)
            if inspection is not None and inspection.status != "completed":
                self.database.update_dataset_source_inspection(
                    item.domain_id,
                    status="cancelled" if item.status == "cancelled" else inspection.status,
                    error=("inspection cancellation requested" if item.status == "cancelled" else inspection.error),
                    completed_at=(
                        datetime.now(timezone.utc).isoformat()
                        if item.status == "cancelled"
                        else inspection.completed_at
                    ),
                )
                if inspection.import_id and item.status == "cancelled":
                    self.database.update_dataset_import(
                        inspection.import_id, status="cancelled"
                    )
        elif item.domain_kind == "document_extraction" and item.domain_id:
            extraction = self.database.get_document_extraction(item.domain_id)
            if (
                extraction is not None
                and extraction.status != "completed"
                and item.status == "cancelled"
            ):
                self.database.update_document_extraction(
                    extraction.id,
                    status="cancelled",
                    error="corpus extraction cancellation requested",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
        elif item.domain_kind == "acquisition_batch" and item.domain_id:
            with self.database._lock:
                self.database._conn.execute(
                    """UPDATE acquisition_batches
                       SET status='cancelled',stage='cancelled',completed_at=?
                       WHERE id=? AND status NOT IN ('ready','cancelled')""",
                    (datetime.now(timezone.utc).isoformat(), item.domain_id),
                )
                self.database._conn.commit()
        elif item.domain_kind == "dataset_repair_session" and item.domain_id:
            with self.database._lock:
                self.database._conn.execute(
                    """UPDATE dataset_repair_sessions
                       SET cancel_requested=1,
                           status=CASE WHEN ?='cancelled' THEN 'cancelled' ELSE status END,
                           stage=CASE WHEN ?='cancelled' THEN 'cancelled' ELSE stage END,
                           updated_at=? WHERE id=?""",
                    (item.status, item.status, datetime.now(timezone.utc).isoformat(), item.domain_id),
                )
                self.database._conn.commit()
        elif item.domain_kind == "dataset_repair_preview" and item.domain_id:
            with self.database._lock:
                row = self.database._conn.execute(
                    "SELECT session_id FROM dataset_repair_previews WHERE id=?",
                    (item.domain_id,),
                ).fetchone()
                if row is not None:
                    self.database._conn.execute(
                        "UPDATE dataset_repair_sessions SET cancel_requested=1,updated_at=? WHERE id=?",
                        (datetime.now(timezone.utc).isoformat(), row["session_id"]),
                    )
                if item.status == "cancelled":
                    self.database._conn.execute(
                        "UPDATE dataset_repair_previews SET status='cancelled',error='preview cancellation requested',completed_at=? WHERE id=?",
                        (datetime.now(timezone.utc).isoformat(), item.domain_id),
                    )
                self.database._conn.commit()
        elif item.domain_kind == "support_bundle" and item.domain_id:
            with self.database._lock:
                self.database._conn.execute(
                    """UPDATE support_bundles SET cancel_requested=1,
                       status=CASE WHEN ?='cancelled' THEN 'cancelled' ELSE status END,
                       error=CASE WHEN ?='cancelled' THEN 'support bundle cancellation requested' ELSE error END,
                       completed_at=CASE WHEN ?='cancelled' THEN ? ELSE completed_at END
                       WHERE id=?""",
                    (
                        item.status, item.status, item.status,
                        datetime.now(timezone.utc).isoformat(), item.domain_id,
                    ),
                )
                self.database._conn.commit()
        elif item.domain_kind == "release_qualification" and item.domain_id:
            with self.database._lock:
                self.database._conn.execute(
                    """UPDATE release_qualifications SET cancel_requested=1,
                       status=CASE WHEN ?='cancelled' THEN 'cancelled' ELSE status END,
                       error=CASE WHEN ?='cancelled' THEN 'release qualification cancellation requested' ELSE error END,
                       completed_at=CASE WHEN ?='cancelled' THEN ? ELSE completed_at END
                       WHERE id=?""",
                    (
                        item.status, item.status, item.status,
                        datetime.now(timezone.utc).isoformat(), item.domain_id,
                    ),
                )
                self.database._conn.commit()
        elif item.domain_kind in {"model_preparation", "training_capacity_check"} and item.domain_id:
            table = (
                "model_preparations"
                if item.domain_kind == "model_preparation"
                else "training_capacity_checks"
            )
            with self.database._lock:
                if item.domain_kind == "training_capacity_check":
                    self.database._conn.execute(
                        f"""UPDATE {table} SET cancel_requested=1,
                           status=CASE WHEN ?='cancelled' THEN 'cancelled' ELSE status END,
                           stage=CASE WHEN ?='cancelled' THEN 'cancelled' ELSE stage END,
                           error=CASE WHEN ?='cancelled' THEN 'capacity check cancellation requested' ELSE error END,
                           completed_at=CASE WHEN ?='cancelled' THEN ? ELSE completed_at END
                           WHERE id=?""",
                        (
                            item.status, item.status, item.status, item.status,
                            datetime.now(timezone.utc).isoformat(), item.domain_id,
                        ),
                    )
                else:
                    self.database._conn.execute(
                        f"""UPDATE {table} SET cancel_requested=1,
                           status=CASE WHEN ?='cancelled' THEN 'cancelled' ELSE status END,
                           error=CASE WHEN ?='cancelled' THEN 'model preparation cancellation requested' ELSE error END,
                           completed_at=CASE WHEN ?='cancelled' THEN ? ELSE completed_at END
                           WHERE id=?""",
                        (
                            item.status, item.status, item.status,
                            datetime.now(timezone.utc).isoformat(), item.domain_id,
                        ),
                    )
                self.database._conn.commit()
        elif item.domain_kind == "verifier_calibration" and item.domain_id:
            changes: dict[str, Any] = {
                "cancel_requested": 1,
                "stage": "cancelling",
            }
            if item.status == "cancelled":
                changes.update(
                    status="cancelled",
                    stage="cancelled",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
            assignments = ", ".join(f"{key}=?" for key in changes)
            with self.database._lock:
                self.database._conn.execute(
                    f"UPDATE verifier_calibrations SET {assignments}, updated_at=? WHERE id=?",
                    (
                        *changes.values(),
                        datetime.now(timezone.utc).isoformat(),
                        item.domain_id,
                    ),
                )
                self.database._conn.commit()
        elif item.domain_kind == "reward_integrity_audit" and item.domain_id:
            changes: dict[str, Any] = {
                "cancel_requested": 1,
                "stage": "cancelling",
            }
            if item.status == "cancelled":
                changes.update(
                    status="cancelled",
                    stage="cancelled",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
            assignments = ", ".join(f"{key}=?" for key in changes)
            with self.database._lock:
                self.database._conn.execute(
                    f"UPDATE reward_integrity_audits SET {assignments}, updated_at=? WHERE id=?",
                    (
                        *changes.values(),
                        datetime.now(timezone.utc).isoformat(),
                        item.domain_id,
                    ),
                )
                self.database._conn.commit()
        return item

    def retry(
        self,
        work_item_id: str,
        *,
        reason: str = "operator requested retry",
        force: bool = True,
        backoff_seconds: Optional[float] = None,
        sync_domain: bool = True,
    ) -> Optional[WorkItemRecord]:
        item = self.database.retry_work_item(
            work_item_id,
            reason=reason,
            force=force,
            backoff_seconds=backoff_seconds,
        )
        if item is None or not sync_domain:
            return item
        try:
            if item.domain_kind == "evaluation" and item.domain_id:
                evaluation = self.database.get_evaluation(item.domain_id)
                if evaluation is not None and evaluation.status in {
                    "failed",
                    "cancelled",
                    "interrupted",
                }:
                    self.database.update_evaluation(
                        evaluation.id,
                        status="queued",
                        stage="queued",
                        processed_samples=0,
                        total_samples=None,
                        error=None,
                        cancel_requested=False,
                        retry_count=evaluation.retry_count + 1,
                        started_at=None,
                        completed_at=None,
                    )
            elif (
                item.domain_kind == "dataset_job"
                and item.domain_id
                and item.launch_spec.get("dataset_root")
            ):
                from halo_forge.data_lab.jobs import SerialJobManager

                manager = SerialJobManager(
                    Path(str(item.launch_spec["dataset_root"])) / ".jobs.json",
                    recover=False,
                )
                try:
                    manager.reset_for_retry(item.domain_id)
                finally:
                    manager.shutdown(wait=False, cancel_futures=False)
            elif item.domain_kind == "dataset_inspection" and item.domain_id:
                inspection = self.database.get_dataset_source_inspection(item.domain_id)
                if inspection is not None and inspection.status in {
                    "failed",
                    "cancelled",
                    "interrupted",
                }:
                    self.database.update_dataset_source_inspection(
                        inspection.id,
                        status="queued",
                        error=None,
                        completed_at=None,
                    )
                    if inspection.import_id:
                        self.database.update_dataset_import(
                            inspection.import_id,
                            status="inspecting",
                            work_item_id=item.id,
                            error=None,
                            completed_at=None,
                        )
            elif item.domain_kind == "document_extraction" and item.domain_id:
                extraction = self.database.get_document_extraction(item.domain_id)
                if extraction is not None and extraction.status in {
                    "failed",
                    "cancelled",
                    "interrupted",
                }:
                    self.database.update_document_extraction(
                        extraction.id,
                        status="queued",
                        work_item_id=item.id,
                        error=None,
                        completed_at=None,
                    )
            elif item.domain_kind == "acquisition_batch" and item.domain_id:
                with self.database._lock:
                    self.database._conn.execute(
                        """UPDATE acquisition_batches
                           SET status='queued',stage='resolving_sources',error=NULL,
                               completed_at=NULL
                           WHERE id=? AND status IN
                               ('failed','cancelled','interrupted','needs_reconciliation')""",
                        (item.domain_id,),
                    )
                    self.database._conn.commit()
            elif item.domain_kind == "verifier_calibration" and item.domain_id:
                with self.database._lock:
                    self.database._conn.execute(
                        """UPDATE verifier_calibrations
                           SET status='queued',stage='resume_pending',error=NULL,
                               cancel_requested=0,retry_count=retry_count+1,
                               completed_at=NULL,updated_at=?
                           WHERE id=? AND status IN
                               ('failed','cancelled','interrupted','needs_reconciliation')""",
                        (datetime.now(timezone.utc).isoformat(), item.domain_id),
                    )
                    self.database._conn.commit()
            elif item.domain_kind == "reward_integrity_audit" and item.domain_id:
                with self.database._lock:
                    self.database._conn.execute(
                        """UPDATE reward_integrity_audits
                           SET status='queued',stage='resume_pending',error=NULL,
                               cancel_requested=0,retry_count=retry_count+1,
                               completed_at=NULL,updated_at=?
                           WHERE id=? AND status IN
                               ('failed','cancelled','interrupted','needs_reconciliation')""",
                        (datetime.now(timezone.utc).isoformat(), item.domain_id),
                    )
                    self.database._conn.commit()
            elif item.domain_kind in {"model_preparation", "training_capacity_check"} and item.domain_id:
                table = (
                    "model_preparations"
                    if item.domain_kind == "model_preparation"
                    else "training_capacity_checks"
                )
                with self.database._lock:
                    if item.domain_kind == "training_capacity_check":
                        self.database._conn.execute(
                            f"""UPDATE {table}
                                SET status='queued',stage='queued',error=NULL,
                                    cancel_requested=0,completed_at=NULL,
                                    progress_json='{{\"stage\":\"queued\",\"retry\":true}}'
                                WHERE id=? AND status IN ('blocked','failed','cancelled','stale')""",
                            (item.domain_id,),
                        )
                    else:
                        self.database._conn.execute(
                            f"""UPDATE {table}
                                SET status='queued',error=NULL,cancel_requested=0,
                                    completed_at=NULL,
                                    progress_json='{{\"stage\":\"queued\",\"retry\":true}}'
                                WHERE id=? AND status IN ('blocked','failed','cancelled')""",
                            (item.domain_id,),
                        )
                    self.database._conn.commit()
        except Exception:
            # A retry that cannot reset its domain record is not executable.
            # Cancel it visibly rather than leaving a queue item that will fail
            # at claim time with a stale terminal domain state.
            return self.database.request_cancel_work_item(item.id)
        return self.database.get_work_item(item.id) or item

    def _owning_worker_is_dead(self, worker_id: str) -> bool:
        """Report positive evidence that another worker's process is gone.

        The training child exiting proves nothing about its owner: a worker
        routinely spends minutes publishing artifacts after the child is
        reaped. Only the owning worker's registered ``(pid, pid_started_at)``
        identity can answer this, and an unknown or incomplete identity is
        treated as alive so a second supervisor cannot strip a healthy claim.
        """

        if not worker_id or worker_id == self.worker_id:
            return False
        identity = self.database.worker_process_identity(worker_id)
        if identity is None:
            return False
        pid, pid_started_at = identity
        if pid is None or pid_started_at is None:
            return False
        return not self.process_probe(pid, pid_started_at)

    def _recoverable_running_items(self) -> list[WorkItemRecord]:
        """Return running items this worker may adopt or reconcile.

        Items claimed by this worker are always recoverable. Another worker's
        items are only in scope once its process identity is provably dead;
        anything else stays with its owner and, if that owner really is gone,
        is released later by lease expiry in ``recover_stale_work_items``.
        """

        owned = self.database.list_work_items(statuses=["running"], worker_id=self.worker_id)
        recoverable = list(owned)
        known = {item.id for item in owned}
        for item in self.database.list_work_items(statuses=["running"]):
            if item.id in known:
                continue
            owner = str(item.worker_id or "")
            if owner and not self._owning_worker_is_dead(owner):
                continue
            recoverable.append(item)
        return recoverable

    def recover_or_adopt(self, *, now: Optional[datetime] = None) -> RecoveryResult:
        """Adopt live child identities and interrupt dead/reused PIDs.

        A live process remains attached to its existing claim token and log
        path; the caller may resume tailing it. Interrupted work is not
        automatically retried, preventing duplicate output directories. Work
        owned by another live worker is left untouched.
        """

        adopted: list[WorkItemRecord] = []
        interrupted: list[WorkItemRecord] = []
        running = self._recoverable_running_items()
        instant = now or datetime.now(timezone.utc)
        for item in running:
            if self.process_probe(item.worker_pid, item.worker_pid_started_at):
                refreshed = (
                    self.database.heartbeat_work_item(
                        item.id,
                        claim_token=item.claim_token or "",
                        lease_ttl_seconds=self.lease_ttl_seconds,
                        now=instant,
                    )
                    if item.claim_token
                    else None
                )
                adopted.append(refreshed or item)
            elif item.claim_token:
                changed = self.database.interrupt_work_item_if_claimed(
                    item.id,
                    claim_token=item.claim_token,
                    error="worker process identity is dead or ambiguous; operator reconciliation required",
                    status="needs_reconciliation",
                )
                if changed is not None:
                    interrupted.append(changed)
        stale = self.database.recover_stale_work_items(
            stale_before=instant - timedelta(seconds=self.lease_ttl_seconds),
            now=instant,
        )
        known = {item.id for item in interrupted}
        interrupted.extend(item for item in stale if item.id not in known)
        inspection_ids = sorted(
            {
                str(item.domain_id)
                for item in interrupted
                if item.domain_kind == "dataset_inspection" and item.domain_id
            }
        )
        for inspection_id in inspection_ids:
            inspection = self.database.get_dataset_source_inspection(inspection_id)
            if inspection is None or inspection.status == "completed":
                continue
            self.database.update_dataset_source_inspection(
                inspection_id,
                status="interrupted",
                error="worker process identity is dead or ambiguous; retry is required",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            if inspection.import_id:
                self.database.update_dataset_import(
                    inspection.import_id,
                    status="failed",
                    error="source inspection was interrupted; retry is required",
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
        extraction_ids = sorted(
            {
                str(item.domain_id)
                for item in interrupted
                if item.domain_kind == "document_extraction" and item.domain_id
            }
        )
        for extraction_id in extraction_ids:
            extraction = self.database.get_document_extraction(extraction_id)
            if extraction is None or extraction.status == "completed":
                continue
            self.database.update_document_extraction(
                extraction_id,
                status="interrupted",
                error="worker process identity is dead or ambiguous; retry is required",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        acquisition_ids = sorted(
            {
                str(item.domain_id)
                for item in interrupted
                if item.domain_kind == "acquisition_batch" and item.domain_id
            }
        )
        if acquisition_ids:
            marks = ",".join("?" for _ in acquisition_ids)
            with self.database._lock:
                self.database._conn.execute(
                    f"""UPDATE acquisition_batches
                        SET status='needs_reconciliation',stage='needs_reconciliation',
                            error='worker process identity is dead or ambiguous'
                        WHERE id IN ({marks}) AND status NOT IN ('ready','cancelled')""",
                    tuple(acquisition_ids),
                )
                self.database._conn.commit()
        calibration_ids = sorted(
            {
                str(item.domain_id)
                for item in interrupted
                if item.domain_kind == "verifier_calibration" and item.domain_id
            }
        )
        if calibration_ids:
            marks = ",".join("?" for _ in calibration_ids)
            with self.database._lock:
                self.database._conn.execute(
                    f"""UPDATE verifier_calibrations
                        SET status='needs_reconciliation',stage='needs_reconciliation',
                            error='worker process identity is dead or ambiguous',
                            cancel_requested=0,updated_at=?
                        WHERE id IN ({marks}) AND status NOT IN ('completed','cancelled')""",
                    (datetime.now(timezone.utc).isoformat(), *calibration_ids),
                )
                self.database._conn.commit()
        audit_ids = sorted(
            {
                str(item.domain_id)
                for item in interrupted
                if item.domain_kind == "reward_integrity_audit" and item.domain_id
            }
        )
        if audit_ids:
            marks = ",".join("?" for _ in audit_ids)
            with self.database._lock:
                self.database._conn.execute(
                    f"""UPDATE reward_integrity_audits
                        SET status='needs_reconciliation',stage='needs_reconciliation',
                            error='worker process identity is dead or ambiguous',
                            cancel_requested=0,updated_at=?
                        WHERE id IN ({marks}) AND status NOT IN ('completed','cancelled')""",
                    (datetime.now(timezone.utc).isoformat(), *audit_ids),
                )
                self.database._conn.commit()
        training_plan_items = [
            item
            for item in interrupted
            if item.domain_kind in {"model_preparation", "training_capacity_check"}
            and item.domain_id
        ]
        if training_plan_items:
            completed_at = datetime.now(timezone.utc).isoformat()
            with self.database._lock:
                for item in training_plan_items:
                    if item.domain_kind == "training_capacity_check":
                        self.database._conn.execute(
                            """UPDATE training_capacity_checks
                               SET status='failed',stage='failed',cancel_requested=0,
                                   error='capacity check was interrupted; retry is required',
                                   completed_at=?
                               WHERE id=? AND status NOT IN ('ready','ready_with_adjustment','cancelled')""",
                            (completed_at, item.domain_id),
                        )
                        root = Path(
                            str(
                                item.launch_spec.get("training_plan_root")
                                or Path.home() / ".halo-forge"
                            )
                        ).expanduser()
                        shutil.rmtree(
                            root / "training-capacity" / "scratch" / str(item.domain_id),
                            ignore_errors=True,
                        )
                    else:
                        self.database._conn.execute(
                            """UPDATE model_preparations
                               SET status='failed',cancel_requested=0,
                                   error='model preparation was interrupted; retry is required',
                                   completed_at=?
                               WHERE id=? AND status NOT IN ('completed','cancelled')""",
                            (completed_at, item.domain_id),
                        )
                self.database._conn.commit()
        return RecoveryResult(tuple(adopted), tuple(interrupted))

    def start_serving(
        self,
        *,
        serving_id: str,
        metadata: Optional[Mapping[str, Any]] = None,
        process_pid: Optional[int] = None,
        process_started_at: Optional[float] = None,
    ) -> Optional[ResourceLeaseRecord]:
        pid = int(process_pid) if process_pid is not None else os.getpid()
        started = (
            float(process_started_at) if process_started_at is not None else process_start_time(pid)
        )
        return self.database.acquire_serving_lease(
            holder_id=serving_id,
            metadata=metadata,
            holder_pid=pid,
            holder_pid_started_at=started,
        )

    def heartbeat_serving(
        self,
        *,
        serving_id: str,
        metadata: Optional[Mapping[str, Any]] = None,
        process_pid: Optional[int] = None,
        process_started_at: Optional[float] = None,
    ) -> Optional[ResourceLeaseRecord]:
        pid = int(process_pid) if process_pid is not None else os.getpid()
        started = (
            float(process_started_at) if process_started_at is not None else process_start_time(pid)
        )
        return self.database.heartbeat_serving_lease(
            holder_id=serving_id,
            holder_pid=pid,
            holder_pid_started_at=started,
            metadata=metadata,
        )

    def recover_stale_serving(self) -> tuple[ResourceLeaseRecord, ...]:
        """Release serving leases whose PID/start identity cannot be verified."""

        recovered: list[ResourceLeaseRecord] = []
        for lease in self.database.list_resource_leases():
            if lease.holder_type != "serving":
                continue
            if self.process_probe(lease.holder_pid, lease.holder_pid_started_at):
                continue
            if self.database.release_serving_lease(
                holder_id=lease.holder_id, resource_key=lease.resource_key
            ):
                recovered.append(lease)
        return tuple(recovered)

    def stop_serving(self, *, serving_id: str) -> bool:
        return self.database.release_serving_lease(holder_id=serving_id)
