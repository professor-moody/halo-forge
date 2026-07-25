"""One supervised workstation worker for dashboard and desktop runtimes."""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from halo_forge.run_db import LabV4Catalog, RunDatabase, WorkItemRecord
from halo_forge.workstation_jobs.resources import raw_telemetry_retention_cutoff
from halo_forge.workstation_jobs.scheduler import (
    WorkstationScheduler,
    process_matches,
    process_start_time,
)
from halo_forge.workstation_jobs.worker import WorkstationWorker


class WorkerSupervisor:
    """Own the dashboard worker and actively monitor processes adopted on restart."""

    MAINTENANCE_INTERVAL_SECONDS = 10.0
    TELEMETRY_PRUNE_INTERVAL_SECONDS = 60.0 * 60.0

    def __init__(self, database: RunDatabase):
        self.database = database
        self.scheduler = WorkstationScheduler(database)
        self.worker = WorkstationWorker(self.scheduler)
        self.catalog = LabV4Catalog(database)
        self._thread: Optional[threading.Thread] = None
        self._maintenance_thread: Optional[threading.Thread] = None
        self._adopted: Dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self._last_telemetry_prune: Optional[datetime] = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> "WorkerSupervisor":
        with self._lock:
            if self.running:
                return self
            self.catalog.register_worker(
                worker_id=self.scheduler.worker_id,
                pid=os.getpid(),
                pid_started_at=process_start_time(os.getpid()),
                version="lab-v4",
                capabilities={"subprocess": True, "single_heavy_operation": True},
                metadata={"managed_by": "dashboard"},
            )
            self.scheduler.recover_stale_serving()
            recovery = self.scheduler.recover_or_adopt()
            for item in recovery.adopted:
                self._start_adopted_monitor(item)
            self.catalog.prune_telemetry(
                before=raw_telemetry_retention_cutoff(now=datetime.now(timezone.utc)).isoformat()
            )
            self._last_telemetry_prune = datetime.now(timezone.utc)
            self.worker.stop_event.clear()
            self._thread = threading.Thread(
                target=self._watch,
                name="halo-forge-workstation-worker",
                daemon=True,
            )
            self._thread.start()
            self._maintenance_thread = threading.Thread(
                target=self._maintain,
                name="halo-forge-workstation-maintenance",
                daemon=True,
            )
            self._maintenance_thread.start()
        return self

    def _watch(self) -> None:
        try:
            self.worker.watch()
        finally:
            self.catalog.stop_worker(self.scheduler.worker_id)

    def _maintenance_once(self, *, now: Optional[datetime] = None) -> None:
        """Refresh idle-worker health and enforce time-based retention.

        Work-item heartbeats naturally cover an active worker. This companion
        heartbeat keeps an idle supervised worker visibly healthy, periodically
        recovers crashed serving leases, and ensures raw telemetry does not live
        past the thirty-day policy merely because the dashboard stayed open for
        weeks without restarting.
        """

        instant = now or datetime.now(timezone.utc)
        self.catalog.heartbeat_worker(self.scheduler.worker_id)
        self.scheduler.recover_stale_serving()
        last = self._last_telemetry_prune
        if (
            last is None
            or (instant - last).total_seconds() >= self.TELEMETRY_PRUNE_INTERVAL_SECONDS
        ):
            self.catalog.prune_telemetry(
                before=raw_telemetry_retention_cutoff(now=instant).isoformat()
            )
            self._last_telemetry_prune = instant

    def _maintain(self) -> None:
        while not self.worker.stop_event.wait(self.MAINTENANCE_INTERVAL_SECONDS):
            if not self.running:
                return
            try:
                self._maintenance_once()
            except Exception:
                # Maintenance is best-effort observability/recovery. The worker
                # queue must remain alive even if a telemetry prune encounters a
                # transient SQLite or filesystem error.
                continue

    def _start_adopted_monitor(self, item: WorkItemRecord) -> None:
        if item.id in self._adopted and self._adopted[item.id].is_alive():
            return

        def monitor() -> None:
            current = item
            while not self.worker.stop_event.wait(max(0.25, self.scheduler.lease_ttl_seconds / 3)):
                if not process_matches(current.worker_pid, current.worker_pid_started_at):
                    if current.claim_token:
                        self.database.interrupt_work_item_if_claimed(
                            current.id,
                            claim_token=current.claim_token,
                            status="needs_reconciliation",
                            error=(
                                "adopted process ended before Halo Forge could verify "
                                "publication; operator reconciliation required"
                            ),
                        )
                    return
                refreshed = self.scheduler.heartbeat(
                    current,
                    stage="adopted_running",
                    progress={"pid": current.worker_pid, "adopted": True},
                )
                if refreshed is None:
                    return
                current = refreshed

        thread = threading.Thread(
            target=monitor,
            name=f"halo-forge-adopted-{item.id}",
            daemon=True,
        )
        self._adopted[item.id] = thread
        thread.start()

    def stop(self, *, timeout: float = 15.0) -> None:
        with self._lock:
            self.worker.stop()
            thread = self._thread
            maintenance = self._maintenance_thread
        if thread and thread.is_alive():
            thread.join(timeout=max(0.0, timeout))
        if maintenance and maintenance.is_alive():
            maintenance.join(timeout=min(max(0.0, timeout), 2.0))
        self.catalog.stop_worker(self.scheduler.worker_id)


_SUPERVISORS: Dict[str, WorkerSupervisor] = {}
_SUPERVISOR_LOCK = threading.RLock()


def get_worker_supervisor(database: RunDatabase) -> WorkerSupervisor:
    path = str(getattr(database, "path", ":memory:"))
    key = (
        str(Path(path).expanduser().resolve()) if path != ":memory:" else f":memory:{id(database)}"
    )
    with _SUPERVISOR_LOCK:
        supervisor = _SUPERVISORS.get(key)
        if supervisor is None:
            supervisor = WorkerSupervisor(database)
            _SUPERVISORS[key] = supervisor
        return supervisor


__all__ = ["WorkerSupervisor", "get_worker_supervisor"]
