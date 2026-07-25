"""Durable single-workstation work dispatcher.

The scheduler is intentionally transport-neutral. API, CLI, and background
worker entrypoints enqueue and claim the same SQLite records through this
package; process spawning remains with the modality-specific launcher.
"""

from typing import TYPE_CHECKING, Any

from halo_forge.workstation_jobs.scheduler import (
    RecoveryResult,
    WorkstationScheduler,
    process_matches,
    process_start_time,
)
from halo_forge.workstation_jobs.resources import (
    AcceleratorCapacity,
    CapacityPreflightPolicy,
    CapacityPreflightResult,
    DiskCapacity,
    MemoryCapacity,
    ProcessCapacity,
    TelemetryRollup,
    WorkstationCapacity,
    WorkstationTelemetrySample,
    aggregate_telemetry,
    evaluate_capacity_preflight,
    sample_workstation_capacity,
)

if TYPE_CHECKING:
    from halo_forge.workstation_jobs.worker import (
        ProcessHandle,
        ProcessRunner,
        SubprocessRunner,
        WorkstationWorker,
    )
    from halo_forge.workstation_jobs.supervisor import WorkerSupervisor

__all__ = [
    "RecoveryResult",
    "AcceleratorCapacity",
    "CapacityPreflightPolicy",
    "CapacityPreflightResult",
    "DiskCapacity",
    "MemoryCapacity",
    "ProcessCapacity",
    "TelemetryRollup",
    "WorkstationCapacity",
    "WorkstationTelemetrySample",
    "WorkstationScheduler",
    "WorkstationWorker",
    "WorkerSupervisor",
    "ProcessHandle",
    "ProcessRunner",
    "SubprocessRunner",
    "process_matches",
    "process_start_time",
    "aggregate_telemetry",
    "evaluate_capacity_preflight",
    "sample_workstation_capacity",
]


def __getattr__(name: str) -> Any:
    # Keep worker imports lazy so ``python -m halo_forge.workstation_jobs.worker``
    # does not preload the module and trigger runpy's duplicate-module warning.
    if name in {"ProcessHandle", "ProcessRunner", "SubprocessRunner", "WorkstationWorker"}:
        from halo_forge.workstation_jobs import worker

        return getattr(worker, name)
    if name == "WorkerSupervisor":
        from halo_forge.workstation_jobs.supervisor import WorkerSupervisor

        return WorkerSupervisor
    raise AttributeError(name)
