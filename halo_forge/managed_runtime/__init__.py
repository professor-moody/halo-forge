"""Managed accelerator runtimes and conservative coexistence probes."""

from .adapters import (
    RUNTIME_EXECUTION_ADAPTERS,
    RuntimeCommand,
    RuntimeExecutionAdapter,
    RuntimeExecutionAdapterRegistry,
    RuntimeMount,
)
from .models import (
    AcceleratorAvailability,
    AcceleratorPreflightDecision,
    ExternalAcceleratorOwner,
    ManagedRuntimeCapability,
    ManagedRuntimeProfile,
    ManagedRuntimeRevision,
    RuntimeBinding,
    RuntimePreparation,
    RuntimeQualification,
    RuntimeQualificationStep,
)
from .occupancy import probe_accelerator, probe_cuda, probe_rocm, wait_for_stable_idle
from .service import ManagedRuntimeError, ManagedRuntimeService

__all__ = [
    "RUNTIME_EXECUTION_ADAPTERS",
    "RuntimeCommand",
    "RuntimeExecutionAdapter",
    "RuntimeExecutionAdapterRegistry",
    "RuntimeMount",
    "ManagedRuntimeCapability",
    "ManagedRuntimeProfile",
    "ManagedRuntimeRevision",
    "RuntimePreparation",
    "RuntimeQualification",
    "RuntimeQualificationStep",
    "RuntimeBinding",
    "AcceleratorAvailability",
    "ExternalAcceleratorOwner",
    "AcceleratorPreflightDecision",
    "probe_accelerator",
    "probe_cuda",
    "probe_rocm",
    "wait_for_stable_idle",
    "ManagedRuntimeError",
    "ManagedRuntimeService",
]
