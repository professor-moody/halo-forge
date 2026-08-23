"""Public V19/V20 managed-runtime and accelerator-coexistence types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


class Serializable:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManagedRuntimeCapability(Serializable):
    accelerator_family: str
    available: bool
    status: str
    summary: str
    runtime_revision_id: Optional[str] = None
    qualification_id: Optional[str] = None
    supported_trainers: tuple[str, ...] = ()
    unavailable_reason: Optional[str] = None


@dataclass(frozen=True)
class ManagedRuntimeProfile(Serializable):
    id: str
    name: str
    accelerator_family: str
    description: Optional[str]
    latest_revision_id: Optional[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ManagedRuntimeRevision(Serializable):
    id: str
    profile_id: str
    revision_number: int
    content_hash: str
    adapter_id: str
    adapter_version: str
    engine: str
    base_image: Optional[str]
    base_image_digest: Optional[str]
    derived_image_ref: Optional[str]
    dependency_lock: Dict[str, str]
    configuration: Dict[str, Any]
    trainer_contracts: tuple[str, ...]
    download_bytes: Optional[int]
    installed_bytes: Optional[int]
    created_at: str


@dataclass(frozen=True)
class RuntimePreparation(Serializable):
    id: str
    runtime_revision_id: str
    status: str
    stage: str
    engine: str
    image_id: Optional[str]
    image_digest: Optional[str]
    storage_path: Optional[str]
    manifest_path: Optional[str]
    manifest_hash: Optional[str]
    progress: Dict[str, Any]
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]


@dataclass(frozen=True)
class RuntimeQualificationStep(Serializable):
    qualification_id: str
    ordinal: int
    step_id: str
    label: str
    status: str
    command_hash: Optional[str]
    result: Dict[str, Any]
    log_path: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


@dataclass(frozen=True)
class RuntimeQualification(Serializable):
    id: str
    runtime_revision_id: str
    preparation_id: Optional[str]
    status: str
    stage: str
    host_identity_hash: str
    device_identity_hash: str
    runtime_identity_hash: str
    qualification_hash: Optional[str]
    evidence_path: Optional[str]
    progress: Dict[str, Any]
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]
    steps: tuple[RuntimeQualificationStep, ...] = ()


@dataclass(frozen=True)
class RuntimeBinding(Serializable):
    id: str
    runtime_revision_id: str
    qualification_id: Optional[str]
    domain_kind: str
    domain_id: str
    role: str
    runtime_identity_hash: str
    details: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class ExternalAcceleratorOwner(Serializable):
    pid: Optional[int]
    executable: str
    elapsed_seconds: Optional[int]
    device: Optional[str] = None
    memory_bytes: Optional[int] = None


@dataclass(frozen=True)
class AcceleratorAvailability(Serializable):
    accelerator_family: str
    state: str
    sampled_at: str
    utilization_percent: Optional[float]
    owners: tuple[ExternalAcceleratorOwner, ...] = ()
    reason: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def idle(self) -> bool:
        return self.state == "idle"


@dataclass(frozen=True)
class AcceleratorPreflightDecision(Serializable):
    id: str
    work_item_id: Optional[str]
    runtime_revision_id: Optional[str]
    accelerator_family: str
    decision: str
    sample_count: int
    evidence_hash: str
    evidence: Dict[str, Any]
    override_reason: Optional[str]
    created_at: str

