"""Public V21 training-path and workstation certification types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


class Serializable:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingPathProfileRevision(Serializable):
    id: str
    profile_id: str
    revision_number: int
    content_hash: str
    runtime_family: str
    backend: str
    scenario_revision_id: Optional[str]
    trainer_mode: str
    model_id: str
    model_revision: str
    tokenizer_processor_hash: str
    fixture_id: str
    fixture_hash: str
    trainer_adapter_version: str
    capacity_adapter_version: str
    configuration: Dict[str, Any]
    expected_artifacts: tuple[str, ...]
    created_at: str


@dataclass(frozen=True)
class TrainingPathCertificationStep(Serializable):
    certification_id: str
    ordinal: int
    step_id: str
    label: str
    status: str
    input_hash: Optional[str]
    result: Dict[str, Any]
    evidence_hash: Optional[str]
    log_path: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


@dataclass(frozen=True)
class TrainingPathCertification(Serializable):
    id: str
    path_revision_id: str
    runtime_revision_id: str
    runtime_qualification_id: str
    status: str
    stage: str
    host_identity_hash: str
    device_identity_hash: str
    runtime_identity_hash: str
    source_identity_hash: str
    certification_hash: Optional[str]
    evidence_path: Optional[str]
    progress: Dict[str, Any]
    resume_cursor: Dict[str, Any]
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]
    steps: tuple[TrainingPathCertificationStep, ...] = ()


@dataclass(frozen=True)
class CertificationRecoveryAction(Serializable):
    action: str
    label: str
    reason: str
    enabled: bool = True


@dataclass(frozen=True)
class TrainingPathCapability(Serializable):
    path_revision_id: str
    profile_id: str
    label: str
    scenario_revision_id: Optional[str]
    trainer_mode: str
    model_id: str
    runtime_family: str
    state: str
    display_status: str
    summary: str
    runtime_revision_id: Optional[str] = None
    runtime_qualification_id: Optional[str] = None
    certification_id: Optional[str] = None
    blocker: Optional[str] = None
    recovery_action: Optional[CertificationRecoveryAction] = None


@dataclass(frozen=True)
class TrainingPathCertificationMatrix(Serializable):
    runtime_family: str
    runtime_ready: bool
    beta_qualified: bool
    paths: tuple[TrainingPathCapability, ...]
    recommended_path_revision_id: Optional[str]


@dataclass(frozen=True)
class WorkstationCertification(Serializable):
    id: str
    runtime_revision_id: str
    runtime_qualification_id: str
    instruction_path_revision_id: str
    instruction_path_certification_id: Optional[str]
    status: str
    stage: str
    host_identity_hash: str
    device_identity_hash: str
    evidence: Dict[str, Any]
    qualification_hash: Optional[str]
    report_path: Optional[str]
    support_bundle_id: Optional[str]
    progress: Dict[str, Any]
    resume_cursor: Dict[str, Any]
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]

