"""Public types for V18 guided training plans and capacity checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


class Serializable:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingPlanReason(Serializable):
    code: str
    summary: str
    detail: str
    kind: str = "recommendation"


@dataclass(frozen=True)
class TrainingResourceForecast(Serializable):
    download_bytes: Optional[int]
    scratch_bytes: Optional[int]
    checkpoint_bytes: Optional[int]
    peak_memory_bytes: Optional[int]
    proof_seconds_range: Optional[tuple[int, int]]
    full_run_seconds_range: Optional[tuple[int, int]]
    provenance: Dict[str, str] = field(default_factory=dict)
    confidence: str = "low"


@dataclass(frozen=True)
class TrainingPlanProfile(Serializable):
    id: str
    version: str
    label: str
    trainer_mode: str
    canonical_shapes: tuple[str, ...]
    proof_max_samples: int
    epochs: int
    cycles: int
    microbatch: int
    gradient_accumulation: int
    learning_rate: float
    max_sequence_length: int
    adaptation: str
    precision: str


@dataclass(frozen=True)
class TrainingCapacityCapability(Serializable):
    id: str
    version: str
    trainer_mode: str
    backends: tuple[str, ...]
    scratch_step: bool
    fallback_steps: tuple[str, ...]
    unavailable_reason: Optional[str] = None


@dataclass(frozen=True)
class TrainingPlanRevision(Serializable):
    id: str
    plan_id: str
    revision_number: int
    status: str
    content_hash: str
    profile_id: str
    profile_version: str
    dataset_version_id: str
    scenario_revision_id: Optional[str]
    trainer_mode: str
    backend: str
    model_id: str
    model_revision: Optional[str]
    resolved_model_commit: Optional[str]
    definition: Dict[str, Any]
    reasons: tuple[TrainingPlanReason, ...]
    forecast: TrainingResourceForecast
    compute_shape_hash: str
    runtime_hash: str
    created_at: str
    runtime_profile_revision_id: Optional[str] = None
    training_path_revision_id: Optional[str] = None
    training_path_certification_id: Optional[str] = None


@dataclass(frozen=True)
class TrainingPlan(Serializable):
    id: str
    dataset_version_id: str
    scenario_revision_id: Optional[str]
    status: str
    latest_revision_id: Optional[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class TrainingPlanRecommendation(Serializable):
    plan: TrainingPlan
    revision: TrainingPlanRevision
    alternatives: tuple[Dict[str, Any], ...]
    summary: str
    primary_action: Dict[str, Any]


@dataclass(frozen=True)
class ModelPreparation(Serializable):
    id: str
    plan_revision_id: str
    status: str
    requested_model_id: str
    requested_revision: Optional[str]
    resolved_commit: Optional[str]
    cache_path: Optional[str]
    manifest_path: Optional[str]
    manifest_hash: Optional[str]
    size_bytes: Optional[int]
    access: Dict[str, Any]
    progress: Dict[str, Any]
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]


@dataclass(frozen=True)
class TrainingCapacityAttempt(Serializable):
    id: str
    capacity_check_id: str
    ordinal: int
    configuration: Dict[str, Any]
    status: str
    sample_identity: Dict[str, Any]
    measurements: Dict[str, Any]
    error_class: Optional[str]
    error: Optional[str]
    scratch_cleaned: bool
    created_at: str
    completed_at: Optional[str]


@dataclass(frozen=True)
class TrainingCapacityCheck(Serializable):
    id: str
    plan_revision_id: str
    model_preparation_id: Optional[str]
    status: str
    stage: str
    capability_id: str
    capability_version: str
    compute_shape_hash: str
    runtime_hash: str
    selected_adjustment: Dict[str, Any]
    forecast: TrainingResourceForecast
    progress: Dict[str, Any]
    primary_remedy: Dict[str, Any]
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]


@dataclass(frozen=True)
class TrainingPlanReadiness(Serializable):
    plan_revision_id: str
    status: str
    display_status: str
    summary: str
    model_preparation: Optional[ModelPreparation]
    capacity_check: Optional[TrainingCapacityCheck]
    blockers: tuple[Dict[str, Any], ...]
    primary_action: Dict[str, Any]
    notices: tuple[Dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TrainingPlanDecision(Serializable):
    id: str
    plan_revision_id: str
    decision: str
    reason: Optional[str]
    details: Dict[str, Any]
    created_at: str
