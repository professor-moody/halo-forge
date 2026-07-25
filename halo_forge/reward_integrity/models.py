"""Transport-neutral models for reward integrity and training signals.

The dataclasses in this module intentionally do not know about any trainer or
HTTP framework.  Trainers, the scheduler, the API, and the CLI can therefore
exchange the same immutable evidence shapes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar


class Serializable:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


@dataclass(frozen=True)
class Page(Generic[T]):
    items: List[T]
    total: int
    limit: int
    offset: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.to_dict() if hasattr(item, "to_dict") else item for item in self.items],
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
        }


@dataclass(frozen=True)
class TrainingRecordRef(Serializable):
    record_id: str
    record_hash: str
    instance_id: str
    group_id: str
    identity_kind: str = "managed"
    lineage: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("record_id", "record_hash", "instance_id", "group_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if self.identity_kind not in {"managed", "virtual", "manual"}:
            raise ValueError("identity_kind must be managed, virtual, or manual")


@dataclass(frozen=True)
class TrainingSignalSnapshot(Serializable):
    snapshot_id: str
    record: TrainingRecordRef
    candidate_ordinal: int
    input: Dict[str, Any]
    output: Any
    expected: Any = None
    media: List[Dict[str, Any]] = field(default_factory=list)
    generation: Dict[str, Any] = field(default_factory=dict)
    optimizer_observation: Dict[str, Any] = field(default_factory=dict)
    selection: Dict[str, Any] = field(default_factory=dict)
    producer_model_hash: str = ""
    checkpoint_hash: str = ""
    runtime_identity: Dict[str, Any] = field(default_factory=dict)
    occurrence_id: Optional[str] = None
    identity_mode: str = "legacy_content_fallback"
    producer_model_identity: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id is required")
        if int(self.candidate_ordinal) < 0:
            raise ValueError("candidate_ordinal cannot be negative")


@dataclass(frozen=True)
class RewardSystem(Serializable):
    id: str
    name: str
    description: Optional[str]
    latest_revision_id: Optional[str]
    archived: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class RewardSystemAuditor(Serializable):
    reward_system_revision_id: str
    ordinal: int
    role: str
    verifier_revision_id: str
    correlated: bool = False
    correlation_reasons: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in {"primary_sentinel", "diagnostic"}:
            raise ValueError("auditor role must be primary_sentinel or diagnostic")
        if self.ordinal < 0:
            raise ValueError("auditor ordinal cannot be negative")


@dataclass(frozen=True)
class RewardSystemRevision(Serializable):
    id: str
    system_id: str
    revision_number: int
    content_hash: str
    optimizer_verifier_revision_id: str
    modality: str
    task_type: str
    input_mapping: Dict[str, Any]
    reward_mapping: Dict[str, Any]
    definition: Dict[str, Any]
    runtime_contract_hash: str
    created_at: str
    auditors: List[RewardSystemAuditor] = field(default_factory=list)

    @property
    def primary_sentinel(self) -> Optional[RewardSystemAuditor]:
        return next((item for item in self.auditors if item.role == "primary_sentinel"), None)


@dataclass(frozen=True)
class RewardAuditProtocol(Serializable):
    id: str
    name: str
    description: Optional[str]
    latest_revision_id: Optional[str]
    archived: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class RewardAuditProtocolRevision(Serializable):
    id: str
    protocol_id: str
    revision_number: int
    content_hash: str
    capture_mode: str
    definition: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class RewardIntegrityProfile(Serializable):
    id: str
    name: str
    description: Optional[str]
    latest_revision_id: Optional[str]
    archived: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class RewardIntegrityProfileRevision(Serializable):
    id: str
    profile_id: str
    revision_number: int
    content_hash: str
    template_kind: str
    promotable: bool
    requirements: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class TrainingSignalShard(Serializable):
    id: str
    run_id: str
    direct_run_segment_id: Optional[str]
    trial_segment_id: Optional[str]
    reward_system_revision_id: str
    protocol_revision_id: str
    capability_id: str
    capture_fidelity: str
    boundary_unit: str
    boundary_value: int
    trace_hash: str
    retained_set_hash: str
    event_count: int
    distinct_record_count: int
    aggregate: Dict[str, Any]
    dataset_identity: Dict[str, Any]
    producer_model_hash: str
    checkpoint_hash: str
    runtime_identity: Dict[str, Any]
    storage_path: str
    manifest_hash: str
    sealed: bool
    created_at: str


@dataclass(frozen=True)
class RewardIntegrityAudit(Serializable):
    id: str
    run_id: str
    direct_run_segment_id: Optional[str]
    trial_segment_id: Optional[str]
    signal_shard_id: str
    reward_system_revision_id: str
    protocol_revision_id: str
    integrity_profile_revision_id: str
    development_suite_revision_id: Optional[str]
    status: str
    stage: str
    processed_samples: int
    total_samples: Optional[int]
    distinct_record_count: int
    request: Dict[str, Any]
    runtime_identity: Dict[str, Any]
    runtime_identity_hash: str
    reuse_key: str
    artifact_path: Optional[str]
    manifest_hash: Optional[str]
    work_item_id: Optional[str]
    cancel_requested: bool
    retry_count: int
    error: Optional[str]
    created_at: str
    updated_at: str
    started_at: Optional[str]
    completed_at: Optional[str]


@dataclass(frozen=True)
class RewardIntegritySample(Serializable):
    audit_id: str
    ordinal: int
    snapshot_id: str
    record_id: str
    record_hash: str
    instance_id: str
    group_id: str
    candidate_ordinal: int
    selection_class: str
    diagnostic: bool
    input: Dict[str, Any]
    output: Any
    expected: Any
    media: List[Dict[str, Any]]
    generation: Dict[str, Any]
    lineage: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class RewardIntegrityObservation(Serializable):
    audit_id: str
    sample_ordinal: int
    role: str
    auditor_ordinal: int
    verifier_revision_id: str
    reward: Optional[float]
    normalized_reward: Optional[float]
    passed: Optional[bool]
    parsed_value: Any
    raw_output: Any
    details: Dict[str, Any]
    component_trace: List[Dict[str, Any]]
    latency_ms: Optional[float]
    error: Optional[str]
    runtime_identity: Dict[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        for name in ("reward", "normalized_reward", "latency_ms"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.normalized_reward is not None and not 0 <= self.normalized_reward <= 1:
            raise ValueError("normalized_reward must be in [0, 1]")


@dataclass(frozen=True)
class RewardIntegrityMetric(Serializable):
    audit_id: str
    name: str
    value: Optional[float]
    available: bool
    record_count: int
    subgroup: str = ""
    population: str = "uniform_core"
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    direction: Optional[str] = None
    missing_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True)
class RewardIntegrityDecision(Serializable):
    id: str
    audit_id: str
    integrity_profile_revision_id: str
    decision: str
    action: str
    reasons: List[str]
    evidence: Dict[str, Any]
    override: bool
    override_note: Optional[str]
    supersedes_decision_id: Optional[str]
    created_at: str


@dataclass(frozen=True)
class RewardIntegrityBinding(Serializable):
    id: str
    reward_system_revision_id: str
    protocol_revision_id: Optional[str]
    integrity_profile_revision_id: Optional[str]
    audit_id: Optional[str]
    domain_kind: str
    domain_id: str
    role: str
    binding_hash: str
    context: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class RewardIntegrityComparisonPair(Serializable):
    """One bounded, explicitly identified pair of retained audit evidence."""

    id: str
    pairing: str
    record_id: str
    snapshot_id: Optional[str]
    left_snapshot_id: str
    right_snapshot_id: str
    same_output: bool
    left: Dict[str, Any]
    right: Dict[str, Any]


@dataclass(frozen=True)
class RewardIntegrityComparison(Serializable):
    left_audit_id: str
    right_audit_id: str
    pairing: str
    pairing_reason: str
    shared_snapshot_count: int
    shared_record_count: int
    metric_deltas: Dict[str, Optional[float]]
    unmatched_left: int
    unmatched_right: int
    pairs: List[RewardIntegrityComparisonPair]
    pair_total: int
    limit: int
    offset: int


@dataclass(frozen=True)
class ResolvedRewardBinding(Serializable):
    reward_system_revision: RewardSystemRevision
    protocol_revision: RewardAuditProtocolRevision
    integrity_profile_revision: RewardIntegrityProfileRevision
    gating_eligible: bool
    blockers: List[str]


__all__ = [
    "Page",
    "ResolvedRewardBinding",
    "RewardAuditProtocol",
    "RewardAuditProtocolRevision",
    "RewardIntegrityAudit",
    "RewardIntegrityBinding",
    "RewardIntegrityComparison",
    "RewardIntegrityComparisonPair",
    "RewardIntegrityDecision",
    "RewardIntegrityMetric",
    "RewardIntegrityObservation",
    "RewardIntegrityProfile",
    "RewardIntegrityProfileRevision",
    "RewardIntegritySample",
    "RewardSystem",
    "RewardSystemAuditor",
    "RewardSystemRevision",
    "TrainingRecordRef",
    "TrainingSignalShard",
    "TrainingSignalSnapshot",
]
