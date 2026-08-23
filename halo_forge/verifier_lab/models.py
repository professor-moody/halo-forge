"""Transport-neutral public models for Verifier Reliability and Reward Studio."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class Serializable:
    """Small common surface used by the API, CLI, and scheduler."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifierCapabilityDescriptor(Serializable):
    id: str
    display_name: str
    family: str
    modalities: List[str]
    task_types: List[str]
    reliability_adapter_id: str
    reliability_adapter_version: str
    fingerprintable: bool = True
    supports_seed: Optional[bool] = None
    supports_batch_consistency: bool = False
    runtime_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierRewardContract(Serializable):
    minimum: float = 0.0
    maximum: float = 1.0
    direction: str = "maximize"
    threshold: Optional[float] = 0.5
    tie_policy: str = "error"
    error_behavior: str = "fail_closed"
    probability_semantics: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("Reward bounds must be finite")
        if self.maximum <= self.minimum:
            raise ValueError("Reward maximum must be greater than minimum")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("Reward direction must be 'maximize' or 'minimize'")
        if self.threshold is not None and (
            not math.isfinite(self.threshold)
            or self.threshold < self.minimum
            or self.threshold > self.maximum
        ):
            raise ValueError("Reward threshold must fall within the declared range")
        if self.probability_semantics and (self.minimum != 0.0 or self.maximum != 1.0):
            raise ValueError("Probability semantics require the [0, 1] reward range")

    @classmethod
    def from_value(cls, value: Any) -> "VerifierRewardContract":
        if isinstance(value, cls):
            return value
        raw = dict(value or {})
        return cls(
            minimum=float(raw.get("minimum", raw.get("min", 0.0))),
            maximum=float(raw.get("maximum", raw.get("max", 1.0))),
            direction=str(raw.get("direction", "maximize")),
            threshold=(
                None if raw.get("threshold", 0.5) is None else float(raw.get("threshold", 0.5))
            ),
            tie_policy=str(raw.get("tie_policy", "error")),
            error_behavior=str(raw.get("error_behavior", "fail_closed")),
            probability_semantics=bool(raw.get("probability_semantics", False)),
        )


@dataclass(frozen=True)
class VerifierObservation(Serializable):
    """Normalized result of any existing verifier implementation."""

    reward: Optional[float]
    passed: Optional[bool]
    parsed_value: Any = None
    raw_output: Any = None
    details: Dict[str, Any] = field(default_factory=dict)
    component_trace: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    runtime_identity: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise ValueError("Verifier rewards must be finite")
        if self.latency_ms is not None and (
            not math.isfinite(float(self.latency_ms)) or self.latency_ms < 0
        ):
            raise ValueError("Verifier latency must be finite and non-negative")


@dataclass(frozen=True)
class VerifierProfile(Serializable):
    id: str
    name: str
    description: Optional[str]
    latest_revision_id: Optional[str]
    archived: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class VerifierRevisionComponent(Serializable):
    revision_id: str
    ordinal: int
    child_revision_id: str
    weight: float = 1.0
    veto: bool = False
    required: bool = True
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierProfileRevision(Serializable):
    id: str
    profile_id: str
    revision_number: int
    content_hash: str
    family: str
    reliability_adapter_id: str
    reliability_adapter_version: str
    implementation_kind: str
    implementation_ref: str
    implementation_fingerprint: Optional[str]
    qualifiable: bool
    qualification_blockers: List[str]
    modality: str
    task_type: str
    input_mapping: Dict[str, Any]
    reward_contract: VerifierRewardContract
    definition: Dict[str, Any]
    sanitized_configuration_hash: str
    runtime_contract: Dict[str, Any]
    runtime_contract_hash: str
    created_at: str
    components: List[VerifierRevisionComponent] = field(default_factory=list)


@dataclass(frozen=True)
class VerifierCalibrationProtocol(Serializable):
    id: str
    name: str
    description: Optional[str]
    latest_revision_id: Optional[str]
    archived: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class VerifierCalibrationProtocolRevision(Serializable):
    id: str
    protocol_id: str
    revision_number: int
    content_hash: str
    definition: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class VerifierQualificationProfile(Serializable):
    id: str
    name: str
    description: Optional[str]
    latest_revision_id: Optional[str]
    archived: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class VerifierQualificationProfileRevision(Serializable):
    id: str
    profile_id: str
    revision_number: int
    content_hash: str
    template_kind: str
    promotable: bool
    requirements: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class VerifierCalibration(Serializable):
    id: str
    verifier_revision_id: str
    protocol_revision_id: str
    qualification_profile_revision_id: str
    source_kind: str
    source_revision_id: str
    source_hash: str
    source_purpose: str
    status: str
    stage: str
    processed_records: int
    total_records: Optional[int]
    sample_count: int
    request: Dict[str, Any]
    partition: Dict[str, Any]
    runtime_identity: Dict[str, Any]
    runtime_identity_hash: str
    protocol_hash: str
    qualification_hash: str
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
class VerifierCalibrationSample(Serializable):
    calibration_id: str
    ordinal: int
    record_id: str
    record_hash: str
    group_id: str
    partition: str
    repeat_index: int
    orientation: str
    probe_kind: str
    seed: Optional[int]
    reference: Dict[str, Any]
    observation: VerifierObservation
    metadata: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class VerifierCalibrationMetric(Serializable):
    calibration_id: str
    name: str
    partition: str
    subgroup: str
    value: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    direction: Optional[str]
    available: bool
    missing_reason: Optional[str]
    record_count: int
    metadata: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class VerifierQualificationDecision(Serializable):
    id: str
    calibration_id: str
    qualification_profile_revision_id: str
    scope: str
    decision: str
    runtime_state: str
    reasons: List[str]
    evidence: Dict[str, Any]
    override: bool
    override_note: Optional[str]
    supersedes_decision_id: Optional[str]
    created_at: str


@dataclass(frozen=True)
class VerifierAlias(Serializable):
    profile_id: str
    alias: str
    revision_id: str
    updated_at: str


@dataclass(frozen=True)
class VerifierAliasEvent(Serializable):
    id: str
    profile_id: str
    alias: str
    previous_revision_id: Optional[str]
    revision_id: str
    qualification_decision_id: Optional[str]
    override: bool
    note: Optional[str]
    created_at: str


@dataclass(frozen=True)
class ResolvedVerifierBinding(Serializable):
    id: str
    verifier_revision_id: str
    domain_kind: str
    domain_id: str
    role: str
    qualification_decision_id: Optional[str]
    legacy_unqualified: bool
    development_exposed: bool
    binding_hash: str
    context: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class VerifierCalibrationComparison(Serializable):
    base_calibration_id: str
    candidate_calibration_id: str
    compatible: bool
    compatibility_reasons: List[str]
    metric_deltas: List[Dict[str, Any]] = field(default_factory=list)
    decision_delta: Optional[Dict[str, Any]] = None


__all__ = [
    "ResolvedVerifierBinding",
    "VerifierAlias",
    "VerifierAliasEvent",
    "VerifierCalibration",
    "VerifierCalibrationComparison",
    "VerifierCalibrationMetric",
    "VerifierCalibrationProtocol",
    "VerifierCalibrationProtocolRevision",
    "VerifierCalibrationSample",
    "VerifierCapabilityDescriptor",
    "VerifierObservation",
    "VerifierProfile",
    "VerifierProfileRevision",
    "VerifierQualificationDecision",
    "VerifierQualificationProfile",
    "VerifierQualificationProfileRevision",
    "VerifierRevisionComponent",
    "VerifierRewardContract",
]
