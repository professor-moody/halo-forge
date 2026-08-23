"""Transport-neutral public models for Human Feedback and Active Data Studio."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class Serializable:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnnotationSchema(Serializable):
    id: str
    name: str
    description: Optional[str]
    archived: bool
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class AnnotationSchemaRevision(Serializable):
    id: str
    schema_id: str
    revision_number: int
    content_hash: str
    modality: str
    task_type: str
    definition: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class AcquisitionStrategy(Serializable):
    kind: str = "explicit"
    quota: Optional[int] = None
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "AcquisitionStrategy":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(kind=value)
        raw = dict(value or {})
        kind = str(raw.pop("kind", raw.pop("strategy", "explicit"))).strip().lower()
        quota = raw.pop("quota", raw.pop("limit", None))
        nested = raw.pop("options", None)
        options = {**(dict(nested) if isinstance(nested, dict) else {}), **raw}
        return cls(kind=kind, quota=None if quota is None else int(quota), options=options)


@dataclass(frozen=True)
class AcquisitionSource(Serializable):
    kind: str
    ref: Optional[str] = None
    split: Optional[str] = None
    base_id: Optional[str] = None
    candidate_id: Optional[str] = None
    source_hash: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcquisitionRequest(Serializable):
    sources: List[AcquisitionSource | Dict[str, Any]] = field(default_factory=list)
    strategies: List[AcquisitionStrategy | Dict[str, Any]] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)
    seed: int = 0
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcquisitionBatch(Serializable):
    id: str
    name: str
    status: str
    stage: str
    request: Dict[str, Any]
    source_hash: str
    content_hash: str
    seed: int
    row_count: int
    processed_records: int
    total_records: Optional[int]
    work_item_id: Optional[str]
    error: Optional[str]
    eligibility: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: str
    completed_at: Optional[str]


@dataclass(frozen=True)
class AcquisitionCandidate(Serializable):
    id: str
    batch_id: str
    ordinal: int
    record_id: str
    record_hash: str
    source_kind: str
    source_ref: Optional[str]
    source_record_id: Optional[str]
    record: Dict[str, Any]
    evidence: Dict[str, Any]
    source: Dict[str, Any]
    stratum: str
    score: Optional[float]
    created_at: str


@dataclass(frozen=True)
class ReviewPolicy(Serializable):
    mode: str = "one_pass"
    blind_second_pass: bool = True
    allow_suggestions: bool = True
    require_adjudication: bool = True

    @property
    def passes(self) -> int:
        return 2 if self.mode == "two_pass" else 1

    @classmethod
    def from_value(cls, value: Any = None, *, default_mode: str = "one_pass") -> "ReviewPolicy":
        if isinstance(value, cls):
            return value
        raw = dict(value or {})
        mode = str(raw.get("mode") or default_mode).strip().lower().replace("-", "_")
        if mode in {"one", "single", "1"}:
            mode = "one_pass"
        if mode in {"two", "double", "2"}:
            mode = "two_pass"
        return cls(
            mode=mode,
            blind_second_pass=bool(raw.get("blind_second_pass", True)),
            allow_suggestions=bool(raw.get("allow_suggestions", True)),
            require_adjudication=bool(raw.get("require_adjudication", True)),
        )


@dataclass(frozen=True)
class ReviewQueue(Serializable):
    id: str
    name: str
    status: str
    acquisition_batch_id: str
    schema_revision_id: str
    policy: Dict[str, Any]
    content_hash: str
    current_pass: int
    latest_label_set_revision_id: Optional[str]
    created_at: str
    updated_at: str
    completed_at: Optional[str]


@dataclass(frozen=True)
class ReviewQueueStatistics(Serializable):
    queue_id: str
    total: int
    resolved: int
    coverage: float
    status_counts: Dict[str, int] = field(default_factory=dict)
    excluded: int = 0
    flagged: int = 0
    conflicts: int = 0
    class_balance: Dict[str, int] = field(default_factory=dict)
    two_pass_compared: int = 0
    two_pass_agreements: int = 0
    two_pass_agreement_rate: Optional[float] = None
    correction_rate: float = 0.0
    suggestion_compared: int = 0
    suggestion_agreements: int = 0
    suggestion_agreement_rate: Optional[float] = None
    unpublished_changes: bool = False
    event_stream_hash: str = ""


@dataclass(frozen=True)
class ReviewItem(Serializable):
    id: str
    queue_id: str
    candidate_id: str
    ordinal: int
    status: str
    active_event_id: Optional[str]
    projection: Dict[str, Any]
    created_at: str
    updated_at: str
    record: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None
    source: Optional[Dict[str, Any]] = None
    record_id: Optional[str] = None
    record_hash: Optional[str] = None


@dataclass(frozen=True)
class ReviewEvent(Serializable):
    id: str
    queue_id: str
    item_id: str
    event_type: str
    pass_number: int
    reviewer_key: str
    idempotency_key: str
    request_hash: str
    expected_active_event_id: Optional[str]
    payload: Dict[str, Any]
    supersedes_event_id: Optional[str]
    created_at: str


@dataclass(frozen=True)
class ReviewSuggestion(Serializable):
    id: str
    item_id: str
    pass_number: int
    provider: str
    model_revision: str
    content_hash: str
    output: Any
    provenance: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class LabelSet(Serializable):
    id: str
    queue_id: str
    name: str
    latest_revision_id: Optional[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class LabelSetRevision(Serializable):
    id: str
    label_set_id: str
    revision_number: int
    content_hash: str
    storage_path: str
    row_count: int
    excluded_count: int
    manifest: Dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class LabelSetItem(Serializable):
    revision_id: str
    ordinal: int
    review_item_id: str
    record_id: str
    record_hash: str
    annotation: Dict[str, Any]
    output_records: List[Dict[str, Any]]
    lineage: Dict[str, Any]
    excluded: bool
    exclusion_reason: Optional[str]


@dataclass(frozen=True)
class LabelSetVerification(Serializable):
    revision_id: str
    valid: bool
    checksums: Dict[str, str]
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DatasetBuildPreview(Serializable):
    label_set_revision_id: str
    dataset_id: str
    parent_version_id: Optional[str]
    build_mode: str
    target_split: str
    source_count: int
    output_count: int
    added_count: int
    removed_count: int
    replaced_count: int
    annotated_count: int
    excluded_count: int
    quarantined_count: int = 0
    split_counts: Dict[str, int] = field(default_factory=dict)
    moved_from_splits: Dict[str, int] = field(default_factory=dict)
    contamination: Dict[str, Any] = field(default_factory=dict)
    sample: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewOutputAdapterDescriptor(Serializable):
    id: str
    version: int
    modalities: List[str]
    task_types: List[str]
    build_modes: List[str]
    default_build_mode: str


__all__ = [
    "AcquisitionBatch",
    "AcquisitionCandidate",
    "AcquisitionRequest",
    "AcquisitionSource",
    "AcquisitionStrategy",
    "AnnotationSchema",
    "AnnotationSchemaRevision",
    "DatasetBuildPreview",
    "LabelSet",
    "LabelSetItem",
    "LabelSetRevision",
    "LabelSetVerification",
    "ReviewEvent",
    "ReviewItem",
    "ReviewPolicy",
    "ReviewQueue",
    "ReviewQueueStatistics",
    "ReviewOutputAdapterDescriptor",
    "ReviewSuggestion",
]
