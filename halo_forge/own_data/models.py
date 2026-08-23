"""Public, JSON-compatible contracts for the guided own-data workflow."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class RemediationAction:
    """One direct, transport-neutral fix for a readiness finding."""

    id: str
    label: str
    action: str
    description: str
    target: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    destructive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioAdviceRequest:
    """Structured inputs for the backend-driven own-data scenario advisor."""

    goal: str = ""
    modality: Optional[str] = None
    source_fields: tuple[str, ...] = ()
    source_layout: Optional[str] = None
    sample_values: Dict[str, Any] = field(default_factory=dict)
    include_unavailable: bool = False

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "ScenarioAdviceRequest":
        fields = value.get("source_fields") or value.get("fields") or ()
        if isinstance(fields, str):
            fields = tuple(part.strip() for part in fields.split(",") if part.strip())
        return cls(
            goal=str(value.get("goal") or value.get("description") or "").strip(),
            modality=(
                str(value.get("modality")).strip().lower()
                if value.get("modality")
                else None
            ),
            source_fields=tuple(str(item) for item in fields),
            source_layout=(
                str(value.get("source_layout")).strip().lower()
                if value.get("source_layout")
                else None
            ),
            sample_values=copy.deepcopy(dict(value.get("sample_values") or {})),
            include_unavailable=bool(value.get("include_unavailable", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["source_fields"] = list(self.source_fields)
        return value


@dataclass(frozen=True)
class ScenarioAdviceResult:
    registry_revision: str
    recommendations: tuple[Dict[str, Any], ...]
    questions: tuple[Dict[str, Any], ...] = ()
    unavailable: tuple[Dict[str, Any], ...] = ()
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_revision": self.registry_revision,
            "recommendations": [copy.deepcopy(item) for item in self.recommendations],
            "questions": [copy.deepcopy(item) for item in self.questions],
            "unavailable": [copy.deepcopy(item) for item in self.unavailable],
            "explanation": self.explanation,
            "requires_confirmation": True,
        }


@dataclass(frozen=True)
class GuidedExampleDescriptor:
    id: str
    scenario_id: str
    scenario_revision_id: str
    label: str
    description: str
    expected_source_shape: str
    expected_outcome: str
    hardware_guidance: str
    fixture_format: str
    fixture_filename: str
    record_count: int
    modality: str
    trainer_modes: tuple[str, ...]
    documentation_anchor: str

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["trainer_modes"] = list(self.trainer_modes)
        return value


@dataclass(frozen=True)
class SemanticRecordPreview:
    """A modality-aware presentation of one source/canonical record pair."""

    kind: str
    ordinal: int
    title: str
    summary: str
    source: Dict[str, Any]
    canonical: Dict[str, Any]
    presentation: Dict[str, Any]
    issues: tuple[Dict[str, Any], ...] = ()
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["issues"] = [copy.deepcopy(item) for item in self.issues]
        return value


@dataclass(frozen=True)
class DatasetReadinessReport:
    """Rich, action-oriented readiness for an inspection or immutable version."""

    scope: str
    subject_id: str
    ready: bool
    scenario_revision_id: Optional[str]
    sampled: bool
    summary: Dict[str, Any] = field(default_factory=dict)
    blockers: tuple[Dict[str, Any], ...] = ()
    warnings: tuple[Dict[str, Any], ...] = ()
    actions: tuple[RemediationAction, ...] = ()
    rejected_examples: tuple[Dict[str, Any], ...] = ()
    distributions: Dict[str, Any] = field(default_factory=dict)
    split_balance: Dict[str, Any] = field(default_factory=dict)
    media: Dict[str, Any] = field(default_factory=dict)
    extraction: Dict[str, Any] = field(default_factory=dict)
    minimum_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["actions"] = [item.to_dict() for item in self.actions]
        for key in ("blockers", "warnings", "rejected_examples"):
            value[key] = [copy.deepcopy(item) for item in value[key]]
        return value


@dataclass(frozen=True)
class CorpusProfile:
    document_count: int
    character_count: int
    paragraph_count: int
    byte_count: int
    language_hints: Dict[str, int] = field(default_factory=dict)
    length_distribution: Dict[str, Any] = field(default_factory=dict)
    duplicate_documents: int = 0
    quarantined_documents: int = 0
    extraction_failures: int = 0
    source_types: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusPackingPlan:
    tokenizer_id: str
    tokenizer_revision: Optional[str]
    tokenizer_hash: str
    max_sequence_length: int
    separator: str
    packing: str
    budget_mode: str
    target_tokens: Optional[int]
    corpus_passes: Optional[float]
    train_tokens: int
    validation_tokens: int
    train_blocks: int
    validation_blocks: int
    padding_tokens: int
    utilization: float
    estimated_steps: int
    effective_batch_size: int
    artifact_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusTrainingConfig:
    dataset_version_id: str
    model: str
    adaptation: str
    max_sequence_length: int
    packing: str
    budget_mode: str
    target_tokens: Optional[int] = None
    corpus_passes: Optional[float] = None
    effective_batch_size: int = 1
    learning_rate: Optional[float] = None
    seed: int = 42
    output: Optional[str] = None

    def __post_init__(self) -> None:
        if self.adaptation not in {"lora", "full"}:
            raise ValueError("adaptation must be explicitly set to 'lora' or 'full'")
        if self.budget_mode not in {"tokens", "passes"}:
            raise ValueError("budget_mode must be 'tokens' or 'passes'")
        if self.budget_mode == "tokens" and not self.target_tokens:
            raise ValueError("target_tokens is required for token-budget training")
        if self.budget_mode == "passes" and not self.corpus_passes:
            raise ValueError("corpus_passes is required for pass-budget training")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InterfaceCapabilityDescriptor:
    id: str
    kind: str
    label: str
    status: str
    available: bool
    reason: Optional[str] = None
    requirements: tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["requirements"] = list(self.requirements)
        # Frequently filtered dimensions are repeated at the top level for
        # simple clients while the complete descriptor remains in metadata.
        for key in (
            "execution_surface",
            "modality",
            "canonical_shape",
            "training_method",
            "trainer_mode",
            "backend",
            "backends",
            "model_family",
            "model_families",
            "source_format",
        ):
            if key in self.metadata:
                value[key] = copy.deepcopy(self.metadata[key])
        return value


@dataclass(frozen=True)
class TrainingScenarioExample:
    id: str
    name: str
    description: str
    format: str
    filename: str
    records: tuple[Dict[str, Any], ...]

    def to_dict(
        self, *, include_records: bool = True, scenario_revision_id: Optional[str] = None
    ) -> Dict[str, Any]:
        value = asdict(self)
        value["label"] = value.pop("name")
        if scenario_revision_id:
            value["scenario_revision_id"] = scenario_revision_id
        value["records"] = (
            [copy.deepcopy(record) for record in self.records] if include_records else []
        )
        return value


@dataclass(frozen=True)
class TrainingScenarioDescriptor:
    id: str
    revision_id: str
    version: int
    label: str
    description: str
    modality: str
    canonical_schema: str
    task: str
    available: bool
    unavailable_reason: Optional[str]
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...]
    field_aliases: Dict[str, tuple[str, ...]]
    safe_constants: Dict[str, Any]
    source_layouts: tuple[str, ...]
    trainer_modes: tuple[str, ...]
    model_families: tuple[str, ...]
    default_recipe: Dict[str, Any]
    proof_budget: Dict[str, Any]
    common_failures: tuple[str, ...]
    documentation_anchor: str
    examples: tuple[TrainingScenarioExample, ...]
    detection_hints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_examples: bool = False) -> Dict[str, Any]:
        value = asdict(self)
        value["revision"] = value["version"]
        value["canonical_shape"] = value["canonical_schema"]
        value["task_type"] = value["task"]
        value["accepted_aliases"] = {
            target: list(aliases) for target, aliases in self.field_aliases.items()
        }
        value["compatible_trainers"] = [
            {
                "adapter_id": self.canonical_schema,
                "adapter_version": "1",
                "trainer_mode": trainer,
                "compatible": self.available,
                "reason": None if self.available else self.unavailable_reason,
                "required_schema": self.canonical_schema,
            }
            for trainer in self.trainer_modes
        ]
        value["proof_run"] = copy.deepcopy(self.proof_budget)
        value["verified"] = bool(self.available)
        value["example_count"] = len(self.examples)
        for key in (
            "required_fields",
            "optional_fields",
            "source_layouts",
            "trainer_modes",
            "model_families",
            "common_failures",
        ):
            value[key] = list(value[key])
        value["field_aliases"] = {
            target: list(aliases) for target, aliases in self.field_aliases.items()
        }
        value["examples"] = [
            item.to_dict(
                include_records=include_examples,
                scenario_revision_id=self.revision_id,
            )
            for item in self.examples
        ]
        return value


@dataclass(frozen=True)
class DatasetImportFile:
    """Public, storage-neutral view of one resumable import file."""

    id: str
    import_id: str
    relative_path: str
    size_bytes: int
    uploaded_bytes: int
    status: str
    content_hash: Optional[str] = None
    media_type: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetImportSession:
    """Public view of a referenced, uploaded, example, or Hugging Face import."""

    id: str
    status: str
    source_kind: str
    display_name: Optional[str] = None
    source_uri: Optional[str] = None
    source_config: Optional[str] = None
    source_split: Optional[str] = None
    source_revision: Optional[str] = None
    resolved_revision: Optional[str] = None
    fingerprint: Optional[str] = None
    scenario_revision_id: Optional[str] = None
    files: tuple[DatasetImportFile, ...] = ()
    total_files: int = 0
    total_bytes: int = 0
    uploaded_bytes: int = 0
    inspection_id: Optional[str] = None
    work_item_id: Optional[str] = None
    published_dataset_id: Optional[str] = None
    published_source_id: Optional[str] = None
    disk_forecast: Dict[str, Any] = field(default_factory=dict)
    readiness: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["files"] = [item.to_dict() for item in self.files]
        return value


@dataclass(frozen=True)
class DatasetSourceInspection:
    """Immutable public view of one completed or in-progress source inspection."""

    id: str
    import_id: Optional[str]
    status: str
    source_fingerprint: str
    import_adapter_version: str
    scenario_registry_revision: str
    scenario_revision_id: Optional[str] = None
    row_count: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    sample_count: int = 0
    size_bytes: int = 0
    fields: tuple[Dict[str, Any], ...] = ()
    preview_records: tuple[Dict[str, Any], ...] = ()
    schema_candidates: tuple[Dict[str, Any], ...] = ()
    parse_errors: tuple[Dict[str, Any], ...] = ()
    media_summary: Dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    work_item_id: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        for key in ("fields", "preview_records", "schema_candidates", "parse_errors"):
            value[key] = [copy.deepcopy(item) for item in value[key]]
        value["warnings"] = list(self.warnings)
        return value


@dataclass(frozen=True)
class SchemaCandidate:
    scenario_id: str
    scenario_revision_id: str
    label: str
    canonical_schema: str
    confidence: str
    confidence_score: float
    required_coverage: float
    required_field_coverage: Dict[str, float]
    safe_transform_count: int
    suggested_mapping: Dict[str, Dict[str, Any]]
    reasons: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["score"] = value["confidence_score"]
        value["coverage"] = value["required_coverage"]
        value["required_coverage"] = copy.deepcopy(value["required_field_coverage"])
        value.pop("required_field_coverage", None)
        value["safe_transforms"] = list(value.get("reasons") or [])
        value["missing_fields"] = list(value.get("blockers") or [])
        value["reasons"] = list(self.reasons)
        value["blockers"] = list(self.blockers)
        return value


@dataclass(frozen=True)
class FieldMappingExpression:
    """A safe, typed mapping expression; no user code is evaluated."""

    kind: str
    source: Optional[str] = None
    path: Optional[str] = None
    value: Any = None
    sources: tuple[str, ...] = ()
    separator: str = "\n"
    role_map: Dict[str, str] = field(default_factory=dict)
    role_field: str = "role"
    content_field: str = "content"
    media_root: Optional[str] = None

    @classmethod
    def from_value(cls, value: Any) -> "FieldMappingExpression":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(kind="direct", source=value)
        if not isinstance(value, Mapping):
            raise ValueError("mapping expression must be a source path or object")
        kind = str(value.get("kind") or value.get("type") or "direct").strip().lower()
        kind = {"role_normalize": "conversation", "media_path": "media_root"}.get(kind, kind)
        sources = value.get("sources") or ()
        if isinstance(sources, str):
            sources = (sources,)
        return cls(
            kind=kind,
            source=(str(value["source"]) if value.get("source") is not None else None),
            path=(str(value["path"]) if value.get("path") is not None else None),
            value=copy.deepcopy(value.get("value")),
            sources=tuple(str(item) for item in sources),
            separator=str(value.get("separator", "\n")),
            role_map={str(k): str(v) for k, v in dict(value.get("role_map") or {}).items()},
            role_field=str(value.get("role_field", "role")),
            content_field=str(value.get("content_field", "content")),
            media_root=(
                str(value.get("root", value.get("media_root")))
                if value.get("root", value.get("media_root")) is not None
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.kind == "direct":
            return {"kind": "direct", "source": self.source}
        if self.kind == "constant":
            return {"kind": "constant", "value": copy.deepcopy(self.value)}
        if self.kind == "concat":
            return {
                "kind": "concat",
                "sources": list(self.sources),
                "separator": self.separator,
            }
        if self.kind == "nested_path":
            return {"kind": "nested_path", "source": self.source, "path": self.path}
        if self.kind == "conversation":
            return {
                "kind": "conversation",
                "source": self.source,
                "role_field": self.role_field,
                "content_field": self.content_field,
                "role_map": copy.deepcopy(self.role_map),
            }
        if self.kind == "media_root":
            return {"kind": "media_root", "source": self.source, "root": self.media_root}
        return {"kind": self.kind, "source": self.source}


@dataclass(frozen=True)
class FieldMappingPlan:
    scenario_revision_id: str
    mappings: Dict[str, FieldMappingExpression]
    confirmed: bool = False
    version: int = 2

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "FieldMappingPlan":
        raw = value.get("mappings") or value.get("mapping") or {}
        if not isinstance(raw, Mapping):
            raise ValueError("mapping plan mappings must be an object")
        return cls(
            scenario_revision_id=str(value.get("scenario_revision_id") or "").strip(),
            mappings={str(k): FieldMappingExpression.from_value(v) for k, v in raw.items()},
            confirmed=bool(value.get("confirmed", False)),
            version=int(value.get("version", 2)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "scenario_revision_id": self.scenario_revision_id,
            "confirmed": self.confirmed,
            "mappings": {key: value.to_dict() for key, value in self.mappings.items()},
        }


@dataclass(frozen=True)
class MappingPreview:
    inspection_id: str
    scenario_revision_id: str
    mapping_plan: Dict[str, Any]
    records: tuple[Dict[str, Any], ...]
    valid_count: int
    invalid_count: int
    field_coverage: Dict[str, float]
    errors: tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        items = []
        for record in self.records:
            issues = [
                {"code": "mapping_invalid", "message": str(message), "severity": "error"}
                for message in record.get("errors", [])
            ]
            items.append(
                {
                    "ordinal": record.get("index"),
                    "source": copy.deepcopy(record.get("source") or {}),
                    "canonical": copy.deepcopy(record.get("canonical") or {}),
                    "issues": issues,
                }
            )
        return {
            "inspection_id": self.inspection_id,
            "scenario_revision_id": self.scenario_revision_id,
            "mapping_plan": copy.deepcopy(self.mapping_plan),
            "items": items,
            "total_sampled": self.valid_count + self.invalid_count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "field_coverage": copy.deepcopy(self.field_coverage),
            "ready": self.invalid_count == 0 and self.valid_count > 0,
            "warnings": [],
        }


@dataclass(frozen=True)
class DatasetPreparationPlan:
    inspection_id: str
    scenario_revision_id: str
    mapping_plan: Dict[str, Any]
    recipe: Dict[str, Any]
    estimated_input_records: int
    estimated_accepted_records: int
    estimated_quarantined_records: int
    estimates_sampled: bool
    warnings: tuple[str, ...] = ()
    split_policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inspection_id": self.inspection_id,
            "scenario_revision_id": self.scenario_revision_id,
            "mapping_plan": copy.deepcopy(self.mapping_plan),
            "recipe": copy.deepcopy(self.recipe),
            "sampled": self.estimates_sampled,
            "estimates": {
                "accepted": self.estimated_accepted_records,
                "quarantined": self.estimated_quarantined_records,
                "duplicates": None,
            },
            "estimated_input_records": self.estimated_input_records,
            "warnings": list(self.warnings),
            "split_policy": copy.deepcopy(self.split_policy),
        }


@dataclass(frozen=True)
class DatasetReadiness:
    ready: bool
    scenario_revision_id: Optional[str]
    blockers: tuple[Dict[str, Any], ...] = ()
    warnings: tuple[Dict[str, Any], ...] = ()
    actions: tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        for key in ("blockers", "warnings", "actions"):
            value[key] = [copy.deepcopy(item) for item in value[key]]
        return value


def bounded_items(items: Sequence[Mapping[str, Any]], *, limit: int, offset: int) -> Dict[str, Any]:
    total = len(items)
    start = max(0, int(offset))
    size = max(1, int(limit))
    return {
        "items": [copy.deepcopy(dict(item)) for item in items[start : start + size]],
        "total": total,
        "limit": size,
        "offset": start,
    }


__all__ = [
    "CorpusPackingPlan",
    "CorpusProfile",
    "CorpusTrainingConfig",
    "DatasetImportFile",
    "DatasetImportSession",
    "DatasetPreparationPlan",
    "DatasetReadiness",
    "DatasetReadinessReport",
    "DatasetSourceInspection",
    "FieldMappingExpression",
    "FieldMappingPlan",
    "GuidedExampleDescriptor",
    "InterfaceCapabilityDescriptor",
    "MappingPreview",
    "RemediationAction",
    "ScenarioAdviceRequest",
    "ScenarioAdviceResult",
    "SchemaCandidate",
    "SemanticRecordPreview",
    "TrainingScenarioDescriptor",
    "TrainingScenarioExample",
    "bounded_items",
]
