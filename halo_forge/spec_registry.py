"""Small public descriptors for structured Dataset and Evaluation editors.

The registry describes existing payloads; it is not a second configuration
format.  Dashboard forms emit the same mappings accepted by Dataset Lab and the
evaluation API, while Advanced mode can continue to edit those mappings
directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SpecFieldDescriptor:
    name: str
    label: str
    value_type: str
    required: bool = False
    default: Any = None
    options: tuple[Any, ...] = ()
    description: str = ""
    placeholder: str = ""
    visible_when: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["options"] = list(self.options)
        return value


@dataclass(frozen=True)
class SpecDescriptor:
    kind: str
    descriptor_id: str
    version: str
    label: str
    description: str
    fields: tuple[SpecFieldDescriptor, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.descriptor_id,
            "version": self.version,
            "label": self.label,
            "description": self.description,
            "fields": [value.to_dict() for value in self.fields],
        }


def _field(
    name: str,
    label: str,
    value_type: str,
    *,
    required: bool = False,
    default: Any = None,
    options: Sequence[Any] = (),
    description: str = "",
    placeholder: str = "",
    visible_when: Optional[Mapping[str, Any]] = None,
) -> SpecFieldDescriptor:
    return SpecFieldDescriptor(
        name=name,
        label=label,
        value_type=value_type,
        required=required,
        default=default,
        options=tuple(options),
        description=description,
        placeholder=placeholder,
        visible_when=dict(visible_when or {}),
    )


_RECIPE_DESCRIPTORS = (
    SpecDescriptor(
        "dataset_recipe_step",
        "map",
        "1",
        "Map fields",
        "Map source columns into a canonical training schema.",
        (
            _field(
                "schema",
                "Canonical schema",
                "select",
                required=True,
                options=("sft", "chat", "preference", "prompt", "tool", "vlm", "audio"),
            ),
            _field(
                "fields",
                "Field mapping",
                "field_mapping",
                required=True,
                description="Canonical target to source-column mapping.",
            ),
        ),
    ),
    SpecDescriptor(
        "dataset_recipe_step",
        "validate",
        "1",
        "Validate records",
        "Validate canonical records and choose how invalid rows are retained.",
        (
            _field(
                "on_error",
                "Invalid records",
                "select",
                default="reject",
                options=("reject", "quarantine"),
            ),
        ),
    ),
    SpecDescriptor(
        "dataset_recipe_step",
        "filter",
        "1",
        "Filter records",
        "Apply a safe field-based predicate.",
        (
            _field("field", "Field", "field_path", required=True),
            _field(
                "op",
                "Condition",
                "select",
                required=True,
                options=(
                    "eq",
                    "ne",
                    "gt",
                    "gte",
                    "lt",
                    "lte",
                    "in",
                    "not_in",
                    "contains",
                    "exists",
                    "regex",
                ),
            ),
            _field("value", "Value", "json_value"),
        ),
    ),
    SpecDescriptor(
        "dataset_recipe_step",
        "dedup",
        "1",
        "Remove duplicates",
        "Deduplicate text, image, or audio records using a supported method.",
        (
            _field(
                "method",
                "Method",
                "select",
                default="exact",
                options=(
                    "exact",
                    "fuzzy",
                    "semantic",
                    "perceptual",
                    "image_exact",
                    "audio_exact",
                ),
            ),
            _field(
                "threshold",
                "Similarity threshold",
                "number",
                default=0.85,
                visible_when={"method": ["fuzzy", "semantic", "perceptual"]},
            ),
            _field("field", "Text or asset field", "field_path"),
        ),
    ),
    SpecDescriptor(
        "dataset_recipe_step",
        "score",
        "1",
        "Score quality",
        "Attach heuristic, verifier, or judge quality evidence.",
        (
            _field(
                "method",
                "Scorer",
                "select",
                default="heuristic",
                options=("heuristic", "verifier", "judge"),
            ),
            _field("threshold", "Minimum score", "number"),
            _field(
                "verifier",
                "Verifier",
                "verifier_picker",
                visible_when={"method": "verifier"},
            ),
            _field(
                "provider",
                "Judge provider",
                "provider_picker",
                visible_when={"method": "judge"},
            ),
        ),
    ),
    SpecDescriptor(
        "dataset_recipe_step",
        "split",
        "1",
        "Create splits",
        "Create deterministic train, validation, test, or canary bindings.",
        (
            _field(
                "method",
                "Method",
                "select",
                default="random",
                options=("random", "stratified", "grouped", "time"),
            ),
            _field("ratios", "Split ratios", "split_ratios", required=True),
            _field(
                "field",
                "Stratification field",
                "field_path",
                visible_when={"method": "stratified"},
            ),
            _field(
                "group_field",
                "Group field",
                "field_path",
                visible_when={"method": "grouped"},
            ),
            _field(
                "time_field",
                "Time field",
                "field_path",
                visible_when={"method": "time"},
            ),
        ),
    ),
    SpecDescriptor(
        "dataset_recipe_step",
        "failure_mining",
        "1",
        "Import reviewed failures",
        "Import a saved failure selection or interaction file.",
        (
            _field("source", "Failure source", "source_picker", required=True),
            _field("selector", "Selection", "failure_selector"),
        ),
    ),
)


_BENCHMARK_DESCRIPTORS = (
    SpecDescriptor(
        "benchmark_suite_item",
        "item",
        "1",
        "Benchmark item",
        "One ordered example evaluated by a registered adapter.",
        (
            _field("id", "Item ID", "text", required=True),
            _field(
                "adapter_id",
                "Evaluator",
                "select",
                default="dataset",
                options=(
                    "dataset",
                    "lm-eval",
                    "benchmark",
                    "verifier",
                    "code",
                    "reasoning",
                    "tool",
                    "vlm",
                    "audio",
                    "performance",
                ),
            ),
            _field("record_id", "Record ID", "text"),
            _field("input", "Input", "multimodal_value"),
            _field("expected", "Expected output", "multimodal_value"),
            _field("task", "Task", "task_picker"),
            _field("verifier", "Verifier", "verifier_picker"),
            _field("weight", "Weight", "number", default=1.0),
            _field("metadata", "Metadata", "key_value"),
        ),
    ),
    SpecDescriptor(
        "benchmark_suite_settings",
        "generation",
        "1",
        "Generation settings",
        "Generation defaults pinned by a benchmark-suite revision.",
        (
            _field("temperature", "Temperature", "number", default=0.0),
            _field("top_p", "Top P", "number", default=1.0),
            _field("max_tokens", "Maximum output tokens", "integer", default=512),
            _field("seed", "Seed", "integer", default=42),
            _field("stop", "Stop sequences", "string_list"),
        ),
    ),
)


def _acquisition_descriptor(
    descriptor_id: str,
    label: str,
    description: str,
    *extra_fields: SpecFieldDescriptor,
) -> SpecDescriptor:
    return SpecDescriptor(
        "acquisition_strategy",
        descriptor_id,
        "1",
        label,
        description,
        (
            _field(
                "quota",
                "Maximum records",
                "integer",
                description="Leave empty to use every eligible match.",
            ),
            *extra_fields,
        ),
    )


_ACQUISITION_DESCRIPTORS = (
    _acquisition_descriptor(
        "explicit", "Explicit selection", "Use the records selected in the source preview.",
        _field("record_ids", "Selected records", "record_picker"),
    ),
    _acquisition_descriptor(
        "candidate_failure", "Candidate failures", "Select valid development examples the candidate failed."
    ),
    _acquisition_descriptor(
        "regression", "Regressions", "Select examples where the candidate failed after the base passed."
    ),
    _acquisition_descriptor(
        "improvement", "Improvements", "Select examples where the candidate improved over the base."
    ),
    _acquisition_descriptor(
        "verifier_disagreement", "Verifier disagreement", "Select examples with conflicting verifier evidence."
    ),
    _acquisition_descriptor(
        "low_score", "Low score", "Select the weakest direction-aware example scores.",
        _field(
            "direction",
            "Metric direction",
            "select",
            default="maximize",
            options=("maximize", "minimize"),
        ),
    ),
    _acquisition_descriptor(
        "low_margin", "Low margin", "Select examples closest to an available decision boundary."
    ),
    _acquisition_descriptor(
        "coverage_gap", "Coverage gaps", "Prioritize underrepresented tasks or categories."
    ),
    _acquisition_descriptor(
        "diversity", "Diverse sample", "Select a deterministic farthest-first embedding sample.",
        _field(
            "embedding_revision",
            "Embedding model revision",
            "model_revision_picker",
            required=True,
            description=(
                "Pinned text, image, or audio embedding identity. Missing vectors are "
                "generated as durable heavy work; no synthetic evidence is substituted."
            ),
            placeholder="image:organization/model@commit",
        ),
    ),
    _acquisition_descriptor(
        "random", "Seeded random", "Provide an auditable random baseline using the batch seed."
    ),
)


def _annotation_descriptor(
    descriptor_id: str,
    label: str,
    description: str,
    *fields: SpecFieldDescriptor,
) -> SpecDescriptor:
    return SpecDescriptor(
        "annotation_task",
        descriptor_id,
        "1",
        label,
        description,
        (
            _field(
                "modality",
                "Modality",
                "select",
                required=True,
                options=("text", "preference", "tool", "vlm", "audio"),
            ),
            *fields,
        ),
    )


_ANNOTATION_DESCRIPTORS = (
    _annotation_descriptor("binary", "Accept or reject", "Record a reviewed binary decision."),
    _annotation_descriptor(
        "categorical",
        "Category",
        "Choose exactly one reviewed label.",
        _field("labels", "Labels", "string_list", required=True),
    ),
    _annotation_descriptor(
        "multi_label",
        "Multiple labels",
        "Choose any number of compatible reviewed labels.",
        _field("labels", "Allowed labels", "string_list"),
    ),
    _annotation_descriptor(
        "scalar",
        "Scalar score",
        "Record a bounded numeric judgment.",
        _field("minimum", "Minimum", "number", default=0.0),
        _field("maximum", "Maximum", "number", default=1.0),
    ),
    _annotation_descriptor(
        "text_correction", "Text correction", "Write a corrected response, answer, or transcript."
    ),
    _annotation_descriptor(
        "structured_correction", "Tool-trace correction", "Correct messages, tools, calls, or results."
    ),
    _annotation_descriptor(
        "pairwise", "Pairwise preference", "Choose between two candidate responses."
    ),
    _annotation_descriptor(
        "ranking", "Response ranking", "Order two or more candidates from strongest to weakest."
    ),
)


_REWARD_SYSTEM_DESCRIPTORS = (
    SpecDescriptor(
        "reward_system",
        "monitored",
        "1",
        "Monitored reward system",
        "Pin the optimization verifier, independent sentinel, and trainer-facing reward mapping.",
        (
            _field("modality", "Modality", "modality_picker", required=True),
            _field("task_type", "Task", "task_picker", required=True),
            _field(
                "optimizer_verifier_revision_id",
                "Training verifier",
                "qualified_verifier_picker",
                required=True,
            ),
            _field(
                "primary_sentinel_revision_id",
                "Independent sentinel",
                "qualified_verifier_picker",
                required=True,
                description="Must be runtime-current and fingerprint-disjoint from the training verifier.",
            ),
            _field(
                "diagnostic_auditor_revision_ids",
                "Diagnostic auditors",
                "qualified_verifier_list",
                description="Optional evidence-only auditors; at most three.",
            ),
            _field(
                "trainer_modes",
                "Compatible trainers",
                "trainer_mode_list",
                required=True,
            ),
            _field("reward_mapping", "Reward mapping", "reward_mapping", required=True),
        ),
    ),
)


_REWARD_PROTOCOL_DESCRIPTORS = (
    SpecDescriptor(
        "reward_audit_protocol",
        "capture",
        "1",
        "Same-output capture protocol",
        "Choose a deterministic evidence budget; the sentinel never regenerates an output.",
        (
            _field(
                "template",
                "Evidence budget",
                "select",
                required=True,
                default="balanced_256",
                options=("balanced_256", "broad_512", "exhaustive"),
            ),
            _field("seed", "Sampling seed", "integer", default=42),
            _field(
                "output_capture",
                "Output capture",
                "select",
                default="full",
                options=("full", "hash_only", "aggregate_only"),
            ),
            _field("bootstrap_resamples", "Bootstrap draws", "integer", default=10000),
        ),
    ),
)


_REWARD_INTEGRITY_DESCRIPTORS = (
    SpecDescriptor(
        "reward_integrity_profile",
        "decision",
        "1",
        "Integrity decision policy",
        "Pin pass, warn, fail, and incomplete-evidence behavior without tuning a verifier.",
        (
            _field(
                "template",
                "Decision template",
                "select",
                required=True,
                default="human_aligned_integrity",
                options=("strict_integrity", "human_aligned_integrity", "exploratory"),
            ),
            _field("minimum_pass_records", "Records required to pass", "integer", default=100),
            _field("minimum_warn_records", "Records required to warn", "integer", default=20),
            _field(
                "incomplete_action",
                "Incomplete evidence",
                "select",
                default="pause",
                options=("pause",),
            ),
        ),
    ),
    SpecDescriptor(
        "reward_integrity_binding",
        "selected_boundaries",
        "1",
        "Selected-boundary audit",
        "Bind one immutable reward system, protocol, and integrity policy to a managed run.",
        (
            _field(
                "reward_system_revision_id",
                "Reward system",
                "reward_system_picker",
                required=True,
            ),
            _field(
                "protocol_revision_id",
                "Capture protocol",
                "reward_protocol_picker",
                required=True,
            ),
            _field(
                "integrity_profile_revision_id",
                "Integrity policy",
                "reward_integrity_profile_picker",
                required=True,
            ),
            _field("boundaries", "Audit boundaries", "boundary_list", required=True),
            _field(
                "development_suite_revision_id",
                "Development quality suite",
                "development_suite_picker",
            ),
        ),
    ),
)


_REGISTRY = {
    "dataset_recipe_step": _RECIPE_DESCRIPTORS,
    "benchmark_suite_item": tuple(
        value for value in _BENCHMARK_DESCRIPTORS if value.kind == "benchmark_suite_item"
    ),
    "benchmark_suite_settings": tuple(
        value for value in _BENCHMARK_DESCRIPTORS if value.kind == "benchmark_suite_settings"
    ),
    "acquisition_strategy": _ACQUISITION_DESCRIPTORS,
    "annotation_task": _ANNOTATION_DESCRIPTORS,
    "reward_system": _REWARD_SYSTEM_DESCRIPTORS,
    "reward_audit_protocol": _REWARD_PROTOCOL_DESCRIPTORS,
    "reward_integrity_profile": tuple(
        value for value in _REWARD_INTEGRITY_DESCRIPTORS if value.kind == "reward_integrity_profile"
    ),
    "reward_integrity_binding": tuple(
        value for value in _REWARD_INTEGRITY_DESCRIPTORS if value.kind == "reward_integrity_binding"
    ),
}


def list_spec_descriptors(kind: str) -> list[SpecDescriptor]:
    normalized = str(kind or "").strip().lower().replace("-", "_")
    if normalized not in _REGISTRY:
        raise KeyError(f"unknown spec descriptor kind: {kind}")
    return list(_REGISTRY[normalized])


def get_spec_descriptor(kind: str, descriptor_id: str) -> SpecDescriptor:
    for descriptor in list_spec_descriptors(kind):
        if descriptor.descriptor_id == descriptor_id:
            return descriptor
    raise KeyError(f"unknown {kind} descriptor: {descriptor_id}")


def validate_structured_spec(
    kind: str, descriptor_id: str, payload: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate a form payload using the same domain parsers as Advanced mode."""

    descriptor = get_spec_descriptor(kind, descriptor_id)
    value = dict(payload)
    errors: list[dict[str, str]] = []
    for field_descriptor in descriptor.fields:
        if field_descriptor.required and value.get(field_descriptor.name) in (None, "", [], {}):
            errors.append(
                {"field": field_descriptor.name, "message": f"{field_descriptor.label} is required"}
            )
    if not errors and descriptor.kind == "dataset_recipe_step":
        try:
            from halo_forge.data_lab.recipe import Recipe, RecipeStep, validate_recipe

            step = RecipeStep.from_value({"kind": descriptor.descriptor_id, **value})
            validate_recipe(Recipe((step,), schema=value.get("schema")))
            value = step.to_dict()
        except Exception as exc:
            errors.append({"field": "", "message": str(exc)})
    elif not errors and descriptor.kind == "benchmark_suite_item":
        adapter_id = str(value.get("adapter_id") or "dataset")
        if adapter_id == "lm-eval" and not value.get("task"):
            errors.append({"field": "task", "message": "Task is required for lm-eval items"})
        if adapter_id == "verifier" and not value.get("verifier"):
            errors.append(
                {"field": "verifier", "message": "Verifier is required for verifier items"}
            )
    elif not errors and descriptor.kind == "acquisition_strategy":
        quota = value.get("quota")
        if quota is not None and int(quota) < 0:
            errors.append({"field": "quota", "message": "Maximum records cannot be negative"})
        if descriptor.descriptor_id == "diversity" and not value.get("embedding_revision"):
            errors.append(
                {
                    "field": "embedding_revision",
                    "message": "Diversity requires a pinned embedding model revision",
                }
            )
        value = {"kind": descriptor.descriptor_id, **value}
    elif not errors and descriptor.kind == "annotation_task":
        try:
            from halo_forge.review_lab.registry import validate_schema_definition

            modality = str(value.pop("modality"))
            definition = validate_schema_definition(
                modality,
                descriptor.descriptor_id,
                value,
            )
            value = {
                "modality": modality,
                "task_type": descriptor.descriptor_id,
                "definition": definition,
            }
        except Exception as exc:
            errors.append({"field": "", "message": str(exc)})
    elif not errors and descriptor.kind == "reward_system":
        diagnostic = list(value.get("diagnostic_auditor_revision_ids") or [])
        if len(diagnostic) > 3:
            errors.append(
                {
                    "field": "diagnostic_auditor_revision_ids",
                    "message": "At most three diagnostic auditors are supported",
                }
            )
        if value.get("optimizer_verifier_revision_id") == value.get(
            "primary_sentinel_revision_id"
        ):
            errors.append(
                {
                    "field": "primary_sentinel_revision_id",
                    "message": "The primary sentinel must differ from the training verifier",
                }
            )
    elif not errors and descriptor.kind == "reward_audit_protocol":
        if int(value.get("bootstrap_resamples", 10000)) <= 0:
            errors.append(
                {"field": "bootstrap_resamples", "message": "Bootstrap draws must be positive"}
            )
        if value.get("output_capture") != "full":
            value["gating_eligible"] = False
    elif not errors and descriptor.kind == "reward_integrity_profile":
        warn_records = int(value.get("minimum_warn_records", 20))
        pass_records = int(value.get("minimum_pass_records", 100))
        if warn_records < 1 or pass_records < warn_records:
            errors.append(
                {
                    "field": "minimum_pass_records",
                    "message": "Pass evidence must be at least the warn evidence minimum",
                }
            )
        value["promotable"] = value.get("template") != "exploratory"
    elif not errors and descriptor.kind == "reward_integrity_binding":
        boundaries = list(value.get("boundaries") or [])
        normalized: list[int | str] = []
        for boundary in boundaries:
            if str(boundary).strip().lower() == "final":
                normalized.append("final")
            elif int(boundary) <= 0:
                errors.append(
                    {"field": "boundaries", "message": "Audit boundaries must be positive"}
                )
                break
            else:
                normalized.append(int(boundary))
        value["boundaries"] = normalized
    return {"valid": not errors, "value": value, "errors": errors}


def serialized_spec_descriptors(kind: str) -> list[Dict[str, Any]]:
    return [value.to_dict() for value in list_spec_descriptors(kind)]


__all__ = [
    "SpecDescriptor",
    "SpecFieldDescriptor",
    "get_spec_descriptor",
    "list_spec_descriptors",
    "serialized_spec_descriptors",
    "validate_structured_spec",
]
