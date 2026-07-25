"""Safe mapping-v2 evaluation and guided preparation plans."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from halo_forge.data_lab.models import get_field, validate_record

from .models import (
    DatasetPreparationPlan,
    FieldMappingExpression,
    FieldMappingPlan,
    MappingPreview,
)
from .registry import TRAINING_SCENARIOS, TrainingScenarioRegistry

_DEFAULT_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}


def _resolve_media(value: Any, media_root: Optional[str]) -> Any:
    if value is None or not media_root or not isinstance(value, str):
        return value
    path = Path(value).expanduser()
    selected_root = Path(media_root).expanduser()
    if selected_root.is_symlink():
        raise ValueError("symbolic-link media roots are not accepted")
    root = selected_root.resolve()
    candidate = path if path.is_absolute() else root / path
    try:
        relative_candidate = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("media path escapes the selected media root") from exc
    cursor = root
    for part in relative_candidate.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("symbolic-link media assets are not accepted")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("media path escapes the selected media root") from exc
    if not resolved.is_file():
        raise ValueError(f"media asset does not exist: {value}")
    return str(resolved)


def evaluate_expression(record: Mapping[str, Any], expression: FieldMappingExpression) -> Any:
    kind = expression.kind.replace("-", "_")
    if kind == "direct":
        if not expression.source:
            raise ValueError(f"{kind} mapping requires source")
        return copy.deepcopy(get_field(record, expression.source))
    if kind == "nested_path":
        if not expression.source:
            raise ValueError("nested_path mapping requires source")
        base = get_field(record, expression.source)
        if expression.path:
            if not isinstance(base, Mapping):
                return None
            return copy.deepcopy(get_field(base, expression.path))
        return copy.deepcopy(base)
    if kind == "constant":
        return copy.deepcopy(expression.value)
    if kind == "concat":
        if not expression.sources:
            raise ValueError("concat mapping requires at least one source")
        values = [get_field(record, source) for source in expression.sources]
        return expression.separator.join(str(value) for value in values if value is not None)
    if kind in {"conversation", "role_normalize"}:
        if not expression.source:
            raise ValueError("role_normalize mapping requires source")
        messages = get_field(record, expression.source)
        if not isinstance(messages, list):
            return messages
        role_map = {**_DEFAULT_ROLE_MAP, **expression.role_map}
        output = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise ValueError("conversation messages must be objects")
            raw_role = (
                str(
                    message.get(
                        expression.role_field,
                        message.get("role", message.get("from", "")),
                    )
                )
                .strip()
                .lower()
            )
            role = role_map.get(raw_role, raw_role)
            content = message.get(
                expression.content_field,
                message.get("content", message.get("value")),
            )
            normalized = {"role": role, "content": content}
            for key in ("tool_calls", "name", "tool_call_id"):
                if key in message:
                    normalized[key] = copy.deepcopy(message[key])
            output.append(normalized)
        return output
    if kind in {"media_root", "media_path"}:
        if not expression.source:
            raise ValueError("media_path mapping requires source")
        return _resolve_media(get_field(record, expression.source), expression.media_root)
    raise ValueError(f"unsupported mapping expression kind: {expression.kind}")


def apply_mapping(
    record: Mapping[str, Any],
    plan: FieldMappingPlan,
    *,
    scenario_registry: TrainingScenarioRegistry = TRAINING_SCENARIOS,
) -> Dict[str, Any]:
    scenario = scenario_registry.get(plan.scenario_revision_id)
    output: Dict[str, Any] = {}
    for target, expression in plan.mappings.items():
        value = evaluate_expression(record, expression)
        if value is not None:
            output[target] = value
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        output["metadata"] = copy.deepcopy(dict(metadata))
    missing = [
        field
        for field in scenario.required_fields
        if output.get(field) is None
        or (isinstance(output.get(field), (str, list, dict)) and not output.get(field))
    ]
    if missing:
        raise ValueError("missing required mapped field(s): " + ", ".join(missing))
    validate_record(output, scenario.canonical_schema)
    return output


def preview_mapping(
    inspection: Mapping[str, Any],
    mapping_plan: FieldMappingPlan | Mapping[str, Any],
    *,
    limit: int = 50,
    scenario_registry: TrainingScenarioRegistry = TRAINING_SCENARIOS,
) -> MappingPreview:
    plan = (
        mapping_plan
        if isinstance(mapping_plan, FieldMappingPlan)
        else FieldMappingPlan.from_value(mapping_plan)
    )
    scenario = scenario_registry.get(plan.scenario_revision_id)
    records = list(inspection.get("preview") or [])[: max(1, min(int(limit), 200))]
    previews: list[Dict[str, Any]] = []
    errors: list[Dict[str, Any]] = []
    field_hits = {field: 0 for field in scenario.required_fields + scenario.optional_fields}
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            errors.append(
                {"index": index, "code": "record_not_object", "message": "Record is not an object."}
            )
            continue
        try:
            canonical = apply_mapping(row, plan, scenario_registry=scenario_registry)
        except (TypeError, ValueError) as exc:
            errors.append({"index": index, "code": "mapping_invalid", "message": str(exc)})
            previews.append(
                {
                    "index": index,
                    "source": copy.deepcopy(dict(row)),
                    "canonical": None,
                    "valid": False,
                    "errors": [str(exc)],
                }
            )
            continue
        for field_name in field_hits:
            value = canonical.get(field_name)
            if value is not None and (not isinstance(value, (str, list, dict)) or bool(value)):
                field_hits[field_name] += 1
        previews.append(
            {
                "index": index,
                "source": copy.deepcopy(dict(row)),
                "canonical": canonical,
                "valid": True,
                "errors": [],
            }
        )
    denominator = max(1, len(records))
    return MappingPreview(
        inspection_id=str(inspection.get("id") or ""),
        scenario_revision_id=scenario.revision_id,
        mapping_plan=plan.to_dict(),
        records=tuple(previews),
        valid_count=sum(1 for item in previews if item["valid"]),
        invalid_count=len(errors),
        field_coverage={key: value / denominator for key, value in field_hits.items()},
        errors=tuple(errors),
    )


def build_preparation_plan(
    inspection: Mapping[str, Any],
    mapping_plan: FieldMappingPlan | Mapping[str, Any],
    *,
    scenario_registry: TrainingScenarioRegistry = TRAINING_SCENARIOS,
) -> DatasetPreparationPlan:
    plan = (
        mapping_plan
        if isinstance(mapping_plan, FieldMappingPlan)
        else FieldMappingPlan.from_value(mapping_plan)
    )
    scenario = scenario_registry.get(plan.scenario_revision_id)
    preview = preview_mapping(inspection, plan, limit=200, scenario_registry=scenario_registry)
    if not plan.confirmed:
        raise ValueError("mapping plan must be explicitly confirmed before preparation")
    recipe = copy.deepcopy(scenario.default_recipe)
    recipe["steps"][0]["fields"] = {
        target: expression.to_dict() for target, expression in plan.mappings.items()
    }
    total = int(inspection.get("total_records") or 0)
    sampled = max(1, preview.valid_count + preview.invalid_count)
    estimated_invalid = round(total * preview.invalid_count / sampled) if total else 0
    warnings: list[str] = []
    if preview.invalid_count:
        warnings.append(
            f"The preview found {preview.invalid_count} invalid mapped record(s); the build will quarantine them."
        )
    if scenario.modality in {"image", "audio"}:
        warnings.append("Shared media is grouped so one asset cannot cross a held-out split.")
    split_step = next(
        (
            step
            for step in recipe.get("steps") or []
            if isinstance(step, Mapping)
            and str(step.get("kind") or step.get("type") or "").lower() == "split"
        ),
        {},
    )
    split_ratios = copy.deepcopy(
        dict(
            split_step.get("ratios")
            or {"train": 0.8, "validation": 0.1, "test": 0.1}
        )
    )
    return DatasetPreparationPlan(
        inspection_id=str(inspection.get("id") or ""),
        scenario_revision_id=scenario.revision_id,
        mapping_plan=plan.to_dict(),
        recipe=recipe,
        estimated_input_records=total,
        estimated_accepted_records=max(0, total - estimated_invalid),
        estimated_quarantined_records=estimated_invalid,
        estimates_sampled=True,
        warnings=tuple(warnings),
        split_policy={
            "method": str(split_step.get("method") or "random"),
            "group_field": split_step.get("group_field"),
            "group_by_asset_hash": bool(
                split_step.get("group_by_asset_hash", False)
            ),
            "ratios": split_ratios,
            "seed": int(split_step.get("seed", recipe.get("seed", 42))),
            "protected_splits": [
                name
                for name in ("test", "canary")
                if float(split_ratios.get(name) or 0) > 0
            ],
        },
    )


__all__ = ["apply_mapping", "build_preparation_plan", "evaluate_expression", "preview_mapping"]
