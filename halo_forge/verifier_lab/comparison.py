"""Direction-aware verifier calibration and sample comparisons."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from .metrics import metric_direction


@dataclass(frozen=True)
class CalibrationComparison:
    base_calibration_id: str
    candidate_calibration_id: str
    compatible: bool
    compatibility_reasons: tuple[str, ...]
    metric_deltas: tuple[Mapping[str, Any], ...] = ()
    decision_delta: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_calibration_id": self.base_calibration_id,
            "candidate_calibration_id": self.candidate_calibration_id,
            "compatible": self.compatible,
            "compatibility_reasons": list(self.compatibility_reasons),
            "metric_deltas": [dict(value) for value in self.metric_deltas],
            "decision_delta": None if self.decision_delta is None else dict(self.decision_delta),
        }


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        result = value.to_dict()
        if isinstance(result, Mapping):
            return result
    raise TypeError("calibration comparison inputs must be mappings or dataclasses")


def _metrics(value: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    source = value.get("metrics") or ()
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    if isinstance(source, Mapping):
        for name, metric in source.items():
            if isinstance(metric, Mapping):
                payload = dict(metric)
                payload.setdefault("name", str(name))
            else:
                payload = {"name": str(name), "value": metric}
            result[
                (
                    str(payload.get("partition", "calibration")),
                    str(payload["name"]),
                    str(payload.get("subgroup", "")),
                )
            ] = payload
    else:
        for metric in source:
            payload = dict(_mapping(metric))
            result[
                (
                    str(payload.get("partition", "calibration")),
                    str(payload.get("name", "")),
                    str(payload.get("subgroup", "")),
                )
            ] = payload
    return result


def _matched_decisions(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> Optional[dict[str, Any]]:
    """Compare only decisions made for the same qualification scope."""

    def by_scope(value: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for raw in value.get("decisions") or ():
            payload = dict(_mapping(raw))
            scope = str(payload.get("scope") or "")
            if scope:
                # Store/service ordering is append-only, so the last decision
                # for a scope is the active comparison point.
                result[scope] = payload
        return result

    left_scopes = by_scope(left)
    right_scopes = by_scope(right)
    common = set(left_scopes).intersection(right_scopes)
    if common:
        priority = ("confirmation", "operational", "development", "overall")
        scope = next((value for value in priority if value in common), sorted(common)[0])
        left_decision = left_scopes[scope].get("decision")
        right_decision = right_scopes[scope].get("decision")
    elif left_scopes or right_scopes:
        return {
            "base": None,
            "candidate": None,
            "scope": None,
            "classification": "missing_matching_scope",
            "base_scopes": sorted(left_scopes),
            "candidate_scopes": sorted(right_scopes),
        }
    else:
        left_decision = left.get("decision")
        right_decision = right.get("decision")
        scope = None
        if left_decision is None and right_decision is None:
            return None
    order = {"fail": 0, "warn": 1, "pass": 2}
    left_rank = order.get(str(left_decision), -1)
    right_rank = order.get(str(right_decision), -1)
    return {
        "base": left_decision,
        "candidate": right_decision,
        "scope": scope,
        "classification": (
            "improved"
            if right_rank > left_rank
            else "regressed"
            if right_rank < left_rank
            else "unchanged"
        ),
    }


def compare_calibrations(base: Any, candidate: Any) -> CalibrationComparison:
    left = _mapping(base)
    right = _mapping(candidate)
    reasons: list[str] = []
    compatibility_fields = (
        "source_hash",
        "protocol_hash",
        "qualification_hash",
        "task_type",
        "reward_contract_hash",
    )
    for field_name in compatibility_fields:
        left_value = left.get(field_name)
        right_value = right.get(field_name)
        if left_value is not None or right_value is not None:
            if left_value != right_value:
                reasons.append(f"{field_name} differs")
    compatible = not reasons
    deltas: list[Mapping[str, Any]] = []
    if compatible:
        left_metrics = _metrics(left)
        right_metrics = _metrics(right)
        for key in sorted(set(left_metrics).union(right_metrics)):
            left_metric = left_metrics.get(key)
            right_metric = right_metrics.get(key)
            if left_metric is None or right_metric is None:
                deltas.append(
                    {
                        "partition": key[0],
                        "name": key[1],
                        "subgroup": key[2],
                        "base": None if left_metric is None else left_metric.get("value"),
                        "candidate": None if right_metric is None else right_metric.get("value"),
                        "delta": None,
                        "classification": "missing_evidence",
                    }
                )
                continue
            left_value = left_metric.get("value")
            right_value = right_metric.get("value")
            left_direction = left_metric.get("direction")
            right_direction = right_metric.get("direction")
            direction = right_direction or left_direction or metric_direction(key[1])
            if not isinstance(left_value, (int, float)) or not isinstance(
                right_value, (int, float)
            ):
                delta = None
                classification = "missing_evidence"
            else:
                delta = float(right_value) - float(left_value)
                if left_direction and right_direction and left_direction != right_direction:
                    classification = "direction_mismatch"
                elif direction not in {"maximize", "minimize"}:
                    classification = "unchanged" if delta == 0 else "descriptive_change"
                else:
                    normalized = delta if direction == "maximize" else -delta
                    classification = (
                        "improved"
                        if normalized > 0
                        else "regressed"
                        if normalized < 0
                        else "unchanged"
                    )
            deltas.append(
                {
                    "partition": key[0],
                    "name": key[1],
                    "subgroup": key[2],
                    "base": left_value,
                    "candidate": right_value,
                    "delta": delta,
                    "direction": direction,
                    "classification": classification,
                }
            )
    decision_delta = _matched_decisions(left, right)
    return CalibrationComparison(
        base_calibration_id=str(left.get("id", left.get("calibration_id", ""))),
        candidate_calibration_id=str(right.get("id", right.get("calibration_id", ""))),
        compatible=compatible,
        compatibility_reasons=tuple(reasons),
        metric_deltas=tuple(deltas),
        decision_delta=decision_delta,
    )


def _sample_key(payload: Mapping[str, Any]) -> tuple[str, str, int, str, str]:
    return (
        str(payload.get("partition", "calibration")),
        str(payload.get("record_id", "")),
        int(payload.get("repeat_index", payload.get("repetition_index", 0))),
        str(payload.get("orientation", "canonical")),
        str(payload.get("probe_kind", payload.get("perturbation", "canonical"))),
    )


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _observation(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    value = sample.get("observation", sample)
    return value if isinstance(value, Mapping) else {}


def _expected(sample: Mapping[str, Any]) -> Any:
    reference = sample.get("reference")
    if isinstance(reference, Mapping) and "expected" in reference:
        return reference.get("expected")
    return sample.get("expected")


def _parsed(observation: Mapping[str, Any]) -> Any:
    if observation.get("parsed_value") is not None:
        return observation.get("parsed_value")
    if observation.get("passed") is not None:
        return observation.get("passed")
    return observation.get("reward")


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _binary_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and float(value) in {0.0, 1.0}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "pass", "passed", "positive", "1"}:
        return True
    if normalized in {"false", "no", "fail", "failed", "negative", "0"}:
        return False
    return None


def _equivalent(left: Any, right: Any, task_type: Optional[str]) -> bool:
    task = str(task_type or "").strip().lower()
    if task == "binary":
        left_binary = _binary_value(left)
        right_binary = _binary_value(right)
        if left_binary is not None and right_binary is not None:
            return left_binary == right_binary
    if task == "multi_label" and isinstance(left, (list, tuple, set)) and isinstance(
        right, (list, tuple, set)
    ):
        return {str(value) for value in left} == {str(value) for value in right}
    if task in {"categorical", "pairwise"} and left is not None and right is not None:
        return str(left) == str(right)
    return _canonical(left) == _canonical(right)


def _correctness(
    expected: Any, predicted: Any, task_type: Optional[str]
) -> Optional[bool]:
    if expected is None or predicted is None:
        return None
    return _equivalent(expected, predicted, task_type)


def _classify_sample_pair(
    base: Optional[Mapping[str, Any]],
    candidate: Optional[Mapping[str, Any]],
    *,
    task_type: Optional[str],
    reward_direction: str,
) -> dict[str, Any]:
    if base is None or candidate is None:
        return {
            "classification": "missing_evidence",
            "base_correct": None,
            "candidate_correct": None,
            "parsed_changed": None,
            "reward_delta": None,
        }
    left = _observation(base)
    right = _observation(candidate)
    left_error = left.get("error")
    right_error = right.get("error")
    left_parsed = _parsed(left)
    right_parsed = _parsed(right)
    parsed_changed = not _equivalent(left_parsed, right_parsed, task_type)
    left_reward = _numeric(left.get("reward"))
    right_reward = _numeric(right.get("reward"))
    reward_delta = (
        right_reward - left_reward
        if left_reward is not None and right_reward is not None
        else None
    )
    expected = _expected(candidate)
    if expected is None:
        expected = _expected(base)
    left_correct = _correctness(expected, left_parsed, task_type)
    right_correct = _correctness(expected, right_parsed, task_type)

    if right_error and not left_error:
        classification = "regressed"
    elif left_error and not right_error:
        classification = "improved"
    elif left_error or right_error:
        classification = (
            "unchanged"
            if _canonical(left_error) == _canonical(right_error) and not parsed_changed
            else "changed"
        )
    elif str(task_type or "").lower() == "scalar":
        target = _numeric(expected)
        left_value = _numeric(left_parsed)
        right_value = _numeric(right_parsed)
        if target is not None and left_value is not None and right_value is not None:
            left_distance = abs(left_value - target)
            right_distance = abs(right_value - target)
            classification = (
                "improved"
                if right_distance < left_distance
                else "regressed"
                if right_distance > left_distance
                else "unchanged"
                if not parsed_changed
                else "changed"
            )
        else:
            classification = "changed" if parsed_changed else "unchanged"
    elif left_correct is not None and right_correct is not None and left_correct != right_correct:
        classification = "improved" if right_correct else "regressed"
    elif expected is None and left.get("passed") != right.get("passed"):
        # Compatibility fallback for legacy samples without reference labels.
        classification = "improved" if right.get("passed") is True else "regressed"
    elif parsed_changed:
        classification = "changed"
    elif reward_delta is not None and reward_delta != 0:
        normalized = reward_delta if reward_direction == "maximize" else -reward_delta
        classification = "improved" if normalized > 0 else "regressed"
    else:
        classification = "unchanged"
    return {
        "classification": classification,
        "base_correct": left_correct,
        "candidate_correct": right_correct,
        "parsed_changed": parsed_changed,
        "reward_delta": reward_delta,
    }


def compare_joined_calibration_samples(
    pairs: Iterable[tuple[Optional[Any], Optional[Any]]],
    *,
    total: int,
    limit: int,
    offset: int,
    task_type: Optional[str] = None,
    reward_direction: str = "maximize",
) -> dict[str, Any]:
    """Classify a pre-paged indexed join without materializing its full domain."""

    items: list[dict[str, Any]] = []
    for base_value, candidate_value in pairs:
        base = None if base_value is None else _mapping(base_value)
        candidate = None if candidate_value is None else _mapping(candidate_value)
        sample = candidate or base or {}
        sample_key = _sample_key(sample)
        delta = _classify_sample_pair(
            base,
            candidate,
            task_type=task_type,
            reward_direction=reward_direction,
        )
        items.append(
            {
                "partition": sample_key[0],
                "record_id": sample_key[1],
                "repeat_index": sample_key[2],
                "orientation": sample_key[3],
                "probe_kind": sample_key[4],
                **delta,
                "base": base,
                "candidate": candidate,
            }
        )
    return {
        "items": items,
        "total": int(total),
        "limit": int(limit),
        "offset": int(offset),
    }


def compare_calibration_samples(
    base_samples: Sequence[Any],
    candidate_samples: Sequence[Any],
    *,
    limit: int = 100,
    offset: int = 0,
    task_type: Optional[str] = None,
    reward_direction: str = "maximize",
) -> dict[str, Any]:
    """Join bounded sample evidence by stable record/protocol expansion identity."""

    if limit < 1 or limit > 1_000:
        raise ValueError("sample comparison limit must be between 1 and 1000")
    if offset < 0:
        raise ValueError("sample comparison offset cannot be negative")

    left = {_sample_key(_mapping(value)): _mapping(value) for value in base_samples}
    right = {_sample_key(_mapping(value)): _mapping(value) for value in candidate_samples}
    keys = sorted(set(left).union(right))
    selected = keys[offset : offset + limit]
    return compare_joined_calibration_samples(
        ((left.get(key), right.get(key)) for key in selected),
        total=len(keys),
        limit=limit,
        offset=offset,
        task_type=task_type,
        reward_direction=reward_direction,
    )


__all__ = [
    "CalibrationComparison",
    "compare_calibration_samples",
    "compare_joined_calibration_samples",
    "compare_calibrations",
]
