"""Deterministic selectors for reviewed verifier-calibration failure mining.

The selectors in this module operate on one stable record at a time.  They do
not reinterpret rewards as probabilities or confidence scores; where a reward
margin is used, the exact threshold, range, and resolved margin are retained in
the returned evidence.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence

VERIFIER_FAILURE_SELECTORS = frozenset(
    {
        "false_accept",
        "false_reject",
        "high_confidence_disagreement",
        "repeat_instability",
        "order_flip",
        "ranking_inversion",
        "threshold_adjacent",
        "parser_runtime",
        "subgroup",
        "chain_component",
    }
)

_SECRET_OPTION_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "client_secret",
        "access_token",
        "refresh_token",
        "token",
    }
)


def _contains_credentials(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SECRET_OPTION_KEYS or _contains_credentials(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_credentials(nested) for nested in value)
    return False


def normalize_failure_selector(value: Any) -> tuple[str, Dict[str, Any]]:
    """Return a validated selector kind and its credential-free options."""

    if isinstance(value, str):
        kind, options = value, {}
    elif isinstance(value, Mapping):
        raw = dict(value)
        kind = str(raw.pop("kind", raw.pop("selector", "")))
        nested = raw.pop("options", {})
        if nested is not None and not isinstance(nested, Mapping):
            raise ValueError("verifier calibration selector options must be an object")
        options = {**dict(nested or {}), **raw}
    else:
        raise ValueError("verifier calibration selector must be a name or object")
    normalized = kind.strip().lower().replace("-", "_")
    if normalized not in VERIFIER_FAILURE_SELECTORS:
        raise ValueError(
            "unsupported verifier calibration selector; expected one of: "
            + ", ".join(sorted(VERIFIER_FAILURE_SELECTORS))
        )
    # Provider credentials are never meaningful selector inputs and must not
    # leak into acquisition manifests through a permissive options mapping.
    if _contains_credentials(options):
        raise ValueError("verifier calibration selector options cannot contain credentials")
    return normalized, options


def validate_failure_selector(
    selector: str,
    *,
    task_type: str,
    verifier_family: str,
) -> None:
    task = str(task_type).strip().lower()
    family = str(verifier_family).strip().lower()
    compatibility = {
        "false_accept": {"binary"},
        "false_reject": {"binary"},
        "order_flip": {"pairwise"},
        "ranking_inversion": {"ranking"},
    }
    supported = compatibility.get(selector)
    if supported is not None and task not in supported:
        raise ValueError(f"{selector} requires a {', '.join(sorted(supported))} verifier task")
    if selector == "chain_component" and family != "chain":
        raise ValueError("chain_component requires a chain verifier profile")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _observation(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    value = sample.get("observation")
    return value if isinstance(value, Mapping) else {}


def _expected(sample: Mapping[str, Any]) -> Any:
    reference = sample.get("reference")
    return reference.get("expected") if isinstance(reference, Mapping) else None


def _predicted(observation: Mapping[str, Any]) -> Any:
    parsed = observation.get("parsed_value")
    return parsed if parsed is not None else observation.get("passed")


def _primary(sample: Mapping[str, Any]) -> bool:
    return (
        int(sample.get("repeat_index") or 0) == 0
        and str(sample.get("probe_kind") or "canonical") == "canonical"
        and str(sample.get("orientation") or "canonical") in {"canonical", "a_b"}
    )


def _sample_summary(sample: Mapping[str, Any]) -> Dict[str, Any]:
    observation = _observation(sample)
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else {}
    return {
        "ordinal": sample.get("ordinal"),
        "repeat_index": int(sample.get("repeat_index") or 0),
        "orientation": str(sample.get("orientation") or "canonical"),
        "probe_kind": str(sample.get("probe_kind") or "canonical"),
        "seed": sample.get("seed"),
        "reward": observation.get("reward"),
        "passed": observation.get("passed"),
        "parsed_value": observation.get("parsed_value"),
        "error": observation.get("error"),
        "details": dict(observation.get("details") or {}),
        "component_trace": list(observation.get("component_trace") or []),
        "latency_ms": observation.get("latency_ms"),
        "runtime_identity": dict(observation.get("runtime_identity") or {}),
        "seed_honored": metadata.get("seed_honored"),
        "subgroup": dict(metadata.get("subgroup") or {}),
    }


def _reward_margin(
    observation: Mapping[str, Any], reward_contract: Mapping[str, Any]
) -> Optional[float]:
    reward = _finite(observation.get("reward"))
    threshold = _finite(reward_contract.get("threshold"))
    if reward is None or threshold is None:
        return None
    return abs(reward - threshold)


def _resolved_margin(options: Mapping[str, Any], reward_contract: Mapping[str, Any]) -> float:
    explicit = options.get("margin", options.get("reward_margin"))
    if explicit is not None:
        margin = _finite(explicit)
        if margin is None or margin < 0:
            raise ValueError("selector margin must be a finite non-negative number")
        return margin
    minimum = _finite(reward_contract.get("minimum"))
    maximum = _finite(reward_contract.get("maximum"))
    if minimum is None or maximum is None or maximum <= minimum:
        raise ValueError("a valid reward range is required to derive the selector margin")
    fraction = _finite(options.get("range_fraction", 0.05))
    if fraction is None or not 0 <= fraction <= 1:
        raise ValueError("selector range_fraction must be between zero and one")
    return (maximum - minimum) * fraction


def _repeat_instability(
    samples: Sequence[Mapping[str, Any]], *, tolerance: float
) -> tuple[bool, list[Dict[str, Any]]]:
    by_condition: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_condition[
            (
                str(sample.get("orientation") or "canonical"),
                str(sample.get("probe_kind") or "canonical"),
            )
        ].append(sample)
    unstable: list[Dict[str, Any]] = []
    for (orientation, probe), rows in sorted(by_condition.items()):
        repeats = {int(row.get("repeat_index") or 0) for row in rows}
        if len(repeats) < 2:
            continue
        observations = [_observation(row) for row in rows]
        passes = {_canonical(value.get("passed")) for value in observations}
        parsed = {_canonical(value.get("parsed_value")) for value in observations}
        errors = {_canonical(value.get("error")) for value in observations}
        rewards = [
            value
            for value in (_finite(observation.get("reward")) for observation in observations)
            if value is not None
        ]
        drift = max(rewards) - min(rewards) if len(rewards) >= 2 else 0.0
        if len(passes) > 1 or len(parsed) > 1 or len(errors) > 1 or drift > tolerance:
            unstable.append(
                {
                    "orientation": orientation,
                    "probe_kind": probe,
                    "repeat_count": len(repeats),
                    "reward_drift": drift,
                    "pass_flip": len(passes) > 1,
                    "parsed_flip": len(parsed) > 1,
                    "error_flip": len(errors) > 1,
                }
            )
    return bool(unstable), unstable


def _order_flip(samples: Sequence[Mapping[str, Any]]) -> tuple[bool, list[Dict[str, Any]]]:
    by_condition: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for sample in samples:
        if str(sample.get("probe_kind") or "canonical") != "canonical":
            continue
        by_condition[
            (int(sample.get("repeat_index") or 0), str(sample.get("probe_kind") or "canonical"))
        ].append(sample)
    flips: list[Dict[str, Any]] = []
    for (repeat_index, probe), rows in sorted(by_condition.items()):
        oriented = {
            str(row.get("orientation") or "canonical"): _predicted(_observation(row))
            for row in rows
        }
        if "a_b" in oriented and "b_a" in oriented:
            consistent = _canonical(oriented["a_b"]) == _canonical(oriented["b_a"])
            if not consistent:
                flips.append(
                    {
                        "repeat_index": repeat_index,
                        "probe_kind": probe,
                        "a_b": oriented["a_b"],
                        "b_a": oriented["b_a"],
                    }
                )
    return bool(flips), flips


def _ranking_inversions(expected: Any, predicted: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(expected, (list, tuple)) or not isinstance(predicted, (list, tuple)):
        return None
    expected_items = [str(value) for value in expected]
    predicted_items = [str(value) for value in predicted]
    if len(expected_items) < 2 or set(expected_items) != set(predicted_items):
        return None
    positions = {value: index for index, value in enumerate(predicted_items)}
    inversions = sum(
        positions[expected_items[left]] > positions[expected_items[right]]
        for left in range(len(expected_items))
        for right in range(left + 1, len(expected_items))
    )
    comparisons = len(expected_items) * (len(expected_items) - 1) // 2
    return {
        "inversions": inversions,
        "implied_comparisons": comparisons,
        "inversion_rate": inversions / comparisons,
        "expected": expected_items,
        "predicted": predicted_items,
    }


def _parser_runtime_failure(observation: Mapping[str, Any]) -> bool:
    if observation.get("error") not in {None, ""}:
        return True
    details = observation.get("details")
    if isinstance(details, Mapping):
        for key in (
            "parse_error",
            "parser_error",
            "runtime_error",
            "timeout",
            "legacy_verifier_error",
        ):
            value = details.get(key)
            if value is not None and value is not False and value != "":
                return True
    return False


def _component_failures(
    samples: Sequence[Mapping[str, Any]], options: Mapping[str, Any]
) -> list[Dict[str, Any]]:
    requested = str(
        options.get("component_revision_id") or options.get("component_id") or ""
    ).strip()
    result: list[Dict[str, Any]] = []
    for sample in samples:
        trace = list(_observation(sample).get("component_trace") or [])
        selected: list[Mapping[str, Any]] = []
        for component in trace:
            if not isinstance(component, Mapping):
                continue
            component_id = str(
                component.get("child_revision_id")
                or component.get("revision_id")
                or component.get("component_id")
                or ""
            )
            if requested and component_id != requested:
                continue
            selected.append(component)
        component_errors: list[str] = []
        identities: set[str] = set()
        for component in selected:
            nested = component.get("observation")
            observation = nested if isinstance(nested, Mapping) else component
            if observation.get("error") not in {None, ""}:
                component_errors.append(str(observation["error"]))
            identities.add(
                _canonical(
                    {
                        "passed": observation.get("passed"),
                        "parsed_value": observation.get("parsed_value"),
                        "reward": observation.get("reward"),
                    }
                )
            )
        if component_errors or len(identities) > 1:
            result.append(
                {
                    "repeat_index": int(sample.get("repeat_index") or 0),
                    "orientation": str(sample.get("orientation") or "canonical"),
                    "component_count": len(selected),
                    "component_errors": component_errors,
                    "component_disagreement": len(identities) > 1,
                }
            )
    return result


def select_calibration_failure(
    *,
    selector: str,
    options: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    task_type: str,
    verifier_family: str,
    reward_contract: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return selector evidence for one stable record, or ``None``.

    ``samples`` must contain calibration-partition observations only.  The
    caller is responsible for joining them to the immutable source record.
    """

    validate_failure_selector(selector, task_type=task_type, verifier_family=verifier_family)
    if not samples:
        return None
    summaries = [_sample_summary(sample) for sample in samples]
    expected = _expected(samples[0])
    primary = [sample for sample in samples if _primary(sample)] or list(samples)
    observations = [_observation(sample) for sample in primary]
    matched_reasons: list[str] = []
    diagnostics: Dict[str, Any] = {}

    if selector == "false_accept":
        matched = expected is False and any(value.get("passed") is True for value in observations)
    elif selector == "false_reject":
        matched = expected is True and any(value.get("passed") is False for value in observations)
    elif selector == "high_confidence_disagreement":
        margin = _resolved_margin({"range_fraction": 0.25, **dict(options)}, reward_contract)
        disagreements = []
        for observation in observations:
            predicted = _predicted(observation)
            reward_margin = _reward_margin(observation, reward_contract)
            if (
                predicted is not None
                and expected is not None
                and _canonical(predicted) != _canonical(expected)
                and reward_margin is not None
                and reward_margin >= margin
            ):
                disagreements.append({"predicted": predicted, "reward_margin": reward_margin})
        matched = bool(disagreements)
        diagnostics.update(disagreements=disagreements, resolved_reward_margin=margin)
    elif selector == "repeat_instability":
        tolerance = _finite(options.get("tolerance", 0.0))
        if tolerance is None or tolerance < 0:
            raise ValueError("repeat instability tolerance must be non-negative")
        matched, unstable = _repeat_instability(samples, tolerance=tolerance)
        diagnostics.update(instability=unstable, reward_tolerance=tolerance)
    elif selector == "order_flip":
        matched, flips = _order_flip(samples)
        diagnostics["order_flips"] = flips
    elif selector == "ranking_inversion":
        inversions = []
        for sample in primary:
            value = _ranking_inversions(expected, _predicted(_observation(sample)))
            if value is not None and value["inversions"] > 0:
                inversions.append(value)
        matched = bool(inversions)
        diagnostics["ranking_inversions"] = inversions
    elif selector == "threshold_adjacent":
        margin = _resolved_margin(options, reward_contract)
        adjacent = [
            value
            for value in (
                _reward_margin(observation, reward_contract) for observation in observations
            )
            if value is not None and value <= margin
        ]
        matched = bool(adjacent)
        diagnostics.update(reward_margins=adjacent, resolved_reward_margin=margin)
    elif selector == "parser_runtime":
        failures = [
            summary
            for summary, sample in zip(summaries, samples)
            if _parser_runtime_failure(_observation(sample))
        ]
        matched = bool(failures)
        diagnostics["parser_runtime_failures"] = failures
    elif selector == "subgroup":
        requested = options.get("subgroup")
        if requested is None and options.get("key") is not None:
            requested = {str(options["key"]): options.get("value")}
        if not isinstance(requested, Mapping) or not requested:
            raise ValueError("subgroup selector requires subgroup or key/value options")
        actual: Dict[str, Any] = {}
        for summary in summaries:
            actual.update(dict(summary.get("subgroup") or {}))
        matched = all(
            key in actual and (value is None or _canonical(actual[key]) == _canonical(value))
            for key, value in requested.items()
        )
        diagnostics.update(requested_subgroup=dict(requested), subgroup=actual)
    else:  # chain_component
        failures = _component_failures(samples, options)
        matched = bool(failures)
        diagnostics["component_failures"] = failures

    if not matched:
        return None
    matched_reasons.append(selector)
    return {
        "selector": selector,
        "selector_version": 1,
        "selector_options": dict(options),
        "matched_reasons": matched_reasons,
        "expected": expected,
        "observation_count": len(samples),
        "observations": summaries,
        **diagnostics,
    }


__all__ = [
    "VERIFIER_FAILURE_SELECTORS",
    "normalize_failure_selector",
    "select_calibration_failure",
    "validate_failure_selector",
]
