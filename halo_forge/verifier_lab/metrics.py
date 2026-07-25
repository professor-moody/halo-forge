"""Reliability metrics with stable-record bootstrap replication."""

from __future__ import annotations

import math
import random
import statistics
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

from .observation import RewardContract, normalize_reward_contract


TASK_TYPES = {"binary", "categorical", "multi_label", "scalar", "pairwise", "ranking"}
PRIMARY_METRIC_BY_TASK = {
    "binary": "balanced_accuracy",
    "categorical": "macro_f1",
    "multi_label": "macro_f1",
    "scalar": "spearman",
    "pairwise": "tie_aware_accuracy",
    "ranking": "kendall_tau",
}

# A direction is meaningful only for a measure whose movement has a reliable
# quality interpretation.  Counts and descriptive distribution statistics are
# intentionally directionless: a larger sample count or reward margin is not,
# by itself, evidence that a verifier became more reliable.
MINIMIZE_METRICS = frozenset(
    {
        "error_rate",
        "parse_error_rate",
        "timeout_rate",
        "false_accept_rate",
        "false_reject_rate",
        "hamming_loss",
        "mae",
        "rmse",
        "normalized_mae",
        "normalized_rmse",
        "inversion_rate",
        "brier_score",
        "ece",
        "mean_repeat_drift",
        "maximum_repeat_drift",
        "pass_flip_rate",
        "seed_ignored_rate",
        "chain_component_error_count",
        "reward_saturation_min_rate",
        "reward_saturation_max_rate",
        "latency_ms_mean",
        "latency_ms_p50",
        "latency_ms_p95",
        "reward_model_batch_max_delta",
    }
)
MAXIMIZE_METRICS = frozenset(
    {
        "coverage",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "macro_f1",
        "micro_f1",
        "exact_match",
        "spearman",
        "kendall_tau",
        "concordance",
        "tie_aware_accuracy",
        "order_consistency",
        "top_1",
        "ndcg",
        "repeat_agreement",
        "exact_repeat_agreement",
        "reward_model_batch_consistency",
        "reward_model_batch_evidence_coverage",
    }
)


def metric_direction(name: str) -> Optional[str]:
    """Return the reliability direction for a metric, if one is defensible."""

    normalized = str(name).rsplit(".", 1)[-1]
    if normalized in MINIMIZE_METRICS:
        return "minimize"
    if normalized in MAXIMIZE_METRICS:
        return "maximize"
    return None


@dataclass(frozen=True)
class CalibrationEvidence:
    record_id: str
    expected: Any
    predicted: Any = None
    reward: Optional[float] = None
    passed: Optional[bool] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None
    timeout: bool = False
    latency_ms: Optional[float] = None
    repetition_index: int = 0
    seed: Optional[int] = None
    seed_honored: Optional[bool] = None
    orientation: str = "canonical"
    perturbation: str = "canonical"
    probability: Optional[float] = None
    subgroup: Mapping[str, Any] = None  # type: ignore[assignment]
    component_trace: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not str(self.record_id).strip():
            raise ValueError("calibration evidence requires record_id")
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise ValueError("calibration evidence reward must be finite")
        if self.latency_ms is not None and (
            not math.isfinite(float(self.latency_ms)) or float(self.latency_ms) < 0
        ):
            raise ValueError("latency_ms must be finite and non-negative")
        if self.probability is not None and not 0.0 <= float(self.probability) <= 1.0:
            raise ValueError("probability must be between zero and one")
        object.__setattr__(self, "subgroup", dict(self.subgroup or {}))

    @property
    def available(self) -> bool:
        return self.error is None and any(
            value is not None for value in (self.predicted, self.reward, self.passed)
        )

    @property
    def primary(self) -> bool:
        return (
            self.repetition_index == 0
            and self.perturbation == "canonical"
            and self.orientation in {"canonical", "a_b"}
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    fields = (
        "record_id",
        "expected",
        "predicted",
        "parsed_value",
        "reward",
        "passed",
        "error",
        "error_kind",
        "timeout",
        "latency_ms",
        "repetition_index",
        "seed",
        "seed_honored",
        "orientation",
        "perturbation",
        "probability",
        "subgroup",
        "metadata",
        "component_trace",
    )
    return {field: getattr(value, field) for field in fields if hasattr(value, field)}


def normalize_evidence(value: CalibrationEvidence | Mapping[str, Any] | Any) -> CalibrationEvidence:
    if isinstance(value, CalibrationEvidence):
        return value
    payload = _as_mapping(value)
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    error_value = payload.get("error")
    error = None if error_value in {None, ""} else str(error_value)
    return CalibrationEvidence(
        record_id=str(payload.get("record_id", "")),
        expected=payload.get("expected"),
        predicted=payload.get("predicted", payload.get("parsed_value")),
        reward=(None if payload.get("reward") is None else float(payload["reward"])),
        passed=payload.get("passed"),
        error=error,
        error_kind=(
            None
            if payload.get("error_kind", metadata.get("error_kind")) in {None, ""}
            else str(payload.get("error_kind", metadata.get("error_kind")))
        ),
        timeout=bool(payload.get("timeout", metadata.get("timeout", False))),
        latency_ms=(
            None if payload.get("latency_ms") is None else float(payload["latency_ms"])
        ),
        repetition_index=int(payload.get("repetition_index", metadata.get("repetition_index", 0))),
        seed=payload.get("seed", metadata.get("seed")),
        seed_honored=payload.get("seed_honored", metadata.get("seed_honored")),
        orientation=str(payload.get("orientation", metadata.get("orientation", "canonical"))),
        perturbation=str(
            payload.get("perturbation", metadata.get("perturbation", "canonical"))
        ),
        probability=(
            None if payload.get("probability") is None else float(payload["probability"])
        ),
        subgroup=payload.get("subgroup", metadata.get("subgroup", {})) or {},
        component_trace=tuple(payload.get("component_trace") or ()),
    )


def _safe_divide(numerator: float, denominator: float) -> Optional[float]:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _mean_available(values: Iterable[Optional[float]]) -> Optional[float]:
    available = [float(value) for value in values if value is not None]
    return statistics.fmean(available) if available else None


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of empty values")
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _primary_rows(rows: Sequence[CalibrationEvidence]) -> list[CalibrationEvidence]:
    available = [row for row in rows if row.primary and row.available]
    # Protocol evidence should have one canonical first repetition per record.
    # Keep the first deterministic row if malformed duplicates reach analysis.
    by_record: dict[str, CalibrationEvidence] = {}
    for row in available:
        by_record.setdefault(row.record_id, row)
    return [by_record[key] for key in sorted(by_record)]


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


def _binary_metrics(rows: Sequence[CalibrationEvidence]) -> dict[str, Any]:
    pairs: list[tuple[bool, bool]] = []
    for row in _primary_rows(rows):
        expected = _binary_value(row.expected)
        predicted = _binary_value(row.predicted if row.predicted is not None else row.passed)
        if expected is not None and predicted is not None:
            pairs.append((expected, predicted))
    tp = sum(expected and predicted for expected, predicted in pairs)
    tn = sum(not expected and not predicted for expected, predicted in pairs)
    fp = sum(not expected and predicted for expected, predicted in pairs)
    fn = sum(expected and not predicted for expected, predicted in pairs)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2.0 * precision * recall / (precision + recall)
    )
    specificity = _safe_divide(tn, tn + fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "record_count": len(pairs),
        "accuracy": _safe_divide(tp + tn, len(pairs)),
        "balanced_accuracy": _mean_available((recall, specificity)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": _safe_divide(tp * tn - fp * fn, denominator),
        "false_accept_rate": _safe_divide(fp, fp + tn),
        "false_reject_rate": _safe_divide(fn, fn + tp),
        "confusion_matrix": {"true_negative": tn, "false_positive": fp, "false_negative": fn, "true_positive": tp},
    }


def _class_metrics(expected: Sequence[str], predicted: Sequence[str], label: str) -> dict[str, Any]:
    tp = sum(left == label and right == label for left, right in zip(expected, predicted))
    fp = sum(left != label and right == label for left, right in zip(expected, predicted))
    fn = sum(left == label and right != label for left, right in zip(expected, predicted))
    tn = len(expected) - tp - fp - fn
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = (
        None
        if precision is None or recall is None or precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return {
        "support": sum(value == label for value in expected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def _categorical_metrics(rows: Sequence[CalibrationEvidence]) -> dict[str, Any]:
    pairs = [
        (str(row.expected), str(row.predicted))
        for row in _primary_rows(rows)
        if row.expected is not None and row.predicted is not None
    ]
    expected = [left for left, _ in pairs]
    predicted = [right for _, right in pairs]
    labels = sorted(set(expected).union(predicted))
    per_class = {label: _class_metrics(expected, predicted, label) for label in labels}
    confusion = {
        actual: {predicted_label: 0 for predicted_label in labels} for actual in labels
    }
    for actual, prediction in pairs:
        confusion[actual][prediction] += 1
    return {
        "record_count": len(pairs),
        "accuracy": _safe_divide(sum(left == right for left, right in pairs), len(pairs)),
        "macro_f1": _mean_available(item["f1"] for item in per_class.values()),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _label_set(value: Any) -> Optional[set[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item) for item in value}
    return None


def _multi_label_metrics(rows: Sequence[CalibrationEvidence]) -> dict[str, Any]:
    pairs: list[tuple[set[str], set[str]]] = []
    for row in _primary_rows(rows):
        expected = _label_set(row.expected)
        predicted = _label_set(row.predicted)
        if expected is not None and predicted is not None:
            pairs.append((expected, predicted))
    labels = sorted(set().union(*(left.union(right) for left, right in pairs))) if pairs else []
    per_label: dict[str, dict[str, Any]] = {}
    total_tp = total_fp = total_fn = 0
    mismatches = 0
    for label in labels:
        expected_binary = [label in left for left, _ in pairs]
        predicted_binary = [label in right for _, right in pairs]
        tp = sum(left and right for left, right in zip(expected_binary, predicted_binary))
        fp = sum(not left and right for left, right in zip(expected_binary, predicted_binary))
        fn = sum(left and not right for left, right in zip(expected_binary, predicted_binary))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = (
            None
            if precision is None or recall is None or precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        per_label[label] = {"support": sum(expected_binary), "precision": precision, "recall": recall, "f1": f1}
        total_tp += tp
        total_fp += fp
        total_fn += fn
        mismatches += sum(left != right for left, right in zip(expected_binary, predicted_binary))
    micro_precision = _safe_divide(total_tp, total_tp + total_fp)
    micro_recall = _safe_divide(total_tp, total_tp + total_fn)
    micro_f1 = (
        None
        if micro_precision is None or micro_recall is None or micro_precision + micro_recall == 0
        else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    )
    return {
        "record_count": len(pairs),
        "label_count": len(labels),
        "macro_f1": _mean_available(value["f1"] for value in per_label.values()),
        "micro_f1": micro_f1,
        "hamming_loss": _safe_divide(mismatches, len(pairs) * len(labels)),
        "exact_match": _safe_divide(sum(left == right for left, right in pairs), len(pairs)),
        "per_label": per_label,
    }


def _numeric_pairs(rows: Sequence[CalibrationEvidence]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for row in _primary_rows(rows):
        prediction = row.predicted if row.predicted is not None else row.reward
        try:
            expected = float(row.expected)
            predicted = float(prediction)
        except (TypeError, ValueError):
            continue
        if math.isfinite(expected) and math.isfinite(predicted):
            result.append((expected, predicted))
    return result


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = average_rank
        position = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return _safe_divide(numerator, denominator)


def _spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _kendall_tau(left: Sequence[Any], right: Sequence[Any]) -> Optional[float]:
    if len(left) < 2 or len(left) != len(right):
        return None
    # Tau-b via an inversion count.  The former pairwise loop was O(n^2),
    # which made the otherwise bounded scalar calibration path unusable at
    # 100,000 records.  Records tied on the left are inserted into the Fenwick
    # tree only after their whole group is queried, so those pairs never become
    # false inversions.  Equal right values are queried inclusively and are
    # likewise excluded from discordance.
    pairs = sorted(zip(left, right), key=lambda value: (value[0], value[1]))
    right_values = sorted(set(right))
    right_rank = {value: index + 1 for index, value in enumerate(right_values)}
    tree = [0] * (len(right_values) + 1)

    def query(index: int) -> int:
        result = 0
        while index > 0:
            result += tree[index]
            index -= index & -index
        return result

    def update(index: int) -> None:
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    discordant = 0
    inserted = 0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        for _, value in pairs[start:end]:
            discordant += inserted - query(right_rank[value])
        for _, value in pairs[start:end]:
            update(right_rank[value])
            inserted += 1
        start = end

    def tied_pairs(values: Sequence[Any]) -> int:
        return sum(count * (count - 1) // 2 for count in Counter(values).values())

    total_pairs = len(left) * (len(left) - 1) // 2
    tied_left_total = tied_pairs(left)
    tied_right_total = tied_pairs(right)
    tied_both = sum(
        count * (count - 1) // 2 for count in Counter(zip(left, right)).values()
    )
    comparable = total_pairs - tied_left_total - tied_right_total + tied_both
    concordant = comparable - discordant
    ties_left = tied_left_total - tied_both
    ties_right = tied_right_total - tied_both
    denominator = math.sqrt(
        (concordant + discordant + ties_left) * (concordant + discordant + ties_right)
    )
    return _safe_divide(concordant - discordant, denominator)


def _scalar_metrics(rows: Sequence[CalibrationEvidence], contract: RewardContract) -> dict[str, Any]:
    pairs = _numeric_pairs(rows)
    errors = [predicted - expected for expected, predicted in pairs]
    span = contract.maximum - contract.minimum
    expected = [left for left, _ in pairs]
    predicted = [right for _, right in pairs]
    return {
        "record_count": len(pairs),
        "mae": statistics.fmean(abs(value) for value in errors) if errors else None,
        "rmse": math.sqrt(statistics.fmean(value * value for value in errors)) if errors else None,
        "normalized_mae": (
            statistics.fmean(abs(value) for value in errors) / span if errors else None
        ),
        "normalized_rmse": (
            math.sqrt(statistics.fmean(value * value for value in errors)) / span if errors else None
        ),
        "spearman": _spearman(expected, predicted),
        "kendall_tau": _kendall_tau(expected, predicted),
    }


def _pairwise_metrics(rows: Sequence[CalibrationEvidence], contract: RewardContract) -> dict[str, Any]:
    primary = _primary_rows(rows)
    pairs = [
        (str(row.expected), str(row.predicted))
        for row in primary
        if row.expected is not None and row.predicted is not None
    ]
    non_tie = [(left, right) for left, right in pairs if left.lower() != "tie"]
    by_record: dict[str, list[CalibrationEvidence]] = defaultdict(list)
    for row in rows:
        if row.available and row.perturbation == "canonical" and row.repetition_index == 0:
            by_record[row.record_id].append(row)
    orientation_consistency: list[bool] = []
    for record_rows in by_record.values():
        orientations = {row.orientation: row for row in record_rows}
        if "a_b" in orientations and "b_a" in orientations:
            left = orientations["a_b"].predicted
            right = orientations["b_a"].predicted
            if left is not None and right is not None:
                orientation_consistency.append(str(left) == str(right))
    margins = [
        abs(float(row.reward) - float(contract.threshold))
        for row in primary
        if row.reward is not None and contract.threshold is not None
    ]
    return {
        "record_count": len(pairs),
        "concordance": _safe_divide(sum(left == right for left, right in non_tie), len(non_tie)),
        "tie_aware_accuracy": _safe_divide(sum(left == right for left, right in pairs), len(pairs)),
        "mean_margin": statistics.fmean(margins) if margins else None,
        "median_margin": statistics.median(margins) if margins else None,
        "order_consistency": _safe_divide(sum(orientation_consistency), len(orientation_consistency)),
        "order_comparison_count": len(orientation_consistency),
    }


def _ranking_items(value: Any) -> Optional[list[str]]:
    if not isinstance(value, (list, tuple)):
        return None
    result = [str(item) for item in value]
    return result if len(result) == len(set(result)) and len(result) >= 2 else None


def _ranking_record_metrics(expected: list[str], predicted: list[str]) -> Optional[dict[str, float]]:
    if set(expected) != set(predicted):
        return None
    expected_position = {item: index for index, item in enumerate(expected)}
    predicted_positions = [predicted.index(item) for item in expected]
    expected_positions = list(range(len(expected)))
    tau = _kendall_tau(expected_positions, predicted_positions)
    pairs = len(expected) * (len(expected) - 1) // 2
    inversions = sum(
        predicted_positions[first] > predicted_positions[second]
        for first in range(len(expected))
        for second in range(first + 1, len(expected))
    )
    relevance = {item: len(expected) - expected_position[item] for item in expected}

    def dcg(order: Sequence[str]) -> float:
        return sum(
            (2 ** relevance[item] - 1) / math.log2(index + 2)
            for index, item in enumerate(order)
        )

    ideal = dcg(expected)
    return {
        "kendall_tau": tau if tau is not None else 0.0,
        "top_1": float(expected[0] == predicted[0]),
        "ndcg": dcg(predicted) / ideal,
        "inversion_rate": inversions / pairs,
        "implied_comparisons": float(pairs),
    }


def _ranking_metrics(rows: Sequence[CalibrationEvidence]) -> dict[str, Any]:
    values: list[dict[str, float]] = []
    for row in _primary_rows(rows):
        expected = _ranking_items(row.expected)
        predicted = _ranking_items(row.predicted)
        if expected is not None and predicted is not None:
            metric = _ranking_record_metrics(expected, predicted)
            if metric is not None:
                values.append(metric)
    return {
        "record_count": len(values),
        "implied_comparisons": int(sum(value["implied_comparisons"] for value in values)),
        "kendall_tau": _mean_available(value["kendall_tau"] for value in values),
        "top_1": _mean_available(value["top_1"] for value in values),
        "ndcg": _mean_available(value["ndcg"] for value in values),
        "inversion_rate": _mean_available(value["inversion_rate"] for value in values),
    }


def compute_task_metrics(
    evidence: Sequence[CalibrationEvidence | Mapping[str, Any] | Any],
    *,
    task_type: str,
    reward_contract: RewardContract | Mapping[str, Any] | Any,
) -> dict[str, Any]:
    rows = [normalize_evidence(value) for value in evidence]
    task = str(task_type).strip().lower()
    if task not in TASK_TYPES:
        raise ValueError("unsupported calibration task type")
    contract = normalize_reward_contract(reward_contract)
    if task == "binary":
        return _binary_metrics(rows)
    if task == "categorical":
        return _categorical_metrics(rows)
    if task == "multi_label":
        return _multi_label_metrics(rows)
    if task == "scalar":
        return _scalar_metrics(rows, contract)
    if task == "pairwise":
        return _pairwise_metrics(rows, contract)
    return _ranking_metrics(rows)


def _universal_metrics(rows: Sequence[CalibrationEvidence], contract: RewardContract) -> dict[str, Any]:
    record_ids = sorted({row.record_id for row in rows})
    available = [row for row in rows if row.available]
    errors = [row for row in rows if row.error is not None]
    timeouts = [row for row in rows if row.timeout or row.error_kind == "timeout"]
    parse_errors = [
        row
        for row in rows
        if row.error_kind == "parse" or (row.error and "parse" in row.error.lower())
    ]
    rewards = [float(row.reward) for row in available if row.reward is not None]
    latencies = [float(row.latency_ms) for row in rows if row.latency_ms is not None]
    by_repeat: dict[tuple[str, str, str], list[CalibrationEvidence]] = defaultdict(list)
    for row in rows:
        by_repeat[(row.record_id, row.orientation, row.perturbation)].append(row)
    repeat_groups = [group for group in by_repeat.values() if len(group) >= 2]
    drifts: list[float] = []
    exact: list[bool] = []
    flips: list[bool] = []
    for group in repeat_groups:
        group_rewards = [float(row.reward) for row in group if row.reward is not None]
        group_passes = [row.passed for row in group if row.passed is not None]
        if len(group_rewards) >= 2:
            drifts.append(max(group_rewards) - min(group_rewards))
        # Exact repeat identity includes the parsed value and failure state.
        # Canonical JSON keeps nested/list/dict values structural and avoids
        # treating two differently parsed outputs as equal merely because
        # their thresholded reward happened to match.
        signatures = {
            json.dumps(
                {
                    "reward": row.reward,
                    "passed": row.passed,
                    "parsed_value": row.predicted,
                    "error": row.error,
                    "error_kind": row.error_kind,
                    "timeout": row.timeout,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )
            for row in group
        }
        exact.append(len(signatures) == 1)
        comparable = len(group_passes) == len(group)
        if comparable:
            flips.append(len(set(group_passes)) > 1)
    covered_records = {row.record_id for row in available}
    seed_claims = [row.seed_honored for row in rows if row.seed_honored is not None]
    component_errors = 0
    for row in rows:
        for component in row.component_trace:
            observation = component.get("observation", component)
            if isinstance(observation, Mapping) and observation.get("error"):
                component_errors += 1
    return {
        "invocation_count": len(rows),
        "record_count": len(record_ids),
        "coverage": _safe_divide(len(covered_records), len(record_ids)),
        "missing_evidence_records": sorted(set(record_ids).difference(covered_records)),
        "parse_error_rate": _safe_divide(len(parse_errors), len(rows)),
        "error_rate": _safe_divide(len(errors), len(rows)),
        "timeout_rate": _safe_divide(len(timeouts), len(rows)),
        "reward_saturation_min_rate": _safe_divide(
            sum(math.isclose(value, contract.minimum, rel_tol=0.0, abs_tol=1e-12) for value in rewards),
            len(rewards),
        ),
        "reward_saturation_max_rate": _safe_divide(
            sum(math.isclose(value, contract.maximum, rel_tol=0.0, abs_tol=1e-12) for value in rewards),
            len(rewards),
        ),
        "repeat_group_count": len(repeat_groups),
        "mean_repeat_drift": statistics.fmean(drifts) if drifts else None,
        "maximum_repeat_drift": max(drifts) if drifts else None,
        "repeat_agreement": _safe_divide(sum(not value for value in flips), len(flips)),
        "exact_repeat_agreement": _safe_divide(sum(exact), len(exact)),
        "pass_flip_rate": _safe_divide(sum(flips), len(flips)),
        "latency_ms_mean": statistics.fmean(latencies) if latencies else None,
        "latency_ms_p50": _percentile(latencies, 0.50) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 0.95) if latencies else None,
        "seed_ignored_rate": _safe_divide(sum(value is False for value in seed_claims), len(seed_claims)),
        "chain_component_error_count": component_errors,
    }


def _probability_metrics(rows: Sequence[CalibrationEvidence], *, bins: int = 10) -> dict[str, Any]:
    values: list[tuple[float, float]] = []
    for row in _primary_rows(rows):
        expected = _binary_value(row.expected)
        if expected is not None and row.probability is not None:
            values.append((float(expected), float(row.probability)))
    if not values:
        return {"brier_score": None, "ece": None, "ece_bins": []}
    brier = statistics.fmean((probability - expected) ** 2 for expected, probability in values)
    ordered = sorted(values, key=lambda item: item[1])
    count = min(max(1, int(bins)), len(ordered))
    summaries: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(count):
        start = math.floor(index * len(ordered) / count)
        end = math.floor((index + 1) * len(ordered) / count)
        bucket = ordered[start:end]
        confidence = statistics.fmean(value[1] for value in bucket)
        accuracy = statistics.fmean(value[0] for value in bucket)
        weight = len(bucket) / len(ordered)
        ece += weight * abs(confidence - accuracy)
        summaries.append(
            {
                "count": len(bucket),
                "mean_probability": confidence,
                "observed_rate": accuracy,
            }
        )
    return {"brier_score": brier, "ece": ece, "ece_bins": summaries}


def grouped_percentile_bootstrap(
    evidence: Sequence[CalibrationEvidence | Mapping[str, Any] | Any],
    metric: Callable[[Sequence[CalibrationEvidence]], Optional[float]],
    *,
    resamples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> Optional[dict[str, float]]:
    """Bootstrap stable records, carrying all repeats/orientations together."""

    if int(resamples) <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    rows = [normalize_evidence(value) for value in evidence]
    by_record: dict[str, list[CalibrationEvidence]] = defaultdict(list)
    for row in rows:
        by_record[row.record_id].append(row)
    record_ids = sorted(by_record)
    if not record_ids:
        return None
    rng = random.Random(int(seed))
    values: list[float] = []
    for _ in range(int(resamples)):
        sampled: list[CalibrationEvidence] = []
        for slot, _ in enumerate(record_ids):
            selected_id = record_ids[rng.randrange(len(record_ids))]
            sampled.extend(
                replace(row, record_id=f"{selected_id}#bootstrap-{slot}")
                for row in by_record[selected_id]
            )
        value = metric(sampled)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return None
    tail = (1.0 - confidence_level) / 2.0
    return {
        "lower": _percentile(values, tail),
        "upper": _percentile(values, 1.0 - tail),
        "confidence_level": confidence_level,
        "resamples": len(values),
        "seed": int(seed),
        "replicate_unit": "stable_record",
    }


# The exact engines below sample multinomial category counts rather than an
# expanded list of record objects.  Categories are sufficient-statistic
# equivalence classes: sampling their counts is distributionally identical to
# sampling individual stable records with replacement.  Work is therefore
# O(resamples * categories), not O(resamples * records), for classification,
# pairwise, and the common fixed-shape ranking/multi-label cases.
_EXACT_BOOTSTRAP_CATEGORY_LIMIT = 4_096
_BOOTSTRAP_TARGET_CELLS = 2_000_000
_BLB_BAGS = 5
_BLB_EXPONENT = 0.60
_BLB_MIN_SUBSAMPLE = 256


def _multinomial_chunks(
    counts: Sequence[int] | np.ndarray,
    *,
    resamples: int,
    rng: np.random.Generator,
    sample_size: Optional[int] = None,
) -> Iterator[np.ndarray]:
    category_counts = np.asarray(counts, dtype=np.int64)
    if category_counts.ndim != 1 or not len(category_counts):
        return
    source_count = int(category_counts.sum())
    if source_count <= 0:
        return
    draw_count = source_count if sample_size is None else int(sample_size)
    probabilities = category_counts.astype(np.float64) / source_count
    chunk_size = max(
        1,
        min(512, _BOOTSTRAP_TARGET_CELLS // max(1, len(category_counts))),
    )
    remaining = int(resamples)
    while remaining:
        size = min(remaining, chunk_size)
        yield rng.multinomial(draw_count, probabilities, size=size)
        remaining -= size


def _percentile_interval(
    values: Sequence[float] | np.ndarray,
    *,
    resamples: int,
    seed: int,
    confidence_level: float = 0.95,
    method: str = "compressed_multinomial_percentile",
    exact: bool = True,
    **metadata: Any,
) -> Optional[dict[str, Any]]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    tail = (1.0 - float(confidence_level)) / 2.0
    return {
        "lower": float(np.quantile(finite, tail)),
        "upper": float(np.quantile(finite, 1.0 - tail)),
        "confidence_level": float(confidence_level),
        "resamples": int(len(finite)),
        "requested_resamples": int(resamples),
        "seed": int(seed),
        "replicate_unit": "stable_record",
        "method": method,
        "exact": bool(exact),
        **metadata,
    }


def _compressed_mean_interval(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> Optional[dict[str, Any]]:
    categories = Counter(float(value) for value in values if math.isfinite(float(value)))
    if not categories:
        return None
    ordered = sorted(categories)
    counts = np.asarray([categories[value] for value in ordered], dtype=np.int64)
    category_values = np.asarray(ordered, dtype=np.float64)
    if len(categories) <= _EXACT_BOOTSTRAP_CATEGORY_LIMIT:
        rng = np.random.default_rng(int(seed))
        samples = [
            draws @ category_values / draws.sum(axis=1)
            for draws in _multinomial_chunks(counts, resamples=resamples, rng=rng)
        ]
        return _percentile_interval(
            np.concatenate(samples),
            resamples=resamples,
            seed=seed,
            category_count=len(categories),
        )
    return _blb_mean_interval(values, resamples=resamples, seed=seed)


def _blb_layout(record_count: int, resamples: int) -> tuple[int, int, list[int]]:
    bag_count = min(_BLB_BAGS, max(1, int(resamples)), max(1, int(record_count)))
    subsample_size = min(
        int(record_count),
        max(_BLB_MIN_SUBSAMPLE, int(math.ceil(record_count ** _BLB_EXPONENT))),
    )
    # Prefer disjoint deterministic bags when the source is large enough.
    if subsample_size * bag_count > record_count:
        subsample_size = max(1, record_count // bag_count)
    quotient, remainder = divmod(int(resamples), bag_count)
    per_bag = [quotient + (1 if index < remainder else 0) for index in range(bag_count)]
    return bag_count, subsample_size, per_bag


def _blb_interval_from_bags(
    bag_values: Sequence[np.ndarray],
    *,
    resamples: int,
    seed: int,
    subsample_size: int,
    source_records: int,
) -> Optional[dict[str, Any]]:
    nonempty = [values[np.isfinite(values)] for values in bag_values if len(values)]
    nonempty = [values for values in nonempty if len(values)]
    if not nonempty:
        return None
    tail = 0.025
    lowers = [float(np.quantile(values, tail)) for values in nonempty]
    uppers = [float(np.quantile(values, 1.0 - tail)) for values in nonempty]
    return {
        "lower": float(statistics.fmean(lowers)),
        "upper": float(statistics.fmean(uppers)),
        "confidence_level": 0.95,
        "resamples": int(sum(len(values) for values in nonempty)),
        "requested_resamples": int(resamples),
        "seed": int(seed),
        "replicate_unit": "stable_record",
        "method": "bag_of_little_bootstraps_percentile",
        "exact": False,
        "approximation": "deterministic_disjoint_bags_with_multinomial_resampling",
        "bag_count": len(nonempty),
        "subsample_size": int(subsample_size),
        "resample_size": int(source_records),
    }


def _blb_mean_interval(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
) -> Optional[dict[str, Any]]:
    source = np.asarray(values, dtype=np.float64)
    source = source[np.isfinite(source)]
    if not len(source):
        return None
    rng = np.random.default_rng(int(seed))
    bag_count, subsample_size, per_bag = _blb_layout(len(source), resamples)
    permutation = rng.permutation(len(source))
    results: list[np.ndarray] = []
    for bag_index in range(bag_count):
        start = bag_index * subsample_size
        indices = permutation[start : start + subsample_size]
        bag = source[indices]
        unique, counts = np.unique(bag, return_counts=True)
        samples = [
            draws @ unique / draws.sum(axis=1)
            for draws in _multinomial_chunks(
                counts,
                resamples=per_bag[bag_index],
                rng=rng,
                sample_size=len(source),
            )
        ]
        results.append(np.concatenate(samples) if samples else np.asarray([]))
    return _blb_interval_from_bags(
        results,
        resamples=resamples,
        seed=seed,
        subsample_size=subsample_size,
        source_records=len(source),
    )


def _binary_primary_interval(
    rows: Sequence[CalibrationEvidence], *, resamples: int, seed: int
) -> Optional[dict[str, Any]]:
    categories: Counter[tuple[bool, bool]] = Counter()
    for row in _primary_rows(rows):
        expected = _binary_value(row.expected)
        predicted = _binary_value(row.predicted if row.predicted is not None else row.passed)
        if expected is not None and predicted is not None:
            categories[(expected, predicted)] += 1
    if not categories:
        return None
    ordered = sorted(categories)
    counts = [categories[key] for key in ordered]
    rng = np.random.default_rng(int(seed))
    values: list[np.ndarray] = []
    for draws in _multinomial_chunks(counts, resamples=resamples, rng=rng):
        zeros = np.zeros(draws.shape[0], dtype=np.float64)

        def cell(key: tuple[bool, bool]) -> np.ndarray:
            return next(
                (
                    draws[:, index].astype(np.float64, copy=False)
                    for index, value in enumerate(ordered)
                    if value == key
                ),
                zeros,
            )

        tp = cell((True, True))
        fn = cell((True, False))
        tn = cell((False, False))
        fp = cell((False, True))
        with np.errstate(divide="ignore", invalid="ignore"):
            recall = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
            specificity = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
            values.append(np.nanmean(np.column_stack((recall, specificity)), axis=1))
    return _percentile_interval(
        np.concatenate(values),
        resamples=resamples,
        seed=seed,
        category_count=len(categories),
    )


def _macro_f1_from_draws(
    draws: np.ndarray,
    *,
    true_positive: np.ndarray,
    expected_positive: np.ndarray,
    predicted_positive: np.ndarray,
) -> np.ndarray:
    tp = draws @ true_positive
    expected = draws @ expected_positive
    predicted = draws @ predicted_positive
    denominator = expected + predicted
    with np.errstate(divide="ignore", invalid="ignore"):
        # Match the public task metric: a class with no true positives has
        # undefined F1 and is not silently converted to zero.
        per_class = np.where(tp > 0, 2.0 * tp / denominator, np.nan)
        available = np.isfinite(per_class)
        counts = available.sum(axis=1)
        totals = np.nansum(per_class, axis=1)
        return np.divide(
            totals,
            counts,
            out=np.full(draws.shape[0], np.nan, dtype=np.float64),
            where=counts > 0,
        )


def _classification_primary_interval(
    rows: Sequence[CalibrationEvidence],
    *,
    multi_label: bool,
    resamples: int,
    seed: int,
) -> Optional[dict[str, Any]]:
    categories: Counter[Any] = Counter()
    if multi_label:
        for row in _primary_rows(rows):
            expected = _label_set(row.expected)
            predicted = _label_set(row.predicted)
            if expected is not None and predicted is not None:
                categories[(tuple(sorted(expected)), tuple(sorted(predicted)))] += 1
        labels = sorted(
            {
                label
                for expected, predicted in categories
                for label in (*expected, *predicted)
            }
        )
    else:
        for row in _primary_rows(rows):
            if row.expected is not None and row.predicted is not None:
                categories[(str(row.expected), str(row.predicted))] += 1
        labels = sorted({label for pair in categories for label in pair})
    if not categories or not labels:
        return None
    ordered = sorted(categories)
    label_index = {label: index for index, label in enumerate(labels)}
    tp = np.zeros((len(ordered), len(labels)), dtype=np.float64)
    expected_positive = np.zeros_like(tp)
    predicted_positive = np.zeros_like(tp)
    for category_index, (expected, predicted) in enumerate(ordered):
        expected_values = set(expected) if multi_label else {expected}
        predicted_values = set(predicted) if multi_label else {predicted}
        for label in expected_values:
            expected_positive[category_index, label_index[label]] = 1.0
        for label in predicted_values:
            predicted_positive[category_index, label_index[label]] = 1.0
        for label in expected_values.intersection(predicted_values):
            tp[category_index, label_index[label]] = 1.0
    counts = np.asarray([categories[key] for key in ordered], dtype=np.int64)
    # Multinomial compression remains exact so long as the sufficient-statistic
    # surface is bounded.  Extremely high-cardinality multi-label sources use
    # the same documented BLB approximation as scalar/ranking calibration.
    work_categories = len(ordered) * max(1, len(labels))
    if work_categories > _EXACT_BOOTSTRAP_CATEGORY_LIMIT:
        return _classification_blb_interval(
            ordered,
            counts,
            tp,
            expected_positive,
            predicted_positive,
            resamples=resamples,
            seed=seed,
        )
    rng = np.random.default_rng(int(seed))
    values = [
        _macro_f1_from_draws(
            draws,
            true_positive=tp,
            expected_positive=expected_positive,
            predicted_positive=predicted_positive,
        )
        for draws in _multinomial_chunks(counts, resamples=resamples, rng=rng)
    ]
    return _percentile_interval(
        np.concatenate(values),
        resamples=resamples,
        seed=seed,
        category_count=len(ordered),
    )


def _classification_blb_interval(
    categories: Sequence[Any],
    counts: np.ndarray,
    tp: np.ndarray,
    expected_positive: np.ndarray,
    predicted_positive: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> Optional[dict[str, Any]]:
    record_count = int(counts.sum())
    rng = np.random.default_rng(int(seed))
    bag_count, subsample_size, per_bag = _blb_layout(record_count, resamples)
    # Sampling source category IDs without materializing record payloads keeps
    # the BLB proposal bounded even when labels themselves are large objects.
    source_categories = rng.choice(
        len(categories),
        size=bag_count * subsample_size,
        replace=False,
        p=counts / record_count,
    ) if np.all(counts == 1) else None
    if source_categories is None:
        expanded_ids = np.repeat(np.arange(len(categories)), counts)
        source_categories = rng.permutation(expanded_ids)[: bag_count * subsample_size]
    results: list[np.ndarray] = []
    for bag_index in range(bag_count):
        selected = source_categories[
            bag_index * subsample_size : (bag_index + 1) * subsample_size
        ]
        bag_counts = np.bincount(selected, minlength=len(categories))
        active = np.flatnonzero(bag_counts)
        values = [
            _macro_f1_from_draws(
                draws,
                true_positive=tp[active],
                expected_positive=expected_positive[active],
                predicted_positive=predicted_positive[active],
            )
            for draws in _multinomial_chunks(
                bag_counts[active],
                resamples=per_bag[bag_index],
                rng=rng,
                sample_size=record_count,
            )
        ]
        results.append(np.concatenate(values) if values else np.asarray([]))
    return _blb_interval_from_bags(
        results,
        resamples=resamples,
        seed=seed,
        subsample_size=subsample_size,
        source_records=record_count,
    )


def _rank_layout(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.flatnonzero(
        np.concatenate((np.asarray([True]), sorted_values[1:] != sorted_values[:-1]))
    )
    sorted_groups = np.cumsum(
        np.concatenate((np.asarray([True]), sorted_values[1:] != sorted_values[:-1]))
    ) - 1
    groups = np.empty(len(values), dtype=np.int64)
    groups[order] = sorted_groups
    return order, starts, groups


def _weighted_spearman(
    weights: np.ndarray, expected: np.ndarray, predicted: np.ndarray
) -> np.ndarray:
    sample_sizes = weights.sum(axis=1).astype(np.float64)

    def ranks(values: np.ndarray) -> np.ndarray:
        order, starts, groups = _rank_layout(values)
        group_weights = np.add.reduceat(weights[:, order], starts, axis=1)
        before = np.cumsum(group_weights, axis=1) - group_weights
        group_ranks = before + (group_weights + 1.0) / 2.0
        return group_ranks[:, groups]

    left = ranks(expected)
    right = ranks(predicted)
    means = (sample_sizes + 1.0) / 2.0
    left_centered = left - means[:, None]
    right_centered = right - means[:, None]
    covariance = np.sum(weights * left_centered * right_centered, axis=1)
    left_variance = np.sum(weights * left_centered * left_centered, axis=1)
    right_variance = np.sum(weights * right_centered * right_centered, axis=1)
    denominator = np.sqrt(left_variance * right_variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, covariance / denominator, np.nan)


def _scalar_primary_interval(
    rows: Sequence[CalibrationEvidence], *, resamples: int, seed: int
) -> Optional[dict[str, Any]]:
    pairs = _numeric_pairs(rows)
    if len(pairs) < 2:
        return None
    categories = Counter(pairs)
    ordered = sorted(categories)
    counts = np.asarray([categories[value] for value in ordered], dtype=np.int64)
    expected = np.asarray([value[0] for value in ordered], dtype=np.float64)
    predicted = np.asarray([value[1] for value in ordered], dtype=np.float64)
    if len(ordered) <= _EXACT_BOOTSTRAP_CATEGORY_LIMIT:
        rng = np.random.default_rng(int(seed))
        values = [
            _weighted_spearman(draws, expected, predicted)
            for draws in _multinomial_chunks(counts, resamples=resamples, rng=rng)
        ]
        return _percentile_interval(
            np.concatenate(values),
            resamples=resamples,
            seed=seed,
            category_count=len(ordered),
        )

    # Spearman needs ranks recomputed after every resample, so no fixed finite
    # sufficient statistic can preserve the full empirical bootstrap at
    # 100,000 continuous-valued records.  A deterministic bag of little
    # bootstraps is used instead and is explicitly identified as approximate.
    source = np.asarray(pairs, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    bag_count, subsample_size, per_bag = _blb_layout(len(source), resamples)
    permutation = rng.permutation(len(source))
    results: list[np.ndarray] = []
    for bag_index in range(bag_count):
        indices = permutation[
            bag_index * subsample_size : (bag_index + 1) * subsample_size
        ]
        bag_categories = Counter(map(tuple, source[indices].tolist()))
        bag_ordered = sorted(bag_categories)
        bag_counts = [bag_categories[value] for value in bag_ordered]
        bag_expected = np.asarray([value[0] for value in bag_ordered], dtype=np.float64)
        bag_predicted = np.asarray([value[1] for value in bag_ordered], dtype=np.float64)
        values = [
            _weighted_spearman(draws, bag_expected, bag_predicted)
            for draws in _multinomial_chunks(
                bag_counts,
                resamples=per_bag[bag_index],
                rng=rng,
                sample_size=len(source),
            )
        ]
        results.append(np.concatenate(values) if values else np.asarray([]))
    return _blb_interval_from_bags(
        results,
        resamples=resamples,
        seed=seed,
        subsample_size=subsample_size,
        source_records=len(source),
    )


def _task_primary_interval(
    rows: Sequence[CalibrationEvidence],
    *,
    task_type: str,
    resamples: int,
    seed: int,
) -> Optional[dict[str, Any]]:
    if int(resamples) <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if task_type == "binary":
        return _binary_primary_interval(rows, resamples=resamples, seed=seed)
    if task_type == "categorical":
        return _classification_primary_interval(
            rows, multi_label=False, resamples=resamples, seed=seed
        )
    if task_type == "multi_label":
        return _classification_primary_interval(
            rows, multi_label=True, resamples=resamples, seed=seed
        )
    if task_type == "scalar":
        return _scalar_primary_interval(rows, resamples=resamples, seed=seed)
    if task_type == "pairwise":
        values = [
            float(str(row.expected) == str(row.predicted))
            for row in _primary_rows(rows)
            if row.expected is not None and row.predicted is not None
        ]
        return _compressed_mean_interval(values, resamples=resamples, seed=seed)
    values: list[float] = []
    for row in _primary_rows(rows):
        expected = _ranking_items(row.expected)
        predicted = _ranking_items(row.predicted)
        if expected is not None and predicted is not None:
            metric = _ranking_record_metrics(expected, predicted)
            if metric is not None:
                values.append(float(metric["kendall_tau"]))
    return _compressed_mean_interval(values, resamples=resamples, seed=seed)


_SUBGROUP_MIN_RECORDS = 20
_SUBGROUP_EXACT_CARDINALITY = 1_000
_SUBGROUP_CANDIDATE_CAPACITY = 10_000


def _subgroup_metric_rows(
    rows: Sequence[CalibrationEvidence],
) -> tuple[
    dict[tuple[str, str], list[CalibrationEvidence]],
    dict[tuple[str, str], int],
    dict[str, Any],
]:
    """Find eligible subgroup slices without retaining unbounded unique keys.

    Low-cardinality inputs preserve the prior behavior, including reporting
    groups below the 20-record threshold.  Once cardinality exceeds 1,000, a
    fixed-capacity Misra-Gries pass identifies possible frequent groups and an
    exact second pass validates them.  With up to 190,000 subgroup
    contributions this is guaranteed to retain every group meeting the
    20-record threshold; larger adversarial metadata is reported as truncated
    instead of silently claiming exhaustive subgroup coverage.
    """

    # Count distinct reference records even when verifier evidence is missing;
    # subgroup availability should not disappear merely because all calls for
    # a record failed.  Task metrics below still consume canonical available
    # primary observations, matching the pre-bounded behavior.
    by_record: dict[str, CalibrationEvidence] = {}
    for row in rows:
        by_record.setdefault(row.record_id, row)
    stable_rows = [by_record[key] for key in sorted(by_record)]
    source_record_count = len(stable_rows)
    exact_counts: dict[tuple[str, str], int] = {}
    candidates: dict[tuple[str, str], int] = {}
    high_cardinality = False
    contribution_count = 0

    for row in stable_rows:
        for key, value in row.subgroup.items():
            pair = (str(key), str(value))
            contribution_count += 1
            if not high_cardinality:
                exact_counts[pair] = exact_counts.get(pair, 0) + 1
                if len(exact_counts) <= _SUBGROUP_EXACT_CARDINALITY:
                    continue
                candidates = exact_counts
                exact_counts = {}
                high_cardinality = True
                continue
            if pair in candidates:
                candidates[pair] += 1
            elif len(candidates) < _SUBGROUP_CANDIDATE_CAPACITY:
                candidates[pair] = 1
            else:
                for candidate in list(candidates):
                    remaining = candidates[candidate] - 1
                    if remaining:
                        candidates[candidate] = remaining
                    else:
                        del candidates[candidate]

    if high_cardinality:
        candidate_keys = set(candidates)
        verified_counts = {key: 0 for key in candidate_keys}
        for row in stable_rows:
            for key, value in row.subgroup.items():
                pair = (str(key), str(value))
                if pair in verified_counts:
                    verified_counts[pair] += 1
        report_counts = {
            key: count
            for key, count in verified_counts.items()
            if count >= _SUBGROUP_MIN_RECORDS
        }
    else:
        report_counts = exact_counts

    del stable_rows, by_record
    primary = _primary_rows(rows)

    eligible = {
        key for key, count in report_counts.items() if count >= _SUBGROUP_MIN_RECORDS
    }
    grouped_rows: dict[tuple[str, str], list[CalibrationEvidence]] = {
        key: [] for key in eligible
    }
    if eligible:
        for row in primary:
            for key, value in row.subgroup.items():
                pair = (str(key), str(value))
                if pair in grouped_rows:
                    grouped_rows[pair].append(row)

    guaranteed_through = _SUBGROUP_CANDIDATE_CAPACITY * (
        _SUBGROUP_MIN_RECORDS - 1
    )
    analysis = {
        "minimum_distinct_records": _SUBGROUP_MIN_RECORDS,
        "source_record_count": source_record_count,
        "contribution_count": contribution_count,
        "high_cardinality": high_cardinality,
        "tracked_candidate_count": (
            len(candidates) if high_cardinality else len(exact_counts)
        ),
        "reported_group_count": len(report_counts),
        "ineligible_groups_omitted": high_cardinality,
        "eligible_group_detection_exhaustive": (
            not high_cardinality or contribution_count <= guaranteed_through
        ),
    }
    if high_cardinality and contribution_count > guaranteed_through:
        analysis["warning"] = (
            "subgroup contribution cardinality exceeded the bounded exact "
            "frequent-group guarantee"
        )
    return grouped_rows, report_counts, analysis


def compute_calibration_metrics(
    evidence: Sequence[CalibrationEvidence | Mapping[str, Any] | Any],
    *,
    task_type: str,
    reward_contract: RewardContract | Mapping[str, Any] | Any,
    probability_semantics: Optional[bool] = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    rows = [normalize_evidence(value) for value in evidence]
    if not rows:
        raise ValueError("calibration metrics require evidence")
    task = str(task_type).strip().lower()
    contract = normalize_reward_contract(reward_contract)
    task_metrics = compute_task_metrics(rows, task_type=task, reward_contract=contract)
    primary_name = PRIMARY_METRIC_BY_TASK[task]
    primary_value = task_metrics.get(primary_name)
    interval = _task_primary_interval(
        rows,
        task_type=task,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    probability_enabled = (
        contract.probability_semantics
        if probability_semantics is None
        else bool(probability_semantics)
    )

    subgroup_values, subgroup_counts, subgroup_analysis = _subgroup_metric_rows(rows)
    subgroups: dict[str, Any] = {}
    for (key, value), record_count in sorted(subgroup_counts.items()):
        group_key = f"{key}={value}"
        if record_count < _SUBGROUP_MIN_RECORDS:
            subgroups[group_key] = {
                "available": False,
                "record_count": record_count,
                "reason": "subgroup_requires_20_distinct_records",
            }
        else:
            subgroup_rows = subgroup_values[(key, value)]
            subgroups[group_key] = {
                "available": True,
                "record_count": record_count,
                "metrics": compute_task_metrics(
                    subgroup_rows, task_type=task, reward_contract=contract
                ),
            }

    result = {
        "task_type": task,
        "universal": _universal_metrics(rows, contract),
        "task": task_metrics,
        "primary_metric": {
            "name": primary_name,
            "value": primary_value,
            "direction": metric_direction(primary_name),
        },
        "primary_metric_interval": interval,
        "bootstrap": {
            "method": (
                interval.get("method", "grouped_percentile")
                if interval
                else "grouped_percentile_unavailable"
            ),
            "replicate_unit": "stable_record",
            "resamples": int(bootstrap_resamples),
            "seed": int(bootstrap_seed),
            "confidence_level": 0.95,
            "exact": interval.get("exact") if interval else None,
            **(
                {
                    key: interval[key]
                    for key in (
                        "approximation",
                        "bag_count",
                        "subsample_size",
                        "resample_size",
                        "category_count",
                    )
                    if key in interval
                }
                if interval
                else {}
            ),
        },
        "probability_semantics": probability_enabled,
        "subgroups": subgroups,
        "subgroup_analysis": subgroup_analysis,
    }
    if probability_enabled:
        result["probability"] = _probability_metrics(rows)
    return result


__all__ = [
    "CalibrationEvidence",
    "PRIMARY_METRIC_BY_TASK",
    "TASK_TYPES",
    "compute_calibration_metrics",
    "compute_task_metrics",
    "grouped_percentile_bootstrap",
    "metric_direction",
    "normalize_evidence",
]
