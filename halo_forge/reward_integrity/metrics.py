"""Deterministic paired metrics for reward-integrity audits."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .models import RewardIntegrityMetric


@dataclass(frozen=True)
class RewardScale:
    minimum: float = 0.0
    maximum: float = 1.0
    direction: str = "maximize"

    @classmethod
    def from_value(cls, value: Any) -> "RewardScale":
        if isinstance(value, cls):
            return value
        if hasattr(value, "minimum") and hasattr(value, "maximum"):
            return cls(
                minimum=float(value.minimum),
                maximum=float(value.maximum),
                direction=str(getattr(value, "direction", "maximize")),
            )
        raw = dict(value or {})
        return cls(
            minimum=float(raw.get("minimum", raw.get("min", 0.0))),
            maximum=float(raw.get("maximum", raw.get("max", 1.0))),
            direction=str(raw.get("direction", "maximize")),
        )

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("reward bounds must be finite")
        if self.maximum <= self.minimum:
            raise ValueError("reward maximum must be greater than minimum")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("reward direction must be maximize or minimize")


@dataclass(frozen=True)
class IntegrityEvidence:
    snapshot_id: str
    group_id: str
    optimizer_reward: Optional[float]
    sentinel_reward: Optional[float]
    optimizer_passed: Optional[bool]
    sentinel_passed: Optional[bool]
    optimizer_error: Optional[str] = None
    sentinel_error: Optional[str] = None
    diagnostic: bool = False
    subgroup: str = ""
    component_disagreement: Optional[bool] = None


def normalize_reward(value: Optional[float], contract: Any) -> Optional[float]:
    """Normalize a finite in-contract reward to [0, 1], higher-is-better."""

    if value is None:
        return None
    scale = RewardScale.from_value(contract)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("reward must be finite")
    if number < scale.minimum or number > scale.maximum:
        raise ValueError(f"reward {number} is outside [{scale.minimum}, {scale.maximum}]")
    result = (number - scale.minimum) / (scale.maximum - scale.minimum)
    if scale.direction == "minimize":
        result = 1.0 - result
    return min(1.0, max(0.0, result))


def _rank(values: Sequence[float]) -> List[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            result[ordered[position]] = average
        cursor = end
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    covariance = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    variance_left = sum((x - mean_left) ** 2 for x in left)
    variance_right = sum((y - mean_right) ** 2 for y in right)
    denominator = math.sqrt(variance_left * variance_right)
    return None if denominator == 0 else covariance / denominator


def spearman_correlation(
    left: Sequence[float], right: Sequence[float], *, minimum_levels: int = 5
) -> Optional[float]:
    """Tie-aware Spearman correlation, unavailable for threshold-like scores."""

    if len(left) != len(right) or len(left) < 2:
        return None
    if len(set(left)) < minimum_levels or len(set(right)) < minimum_levels:
        return None
    return _pearson(_rank(left), _rank(right))


def kendall_tau(
    left: Sequence[float], right: Sequence[float], *, minimum_levels: int = 5
) -> Optional[float]:
    """Tie-aware Kendall tau-b for paired continuous reward evidence."""

    if len(left) != len(right) or len(left) < 2:
        return None
    if len(set(left)) < minimum_levels or len(set(right)) < minimum_levels:
        return None
    concordant = discordant = ties_left = ties_right = 0
    for first in range(len(left) - 1):
        for second in range(first + 1, len(left)):
            left_delta = left[first] - left[second]
            right_delta = right[first] - right[second]
            if left_delta == 0 and right_delta == 0:
                continue
            if left_delta == 0:
                ties_left += 1
            elif right_delta == 0:
                ties_right += 1
            elif left_delta * right_delta > 0:
                concordant += 1
            else:
                discordant += 1
    comparable = concordant + discordant
    denominator = math.sqrt(
        (comparable + ties_left) * (comparable + ties_right)
    )
    return None if denominator == 0 else (concordant - discordant) / denominator


def _percentile(sorted_values: Sequence[float], proportion: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * proportion
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def grouped_percentile_bootstrap(
    evidence: Sequence[IntegrityEvidence],
    statistic: Callable[[Sequence[IntegrityEvidence]], Optional[float]],
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Bootstrap stable record groups; candidates are never replicate units."""

    if resamples <= 0:
        return None, None
    grouped: Dict[str, List[IntegrityEvidence]] = {}
    for item in evidence:
        grouped.setdefault(item.group_id, []).append(item)
    keys = sorted(grouped)
    if not keys:
        return None, None
    rng = random.Random(seed)
    estimates: List[float] = []
    for _ in range(int(resamples)):
        sampled: List[IntegrityEvidence] = []
        for _position in keys:
            sampled.extend(grouped[keys[rng.randrange(len(keys))]])
        value = statistic(sampled)
        if value is not None and math.isfinite(float(value)):
            estimates.append(float(value))
    if not estimates:
        return None, None
    estimates.sort()
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _paired(
    evidence: Sequence[IntegrityEvidence], optimizer: RewardScale, sentinel: RewardScale
) -> List[Tuple[IntegrityEvidence, float, float]]:
    paired: List[Tuple[IntegrityEvidence, float, float]] = []
    for item in evidence:
        if item.optimizer_error or item.sentinel_error:
            continue
        try:
            left = normalize_reward(item.optimizer_reward, optimizer)
            right = normalize_reward(item.sentinel_reward, sentinel)
        except ValueError:
            continue
        if left is not None and right is not None:
            paired.append((item, left, right))
    return paired


def _point_statistics(
    evidence: Sequence[IntegrityEvidence], optimizer: RewardScale, sentinel: RewardScale
) -> Dict[str, Optional[float]]:
    total = len(evidence)
    paired = _paired(evidence, optimizer, sentinel)
    pass_pairs = [
        item
        for item in evidence
        if not item.optimizer_error
        and not item.sentinel_error
        and item.optimizer_passed is not None
        and item.sentinel_passed is not None
    ]
    left = [item[1] for item in paired]
    right = [item[2] for item in paired]
    result: Dict[str, Optional[float]] = {
        "paired_coverage": len(paired) / total if total else None,
        "optimizer_error_rate": (
            sum(bool(item.optimizer_error) for item in evidence) / total if total else None
        ),
        "sentinel_error_rate": (
            sum(bool(item.sentinel_error) for item in evidence) / total if total else None
        ),
        "pass_agreement": (
            sum(item.optimizer_passed == item.sentinel_passed for item in pass_pairs)
            / len(pass_pairs)
            if pass_pairs
            else None
        ),
        "optimizer_only_acceptance": (
            sum(
                item.optimizer_passed is True and item.sentinel_passed is False
                for item in pass_pairs
            )
            / len(pass_pairs)
            if pass_pairs
            else None
        ),
        "sentinel_only_acceptance": (
            sum(
                item.optimizer_passed is False and item.sentinel_passed is True
                for item in pass_pairs
            )
            / len(pass_pairs)
            if pass_pairs
            else None
        ),
        "pass_flip_rate": (
            sum(item.optimizer_passed != item.sentinel_passed for item in pass_pairs)
            / len(pass_pairs)
            if pass_pairs
            else None
        ),
        "optimizer_reward_mean": sum(left) / len(left) if left else None,
        "sentinel_reward_mean": sum(right) / len(right) if right else None,
        "mean_reward_gap": (sum(x - y for x, y in zip(left, right)) / len(left) if left else None),
        "absolute_mean_reward_gap": (
            abs(sum(x - y for x, y in zip(left, right)) / len(left)) if left else None
        ),
        "spearman": spearman_correlation(left, right),
        "kendall_tau": kendall_tau(left, right),
        "optimizer_saturation_rate": (
            sum(value <= 0.01 or value >= 0.99 for value in left) / len(left) if left else None
        ),
        "sentinel_saturation_rate": (
            sum(value <= 0.01 or value >= 0.99 for value in right) / len(right) if right else None
        ),
        "chain_component_disagreement_rate": (
            sum(item.component_disagreement is True for item in evidence)
            / sum(item.component_disagreement is not None for item in evidence)
            if any(item.component_disagreement is not None for item in evidence)
            else None
        ),
    }
    if paired:
        count = max(1, int(math.ceil(len(paired) * 0.1)))
        optimizer_top = {
            item[0].snapshot_id
            for item in sorted(paired, key=lambda value: (-value[1], value[0].snapshot_id))[:count]
        }
        sentinel_top = {
            item[0].snapshot_id
            for item in sorted(paired, key=lambda value: (-value[2], value[0].snapshot_id))[:count]
        }
        result["top_tail_disagreement"] = len(optimizer_top - sentinel_top) / len(optimizer_top)
    else:
        result["top_tail_disagreement"] = None
    return result


_DIRECTIONS = {
    "paired_coverage": "maximize",
    "optimizer_error_rate": "minimize",
    "sentinel_error_rate": "minimize",
    "pass_agreement": "maximize",
    "optimizer_only_acceptance": "minimize",
    "sentinel_only_acceptance": "minimize",
    "pass_flip_rate": "minimize",
    "spearman": "maximize",
    "kendall_tau": "maximize",
    "absolute_mean_reward_gap": "minimize",
    "top_tail_disagreement": "minimize",
    "optimizer_saturation_rate": "minimize",
    "sentinel_saturation_rate": "minimize",
    "chain_component_disagreement_rate": "minimize",
}


def compute_integrity_metrics(
    audit_id: str,
    evidence: Sequence[IntegrityEvidence],
    optimizer_contract: Any,
    sentinel_contract: Any,
    *,
    bootstrap_resamples: int = 10_000,
    seed: int = 42,
    include_diagnostics: bool = False,
) -> List[RewardIntegrityMetric]:
    """Compute paired core-population metrics with grouped intervals.

    Diagnostic strata are deliberately excluded unless explicitly requested;
    they were selected conditionally and cannot estimate population rates.
    """

    source = (
        list(evidence)
        if include_diagnostics
        else [item for item in evidence if not item.diagnostic]
    )
    optimizer = RewardScale.from_value(optimizer_contract)
    sentinel = RewardScale.from_value(sentinel_contract)
    point = _point_statistics(source, optimizer, sentinel)
    record_count = len({item.group_id for item in source})
    results: List[RewardIntegrityMetric] = []
    for name, value in point.items():
        if value is None:
            reason = (
                "requires_at_least_five_reward_levels"
                if name in {"spearman", "kendall_tau"}
                else "missing_paired_evidence"
            )
            results.append(
                RewardIntegrityMetric(
                    audit_id=audit_id,
                    name=name,
                    value=None,
                    available=False,
                    missing_reason=reason,
                    record_count=record_count,
                    direction=_DIRECTIONS.get(name),
                )
            )
            continue

        def statistic(
            sample: Sequence[IntegrityEvidence], metric_name: str = name
        ) -> Optional[float]:
            return _point_statistics(sample, optimizer, sentinel).get(metric_name)

        low, high = grouped_percentile_bootstrap(
            source, statistic, resamples=bootstrap_resamples, seed=seed
        )
        results.append(
            RewardIntegrityMetric(
                audit_id=audit_id,
                name=name,
                value=float(value),
                available=True,
                record_count=record_count,
                ci_low=low,
                ci_high=high,
                direction=_DIRECTIONS.get(name),
                metadata={"bootstrap_resamples": int(bootstrap_resamples), "seed": int(seed)},
            )
        )
    return results


__all__ = [
    "IntegrityEvidence",
    "RewardScale",
    "compute_integrity_metrics",
    "grouped_percentile_bootstrap",
    "kendall_tau",
    "normalize_reward",
    "spearman_correlation",
]
