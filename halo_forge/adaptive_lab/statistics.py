"""Deterministic cohort summaries and matched-seed uncertainty analysis."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .models import (
    CohortAnalysisSnapshot,
    CohortObservation,
    EvidenceCompatibility,
    METRIC_DIRECTIONS,
)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def percentile_bootstrap_interval(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
    resamples: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Return a deterministic percentile-bootstrap interval for the mean."""

    sample = tuple(float(value) for value in values)
    if not sample:
        raise ValueError("bootstrap values cannot be empty")
    if not 0.0 < float(confidence_level) < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if int(resamples) <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if any(not math.isfinite(value) for value in sample):
        raise ValueError("bootstrap values must be finite")
    if len(sample) == 1:
        return sample[0], sample[0]
    rng = random.Random(int(seed))
    count = len(sample)
    means = [
        sum(sample[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(int(resamples))
    ]
    means.sort()
    tail = (1.0 - float(confidence_level)) / 2.0
    return _percentile(means, tail), _percentile(means, 1.0 - tail)


def _normalize_observation(value: CohortObservation | Mapping[str, Any]) -> CohortObservation:
    if isinstance(value, CohortObservation):
        return value
    return CohortObservation(
        subject_id=value["subject_id"],
        seed=value["seed"],
        metric=value["metric"],
        value=value["value"],
        evaluation_id=value.get("evaluation_id"),
        metadata=value.get("metadata") or {},
    )


def _normalize_compatibility(
    value: EvidenceCompatibility | Mapping[str, Any],
) -> EvidenceCompatibility:
    if isinstance(value, EvidenceCompatibility):
        return value
    return EvidenceCompatibility.from_dict(value)


def _numeric_metadata_means(rows: Sequence[CohortObservation]) -> Dict[str, float]:
    values: Dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.metadata.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            result = float(value)
            if math.isfinite(result):
                values[str(key)].append(result)
    return {key: statistics.fmean(items) for key, items in sorted(values.items()) if items}


def _summary(
    rows: Sequence[CohortObservation],
    *,
    required_seeds: Sequence[int],
    confidence_level: float,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    values = [row.value for row in rows]
    seeds = [row.seed for row in rows]
    interval = percentile_bootstrap_interval(
        values,
        confidence_level=confidence_level,
        resamples=resamples,
        seed=seed,
    )
    required = tuple(sorted({int(value) for value in required_seeds}))
    observed = set(seeds)
    return {
        "count": len(values),
        "seeds": sorted(seeds),
        "required_seeds": list(required),
        "complete_seed_coverage": not required or set(required).issubset(observed),
        "missing_seeds": [value for value in required if value not in observed],
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_deviation": statistics.stdev(values) if len(values) >= 2 else None,
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
        "confidence_interval": {"lower": interval[0], "upper": interval[1]},
        "evaluation_ids": sorted(
            {row.evaluation_id for row in rows if row.evaluation_id is not None}
        ),
        "metadata_means": _numeric_metadata_means(rows),
    }


def classify_directional_interval(
    lower: float,
    upper: float,
    *,
    practical_delta: float,
    equivalence_delta: Optional[float] = None,
) -> str:
    """Classify a direction-normalized interval using practical boundaries."""

    practical = float(practical_delta)
    equivalence = practical if equivalence_delta is None else float(equivalence_delta)
    if practical < 0 or equivalence < 0:
        raise ValueError("practical and equivalence deltas must be non-negative")
    if lower > upper:
        raise ValueError("interval lower bound cannot exceed upper bound")
    if lower > practical:
        return "improved"
    if upper < -practical:
        return "regressed"
    if lower >= -equivalence and upper <= equivalence:
        return "practically_equivalent"
    return "inconclusive"


def build_cohort_snapshot(
    observations: Iterable[CohortObservation | Mapping[str, Any]],
    *,
    metric: str,
    direction: str,
    baseline_subject_id: Optional[str] = None,
    confidence_level: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 42,
    practical_delta: float = 0.0,
    equivalence_delta: Optional[float] = None,
    required_seeds: Sequence[int] = (),
    evidence_compatibility: Sequence[EvidenceCompatibility | Mapping[str, Any]] = (),
    context: Optional[Mapping[str, Any]] = None,
) -> CohortAnalysisSnapshot:
    """Create a content-addressed analysis over seed-level observations.

    Seed-level values are the replicates. Candidate comparisons use only seeds
    also present in the baseline and never substitute record-level samples.
    """

    normalized_metric = str(metric).strip()
    normalized_direction = str(direction).strip().lower()
    if not normalized_metric:
        raise ValueError("metric is required")
    if normalized_direction not in METRIC_DIRECTIONS:
        raise ValueError("direction must be maximize or minimize")
    confidence_level = float(confidence_level)
    bootstrap_resamples = int(bootstrap_resamples)
    bootstrap_seed = int(bootstrap_seed)
    practical_delta = float(practical_delta)
    equivalence_delta = practical_delta if equivalence_delta is None else float(equivalence_delta)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if practical_delta < 0 or equivalence_delta < 0:
        raise ValueError("practical and equivalence deltas must be non-negative")

    rows = tuple(_normalize_observation(value) for value in observations)
    if not rows:
        raise ValueError("at least one cohort observation is required")
    if any(row.metric != normalized_metric for row in rows):
        raise ValueError("all observations must use the requested metric")
    by_subject: Dict[str, Dict[int, CohortObservation]] = defaultdict(dict)
    for row in rows:
        if row.seed in by_subject[row.subject_id]:
            raise ValueError(
                f"duplicate observation for subject {row.subject_id!r}, seed {row.seed}"
            )
        by_subject[row.subject_id][row.seed] = row
    if baseline_subject_id is not None and baseline_subject_id not in by_subject:
        raise ValueError(f"unknown baseline subject {baseline_subject_id!r}")
    compatibility_by_subject: Dict[str, EvidenceCompatibility] = {}
    for value in evidence_compatibility:
        compatibility = _normalize_compatibility(value)
        if compatibility.subject_id in compatibility_by_subject:
            raise ValueError(f"duplicate evidence compatibility for {compatibility.subject_id!r}")
        if compatibility.subject_id not in by_subject:
            raise ValueError(
                f"evidence compatibility has unknown subject {compatibility.subject_id!r}"
            )
        compatibility_by_subject[compatibility.subject_id] = compatibility

    subjects: Dict[str, Any] = {}
    for index, subject_id in enumerate(sorted(by_subject)):
        subject_rows = [by_subject[subject_id][seed] for seed in sorted(by_subject[subject_id])]
        subjects[subject_id] = _summary(
            subject_rows,
            required_seeds=required_seeds,
            confidence_level=confidence_level,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed + index * 1_000_003,
        )
        subjects[subject_id]["evidence_compatibility"] = (
            None
            if subject_id not in compatibility_by_subject
            else compatibility_by_subject[subject_id].to_dict()
        )

    comparisons: Dict[str, Any] = {}
    if baseline_subject_id is not None:
        baseline = by_subject[baseline_subject_id]
        sign = 1.0 if normalized_direction == "maximize" else -1.0
        comparison_index = 0
        for candidate_id in sorted(by_subject):
            if candidate_id == baseline_subject_id:
                continue
            candidate = by_subject[candidate_id]
            matched_seeds = sorted(set(baseline).intersection(candidate))
            deltas = [
                sign * (candidate[seed].value - baseline[seed].value) for seed in matched_seeds
            ]
            required = set(int(value) for value in required_seeds)
            complete = not required or required.issubset(matched_seeds)
            comparison: Dict[str, Any] = {
                "baseline_subject_id": baseline_subject_id,
                "candidate_subject_id": candidate_id,
                "matched_seeds": matched_seeds,
                "matched_seed_count": len(matched_seeds),
                "complete_required_seed_coverage": complete,
                "missing_required_seeds": sorted(required.difference(matched_seeds)),
                "direction_normalized_deltas": deltas,
            }
            mismatch_fields: Tuple[str, ...] = ()
            compatibility_reason: Optional[str] = None
            if compatibility_by_subject:
                baseline_compatibility = compatibility_by_subject.get(baseline_subject_id)
                candidate_compatibility = compatibility_by_subject.get(candidate_id)
                if baseline_compatibility is None or candidate_compatibility is None:
                    compatibility_reason = "missing_compatibility_evidence"
                else:
                    mismatch_fields = baseline_compatibility.mismatch_fields(
                        candidate_compatibility
                    )
                    if mismatch_fields:
                        compatibility_reason = "incompatible_evidence"
            comparison["evidence_compatible"] = compatibility_reason is None
            comparison["compatibility_mismatches"] = list(mismatch_fields)
            if compatibility_reason is not None:
                comparison.update(
                    classification="insufficient_evidence",
                    reason=compatibility_reason,
                    mean_delta=statistics.fmean(deltas) if deltas else None,
                    confidence_interval=None,
                )
            elif len(deltas) < 2 or not complete:
                comparison.update(
                    classification="insufficient_evidence",
                    reason=(
                        "fewer_than_two_matched_seeds"
                        if len(deltas) < 2
                        else "incomplete_required_seed_coverage"
                    ),
                    mean_delta=statistics.fmean(deltas) if deltas else None,
                    confidence_interval=None,
                )
            else:
                interval = percentile_bootstrap_interval(
                    deltas,
                    confidence_level=confidence_level,
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed + 10_000_019 + comparison_index * 1_000_003,
                )
                comparison.update(
                    classification=classify_directional_interval(
                        interval[0],
                        interval[1],
                        practical_delta=practical_delta,
                        equivalence_delta=equivalence_delta,
                    ),
                    reason=None,
                    mean_delta=statistics.fmean(deltas),
                    median_delta=statistics.median(deltas),
                    sample_deviation=statistics.stdev(deltas),
                    confidence_interval={"lower": interval[0], "upper": interval[1]},
                )
            comparisons[candidate_id] = comparison
            comparison_index += 1

    request = {
        "metric": normalized_metric,
        "direction": normalized_direction,
        "baseline_subject_id": baseline_subject_id,
        "confidence_level": confidence_level,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "practical_delta": practical_delta,
        "equivalence_delta": equivalence_delta,
        "required_seeds": sorted({int(value) for value in required_seeds}),
        "evidence_compatibility": [
            value.to_dict()
            for value in sorted(compatibility_by_subject.values(), key=lambda item: item.subject_id)
        ],
        "observations": [
            row.to_dict() for row in sorted(rows, key=lambda r: (r.subject_id, r.seed))
        ],
        "context": dict(context or {}),
    }
    analysis = {
        "replicate_unit": "seed",
        "subjects": subjects,
        "comparisons": comparisons,
    }
    return CohortAnalysisSnapshot(request=request, analysis=analysis)


__all__ = [
    "build_cohort_snapshot",
    "classify_directional_interval",
    "percentile_bootstrap_interval",
]
