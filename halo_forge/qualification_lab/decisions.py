"""Pure qualification gates, comparisons, Pareto ranking, and promotion policy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ._canonical import FrozenJsonMap, content_fingerprint
from .profiles import METRIC_DIRECTIONS, QualificationMetricRule, QualificationProfileRevision

QUALIFICATION_STATUSES = frozenset({"pass", "warn", "fail", "not_required"})
_STATUS_RANK = {"not_required": 0, "pass": 0, "warn": 1, "fail": 2}


def _normalize_metric_map(
    metrics: FrozenJsonMap | Mapping[str, Optional[float]], *, name: str
) -> FrozenJsonMap:
    normalized = metrics if isinstance(metrics, FrozenJsonMap) else FrozenJsonMap(metrics)
    for metric, value in normalized.items():
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}.{metric} must be a number or None")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name}.{metric} must be finite")
    return normalized


def _worst_status(statuses: Sequence[str]) -> str:
    return max(statuses, key=lambda status: _STATUS_RANK[status], default="pass")


@dataclass(frozen=True)
class QualificationEvidence:
    """Pinned stage metrics for one artifact under one profile revision."""

    artifact_hash: str
    profile_content_hash: str
    development_metrics: FrozenJsonMap = field(default_factory=FrozenJsonMap)
    operational_metrics: FrozenJsonMap = field(default_factory=FrozenJsonMap)
    holdout_metrics: FrozenJsonMap = field(default_factory=FrozenJsonMap)
    development_complete: bool = True
    operational_complete: bool = True
    holdout_complete: bool = False

    def __post_init__(self) -> None:
        artifact_hash = str(self.artifact_hash).strip()
        profile_hash = str(self.profile_content_hash).strip()
        if not artifact_hash:
            raise ValueError("artifact_hash cannot be empty")
        if not profile_hash:
            raise ValueError("profile_content_hash cannot be empty")
        object.__setattr__(self, "artifact_hash", artifact_hash)
        object.__setattr__(self, "profile_content_hash", profile_hash)
        for name in ("development_metrics", "operational_metrics", "holdout_metrics"):
            object.__setattr__(
                self,
                name,
                _normalize_metric_map(getattr(self, name), name=name),
            )
        for name in ("development_complete", "operational_complete", "holdout_complete"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_hash": self.artifact_hash,
            "profile_content_hash": self.profile_content_hash,
            "development_metrics": self.development_metrics.to_dict(),
            "operational_metrics": self.operational_metrics.to_dict(),
            "holdout_metrics": self.holdout_metrics.to_dict(),
            "development_complete": self.development_complete,
            "operational_complete": self.operational_complete,
            "holdout_complete": self.holdout_complete,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QualificationEvidence":
        return cls(**dict(payload))


@dataclass(frozen=True)
class QualificationMetricDecision:
    stage: str
    metric: str
    direction: str
    status: str
    candidate_value: Optional[float]
    baseline_value: Optional[float]
    raw_delta: Optional[float]
    favorable_delta: Optional[float]
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "metric": self.metric,
            "direction": self.direction,
            "status": self.status,
            "candidate_value": self.candidate_value,
            "baseline_value": self.baseline_value,
            "raw_delta": self.raw_delta,
            "favorable_delta": self.favorable_delta,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class QualificationStageDecision:
    stage: str
    status: str
    complete: bool
    metrics: Tuple[QualificationMetricDecision, ...]
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "complete": self.complete,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ArtifactQualification:
    profile_content_hash: str
    candidate_artifact_hash: str
    parent_artifact_hash: Optional[str]
    development: QualificationStageDecision
    operational: QualificationStageDecision
    holdout: QualificationStageDecision
    overall_status: str
    holdout_required: bool
    reasons: Tuple[str, ...]
    decision_hash: str = field(init=False)

    def __post_init__(self) -> None:
        identity = {
            "profile_content_hash": self.profile_content_hash,
            "candidate_artifact_hash": self.candidate_artifact_hash,
            "parent_artifact_hash": self.parent_artifact_hash,
            "development": self.development.to_dict(),
            "operational": self.operational.to_dict(),
            "holdout": self.holdout.to_dict(),
            "overall_status": self.overall_status,
            "holdout_required": self.holdout_required,
            "reasons": list(self.reasons),
        }
        object.__setattr__(self, "decision_hash", content_fingerprint(identity))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_content_hash": self.profile_content_hash,
            "candidate_artifact_hash": self.candidate_artifact_hash,
            "parent_artifact_hash": self.parent_artifact_hash,
            "development": self.development.to_dict(),
            "operational": self.operational.to_dict(),
            "holdout": self.holdout.to_dict(),
            "overall_status": self.overall_status,
            "holdout_required": self.holdout_required,
            "reasons": list(self.reasons),
            "decision_hash": self.decision_hash,
        }


# Descriptive compatibility name for callers focused on the decision step.
QualificationDecision = ArtifactQualification


def ensure_same_profile(
    profile: QualificationProfileRevision,
    *evidence: QualificationEvidence,
) -> None:
    mismatched = [
        item.profile_content_hash
        for item in evidence
        if item.profile_content_hash != profile.content_hash
    ]
    if mismatched:
        raise ValueError(
            "qualification evidence can only be compared under the same profile revision"
        )


def _metric_decision(
    *,
    stage: str,
    rule: QualificationMetricRule,
    candidate_value: Optional[float],
    baseline_value: Optional[float],
    stage_complete: bool,
    pending_status: str,
) -> QualificationMetricDecision:
    raw_delta = None
    favorable_delta = None
    if candidate_value is not None and baseline_value is not None:
        raw_delta = float(candidate_value) - float(baseline_value)
        favorable_delta = raw_delta if rule.direction == "maximize" else -raw_delta

    if not stage_complete:
        return QualificationMetricDecision(
            stage=stage,
            metric=rule.metric,
            direction=rule.direction,
            status=pending_status,
            candidate_value=candidate_value,
            baseline_value=baseline_value,
            raw_delta=raw_delta,
            favorable_delta=favorable_delta,
            reasons=(f"{stage} evidence is incomplete",),
        )
    if candidate_value is None:
        status = "fail" if rule.required else "warn"
        return QualificationMetricDecision(
            stage=stage,
            metric=rule.metric,
            direction=rule.direction,
            status=status,
            candidate_value=None,
            baseline_value=baseline_value,
            raw_delta=None,
            favorable_delta=None,
            reasons=(
                (
                    f"required metric {rule.metric} is missing"
                    if rule.required
                    else f"optional metric {rule.metric} is unavailable"
                ),
            ),
        )

    statuses = ["pass"]
    reasons = []
    if rule.pass_threshold is not None:
        passes = (
            candidate_value >= rule.pass_threshold
            if rule.direction == "maximize"
            else candidate_value <= rule.pass_threshold
        )
        if not passes:
            warns = False
            if rule.warn_threshold is not None:
                warns = (
                    candidate_value >= rule.warn_threshold
                    if rule.direction == "maximize"
                    else candidate_value <= rule.warn_threshold
                )
            if warns:
                statuses.append("warn")
                reasons.append(
                    f"{rule.metric} missed pass threshold {rule.pass_threshold} "
                    f"but met warning threshold {rule.warn_threshold}"
                )
            else:
                statuses.append("fail")
                reasons.append(
                    f"{rule.metric} failed {rule.direction} threshold {rule.pass_threshold}"
                )
    if rule.maximum_regression is not None:
        if baseline_value is None:
            statuses.append("fail" if rule.required else "warn")
            reasons.append(f"{rule.metric} has no parent value for regression comparison")
        elif favorable_delta is not None and favorable_delta < -rule.maximum_regression:
            statuses.append("fail")
            reasons.append(
                f"{rule.metric} regression {abs(favorable_delta):g} exceeded allowed "
                f"{rule.maximum_regression:g}"
            )
    return QualificationMetricDecision(
        stage=stage,
        metric=rule.metric,
        direction=rule.direction,
        status=_worst_status(statuses),
        candidate_value=float(candidate_value),
        baseline_value=None if baseline_value is None else float(baseline_value),
        raw_delta=raw_delta,
        favorable_delta=favorable_delta,
        reasons=tuple(reasons),
    )


def _stage_decision(
    *,
    stage: str,
    rules: Sequence[QualificationMetricRule],
    candidate_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    complete: bool,
    pending_status: str,
) -> QualificationStageDecision:
    decisions = tuple(
        _metric_decision(
            stage=stage,
            rule=rule,
            candidate_value=candidate_metrics.get(rule.metric),
            baseline_value=baseline_metrics.get(rule.metric),
            stage_complete=complete,
            pending_status=pending_status,
        )
        for rule in rules
    )
    status = _worst_status([decision.status for decision in decisions])
    reasons = tuple(reason for decision in decisions for reason in decision.reasons)
    return QualificationStageDecision(stage, status, complete, decisions, reasons)


def evaluate_qualification(
    profile: QualificationProfileRevision,
    candidate: QualificationEvidence,
    *,
    parent: Optional[QualificationEvidence] = None,
) -> ArtifactQualification:
    """Evaluate a candidate deterministically against one immutable profile."""

    ensure_same_profile(profile, candidate, *(tuple([parent]) if parent is not None else ()))
    empty: Mapping[str, Any] = {}
    development = _stage_decision(
        stage="development",
        rules=profile.development_rules,
        candidate_metrics=candidate.development_metrics,
        baseline_metrics=parent.development_metrics if parent else empty,
        complete=candidate.development_complete,
        pending_status="fail",
    )
    operational = _stage_decision(
        stage="operational",
        rules=profile.operational_rules,
        candidate_metrics=candidate.operational_metrics,
        baseline_metrics=parent.operational_metrics if parent else empty,
        complete=candidate.operational_complete,
        pending_status="fail",
    )
    if profile.holdout_required:
        holdout = _stage_decision(
            stage="holdout",
            rules=profile.holdout_rules,
            candidate_metrics=candidate.holdout_metrics,
            baseline_metrics=parent.holdout_metrics if parent else empty,
            complete=candidate.holdout_complete,
            pending_status="warn",
        )
    else:
        holdout = QualificationStageDecision(
            stage="holdout",
            status="not_required",
            complete=True,
            metrics=(),
            reasons=(),
        )
    stages = (development, operational, holdout)
    overall = _worst_status([stage.status for stage in stages])
    reasons = tuple(reason for stage in stages for reason in stage.reasons)
    return ArtifactQualification(
        profile_content_hash=profile.content_hash,
        candidate_artifact_hash=candidate.artifact_hash,
        parent_artifact_hash=parent.artifact_hash if parent else None,
        development=development,
        operational=operational,
        holdout=holdout,
        overall_status=overall,
        holdout_required=profile.holdout_required,
        reasons=reasons,
    )


@dataclass(frozen=True)
class QualificationMetricDelta:
    stage: str
    metric: str
    direction: str
    parent_value: Optional[float]
    candidate_value: Optional[float]
    raw_delta: Optional[float]
    favorable_delta: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "metric": self.metric,
            "direction": self.direction,
            "parent_value": self.parent_value,
            "candidate_value": self.candidate_value,
            "raw_delta": self.raw_delta,
            "favorable_delta": self.favorable_delta,
        }


@dataclass(frozen=True)
class QualificationComparison:
    profile_content_hash: str
    parent_artifact_hash: str
    candidate_artifact_hash: str
    deltas: Tuple[QualificationMetricDelta, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_content_hash": self.profile_content_hash,
            "parent_artifact_hash": self.parent_artifact_hash,
            "candidate_artifact_hash": self.candidate_artifact_hash,
            "deltas": [delta.to_dict() for delta in self.deltas],
        }


def compare_qualification_evidence(
    profile: QualificationProfileRevision,
    parent: QualificationEvidence,
    candidate: QualificationEvidence,
) -> QualificationComparison:
    """Compute direction-aware deltas after enforcing profile identity."""

    ensure_same_profile(profile, parent, candidate)
    deltas = []
    stage_definitions = (
        (
            "development",
            profile.development_rules,
            parent.development_metrics,
            candidate.development_metrics,
        ),
        (
            "operational",
            profile.operational_rules,
            parent.operational_metrics,
            candidate.operational_metrics,
        ),
        ("holdout", profile.holdout_rules, parent.holdout_metrics, candidate.holdout_metrics),
    )
    for stage, rules, parent_metrics, candidate_metrics in stage_definitions:
        for rule in rules:
            parent_value = parent_metrics.get(rule.metric)
            candidate_value = candidate_metrics.get(rule.metric)
            raw_delta = (
                None
                if parent_value is None or candidate_value is None
                else float(candidate_value) - float(parent_value)
            )
            favorable = (
                None
                if raw_delta is None
                else (raw_delta if rule.direction == "maximize" else -raw_delta)
            )
            deltas.append(
                QualificationMetricDelta(
                    stage,
                    rule.metric,
                    rule.direction,
                    None if parent_value is None else float(parent_value),
                    None if candidate_value is None else float(candidate_value),
                    raw_delta,
                    favorable,
                )
            )
    return QualificationComparison(
        profile.content_hash,
        parent.artifact_hash,
        candidate.artifact_hash,
        tuple(deltas),
    )


@dataclass(frozen=True)
class PromotionEligibility:
    target_alias: str
    eligible: bool
    requires_override: bool
    overridden: bool
    reasons: Tuple[str, ...]
    override_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_alias": self.target_alias,
            "eligible": self.eligible,
            "requires_override": self.requires_override,
            "overridden": self.overridden,
            "reasons": list(self.reasons),
            "override_note": self.override_note,
        }


def promotion_eligibility(
    decision: ArtifactQualification,
    target_alias: str,
    *,
    override_note: Optional[str] = None,
) -> PromotionEligibility:
    """Check candidate/approved gates without performing a promotion."""

    target = str(target_alias).strip().lower()
    if target not in {"candidate", "approved"}:
        raise ValueError("target_alias must be candidate or approved")
    required_stages = [decision.development, decision.operational]
    if target == "approved" and decision.holdout_required:
        required_stages.append(decision.holdout)
    blocked = [stage for stage in required_stages if stage.status != "pass"]
    reasons = tuple(f"{stage.stage} gate is {stage.status}" for stage in blocked)
    note = None if override_note is None else str(override_note).strip() or None
    naturally_eligible = not blocked
    overridden = bool(blocked and note)
    return PromotionEligibility(
        target_alias=target,
        eligible=naturally_eligible or overridden,
        requires_override=bool(blocked and not note),
        overridden=overridden,
        reasons=reasons,
        override_note=note if overridden else None,
    )


@dataclass(frozen=True)
class ParetoPoint:
    identity: str
    metrics: FrozenJsonMap

    def __post_init__(self) -> None:
        identity = str(self.identity).strip()
        if not identity:
            raise ValueError("Pareto point identity cannot be empty")
        metrics = _normalize_metric_map(self.metrics, name=f"pareto.{identity}")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "metrics", metrics)

    def to_dict(self) -> Dict[str, Any]:
        return {"identity": self.identity, "metrics": self.metrics.to_dict()}


@dataclass(frozen=True)
class ParetoResult:
    frontier: Tuple[ParetoPoint, ...]
    dominated: Tuple[ParetoPoint, ...]
    incomplete: Tuple[ParetoPoint, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frontier": [point.to_dict() for point in self.frontier],
            "dominated": [point.to_dict() for point in self.dominated],
            "incomplete": [point.to_dict() for point in self.incomplete],
        }


def pareto_front(
    points: Sequence[ParetoPoint],
    directions: Mapping[str, str],
) -> ParetoResult:
    """Return a deterministic Pareto partition; missing values are incomplete."""

    if not directions:
        raise ValueError("at least one Pareto metric direction is required")
    normalized_directions = {
        str(metric).strip(): str(direction).strip().lower()
        for metric, direction in directions.items()
    }
    if any(not metric for metric in normalized_directions):
        raise ValueError("Pareto metric names cannot be empty")
    if any(direction not in METRIC_DIRECTIONS for direction in normalized_directions.values()):
        raise ValueError("Pareto directions must be maximize or minimize")
    normalized_points = tuple(sorted(points, key=lambda point: point.identity))
    identities = [point.identity for point in normalized_points]
    if len(identities) != len(set(identities)):
        raise ValueError("Pareto point identities must be unique")
    complete = []
    incomplete = []
    for point in normalized_points:
        if any(point.metrics.get(metric) is None for metric in normalized_directions):
            incomplete.append(point)
        else:
            complete.append(point)

    def dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
        never_worse = True
        strictly_better = False
        for metric, direction in normalized_directions.items():
            left_value = float(left.metrics[metric])
            right_value = float(right.metrics[metric])
            if direction == "maximize":
                never_worse = never_worse and left_value >= right_value
                strictly_better = strictly_better or left_value > right_value
            else:
                never_worse = never_worse and left_value <= right_value
                strictly_better = strictly_better or left_value < right_value
        return never_worse and strictly_better

    frontier = []
    dominated = []
    for point in complete:
        if any(other is not point and dominates(other, point) for other in complete):
            dominated.append(point)
        else:
            frontier.append(point)
    return ParetoResult(tuple(frontier), tuple(dominated), tuple(incomplete))


__all__ = [
    "ArtifactQualification",
    "ParetoPoint",
    "ParetoResult",
    "PromotionEligibility",
    "QualificationComparison",
    "QualificationDecision",
    "QualificationEvidence",
    "QualificationMetricDecision",
    "QualificationMetricDelta",
    "QualificationStageDecision",
    "compare_qualification_evidence",
    "ensure_same_profile",
    "evaluate_qualification",
    "pareto_front",
    "promotion_eligibility",
]
