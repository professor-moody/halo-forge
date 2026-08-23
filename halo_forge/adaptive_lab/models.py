"""Immutable public models for adaptive checkpoint and evidence workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ._canonical import FrozenJsonMap, content_fingerprint

METRIC_DIRECTIONS = frozenset({"maximize", "minimize"})
SCHEDULE_MODES = frozenset({"final", "interval", "percentages", "explicit"})
BOUNDARY_UNITS = frozenset({"step", "cycle", "epoch"})
COMPARISONS = frozenset({"absolute", "baseline", "previous", "best"})
GATE_ACTIONS = frozenset({"continue", "pause", "stop"})
RULE_KINDS = frozenset({"objective", "guardrail", "plateau"})


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class CheckpointSchedule:
    """Trainer-independent checkpoint boundary schedule."""

    mode: str = "final"
    unit: str = "step"
    interval: Optional[int] = None
    percentages: Tuple[float, ...] = ()
    boundaries: Tuple[int, ...] = ()
    include_final: bool = True

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        unit = str(self.unit).strip().lower()
        if mode not in SCHEDULE_MODES:
            raise ValueError(f"schedule mode must be one of {sorted(SCHEDULE_MODES)}")
        if unit not in BOUNDARY_UNITS:
            raise ValueError(f"schedule unit must be one of {sorted(BOUNDARY_UNITS)}")
        if not isinstance(self.include_final, bool):
            raise TypeError("include_final must be a boolean")
        interval = None if self.interval is None else int(self.interval)
        percentages = tuple(_finite(value, "percentage") for value in self.percentages)
        boundaries = tuple(int(value) for value in self.boundaries)
        if interval is not None and interval <= 0:
            raise ValueError("schedule interval must be positive")
        if any(value <= 0.0 or value > 1.0 for value in percentages):
            raise ValueError("schedule percentages must be in (0, 1]")
        if any(value <= 0 for value in boundaries):
            raise ValueError("explicit boundaries must be positive")
        if percentages != tuple(sorted(set(percentages))):
            raise ValueError("schedule percentages must be unique and sorted")
        if boundaries != tuple(sorted(set(boundaries))):
            raise ValueError("explicit boundaries must be unique and sorted")
        if mode == "interval" and interval is None:
            raise ValueError("interval schedules require interval")
        if mode == "percentages" and not percentages:
            raise ValueError("percentage schedules require percentages")
        if mode == "explicit" and not boundaries:
            raise ValueError("explicit schedules require boundaries")
        if mode != "interval" and interval is not None:
            raise ValueError("interval is only valid for interval schedules")
        if mode != "percentages" and percentages:
            raise ValueError("percentages are only valid for percentage schedules")
        if mode != "explicit" and boundaries:
            raise ValueError("boundaries are only valid for explicit schedules")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "interval", interval)
        object.__setattr__(self, "percentages", percentages)
        object.__setattr__(self, "boundaries", boundaries)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "unit": self.unit,
            "interval": self.interval,
            "percentages": list(self.percentages),
            "boundaries": list(self.boundaries),
            "include_final": self.include_final,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "CheckpointSchedule":
        values = dict(payload or {})
        return cls(
            mode=values.get("mode", "final"),
            unit=values.get("unit", "step"),
            interval=values.get("interval"),
            percentages=tuple(values.get("percentages") or ()),
            boundaries=tuple(values.get("boundaries") or ()),
            include_final=values.get("include_final", True),
        )


@dataclass(frozen=True)
class CheckpointGateRule:
    """One metric rule evaluated at a verified checkpoint boundary."""

    metric: str
    direction: str
    comparison: str = "absolute"
    kind: str = "guardrail"
    threshold: Optional[float] = None
    minimum_delta: Optional[float] = None
    practical_delta: float = 0.0
    patience: int = 1
    on_breach: str = "stop"
    required: bool = True

    def __post_init__(self) -> None:
        metric = str(self.metric).strip()
        direction = str(self.direction).strip().lower()
        comparison = str(self.comparison).strip().lower()
        kind = str(self.kind).strip().lower()
        on_breach = str(self.on_breach).strip().lower()
        if not metric:
            raise ValueError("gate-rule metric cannot be empty")
        if direction not in METRIC_DIRECTIONS:
            raise ValueError("gate-rule direction must be maximize or minimize")
        if comparison not in COMPARISONS:
            raise ValueError(f"comparison must be one of {sorted(COMPARISONS)}")
        if kind not in RULE_KINDS:
            raise ValueError(f"kind must be one of {sorted(RULE_KINDS)}")
        if on_breach not in {"pause", "stop"}:
            raise ValueError("on_breach must be pause or stop")
        threshold = None if self.threshold is None else _finite(self.threshold, "threshold")
        minimum_delta = (
            None if self.minimum_delta is None else _finite(self.minimum_delta, "minimum_delta")
        )
        practical_delta = _finite(self.practical_delta, "practical_delta")
        patience = int(self.patience)
        if practical_delta < 0:
            raise ValueError("practical_delta must be non-negative")
        if patience <= 0:
            raise ValueError("patience must be positive")
        if comparison == "absolute" and threshold is None:
            raise ValueError("absolute gate rules require threshold")
        if comparison != "absolute" and minimum_delta is None:
            minimum_delta = practical_delta if kind == "plateau" else 0.0
        if kind == "plateau" and comparison not in {"previous", "best"}:
            raise ValueError("plateau rules compare with previous or best evidence")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "comparison", comparison)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "on_breach", on_breach)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "minimum_delta", minimum_delta)
        object.__setattr__(self, "practical_delta", practical_delta)
        object.__setattr__(self, "patience", patience)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "comparison": self.comparison,
            "kind": self.kind,
            "threshold": self.threshold,
            "minimum_delta": self.minimum_delta,
            "practical_delta": self.practical_delta,
            "patience": self.patience,
            "on_breach": self.on_breach,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointGateRule":
        return cls(**dict(payload))


@dataclass(frozen=True)
class CheckpointRetentionPolicy:
    """Reviewed cleanup recommendations for published checkpoint artifacts."""

    keep_last: int = 1
    keep_every_n_boundaries: Optional[int] = None
    keep_best: int = 1
    protect_evaluated: bool = True
    protect_decision_referenced: bool = True
    protect_lineage_referenced: bool = True
    review_before_cleanup: bool = True

    def __post_init__(self) -> None:
        keep_last = int(self.keep_last)
        keep_best = int(self.keep_best)
        periodic = (
            None if self.keep_every_n_boundaries is None else int(self.keep_every_n_boundaries)
        )
        if keep_last < 0 or keep_best < 0:
            raise ValueError("checkpoint retention counts cannot be negative")
        if periodic is not None and periodic <= 0:
            raise ValueError("keep_every_n_boundaries must be positive")
        for name in (
            "protect_evaluated",
            "protect_decision_referenced",
            "protect_lineage_referenced",
            "review_before_cleanup",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if not any(
            (
                keep_last,
                keep_best,
                periodic,
                self.protect_evaluated,
                self.protect_decision_referenced,
                self.protect_lineage_referenced,
            )
        ):
            raise ValueError("checkpoint retention must preserve at least one checkpoint class")
        object.__setattr__(self, "keep_last", keep_last)
        object.__setattr__(self, "keep_best", keep_best)
        object.__setattr__(self, "keep_every_n_boundaries", periodic)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keep_last": self.keep_last,
            "keep_every_n_boundaries": self.keep_every_n_boundaries,
            "keep_best": self.keep_best,
            "protect_evaluated": self.protect_evaluated,
            "protect_decision_referenced": self.protect_decision_referenced,
            "protect_lineage_referenced": self.protect_lineage_referenced,
            "review_before_cleanup": self.review_before_cleanup,
        }

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "CheckpointRetentionPolicy":
        return cls(**dict(payload or {}))


@dataclass(frozen=True)
class CheckpointPolicyRevision:
    """Immutable scientific definition for checkpoint-driven training."""

    policy_id: str
    revision_number: int
    name: str
    development_suite_revision_id: str
    primary_metric: str
    direction: str
    schedule: CheckpointSchedule = field(default_factory=CheckpointSchedule)
    rules: Tuple[CheckpointGateRule, ...] = ()
    retention: CheckpointRetentionPolicy = field(default_factory=CheckpointRetentionPolicy)
    guardrail_suite_revision_ids: Tuple[str, ...] = ()
    automatic_actions: bool = False
    compatible_capabilities: Tuple[str, ...] = ()
    description: Optional[str] = None
    revision_id: Optional[str] = None
    version: int = 1
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        policy_id = str(self.policy_id).strip()
        name = str(self.name).strip()
        suite_id = str(self.development_suite_revision_id).strip()
        metric = str(self.primary_metric).strip()
        direction = str(self.direction).strip().lower()
        if not all((policy_id, name, suite_id, metric)):
            raise ValueError("policy id, name, development suite, and primary metric are required")
        if int(self.revision_number) <= 0:
            raise ValueError("revision_number must be positive")
        if direction not in METRIC_DIRECTIONS:
            raise ValueError("direction must be maximize or minimize")
        if int(self.version) != 1:
            raise ValueError(f"unsupported checkpoint-policy version {self.version}")
        if not isinstance(self.automatic_actions, bool):
            raise TypeError("automatic_actions must be a boolean")
        schedule = (
            self.schedule
            if isinstance(self.schedule, CheckpointSchedule)
            else CheckpointSchedule.from_dict(self.schedule)
        )
        rules = tuple(
            rule if isinstance(rule, CheckpointGateRule) else CheckpointGateRule.from_dict(rule)
            for rule in self.rules
        )
        retention = (
            self.retention
            if isinstance(self.retention, CheckpointRetentionPolicy)
            else CheckpointRetentionPolicy.from_dict(self.retention)
        )
        guardrails = tuple(str(value).strip() for value in self.guardrail_suite_revision_ids)
        capabilities = tuple(
            sorted(
                {
                    str(value).strip().lower()
                    for value in self.compatible_capabilities
                    if str(value).strip()
                }
            )
        )
        if any(not value for value in guardrails):
            raise ValueError("guardrail suite ids cannot be empty")
        if len(guardrails) != len(set(guardrails)):
            raise ValueError("guardrail suite ids must be unique")
        if suite_id in guardrails:
            raise ValueError("development suite cannot also be a guardrail suite")
        rule_keys = [(rule.metric, rule.kind, rule.comparison) for rule in rules]
        if len(rule_keys) != len(set(rule_keys)):
            raise ValueError("checkpoint gate rules must be unique by metric/kind/comparison")
        plateau_metrics = [rule.metric for rule in rules if rule.kind == "plateau"]
        if len(plateau_metrics) != len(set(plateau_metrics)):
            raise ValueError("a metric can define only one plateau rule")
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "revision_number", int(self.revision_number))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "development_suite_revision_id", suite_id)
        object.__setattr__(self, "primary_metric", metric)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "retention", retention)
        object.__setattr__(self, "guardrail_suite_revision_ids", guardrails)
        object.__setattr__(self, "compatible_capabilities", capabilities)
        object.__setattr__(
            self,
            "revision_id",
            None if self.revision_id is None else str(self.revision_id).strip() or None,
        )
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "content_hash", content_fingerprint(self.definition_dict()))

    @property
    def required_suite_revision_ids(self) -> Tuple[str, ...]:
        return (self.development_suite_revision_id, *self.guardrail_suite_revision_ids)

    def definition_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "development_suite_revision_id": self.development_suite_revision_id,
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "schedule": self.schedule.to_dict(),
            "rules": [rule.to_dict() for rule in self.rules],
            "retention": self.retention.to_dict(),
            "guardrail_suite_revision_ids": list(self.guardrail_suite_revision_ids),
            "automatic_actions": self.automatic_actions,
            "compatible_capabilities": list(self.compatible_capabilities),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "revision_number": self.revision_number,
            "revision_id": self.revision_id,
            "name": self.name,
            "description": self.description,
            **self.definition_dict(),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CheckpointPolicyRevision":
        values = dict(payload)
        expected_hash = values.pop("content_hash", None)
        result = cls(
            policy_id=values["policy_id"],
            revision_number=values["revision_number"],
            name=values["name"],
            description=values.get("description"),
            revision_id=values.get("revision_id"),
            development_suite_revision_id=values["development_suite_revision_id"],
            primary_metric=values["primary_metric"],
            direction=values["direction"],
            schedule=CheckpointSchedule.from_dict(values.get("schedule")),
            rules=tuple(values.get("rules") or ()),
            retention=CheckpointRetentionPolicy.from_dict(values.get("retention")),
            guardrail_suite_revision_ids=tuple(values.get("guardrail_suite_revision_ids") or ()),
            automatic_actions=values.get("automatic_actions", False),
            compatible_capabilities=tuple(values.get("compatible_capabilities") or ()),
            version=values.get("version", 1),
        )
        if expected_hash is not None and expected_hash != result.content_hash:
            raise ValueError("checkpoint-policy content_hash does not match its definition")
        return result


@dataclass(frozen=True)
class ResolvedCheckpointPlan:
    policy_revision_id: str
    policy_hash: str
    trainer_mode: str
    unit: str
    total_budget: int
    boundaries: Tuple[int, ...]
    required_suite_revision_ids: Tuple[str, ...]
    automatic_actions: bool
    retention: CheckpointRetentionPolicy = field(default_factory=CheckpointRetentionPolicy)
    capability_notes: Tuple[str, ...] = ()
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not str(self.policy_revision_id).strip() or not str(self.policy_hash).strip():
            raise ValueError("policy revision id and hash are required")
        if not str(self.trainer_mode).strip():
            raise ValueError("trainer_mode is required")
        if self.unit not in BOUNDARY_UNITS:
            raise ValueError("unsupported checkpoint unit")
        total = int(self.total_budget)
        boundaries = tuple(int(value) for value in self.boundaries)
        if total <= 0:
            raise ValueError("total_budget must be positive")
        if not boundaries or boundaries != tuple(sorted(set(boundaries))):
            raise ValueError("resolved boundaries must be non-empty, unique, and sorted")
        if boundaries[0] <= 0 or boundaries[-1] > total:
            raise ValueError("resolved boundaries must fall within total_budget")
        suites = tuple(str(value).strip() for value in self.required_suite_revision_ids)
        if not suites or any(not value for value in suites) or len(suites) != len(set(suites)):
            raise ValueError("required suite revision ids must be non-empty and unique")
        retention = (
            self.retention
            if isinstance(self.retention, CheckpointRetentionPolicy)
            else CheckpointRetentionPolicy.from_dict(self.retention)
        )
        object.__setattr__(self, "policy_revision_id", str(self.policy_revision_id).strip())
        object.__setattr__(self, "policy_hash", str(self.policy_hash).strip())
        object.__setattr__(self, "trainer_mode", str(self.trainer_mode).strip().lower())
        object.__setattr__(self, "total_budget", total)
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "required_suite_revision_ids", suites)
        object.__setattr__(self, "retention", retention)
        object.__setattr__(self, "capability_notes", tuple(str(v) for v in self.capability_notes))
        object.__setattr__(self, "content_hash", content_fingerprint(self.definition_dict()))

    def definition_dict(self) -> Dict[str, Any]:
        return {
            "policy_revision_id": self.policy_revision_id,
            "policy_hash": self.policy_hash,
            "trainer_mode": self.trainer_mode,
            "unit": self.unit,
            "total_budget": self.total_budget,
            "boundaries": list(self.boundaries),
            "required_suite_revision_ids": list(self.required_suite_revision_ids),
            "automatic_actions": self.automatic_actions,
            "retention": self.retention.to_dict(),
            "capability_notes": list(self.capability_notes),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self.definition_dict(), "content_hash": self.content_hash}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResolvedCheckpointPlan":
        values = dict(payload)
        expected_hash = values.pop("content_hash", None)
        result = cls(
            policy_revision_id=values["policy_revision_id"],
            policy_hash=values["policy_hash"],
            trainer_mode=values["trainer_mode"],
            unit=values["unit"],
            total_budget=values["total_budget"],
            boundaries=tuple(values["boundaries"]),
            required_suite_revision_ids=tuple(values["required_suite_revision_ids"]),
            automatic_actions=bool(values["automatic_actions"]),
            retention=CheckpointRetentionPolicy.from_dict(values.get("retention")),
            capability_notes=tuple(values.get("capability_notes") or ()),
        )
        if expected_hash is not None and expected_hash != result.content_hash:
            raise ValueError("resolved checkpoint plan hash mismatch")
        return result


@dataclass(frozen=True)
class CheckpointGateDecision:
    policy_revision_id: str
    plan_hash: str
    boundary_index: int
    action: str
    automatic: bool
    reasons: Tuple[str, ...]
    evidence: FrozenJsonMap = field(default_factory=FrozenJsonMap)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        action = str(self.action).strip().lower()
        if action not in GATE_ACTIONS:
            raise ValueError("gate action must be continue, pause, or stop")
        if int(self.boundary_index) < 0:
            raise ValueError("boundary_index cannot be negative")
        reasons = tuple(str(value).strip() for value in self.reasons if str(value).strip())
        if not reasons:
            raise ValueError("gate decisions require at least one reason")
        evidence = (
            self.evidence
            if isinstance(self.evidence, FrozenJsonMap)
            else FrozenJsonMap(self.evidence)
        )
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "boundary_index", int(self.boundary_index))
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "content_hash", content_fingerprint(self.definition_dict()))

    def definition_dict(self) -> Dict[str, Any]:
        return {
            "policy_revision_id": self.policy_revision_id,
            "plan_hash": self.plan_hash,
            "boundary_index": self.boundary_index,
            "action": self.action,
            "automatic": self.automatic,
            "reasons": list(self.reasons),
            "evidence": self.evidence.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self.definition_dict(), "content_hash": self.content_hash}


@dataclass(frozen=True)
class CohortObservation:
    subject_id: str
    seed: int
    metric: str
    value: float
    evaluation_id: Optional[str] = None
    metadata: FrozenJsonMap = field(default_factory=FrozenJsonMap)

    def __post_init__(self) -> None:
        if not str(self.subject_id).strip() or not str(self.metric).strip():
            raise ValueError("cohort observation subject_id and metric are required")
        metadata = (
            self.metadata
            if isinstance(self.metadata, FrozenJsonMap)
            else FrozenJsonMap(self.metadata)
        )
        object.__setattr__(self, "subject_id", str(self.subject_id).strip())
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "metric", str(self.metric).strip())
        object.__setattr__(self, "value", _finite(self.value, "observation value"))
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "seed": self.seed,
            "metric": self.metric,
            "value": self.value,
            "evaluation_id": self.evaluation_id,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True)
class EvidenceCompatibility:
    """Scientific settings that must match before cohort evidence is joined."""

    subject_id: str
    suite_revision_id: str
    generation_settings_hash: str
    template_hash: Optional[str] = None
    evaluator_versions_hash: Optional[str] = None
    metadata: FrozenJsonMap = field(default_factory=FrozenJsonMap)
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("subject_id", self.subject_id),
            ("suite_revision_id", self.suite_revision_id),
            ("generation_settings_hash", self.generation_settings_hash),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, str(value).strip())
        metadata = (
            self.metadata
            if isinstance(self.metadata, FrozenJsonMap)
            else FrozenJsonMap(self.metadata)
        )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "content_hash", content_fingerprint(self.definition_dict()))

    def definition_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "suite_revision_id": self.suite_revision_id,
            "generation_settings_hash": self.generation_settings_hash,
            "template_hash": self.template_hash,
            "evaluator_versions_hash": self.evaluator_versions_hash,
            "metadata": self.metadata.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self.definition_dict(), "content_hash": self.content_hash}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceCompatibility":
        values = dict(payload)
        expected_hash = values.pop("content_hash", None)
        result = cls(**values)
        if expected_hash is not None and expected_hash != result.content_hash:
            raise ValueError("evidence compatibility hash mismatch")
        return result

    def mismatch_fields(self, other: "EvidenceCompatibility") -> Tuple[str, ...]:
        fields = (
            "suite_revision_id",
            "generation_settings_hash",
            "template_hash",
            "evaluator_versions_hash",
        )
        return tuple(name for name in fields if getattr(self, name) != getattr(other, name))


@dataclass(frozen=True)
class CohortAnalysisSnapshot:
    request: FrozenJsonMap
    analysis: FrozenJsonMap
    status: str = "completed"
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        request = (
            self.request if isinstance(self.request, FrozenJsonMap) else FrozenJsonMap(self.request)
        )
        analysis = (
            self.analysis
            if isinstance(self.analysis, FrozenJsonMap)
            else FrozenJsonMap(self.analysis)
        )
        object.__setattr__(self, "request", request)
        object.__setattr__(self, "analysis", analysis)
        object.__setattr__(
            self,
            "content_hash",
            content_fingerprint({"request": request.to_dict(), "analysis": analysis.to_dict()}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "request": self.request.to_dict(),
            "analysis": self.analysis.to_dict(),
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class ResearchDecisionRecord:
    analysis_snapshot_id: str
    selected_subject: FrozenJsonMap
    rejected_subjects: Tuple[FrozenJsonMap, ...]
    exclusions: Tuple[FrozenJsonMap, ...]
    rationale: str
    fork_spec: FrozenJsonMap = field(default_factory=FrozenJsonMap)
    override_reason: Optional[str] = None
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rationale = str(self.rationale).strip()
        if not rationale:
            raise ValueError("research decisions require a rationale")
        selected = (
            self.selected_subject
            if isinstance(self.selected_subject, FrozenJsonMap)
            else FrozenJsonMap(self.selected_subject)
        )
        rejected = tuple(
            value if isinstance(value, FrozenJsonMap) else FrozenJsonMap(value)
            for value in self.rejected_subjects
        )
        exclusions = tuple(
            value if isinstance(value, FrozenJsonMap) else FrozenJsonMap(value)
            for value in self.exclusions
        )
        fork_spec = (
            self.fork_spec
            if isinstance(self.fork_spec, FrozenJsonMap)
            else FrozenJsonMap(self.fork_spec)
        )
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "selected_subject", selected)
        object.__setattr__(self, "rejected_subjects", rejected)
        object.__setattr__(self, "exclusions", exclusions)
        object.__setattr__(self, "fork_spec", fork_spec)
        object.__setattr__(self, "content_hash", content_fingerprint(self.definition_dict()))

    def definition_dict(self) -> Dict[str, Any]:
        return {
            "analysis_snapshot_id": self.analysis_snapshot_id,
            "selected_subject": self.selected_subject.to_dict(),
            "rejected_subjects": [value.to_dict() for value in self.rejected_subjects],
            "exclusions": [value.to_dict() for value in self.exclusions],
            "rationale": self.rationale,
            "override_reason": self.override_reason,
            "fork_spec": self.fork_spec.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self.definition_dict(), "content_hash": self.content_hash}


@dataclass(frozen=True)
class EvidenceBundle:
    analysis_snapshot_id: str
    research_decision_id: Optional[str]
    request: FrozenJsonMap
    analysis_hash: Optional[str] = None
    decision_hash: Optional[str] = None
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        request = (
            self.request if isinstance(self.request, FrozenJsonMap) else FrozenJsonMap(self.request)
        )
        object.__setattr__(self, "request", request)
        object.__setattr__(
            self,
            "content_hash",
            content_fingerprint(
                {
                    "analysis": self.analysis_hash or self.analysis_snapshot_id,
                    "decision": self.decision_hash or self.research_decision_id,
                    "request": request.to_dict(),
                }
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_snapshot_id": self.analysis_snapshot_id,
            "research_decision_id": self.research_decision_id,
            "analysis_hash": self.analysis_hash,
            "decision_hash": self.decision_hash,
            "request": self.request.to_dict(),
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class WorkspaceDraft:
    draft_kind: str
    content: FrozenJsonMap
    owner_key: str = "local"
    name: str = "default"
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        content = (
            self.content if isinstance(self.content, FrozenJsonMap) else FrozenJsonMap(self.content)
        )
        for name, value in (
            ("draft_kind", self.draft_kind),
            ("owner_key", self.owner_key),
            ("name", self.name),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} cannot be empty")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "content_hash", content_fingerprint(content.to_dict()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "draft_kind": self.draft_kind,
            "owner_key": self.owner_key,
            "name": self.name,
            "content": self.content.to_dict(),
            "content_hash": self.content_hash,
        }


__all__ = [
    "CheckpointGateDecision",
    "CheckpointGateRule",
    "CheckpointPolicyRevision",
    "CheckpointRetentionPolicy",
    "CheckpointSchedule",
    "CohortAnalysisSnapshot",
    "CohortObservation",
    "EvidenceBundle",
    "EvidenceCompatibility",
    "ResearchDecisionRecord",
    "ResolvedCheckpointPlan",
    "WorkspaceDraft",
]
