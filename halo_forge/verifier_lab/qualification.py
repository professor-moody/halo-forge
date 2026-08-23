"""Reviewed verifier qualification templates and runtime compatibility."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from .metrics import CalibrationEvidence, normalize_evidence
from .observation import RewardContract, normalize_reward_contract


QUALIFICATION_SCOPES = {"development", "operational", "confirmation"}


@dataclass(frozen=True)
class QualificationTemplate:
    key: str
    display_name: str
    promotable: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualificationResult:
    decision: str
    template: str
    scope: str
    promotable: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "template": self.template,
            "scope": self.scope,
            "promotable": self.promotable,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "evidence": dict(self.evidence),
        }


QUALIFICATION_TEMPLATES: Mapping[str, QualificationTemplate] = {
    "strict_oracle": QualificationTemplate(
        key="strict_oracle",
        display_name="Strict oracle",
        promotable=True,
        description="Near-exact programmatic-oracle agreement with closed-failure behavior.",
    ),
    "human_aligned": QualificationTemplate(
        key="human_aligned",
        display_name="Human aligned",
        promotable=True,
        description="Task-aware agreement and repeatability against reviewed human labels.",
    ),
    "exploratory": QualificationTemplate(
        key="exploratory",
        display_name="Exploratory",
        promotable=False,
        description="Evidence reporting only; this template never grants a promotable pass.",
    ),
}


def qualification_templates() -> list[dict[str, Any]]:
    return [QUALIFICATION_TEMPLATES[key].to_dict() for key in sorted(QUALIFICATION_TEMPLATES)]


def _metric(metrics: Mapping[str, Any], section: str, name: str) -> Optional[float]:
    value = (metrics.get(section) or {}).get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _primary_metric(metrics: Mapping[str, Any]) -> Optional[float]:
    primary = metrics.get("primary_metric") or {}
    value = primary.get("value") if isinstance(primary, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _evaluate_minimum_evidence(
    metrics: Mapping[str, Any],
    *,
    task_type: str,
    required_classes: Sequence[str],
) -> tuple[str, list[str], dict[str, Any]]:
    universal = metrics.get("universal") or {}
    task = metrics.get("task") or {}
    record_count = int(task.get("record_count", universal.get("record_count", 0)) or 0)
    details: dict[str, Any] = {"record_count": record_count}
    if record_count < 20:
        return "fail", [f"fewer than 20 distinct records ({record_count})"], details
    required = 50 if task_type == "ranking" else 100
    reasons: list[str] = []
    if record_count < required:
        reasons.append(f"promotable evidence requires {required} records; found {record_count}")
    if task_type == "ranking":
        comparisons = int(task.get("implied_comparisons", 0) or 0)
        details["implied_comparisons"] = comparisons
        if comparisons < 150:
            reasons.append(
                f"ranking evidence requires 150 implied comparisons; found {comparisons}"
            )
    if task_type in {"categorical", "multi_label"}:
        class_section = task.get("per_class") if task_type == "categorical" else task.get("per_label")
        class_section = class_section if isinstance(class_section, Mapping) else {}
        observed_counts = {
            str(label): int((value or {}).get("support", 0) or 0)
            for label, value in class_section.items()
            if isinstance(value, Mapping)
        }
        details["class_counts"] = observed_counts
        classes = tuple(required_classes) or tuple(sorted(observed_counts))
        for label in classes:
            if observed_counts.get(str(label), 0) < 20:
                reasons.append(
                    f"required class {label!r} needs 20 records; found {observed_counts.get(str(label), 0)}"
                )
    return ("warn" if reasons else "pass"), reasons, details


def _rule(
    *,
    label: str,
    value: Optional[float],
    pass_threshold: float,
    warn_threshold: float,
    direction: str,
) -> tuple[str, str]:
    if value is None:
        return "fail", f"{label} is unavailable"
    if direction == "maximize":
        if value >= pass_threshold:
            return "pass", f"{label} {value:.4f} meets pass threshold {pass_threshold:.4f}"
        if value >= warn_threshold:
            return "warn", f"{label} {value:.4f} meets warn threshold {warn_threshold:.4f}"
        return "fail", f"{label} {value:.4f} is below warn threshold {warn_threshold:.4f}"
    if value <= pass_threshold:
        return "pass", f"{label} {value:.4f} meets pass maximum {pass_threshold:.4f}"
    if value <= warn_threshold:
        return "warn", f"{label} {value:.4f} meets warn maximum {warn_threshold:.4f}"
    return "fail", f"{label} {value:.4f} exceeds warn maximum {warn_threshold:.4f}"


def _thresholds(
    requirements: Mapping[str, Any],
    *path: str,
    defaults: tuple[float, float],
) -> tuple[float, float]:
    current: Any = requirements
    for part in path:
        if not isinstance(current, Mapping):
            return defaults
        current = current.get(part)
    if isinstance(current, Mapping):
        values = (current.get("pass"), current.get("warn"))
    elif isinstance(current, (list, tuple)) and len(current) >= 2:
        values = (current[0], current[1])
    else:
        return defaults
    try:
        return float(values[0]), float(values[1])
    except (TypeError, ValueError):
        return defaults


def _strict_rules(
    metrics: Mapping[str, Any],
    task_type: str,
    requirements: Mapping[str, Any],
) -> list[tuple[str, str]]:
    if task_type in {"binary", "categorical"}:
        agreement = _metric(metrics, "task", "accuracy")
    elif task_type == "multi_label":
        agreement = _metric(metrics, "task", "exact_match")
    else:
        agreement = _primary_metric(metrics)
    agreement_thresholds = _thresholds(
        requirements, "primary_agreement", defaults=(0.98, 0.95)
    )
    coverage_thresholds = _thresholds(
        requirements, "universal", "coverage", defaults=(0.99, 0.97)
    )
    error_thresholds = _thresholds(
        requirements, "universal", "error_rate", defaults=(0.01, 0.03)
    )
    repeat_thresholds = _thresholds(
        requirements, "exact_repeat_agreement", defaults=(1.0, 1.0)
    )
    rules = [
        _rule(
            label="primary agreement",
            value=agreement,
            pass_threshold=agreement_thresholds[0],
            warn_threshold=agreement_thresholds[1],
            direction="maximize",
        ),
        _rule(
            label="coverage",
            value=_metric(metrics, "universal", "coverage"),
            pass_threshold=coverage_thresholds[0],
            warn_threshold=coverage_thresholds[1],
            direction="maximize",
        ),
        _rule(
            label="error rate",
            value=_metric(metrics, "universal", "error_rate"),
            pass_threshold=error_thresholds[0],
            warn_threshold=error_thresholds[1],
            direction="minimize",
        ),
        _rule(
            label="exact repeat agreement",
            value=_metric(metrics, "universal", "exact_repeat_agreement"),
            pass_threshold=repeat_thresholds[0],
            warn_threshold=repeat_thresholds[1],
            direction="maximize",
        ),
    ]
    if task_type == "binary":
        false_accept_thresholds = _thresholds(
            requirements, "false_accept_rate", defaults=(0.01, 0.03)
        )
        false_reject_thresholds = _thresholds(
            requirements, "false_reject_rate", defaults=(0.02, 0.05)
        )
        rules.extend(
            [
                _rule(
                    label="false accept rate",
                    value=_metric(metrics, "task", "false_accept_rate"),
                    pass_threshold=false_accept_thresholds[0],
                    warn_threshold=false_accept_thresholds[1],
                    direction="minimize",
                ),
                _rule(
                    label="false reject rate",
                    value=_metric(metrics, "task", "false_reject_rate"),
                    pass_threshold=false_reject_thresholds[0],
                    warn_threshold=false_reject_thresholds[1],
                    direction="minimize",
                ),
            ]
        )
    if _metric(metrics, "universal", "reward_model_batch_consistency") is not None:
        rules.append(
            _rule(
                label="reward-model batch consistency",
                value=_metric(metrics, "universal", "reward_model_batch_consistency"),
                pass_threshold=1.0,
                warn_threshold=1.0,
                direction="maximize",
            )
        )
    return rules


def _human_rules(
    metrics: Mapping[str, Any],
    task_type: str,
    requirements: Mapping[str, Any],
) -> list[tuple[str, str]]:
    primary_name_by_task = {
        "binary": "balanced_accuracy",
        "categorical": "macro_f1",
        "multi_label": "macro_f1",
        "pairwise": "tie_aware_accuracy",
        "ranking": "kendall_tau",
        "scalar": "spearman",
    }
    pass_warn = {
        "binary": (0.80, 0.70),
        "categorical": (0.80, 0.70),
        "multi_label": (0.75, 0.65),
        "pairwise": (0.75, 0.65),
        "ranking": (0.60, 0.45),
        "scalar": (0.70, 0.50),
    }
    pass_threshold, warn_threshold = _thresholds(
        requirements,
        "task",
        task_type,
        "primary",
        defaults=pass_warn[task_type],
    )
    repeat_thresholds = _thresholds(
        requirements, "repeat_agreement", defaults=(0.95, 0.90)
    )
    rules = [
        _rule(
            label=primary_name_by_task[task_type].replace("_", " "),
            value=_metric(metrics, "task", primary_name_by_task[task_type]),
            pass_threshold=pass_threshold,
            warn_threshold=warn_threshold,
            direction="maximize",
        ),
        _rule(
            label="repeat agreement",
            value=_metric(metrics, "universal", "repeat_agreement"),
            pass_threshold=repeat_thresholds[0],
            warn_threshold=repeat_thresholds[1],
            direction="maximize",
        ),
    ]
    if task_type == "pairwise":
        order_thresholds = _thresholds(
            requirements,
            "task",
            "pairwise",
            "order_consistency",
            defaults=(0.95, 0.90),
        )
        rules.append(
            _rule(
                label="order consistency",
                value=_metric(metrics, "task", "order_consistency"),
                pass_threshold=order_thresholds[0],
                warn_threshold=order_thresholds[1],
                direction="maximize",
            )
        )
    if task_type == "scalar":
        mae_thresholds = _thresholds(
            requirements,
            "task",
            "scalar",
            "normalized_mae",
            defaults=(0.15, 0.25),
        )
        rules.append(
            _rule(
                label="normalized MAE",
                value=_metric(metrics, "task", "normalized_mae"),
                pass_threshold=mae_thresholds[0],
                warn_threshold=mae_thresholds[1],
                direction="minimize",
            )
        )
    if _metric(metrics, "universal", "reward_model_batch_consistency") is not None:
        rules.append(
            _rule(
                label="reward-model batch consistency",
                value=_metric(metrics, "universal", "reward_model_batch_consistency"),
                pass_threshold=1.0,
                warn_threshold=1.0,
                direction="maximize",
            )
        )
    return rules


def qualify_calibration(
    metrics: Mapping[str, Any],
    *,
    task_type: str,
    template: str,
    scope: str,
    required_classes: Sequence[str] = (),
    requirements: Optional[Mapping[str, Any]] = None,
) -> QualificationResult:
    """Apply a reviewed template without tuning any verifier parameter."""

    task = str(task_type).strip().lower()
    template_key = str(template).strip().lower()
    scope_key = str(scope).strip().lower()
    if template_key not in QUALIFICATION_TEMPLATES:
        raise ValueError("unknown verifier qualification template")
    if scope_key not in QUALIFICATION_SCOPES:
        raise ValueError("qualification scope must be development, operational, or confirmation")
    evidence_state, evidence_reasons, evidence_details = _evaluate_minimum_evidence(
        metrics,
        task_type=task,
        required_classes=required_classes,
    )
    if evidence_state == "fail":
        return QualificationResult(
            decision="fail",
            template=template_key,
            scope=scope_key,
            promotable=False,
            reasons=tuple(evidence_reasons),
            warnings=(),
            evidence=evidence_details,
        )

    if template_key == "exploratory":
        warnings = tuple(evidence_reasons) + (
            "exploratory qualification can never grant a promotable pass",
        )
        return QualificationResult(
            decision="warn",
            template=template_key,
            scope=scope_key,
            promotable=False,
            reasons=("exploratory evidence recorded",),
            warnings=warnings,
            evidence=evidence_details,
        )

    resolved_requirements = dict(requirements or {})
    rules = (
        _strict_rules(metrics, task, resolved_requirements)
        if template_key == "strict_oracle"
        else _human_rules(metrics, task, resolved_requirements)
    )
    states = [state for state, _ in rules]
    decision = "fail" if "fail" in states else "warn" if "warn" in states else "pass"
    if evidence_state == "warn" and decision == "pass":
        decision = "warn"
    reasons = tuple(message for state, message in rules if state == "fail")
    if not reasons:
        reasons = tuple(message for state, message in rules if state == decision)
    warnings = tuple(evidence_reasons) + tuple(
        message for state, message in rules if state == "warn" and message not in reasons
    )
    return QualificationResult(
        decision=decision,
        template=template_key,
        scope=scope_key,
        promotable=decision == "pass",
        reasons=reasons,
        warnings=warnings,
        evidence={
            **evidence_details,
            "rules": [
                {"state": state, "reason": reason} for state, reason in rules
            ],
        },
    )


def promotion_eligibility(
    decisions: Sequence[QualificationResult | Mapping[str, Any]],
    *,
    confirmation_required: bool,
) -> dict[str, Any]:
    """Resolve candidate/approved eligibility from explicit scoped decisions."""

    normalized: dict[str, tuple[str, bool]] = {}
    for value in decisions:
        if isinstance(value, QualificationResult):
            scope = value.scope
            decision = value.decision
            override = False
        else:
            scope = str(value.get("scope", ""))
            decision = str(value.get("decision", ""))
            override = bool(value.get("override", False))
        # Overrides remain visible and excluded from ordinary guided promotion.
        if not override:
            normalized[scope] = (decision, override)
    candidate_missing = [
        scope
        for scope in ("development", "operational")
        if normalized.get(scope, (None, False))[0] != "pass"
    ]
    candidate = not candidate_missing
    approved_missing = list(candidate_missing)
    if confirmation_required and normalized.get("confirmation", (None, False))[0] != "pass":
        approved_missing.append("confirmation")
    return {
        "candidate": candidate,
        "approved": candidate and not approved_missing,
        "candidate_missing_scopes": candidate_missing,
        "approved_missing_scopes": approved_missing,
        "overrides_excluded": True,
    }


def runtime_compatibility(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare every pinned runtime field recursively.

    Missing fields are mismatches.  The function never infers unavailable GPU
    or seed support from neighboring hardware properties.
    """

    mismatches: list[dict[str, Any]] = []

    def compare(prefix: str, left: Any, right: Any) -> None:
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping):
                mismatches.append({"field": prefix, "expected": left, "actual": right})
                return
            for key in sorted(left):
                compare(f"{prefix}.{key}" if prefix else str(key), left[key], right.get(key))
            return
        if left != right:
            mismatches.append({"field": prefix, "expected": left, "actual": right})

    compare("", expected, actual)
    return {
        "status": "compatible" if not mismatches else "stale_runtime",
        "compatible": not mismatches,
        "mismatches": mismatches,
    }


def runtime_contract_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def binary_threshold_curve(
    evidence: Sequence[CalibrationEvidence | Mapping[str, Any] | Any],
    *,
    reward_contract: RewardContract | Mapping[str, Any] | Any,
    thresholds: Optional[Iterable[float]] = None,
) -> list[dict[str, Any]]:
    """Report threshold alternatives without applying or persisting a change."""

    contract = normalize_reward_contract(reward_contract)
    rows = [normalize_evidence(value) for value in evidence]
    primary = [row for row in rows if row.primary and row.reward is not None]
    candidates = (
        sorted({float(value) for value in thresholds})
        if thresholds is not None
        else sorted({float(row.reward) for row in primary if row.reward is not None})
    )
    result: list[dict[str, Any]] = []
    for threshold in candidates:
        if not contract.minimum <= threshold <= contract.maximum:
            raise ValueError("threshold curve value is outside the reward range")
        tp = tn = fp = fn = 0
        for row in primary:
            expected = str(row.expected).strip().lower() in {"true", "1", "pass", "yes"}
            reward = float(row.reward)
            predicted = reward >= threshold if contract.direction == "maximize" else reward <= threshold
            tp += int(expected and predicted)
            tn += int(not expected and not predicted)
            fp += int(not expected and predicted)
            fn += int(expected and not predicted)
        false_accept = fp / (fp + tn) if fp + tn else None
        false_reject = fn / (fn + tp) if fn + tp else None
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else None
        result.append(
            {
                "threshold": threshold,
                "accuracy": accuracy,
                "false_accept_rate": false_accept,
                "false_reject_rate": false_reject,
                "applied": False,
            }
        )
    return result


__all__ = [
    "QUALIFICATION_SCOPES",
    "QUALIFICATION_TEMPLATES",
    "QualificationResult",
    "QualificationTemplate",
    "binary_threshold_curve",
    "promotion_eligibility",
    "qualification_templates",
    "qualify_calibration",
    "runtime_compatibility",
    "runtime_contract_hash",
]
