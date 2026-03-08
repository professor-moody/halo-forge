"""Shared guided-recovery helpers for low-yield training runs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


MAX_RECOVERY_EXAMPLES = 3


def build_recovery_guidance(
    *,
    modality: str,
    yield_diagnostics: Optional[Dict[str, Any]],
    effectiveness: Optional[Dict[str, Any]],
    launch_args: Optional[Dict[str, Any]] = None,
    representative_examples: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a conservative guided-recovery payload from training diagnostics."""
    launch_args = dict(launch_args or {})
    yield_payload = dict(yield_diagnostics or {})
    effectiveness_payload = dict(effectiveness or {})
    reason_code = _derive_reason_code(yield_payload, effectiveness_payload)
    examples = normalize_representative_examples(
        representative_examples or (),
        dominant_reason=reason_code,
    )

    if not reason_code:
        return {
            "status": "unavailable",
            "recommended_action": "No guided recovery available",
            "suggested_overrides": {},
            "reason_code": "",
            "evidence_summary": "This run did not produce a recognized recovery pattern.",
            "representative_examples": examples,
        }

    suggested_overrides: Dict[str, Any] = {}
    recommended_action = "Review suggested fix"
    status = "ready"

    if reason_code == "below_reward_threshold":
        suggested_overrides = _lower_reward_threshold(launch_args)
        recommended_action = "Lower reward threshold"
    elif reason_code == "dropped_by_keep_percent":
        suggested_overrides = _raise_keep_percent(launch_args)
        recommended_action = "Keep more verified samples"
    elif reason_code == "verification_failed":
        suggested_overrides = _increase_sample_budget(modality, launch_args)
        recommended_action = "Increase sample budget"
    elif reason_code == "no_optimizer_steps":
        suggested_overrides = _reduce_minimum_floor(launch_args)
        recommended_action = "Reduce minimum sample floor"
    elif reason_code == "low_sample_budget":
        suggested_overrides = _increase_sample_budget(modality, launch_args)
        recommended_action = "Increase sample budget"
    elif reason_code in {"missing_text", "empty_target", "schema_invalid"}:
        status = "advisory_only"
        recommended_action = "Inspect dataset formatting"
    else:
        status = "advisory_only"
        recommended_action = "Inspect run details"

    if not suggested_overrides and status == "ready":
        status = "advisory_only"
        recommended_action = "Inspect run details"

    return {
        "status": status,
        "recommended_action": recommended_action,
        "suggested_overrides": suggested_overrides,
        "reason_code": reason_code,
        "evidence_summary": _build_evidence_summary(
            reason_code=reason_code,
            status=status,
            examples=examples,
            yield_diagnostics=yield_payload,
        ),
        "representative_examples": examples,
    }


def attach_recovery_guidance(
    summary: Dict[str, Any],
    *,
    modality: str,
    launch_args: Optional[Dict[str, Any]] = None,
    representative_examples: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Attach a guided-recovery block to an existing training summary."""
    summary["recovery_guidance"] = build_recovery_guidance(
        modality=modality,
        yield_diagnostics=summary.get("yield_diagnostics"),
        effectiveness=summary.get("effectiveness"),
        launch_args=launch_args,
        representative_examples=representative_examples,
    )
    return summary


def normalize_representative_examples(
    examples: Iterable[Dict[str, Any]],
    *,
    dominant_reason: Optional[str] = None,
    limit: int = MAX_RECOVERY_EXAMPLES,
) -> list[Dict[str, Any]]:
    """Normalize bounded example payloads for UI display."""
    normalized: list[Dict[str, Any]] = []
    for example in examples:
        if not isinstance(example, dict):
            continue
        reason = str(example.get("reason") or "").strip().lower()
        if dominant_reason and reason and reason != dominant_reason:
            continue
        normalized.append(
            {
                "reason": reason,
                "label": _truncate_text(example.get("label"), limit=48),
                "preview": _truncate_text(example.get("preview"), limit=180),
                "context": _truncate_text(example.get("context"), limit=120),
                "reward": _coerce_optional_float(example.get("reward")),
            }
        )
        if len(normalized) >= limit:
            break
    if not normalized and dominant_reason:
        for example in examples:
            if not isinstance(example, dict):
                continue
            normalized.append(
                {
                    "reason": str(example.get("reason") or "").strip().lower(),
                    "label": _truncate_text(example.get("label"), limit=48),
                    "preview": _truncate_text(example.get("preview"), limit=180),
                    "context": _truncate_text(example.get("context"), limit=120),
                    "reward": _coerce_optional_float(example.get("reward")),
                }
            )
            if len(normalized) >= limit:
                break
    return normalized


def _derive_reason_code(
    yield_diagnostics: Dict[str, Any],
    effectiveness: Dict[str, Any],
) -> str:
    summary = yield_diagnostics.get("summary") if isinstance(yield_diagnostics.get("summary"), dict) else {}
    dominant_reason = str(summary.get("dominant_rejection_reason") or "").strip().lower()
    if dominant_reason in {
        "below_reward_threshold",
        "dropped_by_keep_percent",
        "verification_failed",
        "missing_text",
        "empty_target",
        "schema_invalid",
    }:
        return dominant_reason

    reasons = effectiveness.get("reasons") if isinstance(effectiveness.get("reasons"), list) else []
    if "no_train_steps" in reasons or "optimizer_steps_below_minimum" in reasons:
        return "no_optimizer_steps"

    status = str(summary.get("status") or "").strip().lower()
    keep_rate = _coerce_optional_float(
        ((yield_diagnostics.get("rates") or {}) if isinstance(yield_diagnostics.get("rates"), dict) else {}).get("keep_rate")
    )
    if status in {"low_yield", "no_signal"} or (keep_rate is not None and keep_rate < 0.2):
        return "low_sample_budget"
    return ""


def _lower_reward_threshold(args: Dict[str, Any]) -> Dict[str, Any]:
    current = _coerce_optional_float(args.get("reward_threshold"))
    if current is None:
        return {}
    lowered = round(max(0.0, current - 0.1), 2)
    if lowered >= current:
        return {}
    return {"reward_threshold": lowered}


def _raise_keep_percent(args: Dict[str, Any]) -> Dict[str, Any]:
    current = _coerce_optional_float(args.get("keep_percent"))
    if current is None:
        return {}
    raised = round(min(0.8, current + 0.1), 2)
    if raised <= current:
        return {}
    return {"keep_percent": raised}


def _reduce_minimum_floor(args: Dict[str, Any]) -> Dict[str, Any]:
    current = _coerce_optional_int(args.get("min_samples"))
    if current is None or current <= 1:
        return {}
    lowered = max(1, int(current * 0.75))
    if lowered >= current:
        return {}
    return {"min_samples": lowered}


def _increase_sample_budget(modality: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if modality == "sft":
        current = _coerce_optional_int(args.get("max_samples"))
        target = 64 if current is None else min(512, max(current + 32, current * 2))
        return {"max_samples": target}
    if modality in {"reasoning", "agentic"}:
        current = _coerce_optional_int(args.get("limit"))
        target = 128 if current is None else min(512, max(current + 32, current * 2))
        return {"limit": target}
    current_samples = _coerce_optional_int(args.get("samples_per_prompt"))
    if current_samples is not None:
        return {"samples_per_prompt": min(16, current_samples + 2)}
    current_limit = _coerce_optional_int(args.get("limit"))
    if current_limit is not None:
        return {"limit": min(512, current_limit + 32)}
    return {}


def _build_evidence_summary(
    *,
    reason_code: str,
    status: str,
    examples: list[Dict[str, Any]],
    yield_diagnostics: Dict[str, Any],
) -> str:
    keep_rate = _coerce_optional_float(
        ((yield_diagnostics.get("rates") or {}) if isinstance(yield_diagnostics.get("rates"), dict) else {}).get("keep_rate")
    )
    example_note = f" {len(examples)} representative example(s) are attached." if examples else ""
    if reason_code == "below_reward_threshold":
        return f"Most candidate samples were dropped below the reward threshold.{example_note}"
    if reason_code == "dropped_by_keep_percent":
        return f"Verified samples existed, but keep-percent filtering was too selective.{example_note}"
    if reason_code == "verification_failed":
        return f"Verification rejected too many samples for stable updates.{example_note}"
    if reason_code in {"missing_text", "empty_target", "schema_invalid"}:
        return f"The dataset format needs attention before rerunning.{example_note}"
    if reason_code == "no_optimizer_steps":
        return f"Samples reached the run, but optimizer updates never materialized.{example_note}"
    if keep_rate is not None:
        return f"This run kept only {keep_rate:.0%} of candidate samples.{example_note}"
    return f"Guided recovery is {status.replace('_', ' ')} for this run.{example_note}"


def _truncate_text(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
