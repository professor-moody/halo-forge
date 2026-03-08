"""Shared UI presentation helpers for training workflow status and actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TrainingAction:
    """A normalized UI action for training workflow surfaces."""

    id: str
    label: str
    icon: str
    tone: str = "neutral"


@dataclass(frozen=True)
class TrainingPresentation:
    """Shared presentation model for monitor/results training workflow surfaces."""

    headline_status: str
    supporting_summary: str
    confidence_tone: str
    primary_action: Optional[TrainingAction] = None
    secondary_actions: List[TrainingAction] = field(default_factory=list)


@dataclass(frozen=True)
class LaunchPresentation:
    """Shared presentation model for launch readiness and quality outlook."""

    headline_status: str
    supporting_summary: str
    confidence_tone: str
    recommended_adjustment: str


def _dedupe_actions(
    primary_action: Optional[TrainingAction],
    secondary_actions: List[TrainingAction],
) -> List[TrainingAction]:
    seen = {primary_action.id} if primary_action else set()
    deduped: List[TrainingAction] = []
    for action in secondary_actions:
        if action.id in seen:
            continue
        deduped.append(action)
        seen.add(action.id)
    return deduped


def build_training_run_presentation(
    *,
    job_status: str,
    quality_status: Optional[str],
    quality_summary: Optional[str],
    recovery_status: Optional[str],
    recovery_action: Optional[str],
    recovery_summary: Optional[str],
    failure_reason: Optional[str],
    final_reason: Optional[str],
    has_launch_context: bool,
    can_resume_latest: bool,
    weights_updated: Optional[bool] = None,
    has_quality_details: bool = True,
) -> TrainingPresentation:
    """Build a consistent headline and action hierarchy for a training run."""

    status = str(job_status or "").strip().lower()
    quality = str(quality_status or "").strip().lower()
    recovery = str(recovery_status or "").strip().lower()
    summary_text = (
        str(recovery_summary or "").strip()
        or str(quality_summary or "").strip()
        or str(failure_reason or "").strip()
        or str(final_reason or "").strip()
        or "No additional training summary is available yet."
    )

    review_action = (
        TrainingAction("review_details", "Review Quality", "insights", "neutral")
        if has_quality_details
        else None
    )
    run_again_action = (
        TrainingAction("run_again", "Run Again", "replay", "neutral")
        if has_launch_context
        else None
    )
    resume_action = (
        TrainingAction("resume_latest", "Resume Latest", "history", "neutral")
        if has_launch_context and can_resume_latest
        else None
    )
    edit_action = TrainingAction("edit_config", "Edit Config", "edit", "neutral")
    guided_fix_action = (
        TrainingAction(
            "guided_fix",
            str(recovery_action or "").strip() or "Apply Suggested Fix",
            "auto_fix_high",
            "success",
        )
        if recovery == "ready"
        else None
    )

    primary_action: Optional[TrainingAction] = None
    secondary_actions: List[TrainingAction] = []
    tone = "neutral"
    headline = "Training status unavailable"

    if guided_fix_action is not None:
        tone = "warning" if quality in {"low_yield", "no_signal"} else "success"
        headline = "Suggested fix ready"
        primary_action = guided_fix_action
        secondary_actions.extend([action for action in (review_action, run_again_action, resume_action, edit_action) if action])
    elif status in {"failed", "stopped"}:
        tone = "danger"
        headline = "Run needs attention"
        primary_action = edit_action
        secondary_actions.extend([action for action in (run_again_action, resume_action, review_action) if action])
    elif status == "completed" and quality not in {"low_yield", "no_signal", "error"}:
        tone = "success"
        headline = "Run completed"
        primary_action = review_action
        secondary_actions.extend([action for action in (run_again_action, resume_action, edit_action) if action])
    elif status == "running":
        if quality in {"low_yield", "no_signal"}:
            tone = "warning" if quality == "low_yield" else "danger"
            headline = "Watch training quality"
        else:
            tone = "neutral"
            headline = "Training in progress"
        secondary_actions.extend([action for action in (edit_action,) if action])
    elif status == "pending":
        tone = "neutral"
        headline = "Waiting to start"
        secondary_actions.extend([action for action in (edit_action,) if action])
    else:
        tone = "warning" if quality in {"low_yield", "no_signal"} else "neutral"
        headline = "Review run quality"
        primary_action = review_action
        secondary_actions.extend([action for action in (run_again_action, resume_action, edit_action) if action])

    return TrainingPresentation(
        headline_status=headline,
        supporting_summary=summary_text,
        confidence_tone=tone,
        primary_action=primary_action,
        secondary_actions=_dedupe_actions(primary_action, secondary_actions),
    )


def build_launch_presentation(
    *,
    mode_label: str,
    quality_status: str,
    quality_summary: str,
    suggested_adjustments: List[str],
    yield_safety_note: str,
) -> LaunchPresentation:
    """Build a concise launch-readiness presentation for the training page."""

    status = str(quality_status or "healthy").strip().lower()
    tone = {
        "healthy": "success",
        "caution": "warning",
        "low_yield": "danger",
    }.get(status, "neutral")
    headline = {
        "healthy": f"{mode_label} launch is ready",
        "caution": f"{mode_label} launch has a few quality risks",
        "low_yield": f"{mode_label} launch is at risk of low signal",
    }.get(status, f"{mode_label} launch status")
    recommendation = next(
        (str(item).strip() for item in suggested_adjustments if str(item).strip()),
        str(yield_safety_note or "").strip() or "These settings look balanced for a first run.",
    )
    summary = str(quality_summary or "").strip() or str(yield_safety_note or "").strip() or "Review the settings and launch when ready."
    return LaunchPresentation(
        headline_status=headline,
        supporting_summary=summary,
        confidence_tone=tone,
        recommended_adjustment=recommendation,
    )
