"""Public product view models built on top of internal training/readiness truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional

from ui.services.training_presentation import (
    TrainingAction,
    TrainingPresentation,
    build_training_run_presentation,
)


def _serialize(value: Any) -> Any:
    """Recursively serialize dataclasses and nested containers."""
    if is_dataclass(value):
        return {key: _serialize(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


@dataclass(frozen=True)
class PublicActionView:
    """User-facing action shown in the public product surface."""

    id: str
    label: str
    icon: str
    tone: str = "neutral"


@dataclass(frozen=True)
class ProductUserSummaryView:
    """Plain-language summary used across train, monitor, and results."""

    headline: str
    why_it_matters: str
    next_step: str
    confidence_tone: str
    primary_action: Optional[PublicActionView] = None
    secondary_actions: List[PublicActionView] = field(default_factory=list)


@dataclass(frozen=True)
class TrainingRecoveryView:
    """Recovery guidance safe to expose to public users."""

    status: str
    reason_code: str
    recommended_action: str
    evidence_summary: str
    suggested_overrides: Dict[str, Any] = field(default_factory=dict)
    representative_examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TrainingLaunchPreflightView:
    """Serialized launch preflight for the public training form."""

    mode: str
    ok: bool
    resolved_paths: Dict[str, str]
    errors: List[str]
    warnings: List[str]
    suggested_fixes: List[str]
    user_summary: ProductUserSummaryView
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingRunListItemView:
    """Compact training run row for lists and cards."""

    id: str
    run_id: str
    modality: str
    model_name: str
    status: str
    timestamp: str
    progress_percent: float
    user_summary: ProductUserSummaryView
    details: Dict[str, Any] = field(default_factory=dict)
    research_details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingRunDetailView:
    """Expanded training run detail view."""

    id: str
    run_id: str
    modality: str
    model_name: str
    status: str
    timestamp: str
    progress_percent: float
    user_summary: ProductUserSummaryView
    recovery: TrainingRecoveryView
    details: Dict[str, Any] = field(default_factory=dict)
    research_details: Dict[str, Any] = field(default_factory=dict)
    internal_details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingRunLiveView:
    """Live polling payload for the public monitor page."""

    id: str
    status: str
    progress_percent: float
    current_step: int
    total_steps: int
    current_epoch: float
    total_epochs: int
    current_cycle: int
    total_cycles: int
    latest_loss: Optional[float]
    latest_learning_rate: Optional[float]
    latest_grad_norm: Optional[float]
    user_summary: ProductUserSummaryView
    research_details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModalityReadinessView:
    """Public-safe training readiness summary for one modality."""

    modality: str
    readiness_tier: str
    production_ready: bool
    status: str
    caveat: str
    next_step: str
    eval_metric_name: str
    baseline_value: Optional[float]
    final_value: Optional[float]
    delta: Optional[float]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocsCapabilitySummaryView:
    """Curated doc summary shown in the public docs shell."""

    slug: str
    title: str
    summary: str
    source_path: str
    doc_url: str
    audience: str


def _action_view(action: Optional[TrainingAction]) -> Optional[PublicActionView]:
    if action is None:
        return None
    return PublicActionView(
        id=action.id,
        label=action.label,
        icon=action.icon,
        tone=action.tone,
    )


def build_user_summary(
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
) -> ProductUserSummaryView:
    """Build the public-facing summary from the shared training presentation model."""
    presentation: TrainingPresentation = build_training_run_presentation(
        job_status=job_status,
        quality_status=quality_status,
        quality_summary=quality_summary,
        recovery_status=recovery_status,
        recovery_action=recovery_action,
        recovery_summary=recovery_summary,
        failure_reason=failure_reason,
        final_reason=final_reason,
        has_launch_context=has_launch_context,
        can_resume_latest=can_resume_latest,
        weights_updated=weights_updated,
        has_quality_details=has_quality_details,
    )
    primary = _action_view(presentation.primary_action)
    secondary = [_action_view(action) for action in presentation.secondary_actions]
    secondary = [action for action in secondary if action is not None]
    next_step = (
        primary.label
        if primary is not None
        else (secondary[0].label if secondary else "Review research details")
    )
    return ProductUserSummaryView(
        headline=presentation.headline_status,
        why_it_matters=presentation.supporting_summary,
        next_step=next_step,
        confidence_tone=presentation.confidence_tone,
        primary_action=primary,
        secondary_actions=secondary,
    )


def to_dict(view: Any) -> Dict[str, Any]:
    """Convert a public view model to a JSON-ready dictionary."""
    return _serialize(view)
