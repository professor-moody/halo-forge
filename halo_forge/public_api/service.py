"""Shared public product service built on top of internal halo-forge services."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from halo_forge.training_recovery import build_recovery_guidance
from ui.services.ops_readiness_service import OpsReadinessService, get_ops_readiness_service
from ui.services.quickstart_presets import list_quickstart_presets
from ui.services.results_service import ResultsService, TrainingRunSummary, get_results_service
from ui.services.training_presentation import build_launch_presentation
from ui.services.training_service import TrainingLaunchPreflight, TrainingService
from ui.state import AppState, JobState, state as default_state

from .views import (
    DocsCapabilitySummaryView,
    ModalityReadinessView,
    ProductUserSummaryView,
    TrainingLaunchPreflightView,
    TrainingRecoveryView,
    TrainingRunDetailView,
    TrainingRunListItemView,
    TrainingRunLiveView,
    build_user_summary,
    to_dict,
)


TRAINING_MODALITIES = ("sft", "raft", "vlm", "audio", "reasoning", "agentic")


@dataclass(frozen=True)
class _DocsSource:
    slug: str
    path: Path
    audience: str
    doc_url: str


class PublicApiService:
    """Pure service layer shared by the public API and public frontend."""

    DOC_SOURCES = (
        _DocsSource(
            slug="public-frontend",
            path=Path("website/hugo-docs/content/docs/reference/public-frontend.md"),
            audience="product",
            doc_url="/docs/public-frontend",
        ),
        _DocsSource(
            slug="web-ui-console",
            path=Path("website/hugo-docs/content/docs/reference/web-ui.md"),
            audience="research",
            doc_url="/docs/reference/web-ui",
        ),
        _DocsSource(
            slug="modality-readiness",
            path=Path("website/hugo-docs/content/docs/experimental.md"),
            audience="product",
            doc_url="/docs/experimental",
        ),
        _DocsSource(
            slug="local-docs-index",
            path=Path("docs/README.md"),
            audience="research",
            doc_url="/docs/local",
        ),
    )

    def __init__(
        self,
        *,
        app_state: AppState | None = None,
        results_service: ResultsService | None = None,
        readiness_service: OpsReadinessService | None = None,
        training_service: TrainingService | None = None,
        base_path: Path | None = None,
    ) -> None:
        self.app_state = app_state or default_state
        self.base_path = (base_path or Path.cwd()).resolve()
        self.results_service = results_service or get_results_service()
        self.readiness_service = readiness_service or get_ops_readiness_service()
        self.training_service = training_service or TrainingService(self.app_state)

    def list_training_presets(self) -> list[dict[str, Any]]:
        """Return public-safe quickstart presets for training."""
        items: list[dict[str, Any]] = []
        for preset in list_quickstart_presets("training"):
            items.append(
                {
                    "key": preset.key,
                    "mode": preset.target,
                    "label": preset.label,
                    "description": preset.description,
                    "when_to_use": preset.recommendation.when_to_use,
                    "expected_runtime": preset.recommendation.expected_runtime,
                    "yield_safety": preset.recommendation.yield_safety,
                    "required_fields": list(preset.field_set.required_fields),
                    "optional_fields": list(preset.field_set.optional_fields),
                    "values": dict(preset.values),
                }
            )
        return items

    def preflight_training(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run launch preflight for the requested training mode."""
        mode = str(payload.get("mode") or "").strip().lower()
        if mode not in TRAINING_MODALITIES:
            raise ValueError(f"Unsupported training mode: {mode}")

        if mode == "sft":
            preflight = self.training_service.preflight_sft_launch(
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                epochs=int(payload.get("epochs") or 1),
                batch_size=int(payload.get("batch_size") or 2),
                gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps") or 4),
                max_samples=self._optional_int(payload.get("max_samples")),
            )
        elif mode == "raft":
            preflight = self.training_service.preflight_raft_launch(
                model=str(payload.get("model") or ""),
                prompts=str(payload.get("prompts") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                cycles=int(payload.get("cycles") or 1),
                samples_per_prompt=int(payload.get("samples_per_prompt") or 4),
                keep_percent=float(payload.get("keep_percent") or 0.5),
                reward_threshold=float(payload.get("reward_threshold") or 0.5),
                min_samples=int(payload.get("min_samples") or 1),
                max_new_tokens=int(payload.get("max_new_tokens") or 512),
                checkpoint=self._optional_str(payload.get("checkpoint")),
            )
        else:
            preflight = self.training_service.preflight_modality_train_launch(
                modality=mode,
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                cycles=int(payload.get("cycles") or 1),
                resume_from_cycle=int(payload.get("resume_from_cycle") or 0),
                seed=int(payload.get("seed") or 42),
                allow_prototype_train=bool(payload.get("allow_prototype_train", False)),
                limit=self._optional_int(payload.get("limit")),
                task=self._optional_str(payload.get("task")),
                samples_per_prompt=self._optional_int(payload.get("samples_per_prompt")),
                keep_percent=self._optional_float(payload.get("keep_percent")),
                reward_threshold=self._optional_float(payload.get("reward_threshold")),
            )

        return to_dict(self._build_preflight_view(mode=mode, preflight=preflight))

    async def launch_training(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Launch training from the public API."""
        mode = str(payload.get("mode") or "").strip().lower()
        if mode == "sft":
            job_id = await self.training_service.launch_sft(
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                epochs=int(payload.get("epochs") or 1),
                batch_size=int(payload.get("batch_size") or 2),
                gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps") or 4),
                max_samples=self._optional_int(payload.get("max_samples")),
                source_ui_page="/public/train",
            )
        elif mode == "raft":
            job_id = await self.training_service.launch_raft(
                model=str(payload.get("model") or ""),
                prompts=str(payload.get("prompts") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                verifier=str(payload.get("verifier") or "humaneval"),
                cycles=int(payload.get("cycles") or 1),
                samples_per_prompt=int(payload.get("samples_per_prompt") or 4),
                temperature=float(payload.get("temperature") or 0.7),
                keep_percent=float(payload.get("keep_percent") or 0.5),
                reward_threshold=float(payload.get("reward_threshold") or 0.5),
                min_samples=int(payload.get("min_samples") or 1),
                max_new_tokens=int(payload.get("max_new_tokens") or 512),
                source_ui_page="/public/train",
            )
        elif mode in {"vlm", "audio", "reasoning", "agentic"}:
            job_id = await self.training_service.launch_modality_train(
                modality=mode,
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                cycles=int(payload.get("cycles") or 1),
                learning_rate=self._optional_float(payload.get("learning_rate")),
                lr_decay=self._optional_float(payload.get("lr_decay")),
                samples_per_prompt=self._optional_int(payload.get("samples_per_prompt")),
                temperature=self._optional_float(payload.get("temperature")),
                keep_percent=self._optional_float(payload.get("keep_percent")),
                reward_threshold=self._optional_float(payload.get("reward_threshold")),
                task=self._optional_str(payload.get("task")),
                limit=self._optional_int(payload.get("limit")),
                resume_from_cycle=int(payload.get("resume_from_cycle") or 0),
                seed=int(payload.get("seed") or 42),
                allow_prototype_train=bool(payload.get("allow_prototype_train", False)),
                source_ui_page="/public/train",
            )
        else:
            raise ValueError(f"Unsupported training mode: {mode}")
        return self.get_run_detail(job_id, include_research=True, include_internal=False)

    async def apply_guided_recovery(
        self,
        run_identifier: str,
        *,
        resume_latest: bool = False,
    ) -> Dict[str, Any]:
        """Apply guided recovery using the stored launch context."""
        detail = self._resolve_run_source(run_identifier)
        launch_context = detail["launch_context_path"]
        recovery = detail["recovery"]
        if not launch_context:
            raise ValueError("This run does not have relaunch context.")
        if recovery.status != "ready":
            raise ValueError("Guided recovery is not available for this run.")

        job_id = await self.training_service.relaunch_from_context(
            launch_context,
            resume_latest=resume_latest,
            override_args=recovery.suggested_overrides,
            guided_recovery={
                "reason_code": recovery.reason_code,
                "evidence_summary": recovery.evidence_summary,
            },
            source_ui_page="/public/results",
        )
        return self.get_run_detail(job_id, include_research=True, include_internal=False)

    def list_runs(
        self,
        *,
        include_completed: bool = True,
        active_only: bool = False,
        include_research: bool = False,
    ) -> Dict[str, Any]:
        """List training runs for public monitor and results pages."""
        items: list[TrainingRunListItemView] = []
        seen_keys: set[str] = set()

        summaries = self.results_service.list_training_runs(force_refresh=True) if include_completed else []
        for summary in summaries:
            item = self._summary_to_list_item(summary, include_research=include_research)
            items.append(item)
            seen_keys.add(str(summary.output_dir.resolve()))

        if not include_completed or active_only:
            items = []

        active_jobs = sorted(
            [job for job in self.app_state.jobs.values() if job.type in TRAINING_MODALITIES],
            key=lambda job: job.created_at,
            reverse=True,
        )
        for job in active_jobs:
            output_key = str(job.output_dir.resolve()) if job.output_dir else ""
            if job.status == "completed" and output_key in seen_keys:
                continue
            if active_only and job.status not in {"pending", "running"}:
                continue
            items.append(self._job_to_list_item(job, include_research=include_research))

        items.sort(key=lambda item: item.timestamp, reverse=True)
        return {"items": [to_dict(item) for item in items]}

    def get_run_detail(
        self,
        run_identifier: str,
        *,
        include_research: bool = True,
        include_internal: bool = False,
    ) -> Dict[str, Any]:
        """Resolve a run from active job state or persisted training summaries."""
        resolved = self._resolve_run_source(run_identifier)
        if resolved["kind"] == "job":
            view = self._job_to_detail_view(
                resolved["job"],
                include_research=include_research,
                include_internal=include_internal,
            )
        else:
            view = self._summary_to_detail_view(
                resolved["summary"],
                include_research=include_research,
                include_internal=include_internal,
            )
        return to_dict(view)

    def get_run_live(
        self,
        run_identifier: str,
        *,
        include_research: bool = True,
    ) -> Dict[str, Any]:
        """Return a polling-friendly live view for a run."""
        resolved = self._resolve_run_source(run_identifier)
        if resolved["kind"] == "job":
            return to_dict(
                self._job_to_live_view(
                    resolved["job"],
                    include_research=include_research,
                )
            )

        summary = resolved["summary"]
        detail = self._summary_to_detail_view(
            summary,
            include_research=include_research,
            include_internal=False,
        )
        return to_dict(
            TrainingRunLiveView(
                id=detail.id,
                status=detail.status,
                progress_percent=100.0 if detail.status == "completed" else 0.0,
                current_step=int(detail.details.get("update_steps") or 0),
                total_steps=int(detail.details.get("update_steps") or 0),
                current_epoch=0.0,
                total_epochs=0,
                current_cycle=int(detail.details.get("cycles_executed") or 0),
                total_cycles=int(detail.details.get("cycles_executed") or 0),
                latest_loss=detail.details.get("final_train_loss"),
                latest_learning_rate=None,
                latest_grad_norm=None,
                user_summary=detail.user_summary,
                research_details=detail.research_details,
            )
        )

    async def stream_run(self, run_identifier: str, *, include_research: bool = True):
        """Stream polling snapshots as server-sent events."""
        while True:
            payload = self.get_run_live(run_identifier, include_research=include_research)
            yield f"data: {json.dumps(payload)}\n\n"
            status = str(payload.get("status") or "").lower()
            if status in {"completed", "failed", "stopped"}:
                break
            await asyncio.sleep(1.0)

    def list_training_results(self, *, include_research: bool = False) -> Dict[str, Any]:
        """Return completed training results for the public results page."""
        items = [
            self._summary_to_list_item(summary, include_research=include_research)
            for summary in self.results_service.list_training_runs(force_refresh=True)
        ]
        items.sort(key=lambda item: item.timestamp, reverse=True)
        return {"items": [to_dict(item) for item in items]}

    def list_readiness(self) -> Dict[str, Any]:
        """Return public-safe readiness for training modalities."""
        report = self.readiness_service.load_qualification_report(force_refresh=True)
        items: list[ModalityReadinessView] = []
        counts = {"experimental": 0, "qualified": 0, "production_ready": 0}
        for module in TRAINING_MODALITIES:
            entry = report.modules.get(module)
            if entry is None:
                continue
            tier = str(getattr(entry, "readiness_tier", "") or "experimental")
            if tier in counts:
                counts[tier] += 1
            items.append(
                ModalityReadinessView(
                    modality=module,
                    readiness_tier=tier,
                    production_ready=bool(getattr(entry, "production_ready", False)),
                    status=str(getattr(entry, "status", "warn") or "warn"),
                    caveat=self._readiness_caveat(entry),
                    next_step=str(getattr(entry, "fix_now", "") or "Review readiness details before wider rollout."),
                    eval_metric_name=str(getattr(entry, "eval_metric_name", "") or ""),
                    baseline_value=getattr(entry, "baseline_value", None),
                    final_value=getattr(entry, "final_value", None),
                    delta=getattr(entry, "delta", None),
                    details={
                        "errors": list(getattr(entry, "errors", []) or []),
                        "warnings": list(getattr(entry, "warnings", []) or []),
                        "weights_updated": bool(getattr(entry, "weights_updated", False)),
                        "optimizer_steps": int(getattr(entry, "optimizer_steps", 0) or 0),
                        "samples_kept": int(getattr(entry, "samples_kept", 0) or 0),
                    },
                )
            )
        aggregate_tier = "experimental"
        if counts["production_ready"] == len(items) and items:
            aggregate_tier = "production_ready"
        elif counts["production_ready"] > 0 or counts["qualified"] > 0:
            aggregate_tier = "qualified"
        return {
            "generated_at": getattr(report, "generated_at", None),
            "aggregate_tier": aggregate_tier,
            "items": [to_dict(item) for item in items],
        }

    def list_docs_capabilities(self) -> Dict[str, Any]:
        """Return curated docs summaries for the public docs page."""
        items: list[DocsCapabilitySummaryView] = []
        for source in self.DOC_SOURCES:
            if not source.path.exists():
                continue
            title, summary = self._markdown_title_and_summary(source.path)
            items.append(
                DocsCapabilitySummaryView(
                    slug=source.slug,
                    title=title,
                    summary=summary,
                    source_path=str(source.path),
                    doc_url=source.doc_url,
                    audience=source.audience,
                )
            )
        return {"items": [to_dict(item) for item in items]}

    def _resolve_run_source(self, run_identifier: str) -> Dict[str, Any]:
        identifier = str(run_identifier or "").strip()
        job = self.app_state.get_job(identifier)
        if job is not None and job.type in TRAINING_MODALITIES:
            return {
                "kind": "job",
                "job": job,
                "launch_context_path": str(job.launch_context_file) if job.launch_context_file else None,
                "recovery": self._job_recovery(job),
            }

        summaries = self.results_service.list_training_runs(force_refresh=True)
        for summary in summaries:
            if identifier in {
                summary.id,
                str(summary.run_id or ""),
                summary.output_dir.name,
            }:
                return {
                    "kind": "summary",
                    "summary": summary,
                    "launch_context_path": str(summary.launch_context_path) if summary.launch_context_path else None,
                    "recovery": self._summary_recovery(summary),
                }
        raise KeyError(f"Training run not found: {identifier}")

    def _build_preflight_view(
        self,
        *,
        mode: str,
        preflight: TrainingLaunchPreflight,
    ) -> TrainingLaunchPreflightView:
        outlook = dict(preflight.quality_outlook or {})
        launch_presentation = build_launch_presentation(
            mode_label=mode.upper(),
            quality_status=str(outlook.get("status") or "healthy"),
            quality_summary=str(outlook.get("summary") or ""),
            suggested_adjustments=[
                str(item)
                for item in outlook.get("suggested_adjustments", [])
                if item is not None
            ],
            yield_safety_note=str(outlook.get("yield_safety_note") or ""),
        )
        return TrainingLaunchPreflightView(
            mode=mode,
            ok=bool(preflight.ok),
            resolved_paths=dict(preflight.resolved_paths),
            errors=list(preflight.errors),
            warnings=list(preflight.warnings),
            suggested_fixes=list(preflight.suggested_fixes),
            user_summary=ProductUserSummaryView(
                headline=launch_presentation.headline_status,
                why_it_matters=launch_presentation.supporting_summary,
                next_step=(
                    "Fix required inputs before launch"
                    if preflight.errors
                    else "Launch run when ready"
                ),
                confidence_tone=launch_presentation.confidence_tone,
            ),
            details={
                "quality_outlook": outlook,
                "recommended_adjustment": launch_presentation.recommended_adjustment,
            },
        )

    def _summary_to_list_item(
        self,
        summary: TrainingRunSummary,
        *,
        include_research: bool,
    ) -> TrainingRunListItemView:
        status = "failed" if summary.failure_reason else "completed"
        user_summary = build_user_summary(
            job_status=status,
            quality_status=summary.quality_status,
            quality_summary=summary.quality_summary,
            recovery_status=summary.recovery_status,
            recovery_action=summary.recovery_recommended_action,
            recovery_summary=summary.recovery_summary,
            failure_reason=summary.failure_reason,
            final_reason=summary.final_update_reason,
            has_launch_context=summary.has_relaunch_context,
            can_resume_latest=summary.modality in {"raft", "vlm", "audio", "reasoning", "agentic"},
            weights_updated=summary.weights_updated,
        )
        return TrainingRunListItemView(
            id=summary.id,
            run_id=str(summary.run_id or summary.id),
            modality=summary.modality,
            model_name=summary.model_name,
            status=status,
            timestamp=self._isoformat(summary.timestamp),
            progress_percent=100.0 if status == "completed" else 0.0,
            user_summary=user_summary,
            details={
                "verdict": summary.effectiveness_verdict,
                "keep_rate": summary.keep_rate,
                "quality_status": summary.quality_status,
                "top_issue": summary.dominant_rejection_reason,
                "update_steps": summary.total_train_steps_executed,
                "final_train_loss": summary.final_train_loss,
            },
            research_details=(
                {
                    "yield_diagnostics": summary.yield_diagnostics,
                    "effectiveness": dict(summary.raw_data.get("effectiveness") or {}),
                    "recovery_guidance": self._summary_recovery(summary),
                    "representative_examples": list(summary.representative_examples),
                }
                if include_research
                else {}
            ),
        )

    def _summary_to_detail_view(
        self,
        summary: TrainingRunSummary,
        *,
        include_research: bool,
        include_internal: bool,
    ) -> TrainingRunDetailView:
        item = self._summary_to_list_item(summary, include_research=include_research)
        recovery = self._summary_recovery(summary)
        return TrainingRunDetailView(
            id=item.id,
            run_id=item.run_id,
            modality=item.modality,
            model_name=item.model_name,
            status=item.status,
            timestamp=item.timestamp,
            progress_percent=item.progress_percent,
            user_summary=item.user_summary,
            recovery=recovery,
            details={
                **item.details,
                "cycles_executed": summary.cycles_executed,
                "seed": summary.seed,
                "resume_from_cycle": summary.resume_from_cycle,
                "final_model_available": bool(summary.final_model_path),
            },
            research_details=item.research_details,
            internal_details=(
                {
                    "output_dir": str(summary.output_dir),
                    "final_model_path": summary.final_model_path,
                    "launch_context_path": (
                        str(summary.launch_context_path)
                        if summary.launch_context_path
                        else None
                    ),
                }
                if include_internal
                else {}
            ),
        )

    def _job_to_list_item(
        self,
        job: JobState,
        *,
        include_research: bool,
    ) -> TrainingRunListItemView:
        live_yield = dict(job.latest_yield_snapshot or {})
        yield_summary = live_yield.get("summary") if isinstance(live_yield.get("summary"), dict) else {}
        yield_rates = live_yield.get("rates") if isinstance(live_yield.get("rates"), dict) else {}
        recovery = self._job_recovery(job)
        user_summary = build_user_summary(
            job_status=job.status,
            quality_status=str(yield_summary.get("status") or ""),
            quality_summary=str(yield_summary.get("text") or ""),
            recovery_status=recovery.status,
            recovery_action=recovery.recommended_action,
            recovery_summary=recovery.evidence_summary,
            failure_reason=job.error_message,
            final_reason=job.lifecycle_metadata.get("resume_strategy") if job.lifecycle_metadata else "",
            has_launch_context=bool(job.launch_context_file),
            can_resume_latest=job.type in {"raft", "vlm", "audio", "reasoning", "agentic"},
            weights_updated=job.current_step > 0 or job.current_cycle > 0,
        )
        return TrainingRunListItemView(
            id=job.id,
            run_id=job.id,
            modality=job.type,
            model_name=job.name,
            status=job.status,
            timestamp=self._isoformat(job.created_at),
            progress_percent=job.progress_percent,
            user_summary=user_summary,
            details={
                "quality_status": yield_summary.get("status"),
                "keep_rate": yield_rates.get("keep_rate"),
                "top_issue": yield_summary.get("dominant_rejection_reason"),
                "update_steps": job.current_step,
                "final_train_loss": job.latest_loss,
            },
            research_details=(
                {
                    "yield_diagnostics": live_yield,
                    "yield_history": list(job.yield_history),
                }
                if include_research
                else {}
            ),
        )

    def _job_to_detail_view(
        self,
        job: JobState,
        *,
        include_research: bool,
        include_internal: bool,
    ) -> TrainingRunDetailView:
        item = self._job_to_list_item(job, include_research=include_research)
        recovery = self._job_recovery(job)
        return TrainingRunDetailView(
            id=item.id,
            run_id=item.run_id,
            modality=item.modality,
            model_name=item.model_name,
            status=item.status,
            timestamp=item.timestamp,
            progress_percent=item.progress_percent,
            user_summary=item.user_summary,
            recovery=recovery,
            details={
                **item.details,
                "current_epoch": job.current_epoch,
                "total_epochs": job.total_epochs,
                "current_cycle": job.current_cycle,
                "total_cycles": job.total_cycles,
                "verification_rate": job.verification_rate,
            },
            research_details=item.research_details,
            internal_details=(
                {
                    "output_dir": str(job.output_dir) if job.output_dir else None,
                    "launch_context_path": (
                        str(job.launch_context_file) if job.launch_context_file else None
                    ),
                    "lifecycle_metadata": dict(job.lifecycle_metadata),
                }
                if include_internal
                else {}
            ),
        )

    def _job_to_live_view(
        self,
        job: JobState,
        *,
        include_research: bool,
    ) -> TrainingRunLiveView:
        detail = self._job_to_detail_view(
            job,
            include_research=include_research,
            include_internal=False,
        )
        return TrainingRunLiveView(
            id=detail.id,
            status=detail.status,
            progress_percent=detail.progress_percent,
            current_step=job.current_step,
            total_steps=job.total_steps,
            current_epoch=job.current_epoch,
            total_epochs=job.total_epochs,
            current_cycle=job.current_cycle,
            total_cycles=job.total_cycles,
            latest_loss=job.latest_loss,
            latest_learning_rate=job.latest_lr,
            latest_grad_norm=job.latest_grad_norm,
            user_summary=detail.user_summary,
            research_details=detail.research_details,
        )

    def _summary_recovery(self, summary: TrainingRunSummary) -> TrainingRecoveryView:
        return TrainingRecoveryView(
            status=str(summary.recovery_status or "unavailable"),
            reason_code=str(summary.recovery_reason_code or ""),
            recommended_action=str(summary.recovery_recommended_action or ""),
            evidence_summary=str(summary.recovery_summary or ""),
            suggested_overrides=dict(summary.recovery_suggested_overrides),
            representative_examples=list(summary.representative_examples),
        )

    def _job_recovery(self, job: JobState) -> TrainingRecoveryView:
        live_yield = dict(job.latest_yield_snapshot or {})
        guidance = build_recovery_guidance(
            modality=str(job.type or "unknown"),
            yield_diagnostics=live_yield,
            effectiveness={
                "verdict": "pass" if job.current_step > 0 else "warn",
                "reasons": [],
            },
            launch_args=dict(job.launch_args),
        )
        return TrainingRecoveryView(
            status=str(guidance.get("status") or "unavailable"),
            reason_code=str(guidance.get("reason_code") or ""),
            recommended_action=str(guidance.get("recommended_action") or ""),
            evidence_summary=str(guidance.get("evidence_summary") or ""),
            suggested_overrides=(
                dict(guidance.get("suggested_overrides"))
                if isinstance(guidance.get("suggested_overrides"), dict)
                else {}
            ),
            representative_examples=[
                dict(example)
                for example in guidance.get("representative_examples", [])
                if isinstance(example, dict)
            ],
        )

    def _readiness_caveat(self, entry: Any) -> str:
        if bool(getattr(entry, "production_ready", False)):
            return "Deterministic launch, updates, artifacts, resume, and eval checks are currently passing."
        warnings = [str(item) for item in getattr(entry, "warnings", []) or [] if str(item).strip()]
        errors = [str(item) for item in getattr(entry, "errors", []) or [] if str(item).strip()]
        if errors:
            return errors[0]
        if warnings:
            return warnings[0]
        return str(getattr(entry, "fix_now", "") or "Qualification evidence is incomplete for this modality.")

    def _markdown_title_and_summary(self, path: Path) -> tuple[str, str]:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        title = path.stem.replace("-", " ").title()
        summary = "Documentation summary unavailable."
        index = 0
        if lines and lines[0].strip() == "---":
            index = 1
            while index < len(lines) and lines[index].strip() != "---":
                line = lines[index].strip()
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip('"')
                if line.startswith("description:"):
                    summary = line.split(":", 1)[1].strip().strip('"')
                index += 1
            index += 1
        for line in lines[index:]:
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                continue
            if stripped and not stripped.startswith(("-", "*", "|", "`")):
                summary = stripped
                break
        return title, summary

    @staticmethod
    def _isoformat(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        return int(value)

    @staticmethod
    def _optional_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        return float(value)

    @staticmethod
    def _optional_str(value: Any) -> Optional[str]:
        text = str(value or "").strip()
        return text or None
