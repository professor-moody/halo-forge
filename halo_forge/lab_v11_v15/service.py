"""Persistence and research services for Halo Forge Labs V11-V15."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import shutil
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from halo_forge.run_db.db import RunDatabase

from .environments import ENVIRONMENT_ADAPTERS
from .models import (
    ActionableGuidance,
    AdaptationStudy,
    AdaptationStudyProtocolRevision,
    AgentEnvironment,
    AgentEnvironmentRevision,
    AgentEpisode,
    AgentEpisodeStep,
    ClassificationPrediction,
    EmbeddingResult,
    EnvironmentEvaluationComparison,
    EnvironmentPermissionSummary,
    EnvironmentSubjectExecution,
    EnvironmentToolDescriptor,
    EpisodeSuiteRevision,
    GroundedCandidate,
    GroundedGenerationBatch,
    GroundingCitation,
    GroundingGenerationPreview,
    GroundingProfile,
    GroundingProfileRevision,
    PlannedContrast,
    GuidedAction,
    RerankResult,
    ScenarioEvaluationStarter,
    ScenarioOutcomeProfile,
    SpecializedTaskDescriptor,
    StudyAnalysis,
    StudyArm,
    StudyAssignment,
    StudyDecision,
    StudyDeviation,
    StudyLaunchPlan,
    TaskLabelSchemaRevision,
    TrainingOutcomeAssessment,
    TrainingOutcomeDecision,
    TrainingOutcomeFinding,
    OutcomePreparation,
    TrajectorySet,
    TrajectorySetRevision,
)


class FutureLabError(RuntimeError):
    """Base domain failure for V11-V15 operations."""


class EvidenceEligibilityError(FutureLabError):
    """Protected evidence was supplied to a training-guiding operation."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, value: Any, *, length: int = 24) -> str:
    return f"{prefix}-{_content_hash(value)[:length]}"


def _loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except (TypeError, json.JSONDecodeError):
        return default


def _page(items: Sequence[Any], *, limit: int, offset: int) -> Dict[str, Any]:
    bounded_limit = max(1, min(int(limit), 1000))
    bounded_offset = max(0, int(offset))
    return {
        "items": list(items[bounded_offset : bounded_offset + bounded_limit]),
        "total": len(items),
        "limit": bounded_limit,
        "offset": bounded_offset,
    }


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _atomic_bundle(root: Path, files: Mapping[str, str | bytes]) -> tuple[Path, str]:
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    manifest: Dict[str, str] = {}
    try:
        for relative, content in files.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = content.encode("utf-8") if isinstance(content, str) else bytes(content)
            destination.write_bytes(payload)
            manifest[relative] = hashlib.sha256(payload).hexdigest()
        manifest_payload = {
            "files": manifest,
            "content_hash": _content_hash(manifest),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if root.exists():
            shutil.rmtree(staging, ignore_errors=True)
        else:
            staging.replace(root)
        return root, str(manifest_payload["content_hash"])
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _row_dict(row: Any) -> Dict[str, Any]:
    return {key: row[key] for key in row.keys()}


_OUTCOME_STARTERS = {
    "instruction-sft": ("dataset_split", "score", "maximize"),
    "chat-sft": ("dataset_split", "score", "maximize"),
    "preference-pairs": ("preference", "preference_accuracy", "maximize"),
    "prompt-reward": ("verifier", "pass_rate", "maximize"),
    "reasoning-sft": ("reasoning", "score", "maximize"),
    "tool-agentic": ("tool", "task_success", "maximize"),
    "vlm-captioning": ("vlm", "score", "maximize"),
    "vlm-qa": ("vlm", "score", "maximize"),
    "audio-asr": ("audio", "wer", "minimize"),
    "corpus-adaptation": ("corpus_perplexity", "perplexity", "minimize"),
    "text-classification": ("classification", "macro_f1", "maximize"),
    "text-multilabel": ("classification", "micro_f1", "maximize"),
    "embedding-pairs": ("retrieval", "recall_at_k", "maximize"),
    "reranking": ("reranking", "ndcg", "maximize"),
    "image-classification": ("classification", "macro_f1", "maximize"),
    "audio-classification": ("classification", "macro_f1", "maximize"),
}


_SPECIALIZED_TASKS = (
    SpecializedTaskDescriptor(
        id="text-classification",
        label="Text classification",
        task_kind="classification",
        modality="text",
        canonical_schema="classification",
        trainer_mode="classify",
        metrics=("accuracy", "balanced_accuracy", "macro_f1", "mcc"),
        available=True,
    ),
    SpecializedTaskDescriptor(
        id="text-multilabel",
        label="Text multi-label classification",
        task_kind="multilabel",
        modality="text",
        canonical_schema="classification",
        trainer_mode="classify",
        metrics=("micro_f1", "macro_f1", "hamming_loss", "exact_match"),
        available=True,
    ),
    SpecializedTaskDescriptor(
        id="embedding-pairs",
        label="Embedding model",
        task_kind="embedding",
        modality="text",
        canonical_schema="embedding",
        trainer_mode="embed",
        metrics=("recall_at_k", "mrr", "ndcg"),
        available=True,
    ),
    SpecializedTaskDescriptor(
        id="reranking",
        label="Reranker",
        task_kind="reranking",
        modality="text",
        canonical_schema="reranking",
        trainer_mode="rerank",
        metrics=("mrr", "ndcg", "pairwise_accuracy"),
        available=True,
    ),
    SpecializedTaskDescriptor(
        id="image-classification",
        label="Image classification",
        task_kind="classification",
        modality="image",
        canonical_schema="classification",
        trainer_mode="classify",
        metrics=("accuracy", "balanced_accuracy", "macro_f1"),
        available=True,
    ),
    SpecializedTaskDescriptor(
        id="audio-classification",
        label="Audio classification",
        task_kind="classification",
        modality="audio",
        canonical_schema="classification",
        trainer_mode="classify",
        metrics=("accuracy", "balanced_accuracy", "macro_f1"),
        available=True,
    ),
)


class FutureLabService:
    """Shared implementation for outcome, study, grounding, task, and environment flows."""

    def __init__(
        self,
        db: RunDatabase,
        *,
        root: str | Path | None = None,
        scheduler: Any = None,
        teacher: Optional[Callable[..., str]] = None,
    ) -> None:
        self.db = db
        self.root = Path(root or Path.home() / ".halo-forge").expanduser().resolve()
        self.scheduler = scheduler
        self.teacher = teacher
        self.outcome_root = self.root / "evaluations" / "outcomes"
        self.study_root = self.root / "studies"
        self.grounding_root = self.root / "grounding"
        self.environment_root = self.root / "environments"
        self.episode_root = self.root / "episodes"
        self.trajectory_root = self.root / "trajectories"
        self._ensure_outcome_profiles()

    # ----- common -------------------------------------------------------

    @property
    def conn(self):
        return self.db._conn

    def _commit(self) -> None:
        self.conn.commit()

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        return {
            "outcomes": {
                "states": [
                    "improved",
                    "regressed",
                    "mixed",
                    "no_clear_change",
                    "incomplete_evidence",
                    "technical_failure",
                ],
                "full_run_gate": "assessment_or_reasoned_override",
            },
            "studies": {
                "designs": ["paired_ab", "dose_response", "factorial_2x2"],
                "default_seeds": [17, 42, 101],
                "max_arms": 4,
                "max_doses": 5,
            },
            "grounding": {
                "task_types": [
                    "qa",
                    "instruction",
                    "extraction",
                    "reasoning",
                    "preference",
                    "benchmark",
                ],
                "destinations": ["training", "development_evaluation"],
            },
            "specialized_tasks": [value.to_dict() for value in _SPECIALIZED_TASKS],
            "environments": {
                "adapters": ["state_machine"],
                "adapter_descriptors": [
                    {
                        "id": value.id,
                        "version": value.version,
                        "deterministic": value.deterministic,
                        "supports_snapshot": value.supports_snapshot,
                        "external_writes": value.external_writes,
                    }
                    for value in ENVIRONMENT_ADAPTERS.list()
                ],
                "external_writes": False,
                "online_rl": False,
                "trace_replay": True,
                "model_rerun": True,
            },
        }

    # ----- V16 actionable guidance ------------------------------------

    @staticmethod
    def _action(
        action_id: str,
        label: str,
        *,
        href: Optional[str] = None,
        method: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
        confirmation: bool = False,
        tone: str = "primary",
    ) -> GuidedAction:
        return GuidedAction(
            id=action_id,
            label=label,
            href=href,
            method=method,
            payload=dict(payload or {}),
            requires_confirmation=confirmation,
            tone=tone,
        )

    def outcome_guidance(
        self, assessment: TrainingOutcomeAssessment
    ) -> ActionableGuidance:
        status = assessment.status
        if status in {"queued", "running"}:
            return ActionableGuidance(
                context_kind="training_outcome",
                context_id=assessment.id,
                display_status="Checking training result",
                summary="Halo Forge is comparing the base model and proof model on the same development examples.",
                primary_action=self._action(
                    "open_activity",
                    "View progress",
                    href=f"/runs/{assessment.proof_run_id}?tab=evaluation",
                ),
                technical_details={
                    "status": status,
                    "stage": assessment.stage,
                    "work_item_id": assessment.work_item_id,
                },
            )
        if status == "improved":
            return ActionableGuidance(
                context_kind="training_outcome",
                context_id=assessment.id,
                display_status="Ready to continue",
                summary="The proof model improved on the selected development examples and the training update is technically valid.",
                primary_action=self._action(
                    "start_full_run",
                    "Start full run",
                    method="POST",
                    payload={"assessment_id": assessment.id},
                    confirmation=True,
                ),
                secondary_actions=(
                    self._action(
                        "compare_approaches",
                        "Compare approaches",
                        href=f"/sweeps?section=studies&proofRun={assessment.proof_run_id}",
                        tone="secondary",
                    ),
                ),
                technical_details={"status": status, **assessment.summary},
            )
        if status in {"mixed", "no_clear_change"}:
            return ActionableGuidance(
                context_kind="training_outcome",
                context_id=assessment.id,
                display_status="Review the tradeoff",
                summary="The proof completed, but the development evidence does not show one clear overall improvement.",
                primary_action=self._action(
                    "review_examples",
                    "Review examples",
                    href=f"/runs/{assessment.proof_run_id}?tab=evaluation",
                ),
                secondary_actions=(
                    self._action(
                        "compare_approaches",
                        "Compare approaches",
                        href=f"/sweeps?section=studies&proofRun={assessment.proof_run_id}",
                        tone="secondary",
                    ),
                ),
                technical_details={"status": status, **assessment.summary},
            )
        if status == "regressed":
            return ActionableGuidance(
                context_kind="training_outcome",
                context_id=assessment.id,
                display_status="Needs repair",
                summary="The proof model performed worse on the selected development examples.",
                primary_action=self._action(
                    "repair",
                    "Fix data or settings",
                    href=f"/runs/{assessment.proof_run_id}?tab=evaluation",
                ),
                secondary_actions=(
                    self._action(
                        "continue_anyway",
                        "Continue anyway",
                        method="POST",
                        confirmation=True,
                        tone="warning",
                    ),
                ),
                technical_details={"status": status, **assessment.summary},
            )
        if status == "technical_failure":
            return ActionableGuidance(
                context_kind="training_outcome",
                context_id=assessment.id,
                display_status="Training did not work",
                summary="The proof run did not produce a verified model update.",
                primary_action=self._action(
                    "open_fix_guide",
                    "Open fix guide",
                    href=f"/runs/{assessment.proof_run_id}?tab=logs",
                ),
                secondary_actions=(
                    self._action(
                        "retry",
                        "Retry proof run",
                        href=f"/train?parentRun={assessment.proof_run_id}",
                        tone="secondary",
                    ),
                ),
                technical_details={"status": status, **assessment.summary},
            )
        return ActionableGuidance(
            context_kind="training_outcome",
            context_id=assessment.id,
            display_status="More evidence needed",
            summary="The proof update is available, but Halo Forge still needs a compatible development comparison.",
            primary_action=self._action(
                "complete_evaluation",
                "Complete evaluation",
                href=f"/runs/{assessment.proof_run_id}?tab=evaluation",
            ),
            technical_details={"status": status, **assessment.summary},
        )

    def actionable_guidance(
        self, context_kind: str, context_id: str
    ) -> ActionableGuidance:
        kind = str(context_kind or "").strip().lower()
        if kind == "training_outcome":
            assessment = self.get_outcome_assessment(context_id)
            if assessment is None:
                raise ValueError(f"unknown training outcome: {context_id}")
            return self.outcome_guidance(assessment)
        if kind == "proof_run":
            run = self.db.get_run(context_id)
            if run is None:
                raise ValueError(f"unknown proof run: {context_id}")
            return ActionableGuidance(
                context_kind=kind,
                context_id=context_id,
                display_status="Proof run complete" if run.status == "completed" else "Proof run in progress",
                summary=(
                    "Check whether the model changed in the intended direction before committing to a full run."
                    if run.status == "completed"
                    else "Halo Forge will offer a result check when the proof run finishes."
                ),
                primary_action=(
                    self._action(
                        "check_training_result",
                        "Check training result",
                        method="POST",
                    )
                    if run.status == "completed"
                    else self._action(
                        "open_run",
                        "View progress",
                        href=f"/runs/{context_id}",
                    )
                ),
                technical_details={"status": run.status},
            )
        if kind in {"data", "dataset_version"}:
            version = self.db.get_dataset_version(context_id)
            if version is None:
                raise ValueError(f"unknown dataset version: {context_id}")
            corpus = str(version.schema or "").lower() == "corpus"
            return ActionableGuidance(
                context_kind="dataset_version",
                context_id=context_id,
                display_status="Dataset version is ready",
                summary=(
                    "Create cited examples from these documents, then review them before building training data."
                    if corpus
                    else "Use this immutable version for a proof run before committing to full training."
                ),
                primary_action=(
                    self._action(
                        "create_examples",
                        "Create examples from documents",
                        href=f"/datasets/ground?sourceVersion={context_id}",
                    )
                    if corpus
                    else self._action(
                        "train_single",
                        "Train single",
                        href=f"/train?datasetVersion={context_id}",
                    )
                ),
                secondary_actions=(
                    self._action(
                        "inspect_version",
                        "Inspect version",
                        href=f"/datasets/versions/{context_id}",
                        tone="secondary",
                    ),
                ),
                technical_details={
                    "status": version.status,
                    "schema": version.schema,
                    "content_hash": version.content_hash,
                },
            )
        if kind == "run":
            run = self.db.get_run(context_id)
            if run is None:
                raise ValueError(f"unknown run: {context_id}")
            raw = run.raw
            if bool(raw.get("proof_run") or (raw.get("launch_config") or {}).get("proof_run")):
                return self.actionable_guidance("proof_run", context_id)
            completed = run.status == "completed"
            return ActionableGuidance(
                context_kind="run",
                context_id=context_id,
                display_status="Run complete" if completed else "Training in progress",
                summary=(
                    "Evaluate the completed model before choosing an artifact action."
                    if completed
                    else "Halo Forge is recording progress, resources, logs, data identity, and artifacts."
                ),
                primary_action=(
                    self._action(
                        "evaluate",
                        "Evaluate model",
                        href=f"/runs/{context_id}?tab=evaluation",
                    )
                    if completed
                    else self._action(
                        "view_progress",
                        "View progress",
                        href=f"/runs/{context_id}",
                    )
                ),
                secondary_actions=(
                    self._action(
                        "clone",
                        "Clone in Train",
                        href=f"/train?parentRun={context_id}",
                        tone="secondary",
                    ),
                ) if completed else (),
                technical_details={"status": run.status, "weights_updated": run.weights_updated},
            )
        if kind == "train":
            return ActionableGuidance(
                context_kind="train",
                context_id=context_id,
                display_status="Choose your training goal",
                summary="Start with your data. Halo Forge will inspect it, recommend a compatible scenario, and prepare a proof run.",
                primary_action=self._action(
                    "train_on_your_data",
                    "Train on your data",
                    href="/datasets/new",
                ),
                secondary_actions=(
                    self._action(
                        "working_example",
                        "Try a working example",
                        href="/datasets/new?example=1",
                        tone="secondary",
                    ),
                ),
                technical_details={},
            )
        if kind in {"experiments", "study"}:
            protocol = self.get_study_protocol(context_id)
            if protocol is None:
                raise ValueError(f"unknown study protocol revision: {context_id}")
            plan = self.study_launch_plan(context_id)
            return ActionableGuidance(
                context_kind="study",
                context_id=context_id,
                display_status=(
                    "Study is ready to launch" if not plan.blockers else "Study setup needed"
                ),
                summary=(
                    f"Halo Forge will prepare {plan.run_count} matched runs and keep improvement and retention evidence separate."
                    if not plan.blockers
                    else plan.blockers[0]
                ),
                primary_action=(
                    self._action(
                        "launch_study",
                        "Launch study",
                        method="POST",
                        confirmation=True,
                    )
                    if not plan.blockers
                    else self._action(
                        "complete_setup",
                        "Choose training setup",
                        href=f"/sweeps?section=studies&study={protocol.study_id}",
                    )
                ),
                technical_details=plan.to_dict(),
            )
        if kind in {"evaluate", "evaluation"}:
            evaluation = self.db.get_evaluation(context_id)
            if evaluation is None:
                raise ValueError(f"unknown evaluation: {context_id}")
            complete = evaluation.status == "completed"
            return ActionableGuidance(
                context_kind="evaluation",
                context_id=context_id,
                display_status="Evaluation complete" if complete else "Evaluation in progress",
                summary=(
                    "Inspect example-level changes before deciding whether to build reviewed data."
                    if complete
                    else "Halo Forge is collecting metrics and example evidence."
                ),
                primary_action=(
                    self._action(
                        "review_results",
                        "Review results",
                        href=f"/eval?evaluation={context_id}",
                    )
                    if complete
                    else self._action(
                        "view_progress",
                        "View progress",
                        href=f"/eval?evaluation={context_id}",
                    )
                ),
                secondary_actions=(
                    self._action(
                        "review_failures",
                        "Review failures",
                        href=f"/eval?section=failure-review&evaluation={context_id}",
                        tone="secondary",
                    ),
                ) if complete else (),
                technical_details={"status": evaluation.status, "stage": evaluation.stage},
            )
        raise ValueError(f"unsupported guidance context: {context_kind}")

    def study_launch_plan(self, revision_id: str) -> StudyLaunchPlan:
        protocol = self.get_study_protocol(revision_id)
        if protocol is None:
            raise ValueError(f"unknown study protocol revision: {revision_id}")
        seeds = tuple(int(value) for value in protocol.definition.get("seeds") or (17, 42, 101))
        blockers: list[str] = []
        if not protocol.definition.get("development_suite_revision_id"):
            blockers.append("Choose development evidence before launching the study.")
        for arm in protocol.arms:
            config = dict(arm.launch_config or {})
            if not (config.get("model") or protocol.definition.get("model")):
                blockers.append(f"Choose a model for {arm.name}.")
            if not (
                config.get("dataset_version_id")
                or config.get("dataset")
                or protocol.definition.get("dataset_version_id")
            ):
                blockers.append(f"Choose training data for {arm.name}.")
        historical_seconds = [
            float(value.get("elapsed_seconds") or 0)
            for value in protocol.definition.get("historical_estimates") or []
            if float(value.get("elapsed_seconds") or 0) > 0
        ]
        run_count = len(protocol.arms) * len(seeds)
        mean_seconds = (
            sum(historical_seconds) / len(historical_seconds)
            if historical_seconds
            else None
        )
        return StudyLaunchPlan(
            protocol_revision_id=revision_id,
            arm_count=len(protocol.arms),
            seed_count=len(seeds),
            run_count=run_count,
            estimated_seconds_low=(
                mean_seconds * run_count * 0.8 if mean_seconds is not None else None
            ),
            estimated_seconds_high=(
                mean_seconds * run_count * 1.25 if mean_seconds is not None else None
            ),
            estimated_storage_bytes=(
                int(protocol.definition["estimated_bytes_per_run"]) * run_count
                if protocol.definition.get("estimated_bytes_per_run")
                else None
            ),
            blockers=tuple(dict.fromkeys(blockers)),
            work_item_id=protocol.launch_work_item_id,
        )

    def grounding_preview(
        self, revision_id: str, payload: Mapping[str, Any]
    ) -> GroundingGenerationPreview:
        revision = self.get_grounding_profile_revision(revision_id)
        if revision is None:
            raise ValueError(f"unknown grounding profile revision: {revision_id}")
        preset = str(payload.get("preset") or "standard").strip().lower()
        limits = {"quick": 50, "standard": 250, "thorough": 1000}
        if preset not in limits:
            raise ValueError("grounding preset must be quick, standard, or thorough")
        records = self._source_records(
            source_version_id=str(payload.get("source_version_id") or "").strip() or None,
            records=payload.get("records"),
        )
        seed = int(revision.definition.get("seed") or 42)
        ordered = sorted(
            records,
            key=lambda value: _content_hash(
                {"seed": seed, "identity": value.get("document_id") or value}
            ),
        )
        task_type = str(revision.definition.get("task_type") or "qa")
        preview_items = tuple(
            self._default_grounded_output(task_type, record)
            for record in ordered[:10]
        )
        teacher = dict(revision.definition.get("teacher") or {})
        verifier = dict(revision.definition.get("verifier") or {})
        blockers: list[str] = []
        if str(revision.definition.get("intended_destination") or "") not in {
            "training",
            "development_evaluation",
        }:
            blockers.append("Protected evidence cannot be used for grounded generation.")
        if not records:
            blockers.append("The selected corpus contains no usable records.")
        hosted = str(teacher.get("endpoint_type") or "").lower() in {
            "hosted",
            "openai",
            "anthropic",
        }
        return GroundingGenerationPreview(
            profile_revision_id=revision_id,
            preset=preset,
            candidate_limit=limits[preset],
            preview_items=preview_items,
            teacher=teacher,
            verifier=verifier,
            request_estimate={
                "candidate_requests": min(limits[preset], len(records)),
                "hosted_provider": hosted,
                "requires_confirmation": hosted,
            },
            blockers=tuple(blockers),
        )

    def environment_permissions(
        self, environment_revision_id: str
    ) -> EnvironmentPermissionSummary:
        revision = self.get_environment_revision(environment_revision_id)
        if revision is None:
            raise ValueError(f"unknown environment revision: {environment_revision_id}")
        definition = revision.definition
        return EnvironmentPermissionSummary(
            local_files=bool(definition.get("local_files", True)),
            local_sqlite=bool(definition.get("local_sqlite", True)),
            loopback_services=bool(definition.get("loopback_services", True)),
            external_writes=False,
            max_steps=int(definition.get("max_steps") or 16),
            timeout_seconds=int(definition.get("timeout_seconds") or 60),
            notes=(
                "Every episode runs in an attempt-local workspace.",
                "Live business-system writes are unavailable.",
            ),
        )

    # ----- V11 outcomes -------------------------------------------------

    def _ensure_outcome_profiles(self) -> None:
        now = _now()
        for scenario, (adapter, metric, direction) in _OUTCOME_STARTERS.items():
            scenario_revision_id = f"{scenario}@1"
            definition = {
                "scenario_revision_id": scenario_revision_id,
                "version": "1",
                "practical_margin": 0.01,
                "evaluation_starters": [
                    {
                        "id": f"{scenario}-starter",
                        "adapter_id": adapter,
                        "primary_metric": metric,
                        "direction": direction,
                        "required_fields": [],
                        "minimum_records": 20,
                    }
                ],
                "diagnostic_fields": [
                    "train_loss",
                    "eval_loss",
                    "grad_norm",
                    "throughput",
                    "peak_memory_bytes",
                    "truncation_rate",
                    "padding_rate",
                ],
            }
            content_hash = _content_hash(definition)
            profile_id = _stable_id("outcome-profile", definition)
            self.conn.execute(
                """INSERT OR IGNORE INTO scenario_outcome_profiles
                   (id,scenario_revision_id,version,content_hash,definition_json,created_at)
                   VALUES (?,?,?,?,?,?)""",
                (
                    profile_id,
                    scenario_revision_id,
                    "1",
                    content_hash,
                    _canonical_json(definition),
                    now,
                ),
            )
        self._commit()

    def list_outcome_profiles(
        self, *, scenario_revision_id: Optional[str] = None
    ) -> List[ScenarioOutcomeProfile]:
        if scenario_revision_id:
            rows = self.conn.execute(
                """SELECT * FROM scenario_outcome_profiles
                   WHERE scenario_revision_id=? ORDER BY version""",
                (scenario_revision_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM scenario_outcome_profiles ORDER BY scenario_revision_id,version"
            ).fetchall()
        return [self._outcome_profile(row) for row in rows]

    def get_outcome_profile(self, profile_id: str) -> Optional[ScenarioOutcomeProfile]:
        row = self.conn.execute(
            "SELECT * FROM scenario_outcome_profiles WHERE id=?", (profile_id,)
        ).fetchone()
        return self._outcome_profile(row) if row else None

    @staticmethod
    def _outcome_profile(row: Any) -> ScenarioOutcomeProfile:
        definition = _loads(row["definition_json"], {})
        starters = tuple(
            ScenarioEvaluationStarter(
                id=str(value["id"]),
                scenario_revision_id=str(row["scenario_revision_id"]),
                adapter_id=str(value["adapter_id"]),
                primary_metric=str(value["primary_metric"]),
                direction=str(value["direction"]),
                required_fields=tuple(value.get("required_fields") or ()),
                minimum_records=int(value.get("minimum_records") or 20),
                notes=str(value.get("notes") or ""),
            )
            for value in definition.get("evaluation_starters") or ()
        )
        return ScenarioOutcomeProfile(
            id=str(row["id"]),
            scenario_revision_id=str(row["scenario_revision_id"]),
            version=str(row["version"]),
            content_hash=str(row["content_hash"]),
            practical_margin=float(definition.get("practical_margin") or 0.0),
            evaluation_starters=starters,
            diagnostic_fields=tuple(definition.get("diagnostic_fields") or ()),
        )

    def _evaluation_primary(self, evaluation_id: str) -> Dict[str, Any]:
        evaluation = self.db.get_evaluation(evaluation_id)
        if evaluation is None:
            raise ValueError(f"unknown evaluation: {evaluation_id}")
        if evaluation.status != "completed":
            raise ValueError(f"evaluation {evaluation_id} is not completed")
        revision = self.db.get_benchmark_suite_revision(evaluation.suite_revision_id)
        if revision is None:
            raise ValueError("evaluation suite revision is missing")
        metrics = self.db.list_evaluation_metrics(evaluation_id)
        metric = next(
            (
                value
                for value in metrics
                if value.name == revision.primary_metric and not value.suite_item_id
            ),
            metrics[0] if metrics else None,
        )
        if metric is None:
            raise ValueError(f"evaluation {evaluation_id} has no primary metric")
        return {
            "evaluation": evaluation,
            "revision": revision,
            "metric": metric,
            "value": float(metric.value),
            "direction": str(metric.direction or revision.direction),
        }

    @staticmethod
    def _resource_projection(run: Any) -> Dict[str, Any]:
        raw = run.raw
        duration = raw.get("duration_seconds") or raw.get("training_duration_seconds")
        proof_cap = (
            raw.get("proof_max_samples")
            or (raw.get("launch_config") or {}).get("proof_max_samples")
        )
        total = (
            raw.get("dataset_row_count")
            or (raw.get("launch_config") or {}).get("full_dataset_row_count")
        )
        multiplier = None
        if isinstance(proof_cap, (int, float)) and isinstance(total, (int, float)):
            if float(proof_cap) > 0:
                multiplier = max(1.0, float(total) / float(proof_cap))
        projected = float(duration) * multiplier if duration and multiplier else None
        peak_memory = (
            raw.get("peak_memory_bytes")
            or (raw.get("telemetry") or {}).get("peak_memory_bytes")
        )
        output_bytes = raw.get("output_size_bytes")
        return {
            "elapsed_seconds_low": projected * 0.8 if projected else None,
            "elapsed_seconds_high": projected * 1.25 if projected else None,
            "peak_memory_bytes": int(peak_memory) if peak_memory else None,
            "output_bytes_low": (
                int(float(output_bytes) * multiplier * 0.8)
                if output_bytes and multiplier
                else None
            ),
            "output_bytes_high": (
                int(float(output_bytes) * multiplier * 1.25)
                if output_bytes and multiplier
                else None
            ),
            "training_tokens": raw.get("training_tokens"),
            "confidence": "medium" if projected else "unavailable",
            "basis": {
                "proof_run_id": run.run_id,
                "proof_cap": proof_cap,
                "full_dataset_rows": total,
                "linear_scaling_assumption": bool(projected),
            },
        }

    def assess_outcome(
        self,
        proof_run_id: str,
        payload: Mapping[str, Any],
    ) -> TrainingOutcomeAssessment:
        run = self.db.get_run(proof_run_id)
        if run is None:
            raise ValueError(f"unknown proof run: {proof_run_id}")
        raw = run.raw
        launch = dict(raw.get("launch_config") or raw.get("config") or {})
        proof_run = bool(raw.get("proof_run") or launch.get("proof_run"))
        if not proof_run:
            raise ValueError("training outcome assessments require a proof run")
        scenario_revision_id = str(
            payload.get("scenario_revision_id")
            or raw.get("scenario_revision_id")
            or launch.get("scenario_revision_id")
            or ""
        ).strip()
        if not scenario_revision_id:
            raise ValueError("proof run has no scenario revision identity")
        profile = next(
            iter(self.list_outcome_profiles(scenario_revision_id=scenario_revision_id)),
            None,
        )
        if profile is None:
            raise ValueError(
                f"no outcome profile is registered for {scenario_revision_id}"
            )

        technical_ok = bool(run.weights_updated) and run.status == "completed"
        findings: List[Dict[str, Any]] = []
        diagnostics = {
            key: raw.get(key)
            for key in profile.diagnostic_fields
            if raw.get(key) is not None
        }
        status = "technical_failure" if not technical_ok else "incomplete_evidence"
        quality_status = "unavailable"
        comparison_hash = None
        summary: Dict[str, Any] = {
            "technical_verified": technical_ok,
            "weights_updated": bool(run.weights_updated),
            "artifact_available": bool(run.final_model_path),
        }

        if not technical_ok:
            findings.append(
                {
                    "category": "training",
                    "severity": "error",
                    "summary": "The proof run did not produce a verified weight update.",
                    "evidence": {
                        "status": run.status,
                        "weights_updated": bool(run.weights_updated),
                        "reason": run.final_update_reason or run.failure_reason,
                    },
                    "why_it_matters": "Quality comparisons cannot validate a model that did not update.",
                    "safe_remedies": [
                        "Inspect the proof-run logs and effectiveness contract.",
                        "Repair the data or trainer configuration, then retry the proof run.",
                    ],
                    "available_actions": ["open_run", "retry", "repair_data"],
                }
            )
        else:
            base_id = str(payload.get("base_evaluation_id") or "").strip()
            candidate_id = str(payload.get("candidate_evaluation_id") or "").strip()
            if base_id and candidate_id:
                base = self._evaluation_primary(base_id)
                candidate = self._evaluation_primary(candidate_id)
                if (
                    base["evaluation"].suite_revision_id
                    != candidate["evaluation"].suite_revision_id
                ):
                    raise ValueError(
                        "base and proof evaluations must use the same suite revision"
                    )
                if base["metric"].name != candidate["metric"].name:
                    raise ValueError("base and proof primary metrics do not match")
                direction = candidate["direction"]
                delta = (
                    candidate["value"] - base["value"]
                    if direction == "maximize"
                    else base["value"] - candidate["value"]
                )
                margin = float(
                    payload.get("practical_margin")
                    if payload.get("practical_margin") is not None
                    else profile.practical_margin
                )
                if delta > margin:
                    status = quality_status = "improved"
                elif delta < -margin:
                    status = quality_status = "regressed"
                else:
                    status = quality_status = "no_clear_change"
                base_samples = {
                    str(value.record_id): value
                    for value in self.db.list_evaluation_samples(base_id)
                    if value.record_id and value.score is not None
                }
                candidate_samples = {
                    str(value.record_id): value
                    for value in self.db.list_evaluation_samples(candidate_id)
                    if value.record_id and value.score is not None
                }
                sample_counts = {
                    "improvements": 0,
                    "regressions": 0,
                    "unchanged_failures": 0,
                    "unchanged_passes": 0,
                }
                for record_id in sorted(set(base_samples) & set(candidate_samples)):
                    base_sample = base_samples[record_id]
                    candidate_sample = candidate_samples[record_id]
                    sample_delta = (
                        float(candidate_sample.score) - float(base_sample.score)
                        if direction == "maximize"
                        else float(base_sample.score) - float(candidate_sample.score)
                    )
                    if sample_delta > 0:
                        sample_counts["improvements"] += 1
                    elif sample_delta < 0:
                        sample_counts["regressions"] += 1
                    elif bool(candidate_sample.passed):
                        sample_counts["unchanged_passes"] += 1
                    else:
                        sample_counts["unchanged_failures"] += 1
                if (
                    status == "no_clear_change"
                    and sample_counts["improvements"] > 0
                    and sample_counts["regressions"] > 0
                ):
                    status = quality_status = "mixed"
                comparison = {
                    "suite_revision_id": base["evaluation"].suite_revision_id,
                    "metric": base["metric"].name,
                    "direction": direction,
                    "base": base["value"],
                    "candidate": candidate["value"],
                    "directional_delta": delta,
                    "practical_margin": margin,
                    "sample_counts": sample_counts,
                }
                comparison_hash = _content_hash(comparison)
                summary["comparison"] = comparison
                if status == "regressed":
                    findings.append(
                        {
                            "category": "quality",
                            "severity": "warning",
                            "summary": "The proof artifact regressed on the selected development evidence.",
                            "evidence": comparison,
                            "why_it_matters": "A larger run may amplify the same development regression.",
                            "safe_remedies": [
                                "Inspect sample-level regressions.",
                                "Repair the dataset or reduce the training budget.",
                            ],
                            "available_actions": [
                                "open_comparison",
                                "repair_data",
                                "retry",
                                "override",
                            ],
                        }
                    )
            else:
                findings.append(
                    {
                        "category": "evidence",
                        "severity": "warning",
                        "summary": "No compatible base/proof development comparison is attached.",
                        "evidence": {},
                        "why_it_matters": "Technical completion alone does not show whether behavior improved.",
                        "safe_remedies": [
                            "Launch the scenario evaluation starter for the base and proof artifact."
                        ],
                        "available_actions": ["evaluate", "override"],
                    }
                )

        resource_projection = self._resource_projection(run)
        created_at = _now()
        assessment_identity = {
            "proof_run_id": proof_run_id,
            "scenario_revision_id": scenario_revision_id,
            "profile_hash": profile.content_hash,
            "base_evaluation_id": payload.get("base_evaluation_id"),
            "candidate_evaluation_id": payload.get("candidate_evaluation_id"),
            "comparison_hash": comparison_hash,
            "technical_status": "verified" if technical_ok else "failed",
            "quality_status": quality_status,
            "resource_projection": resource_projection,
            "diagnostics": diagnostics,
        }
        content_hash = _content_hash(assessment_identity)
        assessment_id = str(payload.get("_assessment_id") or "").strip() or _stable_id(
            "outcome", assessment_identity
        )
        existing = self.get_outcome_assessment(assessment_id)
        if existing is not None and existing.status not in {
            "queued",
            "running",
            "retrying",
        }:
            return existing
        if existing is None:
            self.conn.execute(
                """INSERT INTO training_outcome_assessments
                   (id,proof_run_id,scenario_revision_id,profile_id,status,
                    technical_status,quality_status,base_evaluation_id,
                    candidate_evaluation_id,comparison_hash,resource_projection_json,
                    diagnostics_json,summary_json,content_hash,work_item_id,error,
                    created_at,completed_at,stage,progress_json,request_json,
                    resume_cursor_json,cancel_requested)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    assessment_id,
                    proof_run_id,
                    scenario_revision_id,
                    profile.id,
                    status,
                    "verified" if technical_ok else "failed",
                    quality_status,
                    payload.get("base_evaluation_id"),
                    payload.get("candidate_evaluation_id"),
                    comparison_hash,
                    _canonical_json(resource_projection),
                    _canonical_json(diagnostics),
                    _canonical_json(summary),
                    content_hash,
                    payload.get("_work_item_id"),
                    None,
                    created_at,
                    created_at,
                    "completed",
                    _canonical_json({"current": 1, "total": 1, "percent": 100}),
                    _canonical_json(dict(payload)),
                    "{}",
                    0,
                ),
            )
        else:
            self.conn.execute(
                """UPDATE training_outcome_assessments
                   SET status=?,stage='completed',technical_status=?,quality_status=?,
                       base_evaluation_id=?,candidate_evaluation_id=?,
                       comparison_hash=?,resource_projection_json=?,diagnostics_json=?,
                       summary_json=?,content_hash=?,progress_json=?,error=NULL,
                       completed_at=?,cancel_requested=0
                   WHERE id=?""",
                (
                    status,
                    "verified" if technical_ok else "failed",
                    quality_status,
                    payload.get("base_evaluation_id"),
                    payload.get("candidate_evaluation_id"),
                    comparison_hash,
                    _canonical_json(resource_projection),
                    _canonical_json(diagnostics),
                    _canonical_json(summary),
                    content_hash,
                    _canonical_json({"current": 1, "total": 1, "percent": 100}),
                    created_at,
                    assessment_id,
                ),
            )
            self.conn.execute(
                "DELETE FROM training_outcome_findings WHERE assessment_id=?",
                (assessment_id,),
            )
        for ordinal, finding in enumerate(findings):
            finding_hash = _content_hash(finding)
            self.conn.execute(
                """INSERT INTO training_outcome_findings
                   (id,assessment_id,ordinal,category,severity,summary,evidence_json,
                    why_it_matters,safe_remedies_json,available_actions_json,
                    content_hash,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    _stable_id(
                        "outcome-finding",
                        {"assessment_id": assessment_id, "ordinal": ordinal, **finding},
                    ),
                    assessment_id,
                    ordinal,
                    finding["category"],
                    finding["severity"],
                    finding["summary"],
                    _canonical_json(finding["evidence"]),
                    finding["why_it_matters"],
                    _canonical_json(finding["safe_remedies"]),
                    _canonical_json(finding["available_actions"]),
                    finding_hash,
                    created_at,
                ),
            )
        bundle_files = {
            "assessment.json": json.dumps(
                {**assessment_identity, "status": status, "summary": summary},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            "findings.json": json.dumps(findings, indent=2, sort_keys=True) + "\n",
        }
        _atomic_bundle(self.outcome_root / assessment_id, bundle_files)
        self._commit()
        result = self.get_outcome_assessment(assessment_id)
        assert result is not None
        return result

    def prepare_outcome_assessment(
        self,
        proof_run_id: str,
        payload: Mapping[str, Any],
        *,
        work_item_id: Optional[str] = None,
    ) -> TrainingOutcomeAssessment:
        """Create the visible, durable placeholder before evidence work starts."""

        run = self.db.get_run(proof_run_id)
        if run is None:
            raise ValueError(f"unknown proof run: {proof_run_id}")
        raw = run.raw
        launch = dict(raw.get("launch_config") or raw.get("config") or {})
        if not bool(raw.get("proof_run") or launch.get("proof_run")):
            raise ValueError("training outcome preparation requires a proof run")
        scenario_revision_id = str(
            payload.get("scenario_revision_id")
            or raw.get("scenario_revision_id")
            or launch.get("scenario_revision_id")
            or ""
        ).strip()
        profile = next(
            iter(self.list_outcome_profiles(scenario_revision_id=scenario_revision_id)),
            None,
        )
        if profile is None:
            raise ValueError(f"no outcome profile is registered for {scenario_revision_id}")
        identity = {
            "proof_run_id": proof_run_id,
            "scenario_revision_id": scenario_revision_id,
            "suite_revision_id": payload.get("suite_revision_id"),
            "base_evaluation_id": payload.get("base_evaluation_id"),
            "candidate_evaluation_id": payload.get("candidate_evaluation_id"),
        }
        assessment_id = _stable_id("outcome-preparation", identity)
        existing = self.get_outcome_assessment(assessment_id)
        if existing is not None:
            if work_item_id and not existing.work_item_id:
                self.conn.execute(
                    "UPDATE training_outcome_assessments SET work_item_id=? WHERE id=?",
                    (work_item_id, assessment_id),
                )
                self._commit()
                existing = self.get_outcome_assessment(assessment_id)
            assert existing is not None
            return existing
        now = _now()
        self.conn.execute(
            """INSERT INTO training_outcome_assessments
               (id,proof_run_id,scenario_revision_id,profile_id,status,
                technical_status,quality_status,base_evaluation_id,
                candidate_evaluation_id,comparison_hash,resource_projection_json,
                diagnostics_json,summary_json,content_hash,work_item_id,error,
                created_at,completed_at,stage,progress_json,request_json,
                resume_cursor_json,cancel_requested)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                assessment_id,
                proof_run_id,
                scenario_revision_id,
                profile.id,
                "queued",
                "pending",
                "pending",
                payload.get("base_evaluation_id"),
                payload.get("candidate_evaluation_id"),
                None,
                "{}",
                "{}",
                _canonical_json(
                    {
                        "message": "Halo Forge is preparing a fair base and proof comparison.",
                        "suite_revision_id": payload.get("suite_revision_id"),
                    }
                ),
                None,
                work_item_id,
                None,
                now,
                None,
                "waiting_for_evaluations",
                _canonical_json({"current": 0, "total": 2, "percent": 0}),
                _canonical_json(dict(payload)),
                "{}",
                0,
            ),
        )
        self._commit()
        result = self.get_outcome_assessment(assessment_id)
        assert result is not None
        return result

    def _outcome_assessment(self, row: Any) -> TrainingOutcomeAssessment:
        return TrainingOutcomeAssessment(
            id=str(row["id"]),
            proof_run_id=str(row["proof_run_id"]),
            scenario_revision_id=str(row["scenario_revision_id"]),
            profile_id=str(row["profile_id"]),
            status=str(row["status"]),
            technical_status=str(row["technical_status"]),
            quality_status=str(row["quality_status"]),
            base_evaluation_id=row["base_evaluation_id"],
            candidate_evaluation_id=row["candidate_evaluation_id"],
            comparison_hash=row["comparison_hash"],
            resource_projection=_loads(row["resource_projection_json"], {}),
            diagnostics=_loads(row["diagnostics_json"], {}),
            summary=_loads(row["summary_json"], {}),
            content_hash=row["content_hash"],
            work_item_id=row["work_item_id"],
            error=row["error"],
            created_at=str(row["created_at"]),
            completed_at=row["completed_at"],
            stage=str(row["stage"]),
            progress=_loads(row["progress_json"], {}),
            request=_loads(row["request_json"], {}),
            cancel_requested=bool(row["cancel_requested"]),
        )

    def get_outcome_assessment(
        self, assessment_id: str
    ) -> Optional[TrainingOutcomeAssessment]:
        row = self.conn.execute(
            "SELECT * FROM training_outcome_assessments WHERE id=?",
            (assessment_id,),
        ).fetchone()
        return self._outcome_assessment(row) if row else None

    def list_outcome_assessments(
        self,
        *,
        proof_run_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        if proof_run_id:
            rows = self.conn.execute(
                """SELECT * FROM training_outcome_assessments
                   WHERE proof_run_id=? ORDER BY created_at DESC""",
                (proof_run_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM training_outcome_assessments ORDER BY created_at DESC"
            ).fetchall()
        return _page(
            [self._outcome_assessment(row).to_dict() for row in rows],
            limit=limit,
            offset=offset,
        )

    def list_outcome_findings(
        self, assessment_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        rows = self.conn.execute(
            """SELECT * FROM training_outcome_findings
               WHERE assessment_id=? ORDER BY ordinal""",
            (assessment_id,),
        ).fetchall()
        findings = [
            TrainingOutcomeFinding(
                id=str(row["id"]),
                assessment_id=str(row["assessment_id"]),
                ordinal=int(row["ordinal"]),
                category=str(row["category"]),
                severity=str(row["severity"]),
                summary=str(row["summary"]),
                evidence=_loads(row["evidence_json"], {}),
                why_it_matters=str(row["why_it_matters"]),
                safe_remedies=list(_loads(row["safe_remedies_json"], [])),
                available_actions=list(_loads(row["available_actions_json"], [])),
                content_hash=str(row["content_hash"]),
                created_at=str(row["created_at"]),
            ).to_dict()
            for row in rows
        ]
        return _page(findings, limit=limit, offset=offset)

    def create_outcome_decision(
        self,
        proof_run_id: str,
        payload: Mapping[str, Any],
    ) -> TrainingOutcomeDecision:
        decision = str(payload.get("decision") or "").strip().lower()
        if decision not in {
            "evaluate",
            "repair",
            "retry",
            "fork",
            "start_full_run",
            "override",
        }:
            raise ValueError("unsupported training outcome decision")
        reason = str(payload.get("reason") or "").strip()
        if decision in {"override", "fork"} and not reason:
            raise ValueError(f"{decision} requires a reason")
        assessment_id = str(payload.get("assessment_id") or "").strip() or None
        assessment = (
            self.get_outcome_assessment(assessment_id) if assessment_id else None
        )
        if decision == "start_full_run":
            if assessment is None:
                raise ValueError(
                    "starting a full run requires an assessment or an override decision"
                )
            if assessment.technical_status != "verified":
                raise ValueError("a technically failed proof cannot start a full run")
            if assessment.status in {"incomplete_evidence", "technical_failure"}:
                raise ValueError(
                    "starting a full run requires compatible development evidence; "
                    "record an explicit override instead"
                )
        identity = {
            "proof_run_id": proof_run_id,
            "assessment_id": assessment_id,
            "decision": decision,
            "reason": reason,
            "full_run_id": payload.get("full_run_id"),
            "context": dict(payload.get("context") or {}),
            "nonce": uuid.uuid4().hex,
        }
        decision_id = _stable_id("outcome-decision", identity)
        created_at = _now()
        self.conn.execute(
            """INSERT INTO training_outcome_decisions
               (id,assessment_id,proof_run_id,decision,reason,full_run_id,
                context_json,created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                decision_id,
                assessment_id,
                proof_run_id,
                decision,
                reason,
                payload.get("full_run_id"),
                _canonical_json(identity["context"]),
                created_at,
            ),
        )
        self._commit()
        return TrainingOutcomeDecision(
            id=decision_id,
            assessment_id=assessment_id,
            proof_run_id=proof_run_id,
            decision=decision,
            reason=reason,
            full_run_id=payload.get("full_run_id"),
            context=identity["context"],
            created_at=created_at,
        )

    def full_run_context(
        self,
        proof_run_id: str,
        *,
        assessment_id: Optional[str] = None,
        override_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        run = self.db.get_run(proof_run_id)
        if run is None:
            raise ValueError(f"unknown proof run: {proof_run_id}")
        assessment = (
            self.get_outcome_assessment(assessment_id) if assessment_id else None
        )
        if assessment is None and not str(override_reason or "").strip():
            raise ValueError(
                "full-run context requires a completed outcome assessment or override reason"
            )
        if assessment and assessment.technical_status != "verified":
            raise ValueError("proof run did not produce a verified update")
        if assessment and assessment.status in {
            "incomplete_evidence",
            "technical_failure",
        }:
            raise ValueError(
                "the assessment is incomplete; supply an override reason or attach "
                "compatible base/proof development evaluations"
            )
        launch = dict(run.raw.get("launch_config") or run.raw.get("config") or {})
        for key in ("proof_run", "proof_max_samples", "max_samples"):
            launch.pop(key, None)
        launch.update(
            proof_run=False,
            parent_run_id=proof_run_id,
            proof_parent_run_id=proof_run_id,
            full_run_from_proof=True,
            outcome_assessment_id=assessment.id if assessment else None,
            outcome_override_reason=str(override_reason or "").strip() or None,
        )
        return {
            "proof_run_id": proof_run_id,
            "assessment": assessment.to_dict() if assessment else None,
            "override_required": assessment is None,
            "resolved_config": launch,
        }

    # ----- V12 studies --------------------------------------------------

    def create_study(
        self, payload: Mapping[str, Any]
    ) -> AdaptationStudy:
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("study name is required")
        study_id = str(payload.get("id") or uuid.uuid4().hex)
        now = _now()
        self.conn.execute(
            """INSERT INTO adaptation_studies
               (id,name,description,status,latest_protocol_revision_id,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                study_id,
                name,
                payload.get("description"),
                "draft",
                None,
                now,
                now,
            ),
        )
        self._commit()
        return self.get_study(study_id)  # type: ignore[return-value]

    def _study(self, row: Any) -> AdaptationStudy:
        return AdaptationStudy(
            id=str(row["id"]),
            name=str(row["name"]),
            description=row["description"],
            status=str(row["status"]),
            latest_protocol_revision_id=row["latest_protocol_revision_id"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def get_study(self, study_id: str) -> Optional[AdaptationStudy]:
        row = self.conn.execute(
            "SELECT * FROM adaptation_studies WHERE id=?", (study_id,)
        ).fetchone()
        return self._study(row) if row else None

    def list_studies(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT * FROM adaptation_studies ORDER BY updated_at DESC"
        ).fetchall()
        return _page(
            [self._study(row).to_dict() for row in rows],
            limit=limit,
            offset=offset,
        )

    @staticmethod
    def validate_study_protocol(payload: Mapping[str, Any]) -> Dict[str, Any]:
        design = str(payload.get("design_kind") or "paired_ab").strip()
        if design not in {"paired_ab", "dose_response", "factorial_2x2"}:
            raise ValueError("unsupported adaptation study design")
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("study question is required")
        arms = list(payload.get("arms") or [])
        if len(arms) < 2 or len(arms) > 4:
            raise ValueError("adaptation studies require two to four arms")
        if design == "paired_ab" and len(arms) != 2:
            raise ValueError("paired_ab requires exactly two arms")
        if design == "factorial_2x2" and len(arms) != 4:
            raise ValueError("factorial_2x2 requires exactly four arms")
        if design == "dose_response" and len(arms) > 5:
            raise ValueError("dose_response supports at most five doses")
        controls = sum(bool(value.get("is_control")) for value in arms)
        if controls != 1:
            raise ValueError("exactly one study arm must be the control")
        seeds = [int(value) for value in payload.get("seeds") or [17, 42, 101]]
        if len(set(seeds)) < 2:
            raise ValueError("adaptation studies require at least two distinct seeds")
        contrasts = list(payload.get("contrasts") or [])
        if not contrasts:
            raise ValueError("at least one planned contrast is required")
        protected = {"holdout", "operational", "test", "canary"}
        for key in ("development_suite_purpose", "retention_suite_purpose"):
            purpose = str(payload.get(key) or "development").lower()
            if purpose in protected:
                raise EvidenceEligibilityError(
                    f"{key} cannot use {purpose} evidence to guide a study"
                )
        return {
            "valid": True,
            "design_kind": design,
            "arm_count": len(arms),
            "seeds": seeds,
            "planned_contrast_count": len(contrasts),
            "assignment_count": len(arms) * len(seeds),
        }

    def create_study_protocol(
        self, study_id: str, payload: Mapping[str, Any]
    ) -> AdaptationStudyProtocolRevision:
        study = self.get_study(study_id)
        if study is None:
            raise ValueError(f"unknown adaptation study: {study_id}")
        validation = self.validate_study_protocol(payload)
        previous = self.conn.execute(
            """SELECT COALESCE(MAX(revision_number),0) AS value
               FROM adaptation_study_protocol_revisions WHERE study_id=?""",
            (study_id,),
        ).fetchone()
        revision_number = int(previous["value"]) + 1
        definition = dict(payload)
        definition["seeds"] = validation["seeds"]
        content_hash = _content_hash(definition)
        revision_id = _stable_id(
            "study-protocol", {"study_id": study_id, "content_hash": content_hash}
        )
        now = _now()
        self.conn.execute(
            """INSERT OR IGNORE INTO adaptation_study_protocol_revisions
               (id,study_id,revision_number,design_kind,question,definition_json,
                content_hash,created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                revision_id,
                study_id,
                revision_number,
                validation["design_kind"],
                str(payload["question"]).strip(),
                _canonical_json(definition),
                content_hash,
                now,
            ),
        )
        arm_ids: Dict[str, str] = {}
        for ordinal, value in enumerate(payload.get("arms") or []):
            arm_name = str(value.get("name") or f"Arm {ordinal + 1}").strip()
            arm_hash = _content_hash(value)
            arm_id = _stable_id(
                "study-arm",
                {
                    "protocol_revision_id": revision_id,
                    "ordinal": ordinal,
                    "content_hash": arm_hash,
                },
            )
            arm_ids[arm_name] = arm_id
            self.conn.execute(
                """INSERT OR IGNORE INTO adaptation_study_arms
                   (id,protocol_revision_id,ordinal,name,is_control,
                    factor_values_json,launch_config_json,content_hash)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    arm_id,
                    revision_id,
                    ordinal,
                    arm_name,
                    int(bool(value.get("is_control"))),
                    _canonical_json(value.get("factor_values") or {}),
                    _canonical_json(value.get("launch_config") or {}),
                    arm_hash,
                ),
            )
        assignment_ordinal = 0
        for seed in validation["seeds"]:
            for arm in sorted(
                self.list_study_arms(revision_id), key=lambda value: value.ordinal
            ):
                assignment_id = _stable_id(
                    "study-assignment",
                    {
                        "protocol_revision_id": revision_id,
                        "arm_id": arm.id,
                        "seed": seed,
                    },
                )
                self.conn.execute(
                    """INSERT OR IGNORE INTO adaptation_study_assignments
                       (id,protocol_revision_id,arm_id,seed,ordinal,run_group_id,
                        run_id,status,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        assignment_id,
                        revision_id,
                        arm.id,
                        seed,
                        assignment_ordinal,
                        None,
                        None,
                        "planned",
                        now,
                    ),
                )
                assignment_ordinal += 1
        for ordinal, value in enumerate(payload.get("contrasts") or []):
            left = str(value.get("left_arm") or "")
            right = str(value.get("right_arm") or "")
            if left not in arm_ids or right not in arm_ids:
                raise ValueError("planned contrast references an unknown arm")
            conclusion = str(value.get("conclusion_kind") or "superiority")
            if conclusion not in {
                "superiority",
                "equivalence",
                "non_inferiority",
            }:
                raise ValueError("unsupported contrast conclusion kind")
            self.conn.execute(
                """INSERT OR IGNORE INTO adaptation_study_contrasts
                   (id,protocol_revision_id,ordinal,name,left_arm_id,right_arm_id,
                    metric,direction,conclusion_kind,practical_margin,exploratory)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    _stable_id(
                        "study-contrast",
                        {
                            "protocol_revision_id": revision_id,
                            "ordinal": ordinal,
                            "value": value,
                        },
                    ),
                    revision_id,
                    ordinal,
                    str(value.get("name") or f"Contrast {ordinal + 1}"),
                    arm_ids[left],
                    arm_ids[right],
                    str(value.get("metric") or "score"),
                    str(value.get("direction") or "maximize"),
                    conclusion,
                    float(value.get("practical_margin") or 0.0),
                    int(bool(value.get("exploratory", False))),
                ),
            )
        self.conn.execute(
            """UPDATE adaptation_studies
               SET latest_protocol_revision_id=?,updated_at=? WHERE id=?""",
            (revision_id, now, study_id),
        )
        self._commit()
        result = self.get_study_protocol(revision_id)
        assert result is not None
        return result

    def list_study_arms(self, revision_id: str) -> List[StudyArm]:
        rows = self.conn.execute(
            """SELECT * FROM adaptation_study_arms
               WHERE protocol_revision_id=? ORDER BY ordinal""",
            (revision_id,),
        ).fetchall()
        return [
            StudyArm(
                id=str(row["id"]),
                protocol_revision_id=str(row["protocol_revision_id"]),
                ordinal=int(row["ordinal"]),
                name=str(row["name"]),
                is_control=bool(row["is_control"]),
                factor_values=_loads(row["factor_values_json"], {}),
                launch_config=_loads(row["launch_config_json"], {}),
                content_hash=str(row["content_hash"]),
            )
            for row in rows
        ]

    def list_study_assignments(self, revision_id: str) -> List[StudyAssignment]:
        rows = self.conn.execute(
            """SELECT * FROM adaptation_study_assignments
               WHERE protocol_revision_id=? ORDER BY ordinal""",
            (revision_id,),
        ).fetchall()
        return [
            StudyAssignment(
                id=str(row["id"]),
                protocol_revision_id=str(row["protocol_revision_id"]),
                arm_id=str(row["arm_id"]),
                seed=int(row["seed"]),
                ordinal=int(row["ordinal"]),
                run_group_id=row["run_group_id"],
                run_id=row["run_id"],
                status=str(row["status"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def list_study_contrasts(self, revision_id: str) -> List[PlannedContrast]:
        rows = self.conn.execute(
            """SELECT * FROM adaptation_study_contrasts
               WHERE protocol_revision_id=? ORDER BY ordinal""",
            (revision_id,),
        ).fetchall()
        return [
            PlannedContrast(
                id=str(row["id"]),
                name=str(row["name"]),
                left_arm_id=str(row["left_arm_id"]),
                right_arm_id=str(row["right_arm_id"]),
                metric=str(row["metric"]),
                direction=str(row["direction"]),
                conclusion_kind=str(row["conclusion_kind"]),
                practical_margin=float(row["practical_margin"]),
                exploratory=bool(row["exploratory"]),
            )
            for row in rows
        ]

    def get_study_protocol(
        self, revision_id: str
    ) -> Optional[AdaptationStudyProtocolRevision]:
        row = self.conn.execute(
            "SELECT * FROM adaptation_study_protocol_revisions WHERE id=?",
            (revision_id,),
        ).fetchone()
        if row is None:
            return None
        return AdaptationStudyProtocolRevision(
            id=str(row["id"]),
            study_id=str(row["study_id"]),
            revision_number=int(row["revision_number"]),
            design_kind=str(row["design_kind"]),
            question=str(row["question"]),
            definition=_loads(row["definition_json"], {}),
            content_hash=str(row["content_hash"]),
            created_at=str(row["created_at"]),
            arms=tuple(self.list_study_arms(revision_id)),
            assignments=tuple(self.list_study_assignments(revision_id)),
            contrasts=tuple(self.list_study_contrasts(revision_id)),
            launch_status=str(row["launch_status"]),
            launch_progress=_loads(row["launch_progress_json"], {}),
            launch_work_item_id=row["launch_work_item_id"],
            launch_error=row["launch_error"],
        )

    def materialize_study(
        self, revision_id: str
    ) -> Dict[str, Any]:
        revision = self.get_study_protocol(revision_id)
        if revision is None:
            raise ValueError(f"unknown study protocol revision: {revision_id}")
        items = []
        for assignment in revision.assignments:
            arm = next(value for value in revision.arms if value.id == assignment.arm_id)
            config = {
                **dict(arm.launch_config),
                "seed": assignment.seed,
                "study_protocol_revision_id": revision.id,
                "study_arm_id": arm.id,
                "study_assignment_id": assignment.id,
            }
            items.append(
                {
                    **assignment.to_dict(),
                    "arm": arm.to_dict(),
                    "resolved_launch_config": config,
                }
            )
        return {
            "protocol_revision": revision.to_dict(),
            "assignments": items,
            "resource_estimate": {
                "training_runs": len(items),
                "evaluations_per_run": 2,
                "heavy_operations": len(items) * 3,
            },
        }

    def attach_study_run(
        self, assignment_id: str, *, run_id: str, run_group_id: Optional[str] = None
    ) -> StudyAssignment:
        if self.db.get_run(run_id) is None:
            raise ValueError(f"unknown run: {run_id}")
        self.conn.execute(
            """UPDATE adaptation_study_assignments
               SET run_id=?,run_group_id=?,status='completed' WHERE id=?""",
            (run_id, run_group_id, assignment_id),
        )
        self._commit()
        row = self.conn.execute(
            "SELECT protocol_revision_id FROM adaptation_study_assignments WHERE id=?",
            (assignment_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown study assignment: {assignment_id}")
        return next(
            value
            for value in self.list_study_assignments(str(row["protocol_revision_id"]))
            if value.id == assignment_id
        )

    @staticmethod
    def _bootstrap_mean(
        values: Sequence[float], *, seed: int = 42, resamples: int = 10_000
    ) -> tuple[float, float, float, float]:
        if not values:
            raise ValueError("bootstrap requires at least one paired value")
        mean = sum(values) / len(values)
        if len(values) == 1:
            return mean, mean, mean, 1.0
        rng = random.Random(seed)
        distribution = []
        non_positive = 0
        for _ in range(resamples):
            sample = [values[rng.randrange(len(values))] for _ in values]
            estimate = sum(sample) / len(sample)
            distribution.append(estimate)
            non_positive += int(estimate <= 0)
        distribution.sort()
        low = distribution[int(0.025 * (resamples - 1))]
        high = distribution[int(0.975 * (resamples - 1))]
        p_value = min(1.0, 2.0 * min(non_positive, resamples - non_positive) / resamples)
        return mean, low, high, p_value

    @staticmethod
    def _holm(pairs: Sequence[tuple[str, float]]) -> Dict[str, float]:
        ordered = sorted(pairs, key=lambda item: item[1])
        adjusted: Dict[str, float] = {}
        running = 0.0
        count = len(ordered)
        for index, (key, value) in enumerate(ordered):
            corrected = min(1.0, (count - index) * value)
            running = max(running, corrected)
            adjusted[key] = running
        return adjusted

    def analyze_study(
        self, revision_id: str, payload: Mapping[str, Any] | None = None
    ) -> StudyAnalysis:
        revision = self.get_study_protocol(revision_id)
        if revision is None:
            raise ValueError(f"unknown study protocol revision: {revision_id}")
        supplied = dict((payload or {}).get("metrics") or {})
        values: Dict[tuple[str, int, str], float] = {}
        for assignment in revision.assignments:
            arm = next(value for value in revision.arms if value.id == assignment.arm_id)
            assignment_metrics = supplied.get(assignment.id)
            if assignment_metrics is None and assignment.run_id:
                run = self.db.get_run(assignment.run_id)
                assignment_metrics = run.raw.get("metrics") if run else None
            for metric, value in dict(assignment_metrics or {}).items():
                values[(arm.id, assignment.seed, str(metric))] = _finite(
                    value, f"{assignment.id}.{metric}"
                )
        analyses = []
        p_values: List[tuple[str, float]] = []
        complete = True
        for contrast in revision.contrasts:
            paired = []
            seeds = sorted({value.seed for value in revision.assignments})
            for seed in seeds:
                left = values.get((contrast.left_arm_id, seed, contrast.metric))
                right = values.get((contrast.right_arm_id, seed, contrast.metric))
                if left is None or right is None:
                    continue
                delta = right - left if contrast.direction == "maximize" else left - right
                paired.append(delta)
            if len(paired) != len(seeds):
                complete = False
            if not paired:
                analyses.append(
                    {
                        "contrast_id": contrast.id,
                        "name": contrast.name,
                        "classification": "incomplete_evidence",
                        "paired_seed_count": 0,
                    }
                )
                continue
            mean, low, high, p_value = self._bootstrap_mean(paired)
            margin = contrast.practical_margin
            if contrast.conclusion_kind == "equivalence":
                conclusion = (
                    "equivalent" if low >= -margin and high <= margin else "not_established"
                )
            elif contrast.conclusion_kind == "non_inferiority":
                conclusion = "non_inferior" if low >= -margin else "not_established"
            else:
                conclusion = (
                    "superior"
                    if low > margin
                    else ("inferior" if high < -margin else "no_clear_change")
                )
            analyses.append(
                {
                    "contrast_id": contrast.id,
                    "name": contrast.name,
                    "metric": contrast.metric,
                    "direction": contrast.direction,
                    "conclusion_kind": contrast.conclusion_kind,
                    "classification": conclusion,
                    "mean_delta": mean,
                    "ci95": [low, high],
                    "p_value": p_value,
                    "practical_margin": margin,
                    "paired_seed_count": len(paired),
                    "exploratory": contrast.exploratory,
                }
            )
            if not contrast.exploratory:
                p_values.append((contrast.id, p_value))
        adjusted = self._holm(p_values)
        for value in analyses:
            if value.get("contrast_id") in adjusted:
                value["holm_adjusted_p"] = adjusted[value["contrast_id"]]

        randomized = bool(revision.definition.get("randomized", True))
        deviations = self.conn.execute(
            """SELECT COUNT(*) AS value FROM adaptation_study_deviations
               WHERE protocol_revision_id=?""",
            (revision_id,),
        ).fetchone()
        evidence_classification = (
            "causal"
            if complete and randomized and int(deviations["value"]) == 0
            else ("comparative" if analyses else "incomplete")
        )
        analysis_payload = {
            "protocol_revision_id": revision_id,
            "complete": complete,
            "seed_is_replicate": True,
            "bootstrap": {"resamples": 10_000, "seed": 42},
            "multiplicity": "holm",
            "contrasts": analyses,
            "domain_and_retention_kept_separate": True,
        }
        content_hash = _content_hash(analysis_payload)
        analysis_id = str((payload or {}).get("_analysis_id") or "").strip() or _stable_id(
            "study-analysis", analysis_payload
        )
        existing = self.get_study_analysis(analysis_id)
        if existing is not None and existing.status not in {
            "queued",
            "running",
            "retrying",
        }:
            return existing
        bundle = self.study_root / revision.study_id / analysis_id
        csv_stream = io.StringIO(newline="")
        writer = csv.DictWriter(
            csv_stream,
            fieldnames=[
                "name",
                "metric",
                "classification",
                "mean_delta",
                "paired_seed_count",
                "holm_adjusted_p",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(analyses)
        markdown = [
            f"# Adaptation Study: {revision.question}",
            "",
            f"Evidence classification: **{evidence_classification}**",
            "",
        ]
        markdown.extend(
            f"- {value['name']}: {value['classification']}"
            for value in analyses
        )
        bundle_path, _ = _atomic_bundle(
            bundle,
            {
                "analysis.json": json.dumps(
                    analysis_payload, indent=2, sort_keys=True
                )
                + "\n",
                "analysis.csv": csv_stream.getvalue(),
                "report.md": "\n".join(markdown) + "\n",
                "report.html": (
                    "<!doctype html><meta charset='utf-8'><title>Adaptation study</title>"
                    f"<pre>{json.dumps(analysis_payload, indent=2)}</pre>"
                ),
                "intervals.svg": self._study_svg(analyses),
            },
        )
        now = _now()
        if existing is None:
            self.conn.execute(
                """INSERT INTO adaptation_study_analyses
                   (id,protocol_revision_id,status,analysis_json,content_hash,
                    evidence_classification,work_item_id,bundle_path,created_at,
                    completed_at,stage,progress_json,request_json,
                    resume_cursor_json,cancel_requested,error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    analysis_id,
                    revision_id,
                    "completed",
                    _canonical_json(analysis_payload),
                    content_hash,
                    evidence_classification,
                    (payload or {}).get("_work_item_id"),
                    str(bundle_path),
                    now,
                    now,
                    "completed",
                    _canonical_json({"current": 1, "total": 1, "percent": 100}),
                    _canonical_json(dict(payload or {})),
                    "{}",
                    0,
                    None,
                ),
            )
        else:
            self.conn.execute(
                """UPDATE adaptation_study_analyses
                   SET status='completed',stage='completed',analysis_json=?,
                       content_hash=?,evidence_classification=?,bundle_path=?,
                       progress_json=?,cancel_requested=0,error=NULL,completed_at=?
                   WHERE id=?""",
                (
                    _canonical_json(analysis_payload),
                    content_hash,
                    evidence_classification,
                    str(bundle_path),
                    _canonical_json({"current": 1, "total": 1, "percent": 100}),
                    now,
                    analysis_id,
                ),
            )
        self._commit()
        result = self.get_study_analysis(analysis_id)
        assert result is not None
        return result

    def prepare_study_analysis(
        self,
        revision_id: str,
        payload: Mapping[str, Any],
        *,
        analysis_id: str,
        work_item_id: Optional[str] = None,
    ) -> StudyAnalysis:
        if self.get_study_protocol(revision_id) is None:
            raise ValueError(f"unknown study protocol revision: {revision_id}")
        existing = self.get_study_analysis(analysis_id)
        if existing is None:
            now = _now()
            queued_hash = _content_hash(
                {
                    "analysis_id": analysis_id,
                    "protocol_revision_id": revision_id,
                    "request": dict(payload),
                    "state": "queued",
                }
            )
            self.conn.execute(
                """INSERT INTO adaptation_study_analyses
                   (id,protocol_revision_id,status,stage,progress_json,
                    request_json,resume_cursor_json,cancel_requested,
                    analysis_json,content_hash,evidence_classification,
                    work_item_id,bundle_path,error,created_at,completed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    analysis_id,
                    revision_id,
                    "queued",
                    "waiting",
                    _canonical_json({"current": 0, "total": 1, "percent": 0}),
                    _canonical_json(dict(payload)),
                    "{}",
                    0,
                    "{}",
                    queued_hash,
                    "incomplete",
                    work_item_id,
                    None,
                    None,
                    now,
                    None,
                ),
            )
            self._commit()
            existing = self.get_study_analysis(analysis_id)
        elif work_item_id and not existing.work_item_id:
            self.conn.execute(
                "UPDATE adaptation_study_analyses SET work_item_id=? WHERE id=?",
                (work_item_id, analysis_id),
            )
            self._commit()
            existing = self.get_study_analysis(analysis_id)
        assert existing is not None
        return existing

    @staticmethod
    def _study_svg(analyses: Sequence[Mapping[str, Any]]) -> str:
        width = 800
        height = max(100, 44 * len(analyses) + 40)
        rows = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
            "<rect width='100%' height='100%' fill='white'/>",
        ]
        for index, value in enumerate(analyses):
            y = 36 + index * 44
            mean = float(value.get("mean_delta") or 0.0)
            x = 400 + max(-300, min(300, mean * 200))
            rows.append(
                f"<text x='16' y='{y}' font-size='12'>{value.get('name')}</text>"
            )
            rows.append(
                f"<circle cx='{x:.1f}' cy='{y - 4}' r='4' fill='#3768b0'/>"
            )
        rows.append("</svg>")
        return "".join(rows)

    def get_study_analysis(self, analysis_id: str) -> Optional[StudyAnalysis]:
        row = self.conn.execute(
            "SELECT * FROM adaptation_study_analyses WHERE id=?", (analysis_id,)
        ).fetchone()
        if row is None:
            return None
        return StudyAnalysis(
            id=str(row["id"]),
            protocol_revision_id=str(row["protocol_revision_id"]),
            status=str(row["status"]),
            analysis=_loads(row["analysis_json"], {}),
            content_hash=str(row["content_hash"]),
            evidence_classification=str(row["evidence_classification"]),
            work_item_id=row["work_item_id"],
            bundle_path=row["bundle_path"],
            created_at=str(row["created_at"]),
            completed_at=row["completed_at"],
            stage=str(row["stage"]),
            progress=_loads(row["progress_json"], {}),
            request=_loads(row["request_json"], {}),
            cancel_requested=bool(row["cancel_requested"]),
            error=row["error"] if "error" in row.keys() else None,
        )

    def create_study_deviation(
        self, revision_id: str, payload: Mapping[str, Any]
    ) -> StudyDeviation:
        if self.get_study_protocol(revision_id) is None:
            raise ValueError(f"unknown study protocol revision: {revision_id}")
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            raise ValueError("study deviations require a reason")
        change = dict(payload.get("change") or {})
        created_at = _now()
        deviation_id = _stable_id(
            "study-deviation",
            {
                "revision_id": revision_id,
                "reason": reason,
                "change": change,
                "nonce": uuid.uuid4().hex,
            },
        )
        self.conn.execute(
            """INSERT INTO adaptation_study_deviations
               (id,protocol_revision_id,reason,change_json,created_at)
               VALUES (?,?,?,?,?)""",
            (deviation_id, revision_id, reason, _canonical_json(change), created_at),
        )
        self._commit()
        return StudyDeviation(
            id=deviation_id,
            protocol_revision_id=revision_id,
            reason=reason,
            change=change,
            created_at=created_at,
        )

    def create_study_decision(
        self, revision_id: str, payload: Mapping[str, Any]
    ) -> StudyDecision:
        if self.get_study_protocol(revision_id) is None:
            raise ValueError(f"unknown study protocol revision: {revision_id}")
        decision = str(payload.get("decision") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        if not decision or not reason:
            raise ValueError("study decisions require decision and reason")
        created_at = _now()
        decision_id = _stable_id(
            "study-decision",
            {
                "revision_id": revision_id,
                "decision": decision,
                "reason": reason,
                "nonce": uuid.uuid4().hex,
            },
        )
        self.conn.execute(
            """INSERT INTO adaptation_study_decisions
               (id,protocol_revision_id,analysis_id,decision,reason,created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                decision_id,
                revision_id,
                payload.get("analysis_id"),
                decision,
                reason,
                created_at,
            ),
        )
        self._commit()
        return StudyDecision(
            id=decision_id,
            protocol_revision_id=revision_id,
            analysis_id=payload.get("analysis_id"),
            decision=decision,
            reason=reason,
            created_at=created_at,
        )

    # ----- V13 grounding ------------------------------------------------

    def create_grounding_profile(
        self, payload: Mapping[str, Any]
    ) -> GroundingProfile:
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("grounding profile name is required")
        profile_id = str(payload.get("id") or uuid.uuid4().hex)
        now = _now()
        self.conn.execute(
            """INSERT INTO grounding_profiles
               (id,name,description,latest_revision_id,created_at,updated_at)
               VALUES (?,?,?,?,?,?)""",
            (profile_id, name, payload.get("description"), None, now, now),
        )
        self._commit()
        return self.get_grounding_profile(profile_id)  # type: ignore[return-value]

    def _grounding_profile(self, row: Any) -> GroundingProfile:
        return GroundingProfile(
            id=str(row["id"]),
            name=str(row["name"]),
            description=row["description"],
            latest_revision_id=row["latest_revision_id"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def get_grounding_profile(self, profile_id: str) -> Optional[GroundingProfile]:
        row = self.conn.execute(
            "SELECT * FROM grounding_profiles WHERE id=?", (profile_id,)
        ).fetchone()
        return self._grounding_profile(row) if row else None

    def list_grounding_profiles(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT * FROM grounding_profiles ORDER BY updated_at DESC"
        ).fetchall()
        return _page(
            [self._grounding_profile(row).to_dict() for row in rows],
            limit=limit,
            offset=offset,
        )

    def create_grounding_profile_revision(
        self, profile_id: str, payload: Mapping[str, Any]
    ) -> GroundingProfileRevision:
        if self.get_grounding_profile(profile_id) is None:
            raise ValueError(f"unknown grounding profile: {profile_id}")
        task_type = str(payload.get("task_type") or "").strip()
        if task_type not in {
            "qa",
            "instruction",
            "extraction",
            "reasoning",
            "preference",
            "benchmark",
        }:
            raise ValueError("unsupported grounding task type")
        destination = str(payload.get("intended_destination") or "training")
        if destination not in {"training", "development_evaluation"}:
            raise EvidenceEligibilityError(
                "grounding destination must be training or development_evaluation"
            )
        definition = dict(payload)
        definition["task_type"] = task_type
        definition["intended_destination"] = destination
        definition.setdefault("seed", 42)
        definition.setdefault("quota", 100)
        previous = self.conn.execute(
            """SELECT COALESCE(MAX(revision_number),0) AS value
               FROM grounding_profile_revisions WHERE profile_id=?""",
            (profile_id,),
        ).fetchone()
        revision_number = int(previous["value"]) + 1
        content_hash = _content_hash(definition)
        revision_id = _stable_id(
            "grounding-profile-revision",
            {"profile_id": profile_id, "content_hash": content_hash},
        )
        now = _now()
        self.conn.execute(
            """INSERT OR IGNORE INTO grounding_profile_revisions
               (id,profile_id,revision_number,definition_json,content_hash,created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                revision_id,
                profile_id,
                revision_number,
                _canonical_json(definition),
                content_hash,
                now,
            ),
        )
        self.conn.execute(
            "UPDATE grounding_profiles SET latest_revision_id=?,updated_at=? WHERE id=?",
            (revision_id, now, profile_id),
        )
        self._commit()
        result = self.get_grounding_profile_revision(revision_id)
        assert result is not None
        return result

    def get_grounding_profile_revision(
        self, revision_id: str
    ) -> Optional[GroundingProfileRevision]:
        row = self.conn.execute(
            "SELECT * FROM grounding_profile_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        if row is None:
            return None
        return GroundingProfileRevision(
            id=str(row["id"]),
            profile_id=str(row["profile_id"]),
            revision_number=int(row["revision_number"]),
            definition=_loads(row["definition_json"], {}),
            content_hash=str(row["content_hash"]),
            created_at=str(row["created_at"]),
        )

    def _source_records(
        self,
        *,
        source_version_id: Optional[str],
        records: Optional[Sequence[Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if records is not None:
            return [dict(value) for value in records]
        if not source_version_id:
            raise ValueError("grounding generation requires a source version or records")
        version = self.db.get_dataset_version(source_version_id)
        if version is None:
            raise ValueError(f"unknown dataset version: {source_version_id}")
        if version.status != "completed":
            raise ValueError("grounding requires a completed immutable dataset version")
        path = Path(version.storage_path) / "records.jsonl"
        if not path.is_file():
            raise ValueError("corpus version records are missing")
        output = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    output.append(json.loads(line))
        return output

    @staticmethod
    def _grounding_prompt(task_type: str, record: Mapping[str, Any]) -> str:
        text = str(record.get("text") or "")
        source_ref = str(record.get("source_ref") or record.get("document_id") or "source")
        instruction = {
            "qa": "Create one answerable question and concise answer.",
            "instruction": "Create one useful instruction and grounded response.",
            "extraction": "Create one structured extraction task and expected result.",
            "reasoning": "Create one worked problem whose answer is supported by the source.",
            "preference": "Create a prompt plus a strong and weak candidate answer.",
            "benchmark": "Create one development benchmark item with expected answer.",
        }[task_type]
        return (
            f"{instruction}\nReturn JSON and cite character spans from {source_ref}.\n"
            f"SOURCE:\n{text}"
        )

    @staticmethod
    def _default_grounded_output(
        task_type: str, record: Mapping[str, Any]
    ) -> Dict[str, Any]:
        text = str(record.get("text") or "").strip()
        excerpt = text[: min(len(text), 400)]
        title = str(record.get("title") or record.get("source_ref") or "the source")
        base = {
            "citation": {"span_start": 0, "span_end": len(excerpt)},
        }
        if task_type in {"qa", "benchmark"}:
            return {
                **base,
                "question": f"What does {title} state?",
                "answer": excerpt,
            }
        if task_type == "preference":
            return {
                **base,
                "prompt": f"Summarize {title}.",
                "chosen": excerpt,
                "rejected": "The source does not contain relevant information.",
            }
        if task_type == "extraction":
            return {
                **base,
                "instruction": f"Extract the key statement from {title}.",
                "result": {"text": excerpt},
            }
        return {
            **base,
            "instruction": f"Explain the key information in {title}.",
            "response": excerpt,
        }

    def generate_grounded_batch(
        self, payload: Mapping[str, Any]
    ) -> GroundedGenerationBatch:
        revision_id = str(payload.get("profile_revision_id") or "")
        revision = self.get_grounding_profile_revision(revision_id)
        if revision is None:
            raise ValueError(f"unknown grounding profile revision: {revision_id}")
        definition = revision.definition
        destination = str(definition.get("intended_destination") or "training")
        if destination not in {"training", "development_evaluation"}:
            raise EvidenceEligibilityError("protected evidence cannot guide grounding")
        source_version_id = str(payload.get("source_version_id") or "").strip() or None
        source_records = self._source_records(
            source_version_id=source_version_id,
            records=payload.get("records"),
        )
        task_type = str(definition["task_type"])
        seed = int(definition.get("seed") or 42)
        quota = max(
            1,
            min(
                int(payload.get("quota") or definition.get("quota") or 250),
                1_000_000,
            ),
        )
        ordered = sorted(
            source_records,
            key=lambda value: _content_hash(
                {"seed": seed, "identity": value.get("document_id") or value}
            ),
        )[:quota]
        source_hash = _content_hash(source_records)
        batch_identity = {
            "profile_revision_id": revision_id,
            "source_version_id": source_version_id,
            "source_hash": source_hash,
            "task_type": task_type,
            "seed": seed,
            "quota": quota,
        }
        batch_id = _stable_id("grounded-batch", batch_identity)
        batch_id = str(payload.get("_batch_id") or "").strip() or batch_id
        existing = self.get_grounded_batch(batch_id)
        if existing is not None and existing.status not in {
            "queued",
            "running",
            "retrying",
        }:
            return existing
        if existing is not None and existing.source_hash != source_hash:
            raise ValueError(
                "The grounding source changed after the batch was created. Create a new immutable batch."
            )
        resume_ordinal = (
            min(
                len(ordered),
                max(0, int(existing.resume_cursor.get("next_ordinal") or 0)),
            )
            if existing is not None
            else 0
        )
        now = _now()
        if existing is None:
            self.conn.execute(
                """INSERT INTO grounded_generation_batches
                   (id,profile_revision_id,source_version_id,extraction_id,status,stage,
                    intended_destination,request_json,source_hash,content_hash,
                    candidate_count,accepted_count,rejected_count,coverage_json,
                    work_item_id,bundle_path,error,created_at,completed_at,
                    progress_json,resume_cursor_json,cancel_requested)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    batch_id,
                    revision_id,
                    source_version_id,
                    payload.get("extraction_id"),
                    "running",
                    "generating",
                    destination,
                    _canonical_json(dict(payload)),
                    source_hash,
                    None,
                    0,
                    0,
                    0,
                    "{}",
                    payload.get("_work_item_id"),
                    None,
                    None,
                    now,
                    None,
                    _canonical_json({"current": 0, "total": len(ordered), "percent": 0}),
                    "{}",
                    0,
                ),
            )
        else:
            self.conn.execute(
                """UPDATE grounded_generation_batches
                   SET status='running',stage='generating',error=NULL,
                       cancel_requested=0,request_json=?,source_hash=?,
                       progress_json=?,completed_at=NULL
                   WHERE id=?""",
                (
                    _canonical_json(dict(payload)),
                    source_hash,
                    _canonical_json(
                        {
                            "current": resume_ordinal,
                            "total": len(ordered),
                            "percent": (
                                round(100.0 * resume_ordinal / len(ordered), 2)
                                if ordered
                                else 100.0
                            ),
                        }
                    ),
                    batch_id,
                ),
            )
            if resume_ordinal == 0:
                self.conn.execute(
                    "DELETE FROM grounding_citations WHERE candidate_id IN "
                    "(SELECT id FROM grounded_candidates WHERE batch_id=?)",
                    (batch_id,),
                )
                self.conn.execute(
                    "DELETE FROM grounded_candidates WHERE batch_id=?",
                    (batch_id,),
                )
        self._commit()
        existing_candidates = (
            self.conn.execute(
                """SELECT * FROM grounded_candidates
                   WHERE batch_id=? AND ordinal<? ORDER BY ordinal""",
                (batch_id, resume_ordinal),
            ).fetchall()
            if resume_ordinal
            else []
        )
        candidates_payload = []
        accepted = 0
        rejected = 0
        source_counts: Counter[str] = Counter()
        citations_valid = 0
        citations_invalid = 0
        for row in existing_candidates:
            candidate = self._grounded_candidate(row)
            citation = candidate.citations[0] if candidate.citations else None
            structural = bool(citation and citation.structural_valid)
            accepted += int(candidate.status == "accepted")
            rejected += int(candidate.status != "accepted")
            citations_valid += int(structural)
            citations_invalid += int(not structural)
            source_counts[candidate.source_ref] += 1
            candidates_payload.append(
                {
                    "candidate_id": candidate.id,
                    "status": candidate.status,
                    "document_id": candidate.document_id,
                    "source_ref": candidate.source_ref,
                    "output": candidate.output,
                    "citation": {
                        "span_start": citation.span_start if citation else None,
                        "span_end": citation.span_end if citation else None,
                        "structural_valid": structural,
                    },
                }
            )
        for ordinal, record in enumerate(
            ordered[resume_ordinal:], start=resume_ordinal
        ):
            current = self.get_grounded_batch(batch_id)
            if current is not None and current.cancel_requested:
                self.conn.execute(
                    """UPDATE grounded_generation_batches
                       SET status='cancelled',stage='cancelled',error=?,
                           progress_json=?,resume_cursor_json=?,completed_at=?
                       WHERE id=?""",
                    (
                        "Generation was cancelled before publication.",
                        _canonical_json(
                            {
                                "current": ordinal,
                                "total": len(ordered),
                                "percent": (
                                    round(100.0 * ordinal / len(ordered), 2)
                                    if ordered
                                    else 100.0
                                ),
                            }
                        ),
                        _canonical_json({"next_ordinal": ordinal}),
                        _now(),
                        batch_id,
                    ),
                )
                self._commit()
                result = self.get_grounded_batch(batch_id)
                assert result is not None
                return result
            document_id = str(
                record.get("document_id")
                or _stable_id("document", record, length=32)
            )
            source_ref = str(record.get("source_ref") or document_id)
            text = str(record.get("text") or "")
            prompt = self._grounding_prompt(task_type, record)
            output: Dict[str, Any]
            try:
                if self.teacher is not None:
                    raw = self.teacher(prompt, definition, record)
                    output = json.loads(raw) if isinstance(raw, str) else dict(raw)
                else:
                    output = self._default_grounded_output(task_type, record)
            except Exception as exc:
                output = {"error": str(exc)}
            citation = dict(output.get("citation") or {})
            start = citation.get("span_start")
            end = citation.get("span_end")
            structural = (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(text)
            )
            status = "accepted" if structural else "rejected"
            rejection_reason = None if structural else "invalid_citation_span"
            accepted += int(structural)
            rejected += int(not structural)
            citations_valid += int(structural)
            citations_invalid += int(not structural)
            source_counts[source_ref] += 1
            candidate_identity = {
                "batch_id": batch_id,
                "ordinal": ordinal,
                "document_id": document_id,
                "source_ref": source_ref,
                "output": output,
            }
            candidate_id = _stable_id("grounded-candidate", candidate_identity)
            candidate_hash = _content_hash(candidate_identity)
            self.conn.execute(
                """INSERT INTO grounded_candidates
                   (id,batch_id,ordinal,task_type,status,document_id,source_ref,
                    source_hash,prompt_json,output_json,verifier_json,content_hash,
                    rejection_reason,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    candidate_id,
                    batch_id,
                    ordinal,
                    task_type,
                    status,
                    document_id,
                    source_ref,
                    _content_hash(record),
                    _canonical_json({"text": prompt}),
                    _canonical_json(output),
                    _canonical_json(
                        {
                            "structural_citation": structural,
                            "semantic_status": "not_run",
                        }
                    ),
                    candidate_hash,
                    rejection_reason,
                    now,
                ),
            )
            quoted = (
                text[int(start) : int(end)]
                if structural and start is not None and end is not None
                else ""
            )
            citation_id = _stable_id(
                "grounding-citation",
                {"candidate_id": candidate_id, "ordinal": 0, "citation": citation},
            )
            self.conn.execute(
                """INSERT INTO grounding_citations
                   (id,candidate_id,ordinal,document_id,source_ref,span_start,
                    span_end,locator_json,quoted_hash,structural_valid,
                    semantic_status,evidence_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    citation_id,
                    candidate_id,
                    0,
                    document_id,
                    source_ref,
                    start if isinstance(start, int) else None,
                    end if isinstance(end, int) else None,
                    _canonical_json(record.get("source_spans") or {}),
                    _content_hash(quoted),
                    int(structural),
                    "not_run",
                    _canonical_json({"quoted_text": quoted}),
                ),
            )
            candidates_payload.append(
                {
                    "candidate_id": candidate_id,
                    "status": status,
                    "document_id": document_id,
                    "source_ref": source_ref,
                    "output": output,
                    "citation": {
                        "span_start": start,
                        "span_end": end,
                        "structural_valid": structural,
                    },
                }
            )
            if (ordinal + 1) % 100 == 0 or ordinal + 1 == len(ordered):
                self.conn.execute(
                    """UPDATE grounded_generation_batches
                       SET candidate_count=?,accepted_count=?,rejected_count=?,
                           progress_json=?,resume_cursor_json=?
                       WHERE id=?""",
                    (
                        ordinal + 1,
                        accepted,
                        rejected,
                        _canonical_json(
                            {
                                "current": ordinal + 1,
                                "total": len(ordered),
                                "percent": (
                                    round(100.0 * (ordinal + 1) / len(ordered), 2)
                                    if ordered
                                    else 100.0
                                ),
                            }
                        ),
                        _canonical_json({"next_ordinal": ordinal + 1}),
                        batch_id,
                    ),
                )
                self._commit()
        coverage = {
            "documents_total": len(source_records),
            "documents_covered": len(
                {value["document_id"] for value in candidates_payload}
            ),
            "spans_total": len(candidates_payload),
            "citations_valid": citations_valid,
            "citations_invalid": citations_invalid,
            "source_concentration": dict(source_counts),
        }
        content_hash = _content_hash(candidates_payload)
        bundle_path, _ = _atomic_bundle(
            self.grounding_root / batch_id,
            {
                "candidates.jsonl": "".join(
                    json.dumps(value, sort_keys=True) + "\n"
                    for value in candidates_payload
                ),
                "coverage.json": json.dumps(coverage, indent=2, sort_keys=True) + "\n",
                "profile.json": json.dumps(
                    revision.to_dict(), indent=2, sort_keys=True
                )
                + "\n",
                "manifest.json": json.dumps(
                    {
                        "batch_id": batch_id,
                        "profile_revision_id": revision.id,
                        "profile_hash": revision.content_hash,
                        "source_version_id": source_version_id,
                        "source_hash": source_hash,
                        "verification_identity": _content_hash(
                            {
                                "profile": revision.content_hash,
                                "source": source_hash,
                                "task_type": task_type,
                                "seed": seed,
                                "quota": quota,
                            }
                        ),
                        "resume_cursor": {"next_ordinal": len(ordered)},
                        "publication": "atomic",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            },
        )
        completed = _now()
        self.conn.execute(
            """UPDATE grounded_generation_batches
               SET status='completed',stage='completed',content_hash=?,
                   candidate_count=?,accepted_count=?,rejected_count=?,
                   coverage_json=?,bundle_path=?,completed_at=?,
                   progress_json=?,resume_cursor_json=?,cancel_requested=0,error=NULL
               WHERE id=?""",
            (
                content_hash,
                len(candidates_payload),
                accepted,
                rejected,
                _canonical_json(coverage),
                str(bundle_path),
                completed,
                _canonical_json(
                    {"current": len(candidates_payload), "total": len(ordered), "percent": 100}
                ),
                _canonical_json({"next_ordinal": len(ordered)}),
                batch_id,
            ),
        )
        self._commit()
        result = self.get_grounded_batch(batch_id)
        assert result is not None
        return result

    def prepare_grounded_batch(
        self,
        payload: Mapping[str, Any],
        *,
        work_item_id: Optional[str] = None,
    ) -> GroundedGenerationBatch:
        revision_id = str(payload.get("profile_revision_id") or "")
        revision = self.get_grounding_profile_revision(revision_id)
        if revision is None:
            raise ValueError(f"unknown grounding profile revision: {revision_id}")
        destination = str(revision.definition.get("intended_destination") or "training")
        if destination not in {"training", "development_evaluation"}:
            raise EvidenceEligibilityError("protected evidence cannot guide grounding")
        source_version_id = str(payload.get("source_version_id") or "").strip() or None
        records = self._source_records(
            source_version_id=source_version_id,
            records=payload.get("records"),
        )
        quota = max(
            1,
            min(
                int(payload.get("quota") or revision.definition.get("quota") or 250),
                1_000_000,
            ),
        )
        identity = {
            "profile_revision_id": revision_id,
            "source_version_id": source_version_id,
            "source_hash": _content_hash(records),
            "task_type": revision.definition.get("task_type"),
            "seed": int(revision.definition.get("seed") or 42),
            "quota": quota,
        }
        batch_id = _stable_id("grounded-batch", identity)
        existing = self.get_grounded_batch(batch_id)
        if existing is not None:
            if work_item_id and not existing.work_item_id:
                self.conn.execute(
                    "UPDATE grounded_generation_batches SET work_item_id=? WHERE id=?",
                    (work_item_id, batch_id),
                )
                self._commit()
                existing = self.get_grounded_batch(batch_id)
            assert existing is not None
            return existing
        now = _now()
        self.conn.execute(
            """INSERT INTO grounded_generation_batches
               (id,profile_revision_id,source_version_id,extraction_id,status,stage,
                intended_destination,request_json,source_hash,content_hash,
                candidate_count,accepted_count,rejected_count,coverage_json,
                work_item_id,bundle_path,error,created_at,completed_at,
                progress_json,resume_cursor_json,cancel_requested)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                batch_id,
                revision_id,
                source_version_id,
                payload.get("extraction_id"),
                "queued",
                "waiting",
                destination,
                _canonical_json(dict(payload)),
                identity["source_hash"],
                None,
                0,
                0,
                0,
                "{}",
                work_item_id,
                None,
                None,
                now,
                None,
                _canonical_json({"current": 0, "total": min(quota, len(records)), "percent": 0}),
                "{}",
                0,
            ),
        )
        self._commit()
        result = self.get_grounded_batch(batch_id)
        assert result is not None
        return result

    def _grounded_batch(self, row: Any) -> GroundedGenerationBatch:
        return GroundedGenerationBatch(
            id=str(row["id"]),
            profile_revision_id=str(row["profile_revision_id"]),
            source_version_id=row["source_version_id"],
            extraction_id=row["extraction_id"],
            status=str(row["status"]),
            stage=str(row["stage"]),
            intended_destination=str(row["intended_destination"]),
            request=_loads(row["request_json"], {}),
            source_hash=str(row["source_hash"]),
            content_hash=row["content_hash"],
            candidate_count=int(row["candidate_count"]),
            accepted_count=int(row["accepted_count"]),
            rejected_count=int(row["rejected_count"]),
            coverage=_loads(row["coverage_json"], {}),
            work_item_id=row["work_item_id"],
            bundle_path=row["bundle_path"],
            error=row["error"],
            created_at=str(row["created_at"]),
            completed_at=row["completed_at"],
            progress=_loads(row["progress_json"], {}),
            resume_cursor=_loads(row["resume_cursor_json"], {}),
            cancel_requested=bool(row["cancel_requested"]),
        )

    def get_grounded_batch(
        self, batch_id: str
    ) -> Optional[GroundedGenerationBatch]:
        row = self.conn.execute(
            "SELECT * FROM grounded_generation_batches WHERE id=?", (batch_id,)
        ).fetchone()
        return self._grounded_batch(row) if row else None

    def list_grounded_batches(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT * FROM grounded_generation_batches ORDER BY created_at DESC"
        ).fetchall()
        return _page(
            [self._grounded_batch(row).to_dict() for row in rows],
            limit=limit,
            offset=offset,
        )

    def list_grounded_candidates(
        self,
        batch_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        if status:
            rows = self.conn.execute(
                """SELECT * FROM grounded_candidates
                   WHERE batch_id=? AND status=? ORDER BY ordinal""",
                (batch_id, status),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM grounded_candidates WHERE batch_id=? ORDER BY ordinal",
                (batch_id,),
            ).fetchall()
        values = [self._grounded_candidate(row).to_dict() for row in rows]
        return _page(values, limit=limit, offset=offset)

    def _grounded_candidate(self, row: Any) -> GroundedCandidate:
        citations = self.conn.execute(
            "SELECT * FROM grounding_citations WHERE candidate_id=? ORDER BY ordinal",
            (row["id"],),
        ).fetchall()
        return GroundedCandidate(
            id=str(row["id"]),
            batch_id=str(row["batch_id"]),
            ordinal=int(row["ordinal"]),
            task_type=str(row["task_type"]),
            status=str(row["status"]),
            document_id=str(row["document_id"]),
            source_ref=str(row["source_ref"]),
            source_hash=str(row["source_hash"]),
            prompt=_loads(row["prompt_json"], {}),
            output=_loads(row["output_json"], {}),
            verifier=_loads(row["verifier_json"], {}),
            content_hash=str(row["content_hash"]),
            rejection_reason=row["rejection_reason"],
            created_at=str(row["created_at"]),
            citations=tuple(
                GroundingCitation(
                    id=str(value["id"]),
                    candidate_id=str(value["candidate_id"]),
                    ordinal=int(value["ordinal"]),
                    document_id=str(value["document_id"]),
                    source_ref=str(value["source_ref"]),
                    span_start=value["span_start"],
                    span_end=value["span_end"],
                    locator=_loads(value["locator_json"], {}),
                    quoted_hash=str(value["quoted_hash"]),
                    structural_valid=bool(value["structural_valid"]),
                    semantic_status=str(value["semantic_status"]),
                    evidence=_loads(value["evidence_json"], {}),
                )
                for value in citations
            ),
        )

    def grounding_review_proposal(self, batch_id: str) -> Dict[str, Any]:
        batch = self.get_grounded_batch(batch_id)
        if batch is None:
            raise ValueError(f"unknown grounded generation batch: {batch_id}")
        if batch.status != "completed":
            raise ValueError("grounded generation must complete before review")
        return {
            "source_kind": "imported_jsonl",
            "source_ref": str(Path(batch.bundle_path or "") / "candidates.jsonl"),
            "name": f"Review grounded {batch.id}",
            "metadata": {
                "grounded_batch_id": batch.id,
                "intended_destination": batch.intended_destination,
                "source_hash": batch.source_hash,
            },
            "requires_explicit_queue_creation": True,
        }

    # ----- V14 specialized tasks ---------------------------------------

    @staticmethod
    def list_specialized_tasks(
        *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return _page(
            [value.to_dict() for value in _SPECIALIZED_TASKS],
            limit=limit,
            offset=offset,
        )

    @staticmethod
    def specialized_task_readiness(
        payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        task_id = str(payload.get("task_id") or "").strip()
        descriptor = next(
            (value for value in _SPECIALIZED_TASKS if value.id == task_id),
            None,
        )
        if descriptor is None:
            raise ValueError(f"unknown specialized task: {task_id}")
        backend = str(payload.get("backend") or "pytorch").strip().lower()
        blockers: list[str] = []
        warnings: list[str] = []
        if backend not in {"pytorch", "torch"}:
            blockers.append(
                "This task is verified on PyTorch only. The selected backend does not yet have a complete optimizer, evaluation, artifact, and replay contract."
            )
        if not str(payload.get("model") or "").strip():
            blockers.append("Choose a compatible model.")
        if not (
            str(payload.get("dataset_version_id") or "").strip()
            or str(payload.get("dataset") or "").strip()
        ):
            blockers.append("Choose labeled training data.")
        if descriptor.task_kind == "classification" and not payload.get(
            "label_schema_revision_id"
        ):
            warnings.append(
                "Halo Forge will infer the label schema from the complete training split."
            )
        return {
            "task": descriptor.to_dict(),
            "ready": not blockers,
            "display_status": "Ready for a proof run" if not blockers else "Setup needed",
            "blockers": blockers,
            "warnings": warnings,
            "resolved": {
                "backend": "pytorch",
                "trainer_mode": descriptor.trainer_mode,
                "split_policy": (
                    "group identical media across protected splits"
                    if descriptor.modality in {"image", "audio"}
                    else "preserve the immutable dataset split"
                ),
                "loss": {
                    "classification": "cross entropy",
                    "embedding": "multiple-negative ranking",
                    "reranking": "relevance or pairwise margin",
                }.get(descriptor.task_kind, "task-specific verified loss"),
            },
        }

    @staticmethod
    def verify_specialized_artifact(
        artifact_path: str | Path,
        payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        from .specialized import SpecializedServingRuntime

        path = Path(artifact_path).expanduser().resolve()
        required = ["config.json", "task_config.json"]
        missing = [name for name in required if not (path / name).is_file()]
        if missing:
            return {
                "valid": False,
                "artifact_path": str(path),
                "errors": [f"Missing {name}" for name in missing],
                "use_model": False,
            }
        runtime = SpecializedServingRuntime(path)
        fixed = payload.get("fixed_input")
        if runtime.task == "embed":
            result: Any = runtime.embed(
                [str(fixed or "Halo Forge fixed embedding input")]
            )
        elif runtime.task == "rerank":
            result = runtime.rerank(
                str(payload.get("query") or "Halo Forge fixed rerank query"),
                [
                    str(value)
                    for value in payload.get("documents")
                    or ["relevant document", "unrelated document"]
                ],
            )
        else:
            if fixed in (None, ""):
                modality = str(runtime.task_config.get("task_modality") or "text")
                if modality in {"image", "audio"}:
                    raise ValueError(
                        f"fixed_input is required to verify a {modality} artifact"
                    )
                fixed = "Halo Forge fixed classification input"
            result = runtime.classify([fixed], top_k=1)
        return {
            "valid": True,
            "artifact_path": str(path),
            "task": runtime.task,
            "task_config": runtime.task_config,
            "fixed_input_result": result,
            "checks": [
                "model reloaded",
                "processor or tokenizer reloaded",
                "task contract loaded",
                "fixed-input inference completed",
            ],
            "use_model": True,
        }

    def create_task_label_schema(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        task_kind = str(payload.get("task_kind") or "").strip()
        modality = str(payload.get("modality") or "text").strip()
        if not name or task_kind not in {
            "classification",
            "multilabel",
            "embedding",
            "reranking",
        }:
            raise ValueError("valid name and specialized task_kind are required")
        schema_id = str(payload.get("id") or uuid.uuid4().hex)
        now = _now()
        self.conn.execute(
            """INSERT INTO task_label_schemas
               (id,name,task_kind,modality,latest_revision_id,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (schema_id, name, task_kind, modality, None, now, now),
        )
        self._commit()
        return {
            "id": schema_id,
            "name": name,
            "task_kind": task_kind,
            "modality": modality,
            "latest_revision_id": None,
            "created_at": now,
            "updated_at": now,
        }

    def create_task_label_schema_revision(
        self, schema_id: str, payload: Mapping[str, Any]
    ) -> TaskLabelSchemaRevision:
        schema = self.conn.execute(
            "SELECT * FROM task_label_schemas WHERE id=?", (schema_id,)
        ).fetchone()
        if schema is None:
            raise ValueError(f"unknown task label schema: {schema_id}")
        labels = [str(value).strip() for value in payload.get("labels") or []]
        if schema["task_kind"] in {"classification", "multilabel"}:
            if len(set(labels)) < 2:
                raise ValueError("classification schemas require at least two labels")
            if len(set(labels)) != len(labels):
                raise ValueError("task labels must be unique")
        definition = dict(payload)
        definition["labels"] = labels
        previous = self.conn.execute(
            """SELECT COALESCE(MAX(revision_number),0) AS value
               FROM task_label_schema_revisions WHERE schema_id=?""",
            (schema_id,),
        ).fetchone()
        revision_number = int(previous["value"]) + 1
        content_hash = _content_hash(definition)
        revision_id = _stable_id(
            "task-label-revision",
            {"schema_id": schema_id, "content_hash": content_hash},
        )
        now = _now()
        self.conn.execute(
            """INSERT OR IGNORE INTO task_label_schema_revisions
               (id,schema_id,revision_number,definition_json,content_hash,created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                revision_id,
                schema_id,
                revision_number,
                _canonical_json(definition),
                content_hash,
                now,
            ),
        )
        self.conn.execute(
            "UPDATE task_label_schemas SET latest_revision_id=?,updated_at=? WHERE id=?",
            (revision_id, now, schema_id),
        )
        self._commit()
        return TaskLabelSchemaRevision(
            id=revision_id,
            schema_id=schema_id,
            revision_number=revision_number,
            definition=definition,
            content_hash=content_hash,
            created_at=now,
        )

    @staticmethod
    def classification_metrics(
        expected: Sequence[Any],
        predicted: Sequence[Any],
        *,
        multilabel: bool = False,
    ) -> Dict[str, float]:
        if len(expected) != len(predicted) or not expected:
            raise ValueError("expected and predicted must be non-empty and aligned")
        if multilabel:
            expected_sets = [set(value) for value in expected]
            predicted_sets = [set(value) for value in predicted]
            labels = sorted(set().union(*expected_sets, *predicted_sets))
            tp = fp = fn = 0
            exact = 0
            hamming = 0
            for left, right in zip(expected_sets, predicted_sets):
                tp += len(left & right)
                fp += len(right - left)
                fn += len(left - right)
                exact += int(left == right)
                hamming += len(left ^ right)
            micro_f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            per_label = []
            for label in labels:
                label_tp = sum(label in left and label in right for left, right in zip(expected_sets, predicted_sets))
                label_fp = sum(label not in left and label in right for left, right in zip(expected_sets, predicted_sets))
                label_fn = sum(label in left and label not in right for left, right in zip(expected_sets, predicted_sets))
                per_label.append(
                    2 * label_tp / max(1, 2 * label_tp + label_fp + label_fn)
                )
            return {
                "micro_f1": micro_f1,
                "macro_f1": sum(per_label) / max(1, len(per_label)),
                "hamming_loss": hamming / max(1, len(expected_sets) * max(1, len(labels))),
                "exact_match": exact / len(expected_sets),
            }
        expected_labels = [str(value) for value in expected]
        predicted_labels = [str(value) for value in predicted]
        labels = sorted(set(expected_labels) | set(predicted_labels))
        correct = sum(left == right for left, right in zip(expected_labels, predicted_labels))
        recalls = []
        f1s = []
        for label in labels:
            tp = sum(left == label and right == label for left, right in zip(expected_labels, predicted_labels))
            fp = sum(left != label and right == label for left, right in zip(expected_labels, predicted_labels))
            fn = sum(left == label and right != label for left, right in zip(expected_labels, predicted_labels))
            recalls.append(tp / max(1, tp + fn))
            f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
        return {
            "accuracy": correct / len(expected_labels),
            "balanced_accuracy": sum(recalls) / max(1, len(recalls)),
            "macro_f1": sum(f1s) / max(1, len(f1s)),
        }

    @staticmethod
    def retrieval_metrics(
        rankings: Sequence[Sequence[str]],
        relevant: Sequence[Sequence[str]],
        *,
        k: int = 10,
    ) -> Dict[str, float]:
        if len(rankings) != len(relevant) or not rankings:
            raise ValueError("rankings and relevant must be non-empty and aligned")
        recalls = []
        reciprocal = []
        ndcgs = []
        for ranking, expected in zip(rankings, relevant):
            expected_set = set(expected)
            top = list(ranking)[:k]
            recalls.append(len(expected_set & set(top)) / max(1, len(expected_set)))
            first = next(
                (index + 1 for index, value in enumerate(ranking) if value in expected_set),
                None,
            )
            reciprocal.append(0.0 if first is None else 1.0 / first)
            dcg = sum(
                1.0 / math.log2(index + 2)
                for index, value in enumerate(top)
                if value in expected_set
            )
            ideal = sum(
                1.0 / math.log2(index + 2)
                for index in range(min(len(expected_set), k))
            )
            ndcgs.append(dcg / ideal if ideal else 0.0)
        return {
            f"recall_at_{k}": sum(recalls) / len(recalls),
            "mrr": sum(reciprocal) / len(reciprocal),
            "ndcg": sum(ndcgs) / len(ndcgs),
        }

    @staticmethod
    def predict_classification(payload: Mapping[str, Any]) -> ClassificationPrediction:
        scores = {
            str(key): _finite(value, f"score.{key}")
            for key, value in dict(payload.get("scores") or {}).items()
        }
        if not scores:
            raise ValueError("classification prediction requires model scores")
        count = max(1, int(payload.get("top_k") or 1))
        labels = [
            key for key, _value in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:count]
        ]
        return ClassificationPrediction(
            labels=labels,
            scores=scores,
            model_artifact_id=payload.get("model_artifact_id"),
        )

    @staticmethod
    def embedding_result(payload: Mapping[str, Any]) -> EmbeddingResult:
        vector = [_finite(value, "embedding") for value in payload.get("embedding") or []]
        if not vector:
            raise ValueError("embedding result cannot be empty")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise ValueError("embedding result cannot be a zero vector")
        normalized = [value / norm for value in vector]
        return EmbeddingResult(
            embedding=normalized,
            dimensions=len(normalized),
            model_artifact_id=payload.get("model_artifact_id"),
        )

    @staticmethod
    def rerank_result(payload: Mapping[str, Any]) -> RerankResult:
        items = [dict(value) for value in payload.get("items") or []]
        for item in items:
            item["score"] = _finite(item.get("score"), "rerank score")
        items.sort(key=lambda value: value["score"], reverse=True)
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank
        return RerankResult(
            items=items,
            model_artifact_id=payload.get("model_artifact_id"),
        )

    # ----- V15 environments --------------------------------------------

    def create_environment(self, payload: Mapping[str, Any]) -> AgentEnvironment:
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("environment name is required")
        environment_id = str(payload.get("id") or uuid.uuid4().hex)
        now = _now()
        self.conn.execute(
            """INSERT INTO agent_environments
               (id,name,description,latest_revision_id,archived,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                environment_id,
                name,
                payload.get("description"),
                None,
                0,
                now,
                now,
            ),
        )
        self._commit()
        return self.get_environment(environment_id)  # type: ignore[return-value]

    def _environment(self, row: Any) -> AgentEnvironment:
        return AgentEnvironment(
            id=str(row["id"]),
            name=str(row["name"]),
            description=row["description"],
            latest_revision_id=row["latest_revision_id"],
            archived=bool(row["archived"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def get_environment(self, environment_id: str) -> Optional[AgentEnvironment]:
        row = self.conn.execute(
            "SELECT * FROM agent_environments WHERE id=?", (environment_id,)
        ).fetchone()
        return self._environment(row) if row else None

    def list_environments(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT * FROM agent_environments ORDER BY updated_at DESC"
        ).fetchall()
        return _page(
            [self._environment(row).to_dict() for row in rows],
            limit=limit,
            offset=offset,
        )

    def create_environment_revision(
        self, environment_id: str, payload: Mapping[str, Any]
    ) -> AgentEnvironmentRevision:
        if self.get_environment(environment_id) is None:
            raise ValueError(f"unknown agent environment: {environment_id}")
        adapter_id = str(payload.get("adapter_id") or "state_machine")
        try:
            adapter_type = ENVIRONMENT_ADAPTERS.get(adapter_id)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        definition = dict(payload)
        definition.setdefault("initial_state", {})
        definition.setdefault("transitions", {})
        definition.setdefault("max_steps", 16)
        definition["external_writes"] = False
        tools = [dict(value) for value in definition.get("tools") or []]
        adapter_version = adapter_type.descriptor.version
        implementation_hash = _content_hash(
            {"adapter_id": adapter_id, "adapter_version": adapter_version}
        )
        fixture_hash = _content_hash(
            {
                "initial_state": definition["initial_state"],
                "transitions": definition["transitions"],
            }
        )
        content_hash = _content_hash(
            {
                "definition": definition,
                "implementation_hash": implementation_hash,
                "fixture_hash": fixture_hash,
            }
        )
        previous = self.conn.execute(
            """SELECT COALESCE(MAX(revision_number),0) AS value
               FROM agent_environment_revisions WHERE environment_id=?""",
            (environment_id,),
        ).fetchone()
        revision_number = int(previous["value"]) + 1
        revision_id = _stable_id(
            "environment-revision",
            {"environment_id": environment_id, "content_hash": content_hash},
        )
        storage_path, _ = _atomic_bundle(
            self.environment_root / environment_id / revision_id,
            {
                "environment.json": json.dumps(
                    definition, indent=2, sort_keys=True
                )
                + "\n",
                "fixtures.json": json.dumps(
                    {
                        "initial_state": definition["initial_state"],
                        "transitions": definition["transitions"],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            },
        )
        now = _now()
        self.conn.execute(
            """INSERT OR IGNORE INTO agent_environment_revisions
               (id,environment_id,revision_number,adapter_id,adapter_version,
                implementation_hash,definition_json,fixture_hash,content_hash,
                storage_path,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                revision_id,
                environment_id,
                revision_number,
                adapter_id,
                adapter_version,
                implementation_hash,
                _canonical_json(definition),
                fixture_hash,
                content_hash,
                str(storage_path),
                now,
            ),
        )
        for ordinal, tool in enumerate(tools):
            name = str(tool.get("name") or "").strip()
            if not name:
                raise ValueError("environment tools require names")
            self.conn.execute(
                """INSERT OR IGNORE INTO environment_tools
                   (id,environment_revision_id,ordinal,name,definition_json,
                    implementation_hash)
                   VALUES (?,?,?,?,?,?)""",
                (
                    _stable_id(
                        "environment-tool",
                        {
                            "revision_id": revision_id,
                            "ordinal": ordinal,
                            "tool": tool,
                        },
                    ),
                    revision_id,
                    ordinal,
                    name,
                    _canonical_json(tool),
                    _content_hash(tool),
                ),
            )
        self.conn.execute(
            "UPDATE agent_environments SET latest_revision_id=?,updated_at=? WHERE id=?",
            (revision_id, now, environment_id),
        )
        self._commit()
        result = self.get_environment_revision(revision_id)
        assert result is not None
        return result

    def get_environment_revision(
        self, revision_id: str
    ) -> Optional[AgentEnvironmentRevision]:
        row = self.conn.execute(
            "SELECT * FROM agent_environment_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        if row is None:
            return None
        tools = self.conn.execute(
            """SELECT * FROM environment_tools
               WHERE environment_revision_id=? ORDER BY ordinal""",
            (revision_id,),
        ).fetchall()
        return AgentEnvironmentRevision(
            id=str(row["id"]),
            environment_id=str(row["environment_id"]),
            revision_number=int(row["revision_number"]),
            adapter_id=str(row["adapter_id"]),
            adapter_version=str(row["adapter_version"]),
            implementation_hash=str(row["implementation_hash"]),
            definition=_loads(row["definition_json"], {}),
            fixture_hash=str(row["fixture_hash"]),
            content_hash=str(row["content_hash"]),
            storage_path=str(row["storage_path"]),
            created_at=str(row["created_at"]),
            tools=tuple(
                EnvironmentToolDescriptor(
                    id=str(value["id"]),
                    name=str(value["name"]),
                    ordinal=int(value["ordinal"]),
                    definition=_loads(value["definition_json"], {}),
                    implementation_hash=str(value["implementation_hash"]),
                )
                for value in tools
            ),
        )

    def create_episode_suite(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        purpose = str(payload.get("purpose") or "development")
        if not name or purpose not in {"development", "operational", "holdout"}:
            raise ValueError("episode suite requires name and valid purpose")
        suite_id = str(payload.get("id") or uuid.uuid4().hex)
        now = _now()
        self.conn.execute(
            """INSERT INTO episode_suites
               (id,name,purpose,latest_revision_id,created_at,updated_at)
               VALUES (?,?,?,?,?,?)""",
            (suite_id, name, purpose, None, now, now),
        )
        self._commit()
        return {
            "id": suite_id,
            "name": name,
            "purpose": purpose,
            "latest_revision_id": None,
            "created_at": now,
            "updated_at": now,
        }

    def create_episode_suite_revision(
        self, suite_id: str, payload: Mapping[str, Any]
    ) -> EpisodeSuiteRevision:
        suite = self.conn.execute(
            "SELECT * FROM episode_suites WHERE id=?", (suite_id,)
        ).fetchone()
        if suite is None:
            raise ValueError(f"unknown episode suite: {suite_id}")
        environment_revision_id = str(payload.get("environment_revision_id") or "")
        if self.get_environment_revision(environment_revision_id) is None:
            raise ValueError("episode suite requires a valid environment revision")
        definition = dict(payload)
        items = list(definition.get("items") or [])
        if not items:
            raise ValueError("episode suite requires at least one item")
        definition.setdefault("generation", {"seed": 42, "temperature": 0})
        definition.setdefault("max_steps", 16)
        content_hash = _content_hash(definition)
        previous = self.conn.execute(
            """SELECT COALESCE(MAX(revision_number),0) AS value
               FROM episode_suite_revisions WHERE suite_id=?""",
            (suite_id,),
        ).fetchone()
        revision_number = int(previous["value"]) + 1
        revision_id = _stable_id(
            "episode-suite-revision",
            {"suite_id": suite_id, "content_hash": content_hash},
        )
        now = _now()
        self.conn.execute(
            """INSERT OR IGNORE INTO episode_suite_revisions
               (id,suite_id,revision_number,environment_revision_id,
                definition_json,content_hash,created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                revision_id,
                suite_id,
                revision_number,
                environment_revision_id,
                _canonical_json(definition),
                content_hash,
                now,
            ),
        )
        self.conn.execute(
            "UPDATE episode_suites SET latest_revision_id=?,updated_at=? WHERE id=?",
            (revision_id, now, suite_id),
        )
        self._commit()
        return EpisodeSuiteRevision(
            id=revision_id,
            suite_id=suite_id,
            revision_number=revision_number,
            environment_revision_id=environment_revision_id,
            definition=definition,
            content_hash=content_hash,
            created_at=now,
        )

    def get_episode_suite_revision(
        self, revision_id: str
    ) -> Optional[EpisodeSuiteRevision]:
        row = self.conn.execute(
            "SELECT * FROM episode_suite_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        if row is None:
            return None
        return EpisodeSuiteRevision(
            id=str(row["id"]),
            suite_id=str(row["suite_id"]),
            revision_number=int(row["revision_number"]),
            environment_revision_id=str(row["environment_revision_id"]),
            definition=_loads(row["definition_json"], {}),
            content_hash=str(row["content_hash"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _apply_state_delta(state: Mapping[str, Any], delta: Mapping[str, Any]) -> Dict[str, Any]:
        output = json.loads(json.dumps(state))
        for key, value in delta.items():
            if value is None:
                output.pop(str(key), None)
            else:
                output[str(key)] = value
        return output

    def run_episode(
        self, payload: Mapping[str, Any]
    ) -> AgentEpisode:
        revision_id = str(payload.get("suite_revision_id") or "")
        revision = self.get_episode_suite_revision(revision_id)
        if revision is None:
            raise ValueError(f"unknown episode suite revision: {revision_id}")
        environment = self.get_environment_revision(revision.environment_revision_id)
        assert environment is not None
        suite_item_id = str(payload.get("suite_item_id") or "")
        item = next(
            (
                dict(value)
                for value in revision.definition.get("items") or []
                if str(value.get("id") or "") == suite_item_id
            ),
            None,
        )
        if item is None:
            raise ValueError("episode suite item is missing")
        subject_type = str(payload.get("subject_type") or "artifact")
        subject_ref = str(payload.get("subject_ref") or "")
        if not subject_ref:
            raise ValueError("episode subject_ref is required")
        subject_hash = str(payload.get("subject_hash") or _content_hash(subject_ref))
        seed = int(payload.get("seed") or 42)
        episode_identity = {
            "suite_revision_id": revision_id,
            "suite_item_id": suite_item_id,
            "subject_hash": subject_hash,
            "seed": seed,
        }
        episode_id = str(payload.get("_episode_id") or "").strip() or _stable_id(
            "episode", episode_identity
        )
        existing = self.get_episode(episode_id)
        if existing is not None and existing.status not in {
            "queued",
            "running",
            "retrying",
        }:
            return existing
        state = dict(environment.definition.get("initial_state") or {})
        state.update(dict(item.get("initial_state") or {}))
        initial_hash = _content_hash(state)
        now = _now()
        if existing is None:
            self.conn.execute(
                """INSERT INTO agent_episodes
                   (id,suite_revision_id,suite_item_id,subject_type,subject_ref,
                    subject_hash,seed,status,terminal_reason,metrics_json,
                    initial_state_hash,final_state_hash,snapshot_path,trace_hash,
                    work_item_id,error,created_at,completed_at,stage,progress_json,
                    request_json,resume_cursor_json,cancel_requested,parent_episode_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    episode_id,
                    revision_id,
                    suite_item_id,
                    subject_type,
                    subject_ref,
                    subject_hash,
                    seed,
                    "running",
                    None,
                    "{}",
                    initial_hash,
                    None,
                    None,
                    None,
                    payload.get("_work_item_id"),
                    None,
                    now,
                    None,
                    "running_episode",
                    _canonical_json({"current": 0, "total": 1, "percent": 0}),
                    _canonical_json(dict(payload)),
                    "{}",
                    0,
                    payload.get("parent_episode_id"),
                ),
            )
        else:
            self.conn.execute(
                """UPDATE agent_episodes
                   SET status='running',stage='running_episode',error=NULL,
                       cancel_requested=0,request_json=?,completed_at=NULL
                   WHERE id=?""",
                (_canonical_json(dict(payload)), episode_id),
            )
            self.conn.execute(
                "DELETE FROM agent_episode_steps WHERE episode_id=?",
                (episode_id,),
            )
        self._commit()
        transitions = dict(environment.definition.get("transitions") or {})
        actions = [dict(value) for value in payload.get("actions") or []]
        max_steps = min(
            int(revision.definition.get("max_steps") or 16),
            int(environment.definition.get("max_steps") or 16),
        )
        invalid_calls = 0
        tool_errors = 0
        reward_total = 0.0
        terminal_reason = "max_steps"
        trace = []
        for ordinal, action in enumerate(actions[:max_steps]):
            current = self.get_episode(episode_id)
            if current is not None and current.cancel_requested:
                self.conn.execute(
                    """UPDATE agent_episodes
                       SET status='cancelled',stage='cancelled',
                           terminal_reason='cancelled',
                           progress_json=?,resume_cursor_json=?,completed_at=?
                       WHERE id=?""",
                    (
                        _canonical_json(
                            {
                                "current": ordinal,
                                "total": max(1, min(len(actions), max_steps)),
                                "percent": (
                                    round(
                                        100.0
                                        * ordinal
                                        / max(1, min(len(actions), max_steps)),
                                        2,
                                    )
                                ),
                            }
                        ),
                        _canonical_json({"next_step": ordinal}),
                        _now(),
                        episode_id,
                    ),
                )
                self._commit()
                result = self.get_episode(episode_id)
                assert result is not None
                return result
            name = str(action.get("name") or action.get("tool") or "")
            transition = dict(transitions.get(name) or {})
            error = None
            if not transition:
                invalid_calls += 1
                error = f"unknown action: {name}"
                delta: Dict[str, Any] = {}
                reward = 0.0
                terminal = False
            else:
                delta = dict(transition.get("state_delta") or {})
                reward = float(transition.get("reward") or 0.0)
                terminal = bool(transition.get("terminal", False))
                state = self._apply_state_delta(state, delta)
            reward_total += reward
            state_hash = _content_hash(state)
            step_created = _now()
            observation = {
                "goal": item.get("goal"),
                "state": state,
                "available_actions": sorted(transitions),
            }
            self.conn.execute(
                """INSERT INTO agent_episode_steps
                   (episode_id,ordinal,observation_json,raw_output,action_json,
                    tool_call_json,tool_result_json,state_delta_json,state_hash,
                    verifier_json,latency_ms,error,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    episode_id,
                    ordinal,
                    _canonical_json(observation),
                    action.get("raw_output"),
                    _canonical_json(action),
                    _canonical_json(action.get("tool_call"))
                    if action.get("tool_call") is not None
                    else None,
                    _canonical_json(transition.get("result"))
                    if transition.get("result") is not None
                    else None,
                    _canonical_json(delta),
                    state_hash,
                    _canonical_json(
                        {
                            "reward": reward,
                            "passed": error is None,
                        }
                    ),
                    action.get("latency_ms"),
                    error,
                    step_created,
                ),
            )
            trace.append(
                {
                    "ordinal": ordinal,
                    "action": action,
                    "state_delta": delta,
                    "state_hash": state_hash,
                    "reward": reward,
                    "error": error,
                }
            )
            self.conn.execute(
                """UPDATE agent_episodes
                   SET progress_json=?,resume_cursor_json=?
                   WHERE id=?""",
                (
                    _canonical_json(
                        {
                            "current": ordinal + 1,
                            "total": max(1, min(len(actions), max_steps)),
                            "percent": round(
                                100.0
                                * (ordinal + 1)
                                / max(1, min(len(actions), max_steps)),
                                2,
                            ),
                        }
                    ),
                    _canonical_json({"next_step": ordinal + 1}),
                    episode_id,
                ),
            )
            self._commit()
            if terminal:
                terminal_reason = str(transition.get("terminal_reason") or "terminal")
                break
        expected = dict(item.get("expected_state") or {})
        invariant_ok = all(state.get(key) == value for key, value in expected.items())
        status = "completed"
        final_hash = _content_hash(state)
        metrics = {
            "task_success": float(invariant_ok),
            "invariant_satisfaction": float(invariant_ok),
            "invalid_calls": invalid_calls,
            "tool_errors": tool_errors,
            "step_count": len(trace),
            "reward": reward_total,
        }
        snapshot_path, trace_hash = _atomic_bundle(
            self.episode_root / episode_id,
            {
                "initial_state.json": json.dumps(
                    item.get("initial_state") or {}, indent=2, sort_keys=True
                )
                + "\n",
                "final_state.json": json.dumps(state, indent=2, sort_keys=True) + "\n",
                "trace.json": json.dumps(trace, indent=2, sort_keys=True) + "\n",
                "metrics.json": json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                "manifest.json": json.dumps(
                    {
                        "episode_id": episode_id,
                        "environment_revision_id": environment.id,
                        "environment_hash": environment.content_hash,
                        "suite_revision_id": revision.id,
                        "suite_hash": revision.content_hash,
                        "subject": {
                            "type": subject_type,
                            "ref": subject_ref,
                            "hash": subject_hash,
                        },
                        "seed": seed,
                        "parent_episode_id": payload.get("parent_episode_id"),
                        "initial_state_hash": initial_hash,
                        "final_state_hash": final_hash,
                        "snapshot_boundary": "completed_episode",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            },
        )
        completed = _now()
        self.conn.execute(
            """UPDATE agent_episodes
               SET status=?,terminal_reason=?,metrics_json=?,final_state_hash=?,
                   snapshot_path=?,trace_hash=?,completed_at=?,stage='completed',
                   progress_json=?,resume_cursor_json=?,cancel_requested=0,error=NULL
               WHERE id=?""",
            (
                status,
                terminal_reason,
                _canonical_json(metrics),
                final_hash,
                str(snapshot_path),
                trace_hash,
                completed,
                _canonical_json(
                    {"current": len(trace), "total": max(1, len(trace)), "percent": 100}
                ),
                _canonical_json({"next_step": len(trace)}),
                episode_id,
            ),
        )
        self._commit()
        result = self.get_episode(episode_id)
        assert result is not None
        return result

    def prepare_episode(
        self,
        payload: Mapping[str, Any],
        *,
        work_item_id: Optional[str] = None,
    ) -> AgentEpisode:
        revision_id = str(payload.get("suite_revision_id") or "")
        revision = self.get_episode_suite_revision(revision_id)
        if revision is None:
            raise ValueError(f"unknown episode suite revision: {revision_id}")
        suite_item_id = str(payload.get("suite_item_id") or "")
        if not any(
            str(value.get("id") or "") == suite_item_id
            for value in revision.definition.get("items") or []
        ):
            raise ValueError("episode suite item is missing")
        subject_ref = str(payload.get("subject_ref") or "")
        if not subject_ref:
            raise ValueError("episode subject_ref is required")
        subject_hash = str(payload.get("subject_hash") or _content_hash(subject_ref))
        seed = int(payload.get("seed") or 42)
        identity = {
            "suite_revision_id": revision_id,
            "suite_item_id": suite_item_id,
            "subject_hash": subject_hash,
            "seed": seed,
            "parent_episode_id": payload.get("parent_episode_id"),
        }
        episode_id = _stable_id("episode", identity)
        existing = self.get_episode(episode_id)
        if existing is not None:
            if work_item_id and not existing.work_item_id:
                self.conn.execute(
                    "UPDATE agent_episodes SET work_item_id=? WHERE id=?",
                    (work_item_id, episode_id),
                )
                self._commit()
                existing = self.get_episode(episode_id)
            assert existing is not None
            return existing
        environment = self.get_environment_revision(revision.environment_revision_id)
        assert environment is not None
        item = next(
            dict(value)
            for value in revision.definition.get("items") or []
            if str(value.get("id") or "") == suite_item_id
        )
        state = dict(environment.definition.get("initial_state") or {})
        state.update(dict(item.get("initial_state") or {}))
        now = _now()
        self.conn.execute(
            """INSERT INTO agent_episodes
               (id,suite_revision_id,suite_item_id,subject_type,subject_ref,
                subject_hash,seed,status,terminal_reason,metrics_json,
                initial_state_hash,final_state_hash,snapshot_path,trace_hash,
                work_item_id,error,created_at,completed_at,stage,progress_json,
                request_json,resume_cursor_json,cancel_requested,parent_episode_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                episode_id,
                revision_id,
                suite_item_id,
                str(payload.get("subject_type") or "artifact"),
                subject_ref,
                subject_hash,
                seed,
                "queued",
                None,
                "{}",
                _content_hash(state),
                None,
                None,
                None,
                work_item_id,
                None,
                now,
                None,
                "waiting",
                _canonical_json({"current": 0, "total": 1, "percent": 0}),
                _canonical_json(dict(payload)),
                "{}",
                0,
                payload.get("parent_episode_id"),
            ),
        )
        self._commit()
        result = self.get_episode(episode_id)
        assert result is not None
        return result

    def _episode(self, row: Any) -> AgentEpisode:
        return AgentEpisode(
            id=str(row["id"]),
            suite_revision_id=str(row["suite_revision_id"]),
            suite_item_id=str(row["suite_item_id"]),
            subject_type=str(row["subject_type"]),
            subject_ref=str(row["subject_ref"]),
            subject_hash=str(row["subject_hash"]),
            seed=int(row["seed"]),
            status=str(row["status"]),
            terminal_reason=row["terminal_reason"],
            metrics=_loads(row["metrics_json"], {}),
            initial_state_hash=str(row["initial_state_hash"]),
            final_state_hash=row["final_state_hash"],
            snapshot_path=row["snapshot_path"],
            trace_hash=row["trace_hash"],
            work_item_id=row["work_item_id"],
            error=row["error"],
            created_at=str(row["created_at"]),
            completed_at=row["completed_at"],
            stage=str(row["stage"]),
            progress=_loads(row["progress_json"], {}),
            request=_loads(row["request_json"], {}),
            cancel_requested=bool(row["cancel_requested"]),
            parent_episode_id=row["parent_episode_id"],
        )

    def get_episode(self, episode_id: str) -> Optional[AgentEpisode]:
        row = self.conn.execute(
            "SELECT * FROM agent_episodes WHERE id=?", (episode_id,)
        ).fetchone()
        return self._episode(row) if row else None

    def list_episodes(
        self,
        *,
        suite_revision_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        if suite_revision_id:
            rows = self.conn.execute(
                """SELECT * FROM agent_episodes
                   WHERE suite_revision_id=? ORDER BY created_at DESC""",
                (suite_revision_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM agent_episodes ORDER BY created_at DESC"
            ).fetchall()
        return _page(
            [self._episode(row).to_dict() for row in rows],
            limit=limit,
            offset=offset,
        )

    def list_episode_steps(
        self, episode_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT * FROM agent_episode_steps WHERE episode_id=? ORDER BY ordinal",
            (episode_id,),
        ).fetchall()
        values = [
            AgentEpisodeStep(
                episode_id=str(row["episode_id"]),
                ordinal=int(row["ordinal"]),
                observation=_loads(row["observation_json"], {}),
                raw_output=row["raw_output"],
                action=_loads(row["action_json"], {}),
                tool_call=(
                    _loads(row["tool_call_json"], {})
                    if row["tool_call_json"]
                    else None
                ),
                tool_result=(
                    _loads(row["tool_result_json"], {})
                    if row["tool_result_json"]
                    else None
                ),
                state_delta=_loads(row["state_delta_json"], {}),
                state_hash=str(row["state_hash"]),
                verifier=_loads(row["verifier_json"], {}),
                latency_ms=row["latency_ms"],
                error=row["error"],
                created_at=str(row["created_at"]),
            ).to_dict()
            for row in rows
        ]
        return _page(values, limit=limit, offset=offset)

    def replay_episode(self, episode_id: str) -> Dict[str, Any]:
        episode = self.get_episode(episode_id)
        if episode is None:
            raise ValueError(f"unknown agent episode: {episode_id}")
        revision = self.get_episode_suite_revision(episode.suite_revision_id)
        assert revision is not None
        environment = self.get_environment_revision(revision.environment_revision_id)
        assert environment is not None
        item = next(
            value
            for value in revision.definition.get("items") or []
            if str(value.get("id") or "") == episode.suite_item_id
        )
        state = dict(environment.definition.get("initial_state") or {})
        state.update(dict(item.get("initial_state") or {}))
        transitions = dict(environment.definition.get("transitions") or {})
        for step in self.list_episode_steps(episode_id, limit=100_000)["items"]:
            action = dict(step["action"])
            transition = dict(
                transitions.get(str(action.get("name") or action.get("tool") or ""))
                or {}
            )
            state = self._apply_state_delta(
                state, dict(transition.get("state_delta") or {})
            )
            if _content_hash(state) != step["state_hash"]:
                return {
                    "valid": False,
                    "episode_id": episode_id,
                    "failed_ordinal": step["ordinal"],
                    "expected_state_hash": step["state_hash"],
                    "observed_state_hash": _content_hash(state),
                }
        return {
            "valid": _content_hash(state) == episode.final_state_hash,
            "episode_id": episode_id,
            "final_state_hash": _content_hash(state),
            "trace_hash": episode.trace_hash,
        }

    def compare_environment_subjects(
        self,
        *,
        suite_revision_id: str,
        base_subject_hash: str,
        candidate_subject_hash: str,
    ) -> EnvironmentEvaluationComparison:
        rows = self.conn.execute(
            """SELECT * FROM agent_episodes
               WHERE suite_revision_id=? AND subject_hash IN (?,?)
               AND status='completed'""",
            (suite_revision_id, base_subject_hash, candidate_subject_hash),
        ).fetchall()
        by_subject: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            by_subject.setdefault(str(row["subject_hash"]), {})[
                str(row["suite_item_id"])
            ] = self._episode(row)
        base = by_subject.get(base_subject_hash, {})
        candidate = by_subject.get(candidate_subject_hash, {})
        shared = sorted(set(base) & set(candidate))
        counts = {
            "shared": len(shared),
            "improved": 0,
            "regressed": 0,
            "unchanged": 0,
        }
        metric_names = sorted(
            {
                key
                for item in shared
                for key in set(base[item].metrics) | set(candidate[item].metrics)
                if isinstance(base[item].metrics.get(key), (int, float))
                and isinstance(candidate[item].metrics.get(key), (int, float))
            }
        )
        deltas: Dict[str, Optional[float]] = {}
        for metric in metric_names:
            values = [
                float(candidate[item].metrics[metric])
                - float(base[item].metrics[metric])
                for item in shared
            ]
            deltas[metric] = sum(values) / len(values) if values else None
        for item in shared:
            delta = float(candidate[item].metrics.get("task_success", 0)) - float(
                base[item].metrics.get("task_success", 0)
            )
            if delta > 0:
                counts["improved"] += 1
            elif delta < 0:
                counts["regressed"] += 1
            else:
                counts["unchanged"] += 1
        return EnvironmentEvaluationComparison(
            suite_revision_id=suite_revision_id,
            base_subject_hash=base_subject_hash,
            candidate_subject_hash=candidate_subject_hash,
            counts=counts,
            metric_deltas=deltas,
            compatible=bool(shared),
            reasons=[] if shared else ["no shared completed suite item identities"],
        )

    def get_trajectory_set(self, trajectory_set_id: str) -> Optional[TrajectorySet]:
        row = self.conn.execute(
            "SELECT * FROM trajectory_sets WHERE id=?",
            (trajectory_set_id,),
        ).fetchone()
        if row is None:
            return None
        return TrajectorySet(
            id=str(row["id"]),
            name=str(row["name"]),
            latest_revision_id=row["latest_revision_id"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            status=str(row["status"]),
            stage=str(row["stage"]),
            progress=_loads(row["progress_json"], {}),
            work_item_id=row["work_item_id"],
            error=row["error"],
        )

    def prepare_trajectory_set(
        self,
        payload: Mapping[str, Any],
        *,
        work_item_id: Optional[str] = None,
    ) -> TrajectorySet:
        episode_ids = [str(value) for value in payload.get("episode_ids") or []]
        if not episode_ids:
            raise ValueError("trajectory publication requires episode_ids")
        for episode_id in episode_ids:
            episode = self.get_episode(episode_id)
            if episode is None or episode.status != "completed":
                raise ValueError(f"trajectory episode is unavailable: {episode_id}")
        trajectory_set_id = str(
            payload.get("trajectory_set_id")
            or _stable_id(
                "trajectory-set",
                {
                    "episodes": episode_ids,
                    "adapter": payload.get("output_adapter") or "tool_sft",
                },
            )
        )
        existing = self.get_trajectory_set(trajectory_set_id)
        if existing is None:
            now = _now()
            self.conn.execute(
                """INSERT INTO trajectory_sets
                   (id,name,latest_revision_id,status,stage,progress_json,
                    request_json,cancel_requested,work_item_id,error,
                    created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trajectory_set_id,
                    str(payload.get("name") or "Reviewed trajectories").strip(),
                    None,
                    "queued",
                    "waiting",
                    _canonical_json({"current": 0, "total": len(episode_ids)}),
                    _canonical_json(dict(payload)),
                    0,
                    work_item_id,
                    None,
                    now,
                    now,
                ),
            )
            self._commit()
            existing = self.get_trajectory_set(trajectory_set_id)
        elif work_item_id and not existing.work_item_id:
            self.conn.execute(
                "UPDATE trajectory_sets SET work_item_id=?,updated_at=? WHERE id=?",
                (work_item_id, _now(), trajectory_set_id),
            )
            self._commit()
            existing = self.get_trajectory_set(trajectory_set_id)
        assert existing is not None
        return existing

    def publish_trajectory_set(
        self, payload: Mapping[str, Any]
    ) -> TrajectorySetRevision:
        episode_ids = [str(value) for value in payload.get("episode_ids") or []]
        if not episode_ids:
            raise ValueError("trajectory publication requires episode_ids")
        output_adapter = str(payload.get("output_adapter") or "tool_sft")
        if output_adapter not in {
            "tool_sft",
            "preference",
            "reasoning",
            "tool_correction",
            "rlvr",
        }:
            raise ValueError("unsupported trajectory output adapter")
        trajectory_set_id = str(payload.get("trajectory_set_id") or uuid.uuid4().hex)
        existing = self.conn.execute(
            "SELECT * FROM trajectory_sets WHERE id=?", (trajectory_set_id,)
        ).fetchone()
        if existing is not None and bool(existing["cancel_requested"]):
            raise RuntimeError("trajectory publication cancellation requested")
        now = _now()
        if existing is None:
            name = str(payload.get("name") or "Reviewed trajectories").strip()
            self.conn.execute(
                """INSERT INTO trajectory_sets
                   (id,name,latest_revision_id,created_at,updated_at)
                   VALUES (?,?,?,?,?)""",
                (trajectory_set_id, name, None, now, now),
            )
        previous = self.conn.execute(
            """SELECT COALESCE(MAX(revision_number),0) AS value
               FROM trajectory_set_revisions WHERE trajectory_set_id=?""",
            (trajectory_set_id,),
        ).fetchone()
        revision_number = int(previous["value"]) + 1
        records = []
        for episode_id in episode_ids:
            current = self.conn.execute(
                "SELECT cancel_requested FROM trajectory_sets WHERE id=?",
                (trajectory_set_id,),
            ).fetchone()
            if current is not None and bool(current["cancel_requested"]):
                raise RuntimeError("trajectory publication cancellation requested")
            episode = self.get_episode(episode_id)
            if episode is None or episode.status != "completed":
                raise ValueError(f"trajectory episode is unavailable: {episode_id}")
            steps = self.list_episode_steps(episode_id, limit=100_000)["items"]
            if output_adapter == "preference":
                record = {
                    "prompt": f"Complete environment task {episode.suite_item_id}",
                    "chosen": steps,
                    "rejected": [],
                    "metadata": {"episode_id": episode_id},
                }
            elif output_adapter == "rlvr":
                record = {
                    "prompt": f"Complete environment task {episode.suite_item_id}",
                    "reference_answer": episode.metrics,
                    "metadata": {"episode_id": episode_id, "trajectory": steps},
                }
            else:
                record = {
                    "messages": [
                        {"role": "system", "content": "Complete the environment task."},
                        {
                            "role": "user",
                            "content": f"Task {episode.suite_item_id}",
                        },
                        {"role": "assistant", "content": steps},
                    ],
                    "metadata": {"episode_id": episode_id},
                }
            records.append(record)
        content_hash = _content_hash(records)
        revision_id = _stable_id(
            "trajectory-revision",
            {
                "trajectory_set_id": trajectory_set_id,
                "content_hash": content_hash,
                "output_adapter": output_adapter,
            },
        )
        storage_path, _ = _atomic_bundle(
            self.trajectory_root / trajectory_set_id / revision_id,
            {
                "records.jsonl": "".join(
                    json.dumps(record, sort_keys=True) + "\n"
                    for record in records
                ),
                "provenance.json": json.dumps(
                    {
                        "episode_ids": episode_ids,
                        "output_adapter": output_adapter,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            },
        )
        self.conn.execute(
            """INSERT OR IGNORE INTO trajectory_set_revisions
               (id,trajectory_set_id,revision_number,content_hash,storage_path,
                row_count,provenance_json,created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                revision_id,
                trajectory_set_id,
                revision_number,
                content_hash,
                str(storage_path),
                len(records),
                _canonical_json(
                    {"episode_ids": episode_ids, "output_adapter": output_adapter}
                ),
                now,
            ),
        )
        for ordinal, (episode_id, record) in enumerate(zip(episode_ids, records)):
            self.conn.execute(
                """INSERT OR IGNORE INTO trajectory_set_items
                   (revision_id,ordinal,episode_id,output_adapter,record_json,record_hash)
                   VALUES (?,?,?,?,?,?)""",
                (
                    revision_id,
                    ordinal,
                    episode_id,
                    output_adapter,
                    _canonical_json(record),
                    _content_hash(record),
                ),
            )
        self.conn.execute(
            """UPDATE trajectory_sets
               SET latest_revision_id=?,status='completed',stage='completed',
                   progress_json=?,cancel_requested=0,error=NULL,updated_at=?
               WHERE id=?""",
            (
                revision_id,
                _canonical_json(
                    {"current": len(records), "total": len(records), "percent": 100}
                ),
                now,
                trajectory_set_id,
            ),
        )
        self._commit()
        return TrajectorySetRevision(
            id=revision_id,
            trajectory_set_id=trajectory_set_id,
            revision_number=revision_number,
            content_hash=content_hash,
            storage_path=str(storage_path),
            row_count=len(records),
            provenance={"episode_ids": episode_ids, "output_adapter": output_adapter},
            created_at=now,
        )
