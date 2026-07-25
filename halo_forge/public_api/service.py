"""Shared public product service built on top of internal halo-forge services."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import re
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional

from halo_forge.huggingface_access import GATED_MODEL_ACTION, HuggingFaceAccessManager
from halo_forge.training_recovery import build_recovery_guidance
from ui.services.ops_readiness_service import OpsReadinessService, get_ops_readiness_service
from ui.services.quickstart_presets import list_quickstart_presets
from ui.services.results_service import ResultsService, TrainingRunSummary, get_results_service
from ui.services.training_presentation import build_launch_presentation
from ui.services.training_service import TrainingLaunchPreflight, TrainingService
from ui.state import AppState, JobState, state as default_state

from .serve_manager import ManagedServeProcess, ServeStartRequest
from .views import (
    ActiveRunRowView,
    AttentionItemView,
    DashboardSummaryView,
    DocsCapabilitySummaryView,
    ModalityReadinessView,
    ProductUserSummaryView,
    PublicActionView,
    ResearchSectionView,
    RunMetricsSummaryView,
    RunFailureSummaryView,
    TrainingMetricPointView,
    TrainingLaunchPreflightView,
    TrainingRecoveryView,
    TrainingRunDetailView,
    TrainingRunListItemView,
    TrainingRunLiveView,
    TrainingStageView,
    build_user_summary,
    to_dict,
)

TRAINING_MODALITIES = (
    "sft",
    "cpt",
    "raft",
    "dpo",
    "orpo",
    "rm",
    "grpo",
    "vlm",
    "audio",
    "reasoning",
    "agentic",
    "classify",
    "embed",
    "rerank",
)
DEFAULT_RUN_ROOT_ENV = "HALO_FORGE_RUN_ROOT"
GATED_MODEL_MESSAGE = (
    "This model requires Hugging Face access. Connect Hugging Face, accept the model license, "
    "or choose an open model."
)

PUBLIC_TRAIN_ALLOWED_FIELDS: dict[str, set[str]] = {
    "sft": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "accelerator",
        "epochs",
        "max_samples",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_steps",
        "max_sequence_length",
        "gradient_checkpointing",
        "no_caffeinate",
    },
    "cpt": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "output_root",
        "accelerator",
        "adaptation",
        "model_revision",
        "model_hash",
        "tokenizer_hash",
        "training_artifact_hash",
        "expected_packing_plan_hash",
        "max_sequence_length",
        "packing",
        "budget_mode",
        "target_tokens",
        "corpus_passes",
        "effective_batch_size",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_steps",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "use_dora",
        "use_rslora",
        "init_lora_weights",
        "load_in_4bit",
        "optim",
        "no_caffeinate",
    },
    "raft": {
        "mode",
        "model",
        "prompts",
        "output_dir",
        "accelerator",
        "verifier",
        "cycles",
        "limit",
        "max_prompts",
        "samples_per_prompt",
        "keep_percent",
        "reward_threshold",
        "temperature",
        "no_caffeinate",
    },
    "dpo": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "accelerator",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_samples",
        "beta",
        "loss_type",
        "reference_free",
        "no_caffeinate",
    },
    "orpo": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "accelerator",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_samples",
        "beta",
        "no_caffeinate",
    },
    "rm": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "accelerator",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_samples",
        "no_caffeinate",
    },
    "grpo": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "accelerator",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "max_samples",
        "beta",
        "reference_free",
        "verifier",
        "num_generations",
        "epsilon",
        "temperature",
        "reward_threshold",
        "max_steps",
        "no_caffeinate",
    },
    "vlm": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "cycles",
        "limit",
        "task",
        "samples_per_prompt",
        "keep_percent",
        "reward_threshold",
        "temperature",
        "allow_prototype_train",
        "no_caffeinate",
    },
    "audio": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "cycles",
        "limit",
        "samples_per_prompt",
        "keep_percent",
        "reward_threshold",
        "temperature",
        "task",
        "allow_prototype_train",
        "no_caffeinate",
    },
    "reasoning": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "cycles",
        "limit",
        "keep_percent",
        "temperature",
        "learning_rate",
        "allow_prototype_train",
        "no_caffeinate",
    },
    "agentic": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "cycles",
        "limit",
        "keep_percent",
        "temperature",
        "learning_rate",
        "allow_prototype_train",
        "no_caffeinate",
    },
    "classify": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "epochs",
        "batch_size",
        "learning_rate",
        "max_samples",
        "multi_label",
        "label_schema_revision_id",
        "no_caffeinate",
    },
    "embed": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "epochs",
        "batch_size",
        "learning_rate",
        "max_samples",
        "retrieval_corpus_id",
        "no_caffeinate",
    },
    "rerank": {
        "mode",
        "model",
        "dataset",
        "output_dir",
        "epochs",
        "batch_size",
        "learning_rate",
        "max_samples",
        "retrieval_corpus_id",
        "no_caffeinate",
    },
}

for _training_fields in PUBLIC_TRAIN_ALLOWED_FIELDS.values():
    _training_fields.update(
        {
            "seed",
            "dataset_version_id",
            "dataset_split",
            "dataset_bindings",
            "dataset_adapter_id",
            "tokenizer_revision",
            "validation_fraction",
            "validation_file",
            "training_artifact_id",
            "training_artifact_metadata",
            "parent_run_id",
            "run_id",
            "output_root",
            "verifier_profile_revision_id",
            "verifier_binding",
            "reward_system_revision_id",
            "reward_audit_protocol_revision_id",
            "reward_integrity_profile_revision_id",
            "development_suite_revision_id",
            "reward_audit_boundaries",
            "reward_integrity_binding",
            # Immutable lineage for an operator-reviewed reward-audit fork.
            # These values are re-resolved from the audit before preflight and
            # launch; accepting the keys here does not make client paths
            # authoritative.
            "source_reward_integrity_audit_id",
            "source_reward_integrity_decision_id",
            "fork_checkpoint_hash",
            "fork_checkpoint_path",
            "fork_checkpoint_occurrence_id",
            "fork_checkpoint_snapshot_path",
            "fork_boundary_unit",
            "fork_boundary_value",
            "fork_resume_mode",
            # Guided own-data proof/full-run lineage. These fields are
            # evidence only; trainer argv rendering never treats them as
            # arbitrary command-line input.
            "proof_run",
            "scenario_revision_id",
            "field_mapping_plan",
            "dataset_preparation_recipe",
            "proof_sample_identity",
            "proof_max_samples",
            "proof_parent_run_id",
            "full_run_from_proof",
            "full_run_reason",
            "outcome_assessment_id",
            "outcome_override_reason",
            "study_id",
            "study_protocol_revision_id",
            "study_arm_id",
            "study_assignment_id",
            "study_assignment_ids_by_seed",
            "study_factor_values",
            "study_contrast_ids",
            "study_deviation_ids",
            "label_schema_revision_id",
            "retrieval_corpus_id",
            "environment_revision_id",
            "episode_suite_revision_id",
            "trajectory_revision_id",
            # V17 captures the setup snapshot used for preflight and launch.
            # These fields are resolved server-side and are replay evidence,
            # never trainer arguments.
            "workstation_readiness_id",
            "workstation_readiness_hash",
            "distribution_capability",
            "dataset_repair_revision_id",
            # V18 immutable guided-plan and measured-capacity identity.
            "training_plan_revision_id",
            "training_plan_model_id",
            "training_capacity_check_id",
            "model_preparation_id",
            "training_compute_shape_hash",
            "training_capacity_adjustment",
            "training_plan_decision_id",
            "training_plan_content_hash",
            "training_plan_runtime_hash",
            "runtime_profile_revision_id",
            "training_plan_recommendation_reasons",
            "training_plan_forecast",
            "resolved_model_commit",
            "model_preparation_manifest_hash",
        }
    )

PUBLIC_TRAIN_REQUIRED_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "sft": ("model", "dataset", "output_dir"),
    "cpt": ("model", "dataset", "output_dir"),
    "raft": ("model", "prompts", "output_dir"),
    "dpo": ("model", "dataset", "output_dir"),
    "orpo": ("model", "dataset", "output_dir"),
    "rm": ("model", "dataset", "output_dir"),
    "grpo": ("model", "dataset", "output_dir"),
    "vlm": ("model", "dataset", "output_dir"),
    "audio": ("model", "dataset", "output_dir", "task"),
    "reasoning": ("model", "dataset", "output_dir"),
    "agentic": ("model", "dataset", "output_dir"),
    "classify": ("model", "dataset", "output_dir"),
    "embed": ("model", "dataset", "output_dir"),
    "rerank": ("model", "dataset", "output_dir"),
}


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
        serve_manager: ManagedServeProcess | None = None,
        huggingface_manager: HuggingFaceAccessManager | None = None,
        base_path: Path | None = None,
        database: Any | None = None,
        dataset_lab: Any | None = None,
        dataset_storage_root: Path | None = None,
        evaluation_lab: Any | None = None,
        evaluation_storage_root: Path | None = None,
        experiment_operations: Any | None = None,
        artifact_store: Any | None = None,
        artifact_studio: Any | None = None,
        artifact_storage_root: Path | None = None,
        adaptive_lab: Any | None = None,
        evidence_storage_root: Path | None = None,
        workstation_scheduler: Any | None = None,
        review_lab: Any | None = None,
        review_storage_root: Path | None = None,
        review_embedding_engine: Any | None = None,
        verifier_lab: Any | None = None,
        verifier_calibration_storage_root: Path | None = None,
        reward_integrity: Any | None = None,
        reward_integrity_storage_root: Path | None = None,
        guided_own_data: Any | None = None,
        dataset_import_root: Path | None = None,
        corpus_extraction: Any | None = None,
        corpus_storage_root: Path | None = None,
        future_lab: Any | None = None,
        future_lab_storage_root: Path | None = None,
        product_lab: Any | None = None,
        product_lab_storage_root: Path | None = None,
        training_plan: Any | None = None,
        training_plan_storage_root: Path | None = None,
        managed_runtime: Any | None = None,
        managed_runtime_storage_root: Path | None = None,
        training_path_certification: Any | None = None,
        certification_storage_root: Path | None = None,
    ) -> None:
        self.app_state = app_state or default_state
        self.base_path = (base_path or Path.cwd()).resolve()
        self.results_service = results_service or get_results_service()
        self.readiness_service = readiness_service or get_ops_readiness_service()
        self.training_service = training_service or TrainingService(self.app_state)
        self.serve_manager = serve_manager or ManagedServeProcess(base_path=self.base_path)
        self.huggingface_manager = huggingface_manager or HuggingFaceAccessManager()
        self._database = database
        self._dataset_lab = dataset_lab
        self._evaluation_lab = evaluation_lab
        self._experiment_operations = experiment_operations
        self._artifact_store = artifact_store
        self._artifact_studio = artifact_studio
        self._adaptive_lab = adaptive_lab
        self._workstation_scheduler = workstation_scheduler
        self._review_lab = review_lab
        self._review_embedding_engine = review_embedding_engine
        self._verifier_lab = verifier_lab
        self._reward_integrity = reward_integrity
        self._guided_own_data = guided_own_data
        self._corpus_extraction = corpus_extraction
        self._future_lab = future_lab
        self._product_lab = product_lab
        self._training_plan = training_plan
        self._managed_runtime = managed_runtime
        self._training_path_certification = training_path_certification
        self._v4_catalog = None
        self._serving_lease_holder = f"public-api-serving:{self.base_path}"
        self._training_artifact_preparation_lock = threading.RLock()
        self._checkpoint_gate_review_lock = threading.RLock()
        configured_dataset_root = os.environ.get("HALOFORGE_DATASET_ROOT")
        self.dataset_storage_root = (
            dataset_storage_root
            or (Path(configured_dataset_root).expanduser() if configured_dataset_root else None)
            or Path.home() / ".halo-forge" / "datasets"
        ).resolve()
        self.dataset_import_root = (
            dataset_import_root or self.dataset_storage_root.parent / "imports"
        ).resolve()
        self.corpus_storage_root = (
            corpus_storage_root or self.dataset_storage_root.parent / "corpus"
        ).resolve()
        self.future_lab_storage_root = (
            future_lab_storage_root or self.dataset_storage_root.parent
        ).resolve()
        self.product_lab_storage_root = (
            product_lab_storage_root or self.dataset_storage_root.parent
        ).resolve()
        self.training_plan_storage_root = (
            training_plan_storage_root or self.dataset_storage_root.parent
        ).resolve()
        configured_runtime_root = os.environ.get("HALOFORGE_RUNTIME_ROOT")
        self.managed_runtime_storage_root = (
            managed_runtime_storage_root
            or (Path(configured_runtime_root).expanduser() if configured_runtime_root else None)
            or self.training_plan_storage_root / "runtimes"
        ).resolve()
        self.certification_storage_root = (
            certification_storage_root
            or self.training_plan_storage_root / "certifications"
        ).resolve()
        configured_evaluation_root = os.environ.get("HALOFORGE_EVALUATION_ROOT")
        self.evaluation_storage_root = (
            evaluation_storage_root
            or (
                Path(configured_evaluation_root).expanduser()
                if configured_evaluation_root
                else None
            )
            or Path.home() / ".halo-forge" / "evaluations"
        ).resolve()
        configured_artifact_root = os.environ.get("HALOFORGE_ARTIFACT_ROOT")
        self.artifact_storage_root = (
            artifact_storage_root
            or (Path(configured_artifact_root).expanduser() if configured_artifact_root else None)
            or Path.home() / ".halo-forge" / "artifacts"
        ).resolve()
        configured_evidence_root = os.environ.get("HALOFORGE_EVIDENCE_ROOT")
        self.evidence_storage_root = (
            evidence_storage_root
            or (Path(configured_evidence_root).expanduser() if configured_evidence_root else None)
            or Path.home() / ".halo-forge" / "research" / "evidence"
        ).resolve()
        configured_review_root = os.environ.get("HALOFORGE_REVIEW_ROOT")
        self.review_storage_root = (
            review_storage_root
            or (Path(configured_review_root).expanduser() if configured_review_root else None)
            or Path.home() / ".halo-forge" / "reviews"
        ).resolve()
        self.verifier_calibration_storage_root = (
            verifier_calibration_storage_root
            or self.evaluation_storage_root / "verifier-calibrations"
        ).resolve()
        self.reward_integrity_storage_root = (
            reward_integrity_storage_root
            or self.evaluation_storage_root / "reward-audits"
        ).resolve()

    def _scheduler(self):
        if self._workstation_scheduler is None:
            from halo_forge.workstation_jobs import WorkstationScheduler

            self._workstation_scheduler = WorkstationScheduler(self._dataset_database())
        return self._workstation_scheduler

    def _managed_runtime_engine(self):
        if self._managed_runtime is None:
            from halo_forge.managed_runtime import ManagedRuntimeService

            self._managed_runtime = ManagedRuntimeService(
                self._dataset_database(),
                root=self.managed_runtime_storage_root,
                scheduler=self._scheduler(),
                source_root=self.base_path,
            )
        return self._managed_runtime

    def _training_path_engine(self):
        if self._training_path_certification is None:
            from halo_forge.training_path_certification import (
                TrainingPathCertificationService,
            )

            self._training_path_certification = TrainingPathCertificationService(
                self._dataset_database(),
                root=self.certification_storage_root,
                runtime_service=self._managed_runtime_engine(),
                scheduler=self._scheduler(),
                source_root=self.base_path,
            )
        return self._training_path_certification

    def _dataset_database(self):
        if self._database is None:
            from halo_forge.run_db import get_database

            self._database = get_database()
        return self._database

    def _dataset_engine(self):
        """Lazily load the Dataset Lab facade so the rest of the API stays usable
        when optional dataset dependencies are not installed."""
        if self._dataset_lab is None:
            from halo_forge.data_lab import DatasetLab

            self._dataset_lab = DatasetLab(
                self.dataset_storage_root,
                scheduler=self._scheduler(),
                database=self._dataset_database(),
                verifier_service=self._verifier_engine(),
            )
        return self._dataset_lab

    def _guided_data_engine(self):
        if self._guided_own_data is None:
            from halo_forge.own_data import GuidedOwnDataService

            self._guided_own_data = GuidedOwnDataService(
                self._dataset_database(),
                datasets_root=self.dataset_storage_root,
                imports_root=self.dataset_import_root,
                scheduler=self._scheduler(),
            )
        return self._guided_own_data

    def _corpus_extraction_engine(self):
        if self._corpus_extraction is None:
            from halo_forge.corpus_lab import CorpusExtractionService

            self._corpus_extraction = CorpusExtractionService(
                self._dataset_database(),
                root=self.corpus_storage_root,
                scheduler=self._scheduler(),
            )
        return self._corpus_extraction

    def _future_lab_engine(self):
        """Return the shared V11-V15 outcome, study, grounding, task, and environment service."""

        if self._future_lab is None:
            from halo_forge.lab_v11_v15 import FutureLabService

            self._future_lab = FutureLabService(
                self._dataset_database(),
                root=self.future_lab_storage_root,
                scheduler=self._scheduler(),
            )
        return self._future_lab

    def _product_lab_engine(self):
        """Return the shared V17 readiness, repair, and support service."""

        if self._product_lab is None:
            from halo_forge.product_lab import ProductLabService

            self._product_lab = ProductLabService(
                self._dataset_database(),
                root=self.product_lab_storage_root,
                scheduler=self._scheduler(),
            )
        return self._product_lab

    def _training_plan_engine(self):
        """Return the V18 deterministic plan and capacity service."""

        if self._training_plan is None:
            from halo_forge.training_plan import TrainingPlanService

            self._training_plan = TrainingPlanService(
                self._dataset_database(),
                root=self.training_plan_storage_root,
                scheduler=self._scheduler(),
            )
        return self._training_plan

    def _evaluation_engine(self):
        if self._evaluation_lab is None:
            from halo_forge.evaluation_lab import EvaluationLabService
            from halo_forge.workstation_jobs import WorkstationScheduler

            self._evaluation_lab = EvaluationLabService(
                self._dataset_database(),
                self.evaluation_storage_root,
                scheduler=WorkstationScheduler(self._dataset_database()),
            )
        return self._evaluation_lab

    def _verifier_engine(self):
        """Return the transport-neutral verifier reliability facade."""

        if self._verifier_lab is None:
            from halo_forge.verifier_lab import VerifierLabService

            self._verifier_lab = VerifierLabService(
                self._dataset_database(),
                root=self.verifier_calibration_storage_root,
                scheduler=self._scheduler(),
            )
        return self._verifier_lab

    def _reward_integrity_engine(self):
        """Return the shared Reward Integrity service used by every surface."""

        if self._reward_integrity is None:
            from halo_forge.reward_integrity import RewardIntegrityService

            self._reward_integrity = RewardIntegrityService(
                self._dataset_database(),
                root=self.reward_integrity_storage_root,
            )
        return self._reward_integrity

    def _experiment_engine(self):
        """Return the transport-neutral repeat/sweep orchestration facade."""
        if self._experiment_operations is None:
            from halo_forge.orchestration import ExperimentOrchestrationService

            self._experiment_operations = ExperimentOrchestrationService(
                self._dataset_database(), scheduler=self._scheduler()
            )
        return self._experiment_operations

    def _v4_engine(self):
        if self._v4_catalog is None:
            from halo_forge.run_db import LabV4Catalog

            self._v4_catalog = LabV4Catalog(self._dataset_database())
        return self._v4_catalog

    def _artifact_library(self):
        if self._artifact_store is None:
            from halo_forge.artifact_lab import ArtifactStore

            self._artifact_store = ArtifactStore(self.artifact_storage_root)
        return self._artifact_store

    def _artifact_studio_engine(self):
        if self._artifact_studio is None:
            try:
                from halo_forge.artifact_studio import (
                    ArtifactStudioService,
                    SubprocessServingStarter,
                )
            except ImportError:
                return None
            from halo_forge.workstation_jobs import WorkstationScheduler
            from halo_forge.qualification_lab import EvaluationQualificationExecutor

            catalog = self._v4_engine()

            self._artifact_studio = ArtifactStudioService(
                store=self._artifact_library(),
                catalog=catalog,
                scheduler=WorkstationScheduler(self._dataset_database()),
                qualification_executor=EvaluationQualificationExecutor(
                    self._dataset_database(), catalog, self._evaluation_engine()
                ),
                serving_starter=SubprocessServingStarter(
                    catalog,
                    base_path=self.base_path,
                ),
            )
        return self._artifact_studio

    def _adaptive_engine(self):
        """Return the checkpoint-policy and research-evidence service."""

        if self._adaptive_lab is None:
            from halo_forge.adaptive_lab import AdaptiveLabService

            self._adaptive_lab = AdaptiveLabService(
                self._dataset_database(),
                evidence_root=self.evidence_storage_root,
            )
        return self._adaptive_lab

    def _review_engine(self):
        """Return the transport-neutral Human Feedback and Active Data facade."""

        if self._review_lab is None:
            from halo_forge.review_lab import ReviewLabService

            self._review_lab = ReviewLabService(
                self._dataset_database(),
                root=self.review_storage_root,
            )
        return self._review_lab

    def _active_backend_name(self) -> str:
        """Return the active accelerator-kind name for cost / display use.

        Cached per service instance after the first probe — backend
        detection is cheap but not free, and the run-detail endpoint
        fires it on every request.
        """
        cached = getattr(self, "_cached_backend_name", None)
        if cached:
            return cached
        try:
            from halo_forge.backend import get_backend

            name = get_backend().name
        except Exception:
            name = "unknown"
        self._cached_backend_name = name
        return name

    # ----- backend-driven structured form descriptors -------------------

    def list_spec_descriptors(self, kind: str) -> Dict[str, Any]:
        from halo_forge.spec_registry import serialized_spec_descriptors

        items = serialized_spec_descriptors(kind)
        return {
            "items": items,
            "total": len(items),
            "limit": len(items),
            "offset": 0,
        }

    def validate_spec_descriptor(
        self, kind: str, descriptor_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        from halo_forge.spec_registry import validate_structured_spec

        return validate_structured_spec(kind, descriptor_id, payload)

    def get_backend_info(self) -> Dict[str, Any]:
        """Return the active compute backend and its capabilities.

        Used by the frontend to render "Running on Apple Silicon (MPS)" /
        "Running on AMD ROCm" badges and to gate UI affordances (e.g. hide
        4-bit quantization toggles on backends that can't honor them).
        """
        from halo_forge.backend import get_backend
        from dataclasses import asdict

        backend = get_backend()
        chip = None
        try:
            from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry

            parsed_chip = AppleSiliconTelemetry._detect_chip_info(
                AppleSiliconTelemetry._detect_device_name()
            )
            chip = parsed_chip.to_dict() if parsed_chip is not None else None
        except Exception:
            chip = None
        mlx_readiness = self._mlx_readiness_snapshot()
        if (
            not chip or str(chip.get("brand") or "").lower() in {"arm", "apple silicon"}
        ) and mlx_readiness.get("chip"):
            chip = mlx_readiness.get("chip")
        return {
            "name": backend.name,
            "device": backend.device(),
            "chip": chip,
            "capabilities": asdict(backend.capabilities),
            "training_defaults": backend.training_defaults(),
            "inference_defaults": backend.inference_defaults(),
            "mlx_readiness": mlx_readiness,
        }

    # ----- Guided own-data training ------------------------------------

    def list_interface_capabilities(self) -> Dict[str, Any]:
        return self._guided_data_engine().list_capabilities(
            backend_name=self._active_backend_name()
        )

    def list_training_scenarios(
        self,
        *,
        include_unavailable: bool = False,
        modality: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._guided_data_engine().list_scenarios(
            backend_name=self._active_backend_name(),
            include_unavailable=include_unavailable,
            modality=modality,
            limit=limit,
            offset=offset,
        )

    def get_training_scenario(self, scenario_id: str) -> Dict[str, Any]:
        return self._guided_data_engine().get_scenario(
            scenario_id, backend_name=self._active_backend_name()
        )

    def list_training_scenario_examples(self, scenario_id: str) -> Dict[str, Any]:
        return self._guided_data_engine().list_examples(scenario_id)

    def list_guided_training_examples(self) -> Dict[str, Any]:
        return self._guided_data_engine().list_guided_examples(
            backend_name=self._active_backend_name()
        )

    def advise_training_scenario(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._guided_data_engine().scenario_advice(
            payload, backend_name=self._active_backend_name()
        )

    def get_training_scenario_template(
        self, scenario_id: str, *, example_id: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._guided_data_engine().scenario_template(scenario_id, example_id)

    def create_dataset_import(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._guided_data_engine().create_import(payload)

    def get_huggingface_dataset_options(
        self, *, repo_id: str, revision: str
    ) -> Dict[str, Any]:
        return self._guided_data_engine().huggingface_options(repo_id, revision)

    def list_dataset_imports(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._guided_data_engine().list_imports(
            status=status, limit=limit, offset=offset
        )

    def get_dataset_import(self, import_id: str) -> Optional[Dict[str, Any]]:
        return self._guided_data_engine().get_import(import_id)

    def cancel_dataset_import(self, import_id: str) -> Dict[str, Any]:
        return self._guided_data_engine().cancel_import(import_id)

    def retry_dataset_import(self, import_id: str) -> Dict[str, Any]:
        return self._guided_data_engine().retry_import(import_id)

    def create_dataset_import_file(
        self, import_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._guided_data_engine().create_import_file(import_id, payload)

    def upload_dataset_import_chunk(
        self,
        import_id: str,
        file_id: str,
        content: bytes,
        *,
        start: int,
        end: int,
        total: int,
        chunk_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._guided_data_engine().upload_chunk(
            import_id,
            file_id,
            content,
            start=start,
            end=end,
            total=total,
            chunk_sha256=chunk_sha256,
        )

    def inspect_dataset_import(
        self, import_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        values = payload or {}
        return self._guided_data_engine().request_inspection(
            import_id,
            scenario_revision_id=self._optional_str(values.get("scenario_revision_id")),
            force=bool(values.get("force", False)),
        )

    def get_dataset_source_inspection(
        self, inspection_id: str
    ) -> Optional[Dict[str, Any]]:
        return self._guided_data_engine().get_inspection(inspection_id)

    def cancel_dataset_source_inspection(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        return self._guided_data_engine().cancel_inspection(inspection_id)

    def retry_dataset_source_inspection(self, inspection_id: str) -> Dict[str, Any]:
        return self._guided_data_engine().retry_inspection(inspection_id)

    def preview_dataset_mapping(
        self, inspection_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._guided_data_engine().mapping_preview(inspection_id, payload)

    def preview_dataset_semantics(
        self, inspection_id: str, payload: Dict[str, Any], *, limit: int = 50
    ) -> Dict[str, Any]:
        return self._guided_data_engine().semantic_preview(
            inspection_id, payload, limit=limit
        )

    def get_dataset_inspection_readiness(
        self, inspection_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._guided_data_engine().readiness_report(
            inspection_id, payload
        )

    def preview_dataset_preparation(
        self, inspection_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._guided_data_engine().preparation_preview(inspection_id, payload)

    def register_inspected_dataset(
        self, inspection_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        engine = self._guided_data_engine()
        registration = engine.registration_request(inspection_id, payload)
        # Dataset publication is backend-independent. A workstation may
        # inspect and prepare data before optional trainer dependencies are
        # installed; capability filtering and Train preflight remain the
        # authoritative launch gates.
        engine.registry.get(registration["scenario_revision_id"])
        session = self._dataset_database().get_dataset_import(registration["import_id"])
        if session is None:
            raise KeyError(registration["import_id"])
        if session.published_dataset_id and session.published_source_id:
            dataset = self.get_dataset(session.published_dataset_id)
            source = self.get_dataset_source(session.published_source_id)
            return {
                "registration": {
                    "id": session.id,
                    "status": "completed",
                    "dataset_id": session.published_dataset_id,
                    "source_id": session.published_source_id,
                },
                "import": engine.import_view(session),
                "dataset": dataset,
                "source": source,
                "preparation_plan": registration["preparation_plan"],
                "work_item_id": session.work_item_id,
                "reused": True,
            }

        request_hash = hashlib.sha256(
            json.dumps(
                {
                    "inspection_id": inspection_id,
                    "registration": registration,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if session.work_item_id:
            existing_work = self._dataset_database().get_work_item(session.work_item_id)
            if existing_work is not None and existing_work.status in {
                "queued",
                "blocked",
                "running",
            }:
                if existing_work.launch_spec.get("request_hash") != request_hash:
                    raise ValueError(
                        "this import already has a different registration in progress"
                    )
                return {
                    "registration": {
                        "id": session.id,
                        "status": existing_work.status,
                        "dataset_id": existing_work.launch_spec.get("dataset_id"),
                        "source_id": existing_work.launch_spec.get("source_id"),
                    },
                    "import": engine.import_view(session),
                    "dataset": None,
                    "source": None,
                    "preparation_plan": registration["preparation_plan"],
                    "work_item_id": existing_work.id,
                    "reused": True,
                }

        dataset_id = uuid.uuid4().hex
        source_id = uuid.uuid4().hex
        resource_requirements: Optional[Dict[str, Any]] = None
        if registration.get("source_kind") in {"upload", "huggingface"}:
            resource_requirements = {
                "output_path": str(self.dataset_storage_root),
                "projected_disk_bytes": max(
                    0, int(registration.get("source_size_bytes") or 0)
                ),
                "capacity_preflight": True,
            }
        capacity_override_reason = str(
            registration.get("capacity_override_reason")
            or dict(session.metadata.get("capacity") or {}).get("override_reason")
            or ""
        ).strip()
        if capacity_override_reason and resource_requirements is not None:
            resource_requirements["capacity_override_reason"] = capacity_override_reason
        work = self._scheduler().enqueue(
            kind="dataset_registration",
            launch_spec={
                "handler": "own_data.register",
                "inspection_id": inspection_id,
                "import_id": registration["import_id"],
                "registration_payload": json.loads(
                    json.dumps(payload, sort_keys=True, default=str)
                ),
                "request_hash": request_hash,
                "dataset_id": dataset_id,
                "source_id": source_id,
                "dataset_root": str(self.dataset_storage_root),
                "imports_root": str(self.dataset_import_root),
            },
            resource_class="cpu",
            resource_requirements=resource_requirements,
            domain_kind="dataset_registration",
            domain_id=registration["import_id"],
            max_retries=2,
        )
        updated_metadata = dict(session.metadata or {})
        updated_metadata["guided_registration"] = {
            "request_hash": request_hash,
            "inspection_id": inspection_id,
            "dataset_id": dataset_id,
            "source_id": source_id,
        }
        updated = self._dataset_database().update_dataset_import(
            session.id,
            work_item_id=work.id,
            error=None,
            metadata=updated_metadata,
        )
        assert updated is not None
        return {
            "registration": {
                "id": updated.id,
                "status": work.status,
                "dataset_id": dataset_id,
                "source_id": source_id,
            },
            "import": engine.import_view(updated),
            "dataset": None,
            "source": None,
            "preparation_plan": registration["preparation_plan"],
            "work_item_id": work.id,
            "reused": False,
        }

    def execute_inspected_dataset_registration(
        self,
        inspection_id: str,
        payload: Mapping[str, Any],
        *,
        dataset_id: str,
        source_id: str,
    ) -> Dict[str, Any]:
        """Worker entry point for durable guided registration."""

        return self._guided_data_engine().execute_registration(
            inspection_id,
            payload,
            dataset_lab=self._dataset_engine(),
            dataset_id=dataset_id,
            source_id=source_id,
        )

    def cleanup_dataset_imports(
        self, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        values = payload or {}
        approved = bool(values.get("approved", False))
        if approved and not str(values.get("review_note") or "").strip():
            raise ValueError("dataset import cleanup requires a non-empty review_note")
        result = self._guided_data_engine().cleanup(approved=approved)
        if approved:
            result["review_note"] = str(values["review_note"]).strip()
        return result

    def refresh_dataset_source_by_id(self, source_id: str) -> Dict[str, Any]:
        source = self._dataset_database().get_dataset_source(source_id)
        if source is None:
            raise KeyError(source_id)
        if source.kind != "local":
            raise ValueError(
                "only referenced local sources can be refreshed; pinned sources are immutable"
            )
        active = next(
            (
                item
                for item in self._dataset_database().list_work_items(
                    statuses=["queued", "blocked", "running"], limit=10000
                )
                if item.domain_kind == "dataset_source_refresh"
                and item.domain_id == source.id
            ),
            None,
        )
        if active is None:
            active = self._scheduler().enqueue(
                kind="dataset_source_refresh",
                launch_spec={
                    "handler": "own_data.refresh_source",
                    "source_id": source.id,
                    "dataset_root": str(self.dataset_storage_root),
                    "imports_root": str(self.dataset_import_root),
                },
                resource_class="cpu",
                resource_requirements={"output_path": str(self.dataset_storage_root)},
                domain_kind="dataset_source_refresh",
                domain_id=source.id,
                max_retries=2,
            )
        return {
            "refresh": {
                "id": source.id,
                "status": active.status,
                "dataset_id": source.dataset_id,
            },
            "source": source.to_dict(),
            "work_item_id": active.id,
        }

    def get_dataset_version_readiness(
        self,
        version_id: str,
        *,
        trainer_mode: Optional[str] = None,
        model: Optional[str] = None,
        verifier_profile_revision_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        from halo_forge.data_lab import TRAINER_DATASET_ADAPTERS, VersionError

        db = self._dataset_database()
        version = db.get_dataset_version(version_id)
        if version is None:
            raise KeyError(version_id)
        dataset = db.get_dataset(version.dataset_id)
        if dataset is None:
            raise KeyError(version.dataset_id)
        blockers: list[Dict[str, Any]] = []
        warnings: list[Dict[str, Any]] = []
        if version.status != "completed":
            blockers.append(
                {
                    "code": "version_incomplete",
                    "message": "The dataset version has not completed publication.",
                    "remedy": "Open Activity and wait for or retry the build.",
                    "action": "open_activity",
                }
            )
        try:
            verification = self._data_object(
                self._dataset_engine().verify_version(
                    version_id,
                    dataset_id=version.dataset_id,
                    verify_source=not version.assets_materialized,
                )
            )
            if verification.get("valid") is False or verification.get("ok") is False:
                blockers.append(
                    {
                        "code": "version_verification_failed",
                        "message": "; ".join(
                            str(item)
                            for item in (
                                verification.get("problems")
                                or verification.get("errors")
                                or verification.get("missing_assets")
                                or ["Dataset version verification failed."]
                            )
                        ),
                        "remedy": "Refresh a changed source or restore missing assets, then build a new version.",
                        "action": "open_dataset_version",
                    }
                )
        except VersionError as exc:
            # Catalog rows can outlive a moved, cleaned, or only partially
            # published managed version. Readiness is a user-facing check, so
            # report the repair path instead of turning that condition into a
            # 500 response.
            blockers.append(
                {
                    "code": "version_storage_unavailable",
                    "message": str(exc),
                    "remedy": (
                        "Restore the managed version or rebuild it from its "
                        "source before starting training."
                    ),
                    "action": "open_dataset_version",
                }
            )
        except (AttributeError, KeyError):
            pass
        adapters = TRAINER_DATASET_ADAPTERS.list(
            schema=dataset.canonical_schema, trainer_mode=trainer_mode
        )
        compatible = []
        for adapter in adapters:
            modes = (
                [trainer_mode]
                if trainer_mode and trainer_mode in adapter.trainer_modes
                else list(adapter.trainer_modes)
            )
            compatible.extend(
                {
                    "adapter_id": adapter.id,
                    "adapter_version": adapter.version,
                    "trainer_mode": mode,
                    "compatible": True,
                    "reason": None,
                    "required_schema": dataset.canonical_schema,
                }
                for mode in modes
            )
        guided_context = self._guided_version_context(version)
        guided_scenario = None
        scenario_revision_id = self._optional_str(
            guided_context.get("scenario_revision_id")
        )
        if scenario_revision_id:
            try:
                from halo_forge.own_data.registry import TRAINING_SCENARIOS

                scenario = TRAINING_SCENARIOS.get(scenario_revision_id)
                guided_scenario = scenario
                runtime_modes = self._guided_data_engine().runtime_trainer_compatibility(
                    scenario,
                    self._active_backend_name(),
                    trainer_mode=trainer_mode,
                )
                runtime_by_mode = {
                    str(item.get("trainer_mode") or ""): item for item in runtime_modes
                }
                compatible = [
                    {
                        **item,
                        **runtime_by_mode.get(str(item.get("trainer_mode") or ""), {}),
                    }
                    for item in compatible
                    if runtime_by_mode.get(
                        str(item.get("trainer_mode") or ""), {"compatible": False}
                    ).get("compatible")
                ]
                if trainer_mode and not compatible:
                    selected_runtime = runtime_by_mode.get(trainer_mode)
                    blockers.append(
                        {
                            "code": "trainer_runtime_unavailable",
                            "message": str(
                                (selected_runtime or {}).get("reason")
                                or f"{trainer_mode.upper()} is unavailable on the active runtime."
                            ),
                            "remedy": "Choose a verified method shown for this workstation, or install its required runtime.",
                            "action": "choose_trainer",
                        }
                    )
            except KeyError:
                blockers.append(
                    {
                        "code": "scenario_revision_missing",
                        "message": "The guided training scenario revision is no longer available.",
                        "remedy": "Clone the dataset recipe with a current verified scenario.",
                        "action": "open_dataset_version",
                    }
                )
        if guided_scenario is not None and trainer_mode in {"raft", "grpo"}:
            verifier_revision_id = self._optional_str(
                verifier_profile_revision_id
            )
            if not verifier_revision_id:
                blockers.append(
                    {
                        "code": "qualified_verifier_missing",
                        "message": (
                            f"{trainer_mode.upper()} proof training needs a qualified verifier."
                        ),
                        "remedy": (
                            "Choose a compatible candidate- or approved-qualified verifier. "
                            "If none are available, calibrate one in Evaluate → Verifiers first."
                        ),
                        "action": "choose_verifier",
                    }
                )
            else:
                try:
                    self._verifier_engine().resolve_binding(
                        verifier_revision_id,
                        modality="text",
                        require_qualified=True,
                    )
                except (KeyError, ValueError) as exc:
                    blockers.append(
                        {
                            "code": "qualified_verifier_incompatible",
                            "message": str(exc),
                            "remedy": (
                                "Choose a runtime-current candidate- or approved-qualified "
                                "text verifier, or recalibrate the selected revision."
                            ),
                            "action": "choose_verifier",
                        }
                    )
        if not compatible:
            if not any(item["code"] == "trainer_runtime_unavailable" for item in blockers):
                blockers.append(
                    {
                        "code": "trainer_incompatible",
                        "message": (
                            f"No verified {trainer_mode} adapter accepts {dataset.canonical_schema}."
                            if trainer_mode
                            else f"No verified trainer accepts {dataset.canonical_schema}."
                        ),
                        "remedy": "Choose one of the compatible training scenarios.",
                        "action": "choose_trainer",
                    }
                )
        if model and dataset.canonical_schema in {"vlm", "audio"}:
            from halo_forge.capabilities import check_modality_train_capability

            modality = "vlm" if dataset.canonical_schema == "vlm" else "audio"
            check = check_modality_train_capability(
                modality, model, allow_prototype_train=True, dry_run=True
            )
            if not check.allowed:
                blockers.append(
                    {
                        "code": "model_family_incompatible",
                        "message": check.message.splitlines()[-1],
                        "remedy": "Choose a verified model family shown by the scenario.",
                        "action": "choose_model",
                    }
                )
        recommended_model = None
        try:
            from halo_forge.models.catalog import get_model, recommended_models

            backend_name = self._active_backend_name()
            mode_for_models = trainer_mode or (
                str(compatible[0]["trainer_mode"]) if compatible else None
            )
            candidates = recommended_models(
                mode=mode_for_models,
                backend=backend_name,
                modality=(
                    "audio"
                    if dataset.modality == "audio"
                    else "vision"
                    if dataset.modality in {"image", "vlm"}
                    else None
                ),
            )
            try:
                import psutil  # type: ignore

                memory_gb = float(psutil.virtual_memory().total) / 1024**3
            except Exception:
                memory_gb = 0.0
            fitting = [
                item
                for item in candidates
                if not memory_gb
                or not item.get("estimated_memory_gb")
                or float(item["estimated_memory_gb"]) <= memory_gb * 0.8
            ]
            pool = fitting or candidates
            if pool:
                recommended_model = min(
                    pool,
                    key=lambda item: (
                        float(item.get("estimated_memory_gb") or float("inf")),
                        not bool(item.get("recommended_first_run")),
                        str(item.get("id") or ""),
                    ),
                )
            if model:
                selected = get_model(model)
                if selected is not None:
                    supported_modes = set(selected.get("trainer_support") or [])
                    if trainer_mode and trainer_mode not in supported_modes:
                        blockers.append(
                            {
                                "code": "model_trainer_incompatible",
                                "message": f"{model} is not verified for {trainer_mode.upper()} training.",
                                "remedy": "Choose a verified model shown by the scenario.",
                                "action": "choose_model",
                            }
                        )
                    supported_backends = set(selected.get("backend_support") or [])
                    if backend_name and backend_name not in supported_backends:
                        blockers.append(
                            {
                                "code": "model_backend_incompatible",
                                "message": f"{model} is not verified on the active {backend_name} backend.",
                                "remedy": "Choose a model available for this workstation backend.",
                                "action": "choose_model",
                            }
                        )
                    required_gb = float(selected.get("estimated_memory_gb") or 0.0)
                    if memory_gb and required_gb > memory_gb * 0.8:
                        blockers.append(
                            {
                                "code": "model_memory_insufficient",
                                "message": (
                                    f"{model} is estimated to need {required_gb:.1f} GB, "
                                    f"above the guided allowance for this {memory_gb:.1f} GB workstation."
                                ),
                                "remedy": "Choose the smaller recommended model or close memory-heavy applications.",
                                "action": "choose_model",
                            }
                        )
        except Exception:
            # Readiness must remain usable with a minimal installation. The
            # model-aware launch preflight still performs the authoritative
            # access, runtime, RAM, and disk checks.
            recommended_model = None
        return {
            "ready": not blockers,
            "status": "ready" if not blockers else "blocked",
            "blockers": blockers,
            "warnings": warnings,
            "compatible_trainers": compatible,
            "recommended_model": recommended_model,
            "dataset_version_id": version_id,
            "trainer_mode": trainer_mode,
            "model": model,
            "verifier_profile_revision_id": verifier_profile_revision_id,
        }

    def get_version_info(self) -> Dict[str, Any]:
        """Return public product/package version metadata."""

        from halo_forge.version import version_info

        return version_info()

    def get_workspace_info(self) -> Dict[str, Any]:
        """Return local workstation defaults for dashboard-managed runs."""
        root = _default_run_root()
        writable = False
        message = "Halo Forge will save guided runs here."
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".halo-forge-write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            writable = True
        except OSError as exc:
            message = f"Halo Forge could not write to the default run folder: {exc}"
        return {
            "default_run_root": str(root),
            "runs_dir": str(root),
            "writable": writable,
            "message": message,
        }

    def huggingface_status(self) -> Dict[str, Any]:
        return self.huggingface_manager.status(verify=True)

    def huggingface_save_token(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.huggingface_manager.save(str(payload.get("token") or ""))

    def huggingface_clear_token(self) -> Dict[str, Any]:
        return self.huggingface_manager.clear()

    def huggingface_check_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.huggingface_manager.check_model(str(payload.get("model_id") or ""))

    @staticmethod
    def _mlx_readiness_snapshot() -> dict[str, Any]:
        try:
            from halo_forge.backend.mlx_readiness import check_mlx_readiness

            return check_mlx_readiness(timeout_seconds=5.0).to_dict()
        except Exception as exc:
            return {
                "status": "error",
                "executable": False,
                "package_versions": {"mlx": None, "mlx-lm": None},
                "chip": None,
                "macos_version": None,
                "metal_device": None,
                "errors": [f"MLX readiness probe failed: {exc}"],
                "warnings": [],
                "suggested_fixes": [],
                "probe": {},
            }

    async def cancel_run(self, run_identifier: str) -> Dict[str, Any]:
        """Cancel a running training job.

        Only valid for active jobs (`_resolve_run_source` returns
        kind="job"); completed runs in the results service have no
        process to stop. Returns a stable envelope so the frontend can
        render result-or-reason without branching on HTTP status.

        Backed by `TrainingService.stop_job` which sends SIGTERM, waits
        for graceful shutdown (so the trainer can save a checkpoint),
        then SIGKILLs on timeout.
        """
        try:
            source = self._resolve_run_source(run_identifier)
        except KeyError as exc:
            return {
                "ok": False,
                "reason": f"Run not found: {exc}",
                "run_id": run_identifier,
                "status": None,
            }

        if source.get("kind") != "job":
            return {
                "ok": False,
                "reason": "Run is not active; only running jobs can be cancelled.",
                "run_id": run_identifier,
                "status": "completed",
            }

        job = source["job"]
        work_item_id = str((job.lifecycle_metadata or {}).get("work_item_id") or "")
        if work_item_id:
            cancelled = self._scheduler().cancel(work_item_id)
            self._sync_managed_training_job(job)
            return {
                "ok": cancelled is not None,
                "reason": None if cancelled is not None else "Managed work item was not found.",
                "run_id": job.id,
                "work_item_id": work_item_id,
                "status": job.status,
            }
        try:
            stopped = await self.training_service.stop_job(job.id)
        except Exception as exc:
            return {
                "ok": False,
                "reason": f"stop_job failed: {exc}",
                "run_id": job.id,
                "status": job.status,
            }

        return {
            "ok": bool(stopped),
            "reason": None if stopped else "Job was not running.",
            "run_id": job.id,
            "status": job.status,
        }

    def get_run_logs(
        self,
        run_identifier: str,
        *,
        tail: int = 200,
    ) -> Dict[str, Any]:
        """Return the tail of training logs for a run.

        Looks for `run.log` (or `train.log`) inside the run's output_dir
        first; falls back to scanning `logs/` for the newest log file
        whose basename references this run. Honest about unavailability
        — returns `{"available": False, "lines": [], "reason": "..."}`
        rather than 5xx-ing.

        Phase D v2 contract: the frontend polls this every few seconds
        for active runs and renders the last N lines in a virtual-scroll
        panel.
        """
        try:
            source = self._resolve_run_source(run_identifier)
        except Exception as exc:
            return {
                "available": False,
                "lines": [],
                "reason": f"Run not found: {exc}",
                "log_path": None,
                "tail": int(tail),
            }

        from pathlib import Path

        # _resolve_run_source returns either a job (active) or a summary
        # (completed). Both expose an output_dir, but the field lives in
        # different places. Normalize them here.
        output_dir, run_id = _extract_output_dir_and_run_id(source)

        candidates: list[Path] = []
        if output_dir:
            for name in ("run.log", "train.log", "training.log", f"{run_id}_training.log"):
                candidate = output_dir / name
                if candidate.exists():
                    candidates.append(candidate)
            if not candidates:
                matches = sorted(
                    output_dir.glob("*_training.log"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                candidates.extend(matches[:1])

        # Fall back to logs/ scan — newest file whose basename mentions
        # the run_id or output_dir basename.
        if not candidates:
            for logs_dir in _candidate_log_roots(self.base_path):
                if not logs_dir.is_dir():
                    continue
                tokens = [t for t in (run_id, output_dir.name if output_dir else "") if t]
                matches = []
                for log_file in logs_dir.glob("*.log"):
                    if any(tok in log_file.name for tok in tokens):
                        matches.append(log_file)
                # Newest first
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                candidates.extend(matches[:1])
                if candidates:
                    break

        if not candidates:
            return {
                "available": False,
                "lines": [],
                "reason": "No log file found alongside this run.",
                "log_path": None,
                "tail": int(tail),
            }

        log_path = candidates[0]
        # Defensive line-by-line read with a soft cap so an enormous log
        # never blows the API memory budget.
        max_tail = max(1, min(int(tail), 5000))
        try:
            with log_path.open(encoding="utf-8", errors="replace") as f:
                buf: list[str] = []
                for line in f:
                    buf.append(line.rstrip("\n"))
                    if len(buf) > max_tail * 2:
                        # Keep the trailing window only; deque-like prune
                        buf = buf[-max_tail:]
                lines = buf[-max_tail:] if len(buf) > max_tail else buf
        except OSError as exc:
            return {
                "available": False,
                "lines": [],
                "reason": f"Could not read {log_path.name}: {exc}",
                "log_path": str(log_path),
                "tail": max_tail,
            }

        return {
            "available": True,
            "lines": lines,
            "reason": None,
            "log_path": str(log_path),
            "tail": max_tail,
            "total_lines_returned": len(lines),
        }

    def get_run_samples(
        self,
        run_identifier: str,
        *,
        cycle: Optional[int] = None,
        kind: str = "samples",
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Return RAFT-style sample artifacts for a cycle.

        kind="samples"  -> all generated samples for the cycle
                           (cycle_{N}_samples.jsonl)
        kind="accepted" -> the post-filter set fed to SFT
                           (cycle_{N}/accepted.jsonl)

        Returns a stable JSON envelope so the frontend can render an
        "available: false" placeholder when the trainer didn't write
        these artifacts (older summaries, SFT-only runs, or local-only
        files that never reached this host).
        """
        try:
            source = self._resolve_run_source(run_identifier)
        except Exception as exc:
            return {
                "available": False,
                "samples": [],
                "reason": f"Run not found: {exc}",
                "cycle": cycle,
                "kind": kind,
            }

        from pathlib import Path

        out, _ = _extract_output_dir_and_run_id(source)
        if out is None:
            return {
                "available": False,
                "samples": [],
                "reason": "Run has no recorded output_dir.",
                "cycle": cycle,
                "kind": kind,
            }

        # Discover cycles by scanning the output dir for cycle_N folders.
        if cycle is None:
            available_cycles = sorted(
                int(p.name.split("_")[1])
                for p in out.glob("cycle_*")
                if p.name.split("_", 1)[1].isdigit()
            )
            if available_cycles:
                cycle = available_cycles[-1]
            else:
                return {
                    "available": False,
                    "samples": [],
                    "reason": "No cycle artifacts found.",
                    "cycle": None,
                    "kind": kind,
                    "available_cycles": [],
                }
        else:
            cycle = int(cycle)

        if kind == "accepted":
            jsonl_path = out / f"cycle_{cycle}" / "accepted.jsonl"
        else:
            jsonl_path = out / f"cycle_{cycle}_samples.jsonl"

        if not jsonl_path.exists():
            return {
                "available": False,
                "samples": [],
                "reason": f"{jsonl_path.name} not found.",
                "cycle": cycle,
                "kind": kind,
            }

        import json as _json

        samples: list[dict[str, Any]] = []
        max_limit = max(1, min(int(limit), 500))
        try:
            with jsonl_path.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    samples.append(record)
                    if len(samples) >= max_limit:
                        break
        except OSError as exc:
            return {
                "available": False,
                "samples": [],
                "reason": f"Could not read {jsonl_path.name}: {exc}",
                "cycle": cycle,
                "kind": kind,
            }

        # Discover all cycles for the scrubber.
        available_cycles = sorted(
            int(p.name.split("_")[1])
            for p in out.glob("cycle_*")
            if p.name.split("_", 1)[1].isdigit()
        )

        return {
            "available": True,
            "samples": samples,
            "reason": None,
            "cycle": cycle,
            "kind": kind,
            "available_cycles": available_cycles,
            "limit": max_limit,
            "total_returned": len(samples),
            "source_path": str(jsonl_path),
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Live hardware telemetry — the data behind the public_app's
        telemetry strip.

        Polled by the frontend at ~3s intervals. The provider's own
        cache (1s TTL on rocm-smi/nvidia-smi subprocess output) keeps
        the cost bounded even if a flood of clients polls in lockstep.

        Failures inside the provider are caught and surfaced as a
        `note` field on the sample rather than 5xx responses, because
        the strip is meant to *always* render; missing values render
        as "—" but the contract stays stable.
        """
        from halo_forge.telemetry import (
            TelemetryUnavailableError,
            get_telemetry_provider,
        )

        try:
            provider = get_telemetry_provider()
        except TelemetryUnavailableError as exc:
            # Should not happen — the registry falls back to CPU — but
            # we shape the response identically so the frontend never
            # sees an undefined payload.
            return {
                "timestamp": 0.0,
                "backend": "unknown",
                "device_name": None,
                "note": f"Telemetry unavailable: {exc}",
            }
        sample = provider.sample()
        return sample.to_dict()

    def list_training_datasets(self) -> list[dict[str, Any]]:
        """Catalog of known training datasets for the launch configurator.

        Reads `halo_forge.sft.datasets.SFT_DATASETS` (the same registry the
        CLI uses) and projects it down to a JSON-shaped list. Domain
        ('code', 'vlm', 'audio', 'reasoning', 'agentic') is included so
        the frontend can group + filter by modality without re-deriving
        the mapping client-side.
        """
        from halo_forge.sft.datasets import SFT_DATASETS

        items: list[dict[str, Any]] = []
        for spec in SFT_DATASETS.values():
            items.append(
                {
                    "key": spec.name,
                    "huggingface_id": spec.huggingface_id,
                    "description": spec.description,
                    "domain": spec.domain,
                    "size_hint": spec.size_hint,
                    "default_split": spec.default_split,
                }
            )
        return items

    @staticmethod
    def _version_mode_compatibility(version: Any, mode: str) -> tuple[bool, Optional[str]]:
        """Apply loader-level contracts that are stricter than canonical schema."""

        if mode != "reasoning":
            return True, None
        records_path = Path(version.storage_path).expanduser() / "records.jsonl"
        try:
            with records_path.open(encoding="utf-8") as handle:
                for index, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict) or row.get("reference_answer") is None:
                        return (
                            False,
                            "Reasoning training requires reference_answer on every record "
                            f"(first missing at row {index})",
                        )
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"Dataset records are unreadable: {exc}"
        return True, None

    def list_training_dataset_versions(self, mode: Optional[str] = None) -> list[dict[str, Any]]:
        """Completed managed versions, optionally filtered for one trainer."""
        from halo_forge.data_lab import TRAINER_DATASET_ADAPTERS

        normalized_mode = str(mode or "").strip().lower()
        items: list[dict[str, Any]] = []
        db = self._dataset_database()
        for dataset in db.list_datasets(limit=500, offset=0):
            adapters = TRAINER_DATASET_ADAPTERS.list(
                schema=dataset.canonical_schema,
                trainer_mode=normalized_mode or None,
            )
            if normalized_mode and not adapters:
                continue
            for version in db.list_dataset_versions(dataset.id):
                if version.status != "completed":
                    continue
                compatible, _ = self._version_mode_compatibility(version, normalized_mode)
                if normalized_mode and not compatible:
                    continue
                items.append(
                    {
                        **version.to_dict(),
                        "dataset_name": dataset.name,
                        "canonical_schema": dataset.canonical_schema,
                        "modality": dataset.modality,
                        "trainer_compatibility": [
                            {
                                "adapter_id": adapter.id,
                                "adapter_version": adapter.version,
                                "trainer_mode": trainer_mode,
                                "compatible": True,
                                "required_schema": dataset.canonical_schema,
                            }
                            for adapter in adapters
                            for trainer_mode in adapter.trainer_modes
                            if not normalized_mode or trainer_mode == normalized_mode
                        ],
                    }
                )
        return items

    # ----- Dataset Lab ------------------------------------------------------

    @staticmethod
    def _data_object(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        if hasattr(value, "__dict__"):
            return dict(vars(value))
        raise ValueError(f"Dataset Lab returned unsupported value: {type(value).__name__}")

    @staticmethod
    def _source_spec_payload(source: Dict[str, Any]) -> Dict[str, Any]:
        kind = str(source.get("kind") or "local").strip().lower().replace("hf", "huggingface")
        uri = str(source.get("uri") or source.get("path") or source.get("repo_id") or "").strip()
        if not uri:
            raise ValueError("source.uri is required")
        if kind == "local":
            return {"kind": "local", "path": uri}
        if kind == "huggingface":
            return {
                "kind": "huggingface",
                "repo_id": uri,
                "config": source.get("config"),
                "split": str(source.get("split") or "train"),
                "revision": source.get("revision"),
                **({"data_files": source["data_files"]} if "data_files" in source else {}),
            }
        raise ValueError("source.kind must be local or huggingface")

    @staticmethod
    def _source_spec_from_record(source: Any) -> Dict[str, Any]:
        if source.kind == "local":
            return {"kind": "local", "path": source.uri}
        metadata = source.metadata
        return {
            "kind": "huggingface",
            "repo_id": source.uri,
            "config": source.config,
            "split": source.split or "train",
            "revision": source.revision,
            **({"data_files": metadata["data_files"]} if "data_files" in metadata else {}),
        }

    def create_dataset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from halo_forge.data_lab import SourceSpec
        from halo_forge.data_lab.models import infer_schema, validate_record
        from halo_forge.data_lab.sources import load_source

        raw_source = payload.get("source")
        if not isinstance(raw_source, dict):
            raise ValueError("source is required")
        spec_payload = self._source_spec_payload(raw_source)
        spec = SourceSpec.from_value(spec_payload)
        snapshot = load_source(spec)
        inferred_kind = infer_schema(snapshot.records[0]).value if snapshot.records else "sft"
        canonical_schema = str(payload.get("canonical_schema") or inferred_kind).lower()
        modality = str(
            payload.get("modality")
            or ({"vlm": "image", "audio": "audio"}.get(canonical_schema, "text"))
        ).lower()
        uri = str(spec.path or spec.repo_id)
        default_name = Path(uri).stem if spec.kind == "local" else uri.rsplit("/", 1)[-1]
        db = self._dataset_database()
        dataset = db.create_dataset(
            name=str(payload.get("name") or default_name),
            description=payload.get("description"),
            modality=modality,
            canonical_schema=canonical_schema,
            dataset_id=self._optional_str(payload.get("id")),
        )
        try:
            # The facade owns the in-memory/file-backed source cache used by
            # builds. SQLite remains the durable API catalog.
            engine_spec = spec.to_dict()
            engine_spec.update(
                canonical_kind=canonical_schema,
                modality=modality,
                field_mapping=(
                    payload.get("field_mapping") or raw_source.get("field_mapping") or {}
                ),
            )
            engine_source = self._dataset_engine().register_source(
                engine_spec,
                dataset_id=dataset.id,
                name=dataset.name,
                source_id=self._optional_str(raw_source.get("id")),
            )
            engine_data = self._data_object(engine_source)
            source_id = str(
                engine_data.get("id") or engine_data.get("source_id") or snapshot.fingerprint[:24]
            )
            source = db.create_dataset_source(
                dataset_id=dataset.id,
                source_id=source_id,
                kind=spec.kind,
                uri=uri,
                config=spec.config,
                split=spec.split,
                revision=spec.revision,
                fingerprint=snapshot.fingerprint,
                size_bytes=snapshot.size_bytes,
                row_count=len(snapshot.records),
                metadata={
                    "file_count": snapshot.file_count,
                    "assets": [asset.to_dict() for asset in snapshot.assets],
                    "engine": engine_data,
                    **dict(raw_source.get("metadata") or {}),
                    **({"data_files": spec.data_files} if spec.data_files is not None else {}),
                },
            )
        except Exception:
            db.delete_dataset(dataset.id)
            raise
        return self.get_dataset(dataset.id) or {**dataset.to_dict(), "sources": [source.to_dict()]}

    def list_datasets(
        self, *, modality: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        db = self._dataset_database()
        items = []
        for dataset in db.list_datasets(modality=modality, limit=limit, offset=offset):
            data = dataset.to_dict()
            sources = db.list_dataset_sources(dataset.id)
            versions = db.list_dataset_versions(dataset.id)
            jobs = db.list_dataset_jobs(dataset_id=dataset.id, limit=1)
            data.update(
                source=sources[0].to_dict() if sources else None,
                row_count=sources[0].row_count if sources else 0,
                size_bytes=sources[0].size_bytes if sources else 0,
                latest_version=(
                    next((v.to_dict() for v in versions if v.id == dataset.latest_version_id), None)
                ),
                job=jobs[0].to_dict() if jobs else None,
            )
            items.append(data)
        return {"items": items, "limit": limit, "offset": offset}

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        db = self._dataset_database()
        dataset = db.get_dataset(dataset_id)
        if dataset is None:
            return None
        data = dataset.to_dict()
        sources = db.list_dataset_sources(dataset_id)
        versions = db.list_dataset_versions(dataset_id)
        jobs = db.list_dataset_jobs(dataset_id=dataset_id)
        data.update(
            sources=[source.to_dict() for source in sources],
            versions=[version.to_dict() for version in versions],
            jobs=[job.to_dict() for job in jobs],
            latest_version=(
                next((v.to_dict() for v in versions if v.id == dataset.latest_version_id), None)
            ),
        )
        return data

    def list_dataset_sources(self, dataset_id: str) -> Dict[str, Any]:
        db = self._dataset_database()
        if db.get_dataset(dataset_id) is None:
            raise KeyError(dataset_id)
        return {"items": [source.to_dict() for source in db.list_dataset_sources(dataset_id)]}

    def get_dataset_source(self, source_id: str) -> Optional[Dict[str, Any]]:
        source = self._dataset_database().get_dataset_source(source_id)
        return source.to_dict() if source else None

    def refresh_dataset_source(self, dataset_id: str) -> Dict[str, Any]:
        """Legacy direct-service refresh retained for non-HTTP callers.

        Dashboard/browser routes use :meth:`request_dataset_source_refresh` so
        the complete source scan is never performed on an HTTP request.
        """

        db = self._dataset_database()
        sources = db.list_dataset_sources(dataset_id)
        if not sources:
            raise KeyError(dataset_id)
        return self._refresh_dataset_source_record(sources[0])

    def request_dataset_source_refresh(self, dataset_id: str) -> Dict[str, Any]:
        db = self._dataset_database()
        sources = db.list_dataset_sources(dataset_id)
        if not sources:
            raise KeyError(dataset_id)
        return self.refresh_dataset_source_by_id(sources[0].id)

    def execute_dataset_source_refresh(self, source_id: str) -> Dict[str, Any]:
        """Worker entry point that creates, but never mutates, a source revision."""

        source = self._dataset_database().get_dataset_source(source_id)
        if source is None:
            raise KeyError(source_id)
        if source.kind != "local":
            raise ValueError("only referenced local sources can be refreshed")
        return self._refresh_dataset_source_record(source)

    def _refresh_dataset_source_record(self, previous: Any) -> Dict[str, Any]:
        """Refresh one exact source while preserving its guided lineage.

        A refresh is a new source revision, never a mutation.  In particular,
        the confirmed scenario, mapping, and preparation plan must follow the
        source revision so a later proof run remains reproducible.
        """

        db = self._dataset_database()
        refreshed = self._dataset_engine().refresh_source(previous.id)
        data = self._data_object(refreshed)
        if str(data.get("fingerprint")) == previous.fingerprint:
            return {**previous.to_dict(), "refreshed": False, "unchanged": True}
        spec = data.get("spec") or self._source_spec_from_record(previous)
        metadata = dict(previous.metadata or {})
        metadata.update(
            engine=data,
            assets=data.get("asset_fingerprints") or [],
            source_refresh={
                "previous_source_id": previous.id,
                "previous_fingerprint": previous.fingerprint,
                "refreshed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        source = db.create_dataset_source(
            source_id=str(data.get("id")),
            dataset_id=previous.dataset_id,
            kind=str(spec.get("kind") or previous.kind),
            uri=str(spec.get("path") or spec.get("repo_id") or previous.uri),
            config=spec.get("config"),
            split=spec.get("split"),
            revision=spec.get("revision"),
            fingerprint=str(data.get("fingerprint")),
            size_bytes=data.get("size_bytes"),
            row_count=data.get("row_count"),
            metadata=metadata,
            refreshed_from_source_id=previous.id,
        )
        return {**source.to_dict(), "refreshed": True, "unchanged": False}

    def _load_current_source(self, dataset_id: str):
        from halo_forge.data_lab import SourceSpec
        from halo_forge.data_lab.sources import load_source

        db = self._dataset_database()
        sources = db.list_dataset_sources(dataset_id)
        if not sources:
            raise ValueError(f"dataset {dataset_id} has no source")
        source = sources[0]
        snapshot = load_source(SourceSpec.from_value(self._source_spec_from_record(source)))
        if snapshot.fingerprint != source.fingerprint:
            raise ValueError(
                "dataset source has changed; refresh it explicitly to create a new source revision"
            )
        missing = [asset.reference for asset in snapshot.assets if asset.missing]
        if missing:
            raise ValueError(f"dataset source has missing assets: {', '.join(missing[:5])}")
        return source, snapshot

    def preview_dataset(
        self, dataset_id: str, *, offset: int = 0, limit: int = 50
    ) -> Dict[str, Any]:
        if self._dataset_database().get_dataset(dataset_id) is None:
            raise KeyError(dataset_id)
        source, snapshot = self._load_current_source(dataset_id)
        start = max(0, int(offset))
        stop = start + max(1, int(limit))
        return {
            "items": self._preview_asset_urls(snapshot.records[start:stop], source_id=source.id),
            "total": len(snapshot.records),
            "offset": start,
            "limit": limit,
        }

    def dataset_statistics(self, dataset_id: str) -> Dict[str, Any]:
        from halo_forge.data_lab.profiling import profile_records

        dataset = self._dataset_database().get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        source, snapshot = self._load_current_source(dataset_id)
        base_dir: Optional[Path] = None
        if source.kind == "local":
            source_path = Path(source.uri).expanduser().resolve()
            base_dir = source_path.parent if source_path.is_file() else source_path
        profile = profile_records(snapshot.records, base_dir=base_dir)
        return self._data_object(profile)

    def profile_dataset(self, dataset_id: str) -> Dict[str, Any]:
        db = self._dataset_database()
        if db.get_dataset(dataset_id) is None:
            raise KeyError(dataset_id)
        source, _ = self._load_current_source(dataset_id)
        engine_job = self._dataset_engine().start_job(
            "profile", {"source_id": source.id, "dataset_id": dataset_id}
        )
        data = self._normalize_engine_job_data(self._data_object(engine_job))
        job_id = str(data.get("id") or data.get("job_id"))
        job = db.create_dataset_job(
            job_id=job_id,
            dataset_id=dataset_id,
            job_type="profile",
            status=str(data.get("status") or "queued"),
            stage=str(data.get("stage") or "queued"),
            request={"source_id": source.id},
            work_item_id=data.get("work_item_id"),
        )
        return job.to_dict()

    def list_dataset_versions(self, dataset_id: str) -> Dict[str, Any]:
        db = self._dataset_database()
        if db.get_dataset(dataset_id) is None:
            raise KeyError(dataset_id)
        return {"items": [v.to_dict() for v in db.list_dataset_versions(dataset_id)]}

    def get_dataset_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        from halo_forge.data_lab import TRAINER_DATASET_ADAPTERS

        version = self._dataset_database().get_dataset_version(version_id)
        if version is None:
            return None
        data = version.to_dict()
        root = Path(version.storage_path).expanduser()

        def preview_jsonl(path: Path, limit: int = 50) -> list[dict[str, Any]]:
            rows: list[dict[str, Any]] = []
            if not path.is_file():
                return rows
            try:
                with path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        value = json.loads(line)
                        if isinstance(value, dict):
                            rows.append(value)
                        if len(rows) >= limit:
                            break
            except (OSError, json.JSONDecodeError):
                return []
            return rows

        rejected = preview_jsonl(root / "rejected.jsonl")
        quarantined = preview_jsonl(root / "quarantined.jsonl")
        data["rejections"] = {
            "rejected_count": int(
                version.statistics.get("counts", {}).get("rejected", len(rejected))
            ),
            "quarantined_count": int(
                version.statistics.get("counts", {}).get("quarantined", len(quarantined))
            ),
            "rejected_preview": rejected,
            "quarantined_preview": quarantined,
        }
        source = (
            self._dataset_database().get_dataset_source(version.source_id)
            if version.source_id
            else None
        )
        corpus_extraction = (
            copy.deepcopy(
                (
                    (source.metadata or {}).get("guided_own_data") or {}
                ).get("corpus_extraction")
                or {}
            )
            if source is not None
            else {}
        )
        if corpus_extraction:
            data["corpus_extraction"] = corpus_extraction
            data["rejections"]["source_quarantined_count"] = int(
                corpus_extraction.get("quarantined")
                or corpus_extraction.get("quarantined_count")
                or corpus_extraction.get("failed")
                or 0
            )
            data["rejections"]["source_quarantine_preview"] = copy.deepcopy(
                corpus_extraction.get("quarantine_preview") or []
            )
        try:
            data["contamination"] = json.loads(
                (root / "contamination.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            data["contamination"] = {}
        dataset = self._dataset_database().get_dataset(version.dataset_id)
        adapters = (
            TRAINER_DATASET_ADAPTERS.list(schema=dataset.canonical_schema)
            if dataset is not None
            else []
        )
        compatibility = []
        for adapter in adapters:
            for trainer_mode in adapter.trainer_modes:
                compatible, reason = self._version_mode_compatibility(version, trainer_mode)
                compatibility.append(
                    {
                        "adapter_id": adapter.id,
                        "adapter_version": adapter.version,
                        "trainer_mode": trainer_mode,
                        "compatible": compatible,
                        "reason": reason,
                        "required_schema": dataset.canonical_schema if dataset else None,
                    }
                )
        data["trainer_compatibility"] = compatibility
        data["compatible_trainers"] = compatibility
        data["training_artifacts"] = self.list_training_dataset_artifacts(version_id)["items"]
        data["runs"] = self.list_dataset_version_runs(version_id)["items"]
        return data

    def _catalog_training_artifact(self, artifact: Any) -> Dict[str, Any]:
        resolved_counts: dict[tuple[str, str, str], int] = {}
        try:
            manifest = json.loads(
                (Path(artifact.path) / "manifest.json").read_text(encoding="utf-8")
            )
            for value in manifest.get("resolved_bindings") or []:
                key = (
                    str(value.get("role") or "train"),
                    str(value.get("dataset_version_id") or ""),
                    str(value.get("split") or "train"),
                )
                resolved_counts[key] = int(value.get("row_count") or 0)
        except (OSError, json.JSONDecodeError):
            pass
        bindings = []
        for binding in artifact.bindings:
            value = binding.to_dict()
            value["row_count"] = resolved_counts.get(
                (binding.role, binding.dataset_version_id, binding.split),
                int(artifact.row_counts.get(binding.role, 0)),
            )
            bindings.append(value)
        record = self._dataset_database().create_training_artifact(
            artifact_id=artifact.artifact_id,
            artifact_hash=artifact.artifact_hash,
            adapter_id=artifact.adapter_id,
            adapter_version=artifact.adapter_version,
            trainer_mode=artifact.trainer_mode,
            model_id=artifact.model,
            tokenizer_revision=artifact.tokenizer_revision,
            chat_template_hash=artifact.chat_template_hash,
            manifest_path=str(Path(artifact.path) / "manifest.json"),
            bindings=bindings,
            metadata=artifact.to_dict(),
        )
        return self._training_artifact_view({**record.to_dict(), **artifact.to_dict()})

    @staticmethod
    def _training_artifact_view(payload: Dict[str, Any]) -> Dict[str, Any]:
        value = dict(payload)
        bindings = list(value.get("bindings") or [])
        primary = next(
            (binding for binding in bindings if binding.get("role") == "train"),
            bindings[0] if bindings else {},
        )
        validation_policy = dict(value.get("validation_policy") or {})
        value.update(
            id=str(value.get("artifact_id") or value.get("id") or ""),
            artifact_id=str(value.get("artifact_id") or value.get("id") or ""),
            dataset_version_id=primary.get("dataset_version_id"),
            status="ready",
            stage="complete",
            progress_percent=100.0,
            paths=dict(value.get("split_paths") or value.get("paths") or {}),
            storage_path=value.get("path") or value.get("storage_path"),
            asset_root=(value.get("asset_roots") or [None])[0],
            derived_validation=validation_policy.get("kind") == "derived",
        )
        return value

    def create_training_dataset_artifact(
        self, version_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        from halo_forge.data_lab import DatasetBinding

        db = self._dataset_database()
        version = db.get_dataset_version(version_id)
        if version is None:
            raise KeyError(version_id)
        raw_bindings = payload.get("bindings") or [
            {
                "role": "train",
                "dataset_version_id": version_id,
                "split": str(payload.get("split") or "train"),
            }
        ]
        if not isinstance(raw_bindings, list):
            raise ValueError("bindings must be a list")
        bindings = [DatasetBinding.from_value(value) for value in raw_bindings]
        if not bindings:
            raise ValueError("at least one dataset binding is required")
        if not any(binding.dataset_version_id == version_id for binding in bindings):
            raise ValueError("the version in the training-artifact URL must appear in its bindings")
        for binding in bindings:
            bound_version = db.get_dataset_version(binding.dataset_version_id)
            if bound_version is None:
                raise ValueError(f"Unknown dataset version: {binding.dataset_version_id}")
            if bound_version.status != "completed":
                raise ValueError(f"Dataset version {binding.dataset_version_id} is not complete")
            if bound_version.split_counts and binding.split not in bound_version.split_counts:
                raise ValueError(
                    f"Dataset version {binding.dataset_version_id} has no {binding.split!r} split"
                )
        options = {
            "trainer_mode": str(payload.get("trainer_mode") or payload.get("mode") or "sft"),
            "adapter_id": self._optional_str(payload.get("adapter_id")),
            "model": self._optional_str(payload.get("model")),
            "tokenizer_revision": self._optional_str(payload.get("tokenizer_revision")),
            "chat_template": self._optional_str(payload.get("chat_template")),
            "validation_fraction": float(payload.get("validation_fraction", 0.05)),
            "seed": int(payload.get("seed", 42)),
        }
        if options["trainer_mode"] == "cpt":
            options.update(
                {
                    "model_revision": self._optional_str(
                        payload.get("model_revision")
                    ),
                    "model_hash": self._optional_str(payload.get("model_hash")),
                    "tokenizer_hash": self._optional_str(
                        payload.get("tokenizer_hash")
                    ),
                    "max_sequence_length": int(
                        payload.get("max_sequence_length") or 2048
                    ),
                    "packing": str(
                        payload.get("packing")
                        or "paragraph_eos_non_overlap_v1"
                    ),
                    "budget_mode": str(payload.get("budget_mode") or "passes"),
                    "target_tokens": self._optional_int(
                        payload.get("target_tokens")
                    ),
                    "corpus_passes": self._optional_float(
                        payload.get("corpus_passes")
                    ),
                    "effective_batch_size": int(
                        payload.get("effective_batch_size")
                        or (
                            int(payload.get("batch_size") or 1)
                            * int(
                                payload.get("gradient_accumulation_steps")
                                or 8
                            )
                        )
                    ),
                }
            )
        engine = self._dataset_engine()
        starter = getattr(engine, "start_training_artifact_job", None)
        if callable(starter):
            engine_job = starter(bindings, **options)
        else:
            generic_starter = getattr(engine, "start_job", None)
            if not callable(generic_starter):
                raise ValueError(
                    "Dataset Lab engine does not support background training-artifact jobs"
                )
            engine_job = generic_starter(
                "training_artifact",
                {
                    "bindings": [binding.to_dict() for binding in bindings],
                    "options": options,
                },
            )
        engine_data = self._normalize_engine_job_data(self._data_object(engine_job))
        job_id = str(engine_data.get("id") or engine_data.get("job_id") or "")
        if not job_id:
            raise ValueError("Dataset Lab did not return a training-artifact job id")
        request = {
            "version_id": version_id,
            "bindings": [binding.to_dict() for binding in bindings],
            "options": options,
        }
        job = db.get_dataset_job(job_id) or db.create_dataset_job(
            job_id=job_id,
            dataset_id=version.dataset_id,
            version_id=version_id,
            job_type="training_artifact",
            status=str(engine_data.get("status") or "queued"),
            stage=str(engine_data.get("stage") or "queued"),
            request=request,
            work_item_id=engine_data.get("work_item_id"),
        )
        return job.to_dict()

    @staticmethod
    def _training_artifact_job_view(job: Any) -> Dict[str, Any]:
        request = dict(job.request or {})
        options = dict(request.get("options") or {})
        bindings = list(request.get("bindings") or [])
        primary = next(
            (value for value in bindings if value.get("role") == "train"),
            bindings[0] if bindings else {},
        )
        status = str(job.status or "queued")
        visible_status = "rendering" if status == "running" else status
        if status == "completed":
            visible_status = "ready"
        if job.total_records:
            progress = min(
                100.0,
                100.0 * float(job.processed_records or 0) / float(job.total_records),
            )
        elif status == "completed":
            progress = 100.0
        else:
            progress = 0.0
        return {
            "id": job.id,
            "job_id": job.id,
            "artifact_id": None,
            "dataset_version_id": primary.get("dataset_version_id") or job.version_id,
            "status": visible_status,
            "stage": job.stage,
            "progress_percent": progress,
            "adapter_id": options.get("adapter_id") or "",
            "adapter_version": "",
            "trainer_mode": options.get("trainer_mode") or "",
            "model": options.get("model"),
            "tokenizer_revision": options.get("tokenizer_revision"),
            "bindings": bindings,
            "paths": {},
            "row_counts": {},
            "token_statistics": {},
            "artifact_hash": None,
            "storage_path": None,
            "created_at": job.created_at,
            "error": job.error,
        }

    def list_training_dataset_artifacts(self, version_id: str) -> Dict[str, Any]:
        db = self._dataset_database()
        version = db.get_dataset_version(version_id)
        if version is None:
            raise KeyError(version_id)
        ready = [
            self.get_training_dataset_artifact(value.id) or value.to_dict()
            for value in db.list_training_artifacts(dataset_version_id=version_id)
        ]
        preparations: list[dict[str, Any]] = []
        for raw_job in db.list_dataset_jobs(dataset_id=version.dataset_id, limit=500):
            if raw_job.job_type != "training_artifact":
                continue
            job = self._sync_dataset_job(raw_job.id) or raw_job
            bindings = list(job.request.get("bindings") or [])
            if not any(
                str(binding.get("dataset_version_id") or "") == version_id for binding in bindings
            ):
                continue
            artifact_id = str(job.checkpoint.get("training_artifact_id") or "")
            if job.status == "completed" and artifact_id:
                continue
            preparations.append(self._training_artifact_job_view(job))
        items = [*preparations, *ready]
        items.sort(key=lambda value: str(value.get("created_at") or ""), reverse=True)
        return {"items": items}

    def get_training_dataset_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        value = self._dataset_database().get_training_artifact(artifact_id)
        if value is None:
            return None
        payload = value.to_dict()
        if isinstance(payload.get("metadata"), dict):
            payload = {**payload, **payload["metadata"]}
        try:
            payload.update(self._dataset_engine().training_artifacts.get(artifact_id).to_dict())
        except Exception:
            pass
        return self._training_artifact_view(payload)

    def compare_dataset_versions(self, version_id: str, other_version_id: str) -> Dict[str, Any]:
        if self._dataset_database().get_dataset_version(version_id) is None:
            raise KeyError(version_id)
        if self._dataset_database().get_dataset_version(other_version_id) is None:
            raise KeyError(other_version_id)
        value = self._dataset_engine().compare_versions(version_id, other_version_id).to_dict()
        return {
            **value,
            "base_version_id": version_id,
            "other_version_id": other_version_id,
            "changed": value.get("content_changed", []),
            "split_moved": value.get("moved_between_splits", []),
            "recipe_diff": value.get("recipe", {}),
            "statistics_diff": value.get("statistics", {}),
            "source_contribution_diff": value.get("source_contributions", {}),
        }

    def list_dataset_version_runs(self, version_id: str) -> Dict[str, Any]:
        db = self._dataset_database()
        if db.get_dataset_version(version_id) is None:
            raise KeyError(version_id)
        finder = getattr(db, "list_runs_for_dataset_version", None)
        bindings = finder(version_id) if callable(finder) else []
        items: list[dict[str, Any]] = []
        for binding in bindings:
            item = binding.to_dict()
            try:
                detail = self.get_run_detail(binding.run_id, include_research=False)
                item.update(
                    {
                        "run_id": str(detail.get("id") or binding.run_id),
                        "modality": detail.get("modality") or detail.get("type"),
                        "model_name": detail.get("model_name") or detail.get("model"),
                        "status": detail.get("status"),
                        "created_at": detail.get("created_at") or detail.get("timestamp"),
                        "output_dir": detail.get("output_dir"),
                    }
                )
            except (KeyError, ValueError, FileNotFoundError):
                # Bindings outlive optional filesystem summaries, so keep the
                # relationship visible even when the run artifact is missing.
                item.setdefault("modality", "unknown")
                item.setdefault("status", "unavailable")
            items.append(item)
        return {"items": items}

    def build_dataset(self, dataset_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        from halo_forge.data_lab import Recipe

        db = self._dataset_database()
        dataset = db.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        source, _ = self._load_current_source(dataset_id)
        raw_recipe = payload.get("recipe")
        if not isinstance(raw_recipe, dict):
            raise ValueError("recipe is required")
        recipe = Recipe.from_value(raw_recipe)
        engine_job = self._dataset_engine().start_job(
            source.id,
            recipe,
            canonical_kind=dataset.canonical_schema,
            dataset_id=dataset_id,
            materialize_assets=bool(payload.get("materialize_assets", False)),
        )
        engine_data = self._normalize_engine_job_data(self._data_object(engine_job))
        job_id = str(engine_data.get("id") or engine_data.get("job_id"))
        if not job_id:
            raise ValueError("Dataset Lab did not return a job id")
        job = db.get_dataset_job(job_id) or db.create_dataset_job(
            job_id=job_id,
            dataset_id=dataset_id,
            version_id=engine_data.get("version_id"),
            job_type="build",
            status=str(engine_data.get("status") or "queued"),
            stage=str(engine_data.get("stage") or "queued"),
            request={"source_id": source.id, "recipe": raw_recipe},
            work_item_id=engine_data.get("work_item_id"),
        )
        return job.to_dict()

    @staticmethod
    def _normalize_engine_job_data(data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(data)
        status = str(normalized.get("status") or "queued")
        normalized["status"] = "completed" if status == "succeeded" else status
        aliases = {
            "kind": "job_type",
            "payload": "request",
            "processed": "processed_records",
            "total": "total_records",
            "accepted": "accepted_records",
            "rejected": "rejected_records",
            "finished_at": "completed_at",
        }
        for source, target in aliases.items():
            if target not in normalized and source in normalized:
                normalized[target] = normalized[source]
        result = normalized.get("result")
        if isinstance(result, dict):
            normalized.setdefault("version_id", result.get("version_id"))
            normalized.setdefault("dataset_id", result.get("dataset_id"))
        request = normalized.get("request")
        if isinstance(request, dict):
            normalized.setdefault("dataset_id", request.get("dataset_id"))
        return normalized

    def _sync_dataset_job(self, job_id: str) -> Optional[Any]:
        db = self._dataset_database()
        record = db.get_dataset_job(job_id)
        try:
            engine_job = self._dataset_engine().get_job(job_id)
        except Exception:
            return record
        data = self._normalize_engine_job_data(self._data_object(engine_job))
        if record is None:
            record = db.create_dataset_job(
                job_id=job_id,
                dataset_id=data.get("dataset_id"),
                version_id=data.get("version_id"),
                job_type=str(data.get("job_type") or "build"),
                status=str(data.get("status") or "queued"),
                stage=str(data.get("stage") or "queued"),
                request=data.get("request") or {},
                work_item_id=data.get("work_item_id"),
            )
        updates = {
            key: data[key]
            for key in (
                "status",
                "stage",
                "processed_records",
                "total_records",
                "accepted_records",
                "rejected_records",
                "output_size_bytes",
                "error",
                "cancel_requested",
                "started_at",
                "completed_at",
                "work_item_id",
                "logs",
                "checkpoint",
            )
            if key in data
        }
        if "checkpoint" in updates:
            # Public-API synchronization adds catalog identities to the same
            # checkpoint envelope used by the engine.  Preserve those values
            # when the engine subsequently reports its recipe/render boundary.
            updates["checkpoint"] = {
                **dict(record.checkpoint or {}),
                **dict(updates["checkpoint"] or {}),
            }
        record = db.update_dataset_job(job_id, **updates) or record
        record = self._sync_completed_training_artifact(record, data) or record
        self._sync_completed_engine_version(record, data)
        version_id = data.get("version_id")
        if version_id and db.get_dataset_version(str(version_id)) is not None:
            record = db.update_dataset_job(job_id, version_id=str(version_id)) or record
        return record

    def _sync_completed_training_artifact(
        self, job: Any, job_data: Dict[str, Any]
    ) -> Optional[Any]:
        """Catalog a successfully published renderer bundle exactly once.

        The persistent data worker owns filesystem publication.  SQLite is
        intentionally synchronized when API clients poll, matching version-job
        catalog synchronization and keeping the rendering worker independent of
        the public API database.
        """

        if job.status != "completed" or job.job_type != "training_artifact":
            return job
        db = self._dataset_database()
        checkpoint = dict(job.checkpoint or {})
        known_id = str(checkpoint.get("training_artifact_id") or "")
        if known_id and db.get_training_artifact(known_id) is not None:
            return job
        result = job_data.get("result")
        if not isinstance(result, dict):
            return job
        artifact_id = str(result.get("artifact_id") or result.get("id") or "")
        if not artifact_id:
            return job
        artifact: Any = None
        try:
            artifact = self._dataset_engine().training_artifacts.get(artifact_id)
        except Exception:
            # Injected/test Dataset Lab facades may return the serialized
            # artifact directly without exposing a renderer catalog.
            try:
                from halo_forge.data_lab import DatasetBinding, TrainingDatasetArtifact

                values = {
                    key: value
                    for key, value in result.items()
                    if key in TrainingDatasetArtifact.__dataclass_fields__
                }
                values["bindings"] = tuple(
                    DatasetBinding.from_value(value) for value in result.get("bindings") or ()
                )
                values["resolved_bindings"] = tuple(
                    dict(value) for value in result.get("resolved_bindings") or ()
                )
                values["asset_roots"] = tuple(result.get("asset_roots") or ())
                artifact = TrainingDatasetArtifact(**values)
            except (TypeError, ValueError, KeyError):
                return job
        cataloged = self._catalog_training_artifact(artifact)
        checkpoint.update(
            training_artifact_id=str(cataloged.get("artifact_id") or artifact_id),
            artifact_hash=str(cataloged.get("artifact_hash") or artifact.artifact_hash),
        )
        return db.update_dataset_job(job.id, checkpoint=checkpoint) or job

    def _sync_completed_engine_version(self, job: Any, job_data: Dict[str, Any]) -> None:
        if job.status != "completed":
            return
        version_id = job_data.get("version_id") or job.version_id
        if not version_id:
            return
        db = self._dataset_database()
        if db.get_dataset_version(str(version_id)) is not None:
            sync_exposures = getattr(self._dataset_engine(), "sync_version_exposures", None)
            if callable(sync_exposures):
                sync_exposures(
                    str(version_id),
                    database=db,
                    dataset_id=job.dataset_id,
                )
            return
        try:
            value = self._dataset_engine().get_version(str(version_id), dataset_id=job.dataset_id)
            data = self._data_object(value)
        except Exception:
            return
        dataset_id = str(job.dataset_id or data.get("dataset_id") or "")
        if not dataset_id:
            return
        storage_path = str(data.get("storage_path") or data.get("path") or "")
        recipe = data.get("recipe") or job.request.get("recipe") or {}
        manifest: Dict[str, Any] = {}
        if not recipe and storage_path:
            try:
                recipe = json.loads(
                    (Path(storage_path) / "recipe.json").read_text(encoding="utf-8")
                )
                manifest = json.loads(
                    (Path(storage_path) / "manifest.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                recipe = {}
        elif storage_path:
            try:
                manifest = json.loads(
                    (Path(storage_path) / "manifest.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                manifest = {}
        recipe_hash = str(data.get("recipe_hash") or data.get("recipe_id") or "")
        statistics = data.get("statistics") or data.get("profile") or {}
        provenance = data.get("provenance") or {}
        if storage_path:
            try:
                if not statistics:
                    statistics = json.loads(
                        (Path(storage_path) / "stats.json").read_text(encoding="utf-8")
                    )
                if not provenance:
                    provenance = {
                        "steps": json.loads(
                            (Path(storage_path) / "provenance.json").read_text(encoding="utf-8")
                        ),
                        "split_paths": {
                            split: str(Path(storage_path) / "splits" / f"{split}.jsonl")
                            for split in (data.get("split_counts") or {})
                        },
                    }
            except (OSError, json.JSONDecodeError):
                pass
        mixture_parents: list[dict[str, Any]] = []
        source_fingerprints = dict(data.get("source_fingerprints") or {})
        if data.get("source_fingerprint"):
            source_fingerprints.setdefault(
                str(job.request.get("source_id") or "source"),
                str(data["source_fingerprint"]),
            )
        for step in recipe.get("steps", []) if isinstance(recipe, dict) else []:
            if not isinstance(step, dict) or step.get("kind", step.get("type")) != "mix":
                continue
            for parent in step.get("datasets", []):
                if not isinstance(parent, dict) or parent.get("source") in {None, "current"}:
                    continue
                parent_id = str(parent["source"])
                parent_version = db.get_dataset_version(parent_id)
                if parent_version is not None:
                    mixture_parents.append(
                        {
                            "parent_version_id": parent_id,
                            "role": "mixture",
                            "weight": parent.get("weight"),
                        }
                    )
                    if parent_version.content_hash:
                        source_fingerprints[f"version:{parent_id}"] = parent_version.content_hash
        db.create_dataset_version(
            version_id=str(version_id),
            dataset_id=dataset_id,
            source_id=data.get("source_id") or job.request.get("source_id"),
            parent_version_id=data.get("parent_version_id") or manifest.get("parent_version_id"),
            parent_versions=(
                data.get("parents") or data.get("parent_version_ids") or mixture_parents
            ),
            status="completed",
            content_hash=data.get("content_hash") or data.get("version_hash"),
            recipe_hash=recipe_hash,
            recipe=recipe,
            storage_path=storage_path,
            row_count=int(data.get("row_count") or 0),
            size_bytes=int(
                data.get("size_bytes")
                or (
                    sum(
                        path.stat().st_size
                        for path in Path(storage_path).rglob("*")
                        if path.is_file()
                    )
                    if storage_path and Path(storage_path).is_dir()
                    else 0
                )
            ),
            split_counts=data.get("split_counts") or {},
            statistics=statistics,
            provenance=provenance,
            source_fingerprints=source_fingerprints,
            assets_materialized=bool(
                data.get("assets_materialized", data.get("materialized_assets", False))
            ),
        )
        sync_exposures = getattr(self._dataset_engine(), "sync_version_exposures", None)
        if callable(sync_exposures):
            sync_exposures(
                str(version_id),
                database=db,
                dataset_id=dataset_id,
            )

    def list_dataset_jobs(
        self, *, dataset_id: Optional[str] = None, status: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        db = self._dataset_database()
        existing = db.list_dataset_jobs(dataset_id=dataset_id, status=status, limit=limit)
        items = []
        for job in existing:
            synced = self._sync_dataset_job(job.id) or job
            if status is None or synced.status == status:
                items.append(self._dataset_job_view(synced))
        return {"items": items}

    def get_dataset_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._sync_dataset_job(job_id)
        return self._dataset_job_view(job) if job else None

    def _dataset_job_view(self, job: Any) -> Dict[str, Any]:
        value = job.to_dict()
        if job.job_type != "training_artifact":
            return value
        artifact_id = str(job.checkpoint.get("training_artifact_id") or "")
        value["training_artifact_id"] = artifact_id or None
        value["artifact_id"] = artifact_id or None
        if artifact_id:
            value["training_artifact"] = self.get_training_dataset_artifact(artifact_id)
        return value

    def cancel_dataset_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        db = self._dataset_database()
        if db.get_dataset_job(job_id) is None:
            return None
        self._dataset_engine().cancel(job_id)
        self._sync_dataset_job(job_id)
        job = db.cancel_dataset_job(job_id)
        return job.to_dict() if job else None

    def retry_dataset_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        db = self._dataset_database()
        existing = self._sync_dataset_job(job_id)
        if existing is None:
            return None
        engine_job = self._dataset_engine().retry(job_id)
        data = self._normalize_engine_job_data(self._data_object(engine_job))
        new_id = str(data.get("id") or data.get("job_id") or job_id)
        if new_id == job_id:
            job = db.retry_dataset_job(job_id)
        else:
            job = db.create_dataset_job(
                job_id=new_id,
                dataset_id=existing.dataset_id,
                version_id=data.get("version_id"),
                job_type=existing.job_type,
                request=existing.request,
                status=str(data.get("status") or "queued"),
                stage=str(data.get("stage") or "queued"),
                work_item_id=data.get("work_item_id"),
            )
        return job.to_dict() if job else None

    def preview_dataset_version(
        self, version_id: str, *, split: str = "train", offset: int = 0, limit: int = 50
    ) -> Dict[str, Any]:
        if self._dataset_database().get_dataset_version(version_id) is None:
            raise KeyError(version_id)
        value = self._dataset_engine().get_preview(
            version_id,
            split=split,
            offset=offset,
            limit=limit,
            dataset_id=self._dataset_database().get_dataset_version(version_id).dataset_id,
        )
        data = self._data_object(value)
        if "items" not in data and "records" in data:
            data["items"] = data.pop("records")
        version = self._dataset_database().get_dataset_version(version_id)
        data["items"] = self._preview_asset_urls(
            data.get("items") or [],
            source_id=version.source_id,
            version_id=version_id if version.assets_materialized else None,
        )
        data.setdefault("offset", offset)
        data.setdefault("limit", limit)
        return data

    def _preview_asset_urls(
        self,
        records: List[Dict[str, Any]],
        *,
        source_id: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from copy import deepcopy
        from urllib.parse import quote

        output = deepcopy(records)
        for record in output:
            asset_metadata: Dict[str, Any] = {}
            for field_name in ("image", "image_path", "audio", "audio_path"):
                reference = record.get(field_name)
                if not isinstance(reference, str) or reference.startswith(
                    ("http://", "https://", "data:", "/api/")
                ):
                    continue
                if version_id and reference.startswith("assets/"):
                    url = (
                        f"/api/public/dataset-version-assets/{quote(version_id, safe='')}"
                        f"?path={quote(reference, safe='')}"
                    )
                elif source_id:
                    url = (
                        f"/api/public/dataset-source-assets/{quote(source_id, safe='')}"
                        f"?reference={quote(reference, safe='')}"
                    )
                else:
                    continue
                asset_metadata[field_name] = {"reference": reference, "url": url}
                record[field_name] = url
            if asset_metadata:
                existing = record.get("_halo_forge_assets")
                record["_halo_forge_assets"] = {
                    **(existing if isinstance(existing, dict) else {}),
                    **asset_metadata,
                }
        return output

    def dataset_source_asset_path(self, source_id: str, reference: str) -> Path:
        from halo_forge.data_lab.sources import hash_file

        source = self._dataset_database().get_dataset_source(source_id)
        if source is None:
            raise KeyError(source_id)
        for asset in source.metadata.get("assets") or []:
            if asset.get("reference") != reference:
                continue
            resolved = Path(str(asset.get("resolved_path") or "")).expanduser()
            if not resolved.is_file():
                raise ValueError("dataset asset is missing")
            expected = asset.get("fingerprint")
            if expected and hash_file(resolved) != expected:
                raise ValueError("dataset asset changed after source registration")
            return resolved.resolve()
        raise KeyError(reference)

    def dataset_version_asset_path(self, version_id: str, relative_path: str) -> Path:
        version = self._dataset_database().get_dataset_version(version_id)
        if version is None:
            raise KeyError(version_id)
        root = Path(version.storage_path).expanduser().resolve()
        requested = (root / relative_path).resolve()
        try:
            requested.relative_to(root)
        except ValueError as exc:
            raise KeyError(relative_path) from exc
        if not relative_path.startswith("assets/") or not requested.is_file():
            raise KeyError(relative_path)
        return requested

    def dataset_version_statistics(self, version_id: str) -> Dict[str, Any]:
        version = self._dataset_database().get_dataset_version(version_id)
        if version is None:
            raise KeyError(version_id)
        return version.statistics

    def list_document_extractors(self) -> Dict[str, Any]:
        """Describe the reviewed document adapters available in this runtime."""

        import importlib.util
        import shutil

        pdf_available = bool(
            importlib.util.find_spec("pypdf")
            or importlib.util.find_spec("PyPDF2")
            or shutil.which("pdftotext")
        )
        parquet_available = importlib.util.find_spec("pyarrow") is not None
        items = [
            {
                "id": "plain-text@1",
                "label": "Plain text",
                "version": "1",
                "available": True,
                "source_kinds": ["file", "directory", "upload"],
                "media_types": ["text/plain"],
                "extensions": [".txt", ".text"],
                "preserves": [
                    "paragraph boundaries",
                    "relative filename",
                    "source checksum",
                ],
                "limitations": ["Binary or empty files are quarantined."],
                "reason": None,
            },
            {
                "id": "markdown@1",
                "label": "Markdown",
                "version": "1",
                "available": True,
                "source_kinds": ["file", "directory", "upload"],
                "media_types": ["text/markdown"],
                "extensions": [".md", ".markdown", ".mdown", ".mkd"],
                "preserves": [
                    "headings",
                    "code fences",
                    "paragraph boundaries",
                    "relative filename",
                ],
                "limitations": [
                    "Markdown is retained as training text; it is not rendered to HTML."
                ],
                "reason": None,
            },
            {
                "id": "visible-html@1",
                "label": "Visible HTML text",
                "version": "1",
                "available": True,
                "source_kinds": ["file", "directory", "upload"],
                "media_types": ["text/html"],
                "extensions": [".html", ".htm"],
                "preserves": ["visible text", "document title", "block boundaries"],
                "limitations": [
                    "Scripts, styles, hidden nodes, and embedded binary media are excluded."
                ],
                "reason": None,
            },
            {
                "id": "docx@1",
                "label": "Word document text",
                "version": "1",
                "available": True,
                "source_kinds": ["file", "directory", "upload"],
                "media_types": [
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ],
                "extensions": [".docx"],
                "preserves": [
                    "paragraph text",
                    "table cell text",
                    "core title metadata",
                ],
                "limitations": [
                    "Comments, tracked changes, drawings, and embedded media are not training text."
                ],
                "reason": None,
            },
            {
                "id": "text-layer-pdf@1",
                "label": "Text-layer PDF",
                "version": "1",
                "available": pdf_available,
                "source_kinds": ["file", "directory", "upload"],
                "media_types": ["application/pdf"],
                "extensions": [".pdf"],
                "preserves": ["page identity", "page count", "text-layer content"],
                "limitations": [
                    "Image-only or encrypted PDFs are quarantined; OCR is not performed."
                ],
                "reason": (
                    None
                    if pdf_available
                    else "Install the corpus extra or the pdftotext executable."
                ),
            },
            {
                "id": "structured-text@1",
                "label": "Structured text rows",
                "version": "1",
                "available": True,
                "source_kinds": [
                    "file",
                    "directory",
                    "upload",
                    "huggingface",
                ],
                "media_types": [
                    "application/json",
                    "application/x-ndjson",
                    "text/csv",
                    "text/tab-separated-values",
                ],
                "extensions": [
                    ".json",
                    ".jsonl",
                    ".jl",
                    ".csv",
                    ".tsv",
                    ".parquet",
                ],
                "preserves": [
                    "selected text fields",
                    "row identity",
                    "selected metadata",
                ],
                "limitations": [
                    (
                        "Parquet is available."
                        if parquet_available
                        else "Parquet requires pyarrow; JSON, JSONL, CSV, and TSV remain available."
                    )
                ],
                "reason": None,
                "metadata": {"parquet_available": parquet_available},
            },
        ]
        return {
            "items": items,
            "total": len(items),
            "limit": len(items),
            "offset": 0,
        }

    def create_document_extraction(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        path = payload.get("path") or payload.get("source_path")
        return self._corpus_extraction_engine().launch(
            path,
            import_id=(
                str(payload.get("import_id"))
                if payload.get("import_id") is not None
                else None
            ),
            source_id=(
                str(payload.get("source_id"))
                if payload.get("source_id") is not None
                else None
            ),
            config=(
                payload.get("config")
                if isinstance(payload.get("config"), Mapping)
                else None
            ),
            synchronous=False,
        )

    def list_document_extractions(
        self,
        *,
        import_id: Optional[str] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        bounded_limit = max(1, min(500, int(limit)))
        bounded_offset = max(0, int(offset))
        records = self._dataset_database().list_document_extractions(
            import_id=import_id,
            source_id=source_id,
            status=status,
            limit=bounded_limit,
            offset=bounded_offset,
        )
        total = self._dataset_database().count_document_extractions(
            import_id=import_id,
            source_id=source_id,
            status=status,
        )
        return {
            "items": [record.to_dict() for record in records],
            "total": total,
            "limit": bounded_limit,
            "offset": bounded_offset,
        }

    def get_document_extraction(self, extraction_id: str) -> Dict[str, Any]:
        return self._corpus_extraction_engine().status(extraction_id)

    def preview_document_extraction(
        self,
        extraction_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
        include_text: bool = True,
    ) -> Dict[str, Any]:
        engine = self._corpus_extraction_engine()
        status = engine.status(extraction_id)
        bounded_limit = max(0, min(200, int(limit)))
        bounded_offset = max(0, int(offset))
        document_total = int(status.get("document_count") or 0)
        quarantined_total = int(status.get("quarantined_count") or 0)
        remaining = bounded_limit
        records: list[Dict[str, Any]] = []
        quarantine: list[Dict[str, Any]] = []

        if remaining and bounded_offset < document_total:
            document_limit = min(remaining, document_total - bounded_offset)
            document_page = engine.preview(
                extraction_id,
                limit=document_limit,
                offset=bounded_offset,
                include_text=include_text,
            )
            records = copy.deepcopy(document_page.get("records") or [])
            remaining -= len(records)
        if remaining:
            quarantine_offset = max(0, bounded_offset - document_total)
            quarantine_page = engine.preview(
                extraction_id,
                limit=remaining,
                offset=quarantine_offset,
                include_text=include_text,
            )
            quarantine = copy.deepcopy(
                quarantine_page.get("quarantine") or []
            )

        items = copy.deepcopy(records)
        for failure in quarantine:
            items.append(
                {
                    **copy.deepcopy(failure),
                    "issues": [
                        {
                            "code": failure.get("error_code"),
                            "message": failure.get("error"),
                        }
                    ],
                }
            )
        return {
            "extraction": status,
            "records": records,
            "documents": copy.deepcopy(records),
            "quarantine": quarantine,
            "items": items,
            "total": document_total + quarantined_total,
            "document_total": document_total,
            "quarantined_total": quarantined_total,
            "limit": bounded_limit,
            "offset": bounded_offset,
        }

    def verify_document_extraction(self, extraction_id: str) -> Dict[str, Any]:
        return self._corpus_extraction_engine().verify(extraction_id)

    def cancel_document_extraction(self, extraction_id: str) -> Dict[str, Any]:
        return self._corpus_extraction_engine().cancel(extraction_id)

    def retry_document_extraction(self, extraction_id: str) -> Dict[str, Any]:
        return self._corpus_extraction_engine().retry(
            extraction_id, synchronous=False
        )

    def corpus_profile(self, version_id: str) -> Dict[str, Any]:
        """Stream exact document-level statistics from an immutable corpus version."""

        from collections import Counter

        from halo_forge.own_data.models import CorpusProfile

        db = self._dataset_database()
        version = db.get_dataset_version(version_id)
        if version is None:
            raise KeyError(version_id)
        dataset = db.get_dataset(version.dataset_id)
        if dataset is None:
            raise KeyError(version.dataset_id)
        if str(dataset.canonical_schema).lower() != "corpus":
            raise ValueError("corpus profile requires a corpus dataset version")

        engine = self._dataset_engine()
        store = getattr(engine, "store", None)
        lengths: list[int] = []
        byte_count = 0
        paragraph_count = 0
        content_hashes: Counter[str] = Counter()
        source_types: Counter[str] = Counter()
        seen: set[tuple[str, str]] = set()
        split_counts: Dict[str, int] = {}
        split_names = list((version.split_counts or {}).keys()) or ["train"]

        def rows_for(split: str) -> Iterable[Mapping[str, Any]]:
            if store is not None and callable(getattr(store, "iter_records", None)):
                yield from store.iter_records(
                    version_id,
                    dataset_id=version.dataset_id,
                    split=split,
                )
                return
            offset = 0
            while True:
                page = self.preview_dataset_version(
                    version_id, split=split, offset=offset, limit=500
                )
                rows = list(page.get("items") or [])
                yield from rows
                offset += len(rows)
                if not rows or offset >= int(page.get("total") or offset):
                    break

        for split in split_names:
            split_count = 0
            for record in rows_for(split):
                text = str(record.get("text") or "")
                if not text:
                    continue
                document_id = str(
                    record.get("document_id")
                    or record.get("id")
                    or record.get("document_hash")
                    or ""
                )
                document_hash = str(
                    record.get("document_hash")
                    or record.get("content_hash")
                    or hashlib.sha256(text.encode("utf-8")).hexdigest()
                )
                identity = (document_id, document_hash)
                if identity in seen:
                    continue
                seen.add(identity)
                split_count += 1
                lengths.append(len(text))
                byte_count += len(text.encode("utf-8"))
                paragraph_count += len(
                    [
                        value
                        for value in re.split(r"(?:\r?\n[ \t]*){2,}", text)
                        if value.strip()
                    ]
                )
                content_hashes[document_hash] += 1
                metadata = record.get("metadata")
                source_kind = (
                    record.get("source_kind")
                    or (
                        metadata.get("source_kind")
                        if isinstance(metadata, Mapping)
                        else None
                    )
                    or "unknown"
                )
                source_types[str(source_kind)] += 1
            split_counts[split] = split_count

        ordered = sorted(lengths)

        def percentile(fraction: float) -> int:
            if not ordered:
                return 0
            return ordered[
                min(
                    len(ordered) - 1,
                    max(0, math.ceil(len(ordered) * fraction) - 1),
                )
            ]

        counts = dict(version.statistics.get("counts") or {})
        source = db.get_dataset_source(version.source_id) if version.source_id else None
        extraction = (
            dict(
                (
                    (source.metadata or {}).get("guided_own_data") or {}
                ).get("corpus_extraction")
                or {}
            )
            if source is not None
            else {}
        )
        extraction_failures = int(
            extraction.get("quarantined")
            or extraction.get("quarantined_count")
            or extraction.get("failed")
            or version.statistics.get("extraction_failures")
            or counts.get("extraction_failed")
            or 0
        )
        profile = CorpusProfile(
            document_count=len(lengths),
            character_count=sum(lengths),
            paragraph_count=paragraph_count,
            byte_count=byte_count,
            length_distribution={
                "count": len(lengths),
                "min": ordered[0] if ordered else 0,
                "max": ordered[-1] if ordered else 0,
                "mean": sum(lengths) / len(lengths) if lengths else 0.0,
                "p50": percentile(0.5),
                "p95": percentile(0.95),
            },
            duplicate_documents=sum(
                value - 1 for value in content_hashes.values() if value > 1
            ),
            quarantined_documents=int(counts.get("quarantined") or 0)
            + extraction_failures,
            extraction_failures=extraction_failures,
            source_types=dict(sorted(source_types.items())),
        ).to_dict()
        profile.update(
            {
                "dataset_version_id": version_id,
                "dataset_id": version.dataset_id,
                "content_hash": version.content_hash,
                "split_document_counts": split_counts,
                "exact": True,
            }
        )
        return profile

    def _resolved_cpt_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        resolved = self._normalize_public_training_aliases(payload)
        resolved["mode"] = "cpt"
        resolved = self._normalize_cpt_batch_contract(resolved)
        version_id = str(resolved.get("dataset_version_id") or "").strip()
        if not version_id:
            raise ValueError("dataset_version_id is required for managed CPT")
        resolved["dataset_version_id"] = version_id
        resolved.setdefault("dataset_split", "train")
        adaptation = str(resolved.get("adaptation") or "").strip().lower()
        if adaptation not in {"lora", "full"}:
            raise ValueError("CPT requires an explicit LoRA or full adaptation choice")
        resolved["adaptation"] = adaptation
        resolved.setdefault("max_sequence_length", 2048)
        resolved.setdefault("packing", "paragraph_eos_non_overlap_v1")
        resolved.setdefault("budget_mode", "passes")
        if str(resolved["budget_mode"]) == "passes":
            resolved.setdefault("corpus_passes", 1.0)
        resolved.setdefault("seed", 42)
        output_root = Path(
            str(
                resolved.get("output_root")
                or resolved.get("output_dir")
                or _default_run_root()
            )
        ).expanduser()
        resolved["output_root"] = str(output_root)
        # Preflight checks the selected root. launch_training replaces this
        # with an isolated <root>/<run-id> directory before queueing.
        resolved.setdefault("output_dir", str(output_root))
        return resolved

    def preflight_cpt(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        resolved = self._resolved_cpt_payload(payload)
        result = self.preflight_training(resolved)
        version_id = str(resolved["dataset_version_id"])
        try:
            readiness = self.get_dataset_version_readiness(
                version_id,
                trainer_mode="cpt",
                model=str(resolved.get("model") or ""),
            )
        except (KeyError, ValueError) as exc:
            readiness = {
                "ready": False,
                "blockers": [
                    {
                        "code": "corpus_readiness_failed",
                        "message": str(exc),
                        "remedy": "Open the corpus version and resolve the reported issue.",
                        "action": "open_dataset_version",
                    }
                ],
                "warnings": [],
                "actions": [
                    {
                        "id": "open_dataset_version",
                        "label": "Open corpus version",
                        "action": "open_dataset_version",
                        "target": version_id,
                    }
                ],
            }
        result["readiness"] = readiness
        try:
            result["corpus_profile"] = self.corpus_profile(version_id)
        except (KeyError, ValueError):
            result["corpus_profile"] = None
        artifact = dict(
            result.get("training_artifact")
            or resolved.get("training_artifact_metadata")
            or {}
        )
        if artifact:
            result["training_artifact"] = artifact
            result["packing_plan"] = copy.deepcopy(
                artifact.get("packing_plan")
            )
        if readiness.get("ready") is False:
            result["ok"] = False
            for blocker in readiness.get("blockers") or []:
                message = str(blocker.get("message") or "")
                if message and message not in result.setdefault("errors", []):
                    result["errors"].append(message)
        return result

    async def launch_cpt(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        resolved = self._resolved_cpt_payload(payload)
        return await self.launch_training(resolved)

    def corpus_packing_plan(
        self, version_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        request = {
            **dict(payload),
            "dataset_version_id": version_id,
            "dataset_split": "train",
        }
        result = self.preflight_cpt(request)
        if result.get("status") == "preparing_dataset":
            return result
        plan = result.get("packing_plan")
        if not isinstance(plan, Mapping):
            return {
                "status": "unavailable",
                "ready": False,
                "dataset_version_id": version_id,
                "corpus_profile": result.get("corpus_profile"),
                "errors": result.get("errors") or [
                    "The exact tokenizer-aware packing plan has not been published."
                ],
                "warnings": result.get("warnings") or [],
            }
        return {
            **copy.deepcopy(dict(plan)),
            "status": "ready",
            "ready": True,
            "dataset_version_id": version_id,
            "training_artifact": result.get("training_artifact"),
            "corpus_profile": result.get("corpus_profile"),
        }

    def export_dataset_version(self, version_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        version = self._dataset_database().get_dataset_version(version_id)
        if version is None:
            raise KeyError(version_id)
        output = payload.get("output")
        if not output:
            split_suffix = f"-{payload['split']}" if payload.get("split") else ""
            export_format = str(payload.get("format") or "jsonl").lower()
            output = (
                self.dataset_storage_root
                / "exports"
                / f"{version_id}{split_suffix}.{export_format}"
            )
        result = self._dataset_engine().export(
            version_id,
            output=output,
            format=str(payload.get("format") or "jsonl"),
            split=payload.get("split"),
            dataset_id=version.dataset_id,
        )
        return (
            self._data_object(result)
            if not isinstance(result, (str, Path))
            else {"path": str(result)}
        )

    def materialize_dataset_version(
        self, version_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        version = self._dataset_database().get_dataset_version(version_id)
        if version is None:
            raise KeyError(version_id)
        engine_job = self._dataset_engine().materialize(
            version_id, dataset_id=version.dataset_id, **payload
        )
        data = self._normalize_engine_job_data(self._data_object(engine_job))
        job_id = str(data.get("id") or data.get("job_id"))
        if not job_id:
            raise ValueError("Dataset Lab did not return a materialization job id")
        job = self._dataset_database().get_dataset_job(
            job_id
        ) or self._dataset_database().create_dataset_job(
            job_id=job_id,
            dataset_id=version.dataset_id,
            version_id=version_id,
            job_type="materialize",
            request=payload,
            status=str(data.get("status") or "queued"),
            stage=str(data.get("stage") or "queued"),
            work_item_id=data.get("work_item_id"),
        )
        return job.to_dict()

    def list_training_verifiers(self) -> list[dict[str, Any]]:
        """Verifiers available for the RAFT loop. Each entry exposes the
        toolchain dependency the user needs locally so the configurator
        can preflight whether the binary is reachable.
        """
        from halo_forge.cli import RAFT_TRAIN_SUPPORTED_VERIFIERS

        # Hand-curated metadata that the CLI surface already understands.
        # Kept here (not in cli.py) so the API doesn't pull cli imports
        # at module load.
        catalog: dict[str, dict[str, Any]] = {
            "gcc": {
                "label": "GCC (Linux/POSIX C/C++)",
                "toolchain": "gcc",
                "modality": "code",
                "platforms": ["linux", "macos"],
            },
            "mingw": {
                "label": "MinGW (Windows cross-compile)",
                "toolchain": "x86_64-w64-mingw32-g++",
                "modality": "code",
                "platforms": ["linux", "macos"],
            },
            "msvc": {
                "label": "MSVC (remote Windows host)",
                "toolchain": "remote-msvc",
                "modality": "code",
                "platforms": ["any"],
            },
            "humaneval": {
                "label": "HumanEval (Python)",
                "toolchain": "python",
                "modality": "code",
                "platforms": ["any"],
            },
            "mbpp": {
                "label": "MBPP (Python)",
                "toolchain": "python",
                "modality": "code",
                "platforms": ["any"],
            },
            "rust": {
                "label": "Rust (rustc)",
                "toolchain": "rustc",
                "modality": "code",
                "platforms": ["any"],
            },
            "go": {
                "label": "Go (go build)",
                "toolchain": "go",
                "modality": "code",
                "platforms": ["any"],
            },
            "auto": {
                "label": "Auto-detect",
                "toolchain": "any",
                "modality": "code",
                "platforms": ["any"],
            },
            "execution": {
                "label": "Execution (sandboxed runtime)",
                "toolchain": "sandbox",
                "modality": "code",
                "platforms": ["any"],
            },
        }
        return [
            {"key": k, **catalog.get(k, {"label": k, "toolchain": k, "modality": "code"})}
            for k in RAFT_TRAIN_SUPPORTED_VERIFIERS
        ]

    def list_verifier_catalog(self) -> dict[str, Any]:
        """Inventory of every verifier the runtime can resolve (Track F-O).

        Wraps `halo_forge.rlvr.verifiers.registry.inventory()` and adds
        the plugin directory path so the UI can tell users *where* to
        drop a new `.py` to register one.

        Origin counts are also returned to keep the UI from filtering
        the items list just to render headline metrics.
        """
        from halo_forge.rlvr.verifiers.registry import _plugin_dir, inventory

        items = inventory()
        counts = {"builtin": 0, "user_plugin": 0, "entry_point": 0}
        for entry in items:
            origin = str(entry.get("origin", "builtin"))
            counts[origin] = counts.get(origin, 0) + 1
        return {
            "items": items,
            "counts": counts,
            "plugin_dir": str(_plugin_dir()),
            "total": len(items),
        }

    # ----- Verifier Reliability and Reward Studio (Lab v7) ------------

    def get_verifier_reliability_capabilities(self) -> Dict[str, Any]:
        return self._verifier_engine().capabilities()

    def list_verifier_profiles(
        self,
        *,
        query: Optional[str] = None,
        family: Optional[str] = None,
        modality: Optional[str] = None,
        task_type: Optional[str] = None,
        qualified_only: bool = False,
        include_overridden: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._verifier_engine().list_profiles(
            query=query,
            family=family,
            modality=modality,
            task_type=task_type,
            qualified_only=qualified_only,
            include_overridden=include_overridden,
            limit=limit,
            offset=offset,
        )

    def list_verifier_profile_revisions(
        self, profile_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._verifier_engine().list_profile_revisions(
            profile_id, limit=limit, offset=offset
        )

    def create_verifier_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        definition = dict(payload.get("definition") or payload.get("revision") or {})
        return self._verifier_engine().create_profile(
            name=str(payload.get("name") or ""),
            description=self._optional_str(payload.get("description")),
            definition=definition,
        )

    def validate_verifier_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._verifier_engine().validate_profile_definition(
            dict(payload.get("definition") or payload)
        )

    def get_verifier_profile(self, identifier: str) -> Optional[Dict[str, Any]]:
        return self._verifier_engine().get_profile_detail(identifier)

    def revise_verifier_profile(
        self, profile_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._verifier_engine().revise_profile(
            profile_id,
            definition=dict(payload.get("definition") or payload.get("revision") or payload),
        )

    def list_verifier_protocols(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._verifier_engine().list_protocols(limit=limit, offset=offset)

    def create_verifier_protocol(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._verifier_engine().create_protocol(
            name=str(payload.get("name") or ""),
            description=self._optional_str(payload.get("description")),
            definition=dict(payload.get("definition") or payload.get("revision") or {}),
        )

    def get_verifier_protocol(self, identifier: str) -> Optional[Dict[str, Any]]:
        return self._verifier_engine().get_protocol_detail(identifier)

    def revise_verifier_protocol(
        self, protocol_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._verifier_engine().revise_protocol(
            protocol_id,
            definition=dict(payload.get("definition") or payload.get("revision") or payload),
        )

    def list_verifier_qualification_profiles(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._verifier_engine().list_qualification_profiles(
            limit=limit, offset=offset
        )

    def create_verifier_qualification_profile(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._verifier_engine().create_qualification_profile(
            name=str(payload.get("name") or ""),
            description=self._optional_str(payload.get("description")),
            template_kind=str(payload.get("template_kind") or "human_aligned"),
            requirements=(
                dict(payload["requirements"])
                if isinstance(payload.get("requirements"), Mapping)
                else None
            ),
        )

    def get_verifier_qualification_profile(
        self, identifier: str
    ) -> Optional[Dict[str, Any]]:
        return self._verifier_engine().get_qualification_profile_detail(identifier)

    def revise_verifier_qualification_profile(
        self, profile_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._verifier_engine().revise_qualification_profile(
            profile_id,
            template_kind=str(payload.get("template_kind") or "human_aligned"),
            requirements=(
                dict(payload["requirements"])
                if isinstance(payload.get("requirements"), Mapping)
                else None
            ),
        )

    def list_verifier_calibrations(
        self,
        *,
        verifier_revision_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._verifier_engine().list_calibrations(
            verifier_revision_id=verifier_revision_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    def launch_verifier_calibration(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._verifier_engine().launch_calibration(
            verifier_revision_id=str(payload.get("verifier_profile_revision_id") or ""),
            source_kind=str(payload.get("source_kind") or ""),
            source_revision_id=str(payload.get("source_revision_id") or ""),
            protocol_revision_id=str(payload.get("protocol_revision_id") or ""),
            qualification_profile_revision_id=str(
                payload.get("qualification_profile_revision_id") or ""
            ),
            confirmation=bool(payload.get("confirmation", False)),
            runtime_identity=dict(payload.get("runtime_identity") or {}),
        )

    def get_verifier_calibration(self, calibration_id: str) -> Optional[Dict[str, Any]]:
        return self._verifier_engine().get_calibration_detail(calibration_id)

    def list_verifier_calibration_samples(
        self,
        calibration_id: str,
        *,
        partition: Optional[str] = None,
        outcome: Optional[str] = None,
        perturbation: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._verifier_engine().list_calibration_samples(
            calibration_id,
            partition=partition,
            outcome=outcome,
            perturbation=perturbation,
            query=query,
            limit=limit,
            offset=offset,
        )

    def list_verifier_calibration_metrics(
        self,
        calibration_id: str,
        *,
        partition: Optional[str] = None,
        subgroup: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        calibration = self._verifier_engine().get_calibration(calibration_id)
        if calibration is None:
            raise KeyError(calibration_id)
        values, total, page_limit, page_offset = (
            self._verifier_engine().store.query_metrics(
                calibration_id,
                partition=partition,
                subgroup=subgroup,
                limit=limit,
                offset=offset,
            )
        )
        return {
            "items": [value.to_dict() for value in values],
            "total": total,
            "limit": page_limit,
            "offset": page_offset,
        }

    def cancel_verifier_calibration(self, calibration_id: str) -> Dict[str, Any]:
        return self._verifier_engine().cancel_calibration(calibration_id).to_dict()

    def retry_verifier_calibration(self, calibration_id: str) -> Dict[str, Any]:
        return self._verifier_engine().retry_calibration(calibration_id).to_dict()

    def compare_verifier_calibrations(
        self, base_id: str, candidate_id: str, *, offset: int = 0, limit: int = 100
    ) -> Dict[str, Any]:
        return self._verifier_engine().compare_calibrations(
            base_id, candidate_id, offset=offset, limit=limit
        )

    def verify_verifier_calibration(self, calibration_id: str) -> Dict[str, Any]:
        return self._verifier_engine().verify_calibration(calibration_id)

    def qualify_verifier_calibration(
        self, calibration_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._verifier_engine().qualify_calibration(
            calibration_id,
            scope=str(payload.get("scope") or "development"),
            override_note=self._optional_str(payload.get("override_note")),
        ).to_dict()

    def promote_verifier_revision(
        self, revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._verifier_engine().promote_revision(
            revision_id,
            alias=str(payload.get("alias") or "candidate"),
            override_note=self._optional_str(payload.get("override_note")),
        )

    def verifier_revision_usage(
        self, revision_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._verifier_engine().list_usage(
            revision_id, limit=limit, offset=offset
        )

    def list_verifier_qualification_decisions(
        self,
        *,
        verifier_revision_id: Optional[str] = None,
        calibration_id: Optional[str] = None,
        decision: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._verifier_engine().list_qualification_decisions(
            verifier_revision_id=verifier_revision_id,
            calibration_id=calibration_id,
            decision=decision,
            scope=scope,
            limit=limit,
            offset=offset,
        )

    def list_verifier_alias_history(
        self,
        profile_id: str,
        *,
        alias: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._verifier_engine().list_alias_history(
            profile_id,
            alias=alias,
            limit=limit,
            offset=offset,
        )

    def verifier_runtime_compatibility(
        self, revision_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self._verifier_engine().runtime_compatibility(
            revision_id, (payload or {}).get("runtime_identity")
        )

    # ----- Reward Integrity and Training Signal Studio (Lab v8) --------

    @classmethod
    def _reward_value(cls, value: Any) -> Any:
        return cls._review_value(value)

    @classmethod
    def _reward_page(
        cls, value: Any, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        payload = cls._reward_value(value)
        if isinstance(payload, dict) and "items" in payload:
            payload.setdefault("total", len(payload["items"]))
            payload.setdefault("limit", limit)
            payload.setdefault("offset", offset)
            return payload
        items = list(payload or [])
        return {
            "items": items,
            "total": len(items),
            "limit": limit,
            "offset": offset,
        }

    def get_reward_integrity_capabilities(self) -> Dict[str, Any]:
        value = self._reward_value(self._reward_integrity_engine().capabilities())
        from halo_forge.training_signal import TRAINING_SIGNAL_CAPABILITIES

        value["training_signal_capabilities"] = [
            item.to_dict() for item in TRAINING_SIGNAL_CAPABILITIES.list()
        ]
        return value

    def resolve_reward_integrity_binding(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve the guided Train/Experiments binding without launching work."""

        from halo_forge.training_signal import (
            TRAINING_SIGNAL_CAPABILITIES,
            default_audit_boundaries,
        )

        trainer = str(payload.get("trainer_mode") or payload.get("trainer") or "").lower()
        backend = str(payload.get("backend_family") or payload.get("backend") or "hf").lower()
        total = max(1, int(payload.get("total_budget") or payload.get("cycles") or 1))
        raw_boundaries = list(
            payload.get("audit_boundaries")
            or payload.get("reward_audit_boundaries")
            or default_audit_boundaries(total)
        )
        resolved_boundaries: list[int] = []
        for boundary in raw_boundaries:
            raw = str(boundary).strip().lower()
            value = total if raw in {"final", "last"} else int(raw.rsplit(":", 1)[-1])
            if value not in resolved_boundaries:
                resolved_boundaries.append(value)
        capability = TRAINING_SIGNAL_CAPABILITIES.resolve(trainer, backend)
        resolved = self._reward_integrity_engine().resolve_binding(
            str(payload.get("reward_system_revision_id") or ""),
            protocol_revision_id=str(
                payload.get("reward_audit_protocol_revision_id")
                or payload.get("protocol_revision_id")
                or ""
            ),
            integrity_profile_revision_id=str(
                payload.get("reward_integrity_profile_revision_id")
                or payload.get("integrity_profile_revision_id")
                or ""
            ),
            trainer=trainer,
            backend=backend,
            boundaries=raw_boundaries,
            runtime_identity=None,
        )
        errors = list(resolved.blockers)
        if capability.fidelity.value not in {"exact", "sampled"}:
            errors.append(f"capture_fidelity:{capability.fidelity.value}")
        if not capability.resumable and any(value != total for value in resolved_boundaries):
            errors.append("trainer_supports_final_boundary_only")
        if any(value < 1 or value > total for value in resolved_boundaries):
            errors.append("boundary_out_of_range")
        primary = resolved.reward_system_revision.primary_sentinel
        return {
            "reward_system_revision_id": resolved.reward_system_revision.id,
            "reward_audit_protocol_revision_id": resolved.protocol_revision.id,
            "reward_integrity_profile_revision_id": resolved.integrity_profile_revision.id,
            "audit_boundaries": raw_boundaries,
            "reward_system_hash": resolved.reward_system_revision.content_hash,
            "optimizer_verifier_profile_revision_id": (
                resolved.reward_system_revision.optimizer_verifier_revision_id
            ),
            "primary_sentinel_verifier_profile_revision_id": (
                primary.verifier_revision_id if primary else None
            ),
            "capability": capability.to_dict(),
            "capture_fidelity": capability.fidelity.value,
            "boundary_unit": capability.boundary_unit,
            "resolved_boundaries": resolved_boundaries,
            "ready": not errors,
            "warnings": [],
            "errors": sorted(set(errors)),
        }

    def list_reward_systems(
        self,
        *,
        query: Optional[str] = None,
        modality: Optional[str] = None,
        task_type: Optional[str] = None,
        trainer_mode: Optional[str] = None,
        backend_family: str = "hf",
        qualified_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._reward_page(
            self._reward_integrity_engine().list_systems(
                query=query,
                modality=modality,
                task_type=task_type,
                trainer=trainer_mode,
                backend=backend_family,
                qualified_only=qualified_only,
                limit=limit,
                offset=offset,
            ),
            limit=limit,
            offset=offset,
        )

    def create_reward_system(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        definition = dict(payload.get("definition") or payload.get("revision") or {})
        validation = self._reward_integrity_engine().validate_system_definition(
            definition
        )
        blockers = [
            str(value)
            for value in validation.get("blockers", [])
            if not str(value).startswith("primary_sentinel_correlated:")
        ]
        if blockers:
            raise ValueError("invalid reward system: " + "; ".join(blockers))
        value = self._reward_integrity_engine().create_system(
            name=str(payload.get("name") or ""),
            description=self._optional_str(payload.get("description")),
            definition=definition,
        )
        return self._reward_value(value)

    def validate_reward_system(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._reward_value(
            self._reward_integrity_engine().validate_system_definition(
                dict(payload.get("definition") or payload)
            )
        )

    def get_reward_system(self, identifier: str) -> Optional[Dict[str, Any]]:
        engine = self._reward_integrity_engine()
        detail = getattr(engine, "get_system_detail", None)
        if detail is not None:
            value = detail(identifier)
        else:
            value = engine.get_system(identifier) or engine.get_system_revision(identifier)
        return None if value is None else self._reward_value(value)

    def revise_reward_system(
        self, system_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        definition = dict(
            payload.get("definition") or payload.get("revision") or payload
        )
        validation = self._reward_integrity_engine().validate_system_definition(
            definition
        )
        blockers = [
            str(value)
            for value in validation.get("blockers", [])
            if not str(value).startswith("primary_sentinel_correlated:")
        ]
        if blockers:
            raise ValueError("invalid reward system: " + "; ".join(blockers))
        raw = dict(definition)
        return self._reward_value(
            self._reward_integrity_engine().create_system_revision(
                system_id,
                optimizer_verifier_revision_id=str(
                    raw.pop("optimizer_verifier_revision_id")
                ),
                modality=str(raw.pop("modality", "text")),
                task_type=str(raw.pop("task_type", "binary")),
                auditors=list(raw.pop("auditors", [])),
                input_mapping=dict(raw.pop("input_mapping", {})),
                reward_mapping=dict(raw.pop("reward_mapping", {})),
                definition=raw,
            )
        )

    def list_reward_system_usage(
        self, revision_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._reward_page(
            self._reward_integrity_engine().list_usage(
                revision_id, limit=limit, offset=offset
            ),
            limit=limit,
            offset=offset,
        )

    def list_reward_protocols(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._reward_page(
            self._reward_integrity_engine().list_protocols(limit=limit, offset=offset),
            limit=limit,
            offset=offset,
        )

    def create_reward_protocol(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._reward_value(
            self._reward_integrity_engine().create_protocol(
                name=str(payload.get("name") or ""),
                description=self._optional_str(payload.get("description")),
                definition=dict(payload.get("definition") or payload.get("revision") or {}),
            )
        )

    def get_reward_protocol(self, identifier: str) -> Optional[Dict[str, Any]]:
        engine = self._reward_integrity_engine()
        detail = getattr(engine, "get_protocol_detail", None)
        if detail is not None:
            value = detail(identifier)
            return None if value is None else self._reward_value(value)
        try:
            value = engine.get_protocol(identifier)
        except KeyError:
            try:
                value = engine.get_protocol_revision(identifier)
            except KeyError:
                return None
        return self._reward_value(value)

    def revise_reward_protocol(
        self, protocol_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._reward_value(
            self._reward_integrity_engine().create_protocol_revision(
                protocol_id,
                definition=dict(payload.get("definition") or payload.get("revision") or payload),
            )
        )

    def list_reward_integrity_profiles(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._reward_page(
            self._reward_integrity_engine().list_integrity_profiles(
                limit=limit, offset=offset
            ),
            limit=limit,
            offset=offset,
        )

    def create_reward_integrity_profile(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        template_kind = str(
            payload.get("template_kind") or "human_aligned_integrity"
        )
        requirements = (
            dict(payload["requirements"])
            if isinstance(payload.get("requirements"), Mapping)
            else None
        )
        if requirements is None:
            if template_kind == "custom":
                raise ValueError("custom reward integrity profiles require requirements")
            from halo_forge.reward_integrity import PROFILE_DEFAULTS

            requirements = dict(PROFILE_DEFAULTS[template_kind])
        return self._reward_value(
            self._reward_integrity_engine().create_integrity_profile(
                name=str(payload.get("name") or ""),
                description=self._optional_str(payload.get("description")),
                template_kind=template_kind,
                requirements=requirements,
            )
        )

    def get_reward_integrity_profile(
        self, identifier: str
    ) -> Optional[Dict[str, Any]]:
        engine = self._reward_integrity_engine()
        detail = getattr(engine, "get_integrity_profile_detail", None)
        if detail is not None:
            value = detail(identifier)
            return None if value is None else self._reward_value(value)
        try:
            value = engine.get_integrity_profile(identifier)
        except KeyError:
            try:
                value = engine.get_integrity_profile_revision(identifier)
            except KeyError:
                return None
        return self._reward_value(value)

    def revise_reward_integrity_profile(
        self, profile_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        template_kind = str(
            payload.get("template_kind") or "human_aligned_integrity"
        )
        requirements = (
            dict(payload["requirements"])
            if isinstance(payload.get("requirements"), Mapping)
            else None
        )
        if requirements is None:
            if template_kind == "custom":
                raise ValueError("custom reward integrity profiles require requirements")
            from halo_forge.reward_integrity import PROFILE_DEFAULTS

            requirements = dict(PROFILE_DEFAULTS[template_kind])
        return self._reward_value(
            self._reward_integrity_engine().create_integrity_profile_revision(
                profile_id,
                template_kind=template_kind,
                requirements=requirements,
            )
        )

    def list_training_signal_shards(
        self,
        *,
        run_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._reward_page(
            self._reward_integrity_engine().list_signal_shards(
                run_id=run_id, limit=limit, offset=offset
            ),
            limit=limit,
            offset=offset,
        )

    def get_training_signal_shard(self, shard_id: str) -> Optional[Dict[str, Any]]:
        value = self._reward_integrity_engine().get_signal_shard(shard_id)
        return None if value is None else self._reward_value(value)

    def verify_training_signal_shard(self, shard_id: str) -> Dict[str, Any]:
        return self._reward_value(
            self._reward_integrity_engine().verify_signal_shard(shard_id)
        )

    def list_reward_integrity_audits(
        self,
        *,
        run_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        page = self._reward_page(
            self._reward_integrity_engine().list_audits(
                run_id=run_id, status=status, limit=limit, offset=offset
            ),
            limit=limit,
            offset=offset,
        )
        engine = self._reward_integrity_engine()
        for item in page["items"]:
            decisions = engine.store.list_decisions(str(item["id"]), limit=1000)
            item["decision"] = (
                self._reward_value(decisions.items[-1]) if decisions.items else None
            )
        return page

    def launch_reward_integrity_audit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        engine = self._reward_integrity_engine()
        shard_id = str(payload.get("signal_shard_id") or payload.get("trace_id") or "")
        shard = engine.get_signal_shard(shard_id)
        supplied_system = self._optional_str(payload.get("reward_system_revision_id"))
        supplied_protocol = self._optional_str(payload.get("protocol_revision_id"))
        if supplied_system and supplied_system != shard.reward_system_revision_id:
            raise ValueError("reward_system_revision_id conflicts with the sealed trace")
        if supplied_protocol and supplied_protocol != shard.protocol_revision_id:
            raise ValueError("protocol_revision_id conflicts with the sealed trace")
        # ``reward_integrity_audits.work_item_id`` is a foreign key.  Create
        # and hydrate the domain record first, then enqueue and attach the
        # durable work item; inserting the future work-item id here would fail
        # under SQLite foreign-key enforcement.
        work_item_id = f"reward-audit-work-{uuid.uuid4().hex}"
        value = engine.create_audit(
            signal_shard_id=shard_id,
            integrity_profile_revision_id=str(
                payload.get("integrity_profile_revision_id") or ""
            ),
            development_suite_revision_id=self._optional_str(
                payload.get("development_suite_revision_id")
            ),
            runtime_identity=dict(payload.get("runtime_identity") or {}),
            request=dict(payload.get("request") or {}),
            submit=False,
        )
        if value.status == "completed":
            engine.verify_audit_bundle(value.id)
            return {
                **self._reward_value(value),
                "work_item_id": value.work_item_id,
                "reused": True,
            }
        resources = engine.audit_resource_requirements(
            shard.reward_system_revision_id
        )
        resource_class = str(resources["resource_class"])
        work = self._scheduler().enqueue(
            kind="reward_integrity_audit",
            launch_spec={
                "handler": "reward_integrity.execute_audit",
                "operation": "reward_integrity_audit",
                "audit_id": value.id,
                "reward_integrity_root": str(self.reward_integrity_storage_root),
            },
            resource_class=resource_class,
            resource_requirements=resources,
            domain_kind="reward_integrity_audit",
            domain_id=value.id,
            canonical_run_id=value.run_id,
            max_retries=1,
            work_item_id=work_item_id,
        )
        value = engine.store.update_audit(value.id, work_item_id=work.id)
        return {**self._reward_value(value), "work_item_id": work.id}

    def get_reward_integrity_audit(self, audit_id: str) -> Optional[Dict[str, Any]]:
        value = self._reward_integrity_engine().get_audit_detail(audit_id)
        return None if value is None else self._reward_value(value)

    def list_reward_integrity_samples(
        self, audit_id: str, *, limit: int = 100, offset: int = 0, **filters: Any
    ) -> Dict[str, Any]:
        return self._reward_page(
            self._reward_integrity_engine().list_audit_samples(
                audit_id, limit=limit, offset=offset, **filters
            ),
            limit=limit,
            offset=offset,
        )

    def list_reward_integrity_metrics(
        self, audit_id: str, *, limit: int = 100, offset: int = 0, **filters: Any
    ) -> Dict[str, Any]:
        return self._reward_page(
            self._reward_integrity_engine().list_audit_metrics(
                audit_id, limit=limit, offset=offset, **filters
            ),
            limit=limit,
            offset=offset,
        )

    def compare_reward_integrity_audits(
        self, left_id: str, right_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._reward_value(
            self._reward_integrity_engine().compare_audits(
                left_id, right_id, limit=limit, offset=offset
            )
        )

    def verify_reward_integrity_audit(self, audit_id: str) -> Dict[str, Any]:
        return self._reward_value(
            self._reward_integrity_engine().verify_audit_bundle(audit_id)
        )

    def cancel_reward_integrity_audit(self, audit_id: str) -> Dict[str, Any]:
        return self._reward_value(self._reward_integrity_engine().cancel_audit(audit_id))

    def retry_reward_integrity_audit(
        self, audit_id: str, payload: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        reason = str((payload or {}).get("reason") or "").strip()
        if not reason:
            raise ValueError("an operator-forced reward-audit retry requires a reason")
        return self._reward_value(
            self._reward_integrity_engine().retry_audit(audit_id, reason=reason)
        )

    def _reward_audit_checkpoint_context(
        self, audit: Any, shard: Any
    ) -> Dict[str, Any]:
        """Resolve the checkpoint published at an audited boundary.

        A trace hash identifies the exact checkpoint bytes, but a usable fork
        also needs the atomically published local path.  Prefer an indexed
        Artifact Studio occurrence.  Older V8 attempts may only have the
        boundary snapshot and rewritten training summary, so retain that
        truthful fallback without pretending an occurrence exists.
        """

        database = self._dataset_database()
        segment = (
            database.get_direct_run_segment(str(audit.direct_run_segment_id))
            if audit.direct_run_segment_id
            else None
        )
        preferred_occurrence = str(
            (segment or {}).get("checkpoint_occurrence_id") or ""
        )
        occurrence_row = database._conn.execute(
            """
            SELECT occurrence.id AS occurrence_id,
                   occurrence.blob_id AS blob_id,
                   occurrence.artifact_kind AS artifact_kind,
                   blob.content_hash AS content_hash,
                   location.path AS path,
                   location.storage_mode AS storage_mode
              FROM artifact_occurrences occurrence
              JOIN artifact_blobs blob ON blob.id=occurrence.blob_id
              JOIN artifact_locations location ON location.blob_id=blob.id
             WHERE occurrence.run_id=?
               AND blob.content_hash=?
               AND location.state='available'
             ORDER BY CASE WHEN occurrence.id=? THEN 0 ELSE 1 END,
                      CASE location.storage_mode WHEN 'managed' THEN 0 ELSE 1 END,
                      occurrence.created_at DESC,
                      location.created_at DESC
             LIMIT 1
            """,
            (audit.run_id, shard.checkpoint_hash, preferred_occurrence),
        ).fetchone()

        work = None
        if segment and segment.get("work_item_id"):
            work = database.get_work_item(str(segment["work_item_id"]))
        launch_spec = dict(work.launch_spec) if work is not None else {}
        final_segment = bool(launch_spec.get("final_segment"))
        publication_value = (
            launch_spec.get("output_dir")
            if final_segment
            else launch_spec.get("segment_output_dir")
        )
        if not publication_value:
            run = database.get_run(str(audit.run_id))
            publication_value = run.output_dir if run is not None else None
        publication_root = (
            Path(str(publication_value)).expanduser().resolve()
            if publication_value
            else None
        )
        snapshot_path = (
            str(publication_root)
            if publication_root is not None and publication_root.is_dir()
            else None
        )

        checkpoint_path: Optional[str] = None
        occurrence: Optional[Dict[str, Any]] = None
        if occurrence_row is not None:
            indexed_path = Path(str(occurrence_row["path"])).expanduser()
            if indexed_path.exists():
                checkpoint_path = str(indexed_path.resolve())
                occurrence = {
                    "id": str(occurrence_row["occurrence_id"]),
                    "blob_id": str(occurrence_row["blob_id"]),
                    "artifact_kind": str(occurrence_row["artifact_kind"]),
                    "content_hash": str(occurrence_row["content_hash"]),
                    "path": checkpoint_path,
                    "storage_mode": str(occurrence_row["storage_mode"]),
                }

        # Direct-run snapshots rewrite attempt-local summary paths before
        # atomic publication.  This lets legacy V8 boundaries remain forkable
        # even when they predate checkpoint-occurrence indexing.
        if checkpoint_path is None and publication_root is not None:
            summary_path = publication_root / "training_summary.json"
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                summary = {}
            for value in (
                summary.get("final_model_path"),
                summary.get("checkpoint_path"),
            ):
                if not value:
                    continue
                candidate = Path(str(value)).expanduser()
                try:
                    resolved = candidate.resolve(strict=True)
                    resolved.relative_to(publication_root)
                except (OSError, ValueError):
                    continue
                checkpoint_path = str(resolved)
                break

        blockers: List[str] = []
        if shard.checkpoint_hash in {"", "unavailable"}:
            blockers.append("checkpoint_content_identity_unavailable")
        return {
            "content_hash": shard.checkpoint_hash,
            "path": checkpoint_path,
            "occurrence_id": occurrence["id"] if occurrence else None,
            "artifact": occurrence,
            "snapshot_path": snapshot_path,
            "boundary_unit": shard.boundary_unit,
            "boundary_value": shard.boundary_value,
            "segment_id": audit.direct_run_segment_id or audit.trial_segment_id,
            "integrity_source": (
                "verified_artifact_blob" if occurrence else "sealed_trace_and_atomic_summary"
            ),
            "blockers": blockers,
        }

    def get_reward_integrity_fork_context(self, audit_id: str) -> Dict[str, Any]:
        """Return an immutable, server-resolved Train context for a fork review."""

        engine = self._reward_integrity_engine()
        audit = engine.get_audit(audit_id)
        decisions = engine.store.list_decisions(audit_id, limit=1000).items
        if not decisions or decisions[-1].action != "fork":
            raise ValueError(
                "reward audit must have a latest reviewed Fork decision before opening Train"
            )
        decision = decisions[-1]
        shard = engine.get_signal_shard(audit.signal_shard_id)
        checkpoint = self._reward_audit_checkpoint_context(audit, shard)
        try:
            resolved = self.get_resolved_run_launch_config(audit.run_id)
            config = dict(resolved.get("resolved_config") or resolved.get("config") or {})
            datasets = list(resolved.get("datasets") or [])
        except Exception:
            # Durable managed work retains the exact normalized config even if
            # the in-memory job index has not yet been rebuilt after restart.
            work_items = self._dataset_database().list_work_items(
                canonical_run_id=audit.run_id, kinds=["training"], limit=100
            )
            source_work = next(
                (
                    item
                    for item in reversed(work_items)
                    if (item.launch_spec.get("resolved_launch_config") or {}).get("mode")
                ),
                None,
            )
            config = dict(
                (source_work.launch_spec.get("resolved_launch_config") or {})
                if source_work is not None
                else {}
            )
            bindings = self._dataset_database().list_run_datasets(audit.run_id)
            datasets = [self._run_dataset_view(value) for value in bindings]
            if bindings:
                config["dataset_bindings"] = [value.to_dict() for value in bindings]
        config.pop("run_id", None)

        from halo_forge.training_signal import TRAINING_SIGNAL_CAPABILITIES

        try:
            capability = TRAINING_SIGNAL_CAPABILITIES.get(shard.capability_id)
            capability_value = capability.to_dict()
        except KeyError:
            capability = None
            capability_value = None
        resume_mode = (
            "resume_boundary"
            if capability is not None
            and capability.resumable
            and checkpoint.get("snapshot_path")
            else "initialize_from_checkpoint"
        )
        blockers = list(checkpoint.get("blockers") or [])
        if resume_mode == "resume_boundary":
            if not checkpoint.get("snapshot_path"):
                blockers.append("checkpoint_snapshot_missing")
            if not checkpoint.get("path"):
                blockers.append("checkpoint_path_unavailable")
        elif not checkpoint.get("path"):
            blockers.append("checkpoint_path_unavailable")
        if not config.get("mode"):
            blockers.append("parent_launch_config_unavailable")

        train_context = {
            **config,
            "parent_run_id": audit.run_id,
            "source_reward_integrity_audit_id": audit.id,
            "source_reward_integrity_decision_id": decision.id,
            "reward_system_revision_id": audit.reward_system_revision_id,
            "reward_audit_protocol_revision_id": audit.protocol_revision_id,
            "reward_integrity_profile_revision_id": audit.integrity_profile_revision_id,
            "fork_checkpoint_hash": checkpoint.get("content_hash"),
            "fork_checkpoint_path": checkpoint.get("path"),
            "fork_checkpoint_occurrence_id": checkpoint.get("occurrence_id"),
            "fork_checkpoint_snapshot_path": checkpoint.get("snapshot_path"),
            "fork_boundary_unit": checkpoint.get("boundary_unit"),
            "fork_boundary_value": checkpoint.get("boundary_value"),
            "fork_resume_mode": resume_mode,
            # Backward-compatible display field. Execution uses the explicit,
            # server-validated fork fields above.
            "checkpoint": checkpoint.get("path") or checkpoint.get("content_hash"),
        }
        return {
            "audit_id": audit.id,
            "decision": self._reward_value(decision),
            "parent_run_id": audit.run_id,
            "checkpoint": checkpoint,
            "reward_system_revision_id": audit.reward_system_revision_id,
            "reward_audit_protocol_revision_id": audit.protocol_revision_id,
            "reward_integrity_profile_revision_id": audit.integrity_profile_revision_id,
            "signal_capability": capability_value,
            "resume_mode": resume_mode,
            "train_context": train_context,
            "datasets": datasets,
            "launch_ready": not blockers,
            "blockers": sorted(set(blockers)),
            "href": "/train?fork_reward_audit=" + audit.id,
        }

    def review_reward_integrity_audit(
        self, audit_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        engine = self._reward_integrity_engine()
        action = str(payload.get("action") or "")
        audit = engine.get_audit(audit_id)
        checkpoint = self._optional_str(payload.get("checkpoint_hash"))
        if checkpoint is None and action == "fork":
            checkpoint = engine.get_signal_shard(audit.signal_shard_id).checkpoint_hash
        reviewed = engine.review_audit(
            audit_id,
            action=action,
            reason=str(payload.get("reason") or ""),
            checkpoint=checkpoint,
        )
        result = self._reward_value(reviewed)
        replay_sync: Optional[Dict[str, Any]] = None
        if action in {"continue", "stop", "fork"}:
            try:
                replay_sync = self._reward_value(
                    engine.sync_audit_replay(audit_id, decision=reviewed)
                )
            except Exception as exc:
                # The review decision is append-only and has already been
                # committed. Report publication failure without inviting a
                # duplicate operator action on an HTTP retry.
                replay_sync = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        next_work_item = None
        if action == "continue" and audit.direct_run_segment_id:
            from halo_forge.reward_integrity.direct_segments import (
                enqueue_next_direct_segment,
            )

            next_work_item = enqueue_next_direct_segment(
                self._dataset_database(),
                self._scheduler(),
                current_segment_id=audit.direct_run_segment_id,
                dependency_work_item_id=audit.work_item_id,
            )
        if action == "fork":
            fork_context = self.get_reward_integrity_fork_context(audit.id)
            result = {**fork_context, "decision": result}
        if replay_sync is not None:
            result["replay_sync"] = replay_sync
        if next_work_item is not None:
            result["next_work_item_id"] = next_work_item.id
        return result

    def list_model_catalog(
        self,
        *,
        mode: Optional[str] = None,
        backend: Optional[str] = None,
        modality: Optional[str] = None,
        provider: Optional[str] = None,
        status: Optional[str] = None,
        memory_tier: Optional[str] = None,
    ) -> dict[str, Any]:
        """Curated upstream/base-model catalog for docs, UI, and CLI parity."""
        from halo_forge.models.catalog import CATALOG_VERSION, catalog_facets, list_models

        filters = {
            "mode": mode,
            "backend": backend,
            "modality": modality,
            "provider": provider,
            "status": status,
            "memory_tier": memory_tier,
        }
        items = list_models({k: v for k, v in filters.items() if v})
        return {
            "catalog_version": CATALOG_VERSION,
            "items": items,
            "total": len(items),
            "facets": catalog_facets(items),
            "filters": {k: v for k, v in filters.items() if v},
        }

    def list_suggested_models(
        self,
        *,
        mode: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Backend-aware base-model suggestions.

        Sourced from the curated catalog so training quick-picks, docs,
        and CLI recommendations stay aligned.
        """
        from halo_forge.backend import get_backend
        from halo_forge.models.catalog import recommended_models

        backend_name = get_backend().name
        return recommended_models(mode=mode, backend=backend_name, modality=modality)

    # ---- V18 guided training plans -------------------------------------

    def training_plan_capabilities(self) -> Dict[str, Any]:
        return self._training_plan_engine().capabilities()

    def managed_runtime_capabilities(self) -> Dict[str, Any]:
        engine = self._managed_runtime_engine()
        items = [value.to_dict() for value in engine.capabilities()]
        return {"items": items, "total": len(items), "limit": len(items), "offset": 0}

    def list_managed_runtimes(self) -> Dict[str, Any]:
        engine = self._managed_runtime_engine()
        profiles = []
        for profile in engine.list_profiles():
            revision = engine.get_revision(str(profile.latest_revision_id)) if profile.latest_revision_id else None
            qualification = engine.latest_qualification(revision.id) if revision else None
            profiles.append(
                {
                    **profile.to_dict(),
                    "revision": revision.to_dict() if revision else None,
                    "qualification": qualification.to_dict() if qualification else None,
                    "preparations": [value.to_dict() for value in engine.list_preparations(revision.id if revision else None)[:5]],
                }
            )
        return {"items": profiles, "total": len(profiles), "limit": len(profiles), "offset": 0}

    def get_managed_runtime(self, identifier: str) -> Dict[str, Any]:
        engine = self._managed_runtime_engine()
        revision = engine.get_revision(identifier)
        profile = engine.get_profile(identifier)
        if revision is None and profile is not None and profile.latest_revision_id:
            revision = engine.get_revision(str(profile.latest_revision_id))
        if revision is None:
            raise KeyError(identifier)
        return {
            "profile": engine.get_profile(revision.profile_id).to_dict(),
            "revision": revision.to_dict(),
            "preparations": [value.to_dict() for value in engine.list_preparations(revision.id)],
            "qualifications": [value.to_dict() for value in engine.list_qualifications(revision.id)],
            "capability": next(
                (value.to_dict() for value in engine.capabilities() if value.runtime_revision_id == revision.id),
                None,
            ),
        }

    def prepare_managed_runtime(self, revision_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        value = self._managed_runtime_engine().prepare(
            revision_id,
            confirmed=bool(payload.get("confirmed")),
            enqueue=True,
        )
        return value.to_dict()

    def qualify_managed_runtime(self, revision_id: str) -> Dict[str, Any]:
        return self._managed_runtime_engine().qualify(revision_id, enqueue=True).to_dict()

    def get_runtime_preparation(self, identifier: str) -> Dict[str, Any]:
        value = self._managed_runtime_engine().get_preparation(identifier)
        if value is None:
            raise KeyError(identifier)
        return value.to_dict()

    def get_runtime_qualification(self, identifier: str) -> Dict[str, Any]:
        value = self._managed_runtime_engine().get_qualification(identifier)
        if value is None:
            raise KeyError(identifier)
        return value.to_dict()

    def verify_runtime_qualification(self, identifier: str) -> Dict[str, Any]:
        return self._managed_runtime_engine().verify(identifier)

    def cancel_runtime_work(self, kind: str, identifier: str) -> Dict[str, Any]:
        value = self._managed_runtime_engine().cancel(kind, identifier)
        return value.to_dict()

    def retry_runtime_work(
        self, kind: str, identifier: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            raise ValueError("A retry reason is required")
        engine = self._managed_runtime_engine()
        value = (
            engine.get_preparation(identifier)
            if kind == "preparation"
            else engine.get_qualification(identifier)
        )
        if value is None or not value.work_item_id:
            raise KeyError(identifier)
        table = "runtime_preparations" if kind == "preparation" else "runtime_qualifications"
        self._dataset_database()._conn.execute(
            f"UPDATE {table} SET status='queued',stage='queued',error=NULL,cancel_requested=0 WHERE id=?",
            (identifier,),
        )
        self._dataset_database()._conn.commit()
        work = self._scheduler().retry(value.work_item_id, force=True, reason=reason)
        if work is None:
            raise ValueError("Runtime work is not retryable in its current state")
        refreshed = (
            engine.get_preparation(identifier)
            if kind == "preparation"
            else engine.get_qualification(identifier)
        )
        return refreshed.to_dict()

    def accelerator_availability(self, family: str) -> Dict[str, Any]:
        if family not in {"rocm", "cuda"}:
            raise ValueError("accelerator family must be rocm or cuda")
        return self._managed_runtime_engine().availability(family).to_dict()

    # ---- V21 real training-path certification -------------------------

    def training_path_capabilities(self, family: str) -> Dict[str, Any]:
        if family not in {"rocm", "cuda"}:
            raise ValueError("runtime family must be rocm or cuda")
        return self._training_path_engine().capabilities(family).to_dict()

    def launch_training_path_certification(
        self, revision_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        runtime_revision_id = str(payload.get("runtime_profile_revision_id") or "").strip()
        if not runtime_revision_id:
            raise ValueError("runtime_profile_revision_id is required")
        value = self._training_path_engine().certify(
            revision_id, runtime_revision_id, enqueue=True
        )
        return value.to_dict()

    def preview_training_path_certification(
        self, revision_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        runtime_revision_id = str(payload.get("runtime_profile_revision_id") or "").strip()
        if not runtime_revision_id:
            raise ValueError("runtime_profile_revision_id is required")
        return self._training_path_engine().preview(
            revision_id, runtime_revision_id
        )

    def get_training_path_certification(self, identifier: str) -> Dict[str, Any]:
        value = self._training_path_engine().get_certification(identifier)
        if value is None:
            raise KeyError(identifier)
        return value.to_dict()

    def list_training_path_certification_steps(
        self, identifier: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        value = self._training_path_engine().get_certification(identifier)
        if value is None:
            raise KeyError(identifier)
        items = [step.to_dict() for step in value.steps]
        page = items[offset : offset + min(500, limit)]
        return {"items": page, "total": len(items), "limit": limit, "offset": offset}

    def verify_training_path_certification(self, identifier: str) -> Dict[str, Any]:
        return self._training_path_engine().verify(identifier)

    def cancel_training_path_certification(self, identifier: str) -> Dict[str, Any]:
        return self._training_path_engine().cancel(identifier).to_dict()

    def retry_training_path_certification(
        self, identifier: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._training_path_engine().retry(
            identifier, reason=str(payload.get("reason") or "")
        ).to_dict()

    def training_path_certification_evidence(self, identifier: str) -> Dict[str, Any]:
        value = self._training_path_engine().get_certification(identifier)
        if value is None:
            raise KeyError(identifier)
        verification = self._training_path_engine().verify(identifier)
        if not value.evidence_path or not Path(value.evidence_path).is_file():
            raise ValueError("Certification evidence is not published")
        return {
            "certification_id": identifier,
            "verification": verification,
            "bundle": json.loads(Path(value.evidence_path).read_text(encoding="utf-8")),
        }

    def workstation_certify(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        runtime_revision_id = str(payload.get("runtime_profile_revision_id") or "").strip()
        if not runtime_revision_id:
            raise ValueError("runtime_profile_revision_id is required")
        return self._training_path_engine().workstation_certify(
            runtime_revision_id,
            evidence=dict(payload.get("evidence") or {}),
            enqueue=True,
        ).to_dict()

    def workstation_certification_report(self, identifier: str) -> Dict[str, Any]:
        value = self._training_path_engine().get_workstation_certification(identifier)
        if value is None:
            raise KeyError(identifier)
        return value.to_dict()

    def verify_workstation_certification(self, identifier: str) -> Dict[str, Any]:
        return self._training_path_engine().verify_workstation_certification(identifier)

    def recommend_training_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._training_plan_engine().recommend(payload).to_dict()

    def list_training_plans(self, *, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        return self._training_plan_engine().list_plans(limit=limit, offset=offset)

    def get_training_plan(self, plan_id: str) -> Dict[str, Any]:
        plan = self._training_plan_engine().get_plan(plan_id)
        if plan is None:
            raise KeyError(plan_id)
        revision = (
            self._training_plan_engine().get_revision(plan.latest_revision_id)
            if plan.latest_revision_id
            else None
        )
        return {
            "plan": plan.to_dict(),
            "revision": revision.to_dict() if revision else None,
            "readiness": (
                self._training_plan_engine().readiness(revision.id).to_dict()
                if revision
                else None
            ),
        }

    def get_training_plan_revision(self, revision_id: str) -> Dict[str, Any]:
        revision = self._training_plan_engine().get_revision(revision_id)
        if revision is None:
            raise KeyError(revision_id)
        return revision.to_dict()

    def get_training_plan_alternatives(self, revision_id: str) -> Dict[str, Any]:
        return self._training_plan_engine().alternatives(revision_id)

    def choose_training_plan_alternative(
        self, revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        return self._training_plan_engine().choose_alternative(
            revision_id,
            str(payload.get("model_id") or ""),
            reason=str(payload.get("reason") or "Operator selected another compatible model"),
        ).to_dict()

    def confirm_training_plan(self, revision_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        decision = self._training_plan_engine().record_decision(
            revision_id,
            "confirmed",
            reason=self._optional_str(payload.get("reason")),
            details={
                "download_confirmed": bool(payload.get("download_confirmed", False)),
                "hosted_provider_confirmed": bool(
                    payload.get("hosted_provider_confirmed", False)
                ),
            },
        )
        return decision.to_dict()

    def prepare_training_plan_model(
        self, revision_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        request = dict(payload or {})
        if not bool(request.get("download_confirmed", False)):
            raise ValueError("Confirm the displayed model download before preparation")
        if not self._training_plan_engine().is_confirmed(
            revision_id, require_download=True
        ):
            raise ValueError(
                "Confirm the recommended plan and displayed model download before preparation"
            )
        return self._training_plan_engine().prepare_model(
            revision_id,
            enqueue=True,
            allow_download=True,
        ).to_dict()

    def get_model_preparation(self, preparation_id: str) -> Dict[str, Any]:
        value = self._training_plan_engine().get_model_preparation(preparation_id)
        if value is None:
            raise KeyError(preparation_id)
        return value.to_dict()

    def create_training_capacity_check(self, revision_id: str) -> Dict[str, Any]:
        return self._training_plan_engine().create_capacity_check(
            revision_id, enqueue=True
        ).to_dict()

    def get_training_capacity_check(self, check_id: str) -> Dict[str, Any]:
        value = self._training_plan_engine().get_capacity_check(check_id)
        if value is None:
            raise KeyError(check_id)
        return value.to_dict()

    def list_training_capacity_attempts(self, check_id: str) -> Dict[str, Any]:
        return self._training_plan_engine().list_capacity_attempts(check_id)

    def get_training_plan_readiness(self, revision_id: str) -> Dict[str, Any]:
        return self._training_plan_engine().readiness(revision_id).to_dict()

    def cancel_training_plan_work(self, domain_kind: str, domain_id: str) -> Dict[str, Any]:
        return self._training_plan_engine().cancel(domain_kind, domain_id)

    def retry_training_plan_work(
        self, domain_kind: str, domain_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        request = dict(payload or {})
        return self._training_plan_engine().retry(
            domain_kind,
            domain_id,
            reason=str(request.get("reason") or "Operator requested retry after reviewing the failure"),
        )

    async def launch_training_plan_proof(
        self, revision_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        engine = self._training_plan_engine()
        request = dict(payload or {})
        request["training_plan_revision_id"] = revision_id
        request.setdefault("output_root", str(_default_run_root()))
        result = await self.launch_training(request)
        run_id = str(result.get("run_id") or "")
        if (
            run_id
            and result.get("managed") is True
            and result.get("accepted") is True
            and result.get("work_item_id")
        ):
            decision = engine.record_decision(
                revision_id,
                "proof_launched",
                details={
                    "run_id": run_id,
                    "work_item_id": result["work_item_id"],
                },
            )
            result["training_plan_decision_id"] = decision.id
        return result

    def _resolve_public_training_plan_payload(
        self, revision_id: str, payload: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """Resolve once, then expose only executable and replay-safe fields."""

        resolved = self._training_plan_engine().resolved_launch_payload(
            revision_id, payload
        )
        mode = str(resolved.get("mode") or "").strip().lower()
        allowed = PUBLIC_TRAIN_ALLOWED_FIELDS.get(mode)
        if allowed is None:
            raise ValueError(f"Unsupported training mode: {mode}")
        return {
            key: value
            for key, value in resolved.items()
            if key in allowed
        }

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

    def _resolve_public_reward_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve an audited reward system and enforce its field precedence."""

        resolved_payload = dict(payload)
        revision_id = self._optional_str(payload.get("reward_system_revision_id"))
        if not revision_id:
            return resolved_payload
        mode = str(payload.get("mode") or "").strip().lower()
        if mode not in {"raft", "grpo", "vlm", "audio", "reasoning", "agentic"}:
            raise ValueError(f"Training mode {mode!r} cannot consume an audited reward system")
        backend = (
            "mlx"
            if str(payload.get("accelerator") or "").strip().lower() == "mlx"
            or str(payload.get("model") or "").startswith("mlx-community/")
            else "hf"
        )
        try:
            from halo_forge.training_signal import TRAINING_SIGNAL_CAPABILITIES

            capability_model = TRAINING_SIGNAL_CAPABILITIES.resolve(mode, backend)
            capability = capability_model.to_dict()
        except Exception as exc:
            raise ValueError(
                f"No truthful training-signal capability exists for {mode}/{backend}: {exc}"
            ) from exc
        supplied_boundaries = list(payload.get("reward_audit_boundaries") or [])
        if supplied_boundaries:
            boundaries = [
                value for value in supplied_boundaries if str(value).strip()
            ]
            if (
                mode == "grpo"
                and backend == "hf"
                and not payload.get("max_steps")
                and any(str(value).strip().lower() not in {"final", "last"} for value in boundaries)
            ):
                raise ValueError(
                    "HF GRPO intermediate reward-audit boundaries require max_steps"
                )
        else:
            from halo_forge.training_signal.session import default_audit_boundaries

            if not capability_model.resumable:
                boundaries = ["final"]
            else:
                configured_total = (
                    payload.get("max_steps")
                    if mode == "grpo" and backend == "hf"
                    else payload.get("cycles")
                )
                total = int(configured_total or 1)
                boundaries = default_audit_boundaries(max(1, total))
        protocol_id = self._optional_str(payload.get("reward_audit_protocol_revision_id"))
        profile_id = self._optional_str(payload.get("reward_integrity_profile_revision_id"))
        if not protocol_id or not profile_id:
            raise ValueError(
                "Audited training requires reward_audit_protocol_revision_id and "
                "reward_integrity_profile_revision_id"
            )
        reward_engine = self._reward_integrity_engine()
        development_suite_revision_id = self._optional_str(
            payload.get("development_suite_revision_id")
        )
        if development_suite_revision_id:
            reward_engine.validate_development_suite_revision(
                development_suite_revision_id
            )
        resolved = reward_engine.resolve_binding(
            revision_id,
            protocol_revision_id=protocol_id,
            integrity_profile_revision_id=profile_id,
            trainer=mode,
            backend=backend,
            boundaries=boundaries,
            runtime_identity=None,
        )
        value = self._reward_value(resolved)
        if not bool(value.get("gating_eligible", False)):
            blockers = [str(item) for item in value.get("blockers") or []]
            raise ValueError(
                "; ".join(blockers)
                or "The selected reward system is not eligible for a training gate"
            )
        revision = dict(value.get("reward_system_revision") or {})
        optimizer_revision_id = str(
            revision.get("optimizer_verifier_revision_id") or ""
        ).strip()
        supplied_verifier = self._optional_str(payload.get("verifier_profile_revision_id"))
        if supplied_verifier and supplied_verifier != optimizer_revision_id:
            raise ValueError(
                "reward_system_revision_id conflicts with verifier_profile_revision_id; "
                "the optimizer verifier is pinned by the reward system"
            )
        raw_verifier = self._optional_str(payload.get("verifier"))
        if raw_verifier and raw_verifier != f"profile:{optimizer_revision_id}":
            raise ValueError(
                "reward_system_revision_id conflicts with a raw verifier; use the "
                "optimizer verifier pinned by the immutable reward system"
            )
        if payload.get("verifier_config") not in (None, "", {}):
            raise ValueError(
                "reward_system_revision_id conflicts with raw verifier_config"
            )
        if not optimizer_revision_id:
            raise ValueError("Reward system has no optimizer verifier revision")

        if capability.get("fidelity") not in {"exact", "sampled"}:
            raise ValueError(
                "Training-signal capture is report-only for this trainer/backend"
            )
        if not capability.get("resumable") and any(
            str(value).lower() != "final" for value in boundaries
        ):
            raise ValueError(
                f"{mode}/{backend} supports a final audit only; remove intermediate boundaries"
            )

        replay_binding = (
            dict(resolved.to_replay_dict())
            if hasattr(resolved, "to_replay_dict")
            else {
                "reward_system_revision_id": revision.get("id"),
                "reward_system_hash": revision.get("content_hash"),
                "optimizer_verifier_revision_id": optimizer_revision_id,
                "auditors": list(revision.get("auditors") or []),
                "reward_mapping_hash": revision.get("reward_mapping_hash")
                or revision.get("content_hash"),
                "protocol_revision_id": dict(value.get("protocol_revision") or {}).get("id"),
                "protocol_hash": dict(value.get("protocol_revision") or {}).get(
                    "content_hash"
                ),
                "integrity_profile_revision_id": dict(
                    value.get("integrity_profile_revision") or {}
                ).get("id"),
                "integrity_profile_hash": dict(
                    value.get("integrity_profile_revision") or {}
                ).get("content_hash"),
                "runtime_compatibility": {"state": "compatible", "backend": backend},
            }
        )
        replay_binding["boundaries"] = boundaries
        replay_binding["signal_capability"] = capability
        if development_suite_revision_id:
            replay_binding["development_suite_revision_id"] = (
                development_suite_revision_id
            )
        resolved_payload["verifier_profile_revision_id"] = optimizer_revision_id
        resolved_payload["reward_audit_boundaries"] = boundaries
        resolved_payload["reward_integrity_binding"] = replay_binding
        return resolved_payload

    def _resolve_public_verifier_payload(
        self, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve an exact verifier revision without weakening legacy launches.

        The trainer-facing token is deliberately synthetic.  Managed launches
        pass the revision ID to the CLI, which installs the profile-backed
        runtime bridge before trainer construction; the token merely lets the
        existing preflight contracts describe the selected verifier.
        """

        resolved_payload = dict(payload)
        revision_id = self._optional_str(payload.get("verifier_profile_revision_id"))
        if not revision_id:
            raw_name = self._optional_str(payload.get("verifier"))
            if raw_name:
                resolved_payload["verifier_binding"] = {
                    "legacy_unqualified": True,
                    "implementation_ref": raw_name,
                    "warning": "Raw verifier configuration is runnable but not reliability-qualified.",
                }
            return resolved_payload

        mode = str(payload.get("mode") or "").strip().lower()
        supported_modes = {"raft", "grpo", "vlm", "audio", "reasoning", "agentic"}
        if mode not in supported_modes:
            raise ValueError(
                f"Training mode {mode!r} does not consume a verifier profile revision"
            )
        modality = {
            "vlm": "vlm",
            "audio": "audio",
            "agentic": "tool",
        }.get(mode, "text")
        binding = self._verifier_engine().resolve_binding(
            revision_id,
            modality=modality,
        )
        expected_name = str(binding.get("implementation_ref") or "").strip()
        profile_token = f"profile:{revision_id}"
        raw_name = self._optional_str(payload.get("verifier"))
        if raw_name and raw_name not in {expected_name, profile_token}:
            raise ValueError(
                "verifier_profile_revision_id conflicts with raw verifier "
                f"{raw_name!r}; the pinned implementation is {expected_name!r}"
            )
        raw_config = payload.get("verifier_config")
        if raw_config not in (None, "", {}):
            raise ValueError(
                "verifier_profile_revision_id conflicts with raw verifier_config; "
                "configuration is pinned by the immutable profile revision"
            )
        contract = dict(binding.get("reward_contract") or {})
        threshold = contract.get("threshold")
        for key in ("reward_threshold", "pass_threshold", "threshold"):
            supplied = payload.get(key)
            if supplied in (None, "") or threshold is None:
                continue
            try:
                matches = abs(float(supplied) - float(threshold)) <= 1e-12
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise ValueError(
                    f"verifier_profile_revision_id conflicts with {key}; "
                    f"the pinned threshold is {threshold}"
                )

        resolved_payload["verifier"] = profile_token
        if threshold is not None:
            resolved_payload["reward_threshold"] = float(threshold)
        resolved_payload["verifier_binding"] = binding
        return resolved_payload

    def _resolve_reward_fork_payload(
        self, payload: Dict[str, Any], *, verify_checkpoint: bool = False
    ) -> Dict[str, Any]:
        """Re-resolve immutable checkpoint lineage for a reviewed fork.

        The browser only carries the audit ID in its stable URL.  Paths,
        hashes, decision identity, and reward revisions always come back from
        SQLite and the sealed trace, preventing stale tabs or edited requests
        from silently launching a different checkpoint under the same lineage.
        """

        resolved = dict(payload)
        audit_id = self._optional_str(payload.get("source_reward_integrity_audit_id"))
        lineage_fields = {
            "source_reward_integrity_decision_id",
            "fork_checkpoint_hash",
            "fork_checkpoint_path",
            "fork_checkpoint_occurrence_id",
            "fork_checkpoint_snapshot_path",
            "fork_boundary_unit",
            "fork_boundary_value",
            "fork_resume_mode",
        }
        if not audit_id:
            unexpected = sorted(
                key for key in lineage_fields if self._has_public_value(payload.get(key))
            )
            if unexpected:
                raise ValueError(
                    "fork checkpoint lineage requires source_reward_integrity_audit_id"
                )
            return resolved

        context = self.get_reward_integrity_fork_context(audit_id)
        if not context.get("launch_ready"):
            raise ValueError(
                "reward-audit fork checkpoint is not launchable: "
                + ", ".join(str(value) for value in context.get("blockers") or ())
            )
        if verify_checkpoint:
            from halo_forge.artifact_lab.hashing import hash_path

            checkpoint_path = Path(
                str((context.get("checkpoint") or {}).get("path") or "")
            ).expanduser()
            expected_hash = str(
                (context.get("checkpoint") or {}).get("content_hash") or ""
            )
            if not checkpoint_path.exists():
                raise ValueError("reward-audit fork checkpoint disappeared before launch")
            actual_hash = hash_path(checkpoint_path).content_hash
            if actual_hash != expected_hash:
                raise ValueError(
                    "reward-audit fork checkpoint content no longer matches the sealed trace"
                )
        canonical_context = dict(context["train_context"])
        immutable_fields = {
            "parent_run_id",
            "source_reward_integrity_audit_id",
            "source_reward_integrity_decision_id",
            "reward_system_revision_id",
            "reward_audit_protocol_revision_id",
            "reward_integrity_profile_revision_id",
            *lineage_fields,
        }
        for key in sorted(immutable_fields):
            canonical = canonical_context.get(key)
            supplied = payload.get(key)
            if self._has_public_value(supplied) and supplied != canonical:
                # Numeric form controls may serialize an integer as a string.
                # Compare their stable JSON scalar representations before
                # calling it a contradictory immutable identity.
                if str(supplied) != str(canonical):
                    raise ValueError(
                        f"{key} conflicts with the server-resolved reward-audit fork"
                    )
            if self._has_public_value(canonical):
                resolved[key] = canonical
            else:
                resolved.pop(key, None)
        return resolved

    @staticmethod
    def _is_managed_training_payload(payload: Dict[str, Any]) -> bool:
        """Return whether a dashboard launch belongs to the durable control plane."""

        return bool(
            str(payload.get("mode") or "").strip().lower()
            in {"classify", "embed", "rerank"}
            or
            payload.get("output_root")
            or payload.get("dataset_version_id")
            or payload.get("dataset_bindings")
            or payload.get("parent_run_id")
            or payload.get("verifier_profile_revision_id")
            or payload.get("reward_system_revision_id")
        )

    @staticmethod
    def _managed_training_command(payload: Dict[str, Any]) -> list[str]:
        """Render the public-safe training payload into a shell-free argv list.

        Direct CLI and legacy manual-path launches keep using ``TrainingService``.
        This renderer is intentionally limited to the fields accepted by the
        public API, so persisted scheduler specifications cannot smuggle
        arbitrary trainer flags into the worker.
        """

        mode = str(payload["mode"])
        verifier_revision_id = str(payload.get("verifier_profile_revision_id") or "").strip()
        model = str(payload.get("model") or "")
        dataset = str(payload.get("dataset") or "")
        output_dir = str(payload.get("output_dir") or "")
        accelerator = str(payload.get("accelerator") or "").strip()
        command = [sys.executable, "-m", "halo_forge.cli"]
        if accelerator and accelerator != "auto":
            command.extend(["--accelerator", accelerator])

        if mode == "sft":
            command.extend(["sft", "train", "--model", model])
            command.extend(
                ["--data", dataset]
                if dataset and Path(dataset).expanduser().exists()
                else ["--dataset", dataset]
            )
            command.extend(
                [
                    "--output",
                    output_dir,
                    "--epochs",
                    str(int(payload.get("epochs") or 1)),
                    "--batch-size",
                    str(int(payload.get("batch_size") or 2)),
                    "--gradient-accumulation",
                    str(int(payload.get("gradient_accumulation_steps") or 4)),
                    "--learning-rate",
                    str(float(payload.get("learning_rate") or 2e-4)),
                    "--seed",
                    str(int(payload.get("seed") or 42)),
                ]
            )
            if payload.get("max_samples") is not None:
                command.extend(["--max-samples", str(int(payload["max_samples"]))])
            if payload.get("max_steps") is not None:
                command.extend(["--max-steps", str(int(payload["max_steps"]))])
            if payload.get("max_sequence_length") is not None:
                command.extend(
                    ["--max-seq-length", str(int(payload["max_sequence_length"]))]
                )
            if payload.get("gradient_checkpointing") is False:
                command.append("--no-gradient-checkpointing")
            if payload.get("validation_file"):
                command.extend(["--validation-data", str(payload["validation_file"])])
        elif mode == "cpt":
            adaptation = str(payload.get("adaptation") or "").strip().lower()
            if adaptation not in {"lora", "full"}:
                raise ValueError("CPT requires explicit adaptation='lora' or 'full'")
            command.extend(
                [
                    "cpt",
                    "train",
                    "--model",
                    model,
                    "--train-file",
                    dataset,
                    "--output",
                    output_dir,
                    "--adaptation",
                    adaptation,
                    "--max-seq-length",
                    str(int(payload.get("max_sequence_length") or 2048)),
                    "--packing",
                    str(
                        payload.get("packing")
                        or "paragraph_eos_non_overlap_v1"
                    ),
                    "--budget-mode",
                    str(payload.get("budget_mode") or "passes"),
                    "--batch-size",
                    str(int(payload.get("batch_size") or 1)),
                    "--gradient-accumulation",
                    str(
                        int(payload.get("gradient_accumulation_steps") or 8)
                    ),
                    "--learning-rate",
                    str(float(payload.get("learning_rate") or 2e-5)),
                    "--seed",
                    str(int(payload.get("seed") or 42)),
                ]
            )
            if payload.get("validation_file"):
                command.extend(
                    ["--validation-file", str(payload["validation_file"])]
                )
            if str(payload.get("budget_mode") or "passes") == "tokens":
                command.extend(
                    ["--target-tokens", str(int(payload.get("target_tokens") or 0))]
                )
            else:
                command.extend(
                    [
                        "--corpus-passes",
                        str(float(payload.get("corpus_passes") or 1.0)),
                    ]
                )
            if payload.get("max_steps") is not None:
                command.extend(["--max-steps", str(int(payload["max_steps"]))])
            if payload.get("gradient_checkpointing") is False:
                command.append("--no-gradient-checkpointing")
            if payload.get("model_revision"):
                command.extend(["--model-revision", str(payload["model_revision"])])
            if payload.get("model_hash"):
                command.extend(["--model-hash", str(payload["model_hash"])])
            if payload.get("tokenizer_revision"):
                command.extend(
                    ["--tokenizer-revision", str(payload["tokenizer_revision"])]
                )
            if payload.get("tokenizer_hash"):
                command.extend(
                    ["--tokenizer-hash", str(payload["tokenizer_hash"])]
                )
            if payload.get("training_artifact_id"):
                command.extend(
                    ["--training-artifact-id", str(payload["training_artifact_id"])]
                )
            if payload.get("training_artifact_hash"):
                command.extend(
                    ["--training-artifact-hash", str(payload["training_artifact_hash"])]
                )
            if payload.get("expected_packing_plan_hash"):
                command.extend(
                    [
                        "--expected-packing-plan-hash",
                        str(payload["expected_packing_plan_hash"]),
                    ]
                )
            if payload.get("load_in_4bit"):
                command.append("--load-in-4bit")
            for field, flag, caster in (
                ("lora_r", "--lora-rank", int),
                ("lora_alpha", "--lora-alpha", int),
                ("lora_dropout", "--lora-dropout", float),
            ):
                if payload.get(field) is not None:
                    command.extend([flag, str(caster(payload[field]))])
            if payload.get("use_dora"):
                command.append("--use-dora")
            if payload.get("use_rslora"):
                command.append("--use-rslora")
            if payload.get("init_lora_weights"):
                command.extend(
                    [
                        "--init-lora-weights",
                        str(payload["init_lora_weights"]),
                    ]
                )
            if payload.get("optim"):
                command.extend(["--optim", str(payload["optim"])])
        elif mode == "raft":
            command.extend(
                [
                    "raft",
                    "train",
                    "--model",
                    model,
                    "--prompts",
                    str(payload.get("prompts") or dataset),
                    "--output",
                    output_dir,
                    "--cycles",
                    str(int(payload.get("cycles") or 1)),
                    "--samples-per-prompt",
                    str(int(payload.get("samples_per_prompt") or 4)),
                    "--temperature",
                    str(float(payload.get("temperature") or 0.7)),
                    "--seed",
                    str(int(payload.get("seed") or 42)),
                    "--keep-percent",
                    str(float(payload.get("keep_percent") or 0.5)),
                    "--min-samples",
                    str(int(payload.get("min_samples") or 1)),
                    "--max-new-tokens",
                    str(int(payload.get("max_new_tokens") or 512)),
                ]
            )
            if not verifier_revision_id:
                command.extend(
                    [
                        "--verifier",
                        str(payload.get("verifier") or "humaneval"),
                        "--reward-threshold",
                        str(float(payload.get("reward_threshold") or 0.5)),
                    ]
                )
            if payload.get("max_prompts") is not None:
                command.extend(["--max-prompts", str(int(payload["max_prompts"]))])
        elif mode in {"dpo", "orpo", "rm", "grpo"}:
            defaults = {
                "dpo": (1, 16, 5e-6),
                "orpo": (1, 16, 8e-6),
                "rm": (4, 4, 1e-5),
                "grpo": (1, 16, 1e-6),
            }
            default_batch, default_accumulation, default_lr = defaults[mode]
            command.extend([mode, "train", "--model", model])
            command.extend(
                ["--data", dataset]
                if dataset and Path(dataset).expanduser().exists()
                else ["--dataset", dataset]
            )
            command.extend(
                [
                    "--output",
                    output_dir,
                    "--epochs",
                    str(int(payload.get("epochs") or 1)),
                    "--batch-size",
                    str(int(payload.get("batch_size") or default_batch)),
                    "--gradient-accumulation",
                    str(int(payload.get("gradient_accumulation_steps") or default_accumulation)),
                    "--learning-rate",
                    str(float(payload.get("learning_rate") or default_lr)),
                    "--seed",
                    str(int(payload.get("seed") or 42)),
                ]
            )
            if payload.get("max_samples") is not None:
                command.extend(["--max-samples", str(int(payload["max_samples"]))])
            if payload.get("max_steps") is not None:
                command.extend(["--max-steps", str(int(payload["max_steps"]))])
            if payload.get("gradient_checkpointing") is False:
                command.append("--no-gradient-checkpointing")
            if payload.get("validation_file") and mode in {"dpo", "orpo", "rm"}:
                command.extend(["--validation-data", str(payload["validation_file"])])
            if mode in {"dpo", "orpo"} and payload.get("beta") is not None:
                command.extend(["--beta", str(float(payload["beta"]))])
            if mode == "dpo" and payload.get("loss_type"):
                command.extend(["--loss-type", str(payload["loss_type"])])
            if mode == "dpo" and payload.get("reference_free"):
                command.append("--reference-free")
            if mode == "grpo":
                command.extend(
                    [
                        "--num-generations",
                        str(int(payload.get("num_generations") or 4)),
                        "--beta",
                        str(float(payload.get("beta") or 0.04)),
                        "--epsilon",
                        str(float(payload.get("epsilon") or 0.2)),
                        "--temperature",
                        str(float(payload.get("temperature") or 0.9)),
                    ]
                )
                if not verifier_revision_id:
                    command.extend(
                        [
                            "--verifier",
                            str(payload.get("verifier") or "execution"),
                            "--reward-threshold",
                            str(float(payload.get("reward_threshold") or 0.0)),
                        ]
                    )
                if payload.get("reference_free"):
                    command.append("--reference-free")
        elif mode in {"classify", "embed", "rerank"}:
            command.extend(
                [
                    mode,
                    "train",
                    "--model",
                    model,
                    "--dataset",
                    dataset,
                    "--output",
                    output_dir,
                    "--epochs",
                    str(int(payload.get("epochs") or 1)),
                    "--batch-size",
                    str(int(payload.get("batch_size") or 4)),
                    "--learning-rate",
                    str(float(payload.get("learning_rate") or 2e-5)),
                    "--seed",
                    str(int(payload.get("seed") or 42)),
                ]
            )
            if payload.get("max_samples") is not None:
                command.extend(["--max-samples", str(int(payload["max_samples"]))])
            if payload.get("validation_file"):
                command.extend(["--validation-file", str(payload["validation_file"])])
            if mode == "classify" and payload.get("multi_label"):
                command.append("--multi-label")
            if payload.get("label_schema_revision_id"):
                command.extend(
                    [
                        "--label-schema-revision",
                        str(payload["label_schema_revision_id"]),
                    ]
                )
            if payload.get("retrieval_corpus_id"):
                command.extend(
                    ["--retrieval-corpus", str(payload["retrieval_corpus_id"])]
                )
            if payload.get("proof_run"):
                command.append("--proof-run")
        else:
            command.extend(
                [
                    mode,
                    "train",
                    "--model",
                    model,
                    "--dataset",
                    dataset,
                    "--output",
                    output_dir,
                    "--cycles",
                    str(int(payload.get("cycles") or 1)),
                    "--resume-from-cycle",
                    "0",
                    "--seed",
                    str(int(payload.get("seed") or 42)),
                ]
            )
            if payload.get("learning_rate") is not None and mode in {
                "audio",
                "reasoning",
                "agentic",
            }:
                command.extend(["--lr", str(float(payload["learning_rate"]))])
            if payload.get("samples_per_prompt") is not None and mode in {"vlm", "audio"}:
                command.extend(["--samples-per-prompt", str(int(payload["samples_per_prompt"]))])
            for field, flag in (
                ("temperature", "--temperature"),
                ("keep_percent", "--keep-percent"),
            ):
                if payload.get(field) is not None:
                    command.extend([flag, str(float(payload[field]))])
            if payload.get("reward_threshold") is not None and mode in {"vlm", "audio"}:
                command.extend(["--reward-threshold", str(float(payload["reward_threshold"]))])
            if mode == "audio" and payload.get("task"):
                command.extend(["--task", str(payload["task"])])
            if payload.get("limit") is not None and mode in {
                "vlm",
                "audio",
                "reasoning",
                "agentic",
            }:
                command.extend(["--limit", str(int(payload["limit"]))])
            if payload.get("allow_prototype_train"):
                command.append("--allow-prototype-train")

        if payload.get("no_caffeinate"):
            command.append("--no-caffeinate")
        if verifier_revision_id:
            command.extend(["--verifier-profile-revision", verifier_revision_id])
        if payload.get("reward_system_revision_id"):
            command.extend(
                ["--reward-system-revision", str(payload["reward_system_revision_id"])]
            )
            command.extend(
                [
                    "--reward-audit-protocol-revision",
                    str(payload["reward_audit_protocol_revision_id"]),
                    "--reward-integrity-profile-revision",
                    str(payload["reward_integrity_profile_revision_id"]),
                ]
            )
            for boundary in payload.get("reward_audit_boundaries") or []:
                command.extend(["--reward-audit-boundary", str(boundary)])
            if payload.get("development_suite_revision_id"):
                command.extend(
                    [
                        "--reward-development-suite-revision",
                        str(payload["development_suite_revision_id"]),
                    ]
                )
        fork_checkpoint = str(payload.get("fork_checkpoint_path") or "").strip()
        if (
            payload.get("source_reward_integrity_audit_id")
            and payload.get("fork_resume_mode") == "initialize_from_checkpoint"
            and fork_checkpoint
        ):
            if mode == "raft":
                command.extend(["--checkpoint", fork_checkpoint])
            elif mode == "grpo":
                command.extend(["--resume", fork_checkpoint])
        return command

    def _queue_managed_training(
        self,
        payload: Dict[str, Any],
        *,
        canonical_run_id: str,
        dataset_version_metadata: Optional[Dict[str, Any]],
    ) -> tuple[JobState, Any]:
        """Create one canonical run and its first durable training segment."""

        from halo_forge.run_db import RunRecord
        from halo_forge.reward_integrity.direct_segments import (
            build_segment_launch_spec,
            resolve_boundary_values,
        )

        database = self._dataset_database()
        existing_items = database.list_work_items(canonical_run_id=canonical_run_id, limit=10)
        if existing_items:
            existing = existing_items[0]
            job = self.app_state.get_job(canonical_run_id)
            if job is None:
                job = self._create_managed_training_job(
                    payload, canonical_run_id=canonical_run_id, work_item=existing
                )
            return job, existing

        output_dir = Path(str(payload["output_dir"])).expanduser().resolve()
        command = self._managed_training_command(payload)
        log_path = output_dir.parent / ".halo-forge-logs" / f"{canonical_run_id}.log"
        resolved_config = {
            **payload,
            "run_id": canonical_run_id,
            "output_dir": str(output_dir),
        }
        working_directory_resolver = getattr(
            self.training_service, "_launch_working_directory", None
        )
        working_directory = (
            Path(working_directory_resolver())
            if callable(working_directory_resolver)
            else self.base_path
        )
        launch_spec = {
            "operation": "managed_training",
            "command": command,
            "canonical_command": command,
            "cwd": str(working_directory),
            "output_dir": str(output_dir),
            "artifact_root": str(self.artifact_storage_root),
            "resolved_launch_config": resolved_config,
            "dataset_version_metadata": dict(dataset_version_metadata or {}),
            "source_ui_page": "/public/train",
            "no_caffeinate": bool(payload.get("no_caffeinate", False)),
            "runtime_profile_revision_id": payload.get("runtime_profile_revision_id"),
        }
        replay_context = {
            key: copy.deepcopy(payload.get(key))
            for key in (
                "outcome_assessment_id",
                "outcome_override_reason",
                "study_id",
                "study_protocol_revision_id",
                "study_arm_id",
                "study_assignment_id",
                "study_assignment_ids_by_seed",
                "study_factor_values",
                "study_contrast_ids",
                "study_deviation_ids",
            )
            if payload.get(key) not in (None, "", [], {})
        }
        if replay_context:
            launch_spec["env"] = {
                "HALO_FORGE_OPERATIONAL_COMPLETION": json.dumps(
                    replay_context, sort_keys=True, default=str
                )
            }
        requirements = {
            "exclusive_heavy_operation": True,
            "output_path": str(output_dir.parent),
            "projected_disk_bytes": 0,
            "projected_ram_bytes": 0,
            "runtime_profile_revision_id": payload.get("runtime_profile_revision_id"),
            "runtime_root": str(self.managed_runtime_storage_root),
            "accelerator_family": (
                "rocm"
                if str(payload.get("accelerator") or self._active_backend_name()).lower().startswith("rocm")
                else "cuda"
                if str(payload.get("accelerator") or self._active_backend_name()).lower() == "cuda"
                else None
            ),
            "resumable": bool(payload.get("reward_system_revision_id")),
        }
        work_item = None
        segments: list[Dict[str, Any]] = []
        try:
            now = datetime.now(timezone.utc).isoformat()
            database.upsert_run(
                RunRecord(
                    run_id=canonical_run_id,
                    fs_id=canonical_run_id,
                    modality=str(payload["mode"]),
                    model_name=str(
                        payload.get("training_plan_model_id")
                        or payload.get("model")
                        or ""
                    ),
                    status="queued",
                    timestamp=now,
                    output_dir=str(output_dir),
                    seed=int(payload.get("seed") or 42),
                    raw_json=json.dumps(
                        {
                            "run_id": canonical_run_id,
                            "status": "queued",
                            "launch_config": resolved_config,
                        },
                        sort_keys=True,
                        default=str,
                    ),
                )
            )
            if payload.get("reward_system_revision_id"):
                boundaries = list(payload.get("reward_audit_boundaries") or ["final"])
                capability = dict(
                    dict(payload.get("reward_integrity_binding") or {}).get(
                        "signal_capability"
                    )
                    or {}
                )
                mode = str(payload.get("mode") or "")
                backend = (
                    "mlx"
                    if str(payload.get("accelerator") or "").strip().lower() == "mlx"
                    or str(payload.get("model") or "").startswith("mlx-community/")
                    else "hf"
                )
                resolved_config["_resolved_signal_backend"] = backend
                configured_total = (
                    payload.get("max_steps")
                    if mode == "grpo" and backend == "hf"
                    else payload.get("cycles") or payload.get("epochs")
                )
                total = max(1, int(configured_total or 1))
                unit = str(capability.get("boundary_unit") or "final")
                if mode == "grpo" and backend == "hf" and not payload.get("max_steps"):
                    unit = "final"
                if unit not in {"step", "cycle", "epoch", "final"}:
                    unit = "final"
                resolved_boundaries = resolve_boundary_values(
                    boundaries,
                    total=total,
                    resumable=bool(capability.get("resumable")),
                )
                fork_resume = (
                    payload.get("fork_resume_mode") == "resume_boundary"
                    and bool(payload.get("source_reward_integrity_audit_id"))
                )
                fork_boundary = (
                    int(payload.get("fork_boundary_value") or 0) if fork_resume else 0
                )
                if fork_resume:
                    resolved_boundaries = [
                        value for value in resolved_boundaries if value > fork_boundary
                    ]
                    if not resolved_boundaries:
                        raise ValueError(
                            "forked training budget must extend beyond the audited checkpoint "
                            f"{unit}={fork_boundary}"
                        )
                    launch_spec["fork_checkpoint_snapshot_path"] = str(
                        payload["fork_checkpoint_snapshot_path"]
                    )
                    launch_spec["fork_checkpoint_hash"] = str(
                        payload["fork_checkpoint_hash"]
                    )
                    launch_spec["fork_checkpoint_occurrence_id"] = payload.get(
                        "fork_checkpoint_occurrence_id"
                    )
                previous = fork_boundary
                for ordinal, boundary in enumerate(resolved_boundaries):
                    segment = database.create_direct_run_segment(
                        run_id=canonical_run_id,
                        ordinal=ordinal,
                        unit=unit,
                        start_value=previous,
                        end_value=boundary,
                    )
                    if ordinal:
                        segment = database.update_direct_run_segment(
                            str(segment["id"]),
                            status="blocked",
                            decision_reason="waiting for the previous boundary audit",
                        )
                    segments.append(segment)
                    previous = boundary
                launch_spec = build_segment_launch_spec(
                    launch_spec,
                    segment={**segments[0], "is_final": len(segments) == 1},
                )
                work_item_id = f"training-segment-work-{segments[0]['id']}"
            else:
                work_item_id = f"training-work-{uuid.uuid4().hex}"
            work_item = self._scheduler().enqueue(
                kind="training",
                launch_spec=launch_spec,
                resource_class="accelerator",
                resource_requirements=requirements,
                domain_kind="run",
                domain_id=canonical_run_id,
                canonical_run_id=canonical_run_id,
                log_path=str(log_path),
                max_retries=1,
                work_item_id=work_item_id,
            )
            if segments:
                segments[0] = database.update_direct_run_segment(
                    str(segments[0]["id"]), work_item_id=work_item.id
                )
            raw = {
                "run_id": canonical_run_id,
                "status": "queued",
                "work_item_id": work_item.id,
                "launch_config": resolved_config,
            }
            with database._lock:
                database._conn.execute(
                    "UPDATE runs SET raw_json=? WHERE run_id=?",
                    (json.dumps(raw, sort_keys=True, default=str), canonical_run_id),
                )
                database._conn.commit()
            job = self._create_managed_training_job(
                payload, canonical_run_id=canonical_run_id, work_item=work_item
            )
        except Exception:
            if work_item is not None:
                self._scheduler().cancel(work_item.id)
            for segment in segments:
                database.update_direct_run_segment(
                    str(segment["id"]),
                    status="failed",
                    decision_reason="managed segment launch could not be queued",
                )
            raise
        assert work_item is not None
        return job, work_item

    def _create_managed_training_job(
        self,
        payload: Dict[str, Any],
        *,
        canonical_run_id: str,
        work_item: Any,
    ) -> JobState:
        mode = str(payload["mode"])
        job = self.app_state.create_job(
            job_type=mode,
            name=f"{mode.upper()}: {Path(str(payload.get('model') or '')).name}",
            output_dir=Path(str(payload["output_dir"])),
            job_id=canonical_run_id,
        )
        if mode in {"raft", "vlm", "audio", "reasoning", "agentic"}:
            job.total_cycles = int(payload.get("cycles") or 1)
        else:
            job.total_epochs = int(payload.get("epochs") or 1)
        job.launch_args = dict(payload)
        job.lifecycle_metadata = {
            "managed": True,
            "work_item_id": work_item.id,
            "canonical_run_id": canonical_run_id,
        }
        job.log_file_path = Path(str(work_item.log_path)) if work_item.log_path else None
        return job

    def _sync_managed_training_job(self, job: JobState) -> JobState:
        run_id = str((job.lifecycle_metadata or {}).get("canonical_run_id") or job.id)
        segments = self._dataset_database().list_direct_run_segments(run_id)
        if segments:
            active_segment = next(
                (
                    value
                    for value in segments
                    if value.get("work_item_id")
                    and str(value.get("status") or "")
                    not in {"completed", "reviewed", "stopped", "failed"}
                ),
                next(
                    (value for value in reversed(segments) if value.get("work_item_id")),
                    None,
                ),
            )
            if active_segment is not None:
                job.lifecycle_metadata["work_item_id"] = str(
                    active_segment["work_item_id"]
                )
        work_item_id = str((job.lifecycle_metadata or {}).get("work_item_id") or "")
        if not work_item_id:
            return job
        item = self._dataset_database().get_work_item(work_item_id)
        if item is None:
            return job
        status_map = {
            "queued": "pending",
            "blocked": "pending",
            "running": "running",
            "completed": "completed",
            "cancelled": "stopped",
            "failed": "failed",
            "interrupted": "failed",
            "needs_reconciliation": "failed",
        }
        job.status = status_map.get(item.status, job.status)
        job.stage_key = str(item.stage or job.stage_key)
        job.stage_label = job.stage_key.replace("_", " ").title()
        job.stage_message = str(item.error or item.stage or job.stage_message)
        progress = dict(item.progress or {})
        raw_percent = progress.get("percent", progress.get("progress_percent"))
        if raw_percent is not None:
            try:
                job.stage_progress_percent = max(0.0, min(100.0, float(raw_percent)))
            except (TypeError, ValueError):
                pass
        if item.started_at:
            job.started_at = datetime.fromisoformat(item.started_at)
        if item.completed_at:
            job.completed_at = datetime.fromisoformat(item.completed_at)
        job.error_message = item.error
        if item.result.get("artifact_occurrence_ids"):
            job.artifact_state = "final_model"
        if segments:
            run = self._dataset_database().get_run(run_id)
            if run is not None and run.status == "awaiting_review":
                job.status = "pending"
                job.stage_key = "awaiting_review"
                job.stage_label = "Awaiting Review"
                job.stage_message = "A reward-integrity audit requires an operator decision."
            elif run is not None and run.status in {"running", "queued", "audit_pending"}:
                if item.status == "completed":
                    job.status = "running"
                    job.stage_key = "reward_audit"
                    job.stage_label = "Reward Audit"
                    job.stage_message = "Training is stopped at a resumable audit boundary."
            elif run is not None and run.status in {"completed", "succeeded"}:
                job.status = "completed"
        return job

    def _hydrate_managed_training_job(self, run_id: str) -> Optional[JobState]:
        existing = self.app_state.get_job(run_id)
        if existing is not None:
            return self._sync_managed_training_job(existing)
        work_items = self._dataset_database().list_work_items(
            canonical_run_id=run_id,
            kinds=["training"],
            limit=20,
        )
        item = next(
            (
                value
                for value in work_items
                if value.launch_spec.get("operation")
                in {"managed_training", "managed_training_segment"}
            ),
            None,
        )
        if item is None:
            return None
        config = dict(item.launch_spec.get("resolved_launch_config") or {})
        if not config.get("mode") or not config.get("output_dir"):
            return None
        job = self._create_managed_training_job(
            config,
            canonical_run_id=run_id,
            work_item=item,
        )
        job.created_at = datetime.fromisoformat(item.created_at)
        return self._sync_managed_training_job(job)

    def _hydrate_all_managed_training_jobs(self) -> None:
        for item in self._dataset_database().list_work_items(kinds=["training"], limit=1000):
            run_id = str(item.canonical_run_id or "").strip()
            if run_id and item.launch_spec.get("operation") in {
                "managed_training",
                "managed_training_segment",
            }:
                self._hydrate_managed_training_job(run_id)

    def _normalize_guided_proof_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make generic preflight/launch calls obey the guided proof contract.

        The dedicated proof endpoint performs the stricter scenario checks.
        This normalizer also protects clients that preflight the same payload
        through ``/train/preflight`` before using that endpoint.
        """

        resolved = dict(payload)
        if not bool(resolved.get("proof_run")):
            if not resolved.get("output_dir") and resolved.get("output_root"):
                resolved["output_dir"] = str(resolved["output_root"])
            return resolved
        version_id = self._optional_str(resolved.get("dataset_version_id"))
        version = (
            self._dataset_database().get_dataset_version(version_id)
            if version_id
            else None
        )
        dataset = (
            self._dataset_database().get_dataset(version.dataset_id)
            if version is not None
            else None
        )
        mode = str(resolved.get("trainer_mode") or resolved.get("mode") or "").lower()
        if resolved.get("trainer_mode"):
            resolved["mode"] = mode
            resolved.pop("trainer_mode", None)
        default_cap = 50 if str(getattr(dataset, "modality", "")).lower() in {
            "image",
            "vlm",
            "audio",
        } else 200
        raw_cap = resolved.get(
            "max_prompts", resolved.get("limit", resolved.get("max_samples", default_cap))
        )
        try:
            cap = max(1, min(int(raw_cap), default_cap))
        except (TypeError, ValueError):
            cap = default_cap
        if version is not None and int(version.split_counts.get("train", 0)) > 0:
            cap = min(cap, int(version.split_counts["train"]))
            bindings = [
                {
                    "role": "train",
                    "dataset_version_id": version.id,
                    "split": "train",
                }
            ]
            if int(version.split_counts.get("validation", 0)) > 0:
                bindings.append(
                    {
                        "role": "validation",
                        "dataset_version_id": version.id,
                        "split": "validation",
                    }
                )
            resolved["dataset_bindings"] = bindings
            resolved["dataset_split"] = "train"
            guided = self._guided_version_context(version)
            scenario_revision_id = self._optional_str(
                resolved.get("scenario_revision_id") or guided.get("scenario_revision_id")
            )
            if scenario_revision_id:
                resolved["scenario_revision_id"] = scenario_revision_id
                resolved.setdefault("field_mapping_plan", guided["field_mapping_plan"])
                resolved.setdefault("dataset_preparation_recipe", dict(version.recipe or {}))
                resolved["proof_sample_identity"] = self._proof_sample_identity(
                    version=version,
                    scenario_revision_id=scenario_revision_id,
                    trainer_mode=mode,
                    split="train",
                    max_samples=cap,
                    seed=42,
                )
        for key in ("max_samples", "max_prompts", "limit", "max_steps"):
            resolved.pop(key, None)
        resolved["seed"] = 42
        resolved["proof_max_samples"] = cap
        if mode in {"sft", "dpo", "orpo", "rm", "classify", "embed", "rerank"}:
            resolved["epochs"] = 1
            resolved["max_samples"] = cap
            resolved.pop("cycles", None)
        elif mode == "grpo":
            resolved["epochs"] = 1
            resolved["max_samples"] = cap
            resolved["max_steps"] = 1
            resolved.pop("cycles", None)
        elif mode == "raft":
            resolved["cycles"] = 1
            resolved["max_prompts"] = cap
            resolved.pop("epochs", None)
        elif mode:
            resolved["cycles"] = 1
            resolved["limit"] = cap
            resolved.pop("epochs", None)
        if mode == "audio":
            resolved["task"] = "asr"
        if not resolved.get("output_dir"):
            resolved["output_dir"] = str(resolved.get("output_root") or _default_run_root())
        return resolved

    def preflight_training(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run launch preflight for the requested training mode."""
        readiness = self._product_lab_engine().assess_readiness()
        payload = dict(payload)
        plan_revision_id = self._optional_str(payload.get("training_plan_revision_id"))
        if plan_revision_id:
            supplied = dict(payload)
            supplied.pop("training_plan_revision_id", None)
            payload = self._resolve_public_training_plan_payload(
                plan_revision_id, supplied
            )
        payload["workstation_readiness_id"] = readiness.id
        payload["workstation_readiness_hash"] = readiness.content_hash
        payload["distribution_capability"] = readiness.capability.to_dict()
        payload = self._normalize_public_training_aliases(payload)
        payload = self._normalize_guided_proof_payload(payload)
        payload = self._resolve_reward_fork_payload(payload)
        runtime_revision_id = self._optional_str(payload.get("runtime_profile_revision_id"))
        if runtime_revision_id:
            runtime = self._managed_runtime_engine()
            revision = runtime.get_revision(runtime_revision_id)
            qualification = runtime.latest_qualification(runtime_revision_id)
            if revision is None or qualification is None:
                raise ValueError("The selected managed runtime is not prepared and qualified")
            verification = runtime.verify(qualification.id)
            if qualification.status not in {"vendor_supported", "local_verified"} or not verification["valid"]:
                raise ValueError("The selected managed runtime qualification is stale or incomplete")
            # V21 only applies this stricter gate to the guided immutable-plan
            # surface. Advanced/manual commands remain compatible and visibly
            # unverified; a generic runtime qualification never unlocks a plan.
            if plan_revision_id:
                path_revision_id = self._optional_str(
                    payload.get("training_path_revision_id")
                )
                certification_id = self._optional_str(
                    payload.get("training_path_certification_id")
                )
                if not path_revision_id or not certification_id:
                    raise ValueError("Verify this real training path before preparing user data")
                certification = self._training_path_engine().get_certification(
                    certification_id
                )
                if (
                    certification is None
                    or certification.path_revision_id != path_revision_id
                    or certification.runtime_revision_id != runtime_revision_id
                    or not self._training_path_engine().verify(certification_id)["valid"]
                ):
                    raise ValueError("The selected training-path certification is stale or contradictory")
        payload = self._prepare_managed_dataset_payload(payload)
        if payload.get("_managed_dataset_pending"):
            return self._public_training_preparation_payload(payload)
        payload = self._resolve_public_reward_payload(payload)
        payload = self._resolve_public_verifier_payload(payload)
        payload = self._sanitize_public_training_payload(payload)
        mode = str(payload["mode"])
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
        elif mode == "cpt":
            preflight = self.training_service.preflight_sft_launch(
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                epochs=1,
                batch_size=int(
                    payload.get("batch_size")
                    or payload.get("effective_batch_size")
                    or 1
                ),
                gradient_accumulation_steps=int(
                    payload.get("gradient_accumulation_steps")
                    or (1 if payload.get("effective_batch_size") else 8)
                ),
                max_samples=None,
            )
            try:
                from halo_forge.cpt.config import CPTConfig

                config = CPTConfig(
                    adaptation=self._optional_str(payload.get("adaptation")),
                    model_name=str(payload.get("model") or ""),
                    model_revision=self._optional_str(
                        payload.get("model_revision")
                    ),
                    model_hash=self._optional_str(payload.get("model_hash")),
                    tokenizer_revision=self._optional_str(
                        payload.get("tokenizer_revision")
                    ),
                    tokenizer_hash=self._optional_str(
                        payload.get("tokenizer_hash")
                    ),
                    train_file=str(payload.get("dataset") or ""),
                    validation_file=self._optional_str(
                        payload.get("validation_file")
                    ),
                    max_sequence_length=int(
                        payload.get("max_sequence_length") or 2048
                    ),
                    packing=str(
                        payload.get("packing")
                        or "paragraph_eos_non_overlap_v1"
                    ),
                    budget_mode=str(payload.get("budget_mode") or "passes"),
                    target_tokens=self._optional_int(payload.get("target_tokens")),
                    corpus_passes=self._optional_float(
                        payload.get("corpus_passes")
                    ),
                    lora_r=int(payload.get("lora_r") or 16),
                    lora_alpha=int(payload.get("lora_alpha") or 32),
                    lora_dropout=float(payload.get("lora_dropout") or 0.0),
                    use_dora=bool(payload.get("use_dora", False)),
                    use_rslora=bool(payload.get("use_rslora", False)),
                    init_lora_weights=str(
                        payload.get("init_lora_weights") or "true"
                    ),
                    load_in_4bit=bool(payload.get("load_in_4bit", False)),
                    output_dir=str(payload.get("output_dir") or ""),
                    batch_size=int(payload.get("batch_size") or 1),
                    gradient_accumulation_steps=int(
                        payload.get("gradient_accumulation_steps") or 8
                    ),
                    learning_rate=float(payload.get("learning_rate") or 2e-5),
                    max_steps=self._optional_int(payload.get("max_steps")),
                    optim=str(payload.get("optim") or "adamw_torch"),
                    seed=int(payload.get("seed") or 42),
                )
                preflight.resolved_paths.update(
                    {
                        "objective": "causal_next_token",
                        "adaptation": str(config.adaptation),
                        "budget_mode": config.budget_mode,
                        "max_sequence_length": str(config.max_sequence_length),
                        "packing": config.packing,
                    }
                )
                preflight.quality_outlook.update(
                    {
                        "objective": "causal_next_token",
                        "adaptation": config.adaptation,
                        "budget": {
                            "mode": config.budget_mode,
                            "target_tokens": config.target_tokens,
                            "corpus_passes": config.corpus_passes,
                        },
                        "validation_policy": (
                            "supplied_exact"
                            if payload.get("validation_file")
                            else "deterministic_derived_if_needed"
                        ),
                    }
                )
                if config.adaptation == "full":
                    preflight.warnings.append(
                        "Full CPT updates every model weight and normally needs "
                        "substantially more memory and checkpoint storage than LoRA."
                    )
            except (TypeError, ValueError) as exc:
                preflight.errors.append(str(exc))
                preflight.suggested_fixes.append(
                    "Choose LoRA or full adaptation and provide one valid token "
                    "or corpus-pass budget."
                )
            preflight.ok = len(preflight.errors) == 0
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
                max_prompts=self._optional_int(payload.get("max_prompts")),
                checkpoint=self._optional_str(payload.get("checkpoint")),
            )
        elif mode in {"dpo", "orpo", "rm", "grpo"}:
            default_lr = {
                "dpo": 5e-6,
                "orpo": 8e-6,
                "rm": 1e-5,
                "grpo": 1e-6,
            }[mode]
            default_batch = 4 if mode == "rm" else 1
            default_grad_accum = 4 if mode == "rm" else 16
            preflight = self.training_service.preflight_preference_launch(
                mode=mode,
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                epochs=int(payload.get("epochs") or 1),
                batch_size=int(payload.get("batch_size") or default_batch),
                gradient_accumulation_steps=int(
                    payload.get("gradient_accumulation_steps") or default_grad_accum
                ),
                learning_rate=float(payload.get("learning_rate") or default_lr),
                max_samples=self._optional_int(payload.get("max_samples")),
                beta=self._optional_float(payload.get("beta")),
                loss_type=self._optional_str(payload.get("loss_type")),
                reference_free=bool(payload.get("reference_free", False)),
                verifier=self._optional_str(payload.get("verifier")),
                num_generations=self._optional_int(payload.get("num_generations")),
                epsilon=self._optional_float(payload.get("epsilon")),
                temperature=self._optional_float(payload.get("temperature")),
                reward_threshold=self._optional_float(payload.get("reward_threshold")),
            )
        elif mode in {"classify", "embed", "rerank"}:
            preflight = self.training_service.preflight_sft_launch(
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                epochs=int(payload.get("epochs") or 1),
                batch_size=int(payload.get("batch_size") or 4),
                gradient_accumulation_steps=1,
                max_samples=self._optional_int(payload.get("max_samples")),
            )
            preflight.quality_outlook.update(
                {
                    "task": mode,
                    "backend": "pytorch",
                    "loss_adapter": {
                        "classify": "cross_entropy_or_bce_v1",
                        "embed": "multiple_negative_ranking_v1",
                        "rerank": "binary_scalar_cross_encoder_v1",
                    }[mode],
                    "artifact_contract": (
                        "model head, processor/tokenizer, task configuration, "
                        "and fixed-input round trip"
                    ),
                }
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

        self._augment_mlx_preflight(payload=payload, preflight=preflight)
        self._augment_output_dir_preflight(payload=payload, preflight=preflight)
        result = to_dict(self._build_preflight_view(mode=mode, preflight=preflight))
        result["workstation_readiness"] = readiness.to_dict()
        if readiness.status == "blocked":
            result["ok"] = False
            result.setdefault("errors", []).append(readiness.summary)
            result.setdefault("suggested_fixes", []).extend(
                remediation.label for remediation in readiness.remediations
            )
        artifact = dict(payload.get("training_artifact_metadata") or {})
        if artifact:
            result["training_artifact"] = artifact
            result["dataset_bindings"] = copy.deepcopy(
                payload.get("dataset_bindings") or []
            )
            if mode == "cpt":
                result["packing_plan"] = copy.deepcopy(
                    artifact.get("packing_plan")
                )
        if plan_revision_id:
            result["training_plan"] = self._training_plan_engine().get_revision(
                plan_revision_id
            ).to_dict()
            result["training_plan_readiness"] = self._training_plan_engine().readiness(
                plan_revision_id
            ).to_dict()
        return result

    async def launch_training(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Launch training from the public API."""
        from halo_forge.runtime_determinism import build_run_id

        readiness = self._product_lab_engine().assess_readiness()
        if readiness.status == "blocked":
            raise ValueError(readiness.summary)
        payload = dict(payload)
        plan_revision_id = self._optional_str(payload.get("training_plan_revision_id"))
        if plan_revision_id:
            supplied = dict(payload)
            supplied.pop("training_plan_revision_id", None)
            payload = self._resolve_public_training_plan_payload(
                plan_revision_id, supplied
            )
        payload["workstation_readiness_id"] = readiness.id
        payload["workstation_readiness_hash"] = readiness.content_hash
        payload["distribution_capability"] = readiness.capability.to_dict()
        payload = self._normalize_public_training_aliases(payload)
        payload = self._normalize_guided_proof_payload(payload)
        payload = self._resolve_reward_fork_payload(payload, verify_checkpoint=True)
        mode_hint = str(payload.get("mode") or "run").strip().lower()
        canonical_run_id = build_run_id(
            mode_hint, requested=self._optional_str(payload.get("run_id"))
        )
        payload["run_id"] = canonical_run_id
        payload = self._resolve_public_reward_payload(payload)
        payload = self._resolve_public_verifier_payload(payload)
        managed_launch = self._is_managed_training_payload(payload)
        if managed_launch:
            selected_root = Path(
                str(payload.get("output_root") or payload.get("output_dir") or "")
            ).expanduser()
            run_output = (
                selected_root
                if selected_root.name == canonical_run_id
                else selected_root / canonical_run_id
            )
            if run_output.exists():
                raise ValueError(f"Managed run directory already exists: {run_output}")
            payload["output_dir"] = str(run_output)

        payload = self._prepare_managed_dataset_payload(payload)
        if payload.get("_managed_dataset_pending"):
            return self._public_training_preparation_payload(payload)
        payload = self._sanitize_public_training_payload(payload)
        mode = str(payload["mode"])
        dataset_version_metadata = dict(self._dataset_version_metadata(payload) or {})
        dataset_version_metadata = {
            **dataset_version_metadata,
            "product_completion": {
                "workstation_readiness_id": readiness.id,
                "workstation_readiness_hash": readiness.content_hash,
                "distribution_capability": readiness.capability.to_dict(),
            },
        }
        launch_bindings = list(payload.get("dataset_bindings") or [])
        artifact_metadata = dict(payload.get("training_artifact_metadata") or {})
        parent_run_id = self._optional_str(payload.get("parent_run_id"))
        if managed_launch:
            preflight_payload = dict(payload)
            # The plan was resolved above.  Internal preflight must validate
            # that exact executable projection instead of resolving it again.
            preflight_payload.pop("training_plan_revision_id", None)
            preflight = self.preflight_training(preflight_payload)
            if not preflight.get("ok"):
                errors = [str(value) for value in preflight.get("errors") or []]
                raise ValueError("; ".join(errors) or "Managed training preflight failed")
            job, work_item = self._queue_managed_training(
                payload,
                canonical_run_id=canonical_run_id,
                dataset_version_metadata=dataset_version_metadata,
            )
            runtime_binding = None
            runtime_revision_id = self._optional_str(
                payload.get("runtime_profile_revision_id")
            )
            if runtime_revision_id:
                runtime_binding = self._managed_runtime_engine().bind(
                    revision_id=runtime_revision_id,
                    domain_kind="run",
                    domain_id=canonical_run_id,
                    details={"work_item_id": work_item.id, "surface": "public_train"},
                )
            verifier_binding = None
            verifier_revision_id = self._optional_str(
                payload.get("verifier_profile_revision_id")
            )
            if verifier_revision_id:
                verifier_binding = self._verifier_engine().bind_revision(
                    verifier_revision_id,
                    domain_kind="run",
                    domain_id=canonical_run_id,
                    role=f"{mode}_training_reward",
                    context={
                        "work_item_id": work_item.id,
                        "managed": True,
                        "source": "public_train",
                    },
                )
            reward_binding = None
            reward_revision_id = self._optional_str(
                payload.get("reward_system_revision_id")
            )
            if reward_revision_id:
                reward_binding = self._reward_integrity_engine().bind(
                    reward_system_revision_id=reward_revision_id,
                    protocol_revision_id=self._optional_str(
                        payload.get("reward_audit_protocol_revision_id")
                    ),
                    integrity_profile_revision_id=self._optional_str(
                        payload.get("reward_integrity_profile_revision_id")
                    ),
                    domain_kind="run",
                    domain_id=canonical_run_id,
                    role="training_gate",
                    context={
                        "work_item_id": work_item.id,
                        "boundaries": list(payload.get("reward_audit_boundaries") or []),
                        "development_suite_revision_id": self._optional_str(
                            payload.get("development_suite_revision_id")
                        ),
                        "managed": True,
                        "source_reward_integrity_audit_id": self._optional_str(
                            payload.get("source_reward_integrity_audit_id")
                        ),
                        "source_reward_integrity_decision_id": self._optional_str(
                            payload.get("source_reward_integrity_decision_id")
                        ),
                        "fork_checkpoint_hash": self._optional_str(
                            payload.get("fork_checkpoint_hash")
                        ),
                        "fork_checkpoint_occurrence_id": self._optional_str(
                            payload.get("fork_checkpoint_occurrence_id")
                        ),
                    },
                )
            artifact_id = self._optional_str(payload.get("training_artifact_id"))
            if launch_bindings:
                for binding in launch_bindings:
                    self._dataset_database().attach_run_dataset(
                        run_id=canonical_run_id,
                        dataset_version_id=str(binding.get("dataset_version_id") or ""),
                        role=str(binding.get("role") or "train"),
                        split=str(binding.get("split") or binding.get("role") or "train"),
                        training_artifact_id=artifact_id,
                    )
            else:
                version_id = self._optional_str(payload.get("dataset_version_id"))
                if version_id:
                    self._dataset_database().attach_run_dataset(
                        run_id=canonical_run_id,
                        dataset_version_id=version_id,
                        role="train",
                        split=str(payload.get("dataset_split") or "train"),
                        training_artifact_id=artifact_id,
                    )
            if parent_run_id:
                notes = "Forked from resolved launch configuration"
                try:
                    parent_config = self.get_resolved_run_launch_config(parent_run_id)["config"]
                    ignored = {
                        "run_id",
                        "parent_run_id",
                        "output_dir",
                        "training_artifact",
                        "training_artifact_metadata",
                    }
                    current_config = {
                        key: value for key, value in payload.items() if key not in ignored
                    }
                    config_diff = {
                        key: {
                            "parent": parent_config.get(key),
                            "child": current_config.get(key),
                        }
                        for key in sorted(set(parent_config) | set(current_config))
                        if key not in ignored and parent_config.get(key) != current_config.get(key)
                    }
                    notes = json.dumps(
                        {
                            "kind": (
                                "reward_audit_checkpoint_fork"
                                if payload.get("source_reward_integrity_audit_id")
                                else "clone_in_train"
                            ),
                            "config_diff": config_diff,
                            "source_reward_integrity_audit_id": payload.get(
                                "source_reward_integrity_audit_id"
                            ),
                            "source_reward_integrity_decision_id": payload.get(
                                "source_reward_integrity_decision_id"
                            ),
                            "fork_checkpoint_hash": payload.get(
                                "fork_checkpoint_hash"
                            ),
                            "fork_checkpoint_occurrence_id": payload.get(
                                "fork_checkpoint_occurrence_id"
                            ),
                        },
                        sort_keys=True,
                        default=str,
                    )
                except Exception:
                    pass
                self._dataset_database().record_fork(
                    child_run_id=canonical_run_id,
                    parent_run_id=parent_run_id,
                    notes=notes,
                )
            response = self.get_run_detail(job.id, include_research=True, include_internal=False)
            response.update(
                work_item_id=work_item.id,
                accepted=True,
                managed=True,
                queue_position=self._dataset_database().work_item_queue_position(work_item.id),
            )
            if plan_revision_id:
                self._training_plan_engine().bind_run(
                    run_id=canonical_run_id,
                    revision_id=plan_revision_id,
                    capacity_check_id=self._optional_str(
                        payload.get("training_capacity_check_id")
                    ),
                    role="proof" if bool(payload.get("proof_run")) else "full",
                )
                response["training_plan_revision_id"] = plan_revision_id
            if verifier_binding is not None:
                response["verifier_binding"] = verifier_binding.to_dict()
            if reward_binding is not None:
                response["reward_integrity_binding"] = self._reward_value(reward_binding)
            if runtime_binding is not None:
                response["runtime_binding"] = runtime_binding.to_dict()
            return response
        if mode == "sft":
            job_id = await self.training_service.launch_sft(
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                epochs=int(self._value_or_default(payload.get("epochs"), 1)),
                batch_size=int(self._value_or_default(payload.get("batch_size"), 2)),
                gradient_accumulation_steps=int(
                    self._value_or_default(payload.get("gradient_accumulation_steps"), 4)
                ),
                max_samples=self._optional_int(payload.get("max_samples")),
                learning_rate=float(self._value_or_default(payload.get("learning_rate"), 2e-4)),
                seed=int(self._value_or_default(payload.get("seed"), 42)),
                no_caffeinate=bool(payload.get("no_caffeinate", False)),
                accelerator=self._optional_str(payload.get("accelerator")),
                source_ui_page="/public/train",
                dataset_version_id=self._optional_str(payload.get("dataset_version_id")),
                dataset_split=self._optional_str(payload.get("dataset_split")),
                dataset_version_metadata=dataset_version_metadata,
                validation_file=self._optional_str(payload.get("validation_file")),
                dataset_bindings=launch_bindings,
                training_artifact_metadata=artifact_metadata,
                parent_run_id=parent_run_id,
                run_id=canonical_run_id,
            )
        elif mode == "raft":
            job_id = await self.training_service.launch_raft(
                model=str(payload.get("model") or ""),
                prompts=str(payload.get("prompts") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                verifier=str(payload.get("verifier") or "humaneval"),
                cycles=int(self._value_or_default(payload.get("cycles"), 1)),
                samples_per_prompt=int(
                    self._value_or_default(payload.get("samples_per_prompt"), 4)
                ),
                temperature=float(self._value_or_default(payload.get("temperature"), 0.7)),
                keep_percent=float(self._value_or_default(payload.get("keep_percent"), 0.5)),
                reward_threshold=float(
                    self._value_or_default(payload.get("reward_threshold"), 0.5)
                ),
                min_samples=int(self._value_or_default(payload.get("min_samples"), 1)),
                max_new_tokens=int(self._value_or_default(payload.get("max_new_tokens"), 512)),
                max_prompts=self._optional_int(payload.get("max_prompts")),
                seed=int(self._value_or_default(payload.get("seed"), 42)),
                no_caffeinate=bool(payload.get("no_caffeinate", False)),
                accelerator=self._optional_str(payload.get("accelerator")),
                source_ui_page="/public/train",
                dataset_version_id=self._optional_str(payload.get("dataset_version_id")),
                dataset_split=self._optional_str(payload.get("dataset_split")),
                dataset_version_metadata=dataset_version_metadata,
                dataset_bindings=launch_bindings,
                training_artifact_metadata=artifact_metadata,
                parent_run_id=parent_run_id,
                run_id=canonical_run_id,
            )
        elif mode in {"dpo", "orpo", "rm", "grpo"}:
            default_lr = {
                "dpo": 5e-6,
                "orpo": 8e-6,
                "rm": 1e-5,
                "grpo": 1e-6,
            }[mode]
            default_batch = 4 if mode == "rm" else 1
            default_grad_accum = 4 if mode == "rm" else 16
            job_id = await self.training_service.launch_preference_train(
                mode=mode,
                model=str(payload.get("model") or ""),
                dataset=str(payload.get("dataset") or ""),
                output_dir=str(payload.get("output_dir") or ""),
                epochs=int(self._value_or_default(payload.get("epochs"), 1)),
                batch_size=int(self._value_or_default(payload.get("batch_size"), default_batch)),
                gradient_accumulation_steps=int(
                    self._value_or_default(
                        payload.get("gradient_accumulation_steps"), default_grad_accum
                    )
                ),
                learning_rate=float(
                    self._value_or_default(payload.get("learning_rate"), default_lr)
                ),
                seed=int(self._value_or_default(payload.get("seed"), 42)),
                max_samples=self._optional_int(payload.get("max_samples")),
                beta=self._optional_float(payload.get("beta")),
                loss_type=self._optional_str(payload.get("loss_type")),
                reference_free=bool(payload.get("reference_free", False)),
                verifier=self._optional_str(payload.get("verifier")),
                num_generations=self._optional_int(payload.get("num_generations")),
                epsilon=self._optional_float(payload.get("epsilon")),
                temperature=self._optional_float(payload.get("temperature")),
                reward_threshold=self._optional_float(payload.get("reward_threshold")),
                no_caffeinate=bool(payload.get("no_caffeinate", False)),
                accelerator=self._optional_str(payload.get("accelerator")),
                source_ui_page="/public/train",
                dataset_version_id=self._optional_str(payload.get("dataset_version_id")),
                dataset_split=self._optional_str(payload.get("dataset_split")),
                dataset_version_metadata=dataset_version_metadata,
                validation_file=self._optional_str(payload.get("validation_file")),
                dataset_bindings=launch_bindings,
                training_artifact_metadata=artifact_metadata,
                parent_run_id=parent_run_id,
                run_id=canonical_run_id,
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
                no_caffeinate=bool(payload.get("no_caffeinate", False)),
                source_ui_page="/public/train",
                dataset_version_id=self._optional_str(payload.get("dataset_version_id")),
                dataset_split=self._optional_str(payload.get("dataset_split")),
                dataset_version_metadata=dataset_version_metadata,
                dataset_bindings=launch_bindings,
                training_artifact_metadata=artifact_metadata,
                parent_run_id=parent_run_id,
                run_id=canonical_run_id,
            )
        else:
            raise ValueError(f"Unsupported training mode: {mode}")
        artifact_id = self._optional_str(payload.get("training_artifact_id"))
        if launch_bindings:
            for binding in launch_bindings:
                self._dataset_database().attach_run_dataset(
                    run_id=str(job_id),
                    dataset_version_id=str(binding.get("dataset_version_id") or ""),
                    role=str(binding.get("role") or "train"),
                    split=str(binding.get("split") or binding.get("role") or "train"),
                    training_artifact_id=artifact_id,
                )
        else:
            version_id = self._optional_str(payload.get("dataset_version_id"))
            if version_id:
                self._dataset_database().attach_run_dataset(
                    run_id=str(job_id),
                    dataset_version_id=version_id,
                    role="train",
                    split=str(payload.get("dataset_split") or "train"),
                    training_artifact_id=artifact_id,
                )
        if parent_run_id:
            notes = "Forked from resolved launch configuration"
            try:
                parent_config = self.get_resolved_run_launch_config(parent_run_id)["config"]
                ignored = {
                    "run_id",
                    "parent_run_id",
                    "output_dir",
                    "training_artifact",
                    "training_artifact_metadata",
                }
                current_config = {
                    key: value for key, value in payload.items() if key not in ignored
                }
                config_diff = {
                    key: {"parent": parent_config.get(key), "child": current_config.get(key)}
                    for key in sorted(set(parent_config) | set(current_config))
                    if key not in ignored and parent_config.get(key) != current_config.get(key)
                }
                notes = json.dumps(
                    {"kind": "clone_in_train", "config_diff": config_diff},
                    sort_keys=True,
                    default=str,
                )
            except Exception:
                pass
            self._dataset_database().record_fork(
                child_run_id=str(job_id),
                parent_run_id=parent_run_id,
                notes=notes,
            )
        reward_revision_id = self._optional_str(payload.get("reward_system_revision_id"))
        if reward_revision_id:
            self._reward_integrity_engine().bind(
                reward_system_revision_id=reward_revision_id,
                protocol_revision_id=self._optional_str(
                    payload.get("reward_audit_protocol_revision_id")
                ),
                integrity_profile_revision_id=self._optional_str(
                    payload.get("reward_integrity_profile_revision_id")
                ),
                domain_kind="run",
                domain_id=str(job_id),
                role="training_gate",
                context={
                    "boundaries": list(payload.get("reward_audit_boundaries") or []),
                    "managed": False,
                },
            )
        if plan_revision_id:
            self._training_plan_engine().bind_run(
                run_id=str(job_id),
                revision_id=plan_revision_id,
                capacity_check_id=self._optional_str(
                    payload.get("training_capacity_check_id")
                ),
                role="proof" if bool(payload.get("proof_run")) else "full",
            )
        return self.get_run_detail(job_id, include_research=True, include_internal=False)

    def _guided_version_context(self, version: Any) -> Dict[str, Any]:
        """Resolve immutable own-data lineage attached to a dataset version.

        Guided imports persist their scenario and mapping on the source
        revision.  Versions remain immutable, so that source metadata plus the
        resolved version recipe is sufficient to reproduce the handoff.  The
        recipe fallback keeps versions produced by early v9 builds readable.
        """

        db = self._dataset_database()
        source = db.get_dataset_source(version.source_id) if version.source_id else None
        if source is None:
            sources = db.list_dataset_sources(version.dataset_id)
            source = next(
                (
                    item
                    for item in sources
                    if item.id in dict(version.source_fingerprints or {})
                ),
                sources[0] if len(sources) == 1 else None,
            )
        source_metadata = dict(source.metadata if source is not None else {})
        guided = dict(source_metadata.get("guided_own_data") or {})
        recipe = dict(version.recipe or {})
        map_step = next(
            (
                dict(step)
                for step in list(recipe.get("steps") or [])
                if isinstance(step, Mapping)
                and str(step.get("kind") or step.get("type") or "").strip().lower()
                == "map"
            ),
            {},
        )
        scenario_revision_id = self._optional_str(
            guided.get("scenario_revision_id")
            or map_step.get("scenario_revision_id")
            or recipe.get("scenario_revision_id")
        )
        mappings = dict(map_step.get("fields") or guided.get("field_mapping") or {})
        mapping_plan = {
            "version": int(map_step.get("mapping_version") or 2),
            "scenario_revision_id": scenario_revision_id,
            "confirmed": True,
            "mappings": mappings,
        }
        return {
            "scenario_revision_id": scenario_revision_id,
            "field_mapping_plan": mapping_plan,
            "dataset_preparation_recipe": recipe,
            "source_id": source.id if source is not None else version.source_id,
            "import_id": self._optional_str(guided.get("import_id")),
        }

    @staticmethod
    def _proof_sample_identity(
        *,
        version: Any,
        scenario_revision_id: str,
        trainer_mode: str,
        split: str,
        max_samples: int,
        seed: int,
    ) -> str:
        """Fingerprint the deterministic prefix consumed by a proof run."""

        value = {
            "format_version": 1,
            "dataset_version_id": str(version.id),
            "dataset_content_hash": str(version.content_hash or ""),
            "recipe_hash": str(version.recipe_hash or ""),
            "scenario_revision_id": scenario_revision_id,
            "trainer_mode": trainer_mode,
            "split": split,
            "selection": "ordered_prefix",
            "max_samples": int(max_samples),
            "seed": int(seed),
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    async def launch_dataset_proof_run(
        self, version_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Launch a bounded, deterministic proof run from an immutable version."""

        from halo_forge.own_data.registry import TRAINING_SCENARIOS

        db = self._dataset_database()
        version = db.get_dataset_version(version_id)
        if version is None:
            raise KeyError(version_id)
        if version.status != "completed":
            raise ValueError("The dataset version must finish publication before a proof run")
        dataset = db.get_dataset(version.dataset_id)
        if dataset is None:
            raise KeyError(version.dataset_id)
        if not version.split_counts or int(version.split_counts.get("train", 0)) <= 0:
            raise ValueError("The dataset version has no non-empty train split")

        guided = self._guided_version_context(version)
        requested_scenario = self._optional_str(payload.get("scenario_revision_id"))
        recorded_scenario = self._optional_str(guided.get("scenario_revision_id"))
        if requested_scenario and recorded_scenario and requested_scenario != recorded_scenario:
            raise ValueError(
                "scenario_revision_id conflicts with the immutable dataset source revision"
            )
        scenario_revision_id = requested_scenario or recorded_scenario
        if not scenario_revision_id:
            raise ValueError(
                "scenario_revision_id is required because this version has no guided scenario lineage"
            )
        try:
            scenario = TRAINING_SCENARIOS.get(scenario_revision_id)
        except KeyError as exc:
            raise ValueError(f"Unknown training scenario revision: {scenario_revision_id}") from exc
        if not scenario.available:
            raise ValueError(scenario.unavailable_reason or "This training scenario is unavailable")
        if str(dataset.canonical_schema).lower() != str(scenario.canonical_schema).lower():
            raise ValueError(
                f"Scenario {scenario.revision_id} requires {scenario.canonical_schema!r} data, "
                f"but this version is {dataset.canonical_schema!r}"
            )
        dataset_modality = str(dataset.modality or "text").lower()
        scenario_modality = str(scenario.modality or "text").lower()
        modality_aliases = {"vlm": "image", "image": "image"}
        if modality_aliases.get(dataset_modality, dataset_modality) != modality_aliases.get(
            scenario_modality, scenario_modality
        ):
            raise ValueError(
                f"Scenario {scenario.revision_id} requires {scenario.modality!r} data, "
                f"but this version is {dataset.modality!r}"
            )

        mode = str(payload.get("trainer_mode") or payload.get("mode") or "").strip().lower()
        if not mode:
            mode = str(scenario.trainer_modes[0] if scenario.trainer_modes else "")
        if mode not in set(scenario.trainer_modes):
            raise ValueError(
                f"Trainer {mode!r} is not verified for scenario {scenario.revision_id}"
            )
        model = str(payload.get("model") or "").strip()
        if not model:
            raise ValueError("model is required")

        verifier_revision_id = self._optional_str(
            payload.get("verifier_profile_revision_id")
        )
        if mode in {"raft", "grpo"}:
            if not verifier_revision_id:
                raise ValueError(
                    f"Guided {mode.upper()} proof training requires a candidate- or "
                    "approved-qualified verifier profile revision"
                )
            try:
                self._verifier_engine().resolve_binding(
                    verifier_revision_id,
                    modality="text",
                    require_qualified=True,
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"The selected verifier is not ready for guided {mode.upper()} use: {exc}"
                ) from exc

        guided_engine = self._guided_data_engine()
        try:
            mode_compatibility = guided_engine.runtime_trainer_compatibility(
                scenario,
                self._active_backend_name(),
                trainer_mode=mode,
            )
            selected_runtime = next(
                (
                    item
                    for item in mode_compatibility
                    if str(item.get("trainer_mode") or "").lower() == mode
                ),
                None,
            )
            runtime_available = bool(
                selected_runtime and selected_runtime.get("compatible")
            )
            unavailable_reason = (
                str(selected_runtime.get("reason") or "")
                if selected_runtime
                else f"Trainer {mode!r} has no verified runtime contract."
            )
        except (AttributeError, TypeError):
            # Compatibility for embedders that still expose the v9 aggregate
            # facade. The production service always resolves the exact mode.
            runtime_available, unavailable_reason = guided_engine._runtime_scenario_status(
                scenario, self._active_backend_name()
            )
        if not runtime_available:
            raise ValueError(
                unavailable_reason
                or f"Trainer {mode!r} is unavailable on the active runtime"
            )

        budget = int(dict(scenario.proof_budget or {}).get("max_samples") or 200)
        requested_cap = payload.get("max_prompts", payload.get("limit", payload.get("max_samples")))
        if requested_cap is None:
            requested_cap = budget
        try:
            proof_cap = int(requested_cap)
        except (TypeError, ValueError) as exc:
            raise ValueError("proof sample limit must be an integer") from exc
        if proof_cap <= 0:
            raise ValueError("proof sample limit must be greater than zero")
        proof_cap = min(proof_cap, budget)
        proof_cap = min(proof_cap, int(version.split_counts.get("train") or proof_cap))
        seed = int(dict(scenario.proof_budget or {}).get("seed") or 42)

        resolved = dict(payload)
        for key in (
            "trainer_mode",
            "max_samples",
            "max_prompts",
            "limit",
            "max_steps",
            "run_id",
            "parent_run_id",
            "proof_parent_run_id",
            "full_run_from_proof",
            "full_run_reason",
            "dataset",
            "prompts",
            "validation_file",
        ):
            resolved.pop(key, None)
        resolved.update(
            {
                "mode": mode,
                "model": model,
                "dataset_version_id": version.id,
                "dataset_split": "train",
                "dataset_bindings": [
                    {
                        "role": "train",
                        "dataset_version_id": version.id,
                        "split": "train",
                    }
                ],
                "seed": seed,
                "proof_run": True,
                "scenario_revision_id": scenario.revision_id,
                "field_mapping_plan": dict(guided["field_mapping_plan"]),
                "dataset_preparation_recipe": dict(version.recipe or {}),
                "proof_max_samples": proof_cap,
            }
        )
        if mode == "classify" and (
            scenario.id == "text-multilabel"
            or "multi-label" in str(scenario.task or "").lower()
            or "multilabel" in str(scenario.task or "").lower()
        ):
            resolved["multi_label"] = True
        if int(version.split_counts.get("validation", 0)) > 0:
            resolved["dataset_bindings"].append(
                {
                    "role": "validation",
                    "dataset_version_id": version.id,
                    "split": "validation",
                }
            )
        resolved["proof_sample_identity"] = self._proof_sample_identity(
            version=version,
            scenario_revision_id=scenario.revision_id,
            trainer_mode=mode,
            split="train",
            max_samples=proof_cap,
            seed=seed,
        )

        # The route owns proof-run limits.  Client-supplied larger epoch/cycle
        # values cannot silently turn the smoke test into a full run.
        if mode in {"sft", "dpo", "orpo", "rm", "classify", "embed", "rerank"}:
            resolved["epochs"] = 1
            resolved["max_samples"] = proof_cap
            resolved.pop("cycles", None)
        elif mode == "grpo":
            resolved["epochs"] = 1
            resolved["max_samples"] = proof_cap
            resolved["max_steps"] = 1
            resolved.pop("cycles", None)
        elif mode == "raft":
            resolved["cycles"] = 1
            resolved["max_prompts"] = proof_cap
            resolved.pop("epochs", None)
        else:
            resolved["cycles"] = 1
            resolved["limit"] = proof_cap
            resolved.pop("epochs", None)
        if mode == "audio":
            resolved["task"] = "asr"

        selected_root = str(
            resolved.get("output_root")
            or resolved.get("output_dir")
            or _default_run_root()
        )
        resolved["output_root"] = selected_root
        # launch_training allocates the canonical <root>/<run-id> directory.
        resolved["output_dir"] = selected_root
        result = await self.launch_training(resolved)
        result.update(
            proof_run=True,
            dataset_version_id=version.id,
            scenario_revision_id=scenario.revision_id,
            proof_max_samples=proof_cap,
            proof_sample_identity=resolved["proof_sample_identity"],
        )
        return result

    async def launch_full_run_from_proof(
        self, run_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Clone a completed proof run, removing only its proof boundaries."""

        values = dict(payload or {})
        unsupported = sorted(
            key
            for key, value in values.items()
            if key
            not in {
                "reason",
                "output_root",
                "assessment_id",
                "override_reason",
            }
            and self._has_public_value(value)
        )
        if unsupported:
            raise ValueError(
                "Full-run launch only accepts assessment_id, override_reason, "
                "reason, and output_root; "
                f"unsupported fields: {', '.join(unsupported)}"
            )
        detail = self.get_run_detail(run_id, include_research=False, include_internal=False)
        if str(detail.get("status") or "").lower() not in {"completed", "succeeded"}:
            raise ValueError("The proof run must complete successfully before starting a full run")
        resolved_record = self.get_resolved_run_launch_config(run_id)
        parent_id = str(resolved_record.get("run_id") or run_id)
        resolved = dict(resolved_record.get("config") or {})
        if not bool(resolved.get("proof_run")):
            raise ValueError("This run was not launched as a guided proof run")

        assessment_id = self._optional_str(values.get("assessment_id"))
        override_reason = self._optional_str(values.get("override_reason"))
        if not assessment_id and not override_reason:
            # Keep ``reason`` backward-compatible as the explicit override field
            # for older dashboard/CLI clients.
            override_reason = self._optional_str(values.get("reason"))
        outcome_context = self.get_full_run_context(
            parent_id,
            assessment_id=assessment_id,
            override_reason=override_reason,
        )
        proof_plan = self._training_plan_engine().run_binding(parent_id)
        full_plan_revision = None
        if proof_plan and proof_plan.get("revision"):
            full_plan_revision = self._training_plan_engine().derive_full_run_revision(
                str(proof_plan["revision"]["id"])
            )

        output_root = str(values.get("output_root") or resolved.get("output_root") or "").strip()
        if not output_root:
            prior_output = str(resolved.get("output_dir") or detail.get("output_dir") or "")
            output_root = str(Path(prior_output).expanduser().parent) if prior_output else str(
                _default_run_root()
            )
        for key in (
            "run_id",
            "max_samples",
            "max_prompts",
            "limit",
            "max_steps",
            "proof_sample_identity",
            "proof_max_samples",
        ):
            resolved.pop(key, None)
        resolved.update(
            {
                "output_root": output_root,
                "output_dir": output_root,
                "parent_run_id": parent_id,
                "proof_parent_run_id": parent_id,
                "proof_run": False,
                "full_run_from_proof": True,
                "outcome_assessment_id": assessment_id,
                "outcome_override_reason": override_reason,
            }
        )
        if full_plan_revision is not None:
            self._training_plan_engine().record_decision(
                full_plan_revision.id,
                "confirmed",
                reason=override_reason or str(values.get("reason") or "").strip() or None,
                details={
                    "confirmation_surface": "full_run_handoff",
                    "proof_plan_revision_id": str(proof_plan["revision"]["id"]),
                    "outcome_assessment_id": assessment_id,
                    "proof_cap_removed": True,
                },
            )
            # The uncapped immutable revision is authoritative. Remove cloned
            # low-level plan fields so launch resolution cannot see them as
            # contradictory operator overrides.
            resolved = {
                "training_plan_revision_id": full_plan_revision.id,
                "output_root": output_root,
                "output_dir": output_root,
                "parent_run_id": parent_id,
                "proof_parent_run_id": parent_id,
                "full_run_from_proof": True,
                "outcome_assessment_id": assessment_id,
                "outcome_override_reason": override_reason,
            }
        reason = str(values.get("reason") or "").strip()
        if reason:
            resolved["full_run_reason"] = reason
        result = await self.launch_training(resolved)
        self.review_training_outcome(
            parent_id,
            {
                "assessment_id": assessment_id,
                "decision": "start_full_run" if assessment_id else "override",
                "reason": reason or override_reason or "",
                "full_run_id": result.get("run_id"),
                "context": {
                    "resolved_outcome_context": outcome_context,
                },
            },
        )
        result.update(
            proof_run=False,
            full_run_from_proof=True,
            parent_run_id=parent_id,
            training_plan_revision_id=(
                full_plan_revision.id if full_plan_revision is not None else None
            ),
        )
        return result

    def _dataset_version_metadata(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        version_id = self._optional_str(payload.get("dataset_version_id"))
        if not version_id:
            return None
        version = self._dataset_database().get_dataset_version(version_id)
        if version is None:
            return None
        source = (
            self._dataset_database().get_dataset_source(version.source_id)
            if version.source_id
            else None
        )
        source_guidance = (
            dict((source.metadata or {}).get("guided_own_data") or {})
            if source is not None
            else {}
        )
        guided = self._guided_version_context(version)
        return {
            "version_id": version.id,
            "dataset_id": version.dataset_id,
            "content_hash": version.content_hash,
            "recipe_hash": version.recipe_hash,
            "split": str(payload.get("dataset_split") or "train"),
            "source_fingerprints": version.source_fingerprints,
            "assets_materialized": version.assets_materialized,
            "bindings": list(payload.get("dataset_bindings") or []),
            "training_artifact": dict(payload.get("training_artifact_metadata") or {}),
            "corpus_extraction": copy.deepcopy(
                source_guidance.get("corpus_extraction") or {}
            ),
            "guided_own_data": {
                "scenario_revision_id": self._optional_str(
                    payload.get("scenario_revision_id")
                    or guided.get("scenario_revision_id")
                ),
                "field_mapping_plan": dict(
                    payload.get("field_mapping_plan")
                    or guided.get("field_mapping_plan")
                    or {}
                ),
                "dataset_preparation_recipe": dict(
                    payload.get("dataset_preparation_recipe")
                    or guided.get("dataset_preparation_recipe")
                    or {}
                ),
                "proof_run": bool(payload.get("proof_run", False)),
                "proof_sample_identity": self._optional_str(
                    payload.get("proof_sample_identity")
                ),
                "proof_max_samples": self._optional_int(
                    payload.get("proof_max_samples")
                ),
                "proof_parent_run_id": self._optional_str(
                    payload.get("proof_parent_run_id")
                ),
                "full_run_from_proof": bool(payload.get("full_run_from_proof", False)),
            },
        }

    @staticmethod
    def _training_artifact_request(
        payload: Dict[str, Any], bindings: List[Any], mode: str
    ) -> Dict[str, Any]:
        train_binding = next(binding for binding in bindings if binding.role == "train")
        options: Dict[str, Any] = {
            "trainer_mode": mode,
            "adapter_id": PublicApiService._optional_str(
                payload.get("dataset_adapter_id")
            ),
            "model": PublicApiService._optional_str(payload.get("model")),
            "tokenizer_revision": PublicApiService._optional_str(
                payload.get("tokenizer_revision")
            ),
            "chat_template": PublicApiService._optional_str(
                payload.get("chat_template")
            ),
            "validation_fraction": float(payload.get("validation_fraction", 0.05)),
            "seed": int(payload.get("seed", 42)),
        }
        if mode == "cpt":
            effective_tokenizer_revision = PublicApiService._optional_str(
                payload.get("tokenizer_revision")
            ) or PublicApiService._optional_str(payload.get("model_revision"))
            options["tokenizer_revision"] = effective_tokenizer_revision
            options.update(
                {
                    "model_revision": PublicApiService._optional_str(
                        payload.get("model_revision")
                    ),
                    "model_hash": PublicApiService._optional_str(
                        payload.get("model_hash")
                    ),
                    "tokenizer_hash": PublicApiService._optional_str(
                        payload.get("tokenizer_hash")
                    ),
                    "max_sequence_length": int(
                        payload.get("max_sequence_length") or 2048
                    ),
                    "packing": str(
                        payload.get("packing")
                        or "paragraph_eos_non_overlap_v1"
                    ),
                    "budget_mode": str(payload.get("budget_mode") or "passes"),
                    "target_tokens": PublicApiService._optional_int(
                        payload.get("target_tokens")
                    ),
                    "corpus_passes": PublicApiService._optional_float(
                        payload.get("corpus_passes")
                    ),
                    "effective_batch_size": int(
                        payload.get("effective_batch_size")
                        or (
                            int(payload.get("batch_size") or 1)
                            * int(
                                payload.get("gradient_accumulation_steps")
                                or 8
                            )
                        )
                    ),
                }
            )
        return {
            "version_id": train_binding.dataset_version_id,
            "bindings": [binding.to_dict() for binding in bindings],
            "options": options,
        }

    @staticmethod
    def _training_artifact_requests_match(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        def binding_identity(value: Dict[str, Any]) -> Dict[str, str]:
            return {
                "role": str(value.get("role") or "train"),
                "dataset_version_id": str(value.get("dataset_version_id") or ""),
                "split": str(value.get("split") or "train"),
            }

        def normalized(value: Dict[str, Any]) -> Dict[str, Any]:
            options = dict(value.get("options") or {})
            options.setdefault("adapter_id", None)
            options.setdefault("model", None)
            options.setdefault("tokenizer_revision", None)
            options.setdefault("chat_template", None)
            options.setdefault("validation_fraction", 0.05)
            options.setdefault("seed", 42)
            if str(options.get("trainer_mode") or "") == "cpt":
                options.setdefault("model_revision", None)
                options.setdefault("model_hash", None)
                options.setdefault("tokenizer_hash", None)
                options.setdefault("max_sequence_length", 2048)
                options.setdefault(
                    "packing", "paragraph_eos_non_overlap_v1"
                )
                options.setdefault("budget_mode", "passes")
                options.setdefault("target_tokens", None)
                options.setdefault("corpus_passes", 1.0)
                options.setdefault("effective_batch_size", 8)
            return {
                "version_id": str(value.get("version_id") or ""),
                "bindings": [binding_identity(binding) for binding in value.get("bindings") or []],
                "options": options,
            }

        return normalized(left) == normalized(right)

    def _verified_training_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        record = self._dataset_database().get_training_artifact(artifact_id)
        if record is None:
            return None
        engine = self._dataset_engine()
        catalog = getattr(engine, "training_artifacts", None)
        verifier = getattr(catalog, "verify", None)
        if callable(verifier):
            try:
                verification = self._data_object(verifier(artifact_id))
            except Exception:
                return None
            if verification.get("valid") is False or verification.get("ok") is False:
                return None
        else:
            getter = getattr(catalog, "get", None)
            if callable(getter):
                try:
                    getter(artifact_id)
                except Exception:
                    return None
            elif not Path(record.manifest_path).is_file():
                return None
        artifact = self.get_training_dataset_artifact(artifact_id)
        if artifact is None or artifact.get("status") != "ready":
            return None
        return artifact

    @staticmethod
    def _artifact_matches_training_request(
        artifact: Dict[str, Any], request: Dict[str, Any]
    ) -> bool:
        options = dict(request.get("options") or {})
        if str(artifact.get("trainer_mode") or "") != str(options.get("trainer_mode") or ""):
            return False
        requested_adapter = PublicApiService._optional_str(options.get("adapter_id"))
        if requested_adapter and str(artifact.get("adapter_id") or "") != requested_adapter:
            return False
        if PublicApiService._optional_str(
            artifact.get("model") or artifact.get("model_id")
        ) != PublicApiService._optional_str(options.get("model")):
            return False
        if PublicApiService._optional_str(
            artifact.get("tokenizer_revision")
        ) != PublicApiService._optional_str(options.get("tokenizer_revision")):
            return False
        if str(options.get("trainer_mode") or "") == "cpt":
            for artifact_field, option_field in (
                ("model_revision", "model_revision"),
                ("model_hash", "model_hash"),
                ("tokenizer_hash", "tokenizer_hash"),
            ):
                requested = PublicApiService._optional_str(
                    options.get(option_field)
                )
                if requested and PublicApiService._optional_str(
                    artifact.get(artifact_field)
                ) != requested:
                    return False
            plan = dict(artifact.get("packing_plan") or {})
            expected = {
                "max_sequence_length": int(
                    options.get("max_sequence_length") or 2048
                ),
                "packing": str(
                    options.get("packing")
                    or "paragraph_eos_non_overlap_v1"
                ),
                "budget_mode": str(options.get("budget_mode") or "passes"),
                "target_tokens": PublicApiService._optional_int(
                    options.get("target_tokens")
                ),
                "corpus_passes": PublicApiService._optional_float(
                    options.get("corpus_passes")
                ),
                "effective_batch_size": int(
                    options.get("effective_batch_size") or 8
                ),
            }
            for key, value in expected.items():
                actual = plan.get(key)
                if isinstance(value, float) and actual is not None:
                    if abs(float(actual) - value) > 1e-12:
                        return False
                elif actual != value:
                    return False

        def binding_identity(value: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "role": str(value.get("role") or "train"),
                "dataset_version_id": str(value.get("dataset_version_id") or ""),
                "split": str(value.get("split") or "train"),
            }

        expected_bindings = [binding_identity(value) for value in request.get("bindings") or []]
        actual_bindings = [binding_identity(value) for value in artifact.get("bindings") or []]
        if actual_bindings != expected_bindings:
            # The renderer automatically records sibling validation/test/
            # canary splits so supplied validation is preserved and held-out
            # identity remains auditable. Those implicit bindings are allowed
            # after the exact requested prefix; any extra train binding or
            # cross-version sibling remains a mismatch.
            requested_versions = {
                value["dataset_version_id"] for value in expected_bindings
            }
            implicit = actual_bindings[len(expected_bindings) :]
            if (
                actual_bindings[: len(expected_bindings)] != expected_bindings
                or not implicit
                or any(
                    value["role"] not in {"validation", "test", "canary"}
                    or value["dataset_version_id"] not in requested_versions
                    for value in implicit
                )
            ):
                return False

        validation_policy = dict(artifact.get("validation_policy") or {})
        supplied_validation = any(
            value["role"] == "validation" for value in actual_bindings
        )
        if supplied_validation:
            return validation_policy.get("kind") == "supplied"
        if validation_policy.get("kind") == "derived":
            return (
                int(validation_policy.get("seed", -1)) == int(options.get("seed", 42))
                and abs(
                    float(validation_policy.get("fraction", -1.0))
                    - float(options.get("validation_fraction", 0.05))
                )
                <= 1e-12
            )
        # A renderer can legitimately produce no validation split for a
        # one-record dataset. In that case seed/fraction cannot change output.
        return "validation" not in dict(artifact.get("split_paths") or artifact.get("paths") or {})

    def _find_verified_training_artifact(
        self, request: Dict[str, Any], *, explicit_artifact_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        db = self._dataset_database()
        if explicit_artifact_id:
            candidates = [db.get_training_artifact(explicit_artifact_id)]
        else:
            candidates = db.list_training_artifacts(
                dataset_version_id=str(request.get("version_id") or ""),
                trainer_mode=str((request.get("options") or {}).get("trainer_mode") or ""),
                limit=500,
            )
        for record in candidates:
            if record is None:
                continue
            raw = record.to_dict()
            metadata = raw.get("metadata")
            if isinstance(metadata, dict):
                raw = {**raw, **metadata}
            if not self._artifact_matches_training_request(raw, request):
                continue
            verified = self._verified_training_artifact(record.id)
            if verified is not None and self._artifact_matches_training_request(verified, request):
                return verified
        return None

    @staticmethod
    def _apply_training_artifact_payload(
        payload: Dict[str, Any], artifact: Dict[str, Any]
    ) -> Dict[str, Any]:
        bindings = list(artifact.get("bindings") or [])
        train_binding = next(
            (binding for binding in bindings if binding.get("role") == "train"),
            None,
        )
        if train_binding is None:
            raise ValueError("Training artifact did not expose a train binding")
        paths = dict(artifact.get("split_paths") or artifact.get("paths") or {})
        dataset_path = paths.get("train")
        if not dataset_path:
            raise ValueError("Training artifact did not expose a train split")
        resolved = dict(payload)
        resolved["dataset_version_id"] = str(train_binding.get("dataset_version_id") or "")
        resolved["dataset_split"] = str(train_binding.get("split") or "train")
        resolved["dataset_bindings"] = bindings
        resolved["training_artifact_id"] = str(
            artifact.get("artifact_id") or artifact.get("id") or ""
        )
        resolved["training_artifact_metadata"] = artifact
        if str(payload.get("mode") or "").strip().lower() == "cpt":
            resolved["training_artifact_hash"] = str(
                artifact.get("artifact_hash") or ""
            )
            resolved["model"] = artifact.get("model") or artifact.get("model_id")
            resolved["model_revision"] = artifact.get("model_revision")
            resolved["model_hash"] = artifact.get("model_hash")
            resolved["tokenizer_revision"] = artifact.get("tokenizer_revision")
            resolved["tokenizer_hash"] = artifact.get("tokenizer_hash")
            resolved["expected_packing_plan_hash"] = artifact.get(
                "packing_plan_hash"
            )
        if paths.get("validation"):
            resolved["validation_file"] = paths["validation"]
        if str(payload.get("mode") or "").strip().lower() == "raft":
            resolved["prompts"] = dataset_path
        else:
            resolved["dataset"] = dataset_path
        return resolved

    def _training_artifact_pending_payload(
        self, payload: Dict[str, Any], job: Any, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        preparation = self._training_artifact_job_view(job)
        preparation["job_url"] = f"/api/public/dataset-jobs/{job.id}"
        return {
            "_managed_dataset_pending": True,
            "status": "preparing_dataset",
            "ok": True,
            "accepted": True,
            "ready": False,
            "resolved_paths": {},
            "errors": [],
            "warnings": [],
            "suggested_fixes": [],
            "job_id": job.id,
            "run_id": self._optional_str(payload.get("run_id")),
            "mode": str(payload.get("mode") or ""),
            "dataset_version_id": request.get("version_id"),
            "dataset_bindings": list(request.get("bindings") or []),
            "artifact_preparation": preparation,
            "message": "Training dataset artifact preparation is in progress.",
        }

    @staticmethod
    def _public_training_preparation_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in payload.items() if key != "_managed_dataset_pending"}

    def _prepare_managed_dataset_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from halo_forge.data_lab import DatasetBinding

        raw_bindings = payload.get("dataset_bindings") or []
        if raw_bindings and not isinstance(raw_bindings, list):
            raise ValueError("dataset_bindings must be a list")
        bindings = [DatasetBinding.from_value(value) for value in raw_bindings]
        version_id = str(payload.get("dataset_version_id") or "").strip()
        if version_id and not any(binding.role == "train" for binding in bindings):
            bindings.insert(
                0,
                DatasetBinding(
                    role="train",
                    dataset_version_id=version_id,
                    split=str(payload.get("dataset_split") or "train"),
                ),
            )
        if not bindings:
            return dict(payload)
        if not any(binding.role == "train" for binding in bindings):
            raise ValueError("dataset_bindings requires at least one train role")
        mode = str(payload.get("mode") or "").strip().lower()
        resolved_bindings = []
        db = self._dataset_database()
        for binding in bindings:
            version = db.get_dataset_version(binding.dataset_version_id)
            if version is None:
                raise ValueError(f"Unknown dataset version: {binding.dataset_version_id}")
            if version.status != "completed":
                raise ValueError(f"Dataset version {binding.dataset_version_id} is not complete")
            if version.split_counts and binding.split not in version.split_counts:
                raise ValueError(
                    f"Dataset version {binding.dataset_version_id} has no {binding.split!r} split"
                )
            try:
                verification = self._data_object(
                    self._dataset_engine().verify_version(
                        binding.dataset_version_id,
                        dataset_id=version.dataset_id,
                        verify_source=not version.assets_materialized,
                    )
                )
            except (AttributeError, KeyError):
                verification = {"valid": True}
            if verification.get("valid") is False or verification.get("ok") is False:
                problems = (
                    verification.get("problems")
                    or verification.get("errors")
                    or verification.get("missing_assets")
                    or []
                )
                detail = "; ".join(str(item) for item in problems) or "version verification failed"
                raise ValueError(
                    f"Dataset version {binding.dataset_version_id} is not usable: {detail}"
                )
            resolved_bindings.append(
                DatasetBinding(
                    role=binding.role,
                    dataset_version_id=binding.dataset_version_id,
                    split=binding.split,
                    dataset_id=version.dataset_id,
                )
            )

        from halo_forge.data_lab import TRAINER_DATASET_ADAPTERS

        schemas = set()
        for binding in resolved_bindings:
            record = db.get_dataset_version(binding.dataset_version_id)
            dataset = db.get_dataset(record.dataset_id) if record else None
            if dataset:
                schemas.add(dataset.canonical_schema)
        if len(schemas) != 1:
            raise ValueError("Managed bindings must use one canonical schema")
        try:
            TRAINER_DATASET_ADAPTERS.resolve(
                schema=next(iter(schemas)),
                trainer_mode=mode,
                adapter_id=self._optional_str(payload.get("dataset_adapter_id")),
            )
        except Exception as exc:
            raise ValueError(
                f"Dataset schema {next(iter(schemas))!r} is incompatible with {mode}"
            ) from exc

        engine = self._dataset_engine()
        render_artifact = getattr(engine, "render_training_artifact", None)
        if not callable(render_artifact):
            # Compatibility for injected v1 test/extension facades whose
            # versions predate the managed-root layout. Real DatasetLab
            # instances always take the artifact path below.

            def legacy_split_path(binding: Any) -> str:
                record = db.get_dataset_version(binding.dataset_version_id)
                root = Path(record.storage_path).expanduser()
                candidates = [
                    root / "splits" / f"{binding.split}.jsonl",
                    root / f"{binding.split}.jsonl",
                ]
                if binding.split == "train":
                    candidates.append(root / "records.jsonl")
                match = next((value.resolve() for value in candidates if value.is_file()), None)
                if match is None:
                    raise ValueError(
                        f"Dataset version {binding.dataset_version_id} has no readable "
                        f"file for split {binding.split!r}"
                    )
                return str(match)

            train_binding = next(
                binding for binding in resolved_bindings if binding.role == "train"
            )
            resolved = dict(payload)
            resolved["dataset_version_id"] = train_binding.dataset_version_id
            resolved["dataset_split"] = train_binding.split
            resolved["dataset_bindings"] = [binding.to_dict() for binding in resolved_bindings]
            train_path = legacy_split_path(train_binding)
            validation_binding = next(
                (binding for binding in resolved_bindings if binding.role == "validation"), None
            )
            if validation_binding:
                resolved["validation_file"] = legacy_split_path(validation_binding)
            if mode == "raft":
                resolved["prompts"] = train_path
            else:
                resolved["dataset"] = train_path
            return resolved
        request = self._training_artifact_request(payload, resolved_bindings, mode)
        explicit_artifact_id = self._optional_str(payload.get("training_artifact_id"))
        with self._training_artifact_preparation_lock:
            artifact = self._find_verified_training_artifact(
                request, explicit_artifact_id=explicit_artifact_id
            )
            if explicit_artifact_id and artifact is None:
                raise ValueError(
                    f"Training artifact {explicit_artifact_id} is missing, invalid, or does not "
                    "match the requested dataset bindings and trainer configuration"
                )
            if artifact is not None:
                return self._apply_training_artifact_payload(payload, artifact)

            primary_binding = next(
                binding for binding in resolved_bindings if binding.role == "train"
            )
            db_jobs = self._dataset_database().list_dataset_jobs(
                dataset_id=str(primary_binding.dataset_id or ""), limit=1000
            )
            for raw_job in db_jobs:
                if raw_job.job_type != "training_artifact":
                    continue
                if not self._training_artifact_requests_match(raw_job.request, request):
                    continue
                job = self._sync_dataset_job(raw_job.id) or raw_job
                if job.status == "completed":
                    artifact_id = str(job.checkpoint.get("training_artifact_id") or "")
                    completed_artifact = (
                        self._find_verified_training_artifact(
                            request, explicit_artifact_id=artifact_id
                        )
                        if artifact_id
                        else None
                    )
                    if completed_artifact is not None:
                        return self._apply_training_artifact_payload(payload, completed_artifact)
                    continue
                if job.status in {"queued", "running"}:
                    return self._training_artifact_pending_payload(payload, job, request)

            options = dict(request["options"])
            started = self.create_training_dataset_artifact(
                str(request["version_id"]),
                {"bindings": request["bindings"], **options},
            )
            job_id = str(started.get("id") or started.get("job_id") or "")
            job = self._dataset_database().get_dataset_job(job_id)
            if job is None:
                raise ValueError("Dataset Lab did not persist the training-artifact job")
            return self._training_artifact_pending_payload(payload, job, request)

    @classmethod
    def _normalize_public_training_aliases(
        cls, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        payload = dict(payload)
        if cls._has_public_value(payload.get("model_name")) and not cls._has_public_value(
            payload.get("model")
        ):
            payload["model"] = payload["model_name"]
        payload.pop("model_name", None)
        if cls._has_public_value(payload.get("max_seq_length")) and not cls._has_public_value(
            payload.get("max_sequence_length")
        ):
            payload["max_sequence_length"] = payload["max_seq_length"]
        payload.pop("max_seq_length", None)
        if cls._has_public_value(payload.get("output")):
            if not cls._has_public_value(payload.get("output_root")) and not cls._has_public_value(
                payload.get("output_dir")
            ):
                payload["output_root"] = payload["output"]
            payload.pop("output", None)
        if str(payload.get("mode") or payload.get("trainer_mode") or "").strip().lower() == "cpt":
            payload = cls._normalize_cpt_batch_contract(payload)
        return payload

    @classmethod
    def _normalize_cpt_batch_contract(
        cls, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Resolve one effective-batch identity for render, preflight, and argv.

        The guided corpus form exposes an effective batch size while Advanced
        callers may provide micro-batch and accumulation separately.  Artifact
        estimates and the trainer must never silently use different products.
        """

        resolved = dict(payload)
        has_effective = cls._has_public_value(
            resolved.get("effective_batch_size")
        )
        has_batch = cls._has_public_value(resolved.get("batch_size"))
        has_accumulation = cls._has_public_value(
            resolved.get("gradient_accumulation_steps")
        )
        try:
            effective = (
                int(resolved["effective_batch_size"])
                if has_effective
                else None
            )
            batch = int(resolved["batch_size"]) if has_batch else None
            accumulation = (
                int(resolved["gradient_accumulation_steps"])
                if has_accumulation
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CPT batch size and gradient accumulation must be integers"
            ) from exc
        for name, value in (
            ("effective_batch_size", effective),
            ("batch_size", batch),
            ("gradient_accumulation_steps", accumulation),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"CPT {name} must be positive")

        if effective is not None:
            if batch is not None and accumulation is not None:
                if batch * accumulation != effective:
                    raise ValueError(
                        "CPT effective_batch_size must equal "
                        "batch_size * gradient_accumulation_steps"
                    )
            elif batch is not None:
                if effective % batch:
                    raise ValueError(
                        "CPT effective_batch_size must be divisible by batch_size"
                    )
                accumulation = effective // batch
            elif accumulation is not None:
                if effective % accumulation:
                    raise ValueError(
                        "CPT effective_batch_size must be divisible by "
                        "gradient_accumulation_steps"
                    )
                batch = effective // accumulation
            else:
                # Keep the guided default memory-light: one record per device
                # step, accumulated to the requested effective batch.
                batch = 1
                accumulation = effective
        else:
            batch = batch or 1
            accumulation = accumulation or 8
            effective = batch * accumulation

        resolved["batch_size"] = int(batch)
        resolved["gradient_accumulation_steps"] = int(accumulation)
        resolved["effective_batch_size"] = int(effective)
        return resolved

    def _sanitize_public_training_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._normalize_public_training_aliases(payload)
        mode = str(payload.get("mode") or "").strip().lower()
        if mode not in TRAINING_MODALITIES:
            raise ValueError(f"Unsupported training mode: {mode}")

        allowed_fields = PUBLIC_TRAIN_ALLOWED_FIELDS[mode]
        unsupported_fields = sorted(
            key
            for key, value in payload.items()
            if key not in allowed_fields and self._has_public_value(value)
        )
        if unsupported_fields:
            raise ValueError(f"Unsupported fields for {mode}: {', '.join(unsupported_fields)}")

        sanitized: Dict[str, Any] = {"mode": mode}
        for field_name in PUBLIC_TRAIN_REQUIRED_TEXT_FIELDS[mode]:
            text = str(payload.get(field_name) or "").strip()
            if not text:
                raise ValueError(f"{field_name} is required")
            sanitized[field_name] = text

        optional_fields = allowed_fields - {"mode"} - set(PUBLIC_TRAIN_REQUIRED_TEXT_FIELDS[mode])
        for field_name in sorted(optional_fields):
            value = payload.get(field_name)
            if self._has_public_value(value):
                sanitized[field_name] = value
        return sanitized

    def _augment_mlx_preflight(
        self,
        *,
        payload: Dict[str, Any],
        preflight: TrainingLaunchPreflight,
    ) -> None:
        model = str(payload.get("model") or "")
        accelerator = str(payload.get("accelerator") or "").strip().lower()
        wants_mlx = accelerator == "mlx" or model.startswith("mlx-community/")
        if not wants_mlx:
            return

        readiness = self._mlx_readiness_snapshot()
        executable = bool(readiness.get("executable"))
        if accelerator != "mlx" and model.startswith("mlx-community/"):
            preflight.warnings.append(
                "MLX-format model selected; dashboard launch should use accelerator=mlx."
            )
            preflight.suggested_fixes.append(
                "Use the guided Start flow or launch with `halo-forge --accelerator mlx ...`."
            )
        if executable:
            return
        status = str(readiness.get("status") or "unavailable")
        first_error = ""
        errors = readiness.get("errors")
        if isinstance(errors, list) and errors:
            first_error = str(errors[0])
        preflight.warnings.append(
            f"MLX readiness is {status}; MLX training may not launch from this process."
        )
        if first_error:
            preflight.warnings.append(first_error)
        fixes = readiness.get("suggested_fixes")
        if isinstance(fixes, list):
            preflight.suggested_fixes.extend(str(item) for item in fixes if item)
        preflight.quality_outlook.setdefault("mlx_readiness", readiness)

    def _augment_output_dir_preflight(
        self,
        *,
        payload: Dict[str, Any],
        preflight: TrainingLaunchPreflight,
    ) -> None:
        """Make output path failures actionable in the public dashboard."""
        if not preflight.errors:
            return
        output_dir = str(
            payload.get("output_dir") or preflight.resolved_paths.get("output_dir") or ""
        ).strip()
        default_root = str(_default_run_root())

        def friendly(message: str) -> str:
            lower = message.lower()
            if "output_dir" in lower and (
                "not writable" in lower
                or "cannot be created" in lower
                or "current permissions" in lower
                or "parent is not writable" in lower
            ):
                return f"Halo Forge could not write to this folder: {output_dir or 'the selected folder'}"
            return message

        preflight.errors[:] = [friendly(str(message)) for message in preflight.errors]
        if any(
            message.startswith("Halo Forge could not write to this folder:")
            for message in preflight.errors
        ):
            preflight.suggested_fixes[:] = [
                str(fix)
                for fix in preflight.suggested_fixes
                if "output_dir" not in str(fix).lower() and "output parent" not in str(fix).lower()
            ]
            fix = f"Use the default run folder: {default_root}"
            if fix not in preflight.suggested_fixes:
                preflight.suggested_fixes.insert(0, fix)

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

    def search_runs(
        self,
        *,
        modalities: Optional[List[str]] = None,
        statuses: Optional[List[str]] = None,
        model_substring: Optional[str] = None,
        since_iso: Optional[str] = None,
        until_iso: Optional[str] = None,
        has_eval: Optional[bool] = None,
        weights_updated: Optional[bool] = None,
        sort_by: str = "timestamp",
        sort_dir: str = "desc",
        limit: Optional[int] = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """DB-backed run search (Track F-G commit 2).

        Lazily ensures the SQLite index is in sync with the filesystem,
        then queries it with the supplied filters. The existing
        ``list_runs`` surface keeps its filesystem-walk behavior so the
        run-list page is untouched; this endpoint is what the cohort /
        comparison / search-bar surfaces in the upcoming F-J / F-K
        items will target.

        Returns:
            ``{"items": [...], "total": N, "filters": {...},
              "facets": {"modalities": [...], "models": [...]}}``

            ``items`` is the paginated row list, ``total`` is the
            unpaginated match count, and ``facets`` is the distinct
            modality / model values present in the index — useful for
            the filter-chip UI without an extra round trip.
        """
        from halo_forge.run_db import RunFilter, get_database, sync_from_filesystem

        db = get_database()
        # Lazy sync. Cheap if the DB already mirrors the FS (incremental
        # by mtime); the first call after a fresh install pays the full
        # walk once.
        try:
            sync_from_filesystem(db)
        except Exception as exc:  # pragma: no cover - logged at runtime
            self._cached_backend_name = self._cached_backend_name  # touch attr to silence linters
            # Soft failure: serve what's already indexed rather than 5xx.
            # The sync is idempotent so the next call retries.
            import logging

            logging.getLogger(__name__).warning("run_db sync failed; serving cached index: %s", exc)

        filt = RunFilter(
            modalities=list(modalities) if modalities else None,
            statuses=list(statuses) if statuses else None,
            model_substring=model_substring,
            since_iso=since_iso,
            until_iso=until_iso,
            has_eval=has_eval,
            weights_updated=weights_updated,
            sort_by=sort_by,
            sort_dir=sort_dir,
            limit=limit,
            offset=offset,
        )

        records = db.list_runs(filt)
        total = db.count_runs(filt)

        items = [self._db_record_to_list_item(record) for record in records]
        return {
            "items": items,
            "total": total,
            "filters": {
                "modalities": filt.modalities,
                "statuses": filt.statuses,
                "model_substring": filt.model_substring,
                "since_iso": filt.since_iso,
                "until_iso": filt.until_iso,
                "has_eval": filt.has_eval,
                "weights_updated": filt.weights_updated,
                "sort_by": filt.sort_by,
                "sort_dir": filt.sort_dir,
                "limit": filt.limit,
                "offset": filt.offset,
            },
            "facets": {
                "modalities": db.distinct_modalities(),
                "modality_counts": db.modality_counts(),
                "models": db.distinct_models(),
            },
        }

    # ----- Human Feedback and Active Data Studio -------------------------

    @classmethod
    def _review_value(cls, value: Any) -> Any:
        if hasattr(value, "to_dict"):
            return cls._review_value(value.to_dict())
        if isinstance(value, tuple):
            return [cls._review_value(item) for item in value]
        if isinstance(value, list):
            return [cls._review_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): cls._review_value(item) for key, item in value.items()}
        return value

    @classmethod
    def _review_page(
        cls, values: Any, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        if isinstance(values, dict) and "items" in values:
            result = cls._review_value(values)
            result.setdefault("total", len(result["items"]))
            result.setdefault("limit", limit)
            result.setdefault("offset", offset)
            return result
        items = list(values or [])
        start = max(0, int(offset))
        count = max(1, min(1000, int(limit)))
        return {
            "items": [cls._review_value(value) for value in items[start : start + count]],
            "total": len(items),
            "limit": count,
            "offset": start,
        }

    def _review_sql_count(
        self, table: str, *, where: str = "", params: tuple[Any, ...] = ()
    ) -> int:
        allowed = {
            "annotation_schemas",
            "acquisition_batches",
            "review_queues",
            "review_items",
            "label_sets",
            "label_set_items",
        }
        if table not in allowed:
            raise ValueError(f"unsupported review count table: {table}")
        row = self._dataset_database()._conn.execute(
            f"SELECT COUNT(*) AS value FROM {table} {where}", params
        ).fetchone()
        return int(row["value"] if row is not None else 0)

    def _complete_review_activity(
        self,
        *,
        kind: str,
        domain_kind: str,
        domain_id: str,
        result: Mapping[str, Any],
        name: Optional[str] = None,
    ) -> str:
        """Record a short, already-finished review mutation in Activity.

        Heavy review operations use the supervised worker. Small atomic
        catalog publications still receive the same durable work identity so
        API/CLI/dashboard responses and Activity share one lifecycle contract.
        """

        work_item_id = f"review-{uuid.uuid4().hex}"
        scheduler = self._scheduler()
        scheduler.enqueue(
            kind=kind,
            launch_spec={
                "operation": kind,
                "name": name or kind.replace("_", " "),
                "review_root": str(self.review_storage_root),
            },
            resource_class="none",
            domain_kind=domain_kind,
            domain_id=domain_id,
            work_item_id=work_item_id,
        )
        claimed = scheduler.claim(work_item_id=work_item_id)
        if claimed is None:
            raise RuntimeError("could not claim completed Review Studio activity")
        completed = scheduler.complete(claimed, result=dict(result))
        if completed is None:
            raise RuntimeError("could not complete Review Studio activity")
        return completed.id

    def get_review_capabilities(self) -> Dict[str, Any]:
        value = self._review_value(self._review_engine().capabilities())
        return value if isinstance(value, dict) else {"items": value}

    def list_annotation_schemas(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        try:
            values = self._review_engine().list_schemas(limit=limit, offset=offset)
        except TypeError:
            return self._review_page(
                self._review_engine().list_schemas(), limit=limit, offset=offset
            )
        return {
            "items": [self._review_value(value) for value in values],
            "total": self._review_sql_count(
                "annotation_schemas", where="WHERE archived = 0"
            ),
            "limit": limit,
            "offset": offset,
        }

    def validate_annotation_schema(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from halo_forge.review_lab.registry import validate_schema_definition

        modality = str(payload.get("modality") or "")
        task_type = str(payload.get("task_type") or "")
        definition = validate_schema_definition(
            modality,
            task_type,
            dict(payload.get("definition") or {}),
        )
        return {
            "valid": True,
            "schema": {
                "name": str(payload.get("name") or "Validation only"),
                "modality": definition["modality"],
                "task_type": definition["task_type"],
                "definition": definition,
            },
            "errors": [],
        }

    def create_annotation_schema(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        created = self._review_engine().create_schema(
            name=str(payload.get("name") or ""),
            modality=str(payload.get("modality") or ""),
            task_type=str(payload.get("task_type") or ""),
            definition=dict(payload.get("definition") or {}),
            description=self._optional_str(payload.get("description")),
            schema_id=self._optional_str(payload.get("schema_id") or payload.get("id")),
        )
        if isinstance(created, tuple) and len(created) == 2:
            return {
                "schema": self._review_value(created[0]),
                "revision": self._review_value(created[1]),
            }
        return self._review_value(created)

    def get_annotation_schema(self, schema_id: str) -> Optional[Dict[str, Any]]:
        try:
            value = self._review_engine().get_schema(schema_id)
        except KeyError:
            return None
        if value is None:
            return None
        result = self._review_value(value)
        revisions = self.list_annotation_schema_revisions(schema_id, limit=1000)
        if isinstance(result, dict):
            result["revisions"] = revisions["items"]
            result["revision_count"] = revisions["total"]
        return result

    def revise_annotation_schema(
        self, schema_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "definition": dict(payload.get("definition") or {}),
        }
        if payload.get("modality") is not None:
            kwargs["modality"] = str(payload["modality"])
        if payload.get("task_type") is not None:
            kwargs["task_type"] = str(payload["task_type"])
        return self._review_value(self._review_engine().revise_schema(schema_id, **kwargs))

    def list_annotation_schema_revisions(
        self, schema_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        if self._review_engine().get_schema(schema_id) is None:
            raise KeyError(schema_id)
        return self._review_page(
            self._review_engine().list_schema_revisions(schema_id),
            limit=limit,
            offset=offset,
        )

    def get_annotation_schema_revision(
        self, revision_id: str
    ) -> Optional[Dict[str, Any]]:
        try:
            value = self._review_engine().get_schema_revision(revision_id)
        except KeyError:
            return None
        return self._review_value(value) if value is not None else None

    @staticmethod
    def _iter_review_jsonl(path_value: str) -> Iterator[Dict[str, Any]]:
        path = Path(path_value).expanduser().resolve()
        files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
        if not files or any(not value.is_file() for value in files):
            raise ValueError(f"imported JSONL source is missing: {path}")
        for source_path in files:
            with source_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"invalid JSONL at {source_path}:{line_number}: {exc.msg}"
                        ) from exc
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"JSONL record at {source_path}:{line_number} must be an object"
                        )
                    yield value

    @staticmethod
    def _review_comparison_outcome(before: Any, after: Any) -> str:
        if before is None:
            return "missing_base"
        if after is None:
            return "missing_candidate"
        if before.passed is True and after.passed is False:
            return "regression"
        if before.passed is False and after.passed is True:
            return "improvement"
        if before.passed is False and after.passed is False:
            return "unchanged_failure"
        if before.passed is True and after.passed is True:
            return "unchanged_pass"
        return "unchanged_failure" if before.error or after.error else "unchanged_pass"

    @staticmethod
    def _review_source_envelope(
        value: Mapping[str, Any],
        *,
        source: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Retain explicit evidence envelopes without inferring missing evidence."""

        raw = dict(value)
        if isinstance(raw.get("record"), Mapping):
            existing_source = (
                dict(raw.get("source") or {})
                if isinstance(raw.get("source"), Mapping)
                else {}
            )
            return {
                **raw,
                "record": dict(raw["record"]),
                "evidence": (
                    dict(raw.get("evidence") or {})
                    if isinstance(raw.get("evidence"), Mapping)
                    else {}
                ),
                "source": {**existing_source, **dict(source)},
            }
        evidence_fields = {
            "score",
            "score_direction",
            "score_metric",
            "margin",
            "margin_metric",
            "candidate_score",
            "base_score",
            "score_delta",
            "passed",
            "valid",
            "mineable",
            "verifier_disagreement",
            "category",
            "task",
            "failure_reason",
            "embedding",
            "embedding_revision",
            "embedding_model_revision",
            "embedding_provenance",
        }
        return {
            "record_id": raw.get("record_id") or raw.get("id"),
            "record": raw,
            "evidence": {
                key: raw[key] for key in evidence_fields if key in raw
            },
            "source": dict(source),
        }

    @staticmethod
    def _review_record_asset_paths(
        value: Mapping[str, Any], asset_index: Mapping[str, str]
    ) -> Dict[str, str]:
        """Return only media paths referenced by one canonical review record."""

        record = value.get("record") if isinstance(value.get("record"), Mapping) else value
        references: set[str] = set()
        for field in ("image", "image_path", "audio", "audio_path"):
            raw = record.get(field) if isinstance(record, Mapping) else None
            if isinstance(raw, Mapping):
                raw = raw.get("path") or raw.get("filename") or raw.get("ref")
            if isinstance(raw, (str, os.PathLike)) and str(raw):
                references.add(os.fspath(raw))
        return {
            reference: str(asset_index[reference])
            for reference in sorted(references)
            if reference in asset_index
        }

    @staticmethod
    def _review_spool_envelope(
        raw: Mapping[str, Any], *, source_hash_field: str, source_hash: str
    ) -> Dict[str, Any]:
        """Canonicalize raw rows before adding restart-only spool metadata.

        Small synchronous acquisition historically accepts both canonical
        envelopes and plain record objects. Durable acquisition must preserve
        those semantics while ensuring its recovery hash never becomes model
        training content.
        """

        source = dict(raw)
        source.pop(source_hash_field, None)
        if isinstance(source.get("record"), Mapping):
            envelope = source
            record = dict(envelope["record"])
            record.pop(source_hash_field, None)
            envelope["record"] = record
        else:
            record = dict(source)
            record.pop(source_hash_field, None)
            envelope = {"record": record}
            for key in (
                "record_id",
                "source_kind",
                "source_ref",
                "source_record_id",
                "suite_item_id",
            ):
                if source.get(key) is not None:
                    envelope[key] = source[key]
        envelope[source_hash_field] = source_hash
        return envelope

    @staticmethod
    def _review_suite_item_sources(revision: Any) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for item in list(getattr(revision, "items", None) or []):
            if not isinstance(item, Mapping):
                continue
            identifier = str(item.get("id") or item.get("suite_item_id") or "").strip()
            if not identifier:
                continue
            result[identifier] = {
                key: item.get(key)
                for key in ("split", "dataset_version_id")
                if item.get(key) is not None
            }
        return result

    def _iter_evaluation_comparison_acquisition(
        self, *, base_id: str, candidate_id: str
    ) -> Iterator[Dict[str, Any]]:
        db = self._dataset_database()
        base = db.get_evaluation(base_id)
        candidate = db.get_evaluation(candidate_id)
        if base is None:
            raise KeyError(base_id)
        if candidate is None:
            raise KeyError(candidate_id)
        if base.status != "completed" or candidate.status != "completed":
            raise ValueError("evaluation comparison acquisition requires completed evaluations")
        if base.suite_revision_id != candidate.suite_revision_id:
            raise ValueError("evaluation comparison acquisition requires one suite revision")
        revision = db.get_benchmark_suite_revision(candidate.suite_revision_id)
        suite = db.get_benchmark_suite(revision.suite_id) if revision is not None else None
        purpose = suite.purpose if suite is not None else None
        suite_sources = self._review_suite_item_sources(revision)
        total = db.count_evaluation_sample_pairs(base_id, candidate_id)
        offset = 0
        while offset < total:
            pairs = db.list_evaluation_sample_pairs(
                base_id, candidate_id, limit=1000, offset=offset
            )
            if not pairs:
                break
            for pair in pairs:
                before = pair.get("base")
                after = pair.get("candidate")
                sample = after or before
                if sample is None:
                    continue
                before_value = before.to_dict() if before is not None else {}
                after_value = after.to_dict() if after is not None else {}
                direction = str(
                    after_value.get("score_direction")
                    or before_value.get("score_direction")
                    or (revision.direction if revision is not None else "maximize")
                )
                score_delta = None
                if before is not None and after is not None:
                    if before.score is not None and after.score is not None:
                        raw_delta = float(after.score) - float(before.score)
                        score_delta = raw_delta if direction == "maximize" else -raw_delta
                valid = bool(before_value.get("valid", False)) and bool(
                    after_value.get("valid", False)
                )
                mineable = bool(before_value.get("mineable", False)) and bool(
                    after_value.get("mineable", False)
                )
                yield {
                    "record_id": str(pair.get("logical_record_id") or sample.suite_item_id),
                    "record": {
                        "input": after_value.get("input", before_value.get("input")),
                        "expected": after_value.get("expected", before_value.get("expected")),
                        "output": after_value.get("output"),
                        "base_output": before_value.get("output"),
                    },
                    "evidence": {
                        **after_value,
                        "valid": valid,
                        "mineable": mineable,
                        "suite_purpose": purpose,
                        "outcome": self._review_comparison_outcome(before, after),
                        "base_passed": before_value.get("passed"),
                        "candidate_passed": after_value.get("passed"),
                        "base_score": before_value.get("score"),
                        "candidate_score": after_value.get("score"),
                        "score": after_value.get("score"),
                        "score_direction": direction,
                        "score_delta": score_delta,
                        "margin": abs(score_delta) if score_delta is not None else None,
                        "occurrence_index": int(pair.get("occurrence") or 0),
                    },
                    "source": {
                        "kind": "evaluation_comparison",
                        "ref": candidate_id,
                        "base_ref": base_id,
                        "candidate_ref": candidate_id,
                        "purpose": purpose,
                        "suite_revision_id": candidate.suite_revision_id,
                        "suite_item_id": sample.suite_item_id,
                        **suite_sources.get(str(sample.suite_item_id), {}),
                    },
                }
            offset += len(pairs)

    def _iter_verifier_calibration_acquisition(
        self,
        *,
        calibration_id: str,
        selector_value: Any,
        selector_options: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream reviewed failure candidates from development calibration evidence.

        Only the calibration partition is joined.  Source rows are read in
        bounded chunks and observations are fetched with one indexed query per
        chunk, which keeps 100k-record calibrations out of application memory.
        """

        from halo_forge.verifier_lab.failure_mining import (
            normalize_failure_selector,
            select_calibration_failure,
            validate_failure_selector,
        )

        engine = self._verifier_engine()
        calibration = engine.get_calibration(calibration_id)
        if calibration is None:
            raise KeyError(calibration_id)
        if calibration.status != "completed":
            raise ValueError(
                "verifier calibration acquisition requires a completed calibration"
            )
        purpose = str(calibration.source_purpose or "unspecified").strip().lower()
        if purpose not in {"development", "unspecified"}:
            raise ValueError(
                f"{purpose} verifier calibration evidence cannot guide review acquisition"
            )
        partition = dict(calibration.partition or {})
        leakage = partition.get("leakage")
        if isinstance(leakage, Mapping) and any(leakage.values()):
            raise ValueError(
                "verifier calibration has leaked calibration/confirmation identities"
            )
        integrity = engine.verify_calibration(calibration.id)
        if not integrity.get("valid"):
            raise ValueError(
                "verifier calibration evidence failed checksum verification"
            )

        revision = engine.store.get_profile_revision(calibration.verifier_revision_id)
        selector, embedded_options = normalize_failure_selector(selector_value)
        options = {**dict(selector_options or {}), **embedded_options}
        validate_failure_selector(
            selector,
            task_type=revision.task_type,
            verifier_family=revision.family,
        )
        source_meta, source_records = engine._source(  # noqa: SLF001 - same service boundary
            calibration.source_kind,
            calibration.source_revision_id,
            revision.task_type,
        )
        source_purpose = (
            str(source_meta.get("purpose") or "unspecified").strip().lower()
        )
        if source_purpose not in {"development", "unspecified"}:
            raise ValueError(
                f"{source_purpose} verifier source evidence cannot guide review acquisition"
            )
        if str(source_meta.get("hash") or "") != calibration.source_hash:
            raise ValueError(
                "verifier calibration source identity changed after calibration"
            )
        manifest = source_meta.get("manifest")
        if isinstance(manifest, Mapping) and bool(manifest.get("protected_lineage")):
            raise ValueError("protected-lineage verifier evidence cannot guide review acquisition")

        selector_identity = {
            "kind": selector,
            "version": 1,
            "options": options,
        }
        reward_contract = revision.reward_contract.to_dict()
        protected_purposes = {
            "operational",
            "holdout",
            "final_holdout",
            "test",
            "canary",
        }
        protected_splits = {"test", "canary"}

        def emit(chunk: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
            record_ids = [str(value.get("record_id") or "") for value in chunk]
            if any(not value for value in record_ids):
                raise ValueError("calibration source record is missing stable record identity")
            grouped = engine.store.list_samples_for_record_ids(
                calibration.id,
                record_ids,
                partition="calibration",
            )
            for source_record, record_id in zip(chunk, record_ids):
                record_purpose = str(
                    source_record.get("purpose")
                    or source_record.get("suite_purpose")
                    or source_purpose
                ).strip().lower()
                record_split = str(source_record.get("split") or "").strip().lower()
                if (
                    record_purpose in protected_purposes
                    or record_split in protected_splits
                    or bool(source_record.get("protected_lineage"))
                ):
                    raise ValueError(
                        f"protected verifier evidence cannot guide review acquisition: {record_id}"
                    )
                if bool(source_record.get("reward_model_training")):
                    raise ValueError(
                        "reward-model training records cannot guide verifier failure acquisition"
                    )
                sample_values = [
                    value.to_dict() for value in grouped.get(record_id, [])
                ]
                evidence = select_calibration_failure(
                    selector=selector,
                    options=options,
                    samples=sample_values,
                    task_type=revision.task_type,
                    verifier_family=revision.family,
                    reward_contract=reward_contract,
                )
                if evidence is None:
                    continue
                suite_revision_id = (
                    calibration.source_revision_id
                    if calibration.source_kind == "benchmark_suite"
                    else source_record.get("suite_revision_id")
                )
                source = {
                    "kind": "verifier_calibration",
                    "ref": calibration.id,
                    "purpose": "development",
                    "partition": "calibration",
                    "eligible": True,
                    "calibration_id": calibration.id,
                    "calibration_manifest_hash": calibration.manifest_hash,
                    "verifier_profile_revision_id": revision.id,
                    "verifier_profile_revision_hash": revision.content_hash,
                    "protocol_revision_id": calibration.protocol_revision_id,
                    "protocol_hash": calibration.protocol_hash,
                    "qualification_profile_revision_id": (
                        calibration.qualification_profile_revision_id
                    ),
                    "source_kind": calibration.source_kind,
                    "source_revision_id": calibration.source_revision_id,
                    "source_hash": calibration.source_hash,
                    "selector": selector_identity,
                }
                if suite_revision_id:
                    source.update(
                        suite_revision_id=str(suite_revision_id),
                        suite_item_id=str(
                            source_record.get("suite_item_id") or record_id
                        ),
                    )
                yield {
                    "record_id": record_id,
                    "record": dict(source_record),
                    "evidence": {
                        **evidence,
                        "valid": True,
                        "mineable": True,
                        "suite_purpose": "development",
                        "task_type": revision.task_type,
                        "verifier_family": revision.family,
                        "reward_contract": reward_contract,
                        "calibration_id": calibration.id,
                        "calibration_partition": "calibration",
                    },
                    "source": source,
                }

        chunk: List[Dict[str, Any]] = []
        for raw in source_records:
            chunk.append(dict(raw))
            if len(chunk) >= 256:
                yield from emit(chunk)
                chunk = []
        if chunk:
            yield from emit(chunk)

    @staticmethod
    def _reward_audit_protected_evidence_reason(*values: Any) -> Optional[str]:
        """Find protected purpose/split markers in trace or record lineage."""

        protected_purposes = {
            "operational",
            "holdout",
            "final_holdout",
            "test",
            "canary",
        }
        protected_splits = {"test", "canary"}

        def inspect(value: Any) -> Optional[str]:
            if isinstance(value, Mapping):
                if value.get("protected_lineage") is True:
                    return "protected_lineage"
                purpose = str(
                    value.get("purpose") or value.get("suite_purpose") or ""
                ).strip().lower()
                if purpose in protected_purposes:
                    return f"protected_suite_purpose:{purpose}"
                split = str(value.get("split") or "").strip().lower()
                if split in protected_splits:
                    return f"protected_split:{split}"
                for child in value.values():
                    reason = inspect(child)
                    if reason:
                        return reason
            elif isinstance(value, (list, tuple)):
                for child in value:
                    reason = inspect(child)
                    if reason:
                        return reason
            return None

        for value in values:
            reason = inspect(value)
            if reason:
                return reason
        return None

    def _iter_reward_integrity_audit_acquisition(
        self, *, audit_id: str
    ) -> Iterator[Dict[str, Any]]:
        """Stream exact same-output audit evidence into a reviewed proposal.

        This is intentionally a source adapter only: it creates an immutable
        acquisition proposal and never publishes labels, builds data, resolves a
        training pause, or starts a fork. Protected lineage is rejected before
        any acquisition strategy can inspect it.
        """

        engine = self._reward_integrity_engine()
        try:
            audit = engine.get_audit(audit_id)
        except KeyError as exc:
            raise KeyError(audit_id) from exc
        if audit.status != "completed":
            raise ValueError(
                "reward-integrity acquisition requires a completed audit"
            )
        integrity = engine.verify_audit_bundle(audit.id)
        if not bool(integrity.get("valid")):
            raise ValueError("reward-integrity audit evidence failed checksum verification")
        shard = engine.get_signal_shard(audit.signal_shard_id)
        protected = self._reward_audit_protected_evidence_reason(
            shard.dataset_identity
        )
        if protected:
            raise ValueError(
                f"protected reward-integrity evidence cannot guide review acquisition: {protected}"
            )

        offset = 0
        while True:
            page = engine.list_audit_samples(
                audit.id, limit=1000, offset=offset, include_observations=True
            )
            values = list(page.items)
            if not values:
                break
            for raw in values:
                sample = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
                lineage = dict(sample.get("lineage") or {})
                protected = self._reward_audit_protected_evidence_reason(lineage)
                if protected:
                    raise ValueError(
                        "protected reward-integrity evidence cannot guide review "
                        f"acquisition: {sample.get('record_id')} ({protected})"
                    )
                optimizer = dict(sample.get("optimizer_observation") or {})
                sentinel = dict(sample.get("primary_sentinel_observation") or {})
                optimizer_passed = optimizer.get("passed")
                sentinel_passed = sentinel.get("passed")
                optimizer_reward = optimizer.get("normalized_reward")
                sentinel_reward = sentinel.get("normalized_reward")
                disagreement = (
                    optimizer_passed is not None
                    and sentinel_passed is not None
                    and bool(optimizer_passed) != bool(sentinel_passed)
                )
                if optimizer.get("error") or sentinel.get("error"):
                    outcome = "error"
                elif optimizer_passed is True and sentinel_passed is False:
                    outcome = "optimizer_only_accept"
                elif optimizer_passed is False and sentinel_passed is True:
                    outcome = "sentinel_only_accept"
                elif optimizer_passed is not None and sentinel_passed is not None:
                    outcome = "agreement"
                else:
                    outcome = "unclassified"
                score_gap = (
                    abs(float(optimizer_reward) - float(sentinel_reward))
                    if isinstance(optimizer_reward, (int, float))
                    and isinstance(sentinel_reward, (int, float))
                    else None
                )
                input_value = sample.get("input")
                record = (
                    dict(input_value)
                    if isinstance(input_value, Mapping)
                    else {"input": input_value}
                )
                record.update(
                    {
                        "output": sample.get("output"),
                        "expected": sample.get("expected"),
                        "media": list(sample.get("media") or []),
                    }
                )
                purpose = str(
                    lineage.get("purpose")
                    or lineage.get("suite_purpose")
                    or "unspecified"
                ).strip().lower()
                split = str(lineage.get("split") or "").strip().lower()
                yield {
                    "record_id": str(sample.get("record_id") or sample["snapshot_id"]),
                    "record": record,
                    "evidence": {
                        "valid": True,
                        "mineable": True,
                        "passed": sentinel_passed,
                        "score": sentinel_reward,
                        "score_direction": "maximize",
                        "metric_name": "sentinel_normalized_reward",
                        "margin": score_gap,
                        "verifier_disagreement": disagreement,
                        "outcome": outcome,
                        "optimizer_passed": optimizer_passed,
                        "sentinel_passed": sentinel_passed,
                        "optimizer_score": optimizer_reward,
                        "sentinel_score": sentinel_reward,
                        "optimizer_observation": optimizer,
                        "sentinel_observation": sentinel,
                        "diagnostic_observations": list(
                            sample.get("diagnostic_observations") or []
                        ),
                        "capture_stratum": sample.get("selection_class"),
                        "suite_purpose": purpose,
                    },
                    "source": {
                        "kind": "reward_integrity_audit",
                        "ref": audit.id,
                        "purpose": purpose,
                        **({"split": split} if split else {}),
                        "eligible": True,
                        "run_id": audit.run_id,
                        "audit_id": audit.id,
                        "training_signal_shard_id": shard.id,
                        "trace_hash": shard.trace_hash,
                        "reward_system_revision_id": audit.reward_system_revision_id,
                        "protocol_revision_id": audit.protocol_revision_id,
                        "integrity_profile_revision_id": (
                            audit.integrity_profile_revision_id
                        ),
                        "snapshot_id": sample.get("snapshot_id"),
                    },
                }
            offset += len(values)
            if offset >= page.total:
                break

    def _review_acquisition_records(
        self, payload: Dict[str, Any]
    ) -> Iterable[Dict[str, Any]]:
        projection = (
            dict((payload.get("metadata") or {}).get("projection") or {})
            if isinstance(payload.get("metadata"), Mapping)
            else {}
        )
        projection_fields = projection.get("fields")
        if projection_fields is not None and not isinstance(projection_fields, Mapping):
            raise ValueError("acquisition field projection requires a fields object")

        def apply_projection(values: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
            if not projection_fields:
                yield from values
                return
            from halo_forge.data_lab.models import get_field

            for value in values:
                envelope = dict(value)
                source_record = envelope.get("record")
                if not isinstance(source_record, Mapping):
                    raise ValueError("acquisition projection requires object records")
                mapped: Dict[str, Any] = {}
                for target, source_path in dict(projection_fields).items():
                    resolved = get_field(source_record, str(source_path))
                    if resolved is not None:
                        mapped[str(target)] = resolved
                if isinstance(source_record.get("metadata"), Mapping):
                    mapped.setdefault("metadata", dict(source_record["metadata"]))
                envelope["record"] = mapped
                yield envelope

        supplied = payload.get("records")
        if supplied is not None:
            if not isinstance(supplied, list):
                raise ValueError("records must be an array")
            return apply_projection(dict(value) for value in supplied)
        sources = payload.get("sources") or (
            [payload["source"]] if isinstance(payload.get("source"), dict) else []
        )
        if not isinstance(sources, list) or not sources:
            raise ValueError("records or at least one acquisition source is required")

        def generate() -> Iterator[Dict[str, Any]]:
            for source in sources:
                if not isinstance(source, dict):
                    raise ValueError("each acquisition source must be an object")
                kind = str(source.get("kind") or "").strip().lower().replace("-", "_")
                reference = str(source.get("ref") or source.get("id") or "").strip()
                if kind == "evaluation":
                    db = self._dataset_database()
                    evaluation = db.get_evaluation(reference)
                    if evaluation is None:
                        raise KeyError(reference)
                    if evaluation.status != "completed":
                        raise ValueError(
                            "evaluation acquisition requires a completed evaluation"
                        )
                    revision = db.get_benchmark_suite_revision(evaluation.suite_revision_id)
                    suite = db.get_benchmark_suite(revision.suite_id) if revision else None
                    purpose = suite.purpose if suite is not None else None
                    suite_sources = self._review_suite_item_sources(revision)
                    offset = 0
                    while True:
                        samples = db.list_evaluation_samples(reference, limit=1000, offset=offset)
                        if not samples:
                            break
                        for sample in samples:
                            value = sample.to_dict()
                            yield {
                                "record_id": value.get("record_id")
                                or value.get("suite_item_id"),
                                "record": {
                                    "input": value.get("input"),
                                    "expected": value.get("expected"),
                                    "output": value.get("output"),
                                },
                                "evidence": {**value, "suite_purpose": purpose},
                                "source": {
                                    "kind": "evaluation",
                                    "ref": reference,
                                    "purpose": purpose,
                                    "suite_revision_id": evaluation.suite_revision_id,
                                    "suite_item_id": sample.suite_item_id,
                                    **suite_sources.get(str(sample.suite_item_id), {}),
                                },
                            }
                        offset += len(samples)
                elif kind == "evaluation_comparison":
                    base_id = str(source.get("base_id") or "").strip()
                    candidate_id = str(source.get("candidate_id") or reference).strip()
                    if not base_id or not candidate_id:
                        raise ValueError("evaluation comparison requires base_id and candidate_id")
                    yield from self._iter_evaluation_comparison_acquisition(
                        base_id=base_id, candidate_id=candidate_id
                    )
                elif kind == "verifier_calibration":
                    nested_options = source.get("options")
                    if nested_options is not None and not isinstance(
                        nested_options, Mapping
                    ):
                        raise ValueError(
                            "verifier calibration source options must be an object"
                        )
                    options = dict(nested_options or {})
                    selector = (
                        source.get("selector")
                        or source.get("failure_selector")
                        or options.pop("selector", None)
                    )
                    if selector is None:
                        raise ValueError(
                            "verifier calibration acquisition requires a failure selector"
                        )
                    yield from self._iter_verifier_calibration_acquisition(
                        calibration_id=reference,
                        selector_value=selector,
                        selector_options=options,
                    )
                elif kind == "reward_integrity_audit":
                    yield from self._iter_reward_integrity_audit_acquisition(
                        audit_id=reference
                    )
                elif kind == "dataset_version":
                    split = str(source.get("split") or "train")
                    version = self._dataset_database().get_dataset_version(reference)
                    if version is None:
                        raise KeyError(reference)
                    asset_paths: Dict[str, str] = {}
                    try:
                        manifest = json.loads(
                            (Path(version.storage_path) / "manifest.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        for asset in manifest.get("asset_fingerprints") or []:
                            if isinstance(asset, Mapping) and asset.get("reference") and asset.get(
                                "resolved_path"
                            ):
                                asset_paths[str(asset["reference"])] = str(asset["resolved_path"])
                    except (OSError, json.JSONDecodeError):
                        asset_paths = {}
                    offset = 0
                    while True:
                        page = self.preview_dataset_version(
                            reference, split=split, offset=offset, limit=1000
                        )
                        items = list(page.get("items") or [])
                        for value in items:
                            referenced_asset_paths = self._review_record_asset_paths(
                                value, asset_paths
                            )
                            yield self._review_source_envelope(
                                value,
                                source={
                                    "kind": "dataset_version",
                                    "ref": reference,
                                    "split": split,
                                    "asset_root": version.storage_path,
                                    **(
                                        {"asset_paths": referenced_asset_paths}
                                        if referenced_asset_paths
                                        else {}
                                    ),
                                },
                            )
                        offset += len(items)
                        if offset >= int(page.get("total") or 0) or not items:
                            break
                elif kind == "playground_session":
                    session = self.get_playground_session(reference)
                    if session is None:
                        raise KeyError(reference)
                    pending: Optional[Dict[str, Any]] = None
                    for message in session.get("messages") or []:
                        if message.get("role") == "user":
                            pending = message
                        elif message.get("role") == "assistant" and pending is not None:
                            yield {
                                "record": {
                                    "messages": [
                                        {"role": "user", "content": pending.get("content")},
                                        {"role": "assistant", "content": message.get("content")},
                                    ]
                                },
                                "source": {"kind": "playground_session", "ref": reference},
                            }
                            pending = None
                elif kind == "run_samples":
                    for value in self._dataset_engine().failure_resolver(reference):
                        yield self._review_source_envelope(
                            value,
                            source={
                                "kind": "run_samples",
                                "ref": reference,
                                **(
                                    {"split": value.get("split")}
                                    if value.get("split") is not None
                                    else {}
                                ),
                            },
                        )
                elif kind in {"import", "jsonl"}:
                    for value in self._iter_review_jsonl(reference):
                        yield self._review_source_envelope(
                            value,
                            source={
                                "kind": "imported",
                                "ref": reference,
                                "asset_root": str(Path(reference).expanduser().resolve().parent),
                            },
                        )
                else:
                    raise ValueError(f"unsupported acquisition source kind: {kind}")

        return apply_projection(generate())

    @staticmethod
    def _review_diversity_revision(payload: Mapping[str, Any]) -> Optional[str]:
        strategies = payload.get("strategies") or ["explicit"]
        if isinstance(strategies, (str, Mapping)):
            strategies = [strategies]
        for raw in strategies:
            if isinstance(raw, str):
                kind, options = raw, {}
            elif isinstance(raw, Mapping):
                kind = str(raw.get("kind") or raw.get("strategy") or "")
                options = dict(raw.get("options") or {})
                for key in ("embedding_revision", "embedding_model_revision"):
                    if raw.get(key) is not None:
                        options.setdefault(key, raw[key])
            else:
                continue
            if kind.strip().lower().replace("-", "_") != "diversity":
                continue
            revision = str(
                options.get("embedding_revision")
                or options.get("embedding_model_revision")
                or payload.get("embedding_revision")
                or ""
            ).strip()
            if not revision:
                raise ValueError("diversity acquisition requires a pinned embedding_revision")
            return revision
        return None

    def _embed_review_acquisition_chunk(
        self, records: List[Dict[str, Any]], *, embedding_revision: Optional[str]
    ) -> List[Dict[str, Any]]:
        if not embedding_revision:
            return records
        missing_indexes = [
            index
            for index, value in enumerate(records)
            if not isinstance((value.get("evidence") or {}).get("embedding"), list)
        ]
        if not missing_indexes:
            return records
        if self._review_embedding_engine is None:
            from halo_forge.review_lab.embeddings import PinnedEmbeddingEngine

            self._review_embedding_engine = PinnedEmbeddingEngine()
        missing = [records[index] for index in missing_indexes]
        embedded = self._review_embedding_engine.embed_envelopes(
            missing,
            embedding_revision=embedding_revision,
        )
        if len(embedded) != len(missing_indexes):
            raise ValueError("embedding engine returned an incomplete acquisition batch")
        output = list(records)
        for index, value in zip(missing_indexes, embedded):
            output[index] = dict(value)
        return output

    def create_acquisition_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        metadata = dict(payload.get("metadata") or {})
        sources = payload.get("sources") or (
            [payload["source"]] if isinstance(payload.get("source"), dict) else []
        )
        if sources:
            metadata["ordered_sources"] = [dict(value) for value in sources]
        if payload.get("filters") is not None:
            metadata["filters"] = payload.get("filters")
        supplied = payload.get("records")
        diversity_revision = self._review_diversity_revision(payload)
        should_queue = bool(sources) or (
            isinstance(supplied, list) and len(supplied) > 1000
        ) or bool(diversity_revision)
        if should_queue:
            batch_id = str(payload.get("batch_id") or f"acq-{uuid.uuid4().hex}")
            work_item_id = f"review-acquisition-{uuid.uuid4().hex}"
            queued_payload = self._sanitize_review_work_payload(payload)
            queued_payload["metadata"] = metadata
            work = self._scheduler().enqueue(
                kind="review_acquisition",
                launch_spec={
                    "handler": "review_lab.resolve_acquisition",
                    "action": "build_review_batch",
                    "name": str(payload.get("name") or "Create review proposal"),
                    "batch_id": batch_id,
                    "payload": queued_payload,
                    "review_root": str(self.review_storage_root),
                    "dataset_root": str(self.dataset_storage_root),
                    "evaluation_root": str(self.evaluation_storage_root),
                    "artifact_root": str(self.artifact_storage_root),
                },
                resource_class="accelerator" if diversity_revision else "cpu",
                resource_requirements={
                    "output_path": str(self.review_storage_root),
                    **(
                        {"embedding_revision": diversity_revision}
                        if diversity_revision
                        else {}
                    ),
                },
                domain_kind="acquisition_batch",
                domain_id=batch_id,
                log_path=str(
                    self.review_storage_root
                    / ".work"
                    / work_item_id
                    / "acquisition.log"
                ),
                max_retries=2,
                work_item_id=work_item_id,
            )
            try:
                batch = self._review_engine().queue_acquisition(
                    batch_id=batch_id,
                    work_item_id=work.id,
                    request={
                        "sources": queued_payload.get("sources")
                        or (
                            [queued_payload["source"]]
                            if isinstance(queued_payload.get("source"), dict)
                            else []
                        ),
                        "strategies": queued_payload.get("strategies") or ["explicit"],
                        "filters": queued_payload.get("filters") or [],
                        "seed": int(queued_payload.get("seed") or 0),
                    },
                    name=self._optional_str(queued_payload.get("name")),
                    seed=int(queued_payload.get("seed") or 0),
                    metadata=metadata,
                    total_records=(len(supplied) if isinstance(supplied, list) else None),
                )
            except Exception:
                self._scheduler().cancel(work.id)
                raise
            return {**self._review_value(batch), "work_item_id": work.id}
        batch = self._review_engine().create_acquisition(
            self._review_acquisition_records(payload),
            strategies=payload.get("strategies"),
            seed=int(payload.get("seed") or 0),
            filters=payload.get("filters"),
            name=self._optional_str(payload.get("name")),
            metadata=metadata,
        )
        value = self._review_value(batch)
        work_item_id = self._complete_review_activity(
            kind="review_acquisition",
            domain_kind="acquisition_batch",
            domain_id=str(value["id"]),
            result=value,
            name=str(value.get("name") or "Review acquisition proposal"),
        )
        db = self._dataset_database()
        with db._lock:
            db._conn.execute(
                "UPDATE acquisition_batches SET work_item_id=? WHERE id=?",
                (work_item_id, str(value["id"])),
            )
            db._conn.commit()
        return {**value, "work_item_id": work_item_id}

    @staticmethod
    def _sanitize_review_work_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Remove provider credentials before durable review work is persisted."""

        blocked = {"api_key", "token", "authorization", "password", "secret"}

        def clean(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    str(key): clean(item)
                    for key, item in value.items()
                    if str(key).strip().lower() not in blocked
                }
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, tuple):
                return [clean(item) for item in value]
            return value

        return clean(dict(payload))

    def resolve_acquisition_batch(
        self,
        batch_id: str,
        payload: Dict[str, Any],
        *,
        work_item_id: str,
    ) -> Dict[str, Any]:
        """Resolve source pages into a restart-safe spool, then publish once."""

        from halo_forge.review_lab._canonical import bytes_hash, canonical_json
        from halo_forge.review_lab.acquisition_storage import (
            INGESTION_SOURCE_HASH_FIELD,
            AcquisitionRecordSpool,
        )

        review = self._review_engine()
        current = review.get_acquisition(batch_id)
        if current is None:
            raise KeyError(batch_id)
        if current.status == "ready":
            return self._review_value(current)
        if current.status == "cancelled":
            return self._review_value(current)

        class AcquisitionCancellationRequested(RuntimeError):
            pass

        def cancellation_requested() -> bool:
            work = self._dataset_database().get_work_item(work_item_id)
            return bool(work is not None and work.cancel_requested)

        def ensure_not_cancelled() -> None:
            if cancellation_requested():
                raise AcquisitionCancellationRequested(
                    "review acquisition cancellation requested"
                )

        ensure_not_cancelled()
        review.update_acquisition_progress(
            batch_id, status="running", stage="resolving_sources"
        )
        spool = AcquisitionRecordSpool(self.review_storage_root, work_item_id)
        diversity_revision = self._review_diversity_revision(payload)
        checkpoint = spool.checkpoint
        if not bool(checkpoint.get("sealed")):
            iterator = spool.resume_after_verified_prefix(
                self._review_acquisition_records(payload),
                check_cancelled=ensure_not_cancelled,
            )
            chunk: List[Dict[str, Any]] = []
            for raw in iterator:
                ensure_not_cancelled()
                source_record = dict(raw)
                source_record.pop(INGESTION_SOURCE_HASH_FIELD, None)
                source_hash = bytes_hash(
                    canonical_json(source_record).encode("utf-8")
                )
                chunk.append(
                    self._review_spool_envelope(
                        source_record,
                        source_hash_field=INGESTION_SOURCE_HASH_FIELD,
                        source_hash=source_hash,
                    )
                )
                if len(chunk) < 1000:
                    continue
                if diversity_revision:
                    review.update_acquisition_progress(
                        batch_id,
                        status="running",
                        stage="embedding_candidates",
                        processed_records=int(spool.checkpoint["record_count"]),
                    )
                ensure_not_cancelled()
                chunk = self._embed_review_acquisition_chunk(
                    chunk, embedding_revision=diversity_revision
                )
                ensure_not_cancelled()
                progress = spool.append(chunk)
                chunk = []
                review.update_acquisition_progress(
                    batch_id,
                    status="running",
                    stage="resolving_sources",
                    processed_records=int(progress["record_count"]),
                )
            if chunk:
                if diversity_revision:
                    review.update_acquisition_progress(
                        batch_id,
                        status="running",
                        stage="embedding_candidates",
                        processed_records=int(spool.checkpoint["record_count"]),
                    )
                ensure_not_cancelled()
                chunk = self._embed_review_acquisition_chunk(
                    chunk, embedding_revision=diversity_revision
                )
                ensure_not_cancelled()
                progress = spool.append(chunk)
                review.update_acquisition_progress(
                    batch_id,
                    status="running",
                    stage="resolving_sources",
                    processed_records=int(progress["record_count"]),
                )
            ensure_not_cancelled()
            spool.seal()
        ensure_not_cancelled()
        review.update_acquisition_progress(
            batch_id,
            status="running",
            stage="selecting_candidates",
            processed_records=int(spool.checkpoint["record_count"]),
            total_records=int(spool.checkpoint["record_count"]),
        )
        ingestion_pin = spool.source_pin()
        metadata = {
            **dict(payload.get("metadata") or {}),
            "ingestion": {
                key: value
                for key, value in ingestion_pin.items()
                if key != "ref"
            },
        }
        try:
            batch = review.create_acquisition(
                spool.iter_records(),
                strategies=payload.get("strategies"),
                seed=int(payload.get("seed") or 0),
                filters=payload.get("filters"),
                name=self._optional_str(payload.get("name")),
                metadata=metadata,
                batch_id=batch_id,
                work_item_id=work_item_id,
                check_cancelled=ensure_not_cancelled,
            )
        except Exception:
            # The cancellation callback may be surfaced directly from Python
            # loops or as SQLite's generic "interrupted" error. Re-read the
            # durable work state so both cases converge on one truthful domain
            # cancellation before the manifest publication boundary.
            if cancellation_requested():
                return self._review_value(review.cancel_acquisition(batch_id))
            raise
        return self._review_value(batch)

    def list_acquisition_batches(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        operation = getattr(self._review_engine(), "list_acquisitions", None)
        if not callable(operation):
            operation = self._review_engine().list_acquisition_batches
        try:
            values = list(operation(status=status, limit=limit, offset=offset))
            where = "WHERE status = ?" if status else ""
            params = (status,) if status else ()
            return {
                "items": [self._review_value(value) for value in values],
                "total": self._review_sql_count(
                    "acquisition_batches", where=where, params=params
                ),
                "limit": limit,
                "offset": offset,
            }
        except TypeError:
            values = list(operation())
        if status:
            values = [value for value in values if getattr(value, "status", None) == status]
        return self._review_page(values, limit=limit, offset=offset)

    def get_acquisition_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        try:
            operation = getattr(self._review_engine(), "get_acquisition", None)
            if not callable(operation):
                operation = self._review_engine().get_acquisition_batch
            value = operation(batch_id)
        except KeyError:
            return None
        if value is None:
            return None
        resolved = self._review_value(value)
        metadata = resolved.get("metadata") if isinstance(resolved, dict) else None
        reused_batch_id = (
            str(metadata.get("reused_batch_id") or "").strip()
            if isinstance(metadata, dict)
            else ""
        )
        if reused_batch_id and reused_batch_id != batch_id:
            canonical = operation(reused_batch_id)
            if canonical is not None:
                canonical_value = self._review_value(canonical)
                return {
                    **canonical_value,
                    "work_item_id": resolved.get("work_item_id"),
                    "reused_from_batch_id": batch_id,
                }
        return resolved

    def list_acquisition_candidates(
        self, batch_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        if self.get_acquisition_batch(batch_id) is None:
            raise KeyError(batch_id)
        visible_batch = self.get_acquisition_batch(batch_id) or {}
        resolved_batch_id = str(visible_batch.get("id") or batch_id)
        try:
            values = self._review_engine().list_acquisition_candidates(
                resolved_batch_id, limit=limit, offset=offset
            )
        except TypeError:
            all_values = list(
                self._review_engine().list_acquisition_candidates(resolved_batch_id)
            )
            values = all_values[offset : offset + limit]
        batch = visible_batch
        return {
            "items": [self._review_value(value) for value in values],
            "total": int(batch.get("row_count") or len(values)),
            "limit": limit,
            "offset": offset,
        }

    def cancel_acquisition_batch(self, batch_id: str) -> Dict[str, Any]:
        operation = getattr(self._review_engine(), "cancel_acquisition", None)
        if not callable(operation):
            from halo_forge.review_lab import ReviewStateError

            raise ReviewStateError("completed immutable acquisition batches cannot be cancelled")
        visible = self.get_acquisition_batch(batch_id)
        if visible is None:
            raise KeyError(batch_id)
        batch = operation(str(visible.get("id") or batch_id))
        value = self._review_value(batch)
        work_item_id = value.get("work_item_id") if isinstance(value, dict) else None
        if work_item_id:
            self._scheduler().cancel(str(work_item_id))
        return value

    def retry_acquisition_batch(self, batch_id: str) -> Dict[str, Any]:
        operation = getattr(self._review_engine(), "retry_acquisition", None)
        if not callable(operation):
            from halo_forge.review_lab import ReviewStateError

            raise ReviewStateError("this acquisition batch has no retryable background attempt")
        batch = self.get_acquisition_batch(batch_id)
        if batch is None:
            raise KeyError(batch_id)
        work_item_id = self._optional_str(batch.get("work_item_id"))
        if not work_item_id:
            from halo_forge.review_lab import ReviewStateError

            raise ReviewStateError("this acquisition batch has no durable work item")
        retried = self._scheduler().retry(
            work_item_id,
            reason="operator requested acquisition retry",
            force=True,
            sync_domain=False,
        )
        if retried is None:
            from halo_forge.review_lab import ReviewStateError

            raise ReviewStateError("the acquisition work item could not be retried")
        value = self._review_value(operation(str(batch.get("id") or batch_id)))
        return {**value, "work_item_id": retried.id}

    def list_review_queues(
        self,
        *,
        status: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        try:
            values = list(
                self._review_engine().list_queues(
                    status=status, query=query, limit=limit, offset=offset
                )
            )
        except TypeError:
            values = list(self._review_engine().list_queues())
            if status:
                values = [value for value in values if getattr(value, "status", None) == status]
            return self._review_page(values, limit=limit, offset=offset)
        where = "WHERE status = ?" if status else ""
        params = (status,) if status else ()
        counter = getattr(self._review_engine(), "count_queues", None)
        total = (
            int(counter(status=status, query=query))
            if callable(counter)
            else self._review_sql_count("review_queues", where=where, params=params)
        )
        return {
            "items": [self._review_value(value) for value in values],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def create_review_queue(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        requested_batch_id = str(
            payload.get("batch_id") or payload.get("acquisition_batch_id") or ""
        )
        visible_batch = self.get_acquisition_batch(requested_batch_id)
        if visible_batch is None:
            raise KeyError(requested_batch_id)
        queue = self._review_engine().create_queue(
            str(visible_batch.get("id") or requested_batch_id),
            str(payload.get("schema_revision_id") or ""),
            name=self._optional_str(payload.get("name")),
            policy=dict(payload.get("policy") or {}),
        )
        return self._review_value(queue)

    def get_review_queue(self, queue_id: str) -> Optional[Dict[str, Any]]:
        try:
            value = self._review_engine().get_queue(queue_id)
        except KeyError:
            return None
        return self._review_value(value) if value is not None else None

    def clone_review_queue(self, queue_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        source = self._review_engine().get_queue(queue_id)
        if source is None:
            raise KeyError(queue_id)
        source_value = self._review_value(source)
        operation = getattr(self._review_engine(), "clone_queue", None)
        changes_identity = bool(
            payload.get("batch_id") or payload.get("schema_revision_id")
        )
        if callable(operation) and not changes_identity:
            return self._review_value(
                operation(
                    queue_id,
                    name=self._optional_str(payload.get("name")),
                    policy=(
                        dict(payload["policy"]) if isinstance(payload.get("policy"), dict) else None
                    ),
                )
            )
        return self.create_review_queue(
            {
                "batch_id": payload.get("batch_id") or source_value["acquisition_batch_id"],
                "schema_revision_id": (
                    payload.get("schema_revision_id") or source_value["schema_revision_id"]
                ),
                "name": payload.get("name") or f"{source_value['name']} (clone)",
                "policy": payload.get("policy") or source_value.get("policy") or {},
            }
        )

    def update_review_queue_state(
        self, queue_id: str, *, action: str, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        normalized = action.replace("-", "_")
        if normalized == "start_second_pass":
            return self._review_value(self._review_engine().start_second_pass(queue_id))
        operation = getattr(self._review_engine(), f"{normalized}_queue", None)
        if not callable(operation):
            from halo_forge.review_lab import ReviewStateError

            raise ReviewStateError(f"review queue action {action!r} is unavailable")
        try:
            return self._review_value(operation(queue_id, reason=reason))
        except TypeError:
            return self._review_value(operation(queue_id))

    def list_review_items(
        self,
        queue_id: str,
        *,
        status: Optional[str] = None,
        pass_number: Optional[int] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        engine = self._review_engine()
        queue = engine.get_queue(queue_id)
        if queue is None:
            raise KeyError(queue_id)
        page_limit = max(1, min(1000, int(limit)))
        page_offset = max(0, int(offset))
        operation = engine.list_items
        counter = getattr(engine, "count_items", None)
        if callable(counter):
            try:
                values = list(
                    operation(
                        queue_id,
                        status=status,
                        pass_number=pass_number,
                        query=query,
                        limit=page_limit,
                        offset=page_offset,
                    )
                )
                total = int(
                    counter(
                        queue_id,
                        status=status,
                        pass_number=pass_number,
                        query=query,
                    )
                )
                return {
                    "items": [self._review_value(value) for value in values],
                    "total": total,
                    "limit": page_limit,
                    "offset": page_offset,
                }
            except TypeError:
                # Compatibility for injected/legacy review engines that have
                # not adopted SQL-backed paging yet.
                pass
        try:
            values = list(operation(queue_id))
        except TypeError:
            values = list(operation(queue_id, status=status))
        if status:
            values = [value for value in values if getattr(value, "status", None) == status]
        if pass_number is not None:
            queue_pass = int(getattr(queue, "current_pass", 1))
            values = [
                value
                for value in values
                if int((getattr(value, "projection", {}) or {}).get("current_pass", queue_pass))
                == int(pass_number)
            ]
        return self._review_page(values, limit=page_limit, offset=page_offset)

    def list_review_queue_summaries(
        self,
        *,
        status: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Return bounded queue rows with indexed status totals for large studios."""

        page = self.list_review_queues(
            status=status,
            query=query,
            limit=max(1, min(500, int(limit))),
            offset=max(0, int(offset)),
        )
        items = list(page.get("items") or [])
        db = self._dataset_database()
        for value in items:
            queue_id = str(value.get("id") or "")
            rows = db._conn.execute(
                """SELECT status, COUNT(*) AS value FROM review_items
                   WHERE queue_id=? GROUP BY status""",
                (queue_id,),
            ).fetchall()
            counts = {str(row["status"]): int(row["value"]) for row in rows}
            value["item_counts"] = counts
            value["total_items"] = sum(counts.values())
            value["true_queue_position"] = int(
                db._conn.execute(
                    """SELECT COUNT(*) AS value FROM review_queues
                       WHERE status IN ('ready','active','paused')
                         AND created_at < (SELECT created_at FROM review_queues WHERE id=?)""",
                    (queue_id,),
                ).fetchone()["value"]
            )
        return {
            "items": items,
            "total": page.get("total", len(items)),
            "limit": page.get("limit", limit),
            "offset": page.get("offset", offset),
        }

    def get_review_item_neighbors(
        self,
        item_id: str,
        *,
        status: Optional[str] = None,
        pass_number: Optional[int] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve previous/next review items without materializing the queue."""

        item = self.get_review_item(item_id)
        if item is None:
            raise KeyError(item_id)
        queue_id = str(item.get("queue_id") or "")
        ordinal = int(item.get("ordinal") or 0)
        if pass_number is not None:
            queue = self._review_engine().get_queue(queue_id)
            if queue is None or int(queue.current_pass) != int(pass_number):
                return {
                    "item": item,
                    "previous": None,
                    "next": None,
                    "position": 0,
                    "total": 0,
                    "pass_number": int(pass_number),
                }
        db = self._dataset_database()
        where = "i.queue_id=?"
        params: list[Any] = [queue_id]
        if status:
            where += " AND i.status=?"
            params.append(status)
        search = str(query or "").strip()
        if search:
            pattern = f"%{search}%"
            where += (
                " AND (c.record_id LIKE ? OR c.record_json LIKE ? "
                "OR c.evidence_json LIKE ?)"
            )
            params.extend((pattern, pattern, pattern))
        joined = (
            "review_items i JOIN acquisition_candidates c ON c.id=i.candidate_id"
        )

        def neighbor(operator: str, ordering: str) -> Optional[Dict[str, Any]]:
            row = db._conn.execute(
                f"SELECT i.id FROM {joined} WHERE {where} AND i.ordinal {operator} ? "
                f"ORDER BY i.ordinal {ordering} LIMIT 1",
                (*params, ordinal),
            ).fetchone()
            return self.get_review_item(str(row["id"])) if row is not None else None

        position_row = db._conn.execute(
            f"SELECT COUNT(*) AS value FROM {joined} WHERE {where} AND i.ordinal < ?",
            (*params, ordinal),
        ).fetchone()
        total_row = db._conn.execute(
            f"SELECT COUNT(*) AS value FROM {joined} WHERE {where}", params
        ).fetchone()
        return {
            "item": item,
            "previous": neighbor("<", "DESC"),
            "next": neighbor(">", "ASC"),
            "position": int(position_row["value"]) + 1,
            "total": int(total_row["value"]),
            "pass_number": (
                int(pass_number)
                if pass_number is not None
                else int(getattr(self._review_engine().get_queue(queue_id), "current_pass", 1))
            ),
        }

    def get_review_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        try:
            value = self._review_engine().get_item(item_id)
        except KeyError:
            return None
        return self._review_value(value) if value is not None else None

    def submit_review_event(self, item_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        event = self._review_engine().submit_event(
            item_id,
            event_type=str(payload.get("event_type") or "label"),
            payload=dict(payload.get("payload") or {}),
            idempotency_key=str(payload.get("idempotency_key") or ""),
            expected_active_event_id=self._optional_str(payload.get("expected_active_event_id")),
            pass_number=(
                int(payload["pass_number"]) if payload.get("pass_number") is not None else None
            ),
            supersedes_event_id=self._optional_str(payload.get("supersedes_event_id")),
            reviewer_key=str(payload.get("reviewer_key") or "local"),
        )
        return self._review_value(event)

    def list_review_events(
        self, item_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        engine = self._review_engine()
        if engine.get_item(item_id) is None:
            raise KeyError(item_id)
        operation = getattr(engine, "list_events", None)
        if not callable(operation):
            return self._review_page([], limit=limit, offset=offset)
        page_limit = max(1, min(1000, int(limit)))
        page_offset = max(0, int(offset))
        counter = getattr(engine, "count_events", None)
        if callable(counter):
            try:
                values = operation(item_id, limit=page_limit, offset=page_offset)
                return {
                    "items": [self._review_value(value) for value in values],
                    "total": int(counter(item_id)),
                    "limit": page_limit,
                    "offset": page_offset,
                }
            except TypeError:
                pass
        return self._review_page(operation(item_id), limit=page_limit, offset=page_offset)

    def submit_review_event_batch(
        self, queue_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        events = payload.get("events") or []
        if not isinstance(events, list) or not events:
            raise ValueError("events must be a non-empty array")
        if len(events) > 1000:
            raise ValueError("event batches are limited to 1,000 events")
        operation = getattr(self._review_engine(), "submit_event_batch", None)
        if not callable(operation):
            from halo_forge.review_lab import ReviewStateError

            raise ReviewStateError("atomic review event batches are unavailable")
        value = self._review_value(
            operation(
                queue_id,
                events,
                reviewer_key=str(payload.get("reviewer_key") or "local"),
            )
        )
        if isinstance(value, list):
            return {"items": value, "count": len(value), "queue_id": queue_id}
        return value

    def adjudicate_review_item(self, item_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        event = self._review_engine().adjudicate(
            item_id,
            annotation=dict(payload.get("payload") or payload.get("annotation") or {}),
            reason=str(payload.get("reason") or ""),
            idempotency_key=str(payload.get("idempotency_key") or ""),
            expected_active_event_id=self._optional_str(payload.get("expected_active_event_id")),
            reviewer_key=str(payload.get("reviewer_key") or "local"),
        )
        return self._review_value(event)

    def get_review_queue_statistics(self, queue_id: str) -> Dict[str, Any]:
        return self._review_value(self._review_engine().statistics(queue_id))

    def list_review_suggestions(
        self,
        item_id: str,
        *,
        pass_number: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        engine = self._review_engine()
        operation = getattr(engine, "list_suggestions", None)
        if not callable(operation):
            return self._review_page([], limit=limit, offset=offset)
        if pass_number is None:
            item = engine.get_item(item_id)
            if item is not None:
                queue = engine.get_queue(item.queue_id)
                if queue is not None:
                    pass_number = int(queue.current_pass)
        page_limit = max(1, min(1000, int(limit)))
        page_offset = max(0, int(offset))
        counter = getattr(engine, "count_suggestions", None)
        if callable(counter):
            try:
                values = operation(
                    item_id,
                    pass_number=pass_number,
                    limit=page_limit,
                    offset=page_offset,
                )
                return {
                    "items": [self._review_value(value) for value in values],
                    "total": int(counter(item_id, pass_number=pass_number)),
                    "limit": page_limit,
                    "offset": page_offset,
                }
            except TypeError:
                pass
        return self._review_page(
            operation(item_id, pass_number=pass_number),
            limit=page_limit,
            offset=page_offset,
        )

    def generate_review_suggestions(
        self,
        item_id: str,
        payload: Dict[str, Any],
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        payload = dict(payload)
        verifier_revision_id = self._optional_str(
            payload.get("verifier_profile_revision_id")
        )
        resolved_verifier = None
        if verifier_revision_id:
            contradictory = [
                key
                for key in ("verifier", "verifier_config", "threshold", "reward_threshold")
                if payload.get(key) not in (None, "", {})
            ]
            if contradictory:
                raise ValueError(
                    "verifier_profile_revision_id conflicts with raw verifier fields: "
                    + ", ".join(contradictory)
                )
            resolved_verifier = self._verifier_engine().resolve_binding(
                verifier_revision_id
            )
            payload["resolved_verifier_binding"] = resolved_verifier
        if payload.get("output") is None and not bool(payload.get("_execute_now")):
            item = self._review_engine().get_item(item_id)
            if item is None:
                raise KeyError(item_id)
            model_revision = str(
                payload.get("model_revision") or payload.get("model") or ""
            ).strip()
            if not model_revision:
                raise ValueError("model or model_revision is required")
            parameters = {
                key: value
                for key, value in dict(payload.get("parameters") or {}).items()
                if key.lower() not in {"api_key", "token", "authorization", "password"}
            }
            provider = str(payload.get("provider") or "openai_compatible")
            suggestion_id = str(payload.get("suggestion_id") or f"suggestion-{uuid.uuid4().hex}")
            work_item_id = f"review-suggestion-{uuid.uuid4().hex}"
            queued_payload = self._sanitize_review_work_payload({
                **{
                    key: value
                    for key, value in payload.items()
                    if key not in {"api_key", "token", "authorization", "password"}
                },
                "parameters": parameters,
                "provider": provider,
                "model_revision": model_revision,
                "suggestion_id": suggestion_id,
                "_execute_now": True,
            })
            local_provider = provider.strip().lower() in {
                "ollama",
                "openai_compatible",
                "local",
            }
            work = self._scheduler().enqueue(
                kind="review_suggestion",
                launch_spec={
                    "handler": "review_lab.generate_suggestion",
                    "action": "generate_review_suggestion",
                    "name": "Generate review suggestion",
                    "item_id": item_id,
                    "payload": queued_payload,
                    "review_root": str(self.review_storage_root),
                    "dataset_root": str(self.dataset_storage_root),
                    "evaluation_root": str(self.evaluation_storage_root),
                    "artifact_root": str(self.artifact_storage_root),
                },
                resource_class="accelerator" if local_provider else "cpu",
                resource_requirements={
                    "output_path": str(self.review_storage_root),
                    "provider": provider,
                },
                domain_kind="review_suggestion",
                domain_id=suggestion_id,
                log_path=str(
                    self.review_storage_root
                    / ".work"
                    / work_item_id
                    / "suggestion.log"
                ),
                max_retries=2,
                work_item_id=work_item_id,
            )
            return {
                "id": suggestion_id,
                "item_id": item_id,
                "status": "queued",
                "provider": provider,
                "model_revision": model_revision,
                "verifier_profile_revision_id": verifier_revision_id,
                "resolved_verifier_binding": resolved_verifier,
                "work_item_id": work.id,
            }

        operation = getattr(self._review_engine(), "create_suggestion", None)
        if not callable(operation):
            from halo_forge.review_lab import ReviewStateError

            raise ReviewStateError("model-assisted review suggestions are unavailable")
        item = self._review_engine().get_item(item_id)
        if item is None:
            raise KeyError(item_id)
        item_value = self._review_value(item)
        provider = str(payload.get("provider") or "openai_compatible")
        model_revision = str(payload.get("model_revision") or payload.get("model") or "").strip()
        if not model_revision:
            raise ValueError("model or model_revision is required")
        parameters = dict(payload.get("parameters") or {})
        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            prompt = (
                "Propose an annotation for this review item. Return only the annotation value.\n"
                + json.dumps(item_value.get("record") or {}, sort_keys=True, ensure_ascii=False)
            )
        output = payload.get("output")
        if output is None:
            from halo_forge.data_lab.integrations import configured_teacher

            generation_parameters = {
                **parameters,
                "endpoint_type": provider,
                "teacher_model": model_revision,
            }
            output = configured_teacher(
                prompt,
                generation_parameters,
                item_value.get("record") or {},
            )
        if cancellation_check is not None:
            cancellation_check()
        verifier_observation = None
        if verifier_revision_id:
            verifier_observation = self._verifier_engine().invoke_revision(
                verifier_revision_id,
                {
                    **dict(item_value.get("record") or {}),
                    "candidate": output,
                    "output": output,
                    "review_item_id": item_id,
                },
            )
            if cancellation_check is not None:
                cancellation_check()
        extra_provenance = self._sanitize_review_work_payload(
            dict(payload.get("provenance") or {})
        )
        provenance = {
            **extra_provenance,
            "endpoint_type": provider,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "template_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "parameters": {
                key: value
                for key, value in parameters.items()
                if key.lower() not in {"api_key", "token", "authorization", "password"}
            },
            "verifier_profile_revision_id": verifier_revision_id,
            "resolved_verifier_binding": resolved_verifier,
            "verifier_trace": (
                verifier_observation.to_dict()
                if verifier_observation is not None
                else payload.get("verifier_trace")
            ),
            "scores": (
                {"reward": verifier_observation.reward, "passed": verifier_observation.passed}
                if verifier_observation is not None
                else payload.get("scores")
            ),
            "runtime_identity": {
                "python": sys.version.split()[0],
                "platform": sys.platform,
                "provider": provider,
            },
        }
        value = self._review_value(
            operation(
                item_id,
                provider=provider,
                model_revision=model_revision,
                output=output,
                provenance=provenance,
                pass_number=(
                    int(payload["pass_number"])
                    if payload.get("pass_number") is not None
                    else None
                ),
                suggestion_id=self._optional_str(payload.get("suggestion_id")),
            )
        )
        if verifier_revision_id:
            binding = self._verifier_engine().bind_revision(
                verifier_revision_id,
                domain_kind="review_suggestion",
                domain_id=str(value["id"]),
                role="suggestion_verifier",
                context={"review_item_id": item_id},
            )
            value["verifier_binding"] = binding.to_dict()
        if bool(payload.get("_execute_now")):
            return value
        work_item_id = self._complete_review_activity(
            kind="review_suggestion",
            domain_kind="review_suggestion",
            domain_id=str(value["id"]),
            result=value,
            name="Generate review suggestion",
        )
        return {**value, "work_item_id": work_item_id}

    def publish_label_set(self, queue_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        engine = self._review_engine()
        queue = engine.get_queue(queue_id)
        if queue is None:
            raise KeyError(queue_id)
        # Transport-neutral test doubles and third-party embedded review
        # engines may not expose the durable worker contract. Preserve their
        # existing synchronous behavior while the built-in v6 engine always
        # routes publication through the workstation scheduler.
        if not callable(getattr(engine, "execute_work_item", None)):
            value = engine.publish_label_set(
                queue_id,
                name=self._optional_str(payload.get("name")),
                output_adapter_id=self._optional_str(payload.get("output_adapter_id")),
                build_mode=self._optional_str(payload.get("build_mode")),
            )
            result = self._review_value(value)
            work_item_id = self._complete_review_activity(
                kind="review_label_publication",
                domain_kind="label_set_revision",
                domain_id=str(result["id"]),
                result=result,
                name="Publish immutable label set",
            )
            return {**result, "work_item_id": work_item_id}
        publication_id = f"label-publication-{uuid.uuid4().hex}"
        work_item_id = f"review-label-publication-{uuid.uuid4().hex}"
        work = self._scheduler().enqueue(
            kind="review_label_publication",
            launch_spec={
                "handler": "review_lab.execute_work_item",
                "action": "publish_label_set",
                "name": self._optional_str(payload.get("name"))
                or f"{getattr(queue, 'name', 'Review')} labels",
                "queue_id": queue_id,
                "output_adapter_id": self._optional_str(payload.get("output_adapter_id")),
                "build_mode": self._optional_str(payload.get("build_mode")),
                "review_root": str(self.review_storage_root),
            },
            resource_class="cpu",
            resource_requirements={"output_path": str(self.review_storage_root)},
            domain_kind="label_set_publication",
            domain_id=publication_id,
            log_path=str(
                self.review_storage_root
                / ".work"
                / work_item_id
                / "publication.log"
            ),
            max_retries=2,
            work_item_id=work_item_id,
        )
        return {
            "id": publication_id,
            "publication_id": publication_id,
            "queue_id": queue_id,
            "status": "queued",
            "work_item_id": work.id,
            "label_set_revision_id": None,
        }

    def list_label_sets(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        try:
            values = self._review_engine().list_label_sets(limit=limit, offset=offset)
        except TypeError:
            return self._review_page(
                self._review_engine().list_label_sets(), limit=limit, offset=offset
            )
        return {
            "items": [self._review_value(value) for value in values],
            "total": self._review_sql_count("label_sets"),
            "limit": limit,
            "offset": offset,
        }

    def get_label_set(self, label_set_id: str) -> Optional[Dict[str, Any]]:
        operation = getattr(self._review_engine(), "get_label_set", None)
        if not callable(operation):
            return None
        try:
            value = operation(label_set_id)
        except KeyError:
            return None
        if value is None:
            return None
        result = self._review_value(value)
        revisions_operation = getattr(self._review_engine(), "list_label_set_revisions", None)
        if callable(revisions_operation) and isinstance(result, dict):
            result["revisions"] = self._review_value(revisions_operation(label_set_id))
        return result

    def get_label_set_revision(self, revision_id: str) -> Optional[Dict[str, Any]]:
        try:
            value = self._review_engine().get_label_set_revision(revision_id)
        except KeyError:
            return None
        return self._review_value(value) if value is not None else None

    def list_label_set_items(
        self, revision_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        if self._review_engine().get_label_set_revision(revision_id) is None:
            raise KeyError(revision_id)
        operation = getattr(self._review_engine(), "list_label_set_items", None)
        if not callable(operation):
            return self._review_page([], limit=limit, offset=offset)
        try:
            values = operation(revision_id, limit=limit, offset=offset)
        except TypeError:
            return self._review_page(operation(revision_id), limit=limit, offset=offset)
        return {
            "items": [self._review_value(value) for value in values],
            "total": self._review_sql_count(
                "label_set_items", where="WHERE revision_id = ?", params=(revision_id,)
            ),
            "limit": limit,
            "offset": offset,
        }

    def verify_label_set_revision(self, revision_id: str) -> Dict[str, Any]:
        return self._review_value(self._review_engine().verify_label_set(revision_id))

    def _review_label_set_context(self, revision_id: str) -> Dict[str, Any]:
        """Resolve immutable review identities used by Dataset Lab handoff."""

        revision = self._review_engine().get_label_set_revision(revision_id)
        if revision is None:
            raise KeyError(revision_id)
        label_set = self._review_engine().get_label_set(revision.label_set_id)
        if label_set is None:
            raise ValueError("label-set revision has no catalog entry")
        queue = self._review_engine().get_queue(label_set.queue_id)
        if queue is None:
            raise ValueError("label-set revision has no review queue")
        schema_revision = self._review_engine().get_schema_revision(queue.schema_revision_id)
        if schema_revision is None:
            raise ValueError("label-set revision has no annotation schema revision")
        return {
            "revision": revision,
            "label_set": label_set,
            "queue": queue,
            "schema_revision": schema_revision,
        }

    def _all_review_label_set_items(self, revision_id: str) -> list[Any]:
        """Read an immutable label set in bounded pages without truncating it."""

        operation = getattr(self._review_engine(), "list_label_set_items", None)
        if not callable(operation):
            raise ValueError("review label-set items are unavailable")
        values: list[Any] = []
        offset = 0
        while True:
            page = list(operation(revision_id, limit=1000, offset=offset))
            values.extend(page)
            offset += len(page)
            if len(page) < 1000:
                break
        return values

    def _resolve_review_dataset_target(
        self,
        revision_id: str,
        payload: Dict[str, Any],
        *,
        create: bool,
    ) -> tuple[Optional[Any], str]:
        db = self._dataset_database()
        dataset_id = str(payload.get("dataset_id") or "").strip()
        parent_version_id = str(payload.get("parent_version_id") or "").strip()
        if not dataset_id and not parent_version_id:
            context = self._review_label_set_context(revision_id)
            queue = context["queue"]
            source_rows = db._conn.execute(
                """SELECT DISTINCT source_kind,source_ref
                   FROM acquisition_candidates
                   WHERE batch_id=? AND source_ref IS NOT NULL""",
                (queue.acquisition_batch_id,),
            ).fetchall()
            dataset_version_refs = {
                str(row["source_ref"])
                for row in source_rows
                if str(row["source_kind"] or "").strip().lower()
                in {"dataset", "dataset_version"}
            }
            other_sources = [
                row
                for row in source_rows
                if str(row["source_kind"] or "").strip().lower()
                not in {"dataset", "dataset_version"}
            ]
            if len(dataset_version_refs) == 1 and not other_sources:
                inferred_version_id = next(iter(dataset_version_refs))
                inferred_parent = db.get_dataset_version(inferred_version_id)
                if inferred_parent is not None:
                    parent_version_id = inferred_parent.id
                    dataset_id = inferred_parent.dataset_id
                    payload["parent_version_id"] = parent_version_id
                    payload["dataset_id"] = dataset_id
        parent = db.get_dataset_version(parent_version_id) if parent_version_id else None
        if parent_version_id and parent is None:
            raise KeyError(parent_version_id)
        if parent is not None:
            if dataset_id and dataset_id != parent.dataset_id:
                raise ValueError("parent version belongs to a different dataset")
            dataset_id = parent.dataset_id
        dataset = db.get_dataset(dataset_id) if dataset_id else None
        if dataset_id and dataset is None:
            raise KeyError(dataset_id)
        if dataset is not None or not create:
            return dataset, dataset_id or f"new-review-dataset-{revision_id[:16]}"

        from halo_forge.data_lab.models import infer_schema

        context = self._review_label_set_context(revision_id)
        items = self._all_review_label_set_items(revision_id)
        inferred_kind = None
        for item in items:
            for record in list(getattr(item, "output_records", ()) or ()):
                try:
                    candidate_kind = infer_schema(record).value
                    validate_record(record, candidate_kind)
                except Exception:
                    continue
                inferred_kind = candidate_kind
                break
            if inferred_kind is not None:
                break
        if inferred_kind is None:
            raise ValueError("label-set revision has no valid included output records")
        schema_revision = context["schema_revision"]
        canonical_schema = str(payload.get("canonical_schema") or inferred_kind)
        modality = str(
            payload.get("modality")
            or {"vlm": "image", "audio": "audio"}.get(schema_revision.modality, "text")
        )
        dataset = db.create_dataset(
            name=str(payload.get("name") or f"{context['label_set'].name} reviewed"),
            description="Created from immutable Review Studio label-set revision " + revision_id,
            modality=modality,
            canonical_schema=canonical_schema,
        )
        return dataset, dataset.id

    def preview_label_set_dataset(
        self, revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        from halo_forge.data_lab import ReviewDatasetBuilder

        review = self._review_engine()
        required = (
            "get_label_set",
            "get_queue",
            "get_schema_revision",
            "list_label_set_items",
        )
        if not all(callable(getattr(review, name, None)) for name in required):
            rendered = review.render_label_set(
                revision_id,
                output_adapter_id=self._optional_str(payload.get("output_adapter_id")),
                build_mode=self._optional_str(payload.get("build_mode")),
            )
            value = self._review_value(rendered)
            rows = value.get("records", value.get("items", []))
            return {
                **value,
                "revision_id": revision_id,
                "items": list(rows or [])[:100],
                "total": len(rows or []),
                "limit": min(100, len(rows or [])),
                "offset": 0,
                "dataset_id": payload.get("dataset_id"),
                "parent_version_id": payload.get("parent_version_id"),
                "starts_training": False,
            }

        context = self._review_label_set_context(revision_id)
        target_dataset, dataset_id = self._resolve_review_dataset_target(
            revision_id, payload, create=False
        )
        revision = context["revision"]
        rendering = dict(revision.manifest.get("rendering") or {})
        requested_mode = str(
            payload.get("build_mode") or rendering.get("build_mode") or "append"
        )
        adapter = dict(rendering.get("adapter") or {})
        supported_modes = {str(value) for value in adapter.get("build_modes") or []}
        if supported_modes and requested_mode not in supported_modes:
            raise ValueError(
                f"label-set adapter supports {sorted(supported_modes)}, not {requested_mode!r}"
            )
        preview = ReviewDatasetBuilder(self._dataset_engine().store).preview(
            revision,
            self._all_review_label_set_items(revision_id),
            dataset_id=dataset_id,
            parent_version_id=self._optional_str(payload.get("parent_version_id")),
            build_mode=requested_mode,
            target_split=str(payload.get("target_split") or "train"),
            schema=(
                self._optional_str(payload.get("canonical_schema"))
                or (
                    str(target_dataset.canonical_schema)
                    if target_dataset is not None
                    else None
                )
            ),
        )
        value = preview.to_dict()
        return {
            **value,
            "revision_id": revision_id,
            "items": list(value.get("sample") or []),
            "total": int(value.get("output_count") or 0),
            "limit": len(value.get("sample") or []),
            "offset": 0,
            "new_dataset": not bool(payload.get("dataset_id")),
            "starts_training": False,
        }

    def build_label_set_dataset(
        self, revision_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        context = self._review_label_set_context(revision_id)
        dataset, dataset_id = self._resolve_review_dataset_target(
            revision_id, payload, create=True
        )
        assert dataset is not None
        revision = context["revision"]
        rendering = dict(revision.manifest.get("rendering") or {})
        requested_mode = str(
            payload.get("build_mode") or rendering.get("build_mode") or "append"
        )
        adapter = dict(rendering.get("adapter") or {})
        supported_modes = {str(value) for value in adapter.get("build_modes") or []}
        if supported_modes and requested_mode not in supported_modes:
            raise ValueError(
                f"label-set adapter supports {sorted(supported_modes)}, not {requested_mode!r}"
            )
        db = self._dataset_database()
        created_for_build = not bool(payload.get("dataset_id")) and not bool(
            payload.get("parent_version_id")
        )
        try:
            engine_job = self._dataset_engine().start_review_build_job(
                revision_id,
                review_root=self.review_storage_root,
                database_path=str(getattr(db, "path", "") or "") or None,
                dataset_id=dataset_id,
                parent_version_id=self._optional_str(payload.get("parent_version_id")),
                build_mode=requested_mode,
                target_split=str(payload.get("target_split") or "train"),
                materialize_assets=bool(payload.get("materialize_assets", False)),
                schema=dataset.canonical_schema,
            )
            engine_data = self._normalize_engine_job_data(self._data_object(engine_job))
            job_id = str(engine_data.get("id") or engine_data.get("job_id") or "")
            if not job_id:
                raise ValueError("Dataset Lab did not return a review-build job id")
            request = {
                "revision_id": revision_id,
                "dataset_id": dataset_id,
                "parent_version_id": payload.get("parent_version_id"),
                "build_mode": requested_mode,
                "target_split": str(payload.get("target_split") or "train"),
                "materialize_assets": bool(payload.get("materialize_assets", False)),
                "schema": dataset.canonical_schema,
            }
            job = db.get_dataset_job(job_id) or db.create_dataset_job(
                job_id=job_id,
                dataset_id=dataset_id,
                job_type="review_build",
                status=str(engine_data.get("status") or "queued"),
                stage=str(engine_data.get("stage") or "queued"),
                request=request,
                work_item_id=engine_data.get("work_item_id"),
            )
        except Exception:
            if created_for_build and not db.list_dataset_versions(dataset_id):
                db.delete_dataset(dataset_id)
            raise
        return {
            **job.to_dict(),
            "job_id": job.id,
            "dataset_id": dataset_id,
            "new_dataset": created_for_build,
            "starts_training": False,
        }

    # ----- persistent benchmark suites and evaluation --------------------

    @staticmethod
    def _benchmark_revision_view(revision: Any) -> Dict[str, Any]:
        value = revision.to_dict()
        value["revision"] = value.pop("revision_number", 0)
        return value

    def _benchmark_suite_view(
        self, suite: Any, *, include_revisions: bool = False
    ) -> Dict[str, Any]:
        revisions = self._dataset_database().list_benchmark_suite_revisions(suite.id)
        latest = next(
            (value for value in revisions if value.id == suite.latest_revision_id),
            revisions[0] if revisions else None,
        )
        value = {
            **suite.to_dict(),
            "revision_count": len(revisions),
            "latest_revision": (
                self._benchmark_revision_view(latest) if latest is not None else None
            ),
        }
        if include_revisions:
            value["revisions"] = [self._benchmark_revision_view(revision) for revision in revisions]
        return value

    def list_benchmark_suites(self) -> Dict[str, Any]:
        return {
            "items": [
                self._benchmark_suite_view(value)
                for value in self._dataset_database().list_benchmark_suites()
            ]
        }

    def create_benchmark_suite(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        suite, _ = self._evaluation_engine().create_suite(
            name=str(payload.get("name") or ""),
            description=self._optional_str(payload.get("description")),
            purpose=str(payload.get("purpose") or "unspecified"),
            items=list(payload.get("items") or []),
            primary_metric=str(payload.get("primary_metric") or "score"),
            direction=str(payload.get("direction") or "maximize"),
            generation_settings=dict(payload.get("generation_settings") or {}),
            evaluator_versions=dict(payload.get("evaluator_versions") or {}),
        )
        return self._benchmark_suite_view(suite)

    def get_benchmark_suite(self, suite_id: str) -> Optional[Dict[str, Any]]:
        suite = self._dataset_database().get_benchmark_suite(suite_id)
        return (
            self._benchmark_suite_view(suite, include_revisions=True) if suite is not None else None
        )

    def get_benchmark_suite_revision(self, revision_id: str) -> Optional[Dict[str, Any]]:
        revision = self._dataset_database().get_benchmark_suite_revision(revision_id)
        return self._benchmark_revision_view(revision) if revision is not None else None

    def create_benchmark_suite_revision(
        self, suite_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        revision = self._evaluation_engine().create_revision(
            suite_id=suite_id,
            items=list(payload.get("items") or []),
            primary_metric=str(payload.get("primary_metric") or "score"),
            direction=str(payload.get("direction") or "maximize"),
            generation_settings=dict(payload.get("generation_settings") or {}),
            evaluator_versions=dict(payload.get("evaluator_versions") or {}),
        )
        return self._benchmark_revision_view(revision)

    def _evaluation_view(self, evaluation: Any, *, detail: bool = False) -> Dict[str, Any]:
        db = self._dataset_database()
        revision = db.get_benchmark_suite_revision(evaluation.suite_revision_id)
        suite = db.get_benchmark_suite(revision.suite_id) if revision else None
        subject_envelope = dict(evaluation.request.get("subject") or {})
        subject_payload = dict(subject_envelope.get("payload") or {})
        kind = str(subject_envelope.get("subject_type") or evaluation.subject_type)
        subject_value = str(subject_envelope.get("subject_ref") or evaluation.subject_ref)
        run_id = subject_payload.get("run_id")
        if not run_id and kind in {"run", "final_model"}:
            run_id = subject_value
        metrics = [value.to_dict() for value in db.list_evaluation_metrics(evaluation.id)]
        primary = None
        if revision:
            primary = next(
                (
                    value
                    for value in metrics
                    if value.get("name") == revision.primary_metric
                    and not value.get("suite_item_id")
                ),
                next(
                    (value for value in metrics if value.get("name") == revision.primary_metric),
                    metrics[0] if metrics else None,
                ),
            )
        total = evaluation.total_samples
        progress = (
            min(100.0, 100.0 * evaluation.processed_samples / total)
            if total
            else (100.0 if evaluation.status == "completed" else 0.0)
        )
        value = {
            **evaluation.to_dict(),
            "suite_id": revision.suite_id if revision else None,
            "suite_name": suite.name if suite else None,
            "subject": {
                "kind": kind,
                "value": subject_value,
                "run_id": run_id,
                "checkpoint": subject_payload.get("checkpoint"),
                "revision": subject_payload.get("revision"),
                "subject_hash": evaluation.subject_hash,
            },
            "subject_hash": evaluation.subject_hash,
            "run_id": run_id,
            "progress_percent": progress,
            "metrics": metrics,
            "primary_metric": primary,
            "finished_at": evaluation.completed_at,
        }
        if detail:
            value["result"] = evaluation.result
        return value

    def list_evaluations(
        self, *, run_id: Optional[str] = None, status: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        records = self._dataset_database().list_evaluations(
            subject_ref=run_id, status=status, limit=limit
        )
        return {"items": [self._evaluation_view(value) for value in records]}

    def launch_evaluation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        revision_id = str(payload.get("suite_revision_id") or "")
        revision = self._dataset_database().get_benchmark_suite_revision(revision_id)
        if revision is None:
            raise ValueError(f"Unknown benchmark suite revision: {revision_id}")
        adapter_id = self._optional_str(payload.get("adapter_id"))
        subject_value = payload.get("subject")
        if not isinstance(subject_value, dict):
            raise ValueError("subject is required")
        subject_kind = str(subject_value.get("kind") or subject_value.get("type") or "model")
        subject_run_id = self._optional_str(subject_value.get("run_id"))
        if not subject_run_id and subject_kind in {"run", "final_model", "checkpoint"}:
            subject_run_id = self._optional_str(subject_value.get("value"))
            if subject_kind == "checkpoint" and subject_run_id:
                subject_run_id = subject_run_id.split(":", 1)[0]
        if subject_run_id and self._dataset_database().get_run(subject_run_id) is None:
            from halo_forge.run_db import sync_from_filesystem

            sync_from_filesystem(self._dataset_database())
        subject = {
            "type": subject_kind,
            "ref": subject_value.get("value") or subject_value.get("ref"),
            "run_id": subject_value.get("run_id"),
            "checkpoint": subject_value.get("checkpoint"),
            "revision": subject_value.get("revision"),
            "content_hash": subject_value.get("content_hash"),
            "path": subject_value.get("path"),
        }
        request = dict(payload.get("request") or {})
        verifier_revision_id = self._optional_str(
            payload.get("verifier_profile_revision_id")
            or request.get("verifier_profile_revision_id")
        )
        resolved_verifier_binding = None
        if verifier_revision_id:
            contradictory = [
                key
                for key in ("verifier", "verifier_config", "pass_threshold", "reward_threshold")
                if payload.get(key) not in (None, "", {})
                or request.get(key) not in (None, "", {})
            ]
            if contradictory:
                raise ValueError(
                    "verifier_profile_revision_id conflicts with raw verifier fields: "
                    + ", ".join(sorted(set(contradictory)))
                )
            resolved_verifier_binding = self._verifier_engine().resolve_binding(
                verifier_revision_id
            )
            request["verifier_profile_revision_id"] = verifier_revision_id
            request["resolved_verifier_binding"] = resolved_verifier_binding
        request.setdefault("generation_settings", revision.generation_settings)
        request.setdefault("dataset_root", str(self.dataset_storage_root))
        launched = self._evaluation_engine().launch_evaluation(
            suite_revision_id=revision_id,
            adapter_id=adapter_id,
            subject=subject,
            request=request,
        )
        response = self._evaluation_view(launched.evaluation)
        if verifier_revision_id:
            binding = self._verifier_engine().bind_revision(
                verifier_revision_id,
                domain_kind="evaluation",
                domain_id=str(launched.evaluation.id),
                role="evaluation_verifier",
                context={"suite_revision_id": revision_id, "adapter_id": adapter_id},
            )
            response["verifier_binding"] = binding.to_dict()
        return response

    def launch_evaluation_batch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Launch one base and at most four candidates with shared pinned settings."""

        base = payload.get("base") or payload.get("base_subject")
        candidates = list(payload.get("candidates") or payload.get("candidate_subjects") or [])
        if not isinstance(base, Mapping):
            raise ValueError("base subject is required")
        if not 1 <= len(candidates) <= 4 or not all(
            isinstance(value, Mapping) for value in candidates
        ):
            raise ValueError("evaluation batches require one to four candidate subjects")
        revision_id = str(payload.get("suite_revision_id") or "").strip()
        batch_request = dict(payload.get("request") or {})
        verifier_revision_id = self._optional_str(
            payload.get("verifier_profile_revision_id")
            or batch_request.get("verifier_profile_revision_id")
        )
        if verifier_revision_id:
            contradictory = [
                key
                for key in (
                    "verifier",
                    "verifier_config",
                    "pass_threshold",
                    "reward_threshold",
                )
                if payload.get(key) not in (None, "", {})
                or batch_request.get(key) not in (None, "", {})
            ]
            if contradictory:
                raise ValueError(
                    "verifier_profile_revision_id conflicts with raw verifier fields: "
                    + ", ".join(sorted(set(contradictory)))
                )
            # Resolve once before creating any child evaluation so an invalid
            # exact revision cannot leave a partially launched comparison.
            self._verifier_engine().resolve_binding(verifier_revision_id)
        batch_hash = hashlib.sha256(
            json.dumps(
                {
                    "suite_revision_id": revision_id,
                    "adapter_id": payload.get("adapter_id"),
                    "verifier_profile_revision_id": verifier_revision_id,
                    "base": dict(base),
                    "candidates": [dict(value) for value in candidates],
                    "request": batch_request,
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        batch_id = f"evaluation-batch-{batch_hash[:24]}"
        launched: list[Dict[str, Any]] = []
        for ordinal, subject in enumerate([base, *candidates]):
            role = "base" if ordinal == 0 else "candidate"
            request = {
                **batch_request,
                "evaluation_batch_id": batch_id,
                "evaluation_batch_role": role,
                "evaluation_batch_ordinal": ordinal,
            }
            launched.append(
                self.launch_evaluation(
                    {
                        "suite_revision_id": revision_id,
                        "adapter_id": payload.get("adapter_id"),
                        "subject": dict(subject),
                        "request": request,
                        "verifier_profile_revision_id": verifier_revision_id,
                    }
                )
            )
        return {
            "id": batch_id,
            "suite_revision_id": revision_id,
            "verifier_profile_revision_id": verifier_revision_id,
            "base_evaluation_id": launched[0]["id"],
            "candidate_evaluation_ids": [value["id"] for value in launched[1:]],
            "evaluations": launched,
            "work_item_ids": [
                value.get("work_item_id")
                for value in launched
                if value.get("work_item_id")
            ],
        }

    def get_evaluation_batch_comparison_samples(
        self,
        batch_id: str,
        *,
        candidate_id: Optional[str] = None,
        classification: Optional[str] = None,
        query: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Return a bounded, server-filtered sample page for a persisted batch."""

        records = self._dataset_database().list_evaluations(limit=10000)
        matching = [
            value
            for value in records
            if str(value.request.get("evaluation_batch_id") or "") == batch_id
        ]
        matching.sort(
            key=lambda value: int(value.request.get("evaluation_batch_ordinal") or 0)
        )
        base = next(
            (
                value
                for value in matching
                if value.request.get("evaluation_batch_role") == "base"
            ),
            None,
        )
        candidates = [
            value
            for value in matching
            if value.request.get("evaluation_batch_role") == "candidate"
        ]
        if base is None or not candidates:
            raise KeyError(batch_id)
        selected = (
            next((value for value in candidates if value.id == candidate_id), None)
            if candidate_id
            else candidates[0]
        )
        if selected is None:
            raise KeyError(candidate_id or batch_id)
        # Fetch in bounded chunks until enough filtered evidence is available.
        search = str(query or "").strip().casefold()
        desired = max(1, min(1000, int(limit)))
        raw_offset = 0
        filtered: list[Dict[str, Any]] = []
        total_unfiltered = 0
        while len(filtered) < int(offset) + desired:
            page = self.compare_evaluations(
                base.id, selected.id, offset=raw_offset, limit=min(1000, desired * 4)
            )
            values = list(page.get("samples") or [])
            total_unfiltered = int(page.get("sample_total") or len(values))
            if not values:
                break
            for value in values:
                if classification and str(value.get("classification")) != classification:
                    continue
                if search and search not in json.dumps(
                    value, sort_keys=True, default=str
                ).casefold():
                    continue
                filtered.append(value)
            raw_offset += len(values)
            if raw_offset >= total_unfiltered:
                break
        page_items = filtered[int(offset) : int(offset) + desired]
        return {
            "batch_id": batch_id,
            "base_evaluation_id": base.id,
            "candidate_evaluation_id": selected.id,
            "items": page_items,
            "total": len(filtered) if raw_offset >= total_unfiltered else total_unfiltered,
            "limit": desired,
            "offset": max(0, int(offset)),
            "filters": {"classification": classification, "q": query},
        }

    def get_evaluation(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        value = self._dataset_database().get_evaluation(evaluation_id)
        return self._evaluation_view(value, detail=True) if value is not None else None

    def get_evaluation_samples(
        self, evaluation_id: str, *, offset: int = 0, limit: int = 100
    ) -> Dict[str, Any]:
        db = self._dataset_database()
        if db.get_evaluation(evaluation_id) is None:
            raise KeyError(evaluation_id)
        page_offset = max(0, offset)
        page_limit = max(1, limit)
        page = db.list_evaluation_samples(
            evaluation_id,
            offset=page_offset,
            limit=page_limit,
        )
        return {
            "items": [value.to_dict() for value in page],
            "total": db.count_evaluation_samples(evaluation_id),
            "offset": page_offset,
            "limit": page_limit,
        }

    def list_evaluation_jobs(self) -> Dict[str, Any]:
        items = []
        for status in ("running", "queued", "interrupted", "failed", "cancelled"):
            items.extend(self._dataset_database().list_evaluations(status=status, limit=100))
        items.sort(key=lambda value: value.created_at, reverse=True)
        return {"items": [self._evaluation_view(value) for value in items]}

    def cancel_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        return self._evaluation_view(self._evaluation_engine().jobs.cancel(evaluation_id))

    def retry_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        return self._evaluation_view(self._evaluation_engine().jobs.retry(evaluation_id))

    def compare_evaluations(
        self,
        base_id: str,
        candidate_id: str,
        *,
        offset: int = 0,
        limit: int = 200,
    ) -> Dict[str, Any]:
        raw = self._evaluation_engine().compare_page(
            base_id, candidate_id, offset=offset, limit=limit
        )
        sample_page = dict(raw.get("sample_deltas") or {})
        gap_page = dict(raw.get("evidence_gaps") or {})
        primary = str(raw.get("primary_metric") or "score")
        primary_delta = next(
            (
                value
                for value in raw.get("metric_deltas", [])
                if value.get("name") == primary and not value.get("suite_item_id")
            ),
            next(
                (value for value in raw.get("metric_deltas", []) if value.get("name") == primary),
                None,
            ),
        )
        return {
            "base_id": base_id,
            "candidate_id": candidate_id,
            "suite_revision_id": raw["suite_revision_id"],
            "primary_metric": primary,
            "direction": raw["direction"],
            "base_value": primary_delta.get("base") if primary_delta else None,
            "candidate_value": primary_delta.get("candidate") if primary_delta else None,
            "delta": primary_delta.get("delta") if primary_delta else None,
            "counts": raw["counts"],
            "metrics": raw.get("metric_deltas", []),
            "suite_purpose": raw.get("suite_purpose", "unspecified"),
            "failure_mining_allowed": raw.get("failure_mining_allowed", True),
            "evidence_gaps": gap_page.get("items", []),
            "evidence_summary": raw.get("evidence_summary", {}),
            "sample_total": int(sample_page.get("total") or 0),
            "evidence_gap_total": int(gap_page.get("total") or 0),
            "offset": int(sample_page.get("offset") or 0),
            "limit": int(sample_page.get("limit") or limit),
            "samples": [
                {
                    **value,
                    "classification": value.get("outcome"),
                    "delta": (
                        value.get("candidate_score") - value.get("base_score")
                        if value.get("candidate_score") is not None
                        and value.get("base_score") is not None
                        else None
                    ),
                }
                for value in sample_page.get("items", [])
            ],
        }

    def evaluation_history(
        self,
        *,
        subject_ref: Optional[str] = None,
        suite_revision_id: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Return immutable completed evaluations as a longitudinal series."""

        rows = self._dataset_database().list_evaluations(
            subject_ref=subject_ref,
            suite_revision_id=suite_revision_id,
            status="completed",
            limit=max(1, int(limit)),
        )
        items = [self._evaluation_view(value) for value in rows]
        for ordinal, item in enumerate(reversed(items)):
            item["history_ordinal"] = ordinal
            metric = item.get("primary_metric") or {}
            item["primary_value"] = metric.get("value")
        return {
            "items": items,
            "total": len(items),
            "subject_ref": subject_ref,
            "suite_revision_id": suite_revision_id,
            "limit": max(1, int(limit)),
        }

    def evaluation_drift(
        self,
        *,
        base_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        subject_ref: Optional[str] = None,
        suite_revision_id: Optional[str] = None,
        practical_delta: float = 0.0,
    ) -> Dict[str, Any]:
        """Compare two compatible points in an evaluation history."""

        threshold = float(practical_delta)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("practical_delta must be a finite non-negative number")

        db = self._dataset_database()
        base = db.get_evaluation(str(base_id)) if base_id else None
        candidate = db.get_evaluation(str(candidate_id)) if candidate_id else None
        if base_id and base is None:
            raise ValueError(f"unknown evaluation in drift comparison: {base_id}")
        if candidate_id and candidate is None:
            raise ValueError(f"unknown evaluation in drift comparison: {candidate_id}")

        # Supplying one endpoint means "compare it with the nearest compatible
        # historical point", not "compare it with itself" when it is newest.
        anchor = base or candidate
        if anchor is not None:
            if suite_revision_id and anchor.suite_revision_id != suite_revision_id:
                raise ValueError("evaluation does not use the requested suite revision")
            if subject_ref and anchor.subject_ref != subject_ref:
                raise ValueError("evaluation does not use the requested subject reference")
            suite_revision_id = suite_revision_id or anchor.suite_revision_id
            subject_ref = subject_ref or anchor.subject_ref
        elif not subject_ref and not suite_revision_id:
            raise ValueError(
                "evaluation drift requires two evaluation ids, a subject, or a suite revision"
            )

        if base is None or candidate is None:
            history = self.evaluation_history(
                subject_ref=subject_ref,
                suite_revision_id=suite_revision_id,
                limit=100,
            )["items"]
            excluded = {str(value) for value in (base_id, candidate_id) if value is not None}
            available = [value for value in history if value["id"] not in excluded]
            if anchor is None and len(available) >= 2:
                candidate_id = available[0]["id"]
                base_id = available[1]["id"]
            elif base is not None and available:
                candidate_id = available[0]["id"]
            elif candidate is not None and available:
                base_id = available[0]["id"]
            else:
                raise ValueError("evaluation drift requires two completed compatible evaluations")
            base = db.get_evaluation(str(base_id))
            candidate = db.get_evaluation(str(candidate_id))
        if base is None or candidate is None:
            raise ValueError("unknown evaluation in drift comparison")
        if base.id == candidate.id:
            raise ValueError("evaluation drift requires two distinct evaluations")
        if base.status != "completed" or candidate.status != "completed":
            raise ValueError("evaluation drift requires completed evaluations")
        if base.suite_revision_id != candidate.suite_revision_id:
            raise ValueError("evaluation drift requires the same immutable suite revision")
        if suite_revision_id and base.suite_revision_id != suite_revision_id:
            raise ValueError("evaluations do not use the requested suite revision")
        if subject_ref and (
            base.subject_ref != subject_ref or candidate.subject_ref != subject_ref
        ):
            raise ValueError("evaluations do not use the requested subject reference")
        comparison = self.compare_evaluations(str(base_id), str(candidate_id), limit=200)
        delta = comparison.get("delta")
        signed_delta = (
            None
            if delta is None
            else float(delta) * (1.0 if comparison.get("direction") == "maximize" else -1.0)
        )
        if signed_delta is None:
            classification = "unavailable"
        elif abs(signed_delta) <= threshold:
            classification = "practically_equivalent"
        elif signed_delta > 0:
            classification = "improved"
        else:
            classification = "regressed"
        return {
            **comparison,
            "classification": classification,
            "practical_delta": threshold,
            "compatible": True,
            "history_contract": {
                "suite_revision_id": base.suite_revision_id,
                "direction": comparison.get("direction"),
                "comparison": "immutable_evaluation_pair",
            },
        }

    # ----- durable workstation queue -------------------------------------

    # ----- reproducible experiment operations -----------------------------

    def _experiment_work_item_view(self, value: Any) -> Dict[str, Any]:
        """Normalize an embedded queue record without another database lookup."""
        if not isinstance(value, dict):
            return self._work_item_view(value)
        launch_spec = dict(value.get("launch_spec") or value.get("payload") or {})
        progress = dict(value.get("progress") or {})
        item_id = str(value.get("id") or "")
        db = self._dataset_database()
        return {
            **value,
            "payload": launch_spec,
            "run_group_id": launch_spec.get("run_group_id"),
            "trial_id": launch_spec.get("trial_id"),
            "trial_run_id": launch_spec.get("trial_run_id"),
            "segment_id": launch_spec.get("trial_segment_id"),
            "run_id": value.get("canonical_run_id"),
            "attempt": int(value.get("retry_count") or 0) + 1,
            "max_attempts": int(value.get("max_retries") or 0) + 1,
            "progress_current": progress.get("current", progress.get("processed")),
            "progress_total": progress.get("total"),
            "queue_position": db.work_item_queue_position(item_id) if item_id else None,
            "blockers": db.work_item_blockers(item_id) if item_id else [],
        }

    def _run_group_view(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Project durable orchestration records into the public UI contract."""
        value = dict(raw)
        objective = dict(value.get("objective") or {})
        sampler_state = dict(value.get("sampler_state") or {})
        objective = objective or dict(sampler_state.get("development_objective") or {})
        best = value.get("best_trial") or {}
        trials: list[Dict[str, Any]] = []
        for raw_trial in value.get("trials") or []:
            trial = dict(raw_trial)
            cohort = dict(trial.get("cohort") or {})
            runs: list[Dict[str, Any]] = []
            for raw_run in trial.get("runs") or []:
                run = dict(raw_run)
                segments = list(run.get("segments") or [])
                latest = dict(segments[-1]) if segments else {}
                related_work = [
                    self._experiment_work_item_view(item) for item in run.get("work_items") or []
                ]
                evaluation_work = next(
                    (item for item in reversed(related_work) if item.get("kind") == "evaluation"),
                    None,
                )
                runs.append(
                    {
                        **run,
                        "segment_count": len(segments),
                        "model_artifact_id": latest.get("checkpoint_artifact_id"),
                        "evaluation_id": (
                            (evaluation_work or {}).get("result", {}).get("evaluation_id")
                            if isinstance((evaluation_work or {}).get("result"), dict)
                            else None
                        ),
                        "work_items": related_work,
                    }
                )
            prune_reason = next(
                (
                    segment.get("decision_reason")
                    for run in runs
                    for segment in reversed(run.get("segments") or [])
                    if segment.get("decision") == "prune"
                ),
                None,
            )
            trials.append(
                {
                    **trial,
                    "parameters": dict(trial.get("sampled_config") or {}),
                    "aggregate": {
                        **cohort,
                        "count": cohort.get("completed_count", 0),
                        "stddev": cohort.get("standard_deviation"),
                        "direction": objective.get("direction"),
                    },
                    "pruned": trial.get("status") == "pruned",
                    "prune_reason": prune_reason,
                    "runs": runs,
                }
            )
        if not trials and value.get("trial_count"):
            trial_count = int(value["trial_count"])
        else:
            trial_count = len(trials)
        statuses = [str(trial.get("status") or "") for trial in trials]
        completed_trials = (
            statuses.count("completed") if trials else int(value.get("completed_trials") or 0)
        )
        failed_trials = statuses.count("failed") if trials else int(value.get("failed_trials") or 0)
        pruned_trials = statuses.count("pruned") if trials else int(value.get("pruned_trials") or 0)
        work_items = [
            self._experiment_work_item_view(item) for item in value.get("work_items") or []
        ]
        return {
            **value,
            "suite_revision_id": (
                objective.get("suite_revision_id") or value.get("development_suite_revision_id")
            ),
            "primary_metric": objective.get("metric") or value.get("objective_metric"),
            "direction": objective.get("direction") or value.get("objective_direction"),
            "base_config": dict(value.get("resolved_launch_config") or {}),
            "n_trials": trial_count,
            "pruning": dict(value.get("pruning_policy") or {}),
            "best_trial_id": best.get("trial_id"),
            "best_value": best.get("objective_value"),
            "completed_trials": completed_trials,
            "failed_trials": failed_trials,
            "pruned_trials": pruned_trials,
            "trials": trials or value.get("trials"),
            "work_items": work_items,
        }

    def list_run_groups(
        self,
        *,
        kind: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        items = self._experiment_engine().list_run_groups(kind=kind, status=status, limit=limit)
        return {"items": [self._run_group_view(value) for value in items]}

    def create_run_group(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        values = dict(payload)
        config = dict(values.get("base_config") or {})
        if not values.get("dataset_bindings") and config.get("dataset_version_id"):
            values["dataset_bindings"] = [
                {
                    "role": "train",
                    "dataset_version_id": str(config["dataset_version_id"]),
                    "split": str(config.get("dataset_split") or "train"),
                }
            ]
        created = self._experiment_engine().create_group_from_payload(values)
        return self._run_group_view(created)

    def get_run_group(self, run_group_id: str) -> Dict[str, Any]:
        return self._run_group_view(self._experiment_engine().get_run_group_detail(run_group_id))

    def cancel_run_group(self, run_group_id: str) -> Dict[str, Any]:
        return self._run_group_view(self._experiment_engine().cancel_run_group(run_group_id))

    def resume_run_group(
        self, run_group_id: str, *, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        engine = self._experiment_engine()
        if reason is None:
            value = engine.resume_run_group(run_group_id)
        else:
            value = engine.resume_run_group(run_group_id, reason=str(reason).strip())
        return self._run_group_view(value)

    def advance_run_group(self, run_group_id: str, rung_index: int) -> Dict[str, Any]:
        return self._experiment_engine().advance_successive_halving(
            run_group_id, rung_index=int(rung_index)
        )

    def compare_run_group(
        self, run_group_id: str, other_run_group_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if other_run_group_id:
            return self._experiment_engine().compare_run_groups(run_group_id, other_run_group_id)
        detail = self.get_run_group(run_group_id)
        return {
            "run_group_id": run_group_id,
            "suite_revision_id": detail.get("suite_revision_id"),
            "primary_metric": detail.get("primary_metric"),
            "direction": detail.get("direction"),
            "ranking": detail.get("ranking") or [],
            "cohort_aggregates": detail.get("cohort_aggregates") or [],
            "best_trial": detail.get("best_trial"),
        }

    def fork_best_run_group(
        self, run_group_id: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        options = dict(payload or {})
        created = self._experiment_engine().fork_best(
            run_group_id,
            name=self._optional_str(options.get("name")),
            seeds=options.get("seeds"),
            priority=int(options.get("priority", 0)),
            max_retries=int(options.get("max_retries", 0)),
        )
        return self._run_group_view(created)

    @staticmethod
    def _checkpoint_policy_view(
        revision: Any, *, created_at: Optional[str] = None
    ) -> Dict[str, Any]:
        value = revision.to_dict()
        revision_id = value.pop("revision_id", None)
        value["id"] = revision_id
        value["created_at"] = created_at
        return value

    def _validate_adaptive_suite(self, revision_id: str) -> Any:
        db = self._dataset_database()
        revision = db.get_benchmark_suite_revision(str(revision_id))
        if revision is None:
            raise ValueError(f"unknown benchmark suite revision: {revision_id}")
        suite = db.get_benchmark_suite(revision.suite_id)
        purpose = str(getattr(suite, "purpose", None) or "unspecified").lower()
        if purpose != "development":
            raise ValueError(
                "checkpoint policies may use development suites only; "
                f"suite {revision.suite_id} is {purpose}"
            )
        return revision

    def create_checkpoint_policy(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a named policy and one immutable revision."""

        values = dict(payload)
        name = str(values.get("name") or "").strip()
        if not name:
            raise ValueError("checkpoint policy name is required")
        development_id = str(values.get("development_suite_revision_id") or "").strip()
        if not development_id:
            raise ValueError("development_suite_revision_id is required")
        development = self._validate_adaptive_suite(development_id)
        for revision_id in values.get("guardrail_suite_revision_ids") or ():
            self._validate_adaptive_suite(str(revision_id))
        metric = str(values.get("primary_metric") or "").strip()
        if metric != development.primary_metric:
            raise ValueError(
                "checkpoint policy primary_metric must match the development suite revision"
            )
        direction = str(values.get("direction") or "").strip().lower()
        if direction != development.direction:
            raise ValueError(
                "checkpoint policy direction must match the development suite revision"
            )

        policy_id = str(values.get("policy_id") or uuid.uuid4().hex).strip()
        db = self._dataset_database()
        policy = db.get_checkpoint_policy(policy_id)
        if policy is None:
            policy = self._adaptive_engine().create_policy(
                policy_id=policy_id,
                name=name,
                description=self._optional_str(values.get("description")),
            )
        elif policy.name != name:
            raise ValueError(
                f"checkpoint policy {policy_id} already exists with name {policy.name!r}"
            )
        revision = self._adaptive_engine().create_policy_revision(policy.id, values)
        record = db.get_checkpoint_policy_revision(revision.revision_id or "")
        return self._checkpoint_policy_view(
            revision, created_at=record.created_at if record is not None else None
        )

    def list_checkpoint_policies(
        self,
        *,
        trainer_mode: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        from halo_forge.orchestration import resolve_trainer_execution_capability

        db = self._dataset_database()
        capability = resolve_trainer_execution_capability(trainer_mode) if trainer_mode else None
        items: List[Dict[str, Any]] = []
        # Capability filtering happens after hydration, so page over the filtered
        # collection rather than allowing a newer incompatible policy to hide an
        # older compatible one. This is a local, single-user catalog; the public
        # API limit remains bounded below even though we scan all policy heads.
        records = db.list_checkpoint_policies(
            archived=False,
            limit=10_000,
            offset=0,
        )
        for policy in records:
            if not policy.latest_revision_id:
                continue
            revision = self._adaptive_engine().get_policy_revision(policy.latest_revision_id)
            if revision is None:
                continue
            compatible = True
            reasons: List[str] = []
            if capability is not None:
                if not capability.supports_gated_execution:
                    compatible = False
                    reasons.append(capability.reason or "trainer is not resumable")
                if revision.schedule.unit != capability.segment_unit:
                    compatible = False
                    reasons.append(
                        f"policy uses {revision.schedule.unit}; trainer uses {capability.segment_unit}"
                    )
                missing = sorted(
                    set(revision.compatible_capabilities).difference({capability.capability_id})
                )
                if missing:
                    compatible = False
                    reasons.append("missing capabilities: " + ", ".join(missing))
            if trainer_mode and not compatible:
                continue
            revision_record = db.get_checkpoint_policy_revision(revision.revision_id or "")
            item = self._checkpoint_policy_view(
                revision,
                created_at=(revision_record.created_at if revision_record else None),
            )
            item["compatible"] = compatible
            item["compatibility_reasons"] = reasons
            items.append(item)
        page_offset = max(0, int(offset))
        page_limit = max(1, int(limit))
        page = items[page_offset : page_offset + page_limit]
        return {
            "items": page,
            "total": len(items),
            "offset": page_offset,
            "limit": page_limit,
            "has_more": page_offset + len(page) < len(items),
        }

    def get_checkpoint_policy(self, identifier: str) -> Optional[Dict[str, Any]]:
        db = self._dataset_database()
        revision = self._adaptive_engine().get_policy_revision(str(identifier))
        if revision is None:
            policy_id = str(identifier)
            revision_number: Optional[int] = None
            if ":r" in policy_id:
                policy_id, raw_revision = policy_id.rsplit(":r", 1)
                try:
                    revision_number = int(raw_revision)
                except ValueError:
                    return None
            policy = db.get_checkpoint_policy(policy_id)
            if policy is None:
                return None
            if revision_number is None:
                target_id = policy.latest_revision_id
            else:
                target_id = next(
                    (
                        value.id
                        for value in db.list_checkpoint_policy_revisions(policy.id)
                        if value.revision_number == revision_number
                    ),
                    None,
                )
            revision = self._adaptive_engine().get_policy_revision(target_id) if target_id else None
        if revision is None:
            return None
        record = db.get_checkpoint_policy_revision(revision.revision_id or "")
        return self._checkpoint_policy_view(
            revision, created_at=record.created_at if record else None
        )

    def list_checkpoint_policy_revisions(
        self, policy_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        db = self._dataset_database()
        policy = db.get_checkpoint_policy(policy_id)
        if policy is None:
            raise KeyError(policy_id)
        items = []
        for record in db.list_checkpoint_policy_revisions(policy_id):
            revision = self._adaptive_engine().get_policy_revision(record.id)
            if revision is not None:
                items.append(self._checkpoint_policy_view(revision, created_at=record.created_at))
        page_limit = max(1, int(limit))
        page_offset = max(0, int(offset))
        page = items[page_offset : page_offset + page_limit]
        return {
            "items": page,
            "total": len(items),
            "offset": page_offset,
            "limit": page_limit,
            "has_more": page_offset + len(page) < len(items),
            "policy": policy.to_dict(),
        }

    def create_checkpoint_policy_revision(
        self, policy_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        policy = self._dataset_database().get_checkpoint_policy(policy_id)
        if policy is None:
            raise KeyError(policy_id)
        values = dict(payload)
        supplied_policy_id = self._optional_str(values.get("policy_id"))
        if supplied_policy_id is not None and supplied_policy_id != policy.id:
            raise ValueError("checkpoint policy revision belongs to another policy")
        supplied_name = self._optional_str(values.get("name"))
        if supplied_name is not None and supplied_name != policy.name:
            raise ValueError("checkpoint policy revision cannot rename its policy")
        values.update(policy_id=policy.id, name=policy.name)
        values.setdefault("description", policy.description)
        return self.create_checkpoint_policy(values)

    def resolve_checkpoint_policy(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from halo_forge.orchestration import resolve_trainer_execution_capability

        revision_id = str(payload.get("policy_revision_id") or "").strip()
        trainer_mode = str(payload.get("trainer_mode") or "").strip().lower()
        if not revision_id or not trainer_mode:
            raise ValueError("policy_revision_id and trainer_mode are required")
        revision = self._adaptive_engine().get_policy_revision(revision_id)
        if revision is None:
            raise ValueError(f"unknown checkpoint policy revision: {revision_id}")
        config = dict(payload.get("base_config") or {})
        backend = str(
            config.get("backend") or config.get("accelerator") or config.get("device") or "hf"
        )
        capability = resolve_trainer_execution_capability(trainer_mode, backend)
        if not capability.supports_gated_execution:
            raise ValueError(
                capability.reason
                or f"adaptive checkpoint execution is unavailable for {trainer_mode}/{backend}"
            )
        budget_unit = str(payload.get("budget_unit") or revision.schedule.unit).lower()
        if budget_unit != revision.schedule.unit:
            raise ValueError(
                f"budget_unit {budget_unit!r} does not match policy unit "
                f"{revision.schedule.unit!r}"
            )
        plan = self._adaptive_engine().resolve_checkpoint_plan(
            revision,
            trainer_mode=trainer_mode,
            total_budget=int(payload.get("total_budget") or 0),
            supported_units=(capability.segment_unit,),
            capabilities=(capability.capability_id,),
        )
        result = plan.to_dict()
        result.update(
            estimated_checkpoint_count=len(plan.boundaries),
            estimated_evaluation_count=(
                len(plan.boundaries) * len(plan.required_suite_revision_ids)
            ),
            trainer_capability=capability.to_dict(),
        )
        return result

    @staticmethod
    def _gate_view(value: Dict[str, Any]) -> Dict[str, Any]:
        gate = dict(value)
        action = str(gate.get("action") or "")
        overridden = bool(gate.get("override_of_id"))
        if action == "pause" and not overridden:
            gate["action"] = "await_review"
            gate["status"] = "awaiting_review"
        elif overridden:
            gate["status"] = "overridden"
            gate["review_reason"] = gate.get("override_reason")
        else:
            gate["status"] = "decided"
        evidence = dict(gate.get("evidence") or {})
        boundary = dict(evidence.get("boundary") or {})
        gate["boundary_value"] = boundary.get("value")
        return gate

    def get_run_group_trajectory(self, run_group_id: str) -> Dict[str, Any]:
        raw = self._experiment_engine().get_checkpoint_trajectory(run_group_id)
        revision_id = raw.get("checkpoint_policy_revision_id")
        policy = self.get_checkpoint_policy(str(revision_id)) if revision_id else None
        primary_metric = policy.get("primary_metric") if policy else None
        points: List[Dict[str, Any]] = []
        gates: List[Dict[str, Any]] = []
        for run in raw.get("runs") or []:
            for point in run.get("points") or []:
                gate_history = [
                    self._gate_view(dict(value)) for value in point.get("gate_decisions") or ()
                ]
                if not gate_history and point.get("latest_gate_decision"):
                    gate_history = [self._gate_view(dict(point["latest_gate_decision"]))]
                override_by_original = {
                    str(value["override_of_id"]): value.get("id")
                    for value in gate_history
                    if value.get("override_of_id")
                }
                for historical in gate_history:
                    overridden_by_id = override_by_original.get(str(historical.get("id")))
                    if overridden_by_id:
                        if historical.get("action") == "await_review":
                            historical["action"] = "pause"
                        historical.update(status="superseded", overridden_by_id=overridden_by_id)
                gate = gate_history[-1] if gate_history else None
                gates.extend(gate_history)
                evidence = dict((gate or {}).get("evidence") or {})
                metrics = dict(evidence.get("current_metrics") or {})
                evaluations = list(point.get("evaluation_ids") or [])
                points.append(
                    {
                        "id": point.get("id"),
                        "run_id": run.get("run_id"),
                        "trial_id": run.get("trial_id"),
                        "trial_run_id": run.get("trial_run_id"),
                        "seed": run.get("seed"),
                        "boundary_index": int(point.get("ordinal") or 0),
                        "boundary_value": int(point.get("end_value") or 0),
                        "boundary_unit": point.get("unit") or "full_trial",
                        "status": (
                            "awaiting_review"
                            if gate and gate.get("status") == "awaiting_review"
                            else (
                                "stopped"
                                if gate and gate.get("action") == "stop"
                                else point.get("status") or run.get("status")
                            )
                        ),
                        "checkpoint_artifact_id": (
                            point.get("artifact_occurrence_id")
                            or point.get("checkpoint_artifact_id")
                        ),
                        "evaluation_id": evaluations[-1] if evaluations else None,
                        "gate_decision_id": (gate or {}).get("id"),
                        "gate_action": (gate or {}).get("action"),
                        "metric_value": metrics.get(primary_metric) if primary_metric else None,
                        "metrics": metrics,
                        "reason": " · ".join((gate or {}).get("reasons") or []) or None,
                        "created_at": point.get("created_at"),
                    }
                )
        return {
            "run_group_id": run_group_id,
            "policy_revision": policy,
            "resolved_plan": raw.get("resolved_checkpoint_plan"),
            "points": points,
            "gate_decisions": gates,
            "summary": {
                "planned_boundaries": len(
                    (raw.get("resolved_checkpoint_plan") or {}).get("boundaries") or []
                ),
                "published_boundaries": sum(
                    bool(value.get("checkpoint_artifact_id")) for value in points
                ),
                "awaiting_review": sum(
                    value.get("status") == "awaiting_review" for value in points
                ),
            },
        }

    @staticmethod
    def _analysis_view(
        record: Any, *, preferred_subject_id: Optional[str] = None
    ) -> Dict[str, Any]:
        value = record.to_dict()
        raw = dict(value.get("analysis") or {})
        comparisons = dict(raw.get("comparisons") or {})
        selected_id = (
            preferred_subject_id
            if preferred_subject_id in comparisons
            else next(iter(sorted(comparisons)), None)
        )
        selected = dict(comparisons.get(selected_id) or {})
        interval = selected.get("confidence_interval")
        evidence_compatible = bool(selected.get("evidence_compatible", True))
        compatibility = {
            "compatible": evidence_compatible,
            "reasons": [selected.get("reason")] if selected.get("reason") else [],
            "matched_seed_count": selected.get("matched_seed_count", 0),
            "required_seed_count": len(record.request.get("required_seeds") or []),
        }
        dimension_aliases = {
            "latency_ms": ("latency_ms", "total_latency_ms", "time_to_first_token_ms"),
            "throughput": ("throughput", "output_tokens_per_second", "tokens_per_second"),
            "memory_bytes": (
                "memory_bytes",
                "peak_memory_bytes",
                "peak_device_memory_bytes",
                "resource_peak_device_memory_bytes",
                "resource_peak_process_memory_bytes",
            ),
            "energy_kwh": ("energy_kwh", "resource_energy_kwh"),
            "artifact_size_bytes": ("artifact_size_bytes",),
        }
        pareto: List[Dict[str, Any]] = []
        for subject_id, summary in sorted((raw.get("subjects") or {}).items()):
            metadata = dict(summary.get("metadata_means") or {})
            row: Dict[str, Any] = {
                "subject_id": subject_id,
                "primary_metric": summary.get("mean"),
            }
            for dimension, aliases in dimension_aliases.items():
                row[dimension] = next(
                    (metadata[name] for name in aliases if metadata.get(name) is not None),
                    None,
                )
            pareto.append(row)
        directions = {
            "primary_metric": record.direction,
            "latency_ms": "minimize",
            "throughput": "maximize",
            "memory_bytes": "minimize",
            "energy_kwh": "minimize",
            "artifact_size_bytes": "minimize",
        }
        common_dimensions = [
            dimension
            for dimension in directions
            if pareto and all(row.get(dimension) is not None for row in pareto)
        ]
        for candidate in pareto:
            dominated = False
            for other in pareto:
                if other is candidate:
                    continue
                no_worse = True
                strictly_better = False
                for dimension in common_dimensions:
                    left = float(other[dimension])
                    right = float(candidate[dimension])
                    if directions[dimension] == "maximize":
                        no_worse = no_worse and left >= right
                        strictly_better = strictly_better or left > right
                    else:
                        no_worse = no_worse and left <= right
                        strictly_better = strictly_better or left < right
                if no_worse and strictly_better:
                    dominated = True
                    break
            candidate["pareto_efficient"] = not dominated
            candidate["available_dimensions"] = [
                dimension for dimension in directions if candidate.get(dimension) is not None
            ]
        value["analysis"] = {
            **raw,
            "context": dict(record.request.get("context") or {}),
            "objective_boundary": dict(
                (record.request.get("context") or {}).get("objective_boundary") or {}
            ),
            "classification": selected.get("classification", "insufficient_evidence"),
            "primary_metric": record.primary_metric,
            "direction": record.direction,
            "matched_seed_count": selected.get("matched_seed_count", 0),
            "practical_delta": record.request.get("practical_delta"),
            "interval": (
                None
                if not interval
                else {
                    "lower": interval.get("lower"),
                    "upper": interval.get("upper"),
                    "confidence": record.request.get("confidence_level", 0.95),
                }
            ),
            "compatibility": compatibility,
            "pareto": pareto,
            "pareto_dimensions": [
                {"name": name, "direction": directions[name], "common": name in common_dimensions}
                for name in directions
            ],
            "selected_comparison_subject_id": selected_id,
        }
        value["completed_at"] = record.created_at if record.status == "completed" else None
        return value

    def list_run_group_analyses(
        self, run_group_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        group = self._dataset_database().get_run_group(run_group_id)
        if group is None:
            raise KeyError(run_group_id)
        detail = self._experiment_engine().get_run_group_detail(run_group_id)
        preferred = (detail.get("best_trial") or {}).get("trial_id")
        page_limit = max(1, int(limit))
        page_offset = max(0, int(offset))
        db = self._dataset_database()
        rows = db.list_cohort_analysis_snapshots(
            run_group_id=run_group_id, limit=page_offset + page_limit
        )
        with db._lock:
            total_row = db._conn.execute(
                "SELECT COUNT(*) AS value FROM cohort_analysis_snapshots " "WHERE run_group_id = ?",
                (run_group_id,),
            ).fetchone()
        page = rows[page_offset : page_offset + page_limit]
        total = int(total_row["value"] if total_row else len(rows))
        return {
            "items": [self._analysis_view(value, preferred_subject_id=preferred) for value in page],
            "total": total,
            "offset": page_offset,
            "limit": page_limit,
            "has_more": page_offset + len(page) < total,
        }

    def create_run_group_analysis(
        self, run_group_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        from halo_forge.adaptive_lab._canonical import content_fingerprint

        detail = self._experiment_engine().get_run_group_detail(run_group_id)
        objective = dict(detail.get("objective") or {})
        metric = str(objective.get("metric") or "")
        direction = str(objective.get("direction") or "maximize")
        objective_boundary = dict(objective.get("boundary") or {})
        if not objective_boundary:
            raise ValueError("no development-evaluated checkpoint boundary is shared by the cohort")
        observations: List[Dict[str, Any]] = []
        suite_revision = self._dataset_database().get_benchmark_suite_revision(
            str(objective.get("suite_revision_id") or "")
        )
        if suite_revision is None:
            raise ValueError("run group development suite revision is missing")
        generation_hash = content_fingerprint(suite_revision.generation_settings)
        evaluator_hash = content_fingerprint(suite_revision.evaluator_versions)
        compatibility: List[Dict[str, Any]] = []
        for trial in detail.get("trials") or []:
            resolved_config = dict(trial.get("resolved_config") or {})
            compatibility.append(
                {
                    "subject_id": trial["id"],
                    "suite_revision_id": suite_revision.id,
                    "generation_settings_hash": generation_hash,
                    "template_hash": self._optional_str(resolved_config.get("chat_template_hash")),
                    "evaluator_versions_hash": evaluator_hash,
                    "metadata": {
                        "trainer_mode": detail.get("trainer_mode"),
                        "backend": resolved_config.get("backend"),
                        "objective_boundary": objective_boundary,
                        "checkpoint_plan_hash": (
                            (detail.get("resolved_checkpoint_plan") or {}).get("content_hash")
                        ),
                    },
                }
            )
            for run in trial.get("runs") or []:
                value = run.get("objective_value")
                if value is None:
                    continue
                if dict(run.get("objective_boundary") or {}) != objective_boundary:
                    raise ValueError("cohort observations do not share one checkpoint boundary")
                segment = next(
                    (
                        item
                        for item in run.get("segments") or []
                        if int(item.get("ordinal") or 0) == int(objective_boundary["ordinal"])
                    ),
                    None,
                )
                if segment is None:
                    raise ValueError("cohort observation is missing its checkpoint segment")
                segment_id = str(segment.get("id") or "")
                related_work = list(run.get("work_items") or [])
                evaluation_work = next(
                    (
                        item
                        for item in reversed(related_work)
                        if item.get("kind") == "evaluation"
                        and item.get("status") == "completed"
                        and item.get("launch_spec", {}).get("suite_revision_id")
                        == suite_revision.id
                        and item.get("launch_spec", {}).get("trial_segment_id") == segment_id
                        and isinstance(item.get("result"), dict)
                    ),
                    {},
                )
                evaluation_result = dict(evaluation_work.get("result") or {})
                evaluation_id = evaluation_result.get("evaluation_id")
                if not evaluation_id:
                    persistent = next(
                        (
                            item
                            for item in self._dataset_database().list_evaluations(
                                suite_revision_id=suite_revision.id,
                                subject_ref=str(run.get("run_id") or ""),
                                status="completed",
                                limit=100,
                            )
                            if str(item.request.get("trial_segment_id") or "") == segment_id
                        ),
                        None,
                    )
                    evaluation_id = persistent.id if persistent is not None else None
                metadata: Dict[str, Any] = {}
                result_metrics = evaluation_result.get("metrics")
                if isinstance(result_metrics, dict):
                    for name, raw_value in result_metrics.items():
                        candidate = (
                            raw_value.get("value") if isinstance(raw_value, dict) else raw_value
                        )
                        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                            metadata[str(name)] = float(candidate)
                training_work = next(
                    (
                        item
                        for item in reversed(related_work)
                        if item.get("kind") == "training"
                        and item.get("launch_spec", {}).get("trial_segment_id") == segment_id
                    ),
                    None,
                )
                if training_work and training_work.get("id"):
                    rollup = self._v4_engine().get_telemetry_rollup(str(training_work["id"]))
                    for name, raw_value in dict(rollup or {}).items():
                        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                            metadata[f"resource_{name}"] = float(raw_value)
                artifact_sizes = [
                    int(artifact.get("size_bytes") or 0)
                    for artifact in detail.get("artifacts") or []
                    if artifact.get("trial_id") == trial.get("id")
                    and artifact.get("run_id") == run.get("run_id")
                    and (
                        not artifact.get("trial_segment_id")
                        or artifact.get("trial_segment_id") == segment_id
                    )
                ]
                if artifact_sizes:
                    metadata["artifact_size_bytes"] = float(max(artifact_sizes))
                metadata.update(
                    objective_boundary_index=int(objective_boundary["ordinal"]),
                    objective_boundary_value=int(objective_boundary["value"]),
                    objective_boundary_unit=str(objective_boundary["unit"]),
                )
                observations.append(
                    {
                        "subject_id": trial["id"],
                        "seed": int(run["seed"]),
                        "metric": metric,
                        "value": float(value),
                        "evaluation_id": evaluation_id,
                        "metadata": metadata,
                    }
                )
        if not observations:
            raise ValueError("no completed seed-level development evaluations are available")
        trial_ids = [str(value.get("id")) for value in detail.get("trials") or []]
        baseline = self._optional_str(payload.get("baseline_subject_id"))
        if baseline is None and len(trial_ids) > 1:
            baseline = trial_ids[0]
        if baseline is not None and baseline not in trial_ids:
            raise ValueError(f"unknown baseline trial: {baseline}")
        confidence = float(payload.get("confidence_level", payload.get("confidence", 0.95)))
        if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be a finite number between zero and one")
        bootstrap_resamples = int(payload.get("bootstrap_resamples", 10_000))
        if bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")
        practical_delta = float(payload.get("practical_delta", 0.0) or 0.0)
        if not math.isfinite(practical_delta) or practical_delta < 0.0:
            raise ValueError("practical_delta must be a finite non-negative number")
        equivalence_delta = (
            None
            if payload.get("equivalence_delta") is None
            else float(payload["equivalence_delta"])
        )
        if equivalence_delta is not None and (
            not math.isfinite(equivalence_delta) or equivalence_delta < 0.0
        ):
            raise ValueError("equivalence_delta must be a finite non-negative number")
        record = self._adaptive_engine().analyze_cohort(
            observations,
            metric=metric,
            direction=direction,
            baseline_subject_id=baseline,
            run_group_id=run_group_id,
            confidence_level=confidence,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=int(payload.get("bootstrap_seed", 42)),
            practical_delta=practical_delta,
            equivalence_delta=equivalence_delta,
            required_seeds=tuple(int(value) for value in detail.get("seeds") or ()),
            evidence_compatibility=compatibility,
            context={
                "objective_boundary": objective_boundary,
                "checkpoint_policy_revision_id": detail.get("checkpoint_policy_revision_id"),
                "checkpoint_plan_hash": (
                    (detail.get("resolved_checkpoint_plan") or {}).get("content_hash")
                ),
            },
        )
        preferred = (detail.get("best_trial") or {}).get("trial_id")
        return self._analysis_view(record, preferred_subject_id=preferred)

    def review_checkpoint_gate(self, decision_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._checkpoint_gate_review_lock:
            db = self._dataset_database()
            original = db.get_checkpoint_gate_decision(decision_id)
            if original is None:
                raise KeyError(decision_id)
            if original.override_of_id:
                raise ValueError("checkpoint gate overrides cannot be reviewed again")
            action = str(payload.get("action") or "").strip().lower()
            reason = str(payload.get("reason") or "").strip()
            overrides = [
                value
                for value in db.list_checkpoint_gate_decisions(
                    run_group_id=original.run_group_id, limit=10_000
                )
                if value.override_of_id == original.id
            ]
            if overrides:
                latest = overrides[0]
                if latest.action == action and latest.override_reason == reason:
                    return self._gate_view(latest.to_dict())
                raise ValueError("checkpoint gate has already been reviewed")
            result = self._experiment_engine().review_checkpoint_gate(
                decision_id,
                action=action,
                reason=reason,
            )
            return self._gate_view(dict(result.get("gate_decision") or {}))

    def create_research_decision(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        analysis_id = str(payload.get("analysis_snapshot_id") or "").strip()
        analysis = self._dataset_database().get_cohort_analysis_snapshot(analysis_id)
        if analysis is None:
            raise KeyError(f"unknown cohort analysis snapshot: {analysis_id}")
        if analysis.status != "completed":
            raise ValueError("research decisions require a completed analysis snapshot")
        selected = dict(payload.get("selected_subject") or {})
        subject_id = str(selected.get("trial_id") or selected.get("subject_id") or "")
        subjects = dict(analysis.analysis.get("subjects") or {})
        if subject_id and subject_id not in subjects:
            raise ValueError("selected subject is not present in the analysis snapshot")
        if not subject_id:
            raise ValueError("selected_subject must identify a trial or subject")
        selected_group_id = self._optional_str(selected.get("run_group_id"))
        if (
            selected_group_id is not None
            and analysis.run_group_id is not None
            and selected_group_id != analysis.run_group_id
        ):
            raise ValueError("selected subject belongs to another run group")
        if analysis.run_group_id is not None:
            selected["run_group_id"] = analysis.run_group_id

        rejected = [dict(value) for value in payload.get("rejected_subjects") or []]
        for value in rejected:
            rejected_id = str(value.get("trial_id") or value.get("subject_id") or "")
            if not rejected_id or rejected_id not in subjects:
                raise ValueError("rejected subjects must be present in the analysis snapshot")
            if rejected_id == subject_id:
                raise ValueError("selected subject cannot also be rejected")
            rejected_group_id = self._optional_str(value.get("run_group_id"))
            if (
                rejected_group_id is not None
                and analysis.run_group_id is not None
                and rejected_group_id != analysis.run_group_id
            ):
                raise ValueError("rejected subject belongs to another run group")
            if analysis.run_group_id is not None:
                value["run_group_id"] = analysis.run_group_id

        fork_spec = dict(payload.get("fork_spec") or {})
        fork_trial_id = self._optional_str(fork_spec.get("trial_id") or fork_spec.get("subject_id"))
        if fork_trial_id is not None and fork_trial_id not in subjects:
            raise ValueError("fork specification references a subject outside the analysis")
        fork_group_id = self._optional_str(fork_spec.get("run_group_id"))
        if (
            fork_group_id is not None
            and analysis.run_group_id is not None
            and fork_group_id != analysis.run_group_id
        ):
            raise ValueError("fork specification belongs to another run group")
        if fork_spec and analysis.run_group_id is not None:
            fork_spec["run_group_id"] = analysis.run_group_id
        comparisons = dict(analysis.analysis.get("comparisons") or {})
        baseline_subject_id = self._optional_str(analysis.baseline_subject_id)
        selected_comparison = comparisons.get(subject_id)
        if selected_comparison is not None:
            insufficient = selected_comparison.get("classification") == "insufficient_evidence"
        elif subject_id == baseline_subject_id:
            insufficient = not comparisons or all(
                value.get("classification") == "insufficient_evidence"
                for value in comparisons.values()
            )
        else:
            # A snapshot without a baseline can summarize variance, but it
            # cannot provide comparative support for selecting a subject.
            insufficient = True
        if insufficient and not str(payload.get("override_reason") or "").strip():
            raise ValueError("insufficient evidence requires an explicit override_reason")
        record = self._adaptive_engine().create_research_decision(
            analysis_snapshot_id=analysis_id,
            selected_subject=selected,
            rejected_subjects=rejected,
            exclusions=list(payload.get("exclusions") or []),
            rationale=str(payload.get("rationale") or ""),
            fork_spec=fork_spec,
            override_reason=self._optional_str(payload.get("override_reason")),
        )
        value = record.to_dict()
        value["run_group_id"] = analysis.run_group_id
        return value

    def get_research_decision(self, decision_id: str) -> Optional[Dict[str, Any]]:
        record = self._dataset_database().get_research_decision(decision_id)
        if record is None:
            return None
        value = record.to_dict()
        analysis = self._dataset_database().get_cohort_analysis_snapshot(
            record.analysis_snapshot_id
        )
        value["run_group_id"] = analysis.run_group_id if analysis else None
        return value

    def list_research_decisions(
        self, *, run_group_id: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        rows = self._dataset_database().list_research_decisions(limit=max(limit, 1_000))
        items = []
        for row in rows:
            analysis = self._dataset_database().get_cohort_analysis_snapshot(
                row.analysis_snapshot_id
            )
            if run_group_id and (analysis is None or analysis.run_group_id != run_group_id):
                continue
            value = row.to_dict()
            value["run_group_id"] = analysis.run_group_id if analysis else None
            items.append(value)
            if len(items) >= max(1, int(limit)):
                break
        return {"items": items, "total": len(items), "offset": 0, "limit": limit}

    def create_evidence_bundle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import platform

        analysis_id = str(payload.get("analysis_snapshot_id") or "").strip()
        analysis_record = self._dataset_database().get_cohort_analysis_snapshot(analysis_id)
        if analysis_record is None:
            raise KeyError(f"unknown cohort analysis snapshot: {analysis_id}")
        request = {
            key: value
            for key, value in payload.items()
            if key not in {"analysis_snapshot_id", "research_decision_id"}
        }
        if analysis_record.run_group_id:
            detail = self._experiment_engine().get_run_group_detail(analysis_record.run_group_id)
            dataset_versions = []
            for binding in detail.get("dataset_bindings") or []:
                version = self._dataset_database().get_dataset_version(
                    str(binding.get("dataset_version_id") or "")
                )
                dataset_versions.append(
                    {
                        "binding": dict(binding),
                        "identity": version.to_dict() if version is not None else None,
                    }
                )
            request.setdefault(
                "run_group",
                {
                    "id": detail.get("id"),
                    "name": detail.get("name"),
                    "trainer_mode": detail.get("trainer_mode"),
                    "objective": detail.get("objective"),
                    "resolved_launch_config": detail.get("resolved_launch_config"),
                    "dataset_bindings": detail.get("dataset_bindings"),
                    "checkpoint_policy_revision_id": detail.get("checkpoint_policy_revision_id"),
                    "resolved_checkpoint_plan": detail.get("resolved_checkpoint_plan"),
                    "seeds": detail.get("seeds"),
                    "trials": [
                        {
                            "id": trial.get("id"),
                            "ordinal": trial.get("ordinal"),
                            "sampled_config": trial.get("sampled_config"),
                            "cohort": trial.get("cohort"),
                            "runs": [
                                {
                                    "run_id": run.get("run_id"),
                                    "seed": run.get("seed"),
                                    "objective_value": run.get("objective_value"),
                                    "segments": run.get("segments"),
                                }
                                for run in trial.get("runs") or []
                            ],
                        }
                        for trial in detail.get("trials") or []
                    ],
                    "artifacts": detail.get("artifacts") or [],
                },
            )
            request.setdefault("dataset_versions", dataset_versions)
            run_identities = []
            for trial in detail.get("trials") or []:
                for run in trial.get("runs") or []:
                    run_id = str(run.get("run_id") or "")
                    if not run_id:
                        continue
                    resolved_launch = None
                    try:
                        resolved_launch = self.get_resolved_run_launch_config(run_id)
                    except (KeyError, OSError, ValueError):
                        pass
                    indexed = self._dataset_database().get_run(run_id)
                    replay = None
                    output_dir = (
                        Path(indexed.output_dir).expanduser()
                        if indexed is not None and indexed.output_dir
                        else None
                    )
                    replay_path = output_dir / "replay.json" if output_dir else None
                    if replay_path is not None and replay_path.is_file():
                        try:
                            replay = json.loads(replay_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            replay = {"path": str(replay_path), "status": "unreadable"}
                    run_identities.append(
                        {
                            "run_id": run_id,
                            "trial_id": trial.get("id"),
                            "seed": run.get("seed"),
                            "resolved_launch": resolved_launch,
                            "indexed_run": indexed.to_dict() if indexed else None,
                            "replay": replay,
                        }
                    )
            request.setdefault("run_identities", run_identities)
            suite_id = str((detail.get("objective") or {}).get("suite_revision_id") or "")
            suite_revision = self._dataset_database().get_benchmark_suite_revision(suite_id)
            if suite_revision is not None:
                request.setdefault("suite_revision", suite_revision.to_dict())
            request.setdefault(
                "checkpoint_trajectory",
                self.get_run_group_trajectory(analysis_record.run_group_id),
            )
        comparisons = dict(analysis_record.analysis.get("comparisons") or {})
        subjects = dict(analysis_record.analysis.get("subjects") or {})
        request.setdefault(
            "missing_evidence",
            {
                "comparison_reasons": {
                    key: value.get("reason")
                    for key, value in comparisons.items()
                    if value.get("reason")
                },
                "missing_seeds": {
                    key: value.get("missing_seeds")
                    for key, value in subjects.items()
                    if value.get("missing_seeds")
                },
            },
        )
        request.setdefault(
            "analysis_assumptions",
            {
                "replicate_unit": "seed",
                "interval": "percentile_bootstrap",
                "confidence_level": analysis_record.request.get("confidence_level"),
                "bootstrap_resamples": analysis_record.request.get("bootstrap_resamples"),
                "bootstrap_seed": analysis_record.request.get("bootstrap_seed"),
                "per_example_deltas_are_diagnostic": True,
            },
        )
        request.setdefault(
            "runtime_context",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor() or None,
                "accelerator_backend": self._active_backend_name(),
                "halo_forge_version": self.get_version_info().get("package_version"),
            },
        )
        bundle = self._adaptive_engine().queue_evidence_bundle(
            analysis_snapshot_id=analysis_id,
            research_decision_id=self._optional_str(payload.get("research_decision_id")),
            request=request,
        )
        if bundle.status == "completed" or bundle.work_item_id:
            return bundle.to_dict()
        work = self._scheduler().enqueue(
            kind="adaptive_evidence_bundle",
            launch_spec={
                "handler": "adaptive_lab.execute_work_item",
                "action": "build_evidence_bundle",
                "evidence_bundle_id": bundle.id,
                "name": "Build research evidence bundle",
            },
            resource_class="cpu",
            domain_kind="evidence_bundle",
            domain_id=bundle.id,
            run_group_id=analysis_record.run_group_id,
            max_retries=1,
            work_item_id=f"adaptive-evidence-{bundle.id}",
        )
        updated = self._dataset_database().update_evidence_bundle(bundle.id, work_item_id=work.id)
        assert updated is not None
        return updated.to_dict()

    def get_evidence_bundle(self, bundle_id: str) -> Optional[Dict[str, Any]]:
        record = self._dataset_database().get_evidence_bundle(bundle_id)
        return record.to_dict() if record is not None else None

    def list_evidence_bundles(
        self, *, run_group_id: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        rows = self._dataset_database().list_evidence_bundles(limit=max(limit, 1_000))
        items = []
        for row in rows:
            analysis = self._dataset_database().get_cohort_analysis_snapshot(
                row.analysis_snapshot_id
            )
            if run_group_id and (analysis is None or analysis.run_group_id != run_group_id):
                continue
            items.append(row.to_dict())
            if len(items) >= max(1, int(limit)):
                break
        return {"items": items, "total": len(items), "offset": 0, "limit": limit}

    def get_workspace_draft(self, surface: str, draft_key: str) -> Optional[Dict[str, Any]]:
        value = self._dataset_database().get_workspace_draft_by_key(
            draft_kind=str(surface), owner_key="local", name=str(draft_key)
        )
        if value is not None:
            try:
                expired = datetime.fromisoformat(value.expires_at) <= datetime.now(timezone.utc)
            except (TypeError, ValueError):
                expired = True
            if expired:
                self._adaptive_engine().discard_workspace_draft(value.id)
                return None
        return value.to_dict() if value is not None else None

    def save_workspace_draft(
        self, surface: str, draft_key: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        content = payload.get("content")
        if not isinstance(content, dict):
            raise ValueError("workspace draft content must be an object")
        record = self._adaptive_engine().save_workspace_draft(
            draft_kind=str(surface),
            owner_key="local",
            name=str(draft_key),
            content=content,
            ttl_days=int(payload.get("ttl_days", 30)),
        )
        value = record.to_dict()
        value["display_name"] = str(payload.get("name") or draft_key)
        return value

    def delete_workspace_draft(self, surface: str, draft_key: str) -> bool:
        record = self._dataset_database().get_workspace_draft_by_key(
            draft_kind=str(surface), owner_key="local", name=str(draft_key)
        )
        return bool(record and self._adaptive_engine().discard_workspace_draft(record.id))

    def list_trainer_execution_capabilities(self) -> Dict[str, Any]:
        engine = self._experiment_engine()
        return {"items": [value.to_dict() for value in engine.capabilities.list()]}

    def _work_item_view(self, item: Any) -> Dict[str, Any]:
        db = self._dataset_database()
        value = item.to_dict()
        launch_spec = dict(value.pop("launch_spec", {}) or {})
        progress = dict(value.get("progress") or {})
        return {
            **value,
            "payload": launch_spec,
            "run_group_id": launch_spec.get("run_group_id"),
            "trial_id": launch_spec.get("trial_id"),
            "trial_run_id": launch_spec.get("trial_run_id"),
            "segment_id": launch_spec.get("segment_id"),
            "run_id": item.canonical_run_id,
            "attempt": int(item.retry_count) + 1,
            "max_attempts": int(item.max_retries) + 1,
            "progress_current": progress.get("current", progress.get("processed")),
            "progress_total": progress.get("total"),
            "queue_position": db.work_item_queue_position(item.id),
            "blockers": db.work_item_blockers(item.id),
        }

    def list_work_items(
        self,
        *,
        status: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        db = self._dataset_database()
        items = db.list_work_items(
            statuses=[status] if status else None,
            kinds=[kind] if kind else None,
            limit=limit,
            offset=offset,
        )
        lease = db.get_resource_lease("accelerator")
        return {
            "items": [self._work_item_view(item) for item in items],
            "active_lease": lease.to_dict() if lease is not None else None,
            "offset": max(0, int(offset)),
            "limit": max(1, int(limit)),
        }

    def get_work_item(self, work_item_id: str) -> Optional[Dict[str, Any]]:
        item = self._dataset_database().get_work_item(work_item_id)
        return self._work_item_view(item) if item is not None else None

    def cancel_work_item(self, work_item_id: str) -> Dict[str, Any]:
        from halo_forge.workstation_jobs import WorkstationScheduler

        item = WorkstationScheduler(self._dataset_database()).cancel(work_item_id)
        if item is None:
            raise KeyError(work_item_id)
        self._sync_future_lab_domain_from_work_item(item)
        return self._work_item_view(item)

    def retry_work_item(self, work_item_id: str, *, reason: Optional[str] = None) -> Dict[str, Any]:
        from halo_forge.workstation_jobs import WorkstationScheduler

        item = WorkstationScheduler(self._dataset_database()).retry(
            work_item_id, reason=reason or "operator requested retry"
        )
        if item is None:
            raise KeyError(work_item_id)
        self._sync_future_lab_domain_from_work_item(item)
        return self._work_item_view(item)

    def _sync_future_lab_domain_from_work_item(self, item: Any) -> None:
        if str(item.launch_spec.get("handler") or "") != "future_lab.execute_work_item":
            return
        status = str(item.status)
        domain_kind = str(item.domain_kind or "")
        domain_id = str(item.domain_id or "")
        if not domain_id:
            return
        db = self._dataset_database()._conn
        table = {
            "training_outcome_assessment": "training_outcome_assessments",
            "adaptation_study_analysis": "adaptation_study_analyses",
            "grounded_generation_batch": "grounded_generation_batches",
            "agent_episode": "agent_episodes",
            "trajectory_set": "trajectory_sets",
        }.get(domain_kind)
        if table:
            domain_status = "queued" if status == "queued" else status
            stage = "waiting" if status == "queued" else status
            db.execute(
                f"""UPDATE {table}
                    SET status=?,stage=?,cancel_requested=0,
                        error=CASE WHEN ?='queued' THEN NULL ELSE error END
                    WHERE id=?""",
                (domain_status, stage, status, domain_id),
            )
        elif domain_kind == "adaptation_study_protocol_revision":
            db.execute(
                """UPDATE adaptation_study_protocol_revisions
                   SET launch_status=?,
                       launch_error=CASE WHEN ?='queued' THEN NULL ELSE launch_error END
                   WHERE id=?""",
                ("queued" if status == "queued" else status, status, domain_id),
            )
        db.commit()

    def list_workers(self, *, limit: int = 100) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        items = []
        for worker in self._v4_engine().list_workers(limit=limit):
            value = worker.to_dict()
            try:
                age = max(
                    0.0,
                    (now - datetime.fromisoformat(worker.heartbeat_at)).total_seconds(),
                )
            except (TypeError, ValueError):
                age = None
            value.update(
                heartbeat_age_seconds=age,
                healthy=worker.status == "online" and age is not None and age < 90,
            )
            items.append(value)
        return {"items": items, "total": len(items)}

    def storage_status(self) -> Dict[str, Any]:
        inventory = self._artifact_library().inventory()
        value = inventory.to_dict()
        reserve = max(20 * 1024**3, int(value["total_bytes"] * 0.10))
        db = self._dataset_database()
        location_count_row = db._conn.execute(
            "SELECT COUNT(*) AS value FROM artifact_locations WHERE storage_mode = 'managed'"
        ).fetchone()
        location_rows = db._conn.execute(
            "SELECT * FROM artifact_locations WHERE storage_mode = 'managed' "
            "ORDER BY created_at DESC LIMIT 250"
        ).fetchall()
        managed_locations = [
            self._artifact_location_view(
                {
                    **dict(row),
                    "metadata": json.loads(row["metadata_json"] or "{}"),
                }
            )
            for row in location_rows
        ]
        managed_location_count = int(
            location_count_row["value"] if location_count_row else len(managed_locations)
        )
        now = datetime.now(timezone.utc)
        import_staging_items = []
        import_staging_bytes = 0
        imports_root = Path(self.dataset_import_root).expanduser().resolve()
        for dataset_import in db.list_dataset_imports(limit=10_000):
            if dataset_import.status in {"published", "expired"} or not dataset_import.staging_path:
                continue
            staging = Path(dataset_import.staging_path).expanduser().resolve()
            try:
                staging.relative_to(imports_root)
            except ValueError:
                continue
            size = 0
            if staging.exists():
                paths = [staging] if staging.is_file() else staging.rglob("*")
                size = sum(
                    item.stat().st_size
                    for item in paths
                    if item.is_file() and not item.is_symlink()
                )
            import_staging_bytes += size
            try:
                expired = bool(
                    dataset_import.expires_at
                    and datetime.fromisoformat(dataset_import.expires_at) <= now
                )
            except ValueError:
                expired = False
            if len(import_staging_items) < 250:
                import_staging_items.append(
                    {
                        "id": dataset_import.id,
                        "name": dataset_import.display_name or "Dataset import",
                        "status": dataset_import.status,
                        "size_bytes": size,
                        "expires_at": dataset_import.expires_at,
                        "cleanup_eligible": expired,
                        "resource_type": "dataset_import_staging",
                    }
                )
        value.update(
            required_reserve_bytes=reserve,
            minimum_free_bytes=reserve,
            projected_free_bytes=value["free_bytes"],
            low_disk=value["free_bytes"] < reserve,
            forecast_state=("low" if value["free_bytes"] < reserve else "healthy"),
            artifact_bytes=value.get("managed_bytes", 0),
            temporary_bytes=value.get("staging_bytes", 0) + import_staging_bytes,
            import_staging_bytes=import_staging_bytes,
            import_staging_items=import_staging_items,
            managed_locations=managed_locations,
            managed_location_count=managed_location_count,
            managed_locations_truncated=managed_location_count > len(managed_locations),
            cache_items=[],
            forecast={
                "state": "low" if value["free_bytes"] < reserve else "healthy",
                "required_reserve_bytes": reserve,
                "projected_free_bytes": value["free_bytes"],
            },
        )
        return value

    @staticmethod
    def _cleanup_plan_view(plan: Any, *, status: str = "preview") -> Dict[str, Any]:
        value = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
        candidates = list(value.get("candidates") or value.get("items") or [])
        return {
            **value,
            "status": status,
            "items": [
                {
                    "id": item.get("identifier") or item.get("id"),
                    "path": item.get("path"),
                    "size_bytes": item.get("reclaimable_bytes", item.get("size_bytes", 0)),
                    "reason": item.get("resource_type") or item.get("reason"),
                    "protected": False,
                }
                for item in candidates
            ],
            "trash_retention_days": 7,
        }

    def storage_cleanup(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Preview or apply one reviewed, reversible Artifact Studio cleanup."""

        studio = self._artifact_studio_engine()
        if studio is None:
            raise RuntimeError("Artifact Studio is unavailable")
        db = self._dataset_database()
        if bool(payload.get("preview", False)) or not payload.get("plan_id"):
            from halo_forge.artifact_lab import CleanupProtections

            minimum_age_days = max(0.0, float(payload.get("older_than_days", 7)))
            protected_staging = frozenset()
            if not bool(payload.get("include_temporary", False)):
                protected_staging = frozenset(
                    path.name for path in studio.store.staging_dir.iterdir() if path.exists()
                )
            extra = CleanupProtections(active_staging=protected_staging)
            plan = studio.store.preview_cleanup(
                protections=studio.cleanup_protections(extra),
                minimum_blob_age=timedelta(days=minimum_age_days),
            )
            request = {
                key: payload[key]
                for key in ("include_temporary", "include_trash", "older_than_days")
                if key in payload
            }
            value = self._cleanup_plan_view(plan)
            with db._lock:
                db._conn.execute(
                    "INSERT OR REPLACE INTO cleanup_plans "
                    "(id, status, request_json, entries_json, reclaimed_bytes, created_at) "
                    "VALUES (?, 'preview', ?, ?, 0, ?)",
                    (
                        plan.id,
                        json.dumps(request, sort_keys=True),
                        json.dumps(value["items"], sort_keys=True),
                        plan.created_at,
                    ),
                )
                db._conn.commit()
            return value

        plan_id = str(payload.get("plan_id") or "").strip()
        if not bool(payload.get("approved", False)):
            raise ValueError("cleanup execution requires approved=true")
        row = db._conn.execute("SELECT * FROM cleanup_plans WHERE id = ?", (plan_id,)).fetchone()
        if row is None:
            raise KeyError(plan_id)
        if str(row["status"]) in {"approved", "queued", "running", "completed", "failed"}:
            work = db.get_work_item(str(row["work_item_id"])) if row["work_item_id"] else None
            work_status = str(work.status if work is not None else row["status"])
            visible_status = "completed" if work_status == "completed" else work_status
            if visible_status != str(row["status"]):
                with db._lock:
                    db._conn.execute(
                        "UPDATE cleanup_plans SET status = ?, applied_at = CASE "
                        "WHEN ? = 'completed' THEN ? ELSE applied_at END WHERE id = ?",
                        (
                            visible_status,
                            visible_status,
                            datetime.now(timezone.utc).isoformat(),
                            plan_id,
                        ),
                    )
                    db._conn.commit()
            return {
                "id": plan_id,
                "status": visible_status,
                "items": json.loads(row["entries_json"] or "[]"),
                "reclaimable_bytes": sum(
                    int(item.get("size_bytes") or 0)
                    for item in json.loads(row["entries_json"] or "[]")
                ),
                "work_item_id": row["work_item_id"],
                "created_at": row["created_at"],
                "trash_retention_days": 7,
                "reused": True,
            }
        review_note = str(payload.get("review_note") or "").strip()
        if not review_note:
            raise ValueError("cleanup execution requires a non-empty review_note")
        receipt = studio.queue_cleanup(
            plan_id,
            review_note=review_note,
            priority=int(payload.get("priority", 0)),
            max_retries=int(payload.get("max_retries", 1)),
            capacity_override_reason=self._optional_str(payload.get("capacity_override_reason")),
        )
        reviewed_at = datetime.now(timezone.utc).isoformat()
        with db._lock:
            db._conn.execute(
                "UPDATE cleanup_plans SET status = ?, reviewed_at = ?, "
                "work_item_id = ? WHERE id = ?",
                (receipt.status, reviewed_at, receipt.work_item_id, plan_id),
            )
            db._conn.commit()
        return {
            "id": plan_id,
            "status": receipt.status,
            "items": json.loads(row["entries_json"] or "[]"),
            "reclaimable_bytes": sum(
                int(item.get("size_bytes") or 0) for item in json.loads(row["entries_json"] or "[]")
            ),
            "reclaimed_bytes": 0,
            "work_item_id": receipt.work_item_id,
            "domain_id": receipt.domain_id,
            "reused": receipt.reused,
            "created_at": row["created_at"],
            "trash_retention_days": 7,
        }

    def get_activity(self, *, after_sequence: int = 0, limit: int = 200) -> Dict[str, Any]:
        from halo_forge.public_api.activity import activity_item_view, normalize_work_event

        db = self._dataset_database()
        work_items = db.list_work_items(limit=limit, offset=0)
        events = self._v4_engine().list_events(after_sequence=after_sequence, limit=max(limit, 500))
        workers = self.list_workers(limit=10)["items"]
        lease = db.get_resource_lease("accelerator")
        items = []
        for item in work_items:
            view = activity_item_view(db, self._v4_engine(), item)
            view["telemetry"] = (
                self._v4_engine().list_telemetry(item.id, limit=120)
                if item.status == "running"
                else []
            )
            items.append(view)
        gate_rows = db.list_checkpoint_gate_decisions(limit=max(limit, 500))
        overridden_ids = {value.override_of_id for value in gate_rows if value.override_of_id}
        for gate in gate_rows:
            if gate.action != "pause" or gate.id in overridden_ids:
                continue
            boundary = dict(gate.evidence.get("boundary") or {})
            reasons = list(gate.reasons)
            items.append(
                {
                    "id": f"gate-review:{gate.id}",
                    "work_item_id": None,
                    "domain_id": gate.id,
                    "domain_type": "gate_decision",
                    "kind": "checkpoint_gate_review",
                    "title": (
                        f"Review checkpoint {boundary.get('value', gate.boundary_index + 1)} "
                        f"{boundary.get('unit', '')}"
                    ).strip(),
                    "status": "awaiting_review",
                    "stage": "operator decision",
                    "priority": 0,
                    "progress_current": None,
                    "progress_total": None,
                    "progress_percent": None,
                    "queue_position": None,
                    "eta_seconds": None,
                    "blockers": reasons,
                    "resource_requirements": {},
                    "worker_id": None,
                    "attempt": 1,
                    "max_attempts": 1,
                    "attempts": [],
                    "events": [],
                    "logs": [],
                    "error": None,
                    "created_at": gate.created_at,
                    "started_at": None,
                    "completed_at": None,
                    "heartbeat_at": None,
                    "telemetry_rollup": {},
                    "next_actions": ["inspect", "continue", "stop"],
                    "run_group_id": gate.run_group_id,
                }
            )
        items.sort(key=lambda value: str(value.get("created_at") or ""), reverse=True)
        return {
            "items": items,
            "events": [normalize_work_event(value) for value in events],
            "latest_sequence": events[-1].sequence if events else after_sequence,
            "worker": workers[0] if workers else None,
            "workers": workers,
            "resource_lease": lease.to_dict() if lease is not None else None,
            "storage": self.storage_status(),
        }

    def global_search(
        self,
        query: str,
        *,
        types: Optional[List[str]] = None,
        limit: int = 30,
    ) -> Dict[str, Any]:
        """Search the local research catalog without exposing raw identifiers as labels."""

        text_query = str(query or "").strip()
        if not text_query:
            raise ValueError("search query is required")
        requested = {
            str(value).strip().lower().replace("-", "_") for value in (types or []) if value
        }
        aliases = {
            "dataset_version": "dataset_version",
            "version": "dataset_version",
            "dataset": "dataset",
            "run": "run",
            "suite": "suite",
            "benchmark_suite": "suite",
            "run_group": "run_group",
            "experiment": "run_group",
            "artifact": "artifact",
            "model": "artifact",
            "checkpoint_policy": "checkpoint_policy",
            "policy": "checkpoint_policy",
            "activity": "activity",
            "work_item": "activity",
        }
        selected = {aliases.get(value, value) for value in requested}
        allowed = {
            "dataset",
            "dataset_version",
            "run",
            "suite",
            "run_group",
            "artifact",
            "checkpoint_policy",
            "activity",
        }
        if selected:
            selected &= allowed
        else:
            selected = set(allowed)
        escaped = text_query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        per_kind = max(5, min(50, int(limit)))
        db = self._dataset_database()
        results: List[Dict[str, Any]] = []

        def append(
            *,
            kind: str,
            identifier: str,
            label: str,
            description: str,
            status: Optional[str],
            target: str,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> None:
            label_text = str(label or identifier)
            lower_label = label_text.lower()
            lower_query = text_query.lower()
            score = (
                3 if lower_label == lower_query else 2 if lower_label.startswith(lower_query) else 1
            )
            results.append(
                {
                    "id": str(identifier),
                    "type": kind,
                    "label": label_text,
                    "description": str(description or ""),
                    "status": status,
                    "short_id": str(identifier)[:12],
                    "target": target,
                    "metadata": metadata or {},
                    "score": score,
                }
            )

        with db._lock:
            if "dataset" in selected:
                rows = db._conn.execute(
                    "SELECT id, name, modality, canonical_schema, latest_version_id "
                    "FROM datasets WHERE name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (pattern, pattern, per_kind),
                ).fetchall()
                for row in rows:
                    append(
                        kind="dataset",
                        identifier=row["id"],
                        label=row["name"],
                        description=f"{row['modality']} · {row['canonical_schema']}",
                        status="ready" if row["latest_version_id"] else "source",
                        target=f"/datasets/{row['id']}",
                    )
            if "dataset_version" in selected:
                rows = db._conn.execute(
                    "SELECT v.id, v.dataset_id, v.status, v.content_hash, d.name "
                    "FROM dataset_versions v JOIN datasets d ON d.id = v.dataset_id "
                    "WHERE d.name LIKE ? ESCAPE '\\' OR v.id LIKE ? ESCAPE '\\' "
                    "OR v.content_hash LIKE ? ESCAPE '\\' "
                    "ORDER BY v.created_at DESC LIMIT ?",
                    (pattern, pattern, pattern, per_kind),
                ).fetchall()
                for row in rows:
                    append(
                        kind="dataset_version",
                        identifier=row["id"],
                        label=f"{row['name']} · {str(row['id'])[:8]}",
                        description=f"Dataset version · {row['status']}",
                        status=row["status"],
                        target=f"/datasets/{row['dataset_id']}/versions/{row['id']}",
                        metadata={"content_hash": row["content_hash"]},
                    )
            if "run" in selected:
                rows = db._conn.execute(
                    "SELECT run_id, model_name, modality, status FROM runs "
                    "WHERE run_id LIKE ? ESCAPE '\\' OR model_name LIKE ? ESCAPE '\\' "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (pattern, pattern, per_kind),
                ).fetchall()
                for row in rows:
                    append(
                        kind="run",
                        identifier=row["run_id"],
                        label=row["model_name"] or str(row["run_id"])[:12],
                        description=f"{row['modality']} run · {str(row['run_id'])[:12]}",
                        status=row["status"],
                        target=f"/runs/{row['run_id']}",
                    )
            if "suite" in selected:
                rows = db._conn.execute(
                    "SELECT id, name, purpose, latest_revision_id FROM benchmark_suites "
                    "WHERE name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (pattern, pattern, per_kind),
                ).fetchall()
                for row in rows:
                    append(
                        kind="suite",
                        identifier=row["id"],
                        label=row["name"],
                        description=f"{row['purpose'] or 'unspecified'} benchmark suite",
                        status="ready" if row["latest_revision_id"] else "draft",
                        target=f"/eval?suite={row['id']}",
                    )
            if "run_group" in selected:
                rows = db._conn.execute(
                    "SELECT id, name, kind, status, trainer_mode FROM run_groups "
                    "WHERE name LIKE ? ESCAPE '\\' OR id LIKE ? ESCAPE '\\' "
                    "ORDER BY created_at DESC LIMIT ?",
                    (pattern, pattern, per_kind),
                ).fetchall()
                for row in rows:
                    append(
                        kind="run_group",
                        identifier=row["id"],
                        label=row["name"],
                        description=f"{row['kind']} · {row['trainer_mode']}",
                        status=row["status"],
                        target="/sweeps",
                    )
            if "artifact" in selected:
                rows = db._conn.execute(
                    "SELECT id, model_id, artifact_kind, backend, tags_json FROM artifact_occurrences "
                    "WHERE model_id LIKE ? ESCAPE '\\' OR id LIKE ? ESCAPE '\\' "
                    "OR tags_json LIKE ? ESCAPE '\\' ORDER BY created_at DESC LIMIT ?",
                    (pattern, pattern, pattern, per_kind),
                ).fetchall()
                for row in rows:
                    append(
                        kind="artifact",
                        identifier=row["id"],
                        label=row["model_id"],
                        description=f"{str(row['artifact_kind']).replace('_', ' ')} · {row['backend']}",
                        status="available",
                        target=f"/models?tab=artifacts&artifact={row['id']}",
                    )
            if "checkpoint_policy" in selected:
                rows = db._conn.execute(
                    "SELECT id, name, latest_revision_id, archived FROM checkpoint_policies "
                    "WHERE name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\' "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (pattern, pattern, per_kind),
                ).fetchall()
                for row in rows:
                    append(
                        kind="checkpoint_policy",
                        identifier=row["id"],
                        label=row["name"],
                        description="Immutable checkpoint and development-evidence policy",
                        status="archived" if row["archived"] else "ready",
                        target="/sweeps",
                        metadata={"latest_revision_id": row["latest_revision_id"]},
                    )
            if "activity" in selected:
                rows = db._conn.execute(
                    "SELECT id, kind, status, stage, domain_kind, domain_id FROM work_items "
                    "WHERE id LIKE ? ESCAPE '\\' OR kind LIKE ? ESCAPE '\\' "
                    "OR stage LIKE ? ESCAPE '\\' OR domain_id LIKE ? ESCAPE '\\' "
                    "ORDER BY created_at DESC LIMIT ?",
                    (pattern, pattern, pattern, pattern, per_kind),
                ).fetchall()
                for row in rows:
                    append(
                        kind="activity",
                        identifier=row["id"],
                        label=f"{str(row['kind']).replace('_', ' ')} · {str(row['id'])[:8]}",
                        description=str(row["stage"] or row["domain_kind"] or "work item"),
                        status=row["status"],
                        target="/",
                        metadata={
                            "domain_kind": row["domain_kind"],
                            "domain_id": row["domain_id"],
                        },
                    )

        results.sort(key=lambda item: (-int(item["score"]), item["label"].lower(), item["id"]))
        visible = results[: max(1, min(100, int(limit)))]
        return {"items": visible, "total": len(results), "query": text_query}

    async def stream_activity(self, *, after_sequence: int = 0):
        """Stream durable work events with resumable monotonically increasing IDs."""

        from halo_forge.public_api.activity import activity_item_view, normalize_work_event

        sequence = max(0, int(after_sequence))
        yield "retry: 2000\n\n"
        while True:
            events = self._v4_engine().list_events(after_sequence=sequence, limit=500)
            if not events:
                await asyncio.sleep(1.0)
                continue
            for event in events:
                sequence = event.sequence
                payload = normalize_work_event(event)
                item = self._dataset_database().get_work_item(event.work_item_id)
                payload["work_item"] = (
                    activity_item_view(self._dataset_database(), self._v4_engine(), item)
                    if item is not None
                    else None
                )
                yield (
                    f"id: {event.sequence}\n"
                    f"event: {event.event_type}\n"
                    f"data: {json.dumps(payload, default=str)}\n\n"
                )

    def list_artifact_operations(
        self,
        *,
        status: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        db = self._dataset_database()
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if kind:
            clauses.append("operation_type = ?")
            params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total_row = db._conn.execute(
            f"SELECT COUNT(*) AS value FROM artifact_operations {where}", params
        ).fetchone()
        bounded_limit = max(1, min(1000, int(limit)))
        bounded_offset = max(0, int(offset))
        rows = db._conn.execute(
            f"SELECT id FROM artifact_operations {where} "
            "ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            [*params, bounded_limit, bounded_offset],
        ).fetchall()
        items = [
            value
            for row in rows
            if (value := self._v4_engine().get_operation(str(row["id"]))) is not None
        ]
        total = int(total_row["value"] if total_row else len(items))
        return {
            "items": [self._artifact_operation_view(value) for value in items],
            "total": total,
            "offset": bounded_offset,
            "limit": bounded_limit,
            "has_more": bounded_offset + len(items) < total,
        }

    @staticmethod
    def _artifact_operation_view(operation: Any) -> Dict[str, Any]:
        value = operation.to_dict() if hasattr(operation, "to_dict") else dict(operation)
        resolved = dict(value.get("resolved_spec") or {})
        parameters = dict(resolved.get("parameters") or {})
        result = dict(value.get("result") or {})
        return {
            **value,
            "kind": value.get("operation_type") or value.get("kind"),
            "input_artifact_ids": list(
                value.get("input_occurrence_ids") or value.get("input_artifact_ids") or []
            ),
            "output_artifact_id": value.get("output_occurrence_id")
            or value.get("output_artifact_id"),
            "config": parameters,
            "resolved_inputs": list(result.get("resolved_inputs") or []),
            "tool_versions": {
                str(resolved.get("tool_id") or "halo-forge"): str(
                    resolved.get("tool_version") or "unknown"
                )
            },
        }

    def create_artifact_operation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        studio = self._artifact_studio_engine()
        if studio is None:
            raise RuntimeError("Artifact Studio is unavailable")
        kind = str(payload.get("kind") or payload.get("operation_type") or "").lower()
        inputs = [
            str(value)
            for value in (
                payload.get("input_artifact_ids") or payload.get("input_occurrence_ids") or []
            )
            if str(value)
        ]
        config = dict(payload.get("config") or {})
        priority = int(payload.get("priority", config.get("priority", 0)))
        max_retries = int(payload.get("max_retries", config.get("max_retries", 1)))
        if kind in {"merge", "combine", "bake"}:
            mode = "bake" if kind == "bake" else str(config.get("mode") or "combine")
            base_model = str(config.get("base_model") or config.get("model") or "")
            if not base_model and inputs:
                source = self.get_model_artifact(inputs[0])
                base_model = str((source or {}).get("model_name") or "")
            method_aliases = {
                "dare": "dare_linear",
                "magnitude_pruning": "magnitude_prune",
            }
            method = method_aliases.get(
                str(config.get("method") or "dare_ties"),
                str(config.get("method") or "dare_ties"),
            )
            receipt = studio.queue_merge(
                input_occurrence_ids=inputs,
                base_model=base_model,
                mode=mode,
                method=method,
                weights=config.get("weights"),
                bake_after_merge=bool(config.get("bake_after_merge", False)),
                priority=priority,
                max_retries=max_retries,
            )
        elif kind in {"convert", "quantize"}:
            if len(inputs) != 1:
                raise ValueError("conversion requires exactly one input artifact")
            target_aliases = {"huggingface": "hf"}
            target_format = str(config.get("target_format") or config.get("format") or "")
            target_format = target_aliases.get(target_format.lower(), target_format)
            receipt = studio.queue_convert(
                occurrence_id=inputs[0],
                target_format=target_format,
                quantization=str(
                    config.get("quantization")
                    or config.get("precision")
                    or config.get("dtype")
                    or "fp16"
                ),
                priority=priority,
                max_retries=max_retries,
                allow_unquantized_fallback=bool(config.get("allow_unquantized_fallback", False)),
            )
        elif kind == "export":
            if len(inputs) != 1:
                raise ValueError("export requires exactly one input artifact")
            bundle_name = Path(str(config.get("bundle_name") or f"{inputs[0]}-bundle")).name
            if not bundle_name or bundle_name in {".", ".."}:
                raise ValueError("bundle_name must be a safe local name")
            receipt = studio.queue_export(
                occurrence_id=inputs[0],
                destination=self.artifact_storage_root / "exports" / bundle_name,
                replay_identity=dict(config.get("replay_identity") or {}),
                dataset_identity=dict(config.get("dataset_identity") or {}),
                license_metadata=dict(config.get("license_metadata") or {}),
                model_card=self._optional_str(config.get("model_card")),
                priority=priority,
                max_retries=max_retries,
                capacity_override_reason=self._optional_str(config.get("capacity_override_reason")),
            )
        else:
            raise ValueError("kind must be merge, bake, convert, quantize, or export")
        operation = self._v4_engine().get_operation(receipt.domain_id)
        if operation is None:
            return receipt.to_dict()
        return {
            **self._artifact_operation_view(operation),
            "work_item_id": receipt.work_item_id,
            "reused": receipt.reused,
        }

    def get_artifact_operation(self, operation_id: str) -> Dict[str, Any]:
        studio = self._artifact_studio_engine()
        if studio is None:
            raise RuntimeError("Artifact Studio is unavailable")
        operation = self._v4_engine().get_operation(operation_id)
        if operation is None:
            raise KeyError(operation_id)
        return self._artifact_operation_view(operation)

    def list_qualifications(
        self,
        *,
        occurrence_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        db = self._dataset_database()
        clauses: list[str] = []
        params: list[Any] = []
        if occurrence_id:
            clauses.append("occurrence_id = ?")
            params.append(occurrence_id)
        if status:
            clauses.append("(status = ? OR decision = ?)")
            params.extend((status, status))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total_row = db._conn.execute(
            f"SELECT COUNT(*) AS value FROM artifact_qualifications {where}", params
        ).fetchone()
        bounded_limit = max(1, min(1000, int(limit)))
        bounded_offset = max(0, int(offset))
        rows = db._conn.execute(
            f"SELECT id FROM artifact_qualifications {where} "
            "ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            [*params, bounded_limit, bounded_offset],
        ).fetchall()
        items = [
            value
            for row in rows
            if (value := self._v4_engine().get_qualification(str(row["id"]))) is not None
        ]
        total = int(total_row["value"] if total_row else len(items))
        return {
            "items": [self._qualification_view(value) for value in items],
            "total": total,
            "limit": bounded_limit,
            "offset": bounded_offset,
            "has_more": bounded_offset + len(items) < total,
        }

    @staticmethod
    def _qualification_view(qualification: Any) -> Dict[str, Any]:
        value = (
            qualification.to_dict() if hasattr(qualification, "to_dict") else dict(qualification)
        )
        evidence = dict(value.get("metrics") or {})
        decision_evidence = dict(evidence.get("decision") or {})
        flattened_metrics: Dict[str, Any] = {}
        for stage in ("development", "operational", "holdout"):
            stage_metrics = evidence.get(stage)
            if isinstance(stage_metrics, dict):
                for name, metric_value in stage_metrics.items():
                    flattened_metrics[f"{stage}.{name}"] = metric_value
        quality_deltas: Dict[str, Any] = {}
        for stage in ("development", "operational", "holdout"):
            stage_decision = decision_evidence.get(stage)
            if not isinstance(stage_decision, dict):
                continue
            for metric in stage_decision.get("metrics") or []:
                if not isinstance(metric, dict) or not metric.get("metric"):
                    continue
                quality_deltas[f"{stage}.{metric['metric']}"] = metric.get("raw_delta")
        return {
            **value,
            "artifact_id": value.get("occurrence_id") or value.get("artifact_id"),
            "parent_artifact_id": value.get("parent_occurrence_id")
            or value.get("parent_artifact_id"),
            "metrics": flattened_metrics,
            "quality_deltas": quality_deltas,
            "performance": dict(evidence.get("operational") or {}),
            "evidence": evidence,
        }

    def create_qualification(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        studio = self._artifact_studio_engine()
        if studio is None:
            raise RuntimeError("Artifact Studio is unavailable")
        receipt = studio.queue_qualification(
            occurrence_id=str(payload.get("artifact_id") or payload.get("occurrence_id") or ""),
            profile_revision_id=str(payload.get("profile_revision_id") or ""),
            parent_occurrence_id=self._optional_str(
                payload.get("parent_artifact_id") or payload.get("parent_occurrence_id")
            ),
            execution_request=dict(payload.get("execution_request") or {}),
            priority=int(payload.get("priority", 0)),
            max_retries=int(payload.get("max_retries", 1)),
        )
        qualification = self._v4_engine().get_qualification(receipt.domain_id)
        if qualification is None:
            return receipt.to_dict()
        return {
            **self._qualification_view(qualification),
            "work_item_id": receipt.work_item_id,
        }

    def get_qualification(self, qualification_id: str) -> Optional[Dict[str, Any]]:
        value = self._v4_engine().get_qualification(qualification_id)
        return self._qualification_view(value) if value is not None else None

    @staticmethod
    def _qualification_profile_view(revision: Any, *, name: Optional[str] = None) -> Dict[str, Any]:
        value = revision.to_dict() if hasattr(revision, "to_dict") else dict(revision)
        thresholds = list(value.get("thresholds") or [])
        settings = dict(value.get("generation_settings") or {})
        generation = dict(settings.get("generation_settings") or settings)
        return {
            **value,
            "revision": value.get("revision_number"),
            "name": name or value.get("name") or value.get("profile_id"),
            "development_suite_revision_id": value.get("quality_suite_revision_id"),
            "metrics": [
                {
                    "name": item.get("metric") or item.get("name"),
                    "direction": item.get("direction", "maximize"),
                    "threshold": item.get("pass_threshold", item.get("threshold")),
                    "allowed_delta": item.get(
                        "maximum_regression", item.get("allowed_quality_delta")
                    ),
                    "required": bool(item.get("required", True)),
                    "stage": item.get("stage", "development"),
                }
                for item in thresholds
            ],
            "generation_settings": generation,
            "performance_settings": dict(settings.get("performance_settings") or {}),
        }

    def list_qualification_profiles(self, *, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        db = self._dataset_database()
        total_row = db._conn.execute(
            "SELECT COUNT(*) AS value FROM qualification_profile_revisions"
        ).fetchone()
        rows = db._conn.execute(
            "SELECT r.*, p.name FROM qualification_profile_revisions r "
            "JOIN qualification_profiles p ON p.id = r.profile_id "
            "ORDER BY r.created_at DESC LIMIT ? OFFSET ?",
            (max(1, min(1000, int(limit))), max(0, int(offset))),
        ).fetchall()
        items = []
        for row in rows:
            revision = self._v4_engine().get_qualification_profile_revision(str(row["id"]))
            if revision is not None:
                items.append(self._qualification_profile_view(revision, name=str(row["name"])))
        return {
            "items": items,
            "total": int(total_row["value"] if total_row else len(items)),
            "limit": limit,
            "offset": offset,
        }

    def get_qualification_profile(self, revision_id: str) -> Optional[Dict[str, Any]]:
        revision = self._v4_engine().get_qualification_profile_revision(revision_id)
        if revision is None:
            return None
        row = (
            self._dataset_database()
            ._conn.execute(
                "SELECT name FROM qualification_profiles WHERE id = ?",
                (revision.profile_id,),
            )
            .fetchone()
        )
        return self._qualification_profile_view(revision, name=str(row["name"]) if row else None)

    def create_qualification_profile(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from halo_forge.qualification_lab import QualificationProfileRevision

        quality_id = str(
            payload.get("development_suite_revision_id")
            or payload.get("quality_suite_revision_id")
            or ""
        )
        operational_id = str(payload.get("operational_suite_revision_id") or "")
        holdout_id = self._optional_str(payload.get("holdout_suite_revision_id"))
        if not quality_id or not operational_id:
            raise ValueError("development and operational suite revisions are required")
        db = self._dataset_database()

        def require_purpose(revision_id: str, purpose: str) -> None:
            revision = db.get_benchmark_suite_revision(revision_id)
            if revision is None:
                raise KeyError(revision_id)
            suite = db.get_benchmark_suite(revision.suite_id)
            if suite is None or suite.purpose != purpose:
                raise ValueError(f"suite revision {revision_id} must have purpose={purpose}")

        require_purpose(quality_id, "development")
        require_purpose(operational_id, "operational")
        if holdout_id:
            require_purpose(holdout_id, "holdout")
        raw_thresholds = payload.get("thresholds") or payload.get("metrics") or []
        thresholds: list[dict[str, Any]] = []
        stage_rules: Dict[str, list[dict[str, Any]]] = {
            "development": [],
            "operational": [],
            "holdout": [],
        }
        for raw in raw_thresholds:
            if not isinstance(raw, dict):
                raise ValueError("qualification metric rules must be objects")
            stage = str(raw.get("stage") or "development").strip().lower()
            if stage not in stage_rules:
                raise ValueError(
                    "qualification metric stage must be development, operational, or holdout"
                )
            normalized = {
                "metric": str(raw.get("metric") or raw.get("name") or "").strip(),
                "direction": str(raw.get("direction") or "maximize").strip().lower(),
                "pass_threshold": raw.get("pass_threshold", raw.get("threshold")),
                "warn_threshold": raw.get("warn_threshold"),
                "maximum_regression": raw.get(
                    "maximum_regression",
                    raw.get("allowed_quality_delta", raw.get("allowed_delta")),
                ),
                "required": bool(raw.get("required", True)),
            }
            if not normalized["metric"]:
                raise ValueError("qualification metric name is required")
            stage_rules[stage].append(normalized)
            thresholds.append({"stage": stage, **normalized})
        if not stage_rules["development"] or not stage_rules["operational"]:
            raise ValueError(
                "qualification requires at least one development and one operational metric rule"
            )
        if holdout_id and not stage_rules["holdout"]:
            raise ValueError("a holdout suite requires at least one holdout metric rule")
        if not holdout_id and stage_rules["holdout"]:
            raise ValueError("holdout metric rules require a holdout suite revision")
        profile = QualificationProfileRevision(
            profile_id=str(payload.get("profile_id") or "pending"),
            revision_number=1,
            name=str(payload.get("name") or "Qualification profile"),
            development_suite_revision_id=quality_id,
            operational_suite_revision_id=operational_id,
            holdout_suite_revision_id=holdout_id,
            development_rules=tuple(stage_rules["development"]),
            operational_rules=tuple(stage_rules["operational"]),
            holdout_rules=tuple(stage_rules["holdout"]),
            target_backend=str(payload.get("target_backend") or "local"),
            generation_settings=dict(payload.get("generation_settings") or {}),
            performance_settings=dict(payload.get("performance_settings") or {}),
        )
        existing_profile_id = self._optional_str(payload.get("profile_id"))
        if existing_profile_id is None:
            row = db._conn.execute(
                "SELECT id FROM qualification_profiles WHERE name = ?",
                (profile.name,),
            ).fetchone()
            existing_profile_id = str(row["id"]) if row is not None else None
        if existing_profile_id:
            for revision in self._v4_engine().list_qualification_profile_revisions(
                profile_id=existing_profile_id
            ):
                if revision.content_hash == profile.content_hash:
                    return {
                        **self._qualification_profile_view(revision, name=profile.name),
                        "reused": True,
                    }
        stored = self._v4_engine().create_qualification_profile_revision(
            name=profile.name,
            content_hash=profile.content_hash,
            quality_suite_revision_id=quality_id,
            operational_suite_revision_id=operational_id,
            holdout_suite_revision_id=holdout_id,
            thresholds=thresholds,
            target_backend=profile.target_backend,
            generation_settings={
                "generation_settings": profile.generation_settings.to_dict(),
                "performance_settings": profile.performance_settings.to_dict(),
            },
            profile_id=self._optional_str(payload.get("profile_id")),
            description=self._optional_str(payload.get("description")),
        )
        return self._qualification_profile_view(stored, name=profile.name)

    def compare_qualifications(self, base_id: str, candidate_id: str) -> Dict[str, Any]:
        """Return direction-aware deltas for two results under one profile revision."""

        catalog = self._v4_engine()
        base = catalog.get_qualification(base_id)
        candidate = catalog.get_qualification(candidate_id)
        if base is None:
            raise KeyError(base_id)
        if candidate is None:
            raise KeyError(candidate_id)
        if base.status != "completed" or candidate.status != "completed":
            raise ValueError("qualification comparison requires two completed results")
        if base.profile_revision_id != candidate.profile_revision_id:
            raise ValueError("qualifications can only be compared under the same profile revision")
        profile = catalog.get_qualification_profile_revision(base.profile_revision_id)
        if profile is None:
            raise KeyError(base.profile_revision_id)
        base_metrics = dict(base.to_dict().get("metrics") or {})
        candidate_metrics = dict(candidate.to_dict().get("metrics") or {})
        deltas = []
        for rule in profile.to_dict().get("thresholds") or []:
            stage = str(rule.get("stage") or "development")
            metric = str(rule.get("metric") or rule.get("name") or "")
            if not metric:
                continue
            direction = str(rule.get("direction") or "maximize")
            base_value = (base_metrics.get(stage) or {}).get(metric)
            candidate_value = (candidate_metrics.get(stage) or {}).get(metric)
            raw_delta = (
                None
                if base_value is None or candidate_value is None
                else float(candidate_value) - float(base_value)
            )
            deltas.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "direction": direction,
                    "parent_value": base_value,
                    "candidate_value": candidate_value,
                    "raw_delta": raw_delta,
                    "favorable_delta": (
                        None
                        if raw_delta is None
                        else raw_delta if direction == "maximize" else -raw_delta
                    ),
                }
            )
        base_occurrence = catalog.get_occurrence(base.occurrence_id)
        candidate_occurrence = catalog.get_occurrence(candidate.occurrence_id)
        base_blob = catalog.get_blob(base_occurrence.blob_id) if base_occurrence else None
        candidate_blob = (
            catalog.get_blob(candidate_occurrence.blob_id) if candidate_occurrence else None
        )
        return {
            "profile_revision_id": base.profile_revision_id,
            "profile_content_hash": profile.content_hash,
            "base_qualification_id": base.id,
            "candidate_qualification_id": candidate.id,
            "parent_artifact_hash": base_blob.content_hash if base_blob else None,
            "candidate_artifact_hash": (candidate_blob.content_hash if candidate_blob else None),
            "deltas": deltas,
        }

    @staticmethod
    def _model_artifact_view(artifact: Any) -> Dict[str, Any]:
        value = artifact.to_dict()
        return {
            **value,
            "kind": value.get("artifact_kind"),
            "content_hash": value.get("artifact_hash"),
            "model_name": value.get("model_id"),
            "segment_id": value.get("trial_segment_id"),
        }

    @staticmethod
    def _artifact_location_view(location: Any) -> Dict[str, Any]:
        value = location.to_dict() if hasattr(location, "to_dict") else dict(location)
        return {
            **value,
            "kind": value.get("storage_mode") or value.get("location_kind"),
            "available": value.get("state") == "available",
            "verified_at": value.get("last_verified_at"),
        }

    def _artifact_occurrence_view(self, occurrence: Any) -> Dict[str, Any]:
        catalog = self._v4_engine()
        value = occurrence.to_dict()
        blob = catalog.get_blob(occurrence.blob_id)
        locations = catalog.list_locations(occurrence.blob_id)
        available = [item for item in locations if item.state == "available"]
        preferred = next(
            (item for item in available if item.storage_mode == "managed"),
            available[0] if available else (locations[0] if locations else None),
        )
        aliases = catalog.aliases_for(occurrence.id)
        specialized_row = self._dataset_database()._conn.execute(
            "SELECT * FROM specialized_artifact_metadata "
            "WHERE artifact_occurrence_id = ?",
            (occurrence.id,),
        ).fetchone()
        specialized_task = None
        if specialized_row is not None:
            specialized_task = {
                "task_kind": specialized_row["task_kind"],
                "modality": specialized_row["modality"],
                "label_schema_revision_id": specialized_row[
                    "label_schema_revision_id"
                ],
                "model_head_hash": specialized_row["model_head_hash"],
                "processor_hash": specialized_row["processor_hash"],
                "loss_adapter": specialized_row["loss_adapter"],
                "loss_adapter_version": specialized_row["loss_adapter_version"],
                "retrieval_corpus_hash": specialized_row[
                    "retrieval_corpus_hash"
                ],
                "metadata": json.loads(specialized_row["metadata_json"] or "{}"),
                "created_at": specialized_row["created_at"],
            }
        return {
            **value,
            "kind": value.get("artifact_kind"),
            "content_hash": blob.content_hash if blob else None,
            "format": blob.format if blob else None,
            "dtype": blob.dtype if blob else None,
            "quantization": blob.quantization if blob else None,
            "size_bytes": blob.size_bytes if blob else 0,
            "integrity": blob.integrity_state if blob else "missing",
            "manifest": blob.to_dict().get("manifest") if blob else {},
            "blob": (
                {
                    **blob.to_dict(),
                    "integrity": blob.integrity_state,
                }
                if blob
                else None
            ),
            "model_name": value.get("model_id"),
            "segment_id": value.get("trial_segment_id"),
            "path": preferred.path if preferred else None,
            "locations": [self._artifact_location_view(item) for item in locations],
            "aliases": aliases,
            "promoted": bool({"candidate", "approved"}.intersection(aliases)),
            "specialized_task": specialized_task,
        }

    def list_model_artifacts(
        self,
        *,
        run_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        artifact_kind: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        kind_map = {
            "final": "final_model",
            "merged": "merged_model",
            "converted": "converted_model",
            "quantized": "quantized_model",
        }
        resolved_kind = kind_map.get(str(artifact_kind or ""), artifact_kind)
        db = self._dataset_database()
        v4_clauses: list[str] = []
        v4_params: list[Any] = []
        legacy_clauses = [
            "NOT EXISTS (SELECT 1 FROM artifact_occurrences mapped "
            "WHERE mapped.legacy_model_artifact_id = m.id)"
        ]
        legacy_params: list[Any] = []
        for column, value in (
            ("run_id", run_id),
            ("run_group_id", run_group_id),
            ("artifact_kind", resolved_kind),
        ):
            if value is None:
                continue
            v4_clauses.append(f"o.{column} = ?")
            v4_params.append(value)
            legacy_clauses.append(f"m.{column} = ?")
            legacy_params.append(value)
        needle = str(query or "").strip().lower()
        if needle:
            escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            v4_clauses.append(
                "LOWER(o.id || ' ' || o.model_id || ' ' || COALESCE(o.tags_json, '') || "
                "' ' || COALESCE(o.notes, '') || ' ' || b.content_hash) LIKE ? ESCAPE '\\'"
            )
            v4_params.append(pattern)
            legacy_clauses.append(
                "LOWER(m.id || ' ' || m.model_id || ' ' || m.artifact_hash || "
                "' ' || COALESCE(m.metadata_json, '')) LIKE ? ESCAPE '\\'"
            )
            legacy_params.append(pattern)
        v4_where = f"WHERE {' AND '.join(v4_clauses)}" if v4_clauses else ""
        legacy_where = f"WHERE {' AND '.join(legacy_clauses)}"
        union_sql = (
            "SELECT 'v4' AS source, o.id AS id, o.created_at AS created_at "
            "FROM artifact_occurrences o JOIN artifact_blobs b ON b.id = o.blob_id "
            f"{v4_where} UNION ALL "
            "SELECT 'legacy' AS source, m.id AS id, m.created_at AS created_at "
            f"FROM model_artifacts m {legacy_where}"
        )
        params = [*v4_params, *legacy_params]
        total_row = db._conn.execute(
            f"SELECT COUNT(*) AS value FROM ({union_sql})", params
        ).fetchone()
        bounded_limit = max(1, min(1000, int(limit)))
        start = max(0, int(offset))
        rows = db._conn.execute(
            f"SELECT source, id, created_at FROM ({union_sql}) "
            "ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            [*params, bounded_limit, start],
        ).fetchall()
        items: list[dict[str, Any]] = []
        for row in rows:
            if row["source"] == "v4":
                occurrence = self._v4_engine().get_occurrence(str(row["id"]))
                if occurrence is not None:
                    items.append(self._artifact_occurrence_view(occurrence))
            else:
                legacy = db.get_model_artifact(str(row["id"]))
                if legacy is not None:
                    items.append(self._model_artifact_view(legacy))
        total = int(total_row["value"] if total_row else len(items))
        return {
            "items": items,
            "total": total,
            "offset": start,
            "limit": bounded_limit,
            "has_more": start + len(items) < total,
        }

    def get_model_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        occurrence = self._v4_engine().get_occurrence(artifact_id)
        if occurrence is not None:
            lineage = self._v4_engine().lineage(occurrence.id)

            def edge(value: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    **value,
                    "id": value.get("id")
                    or f"{value.get('parent_occurrence_id')}:{value.get('child_occurrence_id')}",
                    "parent_artifact_id": value.get("parent_occurrence_id"),
                    "child_artifact_id": value.get("child_occurrence_id"),
                    "relationship": value.get("relation"),
                }

            return {
                **self._artifact_occurrence_view(occurrence),
                "lineage": lineage,
                "parents": [edge(value) for value in lineage.get("parents") or []],
                "children": [edge(value) for value in lineage.get("children") or []],
            }
        value = self._dataset_database().get_model_artifact(artifact_id)
        if value is None:
            row = (
                self._dataset_database()
                ._conn.execute(
                    "SELECT id FROM artifact_occurrences WHERE legacy_model_artifact_id = ? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (artifact_id,),
                )
                .fetchone()
            )
            if row is not None:
                occurrence = self._v4_engine().get_occurrence(str(row["id"]))
                return self._artifact_occurrence_view(occurrence) if occurrence else None
            return None
        return self._model_artifact_view(value)

    @staticmethod
    def _infer_artifact_format(path: Path, supplied: Optional[str]) -> str:
        if supplied:
            normalized = str(supplied).strip().lower()
            return "hf" if normalized == "huggingface" else normalized
        if path.is_dir():
            return "hf"
        suffix = path.suffix.lower().lstrip(".")
        return {
            "gguf": "gguf",
            "onnx": "onnx",
            "safetensors": "hf",
            "bin": "hf",
            "npz": "mlx",
        }.get(suffix, suffix or "hf")

    def import_model_artifact(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        studio = self._artifact_studio_engine()
        if studio is None:
            raise RuntimeError("Artifact Studio is unavailable")
        raw_path = str(payload.get("path") or "").strip()
        if not raw_path:
            raise ValueError("path is required")
        path = Path(raw_path).expanduser().resolve(strict=True)
        artifact = studio.import_artifact(
            path,
            artifact_kind=str(payload.get("kind") or payload.get("artifact_kind") or "final"),
            artifact_format=self._infer_artifact_format(
                path, self._optional_str(payload.get("format"))
            ),
            model_id=self._optional_str(payload.get("model_id")) or path.name,
            backend=str(payload.get("backend") or "local"),
            managed=bool(payload.get("adopt", payload.get("managed", False))),
            dtype=self._optional_str(payload.get("dtype")),
            quantization=self._optional_str(payload.get("quantization")),
            metadata={
                "import_notes": self._optional_str(payload.get("notes")),
                **dict(payload.get("metadata") or {}),
            },
        )
        occurrence_id = str((artifact.get("occurrence") or {}).get("id") or "")
        if payload.get("notes") and occurrence_id:
            studio.update_annotations(occurrence_id, notes=str(payload.get("notes")).strip())
        view = self.get_model_artifact(occurrence_id)
        if view is None:
            raise RuntimeError("imported artifact was not cataloged")
        return {"artifact": view}

    def get_model_artifact_lineage(self, artifact_id: str) -> Dict[str, Any]:
        occurrence = self._v4_engine().get_occurrence(artifact_id)
        if occurrence is None:
            raise KeyError(artifact_id)
        catalog_lineage = self._v4_engine().lineage(artifact_id)
        parent_edges = list(catalog_lineage.get("parents") or [])
        child_edges = list(catalog_lineage.get("children") or [])

        def edge_view(value: Dict[str, Any]) -> Dict[str, Any]:
            return {
                **value,
                "id": value.get("id")
                or f"{value.get('parent_occurrence_id')}:{value.get('child_occurrence_id')}",
                "parent_artifact_id": value.get("parent_occurrence_id"),
                "child_artifact_id": value.get("child_occurrence_id"),
                "relationship": value.get("relation"),
            }

        parents = []
        for edge in parent_edges:
            parent = self.get_model_artifact(str(edge.get("parent_occurrence_id") or ""))
            if parent:
                parents.append(parent)
        children = []
        for edge in child_edges:
            child = self.get_model_artifact(str(edge.get("child_occurrence_id") or ""))
            if child:
                children.append(child)
        artifact = self.get_model_artifact(artifact_id)
        assert artifact is not None
        return {
            "artifact": artifact,
            "parents": parents,
            "children": children,
            "edges": [edge_view(value) for value in (*parent_edges, *child_edges)],
        }

    def verify_model_artifact(self, artifact_id: str) -> Dict[str, Any]:
        studio = self._artifact_studio_engine()
        if studio is None:
            raise RuntimeError("Artifact Studio is unavailable")
        report = studio.verify_artifact(artifact_id)
        artifact = self.get_model_artifact(artifact_id)
        return {"artifact": artifact, "verification": report}

    def pin_model_artifact(self, artifact_id: str, *, pinned: bool) -> Dict[str, Any]:
        studio = self._artifact_studio_engine()
        if studio is None:
            raise RuntimeError("Artifact Studio is unavailable")
        studio.pin_artifact(artifact_id, pinned=pinned)
        artifact = self.get_model_artifact(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        return artifact

    def tag_model_artifact(self, artifact_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        studio = self._artifact_studio_engine()
        if studio is None:
            raise RuntimeError("Artifact Studio is unavailable")
        tags = payload.get("tags") or []
        if not isinstance(tags, list):
            raise ValueError("tags must be an array")
        studio.tag_artifact(
            artifact_id,
            [str(value) for value in tags],
            replace=bool(payload.get("replace", True)),
        )
        if "notes" in payload:
            studio.update_annotations(
                artifact_id, notes=self._optional_str(payload.get("notes")) or ""
            )
        artifact = self.get_model_artifact(artifact_id)
        if artifact is None:
            raise KeyError(artifact_id)
        return artifact

    def promote_model_artifact(self, artifact_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        studio = self._artifact_studio_engine()
        if studio is None:
            raise RuntimeError("Artifact Studio is unavailable")
        alias = str(payload.get("alias") or payload.get("target_alias") or "").strip()
        override = bool(payload.get("override", False))
        note = self._optional_str(payload.get("note") or payload.get("override_note"))
        if override and not note:
            raise ValueError("a promotion override requires a note")
        result = studio.promote(
            artifact_id,
            alias,
            override_note=note if override else None,
        )
        return {
            "id": f"alias:{result.get('alias')}:{result.get('updated_at')}",
            "alias": result.get("alias"),
            "artifact_id": result.get("occurrence_id"),
            "previous_artifact_id": result.get("previous_occurrence_id"),
            "note": note,
            "created_at": result.get("updated_at"),
            **result,
        }

    def _failure_mining_comparison(
        self, *, candidate_id: str, base_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Resolve immutable evaluation evidence into the mining-core shape."""

        db = self._dataset_database()
        candidate = db.get_evaluation(candidate_id)
        if candidate is None:
            raise KeyError(candidate_id)
        if candidate.status != "completed":
            raise ValueError("Failure mining requires a completed candidate evaluation")
        base = None
        if base_id:
            base = db.get_evaluation(base_id)
            if base is None:
                raise KeyError(base_id)
            if base.status != "completed":
                raise ValueError("Failure mining requires a completed base evaluation")

        policy_gate = getattr(self._evaluation_engine(), "require_failure_mining_allowed", None)
        if callable(policy_gate):
            from halo_forge.evaluation_lab import EvaluationLabError

            try:
                policy_gate(
                    candidate_evaluation_id=candidate_id,
                    base_evaluation_id=base_id,
                )
            except EvaluationLabError as exc:
                raise ValueError(str(exc)) from exc
        else:
            # Compatibility for injected v2 evaluation facades.  Current
            # services centralize this policy in EvaluationLabService.
            revision = db.get_benchmark_suite_revision(candidate.suite_revision_id)
            suite = db.get_benchmark_suite(revision.suite_id) if revision is not None else None
            if suite is not None and suite.purpose == "holdout":
                raise ValueError(
                    "Holdout evaluation evidence is confirmation-only and cannot be mined into training data"
                )
        if base is not None:
            return self._evaluation_engine().compare(base_id, candidate_id)
        return {
            "candidate_evaluation_id": candidate.id,
            "suite_revision_id": candidate.suite_revision_id,
            "samples": [
                value.to_dict()
                for value in db.list_evaluation_samples(candidate.id)
                if value.valid and value.mineable
            ],
        }

    @staticmethod
    def _failure_mining_preview_view(preview: Any) -> Dict[str, Any]:
        items = []
        for candidate in (*preview.selected, *preview.excluded):
            value = candidate.to_dict()
            value.update(
                classification=candidate.outcome,
                base_score=candidate.base.get("score"),
                candidate_score=candidate.candidate.get("score"),
            )
            items.append(value)
        return {
            **preview.to_dict(),
            "items": items,
            "total": preview.matched_count,
        }

    def preview_failure_mining(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from halo_forge.data_lab import preview_failure_mining

        candidate_id = str(payload.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ValueError("candidate_id is required")
        base_id = self._optional_str(payload.get("base_id"))
        comparison = self._failure_mining_comparison(base_id=base_id, candidate_id=candidate_id)
        preview = preview_failure_mining(
            comparison,
            payload.get("selector"),
            exclusions=(payload.get("excluded_record_ids") or payload.get("exclusions") or ()),
        )
        return self._failure_mining_preview_view(preview)

    def build_failure_mined_dataset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a reviewed failure selection for atomic child-version publication."""

        dataset_id = str(payload.get("dataset_id") or "").strip()
        parent_version_id = str(payload.get("parent_version_id") or "").strip()
        candidate_id = str(payload.get("candidate_id") or "").strip()
        if not dataset_id or not parent_version_id or not candidate_id:
            raise ValueError("dataset_id, parent_version_id, and candidate_id are required")
        db = self._dataset_database()
        dataset = db.get_dataset(dataset_id)
        parent = db.get_dataset_version(parent_version_id)
        if dataset is None:
            raise KeyError(dataset_id)
        if parent is None or parent.dataset_id != dataset_id:
            raise ValueError("parent_version_id must belong to dataset_id")
        comparison = self._failure_mining_comparison(
            base_id=self._optional_str(payload.get("base_id")),
            candidate_id=candidate_id,
        )
        request = {
            "dataset_id": dataset_id,
            "parent_version_id": parent_version_id,
            "comparison": comparison,
            "selector": payload.get("selector"),
            "exclusions": (payload.get("excluded_record_ids") or payload.get("exclusions") or []),
            "target_split": str(payload.get("target_split") or "train"),
            "mode": str(payload.get("mode") or "append"),
            "materialize_assets": payload.get("materialize_assets"),
        }
        engine_job = self._dataset_engine().start_job("failure_mining", request)
        data = self._normalize_engine_job_data(self._data_object(engine_job))
        job_id = str(data.get("id") or data.get("job_id") or "")
        if not job_id:
            raise ValueError("Dataset Lab did not return a failure-mining job id")
        job = db.create_dataset_job(
            job_id=job_id,
            dataset_id=dataset_id,
            job_type="failure_mining",
            status=str(data.get("status") or "queued"),
            stage=str(data.get("stage") or "queued"),
            work_item_id=data.get("work_item_id"),
            request={
                "parent_version_id": parent_version_id,
                "base_id": payload.get("base_id"),
                "candidate_id": candidate_id,
                "selector": payload.get("selector"),
                "excluded_record_ids": list(request["exclusions"]),
                "target_split": request["target_split"],
                "mode": request["mode"],
            },
        )
        return {**job.to_dict(), "job_id": job.id}

    def get_run_eval(self, run_identifier: str) -> Dict[str, Any]:
        """Return the lm_eval_summary.json for a run if present.

        Track F-K building block. Looks for `lm_eval_summary.json` inside
        the run's output_dir; honest unavailable shape on miss so the
        cohort dashboard can render a missing-eval column without 5xx.
        """
        try:
            source = self._resolve_run_source(run_identifier)
        except Exception as exc:
            return {
                "available": False,
                "reason": f"Run not found: {exc}",
                "tasks": [],
            }

        if source["kind"] == "summary":
            output_dir = source["summary"].output_dir
        else:
            job = source["job"]
            output_dir = Path(str(job.output_dir)) if job.output_dir else None

        if output_dir is None:
            return {
                "available": False,
                "reason": "Run has no output_dir to inspect",
                "tasks": [],
            }

        eval_path = Path(output_dir) / "lm_eval_summary.json"
        if not eval_path.exists():
            return {
                "available": False,
                "reason": f"No eval summary at {eval_path.name} — run "
                f"`halo-forge eval --output {output_dir}` to populate.",
                "tasks": [],
            }

        try:
            data = json.loads(eval_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "available": False,
                "reason": f"Eval summary unreadable: {exc}",
                "tasks": [],
            }

        return {
            "available": True,
            "model_name": data.get("model_name"),
            "tasks": [
                {
                    "task": tr.get("task"),
                    "primary_metric": tr.get("primary_metric"),
                    "value": tr.get("value"),
                    "n_samples": tr.get("n_samples"),
                    "error": tr.get("error"),
                }
                for tr in data.get("task_results", [])
            ],
            "n_tasks_completed": data.get("n_tasks_completed"),
            "duration_seconds": data.get("duration_seconds"),
            "backend": data.get("backend"),
            "summary_path": str(eval_path),
        }

    def get_eval_cohort(
        self,
        run_ids: List[str],
    ) -> Dict[str, Any]:
        """Aggregate eval summaries across N runs into a cohort table.

        Track F-K. Returns ``{"runs": [{run_id, ...}], "tasks": [task_name],
        "cells": {run_id: {task: value}}}`` so the frontend can render
        a sortable runs-×-tasks grid without per-task fetching.

        Missing eval summaries surface as `available: False` on the run
        entry; the cohort table renders those rows with em-dashes.
        """
        run_entries: List[Dict[str, Any]] = []
        cells: Dict[str, Dict[str, Any]] = {}
        all_tasks: List[str] = []
        seen_tasks: set[str] = set()

        for raw_id in run_ids:
            run_id = str(raw_id or "").strip()
            if not run_id:
                continue
            eval_data = self.get_run_eval(run_id)

            entry = {
                "run_id": run_id,
                "available": eval_data.get("available", False),
                "reason": eval_data.get("reason"),
                "model_name": eval_data.get("model_name"),
                "duration_seconds": eval_data.get("duration_seconds"),
                "backend": eval_data.get("backend"),
            }
            run_entries.append(entry)

            cells[run_id] = {}
            for task in eval_data.get("tasks") or []:
                name = task.get("task")
                if not name:
                    continue
                if name not in seen_tasks:
                    seen_tasks.add(name)
                    all_tasks.append(name)
                cells[run_id][name] = {
                    "primary_metric": task.get("primary_metric"),
                    "value": task.get("value"),
                    "n_samples": task.get("n_samples"),
                    "error": task.get("error"),
                }

        # Per-task best so the UI can highlight winners. Higher is
        # better for accuracy-shaped metrics; lower is better for loss.
        # We can't always tell, so we surface both and let the UI decide
        # based on the metric name (acc / acc_norm / pass@1 / exact_match
        # all higher-is-better; metrics ending in _stderr or _loss don't
        # apply to this dashboard).
        best_per_task_high: Dict[str, Optional[str]] = {}
        for task in all_tasks:
            best_run, best_val = None, None
            for run_id in cells:
                cell = cells[run_id].get(task)
                if cell is None or cell.get("error"):
                    continue
                v = cell.get("value")
                if not isinstance(v, (int, float)):
                    continue
                if best_val is None or v > best_val:
                    best_val = v
                    best_run = run_id
            best_per_task_high[task] = best_run

        return {
            "runs": run_entries,
            "tasks": all_tasks,
            "cells": cells,
            "best_per_task_higher_is_better": best_per_task_high,
        }

    # ----- run stats (Track P3) -------------------------------------------

    def get_run_stats(self) -> Dict[str, Any]:
        """Aggregate counts for the Prometheus `/metrics` endpoint.

        Cheap to compute — single SQLite scan + a dict over the
        in-memory job table. Sub-millisecond on any reasonable run-DB
        size.
        """
        from halo_forge.run_db import get_database

        db = get_database()
        by_modality: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        try:
            cur = db._conn.execute(  # noqa: SLF001 — internal optimization, intentional
                "SELECT modality, status, COUNT(*) AS c FROM runs GROUP BY modality, status"
            )
            for row in cur.fetchall():
                modality = str(row["modality"] or "unknown")
                status = str(row["status"] or "unknown")
                count = int(row["c"])
                by_modality[modality] = by_modality.get(modality, 0) + count
                by_status[status] = by_status.get(status, 0) + count
        except Exception:
            # Empty DB / first call before any sync — leave dicts empty.
            pass

        total = sum(by_modality.values())

        # Active runs come from the in-memory job table; the DB
        # doesn't track runs that are still streaming.
        active = sum(
            1 for job in self.app_state.jobs.values() if job.status in {"pending", "running"}
        )

        return {
            "total_runs": total,
            "by_modality": by_modality,
            "by_status": by_status,
            "active_runs": active,
        }

    # ----- playground proxy (Track F-S) -------------------------------------

    def playground_chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: Optional[List[str]] = None,
        serve_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 120.0,
    ) -> Dict[str, Any]:
        """Forward a chat request to a `halo-forge serve`-style endpoint.

        Avoids CORS by routing through the public API; lets the frontend
        chat UI hit any OpenAI-compatible endpoint (local serve, remote
        host, hosted API) under one auth + origin model. Returns the
        upstream response body verbatim so the UI gets the OpenAI shape
        it expects.

        Defaults the serve URL to `http://127.0.0.1:8001/v1` — exactly
        what `halo-forge serve` exposes locally.
        """
        import os
        import httpx

        resolved_url = (
            serve_url
            or os.environ.get("HALOFORGE_PLAYGROUND_BASE_URL")
            or "http://127.0.0.1:8001/v1"
        )
        resolved_key = api_key or os.environ.get("HALOFORGE_PLAYGROUND_API_KEY") or "EMPTY"

        body: Dict[str, Any] = {
            "model": model or "halo-forge",
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if stop:
            body["stop"] = list(stop)

        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(
                f"{resolved_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {resolved_key}"},
                json=body,
            )
            # Pass upstream errors through so the UI can render the
            # actual problem (model not loaded, OOM, etc.) instead of
            # a generic 500.
            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except Exception:
                    detail = {"error": resp.text}
                return {
                    "upstream_error": True,
                    "status": resp.status_code,
                    "detail": detail,
                    **_friendly_upstream_error(detail),
                }
            return resp.json()

    @staticmethod
    def _process_identity_live(pid: Optional[int], started_at: Optional[float]) -> bool:
        from halo_forge.workstation_jobs import process_matches

        if not process_matches(pid, started_at):
            return False
        try:
            import psutil  # type: ignore[import-not-found]

            return psutil.Process(int(pid)).status() != psutil.STATUS_ZOMBIE
        except Exception:
            return True

    def serve_status(self) -> Dict[str, Any]:
        status = self.serve_manager.status()
        db = self._dataset_database()
        lease = db.get_resource_lease("accelerator")
        if lease is not None and lease.holder_id == self._serving_lease_holder:
            if status.get("running"):
                pid = status.get("pid")
                started_at = None
                if pid is not None:
                    from halo_forge.workstation_jobs import process_start_time

                    started_at = process_start_time(int(pid))
                metadata = {
                    **dict(lease.metadata or {}),
                    "state": status.get("state"),
                    "model": status.get("model"),
                    "backend": status.get("backend"),
                    "url": status.get("url"),
                }
                lease = db.heartbeat_serving_lease(
                    holder_id=self._serving_lease_holder,
                    holder_pid=int(pid) if pid is not None else None,
                    holder_pid_started_at=started_at,
                    metadata=metadata,
                )
            else:
                adopted_alive = self._process_identity_live(
                    lease.holder_pid, lease.holder_pid_started_at
                )
                if adopted_alive:
                    # A dashboard restart can lose the in-memory Popen handle
                    # while the child is still alive. Preserve the lease and make
                    # that reconciliation state explicit instead of double-serving.
                    status = {
                        **status,
                        "state": "needs_reconciliation",
                        "ready_state": "unknown",
                        "running": True,
                        "pid": lease.holder_pid,
                        "message": "The serving process is alive but was adopted after a runtime restart.",
                        "adopted_process": True,
                    }
                else:
                    # Dead or PID-reused children cannot retain the workstation.
                    db.release_serving_lease(holder_id=self._serving_lease_holder)
                    lease = None
        elif lease is not None and lease.holder_type == "serving":
            metadata = dict(lease.metadata or {})
            state = str(metadata.get("state") or "reserved")
            alive = self._process_identity_live(lease.holder_pid, lease.holder_pid_started_at)
            if state == "reserved":
                status = {
                    **status,
                    "state": "reserved",
                    "ready_state": "reserved",
                    "running": False,
                    "message": "The model server is reserved and waiting to start.",
                }
            elif alive:
                launcher = dict(metadata.get("launcher_result") or {})
                status = {
                    **status,
                    "state": "running",
                    "ready_state": "server_ready",
                    "running": True,
                    "pid": lease.holder_pid,
                    "model": launcher.get("model"),
                    "backend": launcher.get("backend"),
                    "host": launcher.get("host"),
                    "port": launcher.get("port"),
                    "url": launcher.get("url"),
                    "log_path": launcher.get("log_path"),
                    "message": "The managed Artifact Studio server is running.",
                    "adopted_process": True,
                }
            else:
                db.release_serving_lease(holder_id=lease.holder_id)
                lease = None
        return {
            **status,
            "resource_lease": lease.to_dict() if lease is not None else None,
            "artifact_id": (lease.metadata or {}).get("occurrence_id") if lease else None,
            "serving_profile_revision_id": (
                (lease.metadata or {}).get("profile_revision_id") if lease else None
            ),
        }

    def serve_start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        current = self.serve_status()
        if current.get("running"):
            raise ValueError(
                "A local model is already being served. Stop it before starting another."
            )
        artifact_id = self._optional_str(
            payload.get("artifact_id")
            or payload.get("occurrence_id")
            or payload.get("model_artifact_id")
        )
        artifact = self.get_model_artifact(artifact_id) if artifact_id else None
        if artifact_id and artifact is None:
            raise KeyError(artifact_id)
        model_ref = str(payload.get("model") or (artifact or {}).get("path") or "")
        if artifact_id and not model_ref:
            raise ValueError("artifact has no available model location")
        request = ServeStartRequest(
            model=model_ref,
            backend=self._optional_str(payload.get("backend") or (artifact or {}).get("backend")),
            host=str(payload.get("host") or "127.0.0.1"),
            port=int(payload.get("port") or 8001),
            trust_remote_code=bool(payload.get("trust_remote_code", False)),
        )
        db = self._dataset_database()
        reservation = None
        if artifact_id:
            studio = self._artifact_studio_engine()
            if studio is None:
                raise RuntimeError("Artifact Studio is unavailable")
            reservation = studio.reserve_serving(
                artifact_id,
                name=str(
                    payload.get("profile_name")
                    or f"Serve {artifact.get('model_name') or artifact_id}"
                ),
                backend=str(request.backend or "local"),
                endpoint_settings={"host": request.host, "port": request.port},
                generation_settings=dict(payload.get("generation_settings") or {}),
                resource_requirements=dict(payload.get("resource_requirements") or {}),
                serving_id=self._serving_lease_holder,
            ).to_dict()
            if reservation.get("state") != "reserved":
                raise ValueError(str(reservation.get("reason") or "serving is blocked"))
            lease = db.get_resource_lease("accelerator")
        else:
            lease = db.acquire_serving_lease(
                holder_id=self._serving_lease_holder,
                metadata={"model": request.model, "backend": request.backend},
            )
        if lease is None:
            blocker = db.get_resource_lease("accelerator")
            owner = blocker.holder_type if blocker is not None else "another operation"
            raise ValueError(
                f"The accelerator is reserved by {owner}; stop or finish that work before serving"
            )
        try:
            status = self.serve_manager.start(request)
        except Exception:
            db.release_serving_lease(holder_id=self._serving_lease_holder)
            raise
        pid = status.get("pid")
        process_started_at = None
        if pid is not None:
            from halo_forge.workstation_jobs import process_start_time

            process_started_at = process_start_time(int(pid))
        metadata = {
            **dict(lease.metadata or {}),
            "occurrence_id": artifact_id,
            "model": request.model,
            "backend": request.backend,
            "url": status.get("url"),
            "profile_revision_id": (((reservation or {}).get("profile_revision") or {}).get("id")),
        }
        lease = (
            db.heartbeat_serving_lease(
                holder_id=self._serving_lease_holder,
                holder_pid=int(pid) if pid is not None else None,
                holder_pid_started_at=process_started_at,
                metadata=metadata,
            )
            or lease
        )
        return {
            **status,
            "artifact_id": artifact_id,
            "serving_profile": (reservation or {}).get("profile_revision"),
            "resource_lease": lease.to_dict(),
        }

    def serve_model_artifact(self, artifact_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        studio = self._artifact_studio_engine()
        if studio is None:
            raise RuntimeError("Artifact Studio is unavailable")
        if getattr(studio, "serving_starter", None) is not None:
            artifact = self.get_model_artifact(artifact_id)
            if artifact is None:
                raise KeyError(artifact_id)
            receipt = studio.queue_serving(
                artifact_id,
                name=str(
                    payload.get("profile_name")
                    or f"Serve {artifact.get('model_name') or artifact_id}"
                ),
                backend=str(payload.get("backend") or artifact.get("backend") or "local"),
                endpoint_settings={
                    "host": str(payload.get("host") or "127.0.0.1"),
                    "port": int(payload.get("port") or 8001),
                },
                generation_settings=dict(payload.get("generation_settings") or {}),
                resource_requirements=dict(payload.get("resource_requirements") or {}),
                start_process=True,
                priority=int(payload.get("priority", 0)),
                max_retries=int(payload.get("max_retries", 1)),
            )
            return {
                **receipt.to_dict(),
                "state": "queued",
                "artifact_id": artifact_id,
            }
        return self.serve_start({**dict(payload), "artifact_id": artifact_id})

    def serve_stop(self) -> Dict[str, Any]:
        db = self._dataset_database()
        lease = db.get_resource_lease("accelerator")
        manager_status = self.serve_manager.status()
        if manager_status.get("running"):
            stopped = self.serve_manager.stop()
            db.release_serving_lease(holder_id=self._serving_lease_holder)
            return {**stopped, "stopped": True, "resource_lease": None}

        # After a runtime restart the retained lease may point at a child for
        # which this process no longer has a Popen handle. Confirm the full
        # PID/start identity, terminate that exact process, and only then free
        # the workstation. PID equality alone is deliberately insufficient.
        if lease is not None and lease.holder_type == "serving":
            pid = lease.holder_pid
            started_at = lease.holder_pid_started_at
            lease_state = str((lease.metadata or {}).get("state") or "")
            if lease_state != "reserved" and self._process_identity_live(pid, started_at):
                if pid == os.getpid():
                    return {
                        **manager_status,
                        "state": "needs_reconciliation",
                        "stopped": False,
                        "message": "Refusing to terminate the dashboard process; reconcile the serving lease manually.",
                        "resource_lease": lease.to_dict(),
                    }
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except (OSError, PermissionError) as exc:
                    return {
                        **manager_status,
                        "state": "needs_reconciliation",
                        "stopped": False,
                        "message": f"The adopted serving process could not be stopped: {exc}",
                        "resource_lease": lease.to_dict(),
                    }
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline and self._process_identity_live(pid, started_at):
                    time.sleep(0.1)
                if self._process_identity_live(pid, started_at):
                    return {
                        **manager_status,
                        "state": "needs_reconciliation",
                        "stopped": False,
                        "message": "The adopted serving process did not exit; its lease remains active.",
                        "resource_lease": lease.to_dict(),
                    }
            db.release_serving_lease(holder_id=lease.holder_id)
        else:
            # Preserve a serving lease owned by another runtime.
            try:
                stopped = self.serve_manager.stop()
            except Exception:
                stopped = manager_status
            if lease is not None:
                return {**stopped, "stopped": False, "resource_lease": lease.to_dict()}
        return {
            **manager_status,
            "running": False,
            "state": "idle",
            "stopped": True,
            "resource_lease": None,
        }

    def serve_logs(self, *, tail: int = 200) -> Dict[str, Any]:
        return self.serve_manager.logs(tail=tail)

    def serve_health(self) -> Dict[str, Any]:
        return self.serve_manager.health()

    def list_serving_profiles(self, *, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        db = self._dataset_database()
        rows = db._conn.execute(
            "SELECT r.*, p.name FROM serving_profile_revisions r "
            "JOIN serving_profiles p ON p.id = r.profile_id "
            "ORDER BY r.created_at DESC LIMIT ? OFFSET ?",
            (max(1, min(1000, int(limit))), max(0, int(offset))),
        ).fetchall()
        total = db._conn.execute(
            "SELECT COUNT(*) AS value FROM serving_profile_revisions"
        ).fetchone()
        return {
            "items": [self._serving_profile_view(dict(row)) for row in rows],
            "total": int(total["value"] if total else len(rows)),
            "limit": limit,
            "offset": offset,
        }

    @staticmethod
    def _serving_profile_view(value: Dict[str, Any]) -> Dict[str, Any]:
        endpoint = json.loads(value.get("endpoint_settings_json") or "{}")
        generation = json.loads(value.get("generation_settings_json") or "{}")
        resources = json.loads(value.get("resource_requirements_json") or "{}")
        return {
            **value,
            "revision": value.get("revision_number"),
            "artifact_id": value.get("occurrence_id"),
            "host": endpoint.get("host"),
            "port": endpoint.get("port"),
            "endpoint_settings": endpoint,
            "generation_defaults": generation,
            "resource_expectations": resources,
            "chat_template": value.get("chat_template_hash"),
        }

    def release_artifact_serving(self, serving_id: str) -> Dict[str, Any]:
        if serving_id == self._serving_lease_holder:
            return {"serving_id": serving_id, **self.serve_stop()}
        studio = self._artifact_studio_engine()
        if studio is None:
            raise RuntimeError("Artifact Studio is unavailable")
        lease = self._dataset_database().get_resource_lease("accelerator")
        if lease is not None and lease.holder_id == serving_id:
            return {"serving_id": serving_id, **self.serve_stop()}
        released = studio.release_serving(serving_id)
        return {"serving_id": serving_id, "released": bool(released)}

    # ----- persistent Playground sessions -------------------------------

    def _playground_session_view(self, row: Any) -> Dict[str, Any]:
        value = dict(row)
        settings = json.loads(value.get("settings_json") or "{}")
        db = self._dataset_database()
        count_row = db._conn.execute(
            "SELECT COUNT(*) AS value FROM playground_messages WHERE session_id = ?",
            (value["id"],),
        ).fetchone()
        messages = db._conn.execute(
            "SELECT * FROM (SELECT * FROM playground_messages WHERE session_id = ? "
            "ORDER BY ordinal DESC LIMIT 500) ORDER BY ordinal",
            (value["id"],),
        ).fetchall()
        message_count = int(count_row["value"] if count_row else len(messages))
        return {
            "id": value["id"],
            "name": value["name"],
            "artifact_id": value.get("primary_occurrence_id"),
            "compare_artifact_id": value.get("comparison_occurrence_id"),
            "endpoint": settings.get("endpoint"),
            "seed": settings.get("seed"),
            "generation_settings": dict(settings.get("generation_settings") or {}),
            "settings": settings,
            "messages": [
                {
                    "id": message["id"],
                    "role": message["role"],
                    "content": message["content"],
                    "artifact_id": message["occurrence_id"],
                    "generation": json.loads(message["generation_json"] or "{}"),
                    "evidence": json.loads(message["evidence_json"] or "{}"),
                    "created_at": message["created_at"],
                }
                for message in messages
            ],
            "message_count": message_count,
            "messages_truncated": message_count > len(messages),
            "created_at": value["created_at"],
            "updated_at": value["updated_at"],
            "archived": bool(value.get("archived")),
        }

    def list_playground_sessions(
        self, *, limit: int = 100, offset: int = 0, include_archived: bool = False
    ) -> Dict[str, Any]:
        db = self._dataset_database()
        where = "" if include_archived else "WHERE archived = 0"
        rows = db._conn.execute(
            f"SELECT * FROM playground_sessions {where} "
            "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (max(1, min(500, int(limit))), max(0, int(offset))),
        ).fetchall()
        total = db._conn.execute(
            f"SELECT COUNT(*) AS value FROM playground_sessions {where}"
        ).fetchone()
        return {
            "items": [self._playground_session_view(row) for row in rows],
            "total": int(total["value"] if total else len(rows)),
            "limit": limit,
            "offset": offset,
        }

    def get_playground_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        row = (
            self._dataset_database()
            ._conn.execute("SELECT * FROM playground_sessions WHERE id = ?", (session_id,))
            .fetchone()
        )
        return self._playground_session_view(row) if row is not None else None

    def _validate_playground_artifact(self, artifact_id: Optional[str]) -> None:
        if artifact_id and self._v4_engine().get_occurrence(artifact_id) is None:
            raise KeyError(artifact_id)

    def create_playground_session(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = str(payload.get("name") or "Untitled session").strip()
        if not name:
            raise ValueError("name cannot be empty")
        primary = self._optional_str(
            payload.get("artifact_id") or payload.get("primary_occurrence_id")
        )
        comparison = self._optional_str(
            payload.get("compare_artifact_id") or payload.get("comparison_occurrence_id")
        )
        self._validate_playground_artifact(primary)
        self._validate_playground_artifact(comparison)
        identifier = str(payload.get("id") or f"playground-{uuid.uuid4().hex}")
        now = datetime.now(timezone.utc).isoformat()
        settings = {
            **dict(payload.get("settings") or {}),
            "endpoint": payload.get("endpoint"),
            "seed": payload.get("seed"),
            "generation_settings": dict(payload.get("generation_settings") or {}),
        }
        db = self._dataset_database()
        with db._lock:
            db._conn.execute(
                "INSERT INTO playground_sessions "
                "(id, name, primary_occurrence_id, comparison_occurrence_id, "
                "settings_json, created_at, updated_at, archived) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    identifier,
                    name,
                    primary,
                    comparison,
                    json.dumps(settings, sort_keys=True),
                    now,
                    now,
                ),
            )
            db._conn.commit()
        return self.get_playground_session(identifier)  # type: ignore[return-value]

    def update_playground_session(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get_playground_session(session_id)
        if current is None:
            raise KeyError(session_id)
        primary = self._optional_str(payload.get("artifact_id", current.get("artifact_id")))
        comparison = self._optional_str(
            payload.get("compare_artifact_id", current.get("compare_artifact_id"))
        )
        self._validate_playground_artifact(primary)
        self._validate_playground_artifact(comparison)
        settings = dict(current.get("settings") or {})
        settings.update(dict(payload.get("settings") or {}))
        for key in ("endpoint", "seed"):
            if key in payload:
                settings[key] = payload[key]
        if "generation_settings" in payload:
            settings["generation_settings"] = dict(payload.get("generation_settings") or {})
        name = str(payload.get("name", current["name"])).strip()
        if not name:
            raise ValueError("name cannot be empty")
        now = datetime.now(timezone.utc).isoformat()
        db = self._dataset_database()
        with db._lock:
            db._conn.execute(
                "UPDATE playground_sessions SET name = ?, primary_occurrence_id = ?, "
                "comparison_occurrence_id = ?, settings_json = ?, updated_at = ?, "
                "archived = ? WHERE id = ?",
                (
                    name,
                    primary,
                    comparison,
                    json.dumps(settings, sort_keys=True),
                    now,
                    int(bool(payload.get("archived", current.get("archived", False)))),
                    session_id,
                ),
            )
            db._conn.commit()
        return self.get_playground_session(session_id)  # type: ignore[return-value]

    def append_playground_message(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.get_playground_session(session_id) is None:
            raise KeyError(session_id)
        role = str(payload.get("role") or "").strip().lower()
        if role not in {"system", "user", "assistant"}:
            raise ValueError("role must be system, user, or assistant")
        content = str(payload.get("content") or "")
        if not content:
            raise ValueError("content is required")
        occurrence_id = self._optional_str(
            payload.get("artifact_id") or payload.get("occurrence_id")
        )
        self._validate_playground_artifact(occurrence_id)
        now = datetime.now(timezone.utc).isoformat()
        identifier = f"playground-message-{uuid.uuid4().hex}"
        db = self._dataset_database()
        with db._lock:
            previous = db._conn.execute(
                "SELECT COALESCE(MAX(ordinal), -1) AS value FROM playground_messages "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            ordinal = int(previous["value"]) + 1
            db._conn.execute(
                "INSERT INTO playground_messages "
                "(id, session_id, ordinal, role, content, occurrence_id, "
                "generation_json, evidence_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identifier,
                    session_id,
                    ordinal,
                    role,
                    content,
                    occurrence_id,
                    json.dumps(dict(payload.get("generation") or {}), sort_keys=True),
                    json.dumps(dict(payload.get("evidence") or {}), sort_keys=True),
                    now,
                ),
            )
            db._conn.execute(
                "UPDATE playground_sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            db._conn.commit()
        return self.get_playground_session(session_id)  # type: ignore[return-value]

    def review_playground_session(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        session = self.get_playground_session(session_id)
        if session is None:
            raise KeyError(session_id)
        pairings = payload.get("pairings")
        message_ids = payload.get("message_ids") or []
        if not isinstance(message_ids, list) or not all(
            isinstance(value, str) for value in message_ids
        ):
            raise ValueError("message_ids must be a list of persisted message IDs")
        review_note = str(payload.get("review_note") or "").strip()
        kind = str(payload.get("kind") or "").strip().lower()
        from halo_forge.public_api.playground_reviews import (
            create_benchmark_revision_from_turns,
            create_dataset_source_draft_from_turns,
            create_review_acquisition_records_from_turns,
        )

        common = {
            "session_id": session_id,
            "session_name": str(session["name"]),
            "messages": session.get("messages") or (),
            "message_ids": message_ids,
            "review_note": review_note,
            "artifact_id": self._optional_str(session.get("artifact_id")),
        }
        if kind == "benchmark_suite":
            result = create_benchmark_revision_from_turns(self._dataset_database(), **common)
            return {
                **result,
                "id": result["revision"]["id"],
                "kind": "benchmark_suite",
                "benchmark_suite_revision_id": result["revision"]["id"],
            }
        if kind == "dataset_source":
            result = create_dataset_source_draft_from_turns(self.dataset_storage_root, **common)
            return {
                **result,
                "kind": "dataset_source",
                "dataset_source_draft_id": result["id"],
            }
        if kind == "review_queue":
            schema_revision_id = str(payload.get("schema_revision_id") or "").strip()
            if not schema_revision_id:
                raise ValueError("schema_revision_id is required for a Playground review queue")
            schema_revision = self._review_engine().get_schema_revision(
                schema_revision_id
            )
            if schema_revision is None:
                raise KeyError(schema_revision_id)
            preference_task = str(schema_revision.task_type) in {"pairwise", "ranking"}
            if pairings is not None and not preference_task:
                raise ValueError(
                    "explicit Playground pairings require a pairwise or ranking schema"
                )
            if preference_task and pairings is None:
                raise ValueError(
                    "Playground pairwise and ranking queues require explicit persisted "
                    "base/candidate pairings"
                )
            normalized = create_review_acquisition_records_from_turns(
                **common,
                pairings=pairings,
            )
            batch = self.create_acquisition_batch(
                {
                    "records": normalized["records"],
                    "name": str(payload.get("name") or f"Playground · {session['name']}"),
                    "seed": int(payload.get("seed") or 0),
                    "strategies": payload.get("strategies") or [{"kind": "explicit"}],
                    "metadata": normalized["provenance"],
                }
            )
            queue = self.create_review_queue(
                {
                    "batch_id": batch["id"],
                    "schema_revision_id": schema_revision_id,
                    "name": payload.get("name") or f"Playground · {session['name']}",
                    "policy": dict(payload.get("policy") or {}),
                }
            )
            return {
                "id": queue["id"],
                "kind": "review_queue",
                "review_queue_id": queue["id"],
                "acquisition_batch": batch,
                "queue": queue,
                "reviewed_turn_count": len(normalized["records"]),
                "pairing_count": int(
                    normalized["provenance"].get("pairing_count") or 0
                ),
                "starts_training": False,
            }
        raise ValueError("kind must be benchmark_suite, dataset_source, or review_queue")

    # ----- run lineage (Track F-Q) -----------------------------------------

    def get_run_lineage(self, run_id: str) -> Dict[str, Any]:
        from halo_forge.run_db import get_database

        db = get_database()
        return db.get_lineage(run_id)

    def record_run_fork(
        self,
        *,
        child_run_id: str,
        parent_run_id: str,
        forked_at_cycle: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        from halo_forge.run_db import get_database

        db = get_database()
        db.record_fork(
            child_run_id=child_run_id,
            parent_run_id=parent_run_id,
            forked_at_cycle=forked_at_cycle,
            notes=notes,
        )
        return db.get_lineage(child_run_id)

    def remove_run_fork(
        self,
        *,
        child_run_id: str,
        parent_run_id: str,
    ) -> bool:
        from halo_forge.run_db import get_database

        db = get_database()
        return db.remove_fork(
            child_run_id=child_run_id,
            parent_run_id=parent_run_id,
        )

    # ----- model registry (Track F-J) ---------------------------------------

    def list_registry_entries(self) -> List[Dict[str, Any]]:
        from halo_forge.run_db import get_database

        db = get_database()
        return [e.to_dict() for e in db.list_registry_entries()]

    def get_registry_entry(self, entry_id: int) -> Optional[Dict[str, Any]]:
        from halo_forge.run_db import get_database

        db = get_database()
        entry = db.get_registry_entry(entry_id)
        return entry.to_dict() if entry else None

    def create_registry_entry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from halo_forge.run_db import get_database

        db = get_database()
        entry = db.create_registry_entry(
            name=str(payload.get("name") or "").strip(),
            description=payload.get("description"),
            base_model=payload.get("base_model"),
            run_ids=payload.get("run_ids") or [],
            tags=payload.get("tags") or [],
        )
        return entry.to_dict()

    def update_registry_entry(
        self, entry_id: int, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        from halo_forge.run_db import get_database

        db = get_database()
        # Only forward the keys the user actually sent so missing keys
        # mean "leave alone", not "set to None".
        kwargs: Dict[str, Any] = {}
        for k in ("description", "base_model", "run_ids", "tags"):
            if k in payload:
                kwargs[k] = payload[k]
        entry = db.update_registry_entry(entry_id, **kwargs)
        return entry.to_dict() if entry else None

    def delete_registry_entry(self, entry_id: int) -> bool:
        from halo_forge.run_db import get_database

        db = get_database()
        return db.delete_registry_entry(entry_id)

    def _db_record_to_list_item(self, record) -> Dict[str, Any]:
        """Project a `RunRecord` to the list-item dict shape the
        frontend already consumes from /runs.

        Keeps the wire shape stable across the two endpoints so the
        run-list components don't have to branch on which fetched them.
        """
        final_model = Path(record.output_dir) / "final_model" if record.output_dir else None
        return {
            "id": record.fs_id or record.run_id,
            "run_id": record.run_id,
            "modality": record.modality,
            "model_name": record.model_name,
            "status": record.status,
            "timestamp": record.timestamp,
            "cycles_executed": record.cycles_executed,
            "weights_updated": record.weights_updated,
            "final_train_loss": record.final_train_loss,
            "effectiveness": (
                {"verdict": record.effectiveness_verdict} if record.effectiveness_verdict else None
            ),
            "quality_status": record.quality_status,
            "keep_rate": record.keep_rate,
            "top_issue": record.dominant_rejection_reason,
            "output_dir": record.output_dir,
            "final_model_available": bool(final_model and final_model.exists()),
            "artifact_path": str(final_model) if final_model and final_model.exists() else None,
        }

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

        summaries = (
            self.results_service.list_training_runs(force_refresh=True) if include_completed else []
        )
        for summary in summaries:
            item = self._summary_to_list_item(summary, include_research=include_research)
            items.append(item)
            seen_keys.add(str(summary.output_dir.resolve()))

        if not include_completed or active_only:
            items = []

        self._hydrate_all_managed_training_jobs()
        for current in self.app_state.jobs.values():
            if current.type in TRAINING_MODALITIES:
                self._sync_managed_training_job(current)
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

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Return the workstation dashboard summary."""
        self._hydrate_all_managed_training_jobs()
        readiness = self.list_readiness()
        active_rows = [
            self._to_active_row(self._job_to_list_item(job, include_research=False))
            for job in sorted(
                [
                    job
                    for job in self.app_state.jobs.values()
                    if job.type in TRAINING_MODALITIES and job.status in {"pending", "running"}
                ],
                key=lambda current: current.created_at,
                reverse=True,
            )
        ]
        completed_runs = [
            self._summary_to_list_item(summary, include_research=False)
            for summary in self.results_service.list_training_runs(force_refresh=True)
        ]
        completed_runs.sort(key=lambda item: item.timestamp, reverse=True)
        attention_source = active_rows[:]
        attention_source.extend(
            self._to_active_row(item)
            for item in completed_runs
            if item.user_summary.confidence_tone in {"warning", "danger"}
        )
        attention_items = [
            AttentionItemView(
                id=row.id,
                modality=row.modality,
                headline=row.headline,
                why_it_matters=row.metrics_summary.eval_metric_name or row.next_step,
                next_step=row.next_step,
                confidence_tone=row.primary_action.tone if row.primary_action else "warning",
                primary_action=row.primary_action,
            )
            for row in attention_source[:5]
        ]
        dashboard = DashboardSummaryView(
            readiness_tier=str(readiness.get("aggregate_tier") or "experimental"),
            generated_at=readiness.get("generated_at"),
            active_runs_count=len(active_rows),
            attention_count=len(attention_items),
            production_ready_count=sum(
                1 for item in readiness.get("items", []) if bool(item.get("production_ready"))
            ),
            modality_count=len(readiness.get("items", [])),
            active_runs=active_rows[:6],
            attention_items=attention_items,
            recent_outcomes=completed_runs[:6],
        )
        return to_dict(dashboard)

    def _run_dataset_view(self, binding: Any) -> Dict[str, Any]:
        db = self._dataset_database()
        value = binding.to_dict()
        version = db.get_dataset_version(binding.dataset_version_id)
        dataset = db.get_dataset(version.dataset_id) if version is not None else None
        artifact = (
            db.get_training_artifact(binding.training_artifact_id)
            if binding.training_artifact_id
            else None
        )
        value.update(
            dataset_id=version.dataset_id if version is not None else None,
            dataset_name=dataset.name if dataset is not None else None,
            content_hash=version.content_hash if version is not None else None,
            recipe_hash=version.recipe_hash if version is not None else None,
            source_fingerprints=(version.source_fingerprints if version is not None else {}),
            assets_materialized=(version.assets_materialized if version is not None else None),
            artifact_hash=artifact.artifact_hash if artifact is not None else None,
            adapter_id=artifact.adapter_id if artifact is not None else None,
            adapter_version=artifact.adapter_version if artifact is not None else None,
        )
        return value

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
        payload = to_dict(view)
        canonical_id = (
            str(resolved["job"].id)
            if resolved["kind"] == "job"
            else str(
                resolved["summary"].run_id
                or payload.get("id")
                or run_identifier
            )
        )
        payload["datasets"] = [
            self._run_dataset_view(value)
            for value in self._dataset_database().list_run_datasets(canonical_id)
        ]
        evaluations = self._dataset_database().list_evaluations(subject_ref=canonical_id, limit=100)
        payload["evaluations"] = [value.to_dict() for value in evaluations]
        payload["evaluated"] = any(value.status == "completed" for value in evaluations)
        payload["model_artifacts"] = [
            self._model_artifact_view(value)
            for value in self._dataset_database().list_model_artifacts(run_id=canonical_id)
        ]
        payload["work_items"] = [
            self._work_item_view(value)
            for value in self._dataset_database().list_work_items(
                canonical_run_id=canonical_id, limit=100
            )
        ]
        payload["training_plan"] = self._training_plan_engine().run_binding(
            canonical_id
        )
        try:
            audit_page = self.list_reward_integrity_audits(
                run_id=canonical_id, limit=100, offset=0
            )
            trace_page = self.list_training_signal_shards(
                run_id=canonical_id, limit=100, offset=0
            )
            payload["reward_integrity_audits"] = audit_page["items"]
            payload["training_signal_shards"] = trace_page["items"]
            terminal = [
                value
                for value in audit_page["items"]
                if str(value.get("status") or "") in {"completed", "awaiting_review"}
            ]
            latest = terminal[0] if terminal else (
                audit_page["items"][0] if audit_page["items"] else None
            )
            payload["reward_integrity"] = {
                "recorded": bool(trace_page["items"]),
                "audit_count": audit_page["total"],
                "trace_count": trace_page["total"],
                "latest_audit": latest,
                "awaiting_review": any(
                    str(value.get("status") or "") == "awaiting_review"
                    or str((value.get("decision") or {}).get("action") or "")
                    == "pause"
                    for value in audit_page["items"]
                ),
            }
        except Exception:
            # Legacy and partially migrated runs remain readable without
            # pretending their aggregate reward history was captured evidence.
            payload["reward_integrity_audits"] = []
            payload["training_signal_shards"] = []
            payload["reward_integrity"] = {
                "recorded": False,
                "audit_count": 0,
                "trace_count": 0,
                "latest_audit": None,
                "awaiting_review": False,
            }
        return payload

    def get_resolved_run_launch_config(self, run_identifier: str) -> Dict[str, Any]:
        """Return the exact normalized launch payload used by a run."""
        from ui.services.launch_context import read_launch_context

        resolved = self._resolve_run_source(run_identifier)
        if resolved["kind"] == "job":
            job = resolved["job"]
            config = dict(job.launch_args or {})
            mode = str(job.type)
            canonical_id = str(job.id)
        else:
            summary = resolved["summary"]
            context_path = resolved.get("launch_context_path")
            if not context_path:
                raise ValueError("This historical run has no resolved launch configuration")
            context = read_launch_context(context_path)
            config = dict(context.args)
            mode = context.job_type
            canonical_id = str(summary.run_id or summary.id)
        bindings = self._dataset_database().list_run_datasets(canonical_id)
        if bindings:
            config["dataset_bindings"] = [value.to_dict() for value in bindings]
            train = next((value for value in bindings if value.role == "train"), bindings[0])
            config["dataset_version_id"] = train.dataset_version_id
            config["dataset_split"] = train.split
        training_plan_binding = self._training_plan_engine().run_binding(canonical_id)
        if training_plan_binding and training_plan_binding.get("revision"):
            config["training_plan_revision_id"] = training_plan_binding["revision"]["id"]
        config.pop("run_id", None)
        dataset_payload = [value.to_dict() for value in bindings]
        return {
            "run_id": canonical_id,
            "mode": mode,
            "config": config,
            "resolved_config": config,
            "dataset_bindings": dataset_payload,
            "datasets": [self._run_dataset_view(value) for value in bindings],
            "parent_run_id": canonical_id,
            "training_plan": training_plan_binding,
        }

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
                stage=TrainingStageView(
                    key="completed" if detail.status == "completed" else "failed",
                    label="Completed" if detail.status == "completed" else "Failed",
                    message=(
                        "Training completed."
                        if detail.status == "completed"
                        else str(
                            detail.failure_summary.message
                            if detail.failure_summary
                            else "Training failed."
                        )
                    ),
                    progress_percent=100.0 if detail.status == "completed" else 0.0,
                    started_at=detail.timestamp,
                ),
                last_event=detail.top_issue or detail.next_step,
                elapsed_seconds=None,
                eta_seconds=None,
                artifact_state=(
                    "final_model"
                    if detail.details.get("final_model_available")
                    else ("failed" if detail.status == "failed" else "none")
                ),
                headline=detail.headline,
                next_step=detail.next_step,
                top_issue=detail.top_issue,
                user_summary=detail.user_summary,
                metrics_summary=detail.metrics_summary,
                metric_points=self._metric_points_from_summary(summary),
                primary_action=detail.primary_action,
                research_sections=detail.research_sections,
            )
        )

    async def stream_run(self, run_identifier: str, *, include_research: bool = True):
        """Stream polling snapshots as server-sent events."""
        while True:
            payload = self.get_run_live(run_identifier, include_research=include_research)
            yield f"data: {json.dumps(payload)}\n\n"
            status = str(payload.get("status") or "").lower()
            if status in {"completed", "failed", "stopped", "cancelled", "canceled"}:
                break
            await asyncio.sleep(1.0)

    async def stream_telemetry(self, *, interval_seconds: float = 2.0):
        """Push hardware telemetry as server-sent events.

        Replaces the 3s polling on the public_app's TelemetryStrip.
        Yields one `data: <json>\\n\\n` event per `interval_seconds`.

        Each event is the same shape `GET /api/public/telemetry` would
        return — the frontend EventSource can parse it identically and
        feed it into the same render path. The provider's own internal
        cache (1s on rocm-smi/nvidia-smi) keeps the actual subprocess
        cost bounded regardless of how aggressive the interval is.

        Streams until the client disconnects (FastAPI raises
        asyncio.CancelledError, which propagates out cleanly).
        """
        from halo_forge.telemetry import (
            TelemetryUnavailableError,
            get_telemetry_provider,
        )

        try:
            provider = get_telemetry_provider()
        except TelemetryUnavailableError as exc:
            # Emit a single error event then exit so the client gets
            # a clear signal instead of a silent hang.
            yield f"data: {json.dumps({'error': f'Telemetry unavailable: {exc}'})}\n\n"
            return

        # SSE retry hint — if the connection drops, the browser will
        # re-open after this many milliseconds. Keep it short so a
        # network blip doesn't leave the strip stale for long.
        yield f"retry: 3000\n\n"

        while True:
            try:
                sample = provider.sample()
                yield f"data: {json.dumps(sample.to_dict())}\n\n"
            except Exception as exc:
                # Don't crash the stream on a single sample failure;
                # surface the error in the event payload and keep
                # the connection alive so the next interval recovers.
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            await asyncio.sleep(max(0.5, float(interval_seconds)))

    async def stream_run_logs(
        self,
        run_identifier: str,
        *,
        initial_tail: int = 200,
        poll_seconds: float = 1.0,
    ):
        """Tail a run's log file and emit new lines as SSE events.

        The first event carries the `initial_tail` last lines so the
        frontend renders content immediately; subsequent events carry
        only newly-appended lines. Each event payload is
        `{"lines": [...], "log_path": "...", "appended_at": ts}`.

        Stops cleanly when the run reaches a terminal status (the file
        won't grow further) or when the client disconnects.
        """
        try:
            source = self._resolve_run_source(run_identifier)
        except Exception as exc:
            yield f"data: {json.dumps({'error': f'Run not found: {exc}'})}\n\n"
            return

        from pathlib import Path

        out_dir, run_id = _extract_output_dir_and_run_id(source)
        log_path: Optional[Path] = None
        if out_dir:
            for name in ("run.log", "train.log", "training.log", f"{run_id}_training.log"):
                if (out_dir / name).exists():
                    log_path = out_dir / name
                    break
            if log_path is None:
                matches = sorted(
                    out_dir.glob("*_training.log"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if matches:
                    log_path = matches[0]

        if log_path is None:
            # Fall back to logs/ scan once at start; if nothing matches,
            # emit a single "unavailable" event and exit.
            for logs_dir in _candidate_log_roots(self.base_path):
                if not logs_dir.is_dir():
                    continue
                tokens = [t for t in (run_id, out_dir.name if out_dir else "") if t]
                matches = [
                    p for p in logs_dir.glob("*.log") if any(tok in p.name for tok in tokens)
                ]
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                if matches:
                    log_path = matches[0]
                    break

        if log_path is None or not log_path.exists():
            yield f"data: {json.dumps({'error': 'No log file alongside this run.'})}\n\n"
            return

        yield "retry: 3000\n\n"

        # Send the initial tail up front so the user sees something
        # without waiting for new lines.
        try:
            with log_path.open(encoding="utf-8", errors="replace") as f:
                buf: list[str] = []
                for line in f:
                    buf.append(line.rstrip("\n"))
                    if len(buf) > initial_tail * 2:
                        buf = buf[-initial_tail:]
                initial = buf[-initial_tail:] if len(buf) > initial_tail else buf
        except OSError as exc:
            yield f"data: {json.dumps({'error': f'Cannot read log: {exc}'})}\n\n"
            return

        yield f"data: {json.dumps({'lines': initial, 'log_path': str(log_path), 'reset': True})}\n\n"

        # Now follow the file for newly-appended bytes. We track byte
        # offset rather than line count so a partial line on disk
        # doesn't get duplicated when its remainder lands.
        offset = log_path.stat().st_size
        while True:
            try:
                size = log_path.stat().st_size
            except OSError:
                # File rotated/deleted — stop the stream gracefully.
                yield f"data: {json.dumps({'error': 'Log file disappeared'})}\n\n"
                return

            if size > offset:
                try:
                    with log_path.open(encoding="utf-8", errors="replace") as f:
                        f.seek(offset)
                        chunk = f.read(size - offset)
                except OSError:
                    chunk = ""
                if chunk:
                    new_lines = chunk.splitlines()
                    if new_lines:
                        yield f"data: {json.dumps({'lines': new_lines, 'log_path': str(log_path)})}\n\n"
                offset = size
            elif size < offset:
                # File was truncated — re-emit the head so the client
                # doesn't render stale state.
                offset = 0
                continue

            # If the underlying job is in a terminal state, stop here
            # so we don't hold the connection open forever after the
            # log stops growing. Re-resolve cheaply each iteration.
            try:
                refreshed = self._resolve_run_source(run_identifier)
                if refreshed.get("kind") == "summary":
                    # Completed run — give one final flush, then close.
                    await asyncio.sleep(poll_seconds)
                    return
            except Exception:
                pass

            await asyncio.sleep(max(0.25, float(poll_seconds)))

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
        try:
            report = self.readiness_service.load_qualification_report(force_refresh=True)
        except Exception as exc:
            items = [
                ModalityReadinessView(
                    modality=module,
                    readiness_tier="experimental",
                    production_ready=False,
                    status="warn",
                    caveat="Readiness report is not available in this runtime.",
                    next_step="Run readiness checks from a source checkout or continue with documented method caveats.",
                    eval_metric_name="",
                    baseline_value=None,
                    final_value=None,
                    delta=None,
                    details={
                        "errors": [],
                        "warnings": [str(exc)],
                        "weights_updated": False,
                        "optimizer_steps": 0,
                        "samples_kept": 0,
                    },
                )
                for module in TRAINING_MODALITIES
            ]
            return {
                "generated_at": None,
                "aggregate_tier": "experimental",
                "items": [to_dict(item) for item in items],
            }
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
                    next_step=str(
                        getattr(entry, "fix_now", "")
                        or "Review readiness details before wider rollout."
                    ),
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
        job = self.app_state.get_job(identifier) or self._hydrate_managed_training_job(identifier)
        if job is not None and job.type in TRAINING_MODALITIES:
            job = self._sync_managed_training_job(job)
            if job.status not in {"completed", "failed", "stopped"}:
                return {
                    "kind": "job",
                    "job": job,
                    "launch_context_path": (
                        str(job.launch_context_file) if job.launch_context_file else None
                    ),
                    "recovery": self._job_recovery(job),
                }
            summary = self.results_service.load_training_run(job.output_dir)
            if summary is not None and identifier in {
                summary.id,
                str(summary.run_id or ""),
                summary.output_dir.name,
                job.id,
            }:
                return {
                    "kind": "summary",
                    "summary": summary,
                    "launch_context_path": (
                        str(summary.launch_context_path)
                        if summary.launch_context_path
                        else None
                    ),
                    "recovery": self._summary_recovery(summary),
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
                    "launch_context_path": (
                        str(summary.launch_context_path) if summary.launch_context_path else None
                    ),
                    "recovery": self._summary_recovery(summary),
                }
        if job is not None and job.type in TRAINING_MODALITIES:
            return {
                "kind": "job",
                "job": job,
                "launch_context_path": (
                    str(job.launch_context_file) if job.launch_context_file else None
                ),
                "recovery": self._job_recovery(job),
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
                str(item) for item in outlook.get("suggested_adjustments", []) if item is not None
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
        metrics_summary = self._metrics_summary(
            progress_percent=100.0 if status == "completed" else 0.0,
            keep_rate=summary.keep_rate,
            update_steps=summary.total_train_steps_executed,
            final_train_loss=summary.final_train_loss,
            effectiveness=dict(summary.raw_data.get("effectiveness") or {}),
        )
        research_sections = (
            self._build_research_sections(
                yield_diagnostics=summary.yield_diagnostics,
                effectiveness=dict(summary.raw_data.get("effectiveness") or {}),
                recovery=self._summary_recovery(summary),
                representative_examples=list(summary.representative_examples),
                lineage={
                    "run_id": summary.run_id,
                    "resume_from_cycle": summary.resume_from_cycle,
                    "final_model_available": bool(summary.final_model_path),
                },
            )
            if include_research
            else []
        )
        return TrainingRunListItemView(
            id=summary.id,
            run_id=str(summary.run_id or summary.id),
            modality=summary.modality,
            model_name=summary.model_name,
            status=status,
            timestamp=self._isoformat(summary.timestamp),
            headline=user_summary.headline,
            next_step=user_summary.next_step,
            top_issue=summary.dominant_rejection_reason,
            user_summary=user_summary,
            metrics_summary=metrics_summary,
            primary_action=user_summary.primary_action,
            details={
                "verdict": summary.effectiveness_verdict,
                "keep_rate": summary.keep_rate,
                "quality_status": summary.quality_status,
                "top_issue": summary.dominant_rejection_reason,
                "update_steps": summary.total_train_steps_executed,
                "final_train_loss": summary.final_train_loss,
            },
            research_sections=research_sections,
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
            headline=item.headline,
            next_step=item.next_step,
            top_issue=item.top_issue,
            user_summary=item.user_summary,
            metrics_summary=item.metrics_summary,
            recovery=recovery,
            failure_summary=self._failure_summary_for_run(
                run_identifier=item.run_id,
                status=item.status,
                modality=item.modality,
                failure_reason=summary.failure_reason,
                output_dir=summary.output_dir,
                launch_context_path=summary.launch_context_path,
            ),
            primary_action=item.primary_action,
            details={
                **item.details,
                "cycles_executed": summary.cycles_executed,
                "seed": summary.seed,
                "resume_from_cycle": summary.resume_from_cycle,
                "final_model_available": bool(summary.final_model_path),
                # Phase D: per-cycle metric series for the live run view.
                # Flat plot-friendly entries so the frontend chart code
                # can hand them straight to recharts without re-shaping.
                "cycle_metrics": _project_cycles_for_charts(summary.raw_data),
                "cycle_losses": list(summary.cycle_losses),
                "yield_diagnostics": summary.yield_diagnostics,
                # Track P2 — energy/cost rollup. Estimated from wall-clock
                # + active backend's nominal training power; `source` flags
                # the provenance for the UI.
                "cost": _project_run_cost(
                    summary.raw_data,
                    backend_name=self._active_backend_name(),
                ),
            },
            research_sections=item.research_sections,
            internal_details=(
                {
                    "output_dir": str(summary.output_dir),
                    "final_model_path": summary.final_model_path,
                    "launch_context_path": (
                        str(summary.launch_context_path) if summary.launch_context_path else None
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
        yield_summary = (
            live_yield.get("summary") if isinstance(live_yield.get("summary"), dict) else {}
        )
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
            final_reason=(
                job.lifecycle_metadata.get("resume_strategy") if job.lifecycle_metadata else ""
            ),
            has_launch_context=bool(job.launch_context_file),
            can_resume_latest=job.type in {"raft", "vlm", "audio", "reasoning", "agentic"},
            weights_updated=job.current_step > 0 or job.current_cycle > 0,
        )
        metrics_summary = self._metrics_summary(
            progress_percent=job.progress_percent,
            keep_rate=self._coerce_float(yield_rates.get("keep_rate")),
            update_steps=job.current_step,
            final_train_loss=job.latest_loss,
            effectiveness={},
        )
        research_sections = (
            self._build_research_sections(
                yield_diagnostics=live_yield,
                effectiveness={},
                recovery=recovery,
                representative_examples=list(recovery.representative_examples),
                lineage={
                    "run_id": job.id,
                    "current_epoch": job.current_epoch,
                    "current_cycle": job.current_cycle,
                    "output_dir": str(job.output_dir) if job.output_dir else "",
                },
            )
            if include_research
            else []
        )
        return TrainingRunListItemView(
            id=job.id,
            run_id=job.id,
            modality=job.type,
            model_name=job.name,
            status=job.status,
            timestamp=self._isoformat(job.created_at),
            headline=user_summary.headline,
            next_step=user_summary.next_step,
            top_issue=(
                str(yield_summary.get("dominant_rejection_reason"))
                if yield_summary.get("dominant_rejection_reason") not in (None, "")
                else None
            ),
            user_summary=user_summary,
            metrics_summary=metrics_summary,
            primary_action=user_summary.primary_action,
            details={
                "quality_status": yield_summary.get("status"),
                "keep_rate": yield_rates.get("keep_rate"),
                "top_issue": yield_summary.get("dominant_rejection_reason"),
                "update_steps": job.current_step,
                "final_train_loss": job.latest_loss,
            },
            research_sections=research_sections,
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
            headline=item.headline,
            next_step=item.next_step,
            top_issue=item.top_issue,
            user_summary=item.user_summary,
            metrics_summary=item.metrics_summary,
            recovery=recovery,
            failure_summary=self._failure_summary_for_run(
                run_identifier=item.run_id,
                status=item.status,
                modality=item.modality,
                failure_reason=job.error_message,
                output_dir=job.output_dir,
                launch_context_path=job.launch_context_file,
            ),
            primary_action=item.primary_action,
            details={
                **item.details,
                "current_epoch": job.current_epoch,
                "total_epochs": job.total_epochs,
                "current_cycle": job.current_cycle,
                "total_cycles": job.total_cycles,
                "verification_rate": job.verification_rate,
            },
            research_sections=item.research_sections,
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
            progress_percent=job.progress_percent,
            current_step=job.current_step,
            total_steps=job.total_steps,
            current_epoch=job.current_epoch,
            total_epochs=job.total_epochs,
            current_cycle=job.current_cycle,
            total_cycles=job.total_cycles,
            latest_loss=job.latest_loss,
            latest_learning_rate=job.latest_lr,
            latest_grad_norm=job.latest_grad_norm,
            stage=self._stage_for_job(job),
            last_event=job.last_event,
            elapsed_seconds=self._elapsed_seconds_for_job(job),
            eta_seconds=self._eta_seconds_for_job(job),
            artifact_state=str(job.artifact_state or "none"),
            headline=detail.headline,
            next_step=detail.next_step,
            top_issue=detail.top_issue,
            user_summary=detail.user_summary,
            metrics_summary=detail.metrics_summary,
            metric_points=self._metric_points_for_job(job),
            primary_action=detail.primary_action,
            research_sections=detail.research_sections,
        )

    def _stage_for_job(self, job: JobState) -> TrainingStageView:
        key = str(getattr(job, "stage_key", "") or "").strip() or self._stage_key_from_status(
            job.status
        )
        label = str(getattr(job, "stage_label", "") or "").strip() or key.replace("_", " ").title()
        message = str(getattr(job, "stage_message", "") or "").strip()
        if not message:
            message = "Waiting for the next training event." if job.status == "running" else label
        progress = getattr(job, "stage_progress_percent", None)
        if progress is None:
            progress = job.progress_percent
        return TrainingStageView(
            key=key,
            label=label,
            message=message,
            progress_percent=float(progress or 0.0),
            started_at=(
                self._isoformat(job.stage_started_at)
                if getattr(job, "stage_started_at", None)
                else None
            ),
        )

    @staticmethod
    def _stage_key_from_status(status: str) -> str:
        value = str(status or "").lower()
        if value == "completed":
            return "completed"
        if value == "failed":
            return "failed"
        if value in {"stopped", "cancelled", "canceled"}:
            return "cancelled"
        if value == "running":
            return "training"
        return "preparing"

    def _metric_points_for_job(self, job: JobState) -> list[TrainingMetricPointView]:
        points: list[TrainingMetricPointView] = []
        for raw in list(getattr(job, "metric_points", []) or [])[-240:]:
            if not isinstance(raw, dict):
                continue
            timestamp = str(raw.get("timestamp") or "")
            if not timestamp:
                timestamp = self._isoformat(datetime.now(timezone.utc))
            points.append(
                TrainingMetricPointView(
                    step=int(raw.get("step") or 0),
                    timestamp=timestamp,
                    train_loss=self._coerce_float(raw.get("train_loss")),
                    eval_loss=self._coerce_float(raw.get("eval_loss")),
                    learning_rate=self._coerce_float(raw.get("learning_rate")),
                    grad_norm=self._coerce_float(raw.get("grad_norm")),
                    throughput=self._coerce_float(raw.get("throughput")),
                )
            )
        return points

    def _metric_points_from_summary(
        self, summary: TrainingRunSummary
    ) -> list[TrainingMetricPointView]:
        points: list[TrainingMetricPointView] = []
        for entry in _project_cycles_for_charts(summary.raw_data):
            step = _coerce_optional_int(entry.get("train_steps_executed")) or int(
                entry.get("cycle") or 0
            )
            train_loss = _coerce_optional_float(entry.get("train_loss"))
            eval_loss = _coerce_optional_float(entry.get("eval_loss"))
            if train_loss is None and eval_loss is None:
                continue
            points.append(
                TrainingMetricPointView(
                    step=step,
                    timestamp=self._isoformat(summary.timestamp),
                    train_loss=train_loss,
                    eval_loss=eval_loss,
                    learning_rate=_coerce_optional_float(entry.get("learning_rate")),
                    grad_norm=None,
                    throughput=None,
                )
            )
        return points[-240:]

    def _elapsed_seconds_for_job(self, job: JobState) -> Optional[float]:
        started_at = job.started_at
        if started_at is None:
            return None
        end = job.completed_at or datetime.now(timezone.utc)
        if started_at.tzinfo is None:
            started_at = started_at.astimezone(timezone.utc)
        else:
            started_at = started_at.astimezone(timezone.utc)
        if end.tzinfo is None:
            end = end.astimezone(timezone.utc)
        else:
            end = end.astimezone(timezone.utc)
        return max(0.0, (end - started_at).total_seconds())

    def _eta_seconds_for_job(self, job: JobState) -> Optional[float]:
        elapsed = self._elapsed_seconds_for_job(job)
        if elapsed is None or job.status != "running":
            return None
        progress = float(job.progress_percent or getattr(job, "stage_progress_percent", 0.0) or 0.0)
        if progress <= 0.0 or progress >= 100.0:
            return None
        return max(0.0, elapsed * ((100.0 - progress) / progress))

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
        return str(
            getattr(entry, "fix_now", "")
            or "Qualification evidence is incomplete for this modality."
        )

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

    def _build_research_sections(
        self,
        *,
        yield_diagnostics: Dict[str, Any],
        effectiveness: Dict[str, Any],
        recovery: TrainingRecoveryView,
        representative_examples: list[dict[str, Any]],
        lineage: Dict[str, Any],
    ) -> list[ResearchSectionView]:
        sections: list[ResearchSectionView] = []
        yield_summary = (
            yield_diagnostics.get("summary")
            if isinstance(yield_diagnostics.get("summary"), dict)
            else {}
        )
        yield_rates = (
            yield_diagnostics.get("rates")
            if isinstance(yield_diagnostics.get("rates"), dict)
            else {}
        )
        sections.append(
            ResearchSectionView(
                key="data_yield",
                title="Data yield",
                summary=str(yield_summary.get("text") or "Yield details unavailable."),
                items=[
                    {"label": "Quality", "value": yield_summary.get("status")},
                    {"label": "Keep rate", "value": yield_rates.get("keep_rate")},
                    {"label": "Top issue", "value": yield_summary.get("dominant_rejection_reason")},
                ],
            )
        )
        update_quality = (
            effectiveness.get("update_quality")
            if isinstance(effectiveness.get("update_quality"), dict)
            else {}
        )
        sections.append(
            ResearchSectionView(
                key="update_quality",
                title="Update quality",
                summary=str(effectiveness.get("verdict") or "No effectiveness verdict."),
                items=[
                    {
                        "label": "Optimizer steps",
                        "value": update_quality.get("optimizer_steps")
                        or update_quality.get("train_steps_executed"),
                    },
                    {
                        "label": "Final loss",
                        "value": update_quality.get("final_train_loss")
                        or update_quality.get("loss_delta"),
                    },
                    {"label": "Weights updated", "value": update_quality.get("weights_updated")},
                ],
            )
        )
        evaluation = (
            effectiveness.get("evaluation")
            if isinstance(effectiveness.get("evaluation"), dict)
            else {}
        )
        sections.append(
            ResearchSectionView(
                key="eval_outcome",
                title="Eval outcome",
                summary=str(
                    evaluation.get("status") or evaluation.get("metric_name") or "Eval unavailable."
                ),
                items=[
                    {"label": "Metric", "value": evaluation.get("metric_name")},
                    {"label": "Current", "value": evaluation.get("final_value")},
                    {"label": "Delta", "value": evaluation.get("delta")},
                ],
            )
        )
        sections.append(
            ResearchSectionView(
                key="recovery_reasoning",
                title="Recovery reasoning",
                summary=recovery.evidence_summary or "No guided recovery recommendation.",
                items=[
                    {"label": "Status", "value": recovery.status},
                    {"label": "Recommended action", "value": recovery.recommended_action},
                    {"label": "Reason", "value": recovery.reason_code},
                ],
            )
        )
        if representative_examples:
            sections.append(
                ResearchSectionView(
                    key="representative_examples",
                    title="Representative examples",
                    summary="Representative evidence from dropped or weak samples.",
                    items=[dict(example) for example in representative_examples[:3]],
                )
            )
        sections.append(
            ResearchSectionView(
                key="artifact_lineage",
                title="Artifact lineage",
                summary="Artifact and resume lineage for this run.",
                items=[
                    {"label": str(key).replace("_", " ").title(), "value": value}
                    for key, value in lineage.items()
                    if value not in (None, "", False)
                ],
            )
        )
        return sections

    def _failure_summary_for_run(
        self,
        *,
        run_identifier: str,
        status: str,
        modality: str,
        failure_reason: Optional[str],
        output_dir: Optional[Path],
        launch_context_path: Optional[Path],
    ) -> Optional[RunFailureSummaryView]:
        if str(status or "").lower() not in {"failed", "stopped", "cancelled", "canceled"}:
            return None

        log_payload = self.get_run_logs(run_identifier, tail=80)
        log_tail = [str(line) for line in log_payload.get("lines", []) if str(line).strip()][-12:]
        log_path = str(log_payload.get("log_path") or "") or None
        launch_text = self._launch_context_excerpt(launch_context_path)
        text = "\n".join(
            item
            for item in [
                str(failure_reason or ""),
                "\n".join(log_tail),
                launch_text,
                str(output_dir or ""),
            ]
            if item
        )
        classified = _classify_training_failure(text, status=status)
        retry_route = f"/train?mode={modality}"
        return RunFailureSummaryView(
            kind=classified["kind"],
            headline=classified["headline"],
            message=classified["message"],
            next_action=classified["next_action"],
            log_path=log_path,
            log_tail=log_tail,
            retry_route=retry_route,
            docs_url=classified.get("docs_url"),
        )

    @staticmethod
    def _launch_context_excerpt(path: Optional[Path]) -> str:
        if not path:
            return ""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return ""
        if not isinstance(payload, dict):
            return ""
        command = payload.get("command")
        args = payload.get("args")
        fields = {
            "command": command,
            "model": args.get("model") if isinstance(args, dict) else payload.get("model"),
            "dataset": args.get("dataset") if isinstance(args, dict) else payload.get("dataset"),
            "prompts": args.get("prompts") if isinstance(args, dict) else payload.get("prompts"),
            "output_dir": (
                args.get("output_dir") if isinstance(args, dict) else payload.get("output_dir")
            ),
        }
        return "\n".join(
            f"{key}: {value}" for key, value in fields.items() if value not in (None, "")
        )

    def _metrics_summary(
        self,
        *,
        progress_percent: float,
        keep_rate: Optional[float],
        update_steps: int,
        final_train_loss: Optional[float],
        effectiveness: Dict[str, Any],
    ) -> RunMetricsSummaryView:
        evaluation = (
            effectiveness.get("evaluation")
            if isinstance(effectiveness.get("evaluation"), dict)
            else {}
        )
        return RunMetricsSummaryView(
            progress_percent=progress_percent,
            keep_rate=keep_rate,
            update_steps=update_steps,
            final_train_loss=final_train_loss,
            eval_metric_name=str(evaluation.get("metric_name") or ""),
            eval_metric_value=self._coerce_float(evaluation.get("final_value")),
            eval_delta=self._coerce_float(evaluation.get("delta")),
        )

    # ----- Labs V11-V15: outcome, studies, grounding, tasks, environments -----

    def get_future_lab_capabilities(self) -> Dict[str, Any]:
        return self._future_lab_engine().capabilities()

    def get_actionable_guidance(
        self, context_kind: str, context_id: str
    ) -> Dict[str, Any]:
        return self._future_lab_engine().actionable_guidance(
            context_kind, context_id
        ).to_dict()

    def _enqueue_future_lab_work(
        self,
        *,
        action: str,
        domain_kind: str,
        domain_id: str,
        payload: Mapping[str, Any],
        dependencies: Sequence[str] = (),
        resource_class: str = "none",
        max_retries: int = 1,
        work_item_id: Optional[str] = None,
    ) -> Any:
        identifier = work_item_id or f"future-lab-{uuid.uuid4().hex}"
        return self._scheduler().enqueue(
            kind=f"future_lab_{action.replace('.', '_')}",
            launch_spec={
                "handler": "future_lab.execute_work_item",
                "action": action,
                "payload": dict(payload),
                "future_lab_root": str(self.future_lab_storage_root),
                "dataset_root": str(self.dataset_storage_root),
                "evaluation_root": str(self.evaluation_storage_root),
            },
            resource_class=resource_class,
            domain_kind=domain_kind,
            domain_id=domain_id,
            dependencies=tuple(dependencies),
            max_retries=max_retries,
            work_item_id=identifier,
        )

    def list_outcome_profiles(
        self,
        *,
        scenario_revision_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        items = [
            value.to_dict()
            for value in self._future_lab_engine().list_outcome_profiles(
                scenario_revision_id=scenario_revision_id
            )
        ]
        return {
            "items": items[offset : offset + min(max(limit, 1), 500)],
            "total": len(items),
            "limit": min(max(limit, 1), 500),
            "offset": max(offset, 0),
        }

    def assess_training_outcome(
        self, proof_run_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().assess_outcome(
            proof_run_id, payload
        ).to_dict()

    def prepare_training_outcome(
        self, proof_run_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        run = self._dataset_database().get_run(proof_run_id)
        if run is None:
            raise ValueError(f"unknown proof run: {proof_run_id}")
        raw = run.raw
        launch = dict(raw.get("launch_config") or raw.get("config") or {})
        if not bool(raw.get("proof_run") or launch.get("proof_run")):
            raise ValueError("Check training result is available after a proof run")

        suite_revision_id = self._optional_str(
            payload.get("suite_revision_id")
            or raw.get("development_suite_revision_id")
            or launch.get("development_suite_revision_id")
        )
        suite_source = "attached"
        if suite_revision_id is None:
            validation = next(
                (
                    value
                    for value in self._dataset_database().list_run_datasets(proof_run_id)
                    if value.role in {"validation", "eval"} or value.split == "validation"
                ),
                None,
            )
            if validation is not None:
                version = self._dataset_database().get_dataset_version(
                    validation.dataset_version_id
                )
                scenario_revision_id = str(
                    payload.get("scenario_revision_id")
                    or raw.get("scenario_revision_id")
                    or launch.get("scenario_revision_id")
                    or ""
                )
                profile = next(
                    iter(
                        self._future_lab_engine().list_outcome_profiles(
                            scenario_revision_id=scenario_revision_id
                        )
                    ),
                    None,
                )
                starter = profile.evaluation_starters[0] if profile and profile.evaluation_starters else None
                suite = self.create_benchmark_suite(
                    {
                        "name": f"Proof validation — {proof_run_id[:12]}",
                        "description": "Development evidence generated from the proof run's preserved validation split.",
                        "purpose": "development",
                        "items": [
                            {
                                "id": "preserved-validation",
                                "dataset_id": version.dataset_id if version else None,
                                "dataset_version_id": validation.dataset_version_id,
                                "split": validation.split,
                            }
                        ],
                        "primary_metric": starter.primary_metric if starter else "score",
                        "direction": starter.direction if starter else "maximize",
                        "generation_settings": {"seed": 42},
                        "evaluator_versions": {"source": "v16-proof-outcome"},
                    }
                )
                suite_revision_id = str((suite.get("latest_revision") or {}).get("id") or "")
                suite_source = "preserved_validation"
        if not suite_revision_id:
            return {
                "status": "needs_suite",
                "proof_run_id": proof_run_id,
                "message": "Choose development examples so Halo Forge can compare the base and proof models fairly.",
                "guidance": {
                    "display_status": "More evidence needed",
                    "summary": "No compatible development suite or preserved validation split is attached.",
                    "primary_action": {
                        "id": "choose_suite",
                        "label": "Choose evaluation examples",
                    },
                    "secondary_actions": [],
                    "blockers": [
                        "A development evaluation suite is required. Test, canary, operational, and holdout evidence cannot be used."
                    ],
                    "technical_details": {"reason": "development_suite_missing"},
                },
            }
        revision = self._dataset_database().get_benchmark_suite_revision(
            suite_revision_id
        )
        suite = (
            self._dataset_database().get_benchmark_suite(revision.suite_id)
            if revision is not None
            else None
        )
        if revision is None or suite is None:
            raise ValueError(f"unknown benchmark suite revision: {suite_revision_id}")
        if str(suite.purpose).lower() != "development":
            raise ValueError(
                "Check training result can use development evidence only"
            )

        try:
            resolved = self.get_resolved_run_launch_config(proof_run_id)
            config = dict(resolved.get("resolved_config") or {})
        except (KeyError, ValueError):
            # Older proof summaries may predate a standalone launch-context
            # file but still contain the exact resolved launch configuration.
            config = dict(launch)
        model_id = str(config.get("model") or launch.get("model") or "").strip()
        if not model_id:
            raise ValueError("the proof run does not record its exact base model")
        adapter_id = self._optional_str(payload.get("adapter_id"))
        base_evaluation = self.launch_evaluation(
            {
                "suite_revision_id": suite_revision_id,
                "adapter_id": adapter_id,
                "subject": {
                    "kind": "model",
                    "value": model_id,
                    "revision": config.get("model_revision"),
                },
                "request": {"outcome_role": "base", "proof_run_id": proof_run_id},
            }
        )
        proof_evaluation = self.launch_evaluation(
            {
                "suite_revision_id": suite_revision_id,
                "adapter_id": adapter_id,
                "subject": {
                    "kind": "final_model",
                    "value": proof_run_id,
                    "run_id": proof_run_id,
                },
                "request": {"outcome_role": "proof", "proof_run_id": proof_run_id},
            }
        )
        work_item_id = f"future-lab-{uuid.uuid4().hex}"
        request = {
            **dict(payload),
            "suite_revision_id": suite_revision_id,
            "suite_source": suite_source,
            "base_evaluation_id": base_evaluation["id"],
            "candidate_evaluation_id": proof_evaluation["id"],
        }
        prepared = self._future_lab_engine().prepare_outcome_assessment(
            proof_run_id,
            request,
        )
        dependencies = [
            str(value)
            for value in (
                base_evaluation.get("work_item_id"),
                proof_evaluation.get("work_item_id"),
            )
            if value
        ]
        work_item = self._enqueue_future_lab_work(
            action="outcome.prepare",
            domain_kind="training_outcome_assessment",
            domain_id=prepared.id,
            payload={
                **request,
                "proof_run_id": proof_run_id,
                "assessment_id": prepared.id,
            },
            dependencies=dependencies,
            work_item_id=work_item_id,
        )
        prepared = self._future_lab_engine().prepare_outcome_assessment(
            proof_run_id,
            request,
            work_item_id=work_item.id,
        )
        return {
            "status": "queued",
            "assessment": prepared.to_dict(),
            "base_evaluation": base_evaluation,
            "proof_evaluation": proof_evaluation,
            "work_item_id": work_item.id,
            "guidance": self._future_lab_engine().outcome_guidance(prepared).to_dict(),
        }

    def get_training_outcome(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        value = self._future_lab_engine().get_outcome_assessment(assessment_id)
        return value.to_dict() if value else None

    def list_training_outcomes(
        self,
        *,
        proof_run_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._future_lab_engine().list_outcome_assessments(
            proof_run_id=proof_run_id, limit=limit, offset=offset
        )

    def list_training_outcome_findings(
        self, assessment_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._future_lab_engine().list_outcome_findings(
            assessment_id, limit=limit, offset=offset
        )

    def review_training_outcome(
        self, proof_run_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().create_outcome_decision(
            proof_run_id, payload
        ).to_dict()

    def get_full_run_context(
        self,
        proof_run_id: str,
        *,
        assessment_id: Optional[str] = None,
        override_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._future_lab_engine().full_run_context(
            proof_run_id,
            assessment_id=assessment_id,
            override_reason=override_reason,
        )

    def list_adaptation_studies(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._future_lab_engine().list_studies(limit=limit, offset=offset)

    def create_adaptation_study(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._future_lab_engine().create_study(payload).to_dict()

    def get_adaptation_study(self, study_id: str) -> Optional[Dict[str, Any]]:
        value = self._future_lab_engine().get_study(study_id)
        return value.to_dict() if value else None

    def validate_adaptation_study_protocol(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().validate_study_protocol(payload)

    def create_adaptation_study_protocol(
        self, study_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().create_study_protocol(
            study_id, payload
        ).to_dict()

    def get_adaptation_study_protocol(
        self, revision_id: str
    ) -> Optional[Dict[str, Any]]:
        value = self._future_lab_engine().get_study_protocol(revision_id)
        return value.to_dict() if value else None

    def get_adaptation_study_launch_plan(
        self, revision_id: str
    ) -> Dict[str, Any]:
        return self._future_lab_engine().study_launch_plan(revision_id).to_dict()

    def launch_adaptation_study(
        self, revision_id: str, payload: Mapping[str, Any] | None = None
    ) -> Dict[str, Any]:
        plan = self._future_lab_engine().study_launch_plan(revision_id)
        if plan.blockers:
            return {
                "status": "blocked",
                "plan": plan.to_dict(),
                "message": "Complete the highlighted study settings before launch.",
            }
        work_item = self._enqueue_future_lab_work(
            action="study.launch",
            domain_kind="adaptation_study_protocol_revision",
            domain_id=revision_id,
            payload={"revision_id": revision_id, **dict(payload or {})},
        )
        self._dataset_database()._conn.execute(
            """UPDATE adaptation_study_protocol_revisions
               SET launch_status='queued',launch_work_item_id=?,
                   launch_progress_json=?,launch_error=NULL
               WHERE id=?""",
            (
                work_item.id,
                json.dumps({"current": 0, "total": plan.run_count}),
                revision_id,
            ),
        )
        self._dataset_database()._conn.commit()
        return {
            "status": "queued",
            "plan": plan.to_dict(),
            "work_item_id": work_item.id,
        }

    def materialize_adaptation_study(self, revision_id: str) -> Dict[str, Any]:
        return self._future_lab_engine().materialize_study(revision_id)

    def attach_adaptation_study_run(
        self, assignment_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().attach_study_run(
            assignment_id,
            run_id=str(payload.get("run_id") or ""),
            run_group_id=self._optional_str(payload.get("run_group_id")),
        ).to_dict()

    def analyze_adaptation_study(
        self, revision_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        revision = self._future_lab_engine().get_study_protocol(revision_id)
        if revision is None:
            raise ValueError(f"unknown study protocol revision: {revision_id}")
        supplied = dict(payload.get("metrics") or {})
        exclusions = {
            str(value)
            for value in payload.get("excluded_assignment_ids") or []
        }
        unfinished = []
        for assignment in revision.assignments:
            if assignment.id in supplied or assignment.id in exclusions:
                continue
            run = (
                self._dataset_database().get_run(assignment.run_id)
                if assignment.run_id
                else None
            )
            if run is None or run.status != "completed":
                unfinished.append(assignment.id)
        if unfinished:
            return {
                "status": "blocked",
                "stage": "waiting_for_matched_runs",
                "protocol_revision_id": revision_id,
                "message": "Analysis will be available after every matched run finishes or is explicitly excluded with a recorded deviation.",
                "unfinished_assignments": unfinished,
            }
        analysis_id = "study-analysis-" + hashlib.sha256(
            json.dumps(
                {"revision_id": revision_id, "payload": dict(payload)},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:24]
        prepared = self._future_lab_engine().prepare_study_analysis(
            revision_id,
            payload,
            analysis_id=analysis_id,
        )
        work_item = self._enqueue_future_lab_work(
            action="study.analyze",
            domain_kind="adaptation_study_analysis",
            domain_id=analysis_id,
            payload={
                "revision_id": revision_id,
                "analysis_id": analysis_id,
                **dict(payload),
            },
        )
        prepared = self._future_lab_engine().prepare_study_analysis(
            revision_id,
            payload,
            analysis_id=analysis_id,
            work_item_id=work_item.id,
        )
        return prepared.to_dict()

    def record_adaptation_study_deviation(
        self, revision_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().create_study_deviation(
            revision_id, payload
        ).to_dict()

    def record_adaptation_study_decision(
        self, revision_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().create_study_decision(
            revision_id, payload
        ).to_dict()

    def list_grounding_profiles(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._future_lab_engine().list_grounding_profiles(
            limit=limit, offset=offset
        )

    def create_grounding_profile(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._future_lab_engine().create_grounding_profile(payload).to_dict()

    def create_grounding_profile_revision(
        self, profile_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().create_grounding_profile_revision(
            profile_id, payload
        ).to_dict()

    def launch_grounded_generation(
        self, profile_revision_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        values = {"profile_revision_id": profile_revision_id, **dict(payload)}
        preset = str(values.get("preset") or "standard").lower()
        values["quota"] = {"quick": 50, "standard": 250, "thorough": 1000}.get(
            preset, values.get("quota") or 250
        )
        work_item_id = f"future-lab-{uuid.uuid4().hex}"
        profile_revision = self._future_lab_engine().get_grounding_profile_revision(
            profile_revision_id
        )
        teacher = dict(profile_revision.definition.get("teacher") or {}) if profile_revision else {}
        hosted = str(teacher.get("endpoint_type") or "").lower() in {
            "hosted",
            "openai",
            "anthropic",
        }
        if hosted and not bool(values.get("hosted_provider_confirmed")):
            raise ValueError(
                "Confirm the hosted-provider request estimate before generation"
            )
        batch = self._future_lab_engine().prepare_grounded_batch(values)
        work_item = self._enqueue_future_lab_work(
            action="grounding.generate",
            domain_kind="grounded_generation_batch",
            domain_id=batch.id,
            payload={**values, "batch_id": batch.id},
            resource_class=(
                "accelerator"
                if bool(
                    profile_revision
                    and profile_revision.definition.get("local_teacher")
                )
                else "none"
            ),
            work_item_id=work_item_id,
        )
        batch = self._future_lab_engine().prepare_grounded_batch(
            values, work_item_id=work_item.id
        )
        return {**batch.to_dict(), "work_item_id": work_item.id}

    def preview_grounded_generation(
        self, profile_revision_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().grounding_preview(
            profile_revision_id, payload
        ).to_dict()

    def get_grounded_generation(self, batch_id: str) -> Optional[Dict[str, Any]]:
        value = self._future_lab_engine().get_grounded_batch(batch_id)
        return value.to_dict() if value else None

    def list_grounded_generations(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._future_lab_engine().list_grounded_batches(
            limit=limit, offset=offset
        )

    def list_grounded_candidates(
        self,
        batch_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._future_lab_engine().list_grounded_candidates(
            batch_id, status=status, limit=limit, offset=offset
        )

    def create_grounding_review_proposal(self, batch_id: str) -> Dict[str, Any]:
        return self._future_lab_engine().grounding_review_proposal(batch_id)

    def list_specialized_tasks(self) -> Dict[str, Any]:
        return self._future_lab_engine().list_specialized_tasks()

    def get_specialized_task_readiness(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().specialized_task_readiness(payload)

    def verify_specialized_task_artifact(
        self, artifact_path: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().verify_specialized_artifact(
            artifact_path, payload
        )

    def create_task_label_schema(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._future_lab_engine().create_task_label_schema(payload)

    def create_task_label_schema_revision(
        self, schema_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().create_task_label_schema_revision(
            schema_id, payload
        ).to_dict()

    def compute_classification_metrics(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().classification_metrics(
            payload.get("expected") or [],
            payload.get("predicted") or [],
            multilabel=bool(payload.get("multi_label", False)),
        )

    def compute_retrieval_metrics(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._future_lab_engine().retrieval_metrics(
            payload.get("rankings") or [],
            payload.get("relevant") or [],
            k=int(payload.get("k") or 10),
        )

    def list_agent_environments(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._future_lab_engine().list_environments(
            limit=limit, offset=offset
        )

    def create_agent_environment(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._future_lab_engine().create_environment(payload).to_dict()

    def get_agent_environment(self, environment_id: str) -> Optional[Dict[str, Any]]:
        value = self._future_lab_engine().get_environment(environment_id)
        return value.to_dict() if value else None

    def create_agent_environment_revision(
        self, environment_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().create_environment_revision(
            environment_id, payload
        ).to_dict()

    def create_episode_suite(
        self, environment_revision_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        # The suite's immutable environment binding is stored on its revision.
        return self._future_lab_engine().create_episode_suite(payload)

    def create_episode_suite_revision(
        self, suite_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._future_lab_engine().create_episode_suite_revision(
            suite_id, payload
        ).to_dict()

    def launch_agent_episode(
        self, suite_revision_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        values = {"suite_revision_id": suite_revision_id, **dict(payload)}
        work_item_id = f"future-lab-{uuid.uuid4().hex}"
        episode = self._future_lab_engine().prepare_episode(
            values
        )
        work_item = self._enqueue_future_lab_work(
            action="environment.run",
            domain_kind="agent_episode",
            domain_id=episode.id,
            payload={**values, "episode_id": episode.id},
            work_item_id=work_item_id,
        )
        episode = self._future_lab_engine().prepare_episode(
            values, work_item_id=work_item.id
        )
        return {**episode.to_dict(), "work_item_id": work_item.id}

    def get_environment_permission_preview(
        self, environment_revision_id: str
    ) -> Dict[str, Any]:
        return self._future_lab_engine().environment_permissions(
            environment_revision_id
        ).to_dict()

    def rerun_agent_episode(
        self, episode_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        original = self._future_lab_engine().get_episode(episode_id)
        if original is None:
            raise ValueError(f"unknown agent episode: {episode_id}")
        values = {
            **dict(payload),
            "suite_revision_id": original.suite_revision_id,
            "suite_item_id": original.suite_item_id,
            "parent_episode_id": episode_id,
            "subject_type": str(payload.get("subject_type") or "served_model"),
            "subject_ref": str(payload.get("subject_ref") or original.subject_ref),
            "seed": int(payload.get("seed") or original.seed),
        }
        work_item_id = f"future-lab-{uuid.uuid4().hex}"
        episode = self._future_lab_engine().prepare_episode(
            values
        )
        work_item = self._enqueue_future_lab_work(
            action="environment.rerun",
            domain_kind="agent_episode",
            domain_id=episode.id,
            payload={**values, "episode_id": episode.id},
            # The managed local server already owns its serving lease. The
            # episode controller is lightweight and must not contend for a
            # second accelerator lease.
            resource_class="none",
            work_item_id=work_item_id,
        )
        episode = self._future_lab_engine().prepare_episode(
            values, work_item_id=work_item.id
        )
        return {**episode.to_dict(), "work_item_id": work_item.id}

    def _environment_model_actions(
        self, values: Mapping[str, Any]
    ) -> list[Dict[str, Any]]:
        from urllib.parse import urlparse

        serve_url = str(
            values.get("serve_url")
            or os.environ.get("HALOFORGE_PLAYGROUND_BASE_URL")
            or "http://127.0.0.1:8001/v1"
        )
        parsed = urlparse(serve_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "Environment model runs use a local loopback serving endpoint only"
            )
        future = self._future_lab_engine()
        suite = future.get_episode_suite_revision(
            str(values.get("suite_revision_id") or "")
        )
        if suite is None:
            raise ValueError("episode suite revision is unavailable")
        environment = future.get_environment_revision(suite.environment_revision_id)
        if environment is None:
            raise ValueError("environment revision is unavailable")
        item = next(
            (
                dict(value)
                for value in suite.definition.get("items") or []
                if str(value.get("id") or "")
                == str(values.get("suite_item_id") or "")
            ),
            None,
        )
        if item is None:
            raise ValueError("episode suite item is unavailable")
        transitions = dict(environment.definition.get("transitions") or {})
        response = self.playground_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are controlling a deterministic local test environment. "
                        "Return only a JSON array of action objects. Each action must "
                        "use one available action name. Do not include prose."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "goal": item.get("goal"),
                            "initial_state": {
                                **dict(environment.definition.get("initial_state") or {}),
                                **dict(item.get("initial_state") or {}),
                            },
                            "available_actions": sorted(transitions),
                            "action_contract": {
                                name: dict(value).get("input_schema") or {}
                                for name, value in transitions.items()
                            },
                            "max_steps": min(
                                int(suite.definition.get("max_steps") or 16),
                                int(environment.definition.get("max_steps") or 16),
                            ),
                        },
                        sort_keys=True,
                    ),
                },
            ],
            model=str(values.get("subject_ref") or "halo-forge"),
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            serve_url=serve_url,
            timeout_s=float(values.get("timeout_seconds") or 120),
        )
        if response.get("upstream_error"):
            raise RuntimeError(
                str(response.get("message") or response.get("detail") or "model rerun failed")
            )
        choices = list(response.get("choices") or [])
        if not choices:
            raise RuntimeError("the local model returned no action plan")
        content = str((choices[0].get("message") or {}).get("content") or "").strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lstrip().startswith("json"):
                content = content.lstrip()[4:].lstrip()
        actions = json.loads(content)
        if not isinstance(actions, list) or not all(
            isinstance(value, Mapping) for value in actions
        ):
            raise ValueError("the local model action plan is not a JSON array")
        return [dict(value) for value in actions]

    def execute_future_lab_work_item(self, item: Any) -> Dict[str, Any]:
        """Execute the single V16 durable handler under worker supervision."""

        action = str(item.launch_spec.get("action") or "").strip()
        values = dict(item.launch_spec.get("payload") or {})
        future = self._future_lab_engine()
        if action == "outcome.prepare":
            assessment = future.assess_outcome(
                str(values["proof_run_id"]),
                {
                    **values,
                    "_assessment_id": values["assessment_id"],
                    "_work_item_id": item.id,
                },
            )
            return {"assessment": assessment.to_dict()}
        if action == "study.launch":
            revision_id = str(values["revision_id"])
            revision = future.get_study_protocol(revision_id)
            if revision is None:
                raise ValueError(f"unknown study protocol revision: {revision_id}")
            definition = dict(revision.definition)
            seeds = [int(value) for value in definition.get("seeds") or [17, 42, 101]]
            total = len(revision.arms) * len(seeds)
            groups = []
            completed = 0
            self._dataset_database()._conn.execute(
                """UPDATE adaptation_study_protocol_revisions
                   SET launch_status='running',launch_progress_json=?,launch_error=NULL
                   WHERE id=?""",
                (json.dumps({"current": 0, "total": total}), revision_id),
            )
            self._dataset_database()._conn.commit()
            for arm in sorted(revision.arms, key=lambda value: value.ordinal):
                launch_state = self._dataset_database()._conn.execute(
                    """SELECT launch_status FROM adaptation_study_protocol_revisions
                       WHERE id=?""",
                    (revision_id,),
                ).fetchone()
                if launch_state is not None and str(launch_state["launch_status"]) == "cancelled":
                    raise RuntimeError("study launch cancellation requested")
                arm_assignments = [
                    assignment
                    for assignment in revision.assignments
                    if assignment.arm_id == arm.id
                ]
                existing_group_id = next(
                    (
                        str(assignment.run_group_id)
                        for assignment in arm_assignments
                        if assignment.run_group_id
                    ),
                    None,
                )
                config = {
                    **dict(definition.get("base_config") or {}),
                    **dict(arm.launch_config or {}),
                }
                config.setdefault("model", definition.get("model"))
                config.setdefault(
                    "dataset_version_id", definition.get("dataset_version_id")
                )
                config.update(
                    study_id=revision.study_id,
                    study_protocol_revision_id=revision.id,
                    study_arm_id=arm.id,
                    study_assignment_ids_by_seed={
                        str(assignment.seed): assignment.id
                        for assignment in arm_assignments
                    },
                    study_factor_values=dict(arm.factor_values),
                    study_contrast_ids=[
                        contrast.id
                        for contrast in revision.contrasts
                        if contrast.left_arm_id == arm.id
                        or contrast.right_arm_id == arm.id
                    ],
                )
                trainer_mode = str(
                    config.get("trainer_mode")
                    or definition.get("trainer_mode")
                    or ""
                )
                bindings = list(definition.get("dataset_bindings") or [])
                if not bindings and config.get("dataset_version_id"):
                    bindings = [
                        {
                            "role": "train",
                            "dataset_version_id": config["dataset_version_id"],
                            "split": str(config.get("dataset_split") or "train"),
                        }
                    ]
                group = (
                    self.get_run_group(existing_group_id)
                    if existing_group_id
                    else self.create_run_group(
                        {
                            "name": f"{revision.question} — {arm.name}",
                            "kind": "repeat",
                            "trainer_mode": trainer_mode,
                            "base_config": config,
                            "seeds": seeds,
                            "development_suite_revision_id": definition[
                                "development_suite_revision_id"
                            ],
                            "holdout_suite_revision_id": definition.get(
                                "holdout_suite_revision_id"
                            ),
                            "dataset_bindings": bindings,
                        }
                    )
                )
                groups.append(group)
                runs = {
                    int(run["seed"]): run
                    for trial in group.get("trials") or []
                    for run in trial.get("runs") or []
                }
                for assignment in arm_assignments:
                    run = runs.get(int(assignment.seed))
                    self._dataset_database()._conn.execute(
                        """UPDATE adaptation_study_assignments
                           SET run_group_id=?,run_id=?,status=?
                           WHERE id=?""",
                        (
                            group["id"],
                            run.get("run_id") if run else None,
                            "queued" if run else "blocked",
                            assignment.id,
                        ),
                    )
                    completed += 1
                    self._dataset_database()._conn.execute(
                        """UPDATE adaptation_study_protocol_revisions
                           SET launch_progress_json=? WHERE id=?""",
                        (
                            json.dumps({"current": completed, "total": total}),
                            revision_id,
                        ),
                    )
                self._dataset_database()._conn.commit()
            self._dataset_database()._conn.execute(
                """UPDATE adaptation_study_protocol_revisions
                   SET launch_status='completed',launch_progress_json=?,
                       launch_error=NULL WHERE id=?""",
                (json.dumps({"current": total, "total": total}), revision_id),
            )
            self._dataset_database()._conn.commit()
            return {"revision_id": revision_id, "run_groups": groups}
        if action == "study.analyze":
            analysis = future.analyze_study(
                str(values["revision_id"]),
                {
                    **values,
                    "_analysis_id": values["analysis_id"],
                    "_work_item_id": item.id,
                },
            )
            return {"analysis": analysis.to_dict()}
        if action == "grounding.generate":
            batch = future.generate_grounded_batch(
                {
                    **values,
                    "_batch_id": values["batch_id"],
                    "_work_item_id": item.id,
                }
            )
            return {"batch": batch.to_dict()}
        if action in {"environment.run", "environment.rerun"}:
            if not values.get("actions"):
                values["actions"] = self._environment_model_actions(values)
            episode = future.run_episode(
                {
                    **values,
                    "_episode_id": values["episode_id"],
                    "_work_item_id": item.id,
                }
            )
            return {
                "episode": episode.to_dict(),
                "subject_execution": {
                    "episode_id": episode.id,
                    "subject_ref": episode.subject_ref,
                    "subject_hash": episode.subject_hash,
                    "execution_kind": (
                        "model_rerun"
                        if action == "environment.rerun"
                        else "model_run"
                    ),
                    "suite_revision_id": episode.suite_revision_id,
                    "environment_revision_id": (
                        future.get_episode_suite_revision(
                            episode.suite_revision_id
                        ).environment_revision_id
                    ),
                    "parent_episode_id": episode.parent_episode_id,
                    "status": episode.status,
                    "work_item_id": item.id,
                },
            }
        if action == "trajectory.publish":
            revision = future.publish_trajectory_set(values)
            return {"trajectory_set_revision": revision.to_dict()}
        raise ValueError(f"unsupported future lab work action: {action}")

    def list_agent_episodes(
        self,
        *,
        environment_revision_id: Optional[str] = None,
        suite_revision_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._future_lab_engine().list_episodes(
            environment_revision_id=environment_revision_id,
            suite_revision_id=suite_revision_id,
            limit=limit,
            offset=offset,
        )

    def get_agent_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        value = self._future_lab_engine().get_episode(episode_id)
        return value.to_dict() if value else None

    def list_agent_episode_steps(
        self, episode_id: str, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._future_lab_engine().list_episode_steps(
            episode_id, limit=limit, offset=offset
        )

    def replay_agent_episode(self, episode_id: str) -> Dict[str, Any]:
        return self._future_lab_engine().replay_episode(episode_id)

    def compare_agent_environments(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        engine = self._future_lab_engine()
        base = engine.get_episode(str(payload.get("base_episode_id") or ""))
        candidate = engine.get_episode(
            str(payload.get("candidate_episode_id") or "")
        )
        if base is None or candidate is None:
            raise ValueError("base and candidate episodes are required")
        if base.suite_revision_id != candidate.suite_revision_id:
            raise ValueError(
                "environment comparisons require the same episode-suite revision"
            )
        return engine.compare_environment_subjects(
            suite_revision_id=base.suite_revision_id,
            base_subject_hash=base.subject_hash,
            candidate_subject_hash=candidate.subject_hash,
        ).to_dict()

    def publish_agent_trajectories(
        self, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        values = dict(payload)
        if not bool(values.get("review_confirmed")):
            return {
                "status": "review_required",
                "message": "Trajectories enter Review Studio before publication.",
                "primary_action": {
                    "id": "create_review_proposal",
                    "label": "Review trajectories",
                },
            }
        trajectory_set_id = str(values.get("trajectory_set_id") or uuid.uuid4().hex)
        values["trajectory_set_id"] = trajectory_set_id
        trajectory_set = self._future_lab_engine().prepare_trajectory_set(values)
        work_item = self._enqueue_future_lab_work(
            action="trajectory.publish",
            domain_kind="trajectory_set",
            domain_id=trajectory_set_id,
            payload=values,
        )
        trajectory_set = self._future_lab_engine().prepare_trajectory_set(
            values, work_item_id=work_item.id
        )
        return trajectory_set.to_dict()

    # ----- Lab v17: readiness, repair overlays, and support bundles ----

    def get_workstation_readiness(self) -> Dict[str, Any]:
        return self._product_lab_engine().assess_readiness().to_dict()

    def apply_setup_remediation(self, action: str) -> Dict[str, Any]:
        return self._product_lab_engine().apply_setup_remediation(action).to_dict()

    def get_distribution_capability(self) -> Dict[str, Any]:
        value = self._product_lab_engine().distribution_capability().to_dict()
        value["managed_runtimes"] = [
            item.to_dict() for item in self._managed_runtime_engine().capabilities()
        ]
        value["verified_training_backends"] = [
            item.accelerator_family
            for item in self._managed_runtime_engine().capabilities()
            if item.available
        ]
        return value

    def get_release_status(self) -> Dict[str, Any]:
        return self._product_lab_engine().release_status()

    def list_dataset_repairs(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._product_lab_engine().list_repair_sessions(
            limit=limit, offset=offset
        )

    def create_dataset_repair(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        session = self._product_lab_engine().create_repair_session(payload)
        return session.to_dict()

    def get_dataset_repair(self, session_id: str) -> Dict[str, Any]:
        return self._product_lab_engine().get_repair_session(session_id).to_dict()

    def list_dataset_repair_issues(
        self,
        session_id: str,
        *,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        return self._product_lab_engine().list_repair_issues(
            session_id,
            category=category,
            severity=severity,
            limit=limit,
            offset=offset,
        )

    def create_dataset_repair_plan(
        self, session_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self._product_lab_engine().create_repair_plan(
            session_id, payload
        ).to_dict()

    def get_dataset_repair_plan(self, revision_id: str) -> Dict[str, Any]:
        return self._product_lab_engine().get_repair_plan(revision_id).to_dict()

    def create_dataset_repair_preview(
        self, session_id: str, payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        revision_id = str(payload.get("plan_revision_id") or "").strip()
        if not revision_id:
            session = self._product_lab_engine().get_repair_session(session_id)
            revision_id = str(session.latest_plan_revision_id or "")
        if not revision_id:
            raise ValueError("A reviewed repair plan revision is required")
        return self._product_lab_engine().prepare_repair_preview(
            session_id, revision_id
        ).to_dict()

    def get_dataset_repair_preview(self, preview_id: str) -> Dict[str, Any]:
        return self._product_lab_engine().get_repair_preview(preview_id).to_dict()

    def publish_dataset_repair(self, preview_id: str) -> Dict[str, Any]:
        return self._product_lab_engine().publish_repair_revision(preview_id)

    def get_dataset_repair_revision(self, revision_id: str) -> Dict[str, Any]:
        return self._product_lab_engine().get_repair_revision(revision_id)

    def rebase_dataset_repair(self, session_id: str) -> Dict[str, Any]:
        return self._product_lab_engine().rebase_repair_session(session_id).to_dict()

    def cancel_dataset_repair(self, session_id: str) -> Dict[str, Any]:
        return self._product_lab_engine().cancel_repair(session_id).to_dict()

    def preview_support_bundle(
        self, categories: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        return self._product_lab_engine().support_bundle_preview(categories).to_dict()

    def create_support_bundle(
        self, categories: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        return self._product_lab_engine().create_support_bundle(categories).to_dict()

    def get_support_bundle(self, bundle_id: str) -> Dict[str, Any]:
        return self._product_lab_engine().get_support_bundle(bundle_id).to_dict()

    def verify_support_bundle(self, bundle_id: str) -> Dict[str, Any]:
        return self._product_lab_engine().verify_support_bundle(bundle_id)

    def delete_support_bundle(self, bundle_id: str) -> Dict[str, Any]:
        return {
            "id": bundle_id,
            "deleted": self._product_lab_engine().delete_support_bundle(bundle_id),
        }

    def qualify_distribution(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return self._product_lab_engine().request_release_qualification(payload).to_dict()

    def list_distribution_qualifications(
        self, *, limit: int = 100, offset: int = 0
    ) -> Dict[str, Any]:
        return self._product_lab_engine().list_release_qualifications(
            limit=limit, offset=offset
        )

    def get_distribution_qualification(self, qualification_id: str) -> Dict[str, Any]:
        return self._product_lab_engine().get_release_qualification(
            qualification_id
        ).to_dict()

    def cancel_distribution_qualification(self, qualification_id: str) -> Dict[str, Any]:
        return self._product_lab_engine().cancel_release_qualification(
            qualification_id
        ).to_dict()

    def execute_product_v17_work_item(self, item: Any) -> Dict[str, Any]:
        return self._product_lab_engine().execute_work_item(item)

    def _to_active_row(self, item: TrainingRunListItemView) -> ActiveRunRowView:
        return ActiveRunRowView(
            id=item.id,
            modality=item.modality,
            model_name=item.model_name,
            status=item.status,
            headline=item.headline,
            next_step=item.next_step,
            primary_action=item.primary_action,
            metrics_summary=item.metrics_summary,
        )

    @staticmethod
    def _isoformat(value: datetime) -> str:
        if value.tzinfo is None:
            value = value.astimezone()
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
    def _value_or_default(value: Any, default: Any) -> Any:
        if value in (None, ""):
            return default
        return value

    @staticmethod
    def _optional_str(value: Any) -> Optional[str]:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _has_public_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value in (None, ""):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Phase D helpers — used by `_summary_to_detail_view` to surface chart-ready
# per-cycle data without leaking the entire training_summary.json across the
# wire. Defined at module scope so tests can exercise it without standing up
# the full PublicApiService.
# ---------------------------------------------------------------------------


def _project_run_cost(raw_data: Dict[str, Any], *, backend_name: str) -> Dict[str, Any]:
    """Roll up wall-clock + nominal-power into a cost estimate (Track P2).

    Sums `cycle_duration_seconds` across the run's cycles and hands them
    to `telemetry.cost.estimate_run_cost`. The backend name comes from
    the *currently active host* — the training_summary doesn't carry the
    backend at write time, so for completed runs displayed on a different
    host the cost is "what would this run cost *here*". Same-host case
    is accurate; cross-host is an honest estimate. The frontend renders
    the `source` field so users know it's an estimate, not a meter
    reading.
    """
    from halo_forge.telemetry.cost import estimate_run_cost

    duration = 0.0
    if isinstance(raw_data, dict):
        cycles = raw_data.get("cycles")
        if isinstance(cycles, list):
            for entry in cycles:
                if isinstance(entry, dict):
                    v = _coerce_optional_float(entry.get("cycle_duration_seconds"))
                    if v:
                        duration += v
    cost = estimate_run_cost(
        duration_seconds=duration,
        backend_name=backend_name or "unknown",
    )
    return cost.to_dict()


def _project_cycles_for_charts(raw_data: Dict[str, Any]) -> list[dict[str, Any]]:
    """Project the cycles array from a training_summary.json payload to a
    flat plot-friendly shape.

    The raw `cycles` list contains everything the trainer emitted, including
    `yield_diagnostics` sub-objects, which are not useful for charts and
    inflate the wire size. We extract just the scalar per-cycle metrics
    that the live run view actually charts: train/eval loss, reward
    averages, success rate, and sample counts.

    Tolerates missing fields (older trainers, partial summaries) by
    returning None for any absent value — the frontend renders gaps as
    breaks in the line, not as zeros.
    """
    if not isinstance(raw_data, dict):
        return []
    cycles = raw_data.get("cycles")
    if not isinstance(cycles, list):
        return []
    projected: list[dict[str, Any]] = []
    for entry in cycles:
        if not isinstance(entry, dict):
            continue
        projected.append(
            {
                "cycle": int(entry.get("cycle") or 0),
                "train_loss": _coerce_optional_float(entry.get("train_loss")),
                "initial_train_loss": _coerce_optional_float(entry.get("initial_train_loss")),
                "eval_loss": _coerce_optional_float(entry.get("eval_loss")),
                "avg_reward": _coerce_optional_float(entry.get("avg_reward")),
                "avg_kept_reward": _coerce_optional_float(entry.get("avg_kept_reward")),
                "success_rate": _coerce_optional_float(entry.get("success_rate")),
                "samples_seen": _coerce_optional_int(entry.get("samples_seen")),
                "samples_kept": _coerce_optional_int(entry.get("samples_kept")),
                "train_steps_executed": _coerce_optional_int(entry.get("train_steps_executed")),
                "cycle_duration_seconds": _coerce_optional_float(
                    entry.get("cycle_duration_seconds")
                ),
                "learning_rate": _coerce_optional_float(entry.get("learning_rate")),
            }
        )
    return projected


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if not (result != result) else None  # filter NaN


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_run_root() -> Path:
    configured = str(os.environ.get(DEFAULT_RUN_ROOT_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".halo-forge" / "runs").expanduser().resolve()


def _default_log_root() -> Path:
    configured = str(os.environ.get("HALO_FORGE_LOG_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".halo-forge" / "logs").expanduser().resolve()


def _candidate_log_roots(base_path: Path) -> list[Path]:
    roots = [(base_path / "logs").expanduser().resolve()]
    app_logs = _default_log_root()
    if app_logs not in roots:
        roots.append(app_logs)
    return roots


def _classify_training_failure(text: str, *, status: str = "failed") -> Dict[str, str]:
    lower = str(text or "").lower()
    if str(status or "").lower() in {"stopped", "cancelled", "canceled"}:
        return {
            "kind": "cancelled",
            "headline": "Run was cancelled",
            "message": "Halo Forge stopped this run before it completed.",
            "next_action": "Review the log tail, then retry from Train if the configuration is still useful.",
            "docs_url": "/docs",
        }
    if any(
        marker in lower
        for marker in (
            "multiprocessing.resource_tracker",
            "from multiprocessing.resource_tracker import main",
            "argument command: invalid choice",
            "invalid choice:",
        )
    ):
        return {
            "kind": "runtime_packaging",
            "headline": "Desktop runtime could not start training",
            "message": "The packaged app routed a Python helper process into the Halo Forge CLI before training could begin.",
            "next_action": "Install the latest desktop build, then retry from Start or Train.",
            "docs_url": "/docs",
        }
    if "read-only file system" in lower and ("logs" in lower or "cwd" in lower):
        return {
            "kind": "logging_cwd",
            "headline": "Run could not create its log file",
            "message": "The trainer started from a folder where the app could not create a logs directory.",
            "next_action": "Update to the latest app build and retry the run; logs should now write under ~/.halo-forge/logs.",
            "docs_url": "/docs",
        }
    if any(
        marker in lower
        for marker in (
            "permission denied",
            "not writable",
            "cannot be created",
            "read-only file system",
        )
    ):
        return {
            "kind": "unwritable_output",
            "headline": "Halo Forge could not write to the output folder",
            "message": "The selected output or working directory is not writable from this app runtime.",
            "next_action": "Retry from Train so Halo Forge can use the default writable run folder under ~/.halo-forge/runs.",
            "docs_url": "/docs",
        }
    if _looks_like_gated_hf_error(text):
        return {
            "kind": "gated_huggingface",
            "headline": "Model requires Hugging Face access",
            "message": GATED_MODEL_MESSAGE,
            "next_action": "Connect Hugging Face, accept the model terms on Hugging Face, or choose an open model.",
            "docs_url": "/docs",
        }
    if any(
        marker in lower
        for marker in ("no such file or directory", "file not found", "does not exist")
    ) and any(token in lower for token in ("dataset", "prompt", "prompts", ".jsonl")):
        return {
            "kind": "missing_data",
            "headline": "Training data was not found",
            "message": "The selected dataset or prompt file could not be resolved by the trainer.",
            "next_action": "Choose a built-in dataset/template or provide a local JSONL path that exists on this workstation.",
            "docs_url": "/docs",
        }
    if any(
        marker in lower
        for marker in (
            "module not found",
            "modulenotfounderror",
            "importerror",
            "command not found",
            "no module named",
        )
    ):
        return {
            "kind": "missing_dependency",
            "headline": "Runtime dependency is missing",
            "message": "The selected method needs a package or command that is not available in this runtime.",
            "next_action": "Review Diagnostics and method docs, then install the missing dependency or choose a supported method.",
            "docs_url": "/docs",
        }
    if any(
        marker in lower
        for marker in ("out of memory", "oom", "mps backend out of memory", "cuda out of memory")
    ):
        return {
            "kind": "out_of_memory",
            "headline": "Run ran out of memory",
            "message": "The model or batch settings exceeded the available workstation memory.",
            "next_action": "Retry with a smaller model, fewer samples, lower batch size, or MLX-friendly settings on Apple Silicon.",
            "docs_url": "/docs",
        }
    if any(
        marker in lower
        for marker in (
            "unsupported backend",
            "not supported on",
            "rollout-gated",
            "capability is still gated",
        )
    ):
        return {
            "kind": "backend_unsupported",
            "headline": "This method is not supported by the current backend",
            "message": "The selected method/model combination is gated or unsupported on this workstation runtime.",
            "next_action": "Choose a supported model/method, or enable the prototype gate only for experimental runs.",
            "docs_url": "/docs",
        }
    if any(
        marker in lower
        for marker in (
            "401 client error",
            "403 client error",
            "unauthorized",
            "token",
            "authentication",
        )
    ):
        return {
            "kind": "model_auth",
            "headline": "Model download needs authentication",
            "message": "The model download failed because the upstream repository requires credentials or access.",
            "next_action": "Connect Hugging Face or choose an open model from Models.",
            "docs_url": "/docs",
        }
    return {
        "kind": "unknown",
        "headline": "Run failed before producing training metrics",
        "message": "Halo Forge could not classify this failure from the available logs.",
        "next_action": "Open Diagnostics, inspect the log tail below, then retry from Train with a known-good preset.",
        "docs_url": "/docs",
    }


def _friendly_upstream_error(detail: Any) -> Dict[str, Any]:
    text = _error_text(detail)
    if _looks_like_gated_hf_error(text):
        result: Dict[str, Any] = {
            "error_kind": "gated_model",
            "message": GATED_MODEL_MESSAGE,
            "action": GATED_MODEL_ACTION,
        }
        model_id = _extract_model_id(detail)
        if model_id:
            result["model_id"] = model_id
            result["model_url"] = f"https://huggingface.co/{model_id}"
        return result
    if text:
        return {"message": _one_line_error(text)}
    return {
        "message": "The local model server returned an error. Check the serve logs for details."
    }


def _extract_model_id(detail: Any) -> str | None:
    if isinstance(detail, dict):
        for key in ("model_id", "model", "repo_id"):
            value = detail.get(key)
            if isinstance(value, str) and "/" in value:
                return value.strip()
        for value in detail.values():
            found = _extract_model_id(value)
            if found:
                return found
    if isinstance(detail, list):
        for item in detail:
            found = _extract_model_id(item)
            if found:
                return found
    return None


def _error_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        parts: list[str] = []
        for key in ("message", "detail", "error", "error_description"):
            value = detail.get(key)
            if value is not None:
                parts.append(_error_text(value))
        if not parts:
            parts.extend(_error_text(value) for value in detail.values())
        return " ".join(part for part in parts if part)
    if isinstance(detail, list):
        return " ".join(_error_text(item) for item in detail)
    return str(detail or "")


def _looks_like_gated_hf_error(text: str) -> bool:
    lower = text.lower()
    gated_markers = (
        "gated repo",
        "cannot access gated repo",
        "restricted",
        "please log in",
        "401 client error",
        "repository not found",
        "private repository",
    )
    return any(marker in lower for marker in gated_markers) and (
        "huggingface.co" in lower or "hugging face" in lower or "hf.co" in lower or "repo" in lower
    )


def _one_line_error(text: str) -> str:
    line = " ".join(part.strip() for part in str(text).splitlines() if part.strip())
    return line[:500] if len(line) > 500 else line


def _extract_output_dir_and_run_id(
    source: Dict[str, Any],
) -> tuple[Optional["Path"], str]:
    """Normalize the (output_dir, run_id) pair across the two run-source
    flavors `_resolve_run_source` returns.

    For active jobs the output_dir lives on `job.output_dir` and the
    identifier is `job.id`. For completed summaries it's
    `summary.output_dir` (already a Path) and `summary.run_id` (or the
    summary id when run_id wasn't recorded). Both are mapped to the
    same shape so the logs/samples endpoints can read uniformly.
    """
    from pathlib import Path

    kind = source.get("kind")
    if kind == "job":
        job = source.get("job")
        out = getattr(job, "output_dir", None)
        out_path = Path(out) if out else None
        run_id = str(getattr(job, "id", "") or "")
        return out_path, run_id

    if kind == "summary":
        summary = source.get("summary")
        out = getattr(summary, "output_dir", None)
        out_path = Path(out) if out else None
        run_id = str(getattr(summary, "run_id", "") or getattr(summary, "id", "") or "")
        return out_path, run_id

    return None, ""
