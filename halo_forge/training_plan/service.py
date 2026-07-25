"""Deterministic V18 training-plan and workstation-capacity service.

The service owns recommendation and durable evidence identity.  Actual trainer
implementations remain unchanged; capacity adapters exercise a disposable
trainer-shaped allocation through an injectable runner so hardware tests can
use the real backend while unit tests stay offline and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from halo_forge.models.catalog import CATALOG_VERSION, get_model, recommended_models
from halo_forge.run_db import RunDatabase
from halo_forge.workstation_jobs.resources import sample_workstation_capacity

from .models import (
    ModelPreparation,
    TrainingCapacityAttempt,
    TrainingCapacityCapability,
    TrainingCapacityCheck,
    TrainingPlan,
    TrainingPlanDecision,
    TrainingPlanProfile,
    TrainingPlanReadiness,
    TrainingPlanReason,
    TrainingPlanRecommendation,
    TrainingPlanRevision,
    TrainingResourceForecast,
)


PROFILE_VERSION = "1"
CAPACITY_VERSION = "1"
GIB = 1024**3


class TrainingPlanError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(prefix: str, value: Any) -> str:
    return f"{prefix}-{_hash(value)[:24]}"


def _backend_name() -> str:
    try:
        from halo_forge.backend import get_backend

        return str(get_backend().name).lower()
    except Exception:
        return str(os.environ.get("HALOFORGE_BACKEND") or "cpu").lower()


_PROFILE_DEFINITIONS: tuple[TrainingPlanProfile, ...] = (
    TrainingPlanProfile("sft_first_proof", PROFILE_VERSION, "Instruction tuning", "sft", ("sft", "chat", "tool"), 200, 1, 1, 1, 8, 2e-4, 1024, "lora", "bf16"),
    TrainingPlanProfile("preference_first_proof", PROFILE_VERSION, "Preference tuning", "dpo", ("preference",), 200, 1, 1, 1, 8, 5e-6, 1024, "lora", "bf16"),
    TrainingPlanProfile("orpo_first_proof", PROFILE_VERSION, "Preference tuning", "orpo", ("preference",), 200, 1, 1, 1, 8, 8e-6, 1024, "lora", "bf16"),
    TrainingPlanProfile("rm_first_proof", PROFILE_VERSION, "Reward scoring", "rm", ("preference",), 200, 1, 1, 2, 4, 1e-5, 1024, "lora", "bf16"),
    TrainingPlanProfile("raft_first_proof", PROFILE_VERSION, "Verifier-guided improvement", "raft", ("prompt", "rlvr"), 200, 1, 1, 1, 8, 2e-5, 1024, "lora", "bf16"),
    TrainingPlanProfile("grpo_first_proof", PROFILE_VERSION, "Verifier-guided optimization", "grpo", ("prompt", "rlvr"), 200, 1, 1, 1, 8, 1e-6, 1024, "lora", "bf16"),
    TrainingPlanProfile("cpt_first_proof", PROFILE_VERSION, "Continue learning from documents", "cpt", ("corpus",), 200, 1, 1, 1, 8, 2e-5, 2048, "lora", "bf16"),
    TrainingPlanProfile("reasoning_first_proof", PROFILE_VERSION, "Worked reasoning tuning", "reasoning", ("sft", "prompt"), 200, 1, 1, 1, 8, 2e-5, 1024, "lora", "bf16"),
    TrainingPlanProfile("agentic_first_proof", PROFILE_VERSION, "Tool-use tuning", "agentic", ("tool",), 200, 1, 1, 1, 8, 2e-5, 1024, "lora", "bf16"),
    TrainingPlanProfile("vlm_first_proof", PROFILE_VERSION, "Visual instruction tuning", "vlm", ("vlm",), 50, 1, 1, 1, 8, 2e-5, 512, "lora", "bf16"),
    TrainingPlanProfile("audio_first_proof", PROFILE_VERSION, "Speech recognition tuning", "audio", ("audio",), 50, 1, 1, 1, 8, 1e-5, 448, "lora", "bf16"),
    TrainingPlanProfile("classify_first_proof", PROFILE_VERSION, "Classification", "classify", ("classification",), 200, 1, 1, 4, 1, 2e-5, 512, "head", "fp32"),
    TrainingPlanProfile("embed_first_proof", PROFILE_VERSION, "Semantic search", "embed", ("embedding",), 200, 1, 1, 8, 1, 2e-5, 512, "full", "fp32"),
    TrainingPlanProfile("rerank_first_proof", PROFILE_VERSION, "Search reranking", "rerank", ("reranking",), 200, 1, 1, 8, 1, 2e-5, 512, "full", "fp32"),
)


_CAPACITY_MODES = tuple(profile.trainer_mode for profile in _PROFILE_DEFINITIONS)


def training_plan_profiles() -> tuple[TrainingPlanProfile, ...]:
    return _PROFILE_DEFINITIONS


def training_capacity_capabilities() -> tuple[TrainingCapacityCapability, ...]:
    backends = ("cpu", "cuda", "rocm", "rocm_gfx1151", "mps", "mlx")
    return tuple(
        TrainingCapacityCapability(
            id=f"{mode}_capacity_v1",
            version=CAPACITY_VERSION,
            trainer_mode=mode,
            backends=backends,
            scratch_step=True,
            fallback_steps=("confirmed", "gradient_checkpointing", "microbatch_one"),
        )
        for mode in _CAPACITY_MODES
    )


class TrainingPlanService:
    def __init__(
        self,
        database: RunDatabase,
        *,
        root: str | Path | None = None,
        scheduler: Any = None,
        capacity_sampler: Callable[..., Any] = sample_workstation_capacity,
        probe_runner: Optional[Callable[[Mapping[str, Any], Path], Mapping[str, Any]]] = None,
    ):
        self.database = database
        self.conn = database._conn
        self.root = Path(root or Path.home() / ".halo-forge").expanduser()
        self.preparation_root = self.root / "models" / "preparations"
        self.scratch_root = self.root / "training-capacity" / "scratch"
        self.runtime_root = Path(
            os.environ.get("HALOFORGE_RUNTIME_ROOT") or self.root / "runtimes"
        ).expanduser().resolve()
        self.scheduler = scheduler
        self.capacity_sampler = capacity_sampler
        self.probe_runner = probe_runner or self._trainer_probe

    # ---- registries and recommendation ---------------------------------

    def capabilities(self) -> Dict[str, Any]:
        return {
            "profile_version": PROFILE_VERSION,
            "capacity_version": CAPACITY_VERSION,
            "profiles": [value.to_dict() for value in training_plan_profiles()],
            "capacity_adapters": [value.to_dict() for value in training_capacity_capabilities()],
            "fallback_lattice": ["confirmed", "gradient_checkpointing", "microbatch_one"],
        }

    def _runtime(self) -> tuple[str, str, Dict[str, Any]]:
        backend = _backend_name()
        capacity = self.capacity_sampler(self.root)
        details = capacity.to_dict()
        identity = {
            "platform": platform.system().lower(),
            "architecture": platform.machine().lower(),
            "python": platform.python_version(),
            "backend": backend,
            "device": (details.get("accelerator") or {}).get("device_name"),
            "memory_total": (details.get("memory") or {}).get("total_bytes"),
            "device_memory_total": (details.get("accelerator") or {}).get("device_memory_total_bytes"),
        }
        return backend, _hash(identity), details

    def _managed_runtime(
        self, backend: str, requested_revision_id: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        family = "rocm" if backend.startswith("rocm") else "cuda" if backend == "cuda" else None
        if family is None:
            if requested_revision_id:
                raise TrainingPlanError(
                    "The selected managed accelerator runtime contradicts the active backend"
                )
            return None, None
        from halo_forge.managed_runtime import ManagedRuntimeService

        service = ManagedRuntimeService(
            self.database,
            root=self.runtime_root,
            scheduler=self.scheduler,
        )
        candidates = [
            capability
            for capability in service.capabilities()
            if capability.accelerator_family == family
        ]
        capability = candidates[0] if candidates else None
        if requested_revision_id:
            revision = service.get_revision(requested_revision_id)
            if revision is None:
                raise TrainingPlanError("The selected managed runtime revision does not exist")
            profile = service.get_profile(revision.profile_id)
            if profile is None or profile.accelerator_family != family:
                raise TrainingPlanError("The selected managed runtime does not match this accelerator")
            qualification = service.latest_qualification(revision.id)
            if qualification is None or not service.verify(qualification.id)["valid"]:
                raise TrainingPlanError("The selected managed runtime is not currently qualified")
            return revision.id, qualification.runtime_identity_hash
        if capability is None or not capability.available or not capability.runtime_revision_id:
            label = "AMD" if family == "rocm" else "NVIDIA"
            raise TrainingPlanError(
                f"Prepare and qualify the managed {label} training runtime before creating a guided plan"
            )
        qualification = service.get_qualification(capability.qualification_id)
        assert qualification is not None
        return capability.runtime_revision_id, qualification.runtime_identity_hash

    def _managed_runtime_service(self):
        from halo_forge.managed_runtime import ManagedRuntimeService

        return ManagedRuntimeService(
            self.database,
            root=self.runtime_root,
            scheduler=self.scheduler,
        )

    @staticmethod
    def _canonical_shape(version: Any) -> str:
        recipe = dict(version.recipe or {})
        return str(recipe.get("schema") or recipe.get("canonical_schema") or "sft").lower()

    @staticmethod
    def _scenario_revision(version: Any) -> Optional[str]:
        recipe = dict(version.recipe or {})
        for step in recipe.get("steps") or []:
            if isinstance(step, Mapping) and str(step.get("kind") or "").lower() == "map":
                value = str(step.get("scenario_revision_id") or "").strip()
                if value:
                    return value
        value = str(recipe.get("scenario_revision_id") or "").strip()
        return value or None

    def _profile(self, *, shape: str, requested_mode: Optional[str]) -> TrainingPlanProfile:
        values = list(_PROFILE_DEFINITIONS)
        if requested_mode:
            values = [value for value in values if value.trainer_mode == requested_mode]
        values = [value for value in values if shape in value.canonical_shapes]
        if not values and shape == "chat":
            values = [value for value in _PROFILE_DEFINITIONS if value.trainer_mode == "sft"]
        if not values:
            raise TrainingPlanError(f"No verified guided training plan supports {shape!r} data")
        return values[0]

    @staticmethod
    def _memory_available(runtime: Mapping[str, Any]) -> Optional[int]:
        accelerator = dict(runtime.get("accelerator") or {})
        total = accelerator.get("device_memory_total_bytes")
        used = accelerator.get("device_memory_used_bytes")
        if total is not None:
            return int(total) - int(used or 0)
        memory = dict(runtime.get("memory") or {})
        value = memory.get("available_bytes")
        return int(value) if value is not None else None

    def _models(
        self,
        profile: TrainingPlanProfile,
        backend: str,
        runtime: Mapping[str, Any],
        *,
        modality: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        catalog_modality = {"image": "vision"}.get(str(modality or "").lower(), modality)
        candidates = recommended_models(
            mode=profile.trainer_mode,
            backend=backend,
            modality=catalog_modality,
        )
        available = self._memory_available(runtime)
        safe = [
            value for value in candidates
            if str(value.get("risk_level") or "safe") != "experimental"
            and (
                available is None
                or value.get("estimated_memory_gb") is None
                or float(value["estimated_memory_gb"]) * GIB <= available * 0.8
            )
        ]
        values = safe or [value for value in candidates if str(value.get("risk_level") or "") != "experimental"]
        return sorted(
            values,
            key=lambda value: (
                not bool(value.get("recommended_first_run")),
                float(value.get("estimated_memory_gb") or float("inf")),
                str(value.get("id") or ""),
            ),
        )

    @staticmethod
    def _dataset_shape(version: Any) -> Dict[str, Any]:
        statistics = dict(version.statistics or {})
        token_stats = dict(
            statistics.get("tokens")
            or statistics.get("token_statistics")
            or statistics.get("text")
            or {}
        )
        media_stats = dict(
            statistics.get("media")
            or statistics.get("image")
            or statistics.get("audio")
            or {}
        )
        label_stats = dict(
            statistics.get("labels")
            or statistics.get("label_statistics")
            or {}
        )
        return {
            "statistics_hash": _hash(statistics),
            "row_count": int(version.row_count or 0),
            "split_counts": dict(version.split_counts or {}),
            "token_p50": token_stats.get("p50"),
            "token_p95": token_stats.get("p95"),
            "token_p99": token_stats.get("p99"),
            "token_max": token_stats.get("max"),
            "label_count": label_stats.get("count") or label_stats.get("class_count"),
            "media_dimensions": {
                key: media_stats.get(key)
                for key in (
                    "width_p95",
                    "height_p95",
                    "duration_p95",
                    "sample_rate",
                    "channels",
                )
                if media_stats.get(key) is not None
            },
        }

    @staticmethod
    def _sequence_length(profile: TrainingPlanProfile, data_shape: Mapping[str, Any]) -> int:
        observed = data_shape.get("token_p99") or data_shape.get("token_max")
        try:
            target = max(128, int(float(observed) * 1.10))
        except (TypeError, ValueError):
            return profile.max_sequence_length
        powers = (128, 256, 512, 1024, 2048, 4096, 8192)
        selected = next((value for value in powers if value >= target), powers[-1])
        return min(profile.max_sequence_length, selected)

    @staticmethod
    def _forecast(model: Mapping[str, Any], profile: TrainingPlanProfile, version: Any) -> TrainingResourceForecast:
        estimated_memory = model.get("estimated_memory_gb")
        peak = int(float(estimated_memory) * GIB) if estimated_memory is not None else None
        if peak is not None and profile.trainer_mode in {"dpo", "grpo"}:
            peak = int(peak * 1.7)
        parameter_label = str(model.get("parameter_count") or "")
        params_b = 0.0
        try:
            params_b = float(parameter_label[:-1]) if parameter_label.upper().endswith("B") else float(parameter_label[:-1]) / 1000 if parameter_label.upper().endswith("M") else 0.0
        except ValueError:
            params_b = 0.0
        download = int(params_b * 2 * GIB) if params_b else None
        checkpoint = int(params_b * (0.08 if profile.adaptation == "lora" else 2.0) * GIB) if params_b else None
        rows = min(int(version.split_counts.get("train", version.row_count) or 0), profile.proof_max_samples)
        proof_low = max(15, rows // max(1, profile.microbatch) * 2)
        proof_high = max(60, proof_low * 5)
        full_factor = max(1.0, float(version.row_count or rows or 1) / max(1, rows))
        return TrainingResourceForecast(
            download_bytes=download,
            scratch_bytes=(checkpoint or 0) + (peak or 0) if checkpoint is not None or peak is not None else None,
            checkpoint_bytes=checkpoint,
            peak_memory_bytes=peak,
            proof_seconds_range=(proof_low, proof_high),
            full_run_seconds_range=(int(proof_low * full_factor), int(proof_high * full_factor)),
            provenance={
                "download_bytes": "estimated",
                "scratch_bytes": "estimated",
                "checkpoint_bytes": "estimated",
                "peak_memory_bytes": "estimated",
                "proof_seconds_range": "estimated",
                "full_run_seconds_range": "estimated",
            },
            confidence="low",
        )

    def recommend(self, payload: Mapping[str, Any]) -> TrainingPlanRecommendation:
        version_id = str(payload.get("dataset_version_id") or "").strip()
        version = self.database.get_dataset_version(version_id)
        if version is None:
            raise KeyError(version_id)
        if version.status != "completed":
            raise TrainingPlanError("The dataset version must be complete before planning")
        if int(version.split_counts.get("train", 0) or 0) <= 0:
            raise TrainingPlanError("The dataset version has no non-empty training split")
        shape = self._canonical_shape(version)
        profile = self._profile(
            shape=shape,
            requested_mode=str(payload.get("trainer_mode") or "").lower() or None,
        )
        backend, runtime_hash, runtime = self._runtime()
        runtime_revision_id, managed_runtime_hash = self._managed_runtime(
            backend,
            str(payload.get("runtime_profile_revision_id") or "").strip() or None,
        )
        if managed_runtime_hash:
            runtime_hash = managed_runtime_hash
        dataset = self.database.get_dataset(version.dataset_id)
        modality = dataset.modality if dataset is not None else None
        models = self._models(profile, backend, runtime, modality=modality)
        requested_model = str(payload.get("model") or "").strip()
        if requested_model:
            selected = get_model(requested_model)
            if selected is None or requested_model not in {value["id"] for value in models}:
                raise TrainingPlanError("The requested model is not a safe verified fit for this plan")
        elif models:
            selected = models[0]
        else:
            raise TrainingPlanError("No verified catalog model fits the active backend and memory")
        scenario = str(payload.get("scenario_revision_id") or self._scenario_revision(version) or "").strip() or None
        training_path_revision_id: Optional[str] = None
        training_path_certification_id: Optional[str] = None
        if runtime_revision_id:
            from halo_forge.training_path_certification import TrainingPathCertificationService

            runtime_engine = self._managed_runtime_service()
            runtime_revision = runtime_engine.get_revision(runtime_revision_id)
            runtime_profile = runtime_engine.get_profile(runtime_revision.profile_id) if runtime_revision else None
            if runtime_profile is None:
                raise TrainingPlanError("The managed runtime profile is missing")
            path_engine = TrainingPathCertificationService(
                self.database,
                runtime_service=runtime_engine,
                scheduler=self.scheduler,
            )
            matrix = path_engine.capabilities(runtime_profile.accelerator_family)
            matching_paths = [
                value
                for value in matrix.paths
                if value.trainer_mode == profile.trainer_mode
                and value.model_id == selected["id"]
                and (
                    not scenario
                    or not value.scenario_revision_id
                    or value.scenario_revision_id == scenario
                )
            ]
            if not matching_paths:
                raise TrainingPlanError(
                    "This exact model and trainer path has no certification profile yet"
                )
            path_capability = matching_paths[0]
            if path_capability.state == "unavailable":
                raise TrainingPlanError(
                    path_capability.blocker
                    or "This exact trainer path is not available for guided use"
                )
            training_path_revision_id = path_capability.path_revision_id
            training_path_certification_id = (
                path_capability.certification_id
                if path_capability.state == "path_verified"
                else None
            )
        proof_count = min(profile.proof_max_samples, int(version.split_counts.get("train", version.row_count)))
        data_shape = self._dataset_shape(version)
        sequence_length = self._sequence_length(profile, data_shape)
        definition = {
            "format_version": 1,
            "recommended": not bool(requested_model),
            "dataset_version_id": version.id,
            "dataset_content_hash": version.content_hash,
            "dataset_split": "train",
            "validation_split": "validation" if version.split_counts.get("validation") else "val" if version.split_counts.get("val") else None,
            "scenario_revision_id": scenario,
            "trainer_mode": profile.trainer_mode,
            "backend": backend,
            "model": selected["id"],
            "model_revision": str(payload.get("model_revision") or "main"),
            "catalog_version": CATALOG_VERSION,
            "adaptation": profile.adaptation,
            "precision": profile.precision,
            "max_sequence_length": sequence_length,
            "batch_size": profile.microbatch,
            "gradient_accumulation_steps": profile.gradient_accumulation,
            "effective_batch_size": profile.microbatch * profile.gradient_accumulation,
            "learning_rate": profile.learning_rate,
            "gradient_checkpointing": False,
            "reference_model": (
                selected["id"] if profile.trainer_mode in {"dpo", "grpo"} else None
            ),
            "reference_model_revision": (
                str(payload.get("model_revision") or "main")
                if profile.trainer_mode in {"dpo", "grpo"}
                else None
            ),
            "epochs": profile.epochs,
            "cycles": profile.cycles,
            "max_samples": proof_count,
            "limit": proof_count,
            "seed": 42,
            "proof_run": True,
            "modality": modality,
            "dataset_shape": data_shape,
            "model_access": {
                "license_note": selected.get("license_note"),
                "license_url": selected.get("license_url"),
                "download_note": selected.get("download_note"),
                "trust_remote_code_required": bool(selected.get("trust_remote_code_required")),
            },
            "expected_artifacts": {
                "kind": (
                    "task_model"
                    if profile.trainer_mode in {"classify", "embed", "rerank"}
                    else "adapter" if profile.adaptation in {"lora", "head"} else "model"
                ),
                "requires_tokenizer": modality in {None, "text", "code"} or profile.trainer_mode == "vlm",
                "requires_processor": modality in {"image", "vision", "audio"} or profile.trainer_mode in {"vlm", "audio"},
            },
            "verifier_profile_revision_id": payload.get("verifier_profile_revision_id"),
            "reward_system_revision_id": payload.get("reward_system_revision_id"),
            "runtime": {
                "platform": platform.system().lower(),
                "architecture": platform.machine().lower(),
                "backend": backend,
                "runtime_profile_revision_id": runtime_revision_id,
            },
            "runtime_profile_revision_id": runtime_revision_id,
            "training_path_revision_id": training_path_revision_id,
            "training_path_certification_id": training_path_certification_id,
        }
        if profile.trainer_mode == "cpt":
            definition.update(
                {
                    "packing": "paragraph_eos_non_overlap_v1",
                    "budget_mode": "passes",
                    "corpus_passes": 1.0,
                    "max_steps": 1,
                }
            )
        compute_shape = {
            key: definition.get(key)
            for key in (
                "trainer_mode", "backend", "model", "model_revision", "adaptation",
                "precision", "max_sequence_length", "batch_size",
                "gradient_accumulation_steps", "effective_batch_size",
                "verifier_profile_revision_id", "reward_system_revision_id",
                "dataset_shape",
                "reference_model", "reference_model_revision",
                "runtime_profile_revision_id",
                "training_path_revision_id", "training_path_certification_id",
            )
        }
        compute_hash = _hash(compute_shape)
        forecast = self._forecast(selected, profile, version)
        reasons = (
            TrainingPlanReason("verified_fit", f"{selected.get('label') or selected['id']} is a verified fit", "It supports the selected training approach and active backend."),
            TrainingPlanReason("memory_headroom", "Chosen for workstation fit", "The catalog estimate stays within the workstation safety margin; the capacity check will measure the real path."),
            TrainingPlanReason("bounded_proof", f"Uses up to {proof_count} training records", "The proof run preserves validation data and limits cost before a full run."),
            TrainingPlanReason("data_shape", f"Uses a {sequence_length}-token maximum for this proof", "The limit comes from the immutable dataset profile and is rechecked against median- and maximum-cost batches."),
            *(
                (TrainingPlanReason("reference_model", "Includes the comparison model", "The capacity forecast includes the reference-model path required by this objective."),)
                if profile.trainer_mode in {"dpo", "grpo"}
                else ()
            ),
        )
        identity = {
            "dataset_version_id": version.id,
            "scenario_revision_id": scenario,
            "profile": [profile.id, profile.version],
            "definition": definition,
            "compute_shape_hash": compute_hash,
            "runtime_hash": runtime_hash,
            "runtime_profile_revision_id": runtime_revision_id,
            "training_path_revision_id": training_path_revision_id,
            "training_path_certification_id": training_path_certification_id,
        }
        plan_id = str(payload.get("plan_id") or _identifier("plan", {"dataset": version.id, "scenario": scenario, "mode": profile.trainer_mode}))
        existing = self.conn.execute("SELECT * FROM training_plans WHERE id=?", (plan_id,)).fetchone()
        now = _now()
        if existing is None:
            self.conn.execute(
                "INSERT INTO training_plans (id,dataset_version_id,scenario_revision_id,status,created_at,updated_at) VALUES (?,?,?,?,?,?)",
                (plan_id, version.id, scenario, "recommended", now, now),
            )
        revision_number = int(self.conn.execute("SELECT COALESCE(MAX(revision_number),0)+1 FROM training_plan_revisions WHERE plan_id=?", (plan_id,)).fetchone()[0])
        content_hash = _hash(identity)
        prior = self.conn.execute("SELECT id FROM training_plan_revisions WHERE content_hash=?", (content_hash,)).fetchone()
        revision_id = str(prior["id"]) if prior else _identifier("planrev", identity)
        if prior is None:
            self.conn.execute(
                """INSERT INTO training_plan_revisions
                   (id,plan_id,revision_number,status,content_hash,profile_id,profile_version,
                    dataset_version_id,scenario_revision_id,trainer_mode,backend,model_id,
                    model_revision,resolved_model_commit,definition_json,reasons_json,
                    forecast_json,compute_shape_hash,runtime_hash,runtime_profile_revision_id,
                    training_path_revision_id,training_path_certification_id,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    revision_id, plan_id, revision_number, "draft", content_hash,
                    profile.id, profile.version, version.id, scenario, profile.trainer_mode,
                    backend, selected["id"], definition["model_revision"], None,
                    _json(definition), _json([value.to_dict() for value in reasons]),
                    _json(forecast.to_dict()), compute_hash, runtime_hash,
                    runtime_revision_id, training_path_revision_id,
                    training_path_certification_id, now,
                ),
            )
        self.conn.execute("UPDATE training_plans SET latest_revision_id=?,status='recommended',updated_at=? WHERE id=?", (revision_id, now, plan_id))
        self.conn.commit()
        revision = self.get_revision(revision_id)
        plan = self.get_plan(plan_id)
        assert revision and plan
        alternatives = tuple(
            {
                "model_id": value["id"],
                "label": value.get("label") or value["id"],
                "estimated_memory_gb": value.get("estimated_memory_gb"),
                "reason_not_selected": "Uses more memory or has a higher first-run risk than the recommendation.",
            }
            for value in models[1:5]
        )
        return TrainingPlanRecommendation(
            plan=plan,
            revision=revision,
            alternatives=alternatives,
            summary=f"Use {selected.get('label') or selected['id']} for a bounded {profile.label.lower()} proof run.",
            primary_action=(
                {"id": "prepare_and_check", "label": "Prepare and check", "plan_revision_id": revision.id}
                if training_path_certification_id
                else {
                    "id": "verify_training_path",
                    "label": "Verify this training path",
                    "training_path_revision_id": training_path_revision_id,
                    "runtime_profile_revision_id": runtime_revision_id,
                }
            ),
        )

    # ---- persistence projections ---------------------------------------

    def get_plan(self, plan_id: str) -> Optional[TrainingPlan]:
        row = self.conn.execute("SELECT * FROM training_plans WHERE id=?", (plan_id,)).fetchone()
        return self._plan(row) if row else None

    def get_revision(self, revision_id: str) -> Optional[TrainingPlanRevision]:
        row = self.conn.execute("SELECT * FROM training_plan_revisions WHERE id=?", (revision_id,)).fetchone()
        return self._revision(row) if row else None

    def _resolved_revision(
        self,
        revision: TrainingPlanRevision,
        resolved_commit: str,
        *,
        model_identity: Optional[Mapping[str, Any]] = None,
    ) -> TrainingPlanRevision:
        if revision.training_path_certification_id:
            evidence_row = self.conn.execute(
                """SELECT result_json FROM training_path_certification_steps
                   WHERE certification_id=? AND step_id='model_preparation'
                         AND status='passed'""",
                (revision.training_path_certification_id,),
            ).fetchone()
            evidence = _loads(evidence_row["result_json"], {}) if evidence_row else {}
            certified_commit = str(evidence.get("resolved_model_commit") or "").strip()
            if not certified_commit:
                raise TrainingPlanError(
                    "The selected training path has no exact certified model commit. Verify this training path again."
                )
            if certified_commit != resolved_commit:
                raise TrainingPlanError(
                    "The prepared model commit differs from the real-path certification. Verify this exact model revision before guided use."
                )
        if revision.status == "resolved" and revision.resolved_model_commit == resolved_commit:
            return revision
        definition = dict(revision.definition)
        definition["resolved_model_commit"] = resolved_commit
        definition["model_revision"] = resolved_commit
        definition.update(dict(model_identity or {}))
        compute_shape = {
            key: definition.get(key)
            for key in (
                "trainer_mode", "backend", "model", "model_revision", "adaptation",
                "precision", "max_sequence_length", "batch_size",
                "gradient_accumulation_steps", "effective_batch_size",
                "verifier_profile_revision_id", "reward_system_revision_id",
                "dataset_shape",
                "tokenizer_identity", "processor_identity", "chat_template_hash",
                "model_inventory_hash",
                "reference_model", "reference_model_revision",
            )
        }
        compute_hash = _hash(compute_shape)
        identity = {
            "parent_revision_id": revision.id,
            "definition": definition,
            "compute_shape_hash": compute_hash,
            "runtime_hash": revision.runtime_hash,
        }
        content_hash = _hash(identity)
        existing = self.conn.execute(
            "SELECT * FROM training_plan_revisions WHERE content_hash=?",
            (content_hash,),
        ).fetchone()
        if existing:
            result = self._revision(existing)
            self._transfer_confirmation(revision.id, result.id)
            return result
        revision_number = int(
            self.conn.execute(
                "SELECT COALESCE(MAX(revision_number),0)+1 FROM training_plan_revisions WHERE plan_id=?",
                (revision.plan_id,),
            ).fetchone()[0]
        )
        identifier = _identifier("planrev", identity)
        created = _now()
        self.conn.execute(
            """INSERT INTO training_plan_revisions
               (id,plan_id,revision_number,status,content_hash,profile_id,profile_version,
                dataset_version_id,scenario_revision_id,trainer_mode,backend,model_id,
                model_revision,resolved_model_commit,definition_json,reasons_json,
                forecast_json,compute_shape_hash,runtime_hash,runtime_profile_revision_id,
                training_path_revision_id,training_path_certification_id,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                identifier, revision.plan_id, revision_number, "resolved", content_hash,
                revision.profile_id, revision.profile_version, revision.dataset_version_id,
                revision.scenario_revision_id, revision.trainer_mode, revision.backend,
                revision.model_id, resolved_commit, resolved_commit, _json(definition),
                _json([value.to_dict() for value in revision.reasons]),
                _json(revision.forecast.to_dict()), compute_hash, revision.runtime_hash,
                revision.runtime_profile_revision_id, revision.training_path_revision_id,
                revision.training_path_certification_id, created,
            ),
        )
        self.conn.execute(
            "UPDATE training_plans SET latest_revision_id=?,status='preparing',updated_at=? WHERE id=?",
            (identifier, created, revision.plan_id),
        )
        self.conn.commit()
        result = self.get_revision(identifier)
        assert result is not None
        self._transfer_confirmation(revision.id, result.id)
        return result

    def _transfer_confirmation(self, source_revision_id: str, target_revision_id: str) -> None:
        if source_revision_id == target_revision_id:
            return
        source = self.conn.execute(
            """SELECT id,reason,details_json FROM training_plan_decisions
               WHERE plan_revision_id=? AND decision='confirmed'
               ORDER BY created_at DESC LIMIT 1""",
            (source_revision_id,),
        ).fetchone()
        if source is None:
            return
        prior_rows = self.conn.execute(
            """SELECT details_json FROM training_plan_decisions
               WHERE plan_revision_id=? AND decision='confirmed'""",
            (target_revision_id,),
        ).fetchall()
        source_id = str(source["id"])
        if any(
            dict(_loads(row["details_json"], {})).get("transferred_from_decision_id") == source_id
            for row in prior_rows
        ):
            return
        details = dict(_loads(source["details_json"], {}))
        details.update(
            {
                "transferred_from_plan_revision_id": source_revision_id,
                "transferred_from_decision_id": source_id,
            }
        )
        self.record_decision(
            target_revision_id,
            "confirmed",
            reason=source["reason"],
            details=details,
        )

    def list_plans(self, *, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        total = int(self.conn.execute("SELECT COUNT(*) FROM training_plans").fetchone()[0])
        rows = self.conn.execute("SELECT * FROM training_plans ORDER BY updated_at DESC LIMIT ? OFFSET ?", (min(500, max(1, limit)), max(0, offset))).fetchall()
        return {"items": [self._plan(row).to_dict() for row in rows], "total": total, "limit": limit, "offset": offset}

    @staticmethod
    def _plan(row: Any) -> TrainingPlan:
        return TrainingPlan(str(row["id"]), str(row["dataset_version_id"]), row["scenario_revision_id"], str(row["status"]), row["latest_revision_id"], str(row["created_at"]), str(row["updated_at"]))

    @staticmethod
    def _forecast_value(value: Mapping[str, Any]) -> TrainingResourceForecast:
        proof = value.get("proof_seconds_range")
        full = value.get("full_run_seconds_range")
        return TrainingResourceForecast(
            download_bytes=value.get("download_bytes"), scratch_bytes=value.get("scratch_bytes"),
            checkpoint_bytes=value.get("checkpoint_bytes"), peak_memory_bytes=value.get("peak_memory_bytes"),
            proof_seconds_range=tuple(proof) if proof else None, full_run_seconds_range=tuple(full) if full else None,
            provenance=dict(value.get("provenance") or {}), confidence=str(value.get("confidence") or "low"),
        )

    @classmethod
    def _revision(cls, row: Any) -> TrainingPlanRevision:
        return TrainingPlanRevision(
            id=str(row["id"]), plan_id=str(row["plan_id"]), revision_number=int(row["revision_number"]),
            status=str(row["status"]), content_hash=str(row["content_hash"]), profile_id=str(row["profile_id"]),
            profile_version=str(row["profile_version"]), dataset_version_id=str(row["dataset_version_id"]),
            scenario_revision_id=row["scenario_revision_id"], trainer_mode=str(row["trainer_mode"]), backend=str(row["backend"]),
            model_id=str(row["model_id"]), model_revision=row["model_revision"], resolved_model_commit=row["resolved_model_commit"],
            definition=dict(_loads(row["definition_json"], {})),
            reasons=tuple(TrainingPlanReason(**value) for value in _loads(row["reasons_json"], [])),
            forecast=cls._forecast_value(_loads(row["forecast_json"], {})), compute_shape_hash=str(row["compute_shape_hash"]),
            runtime_hash=str(row["runtime_hash"]),
            runtime_profile_revision_id=row["runtime_profile_revision_id"] if "runtime_profile_revision_id" in row.keys() else None,
            training_path_revision_id=row["training_path_revision_id"] if "training_path_revision_id" in row.keys() else None,
            training_path_certification_id=row["training_path_certification_id"] if "training_path_certification_id" in row.keys() else None,
            created_at=str(row["created_at"]),
        )

    def alternatives(self, revision_id: str) -> Dict[str, Any]:
        revision = self.get_revision(revision_id)
        if revision is None:
            raise KeyError(revision_id)
        profile = next(value for value in _PROFILE_DEFINITIONS if value.id == revision.profile_id)
        _, _, runtime = self._runtime()
        version = self.database.get_dataset_version(revision.dataset_version_id)
        dataset = self.database.get_dataset(version.dataset_id) if version else None
        models = self._models(
            profile,
            revision.backend,
            runtime,
            modality=dataset.modality if dataset else None,
        )
        return {"items": [value for value in models if value["id"] != revision.model_id], "total": max(0, len(models) - 1), "limit": 50, "offset": 0}

    def derive_full_run_revision(self, proof_revision_id: str) -> TrainingPlanRevision:
        """Create the immutable uncapped child plan used by a reviewed full run."""

        proof = self.get_revision(proof_revision_id)
        if proof is None:
            raise KeyError(proof_revision_id)
        if proof.status != "resolved" or not proof.resolved_model_commit:
            raise TrainingPlanError("A full run requires a resolved proof-plan revision")
        definition = dict(proof.definition)
        for key in (
            "max_samples",
            "limit",
            "max_prompts",
            "max_steps",
            "proof_max_samples",
            "proof_sample_identity",
        ):
            definition.pop(key, None)
        definition["proof_run"] = False
        definition["derived_from_proof_plan_revision_id"] = proof.id
        version = self.database.get_dataset_version(proof.dataset_version_id)
        profile = next(value for value in _PROFILE_DEFINITIONS if value.id == proof.profile_id)
        model = get_model(proof.model_id) or {
            "id": proof.model_id,
            "estimated_memory_gb": (
                float(proof.forecast.peak_memory_bytes) / GIB
                if proof.forecast.peak_memory_bytes is not None
                else None
            ),
        }
        forecast = self._forecast(model, profile, version) if version else proof.forecast
        reasons = (*proof.reasons, TrainingPlanReason(
            "reviewed_full_run",
            "Uses the verified proof configuration without the proof cap",
            "The model, data, objective, and per-step compute shape are unchanged; only the reviewed training budget expands.",
        ))
        identity = {
            "parent_revision_id": proof.id,
            "definition": definition,
            "compute_shape_hash": proof.compute_shape_hash,
            "runtime_hash": proof.runtime_hash,
            "forecast": forecast.to_dict(),
        }
        content_hash = _hash(identity)
        existing = self.conn.execute(
            "SELECT * FROM training_plan_revisions WHERE content_hash=?",
            (content_hash,),
        ).fetchone()
        if existing:
            return self._revision(existing)
        revision_number = int(
            self.conn.execute(
                "SELECT COALESCE(MAX(revision_number),0)+1 FROM training_plan_revisions WHERE plan_id=?",
                (proof.plan_id,),
            ).fetchone()[0]
        )
        identifier = _identifier("planrev", identity)
        created = _now()
        self.conn.execute(
            """INSERT INTO training_plan_revisions
               (id,plan_id,revision_number,status,content_hash,profile_id,profile_version,
                dataset_version_id,scenario_revision_id,trainer_mode,backend,model_id,
                model_revision,resolved_model_commit,definition_json,reasons_json,
                forecast_json,compute_shape_hash,runtime_hash,runtime_profile_revision_id,
                training_path_revision_id,training_path_certification_id,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                identifier, proof.plan_id, revision_number, "resolved", content_hash,
                proof.profile_id, proof.profile_version, proof.dataset_version_id,
                proof.scenario_revision_id, proof.trainer_mode, proof.backend,
                proof.model_id, proof.model_revision, proof.resolved_model_commit,
                _json(definition), _json([value.to_dict() for value in reasons]),
                _json(forecast.to_dict()), proof.compute_shape_hash,
                proof.runtime_hash, proof.runtime_profile_revision_id,
                proof.training_path_revision_id, proof.training_path_certification_id,
                created,
            ),
        )
        self.conn.execute(
            "UPDATE training_plans SET latest_revision_id=?,status='ready',updated_at=? WHERE id=?",
            (identifier, created, proof.plan_id),
        )
        self.conn.commit()
        self.record_decision(
            identifier,
            "full_run_derived",
            details={"proof_plan_revision_id": proof.id},
        )
        result = self.get_revision(identifier)
        assert result is not None
        return result

    def choose_alternative(self, revision_id: str, model_id: str, *, reason: str) -> TrainingPlanRecommendation:
        revision = self.get_revision(revision_id)
        if revision is None:
            raise KeyError(revision_id)
        payload = dict(revision.definition)
        payload.update(plan_id=revision.plan_id, model=model_id, dataset_version_id=revision.dataset_version_id, scenario_revision_id=revision.scenario_revision_id, trainer_mode=revision.trainer_mode)
        recommendation = self.recommend(payload)
        self.record_decision(recommendation.revision.id, "alternative_selected", reason=reason, details={"previous_revision_id": revision.id})
        return recommendation

    # ---- model preparation ---------------------------------------------

    def prepare_model(self, revision_id: str, *, enqueue: bool = True, allow_download: bool = True) -> ModelPreparation:
        revision = self.get_revision(revision_id)
        if revision is None:
            raise KeyError(revision_id)
        identity = {"revision": revision.id, "model": revision.model_id, "requested_revision": revision.model_revision}
        preparation_id = _identifier("modelprep", identity)
        existing = self.get_model_preparation(preparation_id)
        if existing and existing.status == "completed":
            return existing
        now = _now()
        self.conn.execute(
            """INSERT OR REPLACE INTO model_preparations
               (id,plan_revision_id,status,requested_model_id,requested_revision,access_json,progress_json,cancel_requested,created_at)
               VALUES (?,?,?,?,?,?,?,0,?)""",
            (preparation_id, revision.id, "queued" if enqueue else "running", revision.model_id, revision.model_revision, _json({"download_confirmed": bool(allow_download)}), _json({"stage": "queued"}), now),
        )
        self.conn.execute("UPDATE training_plans SET status='preparing',updated_at=? WHERE id=?", (now, revision.plan_id))
        self.conn.commit()
        if enqueue:
            scheduler = self._scheduler()
            item = scheduler.enqueue(
                kind="training_model_preparation",
                launch_spec={"handler": "training_plan.execute_work_item", "operation": "prepare_model", "training_plan_root": str(self.root), "allow_download": bool(allow_download)},
                resource_class="cpu", resource_requirements={
                    "capacity_preflight": True,
                    "output_path": str(self.preparation_root),
                    "projected_disk_bytes": int(revision.forecast.download_bytes or 0),
                },
                domain_kind="model_preparation", domain_id=preparation_id, max_retries=2,
            )
            self.conn.execute("UPDATE model_preparations SET work_item_id=? WHERE id=?", (item.id, preparation_id))
            self.conn.commit()
            return self.get_model_preparation(preparation_id)  # type: ignore[return-value]
        self.run_model_preparation(preparation_id, allow_download=allow_download)
        return self.get_model_preparation(preparation_id)  # type: ignore[return-value]

    def run_model_preparation(self, preparation_id: str, *, allow_download: bool) -> ModelPreparation:
        try:
            return self._perform_model_preparation(
                preparation_id, allow_download=allow_download
            )
        except Exception as exc:
            row = self.conn.execute(
                "SELECT status,cancel_requested FROM model_preparations WHERE id=?",
                (preparation_id,),
            ).fetchone()
            if row is not None and str(row["status"]) != "completed":
                status = "cancelled" if bool(row["cancel_requested"]) else "blocked"
                self.conn.execute(
                    "UPDATE model_preparations SET status=?,error=?,completed_at=? WHERE id=?",
                    (status, str(exc), _now(), preparation_id),
                )
                self.conn.commit()
            raise

    def _perform_model_preparation(self, preparation_id: str, *, allow_download: bool) -> ModelPreparation:
        preparation = self.get_model_preparation(preparation_id)
        if preparation is None:
            raise KeyError(preparation_id)
        self.conn.execute("UPDATE model_preparations SET status='running',progress_json=? WHERE id=?", (_json({"stage": "resolving_model"}), preparation_id))
        self.conn.commit()
        model = preparation.requested_model_id
        requested = preparation.requested_revision or "main"
        source = Path(model).expanduser()
        if source.exists():
            cache_path = source.resolve()
            resolved = _hash({"path": str(cache_path), "mtime": cache_path.stat().st_mtime_ns})
        else:
            try:
                from huggingface_hub import snapshot_download

                cache_path = Path(snapshot_download(repo_id=model, revision=requested, local_files_only=not allow_download)).resolve()
            except Exception as exc:
                self.conn.execute("UPDATE model_preparations SET status='blocked',error=?,completed_at=? WHERE id=?", (str(exc), _now(), preparation_id))
                self.conn.commit()
                raise TrainingPlanError(f"Model preparation failed: {exc}") from exc
            resolved = cache_path.name
        required = [cache_path / "config.json"]
        if not required[0].is_file():
            raise TrainingPlanError("Prepared model is missing config.json")
        files = sorted(path for path in cache_path.rglob("*") if path.is_file())
        has_weights = any(path.suffix.lower() in {".safetensors", ".bin", ".pt", ".gguf"} for path in files)
        if not has_weights:
            raise TrainingPlanError("Prepared model has no supported weight files")
        source_revision = self.get_revision(preparation.plan_revision_id)
        if source_revision is None:
            raise TrainingPlanError("The model preparation lost its training-plan revision")
        modality = str(source_revision.definition.get("modality") or "text").lower()
        tokenizer_files = [
            path
            for path in files
            if path.name in {
                "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                "special_tokens_map.json", "vocab.json", "vocab.txt", "merges.txt",
            }
        ]
        processor_files = [
            path
            for path in files
            if path.name in {
                "preprocessor_config.json", "processor_config.json",
                "feature_extractor_config.json", "image_processor_config.json",
            }
        ]
        if modality in {"text", "code", ""} and not tokenizer_files:
            raise TrainingPlanError("Prepared text model is missing tokenizer files")
        if modality in {"image", "vision", "audio"} and not processor_files:
            raise TrainingPlanError("Prepared media model is missing processor files")
        if source_revision.trainer_mode == "vlm" and not tokenizer_files:
            raise TrainingPlanError("Prepared VLM model is missing tokenizer files")
        file_entries: list[Dict[str, Any]] = []
        for index, path in enumerate(files, 1):
            cancelled = self.conn.execute(
                "SELECT cancel_requested FROM model_preparations WHERE id=?",
                (preparation_id,),
            ).fetchone()
            if cancelled and bool(cancelled["cancel_requested"]):
                raise TrainingPlanError("Model preparation was cancelled")
            file_entries.append(
                {
                    "path": str(path.relative_to(cache_path)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _file_hash(path),
                }
            )
            if index == len(files) or index % 8 == 0:
                self.conn.execute(
                    "UPDATE model_preparations SET progress_json=? WHERE id=?",
                    (_json({"stage": "verifying_files", "processed": index, "total": len(files)}), preparation_id),
                )
                self.conn.commit()
        hashes_by_path = {entry["path"]: entry["sha256"] for entry in file_entries}
        tokenizer_identity = {
            "files": {
                str(path.relative_to(cache_path)): hashes_by_path[str(path.relative_to(cache_path))]
                for path in tokenizer_files
            }
        }
        tokenizer_identity["hash"] = _hash(tokenizer_identity["files"])
        processor_identity = {
            "files": {
                str(path.relative_to(cache_path)): hashes_by_path[str(path.relative_to(cache_path))]
                for path in processor_files
            }
        }
        processor_identity["hash"] = _hash(processor_identity["files"])
        chat_template_hash = None
        tokenizer_config = cache_path / "tokenizer_config.json"
        if tokenizer_config.is_file():
            try:
                chat_template = json.loads(tokenizer_config.read_text(encoding="utf-8")).get("chat_template")
                if chat_template:
                    chat_template_hash = hashlib.sha256(str(chat_template).encode("utf-8")).hexdigest()
            except (OSError, ValueError, TypeError):
                chat_template_hash = None
        manifest = {
            "format_version": 1, "model_id": model, "requested_revision": requested,
            "resolved_commit": resolved, "cache_path": str(cache_path),
            "files": file_entries,
            "tokenizer_identity": tokenizer_identity,
            "processor_identity": processor_identity,
            "chat_template_hash": chat_template_hash,
        }
        manifest_hash = _hash(
            {key: value for key, value in manifest.items() if key != "cache_path"}
        )
        output = self.preparation_root / preparation_id
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "manifest.json"
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, manifest_path)
        size = sum(value["size_bytes"] for value in manifest["files"])
        resolved_revision = self._resolved_revision(
            source_revision,
            resolved,
            model_identity={
                "tokenizer_identity": tokenizer_identity,
                "processor_identity": processor_identity,
                "chat_template_hash": chat_template_hash,
                "model_inventory_hash": manifest_hash,
            },
        )
        now = _now()
        self.conn.execute(
            """UPDATE model_preparations SET status='completed',resolved_commit=?,cache_path=?,manifest_path=?,manifest_hash=?,size_bytes=?,progress_json=?,error=NULL,completed_at=? WHERE id=?""",
            (resolved, str(cache_path), str(manifest_path), manifest_hash, size, _json({"stage": "completed", "files": len(files), "size_bytes": size, "resolved_plan_revision_id": resolved_revision.id}), now, preparation_id),
        )
        self.conn.execute(
            "UPDATE model_preparations SET plan_revision_id=? WHERE id=?",
            (resolved_revision.id, preparation_id),
        )
        self.conn.commit()
        return self.get_model_preparation(preparation_id)  # type: ignore[return-value]

    def get_model_preparation(self, preparation_id: str) -> Optional[ModelPreparation]:
        row = self.conn.execute("SELECT * FROM model_preparations WHERE id=?", (preparation_id,)).fetchone()
        if not row:
            return None
        return ModelPreparation(
            id=str(row["id"]), plan_revision_id=str(row["plan_revision_id"]), status=str(row["status"]),
            requested_model_id=str(row["requested_model_id"]), requested_revision=row["requested_revision"], resolved_commit=row["resolved_commit"],
            cache_path=row["cache_path"], manifest_path=row["manifest_path"], manifest_hash=row["manifest_hash"], size_bytes=row["size_bytes"],
            access=dict(_loads(row["access_json"], {})), progress=dict(_loads(row["progress_json"], {})), work_item_id=row["work_item_id"],
            error=row["error"], created_at=str(row["created_at"]), completed_at=row["completed_at"],
        )

    # ---- capacity checks ------------------------------------------------

    def create_capacity_check(self, revision_id: str, *, enqueue: bool = True) -> TrainingCapacityCheck:
        revision = self.get_revision(revision_id)
        if revision is None:
            raise KeyError(revision_id)
        prep_row = self.conn.execute("SELECT id FROM model_preparations WHERE plan_revision_id=? AND status='completed' ORDER BY completed_at DESC LIMIT 1", (revision.id,)).fetchone()
        if prep_row is None and revision.resolved_model_commit:
            prep_row = self.conn.execute(
                """SELECT mp.id FROM model_preparations mp
                   JOIN training_plan_revisions pr ON pr.id=mp.plan_revision_id
                   WHERE mp.status='completed' AND pr.model_id=?
                     AND mp.resolved_commit=?
                   ORDER BY mp.completed_at DESC LIMIT 1""",
                (revision.model_id, revision.resolved_model_commit),
            ).fetchone()
        if prep_row is None:
            raise TrainingPlanError("Prepare the exact model before checking capacity")
        identity = {"plan_revision_id": revision.id, "compute_shape_hash": revision.compute_shape_hash, "runtime_hash": revision.runtime_hash, "capacity_version": CAPACITY_VERSION}
        check_id = _identifier("capacity", identity)
        existing = self.get_capacity_check(check_id)
        if existing and existing.status in {"ready", "ready_with_adjustment"}:
            return existing
        capability_id = f"{revision.trainer_mode}_capacity_v1"
        now = _now()
        self.conn.execute(
            """INSERT OR REPLACE INTO training_capacity_checks
               (id,plan_revision_id,model_preparation_id,status,stage,capability_id,capability_version,
                compute_shape_hash,runtime_hash,selected_adjustment_json,forecast_json,progress_json,
                primary_remedy_json,cancel_requested,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)""",
            (check_id, revision.id, str(prep_row["id"]), "queued" if enqueue else "running", "queued", capability_id, CAPACITY_VERSION, revision.compute_shape_hash, revision.runtime_hash, _json({}), _json(revision.forecast.to_dict()), _json({"stage": "queued"}), _json({}), now),
        )
        self.conn.execute("UPDATE training_plans SET status='checking',updated_at=? WHERE id=?", (now, revision.plan_id))
        self.conn.commit()
        if enqueue:
            item = self._scheduler().enqueue(
                kind="training_capacity_check",
                launch_spec={"handler": "training_plan.execute_work_item", "operation": "capacity_check", "training_plan_root": str(self.root)},
                resource_class="accelerator", resource_requirements={
                    "capacity_preflight": True,
                    "output_path": str(self.scratch_root),
                    "projected_disk_bytes": int(revision.forecast.scratch_bytes or 0),
                    "projected_ram_bytes": int(revision.forecast.peak_memory_bytes or 0),
                    "runtime_profile_revision_id": revision.runtime_profile_revision_id,
                    "runtime_root": str(self.runtime_root),
                    "accelerator_family": (
                        "rocm" if revision.backend.startswith("rocm")
                        else "cuda" if revision.backend == "cuda" else None
                    ),
                },
                domain_kind="training_capacity_check", domain_id=check_id, max_retries=1,
            )
            self.conn.execute("UPDATE training_capacity_checks SET work_item_id=? WHERE id=?", (item.id, check_id))
            self.conn.commit()
            return self.get_capacity_check(check_id)  # type: ignore[return-value]
        return self.run_capacity_check(check_id)

    def _capacity_configurations(self, revision: TrainingPlanRevision) -> list[Dict[str, Any]]:
        base = dict(revision.definition)
        confirmed = {"step": "confirmed", "batch_size": int(base.get("batch_size") or 1), "gradient_accumulation_steps": int(base.get("gradient_accumulation_steps") or 1), "gradient_checkpointing": bool(base.get("gradient_checkpointing", False))}
        checkpointed = {**confirmed, "step": "gradient_checkpointing", "gradient_checkpointing": True}
        smaller = {**checkpointed, "step": "microbatch_one", "batch_size": 1, "gradient_accumulation_steps": int(confirmed["gradient_accumulation_steps"]) * int(confirmed["batch_size"])}
        result: list[Dict[str, Any]] = []
        for value in (confirmed, checkpointed, smaller):
            if value not in result:
                result.append(value)
        return result

    def _capacity_sample_identity(self, revision: TrainingPlanRevision) -> Dict[str, Any]:
        """Select deterministic train-row identities without retaining row content."""

        version = self.database.get_dataset_version(revision.dataset_version_id)
        if version is None:
            raise TrainingPlanError("The capacity check dataset version is missing")
        costs: list[tuple[int, int]] = []
        for split_index, row_line, _identity in self._iter_training_rows(version):
            costs.append((len(row_line), split_index))
        if not costs:
            raise TrainingPlanError("The selected training split has no readable rows")
        costs.sort(key=lambda value: (value[0], value[1]))
        median_cost, median_ordinal = costs[(len(costs) - 1) // 2]
        max_cost, max_ordinal = costs[-1]
        targets = {median_ordinal: "median_cost", max_ordinal: "maximum_cost"}
        selected: Dict[int, Dict[str, Any]] = {}
        for split_index, _row_line, identity in self._iter_training_rows(version):
            if split_index not in targets:
                continue
            value = dict(identity or {})
            selected[split_index] = {
                "selection": targets[split_index],
                "split_index": split_index,
                "serialized_bytes": median_cost if split_index == median_ordinal else max_cost,
                "record_id": value.get("record_id"),
                "record_hash": value.get("record_hash"),
                "instance_id": value.get("instance_id"),
                "virtual": bool(value.get("virtual", False)),
            }
        for index, selection in targets.items():
            if index not in selected or not selected[index].get("record_id"):
                virtual_hash = _hash(
                    {
                        "dataset_version_id": revision.dataset_version_id,
                        "split": "train",
                        "split_index": index,
                    }
                )
                selected.setdefault(index, {})
                selected[index].update(
                    {
                        "selection": selection,
                        "split_index": index,
                        "serialized_bytes": median_cost if index == median_ordinal else max_cost,
                        "record_id": f"virtual-{virtual_hash[:24]}",
                        "record_hash": None,
                        "instance_id": f"virtual-instance-{virtual_hash[:24]}",
                        "virtual": True,
                    }
                )
        records = [selected[index] for index in sorted(selected)]
        return {
            "dataset_version_id": revision.dataset_version_id,
            "split": "train",
            "records": records,
            "selection": [value["selection"] for value in records],
            "selection_cost": "exact_serialized_row_bytes_before_collation",
            "observed_split_rows": len(costs),
            "record_content_retained": False,
            "identity_hash": _hash(records),
        }

    def _iter_training_rows(self, version: Any):
        """Stream only the selected version's explicit training rows."""

        root = Path(version.storage_path)
        split_path = next(
            (
                candidate
                for candidate in (
                    root / "splits" / "train.jsonl",
                    root / "train.jsonl",
                    root / "records.jsonl",
                )
                if candidate.is_file()
            ),
            None,
        )
        if split_path is None:
            raise TrainingPlanError("The selected training split is missing")
        lineage_path = root / "lineage.jsonl"
        if split_path == root / "records.jsonl" and not lineage_path.is_file():
            raise TrainingPlanError(
                "Capacity checks require an explicit training split or lineage index; protected rows cannot be sampled safely"
            )
        if split_path == root / "records.jsonl" and lineage_path.is_file():
            with split_path.open("rb") as records, lineage_path.open(encoding="utf-8") as lineage:
                split_index = 0
                for row_line, identity_line in zip(records, lineage):
                    if not row_line.strip() or not identity_line.strip():
                        continue
                    identity = json.loads(identity_line)
                    if "train" not in set(identity.get("splits") or ()):
                        continue
                    yield split_index, row_line, identity
                    split_index += 1
        else:
            lineage_by_index: Dict[int, Dict[str, Any]] = {}
            if lineage_path.is_file():
                with lineage_path.open(encoding="utf-8") as lineage:
                    for line in lineage:
                        if not line.strip():
                            continue
                        identity = json.loads(line)
                        split_index = dict(identity.get("split_indices") or {}).get("train")
                        if split_index is not None:
                            lineage_by_index[int(split_index)] = identity
            with split_path.open("rb") as handle:
                split_index = 0
                for line in handle:
                    if not line.strip():
                        continue
                    yield split_index, line, lineage_by_index.get(split_index)
                    split_index += 1

    @staticmethod
    def _resolve_scratch_media(value: Any, source_root: Path, *, key: str = "") -> Any:
        media_keys = {
            "media", "image", "image_ref", "image_path", "audio", "audio_ref", "audio_path"
        }
        if isinstance(value, dict):
            return {
                child_key: TrainingPlanService._resolve_scratch_media(
                    child_value, source_root, key=str(child_key).lower()
                )
                for child_key, child_value in value.items()
            }
        if isinstance(value, list):
            return [TrainingPlanService._resolve_scratch_media(item, source_root, key=key) for item in value]
        if key in media_keys and isinstance(value, str) and value and "://" not in value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (source_root / path).resolve()
            return str(path)
        return value

    def _write_capacity_input(
        self,
        revision: TrainingPlanRevision,
        sample_identity: Mapping[str, Any],
        scratch: Path,
    ) -> Path:
        """Materialize selected rows only inside attempt-local disposable storage."""

        version = self.database.get_dataset_version(revision.dataset_version_id)
        if version is None:
            raise TrainingPlanError("The capacity check dataset version is missing")
        targets = {int(value["split_index"]) for value in sample_identity.get("records") or ()}
        output = scratch / "capacity-input.jsonl"
        written = 0
        with output.open("wb") as handle:
            for split_index, row_line, _identity in self._iter_training_rows(version):
                if split_index not in targets:
                    continue
                try:
                    parsed = json.loads(row_line)
                    resolved = self._resolve_scratch_media(parsed, Path(version.storage_path))
                    row_line = (_json(resolved) + "\n").encode("utf-8")
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
                handle.write(row_line if row_line.endswith(b"\n") else row_line + b"\n")
                written += 1
        if written <= 0:
            raise TrainingPlanError("No capacity-check rows could be staged")
        return output

    def _bounded_probe(self, request: Mapping[str, Any], scratch: Path) -> Mapping[str, Any]:
        """Portable capacity probe used when no backend-specific runner is injected.

        It verifies the exact prepared model inventory and measures current
        workstation headroom. Hardware qualification jobs inject the real
        scratch-step runner and set ``scratch_step_executed``. The portable
        path never fabricates that evidence.
        """

        before = self.capacity_sampler(self.root).to_dict()
        projected = int(request.get("projected_memory_bytes") or 0)
        available = self._memory_available(before)
        if available is not None and projected and projected > available * 0.9:
            raise MemoryError(f"Projected peak {projected} exceeds the current safe memory margin")
        marker = scratch / "capacity-attempt.json"
        marker.write_text(_json({"compute_shape_hash": request.get("compute_shape_hash"), "configuration": request.get("configuration")}), encoding="utf-8")
        after = self.capacity_sampler(self.root).to_dict()
        return {
            "scratch_step_executed": False,
            "measurement_contract": "portable_allocation_preflight_v1",
            "measured_at": _now(),
            "before": before,
            "after": after,
            "peak_process_memory_bytes": (after.get("process") or {}).get("rss_bytes"),
            "peak_device_memory_bytes": (after.get("accelerator") or {}).get("device_memory_used_bytes"),
            "note": "This runtime verified model inventory and measured headroom; a hardware qualification runner can add an exact scratch optimizer step.",
        }

    @staticmethod
    def _optimizer_steps_in(value: Any) -> int:
        if isinstance(value, dict):
            direct = max(
                (
                    int(value.get(key) or 0)
                    for key in (
                        "optimizer_steps",
                        "train_steps_executed",
                        "total_train_steps_executed",
                    )
                    if isinstance(value.get(key), (int, float))
                ),
                default=0,
            )
            nested = [
                TrainingPlanService._optimizer_steps_in(item)
                for item in value.values()
            ]
            return max([direct, *nested])
        if isinstance(value, list):
            return max((TrainingPlanService._optimizer_steps_in(item) for item in value), default=0)
        return 0

    @staticmethod
    def _measured_peak_memory(*values: Any) -> Optional[int]:
        """Choose the largest trustworthy measured memory value.

        Unified-memory runtimes can expose a process RSS that includes the
        allocation while their device counter is absent or only reports a tiny
        bookkeeping allocation.  Treating the first non-null counter as the
        peak made those checks materially under-report workstation usage.
        """

        measured: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            candidate = int(value)
            if candidate >= 0:
                measured.append(candidate)
        return max(measured) if measured else None

    def _trainer_probe(self, request: Mapping[str, Any], scratch: Path) -> Mapping[str, Any]:
        """Run an attempt-local, one-step check through the shipped trainer CLI.

        The staged input, logs, optimizer state, and weights all live below
        ``scratch`` and are deleted by the caller. Only resource measurements,
        hashes, and the reported optimizer-step count leave the attempt.
        """

        from halo_forge.public_api.service import PublicApiService

        definition = dict(request.get("definition") or {})
        configuration = dict(request.get("configuration") or {})
        dataset_path = str(request.get("dataset_path") or "")
        if not dataset_path or not Path(dataset_path).is_file():
            raise TrainingPlanError("The disposable trainer input was not created")
        mode = str(definition.get("trainer_mode") or "").lower()
        output = scratch / "trainer-output"
        payload = {
            **definition,
            **configuration,
            "mode": mode,
            "model": str(request.get("model_cache_path") or definition.get("model") or ""),
            "dataset": dataset_path,
            "prompts": dataset_path,
            "output_dir": str(output),
            "output_root": str(scratch),
            "epochs": 1,
            "cycles": 1,
            "max_steps": 1,
            "max_samples": int(request.get("sample_count") or 1),
            "max_prompts": int(request.get("sample_count") or 1),
            "limit": int(request.get("sample_count") or 1),
            "proof_run": True,
            "no_caffeinate": True,
        }
        # Capacity verifies the local compute shape. It never invokes a hosted
        # verifier or reward provider; those bindings are validated separately.
        provider_binding_present = bool(
            payload.pop("verifier_profile_revision_id", None)
            or payload.pop("reward_system_revision_id", None)
        )
        command = PublicApiService._managed_training_command(payload)
        runtime_revision_id = str(
            request.get("runtime_profile_revision_id") or ""
        ).strip()
        managed_qualification = None
        if runtime_revision_id:
            from halo_forge.managed_runtime import ManagedRuntimeService

            managed_runtime = ManagedRuntimeService(
                self.database,
                root=self.runtime_root,
                scheduler=self.scheduler,
            )
            command, _runtime_cwd, _runtime_env, managed_qualification = (
                managed_runtime.wrap_execution(
                    runtime_revision_id,
                    command,
                    cwd=str(self.root),
                    launch_spec=payload,
                )
            )
        log_path = scratch / "trainer.log"
        environment = {
            key: os.environ[key]
            for key in (
                "PATH",
                "HOME",
                "LANG",
                "LC_ALL",
                "XDG_RUNTIME_DIR",
                "TMPDIR",
            )
            if key in os.environ
        }
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        before = self.capacity_sampler(self.root).to_dict()
        peak_rss: Optional[int] = None
        peak_device: Optional[int] = None
        started = time.monotonic()
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            try:
                try:
                    import psutil

                    observed_process = psutil.Process(process.pid)
                except Exception:
                    observed_process = None
                while process.poll() is None:
                    cancel_requested = request.get("cancel_requested")
                    if callable(cancel_requested) and bool(cancel_requested()):
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        raise TrainingPlanError("Capacity check was cancelled")
                    if observed_process is not None:
                        try:
                            rss = int(observed_process.memory_info().rss)
                            rss += sum(
                                int(child.memory_info().rss)
                                for child in observed_process.children(recursive=True)
                                if child.is_running()
                            )
                            peak_rss = max(peak_rss or 0, rss)
                        except Exception:
                            pass
                    sampled = self.capacity_sampler(self.root).to_dict()
                    used = (sampled.get("accelerator") or {}).get("device_memory_used_bytes")
                    if used is not None:
                        peak_device = max(peak_device or 0, int(used))
                    time.sleep(0.25)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        if process.returncode:
            try:
                lower_log = log_path.read_text(encoding="utf-8", errors="replace").lower()
            except OSError:
                lower_log = ""
            if "out of memory" in lower_log or "cuda oom" in lower_log:
                raise MemoryError("The exact scratch trainer step exceeded available memory")
            raise TrainingPlanError(
                f"The exact scratch trainer step exited with code {process.returncode}"
            )
        optimizer_steps = 0
        for candidate in output.rglob("*.json") if output.exists() else ():
            try:
                optimizer_steps = max(
                    optimizer_steps,
                    self._optimizer_steps_in(json.loads(candidate.read_text(encoding="utf-8"))),
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        if optimizer_steps < 1:
            raise TrainingPlanError(
                "The trainer completed without verifiable optimizer-step evidence"
            )
        after = self.capacity_sampler(self.root).to_dict()
        return {
            "scratch_step_executed": True,
            "measurement_contract": f"{mode}_trainer_scratch_step_v1",
            "optimizer_steps": optimizer_steps,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "peak_process_memory_bytes": peak_rss,
            "peak_device_memory_bytes": peak_device,
            "runtime_before_hash": _hash(before),
            "runtime_after_hash": _hash(after),
            "input_identity_hash": (request.get("sample_identity") or {}).get("identity_hash"),
            "source_content_retained": False,
            "provider_binding_present": provider_binding_present,
            "provider_execution_checked": False if provider_binding_present else None,
            "runtime_profile_revision_id": runtime_revision_id or None,
            "runtime_qualification_id": (
                managed_qualification.id if managed_qualification else None
            ),
        }

    def run_capacity_check(self, check_id: str) -> TrainingCapacityCheck:
        check = self.get_capacity_check(check_id)
        if check is None:
            raise KeyError(check_id)
        revision = self.get_revision(check.plan_revision_id)
        preparation = self.get_model_preparation(str(check.model_preparation_id or ""))
        if revision is None or preparation is None or preparation.status != "completed":
            raise TrainingPlanError("Capacity check inputs are incomplete")
        self.conn.execute("UPDATE training_capacity_checks SET status='running',stage='checking',progress_json=? WHERE id=?", (_json({"stage": "checking", "attempt": 0}), check_id))
        self.conn.commit()
        last_error: Optional[Exception] = None
        passed: Optional[tuple[int, Dict[str, Any], Dict[str, Any]]] = None
        prior_ordinal = int(
            self.conn.execute(
                "SELECT COALESCE(MAX(ordinal),0) FROM training_capacity_attempts WHERE capacity_check_id=?",
                (check_id,),
            ).fetchone()[0]
        )
        configurations = self._capacity_configurations(revision)
        for configuration_index, configuration in enumerate(configurations, 1):
            ordinal = prior_ordinal + configuration_index
            attempt_id = _identifier("capattempt", {"check": check_id, "ordinal": ordinal})
            scratch = self.scratch_root / check_id / f"attempt-{ordinal}-{uuid.uuid4().hex[:8]}"
            scratch.mkdir(parents=True, exist_ok=False)
            sample_identity = self._capacity_sample_identity(revision)
            self.conn.execute(
                "INSERT INTO training_capacity_attempts (id,capacity_check_id,ordinal,configuration_json,status,sample_identity_json,created_at) VALUES (?,?,?,?,?,?,?)",
                (attempt_id, check_id, ordinal, _json(configuration), "running", _json(sample_identity), _now()),
            )
            self.conn.execute("UPDATE training_capacity_checks SET progress_json=? WHERE id=?", (_json({"stage": "checking", "attempt": configuration_index, "total_attempts": len(configurations), "attempt_history": prior_ordinal}), check_id))
            self.conn.commit()
            try:
                dataset_path = self._write_capacity_input(
                    revision, sample_identity, scratch
                )
                measurements = dict(self.probe_runner({
                    "configuration": configuration,
                    "definition": revision.definition,
                    "model_cache_path": preparation.cache_path,
                    "compute_shape_hash": revision.compute_shape_hash,
                    "projected_memory_bytes": revision.forecast.peak_memory_bytes,
                    "sample_identity": sample_identity,
                    "sample_count": len(sample_identity.get("records") or ()),
                    "dataset_path": str(dataset_path),
                    "cancel_requested": lambda: bool(
                        (
                            self.conn.execute(
                                "SELECT cancel_requested FROM training_capacity_checks WHERE id=?",
                                (check_id,),
                            ).fetchone()
                            or {"cancel_requested": 0}
                        )["cancel_requested"]
                    ),
                    "runtime_profile_revision_id": revision.runtime_profile_revision_id,
                }, scratch))
                if not bool(measurements.get("scratch_step_executed")):
                    raise TrainingPlanError(
                        "The capacity adapter did not produce exact scratch-step evidence"
                    )
                self.conn.execute("UPDATE training_capacity_attempts SET status='passed',measurements_json=?,scratch_cleaned=1,completed_at=? WHERE id=?", (_json(measurements), _now(), attempt_id))
                passed = (configuration_index, configuration, measurements)
                break
            except Exception as exc:
                last_error = exc
                error_class = "out_of_memory" if isinstance(exc, MemoryError) or "memory" in str(exc).lower() else "probe_failed"
                self.conn.execute("UPDATE training_capacity_attempts SET status='failed',error_class=?,error=?,scratch_cleaned=1,completed_at=? WHERE id=?", (error_class, str(exc), _now(), attempt_id))
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
                try:
                    scratch.parent.rmdir()
                except OSError:
                    pass
                self.conn.commit()
        now = _now()
        if passed is None:
            remedy = {"id": "smaller_model", "label": "Use a smaller model", "reason": str(last_error or "No capacity configuration passed")}
            self.conn.execute("UPDATE training_capacity_checks SET status='blocked',stage='blocked',primary_remedy_json=?,error=?,completed_at=? WHERE id=?", (_json(remedy), str(last_error or "Capacity check failed"), now, check_id))
            self.conn.execute("UPDATE training_plans SET status='blocked',updated_at=? WHERE id=?", (now, revision.plan_id))
        else:
            ordinal, configuration, measurements = passed
            status = "ready" if ordinal == 1 else "ready_with_adjustment"
            forecast = revision.forecast.to_dict()
            process_peak = measurements.get("peak_process_memory_bytes")
            device_peak = measurements.get("peak_device_memory_bytes")
            peak = self._measured_peak_memory(process_peak, device_peak)
            if peak is not None:
                forecast["peak_memory_bytes"] = int(peak)
                forecast.setdefault("provenance", {})["peak_memory_bytes"] = "measured"
                forecast["confidence"] = "medium"
            self.conn.execute("UPDATE training_capacity_checks SET status=?,stage='completed',selected_adjustment_json=?,forecast_json=?,progress_json=?,primary_remedy_json='{}',error=NULL,completed_at=? WHERE id=?", (status, _json(configuration), _json(forecast), _json({"stage": "completed", "attempt": ordinal}), now, check_id))
            self.conn.execute("UPDATE training_plans SET status='ready',updated_at=? WHERE id=?", (now, revision.plan_id))
        self.conn.commit()
        return self.get_capacity_check(check_id)  # type: ignore[return-value]

    def get_capacity_check(self, check_id: str) -> Optional[TrainingCapacityCheck]:
        row = self.conn.execute("SELECT * FROM training_capacity_checks WHERE id=?", (check_id,)).fetchone()
        if not row:
            return None
        return TrainingCapacityCheck(
            id=str(row["id"]), plan_revision_id=str(row["plan_revision_id"]), model_preparation_id=row["model_preparation_id"], status=str(row["status"]), stage=str(row["stage"]),
            capability_id=str(row["capability_id"]), capability_version=str(row["capability_version"]), compute_shape_hash=str(row["compute_shape_hash"]), runtime_hash=str(row["runtime_hash"]),
            selected_adjustment=dict(_loads(row["selected_adjustment_json"], {})), forecast=self._forecast_value(_loads(row["forecast_json"], {})),
            progress=dict(_loads(row["progress_json"], {})), primary_remedy=dict(_loads(row["primary_remedy_json"], {})), work_item_id=row["work_item_id"], error=row["error"], created_at=str(row["created_at"]), completed_at=row["completed_at"],
        )

    def list_capacity_attempts(self, check_id: str) -> Dict[str, Any]:
        rows = self.conn.execute("SELECT * FROM training_capacity_attempts WHERE capacity_check_id=? ORDER BY ordinal", (check_id,)).fetchall()
        items = [TrainingCapacityAttempt(
            id=str(row["id"]), capacity_check_id=str(row["capacity_check_id"]), ordinal=int(row["ordinal"]), configuration=dict(_loads(row["configuration_json"], {})), status=str(row["status"]),
            sample_identity=dict(_loads(row["sample_identity_json"], {})), measurements=dict(_loads(row["measurements_json"], {})), error_class=row["error_class"], error=row["error"],
            scratch_cleaned=bool(row["scratch_cleaned"]), created_at=str(row["created_at"]), completed_at=row["completed_at"],
        ).to_dict() for row in rows]
        return {"items": items, "total": len(items), "limit": len(items), "offset": 0}

    # ---- readiness, launch binding, work --------------------------------

    def readiness(self, revision_id: str) -> TrainingPlanReadiness:
        revision = self.get_revision(revision_id)
        if revision is None:
            raise KeyError(revision_id)
        prep_row = self.conn.execute("SELECT id FROM model_preparations WHERE plan_revision_id=? ORDER BY created_at DESC LIMIT 1", (revision_id,)).fetchone()
        if prep_row is None and revision.resolved_model_commit:
            prep_row = self.conn.execute(
                """SELECT mp.id FROM model_preparations mp
                   JOIN training_plan_revisions pr ON pr.id=mp.plan_revision_id
                   WHERE mp.status='completed' AND pr.model_id=?
                     AND mp.resolved_commit=?
                   ORDER BY mp.completed_at DESC LIMIT 1""",
                (revision.model_id, revision.resolved_model_commit),
            ).fetchone()
        check_row = self.conn.execute("SELECT id FROM training_capacity_checks WHERE plan_revision_id=? ORDER BY created_at DESC LIMIT 1", (revision_id,)).fetchone()
        if check_row is None:
            check_row = self.conn.execute(
                """SELECT id FROM training_capacity_checks
                   WHERE compute_shape_hash=? AND runtime_hash=?
                     AND status IN ('ready','ready_with_adjustment')
                   ORDER BY completed_at DESC LIMIT 1""",
                (revision.compute_shape_hash, revision.runtime_hash),
            ).fetchone()
        prep = self.get_model_preparation(str(prep_row["id"])) if prep_row else None
        check = self.get_capacity_check(str(check_row["id"])) if check_row else None
        blockers: list[Dict[str, Any]] = []
        notices: list[Dict[str, Any]] = []
        if revision.runtime_profile_revision_id:
            if not revision.training_path_revision_id:
                blockers.append({"code": "path_unprofiled", "summary": "This training path has no real certification profile."})
            elif not revision.training_path_certification_id:
                blockers.append({"code": "path_not_verified", "summary": "Verify this real training path before preparing user data."})
            else:
                from halo_forge.training_path_certification import TrainingPathCertificationService

                verification = TrainingPathCertificationService(
                    self.database,
                    runtime_service=self._managed_runtime_service(),
                    scheduler=self.scheduler,
                ).verify(revision.training_path_certification_id)
                if not verification["valid"]:
                    blockers.append({"code": "path_stale", "summary": "The training-path certification is stale; verify it again."})
        if prep is None or prep.status != "completed" or not prep.cache_path or not Path(prep.cache_path).exists() or not prep.manifest_path or not Path(prep.manifest_path).is_file():
            blockers.append({"code": "model_not_prepared", "summary": "Prepare the exact model files."})
        if check is None or check.status not in {"ready", "ready_with_adjustment"}:
            blockers.append({"code": "capacity_not_verified", "summary": "Complete the workstation capacity check."})
        _, active_runtime_hash, _ = self._runtime()
        if active_runtime_hash != revision.runtime_hash:
            blockers.append({"code": "stale_runtime", "summary": "The workstation runtime changed; run Prepare and check again."})
        if check is not None:
            attempt_row = self.conn.execute(
                """SELECT measurements_json FROM training_capacity_attempts
                   WHERE capacity_check_id=? AND status='passed'
                   ORDER BY ordinal DESC LIMIT 1""",
                (check.id,),
            ).fetchone()
            measurements = dict(
                _loads(attempt_row["measurements_json"], {})
                if attempt_row is not None
                else {}
            )
            if measurements.get("provider_binding_present") and measurements.get("provider_execution_checked") is False:
                notices.append(
                    {
                        "code": "provider_execution_unchecked",
                        "summary": "The local training shape passed. The hosted verifier or reward provider was not contacted during this check and will require separate confirmation.",
                    }
                )
        if blockers:
            action = (
                {
                    "id": "verify_training_path",
                    "label": "Verify this training path",
                    "training_path_revision_id": revision.training_path_revision_id,
                    "runtime_profile_revision_id": revision.runtime_profile_revision_id,
                }
                if blockers[0]["code"] in {"path_unprofiled", "path_not_verified", "path_stale"}
                else {"id": "prepare_and_check", "label": "Prepare and check"}
            )
            return TrainingPlanReadiness(revision_id, "blocked", "Preparation needed", blockers[0]["summary"], prep, check, tuple(blockers), action, tuple(notices))
        return TrainingPlanReadiness(revision_id, "ready", "Ready for proof run", "The exact model and training shape passed the workstation check.", prep, check, (), {"id": "start_proof_run", "label": "Start proof run"}, tuple(notices))

    def resolved_launch_payload(self, revision_id: str, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        revision = self.get_revision(revision_id)
        if revision is None:
            raise KeyError(revision_id)
        readiness = self.readiness(revision_id)
        if readiness.status != "ready":
            raise TrainingPlanError(readiness.summary)
        supplied = dict(payload or {})
        effective_definition = dict(revision.definition)
        if readiness.capacity_check is not None:
            effective_definition.update(
                {
                    key: value
                    for key, value in readiness.capacity_check.selected_adjustment.items()
                    if key
                    in {
                        "batch_size",
                        "gradient_accumulation_steps",
                        "gradient_checkpointing",
                    }
                }
            )
        conflicts = []
        protected = {
            "mode": "trainer_mode",
            "trainer_mode": "trainer_mode",
            "model": "model",
            "model_revision": "model_revision",
            "dataset_version_id": "dataset_version_id",
            "dataset_split": "dataset_split",
            "validation_split": "validation_split",
            "seed": "seed",
            "batch_size": "batch_size",
            "gradient_accumulation": "gradient_accumulation_steps",
            "gradient_accumulation_steps": "gradient_accumulation_steps",
            "gradient_checkpointing": "gradient_checkpointing",
            "learning_rate": "learning_rate",
            "max_samples": "max_samples",
            "limit": "limit",
            "max_steps": "max_steps",
            "max_sequence_length": "max_sequence_length",
            "max_seq_length": "max_sequence_length",
            "adaptation": "adaptation",
            "adaptation_mode": "adaptation",
            "precision": "precision",
            "epochs": "epochs",
            "cycles": "cycles",
            "packing": "packing",
            "budget_mode": "budget_mode",
            "corpus_passes": "corpus_passes",
            "reference_model": "reference_model",
            "verifier_profile_revision_id": "verifier_profile_revision_id",
            "reward_system_revision_id": "reward_system_revision_id",
        }
        for raw_key, plan_key in protected.items():
            if raw_key not in supplied or supplied[raw_key] in (None, ""):
                continue
            if plan_key not in effective_definition:
                conflicts.append(raw_key)
                continue
            actual = supplied[raw_key]
            expected = effective_definition.get(plan_key)
            equivalent = (
                _json(actual) == _json(expected)
                or str(actual).strip().lower() == str(expected).strip().lower()
            )
            if raw_key == "model" and readiness.model_preparation is not None:
                equivalent = equivalent or str(actual).strip() in {
                    revision.model_id,
                    str(readiness.model_preparation.cache_path or "").strip(),
                }
            if not equivalent:
                conflicts.append(raw_key)
        if supplied.get("dataset") or supplied.get("prompts"):
            conflicts.append("dataset")
        if conflicts:
            raise TrainingPlanError("training_plan_revision_id conflicts with explicit fields: " + ", ".join(sorted(set(conflicts))))
        preparation = readiness.model_preparation
        if preparation is None or not preparation.cache_path:
            raise TrainingPlanError("The prepared model path is unavailable")
        resolved = {**supplied, **revision.definition}
        resolved.update(
            mode=revision.trainer_mode,
            # Execute from the verified content-addressed cache.  Keep the
            # catalog identity separately so replay and operator surfaces do
            # not confuse a workstation path with the selected model.
            model=preparation.cache_path,
            training_plan_model_id=revision.model_id,
            dataset_version_id=revision.dataset_version_id,
            training_plan_revision_id=revision.id,
            training_capacity_check_id=readiness.capacity_check.id if readiness.capacity_check else None,
            model_preparation_id=readiness.model_preparation.id if readiness.model_preparation else None,
            training_compute_shape_hash=revision.compute_shape_hash,
            training_capacity_adjustment=(readiness.capacity_check.selected_adjustment if readiness.capacity_check else {}),
            training_plan_content_hash=revision.content_hash,
            training_plan_runtime_hash=revision.runtime_hash,
            runtime_profile_revision_id=revision.runtime_profile_revision_id,
            training_plan_recommendation_reasons=[value.to_dict() for value in revision.reasons],
            training_plan_forecast=revision.forecast.to_dict(),
            resolved_model_commit=revision.resolved_model_commit,
            model_preparation_manifest_hash=(readiness.model_preparation.manifest_hash if readiness.model_preparation else None),
        )
        confirmation = self.conn.execute(
            """SELECT id FROM training_plan_decisions
               WHERE plan_revision_id=? AND decision='confirmed'
               ORDER BY created_at DESC LIMIT 1""",
            (revision.id,),
        ).fetchone()
        if confirmation:
            resolved["training_plan_decision_id"] = str(confirmation["id"])
        adjustment = readiness.capacity_check.selected_adjustment if readiness.capacity_check else {}
        resolved.update({key: value for key, value in adjustment.items() if key in {"batch_size", "gradient_accumulation_steps", "gradient_checkpointing"}})
        return resolved

    def record_decision(self, revision_id: str, decision: str, *, reason: Optional[str] = None, details: Optional[Mapping[str, Any]] = None) -> TrainingPlanDecision:
        if self.get_revision(revision_id) is None:
            raise KeyError(revision_id)
        allowed = {"confirmed", "alternative_selected", "override", "proof_launched", "full_run_derived"}
        if decision not in allowed:
            raise TrainingPlanError(f"Unsupported training-plan decision: {decision}")
        if decision == "override" and not str(reason or "").strip():
            raise TrainingPlanError("Overrides require a retained reason")
        identity = {"revision": revision_id, "decision": decision, "reason": reason, "details": dict(details or {}), "nonce": uuid.uuid4().hex}
        identifier = _identifier("plandecision", identity)
        created = _now()
        self.conn.execute("INSERT INTO training_plan_decisions (id,plan_revision_id,decision,reason,details_json,created_at) VALUES (?,?,?,?,?,?)", (identifier, revision_id, decision, str(reason).strip() if reason else None, _json(dict(details or {})), created))
        self.conn.commit()
        return TrainingPlanDecision(identifier, revision_id, decision, str(reason).strip() if reason else None, dict(details or {}), created)

    def is_confirmed(self, revision_id: str, *, require_download: bool = False) -> bool:
        rows = self.conn.execute(
            """SELECT details_json FROM training_plan_decisions
               WHERE plan_revision_id=? AND decision='confirmed'
               ORDER BY created_at DESC""",
            (revision_id,),
        ).fetchall()
        if not rows:
            return False
        if not require_download:
            return True
        return any(
            bool(dict(_loads(row["details_json"], {})).get("download_confirmed"))
            for row in rows
        )

    def bind_run(self, *, run_id: str, revision_id: str, capacity_check_id: Optional[str], role: str = "proof") -> None:
        if role not in {"proof", "full", "manual"}:
            raise TrainingPlanError("run training-plan role is invalid")
        self.conn.execute("INSERT OR REPLACE INTO run_training_plans (run_id,plan_revision_id,capacity_check_id,role,attached_at) VALUES (?,?,?,?,?)", (run_id, revision_id, capacity_check_id, role, _now()))
        self.conn.commit()

    def run_binding(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM run_training_plans WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        revision = self.get_revision(str(row["plan_revision_id"]))
        capacity = (
            self.get_capacity_check(str(row["capacity_check_id"]))
            if row["capacity_check_id"]
            else None
        )
        return {
            "run_id": str(row["run_id"]),
            "role": str(row["role"]),
            "attached_at": str(row["attached_at"]),
            "revision": revision.to_dict() if revision else None,
            "capacity_check": capacity.to_dict() if capacity else None,
        }

    def cancel(self, domain_kind: str, domain_id: str) -> Dict[str, Any]:
        table = {"model_preparation": "model_preparations", "training_capacity_check": "training_capacity_checks"}.get(domain_kind)
        if not table:
            raise TrainingPlanError("Unsupported cancellable domain")
        row = self.conn.execute(f"SELECT work_item_id FROM {table} WHERE id=?", (domain_id,)).fetchone()
        if not row:
            raise KeyError(domain_id)
        self.conn.execute(f"UPDATE {table} SET cancel_requested=1 WHERE id=?", (domain_id,))
        self.conn.commit()
        if row["work_item_id"]:
            self._scheduler().cancel(str(row["work_item_id"]))
        return {"id": domain_id, "cancel_requested": True, "work_item_id": row["work_item_id"]}

    def retry(self, domain_kind: str, domain_id: str, *, reason: str) -> Dict[str, Any]:
        """Retry durable preparation/check work without erasing attempt history."""

        table = {"model_preparation": "model_preparations", "training_capacity_check": "training_capacity_checks"}.get(domain_kind)
        if not table:
            raise TrainingPlanError("Unsupported retryable domain")
        retained_reason = str(reason or "").strip()
        if not retained_reason:
            raise TrainingPlanError("A retry reason is required")
        row = self.conn.execute(f"SELECT status,work_item_id FROM {table} WHERE id=?", (domain_id,)).fetchone()
        if not row:
            raise KeyError(domain_id)
        if str(row["status"]) not in {"blocked", "failed", "cancelled", "stale"}:
            raise TrainingPlanError("Only blocked, failed, cancelled, or stale work can be retried")
        work_item_id = str(row["work_item_id"] or "")
        if not work_item_id:
            raise TrainingPlanError("This operation has no durable work item to retry")
        if domain_kind == "training_capacity_check":
            self.conn.execute(
                "UPDATE training_capacity_checks SET status='queued',stage='queued',error=NULL,cancel_requested=0,completed_at=NULL,progress_json=? WHERE id=?",
                (_json({"stage": "queued", "retry_reason": retained_reason}), domain_id),
            )
        else:
            self.conn.execute(
                "UPDATE model_preparations SET status='queued',error=NULL,cancel_requested=0,completed_at=NULL,progress_json=? WHERE id=?",
                (_json({"stage": "queued", "retry_reason": retained_reason}), domain_id),
            )
        self.conn.commit()
        item = self._scheduler().retry(
            work_item_id,
            reason=retained_reason,
            force=True,
            sync_domain=False,
        )
        if item is None:
            raise TrainingPlanError("The work item could not be retried")
        return {"id": domain_id, "status": "queued", "work_item_id": item.id, "reason": retained_reason}

    def execute_work_item(self, item: Any) -> Dict[str, Any]:
        operation = str(item.launch_spec.get("operation") or "")
        if operation == "prepare_model":
            return self.run_model_preparation(str(item.domain_id), allow_download=bool(item.launch_spec.get("allow_download", False))).to_dict()
        if operation == "capacity_check":
            return self.run_capacity_check(str(item.domain_id)).to_dict()
        raise TrainingPlanError(f"Unknown training-plan work operation: {operation}")

    def _scheduler(self):
        if self.scheduler is None:
            from halo_forge.workstation_jobs import WorkstationScheduler

            self.scheduler = WorkstationScheduler(self.database)
        return self.scheduler
