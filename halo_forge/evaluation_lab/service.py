"""Persistent evaluation service with truthful per-example evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sqlite3
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from halo_forge.run_db import (
    BenchmarkSuiteRecord,
    BenchmarkSuiteRevisionRecord,
    EvaluationRecord,
    RunDatabase,
)

from .adapters import (
    EvaluationAdapterRegistry,
    EvaluationContext,
    adapter_for_item,
    canonical_adapter_id,
    default_adapter_registry,
    infer_suite_adapter,
)
from .models import (
    EvaluationAdapterResult,
    EvaluationCancelled,
    EvaluationLabError,
    EvaluationLaunch,
    ResolvedSubject,
)

TERMINAL_EVALUATION_STATES = {"completed", "failed", "cancelled"}
EVALUATION_ARTIFACT_FORMAT_VERSION = 3
SUPPORTED_EVALUATION_ARTIFACT_FORMAT_VERSIONS = {2, 3}
EVALUATION_CLAIM_POLL_SECONDS = 0.05

_AGGREGATE_EVIDENCE_KINDS = {
    "aggregate",
    "aggregate_task",
    "aggregate_benchmark",
    "legacy_aggregate",
}

_EVIDENCE_FIELDS = (
    "evidence_kind",
    "valid",
    "mineable",
    "generation_seed",
    "input_tokens",
    "output_tokens",
    "finish_reason",
    "runtime_versions",
    "score_direction",
    "score_threshold",
    "coverage",
    "template_hash",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_is_alive(pid: Any) -> bool:
    try:
        value = int(pid)
        if value <= 0:
            return False
        os.kill(value, 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _record_evidence_value(sample: Any, name: str, default: Any = None) -> Any:
    """Read evidence fields from current records and metadata-backed legacy rows."""

    value = getattr(sample, name, None)
    metadata = getattr(sample, "metadata", {})
    if isinstance(metadata, Mapping):
        evidence = metadata.get("evidence")
        direct_kind = getattr(sample, "evidence_kind", None)
        migration_default = direct_kind in {None, "", "legacy"}
        if (
            isinstance(evidence, Mapping)
            and evidence.get(name) is not None
            and (value is None or migration_default)
        ):
            return evidence[name]
        if metadata.get(name) is not None and (value is None or migration_default):
            return metadata[name]
    if value is not None:
        return value
    return default


def _sample_record_dict(sample: Any) -> Dict[str, Any]:
    """Return one sample with evidence fields promoted for stable API/artifact use."""

    value = sample.to_dict() if hasattr(sample, "to_dict") else dict(sample)
    metadata = value.get("metadata") if isinstance(value.get("metadata"), Mapping) else {}
    nested = metadata.get("evidence") if isinstance(metadata, Mapping) else None
    migration_default = value.get("evidence_kind") in {None, "", "legacy"}
    for name in _EVIDENCE_FIELDS:
        if (
            isinstance(nested, Mapping)
            and name in nested
            and (value.get(name) is None or migration_default)
        ):
            value[name] = nested[name]
    if value.get("evidence_kind") is None:
        value["evidence_kind"] = "legacy"
    if value.get("valid") is None:
        value["valid"] = False
    if value.get("mineable") is None:
        value["mineable"] = False
    return value


def _normalize_sample_evidence(
    sample: Mapping[str, Any],
    *,
    revision: BenchmarkSuiteRevisionRecord,
    adapter_request: Mapping[str, Any],
    adapter_id: str,
    adapter_version: str,
) -> Dict[str, Any]:
    """Normalize evidence provenance and enforce truthful verdict semantics."""

    value = dict(sample)
    metadata = dict(value.get("metadata") or {})
    kind = str(value.get("evidence_kind") or "per_example").strip().lower()
    if metadata.get("aggregate_task_result") and kind == "per_example":
        kind = "aggregate_task"
    elif metadata.get("aggregate_benchmark_result") and kind == "per_example":
        kind = "aggregate_benchmark"
    aggregate = kind in _AGGREGATE_EVIDENCE_KINDS
    valid = bool(value.get("valid", True))
    if value.get("error"):
        valid = False
    mineable = bool(value.get("mineable", True)) and valid and not aggregate
    if aggregate:
        # Successful evaluator execution is not a per-example behavioral pass.
        value["passed"] = None

    settings = dict(revision.generation_settings or {})
    seed = value.get("generation_seed")
    if seed is None:
        seed = adapter_request.get(
            "generation_seed", adapter_request.get("seed", settings.get("seed"))
        )
    if seed is not None and not isinstance(seed, bool):
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None
    else:
        seed = None

    runtime_versions: Dict[str, Any] = {}
    for versions in (
        revision.evaluator_versions,
        adapter_request.get("runtime_versions"),
        value.get("runtime_versions"),
    ):
        if isinstance(versions, Mapping):
            runtime_versions.update({str(name): version for name, version in versions.items()})
    runtime_versions.setdefault(f"evaluation_adapter:{adapter_id}", adapter_version)
    runtime_versions.setdefault("python", platform.python_version())

    for token_field in ("input_tokens", "output_tokens"):
        token_value = value.get(token_field)
        if token_value is not None and not isinstance(token_value, bool):
            try:
                token_value = int(token_value)
            except (TypeError, ValueError):
                token_value = None
        else:
            token_value = None
        if token_value is not None and token_value < 0:
            token_value = None
        value[token_field] = token_value

    threshold = value.get("score_threshold")
    if threshold is None:
        threshold = adapter_request.get("pass_threshold")
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            threshold = None
        if threshold is not None and not math.isfinite(threshold):
            threshold = None

    coverage = value.get("coverage")
    if coverage is None:
        coverage = 1.0 if valid else 0.0
    try:
        coverage = float(coverage)
    except (TypeError, ValueError):
        coverage = 1.0 if valid else 0.0
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        coverage = 1.0 if valid else 0.0

    template_hash = value.get("template_hash") or adapter_request.get(
        "template_hash", adapter_request.get("chat_template_hash")
    )
    value.update(
        evidence_kind=kind,
        valid=valid,
        mineable=mineable,
        generation_seed=seed,
        runtime_versions=runtime_versions,
        score_direction=str(value.get("score_direction") or revision.direction),
        score_threshold=threshold,
        coverage=coverage,
        template_hash=str(template_hash) if template_hash is not None else None,
    )
    # Preserve a metadata mirror so bundles remain readable if opened by an
    # older database implementation that stores unknown fields in metadata.
    metadata["evidence"] = {name: value.get(name) for name in _EVIDENCE_FIELDS}
    value["metadata"] = metadata
    return value


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(b"file\0")
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise EvaluationLabError(f"evaluation subject path does not exist: {path}")
    digest.update(b"directory\0")
    for child in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        relative = child.relative_to(path).as_posix()
        if child.is_symlink():
            digest.update(b"symlink\0" + relative.encode("utf-8") + b"\0")
            digest.update(os.readlink(child).encode("utf-8"))
        elif child.is_file():
            digest.update(b"file\0" + relative.encode("utf-8") + b"\0")
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_artifact_identity(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the immutable portion of a published evaluation manifest."""

    return {
        "format_version": summary.get("format_version"),
        "status": summary.get("status"),
        "evaluation_id": summary.get("evaluation_id"),
        "suite_revision_id": summary.get("suite_revision_id"),
        "suite_content_hash": summary.get("suite_content_hash"),
        "adapter": summary.get("adapter"),
        "subject": summary.get("subject"),
        "reuse_key": summary.get("reuse_key"),
        "result": summary.get("result"),
        "artifact_hashes": summary.get("artifact_hashes"),
    }


def _subject_identity_hint(subject: Mapping[str, Any]) -> tuple[str, str]:
    """Validate the cheap part of a subject without inspecting model contents."""

    payload = dict(subject)
    subject_type = str(
        payload.get("type") or payload.get("subject_type") or payload.get("kind") or "model"
    )
    aliases = {"base_model": "model", "pinned_model": "model", "final": "final_model"}
    subject_type = aliases.get(subject_type, subject_type)
    if subject_type not in {"model", "run", "final_model", "checkpoint"}:
        raise EvaluationLabError(f"unsupported evaluation subject type: {subject_type}")
    subject_ref = str(
        payload.get("ref")
        or payload.get("subject_ref")
        or payload.get("model_id")
        or payload.get("run_id")
        or payload.get("path")
        or payload.get("value")
        or ""
    )
    if not subject_ref:
        raise EvaluationLabError("evaluation subject ref is required")
    return subject_type, subject_ref


def _subject_needs_deferred_resolution(subject: Mapping[str, Any]) -> bool:
    """Return whether resolving this subject may recursively hash local files."""

    payload = dict(subject)
    if payload.get("content_hash"):
        return False
    subject_type, subject_ref = _subject_identity_hint(payload)
    if subject_type in {"run", "final_model", "checkpoint"}:
        return True
    try:
        return Path(subject_ref).expanduser().exists()
    except OSError:
        # A path-like value that cannot be stat'ed is handled by the normal
        # resolver as a model ID. It cannot trigger recursive hashing here.
        return False


def _evaluation_reuse_key(
    *,
    revision: BenchmarkSuiteRevisionRecord,
    adapter_id: str,
    adapter_version: str,
    subject_hash: str,
    adapter_request: Mapping[str, Any],
) -> str:
    return _hash_json(
        {
            "suite_revision_id": revision.id,
            "suite_content_hash": revision.content_hash,
            "adapter_id": adapter_id,
            "adapter_version": adapter_version,
            "subject_hash": subject_hash,
            "request": dict(adapter_request),
        }
    )


def resolve_subject(
    subject: Mapping[str, Any], db: Optional[RunDatabase] = None
) -> ResolvedSubject:
    """Resolve a pinned model/run/final-model/checkpoint into a stable hash."""
    payload = dict(subject)
    subject_type, subject_ref = _subject_identity_hint(payload)

    hash_payload: Dict[str, Any] = {
        "type": subject_type,
        "ref": subject_ref,
        "revision": payload.get("revision"),
    }
    resolved_path: Optional[Path] = None
    if subject_type == "run":
        if db is None:
            raise EvaluationLabError("resolving a run subject requires the run database")
        run = db.get_run(subject_ref)
        if run is None:
            raise EvaluationLabError(f"unknown run subject: {subject_ref}")
        if run.status.lower() not in {"completed", "succeeded", "success"}:
            raise EvaluationLabError(f"run {subject_ref} is not completed")
        if not run.final_model_path:
            raise EvaluationLabError(f"run {subject_ref} has no final model to evaluate")
        candidate = Path(run.final_model_path).expanduser()
        if candidate.exists():
            resolved_path = candidate.resolve()
        elif not payload.get("content_hash"):
            raise EvaluationLabError(f"final model path does not exist: {candidate}")
        hash_payload.update(
            run_id=run.run_id,
            model_name=run.model_name,
            final_model_path=str(resolved_path) if resolved_path else run.final_model_path,
        )
    elif subject_type == "final_model" and payload.get("run_id"):
        if db is None:
            raise EvaluationLabError("resolving a final model run requires the run database")
        run = db.get_run(str(payload["run_id"]))
        if run is None or not run.final_model_path:
            raise EvaluationLabError(f"run has no final model: {payload['run_id']}")
        subject_ref = str(payload["run_id"])
        candidate = Path(run.final_model_path).expanduser()
        if not candidate.exists() and not payload.get("content_hash"):
            raise EvaluationLabError(f"final model path does not exist: {candidate}")
        resolved_path = candidate.resolve() if candidate.exists() else None
        hash_payload.update(run_id=run.run_id, final_model_path=str(candidate))
    elif subject_type == "checkpoint" and payload.get("run_id"):
        if db is None:
            raise EvaluationLabError("resolving a checkpoint run requires the run database")
        run = db.get_run(str(payload["run_id"]))
        if run is None:
            raise EvaluationLabError(f"unknown checkpoint run: {payload['run_id']}")
        checkpoint = str(payload.get("checkpoint") or payload.get("path") or subject_ref)
        candidate = Path(checkpoint).expanduser()
        if not candidate.is_absolute():
            output_root = Path(run.output_dir).expanduser()
            candidates = (
                output_root / checkpoint,
                output_root / "checkpoints" / checkpoint,
                output_root / f"checkpoint-{checkpoint}",
            )
            candidate = next((value for value in candidates if value.exists()), candidates[0])
        if not candidate.exists() and not payload.get("content_hash"):
            raise EvaluationLabError(f"checkpoint path does not exist: {candidate}")
        subject_ref = str(payload["run_id"])
        resolved_path = candidate.resolve() if candidate.exists() else None
        hash_payload.update(run_id=run.run_id, checkpoint=checkpoint, path=str(candidate))
    elif subject_type in {"final_model", "checkpoint"}:
        candidate = Path(str(payload.get("path") or subject_ref)).expanduser()
        if not candidate.exists() and not payload.get("content_hash"):
            raise EvaluationLabError(f"evaluation subject path does not exist: {candidate}")
        resolved_path = candidate.resolve() if candidate.exists() else None
        hash_payload["path"] = str(resolved_path or candidate)
    elif subject_type == "model":
        candidate = Path(subject_ref).expanduser()
        if candidate.exists():
            resolved_path = candidate.resolve()
            hash_payload["path"] = str(resolved_path)

    content_hash = str(payload.get("content_hash") or "")
    if not content_hash and resolved_path is not None:
        content_hash = _hash_path(resolved_path)
    if content_hash:
        hash_payload["content_hash"] = content_hash
    payload["resolved_path"] = str(resolved_path) if resolved_path else None
    payload["content_hash"] = content_hash or None
    return ResolvedSubject(
        subject_type=subject_type,
        subject_ref=subject_ref,
        subject_hash=_hash_json(hash_payload),
        payload=payload,
    )


class EvaluationJobManager:
    """Run at most one persistent evaluation job and publish artifacts atomically."""

    def __init__(
        self,
        db: RunDatabase,
        artifact_root: Path | str | None = None,
        *,
        registry: Optional[EvaluationAdapterRegistry] = None,
        max_workers: int = 1,
        scheduler: Optional[Any] = None,
    ):
        if max_workers != 1:
            raise EvaluationLabError("Dataset Lab v2 supports one active evaluation job")
        self.db = db
        configured = artifact_root or os.environ.get("HALOFORGE_EVALUATION_ROOT")
        self.artifact_root = Path(configured or (Path.home() / ".halo-forge" / "evaluations"))
        self.artifact_root = self.artifact_root.expanduser().resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or default_adapter_registry()
        self.scheduler = scheduler
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evaluation-lab")
        self._futures: Dict[str, Future[Any]] = {}
        self._lock = threading.RLock()
        self.worker_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self.recover_orphaned()

    def _worker_envelope(self) -> Dict[str, Any]:
        return {
            "pid": os.getpid(),
            "worker_id": self.worker_id,
            "artifact_root": str(self.artifact_root),
        }

    def recover_orphaned(self) -> int:
        """Mark only jobs whose owning process is no longer alive interrupted."""

        recovered = 0
        for status in ("queued", "running"):
            for evaluation in self.db.list_evaluations(status=status, limit=100_000):
                if (
                    status == "queued"
                    and evaluation.work_item_id
                    and evaluation.request.get("execution") == "workstation_scheduler"
                ):
                    # The scheduler owns this queue state; it survives the API
                    # process that created the evaluation.
                    continue
                worker = evaluation.request.get("worker") or {}
                if _pid_is_alive(worker.get("pid")):
                    continue
                if self.db.interrupt_evaluation_if_unchanged(
                    evaluation.id,
                    expected_status=status,
                    expected_request_json=evaluation.request_json,
                    error="evaluation worker stopped before completion",
                ):
                    recovered += 1
        return recovered

    def _claim_or_wait(self, evaluation_id: str) -> Optional[EvaluationRecord]:
        """Wait cooperatively for the SQLite-global evaluation slot."""

        while True:
            evaluation = self.db.get_evaluation(evaluation_id)
            if evaluation is None:
                return None
            if evaluation.cancel_requested:
                if evaluation.status in {"queued", "interrupted"}:
                    self.db.update_evaluation(
                        evaluation_id,
                        status="cancelled",
                        stage="cancelled",
                        completed_at=_now(),
                    )
                return None
            if evaluation.status in TERMINAL_EVALUATION_STATES:
                return None
            if evaluation.status == "running":
                # A duplicate worker invocation must never execute a job that
                # another worker already owns.
                return None
            if evaluation.status not in {"queued", "interrupted"}:
                return None

            self.recover_orphaned()
            refreshed = self.db.get_evaluation(evaluation_id)
            if refreshed is None:
                return None
            if refreshed.cancel_requested:
                continue
            subject_pending = (
                isinstance(refreshed.request.get("subject_input"), Mapping)
                and refreshed.request.get("subject_resolution") != "resolved"
            )
            claimed = self.db.claim_evaluation(
                evaluation_id,
                worker=self._worker_envelope(),
                stage="resolving_subject" if subject_pending else "starting",
            )
            if claimed is not None:
                return claimed
            time.sleep(EVALUATION_CLAIM_POLL_SECONDS)

    def launch(
        self,
        *,
        suite_revision_id: str,
        adapter_id: Optional[str] = None,
        subject: Mapping[str, Any],
        request: Optional[Mapping[str, Any]] = None,
        evaluation_id: Optional[str] = None,
        dependencies: Sequence[str] = (),
        submit: bool = True,
    ) -> EvaluationLaunch:
        revision = self.db.get_benchmark_suite_revision(suite_revision_id)
        if revision is None:
            raise EvaluationLabError(f"unknown benchmark suite revision: {suite_revision_id}")
        resolved_adapter_id = (
            canonical_adapter_id(adapter_id) if adapter_id else infer_suite_adapter(revision)
        )
        adapter = self.registry.get(resolved_adapter_id)
        evaluator_versions = dict(revision.evaluator_versions or {})
        selected_adapter_ids = (
            {adapter_for_item(item) for item in revision.items}
            if resolved_adapter_id == "suite"
            else {resolved_adapter_id}
        )
        for selected_id in selected_adapter_ids:
            selected = self.registry.get(selected_id)
            expected = evaluator_versions.get(selected_id)
            if expected is not None and str(expected) != selected.adapter_version:
                raise EvaluationLabError(
                    f"Suite pins {selected_id}@{expected}, but the registered evaluator is "
                    f"{selected_id}@{selected.adapter_version}"
                )
        version_resolver = getattr(adapter, "version_for", None)
        adapter_version = (
            str(version_resolver(revision))
            if callable(version_resolver)
            else adapter.adapter_version
        )
        supplied_request = dict(request or {})
        generation_settings = dict(revision.generation_settings or {})
        adapter_request = {**generation_settings, **supplied_request}
        adapter_request["generation_settings"] = {
            **generation_settings,
            **dict(supplied_request.get("generation_settings") or {}),
        }
        identifier = evaluation_id or uuid.uuid4().hex
        scheduler_submit = self.scheduler is not None and submit
        work_dependencies = tuple(
            dict.fromkeys(str(value).strip() for value in dependencies if str(value).strip())
        )
        work_item_id = f"evaluation-{identifier}" if scheduler_submit else None
        subject_payload = dict(subject)
        subject_type, subject_ref = _subject_identity_hint(subject_payload)
        suite = self.db.get_benchmark_suite(revision.suite_id)
        if suite is None:
            raise EvaluationLabError(f"unknown benchmark suite: {revision.suite_id}")
        if suite.purpose == "holdout":
            if subject_type == "checkpoint":
                raise EvaluationLabError("holdout suites cannot evaluate intermediate checkpoints")
            if subject_type == "model":
                try:
                    local_subject = Path(subject_ref).expanduser().exists()
                except OSError:
                    local_subject = False
                if not (
                    local_subject
                    or subject_payload.get("revision")
                    or subject_payload.get("content_hash")
                ):
                    raise EvaluationLabError(
                        "holdout suites require a pinned model revision, content hash, "
                        "completed run, or final model"
                    )
        deferred = _subject_needs_deferred_resolution(subject_payload)
        if deferred:
            # Use evaluation-scoped placeholders so the row can be committed
            # before a multi-gigabyte checkpoint tree is inspected. These are
            # replaced transactionally by the worker before adapter execution.
            subject_hash = _hash_json(
                {
                    "state": "pending",
                    "evaluation_id": identifier,
                    "subject_type": subject_type,
                    "subject_ref": subject_ref,
                }
            )
            reuse_key = _hash_json(
                {"state": "pending", "evaluation_id": identifier, "kind": "evaluation"}
            )
            envelope = {
                "adapter_request": adapter_request,
                "subject_input": subject_payload,
                "subject_resolution": "pending",
                "work_dependencies": list(work_dependencies),
                "worker": self._worker_envelope(),
                "execution": ("workstation_scheduler" if scheduler_submit else "local_executor"),
            }
        else:
            resolved = resolve_subject(subject_payload, self.db)
            subject_type = resolved.subject_type
            subject_ref = resolved.subject_ref
            subject_hash = resolved.subject_hash
            reuse_key = _evaluation_reuse_key(
                revision=revision,
                adapter_id=adapter.adapter_id,
                adapter_version=adapter_version,
                subject_hash=subject_hash,
                adapter_request=adapter_request,
            )
            reused = self.db.find_completed_evaluation(reuse_key)
            if reused is not None:
                if reused.artifact_path and Path(reused.artifact_path).is_dir():
                    try:
                        self._verify_published(reused, Path(reused.artifact_path))
                    except (EvaluationLabError, OSError) as exc:
                        self.db.update_evaluation(
                            reused.id,
                            status="failed",
                            stage="corrupted_artifact",
                            error=str(exc),
                        )
                    else:
                        # Backfill exposure rows when reusing a completed v2
                        # evaluation that predates the ledger integration.
                        self._record_launch_exposures(
                            evaluation=reused,
                            revision=revision,
                            suite=suite,
                            subject=subject_payload,
                            subject_resolution="resolved",
                        )
                        return EvaluationLaunch(evaluation=reused, reused=True)
                else:
                    self.db.update_evaluation(
                        reused.id,
                        status="failed",
                        stage="missing_artifact",
                        error="completed evaluation artifact is missing",
                    )
            envelope = {
                "adapter_request": adapter_request,
                "subject": resolved.to_dict(),
                "subject_resolution": "resolved",
                "work_dependencies": list(work_dependencies),
                "worker": self._worker_envelope(),
                "execution": ("workstation_scheduler" if scheduler_submit else "local_executor"),
            }
        evaluation = self.db.create_evaluation(
            evaluation_id=identifier,
            suite_revision_id=revision.id,
            adapter_id=adapter.adapter_id,
            adapter_version=adapter_version,
            subject_type=subject_type,
            subject_ref=subject_ref,
            subject_hash=subject_hash,
            reuse_key=reuse_key,
            request=envelope,
            work_item_id=work_item_id,
        )
        self._record_launch_exposures(
            evaluation=evaluation,
            revision=revision,
            suite=suite,
            subject=subject_payload,
            subject_resolution="pending" if deferred else "resolved",
        )
        if submit:
            self._submit(evaluation.id, dependencies=work_dependencies)
        return EvaluationLaunch(evaluation=evaluation, reused=False)

    start = launch

    def _record_launch_exposures(
        self,
        *,
        evaluation: EvaluationRecord,
        revision: BenchmarkSuiteRevisionRecord,
        suite: BenchmarkSuiteRecord,
        subject: Mapping[str, Any],
        subject_resolution: str,
    ) -> None:
        """Append auditable suite-item exposure rows for identifiable subjects.

        A pinned external model has no catalog identity to attach an exposure to,
        so those launches remain valid but do not fabricate a target.  Managed
        runs/checkpoints/artifacts and Dataset Lab-backed suite items do have
        durable identities and are recorded at launch time.  Retries reuse the
        original evaluation row and therefore do not append duplicate events.
        """

        subject_type, subject_ref = _subject_identity_hint(subject)
        run_id: Optional[str] = None
        if subject.get("run_id"):
            run_id = str(subject["run_id"])
        elif subject_type == "run":
            run_id = subject_ref
        elif subject_type in {"final_model", "checkpoint"}:
            # A path-only final model/checkpoint must not be mistaken for a run
            # ID.  Accept the reference only when it is cataloged as a run.
            if self.db.get_run(subject_ref) is not None:
                run_id = subject_ref

        artifact_id = str(
            subject.get("model_artifact_id") or subject.get("artifact_id") or ""
        ).strip()
        artifact = self.db.get_model_artifact(artifact_id) if artifact_id else None
        if artifact is None:
            artifact_id = ""
        else:
            run_id = run_id or artifact.run_id

        run_group_id = str(subject.get("run_group_id") or "").strip()
        if not run_group_id and artifact is not None:
            run_group_id = str(artifact.run_group_id or "")
        if run_group_id and self.db.get_run_group(run_group_id) is None:
            run_group_id = ""

        existing_keys = {
            (
                value.suite_item_id,
                value.dataset_version_id,
                value.run_group_id,
                value.run_id,
                value.model_artifact_id,
            )
            for value in self.db.list_exposures(suite_revision_id=revision.id)
            if value.exposure_type == "evaluation"
            and value.provenance.get("evaluation_id") == evaluation.id
        }

        for index, raw_item in enumerate(revision.items):
            item = dict(raw_item) if isinstance(raw_item, Mapping) else {}
            dataset_version_id = str(item.get("dataset_version_id") or "").strip()
            if dataset_version_id and self.db.get_dataset_version(dataset_version_id) is None:
                # Filesystem-only Dataset Lab versions remain supported.  Their
                # exposure can be synchronized after the SQLite catalog mirrors
                # the version via DatasetLab.sync_version_exposures().
                dataset_version_id = ""
            if not any((dataset_version_id, run_group_id, run_id, artifact_id)):
                continue
            suite_item_id = str(item.get("id") or item.get("suite_item_id") or index)
            exposure_key = (
                suite_item_id,
                dataset_version_id or None,
                run_group_id or None,
                run_id,
                artifact_id or None,
            )
            if exposure_key in existing_keys:
                continue
            self.db.record_exposure(
                suite_revision_id=revision.id,
                suite_item_id=suite_item_id,
                exposure_type="evaluation",
                dataset_version_id=dataset_version_id or None,
                run_group_id=run_group_id or None,
                run_id=run_id,
                model_artifact_id=artifact_id or None,
                provenance={
                    "source": "evaluation_launch",
                    "evaluation_id": evaluation.id,
                    "suite_id": revision.suite_id,
                    "suite_purpose": suite.purpose,
                    "subject_type": subject_type,
                    "subject_ref": subject_ref,
                    "subject_hash": evaluation.subject_hash,
                    "subject_resolution": subject_resolution,
                    "record_id": item.get("record_id"),
                    "dataset_id": item.get("dataset_id"),
                    "dataset_split": item.get("split"),
                },
            )
            existing_keys.add(exposure_key)

    def _submit(
        self,
        evaluation_id: str,
        *,
        dependencies: Sequence[str] = (),
    ) -> None:
        if self.scheduler is not None:
            evaluation = self.get(evaluation_id)
            persisted_dependencies = tuple(
                str(value).strip()
                for value in evaluation.request.get("work_dependencies") or ()
                if str(value).strip()
            )
            work_dependencies = tuple(
                dict.fromkeys(
                    [
                        *(str(value).strip() for value in dependencies if str(value).strip()),
                        *persisted_dependencies,
                    ]
                )
            )
            work_item_id = evaluation.work_item_id or f"evaluation-{evaluation.id}"
            canonical_run_id = (
                evaluation.subject_ref
                if self.db.get_run(evaluation.subject_ref) is not None
                else None
            )
            if not evaluation.work_item_id:
                self.db.update_evaluation(evaluation.id, work_item_id=work_item_id)
            try:
                self.scheduler.enqueue(
                    kind="evaluation",
                    launch_spec={
                        "handler": "evaluation_lab.run_queued",
                        "evaluation_id": evaluation.id,
                        "artifact_root": str(self.artifact_root),
                    },
                    resource_class="accelerator",
                    resource_requirements={"output_path": str(self.artifact_root)},
                    domain_kind="evaluation",
                    domain_id=evaluation.id,
                    canonical_run_id=canonical_run_id,
                    dependencies=work_dependencies,
                    max_retries=2,
                    work_item_id=work_item_id,
                )
            except Exception as exc:
                self.db.update_evaluation(
                    evaluation.id,
                    status="failed",
                    stage="enqueue_failed",
                    error=f"Could not enqueue durable workstation work: {exc}",
                    completed_at=_now(),
                    work_item_id=None,
                )
                raise
            return
        with self._lock:
            self._futures[evaluation_id] = self._executor.submit(self._run, evaluation_id)

    def run_queued(self, evaluation_id: str) -> EvaluationRecord:
        """Claim and execute a queued job in the current process.

        The CLI uses this entry point from a detached worker so a non-waiting
        launch can return immediately without losing the persistent job.
        """

        evaluation = self.get(evaluation_id)
        if evaluation.status not in {"queued", "interrupted"}:
            raise EvaluationLabError(
                f"only queued or interrupted evaluations can be claimed; got {evaluation.status}"
            )
        request = dict(evaluation.request)
        request["worker"] = self._worker_envelope()
        self.db.update_evaluation(
            evaluation_id,
            status="queued",
            stage="queued",
            error=None,
            completed_at=None,
            request=request,
        )
        self._run(evaluation_id)
        return self.get(evaluation_id)

    def _verify_published(
        self, evaluation: EvaluationRecord, path: Path
    ) -> tuple[Dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        """Load a published bundle only after verifying identity and every file."""

        try:
            summary = json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationLabError(
                f"invalid evaluation artifact manifest at {path}: {exc}"
            ) from exc
        if not isinstance(summary, Mapping):
            raise EvaluationLabError(f"evaluation artifact manifest must be an object: {path}")
        if summary.get("format_version") not in SUPPORTED_EVALUATION_ARTIFACT_FORMAT_VERSIONS:
            raise EvaluationLabError(
                f"evaluation artifact has no verifiable integrity manifest: {path}"
            )
        if summary.get("status") != "complete":
            raise EvaluationLabError(f"evaluation artifact is incomplete: {path}")
        if summary.get("evaluation_id") != evaluation.id:
            raise EvaluationLabError(f"published artifact identity mismatch: {path}")
        if summary.get("suite_revision_id") != evaluation.suite_revision_id:
            raise EvaluationLabError(f"published suite revision identity mismatch: {path}")
        adapter_value = summary.get("adapter") or {}
        subject_value = summary.get("subject") or {}
        if not isinstance(adapter_value, Mapping) or not isinstance(subject_value, Mapping):
            raise EvaluationLabError(f"published evaluation identity is malformed: {path}")
        adapter = dict(adapter_value)
        if adapter != {"id": evaluation.adapter_id, "version": evaluation.adapter_version}:
            raise EvaluationLabError(f"published adapter identity mismatch: {path}")
        subject = dict(subject_value)
        if subject.get("subject_hash") != evaluation.subject_hash:
            raise EvaluationLabError(f"published subject identity mismatch: {path}")
        if summary.get("reuse_key") != evaluation.reuse_key:
            raise EvaluationLabError(f"published evaluation configuration mismatch: {path}")
        revision = self.db.get_benchmark_suite_revision(evaluation.suite_revision_id)
        if revision is None or summary.get("suite_content_hash") != revision.content_hash:
            raise EvaluationLabError(f"published suite content identity mismatch: {path}")

        artifact_hash = str(summary.get("artifact_hash") or "")
        computed_hash = _hash_json(_evaluation_artifact_identity(summary))
        if not artifact_hash or artifact_hash != computed_hash:
            raise EvaluationLabError(
                f"evaluation artifact manifest changed after publication: {path}"
            )
        catalog_hash = str((evaluation.result or {}).get("artifact_hash") or "")
        if catalog_hash and catalog_hash != artifact_hash:
            raise EvaluationLabError(
                f"evaluation artifact no longer matches its catalog record: {path}"
            )

        hashes_value = summary.get("artifact_hashes") or {}
        if not isinstance(hashes_value, Mapping):
            raise EvaluationLabError(f"evaluation artifact file hashes are malformed: {path}")
        recorded_hashes = {
            str(relative): str(expected) for relative, expected in hashes_value.items()
        }
        actual_files: set[str] = set()
        for artifact in path.rglob("*"):
            if artifact.is_symlink():
                raise EvaluationLabError(
                    f"evaluation artifact contains a symbolic link: {artifact}"
                )
            if artifact.is_file() and artifact.name != "evaluation.json":
                actual_files.add(artifact.relative_to(path).as_posix())
        if actual_files != set(recorded_hashes):
            raise EvaluationLabError(f"evaluation artifact file inventory changed: {path}")
        for relative, expected in recorded_hashes.items():
            artifact = (path / relative).resolve()
            try:
                artifact.relative_to(path.resolve())
            except ValueError as exc:
                raise EvaluationLabError(
                    f"evaluation artifact path escapes its bundle: {relative}"
                ) from exc
            if not artifact.is_file() or _hash_file(artifact) != expected:
                raise EvaluationLabError(
                    f"evaluation artifact file changed after publication: {relative}"
                )

        try:
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationLabError(f"invalid published evaluation metrics: {exc}") from exc
        if not isinstance(metrics, list):
            raise EvaluationLabError("published evaluation metrics must be a list")
        if not all(isinstance(metric, Mapping) for metric in metrics):
            raise EvaluationLabError("published evaluation metrics must be objects")
        samples: list[dict[str, Any]] = []
        sample_path = path / "samples.jsonl"
        try:
            if sample_path.exists():
                for line in sample_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise EvaluationLabError("published evaluation samples must be objects")
                        samples.append(value)
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationLabError(f"invalid published evaluation samples: {exc}") from exc
        return summary, metrics, samples

    def _read_published(self, evaluation_id: str, path: Path) -> EvaluationRecord:
        evaluation = self.db.get_evaluation(evaluation_id)
        if evaluation is None:
            raise EvaluationLabError(f"unknown evaluation: {evaluation_id}")
        summary, metrics, samples = self._verify_published(evaluation, path)
        result_value = summary.get("result") or {}
        if not isinstance(result_value, Mapping):
            raise EvaluationLabError("published evaluation result must be an object")
        result = dict(result_value)
        result["artifact_hash"] = str(summary["artifact_hash"])
        return self.db.complete_evaluation(
            evaluation_id,
            metrics=metrics,
            samples=samples,
            result=result,
            artifact_path=str(path),
        )

    def _verified_completed_reuse(self, reuse_key: str) -> Optional[
        tuple[
            EvaluationRecord,
            tuple[Dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]],
        ]
    ]:
        """Return a reusable result only after validating its immutable bundle."""

        reused = self.db.find_completed_evaluation(reuse_key)
        if reused is None:
            return None
        if not reused.artifact_path or not Path(reused.artifact_path).is_dir():
            self.db.update_evaluation(
                reused.id,
                status="failed",
                stage="missing_artifact",
                error="completed evaluation artifact is missing",
            )
            return None
        try:
            published = self._verify_published(reused, Path(reused.artifact_path))
        except (EvaluationLabError, OSError) as exc:
            self.db.update_evaluation(
                reused.id,
                status="failed",
                stage="corrupted_artifact",
                error=str(exc),
            )
            return None
        return reused, published

    def _publish_reused_result(
        self,
        *,
        evaluation: EvaluationRecord,
        source: EvaluationRecord,
        published: tuple[Dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]],
        revision: BenchmarkSuiteRevisionRecord,
        subject: ResolvedSubject,
        canonical_reuse_key: str,
        final_path: Path,
    ) -> EvaluationRecord:
        """Copy a reused result into an artifact owned by the allocated job ID.

        Deferred launch has already returned ``evaluation.id`` to its caller, so
        it cannot simply substitute the older evaluation record. Publishing a
        fresh, verified bundle avoids sharing an artifact whose manifest names a
        different evaluation while still skipping adapter/model execution.
        """

        source_summary, metrics, samples = published
        staging = self.artifact_root / f".{evaluation.id}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=False, exist_ok=False)
        try:
            source_path = Path(source.artifact_path or "")
            for child in source_path.iterdir():
                if child.name in {"evaluation.json", "metrics.json", "samples.jsonl"}:
                    continue
                destination = staging / child.name
                if child.is_dir():
                    shutil.copytree(child, destination)
                elif child.is_file():
                    shutil.copy2(child, destination)

            (staging / "metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            with (staging / "samples.jsonl").open("w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample, sort_keys=True, default=str) + "\n")

            result_value = source_summary.get("result") or {}
            if not isinstance(result_value, Mapping):
                raise EvaluationLabError("reused evaluation result must be an object")
            result_payload = dict(result_value)
            adapter_summary = dict(result_payload.get("summary") or {})
            adapter_summary.update(
                reused_from_evaluation_id=source.id,
                canonical_reuse_key=canonical_reuse_key,
            )
            result_payload["summary"] = adapter_summary
            result_payload["metric_count"] = len(metrics)
            result_payload["sample_count"] = len(samples)
            artifact_hashes = {
                artifact.relative_to(staging).as_posix(): _hash_file(artifact)
                for artifact in sorted(staging.rglob("*"))
                if artifact.is_file() and artifact.name != "evaluation.json"
            }
            artifact_summary = {
                "format_version": EVALUATION_ARTIFACT_FORMAT_VERSION,
                "status": "complete",
                "evaluation_id": evaluation.id,
                "suite_revision_id": revision.id,
                "suite_content_hash": revision.content_hash,
                "adapter": {
                    "id": evaluation.adapter_id,
                    "version": evaluation.adapter_version,
                },
                "subject": subject.to_dict(),
                "reuse_key": evaluation.reuse_key,
                "result": result_payload,
                "artifact_hashes": artifact_hashes,
            }
            artifact_summary["artifact_hash"] = _hash_json(
                _evaluation_artifact_identity(artifact_summary)
            )
            (staging / "evaluation.json").write_text(
                json.dumps(artifact_summary, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            self.check_cancelled(evaluation.id)
            os.rename(staging, final_path)
            return self._read_published(evaluation.id, final_path)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def _run(self, evaluation_id: str) -> None:
        staging: Optional[Path] = None
        try:
            evaluation = self._claim_or_wait(evaluation_id)
            if evaluation is None:
                return
            if evaluation.cancel_requested:
                self.db.update_evaluation(
                    evaluation_id, status="cancelled", stage="cancelled", completed_at=_now()
                )
                return
            final_path = self.artifact_root / evaluation_id
            if final_path.is_dir():
                self._read_published(evaluation_id, final_path)
                return
            envelope = dict(evaluation.request)
            # Evaluation rows created before deferred subject resolution have a
            # resolved ``subject`` envelope but no explicit state marker. Keep
            # those legacy jobs runnable rather than mistaking them for a new
            # pending checkpoint that requires ``subject_input``.
            subject_pending = (
                isinstance(envelope.get("subject_input"), Mapping)
                and envelope.get("subject_resolution") != "resolved"
            )
            self.db.update_evaluation(
                evaluation_id,
                stage="resolving_subject" if subject_pending else "starting",
                error=None,
            )
            revision = self.db.get_benchmark_suite_revision(evaluation.suite_revision_id)
            if revision is None:
                raise EvaluationLabError(
                    f"missing benchmark suite revision: {evaluation.suite_revision_id}"
                )
            adapter = self.registry.get(evaluation.adapter_id)
            version_resolver = getattr(adapter, "version_for", None)
            resolved_adapter_version = (
                str(version_resolver(revision))
                if callable(version_resolver)
                else adapter.adapter_version
            )
            if resolved_adapter_version != evaluation.adapter_version:
                raise EvaluationLabError(
                    f"adapter version changed: expected {evaluation.adapter_version}, "
                    f"found {resolved_adapter_version}"
                )
            adapter_request = dict(envelope.get("adapter_request") or {})
            if subject_pending:
                subject_input = envelope.get("subject_input")
                if not isinstance(subject_input, Mapping):
                    raise EvaluationLabError("pending evaluation has no subject input")
                self.check_cancelled(evaluation_id)
                self.log(evaluation_id, "resolving evaluation subject identity")
                subject = resolve_subject(subject_input, self.db)
                self.check_cancelled(evaluation_id)
                canonical_reuse_key = _evaluation_reuse_key(
                    revision=revision,
                    adapter_id=adapter.adapter_id,
                    adapter_version=resolved_adapter_version,
                    subject_hash=subject.subject_hash,
                    adapter_request=adapter_request,
                )
            else:
                subject_value = envelope.get("subject")
                if not isinstance(subject_value, Mapping):
                    raise EvaluationLabError("resolved evaluation has no subject identity")
                subject = ResolvedSubject(**dict(subject_value))
                canonical_reuse_key = str(
                    envelope.get("canonical_reuse_key") or evaluation.reuse_key
                )

            reusable = self._verified_completed_reuse(canonical_reuse_key)
            if reusable is not None and reusable[0].id != evaluation_id:
                source, published = reusable
                alias_reuse_key = _hash_json(
                    {
                        "canonical_reuse_key": canonical_reuse_key,
                        "evaluation_id": evaluation_id,
                        "reused_from_evaluation_id": source.id,
                    }
                )
                envelope.update(
                    subject=subject.to_dict(),
                    subject_resolution="resolved",
                    canonical_reuse_key=canonical_reuse_key,
                    reused_from_evaluation_id=source.id,
                )
                updated = self.db.update_evaluation(
                    evaluation_id,
                    subject_type=subject.subject_type,
                    subject_ref=subject.subject_ref,
                    subject_hash=subject.subject_hash,
                    reuse_key=alias_reuse_key,
                    request=envelope,
                    stage="reusing_completed",
                )
                if updated is None:
                    raise EvaluationLabError(f"unknown evaluation: {evaluation_id}")
                self.check_cancelled(evaluation_id)
                self.log(evaluation_id, f"reusing completed evaluation {source.id}")
                self._publish_reused_result(
                    evaluation=updated,
                    source=source,
                    published=published,
                    revision=revision,
                    subject=subject,
                    canonical_reuse_key=canonical_reuse_key,
                    final_path=final_path,
                )
                return

            if subject_pending:
                envelope.update(
                    subject=subject.to_dict(),
                    subject_resolution="resolved",
                    canonical_reuse_key=canonical_reuse_key,
                )
                updated = self.db.update_evaluation(
                    evaluation_id,
                    subject_type=subject.subject_type,
                    subject_ref=subject.subject_ref,
                    subject_hash=subject.subject_hash,
                    reuse_key=canonical_reuse_key,
                    request=envelope,
                    stage="starting",
                )
                if updated is None:
                    raise EvaluationLabError(f"unknown evaluation: {evaluation_id}")
                evaluation = updated
            staging = self.artifact_root / f".{evaluation_id}.{uuid.uuid4().hex}.tmp"
            staging.mkdir(parents=False, exist_ok=False)
            context = EvaluationContext(self, evaluation_id, staging)
            context.log(f"starting {adapter.adapter_id}@{resolved_adapter_version}")
            result = adapter.evaluate(context, revision, subject, adapter_request)
            if not isinstance(result, EvaluationAdapterResult):
                raise EvaluationLabError("evaluation adapter returned an invalid result")
            context.check_cancelled()
            metrics = [metric.to_dict() for metric in result.metrics]
            samples = [
                _normalize_sample_evidence(
                    sample.to_dict(),
                    revision=revision,
                    adapter_request=adapter_request,
                    adapter_id=adapter.adapter_id,
                    adapter_version=resolved_adapter_version,
                )
                for sample in result.samples
            ]
            seen_metrics: set[tuple[str, str]] = set()
            for metric in metrics:
                metric["name"] = str(metric.get("name") or "").strip()
                metric["suite_item_id"] = str(metric.get("suite_item_id") or "")
                metric["direction"] = str(metric.get("direction") or revision.direction)
                if not metric["name"]:
                    raise EvaluationLabError("evaluation metric name is required")
                if metric["direction"] not in {"maximize", "minimize"}:
                    raise EvaluationLabError(f"invalid metric direction: {metric['direction']}")
                metric["value"] = float(metric["value"])
                if not math.isfinite(metric["value"]):
                    raise EvaluationLabError("evaluation metrics must be finite")
                key = (metric["name"], metric["suite_item_id"])
                if key in seen_metrics:
                    raise EvaluationLabError(f"duplicate evaluation metric: {key}")
                seen_metrics.add(key)
            for index, sample in enumerate(samples):
                sample["suite_item_id"] = str(sample.get("suite_item_id") or index)
                if sample.get("score") is not None:
                    sample["score"] = float(sample["score"])
                    if not math.isfinite(sample["score"]):
                        raise EvaluationLabError("evaluation sample scores must be finite")
                if sample.get("score_direction") not in {"maximize", "minimize"}:
                    raise EvaluationLabError(
                        "evaluation sample score_direction must be maximize or minimize"
                    )
            result_payload = {
                "summary": {
                    **dict(result.summary),
                    "evidence": {
                        "total": len(samples),
                        "valid": sum(bool(sample["valid"]) for sample in samples),
                        "mineable": sum(bool(sample["mineable"]) for sample in samples),
                        "aggregate": sum(
                            sample["evidence_kind"] in _AGGREGATE_EVIDENCE_KINDS
                            for sample in samples
                        ),
                        "invalid": sum(not bool(sample["valid"]) for sample in samples),
                    },
                },
                "metric_count": len(metrics),
                "sample_count": len(samples),
            }
            (staging / "metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            with (staging / "samples.jsonl").open("w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample, sort_keys=True, default=str) + "\n")
            artifact_hashes = {
                artifact.relative_to(staging).as_posix(): _hash_file(artifact)
                for artifact in sorted(staging.rglob("*"))
                if artifact.is_file() and artifact.name != "evaluation.json"
            }
            artifact_summary = {
                "format_version": EVALUATION_ARTIFACT_FORMAT_VERSION,
                "status": "complete",
                "evaluation_id": evaluation_id,
                "suite_revision_id": revision.id,
                "suite_content_hash": revision.content_hash,
                "adapter": {"id": adapter.adapter_id, "version": resolved_adapter_version},
                "subject": subject.to_dict(),
                "reuse_key": evaluation.reuse_key,
                "result": result_payload,
                "artifact_hashes": artifact_hashes,
            }
            artifact_summary["artifact_hash"] = _hash_json(
                _evaluation_artifact_identity(artifact_summary)
            )
            (staging / "evaluation.json").write_text(
                json.dumps(artifact_summary, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            context.check_cancelled()
            os.rename(staging, final_path)
            staging = None
            try:
                self._read_published(evaluation_id, final_path)
            except sqlite3.IntegrityError:
                # Another process may have completed the same canonical
                # evaluation after our pre-execution reuse check. The partial
                # unique index correctly rejects the duplicate. Replace this
                # unpublished bundle with a separately identified verified
                # copy of the winner so the already-returned job ID remains
                # usable without sharing a mismatched artifact manifest.
                canonical_reuse_key = evaluation.reuse_key
                reusable = self._verified_completed_reuse(canonical_reuse_key)
                if reusable is None or reusable[0].id == evaluation_id:
                    raise
                source, published = reusable
                shutil.rmtree(final_path, ignore_errors=True)
                alias_reuse_key = _hash_json(
                    {
                        "canonical_reuse_key": canonical_reuse_key,
                        "evaluation_id": evaluation_id,
                        "reused_from_evaluation_id": source.id,
                    }
                )
                envelope.update(
                    subject=subject.to_dict(),
                    subject_resolution="resolved",
                    canonical_reuse_key=canonical_reuse_key,
                    reused_from_evaluation_id=source.id,
                )
                updated = self.db.update_evaluation(
                    evaluation_id,
                    reuse_key=alias_reuse_key,
                    request=envelope,
                    stage="reusing_completed",
                )
                if updated is None:
                    raise EvaluationLabError(f"unknown evaluation: {evaluation_id}")
                self._publish_reused_result(
                    evaluation=updated,
                    source=source,
                    published=published,
                    revision=revision,
                    subject=subject,
                    canonical_reuse_key=canonical_reuse_key,
                    final_path=final_path,
                )
        except EvaluationCancelled as exc:
            self.db.update_evaluation(
                evaluation_id,
                status="cancelled",
                stage="cancelled",
                error=str(exc),
                completed_at=_now(),
            )
        except Exception as exc:
            current = self.db.get_evaluation(evaluation_id)
            logs = list(current.logs if current else [])
            logs.append(traceback.format_exc())
            self.db.update_evaluation(
                evaluation_id,
                status="failed",
                stage="failed",
                error=f"{type(exc).__name__}: {exc}",
                logs=logs[-1000:],
                completed_at=_now(),
            )
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)

    def get(self, evaluation_id: str) -> EvaluationRecord:
        evaluation = self.db.get_evaluation(evaluation_id)
        if evaluation is None:
            raise EvaluationLabError(f"unknown evaluation: {evaluation_id}")
        return evaluation

    def wait(self, evaluation_id: str, timeout: Optional[float] = None) -> EvaluationRecord:
        future = self._futures.get(evaluation_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.get(evaluation_id)

    def check_cancelled(self, evaluation_id: str) -> None:
        if self.get(evaluation_id).cancel_requested:
            raise EvaluationCancelled("evaluation cancellation requested")

    def update_progress(
        self,
        evaluation_id: str,
        *,
        processed: Optional[int] = None,
        total: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> None:
        changes: Dict[str, Any] = {}
        if processed is not None:
            changes["processed_samples"] = max(0, int(processed))
        if total is not None:
            changes["total_samples"] = max(0, int(total))
        if stage is not None:
            changes["stage"] = str(stage)
        self.db.update_evaluation(evaluation_id, **changes)

    def log(self, evaluation_id: str, message: str) -> None:
        evaluation = self.get(evaluation_id)
        logs = list(evaluation.logs)
        logs.append(str(message))
        self.db.update_evaluation(evaluation_id, logs=logs[-1000:])

    def cancel(self, evaluation_id: str) -> EvaluationRecord:
        evaluation = self.get(evaluation_id)
        if evaluation.status in TERMINAL_EVALUATION_STATES:
            return evaluation
        self.db.update_evaluation(evaluation_id, cancel_requested=True)
        if self.scheduler is not None and evaluation.work_item_id:
            work = self.scheduler.cancel(evaluation.work_item_id)
            if work is not None and work.status == "cancelled":
                self.db.update_evaluation(
                    evaluation_id,
                    status="cancelled",
                    stage="cancelled",
                    completed_at=_now(),
                )
        future = self._futures.get(evaluation_id)
        if future is not None and future.cancel():
            self.db.update_evaluation(
                evaluation_id, status="cancelled", stage="cancelled", completed_at=_now()
            )
        return self.get(evaluation_id)

    def retry(self, evaluation_id: str, *, submit: bool = True) -> EvaluationRecord:
        evaluation = self.get(evaluation_id)
        if evaluation.status not in {"failed", "cancelled", "interrupted"}:
            raise EvaluationLabError(
                f"only failed, cancelled, or interrupted evaluations can retry; got {evaluation.status}"
            )
        if self.scheduler is not None and evaluation.work_item_id and submit:
            retried = self.scheduler.retry(
                evaluation.work_item_id,
                reason="operator requested evaluation retry",
                force=True,
            )
            if retried is None:
                raise EvaluationLabError(
                    f"evaluation {evaluation_id} has no retryable workstation work item"
                )
            return self.get(evaluation_id)
        request = dict(evaluation.request)
        request["worker"] = self._worker_envelope()
        updated = self.db.update_evaluation(
            evaluation_id,
            status="queued",
            stage="queued",
            processed_samples=0,
            total_samples=None,
            error=None,
            cancel_requested=False,
            retry_count=evaluation.retry_count + 1,
            started_at=None,
            completed_at=None,
            request=request,
        )
        assert updated is not None
        if submit:
            self._submit(evaluation_id)
        return updated

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)


class EvaluationLabService:
    """Suite catalog, persistent jobs, results and direction-aware comparisons."""

    def __init__(
        self,
        db: RunDatabase,
        artifact_root: Path | str | None = None,
        *,
        registry: Optional[EvaluationAdapterRegistry] = None,
        scheduler: Optional[Any] = None,
    ):
        self.db = db
        self.jobs = EvaluationJobManager(
            db,
            artifact_root,
            registry=registry,
            scheduler=scheduler,
        )

    @property
    def registry(self) -> EvaluationAdapterRegistry:
        return self.jobs.registry

    def create_suite(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        items: Optional[Sequence[Mapping[str, Any]]] = None,
        primary_metric: str = "score",
        direction: str = "maximize",
        generation_settings: Optional[Mapping[str, Any]] = None,
        evaluator_versions: Optional[Mapping[str, Any]] = None,
        purpose: str = "unspecified",
        suite_id: Optional[str] = None,
    ) -> tuple[BenchmarkSuiteRecord, Optional[BenchmarkSuiteRevisionRecord]]:
        suite = self.db.create_benchmark_suite(
            name=name, description=description, purpose=purpose, suite_id=suite_id
        )
        revision = None
        if items:
            revision = self.create_revision(
                suite_id=suite.id,
                items=items,
                primary_metric=primary_metric,
                direction=direction,
                generation_settings=generation_settings,
                evaluator_versions=evaluator_versions,
            )
            suite = self.db.get_benchmark_suite(suite.id) or suite
        return suite, revision

    def create_revision(
        self,
        *,
        suite_id: str,
        items: Sequence[Mapping[str, Any]],
        primary_metric: str,
        direction: str,
        generation_settings: Optional[Mapping[str, Any]] = None,
        evaluator_versions: Optional[Mapping[str, Any]] = None,
        revision_id: Optional[str] = None,
    ) -> BenchmarkSuiteRevisionRecord:
        normalized_items: list[dict[str, Any]] = []
        seen_item_ids: set[str] = set()
        for index, raw_item in enumerate(items):
            if not isinstance(raw_item, Mapping):
                raise EvaluationLabError("benchmark suite items must be objects")
            item = dict(raw_item)
            item_id = str(item.get("id") or item.get("suite_item_id") or index)
            if item_id in seen_item_ids:
                raise EvaluationLabError(f"duplicate benchmark suite item id: {item_id}")
            seen_item_ids.add(item_id)
            item["id"] = item_id
            normalized_items.append(item)
        definition = {
            "items": normalized_items,
            "generation_settings": dict(generation_settings or {}),
            "evaluator_versions": dict(evaluator_versions or {}),
            "primary_metric": primary_metric,
            "direction": direction,
        }
        return self.db.create_benchmark_suite_revision(
            suite_id=suite_id,
            revision_id=revision_id,
            content_hash=_hash_json(definition),
            items=normalized_items,
            generation_settings=generation_settings,
            evaluator_versions=evaluator_versions,
            primary_metric=primary_metric,
            direction=direction,
        )

    def get_suite(self, suite_id: str) -> BenchmarkSuiteRecord:
        suite = self.db.get_benchmark_suite(suite_id)
        if suite is None:
            raise EvaluationLabError(f"unknown benchmark suite: {suite_id}")
        return suite

    def list_suites(self, *, include_archived: bool = False) -> list[BenchmarkSuiteRecord]:
        return self.db.list_benchmark_suites(include_archived=include_archived)

    def update_suite(
        self,
        suite_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        archived: Optional[bool] = None,
        purpose: Optional[str] = None,
    ) -> BenchmarkSuiteRecord:
        suite = self.db.update_benchmark_suite(
            suite_id,
            name=name,
            description=description,
            archived=archived,
            purpose=purpose,
        )
        if suite is None:
            raise EvaluationLabError(f"unknown benchmark suite: {suite_id}")
        return suite

    def delete_suite(self, suite_id: str) -> bool:
        return self.db.delete_benchmark_suite(suite_id)

    def get_revision(self, revision_id: str) -> BenchmarkSuiteRevisionRecord:
        revision = self.db.get_benchmark_suite_revision(revision_id)
        if revision is None:
            raise EvaluationLabError(f"unknown benchmark suite revision: {revision_id}")
        return revision

    def list_revisions(self, suite_id: str) -> list[BenchmarkSuiteRevisionRecord]:
        self.get_suite(suite_id)
        return self.db.list_benchmark_suite_revisions(suite_id)

    def launch_evaluation(
        self,
        *,
        suite_revision_id: str,
        adapter_id: Optional[str] = None,
        subject: Mapping[str, Any],
        request: Optional[Mapping[str, Any]] = None,
        evaluation_id: Optional[str] = None,
        dependencies: Sequence[str] = (),
        submit: bool = True,
    ) -> EvaluationLaunch:
        return self.jobs.launch(
            suite_revision_id=suite_revision_id,
            adapter_id=adapter_id,
            subject=subject,
            request=request,
            evaluation_id=evaluation_id,
            dependencies=dependencies,
            submit=submit,
        )

    def failure_mining_policy(
        self,
        *,
        candidate_evaluation_id: str,
        base_evaluation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve the immutable suite-purpose gate for reviewed failure mining.

        Public API and CLI surfaces can call this before previewing or building
        a child dataset.  Keeping the decision in the evaluation service avoids
        duplicating purpose and same-revision checks across operator surfaces.
        """

        candidate = self.db.get_evaluation(candidate_evaluation_id)
        if candidate is None:
            raise EvaluationLabError(f"unknown candidate evaluation: {candidate_evaluation_id}")
        base = None
        if base_evaluation_id:
            base = self.db.get_evaluation(base_evaluation_id)
            if base is None:
                raise EvaluationLabError(f"unknown base evaluation: {base_evaluation_id}")
            if base.suite_revision_id != candidate.suite_revision_id:
                raise EvaluationLabError(
                    "failure mining requires evaluations from the same suite revision"
                )
        revision = self.db.get_benchmark_suite_revision(candidate.suite_revision_id)
        if revision is None:
            raise EvaluationLabError(f"missing suite revision: {candidate.suite_revision_id}")
        suite = self.db.get_benchmark_suite(revision.suite_id)
        if suite is None:
            raise EvaluationLabError(f"missing benchmark suite: {revision.suite_id}")
        allowed = suite.purpose not in {"holdout", "operational"}
        return {
            "allowed": allowed,
            "reason": (
                None
                if allowed
                else f"{suite.purpose} suite evidence cannot enter a training dataset"
            ),
            "suite_id": suite.id,
            "suite_revision_id": revision.id,
            "suite_purpose": suite.purpose,
            "base_evaluation_id": base.id if base is not None else None,
            "candidate_evaluation_id": candidate.id,
        }

    def require_failure_mining_allowed(
        self,
        *,
        candidate_evaluation_id: str,
        base_evaluation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the policy or raise before holdout evidence can be mined."""

        policy = self.failure_mining_policy(
            candidate_evaluation_id=candidate_evaluation_id,
            base_evaluation_id=base_evaluation_id,
        )
        if not policy["allowed"]:
            raise EvaluationLabError(str(policy["reason"]))
        return policy

    def get_evaluation(self, evaluation_id: str) -> EvaluationRecord:
        return self.jobs.get(evaluation_id)

    def list_evaluations(self, **filters: Any) -> list[EvaluationRecord]:
        return self.db.list_evaluations(**filters)

    def cancel_evaluation(self, evaluation_id: str) -> EvaluationRecord:
        return self.jobs.cancel(evaluation_id)

    def retry_evaluation(self, evaluation_id: str) -> EvaluationRecord:
        return self.jobs.retry(evaluation_id)

    def evaluation_detail(self, evaluation_id: str) -> Dict[str, Any]:
        evaluation = self.jobs.get(evaluation_id)
        revision = self.db.get_benchmark_suite_revision(evaluation.suite_revision_id)
        suite = self.db.get_benchmark_suite(revision.suite_id) if revision else None
        metrics = [value.to_dict() for value in self.db.list_evaluation_metrics(evaluation_id)]
        primary = next(
            (
                value
                for value in metrics
                if revision is not None
                and value["name"] == revision.primary_metric
                and not value.get("suite_item_id")
            ),
            metrics[0] if metrics else None,
        )
        progress = None
        if evaluation.total_samples:
            progress = min(100.0, 100.0 * evaluation.processed_samples / evaluation.total_samples)
        return {
            **evaluation.to_dict(),
            "suite_id": revision.suite_id if revision else None,
            "suite_name": suite.name if suite else None,
            "subject": {
                "kind": evaluation.subject_type,
                "value": evaluation.subject_ref,
                "run_id": (
                    evaluation.subject_ref
                    if evaluation.subject_type in {"run", "final_model", "checkpoint"}
                    else None
                ),
                "subject_hash": evaluation.subject_hash,
            },
            "run_id": (
                evaluation.subject_ref
                if evaluation.subject_type in {"run", "final_model", "checkpoint"}
                else None
            ),
            "metrics": metrics,
            "primary_metric": primary,
            "progress_percent": progress,
            "finished_at": evaluation.completed_at,
            "samples": [
                _sample_record_dict(value)
                for value in self.db.list_evaluation_samples(evaluation_id)
            ],
        }

    def compare(
        self, base_evaluation_id: str, candidate_evaluation_id: str, *, tolerance: float = 1e-12
    ) -> Dict[str, Any]:
        base = self.jobs.get(base_evaluation_id)
        candidate = self.jobs.get(candidate_evaluation_id)
        if base.status != "completed" or candidate.status != "completed":
            raise EvaluationLabError("only completed evaluations can be compared")
        if base.suite_revision_id != candidate.suite_revision_id:
            raise EvaluationLabError("evaluations must use the same benchmark suite revision")
        revision = self.db.get_benchmark_suite_revision(base.suite_revision_id)
        if revision is None:
            raise EvaluationLabError(f"missing suite revision: {base.suite_revision_id}")
        suite = self.db.get_benchmark_suite(revision.suite_id)
        suite_purpose = suite.purpose if suite is not None else "unspecified"

        base_metrics = {
            (metric.name, metric.suite_item_id): metric
            for metric in self.db.list_evaluation_metrics(base.id)
        }
        candidate_metrics = {
            (metric.name, metric.suite_item_id): metric
            for metric in self.db.list_evaluation_metrics(candidate.id)
        }
        metric_deltas: list[dict[str, Any]] = []
        for key in sorted(set(base_metrics) & set(candidate_metrics)):
            before = base_metrics[key]
            after = candidate_metrics[key]
            direction = after.direction or before.direction or revision.direction
            delta = after.value - before.value
            signed = delta if direction == "maximize" else -delta
            outcome = "unchanged"
            if signed > tolerance:
                outcome = "improvement"
            elif signed < -tolerance:
                outcome = "regression"
            metric_deltas.append(
                {
                    "name": key[0],
                    "suite_item_id": key[1] or None,
                    "base": before.value,
                    "candidate": after.value,
                    "delta": delta,
                    "direction": direction,
                    "outcome": outcome,
                }
            )

        def index_sample_occurrences(samples: Sequence[Any]) -> Dict[tuple[str, str, int], Any]:
            """Index every physical occurrence without losing its logical record ID."""

            indexed: Dict[tuple[str, str, int], Any] = {}
            occurrences: Dict[tuple[str, str], int] = {}
            for sample in samples:
                logical_record_id = str(sample.record_id or sample.suite_item_id)
                suite_item_id = str(sample.suite_item_id or "")
                group = (logical_record_id, suite_item_id)
                occurrence = occurrences.get(group, 0)
                occurrences[group] = occurrence + 1
                indexed[(logical_record_id, suite_item_id, occurrence)] = sample
            return indexed

        base_samples = index_sample_occurrences(self.db.list_evaluation_samples(base.id))
        candidate_samples = index_sample_occurrences(self.db.list_evaluation_samples(candidate.id))
        counts = {
            "regression": 0,
            "improvement": 0,
            "unchanged_failure": 0,
            "unchanged_pass": 0,
            "missing_base": 0,
            "missing_candidate": 0,
        }
        sample_deltas: list[dict[str, Any]] = []
        evidence_gaps: list[dict[str, Any]] = []
        for key in sorted(set(base_samples) | set(candidate_samples)):
            logical_record_id, suite_item_id, occurrence = key
            occurrence_key = _hash_json(
                {
                    "record_id": logical_record_id,
                    "suite_item_id": suite_item_id,
                    "occurrence": occurrence,
                }
            )[:24]
            before = base_samples.get(key)
            after = candidate_samples.get(key)
            if before is None:
                counts["missing_base"] += 1
                evidence_gaps.append(
                    {
                        "record_id": logical_record_id,
                        "suite_item_id": suite_item_id,
                        "occurrence_index": occurrence,
                        "occurrence_key": occurrence_key,
                        "outcome": "missing_base",
                        "classification": "missing_base",
                        "mineable": False,
                        "reason": "base_evidence_missing",
                        "candidate": _sample_record_dict(after),
                    }
                )
                continue
            if after is None:
                counts["missing_candidate"] += 1
                evidence_gaps.append(
                    {
                        "record_id": logical_record_id,
                        "suite_item_id": suite_item_id,
                        "occurrence_index": occurrence,
                        "occurrence_key": occurrence_key,
                        "outcome": "missing_candidate",
                        "classification": "missing_candidate",
                        "mineable": False,
                        "reason": "candidate_evidence_missing",
                        "base": _sample_record_dict(before),
                    }
                )
                continue
            before_kind = str(_record_evidence_value(before, "evidence_kind", "legacy") or "legacy")
            after_kind = str(_record_evidence_value(after, "evidence_kind", "legacy") or "legacy")
            before_valid = bool(_record_evidence_value(before, "valid", False))
            after_valid = bool(_record_evidence_value(after, "valid", False))
            before_mineable = bool(_record_evidence_value(before, "mineable", False))
            after_mineable = bool(_record_evidence_value(after, "mineable", False))
            if not (before_valid and after_valid and before_mineable and after_mineable):
                reasons: list[str] = []
                if not before_valid:
                    reasons.append("base_invalid")
                if not after_valid:
                    reasons.append("candidate_invalid")
                if before_valid and not before_mineable:
                    reasons.append("base_non_mineable")
                if after_valid and not after_mineable:
                    reasons.append("candidate_non_mineable")
                evidence_gaps.append(
                    {
                        "record_id": logical_record_id,
                        "suite_item_id": suite_item_id,
                        "occurrence_index": occurrence,
                        "occurrence_key": occurrence_key,
                        "outcome": "incomplete_evidence",
                        "classification": "incomplete_evidence",
                        "mineable": False,
                        "reason": ",".join(reasons) or "non_mineable_evidence",
                        "base_evidence_kind": before_kind,
                        "candidate_evidence_kind": after_kind,
                        "base": _sample_record_dict(before),
                        "candidate": _sample_record_dict(after),
                    }
                )
                continue
            if before.passed is True and after.passed is False:
                outcome = "regression"
            elif before.passed is False and after.passed is True:
                outcome = "improvement"
            elif before.score is not None and after.score is not None:
                delta = after.score - before.score
                sample_direction = str(
                    _record_evidence_value(
                        after,
                        "score_direction",
                        _record_evidence_value(before, "score_direction", revision.direction),
                    )
                    or revision.direction
                )
                signed = delta if sample_direction == "maximize" else -delta
                if signed > tolerance:
                    outcome = "improvement"
                elif signed < -tolerance:
                    outcome = "regression"
                else:
                    outcome = (
                        "unchanged_failure"
                        if before.passed is False
                        or after.passed is False
                        or before.error
                        or after.error
                        else "unchanged_pass"
                    )
            elif before.passed is False and after.passed is False:
                outcome = "unchanged_failure"
            elif before.passed is True and after.passed is True:
                outcome = "unchanged_pass"
            else:
                outcome = "unchanged_failure" if before.error or after.error else "unchanged_pass"
            counts[outcome] += 1
            sample_deltas.append(
                {
                    "record_id": logical_record_id,
                    "suite_item_id": suite_item_id,
                    "occurrence_index": occurrence,
                    "occurrence_key": occurrence_key,
                    "outcome": outcome,
                    "classification": outcome,
                    "mineable": True,
                    "evidence_complete": True,
                    "base_score": before.score,
                    "candidate_score": after.score,
                    "base_passed": before.passed,
                    "candidate_passed": after.passed,
                    "score_direction": str(
                        _record_evidence_value(
                            after,
                            "score_direction",
                            _record_evidence_value(before, "score_direction", revision.direction),
                        )
                        or revision.direction
                    ),
                    "base": _sample_record_dict(before),
                    "candidate": _sample_record_dict(after),
                }
            )
        primary_delta = next(
            (
                value
                for value in metric_deltas
                if value["name"] == revision.primary_metric and not value["suite_item_id"]
            ),
            None,
        )
        return {
            "base_evaluation_id": base.id,
            "candidate_evaluation_id": candidate.id,
            "base_id": base.id,
            "candidate_id": candidate.id,
            "suite_revision_id": revision.id,
            "suite_purpose": suite_purpose,
            "failure_mining_allowed": suite_purpose not in {"holdout", "operational"},
            "primary_metric": revision.primary_metric,
            "direction": revision.direction,
            "counts": counts,
            "metric_deltas": metric_deltas,
            "metrics": metric_deltas,
            "sample_deltas": sample_deltas,
            "samples": sample_deltas,
            "evidence_gaps": evidence_gaps,
            "evidence_summary": {
                "base_total": len(base_samples),
                "candidate_total": len(candidate_samples),
                "comparable": len(sample_deltas),
                "incomplete": len(evidence_gaps),
                "complete": not evidence_gaps,
                "failure_mining_eligible": len(sample_deltas),
            },
            "base_value": primary_delta["base"] if primary_delta else None,
            "candidate_value": primary_delta["candidate"] if primary_delta else None,
            "delta": primary_delta["delta"] if primary_delta else None,
        }

    def compare_page(
        self,
        base_evaluation_id: str,
        candidate_evaluation_id: str,
        *,
        offset: int = 0,
        limit: int = 100,
        tolerance: float = 1e-12,
    ) -> Dict[str, Any]:
        """Return a comparison page without materializing either evaluation.

        Samples are joined in SQLite by logical record identity and occurrence.
        We stream the join in bounded chunks to retain exact aggregate counts,
        while retaining only the requested page of comparable rows and gaps.
        """

        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 1000))
        base = self.jobs.get(base_evaluation_id)
        candidate = self.jobs.get(candidate_evaluation_id)
        if base.status != "completed" or candidate.status != "completed":
            raise EvaluationLabError("only completed evaluations can be compared")
        if base.suite_revision_id != candidate.suite_revision_id:
            raise EvaluationLabError("evaluations must use the same benchmark suite revision")
        revision = self.db.get_benchmark_suite_revision(base.suite_revision_id)
        if revision is None:
            raise EvaluationLabError(f"missing suite revision: {base.suite_revision_id}")
        suite = self.db.get_benchmark_suite(revision.suite_id)
        suite_purpose = suite.purpose if suite is not None else "unspecified"

        base_metrics = {
            (metric.name, metric.suite_item_id): metric
            for metric in self.db.list_evaluation_metrics(base.id)
        }
        candidate_metrics = {
            (metric.name, metric.suite_item_id): metric
            for metric in self.db.list_evaluation_metrics(candidate.id)
        }
        metric_deltas: list[dict[str, Any]] = []
        for key in sorted(set(base_metrics) & set(candidate_metrics)):
            before_metric = base_metrics[key]
            after_metric = candidate_metrics[key]
            direction = after_metric.direction or before_metric.direction or revision.direction
            delta = after_metric.value - before_metric.value
            signed = delta if direction == "maximize" else -delta
            outcome = "unchanged"
            if signed > tolerance:
                outcome = "improvement"
            elif signed < -tolerance:
                outcome = "regression"
            metric_deltas.append(
                {
                    "name": key[0],
                    "suite_item_id": key[1] or None,
                    "base": before_metric.value,
                    "candidate": after_metric.value,
                    "delta": delta,
                    "direction": direction,
                    "outcome": outcome,
                }
            )

        counts = {
            "regression": 0,
            "improvement": 0,
            "unchanged_failure": 0,
            "unchanged_pass": 0,
            "missing_base": 0,
            "missing_candidate": 0,
        }
        delta_page: list[dict[str, Any]] = []
        gap_page: list[dict[str, Any]] = []
        comparable_total = 0
        gap_total = 0
        pair_total = self.db.count_evaluation_sample_pairs(base.id, candidate.id)

        def retain(page: list[dict[str, Any]], value: dict[str, Any], index: int) -> None:
            if offset <= index < offset + limit:
                page.append(value)

        scan_offset = 0
        while scan_offset < pair_total:
            pairs = self.db.list_evaluation_sample_pairs(
                base.id, candidate.id, limit=1000, offset=scan_offset
            )
            if not pairs:
                break
            for pair in pairs:
                before = pair.get("base")
                after = pair.get("candidate")
                logical_record_id = str(pair.get("logical_record_id") or "")
                occurrence = int(pair.get("occurrence") or 0)
                sample = after or before
                suite_item_id = str(sample.suite_item_id or "") if sample is not None else ""
                occurrence_key = _hash_json(
                    {
                        "record_id": logical_record_id,
                        "suite_item_id": suite_item_id,
                        "occurrence": occurrence,
                    }
                )[:24]
                if before is None or after is None:
                    outcome = "missing_base" if before is None else "missing_candidate"
                    counts[outcome] += 1
                    gap = {
                        "record_id": logical_record_id,
                        "suite_item_id": suite_item_id,
                        "occurrence_index": occurrence,
                        "occurrence_key": occurrence_key,
                        "outcome": outcome,
                        "classification": outcome,
                        "mineable": False,
                        "reason": "base_evidence_missing" if before is None else "candidate_evidence_missing",
                        "candidate" if before is None else "base": _sample_record_dict(after or before),
                    }
                    retain(gap_page, gap, gap_total)
                    gap_total += 1
                    continue

                before_kind = str(
                    _record_evidence_value(before, "evidence_kind", "legacy") or "legacy"
                )
                after_kind = str(
                    _record_evidence_value(after, "evidence_kind", "legacy") or "legacy"
                )
                before_valid = bool(_record_evidence_value(before, "valid", False))
                after_valid = bool(_record_evidence_value(after, "valid", False))
                before_mineable = bool(_record_evidence_value(before, "mineable", False))
                after_mineable = bool(_record_evidence_value(after, "mineable", False))
                if not (before_valid and after_valid and before_mineable and after_mineable):
                    reasons: list[str] = []
                    if not before_valid:
                        reasons.append("base_invalid")
                    if not after_valid:
                        reasons.append("candidate_invalid")
                    if before_valid and not before_mineable:
                        reasons.append("base_non_mineable")
                    if after_valid and not after_mineable:
                        reasons.append("candidate_non_mineable")
                    gap = {
                        "record_id": logical_record_id,
                        "suite_item_id": suite_item_id,
                        "occurrence_index": occurrence,
                        "occurrence_key": occurrence_key,
                        "outcome": "incomplete_evidence",
                        "classification": "incomplete_evidence",
                        "mineable": False,
                        "reason": ",".join(reasons) or "non_mineable_evidence",
                        "base_evidence_kind": before_kind,
                        "candidate_evidence_kind": after_kind,
                        "base": _sample_record_dict(before),
                        "candidate": _sample_record_dict(after),
                    }
                    retain(gap_page, gap, gap_total)
                    gap_total += 1
                    continue

                if before.passed is True and after.passed is False:
                    outcome = "regression"
                elif before.passed is False and after.passed is True:
                    outcome = "improvement"
                elif before.score is not None and after.score is not None:
                    score_delta = after.score - before.score
                    sample_direction = str(
                        _record_evidence_value(
                            after,
                            "score_direction",
                            _record_evidence_value(before, "score_direction", revision.direction),
                        )
                        or revision.direction
                    )
                    signed = score_delta if sample_direction == "maximize" else -score_delta
                    if signed > tolerance:
                        outcome = "improvement"
                    elif signed < -tolerance:
                        outcome = "regression"
                    else:
                        outcome = (
                            "unchanged_failure"
                            if before.passed is False
                            or after.passed is False
                            or before.error
                            or after.error
                            else "unchanged_pass"
                        )
                elif before.passed is False and after.passed is False:
                    outcome = "unchanged_failure"
                elif before.passed is True and after.passed is True:
                    outcome = "unchanged_pass"
                else:
                    outcome = (
                        "unchanged_failure" if before.error or after.error else "unchanged_pass"
                    )
                counts[outcome] += 1
                delta = {
                    "record_id": logical_record_id,
                    "suite_item_id": suite_item_id,
                    "occurrence_index": occurrence,
                    "occurrence_key": occurrence_key,
                    "outcome": outcome,
                    "classification": outcome,
                    "mineable": True,
                    "evidence_complete": True,
                    "base_score": before.score,
                    "candidate_score": after.score,
                    "base_passed": before.passed,
                    "candidate_passed": after.passed,
                    "score_direction": str(
                        _record_evidence_value(
                            after,
                            "score_direction",
                            _record_evidence_value(before, "score_direction", revision.direction),
                        )
                        or revision.direction
                    ),
                    "base": _sample_record_dict(before),
                    "candidate": _sample_record_dict(after),
                }
                retain(delta_page, delta, comparable_total)
                comparable_total += 1
            scan_offset += len(pairs)

        primary_delta = next(
            (
                value
                for value in metric_deltas
                if value["name"] == revision.primary_metric and not value["suite_item_id"]
            ),
            None,
        )
        base_total = self.db.count_evaluation_samples(base.id)
        candidate_total = self.db.count_evaluation_samples(candidate.id)
        return {
            "base_evaluation_id": base.id,
            "candidate_evaluation_id": candidate.id,
            "base_id": base.id,
            "candidate_id": candidate.id,
            "suite_revision_id": revision.id,
            "suite_purpose": suite_purpose,
            "failure_mining_allowed": suite_purpose not in {"holdout", "operational"},
            "primary_metric": revision.primary_metric,
            "direction": revision.direction,
            "counts": counts,
            "metric_deltas": metric_deltas,
            "metrics": metric_deltas,
            "sample_deltas": {
                "items": delta_page,
                "total": comparable_total,
                "offset": offset,
                "limit": limit,
            },
            "evidence_gaps": {
                "items": gap_page,
                "total": gap_total,
                "offset": offset,
                "limit": limit,
            },
            "evidence_summary": {
                "base_total": base_total,
                "candidate_total": candidate_total,
                "comparable": comparable_total,
                "incomplete": gap_total,
                "complete": gap_total == 0,
                "failure_mining_eligible": comparable_total,
            },
            "base_value": primary_delta["base"] if primary_delta else None,
            "candidate_value": primary_delta["candidate"] if primary_delta else None,
            "delta": primary_delta["delta"] if primary_delta else None,
        }

    compare_evaluations = compare

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        self.jobs.shutdown(wait=wait, cancel_futures=cancel_futures)


EvaluationLab = EvaluationLabService


__all__ = [
    "EvaluationJobManager",
    "EvaluationLab",
    "EvaluationLabService",
    "TERMINAL_EVALUATION_STATES",
    "resolve_subject",
]
