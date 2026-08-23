"""Facade shared by Dataset Lab catalog, API, CLI, and dashboard surfaces."""

from __future__ import annotations

import copy
import csv
import json
import os
import re
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .errors import SourceError, VersionError
from .comparison import DatasetVersionComparator, DatasetVersionComparison
from .failure_mining import FailureMiningBuilder
from .integrations import (
    configured_judge,
    configured_semantic_similarity,
    configured_teacher,
    configured_verifier,
)
from .jobs import DatasetJob, JobContext, SerialJobManager
from .identity import seed_record_identities
from .models import infer_schema
from .profiling import profile_records
from .recipe import Recipe, RecipeContext, RecipeResult, RecipeRunner, RecipeStep, StepProvenance
from .sources import AssetFingerprint, SourceSnapshot, SourceSpec, hash_file, load_source
from .storage import DatasetVersion, VersionStore
from .training_artifacts import (
    DatasetBinding,
    TrainingArtifactRenderer,
    TrainingDatasetArtifact,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-.")
    return slug or uuid.uuid4().hex[:12]


def inherit_dataset_version_exposures(
    database: Any,
    *,
    parent_version_ids: Sequence[str],
    child_version_id: str,
    provenance: Optional[Mapping[str, Any]] = None,
) -> List[Any]:
    """Copy append-only exposure lineage from parents to a cataloged child.

    This helper is intentionally independent of the public API.  An operator
    surface calls it only after the newly published filesystem version has been
    mirrored into SQLite, which keeps Dataset Lab publication atomic while also
    satisfying the exposure ledger's dataset-version foreign key.
    """

    if database.get_dataset_version(child_version_id) is None:
        raise VersionError(
            f"Dataset version {child_version_id} must be cataloged before exposure sync"
        )
    existing = database.list_exposures(dataset_version_id=child_version_id)
    by_source = {value.inherited_from_id: value for value in existing if value.inherited_from_id}
    synchronized: List[Any] = []
    for parent_version_id in dict.fromkeys(
        str(value).strip()
        for value in parent_version_ids
        if value is not None and str(value).strip()
    ):
        if database.get_dataset_version(parent_version_id) is None:
            # Old filesystem-only parents remain readable.  Once mirrored, a
            # later idempotent sync can fill in their exposure ancestry.
            continue
        for source in database.list_exposures(dataset_version_id=parent_version_id):
            prior = by_source.get(source.id)
            if prior is not None:
                synchronized.append(prior)
                continue
            inherited = database.inherit_exposures(
                [source],
                dataset_version_id=child_version_id,
                provenance={
                    "operation": "dataset_version_derivation",
                    "parent_version_id": parent_version_id,
                    "child_version_id": child_version_id,
                    "source_exposure_id": source.id,
                    "source_exposure_type": source.exposure_type,
                    "source_provenance": source.provenance,
                    **dict(provenance or {}),
                },
            )[0]
            by_source[source.id] = inherited
            synchronized.append(inherited)
    return synchronized


def record_failure_mining_exposures(
    database: Any,
    *,
    child_version_id: str,
    details: Mapping[str, Any],
) -> List[Any]:
    """Record reviewed evaluation items that entered a training-data child.

    Parent exposure ancestry is handled separately by
    :func:`inherit_dataset_version_exposures`.  These rows describe the new,
    direct feedback-loop exposure introduced by failure mining.  The operation
    is idempotent so job reconciliation and application restarts are safe.
    """

    if database.get_dataset_version(child_version_id) is None:
        raise VersionError(
            f"Dataset version {child_version_id} must be cataloged before exposure sync"
        )
    suite_revision_id = str(details.get("suite_revision_id") or "").strip()
    if not suite_revision_id:
        return []
    revision = database.get_benchmark_suite_revision(suite_revision_id)
    if revision is None:
        # Legacy comparison imports may reference a suite that was never
        # cataloged locally.  Preserve the child version without inventing a
        # ledger foreign-key target.
        return []
    suite = database.get_benchmark_suite(revision.suite_id)
    purpose = suite.purpose if suite is not None else "unspecified"
    selected = details.get("selected_records") or []
    if not selected:
        # v2 manifests predate selected-record envelopes.  Their record IDs were
        # also the best available suite identities, so retain that legacy read.
        selected = [
            {"record_id": value, "suite_item_id": value, "selection_id": None}
            for value in details.get("original_record_ids") or []
        ]
    existing = database.list_exposures(dataset_version_id=child_version_id)
    existing_keys = {
        (
            value.suite_revision_id,
            value.suite_item_id,
            value.exposure_type,
            value.provenance.get("selection_id"),
        ): value
        for value in existing
    }
    synchronized: List[Any] = []
    for raw in selected:
        if not isinstance(raw, Mapping):
            continue
        record_id = str(raw.get("record_id") or "").strip()
        suite_item_id = str(raw.get("suite_item_id") or record_id).strip()
        if not suite_item_id:
            continue
        selection_id = raw.get("selection_id")
        key = (suite_revision_id, suite_item_id, "failure_mining", selection_id)
        prior = existing_keys.get(key)
        if prior is not None:
            synchronized.append(prior)
            continue
        exposure = database.record_exposure(
            suite_revision_id=suite_revision_id,
            suite_item_id=suite_item_id,
            exposure_type="failure_mining",
            dataset_version_id=child_version_id,
            provenance={
                "source": "reviewed_failure_mining",
                "suite_purpose": purpose,
                "child_version_id": child_version_id,
                "parent_version_id": details.get("parent_version_id"),
                "base_evaluation_id": details.get("base_evaluation_id"),
                "candidate_evaluation_id": details.get("candidate_evaluation_id"),
                "evaluation_ids": list(details.get("evaluation_ids") or []),
                "selection_id": selection_id,
                "record_id": record_id or None,
                "outcome": raw.get("outcome"),
                "target_split": details.get("target_split"),
                "selector": copy.deepcopy(details.get("selector") or {}),
                "exclusions_hash": details.get("exclusions_hash"),
            },
        )
        existing_keys[key] = exposure
        synchronized.append(exposure)
    return synchronized


def record_review_label_exposures(
    database: Any,
    *,
    child_version_id: str,
    details: Mapping[str, Any],
) -> List[Any]:
    """Attach development-evidence identity used by reviewed training data."""

    if database.get_dataset_version(child_version_id) is None:
        raise VersionError(
            f"Dataset version {child_version_id} must be cataloged before exposure sync"
        )
    existing = database.list_exposures(dataset_version_id=child_version_id)
    known = {
        (
            value.suite_revision_id,
            value.suite_item_id,
            value.exposure_type,
            value.provenance.get("label_set_revision_id"),
        ): value
        for value in existing
    }
    synchronized: List[Any] = []
    for raw in details.get("exposure_records") or []:
        if not isinstance(raw, Mapping):
            continue
        suite_revision_id = str(raw.get("suite_revision_id") or "").strip()
        suite_item_id = str(raw.get("suite_item_id") or raw.get("record_id") or "").strip()
        if not suite_revision_id or not suite_item_id:
            continue
        if database.get_benchmark_suite_revision(suite_revision_id) is None:
            # Imported manifests can retain external evidence identity without
            # inventing a local foreign-key target.
            continue
        revision_id = str(details.get("label_set_revision_id") or "").strip()
        key = (suite_revision_id, suite_item_id, "human_review_label", revision_id)
        prior = known.get(key)
        if prior is not None:
            synchronized.append(prior)
            continue
        exposure = database.record_exposure(
            suite_revision_id=suite_revision_id,
            suite_item_id=suite_item_id,
            exposure_type="human_review_label",
            dataset_version_id=child_version_id,
            provenance={
                "source": "review_label_set",
                "label_set_revision_id": revision_id or None,
                "label_set_id": details.get("label_set_id"),
                "label_set_content_hash": details.get("label_set_content_hash"),
                "review_item_id": raw.get("review_item_id"),
                "record_id": raw.get("record_id"),
                "suite_purpose": raw.get("purpose"),
                "source_kind": raw.get("source_kind"),
                "source_ref": raw.get("source_ref"),
                "child_version_id": child_version_id,
                "parent_version_id": details.get("parent_version_id"),
            },
        )
        known[key] = exposure
        synchronized.append(exposure)
    return synchronized


@dataclass
class DatasetSource:
    id: str
    dataset_id: str
    name: str
    spec: SourceSpec
    fingerprint: str
    row_count: int
    size_bytes: int
    file_count: int
    canonical_kind: Optional[str] = None
    modality: str = "text"
    field_mapping: Dict[str, str] = field(default_factory=dict)
    asset_fingerprints: List[Dict[str, Any]] = field(default_factory=list)
    parent_source_id: Optional[str] = None
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["spec"] = self.spec.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetSource":
        raw = dict(value)
        raw["spec"] = SourceSpec.from_value(raw["spec"])
        return cls(**raw)


@dataclass
class PreviewPage:
    id: str
    kind: str
    offset: int
    limit: int
    total: int
    records: List[Dict[str, Any]]
    split: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetLab:
    """Local Dataset Lab service with a small JSON source catalog.

    ``root`` is the managed dataset root (normally ``~/.halo-forge/datasets``).
    Operational state uses hidden files ``.catalog.json`` and ``.jobs.json`` there.
    The higher-level SQLite catalog can mirror these records;
    it does not need to own transformation or filesystem semantics.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        teacher: Optional[Callable[..., str]] = None,
        verifier: Optional[Callable[..., Any]] = None,
        judge: Optional[Callable[..., float]] = None,
        semantic_similarity: Optional[Callable[..., float]] = None,
        mixture_resolver: Optional[Callable[[str], Any]] = None,
        failure_resolver: Optional[Callable[[str], Any]] = None,
        scheduler: Optional[Any] = None,
        database: Any = None,
        verifier_service: Any = None,
        repair_overlay_resolver: Optional[Callable[[str], Mapping[str, Any]]] = None,
    ):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / ".catalog.json"
        self._catalog_lock = threading.RLock()
        self._sources: Dict[str, DatasetSource] = self._load_catalog()
        self.store = VersionStore(self.root)
        self.training_artifacts = TrainingArtifactRenderer(self.store)
        # Keep dependency-heavy integrations lazy: these defaults only import
        # their optional clients when the corresponding recipe step executes.
        self._teacher_explicit = teacher is not None
        self.teacher = teacher or configured_teacher
        self._verifier_explicit = verifier is not None
        self.verifier = verifier or configured_verifier
        self.judge = judge or configured_judge
        self.semantic_similarity = semantic_similarity or configured_semantic_similarity
        self.mixture_resolver = mixture_resolver or self._resolve_mixture
        self.failure_resolver = failure_resolver or self._resolve_failures
        self.scheduler = scheduler
        self.database = database or getattr(verifier_service, "db", None)
        self.verifier_service = verifier_service
        self.repair_overlay_resolver = repair_overlay_resolver
        self.job_manager = SerialJobManager(self.root / ".jobs.json")
        self.job_manager.register("build", self._build_job)
        self.job_manager.register("profile", self._profile_job)
        self.job_manager.register("materialize", self._materialize_job)
        self.job_manager.register("failure_mining", self._failure_mining_job)
        self.job_manager.register("review_build", self._review_build_job)
        self.job_manager.register("training_artifact", self._training_artifact_job)

    def _start_background_job(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        submit: Optional[bool] = None,
    ) -> DatasetJob:
        """Use the durable workstation queue when this service is configured for it.

        A plain ``DatasetLab(root)`` keeps the long-standing in-process serial
        executor. Dashboard/desktop runtimes pass a ``WorkstationScheduler``;
        their normal launches then create a durable domain-linked work item and
        leave execution to the supervised worker. ``submit=True`` remains an
        explicit escape hatch for legacy callers, while ``submit=False`` makes
        a persisted job without starting either executor.
        """

        use_scheduler = self.scheduler is not None and submit is None
        if not use_scheduler:
            return self.job_manager.start(
                kind,
                payload,
                submit=True if submit is None else bool(submit),
            )

        work_item_id = f"dataset-{uuid.uuid4().hex}"
        job = self.job_manager.start(
            kind,
            payload,
            submit=False,
            work_item_id=work_item_id,
        )
        try:
            self.scheduler.enqueue(
                kind=f"dataset_{kind}",
                launch_spec={
                    "handler": "dataset_lab.run_queued",
                    "dataset_root": str(self.root),
                    "dataset_job_id": job.id,
                    "database_path": (
                        getattr(self.database, "path", None) if self.database is not None else None
                    ),
                },
                resource_class="accelerator",
                resource_requirements={"output_path": str(self.root)},
                domain_kind="dataset_job",
                domain_id=job.id,
                max_retries=2,
                work_item_id=work_item_id,
            )
        except Exception as exc:
            self.job_manager._fail(
                job.id,
                f"Could not enqueue durable workstation work: {exc}",
            )
            raise
        return self.job_manager.get(job.id)

    def _verifier_engine(self) -> Any:
        if self.verifier_service is None and self.database is not None:
            from halo_forge.verifier_lab import VerifierLabService

            self.verifier_service = VerifierLabService(
                self.database,
                scheduler=self.scheduler,
            )
        return self.verifier_service

    def _resolve_mixture(self, identifier: str) -> List[Dict[str, Any]]:
        """Resolve a registered source or immutable version for a mix step."""
        if identifier in self._sources:
            source, snapshot = self._snapshot(identifier)
            return seed_record_identities(
                snapshot.records,
                source_fingerprint=snapshot.fingerprint,
                source_name=source.id,
            )
        try:
            return self.store.load_records_with_lineage(identifier)
        except VersionError as exc:
            raise VersionError(f"Unknown mixture source or version: {identifier}") from exc

    def _resolve_repair_overlay(self, revision_id: str) -> Mapping[str, Any]:
        """Resolve and verify one immutable V17 overlay without trusting recipe paths."""

        if self.repair_overlay_resolver is not None:
            return self.repair_overlay_resolver(revision_id)
        if self.database is None:
            raise VersionError("Repair overlays require Dataset Lab's SQLite catalog")
        row = self.database._conn.execute(
            "SELECT * FROM dataset_repair_revisions WHERE id=?", (revision_id,)
        ).fetchone()
        if row is None:
            raise VersionError(f"Unknown dataset repair revision: {revision_id}")
        root = Path(str(row["storage_path"])).expanduser().resolve()
        manifest_path = root / "manifest.json"
        overlay_path = root / "overlay.jsonl"
        if not manifest_path.is_file() or not overlay_path.is_file():
            raise VersionError(f"Repair overlay {revision_id} is incomplete")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_overlay_hash = hash_file(overlay_path)
        expected_overlay_hash = str(
            manifest.get("overlay_sha256") or row["repaired_record_set_hash"] or ""
        )
        if not expected_overlay_hash or actual_overlay_hash != expected_overlay_hash:
            raise VersionError(f"Repair overlay {revision_id} failed checksum verification")
        entries: List[Dict[str, Any]] = []
        with overlay_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise VersionError(
                        f"Repair overlay {revision_id} line {line_number} is not an object"
                    )
                entries.append(value)
        return {
            "revision_id": str(row["id"]),
            "source_fingerprint": str(row["source_fingerprint"]),
            "content_hash": str(row["content_hash"]),
            "repaired_record_set_hash": str(row["repaired_record_set_hash"]),
            "entries": entries,
        }

    @staticmethod
    def _resolve_failures(identifier: str) -> List[Dict[str, Any]]:
        """Load interaction samples from a path or an indexed Halo Forge run."""
        from .sources import load_local, load_local_file

        candidate = Path(identifier).expanduser()
        if candidate.exists():
            return load_local(candidate)[0]
        from halo_forge.run_db import get_database

        db = get_database()
        run = db.get_run(identifier) or db.get_run_by_fs_id(identifier)
        if run is None or not run.output_dir:
            raise SourceError(f"Unknown failure-mining run or interaction path: {identifier}")
        output_dir = Path(run.output_dir).expanduser()
        paths = sorted(output_dir.glob("cycle_*_samples.jsonl"))
        paths.extend(sorted(output_dir.glob("cycle_*/accepted.jsonl")))
        paths.extend(sorted(output_dir.glob("cycle_*/completions.jsonl")))
        if not paths:
            paths.extend(sorted(output_dir.glob("*_samples.jsonl")))
        records: List[Dict[str, Any]] = []
        for path in dict.fromkeys(paths):
            records.extend(load_local_file(path))
        if not records:
            raise SourceError(f"Run {identifier!r} has no interaction sample artifacts")
        return records

    def _load_catalog(self) -> Dict[str, DatasetSource]:
        if not self.catalog_path.is_file():
            return {}
        payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        return {item["id"]: DatasetSource.from_dict(item) for item in payload.get("sources", [])}

    def _save_catalog(self) -> None:
        payload = {
            "format_version": 1,
            "sources": [source.to_dict() for source in self._sources.values()],
        }
        fd, name = tempfile.mkstemp(prefix=".catalog.", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, default=str)
                handle.write("\n")
            os.replace(name, self.catalog_path)
        except Exception:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass
            raise

    def add_source(
        self,
        spec: SourceSpec | Mapping[str, Any],
        *,
        dataset_id: Optional[str] = None,
        name: Optional[str] = None,
        source_id: Optional[str] = None,
        parent_source_id: Optional[str] = None,
    ) -> DatasetSource:
        metadata: Dict[str, Any] = {}
        if isinstance(spec, Mapping):
            raw = dict(spec)
            metadata = {
                "canonical_kind": raw.pop("canonical_kind", None),
                "modality": raw.pop("modality", "text"),
                "field_mapping": raw.pop("field_mapping", raw.pop("mapping", {})) or {},
            }
            source_id = source_id or raw.pop("source_id", raw.pop("id", None))
            kind = str(raw.get("kind", "")).lower()
            # CLI/API source dictionaries use dataset_id for the HF repository ID.
            if kind in {"hf", "huggingface"} and not raw.get("repo_id"):
                raw["repo_id"] = raw.pop("dataset_id", None)
            else:
                raw.pop("dataset_id", None)
            resolved_spec = SourceSpec.from_value(raw)
        else:
            resolved_spec = spec
            metadata = {"canonical_kind": None, "modality": "text", "field_mapping": {}}
        snapshot = load_source(resolved_spec)
        default_name = (
            Path(resolved_spec.path).stem if resolved_spec.path else str(resolved_spec.repo_id)
        )
        dataset_identifier = _slug(dataset_id or default_name)
        identifier = _slug(source_id or f"{dataset_identifier}-{snapshot.fingerprint[:12]}")
        source = DatasetSource(
            id=identifier,
            dataset_id=dataset_identifier,
            name=name or default_name,
            spec=resolved_spec,
            fingerprint=snapshot.fingerprint,
            row_count=len(snapshot.records),
            size_bytes=snapshot.size_bytes,
            file_count=snapshot.file_count,
            canonical_kind=metadata["canonical_kind"],
            modality=str(metadata["modality"]),
            field_mapping=dict(metadata["field_mapping"]),
            asset_fingerprints=[asset.to_dict() for asset in snapshot.assets],
            parent_source_id=parent_source_id,
        )
        with self._catalog_lock:
            existing = self._sources.get(identifier)
            if existing:
                if existing.fingerprint != source.fingerprint or existing.spec != source.spec:
                    raise SourceError(
                        f"Source ID {identifier!r} already identifies different content"
                    )
                return copy.deepcopy(existing)
            self._sources[identifier] = source
            self._save_catalog()
        return copy.deepcopy(source)

    register_source = add_source

    def get_source(self, source_id: str) -> DatasetSource:
        # A supervised or headless durable worker may publish a source through
        # another DatasetLab instance. Refresh the small catalog on a cache
        # miss so the browser process can immediately enqueue the dependent
        # build without restarting.
        if source_id not in self._sources and self.catalog_path.is_file():
            with self._catalog_lock:
                self._sources = self._load_catalog()
        try:
            return copy.deepcopy(self._sources[source_id])
        except KeyError as exc:
            raise SourceError(f"Unknown Dataset Lab source: {source_id}") from exc

    def list_sources(self, *, dataset_id: Optional[str] = None) -> List[DatasetSource]:
        values = [
            source
            for source in self._sources.values()
            if dataset_id is None or source.dataset_id == dataset_id
        ]
        return [
            copy.deepcopy(source)
            for source in sorted(values, key=lambda source: source.created_at, reverse=True)
        ]

    def refresh_source(
        self, source_id: str, *, new_source_id: Optional[str] = None
    ) -> DatasetSource:
        old = self.get_source(source_id)
        payload = old.spec.to_dict()
        payload.update(
            canonical_kind=old.canonical_kind,
            modality=old.modality,
            field_mapping=old.field_mapping,
        )
        return self.add_source(
            payload,
            dataset_id=old.dataset_id,
            name=old.name,
            source_id=new_source_id,
            parent_source_id=old.id,
        )

    def _snapshot(self, source_id: str) -> tuple[DatasetSource, SourceSnapshot]:
        source = self.get_source(source_id)
        snapshot = load_source(source.spec)
        if snapshot.fingerprint != source.fingerprint:
            raise SourceError(
                f"Source {source_id!r} changed after registration; call refresh_source to create a new source revision"
            )
        return source, snapshot

    def get_preview(
        self,
        identifier: str,
        *,
        split: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
        dataset_id: Optional[str] = None,
    ) -> PreviewPage:
        if offset < 0 or limit < 0 or limit > 1000:
            raise ValueError("offset must be non-negative and limit must be between 0 and 1000")
        if identifier in self._sources:
            _, snapshot = self._snapshot(identifier)
            rows = snapshot.records
            kind = "source"
            total = len(rows)
            records = copy.deepcopy(rows[offset : offset + limit])
        else:
            rows, total = self.store.preview_records(
                identifier,
                dataset_id=dataset_id,
                split=split,
                offset=offset,
                limit=limit,
            )
            kind = "version"
            records = copy.deepcopy(rows)
        return PreviewPage(identifier, kind, offset, limit, total, records, split)

    def profile(self, source_id: str) -> Dict[str, Any]:
        source, snapshot = self._snapshot(source_id)
        base_dir = Path(source.spec.path).resolve() if source.spec.path else None
        if base_dir and base_dir.is_file():
            base_dir = base_dir.parent
        return profile_records(snapshot.records, base_dir=base_dir)

    def _resolved_recipe(
        self, source: DatasetSource, recipe: Recipe | Mapping[str, Any] | Path | str
    ) -> Recipe:
        resolved = Recipe.from_value(recipe)
        if source.canonical_kind and not any(step.kind == "map" for step in resolved.steps):
            map_params: Dict[str, Any] = {
                "schema": source.canonical_kind,
                "preserve_unmapped_metadata": True,
            }
            if source.field_mapping:
                map_params["fields"] = source.field_mapping
            map_step = RecipeStep(
                "map",
                map_params,
            )
            leading_repairs = tuple(
                step for step in resolved.steps if step.kind == "repair_overlay"
            )
            remaining_steps = tuple(
                step for step in resolved.steps if step.kind != "repair_overlay"
            )
            resolved = Recipe(
                # Repairs are source-shape overlays and must run before the
                # canonical adapter. Every other transformation stays ordered
                # after the automatically inserted map step.
                steps=(*leading_repairs, map_step, *remaining_steps),
                name=resolved.name,
                schema=resolved.schema or source.canonical_kind,
                seed=resolved.seed,
            )
        return resolved

    def build(
        self,
        source_id: str,
        recipe: Recipe | Mapping[str, Any] | Path | str,
        *,
        dataset_id: Optional[str] = None,
        materialize_assets: bool = False,
        progress: Optional[Callable[[int, int, str], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
        checkpoint: Optional[Callable[[int, RecipeResult], None]] = None,
        resume_result: Optional[RecipeResult] = None,
        resume_after_step: int = 0,
    ) -> DatasetVersion:
        source, snapshot = self._snapshot(source_id)
        missing_assets = [asset.reference for asset in snapshot.assets if asset.missing]
        if missing_assets:
            raise SourceError(
                "Source has missing referenced assets: " + ", ".join(missing_assets[:5])
            )
        resolved_recipe = self._resolved_recipe(source, recipe)
        base_dir = Path(source.spec.path).resolve() if source.spec.path else None
        if base_dir and base_dir.is_file():
            base_dir = base_dir.parent
        # Creating the reliability registry can load optional verifier
        # toolchains.  Keep ordinary v1-v6 recipes fast and dependency-light;
        # only resolve it when this immutable recipe actually names a profile.
        has_exact_verifier = any(
            bool(str(step.params.get("verifier_profile_revision_id") or "").strip())
            for step in resolved_recipe.steps
            if step.kind in {"score", "synthesize"}
        )
        verifier_engine = self._verifier_engine() if has_exact_verifier else None

        def resolve_profile(revision_id: str) -> Mapping[str, Any]:
            if verifier_engine is None:
                raise VersionError(
                    "Verifier profile revisions require Dataset Lab to be connected to SQLite"
                )
            return verifier_engine.resolve_binding(
                revision_id,
                modality=source.modality,
            )

        def invoke_profile(revision_id: str, row: Mapping[str, Any]) -> Any:
            if verifier_engine is None:
                raise VersionError(
                    "Verifier profile revisions require Dataset Lab to be connected to SQLite"
                )
            return verifier_engine.invoke_revision(revision_id, row)

        context = RecipeContext(
            base_dir=base_dir,
            teacher=self.teacher,
            synthesis_endpoint_type_default=(
                "injected" if self._teacher_explicit else "openai_compatible"
            ),
            verifier=self.verifier,
            verifier_profile_resolver=(resolve_profile if verifier_engine is not None else None),
            verifier_profile_invoker=(invoke_profile if verifier_engine is not None else None),
            synthesis_verifier_default=self._verifier_explicit,
            judge=self.judge,
            semantic_similarity=self.semantic_similarity,
            mixture_resolver=self.mixture_resolver,
            failure_resolver=self.failure_resolver,
            repair_overlay_resolver=(
                self._resolve_repair_overlay
                if any(step.kind == "repair_overlay" for step in resolved_recipe.steps)
                else None
            ),
            source_fingerprint=snapshot.fingerprint,
            progress=progress,
            cancelled=cancelled,
            checkpoint=checkpoint,
            resume_result=resume_result,
            resume_after_step=resume_after_step,
        )
        identified_records = seed_record_identities(
            snapshot.records,
            source_fingerprint=snapshot.fingerprint,
            source_name=source.id,
        )
        result = RecipeRunner(context).run(identified_records, resolved_recipe)
        version = self.store.publish(
            dataset_id=dataset_id or source.dataset_id,
            recipe=resolved_recipe,
            result=result,
            source=snapshot,
            materialize_assets=materialize_assets,
        )
        self._bind_version_verifiers(
            version,
            recipe=resolved_recipe,
            provenance=result.provenance,
        )
        return version

    def _bind_version_verifiers(
        self,
        version: DatasetVersion,
        *,
        recipe: Recipe,
        provenance: Sequence[StepProvenance],
    ) -> List[Any]:
        """Bind exact verifier inputs only after immutable publication.

        Raw v1-v6 verifier configuration remains represented in provenance as
        ``legacy_unqualified`` and deliberately cannot create a revision FK.
        """

        exact_steps = [
            (step_index, step, step.details.get("verifier_binding"))
            for step_index, step in enumerate(provenance)
            if isinstance(step.details.get("verifier_binding"), Mapping)
            and not step.details["verifier_binding"].get("legacy_unqualified")
            and str(
                step.details["verifier_binding"].get("verifier_profile_revision_id") or ""
            ).strip()
        ]
        if not exact_steps:
            return []
        verifier_engine = self._verifier_engine()
        if verifier_engine is None:
            raise VersionError(
                "Exact verifier provenance was published without a connected verifier catalog"
            )
        bindings: List[Any] = []
        for step_index, step, binding in exact_steps:
            revision_id = str(binding.get("verifier_profile_revision_id") or "").strip()
            role = (
                "dataset_synthesis_verifier"
                if step.kind == "synthesize"
                else "dataset_score_verifier"
            )
            bindings.append(
                verifier_engine.bind_revision(
                    revision_id,
                    domain_kind="dataset_version",
                    domain_id=version.version_id,
                    role=role,
                    context={
                        "dataset_id": version.dataset_id,
                        "dataset_version_id": version.version_id,
                        "dataset_content_hash": version.content_hash,
                        "recipe_hash": recipe.fingerprint,
                        "step_index": step_index,
                        "step_kind": step.kind,
                        "revision_hash": binding.get("revision_hash"),
                        "observation_count": len(step.details.get("observations") or []),
                    },
                )
            )
        return bindings

    def versions(self, dataset_id: Optional[str] = None) -> List[DatasetVersion]:
        return self.store.list(dataset_id)

    def get_version(self, version_id: str, *, dataset_id: Optional[str] = None) -> DatasetVersion:
        return self.store.get_any(version_id, dataset_id)

    def render_training_artifact(
        self,
        bindings: List[DatasetBinding | Mapping[str, Any] | str],
        **options: Any,
    ) -> TrainingDatasetArtifact:
        """Render an immutable, trainer-ready bundle from role bindings."""

        return self.training_artifacts.render(bindings, **options)

    def start_training_artifact_job(
        self,
        bindings: List[DatasetBinding | Mapping[str, Any] | str],
        *,
        submit: Optional[bool] = None,
        **options: Any,
    ) -> DatasetJob:
        """Queue persistent trainer-artifact preparation on the data worker.

        ``render_training_artifact`` deliberately remains synchronous for CLI
        callers and managed training launch preparation.  Dashboard/API callers
        use this method so tokenizer loading and record rendering never hold an
        HTTP request open.
        """

        normalized_bindings = [DatasetBinding.from_value(binding).to_dict() for binding in bindings]
        return self._start_background_job(
            "training_artifact",
            {
                "bindings": normalized_bindings,
                "options": copy.deepcopy(options),
            },
            submit=submit,
        )

    def compare_versions(
        self,
        left_version_id: str,
        right_version_id: str,
        **options: Any,
    ) -> DatasetVersionComparison:
        return DatasetVersionComparator(self.store).compare(
            left_version_id, right_version_id, **options
        )

    def sync_version_exposures(
        self,
        version_id: str,
        *,
        database: Any = None,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronize immutable derivation and failure-mining exposure lineage.

        Filesystem publication deliberately happens before SQLite mirroring.
        Public API reconciliation calls this method after it catalogs the child
        version; repeated calls are idempotent.  Legacy versions without a
        parent or failure-mining metadata simply produce an empty result.
        """

        if database is None:
            from halo_forge.run_db import get_database

            database = get_database()
        version = self.get_version(version_id, dataset_id=dataset_id)
        path = Path(version.path)
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VersionError(f"Invalid version manifest at {path}: {exc}") from exc
        parent_version_id = str(manifest.get("parent_version_id") or "").strip()
        parent_version_ids = [parent_version_id] if parent_version_id else []
        catalog_version = database.get_dataset_version(version.version_id)
        for parent in getattr(catalog_version, "parents", ()) or ():
            if isinstance(parent, Mapping):
                value = str(parent.get("parent_version_id") or "").strip()
                if value and value not in parent_version_ids:
                    parent_version_ids.append(value)
        inherited = inherit_dataset_version_exposures(
            database,
            parent_version_ids=parent_version_ids,
            child_version_id=version.version_id,
            provenance={
                "dataset_id": version.dataset_id,
                "recipe_hash": version.recipe_hash,
                "content_hash": version.content_hash,
            },
        )

        failure_details: Dict[str, Any] = {}
        review_details: Dict[str, Any] = {}
        stats_path = path / "stats.json"
        if stats_path.is_file():
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                if isinstance(stats, Mapping) and isinstance(stats.get("failure_mining"), Mapping):
                    failure_details = dict(stats["failure_mining"])
                if isinstance(stats, Mapping) and isinstance(
                    stats.get("review_label_set"), Mapping
                ):
                    review_details = dict(stats["review_label_set"])
            except (OSError, json.JSONDecodeError):
                # v1/v2 versions remain readable; provenance below is a second
                # route for manifests whose statistics were not retained.
                failure_details = {}
                review_details = {}
        if not failure_details or not review_details:
            provenance_path = path / "provenance.json"
            try:
                steps = json.loads(provenance_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                steps = []
            for step in steps if isinstance(steps, list) else []:
                if not failure_details and (
                    isinstance(step, Mapping)
                    and step.get("kind") == "failure_mining"
                    and isinstance(step.get("details"), Mapping)
                ):
                    failure_details = dict(step["details"])
                if not review_details and (
                    isinstance(step, Mapping)
                    and step.get("kind") == "review_label_set"
                    and isinstance(step.get("details"), Mapping)
                ):
                    review_details = dict(step["details"])
        direct_failure = (
            record_failure_mining_exposures(
                database,
                child_version_id=version.version_id,
                details=failure_details,
            )
            if failure_details
            else []
        )
        direct_review = (
            record_review_label_exposures(
                database,
                child_version_id=version.version_id,
                details=review_details,
            )
            if review_details
            else []
        )
        return {
            "version_id": version.version_id,
            "parent_version_id": parent_version_id or None,
            "inherited": [
                value.to_dict() if hasattr(value, "to_dict") else value for value in inherited
            ],
            "direct": [
                value.to_dict() if hasattr(value, "to_dict") else value
                for value in [*direct_failure, *direct_review]
            ],
        }

    def export(
        self,
        version_id: str,
        destination: Optional[Path | str] = None,
        *,
        output: Optional[Path | str] = None,
        format: str = "jsonl",
        split: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> Path:
        target = destination or output
        if target is None:
            raise VersionError("export requires destination/output")
        export_format = format.lower()
        if export_format in {"jsonl", "jsonlines"}:
            return self.store.export(version_id, target, dataset_id=dataset_id, split=split)
        if export_format not in {"csv", "parquet"}:
            raise VersionError("Dataset Lab exports jsonl, csv, or parquet")
        rows = self.store.load_records(version_id, dataset_id=dataset_id, split=split)
        destination_path = Path(target).expanduser().resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if export_format == "csv":
            fields = sorted({key for row in rows for key in row})
            with destination_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            key: (
                                json.dumps(value, ensure_ascii=False, sort_keys=True)
                                if isinstance(value, (dict, list))
                                else value
                            )
                            for key, value in row.items()
                        }
                    )
            return destination_path
        try:
            import pandas as pd

            pd.DataFrame(rows).to_parquet(destination_path, index=False)
        except (ImportError, ModuleNotFoundError) as exc:
            raise VersionError(
                "Parquet export requires `pip install halo-forge[data-lab]`"
            ) from exc
        return destination_path

    def _load_result(self, version: DatasetVersion) -> tuple[Recipe, RecipeResult, SourceSnapshot]:
        path = Path(version.path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        recipe = Recipe.from_value(json.loads((path / "recipe.json").read_text(encoding="utf-8")))
        records = self.store.load_records_with_lineage(
            version.version_id, dataset_id=version.dataset_id
        )
        splits = {
            name: self.store.load_records_with_lineage(
                version.version_id, dataset_id=version.dataset_id, split=name
            )
            for name in version.split_counts
        }
        rejected_path = path / "rejected.jsonl"
        quarantine_path = path / "quarantined.jsonl"

        def read_rows(file_path: Path) -> List[Dict[str, Any]]:
            return (
                [
                    json.loads(line)
                    for line in file_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if file_path.is_file()
                else []
            )

        provenance = [
            StepProvenance(**entry)
            for entry in json.loads((path / "provenance.json").read_text(encoding="utf-8"))
        ]
        result = RecipeResult(
            records=records,
            splits=splits,
            rejected=read_rows(rejected_path),
            quarantined=read_rows(quarantine_path),
            provenance=provenance,
            contamination=json.loads((path / "contamination.json").read_text(encoding="utf-8")),
            statistics=json.loads((path / "stats.json").read_text(encoding="utf-8")),
        )
        source_data = manifest["source"]
        assets = [AssetFingerprint(**asset) for asset in source_data.get("assets", [])]
        snapshot = SourceSnapshot(
            spec=SourceSpec.from_value(source_data["spec"]),
            records=records,
            fingerprint=source_data["fingerprint"],
            assets=assets,
            size_bytes=int(source_data.get("size_bytes", 0)),
            file_count=int(source_data.get("file_count", 0)),
        )
        return recipe, result, snapshot

    def materialize_version(
        self, version_id: str, *, dataset_id: Optional[str] = None
    ) -> DatasetVersion:
        version = self.get_version(version_id, dataset_id=dataset_id)
        if version.materialized_assets:
            return version
        recipe, result, source = self._load_result(version)
        return self.store.publish(
            dataset_id=version.dataset_id,
            recipe=recipe,
            result=result,
            source=source,
            materialize_assets=True,
            parent_version_id=version.version_id,
        )

    def materialize(
        self,
        version_id: str,
        *,
        dataset_id: Optional[str] = None,
        background: bool = True,
        submit: Optional[bool] = None,
        **_: Any,
    ) -> DatasetJob | DatasetVersion:
        if background:
            return self._start_background_job(
                "materialize",
                {"version_id": version_id, "dataset_id": dataset_id},
                submit=submit,
            )
        return self.materialize_version(version_id, dataset_id=dataset_id)

    def verify_version(
        self, version_id: str, *, dataset_id: Optional[str] = None, verify_source: bool = True
    ) -> Dict[str, Any]:
        return self.store.verify(version_id, dataset_id=dataset_id, verify_source=verify_source)

    def start_job(
        self, kind_or_source_id: str, payload_or_recipe: Any = None, **kwargs: Any
    ) -> DatasetJob:
        """Start a generic job or use the convenient ``start_job(source, recipe)`` build form."""
        submit = kwargs.pop("submit", None)
        if kind_or_source_id in {
            "build",
            "profile",
            "materialize",
            "failure_mining",
            "review_build",
            "training_artifact",
        } and isinstance(payload_or_recipe, Mapping):
            return self._start_background_job(
                kind_or_source_id,
                payload_or_recipe,
                submit=submit,
            )
        payload = {
            "source_id": kind_or_source_id,
            "recipe": self._serializable_recipe(payload_or_recipe),
            "dataset_id": kwargs.get("dataset_id"),
            "materialize_assets": bool(kwargs.get("materialize_assets", False)),
        }
        # canonical_kind is accepted for API compatibility and used only when the source lacks one.
        if kwargs.get("canonical_kind"):
            payload["canonical_kind"] = kwargs["canonical_kind"]
        return self._start_background_job("build", payload, submit=submit)

    def start_review_build_job(
        self,
        revision_id: str,
        *,
        review_root: Path | str,
        database_path: Optional[str] = None,
        submit: Optional[bool] = None,
        **options: Any,
    ) -> DatasetJob:
        """Queue immutable Dataset Lab publication from one label-set revision."""

        return self._start_background_job(
            "review_build",
            {
                "revision_id": str(revision_id),
                "review_root": str(Path(review_root).expanduser().resolve()),
                "database_path": database_path,
                **copy.deepcopy(options),
            },
            submit=submit,
        )

    @staticmethod
    def _serializable_recipe(recipe: Any) -> Any:
        if isinstance(recipe, Recipe):
            return recipe.to_dict()
        if isinstance(recipe, Path):
            return str(recipe)
        return copy.deepcopy(recipe)

    def _build_job(self, context: JobContext, payload: Dict[str, Any]) -> DatasetVersion:
        source_id = payload["source_id"]
        recipe = payload["recipe"]
        if payload.get("canonical_kind"):
            source = self.get_source(source_id)
            if not source.canonical_kind:
                resolved = Recipe.from_value(recipe)
                recipe = Recipe(
                    steps=(
                        RecipeStep("map", {"schema": payload["canonical_kind"]}),
                        *resolved.steps,
                    ),
                    name=resolved.name,
                    schema=payload["canonical_kind"],
                    seed=resolved.seed,
                )
        source = self.get_source(source_id)
        resolved_recipe = self._resolved_recipe(source, recipe)
        resume_result: Optional[RecipeResult] = None
        resume_after_step = 0
        resume_checkpoint_path: Optional[Path] = None
        checkpoint_data = context.checkpoint_data
        checkpoint_path = checkpoint_data.get("checkpoint_path")
        if (
            checkpoint_path
            and checkpoint_data.get("recipe_hash") == resolved_recipe.fingerprint
            and checkpoint_data.get("source_fingerprint") == source.fingerprint
        ):
            candidate = Path(str(checkpoint_path))
            if candidate.is_file():
                resume_result = self._read_recipe_checkpoint(candidate)
                resume_after_step = int(checkpoint_data.get("completed_step") or 0)
                resume_checkpoint_path = candidate
                context.log(f"Resuming after recipe step {resume_after_step}")
        context.log(f"Building source {source_id}")

        def progress(done: int, total: int, stage: str) -> None:
            context.check_cancelled()
            context.progress(stage=stage, processed=done, total=total)

        def checkpoint(index: int, result: RecipeResult) -> None:
            path = self._write_recipe_checkpoint(context.job_id, result)
            context.progress(
                stage=f"completed:{resolved_recipe.steps[index].kind}",
                processed=index + 1,
                total=len(resolved_recipe.steps),
                accepted=len(result.records),
                rejected=len(result.rejected) + len(result.quarantined),
                output_size_bytes=path.stat().st_size,
            )
            context.checkpoint(
                completed_step=index + 1,
                record_count=len(result.records),
                checkpoint_path=str(path),
                recipe_hash=resolved_recipe.fingerprint,
                source_fingerprint=source.fingerprint,
            )

        version = self.build(
            source_id,
            resolved_recipe,
            dataset_id=payload.get("dataset_id"),
            materialize_assets=bool(payload.get("materialize_assets", False)),
            progress=progress,
            cancelled=context.cancelled,
            checkpoint=checkpoint,
            resume_result=resume_result,
            resume_after_step=resume_after_step,
        )
        for path in {
            resume_checkpoint_path,
            (
                Path(str(context.checkpoint_data["checkpoint_path"]))
                if context.checkpoint_data.get("checkpoint_path")
                else None
            ),
        }:
            if path is not None:
                path.unlink(missing_ok=True)
        context.progress(
            stage="published",
            accepted=version.row_count,
            output_size_bytes=sum(
                path.stat().st_size for path in Path(version.path).rglob("*") if path.is_file()
            ),
        )
        return version

    def _write_recipe_checkpoint(self, job_id: str, result: RecipeResult) -> Path:
        directory = self.root / ".job-checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{job_id}.json"
        fd, temporary = tempfile.mkstemp(prefix=f".{job_id}.", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(result.to_dict(include_records=True), handle, sort_keys=True, default=str)
                handle.write("\n")
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return destination

    @staticmethod
    def _read_recipe_checkpoint(path: Path) -> RecipeResult:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return RecipeResult(
                records=list(payload.get("records") or []),
                splits=dict(payload.get("splits") or {}),
                rejected=list(payload.get("rejected") or []),
                quarantined=list(payload.get("quarantined") or []),
                provenance=[StepProvenance(**entry) for entry in payload.get("provenance") or []],
                contamination=dict(payload.get("contamination") or {}),
                statistics=dict(payload.get("statistics") or {}),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise VersionError(f"Invalid Dataset Lab recipe checkpoint {path}: {exc}") from exc

    def _profile_job(self, context: JobContext, payload: Dict[str, Any]) -> Dict[str, Any]:
        context.progress(stage="profiling", processed=0, total=1)
        value = self.profile(payload["source_id"])
        context.progress(stage="profiled", processed=1, total=1, accepted=int(value["row_count"]))
        return value

    def _failure_mining_job(self, context: JobContext, payload: Dict[str, Any]) -> DatasetVersion:
        """Publish one explicitly reviewed evaluation selection as a child version."""

        builder = FailureMiningBuilder(self.store)
        context.progress(stage="reviewing", processed=0, total=1)
        preview = builder.preview(
            payload.get("comparison") or {},
            payload.get("selector"),
            exclusions=payload.get("exclusions") or (),
        )
        context.check_cancelled()
        context.progress(
            stage="building",
            processed=0,
            total=max(1, preview.selected_count),
            accepted=preview.selected_count,
            rejected=len(preview.excluded),
        )
        version = builder.build(
            parent_version_id=str(payload.get("parent_version_id") or ""),
            dataset_id=payload.get("dataset_id"),
            comparison=payload.get("comparison") or {},
            selector=payload.get("selector"),
            exclusions=payload.get("exclusions") or (),
            target_split=str(payload.get("target_split") or "train"),
            mode=str(payload.get("mode") or "append"),
            materialize_assets=payload.get("materialize_assets"),
        )
        output_size = sum(
            path.stat().st_size for path in Path(version.path).rglob("*") if path.is_file()
        )
        context.progress(
            stage="published",
            processed=preview.selected_count,
            total=preview.selected_count,
            accepted=preview.selected_count,
            rejected=len(preview.excluded),
            output_size_bytes=output_size,
        )
        return version

    def _training_artifact_job(
        self, context: JobContext, payload: Dict[str, Any]
    ) -> TrainingDatasetArtifact:
        """Render and atomically publish one trainer-ready artifact bundle."""

        bindings = list(payload.get("bindings") or [])
        if not bindings:
            raise VersionError("training artifact jobs require at least one dataset binding")
        options = dict(payload.get("options") or {})
        context.progress(stage="validating", processed=0, total=len(bindings))
        context.check_cancelled()
        context.log(
            "Preparing training artifact for "
            + ", ".join(
                f"{value.get('role', 'train')}="
                f"{value.get('dataset_version_id')}:{value.get('split')}"
                for value in bindings
            )
        )
        context.progress(stage="rendering", processed=0, total=len(bindings))
        artifact = self.render_training_artifact(bindings, **options)
        context.check_cancelled()
        output_size = sum(
            path.stat().st_size for path in Path(artifact.path).rglob("*") if path.is_file()
        )
        context.progress(
            stage="published",
            processed=len(bindings),
            total=len(bindings),
            accepted=sum(int(value) for value in artifact.row_counts.values()),
            output_size_bytes=output_size,
        )
        context.log(f"Published training artifact {artifact.artifact_id}")
        return artifact

    def _materialize_job(self, context: JobContext, payload: Dict[str, Any]) -> DatasetVersion:
        context.progress(stage="materializing", processed=0, total=1)
        value = self.materialize_version(
            payload["version_id"], dataset_id=payload.get("dataset_id")
        )
        context.progress(stage="published", processed=1, total=1, accepted=value.row_count)
        return value

    def _review_build_job(self, context: JobContext, payload: Dict[str, Any]) -> DatasetVersion:
        """Render a label set and atomically publish its reviewed child version."""

        from halo_forge.review_lab import ReviewLabService
        from halo_forge.run_db import get_database

        from .review_builds import ReviewDatasetBuilder

        context.progress(stage="loading_label_set", processed=0, total=3)
        database = get_database(payload.get("database_path"))
        review = ReviewLabService(database, root=payload["review_root"])
        revision = review.get_label_set_revision(payload["revision_id"])
        if revision is None:
            raise VersionError(f"Unknown label-set revision: {payload['revision_id']}")
        # Label sets are paginated at the service boundary.  Internal builds
        # must consume every page so a large reviewed queue cannot silently
        # publish a truncated child version.
        items = list(review.iter_label_set_items(payload["revision_id"]))
        context.check_cancelled()
        context.progress(
            stage="merging_reviewed_records",
            processed=1,
            total=3,
            accepted=len(items),
        )
        version = ReviewDatasetBuilder(self.store).build(
            revision,
            items,
            dataset_id=str(payload.get("dataset_id") or ""),
            parent_version_id=payload.get("parent_version_id"),
            build_mode=str(payload.get("build_mode") or "append"),
            target_split=str(payload.get("target_split") or "train"),
            materialize_assets=payload.get("materialize_assets"),
            schema=payload.get("schema"),
            check_cancelled=context.check_cancelled,
        )
        context.progress(
            stage="published",
            processed=3,
            total=3,
            accepted=version.row_count,
            output_size_bytes=sum(
                path.stat().st_size for path in Path(version.path).rglob("*") if path.is_file()
            ),
        )
        return version

    def get_job(self, job_id: str) -> DatasetJob:
        return self.job_manager.get(job_id)

    def run_queued(self, job_id: str) -> DatasetJob:
        """Execute a scheduler-owned job without creating a legacy future."""

        return self.job_manager.run_queued(job_id)

    def list_jobs(self, *, status: Optional[str] = None) -> List[DatasetJob]:
        return self.job_manager.list(status=status)

    jobs = list_jobs

    def cancel(self, job_id: str) -> DatasetJob:
        job = self.job_manager.cancel(job_id)
        if self.scheduler is not None and job.work_item_id:
            self.scheduler.cancel(job.work_item_id)
        return self.job_manager.get(job_id)

    def retry(self, job_id: str) -> DatasetJob:
        job = self.job_manager.get(job_id)
        if self.scheduler is None or not job.work_item_id:
            return self.job_manager.retry(job_id)
        retried = self.scheduler.retry(
            job.work_item_id,
            reason="operator requested Dataset Lab retry",
            force=True,
        )
        if retried is None:
            raise VersionError(f"Dataset job {job_id} has no retryable workstation work item")
        return self.job_manager.get(job_id)

    def close(self) -> None:
        self.job_manager.shutdown()

    def __enter__(self) -> "DatasetLab":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


DatasetLabService = DatasetLab

__all__ = ["DatasetLab", "DatasetLabService", "DatasetSource", "PreviewPage"]
