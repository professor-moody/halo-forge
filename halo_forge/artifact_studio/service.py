"""Integrated Artifact Studio application service.

The facade composes the immutable filesystem store, Lab v4 SQLite catalog, and
durable workstation scheduler.  It intentionally contains no HTTP or CLI
logic, making every transport share the same validation and lifecycle rules.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from halo_forge.artifact_lab import (
    ArtifactOperationService,
    ArtifactRegistration,
    ArtifactStore,
    CleanupPlan,
    CleanupProtections,
    CleanupResult,
    OperationSpec,
    PortableExportBundle,
    default_engines,
    fingerprint,
    probe_artifact_loadability,
)
from halo_forge.inference.convert import list_supported_formats, list_supported_quants
from halo_forge.inference.merge import list_supported_methods
from halo_forge.run_db import (
    ArtifactOccurrenceRecord,
    ArtifactOperationRecord,
    ArtifactQualificationRecord,
    LabV4Catalog,
    RunDatabase,
    ServingProfileRevisionRecord,
    WorkItemRecord,
)
from halo_forge.version import PACKAGE_VERSION
from halo_forge.workstation_jobs import WorkstationScheduler

from .models import (
    ArtifactStudioError,
    PromotionBlocked,
    ServingReservation,
    StudioQueueReceipt,
    UnsupportedArtifactCapability,
)

_FILESYSTEM_TO_CATALOG_KIND = {
    "checkpoint": "checkpoint",
    "adapter": "adapter",
    "final": "final_model",
    "merged": "merged_model",
    "converted": "converted_model",
    "quantized": "quantized_model",
    "export_bundle": "export_bundle",
}
_CATALOG_TO_FILESYSTEM_KIND = {value: key for key, value in _FILESYSTEM_TO_CATALOG_KIND.items()}


class QualificationExecutor(Protocol):
    """Worker hook that returns real evaluation-derived qualification output."""

    def __call__(
        self,
        qualification: ArtifactQualificationRecord,
        launch_spec: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class ServingStarter(Protocol):
    """Optional worker hook that starts a real managed model-server process."""

    def __call__(
        self,
        profile_revision: Mapping[str, Any],
        occurrence: ArtifactOccurrenceRecord,
        launch_spec: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


class ArtifactStudioService:
    """One application-level entry point for artifact research operations."""

    def __init__(
        self,
        database: Optional[RunDatabase] = None,
        *,
        store: Optional[ArtifactStore] = None,
        catalog: Optional[LabV4Catalog] = None,
        scheduler: Optional[WorkstationScheduler] = None,
        operation_service: Optional[ArtifactOperationService] = None,
        qualification_executor: Optional[QualificationExecutor] = None,
        serving_starter: Optional[ServingStarter] = None,
        artifact_root: Path | str | None = None,
    ):
        if database is None:
            if catalog is not None:
                database = catalog.database
            elif scheduler is not None:
                database = scheduler.database
            else:
                raise ValueError("database is required unless catalog or scheduler supplies one")
        self.database = database
        self.store = store or ArtifactStore(artifact_root)
        self.catalog = catalog or LabV4Catalog(database)
        self.scheduler = scheduler or WorkstationScheduler(database)
        self.operations = operation_service or ArtifactOperationService(
            self.store, engines=default_engines()
        )
        self.qualification_executor = qualification_executor
        self.serving_starter = serving_starter

    def _evaluation_artifact_root(self) -> Optional[str]:
        """Return the configured evaluation store used by qualification work.

        Artifact work is durable and may be executed by a different process.
        Persisting this root in the launch specification keeps that worker on
        the same evaluation store as the API or desktop process that queued it.
        Custom qualification callables intentionally return no root.
        """

        evaluations = getattr(self.qualification_executor, "evaluations", None)
        jobs = getattr(evaluations, "jobs", None)
        root = getattr(jobs, "artifact_root", None)
        return str(root) if root is not None else None

    # -- catalog synchronization and artifact browsing -----------------

    @staticmethod
    def _filesystem_kind(value: str) -> str:
        normalized = value.strip().lower()
        if normalized in _FILESYSTEM_TO_CATALOG_KIND:
            return normalized
        if normalized in _CATALOG_TO_FILESYSTEM_KIND:
            return _CATALOG_TO_FILESYSTEM_KIND[normalized]
        allowed = sorted(set(_FILESYSTEM_TO_CATALOG_KIND) | set(_CATALOG_TO_FILESYSTEM_KIND))
        raise ValueError(f"Unsupported artifact kind {value!r}; choose from: {', '.join(allowed)}")

    def _sync_registration(
        self,
        registration: ArtifactRegistration,
        *,
        model_id: str,
        backend: str,
        artifact_kind: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        chat_template_hash: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        run_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        trial_segment_id: Optional[str] = None,
    ) -> ArtifactOccurrenceRecord:
        fs_kind = self._filesystem_kind(artifact_kind or registration.occurrence.artifact_kind)
        blob = self.catalog.upsert_blob(
            blob_id=registration.blob.id,
            content_hash=registration.blob.content_hash,
            artifact_type=registration.blob.artifact_kind,
            format=registration.blob.format,
            dtype=registration.blob.dtype,
            quantization=registration.blob.quantization,
            size_bytes=registration.blob.size_bytes,
            integrity_state=registration.blob.integrity,
            manifest=registration.blob.to_dict(),
        )
        self.catalog.add_location(
            location_id=registration.location.id,
            blob_id=blob.id,
            path=registration.location.path,
            storage_mode=registration.location.location_kind,
            state=registration.location.state,
            size_bytes=registration.blob.size_bytes,
            metadata={
                "filesystem_location": registration.location.to_dict(),
                "source_path": registration.location.source_path,
            },
        )
        existing = self.catalog.get_occurrence(registration.occurrence.id)
        if existing is not None:
            if existing.blob_id != blob.id:
                raise ArtifactStudioError(
                    f"Occurrence {existing.id!r} is already bound to different content"
                )
            return existing
        return self.catalog.create_occurrence(
            occurrence_id=registration.occurrence.id,
            blob_id=blob.id,
            artifact_kind=_FILESYSTEM_TO_CATALOG_KIND[fs_kind],
            model_id=model_id,
            backend=backend,
            tokenizer_revision=tokenizer_revision,
            chat_template_hash=chat_template_hash,
            metadata={
                **dict(metadata or {}),
                "filesystem_occurrence": registration.occurrence.to_dict(),
            },
            run_id=run_id or registration.occurrence.run_id,
            run_group_id=run_group_id or registration.occurrence.run_group_id,
            trial_id=trial_id or registration.occurrence.trial_id,
            trial_segment_id=trial_segment_id or registration.occurrence.segment_id,
        )

    def import_artifact(
        self,
        source: Path | str,
        *,
        artifact_kind: str,
        artifact_format: str,
        model_id: Optional[str] = None,
        backend: str = "local",
        managed: bool = False,
        dtype: Optional[str] = None,
        quantization: Optional[str] = None,
        occurrence_id: Optional[str] = None,
        run_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        trial_segment_id: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        chat_template_hash: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        fs_kind = self._filesystem_kind(artifact_kind)
        source_path = Path(source).expanduser().resolve(strict=True)
        registration = self.store.import_artifact(
            source_path,
            artifact_kind=fs_kind,
            artifact_format=artifact_format,
            managed=managed,
            dtype=dtype,
            quantization=quantization,
            quantization_method=(
                "post_training"
                if quantization and quantization.lower() in {"q4", "q8", "int4", "int8"}
                else None
            ),
            occurrence_id=occurrence_id,
            run_id=run_id,
            run_group_id=run_group_id,
            trial_id=trial_id,
            segment_id=trial_segment_id,
            metadata=metadata,
        )
        occurrence = self._sync_registration(
            registration,
            model_id=model_id or source_path.name,
            backend=backend,
            artifact_kind=fs_kind,
            tokenizer_revision=tokenizer_revision,
            chat_template_hash=chat_template_hash,
            metadata=metadata,
            run_id=run_id,
            run_group_id=run_group_id,
            trial_id=trial_id,
            trial_segment_id=trial_segment_id,
        )
        return self.show_artifact(occurrence.id)

    def _all_occurrences(self) -> list[ArtifactOccurrenceRecord]:
        values: list[ArtifactOccurrenceRecord] = []
        offset = 0
        while True:
            page = self.catalog.list_occurrences(limit=1000, offset=offset)
            values.extend(page)
            if len(page) < 1000:
                return values
            offset += len(page)

    def _resolve_occurrence(self, identifier: str) -> Optional[ArtifactOccurrenceRecord]:
        occurrence = self.catalog.get_occurrence(identifier)
        if occurrence is not None:
            return occurrence
        blob = self.catalog.get_blob(identifier) or self.catalog.find_blob(identifier)
        occurrences = self._all_occurrences()
        if blob is not None:
            return next((item for item in occurrences if item.blob_id == blob.id), None)
        normalized_alias = identifier.strip().lower()
        return next(
            (item for item in occurrences if normalized_alias in self.catalog.aliases_for(item.id)),
            None,
        )

    def _artifact_view(self, occurrence: ArtifactOccurrenceRecord) -> dict[str, Any]:
        blob = self.catalog.get_blob(occurrence.blob_id)
        if blob is None:
            raise ArtifactStudioError(f"Artifact occurrence {occurrence.id} has no blob")
        return {
            "occurrence": occurrence.to_dict(),
            "blob": blob.to_dict(),
            "locations": [item.to_dict() for item in self.catalog.list_locations(blob.id)],
            "aliases": self.catalog.aliases_for(occurrence.id),
            "qualifications": [
                item.to_dict()
                for item in self.catalog.list_qualifications(occurrence_id=occurrence.id)
            ],
        }

    def list_artifacts(
        self,
        *,
        artifact_kind: Optional[str] = None,
        pinned: Optional[bool] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        catalog_kind = (
            _FILESYSTEM_TO_CATALOG_KIND[self._filesystem_kind(artifact_kind)]
            if artifact_kind
            else None
        )
        occurrences = self.catalog.list_occurrences(
            artifact_kind=catalog_kind,
            pinned=pinned,
            run_id=run_id,
            limit=limit,
            offset=offset,
        )
        return {
            "items": [self._artifact_view(item) for item in occurrences],
            "limit": max(1, min(1000, int(limit))),
            "offset": max(0, int(offset)),
            "has_more": len(occurrences) >= max(1, min(1000, int(limit))),
        }

    def show_artifact(self, identifier: str) -> dict[str, Any]:
        occurrence = self._resolve_occurrence(identifier)
        if occurrence is None:
            blob = self.catalog.get_blob(identifier) or self.catalog.find_blob(identifier)
            if blob is not None:
                return {
                    "occurrence": None,
                    "blob": blob.to_dict(),
                    "locations": [item.to_dict() for item in self.catalog.list_locations(blob.id)],
                    "aliases": [],
                    "qualifications": [],
                }
            raise KeyError(f"Unknown artifact: {identifier}")
        return self._artifact_view(occurrence)

    def lineage(self, identifier: str) -> dict[str, Any]:
        occurrence = self._resolve_occurrence(identifier)
        if occurrence is None:
            raise KeyError(f"Unknown artifact: {identifier}")
        blob = self.catalog.get_blob(occurrence.blob_id)
        assert blob is not None
        filesystem_lineage = self.store.lineage(blob.content_hash)
        return {
            "catalog": self.catalog.lineage(occurrence.id),
            "content": filesystem_lineage.to_dict(),
        }

    def verify_artifact(
        self,
        identifier: str,
        *,
        loader_probe: Optional[Callable[..., Any]] = None,
        round_trip_report: Optional[Mapping[str, Any]] = None,
        probe_loadability: bool = False,
    ) -> dict[str, Any]:
        occurrence = self._resolve_occurrence(identifier)
        if occurrence is None:
            raise KeyError(f"Unknown artifact: {identifier}")
        blob = self.catalog.get_blob(occurrence.blob_id)
        assert blob is not None
        report = self.store.verify(
            blob.content_hash,
            loader_probe=(
                loader_probe
                if loader_probe is not None
                else probe_artifact_loadability if probe_loadability else None
            ),
            round_trip_report=round_trip_report,
        )
        self._record_verification(blob.id, report)
        return report.to_dict()

    @staticmethod
    def _verification_rank(state: str) -> int:
        return {
            "unverified": 0,
            "hash_verified": 1,
            "verified": 2,  # Compatibility with early Lab v4 rows.
            "structural_verified": 2,
            "load_verified": 3,
            "round_trip_verified": 4,
        }.get(state, 0)

    def _record_verification(self, blob_id: str, report: Any) -> None:
        blob = self.catalog.get_blob(blob_id)
        if blob is None:
            return
        if not report.passed:
            state = "corrupt"
        elif self._verification_rank(report.verification_level) >= self._verification_rank(
            blob.integrity_state
        ):
            state = report.verification_level
        else:
            state = blob.integrity_state
        self.catalog.verify_blob(blob_id, state=state)

    def _require_verification(
        self,
        occurrence: ArtifactOccurrenceRecord,
        *,
        action: str,
        level: str = "structural_verified",
    ) -> dict[str, Any]:
        blob = self._blob_for(occurrence)
        report = self.store.verify(blob.content_hash, structural=True)
        self._record_verification(blob.id, report)
        if not report.satisfies(level):
            detail = "; ".join(report.errors) or (
                f"verification reached {report.verification_level}, requires {level}"
            )
            raise ArtifactStudioError(f"Artifact cannot {action}: {detail}")
        return report.to_dict()

    def adopt_artifact(self, identifier: str) -> dict[str, Any]:
        """Copy referenced bytes into the managed library without losing provenance."""

        occurrence = self._resolve_occurrence(identifier)
        if occurrence is None:
            raise KeyError(f"Unknown artifact: {identifier}")
        blob = self._blob_for(occurrence)
        managed = self.store.adopt(blob.content_hash)
        self.catalog.add_location(
            location_id=managed.id,
            blob_id=blob.id,
            path=managed.path,
            storage_mode="managed",
            state=managed.state,
            size_bytes=blob.size_bytes,
            metadata={"filesystem_location": managed.to_dict()},
        )
        return self._artifact_view(occurrence)

    def update_annotations(
        self,
        occurrence_id: str,
        *,
        pinned: Optional[bool] = None,
        tags: Optional[Sequence[str]] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        occurrence = self.catalog.update_occurrence_annotations(
            occurrence_id, pinned=pinned, tags=tags, notes=notes
        )
        if occurrence is None:
            raise KeyError(f"Unknown artifact occurrence: {occurrence_id}")
        return self._artifact_view(occurrence)

    def pin_artifact(self, occurrence_id: str, *, pinned: bool = True) -> dict[str, Any]:
        return self.update_annotations(occurrence_id, pinned=pinned)

    def tag_artifact(
        self,
        occurrence_id: str,
        tags: Sequence[str],
        *,
        replace: bool = False,
    ) -> dict[str, Any]:
        occurrence = self._require_occurrence(occurrence_id)
        requested = {str(item).strip() for item in tags if str(item).strip()}
        resolved = requested if replace else requested | set(occurrence.to_dict()["tags"])
        return self.update_annotations(occurrence_id, tags=sorted(resolved))

    def set_alias(
        self,
        alias: str,
        occurrence_id: str,
        *,
        override_reason: Optional[str] = None,
    ) -> dict[str, Any]:
        if self.catalog.get_occurrence(occurrence_id) is None:
            raise KeyError(f"Unknown artifact occurrence: {occurrence_id}")
        return self.catalog.set_alias(alias, occurrence_id, override_reason=override_reason)

    # -- durable transformation and qualification jobs ----------------

    def _find_reusable_operation(self, operation_hash: str) -> Optional[ArtifactOperationRecord]:
        completed = self.catalog.find_completed_operation(operation_hash)
        if completed is not None:
            return completed
        for status in ("queued", "running"):
            for operation in self.catalog.list_operations(status=status, limit=1000):
                if operation.operation_hash == operation_hash:
                    return operation
        return None

    def _find_reusable_domain_work(
        self,
        *,
        domain_kind: str,
        domain_id: str,
        kind: str,
    ) -> Optional[WorkItemRecord]:
        matches = []
        offset = 0
        while True:
            page = self.database.list_work_items(kinds=(kind,), limit=1000, offset=offset)
            matches.extend(
                item
                for item in page
                if item.domain_kind == domain_kind and item.domain_id == domain_id
            )
            if len(page) < 1000:
                break
            offset += len(page)
        for status in ("running", "queued", "blocked", "needs_reconciliation", "completed"):
            found = next((item for item in matches if item.status == status), None)
            if found is not None:
                return found
        return None

    def _queue_operation(
        self,
        spec: OperationSpec,
        *,
        input_occurrence_ids: Sequence[str],
        priority: int = 0,
        max_retries: int = 1,
        resource_requirements: Optional[Mapping[str, Any]] = None,
    ) -> StudioQueueReceipt:
        existing = self._find_reusable_operation(spec.fingerprint)
        if existing is not None:
            return StudioQueueReceipt(
                domain_kind="artifact_operation",
                domain_id=existing.id,
                work_item_id=existing.work_item_id,
                status=existing.status,
                reused=True,
            )
        operation_id = f"artifact-operation-{uuid.uuid4().hex}"
        work_item_id = f"artifact-work-{uuid.uuid4().hex}"
        projected_disk_bytes = sum(
            self._blob_for(self._require_occurrence(item)).size_bytes
            for item in input_occurrence_ids
        )
        resolved_requirements = {
            "exclusive_heavy_operation": True,
            "projected_disk_bytes": projected_disk_bytes,
            "output_path": str(self.store.root),
            **dict(resource_requirements or {}),
        }
        launch_spec = {
            "handler": "artifact_studio.execute_work_item",
            "domain_kind": "artifact_operation",
            "domain_id": operation_id,
            "artifact_root": str(self.store.root),
            "operation_spec": spec.to_dict(),
            "input_occurrence_ids": list(input_occurrence_ids),
            "resource_requirements": resolved_requirements,
        }
        work_item = self.scheduler.enqueue(
            kind="artifact_operation",
            launch_spec=launch_spec,
            resource_class="accelerator",
            resource_requirements=resolved_requirements,
            domain_kind="artifact_operation",
            domain_id=operation_id,
            priority=priority,
            max_retries=max_retries,
            work_item_id=work_item_id,
        )
        try:
            operation = self.catalog.create_operation(
                operation_id=operation_id,
                operation_type=spec.operation_type,
                operation_hash=spec.fingerprint,
                resolved_spec=spec.to_dict(),
                input_occurrence_ids=input_occurrence_ids,
                work_item_id=work_item.id,
            )
        except Exception:
            self.scheduler.cancel(work_item.id)
            raise
        if operation.id != operation_id:
            self.scheduler.cancel(work_item.id)
            return StudioQueueReceipt(
                domain_kind="artifact_operation",
                domain_id=operation.id,
                work_item_id=operation.work_item_id,
                status=operation.status,
                reused=True,
            )
        return StudioQueueReceipt(
            domain_kind="artifact_operation",
            domain_id=operation.id,
            work_item_id=work_item.id,
            status=operation.status,
        )

    def queue_merge(
        self,
        *,
        input_occurrence_ids: Sequence[str],
        base_model: str,
        base_occurrence_id: Optional[str] = None,
        base_content_hash: Optional[str] = None,
        base_revision: Optional[str] = None,
        mode: str = "combine",
        method: str = "dare_ties",
        weights: Optional[Sequence[float]] = None,
        bake_after_merge: bool = False,
        priority: int = 0,
        max_retries: int = 1,
    ) -> StudioQueueReceipt:
        mode = mode.strip().lower()
        if mode not in {"bake", "combine"}:
            raise ValueError("merge mode must be bake or combine")
        if not base_model.strip() and not base_occurrence_id and not base_content_hash:
            raise ValueError("A resolved base_model is required")
        if mode == "bake" and len(input_occurrence_ids) != 1:
            raise ValueError("Bake requires exactly one adapter occurrence")
        if mode == "combine" and len(input_occurrence_ids) < 2:
            raise ValueError("Combine requires at least two adapter occurrences")
        if method not in list_supported_methods():
            raise ValueError(
                f"Unsupported merge method {method!r}; choose from: "
                f"{', '.join(list_supported_methods())}"
            )
        if weights is not None and len(weights) != len(input_occurrence_ids):
            raise ValueError("weights must match the number of input occurrences")
        occurrences = [self._require_occurrence(item) for item in input_occurrence_ids]
        for occurrence in occurrences:
            self._require_verification(occurrence, action="be used as a merge adapter")

        explicit_base = base_occurrence_id or base_content_hash
        base_occurrence = self._resolve_occurrence(str(explicit_base or base_model).strip())
        if explicit_base and base_occurrence is None:
            raise KeyError(f"Unknown base artifact: {explicit_base}")
        resolved_base_model = base_model.strip()
        resolved_base_revision = str(base_revision or "").strip() or None
        base_parameters: dict[str, Any]
        if base_occurrence is not None:
            self._require_verification(base_occurrence, action="be used as a merge base")
            base_blob = self._blob_for(base_occurrence)
            operation_occurrences = [base_occurrence, *occurrences]
            hashes = tuple(self._blob_for(item).content_hash for item in operation_occurrences)
            base_parameters = {
                "base_model": base_occurrence.model_id,
                "base_revision": None,
                "base_occurrence_id": base_occurrence.id,
                "base_content_hash": base_blob.content_hash,
                "base_input_index": 0,
            }
        else:
            if resolved_base_revision is None and "@" in resolved_base_model:
                resolved_base_model, resolved_base_revision = resolved_base_model.rsplit("@", 1)
                resolved_base_model = resolved_base_model.strip()
                resolved_base_revision = resolved_base_revision.strip() or None
            if not resolved_base_model or not resolved_base_revision:
                raise ValueError(
                    "Merge base must be an artifact occurrence/content hash or a model "
                    "reference pinned as model@revision"
                )
            operation_occurrences = occurrences
            hashes = tuple(self._blob_for(item).content_hash for item in occurrences)
            base_parameters = {
                "base_model": resolved_base_model,
                "base_revision": resolved_base_revision,
                "base_occurrence_id": None,
                "base_content_hash": None,
            }
        spec = OperationSpec(
            operation_type=mode,
            input_content_hashes=hashes,
            output_kind=("adapter" if mode == "combine" and not bake_after_merge else "merged"),
            output_format="hf",
            parameters={
                **base_parameters,
                "method": method,
                "weights": None if weights is None else list(weights),
                "bake_after_merge": bake_after_merge,
                "model_id": (
                    base_occurrence.model_id
                    if base_occurrence is not None
                    else f"{resolved_base_model}@{resolved_base_revision}"
                ),
                "backend": occurrences[0].backend,
            },
            tool_id="halo_forge.inference.merge",
            tool_version=PACKAGE_VERSION,
        )
        return self._queue_operation(
            spec,
            input_occurrence_ids=[item.id for item in operation_occurrences],
            priority=priority,
            max_retries=max_retries,
        )

    def queue_convert(
        self,
        *,
        occurrence_id: str,
        target_format: str,
        quantization: str,
        priority: int = 0,
        max_retries: int = 1,
        allow_unquantized_fallback: bool = False,
    ) -> StudioQueueReceipt:
        occurrence = self._require_occurrence(occurrence_id)
        target = target_format.strip().lower()
        quant = quantization.strip().lower()
        if target not in list_supported_formats():
            raise UnsupportedArtifactCapability(
                f"No verified {target!r} conversion engine is available; "
                f"current formats: {', '.join(list_supported_formats())}"
            )
        if quant not in list_supported_quants():
            raise ValueError(
                f"Unsupported quantization {quant!r}; choose from: "
                f"{', '.join(list_supported_quants())}"
            )
        if target == "hf" and quant not in {"bf16", "fp16", "fp32"}:
            raise UnsupportedArtifactCapability(
                "Hugging Face export supports dtype conversion only; use MLX or GGUF for q4/q8"
            )
        source_blob = self._blob_for(occurrence)
        operation_type = "quantize" if quant in {"q4", "q8"} else "convert"
        output_kind = "quantized" if operation_type == "quantize" else "converted"
        spec = OperationSpec(
            operation_type=operation_type,
            input_content_hashes=(source_blob.content_hash,),
            output_kind=output_kind,
            output_format=target,
            output_dtype=quant if quant in {"bf16", "fp16", "fp32"} else None,
            output_quantization=quant,
            parameters={
                "allow_unquantized_fallback": allow_unquantized_fallback,
                "model_id": occurrence.model_id,
                "backend": target,
            },
            tool_id="halo_forge.inference.convert",
            tool_version=PACKAGE_VERSION,
        )
        return self._queue_operation(
            spec,
            input_occurrence_ids=(occurrence.id,),
            priority=priority,
            max_retries=max_retries,
        )

    def queue_export(
        self,
        *,
        occurrence_id: str,
        destination: Path | str,
        replay_identity: Optional[Mapping[str, Any]] = None,
        dataset_identity: Optional[Mapping[str, Any]] = None,
        license_metadata: Optional[Mapping[str, Any]] = None,
        model_card: Optional[str] = None,
        priority: int = 0,
        max_retries: int = 1,
        projected_ram_bytes: int = 0,
        capacity_override_reason: Optional[str] = None,
    ) -> StudioQueueReceipt:
        """Queue a portable, atomically published local export bundle."""

        occurrence = self._require_occurrence(occurrence_id)
        self._require_verification(occurrence, action="be exported")
        blob = self._blob_for(occurrence)
        destination_path = Path(destination).expanduser().resolve()
        capacity_path = destination_path.parent
        while not capacity_path.exists() and capacity_path != capacity_path.parent:
            capacity_path = capacity_path.parent
        parameters = {
            "destination": str(destination_path),
            "replay_identity": dict(replay_identity or {}),
            "dataset_identity": dict(dataset_identity or {}),
            "license_metadata": dict(license_metadata or {}),
            "model_card": model_card,
            "model_id": occurrence.model_id,
            "backend": "portable-local",
        }
        spec = OperationSpec(
            operation_type="export",
            input_content_hashes=(blob.content_hash,),
            output_kind="export_bundle",
            output_format="halo-forge-bundle",
            parameters=parameters,
            tool_id="halo-forge-portable-export",
            tool_version=PACKAGE_VERSION,
        )
        requirements: dict[str, Any] = {
            "exclusive_heavy_operation": True,
            "projected_disk_bytes": int(blob.size_bytes) + 16 * 1024**2,
            "projected_ram_bytes": max(0, int(projected_ram_bytes)),
            "output_path": str(capacity_path),
        }
        override = str(capacity_override_reason or "").strip()
        if override:
            requirements["capacity_override_reason"] = override
        return self._queue_operation(
            spec,
            input_occurrence_ids=(occurrence.id,),
            priority=priority,
            max_retries=max_retries,
            resource_requirements=requirements,
        )

    # Explicit long-form name for transports while retaining the concise API.
    queue_export_artifact = queue_export

    def queue_qualification(
        self,
        *,
        occurrence_id: str,
        profile_revision_id: str,
        parent_occurrence_id: Optional[str] = None,
        execution_request: Optional[Mapping[str, Any]] = None,
        priority: int = 0,
        max_retries: int = 1,
    ) -> StudioQueueReceipt:
        occurrence = self._require_occurrence(occurrence_id)
        self._require_verification(occurrence, action="be qualified")
        if parent_occurrence_id:
            parent = self._require_occurrence(parent_occurrence_id)
            self._require_verification(parent, action="be used as a qualification parent")
        if self.catalog.get_qualification_profile_revision(profile_revision_id) is None:
            raise KeyError(f"Unknown qualification profile revision: {profile_revision_id}")
        qualification_id = f"qualification-{uuid.uuid4().hex}"
        work_item_id = f"qualification-work-{uuid.uuid4().hex}"
        launch_spec = {
            "handler": "artifact_studio.execute_work_item",
            "domain_kind": "artifact_qualification",
            "domain_id": qualification_id,
            "artifact_root": str(self.store.root),
            "evaluation_artifact_root": self._evaluation_artifact_root(),
            "occurrence_id": occurrence_id,
            "parent_occurrence_id": parent_occurrence_id,
            "profile_revision_id": profile_revision_id,
            "execution_request": dict(execution_request or {}),
            "resource_requirements": {"exclusive_heavy_operation": True},
        }
        work_item = self.scheduler.enqueue(
            kind="artifact_qualification",
            launch_spec=launch_spec,
            resource_class="accelerator",
            resource_requirements=launch_spec["resource_requirements"],
            domain_kind="artifact_qualification",
            domain_id=qualification_id,
            priority=priority,
            max_retries=max_retries,
            work_item_id=work_item_id,
        )
        try:
            qualification = self.catalog.create_qualification(
                qualification_id=qualification_id,
                profile_revision_id=profile_revision_id,
                occurrence_id=occurrence_id,
                parent_occurrence_id=parent_occurrence_id,
                work_item_id=work_item.id,
            )
        except Exception:
            self.scheduler.cancel(work_item.id)
            raise
        return StudioQueueReceipt(
            domain_kind="artifact_qualification",
            domain_id=qualification.id,
            work_item_id=work_item.id,
            status=qualification.status,
        )

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        operation = self.catalog.get_operation(operation_id)
        if operation is None:
            raise KeyError(f"Unknown artifact operation: {operation_id}")
        value = operation.to_dict()
        value["work_item"] = (
            None
            if not operation.work_item_id
            else _as_dict(self.database.get_work_item(operation.work_item_id))
        )
        return value

    def list_operations(
        self, *, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> dict[str, Any]:
        values = self.catalog.list_operations(status=status, limit=limit, offset=offset)
        return {
            "items": [item.to_dict() for item in values],
            "limit": max(1, min(1000, int(limit))),
            "offset": max(0, int(offset)),
            "has_more": len(values) >= max(1, min(1000, int(limit))),
        }

    def _require_occurrence(self, occurrence_id: str) -> ArtifactOccurrenceRecord:
        occurrence = self.catalog.get_occurrence(occurrence_id)
        if occurrence is None:
            raise KeyError(f"Unknown artifact occurrence: {occurrence_id}")
        return occurrence

    def _blob_for(self, occurrence: ArtifactOccurrenceRecord):
        blob = self.catalog.get_blob(occurrence.blob_id)
        if blob is None:
            raise ArtifactStudioError(f"Artifact occurrence {occurrence.id} has no blob")
        return blob

    def _execute_export_operation(self, operation: ArtifactOperationRecord) -> dict[str, Any]:
        resolved = operation.to_dict()["resolved_spec"]
        inputs = operation.to_dict()["input_occurrence_ids"]
        if len(inputs) != 1:
            raise ArtifactStudioError("A portable export requires exactly one source occurrence")
        parameters = resolved.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ArtifactStudioError("Resolved portable-export parameters are invalid")
        destination = str(parameters.get("destination") or "").strip()
        if not destination:
            raise ArtifactStudioError("Resolved portable export has no destination")
        self.catalog.update_operation(operation.id, status="running")
        exported = self.export_artifact(
            str(inputs[0]),
            destination,
            replay_identity=parameters.get("replay_identity"),
            dataset_identity=parameters.get("dataset_identity"),
            license_metadata=parameters.get("license_metadata"),
            model_card=parameters.get("model_card"),
            _operation_id=operation.id,
        )
        output_occurrence_id = str(exported["artifact"]["occurrence"]["id"])
        result = {
            **exported,
            "output_occurrence_id": output_occurrence_id,
            "reused": bool(exported["bundle"].get("reused", False)),
        }
        self.catalog.update_operation(
            operation.id,
            status="completed",
            output_occurrence_id=output_occurrence_id,
            result=result,
        )
        return result

    def _execute_artifact_operation(self, operation: ArtifactOperationRecord) -> dict[str, Any]:
        if operation.operation_type == "export":
            return self._execute_export_operation(operation)
        self.catalog.update_operation(operation.id, status="running")
        spec = OperationSpec(
            **{
                key: value
                for key, value in operation.to_dict()["resolved_spec"].items()
                if key
                in {
                    "operation_type",
                    "input_content_hashes",
                    "output_kind",
                    "output_format",
                    "parameters",
                    "tool_id",
                    "tool_version",
                    "output_dtype",
                    "output_quantization",
                }
            }
        )
        # JSON deserialization produces a list; OperationSpec's public contract
        # uses a tuple to make input order explicit.
        if not isinstance(spec.input_content_hashes, tuple):
            raise ArtifactStudioError("Resolved operation input identity is invalid")
        completed = self.operations.run(spec)
        if not completed.output_content_hash or not completed.output_location_id:
            raise ArtifactStudioError("Artifact operation completed without a published output")
        filesystem_blob = self.store.get_blob(completed.output_content_hash)
        filesystem_location = self.store.get_location(completed.output_location_id)
        filesystem_occurrence_id = f"artifact-{spec.fingerprint[:24]}"
        filesystem_occurrence = self.store.get_occurrence(filesystem_occurrence_id)
        registration = ArtifactRegistration(
            blob=filesystem_blob,
            location=filesystem_location,
            occurrence=filesystem_occurrence,
            reused_blob=completed.reused,
        )
        input_occurrences = [
            self._require_occurrence(item) for item in operation.to_dict()["input_occurrence_ids"]
        ]
        first = input_occurrences[0]
        output_occurrence = self._sync_registration(
            registration,
            artifact_kind=filesystem_blob.artifact_kind,
            model_id=str(spec.parameters.get("model_id") or first.model_id),
            backend=str(spec.parameters.get("backend") or first.backend),
            tokenizer_revision=first.tokenizer_revision,
            chat_template_hash=first.chat_template_hash,
            metadata={
                "artifact_operation_id": operation.id,
                "operation_evidence": dict(completed.engine_metadata),
            },
        )
        for ordinal, parent in enumerate(input_occurrences):
            base_input_index = spec.parameters.get("base_input_index")
            self.catalog.add_edge(
                child_occurrence_id=output_occurrence.id,
                parent_occurrence_id=parent.id,
                relation=(
                    "base"
                    if base_input_index is not None and ordinal == int(base_input_index)
                    else spec.operation_type
                ),
                ordinal=ordinal,
                operation_id=operation.id,
            )
        result = {
            "operation": completed.to_dict(),
            "output_occurrence_id": output_occurrence.id,
            "reused": completed.reused,
        }
        self.catalog.update_operation(
            operation.id,
            status="completed",
            output_occurrence_id=output_occurrence.id,
            result=result,
        )
        return result

    def _execute_qualification(
        self,
        qualification: ArtifactQualificationRecord,
        launch_spec: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.qualification_executor is None:
            raise UnsupportedArtifactCapability(
                "Qualification execution requires an evidence-producing evaluation executor"
            )
        candidate = self._require_occurrence(qualification.occurrence_id)
        self._require_verification(candidate, action="be qualified")
        if qualification.parent_occurrence_id:
            parent = self._require_occurrence(qualification.parent_occurrence_id)
            self._require_verification(parent, action="be used as a qualification parent")
        self.catalog.update_qualification(qualification.id, status="running")
        result = dict(self.qualification_executor(qualification, launch_spec))
        decision = str(result.get("decision") or "").lower()
        if decision not in {"pass", "warn", "fail"}:
            raise ArtifactStudioError(
                "Qualification executor must return decision=pass, warn, or fail"
            )
        reasons = tuple(str(item) for item in result.get("reasons") or ())
        metrics = dict(result.get("metrics") or {})
        decision_evidence = result.get("decision_evidence")
        if isinstance(decision_evidence, Mapping):
            metrics["decision"] = dict(decision_evidence)
        updated = self.catalog.update_qualification(
            qualification.id,
            status="completed",
            decision=decision,
            reasons=reasons,
            metrics=metrics,
            quality_evaluation_id=result.get("quality_evaluation_id"),
            performance_evaluation_id=result.get("performance_evaluation_id"),
            holdout_evaluation_id=result.get("holdout_evaluation_id"),
        )
        assert updated is not None
        return {"qualification": updated.to_dict()}

    @staticmethod
    def _protections_from_mapping(value: Mapping[str, Any]) -> CleanupProtections:
        keys = (
            "active",
            "pinned",
            "promoted",
            "serving",
            "evaluation_referenced",
            "lineage_required",
            "active_staging",
        )
        return CleanupProtections(
            **{key: frozenset(str(item) for item in value.get(key, ())) for key in keys}
        )

    def _execute_cleanup(
        self,
        plan_id: str,
        launch_spec: Mapping[str, Any],
    ) -> dict[str, Any]:
        plan = self.store.get_cleanup_plan(plan_id)
        if plan.id != plan_id:
            raise ArtifactStudioError("Cleanup plan identity does not match its work link")
        review_note = str(launch_spec.get("review_note") or "").strip()
        extra_value = launch_spec.get("extra_protections") or {}
        if not isinstance(extra_value, Mapping):
            raise ArtifactStudioError("Cleanup work has invalid extra protections")
        result = self.apply_cleanup(
            plan_id,
            review_note=review_note,
            extra_protections=self._protections_from_mapping(extra_value),
        )
        return {
            "plan_id": plan.id,
            "review_note": review_note,
            "cleanup": result.to_dict(),
        }

    def _execute_serving(
        self,
        profile_revision_id: str,
        launch_spec: Mapping[str, Any],
    ) -> dict[str, Any]:
        profile_value = launch_spec.get("profile_revision")
        if not isinstance(profile_value, Mapping):
            raise ArtifactStudioError("Managed serving work has no resolved profile revision")
        profile = dict(profile_value)
        if str(profile.get("id") or "") != profile_revision_id:
            raise ArtifactStudioError(
                "Managed serving profile identity does not match its work link"
            )
        occurrence_id = str(launch_spec.get("occurrence_id") or "")
        if not occurrence_id or occurrence_id != str(profile.get("occurrence_id") or ""):
            raise ArtifactStudioError("Managed serving occurrence identity is invalid")
        occurrence = self._require_occurrence(occurrence_id)
        blob = self._blob_for(occurrence)
        self._require_verification(occurrence, action="start managed serving")

        serving_id = str(launch_spec.get("serving_id") or "").strip()
        if not serving_id:
            raise ArtifactStudioError("Managed serving work has no serving_id")
        start_process = bool(launch_spec.get("start_process", True))
        if start_process and self.serving_starter is None:
            raise UnsupportedArtifactCapability(
                "Managed serving start requires a launcher that reports real process identity"
            )
        base_metadata = {
            "profile_revision_id": profile_revision_id,
            "occurrence_id": occurrence.id,
            "artifact_hash": blob.content_hash,
            "state": "starting" if start_process else "reserved",
        }
        lease = self.scheduler.start_serving(
            serving_id=serving_id,
            metadata=base_metadata,
        )
        if lease is None:
            raise ArtifactStudioError(
                "Managed serving is waiting for the workstation accelerator/memory lease"
            )
        if not start_process:
            return {
                "state": "reserved",
                "serving_id": serving_id,
                "profile_revision": profile,
                "lease": lease.to_dict(),
            }

        try:
            assert self.serving_starter is not None
            started = dict(self.serving_starter(profile, occurrence, launch_spec))
            reported_state = str(started.get("state") or "").strip().lower()
            if reported_state not in {"running", "serving"}:
                raise ArtifactStudioError(
                    "Managed serving launcher must report state=running or state=serving"
                )
            process_id = started.get("process_id", started.get("pid"))
            process_started_at = started.get("process_started_at", started.get("pid_started_at"))
            if isinstance(process_id, bool) or not isinstance(process_id, int) or process_id < 1:
                raise ArtifactStudioError(
                    "Managed serving launcher must report a positive integer process_id"
                )
            if (
                isinstance(process_started_at, bool)
                or not isinstance(process_started_at, (int, float))
                or float(process_started_at) <= 0
            ):
                raise ArtifactStudioError("Managed serving launcher must report process_started_at")
            if not self.scheduler.process_probe(process_id, float(process_started_at)):
                raise ArtifactStudioError(
                    "Managed serving launcher reported a process identity that is not live"
                )
            heartbeat = self.heartbeat_serving(
                serving_id,
                process_id=process_id,
                process_started_at=float(process_started_at),
                metadata={
                    **base_metadata,
                    "state": "serving",
                    "launcher_result": started,
                },
            )
            return {
                "state": "serving",
                "serving_id": serving_id,
                "profile_revision": profile,
                "process_identity": {
                    "process_id": process_id,
                    "process_started_at": float(process_started_at),
                },
                "launcher_result": started,
                "lease": heartbeat,
            }
        except Exception:
            self.release_serving(serving_id)
            raise

    def execute_work_item(self, work_item_id: str) -> dict[str, Any]:
        """In-process worker hook for Artifact Studio work kinds.

        The method claims queued work and owns its terminal transition. A
        supervised worker may call it directly; unsupported capabilities fail
        the durable item rather than being reported as completed.
        """

        item = self.database.get_work_item(work_item_id)
        if item is None:
            raise KeyError(f"Unknown work item: {work_item_id}")
        if item.status == "completed":
            return {"work_item": item.to_dict(), "result": item.result, "reused": True}
        if item.status == "queued":
            queued_domain_kind = str(item.domain_kind or item.launch_spec.get("domain_kind") or "")
            if (
                queued_domain_kind == "artifact_serving"
                and self.database.get_resource_lease("accelerator") is not None
            ):
                raise ArtifactStudioError(
                    "Managed serving remains queued while the workstation resource is busy"
                )
            item = self.scheduler.claim(work_item_id=work_item_id)
            if item is None:
                raise ArtifactStudioError("Work item could not acquire its workstation resource")
        if item.status != "running" or not item.claim_token:
            raise ArtifactStudioError(
                f"Work item {work_item_id} is not executable from status {item.status!r}"
            )
        launch_domain_kind = str(item.launch_spec.get("domain_kind") or "")
        launch_domain_id = str(item.launch_spec.get("domain_id") or "")
        domain_kind = str(item.domain_kind or launch_domain_kind)
        domain_id = str(item.domain_id or launch_domain_id)
        try:
            if item.domain_kind and launch_domain_kind and item.domain_kind != launch_domain_kind:
                raise ArtifactStudioError("Work-item and launch domain kinds do not match")
            if item.domain_id and launch_domain_id and item.domain_id != launch_domain_id:
                raise ArtifactStudioError("Work-item and launch domain identities do not match")
            if domain_kind == "artifact_operation":
                operation = self.catalog.get_operation(domain_id)
                if operation is None:
                    raise ArtifactStudioError(f"Missing artifact operation {domain_id}")
                result = self._execute_artifact_operation(operation)
            elif domain_kind == "artifact_qualification":
                qualification = self.catalog.get_qualification(domain_id)
                if qualification is None:
                    raise ArtifactStudioError(f"Missing artifact qualification {domain_id}")
                result = self._execute_qualification(qualification, item.launch_spec)
            elif domain_kind == "artifact_cleanup":
                result = self._execute_cleanup(domain_id, item.launch_spec)
            elif domain_kind == "artifact_serving":
                result = self._execute_serving(domain_id, item.launch_spec)
            else:
                raise UnsupportedArtifactCapability(
                    f"Artifact Studio cannot execute domain kind {domain_kind!r}"
                )
            finished = self.scheduler.complete(item, result=result)
            return {
                "work_item": _as_dict(finished or item),
                "result": result,
                "reused": False,
            }
        except Exception as exc:
            if domain_kind == "artifact_operation":
                self.catalog.update_operation(domain_id, status="failed", error=str(exc))
            elif domain_kind == "artifact_qualification":
                self.catalog.update_qualification(domain_id, status="failed", reasons=(str(exc),))
            elif domain_kind == "artifact_serving":
                serving_id = str(item.launch_spec.get("serving_id") or "")
                if serving_id:
                    self.release_serving(serving_id)
            self.scheduler.fail(item, error=str(exc))
            raise

    # -- qualification-gated aliases, serving, and export -------------

    def promote(
        self,
        occurrence_id: str,
        target_alias: str,
        *,
        override_note: Optional[str] = None,
    ) -> dict[str, Any]:
        occurrence = self._require_occurrence(occurrence_id)
        self._require_verification(occurrence, action="be promoted")
        target = target_alias.strip().lower()
        if target not in {"candidate", "approved"}:
            raise ValueError("target_alias must be candidate or approved")
        qualifications = self.catalog.list_qualifications(occurrence_id=occurrence.id, limit=1)
        reasons: list[str] = []
        if not qualifications or qualifications[0].status != "completed":
            reasons.append("no completed qualification")
        else:
            qualification = qualifications[0]
            decision_evidence = qualification.to_dict()["metrics"].get("decision", {})
            if isinstance(decision_evidence, Mapping) and decision_evidence:
                required = ["development", "operational"]
                if target == "approved":
                    profile = self.catalog.get_qualification_profile_revision(
                        qualification.profile_revision_id
                    )
                    if profile and profile.holdout_suite_revision_id:
                        required.append("holdout")
                for stage in required:
                    stage_status = (decision_evidence.get(stage) or {}).get("status")
                    if stage_status != "pass":
                        reasons.append(f"{stage} gate is {stage_status or 'missing'}")
            elif qualification.decision != "pass":
                reasons.append(f"qualification decision is {qualification.decision or 'missing'}")
            if target == "approved":
                profile = self.catalog.get_qualification_profile_revision(
                    qualification.profile_revision_id
                )
                if (
                    profile
                    and profile.holdout_suite_revision_id
                    and not qualification.holdout_evaluation_id
                ):
                    reasons.append("holdout confirmation is missing")
        note = str(override_note or "").strip()
        if reasons and not note:
            raise PromotionBlocked("; ".join(reasons))
        alias = self.catalog.set_alias(
            target,
            occurrence.id,
            override_reason=note if reasons else None,
        )
        return {
            **alias,
            "overridden": bool(reasons),
            "reasons": reasons,
            "qualification_id": qualifications[0].id if qualifications else None,
        }

    def _create_serving_profile(
        self,
        occurrence_id: str,
        *,
        name: str,
        backend: str,
        endpoint_settings: Optional[Mapping[str, Any]],
        generation_settings: Optional[Mapping[str, Any]],
        resource_requirements: Optional[Mapping[str, Any]],
        verify_artifact: bool = False,
    ) -> tuple[ArtifactOccurrenceRecord, ServingProfileRevisionRecord]:
        occurrence = self._require_occurrence(occurrence_id)
        resolved_name = name.strip()
        resolved_backend = backend.strip().lower()
        if not resolved_name:
            raise ValueError("Serving profile name is required")
        if not resolved_backend:
            raise ValueError("Serving backend is required")
        blob = self._blob_for(occurrence)
        if verify_artifact:
            self._require_verification(occurrence, action="be reserved for serving")
        profile_definition = {
            "occurrence_id": occurrence.id,
            "artifact_hash": blob.content_hash,
            "backend": resolved_backend,
            "endpoint_settings": dict(endpoint_settings or {}),
            "generation_settings": dict(generation_settings or {}),
            "resource_requirements": dict(resource_requirements or {}),
            "chat_template_hash": occurrence.chat_template_hash,
        }
        profile = self.catalog.create_serving_profile_revision(
            name=resolved_name,
            content_hash=fingerprint(profile_definition),
            occurrence_id=occurrence.id,
            backend=resolved_backend,
            endpoint_settings=endpoint_settings or {},
            generation_settings=generation_settings or {},
            resource_requirements=resource_requirements or {},
            chat_template_hash=occurrence.chat_template_hash,
        )
        return occurrence, profile

    def queue_serving(
        self,
        occurrence_id: str,
        *,
        name: str,
        backend: str,
        endpoint_settings: Optional[Mapping[str, Any]] = None,
        generation_settings: Optional[Mapping[str, Any]] = None,
        resource_requirements: Optional[Mapping[str, Any]] = None,
        serving_id: Optional[str] = None,
        start_process: bool = True,
        priority: int = 0,
        max_retries: int = 1,
    ) -> StudioQueueReceipt:
        """Queue a managed serving reservation or a real launcher-backed start.

        The queue item itself is lightweight. During execution it atomically
        acquires the retained serving lease before any launcher is called, so
        it cannot overlap an accelerator work lease.
        """

        occurrence, profile = self._create_serving_profile(
            occurrence_id,
            name=name,
            backend=backend,
            endpoint_settings=endpoint_settings,
            generation_settings=generation_settings,
            resource_requirements=resource_requirements,
            verify_artifact=True,
        )
        identifier = str(serving_id or f"serving-{uuid.uuid4().hex}").strip()
        if not identifier:
            raise ValueError("serving_id cannot be empty")
        work_item_id = f"serving-work-{uuid.uuid4().hex}"
        resolved_requirements = {
            **dict(resource_requirements or {}),
            "exclusive_heavy_operation": True,
            "lease_type": "serving",
            "output_path": str(self.store.root),
        }
        launch_spec = {
            "handler": "artifact_studio.execute_work_item",
            "domain_kind": "artifact_serving",
            "domain_id": profile.id,
            "artifact_root": str(self.store.root),
            "serving_id": identifier,
            "occurrence_id": occurrence.id,
            "profile_revision": profile.to_dict(),
            "start_process": bool(start_process),
            "resource_requirements": resolved_requirements,
        }
        work_item = self.scheduler.enqueue(
            kind="artifact_serving",
            launch_spec=launch_spec,
            # The retained serving lease is acquired by the handler. Claiming
            # an ordinary accelerator work lease here would make that atomic
            # reservation impossible while the item is running.
            resource_class="none",
            resource_requirements=resolved_requirements,
            domain_kind="artifact_serving",
            domain_id=profile.id,
            priority=priority,
            max_retries=max_retries,
            work_item_id=work_item_id,
        )
        return StudioQueueReceipt(
            domain_kind="artifact_serving",
            domain_id=profile.id,
            work_item_id=work_item.id,
            status=work_item.status,
        )

    queue_managed_serving = queue_serving

    def reserve_serving(
        self,
        occurrence_id: str,
        *,
        name: str,
        backend: str,
        endpoint_settings: Optional[Mapping[str, Any]] = None,
        generation_settings: Optional[Mapping[str, Any]] = None,
        resource_requirements: Optional[Mapping[str, Any]] = None,
        serving_id: Optional[str] = None,
    ) -> ServingReservation:
        occurrence, profile = self._create_serving_profile(
            occurrence_id,
            name=name,
            backend=backend,
            endpoint_settings=endpoint_settings,
            generation_settings=generation_settings,
            resource_requirements=resource_requirements,
            verify_artifact=True,
        )
        blob = self._blob_for(occurrence)
        identifier = serving_id or f"serving-{uuid.uuid4().hex}"
        lease = self.scheduler.start_serving(
            serving_id=identifier,
            metadata={
                "profile_revision_id": profile.id,
                "occurrence_id": occurrence.id,
                "artifact_hash": blob.content_hash,
                "state": "reserved",
            },
        )
        if lease is None:
            return ServingReservation(
                serving_id=identifier,
                profile_revision=profile.to_dict(),
                state="blocked",
                reason="workstation accelerator/memory resource is busy",
            )
        return ServingReservation(
            serving_id=identifier,
            profile_revision=profile.to_dict(),
            state="reserved",
            lease=lease.to_dict(),
        )

    def release_serving(self, serving_id: str) -> bool:
        return self.scheduler.stop_serving(serving_id=serving_id)

    def heartbeat_serving(
        self,
        serving_id: str,
        *,
        process_id: Optional[int] = None,
        process_started_at: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Attach/refresh the actual server process identity after reservation."""

        lease = self.database.heartbeat_serving_lease(
            holder_id=serving_id,
            holder_pid=process_id,
            holder_pid_started_at=process_started_at,
            metadata=metadata,
        )
        if lease is None:
            raise KeyError(f"Unknown serving reservation: {serving_id}")
        return lease.to_dict()

    def export_artifact(
        self,
        occurrence_id: str,
        destination: Path | str,
        *,
        replay_identity: Optional[Mapping[str, Any]] = None,
        dataset_identity: Optional[Mapping[str, Any]] = None,
        license_metadata: Optional[Mapping[str, Any]] = None,
        model_card: Optional[str] = None,
        _operation_id: Optional[str] = None,
    ) -> dict[str, Any]:
        occurrence = self._require_occurrence(occurrence_id)
        blob = self._blob_for(occurrence)
        verification = self._require_verification(occurrence, action="be exported")
        qualifications = self.catalog.list_qualifications(occurrence_id=occurrence.id, limit=1)
        qualification = qualifications[0].to_dict() if qualifications else {}
        bundle: PortableExportBundle = self.store.export_bundle(
            blob.content_hash,
            destination,
            replay_identity=replay_identity,
            dataset_identity=dataset_identity,
            qualification=qualification,
            verification=verification,
            license_metadata=license_metadata,
            model_card=model_card,
        )
        registration = self.store.import_artifact(
            bundle.path,
            artifact_kind="export_bundle",
            artifact_format="halo-forge-bundle",
            managed=False,
            metadata={"source_occurrence_id": occurrence.id, "bundle_id": bundle.id},
        )
        output = self._sync_registration(
            registration,
            model_id=f"{occurrence.model_id}-bundle",
            backend="portable-local",
            artifact_kind="export_bundle",
            metadata={"source_occurrence_id": occurrence.id, "bundle_id": bundle.id},
        )
        self.catalog.add_edge(
            child_occurrence_id=output.id,
            parent_occurrence_id=occurrence.id,
            relation="export",
            operation_id=_operation_id,
        )
        return {"bundle": bundle.to_dict(), "artifact": self._artifact_view(output)}

    # -- storage and reviewed cleanup ----------------------------------

    def storage_inventory(self) -> dict[str, Any]:
        return self.store.inventory().to_dict()

    @staticmethod
    def _merge_protections(
        first: CleanupProtections, second: CleanupProtections
    ) -> CleanupProtections:
        return CleanupProtections(
            active=first.active | second.active,
            pinned=first.pinned | second.pinned,
            promoted=first.promoted | second.promoted,
            serving=first.serving | second.serving,
            evaluation_referenced=(first.evaluation_referenced | second.evaluation_referenced),
            lineage_required=first.lineage_required | second.lineage_required,
            active_staging=first.active_staging | second.active_staging,
        )

    def cleanup_protections(
        self, extra: CleanupProtections = CleanupProtections()
    ) -> CleanupProtections:
        occurrences = self._all_occurrences()
        by_id = {item.id: self._blob_for(item).content_hash for item in occurrences}
        known_hashes = set(by_id.values())
        by_run_id = {item.run_id: by_id[item.id] for item in occurrences if item.run_id}
        by_legacy_id = {
            item.legacy_model_artifact_id: by_id[item.id]
            for item in occurrences
            if item.legacy_model_artifact_id
        }
        by_group_id: dict[str, set[str]] = {}
        by_trial_id: dict[str, set[str]] = {}
        by_segment_id: dict[str, set[str]] = {}
        for occurrence in occurrences:
            content_hash = by_id[occurrence.id]
            for key, target in (
                (occurrence.run_group_id, by_group_id),
                (occurrence.trial_id, by_trial_id),
                (occurrence.trial_segment_id, by_segment_id),
            ):
                if key:
                    target.setdefault(str(key), set()).add(content_hash)
        by_path: dict[str, str] = {}
        for occurrence in occurrences:
            blob_hash = by_id[occurrence.id]
            for location in self.catalog.list_locations(occurrence.blob_id):
                try:
                    resolved = str(Path(location.path).expanduser().resolve())
                except OSError:
                    resolved = str(Path(location.path).expanduser())
                by_path[resolved] = blob_hash
        pinned = {by_id[item.id] for item in occurrences if item.pinned}
        promoted = {
            by_id[item.id]
            for item in occurrences
            if set(self.catalog.aliases_for(item.id)) & {"candidate", "approved"}
        }
        evaluated = {
            by_id[item.id]
            for item in occurrences
            if self.catalog.list_qualifications(occurrence_id=item.id, limit=1)
        }
        # Standalone evaluations are just as immutable and cleanup-sensitive as
        # qualification evaluations. Resolve their persisted subject envelope
        # against content hashes, occurrence IDs, run IDs, and known locations.
        for evaluation in self.database.list_evaluations(limit=1_000_000):
            subject = evaluation.request.get("subject") or evaluation.request.get("subject_input")
            subject = dict(subject) if isinstance(subject, Mapping) else {}
            candidates = {
                str(subject.get("content_hash") or ""),
                str(subject.get("artifact_id") or subject.get("occurrence_id") or ""),
                str(subject.get("ref") or subject.get("path") or ""),
                str(subject.get("resolved_path") or ""),
                str(evaluation.subject_ref or ""),
            }
            for candidate in candidates - {""}:
                if candidate in known_hashes:
                    evaluated.add(candidate)
                if candidate in by_id:
                    evaluated.add(by_id[candidate])
                if candidate in by_run_id:
                    evaluated.add(by_run_id[candidate])
                try:
                    resolved = str(Path(candidate).expanduser().resolve())
                except OSError:
                    resolved = str(Path(candidate).expanduser())
                if resolved in by_path:
                    evaluated.add(by_path[resolved])
        for exposure in self.database.list_exposures():
            if exposure.model_artifact_id in by_id:
                evaluated.add(by_id[exposure.model_artifact_id])
            if exposure.model_artifact_id in by_legacy_id:
                evaluated.add(by_legacy_id[exposure.model_artifact_id])

        # Adaptive gates, reviewed research decisions, and their immutable
        # evidence bundles are durable scientific references. They must keep
        # their checkpoint bytes available even if the trainer-owned source
        # directory is later rotated or a cleanup plan is prepared.
        lineage_required: set[str] = set()
        for gate in self.database.list_checkpoint_gate_decisions(limit=1_000_000):
            if gate.checkpoint_occurrence_id in by_id:
                lineage_required.add(by_id[gate.checkpoint_occurrence_id])
            if gate.trial_segment_id in by_segment_id:
                lineage_required.update(by_segment_id[gate.trial_segment_id])

        def protect_adaptive_reference(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    text = str(child) if isinstance(child, (str, int)) else ""
                    if key in {"artifact_id", "occurrence_id", "checkpoint_occurrence_id"}:
                        if text in by_id:
                            lineage_required.add(by_id[text])
                        if text in by_legacy_id:
                            lineage_required.add(by_legacy_id[text])
                    elif key in {"model_artifact_id", "checkpoint_artifact_id"}:
                        if text in by_legacy_id:
                            lineage_required.add(by_legacy_id[text])
                    elif key == "run_id" and text in by_run_id:
                        lineage_required.add(by_run_id[text])
                    elif key == "run_group_id" and text in by_group_id:
                        lineage_required.update(by_group_id[text])
                    elif key in {"trial_id", "subject_id"} and text in by_trial_id:
                        lineage_required.update(by_trial_id[text])
                    elif key in {"trial_segment_id", "segment_id"} and text in by_segment_id:
                        lineage_required.update(by_segment_id[text])
                    elif key == "content_hash" and text in known_hashes:
                        lineage_required.add(text)
                    elif key in {"path", "ref", "resolved_path"} and text:
                        try:
                            resolved = str(Path(text).expanduser().resolve())
                        except OSError:
                            resolved = str(Path(text).expanduser())
                        if resolved in by_path:
                            lineage_required.add(by_path[resolved])
                    protect_adaptive_reference(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    protect_adaptive_reference(child)

        for decision in self.database.list_research_decisions(limit=1_000_000):
            snapshot = self.database.get_cohort_analysis_snapshot(decision.analysis_snapshot_id)
            if snapshot is not None and snapshot.run_group_id in by_group_id:
                lineage_required.update(by_group_id[snapshot.run_group_id])
            protect_adaptive_reference(decision.selected_subject)
            protect_adaptive_reference(decision.rejected_subjects)
            protect_adaptive_reference(decision.exclusions)
            protect_adaptive_reference(decision.fork_spec)
        for bundle in self.database.list_evidence_bundles(limit=1_000_000):
            if bundle.status in {"failed", "corrupt"}:
                continue
            snapshot = self.database.get_cohort_analysis_snapshot(bundle.analysis_snapshot_id)
            if snapshot is not None and snapshot.run_group_id in by_group_id:
                lineage_required.update(by_group_id[snapshot.run_group_id])
            protect_adaptive_reference(bundle.request)
            protect_adaptive_reference(bundle.manifest)
        active: set[str] = set()
        serving: set[str] = set()
        # Serving profiles are durable deployment references even while their
        # process is stopped. Keep those artifacts until the profile is
        # explicitly removed, in addition to protecting every active lease.
        serving_rows = self.database._conn.execute(
            "SELECT DISTINCT occurrence_id FROM serving_profile_revisions"
        ).fetchall()
        for row in serving_rows:
            occurrence_id = str(row["occurrence_id"])
            if occurrence_id in by_id:
                serving.add(by_id[occurrence_id])
        for lease in self.database.list_resource_leases():
            if lease.holder_type != "serving":
                continue
            serving_occurrence_id = lease.metadata.get("occurrence_id")
            if serving_occurrence_id in by_id:
                serving.add(by_id[serving_occurrence_id])
        for work in self.database.list_work_items(
            statuses=("queued", "blocked", "running", "needs_reconciliation")
        ):
            spec = work.launch_spec
            for occurrence_id in spec.get("input_occurrence_ids") or ():
                if occurrence_id in by_id:
                    active.add(by_id[occurrence_id])
            occurrence_id = spec.get("occurrence_id")
            if occurrence_id in by_id:
                active.add(by_id[occurrence_id])
            active_run_id = str(spec.get("run_id") or "")
            if active_run_id in by_run_id:
                active.add(by_run_id[active_run_id])
        derived = CleanupProtections(
            active=frozenset(active),
            pinned=frozenset(pinned),
            promoted=frozenset(promoted),
            serving=frozenset(serving),
            evaluation_referenced=frozenset(evaluated),
            lineage_required=frozenset(lineage_required),
        )
        return self._merge_protections(derived, extra)

    def preview_cleanup(
        self,
        *,
        extra_protections: CleanupProtections = CleanupProtections(),
    ) -> CleanupPlan:
        return self.store.preview_cleanup(protections=self.cleanup_protections(extra_protections))

    def queue_cleanup(
        self,
        plan_id: str,
        *,
        review_note: str,
        extra_protections: CleanupProtections = CleanupProtections(),
        priority: int = 0,
        max_retries: int = 1,
        projected_ram_bytes: int = 0,
        capacity_override_reason: Optional[str] = None,
    ) -> StudioQueueReceipt:
        """Queue execution of an already reviewed immutable cleanup plan."""

        note = str(review_note or "").strip()
        if not note:
            raise ValueError("A non-empty review_note is required to queue cleanup")
        plan = self.store.get_cleanup_plan(plan_id)
        existing = self._find_reusable_domain_work(
            domain_kind="artifact_cleanup",
            domain_id=plan.id,
            kind="artifact_cleanup",
        )
        if existing is not None:
            return StudioQueueReceipt(
                domain_kind="artifact_cleanup",
                domain_id=plan.id,
                work_item_id=existing.id,
                status=existing.status,
                reused=True,
            )
        requirements: dict[str, Any] = {
            "exclusive_heavy_operation": True,
            "projected_disk_bytes": 0,
            "projected_ram_bytes": max(0, int(projected_ram_bytes)),
            "output_path": str(self.store.root),
            "estimated_reclaimable_bytes": plan.reclaimable_bytes,
        }
        override = str(capacity_override_reason or "").strip()
        if override:
            requirements["capacity_override_reason"] = override
        work_item_id = f"cleanup-work-{uuid.uuid4().hex}"
        launch_spec = {
            "handler": "artifact_studio.execute_work_item",
            "domain_kind": "artifact_cleanup",
            "domain_id": plan.id,
            "artifact_root": str(self.store.root),
            "plan_id": plan.id,
            "review_note": note,
            "candidate_content_hashes": [item.identifier for item in plan.candidates],
            "extra_protections": extra_protections.to_dict(),
            "resource_requirements": requirements,
        }
        work_item = self.scheduler.enqueue(
            kind="artifact_cleanup",
            launch_spec=launch_spec,
            resource_class="accelerator",
            resource_requirements=requirements,
            domain_kind="artifact_cleanup",
            domain_id=plan.id,
            priority=priority,
            max_retries=max_retries,
            work_item_id=work_item_id,
        )
        return StudioQueueReceipt(
            domain_kind="artifact_cleanup",
            domain_id=plan.id,
            work_item_id=work_item.id,
            status=work_item.status,
        )

    queue_reviewed_cleanup = queue_cleanup

    def apply_cleanup(
        self,
        plan_id: str,
        *,
        review_note: str,
        extra_protections: CleanupProtections = CleanupProtections(),
    ) -> CleanupResult:
        return self.store.trash_cleanup(
            plan_id,
            review_note=review_note,
            current_protections=self.cleanup_protections(extra_protections),
        )

    def restore_artifact(self, content_hash: str) -> dict[str, Any]:
        path = self.store.restore(content_hash)
        blob = self.catalog.find_blob(content_hash)
        if blob is None:
            raise ArtifactStudioError(
                f"Restored filesystem blob {content_hash} is absent from the catalog"
            )
        filesystem_location = next(
            (
                item
                for item in self.store.list_locations(content_hash=content_hash)
                if item.managed and item.path == path
            ),
            None,
        )
        self.catalog.add_location(
            blob_id=blob.id,
            path=path,
            storage_mode="managed",
            state="available",
            size_bytes=blob.size_bytes,
            location_id=filesystem_location.id if filesystem_location else None,
            metadata=(
                {"filesystem_location": filesystem_location.to_dict()}
                if filesystem_location
                else {}
            ),
        )
        return {"content_hash": content_hash, "path": path, "state": "available"}

    def purge_trash(
        self, *, extra_protections: CleanupProtections = CleanupProtections()
    ) -> dict[str, Any]:
        return self.store.purge_trash(protections=self.cleanup_protections(extra_protections))


__all__ = ["ArtifactStudioService", "QualificationExecutor", "ServingStarter"]
