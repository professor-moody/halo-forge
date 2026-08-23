"""Database-backed lifecycle service for immutable corpus extractions."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import unquote, urlparse

from halo_forge.own_data.inspection import fingerprint_path

from .extractors import CorpusExtractionCancelled
from .models import (
    CORPUS_EXTRACTOR_VERSION,
    CorpusBundle,
    CorpusExtractionConfig,
    CorpusExtractionResult,
    canonical_json,
    sha256_bytes,
)
from .pipeline import default_corpus_root, extract_source
from .storage import CorpusBundleIntegrityError, CorpusBundleStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class _ResolvedSource:
    path: Path
    source_kind: str
    source_uri: str
    source_fingerprint: str
    size_bytes: int
    file_count: int
    import_id: Optional[str] = None
    source_id: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)


def _local_path(value: str) -> Path:
    parsed = urlparse(str(value))
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).expanduser()
    if parsed.scheme and len(parsed.scheme) > 1:
        raise ValueError(f"corpus extraction requires a local source path, not {parsed.scheme!r}")
    return Path(str(value)).expanduser()


class CorpusExtractionService:
    """Launch, execute, inspect, verify, cancel, and retry corpus extraction."""

    def __init__(
        self,
        database: Any,
        *,
        root: Path | str | None = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        self.db = database
        self.root = Path(root or default_corpus_root()).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = CorpusBundleStore(self.root)
        self.scheduler = scheduler

    def _resolve_source(
        self,
        path: Path | str | None,
        *,
        import_id: Optional[str],
        source_id: Optional[str],
    ) -> _ResolvedSource:
        import_record = self.db.get_dataset_import(import_id) if import_id is not None else None
        if import_id is not None and import_record is None:
            raise KeyError(import_id)
        source_record = self.db.get_dataset_source(source_id) if source_id is not None else None
        if source_id is not None and source_record is None:
            raise KeyError(source_id)

        selected: Optional[Path] = _local_path(str(path)) if path is not None else None
        source_kind = "path"
        source_uri: Optional[str] = str(path) if path is not None else None
        provenance: Dict[str, Any] = {}
        if import_record is not None:
            source_kind = f"dataset_import:{import_record.source_kind}"
            source_uri = str(import_record.source_uri or source_uri or "")
            provenance["dataset_import"] = {
                "id": import_record.id,
                "source_kind": import_record.source_kind,
                "catalog_fingerprint": import_record.fingerprint,
                "resolved_revision": import_record.resolved_revision,
            }
            if selected is None:
                candidate = (
                    import_record.managed_source_path
                    or import_record.staging_path
                    or import_record.source_uri
                )
                if candidate:
                    selected = _local_path(str(candidate))
        if source_record is not None:
            source_kind = f"dataset_source:{source_record.kind}"
            source_uri = str(source_record.uri or source_uri or "")
            provenance["dataset_source"] = {
                "id": source_record.id,
                "dataset_id": source_record.dataset_id,
                "kind": source_record.kind,
                "catalog_fingerprint": source_record.fingerprint,
            }
            if selected is None:
                selected = _local_path(source_record.uri)
        if selected is None:
            raise ValueError("path, import_id, or source_id is required")
        if selected.is_symlink():
            raise ValueError("symbolic-link corpus sources are not accepted")
        resolved = selected.resolve()
        fingerprint, size_bytes, file_count = fingerprint_path(resolved)
        return _ResolvedSource(
            path=resolved,
            source_kind=source_kind,
            source_uri=source_uri or str(resolved),
            source_fingerprint=fingerprint,
            size_bytes=int(size_bytes),
            file_count=int(file_count),
            import_id=import_id,
            source_id=source_id,
            provenance=provenance,
        )

    @staticmethod
    def _reuse_key(source_fingerprint: str, config_hash: str) -> str:
        return sha256_bytes(
            canonical_json(
                {
                    "source_fingerprint": source_fingerprint,
                    "extractor_version": CORPUS_EXTRACTOR_VERSION,
                    "config_hash": config_hash,
                }
            ).encode("utf-8")
        )

    def _enqueue(self, record: Any, source: _ResolvedSource) -> Any:
        if self.scheduler is None:
            return None
        work = self.scheduler.enqueue(
            kind="document_extraction",
            launch_spec={
                "handler": "corpus_lab.extract_source",
                "extraction_id": record.id,
                "source_path": str(source.path),
                "source_uri": source.source_uri,
                "source_kind": source.source_kind,
                "corpus_root": str(self.root),
                "config": copy.deepcopy(record.config),
            },
            resource_class="cpu",
            resource_requirements={
                "output_path": str(self.root),
                "projected_disk_bytes": max(0, int(source.size_bytes) * 2),
                "capacity_preflight": True,
            },
            domain_kind="document_extraction",
            domain_id=record.id,
            max_retries=2,
        )
        return self.db.update_document_extraction(record.id, work_item_id=work.id) or record

    def launch(
        self,
        path: Path | str | None = None,
        *,
        import_id: Optional[str] = None,
        source_id: Optional[str] = None,
        config: CorpusExtractionConfig | Mapping[str, Any] | None = None,
        synchronous: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Create/reuse an extraction and either execute or enqueue it."""

        source = self._resolve_source(path, import_id=import_id, source_id=source_id)
        resolved_config = CorpusExtractionConfig.from_value(config)
        reuse_key = self._reuse_key(source.source_fingerprint, resolved_config.fingerprint)
        existing = self.db.find_document_extraction(reuse_key=reuse_key)
        if existing is not None:
            if existing.status == "completed":
                return {
                    "extraction": self.status(existing.id),
                    "work_item_id": existing.work_item_id,
                    "reused": True,
                }
            if existing.status in {"queued", "running"}:
                should_execute = (
                    self.scheduler is None if synchronous is None else bool(synchronous)
                )
                if should_execute:
                    result = self.execute(existing.id)
                    value = result.to_dict()
                    value["work_item_id"] = existing.work_item_id
                    value["reused"] = result.bundle.reused
                    return value
                return {
                    "extraction": self.status(existing.id),
                    "work_item_id": existing.work_item_id,
                    "reused": True,
                }
            return {
                "extraction": self.status(existing.id),
                "work_item_id": existing.work_item_id,
                "reused": True,
                "retry_required": True,
            }

        record = self.db.create_document_extraction(
            source_kind=source.source_kind,
            source_uri=source.source_uri,
            source_fingerprint=source.source_fingerprint,
            extractor_version=CORPUS_EXTRACTOR_VERSION,
            config_hash=resolved_config.fingerprint,
            config=resolved_config.to_dict(),
            reuse_key=reuse_key,
            import_id=import_id,
            source_id=source_id,
            provenance={
                **source.provenance,
                "source_path": str(source.path),
                "source_size_bytes": source.size_bytes,
                "source_file_count": source.file_count,
            },
        )
        should_execute = self.scheduler is None if synchronous is None else bool(synchronous)
        if should_execute:
            result = self.execute(record.id)
            value = result.to_dict()
            value["work_item_id"] = record.work_item_id
            value["reused"] = result.bundle.reused
            return value
        if self.scheduler is None:
            return {
                "extraction": record.to_dict(),
                "work_item_id": None,
                "reused": False,
            }
        queued = self._enqueue(record, source)
        return {
            "extraction": queued.to_dict(),
            "work_item_id": queued.work_item_id,
            "reused": False,
        }

    launch_extraction = launch

    def _record_source_path(self, record: Any) -> Path:
        value = str(record.provenance.get("source_path") or "").strip()
        if not value:
            value = record.source_uri
        return _local_path(value).resolve()

    def _cancel_check(self, record: Any) -> None:
        if not record.work_item_id:
            current = self.db.get_document_extraction(record.id)
            if current is not None and current.status == "cancelled":
                raise CorpusExtractionCancelled("corpus extraction cancellation requested")
            return
        work = self.db.get_work_item(record.work_item_id)
        if work is None:
            return
        if work.cancel_requested or work.status in {
            "cancelled",
            "failed",
            "interrupted",
            "needs_reconciliation",
        }:
            raise CorpusExtractionCancelled("corpus extraction cancellation requested")

    def _progress(self, record: Any, processed: int, total: int) -> None:
        if self.scheduler is None or not record.work_item_id:
            return
        work = self.db.get_work_item(record.work_item_id)
        if work is None or not work.claim_token:
            return
        self.scheduler.heartbeat(
            work,
            stage="extracting_documents",
            progress={
                "processed_files": int(processed),
                "total_files": int(total),
            },
        )

    def execute(self, extraction_id: str) -> CorpusExtractionResult:
        """Execute one queued extraction and atomically seal its catalog rows."""

        record = self.db.get_document_extraction(extraction_id)
        if record is None:
            raise KeyError(extraction_id)
        if record.status == "completed":
            return self._load_result(record)
        if record.status not in {"queued", "running"}:
            raise ValueError(
                f"document extraction in {record.status!r} state must be retried first"
            )
        record = (
            self.db.update_document_extraction(
                extraction_id, status="running", error=None, completed_at=None
            )
            or record
        )
        try:
            result = extract_source(
                self._record_source_path(record),
                root=self.root,
                config=record.config,
                extraction_id=record.id,
                source_kind=record.source_kind,
                source_uri=record.source_uri,
                source_fingerprint=record.source_fingerprint,
                check_cancelled=lambda: self._cancel_check(record),
                progress=lambda processed, total: self._progress(record, processed, total),
            )
            items = [
                document.to_index_item(bundle_ordinal=index)
                for index, document in enumerate(result.documents)
            ]
            items.extend(
                failure.to_index_item(bundle_ordinal=index)
                for index, failure in enumerate(result.quarantine)
            )
            completed = self.db.complete_document_extraction(
                extraction_id,
                content_hash=result.bundle.content_hash,
                bundle_path=result.bundle.path,
                manifest_hash=result.bundle.manifest_hash,
                items=items,
                statistics=result.statistics,
                provenance={
                    **record.provenance,
                    **result.provenance,
                },
            )
            if completed.content_hash != result.bundle.content_hash:
                raise CorpusBundleIntegrityError(
                    "catalog completion recorded an unexpected bundle identity"
                )
            return result
        except CorpusExtractionCancelled as exc:
            self.db.update_document_extraction(
                extraction_id,
                status="cancelled",
                error=str(exc),
                completed_at=_now(),
            )
            raise
        except Exception as exc:
            self.db.update_document_extraction(
                extraction_id,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                completed_at=_now(),
            )
            raise

    execute_extraction = execute

    def execute_work_item(self, item: Any) -> Dict[str, Any]:
        launch_spec = (
            dict(item.get("launch_spec") or item)
            if isinstance(item, Mapping)
            else dict(getattr(item, "launch_spec", {}) or {})
        )
        extraction_id = str(
            launch_spec.get("extraction_id")
            or (
                item.get("domain_id")
                if isinstance(item, Mapping)
                else getattr(item, "domain_id", "")
            )
            or ""
        )
        if not extraction_id:
            raise ValueError("corpus extraction work item has no extraction_id")
        return self.execute(extraction_id).to_dict()

    def _load_result(self, record: Any) -> CorpusExtractionResult:
        if not record.content_hash:
            raise CorpusBundleIntegrityError("completed extraction is missing its content hash")
        verification = self.store.verify(
            record.content_hash,
            expected_source_fingerprint=record.source_fingerprint,
        )
        if not verification.valid:
            raise CorpusBundleIntegrityError(
                "corpus bundle failed verification: " + "; ".join(verification.errors)
            )
        manifest = self.store.load_manifest(record.content_hash)
        documents = tuple(self.store.iter_documents(record.content_hash))
        quarantine = tuple(self.store.iter_quarantine(record.content_hash))
        bundle = CorpusBundle(
            extraction_id=record.id,
            content_hash=record.content_hash,
            path=str(self.store.path_for(record.content_hash)),
            manifest_hash=str(verification.checksums["manifest.json"]),
            document_count=len(documents),
            quarantined_count=len(quarantine),
            checksums=dict(verification.checksums),
            created_at=str(manifest["created_at"]),
            reused=True,
        )
        return CorpusExtractionResult(
            extraction_id=record.id,
            source_uri=record.source_uri,
            source_kind=record.source_kind,
            source_fingerprint=record.source_fingerprint,
            extractor_version=record.extractor_version,
            config_hash=record.config_hash,
            documents=documents,
            quarantine=quarantine,
            bundle=bundle,
            statistics=record.statistics,
            provenance=record.provenance,
        )

    def status(self, extraction_id: str) -> Dict[str, Any]:
        record = self.db.get_document_extraction(extraction_id)
        if record is None:
            raise KeyError(extraction_id)
        value = record.to_dict()
        if record.content_hash:
            value["verification"] = self.store.verify(
                record.content_hash,
                expected_source_fingerprint=record.source_fingerprint,
            ).to_dict()
        if record.work_item_id:
            work = self.db.get_work_item(record.work_item_id)
            if work is not None:
                value["work_item"] = work.to_dict()
        return value

    get_status = status
    get = status

    def preview(
        self,
        extraction_id: str,
        *,
        limit: int = 20,
        offset: int = 0,
        include_text: bool = True,
    ) -> Dict[str, Any]:
        record = self.db.get_document_extraction(extraction_id)
        if record is None:
            raise KeyError(extraction_id)
        bounded_limit = max(0, min(200, int(limit)))
        bounded_offset = max(0, int(offset))
        records: list[Dict[str, Any]] = []
        quarantine: list[Dict[str, Any]] = []
        if record.status == "completed":
            result = self._load_result(record)
            for document in result.documents[bounded_offset : bounded_offset + bounded_limit]:
                value = document.to_dict()
                if not include_text:
                    value.pop("text", None)
                records.append(value)
            quarantine = [
                value.to_dict()
                for value in result.quarantine[bounded_offset : bounded_offset + bounded_limit]
            ]
        return {
            "extraction": record.to_dict(),
            "records": records,
            "documents": copy.deepcopy(records),
            "quarantine": quarantine,
            "total": record.document_count,
            "quarantined_total": record.quarantined_count,
            "limit": bounded_limit,
            "offset": bounded_offset,
        }

    def verify(self, extraction_id: str) -> Dict[str, Any]:
        record = self.db.get_document_extraction(extraction_id)
        if record is None:
            raise KeyError(extraction_id)
        if record.status != "completed" or not record.content_hash:
            return {
                "extraction_id": extraction_id,
                "valid": False,
                "errors": ["extraction is not completed"],
                "checksums": {},
                "path": record.bundle_path,
            }
        value = self.store.verify(
            record.content_hash,
            expected_source_fingerprint=record.source_fingerprint,
        ).to_dict()
        value["extraction_id"] = extraction_id
        value["catalog_manifest_hash"] = record.manifest_hash
        if value["checksums"].get("manifest.json") != record.manifest_hash:
            value["valid"] = False
            value["errors"].append("catalog manifest hash does not match the bundle")
        if str(value["path"]) != str(record.bundle_path):
            value["valid"] = False
            value["errors"].append("catalog bundle path does not match content-addressed storage")
        return value

    verify_extraction = verify

    def cancel(self, extraction_id: str) -> Dict[str, Any]:
        record = self.db.get_document_extraction(extraction_id)
        if record is None:
            raise KeyError(extraction_id)
        if record.status == "completed":
            raise ValueError("completed document extractions are immutable")
        if record.work_item_id and self.scheduler is not None:
            self.scheduler.cancel(record.work_item_id)
        updated = self.db.update_document_extraction(
            extraction_id,
            status="cancelled",
            error="corpus extraction cancellation requested",
            completed_at=_now(),
        )
        assert updated is not None
        return updated.to_dict()

    cancel_extraction = cancel

    def retry(
        self,
        extraction_id: str,
        *,
        synchronous: Optional[bool] = None,
    ) -> Dict[str, Any]:
        record = self.db.get_document_extraction(extraction_id)
        if record is None:
            raise KeyError(extraction_id)
        if record.status == "completed":
            return {
                "extraction": self.status(extraction_id),
                "work_item_id": record.work_item_id,
                "reused": True,
            }
        if record.status not in {"failed", "cancelled", "interrupted"}:
            raise ValueError(f"document extraction in {record.status!r} state cannot be retried")
        work = None
        if record.work_item_id and self.scheduler is not None:
            work = self.scheduler.retry(
                record.work_item_id,
                reason="operator requested corpus extraction retry",
                force=True,
                sync_domain=False,
            )
        updated = self.db.update_document_extraction(
            extraction_id,
            status="queued",
            error=None,
            completed_at=None,
            work_item_id=work.id if work is not None else record.work_item_id,
        )
        assert updated is not None
        should_execute = self.scheduler is None if synchronous is None else bool(synchronous)
        if should_execute:
            result = self.execute(extraction_id)
            value = result.to_dict()
            value["work_item_id"] = updated.work_item_id
            value["reused"] = result.bundle.reused
            return value
        if self.scheduler is not None and work is None:
            source = self._resolve_source(
                self._record_source_path(updated),
                import_id=updated.import_id,
                source_id=updated.source_id,
            )
            updated = self._enqueue(updated, source)
        return {
            "extraction": updated.to_dict(),
            "work_item_id": updated.work_item_id,
            "reused": False,
        }

    retry_extraction = retry


__all__ = ["CorpusExtractionService"]
