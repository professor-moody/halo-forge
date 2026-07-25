"""Atomic, content-addressed storage for extracted corpus documents."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence

from .models import (
    CORPUS_BUNDLE_FORMAT,
    CORPUS_BUNDLE_FORMAT_VERSION,
    CORPUS_IDENTITY_VERSION,
    CorpusBundle,
    CorpusBundleVerification,
    CorpusDocument,
    CorpusExtractionConfig,
    ExtractionFailure,
    canonical_json,
    sha256_bytes,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CorpusBundleIntegrityError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _write_bytes(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    return sha256_bytes(payload)


def _hash_file(path: Path, *, check_cancelled: Optional[Callable[[], None]] = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if check_cancelled is not None:
                check_cancelled()
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_bytes(values: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((canonical_json(value) + "\n").encode("utf-8") for value in values)


class CorpusBundleStore:
    """Publish complete bundles beneath ``root/bundles/<prefix>/<sha256>``."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()
        self.bundles_root = self.root / "bundles"
        self.bundles_root.mkdir(parents=True, exist_ok=True)

    def path_for(self, content_hash: str) -> Path:
        normalized = str(content_hash or "").strip().lower()
        if not _SHA256.fullmatch(normalized):
            raise ValueError("corpus bundle content_hash must be a lowercase SHA-256")
        return self.bundles_root / normalized[:2] / normalized

    @staticmethod
    def _content_identity(
        *,
        source_fingerprint: str,
        extractor_version: str,
        config_hash: str,
        document_checksum: str,
        quarantine_checksum: str,
        document_count: int,
        quarantined_count: int,
    ) -> str:
        return sha256_bytes(
            canonical_json(
                {
                    "format": CORPUS_BUNDLE_FORMAT,
                    "format_version": CORPUS_BUNDLE_FORMAT_VERSION,
                    "identity_version": CORPUS_IDENTITY_VERSION,
                    "source_fingerprint": source_fingerprint,
                    "extractor_version": extractor_version,
                    "config_hash": config_hash,
                    "documents_sha256": document_checksum,
                    "quarantine_sha256": quarantine_checksum,
                    "document_count": int(document_count),
                    "quarantined_count": int(quarantined_count),
                }
            ).encode("utf-8")
        )

    def publish(
        self,
        *,
        extraction_id: str,
        source_uri: str,
        source_kind: str,
        source_fingerprint: str,
        extractor_version: str,
        config: CorpusExtractionConfig | Mapping[str, Any],
        documents: Sequence[CorpusDocument],
        quarantine: Sequence[ExtractionFailure],
        statistics: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> CorpusBundle:
        cancel = check_cancelled or (lambda: None)
        resolved_config = CorpusExtractionConfig.from_value(config)
        cancel()
        document_payload = _jsonl_bytes([value.to_dict() for value in documents])
        quarantine_payload = _jsonl_bytes([value.to_dict() for value in quarantine])
        document_checksum = sha256_bytes(document_payload)
        quarantine_checksum = sha256_bytes(quarantine_payload)
        content_hash = self._content_identity(
            source_fingerprint=source_fingerprint,
            extractor_version=extractor_version,
            config_hash=resolved_config.fingerprint,
            document_checksum=document_checksum,
            quarantine_checksum=quarantine_checksum,
            document_count=len(documents),
            quarantined_count=len(quarantine),
        )
        final_path = self.path_for(content_hash)
        if final_path.exists():
            verification = self.verify(
                content_hash,
                expected_source_fingerprint=source_fingerprint,
                check_cancelled=check_cancelled,
            )
            if not verification.valid:
                raise CorpusBundleIntegrityError(
                    "existing corpus bundle failed verification: " + "; ".join(verification.errors)
                )
            manifest = self.load_manifest(content_hash)
            expected = manifest.get("payload_checksums") or {}
            if (
                expected.get("documents.jsonl") != document_checksum
                or expected.get("quarantine.jsonl") != quarantine_checksum
            ):
                raise CorpusBundleIntegrityError(
                    "content-addressed corpus bundle contains different payload bytes"
                )
            return CorpusBundle(
                extraction_id=str(extraction_id),
                content_hash=content_hash,
                path=str(final_path),
                manifest_hash=str(
                    verification.checksums.get("manifest.json")
                    or _hash_file(final_path / "manifest.json")
                ),
                document_count=len(documents),
                quarantined_count=len(quarantine),
                checksums=dict(verification.checksums),
                created_at=str(manifest["created_at"]),
                reused=True,
            )

        parent = final_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".stage-{content_hash[:16]}-",
                dir=parent,
            )
        )
        try:
            documents_hash = _write_bytes(stage / "documents.jsonl", document_payload)
            quarantine_hash = _write_bytes(stage / "quarantine.jsonl", quarantine_payload)
            created_at = _utc_now()
            manifest: Dict[str, Any] = {
                "format": CORPUS_BUNDLE_FORMAT,
                "format_version": CORPUS_BUNDLE_FORMAT_VERSION,
                "identity_version": CORPUS_IDENTITY_VERSION,
                "producer_extraction_id": str(extraction_id),
                "content_hash": content_hash,
                "source_uri": str(source_uri),
                "source_kind": str(source_kind),
                "source_fingerprint": str(source_fingerprint),
                "extractor_version": str(extractor_version),
                "config_hash": resolved_config.fingerprint,
                "config": resolved_config.to_dict(),
                "document_count": len(documents),
                "quarantined_count": len(quarantine),
                "item_count": len(documents) + len(quarantine),
                "documents_path": "documents.jsonl",
                "quarantine_path": "quarantine.jsonl",
                "payload_checksums": {
                    "documents.jsonl": documents_hash,
                    "quarantine.jsonl": quarantine_hash,
                },
                "statistics": dict(statistics or {}),
                "provenance": dict(provenance or {}),
                "created_at": created_at,
            }
            manifest_hash = _write_bytes(
                stage / "manifest.json",
                (canonical_json(manifest) + "\n").encode("utf-8"),
            )
            checksums = {
                "documents.jsonl": documents_hash,
                "manifest.json": manifest_hash,
                "quarantine.jsonl": quarantine_hash,
            }
            _write_bytes(
                stage / "checksums.json",
                (canonical_json(checksums) + "\n").encode("utf-8"),
            )
            _fsync_directory(stage)
            cancel()
            try:
                os.replace(stage, final_path)
            except OSError:
                if not final_path.exists():
                    raise
                verification = self.verify(
                    content_hash,
                    expected_source_fingerprint=source_fingerprint,
                )
                if not verification.valid:
                    raise CorpusBundleIntegrityError(
                        "concurrent corpus publication contains incompatible content"
                    )
                manifest = self.load_manifest(content_hash)
                checksums = dict(verification.checksums)
                manifest_hash = checksums["manifest.json"]
                created_at = str(manifest["created_at"])
            _fsync_directory(parent)
            return CorpusBundle(
                extraction_id=str(extraction_id),
                content_hash=content_hash,
                path=str(final_path),
                manifest_hash=manifest_hash,
                document_count=len(documents),
                quarantined_count=len(quarantine),
                checksums=checksums,
                created_at=created_at,
                reused=False,
            )
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def load_manifest(self, content_hash: str) -> Dict[str, Any]:
        path = self.path_for(content_hash) / "manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusBundleIntegrityError(
                f"corpus bundle manifest is unreadable: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise CorpusBundleIntegrityError("corpus bundle manifest must be an object")
        return dict(value)

    def verify(
        self,
        content_hash: str,
        *,
        expected_source_fingerprint: Optional[str] = None,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> CorpusBundleVerification:
        cancel = check_cancelled or (lambda: None)
        normalized = str(content_hash or "").strip().lower()
        path = self.path_for(normalized)
        errors: list[str] = []
        observed: Dict[str, str] = {}
        if not path.is_dir():
            return CorpusBundleVerification(
                normalized, False, str(path), {}, ("bundle directory is missing",)
            )
        cancel()
        try:
            checksums = json.loads((path / "checksums.json").read_text(encoding="utf-8"))
            if not isinstance(checksums, Mapping):
                raise ValueError("checksums.json must be an object")
            checksums = {str(key): str(value) for key, value in checksums.items()}
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return CorpusBundleVerification(
                normalized,
                False,
                str(path),
                {},
                (f"checksums unreadable: {exc}",),
            )
        expected_files = {
            "documents.jsonl",
            "manifest.json",
            "quarantine.jsonl",
        }
        if set(checksums) != expected_files:
            errors.append("checksums.json does not enumerate the exact bundle payload")
        for filename in sorted(expected_files):
            payload_path = path / filename
            if not payload_path.is_file():
                errors.append(f"{filename} is missing")
                continue
            observed[filename] = _hash_file(payload_path, check_cancelled=check_cancelled)
            if checksums.get(filename) != observed[filename]:
                errors.append(f"{filename} checksum mismatch")
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, Mapping):
                raise ValueError("manifest must be an object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"manifest unreadable: {exc}")
            manifest = {}
        if manifest.get("format") != CORPUS_BUNDLE_FORMAT:
            errors.append("manifest format is unsupported")
        if manifest.get("format_version") != CORPUS_BUNDLE_FORMAT_VERSION:
            errors.append("manifest format version is unsupported")
        if manifest.get("identity_version") != CORPUS_IDENTITY_VERSION:
            errors.append("manifest identity version is unsupported")
        if manifest.get("content_hash") != normalized:
            errors.append("manifest content hash does not match its storage path")
        if (
            expected_source_fingerprint is not None
            and manifest.get("source_fingerprint") != expected_source_fingerprint
        ):
            errors.append("manifest source fingerprint does not match the request")
        payload_checksums = dict(manifest.get("payload_checksums") or {})
        if payload_checksums.get("documents.jsonl") != checksums.get("documents.jsonl"):
            errors.append("manifest documents checksum does not match checksums.json")
        if payload_checksums.get("quarantine.jsonl") != checksums.get("quarantine.jsonl"):
            errors.append("manifest quarantine checksum does not match checksums.json")
        try:
            calculated_identity = self._content_identity(
                source_fingerprint=str(manifest["source_fingerprint"]),
                extractor_version=str(manifest["extractor_version"]),
                config_hash=str(manifest["config_hash"]),
                document_checksum=str(checksums["documents.jsonl"]),
                quarantine_checksum=str(checksums["quarantine.jsonl"]),
                document_count=int(manifest["document_count"]),
                quarantined_count=int(manifest["quarantined_count"]),
            )
            if calculated_identity != normalized:
                errors.append("bundle content identity could not be reproduced")
        except (KeyError, TypeError, ValueError):
            errors.append("manifest content identity fields are incomplete")
        cancel()
        return CorpusBundleVerification(
            normalized,
            not errors,
            str(path),
            observed,
            tuple(errors),
        )

    def iter_documents(
        self,
        content_hash: str,
        *,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> Iterator[CorpusDocument]:
        verification = self.verify(content_hash, check_cancelled=check_cancelled)
        if not verification.valid:
            raise CorpusBundleIntegrityError(
                "corpus bundle failed verification: " + "; ".join(verification.errors)
            )
        cancel = check_cancelled or (lambda: None)
        with (self.path_for(content_hash) / "documents.jsonl").open(encoding="utf-8") as handle:
            for ordinal, line in enumerate(handle):
                if ordinal % 128 == 0:
                    cancel()
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise CorpusBundleIntegrityError(
                            "documents.jsonl contains a non-object row"
                        )
                    yield CorpusDocument.from_dict(value)

    def iter_quarantine(
        self,
        content_hash: str,
        *,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> Iterator[ExtractionFailure]:
        verification = self.verify(content_hash, check_cancelled=check_cancelled)
        if not verification.valid:
            raise CorpusBundleIntegrityError(
                "corpus bundle failed verification: " + "; ".join(verification.errors)
            )
        cancel = check_cancelled or (lambda: None)
        with (self.path_for(content_hash) / "quarantine.jsonl").open(encoding="utf-8") as handle:
            for ordinal, line in enumerate(handle):
                if ordinal % 128 == 0:
                    cancel()
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise CorpusBundleIntegrityError(
                            "quarantine.jsonl contains a non-object row"
                        )
                    yield ExtractionFailure.from_dict(value)


__all__ = [
    "CorpusBundleIntegrityError",
    "CorpusBundleStore",
]
