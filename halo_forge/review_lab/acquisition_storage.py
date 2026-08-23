"""Atomic acquisition manifests and restart-safe bounded record ingestion."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional

from ._canonical import bytes_hash, canonical_json
from .acquisition import AcquisitionPlan
from .errors import ReviewIntegrityError, ReviewStateError, ReviewValidationError

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")
INGESTION_SOURCE_HASH_FIELD = "_halo_forge_ingestion_source_sha256"


def _identifier(value: str, name: str) -> str:
    result = str(value or "").strip()
    if not _SAFE_IDENTIFIER.fullmatch(result):
        raise ReviewValidationError(
            f"{name} must use only letters, numbers, dot, underscore, or dash"
        )
    return result


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Some filesystems do not permit directory fsync. File publication is
        # still atomic on the same filesystem through os.replace.
        pass


def _write_bytes(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    return bytes_hash(payload)


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


@dataclass(frozen=True)
class AcquisitionManifestVerification:
    batch_id: str
    valid: bool
    checksums: Dict[str, str]
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "valid": self.valid,
            "checksums": copy.deepcopy(self.checksums),
            "errors": list(self.errors),
        }


class AcquisitionManifestStore:
    """Publish and verify immutable acquisition inputs.

    ``review_root`` is the same root used by :class:`ReviewLabService`.  Each
    successful publication appears atomically at
    ``review_root/acquisitions/<batch-id>``. Identical retries reuse the prior
    directory only after checksum and content-identity verification.
    """

    def __init__(self, review_root: str | Path):
        self.review_root = Path(review_root).expanduser()
        self.root = self.review_root / "acquisitions"

    def path_for(self, batch_id: str) -> Path:
        return self.root / _identifier(batch_id, "batch_id")

    def load_manifest(self, batch_id: str) -> Dict[str, Any]:
        path = self.path_for(batch_id) / "manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewIntegrityError(
                f"acquisition manifest {batch_id} is unreadable: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ReviewIntegrityError(f"acquisition manifest {batch_id} must be an object")
        return dict(value)

    @staticmethod
    def _candidate_checksum(
        path: Path,
        plan: AcquisitionPlan,
        *,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> str:
        digest = hashlib.sha256()
        with path.open("wb") as handle:
            for ordinal, row in enumerate(plan.iter_candidate_payloads()):
                if check_cancelled is not None and ordinal % 128 == 0:
                    check_cancelled()
                payload = (canonical_json(row) + "\n").encode("utf-8")
                handle.write(payload)
                digest.update(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return digest.hexdigest()

    @staticmethod
    def _manifest(batch_id: str, plan: AcquisitionPlan, candidate_hash: str) -> Dict[str, Any]:
        return {
            "format": "halo-forge-review-acquisition",
            "format_version": 1,
            "identity_version": 1,
            "batch_id": batch_id,
            "content_hash": plan.content_hash,
            "source_hash": plan.source_hash,
            "source_pins": copy.deepcopy(list(plan.source_pins)),
            "request": copy.deepcopy(plan.request),
            "eligibility": copy.deepcopy(plan.eligibility),
            "metadata": copy.deepcopy(plan.metadata),
            "row_count": len(plan.selected),
            "candidates_path": "candidates.jsonl",
            "payload_checksums": {"candidates.jsonl": candidate_hash},
        }

    def publish(
        self,
        batch_id: str,
        plan: AcquisitionPlan,
        *,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        cancel = check_cancelled or (lambda: None)
        cancel()
        identifier = _identifier(batch_id, "batch_id")
        final_path = self.path_for(identifier)
        self.root.mkdir(parents=True, exist_ok=True)
        if final_path.exists():
            verification = self.verify(
                identifier,
                expected_content_hash=plan.content_hash,
                check_cancelled=check_cancelled,
            )
            if not verification.valid:
                raise ReviewIntegrityError(
                    "existing acquisition publication failed verification: "
                    + "; ".join(verification.errors)
                )
            manifest = self.load_manifest(identifier)
            expected_candidate_hash = manifest.get("payload_checksums", {}).get("candidates.jsonl")
            candidate_rows_hash = hashlib.sha256()
            for ordinal, row in enumerate(plan.iter_candidate_payloads()):
                if ordinal % 128 == 0:
                    cancel()
                candidate_rows_hash.update((canonical_json(row) + "\n").encode("utf-8"))
            if expected_candidate_hash != candidate_rows_hash.hexdigest():
                raise ReviewIntegrityError(
                    "acquisition batch id already contains different candidate content"
                )
            # Accepting an already-published immutable manifest is the reuse
            # boundary. Cancellation observed after this check is late and the
            # caller must finish its catalog reconciliation truthfully.
            cancel()
            return manifest

        stage = self.root / f".stage-{identifier}-{uuid.uuid4().hex}"
        stage.mkdir(parents=False, exist_ok=False)
        try:
            candidate_hash = self._candidate_checksum(
                stage / "candidates.jsonl",
                plan,
                check_cancelled=check_cancelled,
            )
            manifest = self._manifest(identifier, plan, candidate_hash)
            manifest_hash = _write_bytes(
                stage / "manifest.json", (canonical_json(manifest) + "\n").encode("utf-8")
            )
            checksums = {
                "candidates.jsonl": candidate_hash,
                "manifest.json": manifest_hash,
            }
            _write_bytes(
                stage / "checksums.json", (canonical_json(checksums) + "\n").encode("utf-8")
            )
            _fsync_directory(stage)
            # This is the acquisition publication boundary. Before it, a
            # cancellation removes staging and publishes nothing. After it,
            # callers complete SQLite reconciliation and report success even
            # when a later operator cancellation races with that reconciliation.
            cancel()
            try:
                os.replace(stage, final_path)
            except OSError:
                # A same-ID publisher may have won the race. Reuse only after
                # proving it published this exact plan.
                if not final_path.exists():
                    raise
                verification = self.verify(
                    identifier,
                    expected_content_hash=plan.content_hash,
                )
                if not verification.valid:
                    raise ReviewIntegrityError(
                        "concurrent acquisition publication contains incompatible content"
                    )
            _fsync_directory(self.root)
            return self.load_manifest(identifier)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def verify(
        self,
        batch_id: str,
        *,
        expected_content_hash: Optional[str] = None,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> AcquisitionManifestVerification:
        cancel = check_cancelled or (lambda: None)
        cancel()
        identifier = _identifier(batch_id, "batch_id")
        root = self.path_for(identifier)
        errors: List[str] = []
        observed: Dict[str, str] = {}
        if not root.is_dir():
            return AcquisitionManifestVerification(
                identifier, False, {}, ["storage directory is missing"]
            )
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, Mapping):
                raise ValueError("manifest must be an object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return AcquisitionManifestVerification(
                identifier, False, {}, [f"manifest unreadable: {exc}"]
            )
        if manifest.get("format") != "halo-forge-review-acquisition":
            errors.append("manifest format is unsupported")
        if manifest.get("format_version") != 1:
            errors.append("manifest version is unsupported")
        if manifest.get("batch_id") != identifier:
            errors.append("manifest batch id does not match storage path")
        if (
            expected_content_hash is not None
            and manifest.get("content_hash") != expected_content_hash
        ):
            errors.append("manifest content hash does not match expected acquisition plan")
        try:
            checksum_document = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
            if not isinstance(checksum_document, Mapping):
                raise ValueError("checksums document must be an object")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            checksum_document = {}
            errors.append(f"checksums document unreadable: {exc}")
        expected_names = {"candidates.jsonl", "manifest.json"}
        if set(checksum_document) != expected_names:
            errors.append("checksums document must cover candidates.jsonl and manifest.json")
        for filename in sorted(expected_names):
            path = root / filename
            if not path.is_file():
                errors.append(f"missing file: {filename}")
                continue
            observed[filename] = _hash_file(path, check_cancelled=check_cancelled)
            if observed[filename] != str(checksum_document.get(filename) or ""):
                errors.append(f"checksum mismatch: {filename}")
        payload_checksums = manifest.get("payload_checksums")
        if not isinstance(payload_checksums, Mapping):
            errors.append("manifest payload checksums are missing")
        elif payload_checksums.get("candidates.jsonl") != checksum_document.get("candidates.jsonl"):
            errors.append("candidate checksum does not match manifest")

        row_count = 0
        candidates_path = root / "candidates.jsonl"
        if candidates_path.is_file():
            try:
                with candidates_path.open("rb") as handle:
                    for expected_ordinal, line in enumerate(handle):
                        if expected_ordinal % 128 == 0:
                            cancel()
                        if not line.endswith(b"\n"):
                            raise ValueError("unterminated candidate line")
                        row = json.loads(line)
                        if not isinstance(row, Mapping):
                            raise ValueError("candidate line must be an object")
                        if row.get("ordinal") != expected_ordinal:
                            raise ValueError("candidate ordinals are not contiguous")
                        if line != (canonical_json(row) + "\n").encode("utf-8"):
                            raise ValueError("candidate line is not canonical JSON")
                        row_count += 1
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"candidate records unreadable: {exc}")
        try:
            expected_rows = int(manifest.get("row_count", -1))
        except (TypeError, ValueError):
            expected_rows = -1
            errors.append("manifest row count is invalid")
        if row_count != expected_rows:
            errors.append("candidate row count does not match manifest")
        return AcquisitionManifestVerification(identifier, not errors, observed, errors)

    def iter_candidates(self, batch_id: str) -> Iterator[Dict[str, Any]]:
        verification = self.verify(batch_id)
        if not verification.valid:
            raise ReviewIntegrityError(
                "acquisition publication failed verification: " + "; ".join(verification.errors)
            )
        with (self.path_for(batch_id) / "candidates.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                yield dict(json.loads(line))


class AcquisitionRecordSpool:
    """Append-only, bounded-memory source spool for durable workers.

    One worker owns a spool at a time and appends source pages as they are
    resolved. Every append fsyncs complete canonical JSONL records before an
    atomic checkpoint update. On restart, complete lines written after the last
    checkpoint are adopted; a trailing partial line is removed while the spool
    is open, but is an integrity failure after sealing.
    """

    def __init__(self, review_root: str | Path, ingestion_id: str):
        self.review_root = Path(review_root).expanduser()
        self.ingestion_id = _identifier(ingestion_id, "ingestion_id")
        self.root = self.review_root / "acquisitions" / ".ingest" / self.ingestion_id
        self.records_path = self.root / "records.jsonl"
        self.checkpoint_path = self.root / "checkpoint.json"
        self._lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.records_path.exists():
            self.records_path.touch()
            with self.records_path.open("rb") as handle:
                os.fsync(handle.fileno())
        checkpoint = self._read_checkpoint()
        self._sealed = bool(checkpoint.get("sealed", False)) if checkpoint else False
        self._count, self._bytes, self._hasher = self._scan(repair_partial=not self._sealed)
        observed_hash = self._hasher.hexdigest()
        if self._sealed and checkpoint:
            if (
                checkpoint.get("records_sha256") != observed_hash
                or checkpoint.get("record_count") != self._count
                or checkpoint.get("bytes") != self._bytes
            ):
                raise ReviewIntegrityError("sealed acquisition spool was mutated")
        self._write_checkpoint()

    def _read_checkpoint(self) -> Dict[str, Any]:
        if not self.checkpoint_path.exists():
            return {}
        try:
            value = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewIntegrityError(
                f"acquisition spool checkpoint is unreadable: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ReviewIntegrityError("acquisition spool checkpoint must be an object")
        if value.get("ingestion_id") != self.ingestion_id:
            raise ReviewIntegrityError("acquisition spool checkpoint identity changed")
        return dict(value)

    def _scan(self, *, repair_partial: bool) -> tuple[int, int, Any]:
        count = 0
        size = 0
        digest = hashlib.sha256()
        with self.records_path.open("rb+") as handle:
            while True:
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    if repair_partial:
                        handle.truncate(line_start)
                        handle.flush()
                        os.fsync(handle.fileno())
                        break
                    raise ReviewIntegrityError("sealed acquisition spool has a partial record")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReviewIntegrityError(
                        f"acquisition spool record {count} is invalid JSON"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ReviewIntegrityError(
                        f"acquisition spool record {count} must be an object"
                    )
                if line != (canonical_json(row) + "\n").encode("utf-8"):
                    raise ReviewIntegrityError(
                        f"acquisition spool record {count} is not canonical JSON"
                    )
                digest.update(line)
                size += len(line)
                count += 1
        return count, size, digest

    def _checkpoint(self) -> Dict[str, Any]:
        return {
            "format": "halo-forge-acquisition-spool",
            "format_version": 1,
            "ingestion_id": self.ingestion_id,
            "record_count": self._count,
            "bytes": self._bytes,
            "records_sha256": self._hasher.hexdigest(),
            "sealed": self._sealed,
        }

    def _write_checkpoint(self) -> None:
        payload = (canonical_json(self._checkpoint()) + "\n").encode("utf-8")
        temporary = self.root / f".checkpoint-{uuid.uuid4().hex}.tmp"
        try:
            _write_bytes(temporary, payload)
            os.replace(temporary, self.checkpoint_path)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)

    @property
    def checkpoint(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._checkpoint())

    def append(self, records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            if self._sealed:
                raise ReviewStateError("sealed acquisition spool cannot accept more records")
            with self.records_path.open("ab") as handle:
                try:
                    for ordinal, record in enumerate(records, start=self._count):
                        if not isinstance(record, Mapping):
                            raise ReviewValidationError(
                                f"acquisition spool record {ordinal} must be an object"
                            )
                        line = (canonical_json(dict(record)) + "\n").encode("utf-8")
                        handle.write(line)
                        self._hasher.update(line)
                        self._count += 1
                        self._bytes += len(line)
                finally:
                    handle.flush()
                    os.fsync(handle.fileno())
                    self._write_checkpoint()
            return self.checkpoint

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        with self.records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield dict(json.loads(line))

    def resume_after_verified_prefix(
        self,
        source_records: Iterable[Mapping[str, Any]],
        *,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> Iterator[Mapping[str, Any]]:
        """Return the unspooled source tail after proving its prefix is unchanged.

        Merely skipping ``record_count`` rows can create a hybrid batch when a
        mutable import is reordered between attempts.  Recovery therefore
        compares every persisted canonical record with the newly resolved
        source before allowing any additional append.
        """

        source = iter(source_records)
        for ordinal, persisted in enumerate(self.iter_records()):
            if check_cancelled is not None and ordinal % 128 == 0:
                check_cancelled()
            try:
                current = next(source)
            except StopIteration as exc:
                raise ReviewIntegrityError(
                    "acquisition source became shorter while recovering its durable spool"
                ) from exc
            if not isinstance(current, Mapping):
                raise ReviewValidationError(
                    f"acquisition source record {ordinal} must be an object"
                )
            current_payload = dict(current)
            expected_source_hash = persisted.get(INGESTION_SOURCE_HASH_FIELD)
            if expected_source_hash:
                current_source_hash = bytes_hash(canonical_json(current_payload).encode("utf-8"))
                unchanged = current_source_hash == str(expected_source_hash)
            else:
                unchanged = canonical_json(current_payload) == canonical_json(persisted)
            if not unchanged:
                raise ReviewIntegrityError(
                    "acquisition source changed or reordered while recovering its "
                    f"durable spool at record {ordinal}"
                )
        if check_cancelled is not None:
            check_cancelled()
        return source

    def seal(self) -> Dict[str, Any]:
        with self._lock:
            self._sealed = True
            self._write_checkpoint()
            return self.source_pin()

    def source_pin(self) -> Dict[str, Any]:
        if not self._sealed:
            raise ReviewStateError("acquisition spool must be sealed before it is pinned")
        return {
            "kind": "acquisition_spool",
            "ref": self.ingestion_id,
            "format_version": 1,
            "row_count": self._count,
            "content_hash": self._hasher.hexdigest(),
        }


__all__ = [
    "AcquisitionManifestStore",
    "AcquisitionManifestVerification",
    "AcquisitionRecordSpool",
    "INGESTION_SOURCE_HASH_FIELD",
]
