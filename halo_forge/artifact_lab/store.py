"""Filesystem-backed content-addressed artifact store.

This module deliberately has no database dependency.  SQLite records can point
at these immutable manifests, while tests and headless workflows can use the
same lifecycle directly.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .hashing import (
    ContentDigest,
    atomic_write_json,
    checksum_lines,
    copy_payload,
    fingerprint,
    hash_file,
    hash_path,
    read_json,
)
from .models import (
    ArtifactBlob,
    ArtifactEdge,
    ArtifactLocation,
    ArtifactRegistration,
    ArtifactVerification,
    CleanupCandidate,
    CleanupPlan,
    CleanupProtections,
    CleanupResult,
    LineageGraph,
    ModelArtifactOccurrence,
    PortableExportBundle,
    ProtectedArtifact,
    StorageInventory,
)

DEFAULT_ARTIFACT_ROOT = Path("~/.halo-forge/artifacts")
TRASH_RETENTION_DAYS = 7


class ArtifactStoreError(RuntimeError):
    """Base error for artifact lifecycle failures."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Content does not match its immutable identity."""


class ArtifactProtectedError(ArtifactStoreError):
    """A destructive action targeted protected content."""


class ArtifactLoadProbeUnavailable(ArtifactStoreError):
    """No installed backend can perform a real load probe for this format."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _safe_hash(value: str) -> str:
    if len(value) != 64:
        raise ValueError("Artifact content hash must contain 64 hexadecimal characters")
    int(value, 16)
    return value.lower()


def _safe_id(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def probe_artifact_loadability(path: Path, blob: ArtifactBlob) -> Mapping[str, Any]:
    """Use an installed runtime to perform a real, opt-in load probe.

    This deliberately supports only probes that ask an inference runtime to
    parse/load the artifact.  JSON parsing, filename checks, and GGUF magic
    bytes remain structural evidence and are never relabeled as loadability.
    """

    artifact_format = blob.format.lower()
    if artifact_format == "gguf":
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ArtifactLoadProbeUnavailable(
                "GGUF load verification requires llama-cpp-python"
            ) from exc
        candidate = path if path.is_file() else next(iter(sorted(path.glob("*.gguf"))), None)
        if candidate is None:
            return {"passed": False, "backend": "llama.cpp", "error": "GGUF file is missing"}
        model = Llama(model_path=str(candidate), vocab_only=True, verbose=False)
        close = getattr(model, "close", None)
        if callable(close):
            close()
        return {"passed": True, "backend": "llama.cpp", "mode": "vocab_only"}
    if artifact_format == "onnx":
        try:
            import onnx
        except ImportError as exc:
            raise ArtifactLoadProbeUnavailable("ONNX load verification requires onnx") from exc
        candidate = path if path.is_file() else next(iter(sorted(path.glob("*.onnx"))), None)
        if candidate is None:
            return {"passed": False, "backend": "onnx", "error": "ONNX file is missing"}
        model = onnx.load_model(str(candidate), load_external_data=True)
        onnx.checker.check_model(model)
        return {"passed": True, "backend": "onnx", "checker": True}
    raise ArtifactLoadProbeUnavailable(
        f"No safe built-in load probe is registered for {artifact_format!r}; "
        "supply the trainer or serving backend's loader explicitly"
    )


class ArtifactStore:
    """Manage immutable blobs, locations, occurrences, lineage, and exports."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ):
        self.root = Path(root or DEFAULT_ARTIFACT_ROOT).expanduser().resolve()
        self.clock = clock
        self.blobs_dir = self.root / "blobs"
        self.metadata_dir = self.root / "metadata"
        self.staging_dir = self.root / ".staging"
        self.trash_dir = self.root / "trash"
        for path in (
            self.blobs_dir,
            self.metadata_dir / "blobs",
            self.metadata_dir / "locations",
            self.metadata_dir / "occurrences",
            self.metadata_dir / "edges",
            self.metadata_dir / "operations",
            self.metadata_dir / "exports",
            self.metadata_dir / "cleanup_plans",
            self.metadata_dir / "cleanup_actions",
            self.staging_dir,
            self.trash_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _now(self) -> str:
        return self.clock().astimezone(timezone.utc).isoformat()

    def _blob_manifest_path(self, content_hash: str) -> Path:
        return self.metadata_dir / "blobs" / f"{_safe_hash(content_hash)}.json"

    def _location_manifest_path(self, location_id: str) -> Path:
        return self.metadata_dir / "locations" / f"{_safe_id(location_id, 'location ID')}.json"

    def _occurrence_manifest_path(self, occurrence_id: str) -> Path:
        return (
            self.metadata_dir / "occurrences" / f"{_safe_id(occurrence_id, 'occurrence ID')}.json"
        )

    def _managed_payload_path(self, content_hash: str) -> Path:
        return self.blobs_dir / _safe_hash(content_hash) / "payload"

    @staticmethod
    def _blob_from_dict(value: Mapping[str, Any]) -> ArtifactBlob:
        allowed = {
            "content_hash",
            "artifact_kind",
            "format",
            "size_bytes",
            "file_count",
            "created_at",
            "dtype",
            "quantization",
            "quantization_method",
            "integrity",
            "manifest_version",
            "metadata",
        }
        return ArtifactBlob(**{key: value[key] for key in allowed if key in value})

    @staticmethod
    def _location_from_dict(value: Mapping[str, Any]) -> ArtifactLocation:
        return ArtifactLocation(**dict(value))

    @staticmethod
    def _occurrence_from_dict(value: Mapping[str, Any]) -> ModelArtifactOccurrence:
        return ModelArtifactOccurrence(**dict(value))

    def get_blob(self, content_hash: str) -> ArtifactBlob:
        path = self._blob_manifest_path(content_hash)
        if not path.is_file():
            raise KeyError(f"Unknown artifact blob: {content_hash}")
        return self._blob_from_dict(read_json(path))

    def list_blobs(self) -> list[ArtifactBlob]:
        return [
            self._blob_from_dict(read_json(path))
            for path in sorted((self.metadata_dir / "blobs").glob("*.json"))
        ]

    def get_location(self, location_id: str) -> ArtifactLocation:
        path = self._location_manifest_path(location_id)
        if not path.is_file():
            raise KeyError(f"Unknown artifact location: {location_id}")
        return self._location_from_dict(read_json(path))

    def list_locations(self, *, content_hash: Optional[str] = None) -> list[ArtifactLocation]:
        values = [
            self._location_from_dict(read_json(path))
            for path in sorted((self.metadata_dir / "locations").glob("*.json"))
        ]
        if content_hash is not None:
            values = [item for item in values if item.content_hash == content_hash]
        return values

    def get_occurrence(self, occurrence_id: str) -> ModelArtifactOccurrence:
        path = self._occurrence_manifest_path(occurrence_id)
        if not path.is_file():
            raise KeyError(f"Unknown artifact occurrence: {occurrence_id}")
        return self._occurrence_from_dict(read_json(path))

    def list_occurrences(
        self, *, content_hash: Optional[str] = None, run_id: Optional[str] = None
    ) -> list[ModelArtifactOccurrence]:
        values = [
            self._occurrence_from_dict(read_json(path))
            for path in sorted((self.metadata_dir / "occurrences").glob("*.json"))
        ]
        if content_hash is not None:
            values = [item for item in values if item.content_hash == content_hash]
        if run_id is not None:
            values = [item for item in values if item.run_id == run_id]
        return values

    def _create_blob_manifest(
        self,
        digest: ContentDigest,
        *,
        artifact_kind: str,
        artifact_format: str,
        dtype: Optional[str],
        quantization: Optional[str],
        quantization_method: Optional[str],
        metadata: Optional[Mapping[str, Any]],
    ) -> tuple[ArtifactBlob, bool]:
        manifest_path = self._blob_manifest_path(digest.content_hash)
        if manifest_path.exists():
            blob = self.get_blob(digest.content_hash)
            if blob.size_bytes != digest.size_bytes or blob.file_count != digest.file_count:
                raise ArtifactIntegrityError(
                    f"Content-address collision for {digest.content_hash}: inventory differs"
                )
            declared_identity = {
                "format": artifact_format.strip().lower(),
                "dtype": dtype,
                "quantization": quantization,
                "quantization_method": quantization_method,
            }
            stored_identity = {
                "format": blob.format,
                "dtype": blob.dtype,
                "quantization": blob.quantization,
                "quantization_method": blob.quantization_method,
            }
            if declared_identity != stored_identity:
                raise ArtifactIntegrityError(
                    f"Content {digest.content_hash} is already registered with different "
                    f"format/dtype/quantization identity: {stored_identity}"
                )
            return blob, True
        blob = ArtifactBlob(
            content_hash=digest.content_hash,
            artifact_kind=artifact_kind,
            format=artifact_format.strip().lower(),
            dtype=dtype,
            quantization=quantization,
            quantization_method=quantization_method,
            size_bytes=digest.size_bytes,
            file_count=digest.file_count,
            created_at=self._now(),
            metadata=dict(metadata or {}),
        )
        atomic_write_json(manifest_path, blob.to_dict())
        return blob, False

    def _publish_managed_payload(self, source: Path, digest: ContentDigest) -> Path:
        final_dir = self.blobs_dir / digest.content_hash
        final_payload = final_dir / "payload"
        if final_payload.exists():
            current = hash_path(final_payload)
            if current.content_hash != digest.content_hash:
                raise ArtifactIntegrityError(
                    f"Managed blob {digest.content_hash} is mutated at {final_payload}"
                )
            return final_payload

        stage = Path(tempfile.mkdtemp(prefix="publish-", dir=self.staging_dir))
        try:
            copy_payload(source, stage / "payload")
            copied = hash_path(stage / "payload")
            if copied.content_hash != digest.content_hash:
                raise ArtifactIntegrityError(
                    "Source changed while it was being adopted into the artifact library"
                )
            # The directory rename is the content publication point.  A blob
            # directory is never visible until its complete payload is ready.
            try:
                os.replace(stage, final_dir)
            except OSError:
                if not final_payload.exists():
                    raise
                shutil.rmtree(stage, ignore_errors=True)
                if hash_path(final_payload).content_hash != digest.content_hash:
                    raise ArtifactIntegrityError(
                        f"Concurrent publication produced the wrong content at {final_payload}"
                    )
            return final_payload
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    def import_artifact(
        self,
        source: Path | str,
        *,
        artifact_kind: str,
        artifact_format: str,
        managed: bool = False,
        dtype: Optional[str] = None,
        quantization: Optional[str] = None,
        quantization_method: Optional[str] = None,
        occurrence_id: Optional[str] = None,
        run_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        segment_id: Optional[str] = None,
        step: Optional[int] = None,
        cycle: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ArtifactRegistration:
        """Register an artifact, referencing source bytes unless ``managed``.

        The raw run path is never modified.  Managed imports are copied into a
        same-filesystem staging directory, verified, then published atomically.
        """

        source_path = Path(source).expanduser().resolve(strict=True)
        digest = hash_path(source_path)
        if managed:
            location_path = self._publish_managed_payload(source_path, digest)
            location_kind = "managed"
        else:
            location_path = source_path
            location_kind = "referenced"
        blob, reused = self._create_blob_manifest(
            digest,
            artifact_kind=artifact_kind,
            artifact_format=artifact_format,
            dtype=dtype,
            quantization=quantization,
            quantization_method=quantization_method,
            metadata=metadata,
        )
        now = self._now()
        location_id = f"loc-{fingerprint({'hash': digest.content_hash, 'kind': location_kind, 'path': str(location_path)})[:24]}"
        location = ArtifactLocation(
            id=location_id,
            content_hash=digest.content_hash,
            location_kind=location_kind,
            path=str(location_path),
            source_path=str(source_path) if managed else None,
            created_at=now,
            verified_at=now,
        )
        location_manifest = self._location_manifest_path(location.id)
        if location_manifest.exists():
            existing_location = self.get_location(location.id)
            if existing_location != location:
                # A re-import has a new verified timestamp but the immutable
                # location identity is otherwise the same. Preserve its first
                # publication record.
                location = existing_location
        else:
            atomic_write_json(location_manifest, location.to_dict())

        occurrence = ModelArtifactOccurrence(
            id=occurrence_id or f"artifact-{uuid.uuid4().hex}",
            content_hash=digest.content_hash,
            artifact_kind=artifact_kind,
            location_id=location.id,
            run_id=run_id,
            run_group_id=run_group_id,
            trial_id=trial_id,
            segment_id=segment_id,
            step=step,
            cycle=cycle,
            metadata=dict(metadata or {}),
            created_at=now,
        )
        occurrence_manifest = self._occurrence_manifest_path(occurrence.id)
        if occurrence_manifest.exists():
            existing = self.get_occurrence(occurrence.id)
            comparable = ModelArtifactOccurrence(
                **{**occurrence.to_dict(), "created_at": existing.created_at}
            )
            if existing != comparable:
                raise ArtifactStoreError(
                    f"Occurrence ID {occurrence.id!r} already identifies different content"
                )
            occurrence = existing
        else:
            atomic_write_json(occurrence_manifest, occurrence.to_dict())
        return ArtifactRegistration(
            blob=blob,
            location=location,
            occurrence=occurrence,
            reused_blob=reused,
        )

    def adopt(
        self,
        content_hash: str,
        *,
        source_location_id: Optional[str] = None,
    ) -> ArtifactLocation:
        """Create a verified managed copy without removing the reference."""

        blob = self.get_blob(content_hash)
        managed_path = self._managed_payload_path(blob.content_hash)
        existing_managed = [
            item
            for item in self.list_locations(content_hash=blob.content_hash)
            if item.managed and Path(item.path).exists()
        ]
        if existing_managed:
            report = self.verify(existing_managed[0].id)
            if not report.passed:
                raise ArtifactIntegrityError(
                    f"Existing managed copy failed verification: {existing_managed[0].path}"
                )
            return existing_managed[0]

        if source_location_id:
            source_location = self.get_location(source_location_id)
            if source_location.content_hash != blob.content_hash:
                raise ArtifactStoreError("Source location points at a different artifact blob")
        else:
            candidates = [
                item
                for item in self.list_locations(content_hash=blob.content_hash)
                if Path(item.path).exists()
            ]
            if not candidates:
                raise ArtifactStoreError(f"No available source location for {blob.content_hash}")
            source_location = candidates[0]
        source_path = Path(source_location.path)
        source_digest = hash_path(source_path)
        if source_digest.content_hash != blob.content_hash:
            raise ArtifactIntegrityError(f"Source location {source_location.id} has changed")
        self._publish_managed_payload(source_path, source_digest)
        now = self._now()
        location_id = f"loc-{fingerprint({'hash': blob.content_hash, 'kind': 'managed', 'path': str(managed_path)})[:24]}"
        managed_location = ArtifactLocation(
            id=location_id,
            content_hash=blob.content_hash,
            location_kind="managed",
            path=str(managed_path),
            source_path=source_location.path,
            created_at=now,
            verified_at=now,
        )
        path = self._location_manifest_path(location_id)
        if path.exists():
            return self.get_location(location_id)
        atomic_write_json(path, managed_location.to_dict())
        return managed_location

    def resolve_location(
        self,
        content_hash: str,
        *,
        prefer_managed: bool = True,
        verify: bool = True,
    ) -> ArtifactLocation:
        self.get_blob(content_hash)
        candidates = self.list_locations(content_hash=content_hash)
        candidates.sort(
            key=lambda item: (item.managed == prefer_managed, item.created_at), reverse=True
        )
        for location in candidates:
            if not Path(location.path).exists():
                continue
            if verify and not self.verify(location.id, structural=False).passed:
                continue
            return location
        raise ArtifactStoreError(f"Artifact {content_hash} has no available verified location")

    @staticmethod
    def _structural_checks(
        blob: ArtifactBlob, path: Path
    ) -> tuple[dict[str, Optional[bool]], list[str]]:
        checks: dict[str, Optional[bool]] = {}
        errors: list[str] = []
        fmt = blob.format.lower()
        kind = blob.artifact_kind

        def has_any(patterns: Sequence[str]) -> bool:
            if path.is_file():
                return any(path.match(pattern) for pattern in patterns)
            return any(any(path.glob(pattern)) for pattern in patterns)

        if fmt in {"hf", "huggingface", "transformers", "mlx"}:
            if path.is_file():
                checks["directory_layout"] = False
                errors.append(f"{fmt} artifacts must be directories")
            else:
                if kind == "adapter":
                    # PEFT stores named combined adapters in a subdirectory,
                    # while default adapters use the artifact root. Both are
                    # complete, loadable layouts.
                    checks["adapter_config"] = has_any(
                        ("adapter_config.json", "**/adapter_config.json")
                    )
                    checks["weights"] = has_any(
                        ("*.safetensors", "*.bin", "**/*.safetensors", "**/*.bin")
                    )
                    for label in ("adapter_config", "weights"):
                        if not checks[label]:
                            errors.append(f"Missing required adapter {label.replace('_', ' ')}")
                else:
                    checks["model_config"] = (path / "config.json").is_file()
                    checks["weights"] = has_any(("*.safetensors", "*.bin", "weights*.npz"))
                    checks["tokenizer_config"] = (path / "tokenizer_config.json").is_file()
                    checks["tokenizer_files"] = has_any(
                        ("tokenizer.json", "tokenizer.model", "vocab.json", "vocab.*")
                    )
                    for label, passed in checks.items():
                        if passed is False:
                            errors.append(f"Missing required {label.replace('_', ' ')}")
        elif fmt == "gguf":
            candidate = path if path.is_file() else next(iter(sorted(path.glob("*.gguf"))), None)
            checks["gguf_file"] = candidate is not None
            checks["gguf_magic"] = None
            if candidate is not None:
                with candidate.open("rb") as handle:
                    checks["gguf_magic"] = handle.read(4) == b"GGUF"
            if not checks["gguf_file"]:
                errors.append("Missing GGUF model file")
            elif not checks["gguf_magic"]:
                errors.append("GGUF model file has invalid magic bytes")
        elif fmt == "onnx":
            checks["onnx_file"] = (
                path.suffix.lower() == ".onnx" if path.is_file() else has_any(("*.onnx",))
            )
            if not checks["onnx_file"]:
                errors.append("Missing ONNX model file")
        else:
            checks["nonempty_payload"] = blob.file_count > 0 and blob.size_bytes >= 0
        return checks, errors

    def verify(
        self,
        location_or_hash: str,
        *,
        structural: bool = True,
        loader_probe: Optional[Callable[[Path, ArtifactBlob], bool | Mapping[str, Any]]] = None,
        round_trip_report: Optional[Mapping[str, Any]] = None,
    ) -> ArtifactVerification:
        """Verify content identity plus optional format/load/round-trip evidence.

        Loader and round-trip status are reported as unchecked unless real
        evidence is supplied; the service never fabricates backend support.
        """

        if location_or_hash.startswith("loc-"):
            location = self.get_location(location_or_hash)
        else:
            location = self.resolve_location(location_or_hash, verify=False)
        blob = self.get_blob(location.content_hash)
        path = Path(location.path)
        errors: list[str] = []
        matches = False
        if not path.exists():
            errors.append(f"Artifact location is missing: {path}")
        else:
            try:
                matches = hash_path(path).content_hash == blob.content_hash
            except Exception as exc:
                errors.append(str(exc))
            if not matches:
                errors.append("Artifact content hash does not match its immutable manifest")

        checks: dict[str, Optional[bool]] = {}
        if structural and matches:
            checks, structural_errors = self._structural_checks(blob, path)
            errors.extend(structural_errors)

        details: dict[str, Any] = {}
        loadability_checked = loader_probe is not None and matches
        if loadability_checked:
            try:
                probe_result = loader_probe(path, blob)
                if isinstance(probe_result, Mapping):
                    details["loader_probe"] = dict(probe_result)
                    load_passed = bool(probe_result.get("passed"))
                else:
                    load_passed = bool(probe_result)
                    details["loader_probe"] = {"passed": load_passed}
                checks["loadable"] = load_passed
                if not load_passed:
                    errors.append("Artifact failed its backend loadability probe")
            except ArtifactLoadProbeUnavailable as exc:
                loadability_checked = False
                details["loader_probe"] = {
                    "passed": None,
                    "available": False,
                    "reason": str(exc),
                }
            except Exception as exc:
                checks["loadable"] = False
                errors.append(f"Loadability probe failed: {exc}")

        round_trip_checked = round_trip_report is not None
        if round_trip_checked:
            round_trip_passed = bool(round_trip_report.get("passed"))
            checks["round_trip"] = round_trip_passed
            details["round_trip"] = dict(round_trip_report)
            if not round_trip_passed:
                errors.append("Artifact failed fixed-prompt round-trip verification")
        passed = matches and not errors and all(value is not False for value in checks.values())
        structural_checked = bool(structural and matches)
        if not passed:
            verification_level = "failed"
        elif round_trip_checked:
            verification_level = "round_trip_verified"
        elif loadability_checked:
            verification_level = "load_verified"
        elif structural_checked:
            verification_level = "structural_verified"
        else:
            verification_level = "hash_verified"
        return ArtifactVerification(
            content_hash=blob.content_hash,
            location_id=location.id,
            checked_at=self._now(),
            passed=passed,
            content_hash_matches=matches,
            structural_checks=checks,
            structural_checked=structural_checked,
            loadability_checked=loadability_checked,
            round_trip_checked=round_trip_checked,
            verification_level=verification_level,
            errors=tuple(errors),
            details=details,
        )

    def record_lineage(
        self,
        *,
        child_content_hash: str,
        parent_content_hashes: Sequence[str],
        relationship: str,
        operation_fingerprint: str,
    ) -> tuple[ArtifactEdge, ...]:
        """Persist an immutable ordered parent list for one derived blob."""

        child_content_hash = self.get_blob(child_content_hash).content_hash
        parents = [self.get_blob(item).content_hash for item in parent_content_hashes]
        if child_content_hash in parents:
            raise ArtifactStoreError("An artifact cannot be its own parent")
        for parent in parents:
            try:
                parent_graph = self.lineage(parent, direction="ancestors")
            except KeyError:
                continue
            if any(blob.content_hash == child_content_hash for blob in parent_graph.blobs):
                raise ArtifactStoreError("Lineage edge would create a cycle")
        edges = tuple(
            ArtifactEdge(
                parent_content_hash=parent,
                child_content_hash=child_content_hash,
                ordinal=index,
                relationship=relationship,
                operation_fingerprint=operation_fingerprint,
            )
            for index, parent in enumerate(parents)
        )
        path = self.metadata_dir / "edges" / f"{child_content_hash}.json"
        payload = {
            "child_content_hash": child_content_hash,
            "edges": [item.to_dict() for item in edges],
        }
        if path.exists():
            if read_json(path) != payload:
                raise ArtifactStoreError(
                    f"Artifact {child_content_hash} already has different immutable lineage"
                )
            return edges
        atomic_write_json(path, payload)
        return edges

    def _all_edges(self) -> list[ArtifactEdge]:
        values: list[ArtifactEdge] = []
        for path in sorted((self.metadata_dir / "edges").glob("*.json")):
            values.extend(ArtifactEdge(**item) for item in read_json(path).get("edges", []))
        return values

    def lineage(self, content_hash: str, *, direction: str = "ancestors") -> LineageGraph:
        """Return a recursive ancestor or descendant DAG rooted at a blob."""

        root_blob = self.get_blob(content_hash)
        if direction not in {"ancestors", "descendants"}:
            raise ValueError("direction must be 'ancestors' or 'descendants'")
        all_edges = self._all_edges()
        selected_edges: list[ArtifactEdge] = []
        visited = {root_blob.content_hash}
        frontier = [root_blob.content_hash]
        while frontier:
            current = frontier.pop(0)
            if direction == "ancestors":
                matches = [item for item in all_edges if item.child_content_hash == current]
                next_hashes = [
                    item.parent_content_hash
                    for item in sorted(matches, key=lambda item: item.ordinal)
                ]
            else:
                matches = [item for item in all_edges if item.parent_content_hash == current]
                matches.sort(key=lambda item: (item.child_content_hash, item.ordinal))
                next_hashes = [item.child_content_hash for item in matches]
            for edge in matches:
                if edge not in selected_edges:
                    selected_edges.append(edge)
            for next_hash in next_hashes:
                if next_hash not in visited:
                    visited.add(next_hash)
                    frontier.append(next_hash)
        blobs = tuple(self.get_blob(item) for item in sorted(visited))
        return LineageGraph(
            root_content_hash=root_blob.content_hash,
            direction=direction,
            blobs=blobs,
            edges=tuple(selected_edges),
        )

    def inventory(self) -> StorageInventory:
        usage = shutil.disk_usage(self.root)
        blobs = self.list_blobs()
        managed_count = sum(
            self._managed_payload_path(item.content_hash).exists() for item in blobs
        )
        return StorageInventory(
            root=str(self.root),
            generated_at=self._now(),
            blob_count=len(blobs),
            managed_blob_count=managed_count,
            referenced_location_count=sum(not item.managed for item in self.list_locations()),
            managed_bytes=_path_size(self.blobs_dir),
            metadata_bytes=_path_size(self.metadata_dir),
            staging_bytes=_path_size(self.staging_dir),
            trash_bytes=_path_size(self.trash_dir),
            free_bytes=usage.free,
            total_bytes=usage.total,
        )

    def disk_preflight(
        self,
        projected_output_bytes: int,
        *,
        override_reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """Apply the workstation's greater-of-20GB-or-10% free-space rule."""

        if projected_output_bytes < 0:
            raise ValueError("projected_output_bytes cannot be negative")
        usage = shutil.disk_usage(self.root)
        reserve = max(20 * 1024**3, int(usage.total * 0.10))
        projected_free = usage.free - projected_output_bytes
        passed = projected_free >= reserve
        overridden = not passed and bool(override_reason and override_reason.strip())
        return {
            "passed": passed or overridden,
            "overridden": overridden,
            "override_reason": override_reason if overridden else None,
            "projected_output_bytes": projected_output_bytes,
            "current_free_bytes": usage.free,
            "projected_free_bytes": projected_free,
            "required_reserve_bytes": reserve,
        }

    def _automatic_lineage_protections(self) -> frozenset[str]:
        # Preserve a managed parent while any managed child depends on it.
        managed = {
            blob.content_hash
            for blob in self.list_blobs()
            if self._managed_payload_path(blob.content_hash).exists()
        }
        return frozenset(
            edge.parent_content_hash
            for edge in self._all_edges()
            if edge.child_content_hash in managed
        )

    def _protection_reasons(
        self, content_hash: str, protections: CleanupProtections
    ) -> tuple[str, ...]:
        reasons = list(protections.reasons_for(content_hash))
        if (
            content_hash in self._automatic_lineage_protections()
            and "lineage_required" not in reasons
        ):
            reasons.append("lineage_required")
        return tuple(sorted(reasons))

    def preview_cleanup(
        self,
        *,
        protections: CleanupProtections = CleanupProtections(),
        minimum_blob_age: timedelta = timedelta(0),
        stale_staging_age: timedelta = timedelta(hours=24),
    ) -> CleanupPlan:
        """Persist a reviewed-plan candidate list without deleting anything."""

        now = self.clock().astimezone(timezone.utc)
        candidates: list[CleanupCandidate] = []
        protected: list[ProtectedArtifact] = []
        for blob in self.list_blobs():
            managed_dir = self.blobs_dir / blob.content_hash
            if not managed_dir.exists():
                continue
            age = max(0.0, (now - _parse_time(blob.created_at)).total_seconds())
            reasons = self._protection_reasons(blob.content_hash, protections)
            if reasons:
                protected.append(
                    ProtectedArtifact(
                        content_hash=blob.content_hash,
                        reasons=reasons,
                        size_bytes=_path_size(managed_dir),
                    )
                )
            elif age >= minimum_blob_age.total_seconds():
                candidates.append(
                    CleanupCandidate(
                        resource_type="blob",
                        identifier=blob.content_hash,
                        path=str(managed_dir),
                        reclaimable_bytes=_path_size(managed_dir),
                        age_seconds=age,
                    )
                )
        for staging in sorted(self.staging_dir.iterdir()):
            if staging.name in protections.active_staging:
                continue
            age = max(0.0, now.timestamp() - staging.stat().st_mtime)
            if age >= stale_staging_age.total_seconds():
                candidates.append(
                    CleanupCandidate(
                        resource_type="staging",
                        identifier=staging.name,
                        path=str(staging),
                        reclaimable_bytes=_path_size(staging),
                        age_seconds=age,
                    )
                )
        created_at = self._now()
        plan_payload = {
            "created_at": created_at,
            "candidates": [item.to_dict() for item in candidates],
            "protected": [item.to_dict() for item in protected],
            "protections": protections.to_dict(),
        }
        plan = CleanupPlan(
            id=f"cleanup-{fingerprint(plan_payload)[:24]}",
            created_at=created_at,
            candidates=tuple(candidates),
            protected=tuple(protected),
            protections=protections,
            reclaimable_bytes=sum(item.reclaimable_bytes for item in candidates),
        )
        atomic_write_json(self.metadata_dir / "cleanup_plans" / f"{plan.id}.json", plan.to_dict())
        return plan

    @staticmethod
    def _protections_from_dict(value: Mapping[str, Any]) -> CleanupProtections:
        return CleanupProtections(
            **{
                key: frozenset(value.get(key, []))
                for key in (
                    "active",
                    "pinned",
                    "promoted",
                    "serving",
                    "evaluation_referenced",
                    "lineage_required",
                    "active_staging",
                )
            }
        )

    def get_cleanup_plan(self, plan_id: str) -> CleanupPlan:
        payload = read_json(self.metadata_dir / "cleanup_plans" / f"{plan_id}.json")
        return CleanupPlan(
            id=payload["id"],
            created_at=payload["created_at"],
            candidates=tuple(CleanupCandidate(**item) for item in payload.get("candidates", [])),
            protected=tuple(
                ProtectedArtifact(**{**item, "reasons": tuple(item["reasons"])})
                for item in payload.get("protected", [])
            ),
            protections=self._protections_from_dict(payload.get("protections", {})),
            reclaimable_bytes=int(payload.get("reclaimable_bytes", 0)),
        )

    def trash_cleanup(
        self,
        plan_id: str,
        *,
        review_note: str,
        current_protections: CleanupProtections,
    ) -> CleanupResult:
        """Move reviewed candidates into restorable trash.

        Fresh protections are mandatory so a pin/promotion made after preview
        cannot be bypassed by executing a stale plan.
        """

        if not review_note.strip():
            raise ValueError("A non-empty review_note is required to execute cleanup")
        plan = self.get_cleanup_plan(plan_id)
        now = self._now()
        action_id = f"cleanup-action-{uuid.uuid4().hex}"
        trashed: list[str] = []
        skipped: dict[str, str] = {}
        for candidate in plan.candidates:
            source = Path(candidate.path)
            if not source.exists():
                skipped[candidate.identifier] = "missing"
                continue
            if candidate.resource_type == "blob":
                reasons = self._protection_reasons(candidate.identifier, current_protections)
                if reasons:
                    skipped[candidate.identifier] = f"protected: {', '.join(reasons)}"
                    continue
            elif (
                candidate.resource_type == "staging"
                and candidate.identifier in current_protections.active_staging
            ):
                skipped[candidate.identifier] = "protected: active_staging"
                continue
            trash_id = f"{self.clock().strftime('%Y%m%dT%H%M%S')}-{candidate.resource_type}-{candidate.identifier}"
            destination = self.trash_dir / trash_id
            suffix = 1
            while destination.exists():
                destination = self.trash_dir / f"{trash_id}-{suffix}"
                suffix += 1
            destination.mkdir(parents=True)
            os.replace(source, destination / "payload")
            atomic_write_json(
                destination / "tombstone.json",
                {
                    "trash_id": destination.name,
                    "resource_type": candidate.resource_type,
                    "identifier": candidate.identifier,
                    "original_path": str(source),
                    "trashed_at": now,
                    "plan_id": plan.id,
                    "action_id": action_id,
                    "review_note": review_note.strip(),
                    "size_bytes": candidate.reclaimable_bytes,
                },
            )
            trashed.append(candidate.identifier)
        result = CleanupResult(
            action_id=action_id,
            plan_id=plan.id,
            created_at=now,
            trashed=tuple(trashed),
            skipped=skipped,
            # Trash is reversible and occupies the same filesystem.  Space is
            # only reclaimed by purge after the retention period.
            reclaimed_bytes=0,
        )
        atomic_write_json(
            self.metadata_dir / "cleanup_actions" / f"{action_id}.json",
            {**result.to_dict(), "review_note": review_note.strip()},
        )
        return result

    def restore(self, content_hash: str) -> str:
        """Restore the newest trashed managed blob with this identity."""

        content_hash = _safe_hash(content_hash)
        matches: list[tuple[datetime, Path, dict[str, Any]]] = []
        for tombstone_path in self.trash_dir.glob("*/tombstone.json"):
            tombstone = read_json(tombstone_path)
            if (
                tombstone.get("resource_type") == "blob"
                and tombstone.get("identifier") == content_hash
            ):
                matches.append(
                    (_parse_time(tombstone["trashed_at"]), tombstone_path.parent, tombstone)
                )
        if not matches:
            raise KeyError(f"No trashed managed blob found for {content_hash}")
        _, entry, tombstone = sorted(matches, key=lambda item: item[0], reverse=True)[0]
        final = self.blobs_dir / content_hash
        if final.exists():
            raise ArtifactStoreError(f"Managed blob is already present: {final}")
        os.replace(entry / "payload", final)
        try:
            verification = self.verify(content_hash, structural=False)
            if not verification.passed:
                raise ArtifactIntegrityError("Restored blob failed content verification")
        except Exception:
            os.replace(final, entry / "payload")
            raise
        shutil.rmtree(entry, ignore_errors=True)
        action_id = f"restore-{uuid.uuid4().hex}"
        atomic_write_json(
            self.metadata_dir / "cleanup_actions" / f"{action_id}.json",
            {
                "action_id": action_id,
                "action": "restore",
                "content_hash": content_hash,
                "trash_id": tombstone["trash_id"],
                "created_at": self._now(),
            },
        )
        return str(final / "payload")

    def purge_trash(
        self,
        *,
        retention: timedelta = timedelta(days=TRASH_RETENTION_DAYS),
        protections: CleanupProtections = CleanupProtections(),
    ) -> dict[str, Any]:
        """Permanently remove trash only after the mandatory seven days."""

        if retention < timedelta(days=TRASH_RETENTION_DAYS):
            raise ValueError(
                f"Artifact trash retention cannot be shorter than {TRASH_RETENTION_DAYS} days"
            )
        now = self.clock().astimezone(timezone.utc)
        purged: list[str] = []
        skipped: dict[str, str] = {}
        reclaimed = 0
        for tombstone_path in sorted(self.trash_dir.glob("*/tombstone.json")):
            entry = tombstone_path.parent
            tombstone = read_json(tombstone_path)
            identifier = str(tombstone.get("identifier"))
            age = now - _parse_time(str(tombstone["trashed_at"]))
            if age < retention:
                skipped[identifier] = "retention"
                continue
            if tombstone.get("resource_type") == "blob":
                reasons = self._protection_reasons(identifier, protections)
                if reasons:
                    skipped[identifier] = f"protected: {', '.join(reasons)}"
                    continue
            size = _path_size(entry)
            shutil.rmtree(entry)
            reclaimed += size
            purged.append(identifier)
        result = {
            "created_at": self._now(),
            "purged": purged,
            "skipped": skipped,
            "reclaimed_bytes": reclaimed,
            "retention_seconds": int(retention.total_seconds()),
        }
        atomic_write_json(
            self.metadata_dir / "cleanup_actions" / f"purge-{uuid.uuid4().hex}.json",
            result,
        )
        return result

    def export_bundle(
        self,
        content_hash: str,
        destination: Path | str,
        *,
        replay_identity: Optional[Mapping[str, Any]] = None,
        dataset_identity: Optional[Mapping[str, Any]] = None,
        qualification: Optional[Mapping[str, Any]] = None,
        verification: Optional[Mapping[str, Any]] = None,
        license_metadata: Optional[Mapping[str, Any]] = None,
        model_card: Optional[str] = None,
    ) -> PortableExportBundle:
        """Atomically create a self-contained local directory export."""

        blob = self.get_blob(content_hash)
        source_location = self.resolve_location(content_hash)
        source = Path(source_location.path)
        destination_path = Path(destination).expanduser().resolve()
        if source == destination_path or source in destination_path.parents:
            raise ArtifactStoreError("Export destination cannot be inside the source artifact")
        if destination_path.exists():
            existing_manifest = destination_path / "bundle-manifest.json"
            if existing_manifest.is_file():
                payload = read_json(existing_manifest)
                if payload.get("source_content_hash") == content_hash:
                    if not self._verify_bundle_files(destination_path, payload):
                        raise ArtifactIntegrityError(
                            f"Existing export bundle has changed: {destination_path}"
                        )
                    digest = hash_path(destination_path)
                    return PortableExportBundle(
                        id=str(payload["bundle_id"]),
                        source_content_hash=content_hash,
                        path=str(destination_path),
                        created_at=str(payload["created_at"]),
                        manifest_hash=fingerprint(payload),
                        size_bytes=digest.size_bytes,
                        file_count=digest.file_count,
                        reused=True,
                    )
            raise FileExistsError(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{destination_path.name}.tmp-", dir=destination_path.parent)
        )
        try:
            model_destination = stage / "model"
            if source.is_file():
                model_destination.mkdir()
                copy_payload(source, model_destination / source.name)
            else:
                copy_payload(source, model_destination)
            atomic_write_json(stage / "artifact.json", blob.to_dict())
            atomic_write_json(stage / "lineage.json", self.lineage(content_hash).to_dict())
            atomic_write_json(stage / "replay.json", dict(replay_identity or {}))
            atomic_write_json(stage / "dataset.json", dict(dataset_identity or {}))
            atomic_write_json(stage / "qualification.json", dict(qualification or {}))
            atomic_write_json(stage / "artifact-verification.json", dict(verification or {}))
            atomic_write_json(stage / "license.json", dict(license_metadata or {}))
            card = model_card or self._default_model_card(blob, qualification, license_metadata)
            (stage / "MODEL_CARD.md").write_text(card.rstrip() + "\n", encoding="utf-8")
            content_digest = hash_path(stage)
            (stage / "SHA256SUMS").write_text(
                checksum_lines(content_digest.files), encoding="utf-8"
            )
            with_checksums = hash_path(stage)
            bundle_id = f"bundle-{fingerprint({'source': content_hash, 'files': [item.to_dict() for item in with_checksums.files]})[:24]}"
            manifest = {
                "format_version": 1,
                "bundle_id": bundle_id,
                "created_at": self._now(),
                "source_content_hash": content_hash,
                "source_artifact_kind": blob.artifact_kind,
                "source_format": blob.format,
                "source_dtype": blob.dtype,
                "source_quantization": blob.quantization,
                "source_location_kind": source_location.location_kind,
                "source_verification_level": (verification or {}).get(
                    "verification_level", "not_recorded"
                ),
                "payload_path": "model",
                "checksums_path": "SHA256SUMS",
                "files": [item.to_dict() for item in with_checksums.files],
            }
            atomic_write_json(stage / "bundle-manifest.json", manifest)
            os.replace(stage, destination_path)
            final_digest = hash_path(destination_path)
            bundle = PortableExportBundle(
                id=bundle_id,
                source_content_hash=content_hash,
                path=str(destination_path),
                created_at=manifest["created_at"],
                manifest_hash=fingerprint(manifest),
                size_bytes=final_digest.size_bytes,
                file_count=final_digest.file_count,
            )
            atomic_write_json(self.metadata_dir / "exports" / f"{bundle.id}.json", bundle.to_dict())
            return bundle
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    @staticmethod
    def _verify_bundle_files(bundle_path: Path, manifest: Mapping[str, Any]) -> bool:
        """Verify the immutable files listed by a portable bundle manifest."""

        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            return False
        for item in files:
            if not isinstance(item, Mapping):
                return False
            relative = Path(str(item.get("path") or ""))
            if relative.is_absolute() or ".." in relative.parts:
                return False
            candidate = bundle_path / relative
            if not candidate.is_file():
                return False
            if candidate.stat().st_size != int(item.get("size_bytes", -1)):
                return False
            if hash_file(candidate) != item.get("sha256"):
                return False
        return True

    @staticmethod
    def _default_model_card(
        blob: ArtifactBlob,
        qualification: Optional[Mapping[str, Any]],
        license_metadata: Optional[Mapping[str, Any]],
    ) -> str:
        qualification_status = (qualification or {}).get("decision", "not provided")
        license_name = (license_metadata or {}).get("license", "not provided")
        quantization = blob.quantization or "none"
        method = blob.quantization_method or "not applicable"
        return f"""# Halo Forge Model Artifact

- Content hash: `{blob.content_hash}`
- Artifact kind: `{blob.artifact_kind}`
- Format: `{blob.format}`
- Dtype: `{blob.dtype or 'unspecified'}`
- Quantization: `{quantization}`
- Quantization method: `{method}`
- Qualification decision: `{qualification_status}`
- License: `{license_name}`

This bundle was exported locally by Halo Forge. Verify `SHA256SUMS` before use.
Post-training quantization is reported as such; this bundle does not claim QAT.
"""


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactLoadProbeUnavailable",
    "ArtifactProtectedError",
    "ArtifactStore",
    "ArtifactStoreError",
    "DEFAULT_ARTIFACT_ROOT",
    "TRASH_RETENTION_DAYS",
    "probe_artifact_loadability",
]
