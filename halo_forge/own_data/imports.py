"""Managed upload staging and content-addressed raw-source publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from halo_forge.workstation_jobs.resources import (
    DEFAULT_RESERVE_FRACTION,
    MINIMUM_DISK_RESERVE_BYTES,
    WorkstationCapacity,
)

from .inspection import fingerprint_path
from .registry import TRAINING_SCENARIOS, TrainingScenarioRegistry

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}


class InsufficientDiskCapacityError(ValueError):
    """Raised when a managed import would violate workstation disk reserves."""

    def __init__(self, message: str, *, forecast: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.forecast = dict(forecast)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_relative_path(value: str) -> str:
    raw = str(value or "")
    if not raw or "\x00" in raw or "\\" in raw:
        raise ValueError("relative_path must be a non-empty portable path")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative_path cannot be absolute or contain traversal components")
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_extension(content: bytes, path_hint: Optional[str], kind: str) -> str:
    suffix = Path(path_hint or "").suffix.lower()
    allowed = _IMAGE_SUFFIXES if kind == "image" else _AUDIO_SUFFIXES
    if suffix in allowed:
        return suffix
    if kind == "image":
        if content.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if content.startswith((b"GIF87a", b"GIF89a")):
            return ".gif"
        if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
            return ".webp"
    else:
        if content.startswith(b"RIFF") and content[8:12] == b"WAVE":
            return ".wav"
        if content.startswith(b"fLaC"):
            return ".flac"
        if content.startswith(b"OggS"):
            return ".ogg"
        if content.startswith(b"ID3") or content.startswith(b"\xff\xfb"):
            return ".mp3"
    raise ValueError(f"Hugging Face {kind} bytes use an unrecognized media format")


def _materialize_huggingface_media(
    value: Any,
    *,
    kind: str,
    field_name: str,
    row_index: int,
    staging: Path,
) -> str:
    path_hint: Optional[str] = None
    content: Optional[bytes] = None
    if isinstance(value, Mapping):
        raw_path = value.get("path")
        path_hint = str(raw_path) if raw_path else None
        raw_bytes = value.get("bytes")
        if isinstance(raw_bytes, (bytes, bytearray, memoryview)):
            content = bytes(raw_bytes)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        content = bytes(value)
    elif isinstance(value, str):
        path_hint = value

    source_path = Path(path_hint).expanduser() if path_hint else None
    if content is None and source_path is not None and source_path.is_file():
        if source_path.is_symlink():
            raise ValueError("Hugging Face media cache returned a symbolic-link asset")
        content = source_path.read_bytes()
    if not content:
        raise ValueError(
            f"Hugging Face field {field_name!r} did not expose immutable {kind} bytes"
        )
    suffix = _media_extension(content, path_hint, kind)
    safe_field = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in field_name
    ).strip("-") or kind
    relative = Path("assets") / kind / f"{row_index:09d}-{safe_field}{suffix}"
    target = staging / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return relative.as_posix()


class DatasetImportManager:
    def __init__(
        self,
        database: Any,
        *,
        imports_root: Path | str,
        datasets_root: Path | str,
        scenario_registry: TrainingScenarioRegistry = TRAINING_SCENARIOS,
        capacity_probe: Optional[Callable[[Path], WorkstationCapacity]] = None,
    ) -> None:
        self.db = database
        self.imports_root = Path(imports_root).expanduser().resolve()
        self.sources_root = Path(datasets_root).expanduser().resolve() / "sources"
        self.imports_root.mkdir(parents=True, exist_ok=True)
        self.sources_root.mkdir(parents=True, exist_ok=True)
        self.registry = scenario_registry
        self.capacity_probe = capacity_probe

    @staticmethod
    def _override_reason(session: Any, supplied: Optional[str] = None) -> Optional[str]:
        explicit = str(supplied or "").strip()
        if explicit:
            return explicit
        capacity = dict(session.metadata.get("capacity") or {})
        return str(capacity.get("override_reason") or "").strip() or None

    def disk_forecast(
        self,
        *,
        root: Path,
        projected_disk_bytes: int,
        phase: str,
        override_reason: Optional[str] = None,
        projection_known: bool = True,
    ) -> Dict[str, Any]:
        """Forecast one import phase using the workstation reserve policy.

        Capacity is intentionally optional for transport-neutral/unit-test use.
        The desktop/browser runtime supplies the durable scheduler's probe, so
        production imports use the same greater-of-20GB-or-10% policy as other
        workstation work.  An unavailable probe is reported, never fabricated.
        """

        projected = int(projected_disk_bytes)
        if projected < 0:
            raise ValueError("projected_disk_bytes cannot be negative")
        normalized_reason = str(override_reason or "").strip() or None
        common: Dict[str, Any] = {
            "phase": str(phase),
            "path": str(root.expanduser().resolve(strict=False)),
            "projection_known": bool(projection_known),
            "projected_disk_bytes": projected,
            "policy": "greater_of_20gb_or_10_percent",
        }
        if not projection_known:
            common["warning"] = (
                "Source size is unavailable; projected bytes include only the known minimum."
            )
        if self.capacity_probe is None:
            return {
                **common,
                "available": False,
                "allowed": True,
                "capacity_sufficient": None,
                "overridden": False,
                "override_reason": None,
                "current_free_bytes": None,
                "projected_free_bytes": None,
                "required_reserve_bytes": None,
                "blockers": ["disk_capacity_unavailable"],
                "warning": "Workstation disk capacity could not be measured in this runtime.",
            }
        try:
            capacity = self.capacity_probe(root)
        except Exception as exc:
            return {
                **common,
                "available": False,
                "allowed": True,
                "capacity_sufficient": None,
                "overridden": False,
                "override_reason": None,
                "current_free_bytes": None,
                "projected_free_bytes": None,
                "required_reserve_bytes": None,
                "blockers": ["disk_capacity_unavailable"],
                "warning": f"Workstation disk capacity probe failed: {type(exc).__name__}: {exc}",
            }
        disk = capacity.disk
        if disk is None:
            return {
                **common,
                "available": False,
                "allowed": True,
                "capacity_sufficient": None,
                "overridden": False,
                "override_reason": None,
                "current_free_bytes": None,
                "projected_free_bytes": None,
                "required_reserve_bytes": None,
                "blockers": ["disk_capacity_unavailable"],
                "warning": "Workstation disk capacity is unavailable.",
                "sampled_at": capacity.sampled_at.isoformat(),
            }
        reserve = max(
            MINIMUM_DISK_RESERVE_BYTES,
            math.ceil(disk.total_bytes * DEFAULT_RESERVE_FRACTION),
        )
        projected_free = disk.free_bytes - projected
        sufficient = projected_free >= reserve
        overridden = bool(not sufficient and normalized_reason)
        return {
            **common,
            "available": True,
            "allowed": sufficient or overridden,
            "capacity_sufficient": sufficient,
            "overridden": overridden,
            "override_reason": normalized_reason if overridden else None,
            "current_free_bytes": disk.free_bytes,
            "projected_free_bytes": projected_free,
            "required_reserve_bytes": reserve,
            "blockers": [] if sufficient or overridden else ["insufficient_disk"],
            "sampled_at": capacity.sampled_at.isoformat(),
        }

    @staticmethod
    def _metadata_with_forecast(
        metadata: Mapping[str, Any], forecast: Mapping[str, Any]
    ) -> Dict[str, Any]:
        value = dict(metadata)
        capacity = dict(value.get("capacity") or {})
        forecasts = dict(capacity.get("forecasts") or {})
        forecasts[str(forecast["phase"])] = dict(forecast)
        capacity["forecasts"] = forecasts
        if forecast.get("overridden"):
            reason = str(forecast.get("override_reason") or "").strip()
            capacity["override_reason"] = reason
            history = list(capacity.get("override_history") or [])
            event = {
                "phase": forecast["phase"],
                "reason": reason,
                "projected_disk_bytes": forecast["projected_disk_bytes"],
                "required_reserve_bytes": forecast.get("required_reserve_bytes"),
                "recorded_at": _now(),
            }
            identity = (event["phase"], event["reason"], event["projected_disk_bytes"])
            if not history or (
                history[-1].get("phase"),
                history[-1].get("reason"),
                history[-1].get("projected_disk_bytes"),
            ) != identity:
                history.append(event)
            capacity["override_history"] = history[-100:]
        value["capacity"] = capacity
        return value

    def _record_forecast(self, session: Any, forecast: Mapping[str, Any]) -> Any:
        return self.db.update_dataset_import(
            session.id,
            metadata=self._metadata_with_forecast(session.metadata, forecast),
        )

    def _require_disk_capacity(
        self,
        session: Any,
        *,
        root: Path,
        projected_disk_bytes: int,
        phase: str,
        override_reason: Optional[str] = None,
        projection_known: bool = True,
    ) -> Dict[str, Any]:
        forecast = self.disk_forecast(
            root=root,
            projected_disk_bytes=projected_disk_bytes,
            phase=phase,
            override_reason=self._override_reason(session, override_reason),
            projection_known=projection_known,
        )
        self._record_forecast(session, forecast)
        if not forecast["allowed"]:
            projected = forecast.get("projected_free_bytes")
            reserve = forecast.get("required_reserve_bytes")
            raise InsufficientDiskCapacityError(
                "Dataset import would leave "
                f"{projected} bytes free, below the required {reserve}-byte reserve. "
                "Free disk space or retry with a non-empty capacity_override_reason.",
                forecast=forecast,
            )
        return forecast

    @staticmethod
    def _huggingface_size(info: Any) -> Optional[int]:
        direct = getattr(info, "usedStorage", None)
        if isinstance(direct, int) and direct >= 0:
            return direct
        sizes = []
        for sibling in list(getattr(info, "siblings", None) or []):
            value = getattr(sibling, "size", None)
            if isinstance(value, int) and value >= 0:
                sizes.append(value)
        return sum(sizes) if sizes else None

    @staticmethod
    def _normalize_kind(value: str) -> str:
        aliases = {
            "reference": "workstation_path",
            "path": "workstation_path",
            "desktop": "desktop_reference",
            "example": "upload",
        }
        return aliases.get(str(value).strip().lower(), str(value).strip().lower())

    def create(self, payload: Mapping[str, Any]) -> Any:
        requested_kind = str(payload.get("source_kind") or "").strip().lower()
        kind = self._normalize_kind(requested_kind)
        source_uri = payload.get("source_uri") or payload.get("path")
        supplied_override = str(payload.get("capacity_override_reason") or "").strip() or None
        scenario_revision_id = str(payload.get("scenario_revision_id") or "").strip() or None
        if scenario_revision_id:
            self.registry.get(scenario_revision_id)
        if kind in {"workstation_path", "desktop_reference"}:
            if not source_uri:
                raise ValueError("source_uri is required for a workstation reference")
            selected_source = Path(str(source_uri)).expanduser()
            fingerprint, size_bytes, file_count = fingerprint_path(selected_source)
            source = selected_source.resolve()
            record = self.db.create_dataset_import(
                source_kind=kind,
                display_name=str(payload.get("name") or source.name),
                source_uri=str(source),
                scenario_revision_id=scenario_revision_id,
                expected_size_bytes=size_bytes,
                metadata={"requested_source_kind": requested_kind, "file_count": file_count},
                expires_at=(datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
                status="ready",
            )
            return self.db.update_dataset_import(record.id, fingerprint=fingerprint)
        if kind == "huggingface":
            if not source_uri:
                raise ValueError("source_uri is required for a Hugging Face dataset")
            revision = str(payload.get("revision") or "").strip()
            if not revision:
                raise ValueError("Hugging Face imports require a pinned revision")
            try:
                from huggingface_hub import HfApi  # type: ignore

                info = HfApi().dataset_info(str(source_uri), revision=revision)
                resolved_revision = str(info.sha)
            except Exception as exc:
                raise ValueError(
                    f"could not resolve Hugging Face revision {revision!r} to an immutable commit: {exc}"
                ) from exc
            if not resolved_revision:
                raise ValueError("Hugging Face did not return an immutable dataset commit")
            supplied_size = payload.get("expected_size_bytes")
            expected_size = (
                int(supplied_size)
                if supplied_size is not None
                else self._huggingface_size(info)
            )
            if expected_size is not None and expected_size < 0:
                raise ValueError("expected_size_bytes cannot be negative")
            forecast = self.disk_forecast(
                root=self.imports_root,
                projected_disk_bytes=int(expected_size or 0),
                phase="huggingface_staging",
                override_reason=supplied_override,
                projection_known=expected_size is not None,
            )
            if not forecast["allowed"]:
                raise InsufficientDiskCapacityError(
                    "The pinned Hugging Face source cannot be staged without violating "
                    "the workstation disk reserve. Free space or provide a non-empty "
                    "capacity_override_reason.",
                    forecast=forecast,
                )
            metadata = self._metadata_with_forecast(
                {
                    "requested_source_kind": requested_kind,
                    "size_estimate_source": (
                        "request"
                        if supplied_size is not None
                        else "huggingface_metadata"
                        if expected_size is not None
                        else "unavailable"
                    ),
                },
                forecast,
            )
            return self.db.create_dataset_import(
                source_kind=kind,
                display_name=str(payload.get("name") or source_uri),
                source_uri=str(source_uri),
                source_config=(str(payload["config"]) if payload.get("config") else None),
                source_split=(str(payload["split"]) if payload.get("split") else None),
                source_revision=revision,
                resolved_revision=resolved_revision,
                scenario_revision_id=scenario_revision_id,
                expected_size_bytes=expected_size,
                metadata=metadata,
                expires_at=(datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
                status="ready",
            )
        if kind != "upload":
            raise ValueError(f"unsupported dataset import source kind: {requested_kind}")
        identifier = uuid.uuid4().hex
        supplied_size = payload.get("expected_size_bytes")
        expected_size = int(supplied_size) if supplied_size is not None else None
        if expected_size is not None and expected_size < 0:
            raise ValueError("expected_size_bytes cannot be negative")
        metadata: Dict[str, Any] = {"requested_source_kind": requested_kind}
        if supplied_override:
            metadata["capacity"] = {"override_reason": supplied_override}
        if expected_size:
            forecast = self.disk_forecast(
                root=self.imports_root,
                projected_disk_bytes=expected_size,
                phase="upload_staging",
                override_reason=supplied_override,
            )
            if not forecast["allowed"]:
                raise InsufficientDiskCapacityError(
                    "The upload cannot be staged without violating the workstation disk "
                    "reserve. Free space or provide a non-empty capacity_override_reason.",
                    forecast=forecast,
                )
            metadata = self._metadata_with_forecast(metadata, forecast)
        staging = self.imports_root / identifier
        staging.mkdir(parents=True, exist_ok=False)
        record = self.db.create_dataset_import(
            source_kind="upload",
            display_name=str(payload.get("name") or "Uploaded dataset"),
            scenario_revision_id=scenario_revision_id,
            expected_size_bytes=expected_size,
            staging_path=str(staging),
            metadata=metadata,
            expires_at=(datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
            import_id=identifier,
            status="draft",
        )
        if requested_kind == "example":
            if not scenario_revision_id:
                raise ValueError("example imports require scenario_revision_id")
            filename, fixture_files = self.registry.template_files(
                scenario_revision_id,
                str(payload.get("example_id") or "") or None,
            )
            for relative_path, content in fixture_files.items():
                file_record = self.create_file(
                    record.id,
                    {
                        "relative_path": relative_path,
                        "size_bytes": len(content),
                        "content_type": (
                            "application/x-ndjson"
                            if relative_path == filename
                            else "image/png"
                            if relative_path.endswith(".png")
                            else "audio/wav"
                        ),
                        "content_hash": hashlib.sha256(content).hexdigest(),
                    },
                )
                self.write_chunk(
                    record.id,
                    file_record.id,
                    content,
                    start=0,
                    end=len(content) - 1,
                    total=len(content),
                )
        return self.db.get_dataset_import(record.id)

    def create_file(self, import_id: str, payload: Mapping[str, Any]) -> Any:
        session = self.db.get_dataset_import(import_id)
        if session is None:
            raise KeyError(import_id)
        if session.source_kind != "upload" or session.status not in {
            "draft",
            "uploading",
            "ready",
        }:
            raise ValueError("files can only be added to an active upload import")
        relative = safe_relative_path(str(payload.get("relative_path") or ""))
        size = int(payload.get("size_bytes", -1))
        if size <= 0:
            raise ValueError("empty upload files are not supported")
        expected = payload.get("content_hash") or payload.get("sha256")
        if expected is not None:
            expected = str(expected).lower()
            if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
                raise ValueError("content_hash must be a hexadecimal SHA-256 digest")
        existing_files = self.db.list_dataset_import_files(import_id)
        existing = next(
            (item for item in existing_files if item.relative_path == relative),
            None,
        )
        if existing is not None:
            same_identity = (
                int(existing.size_bytes) == size
                and (
                    expected is None
                    or existing.expected_sha256 is None
                    or existing.expected_sha256 == expected
                )
            )
            if same_identity and existing.status != "failed":
                return existing
            raise ValueError(
                "an upload file with this relative path already exists with different content"
            )
        outstanding = size + sum(
            max(0, int(item.size_bytes) - int(item.received_bytes))
            for item in existing_files
            if item.status not in {"complete", "cancelled"}
        )
        self._require_disk_capacity(
            session,
            root=self.imports_root,
            projected_disk_bytes=outstanding,
            phase="upload_staging",
            override_reason=(
                str(payload.get("capacity_override_reason") or "").strip() or None
            ),
        )
        root = Path(session.staging_path or "").resolve()
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("upload path escapes its staging directory") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.is_symlink():
            raise ValueError("upload target cannot be a symbolic link")
        if not target.exists():
            target.touch(mode=0o600)
        record = self.db.create_dataset_import_file(
            import_id=import_id,
            relative_path=relative,
            size_bytes=size,
            staging_path=str(target),
            media_type=(str(payload.get("content_type")) if payload.get("content_type") else None),
            expected_sha256=expected,
        )
        self.db.update_dataset_import(import_id, status="uploading")
        return record

    def write_chunk(
        self,
        import_id: str,
        file_id: str,
        content: bytes,
        *,
        start: int,
        end: int,
        total: int,
        chunk_sha256: Optional[str] = None,
    ) -> Any:
        session = self.db.get_dataset_import(import_id)
        file_record = self.db.get_dataset_import_file(file_id)
        if session is None or file_record is None or file_record.import_id != import_id:
            raise KeyError(file_id)
        if total != file_record.size_bytes or start < 0 or end < start:
            raise ValueError("Content-Range does not match the declared file size")
        if len(content) != end - start + 1:
            raise ValueError("chunk length does not match Content-Range")
        if chunk_sha256:
            supplied_hash = chunk_sha256.lower()
            # The browser contract may repeat the declared whole-file hash on
            # every range. A distinct value is interpreted as a chunk hash.
            if supplied_hash != file_record.expected_sha256:
                if hashlib.sha256(content).hexdigest() != supplied_hash:
                    raise ValueError("chunk checksum does not match X-Content-SHA256")
        target = Path(file_record.staging_path)
        if target.is_symlink():
            raise ValueError("upload staging file became a symbolic link")
        if start < file_record.received_bytes:
            # Idempotent retry of an already-committed chunk.
            with target.open("rb") as handle:
                handle.seek(start)
                prior = handle.read(len(content))
            if end < file_record.received_bytes and prior == content:
                return file_record
            raise ValueError("upload range overlaps already committed content")
        if start != file_record.received_bytes:
            raise ValueError(f"next upload range must start at byte {file_record.received_bytes}")
        with target.open("r+b") as handle:
            if target.stat().st_size > start:
                # Discard an uncommitted tail left by a crash between fsync and
                # the catalog update; the caller is resuming at the durable boundary.
                handle.truncate(start)
            handle.seek(start)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        received = end + 1
        changes: Dict[str, Any] = {
            "received_bytes": received,
            "status": "uploading",
        }
        if received == total:
            digest = _sha256_file(target)
            if file_record.expected_sha256 and digest != file_record.expected_sha256:
                self.db.update_dataset_import_file(
                    file_id,
                    status="failed",
                    received_bytes=received,
                    content_sha256=digest,
                    error="file checksum mismatch",
                    completed_at=_now(),
                )
                self.db.update_dataset_import(
                    import_id, status="failed", error="file checksum mismatch"
                )
                raise ValueError("uploaded file checksum does not match the declared content hash")
            changes.update(
                status="complete",
                content_sha256=digest,
                completed_at=_now(),
            )
        updated = self.db.update_dataset_import_file(file_id, **changes)
        files = self.db.list_dataset_import_files(import_id)
        uploaded = sum(item.received_bytes for item in files)
        complete = bool(files) and all(item.status == "complete" for item in files)
        self.db.update_dataset_import(
            import_id,
            received_size_bytes=uploaded,
            expected_size_bytes=sum(item.size_bytes for item in files),
            status="ready" if complete else "uploading",
            error=None,
        )
        return updated

    def materialize_huggingface(
        self,
        import_id: str,
        *,
        progress: Optional[Any] = None,
        cancelled: Optional[Any] = None,
    ) -> Path:
        session = self.db.get_dataset_import(import_id)
        if session is None:
            raise KeyError(import_id)
        if session.source_kind != "huggingface":
            raise ValueError("import is not a Hugging Face source")
        self._require_disk_capacity(
            session,
            root=self.imports_root,
            projected_disk_bytes=int(session.expected_size_bytes or 0),
            phase="huggingface_staging",
            projection_known=session.expected_size_bytes is not None,
        )
        try:
            from datasets import Audio, Image, load_dataset  # type: ignore
        except ImportError as exc:
            raise ValueError("Hugging Face dataset import requires the datasets package") from exc
        staging = self.imports_root / import_id
        if staging.is_symlink():
            raise ValueError("Hugging Face import staging cannot be a symbolic link")
        staging.mkdir(parents=True, exist_ok=True)
        target = staging / "data.jsonl"
        # An interrupted attempt is restarted at the immutable source pin.
        # Remove only its unpublished tail so stale media cannot enter the
        # next fingerprint or preview.
        shutil.rmtree(staging / "assets", ignore_errors=True)
        target.unlink(missing_ok=True)
        dataset = load_dataset(
            str(session.source_uri),
            session.source_config,
            split=session.source_split or "train",
            revision=session.resolved_revision,
            streaming=True,
        )
        media_fields: Dict[str, str] = {}
        for field_name, feature in dict(getattr(dataset, "features", {}) or {}).items():
            feature_name = type(feature).__name__.lower()
            kind = (
                "image"
                if isinstance(feature, Image) or feature_name == "image"
                else "audio"
                if isinstance(feature, Audio) or feature_name == "audio"
                else None
            )
            if not kind:
                continue
            media_fields[str(field_name)] = kind
            try:
                dataset = dataset.cast_column(
                    str(field_name), Image(decode=False) if kind == "image" else Audio(decode=False)
                )
            except Exception as exc:
                raise ValueError(
                    f"could not request undecoded Hugging Face {kind} bytes for "
                    f"field {field_name!r}: {exc}"
                ) from exc
        digest = hashlib.sha256()
        count = 0
        with target.open("wb") as handle:
            for record in dataset:
                if cancelled and cancelled():
                    raise RuntimeError("Hugging Face materialization cancelled")
                normalized = dict(record)
                for field_name, kind in media_fields.items():
                    if normalized.get(field_name) is None:
                        continue
                    normalized[field_name] = _materialize_huggingface_media(
                        normalized[field_name],
                        kind=kind,
                        field_name=field_name,
                        row_index=count,
                        staging=staging,
                    )
                line = (json.dumps(normalized, ensure_ascii=False, default=str) + "\n").encode(
                    "utf-8"
                )
                handle.write(line)
                digest.update(line)
                count += 1
                if progress and count % 100 == 0:
                    progress(count)
            handle.flush()
            os.fsync(handle.fileno())
        refreshed_session = self.db.get_dataset_import(import_id) or session
        self.db.update_dataset_import(
            import_id,
            staging_path=str(staging),
            fingerprint=digest.hexdigest(),
            expected_size_bytes=target.stat().st_size,
            received_size_bytes=target.stat().st_size,
            file_count=1,
            metadata={
                **refreshed_session.metadata,
                "row_count": count,
                "materialized_media_fields": media_fields,
            },
            status="ready",
        )
        return staging

    def source_path(
        self,
        import_id: str,
        *,
        progress: Optional[Any] = None,
        cancelled: Optional[Any] = None,
    ) -> Path:
        session = self.db.get_dataset_import(import_id)
        if session is None:
            raise KeyError(import_id)
        if session.source_kind in {"workstation_path", "desktop_reference"}:
            return Path(str(session.source_uri)).expanduser().resolve()
        if session.source_kind == "huggingface" and not session.staging_path:
            return self.materialize_huggingface(
                import_id, progress=progress, cancelled=cancelled
            )
        root = Path(str(session.staging_path or "")).resolve()
        files = self.db.list_dataset_import_files(import_id)
        if len(files) == 1 and Path(files[0].staging_path).suffix.lower() in {
            ".json",
            ".jsonl",
            ".jl",
            ".csv",
            ".tsv",
            ".parquet",
        }:
            return Path(files[0].staging_path)
        return root

    def _verify_published_bundle(self, target: Path, fingerprint: str) -> Path:
        manifest_path = target / "manifest.json"
        data_root = target / "data"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"managed source bundle is corrupt: {exc}") from exc
        if manifest.get("content_hash") != fingerprint or not data_root.exists():
            raise ValueError("managed source bundle identity does not match its directory")
        files = manifest.get("files") or []
        for item in files:
            relative = safe_relative_path(str(item.get("path") or ""))
            path = (data_root / relative).resolve()
            try:
                path.relative_to(data_root.resolve())
            except ValueError as exc:
                raise ValueError("managed source manifest contains an unsafe path") from exc
            if not path.is_file() or _sha256_file(path) != item.get("sha256"):
                raise ValueError(f"managed source file failed checksum verification: {relative}")
        return data_root

    def publish(
        self, import_id: str, *, override_reason: Optional[str] = None
    ) -> Tuple[Path, str]:
        session = self.db.get_dataset_import(import_id)
        if session is None:
            raise KeyError(import_id)
        if session.managed_source_path and Path(session.managed_source_path).exists():
            return Path(session.managed_source_path), str(session.fingerprint)
        source = self.source_path(import_id)
        fingerprint, size_bytes, file_count = fingerprint_path(source)
        target = self.sources_root / fingerprint
        if target.exists():
            data_root = self._verify_published_bundle(target, fingerprint)
            self.db.update_dataset_import(
                import_id,
                managed_source_path=str(data_root),
                fingerprint=fingerprint,
                expected_size_bytes=size_bytes,
                file_count=file_count,
                status="published",
                completed_at=_now(),
            )
            return data_root, fingerprint
        # The staging copy remains until its reviewed seven-day cleanup, so a
        # managed publication needs a second full source copy plus its manifest.
        # A small fixed allowance covers the JSON manifest and directory entries.
        publication_bytes = size_bytes + 1024 * 1024
        self._require_disk_capacity(
            session,
            root=self.sources_root,
            projected_disk_bytes=publication_bytes,
            phase="managed_publication",
            override_reason=override_reason,
        )
        temp = Path(tempfile.mkdtemp(prefix=f".{fingerprint[:12]}-", dir=self.sources_root))
        try:
            data_root = temp / "data"
            if source.is_file():
                data_root.mkdir(parents=True)
                shutil.copy2(source, data_root / source.name)
            else:
                shutil.copytree(source, data_root)
            managed_files = [
                {
                    "path": item.relative_to(data_root).as_posix(),
                    "sha256": _sha256_file(item),
                    "size_bytes": item.stat().st_size,
                }
                for item in sorted(data_root.rglob("*"))
                if item.is_file()
            ]
            manifest = {
                "format_version": 1,
                "content_hash": fingerprint,
                "source_kind": session.source_kind,
                "source_uri": session.source_uri,
                "resolved_revision": session.resolved_revision,
                "size_bytes": size_bytes,
                "file_count": file_count,
                "files": managed_files,
                "published_at": _now(),
            }
            (temp / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temp, target)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        self.db.update_dataset_import(
            import_id,
            managed_source_path=str(target / "data"),
            fingerprint=fingerprint,
            expected_size_bytes=size_bytes,
            file_count=file_count,
            status="published",
            completed_at=_now(),
        )
        return target / "data", fingerprint

    def capacity_status(self, session: Any) -> Dict[str, Any]:
        """Return current staging/publication forecasts and actionable readiness."""

        stored = dict(session.metadata.get("capacity") or {})
        override_reason = str(stored.get("override_reason") or "").strip() or None
        stages: Dict[str, Dict[str, Any]] = {}
        declared_upload_bytes = 0
        if session.source_kind == "upload" and session.status not in {
            "published",
            "expired",
        }:
            files = self.db.list_dataset_import_files(session.id)
            declared_upload_bytes = (
                sum(int(item.size_bytes) for item in files)
                if files
                else int(session.expected_size_bytes or 0)
            )
            outstanding = (
                sum(
                    max(0, int(item.size_bytes) - int(item.received_bytes))
                    for item in files
                    if item.status not in {"complete", "cancelled"}
                )
                if files
                else max(
                    0,
                    int(session.expected_size_bytes or 0)
                    - int(session.received_size_bytes or 0),
                )
            )
            if outstanding:
                stages["upload_staging"] = self.disk_forecast(
                    root=self.imports_root,
                    projected_disk_bytes=outstanding,
                    phase="upload_staging",
                    override_reason=override_reason,
                )
        elif (
            session.source_kind == "huggingface"
            and not session.staging_path
            and session.status not in {"published", "expired"}
        ):
            stages["huggingface_staging"] = self.disk_forecast(
                root=self.imports_root,
                projected_disk_bytes=int(session.expected_size_bytes or 0),
                phase="huggingface_staging",
                override_reason=override_reason,
                projection_known=session.expected_size_bytes is not None,
            )
        if (
            session.source_kind in {"upload", "huggingface"}
            and not session.managed_source_path
            and session.status not in {"published", "expired"}
        ):
            stages["managed_publication"] = self.disk_forecast(
                root=self.sources_root,
                projected_disk_bytes=(
                    int(session.expected_size_bytes or declared_upload_bytes or 0)
                    + 1024 * 1024
                ),
                phase="managed_publication",
                override_reason=override_reason,
                projection_known=(
                    session.expected_size_bytes is not None or declared_upload_bytes > 0
                ),
            )
        blockers = sorted(
            {
                str(blocker)
                for forecast in stages.values()
                if not forecast.get("allowed")
                for blocker in forecast.get("blockers") or []
            }
        )
        warnings = sorted(
            {
                str(forecast["warning"])
                for forecast in stages.values()
                if forecast.get("warning")
            }
        )
        requires_override = any(
            not forecast.get("allowed") and forecast.get("available")
            for forecast in stages.values()
        )
        return {
            "ready": not blockers,
            "requires_override": requires_override,
            "blockers": blockers,
            "warnings": warnings,
            "remedy": (
                "Free disk space or retry with a non-empty capacity_override_reason."
                if requires_override
                else None
            ),
            "stages": stages,
            "last_recorded_forecasts": dict(stored.get("forecasts") or {}),
            "override_history": list(stored.get("override_history") or []),
        }

    def cleanup_expired(
        self, *, now: Optional[datetime] = None, apply: bool = False
    ) -> Dict[str, Any]:
        cutoff = now or datetime.now(timezone.utc)
        removed = []
        reclaimed = 0
        items = []
        for session in self.db.list_dataset_imports(limit=10000):
            if not session.expires_at or session.status in {"published", "expired"}:
                continue
            try:
                expires = datetime.fromisoformat(session.expires_at)
            except ValueError:
                continue
            if expires > cutoff:
                continue
            staging = Path(session.staging_path) if session.staging_path else None
            size = 0
            if staging and staging.exists() and staging.is_relative_to(self.imports_root):
                size = sum(
                    item.stat().st_size for item in staging.rglob("*") if item.is_file()
                )
            reclaimed += size
            items.append(
                {
                    "id": session.id,
                    "name": session.display_name or "Dataset import",
                    "size_bytes": size,
                    "expires_at": session.expires_at,
                    "resource_type": "dataset_import_staging",
                }
            )
            if apply:
                if staging and staging.exists() and staging.is_relative_to(self.imports_root):
                    shutil.rmtree(staging, ignore_errors=True)
                self.db.update_dataset_import(session.id, status="expired")
                removed.append(session.id)
        return {
            "status": "completed" if apply else "preview",
            "items": items,
            "removed_import_ids": removed,
            "reclaimable_bytes": reclaimed,
            "reclaimed_bytes": reclaimed if apply else 0,
            "retention_days": 7,
        }


__all__ = ["DatasetImportManager", "safe_relative_path"]
