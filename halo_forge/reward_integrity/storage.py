"""Atomic checksummed storage for training-signal and reward-audit bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


@dataclass(frozen=True)
class PublishedBundle:
    path: str
    manifest_hash: str
    content_hash: str
    size_bytes: int
    manifest: Dict[str, Any]


class RewardIntegrityStorage:
    """Publish immutable bundles with same-filesystem staging + rename."""

    def __init__(self, root: str | Path | None = None):
        supplied = Path(root or (Path.home() / ".halo-forge")).expanduser()
        # Public/API callers historically supplied the explicit audit
        # directory while CLI callers supplied the Halo Forge base root.  Keep
        # both forms equivalent and retain one canonical base for signal data.
        if supplied.name == "reward-audits" and supplied.parent.name == "evaluations":
            self.root = supplied.parent.parent
            self.audit_root = supplied
        else:
            self.root = supplied
            self.audit_root = supplied / "evaluations" / "reward-audits"

    def signal_path(self, run_id: str, segment_id: str, trace_hash: str) -> Path:
        return self.root / "training-signals" / run_id / segment_id / trace_hash

    def audit_path(self, audit_id: str) -> Path:
        return self.audit_root / audit_id

    @staticmethod
    def _encode(name: str, value: Any) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        if name.endswith(".jsonl"):
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
                raise TypeError(f"{name} must be a sequence for JSONL encoding")
            text = "".join(canonical_json(item) + "\n" for item in value)
            return text.encode("utf-8")
        return (canonical_json(value) + "\n").encode("utf-8")

    def publish(
        self,
        destination: str | Path,
        documents: Mapping[str, Any],
        *,
        identity: Mapping[str, Any],
    ) -> PublishedBundle:
        target = Path(destination)
        if not documents:
            raise ValueError("at least one bundle document is required")
        for name in documents:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or name == "manifest.json":
                raise ValueError(f"unsafe or reserved bundle path: {name}")
        encoded = {name: self._encode(name, value) for name, value in documents.items()}
        entries = {
            name: {"sha256": sha256_bytes(payload), "size_bytes": len(payload)}
            for name, payload in sorted(encoded.items())
        }
        bundle_content_hash = content_hash({"identity": dict(identity), "files": entries})
        manifest = {
            "format": "halo-forge-reward-integrity-bundle-v1",
            "identity": dict(identity),
            "content_hash": bundle_content_hash,
            "files": entries,
        }
        manifest_payload = (canonical_json(manifest) + "\n").encode("utf-8")
        manifest_hash = sha256_bytes(manifest_payload)
        total_size = len(manifest_payload) + sum(len(value) for value in encoded.values())

        if target.exists():
            verified = self.verify(target)
            if verified.manifest_hash != manifest_hash:
                raise FileExistsError(
                    f"immutable bundle already exists with different content: {target}"
                )
            return verified

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
        staging.mkdir(parents=False, exist_ok=False)
        try:
            for name, payload in encoded.items():
                output = staging / name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(payload)
                with output.open("rb") as handle:
                    os.fsync(handle.fileno())
            manifest_file = staging / "manifest.json"
            manifest_file.write_bytes(manifest_payload)
            with manifest_file.open("rb") as handle:
                os.fsync(handle.fileno())
            directory_fd = os.open(staging, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            os.replace(staging, target)
            parent_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return PublishedBundle(
            path=str(target),
            manifest_hash=manifest_hash,
            content_hash=bundle_content_hash,
            size_bytes=total_size,
            manifest=manifest,
        )

    def publish_streaming(
        self,
        destination: str | Path,
        documents: Mapping[str, Any],
        *,
        jsonl_documents: Mapping[str, Iterable[Any]],
        identity: Mapping[str, Any],
    ) -> PublishedBundle:
        """Publish JSONL iterators without assembling their encoded payloads.

        This is used for exhaustive reward-audit evidence.  The resulting
        manifest and file bytes are identical to :meth:`publish` for the same
        ordered values, so existing bundle verification remains compatible.
        """

        target = Path(destination)
        names = [*documents, *jsonl_documents]
        if not names:
            raise ValueError("at least one bundle document is required")
        if len(names) != len(set(names)):
            raise ValueError("bundle document names must be unique")
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or name == "manifest.json":
                raise ValueError(f"unsafe or reserved bundle path: {name}")

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
        staging.mkdir(parents=False, exist_ok=False)
        entries: Dict[str, Dict[str, Any]] = {}
        try:
            for name, value in documents.items():
                payload = self._encode(name, value)
                output = staging / name
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(payload)
                with output.open("rb") as handle:
                    os.fsync(handle.fileno())
                entries[name] = {
                    "sha256": sha256_bytes(payload),
                    "size_bytes": len(payload),
                }
            for name, values in jsonl_documents.items():
                output = staging / name
                output.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with output.open("wb") as handle:
                    for item in values:
                        payload = (canonical_json(item) + "\n").encode("utf-8")
                        handle.write(payload)
                        digest.update(payload)
                        size += len(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                entries[name] = {"sha256": digest.hexdigest(), "size_bytes": size}

            entries = dict(sorted(entries.items()))
            bundle_content_hash = content_hash(
                {"identity": dict(identity), "files": entries}
            )
            manifest = {
                "format": "halo-forge-reward-integrity-bundle-v1",
                "identity": dict(identity),
                "content_hash": bundle_content_hash,
                "files": entries,
            }
            manifest_payload = (canonical_json(manifest) + "\n").encode("utf-8")
            manifest_hash = sha256_bytes(manifest_payload)
            total_size = len(manifest_payload) + sum(
                int(value["size_bytes"]) for value in entries.values()
            )

            if target.exists():
                verified = self.verify(target)
                if verified.manifest_hash != manifest_hash:
                    raise FileExistsError(
                        f"immutable bundle already exists with different content: {target}"
                    )
                shutil.rmtree(staging, ignore_errors=True)
                return verified

            manifest_file = staging / "manifest.json"
            manifest_file.write_bytes(manifest_payload)
            with manifest_file.open("rb") as handle:
                os.fsync(handle.fileno())
            directory_fd = os.open(staging, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            os.replace(staging, target)
            parent_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return PublishedBundle(
            path=str(target),
            manifest_hash=manifest_hash,
            content_hash=bundle_content_hash,
            size_bytes=total_size,
            manifest=manifest,
        )

    def verify(self, path: str | Path) -> PublishedBundle:
        target = Path(path)
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"bundle manifest is missing: {manifest_path}")
        payload = manifest_path.read_bytes()
        try:
            manifest = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"bundle manifest is invalid: {manifest_path}") from exc
        if manifest.get("format") != "halo-forge-reward-integrity-bundle-v1":
            raise ValueError("unsupported reward-integrity bundle format")
        total_size = len(payload)
        entries = dict(manifest.get("files") or {})
        if not entries:
            raise ValueError("bundle manifest contains no files")
        for name, descriptor in entries.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"unsafe path in bundle manifest: {name}")
            source = target / name
            if not source.is_file():
                raise ValueError(f"bundle file is missing: {name}")
            digest = hashlib.sha256()
            size = 0
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != descriptor.get("sha256"):
                raise ValueError(f"bundle checksum mismatch: {name}")
            if size != int(descriptor.get("size_bytes", -1)):
                raise ValueError(f"bundle size mismatch: {name}")
            total_size += size
        expected_content_hash = content_hash(
            {"identity": manifest.get("identity") or {}, "files": entries}
        )
        if expected_content_hash != manifest.get("content_hash"):
            raise ValueError("bundle content hash mismatch")
        return PublishedBundle(
            path=str(target),
            manifest_hash=sha256_bytes(payload),
            content_hash=expected_content_hash,
            size_bytes=total_size,
            manifest=manifest,
        )


__all__ = [
    "PublishedBundle",
    "RewardIntegrityStorage",
    "canonical_json",
    "content_hash",
]
