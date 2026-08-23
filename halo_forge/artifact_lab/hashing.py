"""Deterministic hashing and atomic metadata helpers for Artifact Lab.

The artifact content hash covers payload bytes and logical relative paths.  It
does not cover timestamps, permissions, or Artifact Lab's own manifests, so a
referenced run output and an adopted managed copy have the same identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

HASH_ALGORITHM = "sha256"
_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class HashedFile:
    """One file in a deterministic artifact digest."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContentDigest:
    """Content identity and inventory for a file or directory payload."""

    content_hash: str
    size_bytes: int
    file_count: int
    files: tuple[HashedFile, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": HASH_ALGORITHM,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "files": [item.to_dict() for item in self.files],
        }


def canonical_json(value: Any) -> str:
    """Serialize a JSON-shaped value for stable fingerprints."""

    def default(item: Any) -> Any:
        if is_dataclass(item):
            return asdict(item)
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        if hasattr(item, "to_dict"):
            return item.to_dict()
        raise TypeError(f"Cannot canonically encode {type(item).__name__}")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=default,
    )


def fingerprint(value: Any) -> str:
    """Return a SHA-256 fingerprint for a canonical JSON-shaped value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_file(path: Path | str) -> str:
    """Hash one regular file, following a file symlink to its bytes."""

    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(f"Artifact payload is not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_payload_files(root: Path) -> Iterator[tuple[str, Path]]:
    """Yield logical paths without following directory symlinks.

    File symlinks are intentionally dereferenced.  This lets a Hugging Face
    cache snapshot be adopted into a self-contained managed copy while keeping
    the same content identity.  Directory symlinks are rejected because their
    traversal boundary is ambiguous and can unexpectedly ingest unrelated
    content.
    """

    if root.is_file():
        yield "payload", root
        return
    if not root.is_dir():
        raise FileNotFoundError(f"Artifact payload does not exist: {root}")

    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(directory_names):
            candidate = current_path / name
            if candidate.is_symlink():
                raise ValueError(
                    f"Directory symlinks are not supported in artifact payloads: {candidate}"
                )
        for name in sorted(file_names):
            candidate = current_path / name
            if not candidate.is_file():
                raise ValueError(f"Artifact payload contains a non-regular file: {candidate}")
            yield candidate.relative_to(root).as_posix(), candidate


def hash_path(path: Path | str) -> ContentDigest:
    """Hash a file or directory deterministically.

    A single file is identified by its bytes, independent of its source file
    name.  A directory additionally includes each relative path, preventing two
    layouts with the same concatenated bytes from colliding.
    """

    source = Path(path).expanduser().resolve(strict=True)
    entries: list[HashedFile] = []
    for relative_path, file_path in _iter_payload_files(source):
        entries.append(
            HashedFile(
                path=relative_path,
                sha256=hash_file(file_path),
                size_bytes=file_path.stat().st_size,
            )
        )
    entries.sort(key=lambda item: item.path)
    if not entries:
        raise ValueError(f"Artifact payload is empty: {source}")

    if source.is_file():
        content_hash = entries[0].sha256
    else:
        content_hash = fingerprint(
            [
                {"path": item.path, "sha256": item.sha256, "size_bytes": item.size_bytes}
                for item in entries
            ]
        )
    return ContentDigest(
        content_hash=content_hash,
        size_bytes=sum(item.size_bytes for item in entries),
        file_count=len(entries),
        files=tuple(entries),
    )


def copy_payload(source: Path | str, destination: Path | str) -> Path:
    """Materialize a payload without preserving external symlink dependencies."""

    import shutil

    source_path = Path(source).expanduser().resolve(strict=True)
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    if source_path.is_file():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    else:
        # copytree follows file symlinks by default.  hash_path has already
        # rejected directory symlinks, so this cannot escape through a subtree.
        shutil.copytree(source_path, destination_path, symlinks=False)
    return destination_path


def atomic_write_json(path: Path | str, value: Any) -> None:
    """Durably replace a JSON file using a temporary sibling."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some filesystems do not support directory fsync.  The atomic
            # replacement still provides the publication boundary.
            pass
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object in {path}")
    return dict(value)


def checksum_lines(files: Iterable[HashedFile], *, prefix: str = "") -> str:
    """Render a portable sha256sum-compatible checksum file."""

    normalized_prefix = prefix.rstrip("/")
    return "".join(
        f"{item.sha256}  {normalized_prefix + '/' if normalized_prefix else ''}{item.path}\n"
        for item in files
    )


__all__ = [
    "ContentDigest",
    "HASH_ALGORITHM",
    "HashedFile",
    "atomic_write_json",
    "canonical_json",
    "checksum_lines",
    "copy_payload",
    "fingerprint",
    "hash_file",
    "hash_path",
    "read_json",
]
