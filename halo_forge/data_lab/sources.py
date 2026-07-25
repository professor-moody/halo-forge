"""Local and pinned Hugging Face source loading and fingerprinting."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .errors import SourceError

SUPPORTED_SUFFIXES = {".json", ".jsonl", ".jl", ".csv", ".tsv", ".parquet"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
ASSET_FIELDS = ("image", "image_path", "audio", "audio_path")


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def content_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_path(path: Path | str) -> str:
    """Hash file contents, or a directory's relative paths and contents."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise SourceError(f"Source path does not exist: {resolved}")
    if resolved.is_file():
        return hash_file(resolved)
    digest = hashlib.sha256()
    files = sorted(p for p in resolved.rglob("*") if p.is_file())
    for file_path in files:
        relative = file_path.relative_to(resolved).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hash_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    path: Optional[str] = None
    repo_id: Optional[str] = None
    config: Optional[str] = None
    split: str = "train"
    revision: Optional[str] = None
    data_files: Optional[Any] = None

    def __post_init__(self) -> None:
        normalized = self.kind.strip().lower().replace("hf", "huggingface")
        object.__setattr__(self, "kind", normalized)
        if normalized == "local":
            if not self.path:
                raise SourceError("A local source requires path")
        elif normalized == "huggingface":
            if not self.repo_id:
                raise SourceError("A Hugging Face source requires repo_id")
            if not self.revision or self.revision in {"main", "master", "latest"}:
                raise SourceError(
                    "Hugging Face sources must use a pinned revision, not main/latest"
                )
        else:
            raise SourceError("Source kind must be 'local' or 'huggingface'")

    @classmethod
    def from_value(cls, value: "SourceSpec | Mapping[str, Any]") -> "SourceSpec":
        return value if isinstance(value, cls) else cls(**dict(value))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssetFingerprint:
    field: str
    reference: str
    resolved_path: Optional[str]
    fingerprint: Optional[str]
    size_bytes: Optional[int]
    missing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SourceSnapshot:
    spec: SourceSpec
    records: List[Dict[str, Any]]
    fingerprint: str
    assets: List[AssetFingerprint] = field(default_factory=list)
    size_bytes: int = 0
    file_count: int = 0

    def to_dict(self, *, include_records: bool = False) -> Dict[str, Any]:
        data = {
            "spec": self.spec.to_dict(),
            "fingerprint": self.fingerprint,
            "assets": [asset.to_dict() for asset in self.assets],
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "row_count": len(self.records),
        }
        if include_records:
            data["records"] = self.records
        return data


def _coerce_rows(value: Any, *, source: Path) -> List[Dict[str, Any]]:
    if isinstance(value, Mapping):
        for candidate in ("records", "data", "rows", "items"):
            if isinstance(value.get(candidate), list):
                value = value[candidate]
                break
        else:
            value = [value]
    if not isinstance(value, list):
        raise SourceError(f"Expected an object or array in {source}")
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            row = {"text": row}
        rows.append(dict(row))
    return rows


def load_local_file(path: Path | str) -> List[Dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise SourceError(f"Unsupported source format {suffix!r}: {source}")
    if suffix == ".json":
        try:
            return _coerce_rows(json.loads(source.read_text(encoding="utf-8")), source=source)
        except json.JSONDecodeError as exc:
            raise SourceError(f"Invalid JSON in {source}: {exc}") from exc
    if suffix in {".jsonl", ".jl"}:
        rows: List[Dict[str, Any]] = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SourceError(f"Invalid JSON on {source}:{line_number}: {exc}") from exc
                rows.extend(_coerce_rows([row], source=source))
        return rows
    if suffix in {".csv", ".tsv"}:
        with source.open(newline="", encoding="utf-8-sig") as handle:
            return [
                dict(row)
                for row in csv.DictReader(
                    handle, delimiter="\t" if suffix == ".tsv" else ","
                )
            ]
    try:
        pandas = importlib.import_module("pandas")
        frame = pandas.read_parquet(source)
    except ImportError as exc:
        raise SourceError(
            "Parquet loading requires pandas and a parquet engine such as pyarrow"
        ) from exc
    except Exception as exc:
        raise SourceError(f"Unable to load parquet source {source}: {exc}") from exc
    return [dict(row) for row in frame.to_dict(orient="records")]


def load_local(path: Path | str) -> tuple[List[Dict[str, Any]], int, int]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise SourceError(f"Source path does not exist: {source}")
    files = (
        [source]
        if source.is_file()
        else sorted(
            p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )
    )
    if not files:
        if source.is_dir():
            media = sorted(
                path
                for path in source.rglob("*")
                if path.is_file()
                and path.suffix.lower() in (_IMAGE_SUFFIXES | _AUDIO_SUFFIXES)
            )
            paired: List[Dict[str, Any]] = []
            for path in media:
                sidecar = path.with_suffix(".txt")
                if not sidecar.is_file():
                    continue
                relative = path.relative_to(source).as_posix()
                text = sidecar.read_text(encoding="utf-8").strip()
                if path.suffix.lower() in _IMAGE_SUFFIXES:
                    paired.append({"image": relative, "caption": text})
                else:
                    paired.append({"audio": relative, "transcript": text})
            if paired:
                size = sum(
                    path.stat().st_size + path.with_suffix(".txt").stat().st_size
                    for path in media
                    if path.with_suffix(".txt").is_file()
                )
                return paired, size, len(paired) * 2
        raise SourceError(f"No supported data files found under {source}")
    rows: List[Dict[str, Any]] = []
    for file_path in files:
        loaded = load_local_file(file_path)
        if source.is_dir():
            relative = file_path.relative_to(source).as_posix()
            for row in loaded:
                row.setdefault("_source_file", relative)
        rows.extend(loaded)
    return rows, sum(path.stat().st_size for path in files), len(files)


def _asset_reference(value: Any) -> Optional[str]:
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    if isinstance(value, Mapping):
        candidate = value.get("path") or value.get("filename")
        return os.fspath(candidate) if isinstance(candidate, (str, os.PathLike)) else None
    candidate = getattr(value, "filename", None)
    if isinstance(candidate, (str, os.PathLike)) and candidate:
        return os.fspath(candidate)
    return None


def _jsonable(value: Any) -> Any:
    """Preserve dataset feature payloads while converting media objects to paths."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    filename = getattr(value, "filename", None)
    if isinstance(filename, (str, os.PathLike)) and filename:
        return os.fspath(filename)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def fingerprint_assets(
    records: Sequence[Mapping[str, Any]], *, base_dir: Optional[Path | str] = None
) -> List[AssetFingerprint]:
    root = Path(base_dir).expanduser().resolve() if base_dir is not None else None
    found: Dict[tuple[str, str], AssetFingerprint] = {}
    for row in records:
        for field_name in ASSET_FIELDS:
            raw = row.get(field_name)
            values = raw if isinstance(raw, list) else [raw]
            for value in values:
                reference = _asset_reference(value)
                if not reference or reference.startswith(("http://", "https://", "data:")):
                    continue
                key = (field_name, reference)
                if key in found:
                    continue
                path = Path(reference).expanduser()
                if not path.is_absolute() and root is not None:
                    path = root / path
                path = path.resolve()
                exists = path.is_file()
                found[key] = AssetFingerprint(
                    field=field_name,
                    reference=reference,
                    resolved_path=str(path),
                    fingerprint=hash_file(path) if exists else None,
                    size_bytes=path.stat().st_size if exists else None,
                    missing=not exists,
                )
    return list(found.values())


def load_source(spec: SourceSpec | Mapping[str, Any]) -> SourceSnapshot:
    resolved = SourceSpec.from_value(spec)
    if resolved.kind == "local":
        source_path = Path(resolved.path or "").expanduser().resolve()
        records, size_bytes, file_count = load_local(source_path)
        root = source_path.parent if source_path.is_file() else source_path
        assets = fingerprint_assets(records, base_dir=root)
        fingerprint = content_hash(
            {
                "source": fingerprint_path(source_path),
                "assets": [
                    {
                        "field": asset.field,
                        "reference": asset.reference,
                        "fingerprint": asset.fingerprint,
                        "missing": asset.missing,
                    }
                    for asset in assets
                ],
            }
        )
        return SourceSnapshot(resolved, records, fingerprint, assets, size_bytes, file_count)

    try:
        datasets = importlib.import_module("datasets")
    except ImportError as exc:
        raise SourceError("Hugging Face loading requires the optional `datasets` package") from exc
    kwargs: Dict[str, Any] = {
        "split": resolved.split,
        "revision": resolved.revision,
    }
    if resolved.data_files is not None:
        kwargs["data_files"] = resolved.data_files
    try:
        dataset = datasets.load_dataset(resolved.repo_id, resolved.config, **kwargs)
    except Exception as exc:
        raise SourceError(
            f"Unable to load pinned Hugging Face source {resolved.repo_id}@{resolved.revision}: {exc}"
        ) from exc
    records = [_jsonable(dict(row)) for row in dataset]
    assets = fingerprint_assets(records)
    fingerprint = content_hash(
        {
            "spec": resolved.to_dict(),
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "records": records,
            "assets": [
                {
                    "field": asset.field,
                    "reference": asset.reference,
                    "fingerprint": asset.fingerprint,
                    "missing": asset.missing,
                }
                for asset in assets
            ],
        }
    )
    return SourceSnapshot(resolved, records, fingerprint, assets)


def verify_snapshot(snapshot: SourceSnapshot) -> bool:
    """Return whether the live source and all local assets still match."""
    for asset in snapshot.assets:
        if not asset.resolved_path:
            continue
        path = Path(asset.resolved_path)
        if not path.is_file() or hash_file(path) != asset.fingerprint:
            return False
    if snapshot.spec.kind == "huggingface":
        # The remote identity is pinned; local cached media was checked above.
        return True
    try:
        return load_source(snapshot.spec).fingerprint == snapshot.fingerprint
    except SourceError:
        return False


__all__ = [
    "ASSET_FIELDS",
    "AssetFingerprint",
    "SourceSnapshot",
    "SourceSpec",
    "content_hash",
    "fingerprint_assets",
    "fingerprint_path",
    "hash_file",
    "load_local",
    "load_local_file",
    "load_source",
    "stable_json",
    "verify_snapshot",
]
