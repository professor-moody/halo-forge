"""Immutable, content-addressed Dataset Lab version storage."""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from .errors import VersionError
from .identity import (
    RecordIdentity,
    attach_identities,
    build_version_lineage,
    strip_internal_identity,
)
from .recipe import Recipe, RecipeResult
from .sources import (
    AssetFingerprint,
    SourceSnapshot,
    SourceSpec,
    content_hash,
    hash_file,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _safe_id(value: str, label: str) -> str:
    cleaned = value.strip()
    if not cleaned or cleaned in {".", ".."} or any(char in cleaned for char in "/\\\0"):
        raise VersionError(f"Invalid {label}: {value!r}")
    return cleaned


@dataclass(frozen=True)
class DatasetVersion:
    dataset_id: str
    version_id: str
    path: str
    content_hash: str
    recipe_hash: str
    source_fingerprint: str
    schema: Optional[str]
    materialized_assets: bool
    split_counts: Dict[str, int]
    row_count: int
    created_at: str
    reused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _replace_refs(value: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace_refs(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _replace_refs(item, mapping) for key, item in value.items()}
    return value


class VersionStore:
    """Publish complete versions atomically beneath ``root/<dataset>/<version>``."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _version_path(self, dataset_id: str, version_id: str) -> Path:
        return self.root / _safe_id(dataset_id, "dataset ID") / _safe_id(version_id, "version ID")

    def publish(
        self,
        *,
        dataset_id: str,
        recipe: Recipe | Mapping[str, Any] | Path | str,
        result: RecipeResult,
        source: SourceSnapshot,
        materialize_assets: bool = False,
        parent_version_id: Optional[str] = None,
    ) -> DatasetVersion:
        resolved_recipe = Recipe.from_value(recipe)
        dataset_id = _safe_id(dataset_id, "dataset ID")
        records = copy.deepcopy(result.records)
        splits = copy.deepcopy(result.splits or {"train": records})
        asset_map: Dict[str, str] = {}
        asset_payload = [asset.to_dict() for asset in source.assets]
        identity = {
            "dataset_id": dataset_id,
            "recipe_hash": resolved_recipe.fingerprint,
            "source_fingerprint": source.fingerprint,
            # Build-only lineage envelopes must never alter immutable dataset
            # content identity.
            "records": [strip_internal_identity(row) for row in records],
            "splits": {
                name: [strip_internal_identity(row) for row in rows]
                for name, rows in splits.items()
            },
            "materialized_assets": materialize_assets,
            "asset_fingerprints": [
                {"reference": asset.reference, "fingerprint": asset.fingerprint}
                for asset in source.assets
            ],
            "parent_version_id": parent_version_id,
        }
        version_content_hash = content_hash(identity)
        version_id = version_content_hash[:16]
        final_path = self._version_path(dataset_id, version_id)
        if final_path.exists():
            version = self.get(dataset_id, version_id)
            if version.content_hash != version_content_hash:
                raise VersionError(f"Content-address collision at {final_path}")
            return DatasetVersion(**{**version.to_dict(), "reused": True})

        dataset_dir = final_path.parent
        dataset_dir.mkdir(parents=True, exist_ok=True)
        temp_path = Path(tempfile.mkdtemp(prefix=f".{version_id}.tmp-", dir=dataset_dir))
        try:
            if materialize_assets:
                assets_dir = temp_path / "assets"
                assets_dir.mkdir()
                for asset in source.assets:
                    if asset.missing or not asset.resolved_path or not asset.fingerprint:
                        raise VersionError(f"Cannot materialize missing asset: {asset.reference}")
                    source_path = Path(asset.resolved_path)
                    if not source_path.is_file() or hash_file(source_path) != asset.fingerprint:
                        raise VersionError(
                            f"Cannot materialize changed or missing asset: {asset.reference}"
                        )
                    filename = f"{asset.fingerprint[:12]}-{source_path.name}"
                    destination = assets_dir / filename
                    if not destination.exists():
                        shutil.copy2(source_path, destination)
                    asset_map[asset.reference] = f"assets/{filename}"
                records = [_replace_refs(row, asset_map) for row in records]
                splits = {
                    name: [_replace_refs(row, asset_map) for row in rows]
                    for name, rows in splits.items()
                }

            records, splits, lineage = build_version_lineage(records, splits, version_id=version_id)
            splits_dir = temp_path / "splits"
            splits_dir.mkdir()
            _write_jsonl(temp_path / "records.jsonl", records)
            for name, rows in splits.items():
                _write_jsonl(splits_dir / f"{_safe_id(name, 'split name')}.jsonl", rows)
            _write_jsonl(temp_path / "lineage.jsonl", (entry.to_dict() for entry in lineage))
            _write_jsonl(
                temp_path / "rejected.jsonl",
                (strip_internal_identity(row) for row in result.rejected),
            )
            _write_jsonl(
                temp_path / "quarantined.jsonl",
                (strip_internal_identity(row) for row in result.quarantined),
            )
            _write_json(temp_path / "recipe.json", resolved_recipe.to_dict())
            _write_json(temp_path / "stats.json", result.statistics)
            _write_json(
                temp_path / "provenance.json", [entry.to_dict() for entry in result.provenance]
            )
            _write_json(temp_path / "contamination.json", result.contamination)

            artifact_hashes: Dict[str, str] = {}
            for artifact in sorted(path for path in temp_path.rglob("*") if path.is_file()):
                artifact_hashes[artifact.relative_to(temp_path).as_posix()] = hash_file(artifact)
            manifest = {
                "format_version": 2,
                "status": "complete",
                "dataset_id": dataset_id,
                "version_id": version_id,
                "created_at": _utc_now(),
                "content_hash": version_content_hash,
                "recipe_hash": resolved_recipe.fingerprint,
                "schema": resolved_recipe.schema,
                "source": source.to_dict(include_records=False),
                "source_fingerprint": source.fingerprint,
                "asset_fingerprints": asset_payload,
                "materialized_assets": materialize_assets,
                "asset_mapping": asset_map,
                "parent_version_id": parent_version_id,
                "row_count": len(records),
                "split_counts": {name: len(rows) for name, rows in splits.items()},
                "artifact_hashes": artifact_hashes,
                "lineage": {
                    "path": "lineage.jsonl",
                    "identity_scheme": "content-origin-v1",
                    "record_count": len(lineage),
                    "virtual": False,
                },
            }
            _write_json(temp_path / "manifest.json", manifest)
            # Atomic rename is the publication point; incomplete temp directories are never listed.
            os.replace(temp_path, final_path)
        except Exception:
            shutil.rmtree(temp_path, ignore_errors=True)
            raise
        return self.get(dataset_id, version_id)

    def _find(self, version_id: str, dataset_id: Optional[str] = None) -> Path:
        if dataset_id:
            path = self._version_path(dataset_id, version_id)
            if path.is_dir():
                return path
            raise VersionError(f"Unknown dataset version {dataset_id}/{version_id}")
        matches = [
            path
            for path in self.root.glob(f"*/{_safe_id(version_id, 'version ID')}")
            if path.is_dir()
        ]
        if not matches:
            raise VersionError(f"Unknown dataset version {version_id}")
        if len(matches) > 1:
            raise VersionError(f"Version ID {version_id!r} is ambiguous; supply dataset_id")
        return matches[0]

    def get(self, dataset_id: str, version_id: str) -> DatasetVersion:
        path = self._find(version_id, dataset_id)
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VersionError(f"Invalid version manifest at {path}: {exc}") from exc
        return DatasetVersion(
            dataset_id=manifest["dataset_id"],
            version_id=manifest["version_id"],
            path=str(path),
            content_hash=manifest["content_hash"],
            recipe_hash=manifest["recipe_hash"],
            source_fingerprint=manifest["source_fingerprint"],
            schema=manifest.get("schema"),
            materialized_assets=bool(manifest.get("materialized_assets")),
            split_counts=dict(manifest.get("split_counts", {})),
            row_count=int(manifest.get("row_count", 0)),
            created_at=manifest["created_at"],
        )

    def get_any(self, version_id: str, dataset_id: Optional[str] = None) -> DatasetVersion:
        path = self._find(version_id, dataset_id)
        return self.get(path.parent.name, path.name)

    def list(self, dataset_id: Optional[str] = None) -> List[DatasetVersion]:
        roots = (
            [self.root / _safe_id(dataset_id, "dataset ID")]
            if dataset_id
            else sorted(self.root.iterdir())
        )
        versions: List[DatasetVersion] = []
        for dataset_path in roots:
            if not dataset_path.is_dir():
                continue
            for version_path in sorted(dataset_path.iterdir()):
                if (
                    version_path.is_dir()
                    and not version_path.name.startswith(".")
                    and (version_path / "manifest.json").is_file()
                ):
                    versions.append(self.get(dataset_path.name, version_path.name))
        return sorted(versions, key=lambda version: version.created_at, reverse=True)

    def load_records(
        self,
        version_id: str,
        *,
        dataset_id: Optional[str] = None,
        split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return list(self.iter_records(version_id, dataset_id=dataset_id, split=split))

    def iter_records(
        self,
        version_id: str,
        *,
        dataset_id: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield immutable records in stored order without materializing a split."""

        path = self._find(version_id, dataset_id)
        file_path = (
            path / "records.jsonl"
            if split is None
            else path / "splits" / f"{_safe_id(split, 'split name')}.jsonl"
        )
        if not file_path.is_file():
            raise VersionError(f"Dataset version has no split {split!r}")
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def preview_records(
        self,
        version_id: str,
        *,
        dataset_id: Optional[str] = None,
        split: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Read a bounded page without materializing an immutable version in memory."""

        if offset < 0 or limit < 0 or limit > 1000:
            raise ValueError("offset must be non-negative and limit must be between 0 and 1000")
        path = self._find(version_id, dataset_id)
        split_name = _safe_id(split, "split name") if split is not None else None
        file_path = (
            path / "records.jsonl"
            if split_name is None
            else path / "splits" / f"{split_name}.jsonl"
        )
        if not file_path.is_file():
            raise VersionError(f"Dataset version has no split {split!r}")
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            total = int(
                manifest.get("row_count", 0)
                if split_name is None
                else dict(manifest.get("split_counts") or {}).get(split_name, 0)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise VersionError(f"Invalid version manifest at {path}: {exc}") from exc

        records: List[Dict[str, Any]] = []
        if limit == 0:
            return records, total
        record_index = 0
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if record_index < offset:
                    record_index += 1
                    continue
                if len(records) >= limit:
                    break
                records.append(json.loads(line))
                record_index += 1
        return records, total

    def load_lineage(
        self,
        version_id: str,
        *,
        dataset_id: Optional[str] = None,
        split: Optional[str] = None,
    ) -> List[RecordIdentity]:
        """Load persisted lineage or synthesize deterministic legacy identity."""

        return list(self.iter_lineage(version_id, dataset_id=dataset_id, split=split))

    def iter_lineage(
        self,
        version_id: str,
        *,
        dataset_id: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Iterator[RecordIdentity]:
        """Yield lineage in record/split order using bounded process memory.

        V2 lineage is persisted in record order.  Split order can differ, so a
        temporary on-disk SQLite index performs that ordering without retaining
        every identity in Python memory.  Legacy versions have no sidecar and
        retain the historical in-memory identity synthesis fallback.
        """

        path = self._find(version_id, dataset_id)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        lineage_path = path / str((manifest.get("lineage") or {}).get("path") or "lineage.jsonl")
        if lineage_path.is_file():
            if split is None:
                row_count = int(manifest.get("row_count", 0))
                with lineage_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        identity = RecordIdentity.from_value(json.loads(line))
                        if identity.record_index < row_count:
                            yield identity
                return

            split_name = _safe_id(split, "split name")
            fd, index_name = tempfile.mkstemp(prefix="halo-forge-lineage-index-", suffix=".sqlite3")
            os.close(fd)
            index_path = Path(index_name)
            connection: Optional[sqlite3.Connection] = None
            try:
                connection = sqlite3.connect(index_path)
                connection.execute(
                    "CREATE TABLE lineage_items "
                    "(split_index INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
                )
                pending: List[Tuple[int, str]] = []
                with lineage_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        value = json.loads(line)
                        split_indices = dict(value.get("split_indices") or {})
                        if split_name not in split_indices:
                            continue
                        pending.append(
                            (
                                int(split_indices[split_name]),
                                json.dumps(value, sort_keys=True, ensure_ascii=False, default=str),
                            )
                        )
                        if len(pending) >= 1000:
                            connection.executemany(
                                "INSERT INTO lineage_items VALUES (?, ?)", pending
                            )
                            pending.clear()
                if pending:
                    connection.executemany("INSERT INTO lineage_items VALUES (?, ?)", pending)
                connection.commit()
                cursor = connection.execute(
                    "SELECT payload FROM lineage_items ORDER BY split_index"
                )
                for (payload,) in cursor:
                    yield RecordIdentity.from_value(json.loads(payload))
                return
            finally:
                if connection is not None:
                    connection.close()
                index_path.unlink(missing_ok=True)

        # Legacy versions deliberately remain immutable.  Synthesizing their
        # virtual lineage still uses the historical boundedness exception.
        records = _read_jsonl(path / "records.jsonl")
        split_rows = {
            str(name): _read_jsonl(path / "splits" / f"{_safe_id(str(name), 'split name')}.jsonl")
            for name in dict(manifest.get("split_counts") or {})
        }
        _, _, identities = build_version_lineage(
            records,
            split_rows or {"train": records},
            version_id=str(manifest["version_id"]),
            virtual=True,
        )
        if split is None:
            row_count = int(manifest.get("row_count", len(identities)))
            selected = sorted(
                (identity for identity in identities if identity.record_index < row_count),
                key=lambda identity: identity.record_index,
            )
        else:
            split_name = _safe_id(split, "split name")
            selected = sorted(
                (identity for identity in identities if split_name in identity.splits),
                key=lambda identity: identity.split_indices[split_name],
            )
        yield from selected

    def iter_records_with_lineage(
        self,
        version_id: str,
        *,
        dataset_id: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Iterator[Tuple[Dict[str, Any], RecordIdentity]]:
        """Yield paired records and identities and reject inconsistent sidecars."""

        records = self.iter_records(version_id, dataset_id=dataset_id, split=split)
        identities = self.iter_lineage(version_id, dataset_id=dataset_id, split=split)
        sentinel = object()
        while True:
            record = next(records, sentinel)
            identity = next(identities, sentinel)
            if record is sentinel and identity is sentinel:
                return
            if record is sentinel or identity is sentinel:
                raise VersionError("Dataset version lineage is inconsistent with its records")
            yield record, identity

    def load_records_with_lineage(
        self,
        version_id: str,
        *,
        dataset_id: Optional[str] = None,
        split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load rows with a private lineage envelope for another recipe."""

        records = self.load_records(version_id, dataset_id=dataset_id, split=split)
        identities = self.load_lineage(version_id, dataset_id=dataset_id, split=split)
        try:
            return attach_identities(records, identities)
        except ValueError as exc:
            raise VersionError(f"Dataset version lineage is inconsistent: {exc}") from exc

    def load_identified_records(
        self,
        version_id: str,
        *,
        dataset_id: Optional[str] = None,
        split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return public record/identity envelopes for previews and analysis."""

        records = self.load_records(version_id, dataset_id=dataset_id, split=split)
        identities = self.load_lineage(version_id, dataset_id=dataset_id, split=split)
        if len(records) != len(identities):
            raise VersionError("Dataset version lineage is inconsistent with its records")
        return [
            {"record": copy.deepcopy(record), "identity": identity.to_dict()}
            for record, identity in zip(records, identities)
        ]

    def export(
        self,
        version_id: str,
        destination: Path | str,
        *,
        dataset_id: Optional[str] = None,
        split: Optional[str] = None,
    ) -> Path:
        path = self._find(version_id, dataset_id)
        source = (
            path / "records.jsonl"
            if split is None
            else path / "splits" / f"{_safe_id(split, 'split name')}.jsonl"
        )
        if not source.is_file():
            raise VersionError(f"Dataset version has no split {split!r}")
        output = Path(destination).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        return output

    def verify(
        self, version_id: str, *, dataset_id: Optional[str] = None, verify_source: bool = False
    ) -> Dict[str, Any]:
        path = self._find(version_id, dataset_id)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        problems: List[str] = []
        if manifest.get("status") != "complete":
            problems.append("manifest status is not complete")
        for relative, expected in manifest.get("artifact_hashes", {}).items():
            artifact = path / relative
            if not artifact.is_file():
                problems.append(f"missing artifact: {relative}")
            elif hash_file(artifact) != expected:
                problems.append(f"changed artifact: {relative}")
        for reference, relative in manifest.get("asset_mapping", {}).items():
            if not (path / relative).is_file():
                problems.append(f"missing materialized asset for {reference}: {relative}")
        if verify_source:
            source_data = manifest.get("source", {})
            try:
                snapshot = SourceSnapshot(
                    spec=SourceSpec.from_value(source_data["spec"]),
                    records=[],
                    fingerprint=str(manifest.get("source_fingerprint") or ""),
                    assets=[AssetFingerprint(**asset) for asset in source_data.get("assets", [])],
                )
                from .sources import verify_snapshot

                if not verify_snapshot(snapshot):
                    problems.append("source or referenced assets changed or are missing")
            except Exception:
                problems.append("source or referenced assets are missing or unreadable")
        return {
            "valid": not problems,
            "problems": problems,
            "dataset_id": manifest["dataset_id"],
            "version_id": manifest["version_id"],
        }

    def clean_temporary(self) -> int:
        removed = 0
        for path in self.root.glob("*/.*.tmp-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed


__all__ = ["DatasetVersion", "VersionStore"]
