"""Identity-aware comparison of immutable Dataset Lab versions."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .identity import RecordIdentity
from .storage import DatasetVersion, VersionStore


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return copy.deepcopy(default)


def _group(identities: Sequence[RecordIdentity]) -> Dict[str, List[RecordIdentity]]:
    grouped: Dict[str, List[RecordIdentity]] = {}
    for identity in identities:
        grouped.setdefault(identity.record_id, []).append(identity)
    return grouped


def _source_contributions(version: DatasetVersion) -> Dict[str, Any]:
    path = Path(version.path)
    manifest = _read_json(path / "manifest.json", {})
    provenance = _read_json(path / "provenance.json", [])
    values: Dict[str, Any] = {}
    primary = manifest.get("source_fingerprint")
    if primary:
        values[str(primary)] = {
            "role": "primary",
            "fingerprint": str(primary),
            "available": int((manifest.get("source") or {}).get("row_count", 0)),
        }
    for step in provenance if isinstance(provenance, list) else ():
        if not isinstance(step, Mapping) or step.get("kind") != "mix":
            continue
        for source in (step.get("details") or {}).get("sources") or ():
            if not isinstance(source, Mapping) or not source.get("source"):
                continue
            values[str(source["source"])] = {
                "role": "mixture",
                "source": str(source["source"]),
                "weight": source.get("weight"),
                "available": source.get("available"),
            }
    return values


@dataclass(frozen=True)
class DatasetVersionComparison:
    left: Dict[str, Any]
    right: Dict[str, Any]
    summary: Dict[str, int]
    added: tuple[Dict[str, Any], ...]
    removed: tuple[Dict[str, Any], ...]
    content_changed: tuple[Dict[str, Any], ...]
    repeated: tuple[Dict[str, Any], ...]
    moved_between_splits: tuple[Dict[str, Any], ...]
    recipe: Dict[str, Any]
    statistics: Dict[str, Any]
    source_contributions: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        for name in (
            "added",
            "removed",
            "content_changed",
            "repeated",
            "moved_between_splits",
        ):
            value[name] = [copy.deepcopy(item) for item in getattr(self, name)]
        return value


class DatasetVersionComparator:
    def __init__(self, store: VersionStore):
        self.store = store

    def compare(
        self,
        left_version_id: str,
        right_version_id: str,
        *,
        left_dataset_id: Optional[str] = None,
        right_dataset_id: Optional[str] = None,
    ) -> DatasetVersionComparison:
        left = self.store.get_any(left_version_id, left_dataset_id)
        right = self.store.get_any(right_version_id, right_dataset_id)
        left_lineage = self.store.load_lineage(left.version_id, dataset_id=left.dataset_id)
        right_lineage = self.store.load_lineage(right.version_id, dataset_id=right.dataset_id)
        left_grouped = _group(left_lineage)
        right_grouped = _group(right_lineage)
        left_ids = set(left_grouped)
        right_ids = set(right_grouped)

        added = tuple(
            {
                "record_id": record_id,
                "count": len(right_grouped[record_id]),
                "record_hashes": sorted(
                    {identity.record_hash for identity in right_grouped[record_id]}
                ),
                "instance_ids": [identity.instance_id for identity in right_grouped[record_id]],
            }
            for record_id in sorted(right_ids - left_ids)
        )
        removed = tuple(
            {
                "record_id": record_id,
                "count": len(left_grouped[record_id]),
                "record_hashes": sorted(
                    {identity.record_hash for identity in left_grouped[record_id]}
                ),
                "instance_ids": [identity.instance_id for identity in left_grouped[record_id]],
            }
            for record_id in sorted(left_ids - right_ids)
        )

        changed_values: List[Dict[str, Any]] = []
        repeated_values: List[Dict[str, Any]] = []
        moved_values: List[Dict[str, Any]] = []
        for record_id in sorted(left_ids & right_ids):
            left_rows = left_grouped[record_id]
            right_rows = right_grouped[record_id]
            left_hashes = sorted({identity.record_hash for identity in left_rows})
            right_hashes = sorted({identity.record_hash for identity in right_rows})
            if left_hashes != right_hashes:
                changed_values.append(
                    {
                        "record_id": record_id,
                        "left_record_hashes": left_hashes,
                        "right_record_hashes": right_hashes,
                        "left_instance_ids": [value.instance_id for value in left_rows],
                        "right_instance_ids": [value.instance_id for value in right_rows],
                    }
                )
            if len(left_rows) != len(right_rows) or len(left_rows) > 1 or len(right_rows) > 1:
                repeated_values.append(
                    {
                        "record_id": record_id,
                        "left_count": len(left_rows),
                        "right_count": len(right_rows),
                        "delta": len(right_rows) - len(left_rows),
                        "left_instance_ids": [value.instance_id for value in left_rows],
                        "right_instance_ids": [value.instance_id for value in right_rows],
                    }
                )
            left_split_counts: Dict[str, int] = {}
            right_split_counts: Dict[str, int] = {}
            for identity in left_rows:
                for split in identity.splits:
                    left_split_counts[split] = left_split_counts.get(split, 0) + 1
            for identity in right_rows:
                for split in identity.splits:
                    right_split_counts[split] = right_split_counts.get(split, 0) + 1
            split_membership_changed = set(left_split_counts) != set(right_split_counts)
            redistributed = (
                len(left_rows) == len(right_rows) and left_split_counts != right_split_counts
            )
            if split_membership_changed or redistributed:
                moved_values.append(
                    {
                        "record_id": record_id,
                        "left_splits": sorted(left_split_counts),
                        "right_splits": sorted(right_split_counts),
                        "left_split_counts": left_split_counts,
                        "right_split_counts": right_split_counts,
                        "left_instance_ids": [value.instance_id for value in left_rows],
                        "right_instance_ids": [value.instance_id for value in right_rows],
                    }
                )

        left_path = Path(left.path)
        right_path = Path(right.path)
        left_recipe = _read_json(left_path / "recipe.json", {})
        right_recipe = _read_json(right_path / "recipe.json", {})
        left_stats = _read_json(left_path / "stats.json", {})
        right_stats = _read_json(right_path / "stats.json", {})
        left_sources = _source_contributions(left)
        right_sources = _source_contributions(right)
        source_keys = sorted(set(left_sources) | set(right_sources))
        source_changes = [
            {
                "source": key,
                "left": copy.deepcopy(left_sources.get(key)),
                "right": copy.deepcopy(right_sources.get(key)),
            }
            for key in source_keys
            if left_sources.get(key) != right_sources.get(key)
        ]

        summary = {
            "left_records": len(left_lineage),
            "right_records": len(right_lineage),
            "added": len(added),
            "removed": len(removed),
            "content_changed": len(changed_values),
            "repeated": len(repeated_values),
            "moved_between_splits": len(moved_values),
        }
        return DatasetVersionComparison(
            left=left.to_dict(),
            right=right.to_dict(),
            summary=summary,
            added=added,
            removed=removed,
            content_changed=tuple(changed_values),
            repeated=tuple(repeated_values),
            moved_between_splits=tuple(moved_values),
            recipe={
                "changed": left.recipe_hash != right.recipe_hash,
                "left_hash": left.recipe_hash,
                "right_hash": right.recipe_hash,
                "left": left_recipe,
                "right": right_recipe,
            },
            statistics={
                "changed": left_stats != right_stats,
                "left": left_stats,
                "right": right_stats,
            },
            source_contributions={
                "changed": bool(source_changes),
                "left": left_sources,
                "right": right_sources,
                "differences": source_changes,
            },
        )


def compare_dataset_versions(
    store: VersionStore,
    left_version_id: str,
    right_version_id: str,
    **kwargs: Any,
) -> DatasetVersionComparison:
    return DatasetVersionComparator(store).compare(left_version_id, right_version_id, **kwargs)


__all__ = [
    "DatasetVersionComparator",
    "DatasetVersionComparison",
    "compare_dataset_versions",
]
