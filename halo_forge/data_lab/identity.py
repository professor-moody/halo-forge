"""Stable record identity and immutable-version lineage helpers.

Dataset recipes operate on ordinary dictionaries.  During a build we attach a
private envelope to each row so filtering, mapping, sampling, mixing, and
synthesis can carry origin identity without leaking implementation fields into
the canonical JSONL files.  Publication removes the envelope and writes the
identity information to ``lineage.jsonl`` instead.

Old Dataset Lab versions deliberately remain untouched.  Readers synthesize the
same shape of identity for those versions and mark it ``virtual``.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .sources import content_hash

INTERNAL_LINEAGE_KEY = "__halo_forge_internal_lineage_v2__"


@dataclass(frozen=True)
class RecordIdentity:
    """Identity for one occurrence of a record in an immutable version."""

    record_id: str
    record_hash: str
    instance_id: str
    record_index: int
    splits: tuple[str, ...] = ()
    split_indices: Dict[str, int] = field(default_factory=dict)
    origins: tuple[Dict[str, Any], ...] = ()
    operations: tuple[Dict[str, Any], ...] = ()
    parent_instance_ids: tuple[str, ...] = ()
    virtual: bool = False

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["splits"] = list(self.splits)
        value["origins"] = [copy.deepcopy(item) for item in self.origins]
        value["operations"] = [copy.deepcopy(item) for item in self.operations]
        value["parent_instance_ids"] = list(self.parent_instance_ids)
        return value

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "RecordIdentity":
        raw = dict(value)
        raw["splits"] = tuple(str(item) for item in raw.get("splits") or ())
        raw["split_indices"] = {
            str(name): int(index) for name, index in dict(raw.get("split_indices") or {}).items()
        }
        raw["origins"] = tuple(dict(item) for item in raw.get("origins") or ())
        raw["operations"] = tuple(dict(item) for item in raw.get("operations") or ())
        raw["parent_instance_ids"] = tuple(
            str(item) for item in raw.get("parent_instance_ids") or ()
        )
        return cls(**raw)


def strip_internal_identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep canonical copy of a row without the build-only envelope."""

    return {
        str(key): copy.deepcopy(value)
        for key, value in record.items()
        if key != INTERNAL_LINEAGE_KEY
    }


def record_hash(record: Mapping[str, Any]) -> str:
    """Hash canonical record content, excluding Dataset Lab bookkeeping."""

    return content_hash(strip_internal_identity(record))


def deterministic_record_id(record: Mapping[str, Any]) -> str:
    """Create a stable logical ID for a previously unidentified row.

    Source-provided identity wins when available, allowing a changed payload to
    compare as the same logical record across source revisions.  Content identity
    is the deterministic fallback for datasets without keys.
    """

    clean = strip_internal_identity(record)
    for field_name in ("record_id", "id", "uuid", "key"):
        value = clean.get(field_name)
        if value is not None and str(value).strip():
            return "rec_" + content_hash({"field": field_name, "value": value})
    return f"rec_{content_hash(clean)}"


def _marker_for(
    record: Mapping[str, Any],
    *,
    source_fingerprint: Optional[str] = None,
    source_index: Optional[int] = None,
    source_name: Optional[str] = None,
    virtual: bool = False,
) -> Dict[str, Any]:
    source_record_hash = record_hash(record)
    origin = {
        key: value
        for key, value in {
            "source_fingerprint": source_fingerprint,
            "source_index": source_index,
            "source_name": source_name,
            "source_record_hash": source_record_hash,
        }.items()
        if value is not None
    }
    return {
        "record_id": deterministic_record_id(record),
        "origins": [origin] if origin else [],
        "operations": [],
        "parent_instance_ids": [],
        "virtual": bool(virtual),
    }


def seed_record_identity(
    record: Mapping[str, Any],
    *,
    source_fingerprint: Optional[str] = None,
    source_index: Optional[int] = None,
    source_name: Optional[str] = None,
    virtual: bool = False,
) -> Dict[str, Any]:
    """Copy a row and attach an origin envelope unless it already has one."""

    output = copy.deepcopy(dict(record))
    existing = output.get(INTERNAL_LINEAGE_KEY)
    if not isinstance(existing, Mapping) or not existing.get("record_id"):
        output[INTERNAL_LINEAGE_KEY] = _marker_for(
            output,
            source_fingerprint=source_fingerprint,
            source_index=source_index,
            source_name=source_name,
            virtual=virtual,
        )
    else:
        output[INTERNAL_LINEAGE_KEY] = copy.deepcopy(dict(existing))
    return output


def seed_record_identities(
    records: Sequence[Mapping[str, Any]],
    *,
    source_fingerprint: Optional[str] = None,
    source_name: Optional[str] = None,
    virtual: bool = False,
) -> List[Dict[str, Any]]:
    return [
        seed_record_identity(
            row,
            source_fingerprint=source_fingerprint,
            source_index=index,
            source_name=source_name,
            virtual=virtual,
        )
        for index, row in enumerate(records)
    ]


def derive_record_identity(
    record: MutableMapping[str, Any], operation: str, **details: Any
) -> None:
    """Append derivation provenance while retaining the logical record ID."""

    marker = record.get(INTERNAL_LINEAGE_KEY)
    if not isinstance(marker, MutableMapping):
        seeded = seed_record_identity(record)
        marker = seeded[INTERNAL_LINEAGE_KEY]
        record[INTERNAL_LINEAGE_KEY] = marker
    operations = marker.setdefault("operations", [])
    if not isinstance(operations, list):
        operations = marker["operations"] = []
    operations.append(
        {
            "operation": str(operation),
            **{str(key): copy.deepcopy(value) for key, value in details.items()},
        }
    )


def marker_from_identity(identity: RecordIdentity) -> Dict[str, Any]:
    """Reattach persisted or virtual identity for a downstream recipe."""

    return {
        "record_id": identity.record_id,
        "origins": [copy.deepcopy(item) for item in identity.origins],
        "operations": [copy.deepcopy(item) for item in identity.operations],
        "parent_instance_ids": [identity.instance_id, *identity.parent_instance_ids],
        "virtual": identity.virtual,
    }


def attach_identities(
    records: Sequence[Mapping[str, Any]], identities: Sequence[RecordIdentity]
) -> List[Dict[str, Any]]:
    if len(records) != len(identities):
        raise ValueError("Record and lineage counts do not match")
    output: List[Dict[str, Any]] = []
    for record, identity in zip(records, identities):
        row = strip_internal_identity(record)
        row[INTERNAL_LINEAGE_KEY] = marker_from_identity(identity)
        output.append(row)
    return output


def build_version_lineage(
    records: Sequence[Mapping[str, Any]],
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    version_id: str,
    virtual: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], List[RecordIdentity]]:
    """Strip internal markers and build one identity entry per row occurrence."""

    clean_records: List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []
    occurrence_counts: Dict[tuple[str, str], int] = {}
    by_key: Dict[tuple[str, str], List[int]] = {}

    for record_index, raw in enumerate(records):
        clean = strip_internal_identity(raw)
        clean_records.append(clean)
        marker = raw.get(INTERNAL_LINEAGE_KEY)
        if not isinstance(marker, Mapping):
            marker = _marker_for(clean, virtual=virtual)
        logical_id = str(marker.get("record_id") or deterministic_record_id(clean))
        final_hash = content_hash(clean)
        key = (logical_id, final_hash)
        occurrence = occurrence_counts.get(key, 0)
        occurrence_counts[key] = occurrence + 1
        instance_id = "inst_" + content_hash(
            {
                "version_id": version_id,
                "record_id": logical_id,
                "record_hash": final_hash,
                "occurrence": occurrence,
            }
        )
        entry = {
            "record_id": logical_id,
            "record_hash": final_hash,
            "instance_id": instance_id,
            "record_index": record_index,
            "splits": [],
            "split_indices": {},
            "origins": [dict(item) for item in marker.get("origins") or ()],
            "operations": [dict(item) for item in marker.get("operations") or ()],
            "parent_instance_ids": [str(item) for item in marker.get("parent_instance_ids") or ()],
            "virtual": bool(virtual or marker.get("virtual", False)),
        }
        entries.append(entry)
        by_key.setdefault(key, []).append(record_index)

    clean_splits: Dict[str, List[Dict[str, Any]]] = {}
    split_cursors: Dict[tuple[str, str], int] = {}
    for split_name, raw_rows in splits.items():
        split = str(split_name)
        clean_splits[split] = []
        for split_index, raw in enumerate(raw_rows):
            clean = strip_internal_identity(raw)
            clean_splits[split].append(clean)
            marker = raw.get(INTERNAL_LINEAGE_KEY)
            logical_id = (
                str(marker.get("record_id"))
                if isinstance(marker, Mapping) and marker.get("record_id")
                else deterministic_record_id(clean)
            )
            key = (logical_id, content_hash(clean))
            candidates = by_key.get(key, [])
            if not candidates:
                # A malformed/stale recipe split should not make publication lose
                # lineage.  Represent the split-only occurrence explicitly.
                occurrence = occurrence_counts.get(key, 0)
                occurrence_counts[key] = occurrence + 1
                record_index = len(entries)
                instance_id = "inst_" + content_hash(
                    {
                        "version_id": version_id,
                        "record_id": logical_id,
                        "record_hash": key[1],
                        "occurrence": occurrence,
                    }
                )
                entry = {
                    "record_id": logical_id,
                    "record_hash": key[1],
                    "instance_id": instance_id,
                    "record_index": record_index,
                    "splits": [],
                    "split_indices": {},
                    "origins": (
                        [dict(item) for item in marker.get("origins") or ()]
                        if isinstance(marker, Mapping)
                        else []
                    ),
                    "operations": (
                        [dict(item) for item in marker.get("operations") or ()]
                        if isinstance(marker, Mapping)
                        else []
                    ),
                    "parent_instance_ids": (
                        [str(item) for item in marker.get("parent_instance_ids") or ()]
                        if isinstance(marker, Mapping)
                        else []
                    ),
                    "virtual": bool(virtual),
                }
                entries.append(entry)
                by_key.setdefault(key, []).append(record_index)
                candidates = [record_index]
            cursor = split_cursors.get(key, 0)
            selected = candidates[min(cursor, len(candidates) - 1)]
            split_cursors[key] = cursor + 1
            if split not in entries[selected]["splits"]:
                entries[selected]["splits"].append(split)
            entries[selected]["split_indices"][split] = split_index

    identities = [RecordIdentity.from_value(entry) for entry in entries]
    return clean_records, clean_splits, identities


__all__ = [
    "INTERNAL_LINEAGE_KEY",
    "RecordIdentity",
    "attach_identities",
    "build_version_lineage",
    "derive_record_identity",
    "deterministic_record_id",
    "marker_from_identity",
    "record_hash",
    "seed_record_identities",
    "seed_record_identity",
    "strip_internal_identity",
]
