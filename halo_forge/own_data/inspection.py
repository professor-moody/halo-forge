"""Streaming source inspection, deterministic previews, and schema inference."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

from halo_forge.data_lab.models import get_field

from .models import FieldMappingExpression, SchemaCandidate
from .registry import TRAINING_SCENARIOS, TrainingScenarioRegistry

IMPORT_ADAPTER_VERSION = "guided-source-v2"
PREVIEW_HEAD_COUNT = 100
PREVIEW_RESERVOIR_COUNT = 900

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
_MANIFEST_EXTENSIONS = {".json", ".jsonl", ".jl", ".csv", ".tsv", ".parquet"}
_MEDIA_FIELDS = {
    "image",
    "image_path",
    "images",
    "audio",
    "audio_path",
    "file",
    "path",
    "filename",
    "relative_path",
    "media",
    "media_path",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_path(path: Path | str) -> Tuple[str, int, int]:
    selected = Path(path).expanduser()
    if selected.is_symlink():
        raise ValueError("symbolic-link dataset sources are not accepted")
    root = selected.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        if root.is_symlink():
            raise ValueError("symbolic-link dataset sources are not accepted")
        return _sha256_file(root), root.stat().st_size, 1
    entries: list[tuple[str, str, int]] = []
    total_size = 0
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"unsafe symbolic link in dataset source: {item.relative_to(root)}")
        if not item.is_file():
            continue
        relative = item.relative_to(root).as_posix()
        size = item.stat().st_size
        total_size += size
        entries.append((relative, _sha256_file(item), size))
    payload = json.dumps(entries, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), total_size, len(entries)


def _iter_jsonl(path: Path) -> Iterator[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                yield None, {"code": "invalid_json", "line": line_number, "message": str(exc)}
                continue
            if not isinstance(value, Mapping):
                yield None, {
                    "code": "record_not_object",
                    "line": line_number,
                    "message": "Each JSONL record must be an object.",
                }
                continue
            yield dict(value), None


def _iter_json(path: Path) -> Iterator[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    try:
        import ijson  # type: ignore
    except ImportError:
        ijson = None
    if ijson is not None:
        with path.open("rb") as handle:
            try:
                for index, value in enumerate(ijson.items(handle, "item")):
                    if isinstance(value, Mapping):
                        yield dict(value), None
                    else:
                        yield None, {
                            "code": "record_not_object",
                            "index": index,
                            "message": "Each JSON array entry must be an object.",
                        }
                return
            except Exception as exc:
                yield None, {"code": "invalid_json", "message": str(exc)}
                return
    # Keep JSON-array inspection bounded even when the optional ijson package
    # is absent. A single top-level object is necessarily one record and is
    # decoded normally; array entries are incrementally decoded and discarded.
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def fill() -> bool:
            nonlocal buffer, position, eof
            if eof:
                return False
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(64 * 1024)
            if not chunk:
                eof = True
                return False
            buffer += chunk
            return True

        def skip_space() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or not fill():
                    return

        fill()
        skip_space()
        if position >= len(buffer):
            yield None, {"code": "invalid_json", "message": "JSON source is empty."}
            return
        if buffer[position] != "[":
            try:
                # Reconstruct only the single-record document, including the
                # prefix already read. This branch never materializes an array.
                remainder = buffer[position:] + handle.read()
                value = json.loads(remainder)
            except json.JSONDecodeError as exc:
                yield None, {"code": "invalid_json", "message": str(exc)}
                return
            if isinstance(value, Mapping):
                yield dict(value), None
            else:
                yield None, {
                    "code": "record_not_object",
                    "index": 0,
                    "message": "Each JSON record must be an object.",
                }
            return

        position += 1
        index = 0
        while True:
            skip_space()
            if position < len(buffer) and buffer[position] == "]":
                position += 1
                skip_space()
                if position < len(buffer):
                    yield None, {
                        "code": "invalid_json",
                        "message": "Unexpected content after the JSON array.",
                    }
                return
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError as exc:
                    if fill():
                        continue
                    yield None, {"code": "invalid_json", "index": index, "message": str(exc)}
                    return
            if isinstance(value, Mapping):
                yield dict(value), None
            else:
                yield None, {
                    "code": "record_not_object",
                    "index": index,
                    "message": "Each JSON array entry must be an object.",
                }
            index += 1
            skip_space()
            if position >= len(buffer):
                yield None, {
                    "code": "invalid_json",
                    "index": index,
                    "message": "JSON array ended before its closing bracket.",
                }
                return
            if buffer[position] == ",":
                position += 1
                continue
            if buffer[position] == "]":
                continue
            yield None, {
                "code": "invalid_json",
                "index": index,
                "message": "Expected a comma or closing bracket between JSON records.",
            }
            return


def _iter_delimited(
    path: Path, delimiter: str
) -> Iterator[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            yield None, {
                "code": "missing_header",
                "message": "Delimited files require a header row.",
            }
            return
        for row in reader:
            yield dict(row), None


def _iter_parquet(
    path: Path,
) -> Iterator[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore
    except ImportError as exc:
        raise ValueError("Parquet inspection requires the pyarrow package") from exc
    source = parquet.ParquetFile(path)
    for batch in source.iter_batches(batch_size=1024):
        for row in batch.to_pylist():
            if isinstance(row, Mapping):
                yield dict(row), None


def iter_file_records(
    path: Path,
) -> Iterator[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".jl"}:
        yield from _iter_jsonl(path)
    elif suffix == ".json":
        yield from _iter_json(path)
    elif suffix == ".csv":
        yield from _iter_delimited(path, ",")
    elif suffix == ".tsv":
        yield from _iter_delimited(path, "\t")
    elif suffix == ".parquet":
        yield from _iter_parquet(path)
    else:
        raise ValueError(f"unsupported dataset file format: {suffix or '<none>'}")


def _manifest_in_directory(root: Path) -> Optional[Path]:
    candidates = [
        item
        for item in sorted(root.iterdir())
        if item.is_file() and item.suffix.lower() in _MANIFEST_EXTENSIONS
    ]
    preferred = [
        item
        for item in candidates
        if item.stem.lower() in {"manifest", "metadata", "data", "dataset"}
    ]
    return (preferred or candidates or [None])[0]


def _iter_paired_media(
    root: Path,
) -> Iterator[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    for media in sorted(root.rglob("*")):
        if not media.is_file() or media.suffix.lower() not in _IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS:
            continue
        sidecar = media.with_suffix(".txt")
        if not sidecar.is_file():
            yield None, {
                "code": "missing_sidecar",
                "path": media.relative_to(root).as_posix(),
                "message": "Media file has no same-basename .txt caption or transcript.",
            }
            continue
        text = sidecar.read_text(encoding="utf-8").strip()
        relative = media.relative_to(root).as_posix()
        if media.suffix.lower() in _IMAGE_EXTENSIONS:
            yield {"image": relative, "caption": text, "_media_root": str(root)}, None
        else:
            yield {"audio": relative, "transcript": text, "_media_root": str(root)}, None


def iter_source_records(
    path: Path | str,
) -> Iterator[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
    selected = Path(path).expanduser()
    if selected.is_symlink():
        raise ValueError("symbolic-link dataset sources are not accepted")
    source = selected.resolve()
    if source.is_file():
        for record, issue in iter_file_records(source):
            if record is not None and _MEDIA_FIELDS.intersection(record):
                record.setdefault("_media_root", str(source.parent))
            yield record, issue
        return
    if not source.is_dir():
        raise FileNotFoundError(source)
    manifest = _manifest_in_directory(source)
    if manifest is not None:
        for record, issue in iter_file_records(manifest):
            if record is not None and _MEDIA_FIELDS.intersection(record):
                record.setdefault("_media_root", str(source))
            yield record, issue
        return
    yield from _iter_paired_media(source)


def _media_values(record: Mapping[str, Any]) -> Iterator[Tuple[str, str, str]]:
    """Yield declared local media references without guessing from ordinary paths."""

    for field in sorted(_MEDIA_FIELDS.intersection(record)):
        raw_value = record.get(field)
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            if isinstance(value, Mapping):
                value = value.get("path") or value.get("file")
            if not isinstance(value, str) or not value.strip():
                continue
            suffix = Path(value).suffix.lower()
            modality = (
                "image"
                if suffix in _IMAGE_EXTENSIONS
                else "audio" if suffix in _AUDIO_EXTENSIONS else ""
            )
            if modality:
                yield field, value, modality


def _record_media_evidence(
    record: Mapping[str, Any],
    *,
    source_index: int,
    asset_fingerprints: Optional[Dict[tuple[str, str], Dict[str, Any]]] = None,
) -> Tuple[list[Dict[str, Any]], Dict[str, int]]:
    issues: list[Dict[str, Any]] = []
    counts = {
        "referenced": 0,
        "verified": 0,
        "missing": 0,
        "unsafe": 0,
        "image_references": 0,
        "audio_references": 0,
    }
    root_value = record.get("_media_root")
    references = list(_media_values(record))
    if not references:
        return issues, counts
    if not isinstance(root_value, str) or not root_value:
        counts["referenced"] = len(references)
        counts["unsafe"] = len(references)
        for field, value, modality in references:
            counts[f"{modality}_references"] += 1
            issues.append(
                {
                    "code": "missing_media_root",
                    "index": source_index,
                    "field": field,
                    "path": value,
                    "message": "The media reference has no validated media root.",
                }
            )
            if asset_fingerprints is not None:
                asset_fingerprints.setdefault(
                    (field, value),
                    {
                        "field": field,
                        "reference": value,
                        "resolved_path": None,
                        "fingerprint": None,
                        "size_bytes": None,
                        "missing": False,
                        "unsafe": True,
                    },
                )
        return issues, counts

    root = Path(root_value).expanduser().resolve()
    for field, value, modality in references:
        counts["referenced"] += 1
        counts[f"{modality}_references"] += 1
        raw_candidate = Path(value).expanduser()
        candidate = raw_candidate if raw_candidate.is_absolute() else root / raw_candidate
        try:
            # Reject both a symlink leaf and symlinked path components. The
            # latter otherwise disappear when Path.resolve() is called.
            relative_candidate = candidate.relative_to(root)
            cursor = root
            unsafe_link = False
            for part in relative_candidate.parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    unsafe_link = True
                    break
            resolved = candidate.resolve()
            resolved.relative_to(root)
            if unsafe_link:
                raise ValueError("symbolic-link media paths are not accepted")
        except (OSError, ValueError) as exc:
            counts["unsafe"] += 1
            issues.append(
                {
                    "code": "unsafe_media_path",
                    "index": source_index,
                    "field": field,
                    "path": value,
                    "message": str(exc) or "The media path escapes the selected source.",
                }
            )
            if asset_fingerprints is not None:
                asset_fingerprints.setdefault(
                    (field, value),
                    {
                        "field": field,
                        "reference": value,
                        "resolved_path": None,
                        "fingerprint": None,
                        "size_bytes": None,
                        "missing": False,
                        "unsafe": True,
                    },
                )
            continue
        if not resolved.is_file():
            counts["missing"] += 1
            issues.append(
                {
                    "code": "missing_media_asset",
                    "index": source_index,
                    "field": field,
                    "path": value,
                    "message": "The referenced media asset does not exist.",
                }
            )
            if asset_fingerprints is not None:
                asset_fingerprints.setdefault(
                    (field, value),
                    {
                        "field": field,
                        "reference": value,
                        "resolved_path": str(resolved),
                        "fingerprint": None,
                        "size_bytes": None,
                        "missing": True,
                        "unsafe": False,
                    },
                )
            continue
        counts["verified"] += 1
        if asset_fingerprints is not None and (field, value) not in asset_fingerprints:
            asset_fingerprints[(field, value)] = {
                "field": field,
                "reference": value,
                "resolved_path": str(resolved),
                "fingerprint": _sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
                "missing": False,
                "unsafe": False,
            }
    return issues, counts


def _field_profiles(records: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    names = sorted(
        {str(key) for row in records for key in row.keys() if not str(key).startswith("_")}
    )
    total = max(1, len(records))
    output = []
    for name in names:
        values = [row.get(name) for row in records if row.get(name) is not None]
        types: Dict[str, int] = {}
        examples = []
        for value in values:
            type_name = type(value).__name__
            types[type_name] = types.get(type_name, 0) + 1
            if len(examples) < 3 and value not in examples:
                examples.append(value)
        output.append(
            {"name": name, "coverage": len(values) / total, "types": types, "examples": examples}
        )
    return output


def _update_field_counters(counters: Dict[str, Dict[str, Any]], record: Mapping[str, Any]) -> None:
    for raw_name, value in record.items():
        name = str(raw_name)
        if name.startswith("_"):
            continue
        counter = counters.setdefault(
            name, {"present_count": 0, "null_count": 0, "types": {}, "examples": []}
        )
        if value is None:
            counter["null_count"] += 1
            continue
        counter["present_count"] += 1
        type_name = type(value).__name__
        counter["types"][type_name] = counter["types"].get(type_name, 0) + 1
        if len(counter["examples"]) < 3 and value not in counter["examples"]:
            counter["examples"].append(value)


def _exact_field_profiles(
    counters: Mapping[str, Mapping[str, Any]], *, total_valid: int
) -> list[Dict[str, Any]]:
    denominator = max(1, int(total_valid))
    return [
        {
            "name": name,
            "coverage": int(value.get("present_count", 0)) / denominator,
            "present_count": int(value.get("present_count", 0)),
            "null_count": denominator - int(value.get("present_count", 0)),
            "types": dict(value.get("types") or {}),
            "value_type": max(
                dict(value.get("types") or {}),
                key=dict(value.get("types") or {}).get,
                default="unknown",
            ),
            "examples": list(value.get("examples") or []),
        }
        for name, value in sorted(counters.items())
    ]


def _best_alias(
    records: Sequence[Mapping[str, Any]], aliases: Sequence[str]
) -> Tuple[Optional[str], float]:
    if not records:
        return None, 0.0
    scored = []
    for alias in aliases:
        hits = sum(1 for row in records if get_field(row, alias) not in (None, "", [], {}))
        scored.append((hits / len(records), alias))
    coverage, alias = max(scored, default=(0.0, ""))
    return (alias or None), coverage


def infer_schema_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    registry: TrainingScenarioRegistry = TRAINING_SCENARIOS,
) -> list[SchemaCandidate]:
    candidates: list[SchemaCandidate] = []
    available_fields = {str(key) for row in records for key in row.keys()}
    for scenario in registry.list(include_unavailable=False):
        hints = scenario.detection_hints
        required_any = set(hints.get("require_any_fields") or ())
        if required_any and not required_any.intersection(available_fields):
            continue
        excluded = set(hints.get("exclude_fields") or ())
        if excluded.intersection(available_fields):
            continue
        mapping: Dict[str, Dict[str, Any]] = {}
        coverages: list[float] = []
        field_coverages: Dict[str, float] = {}
        safe_transforms = 0
        reasons: list[str] = []
        for target in scenario.required_fields:
            alias, coverage = _best_alias(records, scenario.field_aliases.get(target, (target,)))
            if coverage >= 0.95 and alias:
                kind = (
                    "conversation"
                    if target == "messages"
                    else "media_root" if target in {"image", "audio"} else "direct"
                )
                expression = FieldMappingExpression(
                    kind=kind,
                    source=alias,
                    media_root=(
                        str(records[0].get("_media_root"))
                        if kind == "media_root" and records and records[0].get("_media_root")
                        else None
                    ),
                )
                mapping[target] = expression.to_dict()
                coverages.append(coverage)
                field_coverages[target] = coverage
            elif target in scenario.safe_constants:
                mapping[target] = FieldMappingExpression(
                    kind="constant", value=scenario.safe_constants[target]
                ).to_dict()
                coverages.append(1.0)
                field_coverages[target] = 1.0
                safe_transforms += 1
                reasons.append(f"{target} can use the scenario default")
            else:
                coverages.append(coverage)
                field_coverages[target] = coverage
        for target in scenario.optional_fields:
            alias, coverage = _best_alias(records, scenario.field_aliases.get(target, (target,)))
            if alias and coverage > 0:
                mapping[target] = FieldMappingExpression(kind="direct", source=alias).to_dict()
        minimum = min(coverages, default=0.0)
        if minimum < 0.5:
            continue
        candidates.append(
            SchemaCandidate(
                scenario_id=scenario.id,
                scenario_revision_id=scenario.revision_id,
                label=scenario.label,
                canonical_schema=scenario.canonical_schema,
                confidence="low",
                confidence_score=minimum,
                required_coverage=minimum,
                required_field_coverage=field_coverages,
                safe_transform_count=safe_transforms,
                suggested_mapping=mapping,
                reasons=tuple(reasons),
                blockers=(() if minimum >= 0.95 else ("Required-field coverage is below 95%.",)),
            )
        )
    eligible = [
        item
        for item in candidates
        if item.required_coverage >= 0.95 and item.safe_transform_count <= 1
    ]
    unique_high = [
        item
        for item in eligible
        if item.required_coverage >= 0.99 and item.safe_transform_count == 0
    ]
    output = []
    for candidate in sorted(candidates, key=lambda item: (-item.confidence_score, item.label)):
        confidence = "low"
        if (
            len(unique_high) == 1
            and candidate.scenario_revision_id == unique_high[0].scenario_revision_id
        ):
            confidence = "high"
        elif candidate in eligible:
            confidence = "medium"
        output.append(SchemaCandidate(**{**candidate.__dict__, "confidence": confidence}))
    return output


def inspect_path(
    path: Path | str,
    *,
    registry: TrainingScenarioRegistry = TRAINING_SCENARIOS,
    progress: Optional[Any] = None,
    cancelled: Optional[Any] = None,
) -> Dict[str, Any]:
    selected = Path(path).expanduser()
    if selected.is_symlink():
        raise ValueError("symbolic-link dataset sources are not accepted")
    source = selected.resolve()
    fingerprint, size_bytes, file_count = fingerprint_path(source)
    rng = random.Random(42)
    head: list[tuple[int, Dict[str, Any]]] = []
    reservoir: list[tuple[int, Dict[str, Any]]] = []
    issues: list[Dict[str, Any]] = []
    field_counters: Dict[str, Dict[str, Any]] = {}
    total = parsed = valid = invalid = 0
    media_summary = {
        "referenced": 0,
        "verified": 0,
        "missing": 0,
        "unsafe": 0,
        "image_references": 0,
        "audio_references": 0,
    }
    asset_fingerprints: Dict[tuple[str, str], Dict[str, Any]] = {}
    for record, issue in iter_source_records(source):
        if cancelled and cancelled():
            raise RuntimeError("inspection cancelled")
        source_index = total
        total += 1
        if issue is not None:
            invalid += 1
            if len(issues) < 500:
                issues.append(dict(issue))
        elif record is not None:
            parsed += 1
            _update_field_counters(field_counters, record)
            media_issues, media_counts = _record_media_evidence(
                record,
                source_index=source_index,
                asset_fingerprints=asset_fingerprints,
            )
            for key, count in media_counts.items():
                media_summary[key] += count
            if media_issues:
                invalid += 1
                if len(issues) < 500:
                    issues.extend(media_issues[: 500 - len(issues)])
            else:
                valid += 1
            if len(head) < PREVIEW_HEAD_COUNT:
                head.append((source_index, dict(record)))
            else:
                seen_after_head = parsed - PREVIEW_HEAD_COUNT
                if len(reservoir) < PREVIEW_RESERVOIR_COUNT:
                    reservoir.append((source_index, dict(record)))
                else:
                    slot = rng.randrange(seen_after_head)
                    if slot < PREVIEW_RESERVOIR_COUNT:
                        reservoir[slot] = (source_index, dict(record))
        if progress and total % 1000 == 0:
            progress(total)
    sampled_pairs = head + sorted(reservoir, key=lambda item: item[0])
    sample = [record for _, record in sampled_pairs]
    candidates = infer_schema_candidates(sample, registry=registry)
    selected = next(
        (item.scenario_revision_id for item in candidates if item.confidence == "high"), None
    )
    assets = [asset_fingerprints[key] for key in sorted(asset_fingerprints)]
    source_identity = hashlib.sha256(
        json.dumps(
            {
                "source_probe_fingerprint": fingerprint,
                "assets": [
                    {
                        "field": item["field"],
                        "reference": item["reference"],
                        "fingerprint": item["fingerprint"],
                        "missing": item["missing"],
                        "unsafe": item["unsafe"],
                    }
                    for item in assets
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "source_fingerprint": source_identity,
        "import_adapter_version": IMPORT_ADAPTER_VERSION,
        "scenario_registry_revision": registry.revision,
        "scenario_revision_id": selected,
        "total_records": total,
        "valid_records": valid,
        "invalid_records": invalid,
        "sample_count": len(sample),
        "size_bytes": size_bytes,
        "file_count": file_count,
        "fields": _exact_field_profiles(field_counters, total_valid=parsed),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "preview": sample,
        "issues": issues,
        "statistics": {
            "source_probe_fingerprint": fingerprint,
            "asset_fingerprints": assets,
            "preview_policy": {
                "head": PREVIEW_HEAD_COUNT,
                "reservoir": PREVIEW_RESERVOIR_COUNT,
                "seed": 42,
            },
            "sampled": len(sample) < parsed,
            "source_path": str(source),
            "media_summary": media_summary,
        },
    }


__all__ = [
    "IMPORT_ADAPTER_VERSION",
    "fingerprint_path",
    "infer_schema_candidates",
    "inspect_path",
    "iter_file_records",
    "iter_source_records",
]
