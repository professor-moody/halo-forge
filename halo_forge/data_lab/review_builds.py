"""Publish immutable Dataset Lab versions from reviewed label-set revisions."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .errors import SchemaError, VersionError
from .identity import (
    INTERNAL_LINEAGE_KEY,
    derive_record_identity,
    record_hash,
    seed_record_identity,
    strip_internal_identity,
)
from .models import infer_schema, validate_record
from .recipe import Recipe, RecipeResult, StepProvenance
from .sources import (
    ASSET_FIELDS,
    AssetFingerprint,
    SourceSnapshot,
    SourceSpec,
    fingerprint_assets,
    load_source,
)
from .storage import DatasetVersion, VersionStore


REVIEW_BUILD_MODES = frozenset(
    {"filter", "replace_by_record_id", "append", "annotate"}
)
PROTECTED_REVIEW_TARGET_SPLITS = frozenset({"test", "canary"})


def _value(value: Any) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise VersionError("label-set revisions and items must be objects")
    return copy.deepcopy(dict(value))


def _record_id(record: Mapping[str, Any]) -> Optional[str]:
    marker = record.get(INTERNAL_LINEAGE_KEY)
    if isinstance(marker, Mapping) and marker.get("record_id"):
        return str(marker["record_id"])
    for key in ("record_id", "source_record_id"):
        if record.get(key):
            return str(record[key])
    return None


def _load_manifest(version: DatasetVersion) -> Dict[str, Any]:
    try:
        return json.loads((Path(version.path) / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VersionError(f"Cannot read parent version manifest: {version.version_id}") from exc


def _parent_assets(manifest: Mapping[str, Any]) -> List[AssetFingerprint]:
    values: List[AssetFingerprint] = []
    for raw in manifest.get("asset_fingerprints") or []:
        if not isinstance(raw, Mapping) or not raw.get("reference"):
            continue
        values.append(
            AssetFingerprint(
                field=str(raw.get("field") or "asset"),
                reference=str(raw["reference"]),
                resolved_path=(
                    str(raw["resolved_path"]) if raw.get("resolved_path") else None
                ),
                fingerprint=(
                    str(raw["fingerprint"]) if raw.get("fingerprint") else None
                ),
                size_bytes=(
                    int(raw["size_bytes"]) if raw.get("size_bytes") is not None else None
                ),
                missing=bool(raw.get("missing", False)),
            )
        )
    return values


def _merge_assets(*groups: Iterable[AssetFingerprint]) -> List[AssetFingerprint]:
    merged: Dict[tuple[str, str], AssetFingerprint] = {}
    for group in groups:
        for asset in group:
            key = (asset.field, asset.reference)
            prior = merged.get(key)
            if prior is None or (prior.missing and not asset.missing):
                merged[key] = asset
    return [merged[key] for key in sorted(merged)]


def _asset_references(record: Mapping[str, Any]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for field_name in ASSET_FIELDS:
        raw = record.get(field_name)
        values = raw if isinstance(raw, list) else [raw]
        for value in values:
            if isinstance(value, (str, Path)):
                reference = str(value)
            elif isinstance(value, Mapping):
                reference = str(value.get("path") or value.get("filename") or "")
            else:
                reference = str(getattr(value, "filename", "") or "")
            if reference:
                result.add((field_name, reference))
    return result


def _asset_split_contamination(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    assets: Sequence[AssetFingerprint] = (),
) -> Dict[str, Any]:
    fingerprints: Dict[tuple[str, str], set[str]] = {}
    for asset in assets:
        identity = (
            f"sha256:{asset.fingerprint}"
            if asset.fingerprint
            else f"reference:{asset.field}:{asset.reference}"
        )
        fingerprints.setdefault((asset.field, asset.reference), set()).add(identity)
    identities: Dict[str, set[str]] = {}
    for split_name, rows in splits.items():
        split_values: set[str] = set()
        for row in rows:
            for field_name, reference in _asset_references(row):
                split_values.update(
                    fingerprints.get(
                        (field_name, reference),
                        {f"reference:{field_name}:{reference}"},
                    )
                )
        identities[str(split_name)] = split_values
    pairs: Dict[str, Any] = {}
    match_count = 0
    names = sorted(identities)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            shared = sorted(identities[left_name].intersection(identities[right_name]))
            match_count += len(shared)
            pairs[f"{left_name}:{right_name}"] = {
                "count": len(shared),
                "asset_identities": shared[:100],
            }
    return {
        "method": "content_fingerprint_with_reference_fallback",
        "match_count": match_count,
        "pairs": pairs,
    }


@dataclass(frozen=True)
class ReviewDatasetBuildPreview:
    label_set_revision_id: str
    dataset_id: str
    parent_version_id: Optional[str]
    build_mode: str
    target_split: str
    source_count: int
    output_count: int
    added_count: int
    removed_count: int
    replaced_count: int
    annotated_count: int
    excluded_count: int
    quarantined_count: int
    split_counts: Dict[str, int]
    moved_from_splits: Dict[str, int] = field(default_factory=dict)
    contamination: Dict[str, Any] = field(default_factory=dict)
    sample: tuple[Dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["sample"] = [copy.deepcopy(row) for row in self.sample]
        value["warnings"] = list(self.warnings)
        value["starts_training"] = False
        return value


@dataclass
class _ResolvedReviewBuild:
    revision: Dict[str, Any]
    items: List[Dict[str, Any]]
    records: List[Dict[str, Any]]
    splits: Dict[str, List[Dict[str, Any]]]
    parent: Optional[DatasetVersion]
    parent_manifest: Dict[str, Any]
    preview: ReviewDatasetBuildPreview
    quarantined: List[Dict[str, Any]] = field(default_factory=list)


class ReviewDatasetBuilder:
    """Merge reviewed outputs into a parent or new Dataset Lab version."""

    def __init__(self, store: VersionStore):
        self.store = store

    @staticmethod
    def _outputs(
        item: Mapping[str, Any], *, revision_id: str, parent_by_id: Mapping[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if bool(item.get("excluded")):
            return []
        record_id = str(item.get("record_id") or "").strip()
        values = item.get("output_records") or []
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise VersionError("label-set output_records must be an array")
        result: List[Dict[str, Any]] = []
        for index, raw in enumerate(values):
            if not isinstance(raw, Mapping):
                raise VersionError("every reviewed output record must be an object")
            output = copy.deepcopy(dict(raw))
            parent = parent_by_id.get(record_id)
            if parent is not None and isinstance(parent.get(INTERNAL_LINEAGE_KEY), Mapping):
                output[INTERNAL_LINEAGE_KEY] = copy.deepcopy(parent[INTERNAL_LINEAGE_KEY])
            else:
                output = seed_record_identity(
                    output,
                    source_name=f"label-set:{revision_id}",
                    source_index=index,
                )
                marker = output.get(INTERNAL_LINEAGE_KEY)
                if record_id and isinstance(marker, dict):
                    marker["record_id"] = record_id
            derive_record_identity(
                output,
                "review_label_set",
                label_set_revision_id=revision_id,
                review_item_id=item.get("review_item_id"),
                output_index=index,
            )
            result.append(output)
        return result

    def resolve(
        self,
        revision: Any,
        label_items: Sequence[Any],
        *,
        dataset_id: str,
        parent_version_id: Optional[str] = None,
        build_mode: str = "append",
        target_split: str = "train",
        schema: Optional[str] = None,
    ) -> _ResolvedReviewBuild:
        revision_value = _value(revision)
        revision_id = str(revision_value.get("id") or "").strip()
        if not revision_id:
            raise VersionError("label-set revision requires an ID")
        mode = str(build_mode or "append").strip().lower().replace("-", "_")
        if mode == "replace":
            mode = "replace_by_record_id"
        if mode not in REVIEW_BUILD_MODES:
            raise VersionError(
                f"review build mode must be one of {sorted(REVIEW_BUILD_MODES)}"
            )
        dataset_id = str(dataset_id or "").strip()
        target_split = str(target_split or "train").strip()
        if not dataset_id or not target_split:
            raise VersionError("review builds require dataset_id and target_split")
        if target_split.lower() in PROTECTED_REVIEW_TARGET_SPLITS:
            raise VersionError(
                f"reviewed outputs cannot be published into protected split {target_split!r}"
            )
        parent: Optional[DatasetVersion] = None
        parent_manifest: Dict[str, Any] = {}
        parent_records: List[Dict[str, Any]] = []
        parent_splits: Dict[str, List[Dict[str, Any]]] = {}
        if parent_version_id:
            parent = self.store.get_any(parent_version_id, dataset_id=dataset_id)
            parent_manifest = _load_manifest(parent)
            parent_records = self.store.load_records_with_lineage(
                parent.version_id, dataset_id=parent.dataset_id
            )
            for split_name in dict(parent_manifest.get("split_counts") or {}):
                parent_splits[str(split_name)] = self.store.load_records_with_lineage(
                    parent.version_id,
                    dataset_id=parent.dataset_id,
                    split=str(split_name),
                )

        parent_by_id: Dict[str, Dict[str, Any]] = {}
        for row in parent_records:
            identity = _record_id(row)
            if identity:
                parent_by_id.setdefault(identity, row)

        items = [_value(item) for item in label_items]
        raw_output_by_item: List[tuple[Dict[str, Any], List[Dict[str, Any]]]] = [
            (
                item,
                self._outputs(item, revision_id=revision_id, parent_by_id=parent_by_id),
            )
            for item in items
        ]
        validation_schema = schema or (parent.schema if parent else None)
        if validation_schema is None:
            first_output = next(
                (
                    output
                    for _, item_outputs in raw_output_by_item
                    for output in item_outputs
                ),
                None,
            )
            if first_output is not None:
                validation_schema = infer_schema(first_output).value
        quarantined: List[Dict[str, Any]] = []
        output_by_item: List[tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
        for item, item_outputs in raw_output_by_item:
            accepted_outputs: List[Dict[str, Any]] = []
            for output in item_outputs:
                if validation_schema is None:
                    accepted_outputs.append(output)
                    continue
                try:
                    validate_record(strip_internal_identity(output), validation_schema)
                    accepted_outputs.append(output)
                except SchemaError as exc:
                    rejected = copy.deepcopy(output)
                    rejected["_rejection_reason"] = f"validation:{exc}"
                    rejected["_review_item_id"] = item.get("review_item_id")
                    quarantined.append(rejected)
            output_by_item.append((item, accepted_outputs))
        excluded_count = sum(bool(item.get("excluded")) for item in items)
        moved: Dict[str, int] = {}
        removed_ids: set[str] = set()
        # Rendered label-set rows are also the complete source for a new
        # logical dataset.  Parent-specific mode semantics only control how
        # those rows are merged when a parent exists.
        outputs: List[Dict[str, Any]] = [
            output
            for _, item_outputs in output_by_item
            for output in item_outputs
        ]
        replacement_ids = {
            str(item["record_id"])
            for item, item_outputs in output_by_item
            if item_outputs and item.get("record_id")
        }

        if mode == "filter":
            removed_ids = {
                str(item.get("record_id"))
                for item, _ in output_by_item
                if not item.get("excluded")
                and isinstance(item.get("annotation"), Mapping)
                and item["annotation"].get("accepted") is False
            }
            records = [row for row in parent_records if _record_id(row) not in removed_ids]
            splits = {
                split: [row for row in rows if _record_id(row) not in removed_ids]
                for split, rows in parent_splits.items()
            }
        else:
            if mode in {"replace_by_record_id", "annotate"}:
                records = [
                    row for row in parent_records if _record_id(row) not in replacement_ids
                ] + copy.deepcopy(outputs)
                splits = {}
                for split_name, rows in parent_splits.items():
                    retained = [row for row in rows if _record_id(row) not in replacement_ids]
                    removed = len(rows) - len(retained)
                    if removed:
                        moved[split_name] = removed
                    splits[split_name] = retained
                splits.setdefault(target_split, []).extend(copy.deepcopy(outputs))
            else:
                records = copy.deepcopy(parent_records) + copy.deepcopy(outputs)
                splits = copy.deepcopy(parent_splits)
                # A reviewed derivative must not leave its source identity in a
                # held-out split while adding it to train.
                for split_name, rows in list(splits.items()):
                    if split_name == target_split:
                        continue
                    retained = [row for row in rows if _record_id(row) not in replacement_ids]
                    removed = len(rows) - len(retained)
                    if removed:
                        moved[split_name] = removed
                    splits[split_name] = retained
                splits.setdefault(target_split, []).extend(copy.deepcopy(outputs))
        if not parent:
            records = copy.deepcopy(outputs)
            splits = {target_split: copy.deepcopy(outputs)}
        removed_count = sum(_record_id(row) in removed_ids for row in parent_records)
        replaced_count = (
            sum(_record_id(row) in replacement_ids for row in parent_records)
            if mode in {"replace_by_record_id", "annotate"}
            else 0
        )
        development_exposure = any(
            isinstance(item.get("lineage"), Mapping)
            and isinstance(item["lineage"].get("source"), Mapping)
            and str(item["lineage"]["source"].get("purpose") or "").strip().lower()
            == "development"
            for item in items
        )
        warnings: List[str] = []
        if moved:
            warnings.append(
                "Reviewed source identities were removed from held-out splits."
            )
        if development_exposure:
            warnings.append(
                "This child version incorporates reviewed development-suite evidence; "
                "descendants cannot treat that suite as untouched evidence."
            )
        contamination: Dict[str, Any] = {"method": "exact_record_hash", "pairs": {}}
        split_hashes = {
            split_name: {
                record_hash(strip_internal_identity(row)): _record_id(row)
                for row in rows
            }
            for split_name, rows in splits.items()
        }
        split_names = sorted(split_hashes)
        contamination_count = 0
        for left_index, left_name in enumerate(split_names):
            for right_name in split_names[left_index + 1 :]:
                shared = sorted(
                    set(split_hashes[left_name]).intersection(split_hashes[right_name])
                )
                contamination_count += len(shared)
                contamination["pairs"][f"{left_name}:{right_name}"] = {
                    "count": len(shared),
                    "record_hashes": shared[:100],
                }
        contamination["match_count"] = contamination_count
        asset_contamination = _asset_split_contamination(splits)
        contamination["assets"] = asset_contamination
        if contamination_count:
            warnings.append(
                f"Exact contamination checks found {contamination_count} record overlap(s) "
                "between dataset splits."
            )
        if asset_contamination["match_count"]:
            warnings.append(
                f"Media contamination checks found {asset_contamination['match_count']} "
                "asset overlap(s) between dataset splits."
            )
        preview = ReviewDatasetBuildPreview(
            label_set_revision_id=revision_id,
            dataset_id=dataset_id,
            parent_version_id=parent.version_id if parent else None,
            build_mode=mode,
            target_split=target_split,
            source_count=len(parent_records),
            output_count=len(records),
            added_count=(
                len(outputs)
                if not parent or mode == "append"
                else (
                    max(0, len(outputs) - replaced_count)
                    if mode in {"replace_by_record_id", "annotate"}
                    else 0
                )
            ),
            removed_count=removed_count,
            replaced_count=replaced_count if mode == "replace_by_record_id" else 0,
            annotated_count=replaced_count if mode == "annotate" else 0,
            excluded_count=excluded_count,
            quarantined_count=len(quarantined),
            split_counts={key: len(value) for key, value in splits.items()},
            moved_from_splits=moved,
            contamination=contamination,
            sample=tuple(strip_internal_identity(row) for row in records[:25]),
            warnings=tuple(warnings),
        )
        return _ResolvedReviewBuild(
            revision=revision_value,
            items=items,
            records=records,
            splits=splits,
            parent=parent,
            parent_manifest=parent_manifest,
            preview=preview,
            quarantined=quarantined,
        )

    def preview(self, revision: Any, label_items: Sequence[Any], **options: Any) -> ReviewDatasetBuildPreview:
        return self.resolve(revision, label_items, **options).preview

    def build(
        self,
        revision: Any,
        label_items: Sequence[Any],
        *,
        dataset_id: str,
        parent_version_id: Optional[str] = None,
        build_mode: str = "append",
        target_split: str = "train",
        materialize_assets: Optional[bool] = None,
        schema: Optional[str] = None,
        check_cancelled: Optional[Callable[[], None]] = None,
    ) -> DatasetVersion:
        cancel = check_cancelled or (lambda: None)
        cancel()
        resolved = self.resolve(
            revision,
            label_items,
            dataset_id=dataset_id,
            parent_version_id=parent_version_id,
            build_mode=build_mode,
            target_split=target_split,
            schema=schema,
        )
        cancel()
        if not resolved.records:
            raise VersionError("reviewed dataset build would publish zero valid records")
        revision_id = resolved.preview.label_set_revision_id
        inferred_schema = schema or (
            resolved.parent.schema if resolved.parent else infer_schema(resolved.records[0]).value
        )
        exposure_records: List[Dict[str, Any]] = []
        for item in resolved.items:
            lineage = item.get("lineage") if isinstance(item.get("lineage"), Mapping) else {}
            source = lineage.get("source") if isinstance(lineage, Mapping) else {}
            if not isinstance(source, Mapping):
                continue
            suite_revision_id = str(source.get("suite_revision_id") or "").strip()
            if not suite_revision_id:
                continue
            exposure_records.append(
                {
                    "suite_revision_id": suite_revision_id,
                    "suite_item_id": str(
                        source.get("suite_item_id")
                        or source.get("record_id")
                        or item.get("record_id")
                        or ""
                    ),
                    "record_id": item.get("record_id"),
                    "review_item_id": item.get("review_item_id"),
                    "purpose": source.get("purpose"),
                    "source_kind": source.get("kind"),
                    "source_ref": source.get("ref"),
                }
            )
        details = {
            **resolved.preview.to_dict(),
            "label_set_id": resolved.revision.get("label_set_id"),
            "label_set_content_hash": resolved.revision.get("content_hash"),
            "review_item_ids": [item.get("review_item_id") for item in resolved.items],
            "exposure_records": exposure_records,
        }
        recipe = Recipe.from_value(
            {
                "name": "review-label-set-build",
                "schema": inferred_schema,
                "steps": [
                    {
                        "kind": "review_label_set",
                        "revision_id": revision_id,
                        "build_mode": resolved.preview.build_mode,
                        "target_split": target_split,
                        "label_set_id": resolved.revision.get("label_set_id"),
                        "content_hash": resolved.revision.get("content_hash"),
                    }
                ],
            }
        )
        result = RecipeResult(
            records=resolved.records,
            splits=resolved.splits,
            provenance=[
                StepProvenance(
                    kind="review_label_set",
                    input_count=resolved.preview.source_count,
                    output_count=resolved.preview.output_count,
                    rejected_count=resolved.preview.removed_count
                    + resolved.preview.excluded_count
                    + resolved.preview.quarantined_count,
                    details=details,
                )
            ],
            statistics={"review_label_set": details},
            quarantined=resolved.quarantined,
            contamination={
                **resolved.preview.contamination,
                "moved_from_splits": resolved.preview.moved_from_splits,
            },
        )

        label_path = Path(str(resolved.revision.get("storage_path") or ".")).expanduser()
        # Label-set publication names the canonical rendered rows
        # ``canonical.jsonl``.  Older hand-authored revisions used
        # ``records.jsonl``, so retain that read-compatible fallback.
        canonical_path = label_path / "canonical.jsonl"
        records_path = canonical_path if canonical_path.is_file() else label_path / "records.jsonl"
        if resolved.parent:
            parent_path = Path(resolved.parent.path) / "records.jsonl"
            source_path = parent_path
            inherited_assets = _parent_assets(resolved.parent_manifest)
        else:
            source_path = records_path
            inherited_assets = []
        discovered_assets = fingerprint_assets(
            [strip_internal_identity(row) for row in resolved.records],
            base_dir=label_path if label_path.is_dir() else None,
        )
        cancel()
        all_assets = _merge_assets(inherited_assets, discovered_assets)
        asset_contamination = _asset_split_contamination(resolved.splits, all_assets)
        result.contamination["assets"] = asset_contamination
        result.statistics["review_label_set"]["contamination"]["assets"] = copy.deepcopy(
            asset_contamination
        )
        if asset_contamination["match_count"]:
            result.statistics["review_label_set"].setdefault("warnings", []).append(
                f"Media content fingerprints overlap across splits "
                f"({asset_contamination['match_count']} match(es))."
            )
        source_spec = SourceSpec(kind="local", path=str(source_path))
        try:
            live_source = load_source(source_spec)
        except Exception as exc:
            raise VersionError(
                f"review build source is missing or unreadable: {source_path}"
            ) from exc
        source = SourceSnapshot(
            spec=source_spec,
            records=[strip_internal_identity(row) for row in resolved.records],
            # The label-set identity remains in the recipe and version payload;
            # this fingerprint deliberately matches the referenced live file so
            # Dataset Lab source verification remains meaningful at train time.
            fingerprint=live_source.fingerprint,
            assets=all_assets,
            size_bytes=live_source.size_bytes,
            file_count=live_source.file_count,
        )
        resolved_materialization = (
            bool(resolved.parent.materialized_assets)
            if materialize_assets is None and resolved.parent
            else bool(materialize_assets)
        )
        # Cancellation is honored until the atomic publication boundary. A
        # request arriving after this point is intentionally too late: the
        # immutable child will be published and the job will complete rather
        # than reporting a cancelled job that secretly produced data.
        cancel()
        return self.store.publish(
            dataset_id=dataset_id,
            recipe=recipe,
            result=result,
            source=source,
            materialize_assets=resolved_materialization,
            parent_version_id=resolved.parent.version_id if resolved.parent else None,
        )


__all__ = [
    "PROTECTED_REVIEW_TARGET_SPLITS",
    "REVIEW_BUILD_MODES",
    "ReviewDatasetBuildPreview",
    "ReviewDatasetBuilder",
]
