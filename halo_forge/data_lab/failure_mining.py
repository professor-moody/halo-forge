"""Reviewed evaluation-failure mining for immutable Dataset Lab versions.

This module deliberately has no API, CLI, or database dependencies.  It accepts
the stable comparison shape produced by :mod:`halo_forge.evaluation_lab` (or a
plain list of evaluation samples), provides a deterministic review preview, and
publishes a child version through :class:`~halo_forge.data_lab.storage.VersionStore`.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .errors import VersionError
from .identity import (
    INTERNAL_LINEAGE_KEY,
    derive_record_identity,
    seed_record_identity,
    strip_internal_identity,
)
from .recipe import Recipe, RecipeResult, StepProvenance
from .sources import (
    AssetFingerprint,
    SourceSnapshot,
    SourceSpec,
    content_hash,
    fingerprint_assets,
    hash_file,
)
from .storage import DatasetVersion, VersionStore

FAILURE_SELECTION_MODES = frozenset(
    {"candidate_failure", "regression", "improvement", "verifier_disagreement"}
)

_MODE_ALIASES = {
    "candidate-failure": "candidate_failure",
    "candidate failed": "candidate_failure",
    "candidate_failed": "candidate_failure",
    "candidate-failed/base-passed": "regression",
    "candidate_failed_base_passed": "regression",
    "verifier-disagreement": "verifier_disagreement",
    "disagreement": "verifier_disagreement",
}


def _mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    raise VersionError(f"Expected an evaluation object, got {type(value).__name__}")


def _values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None and value != ""), None)


def _nested(mapping: Mapping[str, Any], *path: str) -> Any:
    value: Any = mapping
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


@dataclass(frozen=True)
class FailureMiningSelector:
    """Validated, serializable filters used for a reviewed mining pass.

    Modes are ORed.  Task/category/failure-reason and numeric ranges are then
    applied as AND filters.  A selector can be constructed from dashboard/API
    aliases such as ``type``, ``selectors``, ``score: {min, max}``, and
    ``reward_range``.
    """

    modes: tuple[str, ...] = ("candidate_failure",)
    tasks: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    failure_reasons: tuple[str, ...] = ()
    min_score: Optional[float] = None
    max_score: Optional[float] = None
    min_reward: Optional[float] = None
    max_reward: Optional[float] = None

    def __post_init__(self) -> None:
        modes = tuple(
            _MODE_ALIASES.get(str(mode).strip().lower(), str(mode).strip().lower())
            for mode in self.modes
        )
        if not modes:
            modes = ("candidate_failure",)
        unknown = sorted(set(modes) - FAILURE_SELECTION_MODES)
        if unknown:
            raise VersionError(
                "Unknown failure-mining selector(s) "
                f"{', '.join(unknown)}; choose: {', '.join(sorted(FAILURE_SELECTION_MODES))}"
            )
        if (
            self.min_score is not None
            and self.max_score is not None
            and self.min_score > self.max_score
        ):
            raise VersionError("failure-mining min_score cannot exceed max_score")
        if (
            self.min_reward is not None
            and self.max_reward is not None
            and self.min_reward > self.max_reward
        ):
            raise VersionError("failure-mining min_reward cannot exceed max_reward")
        object.__setattr__(self, "modes", tuple(dict.fromkeys(modes)))
        object.__setattr__(self, "tasks", tuple(dict.fromkeys(str(value) for value in self.tasks)))
        object.__setattr__(
            self, "categories", tuple(dict.fromkeys(str(value) for value in self.categories))
        )
        object.__setattr__(
            self,
            "failure_reasons",
            tuple(dict.fromkeys(str(value) for value in self.failure_reasons)),
        )

    @classmethod
    def from_value(
        cls, value: "FailureMiningSelector | Mapping[str, Any] | str | None"
    ) -> "FailureMiningSelector":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(modes=(value,))
        if not isinstance(value, Mapping):
            raise VersionError("A failure-mining selector must be an object or selector name")
        raw = dict(value)
        mode_value = _first(
            raw.get("modes"),
            raw.get("selectors"),
            raw.get("types"),
            raw.get("mode"),
            raw.get("type"),
            raw.get("kind"),
        )
        score_range = raw.get("score") or raw.get("score_range") or {}
        reward_range = raw.get("reward") or raw.get("reward_range") or {}
        if not isinstance(score_range, Mapping):
            score_range = {}
        if not isinstance(reward_range, Mapping):
            reward_range = {}
        return cls(
            modes=_values(mode_value or "candidate_failure"),
            tasks=_values(_first(raw.get("tasks"), raw.get("task"))),
            categories=_values(_first(raw.get("categories"), raw.get("category"))),
            failure_reasons=_values(_first(raw.get("failure_reasons"), raw.get("failure_reason"))),
            min_score=_number(
                _first(raw.get("min_score"), raw.get("score_min"), score_range.get("min"))
            ),
            max_score=_number(
                _first(raw.get("max_score"), raw.get("score_max"), score_range.get("max"))
            ),
            min_reward=_number(
                _first(raw.get("min_reward"), raw.get("reward_min"), reward_range.get("min"))
            ),
            max_reward=_number(
                _first(raw.get("max_reward"), raw.get("reward_max"), reward_range.get("max"))
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modes": list(self.modes),
            "tasks": list(self.tasks),
            "categories": list(self.categories),
            "failure_reasons": list(self.failure_reasons),
            "min_score": self.min_score,
            "max_score": self.max_score,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
        }


@dataclass(frozen=True)
class FailureMiningCandidate:
    selection_id: str
    record_id: str
    suite_item_id: str
    outcome: str
    candidate_failed: bool
    verifier_disagreement: bool
    task: Optional[str] = None
    category: Optional[str] = None
    failure_reason: Optional[str] = None
    score: Optional[float] = None
    reward: Optional[float] = None
    base: Dict[str, Any] = field(default_factory=dict)
    candidate: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FailureMiningPreview:
    base_evaluation_id: Optional[str]
    candidate_evaluation_id: Optional[str]
    suite_revision_id: Optional[str]
    selector: FailureMiningSelector
    exclusions: tuple[str, ...]
    exclusions_hash: str
    examined_count: int
    matched_count: int
    selected: tuple[FailureMiningCandidate, ...]
    excluded: tuple[FailureMiningCandidate, ...] = ()

    @property
    def selected_count(self) -> int:
        return len(self.selected)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_evaluation_id": self.base_evaluation_id,
            "candidate_evaluation_id": self.candidate_evaluation_id,
            "suite_revision_id": self.suite_revision_id,
            "selector": self.selector.to_dict(),
            "exclusions": list(self.exclusions),
            "exclusions_hash": self.exclusions_hash,
            "examined_count": self.examined_count,
            "matched_count": self.matched_count,
            "selected_count": self.selected_count,
            "selected": [item.to_dict() for item in self.selected],
            "excluded": [item.to_dict() for item in self.excluded],
        }


def normalize_exclusions(exclusions: Optional[Iterable[Any]]) -> tuple[str, ...]:
    """Normalize explicit review exclusions independent of order or duplicates."""

    normalized: set[str] = set()
    for value in exclusions or ():
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        if isinstance(value, Mapping):
            value = _first(
                value.get("selection_id"), value.get("record_id"), value.get("suite_item_id")
            )
        if value is not None and str(value).strip():
            normalized.add(str(value).strip())
    return tuple(sorted(normalized))


def exclusions_hash(exclusions: Optional[Iterable[Any]]) -> str:
    """Hash only the normalized review decision, producing a stable audit key."""

    return content_hash({"exclusions": list(normalize_exclusions(exclusions))})


def _candidate_failed(candidate: Mapping[str, Any], outcome: str) -> bool:
    passed = candidate.get("passed")
    if passed is False:
        return True
    if candidate.get("error") not in (None, ""):
        return True
    return outcome in {"regression", "unchanged_failure", "candidate_failure", "failure"}


def _verdict_values(value: Any) -> List[str]:
    if isinstance(value, Mapping):
        for key in ("passed", "accepted", "verdict", "label", "result"):
            if key in value:
                return [json.dumps(value[key], sort_keys=True, default=str)]
        values: List[str] = []
        for child in value.values():
            values.extend(_verdict_values(child))
        return values
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = []
        for child in value:
            values.extend(_verdict_values(child))
        return values
    if value is None:
        return []
    return [json.dumps(value, sort_keys=True, default=str)]


def _has_verifier_disagreement(item: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    sources: List[Any] = [
        item,
        candidate,
        item.get("metadata"),
        candidate.get("metadata"),
        item.get("verifier_trace"),
        candidate.get("verifier_trace"),
    ]
    for value in sources:
        if not isinstance(value, Mapping):
            continue
        if value.get("verifier_disagreement") is True or value.get("disagreement") is True:
            return True
        if value.get("agreement") is False or value.get("agree") is False:
            return True
        for key in ("verdicts", "verifier_results", "results"):
            if key in value:
                verdicts = _verdict_values(value[key])
                if len(set(verdicts)) > 1:
                    return True
    trace = candidate.get("verifier_trace")
    if isinstance(trace, Sequence) and not isinstance(trace, (str, bytes, bytearray)):
        verdicts = _verdict_values(trace)
        return len(set(verdicts)) > 1
    return False


def _outcome(item: Mapping[str, Any], base: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    explicit = item.get("outcome")
    if explicit:
        return str(explicit)
    before = base.get("passed")
    after = candidate.get("passed")
    if before is True and after is False:
        return "regression"
    if before is False and after is True:
        return "improvement"
    if before is False and after is False:
        return "unchanged_failure"
    if before is True and after is True:
        return "unchanged_pass"
    return "candidate_failure" if _candidate_failed(candidate, "") else "unchanged_pass"


def _metadata_value(item: Mapping[str, Any], candidate: Mapping[str, Any], name: str) -> Any:
    return _first(
        item.get(name),
        candidate.get(name),
        _nested(candidate, "metadata", name),
        _nested(item, "metadata", name),
    )


def _candidate_from_item(item: Mapping[str, Any], index: int) -> FailureMiningCandidate:
    nested_candidate = item.get("candidate")
    candidate = (
        _mapping(nested_candidate) if nested_candidate is not None else copy.deepcopy(dict(item))
    )
    base = _mapping(item.get("base")) if item.get("base") is not None else {}
    outcome = _outcome(item, base, candidate)
    record_id = str(
        _first(
            item.get("record_id"),
            candidate.get("record_id"),
            item.get("suite_item_id"),
            candidate.get("suite_item_id"),
            f"sample-{index}",
        )
    )
    suite_item_id = str(
        _first(item.get("suite_item_id"), candidate.get("suite_item_id"), record_id)
    )
    failure_reason_value = _first(
        _metadata_value(item, candidate, "failure_reason"),
        candidate.get("error"),
        item.get("error"),
    )
    score = _number(_first(candidate.get("score"), item.get("candidate_score"), item.get("score")))
    reward = _number(
        _first(
            candidate.get("reward"),
            _nested(candidate, "metadata", "reward"),
            _nested(candidate, "verifier_trace", "reward"),
            item.get("candidate_reward"),
            item.get("reward"),
        )
    )
    selection_id = "sel_" + content_hash(
        {
            "record_id": record_id,
            "suite_item_id": suite_item_id,
            "outcome": outcome,
            "base": base,
            "candidate": candidate,
        }
    )
    return FailureMiningCandidate(
        selection_id=selection_id,
        record_id=record_id,
        suite_item_id=suite_item_id,
        outcome=outcome,
        candidate_failed=_candidate_failed(candidate, outcome),
        verifier_disagreement=_has_verifier_disagreement(item, candidate),
        task=(
            str(value) if (value := _metadata_value(item, candidate, "task")) is not None else None
        ),
        category=(
            str(value)
            if (value := _metadata_value(item, candidate, "category")) is not None
            else None
        ),
        failure_reason=str(failure_reason_value) if failure_reason_value is not None else None,
        score=score,
        reward=reward,
        base=base,
        candidate=candidate,
    )


def _in_range(value: Optional[float], minimum: Optional[float], maximum: Optional[float]) -> bool:
    if minimum is None and maximum is None:
        return True
    if value is None:
        return False
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def _matches(candidate: FailureMiningCandidate, selector: FailureMiningSelector) -> bool:
    mode_matches = {
        "candidate_failure": candidate.candidate_failed,
        "regression": candidate.outcome == "regression",
        "improvement": candidate.outcome == "improvement",
        "verifier_disagreement": candidate.verifier_disagreement,
    }
    if not any(mode_matches[mode] for mode in selector.modes):
        return False
    if selector.tasks and candidate.task not in selector.tasks:
        return False
    if selector.categories and candidate.category not in selector.categories:
        return False
    if selector.failure_reasons and candidate.failure_reason not in selector.failure_reasons:
        return False
    if not _in_range(candidate.score, selector.min_score, selector.max_score):
        return False
    return _in_range(candidate.reward, selector.min_reward, selector.max_reward)


def _comparison_rows(comparison: Any) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if hasattr(comparison, "to_dict"):
        comparison = comparison.to_dict()
    if isinstance(comparison, Mapping):
        payload = copy.deepcopy(dict(comparison))
        values = _first(
            payload.get("sample_deltas"),
            payload.get("samples"),
            payload.get("evaluation_samples"),
        )
        if values is None and any(
            key in payload for key in ("record_id", "suite_item_id", "candidate")
        ):
            values = [payload]
        if values is None:
            values = []
    elif isinstance(comparison, Sequence) and not isinstance(comparison, (str, bytes, bytearray)):
        payload = {}
        values = comparison
    else:
        raise VersionError("Evaluation comparison must be an object or a sample list")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise VersionError("Evaluation comparison samples must be a list")
    rows = [_mapping(value) for value in values]
    return payload, rows


def preview_failure_mining(
    comparison: Any,
    selector: FailureMiningSelector | Mapping[str, Any] | str | None = None,
    *,
    exclusions: Optional[Iterable[Any]] = None,
) -> FailureMiningPreview:
    """Select and preview evidence without writing a dataset version."""

    resolved_selector = FailureMiningSelector.from_value(selector)
    payload, rows = _comparison_rows(comparison)
    normalized_exclusions = normalize_exclusions(exclusions)
    excluded_values = set(normalized_exclusions)
    matched: List[FailureMiningCandidate] = []
    selected: List[FailureMiningCandidate] = []
    excluded: List[FailureMiningCandidate] = []
    for index, row in enumerate(rows):
        candidate = _candidate_from_item(row, index)
        if not _matches(candidate, resolved_selector):
            continue
        matched.append(candidate)
        identifiers = {candidate.selection_id, candidate.record_id, candidate.suite_item_id}
        if identifiers & excluded_values:
            excluded.append(candidate)
        else:
            selected.append(candidate)
    return FailureMiningPreview(
        base_evaluation_id=(
            str(payload["base_evaluation_id"])
            if payload.get("base_evaluation_id") is not None
            else None
        ),
        candidate_evaluation_id=(
            str(payload["candidate_evaluation_id"])
            if payload.get("candidate_evaluation_id") is not None
            else None
        ),
        suite_revision_id=(
            str(payload["suite_revision_id"])
            if payload.get("suite_revision_id") is not None
            else None
        ),
        selector=resolved_selector,
        exclusions=normalized_exclusions,
        exclusions_hash=exclusions_hash(normalized_exclusions),
        examined_count=len(rows),
        matched_count=len(matched),
        selected=tuple(selected),
        excluded=tuple(excluded),
    )


def _record_payload(candidate: FailureMiningCandidate, schema: Optional[str]) -> Dict[str, Any]:
    sample = candidate.candidate
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else {}
    for container in (sample, metadata):
        for key in ("canonical_record", "source_record", "dataset_record", "record"):
            value = container.get(key)
            if isinstance(value, Mapping):
                return copy.deepcopy(dict(value))
    value = sample.get("input")
    expected = sample.get("expected")
    if isinstance(value, Mapping):
        row = copy.deepcopy(dict(value))
        if expected is not None and not any(
            name in row
            for name in ("response", "reference_answer", "ground_truth", "transcript", "label")
        ):
            row["reference_answer" if schema in {"prompt", "rlvr"} else "response"] = copy.deepcopy(
                expected
            )
        return row
    if schema in {"chat", "tool"} and isinstance(value, list):
        row = {"messages": copy.deepcopy(value)}
        tools = _first(sample.get("tools"), metadata.get("tools"))
        if tools is not None:
            row["tools"] = copy.deepcopy(tools)
        return row
    if schema in {"prompt", "rlvr"}:
        return {"prompt": copy.deepcopy(value), "reference_answer": copy.deepcopy(expected)}
    if schema == "vlm":
        row = {
            "image": copy.deepcopy(_first(sample.get("image"), metadata.get("image"))),
            "prompt": copy.deepcopy(value),
        }
        if expected is not None:
            row["response"] = copy.deepcopy(expected)
        return {key: item for key, item in row.items() if item is not None}
    if schema == "audio":
        row = {
            "audio": copy.deepcopy(_first(sample.get("audio"), metadata.get("audio"))),
            "task": copy.deepcopy(_first(sample.get("task"), metadata.get("task"), value)),
        }
        if expected is not None:
            row["transcript"] = copy.deepcopy(expected)
        return {key: item for key, item in row.items() if item is not None}
    return {
        key: item
        for key, item in {
            "prompt": copy.deepcopy(value),
            "response": copy.deepcopy(expected),
        }.items()
        if item is not None
    }


def _evaluation_ids(preview: FailureMiningPreview) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            value
            for value in (preview.base_evaluation_id, preview.candidate_evaluation_id)
            if value
        )
    )


def _load_parent_manifest(parent: DatasetVersion) -> Dict[str, Any]:
    path = Path(parent.path) / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VersionError(f"Invalid parent version manifest at {path}: {exc}") from exc


def _parent_assets(parent: DatasetVersion, manifest: Mapping[str, Any]) -> List[AssetFingerprint]:
    parent_path = Path(parent.path)
    raw_assets = [
        AssetFingerprint(**dict(value))
        for value in _nested(manifest, "source", "assets") or ()
        if isinstance(value, Mapping)
    ]
    materialized = dict(manifest.get("asset_mapping") or {})
    if not materialized:
        return raw_assets
    by_reference = {asset.reference: asset for asset in raw_assets}
    assets: List[AssetFingerprint] = []
    for original, relative in sorted(materialized.items()):
        path = (parent_path / str(relative)).resolve()
        original_asset = by_reference.get(str(original))
        exists = path.is_file()
        assets.append(
            AssetFingerprint(
                field=original_asset.field if original_asset else "asset",
                reference=str(relative),
                resolved_path=str(path),
                fingerprint=hash_file(path) if exists else None,
                size_bytes=path.stat().st_size if exists else None,
                missing=not exists,
            )
        )
    return assets


def _merge_assets(*collections: Sequence[AssetFingerprint]) -> List[AssetFingerprint]:
    values: Dict[tuple[str, str], AssetFingerprint] = {}
    for collection in collections:
        for asset in collection:
            key = (asset.field, asset.reference)
            current = values.get(key)
            if current is None or (current.missing and not asset.missing):
                values[key] = asset
    return [values[key] for key in sorted(values)]


class FailureMiningBuilder:
    """Review evaluation evidence and atomically publish a child version."""

    def __init__(self, store: VersionStore):
        self.store = store

    def preview(
        self,
        comparison: Any,
        selector: FailureMiningSelector | Mapping[str, Any] | str | None = None,
        *,
        exclusions: Optional[Iterable[Any]] = None,
    ) -> FailureMiningPreview:
        return preview_failure_mining(comparison, selector, exclusions=exclusions)

    def build(
        self,
        *,
        parent_version_id: str,
        comparison: Any,
        selector: FailureMiningSelector | Mapping[str, Any] | str | None = None,
        exclusions: Optional[Iterable[Any]] = None,
        dataset_id: Optional[str] = None,
        target_split: str = "train",
        mode: str = "append",
        materialize_assets: Optional[bool] = None,
    ) -> DatasetVersion:
        """Publish a reviewed failure-mined version.

        ``append`` (the default) retains the parent records and split bindings and
        adds selected evidence to ``target_split``.  ``replace`` creates a compact
        failure-only child while retaining the explicit parent relationship.
        """

        if mode not in {"append", "replace"}:
            raise VersionError("failure-mining mode must be append or replace")
        target_split = str(target_split).strip()
        if not target_split:
            raise VersionError("failure-mining target_split cannot be empty")
        parent = self.store.get_any(parent_version_id, dataset_id=dataset_id)
        manifest = _load_parent_manifest(parent)
        preview = self.preview(comparison, selector, exclusions=exclusions)
        if not preview.selected:
            raise VersionError("Failure-mining review selected no records")

        parent_records = self.store.load_records_with_lineage(
            parent.version_id, dataset_id=parent.dataset_id
        )
        parent_identities = self.store.load_lineage(parent.version_id, dataset_id=parent.dataset_id)
        parent_by_record_id: Dict[str, Dict[str, Any]] = {}
        for row, identity in zip(parent_records, parent_identities):
            parent_by_record_id.setdefault(identity.record_id, row)

        evaluation_ids = _evaluation_ids(preview)
        mined: List[Dict[str, Any]] = []
        original_record_ids: List[str] = []
        for candidate in preview.selected:
            if candidate.record_id in parent_by_record_id:
                row = copy.deepcopy(parent_by_record_id[candidate.record_id])
            else:
                row = seed_record_identity(
                    _record_payload(candidate, parent.schema),
                    source_name=f"evaluation:{preview.candidate_evaluation_id or 'sample'}",
                )
                marker = row.get(INTERNAL_LINEAGE_KEY)
                if isinstance(marker, dict):
                    marker["record_id"] = candidate.record_id
            derive_record_identity(
                row,
                "evaluation_failure_mining",
                selection_id=candidate.selection_id,
                outcome=candidate.outcome,
                base_evaluation_id=preview.base_evaluation_id,
                candidate_evaluation_id=preview.candidate_evaluation_id,
                suite_revision_id=preview.suite_revision_id,
            )
            mined.append(row)
            original_record_ids.append(candidate.record_id)

        moved_from_splits: Dict[str, int] = {}
        if mode == "append":
            records = copy.deepcopy(parent_records) + copy.deepcopy(mined)
            splits = {
                str(name): self.store.load_records_with_lineage(
                    parent.version_id, dataset_id=parent.dataset_id, split=str(name)
                )
                for name in dict(manifest.get("split_counts") or {})
            }
            # A reviewed record becomes a new occurrence in target_split. Move
            # matching parent occurrences out of every other split so mining
            # cannot silently create train/validation/test/canary leakage.
            selected_record_ids = set(original_record_ids)
            for split_name, split_rows in list(splits.items()):
                if split_name == target_split:
                    continue
                retained = []
                removed = 0
                for row in split_rows:
                    marker = row.get(INTERNAL_LINEAGE_KEY)
                    record_id = marker.get("record_id") if isinstance(marker, Mapping) else None
                    if record_id in selected_record_ids:
                        removed += 1
                    else:
                        retained.append(row)
                if removed:
                    splits[split_name] = retained
                    moved_from_splits[split_name] = removed
            splits.setdefault(target_split, []).extend(copy.deepcopy(mined))
        else:
            records = copy.deepcopy(mined)
            splits = {target_split: copy.deepcopy(mined)}

        details = {
            "evaluation_ids": list(evaluation_ids),
            "base_evaluation_id": preview.base_evaluation_id,
            "candidate_evaluation_id": preview.candidate_evaluation_id,
            "suite_revision_id": preview.suite_revision_id,
            "selector": preview.selector.to_dict(),
            "exclusions": list(preview.exclusions),
            "exclusions_hash": preview.exclusions_hash,
            "original_record_ids": original_record_ids,
            "selected_records": [
                {
                    "selection_id": candidate.selection_id,
                    "record_id": candidate.record_id,
                    "suite_item_id": candidate.suite_item_id,
                    "outcome": candidate.outcome,
                }
                for candidate in preview.selected
            ],
            "parent_version_id": parent.version_id,
            "parent_dataset_id": parent.dataset_id,
            "mode": mode,
            "target_split": target_split,
            "moved_from_splits": moved_from_splits,
            "examined_count": preview.examined_count,
            "matched_count": preview.matched_count,
            "selected_count": preview.selected_count,
            "excluded_count": len(preview.excluded),
        }
        recipe = Recipe.from_value(
            {
                "name": "evaluation-failure-mining",
                "schema": parent.schema,
                "steps": [
                    {
                        "kind": "failure_mining",
                        "source": (
                            f"evaluation-comparison:{preview.candidate_evaluation_id}"
                            if preview.candidate_evaluation_id
                            else "evaluation-samples"
                        ),
                        **copy.deepcopy(details),
                    }
                ],
            }
        )
        result = RecipeResult(
            records=records,
            splits=splits,
            provenance=[
                StepProvenance(
                    kind="failure_mining",
                    input_count=len(parent_records),
                    output_count=len(records),
                    rejected_count=len(preview.excluded),
                    details=copy.deepcopy(details),
                )
            ],
            statistics={"failure_mining": copy.deepcopy(details)},
        )

        inherited_assets = _parent_assets(parent, manifest)
        discovered_assets = fingerprint_assets(
            [strip_internal_identity(row) for row in mined], base_dir=Path(parent.path)
        )
        assets = _merge_assets(inherited_assets, discovered_assets)
        snapshot_fingerprint = content_hash(
            {
                "parent_version_id": parent.version_id,
                "parent_content_hash": parent.content_hash,
                "evaluation_ids": list(evaluation_ids),
                "suite_revision_id": preview.suite_revision_id,
                "selector": preview.selector.to_dict(),
                "exclusions_hash": preview.exclusions_hash,
                "selected": [candidate.selection_id for candidate in preview.selected],
            }
        )
        parent_records_path = Path(parent.path) / "records.jsonl"
        source = SourceSnapshot(
            spec=SourceSpec(kind="local", path=str(parent_records_path)),
            records=[strip_internal_identity(row) for row in records],
            fingerprint=snapshot_fingerprint,
            assets=assets,
            size_bytes=(parent_records_path.stat().st_size if parent_records_path.is_file() else 0),
            file_count=1,
        )
        resolved_materialization = (
            bool(parent.materialized_assets)
            if materialize_assets is None
            else bool(materialize_assets)
        )
        return self.store.publish(
            dataset_id=dataset_id or parent.dataset_id,
            recipe=recipe,
            result=result,
            source=source,
            materialize_assets=resolved_materialization,
            parent_version_id=parent.version_id,
        )


def build_failure_mined_version(store: VersionStore, **kwargs: Any) -> DatasetVersion:
    """Functional wrapper for service layers that do not retain a builder."""

    return FailureMiningBuilder(store).build(**kwargs)


__all__ = [
    "FAILURE_SELECTION_MODES",
    "FailureMiningBuilder",
    "FailureMiningCandidate",
    "FailureMiningPreview",
    "FailureMiningSelector",
    "build_failure_mined_version",
    "exclusions_hash",
    "normalize_exclusions",
    "preview_failure_mining",
]
