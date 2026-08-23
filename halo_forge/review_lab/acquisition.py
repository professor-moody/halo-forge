"""Deterministic acquisition primitives over supplied records and comparisons."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Iterator, Sequence as SequenceABC
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from halo_forge.data_lab.identity import deterministic_record_id, record_hash

from ._canonical import canonical_json, content_hash
from .errors import ReviewEligibilityError, ReviewValidationError
from .models import AcquisitionStrategy

PROTECTED_PURPOSES = frozenset(
    {
        "operational",
        "holdout",
        "final_holdout",
        "final-holdout",
        "test",
        "canary",
        "protected_lineage",
    }
)
PROTECTED_SPLITS = frozenset({"test", "canary"})
STRATEGY_KINDS = frozenset(
    {
        "explicit",
        "candidate_failure",
        "regression",
        "improvement",
        "verifier_disagreement",
        "low_score",
        "low_margin",
        "coverage_gap",
        "diversity",
        "random",
    }
)
FILTER_SCOPES = frozenset({"record", "evidence", "source"})
FILTER_OPERATORS = frozenset({"eq", "in", "range"})
_FILTER_PATH_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def _check_cancelled(callback: Optional[Callable[[], None]], ordinal: Optional[int] = None) -> None:
    """Poll durable cancellation without making tight record loops DB-bound."""

    if callback is not None and (ordinal is None or ordinal % 128 == 0):
        callback()


@dataclass(frozen=True)
class CandidateInput:
    record_id: str
    record_hash: str
    record: Dict[str, Any]
    evidence: Dict[str, Any]
    source: Dict[str, Any]
    source_kind: str
    source_ref: Optional[str]
    source_record_id: Optional[str]
    input_ordinal: int
    stratum: str = "explicit"

    @property
    def score(self) -> Optional[float]:
        raw = self.evidence.get("score")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        value = float(raw)
        return value if math.isfinite(value) else None

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "record_hash": self.record_hash,
            "record": self.record,
            "evidence": self.evidence,
            "source": self.source,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "source_record_id": self.source_record_id,
        }


@dataclass(frozen=True)
class AcquisitionPlan:
    """Fully resolved, content-addressed acquisition input.

    The plan deliberately contains no timestamps or workstation paths.  Its
    identity can therefore be reused by API, CLI, durable workers, and an
    acquisition-manifest store without changing across retries.
    """

    selected: Sequence[CandidateInput]
    eligibility: Dict[str, Any]
    source_hash: str
    source_pins: Tuple[Dict[str, Any], ...]
    request: Dict[str, Any]
    metadata: Dict[str, Any]
    content_hash: str

    @property
    def default_batch_id(self) -> str:
        # ``content_hash`` is the hash of ``identity_payload``. Deriving the
        # identifier from it avoids rebuilding a potentially large selected
        # array while remaining byte-for-byte compatible with the historical
        # ``stable_id("acq", identity_payload)`` implementation.
        return f"acq_{self.content_hash[:24]}"

    def iter_candidate_payloads(self) -> Iterable[Dict[str, Any]]:
        for ordinal, value in enumerate(self.selected):
            yield {
                "ordinal": ordinal,
                **value.identity_payload(),
                "stratum": value.stratum,
                "score": value.score,
            }

    def candidate_payloads(self) -> List[Dict[str, Any]]:
        return list(self.iter_candidate_payloads())

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "identity_version": 1,
            "request": copy.deepcopy(self.request),
            "source_hash": self.source_hash,
            "source_pins": copy.deepcopy(list(self.source_pins)),
            "selected": [
                {**value.identity_payload(), "stratum": value.stratum} for value in self.selected
            ],
            "metadata": copy.deepcopy(self.metadata),
        }


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ReviewValidationError(f"{name} must be an object")
    return copy.deepcopy(dict(value))


def normalize_candidate(value: Mapping[str, Any], ordinal: int) -> CandidateInput:
    if not isinstance(value, Mapping):
        raise ReviewValidationError(f"acquisition record {ordinal} must be an object")
    wrapper = dict(value)
    has_envelope = isinstance(wrapper.get("record"), Mapping)
    record = _mapping(wrapper.get("record") if has_envelope else wrapper, "record")
    evidence = _mapping(wrapper.get("evidence") if has_envelope else None, "evidence")
    source = _mapping(wrapper.get("source") if has_envelope else None, "source")
    if has_envelope:
        for key in (
            "score",
            "score_direction",
            "metric_direction",
            "score_metric",
            "metric_name",
            "metric",
            "passed",
            "mineable",
            "valid",
            "margin",
            "margin_metric",
            "candidate_score",
            "base_score",
            "score_delta",
            "category",
            "task",
            "failure_reason",
            "verifier_disagreement",
            "embedding",
            "embedding_revision",
            "embedding_model_revision",
            "embedding_provenance",
        ):
            if key in wrapper and key not in evidence:
                evidence[key] = copy.deepcopy(wrapper[key])
    logical_id = str(wrapper.get("record_id") or deterministic_record_id(record)).strip()
    if not logical_id:
        logical_id = deterministic_record_id(record)
    actual_hash = record_hash(record)
    supplied_hash = wrapper.get("record_hash")
    if supplied_hash is not None and str(supplied_hash) != actual_hash:
        raise ReviewValidationError(
            f"record_hash mismatch for acquisition record {logical_id}: supplied content changed"
        )
    source_kind = (
        str(source.get("kind") or wrapper.get("source_kind") or "imported").strip().lower()
    )
    source_ref_raw = source.get("ref", source.get("source_ref", wrapper.get("source_ref")))
    source_record_id_raw = source.get(
        "record_id", wrapper.get("source_record_id", wrapper.get("suite_item_id"))
    )
    return CandidateInput(
        record_id=logical_id,
        record_hash=actual_hash,
        record=record,
        evidence=evidence,
        source=source,
        source_kind=source_kind or "imported",
        source_ref=None if source_ref_raw is None else str(source_ref_raw),
        source_record_id=None if source_record_id_raw is None else str(source_record_id_raw),
        input_ordinal=int(ordinal),
    )


def normalize_candidates(records: Iterable[Mapping[str, Any]]) -> List[CandidateInput]:
    return [normalize_candidate(record, index) for index, record in enumerate(records)]


def eligibility_reason(candidate: CandidateInput) -> Optional[str]:
    purpose = (
        str(candidate.source.get("purpose", candidate.evidence.get("suite_purpose", "")) or "")
        .strip()
        .lower()
    )
    split = (
        str(candidate.source.get("split", candidate.evidence.get("split", "")) or "")
        .strip()
        .lower()
    )
    if purpose in PROTECTED_PURPOSES:
        return f"protected_suite_purpose:{purpose}"
    if split in PROTECTED_SPLITS:
        return f"protected_split:{split}"
    if candidate.source.get("protected_lineage") is True:
        return "protected_lineage"
    if candidate.source_kind == "verifier_calibration":
        partition = str(candidate.source.get("partition") or "").strip().lower()
        if partition != "calibration":
            return f"protected_verifier_partition:{partition or 'unspecified'}"
    if candidate.source.get("eligible") is False:
        return "source_ineligible"
    if candidate.source_kind.startswith("evaluation"):
        if candidate.evidence.get("valid") is False:
            return "invalid_evaluation_evidence"
        if candidate.evidence.get("mineable") is False:
            return "non_mineable_evaluation_evidence"
    return None


def _assert_eligible(values: Sequence[CandidateInput], *, context: str = "acquisition") -> None:
    """Reject protected evidence before it can influence a review proposal.

    Comparison acquisition is especially sensitive here: the candidate row is
    the row that eventually enters the queue, but its base-side result still
    influences regression/improvement selection.  Both sides therefore need
    the same eligibility check before any comparison evidence is derived.
    """

    protected = [(value, eligibility_reason(value)) for value in values]
    protected = [(value, reason) for value, reason in protected if reason is not None]
    if not protected:
        return
    summary: Dict[str, int] = {}
    for _, reason in protected:
        summary[str(reason)] = summary.get(str(reason), 0) + 1
    raise ReviewEligibilityError(
        f"{context} contains protected or non-mineable records: "
        + ", ".join(f"{reason}={count}" for reason, count in sorted(summary.items()))
    )


def _deduplicate(values: Sequence[CandidateInput]) -> Tuple[List[CandidateInput], int]:
    # Sorting before first-wins makes input order irrelevant to record-ID
    # collisions while still retaining stable source/input identity in hashes.
    ordered = sorted(
        values,
        key=lambda value: (
            value.record_id,
            value.record_hash,
            value.source_kind,
            value.source_ref or "",
            value.input_ordinal,
        ),
    )
    result: List[CandidateInput] = []
    seen: set[str] = set()
    for value in ordered:
        if value.record_id in seen:
            continue
        seen.add(value.record_id)
        result.append(value)
    return result, len(values) - len(result)


def _boolean(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _normalized_direction(value: Any, *, field: str = "direction") -> Optional[str]:
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "maximize": "maximize",
        "max": "maximize",
        "higher": "maximize",
        "higher_is_better": "maximize",
        "minimize": "minimize",
        "min": "minimize",
        "lower": "minimize",
        "lower_is_better": "minimize",
    }
    if normalized not in aliases:
        raise ReviewValidationError(f"{field} must be maximize or minimize")
    return aliases[normalized]


def _evidence_direction(candidate: CandidateInput) -> Optional[str]:
    raw = candidate.evidence.get("score_direction", candidate.evidence.get("metric_direction"))
    return _normalized_direction(raw, field=f"record {candidate.record_id} score_direction")


def _normalized_metric(value: Any) -> Optional[str]:
    if value is None:
        return None
    result = str(value).strip().lower()
    return result or None


def _evidence_metric(candidate: CandidateInput, *, margin: bool = False) -> Optional[str]:
    keys = (
        ("margin_metric", "score_metric", "metric_name", "metric")
        if margin
        else ("score_metric", "metric_name", "metric")
    )
    for key in keys:
        value = _normalized_metric(candidate.evidence.get(key))
        if value is not None:
            return value
    return None


def _margin_evidence(candidate: CandidateInput) -> Optional[Tuple[float, str]]:
    direct = _finite_number(candidate.evidence.get("margin"))
    if direct is not None:
        return abs(direct), "margin"
    delta = _finite_number(candidate.evidence.get("score_delta"))
    if delta is not None:
        return abs(delta), "score_delta"
    candidate_score = _finite_number(candidate.evidence.get("candidate_score"))
    base_score = _finite_number(candidate.evidence.get("base_score"))
    if candidate_score is not None and base_score is not None:
        return abs(candidate_score - base_score), "paired_scores"
    return None


def _embedding_revision(candidate: CandidateInput) -> Optional[str]:
    provenance = candidate.evidence.get("embedding_provenance")
    nested_revision = provenance.get("revision") if isinstance(provenance, Mapping) else None
    raw = (
        candidate.evidence.get("embedding_revision")
        or candidate.evidence.get("embedding_model_revision")
        or nested_revision
        or candidate.source.get("embedding_revision")
    )
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _embedding_vector(candidate: CandidateInput) -> Tuple[float, ...]:
    raw = candidate.evidence.get("embedding")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ReviewValidationError(
            f"record {candidate.record_id} lacks a non-empty embedding for diversity acquisition"
        )
    try:
        vector = tuple(float(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ReviewValidationError(
            f"record {candidate.record_id} diversity embedding must be numeric"
        ) from exc
    if any(not math.isfinite(value) for value in vector):
        raise ReviewValidationError(
            f"record {candidate.record_id} diversity embedding must be finite"
        )
    return vector


def resolve_acquisition_filters(value: Any = None) -> List[Dict[str, Any]]:
    """Validate and canonicalize safe, deterministic acquisition filters.

    Filters are an AND-only list.  They can inspect dotted paths in a record,
    its evidence, or its source envelope and support only equality, membership,
    and inclusive numeric ranges.  No expression language or callable path is
    accepted.
    """

    if value is None:
        return []
    if isinstance(value, Mapping):
        requested: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        requested = value
    else:
        raise ReviewValidationError("acquisition filters must be an array of objects")
    result: List[Dict[str, Any]] = []
    for index, raw in enumerate(requested):
        if not isinstance(raw, Mapping):
            raise ReviewValidationError(f"acquisition filter {index} must be an object")
        unknown = set(raw) - {"scope", "field", "op", "value", "min", "max"}
        if unknown:
            raise ReviewValidationError(
                f"acquisition filter {index} has unsupported fields: {sorted(unknown)}"
            )
        raw_field = str(raw.get("field") or "").strip()
        if not raw_field:
            raise ReviewValidationError(f"acquisition filter {index} requires field")
        field_parts = raw_field.split(".")
        explicit_scope = str(raw.get("scope") or "").strip().lower()
        prefixed_scope = field_parts[0].lower() if field_parts[0].lower() in FILTER_SCOPES else None
        if explicit_scope and explicit_scope not in FILTER_SCOPES:
            raise ReviewValidationError(
                f"acquisition filter {index} scope must be record, evidence, or source"
            )
        if explicit_scope and prefixed_scope and explicit_scope != prefixed_scope:
            raise ReviewValidationError(
                f"acquisition filter {index} scope conflicts with its field prefix"
            )
        scope = explicit_scope or prefixed_scope or "record"
        if prefixed_scope:
            field_parts = field_parts[1:]
        if not field_parts or any(not _FILTER_PATH_PART.fullmatch(part) for part in field_parts):
            raise ReviewValidationError(
                f"acquisition filter {index} field must be a safe dotted mapping path"
            )
        field = ".".join(field_parts)
        operator = str(raw.get("op") or "eq").strip().lower().replace("-", "_")
        if operator not in FILTER_OPERATORS:
            raise ReviewValidationError(
                f"acquisition filter {index} operator must be eq, in, or range"
            )
        if operator == "eq":
            if "min" in raw or "max" in raw:
                raise ReviewValidationError(
                    f"acquisition filter {index} equality does not accept range bounds"
                )
            if "value" not in raw:
                raise ReviewValidationError(f"acquisition filter {index} equality requires value")
            resolved = {
                "scope": scope,
                "field": field,
                "op": "eq",
                "value": copy.deepcopy(raw["value"]),
            }
            canonical_json(resolved["value"])
        elif operator == "in":
            if "min" in raw or "max" in raw:
                raise ReviewValidationError(
                    f"acquisition filter {index} membership does not accept range bounds"
                )
            candidates = raw.get("value")
            if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
                raise ReviewValidationError(
                    f"acquisition filter {index} membership value must be an array"
                )
            by_identity = {
                canonical_json(candidate): copy.deepcopy(candidate) for candidate in candidates
            }
            resolved = {
                "scope": scope,
                "field": field,
                "op": "in",
                "value": [by_identity[key] for key in sorted(by_identity)],
            }
        else:
            nested = raw.get("value")
            if nested is not None and not isinstance(nested, Mapping):
                raise ReviewValidationError(
                    f"acquisition filter {index} range value must be an object"
                )
            nested = dict(nested or {})
            unknown_range = set(nested) - {"min", "max"}
            if unknown_range:
                raise ReviewValidationError(
                    f"acquisition filter {index} range has unsupported fields: {sorted(unknown_range)}"
                )
            for bound in ("min", "max"):
                if (
                    bound in raw
                    and bound in nested
                    and canonical_json(raw[bound]) != canonical_json(nested[bound])
                ):
                    raise ReviewValidationError(
                        f"acquisition filter {index} has conflicting range {bound} values"
                    )
            lower = raw.get("min", nested.get("min"))
            upper = raw.get("max", nested.get("max"))
            if lower is None and upper is None:
                raise ReviewValidationError(f"acquisition filter {index} range requires min or max")
            lower_number = _finite_number(lower) if lower is not None else None
            upper_number = _finite_number(upper) if upper is not None else None
            if lower is not None and lower_number is None:
                raise ReviewValidationError(
                    f"acquisition filter {index} range min must be finite numeric"
                )
            if upper is not None and upper_number is None:
                raise ReviewValidationError(
                    f"acquisition filter {index} range max must be finite numeric"
                )
            if (
                lower_number is not None
                and upper_number is not None
                and lower_number > upper_number
            ):
                raise ReviewValidationError(
                    f"acquisition filter {index} range min cannot exceed max"
                )
            resolved = {"scope": scope, "field": field, "op": "range"}
            if lower_number is not None:
                resolved["min"] = lower_number
            if upper_number is not None:
                resolved["max"] = upper_number
        result.append(resolved)
    return sorted(result, key=canonical_json)


def _filter_document(candidate: CandidateInput, scope: str) -> Mapping[str, Any]:
    if scope == "record":
        return candidate.record
    if scope == "evidence":
        return candidate.evidence
    return {
        **candidate.source,
        "kind": candidate.source.get("kind", candidate.source_kind),
        "ref": candidate.source.get("ref", candidate.source_ref),
        "record_id": candidate.source.get("record_id", candidate.source_record_id),
    }


def _field_value(document: Mapping[str, Any], field: str) -> Tuple[bool, Any]:
    current: Any = document
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def apply_acquisition_filters(
    values: Sequence[CandidateInput], filters: Sequence[Mapping[str, Any]]
) -> List[CandidateInput]:
    resolved = resolve_acquisition_filters(filters)
    return [candidate for candidate in values if _candidate_matches_filters(candidate, resolved)]


def _candidate_matches_filters(
    candidate: CandidateInput, filters: Sequence[Mapping[str, Any]]
) -> bool:
    for spec in filters:
        found, actual = _field_value(
            _filter_document(candidate, str(spec["scope"])), str(spec["field"])
        )
        if not found:
            return False
        operator = spec["op"]
        if operator == "eq":
            matched = canonical_json(actual) == canonical_json(spec["value"])
        elif operator == "in":
            actual_identity = canonical_json(actual)
            matched = actual_identity in {canonical_json(value) for value in spec["value"]}
        else:
            numeric = _finite_number(actual)
            matched = numeric is not None
            if matched and "min" in spec:
                matched = numeric >= float(spec["min"])
            if matched and "max" in spec:
                matched = numeric <= float(spec["max"])
        if not matched:
            return False
    return True


def _category(candidate: CandidateInput) -> str:
    return str(
        candidate.evidence.get(
            "category",
            candidate.evidence.get("task", candidate.source.get("category", "uncategorized")),
        )
    )


def _distance(left: Sequence[Any], right: Sequence[Any]) -> float:
    if len(left) != len(right) or not left:
        raise ReviewValidationError("diversity embeddings must be non-empty and equal length")
    try:
        values = [(float(a), float(b)) for a, b in zip(left, right)]
    except (TypeError, ValueError) as exc:
        raise ReviewValidationError("diversity embeddings must be numeric") from exc
    if any(not math.isfinite(a) or not math.isfinite(b) for a, b in values):
        raise ReviewValidationError("diversity embeddings must be finite")
    return sum((a - b) ** 2 for a, b in values)


def _pin_score_strategy(
    strategy: AcquisitionStrategy, values: Sequence[CandidateInput]
) -> AcquisitionStrategy:
    scored = [value for value in values if value.score is not None]
    if not scored:
        raise ReviewValidationError("low_score acquisition requires finite numeric score evidence")
    options = copy.deepcopy(strategy.options)
    evidence_directions = {
        direction for value in scored if (direction := _evidence_direction(value)) is not None
    }
    if len(evidence_directions) > 1:
        raise ReviewValidationError("low_score acquisition received incompatible score directions")
    requested_direction = _normalized_direction(options.get("direction"))
    direction = requested_direction or next(iter(evidence_directions), "maximize")
    if evidence_directions and direction not in evidence_directions:
        raise ReviewValidationError(
            "low_score strategy direction is incompatible with supplied score evidence"
        )

    evidence_metrics = {
        metric for value in scored if (metric := _evidence_metric(value)) is not None
    }
    requested_metric = _normalized_metric(options.get("metric", options.get("metric_name")))
    if requested_metric is None and len(evidence_metrics) > 1:
        raise ReviewValidationError(
            "low_score acquisition received incompatible score metrics; pin a single metric"
        )
    metric = requested_metric or next(iter(evidence_metrics), "score")
    incompatible_metrics = sorted(value for value in evidence_metrics if value != metric)
    if incompatible_metrics:
        raise ReviewValidationError(
            "low_score strategy metric is incompatible with supplied score evidence: "
            + ", ".join(incompatible_metrics)
        )
    options.pop("metric_name", None)
    options.update(
        {
            "direction": direction,
            "metric": metric,
            "score_count": len(scored),
            "score_evidence_hash": content_hash(
                [
                    {
                        "record_id": value.record_id,
                        "record_hash": value.record_hash,
                        "score": value.score,
                        "direction": _evidence_direction(value) or direction,
                        "metric": _evidence_metric(value) or metric,
                    }
                    for value in scored
                ]
            ),
        }
    )
    return AcquisitionStrategy(kind="low_score", quota=strategy.quota, options=options)


def _pin_margin_strategy(
    strategy: AcquisitionStrategy, values: Sequence[CandidateInput]
) -> AcquisitionStrategy:
    evidenced = [
        (value, margin) for value in values if (margin := _margin_evidence(value)) is not None
    ]
    if not evidenced:
        raise ReviewValidationError(
            "low_margin acquisition requires finite margin evidence or compatible paired scores"
        )
    options = copy.deepcopy(strategy.options)
    evidence_directions = {
        direction for value, _ in evidenced if (direction := _evidence_direction(value)) is not None
    }
    if len(evidence_directions) > 1:
        raise ReviewValidationError("low_margin acquisition received incompatible score directions")
    requested_direction = _normalized_direction(options.get("direction"))
    direction = requested_direction or next(iter(evidence_directions), None)
    if direction is not None and evidence_directions and direction not in evidence_directions:
        raise ReviewValidationError(
            "low_margin strategy direction is incompatible with supplied score evidence"
        )

    evidence_metrics = {
        metric
        for value, _ in evidenced
        if (metric := _evidence_metric(value, margin=True)) is not None
    }
    requested_metric = _normalized_metric(options.get("metric", options.get("metric_name")))
    if requested_metric is None and len(evidence_metrics) > 1:
        raise ReviewValidationError(
            "low_margin acquisition received incompatible metrics; pin a single metric"
        )
    metric = requested_metric or next(iter(evidence_metrics), "margin")
    incompatible_metrics = sorted(value for value in evidence_metrics if value != metric)
    if incompatible_metrics:
        raise ReviewValidationError(
            "low_margin strategy metric is incompatible with supplied evidence: "
            + ", ".join(incompatible_metrics)
        )
    options.pop("metric_name", None)
    options.update(
        {
            "metric": metric,
            "margin_semantics": "absolute_distance",
            "evidence_fields": sorted({margin[1] for _, margin in evidenced}),
            "evidence_count": len(evidenced),
            "margin_evidence_hash": content_hash(
                [
                    {
                        "record_id": value.record_id,
                        "record_hash": value.record_hash,
                        "margin": margin[0],
                        "evidence_field": margin[1],
                        "direction": _evidence_direction(value) or direction,
                        "metric": _evidence_metric(value, margin=True) or metric,
                    }
                    for value, margin in evidenced
                ]
            ),
        }
    )
    if direction is not None:
        options["direction"] = direction
    return AcquisitionStrategy(kind="low_margin", quota=strategy.quota, options=options)


def _pin_diversity_strategy(
    strategy: AcquisitionStrategy, values: Sequence[CandidateInput]
) -> AcquisitionStrategy:
    options = copy.deepcopy(strategy.options)
    embedding_revision = str(options.get("embedding_revision") or "").strip()
    if not embedding_revision:
        raise ReviewValidationError("diversity acquisition requires a pinned embedding_revision")
    vectors: List[Tuple[CandidateInput, Tuple[float, ...]]] = []
    for value in values:
        observed_revision = _embedding_revision(value)
        if observed_revision is None:
            raise ReviewValidationError(
                f"record {value.record_id} embedding lacks pinned revision evidence"
            )
        if observed_revision != embedding_revision:
            raise ReviewValidationError(
                f"record {value.record_id} embedding revision {observed_revision!r} "
                f"does not match pinned revision {embedding_revision!r}"
            )
        vectors.append((value, _embedding_vector(value)))
    if not vectors:
        raise ReviewValidationError("diversity acquisition requires records with pinned embeddings")
    dimensions = {len(vector) for _, vector in vectors}
    if len(dimensions) != 1:
        raise ReviewValidationError("diversity embeddings must use one pinned vector dimension")
    options.update(
        {
            "embedding_revision": embedding_revision,
            "embedding_dimension": next(iter(dimensions)),
            "embedding_count": len(vectors),
            "embedding_evidence_hash": content_hash(
                [
                    {
                        "record_id": value.record_id,
                        "record_hash": value.record_hash,
                        "revision": embedding_revision,
                        "embedding": list(vector),
                    }
                    for value, vector in vectors
                ]
            ),
        }
    )
    return AcquisitionStrategy(kind="diversity", quota=strategy.quota, options=options)


def resolve_acquisition_strategies(
    values: Sequence[CandidateInput],
    strategies: Optional[Sequence[AcquisitionStrategy | Mapping[str, Any] | str]] = None,
) -> List[AcquisitionStrategy]:
    requested = strategies if strategies is not None else ["explicit"]
    if not requested:
        raise ReviewValidationError("at least one acquisition strategy is required")
    result: List[AcquisitionStrategy] = []
    for raw in requested:
        strategy = AcquisitionStrategy.from_value(raw)
        kind = strategy.kind.strip().lower().replace("-", "_")
        if kind not in STRATEGY_KINDS:
            raise ReviewValidationError(
                f"unsupported acquisition strategy {kind!r}; choose from {sorted(STRATEGY_KINDS)}"
            )
        if strategy.quota is not None and strategy.quota < 0:
            raise ReviewValidationError("acquisition strategy quota cannot be negative")
        options = copy.deepcopy(strategy.options)
        if kind == "explicit" and options.get("record_ids") is not None:
            record_ids = options["record_ids"]
            if not isinstance(record_ids, Sequence) or isinstance(record_ids, (str, bytes)):
                raise ReviewValidationError("explicit record_ids must be an array")
            options["record_ids"] = sorted({str(value) for value in record_ids})
        strategy = AcquisitionStrategy(kind=kind, quota=strategy.quota, options=options)
        if kind == "low_score":
            strategy = _pin_score_strategy(strategy, values)
        elif kind == "low_margin":
            strategy = _pin_margin_strategy(strategy, values)
        elif kind == "diversity":
            strategy = _pin_diversity_strategy(strategy, values)
        result.append(strategy)
    return result


def _select_diverse(
    values: Sequence[CandidateInput], quota: int, *, embedding_revision: str
) -> List[CandidateInput]:
    if not embedding_revision.strip():
        raise ReviewValidationError("diversity acquisition requires a pinned embedding_revision")
    if not values or quota <= 0:
        return []
    vectors = {value.record_id: _embedding_vector(value) for value in values}
    revisions = {_embedding_revision(value) for value in values}
    if revisions != {embedding_revision}:
        raise ReviewValidationError(
            "diversity embeddings do not match the pinned embedding_revision"
        )
    remaining = sorted(values, key=lambda item: (item.record_id, item.record_hash))
    selected = [remaining.pop(0)]
    while remaining and len(selected) < quota:
        ranked = sorted(
            remaining,
            key=lambda item: (
                -min(
                    _distance(vectors[item.record_id], vectors[prior.record_id])
                    for prior in selected
                ),
                item.record_id,
                item.record_hash,
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _strategy_matches(
    strategy: AcquisitionStrategy,
    available: Sequence[CandidateInput],
    *,
    seed: int,
    stratum_index: int,
) -> List[CandidateInput]:
    kind = strategy.kind.strip().lower().replace("-", "_")
    if kind not in STRATEGY_KINDS:
        raise ReviewValidationError(
            f"unsupported acquisition strategy {kind!r}; choose from {sorted(STRATEGY_KINDS)}"
        )
    options = strategy.options
    ordered = sorted(available, key=lambda value: (value.record_id, value.record_hash))
    if kind == "explicit":
        requested = options.get("record_ids")
        if requested is None:
            matches = ordered
        else:
            requested_set = {str(value) for value in requested}
            matches = [value for value in ordered if value.record_id in requested_set]
    elif kind == "candidate_failure":
        matches = [value for value in ordered if _boolean(value.evidence.get("passed")) is False]
    elif kind == "regression":
        matches = [value for value in ordered if value.evidence.get("outcome") == "regression"]
    elif kind == "improvement":
        matches = [value for value in ordered if value.evidence.get("outcome") == "improvement"]
    elif kind == "verifier_disagreement":
        matches = [value for value in ordered if bool(value.evidence.get("verifier_disagreement"))]
    elif kind == "low_score":
        direction = str(options["direction"])
        matches = [value for value in ordered if value.score is not None]
        matches.sort(
            key=lambda value: (
                value.score if direction == "maximize" else -float(value.score or 0.0),
                value.record_id,
            )
        )
    elif kind == "low_margin":
        margins = {
            value.record_id: evidence[0]
            for value in ordered
            if (evidence := _margin_evidence(value)) is not None
        }
        matches = [value for value in ordered if value.record_id in margins]
        matches.sort(key=lambda value: (margins[value.record_id], value.record_id))
    elif kind == "coverage_gap":
        frequencies: Dict[str, int] = {}
        for value in ordered:
            frequencies[_category(value)] = frequencies.get(_category(value), 0) + 1
        matches = sorted(
            ordered,
            key=lambda value: (frequencies[_category(value)], _category(value), value.record_id),
        )
    elif kind == "diversity":
        quota = strategy.quota if strategy.quota is not None else len(ordered)
        matches = _select_diverse(
            ordered,
            quota,
            embedding_revision=str(options.get("embedding_revision") or ""),
        )
    else:  # seeded random baseline
        matches = list(ordered)
        random.Random(f"halo-forge-review:{seed}:{stratum_index}:{kind}").shuffle(matches)
    if kind != "diversity" and strategy.quota is not None:
        if strategy.quota < 0:
            raise ReviewValidationError("acquisition strategy quota cannot be negative")
        matches = matches[: strategy.quota]
    return matches


def acquire(
    records: Iterable[Mapping[str, Any]],
    *,
    strategies: Optional[Sequence[AcquisitionStrategy | Mapping[str, Any] | str]] = None,
    seed: int = 0,
    filters: Any = None,
) -> Tuple[List[CandidateInput], Dict[str, Any], str]:
    plan = plan_acquisition(records, strategies=strategies, seed=seed, filters=filters)
    return list(plan.selected), copy.deepcopy(plan.eligibility), plan.source_hash


def _source_pins(values: Sequence[CandidateInput]) -> Tuple[Dict[str, Any], ...]:
    grouped: Dict[str, Tuple[Dict[str, Any], List[CandidateInput]]] = {}
    for value in values:
        descriptor = {
            "source_kind": value.source_kind,
            "source_ref": value.source_ref,
            "source": copy.deepcopy(value.source),
        }
        key = canonical_json(descriptor)
        if key not in grouped:
            grouped[key] = (descriptor, [])
        grouped[key][1].append(value)
    pins: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        descriptor, records = grouped[key]
        identities = [value.identity_payload() for value in records]
        pins.append(
            {
                **descriptor,
                "record_count": len(records),
                "record_ids_hash": content_hash([value.record_id for value in records]),
                "records_hash": content_hash(identities),
            }
        )
    return tuple(pins)


class _CanonicalArrayHasher:
    """Hash a canonical JSON array without retaining its elements."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(b"[")
        self._first = True
        self._finished = False

    def append_text(self, canonical_value: str) -> None:
        if self._finished:
            raise RuntimeError("canonical array hash is already finalized")
        if not self._first:
            self._digest.update(b",")
        self._digest.update(canonical_value.encode("utf-8"))
        self._first = False

    def append(self, value: Any) -> None:
        self.append_text(canonical_json(value))

    def finish(self) -> str:
        if not self._finished:
            self._digest.update(b"]")
            self._finished = True
        return self._digest.hexdigest()


def _candidate_from_disk_row(row: sqlite3.Row) -> CandidateInput:
    return CandidateInput(
        record_id=str(row["record_id"]),
        record_hash=str(row["record_hash"]),
        record=dict(json.loads(row["record_json"])),
        evidence=dict(json.loads(row["evidence_json"])),
        source=dict(json.loads(row["source_json"])),
        source_kind=str(row["source_kind"]),
        source_ref=(None if row["source_ref"] is None else str(row["source_ref"])),
        source_record_id=(
            None if row["source_record_id"] is None else str(row["source_record_id"])
        ),
        input_ordinal=int(row["input_ordinal"]),
        stratum=str(row["stratum"]) if "stratum" in row.keys() else "explicit",
    )


class _DiskCandidateSequence(SequenceABC[CandidateInput]):
    """A repeatable selected-candidate sequence backed by a temporary SQLite file."""

    def __init__(self, path: str, count: int) -> None:
        self._path = path
        self._count = int(count)
        self._closed = False

    def __len__(self) -> int:
        return self._count

    def _connect(self) -> sqlite3.Connection:
        if self._closed or not os.path.isfile(self._path):
            raise RuntimeError("disk-backed acquisition plan is closed")
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def __iter__(self) -> Iterator[CandidateInput]:
        connection = self._connect()
        try:
            rows = connection.execute("""SELECT c.*,s.stratum
                   FROM selected s JOIN candidates c ON c.row_index=s.candidate_row_index
                   ORDER BY s.ordinal""")
            for row in rows:
                yield _candidate_from_disk_row(row)
        finally:
            connection.close()

    def __getitem__(self, index: int | slice) -> CandidateInput | List[CandidateInput]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return [self[position] for position in range(start, stop, step)]
        position = int(index)
        if position < 0:
            position += self._count
        if position < 0 or position >= self._count:
            raise IndexError(index)
        connection = self._connect()
        try:
            row = connection.execute(
                """SELECT c.*,s.stratum
                   FROM selected s JOIN candidates c ON c.row_index=s.candidate_row_index
                   WHERE s.ordinal=?""",
                (position,),
            ).fetchone()
            if row is None:
                raise IndexError(index)
            return _candidate_from_disk_row(row)
        finally:
            connection.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass

    def __del__(self) -> None:  # pragma: no cover - cleanup timing is runtime-specific
        try:
            self.close()
        except Exception:
            pass


def _disk_rows(
    connection: sqlite3.Connection,
    *,
    available: bool = False,
    where: str = "",
    params: Sequence[Any] = (),
    order_by: str = "d.record_id,d.record_hash",
) -> Iterator[sqlite3.Row]:
    selected_join = "LEFT JOIN selected s ON s.candidate_row_index=d.row_index" if available else ""
    clauses = ["s.candidate_row_index IS NULL"] if available else []
    if where:
        clauses.append(f"({where})")
    where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""
    cursor = connection.execute(
        f"""SELECT d.* FROM candidates d
             JOIN filtered f ON f.candidate_row_index=d.row_index
             {selected_join}{where_sql} ORDER BY {order_by}""",
        tuple(params),
    )
    yield from cursor


def _disk_pin_score_strategy(
    connection: sqlite3.Connection,
    strategy: AcquisitionStrategy,
    *,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> AcquisitionStrategy:
    score_count = 0
    evidence_directions: set[str] = set()
    evidence_metrics: set[str] = set()
    for ordinal, row in enumerate(_disk_rows(connection, where="d.score IS NOT NULL")):
        _check_cancelled(check_cancelled, ordinal)
        value = _candidate_from_disk_row(row)
        score_count += 1
        if (direction := _evidence_direction(value)) is not None:
            evidence_directions.add(direction)
            if len(evidence_directions) > 1:
                raise ReviewValidationError(
                    "low_score acquisition received incompatible score directions"
                )
        if (metric := _evidence_metric(value)) is not None:
            evidence_metrics.add(metric)
            if len(evidence_metrics) > 1 and not strategy.options.get(
                "metric", strategy.options.get("metric_name")
            ):
                raise ReviewValidationError(
                    "low_score acquisition received incompatible score metrics; "
                    "pin a single metric"
                )
    if not score_count:
        raise ReviewValidationError("low_score acquisition requires finite numeric score evidence")
    options = copy.deepcopy(strategy.options)
    requested_direction = _normalized_direction(options.get("direction"))
    direction = requested_direction or next(iter(evidence_directions), "maximize")
    if evidence_directions and direction not in evidence_directions:
        raise ReviewValidationError(
            "low_score strategy direction is incompatible with supplied score evidence"
        )
    requested_metric = _normalized_metric(options.get("metric", options.get("metric_name")))
    metric = requested_metric or next(iter(evidence_metrics), "score")
    incompatible = sorted(value for value in evidence_metrics if value != metric)
    if incompatible:
        raise ReviewValidationError(
            "low_score strategy metric is incompatible with supplied score evidence: "
            + ", ".join(incompatible)
        )
    evidence_hash = _CanonicalArrayHasher()
    for ordinal, row in enumerate(_disk_rows(connection, where="d.score IS NOT NULL")):
        _check_cancelled(check_cancelled, ordinal)
        value = _candidate_from_disk_row(row)
        evidence_hash.append(
            {
                "record_id": value.record_id,
                "record_hash": value.record_hash,
                "score": value.score,
                "direction": _evidence_direction(value) or direction,
                "metric": _evidence_metric(value) or metric,
            }
        )
    options.pop("metric_name", None)
    options.update(
        direction=direction,
        metric=metric,
        score_count=score_count,
        score_evidence_hash=evidence_hash.finish(),
    )
    return AcquisitionStrategy(kind="low_score", quota=strategy.quota, options=options)


def _disk_pin_margin_strategy(
    connection: sqlite3.Connection,
    strategy: AcquisitionStrategy,
    *,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> AcquisitionStrategy:
    evidence_count = 0
    evidence_directions: set[str] = set()
    evidence_metrics: set[str] = set()
    evidence_fields: set[str] = set()
    for ordinal, row in enumerate(_disk_rows(connection, where="d.margin IS NOT NULL")):
        _check_cancelled(check_cancelled, ordinal)
        value = _candidate_from_disk_row(row)
        evidence_count += 1
        evidence_fields.add(str(row["margin_field"]))
        if (direction := _evidence_direction(value)) is not None:
            evidence_directions.add(direction)
            if len(evidence_directions) > 1:
                raise ReviewValidationError(
                    "low_margin acquisition received incompatible score directions"
                )
        if (metric := _evidence_metric(value, margin=True)) is not None:
            evidence_metrics.add(metric)
            if len(evidence_metrics) > 1 and not strategy.options.get(
                "metric", strategy.options.get("metric_name")
            ):
                raise ReviewValidationError(
                    "low_margin acquisition received incompatible metrics; pin a single metric"
                )
    if not evidence_count:
        raise ReviewValidationError(
            "low_margin acquisition requires finite margin evidence or compatible paired scores"
        )
    options = copy.deepcopy(strategy.options)
    requested_direction = _normalized_direction(options.get("direction"))
    direction = requested_direction or next(iter(evidence_directions), None)
    if direction is not None and evidence_directions and direction not in evidence_directions:
        raise ReviewValidationError(
            "low_margin strategy direction is incompatible with supplied score evidence"
        )
    requested_metric = _normalized_metric(options.get("metric", options.get("metric_name")))
    metric = requested_metric or next(iter(evidence_metrics), "margin")
    incompatible = sorted(value for value in evidence_metrics if value != metric)
    if incompatible:
        raise ReviewValidationError(
            "low_margin strategy metric is incompatible with supplied evidence: "
            + ", ".join(incompatible)
        )
    evidence_hash = _CanonicalArrayHasher()
    for ordinal, row in enumerate(_disk_rows(connection, where="d.margin IS NOT NULL")):
        _check_cancelled(check_cancelled, ordinal)
        value = _candidate_from_disk_row(row)
        evidence_hash.append(
            {
                "record_id": value.record_id,
                "record_hash": value.record_hash,
                "margin": float(row["margin"]),
                "evidence_field": str(row["margin_field"]),
                "direction": _evidence_direction(value) or direction,
                "metric": _evidence_metric(value, margin=True) or metric,
            }
        )
    options.pop("metric_name", None)
    options.update(
        metric=metric,
        margin_semantics="absolute_distance",
        evidence_fields=sorted(evidence_fields),
        evidence_count=evidence_count,
        margin_evidence_hash=evidence_hash.finish(),
    )
    if direction is not None:
        options["direction"] = direction
    return AcquisitionStrategy(kind="low_margin", quota=strategy.quota, options=options)


def _disk_pin_diversity_strategy(
    connection: sqlite3.Connection,
    strategy: AcquisitionStrategy,
    *,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> AcquisitionStrategy:
    options = copy.deepcopy(strategy.options)
    revision = str(options.get("embedding_revision") or "").strip()
    if not revision:
        raise ReviewValidationError("diversity acquisition requires a pinned embedding_revision")
    count = 0
    dimension: Optional[int] = None
    evidence_hash = _CanonicalArrayHasher()
    for ordinal, row in enumerate(_disk_rows(connection)):
        _check_cancelled(check_cancelled, ordinal)
        value = _candidate_from_disk_row(row)
        observed = _embedding_revision(value)
        if observed is None:
            raise ReviewValidationError(
                f"record {value.record_id} embedding lacks pinned revision evidence"
            )
        if observed != revision:
            raise ReviewValidationError(
                f"record {value.record_id} embedding revision {observed!r} "
                f"does not match pinned revision {revision!r}"
            )
        vector = _embedding_vector(value)
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ReviewValidationError("diversity embeddings must use one pinned vector dimension")
        evidence_hash.append(
            {
                "record_id": value.record_id,
                "record_hash": value.record_hash,
                "revision": revision,
                "embedding": list(vector),
            }
        )
        count += 1
    if not count:
        raise ReviewValidationError("diversity acquisition requires records with pinned embeddings")
    options.update(
        embedding_revision=revision,
        embedding_dimension=dimension,
        embedding_count=count,
        embedding_evidence_hash=evidence_hash.finish(),
    )
    return AcquisitionStrategy(kind="diversity", quota=strategy.quota, options=options)


def _disk_resolve_strategies(
    connection: sqlite3.Connection,
    strategies: Optional[Sequence[AcquisitionStrategy | Mapping[str, Any] | str]],
    *,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> List[AcquisitionStrategy]:
    requested = strategies if strategies is not None else ["explicit"]
    if not requested:
        raise ReviewValidationError("at least one acquisition strategy is required")
    result: List[AcquisitionStrategy] = []
    for ordinal, raw in enumerate(requested):
        _check_cancelled(check_cancelled, ordinal)
        strategy = AcquisitionStrategy.from_value(raw)
        kind = strategy.kind.strip().lower().replace("-", "_")
        if kind not in STRATEGY_KINDS:
            raise ReviewValidationError(
                f"unsupported acquisition strategy {kind!r}; choose from {sorted(STRATEGY_KINDS)}"
            )
        if strategy.quota is not None and strategy.quota < 0:
            raise ReviewValidationError("acquisition strategy quota cannot be negative")
        options = copy.deepcopy(strategy.options)
        if kind == "explicit" and options.get("record_ids") is not None:
            record_ids = options["record_ids"]
            if not isinstance(record_ids, Sequence) or isinstance(record_ids, (str, bytes)):
                raise ReviewValidationError("explicit record_ids must be an array")
            options["record_ids"] = sorted({str(value) for value in record_ids})
        strategy = AcquisitionStrategy(kind=kind, quota=strategy.quota, options=options)
        if kind == "low_score":
            strategy = _disk_pin_score_strategy(
                connection, strategy, check_cancelled=check_cancelled
            )
        elif kind == "low_margin":
            strategy = _disk_pin_margin_strategy(
                connection, strategy, check_cancelled=check_cancelled
            )
        elif kind == "diversity":
            strategy = _disk_pin_diversity_strategy(
                connection, strategy, check_cancelled=check_cancelled
            )
        result.append(strategy)
    return result


def _append_disk_selection(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    ordinal: int,
    stratum: str,
) -> None:
    connection.execute(
        "INSERT INTO selected (ordinal,candidate_row_index,stratum) VALUES (?,?,?)",
        (ordinal, int(row["row_index"]), stratum),
    )


def _disk_select_random(
    connection: sqlite3.Connection,
    *,
    seed: int,
    stratum_index: int,
    quota: Optional[int],
    stratum: str,
    ordinal: int,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> int:
    connection.execute("DROP TABLE IF EXISTS random_pool")
    connection.execute(
        "CREATE TEMP TABLE random_pool "
        "(position INTEGER PRIMARY KEY,candidate_row_index INTEGER NOT NULL)"
    )
    count = 0
    for pool_ordinal, row in enumerate(_disk_rows(connection, available=True)):
        _check_cancelled(check_cancelled, pool_ordinal)
        connection.execute("INSERT INTO random_pool VALUES (?,?)", (count, int(row["row_index"])))
        count += 1
    generator = random.Random(f"halo-forge-review:{seed}:{stratum_index}:random")
    for right in range(count - 1, 0, -1):
        _check_cancelled(check_cancelled, count - right)
        left = generator.randrange(right + 1)
        if left == right:
            continue
        left_id = connection.execute(
            "SELECT candidate_row_index FROM random_pool WHERE position=?", (left,)
        ).fetchone()[0]
        right_id = connection.execute(
            "SELECT candidate_row_index FROM random_pool WHERE position=?", (right,)
        ).fetchone()[0]
        connection.execute("UPDATE random_pool SET position=-1 WHERE position=?", (left,))
        connection.execute("UPDATE random_pool SET position=? WHERE position=?", (left, right))
        connection.execute("UPDATE random_pool SET position=? WHERE position=-1", (right,))
        # Reading both values above is intentional: it ensures corruption is
        # detected instead of silently changing the deterministic shuffle.
        if left_id is None or right_id is None:  # pragma: no cover - SQLite invariant
            raise RuntimeError("random acquisition pool is inconsistent")
    limit = count if quota is None else min(count, quota)
    selected = 0
    rows = connection.execute(
        """SELECT c.* FROM random_pool p
           JOIN candidates c ON c.row_index=p.candidate_row_index
           ORDER BY p.position LIMIT ?""",
        (limit,),
    )
    for row in rows:
        _append_disk_selection(connection, row, ordinal=ordinal + selected, stratum=stratum)
        selected += 1
    connection.execute("DROP TABLE random_pool")
    return selected


def _update_diversity_distances(
    connection: sqlite3.Connection,
    selected: CandidateInput,
    *,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> None:
    selected_vector = _embedding_vector(selected)
    updates: List[Tuple[float, int]] = []
    rows = connection.execute("""SELECT c.*,p.min_distance FROM diversity_pool p
           JOIN candidates c ON c.row_index=p.candidate_row_index
           ORDER BY c.record_id,c.record_hash""")
    for row_ordinal, row in enumerate(rows):
        _check_cancelled(check_cancelled, row_ordinal)
        value = _candidate_from_disk_row(row)
        distance = _distance(_embedding_vector(value), selected_vector)
        prior = row["min_distance"]
        updates.append(
            (distance if prior is None else min(float(prior), distance), int(row["row_index"]))
        )
        if len(updates) >= 500:
            connection.executemany(
                "UPDATE diversity_pool SET min_distance=? WHERE candidate_row_index=?",
                updates,
            )
            updates.clear()
    if updates:
        connection.executemany(
            "UPDATE diversity_pool SET min_distance=? WHERE candidate_row_index=?",
            updates,
        )


def _disk_select_diversity(
    connection: sqlite3.Connection,
    *,
    quota: Optional[int],
    stratum: str,
    ordinal: int,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> int:
    connection.execute("DROP TABLE IF EXISTS diversity_pool")
    connection.execute("""CREATE TEMP TABLE diversity_pool
           (candidate_row_index INTEGER PRIMARY KEY,min_distance REAL)""")
    count = 0
    for pool_ordinal, row in enumerate(_disk_rows(connection, available=True)):
        _check_cancelled(check_cancelled, pool_ordinal)
        connection.execute(
            "INSERT INTO diversity_pool (candidate_row_index,min_distance) VALUES (?,NULL)",
            (int(row["row_index"]),),
        )
        count += 1
    limit = count if quota is None else min(count, quota)
    if limit <= 0:
        connection.execute("DROP TABLE diversity_pool")
        return 0
    selected_count = 0
    while selected_count < limit:
        _check_cancelled(check_cancelled)
        if selected_count == 0:
            row = connection.execute("""SELECT c.* FROM diversity_pool p
                   JOIN candidates c ON c.row_index=p.candidate_row_index
                   ORDER BY c.record_id,c.record_hash LIMIT 1""").fetchone()
        else:
            row = connection.execute("""SELECT c.* FROM diversity_pool p
                   JOIN candidates c ON c.row_index=p.candidate_row_index
                   ORDER BY p.min_distance DESC,c.record_id,c.record_hash LIMIT 1""").fetchone()
        if row is None:
            break
        value = _candidate_from_disk_row(row)
        _append_disk_selection(
            connection,
            row,
            ordinal=ordinal + selected_count,
            stratum=stratum,
        )
        connection.execute(
            "DELETE FROM diversity_pool WHERE candidate_row_index=?",
            (int(row["row_index"]),),
        )
        selected_count += 1
        if selected_count < limit:
            _update_diversity_distances(connection, value, check_cancelled=check_cancelled)
    connection.execute("DROP TABLE diversity_pool")
    return selected_count


def _disk_select_strategy(
    connection: sqlite3.Connection,
    strategy: AcquisitionStrategy,
    *,
    seed: int,
    stratum_index: int,
    ordinal: int,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> int:
    kind = strategy.kind
    stratum = f"{stratum_index}:{kind}"
    if kind == "random":
        return _disk_select_random(
            connection,
            seed=seed,
            stratum_index=stratum_index,
            quota=strategy.quota,
            stratum=stratum,
            ordinal=ordinal,
            check_cancelled=check_cancelled,
        )
    if kind == "diversity":
        return _disk_select_diversity(
            connection,
            quota=strategy.quota,
            stratum=stratum,
            ordinal=ordinal,
            check_cancelled=check_cancelled,
        )
    quota = strategy.quota
    if quota == 0:
        return 0
    selected_count = 0
    if kind == "low_score":
        direction = str(strategy.options["direction"])
        order = "d.score ASC,d.record_id" if direction == "maximize" else "d.score DESC,d.record_id"
        rows = _disk_rows(
            connection,
            available=True,
            where="d.score IS NOT NULL",
            order_by=order,
        )
    elif kind == "low_margin":
        rows = _disk_rows(
            connection,
            available=True,
            where="d.margin IS NOT NULL",
            order_by="d.margin ASC,d.record_id",
        )
    elif kind == "coverage_gap":
        rows = connection.execute("""SELECT d.* FROM candidates d
               JOIN filtered f ON f.candidate_row_index=d.row_index
               LEFT JOIN selected s ON s.candidate_row_index=d.row_index
               JOIN (
                 SELECT d2.category AS category,COUNT(*) AS frequency
                 FROM candidates d2
                 JOIN filtered f2 ON f2.candidate_row_index=d2.row_index
                 LEFT JOIN selected s2 ON s2.candidate_row_index=d2.row_index
                 WHERE s2.candidate_row_index IS NULL GROUP BY d2.category
               ) frequencies ON frequencies.category=d.category
               WHERE s.candidate_row_index IS NULL
               ORDER BY frequencies.frequency,d.category,d.record_id""")
    else:
        rows = _disk_rows(connection, available=True)
    explicit_ids = (
        {str(value) for value in strategy.options.get("record_ids") or []}
        if kind == "explicit" and strategy.options.get("record_ids") is not None
        else None
    )
    for row_ordinal, row in enumerate(rows):
        _check_cancelled(check_cancelled, row_ordinal)
        value = _candidate_from_disk_row(row)
        if kind == "explicit":
            matched = explicit_ids is None or value.record_id in explicit_ids
        elif kind == "candidate_failure":
            matched = _boolean(value.evidence.get("passed")) is False
        elif kind == "regression":
            matched = value.evidence.get("outcome") == "regression"
        elif kind == "improvement":
            matched = value.evidence.get("outcome") == "improvement"
        elif kind == "verifier_disagreement":
            matched = bool(value.evidence.get("verifier_disagreement"))
        else:
            matched = True
        if not matched:
            continue
        _append_disk_selection(
            connection,
            row,
            ordinal=ordinal + selected_count,
            stratum=stratum,
        )
        selected_count += 1
        if quota is not None and selected_count >= quota:
            break
    return selected_count


def _stream_plan_identity_hash(
    selected: Iterable[CandidateInput],
    *,
    request: Mapping[str, Any],
    source_hash: str,
    source_pins: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    check_cancelled: Optional[Callable[[], None]] = None,
) -> str:
    digest = hashlib.sha256()
    # Keys are emitted in the same lexicographic order used by
    # ``canonical_json(..., sort_keys=True)``.
    digest.update(b'{"identity_version":1,"metadata":')
    digest.update(canonical_json(metadata).encode("utf-8"))
    digest.update(b',"request":')
    digest.update(canonical_json(request).encode("utf-8"))
    digest.update(b',"selected":[')
    first = True
    for ordinal, value in enumerate(selected):
        _check_cancelled(check_cancelled, ordinal)
        if not first:
            digest.update(b",")
        digest.update(
            canonical_json({**value.identity_payload(), "stratum": value.stratum}).encode("utf-8")
        )
        first = False
    digest.update(b'],"source_hash":')
    digest.update(canonical_json(source_hash).encode("utf-8"))
    digest.update(b',"source_pins":')
    digest.update(canonical_json(list(source_pins)).encode("utf-8"))
    digest.update(b"}")
    return digest.hexdigest()


def _plan_acquisition_disk(
    records: Iterable[Mapping[str, Any]],
    *,
    strategies: Optional[Sequence[AcquisitionStrategy | Mapping[str, Any] | str]],
    seed: int,
    filters: Any,
    metadata: Optional[Mapping[str, Any]],
    check_cancelled: Optional[Callable[[], None]] = None,
) -> AcquisitionPlan:
    _check_cancelled(check_cancelled)
    metadata_filters = metadata.get("filters") if isinstance(metadata, Mapping) else None
    if filters is None and metadata_filters is not None:
        filters = metadata_filters
    resolved_filters = resolve_acquisition_filters(filters)
    if metadata_filters is not None:
        resolved_metadata_filters = resolve_acquisition_filters(metadata_filters)
        if resolved_metadata_filters != resolved_filters:
            raise ReviewValidationError(
                "metadata filters conflict with the acquisition request filters"
            )
    descriptor, path = tempfile.mkstemp(prefix="halo-forge-acquisition-plan-", suffix=".sqlite3")
    os.close(descriptor)
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        if check_cancelled is not None:

            def sqlite_progress() -> int:
                try:
                    check_cancelled()
                except BaseException:
                    return 1
                return 0

            connection.set_progress_handler(sqlite_progress, 10_000)
        connection.executescript("""PRAGMA journal_mode=OFF;
               PRAGMA synchronous=OFF;
               PRAGMA temp_store=FILE;
               PRAGMA cache_size=-4096;
               CREATE TABLE raw_candidates (
                 row_index INTEGER PRIMARY KEY,
                 record_id TEXT NOT NULL,
                 record_hash TEXT NOT NULL,
                 record_json TEXT NOT NULL,
                 evidence_json TEXT NOT NULL,
                 source_json TEXT NOT NULL,
                 source_kind TEXT NOT NULL,
                 source_ref TEXT,
                 source_record_id TEXT,
                 input_ordinal INTEGER NOT NULL,
                 score REAL,
                 margin REAL,
                 margin_field TEXT,
                 category TEXT NOT NULL,
                 identity_json TEXT NOT NULL,
                 descriptor_json TEXT NOT NULL
               );""")
        reason_counts: Counter[str] = Counter()
        supplied_count = 0
        for input_ordinal, raw in enumerate(records):
            _check_cancelled(check_cancelled, input_ordinal)
            value = normalize_candidate(raw, input_ordinal)
            if (reason := eligibility_reason(value)) is not None:
                reason_counts[reason] += 1
            margin = _margin_evidence(value)
            identity = value.identity_payload()
            source_descriptor = {
                "source_kind": value.source_kind,
                "source_ref": value.source_ref,
                "source": copy.deepcopy(value.source),
            }
            connection.execute(
                """INSERT INTO raw_candidates
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    supplied_count,
                    value.record_id,
                    value.record_hash,
                    canonical_json(value.record),
                    canonical_json(value.evidence),
                    canonical_json(value.source),
                    value.source_kind,
                    value.source_ref,
                    value.source_record_id,
                    value.input_ordinal,
                    value.score,
                    margin[0] if margin is not None else None,
                    margin[1] if margin is not None else None,
                    _category(value),
                    canonical_json(identity),
                    canonical_json(source_descriptor),
                ),
            )
            supplied_count += 1
            if supplied_count % 1000 == 0:
                connection.commit()
        connection.commit()
        if reason_counts:
            raise ReviewEligibilityError(
                "acquisition contains protected or non-mineable records: "
                + ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items()))
            )
        connection.executescript("""CREATE TABLE candidates AS
                 SELECT row_index,record_id,record_hash,record_json,evidence_json,
                        source_json,source_kind,source_ref,source_record_id,input_ordinal,
                        score,margin,margin_field,category,identity_json,descriptor_json
                 FROM (
                   SELECT raw_candidates.*,
                          ROW_NUMBER() OVER (
                            PARTITION BY record_id
                            ORDER BY record_hash,source_kind,COALESCE(source_ref,''),input_ordinal
                          ) AS duplicate_rank
                   FROM raw_candidates
                 ) ranked WHERE duplicate_rank=1;
               CREATE UNIQUE INDEX candidates_row_index ON candidates(row_index);
               CREATE UNIQUE INDEX candidates_record_id ON candidates(record_id);
               CREATE INDEX candidates_descriptor ON candidates(descriptor_json);
               CREATE INDEX candidates_score ON candidates(score);
               CREATE INDEX candidates_margin ON candidates(margin);
               CREATE TABLE filtered (candidate_row_index INTEGER PRIMARY KEY);
               CREATE TABLE selected (
                 ordinal INTEGER PRIMARY KEY,
                 candidate_row_index INTEGER NOT NULL UNIQUE,
                 stratum TEXT NOT NULL
               );""")
        deduplicated_count = int(
            connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        )
        source_digest = _CanonicalArrayHasher()
        rows = connection.execute("""SELECT identity_json FROM candidates
               ORDER BY record_id,record_hash,source_kind,COALESCE(source_ref,''),input_ordinal""")
        for ordinal, row in enumerate(rows):
            _check_cancelled(check_cancelled, ordinal)
            source_digest.append_text(str(row["identity_json"]))
        source_hash = source_digest.finish()

        source_pins: List[Dict[str, Any]] = []
        current_descriptor: Optional[str] = None
        record_ids_digest: Optional[_CanonicalArrayHasher] = None
        records_digest: Optional[_CanonicalArrayHasher] = None
        source_count = 0

        def finish_source_pin() -> None:
            nonlocal current_descriptor, record_ids_digest, records_digest, source_count
            if current_descriptor is None or record_ids_digest is None or records_digest is None:
                return
            source_pins.append(
                {
                    **dict(json.loads(current_descriptor)),
                    "record_count": source_count,
                    "record_ids_hash": record_ids_digest.finish(),
                    "records_hash": records_digest.finish(),
                }
            )

        rows = connection.execute("""SELECT descriptor_json,record_id,identity_json FROM candidates
               ORDER BY descriptor_json,record_id,record_hash,source_kind,
                        COALESCE(source_ref,''),input_ordinal""")
        for ordinal, row in enumerate(rows):
            _check_cancelled(check_cancelled, ordinal)
            descriptor_json = str(row["descriptor_json"])
            if descriptor_json != current_descriptor:
                finish_source_pin()
                current_descriptor = descriptor_json
                record_ids_digest = _CanonicalArrayHasher()
                records_digest = _CanonicalArrayHasher()
                source_count = 0
            assert record_ids_digest is not None and records_digest is not None
            record_ids_digest.append(str(row["record_id"]))
            records_digest.append_text(str(row["identity_json"]))
            source_count += 1
        finish_source_pin()

        if not resolved_filters:
            connection.execute("INSERT INTO filtered SELECT row_index FROM candidates")
        else:
            pending: List[Tuple[int]] = []
            rows = connection.execute("""SELECT * FROM candidates
                   ORDER BY record_id,record_hash,source_kind,
                            COALESCE(source_ref,''),input_ordinal""")
            for ordinal, row in enumerate(rows):
                _check_cancelled(check_cancelled, ordinal)
                if _candidate_matches_filters(_candidate_from_disk_row(row), resolved_filters):
                    pending.append((int(row["row_index"]),))
                    if len(pending) >= 1000:
                        connection.executemany("INSERT INTO filtered VALUES (?)", pending)
                        pending.clear()
            if pending:
                connection.executemany("INSERT INTO filtered VALUES (?)", pending)
        connection.commit()
        filtered_count = int(connection.execute("SELECT COUNT(*) FROM filtered").fetchone()[0])
        if resolved_filters and not filtered_count:
            raise ReviewValidationError("acquisition filters selected no eligible records")
        resolved_strategies = _disk_resolve_strategies(
            connection, strategies, check_cancelled=check_cancelled
        )
        selected_count = 0
        stratum_counts: Dict[str, int] = {}
        for index, strategy in enumerate(resolved_strategies):
            count = _disk_select_strategy(
                connection,
                strategy,
                seed=int(seed),
                stratum_index=index,
                ordinal=selected_count,
                check_cancelled=check_cancelled,
            )
            stratum_counts[f"{index}:{strategy.kind}"] = count
            selected_count += count
        if not selected_count:
            raise ReviewValidationError("acquisition strategies selected no eligible records")
        request = {
            "format_version": 1,
            "strategy_engine_version": 1,
            "deduplication": {
                "key": "record_id",
                "collision_policy": "stable_first",
                "version": 1,
            },
            "presentation_order": {
                "policy": "selected_ordinal",
                "version": 1,
            },
            "filters": resolved_filters,
            "strategies": [value.to_dict() for value in resolved_strategies],
            "seed": int(seed),
        }
        resolved_metadata = _mapping(metadata, "metadata")
        if "filters" in resolved_metadata:
            resolved_metadata["filters"] = copy.deepcopy(resolved_filters)
        eligibility = {
            "supplied": supplied_count,
            "eligible": supplied_count,
            "protected": 0,
            "deduplicated": supplied_count - deduplicated_count,
            "after_filters": filtered_count,
            "filtered_out": deduplicated_count - filtered_count,
            "selected": selected_count,
            "strata": stratum_counts,
        }
        connection.commit()
        selected_sequence = _DiskCandidateSequence(path, selected_count)
        _check_cancelled(check_cancelled)
        plan_hash = _stream_plan_identity_hash(
            selected_sequence,
            request=request,
            source_hash=source_hash,
            source_pins=source_pins,
            metadata=resolved_metadata,
            check_cancelled=check_cancelled,
        )
        connection.close()
        connection = None
        return AcquisitionPlan(
            selected=selected_sequence,
            eligibility=eligibility,
            source_hash=source_hash,
            source_pins=tuple(source_pins),
            request=request,
            metadata=resolved_metadata,
            content_hash=plan_hash,
        )
    except Exception:
        if connection is not None:
            connection.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        raise


def plan_acquisition(
    records: Iterable[Mapping[str, Any]],
    *,
    strategies: Optional[Sequence[AcquisitionStrategy | Mapping[str, Any] | str]] = None,
    seed: int = 0,
    filters: Any = None,
    metadata: Optional[Mapping[str, Any]] = None,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> AcquisitionPlan:
    _check_cancelled(check_cancelled)
    # Direct API/CLI payloads remain on the historical implementation so
    # their content identities and small-input latency do not change. Durable
    # spools and other one-pass iterables use the on-disk planner below.
    if not isinstance(records, SequenceABC):
        return _plan_acquisition_disk(
            records,
            strategies=strategies,
            seed=seed,
            filters=filters,
            metadata=metadata,
            check_cancelled=check_cancelled,
        )
    metadata_filters = metadata.get("filters") if isinstance(metadata, Mapping) else None
    if filters is None and metadata_filters is not None:
        filters = metadata_filters
    resolved_filters = resolve_acquisition_filters(filters)
    if metadata_filters is not None:
        resolved_metadata_filters = resolve_acquisition_filters(metadata_filters)
        if resolved_metadata_filters != resolved_filters:
            raise ReviewValidationError(
                "metadata filters conflict with the acquisition request filters"
            )
    normalized = normalize_candidates(records)
    _check_cancelled(check_cancelled)
    _assert_eligible(normalized)
    deduplicated, duplicate_count = _deduplicate(normalized)
    filtered = apply_acquisition_filters(deduplicated, resolved_filters)
    if resolved_filters and not filtered:
        raise ReviewValidationError("acquisition filters selected no eligible records")
    resolved = resolve_acquisition_strategies(filtered, strategies)
    selected: List[CandidateInput] = []
    selected_ids: set[str] = set()
    stratum_counts: Dict[str, int] = {}
    for index, strategy in enumerate(resolved):
        _check_cancelled(check_cancelled)
        available = [value for value in filtered if value.record_id not in selected_ids]
        matches = _strategy_matches(strategy, available, seed=int(seed), stratum_index=index)
        stratum = f"{index}:{strategy.kind}"
        for value in matches:
            _check_cancelled(check_cancelled, len(selected))
            if value.record_id in selected_ids:
                continue
            selected_ids.add(value.record_id)
            selected.append(replace(value, stratum=stratum))
        stratum_counts[stratum] = len(matches)
    if not selected:
        raise ReviewValidationError("acquisition strategies selected no eligible records")
    eligibility = {
        "supplied": len(normalized),
        "eligible": len(normalized),
        "protected": 0,
        "deduplicated": duplicate_count,
        "after_filters": len(filtered),
        "filtered_out": len(deduplicated) - len(filtered),
        "selected": len(selected),
        "strata": stratum_counts,
    }
    source_hash = content_hash([value.identity_payload() for value in deduplicated])
    source_pins = _source_pins(deduplicated)
    request = {
        "format_version": 1,
        "strategy_engine_version": 1,
        "deduplication": {
            "key": "record_id",
            "collision_policy": "stable_first",
            "version": 1,
        },
        "presentation_order": {
            "policy": "selected_ordinal",
            "version": 1,
        },
        "filters": resolved_filters,
        "strategies": [value.to_dict() for value in resolved],
        "seed": int(seed),
    }
    resolved_metadata = _mapping(metadata, "metadata")
    if "filters" in resolved_metadata:
        resolved_metadata["filters"] = copy.deepcopy(resolved_filters)
    identity = {
        "identity_version": 1,
        "request": request,
        "source_hash": source_hash,
        "source_pins": list(source_pins),
        "selected": [{**value.identity_payload(), "stratum": value.stratum} for value in selected],
        "metadata": resolved_metadata,
    }
    return AcquisitionPlan(
        selected=tuple(selected),
        eligibility=eligibility,
        source_hash=source_hash,
        source_pins=source_pins,
        request=request,
        metadata=resolved_metadata,
        content_hash=content_hash(identity),
    )


def comparison_records(
    base_records: Iterable[Mapping[str, Any]],
    candidate_records: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    base = normalize_candidates(base_records)
    candidate = normalize_candidates(candidate_records)
    # A protected base result is still evidence: it changes whether a sample is
    # classified as a regression or improvement.  Reject it before joining so
    # holdout/operational evidence cannot guide reviewed training data through
    # an otherwise eligible candidate evaluation.
    _assert_eligible(base, context="base evaluation comparison")
    _assert_eligible(candidate, context="candidate evaluation comparison")
    base_by_id = {value.record_id: value for value in base}
    result: List[Dict[str, Any]] = []
    for value in candidate:
        reference = base_by_id.get(value.record_id)
        evidence = copy.deepcopy(value.evidence)
        candidate_passed = _boolean(evidence.get("passed"))
        base_passed = _boolean(reference.evidence.get("passed")) if reference else None
        if candidate_passed is False and base_passed is True:
            outcome = "regression"
        elif candidate_passed is True and base_passed is False:
            outcome = "improvement"
        elif candidate_passed is False and base_passed is False:
            outcome = "unchanged_failure"
        elif candidate_passed is True and base_passed is True:
            outcome = "unchanged_pass"
        else:
            outcome = "unmatched"
        evidence.update(
            outcome=outcome,
            candidate_passed=candidate_passed,
            base_passed=base_passed,
            candidate_score=value.score,
            base_score=reference.score if reference else None,
        )
        if value.score is not None and reference is not None and reference.score is not None:
            direction = str(evidence.get("score_direction") or "maximize")
            delta = value.score - reference.score
            evidence["score_delta"] = delta if direction == "maximize" else -delta
            evidence.setdefault("margin", abs(delta))
        source = {
            **value.source,
            "kind": "evaluation_comparison",
            "base_ref": reference.source_ref if reference else None,
            "candidate_ref": value.source_ref,
        }
        result.append(
            {
                "record_id": value.record_id,
                "record_hash": value.record_hash,
                "record": value.record,
                "evidence": evidence,
                "source": source,
                "source_record_id": value.source_record_id,
            }
        )
    return result


__all__ = [
    "AcquisitionPlan",
    "CandidateInput",
    "PROTECTED_PURPOSES",
    "PROTECTED_SPLITS",
    "STRATEGY_KINDS",
    "acquire",
    "comparison_records",
    "eligibility_reason",
    "normalize_candidate",
    "normalize_candidates",
    "plan_acquisition",
    "apply_acquisition_filters",
    "resolve_acquisition_filters",
    "resolve_acquisition_strategies",
]
