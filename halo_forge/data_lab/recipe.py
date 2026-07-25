"""Validated, deterministic ordered recipes for Dataset Lab."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence

import yaml

from halo_forge.data.primitives import jaccard_text, word_shingles

from .errors import RecipeError, SchemaError
from .identity import (
    INTERNAL_LINEAGE_KEY,
    derive_record_identity,
    record_hash,
    seed_record_identities,
    strip_internal_identity,
)
from .models import adapt_record, get_field, normalize_kind, set_field, validate_record
from .profiling import profile_records, record_text
from .sources import hash_file, load_local, stable_json

SUPPORTED_STEPS = {
    "map",
    "normalize",
    "document_clean",
    "document_filter",
    "validate",
    "filter",
    "dedup",
    "score",
    "sample",
    "shuffle",
    "limit",
    "mix",
    "split",
    "contamination",
    "curriculum",
    "failure_mining",
    "repair_overlay",
    "review_label_set",
    "synthesize",
}


@dataclass(frozen=True)
class RecipeStep:
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "RecipeStep":
        raw = dict(value)
        kind = str(raw.pop("kind", raw.pop("type", ""))).strip().lower().replace("-", "_")
        if kind not in SUPPORTED_STEPS:
            raise RecipeError(
                f"Unknown recipe step {kind!r}; choose: {', '.join(sorted(SUPPORTED_STEPS))}"
            )
        if "params" in raw:
            nested = raw.pop("params")
            if not isinstance(nested, Mapping):
                raise RecipeError(f"{kind}.params must be an object")
            raw = {**dict(nested), **raw}
        return cls(kind, raw)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, **copy.deepcopy(self.params)}


@dataclass(frozen=True)
class Recipe:
    steps: tuple[RecipeStep, ...]
    name: str = "dataset-build"
    schema: Optional[str] = None
    seed: int = 0

    @classmethod
    def from_value(cls, value: "Recipe | Mapping[str, Any] | Path | str") -> "Recipe":
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser()
            if not path.is_file():
                raise RecipeError(f"Recipe file does not exist: {path}")
            try:
                if path.suffix.lower() == ".json":
                    payload = json.loads(path.read_text(encoding="utf-8"))
                else:
                    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, yaml.YAMLError) as exc:
                raise RecipeError(f"Invalid recipe file {path}: {exc}") from exc
        else:
            payload = dict(value)
        if not isinstance(payload, Mapping):
            raise RecipeError("Recipe must be an object")
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise RecipeError("Recipe requires a non-empty ordered steps list")
        if any(not isinstance(step, Mapping) for step in raw_steps):
            raise RecipeError("Every recipe step must be an object")
        recipe = cls(
            steps=tuple(RecipeStep.from_value(step) for step in raw_steps),
            name=str(payload.get("name", "dataset-build")),
            schema=str(payload["schema"]) if payload.get("schema") else None,
            seed=int(payload.get("seed", 0)),
        )
        validate_recipe(recipe)
        return recipe

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "name": self.name,
            "seed": self.seed,
            "steps": [s.to_dict() for s in self.steps],
        }
        if self.schema:
            result["schema"] = self.schema
        return result

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(stable_json(self.to_dict()).encode("utf-8")).hexdigest()


def _required(step: RecipeStep, *names: str) -> None:
    missing = [name for name in names if name not in step.params]
    if missing:
        raise RecipeError(f"{step.kind} step requires: {', '.join(missing)}")


def validate_recipe(recipe: Recipe | Mapping[str, Any]) -> Recipe:
    resolved = recipe if isinstance(recipe, Recipe) else Recipe.from_value(recipe)
    split_seen = False
    for step in resolved.steps:
        p = step.params
        if step.kind == "map":
            if not (p.get("schema") or resolved.schema):
                raise RecipeError("map step requires schema (on the step or recipe)")
            if "fields" in p and not isinstance(p["fields"], Mapping):
                raise RecipeError("map.fields must be a target-to-source object")
        elif step.kind == "document_clean":
            if "boilerplate_threshold" in p and not (
                0 < float(p["boilerplate_threshold"]) <= 1
            ):
                raise RecipeError(
                    "document_clean.boilerplate_threshold must be in (0, 1]"
                )
        elif step.kind == "document_filter":
            if int(p.get("min_text_chars", 1)) < 1:
                raise RecipeError(
                    "document_filter.min_text_chars must be at least 1"
                )
        elif step.kind == "filter":
            _required(step, "field", "op")
            if p["op"] not in {
                "eq",
                "ne",
                "gt",
                "gte",
                "lt",
                "lte",
                "in",
                "not_in",
                "contains",
                "exists",
                "regex",
            }:
                raise RecipeError(f"Unsupported safe filter operator: {p['op']}")
        elif step.kind == "dedup":
            method = p.get("method", "exact")
            if method not in {
                "exact",
                "fuzzy",
                "semantic",
                "perceptual",
                "image_exact",
                "audio_exact",
            }:
                raise RecipeError(f"Unsupported dedup method: {method}")
            threshold = float(p.get("threshold", 0.85))
            if method in {"fuzzy", "semantic", "perceptual"} and not 0 < threshold <= 1:
                raise RecipeError("dedup threshold must be in (0, 1]")
        elif step.kind == "score":
            if p.get("method", "heuristic") not in {"heuristic", "judge", "verifier"}:
                raise RecipeError("score.method must be heuristic, judge, or verifier")
            revision_id = str(p.get("verifier_profile_revision_id") or "").strip()
            if revision_id:
                if p.get("method", "heuristic") != "verifier":
                    raise RecipeError("score.verifier_profile_revision_id requires method=verifier")
                contradictory = [
                    name for name in _RAW_VERIFIER_FIELDS if p.get(name) not in (None, "", {})
                ]
                if contradictory:
                    raise RecipeError(
                        "score.verifier_profile_revision_id conflicts with raw verifier fields: "
                        + ", ".join(contradictory)
                    )
        elif step.kind == "sample":
            if ("count" in p) == ("fraction" in p):
                raise RecipeError("sample requires exactly one of count or fraction")
            if "fraction" in p and not 0 <= float(p["fraction"]) <= 1:
                raise RecipeError("sample.fraction must be in [0, 1]")
        elif step.kind == "limit":
            _required(step, "count")
            if int(p["count"]) < 0:
                raise RecipeError("limit.count cannot be negative")
        elif step.kind == "mix":
            _required(step, "datasets")
            if not isinstance(p["datasets"], list) or not p["datasets"]:
                raise RecipeError("mix.datasets must be a non-empty list")
        elif step.kind == "split":
            method = p.get("method", "random")
            if method not in {"random", "stratified", "grouped", "time"}:
                raise RecipeError(f"Unsupported split method: {method}")
            if p.get("group_by_asset_hash") and method != "grouped":
                raise RecipeError("split.group_by_asset_hash requires method=grouped")
            if method in {"stratified", "grouped", "time"} and not (
                p.get("field") or p.get(f"{method[:-2] if method.endswith('ed') else method}_field")
            ):
                # Explicit checks below provide more useful accepted aliases.
                aliases = {"stratified": "field", "grouped": "group_field", "time": "time_field"}
                if not p.get(aliases[method]):
                    raise RecipeError(f"{method} split requires {aliases[method]}")
            split_seen = True
        elif step.kind == "contamination" and not split_seen:
            raise RecipeError("contamination must follow a split step")
        elif step.kind == "curriculum":
            _required(step, "field")
            if p.get("method", "thresholds") not in {
                "thresholds",
                "difficulty",
                "reward",
                "topic",
                "categorical",
            }:
                raise RecipeError(
                    "curriculum.method must be thresholds, difficulty, reward, topic, or categorical"
                )
        elif step.kind == "failure_mining" and not (p.get("path") or p.get("source")):
            raise RecipeError("failure_mining requires path or source")
        elif step.kind == "repair_overlay":
            _required(step, "revision_id")
            if any(key in p for key in ("path", "entries", "patches")):
                raise RecipeError(
                    "repair_overlay accepts an immutable revision_id, not inline patches or paths"
                )
        elif step.kind == "review_label_set":
            _required(step, "revision_id", "build_mode")
            if p["build_mode"] not in {
                "filter",
                "replace_by_record_id",
                "append",
                "annotate",
            }:
                raise RecipeError(
                    "review_label_set.build_mode must be filter, "
                    "replace_by_record_id, append, or annotate"
                )
        elif step.kind == "synthesize":
            revision_id = str(p.get("verifier_profile_revision_id") or "").strip()
            if revision_id:
                contradictory = [
                    name for name in _RAW_VERIFIER_FIELDS if p.get(name) not in (None, "", {})
                ]
                if contradictory:
                    raise RecipeError(
                        "synthesize.verifier_profile_revision_id conflicts with raw verifier "
                        "fields: " + ", ".join(contradictory)
                    )
    return resolved


@dataclass
class StepProvenance:
    kind: str
    input_count: int
    output_count: int
    rejected_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecipeResult:
    records: List[Dict[str, Any]]
    splits: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    quarantined: List[Dict[str, Any]] = field(default_factory=list)
    provenance: List[StepProvenance] = field(default_factory=list)
    contamination: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_records: bool = False) -> Dict[str, Any]:
        data = {
            "counts": {
                "records": len(self.records),
                "splits": {k: len(v) for k, v in self.splits.items()},
                "rejected": len(self.rejected),
                "quarantined": len(self.quarantined),
            },
            "provenance": [step.to_dict() for step in self.provenance],
            "contamination": copy.deepcopy(self.contamination),
            "statistics": copy.deepcopy(self.statistics),
        }
        if include_records:
            data.update(
                records=self.records,
                splits=self.splits,
                rejected=self.rejected,
                quarantined=self.quarantined,
            )
        return data


@dataclass
class RecipeContext:
    base_dir: Optional[Path] = None
    teacher: Optional[Callable[..., str]] = None
    synthesis_endpoint_type_default: str = "injected"
    verifier: Optional[Callable[..., Any]] = None
    verifier_profile_resolver: Optional[Callable[[str], Mapping[str, Any]]] = None
    verifier_profile_invoker: Optional[Callable[[str, Mapping[str, Any]], Any]] = None
    synthesis_verifier_default: bool = True
    judge: Optional[Callable[..., float]] = None
    semantic_similarity: Optional[Callable[[Mapping[str, Any], Mapping[str, Any]], float]] = None
    mixture_resolver: Optional[Callable[[str], Sequence[Mapping[str, Any]]]] = None
    failure_resolver: Optional[Callable[[str], Sequence[Mapping[str, Any]]]] = None
    repair_overlay_resolver: Optional[Callable[[str], Mapping[str, Any]]] = None
    source_fingerprint: Optional[str] = None
    progress: Optional[Callable[[int, int, str], None]] = None
    cancelled: Optional[Callable[[], bool]] = None
    checkpoint: Optional[Callable[[int, RecipeResult], None]] = None
    resume_result: Optional[RecipeResult] = None
    resume_after_step: int = 0


_RAW_VERIFIER_FIELDS = (
    "verifier",
    "verifier_name",
    "verifier_config",
    "verifier_kwargs",
    "verifier_configuration",
    "pass_threshold",
    "reward_threshold",
)


def _record_id(row: Mapping[str, Any]) -> Optional[str]:
    marker = row.get(INTERNAL_LINEAGE_KEY)
    if isinstance(marker, Mapping) and marker.get("record_id"):
        return str(marker["record_id"])
    for field_name in ("record_id", "id"):
        if row.get(field_name) is not None and str(row[field_name]).strip():
            return str(row[field_name])
    return None


def _observation_dict(value: Any) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        raw = value.to_dict()
    elif isinstance(value, Mapping):
        raw = dict(value)
    elif hasattr(value, "reward"):
        raw = {
            "reward": getattr(value, "reward", None),
            "passed": getattr(value, "passed", getattr(value, "success", None)),
            "parsed_value": getattr(value, "parsed_value", None),
            "raw_output": getattr(value, "raw_output", None),
            "details": getattr(value, "details", {}),
            "component_trace": getattr(value, "component_trace", ()),
            "latency_ms": getattr(value, "latency_ms", None),
            "error": getattr(value, "error", None),
            "runtime_identity": getattr(value, "runtime_identity", {}),
        }
    else:
        raw = {"reward": value}
    return {
        "reward": raw.get("reward"),
        "passed": raw.get("passed"),
        "parsed_value": copy.deepcopy(raw.get("parsed_value")),
        "raw_output": copy.deepcopy(raw.get("raw_output")),
        "details": copy.deepcopy(raw.get("details") or {}),
        "component_trace": copy.deepcopy(raw.get("component_trace") or []),
        "latency_ms": raw.get("latency_ms"),
        "error": raw.get("error"),
        "runtime_identity": copy.deepcopy(raw.get("runtime_identity") or {}),
    }


def _check_cancelled(context: RecipeContext) -> None:
    if context.cancelled and context.cancelled():
        raise RecipeError("Recipe execution cancelled")


def _reject(result: RecipeResult, row: Mapping[str, Any], reason: str, mode: str) -> None:
    tagged = copy.deepcopy(dict(row))
    tagged["_rejection_reason"] = reason
    if mode == "quarantine":
        result.quarantined.append(tagged)
    else:
        result.rejected.append(tagged)


def _normalize_value(value: Any, *, trim: bool, lowercase: bool, collapse_whitespace: bool) -> Any:
    if isinstance(value, str):
        if trim:
            value = value.strip()
        if collapse_whitespace:
            value = re.sub(r"\s+", " ", value)
        if lowercase:
            value = value.lower()
    return value


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "exists":
        return (actual is not None) == bool(expected if expected is not None else True)
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "in":
        return actual in expected
    if op == "not_in":
        return actual not in expected
    if op == "contains":
        return expected in actual if actual is not None else False
    if op == "regex":
        return bool(re.search(str(expected), str(actual or "")))
    try:
        return {
            "gt": actual > expected,
            "gte": actual >= expected,
            "lt": actual < expected,
            "lte": actual <= expected,
        }[op]
    except (TypeError, KeyError):
        return False


def _shingles(text: str, n: int = 3) -> set[str]:
    return set(word_shingles(text.strip().lower(), n=n))


def _jaccard(left: str, right: str) -> float:
    return jaccard_text(left, right, n=3)


def _text(row: Mapping[str, Any], field_name: Optional[str]) -> str:
    value = get_field(row, field_name) if field_name else None
    return str(value) if value is not None else record_text(strip_internal_identity(row))


def _perceptual_hash(path: Path) -> int:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RecipeError("Perceptual image dedup requires Pillow") from exc
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8))
        pixels = list(gray.getdata())
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return bits


def _asset_path(
    row: Mapping[str, Any], field_name: str, base_dir: Optional[Path]
) -> Optional[Path]:
    value = get_field(row, field_name)
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, str) or value.startswith(("http://", "https://", "data:")):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir:
        path = base_dir / path
    return path.resolve()


def _partition(
    items: Sequence[Dict[str, Any]], ratios: Mapping[str, float]
) -> Dict[str, List[Dict[str, Any]]]:
    names = list(ratios)
    total_ratio = sum(float(ratios[name]) for name in names)
    if total_ratio <= 0:
        raise RecipeError("split ratios must sum to a positive value")
    normalized = [float(ratios[name]) / total_ratio for name in names]
    raw_counts = [len(items) * ratio for ratio in normalized]
    counts = [math.floor(count) for count in raw_counts]
    for index in sorted(range(len(names)), key=lambda i: raw_counts[i] - counts[i], reverse=True)[
        : len(items) - sum(counts)
    ]:
        counts[index] += 1
    output: Dict[str, List[Dict[str, Any]]] = {}
    cursor = 0
    for name, count in zip(names, counts):
        output[name] = list(items[cursor : cursor + count])
        cursor += count
    return output


def _split(
    records: Sequence[Dict[str, Any]],
    params: Mapping[str, Any],
    seed: int,
    *,
    base_dir: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    method = params.get("method", "random")
    ratios = params.get("ratios", {"train": 0.8, "validation": 0.1, "test": 0.1})
    if not isinstance(ratios, Mapping) or not ratios:
        raise RecipeError("split.ratios must be a non-empty object")
    rng = random.Random(int(params.get("seed", seed)))
    rows = list(records)
    if method == "random":
        rng.shuffle(rows)
        return _partition(rows, ratios)
    field_name = str(params.get("field") or params.get("group_field") or params.get("time_field"))
    if method == "time":
        rows.sort(
            key=lambda row: (
                str(get_field(row, field_name, "")),
                stable_json(strip_internal_identity(row)),
            )
        )
        return _partition(rows, ratios)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if bool(params.get("group_by_asset_hash", False)):
            asset_path = _asset_path(row, field_name, base_dir)
            if asset_path is None or not asset_path.is_file():
                raise RecipeError(
                    f"asset-hash grouping could not resolve {field_name!r} to a local file"
                )
            group_key = f"sha256:{hash_file(asset_path)}"
        else:
            group_key = stable_json(get_field(row, field_name))
        grouped.setdefault(group_key, []).append(row)
    if method == "stratified":
        output = {name: [] for name in ratios}
        for group_name in sorted(grouped):
            group = grouped[group_name]
            rng.shuffle(group)
            portions = _partition(group, ratios)
            for name, values in portions.items():
                output[name].extend(values)
        for values in output.values():
            rng.shuffle(values)
        return output
    # Grouped: assign whole groups, preventing media/entity leakage.
    output = {name: [] for name in ratios}
    targets = {
        name: len(rows) * float(weight) / sum(float(v) for v in ratios.values())
        for name, weight in ratios.items()
    }
    groups = list(grouped.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    for group in groups:
        destination = min(
            output,
            # Prefer the largest requested split when every split is empty.  The
            # previous ``max(target, 1)`` denominator made a one-group media
            # dataset fall through to the alphabetically first split (usually
            # ``test``), leaving no records with which to run the guided proof
            # training.  Keeping the true target in the fill ratio and using
            # the target as the tie-break preserves deterministic grouping
            # while making small datasets useful.
            key=lambda name: (
                len(output[name]) / targets[name]
                if targets[name] > 0
                else float("inf"),
                -targets[name],
                len(output[name]),
                name,
            ),
        )
        output[destination].extend(group)
    return output


class RecipeRunner:
    """Execute ordered transformations over in-memory JSON records."""

    def __init__(self, context: Optional[RecipeContext] = None):
        self.context = context or RecipeContext()

    def run(
        self, records: Sequence[Mapping[str, Any]], recipe: Recipe | Mapping[str, Any] | Path | str
    ) -> RecipeResult:
        resolved = Recipe.from_value(recipe)
        result = (
            copy.deepcopy(self.context.resume_result)
            if self.context.resume_result is not None
            else RecipeResult(records=[copy.deepcopy(dict(row)) for row in records])
        )
        resume_after = max(0, int(self.context.resume_after_step))
        if resume_after > len(resolved.steps):
            raise RecipeError("Recipe checkpoint is beyond the end of the recipe")
        for step_index, step in enumerate(resolved.steps):
            if step_index < resume_after:
                continue
            _check_cancelled(self.context)
            if self.context.progress:
                self.context.progress(step_index, len(resolved.steps), step.kind)
            before = len(result.records)
            rejected_before = len(result.rejected) + len(result.quarantined)
            details = self._apply(result, step, resolved)
            provenance = StepProvenance(
                kind=step.kind,
                input_count=before,
                output_count=len(result.records),
                rejected_count=len(result.rejected) + len(result.quarantined) - rejected_before,
                details=details,
            )
            result.provenance.append(provenance)
            if self.context.checkpoint:
                self.context.checkpoint(step_index, result)
        if not result.splits:
            result.splits = {"train": list(result.records)}
        result.statistics = {
            "counts": {
                "records": len(result.records),
                "splits": {name: len(rows) for name, rows in result.splits.items()},
                "rejected": len(result.rejected),
                "quarantined": len(result.quarantined),
            },
            "profile": profile_records(
                [strip_internal_identity(row) for row in result.records],
                base_dir=self.context.base_dir,
            ),
        }
        if self.context.progress:
            self.context.progress(len(resolved.steps), len(resolved.steps), "complete")
        return result

    def _apply(self, result: RecipeResult, step: RecipeStep, recipe: Recipe) -> Dict[str, Any]:
        handler = getattr(self, f"_step_{step.kind}")
        return handler(result, step.params, recipe)

    def _resolve_verifier_profile(
        self, p: Mapping[str, Any], *, step_kind: str
    ) -> Optional[Dict[str, Any]]:
        revision_id = str(p.get("verifier_profile_revision_id") or "").strip()
        if not revision_id:
            return None
        if not self.context.verifier_profile_resolver or not self.context.verifier_profile_invoker:
            raise RecipeError(
                f"{step_kind}.verifier_profile_revision_id requires Dataset Lab's "
                "VerifierLabService integration"
            )
        contradictory = [name for name in _RAW_VERIFIER_FIELDS if p.get(name) not in (None, "", {})]
        if contradictory:
            raise RecipeError(
                f"{step_kind}.verifier_profile_revision_id conflicts with raw verifier fields: "
                + ", ".join(contradictory)
            )
        try:
            resolved = dict(self.context.verifier_profile_resolver(revision_id))
        except Exception as exc:
            raise RecipeError(
                f"Could not resolve verifier profile revision {revision_id}: {exc}"
            ) from exc
        if str(resolved.get("verifier_profile_revision_id") or "") != revision_id:
            raise RecipeError("Verifier profile resolver returned a different immutable revision")
        contract = dict(resolved.get("reward_contract") or {})
        try:
            minimum = float(contract["minimum"])
            maximum = float(contract["maximum"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RecipeError("Resolved verifier revision has an invalid reward contract") from exc
        if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum <= minimum:
            raise RecipeError("Resolved verifier revision has an invalid reward range")
        direction = str(contract.get("direction") or "maximize")
        if direction not in {"maximize", "minimize"}:
            raise RecipeError("Resolved verifier revision has an invalid reward direction")
        pinned_threshold = contract.get("threshold")
        if pinned_threshold is not None:
            try:
                pinned_threshold = float(pinned_threshold)
            except (TypeError, ValueError) as exc:
                raise RecipeError("Resolved verifier revision has an invalid threshold") from exc
            if (
                not math.isfinite(pinned_threshold)
                or pinned_threshold < minimum
                or pinned_threshold > maximum
            ):
                raise RecipeError("Resolved verifier revision has an out-of-range threshold")
        if "threshold" in p and p.get("threshold") not in (None, ""):
            try:
                supplied_threshold = float(p["threshold"])
            except (TypeError, ValueError) as exc:
                raise RecipeError(f"{step_kind}.threshold must be numeric") from exc
            if pinned_threshold is None or not math.isclose(
                supplied_threshold, pinned_threshold, rel_tol=0.0, abs_tol=1e-12
            ):
                raise RecipeError(
                    f"{step_kind}.verifier_profile_revision_id conflicts with threshold; "
                    f"the immutable reward contract threshold is {pinned_threshold}"
                )
        resolved["reward_contract"] = {
            **contract,
            "minimum": minimum,
            "maximum": maximum,
            "direction": direction,
            "threshold": pinned_threshold,
        }
        return resolved

    @staticmethod
    def _validated_verifier_observation(
        observation: Any,
        *,
        contract: Mapping[str, Any],
        record_id: Optional[str],
    ) -> tuple[Optional[float], Dict[str, Any]]:
        trace = _observation_dict(observation)
        trace["record_id"] = record_id
        error = trace.get("error")
        reward = trace.get("reward")
        if reward is None:
            trace["accepted"] = False
            trace["rejection_reason"] = str(error or "verifier returned no reward")
            return None, trace
        try:
            score = float(reward)
        except (TypeError, ValueError):
            trace["accepted"] = False
            trace["rejection_reason"] = "verifier reward is not numeric"
            return None, trace
        minimum = float(contract["minimum"])
        maximum = float(contract["maximum"])
        if not math.isfinite(score):
            trace["accepted"] = False
            trace["rejection_reason"] = "verifier reward is not finite"
            return None, trace
        if score < minimum or score > maximum:
            trace["accepted"] = False
            trace["rejection_reason"] = f"verifier reward {score} is outside [{minimum}, {maximum}]"
            return None, trace
        trace["reward"] = score
        return score, trace

    @staticmethod
    def _contract_passed(score: float, contract: Mapping[str, Any]) -> Optional[bool]:
        threshold = contract.get("threshold")
        if threshold is None:
            return None
        if str(contract.get("direction") or "maximize") == "minimize":
            return score <= float(threshold)
        return score >= float(threshold)

    def _step_map(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        schema = str(p.get("schema") or recipe.schema)
        output = []
        for row in result.records:
            try:
                output.append(
                    adapt_record(
                        row,
                        schema,
                        mapping=p.get("fields"),
                        preserve_unmapped_metadata=bool(p.get("preserve_unmapped_metadata", True)),
                    )
                )
            except SchemaError as exc:
                mode = str(p.get("on_error", "reject"))
                if mode == "error":
                    raise RecipeError(str(exc)) from exc
                _reject(result, row, f"schema:{exc}", mode)
        result.records = output
        return {"schema": normalize_kind(schema).value, "mapping": dict(p.get("fields", {}))}

    def _step_normalize(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        fields = list(p.get("fields") or [])
        for row in result.records:
            selected = fields or [key for key, value in row.items() if isinstance(value, str)]
            for field_name in selected:
                value = get_field(row, field_name)
                set_field(
                    row,
                    field_name,
                    _normalize_value(
                        value,
                        trim=bool(p.get("trim", True)),
                        lowercase=bool(p.get("lowercase", False)),
                        collapse_whitespace=bool(p.get("collapse_whitespace", True)),
                    ),
                )
        return {"fields": fields or "all_top_level_strings"}

    def _step_document_clean(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        """Conservatively remove repeated edge boilerplate from corpus text.

        Extraction already excludes hidden HTML and binary content. This step
        operates only on visible canonical text and removes a line only when it
        is repeated across several documents and consistently appears near a
        document edge. It therefore avoids treating ordinary repeated prose as
        navigation or a footer.
        """

        preserve_headings = bool(p.get("preserve_headings", True))
        preserve_code_blocks = bool(p.get("preserve_code_blocks", True))
        strip_boilerplate = bool(p.get("strip_boilerplate", True))
        threshold = float(p.get("boilerplate_threshold", 0.6))
        document_lines: list[list[str]] = []
        occurrence_counts: Counter[str] = Counter()
        edge_counts: Counter[str] = Counter()
        for row in result.records:
            raw = str(row.get("text") or "")
            lines = [
                line.rstrip().replace("\u200b", "").replace("\ufeff", "")
                for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            ]
            document_lines.append(lines)
            seen: set[str] = set()
            edge_seen: set[str] = set()
            nonempty = [index for index, line in enumerate(lines) if line.strip()]
            edge_indexes = set(nonempty[:3] + nonempty[-3:])
            for index, line in enumerate(lines):
                normalized = re.sub(r"\s+", " ", line).strip().lower()
                if not normalized or len(normalized) > 160:
                    continue
                seen.add(normalized)
                if index in edge_indexes:
                    edge_seen.add(normalized)
            occurrence_counts.update(seen)
            edge_counts.update(edge_seen)

        minimum_documents = max(
            3,
            math.ceil(max(1, len(result.records)) * threshold),
        )
        boilerplate = {
            line
            for line, count in occurrence_counts.items()
            if strip_boilerplate
            and count >= minimum_documents
            and edge_counts[line] / max(1, count) >= 0.8
            and not line.startswith(("#", "```", "~~~"))
        }
        removed_lines = 0
        changed_documents = 0
        for row, lines in zip(result.records, document_lines):
            cleaned: list[str] = []
            in_fence = False
            changed = False
            for line in lines:
                stripped = line.strip()
                fence = stripped.startswith(("```", "~~~"))
                if fence:
                    in_fence = not in_fence
                    if not preserve_code_blocks:
                        changed = True
                        continue
                normalized = re.sub(r"\s+", " ", stripped).lower()
                if normalized in boilerplate and not in_fence:
                    removed_lines += 1
                    changed = True
                    continue
                if not preserve_headings and re.match(r"^\s{0,3}#{1,6}\s+", line):
                    line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
                    changed = True
                cleaned.append(line)
            text = "\n".join(cleaned)
            text = re.sub(r"\n[ \t]+\n", "\n\n", text)
            text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
            if text != str(row.get("text") or ""):
                changed = True
            row["text"] = text
            if changed:
                changed_documents += 1
        return {
            "strip_boilerplate": strip_boilerplate,
            "boilerplate_threshold": threshold,
            "boilerplate_lines_identified": len(boilerplate),
            "lines_removed": removed_lines,
            "changed_documents": changed_documents,
            "preserve_headings": preserve_headings,
            "preserve_code_blocks": preserve_code_blocks,
        }

    def _step_document_filter(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        output: list[Dict[str, Any]] = []
        mode = (
            "quarantine"
            if bool(p.get("quarantine_extraction_failures", True))
            else str(p.get("on_error", "reject"))
        )
        minimum = int(p.get("min_text_chars", 1))
        require_visible = bool(p.get("require_visible_text", True))
        filtered = Counter()
        for row in result.records:
            metadata = row.get("metadata")
            extraction_error = (
                row.get("extraction_error")
                or (
                    metadata.get("extraction_error")
                    if isinstance(metadata, Mapping)
                    else None
                )
            )
            text = str(row.get("text") or "").strip()
            visible = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
            reason: Optional[str] = None
            if extraction_error:
                reason = "extraction_failure"
            elif len(text) < minimum:
                reason = "document_too_short"
            elif require_visible and not visible:
                reason = "no_visible_text"
            if reason is None:
                output.append(row)
                continue
            filtered[reason] += 1
            _reject(result, row, f"document_filter:{reason}", mode)
        result.records = output
        return {
            "on_error": mode,
            "min_text_chars": minimum,
            "require_visible_text": require_visible,
            "filtered": dict(filtered),
        }

    def _step_validate(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        schema = p.get("schema") or recipe.schema
        if not schema:
            raise RecipeError("validate step requires schema on the step or recipe")
        output = []
        mode = str(p.get("on_error", "reject"))
        for row in result.records:
            try:
                validate_record(row, str(schema))
                output.append(row)
            except SchemaError as exc:
                if mode == "error":
                    raise RecipeError(str(exc)) from exc
                _reject(result, row, f"validation:{exc}", mode)
        result.records = output
        return {"schema": str(schema), "on_error": mode}

    def _step_filter(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        output = []
        for row in result.records:
            if _compare(get_field(row, str(p["field"])), str(p["op"]), p.get("value")):
                output.append(row)
            else:
                _reject(
                    result, row, f"filter:{p['field']}:{p['op']}", str(p.get("on_reject", "reject"))
                )
        result.records = output
        return {"field": p["field"], "op": p["op"]}

    def _step_dedup(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        method = str(p.get("method", "exact"))
        field_name = str(p.get("field", "")) or None
        threshold = float(p.get("threshold", 0.85))
        kept: List[Dict[str, Any]] = []
        signatures: List[Any] = []
        for row in result.records:
            duplicate = False
            if method == "exact":
                text = re.sub(r"\s+", " ", _text(row, field_name).strip())
                if not p.get("case_sensitive", False):
                    text = text.lower()
                signature = hashlib.sha256(text.encode("utf-8")).hexdigest()
                duplicate = signature in signatures
            elif method == "fuzzy":
                text = _text(row, field_name)
                signature = text
                duplicate = any(_jaccard(text, old) >= threshold for old in signatures)
            elif method == "semantic":
                if not self.context.semantic_similarity:
                    raise RecipeError("semantic dedup requires RecipeContext.semantic_similarity")
                signature = row
                clean_row = strip_internal_identity(row)
                signature = clean_row
                duplicate = any(
                    self.context.semantic_similarity(clean_row, old) >= threshold
                    for old in signatures
                )
            else:
                default_field = "audio" if method == "audio_exact" else "image"
                path = _asset_path(row, field_name or default_field, self.context.base_dir)
                if path is None or not path.is_file():
                    signature = f"missing:{stable_json(strip_internal_identity(row))}"
                elif method == "perceptual":
                    signature = _perceptual_hash(path)
                    maximum_distance = round((1 - threshold) * 64)
                    duplicate = any(
                        (signature ^ old).bit_count() <= maximum_distance for old in signatures
                    )
                else:
                    signature = hash_file(path)
                    duplicate = signature in signatures
            if duplicate:
                _reject(result, row, f"dedup:{method}", str(p.get("on_duplicate", "reject")))
            else:
                signatures.append(signature)
                kept.append(row)
        result.records = kept
        return {
            "method": method,
            "threshold": threshold if method in {"fuzzy", "semantic", "perceptual"} else None,
        }

    def _step_score(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        from halo_forge.data.quality import heuristic_score

        method = str(p.get("method", "heuristic"))
        resolved_profile = self._resolve_verifier_profile(p, step_kind="score")
        if resolved_profile and method != "verifier":
            raise RecipeError("score.verifier_profile_revision_id requires method=verifier")
        contract = (
            dict(resolved_profile["reward_contract"])
            if resolved_profile
            else {
                "minimum": 0.0,
                "maximum": 1.0,
                "direction": "maximize",
                "threshold": float(p.get("threshold", 0.5)),
            }
        )
        threshold = (
            contract.get("threshold") if resolved_profile else float(p.get("threshold", 0.5))
        )
        score_field = str(p.get("score_field", "_quality_score"))
        output = []
        observation_traces: List[Dict[str, Any]] = []
        for row in result.records:
            if method == "heuristic":
                score = heuristic_score(strip_internal_identity(row)).score
            elif method == "judge":
                if not self.context.judge:
                    raise RecipeError("judge scoring requires RecipeContext.judge")
                try:
                    score = float(
                        self.context.judge(strip_internal_identity(row), p)  # type: ignore[misc]
                    )
                except TypeError:
                    score = float(self.context.judge(strip_internal_identity(row)))
            else:
                clean_row = strip_internal_identity(row)
                if resolved_profile:
                    assert self.context.verifier_profile_invoker is not None
                    try:
                        verdict = self.context.verifier_profile_invoker(
                            str(resolved_profile["verifier_profile_revision_id"]), clean_row
                        )
                    except Exception as exc:
                        verdict = {
                            "reward": None,
                            "passed": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                else:
                    if not self.context.verifier:
                        raise RecipeError("verifier scoring requires RecipeContext.verifier")
                    try:
                        verdict = self.context.verifier(clean_row, p)  # type: ignore[misc]
                    except TypeError:
                        verdict = self.context.verifier(clean_row)
                score, trace = self._validated_verifier_observation(
                    verdict,
                    contract=contract,
                    record_id=_record_id(row),
                )
                if resolved_profile:
                    trace["verifier_profile_revision_id"] = resolved_profile[
                        "verifier_profile_revision_id"
                    ]
                    trace["revision_hash"] = resolved_profile.get("revision_hash")
                else:
                    trace.update(
                        legacy_unqualified=True,
                        warning=(
                            "Raw verifier configuration has no immutable reliability "
                            "qualification"
                        ),
                    )
                if score is None:
                    observation_traces.append(trace)
                    _reject(
                        result,
                        row,
                        "score:verifier:invalid_observation",
                        str(p.get("on_reject", "reject")),
                    )
                    continue
                passed = (
                    trace.get("passed")
                    if resolved_profile and trace.get("passed") is not None
                    else self._contract_passed(score, contract)
                )
                trace["passed"] = passed
                trace["accepted"] = bool(not p.get("reject_below", True) or passed is not False)
                observation_traces.append(trace)
            if not math.isfinite(float(score)):
                raise RecipeError(f"{method} scoring returned a non-finite score")
            if method != "verifier":
                score = max(0.0, min(1.0, float(score)))
            set_field(row, score_field, score)
            if method == "verifier":
                accepted = bool(
                    not p.get("reject_below", True)
                    or observation_traces[-1].get("passed") is not False
                )
            else:
                accepted = not bool(p.get("reject_below", True)) or score >= float(threshold)
            if accepted:
                output.append(row)
            else:
                _reject(
                    result,
                    row,
                    (
                        f"score:{method}:contract_failed"
                        if resolved_profile
                        else f"score:{method}:below:{threshold}"
                    ),
                    str(p.get("on_reject", "reject")),
                )
        result.records = output
        details: Dict[str, Any] = {
            "method": method,
            "threshold": threshold,
            "score_field": score_field,
        }
        if method == "verifier":
            details.update(
                verifier_binding=(
                    copy.deepcopy(resolved_profile)
                    if resolved_profile
                    else {
                        "implementation_ref": p.get("verifier") or p.get("verifier_name"),
                        "legacy_unqualified": True,
                        "warning": (
                            "Raw verifier configuration has no immutable reliability "
                            "qualification"
                        ),
                    }
                ),
                legacy_unqualified=not bool(resolved_profile),
                observations=observation_traces,
            )
        return details

    def _step_sample(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        count = (
            int(p["count"]) if "count" in p else round(len(result.records) * float(p["fraction"]))
        )
        count = min(max(count, 0), len(result.records))
        selected = sorted(
            random.Random(int(p.get("seed", recipe.seed))).sample(range(len(result.records)), count)
        )
        result.records = [result.records[index] for index in selected]
        return {"count": count, "seed": int(p.get("seed", recipe.seed))}

    def _step_shuffle(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        seed = int(p.get("seed", recipe.seed))
        random.Random(seed).shuffle(result.records)
        return {"seed": seed}

    def _step_limit(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        count = int(p["count"])
        result.records = result.records[:count]
        return {"count": count}

    def _step_mix(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        if not self.context.mixture_resolver:
            raise RecipeError("mix step requires RecipeContext.mixture_resolver")
        sources: List[tuple[str, float, List[Dict[str, Any]]]] = []
        for item in p["datasets"]:
            if not isinstance(item, Mapping) or not item.get("source"):
                raise RecipeError("Each mix dataset requires source")
            name = str(item["source"])
            rows = (
                list(result.records)
                if name == "current"
                else [copy.deepcopy(dict(r)) for r in self.context.mixture_resolver(name)]
            )
            weight = float(item.get("weight", 1.0))
            if weight < 0:
                raise RecipeError("Mix weights cannot be negative")
            sources.append((name, weight, rows))
        total_weight = sum(source[1] for source in sources)
        if total_weight <= 0:
            raise RecipeError("At least one mix weight must be positive")
        size = int(p.get("size", sum(len(source[2]) for source in sources)))
        rng = random.Random(int(p.get("seed", recipe.seed)))
        mixed: List[Dict[str, Any]] = []
        for index, (name, weight, rows) in enumerate(sources):
            target = (
                round(size * weight / total_weight)
                if index < len(sources) - 1
                else size - len(mixed)
            )
            if not rows and target:
                raise RecipeError(f"Mix source {name!r} is empty")
            order = list(range(len(rows)))
            rng.shuffle(order)
            for offset in range(target):
                mixed.append(copy.deepcopy(rows[order[offset % len(rows)]]))
        rng.shuffle(mixed)
        result.records = mixed
        return {
            "sources": [
                {"source": name, "weight": weight, "available": len(rows)}
                for name, weight, rows in sources
            ],
            "size": len(mixed),
        }

    def _step_split(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        result.splits = _split(
            result.records,
            p,
            recipe.seed,
            base_dir=self.context.base_dir,
        )
        return {
            "method": p.get("method", "random"),
            "group_by_asset_hash": bool(p.get("group_by_asset_hash", False)),
            "counts": {name: len(rows) for name, rows in result.splits.items()},
        }

    def _step_contamination(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        names = list(p.get("splits") or result.splits)
        method = str(p.get("method", "exact"))
        threshold = float(p.get("threshold", 0.85))
        field_name = str(p.get("field", "")) or None
        pairs: Dict[str, Any] = {}
        contaminated_train: set[int] = set()
        exact_indexes: Dict[str, Dict[str, List[int]]] = {}
        exact_values: Dict[str, List[str]] = {}
        if method == "exact":
            # Build each split index once.  The former nested comparison called
            # ``_text`` for every cross-split row pair, making the default
            # guided 80/10/10 contamination report quadratic (billions of
            # comparisons at ordinary research-dataset sizes).
            for split_name in names:
                values = [
                    _text(row, field_name).strip().lower()
                    for row in result.splits.get(split_name, [])
                ]
                index: Dict[str, List[int]] = {}
                for row_index, value in enumerate(values):
                    index.setdefault(value, []).append(row_index)
                exact_values[split_name] = values
                exact_indexes[split_name] = index
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                matches = []
                match_count = 0
                if method == "exact":
                    right_index = exact_indexes.get(right_name, {})
                    left_values = exact_values.get(left_name, [])
                    shared_values = set(exact_indexes.get(left_name, {})).intersection(
                        right_index
                    )
                    match_count = sum(
                        len(exact_indexes[left_name][value]) * len(right_index[value])
                        for value in shared_values
                    )
                    for i, value in enumerate(left_values):
                        right_matches = right_index.get(value, ())
                        if right_matches and left_name == "train":
                            contaminated_train.add(i)
                        remaining = 100 - len(matches)
                        if remaining > 0:
                            matches.extend(
                                {"left_index": i, "right_index": j}
                                for j in right_matches[:remaining]
                            )
                    if right_name == "train":
                        for value in shared_values:
                            contaminated_train.update(right_index[value])
                else:
                    for i, left in enumerate(result.splits.get(left_name, [])):
                        left_text = _text(left, field_name)
                        for j, right in enumerate(result.splits.get(right_name, [])):
                            right_text = _text(right, field_name)
                            if _jaccard(left_text, right_text) >= threshold:
                                match_count += 1
                                if len(matches) < 100:
                                    matches.append({"left_index": i, "right_index": j})
                                if left_name == "train":
                                    contaminated_train.add(i)
                                if right_name == "train":
                                    contaminated_train.add(j)
                pairs[f"{left_name}:{right_name}"] = {
                    "count": match_count,
                    "matches": matches,
                }
        removed = 0
        if p.get("action", "report") == "remove" and "train" in result.splits:
            removed = len(contaminated_train)
            result.splits["train"] = [
                row
                for index, row in enumerate(result.splits["train"])
                if index not in contaminated_train
            ]
            result.records = [row for rows in result.splits.values() for row in rows]
        result.contamination = {
            "method": method,
            "threshold": threshold,
            "pairs": pairs,
            "removed_from_train": removed,
        }
        return {
            "method": method,
            "matches": sum(pair["count"] for pair in pairs.values()),
            "removed": removed,
        }

    def _step_curriculum(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        method = str(p.get("method", "thresholds"))
        output_field = str(p.get("output_field", "metadata.curriculum"))
        if method in {"topic", "categorical"}:
            mapping = {str(key): str(value) for key, value in dict(p.get("mapping") or {}).items()}
            counts: Dict[str, int] = {}
            for row in result.records:
                raw = get_field(row, str(p["field"]))
                label = mapping.get(str(raw), str(p.get("default_label") or raw or "unknown"))
                set_field(row, output_field, label)
                counts[label] = counts.get(label, 0) + 1
            return {
                "method": method,
                "field": p["field"],
                "output_field": output_field,
                "counts": counts,
            }
        boundaries = [float(value) for value in p.get("boundaries", [0.33, 0.66])]
        labels = list(p.get("labels", ["easy", "medium", "hard"]))
        if len(labels) != len(boundaries) + 1:
            raise RecipeError("curriculum labels must have exactly one more item than boundaries")
        counts = {label: 0 for label in labels}
        for row in result.records:
            try:
                value = float(get_field(row, str(p["field"])))
            except (TypeError, ValueError):
                value = float("inf")
            index = next(
                (i for i, boundary in enumerate(boundaries) if value <= boundary), len(boundaries)
            )
            label = labels[index]
            set_field(row, output_field, label)
            counts[label] += 1
        return {
            "method": method,
            "field": p["field"],
            "output_field": output_field,
            "counts": counts,
        }

    def _step_failure_mining(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        if p.get("path"):
            mined, _, _ = load_local(str(p["path"]))
        else:
            if not self.context.failure_resolver:
                raise RecipeError("failure_mining source requires RecipeContext.failure_resolver")
            mined = [dict(row) for row in self.context.failure_resolver(str(p["source"]))]
        mined = seed_record_identities(
            mined,
            source_name=str(p.get("source") or p.get("path") or "failure-mining"),
        )
        field_name = str(p.get("failure_field", "success"))
        values = p.get("failure_values", [False, 0, "failed", "error"])
        failures = [row for row in mined if get_field(row, field_name) in values]
        if p.get("mode", "replace") == "append":
            result.records.extend(copy.deepcopy(failures))
        else:
            result.records = copy.deepcopy(failures)
        return {
            "examined": len(mined),
            "failures": len(failures),
            "source": p.get("source") or p.get("path"),
        }

    def _step_repair_overlay(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        """Apply one immutable, verified V17 repair overlay by record identity."""

        revision_id = str(p["revision_id"])
        if not self.context.repair_overlay_resolver:
            raise RecipeError(
                "repair_overlay requires Dataset Lab to be connected to the V17 catalog"
            )
        try:
            overlay = dict(self.context.repair_overlay_resolver(revision_id))
        except Exception as exc:
            raise RecipeError(f"Could not resolve repair overlay {revision_id}: {exc}") from exc
        resolved_revision = str(overlay.get("revision_id") or overlay.get("id") or "")
        if resolved_revision != revision_id:
            raise RecipeError("Repair overlay resolver returned a different immutable revision")
        expected_source = str(overlay.get("source_fingerprint") or "")
        if (
            expected_source
            and self.context.source_fingerprint
            and expected_source != self.context.source_fingerprint
        ):
            raise RecipeError(
                "Repair overlay source fingerprint does not match the registered source"
            )
        raw_entries = list(overlay.get("entries") or [])
        entries: Dict[tuple[str, Optional[int]], Mapping[str, Any]] = {}
        for raw in raw_entries:
            if not isinstance(raw, Mapping) or not raw.get("record_id"):
                raise RecipeError("Repair overlay contains an invalid record entry")
            record_id = str(raw["record_id"])
            source_index = (
                int(raw["source_index"])
                if raw.get("source_index") is not None
                else None
            )
            key = (record_id, source_index)
            if key in entries:
                raise RecipeError(
                    f"Repair overlay repeats source occurrence {record_id}@{source_index}"
                )
            entries[key] = raw
        output: List[Dict[str, Any]] = []
        applied: set[tuple[str, Optional[int]]] = set()
        counts = Counter()
        for row in result.records:
            identity = _record_id(row)
            marker = row.get(INTERNAL_LINEAGE_KEY)
            source_index: Optional[int] = None
            if isinstance(marker, Mapping):
                origins = list(marker.get("origins") or [])
                if origins and isinstance(origins[0], Mapping):
                    raw_index = origins[0].get("source_index")
                    source_index = int(raw_index) if raw_index is not None else None
            exact_key = (str(identity), source_index) if identity else None
            wildcard_key = (str(identity), None) if identity else None
            entry_key = exact_key if exact_key in entries else wildcard_key
            entry = entries.get(entry_key) if entry_key is not None else None
            if entry is None:
                output.append(row)
                counts["unchanged"] += 1
                continue
            before_hash = str(entry.get("before_hash") or "")
            actual_hash = record_hash(row)
            if before_hash and before_hash != actual_hash:
                raise RecipeError(
                    f"Repair overlay record {identity} changed after review; rebase is required"
                )
            operation = str(entry.get("operation") or "")
            if operation == "replace":
                replacement = entry.get("record")
                if not isinstance(replacement, Mapping):
                    raise RecipeError(f"Repair overlay replacement {identity} has no record")
                repaired = copy.deepcopy(dict(replacement))
                marker = row.get(INTERNAL_LINEAGE_KEY)
                if isinstance(marker, Mapping):
                    repaired[INTERNAL_LINEAGE_KEY] = copy.deepcopy(dict(marker))
                derive_record_identity(
                    repaired,
                    "repair_overlay",
                    revision_id=revision_id,
                    before_hash=actual_hash,
                    after_hash=record_hash(repaired),
                )
                output.append(repaired)
                counts["replaced"] += 1
            elif operation == "quarantine":
                _reject(result, row, f"repair_overlay:{revision_id}", "quarantine")
                counts["quarantined"] += 1
            elif operation == "exclude":
                _reject(result, row, f"repair_overlay:{revision_id}", "reject")
                counts["excluded"] += 1
            else:
                raise RecipeError(
                    f"Repair overlay record {identity} has unsupported operation {operation!r}"
                )
            applied.add(entry_key)
        unresolved = sorted(
            set(entries) - applied,
            key=lambda value: (value[0], -1 if value[1] is None else value[1]),
        )
        if unresolved:
            raise RecipeError(
                "Repair overlay references records missing from this source; rebase is required"
            )
        result.records = output
        return {
            "repair_revision_id": revision_id,
            "repair_content_hash": overlay.get("content_hash"),
            "repaired_record_set_hash": overlay.get("repaired_record_set_hash"),
            "counts": dict(counts),
        }

    def _step_synthesize(
        self, result: RecipeResult, p: Mapping[str, Any], recipe: Recipe
    ) -> Dict[str, Any]:
        if not self.context.teacher:
            raise RecipeError("synthesize requires an injected RecipeContext.teacher")
        resolved_profile = self._resolve_verifier_profile(p, step_kind="synthesize")
        contract = (
            dict(resolved_profile["reward_contract"])
            if resolved_profile
            else {
                "minimum": 0.0,
                "maximum": 1.0,
                "direction": "maximize",
                "threshold": float(p.get("threshold", 0)),
            }
        )
        prompt_field = str(p.get("prompt_field", "prompt"))
        output_field = str(p.get("output_field", "response"))
        n_per = int(p.get("n_per_record", 1))
        threshold = contract.get("threshold")
        generated: List[Dict[str, Any]] = []
        accepted = rejected = 0
        observation_traces: List[Dict[str, Any]] = []
        should_verify = bool(
            resolved_profile
            or self.context.verifier
            and (
                self.context.synthesis_verifier_default
                or p.get("verifier")
                or p.get("verifier_name")
            )
        )
        for row in result.records:
            prompt = str(p.get("prompt") or get_field(row, prompt_field, ""))
            if p.get("prompt_template"):

                class _Missing(dict):
                    def __missing__(self, key: str) -> str:
                        return ""

                try:
                    prompt = str(p["prompt_template"]).format_map(_Missing(row))
                except (KeyError, ValueError, TypeError) as exc:
                    raise RecipeError(f"Invalid synthesis prompt_template: {exc}") from exc
            for generation_index in range(n_per):
                _check_cancelled(self.context)
                call_params = dict(p)
                if self.context.base_dir:
                    call_params.setdefault("base_dir", str(self.context.base_dir))
                try:
                    completion = self.context.teacher(  # type: ignore[misc]
                        prompt, call_params, strip_internal_identity(row)
                    )
                except TypeError:
                    try:
                        completion = self.context.teacher(prompt, call_params)  # type: ignore[misc]
                    except TypeError:
                        completion = self.context.teacher(prompt)
                candidate = copy.deepcopy(row)
                set_field(candidate, output_field, str(completion))
                derive_record_identity(
                    candidate,
                    "synthesize",
                    generation_index=generation_index,
                    teacher_model=p.get("teacher_model") or p.get("model") or "default",
                )
                score = 1.0
                trace: Optional[Dict[str, Any]] = None
                if should_verify:
                    clean_candidate = strip_internal_identity(candidate)
                    if resolved_profile:
                        assert self.context.verifier_profile_invoker is not None
                        try:
                            verdict = self.context.verifier_profile_invoker(
                                str(resolved_profile["verifier_profile_revision_id"]),
                                clean_candidate,
                            )
                        except Exception as exc:
                            verdict = {
                                "reward": None,
                                "passed": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                    else:
                        assert self.context.verifier is not None
                        try:
                            verdict = self.context.verifier(  # type: ignore[misc]
                                clean_candidate, call_params
                            )
                        except TypeError:
                            verdict = self.context.verifier(clean_candidate)
                    validated_score, trace = self._validated_verifier_observation(
                        verdict,
                        contract=contract,
                        record_id=_record_id(candidate),
                    )
                    if resolved_profile:
                        trace["verifier_profile_revision_id"] = resolved_profile[
                            "verifier_profile_revision_id"
                        ]
                        trace["revision_hash"] = resolved_profile.get("revision_hash")
                    else:
                        trace.update(
                            legacy_unqualified=True,
                            warning=(
                                "Raw verifier configuration has no immutable reliability "
                                "qualification"
                            ),
                        )
                    if validated_score is None:
                        trace["accepted"] = False
                        observation_traces.append(trace)
                        _reject(
                            result,
                            candidate,
                            "synthesis:invalid_verifier_observation",
                            str(p.get("on_reject", "reject")),
                        )
                        rejected += 1
                        continue
                    score = validated_score
                    passed = (
                        trace.get("passed")
                        if resolved_profile and trace.get("passed") is not None
                        else self._contract_passed(score, contract)
                    )
                    trace["passed"] = passed
                accepted_by_verifier = not should_verify or (
                    trace is not None and trace.get("passed") is not False
                )
                if trace is not None:
                    trace["accepted"] = accepted_by_verifier
                    observation_traces.append(trace)
                metadata = candidate.setdefault("metadata", {})
                if not isinstance(metadata, MutableMapping):
                    metadata = candidate["metadata"] = {"source_metadata": metadata}
                metadata["synthesis"] = {
                    "teacher_model": p.get("teacher_model") or p.get("model") or "default",
                    "endpoint_type": p.get(
                        "endpoint_type", self.context.synthesis_endpoint_type_default
                    ),
                    "prompt": prompt,
                    "prompt_field": prompt_field,
                    "output_field": output_field,
                    "sampling": copy.deepcopy(p.get("sampling", {})),
                    "generation_index": generation_index,
                    "verifier": (
                        {
                            "verifier_profile_revision_id": resolved_profile[
                                "verifier_profile_revision_id"
                            ],
                            "revision_hash": resolved_profile.get("revision_hash"),
                            "legacy_unqualified": False,
                        }
                        if resolved_profile
                        else p.get("verifier") or p.get("verifier_name")
                    ),
                    "verifier_score": score,
                    "accepted": accepted_by_verifier,
                }
                if accepted_by_verifier:
                    generated.append(candidate)
                    accepted += 1
                else:
                    _reject(
                        result,
                        candidate,
                        (
                            "synthesis:verifier_contract_failed"
                            if resolved_profile
                            else f"synthesis:below:{threshold}"
                        ),
                        str(p.get("on_reject", "reject")),
                    )
                    rejected += 1
        result.records = generated
        details: Dict[str, Any] = {
            "generated": accepted + rejected,
            "accepted": accepted,
            "rejected": rejected,
            "teacher_model": p.get("teacher_model") or p.get("model") or "default",
            "endpoint_type": p.get("endpoint_type", self.context.synthesis_endpoint_type_default),
            "verifier": p.get("verifier") or p.get("verifier_name"),
        }
        if should_verify or resolved_profile:
            details.update(
                verifier_binding=(
                    copy.deepcopy(resolved_profile)
                    if resolved_profile
                    else {
                        "implementation_ref": p.get("verifier") or p.get("verifier_name"),
                        "legacy_unqualified": True,
                        "warning": (
                            "Raw verifier configuration has no immutable reliability "
                            "qualification"
                        ),
                    }
                ),
                legacy_unqualified=not bool(resolved_profile),
                observations=observation_traces,
            )
        return details


__all__ = [
    "Recipe",
    "RecipeContext",
    "RecipeResult",
    "RecipeRunner",
    "RecipeStep",
    "SUPPORTED_STEPS",
    "StepProvenance",
    "validate_recipe",
]
