"""Deterministic calibration partitioning and invocation expansion."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence


CALIBRATION_SEED = 42
STOCHASTIC_SEEDS = (17, 42, 101)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CalibrationProtocol:
    family: str
    task_type: str
    deterministic: bool
    seed: int = CALIBRATION_SEED
    confirmation_requested: bool = False
    confirmation_fraction: float = 0.30
    stochastic_seeds: tuple[int, ...] = STOCHASTIC_SEEDS
    temperature: float = 0.0
    top_p: float = 1.0
    concurrency: int = 1
    ranking_orientation_cap: int = 4
    production_batch_size: Optional[int] = None
    reward_model_dtype: Optional[str] = None
    reviewed_probe_kinds: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in {"deterministic", "llm_judge", "reward_model", "chain"}:
            raise ValueError("unsupported verifier family")
        if self.task_type not in {
            "binary",
            "categorical",
            "multi_label",
            "scalar",
            "pairwise",
            "ranking",
        }:
            raise ValueError("unsupported calibration task type")
        if not 0.0 < float(self.confirmation_fraction) < 1.0:
            raise ValueError("confirmation_fraction must be between zero and one")
        if self.temperature != 0.0 or self.top_p != 1.0 or self.concurrency != 1:
            raise ValueError(
                "calibration requires temperature=0, top_p=1, and concurrency=1"
            )
        if self.deterministic and len(self.stochastic_seeds) < 1:
            raise ValueError("at least one protocol seed is required")
        if not self.deterministic and tuple(self.stochastic_seeds) != STOCHASTIC_SEEDS:
            raise ValueError("stochastic calibration seeds are pinned to 17, 42, and 101")
        if self.ranking_orientation_cap < 1 or self.ranking_orientation_cap > 4:
            raise ValueError("ranking_orientation_cap must be between one and four")
        if self.production_batch_size is not None and int(self.production_batch_size) < 1:
            raise ValueError("production_batch_size must be positive")
        if self.family == "reward_model" and self.production_batch_size is not None:
            if not str(self.reward_model_dtype or "").strip():
                raise ValueError(
                    "reward-model batch parity requires a declared production dtype"
                )
            dtype_score_tolerance(str(self.reward_model_dtype))

    @property
    def protocol_hash(self) -> str:
        return _stable_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["protocol_hash"] = self.protocol_hash
        return result


@dataclass(frozen=True)
class ProtocolInvocation:
    invocation_id: str
    record_id: str
    repetition_index: int
    seed: Optional[int]
    orientation: str
    perturbation: str = "canonical"
    fresh_process: bool = True
    generation_settings: Mapping[str, Any] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)
    production_batch_size: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationPartition:
    calibration_record_ids: tuple[str, ...]
    confirmation_record_ids: tuple[str, ...]
    group_count: int
    target_confirmation_records: int
    leakage: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_record_ids": list(self.calibration_record_ids),
            "confirmation_record_ids": list(self.confirmation_record_ids),
            "group_count": self.group_count,
            "target_confirmation_records": self.target_confirmation_records,
            "leakage": {key: list(value) for key, value in self.leakage.items()},
        }


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            keep, replace = sorted((left_root, right_root))
            self.parent[replace] = keep


def _identity_values(record: Mapping[str, Any]) -> tuple[str, ...]:
    result: list[str] = []
    for key in ("group_id", "related_group_id"):
        value = record.get(key)
        if value not in {None, ""}:
            result.append(f"group:{value}")
    if record.get("content_hash") not in {None, ""}:
        result.append(f"content:{record['content_hash']}")
    if record.get("media_hash") not in {None, ""}:
        result.append(f"media:{record['media_hash']}")
    for key in ("media_hashes", "asset_hashes"):
        values = record.get(key) or ()
        if isinstance(values, str):
            values = (values,)
        for value in values:
            result.append(f"media:{value}")
    return tuple(sorted(set(result)))


def _partition_leakage(
    calibration: Sequence[Mapping[str, Any]],
    confirmation: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    def values(rows: Sequence[Mapping[str, Any]], key: str) -> set[str]:
        result: set[str] = set()
        for row in rows:
            if key == "record_id":
                result.add(str(row["record_id"]))
            elif key == "content_hash" and row.get("content_hash"):
                result.add(str(row["content_hash"]))
            elif key == "media_hash":
                if row.get("media_hash"):
                    result.add(str(row["media_hash"]))
                media = row.get("media_hashes", row.get("asset_hashes", ())) or ()
                if isinstance(media, str):
                    media = (media,)
                result.update(str(value) for value in media)
        return result

    return {
        key: tuple(sorted(values(calibration, key).intersection(values(confirmation, key))))
        for key in ("record_id", "content_hash", "media_hash")
    }


def grouped_calibration_confirmation_partition(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = CALIBRATION_SEED,
    confirmation_fraction: float = 0.30,
) -> CalibrationPartition:
    """Create a stable grouped 70/30 partition without shared content/media."""

    if not records:
        raise ValueError("calibration source must contain records")
    if not 0.0 < float(confirmation_fraction) < 1.0:
        raise ValueError("confirmation_fraction must be between zero and one")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        record_id = str(record.get("record_id", "")).strip()
        if not record_id:
            raise ValueError("every calibration record requires record_id")
        if record_id in by_id:
            raise ValueError(f"duplicate calibration record_id {record_id!r}")
        by_id[record_id] = record

    disjoint = _DisjointSet(by_id)
    identity_owner: dict[str, str] = {}
    for record_id in sorted(by_id):
        for identity in _identity_values(by_id[record_id]):
            previous = identity_owner.setdefault(identity, record_id)
            disjoint.union(previous, record_id)

    groups: dict[str, list[str]] = {}
    for record_id in sorted(by_id):
        groups.setdefault(disjoint.find(record_id), []).append(record_id)
    ordered_groups = sorted(
        groups.values(),
        key=lambda group: (_stable_hash({"seed": int(seed), "records": group}), group[0]),
    )
    target = max(1, int(round(len(records) * float(confirmation_fraction))))
    confirmation_ids: list[str] = []
    calibration_ids: list[str] = []
    for group in ordered_groups:
        if len(confirmation_ids) < target:
            confirmation_ids.extend(group)
        else:
            calibration_ids.extend(group)
    if not calibration_ids:
        # A single linked group cannot be split safely.  Report an empty
        # calibration side so the caller can refuse qualification explicitly.
        calibration_ids = []
    calibration_rows = [by_id[value] for value in calibration_ids]
    confirmation_rows = [by_id[value] for value in confirmation_ids]
    leakage = _partition_leakage(calibration_rows, confirmation_rows)
    if any(leakage.values()):
        raise AssertionError("grouped partition leaked record, content, or media identity")
    return CalibrationPartition(
        calibration_record_ids=tuple(sorted(calibration_ids)),
        confirmation_record_ids=tuple(sorted(confirmation_ids)),
        group_count=len(groups),
        target_confirmation_records=target,
        leakage=leakage,
    )


def dtype_score_tolerance(dtype: str) -> float:
    normalized = str(dtype).strip().lower()
    if normalized in {"fp32", "float32", "fp64", "float64", "double"}:
        return 1e-6
    if normalized in {"fp16", "float16", "bf16", "bfloat16"}:
        return 1e-4
    if normalized in {"q4", "q8", "int4", "int8", "quantized", "gguf"}:
        return 1e-3
    raise ValueError(f"unsupported reward-model dtype {dtype!r}")


def _pairwise_orientations(record: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    candidates = list(record.get("candidates") or ())
    if len(candidates) != 2:
        raise ValueError("pairwise calibration records require exactly two candidates")
    canonical = dict(record)
    reversed_payload = dict(record)
    reversed_payload["candidates"] = [candidates[1], candidates[0]]
    return [("a_b", canonical), ("b_a", reversed_payload)]


def _ranking_orientations(
    record: Mapping[str, Any], *, cap: int
) -> list[tuple[str, Mapping[str, Any]]]:
    candidates = list(record.get("candidates") or ())
    if len(candidates) < 2:
        raise ValueError("ranking calibration records require at least two candidates")
    count = min(len(candidates), int(cap))
    result: list[tuple[str, Mapping[str, Any]]] = []
    for offset in range(count):
        payload = dict(record)
        payload["candidates"] = candidates[offset:] + candidates[:offset]
        result.append(("canonical" if offset == 0 else f"rotation_{offset}", payload))
    return result


def _reviewed_probes(
    record: Mapping[str, Any], allowed_kinds: set[str]
) -> list[tuple[str, str, Mapping[str, Any]]]:
    result: list[tuple[str, str, Mapping[str, Any]]] = []
    probes = record.get("reviewed_variants") or ()
    for index, probe in enumerate(probes):
        if not isinstance(probe, Mapping) or probe.get("reviewed") is not True:
            continue
        kind = str(probe.get("kind", "")).strip().lower()
        if kind not in allowed_kinds:
            continue
        payload = dict(record)
        payload.update(dict(probe.get("payload") or {}))
        result.append((kind, f"probe_{kind}_{index}", payload))
    return result


def iter_calibration_protocol(
    records: Sequence[Mapping[str, Any]],
    protocol: CalibrationProtocol,
) -> Iterator[ProtocolInvocation]:
    """Yield repeats, orientations, and probes in deterministic order.

    Calibration can expand each source record into as many as a dozen verifier
    invocations.  Keeping those payload-bearing objects in a tuple made a
    100,000-record calibration needlessly retain the full expansion before the
    first verifier call.  This iterator is the operational API used by the
    worker; :func:`expand_calibration_protocol` remains the backwards-compatible
    materializing convenience API.
    """

    ordered_records = sorted(records, key=lambda row: str(row.get("record_id", "")))
    if any(not str(record.get("record_id", "")).strip() for record in ordered_records):
        raise ValueError("every calibration record requires record_id")
    repetitions: Sequence[tuple[int, Optional[int]]]
    if protocol.deterministic:
        repetitions = ((0, None), (1, None))
    else:
        repetitions = tuple(enumerate(protocol.stochastic_seeds))

    allowed_probes = set(protocol.reviewed_probe_kinds)
    for record in ordered_records:
        record_id = str(record["record_id"])
        if protocol.task_type == "pairwise":
            orientations = _pairwise_orientations(record)
        elif protocol.task_type == "ranking":
            orientations = _ranking_orientations(record, cap=protocol.ranking_orientation_cap)
        else:
            orientations = [("canonical", dict(record))]
        perturbations = [("canonical", orientation, payload) for orientation, payload in orientations]
        perturbations.extend(_reviewed_probes(record, allowed_probes))
        for repetition_index, repetition_seed in repetitions:
            for perturbation, orientation, payload in perturbations:
                identity = {
                    "protocol_hash": protocol.protocol_hash,
                    "record_id": record_id,
                    "repetition_index": repetition_index,
                    "seed": repetition_seed,
                    "orientation": orientation,
                    "perturbation": perturbation,
                }
                yield ProtocolInvocation(
                    invocation_id=f"vinv_{_stable_hash(identity)[:24]}",
                    record_id=record_id,
                    repetition_index=repetition_index,
                    seed=repetition_seed,
                    orientation=orientation,
                    perturbation=perturbation,
                    fresh_process=True,
                    generation_settings={
                        "seed": repetition_seed,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "concurrency": 1,
                    },
                    payload=payload,
                )

    if protocol.family == "reward_model" and protocol.production_batch_size:
        selected = ordered_records[:32]
        for record in selected:
            record_id = str(record["record_id"])
            for batch_size, perturbation in (
                (1, "batch_size_one"),
                (int(protocol.production_batch_size), "production_batch_size"),
            ):
                identity = {
                    "protocol_hash": protocol.protocol_hash,
                    "record_id": record_id,
                    "batch_size": batch_size,
                    "perturbation": perturbation,
                }
                yield ProtocolInvocation(
                    invocation_id=f"vinv_{_stable_hash(identity)[:24]}",
                    record_id=record_id,
                    repetition_index=0,
                    seed=None,
                    orientation="canonical",
                    perturbation=perturbation,
                    fresh_process=True,
                    generation_settings={
                        "seed": None,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "concurrency": 1,
                    },
                    payload={
                        **dict(record),
                        "batch_size": batch_size,
                        "production_batch_size": batch_size,
                    },
                    production_batch_size=batch_size,
                )


def expand_calibration_protocol(
    records: Sequence[Mapping[str, Any]],
    protocol: CalibrationProtocol,
) -> tuple[ProtocolInvocation, ...]:
    """Materialize protocol expansion for callers that need random access."""

    return tuple(iter_calibration_protocol(records, protocol))


def reward_model_batch_consistent(
    batch_one_score: float,
    production_batch_score: float,
    *,
    dtype: str,
) -> bool:
    values = (float(batch_one_score), float(production_batch_score))
    if any(not math.isfinite(value) for value in values):
        return False
    return abs(values[0] - values[1]) <= dtype_score_tolerance(dtype)


__all__ = [
    "CALIBRATION_SEED",
    "STOCHASTIC_SEEDS",
    "CalibrationPartition",
    "CalibrationProtocol",
    "ProtocolInvocation",
    "dtype_score_tolerance",
    "expand_calibration_protocol",
    "iter_calibration_protocol",
    "grouped_calibration_confirmation_partition",
    "reward_model_batch_consistent",
]
