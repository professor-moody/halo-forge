"""Stable identities and normalized observations for training-time signals.

The reward-integrity layer deliberately does not change the historical
``Verifier`` interface.  Trainers translate whatever verifier result they
already receive into these small, JSON-safe records at the point where the
result is used for filtering or optimization.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_PRODUCER_IDENTITY_NAMESPACE = "halo-forge:producer-model-identity:v1"


def normalize_producer_model_identity(
    value: Optional[str],
    identity: Optional[Mapping[str, Any]] = None,
) -> tuple[str, Dict[str, Any]]:
    """Return a hash-shaped producer identity plus truthful provenance.

    Historical trainers passed a Hugging Face name or a filesystem path in a
    field named ``producer_model_hash``.  Preserve that useful reference as
    provenance, but never present it as a content hash.  Callers that have
    actually hashed a managed checkpoint can declare that through
    ``content_available`` and ``content_hash``.
    """

    details = copy.deepcopy(dict(identity or {}))
    supplied = str(value or "").strip()
    declared_hash = str(details.get("content_hash") or "").strip().lower()
    availability_declared = "content_available" in details
    declared_available = bool(details.get("content_available", False))
    if declared_available:
        if not _SHA256_RE.fullmatch(declared_hash):
            raise ValueError(
                "available producer model content must have a SHA-256 content_hash"
            )
        details.update(
            {
                "identity_kind": "content_hash",
                "content_available": True,
                "content_hash": declared_hash,
            }
        )
        return declared_hash, details

    # A bare SHA-256 supplied to the historically hash-named field remains a
    # valid content identity.  New code should also provide explicit
    # provenance, but old integrations need not be rewritten to stay truthful.
    if not availability_declared and _SHA256_RE.fullmatch(supplied):
        digest = supplied.lower()
        details.update(
            {
                "identity_kind": "content_hash",
                "content_available": True,
                "content_hash": digest,
                "identity_source": details.get("identity_source") or "legacy_hash_field",
            }
        )
        return digest, details

    declared_identity_hash = str(details.get("identity_hash") or "").strip().lower()
    if (
        availability_declared
        and not declared_available
        and _SHA256_RE.fullmatch(declared_identity_hash)
        and supplied.lower() == declared_identity_hash
        and details.get("identity_kind") in {"reference_hash", "unavailable_hash"}
    ):
        # Session/sink defaults are normalized once up front.  Retain that
        # exact namespaced identity when each snapshot is constructed.
        return declared_identity_hash, details

    reference = str(details.get("reference") or supplied).strip()
    identity_value = {
        "namespace": _PRODUCER_IDENTITY_NAMESPACE,
        "reference": reference or None,
        "status": "referenced" if reference else "unavailable",
    }
    digest = content_hash(identity_value)
    details.update(
        {
            "identity_kind": "reference_hash" if reference else "unavailable_hash",
            "content_available": False,
            "identity_hash": digest,
            "reference": reference or None,
        }
    )
    details.pop("content_hash", None)
    return digest, details


def _mapping(value: Any) -> Dict[str, Any]:
    return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class TrainingRecordRef:
    """Dataset or deterministic virtual identity for one source occurrence."""

    record_id: str
    record_hash: str
    instance_id: str
    source_index: int
    virtual: bool = False
    source: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id or not self.record_hash or not self.instance_id:
            raise ValueError("training record identity fields cannot be empty")
        if int(self.source_index) < 0:
            raise ValueError("source_index must be non-negative")

    @classmethod
    def from_value(
        cls,
        value: "TrainingRecordRef | Mapping[str, Any] | None",
        *,
        record: Optional[Mapping[str, Any]] = None,
        source_index: int = 0,
        source: Optional[Mapping[str, Any]] = None,
    ) -> "TrainingRecordRef":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            raw = dict(value)
            # Dataset Lab lineage uses ``record_index`` while trainer signal
            # records use the more explicit ``source_index``.
            raw.setdefault("source_index", raw.pop("record_index", source_index))
            raw.setdefault("virtual", False)
            raw.setdefault("source", dict(source or {}))
            allowed = {name for name in cls.__dataclass_fields__}
            return cls(**{key: raw[key] for key in allowed if key in raw})
        return cls.virtual_identity(record or {}, source_index=source_index, source=source)

    @classmethod
    def virtual_identity(
        cls,
        record: Mapping[str, Any],
        *,
        source_index: int,
        source: Optional[Mapping[str, Any]] = None,
    ) -> "TrainingRecordRef":
        clean = copy.deepcopy(dict(record))
        record_digest = content_hash(clean)
        logical_value = next(
            (
                clean[name]
                for name in ("record_id", "id", "uuid", "key")
                if clean.get(name) is not None and str(clean.get(name)).strip()
            ),
            None,
        )
        record_id = (
            "rec_" + content_hash({"source_id": logical_value})
            if logical_value is not None
            else "rec_" + record_digest
        )
        instance_id = "virtual_inst_" + content_hash(
            {
                "record_id": record_id,
                "record_hash": record_digest,
                "source_index": int(source_index),
                "source": dict(source or {}),
            }
        )
        return cls(
            record_id=record_id,
            record_hash=record_digest,
            instance_id=instance_id,
            source_index=int(source_index),
            virtual=True,
            source=copy.deepcopy(dict(source or {})),
        )

    @classmethod
    def from_metadata(
        cls,
        metadata: Any,
        *,
        record: Mapping[str, Any],
        source_index: int,
        source: Optional[Mapping[str, Any]] = None,
    ) -> "TrainingRecordRef":
        """Resolve common Dataset Lab lineage envelopes or derive a virtual ID."""

        if isinstance(metadata, Mapping):
            candidates = (
                metadata.get("lineage"),
                metadata.get("record_identity"),
                metadata,
            )
            for candidate in candidates:
                if isinstance(candidate, Mapping) and all(
                    candidate.get(name)
                    for name in ("record_id", "record_hash", "instance_id")
                ):
                    return cls.from_value(
                        candidate,
                        source_index=source_index,
                        source=source,
                    )
        return cls.virtual_identity(record, source_index=source_index, source=source)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def hashed_media_reference(kind: str, value: Any) -> Dict[str, Any]:
    """Return a credential-free hashed reference without copying media."""

    result: Dict[str, Any] = {"kind": str(kind)}
    if isinstance(value, (str, Path)):
        raw = str(value)
        result["reference"] = raw
        path = Path(raw).expanduser()
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            result.update(
                {
                    "content_hash": digest.hexdigest(),
                    "size_bytes": path.stat().st_size,
                    "available": True,
                }
            )
        else:
            result.update({"reference_hash": content_hash(raw), "available": False})
        return result
    try:
        payload = value.tobytes()
    except Exception:
        payload = canonical_json(value).encode("utf-8")
    result.update(
        {
            "content_hash": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "available": True,
            "in_memory": True,
        }
    )
    for name in ("shape", "mode", "size"):
        attribute = getattr(value, name, None)
        if attribute is not None:
            if name == "mode":
                result[name] = str(attribute)
                continue
            try:
                result[name] = list(attribute)
            except TypeError:
                result[name] = str(attribute)
    return result


def lineage_identity_from_metadata(metadata: Any) -> Optional[Dict[str, Any]]:
    """Return a persisted lineage identity when metadata actually contains one."""

    if not isinstance(metadata, Mapping):
        return None
    for candidate in (
        metadata.get("lineage"),
        metadata.get("record_identity"),
        metadata,
    ):
        if isinstance(candidate, Mapping) and all(
            candidate.get(name) for name in ("record_id", "record_hash", "instance_id")
        ):
            return copy.deepcopy(dict(candidate))
    return None


@dataclass(frozen=True)
class VerifierObservation:
    """JSON-safe verifier result without imposing a new verifier base class."""

    reward: Optional[float]
    passed: Optional[bool]
    parsed_value: Any = None
    raw_output: Any = None
    details: Any = None
    component_trace: tuple[Dict[str, Any], ...] = ()
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    runtime_identity: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise ValueError("verifier reward must be finite")
        if self.latency_ms is not None and (
            not math.isfinite(float(self.latency_ms)) or float(self.latency_ms) < 0
        ):
            raise ValueError("verifier latency must be finite and non-negative")

    @classmethod
    def from_result(
        cls,
        result: Any,
        *,
        latency_ms: Optional[float] = None,
        runtime_identity: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
    ) -> "VerifierObservation":
        if isinstance(result, cls):
            return result
        if isinstance(result, Mapping):
            get = result.get
        else:
            get = lambda name, default=None: getattr(result, name, default)

        raw_reward = get("reward")
        reward = float(raw_reward) if raw_reward is not None else None
        raw_passed = get("passed", get("success"))
        passed = bool(raw_passed) if raw_passed is not None else None
        details = copy.deepcopy(get("details"))
        metadata = _mapping(get("metadata"))
        component_trace = get("component_trace")
        if component_trace is None and isinstance(details, Mapping):
            component_trace = details.get("component_trace")
        if component_trace is None:
            component_trace = metadata.get("component_trace")
        trace = tuple(
            copy.deepcopy(dict(item))
            for item in (component_trace or ())
            if isinstance(item, Mapping)
        )
        parsed_value = get("parsed_value")
        if parsed_value is None and isinstance(details, Mapping):
            parsed_value = details.get("parsed_value", details.get("extracted_answer"))
        raw_output = get("raw_output")
        if raw_output is None:
            raw_output = metadata.get("raw_output")
        observed_error = error or get("error")
        observed_runtime = _mapping(get("runtime_identity"))
        observed_runtime.update(_mapping(runtime_identity))
        return cls(
            reward=reward,
            passed=passed,
            parsed_value=copy.deepcopy(parsed_value),
            raw_output=copy.deepcopy(raw_output),
            details=details,
            component_trace=trace,
            latency_ms=(
                float(latency_ms)
                if latency_ms is not None
                else (
                    float(get("latency_ms")) if get("latency_ms") is not None else None
                )
            ),
            error=str(observed_error) if observed_error else None,
            runtime_identity=observed_runtime,
        )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["component_trace"] = [copy.deepcopy(item) for item in self.component_trace]
        return value


@dataclass(frozen=True)
class TrainingSignalSnapshot:
    """One exact optimizer-verifier observation for one generated output."""

    snapshot_id: str
    record: TrainingRecordRef
    candidate_ordinal: int
    prompt: Any
    context: Any
    output: Any
    expected: Any
    media: tuple[Dict[str, Any], ...]
    generation_settings: Dict[str, Any]
    training_observation: VerifierObservation
    selected: Optional[bool]
    selection_reason: Optional[str]
    producer_model_hash: str
    checkpoint_hash: Optional[str]
    run_id: str
    segment_id: str
    boundary: str
    runtime_identity: Dict[str, Any] = field(default_factory=dict)
    selection_stratum: Optional[str] = None
    occurrence_id: Optional[str] = None
    identity_mode: str = "legacy_content_fallback"
    producer_model_identity: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.candidate_ordinal) < 0:
            raise ValueError("candidate_ordinal must be non-negative")
        if not _SHA256_RE.fullmatch(str(self.producer_model_hash or "")):
            raise ValueError("producer_model_hash must be a SHA-256 identity")
        if self.identity_mode not in {
            "trainer_occurrence",
            "legacy_content_fallback",
        }:
            raise ValueError("unknown training-signal identity mode")
        if self.identity_mode == "trainer_occurrence" and not self.occurrence_id:
            raise ValueError("trainer occurrence identity requires occurrence_id")

    @classmethod
    def create(
        cls,
        *,
        record: TrainingRecordRef | Mapping[str, Any] | None,
        candidate_ordinal: int,
        prompt: Any,
        output: Any,
        training_observation: VerifierObservation | Mapping[str, Any] | Any,
        run_id: str,
        segment_id: str,
        boundary: str,
        context: Any = None,
        expected: Any = None,
        media: Any = (),
        generation_settings: Optional[Mapping[str, Any]] = None,
        selected: Optional[bool] = None,
        selection_reason: Optional[str] = None,
        producer_model_hash: Optional[str] = None,
        producer_model_identity: Optional[Mapping[str, Any]] = None,
        checkpoint_hash: Optional[str] = None,
        runtime_identity: Optional[Mapping[str, Any]] = None,
        occurrence_id: Optional[str] = None,
        source_index: int = 0,
        source: Optional[Mapping[str, Any]] = None,
    ) -> "TrainingSignalSnapshot":
        ref = TrainingRecordRef.from_value(
            record,
            record={"prompt": prompt, "expected": expected},
            source_index=source_index,
            source=source,
        )
        observation = VerifierObservation.from_result(training_observation)
        media_items = tuple(
            copy.deepcopy(dict(item)) for item in (media or ()) if isinstance(item, Mapping)
        )
        resolved_occurrence = str(occurrence_id or "").strip() or None
        identity = {
            "run_id": str(run_id),
            "segment_id": str(segment_id),
            "boundary": str(boundary),
            "instance_id": ref.instance_id,
            "candidate_ordinal": int(candidate_ordinal),
        }
        if resolved_occurrence is not None:
            # Trainer loop coordinates, rather than output bytes, are the
            # event identity.  This keeps retries idempotent while retaining
            # repeated identical generations as separate observations.
            identity["occurrence_id"] = resolved_occurrence
            identity_mode = "trainer_occurrence"
        else:
            # Explicit compatibility path for third-party/legacy sinks that
            # have not adopted occurrence coordinates yet.
            identity["output_hash"] = content_hash(output)
            identity_mode = "legacy_content_fallback"
        resolved_producer_hash, resolved_producer_identity = (
            normalize_producer_model_identity(
                producer_model_hash,
                producer_model_identity,
            )
        )
        return cls(
            snapshot_id="signal_" + content_hash(identity),
            record=ref,
            candidate_ordinal=int(candidate_ordinal),
            prompt=copy.deepcopy(prompt),
            context=copy.deepcopy(context),
            output=copy.deepcopy(output),
            expected=copy.deepcopy(expected),
            media=media_items,
            generation_settings=copy.deepcopy(dict(generation_settings or {})),
            training_observation=observation,
            selected=selected,
            selection_reason=selection_reason,
            producer_model_hash=resolved_producer_hash,
            checkpoint_hash=checkpoint_hash,
            run_id=str(run_id),
            segment_id=str(segment_id),
            boundary=str(boundary),
            runtime_identity=copy.deepcopy(dict(runtime_identity or {})),
            occurrence_id=resolved_occurrence,
            identity_mode=identity_mode,
            producer_model_identity=resolved_producer_identity,
        )

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["record"] = self.record.to_dict()
        value["training_observation"] = self.training_observation.to_dict()
        value["media"] = [copy.deepcopy(item) for item in self.media]
        return value


__all__ = [
    "TrainingRecordRef",
    "TrainingSignalSnapshot",
    "VerifierObservation",
    "canonical_json",
    "content_hash",
    "hashed_media_reference",
    "lineage_identity_from_metadata",
    "normalize_producer_model_identity",
]
