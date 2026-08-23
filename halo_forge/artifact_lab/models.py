"""Public domain types for Halo Forge's content-addressed artifact lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

from halo_forge.version import PACKAGE_VERSION

from .hashing import fingerprint

ARTIFACT_KINDS = frozenset(
    {
        "checkpoint",
        "adapter",
        "final",
        "merged",
        "converted",
        "quantized",
        "export_bundle",
    }
)
LOCATION_KINDS = frozenset({"referenced", "managed"})
OPERATION_TYPES = frozenset({"bake", "combine", "convert", "quantize", "export"})
VERIFICATION_LEVELS = (
    "unverified",
    "hash_verified",
    "structural_verified",
    "load_verified",
    "round_trip_verified",
)
_VERIFICATION_RANK = {name: index for index, name in enumerate(VERIFICATION_LEVELS)}


def _require_choice(value: str, choices: frozenset[str], label: str) -> str:
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"Unknown {label} {value!r}; choose from: {', '.join(sorted(choices))}")
    return normalized


def _safe_metadata(value: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class ArtifactBlob:
    """Unique artifact content, independent of runs and filesystem locations."""

    content_hash: str
    artifact_kind: str
    format: str
    size_bytes: int
    file_count: int
    created_at: str
    dtype: Optional[str] = None
    quantization: Optional[str] = None
    quantization_method: Optional[str] = None
    # Registration hashes the bytes, but it does not prove that a backend can
    # load or execute them.  Later evidence is recorded separately rather than
    # silently upgrading this immutable creation-time fact.
    integrity: str = "hash_verified"
    manifest_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_kind",
            _require_choice(self.artifact_kind, ARTIFACT_KINDS, "artifact kind"),
        )
        if not self.content_hash or len(self.content_hash) != 64:
            raise ValueError("content_hash must be a 64-character SHA-256 hex digest")
        int(self.content_hash, 16)
        object.__setattr__(self, "content_hash", self.content_hash.lower())
        object.__setattr__(self, "format", self.format.strip().lower())
        if self.size_bytes < 0 or self.file_count < 1:
            raise ValueError("ArtifactBlob requires non-negative size and at least one file")
        if self.quantization_method and self.quantization_method.strip().lower() in {
            "qat",
            "quantization-aware-training",
            "quantization_aware_training",
        }:
            raise ValueError(
                "Halo Forge's current artifact quantization is post-training quantization, not QAT"
            )
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata))

    @property
    def id(self) -> str:
        return f"blob-{self.content_hash[:24]}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactLocation:
    """One referenced run path or managed copy of an artifact blob."""

    id: str
    content_hash: str
    location_kind: str
    path: str
    created_at: str
    verified_at: str
    source_path: Optional[str] = None
    state: str = "available"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "location_kind",
            _require_choice(self.location_kind, LOCATION_KINDS, "location kind"),
        )
        if not self.id:
            raise ValueError("ArtifactLocation.id is required")

    @property
    def managed(self) -> bool:
        return self.location_kind == "managed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelArtifactOccurrence:
    """A run/checkpoint occurrence that points at unique artifact content."""

    id: str
    content_hash: str
    artifact_kind: str
    location_id: str
    created_at: str
    run_id: Optional[str] = None
    run_group_id: Optional[str] = None
    trial_id: Optional[str] = None
    segment_id: Optional[str] = None
    step: Optional[int] = None
    cycle: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_kind",
            _require_choice(self.artifact_kind, ARTIFACT_KINDS, "artifact kind"),
        )
        object.__setattr__(self, "metadata", _safe_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactRegistration:
    blob: ArtifactBlob
    location: ArtifactLocation
    occurrence: ModelArtifactOccurrence
    reused_blob: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "blob": self.blob.to_dict(),
            "location": self.location.to_dict(),
            "occurrence": self.occurrence.to_dict(),
            "reused_blob": self.reused_blob,
        }


@dataclass(frozen=True)
class ArtifactEdge:
    """Ordered multi-parent lineage edge."""

    parent_content_hash: str
    child_content_hash: str
    ordinal: int
    relationship: str
    operation_fingerprint: str

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ArtifactEdge.ordinal cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LineageGraph:
    root_content_hash: str
    direction: str
    blobs: tuple[ArtifactBlob, ...]
    edges: tuple[ArtifactEdge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_content_hash": self.root_content_hash,
            "direction": self.direction,
            "blobs": [item.to_dict() for item in self.blobs],
            "edges": [item.to_dict() for item in self.edges],
        }


@dataclass(frozen=True)
class OperationSpec:
    """Resolved immutable request for an artifact-producing operation."""

    operation_type: str
    input_content_hashes: tuple[str, ...]
    output_kind: str
    output_format: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    tool_id: str = "halo-forge"
    tool_version: str = PACKAGE_VERSION
    output_dtype: Optional[str] = None
    output_quantization: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_type",
            _require_choice(self.operation_type, OPERATION_TYPES, "operation type"),
        )
        object.__setattr__(
            self, "output_kind", _require_choice(self.output_kind, ARTIFACT_KINDS, "artifact kind")
        )
        if not self.input_content_hashes:
            raise ValueError("An artifact operation needs at least one input")
        if self.operation_type == "combine" and len(self.input_content_hashes) < 2:
            raise ValueError("A combine operation needs at least two ordered inputs")
        normalized_hashes = []
        for content_hash in self.input_content_hashes:
            if len(content_hash) != 64:
                raise ValueError("Operation input hashes must be 64-character SHA-256 digests")
            int(content_hash, 16)
            normalized_hashes.append(content_hash.lower())
        object.__setattr__(self, "input_content_hashes", tuple(normalized_hashes))
        object.__setattr__(self, "output_format", self.output_format.strip().lower())
        parameters = _safe_metadata(self.parameters)
        if self.output_quantization is None and parameters.get("quantization"):
            object.__setattr__(self, "output_quantization", str(parameters["quantization"]))
        if self.operation_type == "quantize" and not self.output_quantization:
            raise ValueError("A quantize operation requires output_quantization")
        prohibited = {"qat", "quantization-aware-training", "quantization_aware_training"}
        declared_method = (
            str(
                parameters.get("quantization_method")
                or parameters.get("quantization_training_method")
                or ""
            )
            .strip()
            .lower()
        )
        declared_quantization = str(self.output_quantization or "").strip().lower()
        if declared_method in prohibited or declared_quantization in prohibited:
            raise ValueError(
                "Halo Forge's current artifact quantization is post-training quantization, not QAT"
            )
        object.__setattr__(self, "parameters", parameters)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = {
            "operation_type": self.operation_type,
            "input_content_hashes": list(self.input_content_hashes),
            "output_kind": self.output_kind,
            "output_format": self.output_format,
            "output_dtype": self.output_dtype,
            "output_quantization": self.output_quantization,
            "parameters": dict(self.parameters),
            "tool_id": self.tool_id,
            "tool_version": self.tool_version,
        }
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value


@dataclass(frozen=True)
class ArtifactOperation:
    id: str
    fingerprint: str
    spec: OperationSpec
    status: str
    created_at: str
    completed_at: Optional[str] = None
    output_content_hash: Optional[str] = None
    output_location_id: Optional[str] = None
    engine_metadata: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fingerprint": self.fingerprint,
            "spec": self.spec.to_dict(),
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "output_content_hash": self.output_content_hash,
            "output_location_id": self.output_location_id,
            "engine_metadata": dict(self.engine_metadata),
            "error": self.error,
            "reused": self.reused,
        }


@dataclass(frozen=True)
class ArtifactVerification:
    content_hash: str
    location_id: str
    checked_at: str
    passed: bool
    content_hash_matches: bool
    structural_checks: Mapping[str, Optional[bool]]
    structural_checked: bool = False
    loadability_checked: bool = False
    round_trip_checked: bool = False
    verification_level: str = "unverified"
    errors: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.verification_level not in _VERIFICATION_RANK
            and self.verification_level != "failed"
        ):
            raise ValueError(f"unknown verification level: {self.verification_level}")

    def satisfies(self, required_level: str) -> bool:
        """Return whether this report contains successful evidence at ``required_level``."""

        if required_level not in _VERIFICATION_RANK:
            raise ValueError(f"unknown required verification level: {required_level}")
        if not self.passed or self.verification_level == "failed":
            return False
        return _VERIFICATION_RANK[self.verification_level] >= _VERIFICATION_RANK[required_level]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CleanupProtections:
    """Higher-layer references that make content ineligible for cleanup."""

    active: frozenset[str] = frozenset()
    pinned: frozenset[str] = frozenset()
    promoted: frozenset[str] = frozenset()
    serving: frozenset[str] = frozenset()
    evaluation_referenced: frozenset[str] = frozenset()
    lineage_required: frozenset[str] = frozenset()
    active_staging: frozenset[str] = frozenset()

    def reasons_for(self, content_hash: str) -> tuple[str, ...]:
        reasons = []
        for name in (
            "active",
            "pinned",
            "promoted",
            "serving",
            "evaluation_referenced",
            "lineage_required",
        ):
            if content_hash in getattr(self, name):
                reasons.append(name)
        return tuple(reasons)

    def all_hashes(self) -> frozenset[str]:
        return frozenset().union(
            self.active,
            self.pinned,
            self.promoted,
            self.serving,
            self.evaluation_referenced,
            self.lineage_required,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: sorted(getattr(self, name))
            for name in (
                "active",
                "pinned",
                "promoted",
                "serving",
                "evaluation_referenced",
                "lineage_required",
                "active_staging",
            )
        }


@dataclass(frozen=True)
class CleanupCandidate:
    resource_type: str
    identifier: str
    path: str
    reclaimable_bytes: int
    age_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProtectedArtifact:
    content_hash: str
    reasons: tuple[str, ...]
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CleanupPlan:
    id: str
    created_at: str
    candidates: tuple[CleanupCandidate, ...]
    protected: tuple[ProtectedArtifact, ...]
    protections: CleanupProtections
    reclaimable_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "candidates": [item.to_dict() for item in self.candidates],
            "protected": [item.to_dict() for item in self.protected],
            "protections": self.protections.to_dict(),
            "reclaimable_bytes": self.reclaimable_bytes,
        }


@dataclass(frozen=True)
class CleanupResult:
    action_id: str
    plan_id: str
    created_at: str
    trashed: tuple[str, ...]
    skipped: Mapping[str, str]
    reclaimed_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StorageInventory:
    root: str
    generated_at: str
    blob_count: int
    managed_blob_count: int
    referenced_location_count: int
    managed_bytes: int
    metadata_bytes: int
    staging_bytes: int
    trash_bytes: int
    free_bytes: int
    total_bytes: int

    @property
    def used_bytes(self) -> int:
        return self.managed_bytes + self.metadata_bytes + self.staging_bytes + self.trash_bytes

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "used_bytes": self.used_bytes}


@dataclass(frozen=True)
class PortableExportBundle:
    id: str
    source_content_hash: str
    path: str
    created_at: str
    manifest_hash: str
    size_bytes: int
    file_count: int
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def operation_spec_from_dict(value: Mapping[str, Any]) -> OperationSpec:
    return OperationSpec(
        operation_type=str(value["operation_type"]),
        input_content_hashes=tuple(str(item) for item in value["input_content_hashes"]),
        output_kind=str(value["output_kind"]),
        output_format=str(value["output_format"]),
        parameters=dict(value.get("parameters") or {}),
        tool_id=str(value.get("tool_id") or "halo-forge"),
        tool_version=str(value.get("tool_version") or PACKAGE_VERSION),
        output_dtype=value.get("output_dtype"),
        output_quantization=value.get("output_quantization"),
    )


__all__ = [
    "ARTIFACT_KINDS",
    "LOCATION_KINDS",
    "OPERATION_TYPES",
    "VERIFICATION_LEVELS",
    "ArtifactBlob",
    "ArtifactEdge",
    "ArtifactLocation",
    "ArtifactOperation",
    "ArtifactRegistration",
    "ArtifactVerification",
    "CleanupCandidate",
    "CleanupPlan",
    "CleanupProtections",
    "CleanupResult",
    "LineageGraph",
    "ModelArtifactOccurrence",
    "PortableExportBundle",
    "ProtectedArtifact",
    "OperationSpec",
    "StorageInventory",
    "operation_spec_from_dict",
]
