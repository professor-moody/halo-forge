"""Training-time reward observations for Reward Integrity audits."""

from .models import (
    TrainingRecordRef,
    TrainingSignalSnapshot,
    VerifierObservation,
    hashed_media_reference,
    lineage_identity_from_metadata,
)
from .lifecycle import complete_signal_boundary
from .lineage import (
    TrainingArtifactRecordResolver,
    record_resolver_from_training_artifact,
)
from .registry import (
    CaptureFidelity,
    TRAINING_SIGNAL_CAPABILITIES,
    TrainingSignalCapabilityDescriptor,
    TrainingSignalCapabilityRegistry,
    default_training_signal_capabilities,
)
from .sink import (
    PROTOCOLS,
    TrainingSignalShard,
    TrainingSignalSink,
    build_training_runtime_identity,
    load_training_signal_shard,
    verify_training_signal_shard,
)
from .session import BoundarySignalSession, default_audit_boundaries

__all__ = [
    "CaptureFidelity",
    "BoundarySignalSession",
    "PROTOCOLS",
    "TRAINING_SIGNAL_CAPABILITIES",
    "TrainingRecordRef",
    "TrainingArtifactRecordResolver",
    "TrainingSignalCapabilityDescriptor",
    "TrainingSignalCapabilityRegistry",
    "TrainingSignalShard",
    "TrainingSignalSink",
    "TrainingSignalSnapshot",
    "VerifierObservation",
    "build_training_runtime_identity",
    "default_training_signal_capabilities",
    "default_audit_boundaries",
    "complete_signal_boundary",
    "hashed_media_reference",
    "lineage_identity_from_metadata",
    "record_resolver_from_training_artifact",
    "load_training_signal_shard",
    "verify_training_signal_shard",
]
