"""Truthful, versioned trainer signal capability registry."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional


class CaptureFidelity(str, Enum):
    EXACT = "exact"
    SAMPLED = "sampled"
    AGGREGATE_ONLY = "aggregate_only"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class TrainingSignalCapabilityDescriptor:
    id: str
    version: str
    trainer: str
    backend: str
    boundary_unit: str
    resumable: bool
    available_boundaries: tuple[str, ...]
    fidelity: CaptureFidelity
    identity_mapping: Dict[str, str] = field(default_factory=dict)
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_mapping: Dict[str, str] = field(default_factory=dict)
    reference_mapping: Dict[str, str] = field(default_factory=dict)
    media_mapping: Dict[str, str] = field(default_factory=dict)
    verifier_mapping: Dict[str, str] = field(default_factory=dict)
    candidate_multiplicity: str = "one"
    unavailable_fields: tuple[str, ...] = ()
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.version.strip():
            raise ValueError("training signal capabilities require id and version")
        if self.fidelity == CaptureFidelity.UNAVAILABLE and not self.reason:
            raise ValueError("unavailable signal capabilities require a reason")
        if not self.available_boundaries:
            raise ValueError("training signal capabilities require at least one boundary")

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["fidelity"] = self.fidelity.value
        value["available_boundaries"] = list(self.available_boundaries)
        value["unavailable_fields"] = list(self.unavailable_fields)
        return value


class TrainingSignalCapabilityRegistry:
    def __init__(self, values: Iterable[TrainingSignalCapabilityDescriptor] = ()) -> None:
        self._values: Dict[str, TrainingSignalCapabilityDescriptor] = {}
        for value in values:
            self.register(value)

    def register(self, value: TrainingSignalCapabilityDescriptor) -> None:
        existing = self._values.get(value.id)
        if existing is not None and existing != value:
            raise ValueError(f"training signal capability already registered: {value.id}")
        self._values[value.id] = value

    def get(self, capability_id: str) -> TrainingSignalCapabilityDescriptor:
        try:
            return self._values[str(capability_id)]
        except KeyError as exc:
            raise KeyError(f"unknown training signal capability: {capability_id}") from exc

    def resolve(self, trainer: str, backend: str) -> TrainingSignalCapabilityDescriptor:
        matches = [
            value
            for value in self._values.values()
            if value.trainer == trainer and value.backend == backend
        ]
        if len(matches) != 1:
            raise KeyError(f"no unique training signal capability for {trainer}/{backend}")
        return matches[0]

    def list(self) -> tuple[TrainingSignalCapabilityDescriptor, ...]:
        return tuple(copy.deepcopy(self._values[key]) for key in sorted(self._values))


def default_training_signal_capabilities() -> TrainingSignalCapabilityRegistry:
    common_identity = {
        "record_id": "lineage.record_id",
        "record_hash": "lineage.record_hash",
        "instance_id": "lineage.instance_id",
    }
    common_verifier = {
        "reward": "result.reward",
        "passed": "result.success",
        "details": "result.details",
        "component_trace": "result.component_trace|metadata.component_trace",
    }
    values = (
        TrainingSignalCapabilityDescriptor(
            # The PyTorch RAFT loop writes cycle checkpoints, but it cannot
            # resume its optimizer/model state in a new process yet.  Keep the
            # capture truthful and expose a final gate only until that loader
            # exists; a checkpoint-shaped directory is not resumability.
            "raft:hf", "2", "raft", "hf", "final", False, ("final",),
            CaptureFidelity.SAMPLED, common_identity, {"prompt": "prompt"},
            {"output": "completion"}, {}, {}, common_verifier, "samples_per_prompt",
            ("resumable_optimizer_state",),
            "PyTorch RAFT has no verified cross-process cycle resume loader",
        ),
        TrainingSignalCapabilityDescriptor(
            "raft:mlx", "1", "raft", "mlx", "cycle", True, ("cycle", "final"),
            CaptureFidelity.SAMPLED, common_identity, {"prompt": "prompt"},
            {"output": "completion"}, {}, {}, common_verifier, "samples_per_prompt",
        ),
        TrainingSignalCapabilityDescriptor(
            "grpo:hf", "2", "grpo", "hf", "step", True, ("step", "final"),
            CaptureFidelity.SAMPLED, common_identity, {"prompt": "prompt"},
            {"output": "completion"}, {}, {}, common_verifier, "num_generations",
            ("optimizer_step_during_reward_callback",),
        ),
        TrainingSignalCapabilityDescriptor(
            "grpo:mlx", "1", "grpo", "mlx", "final", False, ("final",),
            CaptureFidelity.SAMPLED, common_identity, {"prompt": "prompt"},
            {"output": "completion"}, {}, {}, common_verifier, "num_generations",
        ),
        TrainingSignalCapabilityDescriptor(
            "reasoning:hf", "1", "reasoning", "hf", "cycle", True,
            ("cycle", "final"), CaptureFidelity.SAMPLED, common_identity,
            {"prompt": "sample.question"}, {"output": "completion"},
            {"expected": "sample.answer"}, {}, common_verifier, "samples_per_prompt",
        ),
        TrainingSignalCapabilityDescriptor(
            "agentic:hf", "1", "agentic", "hf", "cycle", True,
            ("cycle", "final"), CaptureFidelity.SAMPLED, common_identity,
            {"prompt": "formatted_prompt", "tools": "sample.tools"},
            {"output": "completion"}, {"expected": "sample.expected_calls"}, {},
            common_verifier, "samples_per_prompt",
        ),
        TrainingSignalCapabilityDescriptor(
            "vlm:hf", "1", "vlm", "hf", "cycle", True, ("cycle", "final"),
            CaptureFidelity.SAMPLED, common_identity, {"prompt": "sample.prompt"},
            {"output": "completion"}, {"expected": "sample.ground_truth"},
            {"image": "sample.image"}, common_verifier, "samples_per_prompt",
        ),
        TrainingSignalCapabilityDescriptor(
            "audio:hf", "1", "audio", "hf", "cycle", True,
            ("cycle", "final"), CaptureFidelity.SAMPLED, common_identity,
            {"task": "sample.task"}, {"output": "prediction"},
            {"expected": "sample.text"}, {"audio": "sample.audio_path"},
            common_verifier, "samples_per_prompt",
        ),
    )
    return TrainingSignalCapabilityRegistry(values)


TRAINING_SIGNAL_CAPABILITIES = default_training_signal_capabilities()


__all__ = [
    "CaptureFidelity",
    "TRAINING_SIGNAL_CAPABILITIES",
    "TrainingSignalCapabilityDescriptor",
    "TrainingSignalCapabilityRegistry",
    "default_training_signal_capabilities",
]
