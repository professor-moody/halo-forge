"""Versioned trainer execution capability declarations.

This registry describes what orchestration may safely do; it does not infer
resume support from a trainer name.  Exact backend declarations take priority
over family declarations and wildcard modality declarations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

_HF_BACKEND_ALIASES = frozenset(
    {
        "auto",
        "cpu",
        "cuda",
        "hf",
        "huggingface",
        "mps",
        "rocm",
        "torch",
        "torch_cpu",
        "torch_cuda",
        "torch_mps",
        "torch_rocm",
        "transformers",
        "trl",
    }
)


def normalize_backend_family(backend: Optional[str]) -> str:
    value = str(backend or "hf").strip().lower().replace("-", "_")
    if value in _HF_BACKEND_ALIASES:
        return "hf"
    if value in {"mlx", "mlx_lm", "apple_mlx"}:
        return "mlx"
    return value


@dataclass(frozen=True)
class TrainerExecutionCapability:
    """One versioned trainer/backend segmentation contract."""

    capability_id: str
    version: int
    trainer_mode: str
    backend_family: str
    segment_unit: str
    supports_gated_execution: bool
    resume_parameter: Optional[str]
    resume_cli_flag: Optional[str]
    checkpoint_pattern: Optional[str]
    checkpoint_index: str = "filesystem"
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        capability_id = str(self.capability_id).strip()
        trainer_mode = str(self.trainer_mode).strip().lower()
        backend_family = str(self.backend_family).strip().lower()
        segment_unit = str(self.segment_unit).strip().lower()
        version = int(self.version)
        if not capability_id or not trainer_mode or not backend_family:
            raise ValueError("capability id, trainer mode, and backend family are required")
        if version <= 0:
            raise ValueError("capability version must be positive")
        if segment_unit not in {"step", "cycle", "full_trial"}:
            raise ValueError("segment_unit must be step, cycle, or full_trial")
        if self.supports_gated_execution:
            if segment_unit == "full_trial":
                raise ValueError("gated execution cannot use full_trial as its segment unit")
            if not self.checkpoint_pattern:
                raise ValueError("gated execution requires a checkpoint pattern")
        object.__setattr__(self, "capability_id", capability_id)
        object.__setattr__(self, "trainer_mode", trainer_mode)
        object.__setattr__(self, "backend_family", backend_family)
        object.__setattr__(self, "segment_unit", segment_unit)
        object.__setattr__(self, "version", version)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainerExecutionCapabilityRegistry:
    """Small deterministic registry with exact-backend override semantics."""

    def __init__(self, capabilities: Iterable[TrainerExecutionCapability] = ()) -> None:
        self._capabilities: Dict[Tuple[str, str], TrainerExecutionCapability] = {}
        for capability in capabilities:
            self.register(capability)

    def register(self, capability: TrainerExecutionCapability) -> None:
        if not isinstance(capability, TrainerExecutionCapability):
            raise TypeError("capability must be a TrainerExecutionCapability")
        key = (capability.trainer_mode, capability.backend_family)
        existing = self._capabilities.get(key)
        if existing and existing.capability_id != capability.capability_id:
            raise ValueError(
                f"capability already registered for {capability.trainer_mode}/{capability.backend_family}"
            )
        self._capabilities[key] = capability

    def resolve(
        self,
        trainer_mode: str,
        backend: Optional[str] = None,
    ) -> TrainerExecutionCapability:
        mode = str(trainer_mode).strip().lower()
        if not mode:
            raise ValueError("trainer_mode cannot be empty")
        family = normalize_backend_family(backend)
        capability = self._capabilities.get((mode, family))
        if capability is None:
            capability = self._capabilities.get((mode, "*"))
        if capability is not None:
            return capability
        return TrainerExecutionCapability(
            capability_id=f"{mode}-{family}-full-trial-v1",
            version=1,
            trainer_mode=mode,
            backend_family=family,
            segment_unit="full_trial",
            supports_gated_execution=False,
            resume_parameter=None,
            resume_cli_flag=None,
            checkpoint_pattern=None,
            checkpoint_index="none",
            reason=f"No gated execution capability is registered for {mode}/{family}.",
        )

    def list(self) -> Tuple[TrainerExecutionCapability, ...]:
        return tuple(
            self._capabilities[key]
            for key in sorted(self._capabilities, key=lambda item: (item[0], item[1]))
        )


def _hf_capability(mode: str) -> TrainerExecutionCapability:
    return TrainerExecutionCapability(
        capability_id=f"hf-{mode}-steps-v1",
        version=1,
        trainer_mode=mode,
        backend_family="hf",
        segment_unit="step",
        supports_gated_execution=True,
        resume_parameter="resume_from_checkpoint",
        resume_cli_flag="--resume",
        checkpoint_pattern="checkpoint-*",
    )


def _cycle_capability(
    mode: str,
    *,
    resume_parameter: str,
    checkpoint_pattern: str = "cycle_*",
) -> TrainerExecutionCapability:
    return TrainerExecutionCapability(
        capability_id=f"{mode}-cycles-v1",
        version=1,
        trainer_mode=mode,
        backend_family="*",
        segment_unit="cycle",
        supports_gated_execution=True,
        resume_parameter=resume_parameter,
        resume_cli_flag="--resume-from-cycle" if mode != "raft" else "--checkpoint",
        checkpoint_pattern=checkpoint_pattern,
    )


def _mlx_full_trial(mode: str) -> TrainerExecutionCapability:
    return TrainerExecutionCapability(
        capability_id=f"mlx-{mode}-full-trial-v1",
        version=1,
        trainer_mode=mode,
        backend_family="mlx",
        segment_unit="full_trial",
        supports_gated_execution=False,
        resume_parameter=None,
        resume_cli_flag=None,
        checkpoint_pattern=None,
        checkpoint_index="none",
        reason=f"MLX {mode.upper()} does not yet provide verified resumable segments.",
    )


DEFAULT_TRAINER_EXECUTION_CAPABILITIES = TrainerExecutionCapabilityRegistry(
    [
        *(_hf_capability(mode) for mode in ("sft", "cpt", "dpo", "orpo", "rm", "grpo")),
        _cycle_capability(
            "raft",
            resume_parameter="sft_checkpoint",
            checkpoint_pattern="cycle_*_final",
        ),
        _cycle_capability("vlm", resume_parameter="resume_from"),
        _cycle_capability("audio", resume_parameter="resume_from_cycle"),
        _cycle_capability("reasoning", resume_parameter="resume_from_cycle"),
        _cycle_capability("agentic", resume_parameter="resume_from_cycle"),
        _mlx_full_trial("dpo"),
        _mlx_full_trial("grpo"),
        _mlx_full_trial("cpt"),
    ]
)


def resolve_trainer_execution_capability(
    trainer_mode: str,
    backend: Optional[str] = None,
) -> TrainerExecutionCapability:
    return DEFAULT_TRAINER_EXECUTION_CAPABILITIES.resolve(trainer_mode, backend)


__all__ = [
    "DEFAULT_TRAINER_EXECUTION_CAPABILITIES",
    "TrainerExecutionCapability",
    "TrainerExecutionCapabilityRegistry",
    "normalize_backend_family",
    "resolve_trainer_execution_capability",
]
