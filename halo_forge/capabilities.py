"""
Capability registry and validation for modality training commands.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class ModalityTrainCapability:
    """Runtime contract for a modality train command."""

    status: str
    supported_model_families: Tuple[str, ...]
    prototype_requires_flag: bool
    notes: str


@dataclass(frozen=True)
class CapabilityCheckResult:
    """Outcome of validating a modality train request."""

    allowed: bool
    reason: str
    message: str
    capability: ModalityTrainCapability


MODALITY_TRAIN_CAPABILITIES: Dict[str, ModalityTrainCapability] = {
    "vlm": ModalityTrainCapability(
        status="prototype",
        supported_model_families=("qwen2-vl", "qwen-vl", "llava"),
        prototype_requires_flag=True,
        notes="Real update loop is implemented but still rollout-gated.",
    ),
    "audio": ModalityTrainCapability(
        status="prototype",
        supported_model_families=("whisper",),
        prototype_requires_flag=True,
        notes="Real update loop is implemented for Whisper-family adapters.",
    ),
    "reasoning": ModalityTrainCapability(
        status="prototype",
        supported_model_families=("*",),
        prototype_requires_flag=True,
        notes="Real update loop is implemented and rollout-gated.",
    ),
    "agentic": ModalityTrainCapability(
        status="prototype",
        supported_model_families=("*",),
        prototype_requires_flag=True,
        notes="Real update loop is implemented and rollout-gated.",
    ),
}


def _looks_like_local_path(model_name: str) -> bool:
    """Return True when model name appears to reference a local path."""
    try:
        return Path(model_name).exists()
    except OSError:
        return False


def _is_model_family_supported(model_name: str, capability: ModalityTrainCapability) -> bool:
    """Check model support against configured family patterns."""
    families = capability.supported_model_families
    if "*" in families:
        return True

    model_lower = (model_name or "").lower()
    if any(token in model_lower for token in families):
        return True

    if _looks_like_local_path(model_name):
        # Local checkpoints are allowed only when the folder name itself
        # still advertises a supported family token.
        path_name = Path(model_name).name.lower()
        return any(token in path_name for token in families)

    return False


def check_modality_train_capability(
    modality: str,
    model_name: str,
    allow_prototype_train: bool = False,
    dry_run: bool = False,
) -> CapabilityCheckResult:
    """
    Validate whether a modality train request is allowed.
    """
    capability = MODALITY_TRAIN_CAPABILITIES[modality]

    if (
        capability.status == "prototype"
        and capability.prototype_requires_flag
        and not allow_prototype_train
        and not dry_run
    ):
        message = (
            "CAPABILITY_ERROR "
            f"modality={modality} "
            "reason=prototype_flag_required "
            "required_flag=--allow-prototype-train\n"
            f"{modality} train is rollout-gated. Re-run with --allow-prototype-train."
        )
        return CapabilityCheckResult(
            allowed=False,
            reason="prototype_flag_required",
            message=message,
            capability=capability,
        )

    if not _is_model_family_supported(model_name, capability):
        families = ",".join(capability.supported_model_families)
        message = (
            "CAPABILITY_ERROR "
            f"modality={modality} "
            "reason=unsupported_model "
            f"model={model_name} "
            f"supported_families={families}\n"
            f"Unsupported model family for {modality} training. "
            f"Supported families: {families}."
        )
        return CapabilityCheckResult(
            allowed=False,
            reason="unsupported_model",
            message=message,
            capability=capability,
        )

    return CapabilityCheckResult(
        allowed=True,
        reason="ok",
        message="",
        capability=capability,
    )
