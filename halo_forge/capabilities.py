"""
Capability registry and validation for modality training commands.
"""

import json
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


def _collect_local_model_hints(model_path: Path) -> list[str]:
    """
    Collect lower-cased identity hints from local checkpoint paths.

    Reads common model metadata files to recover originating base model family
    (for example `adapter_config.json` contains `base_model_name_or_path`).
    """
    hints: list[str] = []
    candidate_dirs: list[Path] = []

    if model_path.is_file():
        candidate_dirs.append(model_path.parent)
    else:
        candidate_dirs.append(model_path)

    # Check nearby folders as many runs point to final_model/checkpoint subdirs.
    candidate_dirs.append(model_path.parent)
    candidate_dirs.append(model_path.parent.parent)

    for path_hint in (
        model_path.name,
        model_path.parent.name,
        model_path.parent.parent.name,
    ):
        if path_hint:
            hints.append(path_hint.lower())

    hint_files = (
        "adapter_config.json",
        "config.json",
        "tokenizer_config.json",
    )
    hint_keys = (
        "base_model_name_or_path",
        "_name_or_path",
        "model_name",
        "model_path",
        "model",
        "model_type",
        "architectures",
    )

    for directory in candidate_dirs:
        try:
            if not directory or not directory.exists() or not directory.is_dir():
                continue
        except OSError:
            continue

        for filename in hint_files:
            metadata_path = directory / filename
            if not metadata_path.exists():
                continue
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            for key in hint_keys:
                value = payload.get(key)
                if isinstance(value, str) and value:
                    hints.append(value.lower())
                elif isinstance(value, list):
                    hints.extend(str(item).lower() for item in value if item)

    return hints


def _is_model_family_supported(model_name: str, capability: ModalityTrainCapability) -> bool:
    """Check model support against configured family patterns."""
    families = capability.supported_model_families
    if "*" in families:
        return True

    model_lower = (model_name or "").lower()
    if any(token in model_lower for token in families):
        return True

    if _looks_like_local_path(model_name):
        # Use model metadata when available so generic local folder names still
        # validate against supported family policies.
        hints = _collect_local_model_hints(Path(model_name))
        return any(token in hint for hint in hints for token in families)

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
