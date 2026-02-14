"""
Shared launch-contract definitions and validation helpers.

These contracts are the single source of truth for UI service preflight checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class LaunchContract:
    """Declarative validation contract for launch payloads."""

    required_text_fields: tuple[str, ...] = ()
    positive_int_fields: tuple[str, ...] = ()
    optional_positive_int_fields: tuple[str, ...] = ()
    non_negative_float_fields: tuple[str, ...] = ()
    ratio_fields: tuple[str, ...] = ()
    required_existing_paths: tuple[str, ...] = ()
    optional_existing_paths: tuple[str, ...] = ()


SFT_LAUNCH_CONTRACT = LaunchContract(
    required_text_fields=("model", "dataset", "output_dir"),
    positive_int_fields=("epochs", "batch_size", "gradient_accumulation_steps"),
)

RAFT_LAUNCH_CONTRACT = LaunchContract(
    required_text_fields=("model", "prompts", "output_dir"),
    positive_int_fields=("cycles", "samples_per_prompt", "min_samples", "max_new_tokens"),
    non_negative_float_fields=("reward_threshold",),
    ratio_fields=("keep_percent",),
    required_existing_paths=("prompts",),
    optional_existing_paths=("checkpoint",),
)

BENCHMARK_LAUNCH_CONTRACT = LaunchContract(
    required_text_fields=("model", "benchmark_name"),
    positive_int_fields=("samples_per_prompt",),
)

VLM_TRAIN_LAUNCH_CONTRACT = LaunchContract(
    required_text_fields=("model", "dataset", "output_dir"),
    positive_int_fields=("cycles", "samples_per_prompt"),
)

AUDIO_TRAIN_LAUNCH_CONTRACT = LaunchContract(
    required_text_fields=("model", "dataset", "task", "output_dir"),
    positive_int_fields=("cycles",),
    optional_positive_int_fields=("limit",),
)

REASONING_TRAIN_LAUNCH_CONTRACT = LaunchContract(
    required_text_fields=("model", "dataset", "output_dir"),
    positive_int_fields=("cycles",),
    optional_positive_int_fields=("limit",),
)

AGENTIC_TRAIN_LAUNCH_CONTRACT = LaunchContract(
    required_text_fields=("model", "dataset", "output_dir"),
    positive_int_fields=("cycles",),
    optional_positive_int_fields=("limit",),
)

INFERENCE_OPTIMIZE_LAUNCH_CONTRACT = LaunchContract(
    required_text_fields=("model", "output_dir", "target_precision"),
    non_negative_float_fields=("target_latency",),
)

INFERENCE_BENCHMARK_LAUNCH_CONTRACT = LaunchContract(
    required_text_fields=("model", "output_dir"),
    positive_int_fields=("num_prompts", "max_tokens", "warmup"),
)

MODALITY_TRAIN_LAUNCH_CONTRACTS: dict[str, LaunchContract] = {
    "vlm": VLM_TRAIN_LAUNCH_CONTRACT,
    "audio": AUDIO_TRAIN_LAUNCH_CONTRACT,
    "reasoning": REASONING_TRAIN_LAUNCH_CONTRACT,
    "agentic": AGENTIC_TRAIN_LAUNCH_CONTRACT,
}

# UI supports launching these training modes end-to-end today.
UI_SUPPORTED_TRAINING_MODES: tuple[str, ...] = (
    "sft",
    "raft",
    "vlm",
    "audio",
    "reasoning",
    "agentic",
)
UI_DEFERRED_TRAINING_MODES: tuple[str, ...] = ()


def validate_launch_payload(
    payload: Mapping[str, Any],
    contract: LaunchContract,
) -> Dict[str, Any]:
    """Validate and normalize a launch payload against a contract."""
    normalized: Dict[str, Any] = dict(payload)

    for field_name in contract.required_text_fields:
        normalized[field_name] = _validate_required_text(
            normalized.get(field_name),
            field_name,
        )

    for field_name in contract.positive_int_fields:
        normalized[field_name] = _validate_positive_int(
            normalized.get(field_name),
            field_name,
            allow_none=False,
        )

    for field_name in contract.optional_positive_int_fields:
        normalized[field_name] = _validate_positive_int(
            normalized.get(field_name),
            field_name,
            allow_none=True,
        )

    for field_name in contract.non_negative_float_fields:
        normalized[field_name] = _validate_non_negative_float(
            normalized.get(field_name),
            field_name,
        )

    for field_name in contract.ratio_fields:
        normalized[field_name] = _validate_ratio(
            normalized.get(field_name),
            field_name,
        )

    for field_name in contract.required_existing_paths:
        _validate_existing_path(normalized.get(field_name), field_name, required=True)

    for field_name in contract.optional_existing_paths:
        _validate_existing_path(normalized.get(field_name), field_name, required=False)

    return normalized


def ensure_local_path_exists_if_pathlike(value: Optional[str], field_name: str) -> None:
    """Fail fast when a path-like field references a local file that does not exist."""
    text = str(value or "").strip()
    if not text:
        return
    if not _looks_like_local_path(text):
        return
    path = Path(text).expanduser()
    if not path.exists():
        raise ValueError(f"{field_name} file does not exist: {text}")


def _validate_required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _validate_positive_int(value: Any, field_name: str, *, allow_none: bool) -> Optional[int]:
    if value is None and allow_none:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be greater than 0")
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return parsed


def _validate_non_negative_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be >= 0")
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return parsed


def _validate_ratio(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be within (0, 1]")
    if parsed <= 0 or parsed > 1:
        raise ValueError(f"{field_name} must be within (0, 1]")
    return parsed


def _validate_existing_path(value: Any, field_name: str, *, required: bool) -> None:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError(f"{field_name} is required")
        return
    if not Path(text).expanduser().exists():
        raise ValueError(f"{field_name} file does not exist: {text}")


def _looks_like_local_path(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    if text.startswith(("/", "./", "../", "~")):
        return True
    if "\\" in text:
        return True
    lower = text.lower()
    if lower.endswith((".jsonl", ".json", ".yaml", ".yml", ".txt", ".csv", ".parquet")):
        return True
    return False
