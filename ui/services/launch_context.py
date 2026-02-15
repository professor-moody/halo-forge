"""
Durable launch context persistence for UI job relaunch operations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


LAUNCH_CONTEXT_FILENAME = "launch_context.json"
LAUNCH_CONTEXT_CONTRACT_VERSION = 1

SUPPORTED_JOB_TYPES = {
    "sft",
    "raft",
    "benchmark",
    "inference",
    "vlm",
    "audio",
    "reasoning",
    "agentic",
    "config",
    "data",
    "info",
    "plot",
}

CYCLE_BASED_TRAINING_JOB_TYPES = {"raft", "vlm", "audio", "reasoning", "agentic"}

_NORMALIZED_ARG_KEYS: Dict[str, tuple[str, ...]] = {
    "sft": (
        "model",
        "dataset",
        "output_dir",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "warmup_ratio",
        "weight_decay",
        "max_grad_norm",
        "use_lora",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "max_seq_length",
        "validation_split",
        "max_samples",
        "save_steps",
        "eval_steps",
        "early_stopping_patience",
        "gradient_checkpointing",
    ),
    "raft": (
        "model",
        "prompts",
        "output_dir",
        "verifier",
        "cycles",
        "samples_per_prompt",
        "temperature",
        "keep_percent",
        "reward_threshold",
        "min_samples",
        "max_new_tokens",
        "lr_decay",
        "min_lr",
        "checkpoint",
        "curriculum",
        "curriculum_stats",
        "curriculum_start",
        "curriculum_increment",
        "reward_shaping",
        "system_prompt",
        "experimental_attention",
    ),
    "vlm": (
        "model",
        "dataset",
        "output_dir",
        "cycles",
        "learning_rate",
        "lr_decay",
        "samples_per_prompt",
        "temperature",
        "keep_percent",
        "reward_threshold",
        "resume_from_cycle",
        "seed",
        "allow_prototype_train",
    ),
    "audio": (
        "model",
        "dataset",
        "output_dir",
        "cycles",
        "learning_rate",
        "lr_decay",
        "samples_per_prompt",
        "temperature",
        "keep_percent",
        "reward_threshold",
        "task",
        "resume_from_cycle",
        "seed",
        "allow_prototype_train",
    ),
    "reasoning": (
        "model",
        "dataset",
        "output_dir",
        "cycles",
        "learning_rate",
        "lr_decay",
        "limit",
        "resume_from_cycle",
        "seed",
        "allow_prototype_train",
    ),
    "agentic": (
        "model",
        "dataset",
        "output_dir",
        "cycles",
        "learning_rate",
        "lr_decay",
        "limit",
        "resume_from_cycle",
        "seed",
        "allow_prototype_train",
    ),
    "benchmark": (
        "model",
        "benchmark_type",
        "benchmark_name",
        "limit",
        "output_path",
        "output_dir",
        "samples_per_prompt",
        "verifier",
        "run_after_compile",
        "task",
    ),
    "inference": (
        "mode",
        "model",
        "output_dir",
        "target_precision",
        "target_latency",
        "calibration_data",
        "dry_run",
        "prompts",
        "num_prompts",
        "max_tokens",
        "warmup",
        "measure_memory",
    ),
    "config": (
        "module",
        "execution_mode",
        "output_root",
        "config_path",
        "config_type",
        "verbose",
    ),
    "data": (
        "module",
        "execution_mode",
        "output_root",
        "data_action",
        "data_file",
        "dataset",
        "topic",
        "backend",
        "backend_model",
        "data_output",
        "template",
        "system_prompt",
    ),
    "info": (
        "module",
        "execution_mode",
        "output_root",
    ),
    "plot": (
        "module",
        "execution_mode",
        "output_root",
        "plot_action",
        "plot_input",
        "plot_output",
        "plot_compare",
    ),
}


@dataclass(frozen=True)
class LaunchContextV1:
    """Versioned launch context schema persisted per run output directory."""

    contract_version: int
    job_type: str
    service: str
    created_at: str
    source_ui_page: str
    command: list[str]
    args: Dict[str, Any]
    relaunch_capabilities: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "job_type": self.job_type,
            "service": self.service,
            "created_at": self.created_at,
            "source_ui_page": self.source_ui_page,
            "command": list(self.command),
            "args": dict(self.args),
            "relaunch_capabilities": dict(self.relaunch_capabilities),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "LaunchContextV1":
        if int(data.get("contract_version", 0)) != LAUNCH_CONTEXT_CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported launch context contract_version: {data.get('contract_version')}"
            )

        job_type = str(data.get("job_type") or "").strip().lower()
        if job_type not in SUPPORTED_JOB_TYPES:
            raise ValueError(f"Unsupported launch context job_type: {job_type}")

        service = str(data.get("service") or "").strip().lower()
        if service not in {"training", "benchmark", "inference", "module_ops"}:
            raise ValueError(f"Unsupported launch context service: {service}")

        command = data.get("command")
        if not isinstance(command, list) or not all(isinstance(v, str) for v in command):
            raise ValueError("launch_context command must be list[str]")

        args = data.get("args")
        if not isinstance(args, dict):
            raise ValueError("launch_context args must be an object")

        relaunch_capabilities = data.get("relaunch_capabilities")
        if not isinstance(relaunch_capabilities, dict):
            raise ValueError("launch_context relaunch_capabilities must be an object")

        return LaunchContextV1(
            contract_version=LAUNCH_CONTEXT_CONTRACT_VERSION,
            job_type=job_type,
            service=service,
            created_at=str(data.get("created_at") or datetime.now().isoformat()),
            source_ui_page=str(data.get("source_ui_page") or ""),
            command=command,
            args=dict(args),
            relaunch_capabilities={
                "can_relaunch": bool(relaunch_capabilities.get("can_relaunch", True)),
                "can_clone": bool(relaunch_capabilities.get("can_clone", True)),
                "can_resume_latest": bool(
                    relaunch_capabilities.get("can_resume_latest", False)
                ),
            },
        )


def normalize_launch_args(job_type: str, args: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize and filter launch args to a stable per-job-type schema."""
    allowed = _NORMALIZED_ARG_KEYS.get(job_type, ())
    normalized: Dict[str, Any] = {}
    for key in allowed:
        if key not in args:
            continue
        value = args.get(key)
        if value is None:
            continue
        if isinstance(value, Path):
            normalized[key] = str(value)
        elif hasattr(value, "value") and isinstance(getattr(value, "value"), str):
            normalized[key] = value.value
        else:
            normalized[key] = value
    return normalized


def launch_context_path_for_output_dir(output_dir: Optional[Path | str]) -> Optional[Path]:
    """Return canonical launch context path for an output dir."""
    if not output_dir:
        return None
    path = Path(output_dir)
    return path / LAUNCH_CONTEXT_FILENAME


def persist_launch_context(
    *,
    output_dir: Optional[Path | str],
    job_type: str,
    service: str,
    source_ui_page: str,
    command: list[str],
    args: Mapping[str, Any],
    relaunch_capabilities: Mapping[str, bool],
) -> Optional[Path]:
    """Persist launch context atomically; returns path when successful."""
    context_path = launch_context_path_for_output_dir(output_dir)
    if context_path is None:
        return None

    context_path.parent.mkdir(parents=True, exist_ok=True)
    payload = LaunchContextV1(
        contract_version=LAUNCH_CONTEXT_CONTRACT_VERSION,
        job_type=job_type,
        service=service,
        created_at=datetime.now().isoformat(),
        source_ui_page=source_ui_page,
        command=list(command),
        args=normalize_launch_args(job_type, args),
        relaunch_capabilities={
            "can_relaunch": bool(relaunch_capabilities.get("can_relaunch", True)),
            "can_clone": bool(relaunch_capabilities.get("can_clone", True)),
            "can_resume_latest": bool(
                relaunch_capabilities.get("can_resume_latest", False)
            ),
        },
    ).to_dict()

    temp_path = context_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(context_path)
    return context_path


def read_launch_context(path: Path | str) -> LaunchContextV1:
    """Read and validate persisted launch context."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("launch_context.json must contain an object")
    return LaunchContextV1.from_dict(raw)


def find_latest_launch_context(
    *,
    root: Path | str,
    job_type: str,
    service: Optional[str] = None,
) -> Optional[Path]:
    """
    Find most recent launch_context.json for a given job type under a root.

    Returns:
        Path to launch_context.json if found and valid, otherwise None.
    """
    search_root = Path(root)
    if not search_root.exists():
        return None

    job_key = str(job_type or "").strip().lower()
    if job_key not in SUPPORTED_JOB_TYPES:
        return None
    service_key = str(service or "").strip().lower() if service else None

    candidates: list[Path] = []
    for context_path in search_root.glob(f"**/{LAUNCH_CONTEXT_FILENAME}"):
        try:
            context = read_launch_context(context_path)
        except Exception:
            continue
        if context.job_type != job_key:
            continue
        if service_key and context.service != service_key:
            continue
        candidates.append(context_path)

    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)
