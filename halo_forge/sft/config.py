"""SFT configuration dataclass.

Lives outside `trainer.py` so it can be imported on hosts without torch
(MLX-only Macs, lightweight CI workers). The PyTorch trainer at
`halo_forge.sft.trainer` and the MLX trainer at `halo_forge.sft.mlx_trainer`
both consume this same `SFTConfig`.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED

logger = logging.getLogger(__name__)

# YAML sections `from_yaml` flattens into a single keyword namespace.
_CONFIG_SECTIONS: tuple[str, ...] = ("model", "data", "lora", "qlora", "training")

# Aliases accepted for backward compatibility with the shipped example configs
# and with HuggingFace naming.
_KEY_ALIASES: Dict[str, str] = {
    "name": "model_name",
    "per_device_train_batch_size": "batch_size",
    "num_train_epochs": "num_epochs",
    "r": "lora_r",
    "alpha": "lora_alpha",
    "dropout": "lora_dropout",
}

# Keys the shipped configs carry that this dataclass deliberately does not
# model — the trainer hardcodes the HuggingFace `TrainingArguments` equivalents.
# They are accepted and ignored; everything *else* unknown is a typo, and a typo
# at these defaults costs a full GPU-day of training, so it must fail loudly.
_IGNORED_KEYS: frozenset[str] = frozenset({
    "dataloader_num_workers",
    "dataloader_pin_memory",
    "eval_strategy",
    "evaluation_strategy",
    "load_best_model_at_end",
    "logging_steps",
    "lr_scheduler_type",
    "per_device_eval_batch_size",
    "save_strategy",
})


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    trust_remote_code: bool = True
    # None -> backend default (eager on ROCm/CPU, sdpa on CUDA/MPS).
    # Pass an explicit string to override.
    attn_implementation: Optional[str] = None
    # Apple M5+ Neural Accelerator surfacing. Experimental annotation only;
    # kernels are unchanged and unsupported backends fail at trainer init.
    enable_neural_accelerators: bool = False

    # Data - supports both local files and HuggingFace datasets
    train_file: Optional[str] = None  # Local JSONL file
    validation_file: Optional[str] = None  # Optional immutable validation JSONL
    dataset: Optional[str] = None  # HuggingFace dataset ID or short name
    max_samples: Optional[int] = None  # Limit number of samples
    validation_split: float = 0.05
    max_seq_length: int = 2048

    # QLoRA (4-bit is slower on Strix Halo - use BF16 by default)
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    # Backends without bitsandbytes (Apple Silicon MPS, CPU) cannot honor
    # load_in_4bit. By default the trainer warns and falls back to unquantized;
    # set this False to fail loudly instead.
    allow_quantization_fallback: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # None -> the Qwen default set filled in by __post_init__. Declared as a
    # plain None default (not a default_factory) precisely so the mutable list
    # is constructed per instance rather than shared across configs.
    target_modules: Optional[List[str]] = None
    # PEFT additions (Track T5):
    #   use_dora    — DoRA decomposes the weight into magnitude + direction;
    #                 typically matches LoRA quality at lower rank but
    #                 trains slightly slower.
    #   use_rslora  — Rank-stabilized LoRA scaling (alpha / sqrt(r) instead
    #                 of alpha / r). Free quality win at high rank.
    #   init_lora_weights — "true" (default), "false", "gaussian", "pissa",
    #                 "pissa_niter_[N]", "loftq", "olora". PiSSA initializes
    #                 from SVD of the base weights (faster convergence).
    use_dora: bool = False
    use_rslora: bool = False
    init_lora_weights: str = "true"

    # Optimizer (Track T4) — string forwarded to TrainingArguments.optim.
    # Common values:
    #   "adamw_torch" (default)
    #   "adamw_bnb_8bit" — bitsandbytes 8-bit AdamW; halves optimizer state.
    #   "lion_8bit" / "lion_32bit" — bitsandbytes Lion.
    #   "paged_adamw_8bit" — paged variant for huge models.
    # See transformers.TrainingArguments for the full list.
    optim: str = "adamw_torch"

    # Training
    output_dir: str = "models/sft"
    num_epochs: int = 3
    # Optional absolute optimizer-step ceiling used by resumable experiment
    # segments. ``None`` preserves the traditional epoch-based behavior.
    max_steps: Optional[int] = None
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3

    # Optimization
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = DEFAULT_TRAINING_SEED
    # V21 real-path certification only. Ordinary training avoids copying all
    # trainable tensors to CPU solely to hash them.
    capture_parameter_hashes: bool = False

    # Saving
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 250

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001

    def __post_init__(self):
        if self.max_steps is not None and int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive when provided")
        if self.target_modules is None:
            # Default for Qwen models
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        """Load config from YAML file.

        Unrecognized keys raise `ValueError` rather than being dropped: a typo
        such as ``learnign_rate`` would otherwise train a whole run at the
        default. Known-but-unmodelled HuggingFace keys (see `_IGNORED_KEYS`)
        are accepted and logged, so the shipped example configs keep loading.

        Raises:
            ValueError: the document is not a mapping, it carries top-level keys
                outside `_CONFIG_SECTIONS`, or a section carries keys that are
                neither `SFTConfig` fields nor documented passthroughs.
        """
        import yaml  # lazy: yaml is a hard dep but keep config.py import-clean

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"SFT config {path} must be a YAML mapping, got "
                f"{type(data).__name__}"
            )

        # Only the known sections are read, so a hyperparameter written at the
        # top level is silently discarded — the same full-GPU-day failure the
        # per-key check below prevents. Reject it, and say where it belongs.
        stray = sorted(set(data) - set(_CONFIG_SECTIONS))
        if stray:
            raise ValueError(
                f"SFT config {path}: top-level key(s) "
                f"{', '.join(repr(key) for key in stray)} sit outside every "
                f"recognized section and would be ignored. Move them under one "
                f"of: {', '.join(_CONFIG_SECTIONS)}."
            )

        # Flatten nested config
        flat: Dict[str, Any] = {}
        for section in _CONFIG_SECTIONS:
            values = data.get(section)
            if isinstance(values, dict):
                flat.update(values)
            elif values is not None:
                raise ValueError(
                    f"SFT config {path}: section {section!r} must be a mapping, "
                    f"got {type(values).__name__}"
                )

        mapped: Dict[str, Any] = {}
        ignored: List[str] = []
        unknown: List[str] = []
        for key, value in flat.items():
            resolved = _KEY_ALIASES.get(key, key)
            if resolved in cls.__dataclass_fields__:
                mapped[resolved] = value
            elif key in _IGNORED_KEYS:
                ignored.append(key)
            else:
                unknown.append(key)

        if ignored:
            logger.debug(
                "SFT config %s: ignoring trainer-managed key(s): %s",
                path,
                ", ".join(sorted(ignored)),
            )

        if unknown:
            valid = sorted(set(cls.__dataclass_fields__) | set(_KEY_ALIASES))
            details = []
            for key in sorted(unknown):
                close = difflib.get_close_matches(key, valid, n=1, cutoff=0.6)
                suffix = f" (did you mean {close[0]!r}?)" if close else ""
                details.append(f"{key!r}{suffix}")
            raise ValueError(
                f"Unknown key(s) in SFT config {path}: {', '.join(details)}. "
                f"Valid keys: {', '.join(valid)}"
            )

        return cls(**mapped)


__all__ = ["SFTConfig"]
