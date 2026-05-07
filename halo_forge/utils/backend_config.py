"""Backend-specific config validation.

Backends don't all support the same set of training-config knobs. The
PyTorch trainer reads every field on ``SFTConfig`` / ``DPOConfig`` and
the underlying transformers / peft / bitsandbytes machinery honors them.
The MLX trainer wraps ``mlx_lm.tuner`` and ignores anything peft-specific
or bnb-specific. Without explicit validation, a user passing
``--use-dora`` on an MLX host gets vanilla LoRA *silently* — the run
finishes, the diff vs LoRA is invisible, and "I asked for DoRA"
becomes a confused-user issue.

This module surfaces those mismatches loudly at trainer-init time so
the user knows which knobs the backend they picked actually honors.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


# Each entry: (attribute name on config, sentinel "default" value, friendly
# description of what the user lost). The MLX SFT trainer can't honor
# any of these — they need peft / bitsandbytes / transformers paths
# that don't exist on Apple Silicon.
_MLX_UNSUPPORTED_SFT_FIELDS: tuple[tuple[str, Any, str], ...] = (
    (
        "use_dora",
        False,
        "DoRA decomposition is peft-only; MLX uses vanilla LoRA. "
        "Run on a torch backend (cuda/rocm/mps/cpu) to honor --use-dora.",
    ),
    (
        "use_rslora",
        False,
        "Rank-stabilized LoRA scaling lives in peft; MLX uses standard alpha/r scaling.",
    ),
    (
        "init_lora_weights",
        "true",
        "Custom LoRA init (pissa / loftq / olora / gaussian) is peft-only; "
        "MLX uses Kaiming init (the LoraConfig default).",
    ),
    (
        "optim",
        "adamw_torch",
        "Optimizer choice is forwarded to transformers.TrainingArguments; "
        "MLX uses mlx.optimizers.AdamW. Bitsandbytes-backed optimizers "
        "(adamw_bnb_8bit, lion_8bit, paged_adamw_*) require CUDA or ROCm.",
    ),
)


def warn_unsupported_for_mlx(
    config: Any,
    *,
    trainer_label: str = "MLX SFT",
    fields: Sequence[tuple[str, Any, str]] = _MLX_UNSUPPORTED_SFT_FIELDS,
) -> list[str]:
    """Surface a warning for every config field MLX can't honor.

    Args:
        config: Trainer config (SFTConfig / DPOConfig / similar).
        trainer_label: Used in the log line so multi-trainer projects
            can tell the source apart.
        fields: Override the default field set. Tests use this to keep
            the surface stable when new fields land.

    Returns:
        List of the warning lines emitted, for tests / structured
        consumers (e.g. the public-API preflight surface) to inspect.
    """
    emitted: list[str] = []
    for attr, default, advice in fields:
        actual = getattr(config, attr, default)
        if actual == default:
            continue
        line = (
            f"[{trainer_label}] Config field {attr!r}={actual!r} is "
            f"not honored on the MLX backend. {advice}"
        )
        logger.warning(line)
        # Also print so users in interactive CLI sessions don't miss it.
        # Logging alone reaches log files but is silent at the terminal
        # without an explicit handler config.
        print(f"WARNING: {line}")
        emitted.append(line)
    return emitted


def warn_experimental_vllm_backend(backend_name: str) -> bool:
    """Print an experimental-status warning for vLLM on Strix Halo.

    Returns True if a warning was emitted (the test harness inspects
    this to assert visibility).
    """
    if backend_name == "rocm_gfx1151":
        line = (
            "[vllm] Strix Halo (gfx1151) vLLM support is experimental. "
            "If rollouts hang or crash, fall back to "
            "--rollout-engine torch or --rollout-engine mlx."
        )
        logger.warning(line)
        print(f"WARNING: {line}")
        return True
    return False


__all__ = [
    "warn_unsupported_for_mlx",
    "warn_experimental_vllm_backend",
]
