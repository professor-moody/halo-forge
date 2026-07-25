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

A second class of mismatch is handled here too: *training-critical* knobs
(``gradient_accumulation_steps``, ``max_steps``, ``max_grad_norm``,
``warmup_ratio``). Those change the optimization itself rather than a LoRA
detail, so instead of a warning the MLX trainers use the helpers below to
actually honor them against the installed mlx / mlx-lm — and fail loudly
when the installed version cannot (see ``MLXUnsupportedConfigError``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

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


# ----------------------------------------------------------------------
# Training-critical knobs (Track T21)
# ----------------------------------------------------------------------


class MLXUnsupportedConfigError(ValueError):
    """Raised when MLX cannot honor a knob that changes the optimization.

    Distinct from the warn-only fields above: ignoring
    ``gradient_accumulation_steps`` silently divides the effective batch
    size (and multiplies the optimizer-step count) by the configured
    value, so the trainers refuse to launch rather than diverge quietly.
    """


@dataclass(frozen=True)
class MLXCapabilities:
    """What the *installed* mlx / mlx-lm can actually honor.

    Feature-detected rather than version-sniffed: the trainers care about
    the presence of ``TrainingArgs.grad_accumulation_steps`` and of the
    ``mlx.optimizers`` schedule / clipping helpers, not about a version
    string. Defaults are all-False so a caller that detected nothing gets
    the conservative answer.
    """

    grad_accumulation: bool = False
    grad_accumulation_kwarg: Optional[str] = None
    lr_schedule: bool = False
    grad_clipping: bool = False


# mlx-lm names the field `grad_accumulation_steps`; accept the fuller
# spelling too in case upstream renames it.
_MLX_GRAD_ACCUM_KWARGS: tuple[str, ...] = (
    "grad_accumulation_steps",
    "gradient_accumulation_steps",
)
_MLX_SCHEDULE_FUNCS: tuple[str, ...] = ("linear_schedule", "cosine_decay", "join_schedules")

# Escape hatch for the grad-accumulation hard error. Set to run anyway on an
# mlx-lm that cannot accumulate — the run then records the *effective*
# value (1), never the requested one.
_MLX_GRAD_ACCUM_OVERRIDE_ENV = "HALOFORGE_MLX_ALLOW_UNSUPPORTED_GRAD_ACCUM"


# Field specs in the `_MLX_UNSUPPORTED_SFT_FIELDS` shape, keyed by the
# capability that would otherwise honor them. These are fed to
# `warn_unsupported_for_mlx` only for the capabilities the installed stack
# is missing — on a supported mlx-lm every one of these is applied for
# real, so warning unconditionally would be its own kind of lie.
_MLX_TRAINING_KNOB_FIELDS: dict[str, tuple[str, Any, str]] = {
    "grad_accumulation": (
        "gradient_accumulation_steps",
        1,
        "The installed mlx-lm has no gradient accumulation; the run used "
        "1 (effective batch size = batch_size). Upgrade mlx-lm or lower "
        "--gradient-accumulation to 1 to match what actually ran.",
    ),
    "lr_schedule": (
        "warmup_ratio",
        0.0,
        "LR warmup needs mlx.optimizers.linear_schedule / join_schedules, "
        "which the installed mlx does not expose; the run used a constant "
        "learning rate.",
    ),
    "grad_clipping": (
        "max_grad_norm",
        0.0,
        "Gradient-norm clipping needs mlx.optimizers.clip_grad_norm, which "
        "the installed mlx does not expose; gradients were not clipped.",
    ),
}


def mlx_grad_accumulation_kwarg(training_args_cls: Any) -> Optional[str]:
    """Return the ``TrainingArgs`` field name for gradient accumulation.

    Returns None when the installed ``mlx_lm.tuner.trainer.TrainingArgs``
    has no such field (older mlx-lm), which is what makes accumulation
    unsupportable on that install.
    """
    fields = getattr(training_args_cls, "__dataclass_fields__", None) or {}
    for name in _MLX_GRAD_ACCUM_KWARGS:
        if name in fields:
            return name
    return None


def detect_mlx_capabilities(
    *,
    training_args_cls: Any = None,
    optim_module: Any = None,
) -> MLXCapabilities:
    """Feature-detect what the installed MLX stack can honor.

    Args:
        training_args_cls: ``mlx_lm.tuner.trainer.TrainingArgs`` (or None
            when the caller owns its own training loop and therefore
            implements accumulation itself).
        optim_module: the ``mlx.optimizers`` module.
    """
    kwarg = (
        mlx_grad_accumulation_kwarg(training_args_cls)
        if training_args_cls is not None
        else None
    )
    return MLXCapabilities(
        grad_accumulation=kwarg is not None,
        grad_accumulation_kwarg=kwarg,
        lr_schedule=(
            optim_module is not None
            and all(hasattr(optim_module, name) for name in _MLX_SCHEDULE_FUNCS)
        ),
        grad_clipping=optim_module is not None and hasattr(optim_module, "clip_grad_norm"),
    )


def mlx_training_knob_gaps(
    capabilities: MLXCapabilities,
) -> tuple[tuple[str, Any, str], ...]:
    """Field specs for training knobs the installed MLX stack can't honor.

    Feed the result to `warn_unsupported_for_mlx(cfg, fields=...)` so the
    gaps reach the user through the same guard as the PEFT fields.
    """
    return tuple(
        spec
        for capability, spec in _MLX_TRAINING_KNOB_FIELDS.items()
        if not getattr(capabilities, capability, False)
    )


def resolve_mlx_gradient_accumulation(
    config: Any,
    capabilities: MLXCapabilities,
    *,
    trainer_label: str,
) -> int:
    """Return the accumulation factor that will actually be applied.

    Raises:
        MLXUnsupportedConfigError: when the config asks for accumulation
            the installed mlx-lm cannot perform. Running anyway would cut
            the effective batch size by that factor without telling anyone,
            which is the failure mode this whole module exists to prevent.
    """
    requested = int(getattr(config, "gradient_accumulation_steps", 1) or 1)
    if requested < 1:
        raise MLXUnsupportedConfigError(
            f"[{trainer_label}] gradient_accumulation_steps must be >= 1; got {requested}."
        )
    if requested == 1 or capabilities.grad_accumulation:
        return requested
    if os.environ.get(_MLX_GRAD_ACCUM_OVERRIDE_ENV, "").strip().lower() in {"1", "true", "yes"}:
        line = (
            f"[{trainer_label}] gradient_accumulation_steps={requested} cannot be "
            "honored by the installed mlx-lm; continuing with 1 because "
            f"{_MLX_GRAD_ACCUM_OVERRIDE_ENV} is set. Effective batch size is "
            f"batch_size, not batch_size x {requested}."
        )
        logger.warning(line)
        print(f"WARNING: {line}")
        return 1
    raise MLXUnsupportedConfigError(
        f"[{trainer_label}] gradient_accumulation_steps={requested} is not supported "
        "by the installed mlx-lm (TrainingArgs has no gradient-accumulation field). "
        "Running anyway would train at 1/"
        f"{requested} of the effective batch size you configured. Upgrade mlx-lm "
        "(>=0.31), pass --gradient-accumulation 1, or set "
        f"{_MLX_GRAD_ACCUM_OVERRIDE_ENV}=1 to accept the 1-step fallback."
    )


def resolve_mlx_step_budget(
    *,
    micro_batches_per_epoch: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
    max_steps: Optional[int] = None,
) -> tuple[int, int]:
    """Convert epochs / ``max_steps`` into an MLX step budget.

    mlx-lm's ``TrainingArgs.iters`` counts *micro* batches and applies the
    optimizer every ``grad_accumulation_steps`` of them, so a budget that
    isn't a whole multiple silently discards the trailing accumulation.

    Args:
        micro_batches_per_epoch: batches of ``batch_size`` per epoch.
        num_epochs: configured epochs.
        gradient_accumulation_steps: accumulation factor actually applied.
        max_steps: optional ceiling in *optimizer* steps (same meaning as
            ``transformers.TrainingArguments.max_steps``).

    Returns:
        ``(micro_batch_iters, optimizer_steps)``.
    """
    accum = max(1, int(gradient_accumulation_steps or 1))
    total_micro = max(1, int(micro_batches_per_epoch)) * max(1, int(num_epochs))
    optimizer_steps = max(1, total_micro // accum)
    if max_steps is not None:
        optimizer_steps = max(1, min(optimizer_steps, int(max_steps)))
    return optimizer_steps * accum, optimizer_steps


def resolve_mlx_warmup_steps(
    config: Any,
    capabilities: MLXCapabilities,
    *,
    optimizer_steps: int,
) -> int:
    """Return the warmup length in optimizer steps (0 when not applicable)."""
    if not capabilities.lr_schedule:
        return 0
    ratio = float(getattr(config, "warmup_ratio", 0.0) or 0.0)
    if ratio <= 0 or optimizer_steps < 2:
        return 0
    return max(1, min(int(round(ratio * optimizer_steps)), optimizer_steps - 1))


def build_mlx_lr_schedule(
    optim_module: Any,
    *,
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> Any:
    """Return a warmup→cosine LR schedule, or the constant learning rate.

    Mirrors the PyTorch path's ``warmup_ratio`` + ``lr_scheduler_type="cosine"``
    so the two backends anneal the same way. Callers pass the result
    straight to an ``mlx.optimizers`` constructor, which accepts either a
    float or a callable schedule.
    """
    if warmup_steps <= 0 or total_steps <= warmup_steps:
        return learning_rate
    warmup = optim_module.linear_schedule(0.0, learning_rate, warmup_steps)
    decay = optim_module.cosine_decay(learning_rate, total_steps - warmup_steps)
    return optim_module.join_schedules([warmup, decay], [warmup_steps])


def build_mlx_optimizer(
    optim_module: Any,
    *,
    learning_rate: Any,
    weight_decay: float,
    max_grad_norm: Optional[float] = None,
) -> Any:
    """Construct the MLX AdamW, clipping gradients when configured.

    mlx-lm's trainer owns the update call, so global-norm clipping is
    installed by overriding ``update`` on an AdamW subclass — that keeps
    every mlx-lm internal (``optimizer.state``, ``learning_rate``,
    ``mx.compile`` tracing) working on a real optimizer instance instead
    of a proxy object.
    """
    adamw = optim_module.AdamW
    clip: Optional[Callable[[Any, float], Any]] = getattr(optim_module, "clip_grad_norm", None)
    norm = float(max_grad_norm or 0.0)
    if norm <= 0 or clip is None:
        return adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    class _ClippedAdamW(adamw):  # type: ignore[misc, valid-type]
        """AdamW that clips the global gradient norm before applying it."""

        def update(self, model: Any, gradients: Any) -> Any:
            clipped, _total_norm = clip(gradients, norm)
            return super().update(model, clipped)

    _ClippedAdamW.__name__ = "ClippedAdamW"
    _ClippedAdamW.__qualname__ = "ClippedAdamW"
    return _ClippedAdamW(learning_rate=learning_rate, weight_decay=weight_decay)


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
    "MLXCapabilities",
    "MLXUnsupportedConfigError",
    "build_mlx_lr_schedule",
    "build_mlx_optimizer",
    "detect_mlx_capabilities",
    "mlx_grad_accumulation_kwarg",
    "mlx_training_knob_gaps",
    "resolve_mlx_gradient_accumulation",
    "resolve_mlx_step_budget",
    "resolve_mlx_warmup_steps",
    "warn_unsupported_for_mlx",
    "warn_experimental_vllm_backend",
]
