"""Backend-aware SFT trainer factory.

`cmd_sft_train` calls `get_sft_trainer(config)` and gets back the right
trainer for the active accelerator without having to import either trainer
module up front. Keeps the CLI agnostic and the import graph clean —
mlx_trainer's `import mlx_lm` only happens when MLX is actually selected.

Routing:
    rocm_gfx1151 / cuda / mps / cpu → halo_forge.sft.trainer.SFTTrainer (PyTorch)
    mlx → halo_forge.sft.mlx_trainer.MLXSFTTrainer

The dispatcher uses `halo_forge.backend.get_backend()` for detection so the
choice honors `--accelerator` / `HALOFORGE_BACKEND` exactly the same way
inference does. `require_training=True` ensures we surface a clean error if
someone wires a non-training backend.
"""

from __future__ import annotations

from typing import Any, Optional

from halo_forge.backend import BackendStrategy, get_backend
from halo_forge.sft.config import SFTConfig


def get_sft_trainer(
    config: Optional[SFTConfig] = None,
    *,
    backend: Optional[BackendStrategy] = None,
) -> Any:
    """Return the SFT trainer matching the active backend.

    Args:
        config: SFTConfig forwarded to the trainer's constructor.
        backend: optional explicit backend strategy. Defaults to
            `get_backend(require_training=True)`, which honors the
            `--accelerator` / `HALOFORGE_BACKEND` choice.

    Returns:
        Either `SFTTrainer` (PyTorch) or `MLXSFTTrainer` (MLX). Both expose
        `train(train_file=None, dataset=None, resume_from_checkpoint=None)`
        and return a canonical summary dict, so `cmd_sft_train` does not
        need to know which one it got.
    """
    backend = backend or get_backend(require_training=True)

    if backend.name == "mlx":
        from halo_forge.sft.mlx_trainer import MLXSFTTrainer

        return MLXSFTTrainer(config)

    # PyTorch path covers rocm_gfx1151, cuda, mps, cpu — the underlying
    # SFTTrainer reads device choice from `device_map` / accelerator helpers
    # already wired through Phase 1.
    from halo_forge.sft.trainer import SFTTrainer

    return SFTTrainer(config)


__all__ = ["get_sft_trainer"]
