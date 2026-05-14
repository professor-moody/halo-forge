"""Backend-aware DPO trainer factory.

Phase Q1 / Track T1 shipped the PyTorch path; T17 lands MLX-native.
The MLX path supports sigmoid, IPO, hinge, and KTO-pair DPO in
reference-free and reference-model modes. Configs that the MLX trainer
can't honor raise NotImplementedError with the exact knob to flip — the
dispatcher itself doesn't pre-check beyond backend selection.
"""

from __future__ import annotations

from typing import Any, Optional

from halo_forge.backend import BackendStrategy, get_backend
from halo_forge.dpo.config import DPOConfig


def get_dpo_trainer(
    config: Optional[DPOConfig] = None,
    *,
    backend: Optional[BackendStrategy] = None,
) -> Any:
    """Return the DPO trainer matching the active backend.

    Args:
        config: DPOConfig forwarded to the trainer's constructor.
        backend: optional explicit backend strategy. Defaults to
            `get_backend(require_training=True)`, which honors the
            `--accelerator` / `HALOFORGE_BACKEND` choice.

    Returns:
        `MLXDPOTrainer` on the MLX backend, otherwise the PyTorch
        `DPOTrainer`.
    """
    backend = backend or get_backend(require_training=True)

    if backend.name == "mlx":
        from halo_forge.dpo.mlx_trainer import MLXDPOTrainer

        return MLXDPOTrainer(config)

    from halo_forge.dpo.trainer import DPOTrainer

    return DPOTrainer(config)


__all__ = ["get_dpo_trainer"]
