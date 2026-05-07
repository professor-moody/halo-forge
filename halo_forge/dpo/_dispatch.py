"""Backend-aware DPO trainer factory.

Phase Q1 / Track T1 ships the PyTorch path; T17 lands MLX-native.
The MLX path supports reference-free DPO (loss_type='sigmoid') in
v1; reference-model DPO and other loss types stay roadmap. Configs
that the MLX trainer can't honor raise NotImplementedError with the
exact knob to flip — the dispatcher itself doesn't pre-check beyond
backend selection.
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
        `DPOTrainer` (PyTorch) — the only path implemented in Q1.

    Raises:
        NotImplementedError if `backend.name == "mlx"`. Phase T17 (MLX
        algorithm parity) will lift this; reference implementation is
        the community mlx-lm-lora fork at
        https://github.com/Goekdeniz-Guelmez/mlx-lm-lora.
    """
    backend = backend or get_backend(require_training=True)

    if backend.name == "mlx":
        from halo_forge.dpo.mlx_trainer import MLXDPOTrainer

        return MLXDPOTrainer(config)

    from halo_forge.dpo.trainer import DPOTrainer

    return DPOTrainer(config)


__all__ = ["get_dpo_trainer"]
