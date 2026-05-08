"""Backend-aware ORPO trainer factory.

Phase Q1 ships the PyTorch path only. MLX-native ORPO is roadmap;
when it lands it will share the same `ORPOConfig` shape.
"""

from __future__ import annotations

from typing import Any, Optional

from halo_forge.backend import BackendStrategy, get_backend
from halo_forge.orpo.config import ORPOConfig


def get_orpo_trainer(
    config: Optional[ORPOConfig] = None,
    *,
    backend: Optional[BackendStrategy] = None,
) -> Any:
    """Return the ORPO trainer matching the active backend.

    Raises:
        NotImplementedError on the MLX backend (roadmap).
    """
    backend = backend or get_backend(require_training=True)

    if backend.name == "mlx":
        raise NotImplementedError(
            "ORPO on MLX is not yet implemented. Use --accelerator cpu "
            "for a slow but functional fallback, or run on a CUDA/ROCm/MPS "
            "host. See halo_forge.dpo.mlx_trainer for the MLX-DPO reference "
            "implementation we'll mirror when this lands."
        )

    from halo_forge.orpo.trainer import ORPOTrainer

    return ORPOTrainer(config)


__all__ = ["get_orpo_trainer"]
