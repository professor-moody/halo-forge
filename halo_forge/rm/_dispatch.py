"""Backend-aware reward-model trainer factory."""

from __future__ import annotations

from typing import Any, Optional

from halo_forge.backend import BackendStrategy, get_backend
from halo_forge.rm.config import RMConfig


def get_rm_trainer(
    config: Optional[RMConfig] = None,
    *,
    backend: Optional[BackendStrategy] = None,
) -> Any:
    """Return the RM trainer matching the active backend."""
    backend = backend or get_backend(require_training=True)

    if backend.name == "mlx":
        raise NotImplementedError(
            "MLX-native reward-model training is on the roadmap. For now "
            "run RM training under --accelerator mps (PyTorch on Apple "
            "Silicon) or use the upstream HF RewardTrainer recipe directly."
        )

    from halo_forge.rm.trainer import RewardModelTrainer

    return RewardModelTrainer(config)


__all__ = ["get_rm_trainer"]
