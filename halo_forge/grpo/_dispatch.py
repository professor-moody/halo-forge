"""Backend-aware GRPO trainer factory.

Mirrors the DPO dispatch shape: PyTorch path covers rocm / cuda / mps /
cpu via TRL's GRPOTrainer; MLX path uses an in-house implementation
gated to reference-free + sigmoid-equivalent variant in v1.
"""

from __future__ import annotations

from typing import Any, Optional

from halo_forge.backend import BackendStrategy, get_backend
from halo_forge.grpo.config import GRPOConfig


def get_grpo_trainer(
    config: Optional[GRPOConfig] = None,
    *,
    backend: Optional[BackendStrategy] = None,
) -> Any:
    """Return the GRPO trainer matching the active backend."""
    backend = backend or get_backend(require_training=True)

    if backend.name == "mlx":
        from halo_forge.grpo.mlx_trainer import MLXGRPOTrainer

        return MLXGRPOTrainer(config)

    from halo_forge.grpo.trainer import GRPOTrainer

    return GRPOTrainer(config)


__all__ = ["get_grpo_trainer"]
