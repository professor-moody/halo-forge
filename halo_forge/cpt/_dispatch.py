"""Backend-aware continued-pretraining trainer factory."""

from __future__ import annotations

from typing import Any, Optional

from halo_forge.backend import BackendStrategy, get_backend
from halo_forge.cpt.config import CPTConfig


def get_cpt_trainer(
    config: CPTConfig,
    *,
    backend: Optional[BackendStrategy] = None,
) -> Any:
    """Return the native trainer for the selected backend."""

    if not isinstance(config, CPTConfig):
        raise TypeError("get_cpt_trainer requires an explicit CPTConfig")
    selected = backend or get_backend(require_training=True)
    if selected.name == "mlx":
        from halo_forge.cpt.mlx_trainer import MLXCPTTrainer

        return MLXCPTTrainer(config)
    from halo_forge.cpt.trainer import CPTTrainer

    return CPTTrainer(config, backend=selected)


__all__ = ["get_cpt_trainer"]
