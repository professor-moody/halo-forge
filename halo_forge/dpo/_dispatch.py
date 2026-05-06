"""Backend-aware DPO trainer factory.

Phase Q1 / Track T1 ships the PyTorch path. The MLX path is stubbed —
when a user runs DPO with `--accelerator mlx` we surface a clean error
pointing at the mlx-lm-lora fork until phase Q3 / T17 lands a native
implementation.
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
        raise NotImplementedError(
            "MLX-native DPO is on the roadmap (T17, phase Q3) but not yet "
            "implemented. For now run DPO under --accelerator mps (PyTorch on "
            "Apple Silicon) or use the community fork: "
            "https://github.com/Goekdeniz-Guelmez/mlx-lm-lora"
        )

    from halo_forge.dpo.trainer import DPOTrainer

    return DPOTrainer(config)


__all__ = ["get_dpo_trainer"]
