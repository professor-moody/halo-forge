"""
Shared determinism helpers for training runtime surfaces.
"""

from __future__ import annotations

import os
import random
import time
from typing import Optional


DEFAULT_TRAINING_SEED = 42


def normalize_seed(seed: Optional[int]) -> int:
    """Normalize and validate seed inputs."""
    if seed is None:
        return DEFAULT_TRAINING_SEED
    try:
        value = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer >= 0") from exc
    if value < 0:
        raise ValueError("seed must be >= 0")
    return value


def set_global_seed(seed: Optional[int]) -> int:
    """
    Set deterministic global RNG state across available libraries.

    Returns the normalized seed value that was applied.
    """
    normalized = normalize_seed(seed)
    random.seed(normalized)
    os.environ["PYTHONHASHSEED"] = str(normalized)

    try:
        import numpy as np

        np.random.seed(normalized)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(normalized)
        # Accelerator-side RNG: routes through halo_forge.utils.accelerator so
        # MPS hosts get torch.mps.manual_seed and ROCm/CUDA hosts get the
        # legacy torch.cuda.manual_seed_all branch. Best-effort — caller
        # already wraps in try/except.
        from halo_forge.utils.accelerator import seed_accelerator

        seed_accelerator(normalized)
    except Exception:
        pass

    # MLX has an independent RNG. Seed it through the backend abstraction so
    # runtime_determinism does not import optional MLX packages directly.
    try:
        from halo_forge.backend import get_backend

        backend = get_backend()
        if backend.name == "mlx":
            backend.seed_all(normalized)
    except Exception:
        pass

    return normalized


def build_run_id(modality: str) -> str:
    """Generate a lightweight deterministic-run identifier."""
    epoch_ms = int(time.time() * 1000)
    return f"{modality}-{epoch_ms}"
