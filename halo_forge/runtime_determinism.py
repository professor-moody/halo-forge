"""
Shared determinism helpers for training runtime surfaces.
"""

from __future__ import annotations

import os
import random
import re
import time
import uuid
from typing import Optional


DEFAULT_TRAINING_SEED = 42
RUN_ID_ENV = "HALO_FORGE_RUN_ID"
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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


def build_run_id(modality: str, requested: Optional[str] = None) -> str:
    """Resolve the canonical run identifier before any trainer work begins.

    Managed launches pass the identifier through ``HALO_FORGE_RUN_ID`` so the
    UI job, subprocess, summaries, replay evidence, and filesystem all refer
    to the same run. Direct CLI launches retain the historical generated-ID
    behavior, with a short random suffix to avoid millisecond collisions.
    """
    supplied = str(requested or os.environ.get(RUN_ID_ENV) or "").strip()
    if supplied:
        if not _RUN_ID_PATTERN.fullmatch(supplied):
            raise ValueError(
                "run_id must start with an alphanumeric character and contain "
                "only letters, numbers, '.', '_' or '-' (maximum 128 characters)"
            )
        return supplied

    safe_modality = re.sub(r"[^A-Za-z0-9._-]+", "-", str(modality or "run")).strip("-.")
    safe_modality = safe_modality or "run"
    epoch_ms = int(time.time() * 1000)
    return f"{safe_modality}-{epoch_ms}-{uuid.uuid4().hex[:8]}"
