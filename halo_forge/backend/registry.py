"""Backend registry + detection + factory.

Three names for callers to remember:

  - `get_backend(name=None, *, require_training=False)` — the public factory.
    Returns a singleton BackendStrategy. Detection precedence (when name is None):
      1. `HALOFORGE_BACKEND` environment variable
      2. `detect_backend()` heuristic — matches `accelerator.detect_gpu_kind()`

  - `detect_backend()` — returns one of the registered names. Never auto-picks
    "mlx"; MLX is opt-in only since most of the codebase is PyTorch.

  - `BACKENDS` — name -> factory dict. Exposed for tests; production code
    should go through `get_backend`.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

from halo_forge.backend.base import BackendStrategy, BackendUnsupportedError


_INSTANCES: Dict[str, BackendStrategy] = {}


def _make_rocm() -> BackendStrategy:
    from halo_forge.backend.torch_rocm import ROCmBackend

    return ROCmBackend()


def _make_cuda() -> BackendStrategy:
    from halo_forge.backend.torch_cuda import CUDABackend

    return CUDABackend()


def _make_mps() -> BackendStrategy:
    from halo_forge.backend.torch_mps import MPSBackend

    return MPSBackend()


def _make_cpu() -> BackendStrategy:
    from halo_forge.backend.torch_cpu import CPUBackend

    return CPUBackend()


def _make_mlx() -> BackendStrategy:
    from halo_forge.backend.mlx import MLXBackend

    return MLXBackend()


# Public registry. Names match `accelerator.detect_gpu_kind()` so users can
# round-trip between detect_gpu_kind() output and get_backend(name=...).
BACKENDS: Dict[str, Callable[[], BackendStrategy]] = {
    "rocm_gfx1151": _make_rocm,
    # Alias: many callers will say "rocm" and expect Strix-Halo defaults.
    # Both names hand out the same backend instance (see _ALIASES below).
    "rocm": _make_rocm,
    "cuda": _make_cuda,
    "mps": _make_mps,
    "cpu": _make_cpu,
    "mlx": _make_mlx,
}


# Canonical name resolution. Keeps `_INSTANCES` keyed by canonical names so
# `get_backend("rocm")` and `get_backend("rocm_gfx1151")` share a singleton.
_ALIASES: Dict[str, str] = {
    "rocm": "rocm_gfx1151",
}


def detect_backend() -> str:
    """Pick a sensible default backend for the current host.

    Mirrors `halo_forge.utils.accelerator.detect_gpu_kind()` but is the
    backend-side answer (which can differ from the torch-side answer if MLX
    is installed but we still want torch as the default).

    MLX is *never* auto-selected — opt in with HALOFORGE_BACKEND=mlx or
    `--backend mlx`. Most halo-forge code is PyTorch and picking MLX silently
    would break trainers that expect torch tensors.
    """
    from halo_forge.utils.accelerator import detect_gpu_kind

    kind = detect_gpu_kind()
    # accelerator.detect_gpu_kind already returns the right registry name for
    # rocm_gfx1151 / cuda / mps / cpu. MLX is excluded by construction.
    if kind in BACKENDS:
        return kind
    return "cpu"


def get_backend(
    name: Optional[str] = None,
    *,
    require_training: bool = False,
) -> BackendStrategy:
    """Return the backend strategy singleton for `name`.

    Args:
        name: explicit backend name. If None, consult `HALOFORGE_BACKEND` env
            var, then fall back to `detect_backend()`.
        require_training: raise BackendUnsupportedError if the resolved
            backend can't train (e.g. MLX in Phase 3). Trainers should pass
            this so users get a clean error instead of an opaque crash later.
    """
    resolved = (
        name
        or os.environ.get("HALOFORGE_BACKEND")
        or detect_backend()
    )
    resolved = resolved.strip().lower()

    if resolved not in BACKENDS:
        valid = sorted(set(BACKENDS) - set(_ALIASES))  # hide aliases from error msg
        raise ValueError(
            f"Unknown backend {resolved!r}. Valid names: {valid}."
        )

    canonical = _ALIASES.get(resolved, resolved)
    if canonical not in _INSTANCES:
        _INSTANCES[canonical] = BACKENDS[canonical]()
    backend = _INSTANCES[canonical]

    if require_training and not backend.capabilities.supports_training:
        raise BackendUnsupportedError(
            f"Backend {backend.name!r} does not support training in this "
            f"halo-forge version. Use --backend mps (or rocm/cuda) for training."
        )
    return backend


def reset_registry_cache() -> None:
    """Drop the singleton cache. Test-only — production code should not call
    this since backends are stateless beyond their dataclass fields.
    """
    _INSTANCES.clear()


__all__ = [
    "BACKENDS",
    "detect_backend",
    "get_backend",
    "reset_registry_cache",
]
