"""Backend abstraction for halo-forge's multi-accelerator support.

Public API:
    from halo_forge.backend import get_backend, BackendStrategy, LoadSpec

    backend = get_backend()                    # auto-detect
    backend = get_backend("mps")               # explicit
    backend = get_backend(require_training=True)  # raises if backend can't train

    spec = LoadSpec(model_name="Qwen/Qwen2.5-0.5B")
    model = backend.load_causal_lm(spec)
    tokenizer = backend.load_tokenizer(spec)

The registry mirrors the existing VLMAdapter / AudioAdapter pattern in
halo_forge — ABC + name-keyed factory + `get_*` lookup.
"""

from halo_forge.backend.base import (
    BackendCapabilities,
    BackendStrategy,
    BackendUnsupportedError,
    LoadSpec,
)
from halo_forge.backend.registry import (
    BACKENDS,
    detect_backend,
    get_backend,
    reset_registry_cache,
)

__all__ = [
    "BACKENDS",
    "BackendCapabilities",
    "BackendStrategy",
    "BackendUnsupportedError",
    "LoadSpec",
    "detect_backend",
    "get_backend",
    "reset_registry_cache",
]
