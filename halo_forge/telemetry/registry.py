"""Factory: pick the right TelemetryProvider for the active backend.

Caches a single provider instance per process — providers may hold open
file handles or subprocess pipes (rocm-smi caches its parsed JSON) so
re-instantiating on every endpoint hit would defeat the cache.
"""

from __future__ import annotations

from typing import Optional

from halo_forge.telemetry.base import TelemetryProvider, TelemetryUnavailableError


_INSTANCE: Optional[TelemetryProvider] = None


def get_telemetry_provider(force_reset: bool = False) -> TelemetryProvider:
    """Return the singleton telemetry provider for the active backend.

    Routes through `halo_forge.utils.accelerator.detect_gpu_kind()` so the
    answer matches what the trainer code sees at runtime. ROCm/CUDA
    providers can raise TelemetryUnavailableError if their underlying
    binary (rocm-smi / nvidia-smi) isn't on PATH; in that case we fall
    back to the CPU provider so the endpoint still serves *something*.

    Args:
        force_reset: drop the cached instance and re-detect. Test-only.
    """
    global _INSTANCE
    if _INSTANCE is not None and not force_reset:
        return _INSTANCE

    from halo_forge.utils.accelerator import (
        GPU_KIND_CPU,
        GPU_KIND_CUDA,
        GPU_KIND_MPS,
        GPU_KIND_ROCM_GFX1151,
        detect_gpu_kind,
    )

    # The "active backend" depends on whether the user explicitly opted
    # into MLX via HALOFORGE_BACKEND. detect_gpu_kind returns the torch
    # answer (which never includes MLX); we layer the env-var override
    # on top so the strip says "Apple · MLX" when the user asked for it.
    import os

    explicit = (os.environ.get("HALOFORGE_BACKEND") or "").strip().lower()
    detected = detect_gpu_kind()

    if explicit == "mlx":
        from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry

        _INSTANCE = AppleSiliconTelemetry(backend_name="mlx")
        return _INSTANCE

    if detected == GPU_KIND_MPS:
        from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry

        _INSTANCE = AppleSiliconTelemetry(backend_name="mps")
        return _INSTANCE

    if detected == GPU_KIND_ROCM_GFX1151:
        try:
            from halo_forge.telemetry.rocm import ROCmTelemetry

            _INSTANCE = ROCmTelemetry(backend_name="rocm_gfx1151")
            return _INSTANCE
        except TelemetryUnavailableError:
            # Binary missing — fall through to CPU provider so the
            # endpoint still returns useful values.
            pass

    if detected == GPU_KIND_CUDA:
        try:
            from halo_forge.telemetry.cuda import CUDATelemetry

            _INSTANCE = CUDATelemetry(backend_name="cuda")
            return _INSTANCE
        except TelemetryUnavailableError:
            pass

    # Floor: CPU-only provider via psutil
    from halo_forge.telemetry.cpu import CPUTelemetry

    if detected == GPU_KIND_CPU:
        _INSTANCE = CPUTelemetry()
    else:
        # Detected something but its provider failed to bring up — still
        # return CPU so the endpoint isn't dead.
        _INSTANCE = CPUTelemetry()
    return _INSTANCE


def reset_provider_cache() -> None:
    """Test hook: drop the cached singleton."""
    global _INSTANCE
    _INSTANCE = None
