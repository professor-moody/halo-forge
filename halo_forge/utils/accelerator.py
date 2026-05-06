"""
Accelerator detection and dispatch helpers.

Cross-backend abstraction over torch.cuda / torch.backends.mps / cpu so the rest
of the codebase doesn't have to inline platform checks at every model-load and
cache-management call site.

Ported from /Users/keys/projects/malagent/malagent/utils/hardware.py with
halo-forge-specific Strix Halo detection re-exported from .hardware so there is
a single source of truth.
"""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any, Dict, Optional, Tuple


# Accelerator kinds returned by detect_gpu_kind(). Strix Halo is its own kind
# because it needs ROCm-specific env vars and attention impl that no other
# CUDA-class device wants.
GPU_KIND_ROCM_GFX1151 = "rocm_gfx1151"
GPU_KIND_CUDA = "cuda"
GPU_KIND_MPS = "mps"
GPU_KIND_CPU = "cpu"


def detect_gpu_kind() -> str:
    """Return the active accelerator family.

    One of: "rocm_gfx1151", "cuda", "mps", "cpu". Cheap and import-safe — does
    not raise if torch is missing or partially mocked (tests replace `torch`
    with a SimpleNamespace that lacks `backends`); falls back to "cpu".
    """
    try:
        import torch
    except ImportError:
        return GPU_KIND_CPU

    cuda_mod = getattr(torch, "cuda", None)
    if cuda_mod is not None and getattr(cuda_mod, "is_available", lambda: False)():
        is_strix, _ = detect_strix_halo()
        if is_strix:
            return GPU_KIND_ROCM_GFX1151
        return GPU_KIND_CUDA

    backends = getattr(torch, "backends", None)
    mps_backend = getattr(backends, "mps", None) if backends is not None else None
    if mps_backend is not None:
        try:
            if mps_backend.is_available() and mps_backend.is_built():
                return GPU_KIND_MPS
        except Exception:
            pass

    return GPU_KIND_CPU


def is_apple_silicon() -> bool:
    """True on macOS arm64 hosts regardless of whether MPS is built into torch."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def get_torch_device() -> "Any":
    """Return the torch.device the rest of the code should target."""
    import torch

    kind = detect_gpu_kind()
    if kind in (GPU_KIND_CUDA, GPU_KIND_ROCM_GFX1151):
        return torch.device("cuda:0")
    if kind == GPU_KIND_MPS:
        return torch.device("mps:0")
    return torch.device("cpu")


def get_device_map() -> Any:
    """Return a value suitable for `from_pretrained(device_map=...)`.

    On MPS we return an explicit mapping rather than "auto" because accelerate's
    auto-placer historically split Qwen2 modules to CPU silently on transformers
    <4.45. Explicit placement avoids that regression.
    """
    kind = detect_gpu_kind()
    if kind == GPU_KIND_MPS:
        return {"": "mps:0"}
    if kind in (GPU_KIND_CUDA, GPU_KIND_ROCM_GFX1151):
        return "auto"
    return {"": "cpu"}


def recommended_attn_impl() -> str:
    """Best `attn_implementation` for the current device.

    - rocm_gfx1151: "eager" (FA2 is broken on gfx1151; SDPA needs AOTriton flag set)
    - cuda / mps: "sdpa"
    - cpu: "eager"
    """
    kind = detect_gpu_kind()
    if kind == GPU_KIND_ROCM_GFX1151:
        return "eager"
    if kind in (GPU_KIND_CUDA, GPU_KIND_MPS):
        return "sdpa"
    return "eager"


def is_accelerator_available() -> bool:
    """True if any non-CPU accelerator is usable."""
    return detect_gpu_kind() != GPU_KIND_CPU


def empty_accelerator_cache() -> None:
    """Free the active device's cache, if it has one. No-op on CPU."""
    try:
        import torch
    except ImportError:
        return

    kind = detect_gpu_kind()
    if kind in (GPU_KIND_CUDA, GPU_KIND_ROCM_GFX1151):
        torch.cuda.empty_cache()
    elif kind == GPU_KIND_MPS:
        # torch.mps.empty_cache exists on torch >= 2.0
        mps_mod = getattr(torch, "mps", None)
        if mps_mod is not None and hasattr(mps_mod, "empty_cache"):
            mps_mod.empty_cache()


def recommended_dtype() -> "Any":
    """Best `torch.dtype` for the current accelerator.

    - rocm_gfx1151 / cuda: bfloat16 (preferred precision; well-supported on
      ROCm gfx1151 and modern CUDA, large dynamic range avoids fp16 overflow).
    - mps: float16. bfloat16 on MPS is patchy pre-macOS 14 — some ops fall back
      to CPU silently. Callers that know they're on macOS 14+ can override.
    - cpu: float32. fp16/bf16 on CPU are slow and many kernels lack support.

    Returns torch.dtype. Raises if torch is unavailable — call from inside a
    block that has already imported torch (or is willing to fail fast).
    """
    import torch

    kind = detect_gpu_kind()
    if kind in (GPU_KIND_CUDA, GPU_KIND_ROCM_GFX1151):
        return torch.bfloat16
    if kind == GPU_KIND_MPS:
        return torch.float16
    return torch.float32


def supports_4bit_quantization() -> bool:
    """True if `BitsAndBytesConfig(load_in_4bit=True)` will work on this host.

    Upstream bitsandbytes ships CUDA-only kernels by default. ROCm has community
    wheels that work on gfx1151. MPS has no upstream kernels.
    """
    return detect_gpu_kind() in (GPU_KIND_CUDA, GPU_KIND_ROCM_GFX1151)


def seed_accelerator(seed: int) -> None:
    """Seed the active accelerator's RNG. Companion to torch.manual_seed.

    Mirrors the old `if torch.cuda.is_available(): torch.cuda.manual_seed_all`
    pattern but extends it to the MPS backend so determinism settings flow
    through on Mac.
    """
    try:
        import torch
    except ImportError:
        return

    kind = detect_gpu_kind()
    if kind in (GPU_KIND_CUDA, GPU_KIND_ROCM_GFX1151):
        torch.cuda.manual_seed_all(seed)
    elif kind == GPU_KIND_MPS:
        mps_mod = getattr(torch, "mps", None)
        if mps_mod is not None and hasattr(mps_mod, "manual_seed"):
            mps_mod.manual_seed(seed)


def detect_strix_halo() -> Tuple[bool, Dict]:
    """Detect if running on AMD Strix Halo (gfx1151).

    Returns:
        (is_strix_halo, info_dict)
    """
    info = {
        "detected": False,
        "gpu_name": "Unknown",
        "total_memory_gb": 0,
        "gfx_version": "unknown",
    }

    try:
        import torch

        if not torch.cuda.is_available():
            return False, info

        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

        info["gpu_name"] = device_name
        info["total_memory_gb"] = round(total_mem, 1)

        # Strix Halo signature: Radeon GPU with >60GB unified memory
        if "Radeon" in device_name and total_mem > 60:
            info["detected"] = True
            info["gfx_version"] = "gfx1151"

        # Authoritative gfx version from rocminfo if present
        try:
            result = subprocess.run(
                ["rocminfo"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "gfx1151" in result.stdout:
                info["gfx_version"] = "gfx1151"
                info["detected"] = True
        except Exception:
            pass

        return info["detected"], info

    except Exception as e:
        info["error"] = str(e)
        return False, info


def get_optimal_config(total_memory_gb: Optional[float] = None) -> Dict:
    """Get optimal training configuration based on available memory.

    Memory-tier driven. Strix Halo (>=80GB) gets the most generous tunings;
    smaller cards get progressively more conservative defaults.
    """
    if total_memory_gb is None:
        _, info = detect_strix_halo()
        total_memory_gb = info.get("total_memory_gb", 16)

    # Conservative base (works on tiny cards / CPU)
    config: Dict[str, Any] = {
        "batch_size": 1,
        "gradient_accumulation_steps": 32,
        "max_seq_length": 1024,
        "gradient_checkpointing": True,
        "quantization": "4bit",
        "lora_r": 8,
        "dataloader_workers": 0,
    }

    if total_memory_gb >= 80:
        # Strix Halo (96GB+ unified)
        config.update({
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 2048,
            "lora_r": 16,
        })
    elif total_memory_gb >= 40:
        config.update({
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "max_seq_length": 2048,
            "lora_r": 16,
        })
    elif total_memory_gb >= 24:
        config.update({
            "batch_size": 2,
            "gradient_accumulation_steps": 16,
            "max_seq_length": 1536,
            "lora_r": 8,
        })

    if total_memory_gb >= 80:
        config["rocm_optimizations"] = {
            "attn_implementation": "eager",
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }

    return config


def print_hardware_info() -> None:
    """Print detected hardware information."""
    is_strix, info = detect_strix_halo()
    kind = detect_gpu_kind()

    print("=" * 50)
    print("HARDWARE DETECTION")
    print("=" * 50)
    print(f"Accelerator: {kind}")
    print(f"GPU: {info['gpu_name']}")
    print(f"Memory: {info['total_memory_gb']} GB")
    print(f"GFX Version: {info['gfx_version']}")
    print(f"Strix Halo: {'Yes' if is_strix else 'No'}")
    print(f"Recommended attn_implementation: {recommended_attn_impl()}")
    print()

    if is_strix:
        config = get_optimal_config(info["total_memory_gb"])
        print("Recommended configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    print_hardware_info()
