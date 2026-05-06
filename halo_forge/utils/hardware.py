"""Hardware detection — thin re-exports over halo_forge.utils.accelerator.

The active accelerator-kind helpers (`detect_gpu_kind`, `get_device_map`,
`recommended_attn_impl`, `empty_accelerator_cache`, `supports_4bit_quantization`,
`get_torch_device`, `seed_accelerator`, `is_accelerator_available`,
`is_apple_silicon`) live in `accelerator.py`. This module exists for backwards
compatibility — older code imports `detect_strix_halo` and `get_optimal_config`
from here.
"""

from halo_forge.utils.accelerator import (
    detect_gpu_kind,
    detect_strix_halo,
    empty_accelerator_cache,
    get_device_map,
    get_optimal_config,
    get_torch_device,
    is_accelerator_available,
    is_apple_silicon,
    print_hardware_info,
    recommended_attn_impl,
    recommended_dtype,
    seed_accelerator,
    supports_4bit_quantization,
    GPU_KIND_CPU,
    GPU_KIND_CUDA,
    GPU_KIND_MPS,
    GPU_KIND_ROCM_GFX1151,
)

__all__ = [
    "detect_gpu_kind",
    "detect_strix_halo",
    "empty_accelerator_cache",
    "get_device_map",
    "get_optimal_config",
    "get_torch_device",
    "is_accelerator_available",
    "is_apple_silicon",
    "print_hardware_info",
    "recommended_attn_impl",
    "recommended_dtype",
    "seed_accelerator",
    "supports_4bit_quantization",
    "GPU_KIND_CPU",
    "GPU_KIND_CUDA",
    "GPU_KIND_MPS",
    "GPU_KIND_ROCM_GFX1151",
]
