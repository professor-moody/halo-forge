"""Utility functions and helpers."""

from halo_forge.utils.accelerator import (
    detect_gpu_kind,
    detect_strix_halo,
    empty_accelerator_cache,
    get_device_map,
    get_optimal_config,
    get_torch_device,
    is_accelerator_available,
    is_apple_silicon,
    recommended_attn_impl,
    recommended_dtype,
    seed_accelerator,
    supports_4bit_quantization,
    GPU_KIND_CPU,
    GPU_KIND_CUDA,
    GPU_KIND_MPS,
    GPU_KIND_ROCM_GFX1151,
)
from halo_forge.utils.metrics import MetricsTracker, CycleMetrics, TrainingHistory, TrainingMonitor

__all__ = [
    # Accelerator dispatch
    "detect_gpu_kind",
    "detect_strix_halo",
    "empty_accelerator_cache",
    "get_device_map",
    "get_optimal_config",
    "get_torch_device",
    "is_accelerator_available",
    "is_apple_silicon",
    "recommended_attn_impl",
    "recommended_dtype",
    "seed_accelerator",
    "supports_4bit_quantization",
    "GPU_KIND_CPU",
    "GPU_KIND_CUDA",
    "GPU_KIND_MPS",
    "GPU_KIND_ROCM_GFX1151",
    # Metrics
    "MetricsTracker",
    "CycleMetrics",
    "TrainingHistory",
    "TrainingMonitor",
]

