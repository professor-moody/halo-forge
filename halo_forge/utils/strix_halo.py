"""
AMD Strix Halo Environment Setup

Centralized configuration for AMD Strix Halo optimization.
Applied automatically by TrainingService, VLMEvalKit integration, and CLI commands.
"""

import os
import platform
from typing import Dict, Optional, Tuple


def _is_rocm_host() -> bool:
    """Cheap, import-safe check for a ROCm-capable host.

    ROCm only runs on Linux. We treat any non-Linux host as definitely-not-ROCm
    so the strix env helpers no-op cleanly on macOS/Windows. On Linux, presence
    of /opt/rocm or rocminfo on PATH is the strong signal; either is sufficient.
    """
    if platform.system() != "Linux":
        return False
    if os.path.isdir("/opt/rocm"):
        return True
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    return any(os.path.isfile(os.path.join(d, "rocminfo")) for d in path_dirs if d)


def setup_strix_halo_env(force: bool = False) -> Dict[str, str]:
    """
    Apply AMD Strix Halo environment optimizations.

    This should be called at startup for:
    - TrainingService before launching jobs
    - VLMEvalKit integration before benchmarks
    - CLI commands at startup

    No-op on non-ROCm hosts (macOS, Windows, Linux without ROCm).

    Args:
        force: Force setting even if already set. Still gated by host check;
            pass `force=True` only when you genuinely want to override an
            existing var on a ROCm host.

    Returns:
        Dictionary of environment variables that were set (empty on non-ROCm hosts)
    """
    if not _is_rocm_host():
        return {}

    set_vars = {}

    def set_if_unset(key: str, value: str):
        if force or key not in os.environ:
            os.environ[key] = value
            set_vars[key] = value

    # GPU architecture for Strix Halo (gfx1151)
    set_if_unset('HSA_OVERRIDE_GFX_VERSION', '11.5.1')
    set_if_unset('PYTORCH_ROCM_ARCH', 'gfx1151')
    set_if_unset('HIP_VISIBLE_DEVICES', '0')

    # Memory management for unified memory architecture
    # This is critical for Strix Halo's unified CPU/GPU memory
    set_if_unset(
        'PYTORCH_HIP_ALLOC_CONF',
        'backend:native,expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512'
    )

    # Stability settings
    # SDMA disabled for better stability on ROCm
    set_if_unset('HSA_ENABLE_SDMA', '0')

    # Dataloader settings (critical for unified memory)
    # With unified memory, pin_memory and multiple workers cause issues
    set_if_unset('OMP_NUM_THREADS', '1')

    return set_vars


def get_strix_halo_env() -> Dict[str, str]:
    """
    Get environment variables for Strix Halo optimization.

    Returns a dictionary that can be passed to subprocess.Popen(env=...).
    Includes all current environment; ROCm-specific keys only added on a
    ROCm-capable host (no-op overlay elsewhere so macOS/Windows subprocesses
    inherit a clean env).

    Returns:
        Environment dictionary for subprocess
    """
    env = os.environ.copy()

    if not _is_rocm_host():
        return env

    # GPU architecture
    env.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.5.1')
    env.setdefault('PYTORCH_ROCM_ARCH', 'gfx1151')
    env.setdefault('HIP_VISIBLE_DEVICES', '0')

    # Memory management
    env.setdefault(
        'PYTORCH_HIP_ALLOC_CONF',
        'backend:native,expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512'
    )

    # Stability
    env.setdefault('HSA_ENABLE_SDMA', '0')

    # Dataloader
    env.setdefault('OMP_NUM_THREADS', '1')

    return env


def get_strix_halo_training_config() -> Dict:
    """
    Get optimal training configuration for Strix Halo.
    
    Returns configuration dict suitable for HuggingFace Trainer.
    
    Returns:
        Training configuration dictionary
    """
    return {
        # Use bf16 for best performance on ROCm
        # NOTE: 4-bit quantization is 2x slower on Strix Halo!
        'bf16': True,
        'fp16': False,
        
        # Dataloader settings for unified memory
        'dataloader_num_workers': 0,      # Required for unified memory
        'dataloader_pin_memory': False,   # Required for unified memory
        
        # Memory efficiency
        'gradient_checkpointing': True,
        'gradient_checkpointing_kwargs': {'use_reentrant': False},
        
        # Optimizer
        'optim': 'adamw_torch',  # Native torch optimizer
        
        # Attention implementation
        'attn_implementation': 'eager',  # Required for ROCm stability
    }


def get_strix_halo_inference_config() -> Dict:
    """
    Get optimal inference configuration for Strix Halo.
    
    Returns:
        Inference configuration dictionary
    """
    return {
        'torch_dtype': 'bfloat16',
        'device_map': 'auto',
        'attn_implementation': 'eager',
        'low_cpu_mem_usage': True,
    }


def is_strix_halo() -> Tuple[bool, Dict]:
    """
    Check if running on AMD Strix Halo.
    
    Returns:
        (is_strix_halo, info_dict)
    """
    from .hardware import detect_strix_halo
    return detect_strix_halo()


def print_strix_halo_status():
    """Print Strix Halo detection and configuration status."""
    is_strix, info = is_strix_halo()
    
    print("=" * 60)
    print("AMD STRIX HALO STATUS")
    print("=" * 60)
    print(f"Detected: {'Yes' if is_strix else 'No'}")
    print(f"GPU: {info.get('gpu_name', 'Unknown')}")
    print(f"Memory: {info.get('total_memory_gb', 'N/A')} GB")
    print(f"GFX Version: {info.get('gfx_version', 'Unknown')}")
    print()
    
    if is_strix:
        print("Environment Variables:")
        env = get_strix_halo_env()
        for key in ['HSA_OVERRIDE_GFX_VERSION', 'PYTORCH_ROCM_ARCH', 'PYTORCH_HIP_ALLOC_CONF']:
            print(f"  {key}={env.get(key, 'not set')}")
        print()
        
        print("Training Config:")
        config = get_strix_halo_training_config()
        for key, value in config.items():
            print(f"  {key}: {value}")
    print("=" * 60)
