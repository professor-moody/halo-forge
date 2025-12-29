"""
Hardware Detection Utilities

Detect AMD Strix Halo and optimize configuration.
"""

import subprocess
import os
from typing import Dict, Optional, Tuple


def detect_strix_halo() -> Tuple[bool, Dict]:
    """
    Detect if running on AMD Strix Halo (gfx1151).
    
    Returns:
        (is_strix_halo, info_dict)
    """
    info = {
        'detected': False,
        'gpu_name': 'Unknown',
        'total_memory_gb': 0,
        'gfx_version': 'unknown'
    }
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False, info
        
        device_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        info['gpu_name'] = device_name
        info['total_memory_gb'] = round(total_mem, 1)
        
        # Check for Strix Halo
        if 'Radeon' in device_name:
            # Check memory (Strix Halo has ~96GB unified)
            if total_mem > 60:
                info['detected'] = True
                info['gfx_version'] = 'gfx1151'
        
        # Try to get exact gfx version from rocminfo
        try:
            result = subprocess.run(
                ['rocminfo'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if 'gfx1151' in result.stdout:
                info['gfx_version'] = 'gfx1151'
                info['detected'] = True
        except Exception:
            pass
        
        return info['detected'], info
        
    except Exception as e:
        info['error'] = str(e)
        return False, info


def get_optimal_config(total_memory_gb: Optional[float] = None) -> Dict:
    """
    Get optimal training configuration based on available memory.
    
    Args:
        total_memory_gb: Override detected memory
    
    Returns:
        Configuration dict with optimized settings
    """
    if total_memory_gb is None:
        _, info = detect_strix_halo()
        total_memory_gb = info.get('total_memory_gb', 16)
    
    # Base configuration (conservative)
    config = {
        'batch_size': 1,
        'gradient_accumulation_steps': 32,
        'max_seq_length': 1024,
        'gradient_checkpointing': True,
        'quantization': '4bit',
        'lora_r': 8,
        'dataloader_workers': 0
    }
    
    # Optimize based on memory
    if total_memory_gb >= 80:
        # Strix Halo (96GB unified)
        config.update({
            'batch_size': 4,
            'gradient_accumulation_steps': 8,
            'max_seq_length': 2048,
            'lora_r': 16,
        })
    elif total_memory_gb >= 40:
        # High-memory (48GB+)
        config.update({
            'batch_size': 2,
            'gradient_accumulation_steps': 16,
            'max_seq_length': 2048,
            'lora_r': 16,
        })
    elif total_memory_gb >= 24:
        # Mid-range (24GB+)
        config.update({
            'batch_size': 2,
            'gradient_accumulation_steps': 16,
            'max_seq_length': 1536,
            'lora_r': 8,
        })
    # else: use base config
    
    # Strix Halo specific optimizations
    if total_memory_gb >= 80:
        config['rocm_optimizations'] = {
            'attn_implementation': 'eager',  # Required for ROCm
            'dataloader_num_workers': 0,     # Unified memory compat
            'dataloader_pin_memory': False,
            'gradient_checkpointing_kwargs': {'use_reentrant': False}
        }
    
    return config


def print_hardware_info():
    """Print detected hardware information."""
    is_strix, info = detect_strix_halo()
    
    print("=" * 50)
    print("HARDWARE DETECTION")
    print("=" * 50)
    print(f"GPU: {info['gpu_name']}")
    print(f"Memory: {info['total_memory_gb']} GB")
    print(f"GFX Version: {info['gfx_version']}")
    print(f"Strix Halo: {'Yes' if is_strix else 'No'}")
    print()
    
    if is_strix:
        config = get_optimal_config(info['total_memory_gb'])
        print("Recommended configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

