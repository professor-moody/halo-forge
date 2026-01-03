"""
GPU metrics monitoring for AMD ROCm GPUs.

Tries rocm-smi first (works without sudo on most setups),
falls back gracefully to show N/A when unavailable.
"""

import subprocess
import json
import re
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class GPUMetrics:
    """GPU metrics data."""
    utilization: Optional[float] = None  # 0-100%
    memory_used: Optional[float] = None  # GB
    memory_total: Optional[float] = None  # GB
    memory_percent: Optional[float] = None  # 0-100%
    temperature: Optional[float] = None  # Celsius
    power_draw: Optional[float] = None  # Watts
    
    @property
    def available(self) -> bool:
        """Check if any metrics are available."""
        return any([
            self.utilization is not None,
            self.memory_used is not None,
            self.temperature is not None
        ])


def get_gpu_metrics() -> GPUMetrics:
    """
    Get GPU metrics from rocm-smi.
    
    Returns:
        GPUMetrics with available values, or all None if unavailable
    """
    metrics = GPUMetrics()
    
    # Try rocm-smi with JSON output first
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--json"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and result.stdout.strip():
            return _parse_rocm_smi_json(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
        pass
    
    # Fallback: Try rocm-smi without JSON
    try:
        result = subprocess.run(
            ["rocm-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return _parse_rocm_smi_text(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Fallback: Try nvidia-smi (for NVIDIA GPUs)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return _parse_nvidia_smi(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return metrics


def _parse_rocm_smi_json(output: str) -> GPUMetrics:
    """Parse rocm-smi JSON output."""
    metrics = GPUMetrics()
    
    try:
        data = json.loads(output)
        
        # Find first GPU (card0 or similar)
        gpu_key = None
        for key in data:
            if key.startswith("card"):
                gpu_key = key
                break
        
        if not gpu_key:
            return metrics
        
        gpu = data[gpu_key]
        
        # GPU utilization
        if "GPU use (%)" in gpu:
            metrics.utilization = float(gpu["GPU use (%)"])
        elif "GPU Usage" in gpu:
            val = gpu["GPU Usage"]
            if isinstance(val, str) and "%" in val:
                metrics.utilization = float(val.replace("%", ""))
        
        # Memory
        if "VRAM Total Memory (B)" in gpu:
            total_bytes = int(gpu["VRAM Total Memory (B)"])
            metrics.memory_total = total_bytes / (1024**3)  # Convert to GB
        if "VRAM Total Used Memory (B)" in gpu:
            used_bytes = int(gpu["VRAM Total Used Memory (B)"])
            metrics.memory_used = used_bytes / (1024**3)
        
        if metrics.memory_total and metrics.memory_used:
            metrics.memory_percent = (metrics.memory_used / metrics.memory_total) * 100
        
        # Temperature
        for temp_key in ["Temperature (Sensor edge) (C)", "Temperature (Sensor junction) (C)", "Temperature"]:
            if temp_key in gpu:
                val = gpu[temp_key]
                if isinstance(val, (int, float)):
                    metrics.temperature = float(val)
                elif isinstance(val, str):
                    # Extract number from string like "65.0 C"
                    match = re.search(r"([\d.]+)", val)
                    if match:
                        metrics.temperature = float(match.group(1))
                break
        
        # Power
        for power_key in ["Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)", "Power"]:
            if power_key in gpu:
                val = gpu[power_key]
                if isinstance(val, (int, float)):
                    metrics.power_draw = float(val)
                elif isinstance(val, str):
                    match = re.search(r"([\d.]+)", val)
                    if match:
                        metrics.power_draw = float(match.group(1))
                break
        
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass
    
    return metrics


def _parse_rocm_smi_text(output: str) -> GPUMetrics:
    """Parse rocm-smi plain text output."""
    metrics = GPUMetrics()
    
    try:
        lines = output.split("\n")
        
        for line in lines:
            # GPU utilization: look for "GPU%" column
            if "%" in line and "GPU" in line:
                match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
                if match:
                    metrics.utilization = float(match.group(1))
            
            # Temperature: look for temperature values
            if "Temp" in line or "°C" in line or "C" in line:
                match = re.search(r"(\d+(?:\.\d+)?)\s*(?:°C|C)", line)
                if match:
                    metrics.temperature = float(match.group(1))
            
            # Power: look for wattage
            if "Power" in line or "W" in line:
                match = re.search(r"(\d+(?:\.\d+)?)\s*W", line)
                if match:
                    metrics.power_draw = float(match.group(1))
        
        # Memory: try to get from separate rocm-smi call
        try:
            mem_result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if mem_result.returncode == 0:
                for line in mem_result.stdout.split("\n"):
                    if "Used" in line:
                        match = re.search(r"(\d+)", line)
                        if match:
                            # Assume MiB if no unit specified
                            metrics.memory_used = int(match.group(1)) / 1024  # Convert to GB
                    if "Total" in line:
                        match = re.search(r"(\d+)", line)
                        if match:
                            metrics.memory_total = int(match.group(1)) / 1024
                
                if metrics.memory_total and metrics.memory_used:
                    metrics.memory_percent = (metrics.memory_used / metrics.memory_total) * 100
        except:
            pass
        
    except Exception:
        pass
    
    return metrics


def _parse_nvidia_smi(output: str) -> GPUMetrics:
    """Parse nvidia-smi CSV output."""
    metrics = GPUMetrics()
    
    try:
        # Output format: utilization, memory_used, memory_total, temperature, power
        parts = output.strip().split(",")
        if len(parts) >= 5:
            metrics.utilization = float(parts[0].strip())
            metrics.memory_used = float(parts[1].strip()) / 1024  # MiB to GB
            metrics.memory_total = float(parts[2].strip()) / 1024
            metrics.temperature = float(parts[3].strip())
            metrics.power_draw = float(parts[4].strip())
            
            if metrics.memory_total:
                metrics.memory_percent = (metrics.memory_used / metrics.memory_total) * 100
    except (ValueError, IndexError):
        pass
    
    return metrics


class GPUMonitor:
    """
    GPU monitor that caches metrics to avoid too-frequent polling.
    
    Usage:
        monitor = GPUMonitor()
        metrics = monitor.get_metrics()  # Cached, polls at most every 2s
    """
    
    def __init__(self, poll_interval: float = 2.0):
        self.poll_interval = poll_interval
        self._last_poll = 0.0
        self._cached_metrics = GPUMetrics()
    
    def get_metrics(self) -> GPUMetrics:
        """Get GPU metrics with caching."""
        import time
        now = time.time()
        
        if now - self._last_poll >= self.poll_interval:
            self._cached_metrics = get_gpu_metrics()
            self._last_poll = now
        
        return self._cached_metrics
    
    def force_refresh(self) -> GPUMetrics:
        """Force a refresh of GPU metrics."""
        import time
        self._cached_metrics = get_gpu_metrics()
        self._last_poll = time.time()
        return self._cached_metrics

