"""
Hardware Monitor for halo-forge benchmarks.

Collects GPU and system metrics during training/generation runs.
Designed for AMD Strix Halo (gfx1151) with ROCm.
"""

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import csv
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class HardwareSample:
    """Single hardware metrics sample."""
    timestamp: float
    gpu_util_pct: float = 0.0
    gpu_mem_used_gb: float = 0.0
    gpu_mem_total_gb: float = 0.0
    gpu_temp_c: float = 0.0
    gpu_power_w: float = 0.0
    sys_mem_used_gb: float = 0.0
    sys_mem_total_gb: float = 0.0
    cpu_util_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "gpu_util_pct": self.gpu_util_pct,
            "gpu_mem_used_gb": round(self.gpu_mem_used_gb, 2),
            "gpu_mem_total_gb": round(self.gpu_mem_total_gb, 2),
            "gpu_temp_c": self.gpu_temp_c,
            "gpu_power_w": round(self.gpu_power_w, 2),
            "sys_mem_used_gb": round(self.sys_mem_used_gb, 2),
            "sys_mem_total_gb": round(self.sys_mem_total_gb, 2),
            "cpu_util_pct": self.cpu_util_pct,
        }


@dataclass
class HardwareSummary:
    """Aggregated hardware metrics summary."""
    duration_sec: float = 0.0
    samples_count: int = 0
    
    # GPU metrics
    gpu_util_avg: float = 0.0
    gpu_util_peak: float = 0.0
    gpu_mem_avg_gb: float = 0.0
    gpu_mem_peak_gb: float = 0.0
    gpu_temp_avg_c: float = 0.0
    gpu_temp_peak_c: float = 0.0
    gpu_power_avg_w: float = 0.0
    gpu_power_peak_w: float = 0.0
    gpu_energy_wh: float = 0.0  # Estimated energy consumption
    
    # System metrics
    sys_mem_avg_gb: float = 0.0
    sys_mem_peak_gb: float = 0.0
    cpu_util_avg: float = 0.0
    cpu_util_peak: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration_sec": round(self.duration_sec, 1),
            "samples_count": self.samples_count,
            "gpu": {
                "utilization_avg_pct": round(self.gpu_util_avg, 1),
                "utilization_peak_pct": round(self.gpu_util_peak, 1),
                "memory_avg_gb": round(self.gpu_mem_avg_gb, 2),
                "memory_peak_gb": round(self.gpu_mem_peak_gb, 2),
                "temp_avg_c": round(self.gpu_temp_avg_c, 1),
                "temp_peak_c": round(self.gpu_temp_peak_c, 1),
                "power_avg_w": round(self.gpu_power_avg_w, 1),
                "power_peak_w": round(self.gpu_power_peak_w, 1),
                "energy_wh": round(self.gpu_energy_wh, 2),
            },
            "system": {
                "memory_avg_gb": round(self.sys_mem_avg_gb, 2),
                "memory_peak_gb": round(self.sys_mem_peak_gb, 2),
                "cpu_util_avg_pct": round(self.cpu_util_avg, 1),
                "cpu_util_peak_pct": round(self.cpu_util_peak, 1),
            }
        }


def _run_cmd(cmd: List[str], default: str = "") -> str:
    """Run a command and return stdout, or default on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else default
    except Exception:
        return default


def _parse_rocm_smi_json() -> Dict[str, Any]:
    """Parse rocm-smi JSON output for GPU metrics."""
    output = _run_cmd(["rocm-smi", "--json"])
    if not output:
        return {}
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {}


def get_gpu_utilization() -> float:
    """Get GPU utilization percentage."""
    # Try rocm-smi first
    output = _run_cmd(["rocm-smi", "--showuse"])
    for line in output.split("\n"):
        if "GPU use" in line or "GPU%" in line:
            # Extract percentage
            parts = line.split()
            for part in parts:
                if "%" in part:
                    try:
                        return float(part.replace("%", ""))
                    except ValueError:
                        pass
    
    # Try amd-smi as fallback
    output = _run_cmd(["amd-smi", "metric", "-g", "0", "--usage"])
    for line in output.split("\n"):
        if "GFX" in line.upper() or "GPU" in line.upper():
            parts = line.split()
            for part in parts:
                try:
                    val = float(part.replace("%", ""))
                    if 0 <= val <= 100:
                        return val
                except ValueError:
                    pass
    
    return 0.0


def get_gtt_memory() -> tuple:
    """Get GTT memory usage (used_gb, total_gb)."""
    output = _run_cmd(["rocm-smi", "--showmeminfo", "gtt"])
    used_gb = 0.0
    total_gb = 128.0  # Default for Strix Halo
    
    for line in output.split("\n"):
        line_lower = line.lower()
        if "used" in line_lower:
            parts = line.split()
            for i, part in enumerate(parts):
                try:
                    val = float(part)
                    # Check if next part is unit
                    if i + 1 < len(parts):
                        unit = parts[i + 1].upper()
                        if "GB" in unit:
                            used_gb = val
                        elif "MB" in unit:
                            used_gb = val / 1024
                        elif "KB" in unit:
                            used_gb = val / (1024 * 1024)
                    else:
                        # Assume bytes if no unit
                        used_gb = val / (1024 ** 3)
                    break
                except ValueError:
                    pass
        elif "total" in line_lower:
            parts = line.split()
            for i, part in enumerate(parts):
                try:
                    val = float(part)
                    if i + 1 < len(parts):
                        unit = parts[i + 1].upper()
                        if "GB" in unit:
                            total_gb = val
                        elif "MB" in unit:
                            total_gb = val / 1024
                    break
                except ValueError:
                    pass
    
    return used_gb, total_gb


def get_gpu_temperature() -> float:
    """Get GPU temperature in Celsius."""
    output = _run_cmd(["rocm-smi", "--showtemp"])
    for line in output.split("\n"):
        if "edge" in line.lower() or "junction" in line.lower() or "temperature" in line.lower():
            parts = line.split()
            for part in parts:
                try:
                    # Remove 'c' or 'C' suffix if present
                    clean = part.rstrip("cC")
                    val = float(clean)
                    if 0 < val < 150:  # Reasonable temp range
                        return val
                except ValueError:
                    pass
    return 0.0


def get_gpu_power() -> float:
    """Get GPU power draw in Watts."""
    output = _run_cmd(["rocm-smi", "--showpower"])
    for line in output.split("\n"):
        if "power" in line.lower():
            parts = line.split()
            for i, part in enumerate(parts):
                try:
                    val = float(part)
                    # Check if reasonable power value
                    if 0 < val < 500:
                        return val
                except ValueError:
                    pass
    return 0.0


def get_system_memory() -> tuple:
    """Get system memory usage (used_gb, total_gb)."""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        return mem.used / (1024 ** 3), mem.total / (1024 ** 3)
    
    # Fallback to /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    val = int(parts[1]) / (1024 ** 2)  # KB to GB
                    meminfo[key] = val
            
            total = meminfo.get("MemTotal", 0)
            available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            used = total - available
            return used, total
    except Exception:
        return 0.0, 0.0


def get_cpu_utilization() -> float:
    """Get CPU utilization percentage."""
    if HAS_PSUTIL:
        return psutil.cpu_percent(interval=0.1)
    return 0.0


class HardwareMonitor:
    """
    Background thread that periodically samples hardware metrics.
    
    Usage:
        monitor = HardwareMonitor(interval_sec=2)
        monitor.start()
        # ... do work ...
        monitor.stop()
        summary = monitor.summarize()
        monitor.save_csv("metrics.csv")
    """
    
    def __init__(self, interval_sec: float = 2.0):
        self.interval = interval_sec
        self.samples: List[HardwareSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()
    
    def sample(self) -> HardwareSample:
        """Take a single hardware sample."""
        gpu_mem_used, gpu_mem_total = get_gtt_memory()
        sys_mem_used, sys_mem_total = get_system_memory()
        
        return HardwareSample(
            timestamp=time.time(),
            gpu_util_pct=get_gpu_utilization(),
            gpu_mem_used_gb=gpu_mem_used,
            gpu_mem_total_gb=gpu_mem_total,
            gpu_temp_c=get_gpu_temperature(),
            gpu_power_w=get_gpu_power(),
            sys_mem_used_gb=sys_mem_used,
            sys_mem_total_gb=sys_mem_total,
            cpu_util_pct=get_cpu_utilization(),
        )
    
    def _sample_loop(self):
        """Background sampling loop."""
        while self._running:
            sample = self.sample()
            with self._lock:
                self.samples.append(sample)
            time.sleep(self.interval)
    
    def start(self):
        """Start background sampling."""
        if self._running:
            return
        
        self._running = True
        self._start_time = time.time()
        self.samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop background sampling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval * 2)
            self._thread = None
    
    def summarize(self) -> HardwareSummary:
        """Calculate summary statistics from collected samples."""
        with self._lock:
            samples = list(self.samples)
        
        if not samples:
            return HardwareSummary()
        
        n = len(samples)
        duration = samples[-1].timestamp - samples[0].timestamp if n > 1 else 0
        
        # Calculate aggregates
        gpu_utils = [s.gpu_util_pct for s in samples]
        gpu_mems = [s.gpu_mem_used_gb for s in samples]
        gpu_temps = [s.gpu_temp_c for s in samples]
        gpu_powers = [s.gpu_power_w for s in samples]
        sys_mems = [s.sys_mem_used_gb for s in samples]
        cpu_utils = [s.cpu_util_pct for s in samples]
        
        # Estimate energy: average power * duration (Wh)
        avg_power = sum(gpu_powers) / n if n else 0
        energy_wh = (avg_power * duration) / 3600 if duration > 0 else 0
        
        return HardwareSummary(
            duration_sec=duration,
            samples_count=n,
            gpu_util_avg=sum(gpu_utils) / n,
            gpu_util_peak=max(gpu_utils),
            gpu_mem_avg_gb=sum(gpu_mems) / n,
            gpu_mem_peak_gb=max(gpu_mems),
            gpu_temp_avg_c=sum(gpu_temps) / n if any(gpu_temps) else 0,
            gpu_temp_peak_c=max(gpu_temps) if any(gpu_temps) else 0,
            gpu_power_avg_w=avg_power,
            gpu_power_peak_w=max(gpu_powers) if any(gpu_powers) else 0,
            gpu_energy_wh=energy_wh,
            sys_mem_avg_gb=sum(sys_mems) / n,
            sys_mem_peak_gb=max(sys_mems),
            cpu_util_avg=sum(cpu_utils) / n,
            cpu_util_peak=max(cpu_utils),
        )
    
    def save_csv(self, path: str):
        """Save time-series samples to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            samples = list(self.samples)
        
        if not samples:
            return
        
        fieldnames = list(samples[0].to_dict().keys())
        
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sample in samples:
                writer.writerow(sample.to_dict())
    
    def save_summary(self, path: str):
        """Save summary statistics to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.summarize()
        with open(path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
    
    def get_samples(self) -> List[Dict[str, Any]]:
        """Get all samples as list of dicts."""
        with self._lock:
            return [s.to_dict() for s in self.samples]


def quick_sample() -> Dict[str, Any]:
    """Take a single quick hardware sample and return as dict."""
    monitor = HardwareMonitor()
    return monitor.sample().to_dict()


if __name__ == "__main__":
    # Quick test
    print("Taking hardware sample...")
    sample = quick_sample()
    print(json.dumps(sample, indent=2))
    
    print("\nMonitoring for 10 seconds...")
    monitor = HardwareMonitor(interval_sec=1)
    monitor.start()
    time.sleep(10)
    monitor.stop()
    
    print(f"\nCollected {len(monitor.samples)} samples")
    summary = monitor.summarize()
    print(json.dumps(summary.to_dict(), indent=2))

