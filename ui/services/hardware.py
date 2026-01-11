"""
Hardware Monitoring Service

Get GPU stats from rocm-smi for AMD GPUs.
"""

import subprocess
import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUStats:
    """GPU statistics."""
    device_id: int = 0
    name: str = "Unknown GPU"
    temperature_c: Optional[float] = None
    utilization_percent: Optional[float] = None
    memory_used_gb: Optional[float] = None
    memory_total_gb: Optional[float] = None
    power_draw_w: Optional[float] = None
    power_cap_w: Optional[float] = None
    
    @property
    def memory_percent(self) -> Optional[float]:
        """Get memory usage as percentage."""
        if self.memory_used_gb is not None and self.memory_total_gb:
            return (self.memory_used_gb / self.memory_total_gb) * 100
        return None
    
    @property
    def power_percent(self) -> Optional[float]:
        """Get power usage as percentage of cap."""
        if self.power_draw_w is not None and self.power_cap_w:
            return (self.power_draw_w / self.power_cap_w) * 100
        return None


def get_gpu_stats() -> Optional[GPUStats]:
    """Get GPU statistics using rocm-smi."""
    try:
        # Try JSON output first (newer rocm-smi)
        result = subprocess.run(
            ['rocm-smi', '--json'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return _parse_rocm_json(result.stdout)
        
        # Fall back to regular output
        result = subprocess.run(
            ['rocm-smi', '--showuse', '--showmemuse', '--showtemp', '--showpower'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return _parse_rocm_text(result.stdout)
            
    except FileNotFoundError:
        # rocm-smi not available
        pass
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    
    return None


def _parse_rocm_json(output: str) -> Optional[GPUStats]:
    """Parse JSON output from rocm-smi --json."""
    try:
        data = json.loads(output)
        
        # Get first GPU
        if not data:
            return None
            
        gpu_key = list(data.keys())[0] if isinstance(data, dict) else None
        if not gpu_key:
            return None
            
        gpu = data[gpu_key]
        
        stats = GPUStats()
        
        # Parse device name
        stats.name = gpu.get('Card series', 'AMD GPU')
        
        # Temperature
        temp = gpu.get('Temperature (Sensor junction)', gpu.get('Temperature', None))
        if temp:
            # Extract number from string like "45.0c"
            match = re.search(r'([\d.]+)', str(temp))
            if match:
                stats.temperature_c = float(match.group(1))
        
        # GPU utilization
        util = gpu.get('GPU use (%)', gpu.get('GPU Utilization', None))
        if util:
            match = re.search(r'([\d.]+)', str(util))
            if match:
                stats.utilization_percent = float(match.group(1))
        
        # Memory
        mem_used = gpu.get('VRAM Total Used Memory (B)', None)
        mem_total = gpu.get('VRAM Total Memory (B)', None)
        
        if mem_used:
            stats.memory_used_gb = float(mem_used) / (1024**3)
        if mem_total:
            stats.memory_total_gb = float(mem_total) / (1024**3)
        
        # Power
        power = gpu.get('Average Graphics Package Power (W)', None)
        power_cap = gpu.get('Max Graphics Package Power (W)', None)
        
        if power:
            match = re.search(r'([\d.]+)', str(power))
            if match:
                stats.power_draw_w = float(match.group(1))
        if power_cap:
            match = re.search(r'([\d.]+)', str(power_cap))
            if match:
                stats.power_cap_w = float(match.group(1))
        
        return stats
        
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _parse_rocm_text(output: str) -> Optional[GPUStats]:
    """Parse text output from rocm-smi."""
    stats = GPUStats()
    
    lines = output.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # GPU name
        if 'GPU' in line and ':' in line:
            stats.name = line.split(':')[-1].strip()
        
        # Temperature
        if 'Temperature' in line:
            match = re.search(r'([\d.]+)\s*[Cc]', line)
            if match:
                stats.temperature_c = float(match.group(1))
        
        # GPU utilization
        if 'GPU use' in line or 'GPU Utilization' in line:
            match = re.search(r'([\d.]+)\s*%', line)
            if match:
                stats.utilization_percent = float(match.group(1))
        
        # Memory
        if 'Memory' in line and 'Used' in line:
            # Try to find GB value
            match = re.search(r'([\d.]+)\s*GB', line, re.IGNORECASE)
            if match:
                stats.memory_used_gb = float(match.group(1))
            else:
                # Try MB
                match = re.search(r'([\d.]+)\s*MB', line, re.IGNORECASE)
                if match:
                    stats.memory_used_gb = float(match.group(1)) / 1024
        
        # Power
        if 'Power' in line:
            match = re.search(r'([\d.]+)\s*W', line)
            if match:
                stats.power_draw_w = float(match.group(1))
    
    # Default memory total for Strix Halo
    if stats.memory_total_gb is None:
        stats.memory_total_gb = 128.0  # Common for Strix Halo
    
    return stats


def get_gpu_summary() -> dict:
    """Get a summary dictionary for UI display."""
    stats = get_gpu_stats()
    
    if stats is None:
        return {
            'available': False,
            'name': 'No GPU detected',
            'util': '--',
            'memory': '--',
            'temp': '--',
        }
    
    return {
        'available': True,
        'name': stats.name,
        'util': f'{stats.utilization_percent:.0f}%' if stats.utilization_percent else '--',
        'memory': f'{stats.memory_used_gb:.1f}/{stats.memory_total_gb:.0f}GB' if stats.memory_used_gb else '--',
        'temp': f'{stats.temperature_c:.0f}°C' if stats.temperature_c else '--',
        'power': f'{stats.power_draw_w:.0f}W' if stats.power_draw_w else '--',
        'memory_percent': stats.memory_percent,
        'util_percent': stats.utilization_percent,
    }
