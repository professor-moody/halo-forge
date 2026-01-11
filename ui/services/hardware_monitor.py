"""
Hardware Monitor

Background task for GPU monitoring with callback support.
Provides live GPU stats updates to the dashboard.
"""

import asyncio
from typing import Optional, Callable, List, Any
from dataclasses import dataclass

from .hardware import GPUStats, get_gpu_stats, get_gpu_summary


@dataclass
class HardwareStats:
    """Combined hardware statistics."""
    gpu: Optional[GPUStats] = None
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    disk_percent: Optional[float] = None


class HardwareMonitor:
    """
    Background task for hardware monitoring.
    
    Provides periodic GPU stats updates via callbacks.
    
    Usage:
        monitor = get_hardware_monitor()
        
        def on_gpu_update(stats: GPUStats):
            print(f"GPU: {stats.utilization_percent}%")
        
        monitor.add_callback(on_gpu_update)
        await monitor.start()
        
        # Later...
        await monitor.stop()
    """
    
    def __init__(self, update_interval: float = 2.0):
        """
        Initialize hardware monitor.
        
        Args:
            update_interval: Seconds between updates
        """
        self.update_interval = update_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[GPUStats], Any]] = []
        self._latest_stats: Optional[GPUStats] = None
    
    def add_callback(self, callback: Callable[[GPUStats], Any]):
        """
        Add callback to receive GPU updates.
        
        Args:
            callback: Function that receives GPUStats
        """
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[GPUStats], Any]):
        """Remove a callback."""
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass
    
    async def start(self):
        """Start background monitoring."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self):
        """Stop background monitoring."""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Get GPU stats
                stats = get_gpu_stats()
                self._latest_stats = stats
                
                # Notify callbacks
                if stats:
                    for callback in self._callbacks:
                        try:
                            result = callback(stats)
                            # Handle async callbacks
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            print(f"Callback error: {e}")
                
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Monitor error: {e}")
                await asyncio.sleep(self.update_interval)
    
    def get_latest(self) -> Optional[GPUStats]:
        """Get most recent stats without waiting."""
        return self._latest_stats
    
    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running


# Singleton instance
_monitor: Optional[HardwareMonitor] = None


def get_hardware_monitor() -> HardwareMonitor:
    """Get the singleton hardware monitor."""
    global _monitor
    if _monitor is None:
        _monitor = HardwareMonitor()
    return _monitor


def get_gpu_stats_sync() -> Optional[GPUStats]:
    """
    Get GPU stats synchronously.
    
    Uses cached value from monitor if running,
    otherwise fetches fresh stats.
    """
    monitor = get_hardware_monitor()
    
    if monitor.is_running and monitor._latest_stats:
        return monitor._latest_stats
    
    return get_gpu_stats()


# Re-export for convenience
__all__ = [
    'HardwareMonitor',
    'HardwareStats',
    'GPUStats',
    'get_hardware_monitor',
    'get_gpu_stats',
    'get_gpu_stats_sync',
    'get_gpu_summary',
]
