"""Shared psutil probes used by every concrete provider.

CPU / system memory are universally readable without elevated permission
on every platform we target, so this stays in one place. Each helper is
defensive: psutil being absent just returns None.
"""

from __future__ import annotations

from typing import Optional, Tuple


def cpu_util_percent() -> Optional[float]:
    """System-wide CPU utilization, 0-100. Non-blocking (uses psutil's
    cached snapshot, valid because the endpoint polls at ~1Hz)."""
    try:
        import psutil  # type: ignore[import]
    except ImportError:
        return None
    # interval=None reads the snapshot since the previous call. The first
    # call after import returns 0.0 — acceptable for our use case since
    # the second poll (~1s later) is the meaningful one.
    return float(psutil.cpu_percent(interval=None))


def sys_memory_gb() -> Tuple[Optional[float], Optional[float]]:
    """(used_gb, total_gb) for system RAM. None on import failure."""
    try:
        import psutil  # type: ignore[import]
    except ImportError:
        return None, None
    vm = psutil.virtual_memory()
    return vm.used / 1e9, vm.total / 1e9
