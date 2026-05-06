"""Apple Silicon telemetry provider.

What we can read without sudo on macOS:
  - torch.mps.driver_allocated_memory   (when torch + MPS available)
  - mlx.core.get_active_memory          (when mlx-lm installed)
  - psutil for CPU / system RAM
  - platform.processor() for the device name (rough)

What we *can't* read without sudo:
  - GPU utilization (powermetrics requires sudo)
  - GPU power (same)
  - GPU temperature (same)

So this provider populates VRAM-like memory + CPU + system RAM, and
leaves the GPU vitals as None. The frontend renders them as "—" and
shows a small "limited on macOS" hint in the tooltip.

Note on "VRAM" semantics: Apple Silicon uses unified memory, so there
is no distinct GPU memory pool. We report the *MPS-allocated* portion
(what torch is currently holding for tensors on the GPU device), not
total system memory. That's the operationally-relevant number for
training: when it gets close to 70-80% of available unified memory,
swap begins and throughput collapses.
"""

from __future__ import annotations

import platform
from typing import Optional, Tuple

from halo_forge.telemetry._psutil_helpers import cpu_util_percent, sys_memory_gb
from halo_forge.telemetry.base import TelemetryProvider, TelemetrySample


class AppleSiliconTelemetry(TelemetryProvider):
    """MPS + MLX + psutil. Everything else is None."""

    name = "apple_silicon"

    def __init__(self, backend_name: str) -> None:
        # backend_name may be "mps" or "mlx" depending on which path
        # was selected via `--accelerator`. We surface that exactly so
        # the strip in the frontend matches the backend chip in the
        # sidebar.
        self._backend_name = backend_name
        self._device_name = self._detect_device_name()

    def sample(self) -> TelemetrySample:
        s = TelemetrySample.now(backend=self._backend_name, device_name=self._device_name)

        # System
        s.cpu_util_percent = cpu_util_percent()
        s.sys_mem_used_gb, s.sys_mem_total_gb = sys_memory_gb()

        # GPU memory — pick the more accurate source for the active
        # backend. MLX gives us total active allocations across the
        # whole MLX runtime; MPS gives us what torch.mps has reserved.
        if self._backend_name == "mlx":
            used, total = self._mlx_memory_gb()
        else:
            used, total = self._mps_memory_gb()
        s.vram_used_gb = used
        s.vram_total_gb = total

        # Honest about the gaps — the strip shows these as "—" and the
        # note travels into the tooltip so users understand why.
        s.note = "GPU util / power / temp require sudo on macOS"
        return s

    # ------------------------------------------------------------------
    # Device identification
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device_name() -> Optional[str]:
        """Read the chip name from sysctl. Returns None on failure."""
        try:
            import subprocess

            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if out.returncode == 0:
                name = out.stdout.strip()
                if name:
                    return name
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            pass
        # Fallback to platform.processor() — typically "arm" on Apple Silicon
        # which isn't useful, but better than crashing.
        return platform.processor() or None

    # ------------------------------------------------------------------
    # Memory probes — both return (used_gb, total_gb).
    # ------------------------------------------------------------------

    @staticmethod
    def _mps_memory_gb() -> Tuple[Optional[float], Optional[float]]:
        """torch.mps allocated bytes, plus the system memory ceiling."""
        try:
            import torch
        except ImportError:
            return None, None

        mps = getattr(torch, "mps", None)
        if mps is None:
            return None, None

        # `driver_allocated_memory()` is what the MPS driver currently
        # holds for the process (closer to "VRAM in use" than
        # `current_allocated_memory()` which only counts active tensors).
        used: Optional[float] = None
        for fn in ("driver_allocated_memory", "current_allocated_memory"):
            probe = getattr(mps, fn, None)
            if callable(probe):
                try:
                    used = float(probe()) / 1e9
                    break
                except Exception:
                    continue

        # On Apple Silicon there is no separate GPU memory pool — total
        # is system RAM. Return that so the bar in the strip has a
        # denominator the operator can read against.
        _, total = sys_memory_gb()
        return used, total

    @staticmethod
    def _mlx_memory_gb() -> Tuple[Optional[float], Optional[float]]:
        """mlx_lm.metal active memory, falling back to mps numbers if mlx
        isn't installed."""
        try:
            import mlx.core as mx  # type: ignore[import]
        except ImportError:
            return AppleSiliconTelemetry._mps_memory_gb()

        used: Optional[float] = None
        # mlx 0.20+ exposes get_active_memory at the top level; older
        # versions had it under mlx.core.metal. Try both.
        for accessor in (
            lambda: mx.get_active_memory(),  # 0.20+
            lambda: mx.metal.get_active_memory(),  # legacy
        ):
            try:
                used = float(accessor()) / 1e9
                break
            except (AttributeError, Exception):
                continue

        _, total = sys_memory_gb()
        return used, total
