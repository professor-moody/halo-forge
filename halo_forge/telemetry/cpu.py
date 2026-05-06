"""CPU-only telemetry provider — the terminal fallback.

When no accelerator is available (or detected as cpu), we still want the
strip in the public_app to render *something* useful. CPU + system memory
via psutil is the floor.
"""

from __future__ import annotations

import platform

from halo_forge.telemetry._psutil_helpers import cpu_util_percent, sys_memory_gb
from halo_forge.telemetry.base import TelemetryProvider, TelemetrySample


class CPUTelemetry(TelemetryProvider):
    name = "cpu"

    def sample(self) -> TelemetrySample:
        s = TelemetrySample.now(
            backend="cpu",
            device_name=platform.processor() or platform.machine() or None,
        )
        s.cpu_util_percent = cpu_util_percent()
        s.sys_mem_used_gb, s.sys_mem_total_gb = sys_memory_gb()
        s.note = "No accelerator detected; running on CPU"
        return s
