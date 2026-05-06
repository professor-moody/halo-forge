"""NVIDIA CUDA telemetry — backed by `nvidia-smi --query-gpu=...`.

The `--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,
temperature.gpu --format=csv,nounits,noheader` form is the stable API
across every NVIDIA driver since at least 2018. No JSON parsing surprises.

Same 1s cache as the ROCm provider — nvidia-smi takes ~50-100ms per call.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Optional

from halo_forge.telemetry._psutil_helpers import cpu_util_percent, sys_memory_gb
from halo_forge.telemetry.base import (
    TelemetryProvider,
    TelemetrySample,
    TelemetryUnavailableError,
)


class CUDATelemetry(TelemetryProvider):
    """nvidia-smi --query-gpu -> TelemetrySample."""

    name = "cuda"

    _QUERY_FIELDS = (
        "name",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "power.draw",
        "temperature.gpu",
    )
    _CACHE_TTL_SECONDS = 1.0

    def __init__(self, backend_name: str = "cuda") -> None:
        self._backend_name = backend_name
        self._nvidia_smi = shutil.which("nvidia-smi")
        if self._nvidia_smi is None:
            raise TelemetryUnavailableError(
                "CUDA telemetry requires `nvidia-smi` on PATH (NVIDIA driver "
                "userspace). Install the driver or fall back to a different backend."
            )
        self._cache: Optional[list[str]] = None
        self._cache_ts: float = 0.0

    def sample(self) -> TelemetrySample:
        s = TelemetrySample.now(backend=self._backend_name)
        s.cpu_util_percent = cpu_util_percent()
        s.sys_mem_used_gb, s.sys_mem_total_gb = sys_memory_gb()

        try:
            row = self._read_nvidia_smi()
        except Exception as exc:
            s.note = f"nvidia-smi failed: {exc}"
            return s

        # CSV row order matches _QUERY_FIELDS
        if len(row) >= 6:
            s.device_name = row[0] or None
            s.gpu_util_percent = _try_float(row[1])
            mem_used_mb = _try_float(row[2])
            mem_total_mb = _try_float(row[3])
            if mem_used_mb is not None:
                s.vram_used_gb = mem_used_mb / 1024.0  # MiB -> GiB approximation
            if mem_total_mb is not None:
                s.vram_total_gb = mem_total_mb / 1024.0
            s.power_watts = _try_float(row[4])
            s.temp_celsius = _try_float(row[5])
        return s

    def _read_nvidia_smi(self) -> list[str]:
        now = time.time()
        if self._cache is not None and (now - self._cache_ts) < self._CACHE_TTL_SECONDS:
            return self._cache

        query = ",".join(self._QUERY_FIELDS)
        result = subprocess.run(
            [self._nvidia_smi, f"--query-gpu={query}", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            timeout=4,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"nvidia-smi exit {result.returncode}: {result.stderr.strip()[:200]}"
            )

        # Take the first GPU only; multi-GPU support is a follow-up.
        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        row = [c.strip() for c in first_line.split(",")]
        self._cache = row
        self._cache_ts = now
        return row


def _try_float(value: Optional[str]) -> Optional[float]:
    if value is None or value in ("", "[N/A]", "N/A"):
        return None
    try:
        return float(value)
    except ValueError:
        return None
