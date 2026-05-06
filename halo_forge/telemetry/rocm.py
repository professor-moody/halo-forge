"""ROCm telemetry — backed by `rocm-smi --json` parsing.

rocm-smi is the only no-sudo way to get GPU vitals on ROCm. Output schema
shifts across versions; we probe each field defensively and treat parse
failures as "value unavailable" rather than crashing the endpoint.

Caching: rocm-smi takes ~50-100ms per invocation. We cache the parsed
JSON for 1s so the endpoint can be hit at any frequency without burning
CPU on subprocess overhead.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any, Dict, Optional

from halo_forge.telemetry._psutil_helpers import cpu_util_percent, sys_memory_gb
from halo_forge.telemetry.base import (
    TelemetryProvider,
    TelemetrySample,
    TelemetryUnavailableError,
)


class ROCmTelemetry(TelemetryProvider):
    """rocm-smi --json -> TelemetrySample."""

    name = "rocm"

    # Cache window for the rocm-smi subprocess. 1s is enough that a
    # 2-3Hz endpoint poll never hits the binary more than once.
    _CACHE_TTL_SECONDS = 1.0

    def __init__(self, backend_name: str) -> None:
        self._backend_name = backend_name
        self._rocm_smi = shutil.which("rocm-smi")
        if self._rocm_smi is None:
            raise TelemetryUnavailableError(
                "ROCm telemetry requires `rocm-smi` on PATH. Install ROCm tools "
                "or fall back to a different backend."
            )
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_ts: float = 0.0

    def sample(self) -> TelemetrySample:
        s = TelemetrySample.now(backend=self._backend_name)
        s.cpu_util_percent = cpu_util_percent()
        s.sys_mem_used_gb, s.sys_mem_total_gb = sys_memory_gb()

        try:
            data = self._read_rocm_smi()
        except Exception as exc:
            s.note = f"rocm-smi failed: {exc}"
            return s

        # rocm-smi --json keys vary across versions — usually "card0",
        # "card1", ... We grab card0 and best-effort each field.
        card_key = next((k for k in data if k.startswith("card")), None)
        if card_key is None:
            s.note = "rocm-smi returned no card data"
            return s
        card = data[card_key]

        s.device_name = _read_first(card, ["Card series", "Card SKU", "Card model"])
        s.gpu_util_percent = _to_float(_read_first(card, ["GPU use (%)", "GPU Use (%)"]))
        s.power_watts = _to_float(_read_first(card, ["Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)"]))
        s.temp_celsius = _to_float(_read_first(card, ["Temperature (Sensor edge) (C)", "Temperature (Sensor junction) (C)"]))

        # Memory: rocm-smi reports in bytes under VRAM keys.
        used_b = _to_float(_read_first(card, ["VRAM Total Used Memory (B)"]))
        total_b = _to_float(_read_first(card, ["VRAM Total Memory (B)"]))
        if used_b is not None:
            s.vram_used_gb = used_b / 1e9
        if total_b is not None:
            s.vram_total_gb = total_b / 1e9

        return s

    def _read_rocm_smi(self) -> Dict[str, Any]:
        now = time.time()
        if self._cache is not None and (now - self._cache_ts) < self._CACHE_TTL_SECONDS:
            return self._cache

        result = subprocess.run(
            [self._rocm_smi, "--showuse", "--showmemuse", "--showpower",
             "--showtemp", "--showmeminfo", "vram", "--showproductname",
             "--json"],
            capture_output=True,
            text=True,
            timeout=4,
        )
        if result.returncode != 0:
            raise RuntimeError(f"rocm-smi exit {result.returncode}: {result.stderr.strip()[:200]}")

        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"rocm-smi output not JSON: {exc}") from exc

        self._cache = parsed
        self._cache_ts = now
        return parsed


def _read_first(d: Dict[str, Any], keys: list[str]) -> Optional[str]:
    """Return the first present key's value as a string, or None."""
    for k in keys:
        if k in d and d[k] not in (None, "", "N/A"):
            return str(d[k])
    return None


def _to_float(value: Optional[str]) -> Optional[float]:
    """Best-effort string -> float. Strips trailing units like ' W' or ' C'."""
    if value is None:
        return None
    cleaned = value.strip()
    # Strip trailing non-numeric chars (e.g. "60.0 C" -> "60.0")
    keep = []
    for ch in cleaned:
        if ch.isdigit() or ch in ".-":
            keep.append(ch)
        elif keep:
            break
    if not keep:
        return None
    try:
        return float("".join(keep))
    except ValueError:
        return None
