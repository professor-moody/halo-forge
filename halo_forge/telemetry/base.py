"""TelemetryProvider ABC + sample dataclass.

Every concrete provider returns a `TelemetrySample` with the same shape;
fields the underlying API can't provide are set to None. This lets the
frontend render the strip identically across backends — Apple Silicon
shows "—" where it has no permission, ROCm/CUDA fill all the values.

Design notes:
  - Samples are stateless. The provider may cache the underlying probe
    output for a short window (1s) but the sample itself is just data.
  - All numbers are SI / human-friendly: GB for memory, watts for power,
    Celsius for temp, percent for utilization, tokens/s for throughput.
  - `device_name` is the human-readable identifier ("AMD Strix Halo",
    "Apple M3 Max", "NVIDIA RTX 4090"). Used in the strip's tooltip.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from time import time
from typing import Any, Dict, Optional


class TelemetryUnavailableError(RuntimeError):
    """Raised when no telemetry provider is wired for the current host.

    Callers should catch this and surface a "telemetry unavailable" state
    rather than crashing the whole API endpoint.
    """


@dataclass
class TelemetrySample:
    """Snapshot of accelerator + system vitals at a point in time.

    Every field is Optional — the contract is "best effort, return None
    rather than fabricate". The frontend treats None as "—".
    """

    # Provenance
    timestamp: float
    backend: str  # accelerator-kind name: rocm_gfx1151, cuda, mps, mlx, cpu
    device_name: Optional[str]  # "AMD Radeon ...", "Apple M3 Max", "NVIDIA RTX 4090"

    # GPU vitals
    gpu_util_percent: Optional[float] = None  # 0-100
    vram_used_gb: Optional[float] = None
    vram_total_gb: Optional[float] = None
    power_watts: Optional[float] = None
    temp_celsius: Optional[float] = None

    # System (always best-effort via psutil)
    cpu_util_percent: Optional[float] = None  # 0-100
    sys_mem_used_gb: Optional[float] = None
    sys_mem_total_gb: Optional[float] = None

    # Training-attached (Phase D will wire these from active run state).
    # For now they're always None; the field exists so the API contract
    # doesn't shift when phase D lands.
    throughput_tokens_per_sec: Optional[float] = None
    active_run_id: Optional[str] = None

    # Free-form notes the provider wants to surface (e.g. "rocm-smi exit
    # code 1 — falling back to torch.cuda probe"). Frontend shows these
    # in a tooltip so debugging telemetry doesn't require a server log.
    note: Optional[str] = None

    @classmethod
    def now(cls, *, backend: str, device_name: Optional[str] = None) -> "TelemetrySample":
        """Construct an empty sample with the timestamp populated.

        Concrete providers fill in the fields they can probe and return.
        """
        return cls(timestamp=time(), backend=backend, device_name=device_name)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict shape for FastAPI serialization."""
        return asdict(self)


class TelemetryProvider(ABC):
    """Strategy that produces TelemetrySamples for one accelerator family.

    Concrete impls live alongside in this package:
      - apple_silicon.py  (mps + mlx; no sudo; limited)
      - rocm.py           (rocm-smi --json)
      - cuda.py           (nvidia-smi)
      - cpu.py            (psutil only — terminal fallback)
    """

    name: str

    @abstractmethod
    def sample(self) -> TelemetrySample:
        """Return current vitals. Should never raise on per-field probe
        failure; set the field to None and (optionally) note the failure
        in `sample.note`. Raise TelemetryUnavailableError only when the
        provider itself can't run at all (binary missing, library
        missing, permission denied for the whole probe).
        """

    def close(self) -> None:
        """Release any cached resources (open subprocess pipes, etc.).

        Default no-op; override for providers that hold persistent state.
        """
