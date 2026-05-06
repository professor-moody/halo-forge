"""Hardware telemetry — the visual signature of halo-forge.

The telemetry strip in the public_app needs live numbers from the actual
accelerator running underneath. ROCm has rocm-smi, CUDA has nvidia-smi,
Apple Silicon has limited unprivileged options (most macOS thermal/power
APIs require sudo), and CPU is psutil. This module abstracts those
differences behind a single TelemetryProvider so the frontend just calls
GET /api/public/telemetry and gets a uniformly-shaped payload.

Honesty about gaps:
  - On macOS without sudo, GPU utilization, power, and temperature are
    not reliably accessible. The provider returns None for those fields
    and the frontend renders "—".
  - VRAM on Apple Silicon is unified system memory; we report
    torch.mps allocated bytes (training-relevant) rather than the
    abstract concept of "GPU memory" which doesn't quite apply.
  - rocm-smi/nvidia-smi calls are gated by a 1s cache so we never
    invoke the binary more than once per second even if the endpoint
    is hammered. Subprocess overhead matters at 1Hz polling.
"""

from halo_forge.telemetry.base import (
    TelemetryProvider,
    TelemetrySample,
    TelemetryUnavailableError,
)
from halo_forge.telemetry.registry import get_telemetry_provider

__all__ = [
    "TelemetryProvider",
    "TelemetrySample",
    "TelemetryUnavailableError",
    "get_telemetry_provider",
]
