"""Run-cost estimation (Track P2).

Computes wall-clock × power × $/kWh = run cost so the public API and
frontend can show operators what each run actually consumed in energy
and dollars. Today this is a *roll-up over training_summary*; per-cycle
power sampling lives in `telemetry/` already but isn't yet persisted
per run, so we estimate from the wall-clock duration plus a backend
nominal-power table. The frontend renders the source ('measured' vs
'nominal') so users know they're looking at an estimate.

Default electricity price: $0.15 / kWh (US 2024 average). Override per
deployment with the `HALOFORGE_COST_PER_KWH` env var.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


# Sustained-training power estimates, in watts, by accelerator-kind name.
# These are deliberately conservative averages from public benchmarks
# and our own measurement; they're a rough estimator, not a meter. Sites
# with real per-run sampling override these via `sampled_power_watts`.
BACKEND_NOMINAL_POWER_WATTS: Dict[str, float] = {
    "rocm_gfx1151": 120.0,    # Strix Halo APU under sustained training
    "rocm": 250.0,             # generic ROCm (e.g. MI210 / 7900 XTX)
    "cuda": 350.0,             # generic single-card CUDA (RTX 4090 class)
    "mps": 30.0,               # Apple Silicon training via PyTorch MPS
    "mlx": 30.0,               # Apple Silicon training via MLX
    "cpu": 25.0,
}

DEFAULT_COST_PER_KWH = 0.15


def _resolve_cost_per_kwh(override: Optional[float] = None) -> float:
    """Pick the active electricity price.

    Order: explicit arg → `HALOFORGE_COST_PER_KWH` env var → US-average default.
    """
    if override is not None and override > 0:
        return float(override)
    env = os.environ.get("HALOFORGE_COST_PER_KWH")
    if env:
        try:
            v = float(env)
            if v > 0:
                return v
        except ValueError:
            pass
    return DEFAULT_COST_PER_KWH


def _resolve_nominal_power(backend_name: str) -> float:
    """Look up the nominal sustained-training power for a backend name.

    Falls back to the generic CUDA estimate if the name isn't in the
    table — better to over-estimate than to silently report zero.
    """
    if not backend_name:
        return BACKEND_NOMINAL_POWER_WATTS["cuda"]
    if backend_name in BACKEND_NOMINAL_POWER_WATTS:
        return BACKEND_NOMINAL_POWER_WATTS[backend_name]
    # rocm_gfx1100 / rocm_mi300 / etc fall back to generic ROCm
    for prefix, watts in BACKEND_NOMINAL_POWER_WATTS.items():
        if backend_name.startswith(prefix):
            return watts
    return BACKEND_NOMINAL_POWER_WATTS["cuda"]


@dataclass
class RunCost:
    """Energy + dollar cost for a training run.

    All fields are present on every response; if the duration is zero
    (no cycles ran) every numeric value is zero too. The `source` field
    tells the UI whether to render an "estimate" badge.
    """

    duration_seconds: float
    duration_hours: float
    power_watts_estimated: float
    energy_kwh: float
    cost_usd: float
    cost_per_kwh: float
    backend: str
    source: str  # "measured" if sampled_power_watts provided, else "nominal"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_run_cost(
    *,
    duration_seconds: float,
    backend_name: str,
    sampled_power_watts: Optional[float] = None,
    cost_per_kwh: Optional[float] = None,
) -> RunCost:
    """Estimate energy + dollar cost for a training run.

    Args:
        duration_seconds: Wall-clock seconds the run trained for. Sum of
            `cycle_duration_seconds` for completed runs; `(now - started_at)`
            for active runs.
        backend_name: accelerator-kind name (rocm_gfx1151, cuda, mps, mlx, cpu).
        sampled_power_watts: Live or historical mean power draw, in watts.
            When provided, marks the result as `source="measured"`. When
            None, falls back to the backend's nominal table.
        cost_per_kwh: Price override; falls back to env or default.

    Returns:
        RunCost — every numeric field is non-negative; `source` flags the
        provenance for the UI.
    """
    duration = max(0.0, float(duration_seconds or 0.0))
    cost_rate = _resolve_cost_per_kwh(cost_per_kwh)

    if sampled_power_watts and sampled_power_watts > 0:
        power = float(sampled_power_watts)
        source = "measured"
    else:
        power = _resolve_nominal_power(backend_name)
        source = "nominal"

    energy_kwh = (duration / 3600.0) * (power / 1000.0)
    cost_usd = energy_kwh * cost_rate

    return RunCost(
        duration_seconds=duration,
        duration_hours=duration / 3600.0,
        power_watts_estimated=power,
        energy_kwh=energy_kwh,
        cost_usd=cost_usd,
        cost_per_kwh=cost_rate,
        backend=backend_name or "unknown",
        source=source,
    )


__all__ = [
    "BACKEND_NOMINAL_POWER_WATTS",
    "DEFAULT_COST_PER_KWH",
    "RunCost",
    "estimate_run_cost",
]
