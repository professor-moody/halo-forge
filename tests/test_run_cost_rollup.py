"""Run-cost rollup tests (Track P2).

Validates the cost estimator math, env-var override, and the public-API
integration that surfaces `details["cost"]` on /runs/{id}.
"""

from __future__ import annotations

import os

import pytest


def test_estimate_run_cost_zero_duration_yields_zero():
    from halo_forge.telemetry.cost import estimate_run_cost

    result = estimate_run_cost(duration_seconds=0.0, backend_name="cuda")
    assert result.duration_seconds == 0.0
    assert result.duration_hours == 0.0
    assert result.energy_kwh == 0.0
    assert result.cost_usd == 0.0
    assert result.power_watts_estimated > 0  # nominal table populated
    assert result.source == "nominal"


def test_estimate_run_cost_uses_nominal_when_no_sample():
    from halo_forge.telemetry.cost import (
        BACKEND_NOMINAL_POWER_WATTS,
        estimate_run_cost,
    )

    # 1 hour at the rocm_gfx1151 nominal of 120W = 0.120 kWh.
    result = estimate_run_cost(duration_seconds=3600.0, backend_name="rocm_gfx1151")
    assert result.power_watts_estimated == BACKEND_NOMINAL_POWER_WATTS["rocm_gfx1151"]
    assert pytest.approx(result.energy_kwh, rel=1e-9) == 0.120
    # default $0.15 / kWh × 0.120 = $0.018
    assert pytest.approx(result.cost_usd, rel=1e-9) == 0.018
    assert result.source == "nominal"


def test_estimate_run_cost_prefers_sampled_power():
    """When a real sample is provided, source flips to 'measured' and the
    nominal-table value is bypassed."""
    from halo_forge.telemetry.cost import estimate_run_cost

    # 30 minutes at sampled 200W = 0.100 kWh.
    result = estimate_run_cost(
        duration_seconds=1800.0,
        backend_name="cuda",
        sampled_power_watts=200.0,
    )
    assert result.source == "measured"
    assert result.power_watts_estimated == 200.0
    assert pytest.approx(result.energy_kwh, rel=1e-9) == 0.100


def test_cost_per_kwh_env_var_override(monkeypatch):
    from halo_forge.telemetry.cost import estimate_run_cost

    monkeypatch.setenv("HALOFORGE_COST_PER_KWH", "0.40")
    result = estimate_run_cost(duration_seconds=3600.0, backend_name="cuda")
    assert result.cost_per_kwh == 0.40
    # 350W (cuda nominal) × 1h = 0.350 kWh × $0.40 = $0.14
    assert pytest.approx(result.cost_usd, rel=1e-9) == 0.14


def test_cost_per_kwh_explicit_arg_wins_over_env(monkeypatch):
    from halo_forge.telemetry.cost import estimate_run_cost

    monkeypatch.setenv("HALOFORGE_COST_PER_KWH", "0.40")
    result = estimate_run_cost(
        duration_seconds=3600.0, backend_name="cuda", cost_per_kwh=0.05
    )
    assert result.cost_per_kwh == 0.05


def test_unknown_backend_falls_back_to_cuda_nominal():
    from halo_forge.telemetry.cost import (
        BACKEND_NOMINAL_POWER_WATTS,
        estimate_run_cost,
    )

    result = estimate_run_cost(
        duration_seconds=60.0, backend_name="some_future_chip_we_dont_know"
    )
    assert result.power_watts_estimated == BACKEND_NOMINAL_POWER_WATTS["cuda"]
    assert result.source == "nominal"


def test_negative_duration_clamps_to_zero():
    """Defensive: a malformed run shouldn't return a negative cost."""
    from halo_forge.telemetry.cost import estimate_run_cost

    result = estimate_run_cost(duration_seconds=-100.0, backend_name="cuda")
    assert result.duration_seconds == 0.0
    assert result.cost_usd == 0.0


def test_project_run_cost_sums_cycle_durations():
    from halo_forge.public_api.service import _project_run_cost

    raw_data = {
        "cycles": [
            {"cycle": 0, "cycle_duration_seconds": 600.0},
            {"cycle": 1, "cycle_duration_seconds": 1200.0},
            {"cycle": 2, "cycle_duration_seconds": 0.0},  # zero ok
            {"cycle": 3, "cycle_duration_seconds": None},  # missing ok
            {"cycle": 4},  # field absent ok
        ]
    }
    result = _project_run_cost(raw_data, backend_name="rocm_gfx1151")
    # 1800 seconds total = 0.5 hour at 120W = 0.06 kWh
    assert pytest.approx(result["energy_kwh"], rel=1e-9) == 0.06
    assert pytest.approx(result["duration_seconds"]) == 1800.0
    assert result["backend"] == "rocm_gfx1151"


def test_project_run_cost_handles_malformed_input():
    from halo_forge.public_api.service import _project_run_cost

    # Non-dict raw_data — common for older / partial summaries.
    result = _project_run_cost(None, backend_name="cuda")  # type: ignore[arg-type]
    assert result["duration_seconds"] == 0.0
    assert result["cost_usd"] == 0.0
    assert result["source"] == "nominal"

    # Cycles missing entirely.
    result = _project_run_cost({"some": "other_payload"}, backend_name="cuda")
    assert result["duration_seconds"] == 0.0


def test_run_cost_dataclass_serializable():
    """The frontend consumes `cost.to_dict()` directly — every field must
    survive round-trip through dataclasses.asdict."""
    from halo_forge.telemetry.cost import estimate_run_cost

    cost = estimate_run_cost(duration_seconds=120.0, backend_name="mlx")
    d = cost.to_dict()
    assert set(d.keys()) == {
        "duration_seconds",
        "duration_hours",
        "power_watts_estimated",
        "energy_kwh",
        "cost_usd",
        "cost_per_kwh",
        "backend",
        "source",
    }
    # No ints sneaking in where floats are expected.
    for k in (
        "duration_seconds",
        "duration_hours",
        "power_watts_estimated",
        "energy_kwh",
        "cost_usd",
        "cost_per_kwh",
    ):
        assert isinstance(d[k], float), f"{k} should be float, got {type(d[k])}"
