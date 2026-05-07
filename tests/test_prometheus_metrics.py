"""Prometheus exposition tests (Track P3)."""

from __future__ import annotations

import sys

import pytest


# ---------- low-level formatter --------------------------------------------


def test_format_metrics_emits_help_type_per_metric():
    from halo_forge.metrics import format_metrics

    out = format_metrics(metrics=[
        {
            "name": "halo_forge_test_gauge",
            "help": "Test gauge",
            "type": "gauge",
            "samples": [{"value": 42, "labels": None}],
        },
    ])
    lines = out.splitlines()
    assert "# HELP halo_forge_test_gauge Test gauge" in lines
    assert "# TYPE halo_forge_test_gauge gauge" in lines
    assert "halo_forge_test_gauge 42" in lines


def test_format_metrics_renders_labels():
    from halo_forge.metrics import format_metrics

    out = format_metrics(metrics=[{
        "name": "halo_forge_runs_by_modality",
        "help": "Runs grouped by modality",
        "type": "gauge",
        "samples": [
            {"value": 12, "labels": {"modality": "sft"}},
            {"value": 5, "labels": {"modality": "raft"}},
        ],
    }])
    assert 'halo_forge_runs_by_modality{modality="sft"} 12' in out
    assert 'halo_forge_runs_by_modality{modality="raft"} 5' in out


def test_format_metrics_escapes_label_values():
    """Backslashes and quotes need escaping per the exposition spec."""
    from halo_forge.metrics import format_metrics

    out = format_metrics(metrics=[{
        "name": "halo_forge_test",
        "help": "x",
        "type": "gauge",
        "samples": [{"value": 1, "labels": {"path": 'a"b\\c'}}],
    }])
    assert 'path="a\\"b\\\\c"' in out


def test_format_metrics_emits_nan_for_missing_value():
    """A None sample value renders as NaN — keeps the metric line
    present (alerting on 'metric stopped reporting' depends on it)."""
    from halo_forge.metrics import format_metrics

    out = format_metrics(metrics=[{
        "name": "halo_forge_x",
        "help": "x",
        "type": "gauge",
        "samples": [{"value": None, "labels": None}],
    }])
    assert "halo_forge_x NaN" in out


def test_format_metrics_empty_samples_keeps_help_type():
    """Empty samples still emit HELP/TYPE so a scraper can tell
    'no samples now' from 'metric doesn't exist'."""
    from halo_forge.metrics import format_metrics

    out = format_metrics(metrics=[{
        "name": "halo_forge_y",
        "help": "y",
        "type": "gauge",
        "samples": [],
    }])
    assert "# HELP halo_forge_y y" in out
    assert "# TYPE halo_forge_y gauge" in out


def test_label_ordering_is_deterministic():
    """Multiple labels render in alpha order so scrape output is
    diff-stable."""
    from halo_forge.metrics import format_metrics

    out = format_metrics(metrics=[{
        "name": "halo_forge_t",
        "help": "t",
        "type": "gauge",
        "samples": [{
            "value": 1,
            "labels": {"zebra": "z", "alpha": "a", "middle": "m"},
        }],
    }])
    # Should be alpha order regardless of dict insertion order.
    assert 'alpha="a",middle="m",zebra="z"' in out


# ---------- halo-forge renderer --------------------------------------------


def test_render_metrics_includes_canonical_metric_names():
    """Every documented metric appears in the output even when telemetry
    is None (HELP/TYPE preamble guarantees the metric exists)."""
    from halo_forge.metrics import render_metrics

    out = render_metrics(telemetry=None, run_stats=None, backend_info=None)
    expected = [
        "halo_forge_gpu_utilization_percent",
        "halo_forge_vram_used_gigabytes",
        "halo_forge_vram_total_gigabytes",
        "halo_forge_power_watts",
        "halo_forge_temperature_celsius",
        "halo_forge_cpu_utilization_percent",
        "halo_forge_system_memory_used_gigabytes",
        "halo_forge_throughput_tokens_per_second",
        "halo_forge_runs_total",
        "halo_forge_active_runs",
        "halo_forge_runs_by_modality",
        "halo_forge_runs_by_status",
        "halo_forge_build_info",
    ]
    for name in expected:
        assert f"# TYPE {name} gauge" in out, f"missing metric {name}"


def test_render_metrics_uses_telemetry_values():
    from halo_forge.metrics import render_metrics

    out = render_metrics(
        telemetry={
            "backend": "rocm_gfx1151",
            "device_name": "AMD Strix Halo",
            "gpu_util_percent": 73.5,
            "power_watts": 124.0,
            "vram_used_gb": 32.5,
        },
    )
    assert "73.5" in out
    assert 'backend="rocm_gfx1151"' in out
    assert 'device="AMD Strix Halo"' in out


def test_render_metrics_run_stats_emits_per_modality_per_status():
    from halo_forge.metrics import render_metrics

    out = render_metrics(
        telemetry=None,
        run_stats={
            "total_runs": 17,
            "active_runs": 2,
            "by_modality": {"sft": 12, "raft": 5},
            "by_status": {"completed": 14, "failed": 3},
        },
    )
    assert "halo_forge_runs_total 17" in out
    assert "halo_forge_active_runs 2" in out
    assert 'halo_forge_runs_by_modality{modality="sft"} 12' in out
    assert 'halo_forge_runs_by_modality{modality="raft"} 5' in out
    assert 'halo_forge_runs_by_status{status="completed"} 14' in out


def test_render_metrics_handles_missing_telemetry_with_nan():
    from halo_forge.metrics import render_metrics

    out = render_metrics(telemetry={
        "backend": "cuda",
        "device_name": "RTX 4090",
        "gpu_util_percent": None,  # explicitly missing
    })
    assert "halo_forge_gpu_utilization_percent" in out
    # NaN sample for the missing field, with the right backend label.
    assert 'backend="cuda"' in out
    assert "NaN" in out


# ---------- end-to-end through TestClient ----------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "runs.db"))
    from halo_forge.run_db import db as db_mod
    db_mod._GLOBAL_DB.clear()
    from halo_forge.auth.dependency import reset_store_for_tests
    reset_store_for_tests(None)

    from fastapi.testclient import TestClient
    from halo_forge.public_api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    db_mod._GLOBAL_DB.clear()


def test_metrics_endpoint_returns_text(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    # Plain-text content type so Prometheus parsers consume it cleanly.
    assert "text/plain" in r.headers["content-type"].lower()
    body = r.text
    assert "# HELP halo_forge_runs_total" in body
    assert "halo_forge_runs_total" in body


def test_metrics_endpoint_skips_auth_for_loopback(client):
    """No token attached and no env override; loopback request still
    gets metrics so the local Prometheus sidecar works zero-config."""
    r = client.get("/metrics")
    assert r.status_code == 200


def test_metrics_endpoint_includes_build_info(client):
    r = client.get("/metrics")
    body = r.text
    assert "halo_forge_build_info" in body
    # Build-info is always 1; labels carry the metadata.
    assert "halo_forge_build_info" in body
