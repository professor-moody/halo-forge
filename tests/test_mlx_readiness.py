"""MLX readiness probe contract tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_mlx_readiness_ready(monkeypatch):
    from halo_forge.backend import mlx_readiness as mod

    monkeypatch.setattr(
        mod,
        "_package_version",
        lambda name: {"mlx": "0.31.2", "mlx-lm": "0.31.3"}.get(name),
    )
    monkeypatch.setattr(mod, "_chip_info", lambda: {"brand": "M3 Max", "generation": 3})
    monkeypatch.setattr(mod, "_metal_device", lambda: {"model": "Apple M3 Max"})
    monkeypatch.setattr(
        mod,
        "_probe_mlx",
        lambda timeout: (0, '{"default_device":"Device(gpu, 0)","value":6.0}\n', ""),
    )

    readiness = mod.check_mlx_readiness()

    assert readiness.status == "ready"
    assert readiness.executable is True
    assert readiness.package_versions["mlx"] == "0.31.2"
    assert readiness.probe["value"] == 6.0


def test_mlx_readiness_enriches_parsed_chip_with_metal_gpu_cores(monkeypatch):
    from halo_forge.backend import mlx_readiness as mod

    monkeypatch.setattr(
        mod,
        "_package_version",
        lambda name: {"mlx": "0.31.2", "mlx-lm": "0.31.3"}.get(name),
    )
    monkeypatch.setattr(
        mod,
        "_chip_info",
        lambda: {
            "brand": "M4 Max",
            "generation": 4,
            "variant": "Max",
            "gpu_cores": None,
            "nominal_memory_bandwidth_gbps": 546,
            "raw_brand": "Apple M4 Max",
        },
    )
    monkeypatch.setattr(
        mod,
        "_metal_device",
        lambda: {"model": "Apple M4 Max", "gpu_cores": 32, "metal_supported": True},
    )
    monkeypatch.setattr(
        mod,
        "_probe_mlx",
        lambda timeout: (0, '{"default_device":"Device(gpu, 0)","value":6.0}\n', ""),
    )

    readiness = mod.check_mlx_readiness()

    assert readiness.chip is not None
    assert readiness.chip["brand"] == "M4 Max"
    assert readiness.chip["gpu_cores"] == 32
    assert readiness.metal_device is not None
    assert readiness.metal_device["metal_supported"] is True


def test_mlx_readiness_infers_metal_supported_from_gpu_probe(monkeypatch):
    from halo_forge.backend import mlx_readiness as mod

    monkeypatch.setattr(
        mod,
        "_package_version",
        lambda name: {"mlx": "0.31.2", "mlx-lm": "0.31.3"}.get(name),
    )
    monkeypatch.setattr(mod, "_chip_info", lambda: {"brand": "M4 Max", "generation": 4})
    monkeypatch.setattr(
        mod,
        "_metal_device",
        lambda: {"model": "Apple M4 Max", "gpu_cores": 32, "metal_supported": None},
    )
    monkeypatch.setattr(
        mod,
        "_probe_mlx",
        lambda timeout: (0, '{"default_device":"Device(gpu, 0)","value":6.0}\n', ""),
    )

    readiness = mod.check_mlx_readiness()

    assert readiness.executable is True
    assert readiness.metal_device is not None
    assert readiness.metal_device["metal_supported"] is True


def test_metal_supported_parses_system_profiler_values():
    from halo_forge.backend import mlx_readiness as mod

    assert mod._metal_supported("spdisplays_supported") is True
    assert mod._metal_supported("Metal 3") is True
    assert mod._metal_supported("Metal: Supported") is True
    assert mod._metal_supported("Unsupported") is False
    assert mod._metal_supported(None) is None


def test_mlx_readiness_unavailable_for_no_metal(monkeypatch):
    from halo_forge.backend import mlx_readiness as mod

    monkeypatch.setattr(mod, "_package_version", lambda name: "0.31.2")
    monkeypatch.setattr(mod, "_chip_info", lambda: None)
    monkeypatch.setattr(mod, "_metal_device", lambda: None)
    monkeypatch.setattr(
        mod,
        "_probe_mlx",
        lambda timeout: (1, "", "[metal::load_device] No Metal device available"),
    )

    readiness = mod.check_mlx_readiness()

    assert readiness.status == "unavailable"
    assert readiness.executable is False
    assert "normal Terminal" in readiness.suggested_fixes[0]


def test_mlx_readiness_missing_package_guidance(monkeypatch):
    from halo_forge.backend import mlx_readiness as mod

    monkeypatch.setattr(
        mod,
        "_package_version",
        lambda name: None if name == "mlx-lm" else "0.31.2",
    )
    monkeypatch.setattr(mod, "_chip_info", lambda: None)
    monkeypatch.setattr(mod, "_metal_device", lambda: None)

    readiness = mod.check_mlx_readiness()

    assert readiness.status == "unavailable"
    assert readiness.executable is False
    assert "mlx-lm" in readiness.errors[0]
    assert "pip install -e" in readiness.suggested_fixes[0]


def test_doctor_mlx_json_exit_code(monkeypatch, capsys):
    from halo_forge.backend.mlx_readiness import MLXReadiness
    from halo_forge import cli

    monkeypatch.setattr(
        "halo_forge.backend.mlx_readiness.check_mlx_readiness",
        lambda: MLXReadiness(
            status="unavailable",
            executable=False,
            package_versions={"mlx": "0.31.2", "mlx-lm": "0.31.3"},
            chip=None,
            macos_version="26.3.1",
            metal_device=None,
            errors=["No Metal device available"],
        ),
    )

    with pytest.raises(SystemExit) as exc:
        cli.cmd_doctor(SimpleNamespace(doctor_command="mlx", json=True))

    assert exc.value.code == 2
    assert '"status": "unavailable"' in capsys.readouterr().out


def test_public_backend_includes_mlx_readiness(monkeypatch, tmp_path):
    from halo_forge.backend.mlx_readiness import MLXReadiness

    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setattr(
        "halo_forge.backend.mlx_readiness.check_mlx_readiness",
        lambda timeout_seconds=5.0: MLXReadiness(
            status="ready",
            executable=True,
            package_versions={"mlx": "0.31.2", "mlx-lm": "0.31.3"},
            chip={"brand": "M3 Max"},
            macos_version="26.3.1",
            metal_device={"model": "Apple M3 Max"},
        ),
    )

    from fastapi.testclient import TestClient
    from halo_forge.public_api.app import create_app

    with TestClient(create_app()) as client:
        body = client.get("/api/public/backend").json()

    assert body["mlx_readiness"]["status"] == "ready"
    assert body["mlx_readiness"]["executable"] is True


def test_preflight_warns_for_mlx_when_unavailable(monkeypatch, tmp_path):
    from halo_forge.backend.mlx_readiness import MLXReadiness
    from halo_forge.public_api.service import PublicApiService

    service = PublicApiService()
    monkeypatch.setattr(
        service,
        "_mlx_readiness_snapshot",
        lambda: MLXReadiness(
            status="unavailable",
            executable=False,
            package_versions={"mlx": "0.31.2", "mlx-lm": "0.31.3"},
            chip=None,
            macos_version=None,
            metal_device=None,
            errors=["No Metal device available"],
            suggested_fixes=["Run from a normal Terminal session with GPU access."],
        ).to_dict(),
    )

    body = service.preflight_training(
        {
            "mode": "sft",
            "model": "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
            "dataset": "codealpaca",
            "output_dir": str(tmp_path / "out"),
            "accelerator": "mlx",
            "epochs": 1,
            "batch_size": 1,
        }
    )

    assert any("MLX readiness is unavailable" in warning for warning in body["warnings"])
    assert any("normal Terminal" in fix for fix in body["suggested_fixes"])
