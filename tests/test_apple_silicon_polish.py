import os
import sys
from types import SimpleNamespace

import pytest


def test_explicit_cpu_accelerator_overrides_available_mps(monkeypatch):
    from halo_forge.utils import accelerator

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: True, is_built=lambda: True)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    monkeypatch.setenv("HALOFORGE_BACKEND", "cpu")
    assert accelerator.detect_gpu_kind() == accelerator.GPU_KIND_CPU

    monkeypatch.delenv("HALOFORGE_BACKEND")
    assert accelerator.detect_gpu_kind() == accelerator.GPU_KIND_MPS


def test_set_global_seed_routes_active_mlx_backend(monkeypatch):
    from halo_forge import backend as backend_pkg
    from halo_forge.runtime_determinism import set_global_seed

    calls: list[int] = []

    class FakeBackend:
        name = "mlx"

        def seed_all(self, seed: int) -> None:
            calls.append(seed)

    monkeypatch.setattr(backend_pkg, "get_backend", lambda: FakeBackend())

    assert set_global_seed("123") == 123
    assert calls == [123]


def test_set_global_seed_ignores_non_mlx_backend(monkeypatch):
    from halo_forge import backend as backend_pkg
    from halo_forge.runtime_determinism import set_global_seed

    calls: list[int] = []

    class FakeBackend:
        name = "mps"

        def seed_all(self, seed: int) -> None:
            calls.append(seed)

    monkeypatch.setattr(backend_pkg, "get_backend", lambda: FakeBackend())

    set_global_seed(9)
    assert calls == []


def test_mps_environment_defaults_preserve_overrides(monkeypatch):
    from halo_forge.backend.torch_mps import MPSBackend

    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")

    applied = MPSBackend().setup_environment()

    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0"
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert applied == {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"}


def test_replay_fingerprint_adds_best_effort_apple_fields(monkeypatch):
    import halo_forge.replay.manifest as manifest
    import halo_forge.utils.apple_chip as apple_chip
    from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry

    monkeypatch.setattr(manifest.sys, "platform", "darwin")
    monkeypatch.setattr(manifest.platform, "mac_ver", lambda: ("26.2", "", ""))
    monkeypatch.setattr(
        AppleSiliconTelemetry,
        "_detect_device_name",
        staticmethod(lambda: "Apple M3 Max"),
    )
    monkeypatch.setattr(apple_chip, "detect_metal_version", lambda: "Metal 4")

    fingerprint = manifest.EnvironmentFingerprint.capture()

    assert fingerprint.chip_name == "Apple M3 Max"
    assert fingerprint.chip_brand == "M3 Max"
    assert fingerprint.macos_version == "26.2"
    assert fingerprint.metal_version == "Metal 4"


def test_caffeinate_command_is_darwin_only_and_best_effort(monkeypatch):
    import halo_forge.utils.macos_runtime as macos_runtime

    monkeypatch.setattr(macos_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(macos_runtime.shutil, "which", lambda name: "/usr/bin/caffeinate")
    assert macos_runtime.caffeinate_command(["halo-forge", "sft", "train"]) == [
        "caffeinate",
        "-i",
        "-m",
        "-s",
        "--",
        "halo-forge",
        "sft",
        "train",
    ]

    monkeypatch.setattr(macos_runtime.shutil, "which", lambda name: None)
    assert macos_runtime.caffeinate_command(["cmd"]) == ["cmd"]

    monkeypatch.setattr(macos_runtime.sys, "platform", "linux")
    assert macos_runtime.caffeinate_command(["cmd"]) == ["cmd"]


def test_mps_fallback_counter_matches_and_expires():
    from halo_forge.telemetry.apple_silicon import MPSFallbackCounter

    now = 1000.0
    counter = MPSFallbackCounter(clock=lambda: now)

    assert counter.record_warning_line("aten::foo will fall back to run on the CPU")
    assert counter.record_warning_line("MPSFallback.mm: unsupported op")
    assert counter.record_warning_line("Set PYTORCH_ENABLE_MPS_FALLBACK=1")
    assert not counter.record_warning_line("ordinary training log")
    assert counter.count_last_60s() == 3

    now = 1061.0
    assert counter.count_last_60s() == 0


def test_apple_telemetry_sample_includes_fallback_count_and_chip(monkeypatch):
    from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry, get_mps_fallback_counter

    monkeypatch.setattr(
        AppleSiliconTelemetry,
        "_detect_device_name",
        staticmethod(lambda: "Apple M3 Max"),
    )
    monkeypatch.setattr(
        AppleSiliconTelemetry,
        "_mps_memory_gb",
        staticmethod(lambda: (None, None)),
    )
    counter = get_mps_fallback_counter()
    counter.clear()
    counter.record_warning_line(
        "aten::foo will fall back to run on the CPU"
    )

    sample = AppleSiliconTelemetry(backend_name="mps").sample()

    assert isinstance(sample.mps_to_cpu_fallbacks_60s, int)
    assert sample.mps_to_cpu_fallbacks_60s >= 1
    assert sample.chip is not None
    assert sample.chip["brand"] == "M3 Max"
    counter.clear()


@pytest.mark.parametrize(
    ("brand", "expected"),
    [
        ("Apple M3 Max", {"generation": 3, "variant": "Max", "brand": "M3 Max"}),
        ("Apple M5", {"generation": 5, "variant": "base", "brand": "M5"}),
        ("Apple M2 Ultra", {"generation": 2, "variant": "Ultra", "brand": "M2 Ultra"}),
    ],
)
def test_parse_chip_brand_known_chips(brand, expected):
    from halo_forge.utils.apple_chip import parse_chip_brand

    chip = parse_chip_brand(brand)
    assert chip is not None
    assert chip.generation == expected["generation"]
    assert chip.variant == expected["variant"]
    assert chip.brand == expected["brand"]


def test_parse_chip_brand_unknown_and_future_return_none():
    from halo_forge.utils.apple_chip import parse_chip_brand

    assert parse_chip_brand("Apple M6 Max") is None
    assert parse_chip_brand("Intel Core i9") is None


def test_mlx_neural_accelerator_capability_gate(monkeypatch):
    import halo_forge.backend.mlx as mlx_backend
    from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry

    monkeypatch.setattr(
        AppleSiliconTelemetry,
        "_detect_device_name",
        staticmethod(lambda: "Apple M5 Max"),
    )
    monkeypatch.setattr(mlx_backend.platform, "mac_ver", lambda: ("26.2", "", ""))
    assert mlx_backend.MLXBackend().capabilities.supports_neural_accelerators is True

    monkeypatch.setattr(mlx_backend.platform, "mac_ver", lambda: ("26.1", "", ""))
    assert mlx_backend.MLXBackend().capabilities.supports_neural_accelerators is False


def test_neural_accelerator_opt_in_validation_rejects_unsupported_backend():
    from halo_forge.backend.base import BackendCapabilities, BackendUnsupportedError
    from halo_forge.utils.neural_accelerators import validate_neural_accelerator_opt_in

    backend = SimpleNamespace(
        name="mps",
        capabilities=BackendCapabilities(
            name="mps",
            supports_bf16=True,
            supports_fp16=True,
            preferred_dtype_str="float16",
            supports_4bit=False,
            supports_8bit=False,
            supports_flash_attn=False,
            preferred_attn_impl="sdpa",
            supports_training=True,
            supports_peft=True,
            supports_neural_accelerators=False,
        ),
    )

    with pytest.raises(BackendUnsupportedError):
        validate_neural_accelerator_opt_in(
            SimpleNamespace(enable_neural_accelerators=True),
            backend=backend,
        )


def test_launch_context_preserves_no_caffeinate():
    from ui.services.launch_context import normalize_launch_args

    normalized = normalize_launch_args(
        "sft",
        {
            "model": "m",
            "dataset": "d",
            "output_dir": "out",
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "no_caffeinate": True,
        },
    )
    assert normalized["no_caffeinate"] is True
