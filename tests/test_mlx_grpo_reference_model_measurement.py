from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "measure_mlx_grpo_reference_model.py"
    spec = importlib.util.spec_from_file_location("measure_mlx_grpo_reference_model", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_model_grpo_measurement_reports_unavailable(monkeypatch):
    module = _load_module()

    readiness = SimpleNamespace(
        executable=False,
        to_dict=lambda: {
            "status": "unavailable",
            "executable": False,
            "errors": ["No Metal device available"],
        },
    )
    monkeypatch.setattr(module, "check_mlx_readiness", lambda: readiness)

    args = module.build_parser().parse_args([])
    result = module.run_measurement(args)

    assert result["status"] == "unavailable"
    assert result["decision"] == "measurement_only"
    assert result["model"] == module.DEFAULT_MODEL
    assert result["readiness"]["status"] == "unavailable"


def test_reference_model_grpo_measurement_detects_metal_unavailable_error():
    module = _load_module()

    assert module._is_metal_unavailable(RuntimeError("[metal::load_device] No Metal device available"))
