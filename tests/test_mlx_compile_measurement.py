from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "measure_mlx_compile.py"
    spec = importlib.util.spec_from_file_location("measure_mlx_compile", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_measure_mlx_compile_imports_without_mlx():
    module = _load_script_module()
    parser = module.build_parser()
    args = parser.parse_args([])
    assert args.batch_size == 32
    assert args.batch_sizes == "32,128,512"
    assert args.candidate == "all"
    assert args.steps == 100
    assert args.warmup == 10
    assert module._batch_sizes(args) == [32, 128, 512]
    assert module._selected_candidates(args) == list(module.CANDIDATES)
    assert "dpo_reference_free_ipo" in module.CANDIDATES
    assert "dpo_reference_model_hinge" in module.CANDIDATES
    assert "dpo_reference_free_kto_pair" in module.CANDIDATES
    assert "dpo_reference_model_kto_pair" in module.CANDIDATES


def test_measure_mlx_compile_reports_missing_mlx_cleanly(monkeypatch):
    module = _load_script_module()

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlx.core":
            raise ImportError("No module named 'mlx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    try:
        module._require_mlx()
    except SystemExit as exc:
        assert "MLX is not installed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("_require_mlx should exit when mlx is unavailable")


def test_measure_mlx_compile_reports_metal_unavailable_as_structured_result():
    module = _load_script_module()
    parser = module.build_parser()
    args = parser.parse_args(["--batch-size", "16"])
    result = module._unavailable_result(
        args,
        "[metal::load_device] No Metal device available",
    )
    assert result["status"] == "unavailable"
    assert result["shapes"] == [{"batch_size": 32}, {"batch_size": 128}, {"batch_size": 512}]
    assert "No Metal device available" in result["reason"]
    assert module._is_metal_unavailable(RuntimeError(result["reason"])) is True


def test_measure_mlx_compile_single_batch_fallback():
    module = _load_script_module()
    parser = module.build_parser()
    args = parser.parse_args(["--batch-sizes", "", "--batch-size", "16"])

    assert module._batch_sizes(args) == [16]
