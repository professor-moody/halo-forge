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
    assert args.steps == 100
    assert args.warmup == 10


def test_measure_mlx_compile_reports_missing_mlx_cleanly():
    module = _load_script_module()
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        try:
            module._require_mlx()
        except SystemExit as exc:
            assert "MLX is not installed" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("_require_mlx should exit when mlx is unavailable")
