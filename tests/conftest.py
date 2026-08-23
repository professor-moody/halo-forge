"""pytest fixtures and backend-aware test gating.

Tests can mark themselves with `@pytest.mark.requires_rocm` /
`@pytest.mark.requires_cuda` / `@pytest.mark.requires_mps` /
`@pytest.mark.requires_mlx` and they will be auto-skipped on hosts that don't
match. Detection routes through `halo_forge.utils.accelerator.detect_gpu_kind`
so the answer matches what the trainer code will see at runtime.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# Establish a disposable home before test modules are imported. A few modules
# resolve default paths at import time, so a fixture alone is too late to keep
# collection from observing or writing an operator's real ~/.halo-forge state.
_SESSION_TEST_ROOT = Path(tempfile.mkdtemp(prefix="halo-forge-pytest-"))
_ORIGINAL_HOME = os.environ.get("HOME")
_ORIGINAL_CARGO_HOME = os.environ.get("CARGO_HOME")
_ORIGINAL_RUSTUP_HOME = os.environ.get("RUSTUP_HOME")
os.environ["HOME"] = str(_SESSION_TEST_ROOT / "home")
Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)

# A rustup-backed cargo executable needs the installed toolchain after HOME is
# isolated. Keep that location read-only in practice, while all Cargo cache and
# build writes go to the disposable session root.
_installed_rustup_value = _ORIGINAL_RUSTUP_HOME or (
    str(Path(_ORIGINAL_HOME) / ".rustup") if _ORIGINAL_HOME else None
)
if _installed_rustup_value and Path(_installed_rustup_value).is_dir():
    os.environ["RUSTUP_HOME"] = _installed_rustup_value
_test_cargo_home = _SESSION_TEST_ROOT / "cargo"
_test_cargo_home.mkdir(parents=True, exist_ok=True)
os.environ["CARGO_HOME"] = str(_test_cargo_home)


_STATE_ENVIRONMENTS = {
    "HALOFORGE_RUN_DB_PATH": "runs.db",
    "HALOFORGE_DATASET_ROOT": "datasets",
    "HALOFORGE_RUNTIME_ROOT": "runtimes",
    "HALOFORGE_EVALUATION_ROOT": "evaluations",
    "HALOFORGE_ARTIFACT_ROOT": "artifacts",
    "HALOFORGE_EVIDENCE_ROOT": "evidence",
    "HALOFORGE_REVIEW_ROOT": "reviews",
    "HALOFORGE_VERIFIER_CALIBRATION_ROOT": "verifier-calibrations",
    "HALOFORGE_TRAINING_SIGNAL_ROOT": "training-signals",
}
_ORIGINAL_STATE_ENVIRONMENTS = {
    name: os.environ.get(name) for name in _STATE_ENVIRONMENTS
}
for _name, _relative_path in _STATE_ENVIRONMENTS.items():
    _target = _SESSION_TEST_ROOT / _relative_path
    if _target.suffix != ".db":
        _target.mkdir(parents=True, exist_ok=True)
    os.environ[_name] = str(_target)


def _clear_process_state() -> None:
    """Stop workers and close cached databases between isolated test roots."""
    try:
        from halo_forge.workstation_jobs import supervisor as supervisor_module

        with supervisor_module._SUPERVISOR_LOCK:
            supervisors = list(supervisor_module._SUPERVISORS.values())
            supervisor_module._SUPERVISORS.clear()
        for supervisor in supervisors:
            supervisor.stop(timeout=1.0)
    except Exception:
        pass

    try:
        from halo_forge.run_db import db as db_module

        with db_module._GLOBAL_DB_LOCK:
            databases = list(db_module._GLOBAL_DB.values())
            db_module._GLOBAL_DB.clear()
        for database in databases:
            try:
                database.close()
            except Exception:
                pass
    except Exception:
        pass

    try:
        from halo_forge.auth.dependency import reset_store_for_tests

        reset_store_for_tests(None)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _isolate_halo_forge_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Give every test a private home, catalog, and managed artifact roots."""
    _clear_process_state()
    state_root = tmp_path / "halo-forge-state"
    home = state_root / "home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    for name, relative_path in _STATE_ENVIRONMENTS.items():
        target = state_root / relative_path
        if target.suffix != ".db":
            target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv(name, str(target))
    yield
    _clear_process_state()


def _detect_gpu_kind_safe() -> str:
    """Detect accelerator without forcing a torch import at collection time."""
    try:
        from halo_forge.utils.accelerator import detect_gpu_kind
    except Exception:
        return "cpu"
    try:
        return detect_gpu_kind()
    except Exception:
        return "cpu"


def _have_mlx() -> bool:
    if importlib.util.find_spec("mlx") is None:
        return False
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import mlx.core as mx; x = mx.array([1.0]); mx.eval(x)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return probe.returncode == 0


_BACKEND = _detect_gpu_kind_safe()
_MLX_AVAILABLE = _have_mlx()


def pytest_configure(config: pytest.Config) -> None:
    """Register backend-gating markers."""
    config.addinivalue_line(
        "markers",
        "requires_rocm: skip unless an AMD ROCm gfx1151 (Strix Halo) host",
    )
    config.addinivalue_line(
        "markers",
        "requires_cuda: skip unless an NVIDIA CUDA host",
    )
    config.addinivalue_line(
        "markers",
        "requires_mps: skip unless an Apple Silicon MPS host",
    )
    config.addinivalue_line(
        "markers",
        "requires_mlx: skip unless MLX can execute a tiny array",
    )
    config.addinivalue_line(
        "markers",
        "requires_accelerator: skip if no GPU/MPS accelerator is detected",
    )


def pytest_unconfigure(config: pytest.Config) -> None:
    """Remove the collection-time test home after the pytest process exits."""
    del config
    _clear_process_state()
    if _ORIGINAL_HOME is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = _ORIGINAL_HOME
    for name, value in (
        ("CARGO_HOME", _ORIGINAL_CARGO_HOME),
        ("RUSTUP_HOME", _ORIGINAL_RUSTUP_HOME),
    ):
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    for name, value in _ORIGINAL_STATE_ENVIRONMENTS.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    shutil.rmtree(_SESSION_TEST_ROOT, ignore_errors=True)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip backend-gated tests on hosts that don't match."""
    skip_rocm = pytest.mark.skip(reason=f"requires ROCm gfx1151; backend is {_BACKEND}")
    skip_cuda = pytest.mark.skip(reason=f"requires CUDA; backend is {_BACKEND}")
    skip_mps = pytest.mark.skip(reason=f"requires Apple Silicon MPS; backend is {_BACKEND}")
    skip_mlx = pytest.mark.skip(reason="requires an executable MLX runtime")
    skip_acc = pytest.mark.skip(reason="requires any non-CPU accelerator")

    for item in items:
        if "requires_rocm" in item.keywords and _BACKEND != "rocm_gfx1151":
            item.add_marker(skip_rocm)
        if "requires_cuda" in item.keywords and _BACKEND != "cuda":
            item.add_marker(skip_cuda)
        if "requires_mps" in item.keywords and _BACKEND != "mps":
            item.add_marker(skip_mps)
        if "requires_mlx" in item.keywords and not _MLX_AVAILABLE:
            item.add_marker(skip_mlx)
        if "requires_accelerator" in item.keywords and _BACKEND == "cpu":
            item.add_marker(skip_acc)
