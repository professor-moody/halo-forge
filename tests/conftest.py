"""pytest fixtures and backend-aware test gating.

Tests can mark themselves with `@pytest.mark.requires_rocm` /
`@pytest.mark.requires_cuda` / `@pytest.mark.requires_mps` /
`@pytest.mark.requires_mlx` and they will be auto-skipped on hosts that don't
match. Detection routes through `halo_forge.utils.accelerator.detect_gpu_kind`
so the answer matches what the trainer code will see at runtime.
"""

from __future__ import annotations

import importlib.util

import pytest


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
    return importlib.util.find_spec("mlx") is not None


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
        "requires_mlx: skip unless the `mlx` package is importable",
    )
    config.addinivalue_line(
        "markers",
        "requires_accelerator: skip if no GPU/MPS accelerator is detected",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip backend-gated tests on hosts that don't match."""
    skip_rocm = pytest.mark.skip(reason=f"requires ROCm gfx1151; backend is {_BACKEND}")
    skip_cuda = pytest.mark.skip(reason=f"requires CUDA; backend is {_BACKEND}")
    skip_mps = pytest.mark.skip(reason=f"requires Apple Silicon MPS; backend is {_BACKEND}")
    skip_mlx = pytest.mark.skip(reason="requires the `mlx` package")
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
