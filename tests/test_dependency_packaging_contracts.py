#!/usr/bin/env python3
"""Dependency and packaging contract regression tests."""

from __future__ import annotations

import re
from pathlib import Path

import tomllib


def _package_name(spec: str) -> str:
    """Extract normalized package name from dependency spec string."""
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", spec or "")
    if not match:
        return ""
    return match.group(1).lower().replace("_", "-")


def _active_requirements(path: Path) -> set[str]:
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        spec = line.split(";", 1)[0].strip()
        name = _package_name(spec)
        if name:
            names.add(name)
    return names


def test_requirements_mirror_pyproject_core_dependencies():
    """requirements.txt should contain all baseline project dependencies."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project_deps = pyproject["project"]["dependencies"]
    core_names = {_package_name(spec) for spec in project_deps}
    req_names = _active_requirements(Path("requirements.txt"))

    missing = sorted(name for name in core_names if name and name not in req_names)
    assert missing == []


def test_expected_modality_and_runtime_extras_are_declared():
    """Install profiles for active modalities/runtime should be explicitly named."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    required = {"audio", "vlm", "reasoning", "agentic", "benchmark", "inference", "dev"}
    assert required.issubset(set(extras.keys()))

    # Keep compatibility alias for older docs/scripts.
    assert extras["benchmark"] == extras["vlm-benchmark"]


def test_quantization_helper_guards_tqdm_scope_and_quantization_dependencies():
    """quantize_model_simple should have deterministic imports/guards for optional deps."""
    source = Path("halo_forge/inference/quantization.py").read_text(encoding="utf-8")
    match = re.search(
        r"def quantize_model_simple\((?:.|\n)*?(?=\n\ndef |\Z)",
        source,
        re.MULTILINE,
    )
    assert match is not None
    source = match.group(0)
    assert "from tqdm import tqdm" in source
    assert "bitsandbytes is required for int4 quantization" in source
    assert "bitsandbytes is required for int8 quantization" in source
    assert "Unsupported precision" in source


def test_packaging_contract_has_no_legacy_tui_entry_points():
    """Packaging metadata should not publish legacy TUI scripts/modules."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert set(scripts.keys()) == {"halo-forge", "halo-forge-desktop-runtime"}
    assert scripts["halo-forge-desktop-runtime"] == "halo_forge.desktop_runtime:main"
    assert all("tui" not in value.lower() for value in scripts.values())
