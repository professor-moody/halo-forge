#!/usr/bin/env python3
"""Phase 5 release-engineering regression guards."""

import shutil
import subprocess
from pathlib import Path


def test_ci_workflow_exists_with_compile_and_core_regression_steps():
    """Repository should include a CI workflow covering compile + core regression tests."""
    workflow = Path(".github/workflows/ci.yml")
    assert workflow.exists()

    content = workflow.read_text(encoding="utf-8")
    assert "python -m compileall -q halo_forge ui tests" in content
    assert "tests/test_phase0_stabilization.py" in content
    assert "tests/test_phase1_hardening.py" in content
    assert "tests/test_phase2_consistency_cleanup.py" in content
    assert "tests/test_phase3_truth_in_advertising.py" in content
    assert "tests/test_phase4_ui_observability_consolidation.py" in content
    assert "tests/test_benchmark_results_trust_matrix.py" in content
    assert "tests/test_training_pipeline_contracts.py" in content
    assert "tests/test_dependency_packaging_contracts.py" in content
    assert "tests/test_runtime_surface_alignment.py" in content
    assert "tests/test_phase6_modality_graduation.py" in content
    assert "tests/test_phase7_modality_reliability_gates.py" in content
    assert "tests/test_phase7b_modality_e2e_surface.py" in content
    assert "tests/test_phase7c_modality_baseline_drift.py" in content
    assert "tests/test_phase8_ui_ops_parity_relaunch.py" in content
    assert "tests/test_phase9_non_code_modality_research_matrix.py" in content
    assert "scripts/generate_modality_baseline.py" in content
    assert "tests/baselines/modality_runtime_baseline.v1.json" in content


def test_modality_tests_use_importorskip_for_optional_heavy_dependencies():
    """Heavy modality tests should skip cleanly when torch/numpy are unavailable."""
    expected_markers = {
        "tests/test_inference.py": "pytest.importorskip(\"torch\")",
        "tests/test_agentic.py": "pytest.importorskip(\"torch\")",
        "tests/test_audio.py": "pytest.importorskip(\"numpy\")",
        "tests/test_reasoning.py": "pytest.importorskip(\"numpy\")",
        "tests/test_vlm.py": "pytest.importorskip(\"numpy\")",
    }
    for rel_path, marker in expected_markers.items():
        content = Path(rel_path).read_text(encoding="utf-8")
        assert marker in content


def test_release_surface_excludes_legacy_tui_and_textual_dependency():
    """Legacy TUI should not be part of tracked source or active dependency contracts."""
    if not shutil.which("git"):
        return

    tracked_tui = subprocess.run(
        ["git", "ls-files", "halo_forge/tui"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked_tui.returncode == 0
    assert tracked_tui.stdout.strip() == ""

    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    active_lines = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any("textual" in line for line in active_lines)

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8").lower()
    assert "textual" not in pyproject
    assert "halo_forge.tui" not in pyproject
