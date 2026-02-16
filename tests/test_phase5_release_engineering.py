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
    assert "tests/test_phase10_non_code_ui_readiness_gates.py" in content
    assert "tests/test_phase11_ops_module_ui_expansion.py" in content
    assert "tests/test_phase12_ops_e2e_launch_reliability.py" in content
    assert "tests/test_phase13_ops_dataset_burnin_baselines.py" in content
    assert "tests/test_phase14_all_module_parity.py" in content
    assert "tests/test_phase16_all_module_ui_execution_parity.py" in content
    assert "tests/test_phase17_ui_execution_truth_and_professional_hardening.py" in content
    assert "tests/test_phase18_all_module_qualification_orchestration.py" in content
    assert "tests/test_phase20_all_module_bootstrap_enablement.py" in content
    assert "tests/test_phase21_all_module_live_execution_closure.py" in content
    assert "tests/test_phase22_dashboard_ops_hub_and_discoverability.py" in content
    assert "tests/test_phase23_training_first_ux_reset.py" in content
    assert "scripts/generate_modality_baseline.py" in content
    assert "scripts/run_ops_module_matrix.py" in content
    assert "scripts/run_ops_e2e_reliability.py" in content
    assert "scripts/run_ops_dataset_burnin.py" in content
    assert "scripts/run_all_module_matrix.py" in content
    assert "scripts/run_all_module_qualification.py" in content
    assert "scripts/run_all_module_bootstrap.py" in content
    assert "scripts/run_all_module_live_matrix.py" in content
    assert "--fixture-pack v1" in content
    assert "tests/baselines/modality_runtime_baseline.v1.json" in content
    assert "ops-readiness-reports" in content
    assert "ops_e2e_launch_reliability.v1.json" in content
    assert "ops_dataset_burnin.v1.json" in content
    assert "all_modules_readiness.v1.json" in content
    assert "all_module_qualification.v1.json" in content
    assert "all_module_bootstrap.v1.json" in content
    assert "all_module_live_execution.v1.json" in content

    nightly_workflow = Path(".github/workflows/nightly_ops_readiness.yml")
    assert nightly_workflow.exists()
    nightly_content = nightly_workflow.read_text(encoding="utf-8")
    assert "schedule:" in nightly_content
    assert "workflow_dispatch:" in nightly_content
    assert "scripts/run_ops_module_matrix.py" in nightly_content
    assert "--fixture-pack v1" in nightly_content
    assert "--strict" in nightly_content

    nightly_e2e_workflow = Path(".github/workflows/nightly_ops_e2e_reliability.yml")
    assert nightly_e2e_workflow.exists()
    nightly_e2e_content = nightly_e2e_workflow.read_text(encoding="utf-8")
    assert "schedule:" in nightly_e2e_content
    assert "workflow_dispatch:" in nightly_e2e_content
    assert "scripts/run_ops_e2e_reliability.py" in nightly_e2e_content
    assert "--fixture-pack v1" in nightly_e2e_content
    assert "--strict" in nightly_e2e_content

    nightly_burnin_workflow = Path(".github/workflows/nightly_ops_dataset_burnin.yml")
    assert nightly_burnin_workflow.exists()
    nightly_burnin_content = nightly_burnin_workflow.read_text(encoding="utf-8")
    assert "schedule:" in nightly_burnin_content
    assert "workflow_dispatch:" in nightly_burnin_content
    assert "scripts/run_ops_dataset_burnin.py" in nightly_burnin_content
    assert "--strict" in nightly_burnin_content
    assert "--compare-baseline" in nightly_burnin_content
    assert "tests/baselines/ops_dataset_burnin_baseline.v1.json" in nightly_burnin_content

    nightly_all_modules_workflow = Path(".github/workflows/nightly_all_module_readiness.yml")
    assert nightly_all_modules_workflow.exists()
    nightly_all_modules_content = nightly_all_modules_workflow.read_text(encoding="utf-8")
    assert "schedule:" in nightly_all_modules_content
    assert "workflow_dispatch:" in nightly_all_modules_content
    assert "scripts/run_all_module_matrix.py" in nightly_all_modules_content
    assert "--fixture-pack v1" in nightly_all_modules_content
    assert "--strict" in nightly_all_modules_content

    nightly_qualification_workflow = Path(".github/workflows/nightly_all_module_qualification.yml")
    assert nightly_qualification_workflow.exists()
    nightly_qualification_content = nightly_qualification_workflow.read_text(encoding="utf-8")
    assert "schedule:" in nightly_qualification_content
    assert "workflow_dispatch:" in nightly_qualification_content
    assert "scripts/run_all_module_qualification.py" in nightly_qualification_content
    assert "--qualification-profile fixture-v1" in nightly_qualification_content
    assert "--strict" in nightly_qualification_content
    assert "--compare-baseline" in nightly_qualification_content
    assert "tests/baselines/all_module_qualification_baseline.v1.json" in nightly_qualification_content

    nightly_live_workflow = Path(".github/workflows/nightly_all_module_live_execution.yml")
    assert nightly_live_workflow.exists()
    nightly_live_content = nightly_live_workflow.read_text(encoding="utf-8")
    assert "schedule:" in nightly_live_content
    assert "workflow_dispatch:" in nightly_live_content
    assert "scripts/run_all_module_live_matrix.py" in nightly_live_content
    assert "--live-profile live-smoke-v1" in nightly_live_content
    assert "--strict" in nightly_live_content

    # Phase 7K walkthroughs are local/operator flows only; no CI gating changes.
    assert "run_all_module_walkthroughs.py" not in content
    assert not Path(".github/workflows/nightly_all_module_walkthroughs.yml").exists()
    assert not Path(".github/workflows/nightly_walkthroughs.yml").exists()


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
