#!/usr/bin/env python3
"""Phase 14 all-module parity regression tests."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from halo_forge.all_module_readiness import (
    ALL_MODULES,
    build_all_module_readiness_report,
    validate_all_module_readiness_payload,
)


def test_all_module_readiness_schema_validation_pass_and_fail():
    """All-module readiness schema validator should catch missing keys and accept valid payloads."""
    report = build_all_module_readiness_report(module_entries={}, seed=42, source="script")
    payload = report.to_dict()
    errors = validate_all_module_readiness_payload(payload)
    assert errors == []
    assert "launch_blocked" in payload["modules"]["config"]
    assert "issue_class" in payload["modules"]["config"]
    assert "action_hint" in payload["modules"]["config"]

    broken = dict(payload)
    broken.pop("modules")
    failed = validate_all_module_readiness_payload(broken)
    assert any("missing top-level key: modules" in err for err in failed)


def test_all_module_matrix_script_strict_and_non_strict_behavior(tmp_path):
    """All-module script should emit ALL_READY lines and enforce strict fail behavior."""
    script = Path("scripts/run_all_module_matrix.py")
    assert script.exists()

    report_file = tmp_path / "all_modules_readiness.v1.json"

    non_strict = subprocess.run(
        [
            sys.executable,
            str(script),
            "--write-report",
            "--report-file",
            str(report_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert non_strict.returncode == 0
    assert "ALL_READY module=config" in non_strict.stdout
    assert report_file.exists()

    strict_pass = subprocess.run(
        [
            sys.executable,
            str(script),
            "--fixture-pack",
            "v1",
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict_pass.returncode == 0
    assert "ALL_READY module=ui_ops" in strict_pass.stdout


def test_all_module_fixture_pack_corruption_fails_strict(tmp_path):
    """Corrupted all-module fixture pack should fail strict mode."""
    script = Path("scripts/run_all_module_matrix.py")
    fixture_root = Path("tests/fixtures/all_modules/v1")
    corrupted_root = tmp_path / "all_modules_corrupt"
    shutil.copytree(fixture_root, corrupted_root)
    (corrupted_root / "raft" / "latest_checkpoint.json").unlink()

    strict_fail = subprocess.run(
        [
            sys.executable,
            str(script),
            "--fixture-pack",
            str(corrupted_root),
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict_fail.returncode == 1
    assert "ERROR: failing modules:" in strict_fail.stdout


def test_cli_surface_exposes_all_modules_level_and_flags():
    """CLI parser should include all-modules level and related flags."""
    cli_source = Path("halo_forge/cli.py").read_text(encoding="utf-8")
    assert "all-modules" in cli_source
    assert "all-module-qualification" in cli_source
    assert "run_all_modules" in cli_source
    assert "walkthroughs" in cli_source
    assert "--profile" in cli_source
    assert "--module" in cli_source


def test_ui_surfaces_include_all_module_readiness_hooks():
    """UI pages/services should expose all-module readiness integration hooks."""
    service_source = Path("ui/services/ops_readiness_service.py").read_text(encoding="utf-8")
    assert "get_effective_all_module_readiness" in service_source
    assert "load_all_module_readiness_report" in service_source

    dashboard_source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "get_dashboard_hub_service" in dashboard_source
    assert "_render_modality_readiness_summary" in dashboard_source

    research_hub_source = Path("ui/pages/research_hub.py").read_text(encoding="utf-8")
    assert "ALL_MODULES" in research_hub_source
    assert "get_effective_all_module_readiness" in research_hub_source

    training_source = Path("ui/pages/training.py").read_text(encoding="utf-8")
    assert "_render_all_module_readiness_banner" in training_source

    ops_console_source = Path("ui/pages/ops_console.py").read_text(encoding="utf-8")
    assert "get_effective_all_module_readiness" in ops_console_source
    assert "module_ops_service" in ops_console_source


def test_all_module_fixture_pack_contains_expected_directories():
    """Fixture pack should include deterministic module directories."""
    fixture_root = Path("tests/fixtures/all_modules/v1")
    assert fixture_root.exists()
    for module in ALL_MODULES:
        if module == "ui_ops":
            continue
        assert (fixture_root / module).exists(), f"missing fixture module directory: {module}"
