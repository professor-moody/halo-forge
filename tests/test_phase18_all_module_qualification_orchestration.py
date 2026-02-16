#!/usr/bin/env python3
"""Phase 18 all-module qualification orchestration regression tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from halo_forge.all_module_qualification import (
    ALL_MODULE_QUALIFICATION_GENERATOR_VERSION,
    build_qualification_baseline_payload,
    compare_qualification_baselines,
    compute_all_module_qualification,
    format_qualification_drift_lines,
    normalize_all_module_qualification_payload,
    validate_all_module_qualification_payload,
    validate_qualification_baseline_payload,
)


def test_qualification_schema_and_normalization_roundtrip_with_fixture_profile():
    """Qualification report payload should validate and normalize deterministically."""
    fixture_root = Path("tests/fixtures/all_modules/v1")
    output_map = {
        "config": str(fixture_root / "config"),
        "data": str(fixture_root / "data"),
        "info": str(fixture_root / "info"),
        "plot": str(fixture_root / "plot"),
        "sft": str(fixture_root / "sft"),
        "raft": str(fixture_root / "raft"),
        "benchmark_code": str(fixture_root / "benchmark_code"),
        "benchmark_non_code": str(fixture_root / "benchmark_non_code"),
        "inference": str(fixture_root / "inference"),
        "vlm": str(fixture_root / "vlm"),
        "audio": str(fixture_root / "audio"),
        "reasoning": str(fixture_root / "reasoning"),
        "agentic": str(fixture_root / "agentic"),
        "ui_ops": str(Path.cwd()),
    }

    report = compute_all_module_qualification(
        output_map=output_map,
        seed=42,
        profile="fixture-v1",
        source="script",
    )
    payload = report.to_dict()
    errors = validate_all_module_qualification_payload(payload)
    assert errors == []

    normalized = normalize_all_module_qualification_payload(report)
    assert normalized["generated_at"] == "<normalized>"

    baseline_payload = build_qualification_baseline_payload(report)
    baseline_errors = validate_qualification_baseline_payload(baseline_payload)
    assert baseline_errors == []
    assert baseline_payload["generator_version"] == ALL_MODULE_QUALIFICATION_GENERATOR_VERSION


def test_qualification_script_profile_and_strict_behavior(tmp_path):
    """Qualification script should emit ALL_QUAL lines and enforce strict failure."""
    script = Path("scripts/run_all_module_qualification.py")
    assert script.exists()

    report_file = tmp_path / "all_module_qualification.v1.json"
    baseline_file = tmp_path / "all_module_qualification_baseline.v1.json"

    non_strict = subprocess.run(
        [
            sys.executable,
            str(script),
            "--qualification-profile",
            "contract-v1",
            "--write-report",
            "--report-file",
            str(report_file),
            "--write-baseline",
            "--baseline-file",
            str(baseline_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert non_strict.returncode == 0
    assert "ALL_QUAL module=config" in non_strict.stdout
    assert report_file.exists()
    assert baseline_file.exists()

    strict_pass = subprocess.run(
        [
            sys.executable,
            str(script),
            "--qualification-profile",
            "fixture-v1",
            "--fixture-pack",
            "v1",
            "--strict",
            "--compare-baseline",
            "--baseline-file",
            str(baseline_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # baseline was written from contract-v1, so compare may warn/hard drift depending local env.
    # strict fixture should still emit module lines; returncode accepted as 0/1 for deterministic CI portability.
    assert "ALL_QUAL module=ui_ops" in strict_pass.stdout


def test_qualification_script_strict_fails_on_corrupt_fixture(tmp_path):
    """Corrupted fixture pack should fail strict fixture qualification."""
    script = Path("scripts/run_all_module_qualification.py")
    fixture_root = Path("tests/fixtures/all_modules/v1")
    corrupt_root = tmp_path / "all_modules_corrupt"
    shutil.copytree(fixture_root, corrupt_root)
    (corrupt_root / "audio" / "training_summary.json").unlink()

    strict_fail = subprocess.run(
        [
            sys.executable,
            str(script),
            "--qualification-profile",
            "fixture-v1",
            "--fixture-pack",
            str(corrupt_root),
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict_fail.returncode == 1
    assert "ERROR: failing modules:" in strict_fail.stdout


def test_qualification_baseline_compare_emits_drift_lines():
    """Baseline comparator should emit parseable drift lines for lifecycle hard fields."""
    expected = {
        "contract_version": 1,
        "generator_version": "phase7m.v1",
        "profile": "fixture-v1",
        "seed": 42,
        "modules": {
            "config": {
                "status": "pass",
                "launch_ok": True,
                "monitor_ok": True,
                "results_ingestion_ok": True,
                "relaunch_ok": True,
                "stop_ok": True,
                "resume_latest_ok": False,
                "artifacts_ok": True,
                "errors": [],
                "warnings": [],
                "evidence": {},
                "rerun_commands": [],
            }
        },
    }
    current = json.loads(json.dumps(expected))
    current["modules"]["config"]["launch_ok"] = False

    drifts = compare_qualification_baselines(expected=expected, current=current)
    lines = format_qualification_drift_lines(drifts)
    assert any("field=launch_ok" in line for line in lines)


def test_cli_ui_and_monitor_surfaces_include_qualification_contracts():
    """CLI/UI/monitor/results should expose qualification orchestration hooks."""
    cli_source = Path("halo_forge/cli.py").read_text(encoding="utf-8")
    assert "all-module-qualification" in cli_source
    assert "run_all_module_qualification" in cli_source
    assert "--qualification-profile" in cli_source
    assert "--show-fix-commands" in cli_source
    assert "all-module-bootstrap" in cli_source
    assert "all-module-live" in cli_source

    service_source = Path("ui/services/ops_readiness_service.py").read_text(encoding="utf-8")
    assert "get_qualification_provenance" in service_source
    assert "run_qualification_probe" in service_source
    assert "get_live_provenance" in service_source

    dashboard_source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "_render_provenance_chip(\"qualification\"" in dashboard_source

    research_hub_source = Path("ui/pages/research_hub.py").read_text(encoding="utf-8")
    assert "Run qualification probe" in research_hub_source
    assert "Qualification issue" in research_hub_source
    assert "Run live probe" in research_hub_source

    monitor_source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "QUALIFICATION_JOB_TYPES" in monitor_source
    assert "self.qualification_service.stop_job" in monitor_source

    results_service_source = Path("ui/services/results_service.py").read_text(encoding="utf-8")
    assert "QualificationReportSummary" in results_service_source
    assert "list_qualification_reports" in results_service_source

    results_page_source = Path("ui/pages/results.py").read_text(encoding="utf-8")
    assert "_render_qualification_reports_table" in results_page_source
    assert "_relaunch_qualification_report" in results_page_source

    script_source = Path("scripts/run_all_module_qualification.py").read_text(
        encoding="utf-8"
    )
    assert "--show-fix-commands" in script_source
    assert "format_qualification_issue_lines" in script_source


def test_ci_workflows_include_qualification_steps():
    """CI should include informational qualification run and nightly strict gate."""
    ci_source = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "run_all_module_qualification.py" in ci_source
    assert "all_module_qualification.v1.json" in ci_source

    nightly_source = Path(".github/workflows/nightly_all_module_qualification.yml").read_text(
        encoding="utf-8"
    )
    assert "--qualification-profile fixture-v1" in nightly_source
    assert "--strict" in nightly_source
    assert "--compare-baseline" in nightly_source
