#!/usr/bin/env python3
"""Phase 19 qualification explainability and UX regression tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from halo_forge.all_module_qualification import (
    format_qualification_issue_lines,
    validate_all_module_qualification_payload,
)
from halo_forge.all_module_readiness import (
    build_all_module_readiness_report,
    validate_all_module_readiness_payload,
)


def test_all_module_readiness_payload_includes_issue_taxonomy_fields():
    """All-module readiness payload should include additive issue taxonomy keys."""
    report = build_all_module_readiness_report(module_entries={}, seed=42, source="script")
    payload = report.to_dict()
    errors = validate_all_module_readiness_payload(payload)
    assert errors == []

    sample = payload["modules"]["config"]
    assert sample["issue_code"]
    assert sample["issue_scope"] in {"module", "cross_module"}
    assert sample["severity"] in {"info", "warning", "error"}
    assert isinstance(sample["what_is_missing"], list)
    assert isinstance(sample["fix_options"], list)


def test_qualification_payload_accepts_issue_taxonomy_fields(tmp_path):
    """Qualification payload validator should accept additive issue diagnostics."""
    script = Path("scripts/run_all_module_qualification.py")
    report_path = tmp_path / "test_phase19_qualification.v1.json"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--qualification-profile",
            "contract-v1",
            "--write-report",
            "--report-file",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0

    from halo_forge.all_module_qualification import load_all_module_qualification_report

    report = load_all_module_qualification_report(report_path)
    payload = report.to_dict()
    errors = validate_all_module_qualification_payload(payload)
    assert errors == []

    sample = payload["modules"]["inference"]
    assert "issue_code" in sample
    assert "fix_now" in sample
    assert "fix_options" in sample


def test_qualification_issue_lines_are_parseable_and_include_fix_commands():
    """Issue formatter should emit ALL_QUAL_ISSUE and ALL_QUAL_FIX lines."""
    from halo_forge.all_module_qualification import AllModuleQualificationResult

    entry = AllModuleQualificationResult(
        module="audio",
        status="fail",
        launch_ok=False,
        monitor_ok=True,
        results_ingestion_ok=False,
        relaunch_ok=True,
        stop_ok=True,
        resume_latest_ok=False,
        artifacts_ok=False,
        errors=["missing training_summary.json"],
        warnings=[],
        evidence={"output_dir": "models/audio_run"},
        rerun_commands=[],
        launch_blocked=True,
        issue_code="AUDIO_PREFLIGHT_BLOCKED",
        issue_scope="module",
        severity="error",
        what_is_missing=["training_summary.json"],
        fix_now="Generate training summary by running audio train once.",
        fix_options=["halo-forge audio train --dry-run"],
    )
    lines = format_qualification_issue_lines(entry, show_fix_commands=True)
    assert any(line.startswith("ALL_QUAL_ISSUE ") for line in lines)
    assert any(line.startswith("ALL_QUAL_FIX ") for line in lines)


def test_ui_surfaces_use_shared_diagnostic_panel_component():
    """All-module surfaces should use shared diagnostic panel rendering."""
    for page_path in (
        "ui/pages/training.py",
        "ui/pages/benchmark.py",
        "ui/pages/benchmark_advanced.py",
        "ui/pages/inference.py",
        "ui/pages/ops_console.py",
        "ui/pages/research_hub.py",
    ):
        source = Path(page_path).read_text(encoding="utf-8")
        assert "render_readiness_diagnostic_panel" in source


def test_cli_and_script_expose_show_fix_commands_flag():
    """Qualification command surfaces should expose show-fix-commands for operators."""
    cli_source = Path("halo_forge/cli.py").read_text(encoding="utf-8")
    assert "--show-fix-commands" in cli_source
    assert "show_fix_commands=args.show_fix_commands" in cli_source

    script_source = Path("scripts/run_all_module_qualification.py").read_text(
        encoding="utf-8"
    )
    assert "--show-fix-commands" in script_source
    assert "format_qualification_issue_lines" in script_source
