#!/usr/bin/env python3
"""Phase 21 all-module live execution closure regression tests."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from halo_forge.all_module_live_execution import (
    ALL_MODULE_LIVE_EXECUTION_CONTRACT_VERSION,
    AllModuleLiveExecutionEntry,
    AllModuleLiveExecutionReport,
    compute_all_module_live_execution,
    normalize_all_module_live_execution_payload,
    validate_all_module_live_execution_payload,
    write_all_module_live_execution_report,
)
from halo_forge.all_module_bootstrap import (
    ALL_MODULE_BOOTSTRAP_CONTRACT_VERSION,
    AllModuleBootstrapEntry,
    AllModuleBootstrapReport,
    write_all_module_bootstrap_report,
)
from halo_forge.all_module_readiness import ALL_MODULES
from ui.services.ops_readiness_service import OpsReadinessService
from ui.services.results_service import ResultsService


def test_live_schema_and_normalization_roundtrip(tmp_path):
    """Live execution report payload should validate and normalize deterministically."""
    report = compute_all_module_live_execution(
        live_profile="live-smoke-v1",
        seed=42,
        source="script",
        output_root=tmp_path / "live",
        module_filters=["config", "data", "info"],
        strict=False,
    )
    payload = report.to_dict()
    errors = validate_all_module_live_execution_payload(payload)
    assert errors == []

    normalized = normalize_all_module_live_execution_payload(report)
    assert normalized["generated_at"] == "<normalized>"


def test_live_script_strict_and_non_strict_behavior(tmp_path):
    """Live matrix script should emit parseable lines and enforce strict failures."""
    script = Path("scripts/run_all_module_live_matrix.py")
    assert script.exists()

    report_file = tmp_path / "all_module_live_execution.v1.json"
    output_root = tmp_path / "live_output"
    non_strict = subprocess.run(
        [
            sys.executable,
            str(script),
            "--live-profile",
            "live-smoke-v1",
            "--module",
            "config",
            "--output-root",
            str(output_root),
            "--write-report",
            "--report-file",
            str(report_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert non_strict.returncode == 0
    assert "ALL_LIVE module=config" in non_strict.stdout
    assert report_file.exists()

    blocked_root = tmp_path / "blocked_root"
    blocked_root.write_text("not a directory", encoding="utf-8")
    strict_fail = subprocess.run(
        [
            sys.executable,
            str(script),
            "--live-profile",
            "live-smoke-v1",
            "--module",
            "config",
            "--output-root",
            str(blocked_root),
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict_fail.returncode == 1
    assert "ERROR: failing modules:" in strict_fail.stdout


def test_ops_readiness_output_map_prefers_live_evidence_roots(monkeypatch, tmp_path):
    """Effective output map should prefer live evidence roots over bootstrap/defaults."""
    live_report_path = tmp_path / "results" / "readiness" / "all_module_live_execution.v1.json"
    bootstrap_report_path = tmp_path / "results" / "readiness" / "all_module_bootstrap.v1.json"
    live_audio_root = tmp_path / "models" / "audio_live"
    bootstrap_audio_root = tmp_path / "models" / "audio_bootstrap"
    live_audio_root.mkdir(parents=True, exist_ok=True)
    bootstrap_audio_root.mkdir(parents=True, exist_ok=True)

    live_entries = {
        module: AllModuleLiveExecutionEntry(
            module=module,
            status="warn",
            probe_attempted=False,
            launch_ok=False,
            monitor_ok=False,
            results_ok=False,
            dependency_status="ok",
            warnings=["module not selected"],
            errors=[],
            evidence_root=str(tmp_path / "results" / "live" / module),
            evidence_files=[],
            rerun_commands=[],
            next_actions=[],
        )
        for module in ALL_MODULES
    }
    live_entries["audio"] = AllModuleLiveExecutionEntry(
        module="audio",
        status="pass",
        probe_attempted=True,
        launch_ok=True,
        monitor_ok=True,
        results_ok=True,
        dependency_status="ok",
        errors=[],
        warnings=[],
        evidence_root=str(live_audio_root),
        evidence_files=[],
        rerun_commands=[],
        next_actions=[],
    )

    live_report = AllModuleLiveExecutionReport(
        contract_version=ALL_MODULE_LIVE_EXECUTION_CONTRACT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        profile="live-smoke-v1",
        seed=42,
        source="script",
        modules=live_entries,
    )
    write_all_module_live_execution_report(live_report_path, live_report)

    bootstrap_entries = {
        module: AllModuleBootstrapEntry(
            module=module,
            status="warn",
            bootstrap_attempted=False,
            warnings=["module not selected"],
            evidence_root=str(tmp_path / "results" / "bootstrap" / module),
            evidence_files=[],
            artifacts_created=[],
            errors=[],
            next_actions=[],
        )
        for module in ALL_MODULES
    }
    bootstrap_entries["audio"] = AllModuleBootstrapEntry(
        module="audio",
        status="pass",
        bootstrap_attempted=True,
        warnings=[],
        errors=[],
        evidence_root=str(bootstrap_audio_root),
        evidence_files=[],
        artifacts_created=[],
        next_actions=[],
    )
    bootstrap_report = AllModuleBootstrapReport(
        contract_version=ALL_MODULE_BOOTSTRAP_CONTRACT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        profile="contract-v1",
        seed=42,
        source="script",
        modules=bootstrap_entries,
    )
    write_all_module_bootstrap_report(bootstrap_report_path, bootstrap_report)

    monkeypatch.setattr(
        "ui.services.ops_readiness_service.get_results_service",
        lambda: ResultsService(base_path=tmp_path),
    )
    service = OpsReadinessService(base_path=tmp_path)
    output_map = service.resolve_effective_output_map(include_all_modules=True, force_refresh=True)
    assert output_map["audio"] == str(live_audio_root)


def test_cli_ui_monitor_results_surfaces_include_live_contracts():
    """CLI/UI/monitor/results should expose live probe orchestration hooks."""
    cli_source = Path("halo_forge/cli.py").read_text(encoding="utf-8")
    assert "all-module-live" in cli_source
    assert "run_all_module_live" in cli_source
    assert "--live-profile" in cli_source

    service_source = Path("ui/services/ops_readiness_service.py").read_text(encoding="utf-8")
    assert "get_live_provenance" in service_source
    assert "run_live_probe" in service_source

    research_hub_source = Path("ui/pages/research_hub.py").read_text(encoding="utf-8")
    assert "Run live probe" in research_hub_source
    assert "Run Live Probe" in research_hub_source

    dashboard_source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "Run Live Probe (All)" in dashboard_source
    assert "_render_provenance_chip" in dashboard_source
    assert "dashboard_hub_service" in dashboard_source

    monitor_source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "LIVE_PROBE_JOB_TYPES" in monitor_source
    assert "self.live_probe_service.stop_job" in monitor_source

    results_service_source = Path("ui/services/results_service.py").read_text(encoding="utf-8")
    assert "LiveProbeReportSummary" in results_service_source
    assert "list_live_probe_reports" in results_service_source

    results_page_source = Path("ui/pages/results.py").read_text(encoding="utf-8")
    assert "_render_live_probe_reports_table" in results_page_source
    assert "_relaunch_live_probe_report" in results_page_source

    script_source = Path("scripts/run_all_module_live_matrix.py").read_text(encoding="utf-8")
    assert "ALL_LIVE" in script_source
