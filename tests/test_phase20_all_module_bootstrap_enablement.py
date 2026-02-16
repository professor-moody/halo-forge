#!/usr/bin/env python3
"""Phase 20 all-module bootstrap enablement regression tests."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from halo_forge.all_module_bootstrap import (
    ALL_MODULE_BOOTSTRAP_CONTRACT_VERSION,
    AllModuleBootstrapEntry,
    AllModuleBootstrapReport,
    compute_all_module_bootstrap,
    normalize_all_module_bootstrap_payload,
    validate_all_module_bootstrap_payload,
    write_all_module_bootstrap_report,
)
from halo_forge.all_module_readiness import ALL_MODULES
from ui.services.ops_readiness_service import OpsReadinessService
from ui.services.results_service import ResultsService


def test_bootstrap_schema_and_normalization_roundtrip(tmp_path):
    """Bootstrap report payload should validate and normalize deterministically."""
    report = compute_all_module_bootstrap(
        bootstrap_profile="contract-v1",
        seed=42,
        source="script",
        output_root=tmp_path / "bootstrap",
        module_filters=["config", "data", "inference"],
        strict=False,
    )
    payload = report.to_dict()
    errors = validate_all_module_bootstrap_payload(payload)
    assert errors == []

    normalized = normalize_all_module_bootstrap_payload(report)
    assert normalized["generated_at"] == "<normalized>"


def test_bootstrap_script_strict_and_non_strict_behavior(tmp_path):
    """Bootstrap script should emit parseable lines and enforce strict failures."""
    script = Path("scripts/run_all_module_bootstrap.py")
    assert script.exists()

    report_file = tmp_path / "all_module_bootstrap.v1.json"
    output_root = tmp_path / "bootstrap_output"
    non_strict = subprocess.run(
        [
            sys.executable,
            str(script),
            "--bootstrap-profile",
            "contract-v1",
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
    assert "ALL_BOOTSTRAP module=config" in non_strict.stdout
    assert report_file.exists()

    blocked_root = tmp_path / "blocked_root"
    blocked_root.write_text("not a directory", encoding="utf-8")
    strict_fail = subprocess.run(
        [
            sys.executable,
            str(script),
            "--bootstrap-profile",
            "contract-v1",
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


def test_ops_readiness_output_map_prefers_bootstrap_evidence_roots(monkeypatch, tmp_path):
    """Effective output map should use bootstrap evidence roots before stale defaults."""
    bootstrap_report_path = tmp_path / "results" / "readiness" / "all_module_bootstrap.v1.json"
    audio_root = tmp_path / "models" / "audio_bootstrap"
    audio_root.mkdir(parents=True, exist_ok=True)

    entries = {
        module: AllModuleBootstrapEntry(
            module=module,
            status="warn",
            bootstrap_attempted=False,
            warnings=["module not selected"],
            evidence_root=str(tmp_path / "results" / "bootstrap" / module),
            evidence_files=[],
            artifacts_created=[],
            next_actions=[],
        )
        for module in ALL_MODULES
    }
    entries["audio"] = AllModuleBootstrapEntry(
        module="audio",
        status="pass",
        bootstrap_attempted=True,
        artifacts_created=[],
        errors=[],
        warnings=[],
        evidence_root=str(audio_root),
        evidence_files=[],
        next_actions=[],
    )

    report = AllModuleBootstrapReport(
        contract_version=ALL_MODULE_BOOTSTRAP_CONTRACT_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        profile="contract-v1",
        seed=42,
        source="script",
        modules=entries,
    )
    write_all_module_bootstrap_report(bootstrap_report_path, report)

    monkeypatch.setattr(
        "ui.services.ops_readiness_service.get_results_service",
        lambda: ResultsService(base_path=tmp_path),
    )
    service = OpsReadinessService(base_path=tmp_path)
    output_map = service.resolve_effective_output_map(include_all_modules=True, force_refresh=True)
    assert output_map["audio"] == str(audio_root)


def test_cli_ui_monitor_results_surfaces_include_bootstrap_contracts():
    """CLI/UI/monitor/results should expose bootstrap orchestration hooks."""
    cli_source = Path("halo_forge/cli.py").read_text(encoding="utf-8")
    assert "all-module-bootstrap" in cli_source
    assert "run_all_module_bootstrap" in cli_source
    assert "--bootstrap-profile" in cli_source
    assert "--output-root" in cli_source

    service_source = Path("ui/services/ops_readiness_service.py").read_text(encoding="utf-8")
    assert "get_bootstrap_provenance" in service_source
    assert "run_bootstrap_probe" in service_source
    assert "get_live_provenance" in service_source

    research_hub_source = Path("ui/pages/research_hub.py").read_text(encoding="utf-8")
    assert "Run bootstrap probe" in research_hub_source
    assert "Generate Evidence" in research_hub_source

    dashboard_source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "_render_provenance_chip(\"bootstrap\"" in dashboard_source

    monitor_source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "BOOTSTRAP_JOB_TYPES" in monitor_source
    assert "self.bootstrap_service.stop_job" in monitor_source
    assert "LIVE_PROBE_JOB_TYPES" in monitor_source

    results_service_source = Path("ui/services/results_service.py").read_text(encoding="utf-8")
    assert "BootstrapReportSummary" in results_service_source
    assert "list_bootstrap_reports" in results_service_source

    results_page_source = Path("ui/pages/results.py").read_text(encoding="utf-8")
    assert "_render_bootstrap_reports_table" in results_page_source
    assert "_relaunch_bootstrap_report" in results_page_source

    script_source = Path("scripts/run_all_module_bootstrap.py").read_text(encoding="utf-8")
    assert "ALL_BOOTSTRAP" in script_source


def test_bootstrap_payload_validator_rejects_missing_keys():
    """Schema validator should fail when required module keys are missing."""
    payload = {
        "contract_version": ALL_MODULE_BOOTSTRAP_CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": "contract-v1",
        "seed": 42,
        "source": "script",
        "modules": {},
    }
    errors = validate_all_module_bootstrap_payload(payload)
    assert errors
    assert any("module entry must be an object" in err for err in errors)
