#!/usr/bin/env python3
"""Phase 13 ops dataset burn-in baseline regression tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from halo_forge.ops_dataset_burnin import (
    DEFAULT_BURNIN_PROFILE,
    OPS_BURNIN_CONTRACT_VERSION,
    OPS_BURNIN_GENERATOR_VERSION,
    OPS_MODULES,
    build_burnin_baseline_payload,
    build_ops_burnin_report,
    compare_burnin_baselines,
    compute_ops_dataset_burnin,
    normalize_ops_burnin_payload,
    validate_burnin_baseline_payload,
    validate_ops_burnin_payload,
)


def test_ops_dataset_burnin_schema_and_normalization():
    """Burn-in report should validate and normalize non-deterministic fields."""
    report = compute_ops_dataset_burnin(
        profile=DEFAULT_BURNIN_PROFILE,
        seed=42,
        source="script",
        fixture_pack="v1",
    )
    payload = report.to_dict()
    assert payload["contract_version"] == OPS_BURNIN_CONTRACT_VERSION
    assert payload["generator_version"] == OPS_BURNIN_GENERATOR_VERSION
    assert validate_ops_burnin_payload(payload) == []

    report.modules["vlm"].errors.append("tmp issue at /tmp/abc123/file.txt")
    normalized = normalize_ops_burnin_payload(report)
    assert normalized["generated_at"] == "<normalized>"
    assert "/tmp/<normalized>" in normalized["modules"]["vlm"]["errors"][-1]


def test_script_writes_report_and_baseline_and_compares(tmp_path):
    """Script should write report/baseline and compare drift with strict behavior."""
    script = Path("scripts/run_ops_dataset_burnin.py")
    assert script.exists()

    report_file = tmp_path / "ops_dataset_burnin.v1.json"
    baseline_file = tmp_path / "ops_dataset_burnin_baseline.v1.json"
    write_result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--burnin-profile",
            "tiny-v1",
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
    assert write_result.returncode == 0
    assert "OPS_BURNIN module=vlm" in write_result.stdout
    assert report_file.exists()
    assert baseline_file.exists()

    baseline_payload = json.loads(baseline_file.read_text(encoding="utf-8"))
    baseline_payload["modules"]["vlm"]["contract_checks"]["launch_ok"] = False
    baseline_file.write_text(json.dumps(baseline_payload), encoding="utf-8")

    compare_result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--burnin-profile",
            "tiny-v1",
            "--baseline-file",
            str(baseline_file),
            "--compare-baseline",
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compare_result.returncode == 1
    assert "BURNIN_DRIFT severity=hard module=vlm path=contract_checks.launch_ok" in compare_result.stdout


def test_baseline_payload_contract_and_comparison():
    """Baseline payload should validate and compare with severity stratification."""
    report = compute_ops_dataset_burnin(
        profile=DEFAULT_BURNIN_PROFILE,
        seed=42,
        source="script",
        fixture_pack="v1",
    )
    baseline = build_burnin_baseline_payload(report)
    assert validate_burnin_baseline_payload(baseline) == []

    current = json.loads(json.dumps(baseline))
    current["modules"]["audio"]["metrics"]["dummy"] = 1
    current["modules"]["reasoning"]["contract_checks"]["launch_ok"] = False
    drifts = compare_burnin_baselines(baseline, current)
    assert any(d["severity"] == "hard" and d["module"] == "reasoning" for d in drifts)
    assert any(d["severity"] == "warn" and d["module"] == "audio" for d in drifts)


def test_cli_surface_exposes_ops_burnin_level_and_flags():
    """CLI parser should include ops-burnin level and profile flag."""
    cli_source = Path("halo_forge/cli.py").read_text(encoding="utf-8")
    assert "ops-burnin" in cli_source
    assert "--burnin-profile" in cli_source
    assert "run_ops_burnin" in cli_source
    assert "ops_dataset_burnin.v1.json" in cli_source


def test_ui_ops_readiness_service_and_pages_include_burnin_provenance_hooks():
    """UI service/pages should expose burn-in provenance metadata."""
    service_source = Path("ui/services/ops_readiness_service.py").read_text(encoding="utf-8")
    assert "get_burnin_provenance" in service_source
    assert "burnin_report_present" in service_source
    assert "burnin_generated_at" in service_source
    assert "burnin_status" in service_source
    assert "get_effective_all_module_readiness" in service_source

    research_hub_source = Path("ui/pages/research_hub.py").read_text(encoding="utf-8")
    assert "burnin report unavailable" in research_hub_source
    assert "burnin status=" in research_hub_source

    dashboard_source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "burnin report unavailable" in dashboard_source
    assert "burnin status=" in dashboard_source


def test_tracked_baseline_and_public_summary_exist():
    """Tracked baseline and public summary contracts should exist."""
    baseline_file = Path("tests/baselines/ops_dataset_burnin_baseline.v1.json")
    summary_file = Path("results/readiness/OPS_DATASET_BURNIN_SUMMARY.md")
    assert baseline_file.exists()
    assert summary_file.exists()
    baseline_payload = json.loads(baseline_file.read_text(encoding="utf-8"))
    assert baseline_payload["contract_version"] == OPS_BURNIN_CONTRACT_VERSION
    assert baseline_payload["generator_version"] == OPS_BURNIN_GENERATOR_VERSION


def test_planning_and_internal_packets_remain_untracked():
    """Internal planning artifacts should remain untracked."""
    if not shutil.which("git"):
        return

    tracked_internal = subprocess.run(
        ["git", "ls-files", ".internal_docs"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked_internal.returncode == 0
    assert tracked_internal.stdout.strip() == ""
