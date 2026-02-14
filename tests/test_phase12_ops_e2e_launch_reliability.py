#!/usr/bin/env python3
"""Phase 12 ops E2E launch reliability regression tests."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from halo_forge.ops_e2e_reliability import (
    OPS_E2E_CONTRACT_VERSION,
    build_ops_e2e_report,
    normalize_ops_e2e_payload,
    validate_ops_e2e_module,
    validate_ops_e2e_payload,
    write_ops_e2e_report,
)
from halo_forge.ops_module_readiness import (
    OPS_MODULES,
    OpsModuleReadiness,
    build_ops_readiness_report,
    write_ops_readiness_report,
)


def test_ops_e2e_schema_validation_and_normalization():
    """E2E report payload should validate and normalize non-deterministic values."""
    entries = {}
    for module in OPS_MODULES:
        entries[module] = validate_ops_e2e_module(
            module=module,
            output_dir=Path("tests/fixtures/ops_e2e/v1") / module
            if module != "ui_ops"
            else Path.cwd(),
            seed=42,
        )

    report = build_ops_e2e_report(
        module_entries=entries,
        seed=42,
        source="script",
        generated_at="2026-02-14T00:00:00+00:00",
    )
    payload = report.to_dict()
    assert payload["contract_version"] == OPS_E2E_CONTRACT_VERSION
    assert validate_ops_e2e_payload(payload) == []

    report.modules["vlm"].evidence["temp_path"] = "/tmp/abc123/model.bin"
    report.modules["vlm"].evidence["run_id"] = "run-1234"
    normalized = normalize_ops_e2e_payload(report)
    assert normalized["generated_at"] == "<normalized>"
    assert normalized["modules"]["vlm"]["evidence"]["temp_path"] == "/tmp/<normalized>"
    assert normalized["modules"]["vlm"]["evidence"]["run_id"] == "<normalized>"


def test_ops_e2e_script_strict_and_non_strict_behavior(tmp_path):
    """Strict mode should fail on module fail; non-strict should remain informational."""
    script = Path("scripts/run_ops_e2e_reliability.py")
    assert script.exists()

    strict_pass = subprocess.run(
        [sys.executable, str(script), "--fixture-pack", "v1", "--strict"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict_pass.returncode == 0
    assert "OPS_E2E module=ui_ops" in strict_pass.stdout

    source_pack = Path("tests/fixtures/ops_e2e/v1")
    corrupted_pack = tmp_path / "corrupted_pack"
    shutil.copytree(source_pack, corrupted_pack)
    (corrupted_pack / "reasoning" / "launch_context.json").unlink()

    non_strict = subprocess.run(
        [sys.executable, str(script), "--fixture-pack", str(corrupted_pack)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert non_strict.returncode == 0
    assert "OPS_E2E module=reasoning status=fail" in non_strict.stdout

    strict_fail = subprocess.run(
        [sys.executable, str(script), "--fixture-pack", str(corrupted_pack), "--strict"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict_fail.returncode == 1
    assert "ERROR: failing modules:" in strict_fail.stdout


def test_cli_and_ui_surface_include_ops_e2e_and_browser_flags():
    """CLI/UI source should expose ops-e2e test level and headless-safe browser flags."""
    cli_source = Path("halo_forge/cli.py").read_text(encoding="utf-8")
    assert "ops-e2e" in cli_source
    assert "--report-file" in cli_source
    assert "--strict" in cli_source
    assert "--fixture-pack" in cli_source
    assert "--open-browser" in cli_source
    assert "--no-browser" in cli_source
    assert "open_browser=False" in cli_source

    app_source = Path("ui/app.py").read_text(encoding="utf-8")
    assert "open_browser: bool = False" in app_source
    assert "show=open_browser" in app_source
    assert "UI_ROUTE root=" in app_source
    assert "UI_ROUTE training=" in app_source
    assert "UI_ROUTE benchmark=" in app_source
    assert "UI_ROUTE inference=" in app_source


def test_resume_latest_scope_only_applies_to_cycle_based_modules():
    """resume_latest is required for cycle-based modalities and N/A for others."""
    pack_root = Path("tests/fixtures/ops_e2e/v1")
    for module in ("vlm", "audio", "reasoning", "agentic"):
        entry = validate_ops_e2e_module(
            module=module,
            output_dir=pack_root / module,
            seed=42,
        )
        assert entry.resume_latest_ok is True

    for module in ("benchmark", "inference"):
        entry = validate_ops_e2e_module(
            module=module,
            output_dir=pack_root / module,
            seed=42,
        )
        assert entry.resume_latest_ok is None

    ui_ops = validate_ops_e2e_module(
        module="ui_ops",
        output_dir=Path.cwd(),
        seed=42,
    )
    assert ui_ops.resume_latest_ok is None


def test_internal_burnin_packet_script_writes_markdown(tmp_path):
    """Burn-in packet generator should combine E2E/readiness reports into markdown."""
    e2e_entries = {}
    readiness_entries = {}
    for module in OPS_MODULES:
        e2e_entries[module] = validate_ops_e2e_module(
            module=module,
            output_dir=Path("tests/fixtures/ops_e2e/v1") / module
            if module != "ui_ops"
            else Path.cwd(),
            seed=42,
        )
        readiness_entries[module] = OpsModuleReadiness(
            module=module,
            status="pass",
            checks={},
            errors=[],
            warnings=[],
            evidence={},
            last_output_dir=str(tmp_path / module),
        )

    e2e_report = build_ops_e2e_report(module_entries=e2e_entries, seed=42, source="script")
    readiness_report = build_ops_readiness_report(
        module_entries=readiness_entries,
        seed=42,
        source="script",
    )

    e2e_path = tmp_path / "ops_e2e.json"
    readiness_path = tmp_path / "ops_readiness.json"
    output_path = tmp_path / "burnin.md"
    write_ops_e2e_report(e2e_path, e2e_report)
    write_ops_readiness_report(readiness_path, readiness_report)

    script = Path("scripts/generate_ops_burnin_packet.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--e2e-report",
            str(e2e_path),
            "--readiness-report",
            str(readiness_path),
            "--output-file",
            str(output_path),
            "--triage-owner",
            "ops-team",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "# Ops Burn-In Packet" in content
    assert "Triage owner: ops-team" in content
    assert "Sign-off:" in content


def test_internal_burnin_artifacts_do_not_leak_to_tracked_docs():
    """Internal burn-in packet paths should not be tracked under public docs surfaces."""
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

    tracked_docs = subprocess.run(
        ["git", "ls-files", "docs"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked_docs.returncode == 0
    leaked = [
        path for path in tracked_docs.stdout.splitlines()
        if "burnin" in Path(path).name.lower() or "packet" in Path(path).name.lower()
    ]
    assert leaked == []
