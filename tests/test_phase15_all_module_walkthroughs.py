#!/usr/bin/env python3
"""Phase 15 all-module walkthrough regression tests."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from halo_forge.all_module_readiness import ALL_MODULES
from halo_forge.all_module_walkthroughs import (
    DEFAULT_WALKTHROUGH_REPORT_FILE,
    compute_walkthroughs,
    validate_walkthrough_payload,
)


def test_walkthrough_schema_validator_pass_and_fail():
    """Walkthrough report schema should validate canonical payloads and reject missing keys."""
    report = compute_walkthroughs(modules=["ui_ops"], seed=42, profile="contract-v1")
    payload = report.to_dict()

    errors = validate_walkthrough_payload(payload)
    assert errors == []

    broken = dict(payload)
    broken.pop("modules")
    broken_errors = validate_walkthrough_payload(broken)
    assert any("missing top-level key: modules" in err for err in broken_errors)


def test_playbook_generator_creates_all_module_markdown_files(tmp_path):
    """Generator should create one playbook per module plus index in target output directory."""
    script = Path("scripts/generate_all_module_walkthrough_playbooks.py")
    assert script.exists()

    output_dir = tmp_path / "walkthroughs"
    dossier_dir = tmp_path / "dossiers"
    dossier_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
            "--dossier-dir",
            str(dossier_dir),
            "--skip-dossier-update",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "PLAYBOOK index=" in result.stdout

    generated_files = list(output_dir.glob("*_E2E_WALKTHROUGH.md"))
    assert len(generated_files) == len(ALL_MODULES)
    assert (output_dir / "INDEX.md").exists()
    assert (output_dir / "reports").exists()


def test_walkthrough_runner_emits_parseable_lines_and_report_shape(tmp_path):
    """Runner should emit WALKTHROUGH lines and write schema-valid report JSON."""
    script = Path("scripts/run_all_module_walkthroughs.py")
    assert script.exists()

    report_file = tmp_path / "all_module_e2e_walkthrough_report.v1.json"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--module",
            "ui_ops",
            "--profile",
            "contract-v1",
            "--strict",
            "--report-file",
            str(report_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "WALKTHROUGH module=ui_ops status=" in result.stdout
    assert "WALKTHROUGH_REPORT file=" in result.stdout
    assert report_file.exists()

    from halo_forge.all_module_walkthroughs import load_walkthrough_report

    report = load_walkthrough_report(report_file)
    assert report.contract_version == 1
    assert report.profile == "contract-v1"
    assert set(report.modules.keys()) == set(ALL_MODULES)


def test_walkthrough_runner_module_filter_scopes_output(tmp_path):
    """Runner output should be scoped to selected modules when filters are provided."""
    script = Path("scripts/run_all_module_walkthroughs.py")
    report_file = tmp_path / "filtered_walkthrough_report.v1.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--module",
            "sft",
            "--module",
            "raft",
            "--profile",
            "contract-v1",
            "--report-file",
            str(report_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    stdout_lines = [line for line in result.stdout.splitlines() if line.startswith("WALKTHROUGH module=")]
    assert len(stdout_lines) == 2
    assert any("module=sft" in line for line in stdout_lines)
    assert any("module=raft" in line for line in stdout_lines)
    assert all("module=ui_ops" not in line for line in stdout_lines)


def test_cli_surface_exposes_walkthroughs_level_and_flags():
    """CLI test parser should expose walkthrough level and local execution flags."""
    source = Path("halo_forge/cli.py").read_text(encoding="utf-8")
    assert "'walkthroughs'" in source
    assert 'elif args.level == "walkthroughs":' in source
    assert "runner.run_walkthroughs(" in source
    assert "--execute" in source


def test_privacy_and_ci_policy_for_walkthrough_artifacts():
    """Walkthrough artifacts stay internal and do not add CI workflow gating."""
    if not shutil.which("git"):
        pytest.skip("git not available")

    tracked_walkthroughs = subprocess.run(
        ["git", "ls-files", ".internal_docs/research_testing/walkthroughs"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked_walkthroughs.returncode == 0
    assert tracked_walkthroughs.stdout.strip() == ""

    ci_workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "run_all_module_walkthroughs.py" not in ci_workflow
    assert not Path(".github/workflows/nightly_all_module_walkthroughs.yml").exists()


def test_default_walkthrough_report_path_is_internal_only():
    """Canonical walkthrough report path should remain under internal docs."""
    assert str(DEFAULT_WALKTHROUGH_REPORT_FILE).startswith(
        ".internal_docs/research_testing/walkthroughs/reports/"
    )
