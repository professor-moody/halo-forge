#!/usr/bin/env python3
"""Phase 24K first-run reliability closure regression tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_data_package_uses_lazy_optional_dependency_imports():
    """`halo_forge.data` should not eagerly import optional public dataset stack."""
    source = Path("halo_forge/data/__init__.py").read_text(encoding="utf-8")
    assert "__getattr__" in source
    assert "_LAZY_EXPORTS" in source
    assert "if TYPE_CHECKING:" in source
    assert "\nfrom halo_forge.data.public_datasets import DatasetPreparer, DatasetSpec" not in source
    assert "\nfrom halo_forge.data.llm_generate import TrainingDataGenerator, TopicSpec" not in source


def test_data_validate_cli_path_is_dependency_light():
    """`data validate` should work in minimal environments without public dataset imports."""
    fixture = Path("tests/fixtures/all_modules/v1/data/sample.jsonl")
    assert fixture.exists()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "halo_forge.cli",
            "data",
            "validate",
            str(fixture),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "DATASET VALIDATION REPORT" in result.stdout
    assert "Status: ✓ VALID" in result.stdout


def test_live_smoke_data_module_passes_in_strict_mode(tmp_path):
    """Strict all-module live probe should pass for the data module."""
    report_file = tmp_path / "all_module_live_execution.v1.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_all_module_live_matrix.py",
            "--live-profile",
            "live-smoke-v1",
            "--module",
            "data",
            "--strict",
            "--report-file",
            str(report_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "ALL_LIVE module=data status=pass" in result.stdout
    payload = json.loads(report_file.read_text(encoding="utf-8"))
    assert payload["modules"]["data"]["status"] == "pass"
    assert payload["modules"]["data"]["launch_ok"] is True


def test_dashboard_is_navigation_first_and_probe_actions_demoted():
    """Dashboard should avoid in-place probe handlers and route users to research hub."""
    source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "import asyncio" in source
    assert "asyncio.create_task(self.hardware_monitor.start())" in source
    assert "Open Advanced Diagnostics Tools" in source
    assert "Generate Setup Artifacts (All • Advanced)" not in source
    assert "Run System Health Check (All • Advanced)" not in source
    assert "def _run_card_action(" not in source
    assert "def _run_bootstrap_all(" not in source
    assert "def _run_live_all(" not in source
    assert "_make_card_action_handler" not in source
