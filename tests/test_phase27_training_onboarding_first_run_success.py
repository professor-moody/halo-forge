#!/usr/bin/env python3
"""Phase 27 first-run onboarding and recovery regression tests."""

from __future__ import annotations

from pathlib import Path

from ui.services.training_service import TrainingService
from ui.state import AppState


def test_training_page_has_guided_onboarding_and_scaffold_controls():
    """Training page should provide guided onboarding and scaffold UX."""
    source = Path("ui/pages/training.py").read_text(encoding="utf-8")
    assert "def _render_guided_onboarding_panel(" in source
    assert "Start Here" in source
    assert "Create output scaffold" in source
    assert "def _run_launch_preflight(" in source
    assert "expected_output_artifacts" in source


def test_training_service_preflight_is_structured_and_warns_non_blocking_missing_output(tmp_path):
    """Preflight should return structured diagnostics and keep missing output dir non-blocking."""
    service = TrainingService(AppState())
    output_dir = tmp_path / "new_sft_output"

    result = service.preflight_sft_launch(
        model="Qwen/Qwen2.5-Coder-1.5B",
        dataset="codealpaca",
        output_dir=str(output_dir),
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_samples=8,
    )

    assert result.ok is True
    assert result.errors == []
    assert result.warnings
    assert "output_dir" in result.resolved_paths
    assert result.suggested_fixes

    invalid = service.preflight_sft_launch(
        model="",
        dataset="codealpaca",
        output_dir=str(output_dir),
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
    )
    assert invalid.ok is False
    assert invalid.errors
    assert invalid.suggested_fixes


def test_training_service_scaffold_creates_output_marker(tmp_path):
    """Output scaffold helper should create directory and marker metadata."""
    service = TrainingService(AppState())
    output_dir = tmp_path / "scaffold_target"

    created = service.scaffold_output_dir(str(output_dir), mode_key="sft")

    assert created.exists()
    assert created.is_dir()
    marker = created / ".halo_forge_output_scaffold.json"
    assert marker.exists()
    assert "expected_artifacts" in marker.read_text(encoding="utf-8")


def test_monitor_failure_recovery_panel_exposes_fix_actions():
    """Failed monitor view should expose concise recovery actions."""
    source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "TRAINING_FIX_ROUTES" in source
    assert "def _render_failure_recovery_panel(" in source
    assert "Fix input" in source
    assert "Re-open launch form" in source
    assert "Retry with same config" in source


def test_dashboard_start_here_sequence_is_training_first():
    """Dashboard should include clear launch-first step sequence."""
    source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "Step 1: Choose training type" in source
    assert "Step 2: Open guided form" in source
    assert "Step 3: Monitor run" in source
    assert "/training?mode=sft&ui_mode=quickstart&preset=sft_fast_local" in source
