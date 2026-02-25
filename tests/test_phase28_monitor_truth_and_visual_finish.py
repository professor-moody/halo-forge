#!/usr/bin/env python3
"""Phase 28 monitor truth, run-state reliability, and visual finish tests."""

from pathlib import Path


def test_monitor_uses_job_type_specific_progress_and_panel_rendering():
    """Monitor should expose explicit run-type progress/panel helpers."""
    source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "TRAINING_JOB_TYPES" in source
    assert "INDETERMINATE_PROGRESS_JOB_TYPES" in source
    assert "def _resolve_progress_display" in source
    assert "def _render_inference_metrics_panel" in source
    assert "def _render_utility_metrics_panel" in source
    assert "def _render_diagnostics_metrics_panel" in source
    assert "This run does not emit training-loss curves." in source
    assert "if self.job and self.job.type not in TRAINING_JOB_TYPES" in source


def test_monitor_stop_flow_remains_context_safe_and_terminal_state_aware():
    """Stop flow should avoid detached tasks and treat terminal transitions as success."""
    source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "on_click=lambda: self._confirm_stop(dialog)" in source
    assert "create_task(self._confirm_stop(dialog))" not in source
    assert "if refreshed_job and refreshed_job.status in {\"stopped\", \"completed\", \"failed\"}" in source
    assert 'ui.navigate.to(f"/monitor/{self.job_id}")' in source


def test_service_stop_paths_are_idempotent_for_terminal_jobs():
    """Job service stop methods should return success for already terminal states."""
    for rel_path in (
        "ui/services/training_service.py",
        "ui/services/benchmark_service.py",
        "ui/services/inference_service.py",
        "ui/services/module_ops_service.py",
        "ui/services/qualification_service.py",
        "ui/services/bootstrap_service.py",
        "ui/services/live_probe_service.py",
    ):
        source = Path(rel_path).read_text(encoding="utf-8")
        assert "if job.status in {\"stopped\", \"completed\", \"failed\"}" in source

