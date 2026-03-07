#!/usr/bin/env python3
"""Phase 26 diagnostics isolation and training clarity regression tests."""

from pathlib import Path


def test_shared_ui_layout_bootstraps_hardware_monitor_once():
    """Shared UI layout should own hardware monitor startup instead of dashboard-only wiring."""
    app_source = Path("ui/app.py").read_text(encoding="utf-8")
    assert "def _ensure_hardware_monitor_started()" in app_source
    assert "_ensure_hardware_monitor_started()" in app_source

    dashboard_source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "asyncio.create_task(self.hardware_monitor.start())" not in dashboard_source


def test_primary_workflow_pages_keep_diagnostics_actions_out_of_main_flow():
    """Training, benchmark, and inference pages should stay launch-first."""
    for rel_path in ("ui/pages/training.py", "ui/pages/benchmark.py", "ui/pages/inference.py"):
        source = Path(rel_path).read_text(encoding="utf-8")
        assert "Run Setup Check (Advanced)" not in source
        assert "Generate Setup Artifacts (Advanced)" not in source
        assert "Run System Health Check (Advanced)" not in source
        assert "Advanced setup checks are available in Advanced Diagnostics Tools." in source


def test_research_hub_is_renamed_and_marked_troubleshooting_only():
    """The troubleshooting surface should be clearly labeled and scoped."""
    app_source = Path("ui/app.py").read_text(encoding="utf-8")
    assert "@ui.page('/research-hub')" in app_source
    assert 'create_layout("Advanced Diagnostics Tools")' in app_source

    sidebar_source = Path("ui/components/sidebar.py").read_text(encoding="utf-8")
    assert "Advanced Diagnostics Tools" in sidebar_source

    dashboard_source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "Open Advanced Diagnostics Tools" in dashboard_source

    research_source = Path("ui/pages/research_hub.py").read_text(encoding="utf-8")
    assert "Advanced Diagnostics Tools" in research_source
    assert "For troubleshooting only." in research_source
    assert "Run Setup Check" in research_source
    assert "Generate Setup Files" in research_source
    assert "Run System Health Check" in research_source


def test_monitor_and_results_hide_advanced_runs_by_default():
    """Diagnostics runs should be opt-in in monitor and results list views."""
    monitor_source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "Show advanced diagnostics runs" in monitor_source
    assert "monitor_show_advanced_diagnostics" in monitor_source

    results_source = Path("ui/pages/results.py").read_text(encoding="utf-8")
    assert "Show advanced diagnostics runs" in results_source
    assert "results_show_advanced_diagnostics" in results_source
    assert "Advanced Diagnostics" in results_source
