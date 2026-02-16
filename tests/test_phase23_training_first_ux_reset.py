#!/usr/bin/env python3
"""Phase 23 training-first UX reset regression tests."""

from __future__ import annotations

from pathlib import Path

from halo_forge.all_module_readiness import AllModuleReadiness, build_all_module_readiness_report
import ui.services.dashboard_hub_service as dashboard_hub_module


class _FakeOpsReadinessService:
    def __init__(self, report):
        self._report = report

    def get_effective_all_module_readiness(self, force_refresh: bool = False):
        return self._report

    def resolve_effective_output_map(self, include_all_modules: bool = True, force_refresh: bool = False):
        return {}

    def get_burnin_provenance(self, force_refresh: bool = False):
        return {"burnin_status": "warn"}

    def get_bootstrap_provenance(self, force_refresh: bool = False):
        return {"bootstrap_status": "warn"}

    def get_qualification_provenance(self, force_refresh: bool = False):
        return {"qualification_status": "warn"}

    def get_live_provenance(self, force_refresh: bool = False):
        return {"live_status": "warn"}


def test_dashboard_uses_training_first_copy_and_advanced_diagnostics_labels():
    """Dashboard should prioritize launch UX and demote diagnostics language."""
    source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "Training Launcher" in source
    assert "Advanced Diagnostics (Optional)" in source
    assert "Run Contract Probe" not in source
    assert "Run Live Probe (All)" not in source
    assert "Run System Health Check (All • Advanced)" in source


def test_dashboard_primary_actions_always_open_surface(monkeypatch):
    """Dashboard cards should keep launch/open as primary action regardless of status."""
    report = build_all_module_readiness_report(
        module_entries={
            "sft": AllModuleReadiness(
                module="sft",
                status="warn",
                warnings=["training_summary.json missing"],
                launch_blocked=False,
                issue_class="evidence_gap",
            ),
            "audio": AllModuleReadiness(
                module="audio",
                status="fail",
                errors=["unsupported model family"],
                launch_blocked=True,
                issue_class="preflight_blocker",
            ),
        },
        seed=42,
        source="script",
    )
    monkeypatch.setattr(
        dashboard_hub_module,
        "get_ops_readiness_service",
        lambda: _FakeOpsReadinessService(report),
    )
    monkeypatch.setattr(dashboard_hub_module, "get_results_service", lambda: object())

    summary = dashboard_hub_module.DashboardHubService().build_summary(force_refresh=True)
    cards = [card for group in summary.cards_by_group.values() for card in group]
    by_module = {card.module: card for card in cards}

    assert by_module["sft"].primary_action.key == "open_surface"
    assert by_module["audio"].primary_action.key == "open_surface"
    assert by_module["audio"].secondary_actions
    assert "Advanced" in by_module["audio"].secondary_actions[0].label


def test_training_page_launch_validation_is_input_driven():
    """Training page should expose explicit input validation for launch controls."""
    source = Path("ui/pages/training.py").read_text(encoding="utf-8")
    assert "def _validate_launch_inputs(" in source
    assert "def _validate_sft_inputs(" in source
    assert "def _validate_raft_inputs(" in source
    assert "def _validate_modality_inputs(" in source
    assert "Advanced setup diagnostics (optional)" in source
    assert "Run Setup Check (Advanced)" in source


def test_diagnostic_panel_uses_setup_check_language():
    """Diagnostics copy should avoid hard launch-blocked wording in primary UX."""
    source = Path("ui/components/diagnostic_panel.py").read_text(encoding="utf-8")
    assert "Run setup check (advanced)" in source
    assert "Setup check not satisfied (advanced diagnostics)." in source
    assert "Launch blocked:" not in source
