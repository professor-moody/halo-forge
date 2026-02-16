#!/usr/bin/env python3
"""Phase 22 dashboard ops hub and discoverability regression tests."""

from __future__ import annotations

from pathlib import Path

from halo_forge.all_module_readiness import (
    ALL_MODULES,
    AllModuleReadiness,
    build_all_module_readiness_report,
)
import ui.services.dashboard_hub_service as dashboard_hub_module


class _FakeOpsReadinessService:
    def __init__(self, report):
        self._report = report

    def get_effective_all_module_readiness(self, force_refresh: bool = False):
        return self._report

    def resolve_effective_output_map(self, include_all_modules: bool = True, force_refresh: bool = False):
        return {module: f"/tmp/{module}" for module in ALL_MODULES}

    def get_burnin_provenance(self, force_refresh: bool = False):
        return {"burnin_status": "warn"}

    def get_bootstrap_provenance(self, force_refresh: bool = False):
        return {"bootstrap_status": "pass"}

    def get_qualification_provenance(self, force_refresh: bool = False):
        return {"qualification_status": "warn"}

    def get_live_provenance(self, force_refresh: bool = False):
        return {"live_status": "pass"}


def test_dashboard_hub_service_builds_all_module_cards(monkeypatch):
    """Hub summary should include all modules with deterministic action mapping."""
    entries = {
        module: AllModuleReadiness(
            module=module,
            status="pass",
            errors=[],
            warnings=[],
            launch_blocked=False,
            issue_class="none",
        )
        for module in ALL_MODULES
    }
    entries["data"] = AllModuleReadiness(
        module="data",
        status="warn",
        errors=[],
        warnings=["missing training_summary.json"],
        launch_blocked=False,
        issue_class="evidence_gap",
    )
    entries["audio"] = AllModuleReadiness(
        module="audio",
        status="fail",
        errors=["unsupported model family"],
        warnings=[],
        launch_blocked=True,
        issue_class="preflight_blocker",
    )
    report = build_all_module_readiness_report(module_entries=entries, seed=42, source="script")

    monkeypatch.setattr(
        dashboard_hub_module,
        "get_ops_readiness_service",
        lambda: _FakeOpsReadinessService(report),
    )
    monkeypatch.setattr(dashboard_hub_module, "get_results_service", lambda: object())
    service = dashboard_hub_module.DashboardHubService()
    summary = service.build_summary(force_refresh=True)

    assert summary.pass_count == len(ALL_MODULES) - 2
    assert summary.warn_count == 1
    assert summary.fail_count == 1

    cards = [
        card
        for cards_in_group in summary.cards_by_group.values()
        for card in cards_in_group
    ]
    assert {card.module for card in cards} == set(ALL_MODULES)
    by_module = {card.module: card for card in cards}

    assert by_module["config"].primary_action.key == "open_surface"
    assert by_module["data"].primary_action.key == "bootstrap_probe"
    assert by_module["audio"].primary_action.key == "contract_probe"
    assert by_module["audio"].launch_blocked is True


def test_dashboard_hub_routes_include_query_driven_deep_links():
    """Module route mapping should expose deterministic query-based deep links."""
    routes = dashboard_hub_module.MODULE_SURFACE_ROUTES
    assert routes["sft"] == "/training?mode=sft"
    assert routes["benchmark_code"] == "/benchmark?view=code"
    assert routes["benchmark_non_code"] == "/benchmark?view=non_code"
    assert routes["inference"] == "/inference?mode=optimize"
    assert routes["data"] == "/ops-console?module=data&execution_mode=contract"


def test_dashboard_and_sidebar_expose_ops_hub_navigation_contract():
    """Dashboard and sidebar should expose grouped operations-hub navigation."""
    dashboard_source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "get_dashboard_hub_service" in dashboard_source
    assert "Operations Hub" in dashboard_source
    assert "Run Bootstrap (All)" in dashboard_source
    assert "Run Live Probe (All)" in dashboard_source
    assert "_render_module_group(" in dashboard_source

    sidebar_source = Path("ui/components/sidebar.py").read_text(encoding="utf-8")
    assert "Overview" in sidebar_source
    assert "Core Workflows" in sidebar_source
    assert "Validation" in sidebar_source
    assert "_nav_groups" in sidebar_source


def test_cross_surface_pages_include_query_preselection_hooks():
    """Key launch surfaces should parse safe query params for preselection."""
    training_source = Path("ui/pages/training.py").read_text(encoding="utf-8")
    assert "_consume_query_params" in training_source
    assert "ignored invalid training mode query param" in training_source

    benchmark_source = Path("ui/pages/benchmark.py").read_text(encoding="utf-8")
    assert "ignored invalid benchmark view query param" in benchmark_source

    bench_adv_source = Path("ui/pages/benchmark_advanced.py").read_text(encoding="utf-8")
    assert "ignored invalid benchmark-advanced domains query param" in bench_adv_source

    inference_source = Path("ui/pages/inference.py").read_text(encoding="utf-8")
    assert "ignored invalid inference mode query param" in inference_source

    ops_source = Path("ui/pages/ops_console.py").read_text(encoding="utf-8")
    assert "ignored invalid ops module query param" in ops_source

    research_source = Path("ui/pages/research_hub.py").read_text(encoding="utf-8")
    assert "Module filter active" in research_source

