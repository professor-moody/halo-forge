"""Dashboard operations-hub aggregation service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal

from halo_forge.all_module_readiness import ALL_MODULES
from ui.state import state

from .ops_readiness_service import get_ops_readiness_service
from .results_service import get_results_service


DashboardActionKey = Literal[
    "open_surface",
    "contract_probe",
    "bootstrap_probe",
    "live_probe",
]


MODULE_SURFACE_ROUTES: dict[str, str] = {
    "config": "/ops-console?module=config&execution_mode=contract",
    "data": "/ops-console?module=data&execution_mode=contract",
    "info": "/ops-console?module=info&execution_mode=contract",
    "plot": "/ops-console?module=plot&execution_mode=contract",
    "sft": "/training?mode=sft",
    "raft": "/training?mode=raft",
    "benchmark_code": "/benchmark?view=code",
    "benchmark_non_code": "/benchmark?view=non_code",
    "inference": "/inference?mode=optimize",
    "vlm": "/training?mode=vlm",
    "audio": "/training?mode=audio",
    "reasoning": "/training?mode=reasoning",
    "agentic": "/training?mode=agentic",
    "ui_ops": "/monitor",
}

MODULE_GROUPS: dict[str, tuple[str, ...]] = {
    "coding": ("config", "data", "info", "plot", "sft", "raft", "benchmark_code"),
    "non_coding": ("benchmark_non_code", "inference", "vlm", "audio", "reasoning", "agentic"),
    "ops": ("ui_ops",),
}


@dataclass
class DashboardAction:
    key: DashboardActionKey
    label: str
    icon: str
    route: str = ""


@dataclass
class DashboardModuleCard:
    module: str
    status: str
    launch_blocked: bool
    issue_class: str
    primary_message: str
    next_action: str
    evidence_root: str
    surface_route: str
    primary_action: DashboardAction
    secondary_actions: List[DashboardAction] = field(default_factory=list)


@dataclass
class DashboardHubSummary:
    source: str
    generated_at: str
    stale: bool
    pass_count: int
    warn_count: int
    fail_count: int
    cards_by_group: Dict[str, List[DashboardModuleCard]]
    burnin_status: str | None
    bootstrap_status: str | None
    qualification_status: str | None
    live_status: str | None
    active_jobs_count: int
    completed_jobs_count: int
    failed_jobs_count: int


class DashboardHubService:
    """Build module-focused dashboard DTOs from existing readiness/results/state services."""

    def __init__(self) -> None:
        self.ops_readiness_service = get_ops_readiness_service()
        self.results_service = get_results_service()

    def build_summary(self, force_refresh: bool = False) -> DashboardHubSummary:
        report = self.ops_readiness_service.get_effective_all_module_readiness(
            force_refresh=force_refresh
        )
        output_map = self.ops_readiness_service.resolve_effective_output_map(
            include_all_modules=True,
            force_refresh=force_refresh,
        )
        burnin_meta = self.ops_readiness_service.get_burnin_provenance(
            force_refresh=force_refresh
        )
        bootstrap_meta = self.ops_readiness_service.get_bootstrap_provenance(
            force_refresh=force_refresh
        )
        qualification_meta = self.ops_readiness_service.get_qualification_provenance(
            force_refresh=force_refresh
        )
        live_meta = self.ops_readiness_service.get_live_provenance(
            force_refresh=force_refresh
        )

        pass_count = 0
        warn_count = 0
        fail_count = 0
        cards_by_group: Dict[str, List[DashboardModuleCard]] = {
            "coding": [],
            "non_coding": [],
            "ops": [],
        }

        for module in ALL_MODULES:
            entry = report.modules.get(module)
            if entry is None:
                continue

            status = str(entry.status or "warn").lower()
            if status == "pass":
                pass_count += 1
            elif status == "warn":
                warn_count += 1
            else:
                fail_count += 1

            surface_route = MODULE_SURFACE_ROUTES.get(module, "/research-hub")
            evidence_root = str(output_map.get(module) or entry.last_output_dir or "")
            launch_blocked = bool(getattr(entry, "launch_blocked", False))
            issue_class = str(getattr(entry, "issue_class", "") or "none")
            primary_message = self._primary_message(entry)
            next_action = self._next_action(entry)
            primary_action = self._select_primary_action(
                module=module,
                status=status,
                launch_blocked=launch_blocked,
                surface_route=surface_route,
            )
            secondary_actions = self._secondary_actions(
                primary_action_key=primary_action.key,
                surface_route=surface_route,
            )

            card = DashboardModuleCard(
                module=module,
                status=status,
                launch_blocked=launch_blocked,
                issue_class=issue_class,
                primary_message=primary_message,
                next_action=next_action,
                evidence_root=evidence_root,
                surface_route=surface_route,
                primary_action=primary_action,
                secondary_actions=secondary_actions,
            )

            for group, modules in MODULE_GROUPS.items():
                if module in modules:
                    cards_by_group[group].append(card)
                    break

        return DashboardHubSummary(
            source=str(report.source),
            generated_at=str(report.generated_at),
            stale=bool(report.stale),
            pass_count=pass_count,
            warn_count=warn_count,
            fail_count=fail_count,
            cards_by_group=cards_by_group,
            burnin_status=burnin_meta.get("burnin_status"),
            bootstrap_status=bootstrap_meta.get("bootstrap_status"),
            qualification_status=qualification_meta.get("qualification_status"),
            live_status=live_meta.get("live_status"),
            active_jobs_count=len(state.get_active_jobs()),
            completed_jobs_count=len(state.get_jobs_by_status("completed")),
            failed_jobs_count=len(state.get_jobs_by_status("failed")),
        )

    def _primary_message(self, entry) -> str:
        errors = list(getattr(entry, "errors", []) or [])
        warnings = list(getattr(entry, "warnings", []) or [])
        status = str(getattr(entry, "status", "warn")).lower()
        launch_blocked = bool(getattr(entry, "launch_blocked", False))

        if status == "pass":
            return "Ready to launch."
        if errors:
            if launch_blocked:
                return "Setup check not satisfied (advanced diagnostics)."
            return f"Needs setup artifacts (launch available): {errors[0]}"
        if warnings:
            return f"Needs setup artifacts (launch available): {warnings[0]}"
        return "No setup diagnostics available."

    def _next_action(self, entry) -> str:
        status = str(getattr(entry, "status", "warn")).lower()
        launch_blocked = bool(getattr(entry, "launch_blocked", False))
        fix_now = str(getattr(entry, "fix_now", "") or "").strip()
        action_hint = str(getattr(entry, "action_hint", "") or "").strip()

        if launch_blocked:
            return fix_now or action_hint or "Open surface and complete required inputs."
        if status == "warn":
            return fix_now or action_hint or "Open surface and run a first launch to create setup artifacts."
        return "Open the module surface and launch a run."

    def _select_primary_action(
        self,
        *,
        module: str,
        status: str,
        launch_blocked: bool,
        surface_route: str,
    ) -> DashboardAction:
        label = "Open Surface"
        if module in {"sft", "raft", "vlm", "audio", "reasoning", "agentic"}:
            label = "Open Training"
        elif module in {"benchmark_code", "benchmark_non_code"}:
            label = "Open Benchmark"
        elif module == "inference":
            label = "Open Inference"
        return DashboardAction(
            key="open_surface",
            label=label,
            icon="open_in_new",
            route=surface_route,
        )

    def _secondary_actions(
        self,
        *,
        primary_action_key: DashboardActionKey,
        surface_route: str,
    ) -> List[DashboardAction]:
        ordered = [
            DashboardAction(
                key="open_surface",
                label="Open Surface",
                icon="open_in_new",
                route=surface_route,
            ),
            DashboardAction(
                key="contract_probe",
                label="Run Setup Check (Advanced)",
                icon="play_arrow",
            ),
            DashboardAction(
                key="bootstrap_probe",
                label="Generate Setup Artifacts (Advanced)",
                icon="build",
            ),
            DashboardAction(
                key="live_probe",
                label="Run System Health Check (Advanced)",
                icon="play_circle",
            ),
        ]
        return [action for action in ordered if action.key != primary_action_key]


_dashboard_hub_service: DashboardHubService | None = None


def get_dashboard_hub_service() -> DashboardHubService:
    """Get singleton dashboard-hub service."""
    global _dashboard_hub_service
    if _dashboard_hub_service is None:
        _dashboard_hub_service = DashboardHubService()
    return _dashboard_hub_service
