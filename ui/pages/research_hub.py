"""
Research Hub Page

Cross-module readiness and operational testing summary for all CLI modules.
"""

from __future__ import annotations
from typing import Dict

from nicegui import ui

from halo_forge.all_module_readiness import ALL_MODULES
from ui.components.diagnostic_panel import render_readiness_diagnostic_panel
from ui.query_params import get_query_param
from ui.services import get_ops_readiness_service
from ui.theme import COLORS


MODULE_ROUTE_MAP: Dict[str, str] = {
    "config": "/ops-console?module=config&execution_mode=contract",
    "data": "/ops-console?module=data&execution_mode=contract",
    "info": "/ops-console?module=info&execution_mode=contract",
    "plot": "/ops-console?module=plot&execution_mode=contract",
    "sft": "/training?mode=sft",
    "raft": "/training?mode=raft",
    "benchmark_code": "/benchmark?view=code",
    "benchmark_non_code": "/benchmark?view=non_code",
    "vlm": "/training?mode=vlm",
    "audio": "/training?mode=audio",
    "reasoning": "/training?mode=reasoning",
    "agentic": "/training?mode=agentic",
    "inference": "/inference?mode=optimize",
    "benchmark": "/benchmark-advanced",
    "ui_ops": "/monitor",
}


class ResearchHub:
    """Render ops readiness report and actionable module diagnostics."""

    def __init__(self):
        self.readiness_service = get_ops_readiness_service()
        self._content_container = None
        self.selected_module: str | None = None
        self._query_warnings: list[str] = []
        self._consume_query_params()

    def _consume_query_params(self) -> None:
        module = get_query_param("module", "").lower()
        if not module:
            return
        if module in ALL_MODULES:
            self.selected_module = module
            return
        self._query_warnings.append(f"ignored invalid research module query param: {module}")

    def render(self) -> None:
        with ui.column().classes("page-content w-full gap-6 p-6"):
            with ui.row().classes("w-full items-center justify-between animate-in"):
                ui.label("Research Hub").classes(
                    f"text-2xl font-bold text-[{COLORS['text_primary']}]"
                )
                with ui.row().classes("items-center gap-2"):
                    ui.button(
                        "Run system health check",
                        icon="play_circle",
                        on_click=self._run_live_probe,
                    ).props("flat").classes(f"text-[{COLORS['accent']}]")
                    ui.button(
                        "Generate setup artifacts",
                        icon="build",
                        on_click=self._run_bootstrap_probe,
                    ).props("flat").classes(f"text-[{COLORS['accent']}]")
                    ui.button(
                        "Run setup check",
                        icon="play_arrow",
                        on_click=self._run_qualification_probe,
                    ).props("flat").classes(f"text-[{COLORS['accent']}]")
                    ui.button(
                        "Refresh readiness",
                        icon="refresh",
                        on_click=self._refresh,
                    ).props("flat").classes(f"text-[{COLORS['text_secondary']}]")
            for warning in self._query_warnings:
                ui.label(warning).classes(f"text-xs text-[{COLORS['warning']}]")
            if self.selected_module:
                ui.label(
                    f"Module filter active: {self.selected_module} (from query param)"
                ).classes(f"text-xs text-[{COLORS['text_muted']}]")

            self._content_container = ui.column().classes("w-full gap-4")
            self._render_content(force_refresh=False)

    def _refresh(self) -> None:
        if self._content_container is None:
            return
        self._content_container.clear()
        with self._content_container:
            self._render_content(force_refresh=True)

    def _render_content(self, force_refresh: bool) -> None:
        report = self.readiness_service.get_effective_all_module_readiness(force_refresh=force_refresh)
        bootstrap_meta = self.readiness_service.get_bootstrap_provenance(force_refresh=force_refresh)
        bootstrap_report = None
        if bootstrap_meta.get("bootstrap_report_present"):
            try:
                bootstrap_report = self.readiness_service.load_bootstrap_report(
                    force_refresh=force_refresh
                )
            except Exception:
                bootstrap_report = None
        live_meta = self.readiness_service.get_live_provenance(force_refresh=force_refresh)
        live_report = None
        if live_meta.get("live_report_present"):
            try:
                live_report = self.readiness_service.load_live_report(force_refresh=force_refresh)
            except Exception:
                live_report = None
        qualification_meta = self.readiness_service.get_qualification_provenance(force_refresh=force_refresh)
        qualification_report = None
        if qualification_meta.get("qualification_report_present"):
            try:
                qualification_report = self.readiness_service.load_qualification_report(
                    force_refresh=force_refresh
                )
            except Exception:
                qualification_report = None
        burnin_meta = self.readiness_service.get_burnin_provenance(force_refresh=force_refresh)
        walkthrough_meta = self.readiness_service.get_walkthrough_provenance(force_refresh=force_refresh)
        burnin_report = None
        if burnin_meta.get("burnin_report_present"):
            try:
                burnin_report = self.readiness_service.load_burnin_report(force_refresh=force_refresh)
            except Exception:
                burnin_report = None
        with ui.column().classes(
            f"w-full gap-2 p-4 rounded-xl bg-[{COLORS['bg_card']}] border border-[#2d343c]"
        ):
            ui.label("Ops Readiness Report").classes(
                f"text-base font-semibold text-[{COLORS['text_primary']}]"
            )
            source_text = f"source={report.source} generated_at={report.generated_at}"
            if report.stale:
                source_text += f" stale={report.age_seconds}s"
            ui.label(source_text).classes(f"text-xs text-[{COLORS['text_muted']}] font-mono")
            if burnin_meta.get("burnin_report_present"):
                ui.label(
                    "burnin "
                    f"status={burnin_meta.get('burnin_status')} "
                    f"source={burnin_meta.get('burnin_source')} "
                    f"generated_at={burnin_meta.get('burnin_generated_at')}"
                ).classes(f"text-xs text-[{COLORS['text_muted']}] font-mono")
            else:
                ui.label("burnin report unavailable (non-blocking)").classes(
                    f"text-xs text-[{COLORS['warning']}]"
                )
            if bootstrap_meta.get("bootstrap_report_present"):
                ui.label(
                    "bootstrap "
                    f"status={bootstrap_meta.get('bootstrap_status')} "
                    f"profile={bootstrap_meta.get('bootstrap_profile')} "
                    f"source={bootstrap_meta.get('bootstrap_source')} "
                    f"generated_at={bootstrap_meta.get('bootstrap_generated_at')}"
                ).classes(f"text-xs text-[{COLORS['text_muted']}] font-mono")
            else:
                ui.label("bootstrap report unavailable (non-blocking)").classes(
                    f"text-xs text-[{COLORS['warning']}]"
                )
            if live_meta.get("live_report_present"):
                ui.label(
                    "live "
                    f"status={live_meta.get('live_status')} "
                    f"profile={live_meta.get('live_profile')} "
                    f"source={live_meta.get('live_source')} "
                    f"generated_at={live_meta.get('live_generated_at')}"
                ).classes(f"text-xs text-[{COLORS['text_muted']}] font-mono")
            else:
                ui.label("live report unavailable (non-blocking)").classes(
                    f"text-xs text-[{COLORS['warning']}]"
                )
            if qualification_meta.get("qualification_report_present"):
                ui.label(
                    "qualification "
                    f"status={qualification_meta.get('qualification_status')} "
                    f"profile={qualification_meta.get('qualification_profile')} "
                    f"source={qualification_meta.get('qualification_source')} "
                    f"generated_at={qualification_meta.get('qualification_generated_at')}"
                ).classes(f"text-xs text-[{COLORS['text_muted']}] font-mono")
            else:
                ui.label("qualification report unavailable (non-blocking)").classes(
                    f"text-xs text-[{COLORS['warning']}]"
                )
            if walkthrough_meta.get("walkthrough_report_present"):
                summary = walkthrough_meta.get("walkthrough_status_summary") or {}
                ui.label(
                    "walkthrough "
                    f"profile={walkthrough_meta.get('walkthrough_profile')} "
                    f"generated_at={walkthrough_meta.get('walkthrough_generated_at')} "
                    f"pass={summary.get('pass', 0)} "
                    f"warn={summary.get('warn', 0)} "
                    f"fail={summary.get('fail', 0)}"
                ).classes(f"text-xs text-[{COLORS['text_muted']}] font-mono")
            else:
                ui.label("walkthrough report unavailable (internal/non-blocking)").classes(
                    f"text-xs text-[{COLORS['warning']}]"
                )
            ui.label(
                "Readiness is warn-but-allow: launches remain enabled for debugging and validation."
            ).classes(f"text-xs text-[{COLORS['text_secondary']}]")

        module_list = [self.selected_module] if self.selected_module else list(ALL_MODULES)
        for module in module_list:
            burnin_entry = None
            bootstrap_entry = None
            qualification_entry = None
            if burnin_report and module in burnin_report.modules:
                burnin_entry = burnin_report.modules[module]
            if bootstrap_report and module in bootstrap_report.modules:
                bootstrap_entry = bootstrap_report.modules[module]
            live_entry = None
            if live_report and module in live_report.modules:
                live_entry = live_report.modules[module]
            if qualification_report and module in qualification_report.modules:
                qualification_entry = qualification_report.modules[module]
            self._render_module_card(
                module,
                report.modules[module],
                burnin_entry,
                bootstrap_entry,
                live_entry,
                qualification_entry,
                report.source,
                bool(report.stale),
            )

    def _render_module_card(
        self,
        module: str,
        entry,
        burnin_entry=None,
        bootstrap_entry=None,
        live_entry=None,
        qualification_entry=None,
        source: str = "ui_live_compute",
        stale: bool = False,
    ) -> None:
        color = {
            "pass": COLORS["success"],
            "warn": COLORS["warning"],
            "fail": COLORS["error"],
        }.get(entry.status, COLORS["text_secondary"])
        with ui.column().classes(
            f"w-full gap-2 p-4 rounded-xl bg-[{COLORS['bg_card']}] border border-[#2d343c]"
        ):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("fact_check", size="18px").classes(f"text-[{color}]")
                    ui.label(module.upper()).classes(
                        f"text-sm font-semibold text-[{COLORS['text_primary']}]"
                    )
                    ui.label(entry.status.upper()).classes(
                        f"text-xs font-mono px-2 py-0.5 rounded bg-[{color}]/20 text-[{color}]"
                    )
                route = MODULE_ROUTE_MAP.get(module)
                if route:
                    ui.button(
                        "Open Surface",
                        icon="open_in_new",
                        on_click=lambda r=route: ui.navigate.to(r),
                    ).props("flat dense").classes(f"text-[{COLORS['accent']}]")
                ui.button(
                    "Run Setup Check (Advanced)",
                    icon="play_arrow",
                    on_click=lambda m=module: self._run_contract_probe(m),
                ).props("flat dense").classes(f"text-[{COLORS['text_secondary']}]")
                ui.button(
                    "Generate Setup Artifacts (Advanced)",
                    icon="build",
                    on_click=self._make_module_bootstrap_handler(module),
                ).props("flat dense").classes(f"text-[{COLORS['accent']}]")
                ui.button(
                    "Run System Health Check (Advanced)",
                    icon="play_circle",
                    on_click=self._make_module_live_probe_handler(module),
                ).props("flat dense").classes(f"text-[{COLORS['accent']}]")

            ui.label(f"errors={len(entry.errors)} warnings={len(entry.warnings)}").classes(
                f"text-xs text-[{COLORS['text_muted']}] font-mono"
            )
            if burnin_entry is not None:
                burnin_color = {
                    "pass": COLORS["success"],
                    "warn": COLORS["warning"],
                    "fail": COLORS["error"],
                }.get(burnin_entry.status, COLORS["text_secondary"])
                ui.label(
                    f"burnin status={burnin_entry.status} "
                    f"errors={len(burnin_entry.errors)} warnings={len(burnin_entry.warnings)}"
                ).classes(f"text-xs text-[{burnin_color}] font-mono")
            if bootstrap_entry is not None:
                boot_color = {
                    "pass": COLORS["success"],
                    "warn": COLORS["warning"],
                    "fail": COLORS["error"],
                }.get(bootstrap_entry.status, COLORS["text_secondary"])
                ui.label(
                    f"bootstrap status={bootstrap_entry.status} "
                    f"attempted={1 if bootstrap_entry.bootstrap_attempted else 0} "
                    f"created={len(bootstrap_entry.artifacts_created)}"
                ).classes(f"text-xs text-[{boot_color}] font-mono")
                if bootstrap_entry.errors:
                    error_color = (
                        COLORS["error"]
                        if str(bootstrap_entry.status).lower() == "fail"
                        else COLORS["warning"]
                    )
                    ui.label(
                        f"Bootstrap issue: {bootstrap_entry.errors[0]}"
                    ).classes(f"text-xs text-[{error_color}]")
                elif bootstrap_entry.warnings:
                    ui.label(
                        f"Bootstrap evidence gap (non-blocking): {bootstrap_entry.warnings[0]}"
                    ).classes(f"text-xs text-[{COLORS['warning']}]")
            if live_entry is not None:
                live_color = {
                    "pass": COLORS["success"],
                    "warn": COLORS["warning"],
                    "fail": COLORS["error"],
                }.get(live_entry.status, COLORS["text_secondary"])
                ui.label(
                    f"live status={live_entry.status} "
                    f"launch={1 if live_entry.launch_ok else 0} "
                    f"monitor={1 if live_entry.monitor_ok else 0} "
                    f"results={1 if live_entry.results_ok else 0} "
                    f"deps={live_entry.dependency_status}"
                ).classes(f"text-xs text-[{live_color}] font-mono")
                if live_entry.errors:
                    error_color = (
                        COLORS["error"]
                        if str(live_entry.status).lower() == "fail"
                        else COLORS["warning"]
                    )
                    ui.label(
                        f"Live probe issue: {live_entry.errors[0]}"
                    ).classes(f"text-xs text-[{error_color}]")
                elif live_entry.warnings:
                    ui.label(
                        f"Live evidence gap (non-blocking): {live_entry.warnings[0]}"
                    ).classes(f"text-xs text-[{COLORS['warning']}]")
            if qualification_entry is not None:
                qual_color = {
                    "pass": COLORS["success"],
                    "warn": COLORS["warning"],
                    "fail": COLORS["error"],
                }.get(qualification_entry.status, COLORS["text_secondary"])
                ui.label(
                    f"qualification status={qualification_entry.status} "
                    f"launch_ok={1 if qualification_entry.launch_ok else 0} "
                    f"monitor_ok={1 if qualification_entry.monitor_ok else 0} "
                    f"results_ok={1 if qualification_entry.results_ingestion_ok else 0}"
                ).classes(f"text-xs text-[{qual_color}] font-mono")
                if qualification_entry.errors:
                    error_color = (
                        COLORS["error"]
                        if str(qualification_entry.status).lower() == "fail"
                        else COLORS["warning"]
                    )
                    ui.label(
                        f"Qualification issue: {qualification_entry.errors[0]}"
                    ).classes(f"text-xs text-[{error_color}]")
                elif qualification_entry.warnings:
                    ui.label(
                        f"Qualification evidence gap (non-blocking): {qualification_entry.warnings[0]}"
                    ).classes(f"text-xs text-[{COLORS['warning']}]")
            render_readiness_diagnostic_panel(
                module=module,
                entry=entry,
                source=source,
                stale=stale,
                expected_path=str(entry.last_output_dir or ""),
                on_probe=None,
                compact=True,
            )

    def _run_contract_probe(self, module: str) -> None:
        ok, message = self.readiness_service.run_contract_probe(
            module=module,
            include_all_modules=True,
        )
        ui.notify(message, type="positive" if ok else "warning", timeout=1800)
        self._refresh()

    def _make_module_bootstrap_handler(self, module: str):
        async def _handler() -> None:
            await self._run_module_bootstrap(module)

        return _handler

    def _make_module_live_probe_handler(self, module: str):
        async def _handler() -> None:
            await self._run_module_live_probe(module)

        return _handler

    async def _run_qualification_probe(self) -> None:
        ok, message, job_id = await self.readiness_service.run_qualification_probe(
            qualification_profile="contract-v1",
            strict=False,
        )
        ui.notify(message, type="positive" if ok else "warning", timeout=2200)
        if ok and job_id:
            ui.navigate.to(f"/monitor/{job_id}")

    async def _run_bootstrap_probe(self) -> None:
        ok, message, job_id = await self.readiness_service.run_bootstrap_probe(
            bootstrap_profile="contract-v1",
            strict=False,
            modules=[],
        )
        ui.notify(message, type="positive" if ok else "warning", timeout=2200)
        if ok and job_id:
            ui.navigate.to(f"/monitor/{job_id}")

    async def _run_module_bootstrap(self, module: str) -> None:
        ok, message, job_id = await self.readiness_service.run_bootstrap_probe(
            bootstrap_profile="contract-v1",
            strict=False,
            modules=[module],
        )
        ui.notify(message, type="positive" if ok else "warning", timeout=2200)
        if ok and job_id:
            ui.navigate.to(f"/monitor/{job_id}")

    async def _run_live_probe(self) -> None:
        ok, message, job_id = await self.readiness_service.run_live_probe(
            live_profile="live-smoke-v1",
            strict=False,
            modules=[],
        )
        ui.notify(message, type="positive" if ok else "warning", timeout=2200)
        if ok and job_id:
            ui.navigate.to(f"/monitor/{job_id}")

    async def _run_module_live_probe(self, module: str) -> None:
        ok, message, job_id = await self.readiness_service.run_live_probe(
            live_profile="live-smoke-v1",
            strict=False,
            modules=[module],
        )
        ui.notify(message, type="positive" if ok else "warning", timeout=2200)
        if ok and job_id:
            ui.navigate.to(f"/monitor/{job_id}")
