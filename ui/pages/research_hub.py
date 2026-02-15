"""
Research Hub Page

Cross-module readiness and operational testing summary for non-coding modules.
"""

from __future__ import annotations

from typing import Dict

from nicegui import ui

from halo_forge.all_module_readiness import ALL_MODULES
from ui.services import get_ops_readiness_service
from ui.theme import COLORS


MODULE_ROUTE_MAP: Dict[str, str] = {
    "config": "/ops-console",
    "data": "/ops-console",
    "info": "/ops-console",
    "plot": "/ops-console",
    "sft": "/training",
    "raft": "/training",
    "benchmark_code": "/benchmark",
    "benchmark_non_code": "/benchmark-advanced",
    "vlm": "/training",
    "audio": "/training",
    "reasoning": "/training",
    "agentic": "/training",
    "inference": "/inference",
    "benchmark": "/benchmark-advanced",
    "ui_ops": "/monitor",
}


class ResearchHub:
    """Render ops readiness report and actionable module diagnostics."""

    def __init__(self):
        self.readiness_service = get_ops_readiness_service()
        self._content_container = None

    def render(self) -> None:
        with ui.column().classes("page-content w-full gap-6 p-6"):
            with ui.row().classes("w-full items-center justify-between animate-in"):
                ui.label("Research Hub").classes(
                    f"text-2xl font-bold text-[{COLORS['text_primary']}]"
                )
                ui.button(
                    "Refresh readiness",
                    icon="refresh",
                    on_click=self._refresh,
                ).props("flat").classes(f"text-[{COLORS['text_secondary']}]")

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

        for module in ALL_MODULES:
            burnin_entry = None
            if burnin_report and module in burnin_report.modules:
                burnin_entry = burnin_report.modules[module]
            self._render_module_card(module, report.modules[module], burnin_entry)

    def _render_module_card(self, module: str, entry, burnin_entry=None) -> None:
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

            if entry.errors:
                ui.label(f"Blocker: {entry.errors[0]}").classes(
                    f"text-xs text-[{COLORS['error']}]"
                )
            elif entry.warnings:
                ui.label(f"Warning: {entry.warnings[0]}").classes(
                    f"text-xs text-[{COLORS['warning']}]"
                )
            else:
                ui.label("All required contract checks passed.").classes(
                    f"text-xs text-[{COLORS['success']}]"
                )

            if entry.evidence:
                evidence_preview = ", ".join(
                    f"{key}={value}" for key, value in list(entry.evidence.items())[:3]
                )
                ui.label(f"evidence: {evidence_preview}").classes(
                    f"text-xs text-[{COLORS['text_muted']}] font-mono"
                )
