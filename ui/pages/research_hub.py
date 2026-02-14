"""
Research Hub Page

Cross-module readiness and operational testing summary for non-coding modules.
"""

from __future__ import annotations

from typing import Dict

from nicegui import ui

from halo_forge.ops_module_readiness import OPS_MODULES
from ui.services import get_ops_readiness_service
from ui.theme import COLORS


MODULE_ROUTE_MAP: Dict[str, str] = {
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
        report = self.readiness_service.get_effective_readiness(force_refresh=force_refresh)
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
            ui.label(
                "Readiness is warn-but-allow: launches remain enabled for debugging and validation."
            ).classes(f"text-xs text-[{COLORS['text_secondary']}]")

        for module in OPS_MODULES:
            self._render_module_card(module, report.modules[module])

    def _render_module_card(self, module: str, entry) -> None:
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
