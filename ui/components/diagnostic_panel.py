"""Shared readiness/qualification diagnostic panel renderer."""

from __future__ import annotations

from typing import Callable, Optional

from nicegui import ui

from ui.theme import COLORS


def status_color(status: str) -> str:
    key = str(status or "").strip().lower()
    if key == "pass":
        return COLORS["success"]
    if key == "warn":
        return COLORS["warning"]
    return COLORS["error"]


def status_icon(status: str) -> str:
    key = str(status or "").strip().lower()
    if key == "pass":
        return "check_circle"
    if key == "warn":
        return "warning"
    return "error"


def render_readiness_diagnostic_panel(
    *,
    module: str,
    entry,
    source: str,
    stale: bool = False,
    expected_path: str = "",
    on_probe: Optional[Callable[[], None]] = None,
    probe_label: str = "Run setup check (advanced)",
    compact: bool = False,
) -> None:
    """Render normalized readiness diagnostics with truthful launch semantics."""
    status = str(getattr(entry, "status", "warn")).lower()
    color = status_color(status)
    icon = status_icon(status)
    launch_blocked = bool(getattr(entry, "launch_blocked", False))

    source_text = source
    if stale:
        source_text += " (stale)"

    with ui.column().classes(
        f"w-full gap-2 p-3 rounded-lg border border-[{color}]/30 bg-[{color}]/10"
    ):
        with ui.row().classes("items-center justify-between gap-2"):
            with ui.row().classes("items-center gap-2"):
                ui.icon(icon, size="16px").classes(f"text-[{color}]")
                ui.label(
                    f"All-module readiness {status.upper()} • module={module} • source={source_text}"
                ).classes(f"text-xs text-[{color}] font-medium")
            if on_probe is not None:
                ui.button(
                    probe_label,
                    icon="play_arrow",
                    on_click=on_probe,
                ).props("flat dense size=sm").classes(
                    f"text-[{COLORS['text_secondary']}]"
                )

        ui.label(_status_summary(status, launch_blocked)).classes(
            f"text-xs text-[{COLORS['text_secondary']}]"
        )

        primary = _primary_message(entry)
        if primary:
            primary_color = COLORS["error"] if launch_blocked else COLORS["warning"]
            if status == "pass":
                primary_color = COLORS["success"]
            ui.label(primary).classes(f"text-xs text-[{primary_color}]")

        missing = list(getattr(entry, "what_is_missing", []) or [])
        if missing:
            text = ", ".join(str(item) for item in missing[:3])
            ui.label(f"What is missing? {text}").classes(
                f"text-xs font-mono text-[{COLORS['text_muted']}]"
            )
        elif expected_path:
            ui.label(f"What is missing? Expected evidence root: {expected_path}").classes(
                f"text-xs font-mono text-[{COLORS['text_muted']}]"
            )

        fix_now = str(getattr(entry, "fix_now", "") or "")
        if fix_now:
            ui.label(f"Fix now: {fix_now}").classes(
                f"text-xs text-[{COLORS['text_secondary']}]"
            )

        issue_code = str(getattr(entry, "issue_code", "") or "")
        severity = str(getattr(entry, "severity", "") or "")
        issue_scope = str(getattr(entry, "issue_scope", "") or "")
        options = list(getattr(entry, "fix_options", []) or [])
        if issue_code or options:
            with ui.expansion(
                text="Technical details",
                icon="terminal",
                value=False,
            ).classes(
                f"w-full rounded bg-[{COLORS['bg_secondary']}] border border-[#2d343c]"
            ).props("dense dark"):
                with ui.column().classes("w-full gap-1 p-2"):
                    if issue_code:
                        ui.label(
                            f"Issue: code={issue_code} • severity={severity or '--'} • scope={issue_scope or '--'}"
                        ).classes(f"text-xs font-mono text-[{COLORS['text_muted']}]")
                    if not compact and options:
                        ui.label("Fix options:").classes(f"text-xs text-[{COLORS['text_muted']}]")
                        for option in options[:3]:
                            ui.label(f"- {option}").classes(
                                f"text-xs font-mono text-[{COLORS['text_muted']}]"
                            )


def _primary_message(entry) -> str:
    errors = list(getattr(entry, "errors", []) or [])
    warnings = list(getattr(entry, "warnings", []) or [])
    status = str(getattr(entry, "status", "warn")).lower()
    if status == "pass":
        return "Launch-ready (contract checks passed)."
    if errors:
        if bool(getattr(entry, "launch_blocked", False)):
            return "Setup check not satisfied (advanced diagnostics)."
        return f"Evidence missing (non-blocking): {errors[0]}"
    if warnings:
        return f"Evidence missing (non-blocking): {warnings[0]}"
    return ""


def _status_summary(status: str, launch_blocked: bool) -> str:
    key = str(status or "").lower()
    if key == "pass":
        return "Launch is available; this status does not imply production readiness."
    if key == "warn":
        return "Evidence is missing or stale; launch remains enabled."
    if launch_blocked:
        return "Setup checks found required issues; fix the listed inputs before launch."
    return "Setup checks found issues; launch may still proceed for troubleshooting."
