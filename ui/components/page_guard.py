"""
Route render guard helpers.

Prevents full-page 500s by converting page render exceptions into explicit
in-app diagnostics while preserving server-side tracebacks.
"""

from __future__ import annotations

import traceback
from typing import Callable

from nicegui import ui

from ui.theme import COLORS


def render_guarded_page(page_name: str, render_fn: Callable[[], None]) -> None:
    """Render a page body with exception guard rails."""
    try:
        render_fn()
    except Exception as exc:
        print(f"[UI_PAGE_GUARD] page={page_name} error={exc}")
        traceback.print_exc()
        with ui.column().classes(
            f"w-full max-w-3xl mx-auto mt-10 gap-4 p-6 rounded-xl bg-[{COLORS['bg_card']}] "
            f"border border-[{COLORS['error']}]/40"
        ):
            ui.label(f"{page_name} failed to render").classes(
                f"text-lg font-semibold text-[{COLORS['text_primary']}]"
            )
            ui.label(str(exc)).classes(f"text-sm text-[{COLORS['error']}] font-mono")
            ui.label(
                "Try refreshing this page. If it persists, review UI logs for the traceback."
            ).classes(f"text-xs text-[{COLORS['text_secondary']}]")
