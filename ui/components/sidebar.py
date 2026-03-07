"""
Sidebar Navigation Component

Left navigation drawer with animated hover effects.
"""

from nicegui import ui
from ui.theme import COLORS
from ui import __version__
from ui.feature_flags import get_ui_feature_flags
from ui.services.dashboard_hub_service import get_dashboard_hub_service


def _is_active_route(item_path: str, current_path: str) -> bool:
    """Match exact routes and nested segments without false prefix collisions."""
    item = str(item_path or "").strip() or "/"
    current = str(current_path or "").strip() or "/"
    if item == "/":
        return current == "/"
    return current == item or current.startswith(f"{item}/")


def _readiness_status() -> tuple[str, str]:
    """Return a non-blocking readiness label and color for the sidebar footer."""
    try:
        summary = get_dashboard_hub_service().build_summary()
    except Exception:
        return "Status unavailable", COLORS["text_muted"]
    if summary.fail_count > 0:
        return "Needs attention", COLORS["error"]
    if summary.warn_count > 0:
        return "Setup pending", COLORS["warning"]
    return "Ready", COLORS["success"]


class Sidebar:
    """Left sidebar navigation component."""
    
    def __init__(self):
        self.render()
    
    def render(self):
        """Render the sidebar content."""
        with ui.column().classes('w-full h-full p-0 m-0'):
            # Logo area
            with ui.row().classes('w-full items-center gap-3 px-4 py-5 border-b border-[#2d343c]'):
                # Logo image (halo-forge branding)
                ui.image('/static/favicon.svg').classes(
                    'w-10 h-10 rounded-lg'
                ).props('fit=contain')
                
                with ui.column().classes('gap-0'):
                    ui.label('HALO-FORGE').classes(
                        f'text-sm font-bold tracking-wider text-[{COLORS["text_primary"]}]'
                    )
                    ui.label('RLVR Training').classes(
                        f'text-xs text-[{COLORS["text_muted"]}]'
                    )
            
            # Navigation items
            with ui.column().classes('w-full flex-1 py-4 gap-1'):
                for group_label, items in self._nav_groups():
                    ui.label(group_label).classes(
                        f'text-[10px] uppercase tracking-wider px-4 pt-2 pb-1 text-[{COLORS["text_muted"]}]'
                    )
                    for item in items:
                        self._render_nav_item(item)
            
            # Footer
            with ui.column().classes('w-full px-4 py-4 border-t border-[#2d343c] gap-2'):
                # Version info (imported from ui package)
                ui.label(f'v{__version__}').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
                
                # Quick status
                status_text, status_color = _readiness_status()
                with ui.row().classes('items-center gap-2'):
                    ui.element('div').classes(
                        f'w-2 h-2 rounded-full bg-[{status_color}]'
                    )
                    ui.label(status_text).classes(
                        f'text-xs text-[{status_color}]'
                    )
    
    def _render_nav_item(self, item: dict):
        """Render a single navigation item with active state."""
        # Detect current route
        try:
            current_path = ui.context.client.page.path
        except Exception:
            current_path = "/"
        
        # Check if this nav item is active
        is_active = _is_active_route(item['path'], current_path)
        
        # Active styling - CSS handles box-shadow via .nav-item.active
        icon_color = COLORS["primary"] if is_active else COLORS["text_secondary"]
        text_color = COLORS["primary"] if is_active else COLORS["text_secondary"]
        active_class = 'active' if is_active else ''
        
        with ui.link(target=item['path']).classes('no-underline w-full'):
            with ui.row().classes(
                f'nav-item w-full items-center gap-3 py-3 cursor-pointer rounded-r-lg {active_class}'
            ):
                ui.icon(item['icon'], size='20px').classes(f'text-[{icon_color}]')
                ui.label(item['label']).classes(f'text-sm font-medium text-[{text_color}]')

    def _nav_groups(self) -> list[tuple[str, list[dict]]]:
        core_workflows = [
            {"icon": "model_training", "label": "Training", "path": "/training"},
            {"icon": "speed", "label": "Benchmark", "path": "/benchmark"},
            {"icon": "terminal", "label": "Ops Console", "path": "/ops-console"},
        ]
        operations = [
            {"icon": "dashboard", "label": "Dashboard", "path": "/"},
            {"icon": "computer", "label": "Monitor", "path": "/monitor"},
            {"icon": "analytics", "label": "Results", "path": "/results"},
        ]
        validation = [
            {"icon": "settings", "label": "Config", "path": "/config"},
            {"icon": "storage", "label": "Datasets", "path": "/datasets"},
            {"icon": "verified", "label": "Verifiers", "path": "/verifiers"},
        ]

        flags = get_ui_feature_flags()
        if flags.enable_inference_page:
            core_workflows.append({"icon": "bolt", "label": "Inference", "path": "/inference"})
        if flags.enable_benchmark_advanced_page:
            core_workflows.append(
                {"icon": "view_array", "label": "Benchmark+", "path": "/benchmark-advanced"}
            )
        if flags.enable_research_hub_page:
            validation.insert(
                0,
                {"icon": "science", "label": "Advanced Diagnostics Tools", "path": "/research-hub"},
            )

        return [
            ("Core Workflows", core_workflows),
            ("Operations", operations),
            ("Validation", validation),
        ]
