"""
Sidebar Navigation Component

Left navigation drawer with animated hover effects.
"""

from nicegui import ui
from ui.theme import COLORS


class Sidebar:
    """Left sidebar navigation component."""
    
    NAV_ITEMS = [
        {"icon": "dashboard", "label": "Dashboard", "path": "/"},
        {"icon": "model_training", "label": "Training", "path": "/training"},
        {"icon": "speed", "label": "Benchmark", "path": "/benchmark"},
        {"icon": "computer", "label": "Monitor", "path": "/monitor"},
        {"icon": "settings", "label": "Config", "path": "/config"},
        {"icon": "verified", "label": "Verifiers", "path": "/verifiers"},
        {"icon": "storage", "label": "Datasets", "path": "/datasets"},
        {"icon": "analytics", "label": "Results", "path": "/results"},
    ]
    
    def __init__(self):
        self.render()
    
    def render(self):
        """Render the sidebar content."""
        with ui.column().classes('w-full h-full p-0 m-0'):
            # Logo area
            with ui.row().classes('w-full items-center gap-3 px-4 py-5 border-b border-[#2d343c]'):
                # Logo icon (forge/anvil concept)
                with ui.element('div').classes(
                    f'w-10 h-10 rounded-lg flex items-center justify-center '
                    f'bg-gradient-to-br from-[{COLORS["primary"]}] to-[{COLORS["secondary"]}]'
                ):
                    ui.icon('hardware', size='24px').classes('text-white')
                
                with ui.column().classes('gap-0'):
                    ui.label('HALO-FORGE').classes(
                        f'text-sm font-bold tracking-wider text-[{COLORS["text_primary"]}]'
                    )
                    ui.label('RLVR Training').classes(
                        f'text-xs text-[{COLORS["text_muted"]}]'
                    )
            
            # Navigation items
            with ui.column().classes('w-full flex-1 py-4 gap-1'):
                for item in self.NAV_ITEMS:
                    self._render_nav_item(item)
            
            # Footer
            with ui.column().classes('w-full px-4 py-4 border-t border-[#2d343c] gap-2'):
                # Version info
                ui.label('v1.1.0').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
                
                # Quick status
                with ui.row().classes('items-center gap-2'):
                    ui.element('div').classes(
                        f'w-2 h-2 rounded-full bg-[{COLORS["success"]}]'
                    )
                    ui.label('System Ready').classes(
                        f'text-xs text-[{COLORS["text_secondary"]}]'
                    )
    
    def _render_nav_item(self, item: dict):
        """Render a single navigation item with active state."""
        # Detect current route
        try:
            current_path = ui.context.client.page.path
        except Exception:
            current_path = "/"
        
        # Check if this nav item is active
        if item['path'] == '/':
            is_active = current_path == '/'
        else:
            is_active = current_path.startswith(item['path'])
        
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
