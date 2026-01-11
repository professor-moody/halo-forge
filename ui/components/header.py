"""
Header Component

Top bar with status indicators and actions.
"""

from nicegui import ui
from ui.theme import COLORS
from ui.state import state


class Header:
    """Top header bar component."""
    
    def __init__(self):
        self.render()
    
    def render(self):
        """Render the header content."""
        with ui.row().classes('w-full items-center justify-between px-6 py-3'):
            # Left: Page title (will be set by each page)
            with ui.row().classes('items-center gap-4'):
                self.page_title = ui.label('Dashboard').classes(
                    f'text-lg font-semibold text-[{COLORS["text_primary"]}]'
                )
            
            # Right: Status indicators and actions
            with ui.row().classes('items-center gap-6'):
                # Active jobs indicator
                self._render_job_indicator()
                
                # GPU status
                self._render_gpu_status()
                
                # Quick actions
                with ui.row().classes('items-center gap-2'):
                    ui.button(icon='add', on_click=lambda: ui.navigate.to('/training')).props(
                        'flat round dense'
                    ).classes(
                        f'text-[{COLORS["text_secondary"]}] hover:text-[{COLORS["primary"]}]'
                    ).tooltip('New Training')
                    
                    ui.button(icon='refresh', on_click=self._refresh).props(
                        'flat round dense'
                    ).classes(
                        f'text-[{COLORS["text_secondary"]}] hover:text-[{COLORS["primary"]}]'
                    ).tooltip('Refresh')
    
    def _render_job_indicator(self):
        """Render the active jobs indicator."""
        active_jobs = state.get_active_jobs()
        job_count = len(active_jobs)
        
        with ui.row().classes('items-center gap-2 px-3 py-1.5 rounded-lg bg-[#1a1f25]'):
            if job_count > 0:
                # Animated running indicator
                ui.element('div').classes(
                    f'w-2 h-2 rounded-full bg-[{COLORS["running"]}] running-glow'
                )
                ui.label(f'{job_count} running').classes(
                    f'text-xs font-medium text-[{COLORS["text_secondary"]}]'
                )
            else:
                ui.element('div').classes(
                    f'w-2 h-2 rounded-full bg-[{COLORS["text_muted"]}]'
                )
                ui.label('Idle').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
    
    def _render_gpu_status(self):
        """Render GPU status indicator."""
        # TODO: Get real GPU stats from hardware module
        with ui.row().classes('items-center gap-2 px-3 py-1.5 rounded-lg bg-[#1a1f25]'):
            ui.icon('memory', size='16px').classes(f'text-[{COLORS["info"]}]')
            ui.label('GPU').classes(f'text-xs text-[{COLORS["text_muted"]}]')
            ui.label('--').classes(
                f'text-xs font-mono font-medium text-[{COLORS["text_secondary"]}]'
            ).bind_text_from(
                self, '_gpu_util', backward=lambda x: f'{x}%' if x else '--'
            )
        
        self._gpu_util = None
    
    async def _refresh(self):
        """Refresh page data."""
        ui.notify('Refreshing...', type='info', timeout=1000)
        # TODO: Trigger data refresh
    
    def set_title(self, title: str):
        """Set the page title."""
        self.page_title.text = title
