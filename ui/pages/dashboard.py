"""
Dashboard Page

Main overview page with system status, active jobs, and recent runs.
"""

from nicegui import ui
from ui.theme import COLORS
from ui.state import state
from ui.services.hardware import get_gpu_summary


class Dashboard:
    """Dashboard page component."""
    
    def __init__(self):
        pass
    
    def render(self):
        """Render the dashboard page."""
        with ui.column().classes('page-content w-full gap-6 p-6'):
            # Welcome header
            with ui.row().classes('w-full items-center justify-between animate-in'):
                with ui.column().classes('gap-1'):
                    ui.label('Welcome to halo-forge').classes(
                        f'text-2xl font-bold text-[{COLORS["text_primary"]}]'
                    )
                    ui.label('RLVR Training Framework for AMD Strix Halo').classes(
                        f'text-sm text-[{COLORS["text_secondary"]}]'
                    )
                
                # Quick action button
                ui.button('New Training', icon='add', on_click=lambda: ui.navigate.to('/training')).props(
                    'unelevated'
                ).classes(
                    f'btn-hover bg-[{COLORS["primary"]}] text-white'
                )
            
            # Stats cards grid
            gpu = get_gpu_summary()
            with ui.element('div').classes('grid-stats w-full'):
                self._render_stat_card(
                    "GPU Status",
                    gpu.get('util', '--'),
                    gpu.get('name', 'AMD GPU')[:20],
                    COLORS["info"],
                    "memory",
                    1
                )
                self._render_stat_card(
                    "Active Jobs",
                    str(len(state.get_active_jobs())),
                    "Running now",
                    COLORS["running"],
                    "play_circle",
                    2
                )
                self._render_stat_card(
                    "Completed",
                    str(len(state.get_jobs_by_status("completed"))),
                    "Total runs",
                    COLORS["success"],
                    "check_circle",
                    3
                )
                self._render_stat_card(
                    "Failed",
                    str(len(state.get_jobs_by_status("failed"))),
                    "Need attention",
                    COLORS["error"],
                    "error",
                    4
                )
            
            # Main content grid
            with ui.element('div').classes('grid-panels w-full'):
                # Active Jobs panel
                with ui.column().classes(
                    f'gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                    f'border border-[#2d343c] animate-in stagger-5 card-hover'
                ):
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label('Active Jobs').classes(
                            f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                        )
                        ui.button(icon='refresh', on_click=self._refresh_jobs).props(
                            'flat round dense size=sm'
                        ).classes(f'text-[{COLORS["text_muted"]}]')
                    
                    self._render_active_jobs()
                
                # Recent Runs panel
                with ui.column().classes(
                    f'gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                    f'border border-[#2d343c] animate-in stagger-6 card-hover'
                ):
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label('Recent Runs').classes(
                            f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                        )
                        ui.link('View all', target='/results').classes(
                            f'text-xs text-[{COLORS["accent"]}] hover:underline'
                        )
                    
                    self._render_recent_runs()
            
            # Quick Actions
            with ui.column().classes(
                f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-6'
            ):
                ui.label('Quick Actions').classes(
                    f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                )
                
                with ui.row().classes('gap-3 flex-wrap'):
                    self._render_action_button('SFT Training', 'school', '/training?type=sft')
                    self._render_action_button('RAFT Training', 'autorenew', '/training?type=raft')
                    self._render_action_button('Run Benchmark', 'speed', '/training?type=benchmark')
                    self._render_action_button('View Configs', 'settings', '/config')
                    self._render_action_button('Test Verifier', 'verified', '/verifiers')
    
    def _render_stat_card(
        self,
        title: str,
        value: str,
        subtitle: str,
        color: str,
        icon: str,
        stagger: int
    ):
        """Render a statistics card."""
        with ui.column().classes(
            f'gap-3 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in stagger-{stagger} card-hover'
        ):
            with ui.row().classes('w-full items-start justify-between'):
                with ui.column().classes('gap-1'):
                    ui.label(title).classes(
                        f'text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
                    )
                    ui.label(value).classes(
                        f'text-3xl font-bold text-[{COLORS["text_primary"]}] font-mono'
                    )
                
                with ui.element('div').classes(
                    f'w-10 h-10 rounded-lg flex items-center justify-center bg-[{color}]/10'
                ):
                    ui.icon(icon, size='20px').classes(f'text-[{color}]')
            
            ui.label(subtitle).classes(
                f'text-xs text-[{COLORS["text_secondary"]}]'
            )
    
    def _render_active_jobs(self):
        """Render the active jobs list."""
        active_jobs = state.get_active_jobs()
        
        if not active_jobs:
            with ui.column().classes('w-full items-center justify-center py-8 gap-2'):
                ui.icon('hourglass_empty', size='32px').classes(
                    f'text-[{COLORS["text_muted"]}]'
                )
                ui.label('No active jobs').classes(
                    f'text-sm text-[{COLORS["text_muted"]}]'
                )
                ui.button('Start Training', on_click=lambda: ui.navigate.to('/training')).props(
                    'flat dense'
                ).classes(f'text-[{COLORS["accent"]}]')
        else:
            for job in active_jobs:
                self._render_job_row(job)
    
    def _render_job_row(self, job):
        """Render a single job row."""
        with ui.row().classes(
            f'w-full items-center gap-3 p-3 rounded-lg bg-[{COLORS["bg_secondary"]}] '
            f'hover:bg-[{COLORS["bg_hover"]}] transition-colors cursor-pointer'
        ).on('click', lambda j=job: ui.navigate.to(f'/monitor/{j.id}')):
            # Status indicator
            ui.element('div').classes(
                f'w-2 h-2 rounded-full bg-[{COLORS[job.status]}] running-glow'
            )
            
            # Job info
            with ui.column().classes('flex-1 gap-0.5'):
                ui.label(job.name).classes(
                    f'text-sm font-medium text-[{COLORS["text_primary"]}]'
                )
                ui.label(f'{job.type.upper()} • {job.duration_str}').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
            
            # Progress
            if job.progress_percent > 0:
                ui.label(f'{job.progress_percent:.0f}%').classes(
                    f'text-xs font-mono text-[{COLORS["text_secondary"]}]'
                )
    
    def _render_recent_runs(self):
        """Render the recent runs list."""
        recent = state.get_recent_jobs(5)
        completed = [j for j in recent if j.status in ('completed', 'failed', 'stopped')]
        
        if not completed:
            with ui.column().classes('w-full items-center justify-center py-8 gap-2'):
                ui.icon('history', size='32px').classes(
                    f'text-[{COLORS["text_muted"]}]'
                )
                ui.label('No completed runs yet').classes(
                    f'text-sm text-[{COLORS["text_muted"]}]'
                )
        else:
            for job in completed[:5]:
                with ui.row().classes(
                    f'w-full items-center gap-3 p-3 rounded-lg '
                    f'hover:bg-[{COLORS["bg_hover"]}] transition-colors'
                ):
                    # Status icon
                    status_icon = 'check_circle' if job.status == 'completed' else 'cancel'
                    ui.icon(status_icon, size='18px').classes(
                        f'text-[{COLORS[job.status]}]'
                    )
                    
                    # Job info
                    with ui.column().classes('flex-1 gap-0'):
                        ui.label(job.name).classes(
                            f'text-sm text-[{COLORS["text_primary"]}]'
                        )
                        ui.label(job.duration_str).classes(
                            f'text-xs text-[{COLORS["text_muted"]}]'
                        )
    
    def _render_action_button(self, label: str, icon: str, path: str):
        """Render a quick action button."""
        with ui.button(on_click=lambda: ui.navigate.to(path)).props('flat').classes(
            f'btn-hover px-4 py-3 bg-[{COLORS["bg_secondary"]}] '
            f'border border-[#2d343c] rounded-lg'
        ):
            with ui.row().classes('items-center gap-2'):
                ui.icon(icon, size='18px').classes(f'text-[{COLORS["accent"]}]')
                ui.label(label).classes(f'text-sm text-[{COLORS["text_primary"]}]')
    
    async def _refresh_jobs(self):
        """Refresh jobs data."""
        ui.notify('Refreshing...', type='info', timeout=1000)
