"""
Header Component

Top bar with status indicators and actions.
"""

from typing import Callable, List
from nicegui import ui
from ui.theme import COLORS
from ui.state import state
from ui.services.hardware import get_gpu_summary
from ui.services import get_event_bus, Event, EventType


class Header:
    """Top header bar component."""
    
    def __init__(self, title: str = "Dashboard"):
        # Initialize bindable attributes BEFORE rendering
        self._gpu_util = None
        self._gpu_mem = None
        self._title = title
        self._unsubscribe_callbacks: List[Callable[[], None]] = []
        self._job_indicator_container = None
        self.render()
        
        # Subscribe to real-time updates
        self._setup_event_subscriptions()
    
    def render(self):
        """Render the header content."""
        with ui.row().classes('w-full items-center justify-between px-6 py-2'):
            # Left: Page title (set from route)
            with ui.row().classes('items-center gap-4'):
                self.page_title = ui.label(self._title).classes(
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
        """Render the active jobs indicator container."""
        self._job_indicator_container = ui.row().classes(
            'items-center gap-2 px-3 py-1.5 rounded-lg bg-[#1a1f25]'
        )
        with self._job_indicator_container:
            self._render_job_indicator_content()
    
    def _render_job_indicator_content(self):
        """Render the actual job indicator content (can be re-rendered)."""
        active_jobs = state.get_active_jobs()
        job_count = len(active_jobs)
        
        if job_count > 0:
            # Animated running indicator
            ui.element('div').classes(
                f'w-2 h-2 rounded-full bg-[{COLORS["running"]}] running-glow'
            )
            ui.label(f'{job_count} running').classes(
                f'text-xs font-medium text-[{COLORS["text_secondary"]}] leading-none'
            )
        else:
            ui.element('div').classes(
                f'w-2 h-2 rounded-full bg-[{COLORS["text_muted"]}]'
            )
            ui.label('Idle').classes(
                f'text-xs text-[{COLORS["text_muted"]}] leading-none'
            )
    
    def _render_gpu_status(self):
        """Render GPU status indicator."""
        with ui.row().classes('items-center gap-2 px-3 py-1.5 rounded-lg bg-[#1a1f25]'):
            ui.icon('memory', size='16px').classes(f'text-[{COLORS["info"]}]')
            ui.label('GPU').classes(f'text-xs text-[{COLORS["text_muted"]}] leading-none')
            self._gpu_label = ui.label('--').classes(
                f'text-xs font-mono font-medium text-[{COLORS["text_secondary"]}] leading-none'
            )
        
        # Initial GPU stats fetch
        self._update_gpu_stats()
    
    def _setup_event_subscriptions(self):
        """Set up event subscriptions for real-time updates."""
        bus = get_event_bus()
        
        # Subscribe to GPU updates
        unsub_gpu = bus.subscribe(EventType.GPU_UPDATE, self._on_gpu_update)
        self._unsubscribe_callbacks.append(unsub_gpu)
        
        # Subscribe to job state changes for the running indicator
        unsub_created = bus.subscribe(EventType.JOB_CREATED, self._on_job_change)
        self._unsubscribe_callbacks.append(unsub_created)
        
        unsub_started = bus.subscribe(EventType.JOB_STARTED, self._on_job_change)
        self._unsubscribe_callbacks.append(unsub_started)
        
        unsub_completed = bus.subscribe(EventType.JOB_COMPLETED, self._on_job_change)
        self._unsubscribe_callbacks.append(unsub_completed)
        
        unsub_failed = bus.subscribe(EventType.JOB_FAILED, self._on_job_change)
        self._unsubscribe_callbacks.append(unsub_failed)
        
        unsub_stopped = bus.subscribe(EventType.JOB_STOPPED, self._on_job_change)
        self._unsubscribe_callbacks.append(unsub_stopped)
        
        # Also do an initial GPU fetch
        self._update_gpu_stats()
    
    def _on_gpu_update(self, event: Event):
        """Handle GPU update event."""
        stats = event.data.get('stats')
        if not stats:
            if hasattr(self, '_gpu_label'):
                self._gpu_label.text = '--'
            return
        
        util = stats.get('utilization_percent')
        if util is not None and hasattr(self, '_gpu_label'):
            self._gpu_label.text = f"{util:.0f}%"
    
    def _on_job_change(self, event: Event):
        """Handle job state change event to update running indicator."""
        if self._job_indicator_container:
            self._job_indicator_container.clear()
            with self._job_indicator_container:
                self._render_job_indicator_content()
    
    def _update_gpu_stats(self):
        """Fetch and update GPU statistics (initial fetch)."""
        try:
            gpu = get_gpu_summary()
            if gpu.get('available'):
                self._gpu_util = gpu.get('util_percent')
                display = gpu.get('util', '--')
                if hasattr(self, '_gpu_label'):
                    self._gpu_label.text = display
            else:
                if hasattr(self, '_gpu_label'):
                    self._gpu_label.text = '--'
        except Exception:
            # Silently ignore GPU polling errors
            pass
    
    async def _refresh(self):
        """Refresh page data."""
        ui.notify('Refreshing...', type='info', timeout=1000)
    
    def set_title(self, title: str):
        """Set the page title."""
        self.page_title.text = title
    
    def _cleanup(self):
        """Clean up event subscriptions when client disconnects."""
        for unsub in self._unsubscribe_callbacks:
            try:
                unsub()
            except Exception:
                pass
        self._unsubscribe_callbacks.clear()
    
    def register_cleanup(self):
        """Register cleanup handler - call this after render in a page context."""
        try:
            ui.context.client.on_disconnect(self._cleanup)
        except Exception:
            pass  # May fail if not in a proper client context
