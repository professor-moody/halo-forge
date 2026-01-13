"""
Monitor Page

Real-time job monitoring with loss charts and log viewer.
"""

from nicegui import ui
from datetime import datetime
from typing import Optional, Callable, List
from pathlib import Path
import asyncio
import json

from ui.theme import COLORS
from ui.state import state, JobState
from ui.services import TrainingService, get_event_bus, Event, EventType
from ui.components.notifications import (
    notify_training_stopped,
    notify_job_completed,
    notify_job_failed,
)


class Monitor:
    """Real-time job monitoring page component."""
    
    def __init__(self, job_id: Optional[str] = None):
        self.job_id = job_id
        self.job: Optional[JobState] = None
        self.update_timer = None
        self.log_lines: list[str] = []  # Legacy - kept for event handler compatibility
        self.training_service = TrainingService(state)
        self._update_task: Optional[asyncio.Task] = None
        self._unsubscribe_callbacks: List[Callable[[], None]] = []
        
        # Persistent log handling
        self._all_log_lines: list[str] = []  # Full log history
        self._displayed_log_count: int = 0   # Track what's already displayed
        
        # References to dynamic UI elements for live updates
        self._duration_label = None
        self._duration_timer = None
        self._progress_percent_label = None
        self._progress_bar = None
        self._epoch_label = None
        self._step_label = None
        self._loss_label = None
        self._lr_label = None
        self._grad_norm_label = None
        self._verification_label = None
        
        if job_id:
            self.job = state.get_job(job_id)
            # Load existing logs from file on mount
            self._load_logs_from_file()
    
    def _load_logs_from_file(self):
        """Load existing logs from persistent log file."""
        if not self.job:
            return
        
        # Try job's log_file_path first
        log_file_path = self.job.log_file_path
        
        # Fallback: construct path from output_dir
        if not log_file_path and self.job.output_dir:
            log_file_path = Path(self.job.output_dir) / f"{self.job_id}_training.log"
        
        if log_file_path and Path(log_file_path).exists():
            try:
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.rstrip('\n')
                        if line:
                            self._all_log_lines.append(line)
            except Exception as e:
                print(f"[Monitor] Error loading log file: {e}")
    
    def _get_log_file_path(self) -> Optional[Path]:
        """Get the path to the persistent log file."""
        if not self.job:
            return None
        
        if self.job.log_file_path:
            return Path(self.job.log_file_path)
        
        if self.job.output_dir:
            return Path(self.job.output_dir) / f"{self.job_id}_training.log"
        
        return None
    
    def render(self):
        """Render the monitor page."""
        with ui.column().classes('page-content w-full gap-6 p-6'):
            if not self.job:
                self._render_no_job()
                return
            
            # Header with job info
            with ui.row().classes(
                'w-full items-center justify-between animate-in'
            ):
                with ui.column().classes('gap-1'):
                    ui.label(self.job.name).classes(
                        f'text-2xl font-bold text-[{COLORS["text_primary"]}]'
                    )
                    with ui.row().classes('items-center gap-3'):
                        # Status badge
                        self._render_status_badge()
                        # Duration (stored for live updates)
                        self._duration_label = ui.label(f'Duration: {self.job.duration_str}').classes(
                            f'text-sm text-[{COLORS["text_secondary"]}]'
                        )
                        # Job ID
                        ui.label(f'ID: {self.job.id}').classes(
                            f'text-sm text-[{COLORS["text_muted"]}] font-mono'
                        )
                
                # Controls
                with ui.row().classes('items-center gap-2'):
                    if self.job.status == 'running':
                        ui.button('Stop', icon='stop', on_click=self._stop_job).props(
                            'flat'
                        ).classes(f'text-[{COLORS["error"]}]')
                    
                    ui.button(icon='refresh', on_click=self._refresh).props(
                        'flat round'
                    ).classes(f'text-[{COLORS["text_secondary"]}]')
            
            # Progress section
            with ui.column().classes(
                f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-1'
            ):
                self._render_progress()
            
            # Main content grid
            with ui.row().classes('w-full gap-6 flex-wrap'):
                # Loss chart
                with ui.column().classes(
                    f'flex-[2] min-w-[400px] gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                    f'border border-[#2d343c] animate-in stagger-2'
                ):
                    self._render_chart_section()
                
                # Metrics panel
                with ui.column().classes(
                    f'flex-1 min-w-[250px] gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                    f'border border-[#2d343c] animate-in stagger-3'
                ):
                    self._render_metrics_panel()
            
            # Log viewer
            with ui.column().classes(
                f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-4'
            ):
                self._render_log_viewer()
            
            # Start live updates if job is running
            if self.job and self.job.status == 'running':
                self._start_live_updates()
                # Timer for duration updates (duration changes every second)
                self._duration_timer = ui.timer(1.0, self._tick_duration)
            
            # Register cleanup on client disconnect (handles navigation away)
            ui.context.client.on_disconnect(self._cleanup_subscriptions)
    
    def _render_no_job(self):
        """Render when no job is selected."""
        with ui.column().classes(
            'w-full items-center justify-center py-16 gap-4 animate-in'
        ):
            ui.icon('hourglass_empty', size='64px').classes(
                f'text-[{COLORS["text_muted"]}]'
            )
            ui.label('No job selected').classes(
                f'text-xl text-[{COLORS["text_secondary"]}]'
            )
            ui.label('Start a training run or select a job from the dashboard.').classes(
                f'text-sm text-[{COLORS["text_muted"]}]'
            )
            ui.button('Go to Dashboard', on_click=lambda: ui.navigate.to('/')).props(
                'flat'
            ).classes(f'text-[{COLORS["accent"]}]')
    
    def _render_status_badge(self):
        """Render the job status badge."""
        status = self.job.status
        color = COLORS.get(status, COLORS['text_secondary'])
        
        with ui.row().classes(
            f'items-center gap-1.5 px-2.5 py-1 rounded-full bg-[{color}]/10'
        ):
            # Animated dot for running
            if status == 'running':
                ui.element('div').classes(
                    f'w-2 h-2 rounded-full bg-[{color}] running-glow'
                )
            else:
                ui.element('div').classes(
                    f'w-2 h-2 rounded-full bg-[{color}]'
                )
            ui.label(status.capitalize()).classes(
                f'text-xs font-medium text-[{color}]'
            )
    
    def _render_progress(self):
        """Render the progress section."""
        with ui.row().classes('w-full items-center justify-between'):
            ui.label('Progress').classes(
                f'text-sm font-semibold text-[{COLORS["text_primary"]}]'
            )
            # Store reference for live updates
            self._progress_percent_label = ui.label(f'{self.job.progress_percent:.1f}%').classes(
                f'text-sm font-mono text-[{COLORS["accent"]}]'
            )
        
        # Progress bar (store reference)
        with ui.element('div').classes(
            f'w-full h-2 rounded-full bg-[{COLORS["bg_secondary"]}] overflow-hidden'
        ):
            self._progress_bar = ui.element('div').classes(
                f'h-full bg-[{COLORS["primary"]}] progress-fill rounded-full'
            ).style(f'width: {self.job.progress_percent}%')
        
        # Progress details (store references)
        with ui.row().classes('w-full gap-6 mt-2'):
            # Epoch/Cycle
            with ui.row().classes('items-center gap-2'):
                epoch_label = 'Cycle' if self.job.type == 'raft' else 'Epoch'
                if self.job.type == 'raft':
                    current = self.job.current_cycle
                else:
                    # Display epoch as float if fractional, int otherwise
                    current = self.job.current_epoch
                    if isinstance(current, float) and current % 1 != 0:
                        current = f'{current:.1f}'
                    else:
                        current = int(current)
                total = self.job.total_cycles if self.job.type == 'raft' else self.job.total_epochs
                ui.label(f'{epoch_label}:').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
                self._epoch_label = ui.label(f'{current}/{total}').classes(
                    f'text-sm font-mono text-[{COLORS["text_secondary"]}]'
                )
            
            # Step
            with ui.row().classes('items-center gap-2'):
                ui.label('Step:').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
                self._step_label = ui.label(f'{self.job.current_step}/{self.job.total_steps or "?"}').classes(
                    f'text-sm font-mono text-[{COLORS["text_secondary"]}]'
                )
    
    def _render_chart_section(self):
        """Render the loss chart section."""
        with ui.row().classes('w-full items-center justify-between'):
            ui.label('Training Loss').classes(
                f'text-base font-semibold text-[{COLORS["text_primary"]}]'
            )
            
            with ui.row().classes('gap-2'):
                ui.button('Loss', on_click=lambda: None).props(
                    'flat dense size=sm'
                ).classes(f'text-[{COLORS["primary"]}] bg-[{COLORS["primary"]}]/10')
                ui.button('LR', on_click=lambda: None).props(
                    'flat dense size=sm'
                ).classes(f'text-[{COLORS["text_muted"]}]')
        
        # Chart container
        self.chart = ui.echart({
            'backgroundColor': 'transparent',
            'grid': {
                'top': 30,
                'right': 20,
                'bottom': 30,
                'left': 50,
            },
            'xAxis': {
                'type': 'value',
                'name': 'Step',
                'nameLocation': 'middle',
                'nameGap': 25,
                'axisLine': {'lineStyle': {'color': COLORS['text_muted']}},
                'axisLabel': {'color': COLORS['text_muted']},
            },
            'yAxis': {
                'type': 'value',
                'name': 'Loss',
                'axisLine': {'lineStyle': {'color': COLORS['text_muted']}},
                'axisLabel': {'color': COLORS['text_muted']},
                'splitLine': {'lineStyle': {'color': '#2d343c'}},
            },
            'tooltip': {
                'trigger': 'axis',
                'backgroundColor': COLORS['bg_card'],
                'borderColor': '#2d343c',
                'textStyle': {'color': COLORS['text_primary']},
            },
            'series': [{
                'type': 'line',
                'smooth': True,
                'symbol': 'none',
                'lineStyle': {
                    'width': 2,
                    'color': COLORS['primary'],
                },
                'areaStyle': {
                    'opacity': 0.1,
                    'color': COLORS['primary'],
                },
                'data': self._get_loss_data(),
            }],
        }).classes('w-full h-64')
    
    def _get_loss_data(self) -> list:
        """Get loss data for the chart."""
        if not self.job_id or self.job_id not in state.metrics_history:
            # Demo data
            return [[i, 2.5 - (i * 0.02) + (0.1 * (i % 5))] for i in range(100)]
        
        loss_points = state.metrics_history[self.job_id].get('loss', [])
        return [[p.step, p.value] for p in loss_points]
    
    def _render_metrics_panel(self):
        """Render the current metrics panel with stored references for live updates."""
        ui.label('Current Metrics').classes(
            f'text-base font-semibold text-[{COLORS["text_primary"]}]'
        )
        
        with ui.column().classes('w-full gap-3 mt-2'):
            # Loss
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Loss').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                val = f'{self.job.latest_loss:.4f}' if self.job.latest_loss else '--'
                self._loss_label = ui.label(val).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            
            # Learning Rate
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Learning Rate').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                val = f'{self.job.latest_lr:.2e}' if self.job.latest_lr else '--'
                self._lr_label = ui.label(val).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            
            # Grad Norm
            with ui.row().classes('w-full items-center justify-between'):
                ui.label('Grad Norm').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                val = f'{self.job.latest_grad_norm:.4f}' if self.job.latest_grad_norm else '--'
                self._grad_norm_label = ui.label(val).classes(
                    f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                )
            
            # Verification (RAFT only)
            if self.job.type == 'raft':
                ui.separator().classes('my-2')
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Verification').classes(f'text-sm text-[{COLORS["text_secondary"]}]')
                    val = f'{self.job.verification_rate:.1%}' if self.job.verification_rate is not None else '--'
                    self._verification_label = ui.label(val).classes(
                        f'text-sm font-mono text-[{COLORS["text_primary"]}]'
                    )
    
    def _render_log_viewer(self):
        """Render the log viewer section."""
        with ui.row().classes('w-full items-center justify-between'):
            ui.label('Training Logs').classes(
                f'text-base font-semibold text-[{COLORS["text_primary"]}]'
            )
            
            with ui.row().classes('gap-2'):
                ui.button(icon='vertical_align_bottom', on_click=self._scroll_to_bottom).props(
                    'flat round dense size=sm'
                ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Scroll to bottom')
                
                ui.button(icon='content_copy', on_click=self._copy_logs).props(
                    'flat round dense size=sm'
                ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Copy logs')
                
                ui.button(icon='download', on_click=self._download_logs).props(
                    'flat round dense size=sm'
                ).classes(f'text-[{COLORS["text_muted"]}]').tooltip('Download logs')
        
        # Log container - with data attribute for scroll targeting, increased height
        self.log_container = ui.column().classes(
            f'w-full h-96 overflow-y-auto p-4 rounded-lg bg-[{COLORS["bg_primary"]}] '
            f'font-mono text-xs leading-relaxed'
        ).props('data-log-container')
        
        with self.log_container:
            # Display existing logs (loaded from file or empty)
            if self._all_log_lines:
                for line in self._all_log_lines:
                    color = self._get_log_color(line)
                    ui.label(line).classes(f'text-[{color}]')
                    self._displayed_log_count += 1
            else:
                # Show placeholder if no logs yet
                ui.label('Waiting for logs...').classes(
                    f'text-[{COLORS["text_muted"]}] italic'
                )
    
    async def _refresh(self):
        """Refresh the monitor data."""
        if self.job_id:
            self.job = state.get_job(self.job_id)
            self._update_metrics_display()
            self._update_logs_display()
        ui.notify('Refreshed', type='info', timeout=1000)
    
    def _start_live_updates(self):
        """Start live updates for running job using event subscriptions."""
        if not self.job_id:
            return
        
        bus = get_event_bus()
        
        # Subscribe to metrics updates
        unsub_metrics = bus.subscribe(EventType.METRICS_UPDATE, self._on_metrics_event)
        self._unsubscribe_callbacks.append(unsub_metrics)
        
        # Subscribe to log lines
        unsub_logs = bus.subscribe(EventType.LOG_LINE, self._on_log_event)
        self._unsubscribe_callbacks.append(unsub_logs)
        
        # Subscribe to job completion
        unsub_completed = bus.subscribe(EventType.JOB_COMPLETED, self._on_job_completed)
        self._unsubscribe_callbacks.append(unsub_completed)
        
        # Subscribe to job failed
        unsub_failed = bus.subscribe(EventType.JOB_FAILED, self._on_job_failed)
        self._unsubscribe_callbacks.append(unsub_failed)
        
        # Subscribe to job stopped
        unsub_stopped = bus.subscribe(EventType.JOB_STOPPED, self._on_job_stopped)
        self._unsubscribe_callbacks.append(unsub_stopped)
        
        # Subscribe to checkpoint saves
        unsub_checkpoint = bus.subscribe(EventType.CHECKPOINT_SAVED, self._on_checkpoint)
        self._unsubscribe_callbacks.append(unsub_checkpoint)
    
    def _on_metrics_event(self, event: Event):
        """Handle metrics update event."""
        if event.job_id != self.job_id:
            return
        
        # Update job state
        self.job = state.get_job(self.job_id)
        
        # Update UI
        self._update_metrics_display()
        self._update_chart()
    
    def _on_log_event(self, event: Event):
        """Handle new log line event."""
        if event.job_id != self.job_id:
            return
        
        line = event.data.get('line', '')
        timestamp = event.data.get('timestamp', '')
        if line:
            # Format with timestamp for consistency with file logs
            formatted_line = f"[{timestamp}] {line}" if timestamp else line
            self._all_log_lines.append(formatted_line)
            self.log_lines.append(line)  # Keep legacy for compatibility
            self._update_logs_display()
    
    def _on_job_completed(self, event: Event):
        """Handle job completion event."""
        if event.job_id != self.job_id:
            return
        
        self.job = state.get_job(self.job_id)
        notify_job_completed(self.job.name if self.job else "Job")
        self._cleanup_subscriptions()
    
    def _on_job_failed(self, event: Event):
        """Handle job failed event."""
        if event.job_id != self.job_id:
            return
        
        self.job = state.get_job(self.job_id)
        error_msg = event.data.get('error', 'Unknown error')
        notify_job_failed(self.job.name if self.job else "Job", error_msg)
        self._cleanup_subscriptions()
    
    def _on_job_stopped(self, event: Event):
        """Handle job stopped event."""
        if event.job_id != self.job_id:
            return
        
        self.job = state.get_job(self.job_id)
        notify_training_stopped(self.job.name if self.job else "Job")
        self._cleanup_subscriptions()
    
    def _on_checkpoint(self, event: Event):
        """Handle checkpoint saved event."""
        if event.job_id != self.job_id:
            return
        
        # Checkpoint notification is already handled by TrainingService
        pass
    
    def _cleanup_subscriptions(self):
        """Unsubscribe from all events and stop timers."""
        for unsub in self._unsubscribe_callbacks:
            try:
                unsub()
            except Exception:
                pass
        self._unsubscribe_callbacks.clear()
        
        # Stop duration timer
        if self._duration_timer:
            self._duration_timer.cancel()
            self._duration_timer = None
    
    def _tick_duration(self):
        """Timer callback to update duration every second."""
        if not self.job_id:
            return
        
        # Always fetch fresh job state to get accurate duration
        self.job = state.get_job(self.job_id)
        
        if self._duration_label and self.job:
            try:
                self._duration_label.set_text(f'Duration: {self.job.duration_str}')
            except Exception:
                pass  # UI element may have been destroyed
        
        # Stop timer if job is no longer running
        if self.job and self.job.status not in ('running', 'pending'):
            if self._duration_timer:
                try:
                    self._duration_timer.cancel()
                    self._duration_timer = None
                except Exception:
                    pass
    
    def _update_metrics_display(self):
        """Update all dynamic UI elements with current job state."""
        if not self.job:
            return
        
        # Refresh job state
        self.job = state.get_job(self.job_id)
        if not self.job:
            return
        
        # Duration
        if self._duration_label:
            self._duration_label.set_text(f'Duration: {self.job.duration_str}')
        
        # Progress
        if self._progress_percent_label:
            self._progress_percent_label.set_text(f'{self.job.progress_percent:.1f}%')
        if self._progress_bar:
            self._progress_bar.style(f'width: {self.job.progress_percent}%')
        
        # Epoch/Cycle
        if self._epoch_label:
            if self.job.type == 'raft':
                self._epoch_label.set_text(f'{self.job.current_cycle}/{self.job.total_cycles}')
            else:
                # Display epoch as float if fractional, int otherwise
                current = self.job.current_epoch
                if isinstance(current, float) and current % 1 != 0:
                    epoch_str = f'{current:.1f}'
                else:
                    epoch_str = str(int(current))
                self._epoch_label.set_text(f'{epoch_str}/{self.job.total_epochs}')
        
        # Step
        if self._step_label:
            self._step_label.set_text(f'{self.job.current_step}/{self.job.total_steps or "?"}')
        
        # Metrics
        if self._loss_label:
            val = f'{self.job.latest_loss:.4f}' if self.job.latest_loss else '--'
            self._loss_label.set_text(val)
        
        if self._lr_label:
            val = f'{self.job.latest_lr:.2e}' if self.job.latest_lr else '--'
            self._lr_label.set_text(val)
        
        if self._grad_norm_label:
            val = f'{self.job.latest_grad_norm:.4f}' if self.job.latest_grad_norm else '--'
            self._grad_norm_label.set_text(val)
        
        if self._verification_label and self.job.verification_rate is not None:
            self._verification_label.set_text(f'{self.job.verification_rate:.1%}')
    
    def _update_chart(self):
        """Update the loss chart with new data."""
        if hasattr(self, 'chart') and self.chart:
            loss_data = self._get_loss_data()
            self.chart.options['series'][0]['data'] = loss_data
            self.chart.update()
    
    def _update_logs_display(self):
        """Update the logs display with new entries (append-only for performance)."""
        if not hasattr(self, 'log_container') or not self.log_container:
            return
        
        # On first render, load from _all_log_lines (includes file-loaded logs)
        # Then append only new lines
        new_lines = self._all_log_lines[self._displayed_log_count:]
        
        if not new_lines and self._displayed_log_count == 0:
            # Fallback: Get logs from training service if no logs loaded
            logs = self.training_service.get_logs(self.job_id, last_n=100)
            for entry in logs:
                line = entry.get('line', '')
                timestamp = entry.get('timestamp', '')
                formatted = f"[{timestamp}] {line}" if timestamp else line
                if formatted not in self._all_log_lines:
                    self._all_log_lines.append(formatted)
            new_lines = self._all_log_lines[self._displayed_log_count:]
        
        if new_lines:
            with self.log_container:
                for line in new_lines:
                    color = self._get_log_color(line)
                    ui.label(line).classes(f'text-[{color}]')
                    self._displayed_log_count += 1
            
            # Auto-scroll to bottom
            self._scroll_to_bottom()
    
    def _get_log_color(self, line: str) -> str:
        """Determine log line color."""
        line_lower = line.lower()
        if 'error' in line_lower or 'failed' in line_lower:
            return COLORS['error']
        elif 'warning' in line_lower:
            return COLORS['warning']
        elif 'saved' in line_lower or 'checkpoint' in line_lower:
            return COLORS['success']
        elif 'loading' in line_lower or 'starting' in line_lower:
            return COLORS['info']
        return COLORS['text_secondary']
    
    async def _stop_job(self):
        """Stop the current job."""
        if not self.job:
            return
        
        with ui.dialog() as dialog, ui.card().classes(f'bg-[{COLORS["bg_card"]}] p-6'):
            ui.label('Stop Training?').classes(
                f'text-lg font-semibold text-[{COLORS["text_primary"]}]'
            )
            ui.label('This will terminate the training process. Progress may be lost.').classes(
                f'text-sm text-[{COLORS["text_secondary"]}] mt-2'
            )
            
            with ui.row().classes('w-full justify-end gap-2 mt-4'):
                ui.button('Cancel', on_click=dialog.close).props('flat').classes(
                    f'text-[{COLORS["text_secondary"]}]'
                )
                ui.button('Stop', on_click=lambda: asyncio.create_task(self._confirm_stop(dialog))).props(
                    'unelevated'
                ).classes(f'bg-[{COLORS["error"]}] text-white')
        
        dialog.open()
    
    async def _confirm_stop(self, dialog):
        """Confirm stopping the job."""
        dialog.close()
        
        if self.job:
            # Actually stop the job via TrainingService
            success = await self.training_service.stop_job(self.job.id)
            
            if success:
                notify_training_stopped(self.job.name)
                self.job = state.get_job(self.job_id)  # Refresh job state
            else:
                notify_job_failed(self.job.name, "Failed to stop training")
    
    def _scroll_to_bottom(self):
        """Scroll log viewer to bottom."""
        if hasattr(self, 'log_container') and self.log_container:
            ui.run_javascript(f'document.querySelector("[data-log-container]").scrollTop = 999999')
    
    def _copy_logs(self):
        """Copy ALL logs to clipboard from persistent file or memory."""
        log_text = ""
        
        # Try to read from persistent log file first (has full history)
        log_file = self._get_log_file_path()
        if log_file and log_file.exists():
            try:
                log_text = log_file.read_text(encoding='utf-8').strip()
            except Exception:
                pass
        
        # Fallback to in-memory logs (use strip() check, not truthiness)
        if not log_text:
            log_text = '\n'.join(self._all_log_lines).strip()
        
        # Final fallback to service buffer
        if not log_text:
            logs = self.training_service.get_logs(self.job_id, last_n=1000)
            log_text = '\n'.join([entry.get('line', '') for entry in logs]).strip()
        
        if not log_text:
            ui.notify('No logs available to copy', type='warning', timeout=1500)
            return
        
        # Use json.dumps for proper JS string escaping with error handling
        ui.run_javascript(f'''
            navigator.clipboard.writeText({json.dumps(log_text)})
                .then(() => {{ /* success */ }})
                .catch((err) => {{ 
                    console.error("Clipboard copy failed:", err);
                    // Fallback: create a temporary textarea
                    const ta = document.createElement('textarea');
                    ta.value = {json.dumps(log_text)};
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand('copy');
                    document.body.removeChild(ta);
                }});
        ''')
        line_count = len(log_text.strip().split('\n'))
        ui.notify(f'Copied {line_count} log lines to clipboard', type='positive', timeout=1500)
    
    async def _download_logs(self):
        """Download ALL logs as a text file from persistent file or memory."""
        log_text = ""
        
        # Try to read from persistent log file first (has full history)
        log_file = self._get_log_file_path()
        if log_file and log_file.exists():
            try:
                log_text = log_file.read_text(encoding='utf-8').strip()
            except Exception:
                pass
        
        # Fallback to in-memory logs (use strip() check, not truthiness)
        if not log_text:
            log_text = '\n'.join(self._all_log_lines).strip()
        
        # Final fallback to service buffer
        if not log_text:
            logs = self.training_service.get_logs(self.job_id, last_n=1000)
            log_text = '\n'.join([entry.get('line', '') for entry in logs]).strip()
        
        if not log_text:
            ui.notify('No logs available to download', type='warning', timeout=1500)
            return
        
        job_name = self.job.name.replace(' ', '_').replace(':', '-') if self.job else 'training'
        filename = f"{job_name}_logs.txt"
        
        # Trigger browser download
        line_count = len(log_text.strip().split('\n'))
        ui.download(log_text.encode('utf-8'), filename)
        ui.notify(f'Downloading {line_count} log lines', type='positive', timeout=1500)


class MonitorList:
    """List of all jobs for monitoring."""
    
    def render(self):
        """Render the job list page."""
        with ui.column().classes('page-content w-full gap-6 p-6'):
            ui.label('Monitor Jobs').classes(
                f'text-2xl font-bold text-[{COLORS["text_primary"]}] animate-in'
            )
            
            # Active jobs
            active_jobs = state.get_active_jobs()
            if active_jobs:
                with ui.column().classes(
                    f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                    f'border border-[#2d343c] animate-in stagger-1'
                ):
                    ui.label('Active Jobs').classes(
                        f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                    )
                    
                    for job in active_jobs:
                        self._render_job_card(job)
            
            # Recent jobs
            recent = state.get_recent_jobs(10)
            completed = [j for j in recent if j.status != 'running']
            
            with ui.column().classes(
                f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-2'
            ):
                ui.label('Recent Jobs').classes(
                    f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                )
                
                if not completed and not active_jobs:
                    with ui.column().classes('w-full items-center py-8 gap-2'):
                        ui.icon('inbox', size='48px').classes(
                            f'text-[{COLORS["text_muted"]}]'
                        )
                        ui.label('No jobs yet').classes(
                            f'text-sm text-[{COLORS["text_muted"]}]'
                        )
                else:
                    for job in completed:
                        self._render_job_card(job)
    
    def _render_job_card(self, job: JobState):
        """Render a job card in the list."""
        status_color = COLORS.get(job.status, COLORS['text_secondary'])
        
        with ui.row().classes(
            f'w-full items-center gap-4 p-4 rounded-lg bg-[{COLORS["bg_secondary"]}] '
            f'hover:bg-[{COLORS["bg_hover"]}] transition-colors cursor-pointer'
        ).on('click', lambda j=job: ui.navigate.to(f'/monitor/{j.id}')):
            # Status indicator
            if job.status == 'running':
                ui.element('div').classes(
                    f'w-3 h-3 rounded-full bg-[{status_color}] running-glow'
                )
            else:
                ui.element('div').classes(
                    f'w-3 h-3 rounded-full bg-[{status_color}]'
                )
            
            # Job info
            with ui.column().classes('flex-1 gap-0.5'):
                ui.label(job.name).classes(
                    f'text-sm font-medium text-[{COLORS["text_primary"]}]'
                )
                ui.label(f'{job.type.upper()} • {job.duration_str}').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
            
            # Progress or status
            if job.status == 'running':
                ui.label(f'{job.progress_percent:.0f}%').classes(
                    f'text-sm font-mono text-[{COLORS["primary"]}]'
                )
            else:
                ui.label(job.status.capitalize()).classes(
                    f'text-xs text-[{status_color}]'
                )
            
            # Arrow
            ui.icon('chevron_right', size='20px').classes(
                f'text-[{COLORS["text_muted"]}]'
            )
