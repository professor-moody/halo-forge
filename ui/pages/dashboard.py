"""
Dashboard Page

Main overview page with system status, active jobs, and recent runs.
"""

import json
from pathlib import Path
from typing import Callable, List
from nicegui import ui
from ui.theme import COLORS
from ui.state import state
from ui.services.hardware import get_gpu_summary
from ui.feature_flags import get_ui_feature_flags
from ui.services import (
    get_results_service,
    get_hardware_monitor,
    get_event_bus,
    get_modality_readiness_service,
    Event,
    EventType,
)


class Dashboard:
    """Dashboard page component."""
    
    def __init__(self):
        self._gpu_value_label = None
        self._gpu_subtitle_label = None
        self._active_jobs_container = None
        self._active_jobs_count_label = None
        self._unsubscribe_callbacks: List[Callable[[], None]] = []
        self.results_service = get_results_service()
        self.hardware_monitor = get_hardware_monitor()
        self.readiness_service = get_modality_readiness_service()
    
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
                    1,
                    is_gpu=True
                )
                self._render_stat_card(
                    "Active Jobs",
                    str(len(state.get_active_jobs())),
                    "Running now",
                    COLORS["running"],
                    "play_circle",
                    2,
                    is_active_jobs=True
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
            
            # Set up live GPU updates
            self._setup_gpu_polling()
            
            # Register cleanup on client disconnect
            ui.context.client.on_disconnect(self._cleanup)

            # Non-code modality readiness summary
            with ui.column().classes(
                f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-2'
            ):
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label('Non-Code Modality Readiness').classes(
                        f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                    )
                    ui.button(
                        'Refresh readiness',
                        icon='refresh',
                        on_click=self._refresh_readiness,
                    ).props('flat dense').classes(
                        f'text-[{COLORS["text_secondary"]}]'
                    )
                self._render_modality_readiness_summary()
            
            # Visualization charts grid
            with ui.element('div').classes('grid-panels w-full'):
                # Training History chart
                with ui.column().classes(
                    f'gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                    f'border border-[#2d343c] animate-in stagger-3 card-hover'
                ):
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label('Training History').classes(
                            f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                        )
                        ui.label('Recent runs').classes(
                            f'text-xs text-[{COLORS["text_muted"]}]'
                        )
                    self._render_training_chart()
                
                # Benchmark Comparison chart
                with ui.column().classes(
                    f'gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                    f'border border-[#2d343c] animate-in stagger-4 card-hover'
                ):
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.label('Benchmark Scores').classes(
                            f'text-base font-semibold text-[{COLORS["text_primary"]}]'
                        )
                        ui.link('View all', target='/results').classes(
                            f'text-xs text-[{COLORS["accent"]}] hover:underline'
                        )
                    self._render_benchmark_chart()
            
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
                    self._render_action_button('Run Benchmark', 'speed', '/benchmark')
                    flags = get_ui_feature_flags()
                    if flags.enable_inference_page:
                        self._render_action_button('Inference', 'bolt', '/inference')
                    if flags.enable_benchmark_advanced_page:
                        self._render_action_button('Benchmark+', 'view_array', '/benchmark-advanced')
                    if flags.enable_research_hub_page:
                        self._render_action_button('Research Hub', 'science', '/research-hub')
                    self._render_action_button('View Configs', 'settings', '/config')
                    self._render_action_button('Test Verifier', 'verified', '/verifiers')
    
    def _render_stat_card(
        self,
        title: str,
        value: str,
        subtitle: str,
        color: str,
        icon: str,
        stagger: int,
        is_gpu: bool = False,
        is_active_jobs: bool = False
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
                    value_label = ui.label(value).classes(
                        f'text-3xl font-bold text-[{COLORS["text_primary"]}] font-mono'
                    )
                    # Store reference for GPU card updates
                    if is_gpu:
                        self._gpu_value_label = value_label
                    # Store reference for Active Jobs count updates
                    if is_active_jobs:
                        self._active_jobs_count_label = value_label
                
                with ui.element('div').classes(
                    f'w-10 h-10 rounded-lg flex items-center justify-center bg-[{color}]/10'
                ):
                    ui.icon(icon, size='20px').classes(f'text-[{color}]')
            
            subtitle_label = ui.label(subtitle).classes(
                f'text-xs text-[{COLORS["text_secondary"]}]'
            )
            if is_gpu:
                self._gpu_subtitle_label = subtitle_label
    
    def _render_active_jobs(self):
        """Render the active jobs list container."""
        # Create a container we can update dynamically
        self._active_jobs_container = ui.column().classes('w-full gap-2')
        with self._active_jobs_container:
            self._render_active_jobs_content()
    
    def _render_active_jobs_content(self):
        """Render the actual active jobs content (can be re-rendered)."""
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
            training_runs = self.results_service.get_recent_training_runs(5)
            if training_runs:
                for run in training_runs:
                    self._render_training_run_row(run)
            else:
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

    def _render_training_run_row(self, run):
        """Render a persisted training summary row when no live state jobs exist."""
        with ui.row().classes(
            f'w-full items-center gap-3 p-3 rounded-lg '
            f'hover:bg-[{COLORS["bg_hover"]}] transition-colors'
        ):
            status_color = COLORS["success"] if run.weights_updated else COLORS["warning"]
            ui.icon("check_circle" if run.weights_updated else "info", size='18px').classes(
                f'text-[{status_color}]'
            )
            with ui.column().classes('flex-1 gap-0'):
                ui.label(f"{run.modality.upper()} • {Path(str(run.model_name)).name}").classes(
                    f'text-sm text-[{COLORS["text_primary"]}]'
                )
                reason = run.failure_reason or run.final_update_reason or "summary"
                ui.label(f"steps={run.total_train_steps_executed} • {reason}").classes(
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

    def _render_modality_readiness_summary(self):
        """Render non-code modality readiness rows."""
        try:
            report = self.readiness_service.get_effective_readiness()
        except Exception as e:
            ui.label(f"Readiness unavailable: {e}").classes(
                f'text-sm text-[{COLORS["warning"]}]'
            )
            return

        source_text = f"source={report.source}"
        if report.generated_at:
            source_text += f" • generated={report.generated_at}"
        if report.stale:
            source_text += " • stale=true"
        ui.label(source_text).classes(
            f'text-xs text-[{COLORS["text_muted"]}]'
        )
        burnin_meta = self.readiness_service.get_burnin_provenance()
        if burnin_meta.get("burnin_report_present"):
            burnin_status = str(burnin_meta.get("burnin_status") or "warn")
            burnin_color = self._status_color(burnin_status)
            ui.label(
                f"burnin status={burnin_status} "
                f"generated={burnin_meta.get('burnin_generated_at')}"
            ).classes(f'text-xs text-[{burnin_color}]')
        else:
            ui.label("burnin report unavailable (non-blocking)").classes(
                f'text-xs text-[{COLORS["warning"]}]'
            )

        for modality in ("vlm", "audio", "reasoning", "agentic"):
            entry = report.modalities.get(modality)
            if not entry:
                continue
            badge_color = self._status_color(entry.status)
            with ui.row().classes(
                f'w-full items-start justify-between gap-4 p-3 rounded-lg bg-[{COLORS["bg_secondary"]}]'
            ):
                with ui.column().classes('gap-1'):
                    with ui.row().classes('items-center gap-2'):
                        ui.label(modality.upper()).classes(
                            f'text-sm font-medium text-[{COLORS["text_primary"]}]'
                        )
                        ui.label(entry.status.upper()).classes(
                            f'text-xs px-2 py-0.5 rounded-full bg-[{badge_color}]/20 text-[{badge_color}]'
                        )
                    ui.label(
                        f"errors={len(entry.errors)} • warnings={len(entry.warnings)}"
                    ).classes(f'text-xs text-[{COLORS["text_muted"]}]')
                    if entry.errors:
                        ui.label(entry.errors[0]).classes(
                            f'text-xs text-[{COLORS["error"]}]'
                        )
                    elif entry.warnings:
                        ui.label(entry.warnings[0]).classes(
                            f'text-xs text-[{COLORS["warning"]}]'
                        )
                ui.label(entry.last_output_dir or "--").classes(
                    f'text-xs font-mono text-[{COLORS["text_muted"]}] max-w-[50%] truncate'
                )

    def _status_color(self, status: str) -> str:
        if status == "pass":
            return COLORS["success"]
        if status == "warn":
            return COLORS["warning"]
        return COLORS["error"]
    
    def _render_training_chart(self):
        """Render training history line chart."""
        # Load recent training data from results
        training_data = self._load_recent_training_data()
        
        if not training_data['runs']:
            with ui.column().classes('w-full items-center justify-center h-48 gap-2'):
                ui.icon('show_chart', size='32px').classes(f'text-[{COLORS["text_muted"]}]')
                ui.label('No training data yet').classes(f'text-sm text-[{COLORS["text_muted"]}]')
                ui.label('Complete a training run to see loss curves').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
            return
        
        # Build ECharts config
        series = []
        for run in training_data['runs'][:3]:  # Show up to 3 recent runs
            series.append({
                'name': run['name'],
                'type': 'line',
                'smooth': True,
                'symbol': 'none',
                'data': run['loss'],
                'lineStyle': {'width': 2}
            })
        
        ui.echart({
            'tooltip': {
                'trigger': 'axis',
                'backgroundColor': COLORS['bg_card'],
                'borderColor': '#2d343c',
                'textStyle': {'color': COLORS['text_primary']}
            },
            'legend': {
                'show': len(series) > 1,
                'bottom': 0,
                'textStyle': {'color': COLORS['text_secondary'], 'fontSize': 10}
            },
            'grid': {
                'left': 45,
                'right': 20,
                'top': 20,
                'bottom': 35 if len(series) > 1 else 20
            },
            'xAxis': {
                'type': 'category',
                'data': training_data['steps'],
                'axisLine': {'lineStyle': {'color': '#2d343c'}},
                'axisLabel': {'color': COLORS['text_muted'], 'fontSize': 10},
                'name': 'Step',
                'nameTextStyle': {'color': COLORS['text_muted'], 'fontSize': 10}
            },
            'yAxis': {
                'type': 'value',
                'name': 'Loss',
                'nameTextStyle': {'color': COLORS['text_muted'], 'fontSize': 10},
                'axisLine': {'lineStyle': {'color': '#2d343c'}},
                'axisLabel': {'color': COLORS['text_muted'], 'fontSize': 10},
                'splitLine': {'lineStyle': {'color': '#2d343c', 'type': 'dashed'}}
            },
            'color': [COLORS['primary'], COLORS['accent'], COLORS['info']],
            'series': series
        }).classes('w-full h-48')
    
    def _render_benchmark_chart(self):
        """Render benchmark comparison bar chart."""
        # Load benchmark results
        benchmark_data = self._load_benchmark_data()
        
        if not benchmark_data['models']:
            with ui.column().classes('w-full items-center justify-center h-48 gap-2'):
                ui.icon('bar_chart', size='32px').classes(f'text-[{COLORS["text_muted"]}]')
                ui.label('No benchmark data yet').classes(f'text-sm text-[{COLORS["text_muted"]}]')
                ui.label('Run benchmarks to see model comparisons').classes(
                    f'text-xs text-[{COLORS["text_muted"]}]'
                )
            return
        
        # Build series for each model
        series = []
        colors = [COLORS['primary'], COLORS['accent'], COLORS['info'], '#9b7ed9']
        for i, model in enumerate(benchmark_data['models']):
            series.append({
                'name': model['name'],
                'type': 'bar',
                'barGap': '10%',
                'data': model['scores'],
                'itemStyle': {'color': colors[i % len(colors)], 'borderRadius': [4, 4, 0, 0]}
            })
        
        ui.echart({
            'tooltip': {
                'trigger': 'axis',
                'axisPointer': {'type': 'shadow'},
                'backgroundColor': COLORS['bg_card'],
                'borderColor': '#2d343c',
                'textStyle': {'color': COLORS['text_primary']}
            },
            'legend': {
                'show': True,
                'bottom': 0,
                'textStyle': {'color': COLORS['text_secondary'], 'fontSize': 10}
            },
            'grid': {
                'left': 45,
                'right': 20,
                'top': 20,
                'bottom': 35
            },
            'xAxis': {
                'type': 'category',
                'data': benchmark_data['domains'],
                'axisLine': {'lineStyle': {'color': '#2d343c'}},
                'axisLabel': {'color': COLORS['text_muted'], 'fontSize': 10}
            },
            'yAxis': {
                'type': 'value',
                'name': 'Score %',
                'max': 100,
                'nameTextStyle': {'color': COLORS['text_muted'], 'fontSize': 10},
                'axisLine': {'lineStyle': {'color': '#2d343c'}},
                'axisLabel': {'color': COLORS['text_muted'], 'fontSize': 10},
                'splitLine': {'lineStyle': {'color': '#2d343c', 'type': 'dashed'}}
            },
            'series': series
        }).classes('w-full h-48')
    
    def _load_recent_training_data(self) -> dict:
        """Load training loss data from recent runs."""
        try:
            return self.results_service.get_dashboard_training_summary(max_runs=3)
        except Exception as e:
            print(f"[Dashboard] Failed to load training summary: {e}")
            return {'runs': [], 'steps': []}
    
    def _load_benchmark_data(self) -> dict:
        """Load benchmark chart data from canonical ResultsService aggregation."""
        try:
            return self.results_service.get_dashboard_benchmark_summary(max_models=5)
        except Exception as e:
            print(f"[Dashboard] Failed to load benchmark summary: {e}")
            return {'models': [], 'domains': ['Code', 'Reasoning', 'VLM', 'Audio', 'Agentic']}
    
    async def _refresh_jobs(self):
        """Refresh jobs data."""
        ui.notify('Refreshing...', type='info', timeout=1000)

    async def _refresh_readiness(self):
        """Trigger readiness refresh."""
        self.readiness_service.get_effective_readiness(force_refresh=True)
        ui.notify('Readiness refreshed', type='info', timeout=1200)
    
    def _setup_gpu_polling(self):
        """Set up GPU stats updates via event subscription."""
        bus = get_event_bus()
        
        # Subscribe to GPU updates
        unsub_gpu = bus.subscribe(EventType.GPU_UPDATE, self._on_gpu_update)
        self._unsubscribe_callbacks.append(unsub_gpu)
        
        # Subscribe to job state changes
        unsub_created = bus.subscribe(EventType.JOB_CREATED, self._on_job_change)
        self._unsubscribe_callbacks.append(unsub_created)
        
        unsub_completed = bus.subscribe(EventType.JOB_COMPLETED, self._on_job_change)
        self._unsubscribe_callbacks.append(unsub_completed)
        
        unsub_failed = bus.subscribe(EventType.JOB_FAILED, self._on_job_change)
        self._unsubscribe_callbacks.append(unsub_failed)
        
        unsub_stopped = bus.subscribe(EventType.JOB_STOPPED, self._on_job_change)
        self._unsubscribe_callbacks.append(unsub_stopped)
        
        # Also start hardware monitor if not running
        import asyncio
        if not self.hardware_monitor.is_running:
            asyncio.create_task(self.hardware_monitor.start())
    
    def _on_gpu_update(self, event: Event):
        """Handle GPU update event."""
        stats = event.data.get('stats')
        if not stats:
            return
        
        util = stats.get('utilization_percent')
        if self._gpu_value_label and util is not None:
            self._gpu_value_label.set_text(f"{util:.0f}%")
        
        if self._gpu_subtitle_label:
            name = (stats.get('name') or 'AMD GPU')[:20]
            temp = stats.get('temperature_c')
            if temp is not None:
                subtitle = f"{name} • {temp:.0f}°C"
            else:
                subtitle = name
            self._gpu_subtitle_label.set_text(subtitle)
    
    def _on_job_change(self, event: Event):
        """Handle job state change event."""
        # Update active jobs count
        active_jobs = state.get_active_jobs()
        if self._active_jobs_count_label:
            self._active_jobs_count_label.set_text(str(len(active_jobs)))
        
        # Re-render active jobs panel
        if self._active_jobs_container:
            self._active_jobs_container.clear()
            with self._active_jobs_container:
                self._render_active_jobs_content()
    
    def _get_recent_benchmark_results(self):
        """Get recent benchmark results from the results service."""
        try:
            return self.results_service.get_latest_results(5)
        except Exception:
            return []
    
    def _cleanup(self):
        """Clean up event subscriptions when client disconnects."""
        for unsub in self._unsubscribe_callbacks:
            try:
                unsub()
            except Exception:
                pass
        self._unsubscribe_callbacks.clear()
