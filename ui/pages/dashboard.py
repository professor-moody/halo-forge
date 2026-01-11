"""
Dashboard Page

Main overview page with system status, active jobs, and recent runs.
"""

import json
from pathlib import Path
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
        result = {'runs': [], 'steps': []}
        
        # Look for training logs/results
        results_dir = Path('results')
        if not results_dir.exists():
            return result
        
        # Search for training JSON files with loss data
        training_files = list(results_dir.glob('**/training*.json'))[:3]
        
        max_steps = 0
        for f in training_files:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    if 'loss_history' in data:
                        losses = data['loss_history']
                        max_steps = max(max_steps, len(losses))
                        result['runs'].append({
                            'name': f.stem[:15],
                            'loss': losses
                        })
            except (json.JSONDecodeError, IOError):
                continue
        
        # Generate step labels
        if max_steps > 0:
            step_interval = max(1, max_steps // 10)
            result['steps'] = [str(i * step_interval) for i in range(max_steps // step_interval + 1)]
        
        return result
    
    def _load_benchmark_data(self) -> dict:
        """Load benchmark results from results directory."""
        result = {'models': [], 'domains': ['Code', 'Reasoning', 'VLM', 'Audio']}
        
        results_dir = Path('results')
        if not results_dir.exists():
            return result
        
        # Aggregate scores by model
        model_scores = {}
        
        for domain in ['code', 'reasoning', 'vlm', 'audio']:
            domain_dir = results_dir / domain
            if not domain_dir.exists():
                continue
            
            for model_dir in domain_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                model_name = model_dir.name
                if model_name not in model_scores:
                    model_scores[model_name] = {'Code': None, 'Reasoning': None, 'VLM': None, 'Audio': None}
                
                # Read first result file
                result_files = list(model_dir.glob('*.json'))
                for rf in result_files[:1]:
                    try:
                        with open(rf) as fp:
                            data = json.load(fp)
                            # Extract score (handle different formats)
                            score = data.get('score') or data.get('accuracy') or data.get('pass_rate')
                            if score is not None:
                                # Convert to percentage if needed
                                if score <= 1:
                                    score *= 100
                                model_scores[model_name][domain.capitalize()] = round(score, 1)
                    except (json.JSONDecodeError, IOError):
                        continue
        
        # Convert to series format
        for model_name, scores in model_scores.items():
            score_list = [scores.get(d) or 0 for d in result['domains']]
            if any(s > 0 for s in score_list):  # Only include models with some data
                result['models'].append({
                    'name': model_name[:12],  # Truncate long names
                    'scores': score_list
                })
        
        return result
    
    async def _refresh_jobs(self):
        """Refresh jobs data."""
        ui.notify('Refreshing...', type='info', timeout=1000)
