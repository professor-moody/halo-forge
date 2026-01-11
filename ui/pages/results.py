"""
Results Page

View and compare benchmark results.
"""

from nicegui import ui, app
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
import json

from ui.theme import COLORS


@dataclass
class BenchmarkResult:
    """A benchmark result entry."""
    id: str
    model: str
    benchmark: str
    pass_at_1: float
    pass_at_5: Optional[float]
    pass_at_10: Optional[float]
    samples: int
    duration_seconds: float
    timestamp: datetime
    notes: Optional[str] = None


class Results:
    """Benchmark results page component."""
    
    RESULTS_DIR = Path("results")
    
    def __init__(self):
        self.results: list[BenchmarkResult] = []
        # Restore selection from storage
        stored_ids = app.storage.user.get('results_selected_ids', [])
        self.selected_ids: set[str] = set(stored_ids)
        self.sort_by: str = app.storage.user.get('results_sort_by', 'timestamp')
        self.sort_desc: bool = app.storage.user.get('results_sort_desc', True)
        
        self._load_results()
    
    def _load_results(self):
        """Load results from the results directory."""
        self.results = []
        
        # Load from results directory
        if self.RESULTS_DIR.exists():
            self._scan_results_directory()
        
        # If no real results, show demo data
        if not self.results:
            self._load_demo_results()
    
    def _scan_results_directory(self):
        """Scan results directory for benchmark JSON files."""
        result_id = 0
        
        # Only scan specific result file patterns - skip model checkpoints, configs, etc.
        # Valid result files: summary.json, baseline_*.json, *_benchmark.json, etc.
        valid_patterns = [
            'summary.json',
            'baseline_*.json',
            '*_benchmark.json',
            '*_test.json',
            '*_test*.json',
            '*_verify*.json',
        ]
        
        # Directories to skip (contain model artifacts, not results)
        skip_dirs = {'checkpoint', 'cycle_', 'adapter', 'tokenizer', 'tensorboard'}
        
        for json_file in self.RESULTS_DIR.glob('**/*.json'):
            try:
                # Skip non-result files by checking filename patterns
                filename = json_file.name.lower()
                
                # Skip common model/training artifact files
                if filename in (
                    'adapter_config.json', 'tokenizer_config.json', 
                    'special_tokens_map.json', 'added_tokens.json',
                    'vocab.json', 'tokenizer.json', 'trainer_state.json',
                    'config.json', 'generation_config.json'
                ):
                    continue
                
                # Skip files in checkpoint directories
                path_str = str(json_file).lower()
                if any(skip in path_str for skip in ['checkpoint-', '/cycle_', '/adapter', 'tensorboard']):
                    continue
                
                with open(json_file) as f:
                    data = json.load(f)
                
                # Handle list format - take first item
                if isinstance(data, list):
                    if not data:
                        continue
                    if isinstance(data[0], dict):
                        data = data[0]
                    else:
                        continue
                
                if not isinstance(data, dict):
                    continue
                
                # Skip if this doesn't look like a benchmark result
                # Must have at least one of these fields
                if not any(k in data for k in ['pass_at_k', 'pass_rate', 'accuracy', 'model_path', 'model_name', 'baseline']):
                    continue
                
                # Extract benchmark info
                result = self._parse_benchmark_result(json_file, data, result_id)
                if result:
                    self.results.append(result)
                    result_id += 1
                
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    
    def _parse_benchmark_result(self, json_file: Path, data: dict, result_id: int) -> Optional[BenchmarkResult]:
        """Parse a benchmark result from JSON data."""
        parts = json_file.relative_to(self.RESULTS_DIR).parts
        
        # Handle nested "baseline" structure (summary.json format)
        if 'baseline' in data and isinstance(data['baseline'], dict):
            baseline = data['baseline']
            model = data.get('model_name') or data.get('model_path') or 'Unknown'
            # Use parent directory as benchmark name
            benchmark = parts[1] if len(parts) > 1 else json_file.stem
            
            pass_at_k = baseline.get('pass_at_k', {})
            if isinstance(pass_at_k, dict):
                pass_at_1 = pass_at_k.get('1') or pass_at_k.get(1) or baseline.get('pass_at_1', 0)
            else:
                pass_at_1 = baseline.get('pass_at_1', 0)
            
            samples = baseline.get('total_samples') or baseline.get('total') or 0
            duration = data.get('total_time_sec') or 0
        else:
            # Direct format (baseline_*.json, *_test.json)
            model = data.get('model_path') or data.get('model_name') or data.get('model') or 'Unknown'
            benchmark = data.get('benchmark') or data.get('dataset') or json_file.stem
            
            # Handle pass_at_k as nested dict
            pass_at_k = data.get('pass_at_k', {})
            if isinstance(pass_at_k, dict):
                pass_at_1 = pass_at_k.get('1') or pass_at_k.get(1) or 0
                pass_at_5 = pass_at_k.get('5') or pass_at_k.get(5)
                pass_at_10 = pass_at_k.get('10') or pass_at_k.get(10)
            else:
                pass_at_1 = data.get('pass@1') or data.get('pass_at_1') or data.get('accuracy') or 0
                pass_at_5 = data.get('pass@5') or data.get('pass_at_5')
                pass_at_10 = data.get('pass@10') or data.get('pass_at_10')
            
            samples = data.get('total') or data.get('samples') or data.get('n_samples') or 0
            
            # Handle timing as nested dict
            timing = data.get('timing', {})
            if isinstance(timing, dict):
                duration = timing.get('total_time') or timing.get('total_time_sec') or 0
            else:
                duration = data.get('duration_seconds') or data.get('duration') or 0
        
        # Extract model short name from path
        if '/' in str(model):
            model = str(model).split('/')[-1]
        
        # Convert pass rates to percentages if they're decimals < 1
        # (they should already be 0-1 range, we'll display as percentage)
        pass_at_1 = float(pass_at_1) if pass_at_1 else 0.0
        
        # Get timestamp
        timestamp_str = data.get('timestamp') or data.get('created_at')
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(str(timestamp_str).replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                timestamp = datetime.fromtimestamp(json_file.stat().st_mtime)
        else:
            timestamp = datetime.fromtimestamp(json_file.stat().st_mtime)
        
        # Format benchmark name nicely
        benchmark_name = str(benchmark).replace('_', ' ').replace('-', ' ').title()
        if 'baseline' in json_file.name.lower():
            benchmark_name = f"Baseline {benchmark_name}"
        
        return BenchmarkResult(
            id=f"result-{result_id}",
            model=str(model),
            benchmark=benchmark_name,
            pass_at_1=pass_at_1,
            pass_at_5=float(pass_at_5) if pass_at_5 else None,
            pass_at_10=float(pass_at_10) if pass_at_10 else None,
            samples=int(samples),
            duration_seconds=float(duration),
            timestamp=timestamp,
            notes=str(json_file.relative_to(self.RESULTS_DIR)),
        )
    
    def _load_demo_results(self):
        """Load demo results for UI development when no real results exist."""
        self.results = [
            BenchmarkResult(
                id="demo-base-qwen-1",
                model="Qwen/Qwen2.5-Coder-3B",
                benchmark="HumanEval",
                pass_at_1=0.421,
                pass_at_5=0.584,
                pass_at_10=0.652,
                samples=164,
                duration_seconds=1847.3,
                timestamp=datetime.now(),
                notes="Demo: Base model baseline"
            ),
            BenchmarkResult(
                id="demo-raft-qwen-1",
                model="models/code_raft/cycle_5",
                benchmark="HumanEval",
                pass_at_1=0.518,
                pass_at_5=0.671,
                pass_at_10=0.732,
                samples=164,
                duration_seconds=2104.7,
                timestamp=datetime.now(),
                notes="Demo: After 5 RAFT cycles"
            ),
        ]
    
    def render(self):
        """Render the results page."""
        with ui.column().classes('page-content w-full gap-6 p-6'):
            # Header
            with ui.row().classes('w-full items-center justify-between animate-in'):
                ui.label('Benchmark Results').classes(
                    f'text-2xl font-bold text-[{COLORS["text_primary"]}]'
                )
                
                with ui.row().classes('gap-2'):
                    if self.selected_ids:
                        ui.button(
                            f'Compare ({len(self.selected_ids)})',
                            icon='compare',
                            on_click=self._show_comparison
                        ).props('unelevated').classes(
                            f'bg-[{COLORS["accent"]}] text-white'
                        )
                    
                    ui.button('Export', icon='download', on_click=self._export).props(
                        'flat'
                    ).classes(f'text-[{COLORS["text_secondary"]}]')
            
            # Summary stats
            with ui.row().classes('w-full gap-4 animate-in stagger-1'):
                self._stat_card('Total Runs', str(len(self.results)), 'analytics')
                best = max(self.results, key=lambda r: r.pass_at_1) if self.results else None
                self._stat_card(
                    'Best pass@1', 
                    f'{best.pass_at_1:.1%}' if best else '--',
                    'emoji_events'
                )
                avg = sum(r.pass_at_1 for r in self.results) / len(self.results) if self.results else 0
                self._stat_card('Avg pass@1', f'{avg:.1%}', 'leaderboard')
            
            # Results table
            with ui.column().classes(
                f'w-full gap-4 p-5 rounded-xl bg-[{COLORS["bg_card"]}] '
                f'border border-[#2d343c] animate-in stagger-2'
            ):
                self._render_table()
    
    def _stat_card(self, label: str, value: str, icon: str):
        """Render a statistics card."""
        with ui.column().classes(
            f'flex-1 min-w-[150px] gap-2 p-4 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c]'
        ):
            with ui.row().classes('items-center gap-2'):
                ui.icon(icon, size='20px').classes(f'text-[{COLORS["accent"]}]')
                ui.label(value).classes(
                    f'text-2xl font-bold text-[{COLORS["text_primary"]}]'
                )
            ui.label(label).classes(
                f'text-xs text-[{COLORS["text_muted"]}]'
            )
    
    def _render_table(self):
        """Render the results table."""
        # Header
        with ui.row().classes('w-full items-center justify-between mb-4'):
            ui.label('All Results').classes(
                f'text-base font-semibold text-[{COLORS["text_primary"]}]'
            )
            
            # Sort control
            with ui.row().classes('items-center gap-2'):
                ui.label('Sort:').classes(f'text-xs text-[{COLORS["text_muted"]}]')
                ui.select(
                    options=['timestamp', 'pass_at_1', 'model'],
                    value=self.sort_by,
                    on_change=lambda e: self._sort_results(e.value)
                ).props('outlined dense dark').classes('w-28')
        
        # Table header
        with ui.row().classes(
            f'w-full items-center gap-4 px-4 py-3 rounded-t-lg bg-[{COLORS["bg_secondary"]}]'
        ):
            ui.checkbox(
                value=len(self.selected_ids) == len(self.results),
                on_change=self._toggle_all
            ).classes('mr-2')
            
            ui.label('Model').classes(
                f'flex-1 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
            )
            ui.label('Benchmark').classes(
                f'w-24 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}]'
            )
            ui.label('pass@1').classes(
                f'w-20 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
            )
            ui.label('pass@5').classes(
                f'w-20 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
            )
            ui.label('Duration').classes(
                f'w-20 text-xs uppercase tracking-wider text-[{COLORS["text_muted"]}] text-right'
            )
        
        # Table rows
        sorted_results = self._get_sorted_results()
        
        for result in sorted_results:
            self._render_result_row(result)
    
    def _render_result_row(self, result: BenchmarkResult):
        """Render a single result row."""
        is_selected = result.id in self.selected_ids
        
        with ui.row().classes(
            f'w-full items-center gap-4 px-4 py-3 '
            f'border-b border-[#2d343c] '
            + (f'bg-[{COLORS["primary"]}]/5' if is_selected else f'hover:bg-[{COLORS["bg_hover"]}]')
        ):
            ui.checkbox(
                value=is_selected,
                on_change=lambda e, r=result: self._toggle_selection(r.id, e.value)
            ).classes('mr-2')
            
            # Model
            with ui.column().classes('flex-1 gap-0'):
                ui.label(result.model).classes(
                    f'text-sm text-[{COLORS["text_primary"]}] truncate'
                )
                if result.notes:
                    ui.label(result.notes).classes(
                        f'text-xs text-[{COLORS["text_muted"]}] truncate'
                    )
            
            # Benchmark
            ui.label(result.benchmark).classes(
                f'w-24 text-sm text-[{COLORS["text_secondary"]}]'
            )
            
            # pass@1
            ui.label(f'{result.pass_at_1:.1%}').classes(
                f'w-20 text-sm font-mono text-[{COLORS["primary"]}] text-right'
            )
            
            # pass@5
            ui.label(f'{result.pass_at_5:.1%}' if result.pass_at_5 else '--').classes(
                f'w-20 text-sm font-mono text-[{COLORS["text_secondary"]}] text-right'
            )
            
            # Duration
            duration_str = f'{result.duration_seconds/60:.1f}m'
            ui.label(duration_str).classes(
                f'w-20 text-sm font-mono text-[{COLORS["text_muted"]}] text-right'
            )
    
    def _get_sorted_results(self) -> list[BenchmarkResult]:
        """Get results sorted by current criteria."""
        key_map = {
            'timestamp': lambda r: r.timestamp,
            'pass_at_1': lambda r: r.pass_at_1,
            'model': lambda r: r.model,
        }
        key_fn = key_map.get(self.sort_by, key_map['timestamp'])
        return sorted(self.results, key=key_fn, reverse=self.sort_desc)
    
    def _sort_results(self, sort_by: str):
        """Update sort criteria."""
        if self.sort_by == sort_by:
            self.sort_desc = not self.sort_desc
        else:
            self.sort_by = sort_by
            self.sort_desc = True
        app.storage.user['results_sort_by'] = self.sort_by
        app.storage.user['results_sort_desc'] = self.sort_desc
        ui.navigate.to('/results')
    
    def _toggle_selection(self, result_id: str, selected: bool):
        """Toggle selection of a result."""
        if selected:
            self.selected_ids.add(result_id)
        else:
            self.selected_ids.discard(result_id)
        app.storage.user['results_selected_ids'] = list(self.selected_ids)
    
    def _toggle_all(self, e):
        """Toggle selection of all results."""
        if e.value:
            self.selected_ids = {r.id for r in self.results}
        else:
            self.selected_ids.clear()
        app.storage.user['results_selected_ids'] = list(self.selected_ids)
        ui.navigate.to('/results')  # Force UI rebuild
    
    def _show_comparison(self):
        """Show comparison chart for selected results."""
        selected_results = [r for r in self.results if r.id in self.selected_ids]
        
        with ui.dialog() as dialog, ui.card().classes(
            f'bg-[{COLORS["bg_card"]}] p-6 min-w-[600px]'
        ):
            ui.label('Comparison').classes(
                f'text-lg font-semibold text-[{COLORS["text_primary"]}]'
            )
            
            # Bar chart
            chart_data = [[r.model.split('/')[-1], r.pass_at_1 * 100] for r in selected_results]
            
            ui.echart({
                'backgroundColor': 'transparent',
                'xAxis': {
                    'type': 'category',
                    'data': [d[0] for d in chart_data],
                    'axisLabel': {'color': COLORS['text_secondary']},
                    'axisLine': {'lineStyle': {'color': COLORS['text_muted']}},
                },
                'yAxis': {
                    'type': 'value',
                    'name': 'pass@1 (%)',
                    'max': 100,
                    'axisLabel': {'color': COLORS['text_secondary']},
                    'axisLine': {'lineStyle': {'color': COLORS['text_muted']}},
                    'splitLine': {'lineStyle': {'color': '#2d343c'}},
                },
                'series': [{
                    'type': 'bar',
                    'data': [d[1] for d in chart_data],
                    'itemStyle': {
                        'color': COLORS['primary'],
                        'borderRadius': [4, 4, 0, 0],
                    },
                    'label': {
                        'show': True,
                        'position': 'top',
                        'formatter': '{c:.1f}%',
                        'color': COLORS['text_secondary'],
                    },
                }],
                'grid': {'top': 40, 'right': 20, 'bottom': 40, 'left': 60},
            }).classes('w-full h-64')
            
            with ui.row().classes('w-full justify-end mt-4'):
                ui.button('Close', on_click=dialog.close).props('flat').classes(
                    f'text-[{COLORS["text_secondary"]}]'
                )
        
        dialog.open()
    
    def _export(self):
        """Export results to JSON."""
        data = [
            {
                'model': r.model,
                'benchmark': r.benchmark,
                'pass_at_1': r.pass_at_1,
                'pass_at_5': r.pass_at_5,
                'pass_at_10': r.pass_at_10,
                'samples': r.samples,
                'duration_seconds': r.duration_seconds,
                'timestamp': r.timestamp.isoformat(),
                'notes': r.notes,
            }
            for r in self.results
        ]
        
        ui.notify('Results exported to console (see browser dev tools)', type='positive')
        print(json.dumps(data, indent=2))
