"""
Results Page

View and compare benchmark results.
"""

from nicegui import ui
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
        self.selected_ids: set[str] = set()
        self.sort_by: str = "timestamp"
        self.sort_desc: bool = True
        
        self._load_results()
    
    def _load_results(self):
        """Load results from the results directory."""
        # Demo results for UI development
        self.results = [
            BenchmarkResult(
                id="base-qwen-1",
                model="Qwen/Qwen2.5-Coder-3B",
                benchmark="HumanEval",
                pass_at_1=0.421,
                pass_at_5=0.584,
                pass_at_10=0.652,
                samples=164,
                duration_seconds=1847.3,
                timestamp=datetime.now(),
                notes="Base model baseline"
            ),
            BenchmarkResult(
                id="sft-qwen-1",
                model="models/code_sft",
                benchmark="HumanEval",
                pass_at_1=0.457,
                pass_at_5=0.612,
                pass_at_10=0.683,
                samples=164,
                duration_seconds=1923.1,
                timestamp=datetime.now(),
                notes="After 3 epochs SFT on Alpaca"
            ),
            BenchmarkResult(
                id="raft-qwen-1",
                model="models/code_raft/cycle_5",
                benchmark="HumanEval",
                pass_at_1=0.518,
                pass_at_5=0.671,
                pass_at_10=0.732,
                samples=164,
                duration_seconds=2104.7,
                timestamp=datetime.now(),
                notes="After 5 RAFT cycles"
            ),
            BenchmarkResult(
                id="base-lfm-1",
                model="amd/lfm-2b",
                benchmark="HumanEval",
                pass_at_1=0.312,
                pass_at_5=0.445,
                pass_at_10=0.521,
                samples=164,
                duration_seconds=1562.4,
                timestamp=datetime.now(),
                notes="LFM base model"
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
        ui.navigate.to('/results')
    
    def _toggle_selection(self, result_id: str, selected: bool):
        """Toggle selection of a result."""
        if selected:
            self.selected_ids.add(result_id)
        else:
            self.selected_ids.discard(result_id)
    
    def _toggle_all(self, e):
        """Toggle selection of all results."""
        if e.value:
            self.selected_ids = {r.id for r in self.results}
        else:
            self.selected_ids.clear()
    
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
