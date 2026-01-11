"""
Datasets Page

Browse and preview available training datasets.
"""

from nicegui import ui
from typing import Optional
from dataclasses import dataclass

from ui.theme import COLORS


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    description: str
    size: str
    domain: str
    format_type: str
    source: str
    example: dict


class Datasets:
    """Dataset browser page component."""
    
    DATASETS = [
        # SFT Datasets
        DatasetInfo(
            name="Alpaca",
            description="Stanford Alpaca instruction-following dataset",
            size="52K examples",
            domain="General",
            format_type="Instruction",
            source="stanford_alpaca/alpaca",
            example={
                "instruction": "Give three tips for staying healthy.",
                "input": "",
                "output": "1. Eat a balanced diet...\n2. Exercise regularly...\n3. Get enough sleep..."
            }
        ),
        DatasetInfo(
            name="MetaMath",
            description="Mathematical reasoning with chain-of-thought",
            size="395K examples",
            domain="Math",
            format_type="QA",
            source="meta-math/MetaMathQA",
            example={
                "question": "Calculate 15% of 240.",
                "answer": "To find 15% of 240:\n15/100 × 240 = 0.15 × 240 = 36"
            }
        ),
        DatasetInfo(
            name="GSM8K",
            description="Grade-school math word problems",
            size="8.5K examples",
            domain="Math",
            format_type="QA",
            source="gsm8k",
            example={
                "question": "James has 3 boxes with 12 apples each. How many apples in total?",
                "answer": "3 × 12 = 36 apples"
            }
        ),
        DatasetInfo(
            name="xLAM",
            description="Function calling and tool use training",
            size="60K examples",
            domain="Agentic",
            format_type="Function Call",
            source="Salesforce/xlam-function-calling-60k",
            example={
                "instruction": "Get the current weather in New York",
                "tools": "[{\"name\": \"get_weather\", \"parameters\": {...}}]",
                "output": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"New York\"}}"
            }
        ),
        # RAFT Prompts
        DatasetInfo(
            name="HumanEval Prompts",
            description="164 Python function completion prompts",
            size="164 prompts",
            domain="Code",
            format_type="Prompt",
            source="data/rlvr/humaneval_prompts.jsonl",
            example={
                "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    ...",
                "task_id": "HumanEval/0"
            }
        ),
        DatasetInfo(
            name="MBPP Prompts",
            description="Mostly Basic Python Problems",
            size="500 prompts",
            domain="Code",
            format_type="Prompt",
            source="data/rlvr/mbpp_train_prompts.jsonl",
            example={
                "prompt": "Write a function to find the factorial of a number.",
                "task_id": "MBPP/1"
            }
        ),
    ]
    
    def __init__(self):
        self.selected_dataset: Optional[DatasetInfo] = None
        self.filter_domain: str = "All"
    
    def render(self):
        """Render the datasets page."""
        with ui.column().classes('page-content w-full gap-6 p-6'):
            # Header
            with ui.row().classes('w-full items-center justify-between animate-in'):
                ui.label('Datasets').classes(
                    f'text-2xl font-bold text-[{COLORS["text_primary"]}]'
                )
                
                # Filter
                with ui.row().classes('items-center gap-2'):
                    ui.label('Filter:').classes(
                        f'text-sm text-[{COLORS["text_muted"]}]'
                    )
                    ui.select(
                        options=['All', 'Code', 'Math', 'General', 'Agentic'],
                        value=self.filter_domain,
                        on_change=lambda e: self._set_filter(e.value)
                    ).props('outlined dense dark').classes('w-32')
            
            # Stats row
            with ui.row().classes('w-full gap-4 animate-in stagger-1'):
                self._stat_card('Total Datasets', str(len(self.DATASETS)), 'storage')
                self._stat_card('Domains', '4', 'category')
                self._stat_card('Total Size', '~500K', 'data_usage')
            
            # Dataset grid
            with ui.row().classes('w-full gap-4 flex-wrap'):
                for i, dataset in enumerate(self._filtered_datasets()):
                    self._render_dataset_card(dataset, i)
            
            # Preview panel
            if self.selected_dataset:
                self._render_preview()
    
    def _filtered_datasets(self) -> list[DatasetInfo]:
        """Get datasets filtered by domain."""
        if self.filter_domain == "All":
            return self.DATASETS
        return [d for d in self.DATASETS if d.domain == self.filter_domain]
    
    def _set_filter(self, domain: str):
        """Set the domain filter."""
        self.filter_domain = domain
        self.selected_dataset = None
        ui.navigate.to('/datasets')
    
    def _stat_card(self, label: str, value: str, icon: str):
        """Render a statistics card."""
        with ui.column().classes(
            f'flex-1 min-w-[120px] gap-2 p-4 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c]'
        ):
            with ui.row().classes('items-center gap-2'):
                ui.icon(icon, size='18px').classes(f'text-[{COLORS["accent"]}]')
                ui.label(value).classes(
                    f'text-xl font-bold text-[{COLORS["text_primary"]}]'
                )
            ui.label(label).classes(
                f'text-xs text-[{COLORS["text_muted"]}]'
            )
    
    def _render_dataset_card(self, dataset: DatasetInfo, index: int):
        """Render a dataset card."""
        is_selected = self.selected_dataset == dataset
        
        with ui.card().classes(
            f'w-[280px] p-4 rounded-xl cursor-pointer card-hover '
            f'animate-in stagger-{min(index + 2, 6)} '
            + (f'border-2 border-[{COLORS["primary"]}] bg-[{COLORS["primary"]}]/5' if is_selected
               else f'border border-[#2d343c] bg-[{COLORS["bg_card"]}]')
        ).on('click', lambda d=dataset: self._select_dataset(d)):
            with ui.column().classes('gap-3'):
                # Header
                with ui.row().classes('w-full items-start justify-between'):
                    ui.label(dataset.name).classes(
                        f'text-sm font-semibold text-[{COLORS["text_primary"]}]'
                    )
                    ui.label(dataset.domain).classes(
                        f'px-2 py-0.5 rounded text-xs bg-[{COLORS["info"]}]/10 text-[{COLORS["info"]}]'
                    )
                
                # Description
                ui.label(dataset.description).classes(
                    f'text-xs text-[{COLORS["text_secondary"]}] line-clamp-2'
                )
                
                # Meta
                with ui.row().classes('w-full items-center justify-between'):
                    ui.label(dataset.size).classes(
                        f'text-xs font-medium text-[{COLORS["accent"]}]'
                    )
                    ui.label(dataset.format_type).classes(
                        f'text-xs text-[{COLORS["text_muted"]}]'
                    )
    
    def _select_dataset(self, dataset: DatasetInfo):
        """Select a dataset to preview."""
        self.selected_dataset = dataset
        ui.navigate.to('/datasets')  # Force refresh
    
    def _render_preview(self):
        """Render the dataset preview panel."""
        dataset = self.selected_dataset
        
        with ui.column().classes(
            f'w-full gap-4 p-6 rounded-xl bg-[{COLORS["bg_card"]}] '
            f'border border-[#2d343c] animate-in'
        ):
            # Header
            with ui.row().classes('w-full items-center justify-between'):
                with ui.column().classes('gap-1'):
                    ui.label(dataset.name).classes(
                        f'text-lg font-semibold text-[{COLORS["text_primary"]}]'
                    )
                    ui.label(dataset.source).classes(
                        f'text-xs text-[{COLORS["text_muted"]}] font-mono'
                    )
                
                ui.button('Use in Training', icon='play_arrow', 
                         on_click=lambda: ui.navigate.to('/training')).props(
                    'unelevated'
                ).classes(f'bg-[{COLORS["primary"]}] text-white')
            
            # Example
            ui.label('Example').classes(
                f'text-sm font-medium text-[{COLORS["text_secondary"]}] mt-2'
            )
            
            import json
            ui.html(f'''<pre class="w-full p-4 rounded-lg font-mono text-xs overflow-x-auto" 
                         style="background: {COLORS["bg_primary"]}; color: {COLORS["text_secondary"]}; white-space: pre-wrap;">{json.dumps(dataset.example, indent=2)}</pre>''')
