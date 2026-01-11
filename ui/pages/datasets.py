"""
Datasets Page

Browse and preview available training datasets.
"""

from nicegui import ui
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import json

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
    is_local: bool = False  # Whether this is a local file


class Datasets:
    """Dataset browser page component."""
    
    # Built-in/HuggingFace datasets
    BUILTIN_DATASETS = [
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
    ]
    
    LOCAL_DATA_DIRS = [Path("data"), Path("data/rlvr"), Path("data/sft")]
    
    def __init__(self):
        self.selected_dataset: Optional[DatasetInfo] = None
        self.filter_domain: str = "All"
        self.datasets: list[DatasetInfo] = []
        self._load_datasets()
        
        # UI container references for dynamic updates
        self._cards_container = None
        self._preview_container = None
        self._stats_container = None
    
    def _load_datasets(self):
        """Load all available datasets (builtin + local)."""
        self.datasets = list(self.BUILTIN_DATASETS)
        
        # Scan local data directories for JSONL files
        for data_dir in self.LOCAL_DATA_DIRS:
            if data_dir.exists():
                self._scan_local_datasets(data_dir)
    
    def _scan_local_datasets(self, data_dir: Path):
        """Scan a directory for local dataset files."""
        for jsonl_file in data_dir.glob("*.jsonl"):
            # Skip if already in builtin
            if any(d.source.endswith(jsonl_file.name) for d in self.BUILTIN_DATASETS):
                continue
            
            try:
                # Count lines and get example
                line_count = 0
                example = {}
                with open(jsonl_file) as f:
                    for i, line in enumerate(f):
                        if i == 0:
                            example = json.loads(line)
                        line_count += 1
                
                # Infer domain from path or content
                domain = "Code"
                if "math" in jsonl_file.name.lower():
                    domain = "Math"
                elif "sft" in str(jsonl_file).lower():
                    domain = "General"
                elif "agentic" in str(jsonl_file).lower() or "xlam" in jsonl_file.name.lower():
                    domain = "Agentic"
                
                # Infer format from content
                format_type = "Prompt"
                if "instruction" in example:
                    format_type = "Instruction"
                elif "question" in example:
                    format_type = "QA"
                elif "tests" in example:
                    format_type = "Prompt+Tests"
                
                self.datasets.append(DatasetInfo(
                    name=jsonl_file.stem.replace("_", " ").title(),
                    description=f"Local dataset from {jsonl_file.parent.name}/",
                    size=f"{line_count} examples",
                    domain=domain,
                    format_type=format_type,
                    source=str(jsonl_file),
                    example=example,
                    is_local=True
                ))
            except (json.JSONDecodeError, IOError):
                continue
    
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
            
            # Stats row - store reference for updates
            self._stats_container = ui.row().classes('w-full gap-4 animate-in stagger-1')
            with self._stats_container:
                self._render_stats()
            
            # Dataset grid - store reference for selection highlighting
            self._cards_container = ui.row().classes('w-full gap-4 flex-wrap')
            with self._cards_container:
                self._render_dataset_cards()
            
            # Preview panel - store reference for content updates
            self._preview_container = ui.column().classes('w-full')
            with self._preview_container:
                if self.selected_dataset:
                    self._render_preview()
    
    def _filtered_datasets(self) -> list[DatasetInfo]:
        """Get datasets filtered by domain."""
        if self.filter_domain == "All":
            return self.datasets
        return [d for d in self.datasets if d.domain == self.filter_domain]
    
    def _render_stats(self):
        """Render stats cards."""
        filtered = self._filtered_datasets()
        domains = set(d.domain for d in filtered)
        local_count = sum(1 for d in filtered if d.is_local)
        
        self._stat_card('Total Datasets', str(len(filtered)), 'storage')
        self._stat_card('Domains', str(len(domains)), 'category')
        self._stat_card('Local Files', str(local_count), 'folder')
    
    def _render_dataset_cards(self):
        """Render all dataset cards."""
        for i, dataset in enumerate(self._filtered_datasets()):
            self._render_dataset_card(dataset, i)
    
    def _set_filter(self, domain: str):
        """Set the domain filter.
        
        FIXED: Updates UI dynamically instead of navigating.
        """
        self.filter_domain = domain
        self.selected_dataset = None
        
        # Update stats
        if self._stats_container:
            self._stats_container.clear()
            with self._stats_container:
                self._render_stats()
        
        # Update cards
        if self._cards_container:
            self._cards_container.clear()
            with self._cards_container:
                self._render_dataset_cards()
        
        # Clear preview
        if self._preview_container:
            self._preview_container.clear()
        
        # NOTE: Removed ui.navigate.to('/datasets') which was causing page reset
    
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
        """Select a dataset to preview.
        
        FIXED: Updates UI dynamically instead of navigating (which killed state).
        """
        self.selected_dataset = dataset
        
        # Re-render cards to update selection highlighting
        if self._cards_container:
            self._cards_container.clear()
            with self._cards_container:
                self._render_dataset_cards()
        
        # Show preview panel
        if self._preview_container:
            self._preview_container.clear()
            with self._preview_container:
                self._render_preview()
        
        # NOTE: Removed ui.navigate.to('/datasets') which was causing full page reload
        # and resetting self.selected_dataset to None
    
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
                         style="background: {COLORS["bg_primary"]}; color: {COLORS["text_secondary"]}; white-space: pre-wrap;">{json.dumps(dataset.example, indent=2)}</pre>''', sanitize=False)
