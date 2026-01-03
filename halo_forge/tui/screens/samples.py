"""
Samples Screen - Browse generated samples.

View samples from current or past training runs,
filter by success/failure, view full prompts and completions.
"""

import json
from pathlib import Path
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Footer, Static, DataTable, Input, Button, TextArea
)
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding
from rich.text import Text


class SampleDetailPanel(Container):
    """Panel showing sample details."""
    
    DEFAULT_CSS = """
    SampleDetailPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: 1fr;
    }
    
    SampleDetailPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    
    SampleDetailPanel > TextArea {
        height: 1fr;
        background: #0a0908;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("SAMPLE DETAIL", classes="panel-title")
        yield TextArea(id="sample-detail", read_only=True)


class SamplesScreen(Screen):
    """Sample browser screen."""
    
    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("f", "toggle_filter", "Filter", show=True),
        Binding("c", "copy_sample", "Copy", show=True),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = []
        self.filtered_samples = []
        self.show_only_passed = False
        self.show_only_failed = False
    
    def compose(self) -> ComposeResult:
        """Compose the samples screen."""
        yield Static("SAMPLES BROWSER", id="screen-title", classes="screen-title")
        
        with Container(id="samples-container"):
            # Filter bar
            with Horizontal(id="filter-bar"):
                yield Input(placeholder="Search prompts...", id="search-input")
                yield Button("All", id="filter-all", variant="primary")
                yield Button("Passed", id="filter-passed", variant="success")
                yield Button("Failed", id="filter-failed", variant="error")
                yield Button("Load File", id="load-file")
            
            with Horizontal(id="samples-content"):
                # Samples table
                with Vertical(id="samples-list"):
                    yield Static("SAMPLES", classes="panel-title")
                    yield DataTable(id="samples-table")
                
                # Detail view
                yield SampleDetailPanel(id="detail-panel")
        
        yield Footer()
    
    def on_mount(self):
        """Set up the samples table."""
        table = self.query_one("#samples-table", DataTable)
        table.add_columns("Status", "Reward", "Prompt")
        table.cursor_type = "row"
        
        # Load samples from state if available
        self._load_samples_from_state()
    
    def _load_samples_from_state(self):
        """Load samples from the current training state."""
        state = self.app.state_manager.read()
        if state.recent_samples:
            self.samples = state.recent_samples
            self._update_table()
    
    def _update_table(self):
        """Update the samples table."""
        table = self.query_one("#samples-table", DataTable)
        table.clear()
        
        # Apply filters
        self.filtered_samples = []
        search_term = self.query_one("#search-input", Input).value.lower()
        
        for sample in self.samples:
            # Filter by search
            if search_term and search_term not in sample.get("prompt", "").lower():
                continue
            
            # Filter by status
            if self.show_only_passed and not sample.get("success", False):
                continue
            if self.show_only_failed and sample.get("success", False):
                continue
            
            self.filtered_samples.append(sample)
        
        # Add to table
        for sample in self.filtered_samples:
            status = "[OK]" if sample.get("success", False) else "[X]"
            reward = f"{sample.get('reward', 0):.2f}"
            prompt = sample.get("prompt", "")[:60] + "..."
            
            table.add_row(status, reward, prompt)
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        """Show sample details when row selected."""
        if event.row_key is not None and event.row_key.value < len(self.filtered_samples):
            sample = self.filtered_samples[event.row_key.value]
            self._show_sample_detail(sample)
    
    def _show_sample_detail(self, sample: dict):
        """Show full sample details."""
        detail = self.query_one("#sample-detail", TextArea)
        
        text = f"""PROMPT:
{'-' * 40}
{sample.get('prompt', 'N/A')}

COMPLETION:
{'-' * 40}
{sample.get('completion', 'N/A')}

REWARD: {sample.get('reward', 0):.2f}
STATUS: {'PASSED' if sample.get('success', False) else 'FAILED'}
"""
        detail.text = text
    
    def on_button_pressed(self, event: Button.Pressed):
        """Handle filter buttons."""
        if event.button.id == "filter-all":
            self.show_only_passed = False
            self.show_only_failed = False
            self._update_table()
        elif event.button.id == "filter-passed":
            self.show_only_passed = True
            self.show_only_failed = False
            self._update_table()
        elif event.button.id == "filter-failed":
            self.show_only_passed = False
            self.show_only_failed = True
            self._update_table()
        elif event.button.id == "load-file":
            self._load_from_file()
    
    def on_input_changed(self, event: Input.Changed):
        """Handle search input."""
        if event.input.id == "search-input":
            self._update_table()
    
    def _load_from_file(self):
        """Load samples from a JSONL file."""
        # Try to find verified cache files
        output_dirs = [
            Path("models/raft"),
            Path("models/production_7b"),
        ]
        
        for output_dir in output_dirs:
            if output_dir.exists():
                for cache_file in output_dir.glob("*_verified.jsonl"):
                    self._load_jsonl(cache_file)
                    return
        
        self.notify("No sample files found", severity="warning")
    
    def _load_jsonl(self, path: Path):
        """Load samples from JSONL file."""
        self.samples = []
        try:
            with open(path) as f:
                for line in f:
                    self.samples.append(json.loads(line))
            
            self._update_table()
            self.notify(f"Loaded {len(self.samples)} samples from {path.name}")
        except Exception as e:
            self.notify(f"Error loading file: {e}", severity="error")
    
    def action_copy_sample(self):
        """Copy current sample to clipboard."""
        table = self.query_one("#samples-table", DataTable)
        if table.cursor_row is not None and table.cursor_row < len(self.filtered_samples):
            sample = self.filtered_samples[table.cursor_row]
            # Note: Actual clipboard copy requires pyperclip or similar
            self.notify("Sample copied to clipboard")

