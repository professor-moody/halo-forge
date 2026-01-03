"""
Comparison Screen - Compare training runs.

Load and compare metrics from multiple training runs.
"""

import json
from pathlib import Path
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Static, DataTable, Button, DirectoryTree
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding
from rich.text import Text


class RunSummaryPanel(Container):
    """Panel showing run summary."""
    
    DEFAULT_CSS = """
    RunSummaryPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
        min-height: 15;
    }
    
    RunSummaryPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    """
    
    def __init__(self, title: str = "RUN", **kwargs):
        super().__init__(**kwargs)
        self._title = title
    
    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="panel-title")
        yield Static("", id="run-summary")


class ComparisonScreen(Screen):
    """Run comparison screen."""
    
    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("1", "select_run_1", "Select Run 1", show=True),
        Binding("2", "select_run_2", "Select Run 2", show=True),
    ]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run1_stats = None
        self.run2_stats = None
    
    def compose(self) -> ComposeResult:
        """Compose the comparison screen."""
        yield Static("RUN COMPARISON", id="screen-title", classes="screen-title")
        
        with Container(id="comparison-container"):
            # Run selection
            with Horizontal(id="run-selection"):
                yield Button("Load Run 1", id="load-run-1", variant="primary")
                yield Button("Load Run 2", id="load-run-2", variant="primary")
                yield Button("Compare", id="compare-btn", variant="success")
            
            # Side by side comparison
            with Horizontal(id="comparison-panels"):
                yield RunSummaryPanel(title="RUN 1", id="run1-panel")
                yield RunSummaryPanel(title="RUN 2", id="run2-panel")
            
            # Comparison table
            with Container(id="comparison-table-container"):
                yield Static("METRICS COMPARISON", classes="panel-title")
                yield DataTable(id="comparison-table")
        
        yield Footer()
    
    def on_mount(self):
        """Set up the comparison table."""
        table = self.query_one("#comparison-table", DataTable)
        table.add_columns("Metric", "Run 1", "Run 2", "Diff")
    
    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id == "load-run-1":
            self._load_run(1)
        elif event.button.id == "load-run-2":
            self._load_run(2)
        elif event.button.id == "compare-btn":
            self._compare_runs()
    
    def _load_run(self, run_num: int):
        """Load a training run from statistics file."""
        # Look for raft_statistics.json files
        search_dirs = [
            Path("models"),
            Path("."),
        ]
        
        stats_files = []
        for d in search_dirs:
            if d.exists():
                stats_files.extend(d.glob("**/raft_statistics.json"))
        
        if not stats_files:
            self.notify("No training runs found", severity="warning")
            return
        
        # For now, just load the first found
        # In a full implementation, we'd show a file picker
        stats_file = stats_files[0] if run_num == 1 else (stats_files[1] if len(stats_files) > 1 else stats_files[0])
        
        try:
            with open(stats_file) as f:
                stats = json.load(f)
            
            if run_num == 1:
                self.run1_stats = stats
                self._update_run_panel(1, stats, stats_file)
            else:
                self.run2_stats = stats
                self._update_run_panel(2, stats, stats_file)
            
            self.notify(f"Loaded run {run_num} from {stats_file.parent.name}")
        except Exception as e:
            self.notify(f"Error loading run: {e}", severity="error")
    
    def _update_run_panel(self, run_num: int, stats: list, path: Path):
        """Update run summary panel."""
        panel_id = f"#run{run_num}-panel"
        summary = self.query_one(f"{panel_id} #run-summary", Static)
        
        if not stats:
            summary.update("No data")
            return
        
        # Calculate summary
        num_cycles = len(stats)
        total_time = sum(s.get("elapsed_minutes", 0) for s in stats)
        avg_kept = sum(s.get("kept", 0) for s in stats) / num_cycles if num_cycles else 0
        final_reward = stats[-1].get("avg_reward", 0) if stats else 0
        
        text = Text()
        text.append(f"Path: {path.parent}\n", style="#6b635a")
        text.append(f"Cycles: {num_cycles}\n", style="bold")
        text.append(f"Total Time: {total_time:.1f} min\n", style="bold")
        text.append(f"Avg Samples Kept: {avg_kept:.0f}\n", style="#2dd4bf")
        text.append(f"Final Avg Reward: {final_reward:.3f}\n", style="#2dd4bf")
        
        text.append("\nCycle Breakdown:\n", style="#6b635a")
        for s in stats:
            text.append(f"  Cycle {s.get('cycle', '?')}: ", style="#6b635a")
            text.append(f"{s.get('kept', 0)} kept, ", style="bold")
            text.append(f"{s.get('avg_reward', 0):.3f} reward\n", style="#2dd4bf")
        
        summary.update(text)
    
    def _compare_runs(self):
        """Compare the two loaded runs."""
        table = self.query_one("#comparison-table", DataTable)
        table.clear()
        
        if not self.run1_stats or not self.run2_stats:
            self.notify("Load both runs first", severity="warning")
            return
        
        # Calculate metrics for comparison
        def calc_metrics(stats):
            if not stats:
                return {}
            return {
                "Cycles": len(stats),
                "Total Time (min)": sum(s.get("elapsed_minutes", 0) for s in stats),
                "Avg Samples Kept": sum(s.get("kept", 0) for s in stats) / len(stats),
                "Final Reward": stats[-1].get("avg_reward", 0),
                "Total Samples": sum(s.get("total_samples", 0) for s in stats),
            }
        
        m1 = calc_metrics(self.run1_stats)
        m2 = calc_metrics(self.run2_stats)
        
        for metric in m1.keys():
            v1 = m1.get(metric, 0)
            v2 = m2.get(metric, 0)
            diff = v2 - v1
            
            if isinstance(v1, float):
                v1_str = f"{v1:.2f}"
                v2_str = f"{v2:.2f}"
                diff_str = f"{diff:+.2f}"
            else:
                v1_str = str(v1)
                v2_str = str(v2)
                diff_str = f"{diff:+}"
            
            table.add_row(metric, v1_str, v2_str, diff_str)
        
        self.notify("Comparison complete")

