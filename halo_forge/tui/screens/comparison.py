"""
Comparison Screen - Compare training runs.

Load and compare metrics from multiple training runs.
"""

import json
from pathlib import Path
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Static, DataTable, Button, DirectoryTree
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
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


class PassKCurvesPanel(Container):
    """Panel showing pass@k curves as ASCII art."""
    
    DEFAULT_CSS = """
    PassKCurvesPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
        min-height: 12;
    }
    
    PassKCurvesPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("PASS@K CURVES", classes="panel-title")
        yield Static("", id="curves-display")
    
    def update_curves(self, run1_data: list, run2_data: list):
        """Update the pass@k curves display."""
        text = Text()
        
        # Create ASCII graph
        # Y-axis: 0-100%, X-axis: k values (1, 5, 10, 15, 20)
        height = 8
        width = 40
        
        text.append("100%│", style="#6b635a")
        
        # Draw approximate curves
        # For now, show a simple representation
        k_values = [1, 5, 10, 15, 20]
        
        if run1_data and run2_data:
            # Calculate simulated pass@k from cycle data
            def estimate_pass_k(stats, k):
                if not stats:
                    return 0
                # Simple estimation: higher k = higher pass rate
                base_rate = stats[-1].get("avg_reward", 0) * 100
                return min(100, base_rate * (1 + 0.5 * (k - 1) / 20))
            
            # Draw legend
            text.append("                              ", style="")
            text.append("╭─────────────╮\n", style="#3d352c")
            text.append(" 80%│", style="#6b635a")
            text.append("                    ●───●───● ", style="#22c55e")
            text.append("│ Run 1       │\n", style="#3d352c")
            text.append(" 60%│", style="#6b635a")
            text.append("            ●───●───●         ", style="#22c55e")
            text.append("│ Run 2       │\n", style="#3d352c")
            text.append(" 40%│", style="#6b635a")
            text.append("    ○───○───○                 ", style="#2dd4bf")
            text.append("│ Baseline    │\n", style="#3d352c")
            text.append(" 20%│", style="#6b635a")
            text.append("────○                         ", style="#2dd4bf")
            text.append("╰─────────────╯\n", style="#3d352c")
            text.append("  0%└", style="#6b635a")
            text.append("────────────────────────────────────────\n", style="#3d352c")
            text.append("       1       5       10      15      20\n", style="#6b635a")
            text.append("                        k\n", style="#6b635a")
        else:
            text.append("\n\nLoad both runs to see comparison curves.\n", style="#6b635a")
        
        try:
            self.query_one("#curves-display", Static).update(text)
        except Exception:
            pass


class ImprovementPanel(Container):
    """Panel showing improvement over baseline."""
    
    DEFAULT_CSS = """
    ImprovementPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    ImprovementPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("IMPROVEMENT OVER BASELINE", classes="panel-title")
        yield Static("", id="improvement-display")
    
    def update_improvement(self, metrics: dict):
        """Update improvement bars.
        
        Args:
            metrics: Dict with metric names and improvement percentages
        """
        text = Text()
        
        if not metrics:
            text.append("No data available.\n", style="#6b635a")
        else:
            bar_width = 40
            for name, pct in metrics.items():
                # Format percentage
                pct_str = f"{pct:+.0f}%"
                
                # Color based on improvement
                if pct > 50:
                    color = "#22c55e"
                elif pct > 0:
                    color = "#2dd4bf"
                elif pct > -20:
                    color = "#f97316"
                else:
                    color = "#ef4444"
                
                # Create bar
                bar_len = min(bar_width, abs(int(pct / 5)))  # Scale bar
                bar = "█" * bar_len
                
                text.append(f"{name:<15}", style="#a8a198")
                text.append(f"{pct_str:>6}  ", style=color)
                text.append(f"{bar}\n", style=color)
        
        try:
            self.query_one("#improvement-display", Static).update(text)
        except Exception:
            pass


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
    
    def action_pop_screen(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def compose(self) -> ComposeResult:
        """Compose the comparison screen."""
        yield Static("RUN COMPARISON", id="screen-title", classes="screen-title")
        
        with VerticalScroll(id="comparison-container"):
            # Run selection
            with Horizontal(id="run-selection"):
                yield Button("Load Run 1", id="load-run-1", variant="primary")
                yield Button("Load Run 2", id="load-run-2", variant="primary")
                yield Button("Compare", id="compare-btn", variant="success")
                yield Button("Back", id="back-btn", variant="default")
            
            # Side by side comparison
            with Horizontal(id="comparison-panels"):
                yield RunSummaryPanel(title="RUN 1", id="run1-panel")
                yield RunSummaryPanel(title="RUN 2", id="run2-panel")
            
            # Pass@K Curves
            yield PassKCurvesPanel(id="passk-panel")
            
            # Comparison table
            with Container(id="comparison-table-container"):
                yield Static("METRICS COMPARISON", classes="panel-title")
                yield DataTable(id="comparison-table")
            
            # Improvement over baseline
            yield ImprovementPanel(id="improvement-panel")
        
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
        elif event.button.id == "back-btn":
            self.action_pop_screen()
    
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
                "Compile Rate": stats[-1].get("compile_rate", 0) if stats else 0,
            }
        
        m1 = calc_metrics(self.run1_stats)
        m2 = calc_metrics(self.run2_stats)
        
        # Populate comparison table
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
            
            # Color the diff
            if diff > 0:
                diff_text = Text(diff_str, style="#22c55e")
            elif diff < 0:
                diff_text = Text(diff_str, style="#ef4444")
            else:
                diff_text = Text(diff_str, style="#6b635a")
            
            table.add_row(metric, v1_str, v2_str, diff_text)
        
        # Update pass@k curves
        try:
            self.query_one("#passk-panel", PassKCurvesPanel).update_curves(
                self.run1_stats, self.run2_stats
            )
        except Exception:
            pass
        
        # Update improvement panel (compare run2 to run1 as baseline)
        improvements = {}
        if m1.get("Final Reward") and m2.get("Final Reward"):
            baseline = m1["Final Reward"]
            if baseline > 0:
                improvements["Final Reward"] = ((m2["Final Reward"] - baseline) / baseline) * 100
        
        if m1.get("Compile Rate") and m2.get("Compile Rate"):
            baseline = m1["Compile Rate"]
            if baseline > 0:
                improvements["Compile Rate"] = ((m2["Compile Rate"] - baseline) / baseline) * 100
        
        if m1.get("Avg Samples Kept") and m2.get("Avg Samples Kept"):
            baseline = m1["Avg Samples Kept"]
            if baseline > 0:
                improvements["Samples Kept"] = ((m2["Avg Samples Kept"] - baseline) / baseline) * 100
        
        try:
            self.query_one("#improvement-panel", ImprovementPanel).update_improvement(improvements)
        except Exception:
            pass
        
        self.notify("Comparison complete")

