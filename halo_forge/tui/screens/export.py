"""
Export Screen - Export training data.

Export logs, samples, and generate training reports.
"""

import json
from pathlib import Path
from datetime import datetime
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Static, Button, Input, TextArea, Checkbox
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.binding import Binding
from rich.text import Text


class ExportPanel(Container):
    """Panel for export options."""
    
    DEFAULT_CSS = """
    ExportPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    ExportPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    
    ExportPanel > .export-row {
        height: 3;
        margin-bottom: 1;
    }
    """


class ExportScreen(Screen):
    """Export screen for training data."""
    
    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
    ]
    
    def action_pop_screen(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def compose(self) -> ComposeResult:
        """Compose the export screen."""
        yield Static("EXPORT DATA", id="screen-title", classes="screen-title")
        
        with Container(id="export-container"):
            # Export options
            with ExportPanel(id="export-options"):
                yield Static("EXPORT OPTIONS", classes="panel-title")
                
                with Horizontal(classes="export-row"):
                    yield Checkbox("Training logs", id="export-logs", value=True)
                
                with Horizontal(classes="export-row"):
                    yield Checkbox("Generated samples (JSONL)", id="export-samples", value=True)
                
                with Horizontal(classes="export-row"):
                    yield Checkbox("Cycle statistics (JSON)", id="export-stats", value=True)
                
                with Horizontal(classes="export-row"):
                    yield Checkbox("Training report (Markdown)", id="export-report", value=True)
                
                with Horizontal(classes="export-row"):
                    yield Static("Output directory:", classes="label")
                    yield Input(value="exports/", id="export-dir", placeholder="Export directory")
            
            # Preview
            with ExportPanel(id="preview-panel"):
                yield Static("PREVIEW", classes="panel-title")
                yield TextArea(id="preview-text", read_only=True)
            
            # Action buttons
            with Horizontal(id="export-buttons"):
                yield Button("Preview Report", id="preview-btn", variant="default")
                yield Button("Export All", id="export-btn", variant="success")
                yield Button("Back", id="back-btn", variant="default")
        
        yield Footer()
    
    def on_mount(self):
        """Initialize preview."""
        self._update_preview()
    
    def on_button_pressed(self, event: Button.Pressed):
        """Handle button presses."""
        if event.button.id == "preview-btn":
            self._generate_preview()
        elif event.button.id == "export-btn":
            self._export_all()
        elif event.button.id == "back-btn":
            self.action_pop_screen()
    
    def _update_preview(self):
        """Update the preview text."""
        preview = self.query_one("#preview-text", TextArea)
        preview.text = "Click 'Preview Report' to generate a preview of the training report."
    
    def _generate_preview(self):
        """Generate a preview of the training report."""
        preview = self.query_one("#preview-text", TextArea)
        
        # Get current state
        state = self.app.state_manager.read()
        
        report = self._generate_report(state)
        preview.text = report
    
    def _generate_report(self, state) -> str:
        """Generate a markdown training report."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# halo-forge Training Report

Generated: {now}

## Configuration

- **Model**: {state.model_name or 'N/A'}
- **Verifier**: {state.verifier or 'N/A'}
- **Output Directory**: {state.output_dir or 'N/A'}
- **Total Cycles**: {state.total_cycles}

## Training Progress

- **Status**: {state.status}
- **Current Cycle**: {state.cycle}/{state.total_cycles}
- **Current Phase**: {state.phase}

## Metrics Summary

| Metric | Value |
|--------|-------|
| Compile Rate | {state.compile_rate:.1f}% |
| Samples Generated | {state.samples_generated} |
| Samples Kept | {state.samples_kept} |
| Current Loss | {state.loss:.4f} |

## Cycle History

| Cycle | Compile Rate | Samples Kept | Loss | Time |
|-------|-------------|--------------|------|------|
"""
        
        for cycle in state.cycle_history:
            report += f"| {cycle.get('cycle', '-')} | {cycle.get('compile_rate', 0):.1f}% | {cycle.get('samples_kept', 0)} | {cycle.get('loss', 0):.4f} | {cycle.get('elapsed_minutes', 0):.1f}m |\n"
        
        report += """
## Hardware Utilization

- **GPU Utilization**: {:.0f}%
- **GPU Memory**: {:.0f}%
- **GPU Temperature**: {:.0f}°C

## Recent Logs

```
""".format(state.gpu_util, state.gpu_mem, state.gpu_temp)
        
        for log in state.logs[-20:]:
            report += f"[{log.get('time', '')}] {log.get('message', '')}\n"
        
        report += "```\n"
        
        return report
    
    def _export_all(self):
        """Export all selected data."""
        export_dir = Path(self.query_one("#export-dir", Input).value)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        state = self.app.state_manager.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        exported = []
        
        # Export logs
        if self.query_one("#export-logs", Checkbox).value:
            logs_path = export_dir / f"training_logs_{timestamp}.json"
            with open(logs_path, 'w') as f:
                json.dump(state.logs, f, indent=2)
            exported.append("logs")
        
        # Export samples
        if self.query_one("#export-samples", Checkbox).value:
            samples_path = export_dir / f"samples_{timestamp}.jsonl"
            with open(samples_path, 'w') as f:
                for sample in state.recent_samples:
                    f.write(json.dumps(sample) + '\n')
            exported.append("samples")
        
        # Export stats
        if self.query_one("#export-stats", Checkbox).value:
            stats_path = export_dir / f"cycle_stats_{timestamp}.json"
            with open(stats_path, 'w') as f:
                json.dump(state.cycle_history, f, indent=2)
            exported.append("stats")
        
        # Export report
        if self.query_one("#export-report", Checkbox).value:
            report_path = export_dir / f"training_report_{timestamp}.md"
            report = self._generate_report(state)
            with open(report_path, 'w') as f:
                f.write(report)
            exported.append("report")
        
        self.notify(f"Exported: {', '.join(exported)} to {export_dir}")

