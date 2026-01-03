"""
Custom widgets for halo-forge TUI.
"""

from textual.app import ComposeResult
from textual.widgets import Static, ProgressBar, DataTable, RichLog
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from rich.text import Text
from rich.table import Table
from typing import List, Dict, Optional

from .state import TrainingState


class Panel(Container):
    """A styled panel with title."""
    
    DEFAULT_CSS = """
    Panel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    Panel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    """
    
    def __init__(self, title: str = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title
    
    def compose(self) -> ComposeResult:
        if self.title:
            yield Static(self.title.upper(), classes="panel-title")


class HeaderBar(Static):
    """Header bar with logo, version, and status."""
    
    DEFAULT_CSS = """
    HeaderBar {
        dock: top;
        height: 3;
        background: #12100e;
        border-bottom: solid #2a2520;
        content-align: center middle;
    }
    """
    
    status = reactive("idle")
    
    def __init__(self, version: str = "0.2.0", **kwargs):
        super().__init__(**kwargs)
        self.version = version
    
    def render(self) -> Text:
        status_colors = {
            "idle": "#6b635a",
            "running": "#2dd4bf",
            "paused": "#f97316",
            "complete": "#22c55e",
            "error": "#ef4444",
        }
        status_color = status_colors.get(self.status, "#6b635a")
        
        text = Text()
        text.append("◆ ", style="bold #2dd4bf")
        text.append("halo-forge", style="bold")
        text.append(f" v{self.version}", style="#6b635a")
        text.append("  │  ", style="#2a2520")
        text.append("● ", style=status_color)
        text.append(self.status.title(), style=status_color)
        
        return text


class ProgressPanel(Container):
    """Progress panel showing cycle and step progress."""
    
    DEFAULT_CSS = """
    ProgressPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    ProgressPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    
    ProgressPanel > .progress-row {
        height: 1;
        margin: 0 0 1 0;
    }
    
    ProgressPanel > .progress-row > .label {
        width: 8;
        color: #6b635a;
    }
    
    ProgressPanel > .progress-row > ProgressBar {
        width: 1fr;
    }
    
    ProgressPanel > .progress-row > .value {
        width: 12;
        text-align: right;
    }
    
    ProgressPanel > .phase-row {
        margin-top: 1;
    }
    
    ProgressPanel > .eta-row {
        color: #6b635a;
        margin-top: 1;
    }
    """
    
    cycle = reactive(0)
    total_cycles = reactive(5)
    step = reactive(0)
    total_steps = reactive(200)
    phase = reactive("idle")
    eta_minutes = reactive(0.0)
    
    def compose(self) -> ComposeResult:
        yield Static("PROGRESS", classes="panel-title")
        
        with Horizontal(classes="progress-row"):
            yield Static("Cycle", classes="label")
            yield ProgressBar(total=100, show_eta=False, id="cycle-bar")
            yield Static("0/5", id="cycle-value", classes="value")
        
        with Horizontal(classes="progress-row"):
            yield Static("Step", classes="label")
            yield ProgressBar(total=100, show_eta=False, id="step-bar")
            yield Static("0/200", id="step-value", classes="value")
        
        yield Static("", id="phase-display", classes="phase-row")
        yield Static("", id="eta-display", classes="eta-row")
    
    def watch_cycle(self, cycle: int):
        self._update_cycle()
    
    def watch_total_cycles(self, total: int):
        self._update_cycle()
    
    def watch_step(self, step: int):
        self._update_step()
    
    def watch_total_steps(self, total: int):
        self._update_step()
    
    def watch_phase(self, phase: str):
        self._update_phase()
    
    def watch_eta_minutes(self, eta: float):
        self._update_eta()
    
    def _update_cycle(self):
        if self.total_cycles > 0:
            pct = (self.cycle / self.total_cycles) * 100
            try:
                bar = self.query_one("#cycle-bar", ProgressBar)
                bar.progress = pct
                val = self.query_one("#cycle-value", Static)
                val.update(f"{self.cycle}/{self.total_cycles}")
            except Exception:
                pass
    
    def _update_step(self):
        if self.total_steps > 0:
            pct = (self.step / self.total_steps) * 100
            try:
                bar = self.query_one("#step-bar", ProgressBar)
                bar.progress = pct
                val = self.query_one("#step-value", Static)
                val.update(f"{self.step}/{self.total_steps}")
            except Exception:
                pass
    
    def _update_phase(self):
        phases = ["generate", "verify", "filter", "train"]
        text = Text()
        text.append("Phase: ", style="#6b635a")
        
        for i, p in enumerate(phases):
            if p == self.phase:
                text.append(f"[{p.upper()}]", style="bold #f97316")
            else:
                text.append(p, style="#6b635a")
            
            if i < len(phases) - 1:
                text.append(" > ", style="#3d352c")
        
        try:
            self.query_one("#phase-display", Static).update(text)
        except Exception:
            pass
    
    def _update_eta(self):
        try:
            if self.eta_minutes > 60:
                eta_str = f"~{self.eta_minutes / 60:.1f}h remaining"
            else:
                eta_str = f"~{self.eta_minutes:.0f}min remaining"
            self.query_one("#eta-display", Static).update(f"ETA: {eta_str}")
        except Exception:
            pass
    
    def update_from_state(self, state: TrainingState):
        """Update all values from state."""
        self.cycle = state.cycle
        self.total_cycles = state.total_cycles
        self.step = state.step
        self.total_steps = state.total_steps
        self.phase = state.phase
        self.eta_minutes = state.eta_minutes


class MetricsPanel(Container):
    """Metrics panel showing compile rate, samples, loss."""
    
    DEFAULT_CSS = """
    MetricsPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    MetricsPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    
    MetricsPanel > .compile-rate {
        height: 4;
    }
    
    MetricsPanel > .metrics-grid {
        height: auto;
        margin-top: 1;
    }
    
    MetricsPanel .metric-label {
        color: #6b635a;
    }
    
    MetricsPanel .metric-value {
        text-style: bold;
    }
    
    MetricsPanel .teal {
        color: #2dd4bf;
    }
    """
    
    compile_rate = reactive(0.0)
    samples_generated = reactive(0)
    samples_kept = reactive(0)
    loss = reactive(0.0)
    grad_norm = reactive(0.0)
    
    def compose(self) -> ComposeResult:
        yield Static("METRICS", classes="panel-title")
        
        yield Static("", id="compile-display", classes="compile-rate")
        yield Static("", id="metrics-display", classes="metrics-grid")
    
    def watch_compile_rate(self, rate: float):
        self._update_display()
    
    def watch_samples_generated(self, n: int):
        self._update_display()
    
    def watch_samples_kept(self, n: int):
        self._update_display()
    
    def watch_loss(self, loss: float):
        self._update_display()
    
    def watch_grad_norm(self, norm: float):
        self._update_display()
    
    def _update_display(self):
        # Compile rate bar
        bar_width = 20
        filled = int(self.compile_rate / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        compile_text = Text()
        compile_text.append("Compile Rate\n", style="#6b635a")
        compile_text.append(bar, style="#2dd4bf")
        compile_text.append(f"\n{self.compile_rate:.1f}%", style="bold #2dd4bf")
        
        try:
            self.query_one("#compile-display", Static).update(compile_text)
        except Exception:
            pass
        
        # Other metrics
        metrics_text = Text()
        metrics_text.append("Samples Generated  ", style="#6b635a")
        metrics_text.append(f"{self.samples_generated:,}\n", style="bold")
        metrics_text.append("Samples Kept       ", style="#6b635a")
        metrics_text.append(f"{self.samples_kept:,}\n", style="bold #2dd4bf")
        metrics_text.append("Loss               ", style="#6b635a")
        metrics_text.append(f"{self.loss:.4f}\n", style="bold")
        metrics_text.append("Grad Norm          ", style="#6b635a")
        metrics_text.append(f"{self.grad_norm:.4f}", style="bold")
        
        try:
            self.query_one("#metrics-display", Static).update(metrics_text)
        except Exception:
            pass
    
    def update_from_state(self, state: TrainingState):
        """Update from state."""
        self.compile_rate = state.compile_rate
        self.samples_generated = state.samples_generated
        self.samples_kept = state.samples_kept
        self.loss = state.loss
        self.grad_norm = state.grad_norm


class HardwarePanel(Container):
    """Hardware panel showing GPU utilization."""
    
    DEFAULT_CSS = """
    HardwarePanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    HardwarePanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    """
    
    gpu_util = reactive(0.0)
    gpu_mem = reactive(0.0)
    gpu_temp = reactive(0.0)
    
    def compose(self) -> ComposeResult:
        yield Static("HARDWARE", classes="panel-title")
        yield Static("", id="hardware-display")
    
    def watch_gpu_util(self, util: float):
        self._update_display()
    
    def watch_gpu_mem(self, mem: float):
        self._update_display()
    
    def watch_gpu_temp(self, temp: float):
        self._update_display()
    
    def _update_display(self):
        bar_width = 15
        
        def make_bar(pct: float, color: str = "#2dd4bf") -> str:
            filled = int(pct / 100 * bar_width)
            return f"[{color}]{'█' * filled}[/][#2a2520]{'░' * (bar_width - filled)}[/]"
        
        temp_color = "#f97316" if self.gpu_temp > 75 else "#2dd4bf"
        
        text = Text.from_markup(
            f"[#6b635a]GPU Util[/]  {make_bar(self.gpu_util)} [bold]{self.gpu_util:.0f}%[/]\n"
            f"[#6b635a]GPU Mem [/]  {make_bar(self.gpu_mem)} [bold]{self.gpu_mem:.0f}%[/]\n"
            f"[#6b635a]GPU Temp[/]  {make_bar(self.gpu_temp, temp_color)} [bold]{self.gpu_temp:.0f}C[/]"
        )
        
        try:
            self.query_one("#hardware-display", Static).update(text)
        except Exception:
            pass
    
    def update_from_state(self, state: TrainingState):
        """Update from state."""
        self.gpu_util = state.gpu_util
        self.gpu_mem = state.gpu_mem
        self.gpu_temp = state.gpu_temp


class HistoryPanel(Container):
    """Cycle history panel with sparkline and table."""
    
    DEFAULT_CSS = """
    HistoryPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    HistoryPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    
    HistoryPanel > DataTable {
        height: auto;
        max-height: 8;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("CYCLE HISTORY", classes="panel-title")
        yield Static("", id="sparkline")
        yield DataTable(id="history-table")
    
    def on_mount(self):
        table = self.query_one("#history-table", DataTable)
        table.add_columns("Cycle", "Rate", "Kept", "Loss")
    
    def update_from_state(self, state: TrainingState):
        """Update from state."""
        # Sparkline
        if state.cycle_history:
            rates = [h["compile_rate"] for h in state.cycle_history]
            min_rate = min(rates) if rates else 0
            max_rate = max(rates) if rates else 1
            range_rate = max_rate - min_rate if max_rate != min_rate else 1
            
            blocks = " ▁▂▃▄▅▆▇█"
            sparkline = ""
            for r in rates:
                idx = min(8, int((r - min_rate) / range_rate * 8))
                sparkline += blocks[idx]
            
            text = Text()
            text.append("Rate: ", style="#6b635a")
            text.append(sparkline, style="#2dd4bf")
            text.append(f" ({min_rate:.0f}>{max_rate:.0f}%)", style="#6b635a")
            
            try:
                self.query_one("#sparkline", Static).update(text)
            except Exception:
                pass
        
        # Table
        try:
            table = self.query_one("#history-table", DataTable)
            table.clear()
            for h in state.cycle_history[-5:]:
                table.add_row(
                    str(h["cycle"]),
                    f"{h['compile_rate']:.1f}%",
                    str(h["samples_kept"]),
                    f"{h['loss']:.3f}"
                )
        except Exception:
            pass


class SamplesPanel(Container):
    """Live samples panel."""
    
    DEFAULT_CSS = """
    SamplesPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
        min-height: 10;
    }
    
    SamplesPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("LIVE SAMPLES", classes="panel-title")
        yield Static("", id="samples-display")
    
    def update_from_state(self, state: TrainingState):
        """Update from state."""
        text = Text()
        
        for sample in state.recent_samples[-6:]:
            if sample["success"]:
                text.append("v ", style="#22c55e")
            else:
                text.append("x ", style="#ef4444")
            
            text.append(f"{sample['reward']:.2f}  ", style="#a8a198")
            text.append(f"{sample['prompt'][:45]}...\n", style="#6b635a")
        
        if not state.recent_samples:
            text.append("Waiting for samples...", style="#6b635a")
        
        try:
            self.query_one("#samples-display", Static).update(text)
        except Exception:
            pass


class LogPanel(Container):
    """Log panel."""
    
    DEFAULT_CSS = """
    LogPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
        min-height: 10;
    }
    
    LogPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("LOG", classes="panel-title")
        yield Static("", id="log-display")
    
    def update_from_state(self, state: TrainingState):
        """Update from state."""
        text = Text()
        
        level_colors = {
            "info": "#a8a198",
            "success": "#2dd4bf",
            "warning": "#f97316",
            "error": "#ef4444",
        }
        
        for log in state.logs[-8:]:
            text.append(f"{log['time']}  ", style="#6b635a")
            color = level_colors.get(log.get("level", "info"), "#a8a198")
            text.append(f"{log['message']}\n", style=color)
        
        if not state.logs:
            text.append("No logs yet...", style="#6b635a")
        
        try:
            self.query_one("#log-display", Static).update(text)
        except Exception:
            pass

