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
        min-height: 10;
    }
    
    ProgressPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    
    ProgressPanel > .progress-row {
        height: 2;
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
    
    def on_mount(self):
        """Initialize display on mount."""
        self._update_cycle()
        self._update_step()
        self._update_phase()
        self._update_eta()
    
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
        min-height: 12;
    }
    
    MetricsPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    
    MetricsPanel > #compile-display {
        height: 4;
    }
    
    MetricsPanel > #metrics-display {
        height: auto;
        min-height: 5;
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
    
    def on_mount(self):
        """Initialize display on mount."""
        self._update_display()
    
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
        min-height: 8;
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
    
    def on_mount(self):
        """Initialize display on mount."""
        self._update_display()
    
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
        """Update from state with enhanced sparklines and trend."""
        # Sparkline with trend indicator
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
            
            # Trend indicator
            if len(rates) >= 2:
                trend = rates[-1] - rates[-2]
                if trend > 2:
                    trend_icon = "↑"
                    trend_color = "#22c55e"
                elif trend < -2:
                    trend_icon = "↓"
                    trend_color = "#ef4444"
                else:
                    trend_icon = "→"
                    trend_color = "#f97316"
            else:
                trend_icon = ""
                trend_color = "#6b635a"
            
            text = Text()
            text.append("Rate  ", style="#6b635a")
            text.append(sparkline, style="#2dd4bf")
            text.append(f" ({min_rate:.0f}→{max_rate:.0f}%)", style="#a8a198")
            if trend_icon:
                text.append(f" {trend_icon}", style=trend_color)
            
            try:
                self.query_one("#sparkline", Static).update(text)
            except Exception:
                pass
        else:
            try:
                self.query_one("#sparkline", Static).update(
                    Text("No history yet...", style="#6b635a")
                )
            except Exception:
                pass
        
        # Table with color-coded rates
        try:
            table = self.query_one("#history-table", DataTable)
            table.clear()
            for h in state.cycle_history[-5:]:
                rate = h.get('compile_rate', 0)
                # Color code the rate
                if rate >= 40:
                    rate_str = f"[#22c55e]{rate:.1f}%[/]"
                elif rate >= 25:
                    rate_str = f"[#2dd4bf]{rate:.1f}%[/]"
                elif rate >= 15:
                    rate_str = f"[#f97316]{rate:.1f}%[/]"
                else:
                    rate_str = f"[#ef4444]{rate:.1f}%[/]"
                
                table.add_row(
                    str(h.get("cycle", "?")),
                    Text.from_markup(rate_str),
                    str(h.get("samples_kept", 0)),
                    f"{h.get('loss', 0):.3f}"
                )
        except Exception:
            pass


class SamplesPanel(Container):
    """Live samples panel with enhanced details."""
    
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
    
    def on_mount(self):
        """Show initial message."""
        text = Text("Waiting for samples...", style="#6b635a")
        try:
            self.query_one("#samples-display", Static).update(text)
        except Exception:
            pass
    
    def update_from_state(self, state: TrainingState):
        """Update from state with enhanced sample details."""
        text = Text()
        
        for sample in state.recent_samples[-6:]:
            # Status icon
            if sample.get("success"):
                text.append("✓ ", style="#22c55e")
            else:
                text.append("✗ ", style="#ef4444")
            
            # Reward
            reward = sample.get('reward', 0)
            reward_color = "#22c55e" if reward >= 0.7 else "#f97316" if reward >= 0.3 else "#ef4444"
            text.append(f"{reward:.2f}  ", style=reward_color)
            
            # Prompt (truncated)
            prompt = sample.get('prompt', '')[:40]
            text.append(f'"{prompt}"', style="#a8a198")
            
            # Error details if failed
            details = sample.get('details', '')
            if not sample.get("success") and details:
                # Extract just the error type
                if "error:" in details.lower():
                    error_msg = details.split("error:")[-1].strip()[:30]
                    text.append(f" → {error_msg}", style="#ef4444")
                elif details:
                    text.append(f" → {details[:30]}", style="#ef4444")
            
            text.append("\n")
        
        if not state.recent_samples:
            text.append("Waiting for samples...", style="#6b635a")
        
        try:
            self.query_one("#samples-display", Static).update(text)
        except Exception:
            pass


class RewardDistributionPanel(Container):
    """Panel showing reward distribution histogram."""
    
    DEFAULT_CSS = """
    RewardDistributionPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
        min-height: 10;
    }
    
    RewardDistributionPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("REWARD DISTRIBUTION", classes="panel-title")
        yield Static("", id="reward-dist-display")
    
    def on_mount(self):
        """Show initial message."""
        text = Text("No data yet...", style="#6b635a")
        try:
            self.query_one("#reward-dist-display", Static).update(text)
        except Exception:
            pass
    
    def update_distribution(self, distribution: Dict):
        """Update the distribution histogram.
        
        Args:
            distribution: Dict with keys '0.0', '0.5', '0.7', '1.0' and counts
        """
        text = Text()
        
        if not distribution:
            text.append("No data yet...", style="#6b635a")
        else:
            # Find max for scaling
            max_count = max(distribution.values()) if distribution.values() else 1
            bar_width = 20
            
            # Histogram bars
            labels = [
                ("1.0", "#22c55e", "correct"),
                ("0.7", "#2dd4bf", "runs"),
                ("0.5", "#f97316", "compiled"),
                ("0.0", "#ef4444", "failed"),
            ]
            
            for key, color, desc in labels:
                count = distribution.get(key, 0)
                pct = (count / max_count * bar_width) if max_count > 0 else 0
                bar = "█" * int(pct)
                
                text.append(f"{key} ", style="#6b635a")
                text.append("▏", style="#3d352c")
                text.append(f"{bar:<{bar_width}}", style=color)
                text.append(f" {count:>4}", style="#a8a198")
                text.append(f" {desc}\n", style="#6b635a")
        
        try:
            self.query_one("#reward-dist-display", Static).update(text)
        except Exception:
            pass


class QuickActionsPanel(Container):
    """Panel with quick action buttons."""
    
    DEFAULT_CSS = """
    QuickActionsPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    QuickActionsPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    
    QuickActionsPanel > .action-item {
        height: 2;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("QUICK ACTIONS", classes="panel-title")
        yield Static("", id="actions-display")
    
    def on_mount(self):
        """Set up quick actions display."""
        text = Text()
        text.append("  [N]", style="bold #f97316")
        text.append(" New Training Run\n", style="#e8e4df")
        text.append("  [R]", style="bold #f97316")
        text.append(" Resume Last Run\n", style="#e8e4df")
        text.append("  [B]", style="bold #f97316")
        text.append(" Run Benchmark\n", style="#e8e4df")
        text.append("  [C]", style="bold #f97316")
        text.append(" Configuration\n", style="#e8e4df")
        text.append("  [V]", style="bold #f97316")
        text.append(" View Samples\n", style="#e8e4df")
        
        try:
            self.query_one("#actions-display", Static).update(text)
        except Exception:
            pass


class RecentRunsPanel(Container):
    """Panel showing recent training runs."""
    
    DEFAULT_CSS = """
    RecentRunsPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
        min-height: 10;
    }
    
    RecentRunsPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("RECENT RUNS", classes="panel-title")
        yield Static("", id="runs-display")
    
    def update_runs(self, runs: List[Dict]):
        """Update recent runs display.
        
        Args:
            runs: List of run dictionaries with name, status, cycles, compile_rate, time
        """
        text = Text()
        
        if not runs:
            text.append("No recent runs.\n", style="#6b635a")
            text.append("Press [N] to start a new run.", style="#a8a198")
        else:
            for run in runs[-5:]:
                # Status indicator
                status = run.get("status", "unknown")
                if status == "complete":
                    text.append("● ", style="#22c55e")
                elif status == "running":
                    text.append("◐ ", style="#2dd4bf")
                elif status == "paused":
                    text.append("◑ ", style="#f97316")
                else:
                    text.append("○ ", style="#6b635a")
                
                # Run name
                name = run.get("name", "unnamed")[:20]
                text.append(f"{name:<20}", style="#e8e4df")
                
                # Time ago
                time_ago = run.get("time_ago", "")
                text.append(f" {time_ago:>8}\n", style="#6b635a")
                
                # Details
                cycles = run.get("cycles", "?/?")
                rate = run.get("compile_rate", 0)
                text.append(f"   Cycle {cycles}", style="#a8a198")
                if status == "complete":
                    text.append(" ✓", style="#22c55e")
                text.append(f"  {rate:.1f}% compile\n", style="#2dd4bf")
        
        try:
            self.query_one("#runs-display", Static).update(text)
        except Exception:
            pass


class SystemInfoPanel(Container):
    """Panel showing system information."""
    
    DEFAULT_CSS = """
    SystemInfoPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    SystemInfoPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static("SYSTEM", classes="panel-title")
        yield Static("", id="system-display")
    
    def update_system_info(self, gpu_name: str = "", vram_used: float = 0, 
                           vram_total: float = 0, temp: float = 0, util: float = 0,
                           rocm_version: str = "", pytorch_version: str = ""):
        """Update system info display."""
        text = Text()
        
        # GPU
        text.append("GPU   ", style="#6b635a")
        text.append(f"{gpu_name or 'Unknown'}\n", style="#e8e4df")
        
        # VRAM
        text.append("VRAM  ", style="#6b635a")
        text.append(f"{vram_used:.1f} / {vram_total:.1f} GB\n", style="#e8e4df")
        
        # Temperature with sparkline
        text.append("Temp  ", style="#6b635a")
        temp_color = "#ef4444" if temp > 80 else "#f97316" if temp > 70 else "#2dd4bf"
        text.append(f"{temp:.0f}°C", style=temp_color)
        text.append("  ▁▂▃▄▃▂▁\n", style="#2dd4bf")
        
        # Utilization bar
        text.append("Util  ", style="#6b635a")
        bar_width = 10
        filled = int(util / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        text.append(f"{util:.0f}%  ", style="#e8e4df")
        text.append(bar, style="#2dd4bf")
        text.append("\n\n", style="")
        
        # Versions
        text.append("ROCm  ", style="#6b635a")
        text.append(f"{rocm_version or 'N/A'}\n", style="#a8a198")
        text.append("PyTorch ", style="#6b635a")
        text.append(f"{pytorch_version or 'N/A'}\n", style="#a8a198")
        
        try:
            self.query_one("#system-display", Static).update(text)
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
    
    def on_mount(self):
        """Show initial message."""
        text = Text("No logs yet...", style="#6b635a")
        try:
            self.query_one("#log-display", Static).update(text)
        except Exception:
            pass
    
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

