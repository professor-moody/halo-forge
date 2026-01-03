"""
halo-forge Terminal User Interface

A Textual-based TUI for monitoring and controlling RAFT training runs.
"""

import subprocess
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Footer, Static
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding
from textual import work

from .state import TrainingState, StateManager, generate_demo_state
from .gpu import GPUMonitor
from .widgets import (
    HeaderBar,
    ProgressPanel,
    MetricsPanel,
    HardwarePanel,
    HistoryPanel,
    SamplesPanel,
    LogPanel,
    QuickActionsPanel,
    RecentRunsPanel,
    SystemInfoPanel,
    RewardDistributionPanel,
)
from .screens import (
    DashboardScreen,
    ConfigScreen,
    SamplesScreen,
    ComparisonScreen,
    ExportScreen,
)


class HaloForgeApp(App):
    """halo-forge Terminal User Interface."""
    
    CSS_PATH = "styles.tcss"
    
    TITLE = "halo-forge"
    
    # Register screens
    SCREENS = {
        "config": ConfigScreen,
        "samples": SamplesScreen,
        "comparison": ComparisonScreen,
        "export": ExportScreen,
    }
    
    BINDINGS = [
        Binding("n", "push_screen('config')", "New", show=True),
        Binding("c", "push_screen('config')", "Config", show=False),
        Binding("v", "push_screen('samples')", "Samples", show=True),
        Binding("m", "push_screen('comparison')", "Compare", show=True),
        Binding("e", "push_screen('export')", "Export", show=True),
        Binding("b", "run_benchmark", "Bench", show=True),
        Binding("p", "pause", "Pause", show=True),
        Binding("r", "resume", "Resume", show=True),
        Binding("s", "stop", "Stop", show=True, priority=True),
        Binding("l", "toggle_log", "Toggle Log", show=False),
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "toggle_demo", "Demo Mode", show=False),
    ]
    
    def __init__(self, demo_mode: bool = False, state_dir: Path = None, **kwargs):
        super().__init__(**kwargs)
        self.demo_mode = demo_mode
        self.state_manager = StateManager(state_dir)
        self.gpu_monitor = GPUMonitor(poll_interval=2.0)
        self.demo_step = 0
        self._state = TrainingState()
        self.pending_config = None  # For config screen to pass config
        self._training_process = None
    
    def compose(self) -> ComposeResult:
        """Compose the app layout (dashboard is the main screen)."""
        yield HeaderBar(id="header")
        
        with Container(id="main-body"):
            with Horizontal(id="top-row"):
                with Vertical(id="left-column"):
                    yield ProgressPanel(id="progress")
                    yield MetricsPanel(id="metrics")
                    yield RewardDistributionPanel(id="reward-dist")
                
                with Vertical(id="right-column"):
                    yield HistoryPanel(id="history")
                    yield HardwarePanel(id="hardware")
                    yield QuickActionsPanel(id="quick-actions")
            
            with Horizontal(id="bottom-row"):
                yield SamplesPanel(id="samples")
                yield LogPanel(id="logs")
        
        yield Footer()
    
    def on_mount(self):
        """Start the update loop when mounted."""
        self.update_state()
        self.set_interval(0.5, self.update_state)
    
    def update_state(self):
        """Update the display from state."""
        if self.demo_mode:
            self._state = generate_demo_state(self.demo_step)
            self.demo_step += 1
        else:
            self._state = self.state_manager.read()
            
            # Get real GPU metrics
            gpu = self.gpu_monitor.get_metrics()
            if gpu.available:
                if gpu.utilization is not None:
                    self._state.gpu_util = gpu.utilization
                if gpu.memory_percent is not None:
                    self._state.gpu_mem = gpu.memory_percent
                if gpu.temperature is not None:
                    self._state.gpu_temp = gpu.temperature
        
        # Update header
        try:
            header = self.query_one("#header", HeaderBar)
            header.status = self._state.status
            
            # Update core panels
            self.query_one("#progress", ProgressPanel).update_from_state(self._state)
            self.query_one("#metrics", MetricsPanel).update_from_state(self._state)
            self.query_one("#hardware", HardwarePanel).update_from_state(self._state)
            self.query_one("#history", HistoryPanel).update_from_state(self._state)
            self.query_one("#samples", SamplesPanel).update_from_state(self._state)
            self.query_one("#logs", LogPanel).update_from_state(self._state)
            
            # Update reward distribution (computed from recent samples)
            reward_dist = self._compute_reward_distribution()
            self.query_one("#reward-dist", RewardDistributionPanel).update_distribution(reward_dist)
        except Exception:
            # Panels might not exist if on a different screen
            pass
    
    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------
    
    def action_pause(self):
        """Pause training."""
        if not self.demo_mode:
            self.state_manager.send_command("pause")
        self.notify("Pause command sent", severity="information")
    
    def action_resume(self):
        """Resume training."""
        if not self.demo_mode:
            self.state_manager.send_command("resume")
        self.notify("Resume command sent", severity="information")
    
    def action_stop(self):
        """Stop training."""
        if not self.demo_mode:
            self.state_manager.send_command("stop")
        self.notify("Stop command sent - training will stop after current step", severity="warning")
    
    def action_toggle_log(self):
        """Toggle log panel visibility."""
        try:
            log_panel = self.query_one("#logs", LogPanel)
            log_panel.display = not log_panel.display
        except Exception:
            pass
    
    def action_toggle_demo(self):
        """Toggle demo mode."""
        self.demo_mode = not self.demo_mode
        self.demo_step = 0
        mode = "Demo" if self.demo_mode else "Live"
        self.notify(f"Switched to {mode} mode", severity="information")
    
    def action_run_benchmark(self):
        """Run a benchmark."""
        self.notify("Benchmark: Coming soon - use CLI for now", severity="information")
    
    def _compute_reward_distribution(self) -> dict:
        """Compute reward distribution from recent samples."""
        dist = {"0.0": 0, "0.5": 0, "0.7": 0, "1.0": 0}
        
        for sample in self._state.recent_samples:
            reward = sample.get("reward", 0)
            if reward >= 1.0:
                dist["1.0"] += 1
            elif reward >= 0.7:
                dist["0.7"] += 1
            elif reward >= 0.5:
                dist["0.5"] += 1
            else:
                dist["0.0"] += 1
        
        return dist
    
    # -------------------------------------------------------------------------
    # Training Control
    # -------------------------------------------------------------------------
    
    def start_training_from_config(self):
        """Start training using the pending config."""
        if not self.pending_config:
            self.notify("No config available", severity="error")
            return
        
        config = self.pending_config
        self.pending_config = None
        
        # Build command
        cmd = [
            "python3", "-m", "halo_forge.cli", "raft", "train",
            "--model", config["model"],
            "--prompts", config["prompts_file"],
            "--verifier", config["verifier"],
            "--cycles", str(config["num_cycles"]),
            "--output", config["output_dir"],
        ]
        
        # Add optional parameters
        if config.get("reward_threshold"):
            cmd.extend(["--reward-threshold", str(config["reward_threshold"])])
        if config.get("keep_top_percent"):
            cmd.extend(["--keep-percent", str(config["keep_top_percent"])])
        
        self.notify(f"Starting training: {' '.join(cmd[:6])}...")
        
        # Start training in background
        self._start_training_process(cmd)
    
    @work(exclusive=True, thread=True)
    def _start_training_process(self, cmd: list):
        """Start the training process in background."""
        try:
            # Create log file
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "tui_training.log"
            
            with open(log_file, "w") as f:
                self._training_process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=Path.cwd()
                )
            
            self.call_from_thread(
                self.notify,
                f"Training started (PID: {self._training_process.pid})",
                severity="success"
            )
            
            # Wait for process
            self._training_process.wait()
            
            self.call_from_thread(
                self.notify,
                "Training process completed",
                severity="information"
            )
        except Exception as e:
            self.call_from_thread(
                self.notify,
                f"Failed to start training: {e}",
                severity="error"
            )


# Standalone CSS for when the file can't be loaded
FALLBACK_CSS = """
Screen {
    background: #0a0908;
}

#header {
    dock: top;
    height: 3;
    background: #12100e;
    border-bottom: solid #2a2520;
    content-align: center middle;
}

#main-body {
    height: 1fr;
}

#top-row {
    height: 1fr;
}

#left-column, #right-column {
    width: 50%;
}

#bottom-row {
    height: auto;
    min-height: 12;
}

#bottom-row > * {
    width: 50%;
}

Footer {
    dock: bottom;
    background: #12100e;
}

/* Screen titles */
.screen-title {
    dock: top;
    height: 3;
    background: #12100e;
    border-bottom: solid #2a2520;
    content-align: center middle;
    text-style: bold;
    color: #2dd4bf;
}

/* Config screen */
#config-container {
    padding: 1;
}

#action-buttons {
    height: 5;
    margin-top: 1;
}

#action-buttons Button {
    margin-right: 1;
}

/* Samples screen */
#samples-container {
    padding: 1;
}

#filter-bar {
    height: 5;
    margin-bottom: 1;
}

#filter-bar Input {
    width: 1fr;
}

#filter-bar Button {
    margin-left: 1;
}

#samples-content {
    height: 1fr;
}

#samples-list {
    width: 50%;
}

#detail-panel {
    width: 50%;
}

/* Comparison screen */
#comparison-container {
    padding: 1;
}

#run-selection {
    height: 5;
    margin-bottom: 1;
}

#run-selection Button {
    margin-right: 1;
}

#comparison-panels {
    height: 1fr;
}

#comparison-table-container {
    height: auto;
    min-height: 10;
}

/* Export screen */
#export-container {
    padding: 1;
}

#preview-panel {
    height: 1fr;
}

#export-buttons {
    height: 5;
    margin-top: 1;
}

#export-buttons Button {
    margin-right: 1;
}
"""


def run(demo: bool = False):
    """Run the halo-forge TUI."""
    app = HaloForgeApp(demo_mode=demo)
    app.run()


if __name__ == "__main__":
    import sys
    demo = "--demo" in sys.argv
    run(demo=demo)
