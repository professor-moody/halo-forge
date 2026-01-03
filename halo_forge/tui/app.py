"""
halo-forge Terminal User Interface

A Textual-based TUI for monitoring and controlling RAFT training runs.
"""

from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Footer, Static
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding
from textual import work

from .state import TrainingState, StateManager, generate_demo_state
from .widgets import (
    HeaderBar,
    ProgressPanel,
    MetricsPanel,
    HardwarePanel,
    HistoryPanel,
    SamplesPanel,
    LogPanel,
)


class HaloForgeApp(App):
    """halo-forge Terminal User Interface."""
    
    CSS_PATH = "styles.tcss"
    
    TITLE = "halo-forge"
    
    BINDINGS = [
        Binding("p", "pause", "Pause", show=True),
        Binding("r", "resume", "Resume", show=True),
        Binding("s", "stop", "Stop", show=True, priority=True),
        Binding("v", "view_sample", "View Sample", show=True),
        Binding("l", "toggle_log", "Toggle Log", show=True),
        Binding("q", "quit", "Quit", show=True),
        Binding("d", "toggle_demo", "Demo Mode", show=False),
    ]
    
    def __init__(self, demo_mode: bool = False, state_dir: Path = None, **kwargs):
        super().__init__(**kwargs)
        self.demo_mode = demo_mode
        self.state_manager = StateManager(state_dir)
        self.demo_step = 0
        self._state = TrainingState()
    
    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield HeaderBar(id="header")
        
        with Container(id="main-body"):
            with Horizontal(id="top-row"):
                with Vertical(id="left-column"):
                    yield ProgressPanel(id="progress")
                    yield MetricsPanel(id="metrics")
                
                with Vertical(id="right-column"):
                    yield HistoryPanel(id="history")
                    yield HardwarePanel(id="hardware")
            
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
        
        # Update header
        header = self.query_one("#header", HeaderBar)
        header.status = self._state.status
        
        # Update panels
        self.query_one("#progress", ProgressPanel).update_from_state(self._state)
        self.query_one("#metrics", MetricsPanel).update_from_state(self._state)
        self.query_one("#hardware", HardwarePanel).update_from_state(self._state)
        self.query_one("#history", HistoryPanel).update_from_state(self._state)
        self.query_one("#samples", SamplesPanel).update_from_state(self._state)
        self.query_one("#logs", LogPanel).update_from_state(self._state)
    
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
    
    def action_view_sample(self):
        """View a sample in detail."""
        # TODO: Implement sample detail view
        self.notify("Sample viewer coming soon", severity="information")
    
    def action_toggle_log(self):
        """Toggle log panel visibility."""
        log_panel = self.query_one("#logs", LogPanel)
        log_panel.display = not log_panel.display
    
    def action_toggle_demo(self):
        """Toggle demo mode."""
        self.demo_mode = not self.demo_mode
        self.demo_step = 0
        mode = "Demo" if self.demo_mode else "Live"
        self.notify(f"Switched to {mode} mode", severity="information")


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
"""


def run(demo: bool = False):
    """Run the halo-forge TUI."""
    app = HaloForgeApp(demo_mode=demo)
    app.run()


if __name__ == "__main__":
    import sys
    demo = "--demo" in sys.argv
    run(demo=demo)

