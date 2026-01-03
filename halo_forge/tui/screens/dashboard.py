"""
Dashboard Screen - Live training monitoring.

This is the main screen showing real-time training progress.
"""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Static
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding

from ..widgets import (
    HeaderBar,
    ProgressPanel,
    MetricsPanel,
    HardwarePanel,
    HistoryPanel,
    SamplesPanel,
    LogPanel,
    QuickActionsPanel,
    RewardDistributionPanel,
)
from ..state import TrainingState


class DashboardScreen(Screen):
    """Main dashboard for monitoring training."""
    
    BINDINGS = [
        Binding("c", "push_screen('config')", "Config", show=True),
        Binding("v", "push_screen('samples')", "Samples", show=True),
        Binding("m", "push_screen('comparison')", "Compare", show=True),
        Binding("e", "push_screen('export')", "Export", show=True),
        Binding("p", "pause", "Pause", show=True),
        Binding("s", "stop", "Stop", show=True),
        Binding("q", "quit", "Quit", show=True),
    ]
    
    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        yield HeaderBar(id="header")
        
        with Vertical(id="main-body"):
            # Row 1: Progress/Metrics + History/Hardware
            with Horizontal(id="top-row", classes="panel-row"):
                with Vertical(id="left-column"):
                    yield ProgressPanel(id="progress")
                    yield MetricsPanel(id="metrics")
                    yield RewardDistributionPanel(id="reward-dist")
                
                with Vertical(id="right-column"):
                    yield HistoryPanel(id="history")
                    yield HardwarePanel(id="hardware")
                    yield QuickActionsPanel(id="quick-actions")
            
            # Row 2: Samples + Logs
            with Horizontal(id="bottom-row", classes="panel-row"):
                yield SamplesPanel(id="samples")
                yield LogPanel(id="logs")
        
        yield Footer()
    
    def update_from_state(self, state: TrainingState):
        """Update all panels from state."""
        # Update header
        header = self.query_one("#header", HeaderBar)
        header.status = state.status
        
        # Update panels
        self.query_one("#progress", ProgressPanel).update_from_state(state)
        self.query_one("#metrics", MetricsPanel).update_from_state(state)
        self.query_one("#hardware", HardwarePanel).update_from_state(state)
        self.query_one("#history", HistoryPanel).update_from_state(state)
        self.query_one("#samples", SamplesPanel).update_from_state(state)
        self.query_one("#logs", LogPanel).update_from_state(state)
    
    def action_pause(self):
        """Pause training."""
        self.app.action_pause()
    
    def action_stop(self):
        """Stop training."""
        self.app.action_stop()

