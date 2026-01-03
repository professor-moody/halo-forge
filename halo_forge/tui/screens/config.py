"""
Configuration Screen - Training configuration and launch.

Allows viewing/editing training parameters before starting a run.
"""

from pathlib import Path
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import (
    Footer, Static, Input, Button, Select, Label,
    RadioSet, RadioButton
)
from textual.containers import Container, Horizontal, Vertical, Grid, VerticalScroll
from textual.binding import Binding
from rich.text import Text


class ConfigPanel(Container):
    """Panel for configuration options."""
    
    DEFAULT_CSS = """
    ConfigPanel {
        background: #12100e;
        border: solid #2a2520;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    ConfigPanel > .panel-title {
        color: #6b635a;
        text-style: bold;
        padding-bottom: 1;
    }
    
    ConfigPanel > .config-row {
        height: 3;
        margin-bottom: 1;
    }
    
    ConfigPanel > .config-row > Label {
        width: 20;
        padding-top: 1;
        color: #a8a198;
    }
    
    ConfigPanel > .config-row > Input {
        width: 1fr;
    }
    """


class ConfigScreen(Screen):
    """Training configuration screen."""
    
    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("enter", "start_training", "Start", show=True),
    ]
    
    def action_pop_screen(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()
    
    def compose(self) -> ComposeResult:
        """Compose the config screen."""
        yield Static("CONFIGURATION", id="screen-title", classes="screen-title")
        
        with VerticalScroll(id="config-container"):
            # Model Configuration
            with ConfigPanel(id="model-config"):
                yield Static("MODEL", classes="panel-title")
                
                with Horizontal(classes="config-row"):
                    yield Label("Base Model:")
                    yield Input(
                        value="Qwen/Qwen2.5-Coder-7B",
                        id="model-name",
                        placeholder="HuggingFace model ID"
                    )
                
                with Horizontal(classes="config-row"):
                    yield Label("SFT Checkpoint:")
                    yield Input(
                        value="models/sft/final_model",
                        id="sft-checkpoint",
                        placeholder="Path to SFT checkpoint"
                    )
                
                with Horizontal(classes="config-row"):
                    yield Label("Output Dir:")
                    yield Input(
                        value="models/raft",
                        id="output-dir",
                        placeholder="Output directory"
                    )
            
            # Training Configuration
            with ConfigPanel(id="training-config"):
                yield Static("TRAINING", classes="panel-title")
                
                with Horizontal(classes="config-row"):
                    yield Label("Cycles:")
                    yield Input(value="5", id="num-cycles", placeholder="Number of RAFT cycles")
                
                with Horizontal(classes="config-row"):
                    yield Label("Samples/Prompt:")
                    yield Input(value="8", id="samples-per-prompt")
                
                with Horizontal(classes="config-row"):
                    yield Label("Reward Threshold:")
                    yield Input(value="0.5", id="reward-threshold")
                
                with Horizontal(classes="config-row"):
                    yield Label("Keep Top %:")
                    yield Input(value="0.5", id="keep-top-percent")
            
            # Generation Configuration  
            with ConfigPanel(id="generation-config"):
                yield Static("GENERATION", classes="panel-title")
                
                with Horizontal(classes="config-row"):
                    yield Label("Max New Tokens:")
                    yield Input(value="1024", id="max-new-tokens")
                
                with Horizontal(classes="config-row"):
                    yield Label("Temperature:")
                    yield Input(value="0.7", id="temperature")
                
                with Horizontal(classes="config-row"):
                    yield Label("Batch Size:")
                    yield Input(value="8", id="batch-size")
            
            # Verifier Selection
            with ConfigPanel(id="verifier-config"):
                yield Static("VERIFIER", classes="panel-title")
                
                with Horizontal(classes="config-row"):
                    yield Label("Verifier:")
                    yield Select(
                        [
                            ("MBPP (Python)", "mbpp"),
                            ("HumanEval (Python)", "humaneval"),
                            ("GCC (C/C++)", "gcc"),
                            ("Pytest", "pytest"),
                        ],
                        id="verifier-select",
                        value="mbpp"
                    )
                
                with Horizontal(classes="config-row"):
                    yield Label("Prompts File:")
                    yield Input(
                        value="data/rlvr/mbpp_train_prompts.jsonl",
                        id="prompts-file",
                        placeholder="Path to prompts JSONL"
                    )
            
            # Action buttons
            with Horizontal(id="action-buttons"):
                yield Button("Start Training", id="start-btn", variant="success")
                yield Button("Load Config", id="load-btn", variant="default")
                yield Button("Save Config", id="save-btn", variant="default")
                yield Button("Back", id="back-btn", variant="default")
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "start-btn":
            self.action_start_training()
        elif event.button.id == "back-btn":
            self.action_pop_screen()
        elif event.button.id == "load-btn":
            self.notify("Load config: Coming soon", severity="information")
        elif event.button.id == "save-btn":
            self.notify("Save config: Coming soon", severity="information")
    
    def action_start_training(self):
        """Start training with current config."""
        # Collect config values
        config = {
            "model": self.query_one("#model-name", Input).value,
            "sft_checkpoint": self.query_one("#sft-checkpoint", Input).value,
            "output_dir": self.query_one("#output-dir", Input).value,
            "num_cycles": int(self.query_one("#num-cycles", Input).value),
            "samples_per_prompt": int(self.query_one("#samples-per-prompt", Input).value),
            "reward_threshold": float(self.query_one("#reward-threshold", Input).value),
            "keep_top_percent": float(self.query_one("#keep-top-percent", Input).value),
            "max_new_tokens": int(self.query_one("#max-new-tokens", Input).value),
            "temperature": float(self.query_one("#temperature", Input).value),
            "batch_size": int(self.query_one("#batch-size", Input).value),
            "verifier": self.query_one("#verifier-select", Select).value,
            "prompts_file": self.query_one("#prompts-file", Input).value,
        }
        
        # Store config in app for later use
        self.app.pending_config = config
        self.notify(f"Config ready: {config['num_cycles']} cycles with {config['verifier']}")
        
        # Pop back to dashboard - actual training start is handled by app
        self.app.pop_screen()
        self.app.start_training_from_config()

