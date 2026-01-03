"""
State management for TUI <-> Training IPC.

The training process writes state to a JSON file.
The TUI reads this file and updates the display.
Commands (pause/stop) are written to a separate file.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime


# Default state directory
STATE_DIR = Path.home() / ".halo-forge"


@dataclass
class CycleStats:
    """Statistics for a completed cycle."""
    cycle: int = 0
    compile_rate: float = 0.0
    samples_kept: int = 0
    samples_total: int = 0
    loss: float = 0.0
    elapsed_minutes: float = 0.0


@dataclass 
class Sample:
    """A generated sample with verification result."""
    prompt: str = ""
    reward: float = 0.0
    success: bool = False
    timestamp: str = ""


@dataclass
class LogEntry:
    """A log message."""
    time: str = ""
    message: str = ""
    level: str = "info"  # info, success, warning, error


@dataclass
class TrainingState:
    """Complete training state for TUI display."""
    
    # Status
    status: str = "idle"  # idle, running, paused, complete, error
    
    # Current cycle info
    cycle: int = 0
    total_cycles: int = 5
    phase: str = "idle"  # idle, generate, verify, filter, train
    
    # Progress within phase
    step: int = 0
    total_steps: int = 0
    
    # Metrics
    compile_rate: float = 0.0
    samples_generated: int = 0
    samples_kept: int = 0
    loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    
    # Hardware
    gpu_util: float = 0.0
    gpu_mem: float = 0.0
    gpu_temp: float = 0.0
    
    # Configuration
    model_name: str = ""
    verifier: str = ""
    output_dir: str = ""
    
    # History
    cycle_history: List[Dict] = field(default_factory=list)
    recent_samples: List[Dict] = field(default_factory=list)
    logs: List[Dict] = field(default_factory=list)
    
    # Timing
    start_time: str = ""
    eta_minutes: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingState":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class StateManager:
    """
    Manages state file I/O for TUI <-> Training communication.
    
    Usage from training:
        state_mgr = StateManager()
        state_mgr.update(cycle=1, phase="generate", step=50, total_steps=100)
        state_mgr.add_log("Starting generation...")
        
    Usage from TUI:
        state_mgr = StateManager()
        state = state_mgr.read()
        if state_mgr.has_command():
            cmd = state_mgr.get_command()
    """
    
    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = Path(state_dir) if state_dir else STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.state_dir / "state.json"
        self.command_file = self.state_dir / "command.json"
        self.lock_file = self.state_dir / "state.lock"
        
        self._state = TrainingState()
    
    # -------------------------------------------------------------------------
    # Training side: Write state
    # -------------------------------------------------------------------------
    
    def update(self, **kwargs):
        """Update state fields and write to file."""
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)
        self._write_state()
    
    def add_log(self, message: str, level: str = "info"):
        """Add a log entry."""
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": message,
            "level": level
        }
        self._state.logs.append(entry)
        # Keep last 50 logs
        self._state.logs = self._state.logs[-50:]
        self._write_state()
    
    def add_sample(self, prompt: str, reward: float, success: bool):
        """Add a sample to recent samples."""
        sample = {
            "prompt": prompt[:100],  # Truncate
            "reward": reward,
            "success": success,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        self._state.recent_samples.append(sample)
        # Keep last 20 samples
        self._state.recent_samples = self._state.recent_samples[-20:]
        self._write_state()
    
    def add_cycle_stats(self, cycle: int, compile_rate: float, samples_kept: int, 
                        samples_total: int, loss: float, elapsed_minutes: float):
        """Add completed cycle stats to history."""
        stats = {
            "cycle": cycle,
            "compile_rate": compile_rate,
            "samples_kept": samples_kept,
            "samples_total": samples_total,
            "loss": loss,
            "elapsed_minutes": elapsed_minutes
        }
        self._state.cycle_history.append(stats)
        self._write_state()
    
    def set_config(self, model_name: str, verifier: str, output_dir: str, total_cycles: int):
        """Set training configuration."""
        self._state.model_name = model_name
        self._state.verifier = verifier
        self._state.output_dir = output_dir
        self._state.total_cycles = total_cycles
        self._state.start_time = datetime.now().isoformat()
        self._write_state()
    
    def _write_state(self):
        """Write state to file atomically."""
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(self._state.to_dict(), f, indent=2)
        temp_file.rename(self.state_file)
    
    # -------------------------------------------------------------------------
    # TUI side: Read state
    # -------------------------------------------------------------------------
    
    def read(self) -> TrainingState:
        """Read current state from file."""
        if not self.state_file.exists():
            return TrainingState()
        
        try:
            with open(self.state_file) as f:
                data = json.load(f)
            return TrainingState.from_dict(data)
        except (json.JSONDecodeError, IOError):
            return TrainingState()
    
    # -------------------------------------------------------------------------
    # Commands (TUI -> Training)
    # -------------------------------------------------------------------------
    
    def send_command(self, command: str, **kwargs):
        """Send a command to the training process."""
        cmd = {
            "command": command,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        with open(self.command_file, "w") as f:
            json.dump(cmd, f)
    
    def has_command(self) -> bool:
        """Check if there's a pending command."""
        return self.command_file.exists()
    
    def get_command(self) -> Optional[Dict]:
        """Get and clear the pending command."""
        if not self.command_file.exists():
            return None
        
        try:
            with open(self.command_file) as f:
                cmd = json.load(f)
            self.command_file.unlink()
            return cmd
        except (json.JSONDecodeError, IOError):
            return None
    
    def clear(self):
        """Clear all state files."""
        for f in [self.state_file, self.command_file]:
            if f.exists():
                f.unlink()


# Demo data generator for testing
def generate_demo_state(step: int = 0) -> TrainingState:
    """Generate demo state for testing the TUI."""
    import random
    
    cycle = (step // 200) % 5 + 1
    phase_idx = (step // 50) % 4
    phases = ["generate", "verify", "filter", "train"]
    
    state = TrainingState(
        status="running",
        cycle=cycle,
        total_cycles=5,
        phase=phases[phase_idx],
        step=step % 200,
        total_steps=200,
        compile_rate=15 + cycle * 8 + random.uniform(-2, 2),
        samples_generated=step * 8,
        samples_kept=int(step * 8 * 0.3),
        loss=0.85 - cycle * 0.05 + random.uniform(-0.02, 0.02),
        grad_norm=0.2 + random.uniform(-0.05, 0.05),
        learning_rate=5e-5,
        gpu_util=85 + random.uniform(-5, 10),
        gpu_mem=75 + random.uniform(-5, 5),
        gpu_temp=65 + random.uniform(-3, 3),
        model_name="Qwen/Qwen2.5-Coder-7B",
        verifier="mbpp",
        output_dir="models/production_7b",
        eta_minutes=max(0, (5 - cycle) * 60 + (200 - step % 200) * 0.5),
    )
    
    # Add cycle history
    for c in range(1, cycle):
        state.cycle_history.append({
            "cycle": c,
            "compile_rate": 15 + c * 8,
            "samples_kept": 150 + c * 30,
            "samples_total": 2992,
            "loss": 0.85 - c * 0.05,
            "elapsed_minutes": 45 + random.uniform(-5, 5)
        })
    
    # Add sample logs with error details
    prompts = [
        ("Write quicksort in C++", "", True),
        ("Implement binary search", "", True),
        ("Create a linked list class", "", True),
        ("HTTP request parser", "error: expected ';' before '}'", False),
        ("Thread pool implementation", "error: 'mutex' not found", False),
        ("Sort array using merge sort", "", True),
        ("Graph traversal BFS", "error: undeclared identifier", False),
    ]
    
    for _ in range(6):
        prompt, error, can_succeed = random.choice(prompts)
        if can_succeed:
            reward = random.choice([0.3, 0.5, 0.5, 0.7, 1.0])
            details = ""
        else:
            reward = 0.0
            details = error
        
        state.recent_samples.append({
            "prompt": prompt,
            "reward": reward,
            "success": reward >= 0.5,
            "details": details,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
    
    # Add logs
    state.logs = [
        {"time": "14:23:45", "message": f"Cycle {cycle}: Generated 200 samples", "level": "info"},
        {"time": "14:24:12", "message": "Verification complete: 79/200 passed (39.5%)", "level": "info"},
        {"time": "14:24:13", "message": "Filtered to 63 samples above threshold", "level": "info"},
        {"time": "14:24:15", "message": "Starting SFT on filtered samples...", "level": "success"},
    ]
    
    return state

