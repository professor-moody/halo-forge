"""
halo-forge Terminal User Interface

A Textual-based TUI for monitoring and controlling RAFT training runs.

Usage:
    halo-forge tui              # Launch TUI
    halo-forge tui --demo       # Run demo mode with simulated data

Features:
    - Live training progress monitoring
    - GPU metrics (utilization, memory, temperature)
    - Cycle history with sparkline visualization
    - Sample browser with filtering
    - Training run comparison
    - Export logs and reports
    - Pause/stop/resume controls
"""

from .app import HaloForgeApp, run
from .state import StateManager, TrainingState
from .gpu import GPUMonitor, get_gpu_metrics

__all__ = [
    "HaloForgeApp",
    "run",
    "StateManager",
    "TrainingState",
    "GPUMonitor",
    "get_gpu_metrics",
]
