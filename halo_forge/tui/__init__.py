"""
halo-forge Terminal User Interface

A Textual-based TUI for monitoring and controlling RAFT training runs.

Usage:
    halo-forge tui              # Launch TUI
    halo-forge tui --demo       # Run demo mode with simulated data
"""

from .app import HaloForgeApp

__all__ = ["HaloForgeApp"]

