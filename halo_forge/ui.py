"""
Rich Terminal UI Components for halo-forge

Provides styled output, progress bars, panels, and tables
for a polished CLI experience.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.text import Text
from rich.style import Style
from rich import box
from typing import Optional, List, Dict, Any


# Global console instance
console = Console()

# Color palette - muted, professional tones
COLORS = {
    "primary": "#7C9885",      # Sage green
    "secondary": "#A8B5A2",    # Light sage
    "accent": "#C9A959",       # Muted gold
    "success": "#7C9885",      # Sage green
    "error": "#B85C5C",        # Muted red
    "warning": "#C9A959",      # Muted gold
    "info": "#6B8E9B",         # Steel blue
    "muted": "#6B6B6B",        # Gray
    "dim": "#4A4A4A",          # Dark gray
}


# ASCII art banner (simple, clean)
BANNER = """
[dim]╭──────────────────────────────────────────────────────────────╮[/dim]
[dim]│[/dim]                                                              [dim]│[/dim]
[dim]│[/dim]   [bold {primary}]HALO[/bold {primary}][{secondary}]-[/{secondary}][bold {primary}]FORGE[/bold {primary}]                                          [dim]│[/dim]
[dim]│[/dim]                                                              [dim]│[/dim]
[dim]│[/dim]   [dim]RLVR Training Framework[/dim]                                   [dim]│[/dim]
[dim]│[/dim]   [dim]Optimized for AMD Strix Halo[/dim]                              [dim]│[/dim]
[dim]│[/dim]                                                              [dim]│[/dim]
[dim]╰──────────────────────────────────────────────────────────────╯[/dim]
""".format(**COLORS)


def print_banner():
    """Print the halo-forge banner."""
    console.print(BANNER)


def print_header(title: str, subtitle: Optional[str] = None):
    """Print a section header."""
    console.print()
    console.print(f"[bold {COLORS['primary']}]{title}[/bold {COLORS['primary']}]")
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]")
    console.print()


def print_step(name: str, status: str = "running", detail: str = "", time_s: Optional[float] = None):
    """Print a step status line."""
    icons = {
        "running": f"[{COLORS['info']}]>[/{COLORS['info']}]",
        "success": f"[{COLORS['success']}]✓[/{COLORS['success']}]",
        "error": f"[{COLORS['error']}]✗[/{COLORS['error']}]",
        "skip": f"[{COLORS['muted']}]-[/{COLORS['muted']}]",
        "pending": f"[{COLORS['muted']}]○[/{COLORS['muted']}]",
    }
    
    icon = icons.get(status, icons["pending"])
    time_str = f"[dim]{time_s:.1f}s[/dim]" if time_s is not None else ""
    detail_str = f"[dim]{detail}[/dim]" if detail else ""
    
    console.print(f"  {icon} {name} {detail_str} {time_str}")


def print_success(message: str):
    """Print a success message."""
    console.print(f"[{COLORS['success']}]✓[/{COLORS['success']}] {message}")


def print_error(message: str):
    """Print an error message."""
    console.print(f"[{COLORS['error']}]✗[/{COLORS['error']}] {message}")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"[{COLORS['warning']}]![/{COLORS['warning']}] {message}")


def print_info(message: str):
    """Print an info message."""
    console.print(f"[{COLORS['info']}]>[/{COLORS['info']}] {message}")


def print_dim(message: str):
    """Print a dim/muted message."""
    console.print(f"[dim]{message}[/dim]")


def print_divider():
    """Print a horizontal divider."""
    console.print(f"[dim]{'─' * 60}[/dim]")


def create_panel(content: str, title: Optional[str] = None, border_style: str = "dim") -> Panel:
    """Create a styled panel."""
    return Panel(
        content,
        title=f"[bold {COLORS['primary']}]{title}[/bold {COLORS['primary']}]" if title else None,
        border_style=border_style,
        box=box.ROUNDED,
        padding=(1, 2)
    )


def print_hardware_info(gpu_name: str, memory_gb: float, rocm_version: str = "", pytorch_version: str = ""):
    """Print hardware information panel."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Label", style="dim")
    table.add_column("Value", style=f"bold {COLORS['primary']}")
    
    table.add_row("GPU", gpu_name)
    table.add_row("Memory", f"{memory_gb:.0f} GB unified")
    if rocm_version:
        table.add_row("ROCm", rocm_version)
    if pytorch_version:
        table.add_row("PyTorch", pytorch_version)
    table.add_row("Status", f"[{COLORS['success']}]Ready[/{COLORS['success']}]")
    
    panel = Panel(
        table,
        title=f"[bold {COLORS['primary']}]Hardware[/bold {COLORS['primary']}]",
        border_style="dim",
        box=box.ROUNDED
    )
    console.print(panel)


def print_test_results(results: Dict[str, Any]):
    """Print test results in a styled format."""
    passed = results.get("passed", [])
    failed = results.get("failed", [])
    skipped = results.get("skipped", [])
    
    total = len(passed) + len(failed) + len(skipped)
    
    console.print()
    print_divider()
    
    # Summary line
    if failed:
        console.print(f"  [{COLORS['error']}]{len(passed)}/{total} passed[/{COLORS['error']}]", end="")
    else:
        console.print(f"  [{COLORS['success']}]{len(passed)}/{total} passed[/{COLORS['success']}]", end="")
    
    if skipped:
        console.print(f", [dim]{len(skipped)} skipped[/dim]", end="")
    if failed:
        console.print(f", [{COLORS['error']}]{len(failed)} failed[/{COLORS['error']}]", end="")
    console.print()
    
    # List failed tests
    if failed:
        console.print()
        console.print(f"  [{COLORS['error']}]Failed:[/{COLORS['error']}]")
        for name in failed:
            console.print(f"    [dim]-[/dim] {name}")
    
    print_divider()


def create_progress() -> Progress:
    """Create a styled progress bar with speed display."""
    return Progress(
        SpinnerColumn(style=COLORS['primary']),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40, style=COLORS['dim'], complete_style=COLORS['primary']),
        MofNCompleteColumn(),
        TextColumn("•"),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("[dim]{task.speed:>.2f} it/s[/dim]", style="dim"),
        console=console,
        transient=False
    )


def print_raft_cycle_header(cycle: int, total_cycles: int):
    """Print RAFT cycle header."""
    console.print()
    console.print(f"[bold {COLORS['primary']}]RAFT Cycle {cycle}/{total_cycles}[/bold {COLORS['primary']}]")
    console.print()


def print_raft_summary(stats: Dict[str, Any]):
    """Print RAFT cycle summary."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="dim")
    table.add_column("Value", style=f"{COLORS['primary']}")
    
    if "generated" in stats:
        table.add_row("Generated", str(stats["generated"]))
    if "verified" in stats:
        table.add_row("Verified", str(stats["verified"]))
    if "kept" in stats:
        table.add_row("Kept", str(stats["kept"]))
    if "compile_rate" in stats:
        table.add_row("Compile Rate", f"{stats['compile_rate']:.1%}")
    if "avg_reward" in stats:
        table.add_row("Avg Reward", f"{stats['avg_reward']:.2f}")
    
    console.print(table)


def print_benchmark_results(results: Dict[str, Any]):
    """Print benchmark results table."""
    table = Table(
        title=f"[bold {COLORS['primary']}]Benchmark Results[/bold {COLORS['primary']}]",
        box=box.ROUNDED,
        border_style="dim"
    )
    
    table.add_column("Metric", style="dim")
    table.add_column("Value", style=f"bold {COLORS['primary']}")
    
    if "pass_rate" in results:
        table.add_row("Pass Rate", f"{results['pass_rate']:.1%}")
    
    if "pass_at_k" in results:
        for k, rate in results["pass_at_k"].items():
            table.add_row(f"pass@{k}", f"{rate:.1%}")
    
    console.print(table)


# Convenience function for main CLI
def cli_header():
    """Print CLI header with banner."""
    print_banner()

