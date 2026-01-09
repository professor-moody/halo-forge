#!/usr/bin/env python3
"""
Generate training visualization charts from TensorBoard event files.

Extracts metrics from HuggingFace Trainer TensorBoard logs and generates
static PNG charts for SFT and RAFT training runs.

Usage:
    python scripts/plot_training.py models/code_sft/logs
    python scripts/plot_training.py models/code_sft/logs --output figures/
    python scripts/plot_training.py models/code_sft/logs models/vlm_sft/logs --compare
    
Available charts:
    - loss_curve.png: Training and eval loss over steps
    - learning_rate.png: Learning rate schedule
    - training_summary.png: Combined 2x2 grid of key metrics
    - grad_norm.png: Gradient norm (training stability indicator)

Consistent with plot_benchmarks.py styling.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
except ImportError:
    print("Error: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("Error: tensorboard required. Install with: pip install tensorboard")
    sys.exit(1)


# =============================================================================
# Style Configuration - Consistent with plot_benchmarks.py
# =============================================================================

plt.style.use('seaborn-v0_8-whitegrid')

COLORS = {
    'primary': '#7C9885',     # Sage green
    'secondary': '#A8B5A2',   # Light sage
    'accent': '#C9A959',      # Muted gold
    'baseline': '#6B8E9B',    # Steel blue
    'final': '#7C9885',       # Sage green
    'hardware': '#B85C5C',    # Muted red
    'train_loss': '#7C9885',  # Sage green
    'eval_loss': '#C9A959',   # Muted gold
    'lr': '#6B8E9B',          # Steel blue
    'grad_norm': '#B85C5C',   # Muted red
}

# For comparison plots with multiple runs
COMPARISON_COLORS = [
    '#7C9885',  # Sage green
    '#C9A959',  # Muted gold
    '#6B8E9B',  # Steel blue
    '#B85C5C',  # Muted red
    '#8B7355',  # Muted brown
    '#9370DB',  # Medium purple
]


# =============================================================================
# Data Loading
# =============================================================================

@dataclass
class TrainingMetrics:
    """Container for extracted training metrics."""
    name: str
    log_dir: Path
    
    # Core metrics (steps, values)
    train_loss: List[Tuple[int, float]] = None
    eval_loss: List[Tuple[int, float]] = None
    learning_rate: List[Tuple[int, float]] = None
    grad_norm: List[Tuple[int, float]] = None
    epoch: List[Tuple[int, float]] = None
    
    # Summary stats
    final_loss: float = None
    min_loss: float = None
    total_steps: int = 0
    total_epochs: float = 0
    
    def __post_init__(self):
        self.train_loss = self.train_loss or []
        self.eval_loss = self.eval_loss or []
        self.learning_rate = self.learning_rate or []
        self.grad_norm = self.grad_norm or []
        self.epoch = self.epoch or []


def find_tfevents(log_dir: Path) -> List[Path]:
    """Find all TensorBoard event files in a directory."""
    events = list(log_dir.glob("**/events.out.tfevents.*"))
    return sorted(events, key=lambda p: p.stat().st_mtime)


def load_training_metrics(log_dir: Path, name: Optional[str] = None) -> TrainingMetrics:
    """
    Load training metrics from TensorBoard event files.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        name: Optional name for this run (defaults to directory name)
    
    Returns:
        TrainingMetrics with extracted data
    """
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    # Find event files
    event_files = find_tfevents(log_dir)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in: {log_dir}")
    
    # Use the most recent event file (or merge multiple if needed)
    # EventAccumulator handles directory-level loading
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()
    
    available_tags = ea.Tags().get('scalars', [])
    
    # Initialize metrics container
    run_name = name or log_dir.parent.name  # e.g., "code_sft" from "models/code_sft/logs"
    metrics = TrainingMetrics(name=run_name, log_dir=log_dir)
    
    # Extract available metrics
    tag_mapping = {
        'train/loss': 'train_loss',
        'eval/loss': 'eval_loss', 
        'train/learning_rate': 'learning_rate',
        'train/grad_norm': 'grad_norm',
        'train/epoch': 'epoch',
        # Alternative names from different trainer versions
        'loss': 'train_loss',
        'learning_rate': 'learning_rate',
    }
    
    for tag, attr in tag_mapping.items():
        if tag in available_tags:
            scalars = ea.Scalars(tag)
            data = [(s.step, s.value) for s in scalars]
            
            # Only set if not already set (prefer train/loss over loss)
            if not getattr(metrics, attr):
                setattr(metrics, attr, data)
    
    # Calculate summary stats
    if metrics.train_loss:
        metrics.final_loss = metrics.train_loss[-1][1]
        metrics.min_loss = min(v for _, v in metrics.train_loss)
        metrics.total_steps = metrics.train_loss[-1][0]
    
    if metrics.epoch:
        metrics.total_epochs = metrics.epoch[-1][1]
    
    return metrics


def load_multiple_runs(log_dirs: List[Path]) -> List[TrainingMetrics]:
    """Load metrics from multiple training runs for comparison."""
    runs = []
    for log_dir in log_dirs:
        try:
            metrics = load_training_metrics(log_dir)
            runs.append(metrics)
            print(f"Loaded: {metrics.name} ({metrics.total_steps} steps)")
        except Exception as e:
            print(f"Warning: Could not load {log_dir}: {e}")
    return runs


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_loss_curve(
    metrics: TrainingMetrics,
    output_path: Path,
    title: Optional[str] = None
):
    """
    Plot training and evaluation loss curves.
    
    Creates a clean line plot showing loss progression over training steps.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if metrics.train_loss:
        steps, values = zip(*metrics.train_loss)
        ax.plot(steps, values, 
                color=COLORS['train_loss'], 
                linewidth=2, 
                label='Training Loss',
                alpha=0.9)
    
    if metrics.eval_loss:
        steps, values = zip(*metrics.eval_loss)
        ax.plot(steps, values,
                color=COLORS['eval_loss'],
                linewidth=2,
                linestyle='--',
                label='Eval Loss',
                alpha=0.9)
    
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title or f'{metrics.name} - Training Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add annotations for min/final loss
    if metrics.train_loss:
        min_step, min_val = min(metrics.train_loss, key=lambda x: x[1])
        ax.annotate(f'Min: {min_val:.4f}',
                    xy=(min_step, min_val),
                    xytext=(10, 20), textcoords='offset points',
                    fontsize=10, color=COLORS['train_loss'],
                    arrowprops=dict(arrowstyle='->', color=COLORS['train_loss'], alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_learning_rate(
    metrics: TrainingMetrics,
    output_path: Path,
    title: Optional[str] = None
):
    """Plot learning rate schedule over training."""
    if not metrics.learning_rate:
        print(f"Skipping learning rate plot: no data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps, values = zip(*metrics.learning_rate)
    ax.plot(steps, values,
            color=COLORS['lr'],
            linewidth=2,
            alpha=0.9)
    
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(title or f'{metrics.name} - Learning Rate Schedule', fontsize=14, fontweight='bold')
    
    # Scientific notation for y-axis
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))
    
    # Add start/end annotations
    if len(steps) > 1:
        ax.annotate(f'Start: {values[0]:.2e}',
                    xy=(steps[0], values[0]),
                    xytext=(10, -20), textcoords='offset points',
                    fontsize=10, color=COLORS['lr'])
        ax.annotate(f'End: {values[-1]:.2e}',
                    xy=(steps[-1], values[-1]),
                    xytext=(-80, 10), textcoords='offset points',
                    fontsize=10, color=COLORS['lr'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_grad_norm(
    metrics: TrainingMetrics,
    output_path: Path,
    title: Optional[str] = None
):
    """Plot gradient norm over training (stability indicator)."""
    if not metrics.grad_norm:
        print(f"Skipping grad norm plot: no data available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps, values = zip(*metrics.grad_norm)
    ax.plot(steps, values,
            color=COLORS['grad_norm'],
            linewidth=1.5,
            alpha=0.7)
    
    # Add smoothed trend line
    if len(values) > 20:
        window = max(len(values) // 20, 5)
        smoothed = _moving_average(values, window)
        smooth_steps = steps[window-1:]
        ax.plot(smooth_steps, smoothed,
                color=COLORS['grad_norm'],
                linewidth=2.5,
                alpha=0.9,
                label=f'Smoothed (window={window})')
        ax.legend(loc='upper right')
    
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title(title or f'{metrics.name} - Gradient Norm', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_summary(
    metrics: TrainingMetrics,
    output_path: Path,
    title: Optional[str] = None
):
    """
    Generate a 2x2 summary grid of key training metrics.
    
    Includes: Loss, Learning Rate, Grad Norm, and Summary Stats
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title or f'{metrics.name} - Training Summary', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Top-left: Loss curve
    ax = axes[0, 0]
    if metrics.train_loss:
        steps, values = zip(*metrics.train_loss)
        ax.plot(steps, values, color=COLORS['train_loss'], linewidth=2, label='Train')
    if metrics.eval_loss:
        steps, values = zip(*metrics.eval_loss)
        ax.plot(steps, values, color=COLORS['eval_loss'], linewidth=2, linestyle='--', label='Eval')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend(loc='upper right')
    
    # Top-right: Learning rate
    ax = axes[0, 1]
    if metrics.learning_rate:
        steps, values = zip(*metrics.learning_rate)
        ax.plot(steps, values, color=COLORS['lr'], linewidth=2)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))
    ax.set_xlabel('Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    
    # Bottom-left: Gradient norm
    ax = axes[1, 0]
    if metrics.grad_norm:
        steps, values = zip(*metrics.grad_norm)
        ax.plot(steps, values, color=COLORS['grad_norm'], linewidth=1, alpha=0.5)
        if len(values) > 20:
            window = max(len(values) // 20, 5)
            smoothed = _moving_average(values, window)
            ax.plot(steps[window-1:], smoothed, color=COLORS['grad_norm'], linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm (Training Stability)')
    
    # Bottom-right: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = _format_stats_text(metrics)
    ax.text(0.1, 0.9, stats_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title('Training Statistics')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison(
    runs: List[TrainingMetrics],
    output_path: Path,
    metric: str = 'train_loss',
    title: Optional[str] = None
):
    """
    Compare a metric across multiple training runs.
    
    Args:
        runs: List of TrainingMetrics from different runs
        output_path: Where to save the chart
        metric: Which metric to compare ('train_loss', 'eval_loss', 'learning_rate')
        title: Optional title override
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_labels = {
        'train_loss': 'Training Loss',
        'eval_loss': 'Eval Loss',
        'learning_rate': 'Learning Rate',
        'grad_norm': 'Gradient Norm',
    }
    
    for i, run in enumerate(runs):
        data = getattr(run, metric, [])
        if not data:
            continue
        
        steps, values = zip(*data)
        color = COMPARISON_COLORS[i % len(COMPARISON_COLORS)]
        ax.plot(steps, values,
                color=color,
                linewidth=2,
                label=run.name,
                alpha=0.85)
    
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)
    ax.set_title(title or f'Comparison: {metric_labels.get(metric, metric)}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    if metric == 'learning_rate':
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Utilities
# =============================================================================

def _moving_average(values: List[float], window: int) -> List[float]:
    """Calculate moving average with given window size."""
    result = []
    for i in range(window - 1, len(values)):
        avg = sum(values[i - window + 1:i + 1]) / window
        result.append(avg)
    return result


def _format_stats_text(metrics: TrainingMetrics) -> str:
    """Format summary statistics as text block."""
    lines = [
        f"Run: {metrics.name}",
        f"",
        f"Total Steps:    {metrics.total_steps:,}",
        f"Total Epochs:   {metrics.total_epochs:.2f}" if metrics.total_epochs else "",
        f"",
        f"Final Loss:     {metrics.final_loss:.4f}" if metrics.final_loss else "",
        f"Minimum Loss:   {metrics.min_loss:.4f}" if metrics.min_loss else "",
    ]
    
    if metrics.learning_rate:
        start_lr = metrics.learning_rate[0][1]
        end_lr = metrics.learning_rate[-1][1]
        lines.extend([
            f"",
            f"Start LR:       {start_lr:.2e}",
            f"End LR:         {end_lr:.2e}",
        ])
    
    if metrics.grad_norm:
        avg_norm = sum(v for _, v in metrics.grad_norm) / len(metrics.grad_norm)
        max_norm = max(v for _, v in metrics.grad_norm)
        lines.extend([
            f"",
            f"Avg Grad Norm:  {avg_norm:.2f}",
            f"Max Grad Norm:  {max_norm:.2f}",
        ])
    
    return '\n'.join(line for line in lines if line is not None)


def generate_all_charts(
    metrics: TrainingMetrics,
    output_dir: Path
):
    """Generate all available charts for a training run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_loss_curve(metrics, output_dir / "loss_curve.png")
    plot_learning_rate(metrics, output_dir / "learning_rate.png")
    plot_grad_norm(metrics, output_dir / "grad_norm.png")
    plot_training_summary(metrics, output_dir / "training_summary.png")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate training visualization charts from TensorBoard logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all charts for a single run
    python scripts/plot_training.py models/code_sft/logs
    
    # Specify output directory
    python scripts/plot_training.py models/code_sft/logs --output figures/
    
    # Compare multiple training runs
    python scripts/plot_training.py models/code_sft/logs models/vlm_sft/logs --compare
    
    # Generate only loss curve
    python scripts/plot_training.py models/code_sft/logs --only loss
        """
    )
    
    parser.add_argument(
        "log_dirs",
        nargs='+',
        help="TensorBoard log directory (e.g., models/code_sft/logs)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for charts (default: <log_dir>/figures)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison charts for multiple runs"
    )
    parser.add_argument(
        "--only",
        choices=['loss', 'lr', 'grad', 'summary'],
        default=None,
        help="Generate only specific chart type"
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Override run name in chart titles"
    )
    
    args = parser.parse_args()
    
    log_dirs = [Path(d) for d in args.log_dirs]
    
    # Comparison mode
    if args.compare and len(log_dirs) > 1:
        runs = load_multiple_runs(log_dirs)
        if not runs:
            print("Error: No valid training runs found")
            sys.exit(1)
        
        output_dir = Path(args.output) if args.output else Path("figures/comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating comparison charts in {output_dir}...")
        plot_comparison(runs, output_dir / "loss_comparison.png", 'train_loss')
        plot_comparison(runs, output_dir / "lr_comparison.png", 'learning_rate')
        
        print(f"\nDone! Comparison charts saved to {output_dir}")
        return
    
    # Single run mode
    log_dir = log_dirs[0]
    
    try:
        metrics = load_training_metrics(log_dir, name=args.name)
    except Exception as e:
        print(f"Error loading training logs: {e}")
        sys.exit(1)
    
    print(f"Loaded {metrics.name}: {metrics.total_steps} steps, final loss {metrics.final_loss:.4f}")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        # Default to <model_dir>/figures (e.g., models/code_sft/figures)
        output_dir = log_dir.parent / "figures"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating charts in {output_dir}...")
    
    # Generate requested charts
    if args.only == 'loss':
        plot_loss_curve(metrics, output_dir / "loss_curve.png")
    elif args.only == 'lr':
        plot_learning_rate(metrics, output_dir / "learning_rate.png")
    elif args.only == 'grad':
        plot_grad_norm(metrics, output_dir / "grad_norm.png")
    elif args.only == 'summary':
        plot_training_summary(metrics, output_dir / "training_summary.png")
    else:
        generate_all_charts(metrics, output_dir)
    
    print(f"\nDone! Charts saved to {output_dir}")


if __name__ == "__main__":
    main()
