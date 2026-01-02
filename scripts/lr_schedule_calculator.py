#!/usr/bin/env python3
"""
Learning Rate Schedule Calculator for RLVR Training

⚠️  EXPERIMENTAL & THEORETICAL

This script helps compute and visualize learning rate schedules
for RAFT training. All strategies are theoretical and untested.

Usage:
    python lr_schedule_calculator.py --cycles 5 --base-lr 5e-5 --decay 0.85
    python lr_schedule_calculator.py --compare
    python lr_schedule_calculator.py --visualize --output lr_curves.png
"""

import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class LRSchedule:
    """Learning rate schedule configuration."""
    name: str
    base_lr: float
    decay_factor: float
    cycles: int
    schedule: List[float]
    
    def to_yaml_block(self) -> str:
        """Generate YAML config block."""
        lines = [
            f"  # {self.name}",
            f"  base_learning_rate: {self.base_lr:.0e}",
            f"  lr_decay_factor: {self.decay_factor}",
            f"  learning_rate_schedule:"
        ]
        for i, lr in enumerate(self.schedule, 1):
            lines.append(f"    cycle_{i}: {lr:.2e}")
        return "\n".join(lines)


def compute_exponential_decay(
    base_lr: float,
    decay_factor: float,
    cycles: int
) -> List[float]:
    """
    Compute exponential decay schedule.
    
    Formula: lr(cycle) = base_lr * (decay_factor ^ (cycle - 1))
    
    Args:
        base_lr: Starting learning rate (cycle 1)
        decay_factor: Multiplicative decay per cycle (e.g., 0.85)
        cycles: Number of RAFT cycles
        
    Returns:
        List of learning rates per cycle
    """
    return [base_lr * (decay_factor ** (i)) for i in range(cycles)]


def compute_step_decay(
    base_lr: float,
    decay_factor: float,
    step_every: int,
    cycles: int
) -> List[float]:
    """
    Compute step decay schedule (decay every N cycles).
    
    Args:
        base_lr: Starting learning rate
        decay_factor: Decay multiplier at each step
        step_every: Decay every N cycles
        cycles: Total cycles
        
    Returns:
        List of learning rates per cycle
    """
    schedule = []
    current_lr = base_lr
    for i in range(cycles):
        schedule.append(current_lr)
        if (i + 1) % step_every == 0:
            current_lr *= decay_factor
    return schedule


def compute_warmup_stable_decay(
    base_lr: float,
    warmup_cycles: int,
    stable_cycles: int,
    decay_cycles: int,
    final_lr_ratio: float = 0.2
) -> List[float]:
    """
    Compute warmup-stable-decay schedule across cycles.
    
    Args:
        base_lr: Peak learning rate (after warmup)
        warmup_cycles: Cycles for warmup phase
        stable_cycles: Cycles at stable LR
        decay_cycles: Cycles for decay phase
        final_lr_ratio: Final LR as ratio of base_lr
        
    Returns:
        List of learning rates per cycle
    """
    schedule = []
    total = warmup_cycles + stable_cycles + decay_cycles
    
    # Warmup phase
    for i in range(warmup_cycles):
        ratio = (i + 1) / warmup_cycles
        schedule.append(base_lr * ratio * 0.5 + base_lr * 0.5)  # Start at 50%
    
    # Stable phase
    for _ in range(stable_cycles):
        schedule.append(base_lr)
    
    # Decay phase
    final_lr = base_lr * final_lr_ratio
    for i in range(decay_cycles):
        ratio = (i + 1) / decay_cycles
        schedule.append(base_lr - (base_lr - final_lr) * ratio)
    
    return schedule


def format_schedule_table(schedules: Dict[str, List[float]]) -> str:
    """Format multiple schedules as comparison table."""
    if not schedules:
        return "No schedules to display"
    
    cycles = len(list(schedules.values())[0])
    
    # Header
    header = "| Cycle | " + " | ".join(schedules.keys()) + " |"
    separator = "|-------|" + "|".join(["---------"] * len(schedules)) + "|"
    
    # Rows
    rows = []
    for i in range(cycles):
        row_values = [f"{schedule[i]:.2e}" for schedule in schedules.values()]
        rows.append(f"| {i+1}     | " + " | ".join(row_values) + " |")
    
    return "\n".join([header, separator] + rows)


def visualize_schedules(
    schedules: Dict[str, List[float]],
    output_path: Optional[str] = None
) -> None:
    """
    Create matplotlib visualization of LR schedules.
    
    Args:
        schedules: Dict of name -> schedule list
        output_path: If provided, save to file instead of showing
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        print("Showing text-based visualization instead:\n")
        print(format_schedule_table(schedules))
        return
    
    plt.figure(figsize=(10, 6))
    
    for name, schedule in schedules.items():
        cycles = list(range(1, len(schedule) + 1))
        plt.plot(cycles, schedule, marker='o', label=name, linewidth=2, markersize=8)
    
    plt.xlabel('RAFT Cycle', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedules Across RAFT Cycles\n(EXPERIMENTAL - Not Validated)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Add annotation
    plt.figtext(0.5, 0.02, '⚠️ These schedules are theoretical and have not been empirically validated',
                ha='center', fontsize=10, style='italic', color='red')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()


def print_recommendations() -> None:
    """Print experimental recommendations."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LEARNING RATE RECOMMENDATIONS                              ║
║                        ⚠️  EXPERIMENTAL & THEORETICAL                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Starting Points (UNTESTED):                                                 ║
║  ─────────────────────────────                                               ║
║  • SFT with LoRA:     2e-4                                                   ║
║  • RAFT Cycle 1:      5e-5  (1/4 of SFT LR)                                 ║
║  • RAFT Cycle 5:      2e-5 to 3e-5  (with decay)                            ║
║                                                                              ║
║  Decay Strategies (THEORETICAL):                                             ║
║  ───────────────────────────────                                             ║
║  • Constant:          5e-5 all cycles  (baseline for comparison)            ║
║  • Moderate decay:    0.85 factor      (5e-5 → 2.6e-5 over 5 cycles)       ║
║  • Aggressive decay:  0.70 factor      (5e-5 → 1.2e-5 over 5 cycles)       ║
║                                                                              ║
║  When to Reduce LR:                                                          ║
║  ──────────────────                                                          ║
║  • Loss oscillates or spikes                                                 ║
║  • Gradients frequently clipped (hitting max_grad_norm)                      ║
║  • Cycle N+1 performs worse than Cycle N                                     ║
║  • Output diversity decreasing rapidly                                       ║
║                                                                              ║
║  When to Increase LR:                                                        ║
║  ───────────────────                                                         ║
║  • Loss barely changes across steps                                          ║
║  • Multiple cycles show no improvement                                       ║
║  • Training seems "stuck"                                                    ║
║                                                                              ║
║  NOTE: All of the above are hypotheses based on general ML principles.       ║
║        RLVR-specific dynamics may require different approaches.              ║
║        Validate empirically before committing to production runs.            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="Learning Rate Schedule Calculator for RLVR Training",
        epilog="⚠️  All schedules are EXPERIMENTAL and THEORETICAL"
    )
    
    parser.add_argument(
        "--cycles", "-c",
        type=int,
        default=5,
        help="Number of RAFT cycles (default: 5)"
    )
    parser.add_argument(
        "--base-lr", "-l",
        type=float,
        default=5e-5,
        help="Base learning rate for cycle 1 (default: 5e-5)"
    )
    parser.add_argument(
        "--decay", "-d",
        type=float,
        default=0.85,
        help="Decay factor per cycle (default: 0.85)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple decay strategies"
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Create visualization plot"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for visualization (shows plot if not specified)"
    )
    parser.add_argument(
        "--yaml",
        action="store_true",
        help="Output as YAML config block"
    )
    parser.add_argument(
        "--recommendations",
        action="store_true",
        help="Print experimental recommendations"
    )
    
    args = parser.parse_args()
    
    if args.recommendations:
        print_recommendations()
        return
    
    if args.compare:
        # Compare multiple strategies
        schedules = {
            "Constant (5e-5)": [5e-5] * args.cycles,
            "Decay 0.95": compute_exponential_decay(args.base_lr, 0.95, args.cycles),
            "Decay 0.85": compute_exponential_decay(args.base_lr, 0.85, args.cycles),
            "Decay 0.70": compute_exponential_decay(args.base_lr, 0.70, args.cycles),
        }
        
        print("\n⚠️  EXPERIMENTAL - These schedules are theoretical\n")
        print(format_schedule_table(schedules))
        
        if args.visualize:
            visualize_schedules(schedules, args.output)
    
    elif args.visualize:
        # Visualize single schedule
        schedule = compute_exponential_decay(args.base_lr, args.decay, args.cycles)
        schedules = {f"Decay {args.decay}": schedule}
        visualize_schedules(schedules, args.output)
    
    else:
        # Compute single schedule
        schedule = compute_exponential_decay(args.base_lr, args.decay, args.cycles)
        
        print(f"\n⚠️  EXPERIMENTAL Learning Rate Schedule")
        print(f"Base LR: {args.base_lr:.0e}, Decay: {args.decay}, Cycles: {args.cycles}\n")
        
        for i, lr in enumerate(schedule, 1):
            print(f"  Cycle {i}: {lr:.2e}")
        
        if args.yaml:
            print("\nYAML config block:")
            print("-" * 40)
            lr_schedule = LRSchedule(
                name=f"Exponential Decay ({args.decay})",
                base_lr=args.base_lr,
                decay_factor=args.decay,
                cycles=args.cycles,
                schedule=schedule
            )
            print(lr_schedule.to_yaml_block())
        
        print(f"\n⚠️  These values are THEORETICAL and should be validated empirically")


if __name__ == "__main__":
    main()
