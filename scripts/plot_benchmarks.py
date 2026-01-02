#!/usr/bin/env python3
"""
Generate benchmark visualization charts from halo-forge benchmark results.

Usage:
    python scripts/plot_benchmarks.py results/benchmarks/
    python scripts/plot_benchmarks.py results/benchmarks/ --output figures/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import sys

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError:
    print("Error: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#7C9885',     # Sage green
    'secondary': '#A8B5A2',   # Light sage
    'accent': '#C9A959',      # Muted gold
    'baseline': '#6B8E9B',    # Steel blue
    'final': '#7C9885',       # Sage green
    'hardware': '#B85C5C',    # Muted red
}


def load_benchmark_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all benchmark results from directory."""
    results = []
    
    # Check for comparison summary first
    comparison_path = results_dir / "comparison_summary.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            data = json.load(f)
            return data.get("models", [])
    
    # Otherwise, load individual model results
    for model_dir in sorted(results_dir.iterdir()):
        if model_dir.is_dir():
            summary_path = model_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    results.append(json.load(f))
    
    return results


def plot_compile_rate_comparison(results: List[Dict], output_path: Path):
    """Plot baseline vs final compile rates."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [r['model_short'] for r in results]
    baseline_rates = [r['baseline']['compile_rate'] * 100 for r in results]
    final_rates = [r['final']['compile_rate'] * 100 for r in results]
    
    x = range(len(models))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], baseline_rates, width, 
                   label='Baseline', color=COLORS['baseline'], alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], final_rates, width,
                   label='After RAFT', color=COLORS['final'], alpha=0.8)
    
    # Add improvement annotations
    for i, (base, final) in enumerate(zip(baseline_rates, final_rates)):
        improvement = final - base
        ax.annotate(f'+{improvement:.1f}%',
                   xy=(i + width/2, final + 1),
                   ha='center', va='bottom',
                   fontsize=10, color=COLORS['accent'], fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Compile Rate (%)', fontsize=12)
    ax.set_title('RAFT Training: Compile Rate Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, max(final_rates) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pass_at_1_comparison(results: List[Dict], output_path: Path):
    """Plot baseline vs final pass@1 rates."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = [r['model_short'] for r in results]
    baseline_rates = [r['baseline']['pass_at_1'] * 100 for r in results]
    final_rates = [r['final']['pass_at_1'] * 100 for r in results]
    
    x = range(len(models))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], baseline_rates, width,
                   label='Baseline', color=COLORS['baseline'], alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], final_rates, width,
                   label='After RAFT', color=COLORS['final'], alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('pass@1 (%)', fontsize=12)
    ax.set_title('RAFT Training: pass@1 Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper left')
    ax.set_ylim(0, max(max(final_rates), max(baseline_rates)) * 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_training_progress(results: List[Dict], output_path: Path):
    """Plot training loss across cycles for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [COLORS['primary'], COLORS['accent'], COLORS['hardware']]
    
    for i, result in enumerate(results):
        cycles = result.get('cycles', [])
        if not cycles:
            continue
        
        cycle_nums = [c['cycle'] for c in cycles]
        losses = [c['training_loss'] for c in cycles]
        
        ax.plot(cycle_nums, losses, 'o-', 
               label=result['model_short'],
               color=colors[i % len(colors)],
               linewidth=2, markersize=8)
    
    ax.set_xlabel('RAFT Cycle', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Across RAFT Cycles', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xticks(cycle_nums)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_gpu_utilization(results: List[Dict], output_path: Path):
    """Plot GPU utilization across models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = []
    avg_utils = []
    peak_utils = []
    
    for result in results:
        hw = result.get('hardware_summary', {})
        if not hw:
            continue
        
        models.append(result['model_short'])
        # Get from phases if available
        phases = hw.get('phases', {})
        if phases:
            all_peaks = [p['gpu']['utilization_peak_pct'] for p in phases.values() if 'gpu' in p]
            all_avgs = [p['gpu']['utilization_avg_pct'] for p in phases.values() if 'gpu' in p]
            peak_utils.append(max(all_peaks) if all_peaks else 0)
            avg_utils.append(sum(all_avgs) / len(all_avgs) if all_avgs else 0)
        else:
            peak_utils.append(hw.get('gpu_peak_utilization_pct', 0))
            avg_utils.append(hw.get('gpu_peak_utilization_pct', 0) * 0.9)  # Estimate
    
    x = range(len(models))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], avg_utils, width,
           label='Average', color=COLORS['primary'], alpha=0.8)
    ax.bar([i + width/2 for i in x], peak_utils, width,
           label='Peak', color=COLORS['accent'], alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax.set_title('GPU Utilization by Model Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Add reference line at 100%
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_memory_usage(results: List[Dict], output_path: Path):
    """Plot memory usage by model size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = []
    gpu_mems = []
    sys_mems = []
    
    for result in results:
        hw = result.get('hardware_summary', {})
        if not hw:
            continue
        
        models.append(result['model_short'])
        gpu_mems.append(hw.get('gpu_peak_memory_gb', 0))
        sys_mems.append(hw.get('sys_peak_memory_gb', 0))
    
    x = range(len(models))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], gpu_mems, width,
           label='GPU Memory (GTT)', color=COLORS['primary'], alpha=0.8)
    ax.bar([i + width/2 for i in x], sys_mems, width,
           label='System RAM', color=COLORS['secondary'], alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Memory Usage (GB)', fontsize=12)
    ax.set_title('Peak Memory Usage by Model Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    
    # Add reference line at 128GB (Strix Halo total)
    ax.axhline(y=128, color='gray', linestyle='--', alpha=0.5, label='128GB unified')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scaling_chart(results: List[Dict], output_path: Path):
    """Plot improvement vs model size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract model sizes (in billions of parameters)
    model_sizes = []
    improvements = []
    models = []
    
    for result in results:
        model = result['model_short']
        models.append(model)
        
        # Parse size from model name
        if '0.5b' in model.lower() or '0-5b' in model.lower():
            size = 0.5
        elif '1.5b' in model.lower() or '1-5b' in model.lower():
            size = 1.5
        elif '3b' in model.lower():
            size = 3.0
        elif '7b' in model.lower():
            size = 7.0
        else:
            size = 1.0  # Default
        
        model_sizes.append(size)
        
        improvement = result.get('improvement', {}).get('compile_rate', 0)
        improvements.append(improvement * 100)
    
    # Scatter plot with trend line
    ax.scatter(model_sizes, improvements, s=150, c=COLORS['primary'], 
               alpha=0.8, edgecolors='white', linewidth=2)
    
    # Add model labels
    for i, model in enumerate(models):
        ax.annotate(model, (model_sizes[i], improvements[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10)
    
    ax.set_xlabel('Model Size (Billion Parameters)', fontsize=12)
    ax.set_ylabel('Compile Rate Improvement (%)', fontsize=12)
    ax.set_title('RAFT Improvement vs Model Size', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(model_sizes) * 1.2)
    ax.set_ylim(0, max(improvements) * 1.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_time_breakdown(results: List[Dict], output_path: Path):
    """Plot time breakdown by phase."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = [r['model_short'] for r in results]
    
    # Calculate time components
    baseline_times = []
    cycle_times = []
    final_times = []
    
    for result in results:
        baseline = result.get('baseline', {})
        cycles = result.get('cycles', [])
        final = result.get('final', {})
        
        baseline_time = baseline.get('generation_time_sec', 0) + baseline.get('verification_time_sec', 0)
        cycle_time = sum(c.get('cycle_time_sec', 0) for c in cycles)
        final_time = final.get('generation_time_sec', 0) + final.get('verification_time_sec', 0)
        
        baseline_times.append(baseline_time / 60)  # Convert to minutes
        cycle_times.append(cycle_time / 60)
        final_times.append(final_time / 60)
    
    x = range(len(models))
    width = 0.6
    
    ax.bar(x, baseline_times, width, label='Baseline Eval', color=COLORS['baseline'], alpha=0.8)
    ax.bar(x, cycle_times, width, bottom=baseline_times, label='RAFT Cycles', color=COLORS['primary'], alpha=0.8)
    ax.bar(x, final_times, width, bottom=[b + c for b, c in zip(baseline_times, cycle_times)],
           label='Final Eval', color=COLORS['accent'], alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Time (minutes)', fontsize=12)
    ax.set_title('Benchmark Time Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(results: List[Dict], output_path: Path):
    """Generate a summary table as text."""
    lines = []
    lines.append("=" * 80)
    lines.append("BENCHMARK SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Header
    lines.append(f"{'Model':<25} {'Baseline':>12} {'Final':>12} {'Improve':>10} {'Time':>10}")
    lines.append("-" * 80)
    
    for result in results:
        model = result['model_short']
        baseline = result.get('baseline', {}).get('compile_rate', 0) * 100
        final = result.get('final', {}).get('compile_rate', 0) * 100
        improvement = final - baseline
        time_min = result.get('total_time_sec', 0) / 60
        
        lines.append(f"{model:<25} {baseline:>11.1f}% {final:>11.1f}% {improvement:>+9.1f}% {time_min:>9.1f}m")
    
    lines.append("-" * 80)
    lines.append("")
    
    # Hardware summary
    lines.append("HARDWARE UTILIZATION")
    lines.append("-" * 40)
    
    for result in results:
        hw = result.get('hardware_summary', {})
        if hw:
            lines.append(f"{result['model_short']}:")
            lines.append(f"  GPU Peak: {hw.get('gpu_peak_utilization_pct', 0):.0f}%")
            lines.append(f"  GPU Memory Peak: {hw.get('gpu_peak_memory_gb', 0):.1f} GB")
            lines.append(f"  Energy: {hw.get('total_energy_wh', 0):.2f} Wh")
    
    lines.append("=" * 80)
    
    content = "\n".join(lines)
    
    with open(output_path, "w") as f:
        f.write(content)
    
    print(f"Saved: {output_path}")
    print(content)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualization charts")
    parser.add_argument("results_dir", help="Directory containing benchmark results")
    parser.add_argument("--output", "-o", default=None, help="Output directory for figures")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Load results
    results = load_benchmark_results(results_dir)
    if not results:
        print(f"Error: No benchmark results found in {results_dir}")
        sys.exit(1)
    
    print(f"Loaded {len(results)} benchmark results")
    
    # Output directory
    output_dir = Path(args.output) if args.output else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print(f"\nGenerating charts in {output_dir}...")
    
    plot_compile_rate_comparison(results, output_dir / "compile_rate_comparison.png")
    plot_pass_at_1_comparison(results, output_dir / "pass_at_1_comparison.png")
    plot_training_progress(results, output_dir / "training_progress.png")
    plot_gpu_utilization(results, output_dir / "gpu_utilization.png")
    plot_memory_usage(results, output_dir / "memory_usage.png")
    plot_scaling_chart(results, output_dir / "scaling_chart.png")
    plot_time_breakdown(results, output_dir / "time_breakdown.png")
    generate_summary_table(results, output_dir / "summary.txt")
    
    print(f"\nDone! Charts saved to {output_dir}")


if __name__ == "__main__":
    main()

