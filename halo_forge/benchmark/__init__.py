"""
Benchmarking and Evaluation Module

This module is for **benchmark reporting** — producing metrics comparable to published papers.

Two evaluation modes in halo-forge:
1. **Training Verification** (halo_forge.rlvr.verifiers): Graduated rewards for RAFT training loop
2. **Benchmark Reporting** (this module): Standard metrics for papers and comparison

Supports multiple backends:
- Native: Code generation benchmarks (HumanEval, MBPP, LiveCodeBench, compile verification)
- VLMEvalKit: Vision-language benchmarks (community standard)

When to use:
- Use this module AFTER training to evaluate your model
- Use halo_forge.rlvr.verifiers DURING training for reward signals

Usage:
    from halo_forge.benchmark import run_benchmark, BenchmarkBackend
    
    # Auto-select backend based on model/benchmark
    result = run_benchmark("Qwen/Qwen2.5-Coder-3B", "humaneval")
    
    # Force specific backend
    result = run_benchmark("LiquidAI/LFM2.5-VL-1.6B", "MMStar", backend=BenchmarkBackend.VLMEVALKIT)

See docs/BENCHMARKS.md for the full benchmarking guide.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

from halo_forge.benchmark.pass_at_k import Benchmark, BenchmarkResult
from halo_forge.benchmark.runner import (
    BenchmarkRunner,
    BenchmarkResult as FullBenchmarkResult,
    EvalResult,
    CycleResult,
    run_benchmark_suite,
    DEFAULT_MODELS,
    BENCHMARK_PROMPTS,
)
from halo_forge.benchmark.prompts import (
    CPP_PROMPTS,
    RUST_PROMPTS,
    GO_PROMPTS,
    get_prompts_for_language,
    get_all_prompts,
    ALL_LANGUAGES,
)


class BenchmarkBackend(Enum):
    """Available benchmark backends."""
    NATIVE = "native"           # halo-forge native verifiers
    VLMEVALKIT = "vlmevalkit"   # VLMEvalKit for VLM benchmarks
    AUTO = "auto"               # Auto-select based on model/benchmark


# VLM benchmarks that should use VLMEvalKit
VLM_BENCHMARKS = {
    'mmstar', 'mmbench', 'mmmu', 'realworldqa', 'textvqa', 'docvqa',
    'chartqa', 'infovqa', 'ocrbench', 'mathvista', 'scienceqa',
    'mm-ifeval', 'blink', 'seedbench', 'pope', 'hallucination',
    'ai2d', 'gqa', 'vqa', 'okvqa', 'vizwiz',
}

# Model patterns that indicate VLM
VLM_MODEL_PATTERNS = [
    'vl', 'vision', 'llava', 'qwen2-vl', 'lfm', 'pixtral', 
    'cogvlm', 'internvl', 'cambrian', 'phi-3-vision',
]


def _select_backend(model: str, benchmark: str) -> BenchmarkBackend:
    """Auto-select appropriate backend based on model/benchmark."""
    bench_lower = benchmark.lower().replace('-', '').replace('_', '')
    
    # Check if it's a VLM benchmark
    if bench_lower in VLM_BENCHMARKS:
        return BenchmarkBackend.VLMEVALKIT
    
    # Check if model is a VLM
    model_lower = model.lower()
    for pattern in VLM_MODEL_PATTERNS:
        if pattern in model_lower:
            # VLM model, use VLMEvalKit for non-code benchmarks
            if bench_lower not in {'humaneval', 'mbpp', 'livecodebench'}:
                return BenchmarkBackend.VLMEVALKIT
    
    # Default to native
    return BenchmarkBackend.NATIVE


def run_benchmark(
    model: str,
    benchmark: str,
    backend: BenchmarkBackend = BenchmarkBackend.AUTO,
    limit: Optional[int] = None,
    output: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run a benchmark with automatic backend selection.
    
    Args:
        model: Model name or path
        benchmark: Benchmark name (e.g., "humaneval", "MMStar", "textvqa")
        backend: Which backend to use (AUTO selects based on model/benchmark)
        limit: Max samples to evaluate
        output: Output path for results
        **kwargs: Backend-specific arguments
    
    Returns:
        Dictionary with benchmark results
    """
    # Apply Strix Halo optimizations
    from halo_forge.utils.strix_halo import setup_strix_halo_env
    setup_strix_halo_env()
    
    # Select backend
    if backend == BenchmarkBackend.AUTO:
        backend = _select_backend(model, benchmark)
    
    if backend == BenchmarkBackend.VLMEVALKIT:
        return _run_vlmevalkit_benchmark(model, benchmark, limit, output, **kwargs)
    else:
        return _run_native_benchmark(model, benchmark, limit, output, **kwargs)


def _run_vlmevalkit_benchmark(
    model: str,
    benchmark: str,
    limit: Optional[int] = None,
    output: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run benchmark using VLMEvalKit."""
    try:
        from vlmeval.config import supported_VLM
        from vlmeval.evaluate import Evaluator
    except ImportError:
        return {
            'error': 'VLMEvalKit not installed. Install with: pip install vlmeval',
            'model': model,
            'benchmark': benchmark,
            'backend': 'vlmevalkit',
        }
    
    print(f"Running VLMEvalKit benchmark: {benchmark}")
    print(f"Model: {model}")
    print(f"Backend: VLMEvalKit (community standard)")
    
    try:
        evaluator = Evaluator(
            model=model,
            data=[benchmark],
            limit=limit,
            **kwargs
        )
        
        results = evaluator.run()
        
        # Convert to standard format
        result = {
            'model': model,
            'benchmark': benchmark,
            'backend': 'vlmevalkit',
            'metrics': {},
            'samples': 0,
        }
        
        if benchmark in results:
            bench_result = results[benchmark]
            # Extract metrics
            for key in ['accuracy', 'acc', 'score', 'overall']:
                if key in bench_result:
                    result['metrics'][key] = float(bench_result[key])
            result['samples'] = bench_result.get('num_samples', 0)
        
        # Save if output specified
        if output:
            import json
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'model': model,
            'benchmark': benchmark,
            'backend': 'vlmevalkit',
        }


def _run_native_benchmark(
    model: str,
    benchmark: str,
    limit: Optional[int] = None,
    output: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run benchmark using native halo-forge."""
    print(f"Running native benchmark: {benchmark}")
    print(f"Model: {model}")
    print(f"Backend: halo-forge native")
    
    # Use existing benchmark runner
    runner = BenchmarkRunner(
        model_name=model,
        n_cycles=0,  # Just evaluation, no training
        samples_per_prompt=kwargs.get('samples_per_prompt', 5),
        **kwargs
    )
    
    result = runner.run_evaluation_only(limit=limit)
    
    # Convert to standard format
    output_result = {
        'model': model,
        'benchmark': benchmark,
        'backend': 'native',
        'metrics': {
            'pass_at_1': result.pass_at_1,
            'compile_rate': result.compile_rate,
            'pass_rate': result.pass_rate,
        },
        'samples': result.total_samples,
    }
    
    # Save if output specified
    if output:
        import json
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(output_result, f, indent=2)
    
    return output_result


__all__ = [
    # Backend selection
    "BenchmarkBackend",
    "run_benchmark",
    # Legacy pass@k benchmark
    "Benchmark",
    "BenchmarkResult",
    # Full RAFT benchmark runner
    "BenchmarkRunner",
    "FullBenchmarkResult",
    "EvalResult",
    "CycleResult",
    "run_benchmark_suite",
    "DEFAULT_MODELS",
    "BENCHMARK_PROMPTS",
    # Language-specific prompts
    "CPP_PROMPTS",
    "RUST_PROMPTS",
    "GO_PROMPTS",
    "get_prompts_for_language",
    "get_all_prompts",
    "ALL_LANGUAGES",
]

