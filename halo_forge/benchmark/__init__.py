"""Benchmarking and evaluation module."""

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

__all__ = [
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
]

