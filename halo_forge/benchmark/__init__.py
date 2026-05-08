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

import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

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
from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult


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

PYTHON_DATASET_BENCHMARKS = {"humaneval", "mbpp", "livecodebench"}


def _prompt_key(prompt: str) -> str:
    """Stable key used to map prompts to benchmark records."""
    return prompt[:200].strip()


def _load_jsonl_records(path: str) -> List[Dict[str, Any]]:
    """Load JSONL benchmark records."""
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _normalize_benchmark_key(name: str) -> str:
    """Normalize benchmark identifiers for backend routing comparisons."""
    return (name or "").strip().lower().replace("-", "").replace("_", "")


class LiveCodeBenchPythonVerifier(Verifier):
    """
    Lightweight verifier for Python LiveCodeBench-style stdin/stdout tasks.

    Expects records with `prompt` and `test_cases` where each test case has
    `input` and `expected`.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        timeout: int = 10,
        max_workers: int = 8,
        max_test_cases: int = 20,
    ):
        super().__init__(max_workers=max_workers)
        self.timeout = timeout
        self.max_test_cases = max_test_cases
        self.prompt_to_tests: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            prompt = str(record.get("prompt", "")).strip()
            if prompt:
                self.prompt_to_tests[_prompt_key(prompt)] = list(record.get("test_cases", []))

    def verify(self, code: str) -> VerifyResult:
        return VerifyResult(
            success=False,
            reward=0.0,
            details="Prompt-aware verification required",
            error="LiveCodeBench verifier requires prompt context",
        )

    def verify_batch(
        self,
        codes: List[str],
        prompts: Optional[List[str]] = None
    ) -> List[VerifyResult]:
        if not prompts or len(prompts) != len(codes):
            return [
                VerifyResult(
                    success=False,
                    reward=0.0,
                    details="Missing prompt context",
                    error="prompts required for LiveCodeBench verification",
                )
                for _ in codes
            ]

        results: List[Optional[VerifyResult]] = [None] * len(codes)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._verify_with_prompt, code, prompt): i
                for i, (code, prompt) in enumerate(zip(codes, prompts))
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:  # pragma: no cover - safety fallback
                    results[idx] = VerifyResult(
                        success=False,
                        reward=0.0,
                        details="Verification exception",
                        error=str(e),
                    )
        return [
            r if r is not None else VerifyResult(
                success=False,
                reward=0.0,
                details="Verification missing result",
                error="internal verifier error",
            )
            for r in results
        ]

    def _verify_with_prompt(self, code: str, prompt: str) -> VerifyResult:
        tests = self.prompt_to_tests.get(_prompt_key(prompt), [])
        if not tests:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="No test cases for prompt",
                error="unknown prompt",
            )
        return self._run_test_cases(code, tests)

    def _run_test_cases(self, code: str, tests: List[Dict[str, Any]]) -> VerifyResult:
        extracted = self.extract_code(code)
        selected = tests[:self.max_test_cases]
        if not selected:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="No executable test cases",
                error="empty test set",
            )

        passed = 0
        failures: List[str] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            script_path = tmp_path / "candidate.py"
            script_path.write_text(extracted, encoding="utf-8")

            for i, test_case in enumerate(selected, 1):
                test_input = str(test_case.get("input", ""))
                expected = str(test_case.get("expected", "")).strip()
                try:
                    proc = subprocess.run(
                        [sys.executable, str(script_path)],
                        input=test_input,
                        text=True,
                        capture_output=True,
                        timeout=self.timeout,
                        cwd=tmpdir,
                    )
                except subprocess.TimeoutExpired:
                    failures.append(f"test_{i}: timeout")
                    continue
                except Exception as e:  # pragma: no cover - safety fallback
                    failures.append(f"test_{i}: {e}")
                    continue

                actual = (proc.stdout or "").strip()
                if proc.returncode == 0 and actual == expected:
                    passed += 1
                else:
                    details = (proc.stderr or proc.stdout or "").strip()
                    snippet = details[:200] if details else "output mismatch"
                    failures.append(f"test_{i}: {snippet}")

        total = len(selected)
        success = passed == total
        reward = passed / total if total else 0.0
        details = f"Passed {passed}/{total} test cases"
        error = None if success else "; ".join(failures[:3])
        return VerifyResult(success=success, reward=reward, details=details, error=error)


def _select_backend(model: str, benchmark: str) -> BenchmarkBackend:
    """Auto-select appropriate backend based on model/benchmark."""
    bench_lower = _normalize_benchmark_key(benchmark)
    normalized_vlm_benchmarks = {
        _normalize_benchmark_key(name) for name in VLM_BENCHMARKS
    }
    
    # Check if it's a VLM benchmark
    if bench_lower in normalized_vlm_benchmarks:
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
            'error': (
                'VLMEvalKit not installed. Install with: '
                'pip install git+https://github.com/open-compass/VLMEvalKit.git'
            ),
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
    
    # Check for language-specific benchmarks (cpp, rust, go)
    language = kwargs.pop('language', None)
    verifier_type = kwargs.pop('verifier', None)
    run_after_compile = kwargs.pop('run_after_compile', True)
    
    if language or benchmark in ('cpp', 'rust', 'go', 'c++'):
        return _run_language_benchmark(
            model, benchmark, language, verifier_type, limit, output,
            run_after_compile=run_after_compile, **kwargs
        )

    if benchmark.lower() in PYTHON_DATASET_BENCHMARKS:
        return _run_python_dataset_benchmark(
            model=model,
            benchmark=benchmark,
            limit=limit,
            output=output,
            **kwargs,
        )

    # Keep internal prompt runner reserved for internal/native suite only.
    # Pop samples_per_prompt to avoid duplicate argument error
    samples = kwargs.pop('samples_per_prompt', 5)
    # Pop output_dir if passed, otherwise derive from output path or use default
    output_dir = kwargs.pop('output_dir', None)
    if output_dir is None:
        if output:
            output_dir = str(output.parent)
        else:
            output_dir = f"results/benchmarks/{Path(model).name}"
    
    runner = BenchmarkRunner(
        model_name=model,
        output_dir=output_dir,
        n_cycles=0,  # Just evaluation, no training
        samples_per_prompt=samples,
        **kwargs
    )
    
    result = runner.run_evaluation_only(limit=limit)
    
    # Convert to standard format
    output_result = {
        'model': model,
        'benchmark': benchmark,
        'backend': 'native',
        'execution_path': 'native-internal-prompts',
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


def _load_python_benchmark_records(
    benchmark: str,
    limit: Optional[int],
    dataset_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load benchmark records for Python benchmark datasets."""
    benchmark = benchmark.lower()

    if benchmark == "humaneval":
        path = Path(dataset_path or "data/rlvr/humaneval_full.jsonl")
        if not path.exists():
            raise FileNotFoundError(
                f"HumanEval dataset not found at {path}. "
                "Prepare it first (expected RLVR JSONL format)."
            )
        records = _load_jsonl_records(str(path))
    elif benchmark == "mbpp":
        path = Path(dataset_path or "data/rlvr/mbpp_train_full.jsonl")
        if not path.exists():
            raise FileNotFoundError(
                f"MBPP dataset not found at {path}. "
                "Prepare it first (expected RLVR JSONL format)."
            )
        records = _load_jsonl_records(str(path))
    elif benchmark == "livecodebench":
        candidates = [dataset_path] if dataset_path else []
        candidates.extend([
            "data/rlvr/livecodebench_full.jsonl",
            "data/livecodebench_rlvr.jsonl",
        ])
        records = []
        for candidate in candidates:
            if not candidate:
                continue
            candidate_path = Path(candidate)
            if candidate_path.exists():
                records = _load_jsonl_records(str(candidate_path))
                break
        if not records:
            try:
                from halo_forge.data.datasets.livecodebench import LiveCodeBenchLoader
                loader = LiveCodeBenchLoader(max_problems=limit or 500)
                records = loader.to_rlvr_format(language="python")
            except Exception as e:
                raise RuntimeError(
                    "Unable to load LiveCodeBench dataset records. "
                    "Provide a local RLVR JSONL dataset path to run_benchmark(...) "
                    "or install dataset dependencies."
                ) from e
    else:
        raise ValueError(f"Unsupported Python benchmark: {benchmark}")

    if limit is not None:
        records = records[:limit]
    return records


def _get_python_dataset_verifier(
    benchmark: str,
    records: List[Dict[str, Any]],
    dataset_path: Optional[str] = None,
):
    """Create verifier matching benchmark dataset semantics."""
    benchmark = benchmark.lower()
    if benchmark == "humaneval":
        from halo_forge.rlvr.verifiers import HumanEvalVerifier
        return HumanEvalVerifier(dataset_path or "data/rlvr/humaneval_full.jsonl")
    if benchmark == "mbpp":
        from halo_forge.rlvr.verifiers import MBPPVerifier
        return MBPPVerifier(dataset_path or "data/rlvr/mbpp_train_full.jsonl")
    if benchmark == "livecodebench":
        return LiveCodeBenchPythonVerifier(records)
    raise ValueError(f"Unsupported Python benchmark: {benchmark}")


def _run_python_dataset_benchmark(
    model: str,
    benchmark: str,
    limit: Optional[int] = None,
    output: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run dataset-faithful Python benchmark evaluation."""
    benchmark_lower = benchmark.lower()
    dataset_path = kwargs.pop("dataset_path", None)
    samples_per_prompt = kwargs.pop("samples_per_prompt", 5)
    max_new_tokens = kwargs.pop("max_new_tokens", 1024)
    temperature = kwargs.pop("temperature", 0.7)
    batch_size = kwargs.pop("batch_size", 4)

    records = _load_python_benchmark_records(
        benchmark=benchmark_lower,
        limit=limit,
        dataset_path=dataset_path,
    )
    prompts = records
    verifier = _get_python_dataset_verifier(
        benchmark=benchmark_lower,
        records=records,
        dataset_path=dataset_path,
    )

    benchmark_runner = Benchmark(
        model_path=model,
        verifier=verifier,
        system_prompt="You are an expert Python programmer. Write clean, correct code.",
    )
    output_path = str(output) if output else None

    result = benchmark_runner.run(
        prompts=prompts,
        samples_per_prompt=samples_per_prompt,
        k_values=[1, 5, 10],
        max_prompts=limit,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        output_path=output_path,
    )

    return {
        "model": model,
        "benchmark": benchmark_lower,
        "backend": "native",
        "execution_path": "native-python-dataset",
        "metrics": {
            "pass_at_1": result.pass_at_k.get(1, 0.0),
            "pass_at_5": result.pass_at_k.get(5, 0.0),
            "pass_at_10": result.pass_at_k.get(10, 0.0),
            "pass_rate": result.pass_rate,
        },
        "samples": result.total,
    }


def _run_language_benchmark(
    model: str,
    benchmark: str,
    language: Optional[str] = None,
    verifier_type: Optional[str] = None,
    limit: Optional[int] = None,
    output: Optional[Path] = None,
    run_after_compile: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run benchmark for compiled languages (C++, Rust, Go) using internal verifiers.
    
    This exposes our RAFT verifiers as a benchmark tool for evaluating models
    on compiled language generation.
    
    Args:
        run_after_compile: If True (MVR), run the compiled code and check output.
                          If False (MVP), only verify compilation.
    """
    from halo_forge.benchmark.prompts import get_prompts_for_language
    from halo_forge.benchmark.pass_at_k import Benchmark
    
    # Determine language from benchmark name if not specified
    if not language:
        language = benchmark.lower()
        if language == 'c++':
            language = 'cpp'
    
    print(f"Running {language.upper()} benchmark with internal verifiers")
    
    # Get prompts for the language
    prompts = get_prompts_for_language(language)
    if not prompts:
        return {
            'error': f'No prompts available for language: {language}',
            'model': model,
            'benchmark': benchmark,
            'backend': 'native',
        }
    
    # Apply limit if specified
    if limit and limit < len(prompts):
        prompts = prompts[:limit]
    
    print(f"Evaluating on {len(prompts)} {language.upper()} prompts")
    is_mingw_compile_only = (
        language in ("cpp", "c++", "c")
        and (verifier_type or "").lower() == "mingw"
    )
    if is_mingw_compile_only and run_after_compile:
        print("MinGW verifier is compile-only; ignoring run_after_compile=True.")
    mode = "MVP (compile-only)" if is_mingw_compile_only else (
        "MVR (full verification)" if run_after_compile else "MVP (compile-only)"
    )
    print(f"Verification mode: {mode}")
    
    # Select verifier based on language and verifier_type
    verifier = _get_verifier_for_language(language, verifier_type, run_after_compile)
    
    if verifier is None:
        return {
            'error': f'No verifier available for language: {language}',
            'model': model,
            'benchmark': benchmark,
            'backend': 'native',
        }
    
    print(f"Using verifier: {verifier.__class__.__name__}")
    
    # Run benchmark using pass@k calculator
    samples_per_prompt = kwargs.pop('samples_per_prompt', 5)
    
    benchmark_runner = Benchmark(
        model_path=model,
        verifier=verifier,
        system_prompt=f"You are an expert {language.upper()} programmer. Write clean, correct code.",
    )
    
    # Build output path
    output_path = None
    if output:
        output_path = str(output)
    
    # Run full benchmark
    result = benchmark_runner.run(
        prompts=prompts,
        samples_per_prompt=samples_per_prompt,
        k_values=[1, 5, 10],
        max_new_tokens=1024,
        temperature=0.7,
        output_path=output_path,
    )
    
    output_result = {
        'model': model,
        'benchmark': benchmark,
        'language': language,
        'backend': 'native-internal',
        'execution_path': 'native-compiled-language',
        'verifier': verifier.__class__.__name__,
        'verification_mode': mode,
        'metrics': {
            'pass_at_1': result.pass_at_k.get(1, 0.0),
            'pass_at_5': result.pass_at_k.get(5, 0.0),
            'pass_at_10': result.pass_at_k.get(10, 0.0),
            'pass_rate': result.pass_rate,
            'total_prompts': result.total,
            'passed': result.passed,
        },
        'by_category': result.by_category,
        'samples': result.total,
    }
    
    print(f"Results: {result.pass_rate:.1%} pass rate, pass@1={result.pass_at_k.get(1, 0):.1%}")
    
    # Save if output specified (already saved by benchmark_runner.run if output_path provided)
    if output and not output_path:
        import json
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(output_result, f, indent=2)
    
    return output_result


def _get_verifier_for_language(
    language: str,
    verifier_type: Optional[str] = None,
    run_after_compile: bool = True
):
    """
    Get the appropriate verifier for a language.
    
    Args:
        language: Target language (cpp, rust, go, python)
        verifier_type: Specific verifier (gcc, mingw, clang, rust, go)
        run_after_compile: If True, use MVR mode (run and check output).
                          If False, use MVP mode (compile-only).
    
    Returns:
        Configured verifier instance
    """
    from halo_forge.rlvr.verifiers import (
        GCCVerifier, MinGWVerifier, ClangVerifier,
        RustVerifier, GoVerifier,
    )
    
    language = language.lower()
    verifier_type = (verifier_type or '').lower()
    
    if language in ('cpp', 'c++', 'c'):
        if verifier_type == 'mingw':
            return MinGWVerifier(timeout=30)
        elif verifier_type == 'clang':
            return ClangVerifier(run_after_compile=run_after_compile, timeout=30)
        else:
            # Default to GCC
            return GCCVerifier(run_after_compile=run_after_compile, timeout=30)
    
    elif language == 'rust':
        return RustVerifier(run_after_compile=run_after_compile, timeout=60)
    
    elif language == 'go':
        return GoVerifier(run_after_compile=run_after_compile, timeout=30)
    
    return None


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
