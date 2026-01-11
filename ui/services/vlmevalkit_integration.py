"""
VLMEvalKit Integration

Wraps VLMEvalKit for standardized VLM benchmarking.
Uses VLMEvalKit as a dependency (not copying code) for community-trusted results.
"""

import os
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List


class BenchmarkBackend(Enum):
    """Available benchmark backends."""
    NATIVE = "native"           # halo-forge native verifiers
    VLMEVALKIT = "vlmevalkit"   # VLMEvalKit for VLM benchmarks
    AUTO = "auto"               # Auto-select based on model/benchmark


@dataclass
class VLMBenchmarkResult:
    """Result from VLMEvalKit benchmark."""
    model: str
    benchmark: str
    metrics: Dict[str, float] = field(default_factory=dict)
    samples: int = 0
    duration_seconds: float = 0.0
    backend: str = "vlmevalkit"
    raw_results: Optional[Dict] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def primary_metric(self) -> Optional[float]:
        """Get primary metric value."""
        for key in ['accuracy', 'acc', 'score', 'overall', 'avg']:
            if key in self.metrics:
                return self.metrics[key]
        return next(iter(self.metrics.values()), None) if self.metrics else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'model': self.model,
            'benchmark': self.benchmark,
            'metrics': self.metrics,
            'samples': self.samples,
            'duration_seconds': self.duration_seconds,
            'backend': self.backend,
            'timestamp': self.timestamp.isoformat(),
        }
    
    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class VLMEvalKitIntegration:
    """
    VLMEvalKit integration for standardized VLM benchmarking.
    
    This provides:
    - Community-trusted evaluation (same tool as Liquid AI)
    - AMD Strix Halo optimization
    - Consistent result format with halo-forge
    
    Usage:
        integration = VLMEvalKitIntegration()
        
        if integration.is_available():
            result = await integration.run_benchmark(
                model="LiquidAI/LFM2.5-VL-1.6B",
                benchmark="MMStar",
                limit=100,
            )
            print(f"Accuracy: {result.metrics.get('accuracy', 0):.2%}")
    """
    
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
    
    def __init__(self):
        """Initialize VLMEvalKit integration."""
        self._vlmeval_available: Optional[bool] = None
    
    def is_available(self) -> bool:
        """Check if VLMEvalKit is installed."""
        if self._vlmeval_available is not None:
            return self._vlmeval_available
        
        try:
            import vlmeval
            self._vlmeval_available = True
        except ImportError:
            self._vlmeval_available = False
        
        return self._vlmeval_available
    
    def _apply_strix_halo_env(self):
        """Apply AMD Strix Halo environment optimizations."""
        # Check if we're on Strix Halo
        try:
            from halo_forge.utils.hardware import detect_strix_halo
            is_strix, info = detect_strix_halo()
            
            if is_strix:
                print(f"Detected AMD Strix Halo ({info.get('total_memory_gb', 'N/A')}GB unified memory)")
        except ImportError:
            is_strix = True  # Assume Strix Halo if can't detect
        
        # GPU architecture
        os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '11.5.1')
        os.environ.setdefault('PYTORCH_ROCM_ARCH', 'gfx1151')
        os.environ.setdefault('HIP_VISIBLE_DEVICES', '0')
        
        # Memory optimization for unified memory
        os.environ.setdefault(
            'PYTORCH_HIP_ALLOC_CONF',
            'backend:native,expandable_segments:True,garbage_collection_threshold:0.9'
        )
        
        # Stability
        os.environ.setdefault('HSA_ENABLE_SDMA', '0')
    
    def list_benchmarks(self) -> List[str]:
        """List available VLM benchmarks."""
        if not self.is_available():
            return list(self.VLM_BENCHMARKS)
        
        try:
            from vlmeval.dataset import SUPPORTED_DATASETS
            return list(SUPPORTED_DATASETS)
        except Exception:
            return list(self.VLM_BENCHMARKS)
    
    def list_models(self) -> List[str]:
        """List supported VLM models."""
        if not self.is_available():
            return []
        
        try:
            from vlmeval.config import supported_VLM
            return list(supported_VLM.keys())
        except Exception:
            return []
    
    async def run_benchmark(
        self,
        model: str,
        benchmark: str,
        limit: Optional[int] = None,
        output: Optional[Path] = None,
        **kwargs
    ) -> VLMBenchmarkResult:
        """
        Run VLM benchmark using VLMEvalKit.
        
        Args:
            model: Model name or path
            benchmark: Benchmark name (e.g., "MMStar", "TextVQA")
            limit: Max samples to evaluate
            output: Output path for results
            **kwargs: Additional VLMEvalKit arguments
            
        Returns:
            VLMBenchmarkResult with metrics
        """
        import time
        start_time = time.time()
        
        if not self.is_available():
            return VLMBenchmarkResult(
                model=model,
                benchmark=benchmark,
                metrics={'error': 1.0},
                backend="vlmevalkit_unavailable",
            )
        
        # Apply Strix Halo optimizations
        self._apply_strix_halo_env()
        
        print(f"Running VLMEvalKit benchmark: {benchmark}")
        print(f"Model: {model}")
        print(f"Backend: VLMEvalKit (community standard)")
        
        try:
            # Import VLMEvalKit
            from vlmeval.config import supported_VLM
            
            # Resolve model name
            vlm_model_name = self._resolve_model_name(model, supported_VLM)
            
            # Use VLMEvalKit's evaluation
            from vlmeval.evaluate import Evaluator
            
            evaluator = Evaluator(
                model=vlm_model_name,
                data=[benchmark],
                limit=limit,
                **kwargs
            )
            
            raw_results = evaluator.run()
            
            # Convert to our format
            result = VLMBenchmarkResult(
                model=model,
                benchmark=benchmark,
                raw_results=raw_results,
                duration_seconds=time.time() - start_time,
            )
            
            # Extract metrics
            if benchmark in raw_results:
                bench_result = raw_results[benchmark]
                result.metrics = self._extract_metrics(bench_result)
                result.samples = bench_result.get('num_samples', 0)
            
        except ImportError as e:
            # VLMEvalKit API might have changed
            result = VLMBenchmarkResult(
                model=model,
                benchmark=benchmark,
                metrics={'error': 1.0, 'message': str(e)},
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            result = VLMBenchmarkResult(
                model=model,
                benchmark=benchmark,
                metrics={'error': 1.0, 'message': str(e)},
                duration_seconds=time.time() - start_time,
            )
        
        # Save if output specified
        if output:
            result.save(output)
            print(f"Results saved to: {output}")
        
        return result
    
    def _resolve_model_name(self, model: str, registry: dict) -> str:
        """Resolve model name to VLMEvalKit registry name."""
        # Exact match
        if model in registry:
            return model
        
        # Case-insensitive match
        model_lower = model.lower()
        for key in registry:
            if key.lower() == model_lower:
                return key
        
        # Partial match
        for key in registry:
            if model_lower in key.lower() or key.lower() in model_lower:
                return key
        
        # Local path - use as-is
        if Path(model).exists():
            return model
        
        # Return original and let VLMEvalKit handle it
        return model
    
    def _extract_metrics(self, bench_result: dict) -> Dict[str, float]:
        """Extract standardized metrics from VLMEvalKit result."""
        metrics = {}
        
        # Common metric names in VLMEvalKit
        metric_keys = [
            'accuracy', 'acc', 'score', 'overall', 'avg',
            'pass@1', 'pass@5', 'exact_match', 'f1',
        ]
        
        for key in metric_keys:
            if key in bench_result:
                try:
                    metrics[key] = float(bench_result[key])
                except (ValueError, TypeError):
                    pass
        
        # Also include any other numeric values
        for key, value in bench_result.items():
            if isinstance(value, (int, float)) and key not in metrics:
                metrics[key] = float(value)
        
        return metrics


# Singleton instance
_integration: Optional[VLMEvalKitIntegration] = None


def get_integration() -> VLMEvalKitIntegration:
    """Get the singleton VLMEvalKit integration."""
    global _integration
    if _integration is None:
        _integration = VLMEvalKitIntegration()
    return _integration


def is_vlmevalkit_available() -> bool:
    """Check if VLMEvalKit is available."""
    return get_integration().is_available()


def list_vlm_benchmarks() -> List[str]:
    """List available VLM benchmarks."""
    return get_integration().list_benchmarks()


def list_vlm_models() -> List[str]:
    """List supported VLM models."""
    return get_integration().list_models()


async def run_vlm_benchmark(
    model: str,
    benchmark: str,
    limit: Optional[int] = None,
    output: Optional[Path] = None,
    **kwargs
) -> VLMBenchmarkResult:
    """
    Run a VLM benchmark.
    
    Convenience function that uses VLMEvalKit integration.
    """
    return await get_integration().run_benchmark(
        model=model,
        benchmark=benchmark,
        limit=limit,
        output=output,
        **kwargs
    )


def should_use_vlmevalkit(model: str, benchmark: str) -> bool:
    """
    Determine if VLMEvalKit should be used for this model/benchmark.
    
    Args:
        model: Model name
        benchmark: Benchmark name
        
    Returns:
        True if VLMEvalKit is appropriate
    """
    integration = get_integration()
    
    # Check benchmark
    bench_lower = benchmark.lower().replace('-', '').replace('_', '')
    if bench_lower in integration.VLM_BENCHMARKS:
        return True
    
    # Check model patterns
    model_lower = model.lower()
    for pattern in integration.VLM_MODEL_PATTERNS:
        if pattern in model_lower:
            # VLM model, but only use VLMEvalKit for VLM benchmarks
            if bench_lower not in {'humaneval', 'mbpp', 'livecodebench'}:
                return True
    
    return False
