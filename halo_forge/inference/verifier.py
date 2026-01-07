"""
Inference Optimization Verifier

Verifies that optimized models maintain quality while meeting
latency and memory targets.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult


@dataclass
class InferenceMetrics:
    """Metrics from inference optimization verification."""
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    tokens_per_second: float
    memory_used_mb: float
    quality_score: float


class InferenceOptimizationVerifier(Verifier):
    """
    Verifies that optimized models maintain quality while meeting performance targets.
    
    Verification Process:
    1. Measure latency on test prompts
    2. Compare output quality to baseline model
    3. Calculate combined reward based on latency + quality
    
    Usage:
        verifier = InferenceOptimizationVerifier(
            baseline_model=baseline,
            target_latency_ms=50,
            quality_threshold=0.95
        )
        result = verifier.verify(optimized_model, test_prompts)
    """
    
    def __init__(
        self,
        baseline_model: Optional[Any] = None,
        baseline_model_name: Optional[str] = None,
        target_latency_ms: float = 50.0,
        quality_threshold: float = 0.95,
        max_new_tokens: int = 100,
        num_warmup: int = 3,
        max_workers: int = 1  # Sequential for accurate timing
    ):
        """
        Initialize the inference optimization verifier.
        
        Args:
            baseline_model: Loaded baseline model for quality comparison
            baseline_model_name: Model name to load as baseline (if not provided)
            target_latency_ms: Target latency in milliseconds
            quality_threshold: Minimum quality score (0-1) for success
            max_new_tokens: Tokens to generate per prompt
            num_warmup: Warmup iterations before timing
            max_workers: Parallel workers (1 recommended for timing)
        """
        super().__init__(max_workers=max_workers)
        
        self.baseline_model = baseline_model
        self.baseline_model_name = baseline_model_name
        self.target_latency_ms = target_latency_ms
        self.quality_threshold = quality_threshold
        self.max_new_tokens = max_new_tokens
        self.num_warmup = num_warmup
        
        self._baseline_loaded = False
        self._tokenizer = None
    
    def _ensure_baseline(self):
        """Load baseline model if not already loaded."""
        if self.baseline_model is not None:
            self._baseline_loaded = True
            return
        
        if self.baseline_model_name and not self._baseline_loaded:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading baseline model: {self.baseline_model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.baseline_model_name,
                trust_remote_code=True
            )
            self.baseline_model = AutoModelForCausalLM.from_pretrained(
                self.baseline_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self._baseline_loaded = True
    
    def measure_latency(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[str]
    ) -> List[float]:
        """
        Measure generation latency for each prompt.
        
        Args:
            model: Model to measure
            tokenizer: Tokenizer for the model
            prompts: List of prompts to generate from
            
        Returns:
            List of latencies in milliseconds
        """
        latencies = []
        
        # Warmup
        for prompt in prompts[:self.num_warmup]:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
        
        # Measure
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        return latencies
    
    def compare_quality(
        self,
        optimized_model: Any,
        baseline_model: Any,
        tokenizer: Any,
        prompts: List[str]
    ) -> float:
        """
        Compare output quality between optimized and baseline models.
        
        Uses token-level agreement as a simple quality metric.
        
        Args:
            optimized_model: Optimized model to evaluate
            baseline_model: Baseline model for comparison
            tokenizer: Tokenizer for both models
            prompts: List of prompts
            
        Returns:
            Quality score (0-1)
        """
        agreements = []
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(optimized_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Generate from optimized
                opt_outputs = optimized_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
                
                # Generate from baseline
                baseline_inputs = {k: v.to(baseline_model.device) for k, v in inputs.items()}
                base_outputs = baseline_model.generate(
                    **baseline_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False
                )
            
            # Compare outputs
            opt_text = tokenizer.decode(opt_outputs[0], skip_special_tokens=True)
            base_text = tokenizer.decode(base_outputs[0], skip_special_tokens=True)
            
            # Simple token overlap metric
            opt_tokens = set(opt_text.split())
            base_tokens = set(base_text.split())
            
            if len(base_tokens) > 0:
                agreement = len(opt_tokens & base_tokens) / len(base_tokens)
            else:
                agreement = 1.0 if len(opt_tokens) == 0 else 0.0
            
            agreements.append(agreement)
        
        return sum(agreements) / len(agreements) if agreements else 0.0
    
    def verify(
        self,
        optimized_model: Any,
        test_prompts: List[str],
        tokenizer: Any = None
    ) -> VerifyResult:
        """
        Verify optimized model meets performance and quality targets.
        
        Args:
            optimized_model: Optimized model to verify
            test_prompts: List of test prompts
            tokenizer: Tokenizer (uses model's tokenizer if not provided)
            
        Returns:
            VerifyResult with combined latency + quality reward
        """
        self._ensure_baseline()
        
        if tokenizer is None:
            if self._tokenizer is not None:
                tokenizer = self._tokenizer
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    self.baseline_model_name or "Qwen/Qwen2.5-Coder-0.5B",
                    trust_remote_code=True
                )
        
        # Measure latency
        latencies = self.measure_latency(optimized_model, tokenizer, test_prompts)
        avg_latency = sum(latencies) / len(latencies)
        
        # Measure quality (if baseline available)
        if self.baseline_model is not None:
            quality = self.compare_quality(
                optimized_model,
                self.baseline_model,
                tokenizer,
                test_prompts
            )
        else:
            quality = 1.0  # Assume full quality without baseline
        
        # Calculate reward
        # Latency factor: 1.0 if at target, scales down if slower
        latency_factor = min(1.0, self.target_latency_ms / avg_latency) if avg_latency > 0 else 0.0
        
        # Combined reward: 50% latency, 50% quality
        reward = 0.5 * latency_factor + 0.5 * quality
        
        # Success if quality meets threshold
        success = quality >= self.quality_threshold
        
        # Calculate tokens per second
        total_tokens = len(test_prompts) * self.max_new_tokens
        total_time_s = sum(latencies) / 1000
        tokens_per_second = total_tokens / total_time_s if total_time_s > 0 else 0
        
        # Memory usage
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            memory_mb = 0
        
        metrics = InferenceMetrics(
            avg_latency_ms=avg_latency,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            tokens_per_second=tokens_per_second,
            memory_used_mb=memory_mb,
            quality_score=quality
        )
        
        return VerifyResult(
            success=success,
            reward=reward,
            details=f"Latency: {avg_latency:.1f}ms, Quality: {quality:.2%}",
            metadata={
                "avg_latency_ms": avg_latency,
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "tokens_per_second": tokens_per_second,
                "memory_mb": memory_mb,
                "quality_score": quality,
                "latency_factor": latency_factor,
                "target_latency_ms": self.target_latency_ms,
                "quality_threshold": self.quality_threshold
            }
        )
    
    def cleanup(self):
        """Clean up baseline model."""
        if self.baseline_model is not None and self._baseline_loaded:
            del self.baseline_model
            self.baseline_model = None
            self._baseline_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
