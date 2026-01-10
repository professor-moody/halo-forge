#!/usr/bin/env python3
"""
Benchmarking Module

Evaluate model code generation with pass@k metrics.
Uses pluggable verifiers for domain-agnostic evaluation.
"""

import json
import time
import math
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    
    # Core metrics
    total: int
    passed: int
    pass_rate: float
    
    # pass@k for different k values
    pass_at_k: Dict[int, float] = field(default_factory=dict)
    
    # Breakdown by category
    by_category: Dict[str, Dict] = field(default_factory=dict)
    
    # Timing
    generation_time: float = 0.0
    verification_time: float = 0.0
    total_time: float = 0.0
    
    # Model info
    model_path: str = ""
    
    # Detailed results
    samples: List[Dict] = field(default_factory=list)
    
    def __repr__(self):
        return f"BenchmarkResult(pass_rate={self.pass_rate:.1%}, total={self.total})"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'total': self.total,
            'passed': self.passed,
            'pass_rate': self.pass_rate,
            'pass_at_k': self.pass_at_k,
            'by_category': self.by_category,
            'timing': {
                'generation_time': self.generation_time,
                'verification_time': self.verification_time,
                'total_time': self.total_time
            },
            'model_path': self.model_path,
            'timestamp': datetime.now().isoformat()
        }
    
    def save(self, path: str):
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Estimate pass@k using unbiased estimator.
    
    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k value
    
    Returns:
        Estimated pass@k probability
    """
    if n - c < k:
        return 1.0
    
    # pass@k = 1 - C(n-c, k) / C(n, k)
    numerator = 1.0
    denominator = 1.0
    
    for i in range(k):
        numerator *= (n - c - i)
        denominator *= (n - i)
    
    return 1.0 - (numerator / denominator)


class Benchmark:
    """
    Model benchmarking with pass@k metrics.
    
    Evaluates a model on a set of prompts using pluggable verifiers.
    
    Example:
        verifier = GCCVerifier()
        benchmark = Benchmark(
            model_path="models/raft/cycle_3_final",
            verifier=verifier
        )
        result = benchmark.run(prompts, k_values=[1, 5, 10])
    """
    
    def __init__(
        self,
        model_path: str,
        verifier: Verifier,
        base_model: str = "Qwen/Qwen2.5-Coder-7B",
        system_prompt: str = "You are an expert Windows systems programmer."
    ):
        """
        Initialize benchmark.
        
        Args:
            model_path: Path to model or checkpoint
            verifier: Verifier instance
            base_model: Base model name (for LoRA checkpoints)
            system_prompt: System prompt for generation
        """
        self.model_path = model_path
        self.verifier = verifier
        self.base_model = base_model
        self.system_prompt = system_prompt
        
        self.model = None
        self.tokenizer = None
    
    def _load_model(self):
        """Load model and tokenizer."""
        if self.model is not None:
            return
        
        print(f"Loading model from {self.model_path}")
        
        # Check if it's a LoRA checkpoint
        adapter_config_path = Path(self.model_path) / "adapter_config.json"
        
        if adapter_config_path.exists():
            # LoRA checkpoint - read base_model from adapter config (authoritative source)
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model = adapter_config.get("base_model_name_or_path", self.base_model)
            print(f"Loading as LoRA checkpoint (base: {base_model})...")
            
            # Load tokenizer from base model
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=True,
                padding_side='left'
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model and apply adapter
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base, self.model_path)
        else:
            # Full model - reload tokenizer from model_path (not base_model)
            print("Loading as full model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side='left'
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager",
                trust_remote_code=True
            )
        
        self.model.eval()
        print("Model loaded")
    
    def generate(
        self,
        prompts: List[str],
        samples_per_prompt: int = 1,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        batch_size: int = 4
    ) -> List[Dict]:
        """
        Generate samples for each prompt.
        
        Args:
            prompts: List of prompts
            samples_per_prompt: Number of samples per prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            batch_size: Generation batch size
        
        Returns:
            List of {prompt, completions, metadata} dicts
        """
        self._load_model()
        
        total = len(prompts) * samples_per_prompt
        print(f"\nGenerating {total} samples...")
        
        all_results = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch = prompts[i:i+batch_size]
            
            # Format prompts
            formatted = []
            for prompt in batch:
                if isinstance(prompt, dict):
                    text = prompt.get('prompt', prompt.get('text', ''))
                    # Preserve metadata, but also capture root-level fields
                    metadata = prompt.get('metadata', {}).copy()
                    # Copy common fields from root level if not in metadata
                    for field in ['category', 'tier', 'difficulty', 'id', 'subcategory', 'api']:
                        if field not in metadata and field in prompt:
                            metadata[field] = prompt[field]
                else:
                    text = prompt
                    metadata = {}
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ]
                
                formatted.append({
                    'text': text,
                    'formatted': self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    ),
                    'metadata': metadata
                })
            
            # Tokenize
            inputs = self.tokenizer(
                [f['formatted'] for f in formatted],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Track input length to strip from output
            input_len = inputs['input_ids'].shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=samples_per_prompt,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode ONLY the new tokens (strip input prompt)
            # This prevents the full chat template from polluting completions
            all_outputs = []
            for output in outputs:
                new_tokens = output[input_len:]
                completion = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_outputs.append(completion)
            
            # Organize by prompt
            for j, item in enumerate(formatted):
                start_idx = j * samples_per_prompt
                end_idx = (j + 1) * samples_per_prompt
                completions = all_outputs[start_idx:end_idx]
                
                all_results.append({
                    'prompt': item['text'],
                    'completions': completions,
                    'metadata': item['metadata']
                })
        
        return all_results
    
    def verify_samples(
        self,
        samples: List[Dict]
    ) -> List[Dict]:
        """
        Verify all samples.
        
        Args:
            samples: List of {prompt, completions, metadata} dicts
        
        Returns:
            Samples with verification results added
        """
        print(f"\nVerifying samples...")
        
        # Flatten all completions for batch verification
        all_completions = []
        indices = []  # Track which sample each completion belongs to
        
        for i, sample in enumerate(samples):
            for completion in sample['completions']:
                all_completions.append(completion)
                indices.append(i)
        
        # Batch verify
        results = self.verifier.verify_batch(all_completions)
        
        # Organize results back to samples
        for sample in samples:
            sample['verification_results'] = []
        
        for idx, result in zip(indices, results):
            samples[idx]['verification_results'].append({
                'success': result.success,
                'reward': result.reward,
                'details': result.details
            })
        
        return samples
    
    def compute_metrics(
        self,
        samples: List[Dict],
        k_values: List[int] = [1, 5, 10]
    ) -> BenchmarkResult:
        """
        Compute pass@k and other metrics.
        
        Args:
            samples: Verified samples
            k_values: k values for pass@k
        
        Returns:
            BenchmarkResult
        """
        total = len(samples)
        passed = 0
        by_category = defaultdict(lambda: {'total': 0, 'passed': 0})
        
        # Per-prompt statistics for pass@k
        n_samples_per_prompt = len(samples[0]['completions']) if samples else 0
        correct_counts = []
        
        for sample in samples:
            # Count correct completions for this prompt
            n_correct = sum(
                1 for r in sample['verification_results']
                if r['success']
            )
            correct_counts.append(n_correct)
            
            # At least one success = passed
            if n_correct > 0:
                passed += 1
            
            # By category - check both metadata.category and root-level category
            category = (
                sample.get('metadata', {}).get('category') or 
                sample.get('category', 'unknown')
            )
            by_category[category]['total'] += 1
            if n_correct > 0:
                by_category[category]['passed'] += 1
        
        # pass@k
        pass_at_k = {}
        for k in k_values:
            if n_samples_per_prompt >= k:
                pass_at_k[k] = sum(
                    estimate_pass_at_k(n_samples_per_prompt, c, k)
                    for c in correct_counts
                ) / len(correct_counts) if correct_counts else 0.0
        
        # Category pass rates
        for cat in by_category:
            cat_data = by_category[cat]
            cat_data['pass_rate'] = cat_data['passed'] / cat_data['total'] if cat_data['total'] > 0 else 0.0
        
        return BenchmarkResult(
            total=total,
            passed=passed,
            pass_rate=passed / total if total > 0 else 0.0,
            pass_at_k=pass_at_k,
            by_category=dict(by_category),
            model_path=self.model_path,
            samples=[{
                'prompt': s['prompt'],
                'success': any(r['success'] for r in s['verification_results']),
                'correct_count': sum(1 for r in s['verification_results'] if r['success']),
                'metadata': s.get('metadata', {})
            } for s in samples]
        )
    
    def run(
        self,
        prompts: Union[List[str], List[Dict], str],
        samples_per_prompt: int = 10,
        k_values: List[int] = [1, 5, 10],
        max_prompts: int = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        batch_size: int = 4,
        output_path: str = None
    ) -> BenchmarkResult:
        """
        Run full benchmark.
        
        Args:
            prompts: Prompts or path to JSONL file
            samples_per_prompt: Samples to generate per prompt
            k_values: k values for pass@k
            max_prompts: Maximum prompts to evaluate
            max_new_tokens: Max tokens per generation
            temperature: Sampling temperature
            batch_size: Generation batch size
            output_path: Path to save results
        
        Returns:
            BenchmarkResult
        """
        start_time = time.time()
        
        # Load prompts from file if needed
        if isinstance(prompts, str):
            print(f"Loading prompts from {prompts}")
            loaded = []
            with open(prompts) as f:
                for line in f:
                    loaded.append(json.loads(line))
            prompts = loaded
        
        # Limit prompts
        if max_prompts and len(prompts) > max_prompts:
            prompts = prompts[:max_prompts]
        
        print(f"\nBenchmarking {len(prompts)} prompts")
        print(f"  samples/prompt: {samples_per_prompt}")
        print(f"  k values: {k_values}")
        
        # Generate
        gen_start = time.time()
        samples = self.generate(
            prompts,
            samples_per_prompt=samples_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size
        )
        gen_time = time.time() - gen_start
        
        # Verify
        verify_start = time.time()
        samples = self.verify_samples(samples)
        verify_time = time.time() - verify_start
        
        # Compute metrics
        result = self.compute_metrics(samples, k_values)
        result.generation_time = gen_time
        result.verification_time = verify_time
        result.total_time = time.time() - start_time
        
        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Total prompts: {result.total}")
        print(f"Passed: {result.passed} ({result.pass_rate:.1%})")
        print()
        
        for k, rate in result.pass_at_k.items():
            print(f"pass@{k}: {rate:.1%}")
        
        if result.by_category:
            print("\nBy category:")
            for cat, data in sorted(result.by_category.items()):
                print(f"  {cat}: {data['passed']}/{data['total']} ({data['pass_rate']:.1%})")
        
        print(f"\nTiming:")
        print(f"  Generation: {result.generation_time:.1f}s")
        print(f"  Verification: {result.verification_time:.1f}s")
        print(f"  Total: {result.total_time:.1f}s")
        
        # Save if requested
        if output_path:
            result.save(output_path)
            print(f"\nSaved to {output_path}")
        
        return result

