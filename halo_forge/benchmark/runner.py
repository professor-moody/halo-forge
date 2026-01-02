"""
Benchmark runner for halo-forge.

Runs systematic benchmarks on models to demonstrate RLVR effectiveness,
collecting both training metrics and hardware utilization data.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import gc

from halo_forge.utils.hw_monitor import HardwareMonitor, HardwareSummary


@dataclass
class EvalResult:
    """Evaluation results."""
    total_prompts: int = 0
    total_samples: int = 0
    compiled: int = 0
    passed: int = 0  # Samples that passed runtime verification
    compile_rate: float = 0.0
    pass_rate: float = 0.0
    pass_at_1: float = 0.0  # At least 1 sample per prompt compiles
    pass_at_k: float = 0.0  # All k samples per prompt
    avg_reward: float = 0.0
    tokens_per_sec: float = 0.0
    generation_time_sec: float = 0.0
    verification_time_sec: float = 0.0
    hardware: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_prompts": self.total_prompts,
            "total_samples": self.total_samples,
            "compiled": self.compiled,
            "passed": self.passed,
            "compile_rate": round(self.compile_rate, 4),
            "pass_rate": round(self.pass_rate, 4),
            "pass_at_1": round(self.pass_at_1, 4),
            "pass_at_k": round(self.pass_at_k, 4),
            "avg_reward": round(self.avg_reward, 4),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "generation_time_sec": round(self.generation_time_sec, 1),
            "verification_time_sec": round(self.verification_time_sec, 1),
            "hardware": self.hardware,
        }


@dataclass
class CycleResult:
    """Single RAFT cycle results."""
    cycle: int
    generated: int = 0
    verified: int = 0
    kept: int = 0
    compile_rate: float = 0.0
    avg_reward: float = 0.0
    training_loss: float = 0.0
    training_steps: int = 0
    cycle_time_sec: float = 0.0
    hardware: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "generated": self.generated,
            "verified": self.verified,
            "kept": self.kept,
            "compile_rate": round(self.compile_rate, 4),
            "avg_reward": round(self.avg_reward, 4),
            "training_loss": round(self.training_loss, 6),
            "training_steps": self.training_steps,
            "cycle_time_sec": round(self.cycle_time_sec, 1),
            "hardware": self.hardware,
        }


@dataclass
class BenchmarkResult:
    """Full benchmark results for a model."""
    model_name: str
    model_short: str
    n_cycles: int
    total_time_sec: float = 0.0
    baseline: Optional[EvalResult] = None
    cycles: List[CycleResult] = field(default_factory=list)
    final: Optional[EvalResult] = None
    hardware_summary: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_short": self.model_short,
            "n_cycles": self.n_cycles,
            "total_time_sec": round(self.total_time_sec, 1),
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "cycles": [c.to_dict() for c in self.cycles],
            "final": self.final.to_dict() if self.final else None,
            "hardware_summary": self.hardware_summary,
            "improvement": {
                "compile_rate": round(
                    (self.final.compile_rate - self.baseline.compile_rate) 
                    if self.final and self.baseline else 0, 4
                ),
                "pass_at_1": round(
                    (self.final.pass_at_1 - self.baseline.pass_at_1)
                    if self.final and self.baseline else 0, 4
                ),
            }
        }
    
    def save(self, path: str):
        """Save results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Built-in benchmark prompts
BENCHMARK_PROMPTS = [
    {
        "id": "hello_world",
        "prompt": "Write a C++ program that prints 'Hello, World!' to stdout.",
        "expected_output": "Hello, World!",
        "difficulty": "easy",
    },
    {
        "id": "sum_function",
        "prompt": "Write a C++ function that returns the sum of two integers a and b, then call it in main to print the result of 5 + 3.",
        "expected_output": "8",
        "difficulty": "easy",
    },
    {
        "id": "count_1_to_5",
        "prompt": "Write a C++ program that prints the numbers 1 through 5, each on a new line.",
        "expected_output": "1\n2\n3\n4\n5",
        "difficulty": "easy",
    },
    {
        "id": "factorial",
        "prompt": "Write a C++ program that computes and prints the factorial of 6.",
        "expected_output": "720",
        "difficulty": "medium",
    },
    {
        "id": "fibonacci",
        "prompt": "Write a C++ program that prints the first 10 Fibonacci numbers, space-separated.",
        "expected_output": "0 1 1 2 3 5 8 13 21 34",
        "difficulty": "medium",
    },
    {
        "id": "reverse_string",
        "prompt": "Write a C++ program that reverses the string 'hello' and prints it.",
        "expected_output": "olleh",
        "difficulty": "medium",
    },
    {
        "id": "is_prime",
        "prompt": "Write a C++ program that checks if 17 is prime and prints 'yes' or 'no'.",
        "expected_output": "yes",
        "difficulty": "medium",
    },
    {
        "id": "array_sum",
        "prompt": "Write a C++ program that calculates and prints the sum of array {1, 2, 3, 4, 5}.",
        "expected_output": "15",
        "difficulty": "easy",
    },
    {
        "id": "max_element",
        "prompt": "Write a C++ program that finds and prints the maximum element in array {3, 1, 4, 1, 5, 9, 2, 6}.",
        "expected_output": "9",
        "difficulty": "medium",
    },
    {
        "id": "bubble_sort",
        "prompt": "Write a C++ program that sorts array {5, 2, 8, 1, 9} using bubble sort and prints the sorted array space-separated.",
        "expected_output": "1 2 5 8 9",
        "difficulty": "medium",
    },
    {
        "id": "gcd",
        "prompt": "Write a C++ program that computes and prints the GCD of 48 and 18.",
        "expected_output": "6",
        "difficulty": "medium",
    },
    {
        "id": "binary_search",
        "prompt": "Write a C++ program that uses binary search to find the index of 7 in sorted array {1, 3, 5, 7, 9, 11} and prints it.",
        "expected_output": "3",
        "difficulty": "hard",
    },
    {
        "id": "palindrome",
        "prompt": "Write a C++ program that checks if 'racecar' is a palindrome and prints 'yes' or 'no'.",
        "expected_output": "yes",
        "difficulty": "medium",
    },
    {
        "id": "count_vowels",
        "prompt": "Write a C++ program that counts and prints the number of vowels in 'hello world'.",
        "expected_output": "3",
        "difficulty": "easy",
    },
    {
        "id": "power_function",
        "prompt": "Write a C++ program that computes and prints 2 raised to the power of 10.",
        "expected_output": "1024",
        "difficulty": "easy",
    },
    {
        "id": "linked_list_sum",
        "prompt": "Write a C++ program with a simple linked list containing 1, 2, 3 and print their sum.",
        "expected_output": "6",
        "difficulty": "hard",
    },
]


class BenchmarkRunner:
    """
    Runs comprehensive benchmarks on a model.
    
    Usage:
        runner = BenchmarkRunner(
            model_name="Qwen/Qwen2.5-Coder-0.5B",
            output_dir="results/qwen-0.5b",
            n_cycles=2
        )
        result = runner.run()
        result.save("results/qwen-0.5b/summary.json")
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        n_cycles: int = 2,
        samples_per_prompt: int = 8,
        prompts: Optional[List[Dict]] = None,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.model_short = model_name.split("/")[-1].lower().replace(".", "-")
        self.output_dir = Path(output_dir)
        self.n_cycles = n_cycles
        self.samples_per_prompt = samples_per_prompt
        self.prompts = prompts or BENCHMARK_PROMPTS
        self.verbose = verbose
        
        # Hardware monitor
        self.hw_monitor = HardwareMonitor(interval_sec=2)
        
        # Results
        self.result = BenchmarkResult(
            model_name=model_name,
            model_short=self.model_short,
            n_cycles=n_cycles,
        )
        
        # State
        self.model = None
        self.tokenizer = None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(msg)
    
    def _load_model(self):
        """Load model and tokenizer with PEFT/LoRA for training."""
        self.log(f"Loading model: {self.model_name}")
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model in BF16 (optimal for Strix Halo)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        
        # Apply LoRA for efficient training
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.enable_input_require_grads()  # Required for gradient checkpointing
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        self.log(f"Model loaded: {total_params / 1e6:.0f}M params, {trainable_params / 1e6:.1f}M trainable ({100 * trainable_params / total_params:.2f}%)")
    
    def _generate_samples(self, prompts: List[Dict]) -> List[Dict]:
        """Generate code samples for prompts with batch processing."""
        import torch
        from tqdm import tqdm
        
        samples = []
        total_tokens = 0
        start_time = time.time()
        batch_size = 4  # Process multiple prompts in parallel
        
        self.model.eval()
        
        # Process prompts in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating", disable=not self.verbose):
            batch_prompts = prompts[i:i+batch_size]
            
            # Format all prompts in batch
            formatted = []
            for prompt_data in batch_prompts:
                messages = [
                    {"role": "system", "content": "You are an expert programmer."},
                    {"role": "user", "content": prompt_data["prompt"]}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                formatted.append(formatted_prompt)
            
            # Tokenize batch
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate multiple samples per prompt
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,  # Match production
                    temperature=0.7,
                    do_sample=True,
                    num_return_sequences=self.samples_per_prompt,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Decode samples
            for j, prompt_data in enumerate(batch_prompts):
                start_idx = j * self.samples_per_prompt
                end_idx = (j + 1) * self.samples_per_prompt
                prompt_outputs = outputs[start_idx:end_idx]
                
                for k, output in enumerate(prompt_outputs):
                    new_tokens = output[inputs.input_ids.shape[1]:]
                    total_tokens += len(new_tokens)
                    code = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    
                    samples.append({
                        "prompt_id": prompt_data.get("id", "unknown"),
                        "prompt": prompt_data["prompt"],
                        "expected_output": prompt_data.get("expected_output"),
                        "code": code,
                        "sample_idx": k,
                    })
        
        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
        
        self.log(f"Generated {len(samples)} samples ({tokens_per_sec:.1f} tok/s)")
        
        return samples, elapsed, tokens_per_sec
    
    def _verify_samples(self, samples: List[Dict]) -> List[Dict]:
        """Verify generated samples using GCC."""
        from halo_forge.rlvr.verifiers import GCCVerifier
        
        start_time = time.time()
        verifier = GCCVerifier(timeout=10)
        
        for sample in samples:
            result = verifier.verify(sample["code"])
            sample["compiles"] = result.success
            sample["reward"] = result.reward
            sample["error"] = result.error if not result.success else None
            
            # Runtime verification if expected output provided
            if result.success and sample.get("expected_output"):
                # Simple runtime check via verifier with run_after_compile
                rt_verifier = GCCVerifier(
                    timeout=10,
                    run_after_compile=True,
                    expected_output=sample["expected_output"],
                )
                rt_result = rt_verifier.verify(sample["code"])
                sample["passes"] = rt_result.reward >= 0.9
                sample["actual_output"] = getattr(rt_result, "stdout", "")
            else:
                sample["passes"] = False
        
        elapsed = time.time() - start_time
        self.log(f"Verified {len(samples)} samples in {elapsed:.1f}s")
        
        return samples, elapsed
    
    def _calculate_eval_metrics(
        self,
        samples: List[Dict],
        gen_time: float,
        verify_time: float,
        tokens_per_sec: float,
    ) -> EvalResult:
        """Calculate evaluation metrics from samples."""
        total_samples = len(samples)
        compiled = sum(1 for s in samples if s.get("compiles", False))
        passed = sum(1 for s in samples if s.get("passes", False))
        
        # Group by prompt
        prompts = {}
        for s in samples:
            pid = s["prompt_id"]
            if pid not in prompts:
                prompts[pid] = []
            prompts[pid].append(s)
        
        # pass@1: at least one sample per prompt compiles
        pass_at_1_count = sum(
            1 for samples in prompts.values()
            if any(s.get("compiles", False) for s in samples)
        )
        
        # pass@k: all samples compile (stricter)
        pass_at_k_count = sum(
            1 for samples in prompts.values()
            if all(s.get("compiles", False) for s in samples)
        )
        
        avg_reward = sum(s.get("reward", 0) for s in samples) / total_samples if total_samples else 0
        
        return EvalResult(
            total_prompts=len(prompts),
            total_samples=total_samples,
            compiled=compiled,
            passed=passed,
            compile_rate=compiled / total_samples if total_samples else 0,
            pass_rate=passed / total_samples if total_samples else 0,
            pass_at_1=pass_at_1_count / len(prompts) if prompts else 0,
            pass_at_k=pass_at_k_count / len(prompts) if prompts else 0,
            avg_reward=avg_reward,
            tokens_per_sec=tokens_per_sec,
            generation_time_sec=gen_time,
            verification_time_sec=verify_time,
        )
    
    def run_baseline_eval(self) -> EvalResult:
        """Run baseline evaluation (before training)."""
        self.log("\n--- Baseline Evaluation ---")
        
        self.hw_monitor.start()
        
        samples, gen_time, tok_per_sec = self._generate_samples(self.prompts)
        samples, verify_time = self._verify_samples(samples)
        
        self.hw_monitor.stop()
        hw_summary = self.hw_monitor.summarize()
        
        # Save samples
        samples_path = self.output_dir / "baseline_samples.jsonl"
        with open(samples_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        
        # Save hardware metrics
        self.hw_monitor.save_csv(self.output_dir / "baseline_hardware.csv")
        
        result = self._calculate_eval_metrics(samples, gen_time, verify_time, tok_per_sec)
        result.hardware = hw_summary.to_dict()
        
        self.log(f"Baseline: {result.compile_rate:.1%} compile, {result.pass_at_1:.1%} pass@1")
        
        return result
    
    def run_raft_cycle(self, cycle: int, checkpoint: Optional[str] = None) -> CycleResult:
        """Run a single RAFT cycle."""
        self.log(f"\n--- RAFT Cycle {cycle} ---")
        
        cycle_start = time.time()
        self.hw_monitor.start()
        
        # Generate
        samples, _, _ = self._generate_samples(self.prompts)
        
        # Verify
        samples, _ = self._verify_samples(samples)
        
        # Filter (keep top 50% by reward)
        compiled_samples = [s for s in samples if s.get("compiles", False)]
        compiled_samples.sort(key=lambda x: x.get("reward", 0), reverse=True)
        kept = compiled_samples[:len(compiled_samples) // 2 + 1] if compiled_samples else []
        
        # Train if we have samples
        training_loss = 0.0
        training_steps = 0
        
        if kept:
            self.log(f"Training on {len(kept)} filtered samples...")
            training_loss, training_steps = self._train_on_samples(kept, cycle)
        
        self.hw_monitor.stop()
        hw_summary = self.hw_monitor.summarize()
        
        cycle_time = time.time() - cycle_start
        
        # Save cycle data
        cycle_dir = self.output_dir / f"cycle_{cycle}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        
        with open(cycle_dir / "samples.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        
        with open(cycle_dir / "kept.jsonl", "w") as f:
            for s in kept:
                f.write(json.dumps(s) + "\n")
        
        self.hw_monitor.save_csv(cycle_dir / "hardware.csv")
        
        compile_rate = len(compiled_samples) / len(samples) if samples else 0
        avg_reward = sum(s.get("reward", 0) for s in samples) / len(samples) if samples else 0
        
        result = CycleResult(
            cycle=cycle,
            generated=len(samples),
            verified=len(samples),
            kept=len(kept),
            compile_rate=compile_rate,
            avg_reward=avg_reward,
            training_loss=training_loss,
            training_steps=training_steps,
            cycle_time_sec=cycle_time,
            hardware=hw_summary.to_dict(),
        )
        
        self.log(f"Cycle {cycle}: {compile_rate:.1%} compile, kept {len(kept)}, loss={training_loss:.4f}")
        
        return result
    
    def _train_on_samples(self, samples: List[Dict], cycle: int) -> Tuple[float, int]:
        """Train model on filtered samples with production config."""
        import torch
        from datasets import Dataset
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        
        self.log(f"Training on {len(samples)} filtered samples...")
        
        # Set model to training mode
        self.model.train()
        
        # Prepare training data
        texts = []
        for s in samples:
            messages = [
                {"role": "system", "content": "You are an expert programmer."},
                {"role": "user", "content": s["prompt"]},
                {"role": "assistant", "content": s["code"]},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        
        dataset = Dataset.from_dict({'text': texts})
        
        # Tokenize
        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=2048
            )
        
        tokenized = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=['text']
        )
        
        # Production-matching training config
        output_path = self.output_dir / f"cycle_{cycle}" / "checkpoint"
        
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,  # Match production
            learning_rate=5e-5,              # Match production
            warmup_steps=10,
            logging_steps=5,
            save_strategy="no",
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch",
            report_to="none",
            dataloader_num_workers=0,        # Required for Strix Halo
            dataloader_pin_memory=False,     # Required for Strix Halo
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        result = trainer.train()
        
        # Clean up
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        return result.training_loss, result.global_step
    
    def run_final_eval(self) -> EvalResult:
        """Run final evaluation (after training)."""
        self.log("\n--- Final Evaluation ---")
        
        self.hw_monitor.start()
        
        samples, gen_time, tok_per_sec = self._generate_samples(self.prompts)
        samples, verify_time = self._verify_samples(samples)
        
        self.hw_monitor.stop()
        hw_summary = self.hw_monitor.summarize()
        
        # Save samples
        samples_path = self.output_dir / "final_samples.jsonl"
        with open(samples_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        
        self.hw_monitor.save_csv(self.output_dir / "final_hardware.csv")
        
        result = self._calculate_eval_metrics(samples, gen_time, verify_time, tok_per_sec)
        result.hardware = hw_summary.to_dict()
        
        self.log(f"Final: {result.compile_rate:.1%} compile, {result.pass_at_1:.1%} pass@1")
        
        return result
    
    def run(self) -> BenchmarkResult:
        """Run full benchmark."""
        total_start = time.time()
        
        self.log(f"\n{'='*60}")
        self.log(f"Benchmark: {self.model_name}")
        self.log(f"Cycles: {self.n_cycles}")
        self.log(f"Prompts: {len(self.prompts)}")
        self.log(f"Output: {self.output_dir}")
        self.log(f"{'='*60}")
        
        # Load model
        self._load_model()
        
        # Baseline evaluation
        self.result.baseline = self.run_baseline_eval()
        
        # RAFT cycles
        for cycle in range(1, self.n_cycles + 1):
            cycle_result = self.run_raft_cycle(cycle)
            self.result.cycles.append(cycle_result)
        
        # Final evaluation
        self.result.final = self.run_final_eval()
        
        # Total time
        self.result.total_time_sec = time.time() - total_start
        
        # Aggregate hardware summary
        self.result.hardware_summary = self._aggregate_hardware()
        
        # Save full results
        self.result.save(self.output_dir / "summary.json")
        
        # Print summary
        self._print_summary()
        
        return self.result
    
    def _aggregate_hardware(self) -> Dict[str, Any]:
        """Aggregate hardware metrics across all phases."""
        phases = []
        
        if self.result.baseline and self.result.baseline.hardware:
            phases.append(("baseline", self.result.baseline.hardware))
        
        for c in self.result.cycles:
            if c.hardware:
                phases.append((f"cycle_{c.cycle}", c.hardware))
        
        if self.result.final and self.result.final.hardware:
            phases.append(("final", self.result.final.hardware))
        
        if not phases:
            return {}
        
        # Calculate overall peaks
        gpu_peak_util = max(p[1]["gpu"]["utilization_peak_pct"] for p in phases)
        gpu_peak_mem = max(p[1]["gpu"]["memory_peak_gb"] for p in phases)
        sys_peak_mem = max(p[1]["system"]["memory_peak_gb"] for p in phases)
        total_energy = sum(p[1]["gpu"]["energy_wh"] for p in phases)
        
        return {
            "gpu_peak_utilization_pct": gpu_peak_util,
            "gpu_peak_memory_gb": gpu_peak_mem,
            "sys_peak_memory_gb": sys_peak_mem,
            "total_energy_wh": round(total_energy, 2),
            "phases": dict(phases),
        }
    
    def _print_summary(self):
        """Print benchmark summary."""
        self.log(f"\n{'='*60}")
        self.log("BENCHMARK SUMMARY")
        self.log(f"{'='*60}")
        self.log(f"Model: {self.model_name}")
        self.log(f"Total time: {self.result.total_time_sec / 60:.1f} minutes")
        self.log("")
        
        if self.result.baseline:
            self.log(f"Baseline compile rate: {self.result.baseline.compile_rate:.1%}")
            self.log(f"Baseline pass@1:       {self.result.baseline.pass_at_1:.1%}")
        
        if self.result.final:
            self.log(f"Final compile rate:    {self.result.final.compile_rate:.1%}")
            self.log(f"Final pass@1:          {self.result.final.pass_at_1:.1%}")
        
        if self.result.baseline and self.result.final:
            improvement = self.result.final.compile_rate - self.result.baseline.compile_rate
            self.log(f"\nImprovement: +{improvement:.1%}")
        
        if self.result.hardware_summary:
            hw = self.result.hardware_summary
            self.log(f"\nHardware:")
            self.log(f"  GPU peak utilization: {hw.get('gpu_peak_utilization_pct', 0):.0f}%")
            self.log(f"  GPU peak memory: {hw.get('gpu_peak_memory_gb', 0):.1f} GB")
            self.log(f"  Energy used: {hw.get('total_energy_wh', 0):.2f} Wh")
        
        self.log(f"{'='*60}\n")


def run_benchmark_suite(
    models: List[str],
    output_dir: str,
    n_cycles: int = 2,
    verbose: bool = True,
) -> List[BenchmarkResult]:
    """Run benchmarks on multiple models."""
    results = []
    output_path = Path(output_dir)
    
    for model in models:
        model_short = model.split("/")[-1].lower().replace(".", "-")
        model_output = output_path / model_short
        
        runner = BenchmarkRunner(
            model_name=model,
            output_dir=str(model_output),
            n_cycles=n_cycles,
            verbose=verbose,
        )
        
        result = runner.run()
        results.append(result)
        
        # Clean up
        import gc
        import torch
        del runner
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save comparison summary
    comparison = {
        "models": [r.to_dict() for r in results],
        "comparison": {
            "baseline_compile_rates": {r.model_short: r.baseline.compile_rate for r in results if r.baseline},
            "final_compile_rates": {r.model_short: r.final.compile_rate for r in results if r.final},
            "improvements": {
                r.model_short: r.final.compile_rate - r.baseline.compile_rate 
                for r in results if r.baseline and r.final
            },
        }
    }
    
    with open(output_path / "comparison_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    return results


# Default benchmark models
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-Coder-0.5B",
    "Qwen/Qwen2.5-Coder-1.5B",
    "Qwen/Qwen2.5-Coder-3B",
]

