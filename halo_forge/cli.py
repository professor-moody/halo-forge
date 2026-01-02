#!/usr/bin/env python3
"""
halo-forge CLI

Unified command-line interface for the halo-forge framework.

Usage:
    halo-forge data prepare --dataset codeforces_cpp --output data/train.jsonl
    halo-forge data generate --topic rust_async --backend deepseek --output data/rust.jsonl
    halo-forge sft train --config configs/sft.yaml
    halo-forge raft train --config configs/raft.yaml
    halo-forge benchmark run --model models/raft/cycle_3 --prompts data/test.jsonl
    halo-forge test --level standard  # Validate pipeline
    halo-forge info  # Show hardware info
"""

import argparse
import sys
import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional


def cmd_data_prepare(args):
    """Prepare dataset from public sources."""
    from halo_forge.data.public_datasets import (
        DatasetPreparer,
        get_dataset_spec,
        list_datasets
    )
    
    if args.list:
        print("Available datasets:")
        for name in list_datasets():
            print(f"  - {name}")
        return
    
    if not args.dataset:
        print("Error: --dataset required")
        print("Use --list to see available datasets")
        sys.exit(1)
    
    spec = get_dataset_spec(args.dataset)
    preparer = DatasetPreparer(spec, system_prompt=args.system_prompt)
    
    output = args.output or f"data/{args.dataset}.jsonl"
    preparer.prepare(output, template=args.template)


def cmd_data_generate(args):
    """Generate data with LLM."""
    from halo_forge.data.llm_generate import (
        TrainingDataGenerator,
        get_backend,
        get_topic_spec,
        list_topics
    )
    
    if args.list:
        print("Available topics:")
        for name in list_topics():
            print(f"  - {name}")
        return
    
    if not args.topic:
        print("Error: --topic required")
        print("Use --list to see available topics")
        sys.exit(1)
    
    spec = get_topic_spec(args.topic)
    backend = get_backend(args.backend, model=args.model)
    generator = TrainingDataGenerator(backend, spec)
    
    output = args.output or f"data/{args.topic}_generated.jsonl"
    generator.generate_all(output, template=args.template)


def cmd_sft_train(args):
    """Run SFT training."""
    from halo_forge.sft.trainer import SFTTrainer, SFTConfig
    
    if args.config:
        config = SFTConfig.from_yaml(args.config)
    else:
        config = SFTConfig(
            train_file=args.data,
            output_dir=args.output,
            num_epochs=args.epochs
        )
    
    trainer = SFTTrainer(config)
    trainer.train(resume_from_checkpoint=args.resume)


def cmd_raft_train(args):
    """Run RAFT training."""
    import yaml
    from halo_forge.rlvr.raft_trainer import RAFTTrainer, RAFTConfig
    from halo_forge.rlvr.verifiers import GCCVerifier, MinGWVerifier, RemoteMSVCVerifier
    
    # Load config
    if args.config:
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
    else:
        cfg_dict = {}
    
    # Setup verifier
    verifier_type = args.verifier or cfg_dict.get('verifier', {}).get('type', 'gcc')
    
    if verifier_type == 'gcc':
        verifier = GCCVerifier()
    elif verifier_type == 'mingw':
        verifier = MinGWVerifier()
    elif verifier_type == 'msvc':
        verifier = RemoteMSVCVerifier(
            host=cfg_dict.get('verifier', {}).get('host', 'localhost'),
            user=cfg_dict.get('verifier', {}).get('user', 'user'),
            ssh_key=cfg_dict.get('verifier', {}).get('ssh_key', '~/.ssh/id_rsa')
        )
    else:
        print(f"Unknown verifier: {verifier_type}")
        sys.exit(1)
    
    # Create config
    config = RAFTConfig(
        sft_checkpoint=args.checkpoint or cfg_dict.get('sft_checkpoint', 'models/sft/final_model'),
        output_dir=args.output or cfg_dict.get('output_dir', 'models/raft'),
        num_cycles=args.cycles or cfg_dict.get('num_cycles', 3)
    )
    
    # Load prompts
    prompts = []
    prompts_file = args.prompts or cfg_dict.get('prompts')
    if prompts_file:
        with open(prompts_file) as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data.get('prompt', data.get('text', '')))
    
    if not prompts:
        print("Error: No prompts provided")
        print("Use --prompts or set in config")
        sys.exit(1)
    
    # Run
    trainer = RAFTTrainer(verifier=verifier, config=config)
    trainer.run(prompts, num_cycles=config.num_cycles)


def cmd_benchmark(args):
    """Run benchmark."""
    from halo_forge.benchmark.pass_at_k import Benchmark
    from halo_forge.rlvr.verifiers import GCCVerifier, MinGWVerifier, RemoteMSVCVerifier
    
    # Setup verifier
    if args.verifier == 'gcc':
        verifier = GCCVerifier()
    elif args.verifier == 'mingw':
        verifier = MinGWVerifier()
    elif args.verifier == 'msvc':
        verifier = RemoteMSVCVerifier(
            host=args.host,
            user=args.user,
            ssh_key=args.ssh_key
        )
    else:
        print(f"Unknown verifier: {args.verifier}")
        sys.exit(1)
    
    # Create benchmark
    benchmark = Benchmark(
        model_path=args.model,
        verifier=verifier,
        base_model=args.base_model,
        system_prompt=args.system_prompt
    )
    
    # Parse k values
    k_values = [int(k) for k in args.k.split(',')]
    
    # Run
    result = benchmark.run(
        prompts=args.prompts,
        samples_per_prompt=args.samples,
        k_values=k_values,
        max_prompts=args.max_prompts,
        output_path=args.output
    )


def cmd_benchmark_full(args):
    """Run comprehensive RAFT benchmark with hardware monitoring."""
    try:
        from halo_forge import ui
        use_rich = True
    except ImportError:
        use_rich = False
    
    from halo_forge.benchmark import BenchmarkRunner, run_benchmark_suite, DEFAULT_MODELS
    
    if use_rich:
        ui.print_banner()
        ui.print_header("RAFT Benchmark", f"Comprehensive training benchmark with metrics")
    
    # Handle --suite option
    if args.suite:
        if args.suite == "all":
            models = DEFAULT_MODELS
        elif args.suite == "small":
            models = [DEFAULT_MODELS[0]]  # Just 0.5B
        elif args.suite == "medium":
            models = DEFAULT_MODELS[:2]  # 0.5B and 1.5B
        else:
            print(f"Unknown suite: {args.suite}")
            print("Valid suites: all, small, medium")
            sys.exit(1)
        
        results = run_benchmark_suite(
            models=models,
            output_dir=args.output,
            n_cycles=args.cycles,
            verbose=not args.quiet,
        )
        
        # Print comparison
        if use_rich:
            ui.print_header("Results Summary")
        print(f"\nBenchmark complete. Results saved to: {args.output}")
        
        for r in results:
            improvement = (r.final.compile_rate - r.baseline.compile_rate) if r.final and r.baseline else 0
            print(f"  {r.model_short}: {r.baseline.compile_rate:.1%} -> {r.final.compile_rate:.1%} (+{improvement:.1%})")
        
    else:
        # Single model benchmark
        runner = BenchmarkRunner(
            model_name=args.model,
            output_dir=args.output,
            n_cycles=args.cycles,
            verbose=not args.quiet,
        )
        
        result = runner.run()
        print(f"\nBenchmark complete. Results saved to: {args.output}/summary.json")


def cmd_info(args):
    """Show hardware info."""
    try:
        from halo_forge import ui
        import torch
        
        ui.print_banner()
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / 1e9
            
            # Try to get versions
            rocm_version = ""
            if hasattr(torch.version, 'hip'):
                rocm_version = torch.version.hip or ""
            
            pytorch_version = torch.__version__
            
            ui.print_hardware_info(
                gpu_name=gpu_name,
                memory_gb=memory_gb,
                rocm_version=rocm_version,
                pytorch_version=pytorch_version
            )
        else:
            ui.print_warning("No GPU detected")
            ui.print_info("PyTorch CUDA/ROCm not available")
    except ImportError:
        # Fallback if rich not installed
        from halo_forge.utils.hardware import print_hardware_info
        print_hardware_info()


# =============================================================================
# Test Command
# =============================================================================

# Built-in test prompts for pipeline validation
TEST_PROMPTS = [
    {
        "prompt": "Write a C++ program that prints 'Hello, World!' to stdout.",
        "expected_output": "Hello, World!"
    },
    {
        "prompt": "Write a C++ function that returns the sum of two integers a and b, then call it in main to print the result of 5 + 3.",
        "expected_output": "8"
    },
    {
        "prompt": "Write a C++ program that prints the numbers 1 through 5, each on a new line.",
        "expected_output": "1\n2\n3\n4\n5"
    },
]


class TestRunner:
    """Pipeline test runner with multiple test levels."""
    
    def __init__(self, verbose: bool = False, model: str = "Qwen/Qwen2.5-Coder-0.5B"):
        self.verbose = verbose
        self.model_name = model
        self.results = {"passed": [], "failed": [], "skipped": []}
        
        # Try to use rich UI
        try:
            from halo_forge import ui
            self.ui = ui
            self.use_rich = True
        except ImportError:
            self.ui = None
            self.use_rich = False
    
    def log(self, msg: str, level: str = "info"):
        """Log message if verbose or if it's an error."""
        if self.verbose or level in ("error", "result"):
            if self.use_rich:
                if level == "ok":
                    self.ui.print_step(msg, "success")
                elif level == "fail":
                    self.ui.print_step(msg, "error")
                elif level == "skip":
                    self.ui.print_step(msg, "skip")
                elif level == "error":
                    self.ui.print_error(msg)
                else:
                    self.ui.print_dim(f"  {msg}")
            else:
                prefix = {"info": "  ", "ok": "  [OK] ", "fail": "  [FAIL] ", "skip": "  [SKIP] ", "error": "  [ERROR] ", "result": ""}
                print(f"{prefix.get(level, '  ')}{msg}")
    
    def run_test(self, name: str, test_fn, skip_condition: bool = False, skip_reason: str = ""):
        """Run a single test with timing."""
        if skip_condition:
            self.results["skipped"].append(name)
            if self.use_rich:
                self.ui.print_step(name, "skip", skip_reason)
            else:
                self.log(f"{name}: {skip_reason}", "skip")
            return None
        
        start = time.time()
        try:
            result = test_fn()
            elapsed = time.time() - start
            self.results["passed"].append(name)
            if self.use_rich:
                self.ui.print_step(name, "success", time_s=elapsed)
            else:
                self.log(f"{name} ({elapsed:.1f}s)", "ok")
            return result
        except Exception as e:
            elapsed = time.time() - start
            self.results["failed"].append(name)
            if self.use_rich:
                self.ui.print_step(name, "error", str(e), time_s=elapsed)
            else:
                self.log(f"{name} ({elapsed:.1f}s): {e}", "fail")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def print_summary(self):
        """Print test summary."""
        if self.use_rich:
            self.ui.print_test_results(self.results)
            return len(self.results["failed"]) == 0
        
        # Fallback plain output
        total = len(self.results["passed"]) + len(self.results["failed"]) + len(self.results["skipped"])
        passed = len(self.results["passed"])
        failed = len(self.results["failed"])
        skipped = len(self.results["skipped"])
        
        print(f"\n{'='*60}")
        print(f"Test Results: {passed}/{total} passed", end="")
        if skipped:
            print(f", {skipped} skipped", end="")
        if failed:
            print(f", {failed} FAILED", end="")
        print()
        
        if failed:
            print(f"\nFailed tests:")
            for name in self.results["failed"]:
                print(f"  - {name}")
        
        print(f"{'='*60}")
        
        return failed == 0
    
    # =========================================================================
    # Smoke Tests (no GPU required)
    # =========================================================================
    
    def test_imports(self) -> bool:
        """Test that all modules import correctly."""
        # Core modules
        from halo_forge.rlvr.verifiers import GCCVerifier, VerifyResult, RewardLevel
        from halo_forge.rlvr.raft_trainer import RAFTTrainer
        from halo_forge.sft.trainer import SFTTrainer
        from halo_forge.utils.hardware import print_hardware_info
        return True
    
    def test_compiler_available(self) -> bool:
        """Test that g++ is available."""
        if not shutil.which("g++"):
            raise RuntimeError("g++ not found in PATH")
        return True
    
    def test_verifier_basic(self) -> bool:
        """Test verifier with known good/bad code."""
        from halo_forge.rlvr.verifiers import GCCVerifier
        
        verifier = GCCVerifier()
        
        # Test valid code
        valid = '#include <iostream>\nint main() { std::cout << "test"; return 0; }'
        result = verifier.verify(valid)
        if result.reward == 0.0:
            raise RuntimeError(f"Valid code got reward 0: {result.details}")
        
        # Test invalid code
        invalid = 'this is not valid C++ code at all'
        result = verifier.verify(invalid)
        if result.reward > 0.0:
            raise RuntimeError("Invalid code got positive reward")
        
        return True
    
    # =========================================================================
    # Standard Tests (GPU required)
    # =========================================================================
    
    def test_gpu_available(self) -> bool:
        """Test GPU availability."""
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available (torch.cuda.is_available() = False)")
        
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / 1e9
        
        self.log(f"GPU: {device_name}, Memory: {mem_gb:.1f} GB")
        return True
    
    def test_model_load(self) -> Any:
        """Test model loading."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.log(f"Loading {self.model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        self.log(f"Loaded: {model.num_parameters() / 1e6:.1f}M parameters")
        
        return model, tokenizer
    
    def test_generation(self, model, tokenizer) -> List[Dict]:
        """Test code generation."""
        import torch
        
        results = []
        
        for i, item in enumerate(TEST_PROMPTS):
            prompt = item["prompt"]
            
            messages = [
                {"role": "system", "content": "You are a helpful coding assistant. Write clean, working C++ code."},
                {"role": "user", "content": prompt}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            self.log(f"Prompt {i+1}: {prompt[:40]}...")
            self.log(f"Generated: {generated[:60]}...")
            
            results.append({
                "prompt": prompt,
                "generated": generated,
                "expected_output": item.get("expected_output"),
            })
        
        return results
    
    def test_verification(self, samples: List[Dict]) -> List[Dict]:
        """Test verification of generated samples."""
        from halo_forge.rlvr.verifiers import GCCVerifier
        
        # Create verifier with run_after_compile to test execution
        verifier = GCCVerifier(run_after_compile=True, timeout=30, run_timeout=5)
        
        verified = []
        for i, sample in enumerate(samples):
            result = verifier.verify(sample["generated"])
            
            status = "PASS" if result.success else "FAIL"
            self.log(f"Sample {i+1}: {status} (reward={result.reward:.2f})")
            
            verified.append({
                **sample,
                "success": result.success,
                "reward": result.reward,
                "details": result.details,
            })
        
        passed = sum(1 for v in verified if v["success"])
        avg_reward = sum(v["reward"] for v in verified) / len(verified) if verified else 0
        
        self.log(f"Verification: {passed}/{len(verified)} passed, avg_reward={avg_reward:.2f}")
        
        return verified
    
    # =========================================================================
    # Full Tests (includes training)
    # =========================================================================
    
    def test_training_step(self, model, tokenizer, verified_samples: List[Dict]) -> bool:
        """Test a minimal SFT training step."""
        from transformers import TrainingArguments
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset
        
        # Prepare data - keep samples with any reward
        kept = [s for s in verified_samples if s["reward"] > 0]
        if not kept:
            self.log("No samples passed verification, using all for test")
            kept = verified_samples
        
        # Format for SFT
        training_data = []
        for sample in kept:
            training_data.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful coding assistant."},
                    {"role": "user", "content": sample["prompt"]},
                    {"role": "assistant", "content": sample["generated"]},
                ]
            })
        
        dataset = Dataset.from_list(training_data)
        
        self.log(f"Training on {len(dataset)} samples...")
        
        # Minimal training config
        with tempfile.TemporaryDirectory(prefix="halo_forge_test_") as tmp_dir:
            training_args = SFTConfig(
                output_dir=tmp_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=2e-5,
                logging_steps=1,
                save_steps=9999,
                max_steps=2,  # Just 2 steps
                bf16=True,
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
                report_to="none",
            )
            
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
            )
            
            result = trainer.train()
            
            self.log(f"Training: {result.global_step} steps, loss={result.training_loss:.4f}")
        
        return True
    
    # =========================================================================
    # Test Level Runners
    # =========================================================================
    
    def run_smoke(self) -> bool:
        """Run smoke tests (no GPU required)."""
        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("Smoke Test", "Quick validation without GPU")
        else:
            print(f"\n{'='*60}")
            print("halo-forge Smoke Test")
            print(f"{'='*60}\n")
        
        self.run_test("Import modules", self.test_imports)
        self.run_test("Compiler available", self.test_compiler_available)
        self.run_test("Verifier basic", self.test_verifier_basic)
        
        return self.print_summary()
    
    def run_standard(self) -> bool:
        """Run standard tests (GPU required)."""
        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("Standard Test", f"Model: {self.model_name}")
        else:
            print(f"\n{'='*60}")
            print("halo-forge Standard Test")
            print(f"Model: {self.model_name}")
            print(f"{'='*60}\n")
        
        # Smoke tests first
        self.run_test("Import modules", self.test_imports)
        self.run_test("Compiler available", self.test_compiler_available)
        
        # GPU tests
        gpu_ok = self.run_test("GPU available", self.test_gpu_available)
        if gpu_ok is None:
            if self.use_rich:
                self.ui.print_error("Cannot continue without GPU")
            else:
                print("\nCannot continue without GPU")
            return self.print_summary()
        
        # Model loading
        result = self.run_test("Model loading", self.test_model_load)
        if result is None:
            return self.print_summary()
        model, tokenizer = result
        
        # Generation
        samples = self.run_test("Code generation", lambda: self.test_generation(model, tokenizer))
        if samples is None:
            return self.print_summary()
        
        # Verification
        self.run_test("Code verification", lambda: self.test_verification(samples))
        
        return self.print_summary()
    
    def run_full(self) -> bool:
        """Run full tests including training."""
        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("Full Pipeline Test", f"Model: {self.model_name}")
        else:
            print(f"\n{'='*60}")
            print("halo-forge Full Pipeline Test")
            print(f"Model: {self.model_name}")
            print(f"{'='*60}\n")
        
        # Smoke tests
        self.run_test("Import modules", self.test_imports)
        self.run_test("Compiler available", self.test_compiler_available)
        
        # GPU tests
        gpu_ok = self.run_test("GPU available", self.test_gpu_available)
        if gpu_ok is None:
            if self.use_rich:
                self.ui.print_error("Cannot continue without GPU")
            else:
                print("\nCannot continue without GPU")
            return self.print_summary()
        
        # Model loading
        result = self.run_test("Model loading", self.test_model_load)
        if result is None:
            return self.print_summary()
        model, tokenizer = result
        
        # Generation
        samples = self.run_test("Code generation", lambda: self.test_generation(model, tokenizer))
        if samples is None:
            return self.print_summary()
        
        # Verification
        verified = self.run_test("Code verification", lambda: self.test_verification(samples))
        if verified is None:
            verified = samples  # Use unverified for training test
        
        # Training step
        self.run_test("Training step", lambda: self.test_training_step(model, tokenizer, verified))
        
        return self.print_summary()


def cmd_test(args):
    """Run pipeline validation tests."""
    runner = TestRunner(verbose=args.verbose, model=args.model)
    
    if args.level == "smoke":
        success = runner.run_smoke()
    elif args.level == "standard":
        success = runner.run_standard()
    elif args.level == "full":
        success = runner.run_full()
    else:
        print(f"Unknown test level: {args.level}")
        print("Valid levels: smoke, standard, full")
        sys.exit(1)
    
    sys.exit(0 if success else 1)


def main():
    parser = argparse.ArgumentParser(
        prog='halo-forge',
        description='Complete RLVR training framework for AMD Strix Halo'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # data command
    data_parser = subparsers.add_parser('data', help='Data preparation')
    data_subparsers = data_parser.add_subparsers(dest='data_command', required=True)
    
    # data prepare
    prepare_parser = data_subparsers.add_parser('prepare', help='Prepare public dataset')
    prepare_parser.add_argument('--dataset', '-d', help='Dataset name')
    prepare_parser.add_argument('--output', '-o', help='Output file path')
    prepare_parser.add_argument('--template', default='qwen', help='Chat template')
    prepare_parser.add_argument('--system-prompt', help='Override system prompt')
    prepare_parser.add_argument('--list', action='store_true', help='List available datasets')
    
    # data generate
    generate_parser = data_subparsers.add_parser('generate', help='Generate with LLM')
    generate_parser.add_argument('--topic', '-t', help='Topic name')
    generate_parser.add_argument('--backend', '-b', default='deepseek', help='LLM backend')
    generate_parser.add_argument('--model', help='Model name for backend')
    generate_parser.add_argument('--output', '-o', help='Output file path')
    generate_parser.add_argument('--template', default='qwen', help='Chat template')
    generate_parser.add_argument('--list', action='store_true', help='List available topics')
    
    # sft command
    sft_parser = subparsers.add_parser('sft', help='SFT training')
    sft_subparsers = sft_parser.add_subparsers(dest='sft_command', required=True)
    
    # sft train
    sft_train_parser = sft_subparsers.add_parser('train', help='Run SFT training')
    sft_train_parser.add_argument('--config', '-c', help='Config file path')
    sft_train_parser.add_argument('--data', help='Training data file')
    sft_train_parser.add_argument('--output', '-o', default='models/sft', help='Output directory')
    sft_train_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    sft_train_parser.add_argument('--resume', help='Resume from checkpoint')
    
    # raft command
    raft_parser = subparsers.add_parser('raft', help='RAFT training')
    raft_subparsers = raft_parser.add_subparsers(dest='raft_command', required=True)
    
    # raft train
    raft_train_parser = raft_subparsers.add_parser('train', help='Run RAFT training')
    raft_train_parser.add_argument('--config', '-c', help='Config file path')
    raft_train_parser.add_argument('--checkpoint', help='SFT checkpoint path')
    raft_train_parser.add_argument('--prompts', '-p', help='Prompts file')
    raft_train_parser.add_argument('--output', '-o', default='models/raft', help='Output directory')
    raft_train_parser.add_argument('--cycles', type=int, help='Number of RAFT cycles')
    raft_train_parser.add_argument('--verifier', default='gcc', help='Verifier type')
    
    # benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Benchmarking')
    bench_subparsers = bench_parser.add_subparsers(dest='bench_command', required=True)
    
    # benchmark run (legacy pass@k benchmark)
    bench_run_parser = bench_subparsers.add_parser('run', help='Run pass@k benchmark')
    bench_run_parser.add_argument('--model', '-m', required=True, help='Model path')
    bench_run_parser.add_argument('--prompts', '-p', required=True, help='Prompts file')
    bench_run_parser.add_argument('--output', '-o', help='Output file path')
    bench_run_parser.add_argument('--samples', type=int, default=10, help='Samples per prompt')
    bench_run_parser.add_argument('--k', default='1,5,10', help='k values (comma-separated)')
    bench_run_parser.add_argument('--max-prompts', type=int, help='Max prompts to evaluate')
    bench_run_parser.add_argument('--verifier', default='gcc', help='Verifier type')
    bench_run_parser.add_argument('--base-model', default='Qwen/Qwen2.5-Coder-7B', help='Base model')
    bench_run_parser.add_argument('--system-prompt', default='You are an expert programmer.', help='System prompt')
    bench_run_parser.add_argument('--host', help='MSVC host')
    bench_run_parser.add_argument('--user', help='MSVC user')
    bench_run_parser.add_argument('--ssh-key', help='MSVC SSH key')
    
    # benchmark full (comprehensive RAFT benchmark with hardware metrics)
    bench_full_parser = bench_subparsers.add_parser('full', help='Run comprehensive RAFT benchmark')
    bench_full_parser.add_argument('--model', '-m', help='Model to benchmark (e.g., Qwen/Qwen2.5-Coder-0.5B)')
    bench_full_parser.add_argument('--suite', '-s', choices=['all', 'small', 'medium'],
                                   help='Run predefined suite: all (0.5B, 1.5B, 3B), small (0.5B), medium (0.5B, 1.5B)')
    bench_full_parser.add_argument('--cycles', '-c', type=int, default=2, help='Number of RAFT cycles (default: 2)')
    bench_full_parser.add_argument('--output', '-o', default='results/benchmarks', help='Output directory')
    bench_full_parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    # info command
    info_parser = subparsers.add_parser('info', help='Show hardware info')
    
    # test command
    test_parser = subparsers.add_parser('test', help='Run pipeline validation tests')
    test_parser.add_argument('--level', '-l', default='standard',
                             choices=['smoke', 'standard', 'full'],
                             help='Test level: smoke (no GPU), standard (with GPU), full (with training)')
    test_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-Coder-0.5B',
                             help='Model to use for testing (default: Qwen2.5-Coder-0.5B)')
    test_parser.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output with detailed logging')
    
    # Parse
    args = parser.parse_args()
    
    # Route to handler
    if args.command == 'data':
        if args.data_command == 'prepare':
            cmd_data_prepare(args)
        elif args.data_command == 'generate':
            cmd_data_generate(args)
    elif args.command == 'sft':
        if args.sft_command == 'train':
            cmd_sft_train(args)
    elif args.command == 'raft':
        if args.raft_command == 'train':
            cmd_raft_train(args)
    elif args.command == 'benchmark':
        if args.bench_command == 'run':
            cmd_benchmark(args)
        elif args.bench_command == 'full':
            if not args.model and not args.suite:
                print("Error: Either --model or --suite is required")
                print("Examples:")
                print("  halo-forge benchmark full --model Qwen/Qwen2.5-Coder-0.5B")
                print("  halo-forge benchmark full --suite all")
                sys.exit(1)
            cmd_benchmark_full(args)
    elif args.command == 'info':
        cmd_info(args)
    elif args.command == 'test':
        cmd_test(args)


if __name__ == '__main__':
    main()

