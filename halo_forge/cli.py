#!/usr/bin/env python3
"""
halo-forge CLI

Unified command-line interface for the halo forge framework.

Usage:
    halo-forge data prepare --dataset codeforces_cpp --output data/train.jsonl
    halo-forge data generate --topic rust_async --backend deepseek --output data/rust.jsonl
    halo-forge sft train --model Qwen/Qwen2.5-Coder-0.5B --data data/train.jsonl
    halo-forge raft train --model Qwen/Qwen2.5-Coder-0.5B --prompts data/prompts.jsonl
    halo-forge benchmark run --model models/raft/cycle_3 --prompts data/test.jsonl
    halo-forge test --level standard  # Validate pipeline
    halo-forge info  # Show hardware info
"""

# Pre-parse for --experimental-attention BEFORE any torch imports
# This must happen before any imports that could trigger torch loading
import sys
import os
if '--experimental-attention' in sys.argv:
    os.environ['TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL'] = '1'

import argparse
import json
import time
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

# ANSI color codes
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
BOLD = "\033[1m"
NC = "\033[0m"  # No Color


def print_banner():
    """Print the halo forge banner."""
    print(f"""
{CYAN}╔═══════════════════════════════════════════════════════════════╗
║                      HALO-FORGE                               ║
║              RAFT Training for AMD Strix Halo                 ║
╚═══════════════════════════════════════════════════════════════╝{NC}
""")


def cmd_data_validate(args):
    """Validate dataset format."""
    from halo_forge.data.validator import validate_dataset
    
    result = validate_dataset(args.file, preview=args.preview)
    
    if not result.valid:
        sys.exit(1)


def cmd_config_validate(args):
    """Validate training config file."""
    import yaml
    from pathlib import Path
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    errors = []
    warnings = []
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML syntax: {e}")
        sys.exit(1)
    
    print(f"Validating config: {config_path}")
    print("=" * 50)
    
    # Required fields based on config type
    config_type = args.type
    if not config_type:
        if 'raft' in str(config_path).lower():
            config_type = 'raft'
        elif 'sft' in str(config_path).lower():
            config_type = 'sft'
        else:
            config_type = 'auto'
    
    if config_type == 'raft':
        required = ['base_model', 'output_dir']
    elif config_type == 'sft':
        required = ['model', 'data_path', 'output_dir']
    else:
        required = []
    
    # Check required fields
    for field in required:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate specific fields
    if 'learning_rate' in config:
        lr = config['learning_rate']
        if not isinstance(lr, (int, float)) or lr <= 0:
            errors.append(f"Invalid learning_rate: {lr} (must be positive number)")
        elif lr > 1e-3:
            warnings.append(f"learning_rate={lr} seems high (typical: 1e-5 to 5e-5)")
    
    if 'lr_decay_per_cycle' in config:
        decay = config['lr_decay_per_cycle']
        if not 0 < decay <= 1:
            errors.append(f"Invalid lr_decay_per_cycle: {decay} (must be 0 < x <= 1)")
    
    if 'num_cycles' in config:
        cycles = config['num_cycles']
        if not isinstance(cycles, int) or cycles < 1:
            errors.append(f"Invalid num_cycles: {cycles} (must be positive integer)")
        elif cycles > 10:
            warnings.append(f"num_cycles={cycles} is high (typical: 3-6)")
    
    if 'temperature' in config:
        temp = config['temperature']
        if not 0 < temp <= 2:
            errors.append(f"Invalid temperature: {temp} (must be 0 < x <= 2)")
    
    if 'reward_threshold' in config:
        threshold = config['reward_threshold']
        if not 0 <= threshold <= 1:
            errors.append(f"Invalid reward_threshold: {threshold} (must be 0 <= x <= 1)")
    
    # Print results
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  ✗ {e}")
    
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ⚠ {w}")
    
    if not errors and not warnings:
        print("✓ Config is valid")
    elif not errors:
        print(f"\n✓ Config is valid ({len(warnings)} warnings)")
    
    # Print config summary
    if args.verbose:
        print("\nConfig contents:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    if errors:
        sys.exit(1)


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
    
    print_banner()
    print(f"{GREEN}SFT Training{NC}")
    print("=" * 60)
    
    # Require either --dataset or --data
    dataset = getattr(args, 'dataset', None)
    data = getattr(args, 'data', None)
    max_samples = getattr(args, 'max_samples', None)
    dry_run = getattr(args, 'dry_run', False)
    
    if not dataset and not data:
        print(f"{RED}Error: Either --dataset or --data is required{NC}")
        print()
        print("Examples:")
        print("  halo-forge sft train --dataset codealpaca --model Qwen/Qwen2.5-Coder-3B")
        print("  halo-forge sft train --data my_data.jsonl --model Qwen/Qwen2.5-Coder-3B")
        print()
        print("Available datasets:")
        print("  codealpaca, metamath, gsm8k_sft, llava, xlam_sft, glaive_sft")
        print("  Run 'halo-forge sft datasets' to see all options")
        sys.exit(1)
    
    if args.config:
        config = SFTConfig.from_yaml(args.config)
        # CLI args override config file
        if args.model:
            config.model_name = args.model
        if dataset:
            config.dataset = dataset
        if data:
            config.train_file = data
        if max_samples:
            config.max_samples = max_samples
        if args.output:
            config.output_dir = args.output
        if args.epochs:
            config.num_epochs = args.epochs
    else:
        config = SFTConfig(
            model_name=args.model or "Qwen/Qwen2.5-Coder-7B",
            dataset=dataset,
            train_file=data,
            max_samples=max_samples,
            output_dir=args.output,
            num_epochs=args.epochs
        )
    
    print(f"Model: {config.model_name}")
    if config.dataset:
        print(f"Dataset: {config.dataset}")
    elif config.train_file:
        print(f"Data file: {config.train_file}")
    if config.max_samples:
        print(f"Max samples: {config.max_samples}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_epochs}")
    print()
    
    if dry_run:
        print(f"{YELLOW}Dry run mode - validating configuration only{NC}")
        print()
        # Validate dataset exists
        if config.dataset:
            from halo_forge.sft.datasets import get_sft_dataset_spec, is_huggingface_id
            spec = get_sft_dataset_spec(config.dataset)
            if spec:
                print(f"{GREEN}✓{NC} Dataset: {spec.name} ({spec.huggingface_id})")
            elif is_huggingface_id(config.dataset):
                print(f"{GREEN}✓{NC} HuggingFace dataset: {config.dataset}")
            else:
                print(f"{RED}✗{NC} Unknown dataset: {config.dataset}")
                sys.exit(1)
        print(f"{GREEN}Configuration valid!{NC}")
        return
    
    trainer = SFTTrainer(config)
    trainer.train(resume_from_checkpoint=args.resume)


def cmd_sft_datasets(args):
    """List available SFT datasets."""
    from halo_forge.sft.datasets import list_sft_datasets
    
    print_banner()
    print(f"{GREEN}Available SFT Datasets{NC}")
    print("=" * 60)
    print()
    
    # Group by domain
    domains = ["code", "reasoning", "vlm", "audio", "agentic"]
    
    for domain in domains:
        datasets = list_sft_datasets(domain)
        if datasets:
            print(f"{YELLOW}{domain.upper()}{NC}")
            for ds in datasets:
                print(f"  {CYAN}{ds.name:<20}{NC} [{ds.size_hint:>6}] {ds.description}")
                print(f"                         HuggingFace: {ds.huggingface_id}")
            print()
    
    print("Usage:")
    print("  halo-forge sft train --dataset codealpaca --model Qwen/Qwen2.5-Coder-3B")
    print("  halo-forge sft train --dataset metamath --model Qwen/Qwen2.5-3B-Instruct")
    print()


def _resolve_model_path(model_path: str) -> tuple:
    """
    Resolve a model path that may be a base model ID or SFT output directory.
    
    Handles three cases:
    1. HuggingFace model ID (e.g., "Qwen/Qwen2.5-Coder-3B") - returns as-is
    2. SFT output directory with final_model/ subdirectory - auto-detects
    3. Direct LoRA adapter directory - reads base_model from adapter_config
    
    Returns:
        tuple: (base_model, sft_checkpoint) where base_model is the HuggingFace ID
               and sft_checkpoint is the path to the LoRA adapters (or None if fresh)
    """
    from pathlib import Path
    
    model_path_obj = Path(model_path)
    
    # Case 1: Not a local path, assume it's a HuggingFace model ID
    if not model_path_obj.exists():
        return (model_path, None)
    
    # Check for final_model subdirectory (SFT output pattern)
    final_model_path = model_path_obj / "final_model"
    if final_model_path.exists() and (final_model_path / "adapter_config.json").exists():
        checkpoint_path = final_model_path
    elif (model_path_obj / "adapter_config.json").exists():
        checkpoint_path = model_path_obj
    else:
        # It's a local path but not a LoRA adapter - might be a merged model
        return (model_path, None)
    
    # Read base model from adapter config
    adapter_config_path = checkpoint_path / "adapter_config.json"
    try:
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model = adapter_config.get("base_model_name_or_path")
        if base_model:
            return (base_model, str(checkpoint_path))
    except (json.JSONDecodeError, IOError):
        pass
    
    # Fallback: couldn't read config
    return (model_path, None)


def cmd_raft_train(args):
    """Run RAFT training."""
    # Note: --experimental-attention is handled at script startup (before imports)
    
    print_banner()
    print(f"{GREEN}RAFT Training{NC}")
    print("=" * 60)
    
    import yaml
    from halo_forge.rlvr.raft_trainer import RAFTTrainer, RAFTConfig
    from halo_forge.rlvr.verifiers import (
        GCCVerifier, MinGWVerifier, RemoteMSVCVerifier,
        HumanEvalVerifier, MBPPVerifier
    )
    
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
        # CLI args take precedence over config file
        msvc_host = getattr(args, 'host', None) or cfg_dict.get('verifier', {}).get('host')
        msvc_user = getattr(args, 'user', None) or cfg_dict.get('verifier', {}).get('user')
        msvc_key = getattr(args, 'ssh_key', None) or cfg_dict.get('verifier', {}).get('ssh_key')
        
        if not msvc_host or not msvc_user or not msvc_key:
            print("Error: MSVC verifier requires --host, --user, and --ssh-key.")
            print("\nExample:")
            print("  halo-forge raft train --verifier msvc \\")
            print("    --host 10.0.0.152 --user keys --ssh-key ~/.ssh/win \\")
            print("    --prompts data/prompts.jsonl")
            print("\nOr in config file (configs/raft_windows_msvc.yaml):")
            print("  verifier:")
            print("    type: msvc")
            print("    host: 10.0.0.152")
            print("    user: keys")
            print("    ssh_key: ~/.ssh/win")
            print("\nOr use MinGW for local cross-compilation (no Windows needed):")
            print("  halo-forge raft train --verifier mingw ...")
            sys.exit(1)
        
        verifier = RemoteMSVCVerifier(
            host=msvc_host,
            user=msvc_user,
            ssh_key=msvc_key
        )
    elif verifier_type == 'humaneval':
        dataset_path = cfg_dict.get('verifier', {}).get('dataset', 'data/rlvr/humaneval_full.jsonl')
        verifier = HumanEvalVerifier(dataset_path)
    elif verifier_type == 'mbpp':
        dataset_path = cfg_dict.get('verifier', {}).get('dataset', 'data/rlvr/mbpp_train_full.jsonl')
        verifier = MBPPVerifier(dataset_path)
    elif verifier_type == 'rust' or verifier_type == 'cargo':
        from halo_forge.rlvr.verifiers import RustVerifier
        run_after = cfg_dict.get('verifier', {}).get('run_after_compile', False)
        verifier = RustVerifier(run_after_compile=run_after)
    elif verifier_type == 'go':
        from halo_forge.rlvr.verifiers import GoVerifier
        run_after = cfg_dict.get('verifier', {}).get('run_after_compile', False)
        verifier = GoVerifier(run_after_compile=run_after)
    elif verifier_type == 'auto':
        from halo_forge.rlvr.verifiers import MultiLanguageVerifier
        run_after = cfg_dict.get('verifier', {}).get('run_after_compile', False)
        binary_cache = cfg_dict.get('verifier', {}).get('binary_cache_dir')
        verifier = MultiLanguageVerifier(
            run_after_compile=run_after,
            binary_cache_dir=binary_cache
        )
    elif verifier_type == 'execution':
        from halo_forge.rlvr.verifiers import ExecutionVerifier
        test_cases = cfg_dict.get('verifier', {}).get('test_cases', [])
        match_mode = cfg_dict.get('verifier', {}).get('match_mode', 'exact')
        verifier = ExecutionVerifier(
            test_cases=test_cases,
            match_mode=match_mode
        )
    else:
        print(f"Unknown verifier: {verifier_type}")
        print("Available: gcc, mingw, msvc, humaneval, mbpp, rust, go")
        sys.exit(1)
    
    # Create config
    keep_percent = getattr(args, 'keep_percent', None) or cfg_dict.get('keep_top_percent', 0.5)
    reward_threshold = getattr(args, 'reward_threshold', None) or cfg_dict.get('reward_threshold', 0.5)
    
    curriculum = getattr(args, 'curriculum', None) or cfg_dict.get('curriculum_strategy', 'none')
    reward_shaping = getattr(args, 'reward_shaping', None) or cfg_dict.get('reward_shaping_strategy', 'fixed')
    system_prompt = getattr(args, 'system_prompt', None) or cfg_dict.get('system_prompt', 'You are an expert Windows systems programmer.')
    lr_decay = getattr(args, 'lr_decay', None) or cfg_dict.get('lr_decay_per_cycle', 0.85)
    min_lr = getattr(args, 'min_lr', None) or cfg_dict.get('min_lr', 1e-6)
    
    # Resolve model path - handles SFT output directories automatically
    # This allows: --model models/code_sft (where adapters are in models/code_sft/final_model)
    model_arg = args.model or cfg_dict.get('base_model', 'Qwen/Qwen2.5-Coder-3B')
    checkpoint_arg = args.checkpoint or cfg_dict.get('sft_checkpoint')
    
    if checkpoint_arg:
        # Explicit checkpoint provided - use as-is
        base_model = model_arg
        sft_checkpoint = checkpoint_arg
    else:
        # Auto-detect from --model argument
        base_model, sft_checkpoint = _resolve_model_path(model_arg)
        if sft_checkpoint:
            print(f"  > Auto-detected SFT adapter: {sft_checkpoint}")
            print(f"  > Base model: {base_model}")
        else:
            # No adapter found - will train from scratch
            sft_checkpoint = cfg_dict.get('sft_checkpoint', 'models/sft/final_model')
    
    config = RAFTConfig(
        base_model=base_model,
        sft_checkpoint=sft_checkpoint,
        output_dir=args.output or cfg_dict.get('output_dir', 'models/raft'),
        num_cycles=args.cycles or cfg_dict.get('num_cycles', 3),
        keep_top_percent=keep_percent,
        reward_threshold=reward_threshold,
        curriculum_strategy=curriculum,
        reward_shaping_strategy=reward_shaping,
        system_prompt=system_prompt,
        lr_decay_per_cycle=lr_decay,
        min_lr=min_lr
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
    # Note: --experimental-attention is handled at script startup (before imports)
    
    print_banner()
    print(f"{GREEN}Benchmark{NC}")
    print("=" * 60)
    
    from halo_forge.benchmark.pass_at_k import Benchmark
    from halo_forge.rlvr.verifiers import (
        GCCVerifier, MinGWVerifier, RemoteMSVCVerifier,
        RustVerifier, GoVerifier, DotNetVerifier, PowerShellVerifier,
        MultiLanguageVerifier, AutoVerifier
    )
    
    # Setup verifier
    if args.verifier == 'gcc':
        verifier = GCCVerifier()
    elif args.verifier == 'mingw':
        verifier = MinGWVerifier()
    elif args.verifier == 'rust':
        verifier = RustVerifier(cross_compile=getattr(args, 'cross_compile', False))
    elif args.verifier == 'go':
        verifier = GoVerifier(cross_compile=getattr(args, 'cross_compile', False))
    elif args.verifier == 'dotnet':
        verifier = DotNetVerifier()
    elif args.verifier == 'powershell':
        verifier = PowerShellVerifier()
    elif args.verifier in ('auto', 'multi'):
        # Auto-detect language from code
        verifier = MultiLanguageVerifier(
            run_after_compile=getattr(args, 'run_after_compile', False)
        )
    elif args.verifier == 'msvc':
        # Validate required MSVC parameters
        missing = []
        if not args.host:
            missing.append('--host')
        if not args.user:
            missing.append('--user')
        if not args.ssh_key:
            missing.append('--ssh-key')
        
        if missing:
            print(f"Error: MSVC verifier requires: {', '.join(missing)}")
            print("\nExample:")
            print("  halo-forge benchmark run --verifier msvc \\")
            print("    --host 10.0.0.152 --user keys --ssh-key ~/.ssh/win \\")
            print("    --model Qwen/Qwen2.5-Coder-0.5B \\")
            print("    --prompts data/prompts.jsonl")
            print("\nOr use MinGW for local cross-compilation (no Windows needed):")
            print("  halo-forge benchmark run --verifier mingw ...")
            sys.exit(1)
        
        verifier = RemoteMSVCVerifier(
            host=args.host,
            user=args.user,
            ssh_key=args.ssh_key
        )
    else:
        print(f"Unknown verifier: {args.verifier}")
        print("Available verifiers: gcc, mingw, msvc, rust, go, dotnet, powershell, auto")
        sys.exit(1)
    
    # Resolve model path - handles SFT/RAFT output directories automatically
    model_arg = args.model
    base_model_arg = args.base_model
    
    if not base_model_arg:
        # Auto-detect from model path
        detected_base, detected_checkpoint = _resolve_model_path(model_arg)
        if detected_checkpoint:
            print(f"  > Auto-detected adapter: {detected_checkpoint}")
            print(f"  > Base model: {detected_base}")
            model_path = detected_checkpoint
            base_model_arg = detected_base
        else:
            model_path = model_arg
    else:
        model_path = model_arg
    
    # Create benchmark
    benchmark = Benchmark(
        model_path=model_path,
        verifier=verifier,
        base_model=base_model_arg,
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
            print("halo forge Smoke Test")
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
            print("halo forge Standard Test")
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
            print("halo forge Full Pipeline Test")
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


def cmd_inference_optimize(args):
    """Optimize model for inference."""
    from halo_forge.inference import (
        InferenceOptimizer, OptimizationConfig,
        check_dependencies, validate_config
    )
    
    print_banner()
    print(f"{GREEN}Inference Optimization{NC}")
    print("=" * 60)
    print(f"Optimizing model: {args.model}")
    print(f"Target precision: {args.target_precision}")
    print(f"Target latency: {args.target_latency}ms")
    
    config = OptimizationConfig(
        target_precision=args.target_precision,
        target_latency_ms=args.target_latency,
        output_dir=args.output
    )
    
    # Handle --dry-run
    if getattr(args, 'dry_run', False):
        print("\n[DRY RUN] Validating configuration and dependencies...")
        
        # Check dependencies
        deps = check_dependencies()
        print("\nDependencies:")
        for dep, available in deps.items():
            status = f"{GREEN}✓{NC}" if available else f"{RED}✗{NC}"
            print(f"  {status} {dep}")
        
        # Validate config
        try:
            warnings = validate_config(config)
            if warnings:
                print("\nWarnings:")
                for w in warnings:
                    print(f"  {YELLOW}⚠{NC} {w}")
            else:
                print(f"\n{GREEN}Configuration valid!{NC}")
        except Exception as e:
            print(f"\n{RED}Configuration error: {e}{NC}")
            sys.exit(1)
        
        # Check model path
        from pathlib import Path
        model_path = Path(args.model)
        if model_path.exists():
            print(f"\n{GREEN}✓{NC} Model path exists: {args.model}")
        else:
            print(f"\n{YELLOW}⚠{NC} Model path not found locally (may be HuggingFace ID)")
        
        print(f"\n{GREEN}[DRY RUN] All checks passed!{NC}")
        return
    
    optimizer = InferenceOptimizer(config)
    
    # Simple eval prompts for verification
    eval_prompts = [
        "Write a function to sort a list.",
        "Implement a binary search.",
        "Create a linked list class."
    ]
    
    result = optimizer.optimize(
        model_path=args.model,
        calibration_data=args.calibration_data,
        eval_prompts=eval_prompts
    )
    
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)
    print(f"Success: {result['success']}")
    if result.get('verification'):
        metrics = result['verification']['metrics']
        print(f"Latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
        print(f"Quality: {metrics.get('quality_score', 0):.2%}")
    print(f"Output: {args.output}")


def cmd_inference_export(args):
    """Export model to deployment format."""
    print_banner()
    print(f"{GREEN}Model Export{NC}")
    print("=" * 60)
    print(f"Exporting model: {args.model}")
    print(f"Format: {args.format}")
    print(f"Output: {args.output}")
    
    if args.format == 'gguf':
        from halo_forge.inference.export import GGUFExporter
        
        print(f"Quantization: {args.quantization}")
        
        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            device_map="cpu"  # Export on CPU
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        exporter = GGUFExporter()
        output_path = exporter.export(
            model,
            args.output,
            tokenizer=tokenizer,
            quantization=args.quantization
        )
        
        print(f"\nExported to: {output_path}")
        
    elif args.format == 'onnx':
        from halo_forge.inference.export import ONNXExporter
        
        # Load model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        exporter = ONNXExporter()
        output_path = exporter.export(
            model,
            args.output,
            tokenizer=tokenizer
        )
        
        print(f"\nExported to: {output_path}")


def cmd_inference_benchmark(args):
    """Benchmark inference latency."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import time
    
    print_banner()
    print(f"{GREEN}Inference Benchmark{NC}")
    print("=" * 60)
    print(f"Benchmarking: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Warmup iterations: {args.warmup}")
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Get test prompts
    if args.prompts:
        with open(args.prompts) as f:
            prompts = [json.loads(line).get('prompt', '') for line in f][:args.num_prompts]
    else:
        prompts = [
            "Write a function to calculate fibonacci numbers.",
            "Implement a binary search tree.",
            "Create a simple HTTP server.",
            "Write a sorting algorithm.",
            "Implement a stack data structure."
        ][:args.num_prompts]
    
    print(f"Testing with {len(prompts)} prompts...\n")
    
    # Warmup
    print("Warmup...")
    for i, prompt in enumerate(prompts[:args.warmup]):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
    
    # Benchmark
    print("Benchmarking...")
    latencies = []
    tokens_generated = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        
        latencies.append(latency_ms)
        tokens_generated.append(num_tokens)
    
    # Calculate metrics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    total_tokens = sum(tokens_generated)
    total_time = sum(latencies) / 1000
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Prompts tested: {len(prompts)}")
    print(f"Avg latency:    {avg_latency:.1f}ms")
    print(f"Min latency:    {min_latency:.1f}ms")
    print(f"Max latency:    {max_latency:.1f}ms")
    print(f"Tokens/second:  {tokens_per_second:.1f}")
    
    if args.measure_memory and torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak memory:    {memory_mb:.0f}MB")


def cmd_vlm_train(args):
    """Train VLM with RAFT."""
    from halo_forge.vlm import VLMRAFTTrainer
    from halo_forge.vlm.trainer import VLMRAFTConfig
    from halo_forge.vlm.data import load_vlm_dataset, list_vlm_datasets
    from halo_forge.vlm.verifiers import check_vlm_dependencies
    
    print_banner()
    
    print(f"\n{GREEN}VLM RAFT Training{NC}")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Dataset:     {args.dataset}")
    print(f"Output:      {args.output}")
    print(f"Cycles:      {args.cycles}")
    print("=" * 60)
    
    # Handle --dry-run
    if getattr(args, 'dry_run', False):
        print("\n[DRY RUN] Validating configuration and dependencies...")
        
        # Check VLM dependencies
        deps = check_vlm_dependencies()
        print("\nVLM Dependencies:")
        for dep, available in deps.items():
            status = f"{GREEN}✓{NC}" if available else f"{YELLOW}⚠{NC}"
            print(f"  {status} {dep}")
        
        # Check dataset
        if args.dataset.endswith('.jsonl'):
            from pathlib import Path
            dataset_path = Path(args.dataset)
            if dataset_path.exists():
                # Count samples
                with open(dataset_path) as f:
                    count = sum(1 for _ in f)
                print(f"\n{GREEN}✓{NC} Dataset: {args.dataset} ({count} samples)")
            else:
                print(f"\n{RED}✗{NC} Dataset not found: {args.dataset}")
                sys.exit(1)
        else:
            available = list_vlm_datasets()
            if args.dataset in available:
                print(f"\n{GREEN}✓{NC} Dataset: {args.dataset} (HuggingFace)")
            else:
                print(f"\n{RED}✗{NC} Unknown dataset: {args.dataset}")
                print(f"  Available: {', '.join(available)}")
                sys.exit(1)
        
        # Validate config values
        print("\nConfiguration:")
        print(f"  Cycles: {args.cycles}")
        print(f"  Samples/prompt: {args.samples_per_prompt}")
        print(f"  Perception weight: {args.perception_weight}")
        print(f"  Reasoning weight: {args.reasoning_weight}")
        print(f"  Output weight: {args.output_weight}")
        print(f"  LR decay: {args.lr_decay}")
        print(f"  Temperature: {args.temperature}")
        
        # Check model (just print - can't validate without loading)
        print(f"\nModel: {args.model}")
        print(f"  (Model will be loaded at training start)")
        
        print(f"\n{GREEN}[DRY RUN] All checks passed!{NC}")
        return
    
    # Create config
    config = VLMRAFTConfig(
        model_name=args.model,
        output_dir=args.output,
        num_cycles=args.cycles,
        samples_per_prompt=args.samples_per_prompt,
        perception_weight=args.perception_weight,
        reasoning_weight=args.reasoning_weight,
        output_weight=args.output_weight,
        lr_decay_per_cycle=args.lr_decay,
        temperature=args.temperature,
    )
    
    # Load dataset
    if args.dataset.endswith('.jsonl'):
        dataset_path = args.dataset
    else:
        available = list_vlm_datasets()
        if args.dataset not in available:
            print(f"{RED}Error: Unknown dataset '{args.dataset}'{NC}")
            print(f"Available: {', '.join(available)}")
            sys.exit(1)
        dataset_path = args.dataset
    
    # Create trainer and run
    trainer = VLMRAFTTrainer(config)
    
    try:
        trainer.train(dataset_path)
    finally:
        trainer.cleanup()
    
    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Output: {args.output}")


def cmd_vlm_sft(args):
    """SFT training for VLM."""
    from halo_forge.sft.trainer import SFTTrainer, SFTConfig
    
    print_banner()
    print(f"{GREEN}VLM SFT Training{NC}")
    print("=" * 60)
    
    dataset = getattr(args, 'dataset', 'llava')
    max_samples = getattr(args, 'max_samples', None)
    dry_run = getattr(args, 'dry_run', False)
    
    print(f"Model: {args.model}")
    print(f"Dataset: {dataset}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print()
    
    if dry_run:
        print(f"{YELLOW}Dry run mode - validating configuration only{NC}")
        from halo_forge.sft.datasets import get_sft_dataset_spec, is_huggingface_id
        spec = get_sft_dataset_spec(dataset)
        if spec:
            print(f"{GREEN}✓{NC} Dataset: {spec.name} ({spec.huggingface_id})")
        elif is_huggingface_id(dataset):
            print(f"{GREEN}✓{NC} HuggingFace dataset: {dataset}")
        else:
            print(f"{RED}✗{NC} Unknown dataset: {dataset}")
            sys.exit(1)
        print(f"{GREEN}Configuration valid!{NC}")
        return
    
    config = SFTConfig(
        model_name=args.model,
        dataset=dataset,
        max_samples=max_samples,
        output_dir=args.output,
        num_epochs=args.epochs
    )
    
    trainer = SFTTrainer(config)
    trainer.train()


def cmd_vlm_benchmark(args):
    """Benchmark VLM on dataset."""
    from halo_forge.vlm.data import load_vlm_dataset
    from halo_forge.vlm.models import get_vlm_adapter
    from halo_forge.vlm.verifiers import VisionVerifier
    
    print_banner()
    
    print(f"\n{GREEN}VLM Benchmark{NC}")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Split:   {args.split}")
    print(f"Limit:   {args.limit}")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_vlm_dataset(args.dataset, split=args.split, limit=args.limit)
    
    # Load model
    print("\nLoading model...")
    adapter = get_vlm_adapter(args.model)
    adapter.load()
    
    # Initialize verifier
    verifier = VisionVerifier()
    
    # Run benchmark
    print(f"\nBenchmarking {len(dataset)} samples...")
    results = []
    correct = 0
    total_reward = 0.0
    
    from tqdm import tqdm
    for sample in tqdm(dataset, desc="Evaluating"):
        # Generate
        output = adapter.generate(
            image=sample.load_image(),
            prompt=sample.prompt,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False
        )
        
        # Verify
        result = verifier.verify(
            image=sample.load_image(),
            prompt=sample.prompt,
            completion=output.text,
            ground_truth=sample.ground_truth
        )
        
        results.append({
            'prompt': sample.prompt[:100],
            'ground_truth': sample.ground_truth,
            'completion': output.text[:200],
            'reward': result.reward,
            'success': result.success
        })
        
        if result.success:
            correct += 1
        total_reward += result.reward
    
    # Print results
    print("\n" + "=" * 60)
    print("VLM BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total samples:  {len(results)}")
    print(f"Correct:        {correct} ({correct/len(results)*100:.1f}%)")
    print(f"Avg reward:     {total_reward/len(results):.3f}")
    
    # Save results if output specified
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump({
                'model': args.model,
                'dataset': args.dataset,
                'split': args.split,
                'accuracy': correct / len(results),
                'avg_reward': total_reward / len(results),
                'results': results
            }, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Cleanup
    adapter.cleanup()
    verifier.cleanup()


def cmd_vlm_datasets(args):
    """List available VLM datasets."""
    from halo_forge.vlm.data import list_vlm_datasets
    
    print_banner()
    
    print(f"\n{GREEN}Available VLM Datasets{NC}")
    print("=" * 60)
    
    datasets = list_vlm_datasets()
    
    dataset_info = {
        'textvqa': 'Text reading in natural images',
        'docvqa': 'Document understanding',
        'chartqa': 'Chart interpretation',
        'realworldqa': 'Real-world visual reasoning',
        'mathvista': 'Mathematical reasoning with visuals',
    }
    
    for name in datasets:
        desc = dataset_info.get(name, 'Vision-language dataset')
        print(f"  {name:15} - {desc}")


# =============================================================================
# Audio Commands
# =============================================================================

def cmd_audio_datasets(args):
    """List available audio datasets."""
    from halo_forge.audio.data import list_audio_datasets
    
    print_banner()
    
    print(f"\n{GREEN}Available Audio Datasets{NC}")
    print("=" * 60)
    
    dataset_info = {
        'librispeech': ('ASR', 'Clean audiobook speech (960h)'),
        'common_voice': ('ASR', 'Crowdsourced multilingual (2000h+)'),
        'audioset': ('Classification', 'Sound event detection (5M clips)'),
        'speech_commands': ('Classification', 'Keyword spotting (105k)'),
    }
    
    datasets = list_audio_datasets()
    
    for name in datasets:
        task, desc = dataset_info.get(name, ('Unknown', 'Audio dataset'))
        print(f"  {name:18} [{task:14}] - {desc}")
    
    print()
    print("Usage:")
    print("  halo-forge audio benchmark --model openai/whisper-small --dataset librispeech")
    print("  halo-forge audio train --model openai/whisper-small --dataset librispeech")


def cmd_audio_sft(args):
    """SFT training for audio."""
    from halo_forge.sft.trainer import SFTTrainer, SFTConfig
    
    print_banner()
    print(f"{GREEN}Audio SFT Training{NC}")
    print("=" * 60)
    
    dataset = getattr(args, 'dataset', 'librispeech_sft')
    max_samples = getattr(args, 'max_samples', None)
    dry_run = getattr(args, 'dry_run', False)
    
    print(f"Model: {args.model}")
    print(f"Dataset: {dataset}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print()
    
    if dry_run:
        print(f"{YELLOW}Dry run mode - validating configuration only{NC}")
        from halo_forge.sft.datasets import get_sft_dataset_spec, is_huggingface_id
        spec = get_sft_dataset_spec(dataset)
        if spec:
            print(f"{GREEN}✓{NC} Dataset: {spec.name} ({spec.huggingface_id})")
        elif is_huggingface_id(dataset):
            print(f"{GREEN}✓{NC} HuggingFace dataset: {dataset}")
        else:
            print(f"{RED}✗{NC} Unknown dataset: {dataset}")
            sys.exit(1)
        print(f"{GREEN}Configuration valid!{NC}")
        return
    
    config = SFTConfig(
        model_name=args.model,
        dataset=dataset,
        max_samples=max_samples,
        output_dir=args.output,
        num_epochs=args.epochs
    )
    
    trainer = SFTTrainer(config)
    trainer.train()


def cmd_audio_benchmark(args):
    """Benchmark audio model."""
    from halo_forge.audio import AudioRAFTTrainer, AudioRAFTConfig
    
    print_banner()
    
    print(f"\n{GREEN}Audio Benchmark{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Limit: {args.limit}")
    
    # Check dependencies
    try:
        from halo_forge.audio.data.processors import check_audio_dependencies
        deps = check_audio_dependencies()
        
        if not deps.get('torchaudio'):
            print(f"\n{YELLOW}Warning: torchaudio not installed{NC}")
            print("Install with: pip install torchaudio")
    except ImportError as e:
        print(f"\n{RED}Error: {e}{NC}")
        sys.exit(1)
    
    # Create config
    config = AudioRAFTConfig(
        model_name=args.model,
        task=args.task,
        wer_threshold=0.3,
    )
    
    # Run benchmark
    trainer = AudioRAFTTrainer(config)
    results = trainer.benchmark(args.dataset, limit=args.limit)
    
    print(f"\n{GREEN}Results:{NC}")
    print(f"  Samples: {results['samples']}")
    print(f"  Success rate: {results['success_rate']:.1%}")
    print(f"  Average reward: {results['average_reward']:.3f}")
    
    if args.task == 'asr':
        print(f"  Average WER: {results.get('average_wer', 'N/A'):.1%}")
    
    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def cmd_audio_train(args):
    """Train audio model with RAFT."""
    from halo_forge.audio import AudioRAFTTrainer, AudioRAFTConfig
    
    print_banner()
    
    print(f"\n{GREEN}Audio RAFT Training{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Cycles: {args.cycles}")
    print(f"Output: {args.output}")
    
    if args.dry_run:
        print(f"\n{YELLOW}Dry run mode - validating configuration only{NC}")
        
        # Check dependencies
        try:
            from halo_forge.audio.data.processors import check_audio_dependencies
            deps = check_audio_dependencies()
            
            print(f"\nDependencies:")
            for dep, installed in deps.items():
                status = f"{GREEN}✓{NC}" if installed else f"{RED}✗{NC}"
                print(f"  {status} {dep}")
            
            # Try loading dataset info
            from halo_forge.audio.data import list_audio_datasets
            if args.dataset in list_audio_datasets():
                print(f"\n{GREEN}✓{NC} Dataset: {args.dataset}")
            else:
                print(f"\n{YELLOW}⚠{NC} Dataset: {args.dataset} (custom path)")
            
            print(f"\n{GREEN}Configuration validated successfully.{NC}")
        except Exception as e:
            print(f"\n{RED}Validation error: {e}{NC}")
            sys.exit(1)
        return
    
    # Create config
    config = AudioRAFTConfig(
        model_name=args.model,
        task=args.task,
        num_cycles=args.cycles,
        learning_rate=args.lr,
        lr_decay_per_cycle=args.lr_decay,
        output_dir=args.output,
    )
    
    # Run training
    trainer = AudioRAFTTrainer(config)
    results = trainer.train(args.dataset)
    
    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Final model saved to: {args.output}")
    
    print("\nUsage:")
    print("  halo-forge vlm train --dataset textvqa --model Qwen/Qwen2-VL-7B-Instruct")
    print("  halo-forge vlm benchmark --dataset docvqa --model path/to/model")


def main():
    parser = argparse.ArgumentParser(
        prog='halo-forge',
        description='Complete RLVR training framework for AMD Strix Halo'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # config command
    config_parser = subparsers.add_parser('config', help='Configuration utilities')
    config_subparsers = config_parser.add_subparsers(dest='config_command', required=True)
    
    # config validate
    config_validate_parser = config_subparsers.add_parser('validate', help='Validate config file')
    config_validate_parser.add_argument('config', help='Path to config file')
    config_validate_parser.add_argument('--type', '-t', choices=['raft', 'sft', 'auto'], default='auto',
                                        help='Config type (auto-detected from filename if not specified)')
    config_validate_parser.add_argument('--verbose', '-v', action='store_true', help='Show config contents')
    
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
    
    # data validate
    validate_parser = data_subparsers.add_parser('validate', help='Validate dataset format')
    validate_parser.add_argument('file', help='Path to JSONL file to validate')
    validate_parser.add_argument('--preview', '-p', action='store_true', help='Show preview of examples')
    
    # sft command
    sft_parser = subparsers.add_parser('sft', help='SFT training')
    sft_subparsers = sft_parser.add_subparsers(dest='sft_command', required=True)
    
    # sft train
    sft_train_parser = sft_subparsers.add_parser('train', help='Run SFT training')
    sft_train_parser.add_argument('--config', '-c', help='Config file path')
    sft_train_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-Coder-7B', help='Base model')
    sft_train_parser.add_argument('--dataset', '-d', help='HuggingFace dataset ID or short name (e.g., codealpaca, metamath)')
    sft_train_parser.add_argument('--data', help='Local training data file (JSONL)')
    sft_train_parser.add_argument('--max-samples', type=int, help='Limit number of training samples')
    sft_train_parser.add_argument('--output', '-o', default='models/sft', help='Output directory')
    sft_train_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    sft_train_parser.add_argument('--resume', help='Resume from checkpoint')
    sft_train_parser.add_argument('--dry-run', action='store_true', help='Validate config without training')
    
    # sft datasets
    sft_datasets_parser = sft_subparsers.add_parser('datasets', help='List available SFT datasets')
    
    # raft command
    raft_parser = subparsers.add_parser('raft', help='RAFT training')
    raft_subparsers = raft_parser.add_subparsers(dest='raft_command', required=True)
    
    # raft train
    raft_train_parser = raft_subparsers.add_parser('train', help='Run RAFT training')
    raft_train_parser.add_argument('--config', '-c', help='Config file path')
    raft_train_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-Coder-3B', help='Base model')
    raft_train_parser.add_argument('--checkpoint', help='SFT checkpoint path (optional)')
    raft_train_parser.add_argument('--prompts', '-p', help='Prompts file')
    raft_train_parser.add_argument('--output', '-o', default='models/raft', help='Output directory')
    raft_train_parser.add_argument('--cycles', type=int, help='Number of RAFT cycles')
    raft_train_parser.add_argument('--verifier', default='gcc',
                                   choices=['gcc', 'mingw', 'msvc', 'rust', 'go', 'dotnet', 'powershell', 'auto'],
                                   help='Verifier type (auto=multi-language)')
    raft_train_parser.add_argument('--keep-percent', type=float, default=0.5, 
                                   help='Keep top X%% of passing samples (0.0-1.0, default: 0.5 = 50%%)')
    raft_train_parser.add_argument('--reward-threshold', type=float, default=0.5,
                                   help='Minimum reward to consider sample passing (default: 0.5)')
    raft_train_parser.add_argument('--curriculum', default='none',
                                   choices=['none', 'complexity', 'progressive', 'adaptive'],
                                   help='Curriculum learning strategy (default: none)')
    raft_train_parser.add_argument('--reward-shaping', default='fixed',
                                   choices=['fixed', 'annealing', 'adaptive', 'warmup'],
                                   help='Reward shaping strategy (default: fixed)')
    raft_train_parser.add_argument('--lr-decay', type=float, default=0.85,
                                   help='Learning rate decay per cycle (default: 0.85)')
    raft_train_parser.add_argument('--min-lr', type=float, default=1e-6,
                                   help='Minimum learning rate floor (default: 1e-6)')
    raft_train_parser.add_argument('--experimental-attention', action='store_true',
                                   help='Enable experimental ROCm attention (needed for LFM2.5, etc.)')
    raft_train_parser.add_argument('--system-prompt', 
                                   default='You are an expert Windows systems programmer.',
                                   help='System prompt for generation')
    raft_train_parser.add_argument('--host', help='MSVC verifier host')
    raft_train_parser.add_argument('--user', help='MSVC verifier user')
    raft_train_parser.add_argument('--ssh-key', help='MSVC verifier SSH key')
    
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
    bench_run_parser.add_argument('--verifier', default='gcc', 
                                   choices=['gcc', 'mingw', 'msvc', 'rust', 'go', 'dotnet', 'powershell', 'auto'],
                                   help='Verifier type (auto=multi-language)')
    bench_run_parser.add_argument('--base-model', default='Qwen/Qwen2.5-Coder-7B', help='Base model')
    bench_run_parser.add_argument('--system-prompt', default='You are an expert Windows systems programmer.', help='System prompt')
    bench_run_parser.add_argument('--host', help='MSVC host')
    bench_run_parser.add_argument('--user', help='MSVC user')
    bench_run_parser.add_argument('--ssh-key', help='MSVC SSH key')
    bench_run_parser.add_argument('--cross-compile', action='store_true', help='Enable Windows cross-compilation for rust/go')
    bench_run_parser.add_argument('--run-after-compile', action='store_true', help='Run compiled code after compile')
    bench_run_parser.add_argument('--experimental-attention', action='store_true',
                                  help='Enable experimental ROCm attention (needed for LFM2.5, etc.)')
    
    # benchmark full (comprehensive RAFT benchmark with hardware metrics)
    bench_full_parser = bench_subparsers.add_parser('full', help='Run comprehensive RAFT benchmark')
    bench_full_parser.add_argument('--model', '-m', help='Model to benchmark (e.g., Qwen/Qwen2.5-Coder-0.5B)')
    bench_full_parser.add_argument('--suite', '-s', choices=['all', 'small', 'medium'],
                                   help='Run predefined suite: all (0.5B, 1.5B, 3B), small (0.5B), medium (0.5B, 1.5B)')
    bench_full_parser.add_argument('--cycles', '-c', type=int, default=2, help='Number of RAFT cycles (default: 2)')
    bench_full_parser.add_argument('--output', '-o', default='results/benchmarks', help='Output directory')
    bench_full_parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    # inference command
    inference_parser = subparsers.add_parser('inference', help='Inference optimization')
    inference_subparsers = inference_parser.add_subparsers(dest='inference_command', required=True)
    
    # inference optimize
    inf_optimize_parser = inference_subparsers.add_parser('optimize', help='Optimize model for inference')
    inf_optimize_parser.add_argument('--model', '-m', required=True, help='Model path')
    inf_optimize_parser.add_argument('--target-precision', default='int4',
                                     choices=['int4', 'int8', 'fp16'],
                                     help='Target precision (default: int4)')
    inf_optimize_parser.add_argument('--target-latency', type=float, default=50.0,
                                     help='Target latency in ms (default: 50)')
    inf_optimize_parser.add_argument('--calibration-data', help='Path to calibration data JSONL')
    inf_optimize_parser.add_argument('--output', '-o', default='models/optimized', help='Output directory')
    inf_optimize_parser.add_argument('--dry-run', action='store_true',
                                     help='Validate config and dependencies without running optimization')
    
    # inference export
    inf_export_parser = inference_subparsers.add_parser('export', help='Export model to deployment format')
    inf_export_parser.add_argument('--model', '-m', required=True, help='Model path')
    inf_export_parser.add_argument('--format', '-f', required=True,
                                   choices=['gguf', 'onnx'],
                                   help='Export format')
    inf_export_parser.add_argument('--quantization', '-q', default='Q4_K_M',
                                   help='GGUF quantization type (default: Q4_K_M)')
    inf_export_parser.add_argument('--output', '-o', required=True, help='Output path')
    
    # inference benchmark
    inf_bench_parser = inference_subparsers.add_parser('benchmark', help='Benchmark inference latency')
    inf_bench_parser.add_argument('--model', '-m', required=True, help='Model path')
    inf_bench_parser.add_argument('--prompts', '-p', help='Test prompts JSONL')
    inf_bench_parser.add_argument('--num-prompts', type=int, default=10, help='Number of prompts to test')
    inf_bench_parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens to generate')
    inf_bench_parser.add_argument('--warmup', type=int, default=3, help='Warmup iterations')
    inf_bench_parser.add_argument('--measure-memory', action='store_true', help='Measure memory usage')
    
    # vlm command
    vlm_parser = subparsers.add_parser('vlm', help='Vision-Language Model training')
    vlm_subparsers = vlm_parser.add_subparsers(dest='vlm_command', required=True)
    
    # vlm train
    vlm_train_parser = vlm_subparsers.add_parser('train', help='Train VLM with RAFT')
    vlm_train_parser.add_argument('--model', '-m', default='Qwen/Qwen2-VL-7B-Instruct',
                                  help='VLM model name')
    vlm_train_parser.add_argument('--dataset', '-d', required=True,
                                  help='Dataset name (textvqa, docvqa, chartqa) or JSONL path')
    vlm_train_parser.add_argument('--output', '-o', default='models/vlm_raft', help='Output directory')
    vlm_train_parser.add_argument('--cycles', type=int, default=6, help='Number of RAFT cycles')
    vlm_train_parser.add_argument('--samples-per-prompt', type=int, default=4,
                                  help='Samples per prompt (default: 4)')
    vlm_train_parser.add_argument('--perception-weight', type=float, default=0.3,
                                  help='Weight for perception verification (default: 0.3)')
    vlm_train_parser.add_argument('--reasoning-weight', type=float, default=0.4,
                                  help='Weight for reasoning verification (default: 0.4)')
    vlm_train_parser.add_argument('--output-weight', type=float, default=0.3,
                                  help='Weight for output verification (default: 0.3)')
    vlm_train_parser.add_argument('--lr-decay', type=float, default=0.85,
                                  help='Learning rate decay per cycle (default: 0.85)')
    vlm_train_parser.add_argument('--temperature', type=float, default=0.7,
                                  help='Generation temperature (default: 0.7)')
    vlm_train_parser.add_argument('--limit', type=int, help='Limit dataset samples')
    vlm_train_parser.add_argument('--dry-run', action='store_true',
                                  help='Validate config and datasets without running training')
    
    # vlm benchmark
    vlm_bench_parser = vlm_subparsers.add_parser('benchmark', help='Benchmark VLM')
    vlm_bench_parser.add_argument('--model', '-m', required=True, help='VLM model path')
    vlm_bench_parser.add_argument('--dataset', '-d', default='textvqa',
                                  help='Dataset name (default: textvqa)')
    vlm_bench_parser.add_argument('--split', default='validation', help='Dataset split')
    vlm_bench_parser.add_argument('--limit', type=int, default=100, help='Limit samples (default: 100)')
    vlm_bench_parser.add_argument('--output', '-o', help='Output file for results')
    
    # vlm datasets
    vlm_datasets_parser = vlm_subparsers.add_parser('datasets', help='List available VLM datasets')
    
    # vlm sft
    vlm_sft_parser = vlm_subparsers.add_parser('sft', help='SFT training for VLM')
    vlm_sft_parser.add_argument('--model', '-m', default='Qwen/Qwen2-VL-2B-Instruct',
                                help='VLM model name')
    vlm_sft_parser.add_argument('--dataset', '-d', default='llava',
                                help='Dataset name (default: llava)')
    vlm_sft_parser.add_argument('--max-samples', type=int, help='Limit training samples')
    vlm_sft_parser.add_argument('--output', '-o', default='models/vlm_sft', help='Output directory')
    vlm_sft_parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    vlm_sft_parser.add_argument('--dry-run', action='store_true', help='Validate config only')
    
    # audio command
    audio_parser = subparsers.add_parser('audio', help='Audio-language training')
    audio_subparsers = audio_parser.add_subparsers(dest='audio_command', required=True)
    
    # audio datasets
    audio_datasets_parser = audio_subparsers.add_parser('datasets', help='List available audio datasets')
    
    # audio benchmark
    audio_bench_parser = audio_subparsers.add_parser('benchmark', help='Benchmark audio model')
    audio_bench_parser.add_argument('--model', '-m', default='openai/whisper-small',
                                    help='Audio model (default: openai/whisper-small)')
    audio_bench_parser.add_argument('--dataset', '-d', default='librispeech',
                                    help='Dataset name (default: librispeech)')
    audio_bench_parser.add_argument('--task', '-t', default='asr',
                                    choices=['asr', 'tts', 'classification'],
                                    help='Task type (default: asr)')
    audio_bench_parser.add_argument('--limit', type=int, default=100,
                                    help='Limit samples (default: 100)')
    audio_bench_parser.add_argument('--output', '-o', help='Output file for results')
    
    # audio train
    audio_train_parser = audio_subparsers.add_parser('train', help='Train audio model with RAFT')
    audio_train_parser.add_argument('--model', '-m', default='openai/whisper-small',
                                    help='Audio model (default: openai/whisper-small)')
    audio_train_parser.add_argument('--dataset', '-d', default='librispeech',
                                    help='Dataset name or path (default: librispeech)')
    audio_train_parser.add_argument('--task', '-t', default='asr',
                                    choices=['asr', 'tts', 'classification'],
                                    help='Task type (default: asr)')
    audio_train_parser.add_argument('--cycles', type=int, default=6,
                                    help='Number of RAFT cycles (default: 6)')
    audio_train_parser.add_argument('--lr', type=float, default=5e-5,
                                    help='Initial learning rate (default: 5e-5)')
    audio_train_parser.add_argument('--lr-decay', type=float, default=0.85,
                                    help='Learning rate decay per cycle (default: 0.85)')
    audio_train_parser.add_argument('--output', '-o', default='models/audio_raft',
                                    help='Output directory (default: models/audio_raft)')
    audio_train_parser.add_argument('--dry-run', action='store_true',
                                    help='Validate config without running training')
    
    # audio sft
    audio_sft_parser = audio_subparsers.add_parser('sft', help='SFT training for audio')
    audio_sft_parser.add_argument('--model', '-m', default='openai/whisper-small',
                                  help='Audio model (default: openai/whisper-small)')
    audio_sft_parser.add_argument('--dataset', '-d', default='librispeech_sft',
                                  help='Dataset name (default: librispeech_sft)')
    audio_sft_parser.add_argument('--max-samples', type=int, help='Limit training samples')
    audio_sft_parser.add_argument('--output', '-o', default='models/audio_sft', help='Output directory')
    audio_sft_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    audio_sft_parser.add_argument('--dry-run', action='store_true', help='Validate config only')
    
    # reasoning command
    reasoning_parser = subparsers.add_parser('reasoning', help='Math/Reasoning training')
    reasoning_subparsers = reasoning_parser.add_subparsers(dest='reasoning_command', required=True)
    
    # reasoning datasets
    reasoning_datasets_parser = reasoning_subparsers.add_parser('datasets', help='List available math datasets')
    
    # reasoning benchmark
    reasoning_bench_parser = reasoning_subparsers.add_parser('benchmark', help='Benchmark math reasoning')
    reasoning_bench_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-7B-Instruct',
                                        help='Model name (default: Qwen/Qwen2.5-7B-Instruct)')
    reasoning_bench_parser.add_argument('--dataset', '-d', default='gsm8k',
                                        help='Dataset name (default: gsm8k)')
    reasoning_bench_parser.add_argument('--split', default='test',
                                        help='Dataset split (default: test)')
    reasoning_bench_parser.add_argument('--limit', type=int, default=100,
                                        help='Limit samples (default: 100)')
    reasoning_bench_parser.add_argument('--output', '-o', help='Output file for results')
    
    # reasoning train
    reasoning_train_parser = reasoning_subparsers.add_parser('train', help='Train with RAFT')
    reasoning_train_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-7B-Instruct',
                                        help='Model name (default: Qwen/Qwen2.5-7B-Instruct)')
    reasoning_train_parser.add_argument('--dataset', '-d', default='gsm8k',
                                        help='Dataset name (default: gsm8k)')
    reasoning_train_parser.add_argument('--cycles', type=int, default=4,
                                        help='Number of RAFT cycles (default: 4)')
    reasoning_train_parser.add_argument('--lr', type=float, default=1e-5,
                                        help='Initial learning rate (default: 1e-5)')
    reasoning_train_parser.add_argument('--lr-decay', type=float, default=0.85,
                                        help='Learning rate decay per cycle (default: 0.85)')
    reasoning_train_parser.add_argument('--output', '-o', default='models/reasoning_raft',
                                        help='Output directory (default: models/reasoning_raft)')
    reasoning_train_parser.add_argument('--limit', type=int, help='Limit dataset samples')
    reasoning_train_parser.add_argument('--dry-run', action='store_true',
                                        help='Validate config without running training')
    
    # reasoning sft
    reasoning_sft_parser = reasoning_subparsers.add_parser('sft', help='SFT training for reasoning')
    reasoning_sft_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-3B-Instruct',
                                      help='Model name (default: Qwen/Qwen2.5-3B-Instruct)')
    reasoning_sft_parser.add_argument('--dataset', '-d', default='metamath',
                                      help='Dataset name (default: metamath)')
    reasoning_sft_parser.add_argument('--max-samples', type=int, help='Limit training samples')
    reasoning_sft_parser.add_argument('--output', '-o', default='models/reasoning_sft', help='Output directory')
    reasoning_sft_parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    reasoning_sft_parser.add_argument('--dry-run', action='store_true', help='Validate config only')
    
    # agentic command (tool calling)
    agentic_parser = subparsers.add_parser('agentic', help='Tool calling / function calling training')
    agentic_subparsers = agentic_parser.add_subparsers(dest='agentic_command', required=True)
    
    # agentic datasets
    agentic_datasets_parser = agentic_subparsers.add_parser('datasets', help='List available tool calling datasets')
    
    # agentic benchmark
    agentic_bench_parser = agentic_subparsers.add_parser('benchmark', help='Benchmark tool calling model')
    agentic_bench_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-7B-Instruct',
                                      help='Model name (default: Qwen/Qwen2.5-7B-Instruct)')
    agentic_bench_parser.add_argument('--dataset', '-d', default='xlam',
                                      help='Dataset name: xlam, glaive (default: xlam)')
    agentic_bench_parser.add_argument('--limit', type=int, default=100,
                                      help='Limit samples (default: 100)')
    agentic_bench_parser.add_argument('--output', '-o', help='Output file for results')
    
    # agentic train
    agentic_train_parser = agentic_subparsers.add_parser('train', help='Train tool calling with RAFT')
    agentic_train_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-7B-Instruct',
                                      help='Model name (default: Qwen/Qwen2.5-7B-Instruct)')
    agentic_train_parser.add_argument('--dataset', '-d', default='xlam',
                                      help='Dataset name: xlam, glaive (default: xlam)')
    agentic_train_parser.add_argument('--cycles', type=int, default=5,
                                      help='Number of RAFT cycles (default: 5)')
    agentic_train_parser.add_argument('--lr', type=float, default=5e-5,
                                      help='Initial learning rate (default: 5e-5)')
    agentic_train_parser.add_argument('--lr-decay', type=float, default=0.85,
                                      help='Learning rate decay per cycle (default: 0.85)')
    agentic_train_parser.add_argument('--output', '-o', default='models/agentic_raft',
                                      help='Output directory (default: models/agentic_raft)')
    agentic_train_parser.add_argument('--limit', type=int, help='Limit dataset samples')
    agentic_train_parser.add_argument('--dry-run', action='store_true',
                                      help='Validate config without running training')
    
    # agentic sft
    agentic_sft_parser = agentic_subparsers.add_parser('sft', help='SFT training for tool calling')
    agentic_sft_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-7B-Instruct',
                                    help='Model name (default: Qwen/Qwen2.5-7B-Instruct)')
    agentic_sft_parser.add_argument('--dataset', '-d', default='xlam_sft',
                                    help='Dataset name (default: xlam_sft)')
    agentic_sft_parser.add_argument('--max-samples', type=int, help='Limit training samples')
    agentic_sft_parser.add_argument('--output', '-o', default='models/agentic_sft', help='Output directory')
    agentic_sft_parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    agentic_sft_parser.add_argument('--dry-run', action='store_true', help='Validate config only')
    
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
    
    # Parse arguments and dispatch
    args = parser.parse_args()
    _dispatch_commands(args)


# =============================================================================
# Reasoning Commands
# =============================================================================

def cmd_reasoning_datasets(args):
    """List available math datasets."""
    from halo_forge.reasoning.data import list_math_datasets
    
    print_banner()
    
    print(f"\n{GREEN}Available Math/Reasoning Datasets{NC}")
    print("=" * 60)
    
    dataset_info = {
        'gsm8k': ('Grade School', '8.5K problems, 2-8 step solutions'),
        'math': ('Competition', '12.5K problems, 7 subjects, 5 levels'),
        'aime': ('Competition', 'AIME problems (hard)'),
    }
    
    datasets = list_math_datasets()
    
    for name in datasets:
        level, desc = dataset_info.get(name, ('Unknown', 'Math dataset'))
        print(f"  {name:12} [{level:12}] - {desc}")
    
    print()
    print("Usage:")
    print("  halo-forge reasoning benchmark --dataset gsm8k")
    print("  halo-forge reasoning train --dataset gsm8k --cycles 4")


def cmd_reasoning_sft(args):
    """SFT training for reasoning."""
    from halo_forge.sft.trainer import SFTTrainer, SFTConfig
    
    print_banner()
    print(f"{GREEN}Reasoning SFT Training{NC}")
    print("=" * 60)
    
    dataset = getattr(args, 'dataset', 'metamath')
    max_samples = getattr(args, 'max_samples', None)
    dry_run = getattr(args, 'dry_run', False)
    
    print(f"Model: {args.model}")
    print(f"Dataset: {dataset}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print()
    
    if dry_run:
        print(f"{YELLOW}Dry run mode - validating configuration only{NC}")
        from halo_forge.sft.datasets import get_sft_dataset_spec, is_huggingface_id
        spec = get_sft_dataset_spec(dataset)
        if spec:
            print(f"{GREEN}✓{NC} Dataset: {spec.name} ({spec.huggingface_id})")
        elif is_huggingface_id(dataset):
            print(f"{GREEN}✓{NC} HuggingFace dataset: {dataset}")
        else:
            print(f"{RED}✗{NC} Unknown dataset: {dataset}")
            sys.exit(1)
        print(f"{GREEN}Configuration valid!{NC}")
        return
    
    config = SFTConfig(
        model_name=args.model,
        dataset=dataset,
        max_samples=max_samples,
        output_dir=args.output,
        num_epochs=args.epochs
    )
    
    trainer = SFTTrainer(config)
    trainer.train()


def cmd_reasoning_benchmark(args):
    """Benchmark math reasoning model."""
    from halo_forge.reasoning import MathVerifier, ReasoningRAFTConfig
    from halo_forge.reasoning.data import load_math_dataset
    
    print_banner()
    
    print(f"\n{GREEN}Reasoning Benchmark{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Limit: {args.limit}")
    
    # Load dataset
    try:
        dataset = load_math_dataset(args.dataset, split=args.split, limit=args.limit)
        print(f"\nLoaded {len(dataset)} samples from {args.dataset}")
    except Exception as e:
        print(f"\n{RED}Error loading dataset: {e}{NC}")
        sys.exit(1)
    
    # Load model
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Model loaded on {model.device}")
    except Exception as e:
        print(f"\n{RED}Error loading model: {e}{NC}")
        sys.exit(1)
    
    # Run benchmark
    verifier = MathVerifier()
    correct = 0
    total = 0
    total_reward = 0
    
    print(f"\nRunning benchmark...")
    for sample in dataset:
        # Format prompt
        prompt = (
            f"Solve the following math problem step by step. "
            f"Put your final answer in \\boxed{{}}.\n\n"
            f"Problem: {sample.question}\n\nSolution:"
        )
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        completion = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Verify
        result = verifier.verify(sample.question, completion, sample.answer)
        
        total += 1
        total_reward += result.reward
        if result.success:
            correct += 1
    
    # Results
    accuracy = correct / total if total > 0 else 0
    avg_reward = total_reward / total if total > 0 else 0
    
    print(f"\n{GREEN}Results:{NC}")
    print(f"  Samples: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Average reward: {avg_reward:.3f}")
    
    if args.output:
        results = {
            'model': args.model,
            'dataset': args.dataset,
            'samples': total,
            'correct': correct,
            'accuracy': accuracy,
            'avg_reward': avg_reward,
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def cmd_reasoning_train(args):
    """Train reasoning model with RAFT."""
    from halo_forge.reasoning import ReasoningRAFTTrainer, ReasoningRAFTConfig
    from halo_forge.reasoning.data import load_math_dataset
    
    print_banner()
    
    print(f"\n{GREEN}Reasoning RAFT Training{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Cycles: {args.cycles}")
    print(f"Output: {args.output}")
    
    if args.dry_run:
        print(f"\n{YELLOW}Dry run mode - validating configuration only{NC}")
        
        # Check dependencies
        try:
            import sympy
            print(f"\n{GREEN}✓{NC} sympy installed")
        except ImportError:
            print(f"\n{RED}✗{NC} sympy not installed (pip install sympy)")
        
        # Check dataset
        try:
            from halo_forge.reasoning.data import list_math_datasets
            if args.dataset in list_math_datasets():
                print(f"{GREEN}✓{NC} Dataset: {args.dataset}")
            else:
                print(f"{RED}✗{NC} Unknown dataset: {args.dataset}")
        except Exception as e:
            print(f"{RED}✗{NC} Error: {e}")
        
        print(f"\n{GREEN}Configuration valid!{NC}")
        return
    
    # Create config
    config = ReasoningRAFTConfig(
        model_name=args.model,
        num_cycles=args.cycles,
        learning_rate=args.lr,
        lr_decay_per_cycle=args.lr_decay,
        output_dir=args.output,
    )
    
    # Load dataset
    dataset = load_math_dataset(args.dataset, split="train", limit=args.limit)
    print(f"\nLoaded {len(dataset)} samples from {args.dataset}")
    
    # Train
    trainer = ReasoningRAFTTrainer(config)
    summary = trainer.train(list(dataset))
    
    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Final accuracy: {summary.get('final_accuracy', 0):.1%}")
    print(f"Results saved to: {args.output}")


# =============================================================================
# Agentic / Tool Calling Commands
# =============================================================================

def cmd_agentic_datasets(args):
    """List available agentic/tool calling datasets."""
    from halo_forge.agentic.data import list_agentic_datasets
    
    print_banner()
    
    datasets = list_agentic_datasets()
    
    print(f"\n{GREEN}Available Agentic / Tool Calling Datasets{NC}")
    print("=" * 60)
    
    for key, info in datasets.items():
        print(f"\n  {CYAN}{key:<12}{NC} [{YELLOW}Tool Calling{NC}]")
        print(f"               {info['description']}")
        print(f"               HuggingFace: {info['hf_path']}")
        print(f"               Size: {info['size']}")
    
    print(f"\n{YELLOW}Note:{NC} Datasets are downloaded on first use via HuggingFace.")


def cmd_agentic_sft(args):
    """SFT training for tool calling."""
    from halo_forge.sft.trainer import SFTTrainer, SFTConfig
    
    print_banner()
    print(f"{GREEN}Agentic SFT Training{NC}")
    print("=" * 60)
    
    dataset = getattr(args, 'dataset', 'xlam_sft')
    max_samples = getattr(args, 'max_samples', None)
    dry_run = getattr(args, 'dry_run', False)
    
    print(f"Model: {args.model}")
    print(f"Dataset: {dataset}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print()
    
    if dry_run:
        print(f"{YELLOW}Dry run mode - validating configuration only{NC}")
        from halo_forge.sft.datasets import get_sft_dataset_spec, is_huggingface_id
        spec = get_sft_dataset_spec(dataset)
        if spec:
            print(f"{GREEN}✓{NC} Dataset: {spec.name} ({spec.huggingface_id})")
        elif is_huggingface_id(dataset):
            print(f"{GREEN}✓{NC} HuggingFace dataset: {dataset}")
        else:
            print(f"{RED}✗{NC} Unknown dataset: {dataset}")
            sys.exit(1)
        print(f"{GREEN}Configuration valid!{NC}")
        return
    
    config = SFTConfig(
        model_name=args.model,
        dataset=dataset,
        max_samples=max_samples,
        output_dir=args.output,
        num_epochs=args.epochs
    )
    
    trainer = SFTTrainer(config)
    trainer.train()


def cmd_agentic_benchmark(args):
    """Run agentic/tool calling benchmark."""
    from halo_forge.agentic import AgenticRAFTTrainer, AgenticRAFTConfig
    from halo_forge.agentic.data import XLAMLoader, GlaiveLoader
    
    print_banner()
    
    print(f"\n{GREEN}Agentic / Tool Calling Benchmark{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Limit: {args.limit}")
    
    # Load dataset
    if args.dataset == "xlam":
        loader = XLAMLoader()
    elif args.dataset == "glaive":
        loader = GlaiveLoader()
    else:
        print(f"{RED}Unknown dataset: {args.dataset}{NC}")
        print("Available: xlam, glaive")
        sys.exit(1)
    
    print(f"\n{YELLOW}Loading dataset...{NC}")
    samples = loader.load(limit=args.limit)
    print(f"Loaded {len(samples)} samples")
    
    # Create trainer for benchmark
    config = AgenticRAFTConfig(
        model_name=args.model,
    )
    trainer = AgenticRAFTTrainer(config)
    
    print(f"\n{YELLOW}Loading model...{NC}")
    trainer.load_model()
    
    print(f"\n{YELLOW}Running benchmark...{NC}")
    results = trainer.benchmark(samples, limit=args.limit)
    
    print(f"\n{GREEN}Benchmark Results{NC}")
    print("=" * 60)
    print(f"  Total samples:     {results['total']}")
    print(f"  Correct:           {results['correct']} ({results['accuracy']:.1%})")
    print(f"  JSON valid:        {results['json_valid']} ({results['json_valid_rate']:.1%})")
    print(f"  Function correct:  {results['function_correct']} ({results['function_accuracy']:.1%})")
    print(f"  Average reward:    {results['avg_reward']:.3f}")
    print(f"  False positives:   {results['false_positives']}")
    
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


def cmd_agentic_train(args):
    """Train agentic/tool calling model with RAFT."""
    from halo_forge.agentic import AgenticRAFTTrainer, AgenticRAFTConfig
    from halo_forge.agentic.data import XLAMLoader, GlaiveLoader
    
    print_banner()
    
    print(f"\n{GREEN}Agentic / Tool Calling RAFT Training{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Cycles: {args.cycles}")
    print(f"Output: {args.output}")
    
    if args.dry_run:
        print(f"\n{YELLOW}Dry run mode - validating configuration only{NC}")
        
        # Check dependencies
        print(f"\n{GREEN}✓{NC} agentic module available")
        
        # Check dataset
        from halo_forge.agentic.data import list_agentic_datasets
        if args.dataset in list_agentic_datasets():
            print(f"{GREEN}✓{NC} Dataset: {args.dataset}")
        else:
            print(f"{RED}✗{NC} Unknown dataset: {args.dataset}")
        
        print(f"\n{GREEN}Configuration valid!{NC}")
        return
    
    # Load dataset
    if args.dataset == "xlam":
        loader = XLAMLoader()
    elif args.dataset == "glaive":
        loader = GlaiveLoader()
    else:
        print(f"{RED}Unknown dataset: {args.dataset}{NC}")
        sys.exit(1)
    
    print(f"\n{YELLOW}Loading dataset...{NC}")
    samples = loader.load(limit=args.limit)
    print(f"Loaded {len(samples)} samples")
    
    # Create config
    config = AgenticRAFTConfig(
        model_name=args.model,
        num_cycles=args.cycles,
        learning_rate=args.lr,
        lr_decay_per_cycle=args.lr_decay,
        output_dir=args.output,
    )
    
    # Train
    trainer = AgenticRAFTTrainer(config)
    results = trainer.train(samples)
    
    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Final accuracy: {results.get('final_success_rate', 0):.1%}")
    print(f"Final avg reward: {results.get('final_avg_reward', 0):.3f}")
    print(f"Results saved to: {args.output}")


# The test parser and dispatch logic is inside main() at line 1598
# These are the remaining handler functions that were placed after main()

def _dispatch_commands(args):
    """Dispatch to appropriate command handler."""
    
    # Route to handler
    if args.command == 'config':
        if args.config_command == 'validate':
            cmd_config_validate(args)
    elif args.command == 'data':
        if args.data_command == 'prepare':
            cmd_data_prepare(args)
        elif args.data_command == 'generate':
            cmd_data_generate(args)
        elif args.data_command == 'validate':
            cmd_data_validate(args)
    elif args.command == 'sft':
        if args.sft_command == 'train':
            cmd_sft_train(args)
        elif args.sft_command == 'datasets':
            cmd_sft_datasets(args)
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
    elif args.command == 'inference':
        if args.inference_command == 'optimize':
            cmd_inference_optimize(args)
        elif args.inference_command == 'export':
            cmd_inference_export(args)
        elif args.inference_command == 'benchmark':
            cmd_inference_benchmark(args)
    elif args.command == 'vlm':
        if args.vlm_command == 'train':
            cmd_vlm_train(args)
        elif args.vlm_command == 'benchmark':
            cmd_vlm_benchmark(args)
        elif args.vlm_command == 'datasets':
            cmd_vlm_datasets(args)
        elif args.vlm_command == 'sft':
            cmd_vlm_sft(args)
    elif args.command == 'audio':
        if args.audio_command == 'datasets':
            cmd_audio_datasets(args)
        elif args.audio_command == 'benchmark':
            cmd_audio_benchmark(args)
        elif args.audio_command == 'train':
            cmd_audio_train(args)
        elif args.audio_command == 'sft':
            cmd_audio_sft(args)
    elif args.command == 'reasoning':
        if args.reasoning_command == 'datasets':
            cmd_reasoning_datasets(args)
        elif args.reasoning_command == 'benchmark':
            cmd_reasoning_benchmark(args)
        elif args.reasoning_command == 'train':
            cmd_reasoning_train(args)
        elif args.reasoning_command == 'sft':
            cmd_reasoning_sft(args)
    elif args.command == 'agentic':
        if args.agentic_command == 'datasets':
            cmd_agentic_datasets(args)
        elif args.agentic_command == 'benchmark':
            cmd_agentic_benchmark(args)
        elif args.agentic_command == 'train':
            cmd_agentic_train(args)
        elif args.agentic_command == 'sft':
            cmd_agentic_sft(args)
    elif args.command == 'info':
        cmd_info(args)
    elif args.command == 'test':
        cmd_test(args)


if __name__ == '__main__':
    main()

