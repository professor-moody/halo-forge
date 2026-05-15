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
import subprocess
import time
import shutil
import tempfile
import webbrowser
from pathlib import Path
from typing import List, Dict, Any, Optional

from halo_forge.capabilities import check_modality_train_capability
from halo_forge.utils.accelerator import (
    detect_gpu_kind,
    empty_accelerator_cache,
    get_device_map,
    recommended_attn_impl,
    recommended_dtype,
)

# ANSI color codes
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
BOLD = "\033[1m"
NC = "\033[0m"  # No Color

# RAFT verifier choices must stay in sync with cmd_raft_train dispatch.
RAFT_TRAIN_SUPPORTED_VERIFIERS = (
    "gcc",
    "mingw",
    "msvc",
    "humaneval",
    "mbpp",
    "rust",
    "go",
    "auto",
    "execution",
)

MODALITY_TRAIN_COMMANDS = ("vlm", "audio", "reasoning", "agentic")


def _enforce_modality_train_contract(modality: str, args) -> None:
    """Validate modality train gating/model support contract."""
    if modality not in MODALITY_TRAIN_COMMANDS:
        return

    check = check_modality_train_capability(
        modality=modality,
        model_name=getattr(args, "model", ""),
        allow_prototype_train=getattr(args, "allow_prototype_train", False),
        dry_run=getattr(args, "dry_run", False),
    )
    if not check.allowed:
        print(f"{RED}{check.message}{NC}")
        sys.exit(2)


def _enforce_training_outcome_or_exit(modality: str, summary: dict) -> None:
    """Fail non-zero when a train command produced no optimizer updates."""
    effectiveness = summary.get("effectiveness")
    if isinstance(effectiveness, dict) and effectiveness.get("verdict") == "fail":
        reason_text = ",".join(effectiveness.get("reasons") or []) or "effectiveness_failed"
        steps = int(summary.get("total_train_steps_executed", 0) or 0)
        print(
            f"{RED}TRAINING_CONTRACT_ERROR modality={modality} "
            f"reason={reason_text} total_train_steps_executed={steps}{NC}"
        )
        print(
            "Training completed but failed the effectiveness contract. "
            "Check sample filtering, optimizer updates, artifact writes, and evaluation deltas."
        )
        sys.exit(2)

    if summary.get("weights_updated", False):
        return

    reason = summary.get("final_update_reason", "no_updates")
    steps = int(summary.get("total_train_steps_executed", 0) or 0)
    print(
        f"{RED}TRAINING_CONTRACT_ERROR modality={modality} "
        f"reason={reason} total_train_steps_executed={steps}{NC}"
    )
    print(
        "Training completed without any optimizer updates. "
        "Check dataset quality, model support, and adapter configuration."
    )
    sys.exit(2)


def _print_training_run_metadata(summary: dict) -> None:
    """Print deterministic runtime metadata when available."""
    run_id = summary.get("run_id")
    if run_id:
        print(f"Run ID: {run_id}")
    if summary.get("seed") is not None:
        print(f"Seed: {summary['seed']}")
    resume_from_cycle = summary.get("resume_from_cycle")
    if resume_from_cycle is not None:
        print(f"Resume from cycle: {resume_from_cycle}")
    resumed_from = summary.get("resumed_from_checkpoint")
    if isinstance(resumed_from, dict) and resumed_from.get("model_dir"):
        print(f"Resumed checkpoint: {resumed_from['model_dir']}")


def _print_completed_training_summary(modality: str, output_dir: str, summary: dict) -> None:
    """Print canonical post-train summary details."""
    _enforce_training_outcome_or_exit(modality, summary)
    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Output: {output_dir}")
    if summary.get("final_model_path"):
        print(f"Final model: {summary['final_model_path']}")
    _print_training_run_metadata(summary)
    print(f"Train steps executed: {int(summary.get('total_train_steps_executed', 0) or 0)}")
    final_loss = summary.get("final_train_loss")
    if isinstance(final_loss, (int, float)):
        print(f"Final train loss: {final_loss:.4f}")
    effectiveness = summary.get("effectiveness")
    if isinstance(effectiveness, dict):
        print(f"Effectiveness verdict: {effectiveness.get('verdict', 'unknown')}")


# =============================================================================
# Auto-Logging System
# =============================================================================

class TeeWriter:
    """
    Write to both stdout and a log file simultaneously.
    
    Implements tee-style output without requiring external commands.
    Used for automatic logging of all training/benchmark commands.
    """
    
    def __init__(self, log_path: Path, quiet: bool = False):
        """
        Initialize TeeWriter.
        
        Args:
            log_path: Path to log file
            quiet: If True, suppress terminal output (log file only)
        """
        self.log_path = log_path
        self.quiet = quiet
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'w', buffering=1)  # Line buffered
    
    def write(self, message: str):
        """Write to both terminal and log file."""
        # Always write to log file
        self.log_file.write(message)
        
        # Write to terminal unless quiet mode
        if not self.quiet:
            self.terminal.write(message)
    
    def flush(self):
        """Flush both outputs."""
        self.log_file.flush()
        if not self.quiet:
            self.terminal.flush()
    
    def close(self):
        """Close log file and restore stdout."""
        self.log_file.close()
        sys.stdout = self.terminal
    
    def isatty(self):
        """Check if terminal is a TTY (for color support)."""
        return not self.quiet and self.terminal.isatty()


def setup_auto_logging(command_name: str, output_dir: str = "logs", quiet: bool = False) -> Path:
    """
    Configure automatic logging with timestamped file.
    
    Creates logs/ directory if needed and redirects stdout to both
    terminal and log file (unless quiet mode).
    
    Args:
        command_name: Name of command being run (e.g., 'raft_train')
        output_dir: Directory for log files (default: 'logs')
        quiet: If True, suppress terminal output
    
    Returns:
        Path to log file
    """
    from datetime import datetime
    
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{command_name}_{timestamp}.log"
    
    # Install TeeWriter
    tee = TeeWriter(log_path, quiet=quiet)
    sys.stdout = tee
    
    # Also capture stderr if not quiet
    if not quiet:
        sys.stderr = tee
    
    return log_path


def print_banner():
    """Print the halo forge banner."""
    # Disable colors when output is piped to file
    if sys.stdout.isatty():
        c, nc = CYAN, NC
    else:
        c, nc = "", ""

    width = 63
    lines = [
        "HALO-FORGE",
        "Local AI Training Framework",
        "ROCm / CUDA / MPS / MLX / CPU",
    ]
    print(f"\n{c}╔{'═' * width}╗")
    for line in lines:
        print(f"║{line.center(width)}║")
    print(f"╚{'═' * width}╝{nc}\n")


def cmd_data_synthesize(args):
    """Synthesize a training dataset (Track D1: teacher → verifier → filter)."""
    from pathlib import Path

    from halo_forge.data.synthesize import synthesize_dataset

    print_banner()
    print(f"{GREEN}halo-forge data synthesize{NC}")
    print("=" * 60)
    print(f"  seeds:    {args.seeds}")
    print(f"  output:   {args.output}")
    print(f"  teacher:  {args.teacher_model} ({args.base_url or 'default endpoint'})")
    print(f"  verifier: {args.verifier}")
    print(f"  shape:    {args.kind} (n={args.n_per_prompt})")
    print(f"  threshold: {args.threshold}")
    print()

    try:
        result = synthesize_dataset(
            seeds=args.seeds,
            output_path=Path(args.output),
            teacher_model=args.teacher_model,
            base_url=args.base_url,
            api_key=args.api_key,
            system_prompt=args.system_prompt,
            verifier_name=args.verifier,
            n_per_prompt=args.n_per_prompt,
            reward_threshold=args.threshold,
            output_kind=args.kind,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except Exception as exc:
        print(f"{RED}Synthesis failed:{NC} {exc}")
        sys.exit(1)

    pct_kept = 100.0 * result.n_accepted / max(1, result.n_generated)
    print(f"{GREEN}Done{NC} in {result.duration_seconds:.1f}s")
    print(f"  seeds:     {result.n_seeds:>8,}")
    print(f"  generated: {result.n_generated:>8,}")
    print(f"  accepted:  {result.n_accepted:>8,}  ({pct_kept:.1f}%)")
    print(f"  avg reward: {result.avg_reward:.3f}")
    print(f"  output:    {result.output_path}")


def cmd_data_score(args):
    """Score a JSONL dataset by quality and filter by threshold or top-K%
    (Track D3)."""
    from pathlib import Path

    from halo_forge.data.quality import score_file

    print_banner()
    print(f"{GREEN}halo-forge data score{NC}")
    print("=" * 60)
    print(f"  input:  {args.input}")
    print(f"  output: {args.output}")
    if args.top_k_pct is not None:
        print(f"  filter: keep top {args.top_k_pct:.0%}")
    else:
        print(f"  filter: score >= {args.threshold}")
    print()

    try:
        result = score_file(
            input_path=Path(args.input),
            output_path=Path(args.output),
            threshold=args.threshold,
            keep_top_k_pct=args.top_k_pct,
        )
    except Exception as exc:
        print(f"{RED}Score failed:{NC} {exc}")
        sys.exit(1)

    pct_kept = 100.0 * result.n_kept / max(1, result.n_input)
    print(f"{GREEN}Done{NC} in {result.duration_seconds:.2f}s")
    print(f"  input:    {result.n_input:>8,}")
    print(f"  kept:     {result.n_kept:>8,}  ({pct_kept:.1f}%)")
    print(f"  rejected: {result.n_rejected:>8,}")
    if result.reasons:
        print(f"  rejection reasons (by weakest component):")
        for reason, count in sorted(
            result.reasons.items(), key=lambda kv: kv[1], reverse=True
        ):
            print(f"    {reason:>14}: {count:>5}")


def cmd_data_dedup(args):
    """Deduplicate a JSONL dataset (Track D2)."""
    from pathlib import Path

    from halo_forge.data.dedup import dedup_file

    print_banner()
    print(f"{GREEN}halo-forge data dedup{NC}")
    print("=" * 60)
    print(f"  input:  {args.input}")
    print(f"  output: {args.output}")
    print(f"  method: {args.method}")
    if args.method == "fuzzy":
        print(f"  threshold: {args.threshold}")
    print()

    try:
        result = dedup_file(
            input_path=Path(args.input),
            output_path=Path(args.output),
            method=args.method,
            threshold=args.threshold,
            key=args.key,
            case_sensitive=args.case_sensitive,
        )
    except Exception as exc:
        print(f"{RED}Dedup failed:{NC} {exc}")
        sys.exit(1)

    pct_removed = (
        100.0 * result.n_removed / max(1, result.n_input)
    )
    print(f"{GREEN}Done{NC} in {result.duration_seconds:.2f}s")
    print(f"  input:    {result.n_input:>8,}")
    print(f"  kept:     {result.n_output:>8,}")
    print(f"  removed:  {result.n_removed:>8,}  ({pct_removed:.1f}%)")


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
    
    def _get_nested(cfg: dict, path: str):
        """Fetch nested config value by dot-delimited path."""
        current = cfg
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    if config_type == 'raft':
        required = ['output_dir', 'prompts']
    elif config_type == 'sft':
        required = ['model.name', 'data.train_file', 'training.output_dir']
    else:
        required = []
    
    # Check required fields
    for field in required:
        if "." in field:
            value = _get_nested(config, field)
            if value is None:
                errors.append(f"Missing required field: {field}")
        else:
            if field not in config:
                errors.append(f"Missing required field: {field}")
    
    # Validate specific fields
    lr = config.get('learning_rate')
    if lr is None:
        lr = _get_nested(config, "training.learning_rate")
    if lr is not None:
        if not isinstance(lr, (int, float)) or lr <= 0:
            errors.append(f"Invalid learning_rate: {lr} (must be positive number)")
        elif lr > 1e-3:
            warnings.append(f"learning_rate={lr} seems high (typical: 1e-5 to 5e-5)")
    
    decay = config.get('lr_decay_per_cycle')
    if decay is None:
        decay = _get_nested(config, "lr_decay_per_cycle")
    if decay is not None:
        if not 0 < decay <= 1:
            errors.append(f"Invalid lr_decay_per_cycle: {decay} (must be 0 < x <= 1)")
    
    cycles = config.get('num_cycles')
    if cycles is None:
        cycles = _get_nested(config, "raft.num_cycles")
    if cycles is not None:
        if not isinstance(cycles, int) or cycles < 1:
            errors.append(f"Invalid num_cycles: {cycles} (must be positive integer)")
        elif cycles > 10:
            warnings.append(f"num_cycles={cycles} is high (typical: 3-6)")
    
    temp = config.get('temperature')
    if temp is None:
        temp = _get_nested(config, "generation.temperature")
    if temp is not None:
        if not 0 < temp <= 2:
            errors.append(f"Invalid temperature: {temp} (must be 0 < x <= 2)")
    
    threshold = config.get('reward_threshold')
    if threshold is None:
        threshold = _get_nested(config, "raft.reward_threshold")
    if threshold is not None:
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
    """Run SFT training. Dispatches to the right backend (PyTorch / MLX)."""
    from halo_forge.sft.trainer import SFTConfig
    from halo_forge.sft._dispatch import get_sft_trainer
    
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
    
    # Extract all CLI arguments with defaults
    batch_size = getattr(args, 'batch_size', 2)
    learning_rate = getattr(args, 'learning_rate', 2e-4)
    warmup_ratio = getattr(args, 'warmup_ratio', 0.03)
    weight_decay = getattr(args, 'weight_decay', 0.01)
    max_grad_norm = getattr(args, 'max_grad_norm', 0.3)
    gradient_accumulation = getattr(args, 'gradient_accumulation', 16)
    lora_rank = getattr(args, 'lora_rank', 16)
    lora_alpha = getattr(args, 'lora_alpha', 32)
    lora_dropout = getattr(args, 'lora_dropout', 0.05)
    no_lora = getattr(args, 'no_lora', False)
    no_gradient_checkpointing = getattr(args, 'no_gradient_checkpointing', False)
    save_steps = getattr(args, 'save_steps', 500)
    eval_steps = getattr(args, 'eval_steps', 250)
    save_total_limit = getattr(args, 'save_total_limit', 3)
    early_stopping_patience = getattr(args, 'early_stopping_patience', 5)
    validation_split = getattr(args, 'validation_split', 0.05)
    max_seq_length = getattr(args, 'max_seq_length', 2048)
    use_dora = getattr(args, 'use_dora', False)
    use_rslora = getattr(args, 'use_rslora', False)
    init_lora_weights = getattr(args, 'init_lora_weights', 'true')
    optim = getattr(args, 'optim', 'adamw_torch')

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
        # Apply other overrides
        config.batch_size = batch_size
        config.learning_rate = learning_rate
        config.warmup_ratio = warmup_ratio
        config.weight_decay = weight_decay
        config.max_grad_norm = max_grad_norm
        config.gradient_accumulation_steps = gradient_accumulation
        config.lora_r = lora_rank
        config.lora_alpha = lora_alpha
        config.lora_dropout = lora_dropout
        config.save_steps = save_steps
        config.eval_steps = eval_steps
        config.save_total_limit = save_total_limit
        config.early_stopping_patience = early_stopping_patience
        config.validation_split = validation_split
        config.max_seq_length = max_seq_length
        config.use_dora = use_dora
        config.use_rslora = use_rslora
        config.init_lora_weights = init_lora_weights
        config.optim = optim
        config.enable_neural_accelerators = getattr(args, "enable_neural_accelerators", False)
        if no_gradient_checkpointing:
            config.gradient_checkpointing = False
    else:
        config = SFTConfig(
            model_name=args.model or "Qwen/Qwen2.5-Coder-7B",
            dataset=dataset,
            train_file=data,
            max_samples=max_samples,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation,
            lora_r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            save_steps=save_steps,
            eval_steps=eval_steps,
            save_total_limit=save_total_limit,
            early_stopping_patience=early_stopping_patience,
            validation_split=validation_split,
            max_seq_length=max_seq_length,
            gradient_checkpointing=not no_gradient_checkpointing,
            use_dora=use_dora,
            use_rslora=use_rslora,
            init_lora_weights=init_lora_weights,
            optim=optim,
            enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        )
    
    # Disable LoRA if requested (full fine-tuning)
    if no_lora:
        config.lora_r = 0  # Trainer will skip LoRA setup if rank is 0
    
    print(f"Model: {config.model_name}")
    if config.dataset:
        print(f"Dataset: {config.dataset}")
    elif config.train_file:
        print(f"Data file: {config.train_file}")
    if config.max_samples:
        print(f"Max samples: {config.max_samples}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size} (x{config.gradient_accumulation_steps} accum = {config.batch_size * config.gradient_accumulation_steps} effective)")
    print(f"Learning rate: {config.learning_rate}")
    if config.lora_r > 0:
        print(f"LoRA: rank={config.lora_r}, alpha={config.lora_alpha}")
    else:
        print(f"LoRA: disabled (full fine-tuning)")
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
    
    trainer = get_sft_trainer(config)
    summary = trainer.train(resume_from_checkpoint=args.resume)
    _print_completed_training_summary("sft", config.output_dir, summary)


def cmd_dpo_train(args):
    """Run DPO training (Track T1 / phase Q1).

    Wraps `trl.DPOTrainer` so we get the published loss-math (sigmoid, IPO,
    hinge, KTO-pair, RPO, cDPO via label-smoothing) for free; halo-forge owns
    the run-id, output_dir, training_summary contract, and recovery guidance
    so the public API + frontend treat DPO runs identically to SFT/RAFT.
    """
    from halo_forge.dpo import DPOConfig, get_dpo_trainer

    print_banner()
    print(f"{GREEN}DPO Training{NC}")
    print("=" * 60)

    dataset = getattr(args, "dataset", None)
    data = getattr(args, "data", None)
    if not dataset and not data:
        print(f"{RED}Error: Either --dataset or --data is required{NC}")
        print()
        print("Examples:")
        print("  halo-forge dpo train --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct")
        print("  halo-forge dpo train --data my_pairs.jsonl --model meta-llama/Llama-3.2-3B")
        print()
        print("Available preference datasets:")
        print("  ultrafeedback, orca_dpo, hh_rlhf, py_dpo")
        print("  Run 'halo-forge dpo datasets' to see all options")
        sys.exit(1)

    config = DPOConfig(
        model_name=args.model,
        train_file=data,
        dataset=dataset,
        max_samples=getattr(args, "max_samples", None),
        validation_split=getattr(args, "validation_split", 0.05),
        max_seq_length=getattr(args, "max_seq_length", 1024),
        max_prompt_length=getattr(args, "max_prompt_length", 512),
        beta=getattr(args, "beta", 0.1),
        loss_type=getattr(args, "loss_type", "sigmoid"),
        reference_free=getattr(args, "reference_free", False),
        label_smoothing=getattr(args, "label_smoothing", 0.0),
        output_dir=args.output,
        num_epochs=getattr(args, "epochs", 1),
        batch_size=getattr(args, "batch_size", 1),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation", 16),
        learning_rate=getattr(args, "learning_rate", 5e-6),
        warmup_ratio=getattr(args, "warmup_ratio", 0.1),
        weight_decay=getattr(args, "weight_decay", 0.0),
        max_grad_norm=getattr(args, "max_grad_norm", 1.0),
        lora_r=getattr(args, "lora_rank", 16),
        lora_alpha=getattr(args, "lora_alpha", 32),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        use_dora=getattr(args, "use_dora", False),
        use_rslora=getattr(args, "use_rslora", False),
        init_lora_weights=getattr(args, "init_lora_weights", "true"),
        optim=getattr(args, "optim", "adamw_torch"),
        save_steps=getattr(args, "save_steps", 200),
        eval_steps=getattr(args, "eval_steps", 100),
        save_total_limit=getattr(args, "save_total_limit", 3),
        load_in_4bit=getattr(args, "load_in_4bit", False),
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        gradient_checkpointing=not getattr(args, "no_gradient_checkpointing", False),
    )

    if getattr(args, "dry_run", False):
        print("Dry run: configuration validated. No training started.")
        print(f"  model={config.model_name} dataset={config.dataset or '(local)'}")
        print(f"  beta={config.beta} loss_type={config.loss_type}")
        return

    trainer = get_dpo_trainer(config)
    summary = trainer.train(resume_from_checkpoint=args.resume)
    _print_completed_training_summary("dpo", config.output_dir, summary)


def cmd_orpo_train(args):
    """Run ORPO training (Track T17b).

    Reference-free preference optimization in a single pass — combines
    NLL on the chosen response with a log-odds preference term against
    the rejected. No reference model copy in memory, no separate RM/PPO,
    competitive with DPO on chat refinement.
    """
    from halo_forge.orpo import ORPOConfig, get_orpo_trainer

    print_banner()
    print(f"{GREEN}ORPO Training{NC}")
    print("=" * 60)

    dataset = getattr(args, "dataset", None)
    data = getattr(args, "data", None)
    if not dataset and not data:
        print(f"{RED}Error: Either --dataset or --data is required{NC}")
        print()
        print("Examples:")
        print("  halo-forge orpo train --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct")
        print("  halo-forge orpo train --data my_pairs.jsonl --model meta-llama/Llama-3.2-3B")
        print()
        print("ORPO consumes the same prompt/chosen/rejected layout as DPO —")
        print("  ultrafeedback, orca_dpo, hh_rlhf, py_dpo all work.")
        sys.exit(1)

    config = ORPOConfig(
        model_name=args.model,
        train_file=data,
        dataset=dataset,
        max_samples=getattr(args, "max_samples", None),
        validation_split=getattr(args, "validation_split", 0.05),
        max_seq_length=getattr(args, "max_seq_length", 1024),
        max_prompt_length=getattr(args, "max_prompt_length", 512),
        beta=getattr(args, "beta", 0.1),
        output_dir=args.output,
        num_epochs=getattr(args, "epochs", 1),
        batch_size=getattr(args, "batch_size", 1),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation", 16),
        learning_rate=getattr(args, "learning_rate", 8e-6),
        warmup_ratio=getattr(args, "warmup_ratio", 0.1),
        weight_decay=getattr(args, "weight_decay", 0.0),
        max_grad_norm=getattr(args, "max_grad_norm", 1.0),
        lora_r=getattr(args, "lora_rank", 16),
        lora_alpha=getattr(args, "lora_alpha", 32),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        use_dora=getattr(args, "use_dora", False),
        use_rslora=getattr(args, "use_rslora", False),
        init_lora_weights=getattr(args, "init_lora_weights", "true"),
        optim=getattr(args, "optim", "adamw_torch"),
        save_steps=getattr(args, "save_steps", 200),
        eval_steps=getattr(args, "eval_steps", 100),
        save_total_limit=getattr(args, "save_total_limit", 3),
        load_in_4bit=getattr(args, "load_in_4bit", False),
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        gradient_checkpointing=not getattr(args, "no_gradient_checkpointing", False),
    )

    if getattr(args, "dry_run", False):
        print("Dry run: configuration validated. No training started.")
        print(f"  model={config.model_name} dataset={config.dataset or '(local)'}")
        print(f"  beta={config.beta}")
        return

    trainer = get_orpo_trainer(config)
    summary = trainer.train(resume_from_checkpoint=args.resume)
    _print_completed_training_summary("orpo", config.output_dir, summary)


def cmd_merge(args):
    """Merge LoRA adapters (Tracks T12 + T13).

    Two operations:
      bake     — merge a single LoRA adapter into its base. Output is a
                 standard HF checkpoint, no LoRA infrastructure required.
      combine  — combine N LoRA adapters into one (linear / ties / dare).
                 Optionally bake the combined adapter into the base in
                 the same step.
    """
    from halo_forge.inference.merge import (
        list_supported_methods,
        merge as run_merge,
    )

    print_banner()
    print(f"{GREEN}halo-forge merge{NC}")
    print("=" * 60)

    if getattr(args, "list", False):
        print("Operations: bake, combine")
        print(f"Combine methods: {', '.join(list_supported_methods())}")
        return

    print(f"  mode:   {args.mode}")
    print(f"  base:   {args.base}")
    print(f"  output: {args.output}")
    if args.mode == "bake":
        print(f"  adapter: {args.adapter}")
    else:
        print(f"  adapters: {args.adapters}")
        print(f"  weights:  {args.weights or '(uniform)'}")
        print(f"  method:   {args.method}")
        if args.bake_after_merge:
            print(f"  + bake-after-merge")
    print()

    adapter_paths = (
        [s.strip() for s in args.adapters.split(",") if s.strip()]
        if args.mode == "combine"
        else None
    )
    weights = None
    if args.mode == "combine" and args.weights:
        weights = [float(w) for w in args.weights.split(",")]

    try:
        result = run_merge(
            operation=args.mode,
            base_model=args.base,
            output_path=args.output,
            adapter_path=args.adapter if args.mode == "bake" else None,
            adapter_paths=adapter_paths,
            weights=weights,
            method=args.method,
            bake_after_merge=args.bake_after_merge,
            trust_remote_code=args.trust_remote_code,
            svd_rank=args.svd_rank,
        )
    except Exception as exc:
        print(f"{RED}Merge failed:{NC} {exc}")
        sys.exit(1)

    size_mb = (result.bytes_written or 0) / (1024 * 1024)
    print(f"{GREEN}Merged{NC} → {result.output_path}")
    print(f"  size: {size_mb:.1f} MB")
    if result.notes:
        print(f"  note: {result.notes}")


def cmd_probe(args):
    """Run a mid-training general-benchmark probe (Track V9)."""
    from pathlib import Path

    from halo_forge.eval import DEFAULT_PROBE_TASKS, MidTrainingProbe

    print_banner()
    print(f"{GREEN}halo-forge probe{NC} (mid-training general-benchmark probe)")
    print("=" * 60)

    tasks = (
        [t.strip() for t in args.tasks.split(",") if t.strip()]
        if args.tasks
        else list(DEFAULT_PROBE_TASKS)
    )
    print(f"  model:    {args.model}")
    print(f"  tasks:    {tasks}")
    print(f"  limit:    {args.limit} samples per task")
    print(f"  baseline: {args.baseline or '(no persistence)'}")
    print(f"  tolerance: {args.tolerance}")
    print()

    probe = MidTrainingProbe(
        model_name=args.model,
        baseline_path=Path(args.baseline) if args.baseline else None,
        tasks=tasks,
        limit=args.limit,
        every_n_cycles=1,  # CLI invocation is one-shot
        regression_tolerance=args.tolerance,
        backend=args.backend,
    )

    try:
        report = probe.run(cycle=args.cycle, notes=args.notes)
    except Exception as exc:
        print(f"{RED}Probe failed:{NC} {exc}")
        sys.exit(1)

    print(f"{GREEN}Done{NC} in {report.duration_seconds:.1f}s")
    if not report.has_baseline:
        print(f"{YELLOW}No baseline yet — current values written as the baseline.{NC}")
    print()
    for d in report.task_deltas:
        marker = (
            f"{RED}REGRESS{NC}" if d.regression
            else (f"{GREEN}    OK{NC}" if d.delta is not None and d.delta >= 0
                  else "       ")
        )
        delta_str = (
            f"  Δ={d.delta:+.4f}" if d.delta is not None else ""
        )
        print(
            f"  {marker} {d.task:<22} "
            f"{d.primary_metric:>22} = {d.value:>7.4f}{delta_str}"
        )

    if report.avg_delta is not None:
        print(f"\n  avg delta vs baseline: {report.avg_delta:+.4f}")
    if report.has_regression:
        regressed = report.regressed_tasks()
        print(
            f"\n{RED}Regression on {len(regressed)} task(s):{NC} "
            f"{', '.join(regressed)}"
        )
        sys.exit(2)


def cmd_eval(args):
    """Run lm-evaluation-harness benchmarks (Track V8)."""
    from pathlib import Path

    from halo_forge.eval import list_curated_task_groups, run_lm_eval

    print_banner()
    print(f"{GREEN}halo-forge eval{NC}")
    print("=" * 60)

    if getattr(args, "list_tasks", False):
        groups = list_curated_task_groups()
        print(f"Curated task groups (use any of these as --tasks <name>):")
        for name, members in groups.items():
            print(f"  {CYAN}{name:<22}{NC} {', '.join(members)}")
        print()
        print("Or pass any lm-eval task name directly (e.g. mmlu_pro_law).")
        return

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"  model:    {args.model}")
    print(f"  backend:  {args.backend}")
    print(f"  tasks:    {tasks}")
    if args.limit:
        print(f"  limit:    {args.limit} per task")
    if args.output:
        print(f"  output:   {args.output}")
    print()

    try:
        result = run_lm_eval(
            model_name=args.model,
            tasks=tasks,
            limit=args.limit,
            batch_size=args.batch_size,
            backend=args.backend,
            output_dir=Path(args.output) if args.output else None,
        )
    except Exception as exc:
        print(f"{RED}Eval failed:{NC} {exc}")
        sys.exit(1)

    print(f"{GREEN}Done{NC} in {result.duration_seconds:.1f}s")
    print(f"  tasks completed: {result.n_tasks_completed}")
    if result.n_tasks_failed:
        print(f"  tasks failed:    {result.n_tasks_failed}")
    print()
    for task in result.task_results:
        marker = f"{RED}✗{NC}" if task.error else f"{GREEN}✓{NC}"
        print(
            f"  {marker} {task.task:<24} "
            f"{task.primary_metric:>22} = "
            f"{task.value:>7.4f}"
            + (f"  (n={task.n_samples})" if task.n_samples else "")
        )
    avg = result.average_score()
    if avg is not None:
        print(f"\n  {GREEN}average primary metric:{NC} {avg:.4f}")
    if result.n_tasks_completed == 0:
        print(f"\n{RED}No eval tasks completed successfully; results are not trustworthy.{NC}")
        sys.exit(1)


def cmd_token(args):
    """Manage API tokens for the public API (Track P1)."""
    from halo_forge.auth import TokenStore, default_store_path

    store = TokenStore()
    sub = args.token_command

    print_banner()
    print(f"{GREEN}halo-forge token{NC}")
    print("=" * 60)

    if sub == "create":
        try:
            secret = store.add_token(name=args.name, note=args.note)
        except ValueError as exc:
            print(f"{RED}Error:{NC} {exc}")
            sys.exit(1)
        print(f"  name:   {args.name}")
        print(f"  store:  {default_store_path()}")
        print()
        print(f"{YELLOW}Save this token now — it won't be shown again:{NC}")
        print(f"  {secret}")
        print()
        print("Use it with:")
        print(f"  curl -H 'Authorization: Bearer {secret}' http://<host>/api/public/health")
    elif sub == "list":
        tokens = store.list_tokens()
        if not tokens:
            print("(no tokens)")
            print(f"Store: {default_store_path()}")
            print("Create one: halo-forge token create <name>")
            return
        print(f"Store: {default_store_path()}")
        print()
        print(f"  {'NAME':<24} {'CREATED':<26} {'LAST USED':<26} NOTE")
        for t in tokens:
            print(
                f"  {t.name:<24} {t.created_at:<26} "
                f"{(t.last_used_at or '—'):<26} {t.note or ''}"
            )
    elif sub == "revoke":
        ok = store.revoke(args.name)
        if ok:
            print(f"{GREEN}Revoked{NC} {args.name!r}")
        else:
            print(f"{YELLOW}No token named{NC} {args.name!r}")
            sys.exit(1)


def cmd_replay(args):
    """Show or execute the replay command for a captured run (Track T15)."""
    from pathlib import Path

    from halo_forge.replay import (
        EnvironmentFingerprint,
        compare_environments,
        load_manifest,
        hash_dataset_file,
    )

    print_banner()
    print(f"{GREEN}halo-forge replay{NC}")
    print("=" * 60)

    try:
        manifest = load_manifest(Path(args.source))
    except FileNotFoundError as exc:
        print(f"{RED}Error:{NC} {exc}")
        sys.exit(1)

    print(f"  run_id:    {manifest.run_id}")
    print(f"  modality:  {manifest.modality}")
    print(f"  model:     {manifest.model_name}")
    print(f"  seed:      {manifest.seed}")
    print(f"  timestamp: {manifest.timestamp}")
    print()

    # Environment diff vs the active host.
    current = EnvironmentFingerprint.capture().to_dict()
    diff = compare_environments(manifest.environment, current)
    if diff["matched"]:
        print(f"{GREEN}Environment matches{NC} the captured run.")
    else:
        print(f"{YELLOW}Environment differs from the captured run:{NC}")
        for d in diff["differences"][:20]:
            print(f"  {d['key']:>26}: {d['captured']!r} -> {d['current']!r}")
        if len(diff["differences"]) > 20:
            print(f"  ... {len(diff['differences']) - 20} more")

    dataset_mismatch = False
    dataset_info = manifest.dataset or {}
    if dataset_info.get("kind") == "local_file" and dataset_info.get("sha256"):
        dataset_path = Path(str(dataset_info.get("path") or ""))
        if not dataset_path.exists():
            dataset_mismatch = True
            print(f"{RED}Dataset missing:{NC} {dataset_path}")
        else:
            current_sha = hash_dataset_file(dataset_path)
            if current_sha != dataset_info.get("sha256"):
                dataset_mismatch = True
                print(f"{RED}Dataset hash mismatch:{NC} {dataset_path}")
                print(f"  captured: {dataset_info.get('sha256')}")
                print(f"  current:  {current_sha}")

    # Reconstruct the launch command.
    cmd = _reconstruct_launch_command(manifest)
    print()
    print(f"{GREEN}Reproducible launch command:{NC}")
    print(f"  {' '.join(cmd)}")

    if args.launch:
        if not diff["matched"] and not args.force:
            print()
            print(
                f"{RED}Refusing to launch{NC}: environment differs. "
                "Pass --force to launch anyway."
            )
            sys.exit(2)
        if dataset_mismatch and not args.allow_dataset_drift:
            print()
            print(
                f"{RED}Refusing to launch{NC}: dataset content differs from replay manifest. "
                "Pass --allow-dataset-drift to launch anyway."
            )
            sys.exit(2)
        print()
        print(f"{GREEN}Launching...{NC}")
        # Replay invokes our own CLI re-entrantly so env vars + logging
        # are wired the same way as a normal launch.
        import subprocess

        completed = subprocess.run(cmd, check=False)
        sys.exit(completed.returncode)


def _reconstruct_launch_command(manifest) -> list[str]:
    """Translate a replay manifest back into a ``halo-forge`` CLI invocation.

    Maps the modality + the captured config onto the right subcommand
    and forwards the small set of fields the CLI exposes. Fields the
    CLI doesn't accept stay in the manifest but aren't part of the
    command — they're already represented by the seed + config keys
    that *are* exposed.
    """
    cfg = manifest.config or {}
    modality = (manifest.modality or "").lower()

    # Map modality → subcommand. Each path picks a small set of keys
    # to forward; the full config is already on disk in replay.json
    # for anyone wanting the exhaustive picture.
    if modality == "sft":
        subcmd = ["sft", "train"]
    elif modality == "raft":
        subcmd = ["raft", "train"]
    elif modality == "dpo":
        subcmd = ["dpo", "train"]
    elif modality.startswith("dpo_mlx"):
        subcmd = ["dpo", "train"]
    elif modality == "grpo":
        subcmd = ["grpo", "train"]
    elif modality.startswith("grpo_mlx"):
        subcmd = ["grpo", "train"]
    else:
        # Unknown modality — show "halo-forge --help" and let the user
        # reconstruct manually from the config they can see.
        return ["halo-forge", "# unknown modality:", modality, "see replay.json"]

    cmd = ["halo-forge", *subcmd]
    if cfg.get("model_name"):
        cmd += ["--model", str(cfg["model_name"])]
    if cfg.get("dataset"):
        cmd += ["--dataset", str(cfg["dataset"])]
    elif cfg.get("train_file"):
        cmd += ["--data", str(cfg["train_file"])]
    if cfg.get("output_dir"):
        cmd += ["--output", str(cfg["output_dir"])]
    if cfg.get("num_epochs"):
        cmd += ["--epochs", str(cfg["num_epochs"])]
    if cfg.get("max_samples"):
        cmd += ["--max-samples", str(cfg["max_samples"])]
    if "seed" in cfg:
        # CLI doesn't have a global --seed on every subcommand yet, but
        # we capture it here so the user knows the exact value to use.
        cmd += [f"# seed={cfg['seed']}"]
    return cmd


def cmd_convert(args):
    """Convert a model between formats (Track I5).

    Wraps mlx_lm.convert / GGUFExporter / HF dtype-recast behind one
    consistent CLI vocabulary. ``--quant q4`` means "4-bit affine
    quantization with group size 64" regardless of which format you
    target; the dispatch translates to the underlying tool's args.
    """
    from halo_forge.inference.convert import (
        convert as run_convert,
        list_supported_formats,
        list_supported_quants,
    )

    print_banner()
    print(f"{GREEN}halo-forge convert{NC}")
    print("=" * 60)

    if getattr(args, "list", False):
        print(f"Supported formats: {', '.join(list_supported_formats())}")
        print(f"Supported quants:  {', '.join(list_supported_quants())}")
        return

    print(f"  source: {args.source}")
    print(f"  format: {args.format}")
    print(f"  quant:  {args.quant}")
    print(f"  output: {args.output}")
    print()
    try:
        result = run_convert(
            source=args.source,
            output_path=args.output,
            target_format=args.format,
            quantization=args.quant,
            trust_remote_code=args.trust_remote_code,
            allow_unquantized_fallback=getattr(args, "allow_unquantized_fallback", False),
        )
    except Exception as exc:
        print(f"{RED}Conversion failed:{NC} {exc}")
        sys.exit(1)

    size_mb = (result.bytes_written or 0) / (1024 * 1024)
    print(f"{GREEN}Converted{NC} -> {result.output_path}")
    print(f"  size: {size_mb:.1f} MB")
    if result.notes:
        print(f"  note: {result.notes}")

    # Track I4 — opt-in round-trip verification right after conversion.
    if getattr(args, 'verify', False):
        from halo_forge.inference.verify_export import verify_export

        print()
        print(f"{GREEN}Verifying export round-trip{NC}")
        try:
            report = verify_export(
                source_model=args.source,
                exported_path=result.output_path,
                target_format=args.format,
            )
        except NotImplementedError as exc:
            print(f"{RED}Verification unsupported:{NC} {exc}")
            sys.exit(1)
        print(f"  prompts:               {report.n_prompts}")
        print(f"  exact match rate:      {report.exact_match_rate:.2%}")
        print(f"  avg char overlap:      {report.avg_char_overlap:.3f}")
        print(f"  first-token match:     {report.avg_first_token_match:.2%}")
        print(f"  duration:              {report.duration_seconds:.1f}s")
        if report.passed:
            print(f"{GREEN}Round-trip verification passed.{NC}")
        else:
            print(f"{RED}Round-trip verification failed{NC} — exported model "
                  f"diverges from source. Failures: {len(report.failures)}/"
                  f"{report.n_prompts}")


def cmd_serve(args):
    """Run an OpenAI-compatible serving endpoint (Track I1).

    Spins up uvicorn on `--host`/`--port` (default 127.0.0.1:8001) serving
    `--model` via the active backend. The model loads lazily on the first
    request so `halo-forge serve` returns control quickly even for large
    weights — the first chat call eats the load cost.
    """
    if args.backend:
        backend_display = f"{args.backend} (forced)"
    else:
        try:
            from halo_forge.backend import get_backend

            backend_display = f"{get_backend().name} (auto)"
        except Exception:
            backend_display = "auto"

    print_banner()
    print(f"{GREEN}halo-forge serve{NC} — OpenAI-compatible endpoint")
    print("=" * 60)
    print(f"  model:               {args.model}")
    print(f"  bind:                {args.host}:{args.port}")
    print(f"  backend:             {backend_display}")
    print(f"  adapter load:        lazy (first generation request)")
    print(f"  streaming:           OpenAI SSE supported")
    print(f"  trust remote code:   {bool(args.trust_remote_code)}")
    print(f"  health:              http://{args.host}:{args.port}/health")
    print(f"  models:              http://{args.host}:{args.port}/v1/models")
    print()

    if getattr(args, "check", False):
        print(f"{GREEN}Serve preflight OK.{NC} No server started.")
        return

    print("Try:")
    print(f"  curl http://{args.host}:{args.port}/v1/models")
    print(
        f"  curl http://{args.host}:{args.port}/v1/chat/completions "
        '-H "Content-Type: application/json" '
        '-d \'{"model":"' + args.model + '","messages":[{"role":"user","content":"hi"}]}\''
    )
    print()

    import uvicorn

    from halo_forge.serving.app import create_serving_app

    app = create_serving_app(
        model_name=args.model,
        backend_name=args.backend,
        trust_remote_code=args.trust_remote_code,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_serve_public(args):
    """Run the dashboard FastAPI (Track F-* surface).

    This is the API the public_app SPA talks to — runs/eval/registry/
    playground/lineage/verifiers/metrics. Different process from the
    OpenAI-compatible inference server (`halo-forge serve`); a typical
    workstation runs both side by side on different ports.
    """
    print_banner()
    print(f"{GREEN}halo-forge serve-public{NC} — dashboard API")
    print("=" * 60)
    is_loopback = str(args.host) in {"127.0.0.1", "localhost", "::1"}
    remote_app_host = "<workstation-host>" if str(args.host) == "0.0.0.0" else str(args.host)
    print(f"  bind:        {args.host}:{args.port}")
    print(f"  health:      http://{args.host}:{args.port}/api/public/health")
    print(f"  local app:   cd public_app && npm run dev")
    print(f"  open app:    http://127.0.0.1:3000")
    print(f"  app command: halo-forge dashboard")
    if not is_loopback:
        print(f"  remote app:  cd public_app && npm run dev -- --host 0.0.0.0")
        print(f"  remote URL:  http://{remote_app_host}:3000")
    print(f"  remote auth: {'loopback bypass' if is_loopback else 'required'}")
    print()

    if getattr(args, "check", False):
        print(f"{GREEN}Dashboard API preflight OK.{NC} No server started.")
        return

    if is_loopback:
        print("Local development:")
        print("  Terminal 1: halo-forge serve-public")
        print("  Terminal 2: cd public_app && npm install && npm run dev")
    else:
        print("Remote development:")
        print("  Terminal 1a: halo-forge token create dashboard")
        print(f"  Terminal 1b: halo-forge serve-public --host {args.host}")
        print("  Terminal 2:  cd public_app && npm install && npm run dev -- --host 0.0.0.0")
        print(f"  Browser:    http://{remote_app_host}:3000")
    print()

    import uvicorn

    from halo_forge.public_api.app import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def _dashboard_is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def _dashboard_open_url(host: str, port: int) -> str:
    display_host = "<workstation-host>" if host == "0.0.0.0" else host
    return f"http://{display_host}:{port}"


def _public_app_source_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "public_app"


def _build_public_app() -> bool:
    app_dir = _public_app_source_dir()
    package_json = app_dir / "package.json"
    if not package_json.is_file():
        print(f"{RED}Dashboard source not found at {app_dir}.{NC}")
        return False
    if shutil.which("npm") is None:
        print(f"{RED}npm is required to build the dashboard assets, but it was not found.{NC}")
        print("Install Node.js/npm, then run:")
        print("  cd public_app && npm install && npm run build")
        return False

    node_modules = app_dir / "node_modules"
    if not node_modules.is_dir():
        print(f"{CYAN}Installing dashboard dependencies...{NC}")
        install = subprocess.run(["npm", "install"], cwd=app_dir)
        if install.returncode != 0:
            return False

    print(f"{CYAN}Building dashboard assets...{NC}")
    build = subprocess.run(["npm", "run", "build"], cwd=app_dir)
    return build.returncode == 0


def cmd_dashboard(args):
    """Run the user-facing dashboard as a single app command."""
    from halo_forge.public_api.app import create_app, find_frontend_dist

    host = str(args.host)
    port = int(args.port)
    is_loopback = _dashboard_is_loopback(host)
    app_url = _dashboard_open_url(host, port)
    dist = find_frontend_dist()

    print_banner()
    print(f"{GREEN}halo-forge dashboard{NC} — workstation app")
    print("=" * 60)
    print(f"  bind:        {host}:{port}")
    print(f"  open app:    {app_url}")
    print(f"  health:      http://{host}:{port}/api/public/health")
    print(f"  remote auth: {'loopback bypass' if is_loopback else 'required'}")
    if dist is None:
        print("  app assets:  missing; will build from public_app/")
        print("  build:       cd public_app && npm install && npm run build")
    else:
        print(f"  app assets:  {dist}")
    print()

    if getattr(args, "check", False):
        print(f"{GREEN}Dashboard preflight OK.{NC} No server started.")
        return

    if dist is None:
        if getattr(args, "no_build", False):
            print(f"{RED}Dashboard assets are not built.{NC}")
            print("Run `cd public_app && npm install && npm run build`, then rerun `halo-forge dashboard`.")
            sys.exit(2)
        if not _build_public_app():
            print(f"{RED}Could not build dashboard assets.{NC}")
            sys.exit(2)
        dist = find_frontend_dist()
        if dist is None:
            print(f"{RED}Dashboard build finished, but public_app/dist/index.html was not found.{NC}")
            sys.exit(2)

    if not is_loopback:
        print("Remote workstation:")
        print("  Create a dashboard token first: halo-forge token create dashboard")
        print(f"  Open from another machine:     {app_url}")
        print("  Paste the hfk_... token in Connection.")
    else:
        print(f"Open {app_url}")
    print()

    if getattr(args, "open", False) and is_loopback:
        webbrowser.open(app_url)
    elif getattr(args, "open", False):
        print("Skipping --open for non-loopback host; open the URL from the remote browser.")

    import uvicorn

    app = create_app(frontend_dist=dist, serve_frontend=True)
    uvicorn.run(app, host=host, port=port, log_level="info")


def cmd_rm_train(args):
    """Train a Bradley-Terry reward model (Track T3)."""
    from halo_forge.rm import RMConfig, get_rm_trainer

    print_banner()
    print(f"{GREEN}Reward-Model Training{NC}")
    print("=" * 60)

    dataset = getattr(args, "dataset", None)
    data = getattr(args, "data", None)
    if not dataset and not data:
        print(f"{RED}Error: Either --dataset or --data is required{NC}")
        print()
        print("Examples:")
        print("  halo-forge rm train --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct")
        print("  halo-forge rm train --data my_pairs.jsonl --model Qwen/Qwen2.5-3B-Instruct")
        sys.exit(1)

    config = RMConfig(
        model_name=args.model,
        train_file=data,
        dataset=dataset,
        max_samples=getattr(args, "max_samples", None),
        max_length=getattr(args, "max_length", 1024),
        output_dir=args.output,
        num_epochs=getattr(args, "epochs", 1),
        batch_size=getattr(args, "batch_size", 4),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation", 4),
        learning_rate=getattr(args, "learning_rate", 1e-5),
        warmup_ratio=getattr(args, "warmup_ratio", 0.05),
        weight_decay=getattr(args, "weight_decay", 0.0),
        max_grad_norm=getattr(args, "max_grad_norm", 1.0),
        lora_r=getattr(args, "lora_rank", 8),
        lora_alpha=getattr(args, "lora_alpha", 16),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        use_dora=getattr(args, "use_dora", False),
        use_rslora=getattr(args, "use_rslora", False),
        init_lora_weights=getattr(args, "init_lora_weights", "true"),
        optim=getattr(args, "optim", "adamw_torch"),
        load_in_4bit=getattr(args, "load_in_4bit", False),
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        center_rewards_coefficient=getattr(args, "center_rewards_coefficient", 0.01),
        gradient_checkpointing=not getattr(args, "no_gradient_checkpointing", False),
    )

    if getattr(args, "dry_run", False):
        print("Dry run: configuration validated. No training started.")
        print(f"  model={config.model_name} dataset={config.dataset or '(local)'}")
        print(f"  lora_rank={config.lora_r} lr={config.learning_rate}")
        return

    trainer = get_rm_trainer(config)
    summary = trainer.train(resume_from_checkpoint=args.resume)
    _print_completed_training_summary("rm", config.output_dir, summary)


def cmd_grpo_train(args):
    """Run GRPO training (Track T2 / phase Q1).

    Wraps trl.GRPOTrainer for the PyTorch path; MLX path uses an
    in-house reference-free/reference-model implementation. Verifier comes from
    the plugin registry (V1) — `--verifier execution` (default), or any
    registered short name (e.g. `llm_judge` from V2).
    """
    from halo_forge.grpo import GRPOConfig, get_grpo_trainer

    print_banner()
    print(f"{GREEN}GRPO Training{NC}")
    print("=" * 60)

    dataset = getattr(args, "dataset", None)
    data = getattr(args, "data", None)
    if not dataset and not data:
        print(f"{RED}Error: Either --dataset or --data is required{NC}")
        print()
        print("Examples:")
        print("  halo-forge grpo train --data prompts.jsonl --model Qwen/Qwen2.5-3B-Instruct --verifier execution")
        print("  halo-forge grpo train --dataset gsm8k --verifier execution --num-generations 8")
        sys.exit(1)

    config = GRPOConfig(
        model_name=args.model,
        train_file=data,
        dataset=dataset,
        max_samples=getattr(args, "max_samples", None),
        max_prompt_length=getattr(args, "max_prompt_length", 512),
        max_completion_length=getattr(args, "max_completion_length", 512),
        num_generations=getattr(args, "num_generations", 4),
        beta=getattr(args, "beta", 0.04),
        epsilon=getattr(args, "epsilon", 0.2),
        temperature=getattr(args, "temperature", 0.9),
        scale_rewards=not getattr(args, "no_scale_rewards", False),
        reference_free=getattr(args, "reference_free", False),
        verifier_name=getattr(args, "verifier", "execution"),
        reward_threshold=getattr(args, "reward_threshold", 0.0),
        output_dir=args.output,
        num_epochs=getattr(args, "epochs", 1),
        batch_size=getattr(args, "batch_size", 1),
        gradient_accumulation_steps=getattr(args, "gradient_accumulation", 16),
        learning_rate=getattr(args, "learning_rate", 1e-6),
        warmup_ratio=getattr(args, "warmup_ratio", 0.1),
        weight_decay=getattr(args, "weight_decay", 0.0),
        max_grad_norm=getattr(args, "max_grad_norm", 1.0),
        lora_r=getattr(args, "lora_rank", 16),
        lora_alpha=getattr(args, "lora_alpha", 32),
        lora_dropout=getattr(args, "lora_dropout", 0.05),
        use_dora=getattr(args, "use_dora", False),
        use_rslora=getattr(args, "use_rslora", False),
        init_lora_weights=getattr(args, "init_lora_weights", "true"),
        optim=getattr(args, "optim", "adamw_torch"),
        load_in_4bit=getattr(args, "load_in_4bit", False),
        rollout_engine=getattr(args, "rollout_engine", "auto"),
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        gradient_checkpointing=not getattr(args, "no_gradient_checkpointing", False),
    )

    if getattr(args, "dry_run", False):
        print("Dry run: configuration validated. No training started.")
        print(f"  model={config.model_name} dataset={config.dataset or '(local)'}")
        print(f"  num_generations={config.num_generations} beta={config.beta}")
        print(f"  verifier={config.verifier_name} reference_free={config.reference_free}")
        return

    trainer = get_grpo_trainer(config)
    summary = trainer.train(resume_from_checkpoint=args.resume)
    _print_completed_training_summary("grpo", config.output_dir, summary)


def cmd_dpo_datasets(args):
    """List the canonical preference datasets halo-forge ships short names for."""
    from halo_forge.dpo.datasets import list_preference_datasets

    print_banner()
    print(f"{GREEN}Available preference datasets (DPO){NC}")
    print("=" * 60)
    print()
    for ds in list_preference_datasets():
        print(f"  {CYAN}{ds.name:<16}{NC} [{ds.size_hint:>6}] {ds.description}")
        print(f"                  HuggingFace: {ds.huggingface_id}")
    print()
    print("Usage:")
    print("  halo-forge dpo train --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct")
    print()


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


def _load_prompts_jsonl(prompts_file: str) -> tuple:
    """
    Load prompts from a JSONL file.

    Returns:
        (prompts, invalid_lines)
    """
    prompts = []
    invalid_lines = []
    with open(prompts_file) as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                invalid_lines.append((line_num, str(e)))
                continue
            prompts.append(data.get('prompt', data.get('text', '')))
    return prompts, invalid_lines


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
        try:
            with open(args.config) as f:
                cfg_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error: Invalid YAML syntax in {args.config}: {e}")
            sys.exit(1)
    else:
        cfg_dict = {}
    
    # Setup verifier
    verifier_type = args.verifier or cfg_dict.get('verifier', {}).get('type', 'gcc')
    verifier_policy = "unsafe_host" if getattr(args, "unsafe_verifier_execution", False) else "sandbox"
    if verifier_policy == "unsafe_host":
        print("WARNING: --unsafe-verifier-execution runs generated-code verifiers on the host.")
    
    if verifier_type == 'gcc':
        verifier = GCCVerifier(execution_policy=verifier_policy)
    elif verifier_type == 'mingw':
        verifier = MinGWVerifier(execution_policy=verifier_policy)
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
        verifier = HumanEvalVerifier(dataset_path, execution_policy=verifier_policy)
    elif verifier_type == 'mbpp':
        dataset_path = cfg_dict.get('verifier', {}).get('dataset', 'data/rlvr/mbpp_train_full.jsonl')
        verifier = MBPPVerifier(dataset_path, execution_policy=verifier_policy)
    elif verifier_type == 'rust' or verifier_type == 'cargo':
        from halo_forge.rlvr.verifiers import RustVerifier
        run_after = cfg_dict.get('verifier', {}).get('run_after_compile', False)
        verifier = RustVerifier(run_after_compile=run_after, execution_policy=verifier_policy)
    elif verifier_type == 'go':
        from halo_forge.rlvr.verifiers import GoVerifier
        run_after = cfg_dict.get('verifier', {}).get('run_after_compile', False)
        verifier = GoVerifier(run_after_compile=run_after, execution_policy=verifier_policy)
    elif verifier_type == 'auto':
        from halo_forge.rlvr.verifiers import MultiLanguageVerifier
        run_after = cfg_dict.get('verifier', {}).get('run_after_compile', False)
        binary_cache = cfg_dict.get('verifier', {}).get('binary_cache_dir')
        verifier = MultiLanguageVerifier(
            run_after_compile=run_after,
            binary_cache_dir=binary_cache,
            execution_policy=verifier_policy,
        )
    elif verifier_type == 'execution':
        from halo_forge.rlvr.verifiers import ExecutionVerifier
        test_cases = cfg_dict.get('verifier', {}).get('test_cases', [])
        match_mode = cfg_dict.get('verifier', {}).get('match_mode', 'exact')
        verifier = ExecutionVerifier(
            test_cases=test_cases,
            match_mode=match_mode,
            execution_policy=verifier_policy,
        )
    else:
        print(f"Unknown verifier: {verifier_type}")
        print(f"Available: {', '.join(RAFT_TRAIN_SUPPORTED_VERIFIERS)}")
        sys.exit(1)
    
    # Create config
    raft_cfg = cfg_dict.get('raft', {}) if isinstance(cfg_dict.get('raft', {}), dict) else {}
    generation_cfg = cfg_dict.get('generation', {}) if isinstance(cfg_dict.get('generation', {}), dict) else {}
    training_cfg = cfg_dict.get('training', {}) if isinstance(cfg_dict.get('training', {}), dict) else {}
    lora_cfg = cfg_dict.get('lora', {}) if isinstance(cfg_dict.get('lora', {}), dict) else {}

    def _prefer_nested(nested, top_level, default, name: str):
        if nested is not None and top_level is not None and nested != top_level:
            print(f"[!] Using raft.{name} ({nested}) over top-level {name} ({top_level})")
        if nested is not None:
            return nested
        if top_level is not None:
            return top_level
        return default

    keep_percent = getattr(args, 'keep_percent', None)
    if keep_percent is None:
        keep_percent = _prefer_nested(
            raft_cfg.get('keep_top_percent'),
            cfg_dict.get('keep_top_percent'),
            0.5,
            "keep_top_percent"
        )

    reward_threshold = getattr(args, 'reward_threshold', None)
    if reward_threshold is None:
        reward_threshold = _prefer_nested(
            raft_cfg.get('reward_threshold'),
            cfg_dict.get('reward_threshold'),
            0.5,
            "reward_threshold"
        )
    
    curriculum = getattr(args, 'curriculum', None) or cfg_dict.get('curriculum_strategy', 'none')
    curriculum_stats = getattr(args, 'curriculum_stats', None) or cfg_dict.get('curriculum_stats_path', None)
    curriculum_start = getattr(args, 'curriculum_start', None) or cfg_dict.get('curriculum_progressive_start', 0.2)
    curriculum_increment = getattr(args, 'curriculum_increment', None) or cfg_dict.get('curriculum_progressive_increment', 0.2)
    reward_shaping = getattr(args, 'reward_shaping', None) or cfg_dict.get('reward_shaping_strategy', 'fixed')
    system_prompt = getattr(args, 'system_prompt', None) or cfg_dict.get('system_prompt', 'You are an expert Windows systems programmer.')
    lr_decay = getattr(args, 'lr_decay', None) or cfg_dict.get('lr_decay_per_cycle', 0.85)
    min_lr = getattr(args, 'min_lr', None) or cfg_dict.get('min_lr', 1e-6)
    
    # New generation parameters
    samples_per_prompt = getattr(args, 'samples_per_prompt', None)
    if samples_per_prompt is None:
        samples_per_prompt = raft_cfg.get('samples_per_prompt', 8)

    temperature = getattr(args, 'temperature', None)
    if temperature is None:
        temperature = generation_cfg.get('temperature', 0.7)

    max_new_tokens = getattr(args, 'max_new_tokens', None)
    if max_new_tokens is None:
        max_new_tokens = generation_cfg.get('max_new_tokens', 1024)

    min_samples = getattr(args, 'min_samples', None)
    if min_samples is None:
        min_samples = raft_cfg.get('min_samples')
    
    # Training hyperparameters from training.* section
    learning_rate = getattr(args, 'learning_rate', None)
    if learning_rate is None:
        learning_rate = training_cfg.get('learning_rate') or training_cfg.get('base_learning_rate') or 5e-5
    
    batch_size = getattr(args, 'batch_size', None)
    if batch_size is None:
        batch_size = training_cfg.get('batch_size') or 2
    
    gradient_accumulation = getattr(args, 'gradient_accumulation', None)
    if gradient_accumulation is None:
        gradient_accumulation = training_cfg.get('gradient_accumulation_steps') or 16
    
    warmup_steps = getattr(args, 'warmup_steps', None)
    if warmup_steps is None:
        warmup_steps = training_cfg.get('warmup_steps') or 10
    
    # Support training.lr_decay_factor as alias for lr_decay_per_cycle
    if lr_decay == 0.85:  # default value, check if config has different
        config_lr_decay = training_cfg.get('lr_decay_factor')
        if config_lr_decay is not None:
            lr_decay = config_lr_decay
    
    # LoRA configuration from lora.* section
    lora_rank = getattr(args, 'lora_rank', None)
    if lora_rank is None:
        lora_rank = lora_cfg.get('r') or 16
    
    lora_alpha = getattr(args, 'lora_alpha', None)
    if lora_alpha is None:
        lora_alpha = lora_cfg.get('alpha') or 32
    
    lora_dropout = lora_cfg.get('dropout') or 0.05
    
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
    
    num_cycles = args.cycles
    if num_cycles is None:
        num_cycles = _prefer_nested(
            raft_cfg.get('num_cycles'),
            cfg_dict.get('num_cycles'),
            3,
            "num_cycles"
        )

    config = RAFTConfig(
        base_model=base_model,
        sft_checkpoint=sft_checkpoint,
        output_dir=args.output or cfg_dict.get('output_dir', 'models/raft'),
        num_cycles=num_cycles,
        keep_top_percent=keep_percent,
        reward_threshold=reward_threshold,
        allow_compile_only_training=getattr(args, 'allow_compile_only_training', False),
        curriculum_strategy=curriculum,
        curriculum_stats_path=curriculum_stats,
        curriculum_progressive_start=curriculum_start,
        curriculum_progressive_increment=curriculum_increment,
        reward_shaping_strategy=reward_shaping,
        system_prompt=system_prompt,
        # Training hyperparameters
        learning_rate=learning_rate,
        train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        warmup_steps=warmup_steps,
        lr_decay_per_cycle=lr_decay,
        min_lr=min_lr,
        # LoRA configuration
        lora_r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # Generation
        samples_per_prompt=samples_per_prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        min_samples_per_cycle=min_samples
    )
    
    # Load prompts
    prompts = []
    prompts_file = args.prompts or cfg_dict.get('prompts')
    if prompts_file:
        prompts, invalid_lines = _load_prompts_jsonl(prompts_file)
        if invalid_lines:
            print(f"Error: Invalid JSONL in prompts file: {prompts_file}")
            for line_num, err in invalid_lines[:5]:
                print(f"  Line {line_num}: {err}")
            if len(invalid_lines) > 5:
                print(f"  ... {len(invalid_lines) - 5} more invalid lines")
            sys.exit(1)
    
    if not prompts:
        print("Error: No prompts provided")
        print("Use --prompts or set in config")
        sys.exit(1)
    
    # Phase 5: when --accelerator mlx is requested we have two paths.
    # Default (5b): MLXRAFTTrainer — rollout + verify + SFT all on MLX.
    # Opt-in (5a): --rollout-only — keep the PyTorch RAFTTrainer but swap
    # in MLXRolloutGenerator for the rollout step. The 5a hybrid is useful
    # when the user has an existing PyTorch checkpoint they want to keep
    # training but wants fast Apple Silicon rollouts.
    if getattr(args, 'accelerator', 'auto') == 'mlx':
        mlx_model = getattr(args, 'rollout_model', None) or args.model
        if getattr(args, 'rollout_only', False):
            from halo_forge.rlvr.mlx_rollout import MLXRolloutGenerator
            print(f"[mlx-5a] Hybrid mode: MLX rollouts ({mlx_model}) + PyTorch policy update ({args.model})")
            trainer = RAFTTrainer(
                verifier=verifier,
                config=config,
                rollout_generator=MLXRolloutGenerator(mlx_model),
            )
            trainer.run(prompts, num_cycles=config.num_cycles)
        else:
            from halo_forge.rlvr.mlx_raft_trainer import MLXRAFTTrainer
            print(f"[mlx-5b] Native MLX RAFT: rollout + verify + SFT on MLX ({mlx_model})")
            mlx_trainer = MLXRAFTTrainer(
                verifier=verifier,
                config=config,
                rollout_model=mlx_model,
            )
            mlx_trainer.run(prompts, num_cycles=config.num_cycles)
    else:
        # Track I6 — pluggable rollout engine. The torch fallback is the
        # historical default; vllm is the CUDA/ROCm fast path; mlx is the
        # Apple Silicon equivalent (vLLM doesn't run on MLX, but
        # mlx_lm.generate is the same throughput story on Apple's
        # unified-memory hardware).
        rollout_engine = getattr(args, 'rollout_engine', 'auto')
        if rollout_engine == 'vllm':
            from halo_forge.rlvr.vllm_rollout import VLLMRolloutGenerator
            print(f"[i6] Using vLLM rollouts for fast continuous-batched generation")
            trainer = RAFTTrainer(
                verifier=verifier,
                config=config,
                rollout_generator=VLLMRolloutGenerator(args.model),
            )
        elif rollout_engine == 'mlx':
            from halo_forge.rlvr.mlx_rollout import MLXRolloutGenerator
            mlx_model = getattr(args, 'rollout_model', None) or args.model
            print(f"[i6] Using MLX rollouts ({mlx_model}) — Apple Silicon fast path")
            trainer = RAFTTrainer(
                verifier=verifier,
                config=config,
                rollout_generator=MLXRolloutGenerator(mlx_model),
            )
        else:
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
    verifier_policy = "unsafe_host" if getattr(args, "unsafe_verifier_execution", False) else "sandbox"
    if verifier_policy == "unsafe_host":
        print("WARNING: --unsafe-verifier-execution runs generated-code verifiers on the host.")
    if args.verifier == 'gcc':
        verifier = GCCVerifier(execution_policy=verifier_policy)
    elif args.verifier == 'mingw':
        verifier = MinGWVerifier(execution_policy=verifier_policy)
    elif args.verifier == 'rust':
        verifier = RustVerifier(cross_compile=getattr(args, 'cross_compile', False), execution_policy=verifier_policy)
    elif args.verifier == 'go':
        verifier = GoVerifier(cross_compile=getattr(args, 'cross_compile', False), execution_policy=verifier_policy)
    elif args.verifier == 'dotnet':
        verifier = DotNetVerifier(execution_policy=verifier_policy)
    elif args.verifier == 'powershell':
        verifier = PowerShellVerifier()
    elif args.verifier in ('auto', 'multi'):
        # Auto-detect language from code
        verifier = MultiLanguageVerifier(
            run_after_compile=getattr(args, 'run_after_compile', False),
            execution_policy=verifier_policy,
        )
    elif args.verifier in ('humaneval', 'python'):
        from halo_forge.rlvr.verifiers import HumanEvalVerifier
        dataset_path = getattr(args, 'dataset', None) or 'data/rlvr/humaneval_full.jsonl'
        verifier = HumanEvalVerifier(dataset_path, execution_policy=verifier_policy)
    elif args.verifier == 'mbpp':
        from halo_forge.rlvr.verifiers import MBPPVerifier
        dataset_path = getattr(args, 'dataset', None) or 'data/rlvr/mbpp_train_full.jsonl'
        verifier = MBPPVerifier(dataset_path, execution_policy=verifier_policy)
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
        print("Available verifiers: gcc, mingw, msvc, rust, go, dotnet, powershell, auto, humaneval, mbpp, python")
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


def cmd_benchmark_eval(args):
    """Run code evaluation on standard benchmarks (HumanEval, MBPP, LiveCodeBench)."""
    from pathlib import Path
    from halo_forge.benchmark import run_benchmark
    
    print_banner()
    print(f"{GREEN}Code Benchmark: {args.benchmark}{NC}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Benchmark: {args.benchmark}")
    if args.limit:
        print(f"Limit: {args.limit}")
    
    run_after_compile = getattr(args, 'run_after_compile', False)
    is_compiled_language = args.benchmark in {"cpp", "rust", "go"}
    if is_compiled_language:
        mode = "MVR (full verification)" if run_after_compile else "MVP (compile-only)"
        print(f"Mode: {mode}")
        if getattr(args, 'language', None):
            print(f"Language: {args.language}")
        if getattr(args, 'verifier', None):
            print(f"Verifier: {args.verifier}")
    else:
        print("Mode: dataset-faithful Python benchmark evaluation")
    print("=" * 60)
    
    output = Path(args.output) if args.output else None
    
    result = run_benchmark(
        model=args.model,
        benchmark=args.benchmark,
        limit=args.limit,
        output=output,
        samples_per_prompt=getattr(args, 'samples_per_prompt', 5),
        run_after_compile=run_after_compile,
        language=getattr(args, 'language', None),
        verifier=getattr(args, 'verifier', None),
    )
    
    if 'error' in result:
        print(f"\n{RED}Error: {result['error']}{NC}")
        sys.exit(1)
    
    print(f"\n{GREEN}Results:{NC}")
    execution_path = result.get("execution_path")
    if execution_path:
        print(f"  execution_path: {execution_path}")
    for key, value in result.get('metrics', {}).items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nSamples evaluated: {result.get('samples', 0)}")
    
    if output:
        print(f"Results saved to: {output}")


def cmd_plot_training(args):
    """Generate charts from TensorBoard training logs."""
    from pathlib import Path
    
    # Import the plotting module
    try:
        from scripts.plot_training import (
            load_training_metrics,
            load_multiple_runs,
            generate_all_charts,
            plot_loss_curve,
            plot_learning_rate,
            plot_grad_norm,
            plot_training_summary,
            plot_comparison,
        )
    except ImportError:
        # Fallback: run as subprocess
        import subprocess
        cmd = [sys.executable, "scripts/plot_training.py"] + args.log_dirs
        if args.output:
            cmd.extend(["--output", args.output])
        if args.compare:
            cmd.append("--compare")
        if args.only:
            cmd.extend(["--only", args.only])
        if args.name:
            cmd.extend(["--name", args.name])
        subprocess.run(cmd)
        return
    
    log_dirs = [Path(d) for d in args.log_dirs]
    
    # Comparison mode
    if args.compare and len(log_dirs) > 1:
        runs = load_multiple_runs(log_dirs)
        if not runs:
            print("Error: No valid training runs found")
            return
        
        output_dir = Path(args.output) if args.output else Path("figures/comparison")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating comparison charts in {output_dir}...")
        plot_comparison(runs, output_dir / "loss_comparison.png", 'train_loss')
        plot_comparison(runs, output_dir / "lr_comparison.png", 'learning_rate')
        print(f"\nDone! Comparison charts saved to {output_dir}")
        return
    
    # Single run mode
    log_dir = log_dirs[0]
    
    try:
        metrics = load_training_metrics(log_dir, name=args.name)
    except Exception as e:
        print(f"Error loading training logs: {e}")
        return
    
    print(f"Loaded {metrics.name}: {metrics.total_steps} steps, final loss {metrics.final_loss:.4f}")
    
    output_dir = Path(args.output) if args.output else log_dir.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating charts in {output_dir}...")
    
    if args.only == 'loss':
        plot_loss_curve(metrics, output_dir / "loss_curve.png")
    elif args.only == 'lr':
        plot_learning_rate(metrics, output_dir / "learning_rate.png")
    elif args.only == 'grad':
        plot_grad_norm(metrics, output_dir / "grad_norm.png")
    elif args.only == 'summary':
        plot_training_summary(metrics, output_dir / "training_summary.png")
    else:
        generate_all_charts(metrics, output_dir)
    
    print(f"\nDone! Charts saved to {output_dir}")


def cmd_plot_benchmarks(args):
    """Generate charts from benchmark results."""
    import subprocess
    cmd = [sys.executable, "scripts/plot_benchmarks.py", args.results_dir]
    if args.output:
        cmd.extend(["--output", args.output])
    subprocess.run(cmd)


def cmd_info(args):
    """Show hardware info."""
    def _backend_info() -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": "unknown",
            "device": "unknown",
            "supports_neural_accelerators": False,
            "chip": None,
        }
        try:
            from halo_forge.backend import get_backend

            backend = get_backend()
            info["name"] = backend.name
            info["device"] = backend.device()
            info["supports_neural_accelerators"] = bool(
                getattr(backend.capabilities, "supports_neural_accelerators", False)
            )
        except Exception:
            pass

        try:
            from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry

            chip = AppleSiliconTelemetry._detect_chip_info(
                AppleSiliconTelemetry._detect_device_name()
            )
            if chip is not None:
                info["chip"] = chip.to_dict()
        except Exception:
            pass
        if not info.get("chip") or str(info.get("chip", {}).get("brand", "")).lower() in {"arm", "apple silicon"}:
            try:
                from halo_forge.backend.mlx_readiness import _metal_device, _optional_int
                from halo_forge.utils.apple_chip import parse_chip_brand, with_gpu_cores

                metal = _metal_device()
                if metal:
                    parsed = with_gpu_cores(
                        parse_chip_brand(str(metal.get("model") or "")),
                        _optional_int(metal.get("gpu_cores")),
                    )
                    if parsed is not None:
                        info["chip"] = parsed.to_dict()
            except Exception:
                pass
        return info

    def _hardware_lines(backend_info: dict[str, Any]) -> list[tuple[str, str]]:
        lines: list[tuple[str, str]] = [
            ("info", f"Backend: {backend_info['name']} ({backend_info['device']})")
        ]
        chip = backend_info.get("chip")
        if isinstance(chip, dict):
            gpu = (
                f", gpu_cores={chip['gpu_cores']}"
                if chip.get("gpu_cores") is not None
                else ""
            )
            lines.append(
                (
                    "info",
                    f"Chip: {chip['brand']} (gen={chip['generation']}, "
                    f"variant={chip.get('variant') or 'base'}{gpu})",
                )
            )
        if backend_info["name"] in {"mps", "mlx"}:
            lines.append(("success", "Apple Silicon accelerator detected"))
        elif backend_info["name"] in {"cuda", "rocm", "rocm_gfx1151"}:
            lines.append(("success", f"{backend_info['name'].upper()} accelerator detected"))
        elif backend_info["name"] == "cpu":
            lines.append(("warning", "No hardware accelerator backend is active"))
        if backend_info["name"] in {"mps", "mlx"}:
            lines.append(("info", "PyTorch CUDA/ROCm not active; using Apple backend"))
        lines.append(
            (
                "info",
                "Neural Accelerators: "
                + (
                    "available"
                    if backend_info.get("supports_neural_accelerators")
                    else "unavailable"
                ),
            )
        )
        return lines

    backend_info = _backend_info()

    try:
        from halo_forge import ui

        ui.print_banner()
        try:
            import torch

            if torch.cuda.is_available() and backend_info["name"] in {"cuda", "rocm", "rocm_gfx1151"}:
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                memory_gb = props.total_memory / 1e9

                rocm_version = ""
                if hasattr(torch.version, 'hip'):
                    rocm_version = torch.version.hip or ""

                ui.print_hardware_info(
                    gpu_name=gpu_name,
                    memory_gb=memory_gb,
                    rocm_version=rocm_version,
                    pytorch_version=torch.__version__,
                )
        except Exception:
            pass

        for level, line in _hardware_lines(backend_info):
            if level == "success":
                ui.print_success(line)
            elif level == "warning":
                ui.print_warning(line)
            else:
                ui.print_info(line)
    except ImportError:
        print_banner()
        for level, line in _hardware_lines(backend_info):
            prefix = {"success": "[OK]", "warning": "[!]", "info": ">"}[level]
            print(f"{prefix} {line}")


def cmd_doctor(args):
    """Run environment readiness checks."""
    if args.doctor_command != "mlx":
        print(f"Unknown doctor check: {args.doctor_command}", file=sys.stderr)
        sys.exit(1)

    from halo_forge.backend.mlx_readiness import check_mlx_readiness

    readiness = check_mlx_readiness()
    payload = readiness.to_dict()
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        versions = payload.get("package_versions") or {}
        chip = payload.get("chip") or {}
        metal = payload.get("metal_device") or {}
        print(f"MLX readiness: {payload['status']}")
        print(f"  executable: {payload['executable']}")
        print(f"  packages: mlx={versions.get('mlx') or 'missing'}, mlx-lm={versions.get('mlx-lm') or 'missing'}")
        if payload.get("macos_version"):
            print(f"  macOS: {payload['macos_version']}")
        if chip:
            print(f"  chip: {chip.get('brand') or chip.get('raw_brand') or chip}")
        if metal:
            print(f"  Metal device: {metal.get('model') or metal}")
        for label in ("warnings", "errors", "suggested_fixes"):
            values = [str(item) for item in payload.get(label, []) if item]
            if values:
                print(f"  {label.replace('_', ' ')}:")
                for value in values:
                    print(f"    - {value}")
        if readiness.executable:
            print()
            print("Next:")
            print(
                "  halo-forge --accelerator mlx sft train "
                "--model mlx-community/Qwen2.5-0.5B-Instruct-bf16 "
                "--dataset codealpaca --output models/sft_mlx_quickstart"
            )

    if readiness.executable:
        return
    sys.exit(2 if readiness.status == "unavailable" else 1)


def cmd_models(args):
    """List and inspect curated upstream/base models."""
    from halo_forge.models.catalog import CATALOG_VERSION, get_model, list_models

    if args.models_command == "show":
        item = get_model(args.model_id)
        if item is None:
            print(f"Unknown model: {args.model_id}", file=sys.stderr)
            sys.exit(1)
        if getattr(args, "json", False):
            print(json.dumps(item, indent=2, sort_keys=True))
            return
        print(f"{item['id']}")
        print(f"  label: {item['label']}")
        print(f"  provider: {item['provider']}")
        print(f"  family: {item['family']} ({item['parameter_count']})")
        print(f"  status: {item['status']}")
        print(f"  memory: {item['memory_tier']}")
        print(f"  modalities: {', '.join(item['modalities'])}")
        print(f"  trainers: {', '.join(item['trainer_support'])}")
        print(f"  backends: {', '.join(item['backend_support'])}")
        if item.get("mlx_variant"):
            print(f"  mlx_variant: {item['mlx_variant']}")
        print(f"  use: {item['recommended_use']}")
        caveats = item.get("known_caveats") or []
        if caveats:
            print("  caveats:")
            for caveat in caveats:
                print(f"    - {caveat}")
        return

    filters = {
        "mode": getattr(args, "mode", None),
        "backend": getattr(args, "backend", None),
        "modality": getattr(args, "modality", None),
        "provider": getattr(args, "provider", None),
        "status": getattr(args, "status", None),
        "memory_tier": getattr(args, "memory_tier", None),
    }
    items = list_models({k: v for k, v in filters.items() if v})
    if getattr(args, "json", False):
        print(json.dumps({"catalog_version": CATALOG_VERSION, "items": items}, indent=2, sort_keys=True))
        return
    print(f"halo-forge model catalog {CATALOG_VERSION} ({len(items)} models)")
    print()
    print(f"{'MODEL':<42} {'STATUS':<12} {'MEM':<7} {'TRAINERS':<22} USE")
    print("-" * 110)
    for item in items:
        trainers = ",".join(item["trainer_support"][:4])
        if len(item["trainer_support"]) > 4:
            trainers += ",..."
        print(
            f"{item['id']:<42.42} {item['status']:<12} {item['memory_tier']:<7} "
            f"{trainers:<22.22} {item['recommended_use']}"
        )


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
        """Test accelerator availability (CUDA/ROCm or Apple Silicon MPS).

        On CPU-only hosts, logs a warning and returns True so smoke tests can
        still proceed against tiny models. The trainer paths themselves still
        prefer an accelerator and tune accordingly.
        """
        import torch
        from halo_forge.utils.accelerator import detect_gpu_kind, GPU_KIND_CPU, GPU_KIND_MPS

        kind = detect_gpu_kind()
        if kind == GPU_KIND_CPU:
            self.log("WARNING: no accelerator detected; running on CPU. Training will be slow.")
            return True

        if kind == GPU_KIND_MPS:
            self.log("Accelerator: Apple Silicon (MPS). Memory probe unavailable on this backend.")
            return True

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
            dtype=recommended_dtype(),
            device_map=get_device_map(),
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

    def test_modality_fixtures(self) -> bool:
        """Validate deterministic modality fixture pack shape."""
        fixture_dir = Path("tests/fixtures/modality")
        required_files = (
            "vlm_samples.jsonl",
            "audio_samples.jsonl",
            "reasoning_samples.jsonl",
            "agentic_samples.jsonl",
        )
        if not fixture_dir.exists():
            raise RuntimeError(f"Missing fixture directory: {fixture_dir}")

        for filename in required_files:
            path = fixture_dir / filename
            if not path.exists():
                raise RuntimeError(f"Missing fixture file: {path}")
            with open(path, encoding="utf-8") as f:
                first_line = f.readline().strip()
            if not first_line:
                raise RuntimeError(f"Fixture file is empty: {path}")
            try:
                json.loads(first_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL fixture in {path}: {exc}") from exc
        return True

    def test_modality_train_smoke(self) -> Dict[str, Dict[str, Any]]:
        """Run tiny deterministic modality train-smoke flows."""
        from types import SimpleNamespace

        from halo_forge.modality_baseline import build_modality_entries_from_runs
        from halo_forge.training_contracts import build_cycle_summary
        from halo_forge.vlm.trainer import VLMRAFTConfig, VLMRAFTTrainer, VLMSampleResult
        from halo_forge.audio.trainer import AudioRAFTConfig, AudioRAFTTrainer, AudioRAFTCycleResult
        from halo_forge.reasoning.trainer import ReasoningRAFTConfig, ReasoningRAFTTrainer
        from halo_forge.reasoning.data import MathSample
        from halo_forge.agentic.trainer import AgenticRAFTConfig, AgenticRAFTTrainer, AgenticRAFTCycleResult

        class _FakeSaveComponent:
            def __init__(self, marker: str):
                self.marker = marker

            def save_pretrained(self, target_dir: str):
                target = Path(target_dir) / f"{self.marker}.txt"
                target.write_text(self.marker, encoding="utf-8")

        with tempfile.TemporaryDirectory(prefix="halo_forge_modality_test_") as tmp_dir:
            output_root = Path(tmp_dir)
            run_payloads: Dict[str, Dict[str, Any]] = {}

            # VLM smoke
            vlm = VLMRAFTTrainer(VLMRAFTConfig(num_cycles=1, output_dir=str(output_root / "vlm"), seed=7))
            vlm._setup = lambda: (
                setattr(
                    vlm,
                    "adapter",
                    SimpleNamespace(
                        model=_FakeSaveComponent("vlm_model"),
                        tokenizer=_FakeSaveComponent("vlm_tokenizer"),
                        processor=_FakeSaveComponent("vlm_processor"),
                        cleanup=lambda: None,
                    ),
                ),
                setattr(vlm, "verifier", SimpleNamespace(cleanup=lambda: None)),
            )
            vlm.generate_samples = lambda prompts, spp: [
                VLMSampleResult(
                    image="fixture.png",
                    prompt="describe",
                    completion="answer",
                    ground_truth="answer",
                    reward=1.0,
                    success=True,
                    details={},
                )
            ]
            vlm.filter_samples = lambda samples: samples
            vlm.train_on_samples = lambda samples, cycle: {
                "train_steps_executed": 1,
                "train_loss": 0.1,
                "weights_updated": True,
                "update_reason": "updated",
                "optimizer_steps": 1,
                "skipped_batches_non_finite": 0,
            }
            vlm_summary = vlm.train(prompts=[SimpleNamespace(image="fixture.png", prompt="describe", ground_truth="answer")])
            if not vlm_summary.get("final_model_path"):
                raise RuntimeError("VLM smoke did not emit final_model_path")
            run_payloads["vlm"] = {"summary": vlm_summary, "output_dir": output_root / "vlm"}

            # Audio smoke
            audio = AudioRAFTTrainer(AudioRAFTConfig(num_cycles=1, output_dir=str(output_root / "audio"), seed=7))
            audio.adapter = SimpleNamespace(
                model=_FakeSaveComponent("audio_model"),
                tokenizer=_FakeSaveComponent("audio_tokenizer"),
                processor=_FakeSaveComponent("audio_processor"),
            )
            audio._init_adapter = lambda: None
            audio._init_verifier = lambda: None
            audio._train_cycle = lambda cycle, samples: AudioRAFTCycleResult(
                cycle=cycle,
                samples_generated=1,
                samples_verified=1,
                samples_kept=1,
                average_reward=1.0,
                learning_rate=1e-5,
                metrics=build_cycle_summary(
                    cycle=cycle,
                    learning_rate=1e-5,
                    samples_seen=1,
                    samples_kept=1,
                    cycle_duration_seconds=0.01,
                    update_metrics={
                        "train_steps_executed": 1,
                        "train_loss": 0.1,
                        "weights_updated": True,
                        "update_reason": "updated",
                        "optimizer_steps": 1,
                        "skipped_batches_non_finite": 0,
                    },
                ),
            )
            audio.train(samples=[SimpleNamespace()])
            if not audio.training_summary.get("final_model_path"):
                raise RuntimeError("Audio smoke did not emit final_model_path")
            run_payloads["audio"] = {
                "summary": audio.training_summary,
                "output_dir": output_root / "audio",
            }

            # Reasoning smoke
            reasoning = ReasoningRAFTTrainer(
                ReasoningRAFTConfig(num_cycles=1, output_dir=str(output_root / "reasoning"), seed=7)
            )
            reasoning.train_cycle = lambda samples, cycle: build_cycle_summary(
                cycle=cycle,
                learning_rate=1e-5,
                samples_seen=1,
                samples_kept=1,
                cycle_duration_seconds=0.01,
                update_metrics={
                    "train_steps_executed": 1,
                    "train_loss": 0.1,
                    "weights_updated": True,
                    "update_reason": "updated",
                    "optimizer_steps": 1,
                    "skipped_batches_non_finite": 0,
                },
                extra={"accuracy": 1.0, "avg_reward": 1.0},
            )
            reasoning_summary = reasoning.train(samples=[MathSample(question="1+1", answer="2")])
            if not reasoning_summary.get("final_model_path"):
                raise RuntimeError("Reasoning smoke did not emit final_model_path")
            run_payloads["reasoning"] = {
                "summary": reasoning_summary,
                "output_dir": output_root / "reasoning",
            }

            # Agentic smoke
            agentic = AgenticRAFTTrainer(
                AgenticRAFTConfig(num_cycles=1, output_dir=str(output_root / "agentic"), seed=7)
            )
            agentic.model = _FakeSaveComponent("agentic_model")
            agentic.tokenizer = _FakeSaveComponent("agentic_tokenizer")
            agentic._run_cycle = lambda samples, cycle: AgenticRAFTCycleResult(
                cycle=cycle,
                total_samples=1,
                verified_samples=1,
                avg_reward=1.0,
                success_rate=1.0,
                training_samples=1,
                metrics=build_cycle_summary(
                    cycle=cycle,
                    learning_rate=1e-5,
                    samples_seen=1,
                    samples_kept=1,
                    cycle_duration_seconds=0.01,
                    update_metrics={
                        "train_steps_executed": 1,
                        "train_loss": 0.1,
                        "weights_updated": True,
                        "update_reason": "updated",
                        "optimizer_steps": 1,
                        "skipped_batches_non_finite": 0,
                    },
                ),
            )
            agentic_summary = agentic.train(
                samples=[SimpleNamespace(prompt="prompt", expected_calls=[], is_irrelevant=False)]
            )
            if not agentic_summary.get("final_model_path"):
                raise RuntimeError("Agentic smoke did not emit final_model_path")
            run_payloads["agentic"] = {
                "summary": agentic_summary,
                "output_dir": output_root / "agentic",
            }

            return build_modality_entries_from_runs(run_payloads)

    def run_modality(
        self,
        baseline_file: Optional[str] = None,
        write_baseline: bool = False,
        compare_baseline: bool = False,
    ) -> bool:
        """Run deterministic modality training smoke checks."""
        from halo_forge.modality_baseline import (
            DEFAULT_MODALITY_BASELINE_FILE,
            build_baseline_payload,
            compare_baseline_payloads,
            compute_fixture_pack_fingerprint,
            format_drift_lines,
            load_baseline_file,
            validate_baseline_payload,
            write_baseline_file,
        )

        baseline_path = Path(baseline_file) if baseline_file else DEFAULT_MODALITY_BASELINE_FILE

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("Modality Smoke", "Deterministic tiny-run validation")
        else:
            print(f"\n{'='*60}")
            print("halo forge Modality Smoke Test")
            print(f"{'='*60}\n")

        self.run_test("Modality fixtures", self.test_modality_fixtures)

        try:
            import torch  # noqa: F401
            has_torch = True
        except Exception:
            has_torch = False

        modality_entries = self.run_test(
            "Modality train smoke",
            self.test_modality_train_smoke,
            skip_condition=not has_torch,
            skip_reason="torch not available in environment",
        )

        def _write_baseline() -> bool:
            if not has_torch or not isinstance(modality_entries, dict):
                raise RuntimeError("Cannot write baseline without modality smoke runtime data")
            payload = build_baseline_payload(modality_entries=modality_entries, seed=42)
            write_baseline_file(baseline_path, payload)
            self.log(f"Wrote modality baseline: {baseline_path}", "info")
            return True

        self.run_test(
            "Modality baseline write",
            _write_baseline,
            skip_condition=not write_baseline,
            skip_reason="--write-baseline not requested",
        )

        def _compare_baseline() -> bool:
            if not baseline_path.exists():
                raise RuntimeError(f"Baseline file not found: {baseline_path}")
            expected = load_baseline_file(baseline_path)
            schema_errors = validate_baseline_payload(expected)
            if schema_errors:
                raise RuntimeError("Invalid baseline schema: " + "; ".join(schema_errors))

            if not has_torch or not isinstance(modality_entries, dict):
                expected_fingerprint = str(expected.get("fixture_pack", ""))
                current_fingerprint = compute_fixture_pack_fingerprint()
                if expected_fingerprint != current_fingerprint:
                    raise RuntimeError(
                        "Fixture pack fingerprint mismatch without runtime smoke coverage. "
                        f"expected={expected_fingerprint} actual={current_fingerprint}"
                    )
                self.log(
                    "Torch runtime unavailable; validated baseline schema + fixture fingerprint only.",
                    "skip",
                )
                return True

            current = build_baseline_payload(modality_entries=modality_entries, seed=42)
            drifts = compare_baseline_payloads(expected, current)
            if drifts:
                raise RuntimeError("\n".join(format_drift_lines(drifts)))
            return True

        self.run_test(
            "Modality baseline compare",
            _compare_baseline,
            skip_condition=not compare_baseline,
            skip_reason="--compare-baseline not requested",
        )
        return self.print_summary()

    def run_ops_e2e(
        self,
        report_file: Optional[str] = None,
        strict: bool = False,
        seed: int = 42,
        fixture_pack: str = "",
    ) -> bool:
        """Run deterministic non-code ops E2E launch reliability checks."""
        from halo_forge.ops_e2e_reliability import (
            DEFAULT_OPS_E2E_REPORT_FILE,
            OPS_E2E_STATUSES,
            OpsE2EModuleResult,
            build_ops_e2e_report,
            compute_ops_e2e_reliability,
            validate_ops_e2e_module,
            write_ops_e2e_report,
        )

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("Ops E2E Reliability", "Launch/stop/relaunch contract checks")
        else:
            print(f"\n{'='*60}")
            print("halo forge Ops E2E Reliability")
            print(f"{'='*60}\n")

        def _resolve_fixture_pack(pack: str) -> Optional[Path]:
            text = str(pack or "").strip()
            if not text:
                return None
            if "/" in text or text.startswith("."):
                root = Path(text).expanduser()
                if not root.is_absolute():
                    root = (Path.cwd() / root).resolve()
                return root
            return (Path.cwd() / "tests" / "fixtures" / "ops_e2e" / text).resolve()

        def _run_ops_e2e() -> bool:
            pack_root = _resolve_fixture_pack(fixture_pack)
            if pack_root:
                if not pack_root.exists() or not pack_root.is_dir():
                    raise RuntimeError(f"Fixture pack directory not found: {pack_root}")
                entries: Dict[str, OpsE2EModuleResult] = {}
                for module in ("vlm", "audio", "reasoning", "agentic", "inference", "benchmark"):
                    module_dir = pack_root / module
                    if not module_dir.exists() or not module_dir.is_dir():
                        raise RuntimeError(f"Fixture pack missing module directory: {module_dir}")
                    entries[module] = validate_ops_e2e_module(
                        module=module,
                        output_dir=module_dir,
                        seed=seed,
                    )
                entries["ui_ops"] = validate_ops_e2e_module(
                    module="ui_ops",
                    output_dir=Path.cwd(),
                    seed=seed,
                )
                report = build_ops_e2e_report(module_entries=entries, seed=seed, source="cli_test")
            else:
                report = compute_ops_e2e_reliability(seed=seed, source="cli_test")

            for module in ("vlm", "audio", "reasoning", "agentic", "inference", "benchmark", "ui_ops"):
                entry = report.modules[module]
                resume_ok = (
                    bool(entry.resume_latest_ok) if entry.resume_latest_ok is not None else False
                )
                print(
                    "OPS_E2E "
                    f"module={module} status={entry.status} "
                    f"launch={1 if entry.launch_ok else 0} "
                    f"stop={1 if entry.stop_ok else 0} "
                    f"relaunch={1 if entry.relaunch_ok else 0} "
                    f"resume={1 if resume_ok else 0}"
                )
                if entry.status not in OPS_E2E_STATUSES:
                    raise RuntimeError(
                        f"Invalid E2E status for module={module}: {entry.status}"
                    )

            report_path = Path(report_file) if report_file else DEFAULT_OPS_E2E_REPORT_FILE
            write_ops_e2e_report(report_path, report)
            self.log(f"Wrote ops E2E report: {report_path}", "info")

            if strict:
                failing = [
                    module
                    for module, entry in report.modules.items()
                    if entry.status == "fail"
                ]
                if failing:
                    raise RuntimeError("Failing modules: " + ", ".join(sorted(failing)))
            return True

        self.run_test("Ops E2E launch reliability", _run_ops_e2e)
        return self.print_summary()

    def run_ops_burnin(
        self,
        *,
        burnin_profile: str = "tiny-v1",
        seed: int = 42,
        report_file: Optional[str] = None,
        baseline_file: Optional[str] = None,
        write_baseline: bool = False,
        compare_baseline: bool = False,
        strict: bool = False,
    ) -> bool:
        """Run bounded dataset-backed non-code burn-in checks."""
        from halo_forge.ops_dataset_burnin import (
            DEFAULT_OPS_BURNIN_BASELINE_FILE,
            DEFAULT_OPS_BURNIN_REPORT_FILE,
            OPS_BURNIN_STATUSES,
            build_burnin_baseline_payload,
            compare_burnin_baselines,
            compute_ops_dataset_burnin,
            format_burnin_drift_lines,
            load_burnin_baseline_file,
            validate_burnin_baseline_payload,
            write_burnin_baseline_file,
            write_ops_burnin_report,
        )

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("Ops Dataset Burn-In", "Bounded non-code runtime trust checks")
        else:
            print(f"\n{'='*60}")
            print("halo forge Ops Dataset Burn-In")
            print(f"{'='*60}\n")

        report_path = Path(report_file) if report_file else DEFAULT_OPS_BURNIN_REPORT_FILE
        baseline_path = Path(baseline_file) if baseline_file else DEFAULT_OPS_BURNIN_BASELINE_FILE

        report = None
        current_baseline = None
        hard_drifts: List[Dict[str, Any]] = []
        warn_drifts: List[Dict[str, Any]] = []

        def _run_burnin() -> bool:
            nonlocal report, current_baseline
            report = compute_ops_dataset_burnin(
                profile=burnin_profile,
                seed=seed,
                source="cli_test",
                execute_commands=False,
                fixture_pack="v1",
            )
            for module in ("vlm", "audio", "reasoning", "agentic", "inference", "benchmark", "ui_ops"):
                entry = report.modules[module]
                print(
                    "OPS_BURNIN "
                    f"module={module} status={entry.status} "
                    f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
                )
                if entry.status not in OPS_BURNIN_STATUSES:
                    raise RuntimeError(
                        f"Invalid burn-in status for module={module}: {entry.status}"
                    )
            write_ops_burnin_report(report_path, report)
            self.log(f"Wrote ops dataset burn-in report: {report_path}", "info")
            current_baseline = build_burnin_baseline_payload(report)
            return True

        self.run_test("Ops dataset burn-in", _run_burnin)

        def _write_baseline() -> bool:
            if current_baseline is None:
                raise RuntimeError("Burn-in baseline payload unavailable")
            write_burnin_baseline_file(baseline_path, current_baseline)
            self.log(f"Wrote burn-in baseline: {baseline_path}", "info")
            return True

        self.run_test(
            "Ops burn-in baseline write",
            _write_baseline,
            skip_condition=not write_baseline,
            skip_reason="--write-baseline not requested",
        )

        def _compare_baseline() -> bool:
            nonlocal hard_drifts, warn_drifts
            if current_baseline is None:
                raise RuntimeError("Burn-in baseline payload unavailable")
            if not baseline_path.exists():
                raise RuntimeError(f"Baseline file not found: {baseline_path}")
            expected = load_burnin_baseline_file(baseline_path)
            schema_errors = validate_burnin_baseline_payload(expected)
            if schema_errors:
                raise RuntimeError("Invalid burn-in baseline schema: " + "; ".join(schema_errors))
            drifts = compare_burnin_baselines(expected=expected, current=current_baseline)
            if drifts:
                for line in format_burnin_drift_lines(drifts):
                    print(line)
            hard_drifts = [drift for drift in drifts if drift.get("severity") == "hard"]
            warn_drifts = [drift for drift in drifts if drift.get("severity") != "hard"]
            if hard_drifts:
                raise RuntimeError("Hard burn-in contract drift detected")
            return True

        self.run_test(
            "Ops burn-in baseline compare",
            _compare_baseline,
            skip_condition=not compare_baseline,
            skip_reason="--compare-baseline not requested",
        )

        if strict and report is not None:
            failing = [
                module for module, entry in report.modules.items() if entry.status == "fail"
            ]
            if failing:
                self.failures += 1
                self.log("Failing modules: " + ", ".join(sorted(failing)), "fail")
            elif warn_drifts:
                self.log(
                    f"Burn-in warning drift detected ({len(warn_drifts)} warn drift(s))",
                    "warn",
                )
        return self.print_summary()

    def run_all_modules(
        self,
        *,
        profile: str = "bounded-v1",
        seed: int = 42,
        report_file: Optional[str] = None,
        module_filters: Optional[List[str]] = None,
        strict: bool = False,
        fixture_pack: str = "",
    ) -> bool:
        """Run all-module readiness checks for coding + non-coding surfaces."""
        from halo_forge.all_module_readiness import (
            ALL_MODULES,
            ALL_MODULE_READINESS_STATUSES,
            DEFAULT_ALL_MODULE_READINESS_REPORT_FILE,
            AllModuleReadiness,
            build_all_module_readiness_report,
            compute_all_module_readiness,
            default_output_map,
            validate_all_module,
            write_all_module_readiness_report,
        )

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("All Module Parity", "Coding + non-coding readiness checks")
        else:
            print(f"\n{'='*60}")
            print("halo forge All Module Parity")
            print(f"{'='*60}\n")

        selected_modules = []
        for module in module_filters or []:
            key = str(module or "").strip().lower()
            if not key:
                continue
            if key not in ALL_MODULES:
                raise RuntimeError(f"Unsupported module filter: {key}")
            if key not in selected_modules:
                selected_modules.append(key)
        if not selected_modules:
            selected_modules = list(ALL_MODULES)

        def _resolve_fixture_pack(pack: str) -> Optional[Path]:
            text = str(pack or "").strip()
            if not text:
                return None
            if "/" in text or text.startswith("."):
                root = Path(text).expanduser()
                if not root.is_absolute():
                    root = (Path.cwd() / root).resolve()
                return root
            return (Path.cwd() / "tests" / "fixtures" / "all_modules" / text).resolve()

        def _run_all_module_checks() -> bool:
            pack_root = _resolve_fixture_pack(fixture_pack)
            if pack_root:
                if not pack_root.exists() or not pack_root.is_dir():
                    raise RuntimeError(f"Fixture pack directory not found: {pack_root}")
                entries: Dict[str, AllModuleReadiness] = {}
                for module in selected_modules:
                    if module == "ui_ops":
                        module_dir = Path.cwd()
                    else:
                        module_dir = pack_root / module
                        if not module_dir.exists() or not module_dir.is_dir():
                            raise RuntimeError(f"Fixture pack missing module directory: {module_dir}")
                    entries[module] = validate_all_module(
                        module=module,
                        output_dir=module_dir,
                        seed=seed,
                        require_artifacts=True,
                    )
                report = build_all_module_readiness_report(
                    module_entries=entries,
                    seed=seed,
                    source="cli_test",
                )
            else:
                base_output_map = default_output_map()
                output_map = {
                    module: base_output_map[module]
                    for module in selected_modules
                }
                report = compute_all_module_readiness(
                    output_map=output_map,
                    seed=seed,
                    source="cli_test",
                    require_artifacts=False,
                )

            for module in selected_modules:
                entry = report.modules[module]
                print(
                    "ALL_READY "
                    f"module={module} status={entry.status} "
                    f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
                )
                if entry.status not in ALL_MODULE_READINESS_STATUSES:
                    raise RuntimeError(
                        f"Invalid all-module status for module={module}: {entry.status}"
                    )

            report_path = Path(report_file) if report_file else DEFAULT_ALL_MODULE_READINESS_REPORT_FILE
            write_all_module_readiness_report(report_path, report)
            self.log(f"Wrote all-module readiness report: {report_path}", "info")

            if strict:
                failing = [
                    module
                    for module in selected_modules
                    if report.modules[module].status == "fail"
                ]
                if failing:
                    raise RuntimeError("Failing modules: " + ", ".join(sorted(failing)))
            return True

        self.run_test(f"All-module readiness ({profile})", _run_all_module_checks)
        return self.print_summary()

    def run_all_module_qualification(
        self,
        *,
        qualification_profile: str = "contract-v1",
        seed: int = 42,
        report_file: Optional[str] = None,
        baseline_file: Optional[str] = None,
        write_baseline: bool = False,
        compare_baseline: bool = False,
        strict: bool = False,
        module_filters: Optional[List[str]] = None,
        fixture_pack: str = "",
        show_fix_commands: bool = False,
    ) -> bool:
        """Run all-module qualification lifecycle checks with optional drift compare."""
        from halo_forge.all_module_readiness import ALL_MODULES
        from halo_forge.all_module_qualification import (
            ALL_MODULE_QUALIFICATION_STATUSES,
            DEFAULT_ALL_MODULE_QUALIFICATION_BASELINE_FILE,
            DEFAULT_ALL_MODULE_QUALIFICATION_REPORT_FILE,
            build_qualification_baseline_payload,
            compare_qualification_baselines,
            compute_all_module_qualification,
            format_qualification_drift_lines,
            format_qualification_issue_lines,
            load_qualification_baseline_file,
            validate_qualification_baseline_payload,
            write_all_module_qualification_report,
            write_qualification_baseline_file,
        )

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header("All-Module Qualification", "Bounded lifecycle qualification checks")
        else:
            print(f"\n{'='*60}")
            print("halo forge All-Module Qualification")
            print(f"{'='*60}\n")

        selected_modules: List[str] = []
        for module in module_filters or []:
            key = str(module or "").strip().lower()
            if not key:
                continue
            if key not in ALL_MODULES:
                raise RuntimeError(f"Unsupported module filter: {key}")
            if key not in selected_modules:
                selected_modules.append(key)
        if not selected_modules:
            selected_modules = list(ALL_MODULES)

        report_path = (
            Path(report_file)
            if report_file
            else DEFAULT_ALL_MODULE_QUALIFICATION_REPORT_FILE
        )
        baseline_path = (
            Path(baseline_file)
            if baseline_file
            else DEFAULT_ALL_MODULE_QUALIFICATION_BASELINE_FILE
        )

        report = None
        current_baseline = None
        hard_drifts: List[Dict[str, Any]] = []
        warn_drifts: List[Dict[str, Any]] = []

        def _resolve_fixture_output_map(pack: str) -> Dict[str, str]:
            text = str(pack or "").strip()
            if not text:
                return {}
            if "/" in text or text.startswith("."):
                pack_root = Path(text).expanduser()
                if not pack_root.is_absolute():
                    pack_root = (Path.cwd() / pack_root).resolve()
            else:
                pack_root = (Path.cwd() / "tests" / "fixtures" / "all_modules" / text).resolve()

            if not pack_root.exists() or not pack_root.is_dir():
                raise RuntimeError(f"Fixture pack directory not found: {pack_root}")

            output_map: Dict[str, str] = {}
            for module in ALL_MODULES:
                if module == "ui_ops":
                    output_map[module] = str(Path.cwd())
                    continue
                module_dir = pack_root / module
                if not module_dir.exists() or not module_dir.is_dir():
                    raise RuntimeError(f"Fixture pack missing module directory: {module_dir}")
                output_map[module] = str(module_dir)
            return output_map

        def _run_qualification() -> bool:
            nonlocal report, current_baseline
            output_map: Dict[str, str] = {}
            if qualification_profile == "fixture-v1":
                output_map = _resolve_fixture_output_map(fixture_pack or "v1")
            elif fixture_pack:
                output_map = _resolve_fixture_output_map(fixture_pack)

            report = compute_all_module_qualification(
                output_map=output_map or None,
                seed=seed,
                profile=qualification_profile,
                source="cli_test",
                module_filters=selected_modules,
            )

            for module in selected_modules:
                entry = report.modules[module]
                print(
                    "ALL_QUAL "
                    f"module={module} status={entry.status} "
                    f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
                )
                for line in format_qualification_issue_lines(
                    entry,
                    show_fix_commands=show_fix_commands,
                ):
                    print(line)
                if entry.status not in ALL_MODULE_QUALIFICATION_STATUSES:
                    raise RuntimeError(
                        f"Invalid qualification status for module={module}: {entry.status}"
                    )

            write_all_module_qualification_report(report_path, report)
            self.log(f"Wrote all-module qualification report: {report_path}", "info")
            current_baseline = build_qualification_baseline_payload(report)
            return True

        self.run_test(f"All-module qualification ({qualification_profile})", _run_qualification)

        def _write_baseline() -> bool:
            if current_baseline is None:
                raise RuntimeError("Qualification baseline payload unavailable")
            write_qualification_baseline_file(baseline_path, current_baseline)
            self.log(f"Wrote qualification baseline: {baseline_path}", "info")
            return True

        self.run_test(
            "Qualification baseline write",
            _write_baseline,
            skip_condition=not write_baseline,
            skip_reason="--write-baseline not requested",
        )

        def _compare_baseline() -> bool:
            nonlocal hard_drifts, warn_drifts
            if current_baseline is None:
                raise RuntimeError("Qualification baseline payload unavailable")
            if not baseline_path.exists():
                raise RuntimeError(f"Baseline file not found: {baseline_path}")
            expected = load_qualification_baseline_file(baseline_path)
            schema_errors = validate_qualification_baseline_payload(expected)
            if schema_errors:
                raise RuntimeError(
                    "Invalid qualification baseline schema: " + "; ".join(schema_errors)
                )
            drifts = compare_qualification_baselines(expected=expected, current=current_baseline)
            if drifts:
                for line in format_qualification_drift_lines(drifts):
                    print(line)
            hard_drifts = [drift for drift in drifts if drift.get("severity") == "hard"]
            warn_drifts = [drift for drift in drifts if drift.get("severity") != "hard"]
            if hard_drifts:
                raise RuntimeError("Hard qualification drift detected")
            return True

        self.run_test(
            "Qualification baseline compare",
            _compare_baseline,
            skip_condition=not compare_baseline,
            skip_reason="--compare-baseline not requested",
        )

        if strict and report is not None:
            failing = [
                module for module in selected_modules if report.modules[module].status == "fail"
            ]
            if failing:
                self.failures += 1
                self.log("Failing modules: " + ", ".join(sorted(failing)), "fail")
            elif warn_drifts:
                self.log(
                    f"Qualification warning drift detected ({len(warn_drifts)} warn drift(s))",
                    "warn",
                )
        return self.print_summary()

    def run_all_module_bootstrap(
        self,
        *,
        bootstrap_profile: str = "contract-v1",
        seed: int = 42,
        output_root: Optional[str] = None,
        report_file: Optional[str] = None,
        module_filters: Optional[List[str]] = None,
        strict: bool = False,
    ) -> bool:
        """Run bounded all-module bootstrap evidence generation."""
        from halo_forge.all_module_readiness import ALL_MODULES
        from halo_forge.all_module_bootstrap import (
            DEFAULT_ALL_MODULE_BOOTSTRAP_OUTPUT_ROOT,
            DEFAULT_ALL_MODULE_BOOTSTRAP_REPORT_FILE,
            compute_all_module_bootstrap,
            write_all_module_bootstrap_report,
        )

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header(
                "All-Module Bootstrap",
                "Bounded evidence generation for readiness remediation",
            )
        else:
            print(f"\n{'='*60}")
            print("halo forge All-Module Bootstrap")
            print(f"{'='*60}\n")

        selected_modules: List[str] = []
        for module in module_filters or []:
            key = str(module or "").strip().lower()
            if not key:
                continue
            if key not in ALL_MODULES:
                raise RuntimeError(f"Unsupported module filter: {key}")
            if key not in selected_modules:
                selected_modules.append(key)
        if not selected_modules:
            selected_modules = list(ALL_MODULES)

        report_path = (
            Path(report_file)
            if report_file
            else DEFAULT_ALL_MODULE_BOOTSTRAP_REPORT_FILE
        )
        output_root_path = (
            Path(output_root)
            if output_root
            else DEFAULT_ALL_MODULE_BOOTSTRAP_OUTPUT_ROOT
        )

        report = None

        def _run_bootstrap() -> bool:
            nonlocal report
            report = compute_all_module_bootstrap(
                bootstrap_profile=bootstrap_profile,
                seed=seed,
                source="cli_test",
                output_root=output_root_path,
                module_filters=selected_modules,
                strict=strict,
            )

            for module in selected_modules:
                entry = report.modules[module]
                print(
                    "ALL_BOOTSTRAP "
                    f"module={module} status={entry.status} "
                    f"created={len(entry.artifacts_created)} "
                    f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
                )

            write_all_module_bootstrap_report(report_path, report)
            self.log(f"Wrote all-module bootstrap report: {report_path}", "info")
            return True

        self.run_test(
            f"All-module bootstrap ({bootstrap_profile})",
            _run_bootstrap,
        )

        if strict and report is not None:
            failing = [
                module for module in selected_modules if report.modules[module].status == "fail"
            ]
            if failing:
                self.failures += 1
                self.log("Failing modules: " + ", ".join(sorted(failing)), "fail")
        return self.print_summary()

    def run_all_module_live(
        self,
        *,
        live_profile: str = "live-smoke-v1",
        seed: int = 42,
        output_root: Optional[str] = None,
        report_file: Optional[str] = None,
        module_filters: Optional[List[str]] = None,
        strict: bool = False,
    ) -> bool:
        """Run bounded all-module live execution probes."""
        from halo_forge.all_module_readiness import ALL_MODULES
        from halo_forge.all_module_live_execution import (
            DEFAULT_ALL_MODULE_LIVE_OUTPUT_ROOT,
            DEFAULT_ALL_MODULE_LIVE_REPORT_FILE,
            compute_all_module_live_execution,
            write_all_module_live_execution_report,
        )

        if self.use_rich:
            self.ui.print_banner()
            self.ui.print_header(
                "All-Module Live Execution",
                "Bounded live-local/smoke probe closure checks",
            )
        else:
            print(f"\n{'='*60}")
            print("halo forge All-Module Live Execution")
            print(f"{'='*60}\n")

        selected_modules: List[str] = []
        for module in module_filters or []:
            key = str(module or "").strip().lower()
            if not key:
                continue
            if key not in ALL_MODULES:
                raise RuntimeError(f"Unsupported module filter: {key}")
            if key not in selected_modules:
                selected_modules.append(key)
        if not selected_modules:
            selected_modules = list(ALL_MODULES)

        report_path = (
            Path(report_file)
            if report_file
            else DEFAULT_ALL_MODULE_LIVE_REPORT_FILE
        )
        output_root_path = (
            Path(output_root)
            if output_root
            else DEFAULT_ALL_MODULE_LIVE_OUTPUT_ROOT
        )

        report = None

        def _run_live() -> bool:
            nonlocal report
            report = compute_all_module_live_execution(
                live_profile=live_profile,
                seed=seed,
                source="cli_test",
                output_root=output_root_path,
                module_filters=selected_modules,
                strict=strict,
            )

            for module in selected_modules:
                entry = report.modules[module]
                print(
                    "ALL_LIVE "
                    f"module={module} status={entry.status} "
                    f"launch={1 if entry.launch_ok else 0} "
                    f"monitor={1 if entry.monitor_ok else 0} "
                    f"results={1 if entry.results_ok else 0} "
                    f"errors={len(entry.errors)} warnings={len(entry.warnings)}"
                )

            write_all_module_live_execution_report(report_path, report)
            self.log(f"Wrote all-module live execution report: {report_path}", "info")
            return True

        self.run_test(
            f"All-module live execution ({live_profile})",
            _run_live,
        )

        if strict and report is not None:
            failing = [
                module for module in selected_modules if report.modules[module].status == "fail"
            ]
            if failing:
                self.failures += 1
                self.log("Failing modules: " + ", ".join(sorted(failing)), "fail")
        return self.print_summary()

    def run_walkthroughs(
        self,
        *,
        profile: str = "contract-v1",
        seed: int = 42,
        report_file: Optional[str] = None,
        module_filters: Optional[List[str]] = None,
        strict: bool = False,
        execute: bool = False,
    ) -> bool:
        """Run all-module walkthrough contracts for local/operator validation."""
        from halo_forge.all_module_walkthroughs import (
            DEFAULT_WALKTHROUGH_REPORT_FILE,
            WALKTHROUGH_PROFILES,
            compute_walkthroughs,
            write_walkthrough_report,
        )

        if profile not in WALKTHROUGH_PROFILES:
            raise RuntimeError(
                f"Invalid walkthrough profile: {profile}. "
                f"Expected one of {', '.join(WALKTHROUGH_PROFILES)}"
            )

        selected_modules: List[str] = []
        for module in module_filters or []:
            key = str(module or "").strip().lower()
            if not key:
                continue
            if key not in (
                "config",
                "data",
                "info",
                "plot",
                "sft",
                "raft",
                "benchmark_code",
                "benchmark_non_code",
                "inference",
                "vlm",
                "audio",
                "reasoning",
                "agentic",
                "ui_ops",
            ):
                raise RuntimeError(f"Unsupported walkthrough module filter: {key}")
            if key not in selected_modules:
                selected_modules.append(key)

        if not selected_modules:
            selected_modules = []

        def _run_walkthroughs() -> bool:
            report = compute_walkthroughs(
                modules=selected_modules,
                seed=seed,
                profile=profile,
                execute=execute,
            )
            modules_to_print = selected_modules or list(report.modules.keys())
            for module in modules_to_print:
                entry = report.modules[module]
                print(
                    "WALKTHROUGH "
                    f"module={module} status={entry.status} "
                    f"steps={len(entry.steps)} errors={len(entry.errors)} warnings={len(entry.warnings)}"
                )

            path = Path(report_file) if report_file else DEFAULT_WALKTHROUGH_REPORT_FILE
            write_walkthrough_report(path, report)
            self.log(f"Wrote walkthrough report: {path}", "info")

            if strict:
                failing = [module for module in modules_to_print if report.modules[module].status == "fail"]
                if failing:
                    raise RuntimeError("Failing modules: " + ", ".join(sorted(failing)))
            return True

        self.run_test(f"All-module walkthroughs ({profile})", _run_walkthroughs)
        return self.print_summary()


def cmd_test(args):
    """Run pipeline validation tests."""
    baseline_levels = {"modality", "ops-burnin", "all-module-qualification"}
    if args.level not in baseline_levels and (args.write_baseline or args.compare_baseline):
        print(
            f"{RED}Error: --write-baseline/--compare-baseline are supported only with "
            f"--level modality, --level ops-burnin, or --level all-module-qualification{NC}"
        )
        sys.exit(2)

    runner = TestRunner(verbose=args.verbose, model=args.model)
    
    if args.level == "smoke":
        success = runner.run_smoke()
    elif args.level == "standard":
        success = runner.run_standard()
    elif args.level == "full":
        success = runner.run_full()
    elif args.level == "modality":
        success = runner.run_modality(
            baseline_file=args.baseline_file,
            write_baseline=args.write_baseline,
            compare_baseline=args.compare_baseline,
        )
    elif args.level == "ops-e2e":
        success = runner.run_ops_e2e(
            report_file=args.report_file,
            strict=args.strict,
            seed=args.seed,
            fixture_pack=args.fixture_pack,
        )
    elif args.level == "ops-burnin":
        report_file = args.report_file
        baseline_file = args.baseline_file
        if report_file == "results/readiness/ops_e2e_launch_reliability.v1.json":
            report_file = "results/readiness/ops_dataset_burnin.v1.json"
        if baseline_file == "tests/baselines/modality_runtime_baseline.v1.json":
            baseline_file = "tests/baselines/ops_dataset_burnin_baseline.v1.json"
        success = runner.run_ops_burnin(
            burnin_profile=args.burnin_profile,
            seed=args.seed,
            report_file=report_file,
            baseline_file=baseline_file,
            write_baseline=args.write_baseline,
            compare_baseline=args.compare_baseline,
            strict=args.strict,
        )
    elif args.level == "all-modules":
        report_file = args.report_file
        if report_file == "results/readiness/ops_e2e_launch_reliability.v1.json":
            report_file = "results/readiness/all_modules_readiness.v1.json"
        success = runner.run_all_modules(
            profile=args.profile,
            seed=args.seed,
            report_file=report_file,
            module_filters=args.module,
            strict=args.strict,
            fixture_pack=args.fixture_pack,
        )
    elif args.level == "walkthroughs":
        report_file = args.report_file
        profile = args.profile
        if report_file == "results/readiness/ops_e2e_launch_reliability.v1.json":
            report_file = (
                ".internal_docs/research_testing/walkthroughs/reports/"
                "all_module_e2e_walkthrough_report.v1.json"
            )
        if profile == "bounded-v1":
            profile = "contract-v1"
        success = runner.run_walkthroughs(
            profile=profile,
            seed=args.seed,
            report_file=report_file,
            module_filters=args.module,
            strict=args.strict,
            execute=args.execute,
        )
    elif args.level == "all-module-qualification":
        report_file = args.report_file
        baseline_file = args.baseline_file
        if report_file == "results/readiness/ops_e2e_launch_reliability.v1.json":
            report_file = "results/readiness/all_module_qualification.v1.json"
        if baseline_file == "tests/baselines/modality_runtime_baseline.v1.json":
            baseline_file = "tests/baselines/all_module_qualification_baseline.v1.json"
        success = runner.run_all_module_qualification(
            qualification_profile=args.qualification_profile,
            seed=args.seed,
            report_file=report_file,
            baseline_file=baseline_file,
            write_baseline=args.write_baseline,
            compare_baseline=args.compare_baseline,
            strict=args.strict,
            module_filters=args.module,
            fixture_pack=args.fixture_pack,
            show_fix_commands=args.show_fix_commands,
        )
    elif args.level == "all-module-bootstrap":
        report_file = args.report_file
        if report_file == "results/readiness/ops_e2e_launch_reliability.v1.json":
            report_file = "results/readiness/all_module_bootstrap.v1.json"
        success = runner.run_all_module_bootstrap(
            bootstrap_profile=args.bootstrap_profile,
            seed=args.seed,
            output_root=args.output_root,
            report_file=report_file,
            module_filters=args.module,
            strict=args.strict,
        )
    elif args.level == "all-module-live":
        report_file = args.report_file
        output_root = args.output_root
        if report_file == "results/readiness/ops_e2e_launch_reliability.v1.json":
            report_file = "results/readiness/all_module_live_execution.v1.json"
        if output_root == "results/bootstrap":
            output_root = "results/live_probes"
        success = runner.run_all_module_live(
            live_profile=args.live_profile,
            seed=args.seed,
            output_root=output_root,
            report_file=report_file,
            module_filters=args.module,
            strict=args.strict,
        )
    else:
        print(f"Unknown test level: {args.level}")
        print(
            "Valid levels: smoke, standard, full, modality, ops-e2e, ops-burnin, "
            "all-modules, walkthroughs, all-module-qualification, all-module-bootstrap, all-module-live"
        )
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

    # MLX path: bitsandbytes-style quantization is not applicable; weights are
    # quantized at conversion time (use mlx-community/...-4bit models or
    # `python -m mlx_lm.convert`). We surface that and run a smoke generation
    # instead of the full PyTorch optimize/calibrate pipeline.
    if getattr(args, 'accelerator', 'auto') == 'mlx':
        from halo_forge.backend.mlx import MLXInferenceAdapter
        print("\n[MLX] Skipping torch optimize/calibrate; running smoke generation.")
        adapter = MLXInferenceAdapter(args.model)
        adapter.load()
        out = adapter.generate(
            "Write a function to sort a list.",
            max_tokens=64,
            temperature=0.7,
        )
        print("\nSample generation:")
        print(out)
        adapter.cleanup()
        print(f"\n{GREEN}MLX smoke OK.{NC} For pre-quantized weights see docs/MLX.md.")
        return
    
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
        dtype=recommended_dtype(),
        device_map=get_device_map(),
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
    from halo_forge.vlm.data import list_vlm_datasets
    from halo_forge.vlm.verifiers import check_vlm_dependencies
    
    print_banner()
    
    print(f"\n{GREEN}VLM RAFT Training{NC}")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Dataset:     {args.dataset}")
    print(f"Output:      {args.output}")
    print(f"Cycles:      {args.cycles}")
    print(f"Seed:        {args.seed}")
    print("=" * 60)

    _enforce_modality_train_contract("vlm", args)
    
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
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        seed=args.seed,
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
        summary = trainer.train(
            dataset_path,
            resume_from=getattr(args, "resume_from_cycle", 0),
        )
    except ValueError as e:
        print(f"{RED}Training error: {e}{NC}")
        sys.exit(2)
    finally:
        trainer.cleanup()

    total_steps = int(summary.get("total_train_steps_executed", 0))
    final_loss = summary.get("final_train_loss")
    _enforce_training_outcome_or_exit("vlm", summary)
    
    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Output: {args.output}")
    if summary.get("final_model_path"):
        print(f"Final model: {summary['final_model_path']}")
    _print_training_run_metadata(summary)
    print(f"Train steps executed: {total_steps}")
    if isinstance(final_loss, (int, float)):
        print(f"Final train loss: {final_loss:.4f}")


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
    summary = trainer.train()
    _print_completed_training_summary("vlm_sft", args.output, summary)


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
    print("  halo-forge audio train --model openai/whisper-small --dataset librispeech --seed 42")


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
    summary = trainer.train()
    _print_completed_training_summary("audio_sft", args.output, summary)


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
    results["model"] = args.model
    results["benchmark"] = args.dataset
    results["task"] = args.task

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
    print(f"Seed: {args.seed}")

    _enforce_modality_train_contract("audio", args)
    
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
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        seed=args.seed,
    )
    
    # Run training
    trainer = AudioRAFTTrainer(config)
    try:
        results = trainer.train(
            args.dataset,
            resume_from_cycle=getattr(args, "resume_from_cycle", 0),
        )
    except ValueError as e:
        print(f"{RED}Training error: {e}{NC}")
        sys.exit(2)
    summary = getattr(trainer, "training_summary", {})
    total_steps = int(summary.get("total_train_steps_executed", 0))
    final_loss = summary.get("final_train_loss")
    _enforce_training_outcome_or_exit("audio", summary)
    
    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Final model saved to: {args.output}")
    if summary.get("final_model_path"):
        print(f"Final model: {summary['final_model_path']}")
    _print_training_run_metadata(summary)
    print(f"Train steps executed: {total_steps}")
    if isinstance(final_loss, (int, float)):
        print(f"Final train loss: {final_loss:.4f}")
    
    print("\nUsage:")
    print("  halo-forge vlm train --dataset textvqa --model Qwen/Qwen2-VL-7B-Instruct --seed 42")
    print("  halo-forge vlm benchmark --dataset docvqa --model path/to/model")


def main():
    if sys.version_info >= (3, 14):
        print(
            "halo-forge supports Python >=3.10,<3.14. "
            f"Current interpreter is {sys.version.split()[0]}. "
            "Create a Python 3.10-3.13 environment and rerun the command.",
            file=sys.stderr,
        )
        sys.exit(2)
    parser = argparse.ArgumentParser(
        prog='halo-forge',
        description='Multi-backend RLVR training framework for AMD ROCm, Apple Silicon, and CUDA'
    )

    # Global flags
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress terminal output (logs still written to file)')
    # Compute accelerator override. Distinct from the per-subcommand `--backend`
    # flag in `data generate` (which selects the LLM API: deepseek/anthropic/...).
    # Default: auto-detect via halo_forge.backend.get_backend().
    # `mlx` requires the `[mlx]` extra and currently only powers inference paths.
    parser.add_argument(
        '--accelerator',
        choices=['auto', 'rocm', 'rocm_gfx1151', 'cuda', 'mps', 'mlx', 'cpu'],
        default='auto',
        help='Compute accelerator to target. "auto" (default) detects ROCm/CUDA/MPS/CPU; '
             'pass "mlx" explicitly to use Apple MLX for supported training and inference paths. '
             'Sets HALOFORGE_BACKEND for downstream code.'
    )

    def add_apple_runtime_flags(train_parser, *, neural_accelerators: bool = True):
        train_parser.add_argument(
            '--no-caffeinate',
            action='store_true',
            help='Dashboard round-trip flag: opt out of macOS caffeinate wrapping for launched training jobs',
        )
        if neural_accelerators:
            train_parser.add_argument(
                '--enable-neural-accelerators',
                action='store_true',
                help='Annotate and validate experimental Apple M5+ Neural Accelerator opt-in (no kernel routing yet)',
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

    # data synthesize (Track D1) — teacher → verifier → filter pipeline.
    synth_parser = data_subparsers.add_parser(
        'synthesize',
        help='Generate synthetic training data from prompts via a teacher model + verifier filter',
    )
    synth_parser.add_argument('--seeds', '-i', required=True,
                              help='JSONL or text file of seed prompts (one per line)')
    synth_parser.add_argument('--output', '-o', required=True,
                              help='Output JSONL path')
    synth_parser.add_argument('--teacher-model', default='default',
                              help='Model name for the OpenAI-compatible teacher endpoint')
    synth_parser.add_argument('--base-url',
                              help='Teacher endpoint base URL (default: http://127.0.0.1:8001/v1 — '
                                   'a local halo-forge serve process)')
    synth_parser.add_argument('--api-key',
                              help='Teacher endpoint API key (env: HALOFORGE_TEACHER_API_KEY)')
    synth_parser.add_argument('--system-prompt',
                              help='System message prepended to every teacher call')
    synth_parser.add_argument('--verifier', default='json_structure',
                              help='V1 verifier short name to score completions '
                                   '(execution, llm_judge, bleu, json_schema, regex_format, ...)')
    synth_parser.add_argument('--n-per-prompt', type=int, default=1,
                              help='Completions sampled per prompt (>=2 required for --kind preference)')
    synth_parser.add_argument('--threshold', type=float, default=0.5,
                              help='Reward threshold for acceptance (default 0.5)')
    synth_parser.add_argument('--kind', default='sft', choices=['sft', 'preference'],
                              help='sft → {prompt, completion}; preference → {prompt, chosen, rejected}')
    synth_parser.add_argument('--max-tokens', type=int, default=512,
                              help='Teacher max_tokens per call (default 512)')
    synth_parser.add_argument('--temperature', type=float, default=0.8,
                              help='Teacher sampling temperature (default 0.8 for diverse generation)')

    # data score (Track D3) — heuristic quality scoring + filter.
    score_parser = data_subparsers.add_parser(
        'score',
        help='Score JSONL records by heuristic quality and filter by threshold / top-K%',
    )
    score_parser.add_argument('--input', '-i', required=True)
    score_parser.add_argument('--output', '-o', required=True)
    score_parser.add_argument('--threshold', type=float, default=0.5,
                              help='Composite score below which rows are dropped (default 0.5)')
    score_parser.add_argument('--top-k-pct', type=float,
                              help='Keep top K%% by score instead of using --threshold (e.g. 0.5 = top 50%%)')

    # data dedup (Track D2)
    dedup_parser = data_subparsers.add_parser(
        'dedup',
        help='Deduplicate a JSONL dataset (exact / fuzzy MinHash)',
    )
    dedup_parser.add_argument('--input', '-i', required=True, help='Input JSONL path')
    dedup_parser.add_argument('--output', '-o', required=True, help='Output JSONL path (deduped)')
    dedup_parser.add_argument('--method', default='exact', choices=['exact', 'fuzzy'],
                              help='exact = SHA256 over normalized text; fuzzy = MinHash + LSH')
    dedup_parser.add_argument('--threshold', type=float, default=0.85,
                              help='[fuzzy] Jaccard similarity threshold (default 0.85)')
    dedup_parser.add_argument('--key', default='text',
                              help='Field name when records are dicts (default "text")')
    dedup_parser.add_argument('--case-sensitive', action='store_true',
                              help='Skip the lowercase-and-trim normalization step')
    dedup_parser.add_argument('--num-perm', type=int, default=128,
                              help='[fuzzy] MinHash permutations (default 128)')
    dedup_parser.add_argument('--shingle-n', type=int, default=5,
                              help='[fuzzy] Word n-gram shingle size (default 5)')

    # sft command
    sft_parser = subparsers.add_parser('sft', help='SFT training')
    sft_subparsers = sft_parser.add_subparsers(dest='sft_command', required=True)
    
    # sft train
    sft_train_parser = sft_subparsers.add_parser('train', help='Run SFT training')
    add_apple_runtime_flags(sft_train_parser)
    sft_train_parser.add_argument('--config', '-c', help='Config file path')
    sft_train_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-Coder-7B', help='Base model')
    sft_train_parser.add_argument('--dataset', '-d', help='HuggingFace dataset ID or short name (e.g., codealpaca, metamath)')
    sft_train_parser.add_argument('--data', help='Local training data file (JSONL)')
    sft_train_parser.add_argument('--output', '-o', default='models/sft', help='Output directory')
    sft_train_parser.add_argument('--resume', help='Resume from checkpoint')
    sft_train_parser.add_argument('--dry-run', action='store_true', help='Validate config without training')
    
    # Training hyperparameters
    sft_train_parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    sft_train_parser.add_argument('--batch-size', type=int, default=2, help='Per-device batch size')
    sft_train_parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    sft_train_parser.add_argument('--warmup-ratio', type=float, default=0.03, help='Warmup ratio for LR scheduler')
    sft_train_parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay for regularization')
    sft_train_parser.add_argument('--max-grad-norm', type=float, default=0.3, help='Max gradient norm for clipping')
    sft_train_parser.add_argument('--gradient-accumulation', type=int, default=16, 
                                  help='Gradient accumulation steps (effective batch = batch_size * accum)')
    
    # LoRA options
    sft_train_parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank')
    sft_train_parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    sft_train_parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
    sft_train_parser.add_argument('--no-lora', action='store_true', help='Disable LoRA (full fine-tuning)')
    # Track T5 — PEFT additions. Vanilla LoRA stays the default; opt in.
    sft_train_parser.add_argument('--use-dora', action='store_true',
                                  help='Use DoRA (decomposed magnitude+direction); slightly slower but typically matches LoRA at lower rank')
    sft_train_parser.add_argument('--use-rslora', action='store_true',
                                  help='Use rank-stabilized LoRA scaling (alpha/sqrt(r) instead of alpha/r)')
    sft_train_parser.add_argument('--init-lora-weights', default='true',
                                  help='LoRA initialization: true (default), pissa, pissa_niter_4, loftq, olora, gaussian, false')
    # Track T4 — optimizer choice.
    sft_train_parser.add_argument('--optim', default='adamw_torch',
                                  help='Optimizer (adamw_torch, adamw_bnb_8bit, lion_8bit, paged_adamw_8bit, ...)')
    
    # Checkpointing
    sft_train_parser.add_argument('--save-steps', type=int, default=500, help='Save checkpoint every N steps')
    sft_train_parser.add_argument('--eval-steps', type=int, default=250, help='Evaluate every N steps')
    sft_train_parser.add_argument('--save-total-limit', type=int, default=3, help='Max checkpoints to keep')
    
    # Early stopping
    sft_train_parser.add_argument('--early-stopping-patience', type=int, default=5, 
                                  help='Stop if no improvement for N evals')
    
    # Data options
    sft_train_parser.add_argument('--max-samples', type=int, help='Limit number of training samples')
    sft_train_parser.add_argument('--validation-split', type=float, default=0.05, help='Validation set fraction')
    sft_train_parser.add_argument('--max-seq-length', type=int, default=2048, help='Maximum sequence length')
    
    # Hardware options
    sft_train_parser.add_argument('--no-gradient-checkpointing', action='store_true',
                                  help='Disable gradient checkpointing (uses more memory)')
    
    # sft datasets
    sft_datasets_parser = sft_subparsers.add_parser('datasets', help='List available SFT datasets')

    # dpo command (Track T1 / phase Q1) - Direct Preference Optimization
    dpo_parser = subparsers.add_parser('dpo', help='DPO (Direct Preference Optimization) training')
    dpo_subparsers = dpo_parser.add_subparsers(dest='dpo_command', required=True)

    # dpo train
    dpo_train_parser = dpo_subparsers.add_parser('train', help='Run DPO training')
    add_apple_runtime_flags(dpo_train_parser)
    dpo_train_parser.add_argument('--config', '-c', help='Config file path')
    dpo_train_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-3B-Instruct',
                                  help='Base / SFT-tuned model to align')
    dpo_train_parser.add_argument('--dataset', '-d',
                                  help='HuggingFace dataset id or short name (ultrafeedback, orca_dpo, hh_rlhf, py_dpo)')
    dpo_train_parser.add_argument('--data',
                                  help='Local JSONL file with prompt/chosen/rejected rows')
    dpo_train_parser.add_argument('--output', '-o', default='models/dpo', help='Output directory')
    dpo_train_parser.add_argument('--resume', help='Resume from checkpoint')
    dpo_train_parser.add_argument('--dry-run', action='store_true', help='Validate config without training')

    # DPO algorithm knobs
    dpo_train_parser.add_argument('--beta', type=float, default=0.1,
                                  help='KL-regularization strength against the reference model (default: 0.1)')
    dpo_train_parser.add_argument('--loss-type', default='sigmoid',
                                  choices=['sigmoid', 'ipo', 'hinge', 'kto_pair', 'rpo'],
                                  help='DPO loss variant (default: sigmoid)')
    dpo_train_parser.add_argument('--reference-free', action='store_true',
                                  help='Skip the reference model (uses policy at step 0); saves memory')
    dpo_train_parser.add_argument('--label-smoothing', type=float, default=0.0,
                                  help='cDPO label smoothing (default: 0.0)')

    # Training hyperparameters
    dpo_train_parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    dpo_train_parser.add_argument('--batch-size', type=int, default=1,
                                  help='Per-device batch size (DPO doubles memory: chosen+rejected)')
    dpo_train_parser.add_argument('--learning-rate', type=float, default=5e-6,
                                  help='Learning rate (DPO needs much smaller LR than SFT)')
    dpo_train_parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')
    dpo_train_parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    dpo_train_parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Max gradient norm')
    dpo_train_parser.add_argument('--gradient-accumulation', type=int, default=16,
                                  help='Gradient accumulation steps')

    # LoRA
    dpo_train_parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank')
    dpo_train_parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    dpo_train_parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
    dpo_train_parser.add_argument('--load-in-4bit', action='store_true',
                                  help='QLoRA: load base model in 4-bit (CUDA/ROCm only)')
    # Track T5 — PEFT additions.
    dpo_train_parser.add_argument('--use-dora', action='store_true',
                                  help='Use DoRA (decomposed magnitude+direction)')
    dpo_train_parser.add_argument('--use-rslora', action='store_true',
                                  help='Use rank-stabilized LoRA scaling')
    dpo_train_parser.add_argument('--init-lora-weights', default='true',
                                  help='LoRA initialization: true, pissa, loftq, olora, gaussian, false')
    # Track T4 — optimizer choice.
    dpo_train_parser.add_argument('--optim', default='adamw_torch',
                                  help='Optimizer (adamw_torch, adamw_bnb_8bit, lion_8bit, ...)')

    # Checkpointing
    dpo_train_parser.add_argument('--save-steps', type=int, default=200, help='Save every N steps')
    dpo_train_parser.add_argument('--eval-steps', type=int, default=100, help='Eval every N steps')
    dpo_train_parser.add_argument('--save-total-limit', type=int, default=3, help='Max checkpoints to keep')

    # Data options
    dpo_train_parser.add_argument('--max-samples', type=int, help='Limit number of training pairs')
    dpo_train_parser.add_argument('--validation-split', type=float, default=0.05, help='Validation fraction')
    dpo_train_parser.add_argument('--max-seq-length', type=int, default=1024,
                                  help='Combined prompt+response length cap')
    dpo_train_parser.add_argument('--max-prompt-length', type=int, default=512,
                                  help='Prompt length cap (DPO truncates from the left after this)')

    # Hardware
    dpo_train_parser.add_argument('--no-gradient-checkpointing', action='store_true',
                                  help='Disable gradient checkpointing (uses more memory)')

    # dpo datasets
    dpo_datasets_parser = dpo_subparsers.add_parser(
        'datasets', help='List available preference datasets'
    )

    # orpo command (Track T17b) — Odds-Ratio Preference Optimization.
    # Same input shape as DPO (prompt/chosen/rejected); reference-free,
    # single-pass — typically half the wall-time of DPO at similar quality.
    orpo_parser = subparsers.add_parser('orpo', help='ORPO (Odds-Ratio Preference Optimization) training')
    orpo_subparsers = orpo_parser.add_subparsers(dest='orpo_command', required=True)

    orpo_train_parser = orpo_subparsers.add_parser('train', help='Run ORPO training')
    add_apple_runtime_flags(orpo_train_parser)
    orpo_train_parser.add_argument('--config', '-c', help='Config file path')
    orpo_train_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-3B-Instruct',
                                   help='Base / SFT-tuned model to align')
    orpo_train_parser.add_argument('--dataset', '-d',
                                   help='HuggingFace dataset id or short name (ultrafeedback, orca_dpo, hh_rlhf, py_dpo)')
    orpo_train_parser.add_argument('--data',
                                   help='Local JSONL with prompt/chosen/rejected rows')
    orpo_train_parser.add_argument('--output', '-o', default='models/orpo', help='Output directory')
    orpo_train_parser.add_argument('--resume', help='Resume from checkpoint')
    orpo_train_parser.add_argument('--dry-run', action='store_true', help='Validate config without training')

    orpo_train_parser.add_argument('--beta', type=float, default=0.1,
                                   help='Relative weight of preference (log-odds) term vs NLL (default: 0.1)')

    orpo_train_parser.add_argument('--epochs', type=int, default=1)
    orpo_train_parser.add_argument('--batch-size', type=int, default=1,
                                   help='Per-device batch size (ORPO sees chosen+rejected per row)')
    orpo_train_parser.add_argument('--learning-rate', type=float, default=8e-6,
                                   help='Learning rate (default: 8e-6 — between SFT and DPO)')
    orpo_train_parser.add_argument('--warmup-ratio', type=float, default=0.1)
    orpo_train_parser.add_argument('--weight-decay', type=float, default=0.0)
    orpo_train_parser.add_argument('--max-grad-norm', type=float, default=1.0)
    orpo_train_parser.add_argument('--gradient-accumulation', type=int, default=16)

    orpo_train_parser.add_argument('--lora-rank', type=int, default=16)
    orpo_train_parser.add_argument('--lora-alpha', type=int, default=32)
    orpo_train_parser.add_argument('--lora-dropout', type=float, default=0.05)
    orpo_train_parser.add_argument('--load-in-4bit', action='store_true',
                                   help='QLoRA: load base model in 4-bit (CUDA/ROCm only)')
    orpo_train_parser.add_argument('--use-dora', action='store_true')
    orpo_train_parser.add_argument('--use-rslora', action='store_true')
    orpo_train_parser.add_argument('--init-lora-weights', default='true')
    orpo_train_parser.add_argument('--optim', default='adamw_torch')

    orpo_train_parser.add_argument('--save-steps', type=int, default=200)
    orpo_train_parser.add_argument('--eval-steps', type=int, default=100)
    orpo_train_parser.add_argument('--save-total-limit', type=int, default=3)
    orpo_train_parser.add_argument('--max-samples', type=int)
    orpo_train_parser.add_argument('--validation-split', type=float, default=0.05)
    orpo_train_parser.add_argument('--max-seq-length', type=int, default=1024)
    orpo_train_parser.add_argument('--max-prompt-length', type=int, default=512)
    orpo_train_parser.add_argument('--no-gradient-checkpointing', action='store_true')

    # rm command (Track T3) — Bradley-Terry reward model trainer.
    rm_parser = subparsers.add_parser('rm', help='Reward model training (Bradley-Terry)')
    rm_subparsers = rm_parser.add_subparsers(dest='rm_command', required=True)

    rm_train_parser = rm_subparsers.add_parser('train', help='Run reward-model training')
    add_apple_runtime_flags(rm_train_parser)
    rm_train_parser.add_argument('--config', '-c', help='Config file path')
    rm_train_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-3B-Instruct',
                                 help='Base / SFT-tuned model')
    rm_train_parser.add_argument('--dataset', '-d',
                                 help='HF preference dataset id (ultrafeedback, orca_dpo, hh_rlhf, py_dpo)')
    rm_train_parser.add_argument('--data',
                                 help='Local JSONL with prompt/chosen/rejected rows')
    rm_train_parser.add_argument('--output', '-o', default='models/rm', help='Output directory')
    rm_train_parser.add_argument('--resume', help='Resume from checkpoint')
    rm_train_parser.add_argument('--dry-run', action='store_true')

    rm_train_parser.add_argument('--epochs', type=int, default=1)
    rm_train_parser.add_argument('--batch-size', type=int, default=4)
    rm_train_parser.add_argument('--learning-rate', type=float, default=1e-5)
    rm_train_parser.add_argument('--warmup-ratio', type=float, default=0.05)
    rm_train_parser.add_argument('--weight-decay', type=float, default=0.0)
    rm_train_parser.add_argument('--max-grad-norm', type=float, default=1.0)
    rm_train_parser.add_argument('--gradient-accumulation', type=int, default=4)
    rm_train_parser.add_argument('--max-length', type=int, default=1024)
    rm_train_parser.add_argument('--max-samples', type=int)
    rm_train_parser.add_argument('--center-rewards-coefficient', type=float, default=0.01,
                                 help='Centering regularizer (default 0.01; 0 disables)')

    rm_train_parser.add_argument('--lora-rank', type=int, default=8)
    rm_train_parser.add_argument('--lora-alpha', type=int, default=16)
    rm_train_parser.add_argument('--lora-dropout', type=float, default=0.05)
    rm_train_parser.add_argument('--load-in-4bit', action='store_true')
    rm_train_parser.add_argument('--use-dora', action='store_true')
    rm_train_parser.add_argument('--use-rslora', action='store_true')
    rm_train_parser.add_argument('--init-lora-weights', default='true')
    rm_train_parser.add_argument('--optim', default='adamw_torch')
    rm_train_parser.add_argument('--no-gradient-checkpointing', action='store_true')

    # grpo command (Track T2 / phase Q1) — Group Relative Policy Optimization
    grpo_parser = subparsers.add_parser('grpo', help='GRPO training (verifier-grounded RL)')
    grpo_subparsers = grpo_parser.add_subparsers(dest='grpo_command', required=True)

    grpo_train_parser = grpo_subparsers.add_parser('train', help='Run GRPO training')
    add_apple_runtime_flags(grpo_train_parser)
    grpo_train_parser.add_argument('--config', '-c', help='Config file path')
    grpo_train_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-3B-Instruct',
                                   help='Base / SFT-tuned model')
    grpo_train_parser.add_argument('--dataset', '-d',
                                   help='HuggingFace dataset id (must have a "prompt" column)')
    grpo_train_parser.add_argument('--data', help='Local JSONL with "prompt" rows')
    grpo_train_parser.add_argument('--output', '-o', default='models/grpo', help='Output directory')
    grpo_train_parser.add_argument('--resume', help='Resume from checkpoint')
    grpo_train_parser.add_argument('--dry-run', action='store_true', help='Validate config without training')

    # GRPO algorithm
    grpo_train_parser.add_argument('--num-generations', type=int, default=4,
                                   help='Group size: completions sampled per prompt (default: 4)')
    grpo_train_parser.add_argument('--beta', type=float, default=0.04,
                                   help='KL-regularization strength (default: 0.04, DeepSeek-R1)')
    grpo_train_parser.add_argument('--epsilon', type=float, default=0.2,
                                   help='PPO ratio clip (default: 0.2)')
    grpo_train_parser.add_argument('--temperature', type=float, default=0.9,
                                   help='Rollout temperature (default: 0.9 — diverse groups)')
    grpo_train_parser.add_argument('--no-scale-rewards', action='store_true',
                                   help='Skip dividing advantages by std(group); RLOO-flavored')
    grpo_train_parser.add_argument('--reference-free', action='store_true',
                                   help='Skip reference model; saves memory')
    grpo_train_parser.add_argument('--verifier', default='execution',
                                   help='Verifier short-name from the V1 plugin registry '
                                        '(execution, llm_judge, ...). Run halo-forge sft datasets '
                                        'to see registered verifiers.')
    grpo_train_parser.add_argument('--reward-threshold', type=float, default=0.0,
                                   help='Below this, advantage is forced to 0')

    # Hyperparameters
    grpo_train_parser.add_argument('--epochs', type=int, default=1)
    grpo_train_parser.add_argument('--batch-size', type=int, default=1)
    grpo_train_parser.add_argument('--learning-rate', type=float, default=1e-6,
                                   help='GRPO LR (much smaller than SFT)')
    grpo_train_parser.add_argument('--warmup-ratio', type=float, default=0.1)
    grpo_train_parser.add_argument('--weight-decay', type=float, default=0.0)
    grpo_train_parser.add_argument('--max-grad-norm', type=float, default=1.0)
    grpo_train_parser.add_argument('--gradient-accumulation', type=int, default=16)

    # LoRA
    grpo_train_parser.add_argument('--lora-rank', type=int, default=16)
    grpo_train_parser.add_argument('--lora-alpha', type=int, default=32)
    grpo_train_parser.add_argument('--lora-dropout', type=float, default=0.05)
    grpo_train_parser.add_argument('--load-in-4bit', action='store_true')
    grpo_train_parser.add_argument('--use-dora', action='store_true')
    grpo_train_parser.add_argument('--use-rslora', action='store_true')
    grpo_train_parser.add_argument('--init-lora-weights', default='true')
    grpo_train_parser.add_argument('--optim', default='adamw_torch')

    # Data lengths
    grpo_train_parser.add_argument('--max-samples', type=int)
    grpo_train_parser.add_argument('--max-prompt-length', type=int, default=512)
    grpo_train_parser.add_argument('--max-completion-length', type=int, default=512)

    # Rollout engine (Track I6)
    grpo_train_parser.add_argument('--rollout-engine', default='auto',
                                   choices=['auto', 'torch', 'vllm', 'mlx'],
                                   help='Generation backend for the rollout stage')

    grpo_train_parser.add_argument('--no-gradient-checkpointing', action='store_true')

    # probe command (Track V9) — mid-training general-benchmark probe.
    probe_parser = subparsers.add_parser(
        'probe',
        help='Run a small held-out benchmark + diff vs baseline (catastrophic-forgetting safeguard)',
    )
    probe_parser.add_argument('--model', '-m', required=True,
                              help='Model id, mlx-community id, or local path')
    probe_parser.add_argument('--tasks', '-t',
                              help='Comma-separated task names (default: small probe set)')
    probe_parser.add_argument('--limit', type=int, default=100,
                              help='Samples per task (default 100; smaller = faster probe)')
    probe_parser.add_argument('--baseline',
                              help='Path to baseline.json. First run writes; subsequent runs diff.')
    probe_parser.add_argument('--tolerance', type=float, default=0.05,
                              help='Regression triggered when Δ < -tolerance (default 0.05)')
    probe_parser.add_argument('--backend', default='hf', choices=['hf', 'vllm', 'mlx'])
    probe_parser.add_argument('--cycle', type=int,
                              help='Tag this probe with the cycle number it ran at')
    probe_parser.add_argument('--notes',
                              help='Free-form annotation for this probe (e.g. "after SFT")')

    # eval command (Track V8) — lm-evaluation-harness wrapper.
    eval_parser = subparsers.add_parser(
        'eval',
        help='Run academic benchmarks via lm-evaluation-harness',
    )
    eval_parser.add_argument('--model', '-m',
                             help='Model id, mlx-community id, or local path')
    eval_parser.add_argument('--tasks', '-t', default='core',
                             help='Comma-separated task names or curated group '
                                  '(core, reasoning, code, instruction_following, knowledge)')
    eval_parser.add_argument('--limit', type=int,
                             help='Cap samples per task (smoke-test mode)')
    eval_parser.add_argument('--batch-size', type=int,
                             help='Per-step batch size (lm-eval default if omitted)')
    eval_parser.add_argument('--backend', default='hf', choices=['hf', 'vllm', 'mlx'],
                             help='lm-eval model adapter (hf works on every backend; '
                                  'vllm faster on CUDA/ROCm; mlx for Apple Silicon)')
    eval_parser.add_argument('--output', '-o',
                             help='Directory to write lm_eval_summary.json + raw results')
    eval_parser.add_argument('--list-tasks', action='store_true',
                             help='Print curated task groups and exit')

    # token command (Track P1) — API token lifecycle.
    token_parser = subparsers.add_parser(
        'token',
        help='Manage API tokens for the public API (auto-required when bound to non-loopback)',
    )
    token_subparsers = token_parser.add_subparsers(dest='token_command', required=True)

    token_create = token_subparsers.add_parser('create', help='Create a new bearer token')
    token_create.add_argument('name', help='Friendly name (e.g. "dashboard", "ci")')
    token_create.add_argument('--note', help='Free-form annotation')

    token_subparsers.add_parser('list', help='List existing tokens (no secrets shown)')

    token_revoke = token_subparsers.add_parser('revoke', help='Revoke a token by name')
    token_revoke.add_argument('name', help='Name of the token to revoke')

    # replay command (Track T15) — deterministic-replay manifest tools.
    replay_parser = subparsers.add_parser(
        'replay',
        help='Show or relaunch a captured run from its replay.json manifest',
    )
    replay_parser.add_argument('source',
                               help='Path to a run directory or replay.json file')
    replay_parser.add_argument('--launch', action='store_true',
                               help='Actually relaunch (subprocess) instead of just printing the command')
    replay_parser.add_argument('--force', action='store_true',
                               help='[--launch] Launch even if the env fingerprint differs')
    replay_parser.add_argument('--allow-dataset-drift', action='store_true',
                               help='[--launch] Launch even if a captured local dataset hash differs')

    # merge command (Tracks T12 + T13) — adapter bake / multi-adapter combine.
    merge_parser = subparsers.add_parser(
        'merge',
        help='Merge LoRA adapters (bake into base, or combine multiple via TIES/DARE)',
    )
    merge_parser.add_argument('--mode', '-m', choices=['bake', 'combine'], default='bake',
                              help='bake = single adapter into base; combine = N adapters into one')
    merge_parser.add_argument('--base', '-b',
                              help='Base model (HF id or local path). Required.')
    merge_parser.add_argument('--output', '-o',
                              help='Output directory.')
    # bake mode
    merge_parser.add_argument('--adapter', '-a',
                              help='[bake] Adapter directory to merge')
    # combine mode
    merge_parser.add_argument('--adapters',
                              help='[combine] Comma-separated adapter paths')
    merge_parser.add_argument('--weights',
                              help='[combine] Comma-separated weights (e.g. "0.5,0.3,0.2"). Defaults to uniform.')
    merge_parser.add_argument('--method',
                              default='dare_ties',
                              help='[combine] Merge method: linear / ties / dare_linear / dare_ties / magnitude_prune')
    merge_parser.add_argument('--bake-after-merge', action='store_true',
                              help='[combine] Also bake the combined adapter into the base; output is a merged checkpoint')
    merge_parser.add_argument('--svd-rank', type=int,
                              help='[combine] Override SVD rank for ties / dare_ties methods')
    merge_parser.add_argument('--trust-remote-code', action='store_true',
                              help='Opt into executing remote model code while loading merge inputs')
    merge_parser.add_argument('--list', action='store_true',
                              help='Print supported operations / methods and exit')

    # convert command (Track I5) — unified format conversion.
    convert_parser = subparsers.add_parser(
        'convert',
        help='Convert a model between formats (HF → MLX / GGUF / HF dtype recast)',
    )
    convert_parser.add_argument('--source', '-s',
                                help='HuggingFace id, mlx-community id, or local path of source model')
    convert_parser.add_argument('--output', '-o',
                                help='Output path (file for GGUF, directory for MLX/HF)')
    convert_parser.add_argument('--format', '-f', default='mlx',
                                choices=['mlx', 'gguf', 'hf'],
                                help='Target format (default: mlx)')
    convert_parser.add_argument('--quant', '-q', default='q4',
                                help='Normalized quant: q4, q8, fp16, bf16, fp32 (default: q4)')
    convert_parser.add_argument('--trust-remote-code', action='store_true',
                                help='Opt into executing remote model code while loading/converting')
    convert_parser.add_argument('--allow-unquantized-fallback', action='store_true',
                                help='For GGUF only: allow FP16 output if requested quantization cannot run')
    convert_parser.add_argument('--list', action='store_true',
                                help='Print supported formats / quants and exit')
    convert_parser.add_argument('--verify', action='store_true',
                                help='Track I4: after conversion, run a fixed prompt set '
                                     'through both source and exported and flag drift. '
                                     'Adds ~30s; catches silently-broken exports.')

    # serve command (Track I1) — OpenAI-compatible serving endpoint.
    serve_parser = subparsers.add_parser(
        'serve',
        help='Run an OpenAI-compatible serving endpoint for a trained model',
    )
    serve_parser.add_argument('--model', '-m', required=True,
                              help='Model id, mlx-community id, or local path to serve')
    serve_parser.add_argument('--host', default='127.0.0.1',
                              help='Bind host (default: 127.0.0.1; loopback only)')
    serve_parser.add_argument('--port', type=int, default=8001,
                              help='Bind port (default: 8001)')
    serve_parser.add_argument('--backend',
                              help='Force a backend (mlx, mps, cuda, rocm_gfx1151, cpu); '
                                   'defaults to autodetect')
    serve_parser.add_argument('--trust-remote-code', action='store_true',
                              help='Opt into executing remote model code while loading the served model')
    serve_parser.add_argument('--check', action='store_true',
                              help='Validate serving configuration and print endpoints without binding a port')

    # dashboard command — user-facing app, API + built React dashboard on one origin.
    dashboard_parser = subparsers.add_parser(
        'dashboard',
        aliases=['app'],
        help='Run the Halo Forge dashboard app',
    )
    dashboard_parser.add_argument('--host', default='127.0.0.1',
                                  help='Bind host (default: 127.0.0.1; use 0.0.0.0 for trusted-network access)')
    dashboard_parser.add_argument('--port', type=int, default=8000,
                                  help='Bind port (default: 8000)')
    dashboard_parser.add_argument('--check', action='store_true',
                                  help='Print dashboard startup details without binding a port')
    dashboard_parser.add_argument('--no-build', action='store_true',
                                  help='Do not auto-build public_app/dist when dashboard assets are missing')
    dashboard_parser.add_argument('--open', action='store_true',
                                  help='Open the dashboard URL in the default browser for loopback launches')

    # serve-public — dashboard FastAPI (the API the public_app SPA talks to)
    serve_public_parser = subparsers.add_parser(
        'serve-public',
        help='Run only the dashboard FastAPI for frontend development',
    )
    serve_public_parser.add_argument('--host', default='127.0.0.1',
                                     help='Bind host (default: 127.0.0.1; loopback skips bearer auth)')
    serve_public_parser.add_argument('--port', type=int, default=8000,
                                     help='Bind port (default: 8000)')
    serve_public_parser.add_argument('--check', action='store_true',
                                     help='Print dashboard API startup details without binding a port')

    # raft command
    raft_parser = subparsers.add_parser('raft', help='RAFT training')
    raft_subparsers = raft_parser.add_subparsers(dest='raft_command', required=True)
    
    # raft train
    raft_train_parser = raft_subparsers.add_parser('train', help='Run RAFT training')
    add_apple_runtime_flags(raft_train_parser, neural_accelerators=False)
    raft_train_parser.add_argument('--config', '-c', help='Config file path')
    raft_train_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-Coder-3B', help='Base model')
    # Phase 5a: when --accelerator mlx is set, rollouts run on MLX while
    # the policy update stays on PyTorch. --rollout-model lets you point at
    # an MLX-format weight set distinct from the torch base. If omitted, we
    # use --model for both (only works if it happens to be MLX-loadable).
    raft_train_parser.add_argument(
        '--rollout-model',
        help='MLX-format model used for rollouts when --accelerator mlx is set '
             '(e.g. mlx-community/Qwen2.5-3B-Instruct-bf16). Defaults to --model.',
    )
    # Phase 5a hybrid: --accelerator mlx --rollout-only keeps the PyTorch
    # RAFT trainer in charge but swaps in MLX-fast rollouts. Without this
    # flag, --accelerator mlx selects the full MLX-native RAFT trainer
    # (Phase 5b).
    raft_train_parser.add_argument(
        '--rollout-only',
        action='store_true',
        help='[--accelerator mlx] Hybrid mode: MLX rollouts + PyTorch policy update. '
             'Without this, --accelerator mlx runs RAFT entirely on MLX.',
    )
    # Track I6 — vLLM rollouts for CUDA/ROCm. Largest single throughput
    # win available for RAFT without changing the algorithm.
    raft_train_parser.add_argument(
        '--rollout-engine',
        default='auto',
        choices=['auto', 'torch', 'vllm', 'mlx'],
        help='Generation engine for the rollout stage. "auto" picks torch '
             '(default; HF generate). "vllm" uses continuous-batched '
             'inference (CUDA / ROCm only). "mlx" uses mlx_lm.generate '
             '(Apple Silicon; equivalent throughput story to vllm on its '
             'native hardware).',
    )
    raft_train_parser.add_argument('--checkpoint', help='SFT checkpoint path (optional)')
    raft_train_parser.add_argument('--prompts', '-p', help='Prompts file')
    raft_train_parser.add_argument('--output', '-o', default='models/raft', help='Output directory')
    raft_train_parser.add_argument('--cycles', type=int, help='Number of RAFT cycles')
    raft_train_parser.add_argument('--verifier', default='gcc',
                                   choices=list(RAFT_TRAIN_SUPPORTED_VERIFIERS),
                                   help='Verifier type (parser matches runtime-supported options)')
    raft_train_parser.add_argument('--keep-percent', type=float, default=0.5, 
                                   help='Keep top X%% of passing samples (0.0-1.0, default: 0.5 = 50%%)')
    raft_train_parser.add_argument('--reward-threshold', type=float, default=0.5,
                                   help='Minimum reward to consider sample passing (default: 0.5)')
    raft_train_parser.add_argument(
        '--allow-compile-only-training',
        action='store_true',
        help='Allow compile-only verifier results to train RAFT samples. Disabled by default.',
    )
    raft_train_parser.add_argument(
        '--unsafe-verifier-execution',
        action='store_true',
        help='Run generated-code verifiers directly on the host instead of the sandbox. Dangerous; disabled by default.',
    )
    raft_train_parser.add_argument('--curriculum', default='none',
                                   choices=['none', 'complexity', 'progressive', 'adaptive', 'historical'],
                                   help='Curriculum learning strategy (default: none)')
    raft_train_parser.add_argument('--curriculum-stats', type=str, default=None,
                                   help='Path to historical stats JSON for historical curriculum')
    raft_train_parser.add_argument('--curriculum-start', type=float, default=0.2,
                                   help='Progressive curriculum: start with this fraction of prompts (default: 0.2)')
    raft_train_parser.add_argument('--curriculum-increment', type=float, default=0.2,
                                   help='Progressive curriculum: add this fraction each cycle (default: 0.2)')
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
    raft_train_parser.add_argument('--samples-per-prompt', type=int, default=8,
                                   help='Samples to generate per prompt (default: 8)')
    raft_train_parser.add_argument('--temperature', type=float, default=0.7,
                                   help='Sampling temperature (default: 0.7)')
    raft_train_parser.add_argument('--max-new-tokens', type=int, default=1024,
                                   help='Maximum tokens to generate (default: 1024)')
    raft_train_parser.add_argument('--min-samples', type=int,
                                   help='Minimum samples per cycle (auto-adjusts threshold if needed)')
    # Training hyperparameters
    raft_train_parser.add_argument('--learning-rate', type=float,
                                   help='Base learning rate (default: 5e-5)')
    raft_train_parser.add_argument('--batch-size', type=int,
                                   help='Per-device batch size (default: 2)')
    raft_train_parser.add_argument('--gradient-accumulation', type=int,
                                   help='Gradient accumulation steps (default: 16)')
    raft_train_parser.add_argument('--warmup-steps', type=int,
                                   help='LR warmup steps (default: 10)')
    # LoRA configuration
    raft_train_parser.add_argument('--lora-rank', type=int,
                                   help='LoRA rank (default: 16)')
    raft_train_parser.add_argument('--lora-alpha', type=int,
                                   help='LoRA alpha (default: 32)')
    # Verifier options
    raft_train_parser.add_argument('--host', help='MSVC verifier host')
    raft_train_parser.add_argument('--user', help='MSVC verifier user')
    raft_train_parser.add_argument('--ssh-key', help='MSVC verifier SSH key')
    
    # benchmark command (for reporting, not training)
    bench_parser = subparsers.add_parser('benchmark', 
        help='Benchmark reporting (compare to papers). For training verification, use RAFT.')
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
                                   choices=['gcc', 'mingw', 'msvc', 'rust', 'go', 'dotnet', 'powershell', 'auto', 'humaneval', 'mbpp', 'python'],
                                   help='Verifier type (humaneval/mbpp/python for Python, auto=multi-language)')
    bench_run_parser.add_argument('--base-model', default='Qwen/Qwen2.5-Coder-7B', help='Base model')
    bench_run_parser.add_argument('--system-prompt', default='You are an expert Windows systems programmer.', help='System prompt')
    bench_run_parser.add_argument('--host', help='MSVC host')
    bench_run_parser.add_argument('--user', help='MSVC user')
    bench_run_parser.add_argument('--ssh-key', help='MSVC SSH key')
    bench_run_parser.add_argument('--cross-compile', action='store_true', help='Enable Windows cross-compilation for rust/go')
    bench_run_parser.add_argument('--run-after-compile', action='store_true', help='Run compiled code after compile')
    bench_run_parser.add_argument('--unsafe-verifier-execution', action='store_true',
                                  help='Run generated-code verifiers directly on the host instead of the sandbox. Dangerous; disabled by default.')
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
    
    # benchmark eval (simple code evaluation on standard datasets)
    bench_eval_parser = bench_subparsers.add_parser('eval', help='Evaluate model on standard code benchmarks')
    bench_eval_parser.add_argument('--model', '-m', required=True, help='Model name or path')
    bench_eval_parser.add_argument('--benchmark', '-b', default='humaneval',
                                   choices=['humaneval', 'mbpp', 'livecodebench', 'cpp', 'rust', 'go'],
                                   help='Benchmark dataset (default: humaneval)')
    bench_eval_parser.add_argument('--limit', type=int, help='Max samples to evaluate')
    bench_eval_parser.add_argument('--output', '-o', help='Output file path')
    bench_eval_parser.add_argument('--samples-per-prompt', type=int, default=5,
                                   help='Samples per prompt for pass@k (default: 5)')
    bench_eval_parser.add_argument('--run-after-compile', action='store_true',
                                   help='Run compiled code (MVR mode). Default: compile-only (MVP)')
    bench_eval_parser.add_argument('--language',
                                   choices=['cpp', 'rust', 'go', 'python'],
                                   help='Target language for native benchmarks')
    bench_eval_parser.add_argument('--verifier',
                                   choices=['gcc', 'mingw', 'clang', 'rust', 'go', 'humaneval', 'mbpp'],
                                   help='Verifier type')
    
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
    add_apple_runtime_flags(vlm_train_parser)
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
    vlm_train_parser.add_argument('--max-new-tokens', type=int, default=512,
                                  help='Maximum tokens to generate (default: 512)')
    vlm_train_parser.add_argument('--keep-percent', type=float, default=0.5,
                                  help='Keep top X%% of passing samples (default: 0.5)')
    vlm_train_parser.add_argument('--reward-threshold', type=float, default=0.5,
                                  help='Minimum reward to consider passing (default: 0.5)')
    vlm_train_parser.add_argument('--limit', type=int, help='Limit dataset samples')
    vlm_train_parser.add_argument('--resume-from-cycle', type=int, default=0,
                                  help='Resume training from this cycle index (default: 0)')
    vlm_train_parser.add_argument('--seed', type=int, default=42,
                                  help='Random seed for deterministic runs (default: 42)')
    vlm_train_parser.add_argument('--dry-run', action='store_true',
                                  help='Validate config and datasets without running training')
    vlm_train_parser.add_argument(
        '--allow-prototype-train',
        action='store_true',
        help='Required while VLM training capability is prototype-gated',
    )
    
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
    add_apple_runtime_flags(audio_train_parser)
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
    audio_train_parser.add_argument('--samples-per-prompt', type=int, default=4,
                                    help='Samples per prompt (default: 4)')
    audio_train_parser.add_argument('--temperature', type=float, default=0.7,
                                    help='Generation temperature (default: 0.7)')
    audio_train_parser.add_argument('--keep-percent', type=float, default=0.5,
                                    help='Keep top X%% of passing samples (default: 0.5)')
    audio_train_parser.add_argument('--reward-threshold', type=float, default=0.5,
                                    help='Minimum reward to consider passing (default: 0.5)')
    audio_train_parser.add_argument('--output', '-o', default='models/audio_raft',
                                    help='Output directory (default: models/audio_raft)')
    audio_train_parser.add_argument('--resume-from-cycle', type=int, default=0,
                                    help='Resume training from this cycle index (default: 0)')
    audio_train_parser.add_argument('--seed', type=int, default=42,
                                    help='Random seed for deterministic runs (default: 42)')
    audio_train_parser.add_argument('--dry-run', action='store_true',
                                    help='Validate config without running training')
    audio_train_parser.add_argument(
        '--allow-prototype-train',
        action='store_true',
        help='Required while audio training capability is prototype-gated',
    )
    
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
    add_apple_runtime_flags(reasoning_train_parser)
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
    reasoning_train_parser.add_argument('--samples-per-prompt', type=int, default=4,
                                        help='Samples per prompt (default: 4)')
    reasoning_train_parser.add_argument('--temperature', type=float, default=0.7,
                                        help='Generation temperature (default: 0.7)')
    reasoning_train_parser.add_argument('--keep-percent', type=float, default=0.5,
                                        help='Keep top X%% of passing samples (default: 0.5)')
    reasoning_train_parser.add_argument('--output', '-o', default='models/reasoning_raft',
                                        help='Output directory (default: models/reasoning_raft)')
    reasoning_train_parser.add_argument('--limit', type=int, help='Limit dataset samples')
    reasoning_train_parser.add_argument('--resume-from-cycle', type=int, default=0,
                                        help='Resume training from this cycle index (default: 0)')
    reasoning_train_parser.add_argument('--seed', type=int, default=42,
                                        help='Random seed for deterministic runs (default: 42)')
    reasoning_train_parser.add_argument('--dry-run', action='store_true',
                                        help='Validate config without running training')
    reasoning_train_parser.add_argument(
        '--allow-prototype-train',
        action='store_true',
        help='Required while reasoning training capability is prototype-gated',
    )
    
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
    add_apple_runtime_flags(agentic_train_parser)
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
    agentic_train_parser.add_argument('--samples-per-prompt', type=int, default=4,
                                      help='Samples per prompt (default: 4)')
    agentic_train_parser.add_argument('--temperature', type=float, default=0.7,
                                      help='Generation temperature (default: 0.7)')
    agentic_train_parser.add_argument('--keep-percent', type=float, default=0.5,
                                      help='Keep top X%% of passing samples (default: 0.5)')
    agentic_train_parser.add_argument('--output', '-o', default='models/agentic_raft',
                                      help='Output directory (default: models/agentic_raft)')
    agentic_train_parser.add_argument('--limit', type=int, help='Limit dataset samples')
    agentic_train_parser.add_argument('--resume-from-cycle', type=int, default=0,
                                      help='Resume training from this cycle index (default: 0)')
    agentic_train_parser.add_argument('--seed', type=int, default=42,
                                      help='Random seed for deterministic runs (default: 42)')
    agentic_train_parser.add_argument('--dry-run', action='store_true',
                                      help='Validate config without running training')
    agentic_train_parser.add_argument(
        '--allow-prototype-train',
        action='store_true',
        help='Required while agentic training capability is prototype-gated',
    )
    
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

    # doctor command
    doctor_parser = subparsers.add_parser('doctor', help='Run environment readiness checks')
    doctor_subparsers = doctor_parser.add_subparsers(dest='doctor_command', required=True)
    doctor_mlx_parser = doctor_subparsers.add_parser('mlx', help='Check Apple MLX package and Metal runtime readiness')
    doctor_mlx_parser.add_argument('--json', action='store_true', help='Emit JSON')

    # models command
    models_parser = subparsers.add_parser('models', help='Browse curated base-model catalog')
    models_subparsers = models_parser.add_subparsers(dest='models_command', required=True)
    models_list_parser = models_subparsers.add_parser('list', help='List recommended and compatible models')
    models_list_parser.add_argument('--mode', help='Filter by trainer/mode (sft, raft, dpo, grpo, vlm, audio, ...)')
    models_list_parser.add_argument('--backend', help='Filter by backend (cuda, rocm, mps, mlx, cpu)')
    models_list_parser.add_argument('--modality', help='Filter by modality (text, code, vision, audio)')
    models_list_parser.add_argument('--provider', help='Filter by provider (Qwen, Liquid AI, Meta, ...)')
    models_list_parser.add_argument('--status', help='Filter by status (recommended, compatible, experimental)')
    models_list_parser.add_argument('--memory-tier', help='Filter by memory tier (tiny, small, medium, large)')
    models_list_parser.add_argument('--json', action='store_true', help='Emit JSON')
    models_show_parser = models_subparsers.add_parser('show', help='Show one model catalog entry')
    models_show_parser.add_argument('model_id', help='Model id, e.g. Qwen/Qwen2.5-Coder-3B')
    models_show_parser.add_argument('--json', action='store_true', help='Emit JSON')
    
    # plot command - visualization tools
    plot_parser = subparsers.add_parser('plot', help='Generate training/benchmark visualizations')
    plot_subparsers = plot_parser.add_subparsers(dest='plot_command')
    
    # plot training
    plot_training_parser = plot_subparsers.add_parser('training', 
        help='Generate charts from TensorBoard training logs')
    plot_training_parser.add_argument('log_dirs', nargs='+',
        help='TensorBoard log directory (e.g., models/code_sft/logs)')
    plot_training_parser.add_argument('--output', '-o', default=None,
        help='Output directory for charts')
    plot_training_parser.add_argument('--compare', action='store_true',
        help='Generate comparison charts for multiple runs')
    plot_training_parser.add_argument('--only', choices=['loss', 'lr', 'grad', 'summary'],
        help='Generate only specific chart type')
    plot_training_parser.add_argument('--name', default=None,
        help='Override run name in chart titles')
    
    # plot benchmarks
    plot_benchmarks_parser = plot_subparsers.add_parser('benchmarks',
        help='Generate charts from benchmark results')
    plot_benchmarks_parser.add_argument('results_dir',
        help='Directory containing benchmark results')
    plot_benchmarks_parser.add_argument('--output', '-o', default=None,
        help='Output directory for charts')
    
    # test command
    test_parser = subparsers.add_parser('test', help='Run pipeline validation tests')
    test_parser.add_argument('--level', '-l', default='standard',
                             choices=['smoke', 'standard', 'full', 'modality', 'ops-e2e', 'ops-burnin', 'all-modules', 'walkthroughs', 'all-module-qualification', 'all-module-bootstrap', 'all-module-live'],
                             help='Test level: smoke, standard, full, modality, ops-e2e, ops-burnin, all-modules, walkthroughs, all-module-qualification, all-module-bootstrap, all-module-live')
    test_parser.add_argument('--model', '-m', default='Qwen/Qwen2.5-Coder-0.5B',
                             help='Model to use for testing (default: Qwen2.5-Coder-0.5B)')
    test_parser.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output with detailed logging')
    test_parser.add_argument(
        '--baseline-file',
        default='tests/baselines/modality_runtime_baseline.v1.json',
        help='Baseline JSON path for modality/ops-burnin/all-module-qualification drift checks',
    )
    test_parser.add_argument(
        '--write-baseline',
        action='store_true',
        help='Write/overwrite modality, ops-burnin, or all-module-qualification baseline snapshot',
    )
    test_parser.add_argument(
        '--compare-baseline',
        action='store_true',
        help='Compare modality, ops-burnin, or all-module-qualification run against baseline and fail on hard drift',
    )
    test_parser.add_argument(
        '--report-file',
        default='results/readiness/ops_e2e_launch_reliability.v1.json',
        help='Output report path for --level ops-e2e, ops-burnin, all-modules, walkthroughs, all-module-qualification, all-module-bootstrap, or all-module-live',
    )
    test_parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail non-zero when module status is fail (used with --level ops-e2e, ops-burnin, all-modules, walkthroughs, all-module-qualification, all-module-bootstrap, all-module-live)',
    )
    test_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Deterministic seed for ops-e2e/ops-burnin/all-modules/walkthroughs/all-module-qualification/all-module-bootstrap/all-module-live checks (default: 42)',
    )
    test_parser.add_argument(
        '--burnin-profile',
        default='tiny-v1',
        help='Burn-in profile for --level ops-burnin (default: tiny-v1)',
    )
    test_parser.add_argument(
        '--profile',
        default='bounded-v1',
        help='Readiness profile for --level all-modules (default: bounded-v1); walkthroughs uses contract-v1/live-local',
    )
    test_parser.add_argument(
        '--qualification-profile',
        default='contract-v1',
        choices=['contract-v1', 'fixture-v1', 'live-local'],
        help='Qualification profile for --level all-module-qualification (default: contract-v1)',
    )
    test_parser.add_argument(
        '--module',
        action='append',
        default=[],
        help='Filter module(s) for --level all-modules, walkthroughs, all-module-qualification, all-module-bootstrap, or all-module-live (repeatable)',
    )
    test_parser.add_argument(
        '--fixture-pack',
        default='',
        help='Fixture pack for ops-e2e/all-modules/all-module-qualification checks (e.g., v1 or tests/fixtures/.../v1)',
    )
    test_parser.add_argument(
        '--show-fix-commands',
        action='store_true',
        help='Emit parseable remediation command lines for qualification issues (--level all-module-qualification)',
    )
    test_parser.add_argument(
        '--bootstrap-profile',
        default='contract-v1',
        choices=['contract-v1', 'live-local'],
        help='Bootstrap profile for --level all-module-bootstrap (default: contract-v1)',
    )
    test_parser.add_argument(
        '--output-root',
        default='results/bootstrap',
        help='Evidence output root for --level all-module-bootstrap or --level all-module-live (default: results/bootstrap)',
    )
    test_parser.add_argument(
        '--live-profile',
        default='live-smoke-v1',
        choices=['live-smoke-v1', 'live-local'],
        help='Live execution profile for --level all-module-live (default: live-smoke-v1)',
    )
    test_parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute bounded command probes (used with --level walkthroughs and profile=live-local)',
    )
    
        # The legacy `halo-forge ui` command launched a NiceGUI web app —
    # retired in favor of the Vite + React frontend at `public_app/`.
    # If you got here from an old script: `cd public_app && npm run dev`.

    # Parse arguments and dispatch
    args = parser.parse_args()

    # Plumb --accelerator into HALOFORGE_BACKEND so every downstream
    # halo_forge.backend.get_backend() call (in trainers, inference,
    # public_api) sees the user's choice without requiring each subcommand
    # handler to thread the flag through manually. Note: distinct from
    # `args.backend` on the `data generate` subcommand (LLM API selection).
    accelerator_choice = getattr(args, 'accelerator', 'auto')
    if accelerator_choice and accelerator_choice != 'auto':
        os.environ['HALOFORGE_BACKEND'] = accelerator_choice

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
    print("  halo-forge reasoning train --dataset gsm8k --cycles 4 --seed 42")


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
    summary = trainer.train()
    _print_completed_training_summary("reasoning_sft", args.output, summary)


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
            dtype=recommended_dtype(),
            device_map=get_device_map(),
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
    from tqdm import tqdm
    for sample in tqdm(dataset, desc="Evaluating", unit="sample"):
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
    print(f"Seed: {args.seed}")

    _enforce_modality_train_contract("reasoning", args)
    
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
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        seed=args.seed,
    )
    
    # Load dataset
    dataset = load_math_dataset(args.dataset, split="train", limit=args.limit)
    print(f"\nLoaded {len(dataset)} samples from {args.dataset}")
    
    # Train
    trainer = ReasoningRAFTTrainer(config)
    try:
        summary = trainer.train(
            list(dataset),
            resume_from_cycle=getattr(args, "resume_from_cycle", 0),
        )
    except ValueError as e:
        print(f"{RED}Training error: {e}{NC}")
        sys.exit(2)
    _enforce_training_outcome_or_exit("reasoning", summary)
    
    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Final accuracy: {summary.get('final_accuracy', 0):.1%}")
    if summary.get("final_model_path"):
        print(f"Final model: {summary['final_model_path']}")
    _print_training_run_metadata(summary)
    total_steps = sum(
        int(c.get("train_steps_executed", 0))
        for c in summary.get("cycles", [])
    )
    print(f"Train steps executed: {total_steps}")
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
    summary = trainer.train()
    _print_completed_training_summary("agentic_sft", args.output, summary)


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
    print(f"Seed: {args.seed}")

    _enforce_modality_train_contract("agentic", args)
    
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
        enable_neural_accelerators=getattr(args, "enable_neural_accelerators", False),
        seed=args.seed,
    )
    
    # Train
    trainer = AgenticRAFTTrainer(config)
    try:
        results = trainer.train(
            samples,
            resume_from_cycle=getattr(args, "resume_from_cycle", 0),
        )
    except ValueError as e:
        print(f"{RED}Training error: {e}{NC}")
        sys.exit(2)
    total_steps = int(results.get("total_train_steps_executed", 0))
    final_loss = results.get("final_train_loss")
    _enforce_training_outcome_or_exit("agentic", results)
    
    print(f"\n{GREEN}Training complete!{NC}")
    print(f"Final accuracy: {results.get('final_success_rate', 0):.1%}")
    print(f"Final avg reward: {results.get('final_avg_reward', 0):.3f}")
    if results.get("final_model_path"):
        print(f"Final model: {results['final_model_path']}")
    _print_training_run_metadata(results)
    print(f"Train steps executed: {total_steps}")
    if isinstance(final_loss, (int, float)):
        print(f"Final train loss: {final_loss:.4f}")
    print(f"Results saved to: {args.output}")


# The test parser and dispatch logic is inside main() at line 1598
# These are the remaining handler functions that were placed after main()

def _dispatch_commands(args):
    """Dispatch to appropriate command handler."""
    
    # Commands that should have auto-logging enabled
    logged_commands = {
        ('raft', 'train'): 'raft_train',
        ('sft', 'train'): 'sft_train',
        ('dpo', 'train'): 'dpo_train',
        ('orpo', 'train'): 'orpo_train',
        ('grpo', 'train'): 'grpo_train',
        ('rm', 'train'): 'rm_train',
        ('vlm', 'train'): 'vlm_train',
        ('audio', 'train'): 'audio_train',
        ('reasoning', 'train'): 'reasoning_train',
        ('agentic', 'train'): 'agentic_train',
        ('benchmark', 'run'): 'benchmark_run',
        ('benchmark', 'full'): 'benchmark_full',
        ('benchmark', 'eval'): 'benchmark_eval',
    }
    
    # Setup auto-logging for training/benchmark commands
    quiet = getattr(args, 'quiet', False)
    subcommand = None
    if args.command == 'raft':
        subcommand = getattr(args, 'raft_command', None)
    elif args.command == 'sft':
        subcommand = getattr(args, 'sft_command', None)
    elif args.command == 'dpo':
        subcommand = getattr(args, 'dpo_command', None)
    elif args.command == 'orpo':
        subcommand = getattr(args, 'orpo_command', None)
    elif args.command == 'grpo':
        subcommand = getattr(args, 'grpo_command', None)
    elif args.command == 'rm':
        subcommand = getattr(args, 'rm_command', None)
    elif args.command == 'vlm':
        subcommand = getattr(args, 'vlm_command', None)
    elif args.command == 'audio':
        subcommand = getattr(args, 'audio_command', None)
    elif args.command == 'reasoning':
        subcommand = getattr(args, 'reasoning_command', None)
    elif args.command == 'agentic':
        subcommand = getattr(args, 'agentic_command', None)
    elif args.command == 'benchmark':
        subcommand = getattr(args, 'bench_command', None)
    
    log_key = (args.command, subcommand) if subcommand else None
    if log_key in logged_commands:
        log_name = logged_commands[log_key]
        log_path = setup_auto_logging(log_name, quiet=quiet)
        if not quiet:
            print(f"Logging to: {log_path}")
    
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
        elif args.data_command == 'dedup':
            cmd_data_dedup(args)
        elif args.data_command == 'score':
            cmd_data_score(args)
        elif args.data_command == 'synthesize':
            cmd_data_synthesize(args)
    elif args.command == 'sft':
        if args.sft_command == 'train':
            cmd_sft_train(args)
        elif args.sft_command == 'datasets':
            cmd_sft_datasets(args)
    elif args.command == 'dpo':
        if args.dpo_command == 'train':
            cmd_dpo_train(args)
        elif args.dpo_command == 'datasets':
            cmd_dpo_datasets(args)
    elif args.command == 'orpo':
        if args.orpo_command == 'train':
            cmd_orpo_train(args)
    elif args.command == 'grpo':
        if args.grpo_command == 'train':
            cmd_grpo_train(args)
    elif args.command == 'rm':
        if args.rm_command == 'train':
            cmd_rm_train(args)
    elif args.command == 'serve':
        cmd_serve(args)
    elif args.command in {'dashboard', 'app'}:
        cmd_dashboard(args)
    elif args.command == 'serve-public':
        cmd_serve_public(args)
    elif args.command == 'convert':
        cmd_convert(args)
    elif args.command == 'merge':
        cmd_merge(args)
    elif args.command == 'replay':
        cmd_replay(args)
    elif args.command == 'token':
        cmd_token(args)
    elif args.command == 'eval':
        cmd_eval(args)
    elif args.command == 'probe':
        cmd_probe(args)
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
        elif args.bench_command == 'eval':
            cmd_benchmark_eval(args)
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
    elif args.command == 'doctor':
        cmd_doctor(args)
    elif args.command == 'models':
        cmd_models(args)
    elif args.command == 'plot':
        if not hasattr(args, 'plot_command') or not args.plot_command:
            print("Usage: halo-forge plot {training|benchmarks} ...")
            print("\nAvailable commands:")
            print("  training    Generate charts from TensorBoard training logs")
            print("  benchmarks  Generate charts from benchmark results")
        elif args.plot_command == 'training':
            cmd_plot_training(args)
        elif args.plot_command == 'benchmarks':
            cmd_plot_benchmarks(args)
    elif args.command == 'test':
        cmd_test(args)


if __name__ == '__main__':
    main()
