#!/usr/bin/env python3
"""
RAFT Training: Reward-Ranked Fine-Tuning

Iterative training loop:
1. Generate samples from model
2. Verify with pluggable verifier
3. Filter by reward threshold
4. SFT on filtered samples
5. Repeat

Works with any Verifier implementation.
"""

import torch
import time
import json
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Type
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import Dataset
from tqdm import tqdm

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult

# Rich UI (optional, falls back to plain print)
try:
    from halo_forge import ui
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


@dataclass
class RAFTConfig:
    """Configuration for RAFT training."""
    
    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-7B"
    sft_checkpoint: str = "models/sft/final_model"
    output_dir: str = "models/raft"
    
    # RAFT parameters
    num_cycles: int = 3
    samples_per_prompt: int = 8
    reward_threshold: float = 0.5  # Minimum reward to keep
    keep_top_percent: float = 0.5  # Keep top X% above threshold
    
    # Generation
    max_new_tokens: int = 1024
    temperature: float = 0.7
    generation_batch_size: int = 8  # Match strix-edr-training
    
    # Training (per cycle)
    train_epochs: int = 1
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    
    # System prompt for generation
    system_prompt: str = "You are an expert programmer."


class RAFTTrainer:
    """
    RAFT (Reward-Ranked Fine-Tuning) trainer.
    
    Iteratively generates, verifies, filters, and trains on high-reward samples.
    Uses pluggable verifiers for domain-agnostic verification.
    
    Example:
        from halo_forge.rlvr.verifiers import GCCVerifier
        
        verifier = GCCVerifier()
        trainer = RAFTTrainer(
            verifier=verifier,
            sft_checkpoint="models/sft/final_model"
        )
        trainer.run(prompts, num_cycles=3)
    """
    
    def __init__(
        self,
        verifier: Verifier,
        config: Optional[RAFTConfig] = None,
        sft_checkpoint: Optional[str] = None
    ):
        """
        Initialize RAFT trainer.
        
        Args:
            verifier: Verifier instance for checking samples
            config: RAFT configuration
            sft_checkpoint: Path to SFT checkpoint (overrides config)
        """
        self.config = config or RAFTConfig()
        if sft_checkpoint:
            self.config.sft_checkpoint = sft_checkpoint
        
        self.verifier = verifier
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # UI helpers
        self.use_rich = HAS_RICH
        
        # Load model and tokenizer
        self._load_model()
        
        # Statistics
        self.cycle_stats = []
    
    def _log(self, msg: str, level: str = "info"):
        """Log a message with optional rich formatting."""
        if self.use_rich:
            if level == "success":
                ui.print_success(msg)
            elif level == "error":
                ui.print_error(msg)
            elif level == "warning":
                ui.print_warning(msg)
            elif level == "dim":
                ui.print_dim(msg)
            elif level == "step":
                ui.print_step(msg, "running")
            else:
                ui.print_info(msg)
        else:
            print(msg)
    
    def _load_model(self):
        """Load base model and optionally SFT adapters."""
        cfg = self.config
        
        self._log("Loading model...", "step")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
            padding_side='left'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self._log(f"Loading base model: {cfg.base_model}", "dim")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )
        
        # Check if SFT checkpoint has PEFT adapters
        checkpoint_path = Path(cfg.sft_checkpoint)
        has_peft = (checkpoint_path.exists() and 
                    (checkpoint_path / "adapter_config.json").exists())
        
        if has_peft:
            # Load existing PEFT adapters
            self._log(f"Loading LoRA adapters from: {cfg.sft_checkpoint}", "dim")
            self.model = PeftModel.from_pretrained(
                self.base_model,
                cfg.sft_checkpoint,
                is_trainable=True
            )
        else:
            # Create new LoRA adapters (fresh start from base model)
            self._log("No SFT checkpoint found, creating new LoRA adapters...", "warning")
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
            self._log(f"Created LoRA adapters with rank={lora_config.r}", "dim")
        
        self.model.enable_input_require_grads()
        self._log("Model loaded", "success")
    
    def generate_samples(
        self,
        prompts: List[str],
        num_samples: int = None,
        max_new_tokens: int = None,
        temperature: float = None,
        batch_size: int = None
    ) -> List[Tuple[str, str]]:
        """
        Generate samples for each prompt.
        
        Args:
            prompts: List of prompts
            num_samples: Samples per prompt (default from config)
            max_new_tokens: Max tokens (default from config)
            temperature: Sampling temperature (default from config)
            batch_size: Generation batch size (default from config)
        
        Returns:
            List of (prompt, completion) tuples
        """
        cfg = self.config
        num_samples = num_samples or cfg.samples_per_prompt
        max_new_tokens = max_new_tokens or cfg.max_new_tokens
        temperature = temperature or cfg.temperature
        batch_size = batch_size or cfg.generation_batch_size
        
        total = len(prompts) * num_samples
        
        if self.use_rich:
            ui.print_header("Generation", f"{total} samples ({len(prompts)} prompts x {num_samples} samples)")
        else:
            print(f"\nGenerating {total} samples...")
            print(f"  {len(prompts)} prompts x {num_samples} samples/prompt")
        
        self.model.eval()
        all_samples = []
        start_time = time.time()
        
        # Use rich progress bar when available
        batch_count = (len(prompts) + batch_size - 1) // batch_size
        if self.use_rich:
            progress = ui.create_progress()
            progress.start()
            task = progress.add_task("Generating", total=batch_count)
        else:
            progress = None
        
        for batch_idx, i in enumerate(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i+batch_size]
            
            if progress:
                progress.update(task, completed=batch_idx + 1, description=f"Generating batch {batch_idx + 1}/{batch_count}")
            
            # Format with chat template
            formatted = []
            for prompt in batch_prompts:
                messages = [
                    {"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted.append(formatted_prompt)
            
            # Tokenize
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_samples,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode
            completions = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            # Pair prompts with completions
            for j, prompt in enumerate(batch_prompts):
                start_idx = j * num_samples
                end_idx = (j + 1) * num_samples
                prompt_completions = completions[start_idx:end_idx]
                
                for completion in prompt_completions:
                    all_samples.append((prompt, completion))
            
            # NOTE: Explicit memory cleanup removed - causes GPU hangs on ROCm/HIP
            # strix-edr-training works fine without cleanup, so we match that behavior
        
        if progress:
            progress.stop()
        
        elapsed = time.time() - start_time
        self._log(f"Generated {len(all_samples)} samples in {elapsed/60:.1f} minutes", "success")
        
        return all_samples
    
    def verify_and_filter(
        self,
        samples: List[Tuple[str, str]]
    ) -> Tuple[List[Dict], Dict]:
        """
        Verify samples and filter by reward.
        
        Uses chunked verification to prevent memory exhaustion on large batches.
        
        Args:
            samples: List of (prompt, completion) tuples
        
        Returns:
            (filtered_samples, stats)
        """
        cfg = self.config
        
        if self.use_rich:
            ui.print_header("Verification", f"{len(samples)} samples")
        else:
            print(f"\nVerifying {len(samples)} samples...")
        
        # Extract prompts and completions for verification
        prompts = [s[0] for s in samples]
        completions = [s[1] for s in samples]
        
        # CHUNKED verification to prevent memory exhaustion
        # Process 200 samples at a time to avoid OOM
        chunk_size = 200
        results = []
        
        start_time = time.time()
        for i in range(0, len(completions), chunk_size):
            chunk_end = min(i + chunk_size, len(completions))
            chunk_prompts = prompts[i:chunk_end]
            chunk_completions = completions[i:chunk_end]
            
            self._log(f"Processing chunk {i//chunk_size + 1}/{(len(completions) + chunk_size - 1)//chunk_size} ({len(chunk_completions)} samples)", "dim")
            
            # Pass both prompts and completions (for verifiers that need context like HumanEval/MBPP)
            chunk_results = self.verifier.verify_batch(chunk_completions, chunk_prompts)
            results.extend(chunk_results)
            
            # Force garbage collection after each chunk
            gc.collect()
        
        elapsed = time.time() - start_time
        self._log(f"Verification completed in {elapsed:.1f}s", "success")
        
        # Combine with prompts and rewards
        all_data = []
        for (prompt, completion), result in zip(samples, results):
            all_data.append({
                'prompt': prompt,
                'completion': completion,
                'reward': result.reward,
                'success': result.success,
                'details': result.details
            })
        
        # Sort by reward
        all_data.sort(key=lambda x: x['reward'], reverse=True)
        
        # Filter by threshold
        above_threshold = [d for d in all_data if d['reward'] >= cfg.reward_threshold]
        
        # Keep top %
        keep_count = max(1, int(len(above_threshold) * cfg.keep_top_percent))
        filtered = above_threshold[:keep_count]
        
        # Statistics
        stats = {
            'total_samples': len(samples),
            'above_threshold': len(above_threshold),
            'kept': len(filtered),
            'avg_reward': sum(d['reward'] for d in all_data) / len(all_data) if all_data else 0,
            'avg_kept_reward': sum(d['reward'] for d in filtered) / len(filtered) if filtered else 0,
            'success_rate': sum(1 for d in all_data if d['success']) / len(all_data) if all_data else 0,
            'reward_distribution': {
                '0.0': sum(1 for d in all_data if d['reward'] < 0.2),
                '0.5': sum(1 for d in all_data if 0.4 <= d['reward'] < 0.6),
                '0.7': sum(1 for d in all_data if 0.6 <= d['reward'] < 0.9),
                '1.0': sum(1 for d in all_data if d['reward'] >= 0.9)
            }
        }
        
        # Print filtering summary
        if self.use_rich:
            ui.print_raft_summary({
                'generated': stats['total_samples'],
                'verified': stats['above_threshold'],
                'kept': stats['kept'],
                'compile_rate': stats['success_rate'],
                'avg_reward': stats['avg_reward']
            })
        else:
            print(f"\nFiltering results:")
            print(f"  Total: {stats['total_samples']}")
            print(f"  Above threshold ({cfg.reward_threshold}): {stats['above_threshold']} ({stats['above_threshold']/stats['total_samples']*100:.1f}%)")
            print(f"  Kept: {stats['kept']} ({stats['kept']/stats['total_samples']*100:.1f}%)")
            print(f"  Avg reward: {stats['avg_reward']:.3f}")
            print(f"  Success rate: {stats['success_rate']*100:.1f}%")
            print(f"\n  Reward distribution:")
            print(f"    0.0 (failed): {stats['reward_distribution']['0.0']}")
            print(f"    0.5 (compiled): {stats['reward_distribution']['0.5']}")
            print(f"    0.7 (runs): {stats['reward_distribution']['0.7']}")
            print(f"    1.0 (correct): {stats['reward_distribution']['1.0']}")
        
        return filtered, stats, all_data
    
    def train_on_filtered(
        self,
        filtered_samples: List[Dict],
        cycle: int
    ):
        """
        SFT on filtered samples.
        
        Args:
            filtered_samples: Samples that passed filtering
            cycle: Current cycle number
        """
        cfg = self.config
        
        if self.use_rich:
            ui.print_header("Training", f"{len(filtered_samples)} filtered samples")
        else:
            print(f"\nTraining on {len(filtered_samples)} filtered samples...")
        
        # Prepare dataset
        texts = []
        for sample in filtered_samples:
            messages = [
                {"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": sample['prompt']},
                {"role": "assistant", "content": sample['completion']}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False
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
        
        # Training args - optimized for Strix Halo
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / f"cycle_{cycle}"),
            num_train_epochs=cfg.train_epochs,
            per_device_train_batch_size=cfg.train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_steps=0,  # No warmup - prevents LR=0 with few training steps
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=1,
            save_steps=500,
            save_total_limit=2,
            optim="adamw_torch",
            report_to="tensorboard",
            logging_dir=str(self.output_dir / f"cycle_{cycle}" / "logs"),
            # CRITICAL: Required for Strix Halo unified memory
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            processing_class=self.tokenizer,  # Updated from deprecated 'tokenizer'
            data_collator=data_collator
        )
        
        trainer.train()
        
        # Save checkpoint
        checkpoint_path = self.output_dir / f"cycle_{cycle}_final"
        trainer.save_model(str(checkpoint_path))
        self._log(f"Saved checkpoint: {checkpoint_path}", "success")
    
    def run_cycle(
        self,
        prompts: List[str],
        cycle: int
    ) -> Dict:
        """
        Run one RAFT cycle.
        
        Args:
            prompts: Training prompts
            cycle: Cycle number
        
        Returns:
            Cycle statistics
        """
        if self.use_rich:
            ui.print_raft_cycle_header(cycle, self.config.num_cycles)
        else:
            print("\n" + "=" * 70)
            print(f"RAFT CYCLE {cycle}")
            print("=" * 70)
        
        cycle_start = time.time()
        
        # Cache paths
        samples_cache = self.output_dir / f"cycle_{cycle}_samples.jsonl"
        verified_cache = self.output_dir / f"cycle_{cycle}_verified.jsonl"
        final_checkpoint = self.output_dir / f"cycle_{cycle}_final"
        
        # Skip if already complete
        if final_checkpoint.exists():
            self._log(f"Cycle {cycle} already complete, skipping...", "dim")
            self._reload_model(str(final_checkpoint))
            return {'cycle': cycle, 'skipped': True}
        
        # Generate (or load from cache)
        if samples_cache.exists():
            self._log("Loading cached samples...", "dim")
            samples = []
            with open(samples_cache) as f:
                for line in f:
                    data = json.loads(line)
                    samples.append((data['prompt'], data['completion']))
            self._log(f"Loaded {len(samples)} samples from cache", "dim")
        else:
            samples = self.generate_samples(prompts)
            
            # Save cache
            with open(samples_cache, 'w') as f:
                for prompt, completion in samples:
                    f.write(json.dumps({'prompt': prompt, 'completion': completion}) + '\n')
        
        # Verify (or load from cache)
        if verified_cache.exists():
            self._log("Loading cached verification...", "dim")
            all_data = []
            with open(verified_cache) as f:
                for line in f:
                    all_data.append(json.loads(line))
            
            # FREE MEMORY: samples not needed when loading from verified cache
            try:
                del samples
            except NameError:
                pass
            gc.collect()
            self._log("Freed cached samples memory", "dim")
            
            # Apply filtering
            cfg = self.config
            above_threshold = [d for d in all_data if d['reward'] >= cfg.reward_threshold]
            above_threshold.sort(key=lambda x: x['reward'], reverse=True)
            keep_count = max(1, int(len(above_threshold) * cfg.keep_top_percent))
            filtered = above_threshold[:keep_count]
            
            stats = {
                'total_samples': len(all_data),
                'above_threshold': len(above_threshold),
                'kept': len(filtered),
                'avg_reward': sum(d['reward'] for d in all_data) / len(all_data),
                'avg_kept_reward': sum(d['reward'] for d in filtered) / len(filtered) if filtered else 0
            }
        else:
            filtered, stats, all_data = self.verify_and_filter(samples)
            
            # FREE MEMORY: samples no longer needed after verification
            del samples
            gc.collect()
            self._log("Freed generation samples memory", "dim")
            
            # Save cache
            with open(verified_cache, 'w') as f:
                for item in all_data:
                    f.write(json.dumps(item) + '\n')
        
        if len(filtered) == 0:
            self._log("No samples passed filtering!", "error")
            return None
        
        # Train
        self.train_on_filtered(filtered, cycle)
        
        # FREE MEMORY: Clear training data before model reload
        del filtered
        try:
            del all_data
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reload for next cycle
        self._reload_model(str(self.output_dir / f"cycle_{cycle}_final"))
        
        cycle_elapsed = time.time() - cycle_start
        
        cycle_stats = {
            'cycle': cycle,
            'elapsed_minutes': cycle_elapsed / 60,
            **stats
        }
        
        self.cycle_stats.append(cycle_stats)
        self._log(f"Cycle {cycle} complete in {cycle_elapsed/60:.1f} minutes", "success")
        
        return cycle_stats
    
    def _reload_model(self, checkpoint_path: str):
        """Reload model from checkpoint."""
        self._log(f"Reloading model from {checkpoint_path}", "step")
        
        # Free memory
        if hasattr(self, 'model'):
            del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Reload
        self.model = PeftModel.from_pretrained(
            self.base_model,
            checkpoint_path,
            is_trainable=True
        )
        self.model.enable_input_require_grads()
    
    def run(
        self,
        prompts: List[str],
        num_cycles: int = None
    ):
        """
        Run full RAFT training.
        
        Args:
            prompts: Training prompts
            num_cycles: Number of cycles (overrides config)
        """
        num_cycles = num_cycles or self.config.num_cycles
        
        cfg = self.config
        if self.use_rich:
            ui.print_banner()
            ui.print_header("RAFT Training", f"{num_cycles} cycles")
            ui.print_dim(f"  Reward threshold: {cfg.reward_threshold}")
            ui.print_dim(f"  Keep top: {cfg.keep_top_percent*100:.0f}% of passing samples")
            ui.print_dim(f"  Samples per prompt: {cfg.samples_per_prompt}")
        else:
            print(f"\nStarting RAFT training: {num_cycles} cycles")
            print(f"  Reward threshold: {cfg.reward_threshold}")
            print(f"  Keep top: {cfg.keep_top_percent*100:.0f}% of passing samples")
            print(f"  Samples per prompt: {cfg.samples_per_prompt}")
        
        for cycle in range(1, num_cycles + 1):
            stats = self.run_cycle(prompts, cycle)
            
            if stats is None:
                self._log(f"Cycle {cycle} failed. Stopping.", "error")
                break
        
        # Save statistics
        self.save_statistics()
        
        # Summary
        if self.use_rich:
            ui.print_divider()
            ui.print_header("RAFT Training Complete")
            
            for stats in self.cycle_stats:
                cycle = stats['cycle']
                if stats.get('skipped'):
                    ui.print_step(f"Cycle {cycle}", "skip")
                else:
                    ui.print_raft_summary({
                        'generated': stats.get('total_samples', 0),
                        'kept': stats.get('kept', 0),
                        'avg_reward': stats.get('avg_reward', 0)
                    })
        else:
            print("\n" + "=" * 70)
            print("RAFT TRAINING COMPLETE")
            print("=" * 70)
            
            for stats in self.cycle_stats:
                cycle = stats['cycle']
                if stats.get('skipped'):
                    print(f"\nCycle {cycle}: SKIPPED")
                else:
                    print(f"\nCycle {cycle}:")
                    print(f"  Time: {stats['elapsed_minutes']:.1f} min")
                    print(f"  Kept: {stats['kept']}/{stats['total_samples']}")
                    print(f"  Avg reward: {stats['avg_reward']:.3f}")
        
        # Cleanup
        self.verifier.cleanup()
        
        final_path = self.output_dir / f"cycle_{num_cycles}_final"
        self._log(f"Final model: {final_path}", "success")
        
        return str(final_path)
    
    def save_statistics(self):
        """Save training statistics."""
        stats_path = self.output_dir / "raft_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.cycle_stats, f, indent=2)
        self._log(f"Saved statistics: {stats_path}", "dim")

