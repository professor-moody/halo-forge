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

# TUI State Manager (optional, for live monitoring)
try:
    from halo_forge.tui.state import StateManager
    HAS_TUI = True
except ImportError:
    HAS_TUI = False


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
        
        # TUI State Manager for live monitoring
        self.state_manager = StateManager() if HAS_TUI else None
        self._paused = False
        self._stop_requested = False
        
        # Load model and tokenizer
        self._load_model()
        
        # Statistics
        self.cycle_stats = []
    
    def _log(self, msg: str, level: str = "info"):
        """Log a message with simple text prefix (works through pipes)."""
        prefixes = {
            "success": "[OK]",
            "error": "[ERROR]",
            "warning": "[!]",
            "dim": "  ",
            "step": "  >",
            "info": "  >",
        }
        prefix = prefixes.get(level, "")
        print(f"{prefix} {msg}", flush=True)
        
        # Also log to TUI state
        if self.state_manager:
            self.state_manager.add_log(msg, level)
    
    def _update_state(self, **kwargs):
        """Update TUI state if available."""
        if self.state_manager:
            self.state_manager.update(**kwargs)
    
    def _check_commands(self):
        """Check for and handle TUI commands (pause/stop)."""
        if not self.state_manager:
            return
        
        cmd = self.state_manager.get_command()
        if not cmd:
            return
        
        command = cmd.get("command", "")
        
        if command == "pause":
            self._paused = True
            self._update_state(status="paused")
            self._log("Training paused by user", "warning")
            
            # Wait loop
            while self._paused:
                time.sleep(0.5)
                cmd = self.state_manager.get_command()
                if cmd and cmd.get("command") == "resume":
                    self._paused = False
                    self._update_state(status="running")
                    self._log("Training resumed", "success")
                elif cmd and cmd.get("command") == "stop":
                    self._paused = False
                    self._stop_requested = True
                    break
        
        elif command == "stop":
            self._stop_requested = True
            self._update_state(status="stopping")
            self._log("Stop requested - will finish current batch", "warning")
        
        elif command == "resume":
            self._paused = False
            self._update_state(status="running")
    
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
        batch_size: int = None,
        cache_path: Optional[Path] = None
    ) -> List[Tuple[str, str]]:
        """
        Generate samples for each prompt with streaming checkpoints.
        
        Args:
            prompts: List of prompts
            num_samples: Samples per prompt (default from config)
            max_new_tokens: Max tokens (default from config)
            temperature: Sampling temperature (default from config)
            batch_size: Generation batch size (default from config)
            cache_path: Path to stream samples to (enables resume)
        
        Returns:
            List of (prompt, completion) tuples
        """
        cfg = self.config
        num_samples = num_samples or cfg.samples_per_prompt
        max_new_tokens = max_new_tokens or cfg.max_new_tokens
        temperature = temperature or cfg.temperature
        batch_size = batch_size or cfg.generation_batch_size
        
        total = len(prompts) * num_samples
        
        # Check for partial cache and resume
        all_samples = []
        start_batch = 0
        if cache_path and cache_path.exists():
            with open(cache_path) as f:
                for line in f:
                    data = json.loads(line)
                    all_samples.append((data['prompt'], data['completion']))
            
            # Calculate how many prompts were completed
            completed_prompts = len(all_samples) // num_samples
            start_batch = completed_prompts // batch_size
            
            if len(all_samples) > 0:
                self._log(f"Resuming from batch {start_batch + 1} ({len(all_samples)} samples already cached)", "dim")
        
        # Plain text header (works through pipes)
        print(f"\nGeneration: {total} samples ({len(prompts)} prompts x {num_samples} samples)")
        
        # Update TUI state
        self._update_state(
            phase="generate",
            step=0,
            total_steps=total,
            samples_generated=len(all_samples)
        )
        
        # If already complete, return cached
        if len(all_samples) >= total:
            self._log(f"Generation already complete ({len(all_samples)} samples cached)", "success")
            return all_samples
        
        self.model.eval()
        start_time = time.time()
        
        # Calculate batch info
        batch_count = (len(prompts) + batch_size - 1) // batch_size
        
        # Open cache file in append mode for streaming writes
        cache_file = open(cache_path, 'a') if cache_path else None
        
        try:
            # Use tqdm for progress (works through pipes)
            batch_iter = range(0, len(prompts), batch_size)
            pbar = tqdm(enumerate(batch_iter), total=batch_count, desc="Generating", 
                       initial=start_batch, unit="batch")
            
            for batch_idx, i in pbar:
                # Skip already-completed batches
                if batch_idx < start_batch:
                    continue
                    
                batch_prompts = prompts[i:i+batch_size]
                
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
                
                # Pair prompts with completions and stream to disk
                batch_samples = []
                for j, prompt in enumerate(batch_prompts):
                    start_idx = j * num_samples
                    end_idx = (j + 1) * num_samples
                    prompt_completions = completions[start_idx:end_idx]
                    
                    for completion in prompt_completions:
                        sample = (prompt, completion)
                        all_samples.append(sample)
                        batch_samples.append(sample)
                        
                        # Stream to cache file immediately
                        if cache_file:
                            cache_file.write(json.dumps({'prompt': prompt, 'completion': completion}) + '\n')
                
                # Flush to disk after each batch (ensures checkpoint)
                if cache_file:
                    cache_file.flush()
                
                # Update tqdm postfix with sample count
                pbar.set_postfix(samples=len(all_samples), refresh=True)
                
                # Update TUI state with progress
                self._update_state(
                    step=len(all_samples),
                    samples_generated=len(all_samples)
                )
                
                # Check for pause/stop commands between batches
                self._check_commands()
                if self._stop_requested:
                    self._log("Generation stopped by user", "warning")
                    break
                
            pbar.close()
                
        finally:
            if cache_file:
                cache_file.close()
        
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
        
        # Simple header (works through pipes)
        print(f"\nVerifying {len(samples)} samples...")
        
        # Update TUI state
        self._update_state(
            phase="verify",
            step=0,
            total_steps=len(samples)
        )
        
        # Extract prompts and completions for verification
        prompts = [s[0] for s in samples]
        completions = [s[1] for s in samples]
        
        # CHUNKED verification to prevent memory exhaustion
        # Process 200 samples at a time to avoid OOM
        chunk_size = 200
        results = []
        num_chunks = (len(completions) + chunk_size - 1) // chunk_size
        
        start_time = time.time()
        for i in tqdm(range(0, len(completions), chunk_size), desc="Verifying", unit="chunk", total=num_chunks):
            chunk_end = min(i + chunk_size, len(completions))
            chunk_prompts = prompts[i:chunk_end]
            chunk_completions = completions[i:chunk_end]
            
            # Pass both prompts and completions (for verifiers that need context like HumanEval/MBPP)
            chunk_results = self.verifier.verify_batch(chunk_completions, chunk_prompts)
            results.extend(chunk_results)
            
            # Update TUI state with progress
            self._update_state(step=len(results))
            
            # Add recent samples to TUI (show last few verified)
            if self.state_manager:
                for j, result in enumerate(chunk_results[-5:]):  # Last 5 from chunk
                    prompt_idx = i + len(chunk_results) - 5 + j
                    if prompt_idx >= 0 and prompt_idx < len(prompts):
                        self.state_manager.add_sample(
                            prompt=prompts[prompt_idx][:100],
                            reward=result.reward,
                            success=result.success
                        )
            
            # Force garbage collection after each chunk
            gc.collect()
        
        elapsed = time.time() - start_time
        print(f"[OK] Verification completed in {elapsed:.1f}s")
        
        # Update TUI state to filtering phase
        self._update_state(phase="filter")
        
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
        
        # Print filtering summary (plain text, works through pipes)
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
        
        # Update TUI with filtering results
        self._update_state(
            compile_rate=stats['success_rate'] * 100,
            samples_kept=stats['kept']
        )
        
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
        
        # Plain text header (works through pipes)
        print(f"\nTraining: {len(filtered_samples)} filtered samples")
        
        # Update TUI state
        self._update_state(
            phase="train",
            step=0,
            total_steps=len(filtered_samples)
        )
        
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
        # Simple stage header (works through pipes)
        print("\n" + "=" * 70)
        print(f"RAFT CYCLE {cycle}/{self.config.num_cycles}")
        print("=" * 70)
        
        # Update TUI state
        self._update_state(cycle=cycle, phase="generate")
        
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
        
        # Generate with streaming checkpoint (handles resume automatically)
        samples = self.generate_samples(prompts, cache_path=samples_cache)
        
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
        
        # Initialize TUI state with config
        if self.state_manager:
            self.state_manager.set_config(
                model_name=cfg.base_model,
                verifier=self.verifier.__class__.__name__,
                output_dir=str(self.output_dir),
                total_cycles=num_cycles
            )
            self._update_state(status="running", total_cycles=num_cycles)
        
        # Print banner with Rich if available (startup only), then plain text
        if self.use_rich:
            ui.print_banner()
        
        # Plain text config summary (works through pipes)
        print("\n" + "=" * 70)
        print(f"RAFT Training: {num_cycles} cycles")
        print("=" * 70)
        print(f"  Reward threshold: {cfg.reward_threshold}")
        print(f"  Keep top: {cfg.keep_top_percent*100:.0f}% of passing samples")
        print(f"  Samples per prompt: {cfg.samples_per_prompt}")
        
        for cycle in range(1, num_cycles + 1):
            # Check for stop request before each cycle
            self._check_commands()
            if self._stop_requested:
                self._log("Training stopped by user", "warning")
                break
            
            stats = self.run_cycle(prompts, cycle)
            
            if stats is None:
                self._log(f"Cycle {cycle} failed. Stopping.", "error")
                self._update_state(status="error")
                break
            
            # Update cycle history in TUI
            if self.state_manager and not stats.get('skipped'):
                self.state_manager.add_cycle_stats(
                    cycle=cycle,
                    compile_rate=stats.get('success_rate', 0) * 100,
                    samples_kept=stats.get('kept', 0),
                    samples_total=stats.get('total_samples', 0),
                    loss=stats.get('avg_reward', 0),
                    elapsed_minutes=stats.get('elapsed_minutes', 0)
                )
        
        # Save statistics
        self.save_statistics()
        
        # Summary (plain text, works through pipes)
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
        
        # Update TUI state
        self._update_state(status="complete", phase="idle")
        
        final_path = self.output_dir / f"cycle_{num_cycles}_final"
        self._log(f"Final model: {final_path}", "success")
        
        return str(final_path)
    
    def save_statistics(self):
        """Save training statistics."""
        stats_path = self.output_dir / "raft_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(self.cycle_stats, f, indent=2)
        self._log(f"Saved statistics: {stats_path}", "dim")

