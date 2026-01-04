"""
GRPO (Generalized Reward-ranked Preference Optimization) Trainer

An alternative to RAFT that uses pairwise preferences based on verification
rewards rather than direct fine-tuning on best samples.

Key differences from RAFT:
- Creates preference pairs (chosen/rejected) from samples
- Uses DPO-style training loss
- Better handles noisy rewards
- More stable but potentially slower convergence

Paper reference: inspired by DPO and RLHF approaches
"""

import gc
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel

from halo_forge.rlvr.verifiers.base import Verifier


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    
    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-3B"
    sft_checkpoint: Optional[str] = None
    output_dir: str = "models/grpo"
    
    # GRPO parameters
    num_cycles: int = 3
    samples_per_prompt: int = 8  # Generate multiple to form pairs
    beta: float = 0.1  # KL divergence weight (higher = more conservative)
    
    # Pair selection
    reward_margin: float = 0.3  # Minimum reward gap between chosen/rejected
    pairs_per_prompt: int = 2   # Number of preference pairs per prompt
    
    # Generation
    max_new_tokens: int = 1024
    temperature: float = 0.8  # Slightly higher for diversity
    generation_batch_size: int = 4
    
    # Training
    train_epochs: int = 1
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6  # Lower for preference learning
    
    # System prompt
    system_prompt: str = "You are an expert programmer."


@dataclass
class PreferencePair:
    """A preference pair for GRPO training."""
    prompt: str
    chosen: str       # Higher reward sample
    rejected: str     # Lower reward sample
    chosen_reward: float
    rejected_reward: float


class GRPOTrainer:
    """
    GRPO (Generalized Reward-ranked Preference Optimization) trainer.
    
    Generates multiple samples per prompt, verifies them, then creates
    preference pairs for DPO-style training.
    
    Example:
        trainer = GRPOTrainer(config, verifier)
        trainer.run(prompts)
    """
    
    def __init__(
        self,
        config: GRPOConfig,
        verifier: Verifier,
        device: str = "auto"
    ):
        self.config = config
        self.verifier = verifier
        self.device = device
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.ref_model = None  # Reference model for KL penalty
        
    def _log(self, message: str, level: str = "info"):
        """Log a message with formatting."""
        prefix = {
            "info": "[GRPO]",
            "dim": "[GRPO]",
            "success": "[GRPO ✓]",
            "warning": "[GRPO ⚠]",
            "error": "[GRPO ✗]",
        }.get(level, "[GRPO]")
        print(f"{prefix} {message}")
    
    def _load_model(self):
        """Load the model and tokenizer."""
        cfg = self.config
        
        self._log(f"Loading model: {cfg.base_model}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": self.device if self.device != "auto" else "auto",
            "trust_remote_code": True,
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            **model_kwargs
        )
        
        # Load SFT checkpoint if provided
        if cfg.sft_checkpoint and Path(cfg.sft_checkpoint).exists():
            self._log(f"Loading SFT checkpoint: {cfg.sft_checkpoint}")
            self.model = PeftModel.from_pretrained(self.model, cfg.sft_checkpoint)
            self.model = self.model.merge_and_unload()
        
        # Create reference model (frozen copy for KL penalty)
        self._log("Creating reference model...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            **model_kwargs
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Add LoRA to training model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.model.gradient_checkpointing_enable()
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt with chat template."""
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        
        # Fallback
        return f"{self.config.system_prompt}\n\n{prompt}\n"
    
    def _generate_samples(self, prompts: List[str]) -> Dict[str, List[Dict]]:
        """Generate multiple samples per prompt."""
        cfg = self.config
        samples_by_prompt = {p: [] for p in prompts}
        
        self.model.eval()
        
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="Generating samples"):
                formatted = self._format_prompt(prompt)
                inputs = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)
                
                for _ in range(cfg.samples_per_prompt):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature,
                        do_sample=True,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    samples_by_prompt[prompt].append({
                        "response": response,
                        "reward": None
                    })
        
        return samples_by_prompt
    
    def _verify_samples(self, samples_by_prompt: Dict[str, List[Dict]]):
        """Verify all samples and assign rewards."""
        all_samples = []
        indices = []
        
        for prompt, samples in samples_by_prompt.items():
            for i, sample in enumerate(samples):
                all_samples.append(sample["response"])
                indices.append((prompt, i))
        
        self._log(f"Verifying {len(all_samples)} samples...")
        
        # Batch verification
        results = self.verifier.verify_batch(all_samples)
        
        # Assign rewards back
        for (prompt, i), result in zip(indices, results):
            samples_by_prompt[prompt][i]["reward"] = result.reward
            samples_by_prompt[prompt][i]["details"] = result.details
    
    def _create_preference_pairs(
        self,
        samples_by_prompt: Dict[str, List[Dict]]
    ) -> List[PreferencePair]:
        """Create preference pairs from verified samples."""
        cfg = self.config
        pairs = []
        
        for prompt, samples in samples_by_prompt.items():
            # Sort by reward (descending)
            sorted_samples = sorted(
                samples,
                key=lambda x: x.get("reward", 0) or 0,
                reverse=True
            )
            
            # Create pairs with sufficient margin
            used_rejected = set()
            pair_count = 0
            
            for i, chosen in enumerate(sorted_samples):
                if pair_count >= cfg.pairs_per_prompt:
                    break
                
                chosen_reward = chosen.get("reward", 0) or 0
                
                # Find a rejected sample with sufficient margin
                for j in range(len(sorted_samples) - 1, i, -1):
                    if j in used_rejected:
                        continue
                    
                    rejected = sorted_samples[j]
                    rejected_reward = rejected.get("reward", 0) or 0
                    
                    if chosen_reward - rejected_reward >= cfg.reward_margin:
                        pairs.append(PreferencePair(
                            prompt=prompt,
                            chosen=chosen["response"],
                            rejected=rejected["response"],
                            chosen_reward=chosen_reward,
                            rejected_reward=rejected_reward
                        ))
                        used_rejected.add(j)
                        pair_count += 1
                        break
        
        return pairs
    
    def _compute_log_probs(
        self,
        model: torch.nn.Module,
        prompt: str,
        response: str
    ) -> torch.Tensor:
        """Compute log probabilities for a response."""
        formatted = self._format_prompt(prompt)
        full_text = formatted + response
        
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(model.device)
        
        prompt_inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        prompt_len = prompt_inputs['input_ids'].shape[1]
        
        with torch.set_grad_enabled(model.training):
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get log probs for response tokens only
        shift_logits = logits[:, prompt_len-1:-1, :]
        shift_labels = inputs['input_ids'][:, prompt_len:]
        
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs.sum()
    
    def _dpo_loss(
        self,
        pair: PreferencePair
    ) -> torch.Tensor:
        """Compute DPO loss for a preference pair."""
        beta = self.config.beta
        
        # Policy model log probs
        policy_chosen = self._compute_log_probs(self.model, pair.prompt, pair.chosen)
        policy_rejected = self._compute_log_probs(self.model, pair.prompt, pair.rejected)
        
        # Reference model log probs
        with torch.no_grad():
            ref_chosen = self._compute_log_probs(self.ref_model, pair.prompt, pair.chosen)
            ref_rejected = self._compute_log_probs(self.ref_model, pair.prompt, pair.rejected)
        
        # DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
        log_ratio_chosen = policy_chosen - ref_chosen
        log_ratio_rejected = policy_rejected - ref_rejected
        
        loss = -torch.nn.functional.logsigmoid(
            beta * (log_ratio_chosen - log_ratio_rejected)
        )
        
        return loss
    
    def _train_on_pairs(self, pairs: List[PreferencePair]):
        """Train model on preference pairs using DPO loss."""
        cfg = self.config
        
        self._log(f"Training on {len(pairs)} preference pairs...")
        
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate
        )
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        total_loss = 0.0
        optimizer.zero_grad()
        
        for i, pair in enumerate(tqdm(pairs, desc="Training")):
            loss = self._dpo_loss(pair)
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()
            
            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        # Handle remaining gradients
        if len(pairs) % cfg.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(pairs)
        self._log(f"Average DPO loss: {avg_loss:.4f}", "success")
        
        return {"avg_loss": avg_loss}
    
    def run_cycle(self, prompts: List[str], cycle: int) -> Dict:
        """Run a single GRPO cycle."""
        self._log(f"=== Cycle {cycle} ===")
        
        # 1. Generate samples
        samples_by_prompt = self._generate_samples(prompts)
        
        # Clean up after generation
        gc.collect()
        torch.cuda.empty_cache()
        
        # 2. Verify samples
        self._verify_samples(samples_by_prompt)
        
        # Clean up after verification
        gc.collect()
        torch.cuda.empty_cache()
        
        # 3. Create preference pairs
        pairs = self._create_preference_pairs(samples_by_prompt)
        self._log(f"Created {len(pairs)} preference pairs")
        
        if not pairs:
            self._log("No valid preference pairs created", "warning")
            return {"pairs": 0}
        
        # 4. Train on pairs
        train_stats = self._train_on_pairs(pairs)
        
        # 5. Save checkpoint
        output_path = Path(self.config.output_dir) / f"cycle_{cycle}"
        output_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        return {
            "pairs": len(pairs),
            **train_stats
        }
    
    def run(self, prompts: List[str], num_cycles: Optional[int] = None):
        """Run full GRPO training."""
        num_cycles = num_cycles or self.config.num_cycles
        
        # Load model
        self._load_model()
        
        print("\n" + "=" * 70)
        print(f"GRPO Training: {num_cycles} cycles")
        print("=" * 70)
        print(f"  Beta (KL weight): {self.config.beta}")
        print(f"  Reward margin: {self.config.reward_margin}")
        print(f"  Pairs per prompt: {self.config.pairs_per_prompt}")
        print(f"  Samples per prompt: {self.config.samples_per_prompt}")
        
        for cycle in range(1, num_cycles + 1):
            stats = self.run_cycle(prompts, cycle)
            
            if stats.get("pairs", 0) == 0:
                self._log("No pairs created. Consider lowering reward_margin.", "error")
        
        # Save final model
        final_path = Path(self.config.output_dir) / "final_model"
        final_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        self._log(f"Final model saved to {final_path}", "success")

