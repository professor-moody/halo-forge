"""
Reasoning RAFT Trainer

RAFT training for mathematical and reasoning tasks.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging
import time

from halo_forge.reasoning.verifiers import MathVerifier, ReasoningVerifyResult
from halo_forge.reasoning.data import MathSample

logger = logging.getLogger(__name__)


@dataclass
class ReasoningRAFTConfig:
    """Configuration for Reasoning RAFT training."""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # Training
    num_cycles: int = 4
    samples_per_prompt: int = 4
    keep_top_percent: float = 0.5
    
    # Learning rate
    learning_rate: float = 1e-5
    lr_decay_per_cycle: float = 0.85
    
    # Generation
    temperature: float = 0.7
    max_new_tokens: int = 512
    
    # Verification
    tolerance: float = 1e-6
    partial_credit: bool = True
    
    # Output
    output_dir: str = "models/reasoning_raft"
    
    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True


@dataclass
class ReasoningCompletion:
    """A single reasoning completion for verification."""
    
    sample: MathSample
    completion: str
    reward: float = 0.0
    verified: bool = False
    result: Optional[ReasoningVerifyResult] = None


class ReasoningRAFTTrainer:
    """
    RAFT trainer for mathematical reasoning.
    
    Training loop:
    1. Generate multiple solutions per problem
    2. Verify each solution using MathVerifier
    3. Filter to keep top-scoring solutions
    4. Train on filtered solutions
    5. Repeat for multiple cycles
    """
    
    def __init__(self, config: ReasoningRAFTConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.verifier = MathVerifier(
            tolerance=config.tolerance,
            partial_credit_for_work=config.partial_credit,
        )
        
        self.model = None
        self.tokenizer = None
        self.current_cycle = 0
        
        # Metrics
        self.metrics: Dict[str, List[float]] = {
            "cycle_rewards": [],
            "cycle_accuracy": [],
            "cycle_samples": [],
        }
    
    def load_model(self) -> None:
        """Load model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers required: pip install transformers")
        
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            trust_remote_code=True,
            device_map="auto",
        )
        
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Model loaded on {self.model.device}")
    
    def get_learning_rate(self, cycle: int) -> float:
        """Get learning rate for current cycle with decay."""
        return self.config.learning_rate * (self.config.lr_decay_per_cycle ** cycle)
    
    def generate_completions(
        self,
        samples: List[MathSample]
    ) -> List[ReasoningCompletion]:
        """
        Generate multiple completions per sample.
        
        Args:
            samples: Math problems
            
        Returns:
            List of completions
        """
        import torch
        
        if self.model is None:
            self.load_model()
        
        completions = []
        
        for sample in samples:
            prompt = self._format_prompt(sample.question)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.model.device)
            
            for _ in range(self.config.samples_per_prompt):
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                # Decode only the generated part
                generated = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                
                completions.append(ReasoningCompletion(
                    sample=sample,
                    completion=generated,
                ))
        
        return completions
    
    def verify_completions(
        self,
        completions: List[ReasoningCompletion]
    ) -> List[ReasoningCompletion]:
        """
        Verify all completions.
        
        Args:
            completions: List of completions to verify
            
        Returns:
            Completions with verification results
        """
        for comp in completions:
            result = self.verifier.verify(
                prompt=comp.sample.question,
                completion=comp.completion,
                expected_answer=comp.sample.answer,
            )
            comp.result = result
            comp.reward = result.reward
            comp.verified = result.success
        
        return completions
    
    def filter_completions(
        self,
        completions: List[ReasoningCompletion]
    ) -> List[ReasoningCompletion]:
        """
        Filter to keep top-scoring completions.
        
        Args:
            completions: Verified completions
            
        Returns:
            Filtered completions
        """
        # Sort by reward descending
        sorted_comps = sorted(completions, key=lambda c: c.reward, reverse=True)
        
        # Keep top percent
        keep_count = max(1, int(len(sorted_comps) * self.config.keep_top_percent))
        filtered = sorted_comps[:keep_count]
        
        # Only keep those with positive reward
        filtered = [c for c in filtered if c.reward > 0]
        
        logger.info(
            f"Filtered: {len(filtered)}/{len(completions)} "
            f"(avg reward: {sum(c.reward for c in filtered)/len(filtered) if filtered else 0:.3f})"
        )
        
        return filtered
    
    def train_cycle(
        self,
        samples: List[MathSample],
        cycle: int
    ) -> Dict[str, Any]:
        """
        Run one RAFT cycle.
        
        Args:
            samples: Training samples
            cycle: Cycle number
            
        Returns:
            Cycle metrics
        """
        logger.info(f"=" * 60)
        logger.info(f"REASONING RAFT CYCLE {cycle + 1}/{self.config.num_cycles}")
        logger.info(f"=" * 60)
        
        start_time = time.time()
        
        # 1. Generate completions
        logger.info(f"Generating {len(samples) * self.config.samples_per_prompt} completions...")
        completions = self.generate_completions(samples)
        
        # 2. Verify completions
        logger.info("Verifying completions...")
        completions = self.verify_completions(completions)
        
        # 3. Filter completions
        filtered = self.filter_completions(completions)
        
        # 4. Calculate metrics
        accuracy = sum(1 for c in completions if c.verified) / len(completions)
        avg_reward = sum(c.reward for c in completions) / len(completions)
        
        metrics = {
            "cycle": cycle,
            "total_completions": len(completions),
            "filtered_completions": len(filtered),
            "accuracy": accuracy,
            "avg_reward": avg_reward,
            "learning_rate": self.get_learning_rate(cycle),
            "duration_seconds": time.time() - start_time,
        }
        
        # Update tracking
        self.metrics["cycle_rewards"].append(avg_reward)
        self.metrics["cycle_accuracy"].append(accuracy)
        self.metrics["cycle_samples"].append(len(filtered))
        
        # 5. Save cycle checkpoint
        checkpoint_path = self.output_dir / f"cycle_{cycle}"
        self._save_cycle_results(checkpoint_path, filtered, metrics)
        
        logger.info(f"Cycle {cycle + 1} complete:")
        logger.info(f"  Accuracy: {accuracy:.1%}")
        logger.info(f"  Avg Reward: {avg_reward:.3f}")
        logger.info(f"  Filtered Samples: {len(filtered)}")
        
        return metrics
    
    def train(self, samples: List[MathSample]) -> Dict[str, Any]:
        """
        Run full RAFT training.
        
        Args:
            samples: Training samples
            
        Returns:
            Training summary
        """
        logger.info(f"Starting Reasoning RAFT training")
        logger.info(f"  Samples: {len(samples)}")
        logger.info(f"  Cycles: {self.config.num_cycles}")
        logger.info(f"  Model: {self.config.model_name}")
        
        all_metrics = []
        
        for cycle in range(self.config.num_cycles):
            self.current_cycle = cycle
            cycle_metrics = self.train_cycle(samples, cycle)
            all_metrics.append(cycle_metrics)
        
        # Save final summary
        summary = {
            "config": {
                "model": self.config.model_name,
                "cycles": self.config.num_cycles,
                "samples": len(samples),
            },
            "cycles": all_metrics,
            "final_accuracy": self.metrics["cycle_accuracy"][-1] if self.metrics["cycle_accuracy"] else 0,
            "final_reward": self.metrics["cycle_rewards"][-1] if self.metrics["cycle_rewards"] else 0,
        }
        
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training complete! Results saved to {self.output_dir}")
        
        return summary
    
    def _format_prompt(self, question: str) -> str:
        """Format math problem as prompt."""
        return (
            f"Solve the following math problem step by step. "
            f"Put your final answer in \\boxed{{}}.\n\n"
            f"Problem: {question}\n\n"
            f"Solution:"
        )
    
    def _save_cycle_results(
        self,
        path: Path,
        completions: List[ReasoningCompletion],
        metrics: Dict[str, Any]
    ) -> None:
        """Save cycle results to disk."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save completions
        comp_data = []
        for c in completions:
            comp_data.append({
                "question": c.sample.question,
                "expected_answer": c.sample.answer,
                "completion": c.completion,
                "reward": c.reward,
                "verified": c.verified,
                "extracted_answer": c.result.extracted_answer if c.result else None,
            })
        
        with open(path / "completions.jsonl", "w") as f:
            for item in comp_data:
                f.write(json.dumps(item) + "\n")
