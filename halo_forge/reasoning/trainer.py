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
from halo_forge.capabilities import is_model_family_supported, get_supported_model_families
from halo_forge.training_updates import run_text_supervised_updates
from halo_forge.training_contracts import (
    attach_effectiveness_contract,
    build_reward_distribution_from_values,
    build_reward_filter_rejection_reasons,
    build_cycle_summary,
    build_training_summary,
    emit_yield_log_line,
    normalize_update_metrics,
    write_json_atomic,
)
from halo_forge.training_recovery import attach_recovery_guidance
from halo_forge.modality_artifacts import (
    persist_cycle_artifacts,
    persist_final_artifacts,
    resolve_resume_checkpoint,
)
from halo_forge.runtime_determinism import (
    DEFAULT_TRAINING_SEED,
    build_run_id,
    set_global_seed,
)

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
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    train_max_steps_per_cycle: int = 8
    train_max_grad_norm: Optional[float] = None
    
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
    seed: int = DEFAULT_TRAINING_SEED


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
        self._all_cycle_metrics: List[Dict[str, Any]] = []
        self.training_summary: Dict[str, Any] = {}
        self.base_model_name = config.model_name
        self.run_id: str = ""
        self.resume_checkpoint_meta: Optional[Dict[str, Any]] = None
        self.representative_examples: list[Dict[str, Any]] = []
    
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
        positive_rewards = sum(1 for c in completions if c.reward > 0)
        cycle_duration = time.time() - start_time
        yield_diagnostics = {
            "stage_counts": {
                "generated": len(completions),
                "verified": len(completions),
                "filtered": positive_rewards,
                "kept": len(filtered),
                "dropped": max(0, len(completions) - len(filtered)),
            },
            "rates": {
                "verification_rate": accuracy,
                "success_rate": accuracy,
            },
            "thresholds": {
                "configured_reward_threshold": 0.0,
                "effective_reward_threshold": 0.0,
                "keep_percent": self.config.keep_top_percent,
                "threshold_adjusted": False,
            },
            "minimums": {"minimum_samples_target": 1},
            "rejection_reasons": build_reward_filter_rejection_reasons(
                total_count=len(completions),
                success_count=sum(1 for c in completions if c.verified),
                above_threshold_count=positive_rewards,
                kept_count=len(filtered),
            ),
            "reward_distribution": build_reward_distribution_from_values(
                c.reward for c in completions
            ),
            "summary": {
                "text": (
                    "Most reasoning completions produced usable supervision."
                    if len(filtered) > 0 and accuracy >= 0.35
                    else "Reasoning yield was low; inspect verifier failures or increase sample budget."
                )
            },
        }
        if not self.representative_examples:
            failed = [
                {
                    "reason": "verification_failed",
                    "label": "Reasoning verifier failure",
                    "preview": c.completion,
                    "context": c.sample.question,
                    "reward": c.reward,
                }
                for c in completions
                if not c.verified
            ][:3]
            dropped = [
                {
                    "reason": "dropped_by_keep_percent",
                    "label": "Dropped completion",
                    "preview": c.completion,
                    "context": c.sample.question,
                    "reward": c.reward,
                }
                for c in completions
                if c.reward > 0 and c not in filtered
            ][:3]
            self.representative_examples = failed or dropped
        train_metrics = normalize_update_metrics(
            self._train_on_filtered(filtered, cycle),
            default_reason="no_filtered_samples",
        )
        metrics = build_cycle_summary(
            cycle=cycle,
            learning_rate=self.get_learning_rate(cycle),
            samples_seen=len(completions),
            samples_kept=len(filtered),
            cycle_duration_seconds=cycle_duration,
            update_metrics=train_metrics,
            yield_diagnostics=yield_diagnostics,
            extra={
                "total_completions": len(completions),
                "filtered_completions": len(filtered),
                "accuracy": accuracy,
                "avg_reward": avg_reward,
                # Backward-compatible alias.
                "duration_seconds": cycle_duration,
            },
        )
        
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
        logger.info(
            "  Update: steps=%d, weights_updated=%s, loss=%s",
            metrics["train_steps_executed"],
            metrics["weights_updated"],
            (
                f"{metrics['train_loss']:.4f}"
                if isinstance(metrics["train_loss"], (float, int))
                else "n/a"
            ),
        )
        emit_yield_log_line(
            {
                "cycle": cycle,
                **metrics["yield_diagnostics"],
            }
        )
        
        return metrics
    
    def train(
        self,
        samples: List[MathSample],
        resume_from_cycle: int = 0,
    ) -> Dict[str, Any]:
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
        if not samples:
            raise ValueError("reasoning training requires at least one sample")

        start_cycle = int(resume_from_cycle or 0)
        self.resume_checkpoint_meta = None
        self._validate_resume_configuration(start_cycle)
        if start_cycle > 0:
            checkpoint = resolve_resume_checkpoint(
                output_dir=self.output_dir,
                resume_from_cycle=start_cycle,
                max_cycles=self.config.num_cycles,
                modality="reasoning",
            )
            self.resume_checkpoint_meta = {
                "cycle": checkpoint.cycle,
                "cycle_dir": str(checkpoint.cycle_dir),
                "model_dir": str(checkpoint.model_dir),
            }
            self.config.model_name = str(checkpoint.model_dir)
            self._load_resume_history(start_cycle)

        if not (
            is_model_family_supported("reasoning", self.config.model_name)
            or is_model_family_supported("reasoning", self.base_model_name)
        ):
            families = ", ".join(get_supported_model_families("reasoning"))
            raise ValueError(
                f"Unsupported model family for reasoning training. Supported families: {families}."
            )

        normalized_seed = set_global_seed(self.config.seed)
        self.config.seed = normalized_seed
        self.run_id = build_run_id("reasoning")

        for cycle in range(start_cycle, self.config.num_cycles):
            self.current_cycle = cycle
            cycle_metrics = self.train_cycle(samples, cycle)
            self._all_cycle_metrics.append(cycle_metrics)
            write_json_atomic(
                self.output_dir / "training_history.json",
                {"cycles": self._all_cycle_metrics},
            )
        
        # Save final summary
        summary = build_training_summary(
            modality="reasoning",
            model_name=self.config.model_name,
            total_cycles_planned=self.config.num_cycles,
            cycles=self._all_cycle_metrics,
            run_id=self.run_id,
            seed=self.config.seed,
            resume_from_cycle=start_cycle,
            resumed_from_checkpoint=self.resume_checkpoint_meta,
            base_model_name=self.base_model_name,
            active_model_name=self.config.model_name,
            extra={
                "config": {
                    "model": self.config.model_name,
                    "cycles": self.config.num_cycles,
                    "samples": len(samples),
                },
                "final_accuracy": self.metrics["cycle_accuracy"][-1] if self.metrics["cycle_accuracy"] else 0,
                "final_reward": self.metrics["cycle_rewards"][-1] if self.metrics["cycle_rewards"] else 0,
            },
        )
        final_state = persist_final_artifacts(
            output_dir=self.output_dir,
            modality="reasoning",
            model_name=self.config.model_name,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        summary["final_model_path"] = final_state["final_model_dir"]
        attach_effectiveness_contract(
            summary,
            minimum_samples_kept=1,
            minimum_optimizer_steps=1,
            evaluation={
                "metric_name": "accuracy",
                "baseline_value": (
                    self.metrics["cycle_accuracy"][0] if self.metrics["cycle_accuracy"] else None
                ),
                "final_value": (
                    self.metrics["cycle_accuracy"][-1] if self.metrics["cycle_accuracy"] else None
                ),
                "higher_is_better": True,
                "tolerance": 0.0,
            },
            evaluation_required=False,
            checkpoint_written=bool(self._all_cycle_metrics),
            final_model_path=final_state["final_model_dir"],
            training_summary_path=self.output_dir / "training_summary.json",
        )
        attach_recovery_guidance(
            summary,
            modality="reasoning",
            launch_args={
                "model": self.base_model_name,
                "dataset": "",
                "output_dir": str(self.output_dir),
                "cycles": self.config.num_cycles,
                "learning_rate": self.config.learning_rate,
                "lr_decay": self.config.lr_decay_per_cycle,
                "limit": len(samples),
                "keep_percent": self.config.keep_top_percent,
            },
            representative_examples=self.representative_examples,
        )
        self.training_summary = summary

        write_json_atomic(self.output_dir / "training_summary.json", summary)
        
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
        write_json_atomic(path / "metrics.json", metrics)
        
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
        persist_cycle_artifacts(
            output_dir=self.output_dir,
            modality="reasoning",
            model_name=self.config.model_name,
            cycle=int(metrics.get("cycle", 0)),
            update_metrics=metrics,
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def _train_on_filtered(
        self,
        completions: List[ReasoningCompletion],
        cycle: int,
    ) -> Dict[str, Any]:
        """Run real supervised updates on filtered completions."""
        if not completions:
            return {
                "train_steps_executed": 0,
                "train_loss": None,
                "weights_updated": False,
                "update_reason": "no_filtered_samples",
            }
        if self.model is None or self.tokenizer is None:
            self.load_model()

        training_texts = []
        for item in completions:
            prompt = self._format_prompt(item.sample.question)
            training_texts.append(f"{prompt}\n{item.completion}")

        return run_text_supervised_updates(
            model=self.model,
            tokenizer=self.tokenizer,
            texts=training_texts,
            learning_rate=self.get_learning_rate(cycle),
            batch_size=self.config.train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_steps=self.config.train_max_steps_per_cycle,
            max_length=1024,
            max_grad_norm=self.config.train_max_grad_norm,
        )

    def _validate_resume_configuration(self, start_cycle: int) -> None:
        """Validate resume prerequisites."""
        if start_cycle < 0 or start_cycle >= self.config.num_cycles:
            raise ValueError(
                f"resume_from_cycle must be in [0, {self.config.num_cycles - 1}]"
            )
        if start_cycle == 0:
            return
        resolve_resume_checkpoint(
            output_dir=self.output_dir,
            resume_from_cycle=start_cycle,
            max_cycles=self.config.num_cycles,
            modality="reasoning",
        )
        history_path = self.output_dir / "training_history.json"
        if not history_path.exists():
            raise ValueError(
                f"resume_from_cycle={start_cycle} requires history file {history_path}"
            )

    def _load_resume_history(self, start_cycle: int) -> None:
        """Load prior cycle metrics for resume mode."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, encoding="utf-8") as f:
            payload = json.load(f)
        cycles = payload.get("cycles") if isinstance(payload, dict) else payload
        if not isinstance(cycles, list):
            raise ValueError(f"Invalid training history format in {history_path}")
        self._all_cycle_metrics = [dict(c) for c in cycles if isinstance(c, dict)]
        if len(self._all_cycle_metrics) < start_cycle:
            raise ValueError(
                f"resume_from_cycle={start_cycle} exceeds available recorded cycles "
                f"({len(self._all_cycle_metrics)}) in {history_path}"
            )
        self.metrics["cycle_accuracy"] = [
            float(c.get("accuracy", 0.0))
            for c in self._all_cycle_metrics
        ]
        self.metrics["cycle_rewards"] = [
            float(c.get("avg_reward", 0.0))
            for c in self._all_cycle_metrics
        ]
