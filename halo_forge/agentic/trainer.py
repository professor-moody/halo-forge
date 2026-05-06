"""
Agentic RAFT Trainer

RAFT (Reward-Ranked Fine-Tuning) for tool calling / function calling models.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging
import time

import torch
from tqdm import tqdm

from halo_forge.agentic.verifiers import ToolCallingVerifier, ToolCallVerifyResult
from halo_forge.agentic.data import ToolCallSample, XLAMLoader
from halo_forge.agentic.data.formatters import HermesFormatter, create_training_sample
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
from halo_forge.utils.metrics import MetricsTracker
from halo_forge.utils.accelerator import (
    get_device_map,
    get_torch_device,
    recommended_dtype,
)

logger = logging.getLogger(__name__)


@dataclass
class AgenticRAFTConfig:
    """Configuration for Agentic RAFT training."""
    
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    trust_remote_code: bool = True
    
    # Training
    num_cycles: int = 5
    samples_per_prompt: int = 8
    reward_threshold: float = 0.5
    keep_top_percent: float = 0.25  # More selective for tool calling precision
    
    # Learning rate
    learning_rate: float = 5e-5  # Lower than SFT
    lr_decay_per_cycle: float = 0.85
    min_lr: float = 1e-6
    
    # Generation
    temperature: float = 0.7
    max_new_tokens: int = 512  # Tool calls are short
    top_p: float = 0.95
    
    # Batch
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_grad_norm: Optional[float] = None
    
    # Output
    output_dir: str = "models/agentic_raft"
    save_every_cycle: bool = True
    
    # Device
    device: Optional[str] = None
    
    # AMD Strix Halo requirements
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = DEFAULT_TRAINING_SEED
    
    def __post_init__(self):
        if self.device is None:
            self.device = str(get_torch_device())


@dataclass
class AgenticCompletion:
    """A single tool calling completion for verification."""
    
    prompt: str
    output: str
    reward: float = 0.0
    verified: bool = False
    result: Optional[ToolCallVerifyResult] = None


@dataclass
class AgenticRAFTCycleResult:
    """Result of a single RAFT cycle."""
    
    cycle: int
    total_samples: int
    verified_samples: int
    avg_reward: float
    success_rate: float
    training_samples: int
    metrics: Dict[str, Any] = field(default_factory=dict)


class AgenticRAFTTrainer:
    """
    RAFT Trainer for Tool Calling / Agentic Models.
    
    Training loop:
    1. Generate tool call outputs for each prompt
    2. Verify using ToolCallingVerifier (JSON, schema, function matching)
    3. Filter by reward threshold and keep top K%
    4. Fine-tune on high-reward samples
    5. Repeat with decayed learning rate
    
    Example:
        config = AgenticRAFTConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            num_cycles=5,
        )
        trainer = AgenticRAFTTrainer(config)
        trainer.load_model()
        
        samples = XLAMLoader().load(limit=1000)
        trainer.train(samples)
    """
    
    def __init__(self, config: AgenticRAFTConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration.
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.verifier = ToolCallingVerifier()
        self.formatter = HermesFormatter()
        
        self.model = None
        self.tokenizer = None
        self.current_cycle = 0
        
        # Track metrics with TensorBoard integration
        self.cycle_results: List[AgenticRAFTCycleResult] = []
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
        self.representative_examples: List[Dict[str, Any]] = []
        
        # Initialize MetricsTracker for TensorBoard and JSON logging
        self.metrics_tracker = MetricsTracker(
            output_dir=str(self.output_dir),
            model_name=config.model_name,
            config={
                "num_cycles": config.num_cycles,
                "samples_per_prompt": config.samples_per_prompt,
                "reward_threshold": config.reward_threshold,
                "keep_top_percent": config.keep_top_percent,
                "learning_rate": config.learning_rate,
                "domain": "agentic",
            },
            enable_tensorboard=True,
            enable_json_logs=True,
        )
    
    def load_model(self) -> None:
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            dtype=recommended_dtype() if self.config.bf16 else torch.float16,
            device_map=get_device_map(),
        )
        
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info("Model loaded successfully")
    
    def get_learning_rate(self, cycle: int) -> float:
        """Get learning rate for current cycle (with decay)."""
        lr = self.config.learning_rate * (self.config.lr_decay_per_cycle ** cycle)
        return max(lr, self.config.min_lr)
    
    def train(
        self,
        samples: List[ToolCallSample],
        resume_from_cycle: int = 0,
    ) -> Dict[str, Any]:
        """
        Run RAFT training loop.
        
        Args:
            samples: Training samples.
            resume_from_cycle: Cycle to resume from.
            
        Returns:
            Training metrics dict.
        """
        self._validate_resume_from_cycle(resume_from_cycle)
        if not samples:
            raise ValueError("agentic training requires at least one sample")
        self.resume_checkpoint_meta = None
        if resume_from_cycle > 0:
            checkpoint = resolve_resume_checkpoint(
                output_dir=self.output_dir,
                resume_from_cycle=resume_from_cycle,
                max_cycles=self.config.num_cycles,
                modality="agentic",
            )
            self.resume_checkpoint_meta = {
                "cycle": checkpoint.cycle,
                "cycle_dir": str(checkpoint.cycle_dir),
                "model_dir": str(checkpoint.model_dir),
            }
            self.config.model_name = str(checkpoint.model_dir)
            self._load_resume_history(resume_from_cycle)

        if not (
            is_model_family_supported("agentic", self.config.model_name)
            or is_model_family_supported("agentic", self.base_model_name)
        ):
            families = ", ".join(get_supported_model_families("agentic"))
            raise ValueError(
                f"Unsupported model family for agentic training. Supported families: {families}."
            )

        normalized_seed = set_global_seed(self.config.seed)
        self.config.seed = normalized_seed
        self.run_id = build_run_id("agentic")

        if self.model is None:
            self.load_model()

        logger.info(f"Starting RAFT training with {len(samples)} samples")
        logger.info(f"Cycles: {self.config.num_cycles}, Samples per prompt: {self.config.samples_per_prompt}")
        
        start_time = time.time()
        
        for cycle in range(resume_from_cycle, self.config.num_cycles):
            self.current_cycle = cycle
            
            # Start cycle tracking
            self.metrics_tracker.start_cycle(cycle)
            
            cycle_result = self._run_cycle(samples, cycle)
            self.cycle_results.append(cycle_result)
            self._all_cycle_metrics.append(cycle_result.metrics)
            write_json_atomic(
                self.output_dir / "training_history.json",
                {"cycles": self._all_cycle_metrics},
            )
            
            # Log cycle to MetricsTracker (TensorBoard + JSON)
            self.metrics_tracker.log_cycle(cycle, {
                "success_rate": cycle_result.success_rate,
                "avg_reward": cycle_result.avg_reward,
                "kept_samples": cycle_result.training_samples,
                "total_samples": cycle_result.total_samples,
                "learning_rate": self.get_learning_rate(cycle),
                "train_steps_executed": cycle_result.metrics.get("train_steps_executed", 0),
                "weights_updated": float(cycle_result.metrics.get("weights_updated", False)),
                "train_loss": cycle_result.metrics.get("train_loss", 0.0) or 0.0,
            })
            
            # Log cycle results
            logger.info(
                f"Cycle {cycle + 1}/{self.config.num_cycles}: "
                f"avg_reward={cycle_result.avg_reward:.3f}, "
                f"success_rate={cycle_result.success_rate:.2%}, "
                f"training_samples={cycle_result.training_samples}"
            )
            
            # Save checkpoint
            if self.config.save_every_cycle:
                self._save_checkpoint(cycle, cycle_result.metrics)
        
        total_time = time.time() - start_time
        
        # Final metrics
        final_metrics = build_training_summary(
            modality="agentic",
            model_name=self.config.model_name,
            total_cycles_planned=self.config.num_cycles,
            cycles=self._all_cycle_metrics,
            run_id=self.run_id,
            seed=self.config.seed,
            resume_from_cycle=resume_from_cycle,
            resumed_from_checkpoint=self.resume_checkpoint_meta,
            base_model_name=self.base_model_name,
            active_model_name=self.config.model_name,
            extra={
                "total_cycles": self.config.num_cycles,
                "total_time_seconds": total_time,
                "final_avg_reward": self.cycle_results[-1].avg_reward if self.cycle_results else 0.0,
                "final_success_rate": self.cycle_results[-1].success_rate if self.cycle_results else 0.0,
                # Backward-compatible alias retained for older consumers.
                "cycle_results": [vars(r) for r in self.cycle_results],
            },
        )
        final_state = persist_final_artifacts(
            output_dir=self.output_dir,
            modality="agentic",
            model_name=self.config.model_name,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        final_metrics["final_model_path"] = final_state["final_model_dir"]
        attach_effectiveness_contract(
            final_metrics,
            minimum_samples_kept=1,
            minimum_optimizer_steps=1,
            evaluation={
                "metric_name": "success_rate",
                "baseline_value": (
                    self.cycle_results[0].success_rate if self.cycle_results else None
                ),
                "final_value": (
                    self.cycle_results[-1].success_rate if self.cycle_results else None
                ),
                "higher_is_better": True,
                "tolerance": 0.0,
            },
            evaluation_required=False,
            checkpoint_written=bool(self.cycle_results),
            final_model_path=final_state["final_model_dir"],
            training_summary_path=self.output_dir / "training_summary.json",
        )
        attach_recovery_guidance(
            final_metrics,
            modality="agentic",
            launch_args={
                "model": self.base_model_name,
                "dataset": "",
                "output_dir": str(self.output_dir),
                "cycles": self.config.num_cycles,
                "learning_rate": self.config.learning_rate,
                "lr_decay": self.config.lr_decay_per_cycle,
                "limit": len(samples),
                "samples_per_prompt": self.config.samples_per_prompt,
                "keep_percent": self.config.keep_top_percent,
            },
            representative_examples=self.representative_examples,
        )
        self.training_summary = final_metrics
        
        # Save metrics summary
        self.metrics_tracker.save_summary()
        
        # Also save standalone metrics file
        metrics_path = self.output_dir / "training_metrics.json"
        write_json_atomic(metrics_path, final_metrics)
        write_json_atomic(self.output_dir / "training_summary.json", final_metrics)
        
        logger.info(f"Training complete. Metrics saved to {metrics_path}")
        logger.info(f"TensorBoard logs: {self.output_dir / 'tensorboard'}")
        
        return final_metrics
    
    def _run_cycle(
        self,
        samples: List[ToolCallSample],
        cycle: int,
    ) -> AgenticRAFTCycleResult:
        """Run a single RAFT cycle.
        
        Args:
            samples: Training samples.
            cycle: Current cycle number.
            
        Returns:
            Cycle results.
        """
        logger.info(f"Starting cycle {cycle + 1}")
        cycle_start = time.time()
        
        # Generate and verify completions
        all_completions: List[AgenticCompletion] = []
        
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                logger.info(f"Processing sample {i + 1}/{len(samples)}")
            
            completions = self._generate_completions(sample)
            
            # Verify each completion
            for completion in completions:
                result = self.verifier.verify(
                    output=completion.output,
                    expected_calls=sample.expected_calls,
                    is_irrelevant=sample.is_irrelevant,
                )
                completion.verified = True
                completion.reward = result.reward
                completion.result = result
            
            all_completions.extend(completions)
        
        # Filter completions
        filtered = self._filter_completions(all_completions)
        
        # Calculate metrics
        total_samples = len(all_completions)
        successful = sum(1 for c in all_completions if c.result and c.result.success)
        avg_reward = sum(c.reward for c in all_completions) / max(total_samples, 1)
        success_rate = successful / max(total_samples, 1)
        above_threshold_count = sum(
            1 for c in all_completions if float(c.reward) >= self.config.reward_threshold
        )
        above_threshold_count = sum(
            1 for c in all_completions if float(c.reward) >= self.config.reward_threshold
        )
        
        # Log sample-level rewards to MetricsTracker
        rewards = [c.reward for c in all_completions]
        self.metrics_tracker.log_samples(cycle, rewards)
        
        # Update metrics
        self.metrics["cycle_rewards"].append(avg_reward)
        self.metrics["cycle_accuracy"].append(success_rate)
        self.metrics["cycle_samples"].append(len(filtered))
        
        train_metrics: Dict[str, Any]
        if filtered:
            train_metrics = self._train_on_samples(filtered, cycle)
        else:
            train_metrics = {
                "train_steps_executed": 0,
                "train_loss": None,
                "weights_updated": False,
                "update_reason": "no_filtered_samples",
            }
        if not self.representative_examples:
            failed = [
                {
                    "reason": "verification_failed",
                    "label": "Tool-call verification failure",
                    "preview": c.output,
                    "context": c.prompt,
                    "reward": c.reward,
                }
                for c in all_completions
                if not (c.result and c.result.success)
            ][:3]
            dropped = [
                {
                    "reason": "below_reward_threshold",
                    "label": "Below threshold",
                    "preview": c.output,
                    "context": c.prompt,
                    "reward": c.reward,
                }
                for c in all_completions
                if c.result and c.result.success and float(c.reward) < self.config.reward_threshold
            ][:3]
            keep_drop = [
                {
                    "reason": "dropped_by_keep_percent",
                    "label": "Dropped by keep percent",
                    "preview": c.output,
                    "context": c.prompt,
                    "reward": c.reward,
                }
                for c in all_completions
                if c.result and c.result.success and float(c.reward) >= self.config.reward_threshold and c not in filtered
            ][:3]
            self.representative_examples = failed or dropped or keep_drop
        
        canonical_metrics = build_cycle_summary(
            cycle=cycle,
            learning_rate=self.get_learning_rate(cycle),
            samples_seen=total_samples,
            samples_kept=len(filtered),
            cycle_duration_seconds=time.time() - cycle_start,
            update_metrics=normalize_update_metrics(
                train_metrics,
                default_reason="no_filtered_samples",
            ),
            yield_diagnostics={
                "stage_counts": {
                    "generated": total_samples,
                    "verified": total_samples,
                    "filtered": above_threshold_count,
                    "kept": len(filtered),
                    "dropped": max(0, total_samples - len(filtered)),
                },
                "rates": {
                    "verification_rate": success_rate,
                    "success_rate": success_rate,
                },
                "thresholds": {
                    "configured_reward_threshold": self.config.reward_threshold,
                    "effective_reward_threshold": self.config.reward_threshold,
                    "keep_percent": self.config.keep_top_percent,
                    "threshold_adjusted": False,
                },
                "minimums": {"minimum_samples_target": 1},
                "rejection_reasons": build_reward_filter_rejection_reasons(
                    total_count=total_samples,
                    success_count=successful,
                    above_threshold_count=above_threshold_count,
                    kept_count=len(filtered),
                ),
                "reward_distribution": build_reward_distribution_from_values(rewards),
                "summary": {
                    "text": (
                        "Agentic verifier yield looks healthy."
                        if len(filtered) > 0 and success_rate >= 0.3
                        else "Agentic yield is low; inspect tool-call formatting or relax the threshold."
                    )
                },
            },
            extra={
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "training_samples": len(filtered),
            },
        )
        emit_yield_log_line(
            {
                "cycle": cycle,
                **canonical_metrics["yield_diagnostics"],
            }
        )

        return AgenticRAFTCycleResult(
            cycle=cycle,
            total_samples=total_samples,
            verified_samples=len(all_completions),
            avg_reward=avg_reward,
            success_rate=success_rate,
            training_samples=len(filtered),
            metrics=canonical_metrics,
        )

    def _validate_resume_from_cycle(self, resume_from_cycle: int) -> None:
        """Fail fast on invalid resume configuration."""
        if resume_from_cycle < 0 or resume_from_cycle >= self.config.num_cycles:
            raise ValueError(
                f"resume_from_cycle must be in [0, {self.config.num_cycles - 1}]"
            )
        if resume_from_cycle == 0:
            return
        resolve_resume_checkpoint(
            output_dir=self.output_dir,
            resume_from_cycle=resume_from_cycle,
            max_cycles=self.config.num_cycles,
            modality="agentic",
        )
        history_path = self.output_dir / "training_history.json"
        if not history_path.exists():
            raise ValueError(
                f"resume_from_cycle={resume_from_cycle} requires history file {history_path}"
            )
    
    def _generate_completions(
        self,
        sample: ToolCallSample,
    ) -> List[AgenticCompletion]:
        """Generate completions for a sample.
        
        Args:
            sample: Tool calling sample.
            
        Returns:
            List of completions.
        """
        prompt = self.formatter.format_prompt(sample)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096 - self.config.max_new_tokens,
        ).to(self.model.device)
        
        completions = []
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                num_return_sequences=self.config.samples_per_prompt,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        for output in outputs:
            generated_text = self.tokenizer.decode(
                output[inputs["input_ids"].shape[1]:],
                skip_special_tokens=False,
            )
            completions.append(AgenticCompletion(
                prompt=prompt,
                output=generated_text,
            ))
        
        return completions
    
    def _filter_completions(
        self,
        completions: List[AgenticCompletion],
    ) -> List[AgenticCompletion]:
        """Filter completions by reward threshold and top K%.
        
        Args:
            completions: All completions.
            
        Returns:
            Filtered completions for training.
        """
        # Filter by threshold
        above_threshold = [
            c for c in completions
            if c.reward >= self.config.reward_threshold
        ]
        
        # Sort by reward
        above_threshold.sort(key=lambda c: c.reward, reverse=True)
        
        # Keep top K%
        n_keep = max(1, int(len(above_threshold) * self.config.keep_top_percent))
        
        return above_threshold[:n_keep]
    
    def _train_on_samples(
        self,
        completions: List[AgenticCompletion],
        cycle: int,
    ) -> Dict[str, Any]:
        """Train on filtered completions.
        
        Args:
            completions: Filtered high-reward completions.
            cycle: Current cycle number.
        """
        logger.info(f"Training on {len(completions)} samples")
        
        # Prepare training data
        training_texts = []
        for completion in completions:
            # Combine prompt and output as full training text
            full_text = completion.prompt + completion.output
            training_texts.append(full_text)
        
        # Get current learning rate
        lr = self.get_learning_rate(cycle)
        logger.info(f"Learning rate: {lr}")
        
        train_metrics = run_text_supervised_updates(
            model=self.model,
            tokenizer=self.tokenizer,
            texts=training_texts,
            learning_rate=lr,
            batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_steps=8,
            max_length=4096,
            max_grad_norm=self.config.max_grad_norm,
        )

        logger.info(
            "Training step complete: steps=%d, weights_updated=%s, loss=%s",
            train_metrics["train_steps_executed"],
            train_metrics["weights_updated"],
            (
                f"{train_metrics['train_loss']:.4f}"
                if isinstance(train_metrics["train_loss"], (float, int))
                else "n/a"
            ),
        )
        return train_metrics
    
    def _save_checkpoint(self, cycle: int, cycle_metrics: Dict[str, Any]) -> None:
        """Save checkpoint for current cycle.
        
        Args:
            cycle: Current cycle number.
        """
        checkpoint_dir = self.output_dir / f"cycle_{cycle}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save config
        config_path = checkpoint_dir / "config.json"
        write_json_atomic(config_path, vars(self.config))
        
        # Save metrics so far
        metrics_path = checkpoint_dir / "metrics.json"
        write_json_atomic(metrics_path, self.metrics)
        persist_cycle_artifacts(
            output_dir=self.output_dir,
            modality="agentic",
            model_name=self.config.model_name,
            cycle=cycle,
            update_metrics=cycle_metrics,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def _load_resume_history(self, start_cycle: int) -> None:
        """Load historical cycle metrics when resuming."""
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
        self.metrics["cycle_rewards"] = [
            float(c.get("avg_reward", 0.0))
            for c in self._all_cycle_metrics
        ]
        self.metrics["cycle_accuracy"] = [
            float(c.get("success_rate", 0.0))
            for c in self._all_cycle_metrics
        ]
    
    def benchmark(
        self,
        samples: List[ToolCallSample],
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run benchmark evaluation.
        
        Args:
            samples: Test samples.
            limit: Optional limit on samples.
            
        Returns:
            Benchmark results.
        """
        if self.model is None:
            self.load_model()
        
        if limit:
            samples = samples[:limit]
        
        logger.info(f"Running benchmark on {len(samples)} samples")
        
        results = {
            "total": len(samples),
            "correct": 0,
            "json_valid": 0,
            "function_correct": 0,
            "args_correct": 0,
            "false_positives": 0,
            "avg_reward": 0.0,
        }
        
        total_reward = 0.0
        
        for sample in tqdm(samples, desc="Evaluating"):
            # Generate single completion (temperature=0 for benchmark)
            prompt = self.formatter.format_prompt(sample)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096 - self.config.max_new_tokens,
            ).to(self.model.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=0.0,  # Deterministic for benchmark
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=False,
            )
            
            # Verify
            result = self.verifier.verify(
                output=generated_text,
                expected_calls=sample.expected_calls,
                is_irrelevant=sample.is_irrelevant,
            )
            
            total_reward += result.reward
            
            if result.success:
                results["correct"] += 1
            if result.json_valid:
                results["json_valid"] += 1
            if result.called_correct_function:
                results["function_correct"] += 1
            if result.arguments_correct:
                results["args_correct"] += 1
            if sample.is_irrelevant and result.parsed_calls:
                results["false_positives"] += 1
        
        results["avg_reward"] = total_reward / max(len(samples), 1)
        results["accuracy"] = results["correct"] / max(len(samples), 1)
        results["json_valid_rate"] = results["json_valid"] / max(len(samples), 1)
        results["function_accuracy"] = results["function_correct"] / max(len(samples), 1)
        
        logger.info(f"Benchmark complete: accuracy={results['accuracy']:.2%}")
        
        return results
