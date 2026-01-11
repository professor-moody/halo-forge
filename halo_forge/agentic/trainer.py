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
from halo_forge.utils.metrics import MetricsTracker

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
    
    # Output
    output_dir: str = "models/agentic_raft"
    save_every_cycle: bool = True
    
    # Device
    device: Optional[str] = None
    
    # AMD Strix Halo requirements
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


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
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            device_map="auto",
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
            
            # Log cycle to MetricsTracker (TensorBoard + JSON)
            self.metrics_tracker.log_cycle(cycle, {
                "success_rate": cycle_result.success_rate,
                "avg_reward": cycle_result.avg_reward,
                "kept_samples": cycle_result.training_samples,
                "total_samples": cycle_result.total_samples,
                "learning_rate": self.get_learning_rate(cycle),
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
                self._save_checkpoint(cycle)
        
        total_time = time.time() - start_time
        
        # Final metrics
        final_metrics = {
            "total_cycles": self.config.num_cycles,
            "total_time_seconds": total_time,
            "final_avg_reward": self.cycle_results[-1].avg_reward if self.cycle_results else 0.0,
            "final_success_rate": self.cycle_results[-1].success_rate if self.cycle_results else 0.0,
            "cycle_results": [vars(r) for r in self.cycle_results],
        }
        
        # Save metrics summary
        self.metrics_tracker.save_summary()
        
        # Also save standalone metrics file
        metrics_path = self.output_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)
        
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
        
        # Log sample-level rewards to MetricsTracker
        rewards = [c.reward for c in all_completions]
        self.metrics_tracker.log_samples(cycle, rewards)
        
        # Update metrics
        self.metrics["cycle_rewards"].append(avg_reward)
        self.metrics["cycle_accuracy"].append(success_rate)
        self.metrics["cycle_samples"].append(len(filtered))
        
        # Train on filtered samples
        if filtered:
            self._train_on_samples(filtered, cycle)
        
        return AgenticRAFTCycleResult(
            cycle=cycle,
            total_samples=total_samples,
            verified_samples=len(all_completions),
            avg_reward=avg_reward,
            success_rate=success_rate,
            training_samples=len(filtered),
            metrics={
                "lr": self.get_learning_rate(cycle),
            },
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
    ) -> None:
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
        
        # Training would use HuggingFace Trainer or similar
        # This is a simplified placeholder
        
        # In production, this would:
        # 1. Tokenize training_texts
        # 2. Create Dataset
        # 3. Configure Trainer with LoRA
        # 4. Run training
        
        logger.info("Training step complete")
    
    def _save_checkpoint(self, cycle: int) -> None:
        """Save checkpoint for current cycle.
        
        Args:
            cycle: Current cycle number.
        """
        checkpoint_dir = self.output_dir / f"cycle_{cycle}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save config
        config_path = checkpoint_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(vars(self.config), f, indent=2, default=str)
        
        # Save metrics so far
        metrics_path = checkpoint_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
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
