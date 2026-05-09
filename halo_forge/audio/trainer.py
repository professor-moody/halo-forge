"""
Audio RAFT Trainer

RAFT (Reward-rAnked Fine-Tuning) for audio-language models.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import logging
import json
import time

import torch
from tqdm import tqdm

from halo_forge.audio.data import AudioSample, AudioProcessor, load_audio_dataset
from halo_forge.audio.verifiers import AudioVerifier, AudioVerifyConfig
from halo_forge.audio.models import (
    AudioAdapter,
    get_audio_adapter,
    supports_audio_training,
)
from halo_forge.capabilities import is_model_family_supported, get_supported_model_families
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
from halo_forge.training_eligibility import is_training_eligible
from halo_forge.training_recovery import attach_recovery_guidance
from halo_forge.modality_artifacts import (
    persist_cycle_artifacts,
    persist_final_artifacts,
    resolve_resume_checkpoint,
)
from halo_forge.utils.accelerator import get_torch_device
from halo_forge.runtime_determinism import (
    DEFAULT_TRAINING_SEED,
    build_run_id,
    set_global_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class AudioRAFTConfig:
    """Configuration for Audio RAFT training."""
    
    # Model
    model_name: str = "openai/whisper-small"
    adapter_type: Optional[str] = None  # auto-detect
    
    # Task
    task: str = "asr"  # asr, tts, classification
    
    # Training
    num_cycles: int = 6
    samples_per_prompt: int = 4
    reward_threshold: float = 0.5
    keep_top_percent: float = 0.5
    
    # Audio
    sample_rate: int = 16000
    max_audio_length: float = 30.0  # seconds
    
    # Learning rate
    learning_rate: float = 5e-5
    lr_decay_per_cycle: float = 0.85
    min_lr: float = 1e-6
    
    # Batch
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Output
    output_dir: str = "models/audio_raft"
    save_every_cycle: bool = True
    
    # Verification
    wer_threshold: float = 0.3
    
    # Device
    device: Optional[str] = None
    seed: int = DEFAULT_TRAINING_SEED

    def __post_init__(self):
        if self.device is None:
            self.device = str(get_torch_device())


@dataclass
class AudioRAFTCycleResult:
    """Result of a single RAFT cycle."""
    
    cycle: int
    samples_generated: int
    samples_verified: int
    samples_kept: int
    average_reward: float
    learning_rate: float
    metrics: Dict[str, Any] = field(default_factory=dict)


class AudioRAFTTrainer:
    """
    RAFT Trainer for Audio-Language Models.
    
    Training loop:
    1. Generate transcriptions/outputs from audio samples
    2. Verify quality with task-specific verifier
    3. Filter by reward threshold
    4. Fine-tune on high-reward samples
    5. Repeat with decayed learning rate
    """
    
    def __init__(self, config: AudioRAFTConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.processor = AudioProcessor(
            sample_rate=config.sample_rate,
            max_duration=config.max_audio_length,
        )
        
        self.adapter: Optional[AudioAdapter] = None
        self.verifier: Optional[AudioVerifier] = None
        
        # Track metrics
        self.cycle_results: List[AudioRAFTCycleResult] = []
        self._all_cycle_metrics: List[Dict[str, Any]] = []
        self.training_summary: Dict[str, Any] = {}
        self.base_model_name = config.model_name
        self.run_id: str = ""
        self.resume_checkpoint_meta: Optional[Dict[str, Any]] = None
        self.representative_examples: List[Dict[str, Any]] = []
    
    def _init_adapter(self) -> None:
        """Initialize model adapter."""
        if self.adapter is None:
            self.adapter = get_audio_adapter(
                self.config.model_name,
                device=self.config.device,
            )
            self.adapter.load()
    
    def _init_verifier(self) -> None:
        """Initialize verifier."""
        if self.verifier is None:
            verify_config = AudioVerifyConfig(
                task=self.config.task,
                wer_threshold=self.config.wer_threshold,
            )
            self.verifier = AudioVerifier(verify_config)
    
    def get_learning_rate(self, cycle: int) -> float:
        """
        Get learning rate for cycle with exponential decay.
        
        Args:
            cycle: Current cycle (0-indexed)
            
        Returns:
            Learning rate for this cycle
        """
        lr = self.config.learning_rate * (self.config.lr_decay_per_cycle ** cycle)
        return max(lr, self.config.min_lr)
    
    def train(
        self,
        samples: Union[str, List[AudioSample]],
        validation_samples: Optional[List[AudioSample]] = None,
        resume_from_cycle: int = 0,
    ) -> List[AudioRAFTCycleResult]:
        """
        Run RAFT training.
        
        Args:
            samples: Dataset name or list of AudioSample
            validation_samples: Optional validation set
            
        Returns:
            List of cycle results
        """
        start_cycle = int(resume_from_cycle or 0)
        self.resume_checkpoint_meta = None
        self._validate_resume_configuration(start_cycle)
        if start_cycle > 0:
            checkpoint = resolve_resume_checkpoint(
                output_dir=self.output_dir,
                resume_from_cycle=start_cycle,
                max_cycles=self.config.num_cycles,
                modality="audio",
            )
            self.resume_checkpoint_meta = {
                "cycle": checkpoint.cycle,
                "cycle_dir": str(checkpoint.cycle_dir),
                "model_dir": str(checkpoint.model_dir),
            }
            self.config.model_name = str(checkpoint.model_dir)
            self._load_resume_history(start_cycle)

        if not (
            is_model_family_supported("audio", self.config.model_name)
            or is_model_family_supported("audio", self.base_model_name)
        ):
            families = ", ".join(get_supported_model_families("audio"))
            raise ValueError(
                f"Unsupported model family for audio training. Supported families: {families}."
            )

        normalized_seed = set_global_seed(self.config.seed)
        self.config.seed = normalized_seed
        self.run_id = build_run_id("audio")

        self._init_adapter()
        self._init_verifier()
        
        # Load dataset if string
        if isinstance(samples, str):
            dataset = load_audio_dataset(samples, limit=None)
            samples = list(dataset)
        if not samples:
            raise ValueError("audio training requires at least one sample")
        
        logger.info(f"Starting AudioRAFT training with {len(samples)} samples")
        logger.info(f"Task: {self.config.task}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Cycles: {self.config.num_cycles}")
        
        for cycle in range(start_cycle, self.config.num_cycles):
            logger.info(f"\n{'='*60}")
            logger.info(f"CYCLE {cycle + 1}/{self.config.num_cycles}")
            logger.info(f"{'='*60}")
            
            result = self._train_cycle(cycle, samples)
            self.cycle_results.append(result)
            self._all_cycle_metrics.append(result.metrics)
            write_json_atomic(
                self.output_dir / "training_history.json",
                {"cycles": self._all_cycle_metrics},
            )
            
            # Save checkpoint
            if self.config.save_every_cycle:
                self._save_checkpoint(cycle, result.metrics)
            
            # Log progress
            logger.info(f"Cycle {cycle + 1} complete:")
            logger.info(f"  Samples kept: {result.samples_kept}/{result.samples_verified}")
            logger.info(f"  Average reward: {result.average_reward:.3f}")
            logger.info(f"  Learning rate: {result.learning_rate:.2e}")
            logger.info(
                "  Update: steps=%d, weights_updated=%s, loss=%s",
                result.metrics.get("train_steps_executed", 0),
                result.metrics.get("weights_updated", False),
                (
                    f"{result.metrics['train_loss']:.4f}"
                    if isinstance(result.metrics.get("train_loss"), (float, int))
                    else "n/a"
                ),
            )
        
        summary = build_training_summary(
            modality="audio",
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
                "task": self.config.task,
                "samples": len(samples),
                "final_average_reward": (
                    self.cycle_results[-1].average_reward if self.cycle_results else 0.0
                ),
                # Backward-compatible alias used by some callsites.
                "cycle_results": [
                    {
                        "cycle": result.cycle,
                        "samples_generated": result.samples_generated,
                        "samples_verified": result.samples_verified,
                        "samples_kept": result.samples_kept,
                        "average_reward": result.average_reward,
                        "learning_rate": result.learning_rate,
                        "metrics": result.metrics,
                    }
                    for result in self.cycle_results
                ],
            },
        )
        final_state = persist_final_artifacts(
            output_dir=self.output_dir,
            modality="audio",
            model_name=self.config.model_name,
            model=getattr(self.adapter, "model", None),
            tokenizer=getattr(self.adapter, "tokenizer", None),
            processor=getattr(self.adapter, "processor", None),
            extra_state={"task": self.config.task},
        )
        summary["final_model_path"] = final_state["final_model_dir"]
        attach_effectiveness_contract(
            summary,
            minimum_samples_kept=1,
            minimum_optimizer_steps=1,
            evaluation={
                "metric_name": "average_reward",
                "baseline_value": (
                    self.cycle_results[0].average_reward if self.cycle_results else None
                ),
                "final_value": (
                    self.cycle_results[-1].average_reward if self.cycle_results else None
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
            summary,
            modality="audio",
            launch_args={
                "model": self.base_model_name,
                "dataset": "",
                "output_dir": str(self.output_dir),
                "cycles": self.config.num_cycles,
                "learning_rate": self.config.learning_rate,
                "lr_decay": self.config.lr_decay_per_cycle,
                "samples_per_prompt": self.config.samples_per_prompt,
                "keep_percent": self.config.keep_top_percent,
                "reward_threshold": self.config.reward_threshold,
                "task": self.config.task,
            },
            representative_examples=self.representative_examples,
        )
        self.training_summary = summary
        write_json_atomic(self.output_dir / "training_summary.json", summary)
        
        return self.cycle_results
    
    def _train_cycle(
        self,
        cycle: int,
        samples: List[AudioSample],
    ) -> AudioRAFTCycleResult:
        """
        Run a single RAFT cycle.
        
        Args:
            cycle: Cycle number (0-indexed)
            samples: Training samples
            
        Returns:
            Cycle result
        """
        cycle_start = time.time()
        lr = self.get_learning_rate(cycle)
        logger.info(f"Learning rate: {lr:.2e}")
        
        # 1. Generate predictions
        logger.info("Generating predictions...")
        predictions = self._generate_predictions(samples)
        
        # 2. Verify
        logger.info("Verifying predictions...")
        verified = self._verify_predictions(predictions, samples)
        
        # 3. Filter by reward
        logger.info("Filtering by reward...")
        kept = self._filter_samples(verified)
        
        # 4. Calculate metrics
        rewards = [v["reward"] for v in verified]
        successful = sum(1 for v in verified if v.get("success"))
        above_threshold = sum(
            1 for v in verified if is_training_eligible(v, self.config.reward_threshold)
        )
        if not self.representative_examples:
            failed = [
                {
                    "reason": "verification_failed",
                    "label": "Verifier mismatch",
                    "preview": v.get("prediction", ""),
                    "context": v.get("ground_truth", ""),
                    "reward": v.get("reward"),
                }
                for v in verified
                if not v.get("success")
            ][:3]
            dropped = [
                {
                    "reason": "below_reward_threshold",
                    "label": "Below threshold",
                    "preview": v.get("prediction", ""),
                    "context": v.get("ground_truth", ""),
                    "reward": v.get("reward"),
                }
                for v in verified
                if v.get("success") and float(v.get("reward", 0.0)) < self.config.reward_threshold
            ][:3]
            self.representative_examples = failed or dropped
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        train_metrics: Dict[str, Any]

        if kept:
            logger.info(f"Training on {len(kept)} samples...")
            train_metrics = self._train_on_samples(kept, lr)
        else:
            train_metrics = {
                "train_steps_executed": 0,
                "train_loss": None,
                "weights_updated": False,
                "update_reason": "no_filtered_samples",
            }

        canonical_metrics = build_cycle_summary(
            cycle=cycle,
            learning_rate=lr,
            samples_seen=len(verified),
            samples_kept=len(kept),
            cycle_duration_seconds=time.time() - cycle_start,
            update_metrics=normalize_update_metrics(
                train_metrics,
                default_reason="no_filtered_samples",
            ),
            yield_diagnostics={
                "stage_counts": {
                    "generated": len(predictions),
                    "verified": len(verified),
                    "filtered": above_threshold,
                    "kept": len(kept),
                    "dropped": max(0, len(predictions) - len(kept)),
                },
                "rates": {
                    "verification_rate": (successful / len(verified)) if verified else 0.0,
                    "success_rate": (successful / len(verified)) if verified else 0.0,
                },
                "thresholds": {
                    "configured_reward_threshold": self.config.reward_threshold,
                    "effective_reward_threshold": self.config.reward_threshold,
                    "keep_percent": self.config.keep_top_percent,
                    "threshold_adjusted": False,
                },
                "minimums": {"minimum_samples_target": 1},
                "rejection_reasons": build_reward_filter_rejection_reasons(
                    total_count=len(verified),
                    success_count=successful,
                    above_threshold_count=above_threshold,
                    kept_count=len(kept),
                ),
                "reward_distribution": build_reward_distribution_from_values(rewards),
                "summary": {
                    "text": (
                        "Audio yield looks healthy for continued updates."
                        if len(kept) > 0 and successful >= max(1, len(verified) // 3)
                        else "Audio yield is low; consider lowering the reward threshold or increasing sample budget."
                    )
                },
            },
            extra={
                "min_reward": min(rewards) if rewards else 0.0,
                "max_reward": max(rewards) if rewards else 0.0,
                "average_reward": avg_reward,
            },
        )
        emit_yield_log_line(
            {
                "cycle": cycle,
                **canonical_metrics["yield_diagnostics"],
            }
        )

        return AudioRAFTCycleResult(
            cycle=cycle,
            samples_generated=len(predictions),
            samples_verified=len(verified),
            samples_kept=len(kept),
            average_reward=avg_reward,
            learning_rate=lr,
            metrics=canonical_metrics,
        )
    
    def _generate_predictions(
        self,
        samples: List[AudioSample],
        show_progress: bool = True,
    ) -> List[str]:
        """Generate predictions for samples."""
        predictions = []
        
        iterator = tqdm(samples, desc="Transcribing") if show_progress else samples
        
        for sample in iterator:
            try:
                # Get audio
                if sample.audio_array is not None:
                    processed = self.processor.load_array(
                        sample.audio_array,
                        sample.sample_rate or 16000,
                    )
                else:
                    processed = self.processor.load(sample.audio_path)
                
                # Transcribe
                result = self.adapter.transcribe(processed.waveform)
                predictions.append(result.text)
            
            except Exception as e:
                logger.warning(f"Failed to process sample: {e}")
                predictions.append("")
        
        return predictions
    
    def _verify_predictions(
        self,
        predictions: List[str],
        samples: List[AudioSample],
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Verify predictions against ground truth."""
        verified = []
        
        pairs = list(zip(predictions, samples))
        iterator = tqdm(pairs, desc="Verifying") if show_progress else pairs
        
        for pred, sample in iterator:
            result = self.verifier.verify(pred, sample.text)
            verified.append({
                "prediction": pred,
                "ground_truth": sample.text,
                "reward": result.reward,
                "success": result.success,
                "details": result.details,
                "sample": sample,
            })
        
        return verified
    
    def _filter_samples(
        self,
        verified: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter samples by reward threshold."""
        eligible = [
            s for s in verified
            if is_training_eligible(s, self.config.reward_threshold)
        ]

        # Sort by reward
        sorted_samples = sorted(eligible, key=lambda x: x["reward"], reverse=True)
        
        # Keep top percent
        keep_count = int(len(sorted_samples) * self.config.keep_top_percent)
        kept = sorted_samples[:keep_count]
        
        return kept

    def _train_on_samples(self, kept: List[Dict[str, Any]], lr: float) -> Dict[str, Any]:
        """Run real optimizer updates for supported audio model families."""
        if not supports_audio_training(self.config.model_name):
            return {
                "train_steps_executed": 0,
                "train_loss": None,
                "weights_updated": False,
                "update_reason": "unsupported_model_family",
                "optimizer_steps": 0,
                "skipped_batches_non_finite": 0,
            }

        if not kept:
            return {
                "train_steps_executed": 0,
                "train_loss": None,
                "weights_updated": False,
                "update_reason": "no_filtered_samples",
                "optimizer_steps": 0,
                "skipped_batches_non_finite": 0,
            }

        model = getattr(self.adapter, "model", None)
        processor = getattr(self.adapter, "processor", None)
        if model is None or processor is None:
            return {
                "train_steps_executed": 0,
                "train_loss": None,
                "weights_updated": False,
                "update_reason": "adapter_not_loaded",
                "optimizer_steps": 0,
                "skipped_batches_non_finite": 0,
            }

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        optimizer_steps = 0
        micro_steps = 0
        grad_accum = max(1, self.config.gradient_accumulation_steps)
        last_loss_value = 0.0
        initial_loss_value = None
        skipped_batches_non_finite = 0

        for item in kept:
            sample: AudioSample = item["sample"]
            try:
                if sample.audio_array is not None:
                    processed = self.processor.load_array(
                        sample.audio_array,
                        sample.sample_rate or self.config.sample_rate,
                    )
                else:
                    processed = self.processor.load(sample.audio_path)
            except Exception as e:
                logger.warning("Skipping sample during train update: %s", e)
                continue

            inputs = processor(
                processed.waveform.cpu().numpy(),
                sampling_rate=self.config.sample_rate,
                return_tensors="pt",
            )
            labels = processor(text=sample.text, return_tensors="pt").input_ids

            model_dtype = next(model.parameters()).dtype
            device = model.device
            input_features = inputs.input_features.to(device=device, dtype=model_dtype)
            labels = labels.to(device)

            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            if loss is None:
                continue
            if not torch.isfinite(loss).item():
                skipped_batches_non_finite += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            if initial_loss_value is None:
                initial_loss_value = float(loss.detach().item())
            (loss / grad_accum).backward()
            last_loss_value = float(loss.detach().item())
            micro_steps += 1

            if micro_steps % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                total_loss += last_loss_value

        if micro_steps % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            total_loss += last_loss_value

        if optimizer_steps == 0:
            return {
                "train_steps_executed": 0,
                "train_loss": None,
                "initial_train_loss": initial_loss_value,
                "weights_updated": False,
                "update_reason": "no_optimizer_steps",
                "optimizer_steps": 0,
                "skipped_batches_non_finite": skipped_batches_non_finite,
            }

        return {
            "train_steps_executed": optimizer_steps,
            "train_loss": total_loss / optimizer_steps if total_loss else 0.0,
            "initial_train_loss": initial_loss_value,
            "weights_updated": True,
            "update_reason": "updated",
            "optimizer_steps": optimizer_steps,
            "skipped_batches_non_finite": skipped_batches_non_finite,
        }
    
    def _save_checkpoint(self, cycle: int, cycle_metrics: Dict[str, Any]) -> None:
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"cycle_{cycle}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = checkpoint_dir / "config.json"
        write_json_atomic(
            config_path,
            {
                "model_name": self.config.model_name,
                "task": self.config.task,
                "cycle": cycle,
                "learning_rate": self.get_learning_rate(cycle),
            },
        )
        
        # Save metrics
        metrics_path = checkpoint_dir / "metrics.json"
        results_data = [
            {
                "cycle": r.cycle + 1,
                "samples_kept": r.samples_kept,
                "average_reward": r.average_reward,
                "learning_rate": r.learning_rate,
                "train_steps_executed": r.metrics.get("train_steps_executed", 0),
                "train_loss": r.metrics.get("train_loss"),
                "weights_updated": r.metrics.get("weights_updated", False),
                "update_reason": r.metrics.get("update_reason"),
            }
            for r in self.cycle_results
        ]
        write_json_atomic(metrics_path, {"cycles": results_data})
        persist_cycle_artifacts(
            output_dir=self.output_dir,
            modality="audio",
            model_name=self.config.model_name,
            cycle=cycle,
            update_metrics=cycle_metrics,
            model=getattr(self.adapter, "model", None),
            tokenizer=getattr(self.adapter, "tokenizer", None),
            processor=getattr(self.adapter, "processor", None),
            extra_state={"task": self.config.task},
        )
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def _validate_resume_configuration(self, start_cycle: int) -> None:
        """Validate resume inputs against checkpoint artifacts."""
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
            modality="audio",
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
    
    def benchmark(
        self,
        samples: Union[str, List[AudioSample]],
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark model on samples.
        
        Args:
            samples: Dataset name or list of samples
            limit: Limit number of samples
            
        Returns:
            Benchmark results
        """
        self._init_adapter()
        self._init_verifier()
        
        # Load dataset if string
        if isinstance(samples, str):
            dataset = load_audio_dataset(samples, limit=limit)
            samples = list(dataset)
        elif limit:
            samples = samples[:limit]
        
        logger.info(f"Benchmarking on {len(samples)} samples...")
        
        # Generate and verify
        predictions = self._generate_predictions(samples)
        verified = self._verify_predictions(predictions, samples)
        
        # Calculate metrics
        rewards = [v["reward"] for v in verified]
        successes = sum(1 for v in verified if v["success"])
        
        results = {
            "samples": len(samples),
            "success_rate": successes / len(samples) if samples else 0.0,
            "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
        }
        
        if self.config.task == "asr":
            wers = [v["details"].get("wer", 1.0) for v in verified]
            results["average_wer"] = sum(wers) / len(wers) if wers else 1.0
        
        return results
