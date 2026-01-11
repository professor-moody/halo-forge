"""
Audio RAFT Trainer

RAFT (Reward-rAnked Fine-Tuning) for audio-language models.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import logging
import json
import os

import torch
from tqdm import tqdm

from halo_forge.audio.data import AudioSample, AudioProcessor, load_audio_dataset
from halo_forge.audio.verifiers import AudioVerifier, AudioVerifyConfig
from halo_forge.audio.models import get_audio_adapter, AudioAdapter

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
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


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
    ) -> List[AudioRAFTCycleResult]:
        """
        Run RAFT training.
        
        Args:
            samples: Dataset name or list of AudioSample
            validation_samples: Optional validation set
            
        Returns:
            List of cycle results
        """
        self._init_adapter()
        self._init_verifier()
        
        # Load dataset if string
        if isinstance(samples, str):
            dataset = load_audio_dataset(samples, limit=None)
            samples = list(dataset)
        
        logger.info(f"Starting AudioRAFT training with {len(samples)} samples")
        logger.info(f"Task: {self.config.task}")
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Cycles: {self.config.num_cycles}")
        
        for cycle in range(self.config.num_cycles):
            logger.info(f"\n{'='*60}")
            logger.info(f"CYCLE {cycle + 1}/{self.config.num_cycles}")
            logger.info(f"{'='*60}")
            
            result = self._train_cycle(cycle, samples)
            self.cycle_results.append(result)
            
            # Save checkpoint
            if self.config.save_every_cycle:
                self._save_checkpoint(cycle)
            
            # Log progress
            logger.info(f"Cycle {cycle + 1} complete:")
            logger.info(f"  Samples kept: {result.samples_kept}/{result.samples_verified}")
            logger.info(f"  Average reward: {result.average_reward:.3f}")
            logger.info(f"  Learning rate: {result.learning_rate:.2e}")
        
        # Save final model
        self._save_checkpoint(self.config.num_cycles - 1, final=True)
        
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
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        # 5. Train on kept samples (placeholder - actual training would happen here)
        if kept:
            logger.info(f"Training on {len(kept)} samples...")
            # self._train_on_samples(kept, lr)
        
        return AudioRAFTCycleResult(
            cycle=cycle,
            samples_generated=len(predictions),
            samples_verified=len(verified),
            samples_kept=len(kept),
            average_reward=avg_reward,
            learning_rate=lr,
            metrics={
                "min_reward": min(rewards) if rewards else 0.0,
                "max_reward": max(rewards) if rewards else 0.0,
            }
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
            })
        
        return verified
    
    def _filter_samples(
        self,
        verified: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter samples by reward threshold."""
        # Sort by reward
        sorted_samples = sorted(verified, key=lambda x: x["reward"], reverse=True)
        
        # Keep top percent
        keep_count = int(len(sorted_samples) * self.config.keep_top_percent)
        kept = sorted_samples[:keep_count]
        
        # Also filter by threshold
        kept = [s for s in kept if s["reward"] >= self.config.reward_threshold]
        
        return kept
    
    def _save_checkpoint(self, cycle: int, final: bool = False) -> None:
        """Save model checkpoint."""
        suffix = "final" if final else f"cycle_{cycle + 1}"
        checkpoint_dir = self.output_dir / suffix
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "model_name": self.config.model_name,
                "task": self.config.task,
                "cycle": cycle + 1,
                "learning_rate": self.get_learning_rate(cycle),
            }, f, indent=2)
        
        # Save metrics
        metrics_path = checkpoint_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            results_data = [
                {
                    "cycle": r.cycle + 1,
                    "samples_kept": r.samples_kept,
                    "average_reward": r.average_reward,
                    "learning_rate": r.learning_rate,
                }
                for r in self.cycle_results
            ]
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
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
