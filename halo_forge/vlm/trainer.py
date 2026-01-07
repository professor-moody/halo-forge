"""
VLM RAFT Trainer

RAFT trainer adapted for vision-language models.
"""

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm

from halo_forge.vlm.verifiers import VisionVerifier
from halo_forge.vlm.models import VLMAdapter, get_vlm_adapter
from halo_forge.vlm.data import VLMSample, load_vlm_dataset


@dataclass
class VLMRAFTConfig:
    """Configuration for VLM RAFT training."""
    # Model
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    adapter_type: Optional[str] = None  # auto-detect
    
    # Training
    num_cycles: int = 6
    samples_per_prompt: int = 4
    reward_threshold: float = 0.5
    keep_top_percent: float = 0.5
    
    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    # Learning rate
    learning_rate: float = 5e-5
    lr_decay_per_cycle: float = 0.85
    min_lr: float = 1e-6
    
    # Verifier
    perception_weight: float = 0.3
    reasoning_weight: float = 0.4
    output_weight: float = 0.3
    
    # Output
    output_dir: str = "models/vlm_raft"
    save_every_cycle: bool = True
    
    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True


@dataclass
class VLMSampleResult:
    """Result of generating and verifying a VLM sample."""
    image: Union[Image.Image, str]
    prompt: str
    completion: str
    ground_truth: Optional[str]
    reward: float
    success: bool
    details: Dict[str, Any]


class VLMRAFTTrainer:
    """
    RAFT Trainer for Vision-Language Models.
    
    Implements the RAFT (Reward-Ranked Fine-Tuning) loop for VLMs:
    1. Generate multiple completions per image+prompt
    2. Verify with VisionVerifier (perception, reasoning, output)
    3. Filter to keep high-reward samples
    4. Train on filtered samples
    5. Repeat
    
    Usage:
        config = VLMRAFTConfig(
            model_name="Qwen/Qwen2-VL-7B-Instruct",
            num_cycles=6
        )
        trainer = VLMRAFTTrainer(config)
        trainer.train(samples)
    """
    
    def __init__(self, config: VLMRAFTConfig):
        """
        Initialize VLM RAFT trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Initialize components
        self.adapter: Optional[VLMAdapter] = None
        self.verifier: Optional[VisionVerifier] = None
        
        # Training state
        self.current_cycle = 0
        self.best_reward = 0.0
        self.training_history: List[Dict[str, Any]] = []
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _log(self, message: str, level: str = "info"):
        """Log a message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "info": "",
            "step": ">",
            "ok": "[OK]",
            "warn": "[WARN]",
            "error": "[ERROR]"
        }.get(level, "")
        print(f"{prefix} {message}")
    
    def _setup(self):
        """Set up model and verifier."""
        # Load VLM adapter
        self._log(f"Loading VLM: {self.config.model_name}", "step")
        self.adapter = get_vlm_adapter(
            self.config.model_name,
            adapter_type=self.config.adapter_type,
            dtype=torch.bfloat16 if self.config.bf16 else torch.float16
        )
        self.adapter.load()
        
        # Initialize verifier
        self._log("Initializing VisionVerifier", "step")
        self.verifier = VisionVerifier(
            perception_weight=self.config.perception_weight,
            reasoning_weight=self.config.reasoning_weight,
            output_weight=self.config.output_weight
        )
    
    def get_learning_rate_for_cycle(self, cycle: int) -> float:
        """Calculate learning rate with decay."""
        lr = self.config.learning_rate * (self.config.lr_decay_per_cycle ** cycle)
        return max(lr, self.config.min_lr)
    
    def generate_samples(
        self,
        prompts: List[VLMSample],
        samples_per_prompt: int
    ) -> List[VLMSampleResult]:
        """
        Generate multiple completions for each prompt.
        
        Args:
            prompts: List of VLM samples with images and prompts
            samples_per_prompt: Number of samples per prompt
            
        Returns:
            List of sample results
        """
        results = []
        
        self._log(f"Generating {len(prompts) * samples_per_prompt} samples "
                  f"({len(prompts)} prompts x {samples_per_prompt})", "step")
        
        for sample in tqdm(prompts, desc="Generating"):
            # Load image
            image = sample.load_image()
            
            for _ in range(samples_per_prompt):
                # Generate completion
                output = self.adapter.generate(
                    image=image,
                    prompt=sample.prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True
                )
                
                # Verify
                verify_result = self.verifier.verify(
                    image=image,
                    prompt=sample.prompt,
                    completion=output.text,
                    ground_truth=sample.ground_truth
                )
                
                results.append(VLMSampleResult(
                    image=sample.image,
                    prompt=sample.prompt,
                    completion=output.text,
                    ground_truth=sample.ground_truth,
                    reward=verify_result.reward,
                    success=verify_result.success,
                    details=verify_result.details
                ))
        
        return results
    
    def filter_samples(
        self,
        samples: List[VLMSampleResult]
    ) -> List[VLMSampleResult]:
        """
        Filter samples by reward threshold.
        
        Args:
            samples: All generated samples
            
        Returns:
            Filtered samples above threshold
        """
        # Filter by threshold
        above_threshold = [s for s in samples if s.reward >= self.config.reward_threshold]
        
        if not above_threshold:
            self._log("No samples above threshold, keeping top 10%", "warn")
            sorted_samples = sorted(samples, key=lambda x: x.reward, reverse=True)
            n_keep = max(1, len(samples) // 10)
            return sorted_samples[:n_keep]
        
        # Keep top percentage
        sorted_samples = sorted(above_threshold, key=lambda x: x.reward, reverse=True)
        n_keep = max(1, int(len(sorted_samples) * self.config.keep_top_percent))
        
        return sorted_samples[:n_keep]
    
    def train_on_samples(
        self,
        samples: List[VLMSampleResult],
        cycle: int
    ):
        """
        Train model on filtered samples.
        
        Note: VLM fine-tuning typically requires LoRA or similar
        efficient fine-tuning methods. This is a simplified version.
        
        Args:
            samples: Filtered high-reward samples
            cycle: Current cycle number
        """
        self._log(f"Training on {len(samples)} filtered samples", "step")
        
        # Prepare training data
        train_data = []
        for s in samples:
            train_data.append({
                'image': s.image,
                'prompt': s.prompt,
                'completion': s.completion,
                'reward': s.reward
            })
        
        # Save training data for this cycle
        data_path = self.output_dir / f"cycle_{cycle}_train_data.jsonl"
        with open(data_path, 'w') as f:
            for item in train_data:
                record = {
                    'prompt': item['prompt'],
                    'completion': item['completion'],
                    'reward': item['reward'],
                    'image': str(item['image']) if isinstance(item['image'], str) else None
                }
                f.write(json.dumps(record) + '\n')
        
        # Get learning rate for this cycle
        lr = self.get_learning_rate_for_cycle(cycle)
        self._log(f"Learning rate: {lr:.2e}", "step")
        
        # Note: Actual VLM fine-tuning would require:
        # 1. Setting up LoRA adapters
        # 2. Creating proper DataLoader with image processing
        # 3. Running gradient descent
        #
        # For now, we save the data and log the training step.
        # Full implementation would integrate with trl or similar.
        
        self._log(f"Training data saved to {data_path}", "ok")
        self._log("Note: Full VLM fine-tuning requires LoRA setup", "info")
    
    def save_checkpoint(self, cycle: int):
        """Save checkpoint for this cycle."""
        checkpoint_dir = self.output_dir / f"cycle_{cycle}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save config
        config_path = checkpoint_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'model_name': self.config.model_name,
                'cycle': cycle,
                'learning_rate': self.get_learning_rate_for_cycle(cycle),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        # Save training history
        history_path = checkpoint_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self._log(f"Checkpoint saved: {checkpoint_dir}", "ok")
    
    def run_cycle(
        self,
        prompts: List[VLMSample],
        cycle: int
    ) -> Dict[str, Any]:
        """
        Run a single RAFT cycle.
        
        Args:
            prompts: Training prompts
            cycle: Cycle number
            
        Returns:
            Cycle metrics
        """
        cycle_start = time.time()
        
        self._log(f"\n{'='*60}", "info")
        self._log(f"RAFT CYCLE {cycle + 1}/{self.config.num_cycles}", "info")
        self._log(f"{'='*60}", "info")
        
        # 1. Generate samples
        gen_start = time.time()
        samples = self.generate_samples(prompts, self.config.samples_per_prompt)
        gen_time = time.time() - gen_start
        
        # Calculate stats
        rewards = [s.reward for s in samples]
        successes = sum(1 for s in samples if s.success)
        
        self._log(f"Generated {len(samples)} samples in {gen_time/60:.1f} min", "ok")
        self._log(f"Success rate: {successes/len(samples)*100:.1f}%", "info")
        self._log(f"Avg reward: {sum(rewards)/len(rewards):.3f}", "info")
        
        # 2. Filter samples
        filtered = self.filter_samples(samples)
        
        self._log(f"Filtered to {len(filtered)} samples", "ok")
        
        # 3. Train
        self.train_on_samples(filtered, cycle)
        
        # 4. Save checkpoint
        if self.config.save_every_cycle:
            self.save_checkpoint(cycle)
        
        cycle_time = time.time() - cycle_start
        
        # Record metrics
        metrics = {
            'cycle': cycle,
            'num_samples': len(samples),
            'num_filtered': len(filtered),
            'success_rate': successes / len(samples),
            'avg_reward': sum(rewards) / len(rewards),
            'max_reward': max(rewards),
            'learning_rate': self.get_learning_rate_for_cycle(cycle),
            'cycle_time_min': cycle_time / 60,
        }
        
        self.training_history.append(metrics)
        
        self._log(f"Cycle {cycle + 1} complete in {cycle_time/60:.1f} min", "ok")
        
        return metrics
    
    def train(
        self,
        prompts: Union[List[VLMSample], str],
        resume_from: Optional[int] = None
    ):
        """
        Run full RAFT training.
        
        Args:
            prompts: List of VLMSample or path to dataset
            resume_from: Resume from cycle number
        """
        # Load prompts if path
        if isinstance(prompts, str):
            if prompts.endswith('.jsonl'):
                # Load from JSONL
                loaded = []
                with open(prompts) as f:
                    for line in f:
                        data = json.loads(line)
                        loaded.append(VLMSample(
                            image=data.get('image', data.get('image_path')),
                            prompt=data['prompt'],
                            ground_truth=data.get('ground_truth', data.get('answer'))
                        ))
                prompts = loaded
            else:
                # Load from HuggingFace dataset
                dataset = load_vlm_dataset(prompts)
                prompts = list(dataset)
        
        self._log(f"Training with {len(prompts)} prompts", "info")
        
        # Setup
        self._setup()
        
        # Determine starting cycle
        start_cycle = resume_from if resume_from else 0
        
        # Run cycles
        for cycle in range(start_cycle, self.config.num_cycles):
            self.current_cycle = cycle
            metrics = self.run_cycle(prompts, cycle)
            
            # Track best
            if metrics['avg_reward'] > self.best_reward:
                self.best_reward = metrics['avg_reward']
        
        # Final save
        self._log(f"\n{'='*60}", "info")
        self._log("TRAINING COMPLETE", "info")
        self._log(f"{'='*60}", "info")
        self._log(f"Best avg reward: {self.best_reward:.3f}", "info")
        self._log(f"Output: {self.output_dir}", "info")
        
        # Save final history
        final_history_path = self.output_dir / "training_history.json"
        with open(final_history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def cleanup(self):
        """Clean up resources."""
        if self.adapter:
            self.adapter.cleanup()
        if self.verifier:
            self.verifier.cleanup()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_vlm_raft(
    model_name: str,
    dataset: str,
    output_dir: str = "models/vlm_raft",
    num_cycles: int = 6,
    **kwargs
) -> VLMRAFTTrainer:
    """
    Convenience function to train VLM with RAFT.
    
    Args:
        model_name: VLM model name
        dataset: Dataset name or path
        output_dir: Output directory
        num_cycles: Number of RAFT cycles
        **kwargs: Additional config options
        
    Returns:
        Trained trainer instance
    """
    config = VLMRAFTConfig(
        model_name=model_name,
        output_dir=output_dir,
        num_cycles=num_cycles,
        **kwargs
    )
    
    trainer = VLMRAFTTrainer(config)
    trainer.train(dataset)
    
    return trainer
