"""Reward Model configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED


@dataclass
class RMConfig:
    """Configuration for reward-model training (Bradley-Terry).

    Trains a sequence-classification head on top of a base LM, fed
    `(prompt + chosen)` and `(prompt + rejected)` as a pair. The loss
    is the Bradley-Terry log-likelihood: -log σ(r(chosen) − r(rejected)).
    """

    # Base model — usually the SFT or instruct version of your target.
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None
    # Apple M5+ Neural Accelerator surfacing. Experimental annotation only.
    enable_neural_accelerators: bool = False

    # Data — preference pairs. Same shape as DPO datasets.
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    dataset: Optional[str] = None
    dataset_split: str = "train"
    max_samples: Optional[int] = None
    validation_split: float = 0.05
    max_length: int = 1024  # combined prompt+response cap

    # QLoRA — same surface as DPOConfig.
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    allow_quantization_fallback: bool = True

    # LoRA / PEFT — adds a tiny extra head on the base; keep ranks small.
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    use_dora: bool = False
    use_rslora: bool = False
    init_lora_weights: str = "true"

    optim: str = "adamw_torch"

    # Training
    output_dir: str = "models/rm"
    num_epochs: int = 1
    # Absolute optimizer-step ceiling for a resumable sweep segment.
    max_steps: Optional[int] = None
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5  # RMs train fast; smaller than SFT
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = DEFAULT_TRAINING_SEED

    save_steps: int = 200
    save_total_limit: int = 3
    eval_steps: int = 100
    logging_steps: int = 10

    # Center the reward range so the RM doesn't drift to producing huge
    # absolute values — most public RM recipes do this.
    center_rewards_coefficient: Optional[float] = 0.01

    def __post_init__(self):
        if self.max_steps is not None and int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive when provided")
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


__all__ = ["RMConfig"]
