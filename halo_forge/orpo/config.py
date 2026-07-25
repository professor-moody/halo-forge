"""ORPO configuration dataclass.

Mirrors `DPOConfig` for shared fields (model, data, LoRA/QLoRA, optim,
schedule). The ORPO-specific knob is `beta`, the relative weight of
the preference (log-odds) term against the NLL term — 0.1 is the value
the original paper recommends.

Reference-free by construction; there's no `reference_free` toggle to
flip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED


@dataclass
class ORPOConfig:
    """Configuration for ORPO training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None
    # Apple M5+ Neural Accelerator surfacing. Experimental annotation only.
    enable_neural_accelerators: bool = False

    # Data
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    dataset: Optional[str] = None
    dataset_split: str = "train"
    max_samples: Optional[int] = None
    validation_split: float = 0.05
    max_seq_length: int = 1024
    max_prompt_length: int = 512

    # ORPO algorithm
    # `beta` blends NLL + log-odds. 0.1 is the value from the original
    # paper (Hong et al., 2024). Lower = more SFT-like; higher = more
    # preference-driven.
    beta: float = 0.1

    # QLoRA — same shape as SFTConfig / DPOConfig.
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    allow_quantization_fallback: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    use_dora: bool = False
    use_rslora: bool = False
    init_lora_weights: str = "true"

    # Optimizer
    optim: str = "adamw_torch"

    # Training
    output_dir: str = "models/orpo"
    num_epochs: int = 1
    # Absolute optimizer-step ceiling for a resumable sweep segment.
    max_steps: Optional[int] = None
    # ORPO trains on chosen+rejected concatenated per row, so peak memory
    # is similar to DPO. Default to small batch + accumulation.
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 8e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    # Optimization
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = DEFAULT_TRAINING_SEED

    # Saving
    save_steps: int = 200
    save_total_limit: int = 3
    eval_steps: int = 100
    logging_steps: int = 10

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0

    def __post_init__(self):
        if self.max_steps is not None and int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive when provided")
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


__all__ = ["ORPOConfig"]
