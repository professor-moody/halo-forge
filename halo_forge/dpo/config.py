"""DPO configuration dataclass.

Lives outside `trainer.py` so it can be imported on hosts without torch
(MLX-only Macs, lightweight CI workers). The PyTorch trainer at
`halo_forge.dpo.trainer` consumes this `DPOConfig`; the future MLX trainer
will share the same shape.

Mirrors `SFTConfig` for the fields we genuinely share, then layers DPO-specific
knobs (`beta`, `loss_type`, `reference_free`, `max_prompt_length`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    trust_remote_code: bool = True
    # None -> backend default. Pass an explicit string ("eager", "sdpa",
    # "flash_attention_2") to override.
    attn_implementation: Optional[str] = None

    # Data
    train_file: Optional[str] = None  # Local JSONL with prompt/chosen/rejected
    dataset: Optional[str] = None  # HuggingFace dataset id
    dataset_split: str = "train"
    max_samples: Optional[int] = None
    validation_split: float = 0.05
    max_seq_length: int = 1024
    max_prompt_length: int = 512  # DPO truncates chosen+rejected after the prompt

    # DPO algorithm
    # `beta` is the KL-regularization strength against the reference model.
    # 0.1 is the canonical DPO value; 0.05 is gentler, 0.5 sharper.
    beta: float = 0.1
    # `loss_type` is forwarded to TRL. Common values:
    #   "sigmoid" - canonical DPO (Rafailov et al., 2023)
    #   "ipo"     - identity preference optimization (Azar et al., 2023)
    #   "hinge"   - max-margin hinge loss
    #   "kto_pair"- KTO loss on paired data
    # See trl.DPOTrainer for the full set.
    loss_type: str = "sigmoid"
    # When True, skip loading the reference model and use a frozen copy
    # of the policy at step 0 implicitly. Saves memory at the cost of some
    # quality; useful on Apple Silicon where you can't fit two copies.
    reference_free: bool = False
    # Label smoothing for the conservative-DPO variant (cDPO). 0.0 disables.
    label_smoothing: float = 0.0

    # QLoRA — same shape as SFTConfig so the public API + UI can re-use
    # the quantization controls without a separate code path.
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
    # PEFT additions (Track T5) — see SFTConfig for documentation.
    use_dora: bool = False
    use_rslora: bool = False
    init_lora_weights: str = "true"

    # Optimizer (Track T4) — see SFTConfig.optim for accepted values.
    optim: str = "adamw_torch"

    # Training
    output_dir: str = "models/dpo"
    num_epochs: int = 1
    batch_size: int = 1  # DPO doubles memory (chosen + rejected per row)
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-6  # DPO needs much smaller LR than SFT
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
        if self.target_modules is None:
            # Default for Qwen / Llama family. Same default as SFTConfig.
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


__all__ = ["DPOConfig"]
