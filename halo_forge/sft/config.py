"""SFT configuration dataclass.

Lives outside `trainer.py` so it can be imported on hosts without torch
(MLX-only Macs, lightweight CI workers). The PyTorch trainer at
`halo_forge.sft.trainer` and the MLX trainer at `halo_forge.sft.mlx_trainer`
both consume this same `SFTConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-7B"
    trust_remote_code: bool = True
    # None -> backend default (eager on ROCm/CPU, sdpa on CUDA/MPS).
    # Pass an explicit string to override.
    attn_implementation: Optional[str] = None

    # Data - supports both local files and HuggingFace datasets
    train_file: Optional[str] = None  # Local JSONL file
    dataset: Optional[str] = None  # HuggingFace dataset ID or short name
    max_samples: Optional[int] = None  # Limit number of samples
    validation_split: float = 0.05
    max_seq_length: int = 2048

    # QLoRA (4-bit is slower on Strix Halo - use BF16 by default)
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    # Backends without bitsandbytes (Apple Silicon MPS, CPU) cannot honor
    # load_in_4bit. By default the trainer warns and falls back to unquantized;
    # set this False to fail loudly instead.
    allow_quantization_fallback: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    # Training
    output_dir: str = "models/sft"
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3

    # Optimization
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = DEFAULT_TRAINING_SEED

    # Saving
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 250

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.001

    def __post_init__(self):
        if self.target_modules is None:
            # Default for Qwen models
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        """Load config from YAML file."""
        import yaml  # lazy: yaml is a hard dep but keep config.py import-clean

        with open(path) as f:
            data = yaml.safe_load(f)

        # Flatten nested config
        flat = {}
        for section in ['model', 'data', 'lora', 'qlora', 'training']:
            if section in data:
                flat.update(data[section])

        # Map config keys
        key_map = {
            'name': 'model_name',
            'train_file': 'train_file',
            'per_device_train_batch_size': 'batch_size',
            'num_train_epochs': 'num_epochs',
            'r': 'lora_r',
            'alpha': 'lora_alpha',
            'dropout': 'lora_dropout',
        }

        mapped = {}
        for k, v in flat.items():
            mapped_key = key_map.get(k, k)
            if hasattr(cls, '__dataclass_fields__') and mapped_key in cls.__dataclass_fields__:
                mapped[mapped_key] = v

        return cls(**mapped)


__all__ = ["SFTConfig"]
