"""GRPO configuration dataclass.

Mirrors `DPOConfig` for the fields they share (model, dataset, LoRA,
QLoRA, optimizer, PEFT additions) and adds GRPO-specific knobs:
``num_generations`` (group size), ``beta`` (KL penalty strength),
``reward_threshold``, ``verifier_name``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None

    # Data — prompts only; GRPO generates the completions.
    train_file: Optional[str] = None  # Local prompts JSONL
    dataset: Optional[str] = None     # HF dataset id
    max_samples: Optional[int] = None
    max_prompt_length: int = 512
    max_completion_length: int = 512

    # GRPO algorithm
    # `num_generations` is the group size — how many completions to sample
    # per prompt. Larger groups give a more reliable advantage estimate
    # but cost more tokens per cycle. 4-8 is the typical range.
    num_generations: int = 4
    # `beta` is the KL-regularization strength against the reference
    # model. 0.04 is the value DeepSeek-R1 published; lower means
    # the policy can drift further from the base; 0 disables KL.
    beta: float = 0.04
    # `reference_free` skips loading a separate reference model — the
    # KL term is computed against a snapshot of the policy at step 0.
    # Saves memory at the cost of some quality. v1 only supports this
    # mode on MLX (full reference-model GRPO on MLX is a follow-up).
    reference_free: bool = False
    # `epsilon` clips the policy ratio (PPO-style) so a single bad
    # update can't blow out the policy. 0.2 is the standard value.
    epsilon: float = 0.2
    # `temperature` for the rollout-stage sampling — higher = more
    # diversity within a group, sharper advantage signal.
    temperature: float = 0.9
    # Whether to scale rewards by 1/std(group). True is the canonical
    # GRPO; False makes it more like RLOO (uses group_mean baseline only).
    scale_rewards: bool = True

    # Verifier (Track V1) — name resolved through the plugin registry.
    # Defaults to "execution" (code execution verifier); pass any
    # registered verifier's short name to swap.
    verifier_name: str = "execution"
    # Threshold below which a completion's reward maps to 0 advantage.
    # Set to 0 to keep all rewards in the group.
    reward_threshold: float = 0.0

    # QLoRA
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

    optim: str = "adamw_torch"

    # Training
    output_dir: str = "models/grpo"
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-6  # GRPO needs an even smaller LR than DPO
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = DEFAULT_TRAINING_SEED

    save_steps: int = 100
    save_total_limit: int = 3
    eval_steps: int = 50
    logging_steps: int = 10

    # Rollout integration (Track I6).
    # When set to "torch", uses HF model.generate via TRL's default path.
    # When "vllm", we wire VLLMRolloutGenerator. When "mlx", the
    # MLX trainer uses mlx_lm.generate. "auto" lets the trainer pick.
    rollout_engine: str = "auto"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


__all__ = ["GRPOConfig"]
