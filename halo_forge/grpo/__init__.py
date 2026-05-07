"""Group Relative Policy Optimization (GRPO) trainer.

Track T2 / phase Q1.

GRPO is the verifier-grounded policy gradient that DeepSeek-R1 popularized
and that has become the default RLVR algorithm in the open-recipe
literature (Tülu 3, Open-R1, Qwen-2.5-Math). Sits next to RAFT
(rejection-sampling) in the RLVR family but is more sample-efficient
because it uses *all* sampled completions (weighted by group-relative
advantage), not just the top-K.

Algorithm:
    For each prompt, sample N completions (a "group").
    Score each completion via the verifier(s).
    advantage_i = (reward_i - group_mean) / group_std
    loss = -E[advantage * log π(completion|prompt)] + β * KL(π || π_ref)

Backend dispatch:
    rocm_gfx1151 / cuda / mps / cpu  →  PyTorch path via trl.GRPOTrainer
    mlx                              →  MLX-native (reference-free in v1)

Both paths reuse:
  - The pluggable rollout engine (I6) — vLLM on CUDA/ROCm, mlx_lm on
    Apple Silicon, HF generate as the universal fallback.
  - The verifier plugin registry (V1) — programmatic verifiers, schema
    verifiers, LLM-as-judge (V2) all just register and the trainer
    picks them up by name.
"""

from halo_forge.grpo.config import GRPOConfig
from halo_forge.grpo._dispatch import get_grpo_trainer

__all__ = ["GRPOConfig", "get_grpo_trainer"]
