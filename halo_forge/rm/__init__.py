"""Reward Model trainer (Track T3).

Trains a Bradley-Terry-style scalar reward model from preference pairs
``(prompt, chosen, rejected)``. The trained RM becomes a learned
verifier — plug it into RAFT or GRPO as the reward signal for tasks
where programmatic verification doesn't exist (creative writing,
summarization, dialog, anything you'd otherwise score with an
LLM-judge but want fast + deterministic + GPU-resident).

Closes the RLHF loop: the RM is what the "RM" in RLHF is. Without it
halo-forge could only reward via:
  - programmatic verifiers (V1 ecosystem) — works for code/structure
  - LLM-as-judge (V2) — works anywhere but is slow and stochastic
  - heuristics (D3) — coarse

A trained RM gives every other modality the verifier-grounded reward
signal that code modalities have for free.

Backend dispatch:
  rocm/cuda/mps/cpu  →  TRL's RewardTrainer (PyTorch)
  mlx                →  not yet implemented; raises with a typed
                        pointer. Native MLX RM is roadmap follow-up.

Pairs with:
  - D1 synthesize ... --kind preference  (produces the training data)
  - GRPO --verifier <rm-as-verifier>     (consumes the trained RM)
  - V2 LLM-judge                          (for the synthetic preference labels)
"""

from halo_forge.rm.config import RMConfig
from halo_forge.rm._dispatch import get_rm_trainer

__all__ = ["RMConfig", "get_rm_trainer"]
