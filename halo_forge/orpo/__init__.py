"""Odds-Ratio Preference Optimization (ORPO) trainer.

Track T17b (added 2026-05-08).

ORPO (Hong, Lee, & Thorne, 2024) does preference optimization in a
single pass *without a reference model* — it combines an NLL term on
the chosen response with a log-odds term that pushes the chosen
likelihood above the rejected likelihood. Cheaper than DPO (no
ref-model copy in memory), simpler than RLHF (no separate RM /
PPO), and competitive on chat refinement benchmarks.

Backend dispatch:
    rocm_gfx1151 / cuda / mps / cpu  ->  halo_forge.orpo.trainer.ORPOTrainer
    mlx                              ->  not yet implemented; raises with a
                                          pointer to the MLX roadmap.

Dataset format:
    JSONL or HuggingFace dataset with `prompt`, `chosen`, `rejected`
    columns (same shape as DPO). We reuse `halo_forge.dpo.datasets.
    load_preference_dataset` directly — there's no point duplicating
    pair loading just because the loss math differs.

Public surface:
    - ORPOConfig — config dataclass.
    - get_orpo_trainer(config) — backend-aware factory.
"""

from halo_forge.orpo.config import ORPOConfig
from halo_forge.orpo._dispatch import get_orpo_trainer

__all__ = [
    "ORPOConfig",
    "get_orpo_trainer",
]
