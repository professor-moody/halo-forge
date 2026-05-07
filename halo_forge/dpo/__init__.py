"""Direct Preference Optimization (DPO) trainer.

Phase Q1 / Track T1.

Halo-forge wraps `trl.DPOTrainer` for the PyTorch path so we get the same
algorithm coverage (sigmoid, IPO, hinge, KTO-pair, RPO, etc.) and tested
loss math as TRL ships, while keeping our config / output / recovery /
public-API surface consistent with SFT and RAFT.

Backend dispatch:
    rocm_gfx1151 / cuda / mps / cpu  ->  halo_forge.dpo.trainer.DPOTrainer
    mlx                              ->  not yet implemented; raises with a
                                          pointer to mlx-lm-lora.

Dataset format:
    JSONL or HuggingFace dataset where each row carries `prompt`,
    `chosen`, and `rejected` text fields (the TRL-canonical preference
    layout). See `halo_forge.dpo.datasets.load_preference_dataset`.

Public surface:
    - DPOConfig — config dataclass.
    - get_dpo_trainer(config) — backend-aware factory.
    - load_preference_dataset(...) — dataset loader.
"""

from halo_forge.dpo.config import DPOConfig
from halo_forge.dpo._dispatch import get_dpo_trainer
from halo_forge.dpo.datasets import load_preference_dataset

__all__ = [
    "DPOConfig",
    "get_dpo_trainer",
    "load_preference_dataset",
]
