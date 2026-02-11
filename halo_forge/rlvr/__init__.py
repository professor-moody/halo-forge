"""RLVR (Reinforcement Learning from Verifier Rewards) training module."""

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult
from halo_forge.rlvr.curriculum import CurriculumScheduler, CurriculumConfig, CurriculumStrategy
from halo_forge.rlvr.reward_shaping import RewardShaper, RewardShapingConfig, RewardShapingStrategy

RAFTTrainer = None
RAFTConfig = None

try:
    from halo_forge.rlvr.raft_trainer import RAFTTrainer, RAFTConfig
except ModuleNotFoundError as e:
    # Allow lightweight imports (e.g. verifiers, benchmark utils) when
    # training dependencies like torch/transformers are unavailable.
    if e.name not in {"torch", "transformers", "peft", "datasets"}:
        raise

__all__ = [
    # Verifiers
    "Verifier", "VerifyResult",
    # Curriculum
    "CurriculumScheduler", "CurriculumConfig", "CurriculumStrategy",
    # Reward Shaping
    "RewardShaper", "RewardShapingConfig", "RewardShapingStrategy",
]

if RAFTTrainer is not None and RAFTConfig is not None:
    __all__.extend(["RAFTTrainer", "RAFTConfig"])
