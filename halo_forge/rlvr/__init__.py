"""RLVR (Reinforcement Learning from Verifier Rewards) training module."""

from halo_forge.rlvr.raft_trainer import RAFTTrainer, RAFTConfig
from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult
from halo_forge.rlvr.curriculum import CurriculumScheduler, CurriculumConfig, CurriculumStrategy
from halo_forge.rlvr.reward_shaping import RewardShaper, RewardShapingConfig, RewardShapingStrategy

__all__ = [
    # Trainers
    "RAFTTrainer", "RAFTConfig",
    # Verifiers
    "Verifier", "VerifyResult",
    # Curriculum
    "CurriculumScheduler", "CurriculumConfig", "CurriculumStrategy",
    # Reward Shaping
    "RewardShaper", "RewardShapingConfig", "RewardShapingStrategy",
]

