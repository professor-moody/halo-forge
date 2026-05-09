"""Direct Preference Optimization (DPO) trainer public surface."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DPOConfig",
    "get_dpo_trainer",
    "load_preference_dataset",
]


def __getattr__(name: str) -> Any:
    if name == "DPOConfig":
        from halo_forge.dpo.config import DPOConfig

        return DPOConfig
    if name == "get_dpo_trainer":
        from halo_forge.dpo._dispatch import get_dpo_trainer

        return get_dpo_trainer
    if name == "load_preference_dataset":
        from halo_forge.dpo.datasets import load_preference_dataset

        return load_preference_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
