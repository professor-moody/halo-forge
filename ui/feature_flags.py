"""
Feature-flag helpers for optional UI surfaces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class UiFeatureFlags:
    enable_inference_page: bool
    enable_benchmark_advanced_page: bool
    enable_research_hub_page: bool


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def get_ui_feature_flags() -> UiFeatureFlags:
    """Load UI feature flags from environment variables."""
    return UiFeatureFlags(
        enable_inference_page=_env_bool("HALO_UI_ENABLE_INFERENCE_PAGE", default=False),
        enable_benchmark_advanced_page=_env_bool(
            "HALO_UI_ENABLE_BENCHMARK_ADVANCED_PAGE",
            default=False,
        ),
        enable_research_hub_page=_env_bool(
            "HALO_UI_ENABLE_RESEARCH_HUB_PAGE",
            default=False,
        ),
    )
