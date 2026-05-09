"""Shared training-sample eligibility helpers.

Assessment and RLVR loops should only train on verifier outputs that actually
passed the check and met the configured reward threshold. Keeping this rule in
one tiny module avoids each modality inventing a subtly different definition.
"""

from __future__ import annotations

from typing import Any


def result_success(result: Any) -> bool:
    """Return the verifier success flag from an object or dict."""
    if isinstance(result, dict):
        return bool(result.get("success", False))
    return bool(getattr(result, "success", False))


def result_reward(result: Any) -> float:
    """Return verifier reward as a float, falling back to zero on bad data."""
    try:
        if isinstance(result, dict):
            return float(result.get("reward", 0.0) or 0.0)
        return float(getattr(result, "reward", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _metadata(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        meta = result.get("metadata", {})
    else:
        meta = getattr(result, "metadata", {})
    return meta if isinstance(meta, dict) else {}


def is_training_eligible(
    result: Any,
    reward_threshold: float,
    *,
    allow_compile_only: bool = False,
) -> bool:
    """A sample may train only if verification passed and reward is high enough."""
    if not result_success(result):
        return False
    if _metadata(result).get("stage") == "compile" and not allow_compile_only:
        return False
    return result_reward(result) >= float(reward_threshold)


__all__ = ["is_training_eligible", "result_reward", "result_success"]
