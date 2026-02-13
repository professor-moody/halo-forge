"""
Canonical training telemetry contracts shared across modality trainers.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def normalize_update_metrics(
    metrics: Optional[Dict[str, Any]],
    *,
    default_reason: str = "no_update",
) -> Dict[str, Any]:
    """Normalize update telemetry fields to a stable shape."""
    payload = dict(metrics or {})
    steps_raw = payload.get("train_steps_executed", 0)
    try:
        steps = int(steps_raw)
    except (TypeError, ValueError):
        steps = 0

    loss_raw = payload.get("train_loss")
    if isinstance(loss_raw, (int, float)):
        train_loss: Optional[float] = float(loss_raw)
    else:
        train_loss = None

    weights_updated = bool(payload.get("weights_updated", False))
    update_reason = payload.get("update_reason")
    if not update_reason:
        update_reason = "updated" if weights_updated else default_reason

    return {
        "train_steps_executed": max(0, steps),
        "train_loss": train_loss,
        "weights_updated": weights_updated,
        "update_reason": str(update_reason),
    }


def build_cycle_summary(
    *,
    cycle: int,
    learning_rate: float,
    samples_seen: int,
    samples_kept: int,
    cycle_duration_seconds: float,
    update_metrics: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build canonical per-cycle telemetry payload."""
    summary = {
        "cycle": int(cycle),
        "learning_rate": float(learning_rate),
        "samples_seen": int(samples_seen),
        "samples_kept": int(samples_kept),
        "cycle_duration_seconds": float(max(0.0, cycle_duration_seconds)),
    }
    summary.update(normalize_update_metrics(update_metrics))
    if extra:
        summary.update(extra)
    return summary


def build_training_summary(
    *,
    modality: str,
    model_name: str,
    total_cycles_planned: int,
    cycles: Iterable[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build canonical final training summary payload."""
    cycle_list = [dict(cycle) for cycle in cycles]
    total_steps = sum(
        int(cycle.get("train_steps_executed", 0))
        for cycle in cycle_list
    )
    weights_updated = any(bool(cycle.get("weights_updated", False)) for cycle in cycle_list)
    final_cycle = cycle_list[-1] if cycle_list else {}
    summary = {
        "contract_version": 1,
        "modality": modality,
        "model_name": model_name,
        "total_cycles_planned": int(total_cycles_planned),
        "cycles_executed": len(cycle_list),
        "cycles": cycle_list,
        "total_train_steps_executed": total_steps,
        "weights_updated": weights_updated,
        "final_train_loss": final_cycle.get("train_loss"),
        "final_update_reason": final_cycle.get("update_reason", "no_cycles"),
    }
    if extra:
        summary.update(extra)
    return summary


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON payload atomically to avoid partial writes."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
    ) as tmp_file:
        json.dump(payload, tmp_file, indent=2)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_path = Path(tmp_file.name)

    os.replace(tmp_path, path)
