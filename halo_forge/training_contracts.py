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

    initial_loss_raw = payload.get("initial_train_loss")
    if isinstance(initial_loss_raw, (int, float)):
        initial_train_loss: Optional[float] = float(initial_loss_raw)
    else:
        initial_train_loss = None

    weights_updated = bool(payload.get("weights_updated", False))
    update_reason = payload.get("update_reason")
    if not update_reason:
        update_reason = "updated" if weights_updated else default_reason

    return {
        "train_steps_executed": max(0, steps),
        "train_loss": train_loss,
        "initial_train_loss": initial_train_loss,
        "weights_updated": weights_updated,
        "update_reason": str(update_reason),
        "optimizer_steps": max(0, int(payload.get("optimizer_steps", steps) or 0)),
        "skipped_batches_non_finite": max(
            0,
            int(payload.get("skipped_batches_non_finite", 0) or 0),
        ),
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
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
    resume_from_cycle: int = 0,
    resumed_from_checkpoint: Optional[Dict[str, Any]] = None,
    base_model_name: Optional[str] = None,
    active_model_name: Optional[str] = None,
    failure_reason: Optional[str] = None,
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
        "run_id": run_id or "",
        "seed": seed,
        "resume_from_cycle": int(max(0, resume_from_cycle)),
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "base_model_name": base_model_name or model_name,
        "active_model_name": active_model_name or model_name,
        "failure_reason": (
            failure_reason
            if failure_reason is not None
            else (final_cycle.get("update_reason", "no_cycles") if not weights_updated else None)
        ),
    }
    if extra:
        summary.update(extra)
    return summary


def build_effectiveness_evaluation(
    *,
    metric_name: Optional[str] = None,
    baseline_value: Optional[float] = None,
    final_value: Optional[float] = None,
    higher_is_better: bool = True,
    tolerance: float = 0.0,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a normalized evaluation block for the training effectiveness contract."""
    normalized_baseline = _coerce_optional_float(baseline_value)
    normalized_final = _coerce_optional_float(final_value)

    if status not in {"available", "not_available", "error"}:
        if metric_name and normalized_final is not None:
            status = "available"
        else:
            status = "not_available"

    delta: Optional[float] = None
    regression = False
    if normalized_baseline is not None and normalized_final is not None:
        delta = normalized_final - normalized_baseline
        if higher_is_better:
            regression = normalized_final + float(tolerance) < normalized_baseline
        else:
            regression = normalized_final - float(tolerance) > normalized_baseline

    return {
        "metric_name": metric_name or "",
        "baseline_value": normalized_baseline,
        "final_value": normalized_final,
        "delta": delta,
        "higher_is_better": bool(higher_is_better),
        "tolerance": float(max(0.0, tolerance)),
        "status": status,
        "regressed": regression,
    }


def build_effectiveness_contract(
    *,
    data_yield: Dict[str, Any],
    update_quality: Dict[str, Any],
    checkpoint_quality: Dict[str, Any],
    evaluation: Optional[Dict[str, Any]] = None,
    failure_reason: Optional[str] = None,
    evaluation_required: bool = False,
) -> Dict[str, Any]:
    """Build the additive effectiveness block with centralized verdict logic."""
    normalized_data = {
        "samples_seen": max(0, _coerce_int(data_yield.get("samples_seen"))),
        "samples_kept": max(0, _coerce_int(data_yield.get("samples_kept"))),
        "keep_rate": float(max(0.0, data_yield.get("keep_rate", 0.0) or 0.0)),
        "minimum_samples_kept": max(0, _coerce_int(data_yield.get("minimum_samples_kept", 0))),
    }
    if normalized_data["samples_seen"] > 0 and not data_yield.get("keep_rate"):
        normalized_data["keep_rate"] = (
            normalized_data["samples_kept"] / normalized_data["samples_seen"]
        )
    normalized_data["min_samples_met"] = (
        normalized_data["samples_kept"] >= normalized_data["minimum_samples_kept"]
    )

    normalized_update = {
        "train_steps_executed": max(0, _coerce_int(update_quality.get("train_steps_executed"))),
        "optimizer_steps": max(0, _coerce_int(update_quality.get("optimizer_steps"))),
        "weights_updated": bool(update_quality.get("weights_updated", False)),
        "initial_train_loss": _coerce_optional_float(update_quality.get("initial_train_loss")),
        "final_train_loss": _coerce_optional_float(update_quality.get("final_train_loss")),
        "skipped_batches_non_finite": max(
            0,
            _coerce_int(update_quality.get("skipped_batches_non_finite")),
        ),
        "update_reason": str(update_quality.get("update_reason") or "no_update"),
        "minimum_optimizer_steps": max(
            0,
            _coerce_int(update_quality.get("minimum_optimizer_steps", 0)),
        ),
    }
    if (
        normalized_update["initial_train_loss"] is not None
        and normalized_update["final_train_loss"] is not None
    ):
        normalized_update["loss_delta"] = (
            normalized_update["final_train_loss"] - normalized_update["initial_train_loss"]
        )
    else:
        normalized_update["loss_delta"] = None

    normalized_checkpoint = {
        "checkpoint_written": bool(checkpoint_quality.get("checkpoint_written", False)),
        "final_model_written": bool(checkpoint_quality.get("final_model_written", False)),
        "training_summary_written": bool(
            checkpoint_quality.get("training_summary_written", False)
        ),
        "resume_contract_ok": bool(checkpoint_quality.get("resume_contract_ok", True)),
    }

    normalized_evaluation = _normalize_effectiveness_evaluation(evaluation)

    reasons = []
    if not normalized_update["weights_updated"]:
        reasons.append("weights_not_updated")
    if normalized_update["train_steps_executed"] <= 0:
        reasons.append("no_train_steps")
    if normalized_update["optimizer_steps"] < normalized_update["minimum_optimizer_steps"]:
        reasons.append("optimizer_steps_below_minimum")
    if not normalized_data["min_samples_met"]:
        reasons.append("samples_kept_below_minimum")
    if not normalized_checkpoint["final_model_written"]:
        reasons.append("final_model_missing")
    if not normalized_checkpoint["training_summary_written"]:
        reasons.append("training_summary_missing")
    if not normalized_checkpoint["resume_contract_ok"]:
        reasons.append("resume_contract_invalid")
    if evaluation_required and normalized_evaluation["status"] != "available":
        reasons.append("evaluation_unavailable")
    if normalized_evaluation["regressed"]:
        reasons.append("evaluation_regressed")
    if failure_reason and failure_reason not in {None, "", "updated"}:
        reasons.append(str(failure_reason))

    if reasons:
        verdict = "fail"
    elif normalized_evaluation["status"] != "available":
        verdict = "warn"
        reasons.append("evaluation_not_available")
    else:
        verdict = "pass"

    return {
        "data_yield": normalized_data,
        "update_quality": normalized_update,
        "checkpoint_quality": normalized_checkpoint,
        "evaluation": normalized_evaluation,
        "verdict": verdict,
        "reasons": reasons,
    }


def attach_effectiveness_contract(
    summary: Dict[str, Any],
    *,
    minimum_samples_kept: int = 1,
    minimum_optimizer_steps: int = 1,
    evaluation: Optional[Dict[str, Any]] = None,
    evaluation_required: bool = False,
    checkpoint_written: Optional[bool] = None,
    final_model_path: Optional[str] = None,
    training_summary_path: Optional[Path] = None,
    resume_contract_ok: Optional[bool] = None,
) -> Dict[str, Any]:
    """Attach the shared effectiveness block to an existing training summary."""
    cycle_list = [dict(cycle) for cycle in summary.get("cycles", [])]
    samples_seen = sum(max(0, _coerce_int(cycle.get("samples_seen"))) for cycle in cycle_list)
    samples_kept = sum(max(0, _coerce_int(cycle.get("samples_kept"))) for cycle in cycle_list)
    optimizer_steps = sum(
        max(0, _coerce_int(cycle.get("optimizer_steps", cycle.get("train_steps_executed", 0))))
        for cycle in cycle_list
    )
    skipped_batches_non_finite = sum(
        max(0, _coerce_int(cycle.get("skipped_batches_non_finite")))
        for cycle in cycle_list
    )
    initial_train_loss = None
    for cycle in cycle_list:
        initial_train_loss = _coerce_optional_float(cycle.get("initial_train_loss"))
        if initial_train_loss is not None:
            break
        initial_train_loss = _coerce_optional_float(cycle.get("train_loss"))
        if initial_train_loss is not None:
            break

    final_train_loss = _coerce_optional_float(summary.get("final_train_loss"))
    if final_train_loss is None:
        for cycle in reversed(cycle_list):
            final_train_loss = _coerce_optional_float(cycle.get("train_loss"))
            if final_train_loss is not None:
                break

    resolved_final_model_path = final_model_path or summary.get("final_model_path")
    if final_model_path is not None:
        summary["final_model_path"] = final_model_path

    effective_resume_contract_ok = (
        resume_contract_ok
        if resume_contract_ok is not None
        else _derive_resume_contract_ok(summary)
    )

    summary["effectiveness"] = build_effectiveness_contract(
        data_yield={
            "samples_seen": samples_seen,
            "samples_kept": samples_kept,
            "minimum_samples_kept": minimum_samples_kept,
        },
        update_quality={
            "train_steps_executed": summary.get("total_train_steps_executed", 0),
            "optimizer_steps": optimizer_steps,
            "weights_updated": summary.get("weights_updated", False),
            "initial_train_loss": initial_train_loss,
            "final_train_loss": final_train_loss,
            "skipped_batches_non_finite": skipped_batches_non_finite,
            "update_reason": summary.get("final_update_reason", "no_updates"),
            "minimum_optimizer_steps": minimum_optimizer_steps,
        },
        checkpoint_quality={
            "checkpoint_written": (
                bool(cycle_list)
                if checkpoint_written is None
                else bool(checkpoint_written)
            ),
            "final_model_written": bool(resolved_final_model_path),
            "training_summary_written": training_summary_path is not None,
            "resume_contract_ok": effective_resume_contract_ok,
        },
        evaluation=evaluation,
        failure_reason=summary.get("failure_reason"),
        evaluation_required=evaluation_required,
    )
    return summary


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _derive_resume_contract_ok(summary: Dict[str, Any]) -> bool:
    resume_from_cycle = max(0, _coerce_int(summary.get("resume_from_cycle")))
    if resume_from_cycle == 0:
        return True
    resumed_from_checkpoint = summary.get("resumed_from_checkpoint")
    return isinstance(resumed_from_checkpoint, dict) and bool(
        resumed_from_checkpoint.get("model_dir")
    )


def _normalize_effectiveness_evaluation(
    evaluation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = dict(evaluation or {})
    return build_effectiveness_evaluation(
        metric_name=payload.get("metric_name"),
        baseline_value=payload.get("baseline_value"),
        final_value=payload.get("final_value"),
        higher_is_better=payload.get("higher_is_better", True),
        tolerance=payload.get("tolerance", 0.0),
        status=payload.get("status"),
    )


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
