"""
Shared artifact persistence helpers for modality RAFT trainers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from halo_forge.training_contracts import normalize_update_metrics, write_json_atomic


@dataclass(frozen=True)
class ResumeCheckpoint:
    """Resolved checkpoint metadata for resume flows."""

    cycle: int
    cycle_dir: Path
    model_dir: Path
    state: Dict[str, Any]


def persist_cycle_artifacts(
    *,
    output_dir: Path,
    modality: str,
    model_name: str,
    cycle: int,
    update_metrics: Optional[Dict[str, Any]],
    model: Any = None,
    tokenizer: Any = None,
    processor: Any = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persist canonical per-cycle training artifacts.
    """
    cycle_dir = output_dir / f"cycle_{cycle}"
    model_dir = cycle_dir / "model"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    saved_components = _save_hf_artifacts(
        model_dir=model_dir,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
    )

    state: Dict[str, Any] = {
        "contract_version": 1,
        "modality": modality,
        "model_name": model_name,
        "cycle": int(cycle),
        "timestamp": datetime.now().isoformat(),
        "model_dir": str(model_dir),
        "update_metrics": normalize_update_metrics(update_metrics),
        "saved_components": saved_components,
    }
    if extra_state:
        state.update(extra_state)

    write_json_atomic(cycle_dir / "checkpoint_state.json", state)
    write_json_atomic(
        output_dir / "latest_checkpoint.json",
        {
            "contract_version": 1,
            "modality": modality,
            "cycle": int(cycle),
            "timestamp": state["timestamp"],
            "cycle_dir": str(cycle_dir),
            "model_dir": str(model_dir),
        },
    )
    return state


def persist_final_artifacts(
    *,
    output_dir: Path,
    modality: str,
    model_name: str,
    model: Any = None,
    tokenizer: Any = None,
    processor: Any = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persist canonical final model artifacts under ``final_model/``.
    """
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    saved_components = _save_hf_artifacts(
        model_dir=final_dir,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
    )
    state: Dict[str, Any] = {
        "contract_version": 1,
        "modality": modality,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "final_model_dir": str(final_dir),
        "saved_components": saved_components,
    }
    if extra_state:
        state.update(extra_state)
    write_json_atomic(output_dir / "final_model_state.json", state)
    return state


def resolve_resume_checkpoint(
    *,
    output_dir: Path,
    resume_from_cycle: int,
    max_cycles: int,
    modality: Optional[str] = None,
    min_contract_version: int = 1,
    required_model_artifacts: tuple[str, ...] = (),
) -> ResumeCheckpoint:
    """
    Resolve and validate checkpoint artifacts for ``resume_from_cycle``.
    """
    if resume_from_cycle < 0 or resume_from_cycle >= max_cycles:
        raise ValueError(f"resume_from_cycle must be in [0, {max_cycles - 1}]")
    if resume_from_cycle == 0:
        raise ValueError("resume_from_cycle=0 does not require checkpoint resolution")

    checkpoint_cycle = resume_from_cycle - 1
    cycle_dir = output_dir / f"cycle_{checkpoint_cycle}"
    state_path = cycle_dir / "checkpoint_state.json"
    if not state_path.exists():
        raise ValueError(
            f"resume_from_cycle={resume_from_cycle} requires checkpoint state {state_path}"
        )

    import json

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid checkpoint state format in {state_path}")

    contract_version = int(state.get("contract_version", 0) or 0)
    if contract_version < min_contract_version:
        raise ValueError(
            f"checkpoint state contract_version={contract_version} is below required "
            f"{min_contract_version} in {state_path}"
        )

    state_modality = str(state.get("modality") or "")
    if modality and state_modality != modality:
        raise ValueError(
            f"resume_from_cycle={resume_from_cycle} expected modality={modality} "
            f"but found modality={state_modality} in {state_path}"
        )

    model_dir_text = state.get("model_dir")
    if not model_dir_text:
        raise ValueError(
            f"checkpoint state missing model_dir in {state_path}"
        )
    model_dir = Path(model_dir_text)
    if not model_dir.exists():
        raise ValueError(
            f"resume_from_cycle={resume_from_cycle} requires model directory {model_dir}"
        )
    for artifact_name in required_model_artifacts:
        candidate = model_dir / artifact_name
        if not candidate.exists():
            raise ValueError(
                f"resume_from_cycle={resume_from_cycle} requires model artifact {candidate}"
            )

    return ResumeCheckpoint(
        cycle=checkpoint_cycle,
        cycle_dir=cycle_dir,
        model_dir=model_dir,
        state=state,
    )


def _save_hf_artifacts(
    *,
    model_dir: Path,
    model: Any = None,
    tokenizer: Any = None,
    processor: Any = None,
) -> Dict[str, bool]:
    """
    Save HuggingFace-compatible artifacts using best-effort behavior.
    """
    saved: Dict[str, bool] = {
        "model": False,
        "tokenizer": False,
        "processor": False,
    }

    if model is not None:
        saved["model"] = _safe_save_pretrained(model, model_dir)
        if not saved["model"] and hasattr(model, "state_dict"):
            try:
                import torch
                torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
                saved["model"] = True
            except Exception:
                saved["model"] = False

    if tokenizer is not None:
        saved["tokenizer"] = _safe_save_pretrained(tokenizer, model_dir)

    if processor is not None and processor is not tokenizer:
        saved["processor"] = _safe_save_pretrained(processor, model_dir)

    return saved


def _safe_save_pretrained(component: Any, target_dir: Path) -> bool:
    if component is None or not hasattr(component, "save_pretrained"):
        return False
    try:
        component.save_pretrained(str(target_dir))
        return True
    except Exception:
        return False
