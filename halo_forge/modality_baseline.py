"""
Deterministic modality baseline schema and drift-comparison helpers.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from halo_forge.modality_artifacts import resolve_resume_checkpoint


BASELINE_CONTRACT_VERSION = 1
BASELINE_GENERATOR_VERSION = "phase7b.v1"
DEFAULT_MODALITY_BASELINE_FILE = Path("tests/baselines/modality_runtime_baseline.v1.json")
DEFAULT_FIXTURE_DIR = Path("tests/fixtures/modality")
REQUIRED_MODALITIES = ("vlm", "audio", "reasoning", "agentic")
MODALITY_ENTRY_KEYS = (
    "cycles_executed",
    "total_train_steps_executed",
    "weights_updated",
    "final_update_reason",
    "failure_reason",
    "optimizer_steps",
    "skipped_batches_non_finite",
    "checkpoint_written",
    "final_model_written",
    "training_summary_written",
    "resume_contract_ok",
)


def compute_fixture_pack_fingerprint(fixture_dir: Path = DEFAULT_FIXTURE_DIR) -> str:
    """Compute stable fingerprint for modality fixture pack files."""
    if not fixture_dir.exists():
        raise ValueError(f"Fixture directory does not exist: {fixture_dir}")

    fixture_files = sorted(fixture_dir.glob("*.jsonl"))
    if not fixture_files:
        raise ValueError(f"No fixture JSONL files found in: {fixture_dir}")

    digest = hashlib.sha256()
    for fixture_path in fixture_files:
        digest.update(fixture_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(fixture_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_modality_entries_from_runs(
    run_payloads: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Build deterministic modality contract entries from raw smoke run payloads.

    run_payloads format:
        {
          "vlm": {"summary": {...}, "output_dir": Path(...)},
          ...
        }
    """
    entries: Dict[str, Dict[str, Any]] = {}
    for modality in REQUIRED_MODALITIES:
        payload = run_payloads.get(modality)
        if not isinstance(payload, dict):
            raise ValueError(f"Missing run payload for modality: {modality}")
        summary = payload.get("summary")
        output_dir = payload.get("output_dir")
        if not isinstance(summary, dict):
            raise ValueError(f"Invalid summary payload for modality: {modality}")
        if not output_dir:
            raise ValueError(f"Missing output_dir for modality: {modality}")
        entries[modality] = _build_modality_entry(
            modality=modality,
            summary=summary,
            output_dir=Path(str(output_dir)),
        )
    return entries


def build_baseline_payload(
    *,
    modality_entries: Dict[str, Dict[str, Any]],
    seed: int = 42,
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
    created_at: str | None = None,
) -> Dict[str, Any]:
    """Build canonical baseline payload from deterministic modality entries."""
    modalities: Dict[str, Dict[str, Any]] = {}
    for modality in REQUIRED_MODALITIES:
        if modality not in modality_entries:
            raise ValueError(f"Missing modality entry: {modality}")
        modalities[modality] = _normalize_modality_entry(modality_entries[modality])

    return {
        "contract_version": BASELINE_CONTRACT_VERSION,
        "generator_version": BASELINE_GENERATOR_VERSION,
        "seed": int(seed),
        "fixture_pack": compute_fixture_pack_fingerprint(fixture_dir),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "modalities": modalities,
    }


def load_baseline_file(path: Path) -> Dict[str, Any]:
    """Load baseline JSON file and return dict payload."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Baseline file must contain a JSON object: {path}")
    return payload


def write_baseline_file(path: Path, payload: Dict[str, Any]) -> None:
    """Write baseline payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_baseline_payload(payload: Dict[str, Any]) -> List[str]:
    """Validate baseline payload shape and required keys."""
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["payload is not a JSON object"]

    for key in (
        "contract_version",
        "generator_version",
        "seed",
        "fixture_pack",
        "created_at",
        "modalities",
    ):
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    modalities = payload.get("modalities")
    if not isinstance(modalities, dict):
        errors.append("modalities must be a JSON object")
        return errors

    modality_keys = tuple(sorted(modalities.keys()))
    if modality_keys != tuple(sorted(REQUIRED_MODALITIES)):
        errors.append(
            f"modalities keys must be exactly {REQUIRED_MODALITIES}, found {modality_keys}"
        )
        return errors

    for modality in REQUIRED_MODALITIES:
        entry = modalities.get(modality)
        if not isinstance(entry, dict):
            errors.append(f"modality entry must be an object: {modality}")
            continue
        for key in MODALITY_ENTRY_KEYS:
            if key not in entry:
                errors.append(f"missing modality key: {modality}.{key}")
    return errors


def compare_baseline_payloads(
    expected: Dict[str, Any],
    current: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compare baseline payloads and return structured drift entries.

    Drift entry format:
      {"modality": "...", "key": "...", "expected": <value>, "actual": <value>}
    """
    drifts: List[Dict[str, Any]] = []

    expected_errors = validate_baseline_payload(expected)
    current_errors = validate_baseline_payload(current)
    for error in expected_errors:
        drifts.append(
            {
                "modality": "__schema__",
                "key": "expected",
                "expected": "valid baseline schema",
                "actual": error,
            }
        )
    for error in current_errors:
        drifts.append(
            {
                "modality": "__schema__",
                "key": "current",
                "expected": "valid baseline schema",
                "actual": error,
            }
        )
    if drifts:
        return drifts

    expected_norm = _normalize_payload_for_compare(expected)
    current_norm = _normalize_payload_for_compare(current)

    for key in ("contract_version", "generator_version", "seed", "fixture_pack"):
        if expected_norm.get(key) != current_norm.get(key):
            drifts.append(
                {
                    "modality": "__global__",
                    "key": key,
                    "expected": expected_norm.get(key),
                    "actual": current_norm.get(key),
                }
            )

    expected_modalities = expected_norm["modalities"]
    current_modalities = current_norm["modalities"]
    for modality in REQUIRED_MODALITIES:
        expected_entry = expected_modalities.get(modality, {})
        current_entry = current_modalities.get(modality, {})
        for key in MODALITY_ENTRY_KEYS:
            if expected_entry.get(key) != current_entry.get(key):
                drifts.append(
                    {
                        "modality": modality,
                        "key": key,
                        "expected": expected_entry.get(key),
                        "actual": current_entry.get(key),
                    }
                )

    return drifts


def format_drift_lines(drifts: Iterable[Dict[str, Any]]) -> List[str]:
    """Format structured drift entries into parseable log lines."""
    lines: List[str] = []
    for drift in drifts:
        lines.append(
            "BASELINE_DRIFT "
            f"modality={drift.get('modality')} "
            f"key={drift.get('key')} "
            f"expected={json.dumps(drift.get('expected'), sort_keys=True)} "
            f"actual={json.dumps(drift.get('actual'), sort_keys=True)}"
        )
    return lines


def _normalize_payload_for_compare(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize payload and strip non-deterministic compare fields."""
    normalized = {
        "contract_version": int(payload.get("contract_version", 0) or 0),
        "generator_version": str(payload.get("generator_version", "")),
        "seed": int(payload.get("seed", 0) or 0),
        "fixture_pack": str(payload.get("fixture_pack", "")),
        "modalities": {},
    }
    modalities = payload.get("modalities") if isinstance(payload.get("modalities"), dict) else {}
    for modality in REQUIRED_MODALITIES:
        normalized["modalities"][modality] = _normalize_modality_entry(
            modalities.get(modality, {})
        )
    return normalized


def _build_modality_entry(
    *,
    modality: str,
    summary: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    cycles = summary.get("cycles") if isinstance(summary.get("cycles"), list) else []

    optimizer_steps = 0
    skipped_non_finite = 0
    for cycle in cycles:
        if not isinstance(cycle, dict):
            continue
        optimizer_steps += _to_int(cycle.get("optimizer_steps"))
        skipped_non_finite += _to_int(cycle.get("skipped_batches_non_finite"))

    if optimizer_steps == 0:
        optimizer_steps = _to_int(summary.get("optimizer_steps"))
    if skipped_non_finite == 0:
        skipped_non_finite = _to_int(summary.get("skipped_batches_non_finite"))

    entry = {
        "cycles_executed": _to_int(summary.get("cycles_executed")),
        "total_train_steps_executed": _to_int(summary.get("total_train_steps_executed")),
        "weights_updated": bool(summary.get("weights_updated", False)),
        "final_update_reason": str(summary.get("final_update_reason") or "no_cycles"),
        "failure_reason": (
            str(summary.get("failure_reason"))
            if summary.get("failure_reason") not in (None, "")
            else None
        ),
        "optimizer_steps": max(0, optimizer_steps),
        "skipped_batches_non_finite": max(0, skipped_non_finite),
        "checkpoint_written": (output_dir / "cycle_0" / "checkpoint_state.json").exists(),
        "final_model_written": (output_dir / "final_model").exists(),
        "training_summary_written": (output_dir / "training_summary.json").exists(),
        "resume_contract_ok": _resume_contract_ok(modality=modality, output_dir=output_dir),
    }
    return _normalize_modality_entry(entry)


def _resume_contract_ok(*, modality: str, output_dir: Path) -> bool:
    try:
        resolve_resume_checkpoint(
            output_dir=output_dir,
            resume_from_cycle=1,
            max_cycles=2,
            modality=modality,
        )
        return True
    except Exception:
        return False


def _normalize_modality_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(entry or {})
    return {
        "cycles_executed": _to_int(payload.get("cycles_executed")),
        "total_train_steps_executed": _to_int(payload.get("total_train_steps_executed")),
        "weights_updated": bool(payload.get("weights_updated", False)),
        "final_update_reason": str(payload.get("final_update_reason") or "no_cycles"),
        "failure_reason": (
            str(payload.get("failure_reason"))
            if payload.get("failure_reason") not in (None, "")
            else None
        ),
        "optimizer_steps": _to_int(payload.get("optimizer_steps")),
        "skipped_batches_non_finite": _to_int(payload.get("skipped_batches_non_finite")),
        "checkpoint_written": bool(payload.get("checkpoint_written", False)),
        "final_model_written": bool(payload.get("final_model_written", False)),
        "training_summary_written": bool(payload.get("training_summary_written", False)),
        "resume_contract_ok": bool(payload.get("resume_contract_ok", False)),
    }


def _to_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
