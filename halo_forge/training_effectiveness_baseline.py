"""
Deterministic baseline helpers for training effectiveness regression gates.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


BASELINE_CONTRACT_VERSION = 1
BASELINE_GENERATOR_VERSION = "phase29a.v1"
DEFAULT_BASELINE_DIR = Path("tests/baselines/training_effectiveness")


def compute_fixture_fingerprint(paths: Iterable[Path] | Path) -> str:
    """Compute a stable fingerprint for one or more deterministic fixture files."""
    candidate_paths = list(_iter_fixture_files(paths))
    if not candidate_paths:
        raise ValueError("No fixture files provided for fingerprinting")

    digest = hashlib.sha256()
    for fixture_path in candidate_paths:
        digest.update(str(fixture_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(fixture_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_baseline_payload(
    *,
    path_id: str,
    entries: Mapping[str, Mapping[str, Any]],
    fixture_paths: Iterable[Path] | Path,
    seed: int = 42,
    created_at: str | None = None,
) -> Dict[str, Any]:
    """Build a deterministic path-oriented baseline payload."""
    normalized_entries = {
        entry_id: _normalize_entry(dict(entry))
        for entry_id, entry in sorted(entries.items())
    }
    return {
        "contract_version": BASELINE_CONTRACT_VERSION,
        "generator_version": BASELINE_GENERATOR_VERSION,
        "path_id": path_id,
        "seed": int(seed),
        "fixture_fingerprint": compute_fixture_fingerprint(fixture_paths),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "entries": normalized_entries,
    }


def validate_baseline_payload(payload: Mapping[str, Any]) -> List[str]:
    """Validate baseline payload shape."""
    errors: List[str] = []
    if not isinstance(payload, Mapping):
        return ["payload must be a JSON object"]

    for key in (
        "contract_version",
        "generator_version",
        "path_id",
        "seed",
        "fixture_fingerprint",
        "created_at",
        "entries",
    ):
        if key not in payload:
            errors.append(f"missing top-level key: {key}")

    entries = payload.get("entries")
    if not isinstance(entries, Mapping) or not entries:
        errors.append("entries must be a non-empty object")
        return errors

    for entry_id, entry in entries.items():
        if not isinstance(entry, Mapping):
            errors.append(f"entry must be an object: {entry_id}")
            continue
        for key in (
            "metric_name",
            "baseline_value",
            "higher_is_better",
            "tolerance",
            "minimum_samples_kept",
            "minimum_optimizer_steps",
            "expected_verdict",
        ):
            if key not in entry:
                errors.append(f"missing entry key: {entry_id}.{key}")
    return errors


def build_actual_entry_from_effectiveness(effectiveness: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the baseline-comparable fields from an effectiveness contract."""
    evaluation = dict(effectiveness.get("evaluation") or {})
    data_yield = dict(effectiveness.get("data_yield") or {})
    update_quality = dict(effectiveness.get("update_quality") or {})
    return {
        "metric_name": str(evaluation.get("metric_name") or ""),
        "final_value": evaluation.get("final_value"),
        "higher_is_better": bool(evaluation.get("higher_is_better", True)),
        "verdict": str(effectiveness.get("verdict") or ""),
        "samples_kept": int(data_yield.get("samples_kept", 0) or 0),
        "optimizer_steps": int(update_quality.get("optimizer_steps", 0) or 0),
        "evaluation_status": str(evaluation.get("status") or "not_available"),
    }


def compare_actuals_to_baseline(
    *,
    expected: Mapping[str, Any],
    actual_entries: Mapping[str, Mapping[str, Any]],
    fixture_paths: Iterable[Path] | Path,
) -> List[Dict[str, Any]]:
    """Compare actual effectiveness entries against a tracked baseline payload."""
    drifts: List[Dict[str, Any]] = []
    validation_errors = validate_baseline_payload(expected)
    if validation_errors:
        for error in validation_errors:
            drifts.append(
                {
                    "entry": "__schema__",
                    "field": "expected",
                    "expected": "valid baseline payload",
                    "actual": error,
                }
            )
        return drifts

    actual_fingerprint = compute_fixture_fingerprint(fixture_paths)
    if actual_fingerprint != expected.get("fixture_fingerprint"):
        drifts.append(
            {
                "entry": "__global__",
                "field": "fixture_fingerprint",
                "expected": expected.get("fixture_fingerprint"),
                "actual": actual_fingerprint,
            }
        )

    expected_entries = dict(expected.get("entries") or {})
    for entry_id, baseline in expected_entries.items():
        actual = dict(actual_entries.get(entry_id) or {})
        if not actual:
            drifts.append(
                {
                    "entry": entry_id,
                    "field": "missing",
                    "expected": "effectiveness entry present",
                    "actual": None,
                }
            )
            continue

        if actual.get("metric_name") != baseline.get("metric_name"):
            drifts.append(
                {
                    "entry": entry_id,
                    "field": "metric_name",
                    "expected": baseline.get("metric_name"),
                    "actual": actual.get("metric_name"),
                }
            )

        if actual.get("evaluation_status") != "available":
            drifts.append(
                {
                    "entry": entry_id,
                    "field": "evaluation_status",
                    "expected": "available",
                    "actual": actual.get("evaluation_status"),
                }
            )

        if actual.get("verdict") != baseline.get("expected_verdict"):
            drifts.append(
                {
                    "entry": entry_id,
                    "field": "verdict",
                    "expected": baseline.get("expected_verdict"),
                    "actual": actual.get("verdict"),
                }
            )

        if int(actual.get("samples_kept", 0) or 0) < int(baseline.get("minimum_samples_kept", 0) or 0):
            drifts.append(
                {
                    "entry": entry_id,
                    "field": "samples_kept",
                    "expected": f">={baseline.get('minimum_samples_kept')}",
                    "actual": actual.get("samples_kept"),
                }
            )

        if int(actual.get("optimizer_steps", 0) or 0) < int(
            baseline.get("minimum_optimizer_steps", 0) or 0
        ):
            drifts.append(
                {
                    "entry": entry_id,
                    "field": "optimizer_steps",
                    "expected": f">={baseline.get('minimum_optimizer_steps')}",
                    "actual": actual.get("optimizer_steps"),
                }
            )

        baseline_value = _coerce_float(baseline.get("baseline_value"))
        final_value = _coerce_float(actual.get("final_value"))
        tolerance = float(baseline.get("tolerance", 0.0) or 0.0)
        higher_is_better = bool(baseline.get("higher_is_better", True))
        if baseline_value is None or final_value is None:
            drifts.append(
                {
                    "entry": entry_id,
                    "field": "final_value",
                    "expected": "numeric value",
                    "actual": actual.get("final_value"),
                }
            )
            continue

        regressed = (
            final_value + tolerance < baseline_value
            if higher_is_better
            else final_value - tolerance > baseline_value
        )
        if regressed:
            drifts.append(
                {
                    "entry": entry_id,
                    "field": "baseline_regression",
                    "expected": baseline_value,
                    "actual": final_value,
                }
            )

    return drifts


def format_drift_lines(drifts: Iterable[Mapping[str, Any]]) -> List[str]:
    """Format drift entries into stable, parseable lines."""
    lines = []
    for drift in drifts:
        lines.append(
            "TRAINING_EFFECTIVENESS_DRIFT "
            f"entry={drift.get('entry')} "
            f"field={drift.get('field')} "
            f"expected={json.dumps(drift.get('expected'), sort_keys=True)} "
            f"actual={json.dumps(drift.get('actual'), sort_keys=True)}"
        )
    return lines


def _normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "metric_name": str(entry.get("metric_name") or ""),
        "baseline_value": _coerce_float(entry.get("baseline_value")),
        "higher_is_better": bool(entry.get("higher_is_better", True)),
        "tolerance": float(max(0.0, entry.get("tolerance", 0.0) or 0.0)),
        "minimum_samples_kept": max(0, int(entry.get("minimum_samples_kept", 0) or 0)),
        "minimum_optimizer_steps": max(
            0, int(entry.get("minimum_optimizer_steps", 0) or 0)
        ),
        "expected_verdict": str(entry.get("expected_verdict") or "pass"),
    }


def _iter_fixture_files(paths: Iterable[Path] | Path) -> Iterable[Path]:
    if isinstance(paths, Path):
        iterable = [paths]
    else:
        iterable = list(paths)

    files: List[Path] = []
    for path in iterable:
        resolved = Path(path)
        if not resolved.exists():
            raise ValueError(f"Fixture path does not exist: {resolved}")
        if resolved.is_file():
            files.append(resolved)
            continue
        files.extend(sorted(p for p in resolved.rglob("*") if p.is_file()))
    return sorted(files)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
