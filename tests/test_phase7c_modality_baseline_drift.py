#!/usr/bin/env python3
"""Phase 7C deterministic modality baseline drift tests."""

import copy
import json
import subprocess
from pathlib import Path

from halo_forge.modality_baseline import (
    MODALITY_ENTRY_KEYS,
    REQUIRED_MODALITIES,
    compare_baseline_payloads,
    compute_fixture_pack_fingerprint,
    validate_baseline_payload,
)


def _sample_modality_entry() -> dict:
    return {
        "cycles_executed": 1,
        "total_train_steps_executed": 1,
        "weights_updated": True,
        "final_update_reason": "updated",
        "failure_reason": None,
        "optimizer_steps": 1,
        "skipped_batches_non_finite": 0,
        "checkpoint_written": True,
        "final_model_written": True,
        "training_summary_written": True,
        "resume_contract_ok": True,
    }


def _sample_payload() -> dict:
    return {
        "contract_version": 1,
        "generator_version": "phase7b.v1",
        "seed": 42,
        "fixture_pack": compute_fixture_pack_fingerprint(),
        "created_at": "2026-02-14T00:00:00+00:00",
        "modalities": {modality: _sample_modality_entry() for modality in REQUIRED_MODALITIES},
    }


def test_cli_modality_baseline_flags_are_exposed():
    """CLI parser should expose baseline controls under test command."""
    source = Path("halo_forge/cli.py").read_text(encoding="utf-8")
    assert "--baseline-file" in source
    assert "--write-baseline" in source
    assert "--compare-baseline" in source


def test_fixture_pack_fingerprint_changes_on_fixture_edit(tmp_path):
    """Fingerprint should change when fixture JSONL content changes."""
    fixture_dir = tmp_path / "fixtures" / "modality"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for modality in REQUIRED_MODALITIES:
        (fixture_dir / f"{modality}_samples.jsonl").write_text(
            '{"sample":"a"}\n',
            encoding="utf-8",
        )

    before = compute_fixture_pack_fingerprint(fixture_dir)
    (fixture_dir / "vlm_samples.jsonl").write_text('{"sample":"b"}\n', encoding="utf-8")
    after = compute_fixture_pack_fingerprint(fixture_dir)
    assert before != after


def test_baseline_compare_ignores_created_at_but_detects_field_drift():
    """Comparison should ignore timestamps and detect contract value drift."""
    expected = _sample_payload()
    current = copy.deepcopy(expected)
    current["created_at"] = "2030-01-01T00:00:00+00:00"

    assert compare_baseline_payloads(expected, current) == []

    current["modalities"]["vlm"]["weights_updated"] = False
    drifts = compare_baseline_payloads(expected, current)
    assert drifts
    assert any(d["modality"] == "vlm" and d["key"] == "weights_updated" for d in drifts)


def test_baseline_validate_fails_when_modality_missing():
    """Schema validation should fail when a modality entry is missing."""
    payload = _sample_payload()
    payload["modalities"].pop("agentic")
    errors = validate_baseline_payload(payload)
    assert errors
    assert any("modalities keys must be exactly" in err for err in errors)


def test_check_script_fails_on_missing_modality_entry(tmp_path):
    """Baseline check script should fail non-zero on malformed baseline schema."""
    baseline_path = tmp_path / "baseline.json"
    payload = _sample_payload()
    payload["modalities"].pop("agentic")
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")

    result = subprocess.run(
        [
            "python3",
            "scripts/generate_modality_baseline.py",
            "--baseline-file",
            str(baseline_path),
            "--check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    combined = f"{result.stdout}\n{result.stderr}"
    assert "baseline schema invalid" in combined


def test_tracked_baseline_file_is_valid_and_complete():
    """Tracked baseline snapshot should validate against schema contract."""
    path = Path("tests/baselines/modality_runtime_baseline.v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_baseline_payload(payload)
    assert errors == []
    assert tuple(sorted(payload["modalities"].keys())) == tuple(sorted(REQUIRED_MODALITIES))
    for modality in REQUIRED_MODALITIES:
        assert tuple(sorted(payload["modalities"][modality].keys())) == tuple(sorted(MODALITY_ENTRY_KEYS))
