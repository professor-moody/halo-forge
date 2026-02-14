#!/usr/bin/env python3
"""Phase 9 non-code modality research/testing matrix regression tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from halo_forge.modality_research import (
    NON_CODE_MODALITIES,
    build_non_code_modality_matrix,
    parse_validation_targets,
    validate_modality_training_artifacts,
)


def test_matrix_covers_all_non_code_modalities_and_core_scenarios():
    """Matrix should include positive/negative/benchmark scenarios for each modality."""
    matrix = build_non_code_modality_matrix(seed=42, cycles=2)
    assert set(matrix.keys()) == set(NON_CODE_MODALITIES)

    for modality in NON_CODE_MODALITIES:
        scenarios = {entry.scenario: entry for entry in matrix[modality]}
        assert "train_positive" in scenarios
        assert "train_negative_unsupported_family" in scenarios
        assert "benchmark_smoke" in scenarios
        assert "--seed" in scenarios["train_positive"].command
        seed_index = scenarios["train_positive"].command.index("--seed")
        assert scenarios["train_positive"].command[seed_index + 1] == "42"
        assert scenarios["train_negative_unsupported_family"].expects_non_zero is True


def test_parse_validation_targets_enforces_modality_and_shape(tmp_path):
    """Validation target parser should enforce modality=path structure."""
    out = tmp_path / "vlm_out"
    targets = parse_validation_targets([f"vlm={out}", f"audio={tmp_path / 'audio_out'}"])
    assert targets[0][0] == "vlm"
    assert targets[0][1] == out

    try:
        parse_validation_targets(["badformat"])
        assert False, "expected ValueError for malformed target"
    except ValueError as exc:
        assert "Expected format" in str(exc)

    try:
        parse_validation_targets([f"code={tmp_path / 'x'}"])
        assert False, "expected ValueError for unsupported modality"
    except ValueError as exc:
        assert "Unsupported modality" in str(exc)


def test_artifact_validator_passes_with_required_files(tmp_path):
    """Artifact validator should pass when canonical files are present and parseable."""
    output_dir = tmp_path / "vlm_ok"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_model").mkdir(parents=True, exist_ok=True)

    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "modality": "vlm",
                "run_id": "vlm-123",
                "seed": 42,
                "weights_updated": True,
                "total_train_steps_executed": 3,
                "final_update_reason": "updated",
                "failure_reason": None,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "launch_context.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "job_type": "vlm",
                "service": "training",
                "created_at": "2026-01-01T00:00:00",
                "source_ui_page": "/training",
                "command": ["halo-forge", "vlm", "train"],
                "args": {},
                "relaunch_capabilities": {"can_relaunch": True, "can_clone": True, "can_resume_latest": True},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "latest_checkpoint.json").write_text(
        json.dumps({"cycle": 0}),
        encoding="utf-8",
    )

    result = validate_modality_training_artifacts(
        modality="vlm",
        output_dir=output_dir,
        expected_seed=42,
    )
    assert result.ok is True
    assert result.errors == []


def test_artifact_validator_fails_when_required_files_missing(tmp_path):
    """Validator should fail with explicit errors when canonical artifacts are missing."""
    output_dir = tmp_path / "audio_missing"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = validate_modality_training_artifacts(
        modality="audio",
        output_dir=output_dir,
        expected_seed=42,
    )
    assert result.ok is False
    assert any("training_summary.json" in err for err in result.errors)
    assert any("launch_context.json" in err for err in result.errors)
    assert any("latest_checkpoint.json" in err for err in result.errors)


def test_matrix_script_prints_json_and_validates_outputs(tmp_path):
    """Script should print matrix JSON and validate outputs with non-zero on failures."""
    script = Path("scripts/run_non_code_modality_matrix.py")
    assert script.exists()

    json_run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--print-matrix",
            "--matrix-format",
            "json",
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert json_run.returncode == 0
    parsed = json.loads(json_run.stdout)
    assert set(parsed.keys()) == set(NON_CODE_MODALITIES)

    fail_run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--validate-training",
            f"vlm={tmp_path / 'missing'}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert fail_run.returncode == 1
    assert "validation failed" in fail_run.stdout.lower()
