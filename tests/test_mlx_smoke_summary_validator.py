from __future__ import annotations

import json

import pytest

from scripts.validate_mlx_smoke_summary import SummaryValidationError, validate_summary


def _passing_summary() -> dict:
    labels = [
        "mlx_sft_raft_live_smoke",
        "mlx_dpo_reference_free_live_smoke",
        "mlx_dpo_reference_model_live_smoke",
        "mlx_dpo_non_sigmoid_variants",
        "mlx_grpo_reference_free_live_smoke",
        "mlx_dpo_loss_unit",
        "mlx_dpo_reference_model_terminal",
        "mlx_grpo_terminal",
    ]
    return {
        "status": "passed",
        "readiness": {
            "status": "ready",
            "executable": True,
            "probe": {"default_device": "Device(gpu, 0)", "value": 6.0},
        },
        "checks": [
            {"label": label, "status": "passed", "returncode": 0}
            for label in labels
        ],
    }


def test_validate_mlx_smoke_summary_accepts_release_shape():
    validate_summary(_passing_summary())


def test_validate_mlx_smoke_summary_rejects_missing_live_check():
    summary = _passing_summary()
    summary["checks"] = [
        check
        for check in summary["checks"]
        if check["label"] != "mlx_grpo_reference_free_live_smoke"
    ]

    with pytest.raises(SummaryValidationError, match="missing labels"):
        validate_summary(summary)


def test_validate_mlx_smoke_summary_cli(tmp_path, capsys):
    from scripts.validate_mlx_smoke_summary import main

    path = tmp_path / "mlx_smoke_summary.json"
    path.write_text(json.dumps(_passing_summary()), encoding="utf-8")

    assert main([str(path)]) == 0
    assert "valid" in capsys.readouterr().out
