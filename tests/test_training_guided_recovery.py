#!/usr/bin/env python3
"""Guided recovery contract and UI regression tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from halo_forge.training_recovery import build_recovery_guidance
from ui.services.results_service import ResultsService
from ui.services.training_service import TrainingService
from ui.state import AppState


def test_recovery_rule_lowers_reward_threshold_for_threshold_pressure():
    guidance = build_recovery_guidance(
        modality="raft",
        yield_diagnostics={
            "summary": {"status": "low_yield", "dominant_rejection_reason": "below_reward_threshold"},
            "rates": {"keep_rate": 0.05},
        },
        effectiveness={"reasons": []},
        launch_args={"reward_threshold": 0.8, "keep_percent": 0.3},
        representative_examples=[{"reason": "below_reward_threshold", "label": "sample", "preview": "code"}],
    )

    assert guidance["status"] == "ready"
    assert guidance["suggested_overrides"]["reward_threshold"] == 0.7


def test_recovery_rule_raises_keep_percent_when_keep_filter_is_too_strict():
    guidance = build_recovery_guidance(
        modality="agentic",
        yield_diagnostics={
            "summary": {"status": "low_yield", "dominant_rejection_reason": "dropped_by_keep_percent"},
        },
        effectiveness={"reasons": []},
        launch_args={"keep_percent": 0.25},
    )

    assert guidance["status"] == "ready"
    assert guidance["suggested_overrides"]["keep_percent"] == 0.35


def test_recovery_rule_is_advisory_only_for_dataset_schema_issues():
    guidance = build_recovery_guidance(
        modality="sft",
        yield_diagnostics={
            "summary": {"status": "no_signal", "dominant_rejection_reason": "missing_text"},
        },
        effectiveness={"reasons": []},
        launch_args={"max_samples": 16},
        representative_examples=[{"reason": "missing_text", "label": "row", "preview": "{...}"}],
    )

    assert guidance["status"] == "advisory_only"
    assert guidance["suggested_overrides"] == {}


def test_recovery_rule_falls_back_to_unavailable_for_unknown_cases():
    guidance = build_recovery_guidance(
        modality="sft",
        yield_diagnostics={"summary": {"status": "healthy"}},
        effectiveness={"reasons": []},
        launch_args={"max_samples": 64},
    )

    assert guidance["status"] == "unavailable"


def test_training_relaunch_from_context_applies_overrides_without_mutating_source(monkeypatch, tmp_path):
    service = TrainingService(AppState())
    output_dir = tmp_path / "models" / "guided-recovery"
    output_dir.mkdir(parents=True, exist_ok=True)
    context_path = output_dir / "launch_context.json"
    original_payload = {
        "contract_version": 1,
        "job_type": "raft",
        "service": "training",
        "created_at": "2026-01-01T00:00:00",
        "source_ui_page": "/training",
        "command": ["python3", "-m", "halo_forge.cli", "raft", "train"],
        "args": {
            "model": "Qwen/Qwen2.5-Coder-1.5B",
            "prompts": "data/rlvr/humaneval_prompts.jsonl",
            "output_dir": str(output_dir),
            "cycles": 3,
            "samples_per_prompt": 4,
            "keep_percent": 0.2,
            "reward_threshold": 0.8,
            "min_samples": 16,
        },
        "relaunch_capabilities": {
            "can_relaunch": True,
            "can_clone": True,
            "can_resume_latest": True,
        },
    }
    context_path.write_text(json.dumps(original_payload), encoding="utf-8")

    captured = {}

    async def fake_launch_raft(**kwargs):
        captured.update(kwargs)
        return "guided-job"

    monkeypatch.setattr(service, "launch_raft", fake_launch_raft)

    job_id = asyncio.run(
        service.relaunch_from_context(
            context_path,
            override_args={"reward_threshold": 0.7, "keep_percent": 0.3},
            guided_recovery={"reason_code": "below_reward_threshold", "evidence_summary": "Too many drops."},
            source_ui_page="/results",
        )
    )

    assert job_id == "guided-job"
    assert captured["reward_threshold"] == 0.7
    assert captured["keep_percent"] == 0.3
    assert captured["guided_recovery"]["reason_code"] == "below_reward_threshold"
    reread = json.loads(context_path.read_text(encoding="utf-8"))
    assert reread["args"]["reward_threshold"] == 0.8
    assert reread["args"]["keep_percent"] == 0.2


def test_results_service_parses_recovery_guidance_and_examples(tmp_path):
    output_dir = tmp_path / "models" / "raft_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "launch_context.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "job_type": "raft",
                "service": "training",
                "created_at": "2026-01-01T00:00:00",
                "source_ui_page": "/training",
                "command": [],
                "args": {"reward_threshold": 0.8, "keep_percent": 0.2},
                "relaunch_capabilities": {
                    "can_relaunch": True,
                    "can_clone": True,
                    "can_resume_latest": True,
                },
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "modality": "raft",
                "model_name": "Qwen/Qwen2.5-Coder-1.5B",
                "cycles": [],
                "yield_diagnostics": {
                    "summary": {"status": "low_yield", "dominant_rejection_reason": "below_reward_threshold"}
                },
                "effectiveness": {"verdict": "fail", "reasons": ["no_train_steps"]},
                "recovery_guidance": {
                    "status": "ready",
                    "recommended_action": "Lower reward threshold",
                    "suggested_overrides": {"reward_threshold": 0.7},
                    "reason_code": "below_reward_threshold",
                    "evidence_summary": "Most samples were dropped below threshold.",
                    "representative_examples": [
                        {"reason": "below_reward_threshold", "label": "Dropped sample", "preview": "completion"}
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    run = ResultsService(base_path=tmp_path).list_training_runs(force_refresh=True)[0]

    assert run.recovery_status == "ready"
    assert run.recovery_recommended_action == "Lower reward threshold"
    assert run.recovery_suggested_overrides["reward_threshold"] == 0.7
    assert run.representative_examples[0]["label"] == "Dropped sample"


def test_monitor_results_and_training_surfaces_expose_guided_recovery_hooks():
    monitor_source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "Apply Suggested Fix" in monitor_source
    assert "def _open_recovery_review_dialog" in monitor_source
    assert "def _apply_recovery_guidance" in monitor_source

    results_source = Path("ui/pages/results.py").read_text(encoding="utf-8")
    assert "Apply suggested fix" in results_source
    assert "def _show_recovery_review_dialog" in results_source

    training_source = Path("ui/pages/training.py").read_text(encoding="utf-8")
    assert "Suggested Recovery Changes" in training_source
    assert "suggested_overrides" in training_source
