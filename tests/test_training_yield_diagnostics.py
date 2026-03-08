#!/usr/bin/env python3
"""Training yield diagnostics contract and UI regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from halo_forge.training_contracts import (
    build_cycle_summary,
    build_training_summary,
    build_yield_diagnostics,
)
from ui.services.metrics_parser import MetricsParser
from ui.services.results_service import ResultsService
from ui.services.training_service import TrainingService
from ui.state import AppState


def test_build_yield_diagnostics_normalizes_reasons_and_summary():
    diagnostics = build_yield_diagnostics(
        stage_counts={"generated": 20, "verified": 10, "filtered": 4, "kept": 2, "dropped": 18},
        thresholds={
            "configured_reward_threshold": 0.8,
            "effective_reward_threshold": 0.6,
            "keep_percent": 0.25,
            "threshold_adjusted": True,
        },
        minimums={"minimum_samples_target": 4},
        rejection_reasons={"verification failed": 10, "dropped by keep percent": 2},
        reward_distribution={"0_8_plus": 2},
    )

    assert diagnostics["summary"]["status"] == "low_yield"
    assert diagnostics["rejection_reasons"]["verification_failed"] == 10
    assert diagnostics["rejection_reasons"]["dropped_by_keep_percent"] == 2
    assert diagnostics["thresholds"]["threshold_adjusted"] is True
    assert diagnostics["minimums"]["minimum_samples_met"] is False


def test_training_summary_aggregates_cycle_yield_diagnostics():
    cycle = build_cycle_summary(
        cycle=0,
        learning_rate=1e-5,
        samples_seen=10,
        samples_kept=3,
        cycle_duration_seconds=0.2,
        update_metrics={
            "train_steps_executed": 2,
            "train_loss": 0.3,
            "weights_updated": True,
            "update_reason": "updated",
        },
        yield_diagnostics={
            "stage_counts": {"generated": 10, "verified": 10, "filtered": 5, "kept": 3, "dropped": 7},
            "minimums": {"minimum_samples_target": 2},
            "rejection_reasons": {"verification_failed": 4, "dropped_by_keep_percent": 2},
        },
    )

    summary = build_training_summary(
        modality="reasoning",
        model_name="org/model",
        total_cycles_planned=1,
        cycles=[cycle],
    )

    assert summary["yield_diagnostics"]["stage_counts"]["kept"] == 3
    assert summary["yield_diagnostics"]["rejection_reasons"]["verification_failed"] == 4


def test_metrics_parser_extracts_machine_readable_yield_snapshot():
    parser = MetricsParser()

    parsed = parser.parse_line(
        'HALO_YIELD {"stage_counts":{"generated":16,"kept":4},"rates":{"keep_rate":0.25},"summary":{"status":"healthy"}}'
    )

    assert parsed is not None
    assert parsed.yield_snapshot["stage_counts"]["generated"] == 16
    assert parsed.yield_snapshot["summary"]["status"] == "healthy"


def test_training_service_preflight_exposes_quality_outlook_for_risky_raft(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt":"one"}\n{"prompt":"two"}\n', encoding="utf-8")
    service = TrainingService(AppState())

    preflight = service.preflight_raft_launch(
        model="Qwen/Qwen2.5-Coder-1.5B",
        prompts=str(prompts),
        output_dir=str(tmp_path / "raft_run"),
        cycles=3,
        samples_per_prompt=2,
        keep_percent=0.1,
        reward_threshold=0.9,
        min_samples=16,
        max_new_tokens=256,
    )

    assert preflight.ok is True
    assert preflight.quality_outlook["status"] == "low_yield"
    assert preflight.quality_outlook["warnings"]
    assert "yield" in preflight.quality_outlook["summary"].lower() or "signal" in preflight.quality_outlook["summary"].lower()


def test_training_service_updates_live_job_yield_snapshot():
    app_state = AppState()
    job = app_state.create_job("raft", "raft")
    service = TrainingService(app_state)

    service._update_job_metrics(
        job.id,
        SimpleNamespace(
            loss=None,
            learning_rate=None,
            epoch=None,
            step=None,
            total_steps=None,
            cycle=1,
            total_cycles=2,
            compile_rate=None,
            grad_norm=None,
            yield_snapshot={"summary": {"status": "low_yield"}},
        ),
    )

    updated = app_state.get_job(job.id)
    assert updated is not None
    assert updated.latest_yield_snapshot["summary"]["status"] == "low_yield"
    assert len(updated.yield_history) == 1


def test_training_service_normalizes_carriage_return_progress_chunks():
    service = TrainingService(AppState())

    normalized = service._normalize_stream_chunk(
        "Map:   0%|          | 0/20022 [00:00<?, ? examples/s]\r"
        "Map: 100%|##########| 20022/20022 [00:00<00:00, 28062.05 examples/s]\n"
        "HALO_YIELD {\"summary\":{\"status\":\"healthy\"}}\n"
    )

    assert normalized == [
        "Map: 100%|##########| 20022/20022 [00:00<00:00, 28062.05 examples/s]",
        'HALO_YIELD {"summary":{"status":"healthy"}}',
    ]


def test_training_service_deduplicates_repeated_progress_redraw_lines():
    service = TrainingService(AppState())

    first = service._should_skip_stream_line("job-1", "Map: 100%|##########| 24/24 [00:22<00:00, 1.05it/s]")
    second = service._should_skip_stream_line("job-1", "Map: 100%|##########| 24/24 [00:22<00:00, 1.05it/s]")
    normal = service._should_skip_stream_line("job-1", "Loaded 200 examples")

    assert first is False
    assert second is True
    assert normal is False


def test_results_service_parses_training_quality_fields(tmp_path):
    output_dir = tmp_path / "models" / "reasoning_run"
    output_dir.mkdir(parents=True)
    payload = {
        "modality": "reasoning",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "cycles_executed": 1,
        "cycles": [],
        "weights_updated": True,
        "total_train_steps_executed": 2,
        "final_train_loss": 0.3,
        "effectiveness": {"verdict": "pass", "reasons": []},
        "yield_diagnostics": {
            "rates": {"keep_rate": 0.4},
            "summary": {
                "status": "healthy",
                "text": "Most samples were usable.",
                "dominant_rejection_reason": "verification_failed",
            },
        },
    }
    (output_dir / "training_summary.json").write_text(json.dumps(payload), encoding="utf-8")

    parsed = ResultsService(base_path=tmp_path).list_training_runs(force_refresh=True)

    assert len(parsed) == 1
    run = parsed[0]
    assert run.effectiveness_verdict == "pass"
    assert run.quality_status == "healthy"
    assert run.keep_rate == pytest.approx(0.4)
    assert run.dominant_rejection_reason == "verification_failed"


def test_sft_load_dataset_tracks_missing_text_and_format_errors(tmp_path):
    try:
        from halo_forge.sft.trainer import SFTConfig, SFTTrainer
    except ModuleNotFoundError as e:
        if e.name and not e.name.startswith("halo_forge"):
            pytest.skip(f"optional dependency missing ({e.name})")
        raise

    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "good sample"}),
                json.dumps({"text": ""}),
                json.dumps({"message": "missing"}),
                json.dumps(["bad", "row"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trainer = SFTTrainer(SFTConfig(train_file=str(dataset_path), validation_split=0.5, max_samples=None))
    trainer.tokenizer = object()

    train_dataset, val_dataset = trainer.load_dataset(file_path=str(dataset_path))

    assert len(train_dataset) + len(val_dataset) == 1
    reasons = trainer.dataset_yield_diagnostics["rejection_reasons"]
    assert reasons["missing_text"] == 2
    assert reasons["format_invalid"] == 1
    assert trainer.dataset_yield_diagnostics["rates"]["keep_rate"] == pytest.approx(0.25)
    assert trainer.dataset_yield_diagnostics["stage_counts"]["dropped"] == 3


def test_raft_verify_and_filter_reports_threshold_adjustment():
    try:
        from halo_forge.rlvr.raft_trainer import RAFTTrainer
    except ModuleNotFoundError as e:
        if e.name and not e.name.startswith("halo_forge"):
            pytest.skip(f"optional dependency missing ({e.name})")
        raise

    trainer = RAFTTrainer.__new__(RAFTTrainer)
    trainer.config = SimpleNamespace(
        verification_chunk_size=32,
        reward_threshold=0.8,
        keep_top_percent=0.5,
        min_samples_per_cycle=3,
    )
    trainer.verifier = SimpleNamespace(
        verify_batch=lambda completions, prompts: [
            SimpleNamespace(reward=0.9, success=True, details={}),
            SimpleNamespace(reward=0.7, success=True, details={}),
            SimpleNamespace(reward=0.6, success=True, details={}),
            SimpleNamespace(reward=0.1, success=False, details={}),
        ]
    )
    trainer._log = lambda *args, **kwargs: None

    filtered, stats, _ = trainer.verify_and_filter(
        [("p1", "a"), ("p2", "b"), ("p3", "c"), ("p4", "d")]
    )

    assert len(filtered) == 3
    assert stats["threshold_adjusted"] is True
    assert stats["effective_threshold"] == pytest.approx(0.6)
