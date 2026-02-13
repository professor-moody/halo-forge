#!/usr/bin/env python3
"""Runtime surface alignment regression tests."""

import asyncio
from pathlib import Path

import pytest

from ui.services.benchmark_service import BenchmarkService, BenchmarkType
from ui.services.training_service import TrainingService
from ui.state import AppState


def test_training_service_fails_fast_for_missing_local_dataset_path(tmp_path):
    """Path-like SFT datasets should be validated before subprocess launch."""
    service = TrainingService(AppState())
    missing_dataset = tmp_path / "missing_dataset.jsonl"

    with pytest.raises(ValueError, match="dataset file does not exist"):
        asyncio.run(
            service.launch_sft(
                model="Qwen/Qwen2.5-Coder-3B",
                dataset=str(missing_dataset),
                output_dir="models/sft",
                epochs=1,
            )
        )


def test_training_service_fails_fast_for_missing_checkpoint_path(tmp_path):
    """RAFT checkpoint path should be validated before job launch."""
    service = TrainingService(AppState())
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text('{"prompt":"p"}\n', encoding="utf-8")
    missing_checkpoint = tmp_path / "missing_checkpoint"

    with pytest.raises(ValueError, match="checkpoint file does not exist"):
        asyncio.run(
            service.launch_raft(
                model="Qwen/Qwen2.5-Coder-3B",
                prompts=str(prompts_path),
                output_dir="models/raft",
                checkpoint=str(missing_checkpoint),
                cycles=1,
            )
        )


def test_training_service_historical_curriculum_validates_stats_file(tmp_path):
    """Historical curriculum must fail fast when stats path is missing."""
    service = TrainingService(AppState())
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text('{"prompt":"p"}\n', encoding="utf-8")
    missing_stats = tmp_path / "missing_curriculum.json"

    with pytest.raises(ValueError, match="curriculum_stats file does not exist"):
        asyncio.run(
            service.launch_raft(
                model="Qwen/Qwen2.5-Coder-3B",
                prompts=str(prompts_path),
                output_dir="models/raft",
                cycles=1,
                curriculum="historical",
                curriculum_stats=str(missing_stats),
            )
        )


def test_benchmark_service_uses_shared_contract_for_numeric_preflight():
    """Benchmark service should fail pre-spawn on invalid samples_per_prompt values."""
    service = BenchmarkService(AppState())

    with pytest.raises(ValueError, match="samples_per_prompt must be greater than 0"):
        asyncio.run(
            service.launch_benchmark(
                model="Qwen/Qwen2.5-Coder-3B",
                benchmark_type=BenchmarkType.CODE,
                benchmark_name="humaneval",
                samples_per_prompt=0,
                output_path="results/benchmarks/test/benchmark.json",
            )
        )


def test_training_page_declares_supported_and_deferred_runtime_modes():
    """Training page should make UI-supported and deferred modes explicit."""
    source = Path("ui/pages/training.py").read_text(encoding="utf-8")
    assert "UI_SUPPORTED_TRAINING_MODES" in source
    assert "UI_DEFERRED_TRAINING_MODES" in source
    assert "Capability-gated modes" in source


def test_monitor_page_reads_canonical_training_summary_fields():
    """Monitor page should consume canonical summary keys when present."""
    source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "training_summary.json" in source
    assert "training_metrics.json" in source
    assert "final_update_reason" in source
    assert "total_train_steps_executed" in source
    assert "final_train_loss" in source
