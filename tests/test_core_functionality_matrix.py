#!/usr/bin/env python3
"""Core functionality matrix tests for training modules and UI services."""

import asyncio
from pathlib import Path

import pytest

from halo_forge.capabilities import (
    CAPABILITY_STATUS_PROTOTYPE,
    check_modality_train_capability,
)
from ui.services.benchmark_service import BenchmarkService, BenchmarkType
from ui.services.training_service import TrainingService
from ui.state import AppState


def test_training_service_validates_sft_launch_payload():
    """SFT launch should fail fast on invalid required payload fields."""
    service = TrainingService(AppState())

    with pytest.raises(ValueError, match="model is required"):
        asyncio.run(
            service.launch_sft(
                model="",
                dataset="codealpaca",
                output_dir="models/sft",
                epochs=1,
            )
        )

    with pytest.raises(ValueError, match="dataset is required"):
        asyncio.run(
            service.launch_sft(
                model="Qwen/Qwen2.5-Coder-3B",
                dataset="",
                output_dir="models/sft",
                epochs=1,
            )
        )

    with pytest.raises(ValueError, match="epochs must be greater than 0"):
        asyncio.run(
            service.launch_sft(
                model="Qwen/Qwen2.5-Coder-3B",
                dataset="codealpaca",
                output_dir="models/sft",
                epochs=0,
            )
        )


def test_training_service_validates_raft_prompts_path(tmp_path):
    """RAFT launch should fail clearly when prompts path is missing."""
    service = TrainingService(AppState())

    missing_path = tmp_path / "does-not-exist.jsonl"
    with pytest.raises(ValueError, match="prompts file does not exist"):
        asyncio.run(
            service.launch_raft(
                model="Qwen/Qwen2.5-Coder-3B",
                prompts=str(missing_path),
                output_dir="models/raft",
                cycles=1,
            )
        )


def test_benchmark_service_validates_launch_payload():
    """Benchmark launch should fail fast on malformed launch payloads."""
    service = BenchmarkService(AppState())

    with pytest.raises(ValueError, match="model is required"):
        asyncio.run(
            service.launch_benchmark(
                model="",
                benchmark_type=BenchmarkType.CODE,
                benchmark_name="humaneval",
                limit=5,
                output_path="results/benchmarks/x/benchmark.json",
            )
        )

    with pytest.raises(ValueError, match="benchmark_name is required"):
        asyncio.run(
            service.launch_benchmark(
                model="Qwen/Qwen2.5-Coder-3B",
                benchmark_type=BenchmarkType.CODE,
                benchmark_name="",
                limit=5,
                output_path="results/benchmarks/x/benchmark.json",
            )
        )

    with pytest.raises(ValueError, match="limit must be greater than 0"):
        asyncio.run(
            service.launch_benchmark(
                model="Qwen/Qwen2.5-Coder-3B",
                benchmark_type=BenchmarkType.CODE,
                benchmark_name="humaneval",
                limit=0,
                output_path="results/benchmarks/x/benchmark.json",
            )
        )


def test_modality_train_capability_matrix_smoke_contract():
    """Each modality should resolve to explicit allow/fail contract behavior."""
    # Capability behavior should match per-modality status + model-family policy.
    for modality in ("vlm", "audio", "reasoning", "agentic"):
        check = check_modality_train_capability(
            modality=modality,
            model_name=(
                "Qwen/Qwen2-VL-7B-Instruct"
                if modality == "vlm"
                else "openai/whisper-small"
                if modality == "audio"
                else "Qwen/Qwen2.5-7B-Instruct"
            ),
            allow_prototype_train=False,
            dry_run=False,
        )
        if check.capability.status == CAPABILITY_STATUS_PROTOTYPE:
            assert check.allowed is False
            assert check.reason == "prototype_flag_required"
        else:
            assert check.allowed is True

    # Restricted modalities should reject unsupported families even with override.
    vlm_check = check_modality_train_capability(
        modality="vlm",
        model_name="org/unsupported-vision-model",
        allow_prototype_train=True,
        dry_run=False,
    )
    assert vlm_check.allowed is False
    assert vlm_check.reason == "unsupported_model"

    audio_check = check_modality_train_capability(
        modality="audio",
        model_name="facebook/wav2vec2-base-960h",
        allow_prototype_train=True,
        dry_run=False,
    )
    assert audio_check.allowed is False
    assert audio_check.reason == "unsupported_model"

    reasoning_check = check_modality_train_capability(
        modality="reasoning",
        model_name="acme/unsupported-reasoning-model",
        allow_prototype_train=True,
        dry_run=False,
    )
    assert reasoning_check.allowed is False
    assert reasoning_check.reason == "unsupported_model"


def test_monitor_page_avoids_demo_loss_data_and_preserves_zero_metrics_display_logic():
    """Monitor should render truthful metrics and no synthetic demo loss series."""
    source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "Demo data" not in source
    assert "if self.job.latest_loss is not None" in source
    assert "if self.job.latest_lr is not None" in source
    assert "if self.job.latest_grad_norm is not None" in source
