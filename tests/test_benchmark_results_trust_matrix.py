#!/usr/bin/env python3
"""Benchmark/results trust-matrix regression tests."""

import json

import pytest

from ui.services.benchmark_service import BenchmarkService, BenchmarkType
from ui.services.results_service import ResultsService
from ui.state import AppState


def test_backend_auto_selection_normalizes_hyphenated_vlm_benchmark_names():
    """Auto backend routing should handle hyphen/underscore variants consistently."""
    try:
        from halo_forge.benchmark import BenchmarkBackend, _select_backend
    except ModuleNotFoundError as e:
        if e.name and not e.name.startswith("halo_forge"):
            pytest.skip(f"optional dependency missing ({e.name})")
        raise

    for name in ("mm-ifeval", "mm_ifeval", "mmifeval"):
        selected = _select_backend(model="org/text-only-model", benchmark=name)
        assert selected == BenchmarkBackend.VLMEVALKIT


def test_results_service_normalizes_sparse_legacy_audio_and_agentic_metric_keys(tmp_path):
    """Legacy metric aliases should map to canonical dashboard/table metrics."""
    audio_file = tmp_path / "results" / "benchmarks" / "whisper-common-voice" / "benchmark.json"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_text(
        json.dumps(
            {
                "model": "openai/whisper-small",
                "dataset": "commonvoice",
                "samples": 20,
                "success_rate": 0.0,
                "average_reward": 0.0,
                "average_wer": 0.35,
            }
        ),
        encoding="utf-8",
    )

    agentic_file = tmp_path / "results" / "benchmarks" / "agentic-xlam" / "benchmark.json"
    agentic_file.parent.mkdir(parents=True, exist_ok=True)
    agentic_file.write_text(
        json.dumps(
            {
                "model": "org/tool-model",
                "dataset": "xlam",
                "accuracy": 0.62,
                "function_accuracy": 0.73,
                "json_valid_rate": 0.9,
            }
        ),
        encoding="utf-8",
    )

    sparse_file = tmp_path / "results" / "benchmarks" / "unknown" / "benchmark.json"
    sparse_file.parent.mkdir(parents=True, exist_ok=True)
    sparse_file.write_text(
        json.dumps({"model": "org/partial-model", "benchmark": "custom"}),
        encoding="utf-8",
    )

    service = ResultsService(base_path=tmp_path)
    results = service.list_results(force_refresh=True)
    assert len(results) == 3

    by_model = {result.model: result for result in results}
    audio = by_model["openai/whisper-small"]
    assert audio.domain == "audio"
    assert audio.normalized_metrics["avg_reward"] == 0.0
    assert audio.normalized_metrics["wer"] == 0.35
    assert audio.normalized_metrics["success_rate"] == 0.0

    agentic = by_model["org/tool-model"]
    assert agentic.domain == "agentic"
    assert agentic.normalized_metrics["function_correctness"] == 0.73

    sparse = by_model["org/partial-model"]
    assert sparse.normalized_metrics == {}
    assert sparse.primary_metric is None


def test_dashboard_summary_keeps_models_with_real_zero_scores(tmp_path):
    """Models with parsed results should appear even when their score is exactly zero."""
    result_file = tmp_path / "results" / "benchmarks" / "model-zero" / "benchmark.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(
        json.dumps(
            {
                "model": "org/model-zero",
                "benchmark": "humaneval",
                "metrics": {"pass_at_1": 0.0},
                "samples": 12,
            }
        ),
        encoding="utf-8",
    )

    service = ResultsService(base_path=tmp_path)
    summary = service.get_dashboard_benchmark_summary(max_models=5)
    assert summary["models"], "Expected zero-score model to remain visible in dashboard summary"
    assert any(model["name"] == "model-zero" for model in summary["models"])


def test_benchmark_command_alias_and_results_ingestion_contract(tmp_path):
    """Alias-normalized benchmark commands should produce discoverable canonical results."""
    benchmark_service = BenchmarkService(AppState())
    output_path = tmp_path / "results" / "benchmarks" / "whisper-commonvoice" / "benchmark.json"

    cmd = benchmark_service._build_command(
        model="openai/whisper-small",
        benchmark_type=BenchmarkType.AUDIO,
        benchmark_name="commonvoice",
        limit=10,
        output_path=str(output_path),
    )
    dataset_idx = cmd.index("--dataset") + 1
    assert cmd[dataset_idx] == "common_voice"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "model": "openai/whisper-small",
                "dataset": "common_voice",
                "samples": 10,
                "success_rate": 0.4,
                "average_reward": 0.12,
            }
        ),
        encoding="utf-8",
    )

    results_service = ResultsService(base_path=tmp_path)
    parsed = results_service.list_results(force_refresh=True)
    assert len(parsed) == 1
    assert parsed[0].benchmark == "common_voice"
    assert parsed[0].domain == "audio"
    assert parsed[0].normalized_metrics["avg_reward"] == 0.12


def test_modality_train_no_update_contract_keys_are_consistent(tmp_path):
    """All modality trainers should expose the same no-update telemetry contract keys."""
    expected = {"train_steps_executed", "train_loss", "weights_updated", "update_reason"}

    try:
        from halo_forge.reasoning.trainer import ReasoningRAFTConfig, ReasoningRAFTTrainer
        from halo_forge.agentic.trainer import AgenticRAFTConfig, AgenticRAFTTrainer
        from halo_forge.audio.trainer import AudioRAFTConfig, AudioRAFTTrainer
        from halo_forge.vlm.trainer import VLMRAFTConfig, VLMRAFTTrainer
    except ModuleNotFoundError as e:
        if e.name and not e.name.startswith("halo_forge"):
            pytest.skip(f"optional dependency missing ({e.name})")
        raise

    reasoning = ReasoningRAFTTrainer(
        ReasoningRAFTConfig(output_dir=str(tmp_path / "reasoning"))
    )
    reasoning_metrics = reasoning._train_on_filtered([], cycle=0)
    assert expected.issubset(reasoning_metrics.keys())
    assert reasoning_metrics["weights_updated"] is False

    agentic = AgenticRAFTTrainer(
        AgenticRAFTConfig(output_dir=str(tmp_path / "agentic"))
    )
    agentic_metrics = agentic._train_on_samples([], cycle=0)
    assert expected.issubset(agentic_metrics.keys())
    assert agentic_metrics["weights_updated"] is False

    audio = AudioRAFTTrainer(
        AudioRAFTConfig(
            model_name="facebook/wav2vec2-base-960h",
            output_dir=str(tmp_path / "audio"),
        )
    )
    audio_metrics = audio._train_on_samples([], lr=1e-5)
    assert expected.issubset(audio_metrics.keys())
    assert audio_metrics["update_reason"] == "unsupported_model_family"

    vlm = VLMRAFTTrainer(
        VLMRAFTConfig(
            model_name="org/unsupported-vlm-family",
            output_dir=str(tmp_path / "vlm"),
        )
    )
    vlm_metrics = vlm.train_on_samples([], cycle=0)
    assert expected.issubset(vlm_metrics.keys())
    assert vlm_metrics["update_reason"] == "unsupported_model_family"
