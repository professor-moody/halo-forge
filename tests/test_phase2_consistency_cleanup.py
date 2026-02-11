#!/usr/bin/env python3
"""Phase 2 consistency and contract cleanup regression tests."""

import sys
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from ui.state import AppState
from ui.services.benchmark_service import (
    BenchmarkService,
    BenchmarkType,
    VLM_PRESETS,
)
from ui.services.hardware import GPUStats, get_gpu_summary
from halo_forge.rlvr.verifiers.base import VerifyResult


def test_training_page_sft_dataset_options_are_canonical():
    """SFT dataset options should map to canonical registry keys."""
    try:
        from ui.pages import training as training_page
    except ModuleNotFoundError as e:
        if e.name == "nicegui":
            pytest.skip("nicegui not installed; skipping UI dataset option test")
        raise

    from halo_forge.sft.datasets import list_sft_datasets

    canonical = {spec.name for spec in list_sft_datasets()}
    option_keys = {key for key, _ in training_page.SFT_DATASETS if key != "custom"}

    assert option_keys
    assert option_keys == canonical


def test_benchmark_service_normalizes_commonvoice_alias():
    """Benchmark service should normalize legacy audio alias keys."""
    service = BenchmarkService(AppState())

    cmd = service._build_command(
        model="openai/whisper-small",
        benchmark_type=BenchmarkType.AUDIO,
        benchmark_name="commonvoice",
        limit=5,
        output_path="results/benchmarks/test/benchmark.json",
    )
    dataset_idx = cmd.index("--dataset") + 1
    assert cmd[dataset_idx] == "common_voice"


def test_vlm_presets_only_include_supported_loader_datasets():
    """UI VLM presets should be runnable by the native VLM loader path."""
    source = Path("halo_forge/vlm/data/loaders.py").read_text(encoding="utf-8")
    module_ast = ast.parse(source)
    supported = set()
    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "VLM_DATASETS":
                    if isinstance(node.value, ast.Dict):
                        for key_node in node.value.keys:
                            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                                supported.add(key_node.value)

    preset_names = {preset.dataset for preset in VLM_PRESETS}

    assert supported
    assert "mmstar" not in preset_names
    assert preset_names.issubset(supported)


def test_vision_verifier_details_contract_is_string_and_structured_metadata():
    """Vision verifier should keep details string-based and store structured details in metadata."""
    try:
        from PIL import Image
    except ModuleNotFoundError:
        pytest.skip("Pillow not installed; skipping vision verifier contract test")

    from halo_forge.vlm.verifiers.base import VisionVerifier
    from halo_forge.vlm.verifiers.perception import PerceptionResult
    from halo_forge.vlm.verifiers.reasoning import ReasoningResult
    from halo_forge.vlm.verifiers.output import OutputResult

    verifier = VisionVerifier()
    verifier.perception_checker = SimpleNamespace(
        verify=lambda image, completion: PerceptionResult(
            object_score=0.8,
            text_score=0.7,
            spatial_score=0.9,
            counting_score=0.6,
            overall_score=0.75,
            details={"objects": 2},
        )
    )
    verifier.reasoning_checker = SimpleNamespace(
        verify_with_context=lambda completion, prompt, ground_truth: ReasoningResult(
            structure_score=0.7,
            consistency_score=0.8,
            grounding_score=0.9,
            overall_score=0.8,
            details={"steps": 3},
        )
    )
    verifier.output_checker = SimpleNamespace(
        verify=lambda completion, ground_truth, expected_format: OutputResult(
            exact_match=True,
            fuzzy_score=1.0,
            semantic_score=1.0,
            format_score=1.0,
            overall_score=1.0,
            details={"answer": "ok"},
        )
    )

    result = verifier.verify(
        image=Image.new("RGB", (4, 4)),
        prompt="prompt",
        completion="completion",
        ground_truth="answer",
    )

    assert isinstance(result.details, str)
    assert isinstance(result.metadata.get("details"), dict)
    assert "perception" in result.metadata["details"]


def test_verify_result_repr_tolerates_non_string_details():
    """Debug repr should remain safe even if legacy callers pass non-string details."""
    result = VerifyResult(success=True, reward=1.0, details={"legacy": "payload"})  # type: ignore[arg-type]
    text = repr(result)
    assert text.startswith("VerifyResult(")


def test_perception_checker_gpu_init_falls_back_to_cpu(monkeypatch):
    """Perception checker should retry OCR on CPU when GPU initialization fails."""
    try:
        from halo_forge.vlm.verifiers.perception import PerceptionChecker
    except ModuleNotFoundError as e:
        pytest.skip(f"VLM verifier deps unavailable ({e})")

    calls = []

    class FakeReader:
        def __init__(self, languages, gpu):
            calls.append(bool(gpu))
            if gpu:
                raise RuntimeError("GPU init failed")

    fake_easyocr = SimpleNamespace(Reader=FakeReader)
    monkeypatch.setitem(sys.modules, "easyocr", fake_easyocr)

    try:
        import torch  # type: ignore
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    except ModuleNotFoundError:
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
        )

    checker = PerceptionChecker(use_ocr=True)
    checker._load_ocr()

    assert calls == [True, False]
    assert checker._ocr_loaded is True
    assert checker._ocr is not None


def test_terminal_state_idempotency_stopped_cannot_be_overwritten():
    """Stopped jobs must not transition to conflicting terminal states."""
    state = AppState()
    job = state.create_job(job_type="benchmark", name="test")

    assert state.update_job_status(job.id, "running") is True
    assert state.update_job_status(job.id, "stopped") is True
    assert state.update_job_status(job.id, "completed") is False
    assert state.update_job_status(job.id, "failed") is False
    assert state.get_job(job.id).status == "stopped"


def test_hardware_summary_preserves_zero_values(monkeypatch):
    """Hardware summary should render valid zero measurements, not placeholders."""
    from ui.services import hardware as hardware_module

    stats = GPUStats(
        name="Test GPU",
        utilization_percent=0.0,
        memory_used_gb=0.0,
        memory_total_gb=128.0,
        temperature_c=0.0,
        power_draw_w=0.0,
    )
    monkeypatch.setattr(hardware_module, "get_gpu_stats", lambda: stats)

    summary = get_gpu_summary()
    assert summary["util"] == "0%"
    assert summary["memory"] == "0.0/128GB"
    assert summary["temp"] == "0°C"
    assert summary["power"] == "0W"


def test_inference_verifier_handles_empty_prompts_gracefully():
    """Inference verifier should return a deterministic failure for empty prompt sets."""
    try:
        from halo_forge.inference.verifier import InferenceOptimizationVerifier
    except ModuleNotFoundError as e:
        if e.name == "torch":
            pytest.skip("torch not installed; skipping inference verifier test")
        raise

    verifier = InferenceOptimizationVerifier()
    result = verifier.verify(optimized_model=object(), test_prompts=[], tokenizer=object())

    assert result.success is False
    assert result.reward == 0.0
    assert "at least one prompt" in (result.error or "").lower()
