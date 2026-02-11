#!/usr/bin/env python3
"""Phase 0 correctness stabilization regression tests."""

import asyncio
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from halo_forge.cli import RAFT_TRAIN_SUPPORTED_VERIFIERS
from halo_forge.rlvr.verifiers.pytest_verifier import RLVRPytestVerifier
from halo_forge.rlvr.verifiers.test_runner import UnittestVerifier
from ui.services.benchmark_service import BenchmarkService, BenchmarkType
from ui.services.results_service import ResultsService
from ui.services.training_service import TrainingService
from ui.state import AppState


@pytest.mark.parametrize("benchmark_name", ["humaneval", "mbpp", "livecodebench"])
def test_native_benchmark_routes_python_datasets_without_internal_prompts(monkeypatch, benchmark_name):
    """Python benchmark names must route to dataset-based flow, not internal prompts."""
    try:
        import halo_forge.benchmark as benchmark_module
    except ModuleNotFoundError as e:
        if e.name == "torch":
            pytest.skip("torch not installed; skipping benchmark runtime import test")
        raise

    called = {"python_dataset": False}

    def fake_python_dataset(**kwargs):
        called["python_dataset"] = True
        assert kwargs["benchmark"] == benchmark_name
        return {"metrics": {"pass_at_1": 0.5}, "samples": 1}

    class ShouldNotInstantiateRunner:
        def __init__(self, *args, **kwargs):
            raise AssertionError("BenchmarkRunner should not be used for Python benchmark datasets")

    monkeypatch.setattr(benchmark_module, "_run_python_dataset_benchmark", fake_python_dataset)
    monkeypatch.setattr(benchmark_module, "BenchmarkRunner", ShouldNotInstantiateRunner)

    result = benchmark_module._run_native_benchmark(
        model="test/model",
        benchmark=benchmark_name,
        limit=1,
        output=None,
    )

    assert called["python_dataset"] is True
    assert result["metrics"]["pass_at_1"] == 0.5


def test_benchmark_service_launch_normalizes_output_to_benchmark_json(monkeypatch):
    """UI launch path should always normalize to a concrete benchmark.json file."""
    state = AppState()
    service = BenchmarkService(state)
    captured = {}

    async def fake_launch_process(job_id, cmd, on_log=None):
        captured["job_id"] = job_id
        captured["cmd"] = cmd

    monkeypatch.setattr(service, "_launch_process", fake_launch_process)

    job_id = asyncio.run(
        service.launch_benchmark(
            model="Qwen/Qwen2.5-Coder-3B",
            benchmark_type=BenchmarkType.CODE,
            benchmark_name="humaneval",
            output_dir="results/benchmarks/qwen-humaneval",
            limit=5,
        )
    )

    cmd = captured["cmd"]
    output_idx = cmd.index("--output") + 1
    assert Path(cmd[output_idx]).name == "benchmark.json"

    job = state.get_job(job_id)
    assert job is not None
    assert str(job.output_dir).endswith("results/benchmarks/qwen-humaneval")


@pytest.mark.parametrize(
    "benchmark_type, benchmark_name",
    [
        (BenchmarkType.CODE, "humaneval"),
        (BenchmarkType.VLM, "textvqa"),
        (BenchmarkType.AUDIO, "librispeech"),
        (BenchmarkType.AGENTIC, "xlam"),
    ],
)
def test_benchmark_service_commands_use_sys_executable(benchmark_type, benchmark_name):
    """All benchmark subprocess commands should launch with the active interpreter."""
    service = BenchmarkService(AppState())
    cmd = service._build_command(
        model="test/model",
        benchmark_type=benchmark_type,
        benchmark_name=benchmark_name,
        limit=1,
        output_path="results/benchmarks/test-run/benchmark.json",
    )
    assert cmd[0] == sys.executable


def test_results_service_parses_metrics_from_benchmark_json(tmp_path):
    """Results discovery should understand nested metrics in benchmark.json files."""
    result_file = tmp_path / "results" / "benchmarks" / "model-humaneval" / "benchmark.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(
        (
            '{'
            '"model":"Qwen/Qwen2.5-Coder-3B",'
            '"benchmark":"humaneval",'
            '"metrics":{"pass_at_1":0.42,"pass_at_5":0.61,"pass_at_10":0.70},'
            '"samples":164'
            '}'
        ),
        encoding="utf-8",
    )

    service = ResultsService(base_path=tmp_path)
    results = service.scan_results(force_refresh=True)
    assert results
    assert results[0].pass_at_1 == 0.42
    assert results[0].pass_at_5 == 0.61
    assert results[0].samples == 164


def test_training_service_commands_use_sys_executable(monkeypatch):
    """SFT and RAFT subprocess launches should use sys.executable."""
    state = AppState()
    service = TrainingService(state)
    captured = {}

    async def fake_launch_process(job_id, cmd, on_log=None):
        captured[job_id] = cmd

    monkeypatch.setattr(service, "_launch_process", fake_launch_process)

    sft_job = asyncio.run(
        service.launch_sft(
            model="Qwen/Qwen2.5-Coder-3B",
            dataset="codealpaca",
            output_dir="models/sft-test",
            epochs=1,
        )
    )
    raft_job = asyncio.run(
        service.launch_raft(
            model="Qwen/Qwen2.5-Coder-3B",
            prompts="tests/fixtures/sample_prompts.jsonl",
            output_dir="models/raft-test",
            cycles=1,
        )
    )

    assert captured[sft_job][0] == sys.executable
    assert captured[raft_job][0] == sys.executable


def test_raft_verifier_option_parity_cli_and_ui():
    """RAFT verifier options should match between CLI and UI."""
    expected = {
        "gcc",
        "mingw",
        "msvc",
        "humaneval",
        "mbpp",
        "rust",
        "go",
        "auto",
        "execution",
    }
    assert set(RAFT_TRAIN_SUPPORTED_VERIFIERS) == expected

    training_page = Path("ui/pages/training.py").read_text(encoding="utf-8")
    match = re.search(r"VERIFIERS\s*=\s*\[(.*?)\]\n\n", training_page, re.DOTALL)
    assert match, "Could not locate VERIFIERS list in ui/pages/training.py"
    ui_values = set(re.findall(r'\("([a-z_]+)",\s*"[^"]*"\)', match.group(1)))
    assert ui_values == expected


def test_rlvr_pytest_verifier_uses_sys_executable(monkeypatch, tmp_path):
    """RLVRPytestVerifier should execute tests with the current interpreter."""
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"task_id":"t1","prompt":"Add numbers","tests":"def test_solution():\\n    assert add(1,2)==3","entry_point":"add"}\n',
        encoding="utf-8",
    )

    verifier = RLVRPytestVerifier(str(dataset_path))
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = verifier.verify("def add(a, b):\n    return a + b", "t1", "mbpp")
    assert result.success is True
    assert captured["cmd"][0] == sys.executable


def test_unittest_verifier_uses_sys_executable(monkeypatch):
    """UnittestVerifier should execute unittest via the current interpreter."""
    verifier = UnittestVerifier(timeout=5)
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    code = """
import unittest

class TestMath(unittest.TestCase):
    def test_add(self):
        self.assertEqual(1 + 1, 2)
"""

    result = verifier.verify(code)
    assert result.success is True
    assert captured["cmd"][0] == sys.executable


def test_optimizer_tracks_immutable_baseline_and_mutable_current_model(monkeypatch, tmp_path):
    """Quantization reload should not overwrite original baseline identity."""
    try:
        from halo_forge.inference.optimizer import InferenceOptimizer, OptimizationConfig
    except ModuleNotFoundError as e:
        if e.name == "torch":
            pytest.skip("torch not installed; skipping inference optimizer test")
        raise

    config = OptimizationConfig(output_dir=str(tmp_path / "optimized"))

    try:
        with patch("transformers.AutoModelForCausalLM") as model_cls, patch("transformers.AutoTokenizer") as tok_cls:
            model_cls.from_pretrained.return_value = MagicMock()
            tok_cls.from_pretrained.return_value = MagicMock()

            optimizer = InferenceOptimizer(config)
            optimizer.load_model("baseline-model")

            assert optimizer.original_baseline_model_name == "baseline-model"
            assert optimizer.current_model_name == "baseline-model"

            from halo_forge.inference import quantization as quant_module

            called = {}

            def fake_quantize_model_simple(model_path, output_path, precision, calibration_data):
                called["model_path"] = model_path
                called["output_path"] = output_path

            monkeypatch.setattr(quant_module, "quantize_model_simple", fake_quantize_model_simple)

            optimizer.quantize(method="post_training")

            assert called["model_path"] == "baseline-model"
            assert optimizer.original_baseline_model_name == "baseline-model"
            assert optimizer.current_model_name.endswith("/quantized")
    except (ImportError, RuntimeError, ModuleNotFoundError) as e:
        pytest.skip(f"transformers unavailable: {e}")


def test_qat_verifier_uses_original_baseline_name(monkeypatch, tmp_path):
    """QAT path should construct verifier against immutable original baseline model name."""
    try:
        from halo_forge.inference.optimizer import InferenceOptimizer, OptimizationConfig
    except ModuleNotFoundError as e:
        if e.name == "torch":
            pytest.skip("torch not installed; skipping QAT baseline test")
        raise

    config = OptimizationConfig(output_dir=str(tmp_path / "optimized"))

    try:
        with patch("transformers.AutoModelForCausalLM") as model_cls, patch("transformers.AutoTokenizer") as tok_cls:
            model = MagicMock()
            model_cls.from_pretrained.return_value = model
            tok_cls.from_pretrained.return_value = MagicMock()

            optimizer = InferenceOptimizer(config)
            optimizer.load_model("baseline-model")
    except (ImportError, RuntimeError, ModuleNotFoundError) as e:
        pytest.skip(f"transformers unavailable: {e}")

    captured = {}

    class FakeVerifier:
        def __init__(self, baseline_model=None, baseline_model_name=None, **kwargs):
            captured["baseline_model"] = baseline_model
            captured["baseline_model_name"] = baseline_model_name

    class FakeQATTrainer:
        def __init__(self, config):
            self.config = config

        def train(self, model, dataloader, verifier=None, eval_prompts=None):
            captured["verifier_instance"] = verifier
            return model

    fake_calibration = SimpleNamespace(get_dataloader=lambda: [])

    from halo_forge.inference import calibration as calibration_module
    from halo_forge.inference import quantization as quant_module
    from halo_forge.inference import verifier as verifier_module

    monkeypatch.setattr(
        calibration_module.CalibrationDataset,
        "from_jsonl",
        staticmethod(lambda calibration_data, tokenizer: fake_calibration),
    )
    monkeypatch.setattr(quant_module, "QATTrainer", FakeQATTrainer)
    monkeypatch.setattr(verifier_module, "InferenceOptimizationVerifier", FakeVerifier)

    optimizer.quantize(method="qat", calibration_data="dummy.jsonl")

    assert captured["baseline_model"] is None
    assert captured["baseline_model_name"] == "baseline-model"
