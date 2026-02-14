#!/usr/bin/env python3
"""Phase 8 UI ops parity and durable relaunch regression tests."""

import asyncio
import json
from pathlib import Path

from ui.services.benchmark_service import BenchmarkService, BenchmarkType
from ui.services.training_service import TrainingService
from ui.state import AppState


def test_launch_context_written_for_training_and_benchmark_jobs(monkeypatch, tmp_path):
    """Training and benchmark launches should persist launch_context.json with schema keys."""
    state = AppState()
    training = TrainingService(state)
    benchmark = BenchmarkService(state)

    class FakeBus:
        def emit_sync(self, event):
            return None

        async def emit(self, event):
            return None

    async def fake_launch_process(job_id, cmd, on_log=None):
        return None

    monkeypatch.setattr("ui.services.training_service.get_event_bus", lambda: FakeBus())
    monkeypatch.setattr("ui.services.benchmark_service.get_event_bus", lambda: FakeBus())
    monkeypatch.setattr(training, "_launch_process", fake_launch_process)
    monkeypatch.setattr(benchmark, "_launch_process", fake_launch_process)

    train_out = tmp_path / "models" / "phase8-sft"
    bench_out = tmp_path / "results" / "phase8-bench" / "benchmark.json"

    asyncio.run(
        training.launch_sft(
            model="Qwen/Qwen2.5-Coder-3B",
            dataset="codealpaca",
            output_dir=str(train_out),
            epochs=1,
        )
    )
    asyncio.run(
        benchmark.launch_benchmark(
            model="Qwen/Qwen2.5-Coder-3B",
            benchmark_type=BenchmarkType.CODE,
            benchmark_name="humaneval",
            output_path=str(bench_out),
            limit=3,
        )
    )

    train_context = train_out / "launch_context.json"
    bench_context = bench_out.parent / "launch_context.json"
    assert train_context.exists()
    assert bench_context.exists()

    train_payload = json.loads(train_context.read_text(encoding="utf-8"))
    bench_payload = json.loads(bench_context.read_text(encoding="utf-8"))
    for payload in (train_payload, bench_payload):
        assert payload["contract_version"] == 1
        assert payload["job_type"]
        assert payload["service"]
        assert isinstance(payload["command"], list)
        assert isinstance(payload["args"], dict)
        assert isinstance(payload["relaunch_capabilities"], dict)


def test_training_resume_latest_cycle_resolution(tmp_path):
    """Resume-latest should derive next cycle from latest_checkpoint.json."""
    service = TrainingService(AppState())
    output_dir = tmp_path / "models" / "phase8-modality"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest_checkpoint.json").write_text(
        json.dumps({"cycle": 2}),
        encoding="utf-8",
    )

    assert service.resolve_resume_latest_cycle(output_dir) == 3


def test_training_relaunch_from_context_applies_resume_latest(monkeypatch, tmp_path):
    """Relaunch from context should map resume-latest into resume_from_cycle."""
    service = TrainingService(AppState())
    output_dir = tmp_path / "models" / "phase8-vlm"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latest_checkpoint.json").write_text(
        json.dumps({"cycle": 1}),
        encoding="utf-8",
    )
    context_path = output_dir / "launch_context.json"
    context_path.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "job_type": "vlm",
                "service": "training",
                "created_at": "2026-01-01T00:00:00",
                "source_ui_page": "/training",
                "command": ["python3", "-m", "halo_forge.cli", "vlm", "train"],
                "args": {
                    "model": "Qwen/Qwen2-VL-7B-Instruct",
                    "dataset": "textvqa",
                    "output_dir": str(output_dir),
                    "cycles": 5,
                    "resume_from_cycle": 0,
                    "seed": 42,
                },
                "relaunch_capabilities": {
                    "can_relaunch": True,
                    "can_clone": True,
                    "can_resume_latest": True,
                },
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    async def fake_launch_modality_train(**kwargs):
        captured.update(kwargs)
        return "job-test"

    monkeypatch.setattr(service, "launch_modality_train", fake_launch_modality_train)
    job_id = asyncio.run(
        service.relaunch_from_context(
            context_path,
            origin_job_id="old-job",
            resume_latest=True,
            source_ui_page="/monitor",
        )
    )

    assert job_id == "job-test"
    assert captured["modality"] == "vlm"
    assert captured["resume_from_cycle"] == 2
    assert captured["origin_job_id"] == "old-job"
    assert captured["relaunch"] is True
    assert captured["resume_strategy"] == "resume_latest"


def test_benchmark_relaunch_from_context_rebuilds_launch(monkeypatch, tmp_path):
    """Benchmark relaunch should rebuild launch_benchmark args from persisted context."""
    service = BenchmarkService(AppState())
    context_path = tmp_path / "launch_context.json"
    context_path.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "job_type": "benchmark",
                "service": "benchmark",
                "created_at": "2026-01-01T00:00:00",
                "source_ui_page": "/benchmark",
                "command": ["python3", "-m", "halo_forge.cli", "benchmark", "eval"],
                "args": {
                    "model": "Qwen/Qwen2.5-Coder-3B",
                    "benchmark_type": "code",
                    "benchmark_name": "humaneval",
                    "output_path": "results/benchmarks/model-humaneval/benchmark.json",
                    "limit": 5,
                    "samples_per_prompt": 3,
                    "run_after_compile": True,
                },
                "relaunch_capabilities": {
                    "can_relaunch": True,
                    "can_clone": True,
                    "can_resume_latest": False,
                },
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    async def fake_launch_benchmark(**kwargs):
        captured.update(kwargs)
        return "bench-job"

    monkeypatch.setattr(service, "launch_benchmark", fake_launch_benchmark)

    job_id = asyncio.run(
        service.relaunch_from_context(
            context_path,
            origin_job_id="bench-old",
            source_ui_page="/monitor",
        )
    )

    assert job_id == "bench-job"
    assert captured["benchmark_type"] == BenchmarkType.CODE
    assert captured["benchmark_name"] == "humaneval"
    assert captured["origin_job_id"] == "bench-old"
    assert captured["relaunch"] is True
    assert captured["resume_strategy"] == "relaunch"


def test_monitor_and_pages_expose_ops_parity_surface_contracts():
    """Monitor/results/training/benchmark should expose clone/relaunch/resume parity hooks."""
    monitor_source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "self.job.type == \"benchmark\"" in monitor_source
    assert "self.benchmark_service.stop_job" in monitor_source
    assert "Resume Latest" in monitor_source
    assert "training_clone_payload" in monitor_source
    assert "benchmark_clone_payload" in monitor_source
    assert "launch context unavailable (invalid JSON)" in monitor_source

    training_source = Path("ui/pages/training.py").read_text(encoding="utf-8")
    assert "training_clone_payload" in training_source
    assert "app.storage.user.pop(\"training_clone_payload\"" in training_source

    benchmark_source = Path("ui/pages/benchmark.py").read_text(encoding="utf-8")
    assert "benchmark_clone_payload" in benchmark_source
    assert "app.storage.user.pop(\"benchmark_clone_payload\"" in benchmark_source

    results_source = Path("ui/pages/results.py").read_text(encoding="utf-8")
    assert "has_relaunch_context" in results_source
    assert "_clone_training_to_form" in results_source
    assert "_clone_benchmark_to_form" in results_source


def test_job_created_event_includes_relaunch_provenance(monkeypatch):
    """Relaunch metadata should be available in JOB_CREATED event payloads."""
    state = AppState()
    service = TrainingService(state)
    captured = []

    class FakeBus:
        def emit_sync(self, event):
            captured.append(event)

        async def emit(self, event):
            captured.append(event)

    async def fake_launch_process(job_id, cmd, on_log=None):
        return None

    monkeypatch.setattr("ui.services.training_service.get_event_bus", lambda: FakeBus())
    monkeypatch.setattr(service, "_launch_process", fake_launch_process)

    asyncio.run(
        service.launch_sft(
            model="Qwen/Qwen2.5-Coder-3B",
            dataset="codealpaca",
            output_dir="models/phase8-provenance",
            epochs=1,
            origin_job_id="source-job",
            relaunch=True,
            resume_strategy="relaunch",
        )
    )

    assert captured
    payload = captured[0].data
    assert payload["origin_job_id"] == "source-job"
    assert payload["relaunch"] is True
    assert payload["resume_strategy"] == "relaunch"
    assert payload["launch_context_file"]
