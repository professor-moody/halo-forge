#!/usr/bin/env python3
"""Phase 4 UI and observability consolidation regression tests."""

import asyncio
import json
import re
from pathlib import Path

from ui.services.benchmark_service import BenchmarkService, BenchmarkType
from ui.services.event_bus import build_transition_payload
from ui.services.results_service import ResultsService
from ui.services.training_service import TrainingService
from ui.state import AppState


def test_results_service_is_authoritative_for_domain_ingestion_and_dashboard_aggregation(tmp_path):
    """Canonical service should normalize multi-domain result schemas for UI consumers."""
    files = {
        tmp_path / "results/code/model-a/benchmark.json": {
            "model": "org/model-a",
            "benchmark": "humaneval",
            "pass_at_k": {"1": 0.42, "5": 0.61, "10": 0.7},
            "samples": 164,
        },
        tmp_path / "results/vlm/model-a/benchmark.json": {
            "model": "org/model-a",
            "dataset": "textvqa",
            "metrics": {"accuracy": 0.55},
            "samples": 500,
        },
        tmp_path / "results/audio/model-b/benchmark.json": {
            "model": "org/model-b",
            "dataset": "librispeech",
            "metrics": {"wer": 0.18},
            "samples": 200,
        },
        tmp_path / "results/agentic/model-c/benchmark.json": {
            "model": "org/model-c",
            "dataset": "xlam",
            "metrics": {"function_correctness": 0.73},
            "samples": 100,
        },
        tmp_path / "results/agentic/model-c/metadata.json": {
            "foo": "bar",
            "not_a_benchmark_result": True,
        },
    }
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    service = ResultsService(base_path=tmp_path)

    all_results = service.list_results(force_refresh=True)
    assert len(all_results) == 4

    grouped = service.get_results_grouped_by_domain()
    assert set(grouped.keys()) >= {"code", "vlm", "audio", "agentic"}

    dashboard = service.get_dashboard_benchmark_summary(max_models=5)
    assert "domains" in dashboard
    assert "models" in dashboard
    assert "Code" in dashboard["domains"]
    assert "Agentic" in dashboard["domains"]
    assert dashboard["models"]


def test_results_page_consumes_service_and_has_no_demo_or_filesystem_parsing():
    """Results page should rely on service DTOs and avoid parser duplication/demo fallback."""
    source = Path("ui/pages/results.py").read_text(encoding="utf-8")

    assert "get_results_service" in source
    assert "glob(" not in source
    assert "json.load(" not in source
    assert "_load_demo_results" not in source
    assert "No benchmark results found" in source


def test_results_domain_table_has_single_actions_header():
    """Domain result header should render one Actions column to preserve alignment."""
    source = Path("ui/pages/results.py").read_text(encoding="utf-8")
    domain_block = re.search(
        r"def _render_domain_table\(self, domain: str, rows: list\[BenchmarkResult\]\):(.*?)for result in rows:",
        source,
        re.DOTALL,
    )
    assert domain_block
    assert domain_block.group(1).count('ui.label("Actions")') == 1
    assert 'ui.label("Actions").classes(\n                    f\'w-36' in source


def test_results_service_inferrs_model_and_benchmark_from_path_for_legacy_audio_outputs(tmp_path):
    """Legacy audio outputs without explicit model/benchmark fields should still normalize cleanly."""
    target = tmp_path / "results/benchmarks/whisper-tiny-librispeech/benchmark.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "success_rate": 1.0,
                "average_reward": 0.95,
                "average_wer": 0.04,
                "samples": 10,
            }
        ),
        encoding="utf-8",
    )

    service = ResultsService(base_path=tmp_path)
    items = service.list_results(force_refresh=True)
    assert len(items) == 1
    assert items[0].model == "whisper-tiny"
    assert items[0].benchmark == "librispeech"
    assert items[0].domain == "audio"


def test_dashboard_benchmark_quick_action_routes_to_benchmark_page():
    """Dashboard quick action should navigate directly to /benchmark."""
    source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "self._render_action_button('Run Benchmark', 'speed', '/benchmark')" in source


def test_dashboard_benchmark_chart_uses_results_service_aggregation():
    """Benchmark chart loading should delegate to ResultsService, not ad-hoc file walking."""
    source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    method_match = re.search(
        r"def _load_benchmark_data\(self\) -> dict:\n(.*?)\n\s+async def _refresh_jobs",
        source,
        re.DOTALL,
    )
    assert method_match, "Could not locate _load_benchmark_data implementation"
    method_body = method_match.group(1)

    assert "get_dashboard_benchmark_summary" in method_body
    assert "Path('results')" not in method_body


def test_transition_trace_records_applied_and_rejected_with_context():
    """AppState should record structured transition attempts, including rejected terminal rewrites."""
    state = AppState()
    job = state.create_job(job_type="benchmark", name="phase4-test")

    assert state.update_job_status(job.id, "running", source="test", reason="start") is True
    assert state.update_job_status(job.id, "stopped", source="test", reason="stop") is True
    assert state.update_job_status(job.id, "completed", source="test", reason="late_complete") is False

    trace = state.get_transition_trace(job.id)
    assert len(trace) == 3

    applied_event = trace[0]
    rejected_event = trace[-1]

    assert applied_event["from_status"] == "pending"
    assert applied_event["to_status"] == "running"
    assert applied_event["applied"] is True
    assert applied_event["source"] == "test"
    assert applied_event["reason"] == "start"
    assert applied_event["timestamp"]

    assert rejected_event["from_status"] == "stopped"
    assert rejected_event["to_status"] == "completed"
    assert rejected_event["applied"] is False
    assert rejected_event["reason"] == "terminal_state_locked"


def test_build_transition_payload_contains_structured_fields():
    """Lifecycle payload helper should expose stable transition metadata keys."""
    payload = build_transition_payload(
        {
            "from_status": "running",
            "to_status": "completed",
            "applied": True,
            "source": "service",
            "reason": "process_exit_ok",
            "timestamp": "2026-02-11T00:00:00",
            "metadata": {"foo": "bar"},
        },
        return_code=0,
    )

    assert payload["from_status"] == "running"
    assert payload["to_status"] == "completed"
    assert payload["applied"] is True
    assert payload["source"] == "service"
    assert payload["reason"] == "process_exit_ok"
    assert payload["timestamp"] == "2026-02-11T00:00:00"
    assert payload["metadata"] == {"foo": "bar"}
    assert payload["return_code"] == 0


def test_training_service_job_created_event_includes_transition_context(monkeypatch):
    """Training service should emit lifecycle payload with transition keys."""
    captured = []

    class FakeBus:
        def emit_sync(self, event):
            captured.append(event)

        async def emit(self, event):
            captured.append(event)

    state = AppState()
    service = TrainingService(state)

    async def fake_launch_process(job_id, cmd, on_log=None):
        return None

    monkeypatch.setattr("ui.services.training_service.get_event_bus", lambda: FakeBus())
    monkeypatch.setattr(service, "_launch_process", fake_launch_process)

    asyncio.run(
        service.launch_sft(
            model="Qwen/Qwen2.5-Coder-3B",
            dataset="codealpaca",
            output_dir="models/phase4-test",
            epochs=1,
        )
    )

    assert captured
    event_data = captured[0].data
    assert event_data["from_status"] is None
    assert event_data["to_status"] == "pending"
    assert event_data["applied"] is True
    assert event_data["source"] == "training_service.launch_sft"
    assert event_data["reason"] == "job_created"
    assert event_data["timestamp"]


def test_benchmark_service_job_created_event_includes_transition_context(monkeypatch):
    """Benchmark service should emit lifecycle payload with transition keys."""
    captured = []

    class FakeBus:
        def emit_sync(self, event):
            captured.append(event)

        async def emit(self, event):
            captured.append(event)

    state = AppState()
    service = BenchmarkService(state)

    async def fake_launch_process(job_id, cmd, on_log=None):
        return None

    monkeypatch.setattr("ui.services.benchmark_service.get_event_bus", lambda: FakeBus())
    monkeypatch.setattr(service, "_launch_process", fake_launch_process)

    asyncio.run(
        service.launch_benchmark(
            model="Qwen/Qwen2.5-Coder-3B",
            benchmark_type=BenchmarkType.CODE,
            benchmark_name="humaneval",
            limit=5,
            output_path="results/benchmarks/phase4/benchmark.json",
        )
    )

    assert captured
    event_data = captured[0].data
    assert event_data["from_status"] is None
    assert event_data["to_status"] == "pending"
    assert event_data["applied"] is True
    assert event_data["source"] == "benchmark_service.launch_benchmark"
    assert event_data["reason"] == "job_created"
    assert event_data["timestamp"]
