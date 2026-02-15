#!/usr/bin/env python3
"""Phase 16 all-module UI execution parity regression tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ui.services.module_ops_service import ModuleOpsService
from ui.services.results_service import ResultsService
from ui.state import AppState


def test_ops_console_route_and_nav_wiring_exist():
    """Ops console should be reachable from app routing and primary navigation surfaces."""
    app_source = Path("ui/app.py").read_text(encoding="utf-8")
    assert "@ui.page('/ops-console')" in app_source

    sidebar_source = Path("ui/components/sidebar.py").read_text(encoding="utf-8")
    assert "/ops-console" in sidebar_source

    dashboard_source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "/ops-console" in dashboard_source


def test_module_ops_service_launch_persists_context(monkeypatch, tmp_path):
    """Utility module launches should persist launch_context.json under results/ops/<module>/<job_id>."""
    state = AppState()
    service = ModuleOpsService(state)

    class FakeBus:
        def emit_sync(self, event):
            return None

        async def emit(self, event):
            return None

    async def fake_launch_process(job_id, cmd, on_log=None):
        return None

    monkeypatch.setattr("ui.services.module_ops_service.get_event_bus", lambda: FakeBus())
    monkeypatch.setattr(service, "_launch_process", fake_launch_process)

    output_root = tmp_path / "results" / "ops"
    job_id = asyncio.run(
        service.launch_module_op(
            module="config",
            execution_mode="contract",
            output_root=str(output_root),
            config_path="configs/sft_example.yaml",
            config_type="sft",
        )
    )

    job = state.get_job(job_id)
    assert job is not None
    assert job.type == "config"
    assert job.output_dir == output_root / "config" / job_id
    launch_context = job.output_dir / "launch_context.json"
    assert launch_context.exists()

    payload = json.loads(launch_context.read_text(encoding="utf-8"))
    assert payload["service"] == "module_ops"
    assert payload["job_type"] == "config"
    assert payload["args"]["module"] == "config"
    assert payload["args"]["execution_mode"] == "contract"


def test_module_ops_service_builds_contract_and_live_commands(tmp_path):
    """Module ops service should produce bounded contract commands and explicit live commands."""
    service = ModuleOpsService(AppState())
    output_dir = tmp_path / "results" / "ops" / "plot" / "abc123"
    output_dir.mkdir(parents=True, exist_ok=True)

    contract_cmd = service._build_command(
        module="plot",
        execution_mode="contract",
        output_dir=output_dir,
        args={},
    )
    assert contract_cmd[:4] == [
        contract_cmd[0],
        "-m",
        "halo_forge.cli",
        "plot",
    ]
    assert "--help" in contract_cmd

    live_cmd = service._build_command(
        module="data",
        execution_mode="live",
        output_dir=output_dir,
        args={
            "data_action": "validate",
            "data_file": "data/rlvr/humaneval_prompts.jsonl",
        },
    )
    assert live_cmd[:5] == [
        live_cmd[0],
        "-m",
        "halo_forge.cli",
        "data",
        "validate",
    ]


def test_monitor_and_results_reference_module_ops_parity_paths():
    """Monitor/results should route utility jobs for stop/relaunch/clone operations."""
    monitor_source = Path("ui/pages/monitor.py").read_text(encoding="utf-8")
    assert "UTILITY_JOB_TYPES" in monitor_source
    assert "self.module_ops_service.stop_job" in monitor_source
    assert "self.module_ops_service.relaunch_from_context" in monitor_source
    assert "ops_clone_payload" in monitor_source
    assert "QUALIFICATION_JOB_TYPES" in monitor_source
    assert "self.qualification_service.stop_job" in monitor_source
    assert "self.qualification_service.relaunch_from_context" in monitor_source

    results_source = Path("ui/pages/results.py").read_text(encoding="utf-8")
    assert "UtilityRunSummary" in results_source
    assert "QualificationReportSummary" in results_source
    assert "_render_utility_runs_table" in results_source
    assert "_render_qualification_reports_table" in results_source
    assert "_clone_utility_to_form" in results_source


def test_results_service_parses_utility_run_summary(tmp_path):
    """Results service should ingest utility run_summary artifacts from results/ops."""
    run_dir = tmp_path / "results" / "ops" / "config" / "job123"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "launch_context.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "job_type": "config",
                "service": "module_ops",
                "created_at": "2026-01-01T00:00:00",
                "source_ui_page": "/ops-console",
                "command": ["python", "-m", "halo_forge.cli", "config", "validate", "configs/sft_example.yaml"],
                "args": {
                    "module": "config",
                    "execution_mode": "contract",
                    "output_root": str(tmp_path / "results" / "ops"),
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
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "job_id": "job123",
                "module": "config",
                "execution_mode": "contract",
                "status": "completed",
                "return_code": 0,
                "output_dir": str(run_dir),
                "command": ["python", "-m", "halo_forge.cli", "config", "validate", "configs/sft_example.yaml"],
                "started_at": "2026-01-01T00:00:00",
                "completed_at": "2026-01-01T00:00:02",
                "duration_seconds": 2.0,
                "artifact_pointers": {
                    "run_summary": str(run_dir / "run_summary.json"),
                    "stdout_log": str(run_dir / "stdout.log"),
                    "launch_context": str(run_dir / "launch_context.json"),
                },
            }
        ),
        encoding="utf-8",
    )

    service = ResultsService(base_path=tmp_path)
    runs = service.list_utility_runs(force_refresh=True)
    assert len(runs) == 1
    run = runs[0]
    assert run.module == "config"
    assert run.execution_mode == "contract"
    assert run.status == "completed"
    assert run.return_code == 0
    assert run.has_relaunch_context is True
    assert run.artifact_pointers["launch_context"].endswith("launch_context.json")


def test_launch_context_supports_module_ops_service_and_utility_job_types():
    """Launch context schema should allow module_ops service and utility job types."""
    source = Path("ui/services/launch_context.py").read_text(encoding="utf-8")
    for token in ('"config"', '"data"', '"info"', '"plot"'):
        assert token in source
    assert '"module_ops"' in source
