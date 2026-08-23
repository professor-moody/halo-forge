#!/usr/bin/env python3
"""Regression coverage for the public API/view-model pivot."""

from __future__ import annotations

import asyncio
import json
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from halo_forge.public_api.service import PublicApiService
from halo_forge.public_api.serve_manager import ManagedServeProcess, ServeStartRequest
from ui.services.results_service import ResultsService
from ui.services.training_service import TrainingLaunchPreflight, TrainingService
from ui.state import AppState


def _fake_readiness_service():
    report = SimpleNamespace(
        generated_at="2026-03-08T12:00:00Z",
        modules={
            "sft": SimpleNamespace(
                readiness_tier="production_ready",
                production_ready=True,
                status="pass",
                fix_now="No action needed.",
                warnings=[],
                errors=[],
                eval_metric_name="pass@1",
                baseline_value=0.3,
                final_value=0.32,
                delta=0.02,
                weights_updated=True,
                optimizer_steps=12,
                samples_kept=190,
            ),
            "vlm": SimpleNamespace(
                readiness_tier="qualified",
                production_ready=False,
                status="warn",
                fix_now="Finish deterministic eval coverage.",
                warnings=["Eval coverage is still qualification-only."],
                errors=[],
                eval_metric_name="accuracy",
                baseline_value=0.5,
                final_value=0.5,
                delta=0.0,
                weights_updated=True,
                optimizer_steps=8,
                samples_kept=24,
            ),
        },
    )
    return SimpleNamespace(load_qualification_report=lambda force_refresh=True: report)


def test_public_api_training_results_expose_product_and_research_layers(tmp_path):
    output_dir = tmp_path / "models" / "sft_public_run"
    output_dir.mkdir(parents=True)
    summary_payload = {
        "modality": "sft",
        "model_name": "Qwen/Qwen2.5-Coder-1.5B",
        "run_id": "sft-public-1",
        "weights_updated": True,
        "final_update_reason": "updated",
        "total_train_steps_executed": 24,
        "final_train_loss": 1.11,
        "effectiveness": {
            "verdict": "pass",
            "reasons": [],
            "evaluation": {
                "metric_name": "pass@1",
                "baseline_value": 0.3,
                "final_value": 0.33,
                "delta": 0.03,
            },
        },
        "yield_diagnostics": {
            "rates": {"keep_rate": 0.95},
            "summary": {
                "status": "healthy",
                "text": "Most samples were usable for SFT.",
                "dominant_rejection_reason": None,
            },
        },
        "recovery_guidance": {
            "status": "unavailable",
            "recommended_action": "",
            "reason_code": "",
            "evidence_summary": "",
            "suggested_overrides": {},
            "representative_examples": [],
        },
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary_payload),
        encoding="utf-8",
    )

    service = PublicApiService(
        app_state=AppState(),
        results_service=ResultsService(base_path=tmp_path),
        readiness_service=_fake_readiness_service(),
        training_service=TrainingService(AppState()),
        base_path=tmp_path,
    )

    payload = service.list_training_results(include_research=True)

    assert len(payload["items"]) == 1
    item = payload["items"][0]
    assert item["headline"] == "Run completed"
    assert item["details"]["keep_rate"] == 0.95
    assert item["research_sections"][0]["key"] == "data_yield"
    assert item["research_sections"][0]["summary"] == "Most samples were usable for SFT."
    assert "internal_details" not in item


def test_public_api_readiness_uses_qualification_truth(tmp_path):
    service = PublicApiService(
        app_state=AppState(),
        results_service=ResultsService(base_path=tmp_path),
        readiness_service=_fake_readiness_service(),
        training_service=TrainingService(AppState()),
        base_path=tmp_path,
    )

    payload = service.list_readiness()

    assert payload["aggregate_tier"] == "qualified"
    assert payload["items"][0]["readiness_tier"] == "production_ready"
    assert payload["items"][1]["next_step"] == "Finish deterministic eval coverage."


def test_public_api_readiness_falls_back_when_report_missing(tmp_path):
    readiness_service = SimpleNamespace(
        load_qualification_report=lambda force_refresh=True: (_ for _ in ()).throw(
            FileNotFoundError("qualification report missing")
        )
    )
    service = PublicApiService(
        app_state=AppState(),
        results_service=ResultsService(base_path=tmp_path),
        readiness_service=readiness_service,
        training_service=TrainingService(AppState()),
        base_path=tmp_path,
    )

    payload = service.list_readiness()

    assert payload["aggregate_tier"] == "experimental"
    assert payload["generated_at"] is None
    assert {item["modality"] for item in payload["items"]} >= {"sft", "raft", "vlm", "audio"}
    assert all(item["status"] == "warn" for item in payload["items"])
    assert "not available" in payload["items"][0]["caveat"]


def test_public_api_dashboard_summary_exposes_workstation_queues(tmp_path):
    output_dir = tmp_path / "models" / "raft_attention_run"
    output_dir.mkdir(parents=True)
    summary_payload = {
        "modality": "raft",
        "model_name": "Qwen/Qwen2.5-Coder-1.5B",
        "run_id": "raft-attention-1",
        "weights_updated": True,
        "final_update_reason": "updated",
        "total_train_steps_executed": 8,
        "final_train_loss": 0.91,
        "effectiveness": {
            "verdict": "warn",
            "reasons": ["evaluation unavailable"],
            "evaluation": {
                "metric_name": "pass@1",
                "baseline_value": 0.3,
                "final_value": None,
                "delta": None,
            },
        },
        "yield_diagnostics": {
            "rates": {"keep_rate": 0.25},
            "summary": {
                "status": "low_yield",
                "text": "Only a quarter of candidates were kept.",
                "dominant_rejection_reason": "below_reward_threshold",
            },
        },
        "recovery_guidance": {
            "status": "ready",
            "recommended_action": "Lower reward threshold",
            "reason_code": "below_reward_threshold",
            "evidence_summary": "Reward filtering removed too many candidates.",
            "suggested_overrides": {"reward_threshold": 0.45},
            "representative_examples": [],
        },
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")

    service = PublicApiService(
        app_state=AppState(),
        results_service=ResultsService(base_path=tmp_path),
        readiness_service=_fake_readiness_service(),
        training_service=TrainingService(AppState()),
        base_path=tmp_path,
    )

    payload = service.get_dashboard_summary()

    assert payload["readiness_tier"] == "qualified"
    assert payload["production_ready_count"] == 1
    assert payload["modality_count"] == 2
    assert payload["attention_count"] == 1
    assert payload["recent_outcomes"][0]["next_step"] == "Lower reward threshold"
    assert payload["attention_items"][0]["headline"] == "Suggested fix ready"


def test_public_api_docs_catalog_tracks_public_frontend_and_internal_console(tmp_path):
    service = PublicApiService(
        app_state=AppState(),
        results_service=ResultsService(base_path=tmp_path),
        readiness_service=_fake_readiness_service(),
        training_service=TrainingService(AppState()),
        base_path=Path.cwd(),
    )

    payload = service.list_docs_capabilities()
    slugs = {item["slug"] for item in payload["items"]}

    assert "public-frontend" in slugs
    assert "web-ui-console" in slugs


def test_public_frontend_scaffold_references_public_workflows():
    api_helper = Path("public_app/src/lib/api.ts").read_text(encoding="utf-8")
    train_source = Path("public_app/src/routes/train.tsx").read_text(encoding="utf-8")
    start_source = Path("public_app/src/routes/start.tsx").read_text(encoding="utf-8")
    run_source = Path("public_app/src/routes/runs.$runId.tsx").read_text(encoding="utf-8")
    results_source = Path("public_app/src/routes/results.tsx").read_text(encoding="utf-8")
    docs_source = Path("public_app/src/routes/docs.tsx").read_text(encoding="utf-8")
    home_source = Path("public_app/src/routes/index.tsx").read_text(encoding="utf-8")
    shell_source = Path("public_app/src/components/shell/index.tsx").read_text(encoding="utf-8")
    sidebar_source = Path("public_app/src/components/shell/sidebar.tsx").read_text(encoding="utf-8")

    assert 'const API_BASE = "/api/public"' in api_helper
    assert "export type VersionInfo" in api_helper
    assert 'versionInfo: () => request<VersionInfo>("/version")' in api_helper
    assert 'from "@/components/shell"' in home_source
    assert "SystemCard" in home_source
    assert "Topbar" in shell_source
    assert "useVersionInfo" in sidebar_source
    assert "display_version" in sidebar_source
    assert "Base model" in train_source
    assert "Launch summary" in train_source
    assert 'to: "/train"' in start_source
    assert "Legacy deep link retained for compatibility" in start_source
    assert "MLX readiness" in train_source
    assert "accelerator" in train_source
    assert "Run detail view" in run_source
    assert 'label="STATUS"' in run_source
    assert "failure_summary" in api_helper
    assert "FailureSummaryCard" in run_source
    assert "StageRail" in run_source
    assert "LiveNowCard" in run_source
    assert "Waiting for the first optimizer step" in run_source
    assert "metric_points" in api_helper
    assert "Run could not create its log file" in Path("halo_forge/public_api/service.py").read_text(encoding="utf-8")
    assert 'to: "/runs"' in results_source
    assert 'view: "completed"' in results_source
    assert "Completed runs live in Runs" in results_source
    assert "Guided training" in docs_source
    assert "Use your own data" in docs_source
    assert "Remote workstation" in docs_source
    assert "Hugging Face access" in docs_source
    assert "CLI reference" in docs_source
    assert "Training that stays understandable" not in home_source
    assert "Public workstation" not in shell_source
    assert "Launch workspace" not in train_source


def test_public_frontend_remote_auth_regression_contract():
    api_helper = Path("public_app/src/lib/api.ts").read_text(encoding="utf-8")
    shell_source = Path("public_app/src/components/shell/index.tsx").read_text(
        encoding="utf-8"
    )
    sidebar_source = Path("public_app/src/components/shell/sidebar.tsx").read_text(
        encoding="utf-8"
    )
    connect_source = Path("public_app/src/routes/connect.tsx").read_text(
        encoding="utf-8"
    )
    stream_source = Path("public_app/src/lib/event-source.ts").read_text(
        encoding="utf-8"
    )

    assert 'AUTH_REQUIRED_EVENT = "halo-forge:auth-required"' in api_helper
    assert "reportAuthRequired(payload)" in api_helper
    assert "isAuthRequiredError(error)" in sidebar_source
    assert 'navigate({ to: "/connect"' in shell_source
    assert "api.health()" in connect_source
    assert "setApiToken(nextToken)" in connect_source
    assert "Hugging Face access" in connect_source
    assert "useHuggingFaceSaveToken" in connect_source
    assert "hf_" in connect_source
    assert 'TOKEN_STORAGE_KEY = "halo-forge:api-token"' in api_helper
    assert 'headers.Authorization = `Bearer ${token}`' in stream_source
    assert 'reportAuthRequired({ source: "stream"' in stream_source


def test_public_frontend_friendly_workstation_contract():
    start_source = Path("public_app/src/routes/start.tsx").read_text(encoding="utf-8")
    train_source = Path("public_app/src/routes/train.tsx").read_text(encoding="utf-8")
    runs_source = Path("public_app/src/routes/runs.index.tsx").read_text(encoding="utf-8")
    run_source = Path("public_app/src/routes/runs.$runId.tsx").read_text(encoding="utf-8")
    models_source = Path("public_app/src/routes/models.tsx").read_text(encoding="utf-8")
    results_source = Path("public_app/src/routes/results.tsx").read_text(encoding="utf-8")
    playground_source = Path("public_app/src/routes/playground.tsx").read_text(encoding="utf-8")
    overview_source = Path("public_app/src/routes/index.tsx").read_text(encoding="utf-8")
    main_source = Path("public_app/src/main.tsx").read_text(encoding="utf-8")
    api_source = Path("public_app/src/lib/api.ts").read_text(encoding="utf-8")
    logs_source = Path("public_app/src/components/run/logs-panel.tsx").read_text(
        encoding="utf-8"
    )

    assert "const GOALS" in train_source
    assert '"code"' in train_source
    assert '"reasoning"' in train_source
    assert '"tool-use"' in train_source
    assert '"apple-silicon"' in train_source
    assert '"gsm8k_sft"' in train_source
    assert '"xlam_sft"' in train_source
    assert "Run started" in train_source
    assert "useWorkspaceInfo" in train_source
    assert "default_run_root" in train_source
    assert 'to: "/train"' in start_source
    assert "models/start-" not in train_source
    assert "export type RunLive" in api_source
    assert "runLive:" in api_source
    assert "/events" in run_source
    assert "LiveSummary" in run_source
    assert "plainRunStatus" in run_source
    assert "Reconnecting to logs" in logs_source
    assert "Fits ${detectedBackend}" in models_source
    assert "showing all catalog models" in models_source
    assert "useServeStart" in models_source
    assert "Use in Train" in models_source
    assert "preferredTrainMode" in models_source
    assert "export type ServeStatus" in api_source
    assert 'state: "idle" | "starting" | "running" | "unhealthy" | "exited" | string' in api_source
    assert "ready_state" in api_source
    assert "active_action" in api_source
    assert "error_hint" in api_source
    assert "model_ready" in api_source
    assert "load_error" in api_source
    assert "serveStart:" in api_source
    assert "huggingFaceStatus" in api_source
    assert "huggingFaceCheckModel" in api_source
    assert "ServeStatusPanel" in playground_source
    assert "useServeLogs" in playground_source
    assert "formatUpstreamError" in playground_source
    assert "This model requires Hugging Face access" in playground_source
    assert "Start safe model" in playground_source
    assert "No model serving" in playground_source
    assert "Start a local model to unlock chat" in playground_source
    assert "Start a model before chatting" in playground_source
    assert "server ready" in playground_source
    assert "chat ready" in playground_source
    assert "Waiting for model to finish loading" in playground_source
    assert "Managed local serving uses Hugging Face access from Connection" in playground_source
    assert "externalEndpoint ? apiKey" in playground_source
    assert "Serve & Test" in models_source
    assert "license_note" in models_source
    assert "download_note" in models_source
    assert "Model page" in models_source
    assert "startServing" in models_source
    assert "servingThis" in models_source
    assert "chat ready" in models_source
    assert "UNIFIED MEM" in Path("public_app/src/components/shell/telemetry.tsx").read_text(encoding="utf-8")
    telemetry_source = Path("public_app/src/components/shell/telemetry.tsx").read_text(encoding="utf-8")
    assert "limited" in telemetry_source
    assert "macOS sensor access" in telemetry_source
    assert "TelemetryStatusChip" in telemetry_source
    assert "Choose open model" in playground_source
    assert 'to: "/runs"' in results_source
    assert 'view: "completed"' in results_source
    assert 'title="Runs"' in runs_source
    assert "Completed" in runs_source
    assert "Collections" in runs_source
    assert "No runs indexed yet. Launch a guided run from Train to populate." in runs_source
    assert "Apple Neural Accelerators (experimental)" in overview_source
    assert "installChunkLoadRecovery" in main_source


def test_public_api_transport_exposes_public_workflows():
    api_source = Path("halo_forge/public_api/app.py").read_text(encoding="utf-8")

    assert "/version" in api_source
    assert "/dashboard" in api_source
    assert "/interface-capabilities" in api_source
    assert "/training-scenarios" in api_source
    assert "/dataset-imports" in api_source
    assert "/dataset-inspections/{inspection_id}/mapping-preview" in api_source
    assert "/dataset-inspections/{inspection_id}/preparation-preview" in api_source
    assert "/dataset-sources/{source_id}/refresh" in api_source
    assert "/dataset-versions/{version_id}/readiness" in api_source
    assert "/dataset-versions/{version_id}/proof-run" in api_source
    assert "/runs/{run_id}/full-run" in api_source
    assert "/train/preflight" in api_source
    assert "/train/launch" in api_source
    assert "/workspace" in api_source
    assert "/huggingface/status" in api_source
    assert "/huggingface/token" in api_source
    assert "/huggingface/check-model" in api_source
    assert "/runs/{run_id}/live" in api_source
    assert "/runs/{run_id}/guided-recovery" in api_source
    assert "/serve/status" in api_source
    assert "/serve/start" in api_source
    assert "/serve/stop" in api_source
    assert "/readiness" in api_source
    assert "/docs" in api_source
    assert "find_frontend_dist" in api_source
    assert "_mount_frontend" in api_source


def test_public_api_version_contract():
    from fastapi.testclient import TestClient
    from halo_forge import __display_version__, __version__
    from halo_forge.public_api.app import create_app
    from halo_forge.version import DISPLAY_VERSION, PACKAGE_VERSION, RELEASE_CHANNEL

    assert __version__ == PACKAGE_VERSION == "2.0.0a2"
    assert __display_version__ == DISPLAY_VERSION == "2.0.0-alpha-2"
    assert RELEASE_CHANNEL == "alpha"

    app = create_app(serve_frontend=False)
    with TestClient(app) as client:
        payload = client.get("/api/public/version").json()

    assert payload["package_version"] == "2.0.0a2"
    assert payload["display_version"] == "2.0.0-alpha-2"
    assert payload["release_channel"] == "alpha"


def test_public_workspace_defaults_to_writable_run_root(tmp_path, monkeypatch):
    run_root = tmp_path / "halo-runs"
    monkeypatch.setenv("HALO_FORGE_RUN_ROOT", str(run_root))
    service = PublicApiService(
        app_state=AppState(),
        results_service=ResultsService(base_path=tmp_path),
        readiness_service=_fake_readiness_service(),
        training_service=TrainingService(AppState()),
        base_path=tmp_path / "repo",
    )

    info = service.get_workspace_info()

    assert info["default_run_root"] == str(run_root.resolve())
    assert info["writable"] is True
    assert run_root.is_dir()


def test_public_preflight_recommends_default_run_root_for_output_permissions(tmp_path, monkeypatch):
    run_root = tmp_path / "halo-runs"
    monkeypatch.setenv("HALO_FORGE_RUN_ROOT", str(run_root))

    class FakeTrainingService:
        def preflight_sft_launch(self, **kwargs):
            return TrainingLaunchPreflight(
                ok=False,
                errors=[
                    f"output_dir cannot be created from current permissions: {kwargs['output_dir']}"
                ],
                warnings=[],
                resolved_paths={"output_dir": kwargs["output_dir"]},
                suggested_fixes=["Choose an output_dir under a writable path."],
                quality_outlook={},
            )

    service = PublicApiService(
        app_state=AppState(),
        results_service=ResultsService(base_path=tmp_path),
        readiness_service=_fake_readiness_service(),
        training_service=FakeTrainingService(),
        base_path=tmp_path / "installed-app-cwd",
    )

    result = service.preflight_training(
        {
            "mode": "sft",
            "model": "Qwen/Qwen2.5-Coder-0.5B",
            "dataset": "codealpaca",
            "output_dir": "models/start-code-qwen",
        }
    )

    assert result["ok"] is False
    assert result["errors"] == [
        "Halo Forge could not write to this folder: models/start-code-qwen"
    ]
    assert result["suggested_fixes"][0] == f"Use the default run folder: {run_root.resolve()}"


def test_public_api_serves_built_frontend_with_spa_fallback(tmp_path):
    from fastapi.testclient import TestClient
    from halo_forge.public_api.app import create_app

    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        '<html><body><div id="root">Halo Forge</div></body></html>',
        encoding="utf-8",
    )
    (dist / "logo.png").write_bytes(b"png")
    (assets / "app.js").write_text("console.log('halo')", encoding="utf-8")

    app = create_app(frontend_dist=dist)
    with TestClient(app) as client:
        root = client.get("/")
        nested = client.get("/start")
        logo = client.get("/logo.png")
        asset = client.get("/assets/app.js")
        missing_api = client.get("/api/not-a-dashboard-route")

    assert root.status_code == 200
    assert root.headers["cache-control"].startswith("no-store")
    assert "Halo Forge" in root.text
    assert nested.status_code == 200
    assert nested.headers["cache-control"].startswith("no-store")
    assert "Halo Forge" in nested.text
    assert logo.status_code == 200
    assert "max-age=31536000" in logo.headers["cache-control"]
    assert logo.content == b"png"
    assert asset.status_code == 200
    assert "immutable" in asset.headers["cache-control"]
    assert "console.log" in asset.text
    assert missing_api.status_code == 404


def test_public_api_finds_desktop_bundled_frontend_dist(tmp_path, monkeypatch):
    from halo_forge.public_api.app import find_frontend_dist

    desktop_frontend = tmp_path / "frontend"
    desktop_frontend.mkdir()
    (desktop_frontend / "index.html").write_text(
        '<html><body><div id="root">Halo Forge Desktop</div></body></html>',
        encoding="utf-8",
    )

    monkeypatch.setenv("HALO_FORGE_FRONTEND_DIST", str(desktop_frontend))

    assert find_frontend_dist() == desktop_frontend.resolve()


def test_public_api_managed_serve_contract_rejects_second_process(tmp_path):
    class FakeServeManager:
        def __init__(self):
            self.running = False
            self.model = None

        def status(self):
            state = "running" if self.running else "idle"
            return {
                "running": self.running,
                "state": state,
                "ready_state": "chat_ready" if self.running else "idle",
                "active_action": "serving" if self.running else None,
                "pid": 123 if self.running else None,
                "model": self.model,
                "backend": "mlx",
                "host": "127.0.0.1",
                "port": 8001,
                "url": "http://127.0.0.1:8001/v1",
                "started_at": 1.0 if self.running else None,
                "exit_code": None,
                "log_path": None,
                "logs_available": self.running,
                "last_error": None,
                "error_hint": None,
                "healthy": self.running,
                "model_ready": self.running,
                "message": "Local model server is ready." if self.running else "No local model is being served.",
            }

        def start(self, request):
            if self.running:
                raise ValueError("already being served")
            assert request.model == "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
            assert request.host == "127.0.0.1"
            assert request.port == 8001
            self.running = True
            self.model = request.model
            return self.status()

        def stop(self):
            self.running = False
            return self.status()

        def logs(self, *, tail=200):
            return {"available": True, "lines": ["ready"], "path": "serve.log"}

        def health(self):
            return self.status()

    fake_serve = FakeServeManager()
    service = PublicApiService(
        app_state=AppState(),
        results_service=ResultsService(base_path=tmp_path),
        readiness_service=_fake_readiness_service(),
        training_service=TrainingService(AppState()),
        base_path=tmp_path,
        serve_manager=fake_serve,
    )

    started = service.serve_start(
        {"model": "mlx-community/Qwen2.5-0.5B-Instruct-bf16", "backend": "mlx"}
    )

    assert started["running"] is True
    assert started["state"] == "running"
    assert started["ready_state"] == "chat_ready"
    assert started["active_action"] == "serving"
    assert service.serve_status()["model"] == "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
    with pytest.raises(ValueError, match="already being served"):
        service.serve_start({"model": "Qwen/Qwen2.5-1.5B-Instruct"})
    assert service.serve_logs()["lines"] == ["ready"]
    assert service.serve_stop()["running"] is False


def test_managed_serve_rejects_busy_port_before_spawn(tmp_path):
    manager = ManagedServeProcess(base_path=tmp_path, log_dir=tmp_path / "serve-logs")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("parent sandbox does not permit loopback socket binding")
        sock.listen(1)
        port = sock.getsockname()[1]

        with pytest.raises(ValueError, match="already in use"):
            manager.start(
                ServeStartRequest(
                    model="mlx-community/Qwen2.5-0.5B-Instruct-bf16",
                    backend="mlx",
                    host="127.0.0.1",
                    port=port,
                )
            )

    assert manager.status()["state"] == "idle"
    assert manager.status()["ready_state"] == "idle"
    assert manager.status()["active_action"] is None


def test_desktop_tauri_foundation_contract():
    tauri_dir = Path("apps/desktop-tauri/src-tauri")
    config = (tauri_dir / "tauri.conf.json").read_text(encoding="utf-8")
    main_rs = (tauri_dir / "src" / "main.rs").read_text(encoding="utf-8")
    capabilities = (tauri_dir / "capabilities" / "default.json").read_text(encoding="utf-8")
    startup_html = Path("apps/desktop-tauri/startup/index.html").read_text(encoding="utf-8")

    assert '"identifier": "ai.haloforge.desktop"' in config
    assert '"version": "2.0.0-alpha-2"' in config
    assert '"frontendDist": "../startup"' in config
    assert '"withGlobalTauri": true' in config
    assert '"devUrl": "http://127.0.0.1:8765"' in config
    assert '"url": "index.html"' in config
    assert '"targets": ["app", "dmg", "appimage", "deb"]' in config
    assert '"../../../public_app/dist": "frontend"' in config
    assert '"../runtime/dist/halo-forge-runtime": "runtime/halo-forge-runtime"' in config
    assert '"macOS"' in config
    assert '"hardenedRuntime": true' in config
    assert '"minimumSystemVersion": "12.0"' in config
    assert '"../runtime/dist/halo-forge-runtime": "runtime/halo-forge-runtime"' in config
    assert 'halo-forge-runtime.exe' in main_rs
    assert Path("halo_forge/desktop_runtime.py").exists()
    assert Path("apps/desktop-tauri/scripts/build_runtime.py").exists()
    assert Path("apps/desktop-tauri/runtime/desktop_runtime_entry.py").exists()
    assert (tauri_dir / "sidecars" / "halo-forge-runtime").exists()
    assert (tauri_dir / "sidecars" / "halo-forge-runtime-aarch64-apple-darwin").exists()
    assert (tauri_dir / "sidecars" / "halo-forge-runtime-x86_64-unknown-linux-gnu").exists()
    assert "StdCommand::new(runtime_exe)" in main_rs
    assert "runtime_executable_in_dir" in main_rs
    assert 'StdCommand::new("taskkill")' in main_rs
    assert "HALO_FORGE_FRONTEND_DIST" in main_rs
    assert "HALO_FORGE_REPO_ROOT" in main_rs
    assert "runtime_self_check_failed" in main_rs
    assert "run_bundled_self_check" in main_rs
    assert "fn dev_repo_root()" in main_rs
    assert "desktop_status" in main_rs
    assert "app_version" in main_rs
    assert 'const RELEASE_CHANNEL: &str = "alpha"' in main_rs
    assert "desktop_retry" in main_rs
    assert "port_conflict" in main_rs
    assert "health_timeout" in main_rs
    assert "backend_exited" in main_rs
    assert "runtime.log" in main_rs
    assert 'const DASHBOARD_PORT: u16 = 8765' in main_rs
    assert '"dashboard"' in main_rs
    assert '"--no-build"' in main_rs
    assert "GET /api/public/health HTTP/1.1" in main_rs
    assert "child.kill()" in main_rs
    assert '"shell:allow-spawn"' in capabilities
    assert "Starting Halo Forge" in startup_html
    assert "2.0.0-alpha-2" in startup_html
    assert "desktop_status" in startup_html
    assert "desktop_retry" in startup_html
    for sidecar_name in [
        "halo-forge-runtime",
        "halo-forge-runtime-aarch64-apple-darwin",
        "halo-forge-runtime-x86_64-unknown-linux-gnu",
    ]:
        sidecar_source = (tauri_dir / "sidecars" / sidecar_name).read_text(encoding="utf-8")
        assert "HALO_FORGE_BUNDLED_RUNTIME" in sidecar_source
        assert "bundled runtime error" in sidecar_source
        assert "HALO_FORGE_REPO_ROOT" in sidecar_source
        assert ".venv/bin/python" in sidecar_source
        assert "desktop dev runtime error" in sidecar_source


def test_desktop_runtime_entrypoint_self_check_contract():
    source = Path("halo_forge/desktop_runtime.py").read_text(encoding="utf-8")
    build_script = Path("apps/desktop-tauri/scripts/build_runtime.py").read_text(encoding="utf-8")

    assert "--desktop-self-check" in source
    assert "multiprocessing.freeze_support()" in source
    assert "HALO_FORGE_FRONTEND_DIST" in source
    assert "halo_forge.public_api.app" in source
    assert "mlx.nn" in source
    assert '"-m", "halo_forge.cli"' in source
    assert "PyInstaller" in build_script
    assert "--onedir" in build_script
    assert ".[mlx]" in build_script
    assert "mlx_lm" in build_script
    assert "mlx._reprlib_fix" in build_script
    assert "mlx.metallib" in build_script
    assert "libjaccl.dylib" in build_script
    assert "halo-forge-runtime" in build_script


def test_packaged_runtime_smoke_script_documents_alpha_training_gate():
    script = Path("apps/desktop-tauri/scripts/packaged_sft_smoke.py").read_text(encoding="utf-8")
    package = Path("apps/desktop-tauri/package.json").read_text(encoding="utf-8")
    checklist = Path("docs/RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    desktop_readme = Path("apps/desktop-tauri/README.md").read_text(encoding="utf-8")

    assert "hf-internal-testing/tiny-random-gpt2" in script
    assert "--desktop-self-check" in script
    assert "--no-lora" in script
    assert "training_summary.json" in script
    assert "final_model" in script
    assert "/api/public/health" in script
    assert '"/start"' in script
    assert '"/runs"' in script
    assert '"/results"' in script
    assert '"/diagnostics"' in script
    assert '"--accelerator"' in script
    assert 'if accelerator != "auto"' in script
    assert '"smoke:runtime": "python3 scripts/packaged_sft_smoke.py"' in package
    assert "npm run smoke:runtime" in checklist
    assert "What users should see during training" in desktop_readme
    assert "Dataset preprocessing" in desktop_readme
    assert "optimizer steps" in desktop_readme


def test_ci_covers_public_dashboard_and_unsigned_desktop_builds():
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    release = Path(".github/workflows/release.yml").read_text(encoding="utf-8")

    assert "public-dashboard-regression" in ci
    assert "tests/test_public_api_pivot.py" in ci
    assert "tests/test_model_catalog.py" in ci
    assert "npm run build" in ci
    assert "desktop-preview-build" in ci
    assert "windows-latest" in ci
    assert "--bundles deb,appimage" in ci
    assert "--bundles nsis" in ci
    assert "macos-14" in ci
    assert "ubuntu-latest" in ci
    assert "Build bundled desktop runtime" in ci
    assert "scripts/build_runtime.py" in ci
    assert "tests/test_playground_proxy.py" in ci
    assert "tests/test_serving.py" in ci
    assert "libwebkit2gtk-4.1-dev" in ci
    assert "timeout-minutes: 60" in ci
    assert "tauri build" in Path("apps/desktop-tauri/package.json").read_text(encoding="utf-8")
    assert "push:" in release
    assert "tags:" in release
    assert '"v*"' in release
    assert "workflow_dispatch" in release
    assert "Desktop release artifact" in release
    assert "Publish GitHub Release" in release
    assert "Detect macOS signing credentials" in release
    assert "Build unsigned macOS preview candidate" in release
    assert "APPLE_CERTIFICATE" in release
    assert "APPLE_CERTIFICATE_PASSWORD" in release
    assert "APPLE_SIGNING_IDENTITY" in release
    assert "APPLE_ID" in release
    assert "APPLE_PASSWORD" in release
    assert "APPLE_TEAM_ID" in release
    assert "Validate signed and notarized macOS app" in release
    assert "codesign --verify --deep --strict" in release
    assert "xcrun stapler validate" in release
    assert "spctl -a -vvv -t execute" in release
    assert "spctl -a -vvv -t open --context context:primary-signature" in release
    assert "Smoke packaged macOS runtime" in release
    assert "Smoke Windows bundled runtime" in release
    assert "npm run smoke:runtime -- --accelerator cpu" in ci
    assert release.count("--accelerator cpu") == 3
    assert "Record distribution qualification evidence" in release
    assert "Normalize macOS release artifact names" in release
    assert 'normalized="$dmg_dir/Halo-Forge_${version}_aarch64${artifact_suffix}.dmg"' in release
    assert 'artifact_suffix="-unsigned-preview"' in release
    assert 'signature_state="unsigned"' in release
    assert "scripts/write_macos_release_manifest.py" in release
    assert "publish_unsigned_macos_preview" in release
    assert '[ "$is_prerelease" = true ]' in release
    assert "Stable releases never publish an unsigned macOS package" in release
    assert "macOS developer-preview warning" in release
    assert "Verify its SHA-256 checksum before opening it" in release
    assert "halo-forge-release-manifest.json" in release
    assert "*.dmg.sha256" in release
    assert "gh release create" in release
    assert "gh release upload" in release
    assert "OVERSIZE_RELEASE_ASSETS.txt" in release
    assert "Unsigned Linux and Windows desktop packages remain workflow preview artifacts" in release
    assert "2147483648" in release
    assert "RELEASE_NOTES_${version}.md" in release
    assert "--prerelease" in release


@pytest.mark.parametrize(
    ("payload", "expected_method", "expected_pairs", "absent_keys"),
    [
        (
            {
                "mode": "sft",
                "model": "Qwen/Qwen2.5-Coder-1.5B",
                "dataset": "codealpaca",
                "output_dir": "models/sft_public_run",
                "epochs": 1,
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "max_samples": 200,
                "learning_rate": 2e-4,
                "accelerator": "mlx",
            },
            "sft",
            {"learning_rate": 2e-4, "max_samples": 200, "accelerator": "mlx"},
            {"temperature", "reward_threshold", "limit", "samples_per_prompt"},
        ),
        (
            {
                "mode": "raft",
                "model": "Qwen/Qwen2.5-Coder-3B",
                "prompts": "data/rlvr/humaneval_prompts.jsonl",
                "output_dir": "models/raft_public_run",
                "verifier": "execution",
                "cycles": 2,
                "samples_per_prompt": 6,
                "keep_percent": 0.6,
                "reward_threshold": 0.5,
                "temperature": 0.7,
            },
            "raft",
            {"verifier": "execution", "samples_per_prompt": 6, "reward_threshold": 0.5, "temperature": 0.7},
            {"dataset", "limit", "learning_rate"},
        ),
        (
            {
                "mode": "dpo",
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "dataset": "ultrafeedback",
                "output_dir": "models/dpo_public_run",
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 16,
                "learning_rate": 5e-6,
                "max_samples": 64,
                "beta": 0.1,
                "loss_type": "sigmoid",
                "reference_free": True,
            },
            "dpo",
            {"beta": 0.1, "loss_type": "sigmoid", "reference_free": True},
            {"prompts", "limit", "samples_per_prompt"},
        ),
        (
            {
                "mode": "orpo",
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "dataset": "ultrafeedback",
                "output_dir": "models/orpo_public_run",
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 16,
                "learning_rate": 8e-6,
                "beta": 0.1,
            },
            "orpo",
            {"beta": 0.1, "learning_rate": 8e-6},
            {"loss_type", "prompts", "verifier"},
        ),
        (
            {
                "mode": "rm",
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "dataset": "ultrafeedback",
                "output_dir": "models/rm_public_run",
                "epochs": 1,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 1e-5,
            },
            "rm",
            {"batch_size": 4, "gradient_accumulation_steps": 4},
            {"beta", "loss_type", "verifier"},
        ),
        (
            {
                "mode": "grpo",
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "dataset": "gsm8k",
                "output_dir": "models/grpo_public_run",
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 16,
                "learning_rate": 1e-6,
                "verifier": "json_schema",
                "num_generations": 4,
                "beta": 0.04,
                "reward_threshold": 0.0,
            },
            "grpo",
            {"verifier": "json_schema", "num_generations": 4, "beta": 0.04},
            {"prompts", "samples_per_prompt", "task"},
        ),
        (
            {
                "mode": "vlm",
                "model": "Qwen/Qwen2-VL-2B-Instruct",
                "dataset": "textvqa",
                "output_dir": "models/vlm_public_run",
                "cycles": 2,
                "limit": 24,
                "samples_per_prompt": 3,
                "keep_percent": 0.6,
                "reward_threshold": 0.5,
                "temperature": 0.7,
                "allow_prototype_train": True,
            },
            "vlm",
            {"limit": 24, "samples_per_prompt": 3, "reward_threshold": 0.5, "allow_prototype_train": True},
            {"learning_rate", "task"},
        ),
        (
            {
                "mode": "audio",
                "model": "openai/whisper-tiny",
                "dataset": "librispeech",
                "output_dir": "models/audio_public_run",
                "cycles": 2,
                "samples_per_prompt": 3,
                "keep_percent": 0.6,
                "reward_threshold": 0.5,
                "temperature": 0.7,
                "task": "asr",
                "allow_prototype_train": True,
            },
            "audio",
            {"samples_per_prompt": 3, "reward_threshold": 0.5, "task": "asr", "allow_prototype_train": True},
            {"limit", "learning_rate"},
        ),
        (
            {
                "mode": "reasoning",
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "dataset": "gsm8k",
                "output_dir": "models/reasoning_public_run",
                "cycles": 2,
                "limit": 64,
                "keep_percent": 0.6,
                "temperature": 0.7,
                "learning_rate": 1e-5,
                "allow_prototype_train": True,
            },
            "reasoning",
            {"limit": 64, "learning_rate": 1e-5, "temperature": 0.7, "allow_prototype_train": True},
            {"reward_threshold", "samples_per_prompt", "task"},
        ),
        (
            {
                "mode": "agentic",
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "dataset": "xlam",
                "output_dir": "models/agentic_public_run",
                "cycles": 2,
                "limit": 64,
                "keep_percent": 0.6,
                "temperature": 0.7,
                "learning_rate": 5e-5,
                "allow_prototype_train": True,
            },
            "agentic",
            {"limit": 64, "learning_rate": 5e-5, "temperature": 0.7, "allow_prototype_train": True},
            {"reward_threshold", "samples_per_prompt", "task"},
        ),
    ],
)
def test_public_api_launch_is_mode_strict(payload, expected_method, expected_pairs, absent_keys, tmp_path):
    """Public launch payloads should forward only fields supported by each mode."""

    class FakeTrainingService:
        def __init__(self):
            self.calls = []

        async def launch_sft(self, **kwargs):
            self.calls.append(("sft", kwargs))
            return "job-sft"

        async def launch_raft(self, **kwargs):
            self.calls.append(("raft", kwargs))
            return "job-raft"

        async def launch_preference_train(self, **kwargs):
            self.calls.append((kwargs["mode"], kwargs))
            return f"job-{kwargs['mode']}"

        async def launch_modality_train(self, **kwargs):
            self.calls.append((kwargs["modality"], kwargs))
            return f"job-{kwargs['modality']}"

    fake_training = FakeTrainingService()
    service = PublicApiService(
        app_state=AppState(),
        results_service=ResultsService(base_path=tmp_path),
        readiness_service=_fake_readiness_service(),
        training_service=fake_training,
        base_path=tmp_path,
    )
    service.get_run_detail = lambda run_identifier, **kwargs: {"id": run_identifier}

    result = asyncio.run(service.launch_training(payload))

    assert result["id"] == f"job-{expected_method}"
    assert fake_training.calls
    method_name, kwargs = fake_training.calls[0]
    assert method_name == expected_method
    for key, value in expected_pairs.items():
        assert kwargs[key] == value
    for key in absent_keys:
        assert key not in kwargs or kwargs[key] is None


def test_public_api_rejects_unsupported_mode_specific_fields(tmp_path):
    """Unsupported public training fields should raise 400-style ValueErrors."""

    class FakeTrainingService:
        calls = []

        def preflight_modality_train_launch(self, **kwargs):
            self.calls.append(kwargs)
            return TrainingLaunchPreflight(
                ok=True,
                errors=[],
                warnings=[],
                resolved_paths={"output_dir": kwargs["output_dir"]},
                suggested_fixes=[],
                quality_outlook={},
            )

    service = PublicApiService(
        app_state=AppState(),
        results_service=ResultsService(base_path=tmp_path),
        readiness_service=_fake_readiness_service(),
        training_service=FakeTrainingService(),
        base_path=tmp_path,
    )

    with pytest.raises(ValueError, match="reward_threshold"):
        service.preflight_training(
            {
                "mode": "reasoning",
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "dataset": "gsm8k",
                "output_dir": "models/reasoning_public_run",
                "cycles": 2,
                "limit": 64,
                "reward_threshold": 0.5,
            }
        )

    audio = service.preflight_training(
        {
            "mode": "audio",
            "model": "openai/whisper-tiny",
            "dataset": "librispeech",
            "output_dir": "models/audio_public_run",
            "cycles": 2,
            "samples_per_prompt": 3,
            "task": "asr",
            "limit": 24,
        }
    )
    assert audio["ok"] is True
    assert FakeTrainingService.calls[0]["limit"] == 24


@pytest.mark.parametrize(
    ("mode", "extra", "expected_tokens"),
    [
        ("dpo", {"beta": 0.1, "loss_type": "sigmoid", "reference_free": True}, ["dpo", "train", "--loss-type", "sigmoid", "--reference-free"]),
        ("orpo", {"beta": 0.1}, ["orpo", "train", "--beta", "0.1"]),
        ("rm", {}, ["rm", "train", "--batch-size", "4"]),
        ("grpo", {"verifier": "json_schema", "num_generations": 4, "beta": 0.04}, ["grpo", "train", "--verifier", "json_schema", "--num-generations", "4"]),
    ],
)
def test_training_service_builds_preference_method_commands(mode, extra, expected_tokens, tmp_path, monkeypatch):
    state = AppState()
    service = TrainingService(state)
    captured: dict[str, list[str]] = {}

    async def fake_launch(job_id, cmd, on_log=None, *, no_caffeinate=False):
        captured["cmd"] = cmd

    monkeypatch.setattr(service, "_launch_process_with_runtime_options", fake_launch)

    job_id = asyncio.run(
        service.launch_preference_train(
            mode=mode,
            model="Qwen/Qwen2.5-1.5B-Instruct",
            dataset="ultrafeedback" if mode != "grpo" else "gsm8k",
            output_dir=str(tmp_path / mode),
            epochs=1,
            batch_size=4 if mode == "rm" else 1,
            gradient_accumulation_steps=4 if mode == "rm" else 16,
            learning_rate=1e-5 if mode == "rm" else 5e-6,
            **extra,
        )
    )

    assert job_id
    cmd = captured["cmd"]
    for token in expected_tokens:
        assert token in cmd
    assert (tmp_path / mode / "launch_context.json").exists()


def test_public_train_client_uses_mode_specific_payloads_and_labels():
    """Public train client should build mode-safe payloads for every dashboard method."""
    train_client_source = Path("public_app/src/routes/train.tsx").read_text(encoding="utf-8")

    assert "buildLaunchPayload(config, workspace.data?.default_run_root)" in train_client_source
    assert 'if (c.modality === "sft")' in train_client_source
    assert 'if (c.modality === "raft")' in train_client_source
    for mode in ["dpo", "orpo", "rm", "grpo", "vlm", "audio", "reasoning", "agentic"]:
        assert f'"{mode}"' in train_client_source
    assert "DEFAULT_RAFT_PROMPTS" in train_client_source
    assert "RAFT_PROMPT_SOURCES" in train_client_source
    assert "resolveRaftPrompts(source)" in train_client_source
    assert "Prompt source" in train_client_source
    assert "stripEmpty" in train_client_source
    assert "Verifier toolchain" in train_client_source
    assert "GoalSection" in train_client_source
    assert "MethodSection" in train_client_source


def test_public_docs_reference_split_between_public_frontend_and_internal_console():
    docs_index = Path("docs/README.md").read_text(encoding="utf-8")
    public_doc = Path("website/hugo-docs/content/docs/reference/public-frontend.md").read_text(
        encoding="utf-8"
    )
    web_ui_doc = Path("website/hugo-docs/content/docs/reference/web-ui.md").read_text(
        encoding="utf-8"
    )

    assert "Public Frontend" in docs_index
    assert "internal ops/research console" in web_ui_doc
    assert "Remote v1 means one Halo Forge machine" in public_doc
    assert "The retired NiceGUI product surface is no longer the primary user workflow" in public_doc


def test_public_docs_stale_copy_and_local_hugo_links():
    docs_root = Path("website/hugo-docs/content")
    docs_text = "\n".join(
        path.read_text(encoding="utf-8") for path in docs_root.rglob("*.md")
    )
    readme = Path("README.md").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    dashboard_docs_route = Path("public_app/src/routes/docs.tsx").read_text(
        encoding="utf-8"
    )
    hugo_config = Path("website/hugo-docs/hugo.toml").read_text(encoding="utf-8")
    home_layout = Path("website/hugo-docs/layouts/index.html").read_text(
        encoding="utf-8"
    )
    header_layout = Path("website/hugo-docs/layouts/partials/header.html").read_text(
        encoding="utf-8"
    )
    download_doc = Path("website/hugo-docs/content/download.md").read_text(
        encoding="utf-8"
    )
    install_doc = (
        docs_root / "docs" / "getting-started" / "install-desktop.md"
    ).read_text(encoding="utf-8")
    public_frontend_doc = (
        docs_root / "docs" / "reference" / "public-frontend.md"
    ).read_text(encoding="utf-8")
    quickstart_doc = (
        docs_root / "docs" / "getting-started" / "quickstart.md"
    ).read_text(encoding="utf-8")

    assert "Next.js" not in docs_text
    assert "NEXT_PUBLIC_HALO_API_BASE" not in docs_text
    assert "8081" not in docs_text
    assert "github.com/professor-moody/halo-forge/tree/main/docs" not in readme
    assert "github.com/professor-moody/halo-forge/tree/main/docs" not in pyproject
    assert "github.com/professor-moody/halo-forge/tree/main/docs" not in dashboard_docs_route
    assert "https://halo-forge.io/docs/" in readme
    assert 'Documentation = "https://halo-forge.io/docs/"' in pyproject
    assert "https://halo-forge.io/docs/" in dashboard_docs_route
    assert 'version = "2.0.0-alpha-2"' in hugo_config
    assert 'href="/download/"' in header_layout
    assert "Install options" in home_layout
    assert "Desktop availability" in home_layout
    assert "Use Quick Start" in home_layout
    assert "checksummed release\nmanifest" in download_doc
    assert "Windows x86-64" in download_doc
    assert "SmartScreen" in download_doc
    assert "runtime version" in download_doc
    assert "Unsigned packages are preview artifacts" in download_doc
    assert "Stable releases never publish an unsigned macOS DMG" in download_doc
    assert "does not remove datasets" in download_doc
    assert "release manifest" in install_doc
    assert "Do not bypass Gatekeeper or Windows SmartScreen" in install_doc
    assert "Unsigned macOS Developer Preview" in install_doc
    assert "shasum -a 256 -c" in install_doc
    assert "support.apple.com/en-us/102445" in install_doc
    assert "Do not disable Gatekeeper" in install_doc
    assert "halo-forge serve --host 127.0.0.1 --port 8000" not in docs_text
    assert "halo-forge serve --host 0.0.0.0 --port 8000" not in docs_text
    assert "halo-forge serve-public" in public_frontend_doc
    assert "halo-forge serve-public --check" in public_frontend_doc
    assert "halo-forge dashboard" in public_frontend_doc
    assert "halo-forge app" in public_frontend_doc
    assert "npm run qa:visual" in public_frontend_doc
    assert "npm run dev -- --host 0.0.0.0" in public_frontend_doc
    assert "http://<workstation-host>:8000" in public_frontend_doc
    assert "halo-forge dashboard" in quickstart_doc
    assert "Use Your Own Data" in docs_text
    assert "Start keeps the model, dataset, sample count, and output path conservative." not in docs_text
    assert "Use in Start" not in docs_text
    for method in ["SFT", "RAFT", "DPO", "ORPO", "RM", "GRPO", "VLM", "audio", "reasoning", "agentic"]:
        assert method in docs_text
    assert "/docs/training-pipeline/methods/" in docs_text
    assert "/docs/reference/dashboard-training/" in docs_text

    missing: list[str] = []
    for path in docs_root.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\]\((/docs/[^)#?]*)(?:#[^)]+)?\)", text):
            target = match.group(1)
            if not _hugo_doc_target_exists(docs_root, target):
                missing.append(f"{path}:{target}")

    assert missing == []


def test_serve_public_check_prints_dashboard_launch_recipe(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "serve-public", "--check"])

    cli_mod.main()

    out = capsys.readouterr().out
    assert "halo-forge serve-public" in out
    assert "bind:        127.0.0.1:8000" in out
    assert "http://127.0.0.1:8000/api/public/health" in out
    assert "cd public_app && npm run dev" in out
    assert "http://127.0.0.1:3000" in out
    assert "remote auth: loopback bypass" in out
    assert "Dashboard API preflight OK" in out
    assert "No server started." in out


def test_cli_version_reports_alpha_identity(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "--version"])

    with pytest.raises(SystemExit) as exc:
        cli_mod.main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "halo-forge 2.0.0-alpha-2" in out
    assert "package 2.0.0a2" in out


def test_cli_auto_logging_uses_dashboard_log_root(tmp_path, monkeypatch):
    from halo_forge.cli import setup_auto_logging

    log_root = tmp_path / "halo-logs"
    monkeypatch.setenv("HALO_FORGE_LOG_DIR", str(log_root))

    log_path = setup_auto_logging("sft_train", quiet=True)
    try:
        print("hello from training")
    finally:
        sys.stdout.close()

    assert log_path.parent == log_root
    assert log_path.read_text(encoding="utf-8").strip() == "hello from training"


def test_training_service_launch_env_and_cwd_are_dashboard_writable(tmp_path, monkeypatch):
    app_dir = tmp_path / "app-data"
    work_dir = tmp_path / "work-dir"
    monkeypatch.setenv("HALO_FORGE_APP_DIR", str(app_dir))
    monkeypatch.setenv("HALO_FORGE_WORKDIR", str(work_dir))

    service = TrainingService(AppState())
    env = service._get_strix_halo_env()

    assert env["HALO_FORGE_APP_DIR"] == str(app_dir.resolve())
    assert env["HALO_FORGE_LOG_DIR"] == str(app_dir.resolve() / "logs")
    assert service._launch_working_directory() == work_dir.resolve()


def test_public_run_logs_find_dashboard_training_log(tmp_path):
    state = AppState()
    output_dir = tmp_path / "halo-runs" / "start-code"
    output_dir.mkdir(parents=True)
    job = state.create_job("sft", "SFT smoke", output_dir=output_dir)
    state.update_job_status(job.id, "failed", source="test", reason="smoke")
    log_path = output_dir / f"{job.id}_training.log"
    log_path.write_text("first\nTraceback boom\nlast\n", encoding="utf-8")

    service = PublicApiService(app_state=state, base_path=tmp_path / "readonly-app-cwd")

    logs = service.get_run_logs(job.id, tail=2)

    assert logs["available"] is True
    assert logs["log_path"] == str(log_path)
    assert logs["lines"] == ["Traceback boom", "last"]


def test_run_detail_exposes_plain_language_failure_summary(tmp_path):
    state = AppState()
    output_dir = tmp_path / "halo-runs" / "start-code"
    output_dir.mkdir(parents=True)
    job = state.create_job("sft", "SFT smoke", output_dir=output_dir)
    job.error_message = "OSError: [Errno 30] Read-only file system: 'logs'"
    log_path = output_dir / f"{job.id}_training.log"
    log_path.write_text(
        "launching\nOSError: [Errno 30] Read-only file system: 'logs'\n",
        encoding="utf-8",
    )
    state.update_job_status(job.id, "failed", source="test", reason="logging failed")

    service = PublicApiService(app_state=state, base_path=tmp_path / "readonly-app-cwd")
    detail = service.get_run_detail(job.id)

    failure = detail["failure_summary"]
    assert failure["kind"] == "logging_cwd"
    assert failure["headline"] == "Run could not create its log file"
    assert failure["log_path"] == str(log_path)
    assert "Read-only file system" in "\n".join(failure["log_tail"])
    assert failure["retry_route"] == "/train?mode=sft"


def test_training_failure_classifier_covers_common_operator_failures():
    from halo_forge.public_api.service import _classify_training_failure

    cases = {
        "Permission denied while creating /x": "unwritable_output",
        "Cannot access gated repo https://huggingface.co/meta/x Please log in": "gated_huggingface",
        "dataset file does not exist: prompts.jsonl": "missing_data",
        "ModuleNotFoundError: No module named 'mlx_lm'": "missing_dependency",
        "MPS backend out of memory": "out_of_memory",
        "rollout-gated. Re-run with --allow-prototype-train": "backend_unsupported",
        "halo-forge: error: argument command: invalid choice: 'from multiprocessing.resource_tracker import main;main(12)'": "runtime_packaging",
    }
    for text, kind in cases.items():
        assert _classify_training_failure(text)["kind"] == kind


def test_run_live_exposes_stage_metric_history_and_elapsed(tmp_path):
    state = AppState()
    output_dir = tmp_path / "runs" / "live"
    output_dir.mkdir(parents=True)
    job = state.create_job("sft", "SFT live", output_dir=output_dir)
    job.started_at = datetime.now(timezone.utc)
    job.stage_key = "training"
    job.stage_label = "Training"
    job.stage_message = "Updating model weights."
    job.stage_progress_percent = 42.0
    job.last_event = "loss=1.234 learning_rate=2e-05"
    job.artifact_state = "checkpoint"
    job.current_step = 3
    job.total_steps = 10
    job.latest_loss = 1.234
    job.metric_points.append(
        {
            "step": 3,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "train_loss": 1.234,
            "learning_rate": 2e-5,
            "grad_norm": 0.8,
        }
    )
    state.update_job_status(job.id, "running", source="test", reason="started")

    service = PublicApiService(app_state=state, base_path=tmp_path)
    live = service.get_run_live(job.id)

    assert live["stage"]["key"] == "training"
    assert live["stage"]["progress_percent"] == 42.0
    assert live["last_event"] == "loss=1.234 learning_rate=2e-05"
    assert live["elapsed_seconds"] >= 0
    assert live["eta_seconds"] is not None
    assert live["artifact_state"] == "checkpoint"
    assert live["metric_points"][0]["train_loss"] == 1.234
    assert live["metric_points"][0]["learning_rate"] == 2e-5


def test_dashboard_check_prints_single_command_app_recipe(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "dashboard", "--check"])

    cli_mod.main()

    out = capsys.readouterr().out
    assert "halo-forge dashboard" in out
    assert "workstation app" in out
    assert "bind:        127.0.0.1:8000" in out
    assert "open app:    http://127.0.0.1:8000" in out
    assert "http://127.0.0.1:8000/api/public/health" in out
    assert "remote auth: loopback bypass" in out
    assert "Dashboard preflight OK" in out
    assert "No server started." in out


def test_app_alias_check_prints_dashboard_recipe(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "app", "--check"])

    cli_mod.main()

    out = capsys.readouterr().out
    assert "halo-forge dashboard" in out
    assert "open app:    http://127.0.0.1:8000" in out
    assert "Dashboard preflight OK" in out


def test_serve_public_check_prints_remote_workstation_recipe(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(
        sys,
        "argv",
        ["halo-forge", "serve-public", "--host", "0.0.0.0", "--check"],
    )

    cli_mod.main()

    out = capsys.readouterr().out
    assert "remote auth: required" in out
    assert "cd public_app && npm run dev -- --host 0.0.0.0" in out
    assert "http://<workstation-host>:3000" in out
    assert "Dashboard API preflight OK" in out


def _hugo_doc_target_exists(content_root: Path, target: str) -> bool:
    slug = target.strip("/")
    if slug == "docs":
        return (content_root / "docs" / "_index.md").exists()
    rel = slug.removeprefix("docs/")
    return (
        (content_root / "docs" / f"{rel}.md").exists()
        or (content_root / "docs" / rel / "_index.md").exists()
    )
