#!/usr/bin/env python3
"""Regression coverage for the public API/view-model pivot."""

from __future__ import annotations

import asyncio
import json
import re
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from halo_forge.public_api.service import PublicApiService
from halo_forge.public_api.serve_manager import ManagedServeProcess, ServeStartRequest
from ui.services.results_service import ResultsService
from ui.services.training_service import TrainingService
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

    assert 'const API_BASE = "/api/public"' in api_helper
    assert 'from "@/components/shell"' in home_source
    assert "SystemCard" in home_source
    assert "Topbar" in shell_source
    assert "Base model" in train_source
    assert "Launch summary" in train_source
    assert "MLX Ready" in start_source
    assert "MLX readiness" in train_source
    assert "accelerator" in train_source
    assert "Run detail view" in run_source
    assert 'label="STATUS"' in run_source
    assert "Run results" in results_source
    assert "Serve model" in results_source
    assert "No completed results yet" in results_source
    assert "First guided run" in docs_source
    assert "Remote workstation" in docs_source
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
    assert 'TOKEN_STORAGE_KEY = "halo-forge:api-token"' in api_helper
    assert 'headers.Authorization = `Bearer ${token}`' in stream_source
    assert 'reportAuthRequired({ source: "stream"' in stream_source


def test_public_frontend_friendly_workstation_contract():
    start_source = Path("public_app/src/routes/start.tsx").read_text(encoding="utf-8")
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

    assert "START_GOALS" in start_source
    assert '"code"' in start_source
    assert '"reasoning"' in start_source
    assert '"tool-use"' in start_source
    assert '"apple-silicon"' in start_source
    assert '"gsm8k_sft"' in start_source
    assert '"xlam_sft"' in start_source
    assert "Run started" in start_source
    assert "export type RunLive" in api_source
    assert "runLive:" in api_source
    assert "/events" in run_source
    assert "LiveSummary" in run_source
    assert "plainRunStatus" in run_source
    assert "Reconnecting to logs" in logs_source
    assert "Fits ${detectedBackend}" in models_source
    assert "showing all catalog models" in models_source
    assert "useServeStart" in models_source
    assert "Use in Start" in models_source
    assert "startGoalForModel" in models_source
    assert "export type ServeStatus" in api_source
    assert 'state: "idle" | "starting" | "running" | "unhealthy" | "exited" | string' in api_source
    assert "serveStart:" in api_source
    assert "ServeStatusPanel" in playground_source
    assert "Results files" in results_source
    assert "View logs" in results_source
    assert "Apple Neural Accelerators (experimental)" in overview_source
    assert "installChunkLoadRecovery" in main_source


def test_public_api_transport_exposes_public_workflows():
    api_source = Path("halo_forge/public_api/app.py").read_text(encoding="utf-8")

    assert "/dashboard" in api_source
    assert "/train/preflight" in api_source
    assert "/train/launch" in api_source
    assert "/runs/{run_id}/live" in api_source
    assert "/runs/{run_id}/guided-recovery" in api_source
    assert "/serve/status" in api_source
    assert "/serve/start" in api_source
    assert "/serve/stop" in api_source
    assert "/readiness" in api_source
    assert "/docs" in api_source
    assert "find_frontend_dist" in api_source
    assert "_mount_frontend" in api_source


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
                "healthy": self.running,
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
    assert service.serve_status()["model"] == "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
    with pytest.raises(ValueError, match="already being served"):
        service.serve_start({"model": "Qwen/Qwen2.5-1.5B-Instruct"})
    assert service.serve_logs()["lines"] == ["ready"]
    assert service.serve_stop()["running"] is False


def test_managed_serve_rejects_busy_port_before_spawn(tmp_path):
    manager = ManagedServeProcess(base_path=tmp_path, log_dir=tmp_path / "serve-logs")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
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


def test_desktop_tauri_foundation_contract():
    tauri_dir = Path("apps/desktop-tauri/src-tauri")
    config = (tauri_dir / "tauri.conf.json").read_text(encoding="utf-8")
    main_rs = (tauri_dir / "src" / "main.rs").read_text(encoding="utf-8")
    capabilities = (tauri_dir / "capabilities" / "default.json").read_text(encoding="utf-8")

    assert '"identifier": "ai.haloforge.desktop"' in config
    assert '"frontendDist": "../../../public_app/dist"' in config
    assert '"targets": ["app", "dmg", "appimage", "deb"]' in config
    assert '"externalBin": ["sidecars/halo-forge-runtime"]' in config
    assert (tauri_dir / "sidecars" / "halo-forge-runtime").exists()
    assert (tauri_dir / "sidecars" / "halo-forge-runtime-aarch64-apple-darwin").exists()
    assert (tauri_dir / "sidecars" / "halo-forge-runtime-x86_64-unknown-linux-gnu").exists()
    assert ".sidecar(\"halo-forge-runtime\")" in main_rs
    assert '"serve-public", "--host", "127.0.0.1", "--port", "8000"' in main_rs
    assert "GET /api/public/health HTTP/1.1" in main_rs
    assert "child.kill()" in main_rs
    assert '"shell:allow-spawn"' in capabilities


def test_ci_covers_public_dashboard_and_unsigned_desktop_builds():
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "public-dashboard-regression" in ci
    assert "tests/test_public_api_pivot.py" in ci
    assert "tests/test_model_catalog.py" in ci
    assert "npm run build" in ci
    assert "desktop-unsigned-build" in ci
    assert "macos-14" in ci
    assert "ubuntu-latest" in ci
    assert "libwebkit2gtk-4.1-dev" in ci
    assert "tauri build" in Path("apps/desktop-tauri/package.json").read_text(encoding="utf-8")


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
                "cycles": 2,
                "samples_per_prompt": 6,
                "keep_percent": 0.6,
                "reward_threshold": 0.5,
                "temperature": 0.7,
            },
            "raft",
            {"samples_per_prompt": 6, "reward_threshold": 0.5, "temperature": 0.7},
            {"dataset", "limit", "learning_rate"},
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
            },
            "vlm",
            {"limit": 24, "samples_per_prompt": 3, "reward_threshold": 0.5},
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
            },
            "audio",
            {"samples_per_prompt": 3, "reward_threshold": 0.5, "task": "asr"},
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
            },
            "reasoning",
            {"limit": 64, "learning_rate": 1e-5, "temperature": 0.7},
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
            },
            "agentic",
            {"limit": 64, "learning_rate": 5e-5, "temperature": 0.7},
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
        def preflight_modality_train_launch(self, **kwargs):
            raise AssertionError("unsupported payload should be rejected before service dispatch")

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

    with pytest.raises(ValueError, match="limit"):
        service.preflight_training(
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


def test_public_train_client_uses_mode_specific_payloads_and_labels():
    """Public train client should build mode-safe payloads and expose audio/VLM-specific labels."""
    train_client_source = Path("public_app/src/routes/train.tsx").read_text(encoding="utf-8")

    assert "buildLaunchPayload(config)" in train_client_source
    assert 'if (c.modality === "sft")' in train_client_source
    assert 'mode: "raft"' in train_client_source
    assert "DEFAULT_RAFT_PROMPTS" in train_client_source
    assert "RAFT_PROMPT_SOURCES" in train_client_source
    assert "resolveRaftPrompts(c.dataset)" in train_client_source
    assert "Prompt source selected" in train_client_source
    assert "stripEmpty" in train_client_source
    assert "VerifierSection" in train_client_source


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
    public_frontend_doc = (
        docs_root / "docs" / "reference" / "public-frontend.md"
    ).read_text(encoding="utf-8")
    quickstart_doc = (
        docs_root / "docs" / "getting-started" / "quickstart.md"
    ).read_text(encoding="utf-8")

    assert "Next.js" not in docs_text
    assert "NEXT_PUBLIC_HALO_API_BASE" not in docs_text
    assert "8081" not in docs_text
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
    assert "Start keeps the model, dataset, sample count, and output path conservative." in docs_text
    assert "Use in Start" in docs_text

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
