#!/usr/bin/env python3
"""Regression coverage for the public API/view-model pivot."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from halo_forge.public_api.service import PublicApiService
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
    assert "Results view in progress" in results_source
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


def test_public_api_transport_exposes_public_workflows():
    api_source = Path("halo_forge/public_api/app.py").read_text(encoding="utf-8")

    assert "/dashboard" in api_source
    assert "/train/preflight" in api_source
    assert "/train/launch" in api_source
    assert "/runs/{run_id}/live" in api_source
    assert "/runs/{run_id}/guided-recovery" in api_source
    assert "/readiness" in api_source
    assert "/docs" in api_source


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
