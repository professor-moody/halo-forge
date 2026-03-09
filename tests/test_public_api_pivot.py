#!/usr/bin/env python3
"""Regression coverage for the public API/view-model pivot."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
    api_helper = Path("public_app/lib/api.ts").read_text(encoding="utf-8")
    train_source = Path("public_app/app/train/page.tsx").read_text(encoding="utf-8")
    train_client_source = Path("public_app/app/train/train-client.tsx").read_text(encoding="utf-8")
    run_source = Path("public_app/app/runs/[id]/page.tsx").read_text(encoding="utf-8")
    run_client_source = Path("public_app/app/runs/[id]/run-client.tsx").read_text(encoding="utf-8")
    results_source = Path("public_app/app/results/page.tsx").read_text(encoding="utf-8")
    results_client_source = Path("public_app/app/results/results-client.tsx").read_text(encoding="utf-8")
    readiness_source = Path("public_app/app/readiness/page.tsx").read_text(encoding="utf-8")
    docs_source = Path("public_app/app/docs/page.tsx").read_text(encoding="utf-8")
    home_source = Path("public_app/app/page.tsx").read_text(encoding="utf-8")
    shell_source = Path("public_app/components/ui.tsx").read_text(encoding="utf-8")

    assert "export const API_BASE" in api_helper
    assert 'from "../lib/api"' in home_source
    assert "System summary" in home_source
    assert "Training platform" in shell_source
    assert "Run configuration" in train_client_source
    assert "Launch review" in train_client_source
    assert "Run monitor" in run_source
    assert "Run status" in run_client_source
    assert "Training outcomes" in results_client_source
    assert "Qualification matrix" in readiness_source
    assert "Documentation catalog" in docs_source
    assert "Training that stays understandable" not in home_source
    assert "Public workstation" not in shell_source
    assert "Launch workspace" not in train_client_source
    assert "Review quality" not in train_client_source


def test_public_api_transport_exposes_public_workflows():
    api_source = Path("halo_forge/public_api/app.py").read_text(encoding="utf-8")

    assert "/dashboard" in api_source
    assert "/train/preflight" in api_source
    assert "/train/launch" in api_source
    assert "/runs/{run_id}/live" in api_source
    assert "/runs/{run_id}/guided-recovery" in api_source
    assert "/readiness" in api_source
    assert "/docs" in api_source


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
    assert "halo-forge ui remains the internal ops/research console" in public_doc
