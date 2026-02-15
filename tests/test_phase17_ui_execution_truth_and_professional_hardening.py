#!/usr/bin/env python3
"""Phase 17 UI execution truth and professional hardening regression tests."""

from __future__ import annotations

import json
from pathlib import Path

from halo_forge.all_module_readiness import (
    build_all_module_readiness_report,
    validate_all_module,
    validate_all_module_readiness_payload,
)
from ui.services.ops_readiness_service import OpsReadinessService
from ui.services.results_service import ResultsService


def test_all_module_readiness_payload_supports_launch_blocking_extensions():
    """Schema should accept launch diagnostics and explainability fields."""
    report = build_all_module_readiness_report(module_entries={}, seed=42, source="script")
    payload = report.to_dict()
    errors = validate_all_module_readiness_payload(payload)
    assert errors == []

    sample = payload["modules"]["audio"]
    assert "launch_blocked" in sample
    assert "issue_class" in sample
    assert "action_hint" in sample
    assert "issue_code" in sample
    assert "severity" in sample
    assert "what_is_missing" in sample
    assert "fix_now" in sample
    assert "fix_options" in sample


def test_non_code_missing_history_is_warn_and_not_launch_blocked_by_default(tmp_path):
    """Missing historical artifacts should be warn-only in live (require_artifacts=False) mode."""
    entry = validate_all_module(
        module="audio",
        output_dir=tmp_path / "missing_audio_dir",
        seed=42,
        require_artifacts=False,
    )
    assert entry.status == "warn"
    assert entry.launch_blocked is False
    assert entry.issue_class == "evidence_gap"


def test_non_code_missing_history_blocks_in_strict_artifact_mode(tmp_path):
    """Strict artifact validation should fail and mark launch_blocked for missing required files."""
    entry = validate_all_module(
        module="audio",
        output_dir=tmp_path / "missing_audio_dir",
        seed=42,
        require_artifacts=True,
    )
    assert entry.status == "fail"
    assert entry.launch_blocked is True
    assert entry.issue_class in {"preflight_blocker", "contract_break"}


def test_ops_readiness_service_resolves_recent_evidence_roots(monkeypatch, tmp_path):
    """Effective output map should prefer recent artifacts over static defaults."""
    run_dir = tmp_path / "models" / "phase7d" / "audio_custom_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "modality": "audio",
                "model_name": "openai/whisper-tiny",
                "seed": 42,
                "run_id": "abc",
                "cycles_executed": 1,
                "total_train_steps_executed": 1,
                "weights_updated": True,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "launch_context.json").write_text(
        json.dumps(
            {
                "contract_version": 1,
                "job_type": "audio",
                "service": "training",
                "created_at": "2026-01-01T00:00:00",
                "source_ui_page": "/training",
                "command": ["python3", "-m", "halo_forge.cli", "audio", "train"],
                "args": {"model": "openai/whisper-tiny", "dataset": "librispeech"},
                "relaunch_capabilities": {
                    "can_relaunch": True,
                    "can_clone": True,
                    "can_resume_latest": True,
                },
            }
        ),
        encoding="utf-8",
    )

    benchmark_dir = tmp_path / "results" / "benchmarks" / "reasoning-fixture"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "dataset": "gsm8k",
                "domain": "reasoning",
                "accuracy": 0.5,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "ui.services.ops_readiness_service.get_results_service",
        lambda: ResultsService(base_path=tmp_path),
    )
    service = OpsReadinessService(base_path=tmp_path)
    output_map = service.resolve_effective_output_map(include_all_modules=True, force_refresh=True)

    assert output_map["audio"] == str(run_dir)
    assert output_map["benchmark_non_code"] == str(benchmark_dir)


def test_runtime_surface_includes_page_guard_and_non_blocking_copy():
    """UI app/pages should include guarded render wrapper and non-blocking readiness wording."""
    app_source = Path("ui/app.py").read_text(encoding="utf-8")
    assert "render_guarded_page" in app_source

    page_guard_source = Path("ui/components/page_guard.py").read_text(encoding="utf-8")
    assert "failed to render" in page_guard_source
    assert "traceback.print_exc" in page_guard_source

    training_source = Path("ui/pages/training.py").read_text(encoding="utf-8")
    assert "render_readiness_diagnostic_panel" in training_source
    assert "self._render_modality_readiness_banner(modality)" not in training_source

    inference_source = Path("ui/pages/inference.py").read_text(encoding="utf-8")
    assert "render_readiness_diagnostic_panel" in inference_source

    ops_service_source = Path("ui/services/ops_readiness_service.py").read_text(encoding="utf-8")
    assert "get_qualification_provenance" in ops_service_source
    assert "run_qualification_probe" in ops_service_source
    assert "get_bootstrap_provenance" in ops_service_source
    assert "run_bootstrap_probe" in ops_service_source

    diag_source = Path("ui/components/diagnostic_panel.py").read_text(encoding="utf-8")
    assert "Evidence missing (non-blocking)" in diag_source
    assert "Launch blocked:" in diag_source
