#!/usr/bin/env python3
"""Phase 10 non-code modality UI readiness gate regression tests."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from halo_forge.modality_research import ArtifactValidationResult, NON_CODE_MODALITIES
from halo_forge.modality_readiness import (
    DEFAULT_READINESS_REPORT_FILE,
    apply_staleness_policy,
    build_readiness_report_from_validations,
    validate_readiness_payload,
    write_readiness_report,
)
from ui.services.modality_readiness_service import ModalityReadinessService


def _pass_validation(modality: str, output_dir: Path) -> ArtifactValidationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    return ArtifactValidationResult(
        modality=modality,
        output_dir=str(output_dir),
        ok=True,
        errors=[],
        warnings=[],
        evidence={
            "training_summary": str(output_dir / "training_summary.json"),
            "launch_context": str(output_dir / "launch_context.json"),
            "latest_checkpoint": str(output_dir / "latest_checkpoint.json"),
            "final_model": str(output_dir / "final_model"),
        },
    )


def test_readiness_report_schema_valid_and_status_mapping(tmp_path):
    """Readiness report builder should emit valid v1 contract payload and statuses."""
    validations = []
    for modality in NON_CODE_MODALITIES:
        validations.append(_pass_validation(modality, tmp_path / modality))
    validations[1] = ArtifactValidationResult(
        modality=NON_CODE_MODALITIES[1],
        output_dir=str(tmp_path / NON_CODE_MODALITIES[1]),
        ok=False,
        errors=["missing launch_context.json"],
        warnings=[],
        evidence=validations[1].evidence,
    )

    report = build_readiness_report_from_validations(validations, seed=42, source="script")
    payload = report.to_dict()
    errors = validate_readiness_payload(payload)
    assert errors == []
    assert report.modalities[NON_CODE_MODALITIES[0]].status == "pass"
    assert report.modalities[NON_CODE_MODALITIES[1]].status == "fail"


def test_readiness_staleness_marks_warn_without_discarding_report(tmp_path):
    """Stale reports should be marked stale and PASS entries should degrade to WARN."""
    validations = [_pass_validation(modality, tmp_path / modality) for modality in NON_CODE_MODALITIES]
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    report = build_readiness_report_from_validations(
        validations,
        seed=42,
        source="script",
    )
    report.generated_at = old_ts

    stale_report = apply_staleness_policy(report)
    assert stale_report.stale is True
    for modality in NON_CODE_MODALITIES:
        assert stale_report.modalities[modality].status == "warn"


def test_script_writes_readiness_report_and_fails_on_validation_failures(tmp_path):
    """Script should write readiness report and exit non-zero when fail statuses exist."""
    script = Path("scripts/run_non_code_modality_matrix.py")
    report_file = tmp_path / "readiness.json"

    # Only one modality target -> missing others should fail readiness contract
    failed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--validate-training",
            f"vlm={tmp_path / 'vlm'}",
            "--write-readiness-report",
            "--readiness-from-validation",
            "--readiness-report-file",
            str(report_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 1
    assert "READINESS modality=vlm" in failed.stdout
    assert report_file.exists()


def test_ui_service_prefers_persisted_report_and_falls_back_on_corruption(tmp_path):
    """UI readiness service should prefer canonical report and fall back to live compute when invalid."""
    report_file = tmp_path / DEFAULT_READINESS_REPORT_FILE
    report_file.parent.mkdir(parents=True, exist_ok=True)
    validations = [_pass_validation(modality, tmp_path / "models" / modality) for modality in NON_CODE_MODALITIES]
    report = build_readiness_report_from_validations(validations, seed=42, source="script")
    write_readiness_report(report_file, report)

    service = ModalityReadinessService(base_path=tmp_path)
    loaded = service.get_effective_readiness(force_refresh=True)
    assert loaded.source == "script"

    report_file.write_text("{bad json", encoding="utf-8")
    fallback = service.get_effective_readiness(
        output_map={modality: str(tmp_path / "missing" / modality) for modality in NON_CODE_MODALITIES},
        force_refresh=True,
    )
    assert fallback.source == "ui_live_compute"


def test_dashboard_and_training_surfaces_include_readiness_hooks():
    """Dashboard/training should include readiness UI wiring while keeping launches enabled."""
    dashboard_source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "get_dashboard_hub_service" in dashboard_source
    assert "_render_modality_readiness_summary" in dashboard_source
    assert "Refresh checks" in dashboard_source

    training_source = Path("ui/pages/training.py").read_text(encoding="utf-8")
    assert "get_ops_readiness_service" in training_source
    assert "_render_all_module_readiness_banner" in training_source
    assert "_render_launch_button(" in training_source
