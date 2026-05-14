#!/usr/bin/env python3
"""Production-readiness qualification and docs accuracy regression tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from halo_forge.all_module_qualification import (
    TRAINING_MODULES,
    TrainingProductionReadinessContract,
    _load_training_production_readiness_contract,
    compute_all_module_qualification,
    validate_all_module_qualification,
    validate_all_module_qualification_payload,
)
from halo_forge.all_module_readiness import ALL_MODULES, AllModuleReadiness, build_all_module_readiness_report
from ui.services.dashboard_hub_service import DashboardHubService
from ui.services.ops_readiness_service import OpsReadinessService


def _fixture_output_map(root: Path) -> dict[str, str]:
    return {
        module: str(root / module) if module != "ui_ops" else str(Path.cwd())
        for module in ALL_MODULES
    }


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_summary(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_fixture_profile_marks_all_training_modalities_production_ready():
    """Deterministic training fixtures should pass the full production-readiness gate."""
    fixture_root = Path("tests/fixtures/all_modules/v1")
    report = compute_all_module_qualification(
        output_map=_fixture_output_map(fixture_root),
        seed=42,
        profile="fixture-v1",
        source="script",
    )
    payload = report.to_dict()
    assert validate_all_module_qualification_payload(payload) == []

    expected_metrics = {
        "sft": "eval_loss",
        "raft": "pass_at_1",
        "vlm": "avg_reward",
        "audio": "average_reward",
        "reasoning": "accuracy",
        "agentic": "success_rate",
    }
    for module, metric_name in expected_metrics.items():
        entry = report.modules[module]
        assert entry.status == "pass"
        assert entry.eval_available is True
        assert entry.eval_metric_name == metric_name
        assert entry.weights_updated is True
        assert entry.optimizer_steps >= 1
        assert entry.samples_kept >= 1
        assert entry.production_ready is True
        assert entry.readiness_tier == "production_ready"


def test_ui_ops_qualification_uses_public_app_contract():
    """UI ops qualification should track the React/FastAPI surface, not retired NiceGUI files."""
    result = validate_all_module_qualification(
        module="ui_ops",
        output_dir=Path.cwd(),
        seed=42,
        require_artifacts=True,
        profile="fixture-v1",
    )

    assert result.status == "pass"
    assert result.artifacts_ok is True
    evidence_text = json.dumps(result.evidence, sort_keys=True)
    assert "public_app" in evidence_text
    assert "halo_forge/public_api" in evidence_text
    assert "ui/app.py" not in evidence_text
    assert "ui/components/sidebar.py" not in evidence_text


def test_training_production_readiness_contract_files_exist_and_validate():
    """Each training modality should have a tracked production-readiness contract file."""
    for module in sorted(TRAINING_MODULES):
        contract = _load_training_production_readiness_contract(module)
        assert contract is not None
        assert isinstance(contract, TrainingProductionReadinessContract)
        assert contract.metric_name
        assert contract.expected_verdict == "pass"


def test_no_optimizer_updates_fail_production_readiness(tmp_path):
    """Artifact presence alone should not mark a run production-ready when weights never changed."""
    fixture_root = Path("tests/fixtures/all_modules/v1")
    copied = tmp_path / "all_modules"
    shutil.copytree(fixture_root, copied)

    summary_path = copied / "audio" / "training_summary.json"
    summary = _load_summary(summary_path)
    summary["weights_updated"] = False
    summary["effectiveness"]["verdict"] = "fail"
    summary["effectiveness"]["reasons"] = ["weights_not_updated"]
    summary["effectiveness"]["update_quality"]["weights_updated"] = False
    _write_summary(summary_path, summary)

    result = validate_all_module_qualification(
        module="audio",
        output_dir=copied / "audio",
        seed=42,
        require_artifacts=True,
        profile="fixture-v1",
    )
    assert result.status == "fail"
    assert result.production_ready is False
    assert result.readiness_tier == "qualified"
    assert any("weights were not updated" in error for error in result.errors)


def test_too_few_samples_fail_production_readiness(tmp_path):
    """Production readiness should fail when too few kept samples reach the optimizer."""
    fixture_root = Path("tests/fixtures/all_modules/v1")
    copied = tmp_path / "all_modules"
    shutil.copytree(fixture_root, copied)

    summary_path = copied / "sft" / "training_summary.json"
    summary = _load_summary(summary_path)
    summary["yield_diagnostics"]["stage_counts"]["kept"] = 1
    summary["effectiveness"]["verdict"] = "fail"
    summary["effectiveness"]["reasons"] = ["samples_kept_below_minimum"]
    summary["effectiveness"]["data_yield"]["samples_kept"] = 1
    summary["effectiveness"]["data_yield"]["keep_rate"] = 0.25
    summary["effectiveness"]["data_yield"]["min_samples_met"] = False
    _write_summary(summary_path, summary)

    result = validate_all_module_qualification(
        module="sft",
        output_dir=copied / "sft",
        seed=42,
        require_artifacts=True,
        profile="fixture-v1",
    )
    assert result.status == "fail"
    assert result.production_ready is False
    assert any("samples_kept below minimum" in error for error in result.errors)


def test_missing_resume_evidence_fails_production_readiness(tmp_path):
    """Cycle-based modalities should not qualify without resume-latest evidence."""
    fixture_root = Path("tests/fixtures/all_modules/v1")
    copied = tmp_path / "all_modules"
    shutil.copytree(fixture_root, copied)
    (copied / "raft" / "latest_checkpoint.json").unlink()

    result = validate_all_module_qualification(
        module="raft",
        output_dir=copied / "raft",
        seed=42,
        require_artifacts=True,
        profile="fixture-v1",
    )
    assert result.status == "fail"
    assert result.production_ready is False
    assert result.readiness_tier == "experimental"
    assert any("resume_latest checkpoint evidence missing" in error for error in result.errors)


def test_missing_eval_fails_production_readiness(tmp_path):
    """A run without post-train evaluation should not qualify as production-ready."""
    fixture_root = Path("tests/fixtures/all_modules/v1")
    copied = tmp_path / "all_modules"
    shutil.copytree(fixture_root, copied)

    summary_path = copied / "vlm" / "training_summary.json"
    summary = _load_summary(summary_path)
    summary["effectiveness"]["verdict"] = "warn"
    summary["effectiveness"]["reasons"] = ["evaluation_not_available"]
    summary["effectiveness"]["evaluation"]["status"] = "not_available"
    summary["effectiveness"]["evaluation"]["final_value"] = None
    summary["effectiveness"]["evaluation"]["delta"] = None
    _write_summary(summary_path, summary)

    result = validate_all_module_qualification(
        module="vlm",
        output_dir=copied / "vlm",
        seed=42,
        require_artifacts=True,
        profile="fixture-v1",
    )
    assert result.status == "fail"
    assert result.production_ready is False
    assert result.readiness_tier == "qualified"
    assert any("post-train evaluation missing" in error for error in result.errors)


def test_eval_regression_fails_production_readiness(tmp_path):
    """The deterministic gate should fail when eval regresses past the tracked baseline."""
    fixture_root = Path("tests/fixtures/all_modules/v1")
    copied = tmp_path / "all_modules"
    shutil.copytree(fixture_root, copied)

    summary_path = copied / "reasoning" / "training_summary.json"
    summary = _load_summary(summary_path)
    summary["effectiveness"]["verdict"] = "fail"
    summary["effectiveness"]["reasons"] = ["evaluation_regressed"]
    summary["effectiveness"]["evaluation"]["final_value"] = 0.8
    summary["effectiveness"]["evaluation"]["delta"] = -0.2
    summary["effectiveness"]["evaluation"]["regressed"] = True
    _write_summary(summary_path, summary)

    result = validate_all_module_qualification(
        module="reasoning",
        output_dir=copied / "reasoning",
        seed=42,
        require_artifacts=True,
        profile="fixture-v1",
    )
    assert result.status == "fail"
    assert result.production_ready is False
    assert any("evaluation regressed beyond tolerance" in error for error in result.errors)


class _FakeOpsReadinessService:
    def __init__(self, readiness_report, qualification_report):
        self._readiness_report = readiness_report
        self._qualification_report = qualification_report

    def get_effective_all_module_readiness(self, force_refresh: bool = False):
        return self._readiness_report

    def resolve_effective_output_map(self, include_all_modules: bool = True, force_refresh: bool = False):
        return {module: f"/tmp/{module}" for module in ALL_MODULES}

    def get_burnin_provenance(self, force_refresh: bool = False):
        return {"burnin_status": "warn"}

    def get_bootstrap_provenance(self, force_refresh: bool = False):
        return {"bootstrap_status": "pass"}

    def get_qualification_provenance(self, force_refresh: bool = False):
        return {
            "qualification_status": "warn",
            "qualification_report_present": True,
            "qualification_training_readiness_tier": "qualified",
        }

    def load_qualification_report(self, force_refresh: bool = False):
        return self._qualification_report

    def get_live_provenance(self, force_refresh: bool = False):
        return {"live_status": "pass"}


def test_dashboard_messages_use_training_readiness_tiers(monkeypatch):
    """Training cards should stop implying full readiness when only contract checks passed."""
    entries = {
        module: AllModuleReadiness(
            module=module,
            status="pass",
            errors=[],
            warnings=[],
            launch_blocked=False,
            issue_class="none",
        )
        for module in ALL_MODULES
    }
    readiness_report = build_all_module_readiness_report(
        module_entries=entries,
        seed=42,
        source="script",
    )

    fixture_root = Path("tests/fixtures/all_modules/v1")
    qualification_report = compute_all_module_qualification(
        output_map=_fixture_output_map(fixture_root),
        seed=42,
        profile="fixture-v1",
        source="script",
    )
    qualification_report.modules["sft"].production_ready = False
    qualification_report.modules["sft"].readiness_tier = "qualified"

    monkeypatch.setattr(
        "ui.services.dashboard_hub_service.get_ops_readiness_service",
        lambda: _FakeOpsReadinessService(readiness_report, qualification_report),
    )
    monkeypatch.setattr("ui.services.dashboard_hub_service.get_results_service", lambda: object())
    summary = DashboardHubService().build_summary(force_refresh=True)
    cards = [
        card
        for cards_in_group in summary.cards_by_group.values()
        for card in cards_in_group
    ]
    by_module = {card.module: card for card in cards}
    assert by_module["sft"].primary_message == "Launch-ready; full train+eval qualification still pending."
    assert by_module["audio"].primary_message == "Production-ready qualification passed."


def test_docs_and_quickstart_copy_match_readiness_tier_language():
    """Docs and UI copy should describe readiness tiers instead of overclaiming readiness."""
    experimental_doc = Path("website/hugo-docs/content/docs/experimental.md").read_text(
        encoding="utf-8"
    )
    assert "production_ready" in experimental_doc
    assert "qualified" in experimental_doc
    assert "Stable" not in experimental_doc

    docs_readme = Path("docs/README.md").read_text(encoding="utf-8")
    assert "readiness tiers and modality caveats" in docs_readme

    quickstart_source = Path("ui/services/quickstart_presets.py").read_text(encoding="utf-8")
    assert "qualification run" in quickstart_source
    assert "qualification-friendly" in quickstart_source


def test_qualification_provenance_reports_training_tier_summary(tmp_path):
    """OpsReadinessService should summarize training readiness tiers from qualification reports."""
    fixture_root = Path("tests/fixtures/all_modules/v1")
    report = compute_all_module_qualification(
        output_map=_fixture_output_map(fixture_root),
        seed=42,
        profile="fixture-v1",
        source="script",
    )
    report_path = tmp_path / "results" / "readiness" / "all_module_qualification.v1.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    service = OpsReadinessService(base_path=tmp_path, qualification_report_path=report_path)
    provenance = service.get_qualification_provenance(force_refresh=True)
    assert provenance["qualification_training_readiness_tier"] == "production_ready"
    assert provenance["qualification_training_production_ready_count"] == 6
    assert provenance["qualification_training_module_count"] == 6
