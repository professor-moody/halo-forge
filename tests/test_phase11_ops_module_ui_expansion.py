#!/usr/bin/env python3
"""Phase 11 ops-module UI expansion regression tests."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from halo_forge.ops_module_readiness import (
    OPS_MODULES,
    build_ops_readiness_report,
    validate_ops_readiness_payload,
)
from ui.services.benchmark_service import BenchmarkService, BenchmarkType
from ui.services.inference_service import InferenceService
from ui.services.ops_readiness_service import OpsReadinessService
from ui.state import AppState
from ui.feature_flags import get_ui_feature_flags


def test_ops_readiness_schema_validation_pass_and_fail():
    """Ops readiness schema validator should catch missing keys and accept valid payloads."""
    report = build_ops_readiness_report(module_entries={}, seed=42, source="script")
    payload = report.to_dict()
    errors = validate_ops_readiness_payload(payload)
    assert errors == []

    broken = dict(payload)
    broken.pop("modules")
    failed = validate_ops_readiness_payload(broken)
    assert any("missing top-level key: modules" in err for err in failed)


def test_ops_matrix_script_flags_and_strict_behavior(tmp_path):
    """Script should emit OPS_READY lines and enforce strict fail behavior."""
    script = Path("scripts/run_ops_module_matrix.py")
    assert script.exists()

    target_dir = tmp_path / "vlm_missing"
    report_file = tmp_path / "ops_readiness.json"

    non_strict = subprocess.run(
        [
            sys.executable,
            str(script),
            "--validate-module",
            f"vlm={target_dir}",
            "--write-report",
            "--report-file",
            str(report_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert non_strict.returncode == 0
    assert "OPS_READY module=vlm" in non_strict.stdout
    assert report_file.exists()

    strict = subprocess.run(
        [
            sys.executable,
            str(script),
            "--validate-module",
            f"vlm={target_dir}",
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict.returncode == 1
    assert "ERROR: failing modules:" in strict.stdout


def test_ops_matrix_fixture_pack_strict_passes_and_corruption_fails(tmp_path):
    """Fixture-pack strict mode should pass for valid fixtures and fail on corruption."""
    script = Path("scripts/run_ops_module_matrix.py")
    assert script.exists()

    strict_pass = subprocess.run(
        [
            sys.executable,
            str(script),
            "--fixture-pack",
            "v1",
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict_pass.returncode == 0
    assert "OPS_READY module=ui_ops" in strict_pass.stdout

    fixture_root = Path("tests/fixtures/ops_readiness/v1")
    corrupted_root = tmp_path / "corrupted_pack"
    shutil.copytree(fixture_root, corrupted_root)
    (corrupted_root / "vlm" / "launch_context.json").unlink()

    strict_fail = subprocess.run(
        [
            sys.executable,
            str(script),
            "--fixture-pack",
            str(corrupted_root),
            "--strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert strict_fail.returncode == 1
    assert "ERROR: failing modules:" in strict_fail.stdout


def test_app_routes_include_feature_flagged_expansion_pages():
    """UI app should register feature-flagged inference/advanced/research routes."""
    source = Path("ui/app.py").read_text(encoding="utf-8")
    assert "@ui.page('/inference')" in source
    assert "@ui.page('/benchmark-advanced')" in source
    assert "@ui.page('/research-hub')" in source
    assert "HALO_UI_ENABLE_INFERENCE_PAGE" in source
    assert "HALO_UI_ENABLE_BENCHMARK_ADVANCED_PAGE" in source
    assert "HALO_UI_ENABLE_RESEARCH_HUB_PAGE" in source
    assert "_render_feature_disabled" in source


def test_sidebar_wires_conditional_nav_items_for_new_pages():
    """Sidebar should include feature-gated nav items for new surfaces."""
    source = Path("ui/components/sidebar.py").read_text(encoding="utf-8")
    assert "get_ui_feature_flags" in source
    assert "/inference" in source
    assert "/benchmark-advanced" in source
    assert "/research-hub" in source


def test_feature_flags_default_on_and_kill_switch(monkeypatch):
    """New UI surfaces should be enabled by default with env kill-switch support."""
    for key in (
        "HALO_UI_ENABLE_INFERENCE_PAGE",
        "HALO_UI_ENABLE_BENCHMARK_ADVANCED_PAGE",
        "HALO_UI_ENABLE_RESEARCH_HUB_PAGE",
    ):
        monkeypatch.delenv(key, raising=False)
    default_flags = get_ui_feature_flags()
    assert default_flags.enable_inference_page is True
    assert default_flags.enable_benchmark_advanced_page is True
    assert default_flags.enable_research_hub_page is True

    monkeypatch.setenv("HALO_UI_ENABLE_RESEARCH_HUB_PAGE", "0")
    disabled_flags = get_ui_feature_flags()
    assert disabled_flags.enable_research_hub_page is False


def test_benchmark_service_supports_reasoning_type_command():
    """Benchmark service should support reasoning benchmark command construction."""
    service = BenchmarkService(AppState())
    cmd = service._build_command(
        model="Qwen/Qwen2.5-7B-Instruct",
        benchmark_type=BenchmarkType.REASONING,
        benchmark_name="gsm8k",
        limit=5,
        output_path="results/benchmarks/gsm8k/benchmark.json",
    )
    assert "reasoning" in cmd
    assert "benchmark" in cmd
    assert "--dataset" in cmd
    assert "gsm8k" in cmd


def test_inference_service_launch_command_contract():
    """Inference service command builder should route through inference optimize/benchmark."""
    service = InferenceService(AppState())
    optimize_cmd = service.build_optimize_command(
        model="Qwen/Qwen2.5-Coder-3B",
        target_precision="int4",
        target_latency=50.0,
        calibration_data=None,
        output_dir="models/optimized",
        dry_run=False,
    )
    assert optimize_cmd[:6] == [sys.executable, "-m", "halo_forge.cli", "inference", "optimize", "--model"]

    benchmark_cmd = service.build_benchmark_command(
        model="Qwen/Qwen2.5-Coder-3B",
        prompts=None,
        num_prompts=8,
        max_tokens=64,
        warmup=2,
        measure_memory=False,
    )
    assert benchmark_cmd[:5] == [sys.executable, "-m", "halo_forge.cli", "inference", "benchmark"]


def test_ops_readiness_service_fallbacks_on_corrupt_report(tmp_path):
    """Ops readiness service should fall back to live compute when persisted report is corrupt."""
    report_file = tmp_path / "results/readiness/ops_modules_readiness.v1.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("{not valid json", encoding="utf-8")

    service = OpsReadinessService(base_path=tmp_path)
    output_map = {module: str(tmp_path / "missing" / module) for module in OPS_MODULES}
    report = service.get_effective_readiness(output_map=output_map, force_refresh=True)
    assert report.source == "ui_live_compute"
    assert set(report.modules.keys()) == set(OPS_MODULES)


def test_ops_e2e_script_and_fixture_pack_exist():
    """Ops E2E reliability script should have a tracked deterministic fixture pack."""
    script = Path("scripts/run_ops_e2e_reliability.py")
    fixture_root = Path("tests/fixtures/ops_e2e/v1")
    assert script.exists()
    assert fixture_root.exists()
    assert (fixture_root / "vlm" / "launch_context.json").exists()
    assert (fixture_root / "benchmark" / "reasoning-fixture" / "launch_context.json").exists()
