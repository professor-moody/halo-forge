#!/usr/bin/env python3
"""Phase 25 quickstart-first user flow and launch-hub regression tests."""

from __future__ import annotations

from pathlib import Path


def test_quickstart_presets_registry_contains_canonical_entries():
    """Quickstart preset registry should provide canonical starter presets."""
    source = Path("ui/services/quickstart_presets.py").read_text(encoding="utf-8")
    for key in (
        "sft_fast_local",
        "raft_safe_default",
        "vlm_tiny",
        "audio_whisper_tiny",
        "reasoning_small",
        "agentic_small",
        "code_smoke",
        "non_code_smoke",
        "optimize_int4_smoke",
        "benchmark_latency_smoke",
    ):
        assert f'"{key}"' in source


def test_training_benchmark_inference_default_to_quickstart_mode():
    """Primary workflow pages should default to quickstart and parse quickstart params."""
    training_source = Path("ui/pages/training.py").read_text(encoding="utf-8")
    assert 'self.ui_mode: str = "quickstart"' in training_source
    assert "ignored invalid training ui_mode query param" in training_source
    assert "ignored invalid training preset query param" in training_source
    assert "def _render_sft_quickstart_form(" in training_source

    benchmark_source = Path("ui/pages/benchmark.py").read_text(encoding="utf-8")
    assert 'self.ui_mode: str = "quickstart"' in benchmark_source
    assert "ignored invalid benchmark ui_mode query param" in benchmark_source
    assert "ignored invalid benchmark preset query param" in benchmark_source
    assert "def _render_quickstart_form(" in benchmark_source

    inference_source = Path("ui/pages/inference.py").read_text(encoding="utf-8")
    assert 'self.ui_mode: str = "quickstart"' in inference_source
    assert "ignored invalid inference ui_mode query param" in inference_source
    assert "ignored invalid inference preset query param" in inference_source
    assert "def _render_quickstart_form(" in inference_source


def test_workflow_pages_keep_probe_controls_off_primary_paths():
    """Primary workflow pages should not expose probe actions directly."""
    for rel_path in ("ui/pages/training.py", "ui/pages/benchmark.py", "ui/pages/inference.py"):
        source = Path(rel_path).read_text(encoding="utf-8")
        assert "Run Contract Probe" not in source
        assert "Run Setup Check (Advanced)" not in source
        assert "Run Live Probe" not in source
        assert "Generate Evidence" not in source
        assert "Advanced setup checks are available in Advanced Diagnostics Tools." in source


def test_dashboard_primary_ctas_deep_link_to_quickstart_routes():
    """Dashboard launch buttons should route users into quickstart flows."""
    source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "/training?mode=sft&ui_mode=quickstart&preset=sft_fast_local" in source
    assert "/benchmark?view=code&ui_mode=quickstart&preset=code_smoke" in source
    assert "/inference?mode=optimize&ui_mode=quickstart&preset=optimize_int4_smoke" in source


def test_launch_buttons_use_input_validation_contracts():
    """Launch controls should be tied to explicit input validation methods."""
    benchmark_source = Path("ui/pages/benchmark.py").read_text(encoding="utf-8")
    assert "def _validate_launch_inputs(" in benchmark_source
    assert "if not is_valid or self.is_running:" in benchmark_source

    inference_source = Path("ui/pages/inference.py").read_text(encoding="utf-8")
    assert "def _validate_launch_inputs(" in inference_source
    assert "if not is_valid or self.is_running:" in inference_source
