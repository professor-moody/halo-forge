#!/usr/bin/env python3
"""Phase 7B modality E2E/runtime-surface regression tests."""

import json
from pathlib import Path

from ui.services.results_service import ResultsService


def test_cli_test_level_exposes_modality_suite():
    """CLI test command should expose and dispatch the modality level."""
    source = Path("halo_forge/cli.py").read_text(encoding="utf-8")
    assert "choices=['smoke', 'standard', 'full', 'modality']" in source
    assert 'elif args.level == "modality":' in source
    assert "runner.run_modality(" in source


def test_modality_fixture_pack_exists_and_is_parseable_jsonl():
    """Deterministic modality fixture files should exist and parse as JSONL."""
    fixture_dir = Path("tests/fixtures/modality")
    required = {
        "vlm_samples.jsonl",
        "audio_samples.jsonl",
        "reasoning_samples.jsonl",
        "agentic_samples.jsonl",
    }
    assert fixture_dir.exists()
    present = {path.name for path in fixture_dir.glob("*.jsonl")}
    assert required.issubset(present)

    for fixture_name in required:
        fixture_path = fixture_dir / fixture_name
        lines = [line for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines, f"{fixture_name} should contain at least one sample"
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)


def test_results_service_training_summaries_feed_dashboard_series(tmp_path):
    """ResultsService should parse modality training summaries for dashboard charts."""
    run_dir = tmp_path / "outputs" / "reasoning_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "modality": "reasoning",
                "model_name": "Qwen/Qwen2.5-7B-Instruct",
                "seed": 7,
                "cycles_executed": 2,
                "total_train_steps_executed": 5,
                "weights_updated": True,
                "cycles": [
                    {"cycle": 0, "train_loss": 0.21},
                    {"cycle": 1, "train_loss": 0.17},
                ],
            }
        ),
        encoding="utf-8",
    )

    service = ResultsService(base_path=tmp_path)
    runs = service.list_training_runs(force_refresh=True)
    assert len(runs) == 1
    assert runs[0].modality == "reasoning"
    assert runs[0].seed == 7

    summary = service.get_dashboard_training_summary(max_runs=3)
    assert summary["steps"] == ["1", "2"]
    assert len(summary["runs"]) == 1
    assert summary["runs"][0]["loss"] == [0.21, 0.17]


def test_dashboard_training_chart_uses_results_service_aggregation():
    """Dashboard training chart path should use canonical ResultsService aggregation."""
    source = Path("ui/pages/dashboard.py").read_text(encoding="utf-8")
    assert "def _load_recent_training_data" in source
    function_body = source.split("def _load_recent_training_data", 1)[1].split(
        "def _load_benchmark_data", 1
    )[0]
    assert "get_dashboard_training_summary" in function_body
    assert "glob(" not in function_body


def test_results_page_renders_training_runs_from_results_service():
    """Results page should render canonical training-run summaries."""
    source = Path("ui/pages/results.py").read_text(encoding="utf-8")
    assert "list_training_runs" in source
    assert "_render_training_runs_table" in source


def test_experimental_docs_use_seed_examples_and_keep_compatibility_note():
    """Public experimental docs should prefer seed-based examples and retain compat guidance."""
    content = Path("website/hugo-docs/content/docs/experimental.md").read_text(
        encoding="utf-8"
    )
    assert content.count("--seed 42") >= 8
    assert "--allow-prototype-train" in content
    assert content.count("--allow-prototype-train") <= 2


def test_command_index_documents_modality_test_level():
    """Public command index should list the modality test level contract."""
    content = Path("website/hugo-docs/content/docs/reference/command-index.md").read_text(
        encoding="utf-8"
    )
    assert "halo-forge test --level modality" in content
    assert "deterministic modality fixture + smoke suite" in content
    assert "--baseline-file" in content
    assert "--write-baseline" in content
    assert "--compare-baseline" in content
