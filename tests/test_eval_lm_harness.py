"""lm-evaluation-harness wrapper tests (Track V8).

Use the runner-injection knob on `run_lm_eval` so we never import
lm-eval (the CI runner doesn't have it).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# ---------- task group resolution ------------------------------------------


def test_curated_groups_include_canonical_benchmarks():
    from halo_forge.eval import list_curated_task_groups

    groups = list_curated_task_groups()
    assert {"core", "reasoning", "code", "instruction_following", "knowledge"}.issubset(
        set(groups)
    )
    assert "mmlu" in groups["core"]
    assert "humaneval" in groups["code"]
    assert "ifeval" in groups["instruction_following"]


def test_resolve_tasks_expands_curated_groups():
    from halo_forge.eval.lm_harness import _resolve_tasks

    expanded = _resolve_tasks(["core"])
    assert "mmlu" in expanded
    assert "gsm8k" in expanded


def test_resolve_tasks_passes_through_raw_names():
    from halo_forge.eval.lm_harness import _resolve_tasks

    expanded = _resolve_tasks(["mmlu_pro_law", "agieval_aqua_rat"])
    assert expanded == ["mmlu_pro_law", "agieval_aqua_rat"]


def test_resolve_tasks_dedupes_across_groups_and_names():
    from halo_forge.eval.lm_harness import _resolve_tasks

    # "core" includes mmlu; passing both "core" and "mmlu" must not duplicate.
    expanded = _resolve_tasks(["core", "mmlu"])
    assert expanded.count("mmlu") == 1


# ---------- primary metric selection ---------------------------------------


def test_primary_metric_uses_hint_when_present():
    from halo_forge.eval.lm_harness import _pick_primary_metric

    metrics = {"acc": 0.65, "acc_norm": 0.7, "acc_stderr": 0.01}
    key, val = _pick_primary_metric("mmlu", metrics)
    assert key == "acc"
    assert val == 0.65


def test_primary_metric_falls_back_for_unknown_task():
    from halo_forge.eval.lm_harness import _pick_primary_metric

    metrics = {"some_score": 0.42, "some_score_stderr": 0.01}
    key, val = _pick_primary_metric("invented_task", metrics)
    assert key == "some_score"
    assert val == 0.42


def test_primary_metric_skips_stderr_in_fallback():
    """Fallback path must not pick *_stderr metrics as primary."""
    from halo_forge.eval.lm_harness import _pick_primary_metric

    metrics = {"acc_stderr": 0.01, "acc": 0.85}
    key, val = _pick_primary_metric("invented_task", metrics)
    assert "stderr" not in key
    assert val == 0.85


# ---------- result projection ----------------------------------------------


def test_project_lm_eval_results_basic_shape():
    from halo_forge.eval.lm_harness import _project_lm_eval_results

    raw = {
        "results": {
            "mmlu": {"acc": 0.65, "acc_stderr": 0.01, "acc_norm": 0.66, "alias": "mmlu"},
            "gsm8k": {"exact_match": 0.42, "exact_match_stderr": 0.02},
        },
        "n-samples": {
            "mmlu": {"effective": 14000, "original": 14000},
            "gsm8k": {"effective": 1319, "original": 1319},
        },
    }
    out = _project_lm_eval_results(raw, ["mmlu", "gsm8k"])
    assert len(out) == 2
    by_task = {r.task: r for r in out}
    assert by_task["mmlu"].primary_metric == "acc"
    assert by_task["mmlu"].value == 0.65
    assert by_task["mmlu"].n_samples == 14000
    assert by_task["mmlu"].error is None
    # alias / version are not metrics; they should be filtered out.
    assert "alias" not in by_task["mmlu"].all_metrics
    # Stderr survives in all_metrics (it's a real number) but not as primary.
    assert "acc_stderr" in by_task["mmlu"].all_metrics


def test_project_lm_eval_results_marks_missing_tasks():
    from halo_forge.eval.lm_harness import _project_lm_eval_results

    raw = {"results": {"mmlu": {"acc": 0.65}}}
    out = _project_lm_eval_results(raw, ["mmlu", "gsm8k"])
    by_task = {r.task: r for r in out}
    assert by_task["gsm8k"].error == "task_not_in_results"
    assert by_task["gsm8k"].value == 0.0


# ---------- run_lm_eval (with runner injection) ----------------------------


def test_run_lm_eval_with_stub_runner():
    from halo_forge.eval import run_lm_eval

    def stub_runner(*, model, tasks, limit):
        # Mirror lm-eval's actual return shape.
        return {
            "results": {
                "mmlu": {"acc": 0.7, "acc_stderr": 0.01},
                "gsm8k": {"exact_match": 0.35, "exact_match_stderr": 0.02},
            },
            "n-samples": {
                "mmlu": {"effective": 100, "original": 14000},
                "gsm8k": {"effective": 100, "original": 1319},
            },
        }

    result = run_lm_eval(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        tasks=["mmlu", "gsm8k"],
        limit=100,
        runner=stub_runner,
    )
    assert result.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert result.n_tasks_completed == 2
    assert result.n_tasks_failed == 0
    assert result.average_score() == pytest.approx((0.7 + 0.35) / 2, rel=1e-9)


def test_run_lm_eval_writes_summary_to_output_dir(tmp_path: Path):
    from halo_forge.eval import run_lm_eval

    def stub_runner(*, model, tasks, limit):
        return {"results": {"mmlu": {"acc": 0.5}}, "n-samples": {}}

    out = tmp_path / "eval_out"
    run_lm_eval(
        model_name="x", tasks=["mmlu"], runner=stub_runner, output_dir=out,
    )
    summary_path = out / "lm_eval_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["model_name"] == "x"
    assert summary["task_results"][0]["task"] == "mmlu"


def test_run_lm_eval_empty_tasks_raises():
    from halo_forge.eval import run_lm_eval

    with pytest.raises(ValueError, match="at least one task"):
        run_lm_eval(model_name="x", tasks=[], runner=lambda **k: {})


def test_run_lm_eval_curated_group_expands_via_stub():
    """`tasks=["core"]` should be expanded before the runner is called
    so the stub sees the actual member tasks."""
    from halo_forge.eval import run_lm_eval

    seen_tasks = []

    def stub_runner(*, model, tasks, limit):
        seen_tasks.extend(tasks)
        return {"results": {t: {"acc": 0.5} for t in tasks}, "n-samples": {}}

    run_lm_eval(model_name="x", tasks=["core"], runner=stub_runner)
    assert "mmlu" in seen_tasks
    assert "gsm8k" in seen_tasks
    assert "core" not in seen_tasks  # group should have been expanded away


def test_run_lm_eval_average_score_is_none_for_all_failed():
    from halo_forge.eval.lm_harness import EvalResult, EvalTaskResult

    r = EvalResult(
        model_name="x", tasks=["a"], task_results=[
            EvalTaskResult(task="a", primary_metric="(missing)",
                           value=0.0, error="task_not_in_results"),
        ],
        n_tasks_completed=0, n_tasks_failed=1,
        duration_seconds=0.1, backend="hf",
    )
    assert r.average_score() is None


# ---------- CLI ------------------------------------------------------------


def test_cli_eval_help_registers(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "eval", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for token in ("--tasks", "--limit", "--backend", "--list-tasks"):
        assert token in out


def test_cli_eval_list_tasks_short_circuits(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "eval", "--list-tasks"])
    cli_mod.main()
    out = capsys.readouterr().out
    assert "core" in out
    assert "reasoning" in out
    assert "mmlu" in out


def test_cli_eval_missing_lm_eval_surfaces_clean_error(monkeypatch, capsys):
    """If lm-eval isn't installed, the user gets a one-line install
    hint instead of a stack trace."""
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(
        sys, "argv",
        ["halo-forge", "eval", "--model", "x", "--tasks", "mmlu", "--limit", "1"],
    )
    # Force the import to look like it failed.
    monkeypatch.setitem(sys.modules, "lm_eval", None)
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "lm-eval" in out or "lm_eval" in out
    assert "pip install" in out
