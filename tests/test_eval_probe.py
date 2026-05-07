"""Mid-training probe tests (Track V9)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# ---------- scheduling ------------------------------------------------------


def test_should_run_fires_at_correct_cycles():
    from halo_forge.eval import MidTrainingProbe

    p = MidTrainingProbe(model_name="x", every_n_cycles=5)
    assert p.should_run(0) is True   # cycle 0 fires
    assert p.should_run(1) is False
    assert p.should_run(4) is False
    assert p.should_run(5) is True
    assert p.should_run(10) is True


def test_should_run_with_every_one():
    from halo_forge.eval import MidTrainingProbe

    p = MidTrainingProbe(model_name="x", every_n_cycles=1)
    for i in range(5):
        assert p.should_run(i) is True


def test_invalid_every_n_cycles_raises():
    from halo_forge.eval import MidTrainingProbe

    with pytest.raises(ValueError):
        MidTrainingProbe(model_name="x", every_n_cycles=0)
    with pytest.raises(ValueError):
        MidTrainingProbe(model_name="x", every_n_cycles=-1)


def test_invalid_tolerance_raises():
    from halo_forge.eval import MidTrainingProbe

    with pytest.raises(ValueError):
        MidTrainingProbe(model_name="x", regression_tolerance=-0.1)


# ---------- baseline IO -----------------------------------------------------


def test_baseline_save_and_load_roundtrip(tmp_path: Path):
    from halo_forge.eval import load_baseline, save_baseline

    bp = tmp_path / "baseline.json"
    save_baseline(bp, {"mmlu": 0.65, "gsm8k": 0.42})
    loaded = load_baseline(bp)
    assert loaded == {"mmlu": 0.65, "gsm8k": 0.42}


def test_load_baseline_missing_returns_none(tmp_path: Path):
    from halo_forge.eval import load_baseline

    assert load_baseline(tmp_path / "nope.json") is None


def test_load_baseline_unreadable_returns_none(tmp_path: Path, caplog):
    from halo_forge.eval import load_baseline

    bp = tmp_path / "b.json"
    bp.write_text("{not valid json")
    with caplog.at_level("WARNING"):
        assert load_baseline(bp) is None
    assert any("unreadable" in rec.message.lower() for rec in caplog.records)


def test_values_from_eval_result_with_dataclass():
    from halo_forge.eval import values_from_eval_result
    from halo_forge.eval.lm_harness import EvalResult, EvalTaskResult

    result = EvalResult(
        model_name="x", tasks=["a", "b"],
        task_results=[
            EvalTaskResult(task="a", primary_metric="acc", value=0.7),
            EvalTaskResult(task="b", primary_metric="acc", value=0.5,
                           error="task_not_in_results"),
        ],
        n_tasks_completed=1, n_tasks_failed=1,
        duration_seconds=1.0, backend="hf",
    )
    out = values_from_eval_result(result)
    # Errored tasks excluded.
    assert out == {"a": 0.7}


def test_values_from_eval_result_with_dict():
    from halo_forge.eval import values_from_eval_result

    raw = {
        "task_results": [
            {"task": "a", "primary_metric": "acc", "value": 0.6, "error": None},
            {"task": "b", "primary_metric": "acc", "value": 0.5, "error": "x"},
        ]
    }
    assert values_from_eval_result(raw) == {"a": 0.6}


# ---------- probe execution -------------------------------------------------


def _stub_runner_factory(scores_per_call):
    """Return a runner stub that returns the next scores on each call."""
    iterator = iter(scores_per_call)

    def runner(*, model, tasks, limit):
        next_scores = next(iterator)
        return {
            "results": {
                t: {"acc": next_scores.get(t, 0.0)} for t in tasks
            },
            "n-samples": {t: {"effective": limit or 100} for t in tasks},
        }
    return runner


def test_first_run_writes_baseline_and_reports_no_baseline(tmp_path: Path):
    from halo_forge.eval import MidTrainingProbe, load_baseline

    bp = tmp_path / "baseline.json"
    runner = _stub_runner_factory([{"mmlu": 0.65, "gsm8k": 0.42}])

    p = MidTrainingProbe(
        model_name="x",
        baseline_path=bp,
        tasks=["mmlu", "gsm8k"],
        runner=runner,
    )
    report = p.run(cycle=0)
    assert report.has_baseline is False
    assert report.has_regression is False
    assert all(d.delta is None for d in report.task_deltas)

    # Baseline written for next run.
    saved = load_baseline(bp)
    assert saved == {"mmlu": 0.65, "gsm8k": 0.42}


def test_second_run_diffs_against_baseline(tmp_path: Path):
    from halo_forge.eval import MidTrainingProbe, save_baseline

    bp = tmp_path / "baseline.json"
    save_baseline(bp, {"mmlu": 0.65, "gsm8k": 0.42})

    runner = _stub_runner_factory([{"mmlu": 0.70, "gsm8k": 0.40}])
    p = MidTrainingProbe(
        model_name="x", baseline_path=bp,
        tasks=["mmlu", "gsm8k"], runner=runner,
        regression_tolerance=0.05,
    )
    report = p.run(cycle=5)
    assert report.has_baseline is True
    by_task = {d.task: d for d in report.task_deltas}
    assert by_task["mmlu"].delta == pytest.approx(0.05, rel=1e-9)
    assert by_task["gsm8k"].delta == pytest.approx(-0.02, rel=1e-9)
    # Neither delta exceeds the tolerance — no regression.
    assert report.has_regression is False
    assert report.avg_delta == pytest.approx((0.05 - 0.02) / 2, rel=1e-9)


def test_regression_detected_when_drop_exceeds_tolerance(tmp_path: Path):
    from halo_forge.eval import MidTrainingProbe, save_baseline

    bp = tmp_path / "baseline.json"
    save_baseline(bp, {"mmlu": 0.65, "gsm8k": 0.42})

    # MMLU drops by 0.10 (>5%) → regression.
    runner = _stub_runner_factory([{"mmlu": 0.55, "gsm8k": 0.40}])
    p = MidTrainingProbe(
        model_name="x", baseline_path=bp,
        tasks=["mmlu", "gsm8k"], runner=runner,
        regression_tolerance=0.05,
    )
    report = p.run()
    assert report.has_regression is True
    assert report.regressed_tasks() == ["mmlu"]


def test_no_regression_when_tolerance_high_enough(tmp_path: Path):
    from halo_forge.eval import MidTrainingProbe, save_baseline

    bp = tmp_path / "baseline.json"
    save_baseline(bp, {"mmlu": 0.65})

    runner = _stub_runner_factory([{"mmlu": 0.55}])
    p = MidTrainingProbe(
        model_name="x", baseline_path=bp,
        tasks=["mmlu"], runner=runner,
        regression_tolerance=0.20,  # generous
    )
    report = p.run()
    assert report.has_regression is False


def test_run_without_baseline_path_doesnt_persist(tmp_path: Path):
    """No baseline_path → probe runs but writes nothing to disk."""
    from halo_forge.eval import MidTrainingProbe

    runner = _stub_runner_factory([{"mmlu": 0.7}])
    p = MidTrainingProbe(
        model_name="x", baseline_path=None,
        tasks=["mmlu"], runner=runner,
    )
    report = p.run()
    assert report.has_baseline is False


def test_probe_uses_default_task_set_when_omitted():
    from halo_forge.eval import DEFAULT_PROBE_TASKS, MidTrainingProbe

    p = MidTrainingProbe(model_name="x")
    assert p.tasks == list(DEFAULT_PROBE_TASKS)
    # The default set hits each capability axis at least once.
    assert "mmlu" in p.tasks  # knowledge
    assert "gsm8k" in p.tasks  # math
    assert "arc_challenge" in p.tasks  # reasoning


def test_report_to_dict_serializable():
    from halo_forge.eval import MidTrainingProbe

    runner = _stub_runner_factory([{"mmlu": 0.65}])
    p = MidTrainingProbe(model_name="x", tasks=["mmlu"], runner=runner)
    report = p.run(cycle=10, notes="after SFT")
    d = report.to_dict()
    # JSON-serializable round-trip.
    serialized = json.dumps(d)
    assert "mmlu" in serialized
    assert d["cycle"] == 10
    assert d["notes"] == "after SFT"


# ---------- CLI -------------------------------------------------------------


def test_cli_probe_help_registers(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "probe", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for token in ("--model", "--tasks", "--baseline", "--tolerance"):
        assert token in out
