"""Hyperparameter sweep tests (Track T10)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest


# ---------- distributions --------------------------------------------------


def test_uniform_samples_inside_bounds():
    from halo_forge.sweep import Uniform

    rng = random.Random(0)
    d = Uniform(low=0.0, high=10.0)
    for _ in range(50):
        v = d.sample(rng)
        assert 0.0 <= v <= 10.0


def test_log_uniform_samples_inside_bounds():
    from halo_forge.sweep import LogUniform

    rng = random.Random(0)
    d = LogUniform(low=1e-6, high=1e-3)
    samples = [d.sample(rng) for _ in range(200)]
    assert all(1e-6 <= s <= 1e-3 for s in samples)
    # Log-uniform shape: roughly half the samples should land below the geometric mean.
    geo_mean = (1e-6 * 1e-3) ** 0.5
    below = sum(1 for s in samples if s < geo_mean)
    assert 50 < below < 150  # tight enough to fail on a uniform-not-log bug


def test_log_uniform_validates_positive_bounds():
    from halo_forge.sweep import LogUniform

    with pytest.raises(ValueError):
        LogUniform(low=0.0, high=1.0)
    with pytest.raises(ValueError):
        LogUniform(low=1.0, high=0.5)


def test_choice_samples_from_values_only():
    from halo_forge.sweep import Choice

    rng = random.Random(0)
    d = Choice(values=[1, 2, 4, 8])
    for _ in range(20):
        assert d.sample(rng) in {1, 2, 4, 8}


def test_choice_requires_non_empty_values():
    from halo_forge.sweep import Choice

    with pytest.raises(ValueError):
        Choice(values=[])


# ---------- search space ---------------------------------------------------


def test_search_space_sample_returns_dict_with_all_params():
    from halo_forge.sweep import Choice, LogUniform, SearchSpace

    space = SearchSpace(params={
        "lr": LogUniform(1e-5, 1e-3),
        "batch": Choice([1, 2, 4]),
    })
    rng = random.Random(0)
    for _ in range(10):
        params = space.sample(rng)
        assert set(params) == {"lr", "batch"}
        assert 1e-5 <= params["lr"] <= 1e-3
        assert params["batch"] in {1, 2, 4}


def test_search_space_to_dict_describes_distributions():
    from halo_forge.sweep import Choice, LogUniform, SearchSpace, Uniform

    space = SearchSpace(params={
        "lr": LogUniform(1e-5, 1e-3),
        "batch": Choice([1, 2, 4]),
        "warmup": Uniform(0.0, 0.2),
    })
    d = space.to_dict()
    assert d["lr"] == {"kind": "log_uniform", "low": 1e-5, "high": 1e-3}
    assert d["batch"] == {"kind": "choice", "values": [1, 2, 4]}
    assert d["warmup"] == {"kind": "uniform", "low": 0.0, "high": 0.2}


# ---------- sweep runner ---------------------------------------------------


def _basic_config(**overrides):
    from halo_forge.sweep import Choice, LogUniform, SearchSpace, SweepConfig

    cfg = SweepConfig(
        name="test_sweep",
        search_space=SearchSpace(params={
            "lr": LogUniform(1e-5, 1e-3),
            "batch": Choice([1, 2, 4]),
        }),
        n_trials=5,
        metric="loss",
        direction="minimize",
        seed=42,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_run_sweep_runs_each_trial_once():
    from halo_forge.sweep import run_sweep

    cfg = _basic_config()
    seen_trial_ids = []
    seen_params = []

    def runner(trial_id, params):
        seen_trial_ids.append(trial_id)
        seen_params.append(params)
        return {"loss": 0.5}

    result = run_sweep(config=cfg, runner=runner)
    assert seen_trial_ids == [0, 1, 2, 3, 4]
    assert all("lr" in p and "batch" in p for p in seen_params)
    assert result.n_completed == 5
    assert result.n_failed == 0


def test_run_sweep_picks_best_minimizing_trial():
    from halo_forge.sweep import run_sweep

    cfg = _basic_config()
    losses = [0.5, 0.3, 0.7, 0.2, 0.4]

    def runner(trial_id, params):
        return {"loss": losses[trial_id]}

    result = run_sweep(config=cfg, runner=runner)
    assert result.best_trial_id == 3
    assert result.best_value == pytest.approx(0.2, rel=1e-9)


def test_run_sweep_picks_best_maximizing_trial():
    from halo_forge.sweep import run_sweep

    cfg = _basic_config(direction="maximize", metric="reward")
    rewards = [0.5, 0.9, 0.4, 0.7, 0.6]

    def runner(trial_id, params):
        return {"reward": rewards[trial_id]}

    result = run_sweep(config=cfg, runner=runner)
    assert result.best_trial_id == 1
    assert result.best_value == pytest.approx(0.9, rel=1e-9)


def test_run_sweep_handles_failing_trial():
    """A runner exception on one trial doesn't crash the sweep."""
    from halo_forge.sweep import run_sweep

    cfg = _basic_config()

    def runner(trial_id, params):
        if trial_id == 2:
            raise RuntimeError("OOM")
        return {"loss": 0.5}

    result = run_sweep(config=cfg, runner=runner)
    assert result.n_completed == 4
    assert result.n_failed == 1
    failed = [t for t in result.trials if t.failed]
    assert len(failed) == 1
    assert failed[0].trial_id == 2
    assert "OOM" in failed[0].error


def test_run_sweep_early_stops_after_no_improvement():
    """`early_stop_after=N` halts when the best value hasn't improved
    in N consecutive trials."""
    from halo_forge.sweep import run_sweep

    cfg = _basic_config(n_trials=20, early_stop_after=3)
    # First trial is best; never improved after.
    losses = [0.1] + [0.5] * 19

    def runner(trial_id, params):
        return {"loss": losses[trial_id]}

    result = run_sweep(config=cfg, runner=runner)
    # Should stop after trial 3 (first best + 3 no-improvement = 4 total).
    assert len(result.trials) <= 5
    assert result.best_trial_id == 0


def test_run_sweep_writes_trials_jsonl_and_summary(tmp_path: Path):
    from halo_forge.sweep import run_sweep

    cfg = _basic_config(output_dir=str(tmp_path))
    losses = [0.5, 0.3, 0.4, 0.2, 0.6]

    def runner(trial_id, params):
        return {"loss": losses[trial_id]}

    result = run_sweep(config=cfg, runner=runner)
    trials_path = tmp_path / "trials.jsonl"
    assert trials_path.exists()
    rows = [json.loads(l) for l in trials_path.read_text().splitlines() if l]
    assert len(rows) == 5
    summary_path = tmp_path / "sweep_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["best_trial_id"] == 3


def test_run_sweep_metric_missing_records_as_none():
    """A runner that returns metrics without the configured `metric` key
    must not crash; the trial just has primary_metric_value=None and
    can't be a candidate for best."""
    from halo_forge.sweep import run_sweep

    cfg = _basic_config()

    def runner(trial_id, params):
        if trial_id == 0:
            return {}  # missing 'loss'
        return {"loss": 0.5}

    result = run_sweep(config=cfg, runner=runner)
    assert result.trials[0].primary_metric_value is None
    # Best is one of the trials that did report.
    assert result.best_trial_id != 0


def test_unknown_sampler_raises():
    from halo_forge.sweep.runner import _build_sampler

    with pytest.raises(ValueError, match="Unknown sampler"):
        _build_sampler("franken", random.Random(0))


def test_grid_sampler_walks_choice_combinations():
    """Grid over discrete choices produces the cartesian product."""
    from halo_forge.sweep import Choice, SearchSpace
    from halo_forge.sweep.runner import _GridSampler

    space = SearchSpace(params={
        "a": Choice([1, 2]),
        "b": Choice(["x", "y", "z"]),
    })
    sampler = _GridSampler(random.Random(0))
    samples = [sampler.next_params(space, trial_id=i) for i in range(6)]
    pairs = {(s["a"], s["b"]) for s in samples}
    assert pairs == {(1, "x"), (1, "y"), (1, "z"), (2, "x"), (2, "y"), (2, "z")}


def test_tpe_falls_back_to_random_without_optuna(monkeypatch, caplog):
    """When optuna isn't importable, tpe sampler downgrades to random
    with a warning rather than crashing."""
    monkeypatch.setitem(sys.modules, "optuna", None)
    from halo_forge.sweep.runner import _build_sampler, _RandomSampler

    with caplog.at_level("WARNING"):
        sampler = _build_sampler("tpe", random.Random(0))
    assert isinstance(sampler, _RandomSampler)
    assert any("optuna" in rec.message.lower() for rec in caplog.records)


def test_sweep_config_to_dict_is_json_serializable():
    """SweepConfig.to_dict() must round-trip through json.dumps for the
    sweep_summary write path."""
    from halo_forge.sweep import Choice, LogUniform, SearchSpace, SweepConfig

    cfg = SweepConfig(
        name="x",
        search_space=SearchSpace(params={
            "lr": LogUniform(1e-5, 1e-3),
            "batch": Choice([1, 2]),
        }),
        n_trials=3,
    )
    d = cfg.to_dict()
    json.dumps(d)  # raises if anything's not serializable
    assert d["search_space"]["lr"]["kind"] == "log_uniform"


def test_is_better_minimize_and_maximize():
    from halo_forge.sweep import SearchSpace, SweepConfig

    cfg_min = SweepConfig(name="x", search_space=SearchSpace(), direction="minimize")
    assert cfg_min.is_better(0.3, None) is True   # any value beats no incumbent
    assert cfg_min.is_better(0.3, 0.5) is True
    assert cfg_min.is_better(0.5, 0.3) is False

    cfg_max = SweepConfig(name="x", search_space=SearchSpace(), direction="maximize")
    assert cfg_max.is_better(0.7, 0.5) is True
    assert cfg_max.is_better(0.5, 0.7) is False
