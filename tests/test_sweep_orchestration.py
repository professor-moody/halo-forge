"""Focused contracts for deterministic sweep/run-group orchestration."""

from __future__ import annotations

import json
import math
import random
import sys
from types import SimpleNamespace

import pytest

from halo_forge.orchestration import (
    CohortObservation,
    RunGroupSpec,
    SuccessiveHalvingConfig,
    aggregate_cohort,
    config_fingerprint,
    decide_successive_halving,
    expected_seeds_for_trials,
    materialize_trials,
    rank_cohort,
    resolve_trainer_execution_capability,
)
from halo_forge.sweep import Choice, LogUniform, SearchSpace, SweepConfig, Uniform


def test_search_space_round_trip_and_yaml_choice_shorthand():
    payload = {
        "learning_rate": {"kind": "log_uniform", "low": 1e-6, "high": 1e-3},
        "batch_size": [1, 2, 4],
        "warmup": {"type": "uniform", "low": 0, "high": 0.2},
    }
    space = SearchSpace.from_dict(payload)
    assert isinstance(space.params["learning_rate"], LogUniform)
    assert isinstance(space.params["batch_size"], Choice)
    assert isinstance(space.params["warmup"], Uniform)
    reparsed = SearchSpace.from_dict(space.to_dict())
    assert reparsed.to_dict() == space.to_dict()
    json.dumps(reparsed.to_dict())


@pytest.mark.parametrize(
    "payload, error",
    [
        ({"x": {"kind": "mystery"}}, "unknown distribution"),
        ({"x": {"kind": "choice", "values": []}}, "at least one"),
        ({"x": {"kind": "uniform", "low": 2, "high": 1}}, "high"),
    ],
)
def test_search_space_parser_rejects_invalid_definitions(payload, error):
    with pytest.raises((TypeError, ValueError), match=error):
        SearchSpace.from_dict(payload)


def test_sweep_config_round_trip_preserves_distribution_types():
    original = SweepConfig(
        name="learning-rate",
        search_space=SearchSpace(params={"lr": LogUniform(1e-6, 1e-3)}),
        direction="maximize",
        sampler="grid",
        n_trials=4,
        seed=19,
    )
    restored = SweepConfig.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    assert isinstance(restored.search_space.params["lr"], LogUniform)


def test_optuna_study_uses_sweep_metric_direction(monkeypatch):
    captured = {}

    class FakeTPESampler:
        def __init__(self, *, seed):
            captured["seed"] = seed

    def create_study(*, direction, sampler):
        captured["direction"] = direction
        captured["sampler"] = sampler
        return SimpleNamespace()

    fake_optuna = SimpleNamespace(
        samplers=SimpleNamespace(TPESampler=FakeTPESampler),
        create_study=create_study,
    )
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)
    from halo_forge.sweep.runner import _build_sampler

    sampler = _build_sampler("tpe", random.Random(9), direction="maximize")
    assert sampler.study is not None
    assert captured["direction"] == "maximize"


def test_config_fingerprint_ignores_only_runtime_identity():
    first = {
        "model": "org/model",
        "learning_rate": 0.001,
        "dataset_artifact_hash": "abc",
        "seed": 7,
        "run_id": "run-a",
        "output_dir": "/tmp/a",
    }
    second = {
        "output_dir": "/tmp/b",
        "run_id": "run-b",
        "seed": 99,
        "dataset_artifact_hash": "abc",
        "learning_rate": 0.001,
        "model": "org/model",
    }
    assert config_fingerprint(first) == config_fingerprint(second)
    changed = dict(second, dataset_artifact_hash="different")
    assert config_fingerprint(first) != config_fingerprint(changed)


def test_repeat_group_materializes_default_three_seeds():
    spec = RunGroupSpec(
        name="three seeds",
        kind="repeat",
        base_config={"model": "org/model", "learning_rate": 0.001},
        base_seed=100,
    )
    trials = materialize_trials(spec)
    assert len(trials) == 1
    assert trials[0].seeds == (100, 101, 102)
    runs = trials[0].materialize_runs()
    assert [run.seed for run in runs] == [100, 101, 102]
    assert len({run.run_fingerprint for run in runs}) == 3
    assert {run.config_fingerprint for run in runs} == {trials[0].config_fingerprint}


def test_repeat_and_sweep_spec_round_trip_with_defaults():
    repeat = RunGroupSpec.from_dict(
        {"name": "repeat", "kind": "repeat", "base_config": {"model": "m"}, "base_seed": 3}
    )
    assert repeat.seeds == (3, 4, 5)
    assert RunGroupSpec.from_dict(repeat.to_dict()).to_dict() == repeat.to_dict()

    sweep = RunGroupSpec.from_dict(
        {
            "name": "sweep",
            "kind": "sweep",
            "base_config": {"model": "m"},
            "search_space": {"lr": [0.001, 0.0001]},
            "n_trials": 2,
            "base_seed": 8,
        }
    )
    assert sweep.seeds == (8,)
    assert RunGroupSpec.from_dict(sweep.to_dict()).to_dict() == sweep.to_dict()


def test_run_group_spec_v1_shape_is_frozen_and_v2_adds_checkpoint_plan():
    v1 = RunGroupSpec(
        name="repeat",
        kind="repeat",
        base_config={"model": "m"},
        seeds=(3,),
    )
    assert v1.to_dict() == {
        "version": 1,
        "name": "repeat",
        "kind": "repeat",
        "base_config": {"model": "m"},
        "search_space": {},
        "n_trials": 1,
        "metric": "final_train_loss",
        "direction": "minimize",
        "sampler": "random",
        "base_seed": 42,
        "sampler_seed": 42,
        "seeds": [3],
        "pruning": {"enabled": False, "reduction_factor": 3, "budgets": []},
    }
    assert "checkpoint_policy_revision_id" not in v1.to_dict()
    assert "resolved_checkpoint_plan" not in v1.to_dict()

    plan = {
        "policy_revision_id": "policy-revision-1",
        "policy_hash": "abc",
        "trainer_mode": "sft",
        "unit": "step",
        "total_budget": 12,
        "boundaries": [4, 8, 12],
        "required_suite_revision_ids": ["suite-1"],
        "automatic_actions": True,
        "capability_notes": [],
        "content_hash": "def",
    }
    v2 = RunGroupSpec.from_dict(
        {
            "name": "adaptive repeat",
            "kind": "repeat",
            "base_config": {"model": "m"},
            "seeds": [7],
            "checkpoint_policy_revision_id": "policy-revision-1",
            "resolved_checkpoint_plan": plan,
        }
    )
    assert v2.version == 2
    assert v2.checkpoint_policy_revision_id == "policy-revision-1"
    assert v2.to_dict()["resolved_checkpoint_plan"] == plan
    assert RunGroupSpec.from_dict(v2.to_dict()).to_dict() == v2.to_dict()

    with pytest.raises(ValueError, match="version 2"):
        RunGroupSpec(
            name="invalid",
            kind="repeat",
            base_config={},
            checkpoint_policy_revision_id="policy-revision-1",
            version=1,
        )


def test_random_sweep_materialization_is_order_independent_and_repeatable():
    left = RunGroupSpec.from_dict(
        {
            "name": "s",
            "kind": "sweep",
            "base_config": {"model": "m"},
            "search_space": {"z": [1, 2], "a": ["x", "y"]},
            "n_trials": 8,
            "sampler_seed": 31,
        }
    )
    right = RunGroupSpec.from_dict(
        {
            "name": "s",
            "kind": "sweep",
            "base_config": {"model": "m"},
            "search_space": {"a": ["x", "y"], "z": [1, 2]},
            "n_trials": 8,
            "sampler_seed": 31,
        }
    )
    assert [trial.to_dict() for trial in materialize_trials(left)] == [
        trial.to_dict() for trial in materialize_trials(right)
    ]


def test_tpe_materialization_requires_persisted_adaptive_suggestions():
    spec = RunGroupSpec.from_dict(
        {
            "name": "adaptive",
            "kind": "sweep",
            "base_config": {"model": "m"},
            "search_space": {"lr": [0.1, 0.01]},
            "n_trials": 2,
            "sampler": "tpe",
        }
    )
    with pytest.raises(ValueError, match="adaptive"):
        materialize_trials(spec)
    trials = materialize_trials(spec, sampled_params=[{"lr": 0.1}, {"lr": 0.01}])
    assert [trial.params["lr"] for trial in trials] == [0.1, 0.01]


def test_group_validation_prevents_pruned_repeats_and_empty_sweeps():
    with pytest.raises(ValueError, match="cannot enable pruning"):
        RunGroupSpec(
            name="r",
            kind="repeat",
            base_config={},
            pruning=SuccessiveHalvingConfig(enabled=True),
        )
    with pytest.raises(ValueError, match="non-empty search space"):
        RunGroupSpec(name="s", kind="sweep", base_config={})


def test_cohort_aggregate_reports_stats_failures_and_missing_coverage():
    observations = [
        CohortObservation("a", 1, 0.2),
        CohortObservation("a", 2, 0.4),
        CohortObservation("b", 1, status="failed"),
    ]
    rows = aggregate_cohort(observations, {"a": [1, 2], "b": [1, 2]})
    by_key = {row.trial_key: row for row in rows}
    assert by_key["a"].eligible is True
    assert by_key["a"].mean == pytest.approx(0.3)
    assert by_key["a"].standard_deviation == pytest.approx(0.1)
    assert by_key["a"].minimum == 0.2
    assert by_key["a"].maximum == 0.4
    assert by_key["b"].coverage == 0.5
    assert by_key["b"].failure_count == 1
    assert by_key["b"].missing_seeds == (2,)
    assert by_key["b"].eligible is False


def test_cohort_ranking_respects_direction_and_stable_ties():
    observations = [
        CohortObservation("a", 1, 0.5),
        CohortObservation("b", 1, 0.7),
        CohortObservation("c", 1, 0.5),
    ]
    rows = aggregate_cohort(observations, {"a": [1], "b": [1], "c": [1]})
    assert [row.trial_key for row in rank_cohort(rows, direction="maximize")] == ["b", "a", "c"]
    assert [row.trial_key for row in rank_cohort(rows, direction="minimize")] == ["a", "c", "b"]


def _halving_rows(values, *, complete=True, failure_key=None):
    expected = {key: [1, 2] for key in values}
    observations = []
    for key, value in values.items():
        observations.append(CohortObservation(key, 1, value))
        if complete:
            if key == failure_key:
                observations.append(CohortObservation(key, 2, status="failed"))
            else:
                observations.append(CohortObservation(key, 2, value))
    return aggregate_cohort(observations, expected)


def test_successive_halving_is_opt_in_and_waits_for_all_seed_coverage():
    rows = _halving_rows({"a": 0.1, "b": 0.2}, complete=False)
    disabled = decide_successive_halving(
        SuccessiveHalvingConfig(), rows, direction="minimize", rung_index=0
    )
    assert disabled.ready is False
    assert disabled.reason == "disabled"

    waiting = decide_successive_halving(
        SuccessiveHalvingConfig(enabled=True),
        rows,
        direction="minimize",
        rung_index=0,
    )
    assert waiting.ready is False
    assert waiting.waiting_trial_keys == ("a", "b")


def test_successive_halving_factor_three_is_deterministic_and_direction_aware():
    values = {"a": 0.1, "b": 0.6, "c": 0.4, "d": 0.9, "e": 0.2, "f": 0.8}
    rows = _halving_rows(values)
    config = SuccessiveHalvingConfig(enabled=True, reduction_factor=3, budgets=(10, 30))
    decision = decide_successive_halving(config, rows, direction="maximize", rung_index=0)
    assert decision.ready is True
    assert decision.promoted_trial_keys == ("d", "f")
    assert decision.pruned_trial_keys == ("b", "c", "e", "a")
    assert decision.next_budget == 30


def test_successive_halving_ranks_terminal_failure_last_and_does_not_prune_at_final_budget():
    rows = _halving_rows({"a": 0.5, "b": 0.9, "c": 0.2}, failure_key="b")
    config = SuccessiveHalvingConfig(enabled=True, budgets=(10,))
    decision = decide_successive_halving(config, rows, direction="maximize", rung_index=0)
    assert decision.reason == "final_budget_complete"
    assert decision.ranking == ("a", "c", "b")
    assert decision.promoted_trial_keys == decision.ranking
    assert decision.pruned_trial_keys == ()


def test_expected_seeds_helper_uses_materialized_trial_keys():
    spec = RunGroupSpec(name="r", kind="repeat", base_config={}, base_seed=4)
    trials = materialize_trials(spec)
    assert expected_seeds_for_trials(trials) == {trials[0].trial_key: (4, 5, 6)}


@pytest.mark.parametrize("mode", ["sft", "dpo", "orpo", "rm", "grpo"])
def test_hf_trainers_declare_step_gating(mode):
    capability = resolve_trainer_execution_capability(mode, "torch_cuda")
    assert capability.backend_family == "hf"
    assert capability.segment_unit == "step"
    assert capability.supports_gated_execution is True
    assert capability.resume_cli_flag == "--resume"
    assert capability.checkpoint_pattern == "checkpoint-*"


@pytest.mark.parametrize("mode", ["raft", "vlm", "audio", "reasoning", "agentic"])
def test_cycle_trainers_declare_cycle_gating(mode):
    capability = resolve_trainer_execution_capability(mode, "cuda")
    assert capability.segment_unit == "cycle"
    assert capability.supports_gated_execution is True
    assert capability.checkpoint_pattern.startswith("cycle_")


@pytest.mark.parametrize("mode", ["dpo", "grpo"])
def test_mlx_preference_trainers_truthfully_override_hf_as_full_trial_only(mode):
    capability = resolve_trainer_execution_capability(mode, "mlx-lm")
    assert capability.backend_family == "mlx"
    assert capability.segment_unit == "full_trial"
    assert capability.supports_gated_execution is False
    assert "does not yet provide" in capability.reason


def test_unregistered_backend_combination_falls_back_to_full_trial_only():
    capability = resolve_trainer_execution_capability("sft", "mlx")
    assert capability.supports_gated_execution is False
    assert capability.segment_unit == "full_trial"
    assert capability.checkpoint_pattern is None


def test_successive_halving_config_validates_factor_and_budgets():
    with pytest.raises(ValueError, match="at least 2"):
        SuccessiveHalvingConfig(enabled=True, reduction_factor=1)
    with pytest.raises(ValueError, match="strictly increasing"):
        SuccessiveHalvingConfig(enabled=True, budgets=(30, 10))
    assert SuccessiveHalvingConfig(enabled=True).reduction_factor == 3
