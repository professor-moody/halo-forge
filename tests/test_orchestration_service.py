"""Persistence/queue contracts for the Lab v3 orchestration facade."""

from __future__ import annotations

from datetime import datetime, timezone
import sys

import pytest

from halo_forge.orchestration import OrchestrationService
from halo_forge.run_db import RunDatabase
from halo_forge.workstation_jobs import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
    WorkstationScheduler,
)

GIB = 1024**3


def _available_capacity(_path):
    return WorkstationCapacity(
        sampled_at=datetime.now(timezone.utc),
        disk=DiskCapacity(
            path="/tmp/halo-forge-tests",
            total_bytes=200 * GIB,
            used_bytes=100 * GIB,
            free_bytes=100 * GIB,
        ),
        memory=MemoryCapacity(
            total_bytes=32 * GIB,
            used_bytes=8 * GIB,
            available_bytes=24 * GIB,
        ),
    )


def _scheduler(db: RunDatabase) -> WorkstationScheduler:
    return WorkstationScheduler(db, worker_id="test", capacity_probe=_available_capacity)


def _suite(db: RunDatabase, *, purpose: str = "development", name: str = "suite"):
    suite = db.create_benchmark_suite(name=name, purpose=purpose)
    return db.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash=f"{name}-hash",
        items=[{"id": "one", "input": "hello", "expected": "world"}],
        primary_metric="accuracy",
        direction="maximize",
    )


def _finish(scheduler: WorkstationScheduler, item_id: str, result=None):
    item = scheduler.claim(work_item_id=item_id)
    assert item is not None
    completed = scheduler.complete(item, result=result)
    assert completed is not None
    return completed


def _complete_group(service: OrchestrationService, values):
    detail = service.get_run_group_detail(values["group_id"], reconcile=False)
    for trial in detail["trials"]:
        for run in trial["runs"]:
            training = next(item for item in run["work_items"] if item["kind"] == "training")
            evaluation = next(item for item in run["work_items"] if item["kind"] == "evaluation")
            _finish(service.scheduler, training["id"])
            _finish(
                service.scheduler,
                evaluation["id"],
                {"metrics": {"accuracy": values[run["seed"]]}},
            )


def test_create_repeat_pins_objective_materializes_ids_and_queues_dependencies(tmp_path):
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    service = OrchestrationService(db, scheduler=scheduler)
    development = _suite(db)
    detail = service.create_group_from_payload(
        {
            "name": "repeat baseline",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {
                "model": "org/model",
                "output_root": str(tmp_path),
            },
            "seeds": [7, 8, 9],
            "development_suite_revision_id": development.id,
            "dataset_bindings": [
                {"role": "train", "dataset_version_id": "version-1", "split": "train"}
            ],
        }
    )

    assert detail["objective"] == {
        "suite_revision_id": development.id,
        "metric": "accuracy",
        "direction": "maximize",
    }
    assert detail["status"] == "queued"
    assert len(detail["trials"]) == 1
    assert len(detail["trials"][0]["runs"]) == 3
    assert len({run["run_id"] for run in detail["trials"][0]["runs"]}) == 3
    assert len(detail["work_items"]) == 6

    run = detail["trials"][0]["runs"][0]
    training = next(item for item in run["work_items"] if item["kind"] == "training")
    evaluation = next(item for item in run["work_items"] if item["kind"] == "evaluation")
    assert training["canonical_run_id"] == run["run_id"]
    assert training["launch_spec"]["command"][:4] == [
        sys.executable,
        "-m",
        "halo_forge.cli",
        "sft",
    ]
    assert training["launch_spec"]["env"]["HALO_FORGE_RUN_ID"] == run["run_id"]
    assert training["launch_spec"]["output_dir"].endswith(run["run_id"])
    assert "train=version-1:train" in training["launch_spec"]["command"]
    dependencies = db.list_work_item_dependencies(evaluation["id"])
    assert [row.depends_on_work_item_id for row in dependencies] == [training["id"]]
    assert evaluation["launch_spec"]["command"][:4] == [
        sys.executable,
        "-m",
        "halo_forge.cli",
        "eval",
    ]


def test_creation_enforces_suite_purpose_metric_and_capability_gating():
    db = RunDatabase(":memory:")
    service = OrchestrationService(db, scheduler=_scheduler(db))
    development = _suite(db, name="dev")
    holdout = _suite(db, purpose="holdout", name="holdout")

    with pytest.raises(ValueError, match="primary metric"):
        service.create_group_from_payload(
            {
                "name": "wrong metric",
                "kind": "repeat",
                "trainer_mode": "sft",
                "base_config": {},
                "metric": "loss",
                "development_suite_revision_id": development.id,
            }
        )
    with pytest.raises(ValueError, match="development suite"):
        service.create_group_from_payload(
            {
                "name": "wrong purpose",
                "kind": "repeat",
                "trainer_mode": "sft",
                "base_config": {},
                "development_suite_revision_id": holdout.id,
            }
        )
    with pytest.raises(ValueError, match="unavailable"):
        service.create_group_from_payload(
            {
                "name": "mlx cannot gate",
                "kind": "sweep",
                "trainer_mode": "dpo",
                "base_config": {"backend": "mlx"},
                "search_space": {"learning_rate": [1e-5, 2e-5, 3e-5]},
                "n_trials": 3,
                "pruning": {"enabled": True, "budgets": [3, 9]},
                "development_suite_revision_id": development.id,
            }
        )

    created = service.create_group_from_payload(
        {
            "name": "separate holdout",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {},
            "seeds": [1],
            "development_suite_revision_id": development.id,
            "holdout_suite_revision_id": holdout.id,
        }
    )
    assert created["holdout_suite_revision_id"] == holdout.id
    assert {
        item["launch_spec"].get("suite_revision_id")
        for item in created["work_items"]
        if item["kind"] == "evaluation"
    } == {development.id}


def test_reconcile_cohort_best_compare_cancel_and_resume():
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    service = OrchestrationService(db, scheduler=scheduler)
    development = _suite(db)

    left = service.create_group_from_payload(
        {
            "name": "left",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {},
            "seeds": [10, 11],
            "development_suite_revision_id": development.id,
        }
    )
    _complete_group(
        service,
        {"group_id": left["id"], 10: 0.6, 11: 0.8},
    )
    reconciled = service.get_run_group_detail(left["id"])
    assert reconciled["status"] == "completed"
    assert reconciled["best_trial"]["objective_value"] == pytest.approx(0.7)
    assert reconciled["best_trial"]["standard_deviation"] == pytest.approx(0.1)
    clone = service.build_fork_best_payload(left["id"], seeds=[90, 91, 92])
    assert clone["kind"] == "repeat"
    assert clone["parent_group_id"] == left["id"]
    assert clone["source_trial_id"] == reconciled["best_trial"]["trial_id"]
    assert clone["seeds"] == [90, 91, 92]
    forked = service.fork_best(left["id"], seeds=[99])
    assert forked["parent_group_id"] == left["id"]
    assert forked["sampler_state"]["fork_context"]["source_trial_id"] == clone["source_trial_id"]

    right = service.create_group_from_payload(
        {
            "name": "right",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {},
            "seeds": [20],
            "development_suite_revision_id": development.id,
        }
    )
    _complete_group(service, {"group_id": right["id"], 20: 0.9})
    comparison = service.compare_run_groups(left["id"], right["id"])
    assert comparison["right_minus_left"] == pytest.approx(0.2)
    assert comparison["winner"] == right["id"]

    queued = service.create_group_from_payload(
        {
            "name": "queued",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {},
            "seeds": [30],
            "development_suite_revision_id": development.id,
        }
    )
    cancelled = service.cancel_run_group(queued["id"])
    assert cancelled["status"] == "cancelled"
    assert all(item["status"] == "cancelled" for item in cancelled["work_items"])
    resumed = service.resume_run_group(queued["id"])
    assert resumed["status"] == "queued"
    assert len(resumed["retried_work_item_ids"]) == 2


def test_synchronous_halving_waits_for_all_trials_then_queues_one_promoted_segment():
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    service = OrchestrationService(db, scheduler=scheduler)
    development = _suite(db)
    created = service.create_group_from_payload(
        {
            "name": "halving",
            "kind": "sweep",
            "trainer_mode": "sft",
            "base_config": {"backend": "hf"},
            "search_space": {"learning_rate": [0.1, 0.2, 0.3]},
            "n_trials": 3,
            "sampler": "grid",
            "seeds": [5],
            "pruning": {
                "enabled": True,
                "reduction_factor": 3,
                "budgets": [3, 9],
            },
            "development_suite_revision_id": development.id,
        }
    )
    scores = [0.1, 0.9, 0.4]
    for trial, score in zip(created["trials"], scores):
        run = trial["runs"][0]
        training = next(item for item in run["work_items"] if item["kind"] == "training")
        evaluation = next(item for item in run["work_items"] if item["kind"] == "evaluation")
        assert training["launch_spec"]["command"][-2:] == ["--max-steps", "3"]
        assert "--learning-rate" in training["launch_spec"]["command"]
        assert training["launch_spec"]["command_transport"]["status"] == "ready"
        _finish(scheduler, training["id"])
        _finish(scheduler, evaluation["id"], {"metrics": {"accuracy": score}})

    decision = service.advance_successive_halving(created["id"], rung_index=0)
    assert decision["ready"] is True
    assert len(decision["promoted_trial_keys"]) == 1
    assert len(decision["pruned_trial_keys"]) == 2
    assert len(decision["queued_work_item_ids"]) == 2
    promoted = db.get_run_group_trial(decision["promoted_trial_keys"][0])
    assert promoted is not None
    assert promoted.sampled_config["learning_rate"] == 0.2
    promoted_run = db.list_trial_runs(promoted.id)[0]
    segments = db.list_trial_segments(promoted_run.id)
    assert [(row.start_value, row.end_value) for row in segments] == [(0, 3), (3, 9)]
    assert segments[0].decision == "continue"


def test_terminal_event_halving_advances_ready_rung_exactly_once():
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    service = OrchestrationService(db, scheduler=scheduler)
    development = _suite(db)
    created = service.create_group_from_payload(
        {
            "name": "event-driven-halving",
            "kind": "sweep",
            "trainer_mode": "sft",
            "base_config": {"backend": "hf"},
            "search_space": {"learning_rate": [0.1, 0.2, 0.3]},
            "n_trials": 3,
            "sampler": "grid",
            "seeds": [5],
            "pruning": {
                "enabled": True,
                "reduction_factor": 3,
                "budgets": [3, 9],
            },
            "development_suite_revision_id": development.id,
        }
    )

    outcomes = []
    for trial, score in zip(created["trials"], [0.1, 0.9, 0.4]):
        run = trial["runs"][0]
        training = next(item for item in run["work_items"] if item["kind"] == "training")
        evaluation = next(item for item in run["work_items"] if item["kind"] == "evaluation")
        _finish(scheduler, training["id"])
        _finish(scheduler, evaluation["id"], {"metrics": {"accuracy": score}})
        outcomes.append(service.advance_ready_successive_halving(created["id"]))

    assert [value["advanced"] for value in outcomes] == [False, False, True]
    assert len(outcomes[-1]["queued_work_item_ids"]) == 2
    count_after_advance = len(db.list_work_items(limit=1000))
    repeated = service.advance_ready_successive_halving(created["id"])
    assert repeated["advanced"] is False
    assert len(db.list_work_items(limit=1000)) == count_after_advance
