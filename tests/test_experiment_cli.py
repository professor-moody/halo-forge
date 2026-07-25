"""CLI contracts for durable experiment groups and workstation jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sys
from pathlib import Path

import pytest

from halo_forge import cli
from halo_forge.orchestration import ExperimentOrchestrationService
from halo_forge.run_db import get_database
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


def _run_cli(monkeypatch, capsys, *args: str):
    monkeypatch.setattr(sys, "argv", ["halo-forge", *args])
    cli.main()
    return json.loads(capsys.readouterr().out)


def _development_revision(database):
    suite = database.create_benchmark_suite(
        name="CLI development suite", purpose="development"
    )
    return database.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="cli-development-v1",
        items=[{"id": "one", "input": "hello", "expected": "world"}],
        primary_metric="accuracy",
        direction="maximize",
    )


def _complete(item_id: str, scheduler: WorkstationScheduler, result=None):
    item = scheduler.claim(work_item_id=item_id)
    assert item is not None
    completed = scheduler.complete(item, result=result)
    assert completed is not None


def test_sweep_cli_create_list_and_show_share_the_catalog(tmp_path, monkeypatch, capsys):
    database_path = tmp_path / "runs.db"
    database = get_database(str(database_path))
    revision = _development_revision(database)

    created = _run_cli(
        monkeypatch,
        capsys,
        "sweep",
        "create",
        "--database",
        str(database_path),
        "--json",
        "--name",
        "three seeds",
        "--kind",
        "repeat",
        "--trainer",
        "sft",
        "--suite",
        revision.id,
        "--model",
        "org/model@revision",
        "--dataset-version",
        "dataset-version-1",
        "--seeds",
        "7,8",
        "9",
        "--output-root",
        str(tmp_path / "runs"),
    )

    assert created["status"] == "queued"
    assert created["objective"]["suite_revision_id"] == revision.id
    assert created["dataset_bindings"] == [
        {
            "role": "train",
            "dataset_version_id": "dataset-version-1",
            "split": "train",
        }
    ]
    assert [run["seed"] for run in created["trials"][0]["runs"]] == [7, 8, 9]
    assert len(created["work_items"]) == 6

    listed = _run_cli(
        monkeypatch,
        capsys,
        "sweep",
        "list",
        "--database",
        str(database_path),
        "--json",
    )
    assert [row["id"] for row in listed] == [created["id"]]

    shown = _run_cli(
        monkeypatch,
        capsys,
        "sweep",
        "show",
        created["id"],
        "--database",
        str(database_path),
        "--json",
    )
    assert shown["resolved_launch_config"]["model"] == "org/model@revision"
    assert shown["id"] == created["id"]


def test_sweep_create_loads_yaml_and_direct_flags_override_it(
    tmp_path, monkeypatch, capsys
):
    database_path = tmp_path / "runs.db"
    database = get_database(str(database_path))
    revision = _development_revision(database)
    spec_path = tmp_path / "sweep.yaml"
    spec_path.write_text(
        """
name: yaml sweep
kind: sweep
trainer_mode: sft
development_suite_revision_id: placeholder
base_config:
  model: old/model
search_space:
  learning_rate:
    kind: choice
    values: [0.0001, 0.0002]
n_trials: 2
seeds: [10]
sampler: grid
""".strip(),
        encoding="utf-8",
    )

    created = _run_cli(
        monkeypatch,
        capsys,
        "sweep",
        "create",
        "--database",
        str(database_path),
        "--json",
        "--spec",
        str(spec_path),
        "--suite",
        revision.id,
        "--model",
        "new/model",
    )

    assert created["kind"] == "sweep"
    assert created["resolved_launch_config"]["model"] == "new/model"
    assert len(created["trials"]) == 2


def test_fork_best_emits_exact_config_and_parent_run_without_launching(
    tmp_path, monkeypatch, capsys
):
    database_path = tmp_path / "runs.db"
    database = get_database(str(database_path))
    scheduler = WorkstationScheduler(database, worker_id="test", capacity_probe=_available_capacity)
    service = ExperimentOrchestrationService(database, scheduler=scheduler)
    revision = _development_revision(database)
    created = service.create_group_from_payload(
        {
            "name": "parent sweep",
            "kind": "sweep",
            "trainer_mode": "sft",
            "base_config": {"model": "org/model", "learning_rate": 1e-4},
            "search_space": {"lora_r": [8, 16]},
            "n_trials": 2,
            "sampler": "grid",
            "seeds": [21, 22],
            "development_suite_revision_id": revision.id,
        }
    )
    scores = {
        (0, 21): 0.4,
        (0, 22): 0.5,
        (1, 21): 0.8,
        (1, 22): 0.9,
    }
    expected_parent_run_id = None
    for trial in created["trials"]:
        for run in trial["runs"]:
            training = next(item for item in run["work_items"] if item["kind"] == "training")
            evaluation = next(item for item in run["work_items"] if item["kind"] == "evaluation")
            _complete(training["id"], scheduler)
            value = scores[(trial["ordinal"], run["seed"])]
            _complete(evaluation["id"], scheduler, {"metrics": {"accuracy": value}})
            if value == 0.9:
                expected_parent_run_id = run["run_id"]

    proposal = _run_cli(
        monkeypatch,
        capsys,
        "sweep",
        "fork-best",
        created["id"],
        "--database",
        str(database_path),
        "--json",
        "--seeds",
        "90",
        "91",
    )

    assert proposal["launch_started"] is False
    assert proposal["parent_group_id"] == created["id"]
    assert proposal["parent_run_id"] == expected_parent_run_id
    assert proposal["parent_context"]["objective_value"] == pytest.approx(0.9)
    assert proposal["resolved_launch_config"]["lora_r"] == 16
    assert proposal["resolved_launch_config"]["seed"] == 22
    assert proposal["seeds"] == [90, 91]
    assert len(database.list_run_groups()) == 1


def test_jobs_cli_inspects_blockers_cancels_retries_and_runs_once(
    tmp_path, monkeypatch, capsys
):
    database_path = tmp_path / "runs.db"
    database = get_database(str(database_path))
    scheduler = WorkstationScheduler(database, worker_id="test", capacity_probe=_available_capacity)
    item = scheduler.enqueue(
        kind="test",
        resource_class="cpu",
        launch_spec={"command": [sys.executable, "-c", "print('ok')"]},
    )

    shown = _run_cli(
        monkeypatch,
        capsys,
        "jobs",
        "show",
        item.id,
        "--database",
        str(database_path),
        "--json",
    )
    assert shown["queue_position"] == 1
    assert shown["blockers"]["dependencies"] == []

    cancelled = _run_cli(
        monkeypatch,
        capsys,
        "jobs",
        "cancel",
        item.id,
        "--database",
        str(database_path),
        "--json",
    )
    assert cancelled["status"] == "cancelled"

    retried = _run_cli(
        monkeypatch,
        capsys,
        "jobs",
        "retry",
        item.id,
        "--database",
        str(database_path),
        "--json",
    )
    assert retried["status"] == "queued"

    worker_result = _run_cli(
        monkeypatch,
        capsys,
        "jobs",
        "worker",
        "--once",
        "--database",
        str(database_path),
        "--json",
    )
    assert worker_result["completed_work_item"]["id"] == item.id
    assert worker_result["completed_work_item"]["status"] == "completed"


@pytest.mark.parametrize("mode", ["sft", "dpo", "orpo", "rm", "grpo"])
def test_hf_training_cli_exposes_bounded_max_steps(mode, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["halo-forge", mode, "train", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert "--max-steps" in capsys.readouterr().out


def test_sft_cli_threads_max_steps_into_the_trainer_config(monkeypatch):
    captured = {}

    def fake_get_sft_trainer(config):
        captured["config"] = config

        class StubTrainer:
            def train(self, **_kwargs):
                return {"summary": "stub"}

        return StubTrainer()

    monkeypatch.setattr(
        "halo_forge.sft._dispatch.get_sft_trainer", fake_get_sft_trainer
    )
    monkeypatch.setattr(
        cli, "_print_completed_training_summary", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "sft",
            "train",
            "--dataset",
            "codealpaca",
            "--model",
            "tiny/model",
            "--max-steps",
            "7",
        ],
    )

    cli.main()

    assert captured["config"].max_steps == 7
