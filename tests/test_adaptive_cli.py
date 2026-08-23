"""CLI parity for adaptive checkpoint and evidence operations."""

from __future__ import annotations

import json
import sys

from halo_forge import cli
from halo_forge.run_db import get_database


def _run_cli(monkeypatch, capsys, *args: str):
    monkeypatch.setattr(cli.sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(sys, "argv", ["halo-forge", *args])
    cli.main()
    return json.loads(capsys.readouterr().out)


def _development_revision(database):
    suite = database.create_benchmark_suite(name="Adaptive CLI development", purpose="development")
    return database.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="adaptive-cli-development-v1",
        items=[{"id": "example", "input": "hello", "expected": "world"}],
        primary_metric="accuracy",
        direction="maximize",
    )


def test_checkpoint_policy_cli_creates_lists_and_pins_a_run_group_plan(
    tmp_path, monkeypatch, capsys
):
    database_path = tmp_path / "runs.db"
    database = get_database(str(database_path))
    revision = _development_revision(database)

    policy = _run_cli(
        monkeypatch,
        capsys,
        "checkpoint-policy",
        "create",
        "--database",
        str(database_path),
        "--json",
        "--policy-id",
        "cli-periodic",
        "--name",
        "CLI periodic checks",
        "--suite-revision",
        revision.id,
        "--metric",
        "accuracy",
        "--direction",
        "maximize",
        "--schedule",
        '{"mode":"percentages","unit":"step","percentages":[0.5,1.0]}',
    )
    assert policy["id"]
    assert policy["schedule"]["unit"] == "step"

    listed = _run_cli(
        monkeypatch,
        capsys,
        "checkpoint-policy",
        "list",
        "--database",
        str(database_path),
        "--trainer",
        "sft",
        "--json",
    )
    assert [value["id"] for value in listed["items"]] == [policy["id"]]

    created = _run_cli(
        monkeypatch,
        capsys,
        "sweep",
        "create",
        "--database",
        str(database_path),
        "--json",
        "--name",
        "adaptive repeat",
        "--kind",
        "repeat",
        "--trainer",
        "sft",
        "--suite",
        revision.id,
        "--model",
        "local/model",
        "--max-steps",
        "100",
        "--checkpoint-policy",
        policy["id"],
        "--seeds",
        "7",
        "11",
    )
    assert created["checkpoint_policy_revision_id"] == policy["id"]
    assert created["resolved_checkpoint_plan"]["boundaries"] == [50, 100]
    assert len(created["work_items"]) == 4

    trajectory = _run_cli(
        monkeypatch,
        capsys,
        "sweep",
        "checkpoints",
        created["id"],
        "--database",
        str(database_path),
        "--json",
    )
    assert trajectory["run_group_id"] == created["id"]
    assert trajectory["resolved_checkpoint_plan"]["content_hash"]
    assert len(trajectory["runs"]) == 2


def test_checkpoint_policy_validate_is_non_persistent(tmp_path, monkeypatch, capsys):
    database_path = tmp_path / "runs.db"
    result = _run_cli(
        monkeypatch,
        capsys,
        "checkpoint-policy",
        "validate",
        "--database",
        str(database_path),
        "--json",
        "--policy-id",
        "validation-only",
        "--name",
        "Validation only",
        "--suite-revision",
        "development-revision",
        "--metric",
        "score",
        "--direction",
        "maximize",
        "--schedule",
        '{"mode":"interval","unit":"cycle","interval":2}',
    )
    assert result["valid"] is True
    assert result["policy"]["schedule"]["interval"] == 2
    database = get_database(str(database_path))
    assert database.list_checkpoint_policies() == []


def test_eval_history_and_drift_cli_use_distinct_compatible_evaluations(
    tmp_path, monkeypatch, capsys
):
    database_path = tmp_path / "runs.db"
    database = get_database(str(database_path))
    revision = _development_revision(database)
    for evaluation_id, value in (("history-old", 0.6), ("history-new", 0.8)):
        database.create_evaluation(
            suite_revision_id=revision.id,
            adapter_id="dataset_split",
            adapter_version="1",
            subject_type="checkpoint",
            subject_ref="run-history",
            subject_hash=f"hash-{evaluation_id}",
            reuse_key=f"reuse-{evaluation_id}",
            request={},
            evaluation_id=evaluation_id,
        )
        database.complete_evaluation(
            evaluation_id,
            metrics=[{"name": "accuracy", "value": value, "direction": "maximize"}],
            samples=[
                {
                    "suite_item_id": "example",
                    "record_id": "example",
                    "score": value,
                    "passed": True,
                    "valid": True,
                    "mineable": True,
                    "score_direction": "maximize",
                }
            ],
            result={"metrics": {"accuracy": value}},
            artifact_path=str(tmp_path / evaluation_id),
        )

    history = _run_cli(
        monkeypatch,
        capsys,
        "eval",
        "history",
        "--database",
        str(database_path),
        "--subject",
        "run-history",
    )
    assert [value["id"] for value in history["items"]] == [
        "history-new",
        "history-old",
    ]

    drift = _run_cli(
        monkeypatch,
        capsys,
        "eval",
        "drift",
        "--database",
        str(database_path),
        "--candidate",
        "history-new",
        "--practical-delta",
        "0.01",
    )
    assert drift["base_id"] == "history-old"
    assert drift["candidate_id"] == "history-new"
    assert drift["classification"] == "improved"
