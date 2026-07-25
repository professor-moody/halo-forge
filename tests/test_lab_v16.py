from __future__ import annotations

import json
from pathlib import Path

from halo_forge.lab_v11_v15 import FutureLabService
from halo_forge.public_api.service import PublicApiService
from halo_forge.replay import capture_manifest
from halo_forge.run_db import RunDatabase, RunRecord
from halo_forge.workstation_jobs import WorkstationScheduler, WorkstationWorker


def _services(tmp_path: Path) -> tuple[RunDatabase, FutureLabService, PublicApiService]:
    database = RunDatabase(tmp_path / "catalog.db")
    scheduler = WorkstationScheduler(database, worker_id="v16-test")
    future = FutureLabService(database, root=tmp_path)
    public = PublicApiService(
        database=database,
        workstation_scheduler=scheduler,
        future_lab=future,
        future_lab_storage_root=tmp_path,
        dataset_storage_root=tmp_path / "datasets",
        evaluation_storage_root=tmp_path / "evaluations",
        base_path=tmp_path,
    )
    return database, future, public


def test_v19_operational_columns_and_plain_guidance(tmp_path: Path) -> None:
    database, future, _ = _services(tmp_path)
    columns = {
        row["name"]
        for row in database._conn.execute(
            "PRAGMA table_info(training_outcome_assessments)"
        ).fetchall()
    }
    assert {
        "stage",
        "progress_json",
        "request_json",
        "resume_cursor_json",
        "cancel_requested",
    } <= columns
    database.upsert_run(
        RunRecord(
            run_id="proof-v16",
            modality="sft",
            status="completed",
            weights_updated=True,
            final_model_path=str(tmp_path / "model"),
            raw_json=json.dumps(
                {
                    "proof_run": True,
                    "scenario_revision_id": "instruction-sft@1",
                    "launch_config": {
                        "proof_run": True,
                        "scenario_revision_id": "instruction-sft@1",
                    },
                }
            ),
        )
    )
    prepared = future.prepare_outcome_assessment("proof-v16", {})
    guidance = future.outcome_guidance(prepared)
    assert guidance.display_status == "Checking training result"
    assert guidance.primary_action.label == "View progress"
    assert len(guidance.secondary_actions) <= 2
    manifest = capture_manifest(
        run_id="v16-replay",
        modality="sft",
        model_name="fixture/model",
        seed=42,
        config={},
        operational_completion_binding={
            "prepared_evaluation_batch_ids": ["batch-1"],
            "reviewed_decision_id": "decision-1",
        },
    )
    assert manifest.manifest_version == 14
    assert manifest.operational_completion["reviewed_decision_id"] == "decision-1"


def test_grounding_launch_is_durable_and_review_is_next(tmp_path: Path) -> None:
    _, future, public = _services(tmp_path)
    profile = future.create_grounding_profile({"name": "Guided grounded data"})
    revision = future.create_grounding_profile_revision(
        profile.id,
        {
            "task_type": "qa",
            "intended_destination": "training",
            "teacher": {"endpoint_type": "local", "model": "fixture"},
            "seed": 42,
        },
    )
    launched = public.launch_grounded_generation(
        revision.id,
        {
            "preset": "quick",
            "records": [
                {
                    "document_id": "doc-1",
                    "source_ref": "guide.md",
                    "text": "Dataset versions are immutable.",
                }
            ],
        },
    )
    assert launched["status"] == "queued"
    terminal = WorkstationWorker(public._scheduler()).run_once(
        work_item_id=launched["work_item_id"]
    )
    assert terminal is not None and terminal.status == "completed"
    batch = public.get_grounded_generation(launched["id"])
    assert batch is not None
    assert batch["status"] == "completed"
    assert batch["accepted_count"] == 1
    proposal = public.create_grounding_review_proposal(launched["id"])
    assert proposal["requires_explicit_queue_creation"] is True


def test_future_lab_cancel_and_retry_keep_domain_in_sync(tmp_path: Path) -> None:
    _, future, public = _services(tmp_path)
    profile = future.create_grounding_profile({"name": "Cancellation fixture"})
    revision = future.create_grounding_profile_revision(
        profile.id,
        {"task_type": "qa", "intended_destination": "training"},
    )
    launched = public.launch_grounded_generation(
        revision.id,
        {
            "preset": "quick",
            "records": [
                {
                    "document_id": "doc-1",
                    "source_ref": "fixture.md",
                    "text": "Cancellation remains explicit.",
                }
            ],
        },
    )
    cancelled = public.cancel_work_item(launched["work_item_id"])
    assert cancelled["status"] == "cancelled"
    assert public.get_grounded_generation(launched["id"])["status"] == "cancelled"
    retried = public.retry_work_item(
        launched["work_item_id"], reason="operator reviewed the source"
    )
    assert retried["status"] == "queued"
    assert public.get_grounded_generation(launched["id"])["status"] == "queued"


def test_check_training_result_prepares_matched_evaluations(tmp_path: Path) -> None:
    database, _, public = _services(tmp_path)
    model_path = tmp_path / "proof-model"
    model_path.mkdir()
    database.upsert_run(
        RunRecord(
            run_id="proof-outcome-v16",
            modality="sft",
            status="completed",
            weights_updated=True,
            final_model_path=str(model_path),
            raw_json=json.dumps(
                {
                    "proof_run": True,
                    "scenario_revision_id": "instruction-sft@1",
                    "launch_config": {
                        "proof_run": True,
                        "scenario_revision_id": "instruction-sft@1",
                        "model": "fixture/base-model",
                        "mode": "sft",
                    },
                }
            ),
        )
    )
    suite = public.create_benchmark_suite(
        {
            "name": "Proof development evidence",
            "purpose": "development",
            "items": [
                {
                    "id": "fixed-score",
                    "input": "prompt",
                    "expected": "answer",
                    "score": 0.75,
                    "passed": True,
                }
            ],
            "primary_metric": "score",
            "direction": "maximize",
        }
    )
    prepared = public.prepare_training_outcome(
        "proof-outcome-v16",
        {
            "suite_revision_id": suite["latest_revision"]["id"],
            "scenario_revision_id": "instruction-sft@1",
            "adapter_id": "dataset",
        },
    )
    assert prepared["status"] == "queued"
    assert (
        prepared["base_evaluation"]["suite_revision_id"]
        == prepared["proof_evaluation"]["suite_revision_id"]
    )
    worker = WorkstationWorker(public._scheduler())
    for _ in range(6):
        current = public.get_work_item(prepared["work_item_id"])
        if current and current["status"] == "completed":
            break
        worker.run_once()
    assessment = public.get_training_outcome(prepared["assessment"]["id"])
    assert assessment is not None
    assert assessment["status"] == "no_clear_change"
    guidance = public.get_actionable_guidance(
        "training_outcome", assessment["id"]
    )
    assert guidance["display_status"] == "Review the tradeoff"
    assert guidance["primary_action"]["label"] == "Review examples"


def test_environment_launch_and_replay_are_separate(tmp_path: Path) -> None:
    _, future, public = _services(tmp_path)
    environment = future.create_environment({"name": "Local task"})
    revision = future.create_environment_revision(
        environment.id,
        {
            "adapter_id": "state_machine",
            "initial_state": {"done": False},
            "transitions": {
                "finish": {
                    "state_delta": {"done": True},
                    "reward": 1,
                    "terminal": True,
                }
            },
            "max_steps": 2,
        },
    )
    permissions = public.get_environment_permission_preview(revision.id)
    assert permissions["external_writes"] is False
    suite = future.create_episode_suite(
        {"name": "Local development", "purpose": "development"}
    )
    suite_revision = future.create_episode_suite_revision(
        suite["id"],
        {
            "environment_revision_id": revision.id,
            "items": [
                {
                    "id": "finish",
                    "goal": "Finish",
                    "expected_state": {"done": True},
                }
            ],
            "max_steps": 2,
        },
    )
    launched = public.launch_agent_episode(
        suite_revision.id,
        {
            "suite_item_id": "finish",
            "subject_type": "recorded_plan",
            "subject_ref": "reviewed-actions",
            "actions": [{"name": "finish"}],
        },
    )
    terminal = WorkstationWorker(public._scheduler()).run_once(
        work_item_id=launched["work_item_id"]
    )
    assert terminal is not None and terminal.status == "completed"
    episode = public.get_agent_episode(launched["id"])
    assert episode is not None and episode["metrics"]["task_success"] == 1.0
    replay = public.replay_agent_episode(launched["id"])
    assert replay["valid"] is True


def test_study_launch_creates_real_matched_run_groups(tmp_path: Path) -> None:
    database, future, public = _services(tmp_path)
    suite = public.create_benchmark_suite(
        {
            "name": "Development score",
            "purpose": "development",
            "items": [{"id": "fixture", "input": "x", "expected": "x"}],
            "primary_metric": "score",
            "direction": "maximize",
        }
    )
    suite_revision_id = suite["latest_revision"]["id"]
    study = future.create_study({"name": "Compare two approaches"})
    protocol = future.create_study_protocol(
        study.id,
        {
            "question": "Which approach works better?",
            "design_kind": "paired_ab",
            "trainer_mode": "sft",
            "development_suite_revision_id": suite_revision_id,
            "dataset_version_id": "fixture-version",
            "seeds": [17, 42, 101],
            "arms": [
                {
                    "name": "Control",
                    "is_control": True,
                    "launch_config": {
                        "model": "fixture/model",
                        "dataset_version_id": "fixture-version",
                        "output_root": str(tmp_path / "runs"),
                    },
                },
                {
                    "name": "Adapted",
                    "launch_config": {
                        "model": "fixture/model",
                        "dataset_version_id": "fixture-version",
                        "output_root": str(tmp_path / "runs"),
                        "learning_rate": 0.0001,
                    },
                },
            ],
            "contrasts": [
                {
                    "name": "Adapted versus control",
                    "left_arm": "Control",
                    "right_arm": "Adapted",
                    "metric": "score",
                    "direction": "maximize",
                    "conclusion_kind": "superiority",
                }
            ],
        },
    )
    plan = public.get_adaptation_study_launch_plan(protocol.id)
    assert plan["run_count"] == 6
    launched = public.launch_adaptation_study(protocol.id)
    terminal = WorkstationWorker(public._scheduler()).run_once(
        work_item_id=launched["work_item_id"]
    )
    assert terminal is not None and terminal.status == "completed"
    assignments = future.list_study_assignments(protocol.id)
    assert all(value.run_group_id and value.run_id for value in assignments)
    assert len({value.run_group_id for value in assignments}) == 2
    assert database.list_run_groups(limit=10)


def test_specialized_readiness_hides_unverified_backend(tmp_path: Path) -> None:
    _, future, _ = _services(tmp_path)
    blocked = future.specialized_task_readiness(
        {
            "task_id": "text-classification",
            "backend": "mlx",
            "model": "fixture/model",
            "dataset": "train.jsonl",
        }
    )
    assert blocked["ready"] is False
    assert "PyTorch only" in blocked["blockers"][0]
    ready = future.specialized_task_readiness(
        {
            "task_id": "text-classification",
            "backend": "pytorch",
            "model": "fixture/model",
            "dataset": "train.jsonl",
        }
    )
    assert ready["ready"] is True
