"""Focused v5 contracts for checkpoint-policy orchestration."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from halo_forge.adaptive_lab import AdaptiveLabService
from halo_forge.orchestration import OrchestrationService
from halo_forge.run_db import RunDatabase
from halo_forge.workstation_jobs import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
    WorkstationScheduler,
    WorkstationWorker,
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


def _scheduler(db: RunDatabase, *, worker_id: str = "test") -> WorkstationScheduler:
    return WorkstationScheduler(db, worker_id=worker_id, capacity_probe=_available_capacity)


def _suite(db: RunDatabase, name: str, metric: str = "accuracy"):
    suite = db.create_benchmark_suite(name=name, purpose="development")
    return db.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash=f"{name}-v1",
        items=[{"id": "one", "input": "hello", "expected": "world"}],
        primary_metric=metric,
        direction="maximize",
    )


def _policy(
    db: RunDatabase,
    development_id: str,
    *,
    automatic: bool = True,
    guardrail_ids=(),
    rules=(),
):
    adaptive = AdaptiveLabService(db)
    policy = adaptive.create_policy(name=f"policy-{automatic}-{len(guardrail_ids)}")
    return adaptive.create_policy_revision(
        policy.id,
        {
            "development_suite_revision_id": development_id,
            "primary_metric": "accuracy",
            "direction": "maximize",
            "schedule": {
                "mode": "explicit",
                "unit": "step",
                "boundaries": [2, 4],
            },
            "rules": list(rules),
            "guardrail_suite_revision_ids": list(guardrail_ids),
            "automatic_actions": automatic,
        },
    )


def _complete(scheduler: WorkstationScheduler, work_item_id: str, result=None):
    item = scheduler.claim(work_item_id=work_item_id)
    assert item is not None
    completed = scheduler.complete(item, result=result or {})
    assert completed is not None
    return completed


def _complete_segment(scheduler: WorkstationScheduler, run, metrics):
    training = next(item for item in run["work_items"] if item["kind"] == "training")
    evaluations = [item for item in run["work_items"] if item["kind"] == "evaluation"]
    _complete(scheduler, training["id"])
    for evaluation in evaluations:
        suite_id = evaluation["launch_spec"]["suite_revision_id"]
        _complete(scheduler, evaluation["id"], {"metrics": metrics[suite_id]})


@pytest.mark.parametrize("trainer_mode", ["sft", "dpo", "orpo", "rm", "grpo"])
def test_hf_step_segments_force_an_exact_boundary_checkpoint(trainer_mode: str):
    command = OrchestrationService._training_command(
        trainer_mode=trainer_mode,
        config={
            "model": "local/model",
            "data": "train.jsonl",
            "output_dir": "runs/example",
            "seed": 42,
            "save_steps": 500,
        },
        bindings=(),
        unit="step",
        start_value=0,
        end_value=25,
    )
    assert command is not None
    assert command[command.index("--max-steps") + 1] == "25"
    assert command[command.index("--save-steps") + 1] == "25"
    assert command.count("--save-steps") == 1
    if trainer_mode in {"sft", "dpo", "orpo"}:
        assert command[command.index("--eval-steps") + 1] == "25"
        assert command.count("--eval-steps") == 1


@pytest.mark.parametrize("trainer_mode", ["sft", "dpo", "orpo"])
def test_bounded_validation_cadence_passes_transformers_argument_validation(
    trainer_mode: str, tmp_path
):
    from transformers import TrainingArguments

    command = OrchestrationService._training_command(
        trainer_mode=trainer_mode,
        config={
            "model": "local/model",
            "data": "train.jsonl",
            "output_dir": str(tmp_path / trainer_mode),
            "seed": 42,
            "save_steps": 500,
            "eval_steps": 100,
        },
        bindings=(),
        unit="step",
        start_value=0,
        end_value=25,
    )
    assert command is not None
    save_steps = int(command[command.index("--save-steps") + 1])
    eval_steps = int(command[command.index("--eval-steps") + 1])

    arguments = TrainingArguments(
        output_dir=str(tmp_path / f"validated-{trainer_mode}"),
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=save_steps,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        report_to=[],
    )

    assert arguments.save_steps == arguments.eval_steps == 25


def test_adaptive_boundaries_continue_idempotently_and_finish_with_trajectory():
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    development = _suite(db, "development")
    policy = _policy(
        db,
        development.id,
        rules=(
            {
                "metric": "accuracy",
                "direction": "maximize",
                "kind": "objective",
                "comparison": "absolute",
                "threshold": 0.5,
                "on_breach": "stop",
            },
        ),
    )
    service = OrchestrationService(db, scheduler=scheduler)
    created = service.create_group_from_payload(
        {
            "version": 2,
            "name": "adaptive repeat",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {"max_steps": 4},
            "seeds": [7],
            "development_suite_revision_id": development.id,
            "checkpoint_policy_revision_id": policy.revision_id,
        }
    )
    assert created["resolved_checkpoint_plan"]["boundaries"] == [2, 4]
    first_run = created["trials"][0]["runs"][0]
    first_segment_id = first_run["segments"][0]["id"]
    _complete_segment(
        scheduler,
        first_run,
        {development.id: {"accuracy": 0.8}},
    )

    advanced = service.advance_ready_checkpoint_policy(
        created["id"], trial_segment_id=first_segment_id
    )
    assert advanced["advanced"] is True
    assert advanced["outcomes"][0]["action"] == "continue"
    assert len(advanced["queued_work_item_ids"]) == 2
    work_count = len(db.list_work_items(limit=1000))
    repeated = service.advance_ready_checkpoint_policy(
        created["id"], trial_segment_id=first_segment_id
    )
    assert repeated["advanced"] is False
    assert repeated["outcomes"][0]["recovered"] is True
    assert len(db.list_work_items(limit=1000)) == work_count

    detail = service.get_run_group_detail(created["id"])
    second_run = detail["trials"][0]["runs"][0]
    second_segment = second_run["segments"][-1]
    second_items = [
        item
        for item in second_run["work_items"]
        if item["launch_spec"].get("trial_segment_id") == second_segment["id"]
    ]
    _complete_segment(
        scheduler,
        {"work_items": second_items},
        {development.id: {"accuracy": 0.9}},
    )
    final = service.advance_ready_checkpoint_policy(
        created["id"], trial_segment_id=second_segment["id"]
    )
    assert final["outcomes"][0]["action"] == "complete"
    completed = service.get_run_group_detail(created["id"])
    assert completed["status"] == "completed"
    points = completed["trajectory"]["runs"][0]["points"]
    assert [(point["start_value"], point["end_value"]) for point in points] == [
        (0, 2),
        (2, 4),
    ]
    assert all(point["latest_gate_decision"] for point in points)


def test_manual_gate_review_and_guardrail_evidence_are_required():
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    development = _suite(db, "development")
    guardrail = _suite(db, "safety", metric="safety")
    policy = _policy(
        db,
        development.id,
        automatic=False,
        guardrail_ids=(guardrail.id,),
        rules=(
            {
                "metric": "safety",
                "direction": "maximize",
                "kind": "guardrail",
                "comparison": "absolute",
                "threshold": 0.7,
                "on_breach": "stop",
            },
        ),
    )
    service = OrchestrationService(db, scheduler=scheduler)
    created = service.create_group_from_payload(
        {
            "version": 2,
            "name": "reviewed repeat",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {"max_steps": 4},
            "seeds": [11],
            "development_suite_revision_id": development.id,
            "checkpoint_policy_revision_id": policy.revision_id,
        }
    )
    run = created["trials"][0]["runs"][0]
    training = next(item for item in run["work_items"] if item["kind"] == "training")
    evaluations = [item for item in run["work_items"] if item["kind"] == "evaluation"]
    primary = next(
        item for item in evaluations if item["launch_spec"]["suite_revision_id"] == development.id
    )
    held = next(item for item in evaluations if item["id"] != primary["id"])
    _complete(scheduler, training["id"])
    _complete(scheduler, primary["id"], {"metrics": {"accuracy": 0.8}})
    waiting = service.advance_ready_checkpoint_policy(created["id"])
    assert waiting["advanced"] is False
    assert waiting["outcomes"][0]["pending_suite_revision_ids"] == [guardrail.id]

    _complete(scheduler, held["id"], {"metrics": {"safety": 0.9}})
    paused = service.advance_ready_checkpoint_policy(created["id"])
    decision = paused["outcomes"][0]["gate_decision"]
    assert decision["action"] == "pause"
    paused_detail = service.get_run_group_detail(created["id"])
    assert paused_detail["status"] == "awaiting_review"
    paused_segment = paused_detail["trajectory"]["runs"][0]["points"][0]
    assert (paused_segment["status"], paused_segment["decision"]) == (
        "awaiting_review",
        "pause",
    )
    reviewed = service.review_checkpoint_gate(
        decision["id"], action="continue", reason="metrics reviewed by operator"
    )
    assert reviewed["action"] == "continue"
    assert reviewed["gate_decision"]["override_of_id"] == decision["id"]
    assert reviewed["run_group"]["status"] == "queued"
    resumed_segment = reviewed["run_group"]["trajectory"]["runs"][0]["points"][0]
    assert (resumed_segment["status"], resumed_segment["decision"]) == (
        "completed",
        "continue",
    )


def test_guardrail_metric_cannot_replace_the_pinned_development_objective():
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    development = _suite(db, "development")
    guardrail = _suite(db, "guardrail-with-same-metric")
    policy = _policy(
        db,
        development.id,
        guardrail_ids=(guardrail.id,),
    )
    service = OrchestrationService(db, scheduler=scheduler)
    created = service.create_group_from_payload(
        {
            "version": 2,
            "name": "suite-pinned objective",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {"max_steps": 4},
            "seeds": [17],
            "development_suite_revision_id": development.id,
            "checkpoint_policy_revision_id": policy.revision_id,
        }
    )
    run = created["trials"][0]["runs"][0]
    _complete_segment(
        scheduler,
        run,
        {
            development.id: {"accuracy": 0.91},
            guardrail.id: {"accuracy": 0.11},
        },
    )

    detail = service.get_run_group_detail(created["id"])
    assert detail["trials"][0]["runs"][0]["objective_value"] == pytest.approx(0.91)
    assert detail["cohort_aggregates"][0]["mean"] == pytest.approx(0.91)


def test_checkpoint_policy_rejects_a_full_trial_only_backend():
    db = RunDatabase(":memory:")
    development = _suite(db, "development")
    policy = _policy(db, development.id)
    service = OrchestrationService(db, scheduler=_scheduler(db))
    with pytest.raises(ValueError, match="adaptive checkpoint execution is unavailable"):
        service.create_group_from_payload(
            {
                "version": 2,
                "name": "unsupported",
                "kind": "repeat",
                "trainer_mode": "dpo",
                "base_config": {"backend": "mlx", "max_steps": 4},
                "seeds": [1],
                "development_suite_revision_id": development.id,
                "checkpoint_policy_revision_id": policy.revision_id,
            }
        )


def test_automatic_stop_is_truthful_and_requires_an_append_only_override():
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    development = _suite(db, "development")
    policy = _policy(
        db,
        development.id,
        rules=(
            {
                "metric": "accuracy",
                "direction": "maximize",
                "kind": "guardrail",
                "comparison": "absolute",
                "threshold": 0.9,
                "on_breach": "stop",
            },
        ),
    )
    service = OrchestrationService(db, scheduler=scheduler)
    created = service.create_group_from_payload(
        {
            "version": 2,
            "name": "guardrail stop",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {"max_steps": 4},
            "seeds": [4],
            "development_suite_revision_id": development.id,
            "checkpoint_policy_revision_id": policy.revision_id,
        }
    )
    run = created["trials"][0]["runs"][0]
    _complete_segment(scheduler, run, {development.id: {"accuracy": 0.2}})
    outcome = service.advance_ready_checkpoint_policy(created["id"])
    decision = outcome["outcomes"][0]["gate_decision"]
    assert decision["action"] == "stop"
    stopped = service.get_run_group_detail(created["id"])
    assert stopped["status"] == "stopped"
    point = stopped["trajectory"]["runs"][0]["points"][0]
    assert (point["status"], point["decision"]) == ("stopped", "stop")
    with pytest.raises(ValueError, match="reviewed gate override"):
        service.resume_run_group(created["id"], reason="do not bypass the gate")

    resumed = service.review_checkpoint_gate(
        decision["id"], action="continue", reason="accepted measured regression"
    )
    assert resumed["run_group"]["status"] == "queued"
    assert resumed["gate_decision"]["override_reason"] == "accepted measured regression"


def test_plateau_patience_resets_after_a_qualifying_improvement():
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    development = _suite(db, "plateau-development")
    adaptive = AdaptiveLabService(db)
    policy_head = adaptive.create_policy(name="plateau patience")
    policy = adaptive.create_policy_revision(
        policy_head.id,
        {
            "development_suite_revision_id": development.id,
            "primary_metric": "accuracy",
            "direction": "maximize",
            "schedule": {
                "mode": "explicit",
                "unit": "step",
                "boundaries": [1, 2, 3, 4, 5, 6],
            },
            "rules": [
                {
                    "metric": "accuracy",
                    "direction": "maximize",
                    "kind": "plateau",
                    "comparison": "best",
                    "minimum_delta": 0.01,
                    "patience": 2,
                    "on_breach": "stop",
                }
            ],
            "automatic_actions": True,
        },
    )
    service = OrchestrationService(db, scheduler=scheduler)
    created = service.create_group_from_payload(
        {
            "version": 2,
            "name": "plateau reset",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {"max_steps": 6},
            "seeds": [5],
            "development_suite_revision_id": development.id,
            "checkpoint_policy_revision_id": policy.revision_id,
        }
    )

    observed_counts = []
    actions = []
    # The small improvement at checkpoint two accumulates patience. The
    # qualifying improvement at checkpoint three resets it; two subsequent
    # sub-threshold checkpoints then stop the run.
    for value in (0.50, 0.505, 0.52, 0.521, 0.522):
        detail = service.get_run_group_detail(created["id"])
        run = detail["trials"][0]["runs"][0]
        segment = run["segments"][-1]
        segment_items = [
            item
            for item in run["work_items"]
            if item["launch_spec"].get("trial_segment_id") == segment["id"]
        ]
        _complete_segment(
            scheduler,
            {"work_items": segment_items},
            {development.id: {"accuracy": value}},
        )
        outcome = service.advance_ready_checkpoint_policy(
            created["id"], trial_segment_id=segment["id"]
        )["outcomes"][0]
        gate = outcome["gate_decision"]
        actions.append(gate["action"])
        observed_counts.append(gate["evidence"]["rule_outcomes"][0].get("plateau_count", 0))

    assert observed_counts == [0, 1, 0, 1, 2]
    assert actions == ["continue", "continue", "continue", "continue", "stop"]
    stopped = service.get_run_group_detail(created["id"])
    assert stopped["status"] == "stopped"
    assert stopped["trajectory"]["runs"][0]["points"][-1]["decision"] == "stop"


def test_worker_startup_recovers_a_completed_evaluation_callback_exactly_once():
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    development = _suite(db, "development")
    policy = _policy(db, development.id)
    service = OrchestrationService(db, scheduler=scheduler)
    created = service.create_group_from_payload(
        {
            "version": 2,
            "name": "restart recovery",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {"max_steps": 4},
            "seeds": [3],
            "development_suite_revision_id": development.id,
            "checkpoint_policy_revision_id": policy.revision_id,
        }
    )
    run = created["trials"][0]["runs"][0]
    _complete_segment(
        scheduler,
        run,
        {development.id: {"accuracy": 0.8}},
    )
    worker = WorkstationWorker(scheduler)
    worker._recover_checkpoint_orchestration()
    assert len(db.list_checkpoint_gate_decisions(run_group_id=created["id"])) == 1
    assert len(db.list_work_items(limit=1000)) == 4
    worker._recover_checkpoint_orchestration()
    assert len(db.list_checkpoint_gate_decisions(run_group_id=created["id"])) == 1
    assert len(db.list_work_items(limit=1000)) == 4


def test_exhausted_evaluation_failure_immediately_enters_missing_evidence_gate():
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    development = _suite(db, "development")
    policy = _policy(db, development.id)
    service = OrchestrationService(db, scheduler=scheduler)
    created = service.create_group_from_payload(
        {
            "version": 2,
            "name": "evaluation failure gate",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {"max_steps": 4},
            "seeds": [3],
            "development_suite_revision_id": development.id,
            "checkpoint_policy_revision_id": policy.revision_id,
            "max_retries": 0,
        }
    )
    run = created["trials"][0]["runs"][0]
    training = next(item for item in run["work_items"] if item["kind"] == "training")
    evaluation = next(item for item in run["work_items"] if item["kind"] == "evaluation")
    _complete(scheduler, training["id"])
    claimed = scheduler.claim(work_item_id=evaluation["id"])
    assert claimed is not None
    failed = scheduler.fail(claimed, error="evaluator exited after its final retry")
    assert failed is not None and failed.status == "failed"

    WorkstationWorker(scheduler)._after_terminal_event(claimed, failed)

    decisions = db.list_checkpoint_gate_decisions(run_group_id=created["id"])
    assert len(decisions) == 1
    assert decisions[0].action == "pause"
    assert "missing_required_evidence" in decisions[0].reasons
    detail = service.get_run_group_detail(created["id"])
    assert detail["status"] == "awaiting_review"


def test_run_level_legacy_evaluation_cannot_fill_a_checkpoint_boundary(tmp_path):
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    development = _suite(db, "development")
    service = OrchestrationService(db, scheduler=scheduler)
    created = service.create_group_from_payload(
        {
            "name": "segment identity",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {"max_steps": 4},
            "seeds": [3],
            "development_suite_revision_id": development.id,
        }
    )
    run_view = created["trials"][0]["runs"][0]
    trial_run = db.get_trial_run(run_view["id"])
    group = db.get_run_group(created["id"])
    assert trial_run is not None and group is not None
    segment = db.list_trial_segments(trial_run.id)[0]
    legacy = db.create_evaluation(
        evaluation_id="legacy-run-level-eval",
        suite_revision_id=development.id,
        adapter_id="legacy",
        adapter_version="1",
        subject_type="run",
        subject_ref=trial_run.run_id,
        subject_hash="legacy-subject",
        reuse_key="legacy-run-level-eval",
        request={},
    )
    db.complete_evaluation(
        legacy.id,
        metrics=[{"name": "accuracy", "value": 0.99, "direction": "maximize"}],
        samples=[
            {
                "suite_item_id": "one",
                "input": "hello",
                "output": "world",
                "score": 0.99,
                "passed": True,
            }
        ],
        result={"primary_metric": "accuracy"},
        artifact_path=str(tmp_path / "legacy-evidence"),
    )
    work_items = service._group_work_items(group.id)

    assert (
        service._objective_for_run(
            group,
            trial_run,
            development,
            work_items,
            segment.id,
        )
        is None
    )
    assert service._objective_for_run(
        group,
        trial_run,
        development,
        work_items,
        None,
    ) == pytest.approx(0.99)


@pytest.mark.parametrize("failure_stage", ["checkpoint", "halving"])
def test_terminal_callback_failure_rearms_startup_reconciliation(monkeypatch, failure_stage):
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    queued = scheduler.enqueue(
        kind="evaluation",
        launch_spec={
            "operation": "evaluate_trial_segment",
            "run_group_id": "group-for-recovery",
            "trial_segment_id": "segment-for-recovery",
        },
    )
    claimed = scheduler.claim(work_item_id=queued.id)
    assert claimed is not None
    terminal = (
        scheduler.fail(claimed, error="terminal evaluator failure")
        if failure_stage == "checkpoint"
        else scheduler.complete(claimed, result={})
    )
    assert terminal is not None

    if failure_stage == "checkpoint":
        monkeypatch.setattr(
            OrchestrationService,
            "advance_ready_checkpoint_policy",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("gate callback failed")),
        )
    else:
        monkeypatch.setattr(
            OrchestrationService,
            "advance_ready_checkpoint_policy",
            lambda *args, **kwargs: {"advanced": False},
        )
        monkeypatch.setattr(
            OrchestrationService,
            "advance_ready_successive_halving",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("halving callback failed")),
        )
    worker = WorkstationWorker(scheduler)
    worker._orchestration_recovery_complete = True

    worker._after_terminal_event(claimed, terminal)

    assert worker._orchestration_recovery_complete is False


def test_gate_recovery_repairs_a_partially_queued_continuation(monkeypatch):
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db)
    development = _suite(db, "development")
    policy = _policy(db, development.id)
    service = OrchestrationService(db, scheduler=scheduler)
    created = service.create_group_from_payload(
        {
            "version": 2,
            "name": "partial continuation",
            "kind": "repeat",
            "trainer_mode": "sft",
            "base_config": {"max_steps": 4},
            "seeds": [9],
            "development_suite_revision_id": development.id,
            "checkpoint_policy_revision_id": policy.revision_id,
        }
    )
    run = created["trials"][0]["runs"][0]
    _complete_segment(scheduler, run, {development.id: {"accuracy": 0.8}})
    original_enqueue = scheduler.enqueue
    failed_once = False

    def flaky_enqueue(**kwargs):
        nonlocal failed_once
        launch = kwargs.get("launch_spec") or {}
        if (
            not failed_once
            and kwargs.get("kind") == "evaluation"
            and launch.get("segment_ordinal") == 1
        ):
            failed_once = True
            raise RuntimeError("simulated crash while linking evaluation")
        return original_enqueue(**kwargs)

    monkeypatch.setattr(scheduler, "enqueue", flaky_enqueue)
    with pytest.raises(RuntimeError, match="simulated crash"):
        service.advance_ready_checkpoint_policy(created["id"])
    assert len(db.list_checkpoint_gate_decisions(run_group_id=created["id"])) == 1
    assert len(db.list_trial_segments(run["id"])) == 2
    assert len(db.list_work_items(limit=1000)) == 3

    monkeypatch.setattr(scheduler, "enqueue", original_enqueue)
    recovered = service.advance_ready_checkpoint_policy(created["id"])
    assert recovered["advanced"] is False
    assert recovered["outcomes"][0]["recovered"] is True
    assert len(db.list_work_items(limit=1000)) == 4


def test_worker_dispatches_adaptive_evidence_handler(monkeypatch):
    db = RunDatabase(":memory:")
    scheduler = _scheduler(db, worker_id="worker")
    item = scheduler.enqueue(
        kind="evidence_bundle",
        resource_class="cpu",
        launch_spec={
            "handler": "adaptive_lab.execute_work_item",
            "operation": "build_evidence_bundle",
            "evidence_bundle_id": "bundle-1",
        },
    )

    def execute(self, claimed):
        finished = self.db.finish_work_item(
            claimed.id,
            claim_token=claimed.claim_token,
            result={"evidence_bundle_id": "bundle-1"},
        )
        assert finished is not None
        return finished.to_dict()

    monkeypatch.setattr(AdaptiveLabService, "execute_work_item", execute)
    terminal = WorkstationWorker(scheduler).run_once()
    assert terminal is not None
    assert terminal.id == item.id
    assert terminal.status == "completed"
    assert terminal.result["evidence_bundle_id"] == "bundle-1"
