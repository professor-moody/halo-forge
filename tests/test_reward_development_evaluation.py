from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from halo_forge.cli import (
    _prepare_reward_integrity,
    _launch_boundary_development_evaluation,
    _project_managed_checkpoint_path,
    _seal_training_signal_session,
)
from halo_forge.evaluation_lab import EvaluationLabService
from halo_forge.public_api.service import PublicApiService
from halo_forge.reward_integrity import RewardIntegrityService
from halo_forge.run_db import RunDatabase, RunRecord
from halo_forge.training_signal import (
    TRAINING_SIGNAL_CAPABILITIES,
    TrainingRecordRef,
    TrainingSignalSink,
)
from halo_forge.verifier_lab.store import VerifierLabStore
from halo_forge.workstation_jobs import WorkstationScheduler


def _dependency_ids(database: RunDatabase, work_item_id: str) -> list[str]:
    return [
        str(row[0])
        for row in database._conn.execute(
            "SELECT depends_on_work_item_id FROM work_item_dependencies "
            "WHERE work_item_id=? ORDER BY depends_on_work_item_id",
            (work_item_id,),
        ).fetchall()
    ]


def _suite(evaluations: EvaluationLabService, *, purpose: str = "development"):
    _, revision = evaluations.create_suite(
        name=f"{purpose} checkpoint suite",
        purpose=purpose,
        items=[
            {
                "id": "sample-1",
                "adapter_id": "dataset",
                "input": "prompt",
                "expected": "answer",
            }
        ],
        primary_metric="accuracy",
        direction="maximize",
    )
    assert revision is not None
    return revision


def _verifier_revision(database: RunDatabase, name: str, fingerprint: str):
    store = VerifierLabStore(database)
    profile = store.create_profile(name=name)
    revision = store.create_profile_revision(
        profile.id,
        {
            "family": "deterministic",
            "implementation": {
                "kind": "builtin",
                "ref": "regex_format",
                "fingerprint": fingerprint,
            },
            "modality": "text",
            "task_type": "binary",
            "reward_contract": {
                "minimum": 0,
                "maximum": 1,
                "direction": "maximize",
                "threshold": 0.5,
            },
            "runtime_requirements": {},
        },
    )
    return revision


def test_development_suite_identity_is_validated_before_reward_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = RunDatabase(str(tmp_path / "suite-preflight.db"))
    evaluations = EvaluationLabService(database, tmp_path / "evaluations")
    development = _suite(evaluations, purpose="development")
    holdout = _suite(evaluations, purpose="holdout")
    rewards = RewardIntegrityService(database, root=tmp_path / "reward")
    try:
        assert rewards.validate_development_suite_revision(development.id) == {
            "id": development.id,
            "purpose": "development",
        }
        with pytest.raises(ValueError, match="holdout suite evidence"):
            rewards.validate_development_suite_revision(holdout.id)
        with pytest.raises(ValueError, match="revision is missing"):
            rewards.validate_development_suite_revision("missing-suite-revision")

        class GuardedRewardEngine:
            def validate_development_suite_revision(self, revision_id: str):
                raise ValueError(f"protected suite rejected before resolve: {revision_id}")

            def resolve_binding(self, *args, **kwargs):  # pragma: no cover - guard
                raise AssertionError("binding resolution must not run after suite refusal")

        public = PublicApiService(
            database=database,
            reward_integrity=GuardedRewardEngine(),
            base_path=tmp_path,
        )
        with pytest.raises(ValueError, match="protected suite rejected before resolve"):
            public._resolve_public_reward_payload(  # noqa: SLF001 - preflight contract
                {
                    "mode": "reasoning",
                    "model": "fixture-model",
                    "reward_system_revision_id": "reward-revision",
                    "reward_audit_protocol_revision_id": "protocol-revision",
                    "reward_integrity_profile_revision_id": "profile-revision",
                    "development_suite_revision_id": holdout.id,
                    "reward_audit_boundaries": ["final"],
                }
            )

        def reject_cli_suite(_service, revision_id: str):
            raise ValueError(f"CLI protected suite rejected before resolve: {revision_id}")

        monkeypatch.setattr(
            RewardIntegrityService,
            "validate_development_suite_revision",
            reject_cli_suite,
        )
        args = Namespace(
            database=str(tmp_path / "cli-preflight.db"),
            reward_system_revision="reward-revision",
            reward_audit_protocol_revision="protocol-revision",
            reward_integrity_profile_revision="profile-revision",
            reward_development_suite_revision=holdout.id,
            reward_audit_boundary=["final"],
            accelerator="cpu",
            model="fixture-model",
            verifier_profile_revision=None,
        )
        with pytest.raises(ValueError, match="CLI protected suite rejected before resolve"):
            _prepare_reward_integrity(args, trainer="reasoning")
    finally:
        evaluations.shutdown(wait=False, cancel_futures=True)
        database.close()


def test_evaluation_scheduler_launch_persists_training_dependency(tmp_path: Path):
    database = RunDatabase(":memory:")
    scheduler = WorkstationScheduler(database)
    parent = scheduler.enqueue(kind="training", work_item_id="training-boundary")
    evaluations = EvaluationLabService(
        database, tmp_path / "evaluations", scheduler=scheduler
    )
    revision = _suite(evaluations)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")

    launched = evaluations.launch_evaluation(
        suite_revision_id=revision.id,
        subject={"type": "checkpoint", "path": str(checkpoint)},
        dependencies=[parent.id],
    )
    work = database.get_work_item(str(launched.evaluation.work_item_id))
    assert work is not None
    assert _dependency_ids(database, work.id) == [parent.id]
    assert launched.evaluation.request["work_dependencies"] == [parent.id]


def test_boundary_evaluation_targets_atomic_checkpoint_publication(tmp_path: Path):
    database = RunDatabase(":memory:")
    scheduler = WorkstationScheduler(database)
    run_id = "managed-boundary-run"
    database.upsert_run(
        RunRecord(
            run_id=run_id,
            modality="reasoning",
            model_name="model",
            status="running",
            output_dir=str(tmp_path / "run"),
            indexed_at="now",
        )
    )
    staging = tmp_path / "attempt"
    checkpoint = staging / "cycle_1" / "model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"model")
    snapshot = tmp_path / "snapshots" / "segment-0000"
    parent = scheduler.enqueue(
        kind="training",
        canonical_run_id=run_id,
        launch_spec={
            "operation": "managed_training_segment",
            "output_dir": str(tmp_path / "run"),
            "segment_output_dir": str(snapshot),
            "final_segment": False,
        },
        work_item_id="managed-training-boundary",
    )
    evaluations = EvaluationLabService(
        database, tmp_path / "evaluation-artifacts", scheduler=scheduler
    )
    revision = _suite(evaluations)
    evaluations.shutdown(wait=False, cancel_futures=False)

    projected = _project_managed_checkpoint_path(
        checkpoint, execution_output=staging, work_item=parent
    )
    assert projected == snapshot / "cycle_1" / "model"
    launched = _launch_boundary_development_evaluation(
        database=database,
        scheduler=scheduler,
        suite_revision_id=revision.id,
        run_id=run_id,
        signal_shard_id="trace-identity",
        direct_run_segment_id="segment-identity",
        checkpoint_hash="checkpoint-hash",
        checkpoint_path=checkpoint,
        execution_output=staging,
        current_work_item=parent,
    )
    evaluation = launched.evaluation
    assert evaluation.request["subject_resolution"] == "pending"
    assert evaluation.request["subject_input"]["path"] == str(projected)
    assert evaluation.request["adapter_request"]["gate_semantics"] == (
        "completion_evidence_only"
    )
    assert database.get_work_item(str(evaluation.work_item_id)).canonical_run_id == run_id
    assert _dependency_ids(database, str(evaluation.work_item_id)) == [parent.id]


def test_managed_signal_seal_orders_training_evaluation_audit(
    tmp_path: Path, monkeypatch
):
    database_path = tmp_path / "runs.db"
    database = RunDatabase(str(database_path))
    scheduler = WorkstationScheduler(database)
    evaluations = EvaluationLabService(
        database, tmp_path / "evaluation-artifacts", scheduler=scheduler
    )
    suite_revision = _suite(evaluations)
    evaluations.shutdown(wait=False, cancel_futures=False)

    optimizer = _verifier_revision(database, "optimizer", "optimizer-fingerprint")
    sentinel = _verifier_revision(database, "sentinel", "sentinel-fingerprint")
    rewards = RewardIntegrityService(database, root=tmp_path / "reward")
    system = rewards.create_system(name="development tracking reward")
    system_revision = rewards.create_system_revision(
        system.id,
        optimizer_verifier_revision_id=optimizer.id,
        modality="text",
        task_type="binary",
        auditors=[
            {
                "role": "primary_sentinel",
                "verifier_revision_id": sentinel.id,
            }
        ],
    )

    run_id = "managed-development-run"
    canonical_output = tmp_path / "runs" / run_id
    staging = tmp_path / "attempt"
    checkpoint = staging / "cycle_1" / "model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    database.upsert_run(
        RunRecord(
            run_id=run_id,
            modality="reasoning",
            model_name="model",
            status="running",
            output_dir=str(canonical_output),
            indexed_at="now",
        )
    )
    segment = database.create_direct_run_segment(
        run_id=run_id,
        ordinal=0,
        unit="cycle",
        start_value=0,
        end_value=2,
    )
    training_work = scheduler.enqueue(
        kind="training",
        canonical_run_id=run_id,
        launch_spec={
            "operation": "managed_training_segment",
            "output_dir": str(canonical_output),
            "segment_output_dir": str(tmp_path / "segments" / "segment-0000"),
            "direct_run_segment_id": segment["id"],
            "final_segment": False,
        },
        work_item_id="training-segment-work",
    )
    database.update_direct_run_segment(segment["id"], work_item_id=training_work.id)

    sink = TrainingSignalSink(
        tmp_path / "signal-capture",
        run_id=run_id,
        segment_id=segment["id"],
        boundary="2",
        capability=TRAINING_SIGNAL_CAPABILITIES.get("reasoning:hf"),
        protocol="balanced_256",
    )
    sink.capture(
        record=TrainingRecordRef.virtual_identity({"prompt": "question"}, source_index=0),
        candidate_ordinal=0,
        prompt="question",
        output="answer",
        training_observation={"reward": 1.0, "passed": True},
        source_index=0,
    )
    captured = sink.seal(checkpoint_hash="placeholder")

    class Session:
        def __init__(self):
            self.run_id = run_id

        def finalize(self, *, checkpoint_hash: str):
            assert checkpoint_hash
            return [captured]

    args = Namespace(
        _training_signal_session=Session(),
        _training_signal_capability=TRAINING_SIGNAL_CAPABILITIES.get("reasoning:hf"),
        reward_system_revision=system_revision.id,
        reward_audit_protocol_revision=rewards.default_ids["protocol:balanced_256"],
        reward_integrity_profile_revision=rewards.default_ids[
            "profile:human_aligned_integrity"
        ],
        reward_development_suite_revision=suite_revision.id,
        accelerator="cuda",
        model="model",
        dataset="dataset.jsonl",
        wait=False,
    )
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(database_path))
    monkeypatch.setenv("HALOFORGE_WORK_ITEM_ID", training_work.id)
    monkeypatch.setenv("HALO_FORGE_RUN_ID", run_id)

    _seal_training_signal_session(
        args,
        {"run_id": run_id, "final_model_path": str(checkpoint)},
        staging,
    )

    audit = rewards.list_audits(run_id=run_id, limit=10).items[0]
    audit_id = str(audit["id"])
    detail = rewards.get_audit_detail(audit_id)
    assert detail is not None
    linked = detail["development_evaluation"]
    assert linked["gate_semantics"] == "completion_evidence_only"
    try:
        rewards.development_evaluation_evidence(audit_id, require_complete=True)
    except ValueError as exc:
        assert "has not completed successfully" in str(exc)
    else:
        raise AssertionError("the reward audit must wait for development evaluation")
    evaluation_id = linked["evaluation_id"]
    evaluation = database.get_evaluation(evaluation_id)
    assert evaluation is not None
    assert evaluation.subject_ref == run_id
    assert _dependency_ids(database, str(evaluation.work_item_id)) == [training_work.id]
    assert _dependency_ids(database, str(audit["work_item_id"])) == [
        str(evaluation.work_item_id)
    ]
    binding = next(
        value
        for value in detail["bindings"]
        if value["audit_id"] == audit_id
    )
    assert binding["context"]["development_evaluation_id"] == evaluation_id
    assert binding["context"]["development_evaluation_semantics"] == (
        "completion_evidence_only"
    )

    delattr(args, "_training_signal_shards")
    _seal_training_signal_session(
        args,
        {"run_id": run_id, "final_model_path": str(checkpoint)},
        staging,
    )
    assert len(database.list_evaluations(subject_ref=run_id, limit=10)) == 1
    assert rewards.list_audits(run_id=run_id, limit=10).total == 1

    with database._lock:
        database._conn.execute(
            "UPDATE work_items SET status='completed',stage='complete' WHERE id=?",
            (training_work.id,),
        )
        database._conn.execute(
            "UPDATE work_items SET status='failed',stage='failed',error=? WHERE id=?",
            ("development evaluator failed", str(evaluation.work_item_id)),
        )
        database._refresh_dependency_states_locked(created_at="now")
        database._conn.commit()
    assert scheduler.claim() is None
    blocked_audit = database.get_work_item(str(audit["work_item_id"]))
    assert blocked_audit is not None
    assert blocked_audit.status == "blocked"
    retried = scheduler.retry(
        str(evaluation.work_item_id),
        reason="operator fixed the development evaluator",
    )
    assert retried is not None and retried.status == "queued"
