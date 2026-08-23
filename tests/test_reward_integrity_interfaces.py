"""V8 transport and durable-execution contract tests.

These tests deliberately cross the API/CLI/worker boundaries.  The core
scientific calculations have their own unit coverage; this file protects the
places where an otherwise valid immutable audit could be stranded by a
foreign key, routed to the generic subprocess worker, or lose its run gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from halo_forge.public_api.activity import activity_item_view
from halo_forge.public_api.service import PublicApiService
from halo_forge.replay import sync_reward_integrity_decision
from halo_forge.reward_integrity import RewardIntegrityMetric, RewardIntegrityService
from halo_forge.run_db import LabV4Catalog, RunDatabase, RunRecord
from halo_forge.training_signal import TrainingRecordRef, TrainingSignalSnapshot
from halo_forge.verifier_lab import VerifierLabService
from halo_forge.workstation_jobs import WorkstationScheduler
from halo_forge.workstation_jobs.worker import WorkstationWorker
from ui.state import AppState


def _verifier_revision(
    service: VerifierLabService, *, name: str, pattern: str
) -> str:
    created = service.create_profile(
        name=name,
        description=None,
        definition={
            "family": "deterministic",
            "implementation": {"ref": "regex_format"},
            "configuration": {"pattern": pattern, "full_match": True},
            "modality": "text",
            "task_type": "binary",
            "input_mapping": {"candidate": "output"},
            "reward_contract": {
                "minimum": 0.0,
                "maximum": 1.0,
                "threshold": 0.5,
                "tie_policy": "fail",
                "error_behavior": "fail_closed",
            },
            "runtime_contract": {},
        },
    )
    return str(created["revision"]["id"])


def _audit_fixture(tmp_path: Path, *, real_checkpoint: bool = False):
    database = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(database, worker_id="v8-interface-worker")
    verifiers = VerifierLabService(
        database,
        root=tmp_path / "verifier-calibrations",
        scheduler=scheduler,
    )
    optimizer_id = _verifier_revision(
        verifiers, name="Training reward", pattern=r"^ok$"
    )
    sentinel_id = _verifier_revision(
        verifiers, name="Independent sentinel", pattern=r"^never$"
    )
    reward_root = tmp_path / "reward-integrity"
    rewards = RewardIntegrityService(
        database, root=reward_root, scheduler=scheduler
    )
    created_system = rewards.create_system(
        name="Audited binary reward",
        definition={
            "optimizer_verifier_revision_id": optimizer_id,
            "modality": "text",
            "task_type": "binary",
            "auditors": [
                {
                    "role": "primary_sentinel",
                    "verifier_revision_id": sentinel_id,
                }
            ],
        },
    )
    system_revision_id = str(created_system["revision"]["id"])
    run_output = tmp_path / "audited-run"
    checkpoint_hash = "checkpoint-hash"
    if real_checkpoint:
        from halo_forge.artifact_lab.hashing import hash_path

        checkpoint = run_output / "final_model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text(
            '{"base_model_name_or_path":"fixture-model"}', encoding="utf-8"
        )
        (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture")
        (run_output / "training_summary.json").write_text(
            json.dumps({"final_model_path": str(checkpoint)}), encoding="utf-8"
        )
        checkpoint_hash = hash_path(checkpoint).content_hash
    database.upsert_run(
        RunRecord(
            run_id="audited-run",
            modality="raft",
            model_name="fixture-model",
            status="running",
            output_dir=str(run_output),
            indexed_at="now",
        )
    )
    snapshots = []
    for index in range(20):
        snapshots.append(
            TrainingSignalSnapshot.create(
                record=TrainingRecordRef.virtual_identity(
                    {"prompt": f"question {index}"}, source_index=index
                ),
                candidate_ordinal=0,
                prompt=f"question {index}",
                output="ok",
                training_observation={"reward": 1.0, "passed": True},
                run_id="audited-run",
                segment_id="segment-final",
                boundary="final",
                producer_model_hash="producer-model-hash",
                checkpoint_hash=checkpoint_hash,
            )
        )
    shard = rewards.create_signal_shard(
        run_id="audited-run",
        segment_id="segment-final",
        reward_system_revision_id=system_revision_id,
        protocol_revision_id=rewards.default_ids["protocol:balanced_256"],
        capability_id="raft:hf",
        capture_fidelity="exact",
        boundary_unit="final",
        boundary_value=0,
        snapshots=snapshots,
        aggregate={"event_count": len(snapshots)},
        dataset_identity={"kind": "fixture"},
        producer_model_hash="producer-model-hash",
        checkpoint_hash=checkpoint_hash,
        runtime_identity={"fixture": True},
    )
    return database, scheduler, rewards, reward_root, shard


def test_core_auto_submit_routes_to_reward_worker(tmp_path: Path) -> None:
    database, _scheduler, rewards, reward_root, shard = _audit_fixture(tmp_path)
    try:
        audit = rewards.create_audit(signal_shard_id=shard.id)
        assert audit.work_item_id
        work = database.get_work_item(str(audit.work_item_id))
        assert work is not None
        assert work.domain_kind == "reward_integrity_audit"
        assert work.domain_id == audit.id
        assert work.launch_spec == {
            "handler": "reward_integrity.execute_audit",
            "operation": "reward_integrity_audit",
            "audit_id": audit.id,
            "reward_integrity_root": str(reward_root),
        }
        rewards.store.add_metrics(
            [
                RewardIntegrityMetric(
                    audit_id=audit.id,
                    name="paired_coverage",
                    value=0.875,
                    available=True,
                    record_count=224,
                    direction="maximize",
                )
            ]
        )
        activity = activity_item_view(
            database, LabV4Catalog(database), database.get_work_item(work.id)
        )
        assert activity["summary_metrics"] == {"paired_coverage": 0.875}
        cancelled = rewards.cancel_audit(audit.id)
        assert (cancelled.status, cancelled.stage) == ("cancelled", "cancelled")
        assert database.get_work_item(work.id).status == "cancelled"
    finally:
        database.close()


def test_worker_finishes_domain_cancellation_and_releases_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from halo_forge.reward_integrity import runtime as reward_runtime

    database, scheduler, rewards, _reward_root, shard = _audit_fixture(tmp_path)
    audit = rewards.create_audit(signal_shard_id=shard.id)

    def cancelled_audit(
        active_database: RunDatabase,
        audit_id: str,
        *,
        root: str | None = None,
        bootstrap_resamples: int = 10_000,
    ):
        del active_database, root, bootstrap_resamples
        return rewards.store.update_audit(
            audit_id, status="cancelled", stage="cancelled"
        )

    monkeypatch.setattr(reward_runtime, "execute_pinned_audit", cancelled_audit)
    try:
        finished = WorkstationWorker(
            scheduler, heartbeat_interval=0.01
        ).run_once()
        assert finished is not None and finished.status == "cancelled"
        assert database.get_work_item(str(audit.work_item_id)).status == "cancelled"
        assert database.list_resource_leases() == []
    finally:
        database.close()


def test_worker_publishes_corrupt_trace_as_incomplete_evidence(
    tmp_path: Path,
) -> None:
    database, scheduler, rewards, _reward_root, shard = _audit_fixture(tmp_path)
    try:
        audit = rewards.create_audit(signal_shard_id=shard.id)
        snapshots = Path(shard.storage_path) / "snapshots.jsonl"
        snapshots.write_text(
            snapshots.read_text(encoding="utf-8") + "{}\n",
            encoding="utf-8",
        )

        finished = WorkstationWorker(
            scheduler, heartbeat_interval=0.01
        ).run_once()

        assert finished is not None and finished.status == "completed"
        completed = rewards.get_audit(audit.id)
        assert completed.status == "completed"
        decision = rewards.store.list_decisions(audit.id, limit=10).items[-1]
        assert decision.decision == "incomplete_evidence"
        assert decision.action == "pause"
        assert decision.evidence["scientific_metrics_computed"] is False
        assert any(
            reason
            == "training_signal_integrity_invalid:bundle checksum mismatch: snapshots.jsonl"
            for reason in decision.reasons
        )
        assert rewards.store.list_metrics(audit.id, limit=10).total == 0
        assert database._conn.execute(
            "SELECT COUNT(*) FROM reward_integrity_observations "
            "WHERE audit_id=? AND role='primary_sentinel'",
            (audit.id,),
        ).fetchone()[0] == 0
        assert database.get_run("audited-run").status == "awaiting_review"
        assert database.get_work_item(str(audit.work_item_id)).retry_count == 0
        assert rewards.verify_audit_bundle(audit.id)["valid"] is True
    finally:
        database.close()


def test_worker_publishes_stale_runtime_as_incomplete_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, scheduler, rewards, _reward_root, shard = _audit_fixture(tmp_path)
    try:
        audit = rewards.create_audit(signal_shard_id=shard.id)
        system = rewards.get_system_revision(audit.reward_system_revision_id)

        def stale_runtime(
            _service: VerifierLabService,
            revision_id: str,
            actual: dict | None = None,
        ) -> dict:
            del actual
            if revision_id != system.optimizer_verifier_revision_id:
                return {
                    "verifier_revision_id": revision_id,
                    "state": "compatible",
                    "compatible": True,
                    "mismatches": [],
                }
            return {
                "verifier_revision_id": revision_id,
                "state": "stale_runtime",
                "compatible": False,
                "mismatches": [{"field": "implementation_identity"}],
            }

        monkeypatch.setattr(
            VerifierLabService, "runtime_compatibility", stale_runtime
        )
        finished = WorkstationWorker(
            scheduler, heartbeat_interval=0.01
        ).run_once()

        assert finished is not None and finished.status == "completed"
        decision = rewards.store.list_decisions(audit.id, limit=10).items[-1]
        assert decision.decision == "incomplete_evidence"
        assert decision.action == "pause"
        assert decision.reasons == [
            "stale_verifier_runtime:optimizer:"
            + system.optimizer_verifier_revision_id
            + ":implementation_identity"
        ]
        assert rewards.store.list_metrics(audit.id, limit=10).total == 0
        assert database._conn.execute(
            "SELECT COUNT(*) FROM reward_integrity_observations "
            "WHERE audit_id=? AND role='primary_sentinel'",
            (audit.id,),
        ).fetchone()[0] == 0
        assert database.get_run("audited-run").status == "awaiting_review"
        assert database.get_work_item(str(audit.work_item_id)).retry_count == 0
        assert rewards.verify_audit_bundle(audit.id)["valid"] is True
    finally:
        database.close()


def test_transient_sentinel_execution_error_remains_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, scheduler, rewards, _reward_root, shard = _audit_fixture(tmp_path)
    try:
        audit = rewards.create_audit(signal_shard_id=shard.id)

        def provider_unavailable(*_args, **_kwargs):
            raise RuntimeError("provider temporarily unavailable")

        monkeypatch.setattr(
            VerifierLabService, "invoke_revision", provider_unavailable
        )
        finished = WorkstationWorker(
            scheduler, heartbeat_interval=0.01
        ).run_once()

        assert finished is not None and finished.status == "queued"
        assert finished.retry_count == 1
        retried_audit = rewards.get_audit(audit.id)
        assert retried_audit.status == "queued"
        assert retried_audit.retry_count == 1
        assert rewards.store.list_decisions(audit.id, limit=10).total == 0
        assert rewards.store.list_metrics(audit.id, limit=10).total == 0
        assert database.get_run("audited-run").status == "running"
    finally:
        database.close()


def test_completed_audit_reuse_requires_verified_bundle(tmp_path: Path) -> None:
    database, _scheduler, rewards, _reward_root, shard = _audit_fixture(tmp_path)
    try:
        audit = rewards.create_audit(signal_shard_id=shard.id, submit=False)
        completed = rewards.execute_audit(
            audit.id,
            sentinel=lambda _sample: {"reward": 1.0, "passed": True},
            bootstrap_resamples=16,
        )
        reused = rewards.create_audit(signal_shard_id=shard.id)
        assert reused.id == completed.id
        assert database._conn.execute(
            "SELECT COUNT(*) FROM work_items "
            "WHERE domain_kind='reward_integrity_audit'"
        ).fetchone()[0] == 0

        bundle = Path(str(completed.artifact_path))
        samples = bundle / "samples.jsonl"
        samples.write_text(samples.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="checksum mismatch"):
            rewards.create_audit(signal_shard_id=shard.id)
        assert database._conn.execute(
            "SELECT COUNT(*) FROM work_items "
            "WHERE domain_kind='reward_integrity_audit'"
        ).fetchone()[0] == 0
    finally:
        database.close()


def test_reward_audit_is_a_reviewable_exact_output_source(tmp_path: Path) -> None:
    database, scheduler, rewards, reward_root, shard = _audit_fixture(tmp_path)
    try:
        audit = rewards.create_audit(signal_shard_id=shard.id, submit=False)
        completed = rewards.execute_audit(
            audit.id,
            sentinel=lambda _sample: {"reward": 0.0, "passed": False},
            bootstrap_resamples=16,
        )
        public = PublicApiService(
            database=database,
            workstation_scheduler=scheduler,
            reward_integrity=rewards,
            reward_integrity_storage_root=reward_root,
            review_storage_root=tmp_path / "reviews",
            base_path=tmp_path,
        )

        records = list(
            public._review_acquisition_records(  # noqa: SLF001 - transport contract
                {
                    "sources": [
                        {"kind": "reward_integrity_audit", "ref": completed.id}
                    ]
                }
            )
        )

        assert len(records) == 20
        assert records[0]["record"]["output"] == "ok"
        assert records[0]["evidence"]["outcome"] == "optimizer_only_accept"
        assert records[0]["evidence"]["verifier_disagreement"] is True
        assert records[0]["source"]["kind"] == "reward_integrity_audit"
        assert records[0]["source"]["trace_hash"] == shard.trace_hash
    finally:
        database.close()


def test_public_comparison_pages_exact_matched_and_aggregate_evidence(
    tmp_path: Path,
) -> None:
    database, scheduler, rewards, reward_root, shard = _audit_fixture(tmp_path)
    try:
        base = rewards.create_audit(
            signal_shard_id=shard.id,
            runtime_identity={"comparison": "base"},
            submit=False,
        )
        candidate = rewards.create_audit(
            signal_shard_id=shard.id,
            runtime_identity={"comparison": "candidate"},
            submit=False,
        )
        for audit, reward in ((base, 0.25), (candidate, 0.75)):
            page = rewards.store.list_samples(audit.id, limit=100)
            rewards.add_audit_evidence(
                audit.id,
                [
                    {
                        "snapshot_id": sample.snapshot_id,
                        "primary_sentinel": {
                            "reward": reward,
                            "passed": reward >= 0.5,
                        },
                    }
                    for sample in page.items
                ],
            )

        public = PublicApiService(
            database=database,
            workstation_scheduler=scheduler,
            reward_integrity=rewards,
            reward_integrity_storage_root=reward_root,
            base_path=tmp_path,
        )
        exact = public.compare_reward_integrity_audits(
            base.id, candidate.id, limit=3, offset=4
        )
        assert exact["pairing"] == "paired_snapshot"
        assert exact["pair_total"] == 20
        assert (exact["limit"], exact["offset"], len(exact["pairs"])) == (3, 4, 3)
        assert all(pair["left_snapshot_id"] == pair["right_snapshot_id"] for pair in exact["pairs"])
        assert all(pair["same_output"] for pair in exact["pairs"])
        assert exact["pairs"][0]["left"]["primary_sentinel_observation"]["normalized_reward"] == 0.25
        assert exact["pairs"][0]["right"]["primary_sentinel_observation"]["normalized_reward"] == 0.75

        source_samples = rewards.store.list_samples(base.id, limit=100).items
        matched_snapshots = [
            TrainingSignalSnapshot.create(
                record=TrainingRecordRef(
                    record_id=sample.record_id,
                    record_hash=sample.record_hash,
                    instance_id=sample.instance_id,
                    source_index=sample.ordinal,
                    virtual=True,
                    source={"group_id": sample.group_id, **sample.lineage},
                ),
                candidate_ordinal=sample.candidate_ordinal,
                prompt=sample.input.get("prompt"),
                output=f"candidate output {sample.ordinal}",
                training_observation={"reward": 0.5, "passed": True},
                run_id="audited-run",
                segment_id="segment-next",
                boundary="step:2",
                producer_model_hash="producer-model-hash-next",
                checkpoint_hash="checkpoint-hash-next",
            )
            for sample in source_samples
        ]
        matched_shard = rewards.create_signal_shard(
            run_id="audited-run",
            segment_id="segment-next",
            reward_system_revision_id=shard.reward_system_revision_id,
            protocol_revision_id=shard.protocol_revision_id,
            capability_id=shard.capability_id,
            capture_fidelity="exact",
            boundary_unit="step",
            boundary_value=2,
            snapshots=matched_snapshots,
            aggregate={"event_count": len(matched_snapshots)},
            dataset_identity={"kind": "fixture"},
            producer_model_hash="producer-model-hash-next",
            checkpoint_hash="checkpoint-hash-next",
            runtime_identity={"fixture": True},
        )
        matched_audit = rewards.create_audit(
            signal_shard_id=matched_shard.id, submit=False
        )
        matched = public.compare_reward_integrity_audits(
            base.id, matched_audit.id, limit=4, offset=5
        )
        assert matched["pairing"] == "matched_input"
        assert matched["pair_total"] == 20
        assert len(matched["pairs"]) == 4
        assert all(pair["snapshot_id"] is None for pair in matched["pairs"])
        assert all(pair["left_snapshot_id"] != pair["right_snapshot_id"] for pair in matched["pairs"])
        assert "non-causal" in matched["pairing_reason"]

        aggregate_shard = rewards.create_signal_shard(
            run_id="audited-run",
            segment_id="segment-aggregate",
            reward_system_revision_id=shard.reward_system_revision_id,
            protocol_revision_id=shard.protocol_revision_id,
            capability_id=shard.capability_id,
            capture_fidelity="aggregate_only",
            boundary_unit="final",
            boundary_value=3,
            snapshots=[],
            aggregate={"event_count": 20, "reward_mean": 0.5},
            dataset_identity={"kind": "fixture"},
            producer_model_hash="aggregate-model-hash",
            checkpoint_hash="aggregate-checkpoint-hash",
            runtime_identity={"fixture": True},
        )
        aggregate_audit = rewards.create_audit(
            signal_shard_id=aggregate_shard.id, submit=False
        )
        aggregate = public.compare_reward_integrity_audits(
            base.id, aggregate_audit.id, limit=10, offset=0
        )
        assert aggregate["pairing"] == "aggregate_only"
        assert aggregate["pair_total"] == 0
        assert aggregate["pairs"] == []
        assert "no retained per-output evidence" in aggregate["pairing_reason"]
    finally:
        database.close()


def test_reward_audit_review_source_refuses_protected_trace_lineage(
    tmp_path: Path,
) -> None:
    database, scheduler, rewards, reward_root, shard = _audit_fixture(tmp_path)
    try:
        snapshots = [
            json.loads(line)
            for line in (Path(shard.storage_path) / "snapshots.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        protected_shard = rewards.create_signal_shard(
            run_id="audited-run",
            segment_id="segment-protected",
            reward_system_revision_id=shard.reward_system_revision_id,
            protocol_revision_id=shard.protocol_revision_id,
            capability_id=shard.capability_id,
            capture_fidelity="exact",
            boundary_unit="final",
            boundary_value=1,
            snapshots=snapshots,
            aggregate={"event_count": len(snapshots)},
            dataset_identity={"bindings": [{"role": "train", "split": "test"}]},
            producer_model_hash=shard.producer_model_hash,
            checkpoint_hash="protected-checkpoint-hash",
            runtime_identity={"fixture": True},
        )
        audit = rewards.create_audit(
            signal_shard_id=protected_shard.id, submit=False
        )
        completed = rewards.execute_audit(
            audit.id,
            sentinel=lambda _sample: {"reward": 0.0, "passed": False},
            bootstrap_resamples=16,
        )
        public = PublicApiService(
            database=database,
            workstation_scheduler=scheduler,
            reward_integrity=rewards,
            reward_integrity_storage_root=reward_root,
            review_storage_root=tmp_path / "reviews",
            base_path=tmp_path,
        )

        with pytest.raises(ValueError, match="protected_split:test"):
            list(
                public._review_acquisition_records(  # noqa: SLF001
                    {
                        "sources": [
                            {
                                "kind": "reward_integrity_audit",
                                "ref": completed.id,
                            }
                        ]
                    }
                )
            )
    finally:
        database.close()


def test_published_audit_bundle_projects_completed_lifecycle(tmp_path: Path) -> None:
    database, _scheduler, rewards, _reward_root, shard = _audit_fixture(tmp_path)
    try:
        audit = rewards.create_audit(signal_shard_id=shard.id, submit=False)
        completed = rewards.execute_audit(
            audit.id,
            sentinel=lambda _sample: {"reward": 1.0, "passed": True},
            bootstrap_resamples=16,
        )

        document = json.loads(
            (Path(str(completed.artifact_path)) / "audit.json").read_text(
                encoding="utf-8"
            )
        )
        assert document["status"] == "completed"
        assert document["stage"] == "published"
        assert document["processed_samples"] == completed.processed_samples == 20
        assert document["artifact_path"] == completed.artifact_path
    finally:
        database.close()


def test_completed_audit_reuse_refuses_current_sentinel_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, _scheduler, rewards, _reward_root, shard = _audit_fixture(tmp_path)
    try:
        audit = rewards.create_audit(signal_shard_id=shard.id, submit=False)
        completed = rewards.execute_audit(
            audit.id,
            sentinel=lambda _sample: {"reward": 1.0, "passed": True},
            bootstrap_resamples=16,
        )
        checked: list[str] = []

        def stale_runtime(
            _service: VerifierLabService,
            revision_id: str,
            actual: dict | None = None,
        ) -> dict:
            del actual
            checked.append(revision_id)
            return {
                "verifier_revision_id": revision_id,
                "state": "stale_runtime",
                "compatible": False,
                "mismatches": [{"field": "implementation_identity"}],
            }

        monkeypatch.setattr(
            VerifierLabService, "runtime_compatibility", stale_runtime
        )
        with pytest.raises(
            ValueError,
            match="completed reward-integrity audit reuse refuses stale sentinel",
        ):
            rewards.create_audit(signal_shard_id=shard.id)

        assert checked
        assert completed.status == "completed"
        assert database._conn.execute(
            "SELECT COUNT(*) FROM work_items "
            "WHERE domain_kind='reward_integrity_audit'"
        ).fetchone()[0] == 0
    finally:
        database.close()


def test_reward_replay_sync_keeps_legacy_manifests_read_only(tmp_path: Path) -> None:
    path = tmp_path / "replay.json"
    path.write_text(
        '{"manifest_version":3,"run_id":"legacy","config":{"kept":true}}\n',
        encoding="utf-8",
    )
    before = path.read_bytes()
    result = sync_reward_integrity_decision(
        path,
        run_id="legacy",
        audit={
            "id": "audit-legacy",
            "status": "completed",
            "manifest_hash": "audit-hash",
            "integrity_profile_revision_id": "profile",
            "work_item_id": None,
        },
        decision={
            "id": "decision-legacy",
            "decision": "pass",
            "action": "continue",
        },
    )
    assert result["status"] == "legacy_read_only"
    assert path.read_bytes() == before


def test_reward_audits_are_globally_serialized_even_on_cpu(tmp_path: Path) -> None:
    database, scheduler, rewards, _reward_root, shard = _audit_fixture(tmp_path)
    second_scheduler = WorkstationScheduler(database, worker_id="second-v8-worker")
    try:
        first = rewards.create_audit(
            signal_shard_id=shard.id, runtime_identity={"variant": "first"}
        )
        second = rewards.create_audit(
            signal_shard_id=shard.id, runtime_identity={"variant": "second"}
        )
        claimed_first = scheduler.claim(work_item_id=str(first.work_item_id))
        assert claimed_first is not None
        assert second_scheduler.claim(work_item_id=str(second.work_item_id)) is None

        completed_first = scheduler.complete(
            claimed_first, result={"test_only": True}
        )
        assert completed_first is not None and completed_first.status == "completed"
        claimed_second = second_scheduler.claim(work_item_id=str(second.work_item_id))
        assert claimed_second is not None
        second_scheduler.cancel(claimed_second.id)
        cancelled_second = second_scheduler.complete(
            claimed_second, result={"test_only": True}
        )
        assert cancelled_second is not None
        assert cancelled_second.status == "cancelled"
    finally:
        database.close()


def test_http_launch_worker_pause_and_review_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from halo_forge.auth.dependency import reset_store_for_tests
    from halo_forge.public_api import app as app_module
    from halo_forge.reward_integrity import runtime as reward_runtime

    database, scheduler, rewards, reward_root, shard = _audit_fixture(
        tmp_path, real_checkpoint=True
    )
    public = PublicApiService(
        database=database,
        app_state=AppState(),
        base_path=tmp_path,
        evaluation_storage_root=tmp_path / "evaluations",
        reward_integrity=rewards,
        reward_integrity_storage_root=reward_root,
        workstation_scheduler=scheduler,
    )
    parent_job = public.app_state.create_job(
        "raft",
        "Audited parent run",
        output_dir=Path(database.get_run("audited-run").output_dir),
        job_id="audited-run",
    )
    parent_job.launch_args = {
        "mode": "raft",
        "model": "fixture-model",
        "prompts": str(tmp_path / "prompts.jsonl"),
        "output_dir": str(tmp_path / "audited-run"),
        "cycles": 2,
        "seed": 42,
        "reward_system_revision_id": shard.reward_system_revision_id,
        "reward_audit_protocol_revision_id": shard.protocol_revision_id,
        "reward_integrity_profile_revision_id": rewards.default_ids[
            "profile:human_aligned_integrity"
        ],
        "reward_audit_boundaries": ["final"],
    }
    monkeypatch.setattr(app_module, "PublicApiService", lambda: public)
    monkeypatch.setenv("HALOFORGE_DISABLE_AUTO_WORKER", "1")
    reset_store_for_tests(None)

    def fast_inverted_audit(
        active_database: RunDatabase,
        audit_id: str,
        *,
        root: str | None = None,
        bootstrap_resamples: int = 10_000,
    ):
        del bootstrap_resamples
        service = RewardIntegrityService(active_database, root=root)
        return service.execute_audit(
            audit_id,
            sentinel=lambda _sample: {"reward": 0.0, "passed": False},
            bootstrap_resamples=16,
        )

    monkeypatch.setattr(reward_runtime, "execute_pinned_audit", fast_inverted_audit)
    try:
        with TestClient(app_module.create_app(serve_frontend=False)) as client:
            capabilities = client.get(
                "/api/public/reward-integrity-capabilities"
            )
            assert capabilities.status_code == 200, capabilities.text
            assert capabilities.json()["schema_version"] == 11

            invalid_system = client.post(
                "/api/public/reward-systems",
                json={
                    "name": "invalid-api-system",
                    "definition": {
                        "optimizer_verifier_revision_id": "missing",
                        "auditors": [],
                    },
                },
            )
            assert invalid_system.status_code == 400, invalid_system.text
            assert (
                database._conn.execute(
                    "SELECT COUNT(*) FROM reward_systems "
                    "WHERE name='invalid-api-system'"
                ).fetchone()[0]
                == 0
            )

            source_system = rewards.get_system_revision(
                shard.reward_system_revision_id
            )
            definition = {
                "optimizer_verifier_revision_id": (
                    source_system.optimizer_verifier_revision_id
                ),
                "modality": "text",
                "task_type": "binary",
                "auditors": [
                    {
                        "role": "primary_sentinel",
                        "verifier_revision_id": (
                            source_system.primary_sentinel.verifier_revision_id
                        ),
                    }
                ],
            }
            created_system = client.post(
                "/api/public/reward-systems",
                json={"name": "API reward system", "definition": definition},
            )
            assert created_system.status_code == 201, created_system.text
            system_payload = created_system.json()
            revised_system = client.post(
                "/api/public/reward-systems/"
                f"{system_payload['system']['id']}/revisions",
                json={
                    "definition": {
                        **definition,
                        "reward_mapping": {"scale": 0.75},
                    }
                },
            )
            assert revised_system.status_code == 201, revised_system.text
            assert revised_system.json()["reward_mapping"]["scale"] == 0.75

            launched = client.post(
                "/api/public/reward-integrity-audits",
                json={
                    "signal_shard_id": shard.id,
                    "integrity_profile_revision_id": rewards.default_ids[
                        "profile:human_aligned_integrity"
                    ],
                },
            )
            assert launched.status_code == 202, launched.text
            launch_payload = launched.json()
            audit_id = str(launch_payload["id"])
            work_item_id = str(launch_payload["work_item_id"])
            work = database.get_work_item(work_item_id)
            assert work is not None
            assert work.domain_id == audit_id
            assert work.launch_spec["handler"] == "reward_integrity.execute_audit"
            assert rewards.get_audit(audit_id).work_item_id == work_item_id

            missing_retry_reason = client.post(
                f"/api/public/reward-integrity-audits/{audit_id}/retry",
                json={},
            )
            assert missing_retry_reason.status_code == 400
            claimed_for_retry = scheduler.claim(work_item_id=work_item_id)
            assert claimed_for_retry is not None
            failed_for_retry = scheduler.fail(
                claimed_for_retry, error="fixture failure before audit execution"
            )
            assert failed_for_retry is not None
            assert failed_for_retry.status == "failed"
            rewards.store.update_audit(
                audit_id,
                status="failed",
                stage="failed",
                error="fixture failure before audit execution",
            )
            retry_reason = "Retry after reviewing the fixture worker failure."
            retried = client.post(
                f"/api/public/reward-integrity-audits/{audit_id}/retry",
                json={"reason": retry_reason},
            )
            assert retried.status_code == 202, retried.text
            retry_events = LabV4Catalog(database).list_events(
                work_item_id=work_item_id, limit=100
            )
            assert any(
                event.event_type == "retry_queued"
                and event.to_dict()["payload"]["reason"] == retry_reason
                for event in retry_events
            )

            run_output = Path(database.get_run("audited-run").output_dir)
            run_output.mkdir(parents=True, exist_ok=True)
            published_checkpoint = run_output / "final_model"
            published_checkpoint.mkdir(exist_ok=True)
            (published_checkpoint / "adapter_config.json").write_text(
                '{"base_model_name_or_path":"fixture-model"}', encoding="utf-8"
            )
            (published_checkpoint / "adapter_model.safetensors").write_bytes(b"fixture")
            (run_output / "training_summary.json").write_text(
                json.dumps({"final_model_path": str(published_checkpoint)}),
                encoding="utf-8",
            )
            replay_path = run_output / "replay.json"
            replay_path.write_text(
                json.dumps(
                    {
                        "manifest_version": 4,
                        "run_id": "audited-run",
                        "config": {"preserved": True},
                        "reward_integrity": {
                            "reward_system_revision_id": (
                                shard.reward_system_revision_id
                            ),
                            "audit_decisions": [
                                {
                                    "audit_id": audit_id,
                                    "status": "queued",
                                    "work_item_id": work_item_id,
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            finished = WorkstationWorker(
                scheduler, heartbeat_interval=0.01
            ).run_once()
            assert finished is not None and finished.status == "completed"
            assert finished.result["decision"]["action"] == "pause"
            assert finished.result["requires_review"] is True
            assert finished.result["replay_sync"]["status"] == "updated"
            assert rewards.get_audit(audit_id).status == "completed"
            assert database.get_run("audited-run").status == "awaiting_review"
            synced_replay = json.loads(replay_path.read_text(encoding="utf-8"))
            assert synced_replay["config"] == {"preserved": True}
            replay_decision = synced_replay["reward_integrity"][
                "audit_decisions"
            ][0]
            assert replay_decision["audit_id"] == audit_id
            assert replay_decision["audit_manifest_hash"] == rewards.get_audit(
                audit_id
            ).manifest_hash
            assert replay_decision["decision"] == "fail"
            assert replay_decision["decision_id"] == finished.result["decision"]["id"]
            assert replay_decision["action"] == "pause"
            assert replay_decision["reasons"] == finished.result["decision"]["reasons"]
            assert replay_decision["result"] == "fail"
            assert replay_decision["status"] == "completed"
            assert replay_decision["work_item_id"] == work_item_id
            assert len(replay_decision["decision_history"]) == 1
            assert replay_decision["decision_history"][0]["action"] == "pause"
            assert not list(run_output.glob(".replay.json.*.tmp"))

            from halo_forge.public_api.activity import activity_item_view

            activity = activity_item_view(
                database, LabV4Catalog(database), finished
            )
            assert activity["status"] == "awaiting_review"
            assert {"open_audit", "continue", "stop", "fork"} <= set(
                activity["next_actions"]
            )
            assert activity["action_links"][0]["href"].startswith(
                "/runs/audited-run?"
            )

            detail = client.get(
                f"/api/public/reward-integrity-audits/{audit_id}"
            )
            assert detail.status_code == 200, detail.text
            detail_payload = detail.json()
            assert detail_payload["audit"]["status"] == "completed"
            assert detail_payload["latest_decision"]["decision"] == "fail"
            assert detail_payload["latest_decision"]["action"] == "pause"

            work_count = database._conn.execute(
                "SELECT COUNT(*) FROM work_items "
                "WHERE domain_kind='reward_integrity_audit'"
            ).fetchone()[0]
            reused = client.post(
                "/api/public/reward-integrity-audits",
                json={
                    "signal_shard_id": shard.id,
                    "integrity_profile_revision_id": rewards.default_ids[
                        "profile:human_aligned_integrity"
                    ],
                },
            )
            assert reused.status_code == 202, reused.text
            assert reused.json()["id"] == audit_id
            assert reused.json()["work_item_id"] == work_item_id
            assert reused.json()["reused"] is True
            assert (
                database._conn.execute(
                    "SELECT COUNT(*) FROM work_items "
                    "WHERE domain_kind='reward_integrity_audit'"
                ).fetchone()[0]
                == work_count
            )

            reviewed = client.post(
                f"/api/public/reward-integrity-audits/{audit_id}/review",
                json={
                    "action": "continue",
                    "reason": "Reviewed the intentionally inverted test sentinel.",
                },
            )
            assert reviewed.status_code == 200, reviewed.text
            assert reviewed.json()["action"] == "continue"
            assert reviewed.json()["override"] is True
            assert reviewed.json()["replay_sync"]["status"] == "updated"
            assert database.get_run("audited-run").status == "running"
            continued_replay = json.loads(replay_path.read_text(encoding="utf-8"))
            continued_decision = continued_replay["reward_integrity"][
                "audit_decisions"
            ][0]
            assert continued_decision["action"] == "continue"
            assert [
                item["action"] for item in continued_decision["decision_history"]
            ] == ["pause", "continue"]
            assert continued_decision["decision_history"][0]["decision_id"] == (
                finished.result["decision"]["id"]
            )
            resolved_activity = activity_item_view(
                database, LabV4Catalog(database), finished
            )
            assert resolved_activity["status"] == "completed"
            assert "continue" not in resolved_activity["next_actions"]

            stopped = client.post(
                f"/api/public/reward-integrity-audits/{audit_id}/review",
                json={
                    "action": "stop",
                    "reason": "Stop the branch after the fixture review.",
                },
            )
            assert stopped.status_code == 200, stopped.text
            assert stopped.json()["action"] == "stop"
            assert stopped.json()["replay_sync"]["status"] == "updated"

            forked = client.post(
                f"/api/public/reward-integrity-audits/{audit_id}/review",
                json={
                    "action": "fork",
                    "reason": "Fork the fixture checkpoint for the contract test.",
                },
            )
            assert forked.status_code == 200, forked.text
            assert forked.json()["decision"]["action"] == "fork"
            assert forked.json()["replay_sync"]["status"] == "updated"
            assert forked.json()["href"] == (
                f"/train?fork_reward_audit={audit_id}"
            )
            assert forked.json()["train_context"]["parent_run_id"] == "audited-run"
            assert forked.json()["train_context"][
                "source_reward_integrity_audit_id"
            ] == audit_id
            assert forked.json()["train_context"][
                "source_reward_integrity_decision_id"
            ] == forked.json()["decision"]["id"]
            assert forked.json()["train_context"][
                "reward_system_revision_id"
            ] == shard.reward_system_revision_id

            restored = client.get(
                f"/api/public/reward-integrity-audits/{audit_id}/fork-context"
            )
            assert restored.status_code == 200, restored.text
            assert restored.json()["decision"]["id"] == forked.json()["decision"]["id"]
            assert restored.json()["train_context"] == forked.json()["train_context"]
            verified_fork = public._resolve_reward_fork_payload(
                {
                    key: value
                    for key, value in restored.json()["train_context"].items()
                    if key != "checkpoint"
                },
                verify_checkpoint=True,
            )
            assert verified_fork["fork_checkpoint_hash"] == shard.checkpoint_hash

            tampered = client.post(
                "/api/public/train/preflight",
                json={
                    "source_reward_integrity_audit_id": audit_id,
                    "fork_checkpoint_hash": "different-checkpoint",
                },
            )
            assert tampered.status_code == 400, tampered.text
            assert "fork_checkpoint_hash conflicts" in tampered.json()["detail"]
            final_replay = json.loads(replay_path.read_text(encoding="utf-8"))
            final_decision = final_replay["reward_integrity"]["audit_decisions"][0]
            assert final_decision["action"] == "fork"
            assert [
                item["action"] for item in final_decision["decision_history"]
            ] == ["pause", "continue", "stop", "fork"]
            assert final_decision["decision_history"][0]["decision_id"] == (
                finished.result["decision"]["id"]
            )
    finally:
        database.close()


def test_reward_cli_parser_validates_before_create_and_revises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from halo_forge import cli

    database_path = tmp_path / "cli.db"
    database = RunDatabase(str(database_path))
    verifiers = VerifierLabService(database, root=tmp_path / "calibrations")
    optimizer_id = _verifier_revision(
        verifiers, name="CLI optimizer", pattern=r"^ok$"
    )
    sentinel_id = _verifier_revision(
        verifiers, name="CLI sentinel", pattern=r"^accepted$"
    )
    database.close()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "reward",
            "audit",
            "retry",
            "audit-without-reason",
            "--database",
            str(database_path),
            "--json",
        ],
    )
    with pytest.raises(SystemExit) as missing_reason:
        cli.main()
    assert missing_reason.value.code == 2
    capsys.readouterr()

    captured_retry: dict[str, str] = {}

    def capture_retry(
        _service: RewardIntegrityService, audit_id: str, *, reason: str
    ) -> dict[str, str]:
        captured_retry.update(audit_id=audit_id, reason=reason)
        return {"id": audit_id, "status": "queued"}

    monkeypatch.setattr(RewardIntegrityService, "retry_audit", capture_retry)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "reward",
            "audit",
            "retry",
            "audit-with-reason",
            "--reason",
            "Retry after inspecting the CLI fixture failure.",
            "--database",
            str(database_path),
            "--json",
        ],
    )
    cli.main()
    assert captured_retry == {
        "audit_id": "audit-with-reason",
        "reason": "Retry after inspecting the CLI fixture failure.",
    }
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "reward",
            "system",
            "create",
            "--name",
            "invalid-system",
            "--optimizer-verifier-revision",
            "missing-optimizer",
            "--primary-sentinel-revision",
            "missing-sentinel",
            "--database",
            str(database_path),
            "--json",
        ],
    )
    with pytest.raises(ValueError, match="optimizer_verifier_revision_unknown"):
        cli.main()
    check = RunDatabase(str(database_path))
    assert (
        check._conn.execute(
            "SELECT COUNT(*) FROM reward_systems WHERE name='invalid-system'"
        ).fetchone()[0]
        == 0
    )
    check.close()
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "reward",
            "system",
            "create",
            "--name",
            "cli-system",
            "--optimizer-verifier-revision",
            optimizer_id,
            "--primary-sentinel-revision",
            sentinel_id,
            "--database",
            str(database_path),
            "--json",
        ],
    )
    cli.main()
    created = json.loads(capsys.readouterr().out)
    system_id = str(created["system"]["id"])
    first_revision_id = str(created["revision"]["id"])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "reward",
            "system",
            "revise",
            system_id,
            "--optimizer-verifier-revision",
            optimizer_id,
            "--primary-sentinel-revision",
            sentinel_id,
            "--reward-mapping",
            '{"scale": 0.5}',
            "--database",
            str(database_path),
            "--json",
        ],
    )
    cli.main()
    revised = json.loads(capsys.readouterr().out)
    assert revised["id"] != first_revision_id
    assert revised["reward_mapping"]["scale"] == 0.5
