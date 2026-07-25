from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from halo_forge.reward_integrity import (
    IntegrityEvidence,
    RewardMappedVerifier,
    RewardIntegrityService,
    RewardIntegrityStorage,
    compute_integrity_metrics,
    normalize_reward,
)
from halo_forge.rlvr.verifiers.base import VerifyResult
from halo_forge.run_db import RunDatabase, RunRecord
from halo_forge.run_db.schema import SCHEMA_VERSION
from halo_forge.training_signal import (
    TRAINING_SIGNAL_CAPABILITIES,
    TrainingRecordRef,
    TrainingSignalSink,
)
from halo_forge.verifier_lab.store import VerifierLabStore


def _verifier_revision(db: RunDatabase, name: str, fingerprint: str, *, task_type: str = "binary"):
    store = VerifierLabStore(db)
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
            "task_type": task_type,
            "reward_contract": {
                "minimum": 0,
                "maximum": 1,
                "direction": "maximize",
                "threshold": 0.5,
            },
            "runtime_requirements": {},
        },
    )
    db._conn.execute(
        "INSERT INTO verifier_aliases (profile_id,alias,revision_id,updated_at) "
        "VALUES (?,?,?,'now')",
        (profile.id, "candidate", revision.id),
    )
    db._conn.commit()
    return revision


def test_v11_schema_is_additive_and_keeps_verifier_tables(tmp_path: Path):
    path = tmp_path / "runs.db"
    db = RunDatabase(str(path))
    revision = _verifier_revision(db, "optimizer", "optimizer-fingerprint")
    verifier_sql = db._conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='verifier_profile_revisions'"
    ).fetchone()[0]
    db._conn.execute("UPDATE schema_meta SET value='10' WHERE key='schema_version'")
    db._conn.commit()
    db.close()

    migrated = RunDatabase(str(path))
    assert SCHEMA_VERSION == 23
    assert (
        migrated._conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0]
        == "23"
    )
    assert migrated._conn.execute(
        "SELECT 1 FROM verifier_profile_revisions WHERE id=?", (revision.id,)
    ).fetchone()
    assert (
        migrated._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='verifier_profile_revisions'"
        ).fetchone()[0]
        == verifier_sql
    )
    tables = {
        row[0]
        for row in migrated._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "reward_systems",
        "reward_system_revisions",
        "reward_system_auditors",
        "reward_audit_protocol_revisions",
        "reward_integrity_profile_revisions",
        "direct_run_segments",
        "training_signal_shards",
        "reward_integrity_audits",
        "reward_integrity_samples",
        "reward_integrity_observations",
        "reward_integrity_metrics",
        "reward_integrity_decisions",
        "reward_integrity_bindings",
    } <= tables


def test_reward_system_revisions_are_deduplicated_immutable_and_paged(tmp_path: Path):
    db = RunDatabase(":memory:")
    optimizer = _verifier_revision(db, "optimizer", "optimizer-fingerprint")
    sentinel = _verifier_revision(db, "sentinel", "sentinel-fingerprint")
    service = RewardIntegrityService(db, root=tmp_path)
    assert len(service.capabilities()["items"]) == 8
    system = service.create_system(name="Production reward")
    revision = service.create_system_revision(
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
        reward_mapping={"scale": 1.0, "center": 0.0},
    )
    same = service.create_system_revision(
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
        reward_mapping={"scale": 1.0, "center": 0.0},
    )
    assert same.id == revision.id
    assert revision.primary_sentinel is not None
    assert revision.primary_sentinel.correlated is False
    assert revision.reward_mapping["normalization"] == {
        "minimum": 0.0,
        "maximum": 1.0,
        "direction": "maximize",
    }
    assert revision.definition["compatible_training_signal_capabilities"] == [
        "grpo:hf",
        "grpo:mlx",
        "raft:hf",
        "raft:mlx",
        "reasoning:hf",
    ]
    assert service.list_systems(limit=1, offset=0).to_dict() == {
        "items": [service.get_system(system.id).to_dict()],
        "total": 1,
        "limit": 1,
        "offset": 0,
    }
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db._conn.execute(
            "UPDATE reward_system_revisions SET modality='audio' WHERE id=?",
            (revision.id,),
        )


def test_reward_system_pins_and_enforces_training_signal_capabilities(tmp_path: Path):
    db = RunDatabase(":memory:")
    optimizer = _verifier_revision(db, "capability-optimizer", "optimizer-fingerprint")
    sentinel = _verifier_revision(db, "capability-sentinel", "sentinel-fingerprint")
    service = RewardIntegrityService(db, root=tmp_path)
    created = service.create_system(
        name="Reasoning-only reward",
        definition={
            "optimizer_verifier_revision_id": optimizer.id,
            "modality": "text",
            "task_type": "binary",
            "compatible_training_signal_capabilities": ["reasoning:hf"],
            "auditors": [
                {
                    "role": "primary_sentinel",
                    "verifier_revision_id": sentinel.id,
                }
            ],
        },
    )
    revision_id = created["revision"]["id"]
    allowed = service.resolve_binding(
        revision_id,
        protocol_revision_id=service.default_ids["protocol:balanced_256"],
        integrity_profile_revision_id=service.default_ids[
            "profile:human_aligned_integrity"
        ],
        trainer="reasoning",
        backend="hf",
        boundaries=["final"],
    )
    assert "training_signal_capability_incompatible" not in allowed.blockers
    refused = service.resolve_binding(
        revision_id,
        protocol_revision_id=service.default_ids["protocol:balanced_256"],
        integrity_profile_revision_id=service.default_ids[
            "profile:human_aligned_integrity"
        ],
        trainer="grpo",
        backend="hf",
        boundaries=["final"],
    )
    assert "training_signal_capability_incompatible" in refused.blockers

    with pytest.raises(ValueError, match="training_signal_capability_unknown"):
        service.create_system(
            name="Unknown trainer reward",
            definition={
                "optimizer_verifier_revision_id": optimizer.id,
                "modality": "text",
                "task_type": "binary",
                "compatible_training_signal_capabilities": ["future:backend"],
                "auditors": [
                    {
                        "role": "primary_sentinel",
                        "verifier_revision_id": sentinel.id,
                    }
                ],
            },
        )


def test_reward_system_revision_recursively_scrubs_credentials(tmp_path: Path):
    db = RunDatabase(":memory:")
    optimizer = _verifier_revision(db, "scrub-optimizer", "optimizer-fingerprint")
    sentinel = _verifier_revision(db, "scrub-sentinel", "sentinel-fingerprint")
    service = RewardIntegrityService(db, root=tmp_path)
    system = service.create_system(name="Credential-free reward")
    revision = service.create_system_revision(
        system.id,
        optimizer_verifier_revision_id=optimizer.id,
        modality="text",
        task_type="binary",
        auditors=[
            {
                "role": "primary_sentinel",
                "verifier_revision_id": sentinel.id,
                "configuration": {
                    "endpoint": "https://user:pass@example.test/v1?api_key=secret&mode=audit",
                    "headers": {"authorization": "Bearer secret", "x-mode": "audit"},
                },
            }
        ],
        definition={"provider": {"token": "secret", "endpoint_type": "hosted"}},
    )
    assert revision.definition == {
        "provider": {"endpoint_type": "hosted"},
        "compatible_training_signal_capabilities": [
            "grpo:hf",
            "grpo:mlx",
            "raft:hf",
            "raft:mlx",
            "reasoning:hf",
        ],
    }
    assert revision.primary_sentinel is not None
    assert revision.primary_sentinel.configuration == {
        "endpoint": "https://example.test/v1?mode=audit",
        "headers": {"x-mode": "audit"},
    }
    persisted = json.dumps(revision.to_dict(), sort_keys=True)
    assert "secret" not in persisted
    assert "user:pass" not in persisted
    binding = service.bind(
        reward_system_revision_id=revision.id,
        domain_kind="run",
        domain_id="credential-free-run",
        context={"provider": {"api_key": "secret", "mode": "audit"}},
    )
    assert binding.context == {"provider": {"mode": "audit"}}


def test_guided_reward_binding_excludes_overridden_verifier_aliases(tmp_path: Path):
    db = RunDatabase(":memory:")
    optimizer = _verifier_revision(db, "alias-optimizer", "optimizer-fingerprint")
    sentinel = _verifier_revision(db, "alias-sentinel", "sentinel-fingerprint")
    service = RewardIntegrityService(db, root=tmp_path)
    created = service.create_system(
        name="Override-inspection reward",
        definition={
            "optimizer_verifier_revision_id": optimizer.id,
            "modality": "text",
            "task_type": "binary",
            "auditors": [
                {
                    "role": "primary_sentinel",
                    "verifier_revision_id": sentinel.id,
                }
            ],
        },
    )
    db._conn.execute(
        "INSERT INTO verifier_alias_events "
        "(id,profile_id,alias,previous_revision_id,revision_id,override,note,created_at) "
        "VALUES (?,?,?,?,?,1,?,'now')",
        (
            "overridden-sentinel-alias",
            sentinel.profile_id,
            "candidate",
            None,
            sentinel.id,
            "inspection only",
        ),
    )
    db._conn.commit()
    resolved = service.resolve_binding(
        created["revision"]["id"],
        protocol_revision_id=service.default_ids["protocol:balanced_256"],
        integrity_profile_revision_id=service.default_ids[
            "profile:human_aligned_integrity"
        ],
        trainer="reasoning",
        backend="hf",
        boundaries=["final"],
    )
    assert resolved.gating_eligible is False
    assert (
        "primary_sentinel:verifier_qualification_override_excluded"
        in resolved.blockers
    )


def test_reward_mapping_is_applied_before_training_selection():
    class FakeVerifier:
        max_workers = 1

        def verify(self, *_args, **_kwargs):
            return VerifyResult(True, 0.75, "ok")

    verifier = RewardMappedVerifier(
        FakeVerifier(),
        {
            "normalization": {"minimum": 0.0, "maximum": 1.0},
            "scale": 0.5,
            "center": 0.25,
            "minimum": 0.0,
            "maximum": 1.0,
            "threshold": 0.7,
        },
    )
    result = verifier.verify("candidate")
    assert result.reward == pytest.approx(0.625)
    assert result.success is False
    assert result.metadata["raw_optimizer_reward"] == 0.75


def test_guided_reward_mapping_is_canonicalized_into_executable_contract(tmp_path: Path):
    db = RunDatabase(":memory:")
    optimizer = _verifier_revision(db, "mapping-optimizer", "optimizer-fingerprint")
    sentinel = _verifier_revision(db, "mapping-sentinel", "sentinel-fingerprint")
    service = RewardIntegrityService(db, root=tmp_path)
    created = service.create_system(
        name="Guided mapping",
        definition={
            "optimizer_verifier_revision_id": optimizer.id,
            "modality": "text",
            "task_type": "binary",
            "reward_mapping": {
                "minimum": 0,
                "maximum": 1,
                "threshold": 0.6,
                "normalization": "linear_0_1",
            },
            "failure_behavior": "fail_closed",
            "filtering": "optimizer_only",
            "scaling": "linear",
            "centering": "none",
            "keep_policy": "trainer_declared",
            "auditors": [
                {
                    "role": "primary_sentinel",
                    "verifier_revision_id": sentinel.id,
                }
            ],
        },
    )
    revision = service.get_system_revision(created["revision"]["id"])
    assert revision.reward_mapping["normalization"] == {
        "minimum": 0.0,
        "maximum": 1.0,
        "direction": "maximize",
    }
    assert revision.reward_mapping["failure_behavior"] == "reject"
    assert revision.reward_mapping["filtering"] == {"mode": "optimizer_only"}
    assert revision.reward_mapping["scaling"] == 1.0
    assert revision.reward_mapping["centering"] == 0.0
    assert "failure_behavior" not in revision.definition

    with pytest.raises(ValueError, match="outside the mapped reward range"):
        service.create_system_revision(
            created["system"]["id"],
            optimizer_verifier_revision_id=optimizer.id,
            modality="text",
            task_type="binary",
            reward_mapping={"threshold": 2.0},
            auditors=[
                {
                    "role": "primary_sentinel",
                    "verifier_revision_id": sentinel.id,
                }
            ],
        )


def test_normalization_metrics_and_grouped_bootstrap_are_deterministic():
    assert normalize_reward(5, {"minimum": 0, "maximum": 10, "direction": "maximize"}) == 0.5
    assert normalize_reward(2, {"minimum": 0, "maximum": 10, "direction": "minimize"}) == 0.8
    with pytest.raises(ValueError, match="outside"):
        normalize_reward(11, {"minimum": 0, "maximum": 10})

    evidence = [
        IntegrityEvidence(
            snapshot_id=f"sample-{index}",
            group_id=f"record-{index}",
            optimizer_reward=index / 99,
            sentinel_reward=index / 99,
            optimizer_passed=index >= 50,
            sentinel_passed=index >= 50,
        )
        for index in range(100)
    ]
    # A deliberately bad diagnostic does not leak into core population rates.
    evidence.append(
        IntegrityEvidence(
            snapshot_id="diagnostic",
            group_id="diagnostic-record",
            optimizer_reward=1.0,
            sentinel_reward=0.0,
            optimizer_passed=True,
            sentinel_passed=False,
            diagnostic=True,
        )
    )
    first = compute_integrity_metrics(
        "audit",
        evidence,
        {"minimum": 0, "maximum": 1},
        {"minimum": 0, "maximum": 1},
        bootstrap_resamples=250,
    )
    second = compute_integrity_metrics(
        "audit",
        evidence,
        {"minimum": 0, "maximum": 1},
        {"minimum": 0, "maximum": 1},
        bootstrap_resamples=250,
    )
    assert [value.to_dict() for value in first] == [value.to_dict() for value in second]
    by_name = {value.name: value for value in first}
    assert by_name["paired_coverage"].value == 1.0
    assert by_name["pass_agreement"].value == 1.0
    assert by_name["optimizer_only_acceptance"].value == 0.0
    assert by_name["pass_flip_rate"].value == 0.0
    assert by_name["spearman"].value == pytest.approx(1.0)
    assert by_name["kendall_tau"].value == pytest.approx(1.0)
    assert by_name["paired_coverage"].record_count == 100


def test_exact_bundle_publication_reuse_and_tamper_detection(tmp_path: Path):
    storage = RewardIntegrityStorage(tmp_path)
    target = tmp_path / "bundle"
    first = storage.publish(
        target,
        {"profile.json": {"name": "audit"}, "samples.jsonl": [{"id": 1}]},
        identity={"kind": "test", "id": "one"},
    )
    same = storage.publish(
        target,
        {"profile.json": {"name": "audit"}, "samples.jsonl": [{"id": 1}]},
        identity={"kind": "test", "id": "one"},
    )
    assert same.manifest_hash == first.manifest_hash
    (target / "samples.jsonl").write_text('{"id":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        storage.verify(target)


def test_storage_accepts_base_or_explicit_reward_audit_root(tmp_path: Path):
    base = tmp_path / ".halo-forge"
    explicit = base / "evaluations" / "reward-audits"
    from_base = RewardIntegrityStorage(base)
    from_explicit = RewardIntegrityStorage(explicit)
    assert from_base.audit_path("audit-1") == explicit / "audit-1"
    assert from_explicit.audit_path("audit-1") == explicit / "audit-1"
    assert from_base.signal_path("run-1", "segment-1", "trace-1") == (
        base / "training-signals" / "run-1" / "segment-1" / "trace-1"
    )
    assert from_explicit.signal_path("run-1", "segment-1", "trace-1") == (
        base / "training-signals" / "run-1" / "segment-1" / "trace-1"
    )


def test_sealed_training_signal_to_same_output_audit(tmp_path: Path):
    db = RunDatabase(":memory:")
    optimizer = _verifier_revision(db, "optimizer", "optimizer-fingerprint")
    sentinel = _verifier_revision(db, "sentinel", "sentinel-fingerprint")
    service = RewardIntegrityService(db, root=tmp_path / "managed")
    system = service.create_system(name="Audited reward")
    system_revision = service.create_system_revision(
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
    db.upsert_run(
        RunRecord(
            run_id="run-1",
            modality="raft",
            model_name="model",
            status="running",
            output_dir=str(tmp_path / "run-1"),
            indexed_at="now",
        )
    )
    segment = db.create_direct_run_segment(
        run_id="run-1",
        ordinal=1,
        unit="cycle",
        start_value=0,
        end_value=1,
    )
    capability = TRAINING_SIGNAL_CAPABILITIES.get("raft:hf")
    sink = TrainingSignalSink(
        tmp_path / "trainer-signals",
        run_id="run-1",
        segment_id="segment-1",
        boundary="cycle:1",
        capability=capability,
        protocol="balanced_256",
    )
    for index in range(100):
        score = index / 99
        record = TrainingRecordRef.virtual_identity(
            {"prompt": f"question {index}"}, source_index=index
        )
        sink.capture(
            record=record,
            candidate_ordinal=0,
            prompt=f"question {index}",
            output=f"answer {index}",
            training_observation={"reward": score, "passed": score >= 0.5},
            source_index=index,
        )
    captured = sink.seal(checkpoint_hash="checkpoint-hash")
    shard = service.register_training_signal_shard(
        captured,
        reward_system_revision_id=system_revision.id,
        protocol_revision_id=service.default_ids["protocol:balanced_256"],
        producer_model_hash="model-hash",
        direct_run_segment_id=segment["id"],
    )
    assert service.verify_signal_shard(shard.id)["valid"] is True
    audit = service.create_audit(signal_shard_id=shard.id)
    assert audit.work_item_id is not None
    assert service.list_audit_samples(audit.id, limit=10).total == 100

    def same_output_sentinel(sample):
        # The sentinel sees the stored output; it never regenerates it.
        index = int(str(sample.output).split()[-1])
        score = index / 99
        return {"reward": score, "passed": score >= 0.5, "raw_output": sample.output}

    completed = service.execute_audit(
        audit.id, sentinel=same_output_sentinel, bootstrap_resamples=100
    )
    assert completed.status == "completed"
    decision = service.store.list_decisions(audit.id).items[-1]
    assert decision.decision == "pass"
    assert decision.action == "continue"
    assert service.verify_audit_bundle(audit.id)["valid"] is True
    agreement = service.list_audit_samples(
        audit.id, outcome="agreement", query="question 9", limit=20
    )
    assert agreement.total == 11
    assert agreement.items[0]["optimizer_observation"] is not None
    assert agreement.items[0]["primary_sentinel_observation"] is not None
    summary = service.list_audits(run_id="run-1").items[0]
    assert summary["boundary_unit"] == "cycle"
    assert summary["boundary_value"] == 1
    assert summary["capture_fidelity"] == "exact"
    reviewed = service.review_audit(
        audit.id, action="continue", reason="Reviewed expected agreement"
    )
    assert reviewed.override is True
    assert reviewed.supersedes_decision_id == decision.id

    # A distinct audit of the same immutable trace can expose reward hacking
    # and projects its pause into the linked run/segment until reviewed.
    failing = service.create_audit(
        signal_shard_id=shard.id, runtime_identity={"audit_variant": "inverted"}
    )

    def inverted_sentinel(sample):
        index = int(str(sample.output).split()[-1])
        score = 1.0 - (index / 99)
        return {"reward": score, "passed": score >= 0.5}

    service.execute_audit(failing.id, sentinel=inverted_sentinel, bootstrap_resamples=50)
    failed_decision = service.store.list_decisions(failing.id).items[-1]
    assert failed_decision.decision == "fail"
    assert failed_decision.action == "pause"
    assert db.get_run("run-1").status == "awaiting_review"
    assert db.get_direct_run_segment(segment["id"])["decision"] == "pause"
    service.review_audit(
        failing.id,
        action="continue",
        reason="Known sentinel inversion; continuing this controlled run",
    )
    assert db.get_run("run-1").status == "completed"
    assert db.get_direct_run_segment(segment["id"])["decision"] == "complete"


def test_exhaustive_trace_registration_and_hydration_stream_jsonl(
    tmp_path: Path, monkeypatch
):
    db = RunDatabase(":memory:")
    optimizer = _verifier_revision(db, "optimizer-stream", "optimizer-stream-fp")
    sentinel = _verifier_revision(db, "sentinel-stream", "sentinel-stream-fp")
    service = RewardIntegrityService(db, root=tmp_path / "managed")
    system = service.create_system(name="Streaming reward")
    revision = service.create_system_revision(
        system.id,
        optimizer_verifier_revision_id=optimizer.id,
        modality="text",
        task_type="binary",
        auditors=[
            {"role": "primary_sentinel", "verifier_revision_id": sentinel.id}
        ],
    )
    db.upsert_run(
        RunRecord(
            run_id="run-stream",
            modality="raft",
            model_name="model",
            status="running",
            output_dir=str(tmp_path / "run-stream"),
            indexed_at="now",
        )
    )
    sink = TrainingSignalSink(
        tmp_path / "signals",
        run_id="run-stream",
        segment_id="segment-stream",
        boundary="final",
        capability=TRAINING_SIGNAL_CAPABILITIES.get("raft:hf"),
        protocol="exhaustive",
    )
    for index in range(40):
        sink.capture(
            record=TrainingRecordRef.virtual_identity(
                {"prompt": f"question {index}"}, source_index=index
            ),
            candidate_ordinal=0,
            prompt=f"question {index}",
            output=f"answer {index}",
            training_observation={"reward": 1.0, "passed": True},
            source_index=index,
        )
    captured = sink.seal(checkpoint_hash="checkpoint-stream")
    manifest = json.loads(Path(captured.path, "manifest.json").read_text())

    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path.name == "samples.jsonl":
            raise AssertionError("registration and hydration must stream samples")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    shard = service.register_training_signal_shard(
        captured,
        reward_system_revision_id=revision.id,
        protocol_revision_id=service.default_ids["protocol:exhaustive"],
        producer_model_hash="model-stream",
    )
    assert shard.retained_set_hash == manifest["retained_ids_hash"]
    assert shard.distinct_record_count == 40
    audit = service.create_audit(signal_shard_id=shard.id, submit=False)
    assert audit.total_samples == 40
    assert audit.distinct_record_count == 40
