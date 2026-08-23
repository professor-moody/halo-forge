"""Service and HTTP contracts for Verifier Reliability and Reward Studio."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import MethodType

import pytest

from halo_forge.run_db import RunDatabase
from halo_forge.verifier_lab import VerifierLabService, VerifierObservation
from halo_forge.workstation_jobs import WorkstationScheduler

V10_TABLES = (
    "verifier_bindings",
    "verifier_alias_events",
    "verifier_aliases",
    "verifier_qualification_decisions",
    "verifier_calibration_metrics",
    "verifier_calibration_samples",
    "verifier_calibrations",
    "verifier_qualification_profile_revisions",
    "verifier_qualification_profiles",
    "verifier_calibration_protocol_revisions",
    "verifier_calibration_protocols",
    "verifier_revision_components",
    "verifier_profile_revisions",
    "verifier_profiles",
)


def _profile_definition() -> dict:
    return {
        "family": "deterministic",
        "implementation": {"ref": "json_structure"},
        "modality": "text",
        "task_type": "binary",
        "input_mapping": {"candidate": "output", "reference": "expected"},
        "reward_contract": {
            "minimum": 0.0,
            "maximum": 1.0,
            "threshold": 0.5,
            "tie_policy": "fail",
            "error_behavior": "fail_closed",
        },
        "runtime_contract": {},
    }


def _calibration_source(db: RunDatabase, *, rows: int = 100):
    suite = db.create_benchmark_suite(
        name="Reviewed deterministic reference", purpose="development"
    )
    items = [
        {
            "id": f"reference-{index:04d}",
            "record_id": f"reference-{index:04d}",
            "input": f"Prompt {index}",
            "output": "accepted" if index % 2 else "rejected",
            "expected": bool(index % 2),
            "group_id": f"group-{index:04d}",
        }
        for index in range(rows)
    ]
    return db.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="reviewed-reference-v1",
        items=items,
        primary_metric="balanced_accuracy",
        direction="maximize",
    )


def _perfect_reference_invocation(self, revision, item, *, runtime=None):
    expected = bool(item["expected"])
    return VerifierObservation(
        reward=1.0 if expected else 0.0,
        passed=expected,
        details={"fixture": "reviewed-reference"},
        runtime_identity=dict(runtime or {}),
    )


def _completed_calibration(tmp_path: Path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(db)
    service = VerifierLabService(
        db,
        root=tmp_path / "calibrations",
        scheduler=scheduler,
    )
    profile = service.create_profile(
        name="Deterministic reference verifier",
        description="A pinned verifier used by the service contract test.",
        definition=_profile_definition(),
    )
    protocol = service.create_protocol(
        name="Replicated deterministic protocol",
        description=None,
        # The production default is 10,000 grouped resamples. A smaller value
        # keeps this end-to-end contract test fast without changing semantics.
        definition={"bootstrap_resamples": 64},
    )
    qualification = service.create_qualification_profile(
        name="Strict oracle policy",
        description=None,
        template_kind="strict_oracle",
    )
    source = _calibration_source(db)
    service._invoke_revision = MethodType(_perfect_reference_invocation, service)
    launched = service.launch_calibration(
        verifier_revision_id=profile["revision"]["id"],
        source_kind="benchmark_suite",
        source_revision_id=source.id,
        protocol_revision_id=protocol["revision"]["id"],
        qualification_profile_revision_id=qualification["revision"]["id"],
    )
    completed = service.run_calibration(launched["id"])
    return db, service, profile, source, completed


def test_v9_to_v11_migration_is_additive_and_preserves_existing_rows(tmp_path):
    path = tmp_path / "v9.db"
    legacy = RunDatabase(str(path))
    suite = legacy.create_benchmark_suite(name="Preserved v9 suite", purpose="operational")
    with legacy._lock:
        legacy._conn.execute("PRAGMA foreign_keys = OFF")
        for table in V10_TABLES:
            legacy._conn.execute(f"DROP TABLE IF EXISTS {table}")
        legacy._conn.execute("UPDATE schema_meta SET value='9' WHERE key='schema_version'")
        legacy._conn.commit()
    legacy.close()

    migrated = RunDatabase(str(path))
    assert migrated.get_benchmark_suite(suite.id).name == "Preserved v9 suite"
    assert migrated.get_benchmark_suite(suite.id).purpose == "operational"
    assert (
        migrated._conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0]
            == "23"
    )
    tables = {
        str(row[0])
        for row in migrated._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert set(V10_TABLES) <= tables
    migrated.close()


def test_deterministic_calibration_qualifies_and_publishes_verified_bundle(tmp_path):
    db, service, profile, _source, calibration = _completed_calibration(tmp_path)
    try:
        assert calibration.status == "completed"
        assert calibration.total_records == 100
        assert calibration.sample_count == 200
        assert service.verify_calibration(calibration.id)["valid"] is True

        metrics = {value.name: value for value in service.store.list_metrics(calibration.id)}
        assert metrics["balanced_accuracy"].value == 1.0
        assert metrics["balanced_accuracy"].ci_low == 1.0
        assert metrics["balanced_accuracy"].ci_high == 1.0
        assert metrics["balanced_accuracy"].metadata["bootstrap"]["method"] == (
            "compressed_multinomial_percentile"
        )
        assert metrics["balanced_accuracy"].metadata["bootstrap"]["exact"] is True
        assert (
            metrics["balanced_accuracy"].metadata["primary_metric_interval"]["replicate_unit"]
            == "stable_record"
        )
        assert metrics["exact_repeat_agreement"].value == 1.0
        assert metrics["record_count"].value == 100.0

        decision = service.qualify_calibration(calibration.id, scope="development")
        assert decision.decision == "pass"
        assert decision.override is False
        assert decision.evidence["record_count"] == 100
        decision_artifact = decision.evidence["decision_artifact"]
        decision_path = Path(decision_artifact["path"])
        assert decision_path.is_file()
        assert hashlib.sha256(decision_path.read_bytes()).hexdigest() == decision_artifact["sha256"]
        decision_bytes = decision_path.read_bytes()
        decision_path.write_text("tampered\n", encoding="utf-8")
        assert service.verify_calibration(calibration.id)["valid"] is False
        decision_path.write_bytes(decision_bytes)
        assert service.verify_calibration(calibration.id)["valid"] is True
        decision_page = service.list_qualification_decisions(
            calibration_id=calibration.id,
            limit=10,
        )
        assert decision_page["total"] == 1
        assert decision_page["items"][0]["id"] == decision.id

        resolved = service.resolve_binding(profile["revision"]["id"])
        assert resolved["revision_hash"] == profile["revision"]["content_hash"]
        assert resolved["runtime_compatibility"]["state"] == "compatible"

        bundle = Path(calibration.artifact_path)
        assert {
            "manifest.json",
            "profile.json",
            "source.json",
            "protocol.json",
            "runtime.json",
            "qualification.json",
            "metrics.json",
            "samples.jsonl",
        } <= {value.name for value in bundle.iterdir()}
        (bundle / "samples.jsonl").write_text("tampered\n", encoding="utf-8")
        verification = service.verify_calibration(calibration.id)
        assert verification["valid"] is False
        assert "checksum mismatch: samples.jsonl" in verification["errors"]
    finally:
        db.close()


def test_atomic_bundle_is_adopted_when_catalog_update_was_interrupted(tmp_path):
    db, service, _profile, _source, calibration = _completed_calibration(tmp_path)
    try:
        bundle = Path(calibration.artifact_path)
        manifest_before = (bundle / "manifest.json").read_bytes()
        metric_count_before = len(service.store.list_metrics(calibration.id))
        # Simulate a crash after os.replace(staging, final) and before the
        # catalog transaction recorded the final directory and manifest hash.
        service.store.update_calibration(
            calibration.id,
            status="failed",
            stage="failed",
            artifact_path=None,
            manifest_hash=None,
        )
        assert service.verify_calibration(calibration.id)["valid"] is False

        recovered = service.run_calibration(calibration.id)

        assert recovered.artifact_path == str(bundle)
        assert recovered.manifest_hash
        assert (bundle / "manifest.json").read_bytes() == manifest_before
        assert len(service.store.list_metrics(calibration.id)) == metric_count_before
        assert service.verify_calibration(calibration.id)["valid"] is True
    finally:
        db.close()


def test_scheduler_reconciles_retries_and_cancels_calibration_domain_row(tmp_path):
    db = RunDatabase(str(tmp_path / "recovery.db"))
    owner = WorkstationScheduler(db, worker_id="original-worker")
    service = VerifierLabService(
        db,
        root=tmp_path / "calibrations",
        scheduler=owner,
    )
    try:
        profile = service.create_profile(
            name="Recovery verifier",
            description=None,
            definition=_profile_definition(),
        )
        protocol = service.create_protocol(
            name="Recovery protocol",
            description=None,
            definition={"bootstrap_resamples": 16},
        )
        qualification = service.create_qualification_profile(
            name="Recovery policy",
            description=None,
            template_kind="strict_oracle",
        )
        source = _calibration_source(db, rows=20)
        launched = service.launch_calibration(
            verifier_revision_id=profile["revision"]["id"],
            source_kind="benchmark_suite",
            source_revision_id=source.id,
            protocol_revision_id=protocol["revision"]["id"],
            qualification_profile_revision_id=qualification["revision"]["id"],
        )
        claimed = owner.claim(child_pid=777, child_pid_started_at=123.0)
        assert claimed is not None and claimed.id == launched["work_item_id"]

        recovering = WorkstationScheduler(
            db,
            worker_id="replacement-worker",
            process_probe=lambda _pid, _started: False,
        )
        outcome = recovering.recover_or_adopt()
        assert [value.id for value in outcome.interrupted] == [claimed.id]
        interrupted = service.store.get_calibration(launched["id"])
        assert interrupted.status == "needs_reconciliation"
        assert interrupted.stage == "needs_reconciliation"

        retried = recovering.retry(claimed.id, reason="operator verified dead worker")
        assert retried is not None and retried.status == "queued"
        queued = service.store.get_calibration(launched["id"])
        assert queued.status == "queued"
        assert queued.stage == "resume_pending"
        assert queued.retry_count == 1

        domain = service.cancel_calibration(launched["id"])
        assert domain.status == "cancelled"
        assert domain.cancel_requested is True
        cancelled = db.get_work_item(claimed.id)
        assert cancelled is not None and cancelled.status == "cancelled"
    finally:
        db.close()


def test_calibration_samples_are_server_filtered_and_bounded(tmp_path):
    db, service, _profile, _source, calibration = _completed_calibration(tmp_path)
    try:
        page = service.list_calibration_samples(
            calibration.id,
            partition="calibration",
            outcome="passed",
            query="reference-009",
            limit=5_000,
            offset=0,
        )
        assert page["limit"] == 1_000
        assert page["offset"] == 0
        assert page["total"] == 10
        assert len(page["items"]) == 10
        assert all(value["observation"]["passed"] is True for value in page["items"])
        assert all("reference-009" in value["record_id"] for value in page["items"])

        tail = service.list_calibration_samples(calibration.id, limit=7, offset=196)
        assert tail["total"] == 200
        assert len(tail["items"]) == 4
        assert [value["ordinal"] for value in tail["items"]] == [196, 197, 198, 199]

        assert (
            service.list_calibration_samples(calibration.id, outcome="false_accept")["total"] == 0
        )
        assert (
            service.list_calibration_samples(calibration.id, outcome="false_reject")["total"] == 0
        )
        assert service.list_calibration_samples(calibration.id, outcome="repeat_flip")["total"] == 0
        canonical = service.list_calibration_samples(
            calibration.id, perturbation="canonical", limit=1
        )
        assert canonical["total"] == 200
        assert canonical["items"][0]["probe_kind"] == "canonical"
    finally:
        db.close()


def test_structured_metric_diagnostics_round_trip_without_loss(tmp_path):
    from halo_forge.verifier_lab.metrics import compute_calibration_metrics
    from halo_forge.verifier_lab.observation import RewardContract

    db, service, _profile, _source, completed = _completed_calibration(tmp_path)
    try:
        pending = service.store.create_calibration(
            verifier_revision_id=completed.verifier_revision_id,
            protocol_revision_id=completed.protocol_revision_id,
            qualification_profile_revision_id=completed.qualification_profile_revision_id,
            source_kind=completed.source_kind,
            source_revision_id=completed.source_revision_id,
            source_hash=completed.source_hash,
            source_purpose=completed.source_purpose,
            request={"diagnostic_round_trip": True},
        )
        result = compute_calibration_metrics(
            [
                {"record_id": "a", "expected": "safe", "predicted": "safe"},
                {"record_id": "b", "expected": "unsafe", "predicted": "safe"},
            ],
            task_type="categorical",
            reward_contract=RewardContract(),
            bootstrap_resamples=16,
        )
        service._persist_metric_result(pending.id, result, partition="calibration")
        persisted = service.store.list_metrics(pending.id)
        nested = service._nested_metrics(persisted)

        assert nested["task"]["confusion_matrix"] == result["task"]["confusion_matrix"]
        assert nested["task"]["per_class"] == result["task"]["per_class"]
        assert nested["task"]["per_class"]["unsafe"]["support"] == 1
        diagnostic_rows = [
            metric for metric in persisted if isinstance(metric.metadata.get("structured"), dict)
        ]
        assert len(diagnostic_rows) >= 1
    finally:
        db.close()


def test_public_api_profile_revision_precedes_raw_verifier_configuration(tmp_path):
    from halo_forge.evaluation_lab import EvaluationLabService
    from halo_forge.public_api.service import PublicApiService

    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(db)
    verifier = VerifierLabService(db, root=tmp_path / "calibrations", scheduler=scheduler)
    profile = verifier.create_profile(
        name="Pinned evaluation verifier",
        description=None,
        definition=_profile_definition(),
    )
    evaluations = EvaluationLabService(db, tmp_path / "evaluations", scheduler=scheduler)
    suite = db.create_benchmark_suite(name="Evaluation contract", purpose="development")
    revision = db.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="evaluation-contract-v1",
        items=[{"id": "one", "input": "Prompt", "expected": True}],
        primary_metric="score",
        direction="maximize",
        evaluator_versions={"verifier": evaluations.registry.get("verifier").adapter_version},
    )
    public = PublicApiService(
        database=db,
        verifier_lab=verifier,
        evaluation_lab=evaluations,
        workstation_scheduler=scheduler,
        base_path=tmp_path,
    )
    revision_id = profile["revision"]["id"]
    subject = {"kind": "model", "value": "example/model", "revision": "pinned-r1"}
    try:
        launched = public.launch_evaluation(
            {
                "suite_revision_id": revision.id,
                "adapter_id": "verifier",
                "subject": subject,
                "verifier_profile_revision_id": revision_id,
            }
        )
        persisted = db.get_evaluation(launched["id"])
        assert persisted.request["adapter_request"]["verifier_profile_revision_id"] == revision_id

        with pytest.raises(ValueError, match="conflicts with raw verifier fields"):
            public.launch_evaluation(
                {
                    "suite_revision_id": revision.id,
                    "adapter_id": "verifier",
                    "subject": subject,
                    "verifier_profile_revision_id": revision_id,
                    "request": {"verifier": "json_structure"},
                }
            )

        batch = public.launch_evaluation_batch(
            {
                "suite_revision_id": revision.id,
                "adapter_id": "verifier",
                "base": subject,
                "candidates": [
                    {"kind": "model", "value": "example/candidate", "revision": "pinned-r2"},
                    {"kind": "model", "value": "example/candidate", "revision": "pinned-r3"},
                ],
                "verifier_profile_revision_id": revision_id,
            }
        )
        assert batch["verifier_profile_revision_id"] == revision_id
        assert len(batch["evaluations"]) == 3
        for child in batch["evaluations"]:
            persisted_child = db.get_evaluation(child["id"])
            assert (
                persisted_child.request["adapter_request"]["verifier_profile_revision_id"]
                == revision_id
            )
            assert child["verifier_binding"]["verifier_revision_id"] == revision_id
        bound_ids = {
            item["domain_id"]
            for item in verifier.list_usage(revision_id, limit=100)["items"]
            if item["domain_kind"] == "evaluation"
        }
        assert {child["id"] for child in batch["evaluations"]} <= bound_ids

        revised_definition = _profile_definition()
        revised_definition["reward_contract"]["threshold"] = 0.75
        revised = verifier.revise_profile(profile["profile"]["id"], definition=revised_definition)
        revised_batch = public.launch_evaluation_batch(
            {
                "suite_revision_id": revision.id,
                "adapter_id": "verifier",
                "base": subject,
                "candidates": [
                    {"kind": "model", "value": "example/candidate", "revision": "pinned-r2"}
                ],
                "verifier_profile_revision_id": revised["revision"]["id"],
            }
        )
        assert revised_batch["id"] != batch["id"]

        before_conflict = len(db.list_evaluations(limit=1000))
        with pytest.raises(ValueError, match="conflicts with raw verifier fields"):
            public.launch_evaluation_batch(
                {
                    "suite_revision_id": revision.id,
                    "adapter_id": "verifier",
                    "base": subject,
                    "candidates": [{"kind": "model", "value": "candidate"}],
                    "verifier_profile_revision_id": revision_id,
                    "request": {"reward_threshold": 0.5},
                }
            )
        assert len(db.list_evaluations(limit=1000)) == before_conflict
    finally:
        db.close()


def test_profile_defaults_pin_observed_runtime_toolchain_and_hardware(tmp_path):
    db = RunDatabase(str(tmp_path / "runtime-contract.db"))
    try:
        service = VerifierLabService(db, root=tmp_path / "calibrations")
        definition = _profile_definition()
        definition.pop("runtime_contract")
        profile = service.create_profile(
            name="Runtime-pinned verifier",
            description=None,
            definition=definition,
        )
        revision = service.store.get_profile_revision(profile["revision"]["id"])
        contract = revision.runtime_contract
        assert contract["schema_version"] == 1
        assert contract["runtime"]["python"]
        assert contract["runtime"]["platform"]
        assert contract["toolchain"]["packages"]["halo-forge"]
        assert contract["hardware"]["machine"]
        assert service.runtime_compatibility(revision.id)["state"] == "compatible"

        stale = service.runtime_compatibility(
            revision.id,
            {
                "runtime": {"python": "0.0-drifted"},
                "toolchain": {
                    "packages": {"halo-forge": "0.0-drifted"},
                },
                "hardware": {"machine": "drifted-machine"},
            },
        )
        assert stale["state"] == "stale_runtime"
        assert {item["field"] for item in stale["mismatches"]} >= {
            "runtime.python",
            "toolchain.packages.halo-forge",
            "hardware.machine",
        }
    finally:
        db.close()


def test_profile_input_mapping_is_applied_during_calibration_invocation(tmp_path):
    db = RunDatabase(str(tmp_path / "mapping.db"))
    try:
        service = VerifierLabService(db, root=tmp_path / "calibrations")
        definition = _profile_definition()
        definition["input_mapping"] = {"candidate": "payload.answer"}
        profile = service.create_profile(
            name="Mapped JSON verifier",
            description=None,
            definition=definition,
        )
        observation = service.invoke_revision(
            profile["revision"]["id"],
            {"payload": {"answer": '{"valid": true}'}},
        )
        assert observation.passed is True
        assert observation.reward == 1.0
    finally:
        db.close()


def test_deterministic_replica_uses_a_fresh_interpreter_process(tmp_path):
    db = RunDatabase(str(tmp_path / "process-isolation.db"))
    try:
        service = VerifierLabService(db, root=tmp_path / "calibrations")
        profile = service.create_profile(
            name="Isolated JSON verifier",
            description=None,
            definition=_profile_definition(),
        )
        revision = service.store.get_profile_revision(profile["revision"]["id"])
        observation = service._invoke_revision(
            revision,
            {"output": '{"valid": true}', "expected": True},
            runtime={"fresh_process_requested": True},
        )
        assert observation.error is None
        assert observation.passed is True
        assert observation.runtime_identity["process_isolation"] == "fresh_interpreter"
        assert observation.runtime_identity["process_id"] != os.getpid()
    finally:
        db.close()


def test_override_promotion_stays_out_of_guided_profile_results(tmp_path):
    db = RunDatabase(str(tmp_path / "override.db"))
    try:
        service = VerifierLabService(db, root=tmp_path / "calibrations")
        profile = service.create_profile(
            name="Overridden verifier",
            description=None,
            definition=_profile_definition(),
        )
        revision_id = profile["revision"]["id"]
        service.store.promote_alias(
            revision_id,
            alias="candidate",
            override=True,
            note="Operator accepts incomplete evidence for an advanced-only trial.",
        )
        assert service.list_profiles(qualified_only=True)["total"] == 0
        advanced = service.list_profiles(
            qualified_only=True,
            include_overridden=True,
        )
        assert advanced["total"] == 1
        assert advanced["items"][0]["guided_eligible"] is False
        assert advanced["items"][0]["overridden_aliases"] == ["candidate"]
        with pytest.raises(ValueError, match="Guided use requires"):
            service.resolve_binding(revision_id, require_qualified=True)
    finally:
        db.close()


def test_calibration_source_revision_aliases_are_canonicalized(tmp_path):
    db = RunDatabase(str(tmp_path / "source-alias.db"))
    try:
        service = VerifierLabService(db, root=tmp_path / "calibrations")
        profile = service.create_profile(
            name="Alias source verifier",
            description=None,
            definition=_profile_definition(),
        )
        protocol = service.create_protocol(
            name="Alias source protocol",
            description=None,
            definition={"bootstrap_resamples": 16},
        )
        qualification = service.create_qualification_profile(
            name="Alias source policy",
            description=None,
            template_kind="strict_oracle",
        )
        source = _calibration_source(db, rows=20)
        launched = service.launch_calibration(
            verifier_revision_id=profile["revision"]["id"],
            source_kind="benchmark_suite_revision",
            source_revision_id=source.id,
            protocol_revision_id=protocol["revision"]["id"],
            qualification_profile_revision_id=qualification["revision"]["id"],
        )
        assert launched["source_kind"] == "benchmark_suite"
    finally:
        db.close()


def test_verifier_http_resources_keep_bounded_page_shape(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from halo_forge.public_api import app as app_module
    from halo_forge.public_api.service import PublicApiService

    db, verifier, profile, _source, calibration = _completed_calibration(tmp_path)
    public = PublicApiService(
        database=db,
        verifier_lab=verifier,
        workstation_scheduler=verifier.scheduler,
        base_path=tmp_path,
    )
    monkeypatch.setattr(app_module, "PublicApiService", lambda: public)
    monkeypatch.setenv("HALOFORGE_DISABLE_AUTO_WORKER", "1")
    try:
        with TestClient(app_module.create_app(serve_frontend=False)) as client:
            response = client.get(
                f"/api/public/verifier-calibrations/{calibration.id}/samples",
                params={"limit": 13, "offset": 187, "outcome": "failed"},
            )
            assert response.status_code == 200, response.text
            page = response.json()
            assert page == {
                **page,
                "total": 100,
                "limit": 13,
                "offset": 187,
            }
            assert page["items"] == []

            oversized = client.get(
                f"/api/public/verifier-calibrations/{calibration.id}/samples",
                params={"limit": 1001},
            )
            assert oversized.status_code == 422

            profiles = client.get(
                "/api/public/verifier-profiles",
                params={"q": "reference", "limit": 1},
            )
            assert profiles.status_code == 200, profiles.text
            assert profiles.json()["total"] == 1
            assert profiles.json()["limit"] == 1

            revisions = client.get(
                f"/api/public/verifier-profiles/{profile['profile']['id']}/revisions",
                params={"limit": 1},
            )
            assert revisions.status_code == 200, revisions.text
            assert revisions.json()["total"] == 1

            metrics = client.get(
                f"/api/public/verifier-calibrations/{calibration.id}/metrics",
                params={"limit": 3, "partition": "calibration"},
            )
            assert metrics.status_code == 200, metrics.text
            assert metrics.json()["limit"] == 3
            assert len(metrics.json()["items"]) == 3

            suite = db.create_benchmark_suite(name="Bound verifier batch", purpose="development")
            suite_revision = db.create_benchmark_suite_revision(
                suite_id=suite.id,
                content_hash="bound-verifier-batch-v1",
                items=[{"id": "one", "input": "Prompt", "expected": True}],
                primary_metric="score",
                direction="maximize",
            )
            revision_id = profile["revision"]["id"]
            batch_response = client.post(
                "/api/public/evaluation-batches",
                json={
                    "suite_revision_id": suite_revision.id,
                    "adapter_id": "verifier",
                    "base": {"kind": "model", "value": "base@r1"},
                    "candidates": [{"kind": "model", "value": "candidate@r1"}],
                    "verifier_profile_revision_id": revision_id,
                },
            )
            assert batch_response.status_code == 202, batch_response.text
            batch_payload = batch_response.json()
            assert batch_payload["verifier_profile_revision_id"] == revision_id
            assert all(
                child["verifier_binding"]["verifier_revision_id"] == revision_id
                for child in batch_payload["evaluations"]
            )

            conflict = client.post(
                "/api/public/evaluation-batches",
                json={
                    "suite_revision_id": suite_revision.id,
                    "adapter_id": "verifier",
                    "base": {"kind": "model", "value": "base@r1"},
                    "candidates": [{"kind": "model", "value": "candidate@r1"}],
                    "verifier_profile_revision_id": revision_id,
                    "verifier": "json_structure",
                },
            )
            assert conflict.status_code == 400
            assert "conflicts with raw verifier fields" in conflict.json()["detail"]
    finally:
        db.close()
