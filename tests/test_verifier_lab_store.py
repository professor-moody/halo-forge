from __future__ import annotations

import json
import sqlite3

import pytest

from halo_forge.run_db import RunDatabase
from halo_forge.verifier_lab import VerifierLabStore, VerifierObservation
from halo_forge.verifier_lab.store import scrub_secrets


@pytest.fixture
def db():
    value = RunDatabase(":memory:")
    yield value
    value.close()


@pytest.fixture
def store(db):
    return VerifierLabStore(db)


def _profile_revision(store: VerifierLabStore, *, name: str = "Exact match"):
    profile = store.create_profile(name=name)
    revision = store.create_profile_revision(
        profile.id,
        {
            "family": "deterministic",
            "implementation": {
                "kind": "builtin",
                "ref": "halo_forge.rlvr.verifiers.ExactMatchVerifier",
                "fingerprint": "sha256:implementation",
                "pinned": True,
            },
            "reliability_adapter": {"id": "programmatic", "version": "1"},
            "modality": "text",
            "task_type": "binary",
            "input_mapping": {"candidate": "output", "reference": "expected"},
            "reward_contract": {
                "minimum": 0.0,
                "maximum": 1.0,
                "threshold": 0.5,
                "direction": "maximize",
            },
            "runtime_requirements": {"python": "3.12", "package": "halo-forge"},
        },
    )
    return profile, revision


def _calibration_foundation(store: VerifierLabStore):
    profile, revision = _profile_revision(store)
    protocol = store.create_protocol(name="Replicated calibration")
    protocol_revision = store.create_protocol_revision(
        protocol.id,
        {
            "deterministic_repeats": 2,
            "fresh_process_per_repeat": True,
            "stochastic_seeds": [17, 42, 101],
            "bootstrap": {"resamples": 10_000, "seed": 42, "grouped": True},
        },
    )
    qualification = store.create_qualification_profile(name="Strict oracle")
    qualification_revision = store.create_qualification_profile_revision(
        qualification.id,
        template_kind="strict_oracle",
        requirements={
            "agreement": {"pass": 0.98, "warn": 0.95},
            "false_accept": {"pass_max": 0.01, "warn_max": 0.03},
            "minimum_records": 100,
        },
    )
    return profile, revision, protocol_revision, qualification_revision


def _calibration(store: VerifierLabStore):
    _, revision, protocol_revision, qualification_revision = _calibration_foundation(store)
    calibration = store.create_calibration(
        verifier_revision_id=revision.id,
        protocol_revision_id=protocol_revision.id,
        qualification_profile_revision_id=qualification_revision.id,
        source_kind="label_set",
        source_revision_id="labels-r1",
        source_hash="source-sha",
        source_purpose="development",
        request={"confirmation": True},
        partition={"seed": 42, "calibration_fraction": 0.7},
        runtime_identity={"python": "3.12", "package": "halo-forge"},
        total_records=100,
    )
    return revision, calibration


def test_large_calibration_comparison_is_index_joined_and_page_bounded(store, db):
    from halo_forge.verifier_lab.service import VerifierLabService

    _, revision, protocol, qualification = _calibration_foundation(store)
    common = {
        "verifier_revision_id": revision.id,
        "protocol_revision_id": protocol.id,
        "qualification_profile_revision_id": qualification.id,
        "source_kind": "label_set",
        "source_revision_id": "large-labels-r1",
        "source_hash": "large-source-sha",
        "source_purpose": "development",
    }
    base = store.create_calibration(**common, request={"variant": "base"})
    candidate = store.create_calibration(**common, request={"variant": "candidate"})
    insert = """
        INSERT INTO verifier_calibration_samples (
            calibration_id, ordinal, record_id, record_hash, group_id,
            partition, repeat_index, orientation, probe_kind, seed,
            reference_json, observation_json, reward, passed, latency_ms,
            error, runtime_identity_json, metadata_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def rows(calibration_id: str, count: int):
        for index in range(count):
            expected = index % 2 == 0
            yield (
                calibration_id,
                index,
                f"record-{index:05d}",
                f"hash-{index:05d}",
                f"group-{index:05d}",
                "calibration",
                0,
                "canonical",
                "canonical",
                42,
                json.dumps({"expected": expected}),
                json.dumps(
                    {
                        "reward": float(expected),
                        "passed": expected,
                        "parsed_value": expected,
                        "details": {},
                        "component_trace": [],
                        "runtime_identity": {},
                    }
                ),
                float(expected),
                int(expected),
                1.0,
                None,
                "{}",
                "{}",
                "2026-07-15T00:00:00+00:00",
            )

    with db._lock:
        db._conn.executemany(insert, rows(base.id, 5_000))
        # The final two base rows deliberately have no candidate evidence.
        db._conn.executemany(insert, rows(candidate.id, 4_998))
        db._conn.commit()

    pairs, total, limit, offset = store.compare_sample_page(
        base.id, candidate.id, limit=10_000, offset=4_995
    )
    assert total == 5_000
    assert limit == 1_000
    assert offset == 4_995
    assert len(pairs) == 5
    assert pairs[-1][1] is None

    service = VerifierLabService(db)
    # The service comparison must never fall back to the old full-scan helper.
    service._all_samples = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("full sample materialization is forbidden")
    )
    comparison = service.compare_calibrations(base.id, candidate.id, limit=3, offset=4_997)
    assert comparison["samples"]["total"] == 5_000
    assert len(comparison["samples"]["items"]) == 3
    assert [item["classification"] for item in comparison["samples"]["items"]] == [
        "unchanged",
        "missing_evidence",
        "missing_evidence",
    ]


def test_schema_v11_is_additive_and_preserves_benchmark_compatibility(tmp_path):
    path = tmp_path / "v9.db"
    legacy = RunDatabase(str(path))
    with legacy._lock:
        legacy._conn.execute("""
            INSERT INTO benchmark_suites
                (id, name, purpose, purpose_v4, created_at, updated_at)
            VALUES ('suite', 'Operational checks', 'unspecified', 'operational', 't', 't')
            """)
        legacy._conn.execute("UPDATE schema_meta SET value = '9' WHERE key = 'schema_version'")
        legacy._conn.commit()
    legacy.close()

    migrated = RunDatabase(str(path))
    assert (
        migrated._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()[0]
            == "23"
    )
    suite = migrated._conn.execute(
        "SELECT purpose, purpose_v4 FROM benchmark_suites WHERE id = 'suite'"
    ).fetchone()
    assert dict(suite) == {"purpose": "unspecified", "purpose_v4": "operational"}
    tables = {
        row[0]
        for row in migrated._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {
        "verifier_profiles",
        "verifier_profile_revisions",
        "verifier_revision_components",
        "verifier_calibrations",
        "verifier_calibration_samples",
        "verifier_calibration_metrics",
        "verifier_qualification_decisions",
        "verifier_alias_events",
        "verifier_bindings",
    } <= tables
    migrated.close()


def test_profile_revision_is_immutable_reused_and_recursively_scrubbed(store, db):
    profile = store.create_profile(name="Hosted judge")
    definition = {
        "family": "llm_judge",
        "implementation": {
            "kind": "entry_point",
            "ref": "example.judges:RubricJudge",
            "fingerprint": "dist-1:source-sha",
        },
        "modality": "text",
        "task_type": "pairwise",
        "reward_contract": {"minimum": -1, "maximum": 1, "threshold": 0},
        "provider": {
            "endpoint": "https://user:password@example.test/v1?api_key=secret&region=us",
            "api_key": "never-store-this",
            "nested": [{"authorization": "Bearer secret", "region": "us"}],
        },
        "runtime_requirements": {"model_revision": "judge@abc123"},
    }
    revision = store.create_profile_revision(profile.id, definition)
    same = store.create_profile_revision(profile.id, definition)
    assert same.id == revision.id
    assert revision.revision_number == 1
    assert revision.qualifiable is True
    serialized = str(revision.to_dict())
    assert "never-store-this" not in serialized
    assert "password" not in serialized
    assert "api_key" not in serialized
    assert "region=us" in serialized
    assert scrub_secrets({"outer": [{"client_secret": "x", "safe": 1}]}) == {"outer": [{"safe": 1}]}

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db._conn.execute(
            "UPDATE verifier_profile_revisions SET task_type = 'scalar' WHERE id = ?",
            (revision.id,),
        )

    unpinned = store.create_profile(name="Unpinned plugin")
    unpinned_revision = store.create_profile_revision(
        unpinned.id,
        {
            "family": "deterministic",
            "implementation": {"kind": "user_plugin", "ref": "/tmp/plugin.py"},
            "modality": "text",
            "task_type": "binary",
            "reward_contract": {"minimum": 0, "maximum": 1},
        },
    )
    assert unpinned_revision.qualifiable is False
    assert "implementation_unfingerprinted" in unpinned_revision.qualification_blockers


def test_ordered_chain_components_and_cycle_detection(store, db):
    _, first = _profile_revision(store, name="First")
    _, second = _profile_revision(store, name="Second")
    chain = store.create_profile(name="Guarded chain")
    chain_revision = store.create_profile_revision(
        chain.id,
        {
            "family": "chain",
            "implementation": {
                "kind": "builtin",
                "ref": "OrderedVerifierChain",
                "fingerprint": "chain-source",
            },
            "modality": "text",
            "task_type": "binary",
            "reward_contract": {"minimum": 0, "maximum": 1},
            "aggregation": {"kind": "weighted_mean"},
        },
        components=[
            {"child_revision_id": second.id, "weight": 0.25},
            {"child_revision_id": first.id, "weight": 0.75, "veto": True},
        ],
    )
    assert [item.child_revision_id for item in chain_revision.components] == [
        second.id,
        first.id,
    ]
    assert chain_revision.components[1].veto is True
    with pytest.raises(ValueError, match="same child"):
        store.create_profile_revision(
            chain.id,
            {
                **chain_revision.definition,
                "aggregation": {"kind": "minimum"},
            },
            components=[
                {"child_revision_id": first.id},
                {"child_revision_id": first.id},
            ],
        )

    # Simulate a corrupt third-party database to prove traversal rejects a
    # recursive chain rather than recursing forever or hiding the cycle.
    db._conn.execute("DROP TRIGGER immutable_verifier_components_update")
    db._conn.execute("PRAGMA ignore_check_constraints = ON")
    db._conn.execute(
        """
        UPDATE verifier_revision_components SET child_revision_id = ?
        WHERE revision_id = ? AND ordinal = 0
        """,
        (chain_revision.id, chain_revision.id),
    )
    with pytest.raises(ValueError, match="cycle"):
        store.create_profile_revision(
            chain.id,
            {
                **chain_revision.definition,
                "aggregation": {"kind": "maximum"},
            },
            components=[{"child_revision_id": chain_revision.id}],
        )


def test_protocol_and_qualification_revisions_are_immutable(store, db):
    protocol = store.create_protocol(name="Default replicated")
    first = store.create_protocol_revision(protocol.id, {"seeds": [17, 42, 101]})
    second = store.create_protocol_revision(protocol.id, {"seeds": [17, 42, 101, 202]})
    assert second.revision_number == 2
    assert store.get_protocol(protocol.id).latest_revision_id == second.id
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db._conn.execute(
            "UPDATE verifier_calibration_protocol_revisions SET definition_json = '{}' WHERE id = ?",
            (first.id,),
        )

    exploratory = store.create_qualification_profile(name="Exploratory")
    with pytest.raises(ValueError, match="never"):
        store.create_qualification_profile_revision(
            exploratory.id,
            template_kind="exploratory",
            promotable=True,
            requirements={},
        )
    revision = store.create_qualification_profile_revision(
        exploratory.id,
        template_kind="exploratory",
        requirements={"report_only": True},
    )
    assert revision.promotable is False


@pytest.mark.parametrize(
    "purpose",
    ["operational", "holdout", "test", "canary", "protected-lineage", "reward-model-training"],
)
def test_protected_calibration_sources_are_refused(store, purpose):
    _, revision, protocol, qualification = _calibration_foundation(store)
    with pytest.raises(ValueError, match="protected"):
        store.create_calibration(
            verifier_revision_id=revision.id,
            protocol_revision_id=protocol.id,
            qualification_profile_revision_id=qualification.id,
            source_kind="benchmark_suite",
            source_revision_id="protected-r1",
            source_hash="protected-hash",
            source_purpose=purpose,
        )


def test_calibration_identity_lifecycle_samples_metrics_and_reuse(store, db):
    revision, calibration = _calibration(store)
    assert calibration.status == "queued"
    with pytest.raises(ValueError, match="Immutable"):
        store.update_calibration(calibration.id, source_hash="changed")
    running = store.update_calibration(
        calibration.id,
        status="running",
        stage="invoking",
        processed_records=1,
        started_at="2026-07-15T00:00:00+00:00",
    )
    assert running.stage == "invoking"

    sample = store.append_sample(
        calibration.id,
        ordinal=0,
        record_id="record-1",
        record_hash="record-hash-1",
        group_id="shared-media-a",
        partition="calibration",
        repeat_index=0,
        orientation="canonical",
        probe_kind="base",
        seed=42,
        reference={"label": True},
        observation=VerifierObservation(
            reward=1.0,
            passed=True,
            parsed_value=True,
            raw_output="PASS",
            latency_ms=3.2,
            runtime_identity={"python": "3.12"},
        ),
    )
    assert sample.observation.reward == 1.0
    assert store.get_calibration(calibration.id).sample_count == 1
    with pytest.raises(ValueError, match="outside"):
        store.append_sample(
            calibration.id,
            ordinal=1,
            record_id="record-2",
            record_hash="record-hash-2",
            group_id=None,
            partition="calibration",
            repeat_index=0,
            orientation="canonical",
            probe_kind="base",
            seed=None,
            reference={},
            observation={"reward": 1.01, "passed": True},
        )
    with pytest.raises(ValueError, match="finite"):
        VerifierObservation(reward=float("nan"), passed=False)

    score = store.append_metric(
        calibration.id,
        name="balanced_accuracy",
        value=0.99,
        ci_low=0.97,
        ci_high=1.0,
        direction="maximize",
        record_count=100,
    )
    unavailable = store.append_metric(
        calibration.id,
        name="device_peak_memory",
        value=None,
        available=False,
        missing_reason="device telemetry unavailable",
    )
    assert score.value == 0.99
    assert unavailable.available is False
    with pytest.raises(ValueError, match="requires a value"):
        store.append_metric(calibration.id, name="bad", value=None)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db._conn.execute(
            """
            UPDATE verifier_calibration_metrics SET value = 0.5
            WHERE calibration_id = ? AND name = 'balanced_accuracy'
            """,
            (calibration.id,),
        )

    completed = store.update_calibration(
        calibration.id,
        status="completed",
        stage="published",
        manifest_hash="manifest-sha",
        completed_at="2026-07-15T00:05:00+00:00",
    )
    assert store.find_reusable_calibration(completed.reuse_key).id == calibration.id
    assert (
        store.runtime_compatibility(revision.id, {"python": "3.12", "package": "halo-forge"})[
            "state"
        ]
        == "compatible"
    )
    assert store.runtime_compatibility(revision.id, {"python": "3.13"})["state"] == "stale_runtime"


def test_decisions_alias_history_and_exact_bindings_are_append_only(store, db):
    revision, calibration = _calibration(store)
    store.update_calibration(calibration.id, status="completed", stage="published")
    decision = store.append_decision(
        calibration.id,
        scope="development",
        decision="pass",
        reasons=["all required gates passed"],
        evidence={"metrics": ["balanced_accuracy"]},
    )
    candidate = store.promote_alias(
        revision.id,
        alias="candidate",
        qualification_decision_id=decision.id,
    )
    assert candidate.revision_id == revision.id
    assert len(store.list_alias_history(revision.profile_id)) == 1
    with pytest.raises(ValueError, match="requires a note"):
        store.promote_alias(revision.id, alias="approved", override=True)
    approved = store.promote_alias(
        revision.id,
        alias="approved",
        override=True,
        note="Operator accepted limited confirmation evidence",
    )
    assert approved.alias == "approved"

    binding = store.bind_revision(
        revision.id,
        domain_kind="evaluation",
        domain_id="evaluation-1",
        role="judge",
        qualification_decision_id=decision.id,
        development_exposed=True,
        context={"provider": {"api_key": "secret", "endpoint": "http://local.test/v1"}},
    )
    same = store.bind_revision(
        revision.id,
        domain_kind="evaluation",
        domain_id="evaluation-1",
        role="judge",
        qualification_decision_id=decision.id,
        development_exposed=True,
        context={"provider": {"api_key": "different-secret", "endpoint": "http://local.test/v1"}},
    )
    assert same.id == binding.id
    assert binding.context == {"provider": {"endpoint": "http://local.test/v1"}}
    assert store.list_bindings(domain_kind="evaluation", domain_id="evaluation-1") == [binding]
    for domain_kind in ("dataset_output", "review_suggestion"):
        downstream = store.bind_revision(
            revision.id,
            domain_kind=domain_kind,
            domain_id=f"{domain_kind}-1",
            role="verifier",
            qualification_decision_id=decision.id,
        )
        assert downstream.domain_kind == domain_kind
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db._conn.execute(
            "UPDATE verifier_bindings SET role = 'changed' WHERE id = ?", (binding.id,)
        )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db._conn.execute(
            "UPDATE verifier_qualification_decisions SET decision = 'fail' WHERE id = ?",
            (decision.id,),
        )
