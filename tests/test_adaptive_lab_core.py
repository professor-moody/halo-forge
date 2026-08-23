from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from halo_forge.adaptive_lab import (
    AdaptiveLabService,
    CheckpointPolicyRevision,
    CheckpointRetentionPolicy,
    build_cohort_snapshot,
    classify_directional_interval,
    percentile_bootstrap_interval,
    publish_evidence_bundle,
)
from halo_forge.run_db import RunDatabase


def _development_revision(db: RunDatabase, *, name: str = "development") -> str:
    suite = db.create_benchmark_suite(name=name, purpose="development")
    revision = db.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash=f"hash-{name}",
        items=[{"id": "item-1", "input": "hello"}],
        primary_metric="accuracy",
        direction="maximize",
    )
    return revision.id


def _policy(service: AdaptiveLabService, suite_revision_id: str, **overrides):
    policy = service.create_policy(name="Guarded training")
    definition = {
        "development_suite_revision_id": suite_revision_id,
        "primary_metric": "accuracy",
        "direction": "maximize",
        "schedule": {
            "mode": "percentages",
            "unit": "step",
            "percentages": [0.25, 0.5, 1.0],
        },
        "rules": [
            {
                "metric": "accuracy",
                "direction": "maximize",
                "comparison": "best",
                "kind": "plateau",
                "minimum_delta": 0.01,
                "patience": 2,
                "on_breach": "stop",
            }
        ],
        "automatic_actions": True,
    }
    definition.update(overrides)
    return service.create_policy_revision(policy.id, definition)


def test_policy_revisions_are_immutable_reused_and_resolve_deterministically(tmp_path):
    db = RunDatabase(":memory:")
    service = AdaptiveLabService(db, tmp_path / "evidence")
    suite_id = _development_revision(db)
    revision = _policy(service, suite_id)

    assert revision.revision_id
    assert revision.retention.keep_last == 1
    assert revision.retention.keep_best == 1
    assert revision.retention.protect_evaluated is True
    assert service.get_policy_revision(revision.revision_id) == revision
    reused = service.create_policy_revision(revision.policy_id, revision)
    assert reused.revision_id == revision.revision_id
    assert len(db.list_checkpoint_policy_revisions(revision.policy_id)) == 1

    first = service.resolve_checkpoint_plan(
        revision,
        trainer_mode="sft",
        total_budget=101,
        supported_units=["step"],
    )
    second = service.resolve_checkpoint_plan(
        revision.revision_id,
        trainer_mode="sft",
        total_budget=101,
        supported_units=["step"],
    )
    assert first == second
    assert first.boundaries == (26, 51, 101)
    assert first.policy_revision_id == revision.revision_id
    assert first.retention == revision.retention

    group = db.create_run_group(
        name="adaptive repeat",
        kind="repeat",
        trainer_mode="sft",
        resolved_launch_config={},
        checkpoint_policy_revision_id=revision.revision_id,
        resolved_checkpoint_plan=first.to_dict(),
    )
    assert group.checkpoint_policy_revision_id == revision.revision_id
    assert group.resolved_checkpoint_plan["content_hash"] == first.content_hash

    revised_values = revision.to_dict()
    revised_values.pop("content_hash")
    revised_values["retention"] = {
        **revision.retention.to_dict(),
        "keep_every_n_boundaries": 2,
    }
    revised = service.create_policy_revision(revision.policy_id, revised_values)
    assert revised.revision_number == 2
    assert revised.retention.keep_every_n_boundaries == 2
    assert revised.content_hash != revision.content_hash

    with pytest.raises(ValueError, match="preserve at least one"):
        CheckpointRetentionPolicy(
            keep_last=0,
            keep_best=0,
            protect_evaluated=False,
            protect_decision_referenced=False,
            protect_lineage_referenced=False,
        )


def test_gate_priority_idempotency_and_append_only_override(tmp_path):
    db = RunDatabase(":memory:")
    service = AdaptiveLabService(db, tmp_path / "evidence")
    revision = _policy(service, _development_revision(db))
    plan = service.resolve_checkpoint_plan(
        revision, trainer_mode="sft", total_budget=100, supported_units=["step"]
    )

    baseline = service.evaluate_gate(
        revision,
        plan,
        boundary_index=0,
        current_metrics={"accuracy": 0.60},
    )
    assert baseline.action == "continue"
    assert baseline.evidence["rule_outcomes"][0]["status"] == "baseline_established"

    first = service.evaluate_and_record_gate(
        revision,
        plan,
        boundary_index=0,
        current_metrics={"accuracy": 0.60},
        best_metrics={"accuracy": 0.60},
        plateau_counts={"accuracy": 2},
        idempotency_key="run:segment:gate",
    )
    duplicate = service.evaluate_and_record_gate(
        revision,
        plan,
        boundary_index=0,
        current_metrics={"accuracy": 0.60},
        best_metrics={"accuracy": 0.60},
        plateau_counts={"accuracy": 2},
        idempotency_key="run:segment:gate",
    )
    assert first.id == duplicate.id
    assert first.action == "stop"
    assert first.automatic is True

    override = service.override_gate(first.id, action="continue", reason="inspect one more segment")
    assert override.id != first.id
    assert override.override_of_id == first.id
    assert override.automatic is False
    assert [value.action for value in db.list_checkpoint_gate_decisions()] == [
        "stop",
        "continue",
    ]

    missing = service.evaluate_gate(
        revision,
        plan,
        boundary_index=1,
        current_metrics={},
        missing_evidence=["development-suite"],
    )
    assert missing.action == "pause"
    assert missing.automatic is False
    assert missing.reasons[0] == "missing_required_evidence"


def test_manual_policy_requires_review_even_when_gates_pass(tmp_path):
    db = RunDatabase(":memory:")
    service = AdaptiveLabService(db, tmp_path / "evidence")
    revision = _policy(
        service,
        _development_revision(db),
        automatic_actions=False,
        rules=[],
    )
    plan = service.resolve_checkpoint_plan(
        revision, trainer_mode="sft", total_budget=10, supported_units=["step"]
    )
    decision = service.evaluate_gate(
        revision, plan, boundary_index=0, current_metrics={"accuracy": 0.8}
    )
    assert decision.action == "pause"
    assert decision.evidence["desired_action"] == "continue"
    assert "manual_review_required" in decision.reasons


def test_checkpoint_policy_rejects_holdout_guidance(tmp_path):
    db = RunDatabase(":memory:")
    service = AdaptiveLabService(db, tmp_path / "evidence")
    suite = db.create_benchmark_suite(name="final holdout", purpose="holdout")
    revision = db.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="holdout-hash",
        items=[{"id": "held-out"}],
        primary_metric="accuracy",
        direction="maximize",
    )
    policy = service.create_policy(name="invalid holdout policy")
    with pytest.raises(ValueError, match="development-purpose"):
        service.create_policy_revision(
            policy.id,
            {
                "development_suite_revision_id": revision.id,
                "primary_metric": "accuracy",
                "direction": "maximize",
            },
        )


def test_bootstrap_and_matched_seed_classification_are_deterministic():
    values = [0.1, 0.2, 0.3, 0.4]
    assert percentile_bootstrap_interval(values) == percentile_bootstrap_interval(values)
    assert classify_directional_interval(0.1, 0.2, practical_delta=0.05) == "improved"
    assert (
        classify_directional_interval(-0.02, 0.02, practical_delta=0.05) == "practically_equivalent"
    )

    observations = [
        {"subject_id": "base", "seed": 1, "metric": "loss", "value": 1.0},
        {"subject_id": "base", "seed": 2, "metric": "loss", "value": 1.1},
        {"subject_id": "candidate", "seed": 1, "metric": "loss", "value": 0.7},
        {"subject_id": "candidate", "seed": 2, "metric": "loss", "value": 0.8},
        {"subject_id": "single", "seed": 1, "metric": "loss", "value": 0.6},
    ]
    first = build_cohort_snapshot(
        observations,
        metric="loss",
        direction="minimize",
        baseline_subject_id="base",
    )
    second = build_cohort_snapshot(
        reversed(observations),
        metric="loss",
        direction="minimize",
        baseline_subject_id="base",
    )
    assert first.content_hash == second.content_hash
    assert first.analysis["comparisons"]["candidate"]["classification"] == "improved"
    assert first.analysis["comparisons"]["single"]["classification"] == "insufficient_evidence"

    incompatible = build_cohort_snapshot(
        observations[:-1],
        metric="loss",
        direction="minimize",
        baseline_subject_id="base",
        bootstrap_resamples=100,
        evidence_compatibility=[
            {
                "subject_id": "base",
                "suite_revision_id": "suite-1",
                "generation_settings_hash": "settings-a",
                "template_hash": "template-a",
            },
            {
                "subject_id": "candidate",
                "suite_revision_id": "suite-1",
                "generation_settings_hash": "settings-b",
                "template_hash": "template-a",
            },
        ],
    )
    comparison = incompatible.analysis["comparisons"]["candidate"]
    assert comparison["classification"] == "insufficient_evidence"
    assert comparison["reason"] == "incompatible_evidence"
    assert comparison["compatibility_mismatches"] == ("generation_settings_hash",)


def test_evidence_bundle_is_atomic_verified_and_worker_dispatchable(tmp_path):
    db = RunDatabase(":memory:")
    service = AdaptiveLabService(db, tmp_path / "evidence")
    snapshot = service.analyze_cohort(
        [
            {"subject_id": "base", "seed": 1, "metric": "accuracy", "value": 0.5},
            {"subject_id": "base", "seed": 2, "metric": "accuracy", "value": 0.5},
            {"subject_id": "candidate", "seed": 1, "metric": "accuracy", "value": 0.7},
            {"subject_id": "candidate", "seed": 2, "metric": "accuracy", "value": 0.7},
        ],
        metric="accuracy",
        direction="maximize",
        baseline_subject_id="base",
        bootstrap_resamples=100,
    )
    decision = service.create_research_decision(
        analysis_snapshot_id=snapshot.id,
        selected_subject={"subject_id": "candidate"},
        rejected_subjects=[{"subject_id": "base"}],
        rationale="Matched seeds improve beyond the practical boundary.",
    )
    bundle = service.queue_evidence_bundle(
        analysis_snapshot_id=snapshot.id,
        research_decision_id=decision.id,
        request={"report_title": "Candidate decision"},
    )
    result = service.execute_work_item(
        {
            "kind": "adaptive_evidence_bundle",
            "domain_id": bundle.id,
            "launch_spec": {
                "action": "build_evidence_bundle",
                "evidence_bundle_id": bundle.id,
            },
        }
    )
    target = Path(result["storage_path"])
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert manifest["content_hash"] == bundle.content_hash
    assert set(manifest["files"]) >= {
        "report.md",
        "report.html",
        "evidence.json",
        "observations.csv",
        "comparisons.csv",
        "comparison.svg",
    }
    assert manifest["formats"] == ["csv", "html", "json", "markdown", "svg"]
    assert service.build_evidence_bundle(bundle.id).id == bundle.id
    assert not list(target.parent.glob(f".{target.name}.staging-*"))
    reused = service.queue_evidence_bundle(
        analysis_snapshot_id=snapshot.id,
        research_decision_id=decision.id,
        request={
            "report_title": "Candidate decision",
            "formats": ["svg", "csv", "json", "html", "markdown"],
        },
    )
    assert reused.id == bundle.id

    with pytest.raises(ValueError, match="unsupported"):
        service.queue_evidence_bundle(
            analysis_snapshot_id=snapshot.id,
            request={"formats": ["pdf"]},
        )

    selected = service.queue_evidence_bundle(
        analysis_snapshot_id=snapshot.id,
        request={"formats": ["json", "svg"]},
    )
    selected_result = service.execute_evidence_bundle(selected.id)
    selected_root = Path(selected_result.storage_path)
    assert {value.name for value in selected_root.iterdir()} == {
        "evidence.json",
        "comparison.svg",
        "manifest.json",
    }


def test_atomic_publisher_rejects_path_escape_and_leaves_no_partial_output(tmp_path):
    destination = tmp_path / "bundle"
    with pytest.raises(ValueError, match="safe relative"):
        publish_evidence_bundle(
            destination,
            content_hash="abc",
            manifest={},
            files={"../escape.txt": "no"},
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".bundle.staging-*"))

    publish_evidence_bundle(
        destination,
        content_hash="verified",
        manifest={"schema_version": 1},
        files={"report.md": "original"},
    )
    (destination / "report.md").write_text("mutated", encoding="utf-8")
    with pytest.raises(RuntimeError, match="verification failed"):
        publish_evidence_bundle(
            destination,
            content_hash="verified",
            manifest={"schema_version": 1},
            files={"report.md": "original"},
        )


def test_corrupt_completed_evidence_is_never_reused_or_rewritten(tmp_path):
    db = RunDatabase(":memory:")
    service = AdaptiveLabService(db, tmp_path / "evidence")
    snapshot = service.analyze_cohort(
        [
            {"subject_id": "base", "seed": 1, "metric": "accuracy", "value": 0.5},
            {"subject_id": "base", "seed": 2, "metric": "accuracy", "value": 0.5},
            {"subject_id": "candidate", "seed": 1, "metric": "accuracy", "value": 0.7},
            {"subject_id": "candidate", "seed": 2, "metric": "accuracy", "value": 0.7},
        ],
        metric="accuracy",
        direction="maximize",
        baseline_subject_id="base",
        bootstrap_resamples=100,
    )
    original = service.queue_evidence_bundle(
        analysis_snapshot_id=snapshot.id,
        request={"report_title": "Tamper check"},
    )
    completed = service.execute_evidence_bundle(original.id)
    Path(completed.storage_path, "report.md").write_text("mutated", encoding="utf-8")

    replacement = service.queue_evidence_bundle(
        analysis_snapshot_id=snapshot.id,
        request={"report_title": "Tamper check"},
    )
    corrupt = db.get_evidence_bundle(original.id)
    assert corrupt is not None and corrupt.status == "corrupt"
    with pytest.raises(ValueError, match="immutable"):
        db.update_evidence_bundle(corrupt.id, content_hash="rewritten-after-corruption")
    with pytest.raises(ValueError, match="immutable"):
        db.update_evidence_bundle(corrupt.id, status="queued")
    assert replacement.id != original.id
    assert replacement.status == "queued"
    rebuilt = service.execute_evidence_bundle(replacement.id)
    assert rebuilt.status == "completed"

    with pytest.raises(ValueError, match="immutable"):
        db.update_evidence_bundle(rebuilt.id, storage_path=str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="requires a reason"):
        db.update_evidence_bundle(rebuilt.id, status="corrupt")


def test_workspace_drafts_upsert_and_expire(tmp_path):
    db = RunDatabase(":memory:")
    service = AdaptiveLabService(db, tmp_path / "evidence")
    first = service.save_workspace_draft(draft_kind="train", content={"step": 1})
    second = service.save_workspace_draft(draft_kind="train", content={"step": 2})
    assert first.id == second.id
    assert second.content == {"step": 2}
    assert len(db.list_workspace_drafts()) == 1
    db._conn.execute(
        "UPDATE workspace_drafts SET expires_at = ? WHERE id = ?",
        ("2000-01-01T00:00:00+00:00", first.id),
    )
    db._conn.commit()
    assert db.get_workspace_draft(first.id) is None
    assert db.get_workspace_draft(first.id, include_expired=True) is not None
    assert db.purge_expired_workspace_drafts(now="9999-12-31T00:00:00+00:00") == 1
    assert db.get_workspace_draft(first.id) is None


def test_v8_migration_preserves_segment_foreign_keys_and_adds_policy_fields(tmp_path):
    path = tmp_path / "legacy-v7.db"
    db = RunDatabase(str(path))
    group = db.create_run_group(
        name="legacy group",
        kind="repeat",
        trainer_mode="sft",
        resolved_launch_config={},
    )
    trial = db.create_run_group_trial(
        run_group_id=group.id,
        ordinal=0,
        config_hash="legacy-config",
        sampled_config={},
    )
    trial_run = db.create_trial_run(
        trial_id=trial.id,
        run_id="legacy-run",
        ordinal=0,
        seed=42,
    )
    segment = db.create_trial_segment(
        trial_run_id=trial_run.id,
        ordinal=0,
        unit="step",
        start_value=0,
        end_value=10,
    )
    db.update_trial_segment(segment.id, decision="continue", decision_reason="legacy")
    db._conn.close()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("ALTER TABLE run_groups DROP COLUMN resolved_checkpoint_plan_json")
    connection.execute("ALTER TABLE run_groups DROP COLUMN checkpoint_policy_revision_id")
    connection.execute("PRAGMA legacy_alter_table = ON")
    connection.execute("ALTER TABLE trial_segments RENAME TO trial_segments_new")
    connection.execute("""
        CREATE TABLE trial_segments (
            id TEXT PRIMARY KEY,
            trial_run_id TEXT NOT NULL REFERENCES trial_runs(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            unit TEXT NOT NULL,
            start_value INTEGER NOT NULL,
            end_value INTEGER NOT NULL,
            work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
            checkpoint_artifact_id TEXT,
            decision TEXT CHECK (
                decision IS NULL OR decision IN ('continue', 'prune', 'complete')
            ),
            decision_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            UNIQUE (trial_run_id, ordinal),
            CHECK (end_value > start_value)
        )
        """)
    connection.execute("INSERT INTO trial_segments SELECT * FROM trial_segments_new")
    connection.execute("DROP TABLE trial_segments_new")
    connection.execute("PRAGMA legacy_alter_table = OFF")
    connection.execute("UPDATE schema_meta SET value = '7' WHERE key = 'schema_version'")
    connection.commit()
    connection.close()

    migrated = RunDatabase(str(path))
    columns = {row["name"] for row in migrated._conn.execute("PRAGMA table_info(run_groups)")}
    assert {"checkpoint_policy_revision_id", "resolved_checkpoint_plan_json"} <= columns
    definition = migrated._conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'trial_segments'"
    ).fetchone()["sql"]
    assert "'pause'" in definition and "'stop'" in definition
    preserved = migrated.get_trial_segment(segment.id)
    assert preserved is not None and preserved.decision == "continue"
    migrated.update_trial_segment(
        segment.id,
        status="awaiting_review",
        decision="pause",
        decision_reason="operator review",
    )
    assert migrated.get_trial_segment(segment.id).decision == "pause"
    for table in ("model_artifacts", "artifact_occurrences"):
        segment_fks = [
            row["table"]
            for row in migrated._conn.execute(f"PRAGMA foreign_key_list({table})")
            if row["from"] == "trial_segment_id"
        ]
        assert segment_fks == ["trial_segments"]
    assert migrated._conn.execute("PRAGMA foreign_key_check").fetchall() == []
