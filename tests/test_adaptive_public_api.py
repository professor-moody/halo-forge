"""Public contracts for adaptive checkpoints and reviewed research evidence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from halo_forge.public_api.service import PublicApiService
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


def _service(tmp_path: Path) -> tuple[PublicApiService, RunDatabase, WorkstationScheduler]:
    database = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(database, capacity_probe=_available_capacity)
    service = PublicApiService(
        database=database,
        workstation_scheduler=scheduler,
        base_path=tmp_path,
        evidence_storage_root=tmp_path / "evidence",
        evaluation_storage_root=tmp_path / "evaluations",
    )
    return service, database, scheduler


def _suite(service: PublicApiService, *, purpose: str = "development") -> str:
    return service.create_benchmark_suite(
        {
            "name": f"{purpose}-quality",
            "purpose": purpose,
            "items": [{"id": "example", "input": "x", "expected": "y"}],
            "primary_metric": "accuracy",
            "direction": "maximize",
        }
    )["latest_revision"]["id"]


def _policy_payload(revision_id: str, *, policy_id: str = "periodic") -> dict:
    return {
        "policy_id": policy_id,
        "revision_number": 1,
        "name": "Periodic quality checks",
        "development_suite_revision_id": revision_id,
        "primary_metric": "accuracy",
        "direction": "maximize",
        "schedule": {
            "mode": "percentages",
            "unit": "step",
            "percentages": [0.25, 0.5, 1.0],
        },
        "rules": [],
        "guardrail_suite_revision_ids": [],
        "automatic_actions": False,
        "compatible_capabilities": [],
        "version": 1,
    }


def test_checkpoint_policy_routes_resolve_a_trainer_compatible_immutable_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from halo_forge.auth.dependency import reset_store_for_tests
    from halo_forge.public_api import app as app_module

    service, database, _scheduler = _service(tmp_path)
    development = _suite(service)
    holdout = _suite(service, purpose="holdout")
    monkeypatch.setattr(app_module, "PublicApiService", lambda: service)
    reset_store_for_tests(None)
    try:
        with TestClient(app_module.create_app(serve_frontend=False)) as client:
            created_response = client.post(
                "/api/public/checkpoint-policies",
                json=_policy_payload(development),
            )
            assert created_response.status_code == 201, created_response.text
            created = created_response.json()
            assert created["id"]
            assert created["content_hash"]

            revision_payload = _policy_payload(development)
            revision_payload["schedule"] = {
                "mode": "interval",
                "unit": "step",
                "interval": 10,
            }
            revised_response = client.post(
                f"/api/public/checkpoint-policies/{created['policy_id']}/revisions",
                json=revision_payload,
            )
            assert revised_response.status_code == 201, revised_response.text
            revised = revised_response.json()
            assert revised["revision_number"] == 2
            revision_page = client.get(
                f"/api/public/checkpoint-policies/{created['policy_id']}/revisions",
                params={"limit": 1, "offset": 1},
            ).json()
            assert revision_page["total"] == 2
            assert revision_page["offset"] == 1
            assert revision_page["items"] == [created]

            listed = client.get(
                "/api/public/checkpoint-policies",
                params={"trainer_mode": "sft"},
            ).json()
            assert [value["id"] for value in listed["items"]] == [revised["id"]]

            resolved_response = client.post(
                "/api/public/checkpoint-policies/resolve",
                json={
                    "policy_revision_id": created["id"],
                    "trainer_mode": "sft",
                    "total_budget": 100,
                    "budget_unit": "step",
                    "base_config": {"backend": "hf", "max_steps": 100},
                },
            )
            assert resolved_response.status_code == 200, resolved_response.text
            resolved = resolved_response.json()
            assert resolved["boundaries"] == [25, 50, 100]
            assert resolved["estimated_evaluation_count"] == 3

            invalid = _policy_payload(holdout, policy_id="holdout-policy")
            assert client.post("/api/public/checkpoint-policies", json=invalid).status_code == 400

            draft = client.put(
                "/api/public/workspace-drafts/experiment/new-group",
                json={"name": "New group", "content": {"step": 3}},
            ).json()
            assert draft["content"] == {"step": 3}
            assert client.get("/api/public/workspace-drafts/experiment/new-group").json()[
                "content"
            ] == {"step": 3}

            search = client.get("/api/public/search", params={"q": "Periodic"}).json()
            assert any(value["type"] == "checkpoint_policy" for value in search["items"])
    finally:
        database.close()


def test_checkpoint_policy_filtering_paginates_after_compatibility_checks(
    tmp_path: Path,
) -> None:
    service, database, _scheduler = _service(tmp_path)
    revision_id = _suite(service)
    try:
        compatible = service.create_checkpoint_policy(
            _policy_payload(revision_id, policy_id="step-policy")
        )
        incompatible_payload = _policy_payload(revision_id, policy_id="newer-cycle-policy")
        incompatible_payload["name"] = "Cycle quality checks"
        incompatible_payload["schedule"] = {
            "mode": "interval",
            "unit": "cycle",
            "interval": 1,
        }
        service.create_checkpoint_policy(incompatible_payload)

        page = service.list_checkpoint_policies(trainer_mode="sft", limit=1, offset=0)
        assert [value["id"] for value in page["items"]] == [compatible["id"]]
        assert page["total"] == 1
        assert page["has_more"] is False
    finally:
        database.close()


def test_matched_seed_analysis_decision_and_evidence_bundle_close_the_loop(
    tmp_path: Path,
) -> None:
    service, database, scheduler = _service(tmp_path)
    revision_id = _suite(service)
    try:
        group = service.create_run_group(
            {
                "name": "learning-rate cohort",
                "kind": "sweep",
                "trainer_mode": "sft",
                "suite_revision_id": revision_id,
                "base_config": {
                    "model": "local/model",
                    "data": str(tmp_path / "train.jsonl"),
                    "output_root": str(tmp_path / "runs"),
                    "max_steps": 10,
                },
                "seeds": [7, 11],
                "n_trials": 2,
                "sampler": "grid",
                "search_space": {"learning_rate": {"kind": "choice", "values": [1e-5, 2e-5]}},
            }
        )
        trial_ordinals = {trial["id"]: int(trial["ordinal"]) for trial in group["trials"]}
        # Complete the already-materialized work graph without running a model.
        # The evaluation result remains a seed-level outcome, never a sample-level replicate.
        for item in database.list_work_items(kinds=["training"], limit=100):
            claimed = scheduler.claim(work_item_id=item.id)
            assert claimed is not None
            scheduler.complete(claimed, result={"status": "trained"})
        for item in database.list_work_items(kinds=["evaluation"], limit=100):
            claimed = scheduler.claim(work_item_id=item.id)
            assert claimed is not None
            ordinal = trial_ordinals[str(item.launch_spec["trial_id"])]
            trial_run = database.get_trial_run(str(item.launch_spec["trial_run_id"]))
            assert trial_run is not None
            seed = int(trial_run.seed)
            value = 0.70 + ordinal * 0.10 + seed * 0.0001
            scheduler.complete(claimed, result={"metrics": {"accuracy": value}})

        analysis = service.create_run_group_analysis(
            group["id"],
            {
                "confidence": 0.95,
                "bootstrap_resamples": 10_000,
                "bootstrap_seed": 42,
                "practical_delta": 0.01,
            },
        )
        assert analysis["analysis"]["classification"] == "improved"
        assert analysis["analysis"]["matched_seed_count"] == 2
        assert analysis["analysis"]["compatibility"]["compatible"] is True

        with pytest.raises(ValueError, match="another run group"):
            service.create_research_decision(
                {
                    "analysis_snapshot_id": analysis["id"],
                    "selected_subject": {
                        "run_group_id": "not-this-group",
                        "trial_id": service.get_run_group(group["id"])["best_trial_id"],
                    },
                    "rationale": "This identity must not enter immutable evidence.",
                }
            )

        refreshed = service.get_run_group(group["id"])
        decision = service.create_research_decision(
            {
                "analysis_snapshot_id": analysis["id"],
                "selected_subject": {
                    "run_group_id": group["id"],
                    "trial_id": refreshed["best_trial_id"],
                },
                "rationale": "The matched-seed interval clears the declared practical delta.",
                "fork_spec": {
                    "run_group_id": group["id"],
                    "trial_id": refreshed["best_trial_id"],
                },
            }
        )
        assert decision["content_hash"]

        queued = service.create_evidence_bundle(
            {
                "analysis_snapshot_id": analysis["id"],
                "research_decision_id": decision["id"],
                "formats": ["markdown", "html", "json", "csv", "svg"],
            }
        )
        assert queued["status"] == "queued"
        assert queued["work_item_id"]
        work = database.get_work_item(queued["work_item_id"])
        assert work is not None
        terminal = WorkstationWorker(scheduler, heartbeat_interval=0.01).run_once()
        assert terminal is not None
        assert terminal.id == work.id
        assert terminal.status == "completed"
        completed = database.get_evidence_bundle(queued["id"])
        assert completed is not None and completed.status == "completed"
        bundle_root = Path(completed.storage_path)
        assert (bundle_root / "manifest.json").is_file()
        assert (bundle_root / "report.md").is_file()
        assert (bundle_root / "evidence.json").is_file()
        assert (bundle_root / "observations.csv").is_file()
        assert (bundle_root / "comparisons.csv").is_file()
        assert (bundle_root / "comparison.svg").is_file()
        report = (bundle_root / "report.md").read_text(encoding="utf-8")
        assert "## Pinned research scope" in report
        assert "## Statistical assumptions" in report
        assert "## Missing evidence" in report
        assert "## Runtime" in report
        evidence = json.loads((bundle_root / "evidence.json").read_text(encoding="utf-8"))
        assert evidence["bundle"]["request"]["run_group"]["id"] == group["id"]
    finally:
        database.close()


def test_workspace_drafts_expire_and_gate_review_missing_identity_is_not_found(
    tmp_path: Path,
) -> None:
    service, database, _scheduler = _service(tmp_path)
    try:
        saved = service.save_workspace_draft("experiment", "expiring", {"content": {"step": 2}})
        expired_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        with database._lock:
            database._conn.execute(
                "UPDATE workspace_drafts SET expires_at = ? WHERE id = ?",
                (expired_at, saved["id"]),
            )
            database._conn.commit()
        assert service.get_workspace_draft("experiment", "expiring") is None
        assert database.get_workspace_draft(saved["id"]) is None

        with pytest.raises(KeyError):
            service.review_checkpoint_gate(
                "missing-gate", {"action": "continue", "reason": "reviewed"}
            )
    finally:
        database.close()


def _completed_evaluation(
    database: RunDatabase,
    *,
    suite_revision_id: str,
    evaluation_id: str,
    subject_ref: str,
    value: float,
) -> None:
    database.create_evaluation(
        suite_revision_id=suite_revision_id,
        adapter_id="dataset_split",
        adapter_version="1",
        subject_type="checkpoint",
        subject_ref=subject_ref,
        subject_hash=f"hash-{evaluation_id}",
        reuse_key=f"reuse-{evaluation_id}",
        request={
            "subject": {
                "subject_type": "checkpoint",
                "subject_ref": subject_ref,
                "payload": {},
            }
        },
        evaluation_id=evaluation_id,
    )
    database.complete_evaluation(
        evaluation_id,
        metrics=[{"name": "accuracy", "value": value, "direction": "maximize"}],
        samples=[
            {
                "suite_item_id": "example",
                "record_id": "example",
                "input": "x",
                "expected": "y",
                "output": "y",
                "score": value,
                "passed": True,
                "valid": True,
                "mineable": True,
                "score_direction": "maximize",
            }
        ],
        result={"metrics": {"accuracy": value}},
        artifact_path=f"/evaluations/{evaluation_id}",
    )


def test_evaluation_history_and_drift_select_distinct_compatible_points(
    tmp_path: Path,
) -> None:
    service, database, _scheduler = _service(tmp_path)
    revision_id = _suite(service)
    try:
        _completed_evaluation(
            database,
            suite_revision_id=revision_id,
            evaluation_id="checkpoint-old",
            subject_ref="run-1",
            value=0.70,
        )
        _completed_evaluation(
            database,
            suite_revision_id=revision_id,
            evaluation_id="checkpoint-new",
            subject_ref="run-1",
            value=0.80,
        )

        history = service.evaluation_history(subject_ref="run-1", suite_revision_id=revision_id)
        assert [value["id"] for value in history["items"]] == [
            "checkpoint-new",
            "checkpoint-old",
        ]
        assert [value["primary_value"] for value in history["items"]] == [0.8, 0.7]

        drift = service.evaluation_drift(candidate_id="checkpoint-new", practical_delta=0.01)
        assert drift["base_id"] == "checkpoint-old"
        assert drift["candidate_id"] == "checkpoint-new"
        assert drift["classification"] == "improved"

        with pytest.raises(ValueError, match="two distinct"):
            service.evaluation_drift(base_id="checkpoint-new", candidate_id="checkpoint-new")
        with pytest.raises(ValueError, match="two evaluation ids"):
            service.evaluation_drift()
        with pytest.raises(ValueError, match="finite non-negative"):
            service.evaluation_drift(
                base_id="checkpoint-old",
                candidate_id="checkpoint-new",
                practical_delta=float("nan"),
            )
    finally:
        database.close()


def test_cohort_analysis_uses_one_shared_development_boundary(tmp_path: Path) -> None:
    service, database, scheduler = _service(tmp_path)
    revision_id = _suite(service)
    policy_payload = _policy_payload(revision_id, policy_id="shared-boundary")
    policy_payload.update(
        automatic_actions=True,
        schedule={
            "mode": "explicit",
            "unit": "step",
            "boundaries": [2, 4],
        },
    )
    policy = service.create_checkpoint_policy(policy_payload)
    try:
        group = service.create_run_group(
            {
                "version": 2,
                "name": "shared boundary cohort",
                "kind": "sweep",
                "trainer_mode": "sft",
                "base_config": {
                    "model": "local/model",
                    "data": str(tmp_path / "train.jsonl"),
                    "output_root": str(tmp_path / "runs"),
                    "max_steps": 4,
                },
                "seeds": [23],
                "n_trials": 2,
                "sampler": "grid",
                "search_space": {
                    "learning_rate": {
                        "kind": "choice",
                        "values": [1e-5, 2e-5],
                    }
                },
                "development_suite_revision_id": revision_id,
                "checkpoint_policy_revision_id": policy["id"],
            }
        )
        for trial in group["trials"]:
            run = trial["runs"][0]
            training = next(item for item in run["work_items"] if item["kind"] == "training")
            evaluation = next(item for item in run["work_items"] if item["kind"] == "evaluation")
            claimed = scheduler.claim(work_item_id=training["id"])
            assert claimed is not None
            scheduler.complete(claimed, result={"status": "trained"})
            claimed = scheduler.claim(work_item_id=evaluation["id"])
            assert claimed is not None
            scheduler.complete(
                claimed,
                result={"metrics": {"accuracy": 0.60 + 0.10 * trial["ordinal"]}},
            )
        service._experiment_engine().advance_ready_checkpoint_policy(group["id"])

        progressed = service.get_run_group(group["id"])
        leading_run = progressed["trials"][1]["runs"][0]
        second_segment = leading_run["segments"][-1]
        second_items = [
            item
            for item in leading_run["work_items"]
            if item.get("launch_spec", {}).get("trial_segment_id") == second_segment["id"]
        ]
        second_training = next(item for item in second_items if item["kind"] == "training")
        second_evaluation = next(item for item in second_items if item["kind"] == "evaluation")
        claimed = scheduler.claim(work_item_id=second_training["id"])
        assert claimed is not None
        scheduler.complete(claimed, result={"status": "trained farther"})
        claimed = scheduler.claim(work_item_id=second_evaluation["id"])
        assert claimed is not None
        scheduler.complete(claimed, result={"metrics": {"accuracy": 0.99}})

        analysis = service.create_run_group_analysis(group["id"], {})
        assert analysis["analysis"]["objective_boundary"] == {
            "ordinal": 0,
            "unit": "step",
            "value": 2,
        }
        assert sorted(
            observation["value"] for observation in analysis["request"]["observations"]
        ) == pytest.approx([0.60, 0.70])
        assert all(
            observation["metadata"]["objective_boundary_value"] == 2
            for observation in analysis["request"]["observations"]
        )
    finally:
        database.close()


def test_selected_insufficient_comparison_requires_an_explicit_override(
    tmp_path: Path,
) -> None:
    service, database, _scheduler = _service(tmp_path)
    try:
        snapshot = service._adaptive_engine().analyze_cohort(
            [
                {"subject_id": "base", "seed": 1, "metric": "score", "value": 0.5},
                {"subject_id": "base", "seed": 2, "metric": "score", "value": 0.5},
                {"subject_id": "good", "seed": 1, "metric": "score", "value": 0.8},
                {"subject_id": "good", "seed": 2, "metric": "score", "value": 0.8},
                {"subject_id": "sparse", "seed": 1, "metric": "score", "value": 0.9},
            ],
            metric="score",
            direction="maximize",
            baseline_subject_id="base",
            required_seeds=(1, 2),
            bootstrap_resamples=100,
        )
        assert snapshot.analysis["comparisons"]["good"]["classification"] == "improved"
        assert (
            snapshot.analysis["comparisons"]["sparse"]["classification"] == "insufficient_evidence"
        )

        request = {
            "analysis_snapshot_id": snapshot.id,
            "selected_subject": {"subject_id": "sparse"},
            "rationale": "The point estimate is interesting but incomplete.",
        }
        with pytest.raises(ValueError, match="insufficient evidence"):
            service.create_research_decision(request)
        accepted = service.create_research_decision(
            {
                **request,
                "override_reason": "Seed 2 is unavailable and explicitly acknowledged.",
            }
        )
        assert accepted["override_reason"]
    finally:
        database.close()


def test_gate_review_is_idempotent_but_cannot_rewrite_append_only_history(
    tmp_path: Path,
) -> None:
    service, database, _scheduler = _service(tmp_path)
    try:
        revision_id = _suite(service)
        policy = service.create_checkpoint_policy(_policy_payload(revision_id))
        original = database.create_checkpoint_gate_decision(
            policy_revision_id=policy["id"],
            plan_hash="plan",
            boundary_index=0,
            action="pause",
            reasons=["manual_review_required"],
            evidence={"boundary": {"value": 10, "unit": "step"}},
            idempotency_key="original-gate",
        )
        override = database.create_checkpoint_gate_decision(
            policy_revision_id=policy["id"],
            plan_hash="plan",
            boundary_index=0,
            action="continue",
            reasons=["operator_override"],
            evidence={"boundary": {"value": 10, "unit": "step"}},
            idempotency_key="reviewed-gate",
            override_of_id=original.id,
            override_reason="Evidence reviewed",
        )
        repeated = service.review_checkpoint_gate(
            original.id,
            {"action": "continue", "reason": "Evidence reviewed"},
        )
        assert repeated["id"] == override.id
        assert len(database.list_checkpoint_gate_decisions(limit=100)) == 2

        with pytest.raises(ValueError, match="already been reviewed"):
            service.review_checkpoint_gate(
                original.id,
                {"action": "stop", "reason": "Changed my mind"},
            )
        with pytest.raises(ValueError, match="cannot be reviewed again"):
            service.review_checkpoint_gate(
                override.id,
                {"action": "stop", "reason": "Nested override"},
            )
        assert len(database.list_checkpoint_gate_decisions(limit=100)) == 2
    finally:
        database.close()


def test_activity_center_surfaces_unresolved_checkpoint_review(tmp_path: Path) -> None:
    service, database, _scheduler = _service(tmp_path)
    revision_id = _suite(service)
    policy = service.create_checkpoint_policy(_policy_payload(revision_id))
    plan = service.resolve_checkpoint_policy(
        {
            "policy_revision_id": policy["id"],
            "trainer_mode": "sft",
            "total_budget": 10,
            "budget_unit": "step",
            "base_config": {"backend": "hf", "max_steps": 10},
        }
    )
    try:
        gate = database.create_checkpoint_gate_decision(
            policy_revision_id=policy["id"],
            plan_hash=plan["content_hash"],
            boundary_index=0,
            action="pause",
            reasons=["manual_review_required"],
            evidence={"boundary": {"value": 3, "unit": "step"}},
            idempotency_key="review-once",
        )
        activity = service.get_activity(limit=50)
        review = next(value for value in activity["items"] if value.get("domain_id") == gate.id)
        assert review["status"] == "awaiting_review"
        assert review["next_actions"] == ["inspect", "continue", "stop"]
    finally:
        database.close()


def test_public_trajectory_preserves_superseded_gate_history(tmp_path: Path) -> None:
    service, database, scheduler = _service(tmp_path)
    revision_id = _suite(service)
    policy = service.create_checkpoint_policy(_policy_payload(revision_id))
    try:
        group = service.create_run_group(
            {
                "version": 2,
                "name": "review history",
                "kind": "repeat",
                "trainer_mode": "sft",
                "base_config": {
                    "model": "local/model",
                    "data": str(tmp_path / "train.jsonl"),
                    "output_root": str(tmp_path / "runs"),
                    "max_steps": 100,
                },
                "seeds": [42],
                "development_suite_revision_id": revision_id,
                "checkpoint_policy_revision_id": policy["id"],
            }
        )
        run = group["trials"][0]["runs"][0]
        training = next(value for value in run["work_items"] if value["kind"] == "training")
        evaluation = next(value for value in run["work_items"] if value["kind"] == "evaluation")
        claimed_training = scheduler.claim(work_item_id=training["id"])
        assert claimed_training is not None
        scheduler.complete(claimed_training, result={"status": "trained"})
        claimed_evaluation = scheduler.claim(work_item_id=evaluation["id"])
        assert claimed_evaluation is not None
        scheduler.complete(claimed_evaluation, result={"metrics": {"accuracy": 0.75}})

        advanced = service._experiment_engine().advance_ready_checkpoint_policy(group["id"])
        original = advanced["outcomes"][0]["gate_decision"]
        assert original["action"] == "pause"
        override = service.review_checkpoint_gate(
            original["id"],
            {"action": "continue", "reason": "reviewed the complete evidence"},
        )

        trajectory = service.get_run_group_trajectory(group["id"])
        assert [value["id"] for value in trajectory["gate_decisions"]] == [
            original["id"],
            override["id"],
        ]
        assert trajectory["gate_decisions"][0]["status"] == "superseded"
        assert trajectory["gate_decisions"][0]["action"] == "pause"
        assert trajectory["gate_decisions"][1]["status"] == "overridden"
        assert trajectory["summary"]["awaiting_review"] == 0
    finally:
        database.close()
