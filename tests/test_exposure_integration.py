"""Operational contracts for evaluation and dataset exposure lineage."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest


@pytest.fixture
def db():
    from halo_forge.run_db import RunDatabase

    value = RunDatabase(":memory:")
    yield value
    value.close()


@pytest.fixture
def evaluation_lab(db, tmp_path):
    from halo_forge.evaluation_lab import EvaluationLabService

    value = EvaluationLabService(db, tmp_path / "evaluations")
    yield value
    value.shutdown()


def _catalog_version(db, version, *, parent_version_id=None):
    return db.create_dataset_version(
        dataset_id=version.dataset_id,
        version_id=version.version_id,
        parent_version_id=parent_version_id,
        recipe_hash=version.recipe_hash,
        recipe={},
        storage_path=version.path,
        status="completed",
        content_hash=version.content_hash,
        row_count=version.row_count,
        split_counts=version.split_counts,
        assets_materialized=version.materialized_assets,
    )


def test_evaluation_launch_records_identifiable_run_and_dataset_exposure(
    evaluation_lab, db, tmp_path
):
    from halo_forge.run_db import RunRecord

    model = tmp_path / "final-model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"model")
    db.upsert_run(
        RunRecord(
            run_id="run-1",
            status="completed",
            output_dir=str(tmp_path),
            final_model_path=str(model),
            indexed_at=datetime.now(timezone.utc).isoformat(),
        )
    )
    db.create_dataset(
        dataset_id="suite-data",
        name="suite data",
        modality="text",
        canonical_schema="sft",
    )
    db.create_dataset_version(
        dataset_id="suite-data",
        version_id="suite-version",
        recipe_hash="recipe",
        recipe={},
        storage_path=str(tmp_path / "suite-version"),
        status="completed",
        content_hash="suite-content",
    )
    _, revision = evaluation_lab.create_suite(
        name="development-evidence",
        purpose="development",
        items=[
            {
                "id": "dataset-item",
                "dataset_version_id": "suite-version",
                "split": "validation",
            },
            {"id": "inline-item", "input": "question", "expected": "answer"},
        ],
    )

    launch = evaluation_lab.launch_evaluation(
        suite_revision_id=revision.id,
        subject={"type": "run", "ref": "run-1"},
        submit=False,
    )

    exposures = db.list_exposures(run_id="run-1")
    assert [value.suite_item_id for value in exposures] == [
        "dataset-item",
        "inline-item",
    ]
    assert exposures[0].dataset_version_id == "suite-version"
    assert exposures[1].dataset_version_id is None
    assert all(value.exposure_type == "evaluation" for value in exposures)
    assert all(
        value.provenance["evaluation_id"] == launch.evaluation.id
        and value.provenance["suite_purpose"] == "development"
        and value.provenance["subject_ref"] == "run-1"
        for value in exposures
    )
    evaluation_lab.cancel_evaluation(launch.evaluation.id)


def test_holdout_failure_mining_policy_is_a_shared_hard_gate(evaluation_lab):
    from halo_forge.evaluation_lab import EvaluationLabError

    _, revision = evaluation_lab.create_suite(
        name="sealed-holdout",
        purpose="holdout",
        items=[{"id": "one", "input": "one", "expected": "one"}],
    )
    launch = evaluation_lab.launch_evaluation(
        suite_revision_id=revision.id,
        subject={"type": "model", "ref": "org/model", "revision": "commit"},
        submit=False,
    )

    policy = evaluation_lab.failure_mining_policy(
        candidate_evaluation_id=launch.evaluation.id
    )
    assert policy == {
        "allowed": False,
        "reason": "holdout suite evidence cannot enter a training dataset",
        "suite_id": revision.suite_id,
        "suite_revision_id": revision.id,
        "suite_purpose": "holdout",
        "base_evaluation_id": None,
        "candidate_evaluation_id": launch.evaluation.id,
    }
    with pytest.raises(EvaluationLabError, match="holdout suite evidence"):
        evaluation_lab.require_failure_mining_allowed(
            candidate_evaluation_id=launch.evaluation.id
        )
    evaluation_lab.cancel_evaluation(launch.evaluation.id)


def test_failure_mined_child_inherits_ancestry_and_records_selected_evidence(
    db, tmp_path
):
    from halo_forge.data_lab import DatasetLab, FailureMiningBuilder
    from halo_forge.evaluation_lab import EvaluationLabService

    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        json.dumps({"prompt": "parent", "response": "answer"}) + "\n",
        encoding="utf-8",
    )
    lab = DatasetLab(tmp_path / "datasets")
    try:
        source = lab.add_source(
            {
                "kind": "local",
                "path": str(source_path),
                "canonical_kind": "sft",
            },
            dataset_id="feedback-data",
        )
        parent = lab.build(
            source.id, {"steps": [{"kind": "normalize", "fields": ["prompt"]}]}
        )
        db.create_dataset(
            dataset_id=parent.dataset_id,
            name="feedback data",
            modality="text",
            canonical_schema="sft",
        )
        _catalog_version(db, parent)
        evaluation_lab = EvaluationLabService(db, tmp_path / "evaluations")
        try:
            _, revision = evaluation_lab.create_suite(
                name="development-loop",
                purpose="development",
                items=[{"id": "prior-item"}, {"id": "failed-item"}],
            )
        finally:
            evaluation_lab.shutdown()
        parent_exposure = db.record_exposure(
            suite_revision_id=revision.id,
            suite_item_id="prior-item",
            exposure_type="evaluation",
            dataset_version_id=parent.version_id,
            provenance={"evaluation_id": "prior-eval"},
        )
        comparison = {
            "candidate_evaluation_id": "candidate-eval",
            "suite_revision_id": revision.id,
            "sample_deltas": [
                {
                    "record_id": "external-record",
                    "suite_item_id": "failed-item",
                    "outcome": "regression",
                    "base": {"passed": True},
                    "candidate": {
                        "input": "failed prompt",
                        "expected": "reviewed answer",
                        "passed": False,
                    },
                }
            ],
        }
        child = FailureMiningBuilder(lab.store).build(
            parent_version_id=parent.version_id,
            comparison=comparison,
            selector="regression",
        )
        _catalog_version(db, child, parent_version_id=parent.version_id)

        first = lab.sync_version_exposures(child.version_id, database=db)
        second = lab.sync_version_exposures(child.version_id, database=db)
        child_exposures = db.list_exposures(dataset_version_id=child.version_id)

        assert len(child_exposures) == 2
        inherited = next(value for value in child_exposures if value.exposure_type == "inherited")
        mined = next(value for value in child_exposures if value.exposure_type == "failure_mining")
        assert inherited.inherited_from_id == parent_exposure.id
        assert inherited.provenance["source_provenance"] == {
            "evaluation_id": "prior-eval"
        }
        assert mined.suite_item_id == "failed-item"
        assert mined.provenance["record_id"] == "external-record"
        assert mined.provenance["suite_purpose"] == "development"
        assert mined.provenance["candidate_evaluation_id"] == "candidate-eval"
        assert {value["id"] for value in first["inherited"] + first["direct"]} == {
            value["id"] for value in second["inherited"] + second["direct"]
        }
    finally:
        lab.close()


def test_reviewed_label_exposure_is_direct_and_idempotent(db, tmp_path):
    from halo_forge.data_lab.service import record_review_label_exposures
    from halo_forge.evaluation_lab import EvaluationLabService

    db.create_dataset(
        dataset_id="review-child-data",
        name="review child data",
        modality="text",
        canonical_schema="sft",
    )
    db.create_dataset_version(
        dataset_id="review-child-data",
        version_id="review-child-version",
        recipe_hash="review-recipe",
        recipe={},
        storage_path=str(tmp_path / "review-child-version"),
        status="completed",
        content_hash="review-child-content",
    )
    evaluation_lab = EvaluationLabService(db, tmp_path / "review-evaluations")
    try:
        _, revision = evaluation_lab.create_suite(
            name="development-review-source",
            purpose="development",
            items=[{"id": "review-source-item", "input": "prompt"}],
        )
    finally:
        evaluation_lab.shutdown()
    details = {
        "label_set_revision_id": "label-revision-1",
        "label_set_id": "label-set-1",
        "label_set_content_hash": "label-content",
        "exposure_records": [
            {
                "suite_revision_id": revision.id,
                "suite_item_id": "review-source-item",
                "record_id": "record-1",
                "review_item_id": "review-item-1",
                "purpose": "development",
                "source_kind": "evaluation",
                "source_ref": "evaluation-1",
            }
        ],
    }

    first = record_review_label_exposures(
        db, child_version_id="review-child-version", details=details
    )
    second = record_review_label_exposures(
        db, child_version_id="review-child-version", details=details
    )

    assert [value.id for value in first] == [value.id for value in second]
    exposure = first[0]
    assert exposure.exposure_type == "human_review_label"
    assert exposure.suite_revision_id == revision.id
    assert exposure.suite_item_id == "review-source-item"
    assert exposure.provenance["label_set_revision_id"] == "label-revision-1"
    assert exposure.provenance["source"] == "review_label_set"


def test_public_version_reconciliation_syncs_exposures_after_catalog_mirror(
    db, tmp_path
):
    from halo_forge.public_api.service import PublicApiService

    db.create_dataset(
        dataset_id="reconciled-data",
        name="reconciled data",
        modality="text",
        canonical_schema="sft",
    )
    calls = []

    class DatasetFacade:
        def get_version(self, version_id, *, dataset_id=None):
            return {
                "dataset_id": dataset_id,
                "version_id": version_id,
                "path": str(tmp_path / version_id),
                "content_hash": "content",
                "recipe_hash": "recipe",
                "source_fingerprint": "source",
                "materialized_assets": False,
                "split_counts": {"train": 1},
                "row_count": 1,
            }

        def sync_version_exposures(
            self, version_id, *, database=None, dataset_id=None
        ):
            assert database is db
            assert database.get_dataset_version(version_id) is not None
            calls.append((version_id, dataset_id))
            return {"inherited": [], "direct": []}

    service = PublicApiService(
        database=db,
        dataset_lab=DatasetFacade(),
        base_path=tmp_path,
        dataset_storage_root=tmp_path / "datasets",
    )
    job = db.create_dataset_job(
        job_id="build-job",
        dataset_id="reconciled-data",
        job_type="build",
        status="completed",
        stage="published",
        request={"recipe": {"steps": [{"kind": "normalize"}]}},
    )

    service._sync_completed_engine_version(
        job,
        {
            "version_id": "child-version",
            "dataset_id": "reconciled-data",
            "recipe_hash": "recipe",
            "content_hash": "content",
            "row_count": 1,
            "split_counts": {"train": 1},
        },
    )
    # Reconciliation can run on every poll without duplicating ledger rows;
    # the Dataset Lab sync operation itself owns idempotence.
    service._sync_completed_engine_version(job, {"version_id": "child-version"})

    assert calls == [
        ("child-version", "reconciled-data"),
        ("child-version", "reconciled-data"),
    ]


def test_public_preview_and_build_both_use_evaluation_holdout_policy(
    db, tmp_path
):
    from halo_forge.evaluation_lab import EvaluationLabService
    from halo_forge.public_api.service import PublicApiService

    evaluations = EvaluationLabService(db, tmp_path / "evaluations")
    try:
        _, revision = evaluations.create_suite(
            name="public-holdout",
            purpose="holdout",
            items=[
                {
                    "id": "one",
                    "input": "one",
                    "expected": "one",
                    "score": 0.0,
                }
            ],
        )
        launched = evaluations.launch_evaluation(
            suite_revision_id=revision.id,
            subject={"type": "model", "ref": "org/model", "revision": "commit"},
        )
        completed = evaluations.jobs.wait(launched.evaluation.id, timeout=5)
        assert completed.status == "completed"

        db.create_dataset(
            dataset_id="public-parent",
            name="public parent",
            modality="text",
            canonical_schema="sft",
        )
        db.create_dataset_version(
            dataset_id="public-parent",
            version_id="parent-version",
            recipe_hash="recipe",
            recipe={},
            storage_path=str(tmp_path / "parent-version"),
            status="completed",
            content_hash="parent-content",
        )
        calls = []
        original = evaluations.require_failure_mining_allowed

        def tracked_policy(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        evaluations.require_failure_mining_allowed = tracked_policy
        service = PublicApiService(
            database=db,
            evaluation_lab=evaluations,
            base_path=tmp_path,
            dataset_storage_root=tmp_path / "datasets",
        )

        with pytest.raises(ValueError, match="holdout suite evidence"):
            service.preview_failure_mining(
                {"candidate_id": completed.id, "selector": "candidate_failure"}
            )
        with pytest.raises(ValueError, match="holdout suite evidence"):
            service.build_failure_mined_dataset(
                {
                    "dataset_id": "public-parent",
                    "parent_version_id": "parent-version",
                    "candidate_id": completed.id,
                    "selector": "candidate_failure",
                }
            )
        assert calls == [
            {
                "candidate_evaluation_id": completed.id,
                "base_evaluation_id": None,
            },
            {
                "candidate_evaluation_id": completed.id,
                "base_evaluation_id": None,
            },
        ]
    finally:
        evaluations.shutdown()
