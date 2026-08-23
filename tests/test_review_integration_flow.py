from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from halo_forge.data_lab import (
    DatasetLab,
    Recipe,
    RecipeResult,
    ReviewDatasetBuilder,
    SourceSnapshot,
    SourceSpec,
    VersionStore,
)
from halo_forge.data_lab.errors import VersionError
from halo_forge.data_lab.identity import INTERNAL_LINEAGE_KEY
from halo_forge.review_lab import ReviewEligibilityError, ReviewLabService
from halo_forge.public_api.service import PublicApiService
from halo_forge.run_db import LabV4Catalog, RunDatabase
from halo_forge.workstation_jobs import WorkstationScheduler, WorkstationWorker
from halo_forge.workstation_jobs.resources import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
)


def _capacity(path: Path) -> WorkstationCapacity:
    gib = 1024**3
    return WorkstationCapacity(
        sampled_at=datetime.now(timezone.utc),
        disk=DiskCapacity(
            path=str(path),
            total_bytes=500 * gib,
            used_bytes=100 * gib,
            free_bytes=400 * gib,
        ),
        memory=MemoryCapacity(
            total_bytes=64 * gib,
            used_bytes=8 * gib,
            available_bytes=56 * gib,
            source="test",
        ),
    )


def _parent_version(root: Path):
    store = VersionStore(root)
    source_path = root.parent / "parent.jsonl"
    rows = [
        {"id": "train-one", "prompt": "train", "response": "old train"},
        {"id": "validation-one", "prompt": "validate", "response": "old validation"},
    ]
    source_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    from halo_forge.data_lab.sources import load_source

    source = load_source(SourceSpec(kind="local", path=str(source_path)))
    version = store.publish(
        dataset_id="reviewed-sft",
        recipe=Recipe.from_value(
            {"name": "parent", "schema": "sft", "steps": [{"kind": "validate"}]}
        ),
        result=RecipeResult(
            records=rows,
            splits={"train": [rows[0]], "validation": [rows[1]]},
        ),
        source=SourceSnapshot(
            spec=source.spec,
            records=rows,
            fingerprint=source.fingerprint,
            assets=source.assets,
            size_bytes=source.size_bytes,
            file_count=source.file_count,
        ),
    )
    validation = store.load_records_with_lineage(
        version.version_id,
        dataset_id=version.dataset_id,
        split="validation",
    )[0]
    validation_identity = store.load_lineage(
        version.version_id,
        dataset_id=version.dataset_id,
        split="validation",
    )[0]
    return store, version, validation, validation_identity


def test_review_to_durable_verified_child_preserves_lineage_and_split_fidelity(tmp_path: Path):
    database_path = tmp_path / "runs.db"
    database = RunDatabase(str(database_path))
    scheduler = WorkstationScheduler(
        database,
        worker_id="review-worker",
        capacity_probe=_capacity,
    )
    dataset_root = tmp_path / "datasets"
    store, parent, source_row, source_identity = _parent_version(dataset_root)
    review_root = tmp_path / "reviews"
    review = ReviewLabService(database, review_root)
    _, schema_revision = review.create_schema(
        name="SFT corrections",
        modality="text",
        task_type="text_correction",
    )
    source_record_id = source_row[INTERNAL_LINEAGE_KEY]["record_id"]
    batch = review.create_acquisition(
        [
            {
                "record_id": source_record_id,
                "record": {
                    key: value
                    for key, value in source_row.items()
                    if key != INTERNAL_LINEAGE_KEY
                },
                "source": {
                    "kind": "dataset_version",
                    "ref": parent.version_id,
                    "split": "validation",
                },
            }
        ]
    )
    queue = review.create_queue(
        batch.id,
        schema_revision.id,
        policy={"mode": "one_pass"},
    )
    item = review.list_items(queue.id)[0]
    review.submit_event(
        item.id,
        "label",
        {"corrected_text": "reviewed answer"},
        idempotency_key="reviewed-correction",
        expected_active_event_id=None,
        reviewer_key="operator",
    )
    label_revision = review.publish_label_set(
        queue.id,
        build_mode="replace_by_record_id",
    )
    label_path = Path(label_revision.storage_path)
    assert (label_path / "canonical.jsonl").is_file()
    assert not (label_path / "records.jsonl").exists()

    database.create_dataset(
        dataset_id=parent.dataset_id,
        name="Reviewed SFT",
        modality="text",
        canonical_schema="sft",
    )
    database.create_dataset_version(
        dataset_id=parent.dataset_id,
        version_id=parent.version_id,
        recipe_hash=parent.recipe_hash,
        recipe={},
        storage_path=parent.path,
        status="completed",
        content_hash=parent.content_hash,
        row_count=parent.row_count,
        split_counts=parent.split_counts,
    )
    api = PublicApiService(
        database=database,
        review_storage_root=review_root,
        dataset_storage_root=dataset_root,
        workstation_scheduler=scheduler,
    )
    inferred_target: dict[str, object] = {}
    inferred_dataset, inferred_dataset_id = api._resolve_review_dataset_target(
        label_revision.id, inferred_target, create=False
    )
    assert inferred_dataset is not None
    assert inferred_dataset_id == parent.dataset_id
    assert inferred_target["parent_version_id"] == parent.version_id

    lab = DatasetLab(dataset_root, scheduler=scheduler)
    job = lab.start_review_build_job(
        label_revision.id,
        review_root=review_root,
        database_path=str(database_path),
        dataset_id=parent.dataset_id,
        parent_version_id=parent.version_id,
        build_mode="replace_by_record_id",
        target_split="train",
    )
    assert job.work_item_id
    terminal = WorkstationWorker(scheduler, heartbeat_interval=0.05).run_once()
    assert terminal is not None and terminal.status == "completed"
    completed_job = lab.get_job(job.id)
    assert completed_job.status == "succeeded"
    child = store.get(parent.dataset_id, completed_job.result["version_id"])
    assert store.verify(
        child.version_id,
        dataset_id=child.dataset_id,
        verify_source=True,
    )["valid"] is True

    training = store.load_records_with_lineage(
        child.version_id,
        dataset_id=child.dataset_id,
        split="train",
    )
    validation = store.load_records_with_lineage(
        child.version_id,
        dataset_id=child.dataset_id,
        split="validation",
    )
    corrected = next(row for row in training if row.get("response") == "reviewed answer")
    assert corrected[INTERNAL_LINEAGE_KEY]["record_id"] == source_record_id
    assert source_identity.instance_id in corrected[INTERNAL_LINEAGE_KEY]["parent_instance_ids"]
    assert all(
        row[INTERNAL_LINEAGE_KEY]["record_id"] != source_record_id for row in validation
    )

    lab.close()
    database.close()


def test_review_durable_handler_is_idempotent_and_protected_comparison_base_is_rejected(
    tmp_path: Path,
):
    database = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(
        database,
        worker_id="review-worker",
        capacity_probe=_capacity,
    )
    work = scheduler.enqueue(
        kind="review_acquisition",
        launch_spec={
            "handler": "review_lab.execute_work_item",
            "review_root": str(tmp_path / "reviews"),
            "action": "build_review_batch",
            "name": "Durable proposal",
            "records": [
                {
                    "record_id": "record-one",
                    "record": {"prompt": "question"},
                    "source": {"kind": "imported"},
                }
            ],
        },
        resource_class="cpu",
        domain_kind="acquisition_batch",
        domain_id="pending",
    )
    first = WorkstationWorker(scheduler, heartbeat_interval=0.05).run_once()
    assert first is not None and first.status == "completed"
    batch_id = first.result["id"]
    review = ReviewLabService(database, tmp_path / "reviews")
    assert review.get_acquisition(batch_id).work_item_id == work.id  # type: ignore[union-attr]

    # Retrying the same resolved launch spec cannot create a second proposal.
    duplicate = review.execute_work_item({"id": "retry", **work.to_dict()})
    assert duplicate["id"] == batch_id
    assert len(review.list_acquisitions()) == 1

    with pytest.raises(ReviewEligibilityError, match="protected_suite_purpose:holdout"):
        review.create_evaluation_comparison_acquisition(
            [
                {
                    "record_id": "shared",
                    "record": {"prompt": "secret"},
                    "evidence": {"passed": True, "valid": True, "mineable": True},
                    "source": {"kind": "evaluation", "purpose": "holdout"},
                }
            ],
            [
                {
                    "record_id": "shared",
                    "record": {"prompt": "development"},
                    "evidence": {"passed": False, "valid": True, "mineable": True},
                    "source": {"kind": "evaluation", "purpose": "development"},
                }
            ],
            strategies=[{"kind": "regression"}],
        )
    database.close()


def test_public_source_acquisition_is_durable_restart_safe_and_manifested(
    tmp_path: Path,
):
    database = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(
        database,
        worker_id="review-source-worker",
        capacity_probe=_capacity,
    )
    source = tmp_path / "unlabeled.jsonl"
    source.write_text(
        "".join(
            json.dumps(value) + "\n"
            for value in [
                {"input": "keep", "output": "accepted", "topic": "target"},
                {"input": "drop", "output": "ignored", "topic": "other"},
            ]
        ),
        encoding="utf-8",
    )
    review_root = tmp_path / "reviews"
    api = PublicApiService(
        database=database,
        review_storage_root=review_root,
        dataset_storage_root=tmp_path / "datasets",
        evaluation_storage_root=tmp_path / "evaluations",
        artifact_storage_root=tmp_path / "artifacts",
        workstation_scheduler=scheduler,
    )

    queued = api.create_acquisition_batch(
        {
            "name": "Imported proposal",
            "sources": [{"kind": "jsonl", "ref": str(source)}],
            "strategies": [{"kind": "explicit"}],
            "filters": [
                {"scope": "record", "field": "topic", "op": "eq", "value": "target"}
            ],
            "metadata": {
                "projection": {
                    "schema": "sft",
                    "fields": {
                        "prompt": "input",
                        "response": "output",
                        "topic": "topic",
                    },
                }
            },
            "seed": 17,
        }
    )
    assert queued["status"] == "queued"
    assert queued["work_item_id"]
    cancelled = api.cancel_acquisition_batch(queued["id"])
    assert cancelled["status"] == "cancelled"
    retried = api.retry_acquisition_batch(queued["id"])
    assert retried["status"] == "queued"
    assert retried["work_item_id"] == queued["work_item_id"]
    # The same state synchronization applies when the operator acts from the
    # global Activity Center rather than the Review Studio route.
    scheduler.cancel(queued["work_item_id"])
    assert api.get_acquisition_batch(queued["id"])["status"] == "cancelled"
    scheduler.retry(
        queued["work_item_id"], reason="resume from Activity Center", force=True
    )
    assert api.get_acquisition_batch(queued["id"])["status"] == "queued"

    terminal = WorkstationWorker(scheduler, heartbeat_interval=0.05).run_once()
    assert terminal is not None and terminal.status == "completed"
    completed = api.get_acquisition_batch(queued["id"])
    assert completed is not None
    assert completed["status"] == "ready"
    assert completed["row_count"] == 1
    candidates = api.list_acquisition_candidates(queued["id"])["items"]
    assert [value["record"]["prompt"] for value in candidates] == ["keep"]
    assert candidates[0]["record"]["response"] == "accepted"
    manifest_root = review_root / "acquisitions" / queued["id"]
    manifest = json.loads((manifest_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["request"]["filters"][0]["field"] == "topic"
    assert ReviewLabService(database, review_root).acquisition_manifests.verify(
        queued["id"], expected_content_hash=completed["content_hash"]
    ).valid

    duplicate = api.create_acquisition_batch(
        {
            "name": "Imported proposal",
            "sources": [{"kind": "jsonl", "ref": str(source)}],
            "strategies": [{"kind": "explicit"}],
            "filters": [
                {"scope": "record", "field": "topic", "op": "eq", "value": "target"}
            ],
            "metadata": {
                "projection": {
                    "schema": "sft",
                    "fields": {
                        "prompt": "input",
                        "response": "output",
                        "topic": "topic",
                    },
                }
            },
            "seed": 17,
        }
    )
    duplicate_terminal = WorkstationWorker(
        scheduler, heartbeat_interval=0.05
    ).run_once()
    assert duplicate_terminal is not None and duplicate_terminal.status == "completed"
    reused = api.get_acquisition_batch(duplicate["id"])
    assert reused is not None
    assert reused["id"] == queued["id"]
    assert reused["reused_from_batch_id"] == duplicate["id"]
    database.close()


@pytest.mark.parametrize(
    ("purpose", "split", "reason"),
    [
        ("operational", "validation", "protected_suite_purpose:operational"),
        ("development", "test", "protected_split:test"),
        ("development", "canary", "protected_split:canary"),
    ],
)
def test_persisted_evaluation_sources_refuse_protected_evidence(
    tmp_path: Path, purpose: str, split: str, reason: str
):
    database = RunDatabase(":memory:")
    suite = database.create_benchmark_suite(
        name=f"{purpose}-{split}", purpose=purpose
    )
    revision = database.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash=f"hash-{purpose}-{split}",
        items=[{"id": "protected-item", "split": split}],
        primary_metric="accuracy",
        direction="maximize",
    )
    evaluation = database.create_evaluation(
        suite_revision_id=revision.id,
        adapter_id="test",
        adapter_version="1",
        subject_type="model",
        subject_ref="candidate",
        subject_hash="candidate-hash",
        reuse_key=f"reuse-{purpose}-{split}",
        request={},
    )
    database.complete_evaluation(
        evaluation.id,
        metrics=[{"name": "accuracy", "value": 0.0, "direction": "maximize"}],
        samples=[
            {
                "suite_item_id": "protected-item",
                "record_id": "protected-record",
                "input": "secret",
                "output": "candidate",
                "score": 0.0,
                "passed": False,
                "valid": True,
                "mineable": True,
                "evidence_kind": "per_example",
            }
        ],
        result={},
        artifact_path=str(tmp_path / "evaluation"),
    )
    api = PublicApiService(
        database=database,
        review_storage_root=tmp_path / "reviews",
    )
    records = api._review_acquisition_records(
        {"sources": [{"kind": "evaluation", "ref": evaluation.id}]}
    )
    with pytest.raises(ReviewEligibilityError, match=reason):
        ReviewLabService(database, tmp_path / "reviews").create_acquisition(records)
    database.close()


def test_on_demand_suggestion_runs_as_durable_work_and_stays_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(
        database,
        worker_id="suggestion-worker",
        capacity_probe=_capacity,
    )
    review_root = tmp_path / "reviews"
    review = ReviewLabService(database, review_root)
    _, schema = review.create_schema(
        name="Suggested binary review",
        modality="text",
        task_type="binary",
    )
    batch = review.create_acquisition(
        [{"record_id": "one", "record": {"prompt": "Question"}}]
    )
    queue = review.create_queue(
        batch.id,
        schema.id,
        policy={"mode": "one_pass", "allow_suggestions": True},
    )
    item = review.list_items(queue.id)[0]
    api = PublicApiService(
        database=database,
        review_storage_root=review_root,
        dataset_storage_root=tmp_path / "datasets",
        evaluation_storage_root=tmp_path / "evaluations",
        artifact_storage_root=tmp_path / "artifacts",
        workstation_scheduler=scheduler,
    )
    monkeypatch.setattr(
        "halo_forge.data_lab.integrations.configured_teacher",
        lambda prompt, parameters, record: {"accepted": True},
    )

    queued = api.generate_review_suggestions(
        item.id,
        {
            "provider": "ollama",
            "model_revision": "teacher@pinned",
            "parameters": {
                "temperature": 0,
                "api_key": "never-store",
                "transport": {"authorization": "also-never-store"},
            },
        },
    )
    assert queued["status"] == "queued"
    persisted_work = database.get_work_item(queued["work_item_id"])
    assert persisted_work is not None
    assert "never-store" not in json.dumps(persisted_work.launch_spec)
    terminal = WorkstationWorker(scheduler, heartbeat_interval=0.05).run_once()
    assert terminal is not None and terminal.status == "completed"
    suggestions = api.list_review_suggestions(item.id)["items"]
    assert len(suggestions) == 1
    assert suggestions[0]["output"] is None
    stored = review.get_suggestion(suggestions[0]["id"], include_hidden=True)
    assert stored is not None
    assert "api_key" not in json.dumps(stored.provenance)
    assert stored.provenance["runtime_identity"]["provider"] == "ollama"

    cancellation = {}

    def cancel_during_generation(prompt, parameters, record):
        scheduler.cancel(cancellation["work_item_id"])
        return {"accepted": False}

    monkeypatch.setattr(
        "halo_forge.data_lab.integrations.configured_teacher",
        cancel_during_generation,
    )
    queued_cancel = api.generate_review_suggestions(
        item.id,
        {
            "provider": "ollama",
            "model_revision": "teacher-cancelled@pinned",
        },
    )
    cancellation["work_item_id"] = queued_cancel["work_item_id"]
    cancelled = WorkstationWorker(scheduler, heartbeat_interval=0.05).run_once()
    assert cancelled is not None and cancelled.status == "cancelled"
    assert len(review.list_suggestions(item.id)) == 1
    database.close()


def test_label_publication_is_durable_and_ignores_only_post_publication_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database = RunDatabase(str(tmp_path / "publication.db"))
    scheduler = WorkstationScheduler(database, capacity_probe=_capacity)
    review_root = tmp_path / "reviews"
    review = ReviewLabService(database, review_root)
    _, schema = review.create_schema(
        name="Durable publication",
        modality="text",
        task_type="binary",
    )
    batch = review.create_acquisition(
        [{"record_id": "publication-record", "record": {"prompt": "Question"}}]
    )
    queue = review.create_queue(batch.id, schema.id)
    item = review.list_items(queue.id)[0]
    review.submit_event(
        item.id,
        "label",
        {"accepted": True},
        idempotency_key="publication-label",
        expected_active_event_id=None,
    )
    public = PublicApiService(
        database=database,
        review_storage_root=review_root,
        workstation_scheduler=scheduler,
    )
    accepted = public.publish_label_set(queue.id, {})
    assert accepted["status"] == "queued"
    assert review.get_queue(queue.id).latest_label_set_revision_id is None

    original = ReviewLabService.publish_label_set

    def publish_then_receive_late_cancel(service, *args, **kwargs):
        revision = original(service, *args, **kwargs)
        scheduler.cancel(accepted["work_item_id"])
        return revision

    monkeypatch.setattr(ReviewLabService, "publish_label_set", publish_then_receive_late_cancel)
    finished = WorkstationWorker(scheduler, heartbeat_interval=0.05).run_once()
    assert finished is not None and finished.status == "completed"
    assert finished.cancel_requested is False
    assert review.get_queue(queue.id).latest_label_set_revision_id is not None
    events = LabV4Catalog(database).list_events(
        work_item_id=accepted["work_item_id"]
    )
    assert events[-1].to_dict()["payload"]["late_cancel_ignored"] is True
    database.close()


def test_acquisition_process_recovery_marks_domain_for_reconciliation(tmp_path: Path):
    database = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(
        database,
        worker_id="departed-worker",
        capacity_probe=_capacity,
    )
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"prompt": "one"}) + "\n", encoding="utf-8")
    api = PublicApiService(
        database=database,
        review_storage_root=tmp_path / "reviews",
        workstation_scheduler=scheduler,
    )
    queued = api.create_acquisition_batch(
        {"sources": [{"kind": "jsonl", "ref": str(source)}]}
    )
    claimed = scheduler.claim(
        work_item_id=queued["work_item_id"],
        child_pid=987654,
        child_pid_started_at=123.0,
    )
    assert claimed is not None

    recovering = WorkstationScheduler(
        database,
        worker_id="replacement-worker",
        process_probe=lambda _pid, _started: False,
        capacity_probe=_capacity,
    )
    outcome = recovering.recover_or_adopt()
    assert [value.id for value in outcome.interrupted] == [queued["work_item_id"]]
    assert database.get_work_item(queued["work_item_id"]).status == "needs_reconciliation"
    assert api.get_acquisition_batch(queued["id"])["status"] == "needs_reconciliation"
    recovering.retry(
        queued["work_item_id"], reason="verified dead worker", force=True
    )
    assert api.get_acquisition_batch(queued["id"])["status"] == "queued"
    database.close()


def test_diversity_source_acquisition_generates_pinned_embeddings_as_heavy_work(
    tmp_path: Path,
):
    database = RunDatabase(str(tmp_path / "diversity.db"))
    scheduler = WorkstationScheduler(database)
    review = ReviewLabService(database, tmp_path / "reviews")

    class FakeEmbeddingEngine:
        def __init__(self):
            self.calls = []

        def embed_envelopes(self, envelopes, *, embedding_revision):
            self.calls.append((len(envelopes), embedding_revision))
            output = []
            for index, envelope in enumerate(envelopes):
                value = dict(envelope)
                value["evidence"] = {
                    **dict(value.get("evidence") or {}),
                    "embedding": [float(index + 1), float(index % 2)],
                    "embedding_revision": embedding_revision,
                    "embedding_provenance": {
                        "engine": "test-model-adapter",
                        "revision": embedding_revision,
                    },
                }
                output.append(value)
            return output

    embedding_engine = FakeEmbeddingEngine()
    public = PublicApiService(
        database=database,
        review_lab=review,
        workstation_scheduler=scheduler,
        review_storage_root=tmp_path / "reviews",
        dataset_storage_root=tmp_path / "datasets",
        evaluation_storage_root=tmp_path / "evaluations",
        artifact_storage_root=tmp_path / "artifacts",
        review_embedding_engine=embedding_engine,
    )
    payload = {
        "name": "Media diversity",
        "records": [
            {
                "record_id": f"image-{index}",
                "record": {"image": f"asset-{index}.png", "prompt": "Describe it"},
                "source": {"kind": "dataset_version", "ref": "version-images"},
            }
            for index in range(4)
        ],
        "strategies": [
            {
                "kind": "diversity",
                "quota": 2,
                "options": {"embedding_revision": "image:org/clip@commit-123"},
            }
        ],
        "seed": 17,
    }
    queued = public.create_acquisition_batch(payload)
    work = database.get_work_item(queued["work_item_id"])
    assert work is not None and work.resource_class == "accelerator"
    assert work.resource_requirements["embedding_revision"] == "image:org/clip@commit-123"

    resolved = public.resolve_acquisition_batch(queued["id"], payload, work_item_id=work.id)
    assert resolved["status"] == "ready"
    assert resolved["row_count"] == 2
    assert embedding_engine.calls == [(4, "image:org/clip@commit-123")]
    candidates = review.list_acquisition_candidates(resolved["id"], limit=10)
    assert len(candidates) == 2
    assert all(
        value.evidence["embedding_revision"] == "image:org/clip@commit-123"
        for value in candidates
    )
    database.close()


def test_large_review_queue_builds_every_page_and_protected_targets_are_refused(
    tmp_path: Path,
):
    database_path = tmp_path / "runs.db"
    database = RunDatabase(str(database_path))
    review_root = tmp_path / "reviews"
    review = ReviewLabService(database, review_root)
    _, revision = review.create_schema(
        name="Topic labels",
        modality="text",
        task_type="categorical",
        definition={"labels": ["keep"]},
    )
    records = [
        {
            "record_id": f"record-{index:04d}",
            "record": {"prompt": f"question {index}", "response": f"answer {index}"},
            "source": {"kind": "imported"},
        }
        for index in range(1001)
    ]
    batch = review.create_acquisition(records)
    queue = review.create_queue(
        batch.id,
        revision.id,
        policy={"mode": "one_pass"},
    )
    items = review.list_items(queue.id, limit=1000)
    final_item = review.list_items(queue.id, limit=1, offset=1000)[0]
    review.submit_event_batch(
        queue.id,
        [
            {
                "item_id": item.id,
                "event_type": "label",
                "payload": {"label": "keep"},
                "idempotency_key": f"label-{item.ordinal}",
                "expected_active_event_id": None,
            }
            for item in items
        ],
    )
    review.submit_event(
        final_item.id,
        "label",
        {"label": "keep"},
        idempotency_key="label-1000",
        expected_active_event_id=None,
    )
    label_revision = review.publish_label_set(queue.id, build_mode="append")
    assert label_revision.row_count == 1001

    lab = DatasetLab(tmp_path / "datasets")
    job = lab.start_review_build_job(
        label_revision.id,
        review_root=review_root,
        database_path=str(database_path),
        dataset_id="large-reviewed-dataset",
        build_mode="append",
        submit=False,
    )
    completed = lab.run_queued(job.id)
    assert completed.status == "succeeded"
    child = lab.get_version(
        completed.result["version_id"],
        dataset_id="large-reviewed-dataset",
    )
    assert child.row_count == 1001
    assert lab.verify_version(child.version_id, dataset_id=child.dataset_id)["valid"] is True

    stored_items = list(review.iter_label_set_items(label_revision.id))
    with pytest.raises(VersionError, match="protected split"):
        ReviewDatasetBuilder(lab.store).preview(
            label_revision,
            stored_items,
            dataset_id="large-reviewed-dataset",
            parent_version_id=child.version_id,
            build_mode="append",
            target_split="canary",
        )

    lab.close()
    database.close()


@pytest.mark.parametrize("build_mode", ["filter", "replace_by_record_id", "annotate"])
def test_reviewed_outputs_can_start_a_new_dataset_in_every_build_mode(
    tmp_path: Path,
    build_mode: str,
):
    label_root = tmp_path / "labels" / build_mode
    label_root.mkdir(parents=True)
    output = {"prompt": "new prompt", "response": "reviewed response"}
    (label_root / "canonical.jsonl").write_text(
        json.dumps(output) + "\n",
        encoding="utf-8",
    )
    revision = {
        "id": f"revision-{build_mode}",
        "label_set_id": "new-label-set",
        "content_hash": f"content-{build_mode}",
        "storage_path": str(label_root),
    }
    items = [
        {
            "review_item_id": "new-item",
            "record_id": "new-record",
            "record_hash": "source-hash",
            "annotation": {"accepted": True},
            "output_records": [output],
            "excluded": False,
        }
    ]
    store = VersionStore(tmp_path / "datasets")
    preview = ReviewDatasetBuilder(store).preview(
        revision,
        items,
        dataset_id=f"new-{build_mode}",
        build_mode=build_mode,
    )
    assert preview.source_count == 0
    assert preview.output_count == 1
    assert preview.added_count == 1

    version = ReviewDatasetBuilder(store).build(
        revision,
        items,
        dataset_id=f"new-{build_mode}",
        build_mode=build_mode,
    )
    assert version.row_count == 1
    assert version.split_counts == {"train": 1}
    assert store.verify(
        version.version_id,
        dataset_id=version.dataset_id,
        verify_source=True,
    )["valid"] is True


def test_label_set_verification_detects_manifest_metadata_mutation(tmp_path: Path):
    database = RunDatabase(":memory:")
    review = ReviewLabService(database, tmp_path / "reviews")
    _, revision = review.create_schema(
        name="Acceptance",
        modality="text",
        task_type="binary",
    )
    batch = review.create_acquisition(
        [{"record_id": "one", "record": {"prompt": "question"}}]
    )
    queue = review.create_queue(
        batch.id,
        revision.id,
        policy={"mode": "one_pass"},
    )
    item = review.list_items(queue.id)[0]
    review.submit_event(
        item.id,
        "label",
        {"accepted": True},
        idempotency_key="accept-one",
        expected_active_event_id=None,
    )
    label_revision = review.publish_label_set(queue.id)
    assert review.verify_label_set(label_revision.id).valid is True

    manifest_path = Path(label_revision.storage_path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verification = review.verify_label_set(label_revision.id)
    assert verification.valid is False
    assert "manifest does not match immutable catalog metadata" in verification.errors
    assert "manifest row count does not match catalog" in verification.errors
    database.close()
