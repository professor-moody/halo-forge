from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from halo_forge.data_lab import DatasetLab
from halo_forge.evaluation_lab import EvaluationLabService
from halo_forge.run_db import RunDatabase
from halo_forge.workstation_jobs import WorkstationScheduler, WorkstationWorker
from halo_forge.workstation_jobs.resources import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
)


def _ample_capacity(path: Path) -> WorkstationCapacity:
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


def _scheduler(db: RunDatabase, worker_id: str = "test") -> WorkstationScheduler:
    return WorkstationScheduler(
        db,
        worker_id=worker_id,
        capacity_probe=_ample_capacity,
    )


def _source(lab: DatasetLab, root: Path):
    path = root / "records.jsonl"
    path.write_text(
        json.dumps({"prompt": "one", "response": "answer"}) + "\n",
        encoding="utf-8",
    )
    return lab.add_source(
        {"kind": "local", "path": str(path), "canonical_kind": "sft"},
        dataset_id="dataset",
    )


def test_scheduler_owned_dataset_job_runs_once_and_refreshes_other_instances(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = _scheduler(db)
    root = tmp_path / "datasets"
    dashboard = DatasetLab(root, scheduler=scheduler)
    source = _source(dashboard, tmp_path)

    job = dashboard.start_job("profile", {"source_id": source.id})
    assert job.status == "queued"
    assert job.work_item_id
    work = db.get_work_item(job.work_item_id)
    assert work is not None
    assert work.domain_kind == "dataset_job"
    assert work.domain_id == job.id
    assert work.launch_spec["handler"] == "dataset_lab.run_queued"

    observer = DatasetLab(root)
    assert observer.get_job(job.id).status == "queued"
    terminal = WorkstationWorker(scheduler, heartbeat_interval=0.05).run_once()
    assert terminal is not None and terminal.status == "completed"
    assert dashboard.get_job(job.id).status == "succeeded"
    assert observer.get_job(job.id).status == "succeeded"

    observer.close()
    dashboard.close()
    db.close()


def test_scheduler_owned_evaluation_publishes_verified_evidence(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = _scheduler(db)
    service = EvaluationLabService(
        db,
        tmp_path / "evaluations",
        scheduler=scheduler,
    )
    _, revision = service.create_suite(
        name="development",
        purpose="development",
        items=[{"id": "one", "record_id": "record-one", "expected": "ok"}],
        primary_metric="score",
        direction="maximize",
    )
    assert revision is not None
    launched = service.launch_evaluation(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "model-at-revision"},
        request={"scores": {"one": 1.0}, "outputs": {"one": "ok"}},
    )

    work_id = launched.evaluation.work_item_id
    assert work_id
    work = db.get_work_item(work_id)
    assert work is not None
    assert work.domain_kind == "evaluation"
    assert work.domain_id == launched.evaluation.id

    terminal = WorkstationWorker(scheduler, heartbeat_interval=0.05).run_once()
    assert terminal is not None and terminal.status == "completed"
    completed = db.get_evaluation(launched.evaluation.id)
    assert completed is not None and completed.status == "completed"
    assert completed.artifact_path and Path(completed.artifact_path).is_dir()
    assert db.count_evaluation_samples(completed.id) == 1
    assert terminal.result["metrics"]["score"] == 1.0

    service.shutdown()
    db.close()


def test_dataset_and_evaluation_jobs_share_the_global_heavy_lease(tmp_path):
    path = tmp_path / "runs.db"
    first_db = RunDatabase(str(path))
    second_db = RunDatabase(str(path))
    first = _scheduler(first_db, "first")
    second = _scheduler(second_db, "second")

    dataset_lab = DatasetLab(tmp_path / "datasets", scheduler=first)
    source = _source(dataset_lab, tmp_path)
    data_job = dataset_lab.start_job("profile", {"source_id": source.id})

    evaluations = EvaluationLabService(
        first_db,
        tmp_path / "evaluations",
        scheduler=first,
    )
    _, revision = evaluations.create_suite(
        name="suite",
        items=[{"id": "one", "expected": "ok"}],
    )
    assert revision is not None
    evaluation = evaluations.launch_evaluation(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "model"},
        request={"scores": {"one": 1.0}},
    ).evaluation

    assert data_job.work_item_id and evaluation.work_item_id
    assert first_db.get_work_item(data_job.work_item_id).resource_class == "accelerator"  # type: ignore[union-attr]
    assert first_db.get_work_item(evaluation.work_item_id).resource_class == "accelerator"  # type: ignore[union-attr]
    claimed = first.claim()
    assert claimed is not None
    assert second.claim() is None

    first.cancel(claimed.id)
    first.complete(claimed)
    dataset_lab.close()
    evaluations.shutdown()
    first_db.close()
    second_db.close()


def test_sqlite_domain_job_helpers_round_trip_work_item_links(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    dataset_job = db.create_dataset_job(
        job_type="profile",
        work_item_id="dataset-work",
    )
    assert dataset_job.work_item_id == "dataset-work"
    assert (
        db.update_dataset_job(dataset_job.id, work_item_id="dataset-work-2").work_item_id
        == "dataset-work-2"
    )  # type: ignore[union-attr]

    suite = db.create_benchmark_suite(name="suite")
    revision = db.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="hash",
        items=[{"id": "one"}],
        primary_metric="score",
        direction="maximize",
    )
    evaluation = db.create_evaluation(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        adapter_version="1",
        subject_type="model",
        subject_ref="model",
        subject_hash="subject-hash",
        reuse_key="reuse",
        request={},
        work_item_id="evaluation-work",
    )
    assert evaluation.work_item_id == "evaluation-work"
    assert (
        db.update_evaluation(evaluation.id, work_item_id="evaluation-work-2").work_item_id
        == "evaluation-work-2"
    )  # type: ignore[union-attr]
    db.close()


def test_v6_catalog_migration_adds_domain_work_item_link_columns(tmp_path):
    path = tmp_path / "legacy-v6.db"
    RunDatabase(str(path)).close()
    connection = sqlite3.connect(path)
    connection.execute("ALTER TABLE dataset_jobs DROP COLUMN work_item_id")
    connection.execute("ALTER TABLE evaluations DROP COLUMN work_item_id")
    connection.execute("UPDATE schema_meta SET value = '6' WHERE key = 'schema_version'")
    connection.commit()
    connection.close()

    migrated = RunDatabase(str(path))
    dataset_columns = {row[1] for row in migrated._conn.execute("PRAGMA table_info(dataset_jobs)")}
    evaluation_columns = {
        row[1] for row in migrated._conn.execute("PRAGMA table_info(evaluations)")
    }
    assert "work_item_id" in dataset_columns
    assert "work_item_id" in evaluation_columns
    migrated.close()


def test_activity_center_cancel_and_retry_stay_in_sync_with_domain_jobs(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = _scheduler(db)
    lab = DatasetLab(tmp_path / "datasets", scheduler=scheduler)
    source = _source(lab, tmp_path)
    job = lab.start_job("profile", {"source_id": source.id})
    assert job.work_item_id

    cancelled = scheduler.cancel(job.work_item_id)
    assert cancelled is not None and cancelled.status == "cancelled"
    assert lab.get_job(job.id).status == "cancelled"
    retried = scheduler.retry(job.work_item_id, reason="operator reviewed failure")
    assert retried is not None and retried.status == "queued"
    assert lab.get_job(job.id).status == "queued"

    evaluations = EvaluationLabService(
        db,
        tmp_path / "evaluations",
        scheduler=scheduler,
    )
    _, revision = evaluations.create_suite(
        name="activity-suite",
        items=[{"id": "one", "expected": "ok"}],
    )
    assert revision is not None
    evaluation = evaluations.launch_evaluation(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "model"},
        request={"scores": {"one": 1.0}},
    ).evaluation
    assert evaluation.work_item_id
    scheduler.cancel(evaluation.work_item_id)
    assert db.get_evaluation(evaluation.id).status == "cancelled"  # type: ignore[union-attr]
    scheduler.retry(evaluation.work_item_id, reason="operator reviewed failure")
    assert db.get_evaluation(evaluation.id).status == "queued"  # type: ignore[union-attr]

    evaluations.shutdown()
    lab.close()
    db.close()
