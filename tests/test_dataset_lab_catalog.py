"""Dataset Lab SQLite catalog and immutable-version contracts."""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def db():
    from halo_forge.run_db.db import RunDatabase

    value = RunDatabase(":memory:")
    yield value
    value.close()


def _dataset(db):
    return db.create_dataset(
        dataset_id="ds-1",
        name="Local SFT",
        modality="text",
        canonical_schema="sft",
    )


def test_current_schema_creates_dataset_lab_tables_and_updates_old_meta(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_meta VALUES ('schema_version', '2')")
    conn.commit()
    conn.close()

    from halo_forge.run_db.db import RunDatabase
    from halo_forge.run_db.schema import SCHEMA_VERSION

    migrated = RunDatabase(str(path))
    tables = {
        row["name"]
        for row in migrated._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {
        "datasets",
        "dataset_sources",
        "dataset_versions",
        "dataset_version_parents",
        "dataset_jobs",
        "run_datasets",
    }.issubset(tables)
    version = migrated._conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    ).fetchone()["value"]
    assert version == str(SCHEMA_VERSION) == "23"
    migrated.close()


def test_dataset_source_version_and_job_roundtrip(db, tmp_path):
    dataset = _dataset(db)
    source = db.create_dataset_source(
        dataset_id=dataset.id,
        source_id="src-1",
        kind="local",
        uri=str(tmp_path / "data.jsonl"),
        fingerprint="abc",
        row_count=4,
        metadata={"assets": []},
    )
    version = db.create_dataset_version(
        dataset_id=dataset.id,
        version_id="v1",
        source_id=source.id,
        recipe_hash="recipe-1",
        recipe={"steps": [{"kind": "split"}]},
        storage_path=str(tmp_path / "v1"),
        status="completed",
        content_hash="content-1",
        row_count=4,
        split_counts={"train": 3, "test": 1},
        statistics={"row_count": 4},
        source_fingerprints={source.id: source.fingerprint},
    )
    job = db.create_dataset_job(
        job_id="job-1",
        dataset_id=dataset.id,
        version_id=version.id,
        job_type="build",
        request={"recipe": version.recipe},
    )

    assert db.get_dataset(dataset.id).latest_version_id == version.id
    assert db.list_dataset_sources(dataset.id)[0].metadata == {"assets": []}
    assert db.get_dataset_version(version.id).split_counts == {"train": 3, "test": 1}
    assert db.get_dataset_job(job.id).request["recipe"]["steps"][0]["kind"] == "split"


def test_completed_version_is_immutable_and_duplicate_build_reuses_identity(db, tmp_path):
    dataset = _dataset(db)
    first = db.create_dataset_version(
        dataset_id=dataset.id,
        version_id="v1",
        recipe_hash="r",
        recipe={"steps": []},
        storage_path=str(tmp_path / "v1"),
        status="completed",
        content_hash="c",
    )
    with pytest.raises(ValueError, match="immutable"):
        db.update_dataset_version(first.id, row_count=99)
    duplicate = db.create_dataset_version(
        dataset_id=dataset.id,
        version_id="v2",
        recipe_hash="r",
        recipe={"steps": []},
        storage_path=str(tmp_path / "v2"),
        status="completed",
        content_hash="c",
    )
    assert duplicate.id == first.id


def test_version_parents_support_weighted_mixtures(db, tmp_path):
    dataset = _dataset(db)
    for version_id in ("parent-a", "parent-b"):
        db.create_dataset_version(
            dataset_id=dataset.id,
            version_id=version_id,
            recipe_hash=version_id,
            recipe={"steps": []},
            storage_path=str(tmp_path / version_id),
            status="completed",
            content_hash=version_id,
        )
    child = db.create_dataset_version(
        dataset_id=dataset.id,
        version_id="mixture",
        recipe_hash="mix",
        recipe={"steps": [{"kind": "mix"}]},
        storage_path=str(tmp_path / "mixture"),
        parent_version_id="parent-a",
        parent_versions=[
            {"parent_version_id": "parent-a", "role": "mixture", "weight": 0.25},
            {"parent_version_id": "parent-b", "role": "mixture", "weight": 0.75},
        ],
    )
    body = child.to_dict()
    assert body["parent_version_ids"] == ["parent-a", "parent-b"]
    assert {parent["weight"] for parent in body["parents"]} == {0.25, 0.75}


def test_jobs_cancel_retry_and_restart_interruption(tmp_path):
    from halo_forge.run_db.db import RunDatabase

    path = tmp_path / "jobs.db"
    first = RunDatabase(str(path))
    _dataset(first)
    first.create_dataset_job(
        job_id="running",
        dataset_id="ds-1",
        job_type="build",
        status="running",
        stage="map",
        request={"recipe": {"steps": []}},
    )
    first.create_dataset_job(
        job_id="queued",
        dataset_id="ds-1",
        job_type="build",
        status="queued",
        request={"recipe": {"steps": []}},
    )
    first.close()

    reopened = RunDatabase(str(path))
    # SQLite is only the searchable mirror; opening a reader must not
    # interrupt work owned by a live DatasetLab process.
    assert reopened.get_dataset_job("running").status == "running"
    assert reopened.get_dataset_job("queued").status == "queued"
    reopened.update_dataset_job("running", status="interrupted", stage="interrupted")
    retried = reopened.retry_dataset_job("running")
    assert retried.status == "queued"
    assert retried.request["recipe"] == {"steps": []}
    cancelled = reopened.cancel_dataset_job("running")
    assert cancelled.status == "cancelled"
    assert cancelled.cancel_requested is True
    reopened.close()


def test_run_dataset_attachment_does_not_require_run_to_be_indexed(db, tmp_path):
    dataset = _dataset(db)
    db.create_dataset_version(
        dataset_id=dataset.id,
        version_id="v1",
        recipe_hash="r",
        recipe={"steps": []},
        storage_path=str(tmp_path / "v1"),
    )
    attached = db.attach_run_dataset(
        run_id="async-job-not-indexed-yet", dataset_version_id="v1", split="train"
    )
    assert attached.to_dict()["dataset_version_id"] == "v1"
    assert db.list_run_datasets("async-job-not-indexed-yet")[0].split == "train"


def test_public_service_register_preview_build_and_version_roundtrip(db, tmp_path):
    from halo_forge.data_lab import DatasetLab
    from halo_forge.public_api.service import PublicApiService

    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        "\n".join(
            [
                '{"question":"one","answer":"1"}',
                '{"question":"two","answer":"2"}',
                '{"question":"three","answer":"3"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    engine = DatasetLab(tmp_path / "lab")
    service = PublicApiService(
        database=db,
        dataset_lab=engine,
        dataset_storage_root=tmp_path / "lab",
        base_path=tmp_path,
    )
    created = service.create_dataset(
        {
            "name": "numbers",
            "canonical_schema": "sft",
            "source": {"kind": "local", "uri": str(source_path)},
        }
    )
    assert created["name"] == "numbers"
    assert created["sources"][0]["row_count"] == 3
    assert service.preview_dataset(created["id"], limit=2)["total"] == 3
    assert service.dataset_statistics(created["id"])["row_count"] == 3

    job = service.build_dataset(
        created["id"],
        {
            "recipe": {
                "seed": 7,
                "steps": [
                    {
                        "kind": "map",
                        "schema": "sft",
                        "fields": {"prompt": "question", "response": "answer"},
                    },
                    {"kind": "split", "ratios": {"train": 0.67, "test": 0.33}},
                ],
            }
        },
    )
    engine.job_manager.wait(job["id"], timeout=5)
    completed = service.get_dataset_job(job["id"])
    assert completed["status"] == "completed"
    assert completed["version_id"]
    version = service.get_dataset_version(completed["version_id"])
    assert version["status"] == "completed"
    assert version["split_counts"] == {"train": 2, "test": 1}
    assert service.preview_dataset_version(version["id"], split="train")["total"] == 2
    engine.close()


def test_public_service_detects_source_change_and_requires_refresh(db, tmp_path):
    from halo_forge.data_lab import DatasetLab
    from halo_forge.public_api.service import PublicApiService

    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"prompt":"p","response":"r"}\n', encoding="utf-8")
    engine = DatasetLab(tmp_path / "lab")
    service = PublicApiService(
        database=db, dataset_lab=engine, dataset_storage_root=tmp_path / "lab"
    )
    created = service.create_dataset(
        {"canonical_schema": "sft", "source": {"kind": "local", "uri": str(source_path)}}
    )
    source_path.write_text('{"prompt":"changed","response":"r"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        service.preview_dataset(created["id"])
    refreshed = service.refresh_dataset_source(created["id"])
    assert refreshed["refreshed"] is True
    assert service.preview_dataset(created["id"])["items"][0]["prompt"] == "changed"
    engine.close()


def test_managed_training_resolution_checks_schema_split_and_identity(db, tmp_path):
    from halo_forge.public_api.service import PublicApiService

    dataset = _dataset(db)
    version_dir = tmp_path / "v1"
    (version_dir / "splits").mkdir(parents=True)
    (version_dir / "splits" / "train.jsonl").write_text(
        '{"prompt":"p","response":"r"}\n', encoding="utf-8"
    )
    db.create_dataset_version(
        dataset_id=dataset.id,
        version_id="v1",
        recipe_hash="recipe-hash",
        recipe={"steps": []},
        storage_path=str(version_dir),
        status="completed",
        content_hash="content-hash",
        split_counts={"train": 1},
        source_fingerprints={"src": "fingerprint"},
    )

    class Engine:
        def verify_version(self, *args, **kwargs):
            return {"valid": True}

        def get_version(self, *args, **kwargs):
            return {"split_paths": {"train": str(version_dir / "splits" / "train.jsonl")}}

    service = PublicApiService(database=db, dataset_lab=Engine())
    assert [item["id"] for item in service.list_training_dataset_versions("sft")] == ["v1"]
    assert service.list_training_dataset_versions("dpo") == []
    resolved = service._prepare_managed_dataset_payload(
        {"mode": "sft", "model": "m", "output_dir": "o", "dataset_version_id": "v1"}
    )
    assert resolved["dataset"] == str((version_dir / "splits" / "train.jsonl").resolve())
    metadata = service._dataset_version_metadata(resolved)
    assert metadata["content_hash"] == "content-hash"
    assert metadata["recipe_hash"] == "recipe-hash"
    with pytest.raises(ValueError, match="incompatible"):
        service._prepare_managed_dataset_payload({"mode": "dpo", "dataset_version_id": "v1"})
