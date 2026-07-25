from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from halo_forge.public_api.service import PublicApiService
from halo_forge.review_lab import ReviewLabService
from halo_forge.review_lab.acquisition import plan_acquisition
from halo_forge.review_lab.acquisition_storage import (
    INGESTION_SOURCE_HASH_FIELD,
    AcquisitionManifestStore,
    AcquisitionRecordSpool,
)
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


def test_durable_raw_records_are_enveloped_without_leaking_recovery_metadata(tmp_path: Path):
    database = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(database, capacity_probe=lambda _: _capacity(tmp_path))

    class Embeddings:
        def embed_envelopes(self, envelopes, *, embedding_revision):
            assert [value["record"]["prompt"] for value in envelopes] == ["one", "two"]
            assert all(INGESTION_SOURCE_HASH_FIELD not in value["record"] for value in envelopes)
            return [
                {
                    **dict(value),
                    "evidence": {
                        "embedding": [float(index), 1.0],
                        "embedding_revision": embedding_revision,
                    },
                }
                for index, value in enumerate(envelopes)
            ]

    api = PublicApiService(
        database=database,
        review_storage_root=tmp_path / "reviews",
        workstation_scheduler=scheduler,
        review_embedding_engine=Embeddings(),
    )
    payload = {
        "records": [
            {"record_id": "raw-1", "prompt": "one"},
            {"record_id": "raw-2", "prompt": "two"},
        ],
        "strategies": [
            {
                "kind": "diversity",
                "quota": 2,
                "options": {"embedding_revision": "text:model@commit"},
            }
        ],
    }
    queued = api.create_acquisition_batch(payload)
    ready = api.resolve_acquisition_batch(
        queued["id"], payload, work_item_id=queued["work_item_id"]
    )
    assert ready["status"] == "ready"
    rows = api.list_acquisition_candidates(ready["id"], limit=10)["items"]
    assert [row["record"]["prompt"] for row in rows] == ["one", "two"]
    assert all(INGESTION_SOURCE_HASH_FIELD not in row["record"] for row in rows)
    database.close()


def test_media_asset_index_is_reduced_to_each_records_references():
    index = {
        "a.png": "/assets/a.png",
        "b.png": "/assets/b.png",
        "clip.wav": "/assets/clip.wav",
        "unused.wav": "/assets/unused.wav",
    }
    assert PublicApiService._review_record_asset_paths(
        {"record": {"image": "a.png", "audio": {"path": "clip.wav"}}}, index
    ) == {
        "a.png": "/assets/a.png",
        "clip.wav": "/assets/clip.wav",
    }


def test_planning_prefix_and_manifest_staging_honor_prepublication_cancel(tmp_path: Path):
    calls = 0

    def cancel_planning() -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RuntimeError("cancelled")

    source = (
        {"record_id": f"record-{index}", "record": {"prompt": str(index)}} for index in range(1_000)
    )
    with pytest.raises(RuntimeError, match="cancelled"):
        plan_acquisition(source, check_cancelled=cancel_planning)

    spool = AcquisitionRecordSpool(tmp_path / "reviews", "prefix-work")
    spool.append([{"record": {"prompt": "persisted"}}])
    with pytest.raises(RuntimeError, match="cancelled"):
        spool.resume_after_verified_prefix(
            [{"record": {"prompt": "persisted"}}],
            check_cancelled=lambda: (_ for _ in ()).throw(RuntimeError("cancelled")),
        )

    plan = plan_acquisition(
        [
            {"record_id": f"publish-{index}", "record": {"prompt": str(index)}}
            for index in range(300)
        ]
    )
    publication_calls = 0

    def cancel_publication() -> None:
        nonlocal publication_calls
        publication_calls += 1
        if publication_calls >= 3:
            raise RuntimeError("cancelled")

    store = AcquisitionManifestStore(tmp_path / "manifest-reviews")
    with pytest.raises(RuntimeError, match="cancelled"):
        store.publish("pre-boundary", plan, check_cancelled=cancel_publication)
    assert not store.path_for("pre-boundary").exists()
    assert not list(store.root.glob(".stage-*"))


def test_cancel_after_acquisition_publication_completes_truthfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database = RunDatabase(str(tmp_path / "late.db"))
    scheduler = WorkstationScheduler(database, capacity_probe=lambda _: _capacity(tmp_path))
    source = tmp_path / "records.jsonl"
    source.write_text(json.dumps({"prompt": "one"}) + "\n", encoding="utf-8")
    api = PublicApiService(
        database=database,
        review_storage_root=tmp_path / "reviews",
        dataset_storage_root=tmp_path / "datasets",
        evaluation_storage_root=tmp_path / "evaluations",
        artifact_storage_root=tmp_path / "artifacts",
        workstation_scheduler=scheduler,
    )
    accepted = api.create_acquisition_batch({"sources": [{"kind": "jsonl", "ref": str(source)}]})
    original = ReviewLabService.create_acquisition

    def publish_then_cancel(service, *args, **kwargs):
        result = original(service, *args, **kwargs)
        scheduler.cancel(accepted["work_item_id"])
        return result

    monkeypatch.setattr(ReviewLabService, "create_acquisition", publish_then_cancel)
    terminal = WorkstationWorker(scheduler, heartbeat_interval=0.05).run_once()
    assert terminal is not None and terminal.status == "completed"
    assert terminal.cancel_requested is False
    assert api.get_acquisition_batch(accepted["id"])["status"] == "ready"
    events = LabV4Catalog(database).list_events(work_item_id=accepted["work_item_id"])
    assert events[-1].to_dict()["payload"]["late_cancel_ignored"] is True
    database.close()
