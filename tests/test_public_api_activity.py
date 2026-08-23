from __future__ import annotations

from pathlib import Path

from halo_forge.public_api.activity import activity_item_view
from halo_forge.public_api.service import PublicApiService
from halo_forge.run_db import LabV4Catalog, RunDatabase
from halo_forge.workstation_jobs import WorkstationScheduler


def test_activity_item_contract_normalizes_blockers_attempts_events_logs_and_progress(
    tmp_path: Path,
) -> None:
    database = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(database, worker_id="activity-worker")
    catalog = LabV4Catalog(database)
    log_path = tmp_path / "activity.log"
    log_path.write_text("first\nsecond\n", encoding="utf-8")
    dependency = scheduler.enqueue(
        kind="training",
        launch_spec={"name": "Dependency"},
        resource_class="none",
    )
    item = scheduler.enqueue(
        kind="artifact_operation",
        launch_spec={"name": "Convert candidate"},
        resource_class="none",
        dependencies=(dependency.id,),
        max_retries=2,
        log_path=str(log_path),
    )
    item = database.update_work_item(
        item.id,
        progress={"current": 2, "total": 8, "latest_log": "publishing"},
    )
    assert item is not None
    catalog.add_event(item.id, "blocked", payload={"reason": "dependency is not complete"})

    view = activity_item_view(database, catalog, database.get_work_item(item.id))

    assert view["title"] == "Convert candidate"
    assert view["progress_percent"] == 25.0
    assert view["queue_position"] is not None
    assert view["blockers"] == [f"Waiting for {dependency.id} (queued)"]
    assert view["events"][-1]["type"] == "blocked"
    assert view["events"][-1]["message"] == "dependency is not complete"
    assert view["logs"] == ["first", "second", "publishing"]
    assert view["attempts"] == []
    assert view["max_attempts"] == 3
    assert view["next_actions"] == ["cancel"]

    inspection_work = scheduler.enqueue(
        kind="dataset_inspection",
        launch_spec={"name": "Inspect own-data source"},
        resource_class="none",
        domain_kind="dataset_inspection",
        domain_id="inspection-123",
    )
    inspection_view = activity_item_view(
        database, catalog, database.get_work_item(inspection_work.id)
    )
    assert inspection_view["action_links"] == [
        {
            "id": "open_source",
            "label": "Resume import",
            "href": "/datasets/new?inspection=inspection-123",
        }
    ]

    import_record = database.create_dataset_import(
        import_id="import-123",
        source_kind="workstation_path",
        source_uri=str(tmp_path / "manuals"),
        status="completed",
    )
    database.update_dataset_import(
        import_record.id,
        latest_inspection_id="inspection-123",
    )
    extraction_work = scheduler.enqueue(
        kind="document_extraction",
        launch_spec={
            "name": "Extract corpus",
            "import_id": import_record.id,
        },
        resource_class="none",
        domain_kind="document_extraction",
        domain_id="extraction-123",
    )
    extraction_view = activity_item_view(
        database, catalog, database.get_work_item(extraction_work.id)
    )
    assert extraction_view["action_links"] == [
        {
            "id": "open_source",
            "label": "Resume corpus import",
            "href": "/datasets/new?inspection=inspection-123",
        }
    ]

    service = PublicApiService(
        database=database,
        base_path=tmp_path,
        dataset_storage_root=tmp_path / "datasets",
        evaluation_storage_root=tmp_path / "evaluations",
        artifact_storage_root=tmp_path / "artifacts",
    )
    snapshot = service.get_activity()
    selected = next(value for value in snapshot["items"] if value["id"] == item.id)
    assert selected["blockers"] == [f"Waiting for {dependency.id} (queued)"]
    assert selected["events"][-1]["type"] == "blocked"
    assert selected["logs"][-1] == "publishing"
    assert any(event["type"] == "blocked" for event in snapshot["events"])
    database.close()
