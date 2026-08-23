from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from halo_forge.data_lab import Recipe, RecipeRunner
from halo_forge.own_data.registry import TRAINING_SCENARIOS
from halo_forge.public_api.service import PublicApiService
from halo_forge.run_db import RunDatabase
from halo_forge.workstation_jobs import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
    WorkstationScheduler,
    WorkstationWorker,
)


def _capacity(root: Path) -> WorkstationCapacity:
    return WorkstationCapacity(
        sampled_at=datetime.now(timezone.utc),
        disk=DiskCapacity(
            path=str(root),
            total_bytes=1024**4,
            used_bytes=100 * 1024**3,
            free_bytes=900 * 1024**3,
        ),
        memory=MemoryCapacity(
            total_bytes=64 * 1024**3,
            used_bytes=8 * 1024**3,
            available_bytes=56 * 1024**3,
            source="test",
        ),
    )


def test_document_recipe_removes_only_repeated_edge_boilerplate() -> None:
    records = [
        {
            "document_id": f"doc-{index}",
            "document_hash": f"hash-{index}",
            "source_ref": f"{index}.md",
            "title": f"Document {index}",
            "text": (
                "Product navigation\n\n"
                f"# Document {index}\n\n"
                f"Unique domain paragraph {index}.\n\n"
                "Copyright example"
            ),
        }
        for index in range(5)
    ]
    recipe = Recipe.from_value(
        {
            "name": "corpus-cleaning",
            "schema": "corpus",
            "seed": 42,
            "steps": [
                {"kind": "map", "schema": "corpus"},
                {
                    "kind": "document_clean",
                    "strip_boilerplate": True,
                    "preserve_headings": True,
                    "preserve_code_blocks": True,
                },
                {
                    "kind": "document_filter",
                    "quarantine_extraction_failures": True,
                    "require_visible_text": True,
                },
                {"kind": "validate", "schema": "corpus", "on_error": "quarantine"},
                {"kind": "dedup", "method": "exact", "field": "text"},
                {
                    "kind": "split",
                    "method": "grouped",
                    "group_field": "document_id",
                    "ratios": {"train": 0.8, "validation": 0.2},
                    "seed": 42,
                },
                {"kind": "contamination", "action": "report"},
            ],
        }
    )

    result = RecipeRunner().run(records, recipe)

    assert len(result.records) == 5
    assert {name: len(rows) for name, rows in result.splits.items()} == {
        "train": 4,
        "validation": 1,
    }
    assert all("Product navigation" not in row["text"] for row in result.records)
    assert all("Copyright example" not in row["text"] for row in result.records)
    assert all("# Document" in row["text"] for row in result.records)
    cleaning = next(step for step in result.provenance if step.kind == "document_clean")
    assert cleaning.details["boilerplate_lines_identified"] == 2
    assert cleaning.details["lines_removed"] == 10


def test_guided_documents_build_an_immutable_corpus_version(tmp_path: Path) -> None:
    source = tmp_path / "documents"
    source.mkdir()
    for index in range(6):
        (source / f"document-{index}.md").write_text(
            f"# Document {index}\n\nA unique reviewed domain paragraph {index}.\n",
            encoding="utf-8",
        )
    (source / "broken.docx").write_bytes(b"not-a-docx")

    database = RunDatabase(str(tmp_path / "runs.db"))
    capacity = _capacity(tmp_path)
    scheduler = WorkstationScheduler(
        database, capacity_probe=lambda _path: capacity
    )
    worker = WorkstationWorker(
        scheduler, telemetry_sampler=lambda *_args, **_kwargs: capacity
    )
    service = PublicApiService(
        database=database,
        dataset_storage_root=tmp_path / "datasets",
        dataset_import_root=tmp_path / "imports",
        workstation_scheduler=scheduler,
    )
    try:
        session = service.create_dataset_import(
            {
                "source_kind": "workstation_path",
                "source_uri": str(source),
                "scenario_revision_id": "corpus-adaptation@1",
                "name": "Reviewed manuals",
            }
        )
        launched = service.inspect_dataset_import(
            session["id"],
            {"scenario_revision_id": "corpus-adaptation@1"},
        )
        terminal = worker.run_once(work_item_id=launched["work_item_id"])
        assert terminal is not None and terminal.status == "completed"

        inspection = service.get_dataset_source_inspection(
            launched["inspection"]["id"]
        )
        assert inspection is not None
        assert inspection["extraction_summary"]["document_count"] == 7
        assert inspection["extraction_summary"]["extracted"] == 6
        assert inspection["extraction_summary"]["quarantined"] == 1
        candidate = next(
            item
            for item in inspection["schema_candidates"]
            if item["scenario_revision_id"] == "corpus-adaptation@1"
        )
        mapping_plan = {
            "version": 2,
            "scenario_revision_id": "corpus-adaptation@1",
            "confirmed": True,
            "mappings": candidate["suggested_mapping"],
        }
        semantic = service.preview_dataset_semantics(
            inspection["id"], {"mapping_plan": mapping_plan}
        )
        assert semantic["canonical_schema"] == "corpus"
        assert semantic["items"][0]["presentation"]["source_ref"]

        scenario = TRAINING_SCENARIOS.get("corpus-adaptation")
        preparation = service.preview_dataset_preparation(
            inspection["id"],
            {
                "preparation_plan": {
                    "mapping_plan": mapping_plan,
                    "recipe": scenario.default_recipe,
                }
            },
        )
        registration = service.register_inspected_dataset(
            inspection["id"],
            {
                "name": "Reviewed manuals",
                "import_id": session["id"],
                "scenario_revision_id": scenario.revision_id,
                "mapping_plan": mapping_plan,
                "preparation_plan": preparation,
            },
        )
        terminal = worker.run_once(work_item_id=registration["work_item_id"])
        assert terminal is not None and terminal.status == "completed"
        published_import = service.get_dataset_import(session["id"])
        assert published_import is not None
        dataset_id = str(published_import["published_dataset_id"])
        source_record = service.list_dataset_sources(dataset_id)["items"][0]
        assert source_record["uri"].endswith("documents.jsonl")
        assert source_record["metadata"]["guided_own_data"]["corpus_extraction"][
            "quarantined"
        ] == 1

        build = service.build_dataset(
            dataset_id, {"recipe": preparation["recipe"]}
        )
        terminal = worker.run_once(work_item_id=build["work_item_id"])
        assert terminal is not None and terminal.status == "completed"
        job = service.get_dataset_job(build["id"])
        assert job is not None and job["status"] == "completed"
        version_id = str(job["version_id"])
        version = service.get_dataset_version(version_id)
        assert version is not None
        assert version["split_counts"] == {"train": 5, "validation": 1}
        assert any(
            item["trainer_mode"] == "cpt" and item["compatible"]
            for item in version["trainer_compatibility"]
        )
        assert version["rejections"]["source_quarantined_count"] == 1
        profile = service.corpus_profile(version_id)
        assert profile["document_count"] == 6
        assert profile["extraction_failures"] == 1
        assert profile["quarantined_documents"] == 1
    finally:
        lab = getattr(service, "_dataset_lab", None)
        if lab is not None:
            lab.close()
        database.close()


def test_activity_retry_reopens_cancelled_document_extraction(
    tmp_path: Path,
) -> None:
    from halo_forge.corpus_lab import CorpusExtractionService

    source = tmp_path / "notes.txt"
    source.write_text("Retryable corpus text.", encoding="utf-8")
    database = RunDatabase(str(tmp_path / "runs.db"))
    capacity = _capacity(tmp_path)
    scheduler = WorkstationScheduler(
        database, capacity_probe=lambda _path: capacity
    )
    worker = WorkstationWorker(
        scheduler, telemetry_sampler=lambda *_args, **_kwargs: capacity
    )
    service = CorpusExtractionService(
        database, root=tmp_path / "corpus", scheduler=scheduler
    )
    try:
        launched = service.launch(source, synchronous=False)
        extraction_id = launched["extraction"]["id"]
        page = database.list_document_extractions(limit=1)
        assert len(page) == 1
        assert database.count_document_extractions() == 1
        scheduler.cancel(launched["work_item_id"])
        assert database.get_document_extraction(extraction_id).status == "cancelled"

        retried = scheduler.retry(
            launched["work_item_id"], reason="reviewed retry"
        )
        assert retried is not None and retried.status == "queued"
        assert database.get_document_extraction(extraction_id).status == "queued"
        terminal = worker.run_once(work_item_id=retried.id)
        assert terminal is not None and terminal.status == "completed"
        assert service.verify(extraction_id)["valid"] is True
    finally:
        database.close()
