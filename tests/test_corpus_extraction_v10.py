from __future__ import annotations

import json
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from halo_forge.corpus_lab import (
    CorpusDocument,
    CorpusBundleStore,
    CorpusExtractionConfig,
    CorpusExtractionService,
    extract_source,
)
from halo_forge.data_lab import RecipeRunner
from halo_forge.own_data.registry import TRAINING_SCENARIOS
from halo_forge.public_api.service import PublicApiService
from halo_forge.run_db import RunDatabase
from halo_forge.run_db.schema import SCHEMA_VERSION
from halo_forge.workstation_jobs import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
    WorkstationScheduler,
    WorkstationWorker,
)


def _write_docx(path: Path, text: str, *, title: str = "Fixture") -> None:
    document = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body>
</w:document>"""
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties
 xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>{title}</dc:title>
</cp:coreProperties>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("docProps/core.xml", core)


def _write_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, value in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(value)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n" f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(payload))


def test_v12_to_v13_migration_is_additive_and_indexes_extraction_items(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-v12.db"
    legacy = RunDatabase(str(path))
    dataset = legacy.create_dataset(
        name="Preserved corpus", modality="text", canonical_schema="sft"
    )
    legacy.close()

    connection = sqlite3.connect(path)
    for trigger in (
        "immutable_completed_document_extractions_update",
        "immutable_completed_document_extractions_delete",
        "immutable_document_extraction_items_update",
        "immutable_document_extraction_items_delete",
        "sealed_document_extraction_items_insert",
    ):
        connection.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    connection.execute("DROP TABLE IF EXISTS document_extraction_items")
    connection.execute("DROP TABLE IF EXISTS document_extractions")
    connection.execute("UPDATE schema_meta SET value='12' WHERE key='schema_version'")
    connection.commit()
    connection.close()

    migrated = RunDatabase(str(path))
    try:
        assert SCHEMA_VERSION == 23
        assert migrated.get_dataset(dataset.id).name == "Preserved corpus"
        assert (
            migrated._conn.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()[0]
                    == "23"
        )
        tables = {
            row[0]
            for row in migrated._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"document_extractions", "document_extraction_items"} <= tables
        indexes = {
            row[1] for row in migrated._conn.execute("PRAGMA index_list(document_extraction_items)")
        }
        assert {
            "idx_document_extraction_items_document",
            "idx_document_extraction_items_content",
            "idx_document_extraction_items_status",
            "idx_document_extraction_items_source_uri",
        } <= indexes
    finally:
        migrated.close()


def test_extract_source_normalizes_text_html_docx_and_structured_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "plain.txt").write_text("Plain corpus text.\n", encoding="utf-8")
    (source / "notes.md").write_text("# Markdown title\n\nMarkdown corpus text.", encoding="utf-8")
    (source / "page.html").write_text(
        """
        <html><head><title>HTML title</title><script>not visible</script></head>
        <body><h1>Visible heading</h1><p>Visible body.</p>
        <div hidden>also not visible</div></body></html>
        """,
        encoding="utf-8",
    )
    _write_docx(source / "report.docx", "DOCX corpus text.")
    (source / "rows.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_id": "row-1",
                        "body": "Structured corpus text.",
                        "timestamp": "2026-07-16T00:00:00Z",
                    }
                ),
                json.dumps({"record_id": "row-2", "body": ""}),
                "{not-json}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = CorpusExtractionConfig.from_value(
        {
            "text_columns": ["body"],
            "id_column": "record_id",
            "metadata_columns": ["timestamp"],
        }
    )

    first = extract_source(source, root=tmp_path / "corpus", config=config)
    assert first["extraction"]["id"] == first.extraction_id
    assert len(first["records"]) == len(first.documents)
    texts = [document.text for document in first.documents]
    assert any("Plain corpus text." in value for value in texts)
    assert any("Markdown corpus text." in value for value in texts)
    html_text = next(value for value in texts if "Visible heading" in value)
    assert "Visible body." in html_text
    assert "not visible" not in html_text
    assert any("DOCX corpus text." in value for value in texts)
    structured = next(
        document for document in first.documents if document.text == "Structured corpus text."
    )
    canonical = structured.to_dict()
    assert canonical["document_id"] == structured.id
    assert canonical["document_hash"] == structured.content_hash
    assert canonical["source_ref"] == "row-1"
    assert canonical["source_spans"][0]["row"] == 0
    assert canonical["timestamp"] == "2026-07-16T00:00:00Z"
    assert {
        "document_id",
        "document_hash",
        "text",
        "title",
        "source_ref",
        "source_spans",
        "timestamp",
        "metadata",
    } <= canonical.keys()
    assert {failure.error_code for failure in first.quarantine} == {
        "invalid_json",
        "structured_row_no_text",
    }

    verification = CorpusBundleStore(tmp_path / "corpus").verify(first.bundle.content_hash)
    assert verification.valid is True
    second = extract_source(source, root=tmp_path / "corpus", config=config)
    assert second.bundle.content_hash == first.bundle.content_hash
    assert second.bundle.path == first.bundle.path
    assert second.bundle.reused is True
    assert [value.id for value in second.documents] == [value.id for value in first.documents]


def test_text_layer_pdf_is_extracted_by_page(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    source = tmp_path / "paper.pdf"
    _write_pdf(source, "PDF corpus text.")
    result = extract_source(source, root=tmp_path / "corpus")
    assert [document.text for document in result.documents] == ["PDF corpus text."]
    assert result.documents[0].locator == {"page": 1, "page_count": 1}
    assert result.documents[0].media_type == "application/pdf"
    assert result.quarantine == ()


def test_corpus_default_split_groups_pdf_pages_but_not_structured_rows() -> None:
    common = {
        "source_uri": "/source/paper.pdf",
        "source_kind": "pdf",
        "media_type": "application/pdf",
        "source_fingerprint": "source-fingerprint",
        "relative_path": "paper.pdf",
    }
    pdf_pages = [
        CorpusDocument.build(
            **common,
            text=f"Unique PDF page text {page}.",
            ordinal=page - 1,
            locator={"page": page, "page_count": 2},
            provenance={"page_documents": True},
        ).to_dict()
        for page in (1, 2)
    ]
    structured = [
        CorpusDocument.build(
            text=f"Unique structured row text {row}.",
            source_uri="/source/rows.jsonl",
            source_kind="structured",
            media_type="application/x-ndjson",
            source_fingerprint="source-fingerprint",
            ordinal=row + 2,
            relative_path="rows.jsonl",
            locator={"row": row},
            provenance={"structured_format": "jsonl"},
        ).to_dict()
        for row in range(10)
    ]
    assert {value["source_ref"] for value in pdf_pages} == {"paper.pdf"}
    assert len({value["source_ref"] for value in structured}) == len(structured)

    recipe = TRAINING_SCENARIOS.get("corpus-adaptation").default_recipe
    result = RecipeRunner().run([*pdf_pages, *structured], recipe)
    page_splits = {
        split
        for split, records in result.splits.items()
        if any(record["source_ref"] == "paper.pdf" for record in records)
    }
    assert len(page_splits) == 1


def test_public_document_preview_uses_one_bounded_combined_page(
    tmp_path: Path,
) -> None:
    database = RunDatabase(str(tmp_path / "runs.db"))
    source = tmp_path / "mixed"
    source.mkdir()
    (source / "good.txt").write_text("Retained text.", encoding="utf-8")
    (source / "broken.docx").write_bytes(b"not-a-docx")
    extraction = CorpusExtractionService(
        database, root=tmp_path / "corpus"
    )
    public = PublicApiService(
        database=database,
        corpus_extraction=extraction,
        dataset_storage_root=tmp_path / "datasets",
    )
    try:
        launched = extraction.launch(source)
        extraction_id = launched["extraction"]["id"]
        first = public.preview_document_extraction(
            extraction_id, limit=1, offset=0
        )
        second = public.preview_document_extraction(
            extraction_id, limit=1, offset=1
        )
        assert first["total"] == second["total"] == 2
        assert len(first["items"]) == len(second["items"]) == 1
        assert first["items"][0]["text"] == "Retained text."
        assert second["items"][0]["error_code"] == "invalid_docx"
    finally:
        database.close()


def test_failures_are_quarantined_without_discarding_good_documents(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mixed"
    source.mkdir()
    (source / "good.txt").write_text("Retained text.", encoding="utf-8")
    (source / "broken.docx").write_bytes(b"not-a-zip")
    result = extract_source(source, root=tmp_path / "corpus")
    assert [document.text for document in result.documents] == ["Retained text."]
    assert len(result.quarantine) == 1
    assert result.quarantine[0].error_code == "invalid_docx"
    manifest = CorpusBundleStore(tmp_path / "corpus").load_manifest(result.bundle.content_hash)
    assert manifest["document_count"] == 1
    assert manifest["quarantined_count"] == 1
    assert set(manifest["payload_checksums"]) == {
        "documents.jsonl",
        "quarantine.jsonl",
    }


def test_service_catalog_reuse_preview_verify_cancel_retry_and_immutability(
    tmp_path: Path,
) -> None:
    database = RunDatabase(str(tmp_path / "runs.db"))
    service = CorpusExtractionService(database, root=tmp_path / "corpus")
    source = tmp_path / "source.txt"
    source.write_text("Service corpus text.", encoding="utf-8")
    try:
        first = service.launch(source)
        extraction_id = first["extraction"]["id"]
        stored = database.get_document_extraction(extraction_id)
        assert stored.status == "completed"
        assert stored.document_count == 1
        items = database.list_document_extraction_items(extraction_id)
        assert len(items) == 1
        assert items[0].status == "extracted"
        assert items[0].content_hash == first["records"][0]["document_hash"]
        assert service.verify(extraction_id)["valid"] is True
        preview = service.preview(extraction_id)
        assert preview["records"][0]["text"] == "Service corpus text."

        reused = service.launch(source)
        assert reused["reused"] is True
        assert reused["extraction"]["id"] == extraction_id
        with pytest.raises(ValueError, match="immutable"):
            database.update_document_extraction(extraction_id, status="failed")
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            database._conn.execute(
                """
                UPDATE document_extraction_items SET title='changed'
                WHERE extraction_id=? AND ordinal=0
                """,
                (extraction_id,),
            )

        queued_source = tmp_path / "queued.txt"
        queued_source.write_text("Queued corpus text.", encoding="utf-8")
        queued = service.launch(queued_source, synchronous=False)
        queued_id = queued["extraction"]["id"]
        assert queued["extraction"]["status"] == "queued"
        assert service.cancel(queued_id)["status"] == "cancelled"
        retried = service.retry(queued_id, synchronous=True)
        assert retried["extraction"]["status"] == "completed"
        assert retried["records"][0]["text"] == "Queued corpus text."
    finally:
        database.close()


def test_bundle_tampering_is_detected(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Checksummed text.", encoding="utf-8")
    result = extract_source(source, root=tmp_path / "corpus")
    documents = Path(result.bundle.path) / "documents.jsonl"
    documents.write_text(documents.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    verification = CorpusBundleStore(tmp_path / "corpus").verify(result.bundle.content_hash)
    assert verification.valid is False
    assert "documents.jsonl checksum mismatch" in verification.errors


def test_scheduler_worker_executes_the_transport_neutral_entrypoint(
    tmp_path: Path,
) -> None:
    database = RunDatabase(str(tmp_path / "runs.db"))
    capacity = WorkstationCapacity(
        sampled_at=datetime.now(timezone.utc),
        disk=DiskCapacity(
            path=str(tmp_path),
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
    scheduler = WorkstationScheduler(
        database,
        capacity_probe=lambda _path: capacity,
    )
    worker = WorkstationWorker(
        scheduler,
        telemetry_sampler=lambda *_args, **_kwargs: capacity,
    )
    service = CorpusExtractionService(
        database,
        root=tmp_path / "corpus",
        scheduler=scheduler,
    )
    source = tmp_path / "queued.txt"
    source.write_text("Worker corpus text.", encoding="utf-8")
    try:
        launched = service.launch(source, synchronous=False)
        work_item_id = launched["work_item_id"]
        work = database.get_work_item(work_item_id)
        assert work.launch_spec["handler"] == "corpus_lab.extract_source"
        assert work.domain_kind == "document_extraction"
        terminal = worker.run_once(work_item_id=work_item_id)
        assert terminal.status == "completed"
        extraction = database.get_document_extraction(launched["extraction"]["id"])
        assert extraction.status == "completed"
        assert service.preview(extraction.id)["records"][0]["text"] == ("Worker corpus text.")
    finally:
        database.close()
