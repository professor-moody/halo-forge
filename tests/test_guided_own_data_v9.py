"""V9 own-data contracts shared by desktop, browser, API, and CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from halo_forge.data_lab.models import adapt_record
from halo_forge.own_data import (
    DatasetImportFile,
    DatasetImportSession,
    DatasetSourceInspection,
    GuidedOwnDataService,
)
from halo_forge.own_data.inspection import infer_schema_candidates, inspect_path
from halo_forge.own_data.mapping import apply_mapping, evaluate_expression
from halo_forge.own_data.models import FieldMappingExpression, FieldMappingPlan
from halo_forge.own_data.registry import TRAINING_SCENARIOS
from halo_forge.public_api.service import PublicApiService
from halo_forge.run_db import RunDatabase


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _service(tmp_path: Path) -> tuple[GuidedOwnDataService, RunDatabase]:
    database = RunDatabase(tmp_path / "runs.db")
    return (
        GuidedOwnDataService(
            database,
            datasets_root=tmp_path / "datasets",
            imports_root=tmp_path / "imports",
            scheduler=None,
        ),
        database,
    )


def test_v11_to_v12_migration_is_additive_and_preserves_catalog_rows(tmp_path: Path):
    path = tmp_path / "legacy-v11.db"
    legacy = RunDatabase(path)
    dataset = legacy.create_dataset(
        name="Preserved dataset", modality="text", canonical_schema="sft"
    )
    with legacy._lock:
        legacy._conn.execute("DROP TABLE IF EXISTS dataset_import_inspections")
        legacy._conn.execute("DROP TABLE IF EXISTS dataset_source_inspections")
        legacy._conn.execute("DROP TABLE IF EXISTS dataset_import_files")
        legacy._conn.execute("DROP TABLE IF EXISTS dataset_imports")
        legacy._conn.execute(
            "UPDATE schema_meta SET value='11' WHERE key='schema_version'"
        )
        legacy._conn.commit()
    legacy.close()

    migrated = RunDatabase(path)
    try:
        assert migrated.get_dataset(dataset.id).name == "Preserved dataset"
        assert migrated._conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0] == "23"
        tables = {
            row[0]
            for row in migrated._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "dataset_imports",
            "dataset_import_files",
            "dataset_source_inspections",
        } <= tables
    finally:
        migrated.close()


def test_scenario_registry_is_truthful_and_media_defaults_are_grouped():
    available = {item.id for item in TRAINING_SCENARIOS.list(include_unavailable=False)}
    assert "audio-asr" in available
    assert "audio-classification" in available
    assert "audio-tts" not in available

    audio = TRAINING_SCENARIOS.get("audio-asr")
    split = next(step for step in audio.default_recipe["steps"] if step["kind"] == "split")
    assert split == {
        "kind": "split",
        "method": "grouped",
        "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "seed": 42,
        "group_field": "audio",
        "group_by_asset_hash": True,
    }
    audio_classification = TRAINING_SCENARIOS.get("audio-classification")
    classification_split = next(
        step
        for step in audio_classification.default_recipe["steps"]
        if step["kind"] == "split"
    )
    assert classification_split["method"] == "grouped"
    assert classification_split["group_field"] == "media"
    assert classification_split["group_by_asset_hash"] is True
    assert audio.task == "automatic_speech_recognition"
    assert audio.proof_budget["max_samples"] == 50
    assert "media_directory_manifest" in audio.source_layouts
    assert "media_directory_sidecar" in audio.source_layouts
    assert "paired_media_text" in audio.source_layouts
    assert "huggingface" in audio.source_layouts

    # Worked reasoning traces are SFT records. The separate verifier-guided
    # reasoning trainer consumes a prompt schema and must not be advertised.
    assert TRAINING_SCENARIOS.get("reasoning-sft").trainer_modes == ("sft",)


def test_runtime_scenario_payload_reports_each_trainer_truthfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, database = _service(tmp_path)
    original_find_spec = importlib.util.find_spec

    def installed(name: str, *args, **kwargs):
        if name in {"mlx", "mlx_lm"}:
            return object()
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr("halo_forge.own_data.service.importlib.util.find_spec", installed)
    try:
        preference = service.get_scenario("preference-pairs", backend_name="mlx")
        compatibility = {
            item["trainer_mode"]: item for item in preference["compatible_trainers"]
        }
        assert preference["available"] is True
        assert preference["trainer_modes"] == ["dpo"]
        assert preference["declared_trainer_modes"] == ["dpo", "orpo", "rm"]
        assert compatibility["dpo"]["compatible"] is True
        assert compatibility["orpo"]["compatible"] is False
        assert compatibility["orpo"]["reason"] == (
            "ORPO training is not implemented on MLX."
        )
        assert compatibility["rm"]["compatible"] is False

        audio = service.get_scenario("audio-asr", backend_name="mlx")
        assert audio["available"] is False
        assert audio["trainer_modes"] == []
        assert "not implemented on MLX" in audio["unavailable_reason"]
    finally:
        database.close()


def test_capabilities_publish_orthogonal_active_runtime_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, database = _service(tmp_path)

    def available(name: str) -> bool:
        # A complete text/VLM torch runtime with neither an audio decoder nor
        # the optional Parquet reader installed.
        return name not in {"torchaudio", "librosa", "soundfile", "pyarrow"}

    monkeypatch.setattr("halo_forge.own_data.service._module_available", available)
    try:
        payload = service.list_capabilities(backend_name="torch")
        by_id = {item["id"]: item for item in payload["items"]}

        assert by_id["backend:torch"]["kind"] == "backend"
        assert by_id["backend:torch"]["available"] is True
        assert by_id["training-method:sft"]["kind"] == "training_method"
        assert by_id["training-method:sft"]["training_method"] == "sft"
        assert by_id["training-method:sft"]["available"] is True

        audio = by_id["training-method:audio"]
        assert audio["available"] is False
        assert "torchaudio or both librosa and soundfile" in audio["reason"]
        assert by_id["model-family:whisper"]["kind"] == "model_family"
        assert by_id["model-family:whisper"]["available"] is False

        parquet = by_id["source-format:parquet"]
        assert parquet["kind"] == "source_format"
        assert parquet["available"] is False
        assert parquet["requirements"] == ["pyarrow"]
        assert "pyarrow" in parquet["reason"]

        guided_audio = service.get_scenario("audio-asr", backend_name="torch")
        assert guided_audio["trainer_modes"] == []
        assert "parquet" not in guided_audio["source_layouts"]
        assert "parquet" in guided_audio["declared_source_layouts"]
    finally:
        database.close()


@pytest.mark.parametrize(
    ("audio_modules", "expected"),
    [
        ({"torchaudio"}, True),
        ({"librosa", "soundfile"}, True),
        ({"librosa"}, False),
    ],
)
def test_audio_trainer_probe_requires_one_complete_decoder_path(
    monkeypatch: pytest.MonkeyPatch,
    audio_modules: set[str],
    expected: bool,
) -> None:
    def available(name: str) -> bool:
        if name in {"torchaudio", "librosa", "soundfile"}:
            return name in audio_modules
        return True

    monkeypatch.setattr("halo_forge.own_data.service._module_available", available)
    compatibility = GuidedOwnDataService.runtime_trainer_compatibility(
        TRAINING_SCENARIOS.get("audio-asr"), "torch"
    )[0]
    assert compatibility["compatible"] is expected
    assert compatibility["requirements"][-1] == (
        "torchaudio or librosa + soundfile"
    )
    assert len(compatibility["alternative_requirements"]) == 2
    if not expected:
        assert "torchaudio or both librosa and soundfile" in compatibility["reason"]


def test_broken_rocm_probe_hides_guided_training_with_plain_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "halo_forge.own_data.service._accelerator_runtime_probe",
        lambda _backend: (
            False,
            "The ROCm runtime crashed during a small isolated GPU check.",
        ),
    )
    monkeypatch.setattr(
        "halo_forge.own_data.service._module_available", lambda _name: True
    )
    compatibility = GuidedOwnDataService.runtime_trainer_compatibility(
        TRAINING_SCENARIOS.get("instruction-sft"), "rocm_gfx1151"
    )[0]
    assert compatibility["compatible"] is False
    assert "isolated GPU check" in compatibility["reason"]


def test_parquet_source_format_is_exposed_when_reader_is_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, database = _service(tmp_path)
    monkeypatch.setattr("halo_forge.own_data.service._module_available", lambda _name: True)
    try:
        capabilities = service.list_capabilities(backend_name="torch")
        parquet = next(
            item
            for item in capabilities["items"]
            if item["id"] == "source-format:parquet"
        )
        assert parquet["available"] is True
        scenario = service.get_scenario("instruction-sft", backend_name="torch")
        assert "parquet" in scenario["source_layouts"]
    finally:
        database.close()


def test_public_import_and_inspection_view_types_are_exported() -> None:
    imported_file = DatasetImportFile(
        id="file-1",
        import_id="import-1",
        relative_path="nested/source.jsonl",
        size_bytes=10,
        uploaded_bytes=10,
        status="verified",
        content_hash="abc",
    )
    session = DatasetImportSession(
        id="import-1",
        status="ready",
        source_kind="huggingface",
        source_uri="org/data",
        source_config="default",
        source_split="train",
        source_revision="main",
        resolved_revision="f" * 40,
        files=(imported_file,),
    )
    inspection = DatasetSourceInspection(
        id="inspection-1",
        import_id=session.id,
        status="completed",
        source_fingerprint="sha256:source",
        import_adapter_version="1",
        scenario_registry_revision="registry",
        fields=({"name": "prompt", "coverage": 1.0},),
        preview_records=({"prompt": "hello"},),
    )
    assert session.to_dict()["resolved_revision"] == "f" * 40
    assert session.to_dict()["files"][0]["status"] == "verified"
    assert inspection.to_dict()["fields"][0]["coverage"] == 1.0


def test_multi_row_inference_enforces_confidence_and_ambiguity_rules():
    high = infer_schema_candidates(
        [{"question": f"q-{index}", "answer": f"a-{index}"} for index in range(100)]
    )
    instruction = next(item for item in high if item.scenario_id == "instruction-sft")
    assert instruction.confidence == "high"
    assert instruction.required_coverage == 1.0

    medium_records = [
        {"question": f"q-{index}", **({"answer": f"a-{index}"} if index < 96 else {})}
        for index in range(100)
    ]
    medium = infer_schema_candidates(medium_records)
    instruction = next(item for item in medium if item.scenario_id == "instruction-sft")
    assert instruction.confidence == "medium"
    assert instruction.required_coverage == pytest.approx(0.96)

    ambiguous = infer_schema_candidates(
        [
            {
                "question": f"q-{index}",
                "response": f"short-{index}",
                "worked_solution": f"worked-{index}",
            }
            for index in range(100)
        ]
    )
    candidates = {
        item.scenario_id: item.confidence
        for item in ambiguous
        if item.scenario_id in {"instruction-sft", "reasoning-sft"}
    }
    assert candidates == {"instruction-sft": "medium", "reasoning-sft": "medium"}
    assert not any(item.confidence == "high" for item in ambiguous)


def test_inspection_streams_exact_coverage_and_seeded_bounded_preview(tmp_path: Path):
    source = tmp_path / "large.jsonl"
    records = [
        {"question": f"q-{index}", "answer": f"a-{index}", "ordinal": index}
        for index in range(1_250)
    ]
    _write_jsonl(source, records)

    first = inspect_path(source)
    second = inspect_path(source)
    assert first["total_records"] == 1_250
    assert first["valid_records"] == 1_250
    assert first["sample_count"] == 1_000
    assert first["preview"] == second["preview"]
    assert [row["ordinal"] for row in first["preview"][:100]] == list(range(100))
    assert len({row["ordinal"] for row in first["preview"]}) == 1_000
    coverage = {field["name"]: field["coverage"] for field in first["fields"]}
    assert coverage == {"answer": 1.0, "ordinal": 1.0, "question": 1.0}
    assert first["statistics"]["sampled"] is True


def test_json_array_inspection_is_incremental_across_read_boundaries(tmp_path: Path):
    source = tmp_path / "large-array.json"
    records = [
        {"question": f"q-{index}", "answer": "a" * 200, "ordinal": index}
        for index in range(750)
    ]
    source.write_text(json.dumps(records), encoding="utf-8")

    inspected = inspect_path(source)
    assert inspected["total_records"] == 750
    assert inspected["valid_records"] == 750
    assert inspected["invalid_records"] == 0
    assert inspected["preview"][0]["ordinal"] == 0


def test_mapping_v2_handles_conversation_nested_concat_and_media(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    image = media / "sample.jpg"
    image.write_bytes(b"fixture")
    record = {
        "context": {"instruction": "Explain"},
        "detail": "briefly",
        "turns": [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi", "tool_call_id": "call-1"},
        ],
        "file": "sample.jpg",
    }
    nested = evaluate_expression(
        record,
        FieldMappingExpression(kind="nested_path", source="context", path="instruction"),
    )
    combined = evaluate_expression(
        record,
        FieldMappingExpression(
            kind="concat", sources=("context.instruction", "detail"), separator=" — "
        ),
    )
    conversation = evaluate_expression(
        record,
        FieldMappingExpression(
            kind="conversation",
            source="turns",
            role_field="from",
            content_field="value",
        ),
    )
    resolved_media = evaluate_expression(
        record,
        FieldMappingExpression(kind="media_root", source="file", media_root=str(media)),
    )
    assert nested == "Explain"
    assert combined == "Explain — briefly"
    assert [item["role"] for item in conversation] == ["user", "assistant"]
    assert conversation[1]["tool_call_id"] == "call-1"
    assert resolved_media == str(image.resolve())

    chat = apply_mapping(
        record,
        FieldMappingPlan(
            scenario_revision_id="chat-sft@1",
            mappings={
                "messages": FieldMappingExpression(
                    kind="conversation",
                    source="turns",
                    role_field="from",
                    content_field="value",
                )
            },
            confirmed=True,
        ),
    )
    assert chat["messages"][0] == {"role": "user", "content": "Hello"}

    built_chat = adapt_record(
        record,
        "chat",
        mapping={
            "messages": {
                "kind": "conversation",
                "source": "turns",
                "role_field": "from",
                "content_field": "value",
            }
        },
    )
    assert built_chat["messages"][1]["tool_call_id"] == "call-1"

    with pytest.raises(ValueError, match="escapes"):
        evaluate_expression(
            {"file": "../outside.jpg"},
            FieldMappingExpression(kind="media_root", source="file", media_root=str(media)),
        )


def test_resumable_upload_checksums_idempotency_and_traversal(tmp_path: Path):
    service, database = _service(tmp_path)
    try:
        session = service.create_import({"source_kind": "upload", "name": "upload"})
        content = b'{"prompt":"hello","response":"world"}\n'
        digest = hashlib.sha256(content).hexdigest()
        file_record = service.create_import_file(
            session["id"],
            {
                "relative_path": "nested/data.jsonl",
                "size_bytes": len(content),
                "content_hash": digest,
            },
        )
        midpoint = len(content) // 2
        first = content[:midpoint]
        second = content[midpoint:]
        service.upload_chunk(
            session["id"],
            file_record["id"],
            first,
            start=0,
            end=midpoint - 1,
            total=len(content),
            chunk_sha256=hashlib.sha256(first).hexdigest(),
        )
        # A network retry of an already committed byte range is idempotent.
        service.upload_chunk(
            session["id"],
            file_record["id"],
            first,
            start=0,
            end=midpoint - 1,
            total=len(content),
            chunk_sha256=hashlib.sha256(first).hexdigest(),
        )
        completed = service.upload_chunk(
            session["id"],
            file_record["id"],
            second,
            start=midpoint,
            end=len(content) - 1,
            total=len(content),
            chunk_sha256=hashlib.sha256(second).hexdigest(),
        )
        assert completed["status"] == "verified"
        assert completed["content_hash"] == digest
        assert service.get_import(session["id"])["status"] == "ready"

        recreated = service.create_import_file(
            session["id"],
            {
                "relative_path": "nested/data.jsonl",
                "size_bytes": len(content),
                "content_hash": digest,
            },
        )
        assert recreated["id"] == file_record["id"]

        # Directory uploads create files sequentially. Completing one file
        # must not prevent the next relative path from joining the session.
        second_file = service.create_import_file(
            session["id"],
            {
                "relative_path": "nested/labels.txt",
                "size_bytes": 1,
                "content_hash": hashlib.sha256(b"x").hexdigest(),
            },
        )
        service.upload_chunk(
            session["id"],
            second_file["id"],
            b"x",
            start=0,
            end=0,
            total=1,
            chunk_sha256=hashlib.sha256(b"x").hexdigest(),
        )
        assert service.get_import(session["id"])["total_files"] == 2

        cancelled = service.cancel_import(session["id"])
        assert cancelled["dataset_import"]["status"] == "cancelled"
        resumed = service.retry_import(session["id"])
        assert resumed["dataset_import"]["status"] == "ready"

        unsafe = service.create_import({"source_kind": "upload", "name": "unsafe"})
        with pytest.raises(ValueError, match="traversal"):
            service.create_import_file(
                unsafe["id"], {"relative_path": "../escape.jsonl", "size_bytes": 1}
            )
    finally:
        database.close()


def test_huggingface_inspection_request_defers_materialization_to_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    service, database = _service(tmp_path)
    try:
        session = database.create_dataset_import(
            source_kind="huggingface",
            source_uri="owner/dataset",
            source_config="default",
            source_split="train",
            source_revision="main",
            resolved_revision="a" * 40,
            status="ready",
        )

        def unexpected_materialization(_import_id: str) -> Path:
            raise AssertionError("request path must not download the dataset")

        monkeypatch.setattr(service.imports, "source_path", unexpected_materialization)
        requested = service.request_inspection(session.id)
        assert requested["inspection"]["status"] == "queued"
        assert requested["work_item_id"] is None
    finally:
        database.close()


def test_huggingface_media_materialization_preserves_existing_binary_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    datasets = pytest.importorskip("datasets")

    service, database = _service(tmp_path)

    class FakeDataset:
        def __init__(self) -> None:
            self.features = {"image": datasets.Image()}

        def cast_column(self, name: str, feature):
            self.features[name] = feature
            return self

        def __iter__(self):
            yield {
                "image": {"bytes": b"\x89PNG\r\n\x1a\nfixture", "path": None},
                "caption": "A retained image",
            }

    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: FakeDataset())
    try:
        session = database.create_dataset_import(
            source_kind="huggingface",
            source_uri="owner/images",
            source_config="default",
            source_split="train",
            source_revision="main",
            resolved_revision="b" * 40,
            status="ready",
        )
        staging = service.imports.materialize_huggingface(session.id)
        manifest = staging / "data.jsonl"
        record = json.loads(manifest.read_text(encoding="utf-8"))
        asset = staging / record["image"]
        assert record["caption"] == "A retained image"
        assert asset.read_bytes() == b"\x89PNG\r\n\x1a\nfixture"
        assert asset.suffix == ".png"
    finally:
        database.close()


def test_media_inspection_reports_missing_and_unsafe_assets_before_build(tmp_path: Path):
    source = tmp_path / "media-source"
    source.mkdir()
    (source / "images").mkdir()
    (source / "images" / "present.jpg").write_bytes(b"fixture")
    _write_jsonl(
        source / "manifest.jsonl",
        [
            {"image": "images/present.jpg", "prompt": "What?", "response": "A."},
            {"image": "images/missing.jpg", "prompt": "Where?", "response": "Gone."},
            {"image": "../outside.jpg", "prompt": "Unsafe?", "response": "Yes."},
        ],
    )

    inspected = inspect_path(source)
    media = inspected["statistics"]["media_summary"]
    assert inspected["total_records"] == 3
    assert inspected["valid_records"] == 1
    assert inspected["invalid_records"] == 2
    assert media == {
        "referenced": 3,
        "verified": 1,
        "missing": 1,
        "unsafe": 1,
        "image_references": 3,
        "audio_references": 0,
    }
    assert {issue["code"] for issue in inspected["issues"]} == {
        "missing_media_asset",
        "unsafe_media_path",
    }

    with pytest.raises(Exception, match="does not exist"):
        adapt_record(
            {"image": "images/missing.jpg", "prompt": "Where?", "response": "Gone."},
            "vlm",
            mapping={
                "image": {
                    "kind": "media_root",
                    "source": "image",
                    "media_root": str(source),
                },
                "prompt": "prompt",
                "response": "response",
            },
        )


def test_media_sidecar_table_accepts_relative_filename_column(tmp_path: Path):
    source = tmp_path / "caption-table"
    source.mkdir()
    (source / "images").mkdir()
    (source / "images" / "one.jpg").write_bytes(b"one")
    (source / "images" / "two.jpg").write_bytes(b"two")
    (source / "manifest.csv").write_text(
        "filename,caption\nimages/one.jpg,First image\nimages/two.jpg,Second image\n",
        encoding="utf-8",
    )

    inspected = inspect_path(source)
    assert inspected["statistics"]["media_summary"]["verified"] == 2
    candidate = next(
        item
        for item in inspected["candidates"]
        if item["scenario_id"] == "vlm-captioning"
    )
    assert candidate["confidence"] == "medium"
    assert candidate["suggested_mapping"]["image"]["source"] == "filename"


@pytest.mark.parametrize(
    ("scenario_revision_id", "expected_media_field"),
    [("vlm-captioning@1", "image"), ("audio-asr@1", "audio")],
)
def test_multimodal_working_examples_include_real_inspectable_assets(
    tmp_path: Path, scenario_revision_id: str, expected_media_field: str
):
    service, database = _service(tmp_path)
    try:
        session = service.create_import(
            {
                "source_kind": "example",
                "scenario_revision_id": scenario_revision_id,
                "name": "Working multimodal example",
            }
        )
        assert session["total_files"] == 2
        requested = service.request_inspection(
            session["id"], scenario_revision_id=scenario_revision_id
        )
        inspected = service.execute_inspection(requested["inspection"]["id"])
        assert inspected["valid_records"] == 1
        assert inspected["invalid_records"] == 0
        assert inspected["media_summary"]["verified"] == 1
        assert expected_media_field in inspected["preview_records"][0]
    finally:
        database.close()


@pytest.mark.parametrize(
    "scenario_revision_id",
    [
        scenario.revision_id
        for scenario in TRAINING_SCENARIOS.list(include_unavailable=False)
    ],
)
def test_every_advertised_working_example_reaches_preparation(
    tmp_path: Path, scenario_revision_id: str
):
    service, database = _service(tmp_path)
    scenario = TRAINING_SCENARIOS.get(scenario_revision_id)
    try:
        session = service.create_import(
            {
                "source_kind": "example",
                "scenario_revision_id": scenario_revision_id,
                "name": f"{scenario.label} working example",
            }
        )
        requested = service.request_inspection(
            session["id"], scenario_revision_id=scenario_revision_id
        )
        inspection = service.execute_inspection(requested["inspection"]["id"])
        candidate = next(
            item
            for item in inspection["schema_candidates"]
            if item["scenario_id"] == scenario.id
        )
        mapping = {
            "version": 2,
            "scenario_revision_id": scenario_revision_id,
            "confirmed": True,
            "mappings": candidate["suggested_mapping"],
        }

        mapped = service.mapping_preview(inspection["id"], mapping)
        prepared = service.preparation_preview(
            inspection["id"], {"mapping_plan": mapping}
        )

        assert mapped["ready"] is True
        assert mapped["valid_count"] >= 1
        assert prepared["estimates"]["accepted"] >= 1
        assert prepared["estimates"]["quarantined"] == 0
        assert prepared["recipe"]["schema"] == scenario.canonical_schema
    finally:
        database.close()


def test_inspection_reuse_is_exact_and_source_drift_creates_new_evidence(tmp_path: Path):
    service, database = _service(tmp_path)
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"question": "q", "answer": "a"}])
    try:
        first_import = service.create_import(
            {"source_kind": "workstation_path", "source_uri": str(source)}
        )
        first_request = service.request_inspection(first_import["id"])
        first = service.execute_inspection(first_request["inspection"]["id"])

        second_import = service.create_import(
            {"source_kind": "workstation_path", "source_uri": str(source)}
        )
        reused = service.request_inspection(second_import["id"])
        assert reused["reused"] is True
        assert reused["inspection"]["id"] == first["id"]
        assert service.get_import(second_import["id"])["status"] == "completed"

        _write_jsonl(
            source,
            [
                {"question": "q", "answer": "a"},
                {"question": "new", "answer": "record"},
            ],
        )
        changed_import = service.create_import(
            {"source_kind": "workstation_path", "source_uri": str(source)}
        )
        changed_request = service.request_inspection(changed_import["id"])
        assert changed_request["inspection"]["id"] != first["id"]
        changed = service.execute_inspection(changed_request["inspection"]["id"])
        assert changed["row_count"] == 2
        assert service.get_inspection(first["id"])["row_count"] == 1
    finally:
        database.close()


def test_reused_inspection_registration_binds_the_active_import(tmp_path: Path):
    from halo_forge.data_lab import DatasetLab

    service, database = _service(tmp_path)
    first_source = tmp_path / "first.jsonl"
    second_source = tmp_path / "second.jsonl"
    rows = [{"question": "Which source?", "answer": "The active one."}]
    _write_jsonl(first_source, rows)
    _write_jsonl(second_source, rows)
    lab = DatasetLab(tmp_path / "datasets")
    try:
        first_import = service.create_import(
            {
                "source_kind": "workstation_path",
                "source_uri": str(first_source),
                "scenario_revision_id": "instruction-sft@1",
            }
        )
        first_request = service.request_inspection(first_import["id"])
        first_inspection = service.execute_inspection(
            first_request["inspection"]["id"]
        )

        second_import = service.create_import(
            {
                "source_kind": "workstation_path",
                "source_uri": str(second_source),
                "scenario_revision_id": "instruction-sft@1",
            }
        )
        reused = service.request_inspection(second_import["id"])
        assert reused["reused"] is True
        assert reused["inspection"]["id"] == first_inspection["id"]
        assert reused["inspection"]["import_id"] == second_import["id"]

        candidate = next(
            item
            for item in first_inspection["schema_candidates"]
            if item["scenario_id"] == "instruction-sft"
        )
        mapping = {
            "version": 2,
            "scenario_revision_id": "instruction-sft@1",
            "confirmed": True,
            "mappings": candidate["suggested_mapping"],
        }
        preparation = service.preparation_preview(
            first_inspection["id"], {"mapping_plan": mapping}
        )
        first_source.unlink()
        result = service.execute_registration(
            first_inspection["id"],
            {
                "name": "Reused evidence",
                "import_id": second_import["id"],
                "scenario_revision_id": "instruction-sft@1",
                "mapping_plan": mapping,
                "preparation_plan": preparation,
            },
            dataset_lab=lab,
            dataset_id="reused-dataset",
            source_id="reused-source",
        )
        assert result["source"]["uri"] == str(second_source.resolve())
        assert database.get_dataset_import(second_import["id"]).status == "published"
        assert database.get_dataset_import(first_import["id"]).status == "completed"
    finally:
        lab.close()
        database.close()


def test_media_content_and_missing_state_are_part_of_inspection_reuse_identity(
    tmp_path: Path,
):
    service, database = _service(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    image = tmp_path / "asset.png"
    _write_jsonl(
        manifest,
        [{"image": "asset.png", "prompt": "Describe it", "response": "A test"}],
    )
    image.write_bytes(b"first-image-bytes")
    try:
        first_import = service.create_import(
            {"source_kind": "workstation_path", "source_uri": str(manifest)}
        )
        first_request = service.request_inspection(first_import["id"])
        first = service.execute_inspection(first_request["inspection"]["id"])
        first_record = database.get_dataset_source_inspection(first["id"])
        first_identity = first_record.statistics["content_identity"]
        assert first_record.statistics["asset_fingerprints"][0]["fingerprint"]

        image.write_bytes(b"changed-image-bytes")
        changed_import = service.create_import(
            {"source_kind": "workstation_path", "source_uri": str(manifest)}
        )
        changed_request = service.request_inspection(changed_import["id"])
        assert changed_request["reused"] is False
        changed = service.execute_inspection(changed_request["inspection"]["id"])
        changed_record = database.get_dataset_source_inspection(changed["id"])
        assert changed_record.statistics["content_identity"] != first_identity

        image.unlink()
        missing_import = service.create_import(
            {"source_kind": "workstation_path", "source_uri": str(manifest)}
        )
        missing_request = service.request_inspection(missing_import["id"])
        assert missing_request["reused"] is False
        missing = service.execute_inspection(missing_request["inspection"]["id"])
        missing_record = database.get_dataset_source_inspection(missing["id"])
        assert missing_record.statistics["content_identity"] not in {
            first_identity,
            changed_record.statistics["content_identity"],
        }
        assert missing["invalid_records"] == 1
        assert missing["media_summary"]["missing"] == 1
    finally:
        database.close()


def test_confirmed_mapping_and_custom_recipe_round_trip_into_registration(tmp_path: Path):
    service, database = _service(tmp_path)
    source = tmp_path / "qa.csv"
    source.write_text("question,answer\nQ?,A.\n", encoding="utf-8")
    try:
        session = service.create_import(
            {
                "source_kind": "workstation_path",
                "source_uri": str(source),
                "scenario_revision_id": "instruction-sft@1",
            }
        )
        requested = service.request_inspection(
            session["id"], scenario_revision_id="instruction-sft@1"
        )
        inspection = service.execute_inspection(requested["inspection"]["id"])
        candidate = next(
            item
            for item in inspection["schema_candidates"]
            if item["scenario_id"] == "instruction-sft"
        )
        mapping = {
            "version": 2,
            "scenario_revision_id": "instruction-sft@1",
            "confirmed": True,
            "mappings": candidate["suggested_mapping"],
        }
        default = service.preparation_preview(inspection["id"], {"mapping_plan": mapping})
        custom_recipe = dict(default["recipe"])
        custom_recipe["steps"] = [dict(step) for step in default["recipe"]["steps"]]
        split = next(step for step in custom_recipe["steps"] if step["kind"] == "split")
        split["ratios"] = {"train": 0.7, "validation": 0.2, "test": 0.1}
        custom = service.preparation_preview(
            inspection["id"],
            {
                "preparation_plan": {
                    "mapping_plan": mapping,
                    "recipe": custom_recipe,
                }
            },
        )
        assert next(step for step in custom["recipe"]["steps"] if step["kind"] == "split")[
            "ratios"
        ] == {"train": 0.7, "validation": 0.2, "test": 0.1}

        registration = service.registration_payload(
            inspection["id"],
            {
                "name": "Questions",
                "scenario_revision_id": "instruction-sft@1",
                "mapping_plan": mapping,
                "preparation_plan": custom,
            },
        )
        assert registration["canonical_schema"] == "sft"
        assert registration["modality"] == "text"
        assert registration["preparation_plan"]["recipe"] == custom["recipe"]
        assert registration["source"]["uri"] == str(source.resolve())
    finally:
        database.close()


def test_source_refresh_targets_exact_revision_and_preserves_guided_metadata(
    tmp_path: Path,
):
    database = RunDatabase(tmp_path / "refresh.db")
    dataset = database.create_dataset(name="refresh", modality="text", canonical_schema="sft")
    first = database.create_dataset_source(
        dataset_id=dataset.id,
        source_id="source-one",
        kind="local",
        uri=str(tmp_path / "one.jsonl"),
        fingerprint="old-one",
        metadata={
            "guided_own_data": {
                "scenario_revision_id": "instruction-sft@1",
                "preparation_plan": {"recipe": {"steps": []}},
            }
        },
    )
    database.create_dataset_source(
        dataset_id=dataset.id,
        source_id="source-two",
        kind="local",
        uri=str(tmp_path / "two.jsonl"),
        fingerprint="old-two",
    )

    class RefreshEngine:
        def __init__(self) -> None:
            self.called_with = ""

        def refresh_source(self, source_id: str):
            self.called_with = source_id
            return {
                "id": "source-one-refreshed",
                "fingerprint": "new-one",
                "size_bytes": 123,
                "row_count": 4,
                "asset_fingerprints": [],
                "spec": {"kind": "local", "path": str(tmp_path / "one.jsonl")},
            }

    engine = RefreshEngine()
    service = PublicApiService(
        database=database,
        dataset_lab=engine,
        dataset_storage_root=tmp_path / "datasets",
    )
    try:
        refreshed = service.execute_dataset_source_refresh(first.id)
        assert engine.called_with == "source-one"
        assert refreshed["refreshed_from_source_id"] == "source-one"
        assert refreshed["fingerprint"] == "new-one"
        assert refreshed["metadata"]["guided_own_data"]["scenario_revision_id"] == (
            "instruction-sft@1"
        )
        assert refreshed["metadata"]["source_refresh"]["previous_fingerprint"] == "old-one"
        assert database.get_dataset_source("source-two").fingerprint == "old-two"
    finally:
        database.close()


def test_local_source_refresh_is_durable_idempotent_and_recoverable(tmp_path: Path):
    from halo_forge.workstation_jobs import WorkstationScheduler, WorkstationWorker

    source_path = tmp_path / "refresh.jsonl"
    _write_jsonl(source_path, [{"prompt": "before", "response": "one"}])
    database = RunDatabase(tmp_path / "refresh-durable.db")
    scheduler = WorkstationScheduler(
        database, process_probe=lambda _pid, _started: False
    )
    service = PublicApiService(
        database=database,
        base_path=tmp_path,
        dataset_storage_root=tmp_path / "datasets",
        dataset_import_root=tmp_path / "imports",
        workstation_scheduler=scheduler,
    )
    try:
        dataset = service.create_dataset(
            {
                "name": "Refresh me",
                "canonical_schema": "sft",
                "source": {"kind": "local", "uri": str(source_path)},
            }
        )
        original = dataset["sources"][0]
        _write_jsonl(
            source_path,
            [
                {"prompt": "before", "response": "one"},
                {"prompt": "after", "response": "two"},
            ],
        )

        first = service.refresh_dataset_source_by_id(original["id"])
        repeated = service.refresh_dataset_source_by_id(original["id"])
        assert first["work_item_id"] == repeated["work_item_id"]
        assert first["source"]["fingerprint"] == original["fingerprint"]

        claimed = scheduler.claim(work_item_id=first["work_item_id"])
        assert claimed is not None and claimed.status == "running"
        recovered = scheduler.recover_or_adopt()
        assert [item.id for item in recovered.interrupted] == [claimed.id]
        assert database.get_work_item(claimed.id).status == "needs_reconciliation"
        retried = scheduler.retry(
            claimed.id, reason="operator verified the interrupted refresh"
        )
        assert retried is not None and retried.status == "queued"

        terminal = WorkstationWorker(scheduler).run_once()
        assert terminal is not None and terminal.status == "completed", terminal
        refreshed = terminal.result["source"]
        assert refreshed["refreshed"] is True
        assert refreshed["refreshed_from_source_id"] == original["id"]
        assert refreshed["fingerprint"] != original["fingerprint"]
        sources = database.list_dataset_sources(dataset["id"])
        assert len(sources) == 2
    finally:
        lab = getattr(service, "_dataset_lab", None)
        if lab is not None:
            lab.close()
        database.close()


def test_guided_api_contract_runs_durable_inspection_and_previews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from halo_forge.auth.dependency import reset_store_for_tests
    from halo_forge.public_api import app as app_module
    from halo_forge.public_api.service import PublicApiService
    from halo_forge.workstation_jobs import WorkstationScheduler, WorkstationWorker

    source = tmp_path / "questions.csv"
    source.write_text("question,answer\nWhy?,Because.\nHow?,Carefully.\n", encoding="utf-8")
    database = RunDatabase(tmp_path / "api-runs.db")
    scheduler = WorkstationScheduler(database)
    public = PublicApiService(
        database=database,
        base_path=tmp_path,
        dataset_storage_root=tmp_path / "datasets",
        dataset_import_root=tmp_path / "imports",
        workstation_scheduler=scheduler,
    )
    public._cached_backend_name = "mps"
    monkeypatch.setenv("HALOFORGE_DISABLE_AUTO_WORKER", "1")
    monkeypatch.setattr(app_module, "PublicApiService", lambda: public)
    reset_store_for_tests(None)

    try:
        with TestClient(app_module.create_app(serve_frontend=False)) as client:
            capabilities = client.get("/api/public/interface-capabilities")
            assert capabilities.status_code == 200
            surfaces = {
                item.get("execution_surface")
                for item in capabilities.json()["items"]
                if item["kind"] == "execution_surface"
            }
            assert {"desktop", "local_browser", "remote_browser", "cli"} <= surfaces
            mac_desktop = next(
                item
                for item in capabilities.json()["items"]
                if item["id"] == "desktop-macos-arm64"
            )
            assert mac_desktop["status"] == "preview"
            assert mac_desktop["metadata"]["signed_public_artifact"] is False

            created = client.post(
                "/api/public/dataset-imports",
                json={
                    "source_kind": "workstation_path",
                    "source_uri": str(source),
                    "scenario_revision_id": "instruction-sft@1",
                },
            )
            assert created.status_code == 201, created.text
            import_id = created.json()["id"]
            queued = client.post(
                f"/api/public/dataset-imports/{import_id}/inspect",
                json={"scenario_revision_id": "instruction-sft@1"},
            )
            assert queued.status_code == 202, queued.text
            assert queued.json()["work_item_id"]

            work = WorkstationWorker(scheduler).run_once()
            assert work is not None and work.status == "completed"
            inspection_id = queued.json()["inspection"]["id"]
            inspected = client.get(f"/api/public/dataset-inspections/{inspection_id}")
            assert inspected.status_code == 200, inspected.text
            inspection = inspected.json()
            assert inspection["row_count"] == 2
            assert isinstance(inspection["preview_policy"], str)
            candidate = next(
                item
                for item in inspection["schema_candidates"]
                if item["scenario_id"] == "instruction-sft"
            )
            mapping = {
                "version": 2,
                "scenario_revision_id": "instruction-sft@1",
                "confirmed": True,
                "mappings": candidate["suggested_mapping"],
            }
            preview = client.post(
                f"/api/public/dataset-inspections/{inspection_id}/mapping-preview",
                json=mapping,
            )
            assert preview.status_code == 200, preview.text
            assert preview.json()["ready"] is True
            preparation = client.post(
                f"/api/public/dataset-inspections/{inspection_id}/preparation-preview",
                json={"mapping_plan": mapping},
            )
            assert preparation.status_code == 200, preparation.text
            assert preparation.json()["recipe"]["steps"][0]["kind"] == "map"

            conflict = client.post(
                f"/api/public/dataset-inspections/{inspection_id}/register",
                json={
                    "name": "Contradictory identity",
                    "import_id": import_id,
                    "scenario_revision_id": "chat-sft@1",
                    "mapping_plan": mapping,
                    "preparation_plan": preparation.json(),
                },
            )
            assert conflict.status_code == 409, conflict.text
            assert "do not match" in conflict.json()["detail"]

            accepted = client.post(
                f"/api/public/dataset-inspections/{inspection_id}/register",
                json={
                    "name": "Questions",
                    "import_id": import_id,
                    "scenario_revision_id": "instruction-sft@1",
                    "mapping_plan": mapping,
                    "preparation_plan": preparation.json(),
                },
            )
            assert accepted.status_code == 202, accepted.text
            assert accepted.json()["dataset"] is None
            assert accepted.json()["work_item_id"]
            registration_work = WorkstationWorker(scheduler).run_once()
            assert registration_work is not None
            assert registration_work.status == "completed", registration_work.error
            published = client.get(f"/api/public/dataset-imports/{import_id}").json()
            assert published["status"] == "published"
            assert published["published_dataset_id"]
            source.write_text(
                "question,answer\nWhy?,Because.\nHow?,Carefully.\nNew?,Record.\n",
                encoding="utf-8",
            )
            refresh = client.post(
                f"/api/public/dataset-sources/{published['published_source_id']}/refresh",
                json={},
            )
            assert refresh.status_code == 202, refresh.text
            assert refresh.json()["work_item_id"]
            refresh_work = WorkstationWorker(scheduler).run_once()
            assert refresh_work is not None
            assert refresh_work.status == "completed", refresh_work.error
            assert refresh_work.result["source"]["refreshed_from_source_id"] == (
                published["published_source_id"]
            )
    finally:
        database.close()


def test_interrupted_inspection_recovery_and_activity_retry_stay_synchronized(
    tmp_path: Path,
) -> None:
    from halo_forge.workstation_jobs import WorkstationScheduler

    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"prompt": "hello", "response": "world"}])
    database = RunDatabase(tmp_path / "recovery.db")
    scheduler = WorkstationScheduler(database, process_probe=lambda _pid, _start: False)
    service = GuidedOwnDataService(
        database,
        datasets_root=tmp_path / "datasets",
        imports_root=tmp_path / "imports",
        scheduler=scheduler,
    )
    try:
        session = service.create_import(
            {"source_kind": "workstation_path", "source_uri": str(source)}
        )
        request = service.request_inspection(session["id"])
        inspection_id = request["inspection"]["id"]
        claimed = scheduler.claim(work_item_id=request["work_item_id"])
        assert claimed is not None and claimed.status == "running"
        database.update_dataset_source_inspection(inspection_id, status="running")

        recovery = scheduler.recover_or_adopt()
        assert [item.id for item in recovery.interrupted] == [claimed.id]
        assert database.get_dataset_source_inspection(inspection_id).status == "interrupted"
        assert database.get_dataset_import(session["id"]).status == "failed"

        retried = scheduler.retry(claimed.id, reason="operator reviewed interruption")
        assert retried is not None and retried.status == "queued"
        assert database.get_dataset_source_inspection(inspection_id).status == "queued"
        assert database.get_dataset_import(session["id"]).status == "inspecting"
    finally:
        database.close()
