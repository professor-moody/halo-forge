from __future__ import annotations

from datetime import datetime, timezone
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from halo_forge.artifact_lab import ArtifactOperationService, ArtifactStore, OperationSpec
from halo_forge.artifact_studio import ArtifactStudioError, ArtifactStudioService
from halo_forge.evaluation_lab import EvaluationLabService
from halo_forge.qualification_lab.executor import EvaluationQualificationExecutor
from halo_forge.run_db import LabV4Catalog, RunDatabase
from halo_forge.workstation_jobs import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
    WorkstationScheduler,
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


def _raw(path: Path, value: str = "weights") -> Path:
    path.mkdir(parents=True)
    (path / "weights.bin").write_text(value, encoding="utf-8")
    return path


def _hf_output(path: Path, value: str = "weights") -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_text(value, encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")


def _studio(tmp_path: Path, *, engines=None):
    database = RunDatabase(str(tmp_path / "runs.db"))
    store = ArtifactStore(tmp_path / "artifacts")
    catalog = LabV4Catalog(database)
    service = ArtifactStudioService(
        database,
        store=store,
        catalog=catalog,
        scheduler=WorkstationScheduler(
            database,
            worker_id="integrity-test",
            capacity_probe=_available_capacity,
        ),
        operation_service=ArtifactOperationService(store, engines=engines or {}),
    )
    return service, database, store, catalog


def test_verification_levels_never_turn_structure_into_load_evidence(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "library")
    registered = store.import_artifact(
        _raw(tmp_path / "model"), artifact_kind="final", artifact_format="raw"
    )

    assert registered.blob.integrity == "hash_verified"
    hash_only = store.verify(registered.location.id, structural=False)
    assert hash_only.verification_level == "hash_verified"
    assert hash_only.satisfies("hash_verified")
    assert not hash_only.satisfies("structural_verified")

    structural = store.verify(registered.location.id)
    assert structural.verification_level == "structural_verified"
    assert structural.loadability_checked is False
    assert not structural.satisfies("load_verified")

    loaded = store.verify(
        registered.location.id,
        loader_probe=lambda _path, _blob: {"passed": True, "backend": "test-runtime"},
    )
    assert loaded.verification_level == "load_verified"
    round_trip = store.verify(
        registered.location.id,
        loader_probe=lambda _path, _blob: True,
        round_trip_report={"passed": True, "prompts": 3},
    )
    assert round_trip.verification_level == "round_trip_verified"


def test_explicit_quantization_fallback_changes_published_artifact_identity(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "library")
    source = store.import_artifact(
        _raw(tmp_path / "source"), artifact_kind="final", artifact_format="raw"
    )

    def fallback_engine(_spec, _inputs, output):
        output.write_bytes(b"GGUFpayload")
        return {
            "actual_output_quantization": "fp16",
            "unquantized_fallback_used": True,
            "result": {
                "actual_quantization": "fp16",
                "unquantized_fallback_used": True,
            },
        }

    operations = ArtifactOperationService(store, engines={"quantize": fallback_engine})
    spec = OperationSpec(
        operation_type="quantize",
        input_content_hashes=(source.blob.content_hash,),
        output_kind="quantized",
        output_format="gguf",
        output_quantization="q4",
        parameters={"allow_unquantized_fallback": True},
        tool_id="test.gguf",
        tool_version="1",
    )
    completed = operations.run(spec)
    blob = store.get_blob(completed.output_content_hash)
    resolved = completed.engine_metadata["resolved_output"]

    assert blob.artifact_kind == "converted"
    assert blob.quantization == "fp16"
    assert blob.quantization_method is None
    assert resolved["requested_quantization"] == "q4"
    assert resolved["actual_quantization"] == "fp16"
    assert resolved["unquantized_fallback_used"] is True
    assert len(resolved["identity_hash"]) == 64

    rejected = OperationSpec(
        operation_type="quantize",
        input_content_hashes=(source.blob.content_hash,),
        output_kind="quantized",
        output_format="gguf",
        output_quantization="q8",
        parameters={"allow_unquantized_fallback": False},
        tool_id="test.gguf",
        tool_version="1",
    )
    with pytest.raises(Exception, match="did not allow"):
        operations.run(rejected)


def test_convert_to_gguf_reports_exporters_actual_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: object()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    from halo_forge.inference.export import GGUFExporter

    def fake_export(self, _model, output_path, **_kwargs):
        Path(output_path).write_bytes(b"GGUFpayload")
        self.last_export_evidence = {
            "requested_backend_quantization": "Q4_K_M",
            "actual_backend_quantization": "F16",
            "unquantized_fallback_used": True,
            "fallback_reason": "test quantizer unavailable",
        }
        return output_path

    monkeypatch.setattr(GGUFExporter, "export", fake_export)
    from halo_forge.inference.convert import convert_to_gguf

    result = convert_to_gguf(
        source="local-source",
        output_path=str(tmp_path / "model.gguf"),
        quantization="q4",
        allow_unquantized_fallback=True,
    )
    assert result.quantization == "q4"
    assert result.actual_quantization == "fp16"
    assert result.actual_backend_quantization == "F16"
    assert result.unquantized_fallback_used is True


def test_merge_requires_pinned_or_content_resolved_base_and_records_base_lineage(
    tmp_path: Path,
) -> None:
    def merge_engine(spec, inputs, output):
        assert spec.parameters["base_input_index"] == 0
        assert len(inputs) == 3
        output.mkdir(parents=True)
        (output / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (output / "adapter_model.safetensors").write_text("merged", encoding="utf-8")
        return {"base_path": str(inputs[0])}

    service, database, _store, catalog = _studio(tmp_path, engines={"combine": merge_engine})
    try:
        base = service.import_artifact(
            _raw(tmp_path / "base", "base"),
            artifact_kind="final_model",
            artifact_format="raw",
            model_id="local/base",
        )
        adapters = [
            service.import_artifact(
                _raw(tmp_path / name, name),
                artifact_kind="adapter",
                artifact_format="raw",
            )
            for name in ("adapter-a", "adapter-b")
        ]
        adapter_ids = [value["occurrence"]["id"] for value in adapters]
        with pytest.raises(ValueError, match="pinned"):
            service.queue_merge(input_occurrence_ids=adapter_ids, base_model="org/base")

        receipt = service.queue_merge(
            input_occurrence_ids=adapter_ids,
            base_model=base["occurrence"]["id"],
            method="ties",
        )
        operation = catalog.get_operation(receipt.domain_id)
        operation_value = operation.to_dict()
        assert operation_value["input_occurrence_ids"][0] == base["occurrence"]["id"]
        assert operation_value["resolved_spec"]["tool_version"] != "unknown"
        assert (
            operation_value["resolved_spec"]["parameters"]["base_content_hash"]
            == base["blob"]["content_hash"]
        )
        executed = service.execute_work_item(receipt.work_item_id)
        assert (
            executed["result"]["operation"]["engine_metadata"]["verification"]["verification_level"]
            == "structural_verified"
        )
        lineage = service.lineage(executed["result"]["output_occurrence_id"])
        parents = lineage["catalog"]["parents"]
        assert parents[0]["parent_occurrence_id"] == base["occurrence"]["id"]
        assert parents[0]["relation"] == "base"
        assert len(lineage["content"]["edges"]) == 3

        pinned = service.queue_merge(
            input_occurrence_ids=adapter_ids,
            base_model="org/base@0123456789abcdef",
            method="ties",
        )
        pinned_spec = catalog.get_operation(pinned.domain_id).to_dict()["resolved_spec"]
        assert pinned_spec["parameters"]["base_model"] == "org/base"
        assert pinned_spec["parameters"]["base_revision"] == "0123456789abcdef"
    finally:
        database.close()


def test_supplied_qualification_evaluation_must_match_subject_and_profile(
    tmp_path: Path,
) -> None:
    service, database, _store, catalog = _studio(tmp_path)
    evaluations = EvaluationLabService(database, tmp_path / "evaluations")
    try:
        candidate = service.import_artifact(
            _raw(tmp_path / "candidate", "candidate"),
            artifact_kind="final_model",
            artifact_format="raw",
            model_id="local/candidate",
        )
        other = service.import_artifact(
            _raw(tmp_path / "other", "other"),
            artifact_kind="final_model",
            artifact_format="raw",
            model_id="local/other",
        )
        suite = database.create_benchmark_suite(name="Development", purpose="development")
        revision = database.create_benchmark_suite_revision(
            suite_id=suite.id,
            content_hash="suite-content",
            items=[{"id": "item", "input": "hello", "expected": "world"}],
            primary_metric="quality",
            direction="maximize",
        )
        executor = EvaluationQualificationExecutor(database, catalog, evaluations)
        candidate_occurrence = catalog.get_occurrence(candidate["occurrence"]["id"])
        other_occurrence = catalog.get_occurrence(other["occurrence"]["id"])
        request = {
            "scores": {"item": 1.0},
            "qualification_binding": {
                "profile_revision_id": "qualification-profile-r1",
                "profile_content_hash": "profile-definition-hash",
                "stored_profile_content_hash": "stored-profile-hash",
                "stage": "development",
                "artifact_occurrence_id": candidate_occurrence.id,
                "artifact_content_hash": candidate["blob"]["content_hash"],
            },
        }
        launched = evaluations.launch_evaluation(
            suite_revision_id=revision.id,
            subject=executor._subject(candidate_occurrence),
            request=request,
        )
        completed = evaluations.jobs.wait(launched.evaluation.id, timeout=10)
        assert completed.status == "completed"
        evaluation_id, metrics = executor._run_stage(
            occurrence=candidate_occurrence,
            suite_revision_id=revision.id,
            stage="development",
            request=request,
            supplied_evaluation_id=completed.id,
            timeout=10,
        )
        assert evaluation_id == completed.id
        assert metrics["quality"] == 1.0

        with pytest.raises(ValueError, match="different artifact subject"):
            executor._run_stage(
                occurrence=other_occurrence,
                suite_revision_id=revision.id,
                stage="development",
                request={
                    **request,
                    "qualification_binding": {
                        **request["qualification_binding"],
                        "artifact_occurrence_id": other_occurrence.id,
                        "artifact_content_hash": other["blob"]["content_hash"],
                    },
                },
                supplied_evaluation_id=completed.id,
                timeout=10,
            )
        with pytest.raises(ValueError, match="qualification profile"):
            executor._run_stage(
                occurrence=candidate_occurrence,
                suite_revision_id=revision.id,
                stage="development",
                request={
                    **request,
                    "qualification_binding": {
                        **request["qualification_binding"],
                        "profile_content_hash": "another-profile",
                    },
                },
                supplied_evaluation_id=completed.id,
                timeout=10,
            )
    finally:
        evaluations.shutdown()
        database.close()


def test_corrupt_artifacts_are_blocked_from_qualification_serving_and_export(
    tmp_path: Path,
) -> None:
    service, database, _store, _catalog = _studio(tmp_path)
    try:
        artifact = service.import_artifact(
            _raw(tmp_path / "mutable"),
            artifact_kind="final_model",
            artifact_format="raw",
        )
        (tmp_path / "mutable" / "weights.bin").write_text("changed", encoding="utf-8")
        occurrence_id = artifact["occurrence"]["id"]
        with pytest.raises(ArtifactStudioError, match="cannot be exported"):
            service.queue_export(occurrence_id=occurrence_id, destination=tmp_path / "bundle")
        with pytest.raises(ArtifactStudioError, match="reserved for serving"):
            service.reserve_serving(
                occurrence_id,
                name="Broken",
                backend="local",
            )
    finally:
        database.close()
