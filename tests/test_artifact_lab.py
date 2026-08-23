from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from halo_forge.artifact_lab import (
    ArtifactIntegrityError,
    ArtifactOperationError,
    ArtifactOperationService,
    ArtifactStore,
    ArtifactStoreError,
    CleanupProtections,
    OperationSpec,
    existing_convert_engine,
    hash_path,
    verify_round_trip,
)


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 14, 15, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: int) -> None:
        self.value += timedelta(**kwargs)


def _raw_payload(path: Path, value: str = "weights") -> Path:
    path.mkdir(parents=True)
    (path / "weights.bin").write_text(value, encoding="utf-8")
    (path / "config.txt").write_text("configuration", encoding="utf-8")
    return path


def _hf_payload(path: Path, value: str = "weights") -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_text(value, encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    return path


def test_hash_path_is_layout_and_content_deterministic(tmp_path: Path) -> None:
    first = _raw_payload(tmp_path / "first")
    second = _raw_payload(tmp_path / "second")
    assert hash_path(first) == hash_path(second)

    (second / "weights.bin").write_text("different", encoding="utf-8")
    assert hash_path(first).content_hash != hash_path(second).content_hash

    renamed = tmp_path / "renamed"
    renamed.mkdir()
    (renamed / "renamed.bin").write_text("weights", encoding="utf-8")
    (renamed / "config.txt").write_text("configuration", encoding="utf-8")
    assert hash_path(first).content_hash != hash_path(renamed).content_hash


def test_single_file_hash_is_independent_of_source_name(tmp_path: Path) -> None:
    first = tmp_path / "a.gguf"
    second = tmp_path / "renamed.gguf"
    first.write_bytes(b"GGUFpayload")
    second.write_bytes(b"GGUFpayload")
    assert hash_path(first).content_hash == hash_path(second).content_hash


def test_directory_symlinks_are_rejected_without_traversal(tmp_path: Path) -> None:
    outside = _raw_payload(tmp_path / "outside")
    root = tmp_path / "payload"
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Directory symlinks"):
        hash_path(root)


def test_referenced_import_and_adoption_preserve_raw_source(tmp_path: Path) -> None:
    clock = Clock()
    store = ArtifactStore(tmp_path / "library", clock=clock)
    source = _raw_payload(tmp_path / "run-output")
    registration = store.import_artifact(
        source,
        artifact_kind="checkpoint",
        artifact_format="raw",
        run_id="run-1",
    )

    assert registration.location.location_kind == "referenced"
    assert Path(registration.location.path) == source.resolve()
    assert not (store.blobs_dir / registration.blob.content_hash).exists()
    before = hash_path(source)

    managed = store.adopt(
        registration.blob.content_hash, source_location_id=registration.location.id
    )
    assert managed.location_kind == "managed"
    assert Path(managed.path).is_dir()
    assert hash_path(Path(managed.path)) == before
    assert hash_path(source) == before
    assert len(store.list_locations(content_hash=registration.blob.content_hash)) == 2

    shutil.rmtree(source)
    assert store.resolve_location(registration.blob.content_hash).id == managed.id


def test_identical_run_occurrences_share_one_blob(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "library")
    first_source = _raw_payload(tmp_path / "run-1")
    second_source = _raw_payload(tmp_path / "run-2")
    first = store.import_artifact(
        first_source,
        artifact_kind="checkpoint",
        artifact_format="raw",
        occurrence_id="checkpoint-run-1",
        run_id="run-1",
    )
    second = store.import_artifact(
        second_source,
        artifact_kind="checkpoint",
        artifact_format="raw",
        occurrence_id="checkpoint-run-2",
        run_id="run-2",
    )

    assert first.blob.content_hash == second.blob.content_hash
    assert second.reused_blob is True
    assert len(store.list_blobs()) == 1
    assert {item.run_id for item in store.list_occurrences()} == {"run-1", "run-2"}


def test_occurrence_ids_cannot_be_rebound(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "library")
    first = _raw_payload(tmp_path / "first", "one")
    second = _raw_payload(tmp_path / "second", "two")
    store.import_artifact(
        first,
        artifact_kind="checkpoint",
        artifact_format="raw",
        occurrence_id="stable-id",
    )
    with pytest.raises(ArtifactStoreError, match="different content"):
        store.import_artifact(
            second,
            artifact_kind="checkpoint",
            artifact_format="raw",
            occurrence_id="stable-id",
        )

    with pytest.raises(ValueError, match="occurrence ID"):
        store.import_artifact(
            first,
            artifact_kind="checkpoint",
            artifact_format="raw",
            occurrence_id="../escape",
        )


def test_verify_detects_mutated_source_and_never_claims_loader_evidence(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "library")
    source = _raw_payload(tmp_path / "source")
    registration = store.import_artifact(source, artifact_kind="final", artifact_format="raw")
    first = store.verify(registration.location.id)
    assert first.passed is True
    assert first.loadability_checked is False
    assert first.round_trip_checked is False

    (source / "weights.bin").write_text("mutated", encoding="utf-8")
    changed = store.verify(registration.location.id)
    assert changed.passed is False
    assert changed.content_hash_matches is False
    assert any("does not match" in error for error in changed.errors)


def test_format_verification_checks_hf_completeness_and_real_probe(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "library")
    complete = _hf_payload(tmp_path / "complete")
    artifact = store.import_artifact(
        complete, artifact_kind="final", artifact_format="hf", managed=True
    )
    report = store.verify(
        artifact.location.id,
        loader_probe=lambda path, blob: {"passed": True, "backend": "fake-loader"},
        round_trip_report={"passed": True, "n_prompts": 2},
    )
    assert report.passed is True
    assert report.loadability_checked is True
    assert report.round_trip_checked is True

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}", encoding="utf-8")
    broken = store.import_artifact(incomplete, artifact_kind="final", artifact_format="hf")
    failed = store.verify(broken.location.id)
    assert failed.passed is False
    assert failed.structural_checks["weights"] is False
    assert failed.structural_checks["tokenizer_config"] is False

    named_adapter = tmp_path / "named-adapter"
    combined = named_adapter / "halo_forge_combined"
    combined.mkdir(parents=True)
    (combined / "adapter_config.json").write_text("{}", encoding="utf-8")
    (combined / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    adapter = store.import_artifact(named_adapter, artifact_kind="adapter", artifact_format="hf")
    assert store.verify(adapter.location.id).passed is True


def test_operation_fingerprints_include_input_order_and_reject_qat() -> None:
    inputs = ("a" * 64, "b" * 64)
    first = OperationSpec(
        operation_type="combine",
        input_content_hashes=inputs,
        output_kind="merged",
        output_format="raw",
        parameters={"method": "ties", "weights": [0.7, 0.3]},
        tool_version="1",
    )
    same = OperationSpec(
        operation_type="combine",
        input_content_hashes=inputs,
        output_kind="merged",
        output_format="raw",
        parameters={"weights": [0.7, 0.3], "method": "ties"},
        tool_version="1",
    )
    reversed_inputs = OperationSpec(
        operation_type="combine",
        input_content_hashes=tuple(reversed(inputs)),
        output_kind="merged",
        output_format="raw",
        parameters={"method": "ties", "weights": [0.7, 0.3]},
        tool_version="1",
    )
    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != reversed_inputs.fingerprint

    with pytest.raises(ValueError, match="not QAT"):
        OperationSpec(
            operation_type="quantize",
            input_content_hashes=("a" * 64,),
            output_kind="quantized",
            output_format="gguf",
            output_quantization="q4",
            parameters={"quantization_method": "qat"},
        )


def test_fake_operation_publishes_atomically_records_lineage_and_reuses(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "library")
    first = store.import_artifact(
        _raw_payload(tmp_path / "first", "one"),
        artifact_kind="adapter",
        artifact_format="raw",
    )
    second = store.import_artifact(
        _raw_payload(tmp_path / "second", "two"),
        artifact_kind="adapter",
        artifact_format="raw",
    )
    calls = []

    def fake_engine(spec: OperationSpec, inputs: tuple[Path, ...], output: Path):
        calls.append((spec, inputs))
        output.mkdir()
        joined = "+".join((item / "weights.bin").read_text() for item in inputs)
        (output / "weights.bin").write_text(joined, encoding="utf-8")
        return {"engine": "lightweight-fake"}

    service = ArtifactOperationService(store, engines={"combine": fake_engine})
    spec = OperationSpec(
        operation_type="combine",
        input_content_hashes=(first.blob.content_hash, second.blob.content_hash),
        output_kind="merged",
        output_format="raw",
        parameters={"method": "linear"},
        tool_id="test",
        tool_version="1",
    )
    completed = service.run(spec)
    assert completed.status == "completed"
    assert completed.output_content_hash
    assert Path(store.resolve_location(completed.output_content_hash).path).is_dir()
    graph = store.lineage(completed.output_content_hash)
    assert [edge.parent_content_hash for edge in graph.edges] == [
        first.blob.content_hash,
        second.blob.content_hash,
    ]
    assert [edge.ordinal for edge in graph.edges] == [0, 1]

    # Completed operations are reusable after a process restart, not merely
    # through an in-memory cache.
    restarted_store = ArtifactStore(store.root)
    restarted_service = ArtifactOperationService(restarted_store, engines={"combine": fake_engine})
    reused = restarted_service.run(spec)
    assert reused.reused is True
    assert len(calls) == 1


def test_adopt_refuses_a_mutated_existing_managed_copy(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "library")
    source = _raw_payload(tmp_path / "source")
    registration = store.import_artifact(
        source,
        artifact_kind="final",
        artifact_format="raw",
    )
    managed = store.adopt(registration.blob.content_hash)
    (Path(managed.path) / "weights.bin").write_text("mutated", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="managed copy failed"):
        store.adopt(registration.blob.content_hash)


def test_failed_operation_records_attempt_without_publication(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "library")
    source = store.import_artifact(
        _raw_payload(tmp_path / "source"),
        artifact_kind="final",
        artifact_format="raw",
    )

    def failing(spec: OperationSpec, inputs: tuple[Path, ...], output: Path):
        output.mkdir()
        (output / "partial.bin").write_text("partial", encoding="utf-8")
        raise RuntimeError("engine crashed")

    service = ArtifactOperationService(store, engines={"convert": failing})
    spec = OperationSpec(
        operation_type="convert",
        input_content_hashes=(source.blob.content_hash,),
        output_kind="converted",
        output_format="raw",
    )
    with pytest.raises(ArtifactOperationError, match="engine crashed"):
        service.run(spec)
    assert len(store.list_blobs()) == 1
    assert list((store.metadata_dir / "operations").glob("*.failed-*.json"))
    assert not list(store.staging_dir.iterdir())


def test_content_preserving_operation_records_manifest_without_self_edge(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "library")
    source = store.import_artifact(
        _raw_payload(tmp_path / "source"),
        artifact_kind="final",
        artifact_format="raw",
    )

    def no_op(spec: OperationSpec, inputs: tuple[Path, ...], output: Path):
        shutil.copytree(inputs[0], output)
        return {"changed": False}

    service = ArtifactOperationService(store, engines={"convert": no_op})
    spec = OperationSpec(
        operation_type="convert",
        input_content_hashes=(source.blob.content_hash,),
        output_kind="converted",
        output_format="raw",
    )
    completed = service.run(spec)
    assert completed.output_content_hash == source.blob.content_hash
    assert completed.engine_metadata["changed"] is False
    assert store.lineage(source.blob.content_hash).edges == ()


def test_output_verifier_must_supply_real_passing_evidence(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "library")
    source = store.import_artifact(
        _raw_payload(tmp_path / "source"),
        artifact_kind="final",
        artifact_format="raw",
    )

    def engine(spec: OperationSpec, inputs: tuple[Path, ...], output: Path):
        _raw_payload(output, "new")
        return {}

    service = ArtifactOperationService(
        store,
        engines={"convert": engine},
        output_verifier=lambda output, spec: {"passed": False, "reason": "load failed"},
    )
    spec = OperationSpec(
        operation_type="convert",
        input_content_hashes=(source.blob.content_hash,),
        output_kind="converted",
        output_format="raw",
    )
    with pytest.raises(ArtifactOperationError, match="verifier rejected"):
        service.run(spec)
    assert len(store.list_blobs()) == 1


def test_lineage_is_immutable_and_descendants_are_queryable(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "library")
    parent = store.import_artifact(
        _raw_payload(tmp_path / "parent", "parent"),
        artifact_kind="checkpoint",
        artifact_format="raw",
    )
    child = store.import_artifact(
        _raw_payload(tmp_path / "child", "child"),
        artifact_kind="merged",
        artifact_format="raw",
    )
    store.record_lineage(
        child_content_hash=child.blob.content_hash,
        parent_content_hashes=[parent.blob.content_hash],
        relationship="bake",
        operation_fingerprint="op-1",
    )
    descendants = store.lineage(parent.blob.content_hash, direction="descendants")
    assert {item.content_hash for item in descendants.blobs} == {
        parent.blob.content_hash,
        child.blob.content_hash,
    }
    with pytest.raises(ArtifactStoreError, match="different immutable lineage"):
        store.record_lineage(
            child_content_hash=child.blob.content_hash,
            parent_content_hashes=[parent.blob.content_hash],
            relationship="convert",
            operation_fingerprint="op-2",
        )


def test_cleanup_preview_protects_references_and_requires_fresh_review(
    tmp_path: Path,
) -> None:
    clock = Clock()
    store = ArtifactStore(tmp_path / "library", clock=clock)
    protected = store.import_artifact(
        _raw_payload(tmp_path / "protected", "keep"),
        artifact_kind="final",
        artifact_format="raw",
        managed=True,
    )
    removable = store.import_artifact(
        _raw_payload(tmp_path / "removable", "remove"),
        artifact_kind="final",
        artifact_format="raw",
        managed=True,
    )
    preview_protections = CleanupProtections(pinned=frozenset({protected.blob.content_hash}))
    plan = store.preview_cleanup(protections=preview_protections)
    assert [item.identifier for item in plan.candidates] == [removable.blob.content_hash]
    assert plan.protected[0].reasons == ("pinned",)
    with pytest.raises(ValueError, match="review_note"):
        store.trash_cleanup(plan.id, review_note="", current_protections=preview_protections)

    # A pin created after preview must win over the stale candidate list.
    fresh = CleanupProtections(
        pinned=frozenset({protected.blob.content_hash, removable.blob.content_hash})
    )
    skipped = store.trash_cleanup(
        plan.id, review_note="Reviewed cleanup", current_protections=fresh
    )
    assert not skipped.trashed
    assert "protected" in skipped.skipped[removable.blob.content_hash]


def test_cleanup_trash_restore_and_seven_day_purge(tmp_path: Path) -> None:
    clock = Clock()
    store = ArtifactStore(tmp_path / "library", clock=clock)
    artifact = store.import_artifact(
        _raw_payload(tmp_path / "source", "temporary"),
        artifact_kind="final",
        artifact_format="raw",
        managed=True,
    )
    plan = store.preview_cleanup()
    result = store.trash_cleanup(
        plan.id,
        review_note="Reviewed and approved",
        current_protections=CleanupProtections(),
    )
    assert result.trashed == (artifact.blob.content_hash,)
    assert result.reclaimed_bytes == 0
    assert not Path(artifact.location.path).exists()
    with pytest.raises(ValueError, match="seven|7 days"):
        store.purge_trash(retention=timedelta(days=1))

    restored = store.restore(artifact.blob.content_hash)
    assert Path(restored).exists()
    assert store.verify(artifact.blob.content_hash, structural=False).passed

    second_plan = store.preview_cleanup()
    store.trash_cleanup(
        second_plan.id,
        review_note="Reviewed again",
        current_protections=CleanupProtections(),
    )
    assert store.purge_trash()["purged"] == []
    clock.advance(days=8)
    purged = store.purge_trash()
    assert purged["purged"] == [artifact.blob.content_hash]
    assert purged["reclaimed_bytes"] > 0


def test_lineage_parent_is_automatically_cleanup_protected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "library")
    parent = store.import_artifact(
        _raw_payload(tmp_path / "parent", "base"),
        artifact_kind="checkpoint",
        artifact_format="raw",
        managed=True,
    )
    child = store.import_artifact(
        _raw_payload(tmp_path / "child", "derived"),
        artifact_kind="converted",
        artifact_format="raw",
        managed=True,
    )
    store.record_lineage(
        child_content_hash=child.blob.content_hash,
        parent_content_hashes=[parent.blob.content_hash],
        relationship="convert",
        operation_fingerprint="operation",
    )
    plan = store.preview_cleanup()
    protected = {item.content_hash: item.reasons for item in plan.protected}
    assert protected[parent.blob.content_hash] == ("lineage_required",)
    assert [item.identifier for item in plan.candidates] == [child.blob.content_hash]


def test_stale_staging_is_previewed_and_trashed(tmp_path: Path) -> None:
    clock = Clock()
    store = ArtifactStore(tmp_path / "library", clock=clock)
    stale = store.staging_dir / "abandoned"
    stale.mkdir()
    (stale / "partial").write_text("partial", encoding="utf-8")
    old_timestamp = (clock.value - timedelta(days=2)).timestamp()
    os.utime(stale, (old_timestamp, old_timestamp))

    plan = store.preview_cleanup()
    candidate = next(item for item in plan.candidates if item.resource_type == "staging")
    assert candidate.identifier == "abandoned"
    result = store.trash_cleanup(
        plan.id,
        review_note="Remove abandoned attempt",
        current_protections=CleanupProtections(),
    )
    assert "abandoned" in result.trashed
    assert not stale.exists()


def test_active_staging_is_rechecked_after_cleanup_preview(tmp_path: Path) -> None:
    clock = Clock()
    store = ArtifactStore(tmp_path / "library", clock=clock)
    staging = store.staging_dir / "active-attempt"
    staging.mkdir()
    (staging / "partial").write_text("partial", encoding="utf-8")
    old_timestamp = (clock.value - timedelta(days=2)).timestamp()
    os.utime(staging, (old_timestamp, old_timestamp))
    plan = store.preview_cleanup()
    result = store.trash_cleanup(
        plan.id,
        review_note="Reviewed, but attempt resumed",
        current_protections=CleanupProtections(active_staging=frozenset({"active-attempt"})),
    )
    assert result.skipped["active-attempt"] == "protected: active_staging"
    assert staging.exists()


def test_portable_export_contains_identity_evidence_checksums_and_model_card(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "library")
    artifact = store.import_artifact(
        _hf_payload(tmp_path / "model"),
        artifact_kind="quantized",
        artifact_format="hf",
        quantization="q8",
        quantization_method="post_training",
        managed=True,
    )
    destination = tmp_path / "portable"
    bundle = store.export_bundle(
        artifact.blob.content_hash,
        destination,
        replay_identity={"run_id": "run-1"},
        dataset_identity={"version_id": "dataset-v1"},
        qualification={"decision": "pass", "profile_revision_id": "qualification-1"},
        license_metadata={"license": "Apache-2.0"},
    )
    manifest = json.loads((destination / "bundle-manifest.json").read_text())
    assert manifest["source_content_hash"] == artifact.blob.content_hash
    assert (destination / "model" / "model.safetensors").is_file()
    assert json.loads((destination / "replay.json").read_text())["run_id"] == "run-1"
    assert json.loads((destination / "dataset.json").read_text())["version_id"] == "dataset-v1"
    assert "model/model.safetensors" in (destination / "SHA256SUMS").read_text()
    card = (destination / "MODEL_CARD.md").read_text()
    assert "post_training" in card
    assert "does not claim QAT" in card
    assert bundle.size_bytes > artifact.blob.size_bytes

    reused = store.export_bundle(artifact.blob.content_hash, destination)
    assert reused.id == bundle.id
    assert reused.reused is True

    (destination / "model" / "model.safetensors").write_text("tampered", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="bundle has changed"):
        store.export_bundle(artifact.blob.content_hash, destination)


def test_storage_inventory_and_disk_preflight_are_truthful(tmp_path: Path, monkeypatch) -> None:
    store = ArtifactStore(tmp_path / "library")
    store.import_artifact(
        _raw_payload(tmp_path / "model"),
        artifact_kind="final",
        artifact_format="raw",
        managed=True,
    )
    inventory = store.inventory()
    assert inventory.blob_count == 1
    assert inventory.managed_blob_count == 1
    assert inventory.managed_bytes > 0

    usage = shutil._ntuple_diskusage(total=100 * 1024**3, used=75 * 1024**3, free=25 * 1024**3)
    monkeypatch.setattr("halo_forge.artifact_lab.store.shutil.disk_usage", lambda path: usage)
    blocked = store.disk_preflight(6 * 1024**3)
    assert blocked["passed"] is False
    assert blocked["required_reserve_bytes"] == 20 * 1024**3
    overridden = store.disk_preflight(6 * 1024**3, override_reason="Operator reviewed disk")
    assert overridden["passed"] is True
    assert overridden["overridden"] is True


def test_existing_convert_adapter_rejects_unverified_onnx_without_loading_models(
    tmp_path: Path,
) -> None:
    source = _raw_payload(tmp_path / "source")
    spec = OperationSpec(
        operation_type="convert",
        input_content_hashes=("a" * 64,),
        output_kind="converted",
        output_format="onnx",
    )
    with pytest.raises(ValueError, match="No verified.*onnx"):
        existing_convert_engine(spec, (source,), tmp_path / "output")


def test_round_trip_wrapper_uses_supplied_generators_only() -> None:
    report = verify_round_trip(
        source_generate=lambda prompt: f"answer {prompt}",
        artifact_generate=lambda prompt: f"answer {prompt}",
        prompts=["one", "two"],
    )
    assert report["passed"] is True
    assert report["n_prompts"] == 2
