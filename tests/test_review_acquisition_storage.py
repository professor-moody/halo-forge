from __future__ import annotations

import json
from pathlib import Path

import pytest

from halo_forge.review_lab._canonical import canonical_json
from halo_forge.review_lab.acquisition import plan_acquisition
from halo_forge.review_lab.acquisition_storage import (
    AcquisitionManifestStore,
    AcquisitionRecordSpool,
    INGESTION_SOURCE_HASH_FIELD,
)
from halo_forge.review_lab.errors import (
    ReviewIntegrityError,
    ReviewStateError,
    ReviewValidationError,
)


def _plan(prompt: str = "one"):
    return plan_acquisition(
        [
            {
                "record_id": "record-1",
                "record": {"prompt": prompt},
                "source": {
                    "kind": "dataset_version",
                    "ref": "version-1",
                    "revision": "sha256:source",
                    "split": "train",
                },
            }
        ],
        metadata={"operator_note": "review source"},
    )


def test_manifest_publication_is_checksummed_atomic_and_idempotent(tmp_path: Path):
    store = AcquisitionManifestStore(tmp_path / "reviews")
    plan = _plan()
    batch_id = plan.default_batch_id
    manifest = store.publish(batch_id, plan)
    root = tmp_path / "reviews" / "acquisitions" / batch_id
    assert root.is_dir()
    assert manifest["content_hash"] == plan.content_hash
    assert manifest["source_hash"] == plan.source_hash
    assert manifest["source_pins"] == list(plan.source_pins)
    assert manifest["request"] == plan.request
    checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    assert set(checksums) == {"candidates.jsonl", "manifest.json"}
    assert store.verify(batch_id, expected_content_hash=plan.content_hash).valid
    assert list(store.iter_candidates(batch_id))[0]["record_id"] == "record-1"
    assert store.publish(batch_id, plan) == manifest
    assert not list((root.parent).glob(".stage-*"))

    with pytest.raises(ReviewIntegrityError, match="content hash"):
        store.publish(batch_id, _plan("different"))


def test_manifest_mutation_and_unsafe_identifiers_are_rejected(tmp_path: Path):
    store = AcquisitionManifestStore(tmp_path / "reviews")
    plan = _plan()
    batch_id = plan.default_batch_id
    store.publish(batch_id, plan)
    candidates = store.path_for(batch_id) / "candidates.jsonl"
    candidates.write_text(candidates.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    verification = store.verify(batch_id, expected_content_hash=plan.content_hash)
    assert verification.valid is False
    assert any("checksum mismatch" in value for value in verification.errors)
    with pytest.raises(ReviewIntegrityError, match="failed verification"):
        list(store.iter_candidates(batch_id))
    with pytest.raises(ReviewIntegrityError, match="failed verification"):
        store.publish(batch_id, plan)
    with pytest.raises(ReviewValidationError, match="only letters"):
        store.path_for("../outside")


def test_restart_safe_spool_recovers_complete_lines_and_discards_partial_tail(
    tmp_path: Path,
):
    review_root = tmp_path / "reviews"
    spool = AcquisitionRecordSpool(review_root, "ingest-1")
    checkpoint = spool.append([{"record_id": "a"}, {"record_id": "b"}])
    assert checkpoint["record_count"] == 2

    # Simulate a crash after a complete record fsync but before checkpoint update.
    with spool.records_path.open("ab") as handle:
        handle.write((canonical_json({"record_id": "c"}) + "\n").encode("utf-8"))
    recovered = AcquisitionRecordSpool(review_root, "ingest-1")
    assert recovered.checkpoint["record_count"] == 3

    # An unsealed partial tail is the only data discarded during recovery.
    with recovered.records_path.open("ab") as handle:
        handle.write(b'{"record_id":"partial"')
    recovered = AcquisitionRecordSpool(review_root, "ingest-1")
    assert recovered.checkpoint["record_count"] == 3
    assert [value["record_id"] for value in recovered.iter_records()] == ["a", "b", "c"]
    recovered.append([{"record_id": "d"}])
    pin = recovered.seal()
    assert pin == {
        "kind": "acquisition_spool",
        "ref": "ingest-1",
        "format_version": 1,
        "row_count": 4,
        "content_hash": recovered.checkpoint["records_sha256"],
    }
    sealed = AcquisitionRecordSpool(review_root, "ingest-1")
    assert sealed.source_pin() == pin
    with pytest.raises(ReviewStateError, match="sealed"):
        sealed.append([{"record_id": "e"}])

    with sealed.records_path.open("ab") as handle:
        handle.write((canonical_json({"record_id": "tampered"}) + "\n").encode("utf-8"))
    with pytest.raises(ReviewIntegrityError, match="mutated"):
        AcquisitionRecordSpool(review_root, "ingest-1")


def test_spool_resume_rejects_a_changed_or_reordered_source_prefix(tmp_path: Path):
    spool = AcquisitionRecordSpool(tmp_path / "reviews", "ingest-prefix")
    spool.append([{"record_id": "a", "value": 1}, {"record_id": "b", "value": 2}])

    tail = spool.resume_after_verified_prefix(
        [
            {"record_id": "a", "value": 1},
            {"record_id": "b", "value": 2},
            {"record_id": "c", "value": 3},
        ]
    )
    assert list(tail) == [{"record_id": "c", "value": 3}]

    with pytest.raises(ReviewIntegrityError, match="changed or reordered"):
        spool.resume_after_verified_prefix(
            [
                {"record_id": "b", "value": 2},
                {"record_id": "a", "value": 1},
                {"record_id": "c", "value": 3},
            ]
        )


def test_spool_recovery_compares_raw_source_identity_before_generated_evidence(
    tmp_path: Path,
):
    import hashlib

    raw = {"record_id": "a", "record": {"prompt": "source"}}
    persisted = {
        **raw,
        INGESTION_SOURCE_HASH_FIELD: hashlib.sha256(
            canonical_json(raw).encode("utf-8")
        ).hexdigest(),
        "evidence": {
            "embedding": [0.1, 0.2],
            "embedding_revision": "text:model@commit",
        },
    }
    spool = AcquisitionRecordSpool(tmp_path / "reviews", "ingest-generated")
    spool.append([persisted])
    tail = spool.resume_after_verified_prefix(
        [raw, {"record_id": "b", "record": {"prompt": "tail"}}]
    )
    assert list(tail) == [{"record_id": "b", "record": {"prompt": "tail"}}]
