import json
from pathlib import Path

import pytest

from halo_forge.data_lab import (
    Recipe,
    RecipeResult,
    ReviewDatasetBuilder,
    SourceSnapshot,
    SourceSpec,
    VersionStore,
)
from halo_forge.data_lab.identity import INTERNAL_LINEAGE_KEY, strip_internal_identity


def _parent_version(tmp_path: Path):
    root = tmp_path / "datasets"
    store = VersionStore(root)
    source_path = tmp_path / "source.jsonl"
    rows = [
        {"prompt": "one", "response": "old one"},
        {"prompt": "two", "response": "old two"},
        {"prompt": "three", "response": "old three"},
    ]
    source_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    version = store.publish(
        dataset_id="reviewed-sft",
        recipe=Recipe.from_value(
            {"name": "parent", "schema": "sft", "steps": [{"kind": "validate"}]}
        ),
        result=RecipeResult(
            records=rows,
            splits={"train": [rows[0], rows[2]], "validation": [rows[1]]},
        ),
        source=SourceSnapshot(
            spec=SourceSpec(kind="local", path=str(source_path)),
            records=rows,
            fingerprint="parent-source",
            size_bytes=source_path.stat().st_size,
            file_count=1,
        ),
    )
    identified = store.load_records_with_lineage(version.version_id, dataset_id=version.dataset_id)
    return store, version, identified


def _revision(tmp_path: Path):
    path = tmp_path / "labels" / "revision-one"
    path.mkdir(parents=True, exist_ok=True)
    (path / "records.jsonl").write_text("{}\n", encoding="utf-8")
    return {
        "id": "label-revision-one",
        "label_set_id": "label-set-one",
        "content_hash": "labels-content",
        "storage_path": str(path),
    }


def test_review_filter_removes_only_explicit_rejections(tmp_path: Path):
    store, parent, rows = _parent_version(tmp_path)
    first_id = rows[0][INTERNAL_LINEAGE_KEY]["record_id"]
    second_id = rows[1][INTERNAL_LINEAGE_KEY]["record_id"]
    items = [
        {
            "review_item_id": "accepted",
            "record_id": first_id,
            "record_hash": "first",
            "annotation": {"accepted": True},
            "output_records": [strip_internal_identity(rows[0])],
            "excluded": False,
        },
        {
            "review_item_id": "rejected",
            "record_id": second_id,
            "record_hash": "second",
            "annotation": {"accepted": False},
            "output_records": [],
            "excluded": False,
        },
    ]
    builder = ReviewDatasetBuilder(store)
    preview = builder.preview(
        _revision(tmp_path),
        items,
        dataset_id=parent.dataset_id,
        parent_version_id=parent.version_id,
        build_mode="filter",
    )
    assert preview.removed_count == 1
    assert preview.output_count == 2

    child = builder.build(
        _revision(tmp_path),
        items,
        dataset_id=parent.dataset_id,
        parent_version_id=parent.version_id,
        build_mode="filter",
    )
    outputs = store.load_records_with_lineage(child.version_id, dataset_id=child.dataset_id)
    assert {row["prompt"] for row in outputs} == {"one", "three"}
    manifest = json.loads((Path(child.path) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parent_version_id"] == parent.version_id


def test_review_replacement_moves_source_out_of_validation(tmp_path: Path):
    store, parent, rows = _parent_version(tmp_path)
    source_id = rows[1][INTERNAL_LINEAGE_KEY]["record_id"]
    items = [
        {
            "review_item_id": "corrected",
            "record_id": source_id,
            "record_hash": "second",
            "annotation": {"corrected_text": "new two"},
            "output_records": [{"prompt": "two", "response": "new two"}],
            "excluded": False,
        }
    ]
    builder = ReviewDatasetBuilder(store)
    child = builder.build(
        _revision(tmp_path),
        items,
        dataset_id=parent.dataset_id,
        parent_version_id=parent.version_id,
        build_mode="replace_by_record_id",
        target_split="train",
    )
    training = store.load_records_with_lineage(
        child.version_id, dataset_id=child.dataset_id, split="train"
    )
    validation = store.load_records_with_lineage(
        child.version_id, dataset_id=child.dataset_id, split="validation"
    )
    assert any(row.get("response") == "new two" for row in training)
    assert all(row[INTERNAL_LINEAGE_KEY]["record_id"] != source_id for row in validation)


def test_new_review_dataset_requires_append_and_publishes_lineage(tmp_path: Path):
    store = VersionStore(tmp_path / "datasets")
    revision = _revision(tmp_path)
    items = [
        {
            "review_item_id": "new-example",
            "record_id": "review-origin-one",
            "record_hash": "new",
            "annotation": {"corrected_text": "answer"},
            "output_records": [{"prompt": "question", "response": "answer"}],
            "excluded": False,
        }
    ]
    version = ReviewDatasetBuilder(store).build(
        revision,
        items,
        dataset_id="new-reviewed-data",
        build_mode="append",
    )
    records = store.load_records_with_lineage(version.version_id, dataset_id=version.dataset_id)
    assert records[0][INTERNAL_LINEAGE_KEY]["record_id"] == "review-origin-one"
    assert records[0][INTERNAL_LINEAGE_KEY]["operations"][0]["operation"] == (
        "review_label_set"
    )


def test_review_build_warns_when_development_evidence_is_no_longer_untouched(
    tmp_path: Path,
):
    store = VersionStore(tmp_path / "datasets")
    preview = ReviewDatasetBuilder(store).preview(
        _revision(tmp_path),
        [
            {
                "review_item_id": "development-example",
                "record_id": "development-record",
                "record_hash": "development-hash",
                "annotation": {"corrected_text": "reviewed"},
                "output_records": [{"prompt": "question", "response": "reviewed"}],
                "lineage": {
                    "source": {
                        "kind": "evaluation",
                        "purpose": "development",
                        "suite_revision_id": "suite-revision",
                    }
                },
                "excluded": False,
            }
        ],
        dataset_id="reviewed-development-data",
        build_mode="append",
    )

    assert any("cannot treat that suite as untouched" in value for value in preview.warnings)


def test_review_build_quarantines_invalid_canonical_outputs(tmp_path: Path):
    store = VersionStore(tmp_path / "datasets")
    revision = _revision(tmp_path)
    items = [
        {
            "review_item_id": "valid",
            "record_id": "valid-record",
            "record_hash": "valid-hash",
            "annotation": {"corrected_text": "answer"},
            "output_records": [{"prompt": "question", "response": "answer"}],
            "excluded": False,
        },
        {
            "review_item_id": "invalid",
            "record_id": "invalid-record",
            "record_hash": "invalid-hash",
            "annotation": {"corrected_text": "missing prompt"},
            "output_records": [{"response": "missing prompt"}],
            "excluded": False,
        },
    ]
    builder = ReviewDatasetBuilder(store)
    preview = builder.preview(
        revision,
        items,
        dataset_id="validated-review-data",
        build_mode="append",
        schema="sft",
    )
    assert preview.output_count == 1
    assert preview.quarantined_count == 1

    version = builder.build(
        revision,
        items,
        dataset_id="validated-review-data",
        build_mode="append",
        schema="sft",
    )
    quarantined = [
        json.loads(line)
        for line in (Path(version.path) / "quarantined.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(quarantined) == 1
    assert quarantined[0]["_review_item_id"] == "invalid"
    assert "missing required field" in quarantined[0]["_rejection_reason"]


def test_review_build_honors_cancellation_before_atomic_publication(tmp_path: Path):
    store = VersionStore(tmp_path / "datasets")
    checks = 0

    def cancel_before_publish():
        nonlocal checks
        checks += 1
        if checks >= 4:
            raise RuntimeError("cancel requested")

    with pytest.raises(RuntimeError, match="cancel requested"):
        ReviewDatasetBuilder(store).build(
            _revision(tmp_path),
            [
                {
                    "review_item_id": "cancelled-item",
                    "record_id": "cancelled-record",
                    "record_hash": "cancelled-hash",
                    "annotation": {"corrected_text": "answer"},
                    "output_records": [{"prompt": "question", "response": "answer"}],
                    "excluded": False,
                }
            ],
            dataset_id="cancelled-review-data",
            build_mode="append",
            schema="sft",
            check_cancelled=cancel_before_publish,
        )
    assert store.list("cancelled-review-data") == []


def test_review_preview_detects_same_media_across_split_record_ids(tmp_path: Path):
    image = tmp_path / "shared.png"
    image.write_bytes(b"same-image-content")
    store = VersionStore(tmp_path / "datasets")
    rows = [
        {"image": str(image), "prompt": "train prompt", "response": "train"},
        {"image": str(image), "prompt": "validation prompt", "response": "validation"},
    ]
    source_path = tmp_path / "vlm.jsonl"
    source_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    parent = store.publish(
        dataset_id="reviewed-vlm",
        recipe=Recipe.from_value(
            {"name": "vlm-parent", "schema": "vlm", "steps": [{"kind": "validate"}]}
        ),
        result=RecipeResult(
            records=rows,
            splits={"train": [rows[0]], "validation": [rows[1]]},
        ),
        source=SourceSnapshot(
            spec=SourceSpec(kind="local", path=str(source_path)),
            records=rows,
            fingerprint="vlm-source",
            size_bytes=source_path.stat().st_size,
            file_count=1,
        ),
    )
    preview = ReviewDatasetBuilder(store).preview(
        _revision(tmp_path),
        [],
        dataset_id=parent.dataset_id,
        parent_version_id=parent.version_id,
        build_mode="append",
        schema="vlm",
    )
    assert preview.contamination["assets"]["match_count"] == 1
    assert any("Media contamination" in warning for warning in preview.warnings)
