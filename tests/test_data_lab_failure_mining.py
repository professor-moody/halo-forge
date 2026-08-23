"""Focused contracts for reviewed, immutable evaluation failure mining."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from halo_forge.data_lab import (
    FailureMiningBuilder,
    FailureMiningSelector,
    VersionError,
    VersionStore,
    exclusions_hash,
    preview_failure_mining,
)
from halo_forge.data_lab.identity import seed_record_identities
from halo_forge.data_lab.recipe import RecipeResult
from halo_forge.data_lab.sources import SourceSnapshot, SourceSpec, fingerprint_assets


def _jsonl(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _comparison():
    def sample(record_id, outcome, passed, *, score, reward, **metadata):
        return {
            "record_id": record_id,
            "suite_item_id": f"item-{record_id}",
            "outcome": outcome,
            "base": {"passed": outcome != "improvement", "score": 0.5},
            "candidate": {
                "record_id": record_id,
                "suite_item_id": f"item-{record_id}",
                "input": f"input-{record_id}",
                "expected": f"expected-{record_id}",
                "passed": passed,
                "score": score,
                "metadata": {"reward": reward, **metadata},
            },
        }

    return {
        "base_evaluation_id": "eval-base",
        "candidate_evaluation_id": "eval-candidate",
        "suite_revision_id": "suite-rev-7",
        "sample_deltas": [
            sample(
                "r-regression",
                "regression",
                False,
                score=0.2,
                reward=0.4,
                task="math",
                category="algebra",
                failure_reason="wrong-answer",
            ),
            sample(
                "r-improvement",
                "improvement",
                True,
                score=0.8,
                reward=0.9,
                task="math",
                category="geometry",
            ),
            sample(
                "r-failure",
                "unchanged_failure",
                False,
                score=0.1,
                reward=0.2,
                task="math",
                category="algebra",
                failure_reason="wrong-answer",
            ),
            {
                **sample(
                    "r-disagreement",
                    "unchanged_pass",
                    True,
                    score=0.9,
                    reward=0.5,
                    task="code",
                ),
                "candidate": {
                    "record_id": "r-disagreement",
                    "suite_item_id": "item-r-disagreement",
                    "input": "compile",
                    "passed": True,
                    "score": 0.9,
                    "verifier_trace": {
                        "verdicts": {
                            "compiler": {"passed": True},
                            "tests": {"passed": False},
                        }
                    },
                    "metadata": {"reward": 0.5, "task": "code"},
                },
            },
        ],
    }


def test_review_preview_supports_all_selectors_filters_and_stable_exclusions():
    comparison = _comparison()
    filtered = preview_failure_mining(
        comparison,
        {
            "type": "candidate_failure",
            "task": "math",
            "category": "algebra",
            "failure_reason": "wrong-answer",
            "score": {"min": 0.0, "max": 0.3},
            "reward_range": {"min": 0.1, "max": 0.5},
        },
        exclusions=["r-failure"],
    )
    assert filtered.examined_count == 4
    assert filtered.matched_count == 2
    assert [item.record_id for item in filtered.selected] == ["r-regression"]
    assert [item.record_id for item in filtered.excluded] == ["r-failure"]
    assert filtered.to_dict()["selected_count"] == 1

    assert [
        item.record_id for item in preview_failure_mining(comparison, "regression").selected
    ] == ["r-regression"]
    assert [
        item.record_id for item in preview_failure_mining(comparison, "improvement").selected
    ] == ["r-improvement"]
    assert [
        item.record_id
        for item in preview_failure_mining(
            comparison, {"selectors": ["verifier_disagreement"]}
        ).selected
    ] == ["r-disagreement"]
    assert exclusions_hash(["b", "a", "b"]) == exclusions_hash(["a", "b"])


def test_selector_rejects_invalid_modes_and_ranges():
    with pytest.raises(VersionError, match="Unknown failure-mining"):
        FailureMiningSelector.from_value("automatic-retrain")
    with pytest.raises(VersionError, match="min_score"):
        FailureMiningSelector.from_value({"min_score": 2, "max_score": 1})


def test_failure_build_appends_child_with_lineage_assets_and_audit_provenance(tmp_path):
    asset = tmp_path / "image.bin"
    asset.write_bytes(b"immutable-image-payload")
    source_path = _jsonl(
        tmp_path / "source.jsonl",
        [{"image": asset.name, "prompt": "what?", "response": "original"}],
    )
    raw_rows = [{"image": asset.name, "prompt": "what?", "response": "original"}]
    rows = seed_record_identities(raw_rows, source_fingerprint="source-v1")
    snapshot = SourceSnapshot(
        spec=SourceSpec(kind="local", path=str(source_path)),
        records=raw_rows,
        fingerprint="source-v1",
        assets=fingerprint_assets(raw_rows, base_dir=tmp_path),
        size_bytes=source_path.stat().st_size,
        file_count=1,
    )
    store = VersionStore(tmp_path / "datasets")
    parent = store.publish(
        dataset_id="vlm-failures",
        recipe={
            "schema": "vlm",
            "steps": [{"kind": "limit", "count": 1}],
        },
        result=RecipeResult(records=rows, splits={"train": rows}),
        source=snapshot,
        materialize_assets=True,
    )
    parent_identity = store.load_lineage(parent.version_id)[0]
    comparison = {
        "base_evaluation_id": "eval-base",
        "candidate_evaluation_id": "eval-candidate",
        "suite_revision_id": "suite-r1",
        "sample_deltas": [
            {
                "record_id": parent_identity.record_id,
                "suite_item_id": "image-item",
                "outcome": "regression",
                "base": {"passed": True, "score": 1.0},
                "candidate": {
                    "record_id": parent_identity.record_id,
                    "suite_item_id": "image-item",
                    "passed": False,
                    "score": 0.0,
                    "input": "what?",
                    "expected": "original",
                },
            }
        ],
    }

    builder = FailureMiningBuilder(store)
    child = builder.build(
        parent_version_id=parent.version_id,
        comparison=comparison,
        selector="regression",
    )
    child_manifest = _read_json(Path(child.path) / "manifest.json")
    recipe = _read_json(Path(child.path) / "recipe.json")
    provenance = _read_json(Path(child.path) / "provenance.json")
    lineage = store.load_lineage(child.version_id)

    assert child_manifest["parent_version_id"] == parent.version_id
    assert child.row_count == 2
    assert child.split_counts == {"train": 2}
    assert child.materialized_assets is True
    assert store.verify(child.version_id, dataset_id=child.dataset_id)["valid"] is True
    assert len(list((Path(child.path) / "assets").iterdir())) == 1
    assert lineage[0].record_id == lineage[1].record_id == parent_identity.record_id
    assert lineage[1].instance_id != parent_identity.instance_id
    assert parent_identity.instance_id in lineage[1].parent_instance_ids
    assert lineage[1].operations[-1]["operation"] == "evaluation_failure_mining"

    audit = recipe["steps"][0]
    assert audit["evaluation_ids"] == ["eval-base", "eval-candidate"]
    assert audit["suite_revision_id"] == "suite-r1"
    assert audit["selector"]["modes"] == ["regression"]
    assert audit["original_record_ids"] == [parent_identity.record_id]
    assert audit["parent_version_id"] == parent.version_id
    assert audit["exclusions_hash"] == exclusions_hash([])
    assert provenance[0]["details"] == {key: audit[key] for key in provenance[0]["details"]}

    reused = builder.build(
        parent_version_id=parent.version_id,
        comparison=comparison,
        selector="regression",
    )
    assert reused.version_id == child.version_id
    assert reused.reused is True


def test_failure_only_child_uses_reviewed_external_canonical_record_and_id(tmp_path):
    source_path = _jsonl(tmp_path / "source.jsonl", [{"prompt": "parent", "response": "answer"}])
    rows = seed_record_identities(
        [{"prompt": "parent", "response": "answer"}],
        source_fingerprint="parent-source",
    )
    store = VersionStore(tmp_path / "datasets")
    parent = store.publish(
        dataset_id="sft-failures",
        recipe={"schema": "sft", "steps": [{"kind": "limit", "count": 1}]},
        result=RecipeResult(records=rows, splits={"train": rows}),
        source=SourceSnapshot(
            spec=SourceSpec(kind="local", path=str(source_path)),
            records=[{"prompt": "parent", "response": "answer"}],
            fingerprint="parent-source",
        ),
    )
    comparison = {
        "candidate_evaluation_id": "candidate-only",
        "suite_revision_id": "suite-external",
        "samples": [
            {
                "record_id": "benchmark-original-id",
                "suite_item_id": "external-item",
                "input": "ignored because canonical record is present",
                "expected": "ignored",
                "passed": False,
                "metadata": {
                    "canonical_record": {
                        "prompt": "reviewed failure",
                        "response": "reviewed answer",
                    }
                },
            }
        ],
    }
    child = FailureMiningBuilder(store).build(
        parent_version_id=parent.version_id,
        comparison=comparison,
        mode="replace",
        exclusions=[],
    )

    assert store.load_records(child.version_id) == [
        {"prompt": "reviewed failure", "response": "reviewed answer"}
    ]
    identity = store.load_lineage(child.version_id)[0]
    assert identity.record_id == "benchmark-original-id"
    assert identity.operations[-1]["candidate_evaluation_id"] == "candidate-only"
    assert child.row_count == 1
    assert child.split_counts == {"train": 1}


def test_failure_build_moves_selected_identity_out_of_heldout_split(tmp_path):
    source_path = _jsonl(
        tmp_path / "source.jsonl", [{"prompt": "held out", "response": "answer"}]
    )
    rows = seed_record_identities(
        [{"prompt": "held out", "response": "answer"}],
        source_fingerprint="heldout-source",
    )
    store = VersionStore(tmp_path / "datasets")
    parent = store.publish(
        dataset_id="heldout-failures",
        recipe={"schema": "sft", "steps": [{"kind": "limit", "count": 1}]},
        result=RecipeResult(records=rows, splits={"train": [], "test": rows}),
        source=SourceSnapshot(
            spec=SourceSpec(kind="local", path=str(source_path)),
            records=[{"prompt": "held out", "response": "answer"}],
            fingerprint="heldout-source",
        ),
    )
    record_id = store.load_lineage(parent.version_id, split="test")[0].record_id
    child = FailureMiningBuilder(store).build(
        parent_version_id=parent.version_id,
        comparison={
            "candidate_evaluation_id": "candidate-heldout",
            "suite_revision_id": "suite-heldout",
            "samples": [
                {
                    "record_id": record_id,
                    "suite_item_id": "heldout-item",
                    "input": "held out",
                    "expected": "answer",
                    "passed": False,
                }
            ],
        },
    )

    assert child.split_counts == {"test": 0, "train": 1}
    provenance = _read_json(Path(child.path) / "provenance.json")
    assert provenance[0]["details"]["moved_from_splits"] == {"test": 1}


def test_failure_build_requires_a_nonempty_review_selection(tmp_path):
    source_path = _jsonl(tmp_path / "source.jsonl", [{"prompt": "p"}])
    rows = seed_record_identities([{"prompt": "p"}])
    store = VersionStore(tmp_path / "datasets")
    parent = store.publish(
        dataset_id="empty",
        recipe={"steps": [{"kind": "limit", "count": 1}]},
        result=RecipeResult(records=rows, splits={"train": rows}),
        source=SourceSnapshot(
            spec=SourceSpec(kind="local", path=str(source_path)),
            records=[{"prompt": "p"}],
            fingerprint="source",
        ),
    )
    with pytest.raises(VersionError, match="selected no records"):
        FailureMiningBuilder(store).build(
            parent_version_id=parent.version_id,
            comparison=[{"record_id": "pass", "passed": True}],
        )
