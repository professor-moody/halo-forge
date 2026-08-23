"""Focused contracts for Dataset Lab v2 identity and trainer artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from halo_forge.data_lab import (
    TRAINER_DATASET_ADAPTERS,
    DatasetBinding,
    DatasetLab,
    DatasetVersionComparator,
    TrainingArtifactRenderer,
    VersionError,
    VersionStore,
)
from halo_forge.data_lab.identity import (
    INTERNAL_LINEAGE_KEY,
    deterministic_record_id,
    seed_record_identities,
)
from halo_forge.data_lab.recipe import RecipeResult
from halo_forge.data_lab.sources import SourceSnapshot, SourceSpec


def _jsonl(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class WordTokenizer:
    chat_template = (
        "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}"
    )

    def encode(self, text, add_special_tokens=True):
        values = str(text).split()
        return ([0] if add_special_tokens else []) + list(range(len(values)))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt=False):
        text = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
        return self.encode(text) if tokenize else text


def test_new_versions_publish_stable_lineage_without_polluting_canonical_rows(tmp_path):
    source_path = _jsonl(
        tmp_path / "source.jsonl",
        [
            {"question": "  One  ", "answer": "1"},
            {"question": "Two", "answer": "2"},
        ],
    )
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {
            "kind": "local",
            "path": str(source_path),
            "canonical_kind": "sft",
            "field_mapping": {"prompt": "question", "response": "answer"},
        },
        dataset_id="identity",
    )
    version = lab.build(
        source.id,
        {
            "steps": [
                {"kind": "normalize", "fields": ["prompt"]},
                {"kind": "shuffle", "seed": 9},
                {"kind": "split", "ratios": {"train": 0.5, "validation": 0.5}},
            ]
        },
    )
    manifest = json.loads((Path(version.path) / "manifest.json").read_text())
    rows = _read_jsonl(Path(version.path) / "records.jsonl")
    lineage = lab.store.load_lineage(version.version_id)

    assert manifest["format_version"] == 2
    assert manifest["lineage"]["path"] == "lineage.jsonl"
    assert all(INTERNAL_LINEAGE_KEY not in row for row in rows)
    assert len(lineage) == len(rows) == 2
    assert len({item.instance_id for item in lineage}) == 2
    assert all(item.record_id.startswith("rec_") for item in lineage)
    assert all(item.record_hash.startswith("") and len(item.record_hash) == 64 for item in lineage)
    assert {split for item in lineage for split in item.splits} == {"train", "validation"}

    reused = lab.build(
        source.id,
        {
            "steps": [
                {"kind": "normalize", "fields": ["prompt"]},
                {"kind": "shuffle", "seed": 9},
                {"kind": "split", "ratios": {"train": 0.5, "validation": 0.5}},
            ]
        },
    )
    assert reused.version_id == version.version_id
    assert reused.reused is True
    assert [item.to_dict() for item in lab.store.load_lineage(version.version_id)] == [
        item.to_dict() for item in lineage
    ]
    identified = lab.store.load_identified_records(version.version_id)
    assert identified[0]["identity"]["record_id"] == lineage[0].record_id
    assert identified[0]["record"] == rows[0]
    lab.close()


def test_legacy_versions_receive_virtual_identity_without_filesystem_changes(tmp_path):
    root = tmp_path / "lab"
    version_path = root / "legacy" / "old-version"
    (version_path / "splits").mkdir(parents=True)
    rows = [{"prompt": "a"}, {"prompt": "b"}]
    _jsonl(version_path / "records.jsonl", rows)
    _jsonl(version_path / "splits" / "train.jsonl", rows)
    (version_path / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "status": "complete",
                "dataset_id": "legacy",
                "version_id": "old-version",
                "created_at": "2026-01-01T00:00:00+00:00",
                "content_hash": "content",
                "recipe_hash": "recipe",
                "source_fingerprint": "source",
                "schema": "prompt",
                "materialized_assets": False,
                "row_count": 2,
                "split_counts": {"train": 2},
                "artifact_hashes": {},
            }
        )
    )
    store = VersionStore(root)
    before = sorted(path.relative_to(version_path).as_posix() for path in version_path.rglob("*"))
    first = store.load_lineage("old-version", dataset_id="legacy")
    second = store.load_lineage("old-version", dataset_id="legacy")
    after = sorted(path.relative_to(version_path).as_posix() for path in version_path.rglob("*"))

    assert all(item.virtual for item in first)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert before == after
    attached = store.load_records_with_lineage("old-version", dataset_id="legacy")
    assert all(INTERNAL_LINEAGE_KEY in row for row in attached)


def test_source_provided_id_survives_payload_changes():
    assert deterministic_record_id({"id": "row-7", "prompt": "before"}) == (
        deterministic_record_id({"id": "row-7", "prompt": "after"})
    )


def test_duplicate_occurrences_receive_distinct_split_instances(tmp_path):
    source_path = _jsonl(tmp_path / "source.jsonl", [{"prompt": "duplicate"}])
    snapshot = SourceSnapshot(SourceSpec("local", path=str(source_path)), [], "source-fingerprint")
    store = VersionStore(tmp_path / "lab")
    rows = seed_record_identities(
        [{"prompt": "duplicate"}, {"prompt": "duplicate"}],
        source_fingerprint="source-fingerprint",
    )
    version = store.publish(
        dataset_id="duplicates",
        recipe={"steps": [{"kind": "limit", "count": 2}]},
        result=RecipeResult(records=rows, splits={"train": [rows[0]], "validation": [rows[1]]}),
        source=snapshot,
    )
    train = store.load_lineage(version.version_id, split="train")
    validation = store.load_lineage(version.version_id, split="validation")
    assert train[0].record_id == validation[0].record_id
    assert train[0].instance_id != validation[0].instance_id


def test_synthesis_and_mixtures_preserve_parent_identity(tmp_path):
    lab = DatasetLab(tmp_path / "lab", teacher=lambda prompt, *_: f"answer:{prompt}")
    seed_source = lab.add_source(
        {
            "kind": "local",
            "path": str(_jsonl(tmp_path / "seed.jsonl", [{"prompt": "seed"}])),
        },
        dataset_id="seed",
    )
    synthesized = lab.build(
        seed_source.id,
        {
            "steps": [
                {"kind": "synthesize", "teacher_model": "teacher", "n_per_record": 2},
                {"kind": "split", "ratios": {"train": 1}},
            ]
        },
    )
    synthesis_lineage = lab.store.load_lineage(synthesized.version_id)
    assert len({value.record_id for value in synthesis_lineage}) == 1
    assert len({value.instance_id for value in synthesis_lineage}) == 2
    assert [value.operations[-1]["generation_index"] for value in synthesis_lineage] == [
        0,
        1,
    ]

    current_source = lab.add_source(
        {
            "kind": "local",
            "path": str(_jsonl(tmp_path / "current.jsonl", [{"prompt": "current"}])),
        },
        dataset_id="mixed",
    )
    mixed = lab.build(
        current_source.id,
        {
            "steps": [
                {
                    "kind": "mix",
                    "datasets": [
                        {"source": "current", "weight": 1},
                        {"source": synthesized.version_id, "weight": 1},
                    ],
                    "size": 2,
                },
                {"kind": "split", "ratios": {"train": 1}},
            ]
        },
    )
    identified = lab.store.load_identified_records(mixed.version_id)
    imported = next(item for item in identified if item["record"]["prompt"] == "seed")
    assert imported["identity"]["parent_instance_ids"]
    lab.close()


def test_version_comparison_uses_origin_identity_for_all_change_classes(tmp_path):
    source_path = _jsonl(tmp_path / "source.jsonl", [{"prompt": "placeholder"}])
    snapshot = SourceSnapshot(
        SourceSpec("local", path=str(source_path)),
        [],
        "same-source",
    )
    store = VersionStore(tmp_path / "lab")
    source_rows = seed_record_identities(
        [
            {"prompt": " A ", "response": "one"},
            {"prompt": "B", "response": "two"},
            {"prompt": "removed", "response": "three"},
        ],
        source_fingerprint="same-source",
    )
    left_rows = [dict(row) for row in source_rows]
    left = store.publish(
        dataset_id="compare",
        recipe={"steps": [{"kind": "limit", "count": 99}]},
        result=RecipeResult(
            records=left_rows,
            splits={"train": left_rows[:2], "test": left_rows[2:]},
        ),
        source=snapshot,
    )

    right_a = dict(source_rows[0])
    right_a["prompt"] = "A"
    right_b = dict(source_rows[1])
    right_b_duplicate = dict(source_rows[1])
    added = seed_record_identities(
        [{"prompt": "added", "response": "four"}], source_fingerprint="same-source"
    )[0]
    right_rows = [right_a, right_b, right_b_duplicate, added]
    right = store.publish(
        dataset_id="compare",
        recipe={"steps": [{"kind": "shuffle", "seed": 3}]},
        result=RecipeResult(
            records=right_rows,
            splits={"validation": [right_a], "train": [right_b, right_b_duplicate, added]},
        ),
        source=snapshot,
    )

    comparison = DatasetVersionComparator(store).compare(left.version_id, right.version_id)
    assert comparison.summary == {
        "left_records": 3,
        "right_records": 4,
        "added": 1,
        "removed": 1,
        "content_changed": 1,
        "repeated": 1,
        "moved_between_splits": 1,
    }
    assert comparison.recipe["changed"] is True
    assert comparison.source_contributions["changed"] is False


@pytest.mark.parametrize(
    ("schema", "mode", "adapter_id"),
    [
        ("sft", "sft", "sft"),
        ("chat", "sft", "chat"),
        ("tool", "sft", "tool"),
        ("preference", "dpo", "preference"),
        ("preference", "orpo", "preference"),
        ("preference", "rm", "preference"),
        ("prompt", "raft", "prompt"),
        ("rlvr", "grpo", "prompt"),
        ("prompt", "reasoning", "prompt"),
        ("vlm", "vlm", "vlm"),
        ("audio", "audio", "audio"),
        ("tool", "agentic", "agentic"),
    ],
)
def test_trainer_adapter_registry_is_schema_and_mode_specific(schema, mode, adapter_id):
    assert TRAINER_DATASET_ADAPTERS.resolve(schema=schema, trainer_mode=mode).id == adapter_id


def test_dataset_binding_parses_cli_shape():
    assert DatasetBinding.from_value("validation=version-1:validation") == DatasetBinding(
        "validation", "version-1", "validation"
    )


def test_adapters_emit_current_trainer_local_shapes():
    tokenizer = WordTokenizer()
    sft = TRAINER_DATASET_ADAPTERS.get("sft").render_record(
        {"system": "rules", "prompt": "question", "response": "answer"},
        tokenizer=tokenizer,
    )
    chat = TRAINER_DATASET_ADAPTERS.get("chat").render_record(
        {"messages": [{"role": "user", "content": "hello"}]}, tokenizer=tokenizer
    )
    preference = TRAINER_DATASET_ADAPTERS.get("preference").render_record(
        {"prompt": "p", "chosen": "yes", "rejected": "no"}
    )
    prompt = TRAINER_DATASET_ADAPTERS.get("prompt").render_record(
        {"prompt": "p", "reference_answer": "a"}
    )
    vlm = TRAINER_DATASET_ADAPTERS.get("vlm").render_record(
        {"image": "image.png", "prompt": "p", "response": "a"}
    )
    audio = TRAINER_DATASET_ADAPTERS.get("audio").render_record(
        {"audio": "audio.wav", "task": "asr", "transcript": "words"}
    )
    agentic = TRAINER_DATASET_ADAPTERS.get("agentic").render_record(
        {
            "messages": [{"role": "user", "content": "call"}],
            "tools": [{"type": "function"}],
            "expected_calls": [{"name": "tool"}],
        }
    )

    assert sft["text"] == "rules\nquestion\nanswer"
    assert chat["text"] == "user: hello"
    assert preference == {"prompt": "p", "chosen": "yes", "rejected": "no"}
    assert prompt == {"prompt": "p", "reference_answer": "a"}
    assert vlm["ground_truth"] == "a"
    assert audio["audio_path"] == "audio.wav" and audio["text"] == "words"
    assert agentic["expected_calls"] == [{"name": "tool"}]


def test_training_artifact_preserves_validation_and_hides_test_canary(tmp_path):
    rows = [{"prompt": f"prompt {index}", "response": f"answer {index}"} for index in range(10)]
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {
            "kind": "local",
            "path": str(_jsonl(tmp_path / "sft.jsonl", rows)),
            "canonical_kind": "sft",
        },
        dataset_id="artifact",
    )
    version = lab.build(
        source.id,
        {
            "seed": 7,
            "steps": [
                {
                    "kind": "split",
                    "seed": 7,
                    "ratios": {"train": 0.6, "validation": 0.2, "test": 0.1, "canary": 0.1},
                }
            ],
        },
    )
    bindings = [DatasetBinding("train", version.version_id, "train")]
    artifact = lab.render_training_artifact(
        bindings,
        trainer_mode="sft",
        model="local/model",
        tokenizer_revision="revision-1",
        tokenizer=WordTokenizer(),
    )

    assert artifact.row_counts == {"train": 6, "validation": 2, "test": 1, "canary": 1}
    assert {binding.role for binding in artifact.bindings} == {
        "train",
        "validation",
        "test",
        "canary",
    }
    assert set(artifact.split_paths) == {"train", "validation"}
    assert artifact.validation_policy == {"kind": "supplied", "preserved": True, "row_count": 2}
    assert artifact.token_statistics["exact"] is True
    assert not (Path(artifact.path) / "splits" / "test.jsonl").exists()
    assert not (Path(artifact.path) / "splits" / "canary.jsonl").exists()
    supplied = lab.store.load_records(version.version_id, split="validation")
    rendered_validation = _read_jsonl(Path(artifact.split_paths["validation"]))
    assert [row["prompt"] for row in rendered_validation] == [row["prompt"] for row in supplied]
    assert all("text" in row for row in rendered_validation)

    reused = lab.render_training_artifact(
        bindings,
        trainer_mode="sft",
        model="local/model",
        tokenizer_revision="revision-1",
        tokenizer=WordTokenizer(),
    )
    assert reused.artifact_hash == artifact.artifact_hash
    assert reused.reused is True
    Path(artifact.split_paths["train"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(VersionError, match="changed after publication"):
        lab.render_training_artifact(
            bindings,
            trainer_mode="sft",
            model="local/model",
            tokenizer_revision="revision-1",
            tokenizer=WordTokenizer(),
        )
    lab.close()


def test_training_artifact_can_run_as_persistent_data_job_without_changing_sync_api(
    tmp_path,
):
    rows = [
        {"prompt": f"prompt {index}", "response": f"answer {index}"}
        for index in range(4)
    ]
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {
            "kind": "local",
            "path": str(_jsonl(tmp_path / "async-sft.jsonl", rows)),
            "canonical_kind": "sft",
        },
        dataset_id="async-artifact",
    )
    version = lab.build(
        source.id,
        {"steps": [{"kind": "split", "ratios": {"train": 1.0}}]},
    )
    binding = DatasetBinding("train", version.version_id, "train")

    started = lab.start_training_artifact_job(
        [binding], trainer_mode="sft", seed=17, validation_fraction=0.25
    )
    assert started.kind == "training_artifact"
    assert started.payload["bindings"] == [binding.to_dict()]
    completed = lab.job_manager.wait(started.id, timeout=5)

    assert completed.status == "succeeded", completed.to_dict()
    assert completed.stage == "complete"
    assert completed.result["artifact_id"]
    assert completed.accepted == 4
    assert any("Published training artifact" in line for line in completed.logs)
    artifact = lab.training_artifacts.get(completed.result["artifact_id"])
    assert set(artifact.split_paths) == {"train", "validation"}

    # CLI and internal train preparation retain the original blocking method,
    # which now reuses the exact content-addressed result from the job.
    synchronous = lab.render_training_artifact(
        [binding], trainer_mode="sft", seed=17, validation_fraction=0.25
    )
    assert synchronous.artifact_id == artifact.artifact_id
    assert synchronous.reused is True
    lab.close()


def test_training_artifact_derives_validation_deterministically_without_mutating_version(tmp_path):
    rows = [{"prompt": f"p{index}", "reference_answer": str(index)} for index in range(10)]
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {
            "kind": "local",
            "path": str(_jsonl(tmp_path / "prompt.jsonl", rows)),
            "canonical_kind": "prompt",
        }
    )
    version = lab.build(source.id, {"steps": [{"kind": "split", "ratios": {"train": 1}}]})
    original = Path(version.path, "splits", "train.jsonl").read_bytes()
    renderer = TrainingArtifactRenderer(lab.store)
    first = renderer.render(
        [DatasetBinding("train", version.version_id, "train")],
        trainer_mode="grpo",
        validation_fraction=0.2,
        seed=11,
    )
    second = renderer.render(
        [DatasetBinding("train", version.version_id, "train")],
        trainer_mode="grpo",
        validation_fraction=0.2,
        seed=11,
    )
    assert first.row_counts["train"] == 8
    assert first.row_counts["validation"] == 2
    assert first.validation_policy["selection"] == "sha256(seed,instance_id)"
    assert first.artifact_hash == second.artifact_hash
    assert Path(version.path, "splits", "train.jsonl").read_bytes() == original
    assert first.token_statistics["exact"] is False
    lab.close()


def test_streaming_renderer_avoids_bulk_version_loaders_and_is_deterministic(
    tmp_path, monkeypatch
):
    rows = [
        {"prompt": f"prompt {index}", "response": f"answer {index}"}
        for index in range(41)
    ]
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {
            "kind": "local",
            "path": str(_jsonl(tmp_path / "streaming.jsonl", rows)),
            "canonical_kind": "sft",
        }
    )
    version = lab.build(
        source.id, {"steps": [{"kind": "split", "ratios": {"train": 1}}]}
    )

    def reject_bulk(*_args, **_kwargs):
        raise AssertionError("training artifacts must use VersionStore iterators")

    monkeypatch.setattr(lab.store, "load_records", reject_bulk)
    monkeypatch.setattr(lab.store, "load_lineage", reject_bulk)
    binding = DatasetBinding("train", version.version_id, "train")
    first = TrainingArtifactRenderer(
        lab.store, root=tmp_path / "artifacts-first"
    ).render(
        [binding],
        trainer_mode="sft",
        tokenizer=WordTokenizer(),
        validation_fraction=0.2,
        seed=73,
    )
    second = TrainingArtifactRenderer(
        lab.store, root=tmp_path / "artifacts-second"
    ).render(
        [binding],
        trainer_mode="sft",
        tokenizer=WordTokenizer(),
        validation_fraction=0.2,
        seed=73,
    )

    assert first.reused is False and second.reused is False
    assert first.artifact_hash == second.artifact_hash
    assert first.row_counts == second.row_counts == {
        "train": 33,
        "validation": 8,
        "test": 0,
        "canary": 0,
    }
    for role in ("train", "validation"):
        assert Path(first.split_paths[role]).read_bytes() == Path(
            second.split_paths[role]
        ).read_bytes()
    assert json.loads(Path(first.path, "lineage.json").read_text()) == json.loads(
        Path(second.path, "lineage.json").read_text()
    )
    lab.close()


def test_streaming_renderer_preserves_supplied_validation_and_hides_heldout(
    tmp_path, monkeypatch
):
    rows = [
        {"prompt": f"prompt {index}", "response": f"answer {index}"}
        for index in range(20)
    ]
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {
            "kind": "local",
            "path": str(_jsonl(tmp_path / "supplied.jsonl", rows)),
            "canonical_kind": "sft",
        }
    )
    version = lab.build(
        source.id,
        {
            "seed": 5,
            "steps": [
                {
                    "kind": "split",
                    "seed": 5,
                    "ratios": {
                        "train": 0.6,
                        "validation": 0.2,
                        "test": 0.1,
                        "canary": 0.1,
                    },
                }
            ],
        },
    )
    expected_validation = _read_jsonl(
        Path(version.path, "splits", "validation.jsonl")
    )

    def reject_bulk(*_args, **_kwargs):
        raise AssertionError("training artifacts must use VersionStore iterators")

    monkeypatch.setattr(lab.store, "load_records", reject_bulk)
    monkeypatch.setattr(lab.store, "load_lineage", reject_bulk)
    artifact = TrainingArtifactRenderer(
        lab.store, root=tmp_path / "artifacts"
    ).render(
        [DatasetBinding("train", version.version_id, "train")],
        trainer_mode="sft",
        tokenizer=WordTokenizer(),
    )

    rendered_validation = _read_jsonl(Path(artifact.split_paths["validation"]))
    assert [row["prompt"] for row in rendered_validation] == [
        row["prompt"] for row in expected_validation
    ]
    assert artifact.validation_policy == {
        "kind": "supplied",
        "preserved": True,
        "row_count": 4,
    }
    assert artifact.row_counts["test"] == artifact.row_counts["canary"] == 2
    assert set(artifact.split_paths) == {"train", "validation"}
    assert not Path(artifact.path, "splits", "test.jsonl").exists()
    assert not Path(artifact.path, "splits", "canary.jsonl").exists()
    lab.close()


def test_training_artifact_refuses_manifest_identity_tampering(tmp_path):
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {
            "kind": "local",
            "path": str(
                _jsonl(
                    tmp_path / "sft-integrity.jsonl",
                    [{"prompt": "question", "response": "answer"}],
                )
            ),
            "canonical_kind": "sft",
        }
    )
    version = lab.build(source.id, {"steps": [{"kind": "split", "ratios": {"train": 1}}]})
    binding = DatasetBinding("train", version.version_id, "train")
    artifact = lab.render_training_artifact([binding], trainer_mode="sft")
    manifest_path = Path(artifact.path) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["adapter_version"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(VersionError, match="manifest identity changed"):
        lab.training_artifacts.get(artifact.artifact_id)
    with pytest.raises(VersionError, match="manifest identity changed"):
        lab.render_training_artifact([binding], trainer_mode="sft")
    lab.close()


def test_multimodal_artifact_resolves_and_verifies_local_assets(tmp_path):
    image = tmp_path / "image.bin"
    image.write_bytes(b"original-image")
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {
            "kind": "local",
            "path": str(
                _jsonl(
                    tmp_path / "vlm.jsonl",
                    [{"image": image.name, "prompt": "what", "response": "answer"}],
                )
            ),
            "canonical_kind": "vlm",
        }
    )
    version = lab.build(source.id, {"steps": [{"kind": "split", "ratios": {"train": 1}}]})
    binding = DatasetBinding("train", version.version_id, "train")
    artifact = lab.render_training_artifact([binding], trainer_mode="vlm")
    rendered = _read_jsonl(Path(artifact.split_paths["train"]))[0]
    assert rendered["image"] == str(image.resolve())
    assert artifact.asset_roots == (str(tmp_path),)

    image.write_bytes(b"changed-image")
    with pytest.raises(VersionError, match="changed after version publication"):
        lab.render_training_artifact([binding], trainer_mode="vlm", seed=99)
    lab.close()


def test_held_out_split_cannot_be_exposed_as_train(tmp_path):
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {
            "kind": "local",
            "path": str(_jsonl(tmp_path / "source.jsonl", [{"prompt": "p", "response": "r"}])),
            "canonical_kind": "sft",
        }
    )
    version = lab.build(source.id, {"steps": [{"kind": "split", "ratios": {"test": 1}}]})
    with pytest.raises(VersionError, match="cannot be exposed"):
        lab.render_training_artifact(
            [DatasetBinding("train", version.version_id, "test")], trainer_mode="sft"
        )
    lab.close()


def test_reasoning_artifact_requires_reference_answers(tmp_path):
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {
            "kind": "local",
            "path": str(_jsonl(tmp_path / "prompts.jsonl", [{"prompt": "solve"}])),
            "canonical_kind": "prompt",
        }
    )
    version = lab.build(
        source.id, {"steps": [{"kind": "split", "ratios": {"train": 1}}]}
    )
    with pytest.raises(VersionError, match="requires reference_answer"):
        lab.render_training_artifact(
            [DatasetBinding("train", version.version_id, "train")],
            trainer_mode="reasoning",
        )
    lab.close()


def test_artifact_refuses_rows_that_do_not_match_declared_schema(tmp_path):
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {
            "kind": "local",
            "path": str(_jsonl(tmp_path / "invalid.jsonl", [{"prompt": "missing response"}])),
        }
    )
    version = lab.build(
        source.id,
        {
            "schema": "sft",
            "steps": [{"kind": "split", "ratios": {"train": 1}}],
        },
    )
    with pytest.raises(VersionError, match="incompatible with adapter"):
        lab.render_training_artifact(
            [DatasetBinding("train", version.version_id, "train")], trainer_mode="sft"
        )
    lab.close()
