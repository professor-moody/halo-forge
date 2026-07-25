from __future__ import annotations

import json
import sys
import threading
import time
import types
import wave
from pathlib import Path

import pytest

from halo_forge.data_lab import (
    CanonicalKind,
    DatasetLab,
    Recipe,
    RecipeContext,
    RecipeError,
    RecipeRunner,
    SchemaError,
    SerialJobManager,
    SourceError,
    SourceSpec,
    VersionError,
    adapt_record,
    configured_teacher,
    configured_verifier,
    infer_schema,
    load_source,
    profile_records,
)


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("kind", "source", "mapping", "expected"),
    [
        ("sft", {"q": "Q", "a": "A"}, {"prompt": "q", "response": "a"}, {"prompt", "response"}),
        ("chat", {"conversations": [{"from": "user", "value": "hi"}]}, None, {"messages"}),
        ("preference", {"prompt": "Q", "chosen": "A", "rejected": "B"}, None, {"prompt", "chosen", "rejected"}),
        ("reasoning", {"problem": "Q", "solution": "A"}, None, {"prompt", "reference_answer"}),
        ("agentic", {"messages": [{"role": "user", "content": "x"}], "functions": [{"name": "f"}]}, None, {"messages", "tools"}),
        ("vlm", {"image_path": "a.png", "question": "Q", "answer": "A"}, None, {"image", "prompt", "response"}),
        ("audio", {"audio_path": "a.wav", "instruction": "transcribe", "text": "hello"}, None, {"audio", "task", "transcript"}),
    ],
)
def test_canonical_adapters(kind, source, mapping, expected):
    adapted = adapt_record(source, kind, mapping=mapping)
    assert expected <= set(adapted)


def test_schema_inference_and_validation_errors():
    assert infer_schema({"image": "x", "prompt": "q"}) is CanonicalKind.VLM
    assert infer_schema({"messages": [], "tools": []}) is CanonicalKind.TOOL
    with pytest.raises(SchemaError, match="response"):
        adapt_record({"instruction": "Q"}, "sft")


def test_tool_schema_accepts_empty_assistant_content_when_a_tool_call_is_present():
    adapted = adapt_record(
        {
            "messages": [
                {"role": "user", "content": "Weather in Austin?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"name": "weather", "arguments": {"city": "Austin"}}
                    ],
                },
            ],
            "tools": [{"name": "weather", "parameters": {"type": "object"}}],
        },
        "tool",
    )

    assert adapted["messages"][1]["content"] == ""
    assert adapted["messages"][1]["tool_calls"][0]["name"] == "weather"

    with pytest.raises(SchemaError, match="tool-call turns"):
        adapt_record(
            {
                "messages": [{"role": "assistant", "content": ""}],
                "tools": [{"name": "weather"}],
            },
            "tool",
        )


def test_load_local_json_jsonl_csv_and_changed_fingerprint(tmp_path):
    json_path = tmp_path / "a.json"
    json_path.write_text(json.dumps({"data": [{"prompt": "one"}]}))
    _jsonl(tmp_path / "b.jsonl", [{"prompt": "two"}])
    (tmp_path / "c.csv").write_text("prompt,response\nthree,3\n")
    snapshot = load_source(SourceSpec(kind="local", path=str(tmp_path)))
    assert len(snapshot.records) == 3
    assert {row["_source_file"] for row in snapshot.records} == {"a.json", "b.jsonl", "c.csv"}
    original = snapshot.fingerprint
    (tmp_path / "c.csv").write_text("prompt,response\nchanged,3\n")
    assert load_source(snapshot.spec).fingerprint != original


def test_huggingface_loader_is_lazy_and_requires_pinned_revision(monkeypatch):
    with pytest.raises(SourceError, match="pinned"):
        SourceSpec(kind="huggingface", repo_id="owner/data", revision="main")

    class FakeDataset(list):
        _fingerprint = "upstream-fingerprint"

    calls = []
    module = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: calls.append((args, kwargs)) or FakeDataset([{"prompt": "Q"}])
    )
    monkeypatch.setitem(sys.modules, "datasets", module)
    snapshot = load_source(
        SourceSpec(kind="huggingface", repo_id="owner/data", config="v1", split="test", revision="abc123")
    )
    assert snapshot.records == [{"prompt": "Q"}]
    assert calls == [(('owner/data', 'v1'), {'split': 'test', 'revision': 'abc123'})]


def test_text_image_audio_profiles(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "pixel.png"
    pil.new("RGB", (12, 8), color="red").save(image_path)
    audio_path = tmp_path / "tone.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\0\0" * 800)
    result = profile_records(
        [{"image": image_path.name, "audio": audio_path.name, "prompt": "hello world"}],
        base_dir=tmp_path,
    )
    assert result["text"]["words"]["mean"] == 2
    assert result["image"]["width"]["mean"] == 12
    assert result["audio"]["sample_rates"] == {"8000": 1}
    assert result["audio"]["total_duration_seconds"] == pytest.approx(0.1)


def test_recipe_mapping_filter_dedup_curriculum_and_deterministic_group_split():
    rows = [
        {"q": " Same  prompt ", "a": "alpha", "difficulty": 0.1, "group": "media-a", "keep": True},
        {"q": "same prompt", "a": "duplicate", "difficulty": 0.2, "group": "media-a", "keep": True},
        {"q": "Different prompt", "a": "beta", "difficulty": 0.9, "group": "media-b", "keep": True},
        {"q": "Filtered", "a": "gamma", "difficulty": 0.4, "group": "media-c", "keep": False},
    ]
    recipe = {
        "schema": "sft",
        "seed": 41,
        "steps": [
            {"kind": "map", "fields": {"prompt": "q", "response": "a"}},
            {"kind": "normalize", "fields": ["prompt"]},
            {"kind": "filter", "field": "metadata.keep", "op": "eq", "value": True},
            {"kind": "dedup", "method": "exact", "field": "prompt"},
            {"kind": "curriculum", "field": "metadata.difficulty", "boundaries": [0.5], "labels": ["easy", "hard"]},
            {"kind": "split", "method": "grouped", "group_field": "metadata.group", "ratios": {"train": 0.5, "test": 0.5}},
            {"kind": "contamination", "field": "prompt"},
        ],
    }
    first = RecipeRunner().run(rows, recipe)
    second = RecipeRunner().run(rows, recipe)
    assert first.splits == second.splits
    assert len(first.records) == 2
    assert len(first.rejected) == 2
    membership = {
        row["metadata"]["group"]: split
        for split, split_rows in first.splits.items()
        for row in split_rows
    }
    assert len(membership) == 2
    assert all("curriculum" in row["metadata"] for row in first.records)
    assert sum(pair["count"] for pair in first.contamination["pairs"].values()) == 0


def test_grouped_split_puts_a_single_asset_group_in_the_primary_split():
    result = RecipeRunner().run(
        [{"image": "sample.png", "prompt": "Describe it", "response": "A sample."}],
        {
            "seed": 42,
            "steps": [
                {
                    "kind": "split",
                    "method": "grouped",
                    "group_field": "image",
                    "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
                }
            ],
        },
    )

    assert len(result.splits["train"]) == 1
    assert result.splits["validation"] == []
    assert result.splits["test"] == []


def test_asset_hash_grouping_keeps_identical_bytes_with_different_paths_together(
    tmp_path: Path,
):
    (tmp_path / "copy-a.png").write_bytes(b"identical-media")
    (tmp_path / "copy-b.png").write_bytes(b"identical-media")
    (tmp_path / "unique-c.png").write_bytes(b"unique-c")
    (tmp_path / "unique-d.png").write_bytes(b"unique-d")
    records = [
        {"image": name, "prompt": name, "response": "ok"}
        for name in ("copy-a.png", "copy-b.png", "unique-c.png", "unique-d.png")
    ]
    result = RecipeRunner(RecipeContext(base_dir=tmp_path)).run(
        records,
        {
            "seed": 42,
            "steps": [
                {
                    "kind": "split",
                    "method": "grouped",
                    "group_field": "image",
                    "group_by_asset_hash": True,
                    "ratios": {"train": 0.5, "validation": 0.25, "test": 0.25},
                }
            ],
        },
    )

    membership = {
        row["image"]: split
        for split, rows in result.splits.items()
        for row in rows
    }
    assert membership["copy-a.png"] == membership["copy-b.png"]
    assert {len(rows) for rows in result.splits.values()} == {1, 2}


def test_exact_contamination_indexes_each_record_once(monkeypatch: pytest.MonkeyPatch):
    from halo_forge.data_lab import recipe as recipe_module

    original_text = recipe_module._text
    calls = 0

    def counted_text(row, field_name=None):
        nonlocal calls
        calls += 1
        return original_text(row, field_name)

    monkeypatch.setattr(recipe_module, "_text", counted_text)
    rows = [{"prompt": f"unique-{index}"} for index in range(5_000)]
    result = RecipeRunner().run(
        rows,
        {
            "seed": 42,
            "steps": [
                {
                    "kind": "split",
                    "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
                },
                {"kind": "contamination", "method": "exact", "field": "prompt"},
            ],
        },
    )

    assert calls == len(rows)
    assert sum(pair["count"] for pair in result.contamination["pairs"].values()) == 0


def test_indexed_exact_contamination_retains_pair_counts_and_train_removal():
    rows = [
        {"prompt": "duplicate", "order": 1},
        {"prompt": "train-only", "order": 2},
        {"prompt": "duplicate", "order": 3},
        {"prompt": "test-only", "order": 4},
    ]
    result = RecipeRunner().run(
        rows,
        {
            "steps": [
                {
                    "kind": "split",
                    "method": "time",
                    "time_field": "order",
                    "ratios": {"train": 0.5, "test": 0.5},
                },
                {
                    "kind": "contamination",
                    "method": "exact",
                    "field": "prompt",
                    "action": "remove",
                },
            ]
        },
    )

    assert result.contamination["pairs"]["train:test"] == {
        "count": 1,
        "matches": [{"left_index": 0, "right_index": 0}],
    }
    assert result.contamination["removed_from_train"] == 1
    assert [row["prompt"] for row in result.splits["train"]] == ["train-only"]


def test_fuzzy_semantic_mix_failure_mining_and_synthesis(tmp_path):
    failures = _jsonl(
        tmp_path / "failures.jsonl",
        [{"prompt": "failed", "success": False}, {"prompt": "passed", "success": True}],
    )
    context = RecipeContext(
        teacher=lambda prompt: f"answer:{prompt}",
        verifier=lambda row: 0.9,
        semantic_similarity=lambda left, right: 1.0 if left["prompt"][0] == right["prompt"][0] else 0.0,
        mixture_resolver=lambda name: [{"prompt": f"{name}-one"}, {"prompt": f"{name}-two"}],
    )
    fuzzy = RecipeRunner().run(
        [{"prompt": "a b c d e"}, {"prompt": "a b c d e"}, {"prompt": "other words"}],
        {"steps": [{"kind": "dedup", "method": "fuzzy", "threshold": 0.8}]},
    )
    assert len(fuzzy.records) == 2
    semantic = RecipeRunner(context).run(
        [{"prompt": "apple"}, {"prompt": "apricot"}, {"prompt": "banana"}],
        {"steps": [{"kind": "dedup", "method": "semantic", "threshold": 0.9}]},
    )
    assert [row["prompt"] for row in semantic.records] == ["apple", "banana"]
    mixed = RecipeRunner(context).run(
        [{"prompt": "current"}],
        {"seed": 3, "steps": [{"kind": "mix", "datasets": [{"source": "current", "weight": 1}, {"source": "extra", "weight": 1}], "size": 4}]},
    )
    assert len(mixed.records) == 4
    mined = RecipeRunner(context).run(
        [{"prompt": "ignored"}],
        {"steps": [{"kind": "failure_mining", "path": str(failures)}, {"kind": "synthesize", "teacher_model": "stub"}]},
    )
    assert mined.records[0]["response"] == "answer:failed"
    assert mined.records[0]["metadata"]["synthesis"]["teacher_model"] == "stub"


def test_configured_teacher_supports_relative_media_and_registered_verifiers(tmp_path, monkeypatch):
    pil = pytest.importorskip("PIL.Image")
    pil.new("RGB", (2, 2), color="green").save(tmp_path / "sample.png")
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "generated annotation"}}]}

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("requests.post", post)
    completion = configured_teacher(
        "Describe this image",
        {
            "endpoint_type": "openai_compatible",
            "base_url": "http://127.0.0.1:9000/v1",
            "teacher_model": "local-teacher",
            "base_dir": str(tmp_path),
        },
        {"image": "sample.png"},
    )
    assert completion == "generated annotation"
    assert calls[0][0] == "http://127.0.0.1:9000/v1/chat/completions"
    content = calls[0][1]["json"]["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert configured_verifier(
        {"response": '{"accepted": true}'}, {"verifier": "json_structure"}
    ) == 1.0


def test_recipe_callbacks_receive_step_configuration_and_source_row():
    observed = {}

    def teacher(prompt, params, row):
        observed.update(prompt=prompt, model=params["teacher_model"], row=dict(row))
        return "answer"

    result = RecipeRunner(RecipeContext(teacher=teacher)).run(
        [{"prompt": "question", "image": "asset.png"}],
        {"steps": [{"kind": "synthesize", "teacher_model": "teacher-a"}]},
    )
    assert observed == {
        "prompt": "question",
        "model": "teacher-a",
        "row": {"prompt": "question", "image": "asset.png"},
    }
    assert result.records[0]["response"] == "answer"


def test_topic_curriculum_labels_categorical_fields():
    result = RecipeRunner().run(
        [{"topic": "math"}, {"topic": "code"}, {"topic": "other"}],
        {
            "steps": [
                {
                    "kind": "curriculum",
                    "method": "topic",
                    "field": "topic",
                    "mapping": {"math": "reasoning", "code": "programming"},
                    "default_label": "general",
                }
            ]
        },
    )
    assert [row["metadata"]["curriculum"] for row in result.records] == [
        "reasoning",
        "programming",
        "general",
    ]


def test_dataset_lab_immutable_version_reuse_export_materialization_and_staleness(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    image = tmp_path / "sample.png"
    pil.new("RGB", (2, 2), color="blue").save(image)
    source_file = _jsonl(tmp_path / "source.jsonl", [{"image_path": image.name, "question": "Q", "answer": "A", "group": "asset-1"}])
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {"kind": "local", "path": str(source_file), "canonical_kind": "vlm"},
        dataset_id="vision",
        source_id="source-one",
    )
    recipe = {"steps": [{"kind": "split", "method": "grouped", "group_field": "metadata.group", "ratios": {"train": 1}}]}
    first = lab.build(source.id, recipe)
    second = lab.build(source.id, recipe)
    assert first.version_id == second.version_id
    assert second.reused is True
    assert lab.verify_version(first.version_id)["valid"] is True
    exported = lab.export(first.version_id, output=tmp_path / "export.jsonl", split="train")
    assert json.loads(exported.read_text())["image"] == image.name
    csv_export = lab.export(
        first.version_id, output=tmp_path / "export.csv", format="csv", split="train"
    )
    assert "image" in csv_export.read_text(encoding="utf-8").splitlines()[0]
    materialized = lab.materialize(first.version_id, background=False)
    assert materialized.materialized_assets is True
    preview = lab.get_preview(materialized.version_id)
    assert preview.records[0]["image"].startswith("assets/")
    assert (Path(materialized.path) / preview.records[0]["image"]).is_file()
    source_file.write_text(json.dumps({"image_path": image.name, "question": "changed", "answer": "A"}) + "\n")
    with pytest.raises(SourceError, match="refresh_source"):
        lab.build(source.id, recipe)
    lab.close()


def test_version_preview_streams_only_the_requested_page(tmp_path, monkeypatch):
    rows = [{"prompt": f"question-{index}"} for index in range(25)]
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {"kind": "local", "path": str(_jsonl(tmp_path / "rows.jsonl", rows))},
        dataset_id="paged",
    )
    version = lab.build(
        source.id,
        {"steps": [{"kind": "split", "ratios": {"train": 1}}]},
    )
    monkeypatch.setattr(
        lab.store,
        "load_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full load")),
    )

    preview = lab.get_preview(version.version_id, split="train", offset=7, limit=3)

    assert preview.total == 25
    split_rows = [
        json.loads(line)
        for line in (Path(version.path) / "splits" / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert preview.records == split_rows[7:10]
    lab.close()


def test_dataset_lab_mixture_resolves_immutable_versions_without_injected_callbacks(tmp_path):
    lab = DatasetLab(tmp_path / "lab")
    first_source = lab.add_source(
        {"kind": "local", "path": str(_jsonl(tmp_path / "first.jsonl", [{"prompt": "first"}]))},
        dataset_id="first",
    )
    first_version = lab.build(
        first_source.id, {"steps": [{"kind": "split", "ratios": {"train": 1}}]}
    )
    second_source = lab.add_source(
        {"kind": "local", "path": str(_jsonl(tmp_path / "second.jsonl", [{"prompt": "second"}]))},
        dataset_id="second",
    )
    mixed = lab.build(
        second_source.id,
        {
            "seed": 4,
            "steps": [
                {
                    "kind": "mix",
                    "datasets": [
                        {"source": "current", "weight": 1},
                        {"source": first_version.version_id, "weight": 1},
                    ],
                    "size": 2,
                }
            ],
        },
    )
    assert {row["prompt"] for row in lab.get_preview(mixed.version_id).records} == {
        "first",
        "second",
    }
    lab.close()


def test_changed_media_requires_source_refresh_before_build_training_or_materialization(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    image = tmp_path / "sample.png"
    pil.new("RGB", (2, 2), color="blue").save(image)
    source_file = _jsonl(
        tmp_path / "source.jsonl",
        [{"image": image.name, "prompt": "Q", "response": "A"}],
    )
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {"kind": "local", "path": str(source_file), "canonical_kind": "vlm"},
        dataset_id="vision",
    )
    version = lab.build(source.id, {"steps": [{"kind": "split", "ratios": {"train": 1}}]})
    pil.new("RGB", (2, 2), color="red").save(image)
    with pytest.raises(SourceError, match="refresh_source"):
        lab.build(source.id, {"steps": [{"kind": "split", "ratios": {"train": 1}}]})
    assert lab.verify_version(version.version_id)["valid"] is False
    with pytest.raises(VersionError, match="changed or missing asset"):
        lab.materialize_version(version.version_id)
    lab.close()


def test_build_refuses_missing_referenced_media(tmp_path):
    source_file = _jsonl(
        tmp_path / "source.jsonl",
        [{"image": "missing.png", "prompt": "Q", "response": "A"}],
    )
    lab = DatasetLab(tmp_path / "lab")
    source = lab.add_source(
        {"kind": "local", "path": str(source_file), "canonical_kind": "vlm"}
    )
    with pytest.raises(SourceError, match="missing referenced assets"):
        lab.build(source.id, {"steps": [{"kind": "split", "ratios": {"train": 1}}]})
    lab.close()


def test_failed_build_retries_from_persisted_recipe_boundary(tmp_path):
    source_path = _jsonl(tmp_path / "source.jsonl", [{"prompt": "  question  "}])

    def failing_teacher(*args):
        raise RuntimeError("teacher unavailable")

    lab = DatasetLab(tmp_path / "lab", teacher=failing_teacher)
    source = lab.add_source({"kind": "local", "path": str(source_path)})
    recipe = {
        "steps": [
            {"kind": "normalize", "fields": ["prompt"]},
            {"kind": "synthesize", "teacher_model": "local-teacher"},
        ]
    }
    failed = lab.start_job(source.id, recipe)
    failed = lab.job_manager.wait(failed.id, 5)
    assert failed.status == "failed"
    assert failed.checkpoint["completed_step"] == 1
    checkpoint_path = Path(failed.checkpoint["checkpoint_path"])
    assert checkpoint_path.is_file()

    lab.teacher = lambda prompt, params, row: f"answer:{prompt}"
    retried = lab.retry(failed.id)
    completed = lab.job_manager.wait(retried.id, 5)
    assert completed.status == "succeeded"
    assert any("Resuming after recipe step 1" in line for line in completed.logs)
    preview = lab.get_preview(completed.result["version_id"])
    assert preview.records[0]["prompt"] == "question"
    assert preview.records[0]["response"] == "answer:question"
    assert preview.records[0]["metadata"]["synthesis"] == {
        "teacher_model": "local-teacher",
        "endpoint_type": "injected",
        "prompt": "question",
        "prompt_field": "prompt",
        "output_field": "response",
        "sampling": {},
        "generation_index": 0,
        "verifier": None,
        "verifier_score": 1.0,
        "accepted": True,
    }
    assert not checkpoint_path.exists()
    lab.close()


def test_serial_jobs_progress_retry_and_recovery(tmp_path):
    state = tmp_path / "jobs.json"
    manager = SerialJobManager(state)
    attempts = {"count": 0}

    def handler(context, payload):
        attempts["count"] += 1
        context.progress(stage="working", processed=1, total=2)
        context.checkpoint(boundary=1)
        if attempts["count"] == 1:
            raise RuntimeError("first attempt fails")
        context.progress(processed=2)
        return {"ok": payload["value"]}

    manager.register("test", handler)
    first = manager.start("test", {"value": 7})
    assert manager.wait(first.id, 5).status == "failed"
    retried = manager.retry(first.id)
    completed = manager.wait(retried.id, 5)
    assert completed.status == "succeeded"
    assert completed.result == {"ok": 7}
    assert completed.checkpoint == {"boundary": 1}
    manager.shutdown()

    payload = json.loads(state.read_text())
    payload["jobs"][0]["status"] = "running"
    payload["jobs"][0]["worker_pid"] = 99_999_999
    payload["jobs"][0]["worker_id"] = "dead-worker"
    state.write_text(json.dumps(payload))
    recovered = SerialJobManager(state)
    assert any(job.status == "interrupted" for job in recovered.list())
    recovered.shutdown()


def test_serial_jobs_do_not_interrupt_work_owned_by_a_live_process(tmp_path):
    state = tmp_path / "jobs.json"
    started = threading.Event()
    release = threading.Event()
    owner = SerialJobManager(state)

    def handler(context, payload):
        started.set()
        assert release.wait(5)
        return {"ok": payload["value"]}

    owner.register("test", handler)
    job = owner.start("test", {"value": 9})
    assert started.wait(5)

    observer = SerialJobManager(state)
    assert observer.get(job.id).status == "running"

    release.set()
    assert owner.wait(job.id, 5).status == "succeeded"
    observer.shutdown()
    owner.shutdown()


def test_background_build_job_and_preview_contract(tmp_path):
    source_path = _jsonl(tmp_path / "data.jsonl", [{"prompt": "Q", "response": "A"}])
    lab = DatasetLab(tmp_path / "lab")
    source = lab.register_source({"kind": "local", "path": str(source_path)}, dataset_id="text")
    job = lab.start_job(source.id, {"schema": "sft", "steps": [{"kind": "map"}, {"kind": "split", "ratios": {"train": 1}}]})
    completed = lab.job_manager.wait(job.id, 5)
    assert completed.status == "succeeded"
    version_id = completed.result["version_id"]
    assert lab.get_preview(version_id, split="train").total == 1
    assert lab.get_version(version_id).row_count == 1
    lab.close()


def test_invalid_recipe_is_rejected_before_execution():
    with pytest.raises(RecipeError, match="safe filter"):
        Recipe.from_value({"steps": [{"kind": "filter", "field": "x", "op": "eval"}]})
    with pytest.raises(RecipeError, match="follow a split"):
        Recipe.from_value({"steps": [{"kind": "contamination"}]})
