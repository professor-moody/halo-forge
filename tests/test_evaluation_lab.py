"""Persistent benchmark/evaluation catalog and one-worker job contracts."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest


@pytest.fixture
def db():
    from halo_forge.run_db import RunDatabase

    value = RunDatabase(":memory:")
    yield value
    value.close()


@pytest.fixture
def lab(db, tmp_path):
    from halo_forge.evaluation_lab import EvaluationLabService

    value = EvaluationLabService(db, tmp_path / "evaluations")
    yield value
    value.shutdown()


def _suite(lab, *, direction="maximize"):
    return lab.create_suite(
        name=f"suite-{direction}",
        items=[
            {"id": "a", "record_id": "r-a", "input": "A", "expected": "A"},
            {"id": "b", "record_id": "r-b", "input": "B", "expected": "B"},
            {"id": "c", "record_id": "r-c", "input": "C", "expected": "C"},
            {"id": "d", "record_id": "r-d", "input": "D", "expected": "D"},
        ],
        primary_metric="score",
        direction=direction,
    )


def test_schema_v4_migrates_run_dataset_role_without_losing_rows(tmp_path):
    path = tmp_path / "v3.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '3');
        CREATE TABLE datasets (
            id TEXT PRIMARY KEY, name TEXT, description TEXT, modality TEXT,
            canonical_schema TEXT, latest_version_id TEXT, created_at TEXT, updated_at TEXT
        );
        CREATE TABLE dataset_versions (
            id TEXT PRIMARY KEY, dataset_id TEXT, source_id TEXT, parent_version_id TEXT,
            status TEXT, content_hash TEXT, recipe_hash TEXT, recipe_json TEXT,
            storage_path TEXT, row_count INTEGER, size_bytes INTEGER,
            split_counts_json TEXT, statistics_json TEXT, provenance_json TEXT,
            source_fingerprints_json TEXT, assets_materialized INTEGER, error TEXT,
            created_at TEXT, completed_at TEXT
        );
        INSERT INTO datasets VALUES ('ds', 'D', NULL, 'text', 'sft', NULL, 't', 't');
        INSERT INTO dataset_versions VALUES (
            'v1', 'ds', NULL, NULL, 'completed', 'c', 'r', '{}', '/tmp/v1',
            1, 1, '{}', '{}', '{}', '{}', 0, NULL, 't', 't'
        );
        CREATE TABLE run_datasets (
            run_id TEXT NOT NULL, dataset_version_id TEXT NOT NULL,
            split TEXT NOT NULL DEFAULT 'train', attached_at TEXT NOT NULL,
            PRIMARY KEY (run_id, dataset_version_id, split)
        );
        INSERT INTO run_datasets VALUES ('run-1', 'v1', 'train', 't');
        """)
    conn.commit()
    conn.close()

    from halo_forge.run_db import RunDatabase
    from halo_forge.run_db.schema import SCHEMA_VERSION

    migrated = RunDatabase(str(path))
    binding = migrated.list_run_datasets("run-1")[0]
    assert SCHEMA_VERSION == 23
    assert binding.role == "train"
    assert binding.training_artifact_id is None
    tables = {
        row["name"]
        for row in migrated._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    assert {
        "training_artifacts",
        "benchmark_suites",
        "benchmark_suite_revisions",
        "evaluations",
        "evaluation_metrics",
        "evaluation_samples",
    }.issubset(tables)
    migrated.close()


def test_training_artifact_catalog_and_role_aware_run_bindings(db, tmp_path):
    dataset = db.create_dataset(
        dataset_id="ds", name="SFT", modality="text", canonical_schema="sft"
    )
    version = db.create_dataset_version(
        dataset_id=dataset.id,
        version_id="v1",
        recipe_hash="recipe",
        recipe={"steps": []},
        storage_path=str(tmp_path / "v1"),
        status="completed",
        content_hash="content",
    )
    mixture_version = db.create_dataset_version(
        dataset_id=dataset.id,
        version_id="v2",
        recipe_hash="recipe-2",
        recipe={"steps": []},
        storage_path=str(tmp_path / "v2"),
        status="completed",
        content_hash="content-2",
    )
    artifact = db.create_training_artifact(
        artifact_id="artifact-1",
        artifact_hash="artifact-hash",
        adapter_id="sft",
        adapter_version="1",
        trainer_mode="sft",
        manifest_path=str(tmp_path / "artifact" / "manifest.json"),
        model_id="model",
        tokenizer_revision="rev",
        bindings=[
            {"role": "train", "dataset_version_id": version.id, "split": "train", "row_count": 9},
            {
                "role": "train",
                "dataset_version_id": mixture_version.id,
                "split": "train",
                "row_count": 4,
            },
            {
                "role": "validation",
                "dataset_version_id": version.id,
                "split": "validation",
                "row_count": 1,
            },
        ],
        metadata={"token_count": 100},
    )
    assert artifact.metadata == {"token_count": 100}
    assert [binding.role for binding in artifact.bindings].count("train") == 2
    assert db.find_training_artifact("artifact-hash").id == artifact.id
    db.attach_run_dataset(
        run_id="run-1",
        dataset_version_id=version.id,
        role="train",
        split="train",
        training_artifact_id=artifact.id,
    )
    db.attach_run_dataset(
        run_id="run-1",
        dataset_version_id=version.id,
        role="validation",
        split="validation",
        training_artifact_id=artifact.id,
    )
    bindings = db.list_run_datasets("run-1")
    assert {binding.role for binding in bindings} == {"train", "validation"}
    assert all(binding.training_artifact_id == artifact.id for binding in bindings)
    linked = db.list_runs_for_dataset_version(version.id)
    assert {(binding.run_id, binding.role) for binding in linked} == {
        ("run-1", "train"),
        ("run-1", "validation"),
    }


def test_suite_revisions_are_immutable_numbered_and_content_reused(lab):
    suite, first = lab.create_suite(
        name="math", items=[{"id": "one", "expected": 1}], primary_metric="accuracy"
    )
    duplicate = lab.create_revision(
        suite_id=suite.id,
        items=[{"id": "one", "expected": 1}],
        primary_metric="accuracy",
        direction="maximize",
    )
    second = lab.create_revision(
        suite_id=suite.id,
        items=[{"id": "one", "expected": 1}, {"id": "two", "expected": 2}],
        primary_metric="accuracy",
        direction="maximize",
    )
    assert duplicate.id == first.id
    assert first.revision_number == 1
    assert second.revision_number == 2
    assert lab.db.get_benchmark_suite(suite.id).latest_revision_id == second.id
    assert [value.id for value in lab.db.list_benchmark_suite_revisions(suite.id)] == [
        second.id,
        first.id,
    ]
    from halo_forge.evaluation_lab import EvaluationLabError

    with pytest.raises(EvaluationLabError, match="duplicate benchmark suite item id"):
        lab.create_revision(
            suite_id=suite.id,
            items=[{"id": "same"}, {"id": "same"}],
            primary_metric="accuracy",
            direction="maximize",
        )


def test_suite_purpose_and_holdout_subject_policy_are_explicit(lab):
    from halo_forge.evaluation_lab import EvaluationLabError

    suite, revision = lab.create_suite(
        name="final-holdout",
        items=[{"id": "one", "expected": "one"}],
        purpose="holdout",
    )
    assert suite.purpose == "holdout"

    with pytest.raises(EvaluationLabError, match="intermediate checkpoints"):
        lab.jobs.launch(
            suite_revision_id=revision.id,
            subject={
                "type": "checkpoint",
                "ref": "/tmp/checkpoint",
                "content_hash": "checkpoint-hash",
            },
        )
    with pytest.raises(EvaluationLabError, match="require a pinned model"):
        lab.jobs.launch(
            suite_revision_id=revision.id,
            subject={"type": "model", "ref": "unpinned/model"},
        )

    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        subject={"type": "model", "ref": "pinned/model", "revision": "commit-sha"},
        request={"scores": {"one": 1.0}},
    )
    assert lab.jobs.wait(launch.evaluation.id, 5).status == "completed"
    candidate = lab.jobs.launch(
        suite_revision_id=revision.id,
        subject={"type": "model", "ref": "candidate/model", "revision": "commit-sha"},
        request={"scores": {"one": 0.0}},
    )
    assert lab.jobs.wait(candidate.evaluation.id, 5).status == "completed"
    comparison = lab.compare(launch.evaluation.id, candidate.evaluation.id)
    assert comparison["suite_purpose"] == "holdout"
    assert comparison["failure_mining_allowed"] is False


def test_subject_hash_is_stable_and_changes_with_local_content(db, tmp_path):
    from halo_forge.evaluation_lab import resolve_subject

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weights = checkpoint / "weights.bin"
    weights.write_bytes(b"first")
    first = resolve_subject({"type": "checkpoint", "ref": str(checkpoint)}, db)
    again = resolve_subject({"type": "checkpoint", "ref": str(checkpoint)}, db)
    weights.write_bytes(b"second")
    changed = resolve_subject({"type": "checkpoint", "ref": str(checkpoint)}, db)
    assert first.subject_hash == again.subject_hash
    assert changed.subject_hash != first.subject_hash


def test_local_subject_hashing_runs_after_evaluation_is_persisted(lab, tmp_path, monkeypatch):
    import halo_forge.evaluation_lab.service as evaluation_service

    checkpoint = tmp_path / "slow-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    entered = threading.Event()
    release = threading.Event()

    def slow_hash(path):
        entered.set()
        assert release.wait(3), "test did not release deferred subject hashing"
        return "stable-local-content"

    monkeypatch.setattr(evaluation_service, "_hash_path", slow_hash)
    _, revision = lab.create_suite(
        name="deferred-local-subject",
        items=[{"id": "one", "input": "one", "expected": "one"}],
    )

    started = time.monotonic()
    launched = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "checkpoint", "ref": str(checkpoint)},
        request={"scores": {"one": 1.0}},
    )
    elapsed = time.monotonic() - started
    try:
        assert elapsed < 0.5
        assert lab.db.get_evaluation(launched.evaluation.id) is not None
        assert launched.evaluation.request["subject_resolution"] == "pending"
        assert entered.wait(1)
    finally:
        release.set()

    completed = lab.jobs.wait(launched.evaluation.id, 5)
    resolved_again = evaluation_service.resolve_subject(
        {"type": "checkpoint", "ref": str(checkpoint)}, lab.db
    )
    assert completed.status == "completed"
    assert completed.subject_hash == resolved_again.subject_hash
    assert completed.subject_hash != launched.evaluation.subject_hash
    assert completed.request["subject_resolution"] == "resolved"
    assert completed.request["subject"]["payload"]["content_hash"] == "stable-local-content"


def test_detached_worker_resolves_deferred_subject_when_claimed(lab, tmp_path, monkeypatch):
    import halo_forge.evaluation_lab.service as evaluation_service

    checkpoint = tmp_path / "detached-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"model")
    calls = []
    original_hash = evaluation_service._hash_path

    def recording_hash(path):
        calls.append(Path(path))
        return original_hash(path)

    monkeypatch.setattr(evaluation_service, "_hash_path", recording_hash)
    _, revision = lab.create_suite(
        name="deferred-detached",
        items=[{"id": "one", "expected": "one"}],
    )
    launched = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "checkpoint", "ref": str(checkpoint)},
        request={"scores": {"one": 1.0}},
        submit=False,
    )

    assert calls == []
    assert launched.evaluation.request["subject_resolution"] == "pending"
    completed = lab.jobs.run_queued(launched.evaluation.id)
    assert completed.status == "completed"
    assert calls == [checkpoint]
    assert completed.request["subject_resolution"] == "resolved"


def test_legacy_resolved_subject_envelope_without_state_marker_remains_runnable(lab):
    _, revision = lab.create_suite(
        name="legacy-resolved-subject",
        items=[{"id": "one", "expected": "one"}],
    )
    launched = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "org/model", "revision": "abc123"},
        request={"scores": {"one": 1.0}},
        submit=False,
    )
    legacy_request = dict(launched.evaluation.request)
    legacy_request.pop("subject_resolution", None)
    legacy_request.pop("subject_input", None)
    lab.db.update_evaluation(launched.evaluation.id, request=legacy_request)

    completed = lab.jobs.run_queued(launched.evaluation.id)

    assert completed.status == "completed"
    assert completed.request["subject"]["subject_ref"] == "org/model"


def test_cancelling_during_subject_hash_can_retry_from_pending_identity(lab, tmp_path, monkeypatch):
    import halo_forge.evaluation_lab.service as evaluation_service

    checkpoint = tmp_path / "cancelled-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"model")
    entered = threading.Event()
    release = threading.Event()

    def controlled_hash(path):
        entered.set()
        assert release.wait(3)
        return "cancelled-then-resolved"

    monkeypatch.setattr(evaluation_service, "_hash_path", controlled_hash)
    _, revision = lab.create_suite(
        name="deferred-cancel",
        items=[{"id": "one", "expected": "one"}],
    )
    launched = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "checkpoint", "ref": str(checkpoint)},
        request={"scores": {"one": 1.0}},
    )
    assert entered.wait(1)
    lab.jobs.cancel(launched.evaluation.id)
    release.set()
    cancelled = lab.jobs.wait(launched.evaluation.id, 5)

    assert cancelled.status == "cancelled"
    assert cancelled.request["subject_resolution"] == "pending"
    retried = lab.jobs.retry(cancelled.id)
    assert retried.status == "queued"
    completed = lab.jobs.wait(cancelled.id, 5)
    assert completed.status == "completed"
    assert completed.request["subject_resolution"] == "resolved"


def test_deferred_resolution_reuses_result_with_its_own_immutable_bundle(
    lab, tmp_path, monkeypatch
):
    checkpoint = tmp_path / "reused-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"same model")
    _, revision = lab.create_suite(
        name="deferred-reuse",
        items=[{"id": "one", "record_id": "r1", "expected": "one"}],
    )
    adapter = lab.jobs.registry.get("dataset")
    original_evaluate = adapter.evaluate
    calls = []

    def counting_evaluate(*args, **kwargs):
        calls.append(args[2].subject_hash)
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(adapter, "evaluate", counting_evaluate)
    arguments = dict(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "checkpoint", "ref": str(checkpoint)},
        request={"scores": {"one": 1.0}},
    )
    first_launch = lab.jobs.launch(**arguments)
    first = lab.jobs.wait(first_launch.evaluation.id, 5)
    second_launch = lab.jobs.launch(**arguments)
    second = lab.jobs.wait(second_launch.evaluation.id, 5)

    assert first.status == second.status == "completed"
    assert second_launch.reused is False  # identity was not known at launch time
    assert first.id != second.id
    assert first.subject_hash == second.subject_hash
    assert calls == [first.subject_hash]
    assert Path(first.artifact_path) != Path(second.artifact_path)
    second_manifest = json.loads(
        Path(second.artifact_path, "evaluation.json").read_text(encoding="utf-8")
    )
    assert second_manifest["evaluation_id"] == second.id
    assert second_manifest["subject"]["subject_hash"] == second.subject_hash
    assert second.request["canonical_reuse_key"] == first.reuse_key
    assert second.reuse_key != first.reuse_key
    assert second.result["summary"]["reused_from_evaluation_id"] == first.id


def test_orphaned_pending_subject_can_be_recovered_and_resolved(tmp_path):
    from halo_forge.evaluation_lab import EvaluationLabService
    from halo_forge.run_db import RunDatabase

    checkpoint = tmp_path / "orphaned-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.bin").write_bytes(b"model")
    database_path = tmp_path / "evaluations.db"
    first_db = RunDatabase(str(database_path))
    first = EvaluationLabService(first_db, tmp_path / "artifacts")
    _, revision = first.create_suite(
        name="orphaned-deferred",
        items=[{"id": "one", "expected": "one"}],
    )
    launched = first.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "checkpoint", "ref": str(checkpoint)},
        request={"scores": {"one": 1.0}},
        submit=False,
    )
    request = dict(launched.evaluation.request)
    request["worker"] = {"pid": 99_999_999, "worker_id": "dead-deferred-worker"}
    first_db.update_evaluation(launched.evaluation.id, request=request)
    first.shutdown()
    first_db.close()

    recovered_db = RunDatabase(str(database_path))
    recovered = EvaluationLabService(recovered_db, tmp_path / "artifacts")
    interrupted = recovered.jobs.get(launched.evaluation.id)
    assert interrupted.status == "interrupted"
    assert interrupted.request["subject_resolution"] == "pending"
    recovered.jobs.retry(interrupted.id, submit=False)
    completed = recovered.jobs.run_queued(interrupted.id)

    assert completed.status == "completed"
    assert completed.request["subject_resolution"] == "resolved"
    recovered.shutdown()
    recovered_db.close()


def test_persistent_job_publishes_atomically_and_reuses_completed_result(lab):
    _, revision = _suite(lab)
    launched = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "org/model", "revision": "abc123"},
        request={"scores": {"a": 1, "b": 0, "c": 1, "d": 1}},
    )
    completed = lab.jobs.wait(launched.evaluation.id, timeout=5)
    assert completed.status == "completed"
    artifact = Path(completed.artifact_path)
    assert (artifact / "evaluation.json").is_file()
    assert (artifact / "metrics.json").is_file()
    assert (artifact / "samples.jsonl").is_file()
    assert not list(artifact.parent.glob(".*.tmp"))
    detail = lab.evaluation_detail(completed.id)
    assert len(detail["samples"]) == 4

    repeated = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "org/model", "revision": "abc123"},
        request={"scores": {"a": 1, "b": 0, "c": 1, "d": 1}},
    )
    assert repeated.reused is True
    assert repeated.evaluation.id == completed.id


def test_deferred_evaluation_can_be_claimed_by_detached_worker(lab):
    _, revision = _suite(lab)
    launched = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "fixture", "revision": "pinned"},
        request={"scores": {"a": 1, "b": 0, "c": 1, "d": 0}},
        submit=False,
    )

    assert lab.jobs.get(launched.evaluation.id).status == "queued"
    assert launched.evaluation.id not in lab.jobs._futures
    completed = lab.jobs.run_queued(launched.evaluation.id)

    assert completed.status == "completed"
    assert completed.processed_samples == 4
    assert Path(completed.artifact_path, "evaluation.json").is_file()


def test_dataset_adapter_generates_real_output_without_echoing_expected(lab, monkeypatch):
    import halo_forge.serving.adapter

    calls = []

    class FakeServingAdapter:
        def generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return "generated answer"

    monkeypatch.setattr(
        halo_forge.serving.adapter,
        "build_serving_adapter",
        lambda *args, **kwargs: FakeServingAdapter(),
    )
    _, revision = lab.create_suite(
        name="real-generation",
        items=[
            {
                "id": "question",
                "record_id": "question-record",
                "input": "What is generated?",
                "expected": "generated answer",
            }
        ],
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "local/model"},
        request={"max_tokens": 17, "temperature": 0.0},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    sample = lab.db.list_evaluation_samples(completed.id)[0]

    assert completed.status == "completed"
    assert sample.output == "generated answer"
    assert sample.score == 1.0 and sample.passed is True
    assert calls == [
        (
            "What is generated?",
            {
                "max_tokens": 17,
                "temperature": 0.0,
                "top_p": 1.0,
                "stop": None,
            },
        )
    ]


def test_dataset_adapter_records_generation_error_instead_of_fabricating_pass(lab, monkeypatch):
    import halo_forge.serving.adapter

    class BrokenServingAdapter:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(
        halo_forge.serving.adapter,
        "build_serving_adapter",
        lambda *args, **kwargs: BrokenServingAdapter(),
    )
    _, revision = lab.create_suite(
        name="generation-error",
        items=[{"id": "one", "input": "prompt", "expected": "secret answer"}],
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "local/model"},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    sample = lab.db.list_evaluation_samples(completed.id)[0]

    assert completed.status == "completed"
    assert sample.output is None
    assert sample.score is None
    assert sample.passed is False
    assert "model unavailable" in sample.error


def test_expected_only_item_is_not_echoed_as_a_model_output(lab):
    _, revision = lab.create_suite(
        name="expected-is-not-output",
        items=[{"id": "only-expected", "expected": "do not copy me"}],
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "local/model"},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    sample = lab.db.list_evaluation_samples(completed.id)[0]

    assert sample.expected == "do not copy me"
    assert sample.output is None
    assert sample.score is None
    assert sample.passed is False
    assert "no input for model generation" in sample.error


def test_score_by_subject_remains_explicit_fixture_evidence(lab, monkeypatch):
    import halo_forge.serving.adapter

    def unexpected_model_load(*args, **kwargs):
        raise AssertionError("explicit fixture scores must not load a model")

    monkeypatch.setattr(
        halo_forge.serving.adapter,
        "build_serving_adapter",
        unexpected_model_load,
    )
    _, revision = lab.create_suite(
        name="subject-fixture",
        items=[
            {
                "id": "fixture",
                "expected": "reference",
                "score_by_subject": {"candidate/model": 0.75},
            }
        ],
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "candidate/model"},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    sample = lab.db.list_evaluation_samples(completed.id)[0]

    assert sample.output is None
    assert sample.score == 0.75
    assert sample.passed is True
    assert sample.error is None


def test_dataset_split_item_expands_canonical_rows_with_lineage(lab, tmp_path):
    from halo_forge.data_lab import DatasetLab

    dataset_root = tmp_path / "datasets"
    source_path = tmp_path / "sft.jsonl"
    source_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"id": "one", "prompt": "One?", "response": "1"},
                {"id": "two", "prompt": "Two?", "response": "2"},
            )
        ),
        encoding="utf-8",
    )
    datasets = DatasetLab(dataset_root)
    source = datasets.add_source(
        {"kind": "local", "path": str(source_path), "canonical_kind": "sft"},
        dataset_id="eval-source",
    )
    version = datasets.build(source.id, {"steps": [{"kind": "normalize", "fields": ["prompt"]}]})
    identities = datasets.store.load_lineage(
        version.version_id, dataset_id=version.dataset_id, split="train"
    )
    outputs = {identity.record_id: expected for identity, expected in zip(identities, ("1", "2"))}

    _, revision = lab.create_suite(
        name="dataset-split",
        items=[
            {
                "id": "heldout",
                "dataset_id": version.dataset_id,
                "dataset_version_id": version.version_id,
                "split": "train",
            }
        ],
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "fixture/model"},
        request={"dataset_root": str(dataset_root), "outputs": outputs},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    samples = lab.db.list_evaluation_samples(completed.id)
    datasets.close()

    assert completed.status == "completed"
    assert [sample.record_id for sample in samples] == [
        identity.record_id for identity in identities
    ]
    assert [sample.input for sample in samples] == ["One?", "Two?"]
    assert [sample.expected for sample in samples] == ["1", "2"]
    assert [sample.output for sample in samples] == ["1", "2"]
    assert all(sample.suite_item_id.startswith("heldout:inst_") for sample in samples)
    assert all(sample.metadata["canonical_schema"] == "sft" for sample in samples)
    assert all(sample.metadata["canonical_record"]["prompt"] for sample in samples)


def test_dataset_split_evaluation_uses_streaming_store_iterator(lab, tmp_path, monkeypatch):
    from halo_forge.data_lab import DatasetLab, VersionStore

    dataset_root = tmp_path / "streaming-datasets"
    source_path = tmp_path / "streaming.jsonl"
    rows = [{"prompt": f"Question {index}?", "response": f"Answer {index}"} for index in range(12)]
    source_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    datasets = DatasetLab(dataset_root)
    source = datasets.add_source(
        {"kind": "local", "path": str(source_path), "canonical_kind": "sft"},
        dataset_id="streaming-eval",
    )
    version = datasets.build(source.id, {"steps": [{"kind": "normalize", "fields": ["prompt"]}]})
    identities = datasets.store.load_lineage(
        version.version_id, dataset_id=version.dataset_id, split="train"
    )
    outputs = {
        identity.record_id: rows[index]["response"] for index, identity in enumerate(identities)
    }

    def forbidden_bulk_load(*args, **kwargs):
        raise AssertionError("evaluation must not bulk-load identified records")

    monkeypatch.setattr(VersionStore, "load_identified_records", forbidden_bulk_load)
    _, revision = lab.create_suite(
        name="streaming-split",
        items=[
            {
                "id": "train-split",
                "dataset_id": version.dataset_id,
                "dataset_version_id": version.version_id,
                "split": "train",
            }
        ],
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "fixture/model"},
        request={"dataset_root": str(dataset_root), "outputs": outputs},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    samples = lab.db.list_evaluation_samples(completed.id)
    datasets.close()

    assert completed.status == "completed"
    assert completed.total_samples == len(rows)
    assert completed.processed_samples == len(rows)
    assert [sample.record_id for sample in samples] == [
        identity.record_id for identity in identities
    ]


def test_dataset_split_expansion_is_sized_and_lazy(tmp_path, monkeypatch):
    from halo_forge.data_lab import DatasetLab, VersionStore
    from halo_forge.evaluation_lab.adapters import _expand_dataset_items

    dataset_root = tmp_path / "lazy-datasets"
    source_path = tmp_path / "lazy.jsonl"
    row_count = 512
    source_path.write_text(
        "".join(
            json.dumps({"prompt": f"P{index}", "response": f"R{index}"}) + "\n"
            for index in range(row_count)
        ),
        encoding="utf-8",
    )
    datasets = DatasetLab(dataset_root)
    source = datasets.add_source(
        {"kind": "local", "path": str(source_path), "canonical_kind": "sft"},
        dataset_id="lazy-eval",
    )
    version = datasets.build(source.id, {"steps": [{"kind": "normalize", "fields": ["prompt"]}]})
    expected_first = datasets.store.load_lineage(
        version.version_id, dataset_id=version.dataset_id, split="train"
    )[0]

    original = VersionStore.iter_records_with_lineage
    consumed = []

    def tracked(self, *args, **kwargs):
        for pair in original(self, *args, **kwargs):
            consumed.append(pair[1].record_id)
            yield pair

    monkeypatch.setattr(VersionStore, "iter_records_with_lineage", tracked)
    expanded = _expand_dataset_items(
        [
            {
                "id": "bounded",
                "dataset_id": version.dataset_id,
                "dataset_version_id": version.version_id,
                "split": "train",
                "limit": 7,
            }
        ],
        {"dataset_root": str(dataset_root)},
    )

    assert len(expanded) == 7
    assert consumed == []
    iterator = iter(expanded)
    first = next(iterator)
    assert consumed == [expected_first.record_id]
    assert first["record_id"] == expected_first.record_id
    iterator.close()
    datasets.close()


def test_missing_modality_asset_is_an_explicit_failure(lab):
    _, revision = lab.create_suite(
        name="audio-without-evaluator",
        items=[
            {
                "id": "audio",
                "adapter": "audio",
                "input": {"audio": "clip.wav", "task": "transcribe"},
                "expected": "words",
            }
        ],
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        subject={"type": "model", "ref": "audio/model"},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    sample = lab.db.list_evaluation_samples(completed.id)[0]

    assert sample.output is None
    assert sample.score is None
    assert sample.passed is False
    assert "audio asset does not exist" in sample.error


@pytest.mark.parametrize("materialize_assets", [False, True])
def test_vlm_dataset_split_generates_with_verified_assets_and_stable_identity(
    lab, tmp_path, monkeypatch, materialize_assets
):
    import huggingface_hub
    import halo_forge.vlm.models

    from halo_forge.data_lab import DatasetLab

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake-png-for-adapter-boundary")
    source_path = tmp_path / "vlm.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "image": image_path.name,
                "prompt": "What color?",
                "ground_truth": "blue",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_root = tmp_path / ("vlm-materialized" if materialize_assets else "vlm-referenced")
    datasets = DatasetLab(dataset_root)
    source = datasets.add_source(
        {"kind": "local", "path": str(source_path), "canonical_kind": "vlm"},
        dataset_id="vision-eval",
    )
    version = datasets.build(
        source.id,
        {"steps": [{"kind": "normalize", "fields": ["prompt"]}]},
        materialize_assets=materialize_assets,
    )
    identity = datasets.store.load_lineage(
        version.version_id, dataset_id=version.dataset_id, split="train"
    )[0]

    model_path = tmp_path / "pinned-vlm"
    model_path.mkdir()
    snapshots = []
    calls = []
    cleaned = []

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: snapshots.append(kwargs) or str(model_path),
    )

    class FakeVLMAdapter:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return type("VLMResult", (), {"text": "blue", "metadata": {"fake": True}})()

        def cleanup(self):
            cleaned.append(True)

    monkeypatch.setattr(
        halo_forge.vlm.models,
        "get_vlm_adapter",
        lambda model_name, **kwargs: (
            calls.append({"model_name": model_name, "adapter_kwargs": kwargs}) or FakeVLMAdapter()
        ),
    )
    _, revision = lab.create_suite(
        name=f"vlm-dataset-{materialize_assets}",
        items=[
            {
                "id": "vision-split",
                "adapter": "vlm",
                "dataset_id": version.dataset_id,
                "dataset_version_id": version.version_id,
                "split": "train",
            }
        ],
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        subject={"type": "model", "ref": "org/vlm", "revision": "commit-vlm"},
        request={"dataset_root": str(dataset_root), "temperature": 0.0},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    sample = lab.db.list_evaluation_samples(completed.id)[0]
    datasets.close()

    assert sample.record_id == identity.record_id
    assert sample.output == "blue"
    assert sample.score == 1.0 and sample.passed is True
    assert sample.metadata["canonical_schema"] == "vlm"
    assert snapshots == [
        {
            "repo_id": "org/vlm",
            "revision": "commit-vlm",
            "local_files_only": False,
        }
    ]
    assert calls[0]["model_name"] == str(model_path)
    resolved_image = Path(calls[1]["image"])
    assert resolved_image.is_file()
    if materialize_assets:
        assert resolved_image.parent == Path(version.path) / "assets"
    else:
        assert resolved_image == image_path.resolve()
    assert cleaned == [True]


@pytest.mark.parametrize("materialize_assets", [False, True])
def test_audio_dataset_split_transcribes_with_verified_assets_and_stable_identity(
    lab, tmp_path, monkeypatch, materialize_assets
):
    import huggingface_hub
    import halo_forge.audio.data.loaders
    import halo_forge.audio.models.adapters

    from halo_forge.data_lab import DatasetLab

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-wav-for-adapter-boundary")
    source_path = tmp_path / "audio.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "audio": audio_path.name,
                "task": "transcribe",
                "transcript": "hello world",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset_root = tmp_path / ("audio-materialized" if materialize_assets else "audio-referenced")
    datasets = DatasetLab(dataset_root)
    source = datasets.add_source(
        {"kind": "local", "path": str(source_path), "canonical_kind": "audio"},
        dataset_id="audio-eval",
    )
    version = datasets.build(
        source.id,
        {"steps": [{"kind": "normalize", "fields": ["task"]}]},
        materialize_assets=materialize_assets,
    )
    identity = datasets.store.load_lineage(
        version.version_id, dataset_id=version.dataset_id, split="train"
    )[0]

    model_path = tmp_path / "pinned-audio"
    model_path.mkdir()
    snapshots = []
    model_calls = []
    decode_calls = []
    transcription_calls = []
    cleaned = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: snapshots.append(kwargs) or str(model_path),
    )

    class FakeAudioResult:
        text = "hello world"
        language = "en"
        confidence = 0.99
        segments = None

    class FakeAudioAdapter:
        sample_rate = 16000

        def transcribe(self, waveform, language=None):
            transcription_calls.append((waveform, language))
            return FakeAudioResult()

        def cleanup(self):
            cleaned.append(True)

    monkeypatch.setattr(
        halo_forge.audio.models.adapters,
        "get_audio_adapter",
        lambda model_name, device=None: (
            model_calls.append((model_name, device)) or FakeAudioAdapter()
        ),
    )
    monkeypatch.setattr(
        halo_forge.audio.data.loaders,
        "decode_audio",
        lambda value, target_sr=16000: (
            decode_calls.append((value, target_sr)) or ([0.0, 0.25], target_sr)
        ),
    )
    _, revision = lab.create_suite(
        name=f"audio-dataset-{materialize_assets}",
        items=[
            {
                "id": "audio-split",
                "adapter": "audio",
                "dataset_id": version.dataset_id,
                "dataset_version_id": version.version_id,
                "split": "train",
            }
        ],
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        subject={
            "type": "model",
            "ref": "org/audio",
            "revision": "commit-audio",
        },
        request={"dataset_root": str(dataset_root), "language": "en"},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    sample = lab.db.list_evaluation_samples(completed.id)[0]
    datasets.close()

    assert sample.record_id == identity.record_id
    assert sample.output == "hello world"
    assert sample.score == 1.0 and sample.passed is True
    assert sample.verifier_trace["verifier"] == "asr"
    assert snapshots[0]["revision"] == "commit-audio"
    assert model_calls == [(str(model_path), None)]
    resolved_audio = Path(decode_calls[0][0]["path"])
    assert resolved_audio.is_file()
    if materialize_assets:
        assert resolved_audio.parent == Path(version.path) / "assets"
    else:
        assert resolved_audio == audio_path.resolve()
    assert decode_calls[0][1] == 16000
    assert transcription_calls == [([0.0, 0.25], "en")]
    assert cleaned == [True]


def test_unsupported_text_backend_is_an_explicit_sample_error(lab, monkeypatch):
    import halo_forge.serving.adapter

    def unexpected_model_load(*args, **kwargs):
        raise AssertionError("unsupported backend must be rejected before model load")

    monkeypatch.setattr(
        halo_forge.serving.adapter,
        "build_serving_adapter",
        unexpected_model_load,
    )
    _, revision = lab.create_suite(
        name="unsupported-backend",
        items=[{"id": "one", "input": "prompt", "expected": "answer"}],
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "local/model"},
        request={"backend": "unsupported-endpoint"},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    sample = lab.db.list_evaluation_samples(completed.id)[0]

    assert sample.output is None and sample.passed is False
    assert "does not support backend" in sample.error


def test_missing_completed_artifact_is_rebuilt_instead_of_silently_reused(lab):
    _, revision = _suite(lab)
    arguments = dict(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "model", "revision": "pinned"},
        request={"scores": {"a": 1, "b": 1, "c": 1, "d": 1}},
    )
    first = lab.jobs.launch(**arguments)
    completed = lab.jobs.wait(first.evaluation.id, 5)
    import shutil

    shutil.rmtree(completed.artifact_path)
    rebuilt = lab.jobs.launch(**arguments)
    assert rebuilt.reused is False
    assert rebuilt.evaluation.id != completed.id
    assert lab.jobs.wait(rebuilt.evaluation.id, 5).status == "completed"


@pytest.mark.parametrize("target", ["metrics.json", "evaluation.json"])
def test_tampered_completed_artifact_is_refused_and_rebuilt(lab, target):
    _, revision = _suite(lab)
    arguments = dict(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "integrity/model", "revision": "pinned"},
        request={"scores": {"a": 1, "b": 0, "c": 1, "d": 1}},
    )
    first = lab.jobs.launch(**arguments)
    completed = lab.jobs.wait(first.evaluation.id, 5)
    path = Path(completed.artifact_path) / target
    if target == "evaluation.json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["adapter"]["version"] = "tampered"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        path.write_text("[]\n", encoding="utf-8")

    rebuilt = lab.jobs.launch(**arguments)
    assert rebuilt.reused is False
    assert rebuilt.evaluation.id != completed.id
    refused = lab.jobs.get(completed.id)
    assert refused.status == "failed"
    assert refused.stage == "corrupted_artifact"
    assert lab.jobs.wait(rebuilt.evaluation.id, 5).status == "completed"


def test_comparison_matches_records_and_respects_direction(lab):
    _, revision = _suite(lab)
    base = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "base", "revision": "1"},
        request={"scores": {"a": 1, "b": 0, "c": 0, "d": 1}},
    )
    candidate = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "candidate", "revision": "1"},
        request={"scores": {"a": 0, "b": 1, "c": 0, "d": 1}},
    )
    lab.jobs.wait(base.evaluation.id, 5)
    lab.jobs.wait(candidate.evaluation.id, 5)
    comparison = lab.compare(base.evaluation.id, candidate.evaluation.id)
    assert comparison["counts"] == {
        "regression": 1,
        "improvement": 1,
        "unchanged_failure": 1,
        "unchanged_pass": 1,
        "missing_base": 0,
        "missing_candidate": 0,
    }
    by_record = {value["record_id"]: value["outcome"] for value in comparison["sample_deltas"]}
    assert by_record["r-a"] == "regression"
    assert by_record["r-b"] == "improvement"

    suite, minimize = lab.create_suite(
        name="latency",
        items=[{"id": "one", "record_id": "one", "expected": None}],
        primary_metric="latency",
        direction="minimize",
    )
    slow = lab.jobs.launch(
        suite_revision_id=minimize.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "slow", "revision": "1"},
        request={"scores": {"one": 0.9}, "pass_threshold": 0.5},
    )
    fast = lab.jobs.launch(
        suite_revision_id=minimize.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "fast", "revision": "1"},
        request={"scores": {"one": 0.1}, "pass_threshold": 0.5},
    )
    lab.jobs.wait(slow.evaluation.id, 5)
    lab.jobs.wait(fast.evaluation.id, 5)
    delta = lab.compare(slow.evaluation.id, fast.evaluation.id)
    assert delta["metric_deltas"][0]["outcome"] == "improvement"


def test_comparison_uses_score_delta_when_pass_state_is_unchanged(lab):
    _, revision = lab.create_suite(
        name="soft-score",
        items=[{"id": "item", "record_id": "record"}],
        primary_metric="reward",
        direction="maximize",
    )
    base = lab.jobs.launch(
        suite_revision_id=revision.id,
        subject={"type": "model", "ref": "base", "revision": "1"},
        request={"scores": {"item": 0.9}, "pass_threshold": 0.5},
    )
    candidate = lab.jobs.launch(
        suite_revision_id=revision.id,
        subject={"type": "model", "ref": "candidate", "revision": "1"},
        request={"scores": {"item": 0.6}, "pass_threshold": 0.5},
    )
    lab.jobs.wait(base.evaluation.id, 5)
    lab.jobs.wait(candidate.evaluation.id, 5)

    comparison = lab.compare(base.evaluation.id, candidate.evaluation.id)

    assert comparison["sample_deltas"][0]["base_passed"] is True
    assert comparison["sample_deltas"][0]["candidate_passed"] is True
    assert comparison["sample_deltas"][0]["outcome"] == "regression"


def test_comparison_preserves_repeated_logical_record_occurrences(lab):
    _, revision = lab.create_suite(
        name="repeated-records",
        items=[
            {"id": "first-occurrence", "record_id": "shared-record"},
            {"id": "second-occurrence", "record_id": "shared-record"},
        ],
    )
    base = lab.jobs.launch(
        suite_revision_id=revision.id,
        subject={"type": "model", "ref": "base", "revision": "1"},
        request={"scores": {"first-occurrence": 1.0, "second-occurrence": 0.0}},
    )
    candidate = lab.jobs.launch(
        suite_revision_id=revision.id,
        subject={"type": "model", "ref": "candidate", "revision": "1"},
        request={"scores": {"first-occurrence": 0.0, "second-occurrence": 1.0}},
    )
    lab.jobs.wait(base.evaluation.id, 5)
    lab.jobs.wait(candidate.evaluation.id, 5)

    comparison = lab.compare(base.evaluation.id, candidate.evaluation.id)

    assert len(comparison["sample_deltas"]) == 2
    assert [delta["record_id"] for delta in comparison["sample_deltas"]] == [
        "shared-record",
        "shared-record",
    ]
    assert {delta["suite_item_id"] for delta in comparison["sample_deltas"]} == {
        "first-occurrence",
        "second-occurrence",
    }
    assert len({delta["occurrence_key"] for delta in comparison["sample_deltas"]}) == 2
    assert comparison["counts"]["regression"] == 1
    assert comparison["counts"]["improvement"] == 1


def test_cancel_and_retry_are_persistent(lab):
    suite, revision = lab.create_suite(
        name="long",
        items=[{"id": str(index), "expected": index} for index in range(25)],
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "fixture", "revision": "1"},
        request={
            "delay_ms": 10,
            "scores": {str(index): 1.0 for index in range(25)},
        },
    )
    deadline = time.time() + 2
    while lab.jobs.get(launch.evaluation.id).status == "queued" and time.time() < deadline:
        time.sleep(0.005)
    lab.jobs.cancel(launch.evaluation.id)
    cancelled = lab.jobs.wait(launch.evaluation.id, 5)
    assert cancelled.status == "cancelled"
    retried = lab.jobs.retry(cancelled.id)
    assert retried.retry_count == 1
    completed = lab.jobs.wait(cancelled.id, 5)
    assert completed.status == "completed"


def test_restart_marks_inflight_evaluations_interrupted(tmp_path):
    from halo_forge.run_db import RunDatabase
    from halo_forge.evaluation_lab import EvaluationLabService

    path = tmp_path / "runs.db"
    first = RunDatabase(str(path))
    suite = first.create_benchmark_suite(name="suite")
    revision = first.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="hash",
        items=[{"id": "x"}],
        primary_metric="score",
        direction="maximize",
    )
    evaluation = first.create_evaluation(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        adapter_version="1",
        subject_type="model",
        subject_ref="m",
        subject_hash="subject",
        reuse_key="reuse",
        request={"worker": {"pid": 99_999_999, "worker_id": "dead-worker"}},
    )
    first.update_evaluation(evaluation.id, status="running", stage="evaluating")
    first.close()
    reopened = RunDatabase(str(path))
    assert reopened.get_evaluation(evaluation.id).status == "running"
    service = EvaluationLabService(reopened, tmp_path / "evaluations")
    interrupted = reopened.get_evaluation(evaluation.id)
    assert interrupted.status == "interrupted"
    assert interrupted.stage == "interrupted"
    service.shutdown()
    reopened.close()


def test_second_live_evaluation_manager_does_not_interrupt_owner(tmp_path):
    from halo_forge.evaluation_lab import EvaluationLabService
    from halo_forge.run_db import RunDatabase

    path = tmp_path / "shared.db"
    owner_db = RunDatabase(str(path))
    owner = EvaluationLabService(owner_db, tmp_path / "owner-evaluations")
    _, revision = owner.create_suite(
        name="live-owner",
        items=[{"id": str(index), "expected": index} for index in range(10)],
    )
    launch = owner.launch_evaluation(
        suite_revision_id=revision.id,
        subject={"type": "model", "ref": "live-model"},
        request={
            "delay_ms": 10,
            "scores": {str(index): 1.0 for index in range(10)},
        },
    )

    observer_db = RunDatabase(str(path))
    observer = EvaluationLabService(observer_db, tmp_path / "observer-evaluations")
    assert observer_db.get_evaluation(launch.evaluation.id).status in {"queued", "running"}
    assert owner.jobs.wait(launch.evaluation.id, 5).status == "completed"
    observer.shutdown()
    observer_db.close()
    owner.shutdown()
    owner_db.close()


def test_two_detached_services_share_one_global_evaluation_slot(tmp_path):
    from halo_forge.evaluation_lab import (
        EvaluationAdapter,
        EvaluationAdapterRegistry,
        EvaluationAdapterResult,
        EvaluationLabService,
        EvaluationMetric,
        EvaluationSample,
    )
    from halo_forge.run_db import RunDatabase

    class ConcurrencyProbeAdapter(EvaluationAdapter):
        adapter_id = "concurrency-probe"
        adapter_version = "1"

        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0
            self.calls = 0

        def evaluate(self, context, revision, subject, request):
            with self.lock:
                self.active += 1
                self.calls += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.12)
                return EvaluationAdapterResult(
                    metrics=[EvaluationMetric(name="score", value=1.0)],
                    samples=[
                        EvaluationSample(
                            suite_item_id="one",
                            record_id=subject.subject_ref,
                            score=1.0,
                            passed=True,
                        )
                    ],
                )
            finally:
                with self.lock:
                    self.active -= 1

    path = tmp_path / "shared-evaluations.db"
    first_db = RunDatabase(str(path))
    second_db = RunDatabase(str(path))
    adapter = ConcurrencyProbeAdapter()
    first = EvaluationLabService(
        first_db,
        tmp_path / "artifacts",
        registry=EvaluationAdapterRegistry([adapter]),
    )
    second = EvaluationLabService(
        second_db,
        tmp_path / "artifacts",
        registry=EvaluationAdapterRegistry([adapter]),
    )
    _, revision = first.create_suite(name="serialized", items=[{"id": "one"}])
    first_launch = first.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id=adapter.adapter_id,
        subject={"type": "model", "ref": "first-model"},
        submit=False,
    )
    second_launch = second.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id=adapter.adapter_id,
        subject={"type": "model", "ref": "second-model"},
        submit=False,
    )
    start = threading.Barrier(3)
    errors = []

    def detached_worker(service, evaluation_id):
        try:
            start.wait(timeout=2)
            service.jobs.run_queued(evaluation_id)
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    workers = [
        threading.Thread(
            target=detached_worker,
            args=(first, first_launch.evaluation.id),
        ),
        threading.Thread(
            target=detached_worker,
            args=(second, second_launch.evaluation.id),
        ),
    ]
    for worker in workers:
        worker.start()
    start.wait(timeout=2)
    for worker in workers:
        worker.join(timeout=5)

    assert errors == []
    assert all(not worker.is_alive() for worker in workers)
    assert first.jobs.get(first_launch.evaluation.id).status == "completed"
    assert second.jobs.get(second_launch.evaluation.id).status == "completed"
    assert adapter.calls == 2
    assert adapter.max_active == 1
    first.shutdown()
    second.shutdown()
    first_db.close()
    second_db.close()


def test_queued_worker_interrupts_dead_global_owner_and_progresses(lab):
    _, revision = _suite(lab)
    dead = lab.db.create_evaluation(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        adapter_version="2",
        subject_type="model",
        subject_ref="dead-model",
        subject_hash="dead-subject",
        reuse_key="dead-reuse",
        request={"worker": {"pid": 99_999_999, "worker_id": "dead-owner"}},
    )
    lab.db.update_evaluation(dead.id, status="running", stage="evaluating")
    queued = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "next-model"},
        request={"scores": {"a": 1, "b": 1, "c": 1, "d": 1}},
        submit=False,
    )

    completed = lab.jobs.run_queued(queued.evaluation.id)

    assert completed.status == "completed"
    assert lab.db.get_evaluation(dead.id).status == "interrupted"


def test_retry_recovers_artifact_published_before_catalog_completion(lab):
    _, revision = _suite(lab)
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "recover", "revision": "pinned"},
        request={"scores": {"a": 1, "b": 1, "c": 1, "d": 1}},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    with lab.db._lock:
        lab.db._conn.execute(
            "DELETE FROM evaluation_metrics WHERE evaluation_id = ?", (completed.id,)
        )
        lab.db._conn.execute(
            "UPDATE evaluations SET status = 'interrupted', stage = 'interrupted' WHERE id = ?",
            (completed.id,),
        )
        lab.db._conn.commit()
    lab.jobs.retry(completed.id)
    recovered = lab.jobs.wait(completed.id, 5)
    assert recovered.status == "completed"
    assert len(lab.db.list_evaluation_metrics(completed.id)) == 1


def test_legacy_lm_eval_summary_import(lab, tmp_path):
    suite, revision = lab.create_suite(
        name="legacy",
        items=[{"adapter": "lm_eval", "task": "mmlu"}],
        primary_metric="average_score",
    )
    summary = tmp_path / "lm_eval_summary.json"
    summary.write_text(
        json.dumps(
            {
                "model_name": "legacy",
                "task_results": [
                    {
                        "task": "mmlu",
                        "primary_metric": "acc",
                        "value": 0.7,
                        "higher_is_better": True,
                        "all_metrics": {"acc": 0.7},
                        "error": None,
                    }
                ],
            }
        )
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        subject={"type": "model", "ref": "legacy/model", "revision": "pinned"},
        request={"legacy_summary_path": str(summary)},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    assert completed.status == "completed"
    assert completed.adapter_id == "lm-eval"
    metrics = lab.db.list_evaluation_metrics(completed.id)
    assert {(metric.name, metric.value) for metric in metrics} == {
        ("average_score", 0.7),
        ("acc", 0.7),
    }
    sample = lab.db.list_evaluation_samples(completed.id)[0]
    assert sample.evidence_kind == "legacy_aggregate"
    assert sample.valid is True
    assert sample.mineable is False
    assert sample.passed is None


def test_verifier_adapter_standardizes_existing_verifier_evidence(lab):
    suite, revision = lab.create_suite(
        name="tool-json",
        items=[
            {"id": "valid", "record_id": "valid", "verifier": "json_structure"},
            {"id": "invalid", "record_id": "invalid", "verifier": "json_structure"},
        ],
        primary_metric="verified_reward",
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="verifier",
        subject={"type": "model", "ref": "tool-model", "revision": "pinned"},
        request={"outputs": {"valid": '{"name": "search"}', "invalid": "not json"}},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    samples = lab.db.list_evaluation_samples(completed.id)
    assert [sample.passed for sample in samples] == [True, False]
    assert samples[0].verifier_trace["reward"] == 1.0


def test_benchmark_adapter_wraps_existing_router(lab, monkeypatch):
    import huggingface_hub
    import halo_forge.benchmark

    calls = []
    snapshots = []

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: snapshots.append(kwargs) or "/models/code-model-pinned",
    )

    def fake_run_benchmark(**kwargs):
        calls.append(kwargs)
        return {
            "benchmark": kwargs["benchmark"],
            "backend": "native",
            "metrics": {"pass_at_1": 0.75},
            "samples": 4,
        }

    monkeypatch.setattr(halo_forge.benchmark, "run_benchmark", fake_run_benchmark)
    suite, revision = lab.create_suite(
        name="code",
        items=[{"id": "humaneval", "benchmark": "humaneval"}],
        primary_metric="pass_at_1",
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="benchmark",
        subject={"type": "model", "ref": "code-model", "revision": "pinned"},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    assert completed.status == "completed"
    assert calls[0]["benchmark"] == "humaneval"
    assert calls[0]["model"] == "/models/code-model-pinned"
    assert snapshots == [
        {
            "repo_id": "code-model",
            "revision": "pinned",
            "local_files_only": False,
        }
    ]
    metrics = lab.db.list_evaluation_metrics(completed.id)
    assert [(metric.name, metric.value) for metric in metrics] == [
        ("pass_at_1", 0.75),
        ("pass_at_1", 0.75),
    ]


def test_benchmark_adapter_publishes_native_per_prompt_evidence(lab, monkeypatch):
    import halo_forge.benchmark

    def fake_run_benchmark(**kwargs):
        return {
            "benchmark": kwargs["benchmark"],
            "backend": "native",
            "metrics": {"pass_at_1": 0.5},
            "samples": 2,
            "sample_results": [
                {
                    "prompt": "first prompt",
                    "success": True,
                    "correct_count": 2,
                    "metadata": {"task_id": "task-one", "category": "easy"},
                },
                {
                    "prompt": "second prompt",
                    "success": False,
                    "correct_count": 0,
                    "metadata": {"task_id": "task-two", "category": "hard"},
                },
            ],
        }

    monkeypatch.setattr(halo_forge.benchmark, "run_benchmark", fake_run_benchmark)
    _, revision = lab.create_suite(
        name="code-samples",
        items=[{"id": "humaneval", "benchmark": "humaneval"}],
        primary_metric="pass_at_1",
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="benchmark",
        subject={"type": "model", "ref": "local-code-model"},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    samples = lab.db.list_evaluation_samples(completed.id)

    assert [sample.record_id for sample in samples] == ["task-one", "task-two"]
    assert [sample.input for sample in samples] == ["first prompt", "second prompt"]
    assert [sample.output for sample in samples] == [None, None]
    assert [sample.score for sample in samples] == [1.0, 0.0]
    assert [sample.passed for sample in samples] == [True, False]
    assert all(not sample.metadata.get("aggregate_benchmark_result") for sample in samples)
    assert samples[0].metadata["dataset_suite_item_id"] == "humaneval"


def test_native_python_benchmark_router_preserves_existing_sample_results(monkeypatch):
    import halo_forge.benchmark as benchmark_module

    expected_samples = [
        {
            "prompt": "solve it",
            "success": True,
            "correct_count": 1,
            "metadata": {"task_id": "problem-one"},
        }
    ]

    class FakeBenchmark:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            return type(
                "Result",
                (),
                {
                    "pass_at_k": {1: 1.0, 5: 1.0, 10: 1.0},
                    "pass_rate": 1.0,
                    "total": 1,
                    "samples": expected_samples,
                },
            )()

    monkeypatch.setattr(
        benchmark_module,
        "_load_python_benchmark_records",
        lambda **kwargs: [{"prompt": "solve it"}],
    )
    monkeypatch.setattr(benchmark_module, "_get_python_dataset_verifier", lambda **kwargs: object())
    monkeypatch.setattr(benchmark_module, "Benchmark", FakeBenchmark)

    result = benchmark_module._run_python_dataset_benchmark(
        model="model", benchmark="humaneval", limit=1
    )

    assert result["sample_results"] == expected_samples


def test_registry_advertises_all_v2_adapter_families_and_shapes_evidence(lab):
    registered = {item["id"] for item in lab.registry.list()}
    assert {"code", "verifier", "reasoning", "tool", "vlm", "audio"}.issubset(registered)
    for adapter_id in ("code", "reasoning", "tool", "vlm", "audio"):
        suite, revision = lab.create_suite(
            name=f"{adapter_id}-evidence",
            items=[
                {
                    "id": "one",
                    "adapter": adapter_id,
                    "record_id": f"{adapter_id}-record",
                    "input": "input",
                    "expected": "expected",
                    "output": "expected",
                }
            ],
        )
        launch = lab.jobs.launch(
            suite_revision_id=revision.id,
            subject={"kind": "model", "value": f"{adapter_id}/model", "revision": "pinned"},
        )
        completed = lab.jobs.wait(launch.evaluation.id, 5)
        assert completed.adapter_id == adapter_id
        sample = lab.db.list_evaluation_samples(completed.id)[0]
        assert sample.to_dict()["record_id"] == f"{adapter_id}-record"
        assert sample.passed is True


def test_composite_suite_groups_item_adapters_and_preserves_order(lab):
    suite, revision = lab.create_suite(
        name="mixed",
        items=[
            {
                "id": "text",
                "adapter": "dataset_fixture",
                "expected": "ok",
                "output": "ok",
            },
            {
                "id": "tool",
                "adapter": "tool_use",
                "expected": {"name": "search"},
                "output": {"name": "search"},
            },
            {
                "id": "audio",
                "adapter": "audio",
                "expected": "transcript",
                "output": "transcript",
            },
        ],
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        subject={"kind": "model", "value": "mixed/model", "revision": "pinned"},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    assert completed.status == "completed"
    assert completed.adapter_id == "suite"
    assert completed.adapter_version.startswith("2+")
    samples = lab.db.list_evaluation_samples(completed.id)
    assert [sample.suite_item_id for sample in samples] == ["text", "tool", "audio"]
    assert all(sample.passed for sample in samples)


def test_comparison_rejects_different_suite_revisions(lab):
    first_suite, first = lab.create_suite(name="first", items=[{"id": "a"}])
    second_suite, second = lab.create_suite(name="second", items=[{"id": "a"}])
    one = lab.jobs.launch(
        suite_revision_id=first.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "one", "revision": "1"},
        request={"scores": {"a": 1.0}},
    )
    two = lab.jobs.launch(
        suite_revision_id=second.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "two", "revision": "1"},
        request={"scores": {"a": 1.0}},
    )
    lab.jobs.wait(one.evaluation.id, 5)
    lab.jobs.wait(two.evaluation.id, 5)
    from halo_forge.evaluation_lab import EvaluationLabError

    with pytest.raises(EvaluationLabError, match="same benchmark suite revision"):
        lab.compare(one.evaluation.id, two.evaluation.id)


def test_per_example_evidence_persists_generation_provenance(lab):
    _, revision = lab.create_suite(
        name="evidence-provenance",
        items=[
            {
                "id": "one",
                "record_id": "record-one",
                "input": "prompt",
                "expected": "answer",
                "output": "answer",
                "input_tokens": 3,
                "output_tokens": 1,
                "finish_reason": "stop",
            }
        ],
        evaluator_versions={"verifier": "2"},
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        subject={"type": "model", "ref": "fixture/model", "revision": "pinned"},
        request={
            "generation_seed": 42,
            "chat_template_hash": "template-sha256",
            "runtime_versions": {"transformers": "test-version"},
        },
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    sample = lab.db.list_evaluation_samples(completed.id)[0]

    assert sample.evidence_kind == "fixture"
    assert sample.valid is True and sample.mineable is True
    assert sample.generation_seed == 42
    assert sample.input_tokens == 3 and sample.output_tokens == 1
    assert sample.finish_reason == "stop"
    assert sample.score_direction == "maximize"
    assert sample.score_threshold == 0.5
    assert sample.coverage == 1.0
    assert sample.template_hash == "template-sha256"
    assert sample.runtime_versions["transformers"] == "test-version"
    assert sample.runtime_versions["verifier"] == "2"
    assert sample.runtime_versions["evaluation_adapter:dataset"] == "3"
    artifact_sample = json.loads(
        Path(completed.artifact_path, "samples.jsonl").read_text(encoding="utf-8")
    )
    assert artifact_sample["mineable"] is True
    assert artifact_sample["metadata"]["evidence"]["generation_seed"] == 42


def test_aggregate_benchmark_result_is_valid_but_never_behavioral_or_mineable(lab, monkeypatch):
    import halo_forge.benchmark

    monkeypatch.setattr(
        halo_forge.benchmark,
        "run_benchmark",
        lambda **kwargs: {
            "benchmark": kwargs["benchmark"],
            "backend": "native",
            "metrics": {"pass_at_1": 0.75},
            "samples": 4,
        },
    )
    _, revision = lab.create_suite(
        name="aggregate-only",
        items=[{"id": "humaneval", "benchmark": "humaneval"}],
        primary_metric="pass_at_1",
    )
    launch = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="benchmark",
        subject={"type": "model", "ref": "code/model"},
    )
    completed = lab.jobs.wait(launch.evaluation.id, 5)
    sample = lab.db.list_evaluation_samples(completed.id)[0]

    assert sample.evidence_kind == "aggregate_benchmark"
    assert sample.valid is True
    assert sample.mineable is False
    assert sample.passed is None
    assert sample.score == 0.75


def test_comparison_withholds_non_mineable_evidence_and_reports_gap(lab, monkeypatch):
    import halo_forge.benchmark

    monkeypatch.setattr(
        halo_forge.benchmark,
        "run_benchmark",
        lambda **kwargs: {
            "benchmark": kwargs["benchmark"],
            "backend": "native",
            "metrics": {"score": 1.0 if "base" in kwargs["model"] else 0.0},
            "samples": 1,
        },
    )
    _, revision = lab.create_suite(
        name="aggregate-compare",
        items=[{"id": "task", "benchmark": "task"}],
    )
    base = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="benchmark",
        subject={"type": "model", "ref": "base/model"},
    )
    candidate = lab.jobs.launch(
        suite_revision_id=revision.id,
        adapter_id="benchmark",
        subject={"type": "model", "ref": "candidate/model"},
    )
    lab.jobs.wait(base.evaluation.id, 5)
    lab.jobs.wait(candidate.evaluation.id, 5)

    comparison = lab.compare(base.evaluation.id, candidate.evaluation.id)

    assert comparison["sample_deltas"] == []
    assert comparison["evidence_summary"] == {
        "base_total": 1,
        "candidate_total": 1,
        "comparable": 0,
        "incomplete": 1,
        "complete": False,
        "failure_mining_eligible": 0,
    }
    assert comparison["evidence_gaps"][0]["mineable"] is False
    assert "non_mineable" in comparison["evidence_gaps"][0]["reason"]


def test_comparison_page_bounds_deltas_and_evidence_gaps(lab):
    _, revision = _suite(lab)
    launches = []
    for subject, scores in (
        ("base", {"a": 1, "b": 0, "c": 1, "d": 0}),
        ("candidate", {"a": 0, "b": 1, "c": 0, "d": 1}),
    ):
        launches.append(
            lab.jobs.launch(
                suite_revision_id=revision.id,
                adapter_id="dataset",
                subject={"type": "model", "ref": subject, "revision": "1"},
                request={"scores": scores},
            )
        )
    for launch in launches:
        lab.jobs.wait(launch.evaluation.id, 5)

    page = lab.compare_page(
        launches[0].evaluation.id,
        launches[1].evaluation.id,
        offset=1,
        limit=2,
    )

    assert page["sample_deltas"]["total"] == 4
    assert len(page["sample_deltas"]["items"]) == 2
    assert page["sample_deltas"]["offset"] == 1
    assert page["evidence_gaps"]["total"] == 0
