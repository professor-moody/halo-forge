from __future__ import annotations

from halo_forge.run_db import RunDatabase


def _evaluation(database: RunDatabase, revision_id: str, evaluation_id: str, samples):
    value = database.create_evaluation(
        suite_revision_id=revision_id,
        adapter_id="test",
        adapter_version="1",
        subject_type="model",
        subject_ref=evaluation_id,
        subject_hash=f"hash-{evaluation_id}",
        reuse_key=f"reuse-{evaluation_id}",
        request={},
        evaluation_id=evaluation_id,
    )
    return database.complete_evaluation(
        value.id,
        metrics=[{"name": "accuracy", "value": 1, "direction": "maximize"}],
        samples=samples,
        result={},
        artifact_path=f"/{evaluation_id}",
    )


def test_bounded_evaluation_pair_join_uses_zero_based_occurrences():
    database = RunDatabase(":memory:")
    suite = database.create_benchmark_suite(name="development", purpose="development")
    revision = database.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="suite-hash",
        items=[{"id": "item"}],
        primary_metric="accuracy",
        direction="maximize",
    )
    shared = {"valid": True, "mineable": True, "evidence_kind": "per_example"}
    _evaluation(
        database,
        revision.id,
        "base",
        [
            {"suite_item_id": "fallback", "record_id": "repeat", "output": "b0", **shared},
            {"suite_item_id": "fallback", "record_id": "repeat", "output": "b1", **shared},
            {"suite_item_id": "base-only", "output": "base", **shared},
        ],
    )
    _evaluation(
        database,
        revision.id,
        "candidate",
        [
            {"suite_item_id": "fallback", "record_id": "repeat", "output": "c0", **shared},
            {"suite_item_id": "fallback", "record_id": "repeat", "output": "c1", **shared},
            {"suite_item_id": "candidate-only", "output": "candidate", **shared},
        ],
    )
    assert database.count_evaluation_sample_pairs("base", "candidate") == 4
    first_page = database.list_evaluation_sample_pairs(
        "base", "candidate", limit=2, offset=0
    )
    second_page = database.list_evaluation_sample_pairs(
        "base", "candidate", limit=2, offset=2
    )
    joined = first_page + second_page
    repeats = [value for value in joined if value["logical_record_id"] == "repeat"]
    assert [value["occurrence"] for value in repeats] == [0, 1]
    assert repeats[0]["base"].output == "b0"
    assert repeats[0]["candidate"].output == "c0"
    assert any(value["base"] is None for value in joined)
    assert any(value["candidate"] is None for value in joined)
    database.close()
