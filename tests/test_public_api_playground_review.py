from __future__ import annotations

from pathlib import Path

import pytest

from halo_forge.public_api.service import PublicApiService
from halo_forge.run_db import RunDatabase


def test_persistent_playground_review_creates_suite_revision_and_dataset_source_draft(
    tmp_path: Path,
) -> None:
    database = RunDatabase(str(tmp_path / "runs.db"))
    service = PublicApiService(
        database=database,
        base_path=tmp_path,
        dataset_storage_root=tmp_path / "datasets",
        evaluation_storage_root=tmp_path / "evaluations",
        artifact_storage_root=tmp_path / "artifacts",
    )
    session = service.create_playground_session({"name": "Reviewed conversation"})
    session = service.append_playground_message(
        session["id"],
        {"role": "user", "content": "What is two plus two?"},
    )
    session = service.append_playground_message(
        session["id"],
        {
            "role": "assistant",
            "content": "Four.",
            "generation": {"seed": 42},
        },
    )
    message_ids = [value["id"] for value in session["messages"]]

    benchmark = service.review_playground_session(
        session["id"],
        {
            "kind": "benchmark_suite",
            "message_ids": message_ids,
            "review_note": "Reviewed expected answer",
        },
    )
    draft = service.review_playground_session(
        session["id"],
        {
            "kind": "dataset_source",
            "message_ids": message_ids,
            "review_note": "Reviewed training example",
        },
    )

    assert benchmark["benchmark_suite_revision_id"]
    assert benchmark["starts_training"] is False
    assert (
        database.get_benchmark_suite_revision(benchmark["benchmark_suite_revision_id"]) is not None
    )
    assert draft["dataset_source_draft_id"]
    assert Path(draft["records_path"]).is_file()
    assert draft["starts_training"] is False
    database.close()


def test_playground_explicit_base_candidate_pair_publishes_real_preference_row(
    tmp_path: Path,
) -> None:
    database = RunDatabase(str(tmp_path / "runs.db"))
    service = PublicApiService(
        database=database,
        base_path=tmp_path,
        dataset_storage_root=tmp_path / "datasets",
        evaluation_storage_root=tmp_path / "evaluations",
        artifact_storage_root=tmp_path / "artifacts",
        review_storage_root=tmp_path / "reviews",
    )
    review = service._review_engine()
    _, schema_revision = review.create_schema(
        name="Playground preference",
        modality="preference",
        task_type="pairwise",
    )
    session = service.create_playground_session({"name": "Compared conversation"})
    for message in (
        {"role": "user", "content": "Name the capital of France."},
        {
            "role": "assistant",
            "content": "The capital is Lyon.",
            "generation": {"model": "base", "seed": 7},
        },
        {
            "role": "assistant",
            "content": "The capital is Paris.",
            "generation": {"model": "candidate", "seed": 7},
        },
    ):
        session = service.append_playground_message(session["id"], message)
    by_content = {value["content"]: value for value in session["messages"]}

    with pytest.raises(ValueError, match="require explicit persisted base/candidate"):
        service.review_playground_session(
            session["id"],
            {
                "kind": "review_queue",
                "schema_revision_id": schema_revision.id,
                "review_note": "Do not infer preference candidates",
                "message_ids": [value["id"] for value in session["messages"]],
            },
        )

    result = service.review_playground_session(
        session["id"],
        {
            "kind": "review_queue",
            "schema_revision_id": schema_revision.id,
            "review_note": "Compare the persisted base and candidate responses",
            "policy": {"mode": "one_pass"},
            "pairings": [
                {
                    "prompt_message_id": by_content["Name the capital of France."]["id"],
                    "base_message_id": by_content["The capital is Lyon."]["id"],
                    "candidate_message_id": by_content["The capital is Paris."]["id"],
                }
            ],
        },
    )

    assert result["kind"] == "review_queue"
    assert result["pairing_count"] == 1
    queue_id = result["review_queue_id"]
    item = review.list_items(queue_id)[0]
    assert item.record["prompt"] == "Name the capital of France."
    assert item.record["alternatives"] == [
        "The capital is Lyon.",
        "The capital is Paris.",
    ]
    assert item.record["metadata"]["comparison"]["representation"] == (
        "playground_base_candidate.v1"
    )

    review.submit_event(
        item.id,
        "label",
        {
            "chosen": "The capital is Paris.",
            "rejected": "The capital is Lyon.",
        },
        idempotency_key="choose-candidate",
        expected_active_event_id=None,
    )
    label_revision = review.publish_label_set(queue_id)
    assert review.render_label_set(label_revision.id)["records"] == [
        {
            "prompt": "Name the capital of France.",
            "chosen": "The capital is Paris.",
            "rejected": "The capital is Lyon.",
        }
    ]
    database.close()
