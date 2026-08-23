from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from halo_forge.review_lab import (
    ReviewConflictError,
    ReviewEligibilityError,
    ReviewLabService,
    ReviewStateError,
    ReviewValidationError,
)
from halo_forge.data_lab.models import validate_record
from halo_forge.public_api.service import PublicApiService
from halo_forge.review_lab.registry import (
    OutputAdapterRegistry,
    validate_schema_definition,
)
from halo_forge.run_db import RunDatabase
from halo_forge.run_db.schema import SCHEMA_VERSION


@pytest.fixture
def service(tmp_path):
    database = RunDatabase(":memory:")
    value = ReviewLabService(database, tmp_path / "reviews")
    try:
        yield value
    finally:
        database.close()


def _queue(service: ReviewLabService, *, task="categorical", definition=None, count=1):
    schema, revision = service.create_schema(
        name=f"{task} labels",
        modality="text" if task != "pairwise" else "preference",
        task_type=task,
        definition=definition
        or ({"labels": ["correct", "incorrect"]} if task == "categorical" else {}),
    )
    records = [
        {
            "record_id": f"r{index}",
            "record": {
                "prompt": f"prompt {index}",
                "alternatives": [f"a{index}", f"b{index}"],
            },
            "source": {"kind": "dataset_version", "ref": "version-1", "split": "train"},
        }
        for index in range(count)
    ]
    batch = service.create_acquisition(records, name="proposal")
    queue = service.create_queue(batch.id, revision.id, name=f"{task} queue")
    return schema, revision, batch, queue


def test_schema_v9_and_deterministic_acquisition(service):
    assert SCHEMA_VERSION == 23
    tables = {
        row[0]
        for row in service.db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "annotation_schemas",
        "annotation_schema_revisions",
        "acquisition_batches",
        "acquisition_candidates",
        "review_queues",
        "review_items",
        "review_events",
        "review_suggestions",
        "label_sets",
        "label_set_revisions",
        "label_set_items",
    }.issubset(tables)
    indexes = {
        row[0]
        for row in service.db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert {
        "idx_review_items_queue_status",
        "idx_review_events_item_pass_page",
        "idx_review_suggestions_item_page",
    }.issubset(indexes)

    records = [
        {
            "record_id": "b",
            "record": {"prompt": "b"},
            "evidence": {"passed": False, "score": 0.2},
            "source": {"kind": "evaluation", "purpose": "development"},
        },
        {
            "record_id": "a",
            "record": {"prompt": "a"},
            "evidence": {"passed": False, "score": 0.1},
            "source": {"kind": "evaluation", "purpose": "development"},
        },
    ]
    first = service.create_acquisition(
        records,
        strategies=[{"kind": "candidate_failure", "quota": 1}],
        seed=7,
    )
    second = service.create_acquisition(
        reversed(records),
        strategies=[{"kind": "candidate_failure", "quota": 1}],
        seed=7,
    )
    assert first.id == second.id
    assert service.list_acquisition_candidates(first.id)[0].record_id == "a"

    with pytest.raises(ReviewEligibilityError, match="protected_split:test"):
        service.create_acquisition(
            [{"record": {"prompt": "secret"}, "source": {"kind": "dataset", "split": "test"}}]
        )


def test_pass_specific_item_pagination_reaches_beyond_one_thousand(service):
    _, _, _, queue = _queue(service, count=1_205)
    api = PublicApiService(database=service.db, review_lab=service)

    page = api.list_review_items(
        queue.id,
        pass_number=1,
        limit=25,
        offset=1_175,
    )

    assert page["total"] == 1_205
    assert page["limit"] == 25
    assert page["offset"] == 1_175
    assert [item["ordinal"] for item in page["items"]] == list(range(1_175, 1_200))
    assert api.list_review_items(queue.id, pass_number=2)["total"] == 0


def test_event_and_suggestion_pages_have_sql_backed_totals(service):
    _, _, _, queue = _queue(service)
    item = service.list_items(queue.id)[0]
    event_values = []
    active_event_id = None
    for index in range(12):
        event = service.submit_event(
            item.id,
            "note",
            {"note": f"audit note {index}"},
            idempotency_key=f"paged-note-{index}",
            expected_active_event_id=active_event_id,
        )
        active_event_id = event.id
        event_values.append(event.to_dict())

    suggestion_ids = [
        service.create_suggestion(
            item.id,
            provider="ollama",
            model_revision="reviewer@sha256:abc",
            output={"annotation": {"label": "correct"}, "ordinal": index},
            provenance={"prompt_hash": f"prompt-{index}"},
        ).id
        for index in range(12)
    ]
    api = PublicApiService(database=service.db, review_lab=service)

    event_page = api.list_review_events(item.id, limit=3, offset=7)
    assert event_page == {
        "items": event_values[7:10],
        "total": 12,
        "limit": 3,
        "offset": 7,
    }
    suggestion_page = api.list_review_suggestions(
        item.id,
        pass_number=1,
        limit=3,
        offset=7,
    )
    assert suggestion_page["total"] == 12
    assert suggestion_page["limit"] == 3
    assert suggestion_page["offset"] == 7
    assert [value["id"] for value in suggestion_page["items"]] == suggestion_ids[7:10]
    assert all(value["output"] is None for value in suggestion_page["items"])
    assert api.list_review_suggestions(item.id, pass_number=2)["total"] == 0


def test_events_are_idempotent_optimistic_append_only_and_atomic(service):
    _, _, _, queue = _queue(service, count=2)
    first, second = service.list_items(queue.id)
    event = service.submit_event(
        first.id,
        "label",
        {"label": "correct"},
        idempotency_key="first-label",
        expected_active_event_id=None,
        reviewer_key="operator-a",
    )
    assert event.reviewer_key == "operator-a"
    assert (
        service.submit_event(
            first.id,
            "label",
            {"label": "correct"},
            idempotency_key="first-label",
            expected_active_event_id=None,
            reviewer_key="operator-a",
        ).id
        == event.id
    )
    with pytest.raises(ReviewConflictError, match="idempotency"):
        service.submit_event(
            first.id,
            "label",
            {"label": "incorrect"},
            idempotency_key="first-label",
            expected_active_event_id=event.id,
            reviewer_key="operator-a",
        )
    with pytest.raises(ReviewConflictError, match="stale"):
        service.submit_event(
            first.id,
            "correct",
            {"annotation": {"label": "incorrect"}, "reason": "rechecked"},
            idempotency_key="stale-correction",
            expected_active_event_id=None,
            supersedes_event_id=event.id,
        )

    with pytest.raises(ReviewConflictError):
        service.submit_event_batch(
            queue.id,
            [
                {
                    "item_id": second.id,
                    "event_type": "label",
                    "payload": {"label": "correct"},
                    "idempotency_key": "batch-duplicate",
                    "expected_active_event_id": None,
                },
                {
                    "item_id": first.id,
                    "event_type": "flag",
                    "payload": {},
                    "idempotency_key": "batch-duplicate",
                    "expected_active_event_id": event.id,
                },
            ],
        )
    assert service.get_item(second.id).active_event_id is None
    rows = service.db._conn.execute(
        "SELECT reviewer_key FROM review_events WHERE item_id=?", (first.id,)
    ).fetchall()
    assert [row[0] for row in rows] == ["operator-a"]

    service.db._conn.execute(
        """UPDATE review_items
           SET status='pending',active_event_id=NULL,projection_json='{}' WHERE id=?""",
        (first.id,),
    )
    service.db._conn.commit()
    assert service.rebuild_queue_projections(queue.id) == {
        "queue_id": queue.id,
        "rebuilt": 2,
    }
    rebuilt = service.get_item(first.id)
    assert rebuilt.status == "resolved"
    assert rebuilt.active_event_id == event.id
    assert rebuilt.projection["active_annotation"] == {"label": "correct"}


def test_blinded_two_pass_conflict_and_adjudication(service):
    _, _, _, queue = _queue(service, task="pairwise")
    item = service.list_items(queue.id)[0]
    pass_one = service.submit_event(
        item.id,
        "label",
        {"chosen": 0},
        idempotency_key="p1",
        expected_active_event_id=None,
        reviewer_key="first-pass",
    )
    queue = service.start_second_pass(queue.id)
    blinded = service.get_item(item.id)
    assert blinded.projection["current_pass"] == 2
    pass_two_page = PublicApiService(
        database=service.db,
        review_lab=service,
    ).list_review_items(queue.id, pass_number=2)
    assert pass_two_page["total"] == 1
    assert blinded.projection["pass_1"] == {"hidden": True}
    assert service.list_events(item.id) == []

    pass_two = service.submit_event(
        item.id,
        "label",
        {"chosen": 1},
        idempotency_key="p2",
        expected_active_event_id=pass_one.id,
        reviewer_key="second-pass",
    )
    assert service.get_item(item.id).status == "conflict"
    final = service.adjudicate(
        item.id,
        {"chosen": 0},
        reason="source evidence supports the first answer",
        idempotency_key="adjudicate",
        expected_active_event_id=pass_two.id,
        reviewer_key="adjudicator",
    )
    assert final.reviewer_key == "adjudicator"
    assert service.get_item(item.id).status == "resolved"
    assert service.statistics(queue.id)["two_pass_agreement_rate"] == 0.0


def test_blind_two_pass_suggestions_require_a_reveal_in_each_pass(service):
    _, _, _, queue = _queue(service, task="pairwise")
    item = service.list_items(queue.id)[0]
    first_suggestion = service.create_suggestion(
        item.id,
        provider="ollama",
        model_revision="teacher@pass-1",
        output={"chosen": 0},
    )
    reveal = service.submit_event(
        item.id,
        "reveal_suggestion",
        {"suggestion_id": first_suggestion.id},
        idempotency_key="pass-1-reveal",
        expected_active_event_id=None,
    )
    label = service.submit_event(
        item.id,
        "label",
        {"chosen": 0},
        idempotency_key="pass-1-label",
        expected_active_event_id=reveal.id,
    )
    service.start_second_pass(queue.id)

    assert service.get_suggestion(first_suggestion.id).output is None
    with pytest.raises(ReviewValidationError, match="review pass"):
        service.submit_event(
            item.id,
            "reveal_suggestion",
            {"suggestion_id": first_suggestion.id},
            idempotency_key="pass-2-cannot-reveal-pass-1",
            expected_active_event_id=label.id,
        )
    second_suggestion = service.create_suggestion(
        item.id,
        provider="ollama",
        model_revision="teacher@pass-2",
        output={"chosen": 1},
    )
    second_reveal = service.submit_event(
        item.id,
        "reveal_suggestion",
        {"suggestion_id": second_suggestion.id},
        idempotency_key="pass-2-reveal",
        expected_active_event_id=label.id,
    )
    assert second_reveal.pass_number == 2
    assert service.get_suggestion(second_suggestion.id).output == {"chosen": 1}


def test_suggestion_reveal_and_atomic_label_set_revisions(service, tmp_path):
    _, _, _, queue = _queue(service)
    item = service.list_items(queue.id)[0]
    suggestion = service.create_suggestion(
        item.id,
        provider="ollama",
        model_revision="teacher@sha256:abc",
        output={"label": "correct"},
        provenance={
            "prompt_hash": "prompt-1",
            "temperature": 0,
            "api_key": "must-not-persist",
            "runtime": {"authorization": "must-not-persist", "engine": "ollama"},
        },
    )
    assert "api_key" not in suggestion.provenance
    assert suggestion.provenance["runtime"] == {"engine": "ollama"}
    assert suggestion.id not in item.projection.get("revealed_suggestion_ids", [])
    reveal = service.submit_event(
        item.id,
        "reveal_suggestion",
        {"suggestion_id": suggestion.id},
        idempotency_key="reveal",
        expected_active_event_id=None,
    )
    label = service.submit_event(
        item.id,
        "label",
        {"label": "correct"},
        idempotency_key="label",
        expected_active_event_id=reveal.id,
    )
    revision_one = service.publish_label_set(queue.id)
    assert revision_one.row_count == 1
    assert service.verify_label_set(revision_one.id).valid
    manifest = json.loads(
        (
            tmp_path
            / "reviews"
            / "label-sets"
            / revision_one.label_set_id
            / revision_one.id
            / "manifest.json"
        ).read_text()
    )
    assert manifest["content_hash"] == revision_one.content_hash
    publication_root = (
        tmp_path / "reviews" / "label-sets" / revision_one.label_set_id / revision_one.id
    )
    assert {
        "canonical.jsonl",
        "items.jsonl",
        "lineage.jsonl",
        "statistics.json",
        "provenance.json",
        "exposure.json",
    }.issubset(manifest["checksums"])
    statistics = json.loads((publication_root / "statistics.json").read_text())
    provenance = json.loads((publication_root / "provenance.json").read_text())
    assert statistics["suggestion_compared"] == 1
    assert provenance["suggestions"][0]["model_revision"] == "teacher@sha256:abc"
    lineage = service.list_label_set_items(revision_one.id)[0].lineage
    assert lineage["suggestions"][0]["revealed"] is True

    correction = service.submit_event(
        item.id,
        "correct",
        {"annotation": {"label": "incorrect"}, "reason": "manual recheck"},
        idempotency_key="correction",
        expected_active_event_id=label.id,
        supersedes_event_id=label.id,
    )
    assert correction.id
    assert service.statistics(queue.id)["unpublished_changes"] is True
    revision_two = service.publish_label_set(queue.id)
    assert revision_two.revision_number == 2
    assert revision_two.content_hash != revision_one.content_hash
    assert service.get_label_set_revision(revision_one.id).content_hash == revision_one.content_hash


def test_label_publication_snapshots_concurrent_correction_atomically(tmp_path, monkeypatch):
    database_path = tmp_path / "review.db"
    publisher_database = RunDatabase(str(database_path))
    writer_database = RunDatabase(str(database_path))
    publisher = ReviewLabService(publisher_database, tmp_path / "reviews")
    writer = ReviewLabService(writer_database, tmp_path / "reviews")
    try:
        _, _, _, queue = _queue(publisher)
        item = publisher.list_items(queue.id)[0]
        label = publisher.submit_event(
            item.id,
            "label",
            {"label": "correct"},
            idempotency_key="snapshot-label",
            expected_active_event_id=None,
        )

        render_started = threading.Event()
        correction_finished = threading.Event()
        correction_errors = []
        adapter = publisher.output_adapters.get("metadata.v1")
        adapter_type = type(adapter)
        original_render = adapter_type._render

        def pause_after_item_projection(adapter_self, record, annotation, build_mode):
            rendered = original_render(adapter_self, record, annotation, build_mode)
            render_started.set()
            assert correction_finished.wait(timeout=5), "concurrent correction did not finish"
            return rendered

        monkeypatch.setattr(adapter_type, "_render", pause_after_item_projection)

        def correct_while_publication_reads():
            if not render_started.wait(timeout=5):
                correction_errors.append(AssertionError("publication render did not start"))
                correction_finished.set()
                return
            try:
                writer.submit_event(
                    item.id,
                    "correct",
                    {
                        "annotation": {"label": "incorrect"},
                        "reason": "committed while publication was reading",
                    },
                    idempotency_key="snapshot-correction",
                    expected_active_event_id=label.id,
                    supersedes_event_id=label.id,
                )
            except BaseException as exc:  # surfaced on the test thread below
                correction_errors.append(exc)
            finally:
                correction_finished.set()

        correction_thread = threading.Thread(target=correct_while_publication_reads)
        correction_thread.start()
        try:
            revision = publisher.publish_label_set(queue.id)
        finally:
            render_started.set()
            correction_finished.set()
            correction_thread.join(timeout=5)
        assert correction_thread.is_alive() is False
        assert correction_errors == []

        publication_root = Path(revision.storage_path)
        manifest = json.loads((publication_root / "manifest.json").read_text())
        snapshot_statistics = json.loads((publication_root / "statistics.json").read_text())
        published_item = publisher.list_label_set_items(revision.id)[0]

        assert published_item.annotation == {"label": "correct"}
        assert [event["id"] for event in published_item.lineage["review_events"]] == [label.id]
        assert snapshot_statistics["event_counts"] == {"label": 1}
        assert manifest["event_stream_hash"] == snapshot_statistics["event_stream_hash"]

        current_statistics = publisher.statistics(queue.id)
        assert current_statistics["event_counts"] == {"correct": 1, "label": 1}
        assert current_statistics["event_stream_hash"] != manifest["event_stream_hash"]
        assert current_statistics["unpublished_changes"] is True
    finally:
        writer_database.close()
        publisher_database.close()


@pytest.mark.parametrize(
    ("modality", "task", "definition", "record", "annotation", "expected_key"),
    [
        (
            "text",
            "text_correction",
            {},
            {"prompt": "p", "response": "bad"},
            {"corrected_text": "good"},
            "response",
        ),
        (
            "preference",
            "pairwise",
            {},
            {"prompt": "p", "alternatives": ["a", "b"]},
            {"chosen": 0},
            "chosen",
        ),
        (
            "tool",
            "structured_correction",
            {},
            {"messages": [], "tools": []},
            {"correction": {"expected_calls": [{"name": "x"}]}},
            "expected_calls",
        ),
        (
            "vlm",
            "text_correction",
            {},
            {"image": "image.png", "prompt": "p"},
            {"corrected_text": "answer"},
            "response",
        ),
        (
            "audio",
            "text_correction",
            {},
            {"audio": "audio.wav", "task": "asr"},
            {"corrected_text": "words"},
            "transcript",
        ),
    ],
)
def test_output_adapters_cover_every_first_release_modality(
    service, modality, task, definition, record, annotation, expected_key
):
    _, revision = service.create_schema(
        name=f"{modality}-{task}",
        modality=modality,
        task_type=task,
        definition=definition,
    )
    batch = service.create_acquisition(
        [{"record_id": "record-1", "record": record, "source": {"kind": "imported"}}]
    )
    queue = service.create_queue(batch.id, revision.id)
    item = service.list_items(queue.id)[0]
    first = service.submit_event(
        item.id,
        "label",
        annotation,
        idempotency_key="pass-1",
        expected_active_event_id=None,
    )
    if queue.policy["mode"] == "two_pass":
        service.start_second_pass(queue.id)
        service.submit_event(
            item.id,
            "label",
            annotation,
            idempotency_key="pass-2",
            expected_active_event_id=first.id,
        )
    published = service.publish_label_set(queue.id)
    output = service.render_label_set(published.id)["records"]
    assert output and expected_key in output[0]


@pytest.mark.parametrize(
    ("modality", "task", "definition"),
    [
        ("preference", "categorical", {"labels": ["good", "bad"]}),
        ("preference", "scalar", {"minimum": 0, "maximum": 5}),
        ("tool", "categorical", {"labels": ["valid", "invalid"]}),
        ("tool", "scalar", {"minimum": 0, "maximum": 1}),
    ],
)
def test_non_transforming_preference_and_tool_tasks_default_to_metadata(
    service, modality, task, definition
):
    normalized = validate_schema_definition(modality, task, definition)
    assert normalized["output_adapter_id"] == "metadata.v1"

    _, revision = service.create_schema(
        name=f"{modality} {task}",
        modality=modality,
        task_type=task,
        definition=definition,
    )
    assert revision.definition["output_adapter_id"] == "metadata.v1"


@pytest.mark.parametrize(
    ("task", "annotation", "expected_rejected"),
    [
        ("pairwise", {"chosen": 1, "rejected": 0}, ["base"]),
        ("ranking", {"ranking": [2, 0, 1]}, ["base", "middle"]),
    ],
)
def test_vlm_preference_outputs_preserve_media_and_valid_vlm_shape(
    task, annotation, expected_rejected
):
    adapter = OutputAdapterRegistry().get("preference.v1")
    record = {
        "image": "diagram.png",
        "prompt": "Which answer matches the diagram?",
        "alternatives": ["base", "middle", "best"],
        "metadata": {"camera": "scan"},
    }
    outputs = adapter.render(record, annotation)

    assert [value["rejected"] for value in outputs] == expected_rejected
    for output in outputs:
        assert output["image"] == "diagram.png"
        assert output["prompt"] == record["prompt"]
        assert output["response"] == output["chosen"]
        assert output["alternatives"] == record["alternatives"]
        assert output["metadata"] == record["metadata"]
        validate_record(output, "vlm")
