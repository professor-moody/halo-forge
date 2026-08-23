from __future__ import annotations

import json
from pathlib import Path

from halo_forge.public_api.playground_reviews import (
    create_benchmark_revision_from_turns,
    create_dataset_source_draft_from_turns,
)
from halo_forge.run_db import RunDatabase


def _messages():
    return [
        {"id": "user-1", "role": "user", "content": "What is two plus two?"},
        {
            "id": "assistant-1",
            "role": "assistant",
            "content": "Four.",
            "generation": {"seed": 42},
        },
    ]


def test_reviewed_playground_turns_create_immutable_suite_and_source_draft(
    tmp_path: Path,
) -> None:
    database = RunDatabase(str(tmp_path / "runs.db"))
    benchmark = create_benchmark_revision_from_turns(
        database,
        session_id="session-one",
        session_name="Arithmetic",
        messages=_messages(),
        message_ids=["user-1", "assistant-1"],
        review_note="Reviewed expected answer",
        artifact_id="artifact-one",
    )
    draft = create_dataset_source_draft_from_turns(
        tmp_path / "datasets",
        session_id="session-one",
        session_name="Arithmetic",
        messages=_messages(),
        message_ids=["user-1", "assistant-1"],
        review_note="Reviewed training example",
        artifact_id="artifact-one",
    )

    assert benchmark["revision"]["items"][0]["expected"] == "Four."
    assert benchmark["starts_training"] is False
    record = json.loads(Path(draft["records_path"]).read_text(encoding="utf-8"))
    assert record["messages"][0]["role"] == "user"
    assert record["messages"][1]["content"] == "Four."
    assert draft["status"] == "draft"
    assert draft["starts_training"] is False
    database.close()
