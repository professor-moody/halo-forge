"""Reviewed Playground turn promotion into evaluation and data drafts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _selected_turns(
    messages: Sequence[Mapping[str, Any]], message_ids: Sequence[str]
) -> list[dict[str, Any]]:
    selected = {str(value) for value in message_ids}
    ordered = [dict(value) for value in messages if str(value.get("id") or "") in selected]
    if not ordered:
        raise ValueError("Select at least one persisted Playground message")
    pairs: list[dict[str, Any]] = []
    pending_user: dict[str, Any] | None = None
    for message in ordered:
        role = str(message.get("role") or "").lower()
        if role == "user":
            pending_user = message
            continue
        if role != "assistant" or pending_user is None:
            continue
        pairs.append(
            {
                "user": pending_user,
                "assistant": message,
            }
        )
        pending_user = None
    if not pairs:
        raise ValueError("Select at least one complete user and assistant turn")
    return pairs


def _review_metadata(
    *,
    session_id: str,
    session_name: str,
    review_note: str,
    artifact_id: str | None,
) -> dict[str, Any]:
    return {
        "source": "playground_review",
        "session_id": session_id,
        "session_name": session_name,
        "review_note": review_note,
        "artifact_id": artifact_id,
    }


def create_benchmark_revision_from_turns(
    database: Any,
    *,
    session_id: str,
    session_name: str,
    messages: Sequence[Mapping[str, Any]],
    message_ids: Sequence[str],
    review_note: str,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    note = str(review_note or "").strip()
    if not note:
        raise ValueError("A review_note is required")
    pairs = _selected_turns(messages, message_ids)
    provenance = _review_metadata(
        session_id=session_id,
        session_name=session_name,
        review_note=note,
        artifact_id=artifact_id,
    )
    items = []
    for index, pair in enumerate(pairs):
        user = pair["user"]
        assistant = pair["assistant"]
        identity = _digest(
            {
                "session_id": session_id,
                "user_id": user.get("id"),
                "assistant_id": assistant.get("id"),
            }
        )
        items.append(
            {
                "id": f"playground-{identity[:20]}",
                "record_id": identity,
                "input": user.get("content"),
                "expected": assistant.get("content"),
                "metadata": {
                    **provenance,
                    "ordinal": index,
                    "generation": assistant.get("generation") or {},
                    "evidence": assistant.get("evidence") or {},
                },
            }
        )
    suite = database.create_benchmark_suite(
        name=f"Playground · {session_name}",
        description=f"Reviewed Playground turns. {note}",
        purpose="development",
    )
    payload = {
        "items": items,
        "provenance": provenance,
        "primary_metric": "score",
        "direction": "maximize",
    }
    revision = database.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash=_digest(payload),
        items=items,
        primary_metric="score",
        direction="maximize",
        generation_settings={"source": "playground_review"},
        evaluator_versions={},
    )
    return {
        "kind": "benchmark_suite_revision",
        "status": "created",
        "suite": suite.to_dict(),
        "revision": revision.to_dict(),
        "reviewed_turn_count": len(items),
        "starts_training": False,
    }


def create_dataset_source_draft_from_turns(
    dataset_root: Path | str,
    *,
    session_id: str,
    session_name: str,
    messages: Sequence[Mapping[str, Any]],
    message_ids: Sequence[str],
    review_note: str,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    note = str(review_note or "").strip()
    if not note:
        raise ValueError("A review_note is required")
    pairs = _selected_turns(messages, message_ids)
    provenance = _review_metadata(
        session_id=session_id,
        session_name=session_name,
        review_note=note,
        artifact_id=artifact_id,
    )
    records = [
        {
            "messages": [
                {"role": "user", "content": pair["user"].get("content")},
                {"role": "assistant", "content": pair["assistant"].get("content")},
            ],
            "metadata": {
                **provenance,
                "source_message_ids": [
                    pair["user"].get("id"),
                    pair["assistant"].get("id"),
                ],
                "generation": pair["assistant"].get("generation") or {},
                "evidence": pair["assistant"].get("evidence") or {},
            },
        }
        for pair in pairs
    ]
    draft_identity = _digest({"records": records, "provenance": provenance})
    draft_id = f"playground-draft-{draft_identity[:24]}"
    root = Path(dataset_root).expanduser().resolve()
    drafts_root = root / "source-drafts"
    drafts_root.mkdir(parents=True, exist_ok=True)
    destination = drafts_root / draft_id
    if not destination.exists():
        staging = drafts_root / f".{draft_id}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True)
        try:
            records_path = staging / "records.jsonl"
            with records_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(_canonical(record) + "\n")
            manifest = {
                "id": draft_id,
                "kind": "dataset_source_draft",
                "canonical_schema": "chat",
                "record_count": len(records),
                "content_hash": _digest(records),
                "provenance": provenance,
                "records_path": "records.jsonl",
                "starts_training": False,
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    return {
        **manifest,
        "status": "draft",
        "path": str(destination),
        "records_path": str(destination / "records.jsonl"),
    }


def create_review_acquisition_records_from_turns(
    *,
    session_id: str,
    session_name: str,
    messages: Sequence[Mapping[str, Any]],
    message_ids: Sequence[str],
    review_note: str,
    artifact_id: str | None = None,
    pairings: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize selected turns for Review Studio without creating label events."""

    note = str(review_note or "").strip()
    if not note:
        raise ValueError("A review_note is required")
    provenance = _review_metadata(
        session_id=session_id,
        session_name=session_name,
        review_note=note,
        artifact_id=artifact_id,
    )
    if pairings is not None:
        return _create_pairwise_review_records(
            session_id=session_id,
            messages=messages,
            pairings=pairings,
            provenance=provenance,
        )

    pairs = _selected_turns(messages, message_ids)
    records: list[dict[str, Any]] = []
    for ordinal, pair in enumerate(pairs):
        user = pair["user"]
        assistant = pair["assistant"]
        identity = _digest(
            {
                "session_id": session_id,
                "user_id": user.get("id"),
                "assistant_id": assistant.get("id"),
            }
        )
        records.append(
            {
                "record_id": identity,
                "record": {
                    "messages": [
                        {"role": "user", "content": user.get("content")},
                        {"role": "assistant", "content": assistant.get("content")},
                    ],
                    "metadata": {
                        **provenance,
                        "source_message_ids": [user.get("id"), assistant.get("id")],
                    },
                },
                "evidence": {
                    "generation": assistant.get("generation") or {},
                    "evidence": assistant.get("evidence") or {},
                },
                "source": {
                    "kind": "playground_session",
                    "ref": session_id,
                    "ordinal": ordinal,
                    "purpose": "development",
                },
            }
        )
    return {"records": records, "provenance": provenance}


def _create_pairwise_review_records(
    *,
    session_id: str,
    messages: Sequence[Mapping[str, Any]],
    pairings: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build explicit Playground base/candidate records for preference review.

    Pairings refer to three persisted messages.  This intentionally never
    substitutes placeholder choices: the actual base and candidate response
    content becomes the ordered alternatives presented to the reviewer.
    """

    if isinstance(pairings, (str, bytes)) or not isinstance(pairings, Sequence):
        raise ValueError("pairings must be a list of persisted message references")
    if not pairings:
        raise ValueError("pairings must contain at least one base/candidate comparison")

    by_id = {
        str(message.get("id")): dict(message)
        for message in messages
        if str(message.get("id") or "").strip()
    }
    records: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(pairings):
        if not isinstance(raw, Mapping):
            raise ValueError("each pairing must be an object")
        prompt_id = str(raw.get("prompt_message_id") or "").strip()
        base_id = str(raw.get("base_message_id") or "").strip()
        candidate_id = str(raw.get("candidate_message_id") or "").strip()
        if not prompt_id or not base_id or not candidate_id:
            raise ValueError(
                "each pairing requires prompt_message_id, base_message_id, "
                "and candidate_message_id"
            )
        if base_id == candidate_id:
            raise ValueError("base_message_id and candidate_message_id must differ")
        try:
            prompt = by_id[prompt_id]
            base = by_id[base_id]
            candidate = by_id[candidate_id]
        except KeyError as exc:
            raise ValueError(f"unknown persisted Playground message: {exc.args[0]}") from exc
        if str(prompt.get("role") or "").lower() != "user":
            raise ValueError("prompt_message_id must identify a user message")
        if str(base.get("role") or "").lower() != "assistant":
            raise ValueError("base_message_id must identify an assistant message")
        if str(candidate.get("role") or "").lower() != "assistant":
            raise ValueError("candidate_message_id must identify an assistant message")

        prompt_text = str(prompt.get("content") or "")
        base_text = str(base.get("content") or "")
        candidate_text = str(candidate.get("content") or "")
        if not prompt_text or not base_text or not candidate_text:
            raise ValueError("paired Playground messages must have non-empty content")
        if base_text == candidate_text:
            raise ValueError("base and candidate responses must differ for preference review")

        identity = _digest(
            {
                "session_id": session_id,
                "prompt_message_id": prompt_id,
                "base_message_id": base_id,
                "candidate_message_id": candidate_id,
            }
        )
        comparison = {
            "representation": "playground_base_candidate.v1",
            "prompt_message_id": prompt_id,
            "base_message_id": base_id,
            "candidate_message_id": candidate_id,
            "base_artifact_id": base.get("artifact_id"),
            "candidate_artifact_id": candidate.get("artifact_id"),
        }
        records.append(
            {
                "record_id": identity,
                "record": {
                    "prompt": prompt_text,
                    "alternatives": [base_text, candidate_text],
                    "metadata": {**dict(provenance), "comparison": comparison},
                },
                "evidence": {
                    "base": {
                        "message_id": base_id,
                        "artifact_id": base.get("artifact_id"),
                        "generation": base.get("generation") or {},
                        "evidence": base.get("evidence") or {},
                    },
                    "candidate": {
                        "message_id": candidate_id,
                        "artifact_id": candidate.get("artifact_id"),
                        "generation": candidate.get("generation") or {},
                        "evidence": candidate.get("evidence") or {},
                    },
                },
                "source": {
                    "kind": "playground_session",
                    "ref": session_id,
                    "ordinal": ordinal,
                    "purpose": "development",
                    "representation": "base_candidate_pair",
                },
            }
        )
    return {
        "records": records,
        "provenance": {
            **dict(provenance),
            "representation": "playground_base_candidate.v1",
            "pairing_count": len(records),
        },
    }


__all__ = [
    "create_benchmark_revision_from_turns",
    "create_dataset_source_draft_from_turns",
    "create_review_acquisition_records_from_turns",
]
