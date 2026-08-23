"""CLI parity for Review Studio's guided operational flow."""

from __future__ import annotations

import json
import sys
from typing import Any

from halo_forge import cli


class _CliReviewService:
    def __init__(self) -> None:
        self.schemas: dict[str, dict[str, Any]] = {}
        self.batches: dict[str, dict[str, Any]] = {}
        self.queues: dict[str, dict[str, Any]] = {}
        self.items: dict[str, dict[str, Any]] = {}
        self.label_revision: dict[str, Any] | None = None

    def get_review_capabilities(self):
        return {"modalities": ["text", "vlm", "audio"]}

    def create_annotation_schema(self, payload):
        schema = {"id": "schema-cli", "name": payload["name"]}
        revision = {
            "id": "schema-cli-r1",
            "schema_id": schema["id"],
            "modality": payload["modality"],
            "task_type": payload["task_type"],
            "definition": payload["definition"],
        }
        self.schemas[schema["id"]] = {**schema, "revisions": [revision]}
        return {"schema": schema, "revision": revision}

    def list_annotation_schemas(self, **_kwargs):
        return {"items": list(self.schemas.values()), "total": len(self.schemas)}

    def get_annotation_schema(self, identifier):
        return self.schemas.get(identifier)

    def get_annotation_schema_revision(self, identifier):
        for schema in self.schemas.values():
            for revision in schema["revisions"]:
                if revision["id"] == identifier:
                    return revision
        return None

    def create_acquisition_batch(self, payload):
        batch = {
            "id": "batch-cli",
            "name": payload.get("name"),
            "seed": payload.get("seed"),
            "row_count": len(payload.get("records") or []),
            "status": "completed",
        }
        self.batches[batch["id"]] = batch
        return batch

    def get_acquisition_batch(self, identifier):
        return self.batches.get(identifier)

    def create_review_queue(self, payload):
        queue = {
            "id": "queue-cli",
            "acquisition_batch_id": payload["batch_id"],
            "schema_revision_id": payload["schema_revision_id"],
            "policy": payload["policy"],
            "status": "active",
        }
        self.queues[queue["id"]] = queue
        item = {
            "id": "item-cli",
            "queue_id": queue["id"],
            "status": "pending",
            "active_event_id": None,
        }
        self.items[item["id"]] = item
        return queue

    def list_review_queues(self, **_kwargs):
        return {"items": list(self.queues.values()), "total": len(self.queues)}

    def get_review_queue(self, identifier):
        return self.queues.get(identifier)

    def list_review_items(self, queue_id, **_kwargs):
        values = [value for value in self.items.values() if value["queue_id"] == queue_id]
        return {"items": values, "total": len(values)}

    def get_review_item(self, item_id):
        return self.items.get(item_id)

    def submit_review_event(self, item_id, payload):
        event = {
            "id": "event-cli",
            "item_id": item_id,
            "event_type": payload["event_type"],
            "payload": payload["payload"],
            "idempotency_key": payload["idempotency_key"],
        }
        self.items[item_id].update(status="resolved", active_event_id=event["id"])
        return event

    def get_review_queue_statistics(self, queue_id):
        items = [value for value in self.items.values() if value["queue_id"] == queue_id]
        return {
            "queue_id": queue_id,
            "total": len(items),
            "resolved": sum(value["status"] == "resolved" for value in items),
        }

    def publish_label_set(self, queue_id, payload):
        self.label_revision = {
            "id": "labels-cli-r1",
            "label_set_id": "labels-cli",
            "queue_id": queue_id,
            "name": payload.get("name"),
            "content_hash": "b" * 64,
        }
        return self.label_revision

    def preview_label_set_dataset(self, revision_id, payload):
        assert self.label_revision and revision_id == self.label_revision["id"]
        return {
            "revision_id": revision_id,
            "build_mode": payload.get("build_mode"),
            "items": [{"prompt": "question", "response": "answer"}],
            "total": 1,
            "starts_training": False,
        }


def _run_cli(monkeypatch, capsys, service: _CliReviewService, *args: str):
    monkeypatch.setattr(cli.sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(cli, "_review_public_service", lambda _args: service)
    monkeypatch.setattr(sys, "argv", ["halo-forge", *args])
    cli.main()
    return json.loads(capsys.readouterr().out)


def test_review_cli_runs_schema_to_label_set_preview(monkeypatch, capsys) -> None:
    service = _CliReviewService()
    schema = _run_cli(
        monkeypatch,
        capsys,
        service,
        "review",
        "schema",
        "create",
        "--json",
        "--name",
        "CLI accept/reject",
        "--modality",
        "text",
        "--task-type",
        "binary",
        "--definition",
        '{"output_adapter_id":"filter"}',
    )
    revision_id = schema["revision"]["id"]

    acquired = _run_cli(
        monkeypatch,
        capsys,
        service,
        "review",
        "acquire",
        "create",
        "--json",
        "--name",
        "CLI proposal",
        "--seed",
        "42",
        "--records",
        '[{"record_id":"cli-record","record":{"prompt":"question","response":"answer"}}]',
        "--strategy",
        '{"kind":"explicit"}',
    )
    assert acquired["row_count"] == 1

    queue = _run_cli(
        monkeypatch,
        capsys,
        service,
        "review",
        "queue",
        "create",
        "--json",
        "--batch",
        acquired["id"],
        "--schema",
        revision_id,
        "--policy",
        '{"mode":"one_pass"}',
    )
    items = _run_cli(
        monkeypatch,
        capsys,
        service,
        "review",
        "items",
        queue["id"],
        "--json",
    )
    item_id = items["items"][0]["id"]
    submitted = _run_cli(
        monkeypatch,
        capsys,
        service,
        "review",
        "submit",
        item_id,
        "--json",
        "--idempotency-key",
        "cli-review-1",
        "--label",
        '{"accepted":true}',
    )
    assert submitted["payload"] == {"accepted": True}
    assert (
        _run_cli(
            monkeypatch,
            capsys,
            service,
            "review",
            "stats",
            queue["id"],
            "--json",
        )["resolved"]
        == 1
    )

    published = _run_cli(
        monkeypatch,
        capsys,
        service,
        "review",
        "label-set",
        "publish",
        "--json",
        "--queue",
        queue["id"],
        "--name",
        "CLI labels",
    )
    preview = _run_cli(
        monkeypatch,
        capsys,
        service,
        "review",
        "label-set",
        "preview",
        published["id"],
        "--json",
        "--mode",
        "filter",
    )
    assert preview["total"] == 1
    assert preview["starts_training"] is False


def test_review_cli_integrates_with_real_review_service(tmp_path, monkeypatch, capsys) -> None:
    database = str(tmp_path / "review-cli.db")
    root = str(tmp_path / "reviews")

    def run(*arguments: str):
        monkeypatch.setattr(cli.sys, "version_info", (3, 13, 0))
        monkeypatch.setattr(sys, "argv", ["halo-forge", *arguments])
        cli.main()
        return json.loads(capsys.readouterr().out)

    schema = run(
        "review",
        "schema",
        "create",
        "--database",
        database,
        "--root",
        root,
        "--json",
        "--name",
        "Real CLI review",
        "--modality",
        "text",
        "--task-type",
        "binary",
        "--definition",
        "{}",
    )
    acquisition = run(
        "review",
        "acquire",
        "create",
        "--database",
        database,
        "--root",
        root,
        "--json",
        "--records",
        '[{"record_id":"real-cli-record","record":{"prompt":"p","response":"r"}}]',
    )
    queue = run(
        "review",
        "queue",
        "create",
        "--database",
        database,
        "--root",
        root,
        "--json",
        "--batch",
        acquisition["id"],
        "--schema",
        schema["revision"]["id"],
    )
    item = run(
        "review",
        "items",
        queue["id"],
        "--database",
        database,
        "--root",
        root,
        "--json",
    )[
        "items"
    ][0]
    event = run(
        "review",
        "submit",
        item["id"],
        "--database",
        database,
        "--root",
        root,
        "--json",
        "--idempotency-key",
        "real-cli-label",
        "--label",
        '{"accepted":true}',
    )
    assert event["reviewer_key"] == "local"
    published = run(
        "review",
        "label-set",
        "publish",
        "--database",
        database,
        "--root",
        root,
        "--json",
        "--queue",
        queue["id"],
    )
    assert published["status"] == "queued"
    worker = run(
        "jobs",
        "worker",
        "--database",
        database,
        "--json",
        "--once",
    )
    assert worker["completed_work_item"]["status"] == "completed"
    refreshed_queue = run(
        "review",
        "queue",
        "show",
        queue["id"],
        "--database",
        database,
        "--root",
        root,
        "--json",
    )
    revision_id = refreshed_queue["latest_label_set_revision_id"]
    preview = run(
        "review",
        "label-set",
        "preview",
        revision_id,
        "--database",
        database,
        "--root",
        root,
        "--json",
    )
    assert preview["total"] == 1
    assert preview["starts_training"] is False
