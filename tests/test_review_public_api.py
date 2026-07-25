"""Public API contracts for Human Feedback and Active Data Studio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from halo_forge.public_api.service import PublicApiService
from halo_forge.run_db import RunDatabase


class _Row(dict):
    def to_dict(self) -> dict[str, Any]:
        return dict(self)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - test helper diagnostics
            raise AttributeError(name) from exc


class _ReviewFacade:
    def __init__(self) -> None:
        self.schemas: dict[str, _Row] = {}
        self.revisions: dict[str, _Row] = {}
        self.batches: dict[str, _Row] = {}
        self.candidates: dict[str, list[_Row]] = {}
        self.queues: dict[str, _Row] = {}
        self.items: dict[str, _Row] = {}
        self.events: list[_Row] = []
        self.suggestions: dict[str, _Row] = {}
        self.label_sets: dict[str, _Row] = {}
        self.label_revisions: dict[str, _Row] = {}

    def capabilities(self) -> dict[str, Any]:
        return {
            "modalities": ["text", "vlm", "audio"],
            "strategies": ["explicit", "regression"],
            "output_adapters": [{"id": "filter", "build_modes": ["filter"]}],
        }

    def create_schema(self, **payload: Any):
        schema_id = payload.get("schema_id") or f"schema-{len(self.schemas) + 1}"
        schema = _Row(
            id=schema_id,
            name=payload["name"],
            description=payload.get("description"),
            archived=False,
        )
        revision = _Row(
            id=f"{schema_id}-r1",
            schema_id=schema_id,
            revision_number=1,
            modality=payload["modality"],
            task_type=payload["task_type"],
            definition=payload["definition"],
        )
        self.schemas[schema_id] = schema
        self.revisions[revision["id"]] = revision
        return schema, revision

    def list_schemas(self):
        return list(self.schemas.values())

    def get_schema(self, schema_id: str):
        return self.schemas.get(schema_id)

    def list_schema_revisions(self, schema_id: str):
        return [row for row in self.revisions.values() if row["schema_id"] == schema_id]

    def get_schema_revision(self, revision_id: str):
        return self.revisions.get(revision_id)

    def revise_schema(self, schema_id: str, **payload: Any):
        prior = self.list_schema_revisions(schema_id)[-1]
        revision = _Row(
            id=f"{schema_id}-r{len(self.list_schema_revisions(schema_id)) + 1}",
            schema_id=schema_id,
            revision_number=len(self.list_schema_revisions(schema_id)) + 1,
            modality=payload.get("modality", prior["modality"]),
            task_type=payload.get("task_type", prior["task_type"]),
            definition=payload["definition"],
        )
        self.revisions[revision["id"]] = revision
        return revision

    def create_acquisition(self, records, **payload: Any):
        records = list(records)
        batch_id = f"batch-{len(self.batches) + 1}"
        batch = _Row(
            id=batch_id,
            name=payload.get("name") or "Acquisition",
            status="completed",
            seed=payload.get("seed", 0),
            row_count=len(records),
        )
        self.batches[batch_id] = batch
        self.candidates[batch_id] = [
            _Row(
                id=f"{batch_id}-candidate-{index}",
                batch_id=batch_id,
                ordinal=index,
                record_id=value.get("record_id") or f"record-{index}",
                record=value.get("record", value),
            )
            for index, value in enumerate(records)
        ]
        return batch

    def list_acquisition_batches(self):
        return list(self.batches.values())

    def get_acquisition_batch(self, batch_id: str):
        return self.batches.get(batch_id)

    def list_acquisition_candidates(self, batch_id: str):
        return self.candidates.get(batch_id, [])

    def create_queue(self, batch_id: str, schema_revision_id: str, **payload: Any):
        queue_id = f"queue-{len(self.queues) + 1}"
        queue = _Row(
            id=queue_id,
            name=payload.get("name") or "Review queue",
            status="active",
            acquisition_batch_id=batch_id,
            schema_revision_id=schema_revision_id,
            policy=payload.get("policy") or {},
            current_pass=1,
        )
        self.queues[queue_id] = queue
        for candidate in self.candidates[batch_id]:
            item = _Row(
                id=f"{queue_id}-item-{candidate['ordinal']}",
                queue_id=queue_id,
                candidate_id=candidate["id"],
                ordinal=candidate["ordinal"],
                status="pending",
                active_event_id=None,
                projection={"current_pass": 1},
                record=candidate["record"],
            )
            self.items[item["id"]] = item
        return queue

    def list_queues(self):
        return list(self.queues.values())

    def get_queue(self, queue_id: str):
        return self.queues.get(queue_id)

    def list_items(self, queue_id: str):
        return [row for row in self.items.values() if row["queue_id"] == queue_id]

    def get_item(self, item_id: str):
        return self.items.get(item_id)

    def submit_event(self, item_id: str, **payload: Any):
        from halo_forge.review_lab.errors import ReviewConflictError

        if payload.get("expected_active_event_id") == "stale":
            raise ReviewConflictError("active event changed")
        event = _Row(
            id=f"event-{len(self.events) + 1}",
            item_id=item_id,
            queue_id=self.items[item_id]["queue_id"],
            **payload,
        )
        self.events.append(event)
        self.items[item_id]["active_event_id"] = event["id"]
        self.items[item_id]["status"] = "resolved"
        return event

    def adjudicate(self, item_id: str, **payload: Any):
        return self.submit_event(item_id, event_type="adjudication", **payload)

    def create_suggestion(self, item_id: str, **payload: Any):
        suggestion_id = payload.get("suggestion_id") or f"suggestion-{len(self.suggestions) + 1}"
        suggestion = _Row(id=suggestion_id, item_id=item_id, **payload)
        self.suggestions[suggestion_id] = suggestion
        return suggestion

    def statistics(self, queue_id: str):
        values = self.list_items(queue_id)
        resolved = sum(row["status"] == "resolved" for row in values)
        return {"queue_id": queue_id, "total": len(values), "resolved": resolved}

    def publish_label_set(
        self,
        queue_id: str,
        *,
        name: str | None = None,
        output_adapter_id: str | None = None,
        build_mode: str | None = None,
    ):
        label_set_id = f"labels-{len(self.label_sets) + 1}"
        revision_id = f"{label_set_id}-r1"
        label_set = _Row(id=label_set_id, queue_id=queue_id, name=name or "Labels")
        revision = _Row(
            id=revision_id,
            label_set_id=label_set_id,
            revision_number=1,
            content_hash="a" * 64,
            row_count=len(self.list_items(queue_id)),
        )
        self.label_sets[label_set_id] = label_set
        self.label_revisions[revision_id] = revision
        return revision

    def list_label_sets(self):
        return list(self.label_sets.values())

    def get_label_set_revision(self, revision_id: str):
        return self.label_revisions.get(revision_id)

    def verify_label_set(self, revision_id: str):
        return _Row(revision_id=revision_id, valid=revision_id in self.label_revisions, errors=[])

    def render_label_set(self, revision_id: str, **_payload: Any):
        if revision_id not in self.label_revisions:
            raise KeyError(revision_id)
        return {"records": [{"prompt": "hello", "response": "world"}]}


def test_review_routes_cover_schema_acquisition_review_and_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from halo_forge.auth.dependency import reset_store_for_tests
    from halo_forge.public_api import app as app_module

    database = RunDatabase(str(tmp_path / "runs.db"))
    facade = _ReviewFacade()
    service = PublicApiService(
        database=database,
        review_lab=facade,
        review_storage_root=tmp_path / "reviews",
    )
    monkeypatch.setattr(app_module, "PublicApiService", lambda: service)
    monkeypatch.setenv("HALOFORGE_DISABLE_AUTO_WORKER", "1")
    reset_store_for_tests(None)
    try:
        with TestClient(app_module.create_app(serve_frontend=False)) as client:
            capabilities = client.get("/api/public/review-capabilities")
            assert capabilities.status_code == 200
            assert "vlm" in capabilities.json()["modalities"]

            descriptors = client.get("/api/public/spec-descriptors/dataset_recipe_step")
            assert descriptors.status_code == 200
            assert descriptors.json()["total"] >= 1
            invalid_step = client.post(
                "/api/public/spec-descriptors/dataset_recipe_step/map/validate",
                json={"schema": "sft"},
            ).json()
            assert invalid_step["valid"] is False

            created = client.post(
                "/api/public/annotation-schemas",
                json={
                    "name": "Accept or reject",
                    "modality": "text",
                    "task_type": "binary",
                    "definition": {"output_adapter_id": "filter"},
                },
            )
            assert created.status_code == 201, created.text
            revision_id = created.json()["revision"]["id"]
            assert client.get("/api/public/annotation-schemas").json()["total"] == 1

            session = client.post(
                "/api/public/playground/sessions", json={"name": "Review source"}
            ).json()
            session = client.post(
                f"/api/public/playground/sessions/{session['id']}/messages",
                json={"role": "user", "content": "Question"},
            ).json()
            session = client.post(
                f"/api/public/playground/sessions/{session['id']}/messages",
                json={"role": "assistant", "content": "Candidate answer"},
            ).json()
            playground_queue = client.post(
                f"/api/public/playground/sessions/{session['id']}/review",
                json={
                    "kind": "review_queue",
                    "schema_revision_id": revision_id,
                    "message_ids": [value["id"] for value in session["messages"]],
                    "review_note": "Review this turn before reuse",
                },
            )
            assert playground_queue.status_code == 201, playground_queue.text
            assert playground_queue.json()["kind"] == "review_queue"
            assert playground_queue.json()["starts_training"] is False

            acquired = client.post(
                "/api/public/acquisition-batches",
                json={
                    "name": "Regression proposal",
                    "seed": 42,
                    "records": [
                        {"record_id": "one", "record": {"prompt": "p1", "response": "r1"}},
                        {"record_id": "two", "record": {"prompt": "p2", "response": "r2"}},
                    ],
                    "strategies": [{"kind": "explicit"}],
                },
            )
            assert acquired.status_code == 202, acquired.text
            batch_id = acquired.json()["id"]
            candidate_page = client.get(
                f"/api/public/acquisition-batches/{batch_id}/candidates",
                params={"limit": 1, "offset": 1},
            ).json()
            assert candidate_page["total"] == 2
            assert candidate_page["items"][0]["record_id"] == "two"

            queued = client.post(
                "/api/public/review-queues",
                json={
                    "batch_id": batch_id,
                    "schema_revision_id": revision_id,
                    "policy": {"mode": "one_pass"},
                },
            )
            assert queued.status_code == 201, queued.text
            queue_id = queued.json()["id"]
            items = client.get(f"/api/public/review-queues/{queue_id}/items").json()
            item_id = items["items"][0]["id"]

            verifier = service._verifier_engine().create_profile(
                name="Review suggestion JSON check",
                description=None,
                definition={
                    "family": "deterministic",
                    "implementation": {"kind": "builtin", "ref": "json_structure"},
                    "modality": "text",
                    "task_type": "binary",
                    "input_mapping": {},
                    "reward_contract": {
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "threshold": 0.5,
                        "tie_policy": "fail",
                        "error_behavior": "fail_closed",
                    },
                },
            )
            suggestion = client.post(
                f"/api/public/review-items/{item_id}/suggestions",
                json={
                    "provider": "injected",
                    "model": "fixture-model@revision-1",
                    "output": "{\"accepted\": true}",
                    "verifier_profile_revision_id": verifier["revision"]["id"],
                },
            )
            assert suggestion.status_code == 202, suggestion.text
            suggestion_payload = suggestion.json()
            assert suggestion_payload["verifier_binding"]["domain_kind"] == "review_suggestion"
            assert service._verifier_engine().store.list_bindings(
                domain_kind="review_suggestion",
                domain_id=suggestion_payload["id"],
            )

            submitted = client.post(
                f"/api/public/review-items/{item_id}/events",
                json={
                    "event_type": "label",
                    "payload": {"accepted": True},
                    "idempotency_key": "api-decision-1",
                },
            )
            assert submitted.status_code == 201, submitted.text
            assert (
                client.get(f"/api/public/review-queues/{queue_id}/statistics").json()["resolved"]
                == 1
            )
            conflict = client.post(
                f"/api/public/review-items/{item_id}/events",
                json={
                    "event_type": "label",
                    "payload": {"accepted": False},
                    "idempotency_key": "api-decision-2",
                    "expected_active_event_id": "stale",
                },
            )
            assert conflict.status_code == 409

            published = client.post(
                f"/api/public/review-queues/{queue_id}/label-set-revisions",
                json={"name": "Reviewed regressions"},
            )
            assert published.status_code == 202, published.text
            label_revision_id = published.json()["id"]
            assert (
                client.post(f"/api/public/label-set-revisions/{label_revision_id}/verify").json()[
                    "valid"
                ]
                is True
            )
            preview = client.post(
                f"/api/public/label-set-revisions/{label_revision_id}/dataset-preview",
                json={"build_mode": "filter"},
            ).json()
            assert preview["total"] == 1
            assert preview["starts_training"] is False
    finally:
        database.close()


def test_review_routes_integrate_with_real_review_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from halo_forge.auth.dependency import reset_store_for_tests
    from halo_forge.public_api import app as app_module

    database = RunDatabase(str(tmp_path / "real-runs.db"))
    service = PublicApiService(
        database=database,
        review_storage_root=tmp_path / "real-reviews",
        dataset_storage_root=tmp_path / "real-datasets",
    )
    monkeypatch.setattr(app_module, "PublicApiService", lambda: service)
    monkeypatch.setenv("HALOFORGE_DISABLE_AUTO_WORKER", "1")
    reset_store_for_tests(None)
    try:
        with TestClient(app_module.create_app(serve_frontend=False)) as client:
            schema = client.post(
                "/api/public/annotation-schemas",
                json={
                    "name": "Real API binary review",
                    "modality": "text",
                    "task_type": "binary",
                    "definition": {},
                },
            )
            assert schema.status_code == 201, schema.text
            acquisition = client.post(
                "/api/public/acquisition-batches",
                json={
                    "records": [
                        {
                            "record_id": "real-api-record",
                            "record": {"prompt": "Question", "response": "Candidate"},
                        }
                    ]
                },
            )
            assert acquisition.status_code == 202, acquisition.text
            queue = client.post(
                "/api/public/review-queues",
                json={
                    "batch_id": acquisition.json()["id"],
                    "schema_revision_id": schema.json()["revision"]["id"],
                },
            )
            assert queue.status_code == 201, queue.text
            item = client.get(f"/api/public/review-queues/{queue.json()['id']}/items").json()[
                "items"
            ][0]
            event = client.post(
                f"/api/public/review-items/{item['id']}/events",
                json={
                    "event_type": "label",
                    "payload": {"accepted": True},
                    "idempotency_key": "real-api-label",
                    "expected_active_event_id": None,
                    "reviewer_key": "local",
                },
            )
            assert event.status_code == 201, event.text
            published = client.post(
                f"/api/public/review-queues/{queue.json()['id']}/label-set-revisions",
                json={},
            )
            assert published.status_code == 202, published.text
            assert published.json()["status"] == "queued"
            from halo_forge.workstation_jobs.worker import WorkstationWorker

            finished = WorkstationWorker(service._scheduler()).run_once()
            assert finished is not None and finished.status == "completed"
            refreshed_queue = client.get(f"/api/public/review-queues/{queue.json()['id']}").json()
            revision_id = refreshed_queue["latest_label_set_revision_id"]
            preview = client.post(
                f"/api/public/label-set-revisions/{revision_id}/dataset-preview",
                json={},
            )
            assert preview.status_code == 200, preview.text
            assert preview.json()["total"] == 1
            assert preview.json()["starts_training"] is False
    finally:
        database.close()
