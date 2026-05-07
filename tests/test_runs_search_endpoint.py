"""/runs/search endpoint tests (Track F-G commit 2).

Exercises the public API's DB-backed search surface end-to-end via
FastAPI's TestClient + an in-memory `RunDatabase`. We pre-seed the
DB and disable the lazy filesystem sync so the test stays focused on
the HTTP→service→DB plumbing rather than the walker.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def search_client(monkeypatch, tmp_path):
    """FastAPI TestClient with the run DB pointed at an isolated path
    and the filesystem sync stubbed to a no-op.
    """
    db_path = tmp_path / "runs.db"
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(db_path))

    # Make sure nothing else has cached a DB instance for the previous
    # path — `get_database` is process-cached.
    from halo_forge.run_db import db as db_mod

    db_mod._GLOBAL_DB.clear()

    # Stub sync so we don't hit the real filesystem during the test.
    import halo_forge.run_db.sync as sync_mod

    monkeypatch.setattr(sync_mod, "sync_from_filesystem", lambda *a, **k: 0)
    # Also patch the import inside the service.
    import halo_forge.public_api.service as service_mod
    monkeypatch.setattr(
        "halo_forge.run_db.sync_from_filesystem", lambda *a, **k: 0, raising=False
    )

    # Build the FastAPI app — pulls in the public service which uses
    # our patched DB path.
    from halo_forge.public_api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    # Seed records directly through the DB so we don't depend on real
    # training_summary.json files.
    from halo_forge.run_db import RunRecord, get_database

    db = get_database()
    for record in _seed_records():
        db.upsert_run(record)

    yield client, db

    db_mod._GLOBAL_DB.clear()


def _seed_records():
    from halo_forge.run_db import RunRecord

    return [
        RunRecord(
            run_id="r_sft_qwen",
            modality="sft",
            model_name="Qwen/Qwen2.5-3B",
            status="completed",
            timestamp="2026-04-15T10:00:00+00:00",
            output_dir="/tmp/r_sft_qwen",
            cycles_executed=2,
            final_train_loss=0.45,
            weights_updated=True,
            effectiveness_verdict="passed",
            raw_json="{}",
        ),
        RunRecord(
            run_id="r_dpo_llama",
            modality="dpo",
            model_name="meta-llama/Llama-3.2-3B",
            status="completed",
            timestamp="2026-04-20T10:00:00+00:00",
            output_dir="/tmp/r_dpo_llama",
            cycles_executed=1,
            final_train_loss=0.33,
            weights_updated=True,
            raw_json="{}",
        ),
        RunRecord(
            run_id="r_raft_qwen",
            modality="raft",
            model_name="Qwen/Qwen2.5-7B",
            status="failed",
            timestamp="2026-04-10T10:00:00+00:00",
            output_dir="/tmp/r_raft_qwen",
            cycles_executed=0,
            final_train_loss=None,
            weights_updated=False,
            failure_reason="oom",
            raw_json="{}",
        ),
    ]


def test_search_default_returns_all(search_client):
    client, _ = search_client
    r = client.get("/api/public/runs/search")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    ids = [item["run_id"] for item in body["items"]]
    # Default sort is timestamp desc.
    assert ids == ["r_dpo_llama", "r_sft_qwen", "r_raft_qwen"]
    # Facets surface distinct modalities + models.
    assert set(body["facets"]["modalities"]) == {"sft", "dpo", "raft"}
    assert "Qwen/Qwen2.5-3B" in body["facets"]["models"]


def test_search_filter_by_modality_in_list(search_client):
    client, _ = search_client
    r = client.get("/api/public/runs/search?modality=sft&modality=dpo")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert {i["run_id"] for i in body["items"]} == {"r_sft_qwen", "r_dpo_llama"}


def test_search_filter_by_status_and_model_substring(search_client):
    client, _ = search_client
    r = client.get(
        "/api/public/runs/search?status=completed&model=Qwen"
    )
    assert r.status_code == 200
    body = r.json()
    # Only r_sft_qwen matches: completed AND Qwen.
    assert {i["run_id"] for i in body["items"]} == {"r_sft_qwen"}


def test_search_has_eval_filter(search_client):
    client, _ = search_client
    r = client.get("/api/public/runs/search?has_eval=false")
    assert r.status_code == 200
    body = r.json()
    assert {i["run_id"] for i in body["items"]} == {"r_raft_qwen"}


def test_search_pagination(search_client):
    client, _ = search_client
    r = client.get("/api/public/runs/search?limit=1&offset=1")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3  # unpaginated
    assert len(body["items"]) == 1


def test_search_sort_by_final_loss_asc(search_client):
    client, _ = search_client
    r = client.get(
        "/api/public/runs/search?sort_by=final_train_loss&sort_dir=asc&has_eval=true"
    )
    assert r.status_code == 200
    body = r.json()
    losses = [i["final_train_loss"] for i in body["items"]]
    assert losses == sorted(losses)


def test_search_invalid_limit_returns_422(search_client):
    """Pydantic enforces ge=1 / le=500 on limit."""
    client, _ = search_client
    r = client.get("/api/public/runs/search?limit=10000")
    assert r.status_code == 422


def test_search_response_shape_matches_RunListItem(search_client):
    """Wire shape stable with the existing /runs surface so frontend
    list components don't have to branch on which endpoint fed them."""
    client, _ = search_client
    r = client.get("/api/public/runs/search?modality=sft")
    body = r.json()
    item = body["items"][0]
    expected = {"run_id", "modality", "model_name", "status", "timestamp"}
    assert expected.issubset(item.keys())
    # Effectiveness verdict round-trips when present.
    assert item["effectiveness"] == {"verdict": "passed"}
