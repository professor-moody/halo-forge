"""Model registry tests (Track F-J).

Library-level (RunDatabase CRUD) plus end-to-end through the FastAPI
TestClient.
"""

from __future__ import annotations

import pytest


# ---------- RunDatabase CRUD -----------------------------------------------


@pytest.fixture
def db(tmp_path):
    from halo_forge.run_db import RunDatabase

    inst = RunDatabase(":memory:")
    yield inst
    inst.close()


def test_create_registry_entry(db):
    entry = db.create_registry_entry(
        name="prod-2026-q2",
        description="Top picks from the Q2 sweep",
        base_model="Qwen/Qwen2.5-3B-Instruct",
        run_ids=["run-a", "run-b", "run-c"],
        tags=["dpo", "production"],
    )
    assert entry.id > 0
    assert entry.name == "prod-2026-q2"
    assert entry.run_ids == ["run-a", "run-b", "run-c"]
    assert entry.tags == ["dpo", "production"]
    assert entry.created_at
    assert entry.updated_at == entry.created_at  # first write


def test_create_registry_entry_dedupes_ids_and_tags(db):
    entry = db.create_registry_entry(
        name="dups",
        run_ids=["a", "b", "a", "c", "b"],
        tags=["x", "x", "y"],
    )
    assert entry.run_ids == ["a", "b", "c"]
    assert entry.tags == ["x", "y"]


def test_create_registry_entry_rejects_empty_name(db):
    with pytest.raises(ValueError, match="name is required"):
        db.create_registry_entry(name="")
    with pytest.raises(ValueError):
        db.create_registry_entry(name="   ")


def test_create_registry_entry_rejects_duplicate_name(db):
    db.create_registry_entry(name="alpha")
    with pytest.raises(ValueError, match="already exists"):
        db.create_registry_entry(name="alpha")


def test_get_registry_entry_by_id_and_by_name(db):
    a = db.create_registry_entry(name="a")
    b = db.create_registry_entry(name="b")

    assert db.get_registry_entry(a.id).name == "a"
    assert db.get_registry_entry(b.id).name == "b"
    assert db.get_registry_entry_by_name("a").id == a.id
    assert db.get_registry_entry_by_name("does_not_exist") is None
    assert db.get_registry_entry(99999) is None


def test_list_registry_entries_orders_by_updated_at_desc(db):
    import time

    a = db.create_registry_entry(name="first")
    time.sleep(0.005)
    b = db.create_registry_entry(name="second")
    time.sleep(0.005)
    db.update_registry_entry(a.id, description="touched")

    items = db.list_registry_entries()
    # `a` was updated most recently, so it should come first.
    assert [e.name for e in items[:2]] == ["first", "second"]


def test_update_registry_entry_only_changes_supplied_fields(db):
    e = db.create_registry_entry(
        name="e", description="original", run_ids=["r1"], tags=["x"],
    )
    updated = db.update_registry_entry(e.id, description="new")
    assert updated.description == "new"
    assert updated.run_ids == ["r1"]    # unchanged
    assert updated.tags == ["x"]


def test_update_registry_entry_replaces_run_ids_when_supplied(db):
    e = db.create_registry_entry(name="e", run_ids=["r1", "r2"])
    updated = db.update_registry_entry(e.id, run_ids=["r3", "r4"])
    assert updated.run_ids == ["r3", "r4"]


def test_update_registry_entry_clears_description_with_empty_string(db):
    e = db.create_registry_entry(name="e", description="initial")
    updated = db.update_registry_entry(e.id, description="")
    assert updated.description == ""


def test_update_registry_entry_returns_none_for_unknown_id(db):
    assert db.update_registry_entry(99999, description="x") is None


def test_delete_registry_entry(db):
    e = db.create_registry_entry(name="x")
    assert db.delete_registry_entry(e.id) is True
    assert db.get_registry_entry(e.id) is None
    # Idempotent: a second delete returns False.
    assert db.delete_registry_entry(e.id) is False


# ---------- end-to-end via FastAPI TestClient ------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "runs.db"))

    from halo_forge.run_db import db as db_mod
    db_mod._GLOBAL_DB.clear()

    from halo_forge.auth.dependency import reset_store_for_tests
    reset_store_for_tests(None)

    from fastapi.testclient import TestClient
    from halo_forge.public_api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c

    db_mod._GLOBAL_DB.clear()


def test_registry_create_list_get_update_delete_roundtrip(client):
    # Create
    r = client.post("/api/public/registry", json={
        "name": "alpha",
        "description": "first",
        "run_ids": ["r1", "r2"],
        "tags": ["sft"],
    })
    assert r.status_code == 200
    created = r.json()
    assert created["name"] == "alpha"
    assert created["run_ids"] == ["r1", "r2"]

    # List
    r = client.get("/api/public/registry")
    assert r.status_code == 200
    items = r.json()["items"]
    assert any(e["name"] == "alpha" for e in items)

    # Get
    r = client.get(f"/api/public/registry/{created['id']}")
    assert r.status_code == 200
    assert r.json()["description"] == "first"

    # Patch — change description, leave run_ids alone
    r = client.patch(f"/api/public/registry/{created['id']}", json={
        "description": "updated",
    })
    assert r.status_code == 200
    patched = r.json()
    assert patched["description"] == "updated"
    assert patched["run_ids"] == ["r1", "r2"]

    # Delete
    r = client.delete(f"/api/public/registry/{created['id']}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True

    # Now 404 on get.
    r = client.get(f"/api/public/registry/{created['id']}")
    assert r.status_code == 404


def test_registry_create_duplicate_name_returns_400(client):
    client.post("/api/public/registry", json={"name": "dup"})
    r = client.post("/api/public/registry", json={"name": "dup"})
    assert r.status_code == 400
    assert "already exists" in r.json()["detail"]


def test_registry_create_empty_name_returns_400(client):
    r = client.post("/api/public/registry", json={"name": ""})
    assert r.status_code == 400


def test_registry_get_unknown_returns_404(client):
    r = client.get("/api/public/registry/99999")
    assert r.status_code == 404


def test_registry_patch_unknown_returns_404(client):
    r = client.patch("/api/public/registry/99999", json={"description": "x"})
    assert r.status_code == 404
