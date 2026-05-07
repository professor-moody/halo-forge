"""Run lineage tests (Track F-Q).

Library-level CRUD + BFS traversal, plus end-to-end through the
FastAPI TestClient.
"""

from __future__ import annotations

import pytest


# ---------- RunDatabase CRUD -----------------------------------------------


@pytest.fixture
def db(tmp_path):
    from halo_forge.run_db import RunDatabase

    inst = RunDatabase(":memory:")
    # Seed parent + child rows so the FK refs satisfy.
    for run_id in ("alpha", "beta", "gamma", "delta", "epsilon"):
        inst.upsert_run(_make_run(run_id))
    yield inst
    inst.close()


def _make_run(run_id):
    from halo_forge.run_db import RunRecord

    return RunRecord(
        run_id=run_id, modality="sft", model_name="x", status="completed",
        timestamp="2026-05-01T00:00:00+00:00", output_dir=f"/tmp/{run_id}",
    )


def test_record_fork_creates_edge(db):
    db.record_fork(child_run_id="beta", parent_run_id="alpha", forked_at_cycle=3)
    parents = db.get_parents("beta")
    assert len(parents) == 1
    assert parents[0]["parent_run_id"] == "alpha"
    assert parents[0]["forked_at_cycle"] == 3


def test_record_fork_is_idempotent_and_updates_cycle(db):
    db.record_fork(child_run_id="beta", parent_run_id="alpha", forked_at_cycle=3)
    db.record_fork(
        child_run_id="beta", parent_run_id="alpha",
        forked_at_cycle=5, notes="updated",
    )
    parents = db.get_parents("beta")
    assert len(parents) == 1
    assert parents[0]["forked_at_cycle"] == 5
    assert parents[0]["notes"] == "updated"


def test_record_fork_rejects_self_parent(db):
    with pytest.raises(ValueError, match="own parent"):
        db.record_fork(child_run_id="alpha", parent_run_id="alpha")


def test_record_fork_rejects_empty_ids(db):
    with pytest.raises(ValueError):
        db.record_fork(child_run_id="", parent_run_id="alpha")
    with pytest.raises(ValueError):
        db.record_fork(child_run_id="beta", parent_run_id="")


def test_remove_fork_deletes_edge(db):
    db.record_fork(child_run_id="beta", parent_run_id="alpha")
    assert db.remove_fork(child_run_id="beta", parent_run_id="alpha") is True
    assert db.get_parents("beta") == []
    # Idempotent: second remove returns False.
    assert db.remove_fork(child_run_id="beta", parent_run_id="alpha") is False


def test_get_children_inverts_get_parents(db):
    db.record_fork(child_run_id="beta", parent_run_id="alpha")
    db.record_fork(child_run_id="gamma", parent_run_id="alpha")
    children = db.get_children("alpha")
    assert {c["child_run_id"] for c in children} == {"beta", "gamma"}


def test_get_lineage_walks_ancestors_transitively(db):
    # alpha → beta → gamma → delta
    db.record_fork(child_run_id="beta", parent_run_id="alpha")
    db.record_fork(child_run_id="gamma", parent_run_id="beta")
    db.record_fork(child_run_id="delta", parent_run_id="gamma")

    lineage = db.get_lineage("delta")
    parent_ids = {a["parent_run_id"] for a in lineage["ancestors"]}
    assert parent_ids == {"gamma", "beta", "alpha"}
    # Depth tracking: closest parent depth=1, root depth=3.
    by_id = {a["parent_run_id"]: a for a in lineage["ancestors"]}
    assert by_id["gamma"]["depth"] == 1
    assert by_id["alpha"]["depth"] == 3


def test_get_lineage_walks_descendants_transitively(db):
    db.record_fork(child_run_id="beta", parent_run_id="alpha")
    db.record_fork(child_run_id="gamma", parent_run_id="beta")
    db.record_fork(child_run_id="delta", parent_run_id="alpha")

    lineage = db.get_lineage("alpha")
    child_ids = {d["child_run_id"] for d in lineage["descendants"]}
    assert child_ids == {"beta", "gamma", "delta"}


def test_get_lineage_handles_a_run_with_no_lineage(db):
    lineage = db.get_lineage("alpha")
    assert lineage["ancestors"] == []
    assert lineage["descendants"] == []
    assert lineage["run_id"] == "alpha"


def test_get_lineage_max_depth_bounds_traversal(db):
    """A pathological cycle (or just deep chain) must not loop forever."""
    # Build a chain: alpha → beta → gamma → delta → epsilon
    db.record_fork(child_run_id="beta", parent_run_id="alpha")
    db.record_fork(child_run_id="gamma", parent_run_id="beta")
    db.record_fork(child_run_id="delta", parent_run_id="gamma")
    db.record_fork(child_run_id="epsilon", parent_run_id="delta")

    lineage = db.get_lineage("epsilon", max_depth=2)
    # Only 2 hops up: delta + gamma. alpha + beta are out of range.
    parent_ids = {a["parent_run_id"] for a in lineage["ancestors"]}
    assert parent_ids == {"delta", "gamma"}


def test_get_lineage_dedupes_when_diamond_inheritance(db):
    """A child with two paths to the same ancestor only emits the
    ancestor once (BFS dedup)."""
    # diamond: alpha → beta, alpha → gamma, both → delta
    db.record_fork(child_run_id="beta", parent_run_id="alpha")
    db.record_fork(child_run_id="gamma", parent_run_id="alpha")
    db.record_fork(child_run_id="delta", parent_run_id="beta")
    db.record_fork(child_run_id="delta", parent_run_id="gamma")

    lineage = db.get_lineage("delta")
    parent_ids = [a["parent_run_id"] for a in lineage["ancestors"]]
    # alpha appears once even though both beta and gamma have it as parent.
    assert parent_ids.count("alpha") == 1


# ---------- end-to-end through TestClient ---------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "runs.db"))
    from halo_forge.run_db import db as db_mod
    db_mod._GLOBAL_DB.clear()
    from halo_forge.auth.dependency import reset_store_for_tests
    reset_store_for_tests(None)

    # Seed runs in the global DB so FK refs hold.
    from halo_forge.run_db import get_database
    g = get_database()
    for run_id in ("alpha", "beta", "gamma"):
        g.upsert_run(_make_run(run_id))

    from fastapi.testclient import TestClient
    from halo_forge.public_api.app import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    db_mod._GLOBAL_DB.clear()


def test_record_fork_returns_lineage(client):
    r = client.post("/api/public/runs/beta/lineage", json={
        "parent_run_id": "alpha",
        "forked_at_cycle": 4,
        "notes": "lr 5e-6 → 1e-6",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["run_id"] == "beta"
    assert len(body["ancestors"]) == 1
    assert body["ancestors"][0]["parent_run_id"] == "alpha"


def test_record_fork_missing_parent_400(client):
    r = client.post("/api/public/runs/beta/lineage", json={})
    assert r.status_code == 400


def test_record_fork_self_parent_400(client):
    r = client.post("/api/public/runs/alpha/lineage", json={
        "parent_run_id": "alpha",
    })
    assert r.status_code == 400


def test_get_run_lineage_walks_chain(client):
    client.post("/api/public/runs/beta/lineage", json={"parent_run_id": "alpha"})
    client.post("/api/public/runs/gamma/lineage", json={"parent_run_id": "beta"})

    r = client.get("/api/public/runs/gamma/lineage")
    assert r.status_code == 200
    body = r.json()
    parent_ids = {a["parent_run_id"] for a in body["ancestors"]}
    assert parent_ids == {"beta", "alpha"}


def test_delete_run_fork(client):
    client.post("/api/public/runs/beta/lineage", json={"parent_run_id": "alpha"})
    r = client.delete("/api/public/runs/beta/lineage/alpha")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    # Now 404 on second delete.
    r = client.delete("/api/public/runs/beta/lineage/alpha")
    assert r.status_code == 404
