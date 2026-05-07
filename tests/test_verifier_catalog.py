"""Verifier-catalog endpoint tests (Track F-O)."""

from __future__ import annotations

import textwrap

import pytest


# ---------- registry inventory --------------------------------------------


def test_inventory_lists_builtins_with_origin_tag():
    from halo_forge.rlvr.verifiers.registry import inventory

    items = inventory()
    by_name = {e["name"]: e for e in items}
    # A handful of canonical builtins must show up.
    for name in ("gcc", "pytest", "json_schema", "bleu", "llm_judge"):
        assert name in by_name, f"missing builtin {name!r}"
        assert by_name[name]["origin"] == "builtin"
        assert by_name[name]["module"].startswith("halo_forge.rlvr.verifiers")
        assert by_name[name]["base"] == "Verifier"


def test_inventory_tags_user_plugin_origin(tmp_path, monkeypatch):
    """A `.py` dropped into the plugin dir is tagged ``user_plugin``."""
    from halo_forge.rlvr.verifiers import registry as reg

    plugin_dir = tmp_path / "verifiers"
    plugin_dir.mkdir()
    (plugin_dir / "my_check.py").write_text(textwrap.dedent("""
        from halo_forge.rlvr.verifiers.base import Verifier
        from halo_forge.rlvr.verifiers.registry import register_verifier

        @register_verifier("my_check_v1")
        class MyCheck(Verifier):
            '''Project-specific verifier for the catalog test.'''
            def verify(self, sample):
                return {"passed": True, "score": 1.0}
    """))

    monkeypatch.setenv("HALOFORGE_VERIFIERS_DIR", str(plugin_dir))
    reg.reset_registry_for_tests()
    reg._seed_builtin_registrations()

    items = reg.inventory()
    by_name = {e["name"]: e for e in items}
    assert "my_check_v1" in by_name
    assert by_name["my_check_v1"]["origin"] == "user_plugin"
    assert by_name["my_check_v1"]["doc"] == "Project-specific verifier for the catalog test."

    # Cleanup so subsequent tests start clean.
    reg.reset_registry_for_tests()
    reg._seed_builtin_registrations()


def test_inventory_is_sorted():
    from halo_forge.rlvr.verifiers.registry import inventory

    names = [e["name"] for e in inventory()]
    assert names == sorted(names), "inventory() should return entries in name order"


# ---------- end-to-end through TestClient ---------------------------------


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


def test_verifier_catalog_endpoint_returns_inventory(client):
    r = client.get("/api/public/verifiers")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert "counts" in body
    assert "plugin_dir" in body
    assert body["total"] == len(body["items"])
    # Counts should sum to total.
    assert sum(body["counts"].values()) == body["total"]


def test_verifier_catalog_includes_canonical_builtins(client):
    r = client.get("/api/public/verifiers")
    assert r.status_code == 200
    body = r.json()
    names = {e["name"] for e in body["items"]}
    for canonical in ("gcc", "pytest", "json_schema", "bleu", "rouge", "chrf", "llm_judge"):
        assert canonical in names, f"catalog missing {canonical!r}"


def test_verifier_catalog_distinct_from_training_verifiers(client):
    """The training-verifier list is the code-execution-only subset; the
    full catalog includes V2/V3/V4 (judge, schema, metrics) too."""
    train = client.get("/api/public/train/verifiers")
    catalog = client.get("/api/public/verifiers")
    assert train.status_code == 200
    assert catalog.status_code == 200
    train_keys = {e.get("key") for e in train.json()["items"]}
    catalog_names = {e["name"] for e in catalog.json()["items"]}
    # Catalog is a superset and includes things training-verifiers list omits.
    assert "llm_judge" in catalog_names
    assert "json_schema" in catalog_names
    assert "llm_judge" not in train_keys
