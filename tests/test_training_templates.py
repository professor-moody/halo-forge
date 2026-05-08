"""Training-template registry tests."""

from __future__ import annotations

import pytest


# ---------- registry contract -------------------------------------------


def test_every_template_uses_a_canonical_modality():
    """Templates must map to a real trainer the runtime supports.

    If this fails, either the trainer was deleted (drop the template)
    or a new modality was introduced that hasn't been added to the
    canonical list yet."""
    from halo_forge.training import TEMPLATES

    canonical = {
        "sft", "raft", "dpo", "grpo", "rm",
        "vlm", "audio", "reasoning", "agentic",
    }
    for tpl in TEMPLATES:
        assert tpl.modality in canonical, (
            f"Template {tpl.id!r} uses non-canonical modality {tpl.modality!r}"
        )


def test_every_template_lives_in_a_known_category():
    from halo_forge.training import CATEGORIES, TEMPLATES

    cat_ids = {c[0] for c in CATEGORIES}
    for tpl in TEMPLATES:
        assert tpl.category in cat_ids, (
            f"Template {tpl.id!r} has unknown category {tpl.category!r}"
        )


def test_template_ids_are_unique():
    from halo_forge.training import TEMPLATES

    ids = [t.id for t in TEMPLATES]
    assert len(ids) == len(set(ids)), "Duplicate template id(s)"


def test_raft_and_grpo_templates_have_a_verifier():
    """RAFT and GRPO are reward-driven loops — a template that doesn't
    name a verifier would fail at launch time."""
    from halo_forge.training import TEMPLATES

    for tpl in TEMPLATES:
        if tpl.modality in ("raft", "grpo"):
            assert tpl.verifier, (
                f"{tpl.modality.upper()} template {tpl.id!r} must declare a verifier"
            )


def test_list_templates_orders_by_category_then_id():
    from halo_forge.training import CATEGORIES, list_templates

    cat_order = {key: idx for idx, (key, _, _) in enumerate(CATEGORIES)}
    seen_indices = [cat_order[t["category"]] for t in list_templates()]
    assert seen_indices == sorted(seen_indices), (
        "Templates must be ordered by category, then by id"
    )


# ---------- get_template + cli_invocation -------------------------------


def test_get_template_returns_known_id():
    from halo_forge.training import get_template

    tpl = get_template("code-python-sft")
    assert tpl is not None
    assert tpl["modality"] == "sft"
    assert "model_hint" in tpl


def test_get_template_returns_none_for_unknown():
    from halo_forge.training import get_template

    assert get_template("does-not-exist") is None


def test_cli_invocation_renders_full_command():
    from halo_forge.training import cli_invocation

    cmd = cli_invocation("code-python-sft")
    assert cmd is not None
    assert cmd.startswith("halo-forge sft train")
    assert "--model Qwen/Qwen2.5-Coder-0.5B" in cmd
    assert "--dataset codealpaca" in cmd
    assert "--epochs 3" in cmd
    assert "--learning-rate 0.0002" in cmd or "--learning-rate 2e-04" in cmd


def test_cli_invocation_skips_dataset_for_custom():
    """`@custom` is the sentinel for "user supplies their own dataset
    file" — it shouldn't render as `--dataset @custom`."""
    from halo_forge.training import cli_invocation

    cmd = cli_invocation("vision-document-extraction")
    assert cmd is not None
    assert "@custom" not in cmd


def test_cli_invocation_includes_verifier_for_raft():
    from halo_forge.training import cli_invocation

    cmd = cli_invocation("code-multi-language-raft")
    assert cmd is not None
    assert "--verifier gcc" in cmd


# ---------- end-to-end through TestClient -------------------------------


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


def test_training_templates_endpoint(client):
    r = client.get("/api/public/train/templates")
    assert r.status_code == 200
    body = r.json()
    assert "items" in body
    assert "categories" in body
    # Categories ship in a known order.
    cat_ids = [c["id"] for c in body["categories"]]
    assert cat_ids == ["code", "reasoning", "vision", "audio", "preference", "agentic"]
    # All canonical trainer kinds should have at least one template.
    seen = {t["modality"] for t in body["items"]}
    assert {"sft", "raft", "dpo", "grpo", "rm", "vlm", "audio", "reasoning", "agentic"}.issubset(seen)


def test_training_template_detail_endpoint(client):
    r = client.get("/api/public/train/templates/pref-dpo-chat")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "pref-dpo-chat"
    assert body["modality"] == "dpo"
    assert body["cli"].startswith("halo-forge dpo train")


def test_training_template_detail_404(client):
    r = client.get("/api/public/train/templates/nope")
    assert r.status_code == 404
