"""API token auth tests (Track P1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


# ---------- token primitives -----------------------------------------------


def test_create_token_has_prefix_and_entropy():
    from halo_forge.auth import TOKEN_PREFIX, create_token

    a = create_token()
    b = create_token()
    assert a != b
    assert a.startswith(TOKEN_PREFIX)
    # Length: prefix + base64url of 32 bytes ≈ 47 chars
    assert len(a) >= 40


def test_hash_token_is_deterministic_and_distinct():
    from halo_forge.auth import hash_token

    h1 = hash_token("hfk_alpha")
    h2 = hash_token("hfk_alpha")
    assert h1 == h2
    assert hash_token("hfk_beta") != h1
    # Hash, not plaintext.
    assert "alpha" not in h1


# ---------- token store ----------------------------------------------------


@pytest.fixture
def store(tmp_path: Path):
    from halo_forge.auth import TokenStore

    return TokenStore(path=tmp_path / "tokens.json")


def test_store_starts_empty(store):
    assert store.list_tokens() == []


def test_add_token_returns_secret_and_persists_hash(store):
    secret = store.add_token(name="dashboard")
    assert secret.startswith("hfk_")
    tokens = store.list_tokens()
    assert len(tokens) == 1
    assert tokens[0].name == "dashboard"
    # Plaintext secret never lands on disk.
    assert tokens[0].secret_hash != secret


def test_add_token_rejects_empty_name(store):
    with pytest.raises(ValueError):
        store.add_token(name="")


def test_add_token_rejects_duplicate_name(store):
    store.add_token(name="ci")
    with pytest.raises(ValueError, match="already exists"):
        store.add_token(name="ci")


def test_revoke_removes_token(store):
    store.add_token(name="ci")
    assert store.revoke("ci") is True
    assert store.list_tokens() == []
    # Idempotent: second revoke returns False.
    assert store.revoke("ci") is False


def test_touch_updates_last_used_for_matching_token(store):
    secret = store.add_token(name="dashboard")
    name = store.touch(secret)
    assert name == "dashboard"
    tokens = store.list_tokens()
    assert tokens[0].last_used_at is not None


def test_touch_returns_none_for_unknown_secret(store):
    assert store.touch("hfk_unknown") is None


def test_store_file_is_chmod_0600(tmp_path: Path):
    """Token file should not be world-readable (Unix only)."""
    if sys.platform.startswith("win"):
        pytest.skip("chmod semantics differ on Windows")
    from halo_forge.auth import TokenStore

    s = TokenStore(path=tmp_path / "tokens.json")
    s.add_token(name="t")
    mode = (tmp_path / "tokens.json").stat().st_mode & 0o777
    assert mode == 0o600


def test_store_handles_corrupt_file(tmp_path: Path, caplog):
    from halo_forge.auth import TokenStore

    p = tmp_path / "tokens.json"
    p.write_text("{not valid json")
    with caplog.at_level("WARNING"):
        s = TokenStore(path=p)
        assert s.list_tokens() == []


# ---------- verify_token ---------------------------------------------------


def test_verify_token_with_store_hit(store):
    secret = store.add_token(name="ci")
    from halo_forge.auth import verify_token

    assert verify_token(secret, store=store) == "ci"


def test_verify_token_miss(store):
    from halo_forge.auth import verify_token

    assert verify_token("hfk_does_not_exist", store=store) is None


def test_verify_token_env_var_override(monkeypatch, store):
    """`HALOFORGE_API_TOKEN` is the single-token deployment shortcut."""
    from halo_forge.auth import verify_token

    monkeypatch.setenv("HALOFORGE_API_TOKEN", "hfk_envonly")
    assert verify_token("hfk_envonly", store=store) == "env"
    # A non-matching secret still lands in the store-check path.
    assert verify_token("hfk_other", store=store) is None


def test_verify_token_empty_string_rejects():
    from halo_forge.auth import verify_token

    assert verify_token("") is None


# ---------- loopback detection ---------------------------------------------


def test_is_loopback_request_for_canonical_addresses():
    from halo_forge.auth import is_loopback_request

    assert is_loopback_request("127.0.0.1") is True
    assert is_loopback_request("::1") is True
    assert is_loopback_request("localhost") is True
    assert is_loopback_request(None) is True  # unknown client = bypass


def test_is_loopback_request_for_external_addresses():
    from halo_forge.auth import is_loopback_request

    assert is_loopback_request("10.0.0.5") is False
    assert is_loopback_request("192.168.1.10") is False
    assert is_loopback_request("203.0.113.42") is False


def test_is_loopback_request_for_127_subrange():
    from halo_forge.auth import is_loopback_request

    # Anything in 127.x.x.x is loopback per RFC.
    assert is_loopback_request("127.0.0.5") is True


# ---------- FastAPI dependency --------------------------------------------


def test_dependency_extract_bearer_handles_shapes():
    from halo_forge.auth.dependency import _extract_bearer

    assert _extract_bearer("Bearer hfk_123") == "hfk_123"
    assert _extract_bearer("bearer hfk_123") == "hfk_123"  # case-insensitive
    assert _extract_bearer("Token hfk_123") is None  # wrong scheme
    assert _extract_bearer("") is None
    assert _extract_bearer(None) is None
    assert _extract_bearer("Bearer  hfk_123") == "hfk_123"  # extra whitespace ok


def test_dependency_passes_for_loopback_request(monkeypatch):
    """Loopback requests bypass auth — that's the local-first contract."""
    from halo_forge.auth.dependency import require_token

    class _FakeClient:
        host = "127.0.0.1"

    class _FakeRequest:
        client = _FakeClient()
        headers: dict = {}

    name = require_token(_FakeRequest())
    assert name == "loopback"


def test_dependency_401s_for_non_loopback_without_token(monkeypatch, store):
    from halo_forge.auth.dependency import require_token, reset_store_for_tests
    from fastapi import HTTPException

    reset_store_for_tests(store)
    try:
        class _FakeClient:
            host = "10.0.0.5"

        class _FakeRequest:
            client = _FakeClient()
            headers: dict = {}

        with pytest.raises(HTTPException) as ei:
            require_token(_FakeRequest())
        assert ei.value.status_code == 401
        assert "invalid_token" in str(ei.value.detail)
    finally:
        reset_store_for_tests(None)


def test_dependency_passes_for_non_loopback_with_valid_token(store):
    from halo_forge.auth.dependency import require_token, reset_store_for_tests

    secret = store.add_token(name="ci")
    reset_store_for_tests(store)
    try:
        class _FakeClient:
            host = "10.0.0.5"

        class _FakeHeaders:
            def get(self, key):
                if key.lower() == "authorization":
                    return f"Bearer {secret}"
                return None

        class _FakeRequest:
            client = _FakeClient()
            headers = _FakeHeaders()

        name = require_token(_FakeRequest())
        assert name == "ci"
    finally:
        reset_store_for_tests(None)


# ---------- end-to-end through TestClient ---------------------------------


def test_health_endpoint_works_loopback(monkeypatch, tmp_path):
    """TestClient mimics loopback → no auth required."""
    monkeypatch.setenv("HALOFORGE_TOKEN_STORE", str(tmp_path / "tokens.json"))
    from halo_forge.auth.dependency import reset_store_for_tests
    reset_store_for_tests(None)

    from fastapi.testclient import TestClient
    from halo_forge.public_api.app import create_app

    app = create_app()
    with TestClient(app) as client:
        r = client.get("/api/public/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


# ---------- CLI ------------------------------------------------------------


def test_cli_token_create_list_revoke_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HALOFORGE_TOKEN_STORE", str(tmp_path / "tokens.json"))
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "token", "create", "ci"])
    cli_mod.main()
    out_create = capsys.readouterr().out
    assert "hfk_" in out_create  # secret printed once

    monkeypatch.setattr(sys, "argv", ["halo-forge", "token", "list"])
    cli_mod.main()
    out_list = capsys.readouterr().out
    assert "ci" in out_list

    monkeypatch.setattr(sys, "argv", ["halo-forge", "token", "revoke", "ci"])
    cli_mod.main()
    out_revoke = capsys.readouterr().out
    assert "Revoked" in out_revoke


def test_cli_token_revoke_unknown_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("HALOFORGE_TOKEN_STORE", str(tmp_path / "tokens.json"))
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "token", "revoke", "nope"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 1


def test_cli_token_help_lists_subcommands(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "token", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for sub in ("create", "list", "revoke"):
        assert sub in out
