"""FastAPI dependency that enforces bearer-token auth (Track P1).

Rule:

  - Requests from loopback (127.0.0.1 / ::1 / localhost) pass without
    auth. This preserves the local-first default.
  - Requests from any other interface require an `Authorization: Bearer
    <token>` header that resolves to a stored token (or matches the
    `HALOFORGE_API_TOKEN` env var).
  - Missing / invalid tokens get 401 with a typed error body.

Wire it onto the FastAPI router with ``Depends(require_token)``; we
attach it once at app construction so every endpoint inherits the
gate without per-handler boilerplate.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from halo_forge.auth.tokens import (
    TokenStore,
    is_loopback_request,
    verify_token,
)

logger = logging.getLogger(__name__)

# Sentinel store; tests can monkeypatch this to inject a fixture.
_default_store: Optional[TokenStore] = None


def get_token_store() -> TokenStore:
    """Process-global token store. Lazily constructed on first access."""
    global _default_store
    if _default_store is None:
        _default_store = TokenStore()
    return _default_store


def reset_store_for_tests(store: Optional[TokenStore] = None) -> None:
    """Test-only — replace the cached store."""
    global _default_store
    _default_store = store


def _extract_bearer(authorization_header: Optional[str]) -> Optional[str]:
    if not authorization_header:
        return None
    parts = authorization_header.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, value = parts
    if scheme.strip().lower() != "bearer":
        return None
    return value.strip() or None


def require_token(request: Any) -> Optional[str]:
    """FastAPI dependency. Returns the authenticated token name (or
    "loopback" for bypassed requests). Raises HTTPException(401) on miss.

    Doesn't import FastAPI at module load so the auth module loads on
    hosts that don't have FastAPI installed (e.g. CLI-only smoke).
    """
    client_host = None
    try:
        client_host = request.client.host if request.client else None
    except AttributeError:
        client_host = None

    if is_loopback_request(client_host):
        return "loopback"

    auth_header = None
    try:
        auth_header = request.headers.get("authorization") or request.headers.get(
            "Authorization"
        )
    except AttributeError:
        auth_header = None

    secret = _extract_bearer(auth_header)
    name = verify_token(secret or "", store=get_token_store())
    if name:
        return name

    from fastapi import HTTPException

    raise HTTPException(
        status_code=401,
        detail={
            "error": "invalid_token",
            "message": (
                "Bearer token missing or invalid. Halo-forge auth is "
                "automatic when bound to non-loopback. Generate a token "
                "with `halo-forge token create <name>` or set "
                "HALOFORGE_API_TOKEN."
            ),
        },
        headers={"WWW-Authenticate": 'Bearer realm="halo-forge"'},
    )


__all__ = [
    "get_token_store",
    "require_token",
    "reset_store_for_tests",
]
