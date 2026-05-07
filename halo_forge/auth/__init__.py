"""API token auth (Track P1).

Halo-forge stays *local-first* by default: bound to 127.0.0.1, no auth
required, no friction. The moment the user binds to a non-loopback
interface — explicit `--host 0.0.0.0` or `0.0.0.0:8000` — auth turns
on automatically. No surprises in either direction.

Three pieces:

  1. **TokenStore**: persisted to ``~/.halo-forge/tokens.json``. A token
     is a (name, secret_hash, created_at, last_used_at) tuple; we store
     the hash, not the secret, so a leaked tokens.json doesn't leak the
     bearer secret.
  2. **FastAPI dependency**: `require_token(request)` checks for a
     `Authorization: Bearer <token>` header, validates against the
     store, and 401s on miss. Skipped automatically when the request
     came in over loopback (matches "local-first by default").
  3. **CLI**: `halo-forge token create / list / revoke` for token
     lifecycle.

Env-var override `HALOFORGE_API_TOKEN` lets a single-token deployment
configure auth without writing to disk — useful for `kubectl create
secret` / docker secret mounting / one-shot remote demos.
"""

from halo_forge.auth.tokens import (
    TOKEN_PREFIX,
    Token,
    TokenStore,
    create_token,
    default_store_path,
    hash_token,
    is_loopback_request,
    verify_token,
)

__all__ = [
    "TOKEN_PREFIX",
    "Token",
    "TokenStore",
    "create_token",
    "default_store_path",
    "hash_token",
    "is_loopback_request",
    "verify_token",
]
