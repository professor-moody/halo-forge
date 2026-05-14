---
title: "Auth + tokens"
description: "Bearer-token API auth that turns on automatically when bound to non-loopback. Local-first stays zero-config."
weight: 70
---

Halo-forge stays local-first by default: bound to `127.0.0.1`, no auth, no friction. The moment the user binds to a non-loopback interface — explicit `--host 0.0.0.0` or any external bind — bearer-token auth turns on automatically.

No surprises in either direction. Operators don't trip over auth on their laptop, and a non-loopback bind is never silently exposed.

## Token lifecycle

```bash
halo-forge token create dashboard
# name:  dashboard
# Save this token now — it won't be shown again:
#   hfk_3R9xV2k...

halo-forge token list
# NAME           CREATED                    LAST USED                  NOTE
# dashboard      2026-05-07T12:00:00+00:00  2026-05-07T12:14:22+00:00

halo-forge token revoke dashboard
```

Tokens are stored at `~/.halo-forge/tokens.json` (overridable with `HALOFORGE_TOKEN_STORE`). The store keeps SHA-256 hashes, **not** the bearer secret — a leaked `tokens.json` doesn't grant API access on its own. Permissions are set to `0o600` on Unix.

## Using a token

```bash
curl -H "Authorization: Bearer hfk_..." http://my-host:8000/api/public/health
```

The frontend at `public_app/` reads from `localStorage["halo-forge:api-token"]` and injects the header automatically. Use the in-app **Connection** screen to paste and test the token. For automation, the same value can be pre-seeded:

```javascript
localStorage.setItem("halo-forge:api-token", "hfk_...")
```

## Single-token deployments

For docker-secret / Kubernetes-secret style deployments, set `HALOFORGE_API_TOKEN` instead of using the file store:

```bash
HALOFORGE_API_TOKEN=hfk_envonly halo-forge serve --host 0.0.0.0
```

The env-var path takes precedence over the file store.

## Loopback bypass

These client hosts skip the auth gate entirely:

- `127.0.0.1`, `::1`, `localhost`
- Any address in `127.x.x.x`
- `testclient` (FastAPI's TestClient default; keeps the unit tests green)
- Empty / unknown client (defensive default)

So `halo-forge serve` (default `--host 127.0.0.1`) is zero-config, and the tests in this repo don't have to attach tokens.

## Programmatic API

```python
from halo_forge.auth import TokenStore, verify_token, is_loopback_request

store = TokenStore()
secret = store.add_token(name="ci")
# Save secret — it's the only time you see it.

# Validate later:
name = verify_token(secret, store=store)
assert name == "ci"

# Check a request:
if not is_loopback_request(request.client.host):
    # require Authorization header
    ...
```

## What auth does + does not cover

- ✅ Every endpoint under `/api/public/*` (the FastAPI router-level dependency catches them all).
- ✅ SSE telemetry / log streams (same router).
- ❌ The frontend's static assets (served by Vite in dev, by the FastAPI host in prod). They're public.
- ❌ The model files on disk. If you bind a server to `0.0.0.0` your filesystem permissions still matter.

## Roadmap

- **Per-token scopes** — read-only / write / admin. Single-tier in v1.
- **TLS termination** — bring-your-own; halo-forge expects to sit behind a reverse proxy for TLS. ACME integration is roadmap.
- **OIDC / SSO** — for organizational deployments. Not a v1 priority.
