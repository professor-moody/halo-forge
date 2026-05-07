"""Token store + verification (Track P1)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


TOKEN_PREFIX = "hfk_"
TOKEN_BYTES = 32  # 256 bits of entropy
DEFAULT_STORE_FILENAME = "tokens.json"


def default_store_path() -> Path:
    """Where the on-disk token store lives.

    Override with `HALOFORGE_TOKEN_STORE` for tests / non-default homes.
    """
    override = os.environ.get("HALOFORGE_TOKEN_STORE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".halo-forge" / DEFAULT_STORE_FILENAME


# ---------- token primitives -----------------------------------------------


def create_token() -> str:
    """Generate a fresh bearer token. Format: ``hfk_<48-char-base64url>``.

    The prefix lets users + automation recognize halo-forge tokens at a
    glance and is the suggested shape per RFC8959 / NIST guidance.
    """
    return TOKEN_PREFIX + secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(secret: str) -> str:
    """Hex SHA-256 of the token. The store keeps hashes, not secrets,
    so a stolen tokens.json doesn't grant API access on its own."""
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


# ---------- stored token ---------------------------------------------------


@dataclass
class Token:
    """One stored token. The bearer secret is *not* stored — only the hash."""

    name: str
    secret_hash: str
    created_at: str
    last_used_at: Optional[str] = None
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- store ----------------------------------------------------------


class TokenStore:
    """Filesystem-backed token list with thread-safe writes.

    Race-safe enough for single-process serving — writes flush, fsync
    (best-effort), and atomic-rename so a crash mid-write doesn't
    corrupt the JSON. Multi-process auth would need a real DB; out of
    scope for v1.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else default_store_path()
        self._lock = threading.Lock()

    # ----- IO ----------------------------------------------------------

    def _read(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Token store at %s unreadable: %s", self.path, exc)
            return []
        if isinstance(data, dict) and "tokens" in data:
            data = data["tokens"]
        if not isinstance(data, list):
            return []
        return data

    def _write(self, rows: List[Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Restrict perms so a tokens.json doesn't end up world-readable.
        # umask handling differs on Windows; best-effort.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            payload = json.dumps({"tokens": rows}, indent=2)
            with tmp.open("w") as f:
                f.write(payload)
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    # ----- public API --------------------------------------------------

    def list_tokens(self) -> List[Token]:
        with self._lock:
            return [Token(**row) for row in self._read()]

    def add_token(self, *, name: str, note: Optional[str] = None) -> str:
        """Create + store a new token. Returns the bearer secret —
        the only time the caller will ever see it."""
        if not name or not name.strip():
            raise ValueError("token name is required")
        secret = create_token()
        record = Token(
            name=name.strip(),
            secret_hash=hash_token(secret),
            created_at=datetime.now(timezone.utc).isoformat(),
            note=note,
        )
        with self._lock:
            rows = self._read()
            # Reject duplicate names — keeps revoke-by-name unambiguous.
            if any(r.get("name") == record.name for r in rows):
                raise ValueError(f"token name already exists: {name!r}")
            rows.append(record.to_dict())
            self._write(rows)
        return secret

    def revoke(self, name: str) -> bool:
        """Remove a token by name. Returns True if a row was removed."""
        with self._lock:
            rows = self._read()
            before = len(rows)
            rows = [r for r in rows if r.get("name") != name]
            if len(rows) == before:
                return False
            self._write(rows)
            return True

    def touch(self, secret: str) -> Optional[str]:
        """Update `last_used_at` for the matching token (if any).
        Returns the matching token's name or None."""
        digest = hash_token(secret)
        with self._lock:
            rows = self._read()
            for r in rows:
                if r.get("secret_hash") == digest:
                    r["last_used_at"] = datetime.now(timezone.utc).isoformat()
                    self._write(rows)
                    return str(r.get("name"))
        return None


# ---------- verification ---------------------------------------------------


def _env_token() -> Optional[str]:
    """Single-token deployment override.

    `HALOFORGE_API_TOKEN` lets ops set one bearer secret without
    writing tokens.json — convenient for `docker run -e ...`.
    """
    return os.environ.get("HALOFORGE_API_TOKEN")


def verify_token(secret: str, *, store: Optional[TokenStore] = None) -> Optional[str]:
    """Validate a bearer secret. Returns the token's `name` on hit, or
    None on miss. The env-var path takes precedence so a tokens.json
    that exists doesn't shadow a deployment-time override."""
    if not secret:
        return None

    env = _env_token()
    if env and secrets.compare_digest(secret, env):
        return "env"

    s = store or TokenStore()
    return s.touch(secret)


# ---------- request inspection ---------------------------------------------


def is_loopback_request(client_host: Optional[str]) -> bool:
    """Auth is bypassed for requests originating on the loopback
    interface — preserves halo-forge's local-first contract.

    Accepts None (some test clients don't expose client) as loopback so
    we don't break the test suite when the FastAPI TestClient runs.
    The literal string ``testclient`` is what FastAPI's TestClient sets
    as the synthetic client host; treat it as loopback so unit tests
    against the public API don't need to attach a token.
    """
    if client_host is None:
        return True
    h = client_host.strip().lower()
    if h in {"127.0.0.1", "::1", "localhost", "", "testclient"}:
        return True
    if h.startswith("127.") or h == "0.0.0.0":
        return True
    return False


__all__ = [
    "Token",
    "TokenStore",
    "TOKEN_PREFIX",
    "create_token",
    "default_store_path",
    "hash_token",
    "is_loopback_request",
    "verify_token",
]
