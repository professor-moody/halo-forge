"""Workstation-scoped Hugging Face credential handling.

The dashboard needs to use gated/private Hugging Face models without
putting HF tokens in browser storage. This module keeps tokens on the
workstation process and only returns sanitized status to the public API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping


HF_TOKEN_ENV = "HF_TOKEN"
HF_TOKEN_FILE_ENV = "HALO_FORGE_HF_TOKEN_FILE"
KEYRING_SERVICE = "halo-forge"
KEYRING_USERNAME = "huggingface-token"
GATED_MODEL_ACTION = "connect_huggingface"


def _default_token_file() -> Path:
    configured = os.environ.get(HF_TOKEN_FILE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".halo-forge" / "secrets" / "huggingface_token"


def _import_keyring() -> Any | None:
    try:
        import keyring  # type: ignore[import-not-found]

        return keyring
    except Exception:
        return None


@dataclass(frozen=True)
class TokenResolution:
    present: bool
    source: str
    token: str | None = None


class HuggingFaceAccessManager:
    """Resolve, verify, store, and clear workstation HF credentials."""

    def __init__(
        self,
        *,
        token_file: Path | None = None,
        env: Mapping[str, str] | None = None,
        keyring_module: Any | None = None,
    ) -> None:
        self.token_file = (token_file or _default_token_file()).expanduser()
        self.env = env if env is not None else os.environ
        self.keyring = keyring_module if keyring_module is not None else _import_keyring()

    def resolve(self, *, include_token: bool = False) -> TokenResolution:
        env_token = str(self.env.get(HF_TOKEN_ENV) or "").strip()
        if env_token:
            return TokenResolution(True, "env", env_token if include_token else None)

        keyring_token = self._read_keyring_token()
        if keyring_token:
            return TokenResolution(True, "keyring", keyring_token if include_token else None)

        file_token = self._read_file_token()
        if file_token:
            return TokenResolution(True, "file", file_token if include_token else None)

        return TokenResolution(False, "none", None)

    def status(self, *, verify: bool = True) -> dict[str, Any]:
        resolution = self.resolve(include_token=verify)
        if not resolution.present:
            return {
                "present": False,
                "source": "none",
                "verified": False,
                "username": None,
                "status": "not_connected",
                "message": "No Hugging Face token is configured on this workstation.",
                "can_clear": False,
            }

        payload: dict[str, Any] = {
            "present": True,
            "source": resolution.source,
            "verified": False,
            "username": None,
            "status": "connected",
            "message": _source_message(resolution.source),
            "can_clear": resolution.source in {"keyring", "file"},
        }
        if not verify or not resolution.token:
            return payload

        verification = verify_huggingface_token(resolution.token)
        payload.update(
            {
                "verified": bool(verification["ok"]),
                "username": verification.get("username"),
                "status": "connected" if verification["ok"] else "needs_attention",
                "message": verification["message"],
            }
        )
        return payload

    def save(self, token: str) -> dict[str, Any]:
        cleaned = str(token or "").strip()
        if not cleaned:
            raise ValueError("Hugging Face token is required.")

        verification = verify_huggingface_token(cleaned)
        if not verification["ok"]:
            raise ValueError(str(verification["message"]))

        if self._write_keyring_token(cleaned):
            self._delete_file_token()
            return self.status(verify=True)

        self._write_file_token(cleaned)
        return self.status(verify=True)

    def clear(self) -> dict[str, Any]:
        resolution = self.resolve(include_token=False)
        self._delete_keyring_token()
        self._delete_file_token()
        if resolution.source == "env":
            status = self.status(verify=False)
            status["message"] = (
                "HF_TOKEN is set in the process environment and cannot be cleared from the dashboard."
            )
            status["can_clear"] = False
            return status
        return self.status(verify=False)

    def check_model(self, model_id: str) -> dict[str, Any]:
        model = str(model_id or "").strip()
        if not model:
            raise ValueError("model_id is required")
        resolution = self.resolve(include_token=True)
        return check_huggingface_model_access(model, token=resolution.token)

    def inject_env(self, env: MutableMapping[str, str]) -> MutableMapping[str, str]:
        if str(env.get(HF_TOKEN_ENV) or "").strip():
            return env
        token = self.resolve(include_token=True).token
        if token:
            env[HF_TOKEN_ENV] = token
        return env

    def _read_keyring_token(self) -> str | None:
        if self.keyring is None:
            return None
        try:
            token = self.keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except Exception:
            return None
        token = str(token or "").strip()
        return token or None

    def _write_keyring_token(self, token: str) -> bool:
        if self.keyring is None:
            return False
        try:
            self.keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, token)
            return True
        except Exception:
            return False

    def _delete_keyring_token(self) -> None:
        if self.keyring is None:
            return
        try:
            self.keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except Exception:
            return

    def _read_file_token(self) -> str | None:
        try:
            token = self.token_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return token or None

    def _write_file_token(self, token: str) -> None:
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        self.token_file.write_text(token.strip() + "\n", encoding="utf-8")
        try:
            os.chmod(self.token_file, 0o600)
        except OSError:
            pass

    def _delete_file_token(self) -> None:
        try:
            self.token_file.unlink()
        except FileNotFoundError:
            return
        except OSError:
            return


def _source_message(source: str) -> str:
    if source == "env":
        return "Using HF_TOKEN from the workstation environment."
    if source == "keyring":
        return "Using the workstation keychain credential."
    if source == "file":
        return "Using the workstation credential file fallback."
    return "No Hugging Face token is configured on this workstation."


def verify_huggingface_token(token: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi

        info = HfApi().whoami(token=token)
        username = None
        if isinstance(info, dict):
            username = info.get("name") or info.get("fullname") or info.get("email")
        return {
            "ok": True,
            "username": username,
            "message": f"Connected to Hugging Face{f' as {username}' if username else ''}.",
        }
    except Exception as exc:
        text = _safe_error_text(exc)
        if _looks_like_bad_token(text):
            message = "Hugging Face rejected this token. Check that it has read access."
        else:
            message = "Could not verify Hugging Face access right now. Check the token or network."
        return {"ok": False, "username": None, "message": message}


def check_huggingface_model_access(model_id: str, *, token: str | None) -> dict[str, Any]:
    model_url = f"https://huggingface.co/{model_id}"
    try:
        from huggingface_hub import HfApi

        HfApi().model_info(model_id, token=token)
        return {
            "model_id": model_id,
            "status": "available",
            "available": True,
            "message": "This model is reachable from the workstation.",
            "model_url": model_url,
            "license_url": model_url,
        }
    except Exception as exc:
        text = _safe_error_text(exc)
        status = _classify_model_error(text, has_token=bool(token))
        message = _model_access_message(status)
        return {
            "model_id": model_id,
            "status": status,
            "available": False,
            "message": message,
            "model_url": model_url,
            "license_url": model_url,
            "action": GATED_MODEL_ACTION if status in {"auth_required", "gated"} else None,
        }


def _classify_model_error(text: str, *, has_token: bool) -> str:
    lower = text.lower()
    if any(marker in lower for marker in ("network", "timed out", "connection", "name resolution")):
        return "network_error"
    if "404" in lower or "not found" in lower or "repository not found" in lower:
        if any(marker in lower for marker in ("private", "gated", "unauthorized", "401")):
            return "gated" if has_token else "auth_required"
        return "missing"
    if any(
        marker in lower
        for marker in ("gated repo", "restricted", "private repository", "please log in", "401", "unauthorized")
    ):
        return "gated" if has_token else "auth_required"
    return "network_error"


def _model_access_message(status: str) -> str:
    if status == "auth_required":
        return "This model requires Hugging Face access. Add a workstation token, then try again."
    if status == "gated":
        return "Hugging Face access is configured, but this model is still restricted. Accept the model license on Hugging Face or choose an open model."
    if status == "missing":
        return "Hugging Face could not find this model repository."
    return "Could not check this model right now. Check the network or try again."


def _looks_like_bad_token(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in ("401", "unauthorized", "invalid token", "forbidden"))


def _safe_error_text(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:1000]


def get_huggingface_access_manager() -> HuggingFaceAccessManager:
    return HuggingFaceAccessManager()


def resolve_huggingface_token() -> str | None:
    return get_huggingface_access_manager().resolve(include_token=True).token


def inject_huggingface_token(env: MutableMapping[str, str]) -> MutableMapping[str, str]:
    return get_huggingface_access_manager().inject_env(env)
