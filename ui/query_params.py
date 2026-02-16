"""Safe query-parameter helpers for NiceGUI page components."""

from __future__ import annotations

from typing import Dict, List

from nicegui import ui


def get_query_params() -> Dict[str, str]:
    """Return current page query parameters as a plain string dict."""
    try:
        client = ui.context.client
        request = getattr(client, "request", None)
        query_params = getattr(request, "query_params", None)
        if query_params is None:
            return {}
        return {str(key): str(value) for key, value in query_params.items()}
    except Exception:
        return {}


def get_query_param(name: str, default: str = "") -> str:
    """Return a single query parameter value."""
    value = get_query_params().get(name, default)
    return str(value).strip()


def get_query_csv(name: str) -> List[str]:
    """Return a comma-separated query parameter as a normalized list."""
    raw = get_query_param(name, "")
    if not raw:
        return []
    return [piece.strip() for piece in raw.split(",") if piece.strip()]

