"""Experimental Apple Neural Accelerator opt-in validation."""

from __future__ import annotations

import logging
from typing import Any

from halo_forge.backend import BackendUnsupportedError, get_backend

_LOGGED_AVAILABLE: set[str] = set()


def validate_neural_accelerator_opt_in(
    config: Any,
    *,
    backend: Any | None = None,
    logger: logging.Logger | None = None,
    label: str = "trainer",
) -> None:
    enabled = bool(getattr(config, "enable_neural_accelerators", False))
    resolved = backend
    if resolved is None:
        try:
            resolved = get_backend()
        except Exception:
            resolved = None
    supports = bool(
        resolved is not None
        and getattr(getattr(resolved, "capabilities", None), "supports_neural_accelerators", False)
    )
    if enabled and not supports:
        name = getattr(resolved, "name", "unknown")
        raise BackendUnsupportedError(
            f"enable_neural_accelerators=True is unsupported on backend {name!r}."
        )
    if supports and not enabled:
        active_logger = logger or logging.getLogger("halo_forge.neural_accelerators")
        key = f"{getattr(resolved, 'name', 'unknown')}:{label}"
        if key not in _LOGGED_AVAILABLE:
            active_logger.info(
                "Neural Accelerators available — opt in via "
                "enable_neural_accelerators=True on your config (experimental)."
            )
            _LOGGED_AVAILABLE.add(key)
