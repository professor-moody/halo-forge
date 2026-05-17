"""Central product/version metadata for Halo Forge."""

from __future__ import annotations

import os
from typing import Any

PACKAGE_VERSION = "2.0.0a1"
DISPLAY_VERSION = "2.0.0-alpha-1"
RELEASE_CHANNEL = "alpha"


def version_info() -> dict[str, Any]:
    """Return sanitized version metadata for CLI, API, and desktop surfaces."""

    git_sha = (
        os.environ.get("HALO_FORGE_GIT_SHA")
        or os.environ.get("GITHUB_SHA")
        or os.environ.get("TAURI_GIT_SHA")
    )
    payload: dict[str, Any] = {
        "package_version": PACKAGE_VERSION,
        "display_version": DISPLAY_VERSION,
        "release_channel": RELEASE_CHANNEL,
    }
    if git_sha:
        payload["git_sha"] = git_sha[:12]
    return payload


def version_line() -> str:
    """Human-readable version line for command-line output."""

    return (
        f"halo-forge {DISPLAY_VERSION} "
        f"(package {PACKAGE_VERSION}, channel {RELEASE_CHANNEL})"
    )
