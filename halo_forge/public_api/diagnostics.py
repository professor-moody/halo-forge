"""Diagnostics — surfaces orphan launches and the logs/ directory.

The /runs page only shows runs that produced a `training_summary.json`.
A run that aborts before that point (CUDA gate, OOM at model load, etc.)
is invisible to the existing surface, which makes Mac-on-MPS debugging
miserable. This module exposes:

- ``inventory_launches()`` — every directory containing
  `launch_context.json` across the standard training roots, classified
  by whether it has a sibling `training_summary.json` ("completed",
  surfaces in /runs) or not ("orphan", probably failed).

- ``inventory_logs()`` — entries in `logs/` with size + mtime, no
  contents (the UI lazy-tails what it actually wants to read).

- ``tail_log(path, n)`` — generic safe-tail. Refuses paths outside
  the repo root so the endpoint can't be abused as arbitrary file
  read.

The "logs/" location and the launch-summary roots mirror
``halo_forge.run_db.sync.DEFAULT_TRAINING_ROOTS`` plus the dashboard
app-data run root — keep them in sync if either moves.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# Same roots the run-db sync walks. If these change, change them in
# halo_forge.run_db.sync too.
LAUNCH_ROOTS: tuple[str, ...] = ("models", "outputs", "results")
LOGS_DIR_NAME = "logs"
LAUNCH_FILENAME = "launch_context.json"
SUMMARY_FILENAME = "training_summary.json"
DEFAULT_RUN_ROOT_ENV = "HALO_FORGE_RUN_ROOT"
DEFAULT_LOG_DIR_ENV = "HALO_FORGE_LOG_DIR"


def _default_run_root() -> Path:
    configured = str(os.environ.get(DEFAULT_RUN_ROOT_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".halo-forge" / "runs"


def _default_log_dir() -> Path:
    configured = str(os.environ.get(DEFAULT_LOG_DIR_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".halo-forge" / "logs"


def _launch_roots(base_path: Path) -> list[Path]:
    roots = [base_path / root_name for root_name in LAUNCH_ROOTS]
    app_run_root = _default_run_root()
    if not any(_same_resolved_path(root, app_run_root) for root in roots):
        roots.append(app_run_root)
    return roots


def _log_roots(base_path: Path) -> list[Path]:
    roots = [base_path / LOGS_DIR_NAME]
    app_log_root = _default_log_dir()
    if not any(_same_resolved_path(root, app_log_root) for root in roots):
        roots.append(app_log_root)
    return roots


def _same_resolved_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


@dataclass
class LaunchInventoryEntry:
    output_dir: str
    status: str  # "completed" | "orphan"
    has_summary: bool
    launched_at: Optional[str]
    command: Optional[list[str]]
    args: dict[str, Any] = field(default_factory=dict)
    log_files: list[str] = field(default_factory=list)
    summary_mtime: Optional[float] = None
    launch_mtime: Optional[float] = None


@dataclass
class LogFileEntry:
    name: str
    path: str
    size_bytes: int
    mtime: float


def _safe_resolve_under(root: Path, target: Path) -> Optional[Path]:
    """Return ``target`` resolved if it lies under ``root``; else ``None``.

    The diagnostics endpoint accepts a path query param. Without this
    check, a malicious caller could pass `../../etc/passwd`. The
    resolved-path comparison defeats both relative segments and symlink
    escapes.
    """
    try:
        resolved = target.resolve()
        resolved.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    return resolved


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Skipping unreadable %s: %s", path, exc)
        return None


def inventory_launches(base_path: Path) -> list[dict[str, Any]]:
    """Walk LAUNCH_ROOTS for every `launch_context.json` and classify.

    Newest-first by launch mtime so the most recent failed/orphan attempt
    surfaces at the top — which is what the user is debugging 90% of the
    time when something didn't show up in /runs.
    """
    base_path = base_path.resolve()
    entries: list[LaunchInventoryEntry] = []
    for root in _launch_roots(base_path):
        if not root.exists():
            continue
        for launch_path in root.rglob(LAUNCH_FILENAME):
            output_dir = launch_path.parent
            summary_path = output_dir / SUMMARY_FILENAME
            has_summary = summary_path.exists()
            launch_payload = _read_json(launch_path) or {}

            log_files: list[str] = []
            for log_path in sorted(output_dir.glob("*_training.log")):
                log_files.append(str(log_path))
            for log_path in sorted(output_dir.glob("*.log")):
                if str(log_path) not in log_files:
                    log_files.append(str(log_path))

            try:
                launch_mtime = launch_path.stat().st_mtime
            except OSError:
                launch_mtime = None
            summary_mtime = None
            if has_summary:
                try:
                    summary_mtime = summary_path.stat().st_mtime
                except OSError:
                    summary_mtime = None

            launched_at = None
            if launch_mtime is not None:
                launched_at = datetime.fromtimestamp(
                    launch_mtime, tz=timezone.utc
                ).isoformat()

            entries.append(LaunchInventoryEntry(
                output_dir=str(output_dir),
                status="completed" if has_summary else "orphan",
                has_summary=has_summary,
                launched_at=launched_at,
                command=launch_payload.get("command"),
                args=launch_payload.get("args") or {},
                log_files=log_files,
                summary_mtime=summary_mtime,
                launch_mtime=launch_mtime,
            ))

    entries.sort(
        key=lambda e: e.launch_mtime if e.launch_mtime is not None else 0.0,
        reverse=True,
    )
    return [e.__dict__ for e in entries]


def inventory_logs(base_path: Path) -> list[dict[str, Any]]:
    """List `logs/*.log` ordered newest-first."""
    items: list[LogFileEntry] = []
    for logs_dir in _log_roots(base_path):
        try:
            logs_dir = logs_dir.resolve()
        except OSError:
            continue
        if not logs_dir.is_dir():
            continue
        for path in logs_dir.glob("*.log"):
            try:
                stat = path.stat()
            except OSError:
                continue
            items.append(LogFileEntry(
                name=path.name,
                path=str(path),
                size_bytes=int(stat.st_size),
                mtime=float(stat.st_mtime),
            ))
    items.sort(key=lambda e: e.mtime, reverse=True)
    return [e.__dict__ for e in items]


def tail_log(
    *,
    base_path: Path,
    requested_path: str,
    tail: int = 200,
    max_bytes: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    """Return the last ``tail`` lines of a log file.

    The path must lie under ``base_path`` (any of LAUNCH_ROOTS or
    LOGS_DIR_NAME). Anything else returns ``available=False``.

    Reads at most ``max_bytes`` from the end of the file so huge
    training logs don't OOM the API process.
    """
    base = base_path.resolve()
    target = Path(requested_path)
    if not target.is_absolute():
        target = base / target
    allowed_roots = [base, _default_run_root(), _default_log_dir()]
    safe = next(
        (
            candidate
            for root in allowed_roots
            if (candidate := _safe_resolve_under(root, target)) is not None
        ),
        None,
    )
    if safe is None or not safe.is_file():
        return {
            "available": False,
            "lines": [],
            "reason": "Log not found under project root.",
            "path": str(target),
            "tail": int(tail),
        }

    # Enforce that we only tail logs under LOGS_DIR_NAME or LAUNCH_ROOTS
    # — refuse arbitrary repo files even though the safe-resolve check
    # already prevents traversal escape.
    permitted = False
    try:
        rel = safe.relative_to(base)
        head = rel.parts[0] if rel.parts else ""
        permitted = head in (*LAUNCH_ROOTS, LOGS_DIR_NAME)
    except ValueError:
        permitted = any(
            _safe_resolve_under(root, safe) is not None
            for root in (_default_run_root(), _default_log_dir())
        )
    if not permitted:
        return {
            "available": False,
            "lines": [],
            "reason": "Log path outside permitted roots.",
            "path": str(safe),
            "tail": int(tail),
        }

    try:
        size = safe.stat().st_size
        with safe.open("rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes, os.SEEK_SET)
                # Drop the (likely partial) first line.
                f.readline()
            buf = f.read()
    except OSError as exc:
        return {
            "available": False,
            "lines": [],
            "reason": f"Read failed: {exc}",
            "path": str(safe),
            "tail": int(tail),
        }

    text = buf.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if tail and tail > 0:
        lines = lines[-tail:]
    return {
        "available": True,
        "lines": lines,
        "reason": None,
        "path": str(safe),
        "tail": int(tail),
        "truncated_head": size > max_bytes,
        "size_bytes": size,
    }


def summary(base_path: Path) -> dict[str, Any]:
    """Top-level diagnostics overview the UI lands on first.

    Cheap aggregate — counts only, no log contents. The detail panels
    fetch the full inventories on demand.
    """
    launches = inventory_launches(base_path)
    logs = inventory_logs(base_path)
    orphan = sum(1 for e in launches if e["status"] == "orphan")
    completed = sum(1 for e in launches if e["status"] == "completed")
    most_recent_orphan = next(
        (e for e in launches if e["status"] == "orphan"), None
    )
    return {
        "base_path": str(base_path.resolve()),
        "launches": {
            "total": len(launches),
            "orphan": orphan,
            "completed": completed,
            "most_recent_orphan": most_recent_orphan,
        },
        "logs": {
            "total": len(logs),
            "newest": logs[0] if logs else None,
        },
    }


__all__ = [
    "LAUNCH_ROOTS",
    "LOGS_DIR_NAME",
    "inventory_launches",
    "inventory_logs",
    "tail_log",
    "summary",
]
