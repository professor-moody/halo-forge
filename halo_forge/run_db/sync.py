"""Filesystem → run database synchronization.

Walks the canonical ``training_summary.json`` files under ``models/``
``outputs/`` ``results/`` and upserts their headline columns into the
SQLite index. Skips files whose mtime hasn't changed since the last
sync — the index is incremental, not a full rebuild every call.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from halo_forge.run_db.db import RunDatabase, RunRecord

logger = logging.getLogger(__name__)


# Same default roots as `ui.services.results_service.ResultsService.TRAINING_DIRS`.
# Keep these mirrored — if those move, this needs to follow.
DEFAULT_TRAINING_ROOTS: tuple[Path, ...] = (
    Path("models"),
    Path("outputs"),
    Path("results"),
)

SUMMARY_FILENAME = "training_summary.json"


def iter_summary_files(roots: Iterable[Path]) -> Iterator[Path]:
    """Yield every ``training_summary.json`` under any of ``roots``.

    Quietly skips roots that don't exist — first-time installs may not
    have all of them yet.
    """
    for root in roots:
        if not root.exists():
            continue
        # rglob is fine here; the run trees are wide but shallow (5-10k
        # files at the high end). For 100k+ runs we'd want an index
        # over the directory listing, but that's a future-future concern.
        yield from root.rglob(SUMMARY_FILENAME)


def _read_summary(path: Path) -> Optional[dict[str, Any]]:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping unreadable summary %s: %s", path, exc)
        return None


def _summary_to_record(*, summary: dict[str, Any], path: Path, mtime: float) -> RunRecord:
    """Project a training_summary.json payload to a RunRecord.

    Tolerates older / partial summaries — every field is best-effort.
    The full payload survives in `raw_json` so the detail page can read
    anything we didn't extract here.
    """
    run_id = (
        summary.get("run_id")
        or summary.get("id")
        or path.parent.name  # stable filesystem identity for legacy summaries
    )
    fs_id = summary.get("id") or path.parent.name
    metrics = summary.get("metrics_summary") or {}
    user_summary = summary.get("user_summary") or {}
    effectiveness = summary.get("effectiveness") or {}

    return RunRecord(
        run_id=str(run_id),
        fs_id=str(fs_id),
        modality=str(summary.get("modality") or "unknown"),
        model_name=str(summary.get("model_name") or ""),
        base_model_name=summary.get("base_model_name"),
        active_model_name=summary.get("active_model_name"),
        status=str(summary.get("status") or "unknown"),
        timestamp=str(summary.get("timestamp") or ""),
        output_dir=str(path.parent),
        cycles_executed=int(summary.get("cycles_executed") or 0),
        total_train_steps=int(
            summary.get("total_train_steps_executed")
            or metrics.get("update_steps")
            or 0
        ),
        final_train_loss=_coerce_float(
            metrics.get("final_train_loss") or summary.get("final_train_loss")
        ),
        weights_updated=bool(summary.get("weights_updated")),
        final_update_reason=summary.get("final_update_reason"),
        failure_reason=summary.get("failure_reason"),
        effectiveness_verdict=(
            effectiveness.get("verdict")
            or summary.get("effectiveness_verdict")
            or user_summary.get("confidence_tone")
        ),
        quality_status=summary.get("quality_status"),
        keep_rate=_coerce_float(
            metrics.get("keep_rate") or summary.get("keep_rate")
        ),
        dominant_rejection_reason=summary.get("dominant_rejection_reason"),
        final_model_path=summary.get("final_model_path"),
        seed=_coerce_int(summary.get("seed")),
        raw_json=json.dumps(summary, default=str),
        source_mtime=mtime,
        indexed_at=datetime.now(timezone.utc).isoformat(),
    )


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sync_from_filesystem(
    db: RunDatabase,
    *,
    roots: Optional[Iterable[Path]] = None,
    force: bool = False,
) -> int:
    """Upsert every readable ``training_summary.json`` into ``db``.

    Args:
        db: Open `RunDatabase`.
        roots: Override the directory set to walk. Defaults to
            `DEFAULT_TRAINING_ROOTS`.
        force: If True, re-upsert every file even if its mtime hasn't
            changed. Useful after a schema bump.

    Returns:
        Number of rows upserted (skipped rows are not counted).
    """
    walked_roots = tuple(roots) if roots is not None else DEFAULT_TRAINING_ROOTS
    upserted = 0
    for path in iter_summary_files(walked_roots):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue

        if not force:
            # Cheap skip: if our row's source_mtime matches, nothing to do.
            cur = db._conn.execute(  # noqa: SLF001 — internal helper, intentional
                "SELECT source_mtime FROM runs WHERE output_dir = ?",
                (str(path.parent),),
            )
            row = cur.fetchone()
            if row and row["source_mtime"] is not None and row["source_mtime"] >= mtime:
                continue

        summary = _read_summary(path)
        if summary is None:
            continue

        record = _summary_to_record(summary=summary, path=path, mtime=mtime)
        db.upsert_run(record)
        upserted += 1

    return upserted


__all__ = [
    "DEFAULT_TRAINING_ROOTS",
    "iter_summary_files",
    "sync_from_filesystem",
]
