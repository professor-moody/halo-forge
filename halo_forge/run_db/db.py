"""SQLite-backed run database.

`RunDatabase` is the connection wrapper + query API. Everything outside
this module talks through `RunRecord` / `RunFilter` shapes; the SQL
stays here.

The default location is ``~/.halo-forge/runs.db``. Override with
``HALOFORGE_RUN_DB_PATH`` for tests or non-default homes. Tests should
prefer creating a `RunDatabase(":memory:")` directly.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

from halo_forge.run_db.schema import SCHEMA_SQL, initial_meta_rows


# Columns we expose on the row-level dataclass. These mirror the
# SQLite schema; everything else lives behind ``raw_json``.
_RUN_COLUMNS = (
    "run_id",
    "fs_id",
    "modality",
    "model_name",
    "base_model_name",
    "active_model_name",
    "status",
    "timestamp",
    "output_dir",
    "cycles_executed",
    "total_train_steps",
    "final_train_loss",
    "weights_updated",
    "final_update_reason",
    "failure_reason",
    "effectiveness_verdict",
    "quality_status",
    "keep_rate",
    "dominant_rejection_reason",
    "final_model_path",
    "seed",
    "raw_json",
    "source_mtime",
    "indexed_at",
)


@dataclass
class RunRecord:
    """Headline columns + raw JSON for a single run.

    Mirrors the SQLite schema exactly. ``raw`` decodes ``raw_json``
    on access for the convenience of API/UI consumers; the column
    itself stays as a string for round-trip stability.
    """

    run_id: str
    fs_id: Optional[str] = None
    modality: str = "unknown"
    model_name: str = ""
    base_model_name: Optional[str] = None
    active_model_name: Optional[str] = None
    status: str = "unknown"
    timestamp: Optional[str] = None
    output_dir: str = ""
    cycles_executed: int = 0
    total_train_steps: int = 0
    final_train_loss: Optional[float] = None
    weights_updated: bool = False
    final_update_reason: Optional[str] = None
    failure_reason: Optional[str] = None
    effectiveness_verdict: Optional[str] = None
    quality_status: Optional[str] = None
    keep_rate: Optional[float] = None
    dominant_rejection_reason: Optional[str] = None
    final_model_path: Optional[str] = None
    seed: Optional[int] = None
    raw_json: str = "{}"
    source_mtime: Optional[float] = None
    indexed_at: str = ""

    @property
    def raw(self) -> dict[str, Any]:
        try:
            return json.loads(self.raw_json or "{}")
        except json.JSONDecodeError:
            return {}

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["weights_updated"] = bool(d["weights_updated"])
        return d


@dataclass
class RunFilter:
    """Common filter / sort options for `RunDatabase.list_runs`.

    None means "any" for every field; ``modalities=[]`` means "exclude
    everything", which is rarely what callers want — the public API
    treats empty lists as "no filter applied" via convention.
    """

    modalities: Optional[List[str]] = None
    statuses: Optional[List[str]] = None
    model_substring: Optional[str] = None
    since_iso: Optional[str] = None  # timestamp >= since_iso
    until_iso: Optional[str] = None  # timestamp <= until_iso
    has_eval: Optional[bool] = None  # final_train_loss IS NOT NULL when True
    weights_updated: Optional[bool] = None
    sort_by: str = "timestamp"  # timestamp / cycles_executed / final_train_loss
    sort_dir: str = "desc"  # asc / desc
    limit: Optional[int] = None
    offset: int = 0


def _default_db_path() -> Path:
    override = os.environ.get("HALOFORGE_RUN_DB_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".halo-forge" / "runs.db"


_GLOBAL_DB: dict[str, "RunDatabase"] = {}
_GLOBAL_DB_LOCK = threading.Lock()


def get_database(path: Optional[str] = None) -> "RunDatabase":
    """Return a process-wide `RunDatabase` for the given path.

    Cached because each `RunDatabase` opens a SQLite connection; we
    don't want one per request. Test suites pass `":memory:"` to get
    a fresh isolated database.
    """
    resolved = str(_default_db_path() if path is None else Path(path).expanduser())
    with _GLOBAL_DB_LOCK:
        if resolved not in _GLOBAL_DB:
            _GLOBAL_DB[resolved] = RunDatabase(resolved)
        return _GLOBAL_DB[resolved]


class RunDatabase:
    """SQLite wrapper exposing the run-list / run-detail / filter API."""

    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False so the FastAPI worker can share the
        # connection. We serialize writes through `_lock`; SQLite is
        # serialized internally on a single connection anyway.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    # ----- internals --------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.executemany(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                initial_meta_rows(),
            )
            self._conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
        kwargs = {c: row[c] for c in _RUN_COLUMNS}
        # SQLite has no native bool — re-cast on the way back out so
        # callers don't have to guess whether they're getting 1 or True.
        kwargs["weights_updated"] = bool(kwargs["weights_updated"])
        return RunRecord(**kwargs)

    # ----- writes -----------------------------------------------------------

    def upsert_run(self, record: RunRecord) -> None:
        """Insert or replace a row keyed by ``run_id``."""
        if not record.run_id:
            raise ValueError("RunRecord.run_id is required")
        if not record.indexed_at:
            record.indexed_at = datetime.now(timezone.utc).isoformat()
        # Coerce booleans to 0/1; SQLite has no real bool type.
        weights_updated_int = 1 if record.weights_updated else 0
        with self._lock:
            self._conn.execute(
                f"""
                INSERT INTO runs ({", ".join(_RUN_COLUMNS)})
                VALUES ({", ".join(["?"] * len(_RUN_COLUMNS))})
                ON CONFLICT(run_id) DO UPDATE SET
                    {", ".join(f"{c}=excluded.{c}" for c in _RUN_COLUMNS if c != "run_id")}
                """,
                (
                    record.run_id,
                    record.fs_id,
                    record.modality,
                    record.model_name,
                    record.base_model_name,
                    record.active_model_name,
                    record.status,
                    record.timestamp,
                    record.output_dir,
                    record.cycles_executed,
                    record.total_train_steps,
                    record.final_train_loss,
                    weights_updated_int,
                    record.final_update_reason,
                    record.failure_reason,
                    record.effectiveness_verdict,
                    record.quality_status,
                    record.keep_rate,
                    record.dominant_rejection_reason,
                    record.final_model_path,
                    record.seed,
                    record.raw_json,
                    record.source_mtime,
                    record.indexed_at,
                ),
            )
            self._conn.commit()

    def upsert_many(self, records: Iterable[RunRecord]) -> int:
        count = 0
        for r in records:
            self.upsert_run(r)
            count += 1
        return count

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ----- reads ------------------------------------------------------------

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        cur = self._conn.execute(
            f"SELECT {', '.join(_RUN_COLUMNS)} FROM runs WHERE run_id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def get_run_by_fs_id(self, fs_id: str) -> Optional[RunRecord]:
        cur = self._conn.execute(
            f"SELECT {', '.join(_RUN_COLUMNS)} FROM runs WHERE fs_id = ?",
            (fs_id,),
        )
        row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def list_runs(self, filters: Optional[RunFilter] = None) -> List[RunRecord]:
        f = filters or RunFilter()
        clauses: list[str] = []
        params: list[Any] = []
        if f.modalities:
            clauses.append(f"modality IN ({', '.join('?' for _ in f.modalities)})")
            params.extend(f.modalities)
        if f.statuses:
            clauses.append(f"status IN ({', '.join('?' for _ in f.statuses)})")
            params.extend(f.statuses)
        if f.model_substring:
            clauses.append("model_name LIKE ?")
            params.append(f"%{f.model_substring}%")
        if f.since_iso:
            clauses.append("timestamp >= ?")
            params.append(f.since_iso)
        if f.until_iso:
            clauses.append("timestamp <= ?")
            params.append(f.until_iso)
        if f.has_eval is True:
            clauses.append("final_train_loss IS NOT NULL")
        elif f.has_eval is False:
            clauses.append("final_train_loss IS NULL")
        if f.weights_updated is True:
            clauses.append("weights_updated = 1")
        elif f.weights_updated is False:
            clauses.append("weights_updated = 0")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        # Whitelist sort columns to keep the surface SQL-injection-safe.
        sort_col = {
            "timestamp": "timestamp",
            "cycles_executed": "cycles_executed",
            "final_train_loss": "final_train_loss",
            "model_name": "model_name",
        }.get(f.sort_by, "timestamp")
        sort_dir = "ASC" if f.sort_dir.lower() == "asc" else "DESC"
        limit_clause = ""
        if f.limit is not None:
            limit_clause = f" LIMIT {int(f.limit)} OFFSET {int(max(0, f.offset))}"

        sql = (
            f"SELECT {', '.join(_RUN_COLUMNS)} "
            f"FROM runs {where} ORDER BY {sort_col} {sort_dir}{limit_clause}"
        )
        cur = self._conn.execute(sql, params)
        return [self._row_to_record(row) for row in cur.fetchall()]

    def count_runs(self, filters: Optional[RunFilter] = None) -> int:
        # We re-run the WHERE assembly so callers can pass the exact
        # same filter to count_runs as to list_runs. Slightly DRY-violating
        # but the query is small and the alternative is forking the SQL.
        f = filters or RunFilter()
        clauses: list[str] = []
        params: list[Any] = []
        if f.modalities:
            clauses.append(f"modality IN ({', '.join('?' for _ in f.modalities)})")
            params.extend(f.modalities)
        if f.statuses:
            clauses.append(f"status IN ({', '.join('?' for _ in f.statuses)})")
            params.extend(f.statuses)
        if f.model_substring:
            clauses.append("model_name LIKE ?")
            params.append(f"%{f.model_substring}%")
        if f.since_iso:
            clauses.append("timestamp >= ?")
            params.append(f.since_iso)
        if f.until_iso:
            clauses.append("timestamp <= ?")
            params.append(f.until_iso)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = self._conn.execute(f"SELECT COUNT(*) FROM runs {where}", params)
        return int(cur.fetchone()[0])

    def distinct_modalities(self) -> List[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT modality FROM runs ORDER BY modality"
        )
        return [row["modality"] for row in cur.fetchall()]

    def distinct_models(self) -> List[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT model_name FROM runs ORDER BY model_name"
        )
        return [row["model_name"] for row in cur.fetchall() if row["model_name"]]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = [
    "RunDatabase",
    "RunFilter",
    "RunRecord",
    "get_database",
]
