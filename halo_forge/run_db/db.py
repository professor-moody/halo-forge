"""SQLite-backed run database.

`RunDatabase` is the connection wrapper + query API. Everything outside
this module talks through `RunRecord` / `RunFilter` shapes; the SQL
stays here.

The default location is ``~/.halo-forge/runs.db``. Override with
``HALOFORGE_RUN_DB_PATH`` for tests or non-default homes. Tests should
prefer creating a `RunDatabase(":memory:")` directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

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


def _content_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    # True only when a completed persistent evaluation targets this run.
    has_eval: Optional[bool] = None
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


class _SerializedCursor:
    """Serialize SQLite cursor calls made from API and worker threads.

    ``check_same_thread=False`` permits a connection to move across threads, but
    it does not make concurrent cursor calls on that connection safe. The
    dashboard deliberately shares one catalog between request handlers and the
    supervised worker, so every SQLite C API call must use the database lock.
    """

    def __init__(self, cursor: sqlite3.Cursor, lock: threading.RLock):
        self._cursor = cursor
        self._lock = lock

    def fetchone(self) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._cursor.fetchone()

    def fetchmany(self, size: Optional[int] = None) -> list[sqlite3.Row]:
        with self._lock:
            if size is None:
                return self._cursor.fetchmany()
            return self._cursor.fetchmany(size)

    def fetchall(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._cursor.fetchall()

    def __iter__(self) -> Iterator[sqlite3.Row]:
        # Materialize while holding the lock so another thread cannot enter the
        # shared connection midway through cursor iteration.
        with self._lock:
            rows = list(self._cursor)
        return iter(rows)

    def __getattr__(self, name: str) -> Any:
        with self._lock:
            return getattr(self._cursor, name)


class _SerializedConnection:
    """Small locked facade over the SQLite methods used by ``RunDatabase``."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.RLock):
        self._connection = connection
        self._lock = lock

    def execute(self, *args: Any, **kwargs: Any) -> _SerializedCursor:
        with self._lock:
            return _SerializedCursor(
                self._connection.execute(*args, **kwargs),
                self._lock,
            )

    def executemany(self, *args: Any, **kwargs: Any) -> _SerializedCursor:
        with self._lock:
            return _SerializedCursor(
                self._connection.executemany(*args, **kwargs),
                self._lock,
            )

    def executescript(self, *args: Any, **kwargs: Any) -> _SerializedCursor:
        with self._lock:
            return _SerializedCursor(
                self._connection.executescript(*args, **kwargs),
                self._lock,
            )

    def commit(self) -> None:
        with self._lock:
            self._connection.commit()

    def rollback(self) -> None:
        with self._lock:
            self._connection.rollback()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "_SerializedConnection":
        """Preserve sqlite transaction contexts while holding serialization.

        A few catalog publication paths intentionally use ``with self._conn``
        so a multi-statement delete/insert or seal operation commits
        atomically.  The facade must retain that behavior instead of exposing
        the underlying connection without its process-local lock.
        """

        self._lock.acquire()
        try:
            self._connection.__enter__()
        except Exception:
            self._lock.release()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        try:
            return self._connection.__exit__(exc_type, exc, traceback)
        finally:
            self._lock.release()


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
        # FastAPI request handlers and the supervised worker share this catalog.
        # ``check_same_thread=False`` permits that topology, while the serialized
        # facade prevents simultaneous SQLite C API calls on one connection.
        self._lock = threading.RLock()
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        self._conn = _SerializedConnection(connection, self._lock)
        self._init_schema()

    # ----- internals --------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(SCHEMA_SQL)
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.executemany(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES (?, ?)",
                initial_meta_rows(),
            )
            current_version = self._conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
            try:
                on_disk_version = int(current_version["value"]) if current_version else 0
            except (TypeError, ValueError):
                on_disk_version = 0
            from halo_forge.run_db.schema import SCHEMA_VERSION

            if on_disk_version < SCHEMA_VERSION:
                self._migrate_schema(on_disk_version)
                self._conn.execute(
                    "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
                    (str(SCHEMA_VERSION),),
                )
            # These indexes reference columns introduced by the v7 in-place
            # migration. Creating them after migration keeps SCHEMA_SQL safe
            # for databases whose older ``work_items`` table already exists.
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_work_items_domain "
                "ON work_items (domain_kind, domain_id, status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_work_items_group_v4 "
                "ON work_items (run_group_id, status, created_at)"
            )
            # Worker owners perform orphan recovery. A read-only CLI or a
            # second app connection must never interrupt another live process.
            self._conn.commit()

    def _migrate_schema(self, on_disk_version: int) -> None:
        """Apply the small in-place migrations not expressible by CREATE IF NOT EXISTS.

        Schema v4 changes the primary key of ``run_datasets`` so that one run
        can bind the same version/split in distinct roles. SQLite cannot alter
        a primary key, therefore the table is rebuilt while preserving every
        v3 row as a ``train`` binding. Schema v5 removes the child-run foreign
        key from ``run_lineage`` because managed launches allocate and attach
        their canonical id before the asynchronous run is indexed.
        """
        if on_disk_version < 4:
            info = self._conn.execute("PRAGMA table_info(run_datasets)").fetchall()
            if info:
                columns = {str(row["name"]) for row in info}
                pk_columns = [
                    str(row["name"])
                    for row in sorted(info, key=lambda value: int(value["pk"]))
                    if int(row["pk"]) > 0
                ]
                expected_pk = ["run_id", "dataset_version_id", "split", "role"]
                if (
                    "role" not in columns
                    or "training_artifact_id" not in columns
                    or pk_columns != expected_pk
                ):
                    self._conn.execute("ALTER TABLE run_datasets RENAME TO run_datasets_v3")
                    self._conn.execute("""
                        CREATE TABLE run_datasets (
                            run_id TEXT NOT NULL,
                            dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(id),
                            role TEXT NOT NULL DEFAULT 'train',
                            split TEXT NOT NULL DEFAULT 'train',
                            training_artifact_id TEXT REFERENCES training_artifacts(id)
                                ON DELETE SET NULL,
                            attached_at TEXT NOT NULL,
                            PRIMARY KEY (run_id, dataset_version_id, split, role)
                        )
                        """)
                    role_expr = "role" if "role" in columns else "'train'"
                    artifact_expr = (
                        "training_artifact_id" if "training_artifact_id" in columns else "NULL"
                    )
                    self._conn.execute(f"""
                        INSERT INTO run_datasets
                            (run_id, dataset_version_id, role, split,
                             training_artifact_id, attached_at)
                        SELECT run_id, dataset_version_id, {role_expr}, split,
                               {artifact_expr}, attached_at
                        FROM run_datasets_v3
                        """)
                    self._conn.execute("DROP TABLE run_datasets_v3")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_run_datasets_version "
                "ON run_datasets (dataset_version_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_run_datasets_role " "ON run_datasets (run_id, role)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_run_datasets_artifact "
                "ON run_datasets (training_artifact_id)"
            )

        if on_disk_version < 5:
            foreign_keys = self._conn.execute("PRAGMA foreign_key_list(run_lineage)").fetchall()
            child_has_fk = any(str(row["from"]) == "child_run_id" for row in foreign_keys)
            if child_has_fk:
                self._conn.execute("ALTER TABLE run_lineage RENAME TO run_lineage_v4")
                self._conn.execute("""
                    CREATE TABLE run_lineage (
                        child_run_id TEXT NOT NULL,
                        parent_run_id TEXT NOT NULL,
                        forked_at_cycle INTEGER,
                        notes TEXT,
                        PRIMARY KEY (child_run_id, parent_run_id)
                    )
                    """)
                self._conn.execute("""
                    INSERT INTO run_lineage
                        (child_run_id, parent_run_id, forked_at_cycle, notes)
                    SELECT child_run_id, parent_run_id, forked_at_cycle, notes
                    FROM run_lineage_v4
                    """)
                self._conn.execute("DROP TABLE run_lineage_v4")

        if on_disk_version < 6:
            suite_columns = {
                str(row["name"])
                for row in self._conn.execute("PRAGMA table_info(benchmark_suites)").fetchall()
            }
            if suite_columns and "purpose" not in suite_columns:
                self._conn.execute(
                    "ALTER TABLE benchmark_suites ADD COLUMN purpose TEXT "
                    "NOT NULL DEFAULT 'unspecified'"
                )

            sample_columns = {
                str(row["name"])
                for row in self._conn.execute("PRAGMA table_info(evaluation_samples)").fetchall()
            }
            sample_additions = {
                "generation_seed": "INTEGER",
                "evidence_kind": "TEXT NOT NULL DEFAULT 'legacy'",
                "valid": "INTEGER NOT NULL DEFAULT 0",
                "mineable": "INTEGER NOT NULL DEFAULT 0",
                "input_tokens": "INTEGER",
                "output_tokens": "INTEGER",
                "finish_reason": "TEXT",
                "template_hash": "TEXT",
                "runtime_versions_json": "TEXT NOT NULL DEFAULT '{}'",
                "score_direction": "TEXT",
                "score_threshold": "REAL",
                "coverage": "REAL",
            }
            for column, declaration in sample_additions.items():
                if sample_columns and column not in sample_columns:
                    self._conn.execute(
                        f"ALTER TABLE evaluation_samples ADD COLUMN {column} {declaration}"
                    )

        if on_disk_version < 7:
            additions = {
                "dataset_jobs": {"work_item_id": "TEXT"},
                "evaluations": {"work_item_id": "TEXT"},
                "benchmark_suites": {"purpose_v4": "TEXT"},
                "work_items": {
                    "resource_requirements_json": "TEXT NOT NULL DEFAULT '{}'",
                    "domain_kind": "TEXT",
                    "domain_id": "TEXT",
                    "run_group_id": "TEXT",
                },
                "resource_leases": {
                    "holder_pid": "INTEGER",
                    "holder_pid_started_at": "REAL",
                },
            }
            for table, columns in additions.items():
                existing = {
                    str(row["name"])
                    for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
                for column, declaration in columns.items():
                    if existing and column not in existing:
                        self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_work_items_domain "
                "ON work_items (domain_kind, domain_id, status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_work_items_group_v4 "
                "ON work_items (run_group_id, status, created_at)"
            )

            # Preserve every legacy artifact occurrence while deduplicating its
            # content identity. Run paths remain referenced rather than copied.
            self._conn.execute("""
                INSERT OR IGNORE INTO artifact_blobs
                    (id, content_hash, artifact_type, format, size_bytes,
                     integrity_state, manifest_json, created_at)
                SELECT 'blob-' || artifact_hash, artifact_hash,
                       artifact_kind, format, size_bytes, verification_status,
                       metadata_json, created_at
                FROM model_artifacts
                """)
            self._conn.execute("""
                INSERT OR IGNORE INTO artifact_locations
                    (id, blob_id, path, storage_mode, state, size_bytes,
                     metadata_json, created_at, last_verified_at)
                SELECT 'location-' || id,
                       'blob-' || artifact_hash, path,
                       'referenced', 'available', size_bytes, '{}', created_at,
                       CASE WHEN verification_status = 'verified' THEN created_at ELSE NULL END
                FROM model_artifacts
                """)
            self._conn.execute("""
                INSERT OR IGNORE INTO artifact_occurrences
                    (id, blob_id, artifact_kind, legacy_model_artifact_id,
                     run_id, run_group_id, trial_id, trial_segment_id, model_id,
                     tokenizer_revision, chat_template_hash, backend, metadata_json,
                     created_at)
                SELECT 'occurrence-' || id,
                       'blob-' || artifact_hash, artifact_kind, id,
                       run_id, run_group_id, trial_id, trial_segment_id, model_id,
                       tokenizer_revision, chat_template_hash, backend, metadata_json,
                       created_at
                FROM model_artifacts
                """)
            self._conn.execute("""
                INSERT OR IGNORE INTO work_item_attempts
                    (id, work_item_id, ordinal, status, worker_id, worker_pid,
                     worker_pid_started_at, claim_token, output_dir, result_json,
                     error, created_at, started_at, completed_at)
                SELECT 'attempt-' || id || '-1', id, 1, status, NULL, worker_pid,
                       worker_pid_started_at, claim_token, NULL, result_json,
                       error, created_at, started_at, completed_at
                FROM work_items
                """)

        if on_disk_version < 8:
            group_columns = {
                str(row["name"])
                for row in self._conn.execute("PRAGMA table_info(run_groups)").fetchall()
            }
            group_additions = {
                "checkpoint_policy_revision_id": (
                    "TEXT REFERENCES checkpoint_policy_revisions(id)"
                ),
                "resolved_checkpoint_plan_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for column, declaration in group_additions.items():
                if group_columns and column not in group_columns:
                    self._conn.execute(f"ALTER TABLE run_groups ADD COLUMN {column} {declaration}")

            segment_sql = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'trial_segments'"
            ).fetchone()
            segment_definition = str(segment_sql["sql"] or "") if segment_sql else ""
            if "'pause'" not in segment_definition or "'stop'" not in segment_definition:
                # Keep dependent foreign-key declarations pointed at the
                # canonical table name while rebuilding SQLite's CHECK. Foreign
                # keys must be disabled outside a transaction; otherwise SQLite
                # rewrites dependent tables to reference the temporary name.
                self._conn.commit()
                self._conn.execute("PRAGMA foreign_keys = OFF")
                self._conn.execute("PRAGMA legacy_alter_table = ON")
                try:
                    self._conn.execute("ALTER TABLE trial_segments RENAME TO trial_segments_v7")
                    self._conn.execute("""
                        CREATE TABLE trial_segments (
                            id TEXT PRIMARY KEY,
                            trial_run_id TEXT NOT NULL
                                REFERENCES trial_runs(id) ON DELETE CASCADE,
                            ordinal INTEGER NOT NULL,
                            status TEXT NOT NULL DEFAULT 'queued',
                            unit TEXT NOT NULL,
                            start_value INTEGER NOT NULL,
                            end_value INTEGER NOT NULL,
                            work_item_id TEXT REFERENCES work_items(id) ON DELETE SET NULL,
                            checkpoint_artifact_id TEXT,
                            decision TEXT CHECK (
                                decision IS NULL OR decision IN (
                                    'continue', 'pause', 'stop', 'prune', 'complete'
                                )
                            ),
                            decision_reason TEXT,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            started_at TEXT,
                            completed_at TEXT,
                            UNIQUE (trial_run_id, ordinal),
                            CHECK (end_value > start_value)
                        )
                        """)
                    self._conn.execute("""
                        INSERT INTO trial_segments
                            (id, trial_run_id, ordinal, status, unit, start_value,
                             end_value, work_item_id, checkpoint_artifact_id,
                             decision, decision_reason, created_at, updated_at,
                             started_at, completed_at)
                        SELECT id, trial_run_id, ordinal, status, unit, start_value,
                               end_value, work_item_id, checkpoint_artifact_id,
                               decision, decision_reason, created_at, updated_at,
                               started_at, completed_at
                        FROM trial_segments_v7
                        """)
                    self._conn.execute("DROP TABLE trial_segments_v7")
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise
                finally:
                    self._conn.execute("PRAGMA legacy_alter_table = OFF")
                    self._conn.execute("PRAGMA foreign_keys = ON")
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trial_segments_run "
                    "ON trial_segments (trial_run_id, ordinal)"
                )

        if on_disk_version < 19:
            additions = {
                "training_outcome_assessments": {
                    "stage": "TEXT NOT NULL DEFAULT 'queued'",
                    "progress_json": "TEXT NOT NULL DEFAULT '{}'",
                    "request_json": "TEXT NOT NULL DEFAULT '{}'",
                    "resume_cursor_json": "TEXT NOT NULL DEFAULT '{}'",
                    "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
                },
                "adaptation_study_protocol_revisions": {
                    "launch_status": "TEXT NOT NULL DEFAULT 'not_started'",
                    "launch_progress_json": "TEXT NOT NULL DEFAULT '{}'",
                    "launch_work_item_id": "TEXT REFERENCES work_items(id) ON DELETE SET NULL",
                    "launch_error": "TEXT",
                },
                "adaptation_study_analyses": {
                    "stage": "TEXT NOT NULL DEFAULT 'queued'",
                    "progress_json": "TEXT NOT NULL DEFAULT '{}'",
                    "request_json": "TEXT NOT NULL DEFAULT '{}'",
                    "resume_cursor_json": "TEXT NOT NULL DEFAULT '{}'",
                    "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
                    "error": "TEXT",
                },
                "grounded_generation_batches": {
                    "progress_json": "TEXT NOT NULL DEFAULT '{}'",
                    "resume_cursor_json": "TEXT NOT NULL DEFAULT '{}'",
                    "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
                },
                "agent_episodes": {
                    "stage": "TEXT NOT NULL DEFAULT 'queued'",
                    "progress_json": "TEXT NOT NULL DEFAULT '{}'",
                    "request_json": "TEXT NOT NULL DEFAULT '{}'",
                    "resume_cursor_json": "TEXT NOT NULL DEFAULT '{}'",
                    "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
                    "parent_episode_id": (
                        "TEXT REFERENCES agent_episodes(id) ON DELETE SET NULL"
                    ),
                },
                "trajectory_sets": {
                    "status": "TEXT NOT NULL DEFAULT 'ready'",
                    "stage": "TEXT NOT NULL DEFAULT 'ready'",
                    "progress_json": "TEXT NOT NULL DEFAULT '{}'",
                    "request_json": "TEXT NOT NULL DEFAULT '{}'",
                    "cancel_requested": "INTEGER NOT NULL DEFAULT 0",
                    "work_item_id": "TEXT REFERENCES work_items(id) ON DELETE SET NULL",
                    "error": "TEXT",
                },
            }
            for table, columns in additions.items():
                existing = {
                    str(row["name"])
                    for row in self._conn.execute(
                        f"PRAGMA table_info({table})"
                    ).fetchall()
                }
                for column, declaration in columns.items():
                    if existing and column not in existing:
                        self._conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
                        )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_future_outcome_work "
                "ON training_outcome_assessments (work_item_id, status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_future_grounding_work "
                "ON grounded_generation_batches (work_item_id, status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_future_episode_work "
                "ON agent_episodes (work_item_id, status)"
            )

        if on_disk_version < 22:
            # V19/V20 add one nullable pointer to the existing immutable plan
            # table. Historical plans deliberately remain native/unbound.
            columns = {
                str(row["name"])
                for row in self._conn.execute(
                    "PRAGMA table_info(training_plan_revisions)"
                ).fetchall()
            }
            if columns and "runtime_profile_revision_id" not in columns:
                self._conn.execute(
                    "ALTER TABLE training_plan_revisions ADD COLUMN "
                    "runtime_profile_revision_id TEXT "
                    "REFERENCES managed_runtime_revisions(id)"
                )

        if on_disk_version < 23:
            # V21 binds a guided plan to the exact path revision and the real
            # certification that unlocked it. Historical plans stay readable
            # and deliberately carry no inferred certification evidence.
            columns = {
                str(row["name"])
                for row in self._conn.execute(
                    "PRAGMA table_info(training_plan_revisions)"
                ).fetchall()
            }
            additions = {
                "training_path_revision_id": (
                    "TEXT REFERENCES training_path_profile_revisions(id)"
                ),
                "training_path_certification_id": (
                    "TEXT REFERENCES training_path_certifications(id)"
                ),
            }
            for column, declaration in additions.items():
                if columns and column not in columns:
                    self._conn.execute(
                        f"ALTER TABLE training_plan_revisions ADD COLUMN {column} {declaration}"
                    )

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
            clauses.append(
                "EXISTS (SELECT 1 FROM evaluations e WHERE e.status = 'completed' "
                "AND e.subject_type IN ('run', 'final_model') "
                "AND e.subject_ref = runs.run_id)"
            )
        elif f.has_eval is False:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM evaluations e WHERE e.status = 'completed' "
                "AND e.subject_type IN ('run', 'final_model') "
                "AND e.subject_ref = runs.run_id)"
            )
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
        if f.has_eval is True:
            clauses.append(
                "EXISTS (SELECT 1 FROM evaluations e WHERE e.status = 'completed' "
                "AND e.subject_type IN ('run', 'final_model') "
                "AND e.subject_ref = runs.run_id)"
            )
        elif f.has_eval is False:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM evaluations e WHERE e.status = 'completed' "
                "AND e.subject_type IN ('run', 'final_model') "
                "AND e.subject_ref = runs.run_id)"
            )
        if f.weights_updated is True:
            clauses.append("weights_updated = 1")
        elif f.weights_updated is False:
            clauses.append("weights_updated = 0")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = self._conn.execute(f"SELECT COUNT(*) FROM runs {where}", params)
        return int(cur.fetchone()[0])

    def distinct_modalities(self) -> List[str]:
        cur = self._conn.execute("SELECT DISTINCT modality FROM runs ORDER BY modality")
        return [row["modality"] for row in cur.fetchall()]

    def modality_counts(self) -> dict[str, int]:
        """Count of indexed runs per modality.

        Used by the /runs filter UI so chips for kinds with rows show
        their tally and chips for kinds with zero runs render dim
        (rather than disappearing — discoverability for trainers the
        user hasn't tried yet)."""
        cur = self._conn.execute("SELECT modality, COUNT(*) AS c FROM runs GROUP BY modality")
        return {row["modality"]: int(row["c"]) for row in cur.fetchall()}

    def distinct_models(self) -> List[str]:
        cur = self._conn.execute("SELECT DISTINCT model_name FROM runs ORDER BY model_name")
        return [row["model_name"] for row in cur.fetchall() if row["model_name"]]

    # ----- Dataset Lab ------------------------------------------------------

    def create_dataset(
        self,
        *,
        name: str,
        modality: str,
        canonical_schema: str,
        description: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> "DatasetRecord":
        if not name or not name.strip():
            raise ValueError("dataset name is required")
        if not modality or not modality.strip():
            raise ValueError("dataset modality is required")
        if not canonical_schema or not canonical_schema.strip():
            raise ValueError("canonical_schema is required")
        now = datetime.now(timezone.utc).isoformat()
        identifier = dataset_id or uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO datasets
                    (id, name, description, modality, canonical_schema,
                     latest_version_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    identifier,
                    name.strip(),
                    description,
                    modality.strip().lower(),
                    canonical_schema.strip().lower(),
                    now,
                    now,
                ),
            )
            self._conn.commit()
        record = self.get_dataset(identifier)
        assert record is not None
        return record

    def get_dataset(self, dataset_id: str) -> Optional["DatasetRecord"]:
        row = self._conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        return _row_to_dataset(row) if row else None

    def list_datasets(
        self,
        *,
        modality: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List["DatasetRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        if modality:
            clauses.append("modality = ?")
            params.append(modality.lower())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        pagination = ""
        if limit is not None:
            pagination = " LIMIT ? OFFSET ?"
            params.extend([max(0, int(limit)), max(0, int(offset))])
        rows = self._conn.execute(
            f"SELECT * FROM datasets {where} ORDER BY updated_at DESC{pagination}",
            params,
        ).fetchall()
        return [_row_to_dataset(row) for row in rows]

    def update_dataset(
        self,
        dataset_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        latest_version_id: Optional[str] = None,
    ) -> Optional["DatasetRecord"]:
        existing = self.get_dataset(dataset_id)
        if existing is None:
            return None
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                UPDATE datasets SET name = ?, description = ?, latest_version_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    name.strip() if name is not None else existing.name,
                    description if description is not None else existing.description,
                    (
                        latest_version_id
                        if latest_version_id is not None
                        else existing.latest_version_id
                    ),
                    now,
                    dataset_id,
                ),
            )
            self._conn.commit()
        return self.get_dataset(dataset_id)

    def delete_dataset(self, dataset_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def create_dataset_source(
        self,
        *,
        dataset_id: str,
        kind: str,
        uri: str,
        fingerprint: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        revision: Optional[str] = None,
        size_bytes: Optional[int] = None,
        row_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        refreshed_from_source_id: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> "DatasetSourceRecord":
        if self.get_dataset(dataset_id) is None:
            raise ValueError(f"unknown dataset: {dataset_id}")
        if not uri or not str(uri).strip():
            raise ValueError("source uri is required")
        if not fingerprint:
            raise ValueError("source fingerprint is required")
        identifier = source_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO dataset_sources
                    (id, dataset_id, kind, uri, config, split, revision,
                     fingerprint, size_bytes, row_count, metadata_json,
                     refreshed_from_source_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    dataset_id,
                    kind,
                    str(uri),
                    config,
                    split,
                    revision,
                    fingerprint,
                    size_bytes,
                    row_count,
                    json.dumps(metadata or {}, sort_keys=True),
                    refreshed_from_source_id,
                    now,
                ),
            )
            self._conn.commit()
        record = self.get_dataset_source(identifier)
        assert record is not None
        return record

    def get_dataset_source(self, source_id: str) -> Optional["DatasetSourceRecord"]:
        row = self._conn.execute(
            "SELECT * FROM dataset_sources WHERE id = ?", (source_id,)
        ).fetchone()
        return _row_to_dataset_source(row) if row else None

    def list_dataset_sources(self, dataset_id: str) -> List["DatasetSourceRecord"]:
        rows = self._conn.execute(
            "SELECT * FROM dataset_sources WHERE dataset_id = ? ORDER BY created_at DESC",
            (dataset_id,),
        ).fetchall()
        return [_row_to_dataset_source(row) for row in rows]

    def create_dataset_version(
        self,
        *,
        dataset_id: str,
        recipe_hash: str,
        recipe: Dict[str, Any],
        storage_path: str,
        source_id: Optional[str] = None,
        parent_version_id: Optional[str] = None,
        parent_versions: Optional[Iterable[Any]] = None,
        status: str = "building",
        content_hash: Optional[str] = None,
        row_count: int = 0,
        size_bytes: int = 0,
        split_counts: Optional[Dict[str, int]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        source_fingerprints: Optional[Dict[str, str]] = None,
        assets_materialized: bool = False,
        error: Optional[str] = None,
        completed_at: Optional[str] = None,
        version_id: Optional[str] = None,
    ) -> "DatasetVersionRecord":
        if self.get_dataset(dataset_id) is None:
            raise ValueError(f"unknown dataset: {dataset_id}")
        identifier = version_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        if status == "completed" and completed_at is None:
            completed_at = now
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO dataset_versions
                        (id, dataset_id, source_id, parent_version_id, status,
                         content_hash, recipe_hash, recipe_json, storage_path,
                         row_count, size_bytes, split_counts_json, statistics_json,
                         provenance_json, source_fingerprints_json,
                         assets_materialized, error, created_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        dataset_id,
                        source_id,
                        parent_version_id,
                        status,
                        content_hash,
                        recipe_hash,
                        json.dumps(recipe, sort_keys=True),
                        storage_path,
                        int(row_count),
                        int(size_bytes),
                        json.dumps(split_counts or {}, sort_keys=True),
                        json.dumps(statistics or {}, sort_keys=True),
                        json.dumps(provenance or {}, sort_keys=True),
                        json.dumps(source_fingerprints or {}, sort_keys=True),
                        1 if assets_materialized else 0,
                        error,
                        now,
                        completed_at,
                    ),
                )
                if status == "completed":
                    self._conn.execute(
                        "UPDATE datasets SET latest_version_id = ?, updated_at = ? WHERE id = ?",
                        (identifier, now, dataset_id),
                    )
                self._conn.commit()
        except sqlite3.IntegrityError as exc:
            if content_hash:
                existing = self.find_dataset_version(
                    dataset_id=dataset_id,
                    content_hash=content_hash,
                    recipe_hash=recipe_hash,
                )
                if existing is not None:
                    return existing
            raise ValueError(f"could not create dataset version: {exc}") from exc
        record = self.get_dataset_version(identifier)
        assert record is not None
        parent_values = list(parent_versions or [])
        if parent_version_id and not any(
            (p == parent_version_id)
            or (isinstance(p, dict) and p.get("parent_version_id") == parent_version_id)
            for p in parent_values
        ):
            parent_values.insert(0, parent_version_id)
        if parent_values:
            self.set_dataset_version_parents(identifier, parent_values)
            record = self.get_dataset_version(identifier) or record
        return record

    def get_dataset_version(self, version_id: str) -> Optional["DatasetVersionRecord"]:
        row = self._conn.execute(
            "SELECT * FROM dataset_versions WHERE id = ?", (version_id,)
        ).fetchone()
        if not row:
            return None
        record = _row_to_dataset_version(row)
        record.parents = self.list_dataset_version_parents(version_id)
        return record

    def find_dataset_version(
        self, *, dataset_id: str, content_hash: str, recipe_hash: str
    ) -> Optional["DatasetVersionRecord"]:
        row = self._conn.execute(
            """
            SELECT * FROM dataset_versions
            WHERE dataset_id = ? AND content_hash = ? AND recipe_hash = ? AND status = 'completed'
            """,
            (dataset_id, content_hash, recipe_hash),
        ).fetchone()
        return _row_to_dataset_version(row) if row else None

    def list_dataset_versions(self, dataset_id: str) -> List["DatasetVersionRecord"]:
        rows = self._conn.execute(
            "SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY created_at DESC",
            (dataset_id,),
        ).fetchall()
        records = [_row_to_dataset_version(row) for row in rows]
        for record in records:
            record.parents = self.list_dataset_version_parents(record.id)
        return records

    def set_dataset_version_parents(
        self, version_id: str, parents: Iterable[Any]
    ) -> List[Dict[str, Any]]:
        if self.get_dataset_version(version_id) is None:
            raise ValueError(f"unknown dataset version: {version_id}")
        normalized: list[tuple[str, str, Optional[float]]] = []
        for value in parents:
            if isinstance(value, str):
                parent_id, role, weight = value, "parent", None
            elif isinstance(value, dict):
                parent_id = str(value.get("parent_version_id") or value.get("id") or "")
                role = str(value.get("role") or "parent")
                weight = float(value["weight"]) if value.get("weight") is not None else None
            else:
                raise ValueError("parent versions must be ids or objects")
            if not parent_id:
                raise ValueError("parent_version_id is required")
            if parent_id == version_id:
                raise ValueError("a dataset version cannot be its own parent")
            normalized.append((parent_id, role, weight))
        with self._lock:
            self._conn.execute(
                "DELETE FROM dataset_version_parents WHERE version_id = ?", (version_id,)
            )
            self._conn.executemany(
                """
                INSERT INTO dataset_version_parents (version_id, parent_version_id, role, weight)
                VALUES (?, ?, ?, ?)
                """,
                [(version_id, parent_id, role, weight) for parent_id, role, weight in normalized],
            )
            self._conn.commit()
        return self.list_dataset_version_parents(version_id)

    def list_dataset_version_parents(self, version_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT parent_version_id, role, weight
            FROM dataset_version_parents WHERE version_id = ?
            ORDER BY role, parent_version_id
            """,
            (version_id,),
        ).fetchall()
        return [
            {
                "parent_version_id": row["parent_version_id"],
                "role": row["role"],
                "weight": row["weight"],
            }
            for row in rows
        ]

    def update_dataset_version(
        self,
        version_id: str,
        *,
        status: Optional[str] = None,
        content_hash: Optional[str] = None,
        row_count: Optional[int] = None,
        size_bytes: Optional[int] = None,
        split_counts: Optional[Dict[str, int]] = None,
        statistics: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        source_fingerprints: Optional[Dict[str, str]] = None,
        error: Optional[str] = None,
    ) -> Optional["DatasetVersionRecord"]:
        existing = self.get_dataset_version(version_id)
        if existing is None:
            return None
        if existing.status == "completed":
            raise ValueError("completed dataset versions are immutable")
        new_status = status or existing.status
        completed_at = (
            datetime.now(timezone.utc).isoformat()
            if new_status == "completed"
            else existing.completed_at
        )
        with self._lock:
            self._conn.execute(
                """
                UPDATE dataset_versions SET status = ?, content_hash = ?, row_count = ?,
                    size_bytes = ?, split_counts_json = ?, statistics_json = ?,
                    provenance_json = ?, source_fingerprints_json = ?, error = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    new_status,
                    content_hash if content_hash is not None else existing.content_hash,
                    row_count if row_count is not None else existing.row_count,
                    size_bytes if size_bytes is not None else existing.size_bytes,
                    json.dumps(
                        split_counts if split_counts is not None else existing.split_counts,
                        sort_keys=True,
                    ),
                    json.dumps(
                        statistics if statistics is not None else existing.statistics,
                        sort_keys=True,
                    ),
                    json.dumps(
                        provenance if provenance is not None else existing.provenance,
                        sort_keys=True,
                    ),
                    json.dumps(
                        (
                            source_fingerprints
                            if source_fingerprints is not None
                            else existing.source_fingerprints
                        ),
                        sort_keys=True,
                    ),
                    error,
                    completed_at,
                    version_id,
                ),
            )
            if new_status == "completed":
                self._conn.execute(
                    "UPDATE datasets SET latest_version_id = ?, updated_at = ? WHERE id = ?",
                    (version_id, completed_at, existing.dataset_id),
                )
            self._conn.commit()
        return self.get_dataset_version(version_id)

    def set_version_assets_materialized(
        self, version_id: str, materialized: bool = True
    ) -> Optional["DatasetVersionRecord"]:
        if self.get_dataset_version(version_id) is None:
            return None
        with self._lock:
            self._conn.execute(
                "UPDATE dataset_versions SET assets_materialized = ? WHERE id = ?",
                (1 if materialized else 0, version_id),
            )
            self._conn.commit()
        return self.get_dataset_version(version_id)

    def create_dataset_job(
        self,
        *,
        job_type: str,
        dataset_id: Optional[str] = None,
        version_id: Optional[str] = None,
        request: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None,
        status: str = "queued",
        stage: str = "queued",
        work_item_id: Optional[str] = None,
    ) -> "DatasetJobRecord":
        identifier = job_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO dataset_jobs
                    (id, dataset_id, version_id, job_type, status, stage,
                     logs_json, request_json, checkpoint_json, created_at, updated_at,
                     work_item_id)
                VALUES (?, ?, ?, ?, ?, ?, '[]', ?, '{}', ?, ?, ?)
                """,
                (
                    identifier,
                    dataset_id,
                    version_id,
                    job_type,
                    status,
                    stage,
                    json.dumps(request or {}, sort_keys=True),
                    now,
                    now,
                    work_item_id,
                ),
            )
            self._conn.commit()
        record = self.get_dataset_job(identifier)
        assert record is not None
        return record

    def get_dataset_job(self, job_id: str) -> Optional["DatasetJobRecord"]:
        row = self._conn.execute("SELECT * FROM dataset_jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_dataset_job(row) if row else None

    def list_dataset_jobs(
        self,
        *,
        dataset_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List["DatasetJobRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        if dataset_id:
            clauses.append("dataset_id = ?")
            params.append(dataset_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        rows = self._conn.execute(
            f"SELECT * FROM dataset_jobs {where} ORDER BY created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_dataset_job(row) for row in rows]

    def update_dataset_job(self, job_id: str, **changes: Any) -> Optional["DatasetJobRecord"]:
        existing = self.get_dataset_job(job_id)
        if existing is None:
            return None
        allowed = {
            "version_id",
            "status",
            "stage",
            "processed_records",
            "total_records",
            "accepted_records",
            "rejected_records",
            "output_size_bytes",
            "error",
            "cancel_requested",
            "started_at",
            "completed_at",
            "work_item_id",
        }
        assignments: list[str] = []
        params: list[Any] = []
        for key, value in changes.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                params.append(1 if key == "cancel_requested" and value else value)
        for key, column in (("logs", "logs_json"), ("checkpoint", "checkpoint_json")):
            if key in changes:
                assignments.append(f"{column} = ?")
                params.append(json.dumps(changes[key], sort_keys=True))
        if not assignments:
            return existing
        assignments.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(job_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE dataset_jobs SET {', '.join(assignments)} WHERE id = ?", params
            )
            self._conn.commit()
        return self.get_dataset_job(job_id)

    def cancel_dataset_job(self, job_id: str) -> Optional["DatasetJobRecord"]:
        existing = self.get_dataset_job(job_id)
        if existing is None:
            return None
        changes: Dict[str, Any] = {"cancel_requested": True}
        if existing.status in {"queued", "interrupted"}:
            changes.update(
                status="cancelled",
                stage="cancelled",
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        return self.update_dataset_job(job_id, **changes)

    def retry_dataset_job(self, job_id: str) -> Optional["DatasetJobRecord"]:
        existing = self.get_dataset_job(job_id)
        if existing is None:
            return None
        if existing.status not in {"failed", "cancelled", "interrupted"}:
            raise ValueError(f"job in {existing.status!r} state cannot be retried")
        return self.update_dataset_job(
            job_id,
            status="queued",
            stage="queued",
            error=None,
            cancel_requested=False,
            completed_at=None,
        )

    # ----- Dataset Lab v9 guided imports ---------------------------------

    def create_dataset_import(
        self,
        *,
        source_kind: str,
        display_name: Optional[str] = None,
        source_uri: Optional[str] = None,
        source_config: Optional[str] = None,
        source_split: Optional[str] = None,
        source_revision: Optional[str] = None,
        resolved_revision: Optional[str] = None,
        scenario_revision_id: Optional[str] = None,
        expected_size_bytes: Optional[int] = None,
        staging_path: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        expires_at: Optional[str] = None,
        import_id: Optional[str] = None,
        status: str = "draft",
    ) -> "DatasetImportRecord":
        allowed_kinds = {"upload", "workstation_path", "huggingface", "desktop_reference"}
        if source_kind not in allowed_kinds:
            raise ValueError(f"unsupported dataset import source_kind: {source_kind}")
        if expected_size_bytes is not None and int(expected_size_bytes) < 0:
            raise ValueError("expected_size_bytes cannot be negative")
        identifier = import_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO dataset_imports
                    (id, source_kind, status, display_name, source_uri, source_config,
                     source_split, source_revision, resolved_revision,
                     scenario_revision_id, expected_size_bytes, staging_path,
                     metadata_json, expires_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    source_kind,
                    status,
                    display_name,
                    source_uri,
                    source_config,
                    source_split,
                    source_revision,
                    resolved_revision,
                    scenario_revision_id,
                    expected_size_bytes,
                    staging_path,
                    json.dumps(dict(metadata or {}), sort_keys=True),
                    expires_at,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        record = self.get_dataset_import(identifier)
        assert record is not None
        return record

    def get_dataset_import(self, import_id: str) -> Optional["DatasetImportRecord"]:
        row = self._conn.execute(
            "SELECT * FROM dataset_imports WHERE id = ?", (import_id,)
        ).fetchone()
        return _row_to_dataset_import(row) if row else None

    def list_dataset_imports(
        self,
        *,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List["DatasetImportRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([max(1, int(limit)), max(0, int(offset))])
        rows = self._conn.execute(
            f"SELECT * FROM dataset_imports {where} "
            "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_row_to_dataset_import(row) for row in rows]

    def update_dataset_import(
        self, import_id: str, **changes: Any
    ) -> Optional["DatasetImportRecord"]:
        existing = self.get_dataset_import(import_id)
        if existing is None:
            return None
        allowed = {
            "status",
            "display_name",
            "source_uri",
            "source_config",
            "source_split",
            "source_revision",
            "resolved_revision",
            "scenario_revision_id",
            "fingerprint",
            "expected_size_bytes",
            "received_size_bytes",
            "file_count",
            "staging_path",
            "managed_source_path",
            "work_item_id",
            "published_dataset_id",
            "published_source_id",
            "latest_inspection_id",
            "error",
            "expires_at",
            "completed_at",
        }
        assignments: list[str] = []
        params: list[Any] = []
        for key, value in changes.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                params.append(value)
        if "metadata" in changes:
            assignments.append("metadata_json = ?")
            params.append(json.dumps(dict(changes["metadata"] or {}), sort_keys=True))
        if not assignments:
            return existing
        assignments.append("updated_at = ?")
        params.extend([datetime.now(timezone.utc).isoformat(), import_id])
        with self._lock:
            self._conn.execute(
                f"UPDATE dataset_imports SET {', '.join(assignments)} WHERE id = ?", params
            )
            self._conn.commit()
        return self.get_dataset_import(import_id)

    def create_dataset_import_file(
        self,
        *,
        import_id: str,
        relative_path: str,
        size_bytes: int,
        staging_path: str,
        media_type: Optional[str] = None,
        expected_sha256: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        file_id: Optional[str] = None,
    ) -> "DatasetImportFileRecord":
        if self.get_dataset_import(import_id) is None:
            raise ValueError(f"unknown dataset import: {import_id}")
        if int(size_bytes) < 0:
            raise ValueError("size_bytes cannot be negative")
        identifier = file_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO dataset_import_files
                    (id, import_id, relative_path, status, media_type, size_bytes,
                     expected_sha256, staging_path, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    import_id,
                    relative_path,
                    media_type,
                    int(size_bytes),
                    expected_sha256,
                    staging_path,
                    json.dumps(dict(metadata or {}), sort_keys=True),
                    now,
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE dataset_imports SET file_count = file_count + 1, "
                "updated_at = ? WHERE id = ?",
                (now, import_id),
            )
            self._conn.commit()
        record = self.get_dataset_import_file(identifier)
        assert record is not None
        return record

    def get_dataset_import_file(
        self, file_id: str
    ) -> Optional["DatasetImportFileRecord"]:
        row = self._conn.execute(
            "SELECT * FROM dataset_import_files WHERE id = ?", (file_id,)
        ).fetchone()
        return _row_to_dataset_import_file(row) if row else None

    def list_dataset_import_files(self, import_id: str) -> List["DatasetImportFileRecord"]:
        rows = self._conn.execute(
            "SELECT * FROM dataset_import_files WHERE import_id = ? ORDER BY relative_path",
            (import_id,),
        ).fetchall()
        return [_row_to_dataset_import_file(row) for row in rows]

    def update_dataset_import_file(
        self, file_id: str, **changes: Any
    ) -> Optional["DatasetImportFileRecord"]:
        existing = self.get_dataset_import_file(file_id)
        if existing is None:
            return None
        allowed = {
            "status",
            "received_bytes",
            "content_sha256",
            "error",
            "completed_at",
        }
        assignments: list[str] = []
        params: list[Any] = []
        for key, value in changes.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                params.append(value)
        if "metadata" in changes:
            assignments.append("metadata_json = ?")
            params.append(json.dumps(dict(changes["metadata"] or {}), sort_keys=True))
        if not assignments:
            return existing
        assignments.append("updated_at = ?")
        params.extend([datetime.now(timezone.utc).isoformat(), file_id])
        with self._lock:
            self._conn.execute(
                f"UPDATE dataset_import_files SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
            self._conn.commit()
        return self.get_dataset_import_file(file_id)

    def create_dataset_source_inspection(
        self,
        *,
        source_fingerprint: str,
        import_adapter_version: str,
        scenario_registry_revision: str,
        import_id: Optional[str] = None,
        source_id: Optional[str] = None,
        scenario_revision_id: Optional[str] = None,
        inspection_id: Optional[str] = None,
        status: str = "queued",
        work_item_id: Optional[str] = None,
    ) -> "DatasetSourceInspectionRecord":
        identifier = inspection_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO dataset_source_inspections
                        (id, import_id, source_id, status, source_fingerprint,
                         import_adapter_version, scenario_registry_revision,
                         scenario_revision_id, work_item_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        import_id,
                        source_id,
                        status,
                        source_fingerprint,
                        import_adapter_version,
                        scenario_registry_revision,
                        scenario_revision_id,
                        work_item_id,
                        now,
                    ),
                )
                self._conn.commit()
        except sqlite3.IntegrityError:
            reused = self.find_dataset_source_inspection(
                source_fingerprint=source_fingerprint,
                import_adapter_version=import_adapter_version,
                scenario_registry_revision=scenario_registry_revision,
            )
            if reused is not None:
                return reused
            raise
        record = self.get_dataset_source_inspection(identifier)
        assert record is not None
        return record

    def get_dataset_source_inspection(
        self, inspection_id: str
    ) -> Optional["DatasetSourceInspectionRecord"]:
        row = self._conn.execute(
            "SELECT * FROM dataset_source_inspections WHERE id = ?", (inspection_id,)
        ).fetchone()
        return _row_to_dataset_source_inspection(row) if row else None

    def link_dataset_import_inspection(
        self, import_id: str, inspection_id: str
    ) -> None:
        if self.get_dataset_import(import_id) is None:
            raise ValueError(f"unknown dataset import: {import_id}")
        if self.get_dataset_source_inspection(inspection_id) is None:
            raise ValueError(f"unknown dataset inspection: {inspection_id}")
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO dataset_import_inspections
                    (import_id, inspection_id, linked_at)
                VALUES (?, ?, ?)
                """,
                (import_id, inspection_id, datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()

    def list_dataset_inspection_import_ids(self, inspection_id: str) -> List[str]:
        rows = self._conn.execute(
            """
            SELECT link.import_id
            FROM dataset_import_inspections link
            JOIN dataset_imports import_record ON import_record.id = link.import_id
            WHERE link.inspection_id = ?
            ORDER BY link.linked_at DESC, import_record.updated_at DESC, link.import_id
            """,
            (inspection_id,),
        ).fetchall()
        return [str(row["import_id"]) for row in rows]

    def dataset_import_uses_inspection(
        self, import_id: str, inspection_id: str
    ) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 FROM dataset_import_inspections
            WHERE import_id = ? AND inspection_id = ?
            """,
            (import_id, inspection_id),
        ).fetchone()
        return row is not None

    def find_dataset_source_inspection(
        self,
        *,
        source_fingerprint: str,
        import_adapter_version: str,
        scenario_registry_revision: str,
    ) -> Optional["DatasetSourceInspectionRecord"]:
        row = self._conn.execute(
            """
            SELECT * FROM dataset_source_inspections
            WHERE source_fingerprint = ? AND import_adapter_version = ?
              AND scenario_registry_revision = ?
            """,
            (
                source_fingerprint,
                import_adapter_version,
                scenario_registry_revision,
            ),
        ).fetchone()
        return _row_to_dataset_source_inspection(row) if row else None

    def list_dataset_source_inspections(
        self,
        *,
        import_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List["DatasetSourceInspectionRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        if import_id:
            clauses.append("import_id = ?")
            params.append(import_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([max(1, int(limit)), max(0, int(offset))])
        rows = self._conn.execute(
            f"SELECT * FROM dataset_source_inspections {where} "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_row_to_dataset_source_inspection(row) for row in rows]

    def update_dataset_source_inspection(
        self, inspection_id: str, **changes: Any
    ) -> Optional["DatasetSourceInspectionRecord"]:
        existing = self.get_dataset_source_inspection(inspection_id)
        if existing is None:
            return None
        if existing.status == "completed":
            raise ValueError("completed dataset inspections are immutable")
        allowed = {
            "status",
            "source_fingerprint",
            "scenario_revision_id",
            "total_records",
            "valid_records",
            "invalid_records",
            "sample_count",
            "size_bytes",
            "work_item_id",
            "error",
            "completed_at",
        }
        assignments: list[str] = []
        params: list[Any] = []
        for key, value in changes.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                params.append(value)
        json_fields = {
            "fields": "fields_json",
            "candidates": "candidates_json",
            "preview": "preview_json",
            "issues": "issues_json",
            "statistics": "statistics_json",
        }
        for key, column in json_fields.items():
            if key in changes:
                assignments.append(f"{column} = ?")
                params.append(json.dumps(changes[key], sort_keys=True, default=str))
        if not assignments:
            return existing
        params.append(inspection_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE dataset_source_inspections SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
            self._conn.commit()
        return self.get_dataset_source_inspection(inspection_id)

    def delete_incomplete_dataset_source_inspection(self, inspection_id: str) -> bool:
        existing = self.get_dataset_source_inspection(inspection_id)
        if existing is None:
            return False
        if existing.status == "completed":
            raise ValueError("completed dataset inspections are immutable")
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM dataset_source_inspections WHERE id = ?", (inspection_id,)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    # ----- Halo Forge Lab v10 corpus extraction ---------------------------

    def create_document_extraction(
        self,
        *,
        source_kind: str,
        source_uri: str,
        source_fingerprint: str,
        extractor_version: str,
        config_hash: str,
        config: Optional[Mapping[str, Any]] = None,
        reuse_key: Optional[str] = None,
        import_id: Optional[str] = None,
        source_id: Optional[str] = None,
        extraction_id: Optional[str] = None,
        status: str = "queued",
        work_item_id: Optional[str] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> "DocumentExtractionRecord":
        """Create or deterministically reuse one corpus-extraction lifecycle row."""

        normalized_kind = str(source_kind or "").strip().lower()
        normalized_uri = str(source_uri or "").strip()
        normalized_fingerprint = str(source_fingerprint or "").strip()
        normalized_version = str(extractor_version or "").strip()
        normalized_config_hash = str(config_hash or "").strip()
        if not normalized_kind:
            raise ValueError("source_kind is required")
        if not normalized_uri:
            raise ValueError("source_uri is required")
        if not normalized_fingerprint:
            raise ValueError("source_fingerprint is required")
        if not normalized_version:
            raise ValueError("extractor_version is required")
        if not normalized_config_hash:
            raise ValueError("config_hash is required")
        if status not in {"queued", "running"}:
            raise ValueError("new document extractions must be queued or running")
        if import_id is not None and self.get_dataset_import(import_id) is None:
            raise ValueError(f"unknown dataset import: {import_id}")
        if source_id is not None and self.get_dataset_source(source_id) is None:
            raise ValueError(f"unknown dataset source: {source_id}")

        identity = reuse_key or _content_hash(
            {
                "source_fingerprint": normalized_fingerprint,
                "extractor_version": normalized_version,
                "config_hash": normalized_config_hash,
            }
        )
        identifier = extraction_id or f"dex-{identity[:24]}"
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO document_extractions
                        (id, import_id, source_id, status, source_kind, source_uri,
                         source_fingerprint, extractor_version, config_hash,
                         reuse_key, config_json, provenance_json, work_item_id,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        import_id,
                        source_id,
                        status,
                        normalized_kind,
                        normalized_uri,
                        normalized_fingerprint,
                        normalized_version,
                        normalized_config_hash,
                        identity,
                        json.dumps(dict(config or {}), sort_keys=True, default=str),
                        json.dumps(dict(provenance or {}), sort_keys=True, default=str),
                        work_item_id,
                        now,
                        now,
                    ),
                )
                self._conn.commit()
        except sqlite3.IntegrityError:
            with self._lock:
                self._conn.rollback()
            existing = self.find_document_extraction(reuse_key=identity)
            if existing is not None:
                return existing
            raise
        record = self.get_document_extraction(identifier)
        assert record is not None
        return record

    def get_document_extraction(
        self, extraction_id: str
    ) -> Optional["DocumentExtractionRecord"]:
        row = self._conn.execute(
            "SELECT * FROM document_extractions WHERE id = ?", (extraction_id,)
        ).fetchone()
        return _row_to_document_extraction(row) if row else None

    def find_document_extraction(
        self,
        *,
        reuse_key: Optional[str] = None,
        source_fingerprint: Optional[str] = None,
        extractor_version: Optional[str] = None,
        config_hash: Optional[str] = None,
        completed_only: bool = False,
    ) -> Optional["DocumentExtractionRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("reuse_key", reuse_key),
            ("source_fingerprint", source_fingerprint),
            ("extractor_version", extractor_version),
            ("config_hash", config_hash),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if not clauses:
            raise ValueError("at least one document-extraction identity field is required")
        if completed_only:
            clauses.append("status = 'completed'")
        row = self._conn.execute(
            "SELECT * FROM document_extractions "
            f"WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT 1",
            params,
        ).fetchone()
        return _row_to_document_extraction(row) if row else None

    def list_document_extractions(
        self,
        *,
        import_id: Optional[str] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List["DocumentExtractionRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("import_id", import_id),
            ("source_id", source_id),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend((max(1, int(limit)), max(0, int(offset))))
        rows = self._conn.execute(
            f"SELECT * FROM document_extractions {where} "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_row_to_document_extraction(row) for row in rows]

    def count_document_extractions(
        self,
        *,
        import_id: Optional[str] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("import_id", import_id),
            ("source_id", source_id),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self._conn.execute(
            f"SELECT COUNT(*) AS total FROM document_extractions {where}",
            params,
        ).fetchone()
        return int(row["total"] if row is not None else 0)

    def update_document_extraction(
        self, extraction_id: str, **changes: Any
    ) -> Optional["DocumentExtractionRecord"]:
        existing = self.get_document_extraction(extraction_id)
        if existing is None:
            return None
        if existing.status == "completed":
            raise ValueError("completed document extractions are immutable")
        allowed = {
            "status",
            "work_item_id",
            "error",
            "completed_at",
        }
        assignments: list[str] = []
        params: list[Any] = []
        for key, value in changes.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                params.append(value)
        for key, column in (
            ("statistics", "statistics_json"),
            ("provenance", "provenance_json"),
        ):
            if key in changes:
                assignments.append(f"{column} = ?")
                params.append(
                    json.dumps(dict(changes[key] or {}), sort_keys=True, default=str)
                )
        if not assignments:
            return existing
        assignments.append("updated_at = ?")
        params.extend((datetime.now(timezone.utc).isoformat(), extraction_id))
        with self._lock:
            self._conn.execute(
                f"UPDATE document_extractions SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
            self._conn.commit()
        return self.get_document_extraction(extraction_id)

    def complete_document_extraction(
        self,
        extraction_id: str,
        *,
        content_hash: str,
        bundle_path: str,
        manifest_hash: str,
        items: Sequence[Mapping[str, Any]],
        statistics: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        completed_at: Optional[str] = None,
    ) -> "DocumentExtractionRecord":
        """Atomically index every item and seal the extraction as immutable."""

        existing = self.get_document_extraction(extraction_id)
        if existing is None:
            raise ValueError(f"unknown document extraction: {extraction_id}")
        if existing.status == "completed":
            if (
                existing.content_hash == content_hash
                and existing.bundle_path == str(bundle_path)
                and existing.manifest_hash == manifest_hash
            ):
                return existing
            raise ValueError("completed document extraction identity cannot change")
        if not str(content_hash or "").strip():
            raise ValueError("content_hash is required")
        if not str(bundle_path or "").strip():
            raise ValueError("bundle_path is required")
        if not str(manifest_hash or "").strip():
            raise ValueError("manifest_hash is required")

        rows: list[tuple[Any, ...]] = []
        document_count = 0
        quarantined_count = 0
        extracted_text_bytes = 0
        document_member_ordinal = 0
        quarantine_member_ordinal = 0
        for ordinal, raw_item in enumerate(items):
            item = dict(raw_item)
            item_status = str(item.get("status") or "extracted")
            if item_status not in {"extracted", "quarantined"}:
                raise ValueError(f"invalid document extraction item status: {item_status}")
            document_id = str(item.get("document_id") or item.get("id") or "").strip()
            if not document_id:
                raise ValueError("document extraction items require document_id")
            if item_status == "extracted":
                item_content_hash = str(
                    item.get("content_hash") or item.get("text_sha256") or ""
                ).strip()
                if not item_content_hash:
                    raise ValueError("extracted document items require content_hash")
                error_code = None
                error = None
                text_char_count = int(item.get("text_char_count", item.get("char_count", 0)))
                text_byte_count = int(item.get("text_byte_count", item.get("byte_count", 0)))
                bundle_member = str(item.get("bundle_member") or "documents.jsonl")
                bundle_ordinal = int(
                    item.get("bundle_ordinal", document_member_ordinal)
                )
                document_member_ordinal += 1
                document_count += 1
                extracted_text_bytes += text_byte_count
            else:
                item_content_hash = None
                error_code = str(item.get("error_code") or "extraction_failed").strip()
                error = str(item.get("error") or "").strip()
                if not error:
                    raise ValueError("quarantined document items require an error")
                text_char_count = 0
                text_byte_count = 0
                bundle_member = str(item.get("bundle_member") or "quarantine.jsonl")
                bundle_ordinal = int(
                    item.get("bundle_ordinal", quarantine_member_ordinal)
                )
                quarantine_member_ordinal += 1
                quarantined_count += 1
            rows.append(
                (
                    extraction_id,
                    ordinal,
                    document_id,
                    item_status,
                    str(item.get("source_uri") or existing.source_uri),
                    str(item.get("relative_path") or ""),
                    str(item.get("source_kind") or existing.source_kind),
                    str(item.get("media_type") or "text/plain"),
                    item.get("title"),
                    item_content_hash,
                    max(0, text_char_count),
                    max(0, text_byte_count),
                    bundle_member,
                    max(0, bundle_ordinal),
                    json.dumps(dict(item.get("locator") or {}), sort_keys=True, default=str),
                    json.dumps(
                        dict(item.get("provenance") or {}), sort_keys=True, default=str
                    ),
                    json.dumps(dict(item.get("metadata") or {}), sort_keys=True, default=str),
                    error_code,
                    error,
                )
            )
        now = completed_at or datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                with self._conn:
                    self._conn.executemany(
                        """
                        INSERT INTO document_extraction_items
                            (extraction_id, ordinal, document_id, status, source_uri,
                             relative_path, source_kind, media_type, title,
                             content_hash, text_char_count, text_byte_count,
                             bundle_member, bundle_ordinal, locator_json,
                             provenance_json, metadata_json, error_code, error)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                    cursor = self._conn.execute(
                        """
                        UPDATE document_extractions
                        SET status = 'completed', content_hash = ?, bundle_path = ?,
                            manifest_hash = ?, document_count = ?, item_count = ?,
                            quarantined_count = ?, extracted_text_bytes = ?,
                            statistics_json = ?, provenance_json = ?, error = NULL,
                            completed_at = ?, updated_at = ?
                        WHERE id = ? AND status != 'completed'
                        """,
                        (
                            content_hash,
                            str(bundle_path),
                            manifest_hash,
                            document_count,
                            len(rows),
                            quarantined_count,
                            extracted_text_bytes,
                            json.dumps(dict(statistics or {}), sort_keys=True, default=str),
                            json.dumps(
                                dict(provenance or existing.provenance),
                                sort_keys=True,
                                default=str,
                            ),
                            now,
                            now,
                            extraction_id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise ValueError(
                            f"document extraction {extraction_id} could not be completed"
                        )
            except Exception:
                self._conn.rollback()
                raise
        completed = self.get_document_extraction(extraction_id)
        assert completed is not None
        return completed

    def get_document_extraction_item(
        self, extraction_id: str, ordinal: int
    ) -> Optional["DocumentExtractionItemRecord"]:
        row = self._conn.execute(
            """
            SELECT * FROM document_extraction_items
            WHERE extraction_id = ? AND ordinal = ?
            """,
            (extraction_id, int(ordinal)),
        ).fetchone()
        return _row_to_document_extraction_item(row) if row else None

    def list_document_extraction_items(
        self,
        extraction_id: str,
        *,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List["DocumentExtractionItemRecord"]:
        clauses = ["extraction_id = ?"]
        params: list[Any] = [extraction_id]
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        pagination = ""
        if limit is not None:
            pagination = " LIMIT ? OFFSET ?"
            params.extend((max(0, int(limit)), max(0, int(offset))))
        rows = self._conn.execute(
            "SELECT * FROM document_extraction_items "
            f"WHERE {' AND '.join(clauses)} ORDER BY ordinal{pagination}",
            params,
        ).fetchall()
        return [_row_to_document_extraction_item(row) for row in rows]

    def delete_incomplete_document_extraction(self, extraction_id: str) -> bool:
        existing = self.get_document_extraction(extraction_id)
        if existing is None:
            return False
        if existing.status == "completed":
            raise ValueError("completed document extractions are immutable")
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM document_extractions WHERE id = ?", (extraction_id,)
            )
            self._conn.commit()
        return cursor.rowcount == 1

    def attach_run_dataset(
        self,
        *,
        run_id: str,
        dataset_version_id: str,
        split: str = "train",
        role: str = "train",
        training_artifact_id: Optional[str] = None,
    ) -> "RunDatasetRecord":
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO run_datasets
                    (run_id, dataset_version_id, role, split, training_artifact_id, attached_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, dataset_version_id, split, role) DO UPDATE SET
                    training_artifact_id = COALESCE(
                        excluded.training_artifact_id, run_datasets.training_artifact_id
                    )
                """,
                (run_id, dataset_version_id, role, split, training_artifact_id, now),
            )
            self._conn.commit()
            row = self._conn.execute(
                """
                SELECT * FROM run_datasets
                WHERE run_id = ? AND dataset_version_id = ? AND split = ? AND role = ?
                """,
                (run_id, dataset_version_id, split, role),
            ).fetchone()
        assert row is not None
        return RunDatasetRecord(
            run_id=row["run_id"],
            dataset_version_id=row["dataset_version_id"],
            split=row["split"],
            attached_at=row["attached_at"],
            role=row["role"],
            training_artifact_id=row["training_artifact_id"],
        )

    def list_run_datasets(self, run_id: str) -> List["RunDatasetRecord"]:
        rows = self._conn.execute(
            "SELECT * FROM run_datasets WHERE run_id = ? ORDER BY attached_at",
            (run_id,),
        ).fetchall()
        return [
            RunDatasetRecord(
                run_id=row["run_id"],
                dataset_version_id=row["dataset_version_id"],
                split=row["split"],
                attached_at=row["attached_at"],
                role=row["role"],
                training_artifact_id=row["training_artifact_id"],
            )
            for row in rows
        ]

    def list_runs_for_dataset_version(self, dataset_version_id: str) -> List["RunDatasetRecord"]:
        rows = self._conn.execute(
            """
            SELECT * FROM run_datasets WHERE dataset_version_id = ?
            ORDER BY attached_at DESC, run_id, role
            """,
            (dataset_version_id,),
        ).fetchall()
        return [
            RunDatasetRecord(
                run_id=row["run_id"],
                dataset_version_id=row["dataset_version_id"],
                split=row["split"],
                attached_at=row["attached_at"],
                role=row["role"],
                training_artifact_id=row["training_artifact_id"],
            )
            for row in rows
        ]

    # ----- Dataset Lab v2 training artifacts -------------------------------

    def create_training_artifact(
        self,
        *,
        artifact_hash: str,
        adapter_id: str,
        adapter_version: str,
        trainer_mode: str,
        manifest_path: str,
        bindings: Sequence[Mapping[str, Any]],
        artifact_id: Optional[str] = None,
        model_id: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        chat_template_hash: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "TrainingArtifactRecord":
        if not artifact_hash:
            raise ValueError("artifact_hash is required")
        existing = self.find_training_artifact(artifact_hash)
        if existing is not None:
            return existing
        identifier = artifact_id or artifact_hash
        now = datetime.now(timezone.utc).isoformat()
        normalized: list[tuple[str, str, str, int]] = []
        seen_bindings: set[tuple[str, str, str]] = set()
        for binding in bindings:
            role = str(binding.get("role") or "train")
            version_id = str(binding.get("dataset_version_id") or "")
            split = str(binding.get("split") or role)
            if not version_id:
                raise ValueError("dataset_version_id is required for artifact bindings")
            if self.get_dataset_version(version_id) is None:
                raise ValueError(f"unknown dataset version: {version_id}")
            key = (role, version_id, split)
            if key in seen_bindings:
                raise ValueError(f"duplicate training artifact binding: {key}")
            seen_bindings.add(key)
            normalized.append((role, version_id, split, int(binding.get("row_count") or 0)))
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO training_artifacts
                        (id, artifact_hash, adapter_id, adapter_version, trainer_mode,
                         model_id, tokenizer_revision, chat_template_hash, manifest_path,
                         metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        artifact_hash,
                        adapter_id,
                        adapter_version,
                        trainer_mode,
                        model_id,
                        tokenizer_revision,
                        chat_template_hash,
                        manifest_path,
                        json.dumps(dict(metadata or {}), sort_keys=True),
                        now,
                    ),
                )
                self._conn.executemany(
                    """
                    INSERT INTO training_artifact_bindings
                        (artifact_id, role, dataset_version_id, split, row_count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [(identifier, *binding) for binding in normalized],
                )
                self._conn.commit()
        except sqlite3.IntegrityError as exc:
            with self._lock:
                self._conn.rollback()
            existing = self.find_training_artifact(artifact_hash)
            if existing is not None:
                return existing
            raise ValueError(f"could not create training artifact: {exc}") from exc
        record = self.get_training_artifact(identifier)
        assert record is not None
        return record

    def find_training_artifact(self, artifact_hash: str) -> Optional["TrainingArtifactRecord"]:
        row = self._conn.execute(
            "SELECT * FROM training_artifacts WHERE artifact_hash = ?", (artifact_hash,)
        ).fetchone()
        return self._training_artifact_from_row(row) if row else None

    def get_training_artifact(self, artifact_id: str) -> Optional["TrainingArtifactRecord"]:
        row = self._conn.execute(
            "SELECT * FROM training_artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        return self._training_artifact_from_row(row) if row else None

    def _training_artifact_from_row(self, row: sqlite3.Row) -> "TrainingArtifactRecord":
        bindings = self._conn.execute(
            """
            SELECT role, dataset_version_id, split, row_count
            FROM training_artifact_bindings WHERE artifact_id = ? ORDER BY role
            """,
            (row["id"],),
        ).fetchall()
        return TrainingArtifactRecord(
            id=row["id"],
            artifact_hash=row["artifact_hash"],
            adapter_id=row["adapter_id"],
            adapter_version=row["adapter_version"],
            trainer_mode=row["trainer_mode"],
            model_id=row["model_id"],
            tokenizer_revision=row["tokenizer_revision"],
            chat_template_hash=row["chat_template_hash"],
            manifest_path=row["manifest_path"],
            metadata_json=row["metadata_json"],
            created_at=row["created_at"],
            bindings=[TrainingArtifactBindingRecord(**dict(binding)) for binding in bindings],
        )

    def list_training_artifacts(
        self,
        *,
        dataset_version_id: Optional[str] = None,
        trainer_mode: Optional[str] = None,
        limit: int = 100,
    ) -> List["TrainingArtifactRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        join = ""
        if dataset_version_id:
            join = "JOIN training_artifact_bindings b ON b.artifact_id = a.id"
            clauses.append("b.dataset_version_id = ?")
            params.append(dataset_version_id)
        if trainer_mode:
            clauses.append("a.trainer_mode = ?")
            params.append(trainer_mode)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        rows = self._conn.execute(
            f"SELECT DISTINCT a.* FROM training_artifacts a {join} {where} "
            "ORDER BY a.created_at DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._training_artifact_from_row(row) for row in rows]

    # ----- Dataset Lab v2 evaluation catalog -------------------------------

    def create_benchmark_suite(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        purpose: str = "unspecified",
        suite_id: Optional[str] = None,
    ) -> "BenchmarkSuiteRecord":
        if not name or not name.strip():
            raise ValueError("benchmark suite name is required")
        identifier = suite_id or uuid.uuid4().hex
        if purpose not in {"development", "holdout", "operational", "unspecified"}:
            raise ValueError(
                "benchmark suite purpose must be development, holdout, operational, "
                "or unspecified"
            )
        stored_purpose = purpose if purpose != "operational" else "unspecified"
        purpose_v4 = purpose if purpose == "operational" else None
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO benchmark_suites
                        (id, name, description, latest_revision_id, purpose, purpose_v4, archived,
                         created_at, updated_at)
                    VALUES (?, ?, ?, NULL, ?, ?, 0, ?, ?)
                    """,
                    (
                        identifier,
                        name.strip(),
                        description,
                        stored_purpose,
                        purpose_v4,
                        now,
                        now,
                    ),
                )
                self._conn.commit()
        except sqlite3.IntegrityError as exc:
            with self._lock:
                self._conn.rollback()
            raise ValueError(f"benchmark suite {name!r} already exists") from exc
        record = self.get_benchmark_suite(identifier)
        assert record is not None
        return record

    def get_benchmark_suite(self, suite_id: str) -> Optional["BenchmarkSuiteRecord"]:
        row = self._conn.execute(
            "SELECT * FROM benchmark_suites WHERE id = ?", (suite_id,)
        ).fetchone()
        return _row_to_benchmark_suite(row) if row else None

    def list_benchmark_suites(
        self, *, include_archived: bool = False
    ) -> List["BenchmarkSuiteRecord"]:
        where = "" if include_archived else "WHERE archived = 0"
        rows = self._conn.execute(
            f"SELECT * FROM benchmark_suites {where} ORDER BY updated_at DESC, name"
        ).fetchall()
        return [_row_to_benchmark_suite(row) for row in rows]

    def update_benchmark_suite(
        self,
        suite_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        archived: Optional[bool] = None,
        purpose: Optional[str] = None,
    ) -> Optional["BenchmarkSuiteRecord"]:
        existing = self.get_benchmark_suite(suite_id)
        if existing is None:
            return None
        if purpose is not None and purpose not in {
            "development",
            "holdout",
            "operational",
            "unspecified",
        }:
            raise ValueError(
                "benchmark suite purpose must be development, holdout, operational, "
                "or unspecified"
            )
        resolved_purpose = purpose if purpose is not None else existing.purpose
        stored_purpose = resolved_purpose if resolved_purpose != "operational" else "unspecified"
        purpose_v4 = resolved_purpose if resolved_purpose == "operational" else None
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                self._conn.execute(
                    """
                    UPDATE benchmark_suites SET name = ?, description = ?, archived = ?,
                        purpose = ?, purpose_v4 = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        name.strip() if name is not None else existing.name,
                        description if description is not None else existing.description,
                        int(archived) if archived is not None else int(existing.archived),
                        stored_purpose,
                        purpose_v4,
                        now,
                        suite_id,
                    ),
                )
                self._conn.commit()
        except sqlite3.IntegrityError as exc:
            with self._lock:
                self._conn.rollback()
            raise ValueError(f"benchmark suite name {name!r} already exists") from exc
        return self.get_benchmark_suite(suite_id)

    def delete_benchmark_suite(self, suite_id: str) -> bool:
        """Archive a suite while retaining immutable revisions and evaluations."""
        return self.update_benchmark_suite(suite_id, archived=True) is not None

    def create_benchmark_suite_revision(
        self,
        *,
        suite_id: str,
        content_hash: str,
        items: Sequence[Mapping[str, Any]],
        primary_metric: str,
        direction: str,
        generation_settings: Optional[Mapping[str, Any]] = None,
        evaluator_versions: Optional[Mapping[str, Any]] = None,
        revision_id: Optional[str] = None,
    ) -> "BenchmarkSuiteRevisionRecord":
        if self.get_benchmark_suite(suite_id) is None:
            raise ValueError(f"unknown benchmark suite: {suite_id}")
        if direction not in {"maximize", "minimize"}:
            raise ValueError("direction must be 'maximize' or 'minimize'")
        if not primary_metric:
            raise ValueError("primary_metric is required")
        if not items:
            raise ValueError("a benchmark suite revision needs at least one item")
        existing = self._conn.execute(
            """
            SELECT * FROM benchmark_suite_revisions
            WHERE suite_id = ? AND content_hash = ?
            """,
            (suite_id, content_hash),
        ).fetchone()
        if existing:
            return _row_to_benchmark_revision(existing)
        identifier = revision_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                next_number = int(
                    self._conn.execute(
                        "SELECT COALESCE(MAX(revision_number), 0) + 1 FROM benchmark_suite_revisions "
                        "WHERE suite_id = ?",
                        (suite_id,),
                    ).fetchone()[0]
                )
                self._conn.execute(
                    """
                    INSERT INTO benchmark_suite_revisions
                        (id, suite_id, revision_number, content_hash, items_json,
                         generation_settings_json, evaluator_versions_json,
                         primary_metric, direction, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        suite_id,
                        next_number,
                        content_hash,
                        json.dumps(list(items), sort_keys=True),
                        json.dumps(dict(generation_settings or {}), sort_keys=True),
                        json.dumps(dict(evaluator_versions or {}), sort_keys=True),
                        primary_metric,
                        direction,
                        now,
                    ),
                )
                self._conn.execute(
                    "UPDATE benchmark_suites SET latest_revision_id = ?, updated_at = ? WHERE id = ?",
                    (identifier, now, suite_id),
                )
                self._conn.commit()
        except sqlite3.IntegrityError as exc:
            with self._lock:
                self._conn.rollback()
            row = self._conn.execute(
                "SELECT * FROM benchmark_suite_revisions WHERE suite_id = ? AND content_hash = ?",
                (suite_id, content_hash),
            ).fetchone()
            if row:
                return _row_to_benchmark_revision(row)
            raise ValueError(f"could not create benchmark revision: {exc}") from exc
        record = self.get_benchmark_suite_revision(identifier)
        assert record is not None
        return record

    def get_benchmark_suite_revision(
        self, revision_id: str
    ) -> Optional["BenchmarkSuiteRevisionRecord"]:
        row = self._conn.execute(
            "SELECT * FROM benchmark_suite_revisions WHERE id = ?", (revision_id,)
        ).fetchone()
        return _row_to_benchmark_revision(row) if row else None

    def list_benchmark_suite_revisions(self, suite_id: str) -> List["BenchmarkSuiteRevisionRecord"]:
        rows = self._conn.execute(
            """
            SELECT * FROM benchmark_suite_revisions
            WHERE suite_id = ? ORDER BY revision_number DESC
            """,
            (suite_id,),
        ).fetchall()
        return [_row_to_benchmark_revision(row) for row in rows]

    def create_evaluation(
        self,
        *,
        suite_revision_id: str,
        adapter_id: str,
        adapter_version: str,
        subject_type: str,
        subject_ref: str,
        subject_hash: str,
        reuse_key: str,
        request: Mapping[str, Any],
        evaluation_id: Optional[str] = None,
        work_item_id: Optional[str] = None,
    ) -> "EvaluationRecord":
        if self.get_benchmark_suite_revision(suite_revision_id) is None:
            raise ValueError(f"unknown benchmark suite revision: {suite_revision_id}")
        identifier = evaluation_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._lock:
                self._conn.execute(
                    """
                    INSERT INTO evaluations
                        (id, suite_revision_id, adapter_id, adapter_version,
                         subject_type, subject_ref, subject_hash, status, stage,
                         request_json, result_json, logs_json, reuse_key,
                         created_at, updated_at, work_item_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 'queued', ?, '{}', '[]', ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        suite_revision_id,
                        adapter_id,
                        adapter_version,
                        subject_type,
                        subject_ref,
                        subject_hash,
                        json.dumps(dict(request), sort_keys=True),
                        reuse_key,
                        now,
                        now,
                        work_item_id,
                    ),
                )
                self._conn.commit()
        except sqlite3.IntegrityError as exc:
            with self._lock:
                self._conn.rollback()
            raise ValueError(f"could not create evaluation: {exc}") from exc
        record = self.get_evaluation(identifier)
        assert record is not None
        return record

    def get_evaluation(self, evaluation_id: str) -> Optional["EvaluationRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
            ).fetchone()
        return _row_to_evaluation(row) if row else None

    def find_completed_evaluation(self, reuse_key: str) -> Optional["EvaluationRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM evaluations WHERE reuse_key = ? AND status = 'completed'",
                (reuse_key,),
            ).fetchone()
        return _row_to_evaluation(row) if row else None

    def list_evaluations(
        self,
        *,
        suite_revision_id: Optional[str] = None,
        subject_type: Optional[str] = None,
        subject_ref: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List["EvaluationRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("suite_revision_id", suite_revision_id),
            ("subject_type", subject_type),
            ("subject_ref", subject_ref),
            ("status", status),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM evaluations {where} ORDER BY created_at DESC LIMIT ?", params
            ).fetchall()
        return [_row_to_evaluation(row) for row in rows]

    def update_evaluation(self, evaluation_id: str, **changes: Any) -> Optional["EvaluationRecord"]:
        if self.get_evaluation(evaluation_id) is None:
            return None
        allowed = {
            "status",
            "stage",
            "processed_samples",
            "total_samples",
            "artifact_path",
            "error",
            "cancel_requested",
            "retry_count",
            "started_at",
            "completed_at",
            # Local model/checkpoint content hashing can be expensive. Evaluation
            # jobs are therefore inserted with a provisional identity and replace
            # these fields from the worker before adapter execution/publication.
            "subject_type",
            "subject_ref",
            "subject_hash",
            "reuse_key",
            "work_item_id",
        }
        assignments: list[str] = []
        params: list[Any] = []
        for key, value in changes.items():
            if key in allowed:
                assignments.append(f"{key} = ?")
                params.append(int(bool(value)) if key == "cancel_requested" else value)
        for key, column in (
            ("request", "request_json"),
            ("result", "result_json"),
            ("logs", "logs_json"),
        ):
            if key in changes:
                assignments.append(f"{column} = ?")
                params.append(json.dumps(changes[key], sort_keys=True, default=str))
        if not assignments:
            return self.get_evaluation(evaluation_id)
        assignments.append("updated_at = ?")
        params.extend((datetime.now(timezone.utc).isoformat(), evaluation_id))
        with self._lock:
            self._conn.execute(
                f"UPDATE evaluations SET {', '.join(assignments)} WHERE id = ?", params
            )
            self._conn.commit()
        return self.get_evaluation(evaluation_id)

    def claim_evaluation(
        self,
        evaluation_id: str,
        *,
        worker: Mapping[str, Any],
        stage: str = "starting",
    ) -> Optional["EvaluationRecord"]:
        """Atomically claim one queued evaluation across all processes.

        A per-process executor cannot enforce the single-workstation evaluation
        limit when CLI workers are detached. ``BEGIN IMMEDIATE`` serializes the
        no-running check and queued-to-running transition in SQLite, so at most
        one catalog row can be running even when several processes poll at once.
        ``None`` means the job is not currently claimable (usually because
        another evaluation owns the global slot).
        """

        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                target = self._conn.execute(
                    "SELECT status, cancel_requested, request_json FROM evaluations WHERE id = ?",
                    (evaluation_id,),
                ).fetchone()
                if (
                    target is None
                    or target["status"] not in {"queued", "interrupted"}
                    or bool(target["cancel_requested"])
                ):
                    self._conn.rollback()
                    return None
                running = self._conn.execute(
                    "SELECT id FROM evaluations WHERE status = 'running' LIMIT 1"
                ).fetchone()
                if running is not None:
                    self._conn.rollback()
                    return None
                request = _load_json(target["request_json"], {})
                request["worker"] = dict(worker)
                cursor = self._conn.execute(
                    """
                    UPDATE evaluations
                    SET status = 'running', stage = ?, request_json = ?, error = NULL,
                        started_at = COALESCE(started_at, ?), updated_at = ?
                    WHERE id = ? AND status IN ('queued', 'interrupted')
                          AND cancel_requested = 0
                    """,
                    (
                        str(stage),
                        json.dumps(request, sort_keys=True, default=str),
                        now,
                        now,
                        evaluation_id,
                    ),
                )
                if cursor.rowcount != 1:
                    self._conn.rollback()
                    return None
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_evaluation(evaluation_id)

    def interrupt_evaluation_if_unchanged(
        self,
        evaluation_id: str,
        *,
        expected_status: str,
        expected_request_json: str,
        error: str,
    ) -> bool:
        """Interrupt a dead owner's job without racing a fresh worker claim."""

        if expected_status not in {"queued", "running"}:
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE evaluations SET status = 'interrupted', stage = 'interrupted',
                    error = ?, updated_at = ?
                WHERE id = ? AND status = ? AND request_json = ?
                """,
                (
                    str(error),
                    now,
                    evaluation_id,
                    expected_status,
                    expected_request_json,
                ),
            )
            self._conn.commit()
        return cursor.rowcount == 1

    def complete_evaluation(
        self,
        evaluation_id: str,
        *,
        metrics: Sequence[Mapping[str, Any]],
        samples: Sequence[Mapping[str, Any]],
        result: Mapping[str, Any],
        artifact_path: str,
    ) -> "EvaluationRecord":
        now = datetime.now(timezone.utc).isoformat()
        metric_rows = [
            (
                evaluation_id,
                str(metric["name"]),
                float(metric["value"]),
                str(metric.get("direction") or "maximize"),
                str(metric.get("suite_item_id") or ""),
                json.dumps(dict(metric.get("metadata") or {}), sort_keys=True),
            )
            for metric in metrics
        ]
        sample_rows = [
            (
                evaluation_id,
                index,
                str(sample.get("suite_item_id") or index),
                sample.get("record_id"),
                json.dumps(sample.get("input"), sort_keys=True, default=str),
                json.dumps(sample.get("expected"), sort_keys=True, default=str),
                json.dumps(sample.get("output"), sort_keys=True, default=str),
                float(sample["score"]) if sample.get("score") is not None else None,
                int(bool(sample["passed"])) if sample.get("passed") is not None else None,
                float(sample["latency_ms"]) if sample.get("latency_ms") is not None else None,
                sample.get("error"),
                (
                    json.dumps(sample.get("verifier_trace"), sort_keys=True, default=str)
                    if sample.get("verifier_trace") is not None
                    else None
                ),
                (
                    int(sample["generation_seed"])
                    if sample.get("generation_seed") is not None
                    else None
                ),
                str(sample.get("evidence_kind") or "legacy"),
                int(bool(sample.get("valid", False))),
                int(bool(sample.get("mineable", False))),
                int(sample["input_tokens"]) if sample.get("input_tokens") is not None else None,
                int(sample["output_tokens"]) if sample.get("output_tokens") is not None else None,
                sample.get("finish_reason"),
                sample.get("template_hash"),
                json.dumps(dict(sample.get("runtime_versions") or {}), sort_keys=True, default=str),
                sample.get("score_direction"),
                (
                    float(sample["score_threshold"])
                    if sample.get("score_threshold") is not None
                    else None
                ),
                float(sample["coverage"]) if sample.get("coverage") is not None else None,
                json.dumps(dict(sample.get("metadata") or {}), sort_keys=True, default=str),
            )
            for index, sample in enumerate(samples)
        ]
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM evaluation_metrics WHERE evaluation_id = ?", (evaluation_id,)
                )
                self._conn.execute(
                    "DELETE FROM evaluation_samples WHERE evaluation_id = ?", (evaluation_id,)
                )
                self._conn.executemany(
                    """
                    INSERT INTO evaluation_metrics
                        (evaluation_id, name, value, direction, suite_item_id, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    metric_rows,
                )
                self._conn.executemany(
                    """
                    INSERT INTO evaluation_samples
                        (evaluation_id, ordinal, suite_item_id, record_id, input_json,
                         expected_json, output_json, score, passed, latency_ms, error,
                         verifier_trace_json, generation_seed, evidence_kind, valid,
                         mineable, input_tokens, output_tokens, finish_reason,
                         template_hash, runtime_versions_json, score_direction,
                         score_threshold, coverage, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?)
                    """,
                    sample_rows,
                )
                cursor = self._conn.execute(
                    """
                    UPDATE evaluations SET status = 'completed', stage = 'complete',
                        processed_samples = ?, total_samples = ?, result_json = ?,
                        artifact_path = ?, error = NULL, cancel_requested = 0,
                        completed_at = ?, updated_at = ? WHERE id = ?
                    """,
                    (
                        len(samples),
                        len(samples),
                        json.dumps(dict(result), sort_keys=True, default=str),
                        artifact_path,
                        now,
                        now,
                        evaluation_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError(f"unknown evaluation: {evaluation_id}")
        record = self.get_evaluation(evaluation_id)
        if record is None:
            raise ValueError(f"unknown evaluation: {evaluation_id}")
        return record

    def list_evaluation_metrics(self, evaluation_id: str) -> List["EvaluationMetricRecord"]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM evaluation_metrics WHERE evaluation_id = ?
                ORDER BY suite_item_id, name
                """,
                (evaluation_id,),
            ).fetchall()
        return [_row_to_evaluation_metric(row) for row in rows]

    def list_evaluation_samples(
        self, evaluation_id: str, *, limit: Optional[int] = None, offset: int = 0
    ) -> List["EvaluationSampleRecord"]:
        params: list[Any] = [evaluation_id]
        suffix = ""
        if limit is not None:
            suffix = " LIMIT ? OFFSET ?"
            params.extend((max(0, int(limit)), max(0, int(offset))))
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM evaluation_samples WHERE evaluation_id = ? ORDER BY ordinal"
                + suffix,
                params,
            ).fetchall()
        return [_row_to_evaluation_sample(row) for row in rows]

    def count_evaluation_samples(self, evaluation_id: str) -> int:
        """Return the number of persisted samples for one evaluation.

        Keep this as a separate aggregate query so paginated API consumers do
        not need to materialize every sample merely to report a total.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS total FROM evaluation_samples WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
        return int(row["total"] if row is not None else 0)

    @staticmethod
    def _evaluation_sample_pair_cte() -> str:
        """Rank duplicate logical examples without materializing either side."""

        return """
            WITH base_ranked AS (
                SELECT ordinal,
                       COALESCE(NULLIF(record_id, ''), suite_item_id) AS logical_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY COALESCE(NULLIF(record_id, ''), suite_item_id)
                           ORDER BY ordinal
                       ) AS occurrence
                FROM evaluation_samples WHERE evaluation_id = ?
            ), candidate_ranked AS (
                SELECT ordinal,
                       COALESCE(NULLIF(record_id, ''), suite_item_id) AS logical_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY COALESCE(NULLIF(record_id, ''), suite_item_id)
                           ORDER BY ordinal
                       ) AS occurrence
                FROM evaluation_samples WHERE evaluation_id = ?
            ), paired AS (
                SELECT base.logical_id, base.occurrence,
                       base.ordinal AS base_ordinal,
                       candidate.ordinal AS candidate_ordinal
                FROM base_ranked base
                LEFT JOIN candidate_ranked candidate
                  ON candidate.logical_id = base.logical_id
                 AND candidate.occurrence = base.occurrence
                UNION ALL
                SELECT candidate.logical_id, candidate.occurrence,
                       NULL AS base_ordinal,
                       candidate.ordinal AS candidate_ordinal
                FROM candidate_ranked candidate
                LEFT JOIN base_ranked base
                  ON base.logical_id = candidate.logical_id
                 AND base.occurrence = candidate.occurrence
                WHERE base.ordinal IS NULL
            )
        """

    def count_evaluation_sample_pairs(self, base_id: str, candidate_id: str) -> int:
        """Count a full record/occurrence join for two persisted evaluations."""

        with self._lock:
            row = self._conn.execute(
                self._evaluation_sample_pair_cte() + " SELECT COUNT(*) AS total FROM paired",
                (base_id, candidate_id),
            ).fetchone()
        return int(row["total"] if row is not None else 0)

    def list_evaluation_sample_pairs(
        self,
        base_id: str,
        candidate_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return one bounded page joined by logical identity and occurrence.

        ``record_id`` is the preferred identity and ``suite_item_id`` is the
        legacy fallback.  Repeated identities are paired by their ordinal
        occurrence, preventing a many-to-many join and preserving duplicates.
        The hard 1,000-row cap protects API/acquisition consumers even when a
        caller accidentally supplies an unbounded page size.
        """

        bounded_limit = min(1000, max(0, int(limit)))
        bounded_offset = max(0, int(offset))
        if bounded_limit == 0:
            return []
        with self._lock:
            key_rows = self._conn.execute(
                self._evaluation_sample_pair_cte()
                + """
                  SELECT logical_id, occurrence, base_ordinal, candidate_ordinal
                  FROM paired
                  ORDER BY logical_id, occurrence
                  LIMIT ? OFFSET ?
                """,
                (base_id, candidate_id, bounded_limit, bounded_offset),
            ).fetchall()

            def load_rows(evaluation_id: str, ordinals: Sequence[int]) -> Dict[int, Any]:
                if not ordinals:
                    return {}
                placeholders = ",".join("?" for _ in ordinals)
                rows = self._conn.execute(
                    "SELECT * FROM evaluation_samples "
                    f"WHERE evaluation_id = ? AND ordinal IN ({placeholders})",
                    (evaluation_id, *ordinals),
                ).fetchall()
                return {int(row["ordinal"]): _row_to_evaluation_sample(row) for row in rows}

            base_ordinals = [
                int(row["base_ordinal"])
                for row in key_rows
                if row["base_ordinal"] is not None
            ]
            candidate_ordinals = [
                int(row["candidate_ordinal"])
                for row in key_rows
                if row["candidate_ordinal"] is not None
            ]
            base_rows = load_rows(base_id, base_ordinals)
            candidate_rows = load_rows(candidate_id, candidate_ordinals)
        return [
            {
                "logical_record_id": str(row["logical_id"]),
                # ROW_NUMBER is 1-based; comparison/failure-mining occurrence
                # indexes are zero-based throughout the existing API.
                "occurrence": int(row["occurrence"]) - 1,
                "base": (
                    base_rows[int(row["base_ordinal"])]
                    if row["base_ordinal"] is not None
                    else None
                ),
                "candidate": (
                    candidate_rows[int(row["candidate_ordinal"])]
                    if row["candidate_ordinal"] is not None
                    else None
                ),
            }
            for row in key_rows
        ]

    # ----- Lab v3 durable workstation queue -------------------------------

    def _append_work_event_locked(
        self,
        work_item_id: str,
        event_type: str,
        *,
        attempt_id: Optional[str] = None,
        payload: Optional[Mapping[str, Any]] = None,
        created_at: Optional[str] = None,
    ) -> None:
        """Append one control-plane event inside the caller's transaction."""

        self._conn.execute(
            """
            INSERT INTO work_item_events
                (id, work_item_id, attempt_id, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                uuid.uuid4().hex,
                work_item_id,
                attempt_id,
                event_type,
                json.dumps(dict(payload or {}), sort_keys=True, default=str),
                created_at or datetime.now(timezone.utc).isoformat(),
            ),
        )

    def _refresh_dependency_states_locked(self, *, created_at: str) -> None:
        """Make failed dependencies explicit and reopen children after recovery."""

        blocked_rows = self._conn.execute("""
            SELECT DISTINCT child.id
            FROM work_items child
            JOIN work_item_dependencies dependency ON dependency.work_item_id = child.id
            JOIN work_items parent ON parent.id = dependency.depends_on_work_item_id
            WHERE child.status = 'queued'
              AND parent.status IN ('failed', 'cancelled', 'interrupted', 'needs_reconciliation')
            """).fetchall()
        for row in blocked_rows:
            identifier = str(row["id"])
            self._conn.execute(
                "UPDATE work_items SET status = 'blocked', stage = 'blocked', "
                "error = 'dependency did not complete', updated_at = ? WHERE id = ?",
                (created_at, identifier),
            )
            self._append_work_event_locked(
                identifier,
                "blocked",
                payload={"reason": "dependency did not complete"},
                created_at=created_at,
            )

        reopened_rows = self._conn.execute("""
            SELECT child.id
            FROM work_items child
            WHERE child.status = 'blocked'
              AND NOT EXISTS (
                  SELECT 1 FROM work_item_dependencies dependency
                  JOIN work_items parent
                    ON parent.id = dependency.depends_on_work_item_id
                  WHERE dependency.work_item_id = child.id
                    AND parent.status <> 'completed'
              )
            """).fetchall()
        for row in reopened_rows:
            identifier = str(row["id"])
            self._conn.execute(
                "UPDATE work_items SET status = 'queued', stage = 'queued', "
                "error = NULL, updated_at = ? WHERE id = ?",
                (created_at, identifier),
            )
            self._append_work_event_locked(
                identifier,
                "reopened",
                payload={"reason": "dependencies completed"},
                created_at=created_at,
            )

    def create_work_item(
        self,
        *,
        kind: str,
        launch_spec: Optional[Mapping[str, Any]] = None,
        resource_class: str = "accelerator",
        resource_requirements: Optional[Mapping[str, Any]] = None,
        priority: int = 0,
        domain_kind: Optional[str] = None,
        domain_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        canonical_run_id: Optional[str] = None,
        log_path: Optional[str] = None,
        dependencies: Sequence[str] = (),
        max_retries: int = 0,
        not_before: Optional[str] = None,
        work_item_id: Optional[str] = None,
    ) -> "WorkItemRecord":
        if not kind or not kind.strip():
            raise ValueError("work item kind is required")
        if resource_class not in {"accelerator", "cpu", "none"}:
            raise ValueError("resource_class must be accelerator, cpu, or none")
        identifier = work_item_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        dependency_ids = list(dict.fromkeys(str(value) for value in dependencies))
        if identifier in dependency_ids:
            raise ValueError("a work item cannot depend on itself")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if dependency_ids:
                    marks = ", ".join("?" for _ in dependency_ids)
                    rows = self._conn.execute(
                        f"SELECT id FROM work_items WHERE id IN ({marks})", dependency_ids
                    ).fetchall()
                    found = {str(row["id"]) for row in rows}
                    missing = sorted(set(dependency_ids) - found)
                    if missing:
                        raise ValueError(f"unknown work item dependencies: {', '.join(missing)}")
                self._conn.execute(
                    """
                    INSERT INTO work_items
                        (id, kind, status, stage, resource_class, priority,
                         launch_spec_json, result_json, progress_json,
                         resource_requirements_json, domain_kind, domain_id,
                         run_group_id, canonical_run_id, log_path, retry_count, max_retries,
                         cancel_requested, not_before, created_at, updated_at)
                    VALUES (?, ?, 'queued', 'queued', ?, ?, ?, '{}', '{}', ?, ?, ?, ?, ?, ?,
                            0, ?, 0, ?, ?, ?)
                    """,
                    (
                        identifier,
                        kind.strip(),
                        resource_class,
                        int(priority),
                        json.dumps(dict(launch_spec or {}), sort_keys=True, default=str),
                        json.dumps(dict(resource_requirements or {}), sort_keys=True, default=str),
                        domain_kind,
                        domain_id,
                        run_group_id,
                        canonical_run_id,
                        log_path,
                        max(0, int(max_retries)),
                        not_before,
                        now,
                        now,
                    ),
                )
                self._conn.executemany(
                    """
                    INSERT INTO work_item_dependencies
                        (work_item_id, depends_on_work_item_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    [(identifier, dependency_id, now) for dependency_id in dependency_ids],
                )
                self._append_work_event_locked(
                    identifier,
                    "queued",
                    payload={
                        "kind": kind.strip(),
                        "resource_class": resource_class,
                        "dependencies": dependency_ids,
                        "domain_kind": domain_kind,
                        "domain_id": domain_id,
                    },
                    created_at=now,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        record = self.get_work_item(identifier)
        assert record is not None
        return record

    def get_work_item(self, work_item_id: str) -> Optional["WorkItemRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM work_items WHERE id = ?", (work_item_id,)
            ).fetchone()
        return _row_to_work_item(row) if row else None

    def list_work_items(
        self,
        *,
        statuses: Optional[Sequence[str]] = None,
        kinds: Optional[Sequence[str]] = None,
        canonical_run_id: Optional[str] = None,
        worker_id: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> List["WorkItemRecord"]:
        """List work items, optionally narrowed to a single owning worker.

        ``worker_id`` defaults to ``None``, which keeps the historical
        cross-worker listing. Recovery uses the filter so a starting supervisor
        does not treat another live worker's claims as its own.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if statuses:
            marks = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({marks})")
            params.extend(str(value) for value in statuses)
        if kinds:
            marks = ", ".join("?" for _ in kinds)
            clauses.append(f"kind IN ({marks})")
            params.extend(str(value) for value in kinds)
        if canonical_run_id is not None:
            clauses.append("canonical_run_id = ?")
            params.append(canonical_run_id)
        if worker_id is not None:
            clauses.append("worker_id = ?")
            params.append(worker_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend((max(1, int(limit)), max(0, int(offset))))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM work_items {where} "
                "ORDER BY priority DESC, created_at, id LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [_row_to_work_item(row) for row in rows]

    def work_item_queue_position(self, work_item_id: str) -> Optional[int]:
        """Return the one-based durable dispatch position for a queued item."""
        with self._lock:
            target = self._conn.execute(
                "SELECT priority, created_at, id, status FROM work_items WHERE id = ?",
                (work_item_id,),
            ).fetchone()
            if target is None or str(target["status"]) != "queued":
                return None
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS ahead FROM work_items
                WHERE status = 'queued' AND (
                    priority > ? OR
                    (priority = ? AND created_at < ?) OR
                    (priority = ? AND created_at = ? AND id < ?)
                )
                """,
                (
                    target["priority"],
                    target["priority"],
                    target["created_at"],
                    target["priority"],
                    target["created_at"],
                    target["id"],
                ),
            ).fetchone()
        return int(row["ahead"]) + 1

    def work_item_blockers(self, work_item_id: str) -> Dict[str, Any]:
        """Describe unresolved dependencies and any accelerator lease blocker."""
        dependencies = self.list_work_item_dependencies(work_item_id)
        item = self.get_work_item(work_item_id)
        lease = (
            self.get_resource_lease("accelerator")
            if item is not None and item.resource_class == "accelerator"
            else None
        )
        return {
            "dependencies": [
                value.to_dict() for value in dependencies if value.dependency_status != "completed"
            ],
            "resource_lease": lease.to_dict() if lease else None,
        }

    def list_work_item_dependencies(self, work_item_id: str) -> List["WorkItemDependencyRecord"]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT d.work_item_id, d.depends_on_work_item_id, d.created_at,
                       w.status AS dependency_status
                FROM work_item_dependencies d
                JOIN work_items w ON w.id = d.depends_on_work_item_id
                WHERE d.work_item_id = ?
                ORDER BY d.created_at, d.depends_on_work_item_id
                """,
                (work_item_id,),
            ).fetchall()
        return [WorkItemDependencyRecord(**dict(row)) for row in rows]

    def add_work_item_dependency(
        self, work_item_id: str, depends_on_work_item_id: str
    ) -> "WorkItemDependencyRecord":
        if work_item_id == depends_on_work_item_id:
            raise ValueError("a work item cannot depend on itself")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                found = self._conn.execute(
                    "SELECT id FROM work_items WHERE id IN (?, ?)",
                    (work_item_id, depends_on_work_item_id),
                ).fetchall()
                if {str(row["id"]) for row in found} != {work_item_id, depends_on_work_item_id}:
                    raise ValueError("both work items must exist")
                cycle = self._conn.execute(
                    """
                    WITH RECURSIVE dependencies(id) AS (
                        SELECT depends_on_work_item_id
                        FROM work_item_dependencies WHERE work_item_id = ?
                        UNION
                        SELECT d.depends_on_work_item_id
                        FROM work_item_dependencies d
                        JOIN dependencies p ON d.work_item_id = p.id
                    )
                    SELECT 1 FROM dependencies WHERE id = ? LIMIT 1
                    """,
                    (depends_on_work_item_id, work_item_id),
                ).fetchone()
                if cycle is not None:
                    raise ValueError("work item dependency would create a cycle")
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO work_item_dependencies
                        (work_item_id, depends_on_work_item_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (work_item_id, depends_on_work_item_id, now),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        dependencies = self.list_work_item_dependencies(work_item_id)
        return next(
            value
            for value in dependencies
            if value.depends_on_work_item_id == depends_on_work_item_id
        )

    def worker_process_identity(
        self, worker_id: str
    ) -> Optional[tuple[Optional[int], Optional[float]]]:
        """Return the registered ``(pid, pid_started_at)`` identity of a worker.

        ``None`` means the worker is unknown to the control plane, which is not
        evidence that its process died. Callers deciding whether another
        worker's claim may be taken over must treat an unknown or incomplete
        identity as "still alive"; PID alone is never sufficient because the
        operating system reuses it.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT pid, pid_started_at FROM workers WHERE id = ?", (worker_id,)
            ).fetchone()
        if row is None:
            return None
        pid = row["pid"]
        pid_started_at = row["pid_started_at"]
        return (
            int(pid) if pid is not None else None,
            float(pid_started_at) if pid_started_at is not None else None,
        )

    def claim_next_work_item(
        self,
        *,
        worker_id: str,
        worker_pid: Optional[int] = None,
        worker_pid_started_at: Optional[float] = None,
        lease_ttl_seconds: int = 30,
        work_item_id: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Optional["WorkItemRecord"]:
        """Claim the next dependency-ready item in transactional priority/FIFO order."""
        if not worker_id:
            raise ValueError("worker_id is required")
        instant = now or datetime.now(timezone.utc)
        now_iso = instant.isoformat()
        expires_at = (instant + timedelta(seconds=max(1, int(lease_ttl_seconds)))).isoformat()
        claim_token = uuid.uuid4().hex
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._recover_expired_leases_locked(now_iso)
                self._refresh_dependency_states_locked(created_at=now_iso)
                clauses = [
                    "w.status = 'queued'",
                    "w.cancel_requested = 0",
                    "(w.not_before IS NULL OR w.not_before <= ?)",
                    "NOT EXISTS (SELECT 1 FROM work_item_dependencies d "
                    "JOIN work_items parent ON parent.id = d.depends_on_work_item_id "
                    "WHERE d.work_item_id = w.id AND parent.status <> 'completed')",
                    "((w.resource_class <> 'accelerator' AND "
                    "COALESCE(json_extract(w.resource_requirements_json, '$.lease_type'), '') "
                    "<> 'serving') OR NOT EXISTS "
                    "(SELECT 1 FROM resource_leases l WHERE l.resource_key = 'accelerator'))",
                    # Verifier calibrations are globally serialized even when
                    # they use hosted/programmatic CPU execution.  The check
                    # lives in the same IMMEDIATE transaction as the claim,
                    # so two dashboard workers cannot race a CPU calibration
                    # against an accelerator-backed calibration.
                    "(COALESCE(w.domain_kind, '') <> 'verifier_calibration' OR "
                    "NOT EXISTS (SELECT 1 FROM work_items active "
                    "WHERE active.domain_kind = 'verifier_calibration' "
                    "AND active.status = 'running' AND active.id <> w.id))",
                    # Reward-integrity audits are likewise serialized across
                    # local, hosted, and deterministic sentinels. This keeps
                    # same-run replay publication ordered and enforces the V8
                    # one-active-audit workstation default even for CPU work.
                    "(COALESCE(w.domain_kind, '') <> 'reward_integrity_audit' OR "
                    "NOT EXISTS (SELECT 1 FROM work_items active "
                    "WHERE active.domain_kind = 'reward_integrity_audit' "
                    "AND active.status = 'running' AND active.id <> w.id))",
                ]
                params: list[Any] = [now_iso]
                if work_item_id is not None:
                    clauses.append("w.id = ?")
                    params.append(work_item_id)
                row = self._conn.execute(
                    "SELECT w.* FROM work_items w WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY w.priority DESC, w.created_at, w.id LIMIT 1",
                    params,
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return None
                identifier = str(row["id"])
                cursor = self._conn.execute(
                    """
                    UPDATE work_items SET status = 'running', stage = 'starting',
                        worker_id = ?, worker_pid = ?, worker_pid_started_at = ?,
                        claim_token = ?, heartbeat_at = ?, error = NULL,
                        started_at = COALESCE(started_at, ?), updated_at = ?
                    WHERE id = ? AND status = 'queued' AND cancel_requested = 0
                    """,
                    (
                        worker_id,
                        worker_pid,
                        worker_pid_started_at,
                        claim_token,
                        now_iso,
                        now_iso,
                        now_iso,
                        identifier,
                    ),
                )
                if cursor.rowcount != 1:
                    self._conn.rollback()
                    return None
                current = self._conn.execute(
                    "SELECT worker_id FROM work_items WHERE id = ?", (work_item_id,)
                ).fetchone()
                if current is not None and current["worker_id"]:
                    self._conn.execute(
                        "UPDATE workers SET status = 'online', heartbeat_at = ? WHERE id = ?",
                        (now_iso, current["worker_id"]),
                    )
                self._conn.execute(
                    """
                    INSERT INTO workers
                        (id, status, pid, pid_started_at, capabilities_json,
                         metadata_json, started_at, heartbeat_at)
                    VALUES (?, 'online', ?, ?, '{}', '{}', ?, ?)
                    ON CONFLICT(id) DO UPDATE SET status = 'online',
                        pid = excluded.pid,
                        pid_started_at = excluded.pid_started_at,
                        heartbeat_at = excluded.heartbeat_at,
                        stopped_at = NULL
                    """,
                    (worker_id, worker_pid, worker_pid_started_at, now_iso, now_iso),
                )
                ordinal_row = self._conn.execute(
                    "SELECT COALESCE(MAX(ordinal), 0) AS value "
                    "FROM work_item_attempts WHERE work_item_id = ?",
                    (identifier,),
                ).fetchone()
                attempt_ordinal = int(ordinal_row["value"]) + 1
                attempt_id = f"{identifier}-attempt-{attempt_ordinal}"
                database_root = (
                    Path(self.path).expanduser().parent
                    if self.path and self.path != ":memory:"
                    else Path.home() / ".halo-forge"
                )
                attempt_output_dir = str(
                    database_root / "work-items" / identifier / f"attempt-{attempt_ordinal}"
                )
                self._conn.execute(
                    """
                    INSERT INTO work_item_attempts
                        (id, work_item_id, ordinal, status, worker_id, worker_pid,
                         worker_pid_started_at, claim_token, output_dir, result_json,
                         created_at, started_at)
                    VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, '{}', ?, ?)
                    """,
                    (
                        attempt_id,
                        identifier,
                        attempt_ordinal,
                        worker_id,
                        worker_pid,
                        worker_pid_started_at,
                        claim_token,
                        attempt_output_dir,
                        now_iso,
                        now_iso,
                    ),
                )
                self._append_work_event_locked(
                    identifier,
                    "claimed",
                    attempt_id=attempt_id,
                    payload={
                        "worker_id": worker_id,
                        "attempt": attempt_ordinal,
                        "output_dir": attempt_output_dir,
                    },
                    created_at=now_iso,
                )
                if str(row["resource_class"]) == "accelerator":
                    self._conn.execute(
                        """
                        INSERT INTO resource_leases
                            (resource_key, holder_type, holder_id, work_item_id,
                             lease_token, retained, acquired_at, heartbeat_at,
                             expires_at, holder_pid, holder_pid_started_at, metadata_json)
                        VALUES ('accelerator', 'work_item', ?, ?, ?, 0, ?, ?, ?, ?, ?, '{}')
                        """,
                        (
                            worker_id,
                            identifier,
                            claim_token,
                            now_iso,
                            now_iso,
                            expires_at,
                            worker_pid,
                            worker_pid_started_at,
                        ),
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_work_item(identifier)

    def heartbeat_work_item(
        self,
        work_item_id: str,
        *,
        claim_token: str,
        stage: Optional[str] = None,
        progress: Optional[Mapping[str, Any]] = None,
        lease_ttl_seconds: int = 30,
        now: Optional[datetime] = None,
    ) -> Optional["WorkItemRecord"]:
        instant = now or datetime.now(timezone.utc)
        now_iso = instant.isoformat()
        expires_at = (instant + timedelta(seconds=max(1, int(lease_ttl_seconds)))).isoformat()
        assignments = ["heartbeat_at = ?", "updated_at = ?"]
        params: list[Any] = [now_iso, now_iso]
        if stage is not None:
            assignments.append("stage = ?")
            params.append(stage)
        if progress is not None:
            assignments.append("progress_json = ?")
            params.append(json.dumps(dict(progress), sort_keys=True, default=str))
        params.extend((work_item_id, claim_token))
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                cursor = self._conn.execute(
                    f"UPDATE work_items SET {', '.join(assignments)} "
                    "WHERE id = ? AND claim_token = ? AND status = 'running'",
                    params,
                )
                if cursor.rowcount != 1:
                    self._conn.rollback()
                    return None
                self._conn.execute(
                    """
                    UPDATE resource_leases SET heartbeat_at = ?, expires_at = ?
                    WHERE work_item_id = ? AND lease_token = ? AND retained = 0
                    """,
                    (now_iso, expires_at, work_item_id, claim_token),
                )
                attempt = self._conn.execute(
                    "SELECT id FROM work_item_attempts "
                    "WHERE work_item_id = ? AND claim_token = ? AND status = 'running' "
                    "ORDER BY ordinal DESC LIMIT 1",
                    (work_item_id, claim_token),
                ).fetchone()
                self._append_work_event_locked(
                    work_item_id,
                    "progress",
                    attempt_id=str(attempt["id"]) if attempt else None,
                    payload={"stage": stage, "progress": dict(progress or {})},
                    created_at=now_iso,
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_work_item(work_item_id)

    def update_work_attempt_output_dir(
        self,
        attempt_id: str,
        output_dir: str,
    ) -> Optional["WorkAttemptRecord"]:
        """Point an active attempt at its actual isolated staging directory.

        Work items normally stage below the database root. Managed training
        needs its attempt directory beside the selected run destination so a
        verified run can be published with one same-filesystem rename. The
        attempt remains immutable after it reaches a terminal state.
        """

        resolved = str(Path(output_dir).expanduser())
        if not resolved:
            raise ValueError("attempt output_dir is required")
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE work_item_attempts SET output_dir = ? "
                "WHERE id = ? AND status = 'running'",
                (resolved, attempt_id),
            )
            self._conn.commit()
        if not cursor.rowcount:
            return None
        row = self._conn.execute(
            "SELECT * FROM work_item_attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
        if row is None:
            return None
        from halo_forge.run_db.v4 import WorkAttemptRecord

        return WorkAttemptRecord(**dict(row))

    def bind_work_item_process(
        self,
        work_item_id: str,
        *,
        claim_token: str,
        worker_pid: int,
        worker_pid_started_at: float,
    ) -> Optional["WorkItemRecord"]:
        """Attach the spawned child identity after a claim and before monitoring."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                cursor = self._conn.execute(
                    """
                    UPDATE work_items SET worker_pid = ?, worker_pid_started_at = ?,
                        updated_at = ?
                    WHERE id = ? AND claim_token = ? AND status = 'running'
                    """,
                    (
                        int(worker_pid),
                        float(worker_pid_started_at),
                        now,
                        work_item_id,
                        claim_token,
                    ),
                )
                if cursor.rowcount:
                    self._conn.execute(
                        "UPDATE work_item_attempts SET worker_pid = ?, "
                        "worker_pid_started_at = ? WHERE work_item_id = ? "
                        "AND claim_token = ? AND status = 'running'",
                        (
                            int(worker_pid),
                            float(worker_pid_started_at),
                            work_item_id,
                            claim_token,
                        ),
                    )
                    self._conn.execute(
                        "UPDATE resource_leases SET holder_pid = ?, "
                        "holder_pid_started_at = ? WHERE work_item_id = ? "
                        "AND lease_token = ?",
                        (
                            int(worker_pid),
                            float(worker_pid_started_at),
                            work_item_id,
                            claim_token,
                        ),
                    )
                    attempt = self._conn.execute(
                        "SELECT id FROM work_item_attempts WHERE work_item_id = ? "
                        "AND claim_token = ? AND status = 'running' "
                        "ORDER BY ordinal DESC LIMIT 1",
                        (work_item_id, claim_token),
                    ).fetchone()
                    self._append_work_event_locked(
                        work_item_id,
                        "process_bound",
                        attempt_id=str(attempt["id"]) if attempt else None,
                        payload={
                            "pid": int(worker_pid),
                            "pid_started_at": float(worker_pid_started_at),
                        },
                        created_at=now,
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_work_item(work_item_id) if cursor.rowcount else None

    def update_work_item(
        self,
        work_item_id: str,
        *,
        stage: Optional[str] = None,
        progress: Optional[Mapping[str, Any]] = None,
        priority: Optional[int] = None,
        log_path: Optional[str] = None,
    ) -> Optional["WorkItemRecord"]:
        assignments = ["updated_at = ?"]
        params: list[Any] = [datetime.now(timezone.utc).isoformat()]
        for column, value in (("stage", stage), ("priority", priority), ("log_path", log_path)):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        if progress is not None:
            assignments.append("progress_json = ?")
            params.append(json.dumps(dict(progress), sort_keys=True, default=str))
        params.append(work_item_id)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE work_items SET {', '.join(assignments)} WHERE id = ?", params
            )
            self._conn.commit()
        return self.get_work_item(work_item_id) if cursor.rowcount else None

    def request_cancel_work_item(self, work_item_id: str) -> Optional["WorkItemRecord"]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT status FROM work_items WHERE id = ?", (work_item_id,)
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return None
                status = str(row["status"])
                if status in {"queued", "blocked", "interrupted", "needs_reconciliation"}:
                    self._conn.execute(
                        """
                        UPDATE work_items SET status = 'cancelled', stage = 'cancelled',
                            cancel_requested = 1, completed_at = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (now, now, work_item_id),
                    )
                    self._conn.execute(
                        "DELETE FROM resource_leases WHERE work_item_id = ?", (work_item_id,)
                    )
                elif status == "running":
                    self._conn.execute(
                        "UPDATE work_items SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                        (now, work_item_id),
                    )
                if status in {
                    "queued",
                    "blocked",
                    "interrupted",
                    "needs_reconciliation",
                    "running",
                }:
                    self._append_work_event_locked(
                        work_item_id,
                        "cancel_requested" if status == "running" else "cancelled",
                        payload={"previous_status": status},
                        created_at=now,
                    )
                    self._refresh_dependency_states_locked(created_at=now)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_work_item(work_item_id)

    def block_claimed_work_item(
        self,
        work_item_id: str,
        *,
        claim_token: str,
        reason: str,
        details: Optional[Mapping[str, Any]] = None,
    ) -> Optional["WorkItemRecord"]:
        """Release a just-claimed item when workstation preflight refuses it."""

        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                attempt = self._conn.execute(
                    "SELECT id FROM work_item_attempts WHERE work_item_id = ? "
                    "AND claim_token = ? AND status = 'running' "
                    "ORDER BY ordinal DESC LIMIT 1",
                    (work_item_id, claim_token),
                ).fetchone()
                cursor = self._conn.execute(
                    """
                    UPDATE work_items SET status = 'blocked', stage = 'blocked_capacity',
                        error = ?, result_json = ?, worker_id = NULL, worker_pid = NULL,
                        worker_pid_started_at = NULL, claim_token = NULL,
                        heartbeat_at = NULL, updated_at = ?
                    WHERE id = ? AND claim_token = ? AND status = 'running'
                    """,
                    (
                        reason,
                        json.dumps(dict(details or {}), sort_keys=True, default=str),
                        now,
                        work_item_id,
                        claim_token,
                    ),
                )
                if cursor.rowcount:
                    self._conn.execute(
                        "DELETE FROM resource_leases WHERE work_item_id = ? AND lease_token = ?",
                        (work_item_id, claim_token),
                    )
                    if attempt is not None:
                        self._conn.execute(
                            "UPDATE work_item_attempts SET status = 'blocked', error = ?, "
                            "result_json = ?, completed_at = ? WHERE id = ?",
                            (
                                reason,
                                json.dumps(dict(details or {}), sort_keys=True, default=str),
                                now,
                                attempt["id"],
                            ),
                        )
                    self._append_work_event_locked(
                        work_item_id,
                        "blocked",
                        attempt_id=str(attempt["id"]) if attempt else None,
                        payload={"reason": reason, "details": dict(details or {})},
                        created_at=now,
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_work_item(work_item_id) if cursor.rowcount else None

    def defer_claimed_work_item_for_accelerator(
        self,
        work_item_id: str,
        *,
        claim_token: str,
        reason: str,
        details: Optional[Mapping[str, Any]] = None,
        not_before: Optional[str] = None,
    ) -> Optional["WorkItemRecord"]:
        """Return a claim to the queue without consuming a retry.

        External accelerator owners are expected workstation activity, not a
        failed Halo Forge attempt. The abandoned claim is retained as a
        ``waiting`` attempt for auditability while the work item becomes
        queue-eligible again after ``not_before``.
        """

        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                attempt = self._conn.execute(
                    "SELECT id FROM work_item_attempts WHERE work_item_id=? "
                    "AND claim_token=? AND status='running' ORDER BY ordinal DESC LIMIT 1",
                    (work_item_id, claim_token),
                ).fetchone()
                cursor = self._conn.execute(
                    """UPDATE work_items SET status='queued',stage='waiting_for_accelerator',
                       error=?,result_json=?,not_before=?,worker_id=NULL,worker_pid=NULL,
                       worker_pid_started_at=NULL,claim_token=NULL,heartbeat_at=NULL,updated_at=?
                       WHERE id=? AND claim_token=? AND status='running'""",
                    (
                        reason,
                        json.dumps(dict(details or {}), sort_keys=True, default=str),
                        not_before,
                        now,
                        work_item_id,
                        claim_token,
                    ),
                )
                if cursor.rowcount:
                    self._conn.execute(
                        "DELETE FROM resource_leases WHERE work_item_id=? AND lease_token=?",
                        (work_item_id, claim_token),
                    )
                    if attempt is not None:
                        self._conn.execute(
                            "UPDATE work_item_attempts SET status='waiting',error=?,result_json=?,completed_at=? WHERE id=?",
                            (
                                reason,
                                json.dumps(dict(details or {}), sort_keys=True, default=str),
                                now,
                                attempt["id"],
                            ),
                        )
                    self._append_work_event_locked(
                        work_item_id,
                        "waiting_for_accelerator",
                        attempt_id=str(attempt["id"]) if attempt else None,
                        payload={"reason": reason, "details": dict(details or {})},
                        created_at=now,
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_work_item(work_item_id) if cursor.rowcount else None

    def finish_work_item(
        self,
        work_item_id: str,
        *,
        claim_token: str,
        result: Optional[Mapping[str, Any]] = None,
        error: Optional[str] = None,
        ignore_late_cancel: bool = False,
    ) -> Optional["WorkItemRecord"]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    """
                    SELECT cancel_requested, worker_id FROM work_items
                    WHERE id = ? AND claim_token = ? AND status = 'running'
                    """,
                    (work_item_id, claim_token),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return None
                late_cancel_ignored = bool(row["cancel_requested"]) and ignore_late_cancel
                if bool(row["cancel_requested"]) and not ignore_late_cancel:
                    status, stage = "cancelled", "cancelled"
                elif error is not None:
                    status, stage = "failed", "failed"
                else:
                    status, stage = "completed", "complete"
                self._conn.execute(
                    """
                    UPDATE work_items SET status = ?, stage = ?, result_json = ?,
                        error = ?, worker_id = NULL, worker_pid = NULL,
                        worker_pid_started_at = NULL, claim_token = NULL,
                        heartbeat_at = NULL, cancel_requested = ?,
                        completed_at = ?, updated_at = ?
                    WHERE id = ? AND claim_token = ?
                    """,
                    (
                        status,
                        stage,
                        json.dumps(dict(result or {}), sort_keys=True, default=str),
                        error,
                        0 if late_cancel_ignored else int(bool(row["cancel_requested"])),
                        now,
                        now,
                        work_item_id,
                        claim_token,
                    ),
                )
                self._conn.execute(
                    "DELETE FROM resource_leases WHERE work_item_id = ? AND lease_token = ?",
                    (work_item_id, claim_token),
                )
                attempt = self._conn.execute(
                    "SELECT id FROM work_item_attempts WHERE work_item_id = ? "
                    "AND claim_token = ? AND status = 'running' "
                    "ORDER BY ordinal DESC LIMIT 1",
                    (work_item_id, claim_token),
                ).fetchone()
                if attempt is not None:
                    self._conn.execute(
                        "UPDATE work_item_attempts SET status = ?, result_json = ?, "
                        "error = ?, completed_at = ? WHERE id = ?",
                        (
                            status,
                            json.dumps(dict(result or {}), sort_keys=True, default=str),
                            error,
                            now,
                            attempt["id"],
                        ),
                    )
                if row["worker_id"]:
                    self._conn.execute(
                        "UPDATE workers SET heartbeat_at = ? WHERE id = ?",
                        (now, row["worker_id"]),
                    )
                self._append_work_event_locked(
                    work_item_id,
                    status,
                    attempt_id=str(attempt["id"]) if attempt else None,
                    payload={
                        "result": dict(result or {}),
                        "error": error,
                        "late_cancel_ignored": late_cancel_ignored,
                    },
                    created_at=now,
                )
                self._refresh_dependency_states_locked(created_at=now)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_work_item(work_item_id)

    def retry_work_item(
        self,
        work_item_id: str,
        *,
        force: bool = True,
        reason: str = "operator requested retry",
        backoff_seconds: Optional[float] = None,
    ) -> Optional["WorkItemRecord"]:
        """Queue another isolated attempt while retaining immutable history.

        Automatic retries (``force=False``) obey ``max_retries`` and receive
        exponential backoff. Operator-forced retries may exceed that limit but
        always carry a durable reason.
        """

        if force and not str(reason or "").strip():
            raise ValueError("operator-forced retry requires a reason")
        instant = datetime.now(timezone.utc)
        now = instant.isoformat()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT status, retry_count, max_retries FROM work_items WHERE id = ?",
                    (work_item_id,),
                ).fetchone()
                if row is None or str(row["status"]) not in {
                    "failed",
                    "interrupted",
                    "cancelled",
                    "needs_reconciliation",
                    "blocked",
                }:
                    self._conn.rollback()
                    return None
                retry_count = int(row["retry_count"])
                max_retries = int(row["max_retries"])
                if not force and retry_count >= max_retries:
                    self._conn.rollback()
                    return None
                delay = (
                    max(0.0, float(backoff_seconds))
                    if backoff_seconds is not None
                    else (min(300.0, float(2**retry_count)) if not force else 0.0)
                )
                not_before = (instant + timedelta(seconds=delay)).isoformat() if delay > 0 else None
                cursor = self._conn.execute(
                    """
                    UPDATE work_items SET status = 'queued', stage = 'queued',
                        retry_count = retry_count + 1, cancel_requested = 0,
                        result_json = '{}', progress_json = '{}', error = NULL,
                        worker_id = NULL, worker_pid = NULL,
                        worker_pid_started_at = NULL, claim_token = NULL,
                        heartbeat_at = NULL, started_at = NULL, completed_at = NULL,
                        not_before = ?, updated_at = ?
                    WHERE id = ? AND status IN (
                        'failed', 'interrupted', 'cancelled', 'needs_reconciliation', 'blocked'
                    )
                    """,
                    (not_before, now, work_item_id),
                )
                if cursor.rowcount:
                    self._conn.execute(
                        "DELETE FROM resource_leases WHERE work_item_id = ?", (work_item_id,)
                    )
                    self._append_work_event_locked(
                        work_item_id,
                        "retry_queued",
                        payload={
                            "forced": bool(force),
                            "reason": str(reason or "automatic retry"),
                            "retry_count": retry_count + 1,
                            "max_retries": max_retries,
                            "backoff_seconds": delay,
                            "not_before": not_before,
                        },
                        created_at=now,
                    )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_work_item(work_item_id) if cursor.rowcount else None

    def interrupt_work_item_if_claimed(
        self,
        work_item_id: str,
        *,
        claim_token: str,
        error: str = "worker process is no longer alive",
        status: str = "interrupted",
    ) -> Optional["WorkItemRecord"]:
        """Interrupt exactly one still-owned item without racing a new claim."""
        if status not in {"interrupted", "needs_reconciliation"}:
            raise ValueError("recovery status must be interrupted or needs_reconciliation")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                cursor = self._conn.execute(
                    """
                    UPDATE work_items SET status = ?, stage = ?,
                        error = ?, worker_id = NULL, worker_pid = NULL,
                        worker_pid_started_at = NULL, claim_token = NULL,
                        heartbeat_at = NULL, updated_at = ?
                    WHERE id = ? AND claim_token = ? AND status = 'running'
                    """,
                    (status, status, error, now, work_item_id, claim_token),
                )
                if cursor.rowcount:
                    self._conn.execute(
                        "DELETE FROM resource_leases WHERE work_item_id = ? AND lease_token = ?",
                        (work_item_id, claim_token),
                    )
                    attempt = self._conn.execute(
                        "SELECT id FROM work_item_attempts WHERE work_item_id = ? "
                        "AND claim_token = ? AND status = 'running' "
                        "ORDER BY ordinal DESC LIMIT 1",
                        (work_item_id, claim_token),
                    ).fetchone()
                    if attempt is not None:
                        self._conn.execute(
                            "UPDATE work_item_attempts SET status = ?, error = ?, "
                            "completed_at = ? WHERE id = ?",
                            (status, error, now, attempt["id"]),
                        )
                    self._append_work_event_locked(
                        work_item_id,
                        status,
                        attempt_id=str(attempt["id"]) if attempt else None,
                        payload={"error": error},
                        created_at=now,
                    )
                    self._refresh_dependency_states_locked(created_at=now)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_work_item(work_item_id) if cursor.rowcount else None

    def recover_stale_work_items(
        self,
        *,
        stale_before: Optional[datetime] = None,
        now: Optional[datetime] = None,
    ) -> List["WorkItemRecord"]:
        instant = now or datetime.now(timezone.utc)
        now_iso = instant.isoformat()
        cutoff = (stale_before or (instant - timedelta(seconds=30))).isoformat()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                expired_ids = self._recover_expired_leases_locked(now_iso)
                stale_rows = self._conn.execute(
                    """
                    SELECT id FROM work_items
                    WHERE status = 'running' AND heartbeat_at IS NOT NULL
                          AND heartbeat_at <= ?
                    """,
                    (cutoff,),
                ).fetchall()
                stale_ids = [str(row["id"]) for row in stale_rows]
                all_ids = list(dict.fromkeys(expired_ids + stale_ids))
                for identifier in stale_ids:
                    attempt = self._conn.execute(
                        "SELECT id FROM work_item_attempts WHERE work_item_id = ? "
                        "AND status = 'running' ORDER BY ordinal DESC LIMIT 1",
                        (identifier,),
                    ).fetchone()
                    self._conn.execute(
                        """
                        UPDATE work_items SET status = 'interrupted', stage = 'interrupted',
                            error = 'worker heartbeat expired', worker_id = NULL,
                            worker_pid = NULL, worker_pid_started_at = NULL,
                            claim_token = NULL, heartbeat_at = NULL, updated_at = ?
                        WHERE id = ? AND status = 'running'
                        """,
                        (now_iso, identifier),
                    )
                    self._conn.execute(
                        "DELETE FROM resource_leases WHERE work_item_id = ? AND retained = 0",
                        (identifier,),
                    )
                    if attempt is not None:
                        self._conn.execute(
                            "UPDATE work_item_attempts SET status = 'interrupted', "
                            "error = 'worker heartbeat expired', completed_at = ? WHERE id = ?",
                            (now_iso, attempt["id"]),
                        )
                    self._append_work_event_locked(
                        identifier,
                        "interrupted",
                        attempt_id=str(attempt["id"]) if attempt else None,
                        payload={"error": "worker heartbeat expired"},
                        created_at=now_iso,
                    )
                self._refresh_dependency_states_locked(created_at=now_iso)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return [record for value in all_ids if (record := self.get_work_item(value))]

    def _recover_expired_leases_locked(self, now_iso: str) -> List[str]:
        rows = self._conn.execute(
            """
            SELECT work_item_id, lease_token FROM resource_leases
            WHERE retained = 0 AND expires_at IS NOT NULL AND expires_at <= ?
            """,
            (now_iso,),
        ).fetchall()
        identifiers = [str(row["work_item_id"]) for row in rows if row["work_item_id"]]
        for row in rows:
            if not row["work_item_id"]:
                continue
            identifier = str(row["work_item_id"])
            attempt = self._conn.execute(
                "SELECT id FROM work_item_attempts WHERE work_item_id = ? "
                "AND claim_token = ? AND status = 'running' "
                "ORDER BY ordinal DESC LIMIT 1",
                (identifier, row["lease_token"]),
            ).fetchone()
            self._conn.execute(
                """
                UPDATE work_items SET status = 'interrupted', stage = 'interrupted',
                    error = 'resource lease expired', worker_id = NULL,
                    worker_pid = NULL, worker_pid_started_at = NULL,
                    claim_token = NULL, heartbeat_at = NULL, updated_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (now_iso, identifier),
            )
            if attempt is not None:
                self._conn.execute(
                    "UPDATE work_item_attempts SET status = 'interrupted', "
                    "error = 'resource lease expired', completed_at = ? WHERE id = ?",
                    (now_iso, attempt["id"]),
                )
            self._append_work_event_locked(
                identifier,
                "interrupted",
                attempt_id=str(attempt["id"]) if attempt else None,
                payload={"error": "resource lease expired"},
                created_at=now_iso,
            )
        self._conn.execute(
            """
            DELETE FROM resource_leases
            WHERE retained = 0 AND expires_at IS NOT NULL AND expires_at <= ?
            """,
            (now_iso,),
        )
        return identifiers

    def acquire_serving_lease(
        self,
        *,
        holder_id: str,
        resource_key: str = "accelerator",
        metadata: Optional[Mapping[str, Any]] = None,
        holder_pid: Optional[int] = None,
        holder_pid_started_at: Optional[float] = None,
    ) -> Optional["ResourceLeaseRecord"]:
        if not holder_id:
            raise ValueError("holder_id is required")
        now = datetime.now(timezone.utc).isoformat()
        token = uuid.uuid4().hex
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._recover_expired_leases_locked(now)
                existing = self._conn.execute(
                    "SELECT * FROM resource_leases WHERE resource_key = ?", (resource_key,)
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing["holder_type"]) == "serving"
                        and str(existing["holder_id"]) == holder_id
                    ):
                        self._conn.execute(
                            "UPDATE resource_leases SET heartbeat_at = ?, holder_pid = ?, "
                            "holder_pid_started_at = ?, metadata_json = ? "
                            "WHERE resource_key = ?",
                            (
                                now,
                                holder_pid,
                                holder_pid_started_at,
                                json.dumps(dict(metadata or {}), sort_keys=True, default=str),
                                resource_key,
                            ),
                        )
                        self._conn.commit()
                        refreshed = self._conn.execute(
                            "SELECT * FROM resource_leases WHERE resource_key = ?",
                            (resource_key,),
                        ).fetchone()
                        return _row_to_resource_lease(refreshed) if refreshed else None
                    self._conn.rollback()
                    return None
                self._conn.execute(
                    """
                    INSERT INTO resource_leases
                        (resource_key, holder_type, holder_id, work_item_id,
                         lease_token, retained, acquired_at, heartbeat_at,
                         expires_at, holder_pid, holder_pid_started_at, metadata_json)
                    VALUES (?, 'serving', ?, NULL, ?, 1, ?, ?, NULL, ?, ?, ?)
                    """,
                    (
                        resource_key,
                        holder_id,
                        token,
                        now,
                        now,
                        holder_pid,
                        holder_pid_started_at,
                        json.dumps(dict(metadata or {}), sort_keys=True, default=str),
                    ),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get_resource_lease(resource_key)

    def heartbeat_serving_lease(
        self,
        *,
        holder_id: str,
        resource_key: str = "accelerator",
        holder_pid: Optional[int] = None,
        holder_pid_started_at: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Optional["ResourceLeaseRecord"]:
        """Refresh a serving lease without changing its immutable lease token."""

        now = datetime.now(timezone.utc).isoformat()
        assignments = ["heartbeat_at = ?"]
        params: list[Any] = [now]
        if holder_pid is not None:
            assignments.append("holder_pid = ?")
            params.append(int(holder_pid))
        if holder_pid_started_at is not None:
            assignments.append("holder_pid_started_at = ?")
            params.append(float(holder_pid_started_at))
        if metadata is not None:
            assignments.append("metadata_json = ?")
            params.append(json.dumps(dict(metadata), sort_keys=True, default=str))
        params.extend((resource_key, holder_id))
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE resource_leases SET {', '.join(assignments)} "
                "WHERE resource_key = ? AND holder_type = 'serving' AND holder_id = ?",
                params,
            )
            self._conn.commit()
        return self.get_resource_lease(resource_key) if cursor.rowcount else None

    def get_resource_lease(self, resource_key: str) -> Optional["ResourceLeaseRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM resource_leases WHERE resource_key = ?", (resource_key,)
            ).fetchone()
        return _row_to_resource_lease(row) if row else None

    def list_resource_leases(self) -> List["ResourceLeaseRecord"]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM resource_leases ORDER BY resource_key"
            ).fetchall()
        return [_row_to_resource_lease(row) for row in rows]

    def release_serving_lease(self, *, holder_id: str, resource_key: str = "accelerator") -> bool:
        with self._lock:
            cursor = self._conn.execute(
                """
                DELETE FROM resource_leases
                WHERE resource_key = ? AND holder_type = 'serving' AND holder_id = ?
                """,
                (resource_key, holder_id),
            )
            self._conn.commit()
        return cursor.rowcount == 1

    # ----- Lab v3 run groups and model artifacts --------------------------

    def create_run_group(
        self,
        *,
        name: str,
        kind: str,
        trainer_mode: str,
        resolved_launch_config: Mapping[str, Any],
        dataset_bindings: Sequence[Mapping[str, Any]] = (),
        base_subject: Optional[Mapping[str, Any]] = None,
        development_suite_revision_id: Optional[str] = None,
        holdout_suite_revision_id: Optional[str] = None,
        search_space: Optional[Mapping[str, Any]] = None,
        seeds: Sequence[int] = (),
        budgets: Optional[Mapping[str, Any]] = None,
        sampler_state: Optional[Mapping[str, Any]] = None,
        pruning_policy: Optional[Mapping[str, Any]] = None,
        checkpoint_policy_revision_id: Optional[str] = None,
        resolved_checkpoint_plan: Optional[Mapping[str, Any]] = None,
        parent_group_id: Optional[str] = None,
        status: str = "draft",
        run_group_id: Optional[str] = None,
    ) -> "RunGroupRecord":
        if kind not in {"repeat", "sweep"}:
            raise ValueError("run group kind must be repeat or sweep")
        if not name.strip() or not trainer_mode.strip():
            raise ValueError("run group name and trainer_mode are required")
        identifier = run_group_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO run_groups
                    (id, name, kind, status, trainer_mode,
                     resolved_launch_config_json, dataset_bindings_json,
                     base_subject_json, development_suite_revision_id,
                     holdout_suite_revision_id, search_space_json, seeds_json,
                     budgets_json, sampler_state_json, pruning_policy_json,
                     checkpoint_policy_revision_id, resolved_checkpoint_plan_json,
                     parent_group_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    name.strip(),
                    kind,
                    status,
                    trainer_mode.strip(),
                    json.dumps(dict(resolved_launch_config), sort_keys=True, default=str),
                    json.dumps(
                        [dict(value) for value in dataset_bindings], sort_keys=True, default=str
                    ),
                    json.dumps(dict(base_subject or {}), sort_keys=True, default=str),
                    development_suite_revision_id,
                    holdout_suite_revision_id,
                    json.dumps(dict(search_space or {}), sort_keys=True, default=str),
                    json.dumps([int(value) for value in seeds]),
                    json.dumps(dict(budgets or {}), sort_keys=True, default=str),
                    json.dumps(dict(sampler_state or {}), sort_keys=True, default=str),
                    json.dumps(dict(pruning_policy or {}), sort_keys=True, default=str),
                    checkpoint_policy_revision_id,
                    json.dumps(dict(resolved_checkpoint_plan or {}), sort_keys=True, default=str),
                    parent_group_id,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        record = self.get_run_group(identifier)
        assert record is not None
        return record

    def get_run_group(self, run_group_id: str) -> Optional["RunGroupRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM run_groups WHERE id = ?", (run_group_id,)
            ).fetchone()
        return _row_to_run_group(row) if row else None

    def list_run_groups(
        self,
        *,
        kind: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List["RunGroupRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM run_groups {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_run_group(row) for row in rows]

    def update_run_group(
        self,
        run_group_id: str,
        *,
        status: Optional[str] = None,
        sampler_state: Optional[Mapping[str, Any]] = None,
        pruning_policy: Optional[Mapping[str, Any]] = None,
        checkpoint_policy_revision_id: Optional[str] = None,
        resolved_checkpoint_plan: Optional[Mapping[str, Any]] = None,
    ) -> Optional["RunGroupRecord"]:
        assignments = ["updated_at = ?"]
        params: list[Any] = [datetime.now(timezone.utc).isoformat()]
        if status is not None:
            assignments.append("status = ?")
            params.append(status)
        if sampler_state is not None:
            assignments.append("sampler_state_json = ?")
            params.append(json.dumps(dict(sampler_state), sort_keys=True, default=str))
        if pruning_policy is not None:
            assignments.append("pruning_policy_json = ?")
            params.append(json.dumps(dict(pruning_policy), sort_keys=True, default=str))
        if checkpoint_policy_revision_id is not None:
            assignments.append("checkpoint_policy_revision_id = ?")
            params.append(checkpoint_policy_revision_id)
        if resolved_checkpoint_plan is not None:
            assignments.append("resolved_checkpoint_plan_json = ?")
            params.append(json.dumps(dict(resolved_checkpoint_plan), sort_keys=True, default=str))
        params.append(run_group_id)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE run_groups SET {', '.join(assignments)} WHERE id = ?", params
            )
            self._conn.commit()
        return self.get_run_group(run_group_id) if cursor.rowcount else None

    def create_run_group_trial(
        self,
        *,
        run_group_id: str,
        ordinal: int,
        config_hash: str,
        sampled_config: Mapping[str, Any],
        required_seed_count: int = 1,
        status: str = "queued",
        trial_id: Optional[str] = None,
    ) -> "RunGroupTrialRecord":
        identifier = trial_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO run_group_trials
                    (id, run_group_id, ordinal, config_hash, sampled_config_json,
                     status, seed_coverage, required_seed_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    identifier,
                    run_group_id,
                    int(ordinal),
                    config_hash,
                    json.dumps(dict(sampled_config), sort_keys=True, default=str),
                    status,
                    max(1, int(required_seed_count)),
                    now,
                    now,
                ),
            )
            self._conn.commit()
        record = self.get_run_group_trial(identifier)
        assert record is not None
        return record

    def get_run_group_trial(self, trial_id: str) -> Optional["RunGroupTrialRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM run_group_trials WHERE id = ?", (trial_id,)
            ).fetchone()
        return _row_to_run_group_trial(row) if row else None

    def list_run_group_trials(self, run_group_id: str) -> List["RunGroupTrialRecord"]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM run_group_trials WHERE run_group_id = ? ORDER BY ordinal",
                (run_group_id,),
            ).fetchall()
        return [_row_to_run_group_trial(row) for row in rows]

    def update_run_group_trial(
        self,
        trial_id: str,
        *,
        status: Optional[str] = None,
        objective_metric: Optional[str] = None,
        objective_direction: Optional[str] = None,
        objective_value: Optional[float] = None,
        seed_coverage: Optional[int] = None,
    ) -> Optional["RunGroupTrialRecord"]:
        if objective_direction is not None and objective_direction not in {"maximize", "minimize"}:
            raise ValueError("objective_direction must be maximize or minimize")
        assignments = ["updated_at = ?"]
        params: list[Any] = [datetime.now(timezone.utc).isoformat()]
        for column, value in (
            ("status", status),
            ("objective_metric", objective_metric),
            ("objective_direction", objective_direction),
            ("objective_value", objective_value),
            ("seed_coverage", seed_coverage),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        params.append(trial_id)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE run_group_trials SET {', '.join(assignments)} WHERE id = ?", params
            )
            self._conn.commit()
        return self.get_run_group_trial(trial_id) if cursor.rowcount else None

    def create_trial_run(
        self,
        *,
        trial_id: str,
        run_id: str,
        ordinal: int,
        seed: int,
        work_item_id: Optional[str] = None,
        status: str = "queued",
        trial_run_id: Optional[str] = None,
    ) -> "TrialRunRecord":
        identifier = trial_run_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO trial_runs
                    (id, trial_id, run_id, ordinal, seed, status, work_item_id,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    trial_id,
                    run_id,
                    int(ordinal),
                    int(seed),
                    status,
                    work_item_id,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        record = self.get_trial_run(identifier)
        assert record is not None
        return record

    def get_trial_run(self, trial_run_id: str) -> Optional["TrialRunRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM trial_runs WHERE id = ?", (trial_run_id,)
            ).fetchone()
        return TrialRunRecord(**dict(row)) if row else None

    def list_trial_runs(self, trial_id: str) -> List["TrialRunRecord"]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trial_runs WHERE trial_id = ? ORDER BY ordinal", (trial_id,)
            ).fetchall()
        return [TrialRunRecord(**dict(row)) for row in rows]

    def update_trial_run(
        self,
        trial_run_id: str,
        *,
        status: str,
        work_item_id: Optional[str] = None,
    ) -> Optional["TrialRunRecord"]:
        now = datetime.now(timezone.utc).isoformat()
        assignments = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status, now]
        if work_item_id is not None:
            assignments.append("work_item_id = ?")
            params.append(work_item_id)
        params.append(trial_run_id)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE trial_runs SET {', '.join(assignments)} WHERE id = ?", params
            )
            self._conn.commit()
        return self.get_trial_run(trial_run_id) if cursor.rowcount else None

    def create_trial_segment(
        self,
        *,
        trial_run_id: str,
        ordinal: int,
        unit: str,
        start_value: int,
        end_value: int,
        work_item_id: Optional[str] = None,
        status: str = "queued",
        segment_id: Optional[str] = None,
    ) -> "TrialSegmentRecord":
        if int(end_value) <= int(start_value):
            raise ValueError("segment end_value must be greater than start_value")
        identifier = segment_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO trial_segments
                    (id, trial_run_id, ordinal, status, unit, start_value,
                     end_value, work_item_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    trial_run_id,
                    int(ordinal),
                    status,
                    unit,
                    int(start_value),
                    int(end_value),
                    work_item_id,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        record = self.get_trial_segment(identifier)
        assert record is not None
        return record

    def get_trial_segment(self, segment_id: str) -> Optional["TrialSegmentRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM trial_segments WHERE id = ?", (segment_id,)
            ).fetchone()
        return TrialSegmentRecord(**dict(row)) if row else None

    def list_trial_segments(self, trial_run_id: str) -> List["TrialSegmentRecord"]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM trial_segments WHERE trial_run_id = ? ORDER BY ordinal",
                (trial_run_id,),
            ).fetchall()
        return [TrialSegmentRecord(**dict(row)) for row in rows]

    def update_trial_segment(
        self,
        segment_id: str,
        *,
        status: Optional[str] = None,
        checkpoint_artifact_id: Optional[str] = None,
        decision: Optional[str] = None,
        decision_reason: Optional[str] = None,
    ) -> Optional["TrialSegmentRecord"]:
        allowed_decisions = {"continue", "pause", "stop", "prune", "complete"}
        if decision is not None and decision not in allowed_decisions:
            raise ValueError("segment decision must be continue, pause, stop, prune, or complete")
        now = datetime.now(timezone.utc).isoformat()
        assignments = ["updated_at = ?"]
        params: list[Any] = [now]
        for column, value in (
            ("status", status),
            ("checkpoint_artifact_id", checkpoint_artifact_id),
            ("decision", decision),
            ("decision_reason", decision_reason),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        if status == "running":
            assignments.append("started_at = COALESCE(started_at, ?)")
            params.append(now)
        if status in {
            "completed",
            "failed",
            "cancelled",
            "pruned",
            "stopped",
            "awaiting_review",
        }:
            assignments.append("completed_at = ?")
            params.append(now)
        params.append(segment_id)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE trial_segments SET {', '.join(assignments)} WHERE id = ?", params
            )
            self._conn.commit()
        return self.get_trial_segment(segment_id) if cursor.rowcount else None

    def create_model_artifact(
        self,
        *,
        artifact_hash: str,
        artifact_kind: str,
        run_id: str,
        model_id: str,
        backend: str,
        format: str,
        path: str,
        run_group_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        trial_segment_id: Optional[str] = None,
        parent_artifact_id: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        chat_template_hash: Optional[str] = None,
        size_bytes: int = 0,
        step: Optional[int] = None,
        cycle: Optional[int] = None,
        verification_status: str = "unverified",
        metadata: Optional[Mapping[str, Any]] = None,
        artifact_id: Optional[str] = None,
    ) -> "ModelArtifactRecord":
        if artifact_kind not in {"checkpoint", "final_model", "adapter"}:
            raise ValueError("artifact_kind must be checkpoint, final_model, or adapter")
        identifier = artifact_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO model_artifacts
                    (id, artifact_hash, artifact_kind, run_id, run_group_id,
                     trial_id, trial_segment_id, parent_artifact_id, model_id,
                     tokenizer_revision, chat_template_hash, backend, format,
                     path, size_bytes, step, cycle, verification_status,
                     metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    artifact_hash,
                    artifact_kind,
                    run_id,
                    run_group_id,
                    trial_id,
                    trial_segment_id,
                    parent_artifact_id,
                    model_id,
                    tokenizer_revision,
                    chat_template_hash,
                    backend,
                    format,
                    path,
                    max(0, int(size_bytes)),
                    step,
                    cycle,
                    verification_status,
                    json.dumps(dict(metadata or {}), sort_keys=True, default=str),
                    now,
                ),
            )
            self._conn.commit()
        record = self.get_model_artifact(identifier)
        assert record is not None
        return record

    def get_model_artifact(self, artifact_id: str) -> Optional["ModelArtifactRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM model_artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        return _row_to_model_artifact(row) if row else None

    def find_model_artifact(self, artifact_hash: str) -> Optional["ModelArtifactRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM model_artifacts WHERE artifact_hash = ?", (artifact_hash,)
            ).fetchone()
        return _row_to_model_artifact(row) if row else None

    def list_model_artifacts(
        self,
        *,
        run_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        artifact_kind: Optional[str] = None,
    ) -> List["ModelArtifactRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("run_id", run_id),
            ("run_group_id", run_group_id),
            ("artifact_kind", artifact_kind),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM model_artifacts {where} ORDER BY created_at, id", params
            ).fetchall()
        return [_row_to_model_artifact(row) for row in rows]

    # ----- Lab v3 evaluation exposure ledger ------------------------------

    def record_exposure(
        self,
        *,
        suite_revision_id: str,
        suite_item_id: str,
        exposure_type: str,
        dataset_version_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        run_id: Optional[str] = None,
        model_artifact_id: Optional[str] = None,
        inherited_from_id: Optional[str] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        exposure_id: Optional[str] = None,
    ) -> "ExposureLedgerRecord":
        if not suite_item_id or not exposure_type:
            raise ValueError("suite_item_id and exposure_type are required")
        if not any((dataset_version_id, run_group_id, run_id, model_artifact_id)):
            raise ValueError("an exposure target is required")
        identifier = exposure_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO exposure_ledger
                    (id, suite_revision_id, suite_item_id, exposure_type,
                     dataset_version_id, run_group_id, run_id, model_artifact_id,
                     inherited_from_id, provenance_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    suite_revision_id,
                    suite_item_id,
                    exposure_type,
                    dataset_version_id,
                    run_group_id,
                    run_id,
                    model_artifact_id,
                    inherited_from_id,
                    json.dumps(dict(provenance or {}), sort_keys=True, default=str),
                    now,
                ),
            )
            self._conn.commit()
        record = self.get_exposure(identifier)
        assert record is not None
        return record

    def get_exposure(self, exposure_id: str) -> Optional["ExposureLedgerRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM exposure_ledger WHERE id = ?", (exposure_id,)
            ).fetchone()
        return _row_to_exposure(row) if row else None

    def list_exposures(
        self,
        *,
        suite_revision_id: Optional[str] = None,
        dataset_version_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        run_id: Optional[str] = None,
        model_artifact_id: Optional[str] = None,
    ) -> List["ExposureLedgerRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("suite_revision_id", suite_revision_id),
            ("dataset_version_id", dataset_version_id),
            ("run_group_id", run_group_id),
            ("run_id", run_id),
            ("model_artifact_id", model_artifact_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM exposure_ledger {where} ORDER BY created_at, id", params
            ).fetchall()
        return [_row_to_exposure(row) for row in rows]

    def inherit_exposures(
        self,
        source: Sequence["ExposureLedgerRecord"],
        *,
        dataset_version_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        run_id: Optional[str] = None,
        model_artifact_id: Optional[str] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> List["ExposureLedgerRecord"]:
        return [
            self.record_exposure(
                suite_revision_id=value.suite_revision_id,
                suite_item_id=value.suite_item_id,
                exposure_type="inherited",
                dataset_version_id=dataset_version_id,
                run_group_id=run_group_id,
                run_id=run_id,
                model_artifact_id=model_artifact_id,
                inherited_from_id=value.id,
                provenance=provenance,
            )
            for value in source
        ]

    # ----- Lab v5 adaptive checkpoints and evidence ----------------------

    def create_checkpoint_policy(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        policy_id: Optional[str] = None,
    ) -> "CheckpointPolicyRecord":
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("checkpoint policy name is required")
        identifier = policy_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO checkpoint_policies
                        (id, name, description, latest_revision_id, archived,
                         created_at, updated_at)
                    VALUES (?, ?, ?, NULL, 0, ?, ?)
                    """,
                    (identifier, normalized_name, description, now, now),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"checkpoint policy {normalized_name!r} already exists") from exc
        record = self.get_checkpoint_policy(identifier)
        assert record is not None
        return record

    def get_checkpoint_policy(self, policy_id: str) -> Optional["CheckpointPolicyRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoint_policies WHERE id = ?", (policy_id,)
            ).fetchone()
        return _row_to_checkpoint_policy(row) if row else None

    def get_checkpoint_policy_by_name(self, name: str) -> Optional["CheckpointPolicyRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoint_policies WHERE name = ?", (str(name).strip(),)
            ).fetchone()
        return _row_to_checkpoint_policy(row) if row else None

    def list_checkpoint_policies(
        self, *, archived: Optional[bool] = False, limit: int = 100, offset: int = 0
    ) -> List["CheckpointPolicyRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        if archived is not None:
            clauses.append("archived = ?")
            params.append(1 if archived else 0)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend((max(1, int(limit)), max(0, int(offset))))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM checkpoint_policies {where} "
                "ORDER BY updated_at DESC, id LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [_row_to_checkpoint_policy(row) for row in rows]

    def archive_checkpoint_policy(
        self, policy_id: str, *, archived: bool = True
    ) -> Optional["CheckpointPolicyRecord"]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE checkpoint_policies SET archived = ?, updated_at = ? WHERE id = ?",
                (1 if archived else 0, now, policy_id),
            )
            self._conn.commit()
        return self.get_checkpoint_policy(policy_id) if cursor.rowcount else None

    def create_checkpoint_policy_revision(
        self,
        *,
        policy_id: str,
        definition: Mapping[str, Any],
        content_hash: str,
        development_suite_revision_id: str,
        primary_metric: str,
        direction: str,
        revision_id: Optional[str] = None,
    ) -> "CheckpointPolicyRevisionRecord":
        if direction not in {"maximize", "minimize"}:
            raise ValueError("checkpoint policy direction must be maximize or minimize")
        if not all(
            str(value).strip()
            for value in (policy_id, content_hash, development_suite_revision_id, primary_metric)
        ):
            raise ValueError("checkpoint policy revision fields cannot be empty")
        identifier = revision_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        encoded_definition = json.dumps(
            dict(definition), sort_keys=True, separators=(",", ":"), default=str
        )
        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM checkpoint_policy_revisions "
                "WHERE policy_id = ? AND content_hash = ?",
                (policy_id, content_hash),
            ).fetchone()
            if existing is not None:
                return _row_to_checkpoint_policy_revision(existing)
            policy = self._conn.execute(
                "SELECT id FROM checkpoint_policies WHERE id = ?", (policy_id,)
            ).fetchone()
            if policy is None:
                raise ValueError(f"unknown checkpoint policy: {policy_id}")
            row = self._conn.execute(
                "SELECT COALESCE(MAX(revision_number), 0) + 1 AS revision_number "
                "FROM checkpoint_policy_revisions WHERE policy_id = ?",
                (policy_id,),
            ).fetchone()
            revision_number = int(row["revision_number"])
            self._conn.execute(
                """
                INSERT INTO checkpoint_policy_revisions
                    (id, policy_id, revision_number, content_hash,
                     development_suite_revision_id, primary_metric, direction,
                     definition_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    policy_id,
                    revision_number,
                    content_hash,
                    development_suite_revision_id,
                    primary_metric,
                    direction,
                    encoded_definition,
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE checkpoint_policies SET latest_revision_id = ?, updated_at = ? "
                "WHERE id = ?",
                (identifier, now, policy_id),
            )
            self._conn.commit()
        record = self.get_checkpoint_policy_revision(identifier)
        assert record is not None
        return record

    def get_checkpoint_policy_revision(
        self, revision_id: str
    ) -> Optional["CheckpointPolicyRevisionRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoint_policy_revisions WHERE id = ?", (revision_id,)
            ).fetchone()
        return _row_to_checkpoint_policy_revision(row) if row else None

    def list_checkpoint_policy_revisions(
        self, policy_id: str
    ) -> List["CheckpointPolicyRevisionRecord"]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM checkpoint_policy_revisions WHERE policy_id = ? "
                "ORDER BY revision_number DESC",
                (policy_id,),
            ).fetchall()
        return [_row_to_checkpoint_policy_revision(row) for row in rows]

    def create_checkpoint_gate_decision(
        self,
        *,
        policy_revision_id: str,
        plan_hash: str,
        boundary_index: int,
        action: str,
        reasons: Sequence[str],
        evidence: Mapping[str, Any],
        idempotency_key: str,
        automatic: bool = False,
        run_group_id: Optional[str] = None,
        trial_run_id: Optional[str] = None,
        trial_segment_id: Optional[str] = None,
        checkpoint_occurrence_id: Optional[str] = None,
        override_of_id: Optional[str] = None,
        override_reason: Optional[str] = None,
        content_hash: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> "CheckpointGateDecisionRecord":
        if action not in {"continue", "pause", "stop"}:
            raise ValueError("checkpoint gate action must be continue, pause, or stop")
        normalized_reasons = [str(value).strip() for value in reasons if str(value).strip()]
        if not normalized_reasons:
            raise ValueError("checkpoint gate decision requires reasons")
        if int(boundary_index) < 0:
            raise ValueError("boundary_index cannot be negative")
        if not str(idempotency_key).strip():
            raise ValueError("idempotency_key is required")
        if override_of_id is not None and not str(override_reason or "").strip():
            raise ValueError("gate overrides require a reason")
        identity = content_hash or _content_hash(
            {
                "policy_revision_id": policy_revision_id,
                "plan_hash": plan_hash,
                "boundary_index": int(boundary_index),
                "action": action,
                "automatic": bool(automatic),
                "reasons": normalized_reasons,
                "evidence": dict(evidence),
                "override_of_id": override_of_id,
                "override_reason": override_reason,
            }
        )
        identifier = decision_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM checkpoint_gate_decisions WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            if existing is not None:
                return _row_to_checkpoint_gate_decision(existing)
            self._conn.execute(
                """
                INSERT INTO checkpoint_gate_decisions
                    (id, idempotency_key, policy_revision_id, plan_hash,
                     run_group_id, trial_run_id, trial_segment_id,
                     checkpoint_occurrence_id, boundary_index, action, automatic,
                     reasons_json, evidence_json, content_hash, override_of_id,
                     override_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    idempotency_key,
                    policy_revision_id,
                    plan_hash,
                    run_group_id,
                    trial_run_id,
                    trial_segment_id,
                    checkpoint_occurrence_id,
                    int(boundary_index),
                    action,
                    1 if automatic else 0,
                    json.dumps(normalized_reasons, sort_keys=True),
                    json.dumps(dict(evidence), sort_keys=True, default=str),
                    identity,
                    override_of_id,
                    override_reason,
                    now,
                ),
            )
            self._conn.commit()
        record = self.get_checkpoint_gate_decision(identifier)
        assert record is not None
        return record

    def get_checkpoint_gate_decision(
        self, decision_id: str
    ) -> Optional["CheckpointGateDecisionRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoint_gate_decisions WHERE id = ?", (decision_id,)
            ).fetchone()
        return _row_to_checkpoint_gate_decision(row) if row else None

    def list_checkpoint_gate_decisions(
        self,
        *,
        run_group_id: Optional[str] = None,
        trial_run_id: Optional[str] = None,
        trial_segment_id: Optional[str] = None,
        policy_revision_id: Optional[str] = None,
        limit: int = 500,
    ) -> List["CheckpointGateDecisionRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("run_group_id", run_group_id),
            ("trial_run_id", trial_run_id),
            ("trial_segment_id", trial_segment_id),
            ("policy_revision_id", policy_revision_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM checkpoint_gate_decisions {where} "
                "ORDER BY created_at, id LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_checkpoint_gate_decision(row) for row in rows]

    def create_cohort_analysis_snapshot(
        self,
        *,
        request: Mapping[str, Any],
        analysis: Mapping[str, Any],
        primary_metric: str,
        direction: str,
        content_hash: Optional[str] = None,
        baseline_subject_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        status: str = "completed",
        snapshot_id: Optional[str] = None,
    ) -> "CohortAnalysisSnapshotRecord":
        if direction not in {"maximize", "minimize"}:
            raise ValueError("cohort direction must be maximize or minimize")
        identity = content_hash or _content_hash(
            {"request": dict(request), "analysis": dict(analysis)}
        )
        identifier = snapshot_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM cohort_analysis_snapshots WHERE content_hash = ?",
                (identity,),
            ).fetchone()
            if existing is not None:
                return _row_to_cohort_analysis_snapshot(existing)
            self._conn.execute(
                """
                INSERT INTO cohort_analysis_snapshots
                    (id, content_hash, run_group_id, baseline_subject_id,
                     primary_metric, direction, status, request_json,
                     analysis_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    identity,
                    run_group_id,
                    baseline_subject_id,
                    primary_metric,
                    direction,
                    status,
                    json.dumps(dict(request), sort_keys=True, default=str),
                    json.dumps(dict(analysis), sort_keys=True, default=str),
                    now,
                ),
            )
            self._conn.commit()
        record = self.get_cohort_analysis_snapshot(identifier)
        assert record is not None
        return record

    def get_cohort_analysis_snapshot(
        self, snapshot_id: str
    ) -> Optional["CohortAnalysisSnapshotRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM cohort_analysis_snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
        return _row_to_cohort_analysis_snapshot(row) if row else None

    def list_cohort_analysis_snapshots(
        self, *, run_group_id: Optional[str] = None, limit: int = 100
    ) -> List["CohortAnalysisSnapshotRecord"]:
        where = "WHERE run_group_id = ?" if run_group_id is not None else ""
        params: list[Any] = [run_group_id] if run_group_id is not None else []
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM cohort_analysis_snapshots {where} "
                "ORDER BY created_at DESC, id LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_cohort_analysis_snapshot(row) for row in rows]

    def create_research_decision(
        self,
        *,
        analysis_snapshot_id: str,
        selected_subject: Mapping[str, Any],
        rejected_subjects: Sequence[Mapping[str, Any]] = (),
        exclusions: Sequence[Mapping[str, Any]] = (),
        rationale: str,
        fork_spec: Optional[Mapping[str, Any]] = None,
        override_reason: Optional[str] = None,
        content_hash: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> "ResearchDecisionRecord":
        normalized_rationale = str(rationale).strip()
        if not normalized_rationale:
            raise ValueError("research decision rationale is required")
        payload = {
            "analysis_snapshot_id": analysis_snapshot_id,
            "selected_subject": dict(selected_subject),
            "rejected_subjects": [dict(value) for value in rejected_subjects],
            "exclusions": [dict(value) for value in exclusions],
            "rationale": normalized_rationale,
            "override_reason": override_reason,
            "fork_spec": dict(fork_spec or {}),
        }
        identity = content_hash or _content_hash(payload)
        identifier = decision_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO research_decisions
                    (id, analysis_snapshot_id, selected_subject_json,
                     rejected_subjects_json, exclusions_json, rationale,
                     override_reason, fork_spec_json, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    analysis_snapshot_id,
                    json.dumps(payload["selected_subject"], sort_keys=True, default=str),
                    json.dumps(payload["rejected_subjects"], sort_keys=True, default=str),
                    json.dumps(payload["exclusions"], sort_keys=True, default=str),
                    normalized_rationale,
                    override_reason,
                    json.dumps(payload["fork_spec"], sort_keys=True, default=str),
                    identity,
                    now,
                ),
            )
            self._conn.commit()
        record = self.get_research_decision(identifier)
        assert record is not None
        return record

    def get_research_decision(self, decision_id: str) -> Optional["ResearchDecisionRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM research_decisions WHERE id = ?", (decision_id,)
            ).fetchone()
        return _row_to_research_decision(row) if row else None

    def list_research_decisions(
        self, *, analysis_snapshot_id: Optional[str] = None, limit: int = 100
    ) -> List["ResearchDecisionRecord"]:
        where = "WHERE analysis_snapshot_id = ?" if analysis_snapshot_id else ""
        params: list[Any] = [analysis_snapshot_id] if analysis_snapshot_id else []
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM research_decisions {where} " "ORDER BY created_at DESC, id LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_research_decision(row) for row in rows]

    def create_evidence_bundle(
        self,
        *,
        analysis_snapshot_id: str,
        content_hash: str,
        storage_path: str,
        research_decision_id: Optional[str] = None,
        request: Optional[Mapping[str, Any]] = None,
        manifest: Optional[Mapping[str, Any]] = None,
        work_item_id: Optional[str] = None,
        status: str = "queued",
        bundle_id: Optional[str] = None,
    ) -> "EvidenceBundleRecord":
        identifier = bundle_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if status == "completed":
                existing = self._conn.execute(
                    "SELECT * FROM evidence_bundles WHERE content_hash = ? "
                    "AND status = 'completed'",
                    (content_hash,),
                ).fetchone()
                if existing is not None:
                    return _row_to_evidence_bundle(existing)
            self._conn.execute(
                """
                INSERT INTO evidence_bundles
                    (id, analysis_snapshot_id, research_decision_id, status,
                     content_hash, storage_path, request_json, manifest_json,
                     work_item_id, error, created_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    identifier,
                    analysis_snapshot_id,
                    research_decision_id,
                    status,
                    content_hash,
                    str(storage_path),
                    json.dumps(dict(request or {}), sort_keys=True, default=str),
                    json.dumps(dict(manifest or {}), sort_keys=True, default=str),
                    work_item_id,
                    now,
                    now,
                    now if status == "completed" else None,
                ),
            )
            self._conn.commit()
        record = self.get_evidence_bundle(identifier)
        assert record is not None
        return record

    def get_evidence_bundle(self, bundle_id: str) -> Optional["EvidenceBundleRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM evidence_bundles WHERE id = ?", (bundle_id,)
            ).fetchone()
        return _row_to_evidence_bundle(row) if row else None

    def find_completed_evidence_bundle(self, content_hash: str) -> Optional["EvidenceBundleRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM evidence_bundles WHERE content_hash = ? "
                "AND status = 'completed' ORDER BY completed_at LIMIT 1",
                (content_hash,),
            ).fetchone()
        return _row_to_evidence_bundle(row) if row else None

    def list_evidence_bundles(
        self,
        *,
        status: Optional[str] = None,
        analysis_snapshot_id: Optional[str] = None,
        limit: int = 100,
    ) -> List["EvidenceBundleRecord"]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (("status", status), ("analysis_snapshot_id", analysis_snapshot_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM evidence_bundles {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_evidence_bundle(row) for row in rows]

    def update_evidence_bundle(
        self,
        bundle_id: str,
        *,
        status: Optional[str] = None,
        content_hash: Optional[str] = None,
        storage_path: Optional[str] = None,
        manifest: Optional[Mapping[str, Any]] = None,
        work_item_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional["EvidenceBundleRecord"]:
        current = self.get_evidence_bundle(bundle_id)
        if current is None:
            return None
        if current.completed_at is not None:
            identity_changes = {
                "content_hash": content_hash,
                "storage_path": storage_path,
                "work_item_id": work_item_id,
                "manifest": manifest,
            }
            changed_identity = any(
                value is not None
                and value != (current.manifest if key == "manifest" else getattr(current, key))
                for key, value in identity_changes.items()
            )
            if changed_identity:
                raise ValueError("completed evidence bundle identity is immutable")
            if current.status == "completed" and status == "corrupt":
                if not str(error or "").strip():
                    raise ValueError(
                        "marking a completed evidence bundle corrupt requires a reason"
                    )
            elif current.status == "completed" and (
                status in {None, "completed"} and error in {None, current.error}
            ):
                return current
            elif current.status == "corrupt" and status in {None, "corrupt"}:
                if status is None and error is None:
                    return current
                if error is not None and not str(error).strip():
                    raise ValueError("corrupt evidence bundle annotation cannot be empty")
            else:
                raise ValueError("completed evidence bundle identity is immutable")
        now = datetime.now(timezone.utc).isoformat()
        assignments = ["updated_at = ?"]
        params: list[Any] = [now]
        for column, value in (
            ("status", status),
            ("content_hash", content_hash),
            ("storage_path", storage_path),
            ("work_item_id", work_item_id),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                params.append(value)
        if manifest is not None:
            assignments.append("manifest_json = ?")
            params.append(json.dumps(dict(manifest), sort_keys=True, default=str))
        if error is not None or status in {"running", "completed"}:
            assignments.append("error = ?")
            params.append(error)
        if status == "completed":
            assignments.append("completed_at = ?")
            params.append(now)
        params.append(bundle_id)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE evidence_bundles SET {', '.join(assignments)} WHERE id = ?", params
            )
            self._conn.commit()
        return self.get_evidence_bundle(bundle_id) if cursor.rowcount else None

    def save_workspace_draft(
        self,
        *,
        draft_kind: str,
        content: Mapping[str, Any],
        owner_key: str = "local",
        name: str = "default",
        draft_id: Optional[str] = None,
        ttl_days: int = 30,
    ) -> "WorkspaceDraftRecord":
        normalized = {
            "draft_kind": str(draft_kind).strip(),
            "owner_key": str(owner_key).strip(),
            "name": str(name).strip(),
        }
        if not all(normalized.values()):
            raise ValueError("workspace draft kind, owner, and name are required")
        if int(ttl_days) <= 0:
            raise ValueError("workspace draft ttl_days must be positive")
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        expires = (now_dt + timedelta(days=int(ttl_days))).isoformat()
        encoded = json.dumps(dict(content), sort_keys=True, default=str)
        identity = _content_hash(dict(content))
        identifier = draft_id or uuid.uuid4().hex
        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM workspace_drafts WHERE owner_key = ? "
                "AND draft_kind = ? AND name = ?",
                (normalized["owner_key"], normalized["draft_kind"], normalized["name"]),
            ).fetchone()
            if existing is not None:
                identifier = str(existing["id"])
                self._conn.execute(
                    "UPDATE workspace_drafts SET content_json = ?, content_hash = ?, "
                    "expires_at = ?, updated_at = ? WHERE id = ?",
                    (encoded, identity, expires, now, identifier),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO workspace_drafts
                        (id, owner_key, draft_kind, name, content_json,
                         content_hash, expires_at, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        normalized["owner_key"],
                        normalized["draft_kind"],
                        normalized["name"],
                        encoded,
                        identity,
                        expires,
                        now,
                        now,
                    ),
                )
            self._conn.commit()
        record = self.get_workspace_draft(identifier)
        assert record is not None
        return record

    def get_workspace_draft(
        self, draft_id: str, *, include_expired: bool = False
    ) -> Optional["WorkspaceDraftRecord"]:
        expiry = "" if include_expired else " AND expires_at > ?"
        params: tuple[Any, ...] = (
            (draft_id,) if include_expired else (draft_id, datetime.now(timezone.utc).isoformat())
        )
        with self._lock:
            row = self._conn.execute(
                f"SELECT * FROM workspace_drafts WHERE id = ?{expiry}", params
            ).fetchone()
        return _row_to_workspace_draft(row) if row else None

    def get_workspace_draft_by_key(
        self,
        *,
        draft_kind: str,
        owner_key: str = "local",
        name: str = "default",
        include_expired: bool = False,
    ) -> Optional["WorkspaceDraftRecord"]:
        expiry = "" if include_expired else " AND expires_at > ?"
        params: tuple[Any, ...] = (
            (owner_key, draft_kind, name)
            if include_expired
            else (
                owner_key,
                draft_kind,
                name,
                datetime.now(timezone.utc).isoformat(),
            )
        )
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM workspace_drafts WHERE owner_key = ? "
                f"AND draft_kind = ? AND name = ?{expiry}",
                params,
            ).fetchone()
        return _row_to_workspace_draft(row) if row else None

    def list_workspace_drafts(
        self, *, owner_key: str = "local", include_expired: bool = False
    ) -> List["WorkspaceDraftRecord"]:
        clauses = ["owner_key = ?"]
        params: list[Any] = [owner_key]
        if not include_expired:
            clauses.append("expires_at > ?")
            params.append(datetime.now(timezone.utc).isoformat())
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM workspace_drafts WHERE "
                + " AND ".join(clauses)
                + " ORDER BY updated_at DESC, id",
                params,
            ).fetchall()
        return [_row_to_workspace_draft(row) for row in rows]

    def delete_workspace_draft(self, draft_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM workspace_drafts WHERE id = ?", (draft_id,))
            self._conn.commit()
        return cursor.rowcount == 1

    def purge_expired_workspace_drafts(self, *, now: Optional[str] = None) -> int:
        cutoff = now or datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM workspace_drafts WHERE expires_at <= ?", (cutoff,)
            )
            self._conn.commit()
        return int(cursor.rowcount)

    # ----- Lab v8 reward integrity -----------------------------------------

    def reward_integrity_store(self) -> Any:
        """Return the shared v8 persistence facade for this connection."""

        store = getattr(self, "_reward_integrity_store_instance", None)
        if store is None:
            from halo_forge.reward_integrity.store import RewardIntegrityStore

            store = RewardIntegrityStore(self)
            self._reward_integrity_store_instance = store
        return store

    def create_reward_system(self, **values: Any) -> Any:
        return self.reward_integrity_store().create_system(**values)

    def get_reward_system(self, system_id: str) -> Any:
        return self.reward_integrity_store().get_system(system_id)

    def list_reward_systems(self, **values: Any) -> Any:
        return self.reward_integrity_store().list_systems(**values)

    def create_reward_system_revision(self, system_id: str, **values: Any) -> Any:
        return self.reward_integrity_store().create_system_revision(system_id, **values)

    def get_reward_system_revision(self, revision_id: str) -> Any:
        return self.reward_integrity_store().get_system_revision(revision_id)

    def list_reward_system_revisions(self, **values: Any) -> Any:
        return self.reward_integrity_store().list_system_revisions(**values)

    def create_reward_audit_protocol(self, **values: Any) -> Any:
        return self.reward_integrity_store().create_protocol(**values)

    def create_reward_audit_protocol_revision(
        self, protocol_id: str, definition: Mapping[str, Any], **values: Any
    ) -> Any:
        return self.reward_integrity_store().create_protocol_revision(
            protocol_id, definition, **values
        )

    def get_reward_audit_protocol_revision(self, revision_id: str) -> Any:
        return self.reward_integrity_store().get_protocol_revision(revision_id)

    def create_reward_integrity_profile(self, **values: Any) -> Any:
        return self.reward_integrity_store().create_integrity_profile(**values)

    def create_reward_integrity_profile_revision(self, profile_id: str, **values: Any) -> Any:
        return self.reward_integrity_store().create_integrity_profile_revision(
            profile_id, **values
        )

    def get_reward_integrity_profile_revision(self, revision_id: str) -> Any:
        return self.reward_integrity_store().get_integrity_profile_revision(revision_id)

    def create_training_signal_shard(self, **values: Any) -> Any:
        return self.reward_integrity_store().create_signal_shard(**values)

    def get_training_signal_shard(self, shard_id: str) -> Any:
        return self.reward_integrity_store().get_signal_shard(shard_id)

    def list_training_signal_shards(self, **values: Any) -> Any:
        return self.reward_integrity_store().list_signal_shards(**values)

    def create_reward_integrity_audit(self, **values: Any) -> Any:
        return self.reward_integrity_store().create_audit(**values)

    def get_reward_integrity_audit(self, audit_id: str) -> Any:
        return self.reward_integrity_store().get_audit(audit_id)

    def list_reward_integrity_audits(self, **values: Any) -> Any:
        return self.reward_integrity_store().list_audits(**values)

    def update_reward_integrity_audit(self, audit_id: str, **values: Any) -> Any:
        return self.reward_integrity_store().update_audit(audit_id, **values)

    def append_reward_integrity_sample(self, value: Mapping[str, Any]) -> Any:
        return self.reward_integrity_store().add_sample(value)

    def append_reward_integrity_observation(self, value: Mapping[str, Any]) -> Any:
        return self.reward_integrity_store().add_observation(value)

    def append_reward_integrity_metrics(self, values: Sequence[Any]) -> Any:
        return self.reward_integrity_store().add_metrics(values)

    def append_reward_integrity_decision(self, **values: Any) -> Any:
        return self.reward_integrity_store().add_decision(**values)

    def bind_reward_integrity(self, **values: Any) -> Any:
        return self.reward_integrity_store().bind(**values)

    def create_direct_run_segment(
        self,
        *,
        run_id: str,
        ordinal: int,
        unit: str,
        start_value: int,
        end_value: int,
        work_item_id: Optional[str] = None,
        segment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if unit not in {"step", "cycle", "epoch", "final"}:
            raise ValueError("direct-run segment unit must be step, cycle, epoch, or final")
        identifier, now = segment_id or f"direct-segment-{uuid.uuid4().hex}", datetime.now(
            timezone.utc
        ).isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO direct_run_segments
                    (id,run_id,ordinal,unit,start_value,end_value,status,
                     work_item_id,created_at,updated_at)
                VALUES (?,?,?,?,?,?,'queued',?,?,?)
                """,
                (
                    identifier,
                    run_id,
                    int(ordinal),
                    unit,
                    int(start_value),
                    int(end_value),
                    work_item_id,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        result = self.get_direct_run_segment(identifier)
        assert result is not None
        return result

    def get_direct_run_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM direct_run_segments WHERE id=?", (segment_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_direct_run_segments(self, run_id: str) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM direct_run_segments WHERE run_id=? ORDER BY ordinal,id",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def update_direct_run_segment(self, segment_id: str, **changes: Any) -> Dict[str, Any]:
        allowed = {
            "status",
            "work_item_id",
            "checkpoint_occurrence_id",
            "decision",
            "decision_reason",
            "started_at",
            "completed_at",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unknown direct-run segment fields: {sorted(unknown)}")
        if not changes:
            result = self.get_direct_run_segment(segment_id)
            if result is None:
                raise KeyError(f"unknown direct-run segment: {segment_id}")
            return result
        values = dict(changes)
        values["updated_at"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE direct_run_segments SET "
                + ",".join(f"{key}=?" for key in values)
                + " WHERE id=?",
                [*values.values(), segment_id],
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise KeyError(f"unknown direct-run segment: {segment_id}")
            self._conn.commit()
        result = self.get_direct_run_segment(segment_id)
        assert result is not None
        return result

    # ----- model registry (Track F-J) ---------------------------------------

    def create_registry_entry(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        base_model: Optional[str] = None,
        run_ids: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> "RegistryEntry":
        """Insert a new registry entry. Name must be unique."""
        if not name or not name.strip():
            raise ValueError("registry entry name is required")
        now = datetime.now(timezone.utc).isoformat()
        run_ids_list = sorted(set(str(r) for r in (run_ids or []) if r))
        tags_list = sorted(set(str(t) for t in (tags or []) if t))
        with self._lock:
            try:
                cur = self._conn.execute(
                    """
                    INSERT INTO model_registry
                        (name, description, base_model, run_ids, tags, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        name.strip(),
                        description,
                        base_model,
                        json.dumps(run_ids_list),
                        json.dumps(tags_list),
                        now,
                        now,
                    ),
                )
                self._conn.commit()
                rowid = cur.lastrowid
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"registry entry with name {name!r} already exists") from exc
        return self.get_registry_entry(rowid)  # type: ignore[arg-type]

    def get_registry_entry(self, entry_id: int) -> Optional["RegistryEntry"]:
        cur = self._conn.execute("SELECT * FROM model_registry WHERE id = ?", (int(entry_id),))
        row = cur.fetchone()
        return _row_to_registry_entry(row) if row else None

    def get_registry_entry_by_name(self, name: str) -> Optional["RegistryEntry"]:
        cur = self._conn.execute("SELECT * FROM model_registry WHERE name = ?", (name,))
        row = cur.fetchone()
        return _row_to_registry_entry(row) if row else None

    def list_registry_entries(self) -> List["RegistryEntry"]:
        cur = self._conn.execute("SELECT * FROM model_registry ORDER BY updated_at DESC")
        return [_row_to_registry_entry(r) for r in cur.fetchall()]

    def update_registry_entry(
        self,
        entry_id: int,
        *,
        description: Optional[str] = None,
        base_model: Optional[str] = None,
        run_ids: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> Optional["RegistryEntry"]:
        """Patch a registry entry. Only non-None fields are updated."""
        existing = self.get_registry_entry(entry_id)
        if existing is None:
            return None

        new_desc = description if description is not None else existing.description
        new_base = base_model if base_model is not None else existing.base_model
        new_runs = (
            sorted(set(str(r) for r in run_ids if r)) if run_ids is not None else existing.run_ids
        )
        new_tags = sorted(set(str(t) for t in tags if t)) if tags is not None else existing.tags
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """
                UPDATE model_registry
                SET description = ?, base_model = ?, run_ids = ?, tags = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_desc,
                    new_base,
                    json.dumps(new_runs),
                    json.dumps(new_tags),
                    now,
                    int(entry_id),
                ),
            )
            self._conn.commit()
        return self.get_registry_entry(entry_id)

    # ----- run lineage (Track F-Q) -----------------------------------------

    def record_fork(
        self,
        *,
        child_run_id: str,
        parent_run_id: str,
        forked_at_cycle: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Record that ``child_run_id`` forked from ``parent_run_id``.

        Idempotent on the (child, parent) pair — re-recording the same
        edge updates the cycle / notes columns rather than failing.
        """
        if not child_run_id or not parent_run_id:
            raise ValueError("both child_run_id and parent_run_id are required")
        if child_run_id == parent_run_id:
            raise ValueError("a run cannot be its own parent")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO run_lineage (child_run_id, parent_run_id, forked_at_cycle, notes)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(child_run_id, parent_run_id) DO UPDATE SET
                    forked_at_cycle = excluded.forked_at_cycle,
                    notes = excluded.notes
                """,
                (str(child_run_id), str(parent_run_id), forked_at_cycle, notes),
            )
            self._conn.commit()

    def remove_fork(
        self,
        *,
        child_run_id: str,
        parent_run_id: str,
    ) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM run_lineage WHERE child_run_id = ? AND parent_run_id = ?",
                (str(child_run_id), str(parent_run_id)),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_parents(self, child_run_id: str) -> List[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT parent_run_id, forked_at_cycle, notes "
            "FROM run_lineage WHERE child_run_id = ?",
            (str(child_run_id),),
        )
        return [
            {
                "parent_run_id": row["parent_run_id"],
                "forked_at_cycle": row["forked_at_cycle"],
                "notes": row["notes"],
            }
            for row in cur.fetchall()
        ]

    def get_children(self, parent_run_id: str) -> List[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT child_run_id, forked_at_cycle, notes "
            "FROM run_lineage WHERE parent_run_id = ?",
            (str(parent_run_id),),
        )
        return [
            {
                "child_run_id": row["child_run_id"],
                "forked_at_cycle": row["forked_at_cycle"],
                "notes": row["notes"],
            }
            for row in cur.fetchall()
        ]

    def get_lineage(
        self,
        run_id: str,
        *,
        max_depth: int = 8,
    ) -> Dict[str, Any]:
        """Return ancestors (transitive parents) + descendants (transitive
        children) for ``run_id``.

        ``max_depth`` caps the BFS to avoid pathological / cyclic
        traversals; the lineage table doesn't enforce a DAG so a buggy
        client could in principle insert a cycle.
        """
        ancestors: List[Dict[str, Any]] = []
        descendants: List[Dict[str, Any]] = []
        seen_ancestors: set[str] = {run_id}
        seen_descendants: set[str] = {run_id}

        # Walk up.
        frontier = [(run_id, 0)]
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for entry in self.get_parents(current):
                pid = entry["parent_run_id"]
                if pid in seen_ancestors:
                    continue
                seen_ancestors.add(pid)
                ancestors.append({**entry, "child_run_id": current, "depth": depth + 1})
                frontier.append((pid, depth + 1))

        # Walk down.
        frontier = [(run_id, 0)]
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for entry in self.get_children(current):
                cid = entry["child_run_id"]
                if cid in seen_descendants:
                    continue
                seen_descendants.add(cid)
                descendants.append({**entry, "parent_run_id": current, "depth": depth + 1})
                frontier.append((cid, depth + 1))

        return {
            "run_id": run_id,
            "ancestors": ancestors,
            "descendants": descendants,
        }

    def delete_registry_entry(self, entry_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM model_registry WHERE id = ?", (int(entry_id),))
            self._conn.commit()
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


@dataclass
class RegistryEntry:
    """A named bundle of run_ids the user wants to compare or promote
    as a unit. The cohort eval dashboard reads these directly so a
    saved entry is one click away from a runs × tasks grid."""

    id: int
    name: str
    description: Optional[str]
    base_model: Optional[str]
    run_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row_to_registry_entry(row: sqlite3.Row) -> RegistryEntry:
    return RegistryEntry(
        id=int(row["id"]),
        name=str(row["name"]),
        description=row["description"],
        base_model=row["base_model"],
        run_ids=json.loads(row["run_ids"] or "[]"),
        tags=json.loads(row["tags"] or "[]"),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _load_json(value: Optional[str], default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except (TypeError, json.JSONDecodeError):
        return default


@dataclass
class DatasetRecord:
    id: str
    name: str
    description: Optional[str]
    modality: str
    canonical_schema: str
    latest_version_id: Optional[str]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSourceRecord:
    id: str
    dataset_id: str
    kind: str
    uri: str
    config: Optional[str]
    split: Optional[str]
    revision: Optional[str]
    fingerprint: str
    size_bytes: Optional[int]
    row_count: Optional[int]
    metadata_json: str
    refreshed_from_source_id: Optional[str]
    created_at: str

    @property
    def metadata(self) -> Dict[str, Any]:
        return _load_json(self.metadata_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("metadata_json", None)
        result["metadata"] = self.metadata
        return result


@dataclass
class DatasetVersionRecord:
    id: str
    dataset_id: str
    source_id: Optional[str]
    parent_version_id: Optional[str]
    status: str
    content_hash: Optional[str]
    recipe_hash: str
    recipe_json: str
    storage_path: str
    row_count: int
    size_bytes: int
    split_counts_json: str
    statistics_json: str
    provenance_json: str
    source_fingerprints_json: str
    assets_materialized: bool
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]
    parents: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def recipe(self) -> Dict[str, Any]:
        return _load_json(self.recipe_json, {})

    @property
    def split_counts(self) -> Dict[str, int]:
        return _load_json(self.split_counts_json, {})

    @property
    def statistics(self) -> Dict[str, Any]:
        return _load_json(self.statistics_json, {})

    @property
    def provenance(self) -> Dict[str, Any]:
        return _load_json(self.provenance_json, {})

    @property
    def source_fingerprints(self) -> Dict[str, str]:
        return _load_json(self.source_fingerprints_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in (
            "recipe_json",
            "split_counts_json",
            "statistics_json",
            "provenance_json",
            "source_fingerprints_json",
        ):
            result.pop(key, None)
        result.update(
            recipe=self.recipe,
            split_counts=self.split_counts,
            statistics=self.statistics,
            provenance=self.provenance,
            source_fingerprints=self.source_fingerprints,
            assets_materialized=bool(self.assets_materialized),
            parent_version_ids=[p["parent_version_id"] for p in self.parents],
        )
        return result


@dataclass
class DatasetJobRecord:
    id: str
    dataset_id: Optional[str]
    version_id: Optional[str]
    job_type: str
    status: str
    stage: str
    processed_records: int
    total_records: Optional[int]
    accepted_records: int
    rejected_records: int
    output_size_bytes: int
    logs_json: str
    request_json: str
    checkpoint_json: str
    error: Optional[str]
    cancel_requested: bool
    created_at: str
    updated_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    work_item_id: Optional[str] = None

    @property
    def logs(self) -> List[Any]:
        return _load_json(self.logs_json, [])

    @property
    def request(self) -> Dict[str, Any]:
        return _load_json(self.request_json, {})

    @property
    def checkpoint(self) -> Dict[str, Any]:
        return _load_json(self.checkpoint_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ("logs_json", "request_json", "checkpoint_json"):
            result.pop(key, None)
        result.update(
            logs=self.logs,
            request=self.request,
            checkpoint=self.checkpoint,
            cancel_requested=bool(self.cancel_requested),
        )
        return result


@dataclass
class DatasetImportRecord:
    id: str
    source_kind: str
    status: str
    display_name: Optional[str]
    source_uri: Optional[str]
    source_config: Optional[str]
    source_split: Optional[str]
    source_revision: Optional[str]
    resolved_revision: Optional[str]
    scenario_revision_id: Optional[str]
    fingerprint: Optional[str]
    expected_size_bytes: Optional[int]
    received_size_bytes: int
    file_count: int
    staging_path: Optional[str]
    managed_source_path: Optional[str]
    work_item_id: Optional[str]
    published_dataset_id: Optional[str]
    published_source_id: Optional[str]
    latest_inspection_id: Optional[str]
    metadata_json: str
    error: Optional[str]
    expires_at: Optional[str]
    created_at: str
    updated_at: str
    completed_at: Optional[str]

    @property
    def metadata(self) -> Dict[str, Any]:
        return _load_json(self.metadata_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("metadata_json", None)
        result["metadata"] = self.metadata
        return result


@dataclass
class DatasetImportFileRecord:
    id: str
    import_id: str
    relative_path: str
    status: str
    media_type: Optional[str]
    size_bytes: int
    received_bytes: int
    expected_sha256: Optional[str]
    content_sha256: Optional[str]
    staging_path: str
    metadata_json: str
    error: Optional[str]
    created_at: str
    updated_at: str
    completed_at: Optional[str]

    @property
    def metadata(self) -> Dict[str, Any]:
        return _load_json(self.metadata_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("metadata_json", None)
        result["metadata"] = self.metadata
        return result


@dataclass
class DatasetSourceInspectionRecord:
    id: str
    import_id: Optional[str]
    source_id: Optional[str]
    status: str
    source_fingerprint: str
    import_adapter_version: str
    scenario_registry_revision: str
    scenario_revision_id: Optional[str]
    sample_seed: int
    total_records: int
    valid_records: int
    invalid_records: int
    sample_count: int
    size_bytes: int
    fields_json: str
    candidates_json: str
    preview_json: str
    issues_json: str
    statistics_json: str
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]

    @property
    def fields(self) -> List[Dict[str, Any]]:
        return _load_json(self.fields_json, [])

    @property
    def candidates(self) -> List[Dict[str, Any]]:
        return _load_json(self.candidates_json, [])

    @property
    def preview(self) -> List[Dict[str, Any]]:
        return _load_json(self.preview_json, [])

    @property
    def issues(self) -> List[Dict[str, Any]]:
        return _load_json(self.issues_json, [])

    @property
    def statistics(self) -> Dict[str, Any]:
        return _load_json(self.statistics_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in (
            "fields_json",
            "candidates_json",
            "preview_json",
            "issues_json",
            "statistics_json",
        ):
            result.pop(key, None)
        result.update(
            fields=self.fields,
            candidates=self.candidates,
            preview=self.preview,
            issues=self.issues,
            statistics=self.statistics,
        )
        return result


@dataclass
class DocumentExtractionRecord:
    id: str
    import_id: Optional[str]
    source_id: Optional[str]
    status: str
    source_kind: str
    source_uri: str
    source_fingerprint: str
    extractor_version: str
    config_hash: str
    reuse_key: str
    config_json: str
    content_hash: Optional[str]
    bundle_path: Optional[str]
    manifest_hash: Optional[str]
    document_count: int
    item_count: int
    quarantined_count: int
    extracted_text_bytes: int
    statistics_json: str
    provenance_json: str
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    updated_at: str
    completed_at: Optional[str]

    @property
    def config(self) -> Dict[str, Any]:
        return _load_json(self.config_json, {})

    @property
    def statistics(self) -> Dict[str, Any]:
        return _load_json(self.statistics_json, {})

    @property
    def provenance(self) -> Dict[str, Any]:
        return _load_json(self.provenance_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ("config_json", "statistics_json", "provenance_json"):
            result.pop(key, None)
        result.update(
            config=self.config,
            statistics=self.statistics,
            provenance=self.provenance,
        )
        return result


@dataclass
class DocumentExtractionItemRecord:
    extraction_id: str
    ordinal: int
    document_id: str
    status: str
    source_uri: str
    relative_path: str
    source_kind: str
    media_type: str
    title: Optional[str]
    content_hash: Optional[str]
    text_char_count: int
    text_byte_count: int
    bundle_member: str
    bundle_ordinal: int
    locator_json: str
    provenance_json: str
    metadata_json: str
    error_code: Optional[str]
    error: Optional[str]

    @property
    def locator(self) -> Dict[str, Any]:
        return _load_json(self.locator_json, {})

    @property
    def provenance(self) -> Dict[str, Any]:
        return _load_json(self.provenance_json, {})

    @property
    def metadata(self) -> Dict[str, Any]:
        return _load_json(self.metadata_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ("locator_json", "provenance_json", "metadata_json"):
            result.pop(key, None)
        result.update(
            id=self.document_id,
            locator=self.locator,
            provenance=self.provenance,
            metadata=self.metadata,
        )
        return result


@dataclass
class RunDatasetRecord:
    run_id: str
    dataset_version_id: str
    split: str
    attached_at: str
    role: str = "train"
    training_artifact_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingArtifactBindingRecord:
    role: str
    dataset_version_id: str
    split: str
    row_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingArtifactRecord:
    id: str
    artifact_hash: str
    adapter_id: str
    adapter_version: str
    trainer_mode: str
    model_id: Optional[str]
    tokenizer_revision: Optional[str]
    chat_template_hash: Optional[str]
    manifest_path: str
    metadata_json: str
    created_at: str
    bindings: List[TrainingArtifactBindingRecord] = field(default_factory=list)

    @property
    def metadata(self) -> Dict[str, Any]:
        return _load_json(self.metadata_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("metadata_json", None)
        result["metadata"] = self.metadata
        result["bindings"] = [binding.to_dict() for binding in self.bindings]
        return result


@dataclass
class BenchmarkSuiteRecord:
    id: str
    name: str
    description: Optional[str]
    latest_revision_id: Optional[str]
    purpose: str
    archived: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["archived"] = bool(self.archived)
        return result


@dataclass
class BenchmarkSuiteRevisionRecord:
    id: str
    suite_id: str
    revision_number: int
    content_hash: str
    items_json: str
    generation_settings_json: str
    evaluator_versions_json: str
    primary_metric: str
    direction: str
    created_at: str

    @property
    def items(self) -> List[Dict[str, Any]]:
        return _load_json(self.items_json, [])

    @property
    def generation_settings(self) -> Dict[str, Any]:
        return _load_json(self.generation_settings_json, {})

    @property
    def evaluator_versions(self) -> Dict[str, Any]:
        return _load_json(self.evaluator_versions_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ("items_json", "generation_settings_json", "evaluator_versions_json"):
            result.pop(key, None)
        result.update(
            items=self.items,
            generation_settings=self.generation_settings,
            evaluator_versions=self.evaluator_versions,
        )
        return result


@dataclass
class EvaluationRecord:
    id: str
    suite_revision_id: str
    adapter_id: str
    adapter_version: str
    subject_type: str
    subject_ref: str
    subject_hash: str
    status: str
    stage: str
    processed_samples: int
    total_samples: Optional[int]
    request_json: str
    result_json: str
    logs_json: str
    artifact_path: Optional[str]
    reuse_key: str
    retry_count: int
    error: Optional[str]
    cancel_requested: bool
    created_at: str
    updated_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    work_item_id: Optional[str] = None

    @property
    def request(self) -> Dict[str, Any]:
        return _load_json(self.request_json, {})

    @property
    def result(self) -> Dict[str, Any]:
        return _load_json(self.result_json, {})

    @property
    def logs(self) -> List[Any]:
        return _load_json(self.logs_json, [])

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ("request_json", "result_json", "logs_json"):
            result.pop(key, None)
        result.update(
            request=self.request,
            result=self.result,
            logs=self.logs,
            cancel_requested=bool(self.cancel_requested),
        )
        return result


@dataclass
class EvaluationMetricRecord:
    evaluation_id: str
    name: str
    value: float
    direction: str
    suite_item_id: str
    metadata_json: str

    @property
    def metadata(self) -> Dict[str, Any]:
        return _load_json(self.metadata_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("metadata_json", None)
        result["metadata"] = self.metadata
        return result


@dataclass
class EvaluationSampleRecord:
    evaluation_id: str
    ordinal: int
    suite_item_id: str
    record_id: Optional[str]
    input_json: Optional[str]
    expected_json: Optional[str]
    output_json: Optional[str]
    score: Optional[float]
    passed: Optional[bool]
    latency_ms: Optional[float]
    error: Optional[str]
    verifier_trace_json: Optional[str]
    generation_seed: Optional[int]
    evidence_kind: str
    valid: bool
    mineable: bool
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    finish_reason: Optional[str]
    template_hash: Optional[str]
    runtime_versions_json: str
    score_direction: Optional[str]
    score_threshold: Optional[float]
    coverage: Optional[float]
    metadata_json: str

    @property
    def input(self) -> Any:
        return _load_json(self.input_json, None)

    @property
    def expected(self) -> Any:
        return _load_json(self.expected_json, None)

    @property
    def output(self) -> Any:
        return _load_json(self.output_json, None)

    @property
    def verifier_trace(self) -> Any:
        return _load_json(self.verifier_trace_json, None)

    @property
    def metadata(self) -> Dict[str, Any]:
        return _load_json(self.metadata_json, {})

    @property
    def runtime_versions(self) -> Dict[str, Any]:
        return _load_json(self.runtime_versions_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in (
            "input_json",
            "expected_json",
            "output_json",
            "verifier_trace_json",
            "runtime_versions_json",
            "metadata_json",
        ):
            result.pop(key, None)
        result.update(
            input=self.input,
            expected=self.expected,
            output=self.output,
            verifier_trace=self.verifier_trace,
            runtime_versions=self.runtime_versions,
            metadata=self.metadata,
            passed=self.passed if self.passed is None else bool(self.passed),
            valid=bool(self.valid),
            mineable=bool(self.mineable),
        )
        return result


@dataclass
class WorkItemRecord:
    id: str
    kind: str
    status: str
    stage: str
    resource_class: str
    priority: int
    launch_spec_json: str
    result_json: str
    progress_json: str
    resource_requirements_json: str
    domain_kind: Optional[str]
    domain_id: Optional[str]
    run_group_id: Optional[str]
    canonical_run_id: Optional[str]
    log_path: Optional[str]
    worker_id: Optional[str]
    worker_pid: Optional[int]
    worker_pid_started_at: Optional[float]
    claim_token: Optional[str]
    heartbeat_at: Optional[str]
    retry_count: int
    max_retries: int
    cancel_requested: bool
    not_before: Optional[str]
    error: Optional[str]
    created_at: str
    updated_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

    @property
    def launch_spec(self) -> Dict[str, Any]:
        return _load_json(self.launch_spec_json, {})

    @property
    def result(self) -> Dict[str, Any]:
        return _load_json(self.result_json, {})

    @property
    def progress(self) -> Dict[str, Any]:
        return _load_json(self.progress_json, {})

    @property
    def resource_requirements(self) -> Dict[str, Any]:
        return _load_json(self.resource_requirements_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in (
            "launch_spec_json",
            "result_json",
            "progress_json",
            "resource_requirements_json",
        ):
            result.pop(key, None)
        result.update(
            launch_spec=self.launch_spec,
            result=self.result,
            progress=self.progress,
            resource_requirements=self.resource_requirements,
            cancel_requested=bool(self.cancel_requested),
        )
        return result


@dataclass
class WorkItemDependencyRecord:
    work_item_id: str
    depends_on_work_item_id: str
    created_at: str
    dependency_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceLeaseRecord:
    resource_key: str
    holder_type: str
    holder_id: str
    work_item_id: Optional[str]
    lease_token: str
    retained: bool
    acquired_at: str
    heartbeat_at: str
    expires_at: Optional[str]
    holder_pid: Optional[int]
    holder_pid_started_at: Optional[float]
    metadata_json: str

    @property
    def metadata(self) -> Dict[str, Any]:
        return _load_json(self.metadata_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("metadata_json", None)
        result.update(metadata=self.metadata, retained=bool(self.retained))
        return result


@dataclass
class RunGroupRecord:
    id: str
    name: str
    kind: str
    status: str
    trainer_mode: str
    resolved_launch_config_json: str
    dataset_bindings_json: str
    base_subject_json: str
    development_suite_revision_id: Optional[str]
    holdout_suite_revision_id: Optional[str]
    search_space_json: str
    seeds_json: str
    budgets_json: str
    sampler_state_json: str
    pruning_policy_json: str
    checkpoint_policy_revision_id: Optional[str]
    resolved_checkpoint_plan_json: str
    parent_group_id: Optional[str]
    created_at: str
    updated_at: str

    @property
    def resolved_launch_config(self) -> Dict[str, Any]:
        return _load_json(self.resolved_launch_config_json, {})

    @property
    def dataset_bindings(self) -> List[Dict[str, Any]]:
        return _load_json(self.dataset_bindings_json, [])

    @property
    def base_subject(self) -> Dict[str, Any]:
        return _load_json(self.base_subject_json, {})

    @property
    def search_space(self) -> Dict[str, Any]:
        return _load_json(self.search_space_json, {})

    @property
    def seeds(self) -> List[int]:
        return _load_json(self.seeds_json, [])

    @property
    def budgets(self) -> Dict[str, Any]:
        return _load_json(self.budgets_json, {})

    @property
    def sampler_state(self) -> Dict[str, Any]:
        return _load_json(self.sampler_state_json, {})

    @property
    def pruning_policy(self) -> Dict[str, Any]:
        return _load_json(self.pruning_policy_json, {})

    @property
    def resolved_checkpoint_plan(self) -> Dict[str, Any]:
        return _load_json(self.resolved_checkpoint_plan_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        fields = {
            "resolved_launch_config_json": ("resolved_launch_config", self.resolved_launch_config),
            "dataset_bindings_json": ("dataset_bindings", self.dataset_bindings),
            "base_subject_json": ("base_subject", self.base_subject),
            "search_space_json": ("search_space", self.search_space),
            "seeds_json": ("seeds", self.seeds),
            "budgets_json": ("budgets", self.budgets),
            "sampler_state_json": ("sampler_state", self.sampler_state),
            "pruning_policy_json": ("pruning_policy", self.pruning_policy),
            "resolved_checkpoint_plan_json": (
                "resolved_checkpoint_plan",
                self.resolved_checkpoint_plan,
            ),
        }
        for raw, (name, value) in fields.items():
            result.pop(raw, None)
            result[name] = value
        return result


@dataclass
class RunGroupTrialRecord:
    id: str
    run_group_id: str
    ordinal: int
    config_hash: str
    sampled_config_json: str
    status: str
    objective_metric: Optional[str]
    objective_direction: Optional[str]
    objective_value: Optional[float]
    seed_coverage: int
    required_seed_count: int
    created_at: str
    updated_at: str

    @property
    def sampled_config(self) -> Dict[str, Any]:
        return _load_json(self.sampled_config_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("sampled_config_json", None)
        result["sampled_config"] = self.sampled_config
        return result


@dataclass
class TrialRunRecord:
    id: str
    trial_id: str
    run_id: str
    ordinal: int
    seed: int
    status: str
    work_item_id: Optional[str]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrialSegmentRecord:
    id: str
    trial_run_id: str
    ordinal: int
    status: str
    unit: str
    start_value: int
    end_value: int
    work_item_id: Optional[str]
    checkpoint_artifact_id: Optional[str]
    decision: Optional[str]
    decision_reason: Optional[str]
    created_at: str
    updated_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelArtifactRecord:
    id: str
    artifact_hash: str
    artifact_kind: str
    run_id: str
    run_group_id: Optional[str]
    trial_id: Optional[str]
    trial_segment_id: Optional[str]
    parent_artifact_id: Optional[str]
    model_id: str
    tokenizer_revision: Optional[str]
    chat_template_hash: Optional[str]
    backend: str
    format: str
    path: str
    size_bytes: int
    step: Optional[int]
    cycle: Optional[int]
    verification_status: str
    metadata_json: str
    created_at: str

    @property
    def metadata(self) -> Dict[str, Any]:
        return _load_json(self.metadata_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("metadata_json", None)
        result["metadata"] = self.metadata
        return result


@dataclass
class ExposureLedgerRecord:
    id: str
    suite_revision_id: str
    suite_item_id: str
    exposure_type: str
    dataset_version_id: Optional[str]
    run_group_id: Optional[str]
    run_id: Optional[str]
    model_artifact_id: Optional[str]
    inherited_from_id: Optional[str]
    provenance_json: str
    created_at: str

    @property
    def provenance(self) -> Dict[str, Any]:
        return _load_json(self.provenance_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("provenance_json", None)
        result["provenance"] = self.provenance
        return result


@dataclass
class CheckpointPolicyRecord:
    id: str
    name: str
    description: Optional[str]
    latest_revision_id: Optional[str]
    archived: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["archived"] = bool(self.archived)
        return result


@dataclass
class CheckpointPolicyRevisionRecord:
    id: str
    policy_id: str
    revision_number: int
    content_hash: str
    development_suite_revision_id: str
    primary_metric: str
    direction: str
    definition_json: str
    created_at: str

    @property
    def definition(self) -> Dict[str, Any]:
        return _load_json(self.definition_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("definition_json", None)
        result["definition"] = self.definition
        return result


@dataclass
class CheckpointGateDecisionRecord:
    id: str
    idempotency_key: str
    policy_revision_id: str
    plan_hash: str
    run_group_id: Optional[str]
    trial_run_id: Optional[str]
    trial_segment_id: Optional[str]
    checkpoint_occurrence_id: Optional[str]
    boundary_index: int
    action: str
    automatic: bool
    reasons_json: str
    evidence_json: str
    content_hash: str
    override_of_id: Optional[str]
    override_reason: Optional[str]
    created_at: str

    @property
    def reasons(self) -> List[str]:
        return _load_json(self.reasons_json, [])

    @property
    def evidence(self) -> Dict[str, Any]:
        return _load_json(self.evidence_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("reasons_json", None)
        result.pop("evidence_json", None)
        result.update(
            reasons=self.reasons,
            evidence=self.evidence,
            automatic=bool(self.automatic),
        )
        return result


@dataclass
class CohortAnalysisSnapshotRecord:
    id: str
    content_hash: str
    run_group_id: Optional[str]
    baseline_subject_id: Optional[str]
    primary_metric: str
    direction: str
    status: str
    request_json: str
    analysis_json: str
    created_at: str

    @property
    def request(self) -> Dict[str, Any]:
        return _load_json(self.request_json, {})

    @property
    def analysis(self) -> Dict[str, Any]:
        return _load_json(self.analysis_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("request_json", None)
        result.pop("analysis_json", None)
        result.update(request=self.request, analysis=self.analysis)
        return result


@dataclass
class ResearchDecisionRecord:
    id: str
    analysis_snapshot_id: str
    selected_subject_json: str
    rejected_subjects_json: str
    exclusions_json: str
    rationale: str
    override_reason: Optional[str]
    fork_spec_json: str
    content_hash: str
    created_at: str

    @property
    def selected_subject(self) -> Dict[str, Any]:
        return _load_json(self.selected_subject_json, {})

    @property
    def rejected_subjects(self) -> List[Dict[str, Any]]:
        return _load_json(self.rejected_subjects_json, [])

    @property
    def exclusions(self) -> List[Dict[str, Any]]:
        return _load_json(self.exclusions_json, [])

    @property
    def fork_spec(self) -> Dict[str, Any]:
        return _load_json(self.fork_spec_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in (
            "selected_subject_json",
            "rejected_subjects_json",
            "exclusions_json",
            "fork_spec_json",
        ):
            result.pop(key, None)
        result.update(
            selected_subject=self.selected_subject,
            rejected_subjects=self.rejected_subjects,
            exclusions=self.exclusions,
            fork_spec=self.fork_spec,
        )
        return result


@dataclass
class EvidenceBundleRecord:
    id: str
    analysis_snapshot_id: str
    research_decision_id: Optional[str]
    status: str
    content_hash: str
    storage_path: str
    request_json: str
    manifest_json: str
    work_item_id: Optional[str]
    error: Optional[str]
    created_at: str
    updated_at: str
    completed_at: Optional[str]

    @property
    def request(self) -> Dict[str, Any]:
        return _load_json(self.request_json, {})

    @property
    def manifest(self) -> Dict[str, Any]:
        return _load_json(self.manifest_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("request_json", None)
        result.pop("manifest_json", None)
        result.update(request=self.request, manifest=self.manifest)
        return result


@dataclass
class WorkspaceDraftRecord:
    id: str
    owner_key: str
    draft_kind: str
    name: str
    content_json: str
    content_hash: str
    expires_at: str
    created_at: str
    updated_at: str

    @property
    def content(self) -> Dict[str, Any]:
        return _load_json(self.content_json, {})

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("content_json", None)
        result["content"] = self.content
        return result


def _row_to_benchmark_suite(row: sqlite3.Row) -> BenchmarkSuiteRecord:
    values = dict(row)
    values["purpose"] = values.pop("purpose_v4", None) or values["purpose"]
    values["archived"] = bool(values["archived"])
    return BenchmarkSuiteRecord(**values)


def _row_to_benchmark_revision(row: sqlite3.Row) -> BenchmarkSuiteRevisionRecord:
    return BenchmarkSuiteRevisionRecord(**dict(row))


def _row_to_evaluation(row: sqlite3.Row) -> EvaluationRecord:
    values = dict(row)
    values["cancel_requested"] = bool(values["cancel_requested"])
    return EvaluationRecord(**values)


def _row_to_evaluation_metric(row: sqlite3.Row) -> EvaluationMetricRecord:
    return EvaluationMetricRecord(**dict(row))


def _row_to_evaluation_sample(row: sqlite3.Row) -> EvaluationSampleRecord:
    values = dict(row)
    if values["passed"] is not None:
        values["passed"] = bool(values["passed"])
    values["valid"] = bool(values["valid"])
    values["mineable"] = bool(values["mineable"])
    return EvaluationSampleRecord(**values)


def _row_to_work_item(row: sqlite3.Row) -> WorkItemRecord:
    values = dict(row)
    values["cancel_requested"] = bool(values["cancel_requested"])
    return WorkItemRecord(**values)


def _row_to_resource_lease(row: sqlite3.Row) -> ResourceLeaseRecord:
    values = dict(row)
    values["retained"] = bool(values["retained"])
    return ResourceLeaseRecord(**values)


def _row_to_run_group(row: sqlite3.Row) -> RunGroupRecord:
    return RunGroupRecord(**dict(row))


def _row_to_run_group_trial(row: sqlite3.Row) -> RunGroupTrialRecord:
    return RunGroupTrialRecord(**dict(row))


def _row_to_model_artifact(row: sqlite3.Row) -> ModelArtifactRecord:
    return ModelArtifactRecord(**dict(row))


def _row_to_exposure(row: sqlite3.Row) -> ExposureLedgerRecord:
    return ExposureLedgerRecord(**dict(row))


def _row_to_checkpoint_policy(row: sqlite3.Row) -> CheckpointPolicyRecord:
    values = dict(row)
    values["archived"] = bool(values["archived"])
    return CheckpointPolicyRecord(**values)


def _row_to_checkpoint_policy_revision(row: sqlite3.Row) -> CheckpointPolicyRevisionRecord:
    return CheckpointPolicyRevisionRecord(**dict(row))


def _row_to_checkpoint_gate_decision(row: sqlite3.Row) -> CheckpointGateDecisionRecord:
    values = dict(row)
    values["automatic"] = bool(values["automatic"])
    return CheckpointGateDecisionRecord(**values)


def _row_to_cohort_analysis_snapshot(row: sqlite3.Row) -> CohortAnalysisSnapshotRecord:
    return CohortAnalysisSnapshotRecord(**dict(row))


def _row_to_research_decision(row: sqlite3.Row) -> ResearchDecisionRecord:
    return ResearchDecisionRecord(**dict(row))


def _row_to_evidence_bundle(row: sqlite3.Row) -> EvidenceBundleRecord:
    return EvidenceBundleRecord(**dict(row))


def _row_to_workspace_draft(row: sqlite3.Row) -> WorkspaceDraftRecord:
    return WorkspaceDraftRecord(**dict(row))


def _row_to_dataset(row: sqlite3.Row) -> DatasetRecord:
    return DatasetRecord(**dict(row))


def _row_to_dataset_source(row: sqlite3.Row) -> DatasetSourceRecord:
    return DatasetSourceRecord(**dict(row))


def _row_to_dataset_version(row: sqlite3.Row) -> DatasetVersionRecord:
    values = dict(row)
    values["assets_materialized"] = bool(values["assets_materialized"])
    return DatasetVersionRecord(**values)


def _row_to_dataset_job(row: sqlite3.Row) -> DatasetJobRecord:
    values = dict(row)
    values["cancel_requested"] = bool(values["cancel_requested"])
    return DatasetJobRecord(**values)


def _row_to_dataset_import(row: sqlite3.Row) -> DatasetImportRecord:
    return DatasetImportRecord(**dict(row))


def _row_to_dataset_import_file(row: sqlite3.Row) -> DatasetImportFileRecord:
    return DatasetImportFileRecord(**dict(row))


def _row_to_dataset_source_inspection(row: sqlite3.Row) -> DatasetSourceInspectionRecord:
    return DatasetSourceInspectionRecord(**dict(row))


def _row_to_document_extraction(row: sqlite3.Row) -> DocumentExtractionRecord:
    return DocumentExtractionRecord(**dict(row))


def _row_to_document_extraction_item(
    row: sqlite3.Row,
) -> DocumentExtractionItemRecord:
    return DocumentExtractionItemRecord(**dict(row))


__all__ = [
    "BenchmarkSuiteRecord",
    "BenchmarkSuiteRevisionRecord",
    "CheckpointGateDecisionRecord",
    "CheckpointPolicyRecord",
    "CheckpointPolicyRevisionRecord",
    "CohortAnalysisSnapshotRecord",
    "DatasetJobRecord",
    "DatasetImportRecord",
    "DatasetImportFileRecord",
    "DatasetRecord",
    "DatasetSourceInspectionRecord",
    "DatasetSourceRecord",
    "DatasetVersionRecord",
    "DocumentExtractionItemRecord",
    "DocumentExtractionRecord",
    "EvaluationMetricRecord",
    "EvaluationRecord",
    "EvaluationSampleRecord",
    "EvidenceBundleRecord",
    "ExposureLedgerRecord",
    "ModelArtifactRecord",
    "RegistryEntry",
    "ResearchDecisionRecord",
    "ResourceLeaseRecord",
    "RunDatasetRecord",
    "RunDatabase",
    "RunFilter",
    "RunGroupRecord",
    "RunGroupTrialRecord",
    "RunRecord",
    "TrialRunRecord",
    "TrialSegmentRecord",
    "TrainingArtifactBindingRecord",
    "TrainingArtifactRecord",
    "WorkItemDependencyRecord",
    "WorkItemRecord",
    "WorkspaceDraftRecord",
    "get_database",
]
