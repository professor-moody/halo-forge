"""SQLite schema for the run database (Track F-G commit 1)."""

from __future__ import annotations


SCHEMA_VERSION = 1


# A single ``runs`` table with all the headline fields the UI filters /
# sorts on; the rest of the training_summary lives behind ``raw_json``.
#
# Indexes are added on the columns we actually filter / sort on:
# modality, status, timestamp, model_name. They cost ~10% on inserts
# and pay back orders of magnitude on the list view at any nontrivial
# row count.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    -- Filesystem id used by the existing ResultsService. Kept so we
    -- can round-trip through that surface during the migration.
    fs_id TEXT,
    modality TEXT NOT NULL,
    model_name TEXT NOT NULL,
    base_model_name TEXT,
    active_model_name TEXT,
    status TEXT NOT NULL,
    -- ISO 8601 timestamp; sortable as text since Python's datetime.isoformat
    -- emits zero-padded fields. Indexed for the "newest first" default sort.
    timestamp TEXT,
    output_dir TEXT NOT NULL,
    -- Headline metrics
    cycles_executed INTEGER DEFAULT 0,
    total_train_steps INTEGER DEFAULT 0,
    final_train_loss REAL,
    weights_updated INTEGER DEFAULT 0,
    final_update_reason TEXT,
    failure_reason TEXT,
    effectiveness_verdict TEXT,
    quality_status TEXT,
    keep_rate REAL,
    dominant_rejection_reason TEXT,
    final_model_path TEXT,
    seed INTEGER,
    -- Free-form JSON for everything else the UI might want to read on
    -- the detail page (yield_diagnostics, recovery, cycle_losses, etc).
    raw_json TEXT,
    -- Sync provenance — the mtime of the source training_summary.json
    -- when this row was last upserted. Lets sync skip unchanged rows.
    source_mtime REAL,
    indexed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_modality ON runs (modality);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs (timestamp);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs (model_name);

-- Track F-Q (run forking) reservation. Empty in commit 1.
-- Each row is "child run forked from parent at cycle N".
CREATE TABLE IF NOT EXISTS run_lineage (
    child_run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    parent_run_id TEXT NOT NULL,
    forked_at_cycle INTEGER,
    notes TEXT,
    PRIMARY KEY (child_run_id, parent_run_id)
);
"""


def initial_meta_rows() -> list[tuple[str, str]]:
    """Rows the schema needs to be self-describing on disk."""
    return [("schema_version", str(SCHEMA_VERSION))]
