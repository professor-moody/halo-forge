"""Run database (Track F-G).

A SQLite-backed index over training runs so the public API can offer
search / filter / sort / pagination without doing an O(n) filesystem
walk per request.

The filesystem is still the source of truth — every run writes a
``training_summary.json`` next to its checkpoints, and the DB is
re-derivable from those summaries at any time. The DB is an *index*,
not durable state. Losing the DB just means re-running ``sync_from_filesystem``.

Phase plan:
  - **Commit 1 (this commit):** schema, ``RunDatabase`` class, sync
    walker, tests. No public-API or frontend changes.
  - Commit 2: ``/runs`` endpoint queries the DB with filters.
  - Commit 3: frontend filter chips on the runs list.

Design choices:
  - SQLite, not DuckDB: simpler runtime, every Python install ships it,
    and the row counts (low thousands at scale) don't justify DuckDB's
    columnar story.
  - One row per run keyed by ``run_id``. Cycle metrics + raw_data live
    behind a JSON column rather than child tables: querying inside
    them is rare enough that the join cost outweighs the schema cost.
  - WAL journaling so live writes don't block reads.
"""

from halo_forge.run_db.db import (
    RegistryEntry,
    RunDatabase,
    RunFilter,
    RunRecord,
    get_database,
)
from halo_forge.run_db.sync import sync_from_filesystem

__all__ = [
    "RegistryEntry",
    "RunDatabase",
    "RunFilter",
    "RunRecord",
    "get_database",
    "sync_from_filesystem",
]
