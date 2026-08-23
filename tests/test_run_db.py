"""Run database tests (Track F-G commit 1).

Exercises the SQLite schema, RunDatabase CRUD/query API, and the
filesystem sync. All tests use ``RunDatabase(":memory:")`` so they
leave nothing on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_record(**overrides):
    from halo_forge.run_db import RunRecord

    base = dict(
        run_id="run_alpha",
        modality="sft",
        model_name="Qwen/Qwen2.5-3B",
        status="completed",
        timestamp="2026-05-01T12:00:00+00:00",
        output_dir="/tmp/run_alpha",
        cycles_executed=3,
        final_train_loss=0.42,
        weights_updated=True,
        raw_json='{"foo": "bar"}',
    )
    base.update(overrides)
    return RunRecord(**base)


def _attach_evaluation(db, run_id: str, *, subject_type: str = "run", status: str = "completed"):
    suite = db.create_benchmark_suite(name=f"suite-{run_id}-{subject_type}-{status}")
    revision = db.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash=f"hash-{run_id}-{subject_type}-{status}",
        items=[{"id": "one"}],
        primary_metric="score",
        direction="maximize",
    )
    evaluation = db.create_evaluation(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        adapter_version="1",
        subject_type=subject_type,
        subject_ref=run_id,
        subject_hash=f"subject-{run_id}",
        reuse_key=f"reuse-{run_id}-{subject_type}-{status}",
        request={},
    )
    return db.update_evaluation(evaluation.id, status=status, stage=status)


@pytest.fixture
def db():
    from halo_forge.run_db import RunDatabase

    inst = RunDatabase(":memory:")
    yield inst
    inst.close()


def test_schema_initializes_clean(db):
    """Opening a fresh DB sets schema_version and creates the runs table."""
    cur = db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row["name"] for row in cur.fetchall()]
    assert "runs" in tables
    assert "schema_meta" in tables
    assert "run_lineage" in tables
    cur = db._conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'schema_version'"
    )
    # SCHEMA_VERSION is the source of truth; tests assert against it
    # rather than a hard-coded number so adding a table (e.g. F-J's
    # model_registry) doesn't require touching this test.
    from halo_forge.run_db.schema import SCHEMA_VERSION
    assert cur.fetchone()["value"] == str(SCHEMA_VERSION)


def test_run_lineage_accepts_preallocated_child_before_run_indexing(db):
    db.record_fork(child_run_id="pending-child", parent_run_id="completed-parent")
    lineage = db.get_lineage("pending-child")
    assert lineage["ancestors"][0]["parent_run_id"] == "completed-parent"
    assert db.get_run("pending-child") is None


def test_schema_v5_removes_lineage_child_fk_without_losing_edges(tmp_path):
    from halo_forge.run_db import RunDatabase

    path = tmp_path / "v4-lineage.db"
    legacy = RunDatabase(str(path))
    legacy.upsert_run(_make_record(run_id="legacy-child"))
    with legacy._lock:
        legacy._conn.execute("DROP TABLE run_lineage")
        legacy._conn.execute(
            """
            CREATE TABLE run_lineage (
                child_run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                parent_run_id TEXT NOT NULL,
                forked_at_cycle INTEGER,
                notes TEXT,
                PRIMARY KEY (child_run_id, parent_run_id)
            )
            """
        )
        legacy._conn.execute(
            "INSERT INTO run_lineage VALUES ('legacy-child', 'legacy-parent', 2, 'kept')"
        )
        legacy._conn.execute(
            "UPDATE schema_meta SET value = '4' WHERE key = 'schema_version'"
        )
        legacy._conn.commit()
    legacy.close()

    migrated = RunDatabase(str(path))
    assert migrated._conn.execute("PRAGMA foreign_key_list(run_lineage)").fetchall() == []
    assert migrated.get_parents("legacy-child") == [
        {"parent_run_id": "legacy-parent", "forked_at_cycle": 2, "notes": "kept"}
    ]
    migrated.close()


def test_upsert_and_get(db):
    db.upsert_run(_make_record())
    got = db.get_run("run_alpha")
    assert got is not None
    assert got.modality == "sft"
    assert got.cycles_executed == 3
    assert got.weights_updated is True
    # raw decoded lazily
    assert got.raw == {"foo": "bar"}


def test_upsert_replaces_on_conflict(db):
    db.upsert_run(_make_record(status="running"))
    db.upsert_run(_make_record(status="completed", final_train_loss=0.31))
    got = db.get_run("run_alpha")
    assert got.status == "completed"
    assert got.final_train_loss == 0.31


def test_delete_run(db):
    db.upsert_run(_make_record())
    assert db.delete_run("run_alpha") is True
    assert db.get_run("run_alpha") is None
    assert db.delete_run("does_not_exist") is False


def test_filter_by_modality(db):
    db.upsert_run(_make_record(run_id="r_sft", modality="sft"))
    db.upsert_run(_make_record(run_id="r_raft", modality="raft"))
    db.upsert_run(_make_record(run_id="r_dpo", modality="dpo"))
    from halo_forge.run_db import RunFilter

    only_dpo = db.list_runs(RunFilter(modalities=["dpo"]))
    assert {r.run_id for r in only_dpo} == {"r_dpo"}
    pair = db.list_runs(RunFilter(modalities=["sft", "raft"]))
    assert {r.run_id for r in pair} == {"r_sft", "r_raft"}


def test_filter_by_status_and_model_substring(db):
    db.upsert_run(_make_record(run_id="r1", status="completed", model_name="Qwen/Qwen2.5-3B"))
    db.upsert_run(_make_record(run_id="r2", status="completed", model_name="meta-llama/Llama-3.2-3B"))
    db.upsert_run(_make_record(run_id="r3", status="failed", model_name="Qwen/Qwen2.5-7B"))
    from halo_forge.run_db import RunFilter

    qwen = db.list_runs(RunFilter(model_substring="Qwen"))
    assert {r.run_id for r in qwen} == {"r1", "r3"}
    completed_qwen = db.list_runs(
        RunFilter(model_substring="Qwen", statuses=["completed"])
    )
    assert {r.run_id for r in completed_qwen} == {"r1"}


def test_filter_by_timestamp_range(db):
    db.upsert_run(_make_record(run_id="r_old", timestamp="2026-01-01T00:00:00Z"))
    db.upsert_run(_make_record(run_id="r_mid", timestamp="2026-04-01T00:00:00Z"))
    db.upsert_run(_make_record(run_id="r_new", timestamp="2026-08-01T00:00:00Z"))
    from halo_forge.run_db import RunFilter

    spring = db.list_runs(
        RunFilter(since_iso="2026-02-01", until_iso="2026-06-01")
    )
    assert {r.run_id for r in spring} == {"r_mid"}


def test_filter_by_eval_present(db):
    db.upsert_run(_make_record(run_id="with", final_train_loss=None))
    db.upsert_run(_make_record(run_id="without", final_train_loss=0.5))
    db.upsert_run(_make_record(run_id="running", final_train_loss=0.2))
    _attach_evaluation(db, "with")
    _attach_evaluation(db, "running", status="running")
    from halo_forge.run_db import RunFilter

    has = db.list_runs(RunFilter(has_eval=True))
    assert {r.run_id for r in has} == {"with"}
    none = db.list_runs(RunFilter(has_eval=False))
    assert {r.run_id for r in none} == {"without", "running"}


def test_sort_and_pagination(db):
    for i, ts in enumerate(["2026-01", "2026-02", "2026-03", "2026-04"]):
        db.upsert_run(
            _make_record(run_id=f"r{i}", timestamp=f"{ts}-01T00:00:00Z")
        )
    from halo_forge.run_db import RunFilter

    desc = db.list_runs(RunFilter(sort_by="timestamp", sort_dir="desc", limit=2))
    assert [r.run_id for r in desc] == ["r3", "r2"]

    asc_page2 = db.list_runs(
        RunFilter(sort_by="timestamp", sort_dir="asc", limit=2, offset=2)
    )
    assert [r.run_id for r in asc_page2] == ["r2", "r3"]


def test_unknown_sort_falls_back_to_timestamp(db):
    """Whitelisted sort columns: sneaking SQL through `sort_by` must
    fall back to a known-safe default rather than raise."""
    db.upsert_run(_make_record(run_id="r1", timestamp="2026-01-01T00:00:00Z"))
    db.upsert_run(_make_record(run_id="r2", timestamp="2026-02-01T00:00:00Z"))
    from halo_forge.run_db import RunFilter

    # `; DROP TABLE runs;` — should be ignored, no SQL injection.
    rs = db.list_runs(RunFilter(sort_by="; DROP TABLE runs; --"))
    assert {r.run_id for r in rs} == {"r1", "r2"}
    # Table still exists.
    assert db.count_runs() == 2


def test_count_runs_with_and_without_filter(db):
    db.upsert_run(_make_record(run_id="a", modality="sft"))
    db.upsert_run(_make_record(run_id="b", modality="sft"))
    db.upsert_run(_make_record(run_id="c", modality="raft"))
    from halo_forge.run_db import RunFilter

    assert db.count_runs() == 3
    assert db.count_runs(RunFilter(modalities=["sft"])) == 2


def test_count_runs_applies_eval_and_update_filters(db):
    db.upsert_run(_make_record(run_id="eval-updated", final_train_loss=0.5, weights_updated=True))
    db.upsert_run(_make_record(run_id="eval-not-updated", final_train_loss=0.7, weights_updated=False))
    db.upsert_run(_make_record(run_id="no-eval", final_train_loss=None, weights_updated=True))
    _attach_evaluation(db, "eval-updated")
    _attach_evaluation(db, "eval-not-updated", subject_type="final_model")
    from halo_forge.run_db import RunFilter

    assert db.count_runs(RunFilter(has_eval=True)) == 2
    assert db.count_runs(RunFilter(has_eval=False)) == 1
    assert db.count_runs(RunFilter(weights_updated=True)) == 2
    assert db.count_runs(RunFilter(weights_updated=False)) == 1


def test_distinct_helpers(db):
    db.upsert_run(_make_record(run_id="a", modality="sft", model_name="Qwen/A"))
    db.upsert_run(_make_record(run_id="b", modality="raft", model_name="Qwen/A"))
    db.upsert_run(_make_record(run_id="c", modality="sft", model_name="meta/B"))
    assert db.distinct_modalities() == ["raft", "sft"]
    assert db.distinct_models() == ["Qwen/A", "meta/B"]


def test_modality_counts(db):
    db.upsert_run(_make_record(run_id="a", modality="sft"))
    db.upsert_run(_make_record(run_id="b", modality="sft"))
    db.upsert_run(_make_record(run_id="c", modality="dpo"))
    db.upsert_run(_make_record(run_id="d", modality="grpo"))
    counts = db.modality_counts()
    assert counts == {"sft": 2, "dpo": 1, "grpo": 1}


def test_modality_counts_empty(db):
    assert db.modality_counts() == {}


def test_get_database_caches_per_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "runs.db"))
    from halo_forge.run_db import get_database

    a = get_database()
    b = get_database()
    assert a is b
    a.close()
    # Reopen via fresh path is a different cached instance.
    other = get_database(str(tmp_path / "other.db"))
    assert other is not a


# ----- sync tests -----------------------------------------------------------


def _write_summary(path: Path, **overrides) -> None:
    payload = {
        "id": path.parent.name,
        "run_id": overrides.get("run_id", path.parent.name),
        "modality": overrides.get("modality", "sft"),
        "model_name": overrides.get("model_name", "Qwen/Qwen2.5-3B"),
        "status": overrides.get("status", "completed"),
        "timestamp": overrides.get("timestamp", "2026-04-15T12:00:00+00:00"),
        "cycles_executed": overrides.get("cycles_executed", 1),
        "weights_updated": overrides.get("weights_updated", True),
        "metrics_summary": {
            "final_train_loss": overrides.get("final_train_loss", 0.5),
            "update_steps": overrides.get("update_steps", 100),
        },
    }
    payload.update(
        {k: v for k, v in overrides.items() if k not in payload and k != "metrics_summary"}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_sync_walks_summaries(tmp_path, db):
    from halo_forge.run_db import sync_from_filesystem

    _write_summary(tmp_path / "models" / "alpha" / "training_summary.json", run_id="alpha")
    _write_summary(
        tmp_path / "outputs" / "beta" / "training_summary.json",
        run_id="beta",
        modality="raft",
    )
    # A non-summary file in the tree should be ignored.
    (tmp_path / "outputs" / "junk.txt").write_text("hello")

    upserted = sync_from_filesystem(
        db, roots=[tmp_path / "models", tmp_path / "outputs"]
    )
    assert upserted == 2
    runs = db.list_runs()
    assert {r.run_id for r in runs} == {"alpha", "beta"}


def test_sync_is_incremental(tmp_path, db):
    from halo_forge.run_db import sync_from_filesystem

    summary = tmp_path / "models" / "x" / "training_summary.json"
    _write_summary(summary, run_id="x")

    assert sync_from_filesystem(db, roots=[tmp_path / "models"]) == 1
    # Running again without changes should upsert nothing.
    assert sync_from_filesystem(db, roots=[tmp_path / "models"]) == 0


def test_sync_force_re_upserts_everything(tmp_path, db):
    from halo_forge.run_db import sync_from_filesystem

    summary = tmp_path / "models" / "x" / "training_summary.json"
    _write_summary(summary, run_id="x")

    assert sync_from_filesystem(db, roots=[tmp_path / "models"]) == 1
    assert sync_from_filesystem(db, roots=[tmp_path / "models"], force=True) == 1


def test_sync_picks_up_modified_summaries(tmp_path, db):
    """When the JSON gets rewritten with a newer mtime, sync re-upserts."""
    import os
    import time
    from halo_forge.run_db import sync_from_filesystem

    summary = tmp_path / "models" / "x" / "training_summary.json"
    _write_summary(summary, run_id="x", final_train_loss=0.5)
    sync_from_filesystem(db, roots=[tmp_path / "models"])

    time.sleep(0.01)  # ensure mtime tick
    _write_summary(summary, run_id="x", final_train_loss=0.1)
    # Force the mtime forward in case the FS resolution is coarse.
    new_mtime = summary.stat().st_mtime + 5
    os.utime(summary, (new_mtime, new_mtime))

    assert sync_from_filesystem(db, roots=[tmp_path / "models"]) == 1
    record = db.get_run("x")
    assert record.final_train_loss == 0.1


def test_sync_skips_unreadable_summaries(tmp_path, db):
    """Bad JSON should log a warning and be skipped — not fail the whole run."""
    from halo_forge.run_db import sync_from_filesystem

    bad = tmp_path / "models" / "broken" / "training_summary.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not valid json")
    _write_summary(tmp_path / "models" / "good" / "training_summary.json", run_id="good")

    upserted = sync_from_filesystem(db, roots=[tmp_path / "models"])
    assert upserted == 1
    assert db.get_run("good") is not None


def test_sync_handles_missing_root(tmp_path, db):
    from halo_forge.run_db import sync_from_filesystem

    # Path doesn't exist; should be silently skipped, not crash.
    upserted = sync_from_filesystem(db, roots=[tmp_path / "does_not_exist"])
    assert upserted == 0


def test_sync_record_pulls_metrics_from_metrics_summary(tmp_path, db):
    """Final loss / update steps come from the nested metrics_summary
    block when the top-level keys are absent."""
    from halo_forge.run_db import sync_from_filesystem

    _write_summary(
        tmp_path / "models" / "x" / "training_summary.json",
        run_id="x",
        final_train_loss=0.123,
        update_steps=42,
    )
    sync_from_filesystem(db, roots=[tmp_path / "models"])
    record = db.get_run("x")
    assert record.final_train_loss == 0.123
    assert record.total_train_steps == 42
