from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3
import threading

from halo_forge.run_db import RunDatabase
from halo_forge.workstation_jobs import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
    WorkstationScheduler,
)

GIB = 1024**3


def _available_capacity(_path):
    return WorkstationCapacity(
        sampled_at=datetime.now(timezone.utc),
        disk=DiskCapacity(
            path="/tmp/halo-forge-tests",
            total_bytes=200 * GIB,
            used_bytes=100 * GIB,
            free_bytes=100 * GIB,
        ),
        memory=MemoryCapacity(
            total_bytes=32 * GIB,
            used_bytes=8 * GIB,
            available_bytes=24 * GIB,
        ),
    )


def test_schema_v7_migrates_suite_purpose_evidence_and_control_plane_defaults(tmp_path):
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta VALUES ('schema_version', '5');
        CREATE TABLE benchmark_suites (
            id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, description TEXT,
            latest_revision_id TEXT, archived INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        INSERT INTO benchmark_suites VALUES
            ('suite', 'Legacy', NULL, NULL, 0, '2026-01-01', '2026-01-01');
        CREATE TABLE evaluation_samples (
            evaluation_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
            suite_item_id TEXT NOT NULL, record_id TEXT, input_json TEXT,
            expected_json TEXT, output_json TEXT, score REAL, passed INTEGER,
            latency_ms REAL, error TEXT, verifier_trace_json TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (evaluation_id, ordinal)
        );
        INSERT INTO evaluation_samples
            (evaluation_id, ordinal, suite_item_id, metadata_json)
        VALUES ('evaluation', 0, 'item', '{}');
        """)
    connection.commit()
    connection.close()

    db = RunDatabase(str(path))
    suite = db.get_benchmark_suite("suite")
    assert suite is not None and suite.purpose == "unspecified"
    sample = db.list_evaluation_samples("evaluation")[0]
    assert sample.evidence_kind == "legacy"
    assert sample.valid is False and sample.mineable is False
    assert (
            db._conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()[
                "value"
            ]
                == "23"
        )


def test_transactional_priority_dependency_and_single_heavy_lease(tmp_path):
    path = tmp_path / "runs.db"
    first_db = RunDatabase(str(path))
    second_db = RunDatabase(str(path))
    first = WorkstationScheduler(first_db, worker_id="first", capacity_probe=_available_capacity)
    second = WorkstationScheduler(second_db, worker_id="second", capacity_probe=_available_capacity)

    prerequisite = first.enqueue(kind="render", priority=1, work_item_id="prerequisite")
    first.enqueue(
        kind="train",
        priority=100,
        dependencies=[prerequisite.id],
        work_item_id="dependent",
    )
    first.enqueue(kind="evaluate", priority=10, work_item_id="ready")

    claimed = first.claim()
    assert claimed is not None and claimed.id == "ready"
    assert second.claim() is None
    finished = first.complete(claimed, result={"ok": True})
    assert finished is not None and finished.status == "completed"

    claimed_prerequisite = second.claim()
    assert claimed_prerequisite is not None
    assert claimed_prerequisite.id == "prerequisite"
    second.complete(claimed_prerequisite)
    claimed_dependent = first.claim()
    assert claimed_dependent is not None and claimed_dependent.id == "dependent"

    first_db.close()
    second_db.close()


def test_concurrent_connections_cannot_double_claim_heavy_work(tmp_path):
    path = tmp_path / "runs.db"
    setup = RunDatabase(str(path))
    setup.create_work_item(kind="train", work_item_id="one")
    setup.create_work_item(kind="evaluate", work_item_id="two")
    setup.close()
    barrier = threading.Barrier(2)
    claimed: list[str | None] = []

    def claim(worker_id: str) -> None:
        database = RunDatabase(str(path))
        scheduler = WorkstationScheduler(
            database, worker_id=worker_id, capacity_probe=_available_capacity
        )
        barrier.wait()
        item = scheduler.claim()
        claimed.append(item.id if item else None)
        database.close()

    first = threading.Thread(target=claim, args=("first",))
    second = threading.Thread(target=claim, args=("second",))
    first.start()
    second.start()
    first.join()
    second.join()
    assert len([value for value in claimed if value is not None]) == 1


def test_cpu_and_accelerator_calibrations_are_globally_serialized(tmp_path):
    path = tmp_path / "calibrations.db"
    first_db = RunDatabase(str(path))
    second_db = RunDatabase(str(path))
    cpu_worker = WorkstationScheduler(
        first_db, worker_id="cpu-worker", capacity_probe=_available_capacity
    )
    accelerator_worker = WorkstationScheduler(
        second_db, worker_id="accelerator-worker", capacity_probe=_available_capacity
    )
    try:
        cpu_worker.enqueue(
            kind="verifier_calibration",
            resource_class="cpu",
            priority=2,
            domain_kind="verifier_calibration",
            domain_id="cpu-calibration",
            work_item_id="cpu-calibration-work",
        )
        cpu_worker.enqueue(
            kind="verifier_calibration",
            resource_class="accelerator",
            priority=1,
            domain_kind="verifier_calibration",
            domain_id="accelerator-calibration",
            work_item_id="accelerator-calibration-work",
        )

        cpu_claim = cpu_worker.claim()
        assert cpu_claim is not None and cpu_claim.id == "cpu-calibration-work"
        assert accelerator_worker.claim() is None

        cpu_worker.complete(cpu_claim)
        accelerator_claim = accelerator_worker.claim()
        assert accelerator_claim is not None
        assert accelerator_claim.id == "accelerator-calibration-work"
    finally:
        first_db.close()
        second_db.close()


def test_cpu_work_can_run_while_accelerator_is_leased(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(db, worker_id="worker", capacity_probe=_available_capacity)
    scheduler.enqueue(kind="train", resource_class="accelerator", priority=2)
    scheduler.enqueue(kind="profile", resource_class="cpu", priority=1)

    heavy = scheduler.claim()
    assert heavy is not None and heavy.resource_class == "accelerator"
    cpu = scheduler.claim()
    assert cpu is not None and cpu.resource_class == "cpu"
    assert len(db.list_resource_leases()) == 1


def test_retained_serving_lease_requires_explicit_release(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(db, worker_id="worker", capacity_probe=_available_capacity)
    scheduler.enqueue(kind="train", work_item_id="train")

    lease = scheduler.start_serving(serving_id="server-1", metadata={"model": "local"})
    assert lease is not None and lease.retained is True and lease.expires_at is None
    assert scheduler.claim() is None

    future = datetime.now(timezone.utc) + timedelta(days=30)
    assert db.recover_stale_work_items(now=future) == []
    assert db.get_resource_lease("accelerator") is not None
    assert scheduler.stop_serving(serving_id="other") is False
    assert scheduler.stop_serving(serving_id="server-1") is True
    assert scheduler.claim() is not None


def test_expired_lease_interrupts_owner_and_unblocks_next_item(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(
        db,
        worker_id="worker",
        lease_ttl_seconds=5,
        capacity_probe=_available_capacity,
    )
    scheduler.enqueue(kind="train", priority=2, work_item_id="first")
    scheduler.enqueue(kind="evaluate", priority=1, work_item_id="second")
    start = datetime(2026, 7, 14, tzinfo=timezone.utc)

    claimed = scheduler.claim(now=start)
    assert claimed is not None and claimed.id == "first"
    next_item = scheduler.claim(now=start + timedelta(seconds=6))
    assert next_item is not None and next_item.id == "second"
    interrupted = db.get_work_item("first")
    assert interrupted is not None and interrupted.status == "interrupted"
    assert interrupted.error == "resource lease expired"


def test_cancel_finish_and_explicit_retry(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(db, worker_id="worker", capacity_probe=_available_capacity)
    scheduler.enqueue(kind="train", work_item_id="item")
    claimed = scheduler.claim()
    assert claimed is not None

    requested = scheduler.cancel(claimed.id)
    assert requested is not None and requested.cancel_requested is True
    finished = scheduler.complete(claimed)
    assert finished is not None and finished.status == "cancelled"
    retried = scheduler.retry(claimed.id)
    assert retried is not None and retried.status == "queued"
    assert retried.retry_count == 1 and retried.cancel_requested is False


def test_restart_adopts_matching_pid_and_reconciles_dead_pid(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    owner = WorkstationScheduler(db, worker_id="owner")
    owner.enqueue(kind="train", resource_class="cpu", work_item_id="live")
    live = owner.claim(child_pid=11, child_pid_started_at=101.0)
    assert live is not None
    owner.enqueue(kind="profile", resource_class="cpu", work_item_id="dead")
    dead = owner.claim(child_pid=12, child_pid_started_at=102.0)
    assert dead is not None

    recovering = WorkstationScheduler(
        db,
        worker_id="new-owner",
        process_probe=lambda pid, started: pid == 11 and started == 101.0,
    )
    outcome = recovering.recover_or_adopt()
    assert [item.id for item in outcome.adopted] == ["live"]
    assert [item.id for item in outcome.interrupted] == ["dead"]
    assert db.get_work_item("live").status == "running"  # type: ignore[union-attr]
    assert db.get_work_item("dead").status == "needs_reconciliation"  # type: ignore[union-attr]


def test_dependency_cycle_is_rejected():
    db = RunDatabase(":memory:")
    db.create_work_item(kind="a", resource_class="cpu", work_item_id="a")
    db.create_work_item(kind="b", resource_class="cpu", dependencies=["a"], work_item_id="b")
    try:
        db.add_work_item_dependency("a", "b")
    except ValueError as exc:
        assert "cycle" in str(exc)
    else:
        raise AssertionError("dependency cycle was accepted")


def test_run_group_artifact_and_exposure_records_round_trip():
    db = RunDatabase(":memory:")
    suite = db.create_benchmark_suite(name="dev", purpose="development")
    revision = db.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="suite-hash",
        items=[{"id": "item-1", "input": "x"}],
        primary_metric="accuracy",
        direction="maximize",
    )
    group = db.create_run_group(
        name="repeat",
        kind="repeat",
        trainer_mode="sft",
        resolved_launch_config={"epochs": 1},
        development_suite_revision_id=revision.id,
        seeds=[7, 8, 9],
    )
    trial = db.create_run_group_trial(
        run_group_id=group.id,
        ordinal=0,
        config_hash="cfg",
        sampled_config={"learning_rate": 0.001},
        required_seed_count=3,
    )
    trial_run = db.create_trial_run(trial_id=trial.id, run_id="run-7", ordinal=0, seed=7)
    segment = db.create_trial_segment(
        trial_run_id=trial_run.id,
        ordinal=0,
        unit="step",
        start_value=0,
        end_value=100,
    )
    artifact = db.create_model_artifact(
        artifact_hash="hash",
        artifact_kind="checkpoint",
        run_id="run-7",
        run_group_id=group.id,
        trial_id=trial.id,
        trial_segment_id=segment.id,
        model_id="model",
        backend="transformers",
        format="safetensors",
        path="/tmp/checkpoint",
        step=100,
    )
    db.update_trial_segment(
        segment.id, status="completed", checkpoint_artifact_id=artifact.id, decision="continue"
    )
    exposure = db.record_exposure(
        suite_revision_id=revision.id,
        suite_item_id="item-1",
        exposure_type="failure_mining",
        run_id="run-7",
        provenance={"evaluation_id": "eval-1"},
    )
    inherited = db.inherit_exposures(
        [exposure], model_artifact_id=artifact.id, provenance={"via": "training"}
    )

    assert db.get_run_group(group.id).seeds == [7, 8, 9]  # type: ignore[union-attr]
    assert db.list_run_group_trials(group.id)[0].sampled_config["learning_rate"] == 0.001
    assert db.list_model_artifacts(run_id="run-7")[0].id == artifact.id
    assert inherited[0].inherited_from_id == exposure.id
    assert db.list_exposures(model_artifact_id=artifact.id)[0].provenance == {"via": "training"}
