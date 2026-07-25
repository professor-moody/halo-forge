"""Cross-process adoption safety for the workstation recovery path.

A second supervisor must never take over work that a *live* worker process
still owns. The only safe evidence that a foreign claim is abandoned is the
registered ``(pid, pid_started_at)`` identity of the owning worker process --
never the liveness of the training child, which legitimately exits minutes
before the owning worker finishes publication.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from halo_forge.run_db import RunDatabase
from halo_forge.workstation_jobs import WorkstationScheduler

WORKER_A_PID = 4242
WORKER_A_PID_STARTED_AT = 1000.0
CHILD_PID = 5150
CHILD_PID_STARTED_AT = 2000.0


def _worker_a_with_finished_child(db: RunDatabase) -> tuple[WorkstationScheduler, str]:
    """Claim one item as worker A, then attach a training child that has exited."""
    owner = WorkstationScheduler(db, worker_id="worker-a")
    owner.enqueue(kind="train", resource_class="cpu", work_item_id="training")
    claimed = owner.claim(
        child_pid=WORKER_A_PID,
        child_pid_started_at=WORKER_A_PID_STARTED_AT,
    )
    assert claimed is not None
    bound = owner.bind_process(
        claimed,
        child_pid=CHILD_PID,
        child_pid_started_at=CHILD_PID_STARTED_AT,
    )
    assert bound is not None and bound.claim_token
    return owner, str(bound.claim_token)


def _probe(alive: set[tuple[Optional[int], Optional[float]]]):
    def probe(pid: Optional[int], started: Optional[float]) -> bool:
        return (pid, started) in alive

    return probe


def test_list_work_items_worker_filter_is_backward_compatible(tmp_path):
    db = RunDatabase(str(tmp_path / "runs.db"))
    try:
        _worker_a_with_finished_child(db)
        assert [item.id for item in db.list_work_items(statuses=["running"])] == ["training"]
        assert [
            item.id
            for item in db.list_work_items(statuses=["running"], worker_id="worker-a")
        ] == ["training"]
        assert db.list_work_items(statuses=["running"], worker_id="worker-b") == []
    finally:
        db.close()


def test_recovery_leaves_items_owned_by_a_live_worker_alone(tmp_path):
    """Worker B must not adopt or interrupt work owned by a live worker A."""
    db = RunDatabase(str(tmp_path / "runs.db"))
    try:
        _owner, claim_token = _worker_a_with_finished_child(db)

        # Worker A's supervisor process is alive and publishing; its training
        # child has already exited.
        recovering = WorkstationScheduler(
            db,
            worker_id="worker-b",
            process_probe=_probe({(WORKER_A_PID, WORKER_A_PID_STARTED_AT)}),
        )
        outcome = recovering.recover_or_adopt()

        assert [item.id for item in outcome.adopted] == []
        assert [item.id for item in outcome.interrupted] == []
        item = db.get_work_item("training")
        assert item is not None
        assert item.status == "running"
        assert item.worker_id == "worker-a"
        assert item.claim_token == claim_token
    finally:
        db.close()


def test_recovery_reconciles_items_whose_owning_worker_is_dead(tmp_path):
    """A provably dead worker identity releases its claim to the new supervisor."""
    db = RunDatabase(str(tmp_path / "runs.db"))
    try:
        _worker_a_with_finished_child(db)

        recovering = WorkstationScheduler(
            db,
            worker_id="worker-b",
            process_probe=_probe(set()),
        )
        outcome = recovering.recover_or_adopt()

        assert [item.id for item in outcome.interrupted] == ["training"]
        item = db.get_work_item("training")
        assert item is not None and item.status == "needs_reconciliation"
    finally:
        db.close()


def test_recovery_adopts_live_child_of_a_dead_worker(tmp_path):
    """A surviving child of a crashed worker is still adoptable."""
    db = RunDatabase(str(tmp_path / "runs.db"))
    try:
        _owner, claim_token = _worker_a_with_finished_child(db)

        recovering = WorkstationScheduler(
            db,
            worker_id="worker-b",
            process_probe=_probe({(CHILD_PID, CHILD_PID_STARTED_AT)}),
        )
        outcome = recovering.recover_or_adopt()

        assert [item.id for item in outcome.adopted] == ["training"]
        assert [item.id for item in outcome.interrupted] == []
        item = db.get_work_item("training")
        assert item is not None
        assert item.status == "running"
        assert item.claim_token == claim_token
    finally:
        db.close()


def test_incomplete_owner_identity_defers_to_lease_expiry(tmp_path):
    """An unverifiable owner is left alone now, but is not stranded forever.

    Treating "cannot prove dead" as alive is only safe because the heartbeat
    lease still reaps the claim. Without that backstop the conservative branch
    would hold a crashed worker's item in ``running`` indefinitely.
    """
    db = RunDatabase(str(tmp_path / "runs.db"))
    try:
        owner = WorkstationScheduler(db, worker_id="worker-a")
        owner.enqueue(kind="train", resource_class="cpu", work_item_id="training")
        # A pid with no resolvable start time registers an incomplete identity.
        assert owner.claim(child_pid=99999999) is not None
        assert db.worker_process_identity("worker-a") == (99999999, None)

        recovering = WorkstationScheduler(
            db,
            worker_id="worker-b",
            process_probe=_probe(set()),
            lease_ttl_seconds=30,
        )

        immediate = recovering.recover_or_adopt()
        assert [item.id for item in immediate.adopted] == []
        assert [item.id for item in immediate.interrupted] == []
        held = db.get_work_item("training")
        assert held is not None and held.status == "running"

        expired = recovering.recover_or_adopt(
            now=datetime.now(timezone.utc) + timedelta(seconds=120)
        )
        assert [item.id for item in expired.interrupted] == ["training"]
        released = db.get_work_item("training")
        assert released is not None and released.status == "interrupted"
    finally:
        db.close()


def test_recovery_still_handles_its_own_items(tmp_path):
    """Self-recovery after a restart is unchanged by the ownership filter."""
    db = RunDatabase(str(tmp_path / "runs.db"))
    try:
        owner = WorkstationScheduler(
            db,
            worker_id="worker-a",
            process_probe=_probe(set()),
        )
        owner.enqueue(kind="train", resource_class="cpu", work_item_id="training")
        claimed = owner.claim(child_pid=CHILD_PID, child_pid_started_at=CHILD_PID_STARTED_AT)
        assert claimed is not None

        outcome = owner.recover_or_adopt()

        assert [item.id for item in outcome.interrupted] == ["training"]
        item = db.get_work_item("training")
        assert item is not None and item.status == "needs_reconciliation"
    finally:
        db.close()
