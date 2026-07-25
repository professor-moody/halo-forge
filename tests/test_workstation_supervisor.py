from __future__ import annotations

from datetime import datetime, timedelta, timezone


def test_supervisor_maintenance_keeps_idle_worker_online_and_prunes_raw_telemetry(
    tmp_path,
):
    from halo_forge.run_db import LabV4Catalog, RunDatabase
    from halo_forge.workstation_jobs.supervisor import WorkerSupervisor

    database = RunDatabase(str(tmp_path / "runs.db"))
    supervisor = WorkerSupervisor(database)
    catalog = LabV4Catalog(database)
    try:
        catalog.register_worker(
            worker_id=supervisor.scheduler.worker_id,
            pid=1,
            pid_started_at=1.0,
        )
        database.create_work_item(kind="telemetry-fixture", work_item_id="work")
        old = datetime.now(timezone.utc) - timedelta(days=31)
        catalog.record_telemetry(
            {
                "work_item_id": "work",
                "sampled_at": old.isoformat(),
                "disk": {},
                "memory": {},
                "errors": [],
            }
        )
        supervisor._last_telemetry_prune = old

        before = catalog.get_worker(supervisor.scheduler.worker_id)
        supervisor._maintenance_once(now=datetime.now(timezone.utc))
        after = catalog.get_worker(supervisor.scheduler.worker_id)

        assert after is not None and after.status == "online"
        assert before is not None and after.heartbeat_at >= before.heartbeat_at
        assert catalog.list_telemetry("work") == []
    finally:
        database.close()
