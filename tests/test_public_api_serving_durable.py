from __future__ import annotations

import os
from pathlib import Path

from halo_forge.artifact_studio import SubprocessServingStarter
from halo_forge.public_api.service import PublicApiService
from halo_forge.run_db import RunDatabase


def _service(tmp_path: Path):
    database = RunDatabase(str(tmp_path / "runs.db"))
    service = PublicApiService(
        database=database,
        base_path=tmp_path,
        dataset_storage_root=tmp_path / "datasets",
        evaluation_storage_root=tmp_path / "evaluations",
        artifact_storage_root=tmp_path / "artifacts",
    )
    return service, database


def test_public_api_uses_durable_serving_launcher_and_projects_adopted_status(
    tmp_path: Path,
) -> None:
    service, database = _service(tmp_path)
    assert isinstance(service._artifact_studio_engine().serving_starter, SubprocessServingStarter)
    lease = database.acquire_serving_lease(
        holder_id="artifact-serving-one",
        holder_pid=4242,
        holder_pid_started_at=123.5,
        metadata={
            "state": "serving",
            "occurrence_id": "artifact-one",
            "profile_revision_id": "profile-one",
            "launcher_result": {
                "model": "/models/candidate",
                "backend": "mlx",
                "host": "127.0.0.1",
                "port": 8123,
                "url": "http://127.0.0.1:8123/v1",
                "log_path": "/tmp/serve.log",
            },
        },
    )
    assert lease is not None
    service._process_identity_live = lambda pid, started_at: (  # type: ignore[method-assign]
        pid,
        started_at,
    ) == (4242, 123.5)

    status = service.serve_status()

    assert status["running"] is True
    assert status["state"] == "running"
    assert status["url"] == "http://127.0.0.1:8123/v1"
    assert status["artifact_id"] == "artifact-one"
    assert status["serving_profile_revision_id"] == "profile-one"
    database.close()


def test_reserved_serving_lease_can_be_stopped_without_terminating_worker(
    tmp_path: Path,
) -> None:
    service, database = _service(tmp_path)
    lease = database.acquire_serving_lease(
        holder_id="reserved-serving",
        holder_pid=os.getpid(),
        holder_pid_started_at=1.0,
        metadata={"state": "reserved"},
    )
    assert lease is not None

    stopped = service.serve_stop()

    assert stopped["stopped"] is True
    assert stopped["resource_lease"] is None
    assert database.get_resource_lease("accelerator") is None
    database.close()
