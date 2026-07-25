from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from halo_forge.managed_runtime import (
    ManagedRuntimeService,
    RuntimeMount,
    probe_cuda,
    probe_rocm,
    wait_for_stable_idle,
)
from halo_forge.managed_runtime.adapters import PodmanRocmRuntimeAdapter
from halo_forge.managed_runtime.models import (
    AcceleratorAvailability,
    ExternalAcceleratorOwner,
)
from halo_forge.run_db import RunDatabase
from halo_forge.run_db.schema import SCHEMA_VERSION
from halo_forge.workstation_jobs import WorkstationScheduler


def _availability(state: str, family: str = "rocm") -> AcceleratorAvailability:
    owners = (
        (ExternalAcceleratorOwner(88, "other-trainer", 42),)
        if state == "busy"
        else ()
    )
    return AcceleratorAvailability(
        family,
        state,
        "2026-07-17T00:00:00+00:00",
        100.0 if state == "busy" else 0.0 if state == "idle" else None,
        owners,
        "external compute owns the accelerator" if state == "busy" else None,
    )


def test_schema_v22_and_builtin_runtime_identity(tmp_path: Path) -> None:
    from halo_forge.managed_runtime.cli import _serializable

    db = RunDatabase(str(tmp_path / "runs.db"))
    service = ManagedRuntimeService(db, root=tmp_path / "runtimes")
    assert SCHEMA_VERSION == 23
    assert db._conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()[0] == "23"
    strix = service.get_profile("strix-halo-rocm-7.2.1")
    revision = service.get_revision(str(strix.latest_revision_id))
    assert revision.base_image_digest == (
        "sha256:96a2fb24dec9896e2f8238178f0c49d0dcc4c7dcc597be09e4564316bd86d191"
    )
    assert revision.dependency_lock["transformers"] == "5.13.1"
    assert _serializable(service.list_profiles())[0]["id"]
    with pytest.raises(Exception, match="immutable"):
        db._conn.execute(
            "UPDATE managed_runtime_revisions SET engine='docker' WHERE id=?",
            (revision.id,),
        )


def test_podman_rocm_is_argv_only_rootless_and_path_preserving(tmp_path: Path) -> None:
    adapter = PodmanRocmRuntimeAdapter()
    source = tmp_path / "data"
    source.mkdir()
    wrapped = adapter.wrap(
        ("python", "-m", "halo_forge.cli", "sft", "train"),
        image="localhost/halo-forge-rocm:test",
        mounts=(RuntimeMount(source, source),),
        env={"HF_HOME": str(tmp_path / "cache")},
        name="safe-name",
    )
    assert wrapped.argv[:2] == ("podman", "run")
    assert "--privileged" not in wrapped.argv
    assert "/var/run/docker.sock" not in " ".join(wrapped.argv)
    assert "/dev/kfd" in wrapped.argv and "/dev/dri" in wrapped.argv
    assert str(source.resolve()) in " ".join(wrapped.argv)
    with pytest.raises(ValueError, match="credentials"):
        adapter.wrap(("true",), image="x", env={"HF_TOKEN": "secret"})


def test_rocm_and_cuda_occupancy_are_conservative(monkeypatch: pytest.MonkeyPatch) -> None:
    import halo_forge.managed_runtime.occupancy as occupancy

    monkeypatch.setattr(occupancy, "_kfd_owners", lambda excluded: ())
    rocm = probe_rocm(
        runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, json.dumps({"card0": {"GPU use (%)": "0"}}), ""
        )
    )
    assert rocm.state == "idle"
    malformed = probe_rocm(
        runner=lambda argv: subprocess.CompletedProcess(argv, 0, "{}", "")
    )
    assert malformed.state == "unknown"
    cuda = probe_cuda(
        runner=lambda argv: subprocess.CompletedProcess(
            argv, 0, "445, /opt/train.py, 1024\n", ""
        )
    )
    assert cuda.state == "busy"
    assert cuda.owners[0].executable == "train.py"
    assert cuda.owners[0].memory_bytes == 1024 * 1024 * 1024


def test_three_idle_samples_and_busy_short_circuit() -> None:
    states = iter((_availability("idle"), _availability("idle"), _availability("idle")))
    idle, samples = wait_for_stable_idle(
        "rocm", probe=lambda family: next(states), sleeper=lambda seconds: None
    )
    assert idle and len(samples) == 3
    states = iter((_availability("idle"), _availability("busy"), _availability("idle")))
    idle, samples = wait_for_stable_idle(
        "rocm", probe=lambda family: next(states), sleeper=lambda seconds: None
    )
    assert not idle and len(samples) == 2


def test_scheduler_waits_without_consuming_retry(tmp_path: Path) -> None:
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(
        db,
        accelerator_probe=lambda family: _availability("busy", family),
        idle_sample_interval_seconds=0,
    )
    item = scheduler.enqueue(
        kind="training",
        launch_spec={"command": ["true"]},
        resource_class="accelerator",
        resource_requirements={
            "accelerator_family": "rocm",
            "output_path": str(tmp_path),
            "capacity_override_reason": "unit test isolates accelerator queue semantics",
        },
        max_retries=2,
    )
    assert scheduler.claim(work_item_id=item.id) is None
    waiting = db.get_work_item(item.id)
    assert waiting.status == "queued"
    assert waiting.stage == "waiting_for_accelerator"
    assert waiting.retry_count == 0
    assert db.get_resource_lease("accelerator") is None
    decision = db._conn.execute(
        "SELECT decision,evidence_json FROM accelerator_preflight_decisions WHERE work_item_id=?",
        (item.id,),
    ).fetchone()
    assert decision["decision"] == "waiting"
    assert "other-trainer" in decision["evidence_json"]


def test_fake_full_qualification_publishes_current_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "halo_forge.managed_runtime.service.wait_for_stable_idle",
        lambda family, probe: (
            True,
            (probe(family), probe(family), probe(family)),
        ),
    )
    monkeypatch.setattr(
        "halo_forge.managed_runtime.adapters.PodmanRocmRuntimeAdapter.available",
        lambda self: (True, None),
    )
    db = RunDatabase(str(tmp_path / "runs.db"))

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, json.dumps({"ok": True}), "")

    service = ManagedRuntimeService(
        db,
        root=tmp_path / "runtimes",
        runner=runner,
        occupancy_probe=lambda family: _availability("idle", family),
    )
    revision = service.list_revisions("strix-halo-rocm-7.2.1")[0]
    now = "2026-07-17T00:00:00+00:00"
    db._conn.execute(
        """INSERT INTO runtime_preparations
           (id,runtime_revision_id,status,stage,engine,image_id,image_digest,
            storage_path,progress_json,created_at,completed_at)
           VALUES (?,?, 'completed','completed','podman','image',?,?, '{}',?,?)""",
        (
            "prep",
            revision.id,
            revision.base_image_digest,
            str(tmp_path / "runtimes"),
            now,
            now,
        ),
    )
    db._conn.commit()
    qualification = service.qualify(revision.id, enqueue=False)
    completed = service.run_qualification(qualification.id)
    assert completed.status in {"local_verified", "vendor_supported"}
    assert all(step.status == "passed" for step in completed.steps)
    capability = next(
        value for value in service.capabilities() if value.accelerator_family == "rocm"
    )
    assert capability.available is True
    assert service.verify(completed.id)["valid"] is True


def test_runtime_api_lists_profiles_and_requires_download_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient
    from halo_forge.public_api.app import create_app

    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("HALOFORGE_RUNTIME_ROOT", str(tmp_path / "runtimes"))
    client = TestClient(create_app(serve_frontend=False))
    listed = client.get("/api/public/runtimes")
    assert listed.status_code == 200
    assert listed.json()["total"] == 2
    revision_id = listed.json()["items"][0]["revision"]["id"]
    refused = client.post(
        f"/api/public/runtime-revisions/{revision_id}/prepare", json={"confirmed": False}
    )
    assert refused.status_code == 400
    paths = set(client.get("/api/public/openapi.json").json()["paths"])
    assert "/api/public/accelerator/availability" in paths
    assert "/api/public/runtime-qualifications/{qualification_id}/verify" in paths
