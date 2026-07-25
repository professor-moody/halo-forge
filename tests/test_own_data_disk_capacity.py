"""Focused V9 disk forecasts for managed own-data imports."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from halo_forge.own_data import (
    GuidedOwnDataService,
    InsufficientDiskCapacityError,
)
from halo_forge.run_db import RunDatabase
from halo_forge.workstation_jobs import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
    WorkstationScheduler,
)


GIB = 1024**3


def _capacity(path: Path, *, free_gib: int) -> WorkstationCapacity:
    return WorkstationCapacity(
        sampled_at=datetime.now(timezone.utc),
        disk=DiskCapacity(
            path=str(path),
            total_bytes=100 * GIB,
            used_bytes=(100 - free_gib) * GIB,
            free_bytes=free_gib * GIB,
        ),
        memory=MemoryCapacity(
            total_bytes=32 * GIB,
            used_bytes=8 * GIB,
            available_bytes=24 * GIB,
        ),
    )


def test_upload_staging_refuses_low_disk_and_records_explicit_override(
    tmp_path: Path,
) -> None:
    database = RunDatabase(tmp_path / "runs.db")
    scheduler = WorkstationScheduler(
        database, capacity_probe=lambda path: _capacity(path, free_gib=21)
    )
    service = GuidedOwnDataService(
        database,
        datasets_root=tmp_path / "datasets",
        imports_root=tmp_path / "imports",
        scheduler=scheduler,
    )
    try:
        session = service.create_import({"source_kind": "upload", "name": "Large"})
        declaration = {
            "relative_path": "data.jsonl",
            "size_bytes": 2 * GIB,
            "content_hash": "a" * 64,
        }
        with pytest.raises(InsufficientDiskCapacityError) as caught:
            service.create_import_file(session["id"], declaration)
        assert caught.value.forecast["phase"] == "upload_staging"
        assert caught.value.forecast["blockers"] == ["insufficient_disk"]
        assert caught.value.forecast["projected_free_bytes"] == 19 * GIB

        created = service.create_import_file(
            session["id"],
            {
                **declaration,
                "capacity_override_reason": "This is a reviewed disposable test import.",
            },
        )
        assert created["size_bytes"] == 2 * GIB
        refreshed = service.get_import(session["id"])
        assert refreshed is not None
        assert refreshed["disk_forecast"]["requires_override"] is False
        history = refreshed["disk_forecast"]["override_history"]
        assert history[-1]["phase"] == "upload_staging"
        assert history[-1]["reason"] == "This is a reviewed disposable test import."
        assert refreshed["readiness"]["ready"] is True
    finally:
        database.close()


def test_managed_publication_rechecks_capacity_and_persists_override(
    tmp_path: Path,
) -> None:
    free = {"gib": 100}
    database = RunDatabase(tmp_path / "runs.db")
    scheduler = WorkstationScheduler(
        database, capacity_probe=lambda path: _capacity(path, free_gib=free["gib"])
    )
    service = GuidedOwnDataService(
        database,
        datasets_root=tmp_path / "datasets",
        imports_root=tmp_path / "imports",
        scheduler=scheduler,
    )
    content = b'{"prompt":"hello","response":"world"}\n'
    try:
        session = service.create_import({"source_kind": "upload", "name": "Tiny"})
        file_record = service.create_import_file(
            session["id"],
            {
                "relative_path": "data.jsonl",
                "size_bytes": len(content),
                "content_hash": hashlib.sha256(content).hexdigest(),
            },
        )
        service.upload_chunk(
            session["id"],
            file_record["id"],
            content,
            start=0,
            end=len(content) - 1,
            total=len(content),
        )

        free["gib"] = 20
        with pytest.raises(InsufficientDiskCapacityError) as caught:
            service.imports.publish(session["id"])
        assert caught.value.forecast["phase"] == "managed_publication"

        path, fingerprint = service.imports.publish(
            session["id"], override_reason="Reviewed publication despite low reserve."
        )
        assert path.is_dir()
        assert len(fingerprint) == 64
        stored = database.get_dataset_import(session["id"])
        history = stored.metadata["capacity"]["override_history"]
        assert history[-1]["phase"] == "managed_publication"
        assert history[-1]["reason"] == "Reviewed publication despite low reserve."
    finally:
        database.close()


def test_cpu_work_with_disk_projection_uses_scheduler_capacity_preflight(
    tmp_path: Path,
) -> None:
    database = RunDatabase(tmp_path / "runs.db")
    scheduler = WorkstationScheduler(
        database, capacity_probe=lambda path: _capacity(path, free_gib=20)
    )
    try:
        work = scheduler.enqueue(
            kind="dataset_registration",
            launch_spec={"handler": "own_data.register"},
            resource_class="cpu",
            resource_requirements={
                "capacity_preflight": True,
                "output_path": str(tmp_path),
                "projected_disk_bytes": 1,
            },
        )
        assert scheduler.claim(work_item_id=work.id) is None
        blocked = database.get_work_item(work.id)
        assert blocked is not None and blocked.status == "blocked"
        assert "capacity preflight failed" in str(blocked.error)
        assert blocked.result["preflight"]["blockers"] == ["insufficient_disk"]
    finally:
        database.close()
