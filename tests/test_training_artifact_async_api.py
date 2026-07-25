"""Non-blocking managed training artifact preparation API contracts."""

from __future__ import annotations

import threading
import time

import pytest


def _wait_for_dataset_job(service, job_id: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = service.get_dataset_job(job_id)
        if job and job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"dataset job {job_id} did not finish")


def test_managed_preflight_and_launch_return_reused_render_job_then_launch_ready(
    tmp_path, monkeypatch
):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from halo_forge.auth.dependency import reset_store_for_tests
    from halo_forge.data_lab import DatasetLab
    from halo_forge.public_api import app as app_module
    from halo_forge.public_api.service import PublicApiService
    from halo_forge.run_db import RunDatabase
    from ui.services.training_service import TrainingLaunchPreflight

    class FakeTrainingService:
        def __init__(self):
            self.preflights = []
            self.launches = []

        def preflight_sft_launch(self, **kwargs):
            self.preflights.append(kwargs)
            return TrainingLaunchPreflight(
                ok=True,
                errors=[],
                warnings=[],
                resolved_paths={"dataset": kwargs["dataset"]},
                suggested_fixes=[],
                quality_outlook={},
            )

        async def launch_sft(self, **kwargs):
            self.launches.append(kwargs)
            return kwargs["run_id"]

    database = RunDatabase(str(tmp_path / "runs.db"))
    dataset_lab = DatasetLab(tmp_path / "datasets")
    training = FakeTrainingService()
    service = PublicApiService(
        database=database,
        dataset_lab=dataset_lab,
        training_service=training,
        base_path=tmp_path,
    )
    service.get_run_detail = lambda run_id, **_: {"id": run_id, "status": "running"}

    source = tmp_path / "managed.jsonl"
    source.write_text(
        '{"prompt":"one","response":"1"}\n'
        '{"prompt":"two","response":"2"}\n'
        '{"prompt":"three","response":"3"}\n',
        encoding="utf-8",
    )
    dataset = service.create_dataset(
        {
            "name": "managed",
            "canonical_schema": "sft",
            "source": {"kind": "local", "uri": str(source)},
        }
    )
    build = service.build_dataset(
        dataset["id"],
        {
            "recipe": {
                "seed": 7,
                "steps": [{"kind": "split", "ratios": {"train": 1.0}}],
            }
        },
    )
    built = _wait_for_dataset_job(service, build["id"])
    assert built["status"] == "completed", built
    version_id = built["version_id"]

    renderer_entered = threading.Event()
    release_renderer = threading.Event()
    original_render = dataset_lab.render_training_artifact

    def blocking_render(bindings, **options):
        renderer_entered.set()
        assert release_renderer.wait(5), "test did not release artifact renderer"
        return original_render(bindings, **options)

    monkeypatch.setattr(dataset_lab, "render_training_artifact", blocking_render)
    monkeypatch.setattr(app_module, "PublicApiService", lambda: service)
    reset_store_for_tests(None)
    common = {
        "mode": "sft",
        "model": "local/test-model",
        "dataset_version_id": version_id,
        "dataset_split": "train",
        "validation_fraction": 0.34,
        "output_dir": str(tmp_path / "runs"),
    }

    try:
        with TestClient(app_module.create_app(serve_frontend=False)) as client:
            started_at = time.monotonic()
            preflight = client.post("/api/public/train/preflight", json=common)
            preflight_elapsed = time.monotonic() - started_at
            assert preflight.status_code == 202, preflight.text
            assert preflight_elapsed < 0.5
            pending = preflight.json()
            assert pending["status"] == "preparing_dataset"
            assert pending["ready"] is False and pending["accepted"] is True
            assert pending["resolved_paths"] == {}
            assert pending["errors"] == pending["warnings"] == pending["suggested_fixes"] == []
            assert pending["artifact_preparation"]["job_id"] == pending["job_id"]
            assert pending["artifact_preparation"]["job_url"].endswith(pending["job_id"])
            assert renderer_entered.wait(1)

            started_at = time.monotonic()
            launch = client.post(
                "/api/public/train/launch",
                json={**common, "output_root": str(tmp_path / "managed-runs")},
            )
            launch_elapsed = time.monotonic() - started_at
            assert launch.status_code == 202, launch.text
            assert launch_elapsed < 0.5
            launch_pending = launch.json()
            assert launch_pending["job_id"] == pending["job_id"]
            assert launch_pending["run_id"]
            assert training.preflights == []
            assert training.launches == []

            repeated = client.post("/api/public/train/preflight", json=common)
            assert repeated.status_code == 202
            assert repeated.json()["job_id"] == pending["job_id"]

            release_renderer.set()
            deadline = time.monotonic() + 5
            completed = None
            while time.monotonic() < deadline:
                completed_response = client.get(f"/api/public/dataset-jobs/{pending['job_id']}")
                assert completed_response.status_code == 200
                completed = completed_response.json()
                if completed["status"] in {"completed", "failed", "cancelled"}:
                    break
                time.sleep(0.01)
            assert completed and completed["status"] == "completed", completed
            assert completed["training_artifact_id"]

            ready = client.post(
                "/api/public/train/launch",
                json={
                    **common,
                    "output_root": str(tmp_path / "managed-runs"),
                    "run_id": launch_pending["run_id"],
                },
            )
            assert ready.status_code == 200, ready.text
            assert ready.json()["id"] == launch_pending["run_id"]
            assert ready.json()["managed"] is True
            assert ready.json()["work_item_id"]
            assert training.launches == []
            queued = database.get_work_item(ready.json()["work_item_id"])
            assert queued is not None and queued.status == "queued"
            resolved = queued.launch_spec["resolved_launch_config"]
            assert (
                resolved["training_artifact_metadata"]["artifact_id"]
                == completed["training_artifact_id"]
            )
            assert resolved["dataset"].endswith("splits/train.jsonl")
            assert resolved["validation_file"].endswith("splits/validation.jsonl")
    finally:
        release_renderer.set()
        dataset_lab.close()
        database.close()
