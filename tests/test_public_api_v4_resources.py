from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from halo_forge.public_api.service import PublicApiService
from halo_forge.run_db import LabV4Catalog, RunDatabase


def _raw_model(path: Path, value: str = "weights") -> Path:
    path.mkdir(parents=True)
    (path / "weights.bin").write_text(value, encoding="utf-8")
    return path


def _service(tmp_path: Path, **kwargs) -> tuple[PublicApiService, RunDatabase]:
    database = RunDatabase(str(tmp_path / "runs.db"))
    service = PublicApiService(
        database=database,
        base_path=tmp_path,
        artifact_storage_root=tmp_path / "artifacts",
        evaluation_storage_root=tmp_path / "evaluations",
        **kwargs,
    )
    return service, database


def _suite_revision(service: PublicApiService, purpose: str, metric: str) -> str:
    return service.create_benchmark_suite(
        {
            "name": f"{purpose}-{metric}",
            "purpose": purpose,
            "items": [{"id": "item", "input": "hello", "expected": "world"}],
            "primary_metric": metric,
            "direction": "minimize" if "latency" in metric else "maximize",
        }
    )["latest_revision"]["id"]


def test_artifact_api_routes_are_flattened_persistent_and_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from halo_forge.auth.dependency import reset_store_for_tests
    from halo_forge.public_api import app as app_module

    service, database = _service(tmp_path)
    monkeypatch.setattr(app_module, "PublicApiService", lambda: service)
    reset_store_for_tests(None)
    source = _raw_model(tmp_path / "run-model")

    try:
        with TestClient(app_module.create_app(serve_frontend=False)) as client:
            imported = client.post(
                "/api/public/model-artifacts/import",
                json={
                    "path": str(source),
                    "kind": "final",
                    "format": "raw",
                    "adopt": True,
                    "notes": "candidate from run",
                },
            )
            assert imported.status_code == 201, imported.text
            artifact = imported.json()["artifact"]
            artifact_id = artifact["id"]
            assert artifact["content_hash"]
            assert artifact["blob"]["integrity"] == "hash_verified"
            assert artifact["locations"][0]["available"] is True

            page = client.get(
                "/api/public/model-artifacts",
                params={"query": "candidate", "limit": 1, "offset": 0},
            ).json()
            assert page["total"] == 1
            assert page["items"][0]["id"] == artifact_id

            assert (
                client.post(f"/api/public/model-artifacts/{artifact_id}/pin").json()["pinned"]
                is True
            )
            tagged = client.post(
                f"/api/public/model-artifacts/{artifact_id}/tags",
                json={"tags": ["reviewed", "local"]},
            ).json()
            assert tagged["tags"] == ["local", "reviewed"]
            verified = client.post(
                f"/api/public/model-artifacts/{artifact_id}/verify", json={}
            ).json()
            assert verified["verification"]["passed"] is True
            lineage = client.get(f"/api/public/model-artifacts/{artifact_id}/lineage").json()
            assert lineage["artifact"]["id"] == artifact_id
            assert lineage["edges"] == []

            export = client.post(
                "/api/public/artifact-operations",
                json={
                    "kind": "export",
                    "input_artifact_ids": [artifact_id],
                    "config": {"bundle_name": "portable-candidate"},
                },
            )
            assert export.status_code == 202, export.text
            operation = export.json()
            assert operation["kind"] == "export"
            assert operation["status"] == "queued"
            assert operation["work_item_id"]
            assert operation["input_artifact_ids"] == [artifact_id]

            plan = client.post(
                "/api/public/storage/cleanup",
                json={"preview": True},
            ).json()
            assert plan["status"] == "preview"
            # Pinned artifacts are protected and therefore absent from candidates.
            assert plan["items"] == []
            client.delete(f"/api/public/model-artifacts/{artifact_id}/pin")
            plan = client.post(
                "/api/public/storage/cleanup",
                json={"preview": True, "older_than_days": 0},
            ).json()
            queued = client.post(
                "/api/public/storage/cleanup",
                json={
                    "plan_id": plan["id"],
                    "approved": True,
                    "review_note": "Reviewed cleanup candidates",
                },
            ).json()
            assert queued["status"] == "queued"
            assert queued["work_item_id"]

            session = client.post(
                "/api/public/playground/sessions",
                json={"name": "Candidate review", "artifact_id": artifact_id, "seed": 7},
            ).json()
            updated = client.post(
                f"/api/public/playground/sessions/{session['id']}/messages",
                json={"role": "user", "content": "Explain this result."},
            ).json()
            assert updated["seed"] == 7
            assert updated["messages"][0]["content"] == "Explain this result."
    finally:
        database.close()


def test_qualification_profile_normalizes_public_rules_and_comparison(tmp_path: Path) -> None:
    service, database = _service(tmp_path)
    try:
        development = _suite_revision(service, "development", "accuracy")
        operational = _suite_revision(service, "operational", "total_latency_ms")
        profile = service.create_qualification_profile(
            {
                "name": "Local gates",
                "development_suite_revision_id": development,
                "operational_suite_revision_id": operational,
                "target_backend": "mlx",
                "thresholds": [
                    {
                        "stage": "development",
                        "name": "accuracy",
                        "direction": "maximize",
                        "threshold": 0.8,
                        "allowed_delta": 0.02,
                    },
                    {
                        "stage": "operational",
                        "name": "total_latency_ms",
                        "direction": "minimize",
                        "threshold": 100.0,
                    },
                ],
            }
        )
        assert profile["development_suite_revision_id"] == development
        assert profile["revision"] == 1
        assert {item["name"] for item in profile["metrics"]} == {
            "accuracy",
            "total_latency_ms",
        }
        assert service.list_qualification_profiles()["items"][0]["name"] == "Local gates"

        reused = service.create_qualification_profile(
            {
                "name": "Local gates",
                "development_suite_revision_id": development,
                "operational_suite_revision_id": operational,
                "target_backend": "mlx",
                "metrics": [
                    {
                        "stage": "development",
                        "name": "accuracy",
                        "direction": "maximize",
                        "threshold": 0.8,
                        "allowed_delta": 0.02,
                    },
                    {
                        "stage": "operational",
                        "name": "total_latency_ms",
                        "direction": "minimize",
                        "threshold": 100.0,
                    },
                ],
            }
        )
        assert reused["id"] == profile["id"]
        assert reused["reused"] is True

        with pytest.raises(ValueError, match="development and one operational"):
            service.create_qualification_profile(
                {
                    "name": "Missing operational rule",
                    "development_suite_revision_id": development,
                    "operational_suite_revision_id": operational,
                    "thresholds": [
                        {
                            "stage": "development",
                            "metric": "accuracy",
                            "pass_threshold": 0.8,
                        }
                    ],
                }
            )

        first = service.import_model_artifact(
            {"path": str(_raw_model(tmp_path / "base", "base")), "format": "raw"}
        )["artifact"]["id"]
        second = service.import_model_artifact(
            {"path": str(_raw_model(tmp_path / "candidate", "candidate")), "format": "raw"}
        )["artifact"]["id"]
        catalog = LabV4Catalog(database)
        base = catalog.create_qualification(profile_revision_id=profile["id"], occurrence_id=first)
        candidate = catalog.create_qualification(
            profile_revision_id=profile["id"], occurrence_id=second
        )
        catalog.update_qualification(
            base.id,
            status="completed",
            decision="pass",
            metrics={
                "development": {"accuracy": 0.80},
                "operational": {"total_latency_ms": 100.0},
            },
        )
        catalog.update_qualification(
            candidate.id,
            status="completed",
            decision="pass",
            metrics={
                "development": {"accuracy": 0.82},
                "operational": {"total_latency_ms": 90.0},
            },
        )
        comparison = service.compare_qualifications(base.id, candidate.id)
        deltas = {item["metric"]: item for item in comparison["deltas"]}
        assert deltas["accuracy"]["favorable_delta"] == pytest.approx(0.02)
        assert deltas["total_latency_ms"]["favorable_delta"] == pytest.approx(10.0)
        listed = service.list_qualifications(status="pass")
        assert listed["total"] == 2
        assert listed["items"][0]["artifact_id"] in {first, second}
    finally:
        database.close()


def test_model_artifact_pagination_is_not_capped_at_one_thousand(tmp_path: Path) -> None:
    service, database = _service(tmp_path)
    try:
        catalog = LabV4Catalog(database)
        blob = catalog.upsert_blob(
            content_hash="a" * 64,
            artifact_type="final",
            format="raw",
            size_bytes=1,
            integrity_state="verified",
        )
        for index in range(1005):
            catalog.create_occurrence(
                occurrence_id=f"artifact-{index:04d}",
                blob_id=blob.id,
                artifact_kind="final_model",
                model_id=f"model-{index:04d}",
                backend="local",
            )
        page = service.list_model_artifacts(limit=10, offset=1000)
        assert page["total"] == 1005
        assert len(page["items"]) == 5
        assert page["has_more"] is False
    finally:
        database.close()


class _IdleServeManager:
    def status(self):
        return {"running": False, "state": "idle", "pid": None, "model": None}

    def stop(self):
        return self.status()


class _ManagedServeManager:
    def __init__(self) -> None:
        self.running = False
        self.model = None

    def status(self):
        return {
            "running": self.running,
            "state": "running" if self.running else "idle",
            "pid": os.getpid() if self.running else None,
            "model": self.model,
            "backend": "local",
            "url": "http://127.0.0.1:8001/v1",
        }

    def start(self, request):
        if self.running:
            raise ValueError("already being served")
        self.running = True
        self.model = request.model
        return self.status()

    def stop(self):
        self.running = False
        return self.status()


def test_second_serve_start_preserves_the_live_process_lease(tmp_path: Path) -> None:
    manager = _ManagedServeManager()
    service, database = _service(tmp_path, serve_manager=manager)
    try:
        artifact_id = service.import_model_artifact(
            {
                "path": str(_raw_model(tmp_path / "served-model")),
                "format": "raw",
            }
        )["artifact"]["id"]
        started = service.serve_start({"artifact_id": artifact_id})
        assert started["running"] is True
        lease = database.get_resource_lease("accelerator")
        assert lease is not None
        with pytest.raises(ValueError, match="already being served"):
            service.serve_start({"model": "another-model"})
        retained = database.get_resource_lease("accelerator")
        assert retained is not None
        assert retained.lease_token == lease.lease_token
        service.serve_stop()
        assert database.get_resource_lease("accelerator") is None
    finally:
        database.close()


def test_serve_stop_terminates_adopted_process_before_releasing_lease(tmp_path: Path) -> None:
    from halo_forge.workstation_jobs import process_start_time

    manager = _IdleServeManager()
    service, database = _service(tmp_path, serve_manager=manager)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        started_at = process_start_time(child.pid)
        assert started_at is not None
        lease = database.acquire_serving_lease(
            holder_id=service._serving_lease_holder,
            holder_pid=child.pid,
            holder_pid_started_at=started_at,
            metadata={"state": "serving"},
        )
        assert lease is not None
        stopped = service.serve_stop()
        child.wait(timeout=3)
        assert stopped["stopped"] is True
        assert database.get_resource_lease("accelerator") is None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)
        database.close()
