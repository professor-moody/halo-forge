"""Public API contracts for reproducible experiment operations."""

from __future__ import annotations


def _development_suite(service):
    return service.create_benchmark_suite(
        {
            "name": "development",
            "purpose": "development",
            "items": [{"id": "example", "input": "x", "expected": "y"}],
            "primary_metric": "accuracy",
            "direction": "maximize",
        }
    )["latest_revision"]["id"]


def test_run_group_api_materializes_seed_runs_and_shared_queue(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from halo_forge.auth.dependency import reset_store_for_tests
    from halo_forge.public_api import app as app_module
    from halo_forge.public_api.service import PublicApiService
    from halo_forge.run_db import RunDatabase

    database = RunDatabase(str(tmp_path / "runs.db"))
    service = PublicApiService(
        database=database,
        base_path=tmp_path,
        evaluation_storage_root=tmp_path / "evaluations",
    )
    revision_id = _development_suite(service)
    monkeypatch.setattr(app_module, "PublicApiService", lambda: service)
    reset_store_for_tests(None)

    with TestClient(app_module.create_app(serve_frontend=False)) as client:
        response = client.post(
            "/api/public/run-groups",
            json={
                "name": "three-seed repeat",
                "kind": "repeat",
                "trainer_mode": "sft",
                "suite_revision_id": revision_id,
                "base_config": {
                    "model": "local/test-model",
                    "data": str(tmp_path / "train.jsonl"),
                    "output_root": str(tmp_path / "outputs"),
                },
                "seeds": [7, 11, 19],
            },
        )
        assert response.status_code == 202, response.text
        group = response.json()
        assert group["primary_metric"] == "accuracy"
        assert group["direction"] == "maximize"
        assert group["n_trials"] == 1
        assert [run["seed"] for run in group["trials"][0]["runs"]] == [7, 11, 19]

        queue = client.get("/api/public/work-items").json()["items"]
        assert len(queue) == 6
        assert [item["kind"] for item in queue].count("training") == 3
        assert [item["kind"] for item in queue].count("evaluation") == 3
        assert all(item["run_group_id"] == group["id"] for item in queue)

        detail = client.get(f"/api/public/run-groups/{group['id']}")
        assert detail.status_code == 200
        assert detail.json()["suite_revision_id"] == revision_id
        assert client.get("/api/public/trainer-execution-capabilities").json()["items"]

        cancelled = client.post(f"/api/public/run-groups/{group['id']}/cancel")
        assert cancelled.status_code == 200
        # The supervised dashboard worker may already have claimed one item.
        # Claimed work reports the truthful intermediate state until the
        # worker observes the cancellation request.
        assert cancelled.json()["status"] in {"cancelling", "cancelled"}

    database.close()


def test_run_group_api_rejects_holdout_as_optimization_objective(tmp_path):
    from halo_forge.public_api.service import PublicApiService
    from halo_forge.run_db import RunDatabase

    database = RunDatabase(str(tmp_path / "runs.db"))
    service = PublicApiService(database=database, base_path=tmp_path)
    holdout_id = service.create_benchmark_suite(
        {
            "name": "confirmation",
            "purpose": "holdout",
            "items": [{"id": "example", "input": "x", "expected": "y"}],
            "primary_metric": "accuracy",
            "direction": "maximize",
        }
    )["latest_revision"]["id"]

    try:
        service.create_run_group(
            {
                "name": "invalid",
                "kind": "repeat",
                "trainer_mode": "sft",
                "suite_revision_id": holdout_id,
                "base_config": {"model": "m", "data": "train.jsonl"},
                "seeds": [1],
            }
        )
    except ValueError as exc:
        assert "development" in str(exc)
    else:  # pragma: no cover - the assertion above should always be reached
        raise AssertionError("holdout suite was accepted as a development objective")
    finally:
        database.close()
