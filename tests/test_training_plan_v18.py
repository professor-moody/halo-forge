from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from halo_forge.replay import MANIFEST_VERSION, capture_manifest
from halo_forge.run_db import RunDatabase
from halo_forge.run_db.schema import SCHEMA_VERSION
from halo_forge.training_plan import TrainingPlanError, TrainingPlanService
from halo_forge.workstation_jobs.resources import (
    AcceleratorCapacity,
    DiskCapacity,
    MemoryCapacity,
    ProcessCapacity,
    WorkstationCapacity,
)


def _capacity(_: Path) -> WorkstationCapacity:
    gib = 1024**3
    return WorkstationCapacity(
        sampled_at=datetime.now(timezone.utc),
        disk=DiskCapacity(str(_), 500 * gib, 100 * gib, 400 * gib),
        memory=MemoryCapacity(64 * gib, 8 * gib, 56 * gib, source="test"),
        process=ProcessCapacity(123, rss_bytes=256 * 1024**2),
        accelerator=AcceleratorCapacity(
            "cpu", device_name="test-cpu", device_memory_total_bytes=None
        ),
    )


def _version(database: RunDatabase, root: Path):
    dataset = database.create_dataset(
        name="Guided SFT", modality="text", canonical_schema="sft"
    )
    source = database.create_dataset_source(
        dataset_id=dataset.id,
        source_id="source-v18",
        kind="local",
        uri=str(root / "source.jsonl"),
        fingerprint="source-fingerprint-v18",
        row_count=120,
    )
    version_root = root / "version"
    version_root.mkdir()
    (version_root / "train.jsonl").write_text("{}\n", encoding="utf-8")
    return database.create_dataset_version(
        dataset_id=dataset.id,
        version_id="version-v18",
        source_id=source.id,
        recipe_hash="recipe-v18",
        recipe={"schema": "sft", "steps": []},
        storage_path=str(version_root),
        status="completed",
        content_hash="content-v18",
        row_count=120,
        split_counts={"train": 100, "validation": 10, "test": 10},
        statistics={"tokens": {"p50": 72, "p99": 240}},
        source_fingerprints={source.id: source.fingerprint},
    )


def _prepared_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe):
    database = RunDatabase(str(tmp_path / "catalog.sqlite"))
    version = _version(database, tmp_path)
    monkeypatch.setattr("halo_forge.training_plan.service._backend_name", lambda: "cpu")
    service = TrainingPlanService(
        database,
        root=tmp_path / "managed",
        capacity_sampler=_capacity,
        probe_runner=probe,
    )
    recommendation = service.recommend(
        {"dataset_version_id": version.id, "trainer_mode": "sft"}
    )
    model_root = tmp_path / "model-cache"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_root / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **_: str(model_root)),
    )
    service.record_decision(
        recommendation.revision.id,
        "confirmed",
        details={"download_confirmed": True, "confirmation_surface": "test"},
    )
    preparation = service.prepare_model(
        recommendation.revision.id, enqueue=False, allow_download=True
    )
    return database, version, service, recommendation, preparation


def test_schema_v21_and_replay_v12_are_additive(tmp_path: Path) -> None:
    database = RunDatabase(str(tmp_path / "catalog.sqlite"))
    assert SCHEMA_VERSION == 23
    tables = {
        row["name"]
        for row in database._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "training_plans",
        "training_plan_revisions",
        "model_preparations",
        "training_capacity_checks",
        "training_capacity_attempts",
        "training_plan_decisions",
        "run_training_plans",
    } <= tables
    manifest = capture_manifest(
        run_id="proof-v18",
        modality="sft",
        model_name="local-model",
        seed=42,
        config={},
        training_plan_binding={
            "revision_id": "planrev-v18",
            "capacity_check_id": "capacity-v18",
            "confirmation": "confirmed",
        },
    )
    assert MANIFEST_VERSION == manifest.manifest_version == 14
    assert manifest.training_plan["revision_id"] == "planrev-v18"


def test_schema_v20_to_v21_upgrade_preserves_existing_catalog(tmp_path: Path) -> None:
    path = tmp_path / "legacy-v20.sqlite"
    legacy = RunDatabase(str(path))
    dataset = legacy.create_dataset(
        name="Preserved v20 dataset", modality="text", canonical_schema="sft"
    )
    v21_tables = (
        "run_training_plans",
        "training_plan_decisions",
        "training_capacity_attempts",
        "training_capacity_checks",
        "model_preparations",
        "training_plan_revisions",
        "training_plans",
    )
    for table in v21_tables:
        legacy._conn.execute(f"DROP TABLE IF EXISTS {table}")
    legacy._conn.execute(
        "UPDATE schema_meta SET value='20' WHERE key='schema_version'"
    )
    legacy._conn.commit()
    legacy.close()

    upgraded = RunDatabase(str(path))
    assert upgraded.get_dataset(dataset.id).name == "Preserved v20 dataset"
    assert upgraded._conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()[0] == "23"
    tables = {
        row["name"]
        for row in upgraded._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert set(v21_tables) <= tables


def test_v18_public_routes_are_exposed_without_internal_ids_in_path_names() -> None:
    pytest.importorskip("fastapi")
    from halo_forge.public_api.app import create_app

    paths = create_app(serve_frontend=False).openapi()["paths"]
    assert {
        "/api/public/training-plan-capabilities",
        "/api/public/training-plans/recommend",
        "/api/public/training-plan-revisions/{revision_id}/prepare",
        "/api/public/training-plan-revisions/{revision_id}/capacity-check",
        "/api/public/training-plan-revisions/{revision_id}/readiness",
        "/api/public/training-plan-revisions/{revision_id}/proof",
        "/api/public/model-preparations/{preparation_id}/retry",
        "/api/public/training-capacity-checks/{check_id}/retry",
    } <= set(paths)


def test_dataset_readiness_reports_missing_managed_storage_instead_of_500(
    tmp_path: Path,
) -> None:
    from halo_forge.data_lab import VersionError
    from halo_forge.public_api.service import PublicApiService

    database = RunDatabase(str(tmp_path / "catalog.sqlite"))
    version = _version(database, tmp_path)

    class MissingVersionEngine:
        def verify_version(self, *_args, **_kwargs):
            raise VersionError("The managed dataset version is unavailable.")

    readiness = PublicApiService(
        database=database,
        dataset_lab=MissingVersionEngine(),
        base_path=tmp_path,
    ).get_dataset_version_readiness(version.id, trainer_mode="sft")

    assert readiness["ready"] is False
    assert readiness["status"] == "blocked"
    blocker = next(
        item
        for item in readiness["blockers"]
        if item["code"] == "version_storage_unavailable"
    )
    assert blocker["action"] == "open_dataset_version"
    assert "rebuild" in blocker["remedy"].lower()


def test_recommendation_is_deterministic_explainable_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = RunDatabase(str(tmp_path / "catalog.sqlite"))
    version = _version(database, tmp_path)
    monkeypatch.setattr("halo_forge.training_plan.service._backend_name", lambda: "cpu")
    service = TrainingPlanService(
        database, root=tmp_path / "managed", capacity_sampler=_capacity
    )
    first = service.recommend({"dataset_version_id": version.id, "trainer_mode": "sft"})
    second = service.recommend({"dataset_version_id": version.id, "trainer_mode": "sft"})
    assert first.revision.id == second.revision.id
    assert first.revision.definition["seed"] == 42
    assert first.revision.definition["validation_split"] == "validation"
    assert first.revision.definition["max_samples"] == 100
    assert len(first.revision.reasons) >= 3
    with pytest.raises(Exception, match="immutable"):
        database._conn.execute(
            "UPDATE training_plan_revisions SET model_id='changed' WHERE id=?",
            (first.revision.id,),
        )


def test_specialized_classification_recommendation_respects_dataset_modality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = RunDatabase(str(tmp_path / "catalog.sqlite"))
    dataset = database.create_dataset(
        name="Image labels", modality="image", canonical_schema="classification"
    )
    version_root = tmp_path / "image-version"
    version_root.mkdir()
    (version_root / "train.jsonl").write_text(
        '{"media":"asset.jpg","label":"ok"}\n', encoding="utf-8"
    )
    version = database.create_dataset_version(
        dataset_id=dataset.id,
        version_id="image-classification-v18",
        recipe_hash="image-recipe-v18",
        recipe={"schema": "classification", "steps": []},
        storage_path=str(version_root),
        status="completed",
        content_hash="image-content-v18",
        row_count=40,
        split_counts={"train": 32, "validation": 4, "test": 4},
        statistics={
            "labels": {"class_count": 3},
            "image": {"width_p95": 512, "height_p95": 512},
        },
    )
    monkeypatch.setattr("halo_forge.training_plan.service._backend_name", lambda: "cpu")
    service = TrainingPlanService(
        database, root=tmp_path / "managed", capacity_sampler=_capacity
    )
    recommendation = service.recommend(
        {"dataset_version_id": version.id, "trainer_mode": "classify"}
    )
    assert recommendation.revision.model_id == "google/vit-base-patch16-224-in21k"
    assert recommendation.revision.definition["expected_artifacts"]["requires_processor"] is True
    assert recommendation.revision.definition["dataset_shape"]["label_count"] == 3


def test_model_resolution_and_capacity_fallback_preserve_attempt_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    def probe(request, scratch):
        calls.append(dict(request["configuration"]))
        assert scratch.is_dir()
        if len(calls) == 1:
            raise MemoryError("confirmed microbatch did not fit")
        return {
            "scratch_step_executed": True,
            "measurement_contract": "test_exact_trainer_adapter_v1",
            "peak_process_memory_bytes": 2 * 1024**3,
            # Unified-memory runtimes may expose a tiny bookkeeping value here.
            "peak_device_memory_bytes": 475 * 1024,
        }

    database, version, service, recommendation, preparation = _prepared_service(
        tmp_path, monkeypatch, probe
    )
    assert preparation.status == "completed"
    assert preparation.resolved_commit
    assert preparation.plan_revision_id != recommendation.revision.id
    resolved = service.get_revision(preparation.plan_revision_id)
    assert resolved is not None and resolved.status == "resolved"
    assert resolved.resolved_model_commit == preparation.resolved_commit
    assert service.is_confirmed(resolved.id, require_download=True)

    check = service.create_capacity_check(resolved.id, enqueue=False)
    assert check.status == "ready_with_adjustment"
    assert check.selected_adjustment["gradient_checkpointing"] is True
    assert check.forecast.peak_memory_bytes == 2 * 1024**3
    attempts = service.list_capacity_attempts(check.id)["items"]
    assert [item["status"] for item in attempts] == ["failed", "passed"]
    assert all(item["sample_identity"]["record_content_retained"] is False for item in attempts)
    assert not (tmp_path / "managed" / "training-capacity" / "scratch" / check.id).exists()

    readiness = service.readiness(resolved.id)
    assert readiness.status == "ready"
    payload = service.resolved_launch_payload(resolved.id, {})
    assert payload["training_plan_revision_id"] == resolved.id
    assert payload["training_plan_model_id"] == resolved.model_id
    assert payload["model"] == preparation.cache_path
    assert payload["dataset_version_id"] == version.id
    assert payload["gradient_checkpointing"] is True
    round_trip = service.resolved_launch_payload(resolved.id, payload)
    assert round_trip["batch_size"] == payload["batch_size"]
    assert round_trip["gradient_checkpointing"] is True
    with pytest.raises(TrainingPlanError, match="conflicts"):
        service.resolved_launch_payload(resolved.id, {"model": "another/model"})
    with pytest.raises(TrainingPlanError, match="max_sequence_length"):
        service.resolved_launch_payload(
            resolved.id, {"max_sequence_length": 32_768}
        )
    with pytest.raises(TrainingPlanError, match="gradient_accumulation_steps"):
        service.resolved_launch_payload(
            resolved.id, {"gradient_accumulation_steps": 999}
        )

    service.bind_run(
        run_id="proof-run-v18",
        revision_id=resolved.id,
        capacity_check_id=check.id,
        role="proof",
    )
    binding = service.run_binding("proof-run-v18")
    assert binding and binding["revision"]["content_hash"] == resolved.content_hash
    assert binding["capacity_check"]["compute_shape_hash"] == resolved.compute_shape_hash

    full = service.derive_full_run_revision(resolved.id)
    assert full.id != resolved.id
    assert full.definition["proof_run"] is False
    assert "max_samples" not in full.definition
    assert full.definition["validation_split"] == "validation"
    assert full.compute_shape_hash == resolved.compute_shape_hash
    full_readiness = service.readiness(full.id)
    assert full_readiness.status == "ready"
    assert full_readiness.capacity_check.id == check.id
    # A fresh full-plan check reuses the same verified content-addressed model
    # preparation instead of requiring a duplicate download record.
    full_check = service.create_capacity_check(full.id, enqueue=False)
    assert full_check.status == "ready"
    assert full_check.model_preparation_id == preparation.id


def test_capacity_retry_appends_attempts_instead_of_replacing_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always_fail(request, scratch):
        raise MemoryError("no safe shape")

    database, _, service, _, preparation = _prepared_service(
        tmp_path, monkeypatch, always_fail
    )
    check = service.create_capacity_check(preparation.plan_revision_id, enqueue=False)
    assert check.status == "blocked"
    before = service.list_capacity_attempts(check.id)["items"]
    assert len(before) >= 1

    # A direct rerun exercises the same append-only retry boundary as the
    # durable scheduler path without needing a live worker in this unit test.
    service.probe_runner = lambda request, scratch: {
        "scratch_step_executed": True,
        "measurement_contract": "test_exact_trainer_adapter_v1",
    }
    database._conn.execute(
        "UPDATE training_capacity_checks SET status='queued',stage='queued',error=NULL WHERE id=?",
        (check.id,),
    )
    database._conn.commit()
    retried = service.run_capacity_check(check.id)
    after = service.list_capacity_attempts(check.id)["items"]
    assert retried.status == "ready"
    assert len(after) == len(before) + 1
    assert after[-1]["ordinal"] > before[-1]["ordinal"]


def test_default_capacity_runner_requires_reported_optimizer_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from halo_forge.public_api.service import PublicApiService

    database = RunDatabase(str(tmp_path / "catalog.sqlite"))
    service = TrainingPlanService(
        database, root=tmp_path / "managed", capacity_sampler=_capacity
    )
    dataset = tmp_path / "capacity.jsonl"
    dataset.write_text('{"prompt":"private input","response":"private output"}\n', encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    def command(payload):
        output = repr(str(payload["output_dir"]))
        code = (
            "import json; from pathlib import Path; "
            f"root=Path({output}); root.mkdir(parents=True); "
            "(root/'summary.json').write_text(json.dumps({'optimizer_steps':1}))"
        )
        return [sys.executable, "-c", code]

    monkeypatch.setattr(
        PublicApiService,
        "_managed_training_command",
        staticmethod(command),
    )
    result = service._trainer_probe(
        {
            "definition": {"trainer_mode": "sft", "model": "unused"},
            "configuration": {"batch_size": 1},
            "model_cache_path": str(tmp_path),
            "dataset_path": str(dataset),
            "sample_count": 1,
            "sample_identity": {"identity_hash": "identity-v18"},
        },
        scratch,
    )
    assert result["scratch_step_executed"] is True
    assert result["optimizer_steps"] == 1
    assert result["source_content_retained"] is False
    assert "private input" not in str(result)


def test_optimizer_evidence_handles_nested_empty_objects_and_canonical_total() -> None:
    summary = {
        "run_id": "real-sft-shape",
        "cycles": [
            {
                "optimizer_steps": 1,
                "diagnostics": {},
                "metrics": {"validation": {}},
            }
        ],
        "total_train_steps_executed": 1,
        "optional": {},
    }
    assert TrainingPlanService._optimizer_steps_in(summary) == 1


def test_public_plan_projection_separates_execution_from_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from halo_forge.public_api.service import PublicApiService

    def probe(_request, _scratch):
        return {
            "scratch_step_executed": True,
            "measurement_contract": "test_exact_trainer_adapter_v1",
        }

    database, _version_value, plan_service, _recommendation, preparation = (
        _prepared_service(tmp_path, monkeypatch, probe)
    )
    revision_id = preparation.plan_revision_id
    plan_service.create_capacity_check(revision_id, enqueue=False)
    public = PublicApiService(database=database, base_path=tmp_path)
    monkeypatch.setattr(public, "_training_plan_engine", lambda: plan_service)

    projected = public._resolve_public_training_plan_payload(revision_id, {})
    assert projected["model"] == preparation.cache_path
    assert projected["training_plan_model_id"] == "Qwen/Qwen2.5-Coder-0.5B"
    assert projected["max_sequence_length"] == plan_service.get_revision(
        revision_id
    ).definition["max_sequence_length"]
    assert projected["gradient_checkpointing"] is False
    assert "dataset_shape" not in projected
    assert "expected_artifacts" not in projected
    assert "precision" not in projected
    assert "adaptation" not in projected
