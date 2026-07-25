from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from halo_forge.managed_runtime import ManagedRuntimeService
from halo_forge.managed_runtime.models import AcceleratorAvailability
from halo_forge.replay import MANIFEST_VERSION, capture_manifest
from halo_forge.run_db import RunDatabase
from halo_forge.run_db.schema import SCHEMA_VERSION
from halo_forge.training_path_certification import TrainingPathCertificationService


def _idle(family: str = "rocm") -> AcceleratorAvailability:
    return AcceleratorAvailability(
        family,
        "idle",
        "2026-07-17T00:00:00+00:00",
        0.0,
    )


def _qualified_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RunDatabase, ManagedRuntimeService, str]:
    monkeypatch.setattr(
        "halo_forge.managed_runtime.service.wait_for_stable_idle",
        lambda family, probe: (True, (probe(family), probe(family), probe(family))),
    )
    monkeypatch.setattr(
        "halo_forge.managed_runtime.adapters.PodmanRocmRuntimeAdapter.available",
        lambda self: (True, None),
    )
    database = RunDatabase(str(tmp_path / "runs.db"))
    runtime = ManagedRuntimeService(
        database,
        root=tmp_path / "runtimes",
        source_root=tmp_path / "source",
        runner=lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, json.dumps({"passed": True}), ""
        ),
        occupancy_probe=_idle,
    )
    revision = runtime.list_revisions("strix-halo-rocm-7.2.1")[0]
    now = "2026-07-17T00:00:00+00:00"
    database._conn.execute(
        """INSERT INTO runtime_preparations
           (id,runtime_revision_id,status,stage,engine,image_id,image_digest,
            storage_path,progress_json,created_at,completed_at)
           VALUES (?,?, 'completed','completed','podman','image',?,?, '{}',?,?)""",
        ("prep", revision.id, revision.base_image_digest, str(tmp_path), now, now),
    )
    database._conn.commit()
    qualification = runtime.qualify(revision.id, enqueue=False)
    assert runtime.run_qualification(qualification.id).status == "local_verified"
    return database, runtime, revision.id


def _step(step: str, context: dict) -> dict:
    hashes = {
        "before": "1" * 64,
        "after": "2" * 64,
    }
    return {
        "fixture_dataset": {
            "passed": True,
            "dataset_version_id": "version-fixture",
            "dataset_content_hash": "a" * 64,
        },
        "trainer_artifact": {
            "passed": True,
            "format_version": 3,
            "artifact_hash": "b" * 64,
        },
        "model_preparation": {
            "passed": True,
            "resolved_model_commit": "0123456789abcdef",
            "tokenizer_processor_hash": "c" * 64,
        },
        "capacity_step": {
            "passed": True,
            "optimizer_step_executed": True,
            "scratch_cleaned": True,
        },
        "optimizer_update": {
            "passed": True,
            "real_trainer_entrypoint": True,
            "weights_updated": True,
        },
        "parameter_delta": {"passed": True, "changed": True, **hashes},
        "artifact_files": {"passed": True, "verified": True, "missing": []},
        "artifact_reload": {"passed": True, "reloaded": True, "finite_output": True},
        "replay_lineage": {"passed": True, "replay_version": 14, "lineage_complete": True},
        "scratch_cleanup": {"passed": True, "scratch_cleaned": True},
    }[step]


def test_schema_v23_and_replay_v14_are_additive(tmp_path: Path) -> None:
    database = RunDatabase(str(tmp_path / "runs.db"))
    assert SCHEMA_VERSION == 23
    assert MANIFEST_VERSION == 14
    for table in (
        "training_path_profiles",
        "training_path_profile_revisions",
        "training_path_certifications",
        "training_path_certification_steps",
        "training_path_certification_attempts",
        "training_path_evidence_bindings",
        "workstation_certifications",
    ):
        assert database._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
    manifest = capture_manifest(
        run_id="v21",
        modality="sft",
        model_name="Qwen/Qwen2.5-Coder-0.5B",
        seed=42,
        config={},
        training_path_binding={"training_path_revision_id": "path-1"},
    )
    assert manifest.training_path["training_path_revision_id"] == "path-1"


def test_generic_tensor_diagnostics_never_unlock_a_guided_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, runtime, runtime_revision_id = _qualified_runtime(tmp_path, monkeypatch)
    core = next(value for value in runtime.capabilities() if value.accelerator_family == "rocm")
    assert core.available is True
    assert core.supported_trainers == ()
    service = TrainingPathCertificationService(
        database,
        root=tmp_path / "certifications",
        runtime_service=runtime,
        source_root=tmp_path / "source",
        step_executor=_step,
        occupancy_probe=_idle,
    )
    matrix = service.capabilities("rocm")
    assert matrix.runtime_ready is True
    assert all(value.state != "path_verified" for value in matrix.paths)
    assert matrix.recommended_path_revision_id
    unavailable = [value for value in matrix.paths if value.trainer_mode != "sft"]
    assert unavailable
    assert all(value.state == "unavailable" for value in unavailable)
    assert all(value.recovery_action and not value.recovery_action.enabled for value in unavailable)


def test_real_evidence_progressively_verifies_sft_and_detects_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, runtime, runtime_revision_id = _qualified_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "halo_forge.training_path_certification.service.wait_for_stable_idle",
        lambda family, probe: (True, (probe(family), probe(family), probe(family))),
    )
    source = tmp_path / "source"
    (source / "halo_forge").mkdir(parents=True, exist_ok=True)
    code = source / "halo_forge" / "trainer.py"
    code.write_text("VERSION = 1\n", encoding="utf-8")
    service = TrainingPathCertificationService(
        database,
        root=tmp_path / "certifications",
        runtime_service=runtime,
        source_root=source,
        step_executor=_step,
        occupancy_probe=_idle,
    )
    matrix = service.capabilities("rocm")
    certification = service.certify(
        matrix.recommended_path_revision_id, runtime_revision_id, enqueue=False
    )
    completed = service.run_certification(certification.id)
    assert completed.status == "verified"
    assert len(completed.steps) == 10
    assert all(step.status == "passed" for step in completed.steps)
    assert service.verify(completed.id)["valid"] is True
    verified = service.capabilities("rocm")
    instruction = next(
        value
        for value in verified.paths
        if value.path_revision_id == verified.recommended_path_revision_id
    )
    assert instruction.state == "path_verified"

    code.write_text("VERSION = 2\n", encoding="utf-8")
    drift = service.verify(completed.id)
    assert drift["valid"] is False
    assert drift["stale"] is True


def test_parameter_hashes_must_be_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, runtime, runtime_revision_id = _qualified_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "halo_forge.training_path_certification.service.wait_for_stable_idle",
        lambda family, probe: (True, (probe(family), probe(family), probe(family))),
    )

    def bad_step(step: str, context: dict) -> dict:
        result = _step(step, context)
        if step == "parameter_delta":
            result = {"passed": True, "changed": True, "before": "1" * 64, "after": "1" * 64}
        return result

    service = TrainingPathCertificationService(
        database,
        root=tmp_path / "certifications",
        runtime_service=runtime,
        source_root=tmp_path / "source",
        step_executor=bad_step,
        occupancy_probe=_idle,
    )
    path = service.capabilities("rocm").recommended_path_revision_id
    value = service.certify(path, runtime_revision_id, enqueue=False)
    assert service.run_certification(value.id).status == "failed"


def test_workstation_beta_refuses_unbound_boolean_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, runtime, runtime_revision_id = _qualified_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "halo_forge.training_path_certification.service.wait_for_stable_idle",
        lambda family, probe: (True, (probe(family), probe(family), probe(family))),
    )
    service = TrainingPathCertificationService(
        database,
        root=tmp_path / "certifications",
        runtime_service=runtime,
        source_root=tmp_path / "source",
        step_executor=_step,
        occupancy_probe=_idle,
    )
    path = service.capabilities("rocm").recommended_path_revision_id
    certification = service.certify(path, runtime_revision_id, enqueue=False)
    assert service.run_certification(certification.id).status == "verified"

    value = service.workstation_certify(
        runtime_revision_id,
        evidence={
            "managed_capacity_check": True,
            "own_data_proof": True,
            "parameter_hash_delta": True,
            "artifact_reload": True,
            "outcome_assessment": True,
            "scheduler_restart_recovery": True,
            "external_workload_waiting": True,
            "twelve_hour_soak": True,
            "sequential_proof_runs": 1,
        },
        enqueue=False,
    )
    assert value.status == "incomplete"
    assert value.evidence["requirements"]["managed_capacity_check"] is False
    assert value.evidence["requirements"]["own_data_proof"] is False
    manifest = Path(value.report_path)
    assert manifest.name == "manifest.json"
    assert (manifest.parent / "qualification.md").is_file()


def test_real_sft_executor_uses_the_shipped_step_count_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from halo_forge.training_path_certification import certify as executor

    monkeypatch.setattr(
        executor,
        "_run_sft",
        lambda context, output, max_samples: {
            "summary": {
                "weights_updated": True,
                "total_train_steps_executed": 1,
                "parameter_evidence": {
                    "algorithm": "sha256-trainable-tensors-v1",
                    "before": "1" * 64,
                    "after": "2" * 64,
                    "changed": True,
                },
            },
            "summary_path": str(tmp_path / "training_summary.json"),
        },
    )
    value = executor.optimizer_update({"attempt_dir": str(tmp_path)})
    assert value["passed"] is True
    assert value["total_train_steps"] == 1


def test_failed_certification_resumes_from_last_checksummed_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database, runtime, runtime_revision_id = _qualified_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "halo_forge.training_path_certification.service.wait_for_stable_idle",
        lambda family, probe: (True, (probe(family), probe(family), probe(family))),
    )
    calls: list[str] = []
    fail_reload = True

    def resumable(step: str, context: dict) -> dict:
        nonlocal fail_reload
        calls.append(step)
        if step == "artifact_reload" and fail_reload:
            fail_reload = False
            raise RuntimeError("simulated reload interruption")
        return _step(step, context)

    service = TrainingPathCertificationService(
        database,
        root=tmp_path / "certifications",
        runtime_service=runtime,
        source_root=tmp_path / "source",
        step_executor=resumable,
        occupancy_probe=_idle,
    )
    path = service.capabilities("rocm").recommended_path_revision_id
    value = service.certify(path, runtime_revision_id, enqueue=False)
    assert service.run_certification(value.id).status == "failed"
    assert service.get_certification(value.id).resume_cursor["last_complete_step"] == 7
    service.retry(value.id, reason="simulated issue repaired")
    completed = service.run_certification(value.id)
    assert completed.status == "verified"
    assert calls.count("fixture_dataset") == 1
    assert calls.count("artifact_reload") == 2


def test_certification_fixture_really_builds_and_renders_v3_deterministically(
    tmp_path: Path,
) -> None:
    from halo_forge.training_path_certification.certify import (
        fixture_dataset,
        trainer_artifact,
    )
    from halo_forge.training_path_certification.registry import (
        PATH_DEFINITIONS,
        normalized_definition,
    )

    identities = []
    for ordinal in range(2):
        attempt = tmp_path / f"attempt-{ordinal}"
        attempt.mkdir()
        revision = normalized_definition(dict(PATH_DEFINITIONS[0]), "rocm")
        revision["id"] = "certification-smoke-path"
        context = {"attempt_dir": str(attempt), "path_revision": revision, "evidence": {}}
        version = fixture_dataset(context)
        context["evidence"]["fixture_dataset"] = version
        artifact = trainer_artifact(context)
        assert version["split_counts"] == {"train": 9, "validation": 3}
        assert artifact["format_version"] == 3
        assert artifact["row_counts"]["test"] == 0
        assert artifact["row_counts"]["canary"] == 0
        identities.append((version["dataset_content_hash"], artifact["artifact_hash"]))
    assert identities[0] == identities[1]


def test_v21_api_contract_is_exposed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    from halo_forge.public_api.app import create_app

    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("HALOFORGE_RUNTIME_ROOT", str(tmp_path / "runtimes"))
    client = TestClient(create_app(serve_frontend=False))
    paths = set(client.get("/api/public/openapi.json").json()["paths"])
    assert "/api/public/runtime/paths" in paths
    assert "/api/public/training-path-revisions/{revision_id}/certify" in paths
    assert "/api/public/training-path-revisions/{revision_id}/certification-preview" in paths
    assert "/api/public/training-path-certifications/{certification_id}/verify" in paths
    assert "/api/public/training-path-certifications/{certification_id}/resume" in paths
    assert "/api/public/training-path-certifications/{certification_id}/evidence" in paths
    assert "/api/public/release/workstation-certify" in paths
    assert "/api/public/release/workstation-certifications/{certification_id}/verify" in paths
