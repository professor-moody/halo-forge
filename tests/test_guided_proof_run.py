"""Focused contracts for the v9 guided proof/full-run handoff."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from halo_forge.public_api.service import PublicApiService
from halo_forge.run_db import RunDatabase


class _VersionVerifier:
    def verify_version(self, *_args: Any, **_kwargs: Any) -> dict[str, bool]:
        return {"valid": True}


class _QualifiedVerifierEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def resolve_binding(self, revision_id: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((revision_id, kwargs))
        if revision_id != "qualified-verifier-revision":
            raise ValueError("Guided use requires a candidate- or approved-qualified verifier")
        if kwargs.get("require_qualified") is not True:
            raise AssertionError("guided proof resolution must require qualification")
        return {
            "verifier_profile_revision_id": revision_id,
            "implementation_ref": "math_verifier",
            "qualification_alias": "candidate",
        }


def _guided_version(
    db: RunDatabase,
    root: Path,
    *,
    scenario_revision_id: str,
    schema: str,
    modality: str,
    train_rows: int = 400,
    validation_rows: int = 20,
) -> str:
    dataset = db.create_dataset(
        dataset_id=f"dataset-{schema}",
        name=f"Guided {schema}",
        modality=modality,
        canonical_schema=schema,
    )
    source = db.create_dataset_source(
        dataset_id=dataset.id,
        source_id=f"source-{schema}",
        kind="local",
        uri=str(root / "source.jsonl"),
        fingerprint=f"source-fingerprint-{schema}",
        metadata={
            "guided_own_data": {
                "format_version": 1,
                "scenario_revision_id": scenario_revision_id,
                "field_mapping": {
                    "prompt": {"kind": "direct", "source": "question"},
                    "response": {"kind": "direct", "source": "answer"},
                },
            }
        },
    )
    version_root = root / f"version-{schema}"
    (version_root / "splits").mkdir(parents=True)
    for split in ("train", "validation"):
        (version_root / "splits" / f"{split}.jsonl").write_text(
            '{"prompt":"p","response":"r"}\n', encoding="utf-8"
        )
    recipe = {
        "seed": 42,
        "steps": [
            {
                "kind": "map",
                "mapping_version": 2,
                "scenario_revision_id": scenario_revision_id,
                "fields": {
                    "prompt": {"kind": "direct", "source": "question"},
                    "response": {"kind": "direct", "source": "answer"},
                },
            },
            {
                "kind": "split",
                "seed": 42,
                "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            },
        ],
    }
    version = db.create_dataset_version(
        dataset_id=dataset.id,
        version_id=f"version-{schema}",
        source_id=source.id,
        recipe_hash=f"recipe-{schema}",
        recipe=recipe,
        storage_path=str(version_root),
        status="completed",
        content_hash=f"content-{schema}",
        row_count=train_rows + validation_rows,
        split_counts={"train": train_rows, "validation": validation_rows, "test": 10},
        source_fingerprints={source.id: source.fingerprint},
    )
    return version.id


def _service(db: RunDatabase, root: Path) -> PublicApiService:
    service = PublicApiService(
        database=db,
        dataset_lab=_VersionVerifier(),
        dataset_storage_root=root / "datasets",
        base_path=root,
    )
    # The scenario registry remains authoritative in these unit tests; runtime
    # dependency qualification is covered separately by capability tests.
    service._guided_data_engine = lambda: SimpleNamespace(  # type: ignore[method-assign]
        _runtime_scenario_status=lambda _scenario, _backend: (True, None),
        runtime_trainer_compatibility=lambda scenario, _backend, trainer_mode=None: [
            {
                "trainer_mode": mode,
                "compatible": True,
                "reason": None,
            }
            for mode in (
                [trainer_mode]
                if trainer_mode
                else list(scenario.trainer_modes)
            )
        ],
    )
    return service


def test_guided_sft_proof_caps_budget_and_preserves_validation(tmp_path: Path) -> None:
    db = RunDatabase(":memory:")
    version_id = _guided_version(
        db,
        tmp_path,
        scenario_revision_id="instruction-sft@1",
        schema="sft",
        modality="text",
    )
    service = _service(db, tmp_path)
    captured: list[dict[str, Any]] = []

    async def fake_launch(payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(payload)
        return {"id": "proof-run", "run_id": "proof-run", "status": "pending"}

    service.launch_training = fake_launch  # type: ignore[method-assign]
    result = asyncio.run(
        service.launch_dataset_proof_run(
            version_id,
            {
                "scenario_revision_id": "instruction-sft@1",
                "trainer_mode": "sft",
                "model": "local/tiny-model",
                "max_samples": 9999,
                "epochs": 8,
                "output_root": str(tmp_path / "runs"),
            },
        )
    )

    launch = captured[0]
    assert launch["proof_run"] is True
    assert launch["max_samples"] == launch["proof_max_samples"] == 200
    assert launch["epochs"] == 1
    assert launch["seed"] == 42
    assert launch["scenario_revision_id"] == "instruction-sft@1"
    assert launch["field_mapping_plan"]["confirmed"] is True
    assert [binding["role"] for binding in launch["dataset_bindings"]] == [
        "train",
        "validation",
    ]
    assert launch["dataset_bindings"][1]["split"] == "validation"
    assert len(launch["proof_sample_identity"]) == 64
    assert result["proof_sample_identity"] == launch["proof_sample_identity"]

    metadata = service._dataset_version_metadata(launch)
    assert metadata is not None
    assert metadata["guided_own_data"]["proof_run"] is True
    assert (
        metadata["guided_own_data"]["dataset_preparation_recipe"]
        == launch["dataset_preparation_recipe"]
    )
    db.close()


def test_guided_raft_proof_uses_real_prompt_cap(tmp_path: Path) -> None:
    db = RunDatabase(":memory:")
    version_id = _guided_version(
        db,
        tmp_path,
        scenario_revision_id="prompt-reward@1",
        schema="prompt",
        modality="text",
    )
    service = _service(db, tmp_path)
    verifier_engine = _QualifiedVerifierEngine()
    service._verifier_engine = lambda: verifier_engine  # type: ignore[method-assign]
    captured: list[dict[str, Any]] = []

    missing = service.get_dataset_version_readiness(
        version_id,
        trainer_mode="raft",
        model="local/tiny-model",
    )
    assert "qualified_verifier_missing" in {
        blocker["code"] for blocker in missing["blockers"]
    }

    normalized_preflight = service._normalize_guided_proof_payload(
        {
            "mode": "raft",
            "model": "local/tiny-model",
            "dataset_version_id": version_id,
            "proof_run": True,
            "max_samples": 300,
            "cycles": 9,
            "output_root": str(tmp_path / "runs"),
        }
    )
    assert normalized_preflight["max_prompts"] == 200
    assert normalized_preflight["cycles"] == 1
    assert normalized_preflight["output_dir"] == str(tmp_path / "runs")
    assert [item["role"] for item in normalized_preflight["dataset_bindings"]] == [
        "train",
        "validation",
    ]

    async def fake_launch(payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(payload)
        return {"id": "raft-proof", "run_id": "raft-proof", "status": "pending"}

    service.launch_training = fake_launch  # type: ignore[method-assign]
    asyncio.run(
        service.launch_dataset_proof_run(
            version_id,
            {
                "scenario_revision_id": "prompt-reward@1",
                "trainer_mode": "raft",
                "model": "local/tiny-model",
                # This is what the generic frontend proof payload historically
                # called the bound. The proof endpoint translates it truthfully.
                "max_samples": 300,
                "cycles": 9,
                "verifier_profile_revision_id": "qualified-verifier-revision",
                "output_root": str(tmp_path / "runs"),
            },
        )
    )
    launch = captured[0]
    assert launch["cycles"] == 1
    assert launch["max_prompts"] == launch["proof_max_samples"] == 200
    assert "max_samples" not in launch
    assert launch["verifier_profile_revision_id"] == "qualified-verifier-revision"
    assert verifier_engine.calls[-1] == (
        "qualified-verifier-revision",
        {"modality": "text", "require_qualified": True},
    )
    command = PublicApiService._managed_training_command(
        {
            **launch,
            "prompts": str(tmp_path / "artifact" / "train.jsonl"),
            "output_dir": str(tmp_path / "run"),
        }
    )
    assert command[command.index("--max-prompts") + 1] == "200"
    db.close()


def test_guided_reward_proof_refuses_missing_or_unqualified_verifier(
    tmp_path: Path,
) -> None:
    db = RunDatabase(":memory:")
    version_id = _guided_version(
        db,
        tmp_path,
        scenario_revision_id="prompt-reward@1",
        schema="prompt",
        modality="text",
    )
    service = _service(db, tmp_path)
    service._verifier_engine = lambda: _QualifiedVerifierEngine()  # type: ignore[method-assign]

    base = {
        "scenario_revision_id": "prompt-reward@1",
        "trainer_mode": "grpo",
        "model": "local/tiny-model",
        "output_root": str(tmp_path / "runs"),
    }
    with pytest.raises(ValueError, match="requires a candidate- or approved-qualified"):
        asyncio.run(service.launch_dataset_proof_run(version_id, base))
    with pytest.raises(ValueError, match="not ready for guided GRPO"):
        asyncio.run(
            service.launch_dataset_proof_run(
                version_id,
                {**base, "verifier_profile_revision_id": "unqualified-revision"},
            )
        )
    db.close()


def test_guided_proof_rejects_selected_mode_blocked_by_active_runtime(
    tmp_path: Path,
) -> None:
    db = RunDatabase(":memory:")
    version_id = _guided_version(
        db,
        tmp_path,
        scenario_revision_id="preference-pairs@1",
        schema="preference",
        modality="text",
    )
    service = _service(db, tmp_path)
    service._guided_data_engine = lambda: SimpleNamespace(  # type: ignore[method-assign]
        _runtime_scenario_status=lambda _scenario, _backend: (True, None),
        runtime_trainer_compatibility=lambda _scenario, _backend, trainer_mode=None: [
            {
                "trainer_mode": str(trainer_mode),
                "compatible": False,
                "reason": "ORPO training is not implemented on MLX.",
            }
        ],
    )
    service._cached_backend_name = "mlx"

    with pytest.raises(ValueError, match="ORPO training is not implemented on MLX"):
        asyncio.run(
            service.launch_dataset_proof_run(
                version_id,
                {
                    "scenario_revision_id": "preference-pairs@1",
                    "trainer_mode": "orpo",
                    "model": "mlx-community/tiny-model",
                    "output_root": str(tmp_path / "runs"),
                },
            )
        )
    db.close()


def test_full_run_clones_completed_proof_and_removes_only_proof_caps(
    tmp_path: Path,
) -> None:
    service = PublicApiService(base_path=tmp_path, database=RunDatabase(":memory:"))
    parent_config = {
        "mode": "sft",
        "model": "local/tiny-model",
        "dataset": str(tmp_path / "artifact" / "train.jsonl"),
        "dataset_version_id": "version-sft",
        "dataset_split": "train",
        "dataset_bindings": [
            {"role": "train", "dataset_version_id": "version-sft", "split": "train"},
            {
                "role": "validation",
                "dataset_version_id": "version-sft",
                "split": "validation",
            },
        ],
        "output_root": str(tmp_path / "runs"),
        "output_dir": str(tmp_path / "runs" / "proof-1"),
        "epochs": 1,
        "max_samples": 200,
        "proof_run": True,
        "proof_max_samples": 200,
        "proof_sample_identity": "a" * 64,
        "scenario_revision_id": "instruction-sft@1",
        "field_mapping_plan": {"version": 2, "confirmed": True, "mappings": {}},
        "dataset_preparation_recipe": {"steps": []},
        "training_artifact_id": "artifact-1",
    }
    service.get_run_detail = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "id": "proof-1",
        "run_id": "proof-1",
        "status": "completed",
        "output_dir": parent_config["output_dir"],
    }
    service.get_resolved_run_launch_config = lambda _run_id: {  # type: ignore[method-assign]
        "run_id": "proof-1",
        "config": dict(parent_config),
    }
    captured: list[dict[str, Any]] = []

    async def fake_launch(payload: dict[str, Any]) -> dict[str, Any]:
        captured.append(payload)
        return {"id": "full-1", "run_id": "full-1", "status": "pending"}

    service.launch_training = fake_launch  # type: ignore[method-assign]
    service.get_full_run_context = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "allowed": True,
        "decision": "override",
    }
    service.review_training_outcome = lambda *_args, **_kwargs: {}  # type: ignore[method-assign]
    result = asyncio.run(
        service.launch_full_run_from_proof(
            "proof-1",
            {"override_reason": "Reviewed legacy proof before the V11 assessment layer."},
        )
    )
    launch = captured[0]
    assert launch["proof_run"] is False
    assert launch["full_run_from_proof"] is True
    assert launch["parent_run_id"] == launch["proof_parent_run_id"] == "proof-1"
    assert "max_samples" not in launch
    assert "proof_max_samples" not in launch
    assert "proof_sample_identity" not in launch
    assert launch["dataset_bindings"] == parent_config["dataset_bindings"]
    assert launch["training_artifact_id"] == "artifact-1"
    assert launch["outcome_override_reason"]
    assert result["parent_run_id"] == "proof-1"


def test_full_run_refuses_incomplete_or_non_proof_parent(tmp_path: Path) -> None:
    service = PublicApiService(base_path=tmp_path, database=RunDatabase(":memory:"))
    service.get_run_detail = lambda *_args, **_kwargs: {"status": "running"}  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="complete successfully"):
        asyncio.run(service.launch_full_run_from_proof("proof-1"))

    service.get_run_detail = lambda *_args, **_kwargs: {"status": "completed"}  # type: ignore[method-assign]
    service.get_resolved_run_launch_config = lambda _run_id: {  # type: ignore[method-assign]
        "run_id": "ordinary-1",
        "config": {"mode": "sft", "proof_run": False},
    }
    with pytest.raises(ValueError, match="not launched as a guided proof"):
        asyncio.run(service.launch_full_run_from_proof("ordinary-1"))


def test_public_routes_expose_guided_proof_and_full_run(monkeypatch: Any) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from halo_forge.public_api import app as app_module

    calls: list[tuple[str, str, dict[str, Any]]] = []

    class StubService:
        async def launch_dataset_proof_run(
            self, version_id: str, payload: dict[str, Any]
        ) -> dict[str, Any]:
            calls.append(("proof", version_id, payload))
            return {"run_id": "proof-1", "proof_run": True, "status": "pending"}

        async def launch_full_run_from_proof(
            self, run_id: str, payload: dict[str, Any]
        ) -> dict[str, Any]:
            calls.append(("full", run_id, payload))
            return {
                "run_id": "full-1",
                "parent_run_id": run_id,
                "full_run_from_proof": True,
                "status": "pending",
            }

    monkeypatch.setenv("HALOFORGE_DISABLE_AUTO_WORKER", "1")
    monkeypatch.setattr(app_module, "PublicApiService", StubService)
    with TestClient(app_module.create_app(serve_frontend=False)) as client:
        proof = client.post(
            "/api/public/dataset-versions/version-1/proof-run",
            json={"model": "local/tiny", "trainer_mode": "sft"},
        )
        full = client.post("/api/public/runs/proof-1/full-run", json={})
    assert proof.status_code == 200
    assert proof.json()["proof_run"] is True
    assert full.status_code == 200
    assert full.json()["parent_run_id"] == "proof-1"
    assert calls == [
        ("proof", "version-1", {"model": "local/tiny", "trainer_mode": "sft"}),
        ("full", "proof-1", {}),
    ]
