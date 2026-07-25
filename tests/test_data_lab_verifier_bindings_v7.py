"""Exact verifier-revision contracts in Dataset Lab recipe execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from halo_forge.data_lab import DatasetLab, RecipeContext, RecipeError, RecipeRunner
from halo_forge.run_db import RunDatabase
from halo_forge.verifier_lab import VerifierLabService
from halo_forge.workstation_jobs import WorkstationScheduler


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _verifier_lab(tmp_path: Path) -> tuple[RunDatabase, VerifierLabService, dict]:
    database = RunDatabase(str(tmp_path / "runs.db"))
    service = VerifierLabService(
        database,
        root=tmp_path / "calibrations",
        scheduler=WorkstationScheduler(database),
    )
    created = service.create_profile(
        name="Pinned JSON quality verifier",
        description=None,
        definition={
            "family": "deterministic",
            "implementation": {"ref": "json_structure"},
            "modality": "text",
            "task_type": "binary",
            "input_mapping": {"candidate": "response"},
            "reward_contract": {
                "minimum": 0.0,
                "maximum": 1.0,
                "direction": "maximize",
                "threshold": 0.5,
                "tie_policy": "fail",
                "error_behavior": "fail_closed",
            },
            "runtime_contract": {},
        },
    )
    return database, service, created["revision"]


def test_exact_score_revision_persists_trace_and_binds_published_version(tmp_path):
    database, verifier_service, revision = _verifier_lab(tmp_path)
    source_path = _jsonl(
        tmp_path / "source.jsonl",
        [
            {"id": "valid", "response": '{"accepted": true}'},
            {"id": "invalid", "response": "not-json"},
        ],
    )
    lab = DatasetLab(
        tmp_path / "datasets",
        database=database,
        verifier_service=verifier_service,
    )
    source = lab.add_source({"kind": "local", "path": str(source_path)})
    version = lab.build(
        source.id,
        {
            "steps": [
                {
                    "kind": "score",
                    "method": "verifier",
                    "verifier_profile_revision_id": revision["id"],
                    "threshold": 0.5,
                }
            ]
        },
    )

    provenance = json.loads((Path(version.path) / "provenance.json").read_text())
    details = provenance[0]["details"]
    assert details["legacy_unqualified"] is False
    assert details["verifier_binding"]["verifier_profile_revision_id"] == revision["id"]
    assert details["verifier_binding"]["revision_hash"] == revision["content_hash"]
    assert [value["reward"] for value in details["observations"]] == [1.0, 0.0]
    assert [value["accepted"] for value in details["observations"]] == [True, False]
    assert all(value["record_id"] for value in details["observations"])

    bindings = verifier_service.store.list_bindings(
        domain_kind="dataset_version",
        domain_id=version.version_id,
    )
    assert len(bindings) == 1
    assert bindings[0].verifier_revision_id == revision["id"]
    assert bindings[0].role == "dataset_score_verifier"
    assert bindings[0].context["dataset_content_hash"] == version.content_hash
    assert bindings[0].context["revision_hash"] == revision["content_hash"]


def test_exact_synthesis_revision_uses_contract_and_records_component_evidence(tmp_path):
    database, verifier_service, revision = _verifier_lab(tmp_path)
    source_path = _jsonl(tmp_path / "seeds.jsonl", [{"id": "seed", "prompt": "annotate"}])
    lab = DatasetLab(
        tmp_path / "datasets",
        teacher=lambda *_args: '{"label": "safe"}',
        database=database,
        verifier_service=verifier_service,
    )
    source = lab.add_source({"kind": "local", "path": str(source_path)})
    version = lab.build(
        source.id,
        {
            "steps": [
                {
                    "kind": "synthesize",
                    "teacher_model": "teacher-revision-a",
                    "verifier_profile_revision_id": revision["id"],
                }
            ]
        },
    )

    provenance = json.loads((Path(version.path) / "provenance.json").read_text())
    details = provenance[0]["details"]
    assert details["verifier_binding"]["revision_hash"] == revision["content_hash"]
    assert details["observations"][0]["reward"] == 1.0
    record = json.loads((Path(version.path) / "records.jsonl").read_text().strip())
    assert record["metadata"]["synthesis"]["verifier"] == {
        "legacy_unqualified": False,
        "revision_hash": revision["content_hash"],
        "verifier_profile_revision_id": revision["id"],
    }
    bindings = verifier_service.store.list_bindings(
        domain_kind="dataset_version",
        domain_id=version.version_id,
    )
    assert [value.role for value in bindings] == ["dataset_synthesis_verifier"]


def test_exact_revision_rejects_raw_verifier_and_conflicting_threshold(tmp_path):
    database, verifier_service, revision = _verifier_lab(tmp_path)
    context = RecipeContext(
        teacher=lambda _prompt: "{}",
        verifier_profile_resolver=verifier_service.resolve_binding,
        verifier_profile_invoker=verifier_service.invoke_revision,
    )
    with pytest.raises(RecipeError, match="conflicts with raw verifier fields"):
        RecipeRunner(context).run(
            [{"response": "{}"}],
            {
                "steps": [
                    {
                        "kind": "score",
                        "method": "verifier",
                        "verifier_profile_revision_id": revision["id"],
                        "verifier": "json_structure",
                    }
                ]
            },
        )
    with pytest.raises(RecipeError, match="immutable reward contract threshold is 0.5"):
        RecipeRunner(context).run(
            [{"response": "{}"}],
            {
                "steps": [
                    {
                        "kind": "synthesize",
                        "verifier_profile_revision_id": revision["id"],
                        "threshold": 0.8,
                    }
                ]
            },
        )


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), 1.01, -0.01])
def test_exact_revision_rejects_invalid_observation_rewards_without_clamping(tmp_path, reward):
    _database, verifier_service, revision = _verifier_lab(tmp_path)
    result = RecipeRunner(
        RecipeContext(
            verifier_profile_resolver=verifier_service.resolve_binding,
            verifier_profile_invoker=lambda _revision_id, _row: {
                "reward": reward,
                "passed": True,
                "details": {"fixture": "invalid-reward"},
            },
        )
    ).run(
        [{"id": "row", "response": "{}"}],
        {
            "steps": [
                {
                    "kind": "score",
                    "method": "verifier",
                    "verifier_profile_revision_id": revision["id"],
                }
            ]
        },
    )
    assert result.records == []
    assert len(result.rejected) == 1
    trace = result.provenance[0].details["observations"][0]
    assert trace["accepted"] is False
    assert "verifier reward" in trace["rejection_reason"]


@pytest.mark.parametrize("reward", [float("inf"), 1.25, -0.01])
def test_legacy_raw_verifier_rewards_are_rejected_not_clamped(reward):
    result = RecipeRunner(RecipeContext(verifier=lambda _row: reward)).run(
        [{"id": "row", "response": "candidate"}],
        {
            "steps": [
                {
                    "kind": "score",
                    "method": "verifier",
                    "verifier": "legacy-plugin",
                }
            ]
        },
    )
    assert result.records == []
    assert len(result.rejected) == 1
    details = result.provenance[0].details
    assert details["legacy_unqualified"] is True
    assert details["verifier_binding"]["legacy_unqualified"] is True
    assert details["observations"][0]["accepted"] is False
    assert "rejection_reason" in details["observations"][0]


def test_valid_legacy_score_and_synthesis_remain_runnable_but_unqualified():
    score = RecipeRunner(RecipeContext(verifier=lambda _row: 0.75)).run(
        [{"id": "score", "response": "candidate"}],
        {
            "steps": [
                {
                    "kind": "score",
                    "method": "verifier",
                    "verifier": "legacy-plugin",
                    "threshold": 0.5,
                }
            ]
        },
    )
    assert len(score.records) == 1
    assert score.records[0]["_quality_score"] == 0.75
    assert score.provenance[0].details["legacy_unqualified"] is True

    synthesis = RecipeRunner(
        RecipeContext(
            teacher=lambda _prompt: "completion",
            verifier=lambda _row: 0.8,
        )
    ).run(
        [{"id": "synthesis", "prompt": "question"}],
        {
            "steps": [
                {
                    "kind": "synthesize",
                    "verifier": "legacy-plugin",
                    "threshold": 0.5,
                }
            ]
        },
    )
    assert len(synthesis.records) == 1
    details = synthesis.provenance[0].details
    assert details["legacy_unqualified"] is True
    assert details["verifier_binding"]["legacy_unqualified"] is True
    assert details["observations"][0]["reward"] == 0.8
