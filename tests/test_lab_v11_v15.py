from __future__ import annotations

import json
from pathlib import Path

import pytest

from halo_forge.data_lab.models import adapt_record, infer_schema, validate_record
from halo_forge.data_lab.training_artifacts import TRAINER_DATASET_ADAPTERS
from halo_forge.lab_v11_v15 import EvidenceEligibilityError, FutureLabService
from halo_forge.public_api.service import PublicApiService
from halo_forge.replay import MANIFEST_VERSION, capture_manifest, load_manifest, save_manifest
from halo_forge.run_db import RunDatabase, RunRecord
from halo_forge.run_db.schema import SCHEMA_VERSION
from halo_forge.serving.app import create_serving_app


@pytest.fixture
def lab(tmp_path: Path) -> tuple[RunDatabase, FutureLabService]:
    database = RunDatabase(str(tmp_path / "catalog.db"))
    return database, FutureLabService(database, root=tmp_path)


def test_schema_v19_additive_catalog(lab: tuple[RunDatabase, FutureLabService]) -> None:
    database, _ = lab
    assert SCHEMA_VERSION == 23
    assert (
        database._conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()[0]
        == "23"
    )
    names = {
        row[0]
        for row in database._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "training_outcome_assessments",
        "adaptation_studies",
        "grounded_generation_batches",
        "task_label_schema_revisions",
        "agent_episodes",
        "trajectory_set_revisions",
    } <= names


@pytest.mark.parametrize(
    ("record", "kind"),
    [
        ({"text": "hello", "label": "greeting"}, "classification"),
        ({"anchor": "reset password", "positive": "Open settings"}, "embedding"),
        ({"query": "refund", "document": "30 days", "relevance": 1}, "reranking"),
    ],
)
def test_specialized_canonical_shapes(record: dict, kind: str) -> None:
    assert infer_schema(record).value == kind
    canonical = adapt_record(record, kind)
    validate_record(canonical, kind)
    adapter = TRAINER_DATASET_ADAPTERS.resolve(
        schema=kind,
        trainer_mode={
            "classification": "classify",
            "embedding": "embed",
            "reranking": "rerank",
        }[kind],
    )
    assert adapter.render_record(canonical)


def test_outcome_gate_requires_evidence_or_reason(
    lab: tuple[RunDatabase, FutureLabService], tmp_path: Path
) -> None:
    database, service = lab
    database.upsert_run(
        RunRecord(
            run_id="proof-1",
            modality="sft",
            status="completed",
            weights_updated=True,
            final_model_path=str(tmp_path / "model"),
            raw_json=json.dumps(
                {
                    "proof_run": True,
                    "scenario_revision_id": "instruction-sft@1",
                    "launch_config": {
                        "proof_run": True,
                        "scenario_revision_id": "instruction-sft@1",
                        "mode": "sft",
                    },
                }
            ),
        )
    )
    assessment = service.assess_outcome("proof-1", {})
    assert assessment.technical_status == "verified"
    assert assessment.status == "incomplete_evidence"
    with pytest.raises(ValueError, match="incomplete"):
        service.full_run_context("proof-1", assessment_id=assessment.id)
    context = service.full_run_context(
        "proof-1", override_reason="The proof is a time-bounded infrastructure check."
    )
    assert context["override_required"] is True
    assert context["resolved_config"]["parent_run_id"] == "proof-1"


def test_paired_study_materialization_and_analysis(
    lab: tuple[RunDatabase, FutureLabService]
) -> None:
    _, service = lab
    study = service.create_study({"name": "Corpus validity"})
    protocol = service.create_study_protocol(
        study.id,
        {
            "design_kind": "paired_ab",
            "question": "Does the corpus improve the domain metric?",
            "arms": [
                {"name": "Control", "is_control": True, "launch_config": {}},
                {"name": "Adapted", "launch_config": {}},
            ],
            "seeds": [17, 42, 101],
            "contrasts": [
                {
                    "name": "Adapted versus control",
                    "left_arm": "Control",
                    "right_arm": "Adapted",
                    "metric": "domain_score",
                    "direction": "maximize",
                }
            ],
        },
    )
    assert len(protocol.assignments) == 6
    metrics = {}
    for assignment in protocol.assignments:
        arm = next(item for item in protocol.arms if item.id == assignment.arm_id)
        metrics[assignment.id] = {
            "domain_score": 0.8 if arm.name == "Adapted" else 0.5
        }
    analysis = service.analyze_study(protocol.id, {"metrics": metrics})
    contrast = analysis.analysis["contrasts"][0]
    assert contrast["classification"] == "superior"
    assert contrast["paired_seed_count"] == 3
    assert analysis.evidence_classification == "causal"
    assert Path(analysis.bundle_path or "", "report.md").is_file()


def test_grounding_is_cited_review_input_only(
    lab: tuple[RunDatabase, FutureLabService]
) -> None:
    _, service = lab
    profile = service.create_grounding_profile({"name": "Grounded QA"})
    revision = service.create_grounding_profile_revision(
        profile.id,
        {"task_type": "qa", "intended_destination": "training", "quota": 2},
    )
    batch = service.generate_grounded_batch(
        {
            "profile_revision_id": revision.id,
            "records": [
                {
                    "document_id": "doc-1",
                    "source_ref": "guide.md",
                    "text": "Dataset versions are immutable.",
                }
            ],
        }
    )
    assert batch.accepted_count == 1
    candidate = service.list_grounded_candidates(batch.id)["items"][0]
    assert candidate["citations"][0]["structural_valid"] is True
    proposal = service.grounding_review_proposal(batch.id)
    assert proposal["requires_explicit_queue_creation"] is True
    with pytest.raises(EvidenceEligibilityError):
        service.create_grounding_profile_revision(
            profile.id,
            {"task_type": "qa", "intended_destination": "holdout"},
        )


def test_specialized_metrics_and_command_contract(
    lab: tuple[RunDatabase, FutureLabService], tmp_path: Path
) -> None:
    _, service = lab
    metrics = service.classification_metrics(
        ["a", "b", "b"], ["a", "a", "b"]
    )
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    retrieval = service.retrieval_metrics(
        [["d1", "d2"], ["d3", "d4"]],
        [["d2"], ["d3"]],
        k=2,
    )
    assert retrieval["recall_at_2"] == 1.0
    command = PublicApiService._managed_training_command(
        {
            "mode": "classify",
            "model": "distilbert/distilbert-base-uncased",
            "dataset": str(tmp_path / "train.jsonl"),
            "output_dir": str(tmp_path / "run"),
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 2e-5,
            "seed": 42,
            "proof_run": True,
        }
    )
    assert command[3:5] == ["classify", "train"]
    assert "--proof-run" in command


def test_deterministic_environment_replay_compare_and_trajectory(
    lab: tuple[RunDatabase, FutureLabService]
) -> None:
    _, service = lab
    environment = service.create_environment({"name": "Local fixture"})
    revision = service.create_environment_revision(
        environment.id,
        {
            "adapter_id": "state_machine",
            "initial_state": {"done": False},
            "transitions": {
                "finish": {
                    "state_delta": {"done": True},
                    "reward": 1,
                    "terminal": True,
                }
            },
        },
    )
    suite = service.create_episode_suite(
        {"name": "Development episodes", "purpose": "development"}
    )
    suite_revision = service.create_episode_suite_revision(
        suite["id"],
        {
            "environment_revision_id": revision.id,
            "items": [
                {
                    "id": "finish",
                    "goal": "Finish",
                    "expected_state": {"done": True},
                }
            ],
        },
    )
    base = service.run_episode(
        {
            "suite_revision_id": suite_revision.id,
            "suite_item_id": "finish",
            "subject_ref": "base",
            "actions": [{"name": "unknown"}],
        }
    )
    candidate = service.run_episode(
        {
            "suite_revision_id": suite_revision.id,
            "suite_item_id": "finish",
            "subject_ref": "candidate",
            "actions": [{"name": "finish"}],
        }
    )
    assert service.replay_episode(candidate.id)["valid"] is True
    comparison = service.compare_environment_subjects(
        suite_revision_id=suite_revision.id,
        base_subject_hash=base.subject_hash,
        candidate_subject_hash=candidate.subject_hash,
    )
    assert comparison.counts["improved"] == 1
    trajectories = service.publish_trajectory_set(
        {"episode_ids": [candidate.id], "output_adapter": "tool_sft"}
    )
    assert trajectories.row_count == 1
    assert Path(trajectories.storage_path, "records.jsonl").is_file()


def test_replay_v9_captures_new_domain_identities(tmp_path: Path) -> None:
    manifest = capture_manifest(
        run_id="run-v9",
        modality="classify",
        model_name="distilbert",
        seed=42,
        config={"epochs": 1},
        training_outcome_binding={"assessment_id": "outcome-1"},
        adaptation_study_binding={"protocol_revision_id": "study-r1"},
        specialized_task_binding={"task": "classification", "label_schema_hash": "abc"},
        agent_environment_binding={"environment_revision_id": "env-r1"},
    )
    assert MANIFEST_VERSION == 14
    path = save_manifest(manifest, tmp_path)
    loaded = load_manifest(path)
    assert loaded.training_outcome["assessment_id"] == "outcome-1"
    assert loaded.adaptation_study["protocol_revision_id"] == "study-r1"
    assert loaded.specialized_task["task"] == "classification"
    assert loaded.agent_environment["environment_revision_id"] == "env-r1"


def test_specialized_openai_compatible_serving_endpoints() -> None:
    from fastapi.testclient import TestClient

    class Runtime:
        def embed(self, inputs):
            return [[float(index), 1.0] for index, _ in enumerate(inputs)]

        def classify(self, inputs, *, top_k):
            return [[{"label": "ok", "score": 0.9}][:top_k] for _ in inputs]

        def rerank(self, query, documents, *, top_n=None):
            values = [
                {"index": index, "document": value, "score": float(len(value))}
                for index, value in enumerate(documents)
            ]
            values.sort(key=lambda item: item["score"], reverse=True)
            return values[:top_n] if top_n else values

    client = TestClient(
        create_serving_app(model_name="task-artifact", adapter=Runtime())
    )
    embeddings = client.post(
        "/v1/embeddings",
        json={"model": "task-artifact", "input": ["a", "b"]},
    )
    assert embeddings.status_code == 200
    assert len(embeddings.json()["data"]) == 2
    classifications = client.post(
        "/v1/classifications",
        json={"model": "task-artifact", "input": ["hello"], "top_k": 1},
    )
    assert classifications.json()["data"][0]["predictions"][0]["label"] == "ok"
    reranked = client.post(
        "/v1/rerank",
        json={
            "model": "task-artifact",
            "query": "q",
            "documents": ["short", "much longer"],
            "top_n": 1,
        },
    )
    assert reranked.json()["results"][0]["document"] == "much longer"


def test_v11_v15_guided_api_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from halo_forge.auth.dependency import reset_store_for_tests
    from halo_forge.public_api import app as app_module

    database = RunDatabase(tmp_path / "public.db")
    service = PublicApiService(
        database=database,
        base_path=tmp_path,
        dataset_storage_root=tmp_path / "datasets",
    )
    monkeypatch.setenv("HALOFORGE_DISABLE_AUTO_WORKER", "1")
    monkeypatch.setattr(app_module, "PublicApiService", lambda: service)
    reset_store_for_tests(None)

    with TestClient(app_module.create_app(serve_frontend=False)) as client:
        capabilities = client.get("/api/public/lab-capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["studies"]["max_arms"] == 4

        study = client.post(
            "/api/public/adaptation-studies", json={"name": "API study"}
        )
        assert study.status_code == 201, study.text
        protocol = client.post(
            f"/api/public/adaptation-studies/{study.json()['id']}/protocols",
            json={
                "question": "Does the adapted data improve domain evidence?",
                "design_kind": "paired_ab",
                "seeds": [17, 42, 101],
                "development_suite_purpose": "development",
                "retention_suite_purpose": "development",
                "arms": [
                    {
                        "name": "Control",
                        "is_control": True,
                        "factor_values": {},
                        "launch_config": {},
                    },
                    {
                        "name": "Adapted",
                        "factor_values": {},
                        "launch_config": {},
                    },
                ],
                "contrasts": [
                    {
                        "name": "Adapted versus control",
                        "left_arm": "Control",
                        "right_arm": "Adapted",
                        "metric": "primary_metric",
                        "direction": "maximize",
                        "conclusion_kind": "superiority",
                    }
                ],
            },
        )
        assert protocol.status_code == 201, protocol.text
        assert len(protocol.json()["assignments"]) == 6

        environment = client.post(
            "/api/public/environments", json={"name": "API environment"}
        )
        assert environment.status_code == 201, environment.text
        revision = client.post(
            f"/api/public/environments/{environment.json()['id']}/revisions",
            json={
                "adapter_id": "state_machine",
                "initial_state": {"done": False},
                "transitions": {
                    "finish": {
                        "state_delta": {"done": True},
                        "reward": 1,
                        "terminal": True,
                    }
                },
                "tools": [{"name": "finish", "input_schema": {"type": "object"}}],
                "max_steps": 2,
            },
        )
        assert revision.status_code == 201, revision.text
        suite = client.post(
            f"/api/public/environment-revisions/{revision.json()['id']}/suites",
            json={"name": "API episodes", "purpose": "development"},
        )
        assert suite.status_code == 201, suite.text
        suite_revision = client.post(
            f"/api/public/environment-suites/{suite.json()['id']}/revisions",
            json={
                "environment_revision_id": revision.json()["id"],
                "items": [
                    {
                        "id": "finish",
                        "goal": "Finish",
                        "expected_state": {"done": True},
                    }
                ],
                "generation": {"seed": 42, "temperature": 0},
                "max_steps": 2,
            },
        )
        assert suite_revision.status_code == 201, suite_revision.text
        episode = client.post(
            "/api/public/environment-suite-revisions/"
            f"{suite_revision.json()['id']}/episodes",
            json={
                "suite_item_id": "finish",
                "subject_type": "recorded_plan",
                "subject_ref": "api-smoke",
                "seed": 42,
                "actions": [{"name": "finish", "raw_output": "finish"}],
            },
        )
        assert episode.status_code == 202, episode.text
        assert episode.json()["status"] == "queued"
        from halo_forge.workstation_jobs import WorkstationWorker

        terminal = WorkstationWorker(service._scheduler()).run_once(
            work_item_id=episode.json()["work_item_id"]
        )
        assert terminal is not None
        assert terminal.status == "completed"
        replay = client.post(
            f"/api/public/environment-episodes/{episode.json()['id']}/replay"
        )
        assert replay.status_code == 200, replay.text
        assert replay.json()["valid"] is True

    database.close()
