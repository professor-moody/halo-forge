"""FastAPI coverage for the Dataset Lab public resources."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("HALOFORGE_DATASET_ROOT", str(tmp_path / "datasets"))

    from halo_forge.run_db import db as db_mod
    from halo_forge.workstation_jobs import scheduler as scheduler_mod
    from halo_forge.workstation_jobs import DiskCapacity, MemoryCapacity, WorkstationCapacity

    gib = 1024**3
    monkeypatch.setattr(
        scheduler_mod,
        "sample_workstation_capacity",
        lambda path, **_: WorkstationCapacity(
            sampled_at=datetime.now(timezone.utc),
            disk=DiskCapacity(
                path=str(path),
                total_bytes=200 * gib,
                used_bytes=100 * gib,
                free_bytes=100 * gib,
            ),
            memory=MemoryCapacity(
                total_bytes=32 * gib,
                used_bytes=8 * gib,
                available_bytes=24 * gib,
            ),
        ),
    )

    db_mod._GLOBAL_DB.clear()
    from halo_forge.auth.dependency import reset_store_for_tests
    from fastapi.testclient import TestClient
    from halo_forge.public_api.app import create_app

    reset_store_for_tests(None)
    with TestClient(create_app(serve_frontend=False)) as value:
        yield value, tmp_path
    db_mod._GLOBAL_DB.clear()


def _register(client, tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(
        '{"prompt":"one","response":"1"}\n'
        '{"prompt":"two","response":"2"}\n'
        '{"prompt":"three","response":"3"}\n',
        encoding="utf-8",
    )
    response = client.post(
        "/api/public/datasets",
        json={
            "name": "numbers",
            "canonical_schema": "sft",
            "source": {"kind": "local", "uri": str(source)},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _await_guided_registration(http, response):
    assert response.status_code == 202, response.text
    accepted = response.json()
    assert accepted["work_item_id"]
    import_id = accepted["import"]["id"]
    session = accepted["import"]
    for _ in range(300):
        session = http.get(f"/api/public/dataset-imports/{import_id}").json()
        if session.get("published_dataset_id") or session["status"] in {
            "failed",
            "cancelled",
        }:
            break
        time.sleep(0.01)
    assert session["status"] == "published", session
    work = http.get(f"/api/public/work-items/{accepted['work_item_id']}").json()
    assert work["status"] == "completed", work
    return http.get(
        f"/api/public/datasets/{session['published_dataset_id']}"
    ).json()


def _assert_scenario_availability_contract(scenarios) -> None:
    """Assert the runtime-availability contract for training scenarios.

    `available` must be a bool, and `unavailable_reason` must be populated if
    and only if `available` is False. Whether any individual scenario is
    available depends on what the host has installed, so that is deliberately
    not asserted here.
    """
    for scenario in scenarios:
        available = scenario["available"]
        assert isinstance(available, bool), scenario
        reason = scenario.get("unavailable_reason")
        if available:
            assert not reason, scenario
        else:
            assert isinstance(reason, str) and reason.strip(), scenario


def test_training_scenario_availability_contract_holds_in_both_branches(
    client, monkeypatch
):
    """Both availability branches must honour the reason contract.

    The availability probe (`_module_available`) is the host-dependent seam;
    pinning it makes the available and unavailable branches deterministic on
    any runner, including CI jobs that never install torch.
    """
    http, _ = client
    from halo_forge.own_data import service as own_data_service
    from halo_forge.public_api.service import PublicApiService

    monkeypatch.setattr(PublicApiService, "_active_backend_name", lambda self: "cpu")

    def _scenarios():
        response = http.get("/api/public/training-scenarios?include_unavailable=true")
        assert response.status_code == 200, response.text
        return {item["id"]: item for item in response.json()["items"]}

    monkeypatch.setattr(own_data_service, "_module_available", lambda _name: True)
    ready = _scenarios()
    _assert_scenario_availability_contract(ready.values())
    assert ready["audio-classification"]["available"] is True
    assert not ready["audio-classification"]["unavailable_reason"]
    assert ready["audio-classification"]["trainer_modes"] == ["classify"]

    monkeypatch.setattr(own_data_service, "_module_available", lambda _name: False)
    thin = _scenarios()
    _assert_scenario_availability_contract(thin.values())
    assert thin["audio-classification"]["available"] is False
    assert thin["audio-classification"]["unavailable_reason"]
    assert thin["audio-classification"]["trainer_modes"] == []


def test_dataset_registration_list_detail_preview_and_statistics(client):
    http, root = client
    created = _register(http, root)

    listed = http.get("/api/public/datasets").json()["items"]
    assert listed[0]["id"] == created["id"]
    assert listed[0]["row_count"] == 3
    assert http.get(f"/api/public/datasets/{created['id']}").json()["sources"]

    preview = http.get(f"/api/public/datasets/{created['id']}/preview?offset=1&limit=1").json()
    assert preview["total"] == 3
    assert preview["items"] == [{"prompt": "two", "response": "2"}]
    stats = http.get(f"/api/public/datasets/{created['id']}/statistics").json()
    assert stats["row_count"] == 3


def test_dataset_build_job_version_preview_and_export(client):
    http, root = client
    created = _register(http, root)
    build = http.post(
        f"/api/public/datasets/{created['id']}/build",
        json={
            "recipe": {
                "seed": 11,
                "steps": [{"kind": "split", "ratios": {"train": 0.67, "test": 0.33}}],
            }
        },
    )
    assert build.status_code == 202, build.text
    job_id = build.json()["id"]
    job = None
    for _ in range(100):
        response = http.get(f"/api/public/dataset-jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert job["status"] == "completed", job
    version_id = job["version_id"]

    version = http.get(f"/api/public/dataset-versions/{version_id}").json()
    assert version["content_hash"]
    assert version["split_counts"] == {"train": 2, "test": 1}
    assert {
        (item["adapter_id"], item["trainer_mode"])
        for item in version["trainer_compatibility"]
    } == {("sft", "sft")}
    assert version["rejections"]["rejected_count"] == 0
    assert "contamination" in version
    preview = http.get(f"/api/public/dataset-versions/{version_id}/preview?split=test").json()
    assert preview["total"] == 1

    destination = root / "exported.jsonl"
    exported = http.post(
        f"/api/public/dataset-versions/{version_id}/export",
        json={"output": str(destination), "split": "train"},
    )
    assert exported.status_code == 200, exported.text
    assert destination.is_file()
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 2


def test_training_artifact_render_returns_job_and_catalogs_after_poll(client):
    http, root = client
    created = _register(http, root)
    build = http.post(
        f"/api/public/datasets/{created['id']}/build",
        json={
            "recipe": {
                "seed": 23,
                "steps": [{"kind": "split", "ratios": {"train": 1.0}}],
            }
        },
    )
    assert build.status_code == 202, build.text
    build_id = build.json()["id"]
    for _ in range(100):
        build_job = http.get(f"/api/public/dataset-jobs/{build_id}").json()
        if build_job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert build_job["status"] == "completed", build_job
    version_id = build_job["version_id"]

    response = http.post(
        f"/api/public/dataset-versions/{version_id}/training-artifacts",
        json={
            "adapter_id": "sft",
            "trainer_mode": "sft",
            "seed": 23,
            "validation_fraction": 0.34,
            "bindings": [
                {
                    "role": "train",
                    "dataset_version_id": version_id,
                    "split": "train",
                }
            ],
        },
    )
    assert response.status_code == 202, response.text
    started = response.json()
    assert started["job_type"] == "training_artifact"
    assert started["id"]
    assert started["version_id"] == version_id

    job_id = started["id"]
    for _ in range(100):
        job_response = http.get(f"/api/public/dataset-jobs/{job_id}")
        assert job_response.status_code == 200, job_response.text
        job = job_response.json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert job["status"] == "completed", job
    assert job["training_artifact_id"]
    assert job["training_artifact"]["status"] == "ready"
    artifact_id = job["training_artifact_id"]

    artifact = http.get(f"/api/public/training-artifacts/{artifact_id}")
    assert artifact.status_code == 200, artifact.text
    artifact_data = artifact.json()
    assert artifact_data["artifact_id"] == artifact_id
    assert artifact_data["dataset_version_id"] == version_id
    assert set(artifact_data["paths"]) == {"train", "validation"}

    listed = http.get(
        f"/api/public/dataset-versions/{version_id}/training-artifacts"
    ).json()["items"]
    assert [item["artifact_id"] for item in listed] == [artifact_id]
    version = http.get(f"/api/public/dataset-versions/{version_id}").json()
    assert [item["artifact_id"] for item in version["training_artifacts"]] == [
        artifact_id
    ]


def test_dataset_api_reports_invalid_and_missing_resources(client):
    http, root = client
    created = _register(http, root)
    invalid = http.post(
        f"/api/public/datasets/{created['id']}/build",
        json={"recipe": {"steps": []}},
    )
    assert invalid.status_code == 400
    assert http.get("/api/public/datasets/missing").status_code == 404
    assert http.get("/api/public/dataset-jobs/missing").status_code == 404


def test_document_extraction_api_is_durable_bounded_and_verifiable(client):
    http, root = client
    launched = []
    for index in range(2):
        source = root / f"manual-{index}.txt"
        source.write_text(
            f"Reviewed manual {index}\n\nA distinct corpus paragraph.",
            encoding="utf-8",
        )
        response = http.post(
            "/api/public/document-extractions",
            json={"path": str(source)},
        )
        assert response.status_code == 202, response.text
        value = response.json()
        assert value["extraction"]["id"]
        assert value["work_item_id"]
        launched.append(value)

    for value in launched:
        extraction_id = value["extraction"]["id"]
        for _ in range(300):
            extraction = http.get(
                f"/api/public/document-extractions/{extraction_id}"
            ).json()
            if extraction["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        assert extraction["status"] == "completed", extraction

    page = http.get("/api/public/document-extractions?limit=1").json()
    assert len(page["items"]) == 1
    assert page["total"] == 2
    assert page["limit"] == 1
    assert page["offset"] == 0

    extraction_id = launched[0]["extraction"]["id"]
    preview = http.post(
        f"/api/public/document-extractions/{extraction_id}/preview",
        json={"limit": 1, "include_text": True},
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["total"] == 1
    assert "Reviewed manual" in preview.json()["items"][0]["text"]
    verification = http.post(
        f"/api/public/document-extractions/{extraction_id}/verify"
    )
    assert verification.status_code == 200, verification.text
    assert verification.json()["valid"] is True
    assert http.get("/api/public/dataset-versions/missing").status_code == 404


def test_local_multimodal_preview_uses_authenticated_asset_url(client):
    http, root = client
    image = root / "pixel.png"
    image.write_bytes(b"not-a-real-png-but-an-asset")
    source = root / "vlm.jsonl"
    source.write_text(
        '{"image":"pixel.png","prompt":"what?","response":"pixel"}\n',
        encoding="utf-8",
    )
    created = http.post(
        "/api/public/datasets",
        json={
            "name": "vision",
            "canonical_schema": "vlm",
            "source": {"kind": "local", "uri": str(source)},
        },
    ).json()
    row = http.get(f"/api/public/datasets/{created['id']}/preview").json()["items"][0]
    assert row["image"].startswith("/api/public/dataset-source-assets/")
    assert row["_halo_forge_assets"]["image"]["reference"] == "pixel.png"
    asset = http.get(row["image"])
    assert asset.status_code == 200
    assert asset.content == image.read_bytes()


def test_closed_loop_evaluation_preview_and_failure_child_job(client):
    http, root = client
    created = _register(http, root)
    build = http.post(
        f"/api/public/datasets/{created['id']}/build",
        json={
            "recipe": {
                "seed": 7,
                "steps": [
                    {"kind": "split", "ratios": {"train": 0.67, "test": 0.33}}
                ],
            }
        },
    ).json()
    for _ in range(100):
        parent_job = http.get(f"/api/public/dataset-jobs/{build['id']}").json()
        if parent_job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert parent_job["status"] == "completed"
    parent_version_id = parent_job["version_id"]

    suite_response = http.post(
        "/api/public/benchmark-suites",
        json={
            "name": "feedback-loop",
            "primary_metric": "score",
            "direction": "maximize",
            "items": [
                {
                    "id": "failure-1",
                    "record_id": "external-failure-1",
                    "input": "repair this",
                    "expected": "repaired",
                    "score_by_subject": {"base-model": 1.0, "candidate-model": 0.0},
                }
            ],
        },
    )
    assert suite_response.status_code == 201, suite_response.text
    suite_payload = suite_response.json()
    revision_id = suite_payload["latest_revision"]["id"]
    suite_detail = http.get(
        f"/api/public/benchmark-suites/{suite_payload['id']}"
    ).json()
    assert [value["id"] for value in suite_detail["revisions"]] == [revision_id]
    assert http.get(
        f"/api/public/benchmark-suite-revisions/{revision_id}"
    ).json()["content_hash"] == suite_detail["revisions"][0]["content_hash"]

    evaluation_ids = []
    for subject in ("base-model", "candidate-model"):
        launched = http.post(
            "/api/public/evaluations",
            json={
                "suite_revision_id": revision_id,
                "subject": {"kind": "model", "value": subject},
            },
        )
        assert launched.status_code == 202, launched.text
        evaluation_id = launched.json()["id"]
        evaluation_ids.append(evaluation_id)
        for _ in range(100):
            evaluation = http.get(f"/api/public/evaluations/{evaluation_id}").json()
            if evaluation["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert evaluation["status"] == "completed", evaluation

    base_id, candidate_id = evaluation_ids
    comparison = http.get(
        f"/api/public/evaluations/compare?base_id={base_id}&candidate_id={candidate_id}"
    )
    assert comparison.status_code == 200, comparison.text
    assert comparison.json()["counts"]["regression"] == 1

    preview = http.post(
        "/api/public/evaluation-mining/preview",
        json={
            "base_id": base_id,
            "candidate_id": candidate_id,
            "selector": {"kind": "regression"},
            "excluded_record_ids": [],
        },
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["total"] == 1
    assert preview.json()["items"][0]["classification"] == "regression"

    mined = http.post(
        "/api/public/evaluation-mining/build",
        json={
            "dataset_id": created["id"],
            "parent_version_id": parent_version_id,
            "base_id": base_id,
            "candidate_id": candidate_id,
            "selector": {"kind": "regression"},
            "excluded_record_ids": [],
        },
    )
    assert mined.status_code == 202, mined.text
    mined_job_id = mined.json()["id"]
    for _ in range(100):
        mined_job = http.get(f"/api/public/dataset-jobs/{mined_job_id}").json()
        if mined_job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)
    assert mined_job["status"] == "completed", mined_job
    child = http.get(
        f"/api/public/dataset-versions/{mined_job['version_id']}"
    ).json()
    assert child["parent_version_id"] == parent_version_id
    assert child["row_count"] == 4
    assert child["provenance"]["steps"][0]["details"]["evaluation_ids"] == [
        base_id,
        candidate_id,
    ]


def test_guided_own_data_example_to_immutable_version(client, monkeypatch):
    http, _ = client

    # This walkthrough asserts capability outcomes (`available`, `ready`)
    # end-to-end, so it needs a runtime with every trainer dependency present.
    # Pin the host-dependent seams instead of inheriting whatever the runner
    # happens to have installed.
    from halo_forge.own_data import service as own_data_service
    from halo_forge.public_api.service import PublicApiService

    monkeypatch.setattr(own_data_service, "_module_available", lambda _name: True)
    monkeypatch.setattr(PublicApiService, "_active_backend_name", lambda self: "cpu")

    capabilities = http.get("/api/public/interface-capabilities")
    assert capabilities.status_code == 200, capabilities.text
    capability_items = capabilities.json()["items"]
    assert any(item["id"] == "browser-local" for item in capability_items)
    assert any(item["id"] == "cli" for item in capability_items)

    scenarios = http.get(
        "/api/public/training-scenarios?include_unavailable=true"
    )
    assert scenarios.status_code == 200, scenarios.text
    by_id = {item["id"]: item for item in scenarios.json()["items"]}
    assert by_id["instruction-sft"]["revision_id"] == "instruction-sft@1"
    assert by_id["audio-asr"]["task_type"] == "automatic_speech_recognition"
    # `available` is a runtime probe of the host, not a static contract; it is
    # deterministic here only because the probe is pinned above. Both branches
    # are covered by
    # `test_training_scenario_availability_contract_holds_in_both_branches`.
    _assert_scenario_availability_contract(by_id.values())
    assert by_id["audio-classification"]["available"] is True
    assert by_id["audio-classification"]["trainer_modes"] == ["classify"]
    # audio-tts is unavailable in the registry itself (no verified trainer
    # contract exists on any host), so this one *is* a static contract.
    assert by_id["audio-tts"]["available"] is False
    assert by_id["audio-tts"]["unavailable_reason"]

    created = http.post(
        "/api/public/dataset-imports",
        json={
            "source_kind": "example",
            "scenario_revision_id": "instruction-sft@1",
            "example_id": "instruction-text",
            "name": "Working example",
        },
    )
    assert created.status_code == 201, created.text
    import_session = created.json()
    assert import_session["status"] == "ready"
    assert import_session["files"][0]["status"] == "verified"

    launched = http.post(
        f"/api/public/dataset-imports/{import_session['id']}/inspect",
        json={"scenario_revision_id": "instruction-sft@1"},
    )
    assert launched.status_code == 202, launched.text
    inspection_id = launched.json()["inspection"]["id"]
    inspection = None
    for _ in range(200):
        response = http.get(f"/api/public/dataset-inspections/{inspection_id}")
        assert response.status_code == 200, response.text
        inspection = response.json()
        if inspection["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert inspection["status"] == "completed", inspection
    assert inspection["row_count"] == 2
    candidate = next(
        item
        for item in inspection["schema_candidates"]
        if item["scenario_id"] == "instruction-sft"
    )
    mapping_plan = {
        "version": 2,
        "scenario_revision_id": candidate["scenario_revision_id"],
        "mappings": candidate["suggested_mapping"],
        "confirmed": True,
    }
    mapped = http.post(
        f"/api/public/dataset-inspections/{inspection_id}/mapping-preview",
        json={"mapping_plan": mapping_plan},
    )
    assert mapped.status_code == 200, mapped.text
    assert mapped.json()["ready"] is True
    assert mapped.json()["items"][0]["canonical"]["prompt"]

    prepared = http.post(
        f"/api/public/dataset-inspections/{inspection_id}/preparation-preview",
        json={"mapping_plan": mapping_plan},
    )
    assert prepared.status_code == 200, prepared.text
    preparation = prepared.json()
    assert preparation["sampled"] is True
    assert [step["kind"] for step in preparation["recipe"]["steps"]] == [
        "map",
        "normalize",
        "validate",
        "dedup",
        "split",
        "contamination",
    ]

    registered = http.post(
        f"/api/public/dataset-inspections/{inspection_id}/register",
        json={
            "name": "Guided example",
            "scenario_revision_id": "instruction-sft@1",
            "mapping_plan": mapping_plan,
            "preparation_plan": preparation,
        },
    )
    dataset = _await_guided_registration(http, registered)
    assert dataset["canonical_schema"] == "sft"
    assert dataset["sources"][0]["metadata"]["guided_own_data"][
        "scenario_revision_id"
    ] == "instruction-sft@1"

    built = http.post(
        f"/api/public/datasets/{dataset['id']}/build",
        json={"recipe": preparation["recipe"]},
    )
    assert built.status_code == 202, built.text
    job_id = built.json()["id"]
    job = None
    for _ in range(200):
        response = http.get(f"/api/public/dataset-jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert job["status"] == "completed", job
    version = http.get(
        f"/api/public/dataset-versions/{job['version_id']}"
    ).json()
    assert version["split_counts"]["train"] > 0
    assert version["content_hash"]
    readiness = http.get(
        f"/api/public/dataset-versions/{version['id']}/readiness?trainer_mode=sft"
    )
    assert readiness.status_code == 200, readiness.text
    assert readiness.json()["ready"] is True
    assert readiness.json()["recommended_model"]["id"]


@pytest.mark.parametrize(
    ("scenario_revision_id", "scenario_id", "schema"),
    [
        ("vlm-captioning@1", "vlm-captioning", "vlm"),
        ("audio-asr@1", "audio-asr", "audio"),
    ],
)
def test_multimodal_working_example_builds_without_manual_asset_copy(
    client,
    monkeypatch,
    scenario_revision_id: str,
    scenario_id: str,
    schema: str,
):
    # Fixture publication is a data-path contract. Simulate the optional
    # modality dependencies so this test remains independent of which extras
    # are installed on the machine running the suite; runtime filtering is
    # covered separately by the guided capability tests.
    monkeypatch.setattr("halo_forge.own_data.service._module_available", lambda _name: True)
    http, _ = client
    created = http.post(
        "/api/public/dataset-imports",
        json={
            "source_kind": "example",
            "scenario_revision_id": scenario_revision_id,
            "name": f"{scenario_id} working example",
        },
    )
    assert created.status_code == 201, created.text
    session = created.json()
    assert session["total_files"] == 2

    launched = http.post(
        f"/api/public/dataset-imports/{session['id']}/inspect",
        json={"scenario_revision_id": scenario_revision_id},
    )
    assert launched.status_code == 202, launched.text
    inspection_id = launched.json()["inspection"]["id"]
    for _ in range(200):
        inspection = http.get(
            f"/api/public/dataset-inspections/{inspection_id}"
        ).json()
        if inspection["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert inspection["status"] == "completed", inspection
    assert inspection["valid_records"] == 1
    assert inspection["media_summary"]["verified"] == 1
    candidate = next(
        item
        for item in inspection["schema_candidates"]
        if item["scenario_id"] == scenario_id
    )
    mapping_plan = {
        "version": 2,
        "scenario_revision_id": scenario_revision_id,
        "mappings": candidate["suggested_mapping"],
        "confirmed": True,
    }
    prepared = http.post(
        f"/api/public/dataset-inspections/{inspection_id}/preparation-preview",
        json={"mapping_plan": mapping_plan},
    )
    assert prepared.status_code == 200, prepared.text
    preparation = prepared.json()
    registered = http.post(
        f"/api/public/dataset-inspections/{inspection_id}/register",
        json={
            "name": f"{scenario_id} fixture",
            "scenario_revision_id": scenario_revision_id,
            "mapping_plan": mapping_plan,
            "preparation_plan": preparation,
        },
    )
    dataset = _await_guided_registration(http, registered)
    assert dataset["canonical_schema"] == schema
    built = http.post(
        f"/api/public/datasets/{dataset['id']}/build",
        json={"recipe": preparation["recipe"]},
    )
    assert built.status_code == 202, built.text
    job_id = built.json()["id"]
    for _ in range(200):
        job = http.get(f"/api/public/dataset-jobs/{job_id}").json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert job["status"] == "completed", job
    version = http.get(
        f"/api/public/dataset-versions/{job['version_id']}"
    ).json()
    assert version["split_counts"]["train"] == 1
    assert version["row_count"] == 1
