from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from halo_forge.artifact_lab import ArtifactOperationService, ArtifactStore
from halo_forge.artifact_studio import (
    ArtifactStudioError,
    ArtifactStudioService,
    PromotionBlocked,
    UnsupportedArtifactCapability,
)
from halo_forge.run_db import LabV4Catalog, RunDatabase
from halo_forge.workstation_jobs import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
    WorkstationScheduler,
)
from halo_forge.workstation_jobs.worker import WorkstationWorker


GIB = 1024**3


def _available_capacity(_path):
    return WorkstationCapacity(
        sampled_at=datetime.now(timezone.utc),
        disk=DiskCapacity(
            path="/tmp/halo-forge-tests",
            total_bytes=200 * GIB,
            used_bytes=100 * GIB,
            free_bytes=100 * GIB,
        ),
        memory=MemoryCapacity(
            total_bytes=32 * GIB,
            used_bytes=8 * GIB,
            available_bytes=24 * GIB,
        ),
    )


def _raw_model(path: Path, value: str = "weights") -> Path:
    path.mkdir(parents=True)
    (path / "weights.bin").write_text(value, encoding="utf-8")
    return path


def _hf_output(path: Path, value: str = "converted") -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_text(value, encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")


@pytest.fixture
def studio(tmp_path: Path):
    database = RunDatabase(str(tmp_path / "runs.db"))
    store = ArtifactStore(tmp_path / "artifacts")
    catalog = LabV4Catalog(database)
    scheduler = WorkstationScheduler(
        database,
        worker_id="studio-test-worker",
        capacity_probe=_available_capacity,
    )

    def convert_engine(spec, inputs, output):
        _hf_output(output, f"converted:{inputs[0].name}")
        return {"engine": "test-converter"}

    operations = ArtifactOperationService(
        store,
        engines={
            "convert": convert_engine,
            "quantize": convert_engine,
        },
    )

    def qualification_executor(qualification, launch_spec):
        return {
            "decision": "pass",
            "reasons": [],
            "metrics": {"quality": 0.9, "tokens_per_second": 42.0},
            "decision_evidence": {
                "development": {"status": "pass"},
                "operational": {"status": "pass"},
                "holdout": {"status": "not_required"},
            },
        }

    service = ArtifactStudioService(
        database,
        store=store,
        catalog=catalog,
        scheduler=scheduler,
        operation_service=operations,
        qualification_executor=qualification_executor,
    )
    try:
        yield service, database, store, catalog
    finally:
        database.close()


def _import(
    service: ArtifactStudioService,
    path: Path,
    *,
    managed: bool = False,
    value: str = "weights",
):
    return service.import_artifact(
        _raw_model(path, value),
        artifact_kind="final_model",
        artifact_format="raw",
        model_id="local/test-model",
        backend="local",
        managed=managed,
    )


def _qualification_profile(database: RunDatabase, catalog: LabV4Catalog):
    development = database.create_benchmark_suite(name="Development", purpose="development")
    operational = database.create_benchmark_suite(name="Operational", purpose="operational")
    development_revision = database.create_benchmark_suite_revision(
        suite_id=development.id,
        content_hash="dev-hash",
        items=[{"id": "dev-item", "input": "hello"}],
        primary_metric="quality",
        direction="maximize",
    )
    operational_revision = database.create_benchmark_suite_revision(
        suite_id=operational.id,
        content_hash="ops-hash",
        items=[{"id": "ops-item", "input": "hello"}],
        primary_metric="latency_ms",
        direction="minimize",
    )
    return catalog.create_qualification_profile_revision(
        name="Local qualification",
        content_hash="qualification-profile-hash",
        quality_suite_revision_id=development_revision.id,
        operational_suite_revision_id=operational_revision.id,
        thresholds=[
            {
                "stage": "development",
                "metric": "quality",
                "direction": "maximize",
                "pass_threshold": 0.8,
            },
            {
                "stage": "operational",
                "metric": "latency_ms",
                "direction": "minimize",
                "pass_threshold": 1000,
            },
        ],
        generation_settings={"seed": 42},
        target_backend="local",
    )


def test_import_list_show_verify_annotations_and_alias(studio, tmp_path: Path) -> None:
    service, _database, _store, _catalog = studio
    imported = _import(service, tmp_path / "run-output")
    occurrence_id = imported["occurrence"]["id"]
    content_hash = imported["blob"]["content_hash"]

    assert imported["locations"][0]["storage_mode"] == "referenced"
    assert service.show_artifact(occurrence_id)["blob"]["content_hash"] == content_hash
    assert service.show_artifact(content_hash)["occurrence"]["id"] == occurrence_id
    assert service.list_artifacts()["items"][0]["occurrence"]["id"] == occurrence_id
    assert service.verify_artifact(occurrence_id)["passed"] is True

    adopted = service.adopt_artifact(occurrence_id)
    assert {item["storage_mode"] for item in adopted["locations"]} == {
        "referenced",
        "managed",
    }

    service.pin_artifact(occurrence_id)
    service.tag_artifact(occurrence_id, ["research", "research"])
    annotated = service.tag_artifact(occurrence_id, ["candidate-source"])
    annotated = service.update_annotations(occurrence_id, notes="Useful checkpoint")
    assert annotated["occurrence"]["pinned"] is True
    assert annotated["occurrence"]["tags"] == ["candidate-source", "research"]
    service.set_alias("favorite", occurrence_id)
    assert service.show_artifact("favorite")["occurrence"]["id"] == occurrence_id


def test_constructor_infers_database_from_transport_supplied_catalog(tmp_path: Path) -> None:
    database = RunDatabase(str(tmp_path / "runs.db"))
    try:
        store = ArtifactStore(tmp_path / "artifacts")
        catalog = LabV4Catalog(database)
        service = ArtifactStudioService(
            store=store,
            catalog=catalog,
            scheduler=WorkstationScheduler(
                database,
                worker_id="transport-worker",
                capacity_probe=_available_capacity,
            ),
            operation_service=ArtifactOperationService(store, engines={}),
        )
        assert service.database is database
    finally:
        database.close()


def test_queue_convert_returns_both_ids_and_rejects_unsupported_capabilities(
    studio, tmp_path: Path
) -> None:
    service, database, _store, catalog = studio
    occurrence_id = _import(service, tmp_path / "source")["occurrence"]["id"]

    before = len(database.list_work_items())
    with pytest.raises(UnsupportedArtifactCapability, match="No verified.*onnx"):
        service.queue_convert(
            occurrence_id=occurrence_id,
            target_format="onnx",
            quantization="fp16",
        )
    assert len(database.list_work_items()) == before

    receipt = service.queue_convert(
        occurrence_id=occurrence_id,
        target_format="hf",
        quantization="fp16",
    )
    assert receipt.domain_id
    assert receipt.work_item_id
    operation = catalog.get_operation(receipt.domain_id)
    work = database.get_work_item(receipt.work_item_id)
    assert operation is not None and operation.work_item_id == work.id
    assert work.launch_spec["domain_id"] == operation.id
    assert work.domain_kind == "artifact_operation"
    assert work.domain_id == operation.id
    assert work.resource_requirements == work.launch_spec["resource_requirements"]
    assert work.resource_requirements["exclusive_heavy_operation"] is True
    assert work.status == "queued"

    duplicate = service.queue_convert(
        occurrence_id=occurrence_id,
        target_format="hf",
        quantization="fp16",
    )
    assert duplicate.reused is True
    assert duplicate.domain_id == receipt.domain_id
    assert duplicate.work_item_id == receipt.work_item_id
    assert service.get_operation(receipt.domain_id)["work_item"]["id"] == receipt.work_item_id
    assert service.list_operations(status="queued")["items"][0]["id"] == receipt.domain_id

    with pytest.raises(UnsupportedArtifactCapability, match="dtype conversion"):
        service.queue_convert(
            occurrence_id=occurrence_id,
            target_format="hf",
            quantization="q4",
        )


def test_execute_work_item_publishes_catalog_output_and_lineage(studio, tmp_path: Path) -> None:
    service, database, store, catalog = studio
    source = _import(service, tmp_path / "source")
    receipt = service.queue_convert(
        occurrence_id=source["occurrence"]["id"],
        target_format="hf",
        quantization="fp16",
    )
    result = service.execute_work_item(receipt.work_item_id)

    finished = database.get_work_item(receipt.work_item_id)
    operation = catalog.get_operation(receipt.domain_id)
    assert finished.status == "completed"
    assert operation.status == "completed"
    output_id = result["result"]["output_occurrence_id"]
    output = service.show_artifact(output_id)
    assert output["occurrence"]["artifact_kind"] == "converted_model"
    assert output["blob"]["format"] == "hf"
    assert Path(output["locations"][0]["path"]).is_dir()
    assert store.verify(output["blob"]["content_hash"]).passed is True

    lineage = service.lineage(output_id)
    assert lineage["catalog"]["parents"][0]["parent_occurrence_id"] == source["occurrence"]["id"]
    assert lineage["content"]["edges"][0]["relationship"] == "convert"

    replay = service.execute_work_item(receipt.work_item_id)
    assert replay["reused"] is True
    assert replay["result"]["output_occurrence_id"] == output_id


def test_worker_uses_the_artifact_root_persisted_by_the_queuing_service(
    tmp_path: Path,
) -> None:
    database = RunDatabase(str(tmp_path / "runs.db"))
    artifact_root = tmp_path / "custom-artifact-library"
    scheduler = WorkstationScheduler(
        database,
        worker_id="custom-root-worker",
        capacity_probe=_available_capacity,
    )
    service = ArtifactStudioService(
        database,
        artifact_root=artifact_root,
        scheduler=scheduler,
    )
    try:
        source = _import(service, tmp_path / "source")
        destination = tmp_path / "portable-export"
        receipt = service.queue_export(
            occurrence_id=source["occurrence"]["id"],
            destination=destination,
        )

        finished = WorkstationWorker(scheduler).run_once()

        assert finished is not None
        assert finished.id == receipt.work_item_id
        assert finished.status == "completed"
        assert destination.is_dir()
        operation = service.get_operation(receipt.domain_id)
        assert operation["status"] == "completed"
        assert operation["work_item"]["launch_spec"]["artifact_root"] == str(
            artifact_root.resolve()
        )
    finally:
        database.close()


def test_cleanup_protects_standalone_evaluation_and_serving_profile_references(
    studio, tmp_path: Path
) -> None:
    service, database, _store, catalog = studio
    imported = _import(service, tmp_path / "protected", managed=True)
    occurrence_id = imported["occurrence"]["id"]
    content_hash = imported["blob"]["content_hash"]
    location = imported["locations"][0]["path"]

    suite = database.create_benchmark_suite(name="Standalone", purpose="development")
    revision = database.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="standalone-suite",
        items=[{"id": "one", "input": "hello"}],
        primary_metric="quality",
        direction="maximize",
    )
    database.create_evaluation(
        evaluation_id="standalone-evaluation",
        suite_revision_id=revision.id,
        adapter_id="dataset-split",
        adapter_version="1",
        subject_type="model",
        subject_ref=location,
        subject_hash="resolved-subject-hash",
        reuse_key="standalone-reuse",
        request={"subject": {"type": "model", "ref": location, "content_hash": content_hash}},
    )
    catalog.create_serving_profile_revision(
        name="Stopped profile",
        content_hash="serving-profile-content",
        occurrence_id=occurrence_id,
        backend="local",
        endpoint_settings={"port": 8001},
        generation_settings={"seed": 42},
        resource_requirements={},
    )

    plan = service.preview_cleanup()
    protected = {item.content_hash: set(item.reasons) for item in plan.protected}

    assert content_hash in protected
    assert "evaluation_referenced" in protected[content_hash]
    assert "serving" in protected[content_hash]


def test_cleanup_protects_artifacts_pinned_by_research_decisions_and_evidence(
    studio, tmp_path: Path
) -> None:
    service, database, _store, _catalog = studio
    imported = _import(service, tmp_path / "research-checkpoint", managed=True)
    occurrence_id = imported["occurrence"]["id"]
    content_hash = imported["blob"]["content_hash"]
    snapshot = database.create_cohort_analysis_snapshot(
        request={"observations": []},
        analysis={"subjects": {"candidate": {}}},
        primary_metric="accuracy",
        direction="maximize",
    )
    decision = database.create_research_decision(
        analysis_snapshot_id=snapshot.id,
        selected_subject={"subject_id": "candidate", "occurrence_id": occurrence_id},
        rationale="Keep the exact checkpoint used by the reviewed conclusion.",
    )
    database.create_evidence_bundle(
        analysis_snapshot_id=snapshot.id,
        research_decision_id=decision.id,
        content_hash="research-evidence-content",
        storage_path=str(tmp_path / "evidence"),
        request={"artifact_id": occurrence_id},
    )

    plan = service.preview_cleanup()
    protected = {item.content_hash: set(item.reasons) for item in plan.protected}

    assert "lineage_required" in protected[content_hash]


def test_merge_queue_validates_inputs_without_loading_models(studio, tmp_path: Path) -> None:
    service, _database, _store, _catalog = studio
    first = service.import_artifact(
        _raw_model(tmp_path / "adapter-a", "a"),
        artifact_kind="adapter",
        artifact_format="raw",
    )
    second = service.import_artifact(
        _raw_model(tmp_path / "adapter-b", "b"),
        artifact_kind="adapter",
        artifact_format="raw",
    )
    with pytest.raises(ValueError, match="at least two"):
        service.queue_merge(
            input_occurrence_ids=[first["occurrence"]["id"]],
            base_model="pinned/base@revision",
        )
    receipt = service.queue_merge(
        input_occurrence_ids=[
            first["occurrence"]["id"],
            second["occurrence"]["id"],
        ],
        base_model="pinned/base@revision",
        method="ties",
        weights=[0.7, 0.3],
    )
    assert receipt.status == "queued"
    assert service.get_operation(receipt.domain_id)["resolved_spec"]["output_kind"] == "adapter"


def test_qualification_queue_execution_and_promotion(studio, tmp_path: Path) -> None:
    service, database, _store, catalog = studio
    artifact = _import(service, tmp_path / "candidate")
    occurrence_id = artifact["occurrence"]["id"]
    profile = _qualification_profile(database, catalog)

    with pytest.raises(PromotionBlocked, match="no completed qualification"):
        service.promote(occurrence_id, "candidate")
    receipt = service.queue_qualification(
        occurrence_id=occurrence_id,
        profile_revision_id=profile.id,
        execution_request={"evaluate": ["development", "operational"]},
    )
    executed = service.execute_work_item(receipt.work_item_id)
    qualification = executed["result"]["qualification"]
    assert qualification["decision"] == "pass"
    assert qualification["metrics"]["decision"]["operational"]["status"] == "pass"

    promotion = service.promote(occurrence_id, "candidate")
    assert promotion["alias"] == "candidate"
    assert promotion["overridden"] is False
    assert "candidate" in service.show_artifact(occurrence_id)["aliases"]


def test_promotion_override_requires_and_records_a_note(studio, tmp_path: Path) -> None:
    service, _database, _store, catalog = studio
    artifact = _import(service, tmp_path / "untested")
    occurrence_id = artifact["occurrence"]["id"]
    with pytest.raises(PromotionBlocked):
        service.promote(occurrence_id, "approved")
    override = service.promote(
        occurrence_id,
        "approved",
        override_note="Reviewed for a local diagnostic deployment only",
    )
    assert override["overridden"] is True
    event = catalog._conn.execute(
        "SELECT override_reason FROM artifact_alias_events WHERE alias = 'approved'"
    ).fetchone()
    assert event["override_reason"] == "Reviewed for a local diagnostic deployment only"


def test_serving_is_reserved_not_claimed_as_started(studio, tmp_path: Path) -> None:
    service, _database, _store, _catalog = studio
    artifact = _import(service, tmp_path / "serve-model")
    reservation = service.reserve_serving(
        artifact["occurrence"]["id"],
        name="Local serving",
        backend="local",
        endpoint_settings={"port": 8000},
        generation_settings={"temperature": 0.0},
        resource_requirements={"accelerator": True},
        serving_id="serve-test",
    )
    assert reservation.state == "reserved"
    assert reservation.lease is not None
    heartbeat = service.heartbeat_serving(
        "serve-test",
        process_id=123,
        process_started_at=456.0,
        metadata={"state": "server_started"},
    )
    assert heartbeat["holder_pid"] == 123
    assert heartbeat["metadata"]["state"] == "server_started"
    assert service.release_serving("serve-test") is True


def test_export_registers_bundle_occurrence_and_parent_lineage(studio, tmp_path: Path) -> None:
    service, _database, _store, _catalog = studio
    artifact = _import(service, tmp_path / "export-model", managed=True)
    exported = service.export_artifact(
        artifact["occurrence"]["id"],
        tmp_path / "portable",
        replay_identity={"run_id": "run-1"},
        dataset_identity={"version_id": "dataset-v1"},
        license_metadata={"license": "Apache-2.0"},
    )
    output = exported["artifact"]
    assert output["occurrence"]["artifact_kind"] == "export_bundle"
    assert Path(exported["bundle"]["path"], "bundle-manifest.json").is_file()
    lineage = service.lineage(output["occurrence"]["id"])
    assert lineage["catalog"]["parents"][0]["relation"] == "export"


def test_queued_export_has_durable_operation_link_and_reuses_completed_work(
    studio, tmp_path: Path
) -> None:
    service, database, _store, catalog = studio
    artifact = _import(service, tmp_path / "queued-export-model", managed=True)
    destination = tmp_path / "queued-portable"
    receipt = service.queue_export(
        occurrence_id=artifact["occurrence"]["id"],
        destination=destination,
        replay_identity={"run_id": "run-export"},
        dataset_identity={"version_id": "dataset-export"},
        license_metadata={"license": "Apache-2.0"},
    )
    work = database.get_work_item(receipt.work_item_id)
    operation = catalog.get_operation(receipt.domain_id)
    assert receipt.domain_kind == "artifact_operation"
    assert work.domain_kind == receipt.domain_kind
    assert work.domain_id == receipt.domain_id
    assert work.resource_requirements["projected_disk_bytes"] > artifact["blob"]["size_bytes"]
    assert Path(work.resource_requirements["output_path"]).exists()
    assert operation.operation_type == "export"

    executed = service.execute_work_item(receipt.work_item_id)
    output_id = executed["result"]["output_occurrence_id"]
    assert database.get_work_item(receipt.work_item_id).status == "completed"
    assert catalog.get_operation(receipt.domain_id).output_occurrence_id == output_id
    assert (destination / "bundle-manifest.json").is_file()
    parent = service.lineage(output_id)["catalog"]["parents"][0]
    assert parent["relation"] == "export"
    assert parent["operation_id"] == receipt.domain_id

    duplicate = service.queue_export(
        occurrence_id=artifact["occurrence"]["id"],
        destination=destination,
        replay_identity={"run_id": "run-export"},
        dataset_identity={"version_id": "dataset-export"},
        license_metadata={"license": "Apache-2.0"},
    )
    assert duplicate.reused is True
    assert duplicate.domain_id == receipt.domain_id
    assert duplicate.work_item_id == receipt.work_item_id


def test_cleanup_derives_pin_and_promotion_protections(studio, tmp_path: Path) -> None:
    service, _database, _store, _catalog = studio
    pinned = _import(service, tmp_path / "pinned", managed=True, value="pinned")
    promoted = _import(service, tmp_path / "promoted", managed=True, value="promoted")
    removable = _import(service, tmp_path / "removable", managed=True, value="removable")
    service.update_annotations(pinned["occurrence"]["id"], pinned=True)
    service.promote(
        promoted["occurrence"]["id"],
        "candidate",
        override_note="Reviewed experimental candidate",
    )

    plan = service.preview_cleanup()
    protected = {item.content_hash: item.reasons for item in plan.protected}
    assert "pinned" in protected[pinned["blob"]["content_hash"]]
    assert "promoted" in protected[promoted["blob"]["content_hash"]]
    assert [item.identifier for item in plan.candidates] == [removable["blob"]["content_hash"]]
    result = service.apply_cleanup(plan.id, review_note="Reviewed removable artifact")
    assert result.trashed == (removable["blob"]["content_hash"],)
    assert service.storage_inventory()["trash_bytes"] > 0


def test_queued_reviewed_cleanup_rechecks_protections_before_execution(
    studio, tmp_path: Path
) -> None:
    service, database, _store, _catalog = studio
    protected_late = _import(
        service, tmp_path / "protect-after-review", managed=True, value="protect-late"
    )
    removable = _import(service, tmp_path / "queued-removable", managed=True, value="remove")
    plan = service.preview_cleanup()

    with pytest.raises(ValueError, match="review_note"):
        service.queue_cleanup(plan.id, review_note="")
    receipt = service.queue_cleanup(plan.id, review_note="Operator reviewed both candidates")
    duplicate = service.queue_reviewed_cleanup(
        plan.id, review_note="Operator reviewed both candidates"
    )
    assert duplicate.reused is True
    assert duplicate.work_item_id == receipt.work_item_id
    work = database.get_work_item(receipt.work_item_id)
    assert receipt.domain_kind == "artifact_cleanup"
    assert work.domain_kind == "artifact_cleanup"
    assert work.domain_id == plan.id
    assert work.resource_requirements["estimated_reclaimable_bytes"] == plan.reclaimable_bytes
    assert "input_occurrence_ids" not in work.launch_spec

    # This post-review pin must win over the stale preview at execution time.
    service.pin_artifact(protected_late["occurrence"]["id"])
    executed = service.execute_work_item(receipt.work_item_id)
    cleanup = executed["result"]["cleanup"]
    assert removable["blob"]["content_hash"] in cleanup["trashed"]
    assert cleanup["skipped"][protected_late["blob"]["content_hash"]] == "protected: pinned"
    assert database.get_work_item(receipt.work_item_id).status == "completed"


def test_queued_serving_can_reserve_without_claiming_a_server_start(studio, tmp_path: Path) -> None:
    service, database, _store, _catalog = studio
    artifact = _import(service, tmp_path / "queued-reservation")
    receipt = service.queue_serving(
        artifact["occurrence"]["id"],
        name="Queued reservation",
        backend="local",
        serving_id="queued-reserved",
        start_process=False,
        resource_requirements={"accelerator_memory_bytes": 1024},
    )
    work = database.get_work_item(receipt.work_item_id)
    assert receipt.domain_kind == "artifact_serving"
    assert work.domain_kind == "artifact_serving"
    assert work.domain_id == receipt.domain_id
    assert work.resource_class == "none"
    assert work.resource_requirements["lease_type"] == "serving"

    executed = service.execute_work_item(receipt.work_item_id)
    assert executed["result"]["state"] == "reserved"
    lease = database.get_resource_lease("accelerator")
    assert lease.holder_id == "queued-reserved"
    assert lease.metadata["state"] == "reserved"
    assert service.release_serving("queued-reserved") is True


def test_queued_serving_waits_without_overlapping_an_existing_lease(studio, tmp_path: Path) -> None:
    service, database, _store, _catalog = studio
    artifact = _import(service, tmp_path / "serving-waits")
    current = service.reserve_serving(
        artifact["occurrence"]["id"],
        name="Current server",
        backend="local",
        serving_id="current-server",
    )
    assert current.state == "reserved"
    queued = service.queue_serving(
        artifact["occurrence"]["id"],
        name="Next server",
        backend="local",
        serving_id="next-server",
        start_process=False,
    )
    with pytest.raises(ArtifactStudioError, match="remains queued"):
        service.execute_work_item(queued.work_item_id)
    assert database.get_work_item(queued.work_item_id).status == "queued"

    assert service.release_serving("current-server") is True
    executed = service.execute_work_item(queued.work_item_id)
    assert executed["result"]["state"] == "reserved"
    assert database.get_resource_lease("accelerator").holder_id == "next-server"
    assert service.release_serving("next-server") is True


def test_queued_serving_start_requires_and_records_real_process_identity(
    studio, tmp_path: Path
) -> None:
    service, database, _store, _catalog = studio
    artifact = _import(service, tmp_path / "queued-serving-start")

    def serving_starter(profile, occurrence, launch_spec):
        assert profile["occurrence_id"] == occurrence.id
        assert launch_spec["serving_id"] == "managed-start"
        return {
            "state": "serving",
            "process_id": 43210,
            "process_started_at": 9876.5,
            "endpoint": "http://127.0.0.1:9000",
        }

    service.serving_starter = serving_starter
    service.scheduler.process_probe = lambda process_id, started_at: (
        process_id == 43210 and started_at == 9876.5
    )
    receipt = service.queue_managed_serving(
        artifact["occurrence"]["id"],
        name="Managed start",
        backend="local",
        serving_id="managed-start",
    )
    executed = service.execute_work_item(receipt.work_item_id)
    assert executed["result"]["state"] == "serving"
    assert executed["result"]["process_identity"] == {
        "process_id": 43210,
        "process_started_at": 9876.5,
    }
    lease = database.get_resource_lease("accelerator")
    assert lease.holder_id == "managed-start"
    assert lease.holder_pid == 43210
    assert lease.holder_pid_started_at == 9876.5
    assert lease.metadata["state"] == "serving"
    assert service.release_serving("managed-start") is True


def test_missing_managed_serving_launcher_fails_without_stale_lease(studio, tmp_path: Path) -> None:
    service, database, _store, _catalog = studio
    artifact = _import(service, tmp_path / "missing-serving-launcher")
    receipt = service.queue_serving(
        artifact["occurrence"]["id"],
        name="Unavailable launcher",
        backend="local",
        serving_id="missing-launcher",
    )
    with pytest.raises(UnsupportedArtifactCapability, match="real process identity"):
        service.execute_work_item(receipt.work_item_id)
    assert database.get_work_item(receipt.work_item_id).status == "failed"
    assert database.get_resource_lease("accelerator") is None


def test_missing_qualification_executor_fails_durable_work_truthfully(
    tmp_path: Path,
) -> None:
    database = RunDatabase(str(tmp_path / "runs.db"))
    store = ArtifactStore(tmp_path / "artifacts")
    catalog = LabV4Catalog(database)
    service = ArtifactStudioService(
        database,
        store=store,
        catalog=catalog,
        scheduler=WorkstationScheduler(
            database,
            worker_id="no-evaluator",
            capacity_probe=_available_capacity,
        ),
        operation_service=ArtifactOperationService(store, engines={}),
    )
    try:
        artifact = _import(service, tmp_path / "artifact")
        profile = _qualification_profile(database, catalog)
        receipt = service.queue_qualification(
            occurrence_id=artifact["occurrence"]["id"],
            profile_revision_id=profile.id,
        )
        with pytest.raises(UnsupportedArtifactCapability, match="evidence-producing"):
            service.execute_work_item(receipt.work_item_id)
        assert database.get_work_item(receipt.work_item_id).status == "failed"
        assert catalog.get_qualification(receipt.domain_id).status == "failed"
    finally:
        database.close()
