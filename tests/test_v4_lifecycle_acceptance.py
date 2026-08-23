from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from halo_forge.artifact_lab import ArtifactOperationService, ArtifactStore
from halo_forge.artifact_studio import ArtifactStudioService
from halo_forge.evaluation_lab import (
    EvaluationAdapter,
    EvaluationAdapterResult,
    EvaluationLabService,
    EvaluationMetric,
    EvaluationSample,
    default_adapter_registry,
)
from halo_forge.qualification_lab import (
    PerformanceSettings,
    QualificationMetricRule,
    QualificationProfileRevision,
)
from halo_forge.qualification_lab.executor import EvaluationQualificationExecutor
from halo_forge.run_db import LabV4Catalog, RunDatabase
from halo_forge.workstation_jobs import (
    DiskCapacity,
    MemoryCapacity,
    WorkstationCapacity,
    WorkstationScheduler,
)


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


def _raw_artifact(path: Path, value: str) -> Path:
    path.mkdir(parents=True)
    (path / "weights.bin").write_text(value, encoding="utf-8")
    return path


def _hf_artifact(path: Path, value: str) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_text(value, encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}\n", encoding="utf-8")


class _FixturePerformanceAdapter(EvaluationAdapter):
    """Deterministic stand-in for the external inference runtime only."""

    adapter_id = "performance"
    adapter_version = "fixture-1"

    def __init__(self, latency_by_hash: Mapping[str, float]):
        self.latency_by_hash = latency_by_hash

    def evaluate(self, context, revision, subject, request) -> EvaluationAdapterResult:
        settings = dict(request["performance_settings"])
        assert settings == {
            "warmup_runs": 2,
            "measured_repeats": 5,
            "concurrency": 1,
            "generation_seed": 17,
        }
        content_hash = str(subject.payload["content_hash"])
        latency_ms = float(self.latency_by_hash[content_hash])
        total = settings["warmup_runs"] + settings["measured_repeats"]
        samples = []
        context.progress(stage="fixture-performance", processed=0, total=total)
        for index in range(total):
            phase = "warmup" if index < settings["warmup_runs"] else "measure"
            samples.append(
                EvaluationSample(
                    suite_item_id=f"performance-{phase}-{index}",
                    record_id=f"performance-{phase}-{index}",
                    input={"phase": phase, "seed": settings["generation_seed"]},
                    output={"total_latency_ms": latency_ms},
                    score=latency_ms,
                    passed=True,
                    latency_ms=latency_ms,
                    evidence_kind="operational_measurement",
                    valid=True,
                    mineable=False,
                    generation_seed=settings["generation_seed"],
                    score_direction="minimize",
                    runtime_versions={"fixture-runtime": "1"},
                )
            )
            context.progress(processed=index + 1, total=total)
        return EvaluationAdapterResult(
            metrics=[
                EvaluationMetric(
                    name=revision.primary_metric,
                    value=latency_ms,
                    direction="minimize",
                )
            ],
            samples=samples,
            summary={"fixture": True, "warmups": 2, "measured_repeats": 5},
        )


def test_v4_artifact_lifecycle_from_adapter_to_approved_bundle(tmp_path: Path) -> None:
    """Exercise the v4 workstation loop through durable, public service seams."""

    database_path = tmp_path / "runs.db"
    database = RunDatabase(str(database_path))
    store = ArtifactStore(tmp_path / "artifact-library")
    catalog = LabV4Catalog(database)
    scheduler = WorkstationScheduler(
        database,
        worker_id="v4-lifecycle-worker",
        capacity_probe=_available_capacity,
    )

    def bake_engine(spec, inputs, output):
        assert spec.operation_type == "bake"
        assert spec.parameters["base_input_index"] == 0
        assert len(inputs) == 2
        source = "|".join((path / "weights.bin").read_text(encoding="utf-8") for path in inputs)
        _hf_artifact(output, f"baked:{source}")
        return {"runtime": "fixture-bake", "resolved_input_count": len(inputs)}

    def convert_engine(spec, inputs, output):
        assert spec.operation_type == "convert"
        assert len(inputs) == 1
        source = (inputs[0] / "model.safetensors").read_text(encoding="utf-8")
        _hf_artifact(output, f"converted:{source}")
        return {"runtime": "fixture-convert", "target_format": spec.output_format}

    operations = ArtifactOperationService(
        store,
        engines={"bake": bake_engine, "convert": convert_engine},
    )
    latency_by_hash: dict[str, float] = {}
    registry = default_adapter_registry()
    registry.register(_FixturePerformanceAdapter(latency_by_hash), replace=True)
    evaluations = EvaluationLabService(
        database,
        tmp_path / "evaluations",
        registry=registry,
    )
    qualification_executor = EvaluationQualificationExecutor(database, catalog, evaluations)
    studio = ArtifactStudioService(
        database,
        store=store,
        catalog=catalog,
        scheduler=scheduler,
        operation_service=operations,
        qualification_executor=qualification_executor,
    )

    try:
        base = studio.import_artifact(
            _raw_artifact(tmp_path / "base", "base-v1"),
            artifact_kind="final_model",
            artifact_format="raw",
            model_id="local/base",
            backend="fixture",
        )
        adapter = studio.import_artifact(
            _raw_artifact(tmp_path / "adapter", "adapter-v1"),
            artifact_kind="adapter",
            artifact_format="raw",
            model_id="local/base-adapter",
            backend="fixture",
        )
        base_id = base["occurrence"]["id"]
        adapter_id = adapter["occurrence"]["id"]

        bake = studio.queue_merge(
            input_occurrence_ids=[adapter_id],
            base_model=base_id,
            base_occurrence_id=base_id,
            mode="bake",
            method="linear",
        )
        assert bake.domain_id and bake.work_item_id
        assert database.get_work_item(bake.work_item_id).domain_id == bake.domain_id
        assert catalog.get_operation(bake.domain_id).work_item_id == bake.work_item_id
        baked_id = studio.execute_work_item(bake.work_item_id)["result"]["output_occurrence_id"]

        conversion = studio.queue_convert(
            occurrence_id=baked_id,
            target_format="hf",
            quantization="fp16",
        )
        assert conversion.domain_id and conversion.work_item_id
        assert database.get_work_item(conversion.work_item_id).domain_id == conversion.domain_id
        assert catalog.get_operation(conversion.domain_id).work_item_id == conversion.work_item_id
        candidate_id = studio.execute_work_item(conversion.work_item_id)["result"][
            "output_occurrence_id"
        ]
        candidate = studio.show_artifact(candidate_id)
        candidate_hash = candidate["blob"]["content_hash"]
        baked = studio.show_artifact(baked_id)
        baked_hash = baked["blob"]["content_hash"]

        baked_parents = studio.lineage(baked_id)["catalog"]["parents"]
        assert [parent["parent_occurrence_id"] for parent in baked_parents] == [
            base_id,
            adapter_id,
        ]
        assert [parent["relation"] for parent in baked_parents] == ["base", "bake"]
        converted_parent = studio.lineage(candidate_id)["catalog"]["parents"]
        assert converted_parent == [
            {
                **converted_parent[0],
                "parent_occurrence_id": baked_id,
                "relation": "convert",
                "operation_id": conversion.domain_id,
            }
        ]

        candidate_ref = next(
            location["path"]
            for location in candidate["locations"]
            if location["storage_mode"] == "managed"
        )
        baked_ref = next(
            location["path"]
            for location in baked["locations"]
            if location["storage_mode"] == "managed"
        )
        latency_by_hash.update({candidate_hash: 8.0, baked_hash: 10.0})

        _, development = evaluations.create_suite(
            name="Lifecycle development",
            purpose="development",
            items=[
                {
                    "id": "quality-1",
                    "input": "Name the lifecycle stage.",
                    "score_by_subject": {candidate_ref: 0.94, baked_ref: 0.90},
                }
            ],
            primary_metric="quality",
            direction="maximize",
        )
        _, operational = evaluations.create_suite(
            name="Lifecycle operational",
            purpose="operational",
            items=[{"id": "performance-1", "input": "Warm up then measure."}],
            primary_metric="total_latency_ms",
            direction="minimize",
        )
        _, holdout = evaluations.create_suite(
            name="Lifecycle holdout",
            purpose="holdout",
            items=[
                {
                    "id": "holdout-1",
                    "input": "Final confirmation.",
                    "score_by_subject": {candidate_ref: 0.92, baked_ref: 0.89},
                }
            ],
            primary_metric="quality",
            direction="maximize",
        )
        assert development is not None and operational is not None and holdout is not None

        development_rule = QualificationMetricRule(
            "quality", "maximize", pass_threshold=0.80, maximum_regression=0.05
        )
        operational_rule = QualificationMetricRule(
            "total_latency_ms", "minimize", pass_threshold=20.0, maximum_regression=2.0
        )
        holdout_rule = QualificationMetricRule(
            "quality", "maximize", pass_threshold=0.85, maximum_regression=0.05
        )
        profile_definition = QualificationProfileRevision(
            profile_id="lifecycle-profile",
            revision_number=1,
            name="Lifecycle release profile",
            development_suite_revision_id=development.id,
            operational_suite_revision_id=operational.id,
            holdout_suite_revision_id=holdout.id,
            development_rules=(development_rule,),
            operational_rules=(operational_rule,),
            holdout_rules=(holdout_rule,),
            target_backend="fixture",
            generation_settings={"seed": 17, "temperature": 0.0},
            performance_settings=PerformanceSettings(generation_seed=17),
        )
        profile = catalog.create_qualification_profile_revision(
            name=profile_definition.name,
            profile_id=profile_definition.profile_id,
            content_hash=profile_definition.content_hash,
            quality_suite_revision_id=development.id,
            operational_suite_revision_id=operational.id,
            holdout_suite_revision_id=holdout.id,
            thresholds=[
                {"stage": "development", **development_rule.to_dict()},
                {"stage": "operational", **operational_rule.to_dict()},
                {"stage": "holdout", **holdout_rule.to_dict()},
            ],
            generation_settings={
                "generation_settings": profile_definition.generation_settings.to_dict(),
                "performance_settings": profile_definition.performance_settings.to_dict(),
            },
            target_backend=profile_definition.target_backend,
        )

        qualification = studio.queue_qualification(
            occurrence_id=candidate_id,
            parent_occurrence_id=baked_id,
            profile_revision_id=profile.id,
        )
        assert qualification.domain_id and qualification.work_item_id
        assert (
            database.get_work_item(qualification.work_item_id).domain_id == qualification.domain_id
        )
        first_result = studio.execute_work_item(qualification.work_item_id)
        first = first_result["result"]["qualification"]
        assert first["decision"] == "warn"
        assert first["holdout_evaluation_id"] is None
        assert first["metrics"]["decision"]["development"]["status"] == "pass"
        assert first["metrics"]["decision"]["operational"]["status"] == "pass"
        assert first["metrics"]["decision"]["holdout"]["status"] == "warn"

        candidate_promotion = studio.promote(candidate_id, "candidate")
        assert candidate_promotion["overridden"] is False

        holdout_confirmation = studio.queue_qualification(
            occurrence_id=candidate_id,
            parent_occurrence_id=baked_id,
            profile_revision_id=profile.id,
            execution_request={"confirm_holdout": True},
        )
        assert holdout_confirmation.domain_id and holdout_confirmation.work_item_id
        assert (
            database.get_work_item(holdout_confirmation.work_item_id).domain_id
            == holdout_confirmation.domain_id
        )
        second_result = studio.execute_work_item(holdout_confirmation.work_item_id)
        confirmed = second_result["result"]["qualification"]
        assert confirmed["decision"] == "pass"
        assert confirmed["holdout_evaluation_id"]
        assert confirmed["metrics"]["decision"]["holdout"]["status"] == "pass"

        approved_promotion = studio.promote(candidate_id, "approved")
        assert approved_promotion["overridden"] is False
        assert studio.show_artifact(candidate_id)["aliases"] == ["approved", "candidate"]
        alias_events = catalog._conn.execute(
            "SELECT alias, occurrence_id, override_reason FROM artifact_alias_events "
            "WHERE occurrence_id = ? ORDER BY created_at",
            (candidate_id,),
        ).fetchall()
        assert {(row["alias"], row["occurrence_id"]) for row in alias_events} == {
            ("candidate", candidate_id),
            ("approved", candidate_id),
        }
        assert all(row["override_reason"] is None for row in alias_events)

        evidence_ids = {
            first["quality_evaluation_id"],
            first["performance_evaluation_id"],
            confirmed["holdout_evaluation_id"],
        }
        assert None not in evidence_ids
        for evaluation_id in evidence_ids:
            evaluation = database.get_evaluation(evaluation_id)
            assert evaluation is not None and evaluation.status == "completed"
            evidence_path = Path(evaluation.artifact_path)
            assert (evidence_path / "evaluation.json").is_file()
            assert (evidence_path / "metrics.json").is_file()
            assert (evidence_path / "samples.jsonl").is_file()
            assert database.count_evaluation_samples(evaluation_id) > 0

        def serving_starter(profile_revision, occurrence, launch_spec):
            assert profile_revision["occurrence_id"] == occurrence.id == candidate_id
            assert launch_spec["serving_id"] == "lifecycle-server"
            return {
                "state": "serving",
                "process_id": 42420,
                "process_started_at": 1720972800.0,
                "endpoint": "http://127.0.0.1:8123",
            }

        studio.serving_starter = serving_starter
        scheduler.process_probe = lambda process_id, started_at: (
            process_id == 42420 and started_at == 1720972800.0
        )
        serving = studio.queue_serving(
            candidate_id,
            name="Lifecycle local server",
            backend="fixture",
            endpoint_settings={"host": "127.0.0.1", "port": 8123},
            generation_settings={"seed": 17},
            resource_requirements={"accelerator_memory_bytes": 1024},
            serving_id="lifecycle-server",
        )
        assert serving.domain_id and serving.work_item_id
        assert database.get_work_item(serving.work_item_id).domain_id == serving.domain_id
        served = studio.execute_work_item(serving.work_item_id)["result"]
        assert served["process_identity"] == {
            "process_id": 42420,
            "process_started_at": 1720972800.0,
        }
        lease = database.get_resource_lease("accelerator")
        assert lease is not None
        assert lease.holder_id == "lifecycle-server"
        assert lease.holder_pid == 42420
        assert lease.holder_pid_started_at == 1720972800.0
        assert lease.metadata["artifact_hash"] == candidate_hash
        assert lease.metadata["profile_revision_id"] == serving.domain_id
        assert studio.release_serving("lifecycle-server") is True

        bundle_path = tmp_path / "portable-approved"
        export = studio.queue_export(
            occurrence_id=candidate_id,
            destination=bundle_path,
            replay_identity={"run_id": "lifecycle-run", "artifact_hash": candidate_hash},
            dataset_identity={"version_id": "dataset-v4", "split": "train"},
            license_metadata={"license": "Apache-2.0"},
        )
        assert export.domain_id and export.work_item_id
        assert database.get_work_item(export.work_item_id).domain_id == export.domain_id
        exported = studio.execute_work_item(export.work_item_id)["result"]
        export_occurrence_id = exported["output_occurrence_id"]
        assert catalog.get_operation(export.domain_id).output_occurrence_id == export_occurrence_id
        export_parent = studio.lineage(export_occurrence_id)["catalog"]["parents"][0]
        assert export_parent["parent_occurrence_id"] == candidate_id
        assert export_parent["relation"] == "export"
        assert export_parent["operation_id"] == export.domain_id

        manifest = json.loads((bundle_path / "bundle-manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_content_hash"] == candidate_hash
        assert json.loads((bundle_path / "replay.json").read_text())["run_id"] == "lifecycle-run"
        assert json.loads((bundle_path / "dataset.json").read_text())["version_id"] == "dataset-v4"
        qualification_evidence = json.loads((bundle_path / "qualification.json").read_text())
        assert qualification_evidence["decision"] == "pass"
        assert qualification_evidence["holdout_evaluation_id"] == confirmed["holdout_evaluation_id"]
        assert (bundle_path / "SHA256SUMS").is_file()
        assert (bundle_path / "MODEL_CARD.md").is_file()

        durable_work_ids = [
            bake.work_item_id,
            conversion.work_item_id,
            qualification.work_item_id,
            holdout_confirmation.work_item_id,
            serving.work_item_id,
            export.work_item_id,
        ]
        for work_item_id in durable_work_ids:
            assert database.get_work_item(work_item_id).status == "completed"
            events = catalog.list_events(work_item_id=work_item_id)
            assert events[0].event_type == "queued"
            assert events[-1].event_type == "completed"
    finally:
        evaluations.shutdown()
        database.close()

    reopened = RunDatabase(str(database_path))
    try:
        durable_catalog = LabV4Catalog(reopened)
        assert reopened.get_work_item(export.work_item_id).status == "completed"
        assert durable_catalog.get_operation(export.domain_id).output_occurrence_id == (
            export_occurrence_id
        )
        assert durable_catalog.aliases_for(candidate_id) == ["approved", "candidate"]
        assert durable_catalog.get_qualification(holdout_confirmation.domain_id).decision == "pass"
        assert (bundle_path / "bundle-manifest.json").is_file()
    finally:
        reopened.close()
