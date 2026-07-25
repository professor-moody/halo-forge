"""Persistent evaluation-backed execution for artifact qualification."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from halo_forge.evaluation_lab import (
    EvaluationLabService,
    canonical_adapter_id,
    infer_suite_adapter,
    resolve_subject,
)
from halo_forge.run_db import (
    ArtifactOccurrenceRecord,
    ArtifactQualificationRecord,
    LabV4Catalog,
    RunDatabase,
)

from .decisions import QualificationEvidence, evaluate_qualification
from .profiles import QualificationMetricRule, QualificationProfileRevision


class EvaluationQualificationExecutor:
    """Run/resolve each profile stage and apply the immutable decision rules."""

    def __init__(
        self,
        database: RunDatabase,
        catalog: LabV4Catalog,
        evaluations: EvaluationLabService,
    ):
        self.database = database
        self.catalog = catalog
        self.evaluations = evaluations

    @staticmethod
    def _rules(
        thresholds: Sequence[Mapping[str, Any]], stage: str
    ) -> tuple[QualificationMetricRule, ...]:
        result = []
        for value in thresholds:
            if str(value.get("stage") or "development").lower() != stage:
                continue
            result.append(
                QualificationMetricRule(
                    metric=str(value.get("metric") or value.get("name") or ""),
                    direction=str(value.get("direction") or "maximize"),
                    pass_threshold=value.get("pass_threshold", value.get("threshold")),
                    warn_threshold=value.get("warn_threshold"),
                    maximum_regression=value.get(
                        "maximum_regression", value.get("allowed_quality_delta")
                    ),
                    required=bool(value.get("required", True)),
                )
            )
        return tuple(result)

    def _profile(self, revision: Any) -> QualificationProfileRevision:
        row = self.database._conn.execute(
            "SELECT name FROM qualification_profiles WHERE id = ?",
            (revision.profile_id,),
        ).fetchone()
        revision_value = revision.to_dict()
        settings = dict(revision_value.get("generation_settings") or {})
        thresholds = tuple(revision_value.get("thresholds") or ())
        return QualificationProfileRevision(
            profile_id=revision.profile_id,
            revision_number=revision.revision_number,
            name=str(row["name"] if row else revision.profile_id),
            development_suite_revision_id=revision.quality_suite_revision_id,
            operational_suite_revision_id=revision.operational_suite_revision_id,
            holdout_suite_revision_id=revision.holdout_suite_revision_id,
            development_rules=self._rules(thresholds, "development"),
            operational_rules=self._rules(thresholds, "operational"),
            holdout_rules=self._rules(thresholds, "holdout"),
            target_backend=str(revision.target_backend or "local"),
            generation_settings=settings.get("generation_settings", settings),
            performance_settings=settings.get("performance_settings", {}),
        )

    def _subject(self, occurrence: ArtifactOccurrenceRecord) -> Dict[str, Any]:
        blob = self.catalog.get_blob(occurrence.blob_id)
        if blob is None:
            raise ValueError(f"artifact {occurrence.id} has no content blob")
        locations = [
            value for value in self.catalog.list_locations(blob.id) if value.state == "available"
        ]
        location = next(
            (value for value in locations if value.storage_mode == "managed"),
            locations[0] if locations else None,
        )
        if location is None:
            raise ValueError(f"artifact {occurrence.id} has no available location")
        return {
            "type": "model",
            "ref": location.path,
            "content_hash": blob.content_hash,
            "backend": occurrence.backend,
        }

    def _run_stage(
        self,
        *,
        occurrence: ArtifactOccurrenceRecord,
        suite_revision_id: str,
        stage: str,
        request: Mapping[str, Any],
        supplied_evaluation_id: Optional[str],
        timeout: float,
    ) -> tuple[str, Dict[str, float]]:
        evaluation = (
            self.database.get_evaluation(supplied_evaluation_id) if supplied_evaluation_id else None
        )
        if evaluation is None:
            launched = self.evaluations.launch_evaluation(
                suite_revision_id=suite_revision_id,
                adapter_id="performance" if stage == "operational" else None,
                subject=self._subject(occurrence),
                request=request,
            )
            evaluation = launched.evaluation
        if evaluation.suite_revision_id != suite_revision_id:
            raise ValueError(f"{stage} evaluation uses a different suite revision")
        if evaluation.status not in {"completed", "failed", "cancelled"}:
            evaluation = self.evaluations.jobs.wait(evaluation.id, timeout=timeout)
        if evaluation.status != "completed":
            raise ValueError(
                f"{stage} evaluation {evaluation.id} did not complete: "
                f"{evaluation.error or evaluation.status}"
            )
        expected_subject = resolve_subject(self._subject(occurrence), self.database)
        if (
            evaluation.subject_type != expected_subject.subject_type
            or evaluation.subject_ref != expected_subject.subject_ref
            or evaluation.subject_hash != expected_subject.subject_hash
        ):
            raise ValueError(f"{stage} evaluation is bound to a different artifact subject")
        suite = self.database.get_benchmark_suite_revision(suite_revision_id)
        if suite is None:
            raise ValueError(f"{stage} benchmark suite revision is missing")
        expected_adapter = "performance" if stage == "operational" else infer_suite_adapter(suite)
        if canonical_adapter_id(evaluation.adapter_id) != canonical_adapter_id(expected_adapter):
            raise ValueError(f"{stage} evaluation uses a different evaluation adapter")
        expected_request = {**dict(suite.generation_settings or {}), **dict(request)}
        expected_request["generation_settings"] = {
            **dict(suite.generation_settings or {}),
            **dict(request.get("generation_settings") or {}),
        }
        actual_request = evaluation.request.get("adapter_request")
        if not isinstance(actual_request, Mapping) or dict(actual_request) != expected_request:
            raise ValueError(
                f"{stage} evaluation configuration is not bound to this qualification profile"
            )
        if not evaluation.artifact_path:
            raise ValueError(f"{stage} evaluation has no published evidence bundle")
        # Re-read through Evaluation Lab's integrity verifier so a completed
        # catalog row cannot qualify a model after its evidence was mutated.
        evaluation = self.evaluations.jobs._read_published(  # noqa: SLF001
            evaluation.id, self.evaluations.jobs.artifact_root / evaluation.id
        )
        metrics = {
            metric.name: float(metric.value)
            for metric in self.database.list_evaluation_metrics(evaluation.id)
            if not metric.suite_item_id
        }
        if suite is None or suite.primary_metric not in metrics:
            raise ValueError(f"{stage} evaluation is missing its primary metric")
        if self.database.count_evaluation_samples(evaluation.id) <= 0:
            raise ValueError(f"{stage} evaluation has no sample evidence")
        return evaluation.id, metrics

    def __call__(
        self,
        qualification: ArtifactQualificationRecord,
        launch_spec: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        revision = self.catalog.get_qualification_profile_revision(
            qualification.profile_revision_id
        )
        if revision is None:
            raise ValueError("qualification profile revision is missing")
        profile = self._profile(revision)
        candidate = self.catalog.get_occurrence(qualification.occurrence_id)
        if candidate is None:
            raise ValueError("candidate artifact occurrence is missing")
        parent = (
            self.catalog.get_occurrence(qualification.parent_occurrence_id)
            if qualification.parent_occurrence_id
            else None
        )
        execution = dict(launch_spec.get("execution_request") or {})
        stored_settings = dict(revision.to_dict().get("generation_settings") or {})
        reward_requirement = dict(stored_settings.get("reward_integrity") or {})
        required_audit = None
        if reward_requirement.get("require_final_audit_pass"):
            if not candidate.run_id:
                return {
                    "decision": "fail",
                    "reasons": ["artifact has no audited training-run identity"],
                    "metrics": {},
                    "decision_evidence": {
                        "reward_integrity": {"required": True, "status": "missing_run"}
                    },
                }
            params: list[Any] = [candidate.run_id]
            profile_clause = ""
            required_profile = reward_requirement.get(
                "integrity_profile_revision_id"
            )
            if required_profile:
                profile_clause = " AND a.integrity_profile_revision_id=?"
                params.append(str(required_profile))
            row = self.database._conn.execute(
                """SELECT a.id,a.manifest_hash,s.boundary_value
                     FROM reward_integrity_audits a
                     JOIN training_signal_shards s ON s.id=a.signal_shard_id
                    WHERE a.run_id=? AND a.status='completed'"""
                + profile_clause
                + " ORDER BY s.boundary_value DESC,a.completed_at DESC LIMIT 1",
                params,
            ).fetchone()
            if row is None:
                return {
                    "decision": "fail",
                    "reasons": ["required final reward-integrity audit is missing"],
                    "metrics": {},
                    "decision_evidence": {
                        "reward_integrity": {"required": True, "status": "missing"}
                    },
                }
            decision_row = self.database._conn.execute(
                "SELECT * FROM reward_integrity_decisions WHERE audit_id=? "
                "ORDER BY created_at DESC,id DESC LIMIT 1",
                (row["id"],),
            ).fetchone()
            if decision_row is None or str(decision_row["decision"]) != "pass":
                return {
                    "decision": "fail",
                    "reasons": ["required final reward-integrity audit did not pass"],
                    "metrics": {},
                    "decision_evidence": {
                        "reward_integrity": {
                            "required": True,
                            "audit_id": str(row["id"]),
                            "status": (
                                str(decision_row["decision"])
                                if decision_row is not None
                                else "missing_decision"
                            ),
                        }
                    },
                }
            from halo_forge.reward_integrity import RewardIntegrityService

            reward_service = RewardIntegrityService(self.database)
            reward_service.verify_audit_bundle(str(row["id"]))
            reward_service.bind(
                reward_system_revision_id=reward_service.get_audit(
                    str(row["id"])
                ).reward_system_revision_id,
                integrity_profile_revision_id=reward_service.get_audit(
                    str(row["id"])
                ).integrity_profile_revision_id,
                audit_id=str(row["id"]),
                domain_kind="artifact_occurrence",
                domain_id=candidate.id,
                role="final_qualification_gate",
                context={"qualification_id": qualification.id},
            )
            required_audit = {
                "required": True,
                "audit_id": str(row["id"]),
                "manifest_hash": str(row["manifest_hash"]),
                "boundary_value": int(row["boundary_value"]),
                "status": "pass",
            }
        timeout = float(execution.get("timeout_seconds", 24 * 60 * 60))
        evaluation_ids = dict(execution.get("evaluation_ids") or {})
        adapter_request = {
            "generation_settings": dict(profile.generation_settings),
            "performance_settings": profile.performance_settings.to_dict(),
            "backend": profile.target_backend,
        }
        qualification_binding = {
            "profile_revision_id": revision.id,
            "profile_content_hash": profile.content_hash,
            "stored_profile_content_hash": revision.content_hash,
        }

        candidate_metrics: Dict[str, Dict[str, float]] = {}
        parent_metrics: Dict[str, Dict[str, float]] = {}
        candidate_evaluations: Dict[str, str] = {}
        parent_evaluations: Dict[str, str] = {}
        stage_suites = {
            "development": profile.development_suite_revision_id,
            "operational": profile.operational_suite_revision_id,
        }
        # Holdout confirmation is intentionally a separate, explicit action.
        # Running it for every candidate qualification would expose final-gate
        # evidence during iterative selection and turn the holdout into another
        # development signal. A supplied immutable holdout evaluation also
        # counts as an explicit confirmation request.
        confirm_holdout = bool(execution.get("confirm_holdout")) or bool(
            evaluation_ids.get("holdout")
        )
        if profile.holdout_suite_revision_id and confirm_holdout:
            stage_suites["holdout"] = profile.holdout_suite_revision_id
        for stage, suite_revision_id in stage_suites.items():
            candidate_blob = self.catalog.get_blob(candidate.blob_id)
            candidate_request = {
                **adapter_request,
                "qualification_binding": {
                    **qualification_binding,
                    "stage": stage,
                    "artifact_occurrence_id": candidate.id,
                    "artifact_content_hash": (
                        candidate_blob.content_hash if candidate_blob else candidate.blob_id
                    ),
                },
            }
            candidate_id, metrics = self._run_stage(
                occurrence=candidate,
                suite_revision_id=suite_revision_id,
                stage=stage,
                request=candidate_request,
                supplied_evaluation_id=evaluation_ids.get(stage),
                timeout=timeout,
            )
            candidate_evaluations[stage] = candidate_id
            candidate_metrics[stage] = metrics
            if parent is not None:
                parent_blob = self.catalog.get_blob(parent.blob_id)
                parent_request = {
                    **adapter_request,
                    "qualification_binding": {
                        **qualification_binding,
                        "stage": stage,
                        "role": "parent",
                        "artifact_occurrence_id": parent.id,
                        "artifact_content_hash": (
                            parent_blob.content_hash if parent_blob else parent.blob_id
                        ),
                    },
                }
                parent_id, baseline = self._run_stage(
                    occurrence=parent,
                    suite_revision_id=suite_revision_id,
                    stage=stage,
                    request=parent_request,
                    supplied_evaluation_id=evaluation_ids.get(f"parent_{stage}"),
                    timeout=timeout,
                )
                parent_evaluations[stage] = parent_id
                parent_metrics[stage] = baseline

        candidate_blob = self.catalog.get_blob(candidate.blob_id)
        parent_blob = self.catalog.get_blob(parent.blob_id) if parent else None
        candidate_evidence = QualificationEvidence(
            artifact_hash=candidate_blob.content_hash if candidate_blob else candidate.blob_id,
            profile_content_hash=profile.content_hash,
            development_metrics=candidate_metrics.get("development", {}),
            operational_metrics=candidate_metrics.get("operational", {}),
            holdout_metrics=candidate_metrics.get("holdout", {}),
            holdout_complete="holdout" in candidate_metrics,
        )
        parent_evidence = (
            QualificationEvidence(
                artifact_hash=parent_blob.content_hash if parent_blob else parent.blob_id,
                profile_content_hash=profile.content_hash,
                development_metrics=parent_metrics.get("development", {}),
                operational_metrics=parent_metrics.get("operational", {}),
                holdout_metrics=parent_metrics.get("holdout", {}),
                holdout_complete="holdout" in parent_metrics,
            )
            if parent is not None
            else None
        )
        decision = evaluate_qualification(profile, candidate_evidence, parent=parent_evidence)
        return {
            "decision": decision.overall_status,
            "reasons": list(decision.reasons),
            "metrics": candidate_metrics,
            "decision_evidence": decision.to_dict(),
            "quality_evaluation_id": candidate_evaluations.get("development"),
            "performance_evaluation_id": candidate_evaluations.get("operational"),
            "holdout_evaluation_id": candidate_evaluations.get("holdout"),
            "parent_evaluation_ids": parent_evaluations,
            "reward_integrity": required_audit,
        }


__all__ = ["EvaluationQualificationExecutor"]
