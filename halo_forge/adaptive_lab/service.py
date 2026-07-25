"""Service layer for adaptive checkpoint, analysis, and evidence workflows."""

from __future__ import annotations

import csv
import io
import math
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from halo_forge.run_db.db import RunDatabase

from ._canonical import content_fingerprint
from .models import (
    CheckpointGateDecision,
    CheckpointPolicyRevision,
    EvidenceBundle,
    ResearchDecisionRecord as ResearchDecisionModel,
    ResolvedCheckpointPlan,
    WorkspaceDraft,
)
from .reports import (
    comparison_interval_svg,
    markdown_report_html,
    publish_evidence_bundle,
    verify_evidence_bundle,
)
from .statistics import build_cohort_snapshot


class AdaptiveLabError(RuntimeError):
    """Base error for adaptive research operations."""


class EvidenceBundleExecutionError(AdaptiveLabError):
    """An evidence bundle could not be built or atomically published."""


EVIDENCE_FORMATS = frozenset({"markdown", "html", "json", "csv", "svg"})


def _resolve_evidence_formats(value: Any = None) -> tuple[str, ...]:
    if value is None:
        formats = set(EVIDENCE_FORMATS)
    elif isinstance(value, str):
        formats = {value.strip().lower()}
    elif isinstance(value, Sequence):
        formats = {str(item).strip().lower() for item in value}
    else:
        raise ValueError("evidence formats must be a string or list of strings")
    if not formats or not formats.issubset(EVIDENCE_FORMATS):
        unsupported = sorted(formats.difference(EVIDENCE_FORMATS))
        raise ValueError(
            "evidence formats must be a non-empty subset of "
            f"{sorted(EVIDENCE_FORMATS)}; unsupported={unsupported}"
        )
    return tuple(sorted(formats))


def _finite_metric_map(values: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key, value in dict(values or {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"metric {key!r} must be numeric")
        metric = float(value)
        if not math.isfinite(metric):
            raise ValueError(f"metric {key!r} must be finite")
        result[str(key)] = metric
    return result


def _directional_delta(current: float, reference: float, direction: str) -> float:
    return current - reference if direction == "maximize" else reference - current


def _csv_rows(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return stream.getvalue()


class AdaptiveLabService:
    """Transport-neutral facade shared by dashboard, CLI, and workers."""

    def __init__(
        self,
        db: RunDatabase,
        evidence_root: str | Path | None = None,
    ) -> None:
        self.db = db
        self.evidence_root = Path(
            evidence_root or Path.home() / ".halo-forge" / "research" / "evidence"
        ).expanduser()

    # ----- checkpoint policies ------------------------------------------

    def create_policy(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        policy_id: Optional[str] = None,
    ) -> Any:
        return self.db.create_checkpoint_policy(
            name=name, description=description, policy_id=policy_id
        )

    def list_policies(
        self, *, archived: Optional[bool] = False, limit: int = 100, offset: int = 0
    ) -> list[Any]:
        return self.db.list_checkpoint_policies(archived=archived, limit=limit, offset=offset)

    def create_policy_revision(
        self,
        policy_id: str,
        definition: CheckpointPolicyRevision | Mapping[str, Any],
    ) -> CheckpointPolicyRevision:
        policy = self.db.get_checkpoint_policy(policy_id)
        if policy is None:
            raise ValueError(f"unknown checkpoint policy: {policy_id}")
        next_revision = len(self.db.list_checkpoint_policy_revisions(policy_id)) + 1
        if isinstance(definition, CheckpointPolicyRevision):
            if definition.policy_id != policy_id:
                raise ValueError("checkpoint policy revision belongs to another policy")
            values = definition.to_dict()
            values.pop("content_hash", None)
            values["revision_id"] = None
            values["revision_number"] = next_revision
            values["name"] = policy.name
            values["description"] = policy.description
            revision = CheckpointPolicyRevision.from_dict(values)
        else:
            values = dict(definition)
            if values.get("policy_id") not in {None, policy_id}:
                raise ValueError("checkpoint policy revision belongs to another policy")
            values.update(
                policy_id=policy_id,
                revision_id=None,
                revision_number=next_revision,
                name=values.get("name") or policy.name,
                description=values.get("description", policy.description),
            )
            values.pop("content_hash", None)
            revision = CheckpointPolicyRevision.from_dict(values)
        for suite_revision_id in revision.required_suite_revision_ids:
            suite_revision = self.db.get_benchmark_suite_revision(suite_revision_id)
            if suite_revision is None:
                raise ValueError(f"unknown benchmark suite revision: {suite_revision_id}")
            suite = self.db.get_benchmark_suite(suite_revision.suite_id)
            if suite is None or suite.purpose != "development":
                raise ValueError("checkpoint policies may use development-purpose suites only")
        record = self.db.create_checkpoint_policy_revision(
            policy_id=policy_id,
            definition=revision.to_dict(),
            content_hash=revision.content_hash,
            development_suite_revision_id=revision.development_suite_revision_id,
            primary_metric=revision.primary_metric,
            direction=revision.direction,
        )
        hydrated = revision.to_dict()
        hydrated.update(
            revision_id=record.id,
            revision_number=record.revision_number,
            content_hash=record.content_hash,
        )
        return CheckpointPolicyRevision.from_dict(hydrated)

    def get_policy_revision(self, revision_id: str) -> Optional[CheckpointPolicyRevision]:
        record = self.db.get_checkpoint_policy_revision(revision_id)
        if record is None:
            return None
        policy = self.db.get_checkpoint_policy(record.policy_id)
        values = record.definition
        values.update(
            policy_id=record.policy_id,
            revision_id=record.id,
            revision_number=record.revision_number,
            name=values.get("name") or (policy.name if policy else record.policy_id),
            description=values.get("description") if policy is None else policy.description,
            development_suite_revision_id=record.development_suite_revision_id,
            primary_metric=record.primary_metric,
            direction=record.direction,
            content_hash=record.content_hash,
        )
        return CheckpointPolicyRevision.from_dict(values)

    def list_policy_revisions(self, policy_id: str) -> list[CheckpointPolicyRevision]:
        revisions: list[CheckpointPolicyRevision] = []
        for value in self.db.list_checkpoint_policy_revisions(policy_id):
            revision = self.get_policy_revision(value.id)
            if revision is not None:
                revisions.append(revision)
        return revisions

    def _policy_revision(
        self, value: str | CheckpointPolicyRevision | Mapping[str, Any]
    ) -> CheckpointPolicyRevision:
        if isinstance(value, str):
            revision = self.get_policy_revision(value)
            if revision is None:
                raise ValueError(f"unknown checkpoint policy revision: {value}")
            return revision
        revision = (
            value
            if isinstance(value, CheckpointPolicyRevision)
            else CheckpointPolicyRevision.from_dict(value)
        )
        if not revision.revision_id:
            raise ValueError("checkpoint policy revision must be persisted before resolution")
        stored = self.get_policy_revision(revision.revision_id)
        if stored is None:
            raise ValueError(f"unknown checkpoint policy revision: {revision.revision_id}")
        if stored.content_hash != revision.content_hash:
            raise ValueError("checkpoint policy revision content does not match persistence")
        return stored

    def resolve_checkpoint_plan(
        self,
        revision: str | CheckpointPolicyRevision | Mapping[str, Any],
        *,
        trainer_mode: str,
        total_budget: int,
        supported_units: Sequence[str] = (),
        capabilities: Sequence[str] = (),
    ) -> ResolvedCheckpointPlan:
        policy = self._policy_revision(revision)
        total = int(total_budget)
        if total <= 0:
            raise ValueError("total_budget must be positive")
        units = {str(value).strip().lower() for value in supported_units}
        available = {str(value).strip().lower() for value in capabilities}
        if units and policy.schedule.unit not in units:
            raise ValueError(
                f"trainer {trainer_mode!r} does not support {policy.schedule.unit} boundaries"
            )
        missing = sorted(set(policy.compatible_capabilities).difference(available))
        if missing:
            raise ValueError(
                "trainer is missing required checkpoint capabilities: " + ", ".join(missing)
            )

        schedule = policy.schedule
        notes: list[str] = []
        if schedule.mode == "final":
            boundaries = [total]
            notes.append("final_only")
        elif schedule.mode == "interval":
            assert schedule.interval is not None
            boundaries = list(range(schedule.interval, total + 1, schedule.interval))
        elif schedule.mode == "percentages":
            boundaries = [max(1, math.ceil(total * value)) for value in schedule.percentages]
        else:
            boundaries = list(schedule.boundaries)
            if boundaries[-1] > total:
                raise ValueError("an explicit checkpoint boundary exceeds total_budget")
        if schedule.include_final:
            boundaries.append(total)
        boundaries = sorted(set(boundaries))
        if not boundaries:
            raise ValueError("checkpoint schedule resolved to no boundaries")
        if len(boundaries) == 1 and boundaries[0] == total and "final_only" not in notes:
            notes.append("resolved_to_final_only")
        return ResolvedCheckpointPlan(
            policy_revision_id=policy.revision_id or "",
            policy_hash=policy.content_hash,
            trainer_mode=trainer_mode,
            unit=schedule.unit,
            total_budget=total,
            boundaries=tuple(boundaries),
            required_suite_revision_ids=policy.required_suite_revision_ids,
            automatic_actions=policy.automatic_actions,
            retention=policy.retention,
            capability_notes=tuple(notes),
        )

    # ----- gate decisions ------------------------------------------------

    def evaluate_gate(
        self,
        policy: str | CheckpointPolicyRevision | Mapping[str, Any],
        plan: ResolvedCheckpointPlan | Mapping[str, Any],
        *,
        boundary_index: int,
        current_metrics: Mapping[str, Any],
        baseline_metrics: Optional[Mapping[str, Any]] = None,
        previous_metrics: Optional[Mapping[str, Any]] = None,
        best_metrics: Optional[Mapping[str, Any]] = None,
        missing_evidence: Sequence[str] = (),
        plateau_counts: Optional[Mapping[str, int]] = None,
        automatic: Optional[bool] = None,
    ) -> CheckpointGateDecision:
        revision = self._policy_revision(policy)
        resolved = (
            plan
            if isinstance(plan, ResolvedCheckpointPlan)
            else ResolvedCheckpointPlan.from_dict(plan)
        )
        if resolved.policy_revision_id != revision.revision_id:
            raise ValueError("resolved plan does not belong to the checkpoint policy revision")
        if resolved.policy_hash != revision.content_hash:
            raise ValueError("resolved plan policy hash does not match the policy revision")
        boundary_index = int(boundary_index)
        if boundary_index < 0 or boundary_index >= len(resolved.boundaries):
            raise ValueError("boundary_index is outside the resolved checkpoint plan")
        if automatic is True and not revision.automatic_actions:
            raise ValueError("automatic checkpoint actions are disabled by this policy revision")
        auto_enabled = revision.automatic_actions if automatic is None else bool(automatic)

        current = _finite_metric_map(current_metrics)
        references = {
            "baseline": _finite_metric_map(baseline_metrics),
            "previous": _finite_metric_map(previous_metrics),
            "best": _finite_metric_map(best_metrics),
        }
        counts = {str(key): max(0, int(value)) for key, value in dict(plateau_counts or {}).items()}
        missing = {str(value).strip() for value in missing_evidence if str(value).strip()}
        if revision.primary_metric not in current:
            missing.add(f"metric:{revision.primary_metric}")

        outcomes: list[Dict[str, Any]] = []
        breaches: list[tuple[int, str, str]] = []
        priority = {"guardrail": 0, "plateau": 1, "objective": 2}
        for rule in sorted(
            revision.rules,
            key=lambda value: (priority[value.kind], value.metric, value.comparison),
        ):
            outcome: Dict[str, Any] = {
                "metric": rule.metric,
                "kind": rule.kind,
                "comparison": rule.comparison,
                "direction": rule.direction,
                "breached": False,
            }
            if rule.metric not in current:
                if rule.required:
                    missing.add(f"metric:{rule.metric}")
                outcome["status"] = "missing_current_metric"
                outcomes.append(outcome)
                continue
            value = current[rule.metric]
            outcome["current"] = value
            if rule.comparison == "absolute":
                assert rule.threshold is not None
                passed = (
                    value >= rule.threshold
                    if rule.direction == "maximize"
                    else value <= rule.threshold
                )
                outcome.update(threshold=rule.threshold, passed=passed)
                breached = not passed
            else:
                reference = references[rule.comparison].get(rule.metric)
                if reference is None:
                    if rule.kind == "plateau":
                        outcome.update(
                            status="baseline_established",
                            passed=True,
                            plateau_count=0,
                            patience=rule.patience,
                        )
                    elif rule.required:
                        missing.add(f"{rule.comparison}:{rule.metric}")
                        outcome["status"] = "missing_reference_metric"
                    else:
                        outcome["status"] = "optional_reference_missing"
                    outcomes.append(outcome)
                    continue
                delta = _directional_delta(value, reference, rule.direction)
                minimum_delta = float(rule.minimum_delta or 0.0)
                breached = delta < minimum_delta
                outcome.update(
                    reference=reference,
                    direction_normalized_delta=delta,
                    minimum_delta=minimum_delta,
                    passed=not breached,
                )
            if breached and rule.kind == "plateau":
                observed_count = counts.get(rule.metric, 1)
                outcome.update(plateau_count=observed_count, patience=rule.patience)
                breached = observed_count >= rule.patience
            outcome["breached"] = breached
            if breached:
                breaches.append((priority[rule.kind], rule.on_breach, f"{rule.kind}:{rule.metric}"))
            outcomes.append(outcome)

        if missing:
            desired_action = "pause"
            reasons = [
                "missing_required_evidence",
                *[f"missing:{value}" for value in sorted(missing)],
            ]
        elif breaches:
            strongest = sorted(
                breaches,
                key=lambda value: (value[0], 0 if value[1] == "stop" else 1, value[2]),
            )[0]
            desired_action = strongest[1]
            reasons = [f"gate_breach:{value[2]}" for value in breaches]
        else:
            desired_action = "continue"
            reasons = ["all_required_gates_passed"]

        requires_review = bool(missing) or not auto_enabled
        if requires_review:
            action = "pause"
            is_automatic = False
            if not missing:
                reasons = ["manual_review_required", *reasons]
        else:
            action = desired_action
            is_automatic = True
        evidence = {
            "boundary": {
                "index": boundary_index,
                "unit": resolved.unit,
                "value": resolved.boundaries[boundary_index],
            },
            "current_metrics": current,
            "baseline_metrics": references["baseline"],
            "previous_metrics": references["previous"],
            "best_metrics": references["best"],
            "rule_outcomes": outcomes,
            "desired_action": desired_action,
            "manual_review_required": requires_review,
        }
        return CheckpointGateDecision(
            policy_revision_id=resolved.policy_revision_id,
            plan_hash=resolved.content_hash,
            boundary_index=boundary_index,
            action=action,
            automatic=is_automatic,
            reasons=tuple(reasons),
            evidence=evidence,
        )

    def evaluate_and_record_gate(
        self,
        policy: str | CheckpointPolicyRevision | Mapping[str, Any],
        plan: ResolvedCheckpointPlan | Mapping[str, Any],
        *,
        boundary_index: int,
        current_metrics: Mapping[str, Any],
        idempotency_key: str,
        baseline_metrics: Optional[Mapping[str, Any]] = None,
        previous_metrics: Optional[Mapping[str, Any]] = None,
        best_metrics: Optional[Mapping[str, Any]] = None,
        missing_evidence: Sequence[str] = (),
        plateau_counts: Optional[Mapping[str, int]] = None,
        automatic: Optional[bool] = None,
        run_group_id: Optional[str] = None,
        trial_run_id: Optional[str] = None,
        trial_segment_id: Optional[str] = None,
        checkpoint_occurrence_id: Optional[str] = None,
    ) -> Any:
        decision = self.evaluate_gate(
            policy,
            plan,
            boundary_index=boundary_index,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            previous_metrics=previous_metrics,
            best_metrics=best_metrics,
            missing_evidence=missing_evidence,
            plateau_counts=plateau_counts,
            automatic=automatic,
        )
        return self.db.create_checkpoint_gate_decision(
            policy_revision_id=decision.policy_revision_id,
            plan_hash=decision.plan_hash,
            boundary_index=decision.boundary_index,
            action=decision.action,
            reasons=decision.reasons,
            evidence=decision.evidence.to_dict(),
            idempotency_key=idempotency_key,
            automatic=decision.automatic,
            run_group_id=run_group_id,
            trial_run_id=trial_run_id,
            trial_segment_id=trial_segment_id,
            checkpoint_occurrence_id=checkpoint_occurrence_id,
            content_hash=decision.content_hash,
        )

    def override_gate(
        self,
        decision_id: str,
        *,
        action: str,
        reason: str,
        idempotency_key: Optional[str] = None,
    ) -> Any:
        existing = self.db.get_checkpoint_gate_decision(decision_id)
        if existing is None:
            raise ValueError(f"unknown checkpoint gate decision: {decision_id}")
        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError("checkpoint gate override reason is required")
        key = idempotency_key or content_fingerprint(
            {"override_of_id": decision_id, "action": action, "reason": normalized_reason}
        )
        evidence = dict(existing.evidence)
        evidence["operator_override"] = {
            "previous_action": existing.action,
            "action": action,
            "reason": normalized_reason,
        }
        return self.db.create_checkpoint_gate_decision(
            policy_revision_id=existing.policy_revision_id,
            plan_hash=existing.plan_hash,
            boundary_index=existing.boundary_index,
            action=action,
            reasons=("operator_override",),
            evidence=evidence,
            idempotency_key=key,
            automatic=False,
            run_group_id=existing.run_group_id,
            trial_run_id=existing.trial_run_id,
            trial_segment_id=existing.trial_segment_id,
            checkpoint_occurrence_id=existing.checkpoint_occurrence_id,
            override_of_id=existing.id,
            override_reason=normalized_reason,
        )

    def list_gate_decisions(self, **filters: Any) -> list[Any]:
        return self.db.list_checkpoint_gate_decisions(**filters)

    # ----- cohort analysis and reviewed decisions -----------------------

    def analyze_cohort(
        self,
        observations: Iterable[Any],
        *,
        metric: str,
        direction: str,
        baseline_subject_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        confidence_level: float = 0.95,
        bootstrap_resamples: int = 10_000,
        bootstrap_seed: int = 42,
        practical_delta: float = 0.0,
        equivalence_delta: Optional[float] = None,
        required_seeds: Sequence[int] = (),
        evidence_compatibility: Sequence[Any] = (),
        context: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        snapshot = build_cohort_snapshot(
            observations,
            metric=metric,
            direction=direction,
            baseline_subject_id=baseline_subject_id,
            confidence_level=confidence_level,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            practical_delta=practical_delta,
            equivalence_delta=equivalence_delta,
            required_seeds=required_seeds,
            evidence_compatibility=evidence_compatibility,
            context=context,
        )
        return self.db.create_cohort_analysis_snapshot(
            request=snapshot.request.to_dict(),
            analysis=snapshot.analysis.to_dict(),
            primary_metric=metric,
            direction=direction,
            content_hash=snapshot.content_hash,
            baseline_subject_id=baseline_subject_id,
            run_group_id=run_group_id,
        )

    def create_research_decision(
        self,
        *,
        analysis_snapshot_id: str,
        selected_subject: Mapping[str, Any],
        rejected_subjects: Sequence[Mapping[str, Any]] = (),
        exclusions: Sequence[Mapping[str, Any]] = (),
        rationale: str,
        fork_spec: Optional[Mapping[str, Any]] = None,
        override_reason: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> Any:
        model = ResearchDecisionModel(
            analysis_snapshot_id=analysis_snapshot_id,
            selected_subject=selected_subject,
            rejected_subjects=tuple(rejected_subjects),
            exclusions=tuple(exclusions),
            rationale=rationale,
            fork_spec=fork_spec or {},
            override_reason=override_reason,
        )
        return self.db.create_research_decision(
            analysis_snapshot_id=analysis_snapshot_id,
            selected_subject=model.selected_subject.to_dict(),
            rejected_subjects=[value.to_dict() for value in model.rejected_subjects],
            exclusions=[value.to_dict() for value in model.exclusions],
            rationale=model.rationale,
            fork_spec=model.fork_spec.to_dict(),
            override_reason=model.override_reason,
            content_hash=model.content_hash,
            decision_id=decision_id,
        )

    def list_analyses(self, *, run_group_id: Optional[str] = None, limit: int = 100) -> list[Any]:
        return self.db.list_cohort_analysis_snapshots(run_group_id=run_group_id, limit=limit)

    def get_analysis(self, snapshot_id: str) -> Any:
        return self.db.get_cohort_analysis_snapshot(snapshot_id)

    def list_research_decisions(
        self, *, analysis_snapshot_id: Optional[str] = None, limit: int = 100
    ) -> list[Any]:
        return self.db.list_research_decisions(
            analysis_snapshot_id=analysis_snapshot_id, limit=limit
        )

    def get_research_decision(self, decision_id: str) -> Any:
        return self.db.get_research_decision(decision_id)

    # ----- evidence bundles ---------------------------------------------

    def queue_evidence_bundle(
        self,
        *,
        analysis_snapshot_id: str,
        research_decision_id: Optional[str] = None,
        request: Optional[Mapping[str, Any]] = None,
        work_item_id: Optional[str] = None,
        bundle_id: Optional[str] = None,
    ) -> Any:
        snapshot = self.db.get_cohort_analysis_snapshot(analysis_snapshot_id)
        if snapshot is None:
            raise ValueError(f"unknown cohort analysis snapshot: {analysis_snapshot_id}")
        decision = None
        if research_decision_id is not None:
            decision = self.db.get_research_decision(research_decision_id)
            if decision is None:
                raise ValueError(f"unknown research decision: {research_decision_id}")
            if decision.analysis_snapshot_id != analysis_snapshot_id:
                raise ValueError("research decision belongs to another analysis snapshot")
        model = EvidenceBundle(
            analysis_snapshot_id=analysis_snapshot_id,
            research_decision_id=research_decision_id,
            request={
                **dict(request or {}),
                "formats": list(_resolve_evidence_formats((request or {}).get("formats"))),
            },
            analysis_hash=snapshot.content_hash,
            decision_hash=None if decision is None else decision.content_hash,
        )
        existing = self.db.find_completed_evidence_bundle(model.content_hash)
        if existing is not None:
            try:
                verify_evidence_bundle(
                    existing.storage_path,
                    content_hash=existing.content_hash,
                )
            except RuntimeError as exc:
                self.db.update_evidence_bundle(
                    existing.id,
                    status="corrupt",
                    error=f"checksum verification failed before reuse: {exc}",
                )
            else:
                return existing
        identifier = bundle_id or uuid.uuid4().hex
        return self.db.create_evidence_bundle(
            analysis_snapshot_id=analysis_snapshot_id,
            research_decision_id=research_decision_id,
            content_hash=model.content_hash,
            storage_path=str(self.evidence_root / identifier),
            request=model.request.to_dict(),
            work_item_id=work_item_id,
            bundle_id=identifier,
        )

    def _report_markdown(self, bundle: Any, snapshot: Any, decision: Any) -> str:
        request = bundle.request
        title = str(request.get("report_title") or "Halo Forge Research Evidence")
        lines = [f"# {title}", "", f"Evidence bundle: `{bundle.id}`", ""]

        run_group = dict(request.get("run_group") or {})
        suite_revision = dict(request.get("suite_revision") or {})
        datasets = list(request.get("dataset_versions") or ())
        run_identities = list(request.get("run_identities") or ())
        if run_group or suite_revision or datasets or run_identities:
            lines.extend(["## Pinned research scope", ""])
            if run_group:
                lines.append(
                    f"- Run group: `{run_group.get('id')}`"
                    + (f" — {run_group.get('name')}" if run_group.get("name") else "")
                )
                if run_group.get("checkpoint_policy_revision_id"):
                    lines.append(
                        "- Checkpoint policy revision: "
                        f"`{run_group.get('checkpoint_policy_revision_id')}`"
                    )
            if suite_revision:
                lines.append(f"- Benchmark suite revision: `{suite_revision.get('id')}`")
            for value in datasets:
                binding = dict(value.get("binding") or {})
                identity = dict(value.get("identity") or {})
                lines.append(
                    "- Dataset binding: "
                    f"`{binding.get('role', 'train')}={binding.get('dataset_version_id')}:"
                    f"{binding.get('split', 'train')}`"
                    + (
                        f" (content `{identity.get('content_hash')}`)"
                        if identity.get("content_hash")
                        else ""
                    )
                )
            for value in run_identities:
                lines.append(
                    f"- Run: `{value.get('run_id')}` · seed `{value.get('seed')}`"
                    + (f" · trial `{value.get('trial_id')}`" if value.get("trial_id") else "")
                )
            lines.append("")

        lines.extend(
            [
                "## Analysis",
                "",
                f"Primary metric: `{snapshot.primary_metric}` ({snapshot.direction})",
                f"Replicate unit: `{snapshot.analysis.get('replicate_unit', 'seed')}`",
                f"Bootstrap: `{snapshot.request.get('bootstrap_resamples', 10_000)}` "
                f"resamples · seed `{snapshot.request.get('bootstrap_seed', 42)}` · "
                f"confidence `{snapshot.request.get('confidence_level', 0.95)}`",
                "",
            ]
        )
        comparisons = snapshot.analysis.get("comparisons") or {}
        if comparisons:
            lines.extend(["## Comparisons", ""])
            confidence_percent = float(snapshot.request.get("confidence_level", 0.95)) * 100.0
            for subject_id, value in sorted(comparisons.items()):
                interval = value.get("confidence_interval") or {}
                interval_text = (
                    "unavailable"
                    if not interval
                    else f"{interval.get('lower'):.6g} to {interval.get('upper'):.6g}"
                )
                lines.append(
                    f"- `{subject_id}`: {value.get('classification')} "
                    f"(mean delta {value.get('mean_delta')}, "
                    f"{confidence_percent:g}% interval {interval_text})"
                )
            lines.append("")
        if decision is not None:
            lines.extend(
                [
                    "## Reviewed decision",
                    "",
                    f"Selected: `{decision.selected_subject}`",
                    "",
                    decision.rationale,
                    "",
                ]
            )
            if decision.override_reason:
                lines.extend([f"Override: {decision.override_reason}", ""])

        assumptions = dict(request.get("analysis_assumptions") or {})
        if assumptions:
            lines.extend(["## Statistical assumptions", ""])
            for key, value in sorted(assumptions.items()):
                lines.append(f"- {key.replace('_', ' ')}: `{value}`")
            lines.append("")

        missing_evidence = dict(request.get("missing_evidence") or {})
        missing_rows = {
            key: value
            for key, value in missing_evidence.items()
            if value not in (None, "", [], {}, ())
        }
        lines.extend(["## Missing evidence", ""])
        if missing_rows:
            for key, value in sorted(missing_rows.items()):
                lines.append(f"- {key.replace('_', ' ')}: `{value}`")
        else:
            lines.append("- No missing evidence was declared for this snapshot.")
        lines.append("")

        runtime = dict(request.get("runtime_context") or {})
        if runtime:
            lines.extend(["## Runtime", ""])
            for key in (
                "halo_forge_version",
                "python",
                "platform",
                "machine",
                "accelerator_backend",
            ):
                if runtime.get(key) is not None:
                    lines.append(f"- {key.replace('_', ' ')}: `{runtime[key]}`")
            lines.append("")

        notes = request.get("notes")
        if notes:
            lines.extend(["## Notes", "", str(notes), ""])
        lines.extend(
            [
                "## Reproducibility",
                "",
                f"Analysis hash: `{snapshot.content_hash}`",
                f"Bundle content hash: `{bundle.content_hash}`",
                "",
            ]
        )
        return "\n".join(lines)

    def execute_evidence_bundle(self, bundle_id: str) -> Any:
        bundle = self.db.get_evidence_bundle(bundle_id)
        if bundle is None:
            raise ValueError(f"unknown evidence bundle: {bundle_id}")
        if bundle.status == "completed":
            try:
                verify_evidence_bundle(bundle.storage_path, content_hash=bundle.content_hash)
            except RuntimeError as exc:
                self.db.update_evidence_bundle(
                    bundle.id,
                    status="corrupt",
                    error=f"checksum verification failed: {exc}",
                )
                raise EvidenceBundleExecutionError(str(exc)) from exc
            return bundle
        snapshot = self.db.get_cohort_analysis_snapshot(bundle.analysis_snapshot_id)
        if snapshot is None:
            raise EvidenceBundleExecutionError("evidence bundle analysis snapshot is missing")
        decision = (
            None
            if bundle.research_decision_id is None
            else self.db.get_research_decision(bundle.research_decision_id)
        )
        self.db.update_evidence_bundle(bundle.id, status="running", error=None)
        try:
            markdown = self._report_markdown(bundle, snapshot, decision)
            title = str(bundle.request.get("report_title") or "Halo Forge Research Evidence")
            observations = snapshot.request.get("observations") or []
            comparisons = snapshot.analysis.get("comparisons") or {}
            comparison_rows = [
                {
                    "candidate_subject_id": key,
                    "classification": value.get("classification"),
                    "matched_seed_count": value.get("matched_seed_count"),
                    "mean_delta": value.get("mean_delta"),
                    "interval_lower": (value.get("confidence_interval") or {}).get("lower"),
                    "interval_upper": (value.get("confidence_interval") or {}).get("upper"),
                }
                for key, value in sorted(comparisons.items())
            ]
            formats = set(_resolve_evidence_formats(bundle.request.get("formats")))
            files: Dict[str, Any] = {}
            if "markdown" in formats:
                files["report.md"] = markdown
            if "html" in formats:
                files["report.html"] = markdown_report_html(markdown, title=title)
            if "json" in formats:
                files["evidence.json"] = {
                    "bundle": {
                        "id": bundle.id,
                        "content_hash": bundle.content_hash,
                        "request": bundle.request,
                    },
                    "analysis": snapshot.to_dict(),
                    "decision": None if decision is None else decision.to_dict(),
                }
            if "csv" in formats:
                files["observations.csv"] = _csv_rows(
                    ("subject_id", "seed", "metric", "value", "evaluation_id"),
                    observations,
                )
                files["comparisons.csv"] = _csv_rows(
                    (
                        "candidate_subject_id",
                        "classification",
                        "matched_seed_count",
                        "mean_delta",
                        "interval_lower",
                        "interval_upper",
                    ),
                    comparison_rows,
                )
            if "svg" in formats:
                files["comparison.svg"] = comparison_interval_svg(comparisons)
            extra_files = bundle.request.get("extra_files") or {}
            if not isinstance(extra_files, Mapping):
                raise ValueError("evidence bundle extra_files must be a mapping")
            collisions = sorted(set(str(key) for key in extra_files).intersection(files))
            if collisions:
                raise ValueError(
                    "evidence extra_files cannot replace generated files: " + ", ".join(collisions)
                )
            files.update({str(key): value for key, value in extra_files.items()})
            manifest = {
                "schema_version": 1,
                "bundle_id": bundle.id,
                "created_at": bundle.created_at,
                "analysis_snapshot_id": bundle.analysis_snapshot_id,
                "analysis_hash": snapshot.content_hash,
                "research_decision_id": bundle.research_decision_id,
                "decision_hash": None if decision is None else decision.content_hash,
                "formats": sorted(formats),
                "request": bundle.request,
            }
            published = publish_evidence_bundle(
                bundle.storage_path,
                content_hash=bundle.content_hash,
                manifest=manifest,
                files=files,
            )
            completed = self.db.update_evidence_bundle(
                bundle.id,
                status="completed",
                storage_path=str(published.path),
                manifest=published.manifest,
                error=None,
            )
            assert completed is not None
            return completed
        except Exception as exc:
            self.db.update_evidence_bundle(bundle.id, status="failed", error=str(exc))
            if isinstance(exc, (ValueError, EvidenceBundleExecutionError)):
                raise
            raise EvidenceBundleExecutionError(str(exc)) from exc

    def build_evidence_bundle(self, bundle_id: str) -> Any:
        return self.execute_evidence_bundle(bundle_id)

    def list_evidence_bundles(
        self,
        *,
        status: Optional[str] = None,
        analysis_snapshot_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[Any]:
        return self.db.list_evidence_bundles(
            status=status,
            analysis_snapshot_id=analysis_snapshot_id,
            limit=limit,
        )

    def get_evidence_bundle(self, bundle_id: str) -> Any:
        return self.db.get_evidence_bundle(bundle_id)

    def execute_work_item(self, work_item: str | Mapping[str, Any] | Any) -> Dict[str, Any]:
        if isinstance(work_item, str):
            record = self.db.get_work_item(work_item)
            if record is None:
                raise ValueError(f"unknown work item: {work_item}")
            payload: Mapping[str, Any] = record.to_dict()
        elif isinstance(work_item, Mapping):
            payload = work_item
        elif hasattr(work_item, "to_dict"):
            payload = work_item.to_dict()
        else:
            raise TypeError("work_item must be an id, mapping, or work-item record")
        spec = payload.get("launch_spec") or payload.get("launch_spec_json") or payload
        if isinstance(spec, str):
            import json

            spec = json.loads(spec)
        if not isinstance(spec, Mapping):
            raise TypeError("adaptive work-item launch_spec must be a mapping")
        action = (
            str(spec.get("action") or spec.get("operation") or payload.get("kind") or "")
            .strip()
            .lower()
        )
        if action not in {
            "build_evidence_bundle",
            "adaptive_evidence_bundle",
            "evidence_bundle",
            "adaptive.evidence_bundle",
        }:
            raise ValueError(f"unsupported adaptive work-item action: {action!r}")
        bundle_id = (
            spec.get("evidence_bundle_id") or spec.get("bundle_id") or payload.get("domain_id")
        )
        if not bundle_id:
            raise ValueError("adaptive evidence work item requires evidence_bundle_id")
        result = self.execute_evidence_bundle(str(bundle_id))
        return result.to_dict()

    # ----- workspace drafts ---------------------------------------------

    def save_workspace_draft(
        self,
        *,
        draft_kind: str,
        content: Mapping[str, Any],
        owner_key: str = "local",
        name: str = "default",
        draft_id: Optional[str] = None,
        ttl_days: int = 30,
    ) -> Any:
        model = WorkspaceDraft(
            draft_kind=draft_kind,
            content=content,
            owner_key=owner_key,
            name=name,
        )
        return self.db.save_workspace_draft(
            draft_kind=model.draft_kind,
            content=model.content.to_dict(),
            owner_key=model.owner_key,
            name=model.name,
            draft_id=draft_id,
            ttl_days=ttl_days,
        )

    def discard_workspace_draft(self, draft_id: str) -> bool:
        return self.db.delete_workspace_draft(draft_id)

    def list_workspace_drafts(
        self, *, owner_key: str = "local", include_expired: bool = False
    ) -> list[Any]:
        return self.db.list_workspace_drafts(owner_key=owner_key, include_expired=include_expired)

    def get_workspace_draft(self, draft_id: str) -> Any:
        return self.db.get_workspace_draft(draft_id)


__all__ = ["AdaptiveLabError", "AdaptiveLabService", "EvidenceBundleExecutionError"]
