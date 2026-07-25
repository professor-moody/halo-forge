"""Persistent, transport-neutral experiment orchestration.

This module connects the deterministic run-group primitives to the SQLite
catalog and the workstation queue.  It intentionally does not import a
trainer, spawn a process, or run an evaluator: queue consumers remain free to
use the existing CLI, API, or dashboard launch transports.
"""

from __future__ import annotations

import math
import json
import statistics
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from halo_forge.orchestration.capabilities import (
    DEFAULT_TRAINER_EXECUTION_CAPABILITIES,
    TrainerExecutionCapability,
    TrainerExecutionCapabilityRegistry,
)
from halo_forge.orchestration.models import (
    CohortAggregate,
    CohortObservation,
    RunGroupSpec,
    aggregate_cohort,
    canonical_fingerprint,
    materialize_trials,
    rank_cohort,
)
from halo_forge.orchestration.policies import decide_successive_halving
from halo_forge.run_db import (
    BenchmarkSuiteRevisionRecord,
    RunDatabase,
    RunGroupRecord,
    RunGroupTrialRecord,
    TrialRunRecord,
    TrialSegmentRecord,
    WorkItemRecord,
)
from halo_forge.runtime_determinism import build_run_id
from halo_forge.workstation_jobs import WorkstationScheduler

_TERMINAL_WORK_STATUSES = frozenset({"completed", "failed", "cancelled"})
_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled", "pruned", "stopped"})


class ExperimentOrchestrationService:
    """Facade for repeat/sweep persistence, queueing, and result aggregation."""

    def __init__(
        self,
        database: RunDatabase,
        *,
        scheduler: Optional[WorkstationScheduler] = None,
        capabilities: Optional[TrainerExecutionCapabilityRegistry] = None,
    ) -> None:
        self.database = database
        self.scheduler = scheduler or WorkstationScheduler(database)
        self.capabilities = capabilities or DEFAULT_TRAINER_EXECUTION_CAPABILITIES

    # ----- creation -----------------------------------------------------

    def create_group_from_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Create and queue a run group from an API/CLI-shaped mapping."""

        if not isinstance(payload, Mapping):
            raise TypeError("run-group payload must be a mapping")
        values = dict(payload)
        raw_spec = dict(values.get("spec") or values)
        for key in ("checkpoint_policy_revision_id", "resolved_checkpoint_plan"):
            if key in values and key not in raw_spec:
                raw_spec[key] = values[key]
        if "version" not in raw_spec and (
            raw_spec.get("checkpoint_policy_revision_id")
            or raw_spec.get("resolved_checkpoint_plan") is not None
        ):
            raw_spec["version"] = 2
        trainer_mode = str(
            values.get("trainer_mode")
            or (raw_spec.get("base_config") or {}).get("trainer_mode")
            or ""
        ).strip()
        development_revision_id = str(
            values.get("development_suite_revision_id") or values.get("suite_revision_id") or ""
        ).strip()
        if not trainer_mode:
            raise ValueError("trainer_mode is required")
        if not development_revision_id:
            raise ValueError("development_suite_revision_id is required")
        dataset_bindings = values.get("dataset_bindings") or ()
        base_config = dict(raw_spec.get("base_config") or {})
        if not dataset_bindings and base_config.get("dataset_version_id"):
            dataset_bindings = (
                {
                    "role": "train",
                    "dataset_version_id": str(base_config["dataset_version_id"]),
                    "split": str(base_config.get("dataset_split") or "train"),
                },
            )
        created = self.create_run_group(
            raw_spec,
            trainer_mode=trainer_mode,
            development_suite_revision_id=development_revision_id,
            holdout_suite_revision_id=values.get("holdout_suite_revision_id"),
            dataset_bindings=dataset_bindings,
            base_subject=values.get("base_subject"),
            parent_group_id=values.get("parent_group_id"),
            priority=int(values.get("priority", 0)),
            max_retries=int(values.get("max_retries", 0)),
            sampled_params=values.get("sampled_params"),
            run_group_id=values.get("run_group_id"),
        )
        if values.get("source_trial_id") or values.get("source_config_hash"):
            group = self._require_group(created["id"])
            state = dict(group.sampler_state)
            state["fork_context"] = {
                "parent_group_id": values.get("parent_group_id"),
                "source_trial_id": values.get("source_trial_id"),
                "source_config_hash": values.get("source_config_hash"),
            }
            self.database.update_run_group(group.id, sampler_state=state)
            created = self.get_run_group_detail(group.id, reconcile=False)
        return created

    def create_run_group(
        self,
        spec: RunGroupSpec | Mapping[str, Any],
        *,
        trainer_mode: str,
        development_suite_revision_id: str,
        holdout_suite_revision_id: Optional[str] = None,
        dataset_bindings: Sequence[Mapping[str, Any]] = (),
        base_subject: Optional[Mapping[str, Any]] = None,
        parent_group_id: Optional[str] = None,
        priority: int = 0,
        max_retries: int = 0,
        sampled_params: Optional[Sequence[Mapping[str, Any]]] = None,
        run_group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist a deterministic materialization and enqueue its first rung.

        The development revision is the sole source of truth for objective
        name and direction.  The optional holdout revision is recorded for a
        later explicit evaluation and is never queued as an optimization
        dependency here.
        """

        development = self._require_suite_revision(
            development_suite_revision_id, purpose="development"
        )
        holdout = None
        if holdout_suite_revision_id:
            if holdout_suite_revision_id == development_suite_revision_id:
                raise ValueError("development and holdout suite revisions must be different")
            holdout = self._require_suite_revision(
                str(holdout_suite_revision_id), purpose="holdout"
            )
        resolved_spec = self._pin_spec_to_development_revision(spec, development)
        mode = str(trainer_mode).strip().lower()
        if not mode:
            raise ValueError("trainer_mode is required")
        backend = self._backend_from_config(resolved_spec.base_config)
        capability = self.capabilities.resolve(mode, backend)
        resolved_spec = self._resolve_checkpoint_spec(
            resolved_spec,
            development=development,
            trainer_mode=mode,
            capability=capability,
        )
        self._validate_capability(resolved_spec, capability)
        normalized_bindings = self._normalize_dataset_bindings(dataset_bindings)
        trials = materialize_trials(resolved_spec, sampled_params=sampled_params)

        group = self.database.create_run_group(
            name=resolved_spec.name,
            kind=resolved_spec.kind,
            trainer_mode=mode,
            resolved_launch_config=resolved_spec.base_config,
            dataset_bindings=normalized_bindings,
            base_subject=base_subject,
            development_suite_revision_id=development.id,
            holdout_suite_revision_id=holdout.id if holdout else None,
            search_space=resolved_spec.search_space.to_dict(),
            seeds=resolved_spec.seeds,
            budgets=self._resolved_group_budgets(resolved_spec, capability),
            sampler_state={
                "spec": resolved_spec.to_dict(),
                "capability": capability.to_dict(),
                "development_objective": {
                    "suite_revision_id": development.id,
                    "metric": development.primary_metric,
                    "direction": development.direction,
                },
                "sampled_params": [dict(value) for value in sampled_params or ()],
                "trial_keys": {},
                "checkpoint_policy_revision_id": (resolved_spec.checkpoint_policy_revision_id),
                "resolved_checkpoint_plan": deepcopy(resolved_spec.resolved_checkpoint_plan),
            },
            pruning_policy=resolved_spec.pruning.to_dict(),
            checkpoint_policy_revision_id=resolved_spec.checkpoint_policy_revision_id,
            resolved_checkpoint_plan=resolved_spec.resolved_checkpoint_plan,
            parent_group_id=parent_group_id,
            status="materializing",
            run_group_id=run_group_id,
        )

        enqueued: list[str] = []
        trial_keys: dict[str, str] = {}
        try:
            for materialized in trials:
                trial_id = self._stable_id(group.id, "trial", materialized.trial_key)
                trial_keys[trial_id] = materialized.trial_key
                trial = self.database.create_run_group_trial(
                    run_group_id=group.id,
                    ordinal=materialized.trial_index,
                    config_hash=materialized.config_fingerprint,
                    sampled_config=materialized.params,
                    required_seed_count=len(materialized.seeds),
                    status="queued",
                    trial_id=trial_id,
                )
                for materialized_run in materialized.materialize_runs():
                    canonical_run_id = build_run_id(mode)
                    trial_run_id = self._stable_id(
                        group.id,
                        "trial-run",
                        materialized.trial_key,
                        materialized_run.seed_index,
                        materialized_run.seed,
                    )
                    trial_run = self.database.create_trial_run(
                        trial_id=trial.id,
                        run_id=canonical_run_id,
                        ordinal=materialized_run.seed_index,
                        seed=materialized_run.seed,
                        status="materializing",
                        trial_run_id=trial_run_id,
                    )
                    queued = self._queue_segment(
                        group=group,
                        trial=trial,
                        trial_run=trial_run,
                        capability=capability,
                        resolved_config=materialized_run.resolved_config,
                        development=development,
                        ordinal=0,
                        start_value=0,
                        end_value=self._initial_segment_end(resolved_spec),
                        unit=self._initial_segment_unit(resolved_spec, capability),
                        priority=priority,
                        max_retries=max_retries,
                    )
                    enqueued.extend(queued)
            state = dict(group.sampler_state)
            state["trial_keys"] = trial_keys
            self.database.update_run_group(group.id, status="queued", sampler_state=state)
        except Exception:
            for work_item_id in enqueued:
                self.scheduler.cancel(work_item_id)
            self.database.update_run_group(group.id, status="failed")
            raise
        return self.get_run_group_detail(group.id, reconcile=False)

    def _queue_segment(
        self,
        *,
        group: RunGroupRecord,
        trial: RunGroupTrialRecord,
        trial_run: TrialRunRecord,
        capability: TrainerExecutionCapability,
        resolved_config: Mapping[str, Any],
        development: BenchmarkSuiteRevisionRecord,
        ordinal: int,
        start_value: int,
        end_value: int,
        unit: str,
        priority: int,
        max_retries: int,
        dependencies: Sequence[str] = (),
    ) -> list[str]:
        segment_id = self._stable_id(trial_run.id, "segment", ordinal)
        existing_segment = self.database.get_trial_segment(segment_id)
        if existing_segment is not None:
            if (
                existing_segment.trial_run_id != trial_run.id
                or existing_segment.ordinal != ordinal
                or existing_segment.unit != unit
                or existing_segment.start_value != start_value
                or existing_segment.end_value != end_value
            ):
                raise ValueError(f"segment identity collision for {segment_id}")
        run_config = deepcopy(dict(resolved_config))
        run_config.update(
            seed=trial_run.seed,
            run_id=trial_run.run_id,
            canonical_run_id=trial_run.run_id,
        )
        output_dir = self._managed_output_dir(run_config, trial_run.run_id)
        run_config["output_dir"] = output_dir
        checkpoint_plan = self._checkpoint_plan(group)
        checkpoint_policy_revision_id = self._checkpoint_policy_revision_id(group)
        command = self._training_command(
            trainer_mode=group.trainer_mode,
            config=run_config,
            bindings=group.dataset_bindings,
            unit=unit,
            start_value=start_value,
            end_value=end_value,
        )
        replay_context = {
            key: deepcopy(run_config.get(key))
            for key in (
                "study_id",
                "study_protocol_revision_id",
                "study_arm_id",
                "study_assignment_id",
                "study_assignment_ids_by_seed",
                "study_factor_values",
                "study_contrast_ids",
                "study_deviation_ids",
            )
            if run_config.get(key) not in (None, "", [], {})
        }
        train_spec = {
            "operation": "train_trial_segment",
            "run_group_id": group.id,
            "trial_id": trial.id,
            "trial_run_id": trial_run.id,
            "trial_segment_id": segment_id,
            "segment_ordinal": ordinal,
            "trainer_mode": group.trainer_mode,
            "capability": capability.to_dict(),
            "segment": {"unit": unit, "start": start_value, "end": end_value},
            "resolved_launch_config": run_config,
            "output_dir": output_dir,
            "env": {
                "HALO_FORGE_RUN_ID": trial_run.run_id,
                **(
                    {
                        "HALO_FORGE_OPERATIONAL_COMPLETION": json.dumps(
                            replay_context, sort_keys=True, default=str
                        )
                    }
                    if replay_context
                    else {}
                ),
            },
            "dataset_bindings": group.dataset_bindings,
            "base_subject": group.base_subject,
            "checkpoint_policy_revision_id": checkpoint_policy_revision_id,
            "checkpoint_plan_hash": checkpoint_plan.get("content_hash"),
        }
        if command is not None:
            train_spec["command"] = command
            train_spec["command_transport"] = {"status": "ready", "version": 1}
        else:
            command_template = self._training_command(
                trainer_mode=group.trainer_mode,
                config=run_config,
                bindings=group.dataset_bindings,
                unit=unit,
                start_value=0,
                end_value=end_value,
            )
            train_spec["command_transport"] = {
                "status": (
                    "requires_checkpoint_resolution"
                    if start_value and capability.resume_cli_flag
                    else "requires_segment_adapter"
                ),
                "version": 1,
                "reason": (
                    "The current trainer CLI cannot express this bounded resumable segment; "
                    "a capability-specific worker must resolve it."
                ),
                "command_template": command_template,
                "checkpoint_resolution": {
                    "from_previous_segment": ordinal - 1,
                    "checkpoint_pattern": capability.checkpoint_pattern,
                    "append_flag": capability.resume_cli_flag,
                },
            }
        training_work_item_id = self._stable_id(segment_id, "training")
        training = self.database.get_work_item(training_work_item_id)
        if training is None:
            training = self.scheduler.enqueue(
                kind="training",
                launch_spec=train_spec,
                resource_class="accelerator",
                priority=priority,
                canonical_run_id=trial_run.run_id,
                dependencies=dependencies,
                max_retries=max_retries,
                run_group_id=group.id,
                work_item_id=training_work_item_id,
            )
        elif (
            training.kind != "training"
            or training.launch_spec.get("trial_segment_id") != segment_id
        ):
            raise ValueError(f"training work identity collision for {training_work_item_id}")
        if existing_segment is None:
            self.database.create_trial_segment(
                trial_run_id=trial_run.id,
                ordinal=ordinal,
                unit=unit,
                start_value=start_value,
                end_value=end_value,
                work_item_id=training.id,
                status="queued",
                segment_id=segment_id,
            )
        required_revision_ids = [development.id]
        for revision_id in checkpoint_plan.get("required_suite_revision_ids") or ():
            revision_id = str(revision_id)
            if revision_id not in required_revision_ids:
                required_revision_ids.append(revision_id)
        evaluation_ids: list[str] = []
        for revision_id in required_revision_ids:
            revision = (
                development
                if revision_id == development.id
                else self._require_suite_revision(revision_id, purpose="development")
            )
            evaluation_spec = {
                "operation": "evaluate_trial_segment",
                "run_group_id": group.id,
                "trial_id": trial.id,
                "trial_run_id": trial_run.id,
                "trial_segment_id": segment_id,
                "segment_ordinal": ordinal,
                "canonical_run_id": trial_run.run_id,
                "suite_revision_id": revision.id,
                "suite_role": "primary" if revision.id == development.id else "guardrail",
                "metric": revision.primary_metric,
                "direction": revision.direction,
                "checkpoint_policy_revision_id": checkpoint_policy_revision_id,
                "checkpoint_plan_hash": checkpoint_plan.get("content_hash"),
                "subject": {
                    "type": "checkpoint" if unit != "full_trial" else "run",
                    "ref": trial_run.run_id,
                    "resolve_from_segment": segment_id,
                },
            }
            evaluation_command = self._evaluation_command(
                suite_revision_id=revision.id,
                run_id=trial_run.run_id,
                segment_id=segment_id,
                unit=unit,
            )
            if evaluation_command is not None:
                evaluation_spec["command"] = evaluation_command
                evaluation_spec["command_transport"] = {"status": "ready", "version": 1}
            else:
                evaluation_spec["command_transport"] = {
                    "status": "requires_checkpoint_resolution",
                    "version": 1,
                    "reason": (
                        "The checkpoint artifact path is produced by the training dependency and "
                        "must be resolved from trial_segment_id before launching eval run."
                    ),
                    "command_prefix": [
                        sys.executable,
                        "-m",
                        "halo_forge.cli",
                        "eval",
                        "run",
                        "--suite-revision",
                        revision.id,
                        "--subject-type",
                        "checkpoint",
                    ],
                }
            evaluation_work_item_id = self._stable_id(segment_id, "evaluation", revision.id)
            evaluation = self.database.get_work_item(evaluation_work_item_id)
            if evaluation is None:
                evaluation = self.scheduler.enqueue(
                    kind="evaluation",
                    launch_spec=evaluation_spec,
                    resource_class="accelerator",
                    priority=priority,
                    canonical_run_id=trial_run.run_id,
                    dependencies=(training.id,),
                    max_retries=max_retries,
                    run_group_id=group.id,
                    work_item_id=evaluation_work_item_id,
                )
            elif (
                evaluation.kind != "evaluation"
                or evaluation.launch_spec.get("trial_segment_id") != segment_id
                or evaluation.launch_spec.get("suite_revision_id") != revision.id
            ):
                raise ValueError(
                    f"evaluation work identity collision for {evaluation_work_item_id}"
                )
            evaluation_ids.append(evaluation.id)
        self.database.update_trial_run(trial_run.id, status="queued", work_item_id=training.id)
        return [training.id, *evaluation_ids]

    # ----- views and reconciliation ------------------------------------

    def list_run_groups(
        self,
        *,
        kind: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        reconcile: bool = True,
    ) -> list[Dict[str, Any]]:
        groups = self.database.list_run_groups(kind=kind, status=status, limit=limit)
        return [self._group_summary(group, reconcile=reconcile) for group in groups]

    def get_run_group_detail(self, run_group_id: str, *, reconcile: bool = True) -> Dict[str, Any]:
        group = self._require_group(run_group_id)
        if reconcile:
            group = self.reconcile_group(run_group_id)
        development = self._require_revision(group.development_suite_revision_id)
        work_items = self._group_work_items(group.id)
        work_by_id = {item.id: item for item in work_items}
        trial_rows = self.database.list_run_group_trials(group.id)
        objective_boundary = self._latest_common_objective_boundary(
            group, trial_rows, work_items, development
        )
        aggregates, observations = self._cohort_state(group, trial_rows, work_items, development)
        aggregate_by_trial = {row.trial_key: row for row in aggregates}
        trials: list[Dict[str, Any]] = []
        for trial in trial_rows:
            runs = []
            for trial_run in self.database.list_trial_runs(trial.id):
                segments = self.database.list_trial_segments(trial_run.id)
                related = [
                    item.to_dict()
                    for item in work_items
                    if item.launch_spec.get("trial_run_id") == trial_run.id
                ]
                run_value = self._objective_for_run(
                    group,
                    trial_run,
                    development,
                    work_items,
                    next(
                        (
                            segment.id
                            for segment in segments
                            if objective_boundary is not None
                            and segment.ordinal == objective_boundary["ordinal"]
                        ),
                        None,
                    ),
                )
                if objective_boundary is None:
                    run_value = None
                runs.append(
                    {
                        **trial_run.to_dict(),
                        "objective_value": run_value,
                        "objective_boundary": objective_boundary,
                        "segments": [segment.to_dict() for segment in segments],
                        "work_items": related,
                    }
                )
            resolved_config = dict(group.resolved_launch_config)
            resolved_config.update(trial.sampled_config)
            trials.append(
                {
                    **trial.to_dict(),
                    "resolved_config": resolved_config,
                    "runs": runs,
                    "cohort": aggregate_by_trial[trial.id].to_dict(),
                }
            )
        ranking = self._rank_active_trials(aggregates, trial_rows, direction=development.direction)
        best = self._best_from_rows(trial_rows, ranking)
        checkpoint_policy_revision_id = self._checkpoint_policy_revision_id(group)
        resolved_checkpoint_plan = self._checkpoint_plan(group) or None
        objective = {
            "suite_revision_id": development.id,
            "metric": development.primary_metric,
            "direction": development.direction,
        }
        if objective_boundary is not None:
            objective["boundary"] = objective_boundary
        return {
            **group.to_dict(),
            "objective": objective,
            "capability": group.sampler_state.get("capability", {}),
            "checkpoint_policy_revision_id": checkpoint_policy_revision_id,
            "resolved_checkpoint_plan": resolved_checkpoint_plan,
            "trials": trials,
            "cohort_aggregates": [row.to_dict() for row in aggregates],
            "ranking": [row.to_dict() for row in ranking],
            "best_trial": best,
            "work_items": [item.to_dict() for item in work_items],
            "artifacts": [
                value.to_dict()
                for value in self.database.list_model_artifacts(run_group_id=group.id)
            ],
            "exposures": [
                value.to_dict() for value in self.database.list_exposures(run_group_id=group.id)
            ],
            "observation_count": len(observations),
            "work_item_count": len(work_by_id),
            "trajectory": self.get_checkpoint_trajectory(
                group.id,
                reconcile=False,
                work_items=work_items,
            ),
        }

    def get_checkpoint_trajectory(
        self,
        run_group_id: str,
        *,
        reconcile: bool = True,
        work_items: Optional[Sequence[WorkItemRecord]] = None,
    ) -> Dict[str, Any]:
        """Return the ordered checkpoint/evaluation/gate history for a group."""

        group = (
            self.reconcile_group(run_group_id) if reconcile else self._require_group(run_group_id)
        )
        items = list(work_items) if work_items is not None else self._group_work_items(group.id)
        gates_by_segment: dict[str, list[Dict[str, Any]]] = {}
        for gate in self._list_gate_decisions(group.id):
            payload = gate.to_dict() if hasattr(gate, "to_dict") else dict(gate)
            segment_id = str(payload.get("trial_segment_id") or payload.get("segment_id") or "")
            if segment_id:
                gates_by_segment.setdefault(segment_id, []).append(payload)

        runs: list[Dict[str, Any]] = []
        for trial in self.database.list_run_group_trials(group.id):
            for trial_run in self.database.list_trial_runs(trial.id):
                points: list[Dict[str, Any]] = []
                for segment in self.database.list_trial_segments(trial_run.id):
                    training = self._segment_work(items, segment, "training")
                    evaluations = [
                        item
                        for item in items
                        if item.kind == "evaluation"
                        and item.launch_spec.get("trial_segment_id") == segment.id
                    ]
                    artifact = (
                        self.database.get_model_artifact(segment.checkpoint_artifact_id)
                        if segment.checkpoint_artifact_id
                        else None
                    )
                    occurrence_id = None
                    if artifact is not None:
                        occurrence = self.database._conn.execute(
                            "SELECT id FROM artifact_occurrences "
                            "WHERE legacy_model_artifact_id = ? "
                            "ORDER BY created_at DESC LIMIT 1",
                            (artifact.id,),
                        ).fetchone()
                        occurrence_id = str(occurrence["id"]) if occurrence else None
                    gates = gates_by_segment.get(segment.id, [])
                    points.append(
                        {
                            **segment.to_dict(),
                            "training_work_item": training.to_dict() if training else None,
                            "evaluation_work_items": [value.to_dict() for value in evaluations],
                            "evaluation_ids": [
                                str(value.result.get("evaluation_id"))
                                for value in evaluations
                                if value.result.get("evaluation_id")
                            ],
                            "artifact": artifact.to_dict() if artifact else None,
                            "artifact_occurrence_id": occurrence_id,
                            "gate_decisions": gates,
                            "latest_gate_decision": gates[-1] if gates else None,
                        }
                    )
                runs.append(
                    {
                        "trial_id": trial.id,
                        "trial_run_id": trial_run.id,
                        "run_id": trial_run.run_id,
                        "seed": trial_run.seed,
                        "status": trial_run.status,
                        "points": points,
                    }
                )
        return {
            "run_group_id": group.id,
            "checkpoint_policy_revision_id": self._checkpoint_policy_revision_id(group),
            "resolved_checkpoint_plan": self._checkpoint_plan(group) or None,
            "runs": runs,
        }

    def reconcile_group(self, run_group_id: str) -> RunGroupRecord:
        """Project durable work/evaluation outcomes onto trial/group status."""

        group = self._require_group(run_group_id)
        development = self._require_revision(group.development_suite_revision_id)
        work_items = self._group_work_items(group.id)
        trial_rows = self.database.list_run_group_trials(group.id)
        checkpoint_plan = self._checkpoint_plan(group)
        checkpoint_boundaries = tuple(checkpoint_plan.get("boundaries") or ())
        gates_by_segment: dict[str, Dict[str, Any]] = {}
        for gate in self._list_gate_decisions(group.id):
            payload = gate.to_dict() if hasattr(gate, "to_dict") else dict(gate)
            segment_id = str(payload.get("trial_segment_id") or payload.get("segment_id") or "")
            if segment_id:
                gates_by_segment[segment_id] = payload

        for trial in trial_rows:
            run_statuses: list[str] = []
            for trial_run in self.database.list_trial_runs(trial.id):
                segments = self.database.list_trial_segments(trial_run.id)
                latest = segments[-1] if segments else None
                training = self._segment_work(work_items, latest, "training")
                evaluation = self._segment_work(work_items, latest, "evaluation")
                gate = gates_by_segment.get(latest.id) if latest else None
                gate_action = str((gate or {}).get("action") or "")
                if (
                    latest
                    and training
                    and latest.status not in {"awaiting_review", "stopped"}
                    and latest.status != training.status
                ):
                    segment_status = (
                        "completed" if training.status == "completed" else training.status
                    )
                    if segment_status in {
                        "queued",
                        "running",
                        "completed",
                        "failed",
                        "cancelled",
                        "interrupted",
                    }:
                        self.database.update_trial_segment(latest.id, status=segment_status)
                        latest = self.database.get_trial_segment(latest.id) or latest
                value = self._objective_for_run(
                    group,
                    trial_run,
                    development,
                    work_items,
                    latest.id if latest else None,
                )
                if gate_action == "pause":
                    status = "awaiting_review"
                elif gate_action == "stop":
                    status = "stopped"
                elif value is not None:
                    status = (
                        "awaiting_decision"
                        if (group.pruning_policy.get("enabled") or checkpoint_boundaries)
                        and latest is not None
                        and (
                            latest.decision is None
                            or (
                                checkpoint_boundaries
                                and latest.ordinal < len(checkpoint_boundaries) - 1
                                and gate_action != "continue"
                            )
                        )
                        else "completed"
                    )
                elif training and training.status in {"failed", "cancelled", "interrupted"}:
                    status = training.status
                elif evaluation and evaluation.status in {"failed", "cancelled", "interrupted"}:
                    status = evaluation.status
                elif (training and training.status == "running" and training.cancel_requested) or (
                    evaluation and evaluation.status == "running" and evaluation.cancel_requested
                ):
                    status = "cancelling"
                elif (training and training.status == "running") or (
                    evaluation and evaluation.status == "running"
                ):
                    status = "running"
                elif trial_run.status in {"pruned", "stopped", "awaiting_review"}:
                    status = trial_run.status
                else:
                    status = "queued"
                self.database.update_trial_run(
                    trial_run.id,
                    status=status,
                    work_item_id=training.id if training else None,
                )
                run_statuses.append(status)

            aggregate_rows, _ = self._cohort_state(group, [trial], work_items, development)
            aggregate = aggregate_rows[0]
            if "awaiting_review" in run_statuses:
                trial_status = "awaiting_review"
            elif "stopped" in run_statuses:
                trial_status = "stopped"
            elif "cancelling" in run_statuses:
                trial_status = "cancelling"
            elif "running" in run_statuses:
                trial_status = "running"
            elif "awaiting_decision" in run_statuses:
                trial_status = "awaiting_decision"
            elif "queued" in run_statuses:
                trial_status = "queued"
            elif aggregate.eligible and all(value == "completed" for value in run_statuses):
                trial_status = "completed"
            elif run_statuses and all(value in _TERMINAL_RUN_STATUSES for value in run_statuses):
                if all(value == "pruned" for value in run_statuses):
                    trial_status = "pruned"
                elif all(value == "cancelled" for value in run_statuses):
                    trial_status = "cancelled"
                else:
                    trial_status = "failed"
            else:
                trial_status = "queued"
            self.database.update_run_group_trial(
                trial.id,
                status=trial_status,
                objective_metric=development.primary_metric,
                objective_direction=development.direction,
                objective_value=aggregate.mean,
                seed_coverage=aggregate.completed_count,
            )

        refreshed_trials = self.database.list_run_group_trials(group.id)
        statuses = [trial.status for trial in refreshed_trials]
        if statuses and all(value == "stopped" for value in statuses):
            group_status = "stopped"
        elif statuses and all(value in {"completed", "pruned", "stopped"} for value in statuses):
            group_status = "completed"
        elif statuses and all(value == "cancelled" for value in statuses):
            group_status = "cancelled"
        elif (
            statuses
            and all(value in {"failed", "cancelled", "pruned", "completed"} for value in statuses)
            and any(value == "failed" for value in statuses)
        ):
            group_status = "failed"
        elif any(value == "cancelling" for value in statuses):
            group_status = "cancelling"
        elif any(value == "running" for value in statuses):
            group_status = "running"
        elif any(value == "awaiting_review" for value in statuses):
            group_status = "awaiting_review"
        elif any(value == "awaiting_decision" for value in statuses):
            group_status = "awaiting_decision"
        else:
            group_status = "queued"
        return self.database.update_run_group(group.id, status=group_status) or group

    # ----- lifecycle and halving ---------------------------------------

    def cancel_run_group(self, run_group_id: str) -> Dict[str, Any]:
        group = self._require_group(run_group_id)
        items = self._group_work_items(group.id)
        still_running = False
        for item in items:
            if item.status not in _TERMINAL_WORK_STATUSES:
                updated = self.scheduler.cancel(item.id)
                still_running = still_running or bool(updated and updated.status == "running")
        for trial in self.database.list_run_group_trials(group.id):
            for trial_run in self.database.list_trial_runs(trial.id):
                if trial_run.status not in _TERMINAL_RUN_STATUSES:
                    self.database.update_trial_run(trial_run.id, status="cancelled")
                for segment in self.database.list_trial_segments(trial_run.id):
                    if segment.status not in _TERMINAL_RUN_STATUSES:
                        self.database.update_trial_segment(segment.id, status="cancelled")
            self.database.update_run_group_trial(trial.id, status="cancelled")
        self.database.update_run_group(
            group.id, status="cancelling" if still_running else "cancelled"
        )
        return self.get_run_group_detail(group.id, reconcile=False)

    def resume_run_group(
        self, run_group_id: str, *, reason: Optional[str] = None
    ) -> Dict[str, Any]:
        group = self._require_group(run_group_id)
        if group.status in {"awaiting_review", "stopped"}:
            raise ValueError(
                "checkpoint-gated groups require a reviewed gate override with a recorded reason"
            )
        retried = []
        retry_reason = None if reason is None else str(reason).strip()
        if reason is not None and not retry_reason:
            raise ValueError("resume reason cannot be empty")
        items = self._group_work_items(group.id)
        if any(item.status == "running" for item in items):
            raise ValueError("cannot resume while run-group work is still running")
        for item in items:
            if item.status in {"failed", "interrupted", "cancelled"}:
                changed = self.scheduler.retry(
                    item.id,
                    **({"reason": retry_reason} if retry_reason is not None else {}),
                )
                if changed is not None:
                    retried.append(changed.id)
        if not retried:
            raise ValueError("run group has no failed, interrupted, or cancelled work to resume")
        for trial in self.database.list_run_group_trials(group.id):
            self.database.update_run_group_trial(trial.id, status="queued")
            for trial_run in self.database.list_trial_runs(trial.id):
                self.database.update_trial_run(trial_run.id, status="queued")
                for segment in self.database.list_trial_segments(trial_run.id):
                    if segment.status in {"failed", "interrupted", "cancelled"}:
                        self.database.update_trial_segment(segment.id, status="queued")
        self.database.update_run_group(group.id, status="queued")
        detail = self.get_run_group_detail(group.id, reconcile=False)
        detail["retried_work_item_ids"] = retried
        return detail

    def advance_successive_halving(self, run_group_id: str, *, rung_index: int) -> Dict[str, Any]:
        """Decide one synchronous rung and queue the promoted next segments."""

        group = self.reconcile_group(run_group_id)
        spec = RunGroupSpec.from_dict(group.sampler_state.get("spec") or {})
        if group.kind == "repeat" or not spec.pruning.enabled:
            raise ValueError("successive halving is only available for pruning-enabled sweeps")
        development = self._require_revision(group.development_suite_revision_id)
        capability = self.capabilities.resolve(
            group.trainer_mode, self._backend_from_config(group.resolved_launch_config)
        )
        self._validate_capability(spec, capability)
        trials = self.database.list_run_group_trials(group.id)
        if any(trial.status == "awaiting_review" for trial in trials):
            raise ValueError("successive halving is waiting for checkpoint gate review")
        if self._checkpoint_plan(group):
            waiting_for_gates: list[str] = []
            for trial in trials:
                if trial.status in {"pruned", "cancelled", "failed", "stopped"}:
                    continue
                for trial_run in self.database.list_trial_runs(trial.id):
                    segments = self.database.list_trial_segments(trial_run.id)
                    if not segments:
                        waiting_for_gates.append(trial_run.id)
                        continue
                    gate = self._latest_gate_for_segment(group.id, segments[-1].id)
                    if gate is None or gate.get("action") != "continue":
                        waiting_for_gates.append(trial_run.id)
            if waiting_for_gates:
                return {
                    "ready": False,
                    "reason": "waiting_for_checkpoint_gates",
                    "waiting_trial_run_ids": sorted(waiting_for_gates),
                    "queued_work_item_ids": [],
                }
        work_items = self._group_work_items(group.id)
        aggregates, _ = self._cohort_state(group, trials, work_items, development)
        active = [
            trial.id
            for trial in trials
            if trial.status not in {"pruned", "cancelled", "failed", "stopped", "awaiting_review"}
        ]
        decision = decide_successive_halving(
            spec.pruning,
            aggregates,
            direction=development.direction,
            rung_index=rung_index,
            active_trial_keys=active,
        )
        response = decision.to_dict()
        response["queued_work_item_ids"] = []
        if not decision.ready:
            return response

        by_id = {trial.id: trial for trial in trials}
        for trial_id in decision.pruned_trial_keys:
            trial = by_id[trial_id]
            self.database.update_run_group_trial(trial.id, status="pruned")
            for trial_run in self.database.list_trial_runs(trial.id):
                self.database.update_trial_run(trial_run.id, status="pruned")
                segments = self.database.list_trial_segments(trial_run.id)
                if segments:
                    self.database.update_trial_segment(
                        segments[-1].id,
                        status="completed",
                        decision="prune",
                        decision_reason=f"successive-halving rung {rung_index}",
                    )

        for trial_id in decision.promoted_trial_keys:
            trial = by_id[trial_id]
            for trial_run in self.database.list_trial_runs(trial.id):
                segments = self.database.list_trial_segments(trial_run.id)
                current = segments[-1]
                if decision.next_budget is None:
                    self.database.update_trial_segment(
                        current.id,
                        status="completed",
                        decision="complete",
                        decision_reason="final successive-halving budget complete",
                    )
                    self.database.update_trial_run(trial_run.id, status="completed")
                    continue
                if len(segments) > rung_index + 1:
                    continue
                evaluations = [
                    item
                    for item in work_items
                    if item.kind == "evaluation"
                    and item.launch_spec.get("trial_segment_id") == current.id
                ]
                if not evaluations or any(item.status != "completed" for item in evaluations):
                    raise ValueError(
                        f"trial run {trial_run.id} has no completed development evaluation "
                        f"for rung {rung_index}"
                    )
                if self._checkpoint_plan(group):
                    gate = self._latest_gate_for_segment(group.id, current.id)
                    if gate is None or gate.get("action") != "continue":
                        raise ValueError(
                            f"trial run {trial_run.id} has not passed its checkpoint gate"
                        )
                self.database.update_trial_segment(
                    current.id,
                    status="completed",
                    decision="continue",
                    decision_reason=f"promoted at successive-halving rung {rung_index}",
                )
                resolved = dict(group.resolved_launch_config)
                resolved.update(trial.sampled_config)
                resolved["seed"] = trial_run.seed
                queued = self._queue_segment(
                    group=group,
                    trial=trial,
                    trial_run=trial_run,
                    capability=capability,
                    resolved_config=resolved,
                    development=development,
                    ordinal=rung_index + 1,
                    start_value=int(decision.budget or current.end_value),
                    end_value=int(decision.next_budget),
                    unit=capability.segment_unit,
                    priority=max(value.priority for value in evaluations),
                    max_retries=max(value.max_retries for value in evaluations),
                    dependencies=tuple(value.id for value in evaluations),
                )
                response["queued_work_item_ids"].extend(queued)
            self.database.update_run_group_trial(
                trial.id,
                status="completed" if decision.next_budget is None else "queued",
            )
        self.database.update_run_group(
            group.id,
            status="completed" if decision.next_budget is None else "queued",
        )
        return response

    def advance_ready_successive_halving(self, run_group_id: str) -> Dict[str, Any]:
        """Advance the one eligible undecided rung, if its evidence is complete.

        This is the terminal-work-event counterpart to the operator-facing
        ``advance_successive_halving`` method. It deliberately performs the
        same policy decision and only discovers the rung automatically. The
        current segment's immutable decision (or the already-created next
        segment) is the idempotency marker, so repeated completion callbacks
        cannot enqueue the same rung twice.
        """

        group = self.reconcile_group(run_group_id)
        spec = RunGroupSpec.from_dict(group.sampler_state.get("spec") or {})
        if group.kind == "repeat" or not spec.pruning.enabled:
            return {
                "run_group_id": group.id,
                "ready": False,
                "advanced": False,
                "reason": "successive halving is not enabled",
            }
        if group.status == "awaiting_review":
            return {
                "run_group_id": group.id,
                "ready": False,
                "advanced": False,
                "reason": "checkpoint gate review is required",
            }

        work_items = self._group_work_items(group.id)
        undecided_ordinals: set[int] = set()
        active_run_count = 0
        for trial in self.database.list_run_group_trials(group.id):
            if trial.status in {"pruned", "cancelled", "failed", "stopped", "awaiting_review"}:
                continue
            for trial_run in self.database.list_trial_runs(trial.id):
                if trial_run.status in {
                    "pruned",
                    "cancelled",
                    "failed",
                    "stopped",
                    "awaiting_review",
                }:
                    continue
                segments = self.database.list_trial_segments(trial_run.id)
                if not segments:
                    return {
                        "run_group_id": group.id,
                        "ready": False,
                        "advanced": False,
                        "reason": f"trial run {trial_run.id} has no segment",
                    }
                active_run_count += 1
                current = segments[-1]
                if current.decision is None:
                    development_evaluations = [
                        item
                        for item in work_items
                        if item.kind == "evaluation"
                        and item.launch_spec.get("trial_segment_id") == current.id
                        and item.launch_spec.get("suite_revision_id")
                        == group.development_suite_revision_id
                    ]
                    if not development_evaluations or any(
                        item.status != "completed" for item in development_evaluations
                    ):
                        return {
                            "run_group_id": group.id,
                            "ready": False,
                            "advanced": False,
                            "reason": "waiting for current-rung development evaluations",
                            "trial_segment_id": current.id,
                        }
                    undecided_ordinals.add(int(current.ordinal))

        if active_run_count == 0 or not undecided_ordinals:
            return {
                "run_group_id": group.id,
                "ready": False,
                "advanced": False,
                "reason": "no undecided active rung",
            }
        if len(undecided_ordinals) != 1:
            return {
                "run_group_id": group.id,
                "ready": False,
                "advanced": False,
                "reason": "active trial runs are on different rungs",
                "rungs": sorted(undecided_ordinals),
            }

        rung_index = next(iter(undecided_ordinals))
        result = self.advance_successive_halving(
            group.id,
            rung_index=rung_index,
        )
        result.update(
            run_group_id=group.id,
            rung_index=rung_index,
            advanced=bool(result.get("ready")),
        )
        return result

    def advance_ready_checkpoint_policy(
        self,
        run_group_id: str,
        *,
        trial_segment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate ready checkpoint gates and durably queue continuations.

        Evaluation completion callbacks may invoke this more than once.  The
        gate idempotency key and stable next-segment/work identifiers make
        those callbacks safe across retries and process restarts.
        """

        group = self._require_group(run_group_id)
        plan_payload = self._checkpoint_plan(group)
        policy_revision_id = self._checkpoint_policy_revision_id(group)
        if not plan_payload or not policy_revision_id:
            return {
                "run_group_id": group.id,
                "enabled": False,
                "advanced": False,
                "reason": "checkpoint policy is not enabled",
            }

        from halo_forge.adaptive_lab import (
            AdaptiveLabService,
            ResolvedCheckpointPlan,
        )

        policy_record = self.database.get_checkpoint_policy_revision(policy_revision_id)
        if policy_record is None:
            raise ValueError(f"unknown checkpoint policy revision: {policy_revision_id}")
        policy = self._checkpoint_policy_model(policy_record)
        plan = ResolvedCheckpointPlan.from_dict(plan_payload)
        adaptive = AdaptiveLabService(self.database)
        work_items = self._group_work_items(group.id)

        candidates: list[tuple[RunGroupTrialRecord, TrialRunRecord, TrialSegmentRecord]] = []
        for trial in self.database.list_run_group_trials(group.id):
            for trial_run in self.database.list_trial_runs(trial.id):
                for segment in self.database.list_trial_segments(trial_run.id):
                    if trial_segment_id and segment.id != trial_segment_id:
                        continue
                    if segment.ordinal >= len(plan.boundaries):
                        continue
                    candidates.append((trial, trial_run, segment))

        outcomes: list[Dict[str, Any]] = []
        queued_ids: list[str] = []
        for trial, trial_run, segment in candidates:
            outcome = self._advance_checkpoint_segment(
                group=group,
                trial=trial,
                trial_run=trial_run,
                segment=segment,
                policy=policy,
                plan=plan,
                adaptive=adaptive,
                work_items=work_items,
            )
            outcomes.append(outcome)
            queued_ids.extend(outcome.get("queued_work_item_ids") or ())

        self.reconcile_group(group.id)
        return {
            "run_group_id": group.id,
            "enabled": True,
            "advanced": any(value.get("advanced") for value in outcomes),
            "outcomes": outcomes,
            "queued_work_item_ids": queued_ids,
        }

    def _advance_checkpoint_segment(
        self,
        *,
        group: RunGroupRecord,
        trial: RunGroupTrialRecord,
        trial_run: TrialRunRecord,
        segment: TrialSegmentRecord,
        policy: Any,
        plan: Any,
        adaptive: Any,
        work_items: Sequence[WorkItemRecord],
    ) -> Dict[str, Any]:
        existing = self._latest_gate_for_segment(group.id, segment.id)
        if existing is not None:
            gate_record = self.database.get_checkpoint_gate_decision(str(existing["id"]))
            recovered: Dict[str, Any] = {}
            if gate_record is not None:
                recovered = self._apply_checkpoint_gate_action(
                    group=group,
                    trial=trial,
                    trial_run=trial_run,
                    segment=segment,
                    plan=plan,
                    gate=gate_record,
                    work_items=work_items,
                )
            return {
                "trial_segment_id": segment.id,
                "ready": True,
                "advanced": False,
                "recovered": bool(gate_record),
                "reason": "checkpoint boundary already has a gate decision",
                "gate_decision": existing,
                **recovered,
            }

        current_metrics, missing, pending, evidence_ids = self._segment_metrics(
            segment,
            work_items=work_items,
            required_suite_revision_ids=plan.required_suite_revision_ids,
            development_suite_revision_id=policy.development_suite_revision_id,
        )
        if pending:
            return {
                "trial_segment_id": segment.id,
                "ready": False,
                "advanced": False,
                "reason": "required checkpoint evaluations are not terminal",
                "pending_suite_revision_ids": pending,
            }

        previous_metrics: Optional[Mapping[str, float]] = None
        best_metrics: Optional[Mapping[str, float]] = None
        plateau_counts: Dict[str, int] = {}
        previous_segments = [
            value
            for value in self.database.list_trial_segments(trial_run.id)
            if value.ordinal < segment.ordinal
        ]
        history: list[Mapping[str, float]] = []
        for previous in previous_segments:
            metrics, _, pending_previous, _ = self._segment_metrics(
                previous,
                work_items=work_items,
                required_suite_revision_ids=plan.required_suite_revision_ids,
                development_suite_revision_id=policy.development_suite_revision_id,
            )
            if not pending_previous and metrics:
                history.append(metrics)
        if history:
            previous_metrics = history[-1]
            best_metrics = self._best_metric_history(history, policy)
        plateau_best, plateau_counts = self._checkpoint_plateau_state(
            history,
            current_metrics,
            policy,
        )
        if plateau_best:
            best_metrics = {**dict(best_metrics or {}), **plateau_best}

        decision = adaptive.evaluate_gate(
            policy,
            plan,
            boundary_index=segment.ordinal,
            current_metrics=current_metrics,
            baseline_metrics=self._baseline_metrics(group, policy),
            previous_metrics=previous_metrics,
            best_metrics=best_metrics,
            missing_evidence=missing,
            plateau_counts=plateau_counts,
        )
        occurrence_id = self._checkpoint_occurrence_id(segment.id)
        idempotency_key = canonical_fingerprint(
            {
                "run_group_id": group.id,
                "trial_run_id": trial_run.id,
                "trial_segment_id": segment.id,
                "plan_hash": plan.content_hash,
                "boundary_index": segment.ordinal,
                "evaluation_ids": evidence_ids,
            }
        )
        gate = self.database.create_checkpoint_gate_decision(
            policy_revision_id=plan.policy_revision_id,
            plan_hash=plan.content_hash,
            boundary_index=segment.ordinal,
            action=decision.action,
            reasons=decision.reasons,
            evidence=decision.evidence.to_dict(),
            idempotency_key=idempotency_key,
            automatic=decision.automatic,
            run_group_id=group.id,
            trial_run_id=trial_run.id,
            trial_segment_id=segment.id,
            checkpoint_occurrence_id=occurrence_id,
            content_hash=decision.content_hash,
            decision_id=self._stable_id(segment.id, "checkpoint-gate", plan.content_hash),
        )
        applied = self._apply_checkpoint_gate_action(
            group=group,
            trial=trial,
            trial_run=trial_run,
            segment=segment,
            plan=plan,
            gate=gate,
            work_items=work_items,
        )
        return {
            "trial_segment_id": segment.id,
            "ready": True,
            "advanced": True,
            "gate_decision": gate.to_dict(),
            **applied,
        }

    def _apply_checkpoint_gate_action(
        self,
        *,
        group: RunGroupRecord,
        trial: RunGroupTrialRecord,
        trial_run: TrialRunRecord,
        segment: TrialSegmentRecord,
        plan: Any,
        gate: Any,
        work_items: Sequence[WorkItemRecord],
    ) -> Dict[str, Any]:
        action = str(gate.action)
        reason = "; ".join(gate.reasons)
        final_boundary = segment.ordinal >= len(plan.boundaries) - 1
        if action == "pause":
            self.database.update_trial_segment(
                segment.id,
                status="awaiting_review",
                decision="pause",
                decision_reason=f"awaiting review: {reason}",
            )
            self.database.update_trial_run(trial_run.id, status="awaiting_review")
            self.database.update_run_group_trial(trial.id, status="awaiting_review")
            self.database.update_run_group(group.id, status="awaiting_review")
            return {"action": action, "queued_work_item_ids": []}
        if action == "stop":
            self.database.update_trial_segment(
                segment.id,
                status="stopped",
                decision="stop",
                decision_reason=f"checkpoint policy stopped training: {reason}",
            )
            self.database.update_trial_run(trial_run.id, status="stopped")
            return {"action": action, "queued_work_item_ids": []}
        if final_boundary:
            self.database.update_trial_segment(
                segment.id,
                status="completed",
                decision="complete",
                decision_reason="final checkpoint boundary complete",
            )
            self.database.update_trial_run(trial_run.id, status="completed")
            return {"action": "complete", "queued_work_item_ids": []}

        if group.pruning_policy.get("enabled"):
            # Synchronous halving owns promotion after every required seed has
            # passed its per-checkpoint gate.
            self.database.update_trial_segment(segment.id, status="completed")
            return {
                "action": "continue",
                "awaiting_successive_halving": True,
                "queued_work_item_ids": [],
            }

        self.database.update_trial_segment(
            segment.id,
            status="completed",
            decision="continue",
            decision_reason=reason,
        )

        capability = self.capabilities.resolve(
            group.trainer_mode,
            self._backend_from_config(group.resolved_launch_config),
        )
        resolved = dict(group.resolved_launch_config)
        resolved.update(trial.sampled_config)
        resolved["seed"] = trial_run.seed
        evaluations = [
            item
            for item in work_items
            if item.kind == "evaluation" and item.launch_spec.get("trial_segment_id") == segment.id
        ]
        if not evaluations or any(value.status != "completed" for value in evaluations):
            raise ValueError("checkpoint continuation requires all evaluations to complete")
        queued = self._queue_segment(
            group=group,
            trial=trial,
            trial_run=trial_run,
            capability=capability,
            resolved_config=resolved,
            development=self._require_revision(group.development_suite_revision_id),
            ordinal=segment.ordinal + 1,
            start_value=int(plan.boundaries[segment.ordinal]),
            end_value=int(plan.boundaries[segment.ordinal + 1]),
            unit=plan.unit,
            priority=max(value.priority for value in evaluations),
            max_retries=max(value.max_retries for value in evaluations),
            dependencies=tuple(value.id for value in evaluations),
        )
        return {"action": "continue", "queued_work_item_ids": queued}

    def review_checkpoint_gate(
        self,
        decision_id: str,
        *,
        action: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Record an operator override and apply it without mutating history."""

        normalized_action = str(action).strip().lower()
        normalized_reason = str(reason).strip()
        if normalized_action not in {"continue", "stop"}:
            raise ValueError("checkpoint review action must be continue or stop")
        if not normalized_reason:
            raise ValueError("checkpoint review requires a reason")
        original = self.database.get_checkpoint_gate_decision(str(decision_id))
        if original is None:
            raise ValueError(f"unknown checkpoint gate decision: {decision_id}")
        if original.action not in {"pause", "stop"}:
            raise ValueError("only paused or stopped checkpoint decisions can be reviewed")
        if normalized_action == original.action:
            raise ValueError("checkpoint review must change the current gate action")

        from halo_forge.adaptive_lab import AdaptiveLabService, ResolvedCheckpointPlan

        override = AdaptiveLabService(self.database).override_gate(
            original.id,
            action=normalized_action,
            reason=normalized_reason,
            idempotency_key=canonical_fingerprint(
                {
                    "override_of_id": original.id,
                    "action": normalized_action,
                    "reason": normalized_reason,
                }
            ),
        )
        if not original.run_group_id or not original.trial_run_id or not original.trial_segment_id:
            raise ValueError("checkpoint gate is not linked to a complete run-group boundary")
        group = self._require_group(original.run_group_id)
        trial_run = self.database.get_trial_run(original.trial_run_id)
        segment = self.database.get_trial_segment(original.trial_segment_id)
        if trial_run is None or segment is None:
            raise ValueError("checkpoint gate references missing trial state")
        trial = self.database.get_run_group_trial(trial_run.trial_id)
        if trial is None:
            raise ValueError("checkpoint gate references a missing trial")
        plan = ResolvedCheckpointPlan.from_dict(self._checkpoint_plan(group))
        applied = self._apply_checkpoint_gate_action(
            group=group,
            trial=trial,
            trial_run=trial_run,
            segment=segment,
            plan=plan,
            gate=override,
            work_items=self._group_work_items(group.id),
        )
        self.reconcile_group(group.id)
        return {
            "gate_decision": override.to_dict(),
            **applied,
            "run_group": self.get_run_group_detail(group.id, reconcile=False),
        }

    resume_checkpoint_gate = review_checkpoint_gate

    # ----- comparison and artifacts -----------------------------------

    def best_trial(self, run_group_id: str) -> Optional[Dict[str, Any]]:
        group = self.reconcile_group(run_group_id)
        development = self._require_revision(group.development_suite_revision_id)
        trials = self.database.list_run_group_trials(group.id)
        aggregates, _ = self._cohort_state(
            group, trials, self._group_work_items(group.id), development
        )
        ranking = self._rank_active_trials(aggregates, trials, direction=development.direction)
        return self._best_from_rows(trials, ranking)

    def compare_run_groups(self, left_group_id: str, right_group_id: str) -> Dict[str, Any]:
        left = self.reconcile_group(left_group_id)
        right = self.reconcile_group(right_group_id)
        if left.development_suite_revision_id != right.development_suite_revision_id:
            raise ValueError("run groups can only be compared on the same suite revision")
        development = self._require_revision(left.development_suite_revision_id)
        left_best = self.best_trial(left.id)
        right_best = self.best_trial(right.id)
        delta = None
        winner = None
        if left_best is not None and right_best is not None:
            left_value = float(left_best["objective_value"])
            right_value = float(right_best["objective_value"])
            delta = right_value - left_value
            if math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12):
                winner = "tie"
            elif (development.direction == "maximize" and right_value > left_value) or (
                development.direction == "minimize" and right_value < left_value
            ):
                winner = right.id
            else:
                winner = left.id
        return {
            "suite_revision_id": development.id,
            "metric": development.primary_metric,
            "direction": development.direction,
            "left": {"run_group": left.to_dict(), "best_trial": left_best},
            "right": {"run_group": right.to_dict(), "best_trial": right_best},
            "right_minus_left": delta,
            "winner": winner,
        }

    def build_fork_best_payload(
        self,
        run_group_id: str,
        *,
        name: Optional[str] = None,
        seeds: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Return a directly reusable repeat-group payload for the best trial."""

        group = self.reconcile_group(run_group_id)
        best = self.best_trial(group.id)
        if best is None:
            raise ValueError("run group has no eligible fully evaluated trial to fork")
        resolved = dict(group.resolved_launch_config)
        resolved.update(best["sampled_config"])
        payload = {
            "name": name or f"{group.name} — best trial fork",
            "kind": "repeat",
            "trainer_mode": group.trainer_mode,
            "base_config": resolved,
            "seeds": [int(value) for value in (seeds or group.seeds)],
            "development_suite_revision_id": group.development_suite_revision_id,
            "holdout_suite_revision_id": group.holdout_suite_revision_id,
            "dataset_bindings": group.dataset_bindings,
            "base_subject": group.base_subject,
            "parent_group_id": group.id,
            "source_trial_id": best["trial_id"],
            "source_config_hash": best["config_hash"],
        }
        checkpoint_policy_revision_id = self._checkpoint_policy_revision_id(group)
        checkpoint_plan = self._checkpoint_plan(group)
        if checkpoint_policy_revision_id and checkpoint_plan:
            payload.update(
                version=2,
                checkpoint_policy_revision_id=checkpoint_policy_revision_id,
                resolved_checkpoint_plan=checkpoint_plan,
            )
        return payload

    def fork_best(
        self,
        run_group_id: str,
        *,
        name: Optional[str] = None,
        seeds: Optional[Sequence[int]] = None,
        priority: int = 0,
        max_retries: int = 0,
    ) -> Dict[str, Any]:
        """Create a child repeat group from the selected best configuration."""

        payload = self.build_fork_best_payload(run_group_id, name=name, seeds=seeds)
        # Source-only keys remain useful in the returned clone payload but are
        # not part of RunGroupSpec. Creation safely ignores them.
        payload["priority"] = int(priority)
        payload["max_retries"] = int(max_retries)
        return self.create_group_from_payload(payload)

    def list_model_artifacts(
        self,
        *,
        run_id: Optional[str] = None,
        run_group_id: Optional[str] = None,
        artifact_kind: Optional[str] = None,
    ) -> list[Dict[str, Any]]:
        return [
            value.to_dict()
            for value in self.database.list_model_artifacts(
                run_id=run_id,
                run_group_id=run_group_id,
                artifact_kind=artifact_kind,
            )
        ]

    def register_model_artifact(self, **payload: Any) -> Dict[str, Any]:
        return self.database.create_model_artifact(**payload).to_dict()

    # ----- internal helpers -------------------------------------------

    def _group_summary(self, group: RunGroupRecord, *, reconcile: bool) -> Dict[str, Any]:
        if reconcile:
            group = self.reconcile_group(group.id)
        trials = self.database.list_run_group_trials(group.id)
        best = self.best_trial(group.id) if reconcile else None
        development = self._require_revision(group.development_suite_revision_id)
        statuses = [trial.status for trial in trials]
        return {
            **group.to_dict(),
            "objective": {
                "suite_revision_id": development.id,
                "metric": development.primary_metric,
                "direction": development.direction,
            },
            "trial_count": len(trials),
            "run_count": sum(len(self.database.list_trial_runs(row.id)) for row in trials),
            "completed_trials": statuses.count("completed"),
            "failed_trials": statuses.count("failed"),
            "pruned_trials": statuses.count("pruned"),
            "stopped_trials": statuses.count("stopped"),
            "awaiting_review_trials": statuses.count("awaiting_review"),
            "best_trial": best,
            "queued_work_count": sum(
                value.status == "queued" for value in self._group_work_items(group.id)
            ),
        }

    def _cohort_state(
        self,
        group: RunGroupRecord,
        trials: Sequence[RunGroupTrialRecord],
        work_items: Sequence[WorkItemRecord],
        development: BenchmarkSuiteRevisionRecord,
    ) -> tuple[tuple[CohortAggregate, ...], list[CohortObservation]]:
        expected: dict[str, list[int]] = {}
        observations: list[CohortObservation] = []
        objective_boundary = self._latest_common_objective_boundary(
            group, trials, work_items, development
        )
        for trial in trials:
            runs = self.database.list_trial_runs(trial.id)
            expected[trial.id] = [run.seed for run in runs]
            for trial_run in runs:
                segments = self.database.list_trial_segments(trial_run.id)
                segment_id = next(
                    (
                        segment.id
                        for segment in segments
                        if objective_boundary is not None
                        and segment.ordinal == objective_boundary["ordinal"]
                    ),
                    None,
                )
                value = (
                    self._objective_for_run(group, trial_run, development, work_items, segment_id)
                    if segment_id is not None
                    else None
                )
                if value is not None:
                    observations.append(
                        CohortObservation(trial.id, trial_run.seed, value, "completed")
                    )
                elif trial_run.status in {"failed", "cancelled", "pruned"}:
                    observations.append(
                        CohortObservation(trial.id, trial_run.seed, status=trial_run.status)
                    )
        return aggregate_cohort(observations, expected), observations

    def _latest_common_objective_boundary(
        self,
        group: RunGroupRecord,
        trials: Sequence[RunGroupTrialRecord],
        work_items: Sequence[WorkItemRecord],
        development: BenchmarkSuiteRevisionRecord,
    ) -> Optional[Dict[str, Any]]:
        """Return the latest development-evaluated boundary shared by active runs.

        A run that terminated before producing any development evidence remains
        an ineligible cohort member, but it does not erase the comparable
        boundary used to rank the successful cohorts. Non-terminal runs must
        publish development evidence before a group receives an objective.
        """

        available_by_run: list[Dict[int, TrialSegmentRecord]] = []
        ignored_terminal = {"failed", "cancelled", "pruned"}
        for trial in trials:
            for trial_run in self.database.list_trial_runs(trial.id):
                available: Dict[int, TrialSegmentRecord] = {}
                for segment in self.database.list_trial_segments(trial_run.id):
                    value = self._objective_for_run(
                        group,
                        trial_run,
                        development,
                        work_items,
                        segment.id,
                    )
                    if value is not None:
                        available[segment.ordinal] = segment
                if available:
                    available_by_run.append(available)
                elif trial_run.status not in ignored_terminal:
                    return None
        if not available_by_run:
            return None
        common = set(available_by_run[0])
        for values in available_by_run[1:]:
            common.intersection_update(values)
        if not common:
            return None
        ordinal = max(common)
        segments = [values[ordinal] for values in available_by_run]
        contracts = {(value.unit, value.end_value) for value in segments}
        if len(contracts) != 1:
            return None
        unit, end_value = next(iter(contracts))
        return {"ordinal": ordinal, "unit": unit, "value": int(end_value)}

    def _objective_for_run(
        self,
        group: RunGroupRecord,
        trial_run: TrialRunRecord,
        development: BenchmarkSuiteRevisionRecord,
        work_items: Sequence[WorkItemRecord],
        segment_id: Optional[str],
    ) -> Optional[float]:
        evaluation_work = [
            item
            for item in work_items
            if item.kind == "evaluation"
            and item.status == "completed"
            and item.launch_spec.get("trial_run_id") == trial_run.id
            and item.launch_spec.get("suite_revision_id") == development.id
            and (segment_id is None or item.launch_spec.get("trial_segment_id") == segment_id)
        ]
        for item in reversed(evaluation_work):
            value = self._metric_from_result(item.result, development.primary_metric)
            if value is not None:
                return value
            evaluation_id = item.result.get("evaluation_id")
            if evaluation_id:
                value = self._metric_from_evaluation(str(evaluation_id), development.primary_metric)
                if value is not None:
                    return value

        evaluations = self.database.list_evaluations(
            suite_revision_id=development.id,
            subject_ref=trial_run.run_id,
            status="completed",
            limit=100,
        )
        for evaluation in evaluations:
            requested_segment = evaluation.request.get("trial_segment_id")
            if segment_id is not None and requested_segment != segment_id:
                continue
            value = self._metric_from_evaluation(evaluation.id, development.primary_metric)
            if value is not None:
                return value
        return None

    def _metric_from_evaluation(self, evaluation_id: str, metric: str) -> Optional[float]:
        rows = [
            row
            for row in self.database.list_evaluation_metrics(evaluation_id)
            if row.name == metric
        ]
        if not rows:
            return None
        aggregate = [row.value for row in rows if not row.suite_item_id]
        values = aggregate or [row.value for row in rows]
        return float(statistics.fmean(values))

    @staticmethod
    def _metric_from_result(result: Mapping[str, Any], metric: str) -> Optional[float]:
        direct = result.get("objective_value")
        if direct is not None:
            return float(direct)
        metrics = result.get("metrics")
        if isinstance(metrics, Mapping) and metrics.get(metric) is not None:
            value = metrics[metric]
            if isinstance(value, Mapping):
                value = value.get("value")
            if value is not None:
                return float(value)
        if isinstance(metrics, Sequence) and not isinstance(metrics, (str, bytes)):
            values = [
                float(row["value"])
                for row in metrics
                if isinstance(row, Mapping)
                and row.get("name") == metric
                and row.get("value") is not None
            ]
            if values:
                return float(statistics.fmean(values))
        return None

    @staticmethod
    def _segment_work(
        work_items: Sequence[WorkItemRecord],
        segment: Optional[TrialSegmentRecord],
        kind: str,
    ) -> Optional[WorkItemRecord]:
        if segment is None:
            return None
        for item in reversed(work_items):
            if item.kind == kind and item.launch_spec.get("trial_segment_id") == segment.id:
                return item
        return None

    def _group_work_items(self, run_group_id: str) -> list[WorkItemRecord]:
        return [
            item
            for item in self.database.list_work_items(limit=100000)
            if item.launch_spec.get("run_group_id") == run_group_id
        ]

    def _list_gate_decisions(self, run_group_id: str) -> list[Any]:
        list_decisions = getattr(self.database, "list_checkpoint_gate_decisions", None)
        if list_decisions is None:
            return []
        try:
            values = list(list_decisions(run_group_id=run_group_id, limit=100000))
        except TypeError:
            try:
                values = list(list_decisions(run_group_id=run_group_id))
            except TypeError:
                try:
                    values = list(list_decisions(limit=100000))
                except TypeError:
                    values = list(list_decisions())
            values = [
                value
                for value in values
                if str(
                    getattr(value, "run_group_id", None)
                    or (value.to_dict().get("run_group_id") if hasattr(value, "to_dict") else "")
                )
                == run_group_id
            ]
        return sorted(
            values,
            key=lambda value: (
                str(getattr(value, "created_at", "")),
                str(getattr(value, "id", "")),
            ),
        )

    def _latest_gate_for_segment(
        self, run_group_id: str, segment_id: str
    ) -> Optional[Dict[str, Any]]:
        matches = []
        for value in self._list_gate_decisions(run_group_id):
            payload = value.to_dict() if hasattr(value, "to_dict") else dict(value)
            if (
                str(payload.get("trial_segment_id") or payload.get("segment_id") or "")
                == segment_id
            ):
                matches.append(payload)
        return matches[-1] if matches else None

    @staticmethod
    def _checkpoint_plan(group: RunGroupRecord) -> Dict[str, Any]:
        value = getattr(group, "resolved_checkpoint_plan", None)
        if isinstance(value, Mapping) and value:
            return deepcopy(dict(value))
        return deepcopy(dict(group.sampler_state.get("resolved_checkpoint_plan") or {}))

    @staticmethod
    def _checkpoint_policy_revision_id(group: RunGroupRecord) -> Optional[str]:
        value = getattr(group, "checkpoint_policy_revision_id", None)
        if value:
            return str(value)
        value = group.sampler_state.get("checkpoint_policy_revision_id")
        return str(value) if value else None

    def _checkpoint_policy_model(self, record: Any) -> Any:
        from halo_forge.adaptive_lab import CheckpointPolicyRevision

        definition = deepcopy(dict(getattr(record, "definition", {}) or {}))
        if isinstance(definition.get("definition"), Mapping):
            definition = deepcopy(dict(definition["definition"]))
        parent = self.database.get_checkpoint_policy(record.policy_id)
        definition.update(
            revision_id=record.id,
            policy_id=record.policy_id,
            revision_number=record.revision_number,
            name=definition.get("name") or (parent.name if parent else record.policy_id),
            description=(
                definition.get("description")
                if definition.get("description") is not None
                else (parent.description if parent else None)
            ),
            development_suite_revision_id=record.development_suite_revision_id,
            primary_metric=record.primary_metric,
            direction=record.direction,
            content_hash=record.content_hash,
        )
        return CheckpointPolicyRevision.from_dict(definition)

    def _segment_metrics(
        self,
        segment: TrialSegmentRecord,
        *,
        work_items: Sequence[WorkItemRecord],
        required_suite_revision_ids: Sequence[str],
        development_suite_revision_id: str,
    ) -> tuple[Dict[str, float], list[str], list[str], list[str]]:
        by_suite: dict[str, WorkItemRecord] = {}
        for item in work_items:
            if item.kind == "evaluation" and item.launch_spec.get("trial_segment_id") == segment.id:
                by_suite[str(item.launch_spec.get("suite_revision_id") or "")] = item
        metrics: Dict[str, float] = {}
        missing: list[str] = []
        pending: list[str] = []
        evaluation_ids: list[str] = []
        for revision_id in required_suite_revision_ids:
            revision_id = str(revision_id)
            item = by_suite.get(revision_id)
            if item is None:
                pending.append(revision_id)
                continue
            if item.status in {"queued", "running", "blocked", "retrying"}:
                pending.append(revision_id)
                continue
            if item.status != "completed":
                missing.append(revision_id)
                continue
            evaluation_id = str(item.result.get("evaluation_id") or "").strip()
            if evaluation_id:
                evaluation_ids.append(evaluation_id)
            suite_metrics: Dict[str, float] = {}
            result_metrics = item.result.get("metrics")
            if isinstance(result_metrics, Mapping):
                for name, raw in result_metrics.items():
                    value = raw.get("value") if isinstance(raw, Mapping) else raw
                    if value is not None:
                        suite_metrics[str(name)] = float(value)
            if evaluation_id:
                for row in self.database.list_evaluation_metrics(evaluation_id):
                    if row.suite_item_id:
                        continue
                    suite_metrics[str(row.name)] = float(row.value)
            if not suite_metrics:
                missing.append(revision_id)
                continue
            for name, value in suite_metrics.items():
                metrics[f"{revision_id}:{name}"] = value
                if revision_id == development_suite_revision_id or name not in metrics:
                    metrics[name] = value
        return metrics, missing, pending, sorted(set(evaluation_ids))

    @staticmethod
    def _best_metric_history(
        history: Sequence[Mapping[str, float]], policy: Any
    ) -> Dict[str, float]:
        directions = {str(rule.metric): str(rule.direction) for rule in policy.rules}
        directions.setdefault(str(policy.primary_metric), str(policy.direction))
        result: Dict[str, float] = {}
        keys = sorted({key for row in history for key in row})
        for key in keys:
            values = [float(row[key]) for row in history if key in row]
            bare_key = key.split(":", 1)[-1]
            direction = directions.get(key, directions.get(bare_key, "maximize"))
            result[key] = min(values) if direction == "minimize" else max(values)
        return result

    @staticmethod
    def _checkpoint_plateau_state(
        history: Sequence[Mapping[str, float]],
        current_metrics: Mapping[str, float],
        policy: Any,
    ) -> tuple[Dict[str, float], Dict[str, int]]:
        """Resolve qualifying-best references and consecutive plateau counts.

        A checkpoint resets patience only when it clears the rule's declared
        minimum delta. Smaller improvements remain visible but accumulate
        patience against the last qualifying best. Recomputing from immutable
        evaluation history makes restart and duplicate callback behavior exact.
        """

        best_references: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for rule in policy.rules:
            if str(rule.kind) != "plateau":
                continue
            metric = str(rule.metric)
            direction = str(rule.direction)
            minimum_delta = float(rule.minimum_delta or 0.0)
            values = [float(row[metric]) for row in history if metric in row]
            current = current_metrics.get(metric)
            if not values:
                counts[metric] = 0
                continue

            consecutive = 0
            if str(rule.comparison) == "previous":
                reference = values[0]
                for value in values[1:]:
                    delta = value - reference if direction == "maximize" else reference - value
                    consecutive = consecutive + 1 if delta < minimum_delta else 0
                    reference = value
                if current is not None:
                    delta = (
                        float(current) - reference
                        if direction == "maximize"
                        else reference - float(current)
                    )
                    consecutive = consecutive + 1 if delta < minimum_delta else 0
                counts[metric] = consecutive
                continue

            qualifying_best = values[0]
            for value in values[1:]:
                delta = (
                    value - qualifying_best if direction == "maximize" else qualifying_best - value
                )
                if delta >= minimum_delta:
                    qualifying_best = value
                    consecutive = 0
                else:
                    consecutive += 1
            best_references[metric] = qualifying_best
            if current is not None:
                delta = (
                    float(current) - qualifying_best
                    if direction == "maximize"
                    else qualifying_best - float(current)
                )
                if delta >= minimum_delta:
                    consecutive = 0
                else:
                    consecutive += 1
            counts[metric] = consecutive
        return best_references, counts

    def _baseline_metrics(self, group: RunGroupRecord, policy: Any) -> Optional[Dict[str, float]]:
        subject = dict(group.base_subject or {})
        result: Dict[str, float] = {}
        raw_metrics = subject.get("metrics")
        if isinstance(raw_metrics, Mapping):
            for name, raw in raw_metrics.items():
                value = raw.get("value") if isinstance(raw, Mapping) else raw
                if value is not None:
                    result[str(name)] = float(value)

        evaluation_ids: dict[str, str] = {}
        if isinstance(subject.get("evaluation_ids"), Mapping):
            evaluation_ids.update(
                {str(key): str(value) for key, value in subject["evaluation_ids"].items()}
            )
        if subject.get("evaluation_id"):
            evaluation_ids.setdefault(
                policy.development_suite_revision_id,
                str(subject["evaluation_id"]),
            )
        subject_ref = str(subject.get("ref") or subject.get("subject_ref") or "").strip()
        for revision_id in policy.required_suite_revision_ids:
            evaluation_id = evaluation_ids.get(revision_id)
            if not evaluation_id and subject_ref:
                evaluations = self.database.list_evaluations(
                    suite_revision_id=revision_id,
                    subject_ref=subject_ref,
                    status="completed",
                    limit=10,
                )
                evaluation_id = evaluations[-1].id if evaluations else None
            if not evaluation_id:
                continue
            for row in self.database.list_evaluation_metrics(evaluation_id):
                if row.suite_item_id:
                    continue
                key = str(row.name)
                result[f"{revision_id}:{key}"] = float(row.value)
                if revision_id == policy.development_suite_revision_id or key not in result:
                    result[key] = float(row.value)
        return result or None

    def _checkpoint_occurrence_id(self, segment_id: str) -> Optional[str]:
        row = self.database._conn.execute(
            "SELECT id FROM artifact_occurrences WHERE trial_segment_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (segment_id,),
        ).fetchone()
        return str(row["id"]) if row is not None else None

    def _require_group(self, run_group_id: str) -> RunGroupRecord:
        group = self.database.get_run_group(run_group_id)
        if group is None:
            raise ValueError(f"unknown run group: {run_group_id}")
        return group

    def _require_revision(self, revision_id: Optional[str]) -> BenchmarkSuiteRevisionRecord:
        if not revision_id:
            raise ValueError("run group has no development suite revision")
        revision = self.database.get_benchmark_suite_revision(revision_id)
        if revision is None:
            raise ValueError(f"unknown benchmark suite revision: {revision_id}")
        return revision

    def _require_suite_revision(
        self, revision_id: str, *, purpose: str
    ) -> BenchmarkSuiteRevisionRecord:
        revision = self._require_revision(revision_id)
        suite = self.database.get_benchmark_suite(revision.suite_id)
        if suite is None:
            raise ValueError(f"benchmark suite for revision {revision_id} is missing")
        if suite.purpose != purpose:
            raise ValueError(
                f"suite revision {revision_id} must belong to a {purpose} suite; "
                f"found {suite.purpose}"
            )
        return revision

    @staticmethod
    def _pin_spec_to_development_revision(
        spec: RunGroupSpec | Mapping[str, Any],
        development: BenchmarkSuiteRevisionRecord,
    ) -> RunGroupSpec:
        if isinstance(spec, RunGroupSpec):
            resolved = spec
            if resolved.metric != development.primary_metric:
                raise ValueError(
                    "run-group metric must match the pinned development suite primary metric"
                )
            if resolved.direction != development.direction:
                raise ValueError(
                    "run-group direction must match the pinned development suite direction"
                )
            return resolved
        if not isinstance(spec, Mapping):
            raise TypeError("spec must be a RunGroupSpec or mapping")
        values = dict(spec)
        if values.get("metric") not in (None, development.primary_metric):
            raise ValueError(
                "run-group metric must match the pinned development suite primary metric"
            )
        if values.get("direction") not in (None, development.direction):
            raise ValueError(
                "run-group direction must match the pinned development suite direction"
            )
        values["metric"] = development.primary_metric
        values["direction"] = development.direction
        return RunGroupSpec.from_dict(values)

    @staticmethod
    def _backend_from_config(config: Mapping[str, Any]) -> str:
        return str(
            config.get("backend")
            or config.get("training_backend")
            or config.get("accelerator")
            or config.get("device")
            or "hf"
        )

    def _resolve_checkpoint_spec(
        self,
        spec: RunGroupSpec,
        *,
        development: BenchmarkSuiteRevisionRecord,
        trainer_mode: str,
        capability: TrainerExecutionCapability,
    ) -> RunGroupSpec:
        """Resolve and pin a v2 checkpoint policy at group creation.

        The resolved plan is stored inside the existing immutable sampler-state
        envelope.  That keeps the run-group schema backward compatible while
        preventing a mutable policy head from changing an already-created
        group's execution boundaries.
        """

        raw_plan = dict(spec.resolved_checkpoint_plan or {})
        policy_revision_id = str(
            spec.checkpoint_policy_revision_id or raw_plan.get("policy_revision_id") or ""
        ).strip()
        if not policy_revision_id:
            if raw_plan:
                raise ValueError("resolved_checkpoint_plan requires checkpoint_policy_revision_id")
            return spec

        get_revision = getattr(self.database, "get_checkpoint_policy_revision", None)
        if get_revision is None:
            raise RuntimeError("checkpoint policy persistence is unavailable")
        record = get_revision(policy_revision_id)
        if record is None:
            raise ValueError(f"unknown checkpoint policy revision: {policy_revision_id}")

        from halo_forge.adaptive_lab import AdaptiveLabService

        policy = self._checkpoint_policy_model(record)
        if policy.development_suite_revision_id != development.id:
            raise ValueError(
                "checkpoint policy development suite must match the run-group development suite"
            )
        if policy.primary_metric != development.primary_metric:
            raise ValueError("checkpoint policy primary metric must match the development suite")
        if policy.direction != development.direction:
            raise ValueError("checkpoint policy direction must match the development suite")
        if not capability.supports_gated_execution:
            reason = capability.reason or "trainer does not support resumable segments"
            raise ValueError(f"adaptive checkpoint execution is unavailable: {reason}")

        total_budget = self._adaptive_total_budget(
            spec.base_config,
            capability=capability,
            supplied_plan=raw_plan,
            schedule=policy.schedule,
        )
        supported_units = (capability.segment_unit,)
        resolved = AdaptiveLabService(self.database).resolve_checkpoint_plan(
            policy,
            trainer_mode=trainer_mode,
            total_budget=total_budget,
            supported_units=supported_units,
            capabilities=(capability.capability_id,),
        )
        resolved_payload = resolved.to_dict()
        if resolved.policy_revision_id != policy_revision_id:
            raise ValueError("resolved checkpoint plan references a different policy revision")
        if raw_plan:
            supplied_hash = str(raw_plan.get("content_hash") or "").strip()
            if supplied_hash and supplied_hash != resolved.content_hash:
                raise ValueError(
                    "resolved checkpoint plan does not match the pinned policy and trainer capability"
                )

        for revision_id in resolved.required_suite_revision_ids:
            self._require_suite_revision(str(revision_id), purpose="development")
        if spec.pruning.enabled and tuple(resolved.boundaries) != tuple(spec.pruning.budgets):
            raise ValueError(
                "checkpoint boundaries must equal successive-halving budgets when both are enabled"
            )

        values = spec.to_dict()
        values.update(
            version=2,
            checkpoint_policy_revision_id=policy_revision_id,
            resolved_checkpoint_plan=resolved_payload,
        )
        return RunGroupSpec.from_dict(values)

    @staticmethod
    def _adaptive_total_budget(
        config: Mapping[str, Any],
        *,
        capability: TrainerExecutionCapability,
        supplied_plan: Mapping[str, Any],
        schedule: Any,
    ) -> int:
        supplied = supplied_plan.get("total_budget")
        if capability.segment_unit == "step":
            configured = int(config.get("max_steps") or config.get("training_steps") or 0)
        elif capability.segment_unit == "cycle":
            configured = int(config.get("cycles") or config.get("num_cycles") or 0)
        else:
            configured = 1
        if supplied is not None:
            total = int(supplied)
            if configured > 0 and total != configured:
                raise ValueError(
                    "resolved checkpoint plan total_budget does not match the training config"
                )
        else:
            total = configured
        if total <= 0 and getattr(schedule, "mode", None) == "explicit":
            boundaries = tuple(getattr(schedule, "boundaries", ()) or ())
            total = int(boundaries[-1]) if boundaries else 0
        if total <= 0:
            key = "max_steps" if capability.segment_unit == "step" else "cycles"
            raise ValueError(
                f"adaptive checkpoint policy requires a positive {key} training budget"
            )
        return total

    @staticmethod
    def _validate_capability(spec: RunGroupSpec, capability: TrainerExecutionCapability) -> None:
        if spec.pruning.enabled:
            if not capability.supports_gated_execution:
                reason = capability.reason or "trainer does not support resumable segments"
                raise ValueError(f"successive halving is unavailable: {reason}")
            if len(spec.pruning.budgets) < 2:
                raise ValueError("successive halving requires at least two increasing budgets")
        plan = dict(spec.resolved_checkpoint_plan or {})
        boundaries = tuple(int(value) for value in plan.get("boundaries") or ())
        if len(boundaries) > 1 and not capability.supports_gated_execution:
            reason = capability.reason or "trainer does not support resumable segments"
            raise ValueError(f"adaptive checkpoint execution is unavailable: {reason}")
        if boundaries and plan.get("unit") != (
            capability.segment_unit if capability.supports_gated_execution else "full_trial"
        ):
            raise ValueError("resolved checkpoint plan is incompatible with the trainer backend")

    @staticmethod
    def _normalize_dataset_bindings(
        bindings: Sequence[Mapping[str, Any]],
    ) -> list[Dict[str, Any]]:
        normalized = []
        seen = set()
        for raw in bindings:
            if not isinstance(raw, Mapping):
                raise TypeError("dataset bindings must be mappings")
            version_id = str(raw.get("dataset_version_id") or "").strip()
            role = str(raw.get("role") or "train").strip().lower()
            split = str(raw.get("split") or role).strip()
            if not version_id or not role or not split:
                raise ValueError("dataset binding role, version, and split are required")
            key = (role, version_id, split)
            if key in seen:
                raise ValueError(f"duplicate dataset binding: {key}")
            seen.add(key)
            normalized.append({"role": role, "dataset_version_id": version_id, "split": split})
        return normalized

    @staticmethod
    def _initial_segment_end(spec: RunGroupSpec) -> int:
        plan = dict(spec.resolved_checkpoint_plan or {})
        boundaries = tuple(int(value) for value in plan.get("boundaries") or ())
        if boundaries:
            return boundaries[0]
        return int(spec.pruning.budgets[0]) if spec.pruning.enabled else 1

    @staticmethod
    def _initial_segment_unit(spec: RunGroupSpec, capability: TrainerExecutionCapability) -> str:
        plan = dict(spec.resolved_checkpoint_plan or {})
        if plan.get("unit"):
            return str(plan["unit"])
        return capability.segment_unit if spec.pruning.enabled else "full_trial"

    @staticmethod
    def _resolved_group_budgets(
        spec: RunGroupSpec, capability: TrainerExecutionCapability
    ) -> Dict[str, Any]:
        plan = dict(spec.resolved_checkpoint_plan or {})
        if plan:
            return {
                "values": [int(value) for value in plan.get("boundaries") or ()],
                "unit": str(plan.get("unit") or capability.segment_unit),
                "total": int(plan.get("total_budget") or 0),
                "source": "checkpoint_policy",
            }
        return {
            "values": list(spec.pruning.budgets),
            "unit": capability.segment_unit if spec.pruning.enabled else "full_trial",
        }

    @staticmethod
    def _managed_output_dir(config: Mapping[str, Any], run_id: str) -> str:
        raw_root = (
            config.get("output_root")
            or config.get("output_dir")
            or config.get("output")
            or "~/.halo-forge/runs"
        )
        root = Path(str(raw_root)).expanduser()
        if root.name == run_id:
            return str(root)
        return str(root / run_id)

    @staticmethod
    def _training_command(
        *,
        trainer_mode: str,
        config: Mapping[str, Any],
        bindings: Sequence[Mapping[str, Any]],
        unit: str,
        start_value: int,
        end_value: int,
    ) -> Optional[list[str]]:
        """Build a conservative argv using flags shared by the existing trainers."""

        mode = str(trainer_mode).strip().lower()
        if unit == "step" and start_value:
            # The next checkpoint path does not exist until the previous
            # segment publishes. The worker resolves it and appends --resume;
            # executing this command without that resolution would restart.
            return None
        command = [sys.executable, "-m", "halo_forge.cli"]
        accelerator = (
            str(config.get("accelerator") or config.get("device") or config.get("backend") or "")
            .strip()
            .lower()
            .replace("-", "_")
        )
        accelerator_aliases = {
            "torch_cuda": "cuda",
            "torch_mps": "mps",
            "torch_rocm": "rocm",
            "torch_cpu": "cpu",
        }
        accelerator = accelerator_aliases.get(accelerator, accelerator)
        if accelerator in {"auto", "rocm", "rocm_gfx1151", "cuda", "mps", "mlx", "cpu"}:
            command.extend(("--accelerator", accelerator))
        command.extend((mode, "train"))
        model = config.get("model") or config.get("model_name") or config.get("base_model")
        if model:
            command.extend(("--model", str(model)))
        command.extend(("--output", str(config["output_dir"])))
        command.extend(("--seed", str(int(config.get("seed", 42)))))
        for binding in bindings:
            command.extend(
                (
                    "--dataset-binding",
                    (f"{binding['role']}={binding['dataset_version_id']}:" f"{binding['split']}"),
                )
            )
        if mode == "vlm" and bindings and not config.get("dataset"):
            train_binding = next(
                (binding for binding in bindings if binding.get("role") == "train"),
                bindings[0],
            )
            # argparse currently requires --dataset before the managed binding
            # renderer replaces it with the content-addressed artifact path.
            command.extend(("--dataset", str(train_binding["dataset_version_id"])))
        if not bindings:
            if config.get("dataset"):
                command.extend(("--dataset", str(config["dataset"])))
            elif config.get("train_file") or config.get("data"):
                command.extend(("--data", str(config.get("train_file") or config.get("data"))))
        ExperimentOrchestrationService._append_config_flags(command, mode, config)
        if unit == "step":
            # A gated boundary is not useful unless the trainer publishes an
            # exact resumable checkpoint there. Override user/default cadence
            # for this bounded invocation; the original resolved config stays
            # immutable and is still captured in the launch manifest.
            ExperimentOrchestrationService._set_command_value(
                command, "--save-steps", str(int(end_value))
            )
            # These trainers load the best validation checkpoint at the end
            # of a bounded segment. Transformers requires the save cadence to
            # be an exact multiple of the evaluation cadence; pinning both to
            # the declared boundary keeps short adaptive segments valid.
            if mode in {"sft", "dpo", "orpo"}:
                ExperimentOrchestrationService._set_command_value(
                    command, "--eval-steps", str(int(end_value))
                )
            command.extend(("--max-steps", str(int(end_value))))
        elif config.get("max_steps") is not None:
            command.extend(("--max-steps", str(int(config["max_steps"]))))
        if unit == "cycle":
            command.extend(("--cycles", str(int(end_value))))
            if start_value:
                if mode in {"vlm", "audio", "reasoning", "agentic"}:
                    command.extend(("--resume-from-cycle", str(int(start_value))))
                else:
                    # RAFT resumes from a concrete checkpoint path, not an
                    # integer cycle. The worker must resolve that artifact.
                    return None
        return command

    @staticmethod
    def _set_command_value(command: list[str], flag: str, value: str) -> None:
        while flag in command:
            index = command.index(flag)
            del command[index : min(len(command), index + 2)]
        command.extend((flag, value))

    @staticmethod
    def _append_config_flags(command: list[str], mode: str, config: Mapping[str, Any]) -> None:
        """Forward trainer knobs that are explicitly represented by current CLIs."""

        common = {
            "epochs": "--epochs",
            "num_epochs": "--epochs",
            "batch_size": "--batch-size",
            "learning_rate": "--learning-rate",
            "warmup_ratio": "--warmup-ratio",
            "weight_decay": "--weight-decay",
            "max_grad_norm": "--max-grad-norm",
            "gradient_accumulation": "--gradient-accumulation",
            "gradient_accumulation_steps": "--gradient-accumulation",
            "lora_rank": "--lora-rank",
            "lora_alpha": "--lora-alpha",
            "lora_dropout": "--lora-dropout",
            "optim": "--optim",
            "max_samples": "--max-samples",
        }
        preference = {
            "beta": "--beta",
            "save_steps": "--save-steps",
            "eval_steps": "--eval-steps",
            "save_total_limit": "--save-total-limit",
            "validation_split": "--validation-split",
            "max_seq_length": "--max-seq-length",
            "max_prompt_length": "--max-prompt-length",
            "init_lora_weights": "--init-lora-weights",
        }
        by_mode: dict[str, dict[str, str]] = {
            "sft": {
                **common,
                "save_steps": "--save-steps",
                "eval_steps": "--eval-steps",
                "save_total_limit": "--save-total-limit",
                "early_stopping_patience": "--early-stopping-patience",
                "validation_split": "--validation-split",
                "max_seq_length": "--max-seq-length",
                "init_lora_weights": "--init-lora-weights",
            },
            "dpo": {
                **common,
                **preference,
                "loss_type": "--loss-type",
                "label_smoothing": "--label-smoothing",
            },
            "orpo": {**common, **preference},
            "rm": {
                **common,
                "max_length": "--max-length",
                "center_rewards_coefficient": "--center-rewards-coefficient",
                "init_lora_weights": "--init-lora-weights",
                "save_steps": "--save-steps",
                "save_total_limit": "--save-total-limit",
            },
            "grpo": {
                **common,
                "num_generations": "--num-generations",
                "beta": "--beta",
                "epsilon": "--epsilon",
                "temperature": "--temperature",
                "verifier": "--verifier",
                "reward_threshold": "--reward-threshold",
                "max_prompt_length": "--max-prompt-length",
                "max_completion_length": "--max-completion-length",
                "rollout_engine": "--rollout-engine",
                "init_lora_weights": "--init-lora-weights",
                "save_steps": "--save-steps",
                "save_total_limit": "--save-total-limit",
            },
            "raft": {
                "prompts": "--prompts",
                "verifier": "--verifier",
                "keep_percent": "--keep-percent",
                "reward_threshold": "--reward-threshold",
                "curriculum": "--curriculum",
                "curriculum_stats": "--curriculum-stats",
                "curriculum_start": "--curriculum-start",
                "curriculum_increment": "--curriculum-increment",
                "reward_shaping": "--reward-shaping",
                "lr_decay": "--lr-decay",
                "min_lr": "--min-lr",
                "system_prompt": "--system-prompt",
                "samples_per_prompt": "--samples-per-prompt",
                "temperature": "--temperature",
                "rollout_engine": "--rollout-engine",
                "rollout_model": "--rollout-model",
            },
            "vlm": {
                "samples_per_prompt": "--samples-per-prompt",
                "perception_weight": "--perception-weight",
                "reasoning_weight": "--reasoning-weight",
                "output_weight": "--output-weight",
                "lr_decay": "--lr-decay",
                "temperature": "--temperature",
                "max_new_tokens": "--max-new-tokens",
                "keep_percent": "--keep-percent",
                "reward_threshold": "--reward-threshold",
                "limit": "--limit",
            },
            "audio": {
                "task": "--task",
                "learning_rate": "--lr",
                "lr": "--lr",
                "lr_decay": "--lr-decay",
                "samples_per_prompt": "--samples-per-prompt",
                "temperature": "--temperature",
                "keep_percent": "--keep-percent",
                "reward_threshold": "--reward-threshold",
            },
            "reasoning": {
                "learning_rate": "--lr",
                "lr": "--lr",
                "lr_decay": "--lr-decay",
                "samples_per_prompt": "--samples-per-prompt",
                "temperature": "--temperature",
                "keep_percent": "--keep-percent",
                "limit": "--limit",
            },
            "agentic": {
                "learning_rate": "--lr",
                "lr": "--lr",
                "lr_decay": "--lr-decay",
                "samples_per_prompt": "--samples-per-prompt",
                "temperature": "--temperature",
                "keep_percent": "--keep-percent",
                "limit": "--limit",
            },
        }
        boolean_by_mode = {
            "sft": {
                "no_lora": "--no-lora",
                "use_dora": "--use-dora",
                "use_rslora": "--use-rslora",
                "no_gradient_checkpointing": "--no-gradient-checkpointing",
            },
            "dpo": {
                "reference_free": "--reference-free",
                "load_in_4bit": "--load-in-4bit",
                "use_dora": "--use-dora",
                "use_rslora": "--use-rslora",
                "no_gradient_checkpointing": "--no-gradient-checkpointing",
            },
            "orpo": {
                "load_in_4bit": "--load-in-4bit",
                "use_dora": "--use-dora",
                "use_rslora": "--use-rslora",
                "no_gradient_checkpointing": "--no-gradient-checkpointing",
            },
            "rm": {
                "load_in_4bit": "--load-in-4bit",
                "use_dora": "--use-dora",
                "use_rslora": "--use-rslora",
                "no_gradient_checkpointing": "--no-gradient-checkpointing",
            },
            "grpo": {
                "no_scale_rewards": "--no-scale-rewards",
                "reference_free": "--reference-free",
                "load_in_4bit": "--load-in-4bit",
                "use_dora": "--use-dora",
                "use_rslora": "--use-rslora",
                "no_gradient_checkpointing": "--no-gradient-checkpointing",
            },
            "raft": {
                "rollout_only": "--rollout-only",
                "allow_compile_only_training": "--allow-compile-only-training",
                "experimental_attention": "--experimental-attention",
            },
        }
        used_flags = set(command)
        for key, flag in by_mode.get(mode, {}).items():
            value = config.get(key)
            if value is None or flag in used_flags:
                continue
            command.extend((flag, str(value)))
            used_flags.add(flag)
        for key, flag in boolean_by_mode.get(mode, {}).items():
            if bool(config.get(key)) and flag not in used_flags:
                command.append(flag)
                used_flags.add(flag)

    @staticmethod
    def _evaluation_command(
        *,
        suite_revision_id: str,
        run_id: str,
        segment_id: str,
        unit: str,
    ) -> Optional[list[str]]:
        if unit != "full_trial":
            return None
        return [
            sys.executable,
            "-m",
            "halo_forge.cli",
            "eval",
            "run",
            "--suite-revision",
            suite_revision_id,
            "--subject-type",
            "run",
            "--subject",
            run_id,
            "--request",
            '{"trial_segment_id":"' + segment_id + '"}',
            "--wait",
        ]

    @staticmethod
    def _stable_id(*parts: Any) -> str:
        text = ":".join(str(value) for value in parts)
        return uuid.uuid5(uuid.NAMESPACE_URL, f"halo-forge:{text}").hex

    @staticmethod
    def _best_from_rows(
        trials: Sequence[RunGroupTrialRecord],
        ranking: Sequence[CohortAggregate],
    ) -> Optional[Dict[str, Any]]:
        if not ranking:
            return None
        trial = {row.id: row for row in trials}[ranking[0].trial_key]
        return {
            "trial_id": trial.id,
            "ordinal": trial.ordinal,
            "sampled_config": trial.sampled_config,
            "config_hash": trial.config_hash,
            "objective_value": ranking[0].mean,
            "standard_deviation": ranking[0].standard_deviation,
            "seed_coverage": ranking[0].completed_count,
            "required_seed_count": len(ranking[0].expected_seeds),
        }

    @staticmethod
    def _rank_active_trials(
        aggregates: Sequence[CohortAggregate],
        trials: Sequence[RunGroupTrialRecord],
        *,
        direction: str,
    ) -> tuple[CohortAggregate, ...]:
        selectable = {
            trial.id
            for trial in trials
            if trial.status not in {"pruned", "failed", "cancelled", "stopped", "awaiting_review"}
        }
        return rank_cohort(
            [row for row in aggregates if row.trial_key in selectable],
            direction=direction,
        )


# A shorter public alias reads well at API/CLI construction sites.
OrchestrationService = ExperimentOrchestrationService


__all__ = ["ExperimentOrchestrationService", "OrchestrationService"]
