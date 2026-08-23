"""Deterministic run-group orchestration primitives."""

from halo_forge.orchestration.capabilities import (
    DEFAULT_TRAINER_EXECUTION_CAPABILITIES,
    TrainerExecutionCapability,
    TrainerExecutionCapabilityRegistry,
    normalize_backend_family,
    resolve_trainer_execution_capability,
)
from halo_forge.orchestration.models import (
    CohortAggregate,
    CohortObservation,
    MaterializedRun,
    MaterializedTrial,
    RunGroupSpec,
    SuccessiveHalvingConfig,
    aggregate_cohort,
    canonical_fingerprint,
    canonical_json,
    config_fingerprint,
    expected_seeds_for_trials,
    materialize_trials,
    rank_cohort,
)
from halo_forge.orchestration.policies import HalvingDecision, decide_successive_halving
from halo_forge.orchestration.service import (
    ExperimentOrchestrationService,
    OrchestrationService,
)

__all__ = [
    "CohortAggregate",
    "CohortObservation",
    "DEFAULT_TRAINER_EXECUTION_CAPABILITIES",
    "ExperimentOrchestrationService",
    "HalvingDecision",
    "MaterializedRun",
    "MaterializedTrial",
    "OrchestrationService",
    "RunGroupSpec",
    "SuccessiveHalvingConfig",
    "TrainerExecutionCapability",
    "TrainerExecutionCapabilityRegistry",
    "aggregate_cohort",
    "canonical_fingerprint",
    "canonical_json",
    "config_fingerprint",
    "decide_successive_halving",
    "expected_seeds_for_trials",
    "materialize_trials",
    "normalize_backend_family",
    "rank_cohort",
    "resolve_trainer_execution_capability",
]
