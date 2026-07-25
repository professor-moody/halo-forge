"""Pure domain primitives for artifact qualification and serving profiles.

This package intentionally has no database, process, HTTP, or trainer
dependencies. Persistence and scheduler integrations can consume these stable
types without changing their scientific decision semantics.
"""

from .decisions import (
    ArtifactQualification,
    ParetoPoint,
    ParetoResult,
    PromotionEligibility,
    QualificationComparison,
    QualificationDecision,
    QualificationEvidence,
    QualificationMetricDecision,
    QualificationMetricDelta,
    QualificationStageDecision,
    compare_qualification_evidence,
    ensure_same_profile,
    evaluate_qualification,
    pareto_front,
    promotion_eligibility,
)
from .performance import (
    DEFAULT_CONCURRENCY,
    DEFAULT_GENERATION_SEED,
    DEFAULT_MEASURED_REPEATS,
    DEFAULT_WARMUP_RUNS,
    PERFORMANCE_METRIC_POLICIES,
    InferencePerformanceAdapter,
    PerformanceAggregate,
    PerformanceMetricAggregate,
    PerformanceRunRequest,
    PerformanceRunner,
    PerformanceSample,
    PerformanceSettings,
)
from .profiles import (
    METRIC_DIRECTIONS,
    QualificationMetricRule,
    QualificationProfileRevision,
    ServingProfileRevision,
)
from .executor import EvaluationQualificationExecutor

__all__ = [
    "ArtifactQualification",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_GENERATION_SEED",
    "DEFAULT_MEASURED_REPEATS",
    "DEFAULT_WARMUP_RUNS",
    "InferencePerformanceAdapter",
    "EvaluationQualificationExecutor",
    "METRIC_DIRECTIONS",
    "ParetoPoint",
    "ParetoResult",
    "PERFORMANCE_METRIC_POLICIES",
    "PerformanceAggregate",
    "PerformanceMetricAggregate",
    "PerformanceRunRequest",
    "PerformanceRunner",
    "PerformanceSample",
    "PerformanceSettings",
    "PromotionEligibility",
    "QualificationComparison",
    "QualificationDecision",
    "QualificationEvidence",
    "QualificationMetricDecision",
    "QualificationMetricDelta",
    "QualificationMetricRule",
    "QualificationProfileRevision",
    "QualificationStageDecision",
    "ServingProfileRevision",
    "compare_qualification_evidence",
    "ensure_same_profile",
    "evaluate_qualification",
    "pareto_front",
    "promotion_eligibility",
]
