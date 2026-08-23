"""Stable public shapes shared by Dataset Lab evaluation services."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class EvaluationLabError(RuntimeError):
    """Base error for persistent evaluation operations."""


class EvaluationCancelled(EvaluationLabError):
    """Raised cooperatively when an evaluation cancellation is requested."""


@dataclass(frozen=True)
class ResolvedSubject:
    subject_type: str
    subject_ref: str
    subject_hash: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationMetric:
    name: str
    value: float
    direction: str = "maximize"
    suite_item_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSample:
    suite_item_id: str
    record_id: Optional[str] = None
    input: Any = None
    expected: Any = None
    output: Any = None
    score: Optional[float] = None
    passed: Optional[bool] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    verifier_trace: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Evidence semantics are explicit so aggregate task summaries and runtime
    # failures cannot be mistaken for behavioral, mineable examples.  These
    # fields are also persisted by RunDatabase; older rows surface as
    # ``evidence_kind=legacy, valid=False, mineable=False``.
    evidence_kind: str = "per_example"
    valid: bool = True
    mineable: bool = True
    generation_seed: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    runtime_versions: Dict[str, Any] = field(default_factory=dict)
    score_direction: Optional[str] = None
    score_threshold: Optional[float] = None
    coverage: Optional[float] = None
    template_hash: Optional[str] = None

    def __post_init__(self) -> None:
        self.evidence_kind = str(self.evidence_kind or "per_example").strip().lower()
        if self.evidence_kind in {
            "aggregate",
            "aggregate_task",
            "aggregate_benchmark",
            "legacy_aggregate",
        }:
            # Aggregate success means that an evaluator completed, not that a
            # particular model response passed.  Keep its numeric score for
            # reporting while withholding a per-example verdict.
            self.passed = None
            self.mineable = False
        if not self.valid:
            self.mineable = False
        if self.score_direction not in {None, "maximize", "minimize"}:
            raise ValueError("score_direction must be maximize, minimize, or None")
        for name in ("input_tokens", "output_tokens"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} cannot be negative")
            if value is not None:
                setattr(self, name, int(value))
        if self.coverage is not None:
            self.coverage = float(self.coverage)
            if not 0.0 <= self.coverage <= 1.0:
                raise ValueError("coverage must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationAdapterResult:
    metrics: List[EvaluationMetric] = field(default_factory=list)
    samples: List[EvaluationSample] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": [metric.to_dict() for metric in self.metrics],
            "samples": [sample.to_dict() for sample in self.samples],
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class EvaluationLaunch:
    evaluation: Any
    reused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        value = (
            self.evaluation.to_dict() if hasattr(self.evaluation, "to_dict") else self.evaluation
        )
        return {"evaluation": value, "reused": self.reused}


__all__ = [
    "EvaluationAdapterResult",
    "EvaluationCancelled",
    "EvaluationLabError",
    "EvaluationLaunch",
    "EvaluationMetric",
    "EvaluationSample",
    "ResolvedSubject",
]
