"""Deterministic and dependency-free inference performance evidence."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from ._canonical import FrozenJsonMap

DEFAULT_WARMUP_RUNS = 2
DEFAULT_MEASURED_REPEATS = 5
DEFAULT_CONCURRENCY = 1
DEFAULT_GENERATION_SEED = 42

_METRIC_POLICIES = {
    "load_time_ms": "median",
    "time_to_first_token_ms": "median",
    "total_latency_ms": "median",
    "output_tokens_per_second": "median",
    "peak_process_memory_bytes": "maximum",
    "peak_system_memory_bytes": "maximum",
    "peak_device_memory_bytes": "maximum",
    "artifact_size_bytes": "maximum",
}
PERFORMANCE_METRIC_POLICIES = FrozenJsonMap(_METRIC_POLICIES)


def _integer(value: Any, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _optional_number(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _optional_count(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer or None")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class PerformanceSettings:
    """The reproducible execution policy for one performance evaluation."""

    warmup_runs: int = DEFAULT_WARMUP_RUNS
    measured_repeats: int = DEFAULT_MEASURED_REPEATS
    concurrency: int = DEFAULT_CONCURRENCY
    generation_seed: int = DEFAULT_GENERATION_SEED

    def __post_init__(self) -> None:
        warmup_runs = _integer(self.warmup_runs, "warmup_runs", minimum=0)
        measured_repeats = _integer(self.measured_repeats, "measured_repeats", minimum=1)
        concurrency = _integer(self.concurrency, "concurrency", minimum=1)
        generation_seed = _integer(self.generation_seed, "generation_seed")
        if concurrency != 1:
            raise ValueError("qualification performance measurements require concurrency=1")
        object.__setattr__(self, "warmup_runs", warmup_runs)
        object.__setattr__(self, "measured_repeats", measured_repeats)
        object.__setattr__(self, "concurrency", 1)
        object.__setattr__(self, "generation_seed", generation_seed)

    def to_dict(self) -> Dict[str, int]:
        return {
            "warmup_runs": self.warmup_runs,
            "measured_repeats": self.measured_repeats,
            "concurrency": self.concurrency,
            "generation_seed": self.generation_seed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "PerformanceSettings":
        values = dict(payload or {})
        return cls(
            warmup_runs=values.get("warmup_runs", DEFAULT_WARMUP_RUNS),
            measured_repeats=values.get("measured_repeats", DEFAULT_MEASURED_REPEATS),
            concurrency=values.get("concurrency", DEFAULT_CONCURRENCY),
            generation_seed=values.get("generation_seed", DEFAULT_GENERATION_SEED),
        )


@dataclass(frozen=True)
class PerformanceRunRequest:
    artifact_ref: str
    backend: str
    prompt: Any
    phase: str
    iteration: int
    generation_seed: int
    generation_settings: FrozenJsonMap = field(default_factory=FrozenJsonMap)

    def __post_init__(self) -> None:
        artifact_ref = str(self.artifact_ref).strip()
        backend = str(self.backend).strip()
        phase = str(self.phase).strip().lower()
        if not artifact_ref:
            raise ValueError("artifact_ref cannot be empty")
        if not backend:
            raise ValueError("backend cannot be empty")
        if phase not in {"warmup", "measure"}:
            raise ValueError("phase must be warmup or measure")
        iteration = _integer(self.iteration, "iteration", minimum=0)
        generation_seed = _integer(self.generation_seed, "generation_seed")
        settings = (
            self.generation_settings
            if isinstance(self.generation_settings, FrozenJsonMap)
            else FrozenJsonMap(self.generation_settings)
        )
        object.__setattr__(self, "artifact_ref", artifact_ref)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "generation_seed", generation_seed)
        object.__setattr__(self, "generation_settings", settings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_ref": self.artifact_ref,
            "backend": self.backend,
            "prompt": self.prompt,
            "phase": self.phase,
            "iteration": self.iteration,
            "generation_seed": self.generation_seed,
            "generation_settings": self.generation_settings.to_dict(),
        }


@dataclass(frozen=True)
class PerformanceSample:
    """One warmup or measured run; unavailable readings remain ``None``."""

    phase: str
    iteration: int
    generation_seed: int
    load_time_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    output_tokens_per_second: Optional[float] = None
    peak_process_memory_bytes: Optional[int] = None
    peak_system_memory_bytes: Optional[int] = None
    peak_device_memory_bytes: Optional[int] = None
    artifact_size_bytes: Optional[int] = None
    error: Optional[str] = None
    runtime_versions: FrozenJsonMap = field(default_factory=FrozenJsonMap)
    hardware_identity: FrozenJsonMap = field(default_factory=FrozenJsonMap)

    def __post_init__(self) -> None:
        phase = str(self.phase).strip().lower()
        if phase not in {"warmup", "measure"}:
            raise ValueError("phase must be warmup or measure")
        iteration = _integer(self.iteration, "iteration", minimum=0)
        generation_seed = _integer(self.generation_seed, "generation_seed")
        for name in (
            "load_time_ms",
            "time_to_first_token_ms",
            "total_latency_ms",
            "output_tokens_per_second",
        ):
            object.__setattr__(self, name, _optional_number(getattr(self, name), name))
        for name in (
            "input_tokens",
            "output_tokens",
            "peak_process_memory_bytes",
            "peak_system_memory_bytes",
            "peak_device_memory_bytes",
            "artifact_size_bytes",
        ):
            object.__setattr__(self, name, _optional_count(getattr(self, name), name))
        runtime_versions = (
            self.runtime_versions
            if isinstance(self.runtime_versions, FrozenJsonMap)
            else FrozenJsonMap(self.runtime_versions)
        )
        hardware_identity = (
            self.hardware_identity
            if isinstance(self.hardware_identity, FrozenJsonMap)
            else FrozenJsonMap(self.hardware_identity)
        )
        error = None if self.error is None else str(self.error).strip() or None
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "generation_seed", generation_seed)
        object.__setattr__(self, "runtime_versions", runtime_versions)
        object.__setattr__(self, "hardware_identity", hardware_identity)
        object.__setattr__(self, "error", error)

    @property
    def successful(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "iteration": self.iteration,
            "generation_seed": self.generation_seed,
            "load_time_ms": self.load_time_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "total_latency_ms": self.total_latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "output_tokens_per_second": self.output_tokens_per_second,
            "peak_process_memory_bytes": self.peak_process_memory_bytes,
            "peak_system_memory_bytes": self.peak_system_memory_bytes,
            "peak_device_memory_bytes": self.peak_device_memory_bytes,
            "artifact_size_bytes": self.artifact_size_bytes,
            "error": self.error,
            "runtime_versions": self.runtime_versions.to_dict(),
            "hardware_identity": self.hardware_identity.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PerformanceSample":
        return cls(**dict(payload))


@dataclass(frozen=True)
class PerformanceMetricAggregate:
    name: str
    count: int
    minimum: Optional[float]
    maximum: Optional[float]
    mean: Optional[float]
    median: Optional[float]
    policy: str
    policy_value: Optional[float]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("performance metric aggregate name cannot be empty")
        count = _integer(self.count, "count", minimum=0)
        if self.policy not in {"median", "maximum"}:
            raise ValueError("performance metric policy must be median or maximum")
        values = (self.minimum, self.maximum, self.mean, self.median, self.policy_value)
        if count == 0 and any(value is not None for value in values):
            raise ValueError("an unavailable metric aggregate cannot contain values")
        if count > 0 and any(value is None for value in values):
            raise ValueError("an available metric aggregate requires complete statistics")
        normalized = tuple(_optional_number(value, self.name) for value in values)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "count", count)
        for field_name, value in zip(
            ("minimum", "maximum", "mean", "median", "policy_value"), normalized
        ):
            object.__setattr__(self, field_name, value)

    @classmethod
    def from_values(
        cls, name: str, values: Sequence[float], *, policy: str
    ) -> "PerformanceMetricAggregate":
        normalized = tuple(float(value) for value in values)
        if not normalized:
            return cls(name, 0, None, None, None, None, policy, None)
        minimum = min(normalized)
        maximum = max(normalized)
        mean = statistics.fmean(normalized)
        median = statistics.median(normalized)
        policy_value = median if policy == "median" else maximum
        return cls(name, len(normalized), minimum, maximum, mean, median, policy, policy_value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "policy": self.policy,
            "policy_value": self.policy_value,
        }


@dataclass(frozen=True)
class PerformanceAggregate:
    settings: PerformanceSettings
    samples: Tuple[PerformanceSample, ...]
    warmup_count: int
    measured_count: int
    successful_count: int
    failed_count: int
    metrics: Tuple[PerformanceMetricAggregate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.settings, PerformanceSettings):
            raise TypeError("settings must be PerformanceSettings")
        samples = tuple(self.samples)
        metrics = tuple(self.metrics)
        if any(not isinstance(sample, PerformanceSample) for sample in samples):
            raise TypeError("samples must contain PerformanceSample values")
        if any(not isinstance(metric, PerformanceMetricAggregate) for metric in metrics):
            raise TypeError("metrics must contain PerformanceMetricAggregate values")
        metric_names = [metric.name for metric in metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("performance aggregate metric names must be unique")
        expected = {
            "warmup_count": sum(sample.phase == "warmup" for sample in samples),
            "measured_count": sum(sample.phase == "measure" for sample in samples),
            "successful_count": sum(
                sample.phase == "measure" and sample.successful for sample in samples
            ),
        }
        expected["failed_count"] = expected["measured_count"] - expected["successful_count"]
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} does not match performance samples")
        successful = tuple(
            sample for sample in samples if sample.phase == "measure" and sample.successful
        )
        expected_metrics = []
        for name, policy in _METRIC_POLICIES.items():
            values = [
                float(getattr(sample, name))
                for sample in successful
                if getattr(sample, name) is not None
            ]
            expected_metrics.append(
                PerformanceMetricAggregate.from_values(name, values, policy=policy)
            )
        if metrics != tuple(expected_metrics):
            raise ValueError("performance aggregate metrics do not match measured samples")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "metrics", metrics)

    @classmethod
    def from_samples(
        cls, settings: PerformanceSettings, samples: Sequence[PerformanceSample]
    ) -> "PerformanceAggregate":
        normalized = tuple(samples)
        measured = tuple(sample for sample in normalized if sample.phase == "measure")
        successful = tuple(sample for sample in measured if sample.successful)
        metrics = []
        for name, policy in _METRIC_POLICIES.items():
            values = [
                float(getattr(sample, name))
                for sample in successful
                if getattr(sample, name) is not None
            ]
            metrics.append(PerformanceMetricAggregate.from_values(name, values, policy=policy))
        return cls(
            settings=settings,
            samples=normalized,
            warmup_count=sum(sample.phase == "warmup" for sample in normalized),
            measured_count=len(measured),
            successful_count=len(successful),
            failed_count=len(measured) - len(successful),
            metrics=tuple(metrics),
        )

    def metric_values(self) -> Dict[str, Optional[float]]:
        """Return each policy value, preserving unavailable metrics as ``None``."""

        return {metric.name: metric.policy_value for metric in self.metrics}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "settings": self.settings.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
            "warmup_count": self.warmup_count,
            "measured_count": self.measured_count,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "metric_values": self.metric_values(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PerformanceAggregate":
        values = dict(payload)
        result = cls.from_samples(
            PerformanceSettings.from_dict(values.get("settings")),
            [PerformanceSample.from_dict(sample) for sample in values.get("samples") or ()],
        )
        for name in ("warmup_count", "measured_count", "successful_count", "failed_count"):
            if name in values and values[name] != getattr(result, name):
                raise ValueError(f"stored {name} does not match performance samples")
        if "metrics" in values and values["metrics"] != [
            metric.to_dict() for metric in result.metrics
        ]:
            raise ValueError("stored performance metrics do not match performance samples")
        return result


PerformanceRunner = Callable[[PerformanceRunRequest], Mapping[str, Any] | PerformanceSample]


class InferencePerformanceAdapter:
    """Sequential benchmark adapter whose inference runner is injected by callers."""

    def __init__(
        self,
        runner: PerformanceRunner,
        *,
        settings: PerformanceSettings | None = None,
    ) -> None:
        if not callable(runner):
            raise TypeError("runner must be callable")
        self._runner = runner
        self.settings = settings or PerformanceSettings()

    def run(
        self,
        *,
        artifact_ref: str,
        backend: str,
        prompt: Any,
        generation_settings: Mapping[str, Any] | None = None,
        artifact_size_bytes: Optional[int] = None,
    ) -> PerformanceAggregate:
        samples = []
        phases = [("warmup", index) for index in range(self.settings.warmup_runs)] + [
            ("measure", index) for index in range(self.settings.measured_repeats)
        ]
        for phase, iteration in phases:
            request = PerformanceRunRequest(
                artifact_ref=artifact_ref,
                backend=backend,
                prompt=prompt,
                phase=phase,
                iteration=iteration,
                generation_seed=self.settings.generation_seed,
                generation_settings=FrozenJsonMap(generation_settings),
            )
            try:
                result = self._runner(request)
                sample = self._sample_from_result(request, result, artifact_size_bytes)
            except Exception as exc:  # Runner failures are evidence, not adapter crashes.
                sample = PerformanceSample(
                    phase=phase,
                    iteration=iteration,
                    generation_seed=self.settings.generation_seed,
                    artifact_size_bytes=artifact_size_bytes,
                    error=f"{type(exc).__name__}: {exc}",
                )
            samples.append(sample)
        return PerformanceAggregate.from_samples(self.settings, samples)

    @staticmethod
    def _sample_from_result(
        request: PerformanceRunRequest,
        result: Mapping[str, Any] | PerformanceSample,
        artifact_size_bytes: Optional[int],
    ) -> PerformanceSample:
        if isinstance(result, PerformanceSample):
            if (
                result.phase != request.phase
                or result.iteration != request.iteration
                or result.generation_seed != request.generation_seed
            ):
                raise ValueError("runner sample identity does not match its request")
            if artifact_size_bytes is not None:
                if (
                    result.artifact_size_bytes is not None
                    and result.artifact_size_bytes != artifact_size_bytes
                ):
                    raise ValueError("runner artifact_size_bytes does not match the artifact")
                if result.artifact_size_bytes is None:
                    values = result.to_dict()
                    values["artifact_size_bytes"] = artifact_size_bytes
                    return PerformanceSample.from_dict(values)
            return result
        if not isinstance(result, Mapping):
            raise TypeError("runner must return a mapping or PerformanceSample")
        values = dict(result)
        if "ttft_ms" in values and "time_to_first_token_ms" not in values:
            values["time_to_first_token_ms"] = values.pop("ttft_ms")
        allowed = {
            "load_time_ms",
            "time_to_first_token_ms",
            "total_latency_ms",
            "input_tokens",
            "output_tokens",
            "output_tokens_per_second",
            "peak_process_memory_bytes",
            "peak_system_memory_bytes",
            "peak_device_memory_bytes",
            "artifact_size_bytes",
            "error",
            "runtime_versions",
            "hardware_identity",
        }
        unexpected = sorted(set(values) - allowed)
        if unexpected:
            raise ValueError(f"runner returned unsupported fields: {', '.join(unexpected)}")
        values.setdefault("artifact_size_bytes", artifact_size_bytes)
        return PerformanceSample(
            phase=request.phase,
            iteration=request.iteration,
            generation_seed=request.generation_seed,
            **values,
        )


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_GENERATION_SEED",
    "DEFAULT_MEASURED_REPEATS",
    "DEFAULT_WARMUP_RUNS",
    "InferencePerformanceAdapter",
    "PerformanceAggregate",
    "PERFORMANCE_METRIC_POLICIES",
    "PerformanceMetricAggregate",
    "PerformanceRunRequest",
    "PerformanceRunner",
    "PerformanceSample",
    "PerformanceSettings",
]
