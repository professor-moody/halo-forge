"""Immutable qualification and serving profile revisions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ._canonical import FrozenJsonMap, content_fingerprint
from .performance import PerformanceSettings

METRIC_DIRECTIONS = frozenset({"maximize", "minimize"})


def _optional_finite(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class QualificationMetricRule:
    """One absolute and/or relative gate for a named metric.

    ``maximum_regression`` is an absolute adverse delta. For a maximize metric,
    candidate - parent must be at least ``-maximum_regression``. For a minimize
    metric, parent - candidate must meet the same bound.
    """

    metric: str
    direction: str
    pass_threshold: Optional[float] = None
    warn_threshold: Optional[float] = None
    maximum_regression: Optional[float] = None
    required: bool = True

    def __post_init__(self) -> None:
        metric = str(self.metric).strip()
        direction = str(self.direction).strip().lower()
        if not metric:
            raise ValueError("metric cannot be empty")
        if direction not in METRIC_DIRECTIONS:
            raise ValueError("direction must be maximize or minimize")
        pass_threshold = _optional_finite(self.pass_threshold, "pass_threshold")
        warn_threshold = _optional_finite(self.warn_threshold, "warn_threshold")
        maximum_regression = _optional_finite(self.maximum_regression, "maximum_regression")
        if pass_threshold is None and maximum_regression is None:
            raise ValueError("a metric rule requires pass_threshold or maximum_regression")
        if maximum_regression is not None and maximum_regression < 0:
            raise ValueError("maximum_regression must be non-negative")
        if warn_threshold is not None and pass_threshold is None:
            raise ValueError("warn_threshold requires pass_threshold")
        if warn_threshold is not None and pass_threshold is not None:
            if direction == "maximize" and warn_threshold > pass_threshold:
                raise ValueError("maximize warn_threshold must not exceed pass_threshold")
            if direction == "minimize" and warn_threshold < pass_threshold:
                raise ValueError("minimize warn_threshold must not be below pass_threshold")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a boolean")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "pass_threshold", pass_threshold)
        object.__setattr__(self, "warn_threshold", warn_threshold)
        object.__setattr__(self, "maximum_regression", maximum_regression)
        object.__setattr__(self, "required", self.required)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "pass_threshold": self.pass_threshold,
            "warn_threshold": self.warn_threshold,
            "maximum_regression": self.maximum_regression,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QualificationMetricRule":
        return cls(**dict(payload))


def _normalize_rules(
    rules: Sequence[QualificationMetricRule | Mapping[str, Any]], *, stage: str
) -> Tuple[QualificationMetricRule, ...]:
    normalized = tuple(
        (
            rule
            if isinstance(rule, QualificationMetricRule)
            else QualificationMetricRule.from_dict(rule)
        )
        for rule in rules
    )
    names = [rule.metric for rule in normalized]
    if len(names) != len(set(names)):
        raise ValueError(f"{stage} metric names must be unique")
    return normalized


@dataclass(frozen=True)
class QualificationProfileRevision:
    """Immutable scientific definition for artifact qualification."""

    profile_id: str
    revision_number: int
    name: str
    development_suite_revision_id: str
    operational_suite_revision_id: str
    development_rules: Tuple[QualificationMetricRule, ...]
    operational_rules: Tuple[QualificationMetricRule, ...]
    target_backend: str
    holdout_suite_revision_id: Optional[str] = None
    holdout_rules: Tuple[QualificationMetricRule, ...] = ()
    generation_settings: FrozenJsonMap = field(default_factory=FrozenJsonMap)
    performance_settings: PerformanceSettings = field(default_factory=PerformanceSettings)
    version: int = 1
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        name = str(self.name).strip()
        development_suite = str(self.development_suite_revision_id).strip()
        operational_suite = str(self.operational_suite_revision_id).strip()
        target_backend = str(self.target_backend).strip().lower()
        holdout_suite = (
            None
            if self.holdout_suite_revision_id is None
            else str(self.holdout_suite_revision_id).strip() or None
        )
        if not profile_id:
            raise ValueError("profile_id cannot be empty")
        if not name:
            raise ValueError("name cannot be empty")
        if isinstance(self.revision_number, bool) or not isinstance(self.revision_number, int):
            raise TypeError("revision_number must be an integer")
        if self.revision_number <= 0:
            raise ValueError("revision_number must be positive")
        if not development_suite or not operational_suite:
            raise ValueError("development and operational suite revisions are required")
        suites = [development_suite, operational_suite]
        if holdout_suite:
            suites.append(holdout_suite)
        if len(suites) != len(set(suites)):
            raise ValueError("qualification stages must use distinct suite revisions")
        if not target_backend:
            raise ValueError("target_backend cannot be empty")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("version must be an integer")
        if self.version != 1:
            raise ValueError(f"unsupported qualification profile version {self.version}")
        development_rules = _normalize_rules(self.development_rules, stage="development")
        operational_rules = _normalize_rules(self.operational_rules, stage="operational")
        holdout_rules = _normalize_rules(self.holdout_rules, stage="holdout")
        if not development_rules:
            raise ValueError("at least one development metric rule is required")
        if not operational_rules:
            raise ValueError("at least one operational metric rule is required")
        if bool(holdout_suite) != bool(holdout_rules):
            raise ValueError("holdout suite and holdout rules must be configured together")
        generation_settings = (
            self.generation_settings
            if isinstance(self.generation_settings, FrozenJsonMap)
            else FrozenJsonMap(self.generation_settings)
        )
        performance_settings = (
            self.performance_settings
            if isinstance(self.performance_settings, PerformanceSettings)
            else PerformanceSettings.from_dict(self.performance_settings)
        )
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "development_suite_revision_id", development_suite)
        object.__setattr__(self, "operational_suite_revision_id", operational_suite)
        object.__setattr__(self, "holdout_suite_revision_id", holdout_suite)
        object.__setattr__(self, "target_backend", target_backend)
        object.__setattr__(self, "development_rules", development_rules)
        object.__setattr__(self, "operational_rules", operational_rules)
        object.__setattr__(self, "holdout_rules", holdout_rules)
        object.__setattr__(self, "generation_settings", generation_settings)
        object.__setattr__(self, "performance_settings", performance_settings)
        object.__setattr__(self, "content_hash", content_fingerprint(self.definition_dict()))

    @property
    def holdout_required(self) -> bool:
        return self.holdout_suite_revision_id is not None

    def definition_dict(self) -> Dict[str, Any]:
        """Return only fields that affect scientific comparability."""

        return {
            "version": self.version,
            "development_suite_revision_id": self.development_suite_revision_id,
            "operational_suite_revision_id": self.operational_suite_revision_id,
            "holdout_suite_revision_id": self.holdout_suite_revision_id,
            "development_rules": [rule.to_dict() for rule in self.development_rules],
            "operational_rules": [rule.to_dict() for rule in self.operational_rules],
            "holdout_rules": [rule.to_dict() for rule in self.holdout_rules],
            "target_backend": self.target_backend,
            "generation_settings": self.generation_settings.to_dict(),
            "performance_settings": self.performance_settings.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "revision_number": self.revision_number,
            "name": self.name,
            **self.definition_dict(),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QualificationProfileRevision":
        values = dict(payload)
        expected_hash = values.pop("content_hash", None)
        result = cls(
            profile_id=values["profile_id"],
            revision_number=values["revision_number"],
            name=values["name"],
            development_suite_revision_id=values["development_suite_revision_id"],
            operational_suite_revision_id=values["operational_suite_revision_id"],
            holdout_suite_revision_id=values.get("holdout_suite_revision_id"),
            development_rules=tuple(values.get("development_rules") or ()),
            operational_rules=tuple(values.get("operational_rules") or ()),
            holdout_rules=tuple(values.get("holdout_rules") or ()),
            target_backend=values["target_backend"],
            generation_settings=FrozenJsonMap(values.get("generation_settings")),
            performance_settings=PerformanceSettings.from_dict(values.get("performance_settings")),
            version=values.get("version", 1),
        )
        if expected_hash is not None and expected_hash != result.content_hash:
            raise ValueError("qualification profile content_hash does not match its definition")
        return result


@dataclass(frozen=True)
class ServingProfileRevision:
    """Immutable local serving configuration pinned to one artifact."""

    profile_id: str
    revision_number: int
    name: str
    artifact_id: str
    artifact_hash: str
    backend: str
    endpoint_settings: FrozenJsonMap = field(default_factory=FrozenJsonMap)
    chat_template: Optional[str] = None
    generation_defaults: FrozenJsonMap = field(default_factory=FrozenJsonMap)
    resource_expectations: FrozenJsonMap = field(default_factory=FrozenJsonMap)
    version: int = 1
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text_fields = {
            "profile_id": self.profile_id,
            "name": self.name,
            "artifact_id": self.artifact_id,
            "artifact_hash": self.artifact_hash,
            "backend": self.backend,
        }
        normalized = {key: str(value).strip() for key, value in text_fields.items()}
        for key, value in normalized.items():
            if not value:
                raise ValueError(f"{key} cannot be empty")
        if isinstance(self.revision_number, bool) or not isinstance(self.revision_number, int):
            raise TypeError("revision_number must be an integer")
        if self.revision_number <= 0:
            raise ValueError("revision_number must be positive")
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise TypeError("version must be an integer")
        if self.version != 1:
            raise ValueError(f"unsupported serving profile version {self.version}")
        for key, value in normalized.items():
            object.__setattr__(self, key, value.lower() if key == "backend" else value)
        for key in ("endpoint_settings", "generation_defaults", "resource_expectations"):
            value = getattr(self, key)
            object.__setattr__(
                self,
                key,
                value if isinstance(value, FrozenJsonMap) else FrozenJsonMap(value),
            )
        chat_template = None if self.chat_template is None else str(self.chat_template)
        object.__setattr__(self, "chat_template", chat_template)
        object.__setattr__(self, "content_hash", content_fingerprint(self.definition_dict()))

    def definition_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "artifact_id": self.artifact_id,
            "artifact_hash": self.artifact_hash,
            "backend": self.backend,
            "endpoint_settings": self.endpoint_settings.to_dict(),
            "chat_template": self.chat_template,
            "generation_defaults": self.generation_defaults.to_dict(),
            "resource_expectations": self.resource_expectations.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "revision_number": self.revision_number,
            "name": self.name,
            **self.definition_dict(),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ServingProfileRevision":
        values = dict(payload)
        expected_hash = values.pop("content_hash", None)
        result = cls(
            profile_id=values["profile_id"],
            revision_number=values["revision_number"],
            name=values["name"],
            artifact_id=values["artifact_id"],
            artifact_hash=values["artifact_hash"],
            backend=values["backend"],
            endpoint_settings=FrozenJsonMap(values.get("endpoint_settings")),
            chat_template=values.get("chat_template"),
            generation_defaults=FrozenJsonMap(values.get("generation_defaults")),
            resource_expectations=FrozenJsonMap(values.get("resource_expectations")),
            version=values.get("version", 1),
        )
        if expected_hash is not None and expected_hash != result.content_hash:
            raise ValueError("serving profile content_hash does not match its definition")
        return result


__all__ = [
    "METRIC_DIRECTIONS",
    "QualificationMetricRule",
    "QualificationProfileRevision",
    "ServingProfileRevision",
]
