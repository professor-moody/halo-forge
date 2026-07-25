"""Normalized verifier invocation contracts.

This module deliberately wraps the legacy :class:`Verifier` API instead of
changing it.  Reliability work needs stricter semantics than training's
historical ``VerifyResult`` shape: rewards are validated against an immutable
contract, errors stay visible, and every invocation carries its runtime
identity.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Optional


class RewardContractError(ValueError):
    """Raised when an observation violates its declared reward contract."""


@dataclass(frozen=True)
class RewardContract:
    """Executable, credential-free reward semantics for one verifier revision."""

    minimum: float = 0.0
    maximum: float = 1.0
    direction: str = "maximize"
    threshold: Optional[float] = 0.5
    tie_policy: str = "error"
    error_behavior: str = "fail_closed"
    probability_semantics: bool = False

    def __post_init__(self) -> None:
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        threshold = None if self.threshold is None else float(self.threshold)
        if not all(math.isfinite(value) for value in (minimum, maximum)):
            raise RewardContractError("reward bounds must be finite")
        if threshold is not None and not math.isfinite(threshold):
            raise RewardContractError("reward threshold must be finite when supplied")
        if minimum >= maximum:
            raise RewardContractError("reward minimum must be lower than maximum")
        if threshold is not None and not minimum <= threshold <= maximum:
            raise RewardContractError("reward threshold must lie inside the reward range")
        if self.direction not in {"maximize", "minimize"}:
            raise RewardContractError("reward direction must be maximize or minimize")
        if self.tie_policy not in {"pass", "fail", "tie", "error"}:
            raise RewardContractError("tie_policy must be pass, fail, tie, or error")
        if self.error_behavior not in {"fail", "fail_closed", "fail_open", "error", "abstain"}:
            raise RewardContractError(
                "error_behavior must be fail_closed, fail_open, error, or abstain"
            )
        if self.probability_semantics and (minimum != 0.0 or maximum != 1.0):
            raise RewardContractError("probability semantics require the [0, 1] reward range")

    @property
    def fails_closed(self) -> bool:
        return self.error_behavior in {"fail", "fail_closed"}

    def validate_reward(self, reward: Any) -> float:
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise RewardContractError("verifier reward must be numeric")
        result = float(reward)
        if not math.isfinite(result):
            raise RewardContractError("verifier reward must be finite")
        if result < self.minimum or result > self.maximum:
            raise RewardContractError(
                f"verifier reward {result} is outside [{self.minimum}, {self.maximum}]"
            )
        return result

    def classify(self, reward: float) -> Optional[bool]:
        value = self.validate_reward(reward)
        if self.threshold is None:
            return None
        if value == self.threshold:
            if self.tie_policy == "pass":
                return True
            if self.tie_policy == "fail":
                return False
            if self.tie_policy == "error":
                raise RewardContractError("verifier reward equals an error-policy tie threshold")
            return None
        if self.direction == "maximize":
            return value > self.threshold
        return value < self.threshold

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifierObservation:
    """One normalized verifier invocation.

    ``component_trace`` is intentionally structured rather than flattened so
    an aggregate chain can never conceal a child failure.
    """

    reward: Optional[float]
    passed: Optional[bool]
    parsed_value: Any = None
    raw_output: Any = None
    details: Any = None
    component_trace: tuple[Mapping[str, Any], ...] = ()
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    runtime_identity: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "passed": self.passed,
            "parsed_value": self.parsed_value,
            "raw_output": self.raw_output,
            "details": self.details,
            "component_trace": [dict(item) for item in self.component_trace],
            "latency_ms": self.latency_ms,
            "error": self.error,
            "runtime_identity": dict(self.runtime_identity),
        }


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    fields = (
        "minimum",
        "maximum",
        "min_reward",
        "max_reward",
        "direction",
        "threshold",
        "tie_policy",
        "error_behavior",
        "probability_semantics",
    )
    return {field: getattr(value, field) for field in fields if hasattr(value, field)}


def normalize_reward_contract(value: RewardContract | Mapping[str, Any] | Any) -> RewardContract:
    """Accept public-model, mapping, or internal reward-contract shapes."""

    if isinstance(value, RewardContract):
        return value
    payload = _mapping(value)
    return RewardContract(
        minimum=float(payload.get("minimum", payload.get("min_reward", 0.0))),
        maximum=float(payload.get("maximum", payload.get("max_reward", 1.0))),
        direction=str(payload.get("direction", "maximize")).strip().lower(),
        threshold=(
            None
            if payload.get("threshold", 0.5) is None
            else float(payload.get("threshold", 0.5))
        ),
        tie_policy=str(payload.get("tie_policy", "error")).strip().lower(),
        error_behavior=str(payload.get("error_behavior", "fail_closed")).strip().lower(),
        probability_semantics=bool(payload.get("probability_semantics", False)),
    )


def normalize_verifier_result(
    result: Any,
    *,
    contract: RewardContract | Mapping[str, Any] | Any,
    latency_ms: Optional[float] = None,
    runtime_identity: Optional[Mapping[str, Any]] = None,
    raw_output: Any = None,
    component_trace: tuple[Mapping[str, Any], ...] = (),
) -> VerifierObservation:
    """Normalize legacy verifier results without silently repairing them.

    A result containing a non-finite or out-of-range reward raises
    :class:`RewardContractError`.  Callers should persist that invocation as a
    protocol error; clamping would make the reliability analysis misleading.
    """

    reward_contract = normalize_reward_contract(contract)
    payload = _mapping(result)
    error_value = payload.get("error")
    legacy_error = None if error_value in {None, ""} else str(error_value)
    reward_value = payload.get("reward")
    reward: Optional[float]
    passed: Optional[bool]
    if reward_value is None:
        reward = None
        error = legacy_error
        passed = False if error and reward_contract.fails_closed else None
    else:
        reward = reward_contract.validate_reward(reward_value)
        passed = reward_contract.classify(reward)
        # Historical VerifyResult implementations use ``error`` for a normal
        # negative verdict (for example ``invalid_json``) while also returning
        # a valid reward.  V7 reserves observation.error for invocation,
        # parser-contract, timeout, and runtime failures so negative reference
        # examples remain available reliability evidence.
        error = None

    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    explicit_parsed = payload.get("parsed_value", metadata.get("parsed_value"))
    explicit_raw = payload.get("raw_output", metadata.get("raw_output", raw_output))
    details = payload.get("details")
    legacy_success = payload.get("success")
    if legacy_success is not None:
        if isinstance(details, Mapping):
            details = {**details, "legacy_success": bool(legacy_success)}
        else:
            details = {"message": details, "legacy_success": bool(legacy_success)}
    if legacy_error is not None and reward_value is not None:
        if isinstance(details, Mapping):
            details = {**details, "legacy_verifier_error": legacy_error}
        else:
            details = {"message": details, "legacy_verifier_error": legacy_error}

    normalized_latency: Optional[float] = None
    if latency_ms is not None:
        normalized_latency = float(latency_ms)
        if not math.isfinite(normalized_latency) or normalized_latency < 0:
            raise ValueError("latency_ms must be a finite non-negative value")

    return VerifierObservation(
        reward=reward,
        passed=passed,
        parsed_value=explicit_parsed,
        raw_output=explicit_raw,
        details=details,
        component_trace=component_trace,
        latency_ms=normalized_latency,
        error=error,
        runtime_identity=dict(runtime_identity or {}),
    )


__all__ = [
    "RewardContract",
    "RewardContractError",
    "VerifierObservation",
    "normalize_reward_contract",
    "normalize_verifier_result",
]
