"""Validation and transparent aggregation for ordered verifier chains."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, Sequence

from .observation import RewardContract, VerifierObservation, normalize_reward_contract


@dataclass(frozen=True)
class ChainComponent:
    revision_id: str
    order_index: int
    weight: float = 1.0
    veto: bool = False

    @classmethod
    def from_value(cls, value: "ChainComponent | Mapping[str, Any] | Any") -> "ChainComponent":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            payload = value
        elif is_dataclass(value):
            payload = asdict(value)
        else:
            payload = {
                name: getattr(value, name)
                for name in ("revision_id", "child_revision_id", "order_index", "weight", "veto")
                if hasattr(value, name)
            }
        revision_id = str(payload.get("revision_id", payload.get("child_revision_id", ""))).strip()
        if not revision_id:
            raise ValueError("chain component revision_id is required")
        order_index = int(payload.get("order_index", 0))
        weight = float(payload.get("weight", 1.0))
        if order_index < 0:
            raise ValueError("chain component order_index cannot be negative")
        if weight <= 0:
            raise ValueError("chain component weight must be positive")
        return cls(
            revision_id=revision_id,
            order_index=order_index,
            weight=weight,
            veto=bool(payload.get("veto", False)),
        )


def normalize_components(
    values: Sequence[ChainComponent | Mapping[str, Any] | Any],
) -> tuple[ChainComponent, ...]:
    result = tuple(sorted((ChainComponent.from_value(value) for value in values), key=lambda v: v.order_index))
    orders = [value.order_index for value in result]
    if len(set(orders)) != len(orders):
        raise ValueError("chain component order_index values must be unique")
    revisions = [value.revision_id for value in result]
    if len(set(revisions)) != len(revisions):
        raise ValueError("a verifier revision may appear only once in one chain")
    if not result:
        raise ValueError("a chain must contain at least one component")
    return result


def validate_chain_graph(
    root_revision_id: str,
    components_by_revision: Mapping[str, Sequence[ChainComponent | Mapping[str, Any] | Any]],
    *,
    qualification_by_revision: Mapping[str, str],
) -> tuple[ChainComponent, ...]:
    """Validate order, cycles, and child candidate qualification recursively."""

    root = str(root_revision_id).strip()
    if not root:
        raise ValueError("root_revision_id is required")
    normalized: dict[str, tuple[ChainComponent, ...]] = {
        str(revision_id): normalize_components(components)
        for revision_id, components in components_by_revision.items()
    }
    if root not in normalized:
        raise ValueError("root verifier revision does not define chain components")

    visiting: list[str] = []
    visited: set[str] = set()

    def visit(revision_id: str) -> None:
        if revision_id in visiting:
            cycle_start = visiting.index(revision_id)
            cycle = visiting[cycle_start:] + [revision_id]
            raise ValueError("verifier chain cycle: " + " -> ".join(cycle))
        if revision_id in visited:
            return
        visiting.append(revision_id)
        for component in normalized.get(revision_id, ()):
            status = str(qualification_by_revision.get(component.revision_id, "")).lower()
            if status not in {"candidate", "approved"}:
                raise ValueError(
                    f"chain child {component.revision_id!r} must be candidate-qualified"
                )
            if component.revision_id in normalized:
                visit(component.revision_id)
        visiting.pop()
        visited.add(revision_id)

    visit(root)
    return normalized[root]


def _observation_mapping(value: VerifierObservation | Mapping[str, Any]) -> Mapping[str, Any]:
    return value.to_dict() if isinstance(value, VerifierObservation) else value


def aggregate_chain_observations(
    components: Sequence[ChainComponent | Mapping[str, Any] | Any],
    observations: Mapping[str, VerifierObservation | Mapping[str, Any]],
    *,
    aggregation: str,
    contract: RewardContract | Mapping[str, Any] | Any,
    runtime_identity: Mapping[str, Any] | None = None,
) -> VerifierObservation:
    """Aggregate a chain while preserving every component result and error."""

    ordered = normalize_components(components)
    missing = [item.revision_id for item in ordered if item.revision_id not in observations]
    if missing:
        raise ValueError("missing chain observations: " + ", ".join(missing))
    rule = str(aggregation).strip().lower()
    if rule not in {"weighted_mean", "mean", "minimum", "maximum", "all", "any"}:
        raise ValueError("unsupported chain aggregation rule")
    reward_contract = normalize_reward_contract(contract)

    trace: list[Mapping[str, Any]] = []
    errors: list[str] = []
    rewards: list[tuple[float, float]] = []
    veto_failure = False
    latencies: list[float] = []
    for component in ordered:
        value = _observation_mapping(observations[component.revision_id])
        component_error = value.get("error")
        if component_error:
            errors.append(f"{component.revision_id}: {component_error}")
        reward = value.get("reward")
        if reward is not None:
            rewards.append((reward_contract.validate_reward(reward), component.weight))
        if component.veto and value.get("passed") is not True:
            veto_failure = True
        latency = value.get("latency_ms")
        if latency is not None:
            latencies.append(float(latency))
        trace.append(
            {
                "revision_id": component.revision_id,
                "order_index": component.order_index,
                "weight": component.weight,
                "veto": component.veto,
                "observation": dict(value),
            }
        )

    total_latency = sum(latencies) if latencies else None
    if errors:
        return VerifierObservation(
            reward=None,
            passed=False if reward_contract.fails_closed else None,
            details={"aggregation": rule, "component_count": len(ordered)},
            component_trace=tuple(trace),
            latency_ms=total_latency,
            error="; ".join(errors),
            runtime_identity=dict(runtime_identity or {}),
        )
    if len(rewards) != len(ordered):
        return VerifierObservation(
            reward=None,
            passed=None,
            details={"aggregation": rule, "component_count": len(ordered)},
            component_trace=tuple(trace),
            latency_ms=total_latency,
            error="one or more chain components abstained without a reward",
            runtime_identity=dict(runtime_identity or {}),
        )

    values = [value for value, _ in rewards]
    if rule == "weighted_mean":
        total_weight = sum(weight for _, weight in rewards)
        aggregate = sum(value * weight for value, weight in rewards) / total_weight
    elif rule == "mean":
        aggregate = statistics.fmean(values)
    elif rule == "minimum":
        aggregate = min(values)
    elif rule == "maximum":
        aggregate = max(values)
    elif rule == "all":
        aggregate = reward_contract.maximum if all(
            _observation_mapping(observations[item.revision_id]).get("passed") is True
            for item in ordered
        ) else reward_contract.minimum
    else:  # any
        aggregate = reward_contract.maximum if any(
            _observation_mapping(observations[item.revision_id]).get("passed") is True
            for item in ordered
        ) else reward_contract.minimum

    aggregate = reward_contract.validate_reward(aggregate)
    passed = False if veto_failure else reward_contract.classify(aggregate)
    return VerifierObservation(
        reward=aggregate,
        passed=passed,
        parsed_value=aggregate,
        details={
            "aggregation": rule,
            "component_count": len(ordered),
            "veto_triggered": veto_failure,
        },
        component_trace=tuple(trace),
        latency_ms=total_latency,
        runtime_identity=dict(runtime_identity or {}),
    )


__all__ = [
    "ChainComponent",
    "aggregate_chain_observations",
    "normalize_components",
    "validate_chain_graph",
]
