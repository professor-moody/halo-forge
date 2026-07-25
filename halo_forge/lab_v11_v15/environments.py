"""Versioned deterministic environment adapter registry."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class EnvironmentStepResult:
    observation: Dict[str, Any]
    state_delta: Dict[str, Any]
    reward: float
    terminal: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class EnvironmentAdapterDescriptor:
    id: str
    version: str
    deterministic: bool
    supports_snapshot: bool
    external_writes: bool


class StateMachineEnvironmentAdapter:
    descriptor = EnvironmentAdapterDescriptor(
        id="state_machine",
        version="1",
        deterministic=True,
        supports_snapshot=True,
        external_writes=False,
    )

    def __init__(self, definition: Mapping[str, Any]):
        self.definition = copy.deepcopy(dict(definition))
        self.state: Dict[str, Any] = {}
        self.seed = 42

    def reset(self, seed: int) -> Dict[str, Any]:
        self.seed = int(seed)
        random.seed(self.seed)
        self.state = copy.deepcopy(dict(self.definition.get("initial_state") or {}))
        return self._observation()

    def step(self, action: Mapping[str, Any]) -> EnvironmentStepResult:
        name = str(action.get("name") or action.get("tool") or "")
        transition = dict(
            (self.definition.get("transitions") or {}).get(name) or {}
        )
        if not transition:
            return EnvironmentStepResult(
                observation=self._observation(),
                state_delta={},
                reward=0.0,
                terminal=False,
                error=f"unknown action: {name}",
            )
        delta = copy.deepcopy(dict(transition.get("state_delta") or {}))
        for key, value in delta.items():
            if value is None:
                self.state.pop(str(key), None)
            else:
                self.state[str(key)] = value
        return EnvironmentStepResult(
            observation=self._observation(),
            state_delta=delta,
            reward=float(transition.get("reward") or 0.0),
            terminal=bool(transition.get("terminal", False)),
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "state": copy.deepcopy(self.state),
            "state_hash": self._state_hash(),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> Dict[str, Any]:
        state = copy.deepcopy(dict(snapshot.get("state") or {}))
        expected = str(snapshot.get("state_hash") or "")
        actual = hashlib.sha256(
            json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if expected and expected != actual:
            raise ValueError("environment snapshot hash does not match")
        self.seed = int(snapshot.get("seed") or 42)
        self.state = state
        return self._observation()

    def verify_final_state(
        self, expected: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        invariants = dict(expected or self.definition.get("expected_state") or {})
        failures = {
            key: {"expected": value, "observed": self.state.get(key)}
            for key, value in invariants.items()
            if self.state.get(key) != value
        }
        return {
            "passed": not failures,
            "failures": failures,
            "state_hash": self._state_hash(),
        }

    def _state_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.state, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _observation(self) -> Dict[str, Any]:
        return {
            "state": copy.deepcopy(self.state),
            "available_actions": sorted(
                dict(self.definition.get("transitions") or {})
            ),
            "state_hash": self._state_hash(),
        }


class EnvironmentAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: Dict[str, type[StateMachineEnvironmentAdapter]] = {}

    def register(
        self,
        adapter: type[StateMachineEnvironmentAdapter],
        *,
        replace: bool = False,
    ) -> None:
        identifier = adapter.descriptor.id
        if identifier in self._adapters and not replace:
            raise ValueError(f"environment adapter {identifier!r} is already registered")
        if adapter.descriptor.external_writes:
            raise ValueError("V15 environment adapters cannot enable external writes")
        self._adapters[identifier] = adapter

    def get(self, identifier: str) -> type[StateMachineEnvironmentAdapter]:
        try:
            return self._adapters[str(identifier)]
        except KeyError as exc:
            raise KeyError(f"unknown environment adapter: {identifier}") from exc

    def create(
        self, identifier: str, definition: Mapping[str, Any]
    ) -> StateMachineEnvironmentAdapter:
        return self.get(identifier)(definition)

    def list(self) -> list[EnvironmentAdapterDescriptor]:
        return [
            adapter.descriptor
            for adapter in sorted(
                self._adapters.values(), key=lambda value: value.descriptor.id
            )
        ]


ENVIRONMENT_ADAPTERS = EnvironmentAdapterRegistry()
ENVIRONMENT_ADAPTERS.register(StateMachineEnvironmentAdapter)


__all__ = [
    "ENVIRONMENT_ADAPTERS",
    "EnvironmentAdapterDescriptor",
    "EnvironmentAdapterRegistry",
    "EnvironmentStepResult",
    "StateMachineEnvironmentAdapter",
]
