"""Search space + sweep config dataclasses (Track T10)."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

# ---------- per-knob distributions -----------------------------------------


@dataclass
class Uniform:
    """Continuous uniform sample in [low, high]."""

    low: float
    high: float

    def __post_init__(self) -> None:
        self.low = float(self.low)
        self.high = float(self.high)
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("Uniform requires finite bounds")
        if self.high < self.low:
            raise ValueError("Uniform: high must be >= low")

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "uniform", "low": self.low, "high": self.high}


@dataclass
class LogUniform:
    """Log-uniform sample in [low, high]. Right shape for learning
    rates and other rate-like quantities where 1e-5 vs 1e-4 is the
    interesting comparison, not 1e-5 vs 5e-5."""

    low: float
    high: float

    def __post_init__(self) -> None:
        self.low = float(self.low)
        self.high = float(self.high)
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("LogUniform requires finite bounds")
        if self.low <= 0 or self.high <= 0:
            raise ValueError("LogUniform requires strictly positive bounds")
        if self.high < self.low:
            raise ValueError("LogUniform: high must be >= low")

    def sample(self, rng: random.Random) -> float:
        log_low = math.log(self.low)
        log_high = math.log(self.high)
        return math.exp(rng.uniform(log_low, log_high))

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "log_uniform", "low": self.low, "high": self.high}


@dataclass
class Choice:
    """Discrete choice from a fixed list."""

    values: List[Any]

    def __post_init__(self) -> None:
        self.values = list(self.values)
        if not self.values:
            raise ValueError("Choice requires at least one value")

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.values)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": "choice", "values": list(self.values)}


# ---------- search space ---------------------------------------------------


@dataclass
class SearchSpace:
    """Mapping `param_name → Distribution` for the sweep to walk.

    Example:
        SearchSpace(
            params={
                "learning_rate": LogUniform(1e-6, 1e-3),
                "batch_size": Choice([1, 2, 4]),
                "lora_rank": Choice([8, 16, 32, 64]),
                "warmup_ratio": Uniform(0.0, 0.2),
            }
        )
    """

    params: Dict[str, Any] = field(default_factory=dict)

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        return {name: self.params[name].sample(rng) for name in sorted(self.params)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: (
                self.params[name].to_dict()
                if hasattr(self.params[name], "to_dict")
                else str(self.params[name])
            )
            for name in sorted(self.params)
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchSpace":
        """Parse the JSON/YAML representation emitted by :meth:`to_dict`.

        Both the canonical ``{"learning_rate": {"kind": ...}}`` shape and
        the convenient ``{"params": {...}}`` wrapper are accepted.  Bare
        lists are treated as discrete choices so hand-authored YAML remains
        concise.  Unknown distribution kinds fail closed instead of silently
        becoming string values.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("search space must be a mapping")
        raw_params: Any = payload.get("params", payload)
        if not isinstance(raw_params, Mapping):
            raise TypeError("search_space.params must be a mapping")
        params: Dict[str, Any] = {}
        for raw_name, raw_distribution in raw_params.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("search-space parameter names cannot be empty")
            params[name] = distribution_from_dict(raw_distribution)
        return cls(params=params)


def distribution_from_dict(payload: Any) -> Any:
    """Return a supported distribution from a serializable definition."""

    if isinstance(payload, (Uniform, LogUniform, Choice)):
        return payload
    if isinstance(payload, (list, tuple)):
        return Choice(list(payload))
    if not isinstance(payload, Mapping):
        raise TypeError(
            "distribution must be a mapping with kind=uniform/log_uniform/choice "
            "or a list of choices"
        )
    kind = str(payload.get("kind") or payload.get("type") or "").strip().lower()
    kind = kind.replace("-", "_")
    if kind == "uniform":
        return Uniform(low=payload["low"], high=payload["high"])
    if kind in {"log_uniform", "loguniform"}:
        return LogUniform(low=payload["low"], high=payload["high"])
    if kind in {"choice", "categorical"}:
        values = payload.get("values", payload.get("choices"))
        if values is None:
            raise ValueError("choice distribution requires values")
        if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
            raise TypeError("choice values must be a list")
        return Choice(list(values))
    raise ValueError(f"unknown distribution kind {kind!r}")


# ---------- sweep config ---------------------------------------------------


@dataclass
class SweepConfig:
    """Top-level sweep configuration."""

    name: str
    search_space: SearchSpace
    n_trials: int = 16
    metric: str = "final_train_loss"  # field name on the trial result
    direction: str = "minimize"  # "minimize" or "maximize"
    sampler: str = "random"  # "random", "tpe" (requires optuna), "grid"
    seed: int = 42
    early_stop_after: Optional[int] = None  # stop entire sweep early if no improvement for N trials
    output_dir: Optional[str] = None  # writes trial JSONL + summary here
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.search_space, SearchSpace):
            self.search_space = SearchSpace.from_dict(self.search_space)
        self.n_trials = int(self.n_trials)
        self.seed = int(self.seed)
        self.direction = str(self.direction).strip().lower()
        self.sampler = str(self.sampler).strip().lower()
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if self.direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'")
        if self.sampler not in {"random", "tpe", "grid"}:
            raise ValueError("sampler must be random, tpe, or grid")
        if self.early_stop_after is not None:
            self.early_stop_after = int(self.early_stop_after)
            if self.early_stop_after <= 0:
                raise ValueError("early_stop_after must be positive")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # The dataclass `asdict` walks too deep; replace search_space with a
        # JSON-friendly version that knows how to serialize distributions.
        d["search_space"] = self.search_space.to_dict()
        return d

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SweepConfig":
        """Parse a persisted sweep configuration without losing types."""

        if not isinstance(payload, Mapping):
            raise TypeError("sweep config must be a mapping")
        values = dict(payload)
        if "name" not in values:
            raise ValueError("sweep config requires name")
        values["search_space"] = SearchSpace.from_dict(values.get("search_space") or {})
        return cls(**values)

    def is_better(self, candidate: float, incumbent: Optional[float]) -> bool:
        if incumbent is None:
            return True
        if self.direction == "minimize":
            return candidate < incumbent
        return candidate > incumbent


__all__ = [
    "Choice",
    "distribution_from_dict",
    "LogUniform",
    "SearchSpace",
    "SweepConfig",
    "Uniform",
]
