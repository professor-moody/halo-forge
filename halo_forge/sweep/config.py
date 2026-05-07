"""Search space + sweep config dataclasses (Track T10)."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ---------- per-knob distributions -----------------------------------------


@dataclass
class Uniform:
    """Continuous uniform sample in [low, high]."""

    low: float
    high: float

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

    def __post_init__(self):
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

    def __post_init__(self):
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
        return {name: dist.sample(rng) for name, dist in self.params.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: dist.to_dict() if hasattr(dist, "to_dict") else str(dist)
            for name, dist in self.params.items()
        }


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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # The dataclass `asdict` walks too deep; replace search_space with a
        # JSON-friendly version that knows how to serialize distributions.
        d["search_space"] = self.search_space.to_dict()
        return d

    def is_better(self, candidate: float, incumbent: Optional[float]) -> bool:
        if incumbent is None:
            return True
        if self.direction == "minimize":
            return candidate < incumbent
        return candidate > incumbent


__all__ = [
    "Choice",
    "LogUniform",
    "SearchSpace",
    "SweepConfig",
    "Uniform",
]
