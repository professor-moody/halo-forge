"""Sweep runner — sample trials, run them, aggregate, persist (Track T10)."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from halo_forge.sweep.config import SweepConfig

logger = logging.getLogger(__name__)


# A trial runner is anything ``(trial_id: int, params: dict) -> dict``.
# Returns the metric dict — the runner is what trains the model and
# reports `{"final_train_loss": 0.42, ...}` so the sweep doesn't need
# to know about trainer internals.
TrialRunnerFn = Callable[[int, Dict[str, Any]], Dict[str, Any]]


@dataclass
class TrialResult:
    """Outcome of one trial."""

    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, Any] = field(default_factory=dict)
    primary_metric_value: Optional[float] = None
    duration_seconds: float = 0.0
    failed: bool = False
    error: Optional[str] = None
    pruned: bool = False  # ASHA-style early-stopped (not implemented in v1)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SweepResult:
    """Aggregate over all trials."""

    sweep_name: str
    config: Dict[str, Any]
    trials: List[TrialResult]
    best_trial_id: Optional[int]
    best_value: Optional[float]
    n_completed: int
    n_failed: int
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["trials"] = [t.to_dict() for t in self.trials]
        return d


# ---------- samplers --------------------------------------------------------


def _build_sampler(
    name: str,
    rng: random.Random,
    direction: str = "minimize",
):
    """Return a sampler with `next_params(search_space) -> dict`.

    Random is the always-available baseline. TPE (Optuna-backed) is the
    upgrade when the user has Optuna installed and wants smarter
    suggestions; falls back to random with a warning when not.
    """
    name = (name or "random").lower()
    if name == "random":
        return _RandomSampler(rng)
    if name == "tpe":
        try:
            return _OptunaSampler(rng, direction=direction)
        except ImportError as exc:
            logger.warning(
                "Optuna not installed; falling back to random sampling. (%s)",
                exc,
            )
            return _RandomSampler(rng)
    if name == "grid":
        return _GridSampler(rng)
    raise ValueError(f"Unknown sampler {name!r}; choose random / tpe / grid")


class _RandomSampler:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def next_params(self, space, *, trial_id: int) -> Dict[str, Any]:
        return space.sample(self.rng)

    def report(self, trial_id: int, params: Dict[str, Any], value: Optional[float]):
        pass  # random doesn't learn from history


class _GridSampler:
    """Brute-force grid over `Choice` distributions; falls back to
    random for `Uniform` / `LogUniform` (continuous spaces don't grid)."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._idx = 0

    def _build_grid(self, space) -> List[Dict[str, Any]]:
        # Cartesian product over the discrete choices; for continuous
        # distributions we sample once at build time.
        from halo_forge.sweep.config import Choice

        keys = sorted(space.params)
        per_key_values: List[List[Any]] = []
        for k in keys:
            dist = space.params[k]
            if isinstance(dist, Choice):
                per_key_values.append(list(dist.values))
            else:
                # Continuous: sample 4 representative points.
                per_key_values.append([dist.sample(self.rng) for _ in range(4)])

        grid: List[Dict[str, Any]] = [{}]
        for key, values in zip(keys, per_key_values):
            grid = [{**row, key: v} for row in grid for v in values]
        return grid

    def next_params(self, space, *, trial_id: int) -> Dict[str, Any]:
        if self._cache is None:
            self._cache = self._build_grid(space)
        if self._idx >= len(self._cache):
            # Exhausted grid — fall through to random for over-budget trials.
            return space.sample(self.rng)
        params = self._cache[self._idx]
        self._idx += 1
        return params

    def report(self, trial_id: int, params: Dict[str, Any], value: Optional[float]):
        pass


class _OptunaSampler:
    """Lazy-imported TPE sampler. Each trial pulls from the same study
    so suggestions are informed by the in-progress history."""

    def __init__(self, rng: random.Random, *, direction: str = "minimize"):
        import optuna  # noqa: F401  (forces ImportError early)

        self._optuna = optuna
        normalized_direction = str(direction).strip().lower()
        if normalized_direction not in {"minimize", "maximize"}:
            raise ValueError("direction must be 'minimize' or 'maximize'")
        # Build the study with a fixed seed so the sweep is reproducible
        # under the same SweepConfig.seed.
        sampler = optuna.samplers.TPESampler(seed=rng.randint(0, 2**31 - 1))
        self.study = optuna.create_study(direction=normalized_direction, sampler=sampler)
        self._trial_handles: Dict[int, Any] = {}
        self._fallback_rng = rng

    def next_params(self, space, *, trial_id: int) -> Dict[str, Any]:
        from halo_forge.sweep.config import Choice, LogUniform, Uniform

        trial = self.study.ask()
        params: Dict[str, Any] = {}
        for name in sorted(space.params):
            dist = space.params[name]
            if isinstance(dist, Choice):
                params[name] = trial.suggest_categorical(name, dist.values)
            elif isinstance(dist, LogUniform):
                params[name] = trial.suggest_float(name, dist.low, dist.high, log=True)
            elif isinstance(dist, Uniform):
                params[name] = trial.suggest_float(name, dist.low, dist.high)
            else:
                params[name] = dist.sample(self._fallback_rng)  # fallback
        self._trial_handles[trial_id] = trial
        return params

    def report(self, trial_id: int, params: Dict[str, Any], value: Optional[float]):
        trial = self._trial_handles.pop(trial_id, None)
        if trial is None:
            return
        if value is None:
            self.study.tell(trial, state=self._optuna.trial.TrialState.FAIL)
        else:
            self.study.tell(trial, value)


# ---------- runner ---------------------------------------------------------


def run_sweep(
    *,
    config: SweepConfig,
    runner: TrialRunnerFn,
) -> SweepResult:
    """Run `config.n_trials` trials, each with `runner` doing the work.

    Args:
        config: SweepConfig with search space + budget + sampler choice.
        runner: ``(trial_id, params) -> metrics``. Halo-forge's caller
            (CLI / programmatic) builds a runner that invokes the
            chosen trainer with the sampled params and returns the
            relevant metrics dict at the end.

    Returns:
        SweepResult with per-trial outcomes and the best-so-far pointer.
    """
    rng = random.Random(config.seed)
    sampler = _build_sampler(config.sampler, rng, direction=config.direction)

    output_dir = Path(config.output_dir) if config.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    trials_jsonl = (output_dir / "trials.jsonl") if output_dir else None
    jsonl_handle = trials_jsonl.open("w") if trials_jsonl else None

    trials: List[TrialResult] = []
    best_value: Optional[float] = None
    best_id: Optional[int] = None
    no_improvement_streak = 0
    t0 = time.time()

    try:
        for trial_id in range(config.n_trials):
            params = sampler.next_params(config.search_space, trial_id=trial_id)
            trial_t0 = time.time()
            try:
                metrics = runner(trial_id, params)
                primary = metrics.get(config.metric)
                primary_val = float(primary) if isinstance(primary, (int, float)) else None
                trial = TrialResult(
                    trial_id=trial_id,
                    params=params,
                    metrics=dict(metrics),
                    primary_metric_value=primary_val,
                    duration_seconds=time.time() - trial_t0,
                )
            except Exception as exc:
                logger.warning("Trial %d failed: %s", trial_id, exc)
                trial = TrialResult(
                    trial_id=trial_id,
                    params=params,
                    duration_seconds=time.time() - trial_t0,
                    failed=True,
                    error=str(exc),
                )

            trials.append(trial)
            sampler.report(trial_id, params, trial.primary_metric_value)

            # Update best-so-far.
            if trial.primary_metric_value is not None and config.is_better(
                trial.primary_metric_value, best_value
            ):
                best_value = trial.primary_metric_value
                best_id = trial.trial_id
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1

            if jsonl_handle:
                jsonl_handle.write(json.dumps(trial.to_dict(), default=str) + "\n")
                jsonl_handle.flush()

            # Early-stop the whole sweep when no improvement for N trials.
            if (
                config.early_stop_after is not None
                and no_improvement_streak >= config.early_stop_after
            ):
                logger.info(
                    "Sweep early-stopped after %d trials with no improvement",
                    no_improvement_streak,
                )
                break
    finally:
        if jsonl_handle:
            jsonl_handle.close()

    n_failed = sum(1 for t in trials if t.failed)
    result = SweepResult(
        sweep_name=config.name,
        config=config.to_dict(),
        trials=trials,
        best_trial_id=best_id,
        best_value=best_value,
        n_completed=len(trials) - n_failed,
        n_failed=n_failed,
        duration_seconds=time.time() - t0,
    )

    if output_dir:
        (output_dir / "sweep_summary.json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str)
        )

    return result


__all__ = [
    "TrialResult",
    "SweepResult",
    "run_sweep",
]
