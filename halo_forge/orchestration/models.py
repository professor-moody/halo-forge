"""Pure, serializable models for repeated and swept training runs.

The orchestration service persists these values, but this module deliberately
does not know about SQLite, subprocesses, HTTP, or trainer implementations.
That keeps materialization deterministic and makes replay validation possible
without starting a runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from halo_forge.sweep.config import Choice, SearchSpace

GROUP_KINDS = frozenset({"repeat", "sweep"})
METRIC_DIRECTIONS = frozenset({"minimize", "maximize"})
SAMPLERS = frozenset({"random", "grid", "tpe"})
TERMINAL_SEED_STATUSES = frozenset({"completed", "failed", "cancelled", "pruned", "stopped"})
RUNTIME_CONFIG_KEYS = frozenset(
    {
        "canonical_run_id",
        "output",
        "output_dir",
        "output_root",
        "parent_run_id",
        "run_id",
        "seed",
    }
)


def _json_value(value: Any) -> Any:
    """Normalize supported values before canonical JSON encoding."""

    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fingerprinted values must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if text_key in normalized:
                raise ValueError(f"mapping keys collide after string conversion: {text_key!r}")
            normalized[text_key] = _json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_value(item) for item in value]
        return sorted(items, key=canonical_json)
    raise TypeError(f"value of type {type(value).__name__} is not canonically serializable")


def canonical_json(value: Any) -> str:
    """Encode a value in the stable form used by orchestration hashes."""

    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_fingerprint(value: Any) -> str:
    """Return a SHA-256 fingerprint for an arbitrary canonical payload."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def config_fingerprint(
    config: Mapping[str, Any],
    *,
    exclude_runtime: bool = True,
    excluded_keys: Iterable[str] = (),
) -> str:
    """Fingerprint a resolved training configuration.

    Runtime identity is excluded by default, so the same parameterization run
    under several seeds or output directories has one configuration identity.
    Dataset/artifact hashes, model revisions, and all other scientific inputs
    remain part of the fingerprint.
    """

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    ignored = {str(key) for key in excluded_keys}
    if exclude_runtime:
        ignored.update(RUNTIME_CONFIG_KEYS)
    scientific_config = {
        str(key): value for key, value in config.items() if str(key) not in ignored
    }
    return canonical_fingerprint(scientific_config)


@dataclass(frozen=True)
class SuccessiveHalvingConfig:
    """Configuration for opt-in synchronous successive halving."""

    enabled: bool = False
    reduction_factor: int = 3
    budgets: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        factor = int(self.reduction_factor)
        budgets = tuple(int(value) for value in self.budgets)
        if factor < 2:
            raise ValueError("successive-halving reduction_factor must be at least 2")
        if any(value <= 0 for value in budgets):
            raise ValueError("successive-halving budgets must be positive")
        if any(current >= following for current, following in zip(budgets, budgets[1:])):
            raise ValueError("successive-halving budgets must be strictly increasing")
        object.__setattr__(self, "reduction_factor", factor)
        object.__setattr__(self, "budgets", budgets)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "reduction_factor": self.reduction_factor,
            "budgets": list(self.budgets),
        }

    @classmethod
    def from_dict(cls, payload: Optional[Mapping[str, Any]]) -> "SuccessiveHalvingConfig":
        values = dict(payload or {})
        return cls(
            enabled=bool(values.get("enabled", False)),
            reduction_factor=int(values.get("reduction_factor", 3)),
            budgets=tuple(values.get("budgets") or ()),
        )


@dataclass(frozen=True)
class RunGroupSpec:
    """Resolved intent for a repeated-seed group or parameter sweep."""

    name: str
    kind: str
    base_config: Mapping[str, Any]
    search_space: SearchSpace = field(default_factory=SearchSpace)
    n_trials: int = 1
    metric: str = "final_train_loss"
    direction: str = "minimize"
    sampler: str = "random"
    base_seed: int = 42
    sampler_seed: Optional[int] = None
    seeds: Tuple[int, ...] = ()
    pruning: SuccessiveHalvingConfig = field(default_factory=SuccessiveHalvingConfig)
    checkpoint_policy_revision_id: Optional[str] = None
    resolved_checkpoint_plan: Optional[Mapping[str, Any]] = None
    version: int = 1

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        name = str(self.name).strip()
        direction = str(self.direction).strip().lower()
        sampler = str(self.sampler).strip().lower()
        metric = str(self.metric).strip()
        n_trials = int(self.n_trials)
        base_seed = int(self.base_seed)
        sampler_seed = base_seed if self.sampler_seed is None else int(self.sampler_seed)
        version = int(self.version)
        checkpoint_policy_revision_id = (
            str(self.checkpoint_policy_revision_id).strip()
            if self.checkpoint_policy_revision_id is not None
            else None
        )
        if checkpoint_policy_revision_id == "":
            checkpoint_policy_revision_id = None
        resolved_checkpoint_plan = self.resolved_checkpoint_plan
        if resolved_checkpoint_plan is not None and not isinstance(
            resolved_checkpoint_plan, Mapping
        ):
            raise TypeError("resolved_checkpoint_plan must be a mapping")
        normalized_checkpoint_plan = (
            deepcopy(dict(resolved_checkpoint_plan))
            if resolved_checkpoint_plan is not None
            else None
        )
        search_space = self.search_space
        if not isinstance(search_space, SearchSpace):
            search_space = SearchSpace.from_dict(search_space)
        pruning = self.pruning
        if not isinstance(pruning, SuccessiveHalvingConfig):
            pruning = SuccessiveHalvingConfig.from_dict(pruning)
        seeds = tuple(int(seed) for seed in self.seeds)

        if not name:
            raise ValueError("run-group name cannot be empty")
        if kind not in GROUP_KINDS:
            raise ValueError("run-group kind must be 'repeat' or 'sweep'")
        if direction not in METRIC_DIRECTIONS:
            raise ValueError("direction must be 'minimize' or 'maximize'")
        if sampler not in SAMPLERS:
            raise ValueError("sampler must be random, grid, or tpe")
        if not metric:
            raise ValueError("primary metric cannot be empty")
        if n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if version not in {1, 2}:
            raise ValueError(f"unsupported run-group spec version {version}")
        if version == 1 and (
            checkpoint_policy_revision_id is not None or normalized_checkpoint_plan is not None
        ):
            raise ValueError("checkpoint policies require run-group spec version 2")
        if not isinstance(self.base_config, Mapping):
            raise TypeError("base_config must be a mapping")
        if len(seeds) != len(set(seeds)):
            raise ValueError("run-group seeds must be unique")

        if kind == "repeat":
            if n_trials != 1:
                raise ValueError("repeat groups contain exactly one parameter trial")
            if search_space.params:
                raise ValueError("repeat groups cannot define a search space")
            if pruning.enabled:
                raise ValueError("repeat groups cannot enable pruning")
            if not seeds:
                seeds = (base_seed, base_seed + 1, base_seed + 2)
        elif not search_space.params:
            raise ValueError("sweep groups require a non-empty search space")
        elif not seeds:
            # Every parameter configuration uses the same default seed so the
            # comparison changes one scientific variable at a time.
            seeds = (base_seed,)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "base_config", deepcopy(dict(self.base_config)))
        object.__setattr__(self, "search_space", search_space)
        object.__setattr__(self, "n_trials", n_trials)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "base_seed", base_seed)
        object.__setattr__(self, "sampler_seed", sampler_seed)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "pruning", pruning)
        object.__setattr__(self, "checkpoint_policy_revision_id", checkpoint_policy_revision_id)
        object.__setattr__(self, "resolved_checkpoint_plan", normalized_checkpoint_plan)
        object.__setattr__(self, "version", version)

        # Fail at the boundary rather than much later during persistence.
        canonical_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "version": self.version,
            "name": self.name,
            "kind": self.kind,
            "base_config": deepcopy(dict(self.base_config)),
            "search_space": self.search_space.to_dict(),
            "n_trials": self.n_trials,
            "metric": self.metric,
            "direction": self.direction,
            "sampler": self.sampler,
            "base_seed": self.base_seed,
            "sampler_seed": self.sampler_seed,
            "seeds": list(self.seeds),
            "pruning": self.pruning.to_dict(),
        }
        # Version 1 is a frozen compatibility format. Keep its serialized
        # shape byte-for-byte stable and append adaptive fields only in v2.
        if self.version == 2:
            payload["checkpoint_policy_revision_id"] = self.checkpoint_policy_revision_id
            payload["resolved_checkpoint_plan"] = deepcopy(self.resolved_checkpoint_plan)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunGroupSpec":
        if not isinstance(payload, Mapping):
            raise TypeError("run-group spec must be a mapping")
        values = dict(payload)
        kind = str(values.get("kind") or "").strip().lower()
        default_trials = 1 if kind == "repeat" else 16
        inferred_version = (
            2
            if "version" not in values
            and ("checkpoint_policy_revision_id" in values or "resolved_checkpoint_plan" in values)
            else int(values.get("version", 1))
        )
        return cls(
            version=inferred_version,
            name=values.get("name", ""),
            kind=kind,
            base_config=values.get("base_config") or {},
            search_space=SearchSpace.from_dict(values.get("search_space") or {}),
            n_trials=int(values.get("n_trials", default_trials)),
            metric=values.get("metric", "final_train_loss"),
            direction=values.get("direction", "minimize"),
            sampler=values.get("sampler", "random"),
            base_seed=int(values.get("base_seed", 42)),
            sampler_seed=(
                None if values.get("sampler_seed") is None else int(values["sampler_seed"])
            ),
            seeds=tuple(values.get("seeds") or ()),
            pruning=SuccessiveHalvingConfig.from_dict(values.get("pruning")),
            checkpoint_policy_revision_id=values.get("checkpoint_policy_revision_id"),
            resolved_checkpoint_plan=values.get("resolved_checkpoint_plan"),
        )


@dataclass(frozen=True)
class MaterializedRun:
    """One seeded run belonging to a materialized parameter trial."""

    trial_key: str
    trial_index: int
    seed_index: int
    seed: int
    resolved_config: Mapping[str, Any]
    config_fingerprint: str
    run_fingerprint: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_key": self.trial_key,
            "trial_index": self.trial_index,
            "seed_index": self.seed_index,
            "seed": self.seed,
            "resolved_config": deepcopy(dict(self.resolved_config)),
            "config_fingerprint": self.config_fingerprint,
            "run_fingerprint": self.run_fingerprint,
        }


@dataclass(frozen=True)
class MaterializedTrial:
    """A sampled configuration and all of its explicitly seeded runs."""

    trial_key: str
    trial_index: int
    params: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    config_fingerprint: str
    seeds: Tuple[int, ...]

    def materialize_runs(self) -> Tuple[MaterializedRun, ...]:
        runs = []
        for seed_index, seed in enumerate(self.seeds):
            run_config = deepcopy(dict(self.resolved_config))
            run_config["seed"] = int(seed)
            runs.append(
                MaterializedRun(
                    trial_key=self.trial_key,
                    trial_index=self.trial_index,
                    seed_index=seed_index,
                    seed=int(seed),
                    resolved_config=run_config,
                    config_fingerprint=self.config_fingerprint,
                    run_fingerprint=canonical_fingerprint(
                        {
                            "trial_key": self.trial_key,
                            "seed": int(seed),
                            "config": run_config,
                        }
                    ),
                )
            )
        return tuple(runs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_key": self.trial_key,
            "trial_index": self.trial_index,
            "params": deepcopy(dict(self.params)),
            "resolved_config": deepcopy(dict(self.resolved_config)),
            "config_fingerprint": self.config_fingerprint,
            "seeds": list(self.seeds),
            "runs": [run.to_dict() for run in self.materialize_runs()],
        }


def _grid_params(spec: RunGroupSpec) -> Tuple[Mapping[str, Any], ...]:
    rng = random.Random(spec.sampler_seed)
    rows: list[Dict[str, Any]] = [{}]
    for name in sorted(spec.search_space.params):
        distribution = spec.search_space.params[name]
        if isinstance(distribution, Choice):
            values = list(distribution.values)
        else:
            values = [distribution.sample(rng) for _ in range(4)]
        rows = [{**row, name: deepcopy(value)} for row in rows for value in values]
    if not rows:
        return ({},)
    while len(rows) < spec.n_trials:
        rows.append(spec.search_space.sample(rng))
    return tuple(rows[: spec.n_trials])


def _validate_sampled_params(
    spec: RunGroupSpec,
    sampled_params: Sequence[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], ...]:
    if len(sampled_params) != spec.n_trials:
        raise ValueError(
            f"expected {spec.n_trials} sampled parameter sets, got {len(sampled_params)}"
        )
    expected = set(spec.search_space.params)
    normalized = []
    for index, params in enumerate(sampled_params):
        if not isinstance(params, Mapping):
            raise TypeError(f"sampled_params[{index}] must be a mapping")
        actual = {str(key) for key in params}
        if actual != expected:
            raise ValueError(
                f"sampled_params[{index}] keys must be {sorted(expected)}, got {sorted(actual)}"
            )
        normalized.append({str(key): deepcopy(value) for key, value in params.items()})
    return tuple(normalized)


def materialize_trials(
    spec: RunGroupSpec,
    *,
    sampled_params: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[MaterializedTrial, ...]:
    """Deterministically materialize configuration trials and seeded runs.

    TPE is adaptive: later suggestions depend on completed objective values.
    Callers therefore pass its persisted sequential suggestions through
    ``sampled_params``.  Random and grid groups can be recreated from the spec
    alone on any restart.
    """

    if not isinstance(spec, RunGroupSpec):
        raise TypeError("spec must be a RunGroupSpec")
    if spec.kind == "repeat":
        if sampled_params not in (None, (), []):
            raise ValueError("repeat groups do not accept sampled parameters")
        parameter_sets: Tuple[Mapping[str, Any], ...] = ({},)
    elif sampled_params is not None:
        parameter_sets = _validate_sampled_params(spec, sampled_params)
    elif spec.sampler == "tpe":
        raise ValueError(
            "TPE trials are adaptive; pass the persisted sequential suggestions as sampled_params"
        )
    elif spec.sampler == "grid":
        parameter_sets = _grid_params(spec)
    else:
        rng = random.Random(spec.sampler_seed)
        parameter_sets = tuple(spec.search_space.sample(rng) for _ in range(spec.n_trials))

    trials = []
    for trial_index, params in enumerate(parameter_sets):
        resolved_config = deepcopy(dict(spec.base_config))
        resolved_config.update(deepcopy(dict(params)))
        fingerprint = config_fingerprint(resolved_config)
        trial_key = f"trial-{trial_index:04d}-{fingerprint[:12]}"
        trials.append(
            MaterializedTrial(
                trial_key=trial_key,
                trial_index=trial_index,
                params=deepcopy(dict(params)),
                resolved_config=resolved_config,
                config_fingerprint=fingerprint,
                seeds=spec.seeds,
            )
        )
    return tuple(trials)


@dataclass(frozen=True)
class CohortObservation:
    """Terminal or completed outcome for one trial seed."""

    trial_key: str
    seed: int
    value: Optional[float] = None
    status: str = "completed"

    def __post_init__(self) -> None:
        trial_key = str(self.trial_key).strip()
        status = str(self.status).strip().lower()
        value = None if self.value is None else float(self.value)
        if not trial_key:
            raise ValueError("trial_key cannot be empty")
        if status not in TERMINAL_SEED_STATUSES:
            raise ValueError(f"unsupported cohort status {status!r}")
        if status == "completed" and value is None:
            raise ValueError("completed cohort observations require a metric value")
        if value is not None and not math.isfinite(value):
            raise ValueError("cohort metric values must be finite")
        object.__setattr__(self, "trial_key", trial_key)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "status", status)


@dataclass(frozen=True)
class CohortAggregate:
    """Direction-neutral aggregate for one parameter trial."""

    trial_key: str
    expected_seeds: Tuple[int, ...]
    observed_seeds: Tuple[int, ...]
    missing_seeds: Tuple[int, ...]
    completed_count: int
    failure_count: int
    coverage: float
    complete_seed_coverage: bool
    eligible: bool
    mean: Optional[float]
    standard_deviation: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def aggregate_cohort(
    observations: Sequence[CohortObservation],
    expected_seeds: Mapping[str, Sequence[int]],
    *,
    require_complete_seed_coverage: bool = True,
) -> Tuple[CohortAggregate, ...]:
    """Aggregate metrics per trial while making missing coverage explicit."""

    expected = {
        str(trial_key): tuple(int(seed) for seed in seeds)
        for trial_key, seeds in expected_seeds.items()
    }
    for trial_key, seeds in expected.items():
        if not trial_key:
            raise ValueError("expected trial keys cannot be empty")
        if not seeds:
            raise ValueError(f"trial {trial_key!r} must expect at least one seed")
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"trial {trial_key!r} has duplicate expected seeds")

    indexed: Dict[Tuple[str, int], CohortObservation] = {}
    for observation in observations:
        if not isinstance(observation, CohortObservation):
            observation = CohortObservation(**dict(observation))
        key = (observation.trial_key, observation.seed)
        if observation.trial_key not in expected:
            raise ValueError(f"observation references unknown trial {observation.trial_key!r}")
        if observation.seed not in expected[observation.trial_key]:
            raise ValueError(
                f"observation seed {observation.seed} is not expected for {observation.trial_key!r}"
            )
        if key in indexed:
            raise ValueError(f"duplicate observation for trial/seed {key!r}")
        indexed[key] = observation

    aggregates = []
    for trial_key in sorted(expected):
        seeds = expected[trial_key]
        rows = [indexed[(trial_key, seed)] for seed in seeds if (trial_key, seed) in indexed]
        values = [row.value for row in rows if row.status == "completed" and row.value is not None]
        observed = tuple(row.seed for row in rows)
        missing = tuple(seed for seed in seeds if (trial_key, seed) not in indexed)
        failure_count = sum(row.status != "completed" for row in rows)
        complete_coverage = not missing
        eligible = failure_count == 0 and bool(values)
        if require_complete_seed_coverage:
            eligible = eligible and complete_coverage and len(values) == len(seeds)
        mean = statistics.fmean(values) if values else None
        deviation = statistics.pstdev(values) if values else None
        aggregates.append(
            CohortAggregate(
                trial_key=trial_key,
                expected_seeds=seeds,
                observed_seeds=observed,
                missing_seeds=missing,
                completed_count=len(values),
                failure_count=failure_count,
                coverage=len(rows) / len(seeds),
                complete_seed_coverage=complete_coverage,
                eligible=eligible,
                mean=mean,
                standard_deviation=deviation,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
            )
        )
    return tuple(aggregates)


def rank_cohort(
    aggregates: Sequence[CohortAggregate],
    *,
    direction: str,
    eligible_only: bool = True,
) -> Tuple[CohortAggregate, ...]:
    """Return a deterministic, metric-direction-aware trial ranking."""

    normalized_direction = str(direction).strip().lower()
    if normalized_direction not in METRIC_DIRECTIONS:
        raise ValueError("direction must be 'minimize' or 'maximize'")
    rows = [row for row in aggregates if not eligible_only or row.eligible]

    def key(row: CohortAggregate) -> Tuple[int, float, str]:
        if row.mean is None:
            return (1, 0.0, row.trial_key)
        metric = row.mean if normalized_direction == "minimize" else -row.mean
        return (0, metric, row.trial_key)

    return tuple(sorted(rows, key=key))


def expected_seeds_for_trials(
    trials: Sequence[MaterializedTrial],
) -> Dict[str, Tuple[int, ...]]:
    return {trial.trial_key: trial.seeds for trial in trials}


__all__ = [
    "CohortAggregate",
    "CohortObservation",
    "MaterializedRun",
    "MaterializedTrial",
    "RunGroupSpec",
    "SuccessiveHalvingConfig",
    "aggregate_cohort",
    "canonical_fingerprint",
    "canonical_json",
    "config_fingerprint",
    "expected_seeds_for_trials",
    "materialize_trials",
    "rank_cohort",
]
