"""Deterministic orchestration policies with no persistence side effects."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from halo_forge.orchestration.models import (
    CohortAggregate,
    METRIC_DIRECTIONS,
    SuccessiveHalvingConfig,
)


@dataclass(frozen=True)
class HalvingDecision:
    """One synchronous rung decision.

    ``ready=False`` means the caller must not mutate trial state.  Once ready,
    promotion and pruning are exhaustive over the active trial set unless this
    is the final configured budget, where every already-completed trial remains
    terminal and the direction-aware ranking is the selection result.
    """

    ready: bool
    reason: str
    rung_index: int
    budget: Optional[int]
    next_budget: Optional[int]
    reduction_factor: int
    active_trial_keys: Tuple[str, ...]
    ranking: Tuple[str, ...]
    promoted_trial_keys: Tuple[str, ...]
    pruned_trial_keys: Tuple[str, ...]
    waiting_trial_keys: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _rank_for_halving(
    rows: Sequence[CohortAggregate],
    *,
    direction: str,
) -> Tuple[CohortAggregate, ...]:
    """Rank successes first and terminal failures deterministically last."""

    def key(row: CohortAggregate) -> Tuple[int, float, str]:
        if not row.eligible or row.mean is None:
            return (1, 0.0, row.trial_key)
        metric = row.mean if direction == "minimize" else -row.mean
        return (0, metric, row.trial_key)

    return tuple(sorted(rows, key=key))


def decide_successive_halving(
    config: SuccessiveHalvingConfig,
    aggregates: Sequence[CohortAggregate],
    *,
    direction: str,
    rung_index: int,
    active_trial_keys: Optional[Sequence[str]] = None,
) -> HalvingDecision:
    """Apply one opt-in synchronous successive-halving decision.

    No trial is compared until every active trial has a terminal observation
    for every configured seed.  This prevents a fast single seed from pruning a
    slower repeated-seed configuration.  Successful complete cohorts rank by
    mean; cohorts containing a failed/cancelled/pruned seed rank after them.
    Ties are resolved by stable trial key.
    """

    if not isinstance(config, SuccessiveHalvingConfig):
        config = SuccessiveHalvingConfig.from_dict(config)
    normalized_direction = str(direction).strip().lower()
    if normalized_direction not in METRIC_DIRECTIONS:
        raise ValueError("direction must be 'minimize' or 'maximize'")
    rung_index = int(rung_index)
    if rung_index < 0:
        raise ValueError("rung_index cannot be negative")
    if config.budgets and rung_index >= len(config.budgets):
        raise ValueError("rung_index exceeds configured successive-halving budgets")

    by_key: Dict[str, CohortAggregate] = {}
    for row in aggregates:
        if row.trial_key in by_key:
            raise ValueError(f"duplicate cohort aggregate for {row.trial_key!r}")
        by_key[row.trial_key] = row
    if active_trial_keys is None:
        active = tuple(sorted(by_key))
    else:
        active = tuple(sorted(str(key) for key in active_trial_keys))
        if len(active) != len(set(active)):
            raise ValueError("active_trial_keys cannot contain duplicates")
    if not active:
        raise ValueError("successive halving requires at least one active trial")

    budget = config.budgets[rung_index] if config.budgets else None
    next_budget = (
        config.budgets[rung_index + 1]
        if config.budgets and rung_index + 1 < len(config.budgets)
        else None
    )
    if not config.enabled:
        return HalvingDecision(
            ready=False,
            reason="disabled",
            rung_index=rung_index,
            budget=budget,
            next_budget=next_budget,
            reduction_factor=config.reduction_factor,
            active_trial_keys=active,
            ranking=(),
            promoted_trial_keys=(),
            pruned_trial_keys=(),
        )

    waiting = tuple(
        key for key in active if key not in by_key or not by_key[key].complete_seed_coverage
    )
    if waiting:
        return HalvingDecision(
            ready=False,
            reason="waiting_for_complete_seed_coverage",
            rung_index=rung_index,
            budget=budget,
            next_budget=next_budget,
            reduction_factor=config.reduction_factor,
            active_trial_keys=active,
            ranking=(),
            promoted_trial_keys=(),
            pruned_trial_keys=(),
            waiting_trial_keys=waiting,
        )

    ranked_rows = _rank_for_halving([by_key[key] for key in active], direction=normalized_direction)
    ranking = tuple(row.trial_key for row in ranked_rows)
    if config.budgets and next_budget is None:
        return HalvingDecision(
            ready=True,
            reason="final_budget_complete",
            rung_index=rung_index,
            budget=budget,
            next_budget=None,
            reduction_factor=config.reduction_factor,
            active_trial_keys=active,
            ranking=ranking,
            promoted_trial_keys=ranking,
            pruned_trial_keys=(),
        )

    survivor_count = max(1, math.ceil(len(active) / config.reduction_factor))
    promoted = ranking[:survivor_count]
    pruned = ranking[survivor_count:]
    return HalvingDecision(
        ready=True,
        reason="rung_complete",
        rung_index=rung_index,
        budget=budget,
        next_budget=next_budget,
        reduction_factor=config.reduction_factor,
        active_trial_keys=active,
        ranking=ranking,
        promoted_trial_keys=promoted,
        pruned_trial_keys=pruned,
    )


__all__ = ["HalvingDecision", "decide_successive_halving"]
