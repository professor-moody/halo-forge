"""Mid-training general-benchmark probe (Track V9).

The single biggest quality safeguard for finetuning: run a small held-out
general benchmark every N cycles, compare to the baseline, and flag
catastrophic forgetting *during* training instead of discovering it
weeks later when someone runs MMLU on the shipped model and sees a
regression.

Built on V8 lm-eval-harness (so the probe and the full eval share one
implementation, one task vocabulary, one result projection). Differs
from V8 in three ways:

  - **Small task subset by default**. The DEFAULT_PROBE_TASKS is sized
    so the probe completes in single-digit minutes; the full V8 eval
    can take hours.
  - **Baseline tracking**. The probe stores its first run as the
    baseline (or accepts a user-supplied baseline JSON) and reports
    deltas on every subsequent call.
  - **Regression detection**. Any task that drops more than the
    configured tolerance vs baseline is flagged as a regression
    with an explicit code so trainers can branch on the result.

v1 ships the probe as a standalone library + CLI. Direct integration
with the trainer cycle loops (so SFT / RAFT / DPO / GRPO automatically
fire it every N cycles) lands as a follow-up — the machinery here is
what that integration consumes.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# Small, fast probe set — single-digit minutes on a 3B model with limit=100.
# Picks one task per capability so a regression on any axis surfaces.
DEFAULT_PROBE_TASKS: tuple[str, ...] = (
    "mmlu",          # knowledge
    "arc_challenge", # commonsense reasoning
    "gsm8k",         # math
    "hellaswag",     # commonsense
)
DEFAULT_PROBE_LIMIT = 100  # samples per task


@dataclass
class ProbeTaskDelta:
    """One task's probe result + delta from baseline."""

    task: str
    primary_metric: str
    value: float
    baseline_value: Optional[float] = None
    delta: Optional[float] = None
    regression: bool = False  # True when delta < -tolerance
    n_samples: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProbeReport:
    """Outcome of one probe call."""

    model_name: str
    cycle: Optional[int]
    tasks: List[str]
    task_deltas: List[ProbeTaskDelta]
    has_baseline: bool
    has_regression: bool
    avg_delta: Optional[float]
    duration_seconds: float
    timestamp: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["task_deltas"] = [t.to_dict() for t in self.task_deltas]
        return d

    def regressed_tasks(self) -> List[str]:
        return [t.task for t in self.task_deltas if t.regression]


# ---------- baseline IO ----------------------------------------------------


def load_baseline(path: Path) -> Optional[Dict[str, float]]:
    """Read a baseline JSON written by `save_baseline`. Returns None if
    the file doesn't exist (first probe run)."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        logger.warning("Probe baseline at %s is unreadable: %s", path, exc)
        return None
    return {k: float(v) for k, v in data.items()}


def save_baseline(path: Path, task_values: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task_values, indent=2, sort_keys=True))


def values_from_eval_result(result: Any) -> Dict[str, float]:
    """Extract `{task: primary_value}` from an `EvalResult`. Accepts
    either the dataclass or its `.to_dict()` shape."""
    items: Dict[str, float] = {}
    task_results = getattr(result, "task_results", None)
    if task_results is None and isinstance(result, dict):
        task_results = result.get("task_results", [])
    for tr in task_results or []:
        if isinstance(tr, dict):
            if tr.get("error") is None:
                items[str(tr["task"])] = float(tr["value"])
        else:
            if tr.error is None:
                items[tr.task] = float(tr.value)
    return items


# ---------- the probe ------------------------------------------------------


class MidTrainingProbe:
    """Schedule + execute mid-training benchmark probes.

    Args:
        model_name: Model under test. For LoRA training, point at the
            base + adapter path or the post-bake checkpoint.
        baseline_path: Where to load / store the per-task baseline.
            None disables persistence (probe runs fire-and-forget).
        tasks: Override the default probe task set.
        limit: Samples per task. None uses lm-eval's full eval (slow).
        every_n_cycles: Fire on cycles `c` where ``c % every_n_cycles == 0``.
        regression_tolerance: Delta below this fraction triggers a
            regression flag. 0.05 = 5% absolute drop is loud-warn.
        backend: lm-eval model adapter ("hf" / "vllm" / "mlx").
        runner: Optional injected runner for tests (signature mirrors
            `halo_forge.eval.run_lm_eval`'s `runner=` knob).
    """

    def __init__(
        self,
        *,
        model_name: str,
        baseline_path: Optional[Path] = None,
        tasks: Optional[Sequence[str]] = None,
        limit: Optional[int] = DEFAULT_PROBE_LIMIT,
        every_n_cycles: int = 5,
        regression_tolerance: float = 0.05,
        backend: str = "hf",
        runner: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        if every_n_cycles < 1:
            raise ValueError("every_n_cycles must be >= 1")
        if regression_tolerance < 0:
            raise ValueError("regression_tolerance must be >= 0")
        self.model_name = model_name
        self.baseline_path = (
            Path(baseline_path) if baseline_path is not None else None
        )
        self.tasks = list(tasks or DEFAULT_PROBE_TASKS)
        self.limit = limit
        self.every_n_cycles = every_n_cycles
        self.regression_tolerance = regression_tolerance
        self.backend = backend
        self._runner = runner

    def should_run(self, cycle: int) -> bool:
        """Trainer cycle hook — return True when the probe should fire."""
        return cycle >= 0 and cycle % self.every_n_cycles == 0

    def run(self, *, cycle: Optional[int] = None, notes: Optional[str] = None) -> ProbeReport:
        """Execute one probe pass and return the report."""
        from halo_forge.eval import run_lm_eval

        t0 = time.time()
        result = run_lm_eval(
            model_name=self.model_name,
            tasks=self.tasks,
            limit=self.limit,
            backend=self.backend,
            runner=self._runner,
        )
        current = values_from_eval_result(result)

        baseline = (
            load_baseline(self.baseline_path) if self.baseline_path else None
        )
        deltas: List[ProbeTaskDelta] = []
        sum_delta = 0.0
        n_with_delta = 0
        any_regression = False

        for tr in result.task_results:
            baseline_val = baseline.get(tr.task) if baseline else None
            delta = (
                (tr.value - baseline_val) if baseline_val is not None else None
            )
            regression = (
                delta is not None and delta < -self.regression_tolerance
            )
            if regression:
                any_regression = True
            if delta is not None:
                sum_delta += delta
                n_with_delta += 1
            deltas.append(ProbeTaskDelta(
                task=tr.task,
                primary_metric=tr.primary_metric,
                value=tr.value,
                baseline_value=baseline_val,
                delta=delta,
                regression=regression,
                n_samples=tr.n_samples,
                error=tr.error,
            ))

        avg_delta = (sum_delta / n_with_delta) if n_with_delta else None
        has_baseline = baseline is not None

        # First run with no baseline: persist current as the baseline so
        # subsequent calls have something to compare against.
        if self.baseline_path is not None and baseline is None:
            save_baseline(self.baseline_path, current)
            logger.info(
                "Probe wrote first-run baseline to %s with %d tasks",
                self.baseline_path, len(current),
            )

        return ProbeReport(
            model_name=self.model_name,
            cycle=cycle,
            tasks=self.tasks,
            task_deltas=deltas,
            has_baseline=has_baseline,
            has_regression=any_regression,
            avg_delta=avg_delta,
            duration_seconds=time.time() - t0,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z") or "",
            notes=notes,
        )


__all__ = [
    "DEFAULT_PROBE_TASKS",
    "DEFAULT_PROBE_LIMIT",
    "ProbeTaskDelta",
    "ProbeReport",
    "MidTrainingProbe",
    "load_baseline",
    "save_baseline",
    "values_from_eval_result",
]
