"""lm-evaluation-harness wrapper (Track V8)."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# Curated task groups — sensible defaults for "report your numbers"
# without making users learn the full lm-eval task taxonomy.
CURATED_TASK_GROUPS: Dict[str, List[str]] = {
    "core": ["mmlu", "gsm8k", "hellaswag", "arc_challenge", "winogrande", "truthfulqa_mc2"],
    "reasoning": ["gsm8k", "math", "arc_challenge", "bbh"],
    "code": ["humaneval", "mbpp"],
    "instruction_following": ["ifeval", "mt_bench"],
    "knowledge": ["mmlu", "mmlu_pro", "agieval"],
}


def list_curated_task_groups() -> Dict[str, List[str]]:
    return {k: list(v) for k, v in CURATED_TASK_GROUPS.items()}


# ---------- result shape ---------------------------------------------------


@dataclass
class EvalTaskResult:
    """One task's outcome — name, primary metric, value, plus the full
    metric dict for callers that want secondary numbers (e.g. byte
    accuracy alongside acc_norm)."""

    task: str
    primary_metric: str
    value: float
    n_samples: Optional[int] = None
    higher_is_better: bool = True
    all_metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    """Aggregate of an `lm-eval` run across multiple tasks. Shaped to
    match halo-forge's training_summary contract loosely so the F-K
    cohort eval dashboard can consume it without a separate path."""

    model_name: str
    tasks: List[str]
    task_results: List[EvalTaskResult]
    n_tasks_completed: int
    n_tasks_failed: int
    duration_seconds: float
    backend: str
    output_dir: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["task_results"] = [r.to_dict() for r in self.task_results]
        return d

    def average_score(self) -> Optional[float]:
        """Mean of every task's primary metric value. None if no
        successful tasks."""
        successful = [
            r.value for r in self.task_results
            if r.error is None and isinstance(r.value, (int, float))
        ]
        if not successful:
            return None
        return sum(successful) / len(successful)


# ---------- task list resolution -------------------------------------------


def _resolve_tasks(tasks: Sequence[str]) -> List[str]:
    """Expand any curated group names ("core", "reasoning", ...) into
    their member tasks; pass through anything we don't recognize so
    upstream-lm-eval's own task names continue to work."""
    resolved: List[str] = []
    seen: set[str] = set()
    for t in tasks:
        if t in CURATED_TASK_GROUPS:
            for member in CURATED_TASK_GROUPS[t]:
                if member not in seen:
                    resolved.append(member)
                    seen.add(member)
        elif t not in seen:
            resolved.append(t)
            seen.add(t)
    return resolved


# ---------- result extraction from lm-eval's output -----------------------


# lm-eval's task → primary-metric heuristic. The harness exposes many
# metrics per task; we pick the one papers report in their leaderboard
# tables. Falls back to the first numeric metric for unknown tasks.
_PRIMARY_METRIC_HINTS: Dict[str, str] = {
    "mmlu": "acc",
    "mmlu_pro": "acc",
    "gsm8k": "exact_match",
    "math": "exact_match",
    "hellaswag": "acc_norm",
    "arc_challenge": "acc_norm",
    "arc_easy": "acc_norm",
    "winogrande": "acc",
    "truthfulqa_mc2": "acc",
    "ifeval": "prompt_level_loose_acc",
    "humaneval": "pass@1",
    "mbpp": "pass@1",
    "agieval": "acc",
    "bbh": "exact_match",
}


def _pick_primary_metric(task: str, metrics: Dict[str, Any]) -> tuple[str, float]:
    """Choose which metric to use as the headline for ``task``."""
    hint = _PRIMARY_METRIC_HINTS.get(task)
    if hint and hint in metrics:
        try:
            return hint, float(metrics[hint])
        except (TypeError, ValueError):
            pass

    # Fall back to the first numeric metric that's not stderr.
    for key, val in metrics.items():
        if "stderr" in key:
            continue
        try:
            return key, float(val)
        except (TypeError, ValueError):
            continue
    return ("(none)", 0.0)


def _project_lm_eval_results(
    raw: Dict[str, Any],
    task_list: Sequence[str],
) -> List[EvalTaskResult]:
    """Turn lm-eval's results dict into our `EvalTaskResult` list.

    lm-eval exposes the per-task metrics under ``raw["results"][task_name]``;
    we extract the primary metric, the sample count, and the full
    metric dict for each requested task.
    """
    results_table = raw.get("results", {}) if isinstance(raw, dict) else {}
    n_samples_table = raw.get("n-samples", {}) if isinstance(raw, dict) else {}

    out: List[EvalTaskResult] = []
    for task in task_list:
        metrics = results_table.get(task)
        if not metrics:
            out.append(EvalTaskResult(
                task=task, primary_metric="(missing)", value=0.0,
                error="task_not_in_results",
            ))
            continue
        # Drop the "alias" / "version" rows lm-eval adds — they're not metrics.
        cleaned: Dict[str, float] = {}
        for k, v in metrics.items():
            if k in ("alias", "version"):
                continue
            try:
                cleaned[k] = float(v)
            except (TypeError, ValueError):
                continue
        primary_key, primary_val = _pick_primary_metric(task, cleaned)
        n_samples = None
        if isinstance(n_samples_table.get(task), dict):
            n_samples = n_samples_table[task].get("effective")
        out.append(EvalTaskResult(
            task=task,
            primary_metric=primary_key,
            value=primary_val,
            n_samples=int(n_samples) if isinstance(n_samples, int) else None,
            all_metrics=cleaned,
        ))
    return out


# ---------- main entry point -----------------------------------------------


def run_lm_eval(
    *,
    model_name: str,
    tasks: Sequence[str],
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
    backend: str = "hf",
    output_dir: Optional[Path] = None,
    runner=None,
) -> EvalResult:
    """Run `lm-eval` on `tasks` and return halo-forge-shaped results.

    Args:
        model_name: HF id, mlx-community id, or local path.
        tasks: List of task names. Curated group names ("core",
            "reasoning", ...) expand to their members.
        limit: Cap samples per task (useful for smoke tests; None
            for full eval).
        batch_size: Per-step batch size. None lets lm-eval pick.
        backend: "hf" (transformers + accelerate; works on every
            backend halo-forge supports), "vllm" (CUDA/ROCm only;
            faster), or "mlx" (Apple Silicon).
        output_dir: Optional dir to write the raw + projected results
            JSON. None skips the dump.
        runner: Optional `(model, tasks, **kwargs) -> dict` callable
            used by tests to inject stub results without importing
            lm-eval. Production calls this through `lm_eval.simple_evaluate`.
    """
    expanded = _resolve_tasks(tasks)
    if not expanded:
        raise ValueError("run_lm_eval needs at least one task")

    t0 = time.time()
    if runner is None:
        try:
            from lm_eval import simple_evaluate
        except ImportError as exc:
            raise ImportError(
                "lm-evaluation-harness is not installed. Install with "
                "`pip install lm-eval`."
            ) from exc

        kwargs: Dict[str, Any] = {
            "model": backend,  # lm-eval's `model` arg is "hf", "vllm", "mlx", ...
            "model_args": f"pretrained={model_name}",
            "tasks": list(expanded),
        }
        if limit is not None:
            kwargs["limit"] = int(limit)
        if batch_size is not None:
            kwargs["batch_size"] = int(batch_size)
        raw = simple_evaluate(**kwargs)
    else:
        raw = runner(model=model_name, tasks=list(expanded), limit=limit)

    task_results = _project_lm_eval_results(raw or {}, expanded)
    completed = sum(1 for r in task_results if r.error is None)
    failed = sum(1 for r in task_results if r.error is not None)

    result = EvalResult(
        model_name=model_name,
        tasks=list(expanded),
        task_results=task_results,
        n_tasks_completed=completed,
        n_tasks_failed=failed,
        duration_seconds=time.time() - t0,
        backend=backend,
        output_dir=str(output_dir) if output_dir else None,
    )

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "lm_eval_summary.json").write_text(
            json.dumps(result.to_dict(), indent=2, default=str)
        )
        if raw:
            try:
                (out / "lm_eval_raw.json").write_text(
                    json.dumps(raw, indent=2, default=str)
                )
            except (TypeError, ValueError):
                # lm-eval sometimes returns un-serializable bits; best-effort.
                pass

    return result


__all__ = [
    "CURATED_TASK_GROUPS",
    "EvalTaskResult",
    "EvalResult",
    "list_curated_task_groups",
    "run_lm_eval",
]
