"""Standardized evaluation (Track V8).

Wraps EleutherAI's `lm-evaluation-harness` so a halo-forge-trained
model can be benchmarked against the academic-standard task suites
(MMLU, GSM8K, HumanEval, IFEval, ARC, HellaSwag, BBH, …) with one
command and one consistent result shape.

Why integrate vs. roll our own:
  - lm-eval-harness is the published reference for every academic
    paper. If we want comparable numbers, we need their tokenization,
    their few-shot setup, their answer-extraction logic.
  - The task catalogue is enormous (>200 tasks) and growing; we'd
    never keep up.
  - Their HF-model adapter handles both vLLM and HF generate; that
    lines up exactly with halo-forge's I6 rollout dispatch.

What this module owns:
  - A thin `run_lm_eval` wrapper that handles I/O + result shaping.
  - Result projection into a halo-forge-flavored summary so the
    F-K cohort eval dashboard can consume eval runs the same way it
    consumes training runs.
  - A list of curated task aliases (the canonical benchmarks every
    finetune should report) so users don't need to read the
    lm-eval task index to get started.
"""

from halo_forge.eval.lm_harness import (
    EvalResult,
    EvalTaskResult,
    list_curated_task_groups,
    run_lm_eval,
)
from halo_forge.eval.probe import (
    DEFAULT_PROBE_LIMIT,
    DEFAULT_PROBE_TASKS,
    MidTrainingProbe,
    ProbeReport,
    ProbeTaskDelta,
    load_baseline,
    save_baseline,
    values_from_eval_result,
)

__all__ = [
    "DEFAULT_PROBE_LIMIT",
    "DEFAULT_PROBE_TASKS",
    "EvalResult",
    "EvalTaskResult",
    "MidTrainingProbe",
    "ProbeReport",
    "ProbeTaskDelta",
    "list_curated_task_groups",
    "load_baseline",
    "run_lm_eval",
    "save_baseline",
    "values_from_eval_result",
]
