# Evaluation

`halo-forge eval` wraps EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) so a halo-forge-trained model can be benchmarked against the published academic suites with one command and one consistent result shape.

```bash
halo-forge eval --model ./models/sft/final_model --tasks core
halo-forge eval --model X --tasks reasoning,code --limit 100
halo-forge eval --backend vllm --tasks core    # CUDA/ROCm fast path
halo-forge eval --backend mlx  --tasks core    # Apple Silicon
halo-forge eval --list-tasks                   # show curated groups
```

## Curated task groups

Pass any of these as `--tasks <name>` to run the canonical benchmarks for that capability without learning the full lm-eval task index:

| Group | Tasks |
|---|---|
| `core` | mmlu, gsm8k, hellaswag, arc_challenge, winogrande, truthfulqa_mc2 |
| `reasoning` | gsm8k, math, arc_challenge, bbh |
| `code` | humaneval, mbpp |
| `instruction_following` | ifeval, mt_bench |
| `knowledge` | mmlu, mmlu_pro, agieval |

Or pass any lm-eval task name directly (`--tasks mmlu_pro_law,agieval_aqua_rat`).

Curated groups are deduplicated before the runner sees them, so passing both `core` and `mmlu` doesn't run MMLU twice.

## Backend choice

`--backend` selects which lm-eval model adapter to use:

- **`hf`** (default) — transformers + accelerate. Works on every halo-forge backend (rocm/cuda/mps/cpu/mlx-via-mps).
- **`vllm`** — continuous-batched inference. CUDA/ROCm only. 5-10× faster on long generations.
- **`mlx`** — Apple Silicon native via `mlx_lm`.

## Result shape

Halo-forge projects lm-eval's per-task metrics into a single `EvalResult` dataclass:

```python
{
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "tasks": ["mmlu", "gsm8k", "hellaswag", "arc_challenge", "winogrande", "truthfulqa_mc2"],
  "task_results": [
    {
      "task": "mmlu",
      "primary_metric": "acc",
      "value": 0.643,
      "n_samples": 14000,
      "all_metrics": {"acc": 0.643, "acc_stderr": 0.004, "acc_norm": 0.651},
      "error": null
    },
    ...
  ],
  "n_tasks_completed": 6,
  "n_tasks_failed": 0,
  "duration_seconds": 412.3,
  "backend": "vllm"
}
```

The **primary metric** for each task is hint-table-driven: `acc` for MMLU, `exact_match` for GSM8K, `pass@1` for HumanEval, etc. The full metric dict is preserved under `all_metrics` for callers that want secondary numbers (acc_norm, byte accuracy, etc.).

## Output directory

`--output <dir>` writes two files:

- `lm_eval_summary.json` — the projected `EvalResult` shape above.
- `lm_eval_raw.json` — lm-eval's full unprocessed output dict (when serializable).

The summary shape is intentionally close to halo-forge's `training_summary.json` shape so the upcoming F-K cohort eval dashboard can consume eval runs alongside training runs without a separate path.

## Smoke tests

Cap samples per task with `--limit` to validate a model in seconds rather than hours:

```bash
halo-forge eval --model ./final_model --tasks core --limit 50
```

50 samples per task is enough to catch broken-export failures (the kind I4 round-trip verify also catches) without paying the full eval cost.

## Programmatic use

```python
from halo_forge.eval import run_lm_eval

result = run_lm_eval(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    tasks=["mmlu", "gsm8k"],
    limit=100,
    backend="hf",
)
print(result.average_score())
for r in result.task_results:
    print(f"{r.task}: {r.primary_metric}={r.value:.4f}")
```

The `runner=` knob lets tests inject a stub instead of importing lm-eval — useful when validating result-projection logic without the heavy dep.

## Installation

lm-eval is **lazy-imported**. Halo-forge's eval module loads on installs without it; the failure path surfaces a one-line install hint instead of a stack trace. Install with:

```bash
pip install lm-eval
# or for the CUDA-accelerated `vllm` backend choice:
pip install lm-eval[vllm]
```

## Roadmap

- **F-K cohort eval dashboard** — UI for "run this eval suite across N adapters, see a sortable table + per-task drill-down". Depends on this module being in place.
- **V9 mid-training general-benchmark probe** — run a small held-out general benchmark every N cycles during training to catch catastrophic forgetting in real time.
- **V7 judge reliability harness** — measure judge agreement vs human labels; flag judges that disagree with themselves on re-runs (calibration for V2 LLM-as-judge).
