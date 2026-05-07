---
title: "Hyperparameter sweeps"
description: "Optuna-style search over learning rate, batch, LoRA rank, etc., with random / TPE / grid samplers and ASHA-style sweep-level early stop."
weight: 60
---

The sweep library composes three pieces:

1. **Search space** — per-knob distributions (`Uniform`, `LogUniform`, `Choice`).
2. **Sampler** — `random` / `tpe` (Optuna-backed) / `grid`. TPE falls back to random with a warning when Optuna isn't installed.
3. **Pruner** — sweep-level early stop after N trials with no improvement.

## Search space

```python
from halo_forge.sweep import SearchSpace, LogUniform, Choice, Uniform

space = SearchSpace(params={
    "learning_rate": LogUniform(1e-6, 1e-3),
    "batch_size":    Choice([1, 2, 4]),
    "lora_rank":     Choice([8, 16, 32, 64]),
    "warmup_ratio":  Uniform(0.0, 0.2),
})
```

Distributions:

| | What it does | Right shape for |
|---|---|---|
| `Uniform(low, high)` | Continuous flat | rates, fractions |
| `LogUniform(low, high)` | Log-uniform | learning rates (1e-5 vs 1e-4 is the interesting comparison, not 1e-5 vs 5e-5) |
| `Choice([v1, ...])` | Discrete | batch sizes, ranks, dtype names |

## Running a sweep

```python
from halo_forge.sweep import SweepConfig, run_sweep

cfg = SweepConfig(
    name="dpo_lr_search",
    search_space=space,
    n_trials=16,
    metric="final_train_loss",
    direction="minimize",
    sampler="random",  # or "tpe", "grid"
    seed=42,
    early_stop_after=5,            # halt sweep after 5 trials with no improvement
    output_dir="sweeps/dpo_lr_search",
)

def runner(trial_id, params):
    # Build a trainer with `params` overlaid on a base config and run it;
    # return the metrics dict the sweep watches.
    summary = launch_dpo_with_overrides(params)
    return {"final_train_loss": summary["final_train_loss"]}

result = run_sweep(config=cfg, runner=runner)
print("best:", result.best_trial_id, result.best_value)
```

The runner is callable-driven: any `(trial_id, params) -> metrics` works. That decouples the sweep machinery from the trainer surface — when a new trainer ships, sweeping it is "wire a runner".

## Output

```
sweeps/dpo_lr_search/
  ├── trials.jsonl         # one line per trial; streamed live so a dashboard can tail it
  └── sweep_summary.json   # full result + best-trial pointer
```

`trials.jsonl` is appended on every trial completion, so a dashboard (F-P) can render in-progress sweeps without waiting for the budget to exhaust.

## Samplers

| Sampler | Backed by | Best for |
|---|---|---|
| `random` | stdlib `random.Random(seed)` | always available; the baseline |
| `tpe` | Optuna's `TPESampler` | smarter suggestions when 16+ trials in budget |
| `grid` | Cartesian product over `Choice` (continuous distributions sampled once) | small discrete spaces |

`pip install optuna` to enable `tpe`. Without it, `tpe` logs a warning and falls back to random.

## Early stopping

Two layers:

- **Sweep-level** (`early_stop_after`) — halt the whole sweep when N consecutive trials don't improve the best.
- **Per-trial pruning** — roadmap. Will integrate ASHA / Hyperband once the trainer-side intermediate-metric reporting lands.

## CLI integration

The library is the foundation; a `halo-forge sweep` CLI that orchestrates SFT/DPO/GRPO trials with sampled config overlays is a follow-up — it requires per-trainer config-overlay machinery beyond the v1 surface. Use the programmatic API today.
