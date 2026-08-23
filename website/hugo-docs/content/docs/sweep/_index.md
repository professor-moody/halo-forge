---
title: "Repeats and hyperparameter sweeps"
description: "Run deterministic seed repeats and parameter searches through Halo Forge's durable workstation queue."
weight: 60
---

Halo Forge groups related training into two reproducible operations:

- A **repeat** runs one resolved configuration with several explicit seeds.
- A **sweep** materializes several parameter configurations and runs the same
  configured seed cohort for each one.

Both use a pinned development benchmark revision and the durable workstation
queue. The group records its dataset bindings, trainer, sampler seed, run seeds,
objective direction, trials, evaluations, and artifacts.

## Before creating a group

Create an immutable benchmark suite revision with purpose `development`. Its
primary metric and `maximize` or `minimize` direction become the group's
objective. Halo Forge rejects an `unspecified` or `holdout` suite as the
optimization objective.

A separate suite with purpose `holdout` can be attached for later final
confirmation. It is not used for ranking, pruning, failure mining, or selecting
the best trial.

Start a queue worker in another terminal:

```bash
halo-forge jobs worker
```

The worker executes one accelerator-heavy item at a time and persists process
heartbeats, logs, cancellation, dependencies, and retry state.

## Repeat several seeds

Create `repeat.yaml`:

```yaml
name: support-sft-seeds
kind: repeat
trainer_mode: sft
development_suite_revision_id: <development-suite-revision>
dataset_bindings:
  - role: train
    dataset_version_id: <dataset-version>
    split: train
base_config:
  model: Qwen/Qwen2.5-3B-Instruct
  backend: hf
  output_root: ./runs
  learning_rate: 0.00002
  epochs: 1
seeds: [41, 42, 43]
```

Then create the group:

```bash
halo-forge sweep create --spec ./repeat.yaml
```

`sweep` is the experiment command family, so it manages repeats as well as
parameter sweeps. You can also create a repeat without a file:

```bash
halo-forge sweep create \
  --name support-sft-seeds \
  --kind repeat \
  --trainer sft \
  --suite <development-suite-revision> \
  --model Qwen/Qwen2.5-3B-Instruct \
  --dataset-version <dataset-version> \
  --seeds 41 42 43
```

## Create a parameter sweep

A search space accepts uniform, log-uniform, and discrete choice
distributions. Bare YAML lists are treated as choices.

```yaml
name: dpo-learning-rate
kind: sweep
trainer_mode: dpo
development_suite_revision_id: <development-suite-revision>
base_config:
  model: Qwen/Qwen2.5-3B-Instruct
  backend: hf
  output_root: ./runs
dataset_bindings:
  - role: train
    dataset_version_id: <preference-version>
    split: train
  - role: validation
    dataset_version_id: <preference-version>
    split: validation
n_trials: 12
sampler: random
sampler_seed: 2026
seeds: [17, 29]
search_space:
  learning_rate:
    kind: log_uniform
    low: 0.000001
    high: 0.0001
  batch_size: [1, 2, 4]
  lora_rank: [8, 16, 32]
```

```bash
halo-forge sweep create --spec ./dpo-sweep.yaml
```

Durable groups currently materialize `random` and `grid` samplers directly.
Unchanged input and `sampler_seed` produce the same trial set. Adaptive TPE is
available through the programmatic library, where completed objective values
can be reported before the next suggestion is chosen.

## Optional successive halving

Pruning is off by default and repeats are never pruned. A sweep can enable
synchronous successive halving:

```yaml
pruning:
  enabled: true
  reduction_factor: 3
  budgets: [100, 300, 900]
```

At a rung, Halo Forge waits for all active trials to have the required completed
seed coverage on the development objective. It then ranks in the suite's metric
direction, records the decision, and promotes the top cohort. The budgets are
trainer steps or cycles according to the verified trainer capability.

Current gated capabilities are Hugging Face SFT, DPO, ORPO, RM, and GRPO by
step; RAFT, VLM, audio, reasoning, and agentic trainers by cycle. MLX DPO and
GRPO currently run as full trials and cannot enable pruning.

The same policy can be supplied from the command line:

```bash
halo-forge sweep create \
  --spec ./dpo-sweep.yaml \
  --prune \
  --budgets 100 300 900 \
  --reduction-factor 3
```

## Adaptive checkpoint policies

Checkpoint policies are immutable revisions. They pin an ordered schedule, a
development-suite metric and direction, optional guardrail or plateau rules,
practical thresholds, patience, reviewed/automatic boundary behavior, and
retention guidance. Holdout and operational suites cannot guide continuation.

Create a policy from a reviewed YAML/JSON definition:

```bash
halo-forge checkpoint-policy validate --spec ./checkpoint-policy.yaml
halo-forge checkpoint-policy create --spec ./checkpoint-policy.yaml
halo-forge checkpoint-policy list --trainer sft
```

Then pin the revision when creating a group. Step-gated Hugging Face trainers
require an explicit step budget; cycle-gated trainers require a cycle budget:

```bash
halo-forge sweep create \
  --spec ./dpo-sweep.yaml \
  --checkpoint-policy <policy-revision-id> \
  --max-steps 900
```

The resolved boundaries and policy hash are stored with the group. Halo Forge
refuses unsupported boundary units and pauses when required evidence is
missing or incompatible.

## Analyze and record the decision

After the seed cohort completes, inspect the trajectory and create a
content-addressed matched-seed analysis:

```bash
halo-forge sweep checkpoints <run-group-id>
halo-forge sweep analyze <run-group-id> \
  --confidence 0.95 \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 42
```

The default interval is a deterministic percentile bootstrap over seed-level
outcomes. Evaluation examples remain diagnostic evidence; they are never
treated as independent training replicates.

Record the reviewed selection and queue a reproducibility bundle separately:

```bash
halo-forge sweep decide <run-group-id> \
  --select <trial-id> \
  --rationale "Matched-seed evidence clears the declared practical delta"

halo-forge sweep report <run-group-id> --decision <decision-id>
```

Reports are published atomically under
`~/.halo-forge/research/evidence/<bundle-id>/`. Recording a decision or report
never launches training, promotes an artifact, or creates data.

## Inspect and operate groups

```bash
halo-forge sweep list
halo-forge sweep show <run-group-id>
halo-forge sweep cancel <run-group-id>
halo-forge sweep resume <run-group-id> --reason "Reviewed retry after interruption"
halo-forge sweep compare <left-group-id> <right-group-id>
```

Group comparison requires the same development suite revision and respects its
metric direction. Cohort results include mean, standard deviation, range,
coverage, and terminal failures across the requested seeds.

To continue from the selected configuration:

```bash
halo-forge sweep fork-best <run-group-id> --json
```

This emits a reusable repeat specification with the resolved configuration and
parent group/trial/run context. It does **not** launch automatically; review the
specification and pass it to `sweep create` when ready.

## Inspect and operate queued work

```bash
halo-forge jobs list
halo-forge jobs show <work-item-id>
halo-forge jobs cancel <work-item-id>
halo-forge jobs retry <work-item-id>
halo-forge jobs worker --once
```

Ready items are chosen by priority and then FIFO order. Training and its pinned
development evaluation are linked by durable dependencies. A retained serving
lease blocks incompatible accelerator work until serving stops. On restart,
Halo Forge can adopt a matching live child or retain interrupted work for an
explicit retry.

Add `--json` to any `sweep` or `jobs` command for machine-readable output. Add
`--database` to point both commands at a non-default Halo Forge catalog.

## Dashboard

The **Experiments** workspace exposes the same service as the CLI. Its left pane
selects or creates groups, the main pane shows trials, seed coverage, objective
values, and artifacts, and the queue pane shows running work, blockers, and
progress. The Eval workspace labels development and holdout evidence and
disables failure mining from holdout results.

## Programmatic sweep library

The original callable-driven library remains available for lightweight custom
Python searches:

```python
from halo_forge.sweep import Choice, LogUniform, SearchSpace, SweepConfig, run_sweep

space = SearchSpace(params={
    "learning_rate": LogUniform(1e-6, 1e-3),
    "batch_size": Choice([1, 2, 4]),
})

config = SweepConfig(
    name="custom-search",
    search_space=space,
    n_trials=12,
    metric="accuracy",
    direction="maximize",
    sampler="random",
    seed=42,
)

def runner(trial_id, params):
    summary = launch_training_with_overrides(params)
    return {"accuracy": summary["accuracy"]}

result = run_sweep(config=config, runner=runner)
print(result.best_trial_id, result.best_value)
```

Use the programmatic library when a Python callable is the desired runner. Use
`halo-forge sweep` when training, evaluation, recovery, seed aggregation, and
the dashboard should share one durable record.
