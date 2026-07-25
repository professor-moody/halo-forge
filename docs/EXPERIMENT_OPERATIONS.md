# Reproducible Experiment Operations

Halo Forge experiment operations turn an immutable dataset version and a
development benchmark into a reproducible group of training runs:

```text
dataset version + development suite revision
  → repeat or sweep
  → queued training/evaluation work
  → seed-aware comparison
  → selected artifact or fork
```

The implementation is designed for one researcher on one workstation. It adds
durable orchestration and traceability without adding a project or general
experiment-matrix object.

## Run groups

A **run group** is either:

- a **repeat**, which runs one resolved configuration with several explicit
  seeds; or
- a **sweep**, which deterministically materializes several configurations from
  a search space and runs the configured seeds for each one.

Each group pins its trainer mode, base configuration, dataset bindings,
development suite revision, metric direction, sampler seed, run seeds, and
optional pruning policy. Every parameter trial, seed run, and training segment
has a persistent identity. Training receives the final run ID before data
preparation or process launch, so the database, output directory, logs, replay
manifest, evaluation, and artifact relationships use the same ID.

Run seeds are scientific inputs, not labels added after execution. Unchanged
group input, search space, sampler, and sampler seed produce the same trial
materialization. Each seed still receives its own immutable run directory.

## Development and holdout suites

Benchmark suites declare a purpose:

- `development` guides trial ranking, checkpoint decisions, selection, and
  reviewed failure mining;
- `holdout` is reserved for final confirmation and is never an optimization
  dependency; and
- `unspecified` preserves the meaning of legacy suites, but cannot be used as a
  pinned development objective for a new group.

The group's primary metric and `maximize` or `minimize` direction come from its
pinned development suite revision. Halo Forge rejects a conflicting metric or
direction instead of silently ranking the cohort with different semantics.
Comparisons only combine compatible evidence, and holdout results cannot be
used to build failure-mined training data. Checkpoint and unpinned remote-model
subjects are also rejected for holdout launches.

Evaluation samples carry evidence validity and mining eligibility in addition
to score, latency, generation seed, token counts, finish reason, runtime
versions, threshold/direction, coverage, and template identity. Legacy summary
files remain readable, but are marked as legacy evidence rather than treated as
complete sample-level evaluation.

## Trainer capability gating

Halo Forge uses a versioned trainer/backend capability registry before enabling
segmented execution or pruning. Current verified execution units are:

| Trainer/backend | Execution unit | Gated continuation |
|---|---:|---:|
| Hugging Face SFT, DPO, ORPO, RM, GRPO | optimizer step | yes |
| RAFT | cycle | yes |
| VLM, audio, reasoning, agentic | cycle | yes |
| MLX DPO and GRPO | full trial | no |

An unregistered trainer/backend can still run as a full trial when its normal
training command is available. It cannot opt into pruning until a resumable
segment and checkpoint contract has been verified. This prevents the dashboard
or API from promising checkpoint continuation that the trainer cannot perform.

## Optional successive halving

Pruning is off by default and is never available for repeat groups. A sweep may
opt into synchronous successive halving when its trainer capability supports
gated continuation.

The default reduction factor is three. Budgets must be positive and strictly
increasing. At each rung Halo Forge waits for every active trial to have the
required completed seed coverage on the pinned development metric, ranks in the
declared metric direction, then promotes the top cohort and records a durable
continue/prune decision. A terminal failure ranks behind a valid completed
observation. Test and holdout suites do not participate in these decisions.

## Durable workstation queue

Managed run-group training and its dependent checkpoint evaluations share one
SQLite-backed work queue. A work item records its launch specification,
dependencies, resource class, priority, queue position, progress, logs, retry
count, process identity, and result. Existing standalone Dataset Lab and
evaluation jobs retain their compatible job managers; launching evaluation as
part of a run group places it in this experiment queue.

Only one accelerator-heavy item is leased at a time. Ready work is chosen by
priority and then FIFO order; dependency-blocked work stays queued. A retained
serving lease also blocks incompatible accelerator work until serving stops.
The worker executes argv directly without a shell, binds the child process to
the durable claim, and heartbeats its lease while it runs.

Run the worker in the foreground while using queued experiments:

```bash
halo-forge jobs worker

# Process at most one ready item, which is useful for scripts and testing.
halo-forge jobs worker --once
```

Inspect or control work without losing its provenance:

```bash
halo-forge jobs list
halo-forge jobs show <work-item-id>
halo-forge jobs cancel <work-item-id>
halo-forge jobs retry <work-item-id>
```

Cancellation requests are persisted. A running child receives a graceful
termination request and is force-stopped only after the configured timeout.
Interrupted or stale claims are recovered on restart; a live matching child
can be adopted, while an item whose process is gone becomes retryable. Retrying
keeps the original work identity and increments its attempt history.

## CLI workflow

Create an immutable development suite revision first, then describe the group
in YAML or JSON. A repeat example:

```yaml
name: sft-seed-check
kind: repeat
trainer_mode: sft
development_suite_revision_id: <development-revision-id>
dataset_bindings:
  - role: train
    dataset_version_id: <dataset-version-id>
    split: train
base_config:
  model: Qwen/Qwen2.5-3B-Instruct
  learning_rate: 0.00002
  epochs: 1
seeds: [41, 42, 43]
```

Create and operate on groups with the `sweep` command family. Despite the
historical name, it manages both repeat and sweep groups:

```bash
halo-forge sweep create --spec ./repeat.yaml
halo-forge sweep list
halo-forge sweep show <run-group-id>
halo-forge sweep cancel <run-group-id>
halo-forge sweep resume <run-group-id>
halo-forge sweep compare <left-group-id> <right-group-id>
halo-forge sweep fork-best <run-group-id> --json
```

A sweep adds `search_space`, `n_trials`, `sampler`, and `sampler_seed`. It may
also add `pruning.enabled`, `reduction_factor`, and ordered budgets. The existing
programmatic `halo_forge.sweep` library remains available for callable-driven
local searches; the CLI group path adds persistence, trainer launches,
evaluation dependencies, and restart recovery.

`fork-best` prints a reusable repeat specification with the resolved best
configuration and parent group/trial/run context. It does not launch the fork
until that specification is reviewed and passed to `sweep create`.

Existing training commands also accept explicit `--seed`,
`--dataset-version`, and repeatable `--dataset-binding` flags. Direct manual
training remains immediate and backward compatible; use a run group when the
work should be queued, repeated, compared, and recovered as one operation.

## Dashboard and API

The dashboard's **Experiments** workspace provides three linked views:

- a run-group list and repeat/sweep composer;
- trial, seed, objective, artifact, and cohort detail for the selected group;
  and
- the live workstation queue with progress and blockers.

The Eval workspace distinguishes development from holdout suites and shows
evidence gaps before enabling failure mining. Dataset Version and Run views
retain links to the exact group, bindings, evaluation, and artifacts.

The HTTP API exposes the same service layer through run-group, work-item, and
model-artifact resources. Run-group creation returns persistent group and work
identities immediately; large work is never held open by the request. Resource
detail includes trials, seed runs, segments, cohort aggregates, queue items,
artifacts, and recorded exposures. Lifecycle actions cover cancellation,
resume/retry, comparison, and best-trial forking.

## Artifacts and exposure lineage

Completed checkpoints, adapters, and final models can be registered as minimal
content-identified model artifacts linked to their run, run group, trial,
segment, base subject, and source path. This is the reproducible handoff for
evaluation and forking; conversion, retention, and richer artifact lifecycle
management remain separate concerns.

The append-only exposure ledger records identifiable suite-item use by dataset
versions, groups, runs, and model artifacts. Evaluations record direct exposure.
Reviewed failure-mined child datasets record their selector and source
evaluation, and inherit applicable parent exposure. This makes it possible to
distinguish development evidence from genuinely held-out confirmation after
several train/evaluate/mine cycles.

## Compatibility

Dataset Lab v1 versions without lineage sidecars receive stable virtual record
identities when read. Historical runs remain readable without run-group rows,
old replay manifests remain supported, manual dataset paths still work, and
legacy `lm_eval_summary.json` results are exposed read-only. The durable queue
applies to new managed work; it does not rewrite completed output directories or
silently enroll old direct CLI launches in a group.
