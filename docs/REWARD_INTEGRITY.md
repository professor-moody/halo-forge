# Reward Integrity and Training Signal Studio

Reward Integrity checks whether a qualified training verifier remains
trustworthy while a model optimizes against it. It complements static verifier
calibration; it does not replace it.

```text
qualified reward system
  -> capture the outputs actually scored during training
  -> seal a deterministic retained snapshot at a selected boundary
  -> rescore those same outputs with an independent sentinel
  -> analyze agreement, reward gaps, and boundary trends
  -> continue, pause for review, stop, or fork from the checkpoint
```

The sentinel never regenerates a candidate and never contributes to filtering,
selection, gradients, or the training reward. A person must review every stop,
fork, data proposal, and override. Halo Forge never changes a verifier, reward
mapping, threshold, prompt, or shaping rule from an audit result.

## Immutable identity

A **reward-system revision** pins the optimization-verifier revision, modality,
task, input mapping, reward normalization and shaping, failure and keep policy,
one primary sentinel, and up to three diagnostic auditors. The revision also
pins implementation, model, tokenizer, adapter, and sanitized configuration
hashes inherited from its verifier profiles.

Guided gating requires both the optimizer and primary sentinel to be qualified,
runtime-current, and compatible with the task and modality. Their verifier
revision, model artifact, and chain-leaf implementation fingerprints must be
disjoint. A correlated or unfingerprintable auditor can still appear in an
inspection report, but it cannot decide a training gate.

Audit behavior is also immutable:

- a **reward-audit protocol revision** pins retention mode, seed, core and
  diagnostic limits, and whether complete snapshots are required for gating;
- a **reward-integrity profile revision** pins the metrics, directions,
  pass/warn limits, evidence minimums, and report-only status; and
- a **training-signal shard** pins the run, boundary, dataset lineage,
  checkpoint and producer hashes, capability and fidelity, retained-set hash,
  aggregate counts, runtime, and a checksum manifest.

SQLite schema v11 adds reward systems, auditors, protocol/profile revisions,
direct-run segments, sealed shards, audits, samples, component observations,
metrics, decisions, and downstream bindings additively. Existing v1-v7 rows
are not rebuilt and historical runs remain `not_recorded` unless a real shard
was captured.

Managed Dataset Lab signal-capable training artifacts use the v3 row-to-lineage
contract. CPT corpus artifacts use format v4 to add exact model, tokenizer, and
packing identity. Manual files and older artifacts get explicit virtual
identities; Halo Forge does not infer historical signal traces from loss or
aggregate reward summaries.

## Trainer capture capabilities

`halo-forge reward capabilities` is authoritative for the installed runtime.
The built-in registry currently declares:

| Trainer/backend | Boundary | Resumable audit gate | Capture fidelity |
|---|---|---:|---|
| RAFT / Hugging Face | final | no | sampled |
| RAFT / MLX | cycle and final | yes | sampled |
| GRPO / Hugging Face | step and final when `max_steps` is configured | yes | sampled |
| GRPO / MLX | final | no | sampled |
| Reasoning / Hugging Face | cycle and final | yes | sampled |
| Agentic/tool / Hugging Face | cycle and final | yes | sampled |
| VLM / Hugging Face | cycle and final | yes | sampled, with image hash |
| Audio / Hugging Face | cycle and final | yes | sampled, with audio hash |

Here, `sampled` describes retention, not reconstruction: every retained row
contains the exact output that the training verifier saw. Unsupported and
third-party backends can truthfully declare `aggregate_only` or `unavailable`;
those modes produce reports but cannot gate training. Hugging Face RAFT and MLX
GRPO are explicitly final-only. Hugging Face GRPO exposes step boundaries only
when the managed launch resolves a positive `max_steps`; otherwise it also has
only a final audit. A final audit can delay completion and promotion review,
but it is not described as a mid-training pause.

Each captured observation retains stable source identity, candidate ordinal,
prompt or task context, exact output, available reference data, hashed media,
generation settings, the normalized optimization-verifier observation and
component trace, selection/drop outcome, runtime identity, and checkpoint
identity. Aggregate counts and reward distributions are retained for all
observations, including rows outside the snapshot.

## Retention protocols

The built-in protocols are deterministic with seed 42:

- `balanced_256` keeps a 192-record uniform core plus up to 64 diagnostic rows
  across verifier errors, threshold-adjacent outputs, highest optimizer rewards,
  and chain-component disagreement;
- `broad_512` uses a 384-record uniform core plus up to 128 diagnostic rows; and
- `exhaustive` retains every captured output.

If a boundary contains no more rows than the protocol limit, all rows are kept
and the shard is marked exact. Diagnostic rows are reported separately and are
never pooled into population-rate estimates. Snapshot IDs and stable event IDs
make retry/resume idempotent. Unsealed attempt tails are not accepted as audit
evidence.

Sealed traces live under:

```text
~/.halo-forge/training-signals/<run-id>/<segment-id>/<trace-hash>/
```

## Same-output audits

The primary sentinel receives the retained input, context, media identity, and
exact output. It does not sample from the model again. Rewards are normalized to
`[0,1]`, with larger values meaning better, before compatible cross-verifier
comparisons.

Reports include paired coverage, errors and timeouts, pass agreement,
optimizer-only and sentinel-only acceptance, normalized reward gap, rank
agreement where the contracts expose enough levels, top-tail disagreement,
saturation, flips, component disagreement, subgroups, and matched-identity
boundary trends. Confidence intervals use 10,000 grouped percentile-bootstrap
resamples with seed 42. The stable source record—not candidates, repetitions,
or chain components—is the replicate unit. Diagnostic rows remain separate.

The built-in profiles are:

- `strict_integrity`, intended for independently fingerprinted deterministic,
  strictly qualified verifiers;
- `human_aligned_integrity`, the normal guided default; and
- `exploratory`, which is report-only and cannot grant a gating pass.

At least 100 distinct records are required for `pass`. From 20 through 99,
otherwise passing evidence is capped at `warn`. Fewer than 20 records, corrupt
evidence, stale identity, or missing required fields produces
`incomplete_evidence`; Halo Forge does not turn missing evidence into a
scientific failure. A matched optimizer-up/sentinel-down trend whose grouped
intervals both exclude zero is a failure.

`pass` and `warn` continue. `fail` and `incomplete_evidence` pause a resumable
run at its checkpoint boundary. The review actions are **Continue**, **Stop**,
**Fork from checkpoint**, and **Create review proposal**. Each action requires a
reason and is append-only. Creating a Review Studio proposal does not resolve
the pause or launch training.

An optional development or unspecified-purpose suite revision can be pinned on
the managed training binding. After each selected checkpoint is verified and
published, Halo Forge evaluates that exact checkpoint as durable scheduler work.
The evaluation depends on the training segment, and the same-output reward
audit depends on the evaluation, so a failed evaluation is visible and blocks
the audit until it is retried successfully. The evaluation ID, work item, suite,
checkpoint identity, and status remain linked in the audit and run evidence.

This is completion and evidence tracking only. V8 does not define a development
quality threshold, so development metrics do not change reward-integrity
`pass`, `warn`, `fail`, or `incomplete_evidence`. The reward-integrity decision
remains the only automated input to the checkpoint gate.

## Command-line workflow

Discover capabilities and define immutable inputs:

```bash
halo-forge reward capabilities
halo-forge reward system validate --spec ./reward-system.json
halo-forge reward system create --name "Code reward + sentinel" \
  --spec ./reward-system.json
halo-forge reward protocol list
halo-forge reward integrity-profile list
```

Enable managed capture on a compatible verifier-guided training command:

```bash
halo-forge grpo train \
  --max-steps 400 \
  --reward-system-revision <reward-system-revision-id> \
  --reward-audit-protocol-revision <protocol-revision-id> \
  --reward-integrity-profile-revision <profile-revision-id> \
  --reward-development-suite-revision <optional-development-suite-revision-id> \
  --reward-audit-boundary 100 \
  --reward-audit-boundary 400
```

The three revision options are a single binding and must be supplied together.
They conflict with a separate verifier-profile revision because the reward
system already pins the optimization verifier. Existing raw `--verifier`
launches remain compatible, but they are unmonitored and cannot satisfy audit
readiness.

Inspect traces and durable audit work:

```bash
halo-forge reward trace list --run-id <run-id>
halo-forge reward trace verify <trace-id>
halo-forge reward audit list --run-id <run-id>
halo-forge reward audit show <audit-id>
halo-forge reward audit samples <audit-id>
halo-forge reward audit metrics <audit-id>
halo-forge reward audit compare <base-audit-id> <candidate-audit-id>
halo-forge reward audit review <audit-id> \
  --action continue --reason "Reviewed the paired evidence"
```

Audit launch, cancellation, retry, bundle verification, and reviewed fork/stop
actions are available under `halo-forge reward audit`. Commands support
`--database` and `--json`; evidence lists are bounded and paginated.

## HTTP and dashboard surfaces

The public API exposes bounded resources for:

- `/reward-integrity-capabilities`;
- `/reward-systems`, `/reward-audit-protocols`, and
  `/reward-integrity-profiles`, including immutable revisions;
- `/training-signals` and run-linked trace lists; and
- `/reward-integrity-audits`, samples, metrics, comparisons, integrity
  verification, cancellation, retry, and gate review.

Large launches return the domain audit ID and durable `work_item_id`. List
responses use `{items,total,limit,offset}`. The dashboard surfaces reward-system
setup in Train and Experiments, boundary audit status on Runs and Activity, and
same-output evidence under Evaluate -> Verifiers -> Training audits.

## Replay v5 and limits

Replay v4 introduced the reward-system revision/hash, optimizer and ordered
auditors, reward mapping, protocol and integrity profile, boundary schedule,
signal capability, trace manifests, audit decisions, and runtime compatibility.
Current replay manifests are v5 and retain those fields while adding
corpus-training identity. Older manifests remain readable and are never
presented as monitored when the corresponding evidence is absent.

Exact replay refuses missing or changed reward identity. An intentional
override requires both `--allow-reward-drift` and a non-empty
`--reward-drift-reason`; Halo Forge records that reason beside the manifest.
See [REPLAY.md](REPLAY.md) for the complete compatibility rules.

Reward Integrity audits the existing reward system. It does not introduce a
new reward algorithm, tune a verifier, generate replacement outputs, create a
dataset, launch a fork, or promote an artifact. Protected operational, holdout,
test, and canary evidence remains evidence-only and cannot guide a training
gate or seed a review proposal.
