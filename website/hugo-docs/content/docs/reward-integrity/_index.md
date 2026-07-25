---
title: "Reward Integrity"
description: "Capture verifier-guided training outputs and rescore the same outputs with an independent qualified sentinel."
weight: 32
---

Reward Integrity checks whether a qualified verifier stays trustworthy while a
model optimizes against it.

```text
qualified reward system -> exact training output capture
-> independent same-output sentinel -> integrity decision
-> continue or reviewed pause / stop / checkpoint fork
```

The sentinel receives the exact captured input, context, media identity, and
output. It never regenerates a candidate and its scores never enter training
selection, filtering, gradients, or reward. Halo Forge does not tune thresholds,
prompts, chains, mappings, or reward shaping from an audit.

## Reward systems

An immutable reward-system revision pins:

- the qualified optimization-verifier revision;
- modality, task, input mapping, reward normalization and shaping, failure and
  keep behavior;
- one qualified primary sentinel and up to three optional diagnostic auditors;
  and
- implementation, chain-leaf, model, artifact, tokenizer, adapter,
  configuration, and runtime hashes.

Guided gating requires the optimizer and primary sentinel to be compatible,
runtime-current, and disjoint across revision, artifact, and implementation
fingerprints. Correlated or unfingerprintable auditors are inspection-only.

The retention protocol and integrity policy are separate immutable revisions.
Changing any verifier, mapping, boundary protocol, or decision rule creates a
new identity.

SQLite schema v11 is additive. Older runs remain readable and are marked
`not_recorded`; Halo Forge never reconstructs a trace from aggregate summaries.

## Capture coverage

The installed capability registry is authoritative:

```bash
halo-forge reward capabilities
```

| Trainer/backend | Boundaries | Resumable | Fidelity |
|---|---|---:|---|
| RAFT / Hugging Face | final | no | sampled |
| RAFT / MLX | cycle, final | yes | sampled |
| GRPO / Hugging Face | step, final when `max_steps` is configured | yes | sampled |
| GRPO / MLX | final | no | sampled |
| Reasoning / Hugging Face | cycle, final | yes | sampled |
| Agentic/tool / Hugging Face | cycle, final | yes | sampled |
| VLM / Hugging Face | cycle, final | yes | sampled + image hash |
| Audio / Hugging Face | cycle, final | yes | sampled + audio hash |

`sampled` means deterministic retention; retained rows still contain the exact
training output. Unsupported backends can declare `aggregate_only` or
`unavailable`, which is report-only. Hugging Face RAFT and MLX GRPO are
explicitly final-only. Hugging Face GRPO exposes step boundaries only when the
managed launch resolves a positive `max_steps`; otherwise it also has only a
final audit. A final failure delays completion review but is not described as a
mid-training pause.

Managed Dataset Lab artifacts preserve `record_id`, `record_hash`, and
`instance_id`; legacy and manual sources get explicit virtual identities.
Captured rows include candidate ordinal, selection outcome, generation settings,
optimizer observation and component trace, checkpoint/runtime identity, and
hashed media where applicable.

## Retention and analysis

Built-in protocols use seed 42:

- `balanced_256`: 192-record uniform core plus up to 64 separately reported
  diagnostic rows;
- `broad_512`: 384-record core plus up to 128 diagnostics; and
- `exhaustive`: retain everything.

Diagnostics cover verifier errors, threshold-adjacent outputs, highest optimizer
rewards, and chain-component disagreement. They are not pooled into population
rates. If a boundary fits within its protocol limit, every output is retained
and the shard is marked exact.

The audit reports coverage, errors, pass agreement, asymmetric acceptance,
normalized reward gaps, rank agreement when applicable, top-tail disagreement,
saturation, component traces, subgroups, and matched-identity boundary trends.
Intervals use 10,000 grouped bootstrap resamples with seed 42; the stable source
record is the replicate unit.

`human_aligned_integrity` is the guided default. `strict_integrity` is intended
for independently fingerprinted deterministic strict verifiers.
`exploratory` is report-only. At least 100 distinct records are needed for pass;
20-99 caps evidence at warn; fewer than 20 or corrupt, stale, or incomplete
evidence yields `incomplete_evidence`.

Pass and warn continue. Fail and incomplete evidence pause a resumable run.
Continue, Stop, and Fork from checkpoint require a reason and append a visible
decision. **Create review proposal** opens the reviewed Review Studio flow and
does not resolve the pause.

An optional development or unspecified-purpose suite revision can be pinned on
managed training. Halo Forge evaluates the exact published checkpoint after the
training segment; the reward audit waits for that durable evaluation. Evaluation
failures stay visible and retryable, and block the audit instead of being
ignored. The linked evaluation and work-item identities remain in audit and run
evidence.

This is completion and evidence tracking only. V8 defines no development-quality
threshold, so suite metrics do not change the reward-integrity decision. Only the
reward-integrity decision is an automated checkpoint-gate input.

## Command line

```bash
halo-forge reward system validate --spec ./reward-system.json
halo-forge reward system create --name "Code reward + sentinel" \
  --spec ./reward-system.json
halo-forge reward protocol list
halo-forge reward integrity-profile list

halo-forge grpo train \
  --max-steps 400 \
  --reward-system-revision <system-revision-id> \
  --reward-audit-protocol-revision <protocol-revision-id> \
  --reward-integrity-profile-revision <profile-revision-id> \
  --reward-development-suite-revision <optional-development-suite-revision-id> \
  --reward-audit-boundary 100 --reward-audit-boundary 400

halo-forge reward trace list --run-id <run-id>
halo-forge reward trace verify <trace-id>
halo-forge reward audit show <audit-id>
halo-forge reward audit samples <audit-id>
halo-forge reward audit metrics <audit-id>
halo-forge reward audit review <audit-id> \
  --action continue --reason "Reviewed the paired evidence"
```

The three training revision flags form one managed binding and must be supplied
together. Existing raw verifier commands remain runnable, but are unmonitored
and cannot satisfy audit readiness.

## API, storage, and replay

The dashboard and CLI share `/api/public` resources for capabilities, reward
systems, protocols, integrity profiles, run-linked signal shards, audits,
samples, metrics, comparison, verification, cancellation, retry, and gate
review. Lists are bounded and large launches return a durable `work_item_id`.

Sealed traces and audit bundles live under:

```text
~/.halo-forge/training-signals/<run-id>/<segment-id>/<trace-hash>/
~/.halo-forge/evaluations/reward-audits/<audit-id>/
```

Replay v4 introduced reward-system, auditor, mapping, protocol, integrity-
profile, boundary, capability, trace, audit, decision, and runtime identity.
Current v5 manifests retain that evidence and add corpus-training identity.
Exact replay refuses reward drift unless the operator supplies
`--allow-reward-drift` and a recorded `--reward-drift-reason`. Older manifests
remain readable but are never presented as monitored.

Operational, holdout, test, and canary evidence cannot guide a training gate or
seed a review proposal. An audit never creates data, launches a fork, or
promotes an artifact automatically.
