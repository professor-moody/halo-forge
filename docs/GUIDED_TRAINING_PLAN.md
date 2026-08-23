# Guided training plans

After Dataset Lab publishes a clean immutable version, Halo Forge recommends
one bounded training plan. The recommendation is computed by the backend from
the scenario, exact dataset statistics, verified model/trainer compatibility,
and the active workstation. Normal mode does not ask the operator to choose
precision, microbatch size, gradient accumulation, sequence configuration, or
adapter details.

The normal flow is:

```text
Recommended plan → Prepare and check → Ready for proof run → Start proof run
```

**Prepare and check** is explicit confirmation for any displayed public-model
download. Gated model access and hosted-provider work require their own consent.
Halo Forge resolves the model to an immutable commit in the standard cache,
verifies the model inventory, then runs disposable capacity work. Scratch state
is isolated and removed after success, failure, cancellation, or retry.
The check uses the shipped trainer adapter to load and collate deterministic
median- and maximum-cost training records, run forward and backward passes,
allocate the optimizer, and verify at least one scratch optimizer step. Only
record identities, tensor or media dimensions, hashes, timings, and resource
measurements are retained. Source content and scratch weights are not retained.

Instruction tuning, preference methods, verifier-guided training, continued
pretraining, VLM, ASR, and specialized task models all use this same normal
workflow. A trainer/backend pair is hidden from guided mode unless it has a
registered capacity adapter.

The capacity fallback lattice is intentionally narrow:

1. the confirmed configuration;
2. verified gradient checkpointing;
3. a smaller microbatch with compensating accumulation.

Halo Forge never silently changes the model, objective, sequence length,
learning rate, verifier, reward behavior, or data. If those safe memory changes
do not fit, the screen recommends one repair such as choosing a smaller
compatible model or freeing memory.

## CLI

```bash
halo-forge train-plan recommend --dataset-version VERSION_ID
halo-forge train-plan show REVISION_ID
halo-forge train-plan alternatives REVISION_ID
halo-forge train-plan prepare REVISION_ID --wait
halo-forge train-plan check RESOLVED_REVISION_ID --wait
halo-forge train-plan proof RESOLVED_REVISION_ID --wait
```

Existing trainer commands accept `--training-plan-revision`. A plan takes
precedence over raw training settings; contradictory raw settings are rejected.
Advanced/manual launches remain supported and are labeled operator-configured.

## Identity and replay

Schema v21 stores immutable plan revisions, model preparations, capacity
attempts, append-only decisions, and run bindings. Replay format v12 records
the recommendation reasons, resolved model commit, complete compute-shape hash,
capacity evidence, selected adjustment, confirmation, and bound run identity.
Capacity evidence is reused only when both compute-shape and runtime hashes
match exactly.
