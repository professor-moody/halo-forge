# Verifier Reliability and Reward Studio

Verifier Reliability turns a verifier from an informal configuration into a
pinned, inspectable research input. It covers deterministic program checks,
LLM judges, verified reward-model artifacts, and ordered verifier chains
without changing the existing verifier plugin interface.

The operational loop is:

```text
Human-reference labels
  → immutable verifier profile revision
  → replicated calibration
  → reliability evidence and explicit qualification
  → candidate/approved alias
  → exact binding in data, evaluation, review, or training
```

Nothing tunes, promotes, or starts training automatically. Threshold curves
are diagnostic. Changing a threshold, rubric, parser, chain, prompt, model, or
reward mapping creates a new profile revision and requires a new calibration.

## The objects that carry identity

- A **verifier profile revision** pins the family, implementation fingerprint,
  reliability-adapter version, modality and task, input mapping, reward
  contract, sanitized configuration hash, and runtime contract. Judge and
  reward-model profiles additionally pin their model or artifact identity,
  prompt/parser settings, and credential-free endpoint requirements.
- A **calibration protocol revision** pins repeats, orientations, perturbation
  policy, seeds, generation settings, batch-consistency checks, and bootstrap
  settings.
- A **qualification profile revision** pins a reviewed policy such as
  `strict_oracle`, `human_aligned`, or evidence-only `exploratory`.
- A **calibration** pins all three revisions plus one immutable reference
  source. Its samples and metrics are append-only and its published bundle is
  checksummed.
- A **qualification decision** is append-only. A later decision supersedes it;
  it does not rewrite history.
- The `candidate` and `approved` aliases are convenient pointers with
  append-only history. They do not change verifier identity.
- A **verifier binding** records the exact revision used by a dataset version,
  run, evaluation, review suggestion, evidence bundle, training artifact, or
  replay.

Secrets are removed recursively before configuration is stored or hashed.
Credentials remain in the provider/runtime configuration that already owns
them; they are never verifier-profile data.

## Reference evidence and exposure

Calibration accepts an immutable published Label Set revision or a compatible
development/unspecified benchmark-suite revision. Human labels are reference
labels for the declared task; Halo Forge does not call them population truth or
inter-annotator agreement.

Operational, holdout, test, canary, protected-lineage, unresolved, and
reward-model-training evidence is refused. Development exposure is recorded
and follows downstream bindings. Development-only reliability failures can be
sent to the reviewed Review Studio proposal flow; protected evidence remains
evidence-only.

When confirmation is requested, related records and shared media are kept
together in a deterministic 70/30 calibration/confirmation partition using
seed 42. Record, content, and media identities are checked for leakage.

## Replication and diagnostics

Deterministic profiles request two independent repetitions. Stochastic profiles
use seeds 17, 42, and 101 with temperature 0, top-p 1, and concurrency one.
Pairwise tasks include A/B and B/A; rankings include the canonical order and
deterministic cyclic rotations, capped at four. Length, style, and paraphrase
probes run only when the source contains explicitly reviewed variants.

Every invocation is normalized to one observation containing reward, pass
state, parsed and raw output, details, component trace, latency, error, and
runtime identity. Non-finite and out-of-contract rewards are rejected rather
than clamped. Chain evidence retains every child trace and a child error cannot
be hidden by the aggregate.

The universal report includes coverage, parse/error/timeout rates, reward
saturation, repeat drift and flips, latency, and missing evidence. The task
report adds the appropriate binary, categorical, multi-label, scalar,
pairwise, or ranking metrics. Brier score and equal-frequency ECE appear only
for profiles that explicitly declare probability semantics.

Confidence intervals use a grouped 95% percentile bootstrap. Stable records,
not repeats or pair/ranking expansions, are the replicate unit. The default is
10,000 resamples with seed 42. Subgroup metrics remain unavailable until there
are at least 20 distinct records in the subgroup.

## Qualification and promotion

The shipped policies are:

- `strict_oracle`: near-exact agreement, low false acceptance/rejection, high
  coverage, low errors, and exact deterministic repeat agreement;
- `human_aligned`: task-aware agreement plus repeatability and the applicable
  order-consistency or scalar-error gates; and
- `exploratory`: reports evidence but can never produce a promotable pass.

Fewer than 20 distinct records fails qualification. Evidence below the
promotable minimum is capped at `warn`, even when observed metrics pass. Every
`pass`, `warn`, or `fail` includes its exact reasons and the runtime scope used.

Normal `candidate` promotion requires passing development and operational
decisions. `approved` also requires confirmation when the calibration requested
it. An override requires a note, remains visible in history, and is excluded
from normal guided pickers. A changed implementation, parser, tokenizer,
toolchain, hardware contract, or runtime requirement produces
`stale_runtime` until recalibrated.

## Durable execution and artifacts

Calibration is durable workstation work with cancellation, retry, restart
recovery, isolated attempts, and one active calibration by default. Local
model-backed work uses the accelerator lease; hosted and programmatic work use
bounded CPU/provider execution.

Successful bundles live under:

```text
~/.halo-forge/evaluations/verifier-calibrations/<calibration-id>/
```

Each bundle contains the profile, source identity, protocol, runtime, samples,
metrics, qualification policy/evidence, and a checksum manifest. Reuse requires
the exact profile, source, protocol, runtime, perturbation, seed, and
qualification identities and a valid bundle.

## Guided and legacy use

Supplying `verifier_profile_revision_id` resolves one exact verifier. A launch
that also supplies contradictory raw verifier, threshold, parser, or verifier
configuration fields is refused. Guided pickers show compatible qualified
revisions by default.

Existing raw `--verifier` configurations remain runnable and are marked
`legacy_unqualified`. Replay manifests v3 carry verifier revision and hash,
implementation fingerprint, adapter, sanitized configuration hash, reward
contract, qualification scope, runtime compatibility, and the legacy warning.
Older v1/v2 manifests remain readable. Exact replay refuses missing or drifted
verifier identity unless the operator records a drift override reason.

## CLI workflow

Discover and define identity:

```bash
halo-forge verifier catalog
halo-forge verifier profile create --help
halo-forge verifier protocol create --help
halo-forge verifier qualification-profile create --help
```

Calibrate and inspect evidence:

```bash
halo-forge verifier calibration create --help
halo-forge verifier calibration list
halo-forge verifier calibration show <calibration-id>
halo-forge verifier calibration samples <calibration-id>
halo-forge verifier calibration compare <base-id> <candidate-id>
```

Record a decision, promote explicitly, and inspect bindings:

```bash
halo-forge verifier qualify --help
halo-forge verifier promote --help
halo-forge verifier usage --help
```

Commands that consume verifiers accept `--verifier-profile-revision` when the
selected capability supports a verifier. Existing raw verifier flags remain
available for compatibility.

## HTTP resources

The public API exposes bounded resources for capabilities, profiles and
immutable revisions, protocols, qualification profiles, calibration launch and
jobs, samples, metrics, cancellation, retry, comparison, integrity
verification, decisions, promotion history, runtime compatibility, and linked
usage. Large launches return both the domain calibration ID and its durable
`work_item_id`.

List responses retain the standard shape:

```json
{"items": [], "total": 0, "limit": 100, "offset": 0}
```

Calibration and evaluation samples are searched and paged on the server. The
dashboard never needs to materialize the full evidence catalog in the browser.

## Training-time integrity

Qualification answers whether a verifier agrees with pinned human-reference
evidence before use. It does not prove that the verifier will remain reliable
after a model optimizes against it. Reward Integrity pins a qualified optimizer
and independent sentinel, captures the outputs actually scored during training,
and rescales those same outputs without regeneration. See
[REWARD_INTEGRITY.md](REWARD_INTEGRITY.md).

## Relationship to the older verifier guide

[VERIFIERS.md](VERIFIERS.md) documents the verifier implementations and the
historical training interface. This document covers reproducible identity,
calibration, qualification, and exact downstream use. Verifier evidence is
still training infrastructure; benchmark reporting remains a separate
evaluation concern.
