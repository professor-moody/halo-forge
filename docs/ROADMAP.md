# Halo Forge Lab Roadmap

**Updated:** 2026-07-17

Halo Forge is a local workbench for model-training research. Its operating loop
is now:

```text
Development evidence or unlabeled data
  → acquisition proposal → approved review queue → human labels/adjudication
  → approved immutable label set → reviewed child dataset version
  → explicit training launch or repeat

Dataset version → repeat/sweep → train → evaluate → compare
                ↑                                      ↓
                └─── fork or failure-mined child version ───┘

Completed run → artifact → transform → qualify → promote → serve/export

Qualified reward system → exact training output capture → independent sentinel
                        → integrity decision → continue or reviewed pause

Own data → proof outcome → controlled adaptation study → grounded reviewed data
         → specialized task model → deterministic agent trajectory
```

The Lab is organized around immutable dataset and label-set versions, review
queues, run groups, runs, benchmark-suite revisions, and model artifacts. It
deliberately does not add a general research-project or experiment-matrix
object. Acquisition, queue creation, label publication, dataset building, and
training remain separate reviewed actions.

## Delivered foundation

### Dataset Lab v1 — local multimodal data factory

Delivered:

- persistent local and pinned Hugging Face dataset sources;
- canonical text, chat, preference, reasoning/RLVR, tool, VLM, and audio
  records;
- ordered YAML/JSON recipes for mapping, validation, filtering, deduplication,
  scoring, sampling, mixtures, splits, contamination checks, curricula,
  failure imports, and annotation synthesis;
- immutable, content-identified versions with source and asset fingerprints;
- profiling, previews, quarantines, provenance, optional media
  materialization, atomic publication, and restart-safe jobs;
- dashboard, API, and CLI registration, build, inspection, export, and direct
  Train handoff; and
- compatibility for existing `prepare`, `validate`, `dedup`, `score`, and
  `synthesize` workflows.

See [Dataset Lab](DATASET_LAB.md).

### Dataset Lab v2 — closed-loop training and evaluation

Delivered:

- stable record, content, and per-version instance identities with lineage;
- versioned trainer-dataset adapters and content-addressed trainer artifacts;
- exact preservation of supplied validation data and exclusion of test/canary
  data from trainers;
- canonical run identity, explicit dataset roles, replay-complete dataset and
  tokenizer identity, and run/dataset traceability;
- immutable benchmark suite revisions, persistent evaluations, metric
  direction, sample-level evidence, subject hashing, and compatible-result
  reuse;
- base/candidate comparison and reviewed, provenance-preserving failure mining;
- dataset-version comparison by membership, content, split movement, recipe,
  statistics, and source contribution; and
- dashboard, API, and CLI flows for render, train, evaluate, compare, mine, and
  fork while retaining legacy versions, paths, manifests, and evaluation
  summaries.

### Lab v3 — reproducible experiment operations

Delivered and now the current operational layer:

- persistent repeat and sweep groups with deterministic trial materialization,
  explicit run seeds, canonical run IDs, and immutable output directories;
- a pinned development-suite objective with explicit metric direction, plus a
  separate holdout purpose reserved for final confirmation;
- trainer/backend capability gating for step-, cycle-, or full-trial execution;
- opt-in synchronous successive halving with seed-complete rung decisions and a
  default reduction factor of three; repeats are never pruned;
- a durable SQLite work/dependency queue, one global accelerator lease,
  priority/FIFO ordering, process heartbeats, cancellation, retry, restart
  recovery, and serving-aware resource blocking;
- cohort aggregation and ranking across seeds, group comparison, and reviewed
  best-trial fork specifications;
- minimal checkpoint/adapter/final-model artifact records and append-only
  evaluation/failure-mining exposure lineage;
- evidence-validity and mining-eligibility metadata, bounded comparison pages,
  and honest treatment of legacy aggregate-only evaluation; and
- the Experiments dashboard plus `halo-forge sweep`, `halo-forge jobs`, and
  matching HTTP resources.

See [Reproducible Experiment Operations](EXPERIMENT_OPERATIONS.md).

### Lab v4 — workstation control plane and Artifact Studio

Delivered:

- one durable workstation scheduler and supervised dashboard worker for data,
  training, evaluation, experiments, artifact operations, cleanup, and serving;
- attempt isolation, event history, dependency blocking/reopening, bounded
  retries, PID/start-identity recovery, two-second resource samples, permanent
  telemetry rollups, and RAM/disk preflight;
- content-addressed artifact blobs with occurrence, location, operation, and
  ordered multi-parent lineage records;
- verified adapter bake/composition, conversion, post-training quantization,
  atomic publication, operation reuse, pins, tags, notes, and append-only alias
  history;
- immutable qualification and serving profiles, operational performance
  evaluation, direction-aware parent/candidate comparison, explicit promotion,
  and local portable export;
- storage inventory, protected reviewed cleanup plans, seven-day trash,
  restoration, and expired-trash purge; and
- a consolidated, responsive dashboard with structured forms, compatible
  pickers, contextual inspectors, command palette, and global Activity Center.

See [Artifact Studio](ARTIFACT_STUDIO.md).

### Lab v5 — adaptive training and Evidence Studio

Delivered:

- immutable checkpoint-policy revisions with explicit step/cycle schedules,
  development-only objectives and guardrails, practical thresholds, patience,
  reviewed or opt-in automatic boundary actions, and retention guidance;
- resolved trainer-compatible plans and a durable train boundary → checkpoint
  → evaluation → gate → continuation graph with idempotent recovery;
- truthful pause, review, stop, and continuation states, append-only operator
  overrides, checkpoint trajectory views, and Activity Center review actions;
- deterministic matched-seed cohort snapshots with 95% percentile-bootstrap
  intervals, 10,000 resamples, seed 42, practical-equivalence conclusions,
  compatibility checks, and no sample-level pseudo-replication;
- primary-objective ranking, explicit secondary evidence, and a post-training
  Pareto view that preserves unavailable speed, memory, energy, and size data;
- append-only reviewed research decisions and immutable Markdown, HTML, JSON,
  CSV, and SVG evidence bundles containing data, suite, run, checkpoint,
  runtime, assumptions, and missing-evidence identities;
- longitudinal evaluation history and direction-aware drift comparison; and
- a guided, autosaved Experiments workflow with labeled search pickers,
  server-side drafts, global search, checkpoint timelines, evidence review,
  responsive navigation, and matching API/CLI operations.

See [Adaptive Training and Evidence Studio](ADAPTIVE_EVIDENCE.md).

### Lab v6 — Human Feedback and Active Data Studio

Delivered:

- immutable annotation-schema revisions for text, preference, tool-use, VLM,
  and audio review, with binary, categorical, multi-label, scalar, correction,
  pairwise, and ranking tasks;
- deterministic, checksummed acquisition batches over development evaluations,
  comparisons, run samples, Dataset Lab versions, Playground sessions, and
  imported JSONL, including ordered quota strata and stable record deduplication;
- evidence-aware explicit, failure, regression, improvement, disagreement,
  low-score, low-margin, coverage, diversity, and seeded-random selection that
  refuses to fabricate unavailable scores, margins, or embeddings;
- protected-evidence enforcement: operational, holdout, test, and canary
  records cannot enter review acquisition, while development exposure is
  recorded and propagated to descendant label sets and dataset versions;
- one-pass and blinded, counterbalanced two-pass queues with explicit
  adjudication, append-only events, idempotent writes, stale-write rejection,
  drafts, deferral, flags, exclusions, correction history, and projection
  recovery;
- optional on-demand model suggestions with separate provenance and reveal
  history that never count as human decisions;
- immutable, content-addressed label-set revisions with canonical records,
  lineage, statistics, checksums, suggestion provenance, and exposure identity;
- reviewed `filter`, `replace_by_record_id`, `append`, and `annotate` handoff to
  immutable Dataset Lab child versions, followed by a separate Train or repeat
  launch; and
- Data → Review Queues dashboard workflows, durable Activity integration,
  matching HTTP resources, structured spec descriptors, and the complete
  `halo-forge review` CLI hierarchy.

See [Human Feedback and Active Data Studio](REVIEW_STUDIO.md).

### Lab v7 — Verifier Reliability and Reward Studio

Delivered:

- immutable verifier profiles for deterministic checks, LLM judges, verified
  reward-model artifacts, and ordered chains, with implementation,
  configuration, reward-contract, adapter, and runtime fingerprints;
- protected-source eligibility, deterministic grouped calibration/confirmation
  partitioning, replicated runs, pair/ranking counterbalancing, reviewed
  perturbations, reward-model batch checks, and normalized observations;
- task-aware reliability metrics, stable-record grouped bootstrap intervals,
  explicit missing evidence, reviewed qualification policies, append-only
  decisions, and candidate/approved alias history;
- durable, recoverable calibration work and atomic checksummed evidence bundles;
- exact qualified-by-default bindings in data, evaluation, review, training,
  evidence, and replay while retaining visibly unqualified legacy behavior;
- guided Evaluate and Verifier workspaces, server-paged evidence inspection,
  bounded multi-subject comparison, and reviewed reliability-failure handoff;
  and
- matching HTTP resources and the complete `halo-forge verifier` CLI hierarchy.

See [Verifier Reliability and Reward Studio](VERIFIER_RELIABILITY.md).

### Lab v8 — Reward Integrity and Training Signal Studio

Delivered:

- additive SQLite schema v11 records for immutable reward systems, protocols,
  integrity profiles, direct-run segments, sealed traces, audits, observations,
  metrics, decisions, and domain bindings without rebuilding older records;
- immutable reward-system revisions pinning one qualified optimization verifier,
  a disjoint qualified primary sentinel, optional diagnostic auditors, reward
  mapping and shaping, task/modality compatibility, and implementation,
  artifact, tokenizer, configuration, and runtime identities;
- a versioned trainer-signal capability registry and one identity-aware,
  bounded observation sink for RAFT, GRPO, reasoning, agentic/tool, VLM, and
  audio, including candidate ordinals, selection outcomes, component traces,
  hashed media, and exact captured outputs;
- deterministic `balanced_256`, `broad_512`, and `exhaustive` retention,
  append-only attempt-scoped shards, atomic sealing, checksum verification,
  resume deduplication, and explicit virtual identities for legacy/manual data;
- independent same-output sentinel rescoring without regeneration or influence
  on gradients, selection, filtering, or the optimization reward;
- paired coverage, agreement, acceptance asymmetry, normalized reward gaps,
  rank and top-tail diagnostics, grouped-bootstrap intervals, and matched-
  identity boundary trends with diagnostic/core populations kept separate;
- immutable strict, human-aligned, and report-only integrity profiles, explicit
  `pass`, `warn`, `fail`, and `incomplete_evidence` decisions, and reviewed
  Continue, Stop, Fork, or Review Studio proposal actions;
- durable trace and audit work, run/checkpoint/artifact bindings, bounded
  evidence APIs, dashboard audit workspaces, and the complete
  `halo-forge reward` CLI hierarchy; and
- replay manifest v4 reward, auditor, capability, trace, boundary, audit, and
  decision identities while keeping v1-v3 readable and requiring a recorded
  reason for exact-replay reward drift.

Hugging Face RAFT and MLX GRPO are truthfully final-boundary-only. MLX RAFT is
cycle-resumable, while Hugging Face GRPO supports resumable step audits when a
positive `max_steps` is resolved. Reasoning, agentic/tool, VLM, and audio remain
cycle-resumable. Optional development-suite identity is pinned on each audit;
V8 evaluates every exact published checkpoint through durable Evaluation Lab
work before its reward audit can run. Development results are completion and
evidence tracking only: V8 defines no quality threshold and does not combine
suite metrics with the checkpoint gate. Existing raw verifier launches remain
compatible but unmonitored. No audit result automatically tunes a verifier,
changes reward mapping, creates data, starts a fork, or promotes an artifact.

See [Reward Integrity and Training Signal Studio](REWARD_INTEGRITY.md).

### Labs v11–v15 — outcomes, studies, grounding, task models, and environments

Delivered as five independently releasable capability layers:

- V11 separates technical proof completion from development-quality evidence,
  records normalized findings and resource projections, and requires a
  compatible assessment or retained override before a full run;
- V12 adds immutable paired A/B, dose-response, and bounded 2×2 adaptation
  studies with paired seeds, domain/retention evidence, planned contrasts,
  grouped intervals, Holm correction, deviations, and reviewed decisions;
- V13 turns immutable corpora into cited generation proposals with exact
  source-span identity, structural verification, coverage reports, and an
  explicit Review Studio handoff;
- V14 adds verified PyTorch classification, multi-label, embedding, reranking,
  image-classification, and audio-classification data, training, evaluation,
  artifact, serving, and replay contracts; and
- V15 adds deterministic local state-machine environments, immutable episode
  suites, exact trace replay, comparison, step evidence, and reviewed
  trajectory publication for existing trainers.

Schema v18 and replay v9 remain backward readable. See
[Halo Forge Labs V11–V15](LABS_V11_V15.md).

## Current work

The current phase is V21 beta closure: progressively certifying real trainer
paths instead of treating hardware detection or a generic tensor update as
proof that guided training works. Runtime core qualification, exact trainer-path
certification, user-plan capacity evidence, and full workstation beta evidence
are separate states.

Delivered in the V21 implementation:

- schema v23 and replay v14 identity for immutable path profiles,
  certifications, evidence steps, attempts, bindings, and workstation reports;
- a ten-step real Dataset Lab → shipped trainer → parameter delta → artifact
  reload certification contract;
- automatic instruction-SFT certification after core AMD runtime qualification,
  with other paths verified progressively on demand;
- backend capability filtering that cannot be unlocked by lightweight tensor
  diagnostics;
- durable waiting, retry, resume, Activity, API, CLI, Setup, and Train surfaces;
- workstation reports that resolve concrete capacity, proof, artifact, outcome,
  recovery, coexistence, and soak records instead of accepting boolean claims;
  and
- CUDA retained as hardware-unqualified until equivalent evidence exists on a
  real NVIDIA host.

Remaining release evidence is operational rather than a new capability:

- add long-running crash/recovery soak coverage on representative workstations;
- complete an independently idle Strix instruction-SFT certification, managed
  capacity check, own-data proof, matched outcome assessment, external-workload
  wait/release exercise, and twelve-hour soak;
- progressively implement and certify the remaining advertised path executors;
- run the same real-path ladder on NVIDIA hardware before enabling CUDA guided
  scenarios;
- soak signal capture, trace sealing, same-output rescoring, boundary pause,
  reviewed continuation, and crash recovery across supported trainers;
- expand independent-sentinel fixtures and matched-identity reward-inversion
  scenarios without inventing unavailable scores or causal claims;
- improve backend-specific performance instrumentation while retaining nulls
  for metrics the runtime cannot measure;
- run restart, retry, and bounded-memory soak tests for large acquisition,
  suggestion, label publication, and reviewed dataset-build work;
- expand representative review fixtures for preference, tool, VLM, and audio
  records without generating or editing binary media; and
- refine keyboard, mobile, reconnect, and accessibility behavior using real
  large catalogs and queues;
- soak proof-outcome assessment and reviewed full-run gating;
- expand real task-model fixtures and serving round trips across supported
  text, image, and audio families; and
- validate deterministic environment snapshot/replay behavior over large
  episode and step catalogs.

## True future phases

These remain intentionally outside the current single-workstation Lab scope:

### Larger search and scheduling

- additional verified pruning policies and multi-objective search;
- multi-objective selection across quality, speed, memory, energy, and size;
- repeated-seed and budget orchestration across several workstations;
- distributed training and cluster scheduling; and
- statistically valid sequential-testing and budget-allocation policies beyond
  the declared checkpoint gates now supported.

### Richer artifact operations

- verified QAT backed by an actual quantization-aware training implementation;
- remote Hugging Face or registry publishing with reviewable credentials and
  release metadata;
- additional conversion backends once round-trip and loadability contracts are
  available; and
- adapter routing and continued-training policies beyond explicit operator
  actions.

### More autonomous data research

- autonomous active-learning proposals built from valid development failures;
- multi-reviewer assignment, accounts, and agreement studies beyond the
  current local one-reviewer/two-pass workflow;
- additional acquisition algorithms after they have pinned, reproducible
  evidence contracts;
- live or nondeterministic agent environments with explicit side-effect
  controls;
- binary image/audio generation and broader multimodal synthesis; and
- a bounded-memory streaming rewrite for every legacy recipe transform.

### Broader research analysis

- larger randomized designs beyond the delivered paired, dose-response, and
  bounded 2×2 adaptation-study templates;
- cross-backend reproducibility studies with normalized hardware telemetry; and
- cross-group meta-analysis and publication templates beyond the immutable
  evidence bundles now supported.

## Operational research angles

The delivered Lab can already support controlled studies of:

- raw versus cleaned, deduplicated, scored, mixed, curriculum, or synthetic
  data;
- dataset size, source contribution, mixture ratio, and sampling policy;
- SFT versus preference optimization or verifier-guided training on a shared
  base and suite;
- learning rate, batch size, LoRA configuration, epochs/cycles, and repeated
  seed variance;
- verifier choice, reward threshold, and development-to-holdout transfer;
- optimization-verifier versus independent-sentinel agreement, reward-gap,
  top-tail disagreement, and boundary-drift behavior on the same outputs;
- checkpoint/cycle continuation for trainers with a verified capability;
- model family, backend, modality, training cost, and inference behavior;
- acquisition-strategy and annotation-policy effects across deterministic
  review batches; and
- reviewed failures, corrections, preferences, tool traces, VLM annotations,
  or audio labels turned into child datasets and forked training groups without
  losing record, review, or exposure history.

The next roadmap phase should be selected from measured friction in these
workflows, not from adding another layer of organizational objects.
