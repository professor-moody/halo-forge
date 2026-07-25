# Adaptive Training and Evidence Studio

Adaptive Training and Evidence Studio connects durable experiment groups to
checkpoint evaluation and reviewed research decisions:

```text
run group -> train boundary -> checkpoint -> development evaluation
          -> continue / pause / stop -> cohort analysis -> decision -> fork or report
```

The feature is opt-in. Existing run groups and new groups without a checkpoint
policy keep their final-only behavior.

## Checkpoint policies

A checkpoint policy has immutable revisions. A revision pins the schedule,
development-suite evidence, metric direction, guardrails, plateau rules,
actions, and retention recommendations. When a group is created, Halo Forge
resolves percentages or intervals into explicit step or cycle boundaries and
stores the resulting plan with the group.

Only trainer/backend pairs with a verified resumable boundary contract can use
mid-training gates. Full-trial backends remain final-only. A gate can:

- continue to the next verified boundary;
- pause for operator review; or
- stop at the current checkpoint.

Missing or incompatible evidence pauses instead of silently passing. An
operator can override a pause or stop, but the reason is append-only. Test,
canary, operational, and holdout suites never guide continuation or pruning.

Every gated boundary must publish an exact resumable checkpoint. Halo Forge
aligns Hugging Face save and validation cadence to that boundary, verifies the
trainer's recorded step, and atomically adopts the checkpoint into the managed
content-addressed artifact store before evaluation or continuation begins.
Trainer-owned checkpoint rotation therefore cannot invalidate a prior gate.

Checkpoint retention is advisory. Policies can identify last, periodic, or
best checkpoints as cleanup candidates, but deletion still follows Artifact
Studio's reviewed trash workflow. Evaluated checkpoints and checkpoints used
by decisions or evidence bundles remain protected.

A compact step-based policy looks like this:

```yaml
policy_id: guarded-sft
name: Guarded SFT checkpoints
development_suite_revision_id: <development-revision-id>
primary_metric: accuracy
direction: maximize
schedule:
  mode: percentages
  unit: step
  percentages: [0.25, 0.5, 0.75, 1.0]
rules:
  - kind: plateau
    metric: accuracy
    direction: maximize
    comparison: previous
    minimum_delta: 0.002
    practical_delta: 0.002
    patience: 2
    on_breach: pause
automatic_actions: true
retention:
  keep_last: 2
  keep_best: 1
  keep_every_n_boundaries: 2
  protect_evaluated: true
  protect_decision_referenced: true
  protect_lineage_referenced: true
  review_before_cleanup: true
```

For a cycle-based trainer, change the schedule unit to `cycle` and supply the
group's cycle budget. Percentages are resolved to explicit integer boundaries
before any work is queued.

## Cohort evidence

Analysis snapshots are immutable and content-addressed. The default analysis
uses matched seeds, a deterministic 95% percentile-bootstrap interval, 10,000
resamples, and seed 42. Seed runs are the experimental replicates; per-record
evaluation deltas remain diagnostic evidence and are not counted as independent
training replicates.

Direction-normalized comparisons are reported as improved, regressed,
practically equivalent, inconclusive, or insufficient evidence. Missing seed
coverage, mismatched suite revisions, generation settings, templates, or
unavailable metrics remain visible rather than being normalized away.

Automatic ranking keeps one development metric as its primary objective and
uses secondary development metrics only as constraints. The dashboard also
shows a post-training Pareto view over available quality, latency, throughput,
memory, energy, and artifact-size measurements. It does not fabricate missing
measurements or hide tradeoffs behind a weighted score.

## Reviewed decisions and reports

A research decision records the analysis snapshot, selected and rejected
checkpoints, exclusions, missing-evidence acknowledgement, rationale, and an
optional fork specification. Recording a decision does not promote an
artifact, create data, or launch training.

Evidence bundles are built by the workstation scheduler and published
atomically under:

```text
~/.halo-forge/research/evidence/<bundle-id>/
```

Bundles contain a checksummed manifest, Markdown and HTML reports, machine-
readable evidence, dataset/suite/run/artifact identities, replay information,
runtime and hardware context, assumptions, and missing-evidence inventory.
Completed bundles are checksum-verified before every reuse. A failed check is
recorded as corruption without changing the original identity, and rebuilding
creates a new immutable bundle occurrence.

## Dashboard workflow

Experiments adds a guided Checkpoints and gates step. Group and Run views show
the resolved plan as a checkpoint trajectory with evaluations, decisions,
artifacts, and resource use on one timeline. Paused gates appear in Activity
with direct Inspect, Continue, and Stop actions. Evidence comparison and
decision review use searchable labeled pickers; full identifiers and raw
specifications remain available in the contextual inspector.

Long forms autosave as workstation drafts and can be resumed or discarded.
The command palette searches datasets, versions, runs, suites, groups,
artifacts, policies, and active work.

## CLI overview

```bash
halo-forge checkpoint-policy list
halo-forge checkpoint-policy create --spec ./policy.yaml
halo-forge checkpoint-policy show <policy-or-revision-id>
halo-forge checkpoint-policy validate --spec ./policy.yaml

halo-forge sweep create --spec ./repeat.yaml --checkpoint-policy <revision-id>
halo-forge sweep checkpoints <group-id>
halo-forge sweep analyze <group-id>
halo-forge sweep decide <group-id> --analysis <snapshot-id> --select <trial-id> \
  --rationale "Best supported quality/cost tradeoff"
halo-forge sweep report <group-id>
halo-forge sweep resume <group-id> --reason "Reviewed the paused guardrail"

halo-forge eval history --suite-revision <revision-id>
halo-forge eval drift --base <evaluation-id> --candidate <evaluation-id>
```

For Hugging Face trainers, adaptive groups declare `--max-steps`; RAFT and
the multimodal trainers declare `--cycles`. Halo Forge refuses a checkpoint
policy whose boundary unit does not match the trainer's verified resume
contract.

The existing training, sweep, evaluation, artifact, replay, and manual-path
commands remain compatible.
