---
title: "Review Studio"
description: "Turn development evidence and unlabeled multimodal records into reviewed, immutable training data."
weight: 22
---

Review Studio closes the reviewed data loop:

```text
development evidence or unlabeled data -> acquisition proposal
-> approved review queue -> human labels and adjudication
-> approved immutable label set -> reviewed Dataset Lab child version
-> explicit training launch or repeat
```

Halo Forge proposes deterministic batches, but a person controls every state
change. Queue creation, label-set publication, dataset building, and training
launch are separate actions. Completing an acquisition job does not open a
queue or change training data.

Open **Data -> Review Queues** at `/datasets/review`. A focused queue opens at
`/datasets/review/<queue-id>`. Contextual **Create review proposal** actions are
available from development evaluation comparisons, run samples, Dataset Lab
records, and Models -> Serve & Test.

## Deterministic acquisition

Build a proposal from:

- a development evaluation or compatible base/candidate comparison;
- run samples;
- a Dataset Lab version and split;
- a Playground session; or
- imported JSONL.

Strategies include explicit selection, candidate failures, regressions,
improvements, verifier disagreement, low score, low margin, coverage gaps,
diversity, and seeded random sampling. Ordered quota strata are deterministic
and deduplicate by stable record identity. Halo Forge records source hashes,
strategy versions, filters, quotas, seed, embedding revision when required, and
the exact presentation order.

Low-margin selection requires compatible scores. Diversity requires a pinned,
available embedding revision. If vectors are absent, Halo Forge runs a
compatible pinned text, image, or audio embedding model as durable heavy work;
use `modality:model@revision`. The Lab records the model/runtime provenance and
does not invent either signal.

### Protected evidence

Operational, holdout, test, and canary evidence cannot enter an acquisition
batch. Development evidence can be reviewed, but its exposure identity follows
the label set and every descendant dataset version. Those descendants cannot be
reported as untouched evidence from the same suite.

## What you can review

- accept/reject, category, multi-label, and scalar judgments;
- SFT or chat corrections;
- pairwise preferences and rankings;
- structured tool calls and results;
- VLM responses over existing images; and
- audio transcripts and labels over existing audio.

Queues support fast one-pass review or a blinded second pass with explicit
adjudication. Pass two hides pass-one decisions and deterministically
counterbalances candidate order. Labels, corrections, deferrals, exclusions,
flags, notes, reveals, retractions, and adjudications are append-only events.
Idempotency keys and active-event checks prevent a stale browser from
overwriting a newer decision.

Optional model suggestions are generated only when requested. They remain
separate from human decisions, stay hidden during a blind pass until explicitly
revealed, and never count as a human label. Suggestion provenance records the
model revision, endpoint type, prompt/template hash, parameters, verifier trace,
scores, and runtime identity without storing credentials.

Playground preference review binds a persisted user prompt to distinct base and
candidate assistant responses through guided pickers. Missing alternatives are
rejected rather than replaced with synthetic choices.

The focused workspace supports keyboard review, autosaved drafts, reconnect
recovery, image zoom and metadata, accessible audio and transcript controls,
structured tool-trace correction, preference ranking, conflict resolution, and
a provenance inspector. The mobile layout uses a full-screen item and persistent
action bar; no review control depends on hover.

## Publish and build

Publication blocks while an item is pending, flagged, or conflicting. An
unresolved item must be explicitly excluded with a retained reason; **Skip**
only defers it. A published label-set revision contains checksummed canonical
JSONL, lineage, statistics, acquisition/schema/suggestion provenance, active
review-event identities, and exposure records. Later corrections create
unpublished changes; republishing creates a new revision without changing the
old one.

Label publication is durable Activity work with cancellation, retry, recovery,
and atomic checksum publication. The revision becomes selectable only after the
worker verifies and catalogs it.

Preview the exact Dataset Lab change before building:

- `filter` keeps accepted source records and removes rejected records;
- `replace_by_record_id` applies corrections while preserving lineage;
- `append` adds new annotations or preference pairs; and
- `annotate` adds metadata labels without replacing source content.

The preview reports added, removed, replaced, quarantined, and split-affected
records and reuses Dataset Lab validation, media hashing, contamination checks,
and optional asset materialization. Building publishes a new immutable dataset
version. **Train single** and **Start repeat** remain separate explicit actions.

## Command line

```bash
halo-forge review capabilities
halo-forge review schema list
halo-forge review acquire create --spec ./acquisition.yaml
halo-forge review acquire show <batch-id> --candidates
halo-forge review queue create --batch <batch-id> --schema <revision-id>
halo-forge review items <queue-id>
halo-forge review item <item-id>
halo-forge review submit <item-id> --label '{"accepted": true}'
halo-forge review stats <queue-id>
halo-forge review label-set publish --queue <queue-id>
halo-forge review label-set verify <revision-id>
halo-forge review label-set preview <revision-id> --dataset <dataset-id>
halo-forge review label-set build-dataset <revision-id> --dataset <dataset-id>
```

Every normal command supports `--database` and `--json`; use `--root` to choose
a Review Studio storage root. Simple acquisition sources and strategies have
structured flags. Multiple ordered sources or strata use `--spec`. Existing
`halo-forge data mine`, evaluation-mining APIs, and Playground dataset or suite
outputs remain compatible.

## HTTP resources

The dashboard uses the same `/api/public` service as the CLI. Principal
resources are:

- `/review-capabilities` and `/spec-descriptors/{kind}`;
- `/annotation-schemas` and immutable schema revisions;
- `/acquisition-batches` and paginated candidate previews;
- `/review-queues`, queue statistics, and paginated items;
- `/review-items/{id}/events` and `/review-items/{id}/suggestions`; and
- `/label-sets`, immutable label-set revisions, verification, Dataset Lab
  preview, and dataset build.

Acquisition, suggestion, label publication, and dataset-build responses include
a durable `work_item_id` for the Activity Center. Lists use bounded
`{items,total,limit,offset}` responses.
