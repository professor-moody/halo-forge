# Human Feedback and Active Data Studio

Review Studio turns development evidence and unlabeled records into reviewed,
versioned training data without automatically changing a dataset or launching a
trainer.

```text
development evidence or unlabeled data
  -> deterministic acquisition batch
  -> local review and optional second pass
  -> immutable label-set revision
  -> reviewed Dataset Lab child version
  -> explicit training launch or repeat
```

Review Studio is part of **Data** in the dashboard. It is designed for one local
researcher and supports text, chat, preference, tool-use, VLM, and audio records.
Image and audio files remain hashed references until Dataset Lab materializes
them.

## Human gates and protected evidence

Halo Forge can propose and order candidates, but it does not silently advance
the workflow. The operator separately approves queue creation, publishes a
label-set revision, builds a child dataset version, and launches Train or a
repeat. A completed acquisition job is still only a proposal.

Operational, holdout, test, and canary evidence is never eligible for review
acquisition. Development evidence is eligible, but its exposure identity is
recorded and propagated so derived data cannot later be presented as untouched
evidence from the same suite.

## Acquisition batches

An acquisition batch is an immutable proposal. It records its source identities,
source hashes, strategy and version, ordered quotas, filters, seed, record-ID
deduplication, and the exact selected order. Supported sources include:

- evaluations and evaluation comparisons;
- run samples;
- Dataset Lab versions and splits;
- Playground sessions; and
- imported JSONL.

Strategies include explicit selection, candidate failures, regressions,
improvements, verifier disagreement, low score or margin, coverage gaps,
diversity, and seeded random sampling. Strategies only use evidence that is
actually available. Diversity pins its embedding-model revision and, when the
source has no stored vectors, runs the compatible text, image, or audio model as
durable heavy work. Generated embeddings retain the exact model revision and
runtime provenance; the CLI/UI use `modality:model@revision` (for example,
`image:org/clip@commit`). Low-margin selection is unavailable when the source
has no compatible scores.

Holdout, operational, test, and canary evidence cannot be acquired. Development
exposure is retained in the exposure ledger and follows every derived label set
and dataset version.

Creating a batch does not create a review queue. The operator previews the
selection, chooses an annotation schema and review policy, and explicitly opens
the queue.

## Review policies

Annotation-schema revisions define the task, validation rules, presentation,
and output adapter. Review Studio supports:

- binary, categorical, multi-label, and scalar labels;
- text and chat corrections;
- structured tool-call and result corrections;
- pairwise preferences and rankings;
- VLM annotations over existing images; and
- audio transcripts or labels over existing audio.

Queues can use one pass or a blinded two-pass workflow. The second pass begins
after the first pass is resolved, hides the first decision, and deterministically
counterbalances preference order. Conflicts require an explicit adjudication.

Labels, corrections, deferrals, exclusions, flags, notes, suggestion reveals,
retractions, and adjudications are append-only events. Corrections reference the
event they supersede and retain a reason. Submitted events use idempotency and
optimistic concurrency so a stale browser cannot overwrite a newer decision.

Optional model suggestions are generated only when requested. Their model,
endpoint type, prompt hash, sampling settings, verifier evidence, and runtime are
recorded separately. Suggestions remain hidden during a blind pass until the
reviewer explicitly reveals them, and never become human labels automatically.

## Label sets and Dataset Lab

Publishing creates an immutable, checksummed label-set revision with canonical
JSONL, lineage JSONL, statistics, acquisition and annotation identities, active
review-event IDs, suggestion provenance, and exposure records. Pending, flagged,
or conflicting items block publication; an omitted item needs an explicit
exclusion reason.

Publication itself is durable scheduler work. The dashboard shows its Activity
item, cancellation/retry state, and checksum publication progress, then opens
the revision only after the atomic result is cataloged.

After label-set publication, Dataset Lab previews one of four reviewed build
modes before a child dataset version is built:

- `filter` keeps accepted source records and removes rejected records;
- `replace_by_record_id` replaces corrected records while retaining lineage;
- `append` adds newly labeled examples or preferences; and
- `annotate` attaches reviewed metadata without changing the source content.

The preview shows added, removed, replaced, quarantined, and split-affected
records and runs the normal contamination and media-reference checks. The
operator then explicitly builds a new immutable version. Training remains a
separate action.

## Dashboard workflow

Open **Data -> Review Queues**. Contextual **Create review proposal** actions are
also available from evaluation comparisons, run samples, dataset records, and
Models -> Serve & Test.

Pairwise Playground review uses an explicit persisted prompt, base response,
and candidate response. Halo Forge never fabricates missing alternatives; the
guided picker records both response identities and their generation provenance.

The queue browser is `/datasets/review`; a focused queue workspace is
`/datasets/review/<queue-id>`. The dashboard uses compatibility-aware pickers,
so the normal workflow does not require copying internal IDs.

The queue workspace keeps the current sample dominant, with the queue and
progress on one side and provenance on the other. Submitted decisions save
atomically; unfinished item drafts resume after navigation or reconnect. The
mobile view becomes a full-screen sample with a persistent action bar.

Common Dataset recipe steps and benchmark-suite settings use structured forms.
The existing YAML/JSON representation remains available under **Advanced** and
round-trips without loss.

## CLI overview

```bash
halo-forge review schema list
halo-forge review acquire create --spec ./acquisition.yaml
halo-forge review queue create --batch <batch-id> --schema <revision-id>
halo-forge review items <queue-id>
halo-forge review item <item-id>
halo-forge review submit <item-id> --label '{"accepted": true}'
halo-forge review stats <queue-id>
halo-forge review label-set publish --queue <queue-id>
halo-forge review label-set preview <revision-id>
halo-forge review label-set build-dataset <revision-id> --dataset <dataset-id>
```

Use `--json` for automation. Multiple ordered acquisition sources or strategies
use `--spec`; the dashboard does not require raw JSON or internal-ID entry.

Existing `halo-forge data mine`, evaluation-mining endpoints, and Playground
benchmark/dataset review targets remain compatible. They share normalization and
selection primitives but do not fabricate historical human-review events.

## Public API

The dashboard and CLI share the `/api/public` resources for review
capabilities, annotation schemas and immutable revisions, acquisition batches
and candidates, queues and statistics, item events and suggestions, label sets,
verification, Dataset Lab preview/build, and structured spec descriptors.
Acquisition, suggestion, label publication, and dataset-build responses include
a durable `work_item_id`; list responses retain the bounded
`{items,total,limit,offset}` shape.
