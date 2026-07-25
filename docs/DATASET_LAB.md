# Dataset Lab

Dataset Lab is Halo Forge's local data factory for text, vision-language, and
audio training data. It keeps raw sources unchanged, records fingerprints, and
publishes immutable prepared versions that can be launched directly from the
dashboard or CLI.

Install the optional local profiling and semantic-deduplication dependencies
with `pip install "halo-forge[data-lab]"`. Audio formats other than WAV use the
existing `audio` extra.

## Workflow

```text
register → inspect/map → build version → render trainer artifact → train
        → evaluate base/candidate → inspect deltas → mine failures → fork training
```

The default managed root is `~/.halo-forge/datasets`. Override it with
`HALOFORGE_DATASET_ROOT` or the Dataset Lab CLI `--root` option.

## Register sources

Local JSON, JSONL, CSV, Parquet, image-manifest, and audio-manifest sources:

```bash
halo-forge data add \
  --name support-sft \
  --path ./data/support.jsonl \
  --kind sft \
  --mapping '{"prompt":"question","response":"answer"}'
```

Pinned Hugging Face sources:

```bash
halo-forge data add \
  --name preference-pairs \
  --hf-id HuggingFaceH4/ultrafeedback_binarized \
  --split train \
  --revision <commit-or-tag> \
  --kind preference
```

Use `halo-forge data list`, `show`, and `preview` to inspect registered sources.
If a referenced local file or media asset changes, Dataset Lab refuses to build
or train from the stale fingerprint until the source is explicitly refreshed.

## Recipes

Recipes are ordered YAML or JSON step lists. They are deterministic when their
inputs and seed are unchanged.

```yaml
version: 1
schema: sft
seed: 42
steps:
  - type: map
    schema: sft
    fields:
      prompt: question
      response: answer
  - type: validate
    on_error: quarantine
  - type: dedup
    method: fuzzy
    threshold: 0.85
  - type: score
    method: heuristic
    threshold: 0.55
  - type: split
    method: random
    ratios:
      train: 0.9
      validation: 0.05
      test: 0.05
```

Build and inspect a version:

```bash
halo-forge data build <dataset-id> --recipe ./recipe.yaml
halo-forge data versions
halo-forge data export <version-id> --split train --output ./train.jsonl
```

Supported recipe operations include mapping, validation, safe filters,
exact/fuzzy/semantic text deduplication, media deduplication, quality scoring,
sampling, shuffling, limits, mixtures, split strategies, contamination checks,
curriculum labels, failure mining, and teacher-generated text or annotations.

## Media assets

VLM and audio records keep hashed references to their source assets by default.
Use the dashboard action or `halo-forge data materialize <version-id>` to copy
assets into managed storage when a self-contained version is needed. Dataset Lab
does not generate binary image or audio content; multimodal synthesis generates
captions, questions, answers, transcripts, labels, or preferences for existing
assets.

## Training handoff

Choose **Train** on a completed Dataset Lab version to open the training
configurator with `dataset_version_id` and `dataset_split` prefilled. The backend
resolves the immutable split, verifies trainer/schema compatibility and asset
availability, and records the version identity in the run relationship and
`replay.json`. If rendering is required, preflight returns immediately with a
persistent job, the Train screen shows its progress, and preflight resumes after
the verified content-addressed artifact is published.

Manual `--dataset` and `--data` launch paths remain supported.

## Closed-loop evaluation

Dataset Lab preserves stable record identity through recipes, mixtures, splits,
trainer rendering, evaluation, and reviewed failure mining. Rendered artifacts
contain only trainer-visible train and validation data; test and canary bindings
remain held out.

```bash
# Render exactly what a trainer will load.
halo-forge data render <version-id> --trainer sft --model <model-id>

# Bind immutable versions directly from any training command.
halo-forge sft train --model <model-id> --dataset-version <version-id>
halo-forge dpo train --model <model-id> \
  --dataset-binding train=<version-id>:train \
  --dataset-binding validation=<version-id>:validation

# Compare version membership, content, split movement, recipes, and sources.
halo-forge data compare <parent-version-id> <child-version-id>
```

Benchmark suites are immutable revisions. A persistent evaluation pins its
subject identity, adapter version, generation settings, metrics, and per-example
evidence.

```bash
halo-forge eval suite create \
  --name support-regression \
  --purpose development \
  --items ./suite-items.json \
  --primary-metric accuracy \
  --direction maximize

halo-forge eval run \
  --suite-revision <suite-revision-id> \
  --subject <model-id> \
  --subject-revision <model-revision> \
  --wait

halo-forge eval compare <base-evaluation-id> <candidate-evaluation-id>
halo-forge data mine \
  --base <base-evaluation-id> \
  --candidate <candidate-evaluation-id> \
  --selector regression
```

The mining command previews its selection by default. Add `--dataset` and
`--parent-version` to explicitly build an immutable child version after review.
The dashboard exposes the same loop from Dataset Version, Train, Run, and Eval.

## Jobs

Dataset Lab keeps its persistent serial job manager for build, synthesis,
profiling, trainer-artifact rendering, failure mining, and materialization.
Run-group training and its dependent checkpoint evaluations use the durable
experiment queue, where dependency-ready work is ordered by priority and then
FIFO. A retained serving lease pauses incompatible accelerator work in that
queue.

Dataset-specific commands remain available for familiar Dataset Lab workflows:

```bash
halo-forge data jobs
halo-forge data jobs --job <job-id>
halo-forge data jobs --cancel <job-id>
halo-forge data jobs --retry <job-id>
```

The experiment queue shows run-group training and dependent evaluation work:

```bash
halo-forge jobs list
halo-forge jobs show <work-item-id>
halo-forge jobs cancel <work-item-id>
halo-forge jobs retry <work-item-id>
halo-forge jobs worker
```

Experiment work items persist their dependencies, progress, logs, process
identity, and attempt history. Interrupted work is retained as retryable
instead of being reported as completed, and a live matching child can be
adopted after restart. Dataset build retries continue to resume from the last
persisted recipe boundary through Dataset Lab's job manager. See
[Reproducible Experiment Operations](EXPERIMENT_OPERATIONS.md) for queue and
run-group behavior.

Exports support canonical JSONL, CSV, and Parquet (`--format`); JSONL remains
the default training and replay interchange.
