---
title: "Artifact Studio"
description: "Transform, qualify, promote, serve, export, and safely retain local model artifacts."
weight: 65
---

Artifact Studio connects a completed run or checkpoint to a verified local
model you can use:

```text
run/checkpoint -> merge or convert -> qualify -> promote -> serve/export
```

Open **Models** in the dashboard. Trained artifacts, downloaded models,
serving, and test sessions now live in one workspace. Open **Activity** at any
time to see the current resource owner, queue, progress, blockers, telemetry,
logs, retry history, and disk forecast.

## Why blobs and occurrences are separate

Halo Forge stores one immutable **blob** for each unique content hash. An
artifact **occurrence** records where those bytes appeared: a run checkpoint,
final model, merge, conversion, quantized variant, or export. Locations track
the original run path and any managed library copy.

This means two runs can produce identical bytes without losing either run's
history. Ordered lineage also preserves every parent in multi-adapter merges.

Run artifacts are referenced in place by default. **Adopt into library** makes
a verified managed copy under `~/.halo-forge/artifacts/blobs/` without changing
the run output.

## Transform an artifact

Select an artifact, then choose a structured operation:

- **Bake** folds one adapter into its recorded base.
- **Combine** supports linear, TIES, DARE, and magnitude-pruning methods.
- **Convert** targets Hugging Face, MLX, or GGUF when that combination is
  supported by the installed runtime.
- **Quantize** creates a supported q4 or q8 post-training variant.

q4/q8 are post-training quantization, not quantization-aware training. ONNX is
rejected when no verified local converter is installed. Outputs stay hidden in
a same-filesystem staging directory until hashes and format checks pass, then
publish atomically. An identical completed operation is reused.

## Qualify and promote

A qualification profile pins a development quality suite, an operational
performance suite, an optional final holdout suite, metric directions,
thresholds, permitted quality deltas, backend, and generation settings.

Operational measurement defaults to two warmups and five measured repetitions
at concurrency one with a fixed seed. Missing device or token-timing metrics
remain unavailable; Halo Forge never fills them with estimates.

Compare only artifacts evaluated under the same immutable profile revision.
The result is `pass`, `warn`, or `fail` with reasons and a quality, speed,
memory, and size view.

Promotion is explicit:

- `candidate` requires the development and operational gates;
- `approved` also requires holdout confirmation when configured; and
- overriding a gate requires a retained note.

Operational and holdout suites cannot drive training or failure mining.

## Serve, test, and export

Use **Models -> Serve & Test** to select an artifact, keep named sessions,
record seeds and generation settings, and compare a base and candidate
sequentially. Managed serving holds the workstation resource lease for the
server process lifetime.

A portable local export includes model files, tokenizer/configuration,
checksums, lineage, qualification evidence, replay and dataset identities,
license metadata, and a model card. Remote registry publishing is not included
in this phase.

## Safe cleanup

Open **Models -> Cached Models** or run `halo-forge storage status` to inspect
managed, referenced, temporary, and trash storage.

Cleanup is always previewed and reviewed. Active, pinned, promoted, serving,
evaluation-referenced, and lineage-required artifacts are protected. Applied
plans move content to trash for seven days before it can be permanently purged.

## CLI quick reference

```bash
halo-forge artifact list
halo-forge artifact show <artifact>
halo-forge artifact lineage <artifact>
halo-forge artifact verify <artifact>

halo-forge artifact merge <adapter-a> <adapter-b> \
  --base-model Qwen/Qwen2.5-7B --method dare_ties
halo-forge artifact convert <artifact> --format gguf --quantization q4
halo-forge artifact qualify <artifact> --profile <profile-revision>
halo-forge artifact promote <artifact> candidate
halo-forge artifact serve <artifact>
halo-forge artifact export <artifact> ./portable-model

halo-forge storage status
halo-forge storage cleanup
halo-forge storage cleanup --apply <plan-id> --review-note "Reviewed candidates"
```

Large operations return a domain operation ID and a durable work-item ID. The
dashboard starts one supervised worker automatically. For headless use, run
`halo-forge jobs worker`.
