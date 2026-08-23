# Artifact Studio

Artifact Studio is Halo Forge's local model-artifact lifecycle. It connects the
end of training to a verified model that can be compared, promoted, served, and
exported without manually moving checkpoint files:

```text
run or checkpoint -> merge/convert/quantize -> qualify -> promote -> serve/export
```

The dashboard presents this workflow under **Models**. **Activity** shows the
workstation owner, queue, progress, blockers, telemetry, retry history, and disk
forecast for every managed operation.

## Artifact identity

Halo Forge separates bytes from their history:

- A **blob** is one unique content hash and immutable manifest.
- A **location** is a referenced run path, managed library copy, or trash
  location containing those bytes.
- An **occurrence** explains where the artifact came from: a run, checkpoint,
  merge, conversion, quantization, or export.
- An **edge** records ordered parentage. Multi-adapter operations therefore keep
  the input ordering and do not collapse lineage when inputs have identical
  bytes.

Run outputs are referenced by default. **Adopt into library** verifies and
copies the content to
`~/.halo-forge/artifacts/blobs/<content-hash>/` while retaining the original
occurrence and path.

## Guided workflow

1. Open **Models -> Trained Artifacts** and choose a completed run artifact.
2. Use the structured operation form to bake an adapter, combine adapters, or
   convert the selected artifact.
3. Follow the operation in **Activity**. Every retry gets an isolated attempt
   directory; a completed identical operation is reused.
4. Choose a qualification profile and compare the candidate with its parent.
5. Inspect the decision, metric deltas, and quality/speed/memory/size tradeoff.
6. Explicitly promote the artifact to `candidate` or `approved`.
7. Serve it in **Models -> Serve & Test**, or export a portable local bundle.

The contextual inspector exposes hashes, manifests, provenance, logs, and raw
specifications without requiring those values in normal forms.

## Transformations

Artifact Studio delegates to Halo Forge's verified merge and conversion
engines. Supported operations are checked before work is queued:

- bake one PEFT adapter into its base model;
- combine adapters with linear, TIES, DARE, or magnitude-pruning methods;
- convert to Hugging Face, MLX, or GGUF when the installed engine supports the
  source/target combination; and
- produce q4, q8, fp16, bf16, or fp32 variants when supported.

ONNX requests fail truthfully when no verified local engine is available. q4
and q8 conversions are **post-training quantization**, not QAT. Halo Forge does
not claim real QAT without a verified quantization-aware training path.

Outputs are built in same-filesystem staging directories, verified for content
identity and format completeness, and exposed atomically only after validation.

## Qualification and promotion

A qualification-profile revision pins:

- one development quality-suite revision;
- one operational performance-suite revision;
- an optional final holdout revision;
- metric directions, thresholds, and permitted quality deltas; and
- the target backend and generation settings.

Operational performance uses two warmups and five measured repetitions at
concurrency one with a fixed seed by default. Halo Forge records only metrics
the runtime can actually measure; unavailable GPU or token-timing values remain
empty instead of being estimated.

Parent and candidate comparisons require the same immutable profile revision.
Qualification produces `pass`, `warn`, or `fail` with exact reasons. Promotion
is never automatic:

- `candidate` requires the configured development and operational gates;
- `approved` additionally requires holdout confirmation when the profile has a
  holdout; and
- an override requires a visible, retained operator note.

Operational and holdout suites cannot be used for failure mining or to guide
training.

## Serving and Serve & Test

Managed serving retains the workstation accelerator/memory lease for the life
of the server. The lease stores the real process identity and receives
heartbeats so a crashed or reused PID cannot block the workstation forever.

Serve & Test supports named sessions, streaming where the selected backend
supports it, generation seeds and metadata, and sequential base/candidate
turns. Reviewed turns can become a new benchmark-suite revision or Dataset Lab
source draft. They do not create a dataset or start training automatically.

## Portable export

Local export bundles include model files, tokenizer/configuration, checksums,
lineage, qualification evidence, replay and dataset identities, license
metadata, and a generated or supplied model card. Remote registry and
Hugging Face publishing are intentionally outside this phase.

## Storage and cleanup

`halo-forge storage status` reports unique content, locations, managed size,
referenced size, trash, and disk capacity. Cleanup always follows a review
flow:

1. Preview a cleanup plan.
2. Inspect every proposed item and protection reason.
3. Apply the exact plan with a review note.
4. Restore during the seven-day trash window, or purge only expired trash.

Active, pinned, promoted, serving, evaluation-referenced, and lineage-required
artifacts are protected.

## CLI examples

```bash
halo-forge artifact import ./run/final_model --kind final --format hf
halo-forge artifact list
halo-forge artifact lineage <occurrence-id>
halo-forge artifact verify <occurrence-id>

halo-forge artifact merge <adapter-a> <adapter-b> \
  --base-model Qwen/Qwen2.5-7B --mode combine --method dare_ties
halo-forge artifact convert <occurrence-id> --format gguf --quantization q4
halo-forge artifact qualify <occurrence-id> --profile <profile-revision-id>
halo-forge artifact compare <parent-id> <candidate-id>
halo-forge artifact promote <occurrence-id> candidate

halo-forge artifact serve <occurrence-id> --backend local
halo-forge artifact export <occurrence-id> ./portable-model
halo-forge storage status
halo-forge storage cleanup
halo-forge storage cleanup --apply <plan-id> --review-note "Reviewed candidates"
```

Merge, conversion, qualification, serving, export, and approved cleanup return
both a domain operation ID and a durable work-item ID. The dashboard
automatically supervises one worker; headless operators can continue using
`halo-forge jobs worker`.
