# Corpus Adaptation

Halo Forge can continue causal language-model training on a local document
collection without converting the source by hand. The managed workflow is:

```text
Select documents → extract visible text → inspect provenance and failures
                 → prepare immutable corpus version → render token blocks
                 → choose LoRA or full adaptation → launch
```

Corpus adaptation is the `corpus-adaptation@1` guided scenario, documented by
the stable registry anchor `own-data/corpus-adaptation`. It uses canonical
schema `corpus` and trainer mode `cpt`.

## Supported documents

The first release accepts:

- plain text and Markdown;
- visible HTML, excluding scripts, styles, and hidden nodes;
- PDFs that contain a readable text layer;
- DOCX paragraphs, headings, and tables; and
- existing JSON, JSONL, CSV, TSV, Parquet, or pinned Hugging Face text
  columns.

Encrypted PDFs, image-only PDFs, and empty documents are reported and
quarantined. OCR, crawling, EPUB, remote object storage, and binary media
generation are not part of this phase.

Extraction records page, section, paragraph, and table provenance where the
source format exposes it. Checksummed extraction bundles are stored under:

```text
~/.halo-forge/corpus/bundles/<hash-prefix>/<content-hash>/
```

The source remains unchanged. A changed source creates a new extraction and
source revision.

## Canonical corpus record

```text
CorpusDocument {
  document_id,
  document_hash,
  text,
  optional title,
  source_ref,
  source_spans,
  optional timestamp,
  metadata
}
```

Document and content hashes are deterministic. Dataset splitting groups by
`source_ref`, so page-level chunks from one PDF remain together while
independent structured rows retain distinct source references.

## Reviewed preparation defaults

The default recipe:

1. preserves Markdown and code structure;
2. normalizes encoding and surrounding whitespace without collapsing
   paragraphs;
3. quarantines empty or failed extractions;
4. removes exact and near-duplicate documents;
5. creates a deterministic seed-42 90/10 train/validation split grouped by
   document; and
6. reports contamination between protected splits.

A test split is optional in Advanced and is never given to the trainer.

## Tokenization and packing

Dataset versions remain model-independent. Rendering a CPT training artifact
pins the model, tokenizer revision, tokenizer hash, sequence length, separator,
and packing policy.

The normal default is paragraph-aware, non-overlapping packing with an EOS
separator and a reviewed maximum sequence length of 2,048 tokens. Choose a
shorter supported limit when the selected model or workstation requires it. A
paragraph longer than the limit is split at token boundaries. The artifact
reports exact train and validation token counts,
blocks, padding, utilization, effective batch size, and estimated update
steps, plus block-to-document lineage.

Managed launches carry the artifact, model, tokenizer, and packing identities
into the trainer process. The trainer verifies the bundle again and recomputes
the packing plan before optimization; changed files or a different plan stop
the launch instead of silently training on drifted input.

The Train screen always shows both:

- target training tokens; and
- equivalent corpus passes.

One selected budget mode controls the launch. The default is one corpus pass.

## Adaptation choice

Every launch requires an explicit choice:

- **LoRA** updates adapters and is normally the lower-memory option.
- **Full** updates all model weights and requires substantially more memory
  and storage.

Base and instruct causal language models are treated equally when the active
runtime verifies them. Halo Forge does not apply a special warning based only
on that naming convention.

PyTorch supports CUDA, ROCm, MPS, and CPU execution. Native MLX is supported on
verified Apple Silicon runtimes. Hugging Face CPT supports step-level resume;
MLX remains final/full-trial unless its runtime advertises verified token-block
resume.

## CLI

```bash
halo-forge data scenarios advise \
  --goal "Adapt a model to our Markdown and PDF manuals" \
  --source-layout pdf

halo-forge data extract --path ./manuals
halo-forge data inspect --path ./manuals --scenario corpus-adaptation
halo-forge data corpus-profile <version-id>
halo-forge data build <dataset-id> --recommended-recipe
halo-forge data render <version-id> --trainer cpt \
  --model Qwen/Qwen2.5-1.5B

halo-forge cpt train \
  --dataset-version <version-id> \
  --model Qwen/Qwen2.5-1.5B \
  --adaptation lora \
  --budget-mode passes \
  --corpus-passes 1 \
  --max-seq-length 2048 \
  --packing paragraph_eos_non_overlap_v1
```

Extraction, version publication, artifact rendering, and training remain
separate explicit operations. Nothing starts training automatically.
