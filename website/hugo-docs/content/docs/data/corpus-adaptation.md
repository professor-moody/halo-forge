---
title: "Adapt a Model to Documents"
description: "Extract a local document corpus, build an immutable version, pack exact token blocks, and run continued pretraining"
weight: 3
---

The `corpus-adaptation@1` scenario turns existing documents into a
model-independent corpus version and then renders a tokenizer-specific
continued-pretraining artifact.

```text
Documents → extraction → corpus review → immutable version
          → token packing → LoRA or full adaptation → explicit launch
```

Its stable scenario anchor is `own-data/corpus-adaptation`.

## Supported Sources

Halo Forge accepts plain text, Markdown, visible HTML, text-layer PDF, DOCX,
structured text columns, and pinned Hugging Face datasets. Core extraction
handles text, Markdown, HTML, and DOCX; the `corpus` extra installs the
reviewed PDF parser and the full optional extraction stack:

```bash
pip install "halo-forge[corpus]"
```

Encrypted, image-only, and empty PDFs are reported and quarantined. OCR,
crawling, EPUB, and remote object storage are not included.

Extraction preserves page, section, paragraph, or table provenance where
available. The checksummed result is stored under
`~/.halo-forge/corpus/bundles/<hash-prefix>/<content-hash>/`; the original source is
never modified.

## Preparation

The normal recipe preserves Markdown/code structure, normalizes encoding,
quarantines extraction failures, performs exact and fuzzy document
deduplication, and creates a seed-42 90/10 train/validation split grouped by
source reference. Page-level chunks from one PDF stay together, while
independent structured rows keep distinct row references.

The corpus preview shows extracted text beside the source reference and its
page or section spans. Exact accepted, quarantined, duplicate, document, and
character counts are published with the immutable version.

## Packing and Budget

The dataset version does not depend on a model. Rendering for trainer `cpt`
pins the tokenizer and produces non-overlapping, paragraph-aware token blocks.
The reviewed default maximum sequence length is 2,048 tokens; choose a shorter
supported value when the selected model or workstation requires it. Long
paragraphs are split at token boundaries and packing adds an EOS separator.

Halo Forge reports exact tokens, blocks, padding, utilization, effective batch
size, update steps, and block-to-document lineage. The training form displays
both target tokens and equivalent corpus passes; the selected budget mode
controls the launch. One corpus pass is the default.

Before the first optimizer update, a managed launch re-verifies the
content-addressed artifact and recomputes its packing-plan hash. Changed files
or a different model/tokenizer packing identity stop the launch.

## Training

Every launch requires **LoRA** or **Full** adaptation. Base and instruct causal
language models are equally eligible when the active PyTorch or MLX runtime
verifies them.

```bash
halo-forge data extract --path ./manuals
halo-forge data inspect --path ./manuals --scenario corpus-adaptation
halo-forge data build <dataset-id> --recommended-recipe
halo-forge data render <version-id> --trainer cpt \
  --model Qwen/Qwen2.5-1.5B
halo-forge cpt train --dataset-version <version-id> \
  --model Qwen/Qwen2.5-1.5B \
  --adaptation lora \
  --budget-mode passes --corpus-passes 1
```

Version publication, artifact rendering, and training are separate reviewed
actions. Halo Forge never launches training automatically.
