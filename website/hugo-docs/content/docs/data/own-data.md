---
title: "Use Your Own Data"
description: "Guided local file/folder import, scenario mapping, validation, immutable versions, and training handoff"
weight: 2
---

The managed own-data path is:

```text
Choose scenario → select source → inspect → confirm mapping
                → preview quality and splits → build immutable version
                → train (optional proof for labeled scenarios)
```

Halo Forge references and fingerprints the source rather than rewriting it.
Derived versions are published atomically under `~/.halo-forge/datasets/` with
their recipe, lineage, statistics, rejected rows, and source hashes.

## Source Containers

Local sources may be `.json`, `.jsonl`, `.jl`, `.csv`, `.tsv`, or `.parquet`
files. Corpus adaptation additionally accepts plain text, Markdown, visible
HTML, text-layer PDF, and DOCX. A media folder may use an explicit manifest, a sidecar table keyed by
relative filename, or same-basename `.txt` captions/transcripts. Pinned
Hugging Face dataset revisions remain supported.

Choose the scenario that matches the information in each record:

| Scenario | Minimum shape | Trainer family |
|---|---|---|
| SFT | prompt/instruction + response/completion | SFT |
| Chat | ordered role/content messages | chat SFT |
| Preference | prompt + chosen + rejected | DPO, ORPO, RM |
| Reasoning/RLVR | prompt + optional reference answer/metadata | RAFT, GRPO, reasoning |
| Tool use | messages + tools + expected calls/results | agentic/tool |
| VLM | image reference + prompt + response/ground truth | VLM |
| Audio | audio reference + task + transcript/label | audio |
| Corpus | extracted document text + source provenance | CPT |

## Scenario Catalog

The versioned registry exposes these available revision-1 scenarios. The
registry anchor is stable across desktop, browser, CLI, and API surfaces.

| Scenario ID | Registry anchor | Trainer modes |
|---|---|---|
| `instruction-sft` | `own-data/instruction-sft` | SFT |
| `chat-sft` | `own-data/chat-sft` | SFT |
| `preference-pairs` | `own-data/preference-pairs` | DPO, ORPO, RM |
| `prompt-reward` | `own-data/prompt-reward` | RAFT, GRPO |
| `reasoning-sft` | `own-data/reasoning-sft` | SFT |
| `tool-agentic` | `own-data/tool-agentic` | SFT, agentic |
| `vlm-captioning` | `own-data/vlm-captioning` | VLM |
| `vlm-qa` | `own-data/vlm-qa` | VLM |
| `audio-asr` | `own-data/audio-asr` | audio |
| `corpus-adaptation` | `own-data/corpus-adaptation` | CPT |
| `text-classification` | `own-data/text-classification` | classify |
| `text-multilabel` | `own-data/text-multilabel` | classify |
| `embedding-pairs` | `own-data/embedding-pairs` | embed |
| `reranking` | `own-data/reranking` | rerank |
| `image-classification` | `own-data/image-classification` | classify |
| `audio-classification` | `own-data/audio-classification` | classify |
| `audio-tts` | `own-data/audio-tts` | unavailable |

`audio-tts` remains visible but unavailable because Halo Forge has no verified
data-to-weight-update trainer contract for it. Specialized task-model
scenarios use the verified PyTorch path; unsupported runtimes remain hidden
with an exact reason.

**Help me decide** uses the backend scenario advisor. It explains the goal,
field, modality, and source-layout evidence behind each suggestion and still
requires confirmation. **Try a working example** opens a gallery covering
every verified scenario, including the document corpus path.

## Checked JSONL Shapes

These examples are parsed and matched to the versioned scenario shapes by Halo
Forge's release-interface check. This verifies structure, not the existence or
decodability of referenced media.

```jsonl
{"prompt":"Summarize the incident report.","response":"The service recovered after the cache was rebuilt."}
```

```jsonl
{"prompt":"Explain the failure.","chosen":"The upstream stopped responding and the request timed out.","rejected":"It broke for no reason."}
```

```jsonl
{"image":"assets/panel-001.png","prompt":"Which warning light is active?","response":"The amber temperature warning is active."}
```

```jsonl
{"audio":"assets/call-001.wav","task":"asr","transcript":"Please restart the service after the backup completes."}
```

Corpus adaptation starts with a normal document:

```markdown
# Service recovery

Restore the last verified snapshot, confirm integrity, and restart the worker.
```

Media paths refer to the workstation running Halo Forge. A remote browser's
local filesystem is not the host filesystem. These standalone snippets require
the referenced files. **Try a working example** and the CLI scenario-template
command include tiny checksummed image/audio companions and are inspectable as
delivered. Supply your own assets for real work; Halo Forge resolves, hashes,
and decodes them. Assets are referenced by default; **Materialize assets** makes
a managed copy when needed.

## Dashboard

1. Choose **Train on your data** from Overview, Data, or Train.
2. Choose a file, folder, or pinned Hugging Face revision.
3. Describe the goal or choose a scenario and inspect the advisor's reasons.
4. Confirm the visual mapping and semantic chat, preference, tool, image,
   audio, or corpus preview.
5. Review rejected examples, duplicates, distributions, split balance,
   media/extraction failures, and direct remediation actions.
6. Build the version and monitor it in **Activity**.
7. Confirm the model and method, then explicitly launch. Labeled scenarios may
   use the optional proof flow. Corpus adaptation requires an explicit
   LoRA/full choice and token/pass budget.

RAFT and GRPO proof runs require an immutable candidate- or approved-qualified
verifier revision. The guided Train step shows compatible choices by name and
links to **Evaluate → Verifiers** when one must be calibrated first. Raw
verifier configuration remains an Advanced, visibly unqualified path.

Version publication reports exact rows, quarantines, duplicates, splits,
assets, and characters. Token counts are estimates until a training artifact
pins the model, tokenizer revision, and chat template; only then are they
reported as exact.

The desktop shell offers a native file/folder chooser. Local and remote browser
surfaces retain upload or explicit workstation-path flows. They all produce the
same Dataset Lab source and version records.

## CLI

```bash
halo-forge data inspect --path ./support-sft.jsonl --scenario instruction-sft
halo-forge data add --name support-sft --path ./support-sft.jsonl \
  --scenario instruction-sft \
  --map prompt=instruction --map response=answer \
  --accept-recommended
halo-forge data show <dataset-id>
halo-forge data build <dataset-id> --recommended-recipe
halo-forge data versions <dataset-id>
halo-forge data render <version-id> --trainer sft
halo-forge sft train --dataset-version <version-id>
```

Document corpus example:

```bash
halo-forge data scenarios advise --goal "Adapt to our manuals" --source-layout pdf
halo-forge data extract --path ./manuals
halo-forge data inspect --path ./manuals --scenario corpus-adaptation
halo-forge data corpus-profile <version-id>
halo-forge data render <version-id> --trainer cpt --model Qwen/Qwen2.5-1.5B
halo-forge cpt train --dataset-version <version-id> \
  --model Qwen/Qwen2.5-1.5B --adaptation lora \
  --budget-mode passes --corpus-passes 1
```

See [Adapt a Model to Documents](/docs/data/corpus-adaptation/) for extraction
and packing details.

Use `halo-forge data preview` while developing a mapping. A changed or missing
source requires an explicit refresh and a new revision; it cannot silently
alter an existing version.
