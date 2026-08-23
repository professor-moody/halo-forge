# Managed Own-Data Workflow

Halo Forge turns workstation data into an immutable Dataset Lab version before
training. The managed path is:

```text
Choose scenario → select source → inspect → confirm mapping
                → preview quality and splits → build immutable version
                → train (optional proof for labeled scenarios)
```

The source is never rewritten. Halo Forge fingerprints it, records rejected or
quarantined rows, and publishes derived data atomically under
`~/.halo-forge/datasets/`.

## Supported source containers

The local source reader accepts `.json`, `.jsonl`, `.jl`, `.csv`, `.tsv`, and
`.parquet` files. Corpus adaptation also accepts `.txt`, Markdown, visible
HTML, text-layer PDF, and DOCX. A folder source may use an explicit manifest, a sidecar table
keyed by relative filename, or same-basename `.txt` captions/transcripts beside
image or audio assets. Pinned Hugging Face dataset revisions remain supported.

The container format does not decide the training task. In the guided flow,
choose a scenario and map source fields to one of these canonical targets:

| Canonical target | Required information | Typical trainer |
|---|---|---|
| SFT | prompt/instruction and response/completion | SFT |
| Chat | ordered role/content messages | chat SFT |
| Preference | prompt, chosen, rejected | DPO, ORPO, RM |
| Reasoning/RLVR | prompt plus optional reference answer/metadata | RAFT, GRPO, reasoning |
| Tool use | messages, tool definitions, expected calls/results | agentic/tool SFT |
| VLM | image reference, prompt, response or ground truth | VLM |
| Audio | audio reference, task, transcript or label | audio |
| Corpus | extracted document text and source provenance | CPT |

## Versioned scenario catalog

The guided workflow uses these stable scenario IDs and revision-1 documentation
anchors. Halo Forge returns unavailable scenarios too, with an exact reason,
instead of silently offering an unsupported trainer.

| Scenario ID | Registry anchor | Canonical target | Trainer modes |
|---|---|---|---|
| `instruction-sft` | `own-data/instruction-sft` | SFT | SFT |
| `chat-sft` | `own-data/chat-sft` | Chat | SFT |
| `preference-pairs` | `own-data/preference-pairs` | Preference | DPO, ORPO, RM |
| `prompt-reward` | `own-data/prompt-reward` | Prompt | RAFT, GRPO |
| `reasoning-sft` | `own-data/reasoning-sft` | SFT | SFT |
| `tool-agentic` | `own-data/tool-agentic` | Tool | SFT, agentic |
| `vlm-captioning` | `own-data/vlm-captioning` | VLM | VLM |
| `vlm-qa` | `own-data/vlm-qa` | VLM | VLM |
| `audio-asr` | `own-data/audio-asr` | Audio | audio |
| `corpus-adaptation` | `own-data/corpus-adaptation` | Corpus | CPT |
| `text-classification` | `own-data/text-classification` | Classification | classify |
| `text-multilabel` | `own-data/text-multilabel` | Classification | classify |
| `embedding-pairs` | `own-data/embedding-pairs` | Embedding | embed |
| `reranking` | `own-data/reranking` | Reranking | rerank |
| `image-classification` | `own-data/image-classification` | Classification | classify |
| `audio-classification` | `own-data/audio-classification` | Classification | classify |
| `audio-tts` | `own-data/audio-tts` | Audio | unavailable |

`audio-tts` is visible but unavailable because the repository does not have a
verified data-to-weight-update trainer contract for it. Specialized task-model
scenarios use the verified PyTorch path; unsupported runtimes remain hidden
with an exact reason.

The scenario advisor accepts a plain-language goal and explains why each
verified scenario fits, which source evidence it used, and what remains
ambiguous. It never selects a scenario automatically. **Try a working
example** opens the same checked fixture gallery used by documentation and
contract tests.

## Small checked shapes

Each fenced JSONL example below is parsed and checked against the versioned
scenario shapes by the release-interface check. This verifies record structure,
not the existence or decodability of referenced media. Field mapping can
translate different source names, but starting with these names reduces setup
work.

SFT:

```jsonl
{"prompt":"Summarize the incident report.","response":"The service recovered after the cache was rebuilt."}
{"system":"Answer as a concise analyst.","prompt":"What changed?","response":"The retry policy moved from three attempts to five."}
```

Preference:

```jsonl
{"prompt":"Explain the failure.","chosen":"The request timed out after the upstream stopped responding.","rejected":"It broke for no reason."}
```

Tool use:

```jsonl
{"messages":[{"role":"user","content":"Find ticket 42."}],"tools":[{"type":"function","function":{"name":"get_ticket","parameters":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}}}],"expected_calls":[{"name":"get_ticket","arguments":{"id":42}}]}
```

VLM over an existing image:

```jsonl
{"image":"assets/panel-001.png","prompt":"Which warning light is active?","response":"The amber temperature warning is active."}
```

Audio over an existing recording:

```jsonl
{"audio":"assets/call-001.wav","task":"asr","transcript":"Please restart the service after the backup completes."}
```

Corpus adaptation starts from documents rather than a labeled JSONL row:

```markdown
# Service recovery

Restore the last verified snapshot, confirm database integrity, and then
restart the worker.
```

Image and audio paths refer to assets on the Halo Forge workstation, not the
machine running a remote browser. The standalone snippets above still require
the referenced files. **Try a working example** and `data scenarios template
--output DIRECTORY` include tiny checksummed image/audio assets, so those
fixtures can traverse inspection and proof-run preflight as delivered. For real
data, Halo Forge resolves, hashes, and decodes each asset first. Assets are
referenced by default; choose **Materialize assets** only when a managed copy is
required.

## Dashboard workflow

1. Choose **Train on your data** from Overview, Data, or Train.
2. Choose a file, folder, or pinned Hugging Face revision. The desktop shell
   offers a native chooser; browser surfaces use upload or an explicit
   workstation path.
3. Describe the goal or choose a scenario, then inspect the advisor's reasons
   and inferred fields.
4. Confirm the visual mapping and semantic record preview. Chat, preference,
   tool, image, audio, and corpus records are shown in their working form.
5. Review rejected examples, duplicates, length/token estimates, balance,
   proposed splits, media/extraction failures, and direct remediation actions.
6. Build the immutable version and monitor it in **Activity**.
7. Choose **Continue to training** and explicitly launch. Labeled scenarios
   may use the optional bounded proof flow. Corpus adaptation instead requires
   an explicit LoRA/full choice and token/pass budget; it does not imply a
   proof qualification gate.

RAFT and GRPO proof runs also require an immutable candidate- or
approved-qualified verifier revision. The Train step lists compatible qualified
verifiers by name and links to **Evaluate → Verifiers** when calibration is
still needed; a raw verifier configuration cannot satisfy guided readiness.

The published dataset version contains exact accepted, quarantined,
deduplicated, split, asset, and character counts. Token counts shown before a
training artifact is rendered are estimates. Exact token counts require the
artifact's pinned model, tokenizer revision, and chat template.

## CLI workflow

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

For a document corpus:

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

See [Corpus Adaptation](CORPUS_ADAPTATION.md) for extraction, packing, and
runtime details.

Use `halo-forge data preview` before a build when scripting a new mapping. CLI,
desktop, and browser launches use the same catalog, version identity, and
trainer-artifact renderer.

## Source changes

Halo Forge refuses a managed rebuild or launch when referenced source files or
assets are missing or no longer match their fingerprints. Refreshing a source
creates a new source revision; it never mutates a completed dataset version.
