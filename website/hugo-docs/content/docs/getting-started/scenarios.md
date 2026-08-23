---
title: "Own-Data Scenarios"
description: "Versioned source shapes that Halo Forge can inspect, map, validate, and hand to a compatible trainer"
weight: 3
---

Halo Forge's guided own-data workflow is driven by a versioned scenario
registry. A scenario describes the minimum source fields, safe aliases and
constants, canonical Dataset Lab shape, compatible trainer modes, and a small
launch budget or proof recommendation where applicable. It does not bypass data validation, backend preflight, or the
explicit launch step.

Choose **Train on your data** in the desktop or browser dashboard, or use the
same Dataset Lab catalog from the CLI. See [Use Your Own Data](/docs/data/own-data/)
for the complete source-to-version workflow.

## Available Scenarios

The stable registry anchor is shown exactly as returned by Halo Forge. All
currently published scenarios are revision 1.

| Scenario ID | Registry anchor | Canonical shape | Registry trainer modes |
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

`audio-tts` is intentionally returned as unavailable. Halo Forge has no
verified data-to-weight-update contract for that task. Specialized task-model
scenarios use the verified PyTorch runtime and are filtered out when their
model, dependency, or backend contract is unavailable.

Use **Help me decide** or `halo-forge data scenarios advise` to receive ranked,
explainable suggestions from a plain-language goal and the observed source
shape. Halo Forge still requires the operator to confirm the scenario.

## Registry-Matched Examples

These JSONL records match examples in the checked-in scenario registry and are
parsed by the release-interface check.

Instruction SFT (`instruction-sft@1`):

```jsonl
{"instruction":"Summarize the water cycle.","answer":"Water evaporates, condenses, and returns as precipitation."}
{"instruction":"Name the capital of Japan.","answer":"Tokyo."}
```

Preference pairs (`preference-pairs@1`):

```jsonl
{"prompt":"Explain gravity simply.","chosen":"Gravity pulls masses toward one another.","rejected":"Gravity is when things always fall down."}
```

Prompt reward (`prompt-reward@1`):

```jsonl
{"problem":"What is 12 * 8?","reference_answer":"96"}
{"problem":"Return the first prime after 10.","reference_answer":"11"}
```

Tool and agentic (`tool-agentic@1`):

```jsonl
{"messages":[{"role":"user","content":"Weather in Austin?"},{"role":"assistant","content":"","tool_calls":[{"name":"weather","arguments":{"city":"Austin"}}]}],"tools":[{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]}
```

VLM caption manifest (`vlm-captioning@1`):

```jsonl
{"image":"images/sample.png","caption":"A red bicycle beside a brick wall."}
```

Audio transcription manifest (`audio-asr@1`):

```jsonl
{"audio":"clips/hello.wav","transcript":"Hello from Halo Forge."}
```

Corpus adaptation (`corpus-adaptation@1`) uses a real document fixture rather
than a labeled JSONL record:

```markdown
# Dataset versions

Completed versions are immutable. Refreshing a changed source creates a new
source revision.
```

The inline rows require the referenced assets. Halo Forge's **Try a working
example** action and `halo-forge data scenarios template ... --output
DIRECTORY` include tiny checksummed PNG/WAV companions, so the downloadable
fixtures are train-path-ready. Replace them with real local assets for useful
training; Halo Forge resolves, hashes, decodes, and validates those assets
before building a version.

## Small Managed Launch

For a local file, inspect first and then use the same scenario-backed defaults:

```bash
halo-forge data inspect --path ./support-sft.jsonl --scenario instruction-sft
halo-forge data add --name support-sft --path ./support-sft.jsonl \
  --scenario instruction-sft \
  --map prompt=instruction --map response=answer \
  --accept-recommended
halo-forge data show <dataset-id>
halo-forge data build <dataset-id> --recommended-recipe
halo-forge data render <version-id> --trainer sft
halo-forge sft train --dataset-version <version-id>
```

Labeled scenarios can recommend a bounded proof budget, but the final command
still runs the selected trainer. Corpus adaptation uses an explicit CPT
token/pass budget instead. Verify the semantic preview, quarantined rows,
split policy, hardware preflight, and output path before launching.

Prompt-only RAFT/GRPO proof runs additionally require a compatible qualified
verifier revision. The dashboard offers qualified profiles by name; raw
verifier configuration remains an Advanced, unqualified launch.

For corpus adaptation:

```bash
halo-forge data extract --path ./manuals
halo-forge data inspect --path ./manuals --scenario corpus-adaptation
halo-forge data render <version-id> --trainer cpt --model Qwen/Qwen2.5-1.5B
halo-forge cpt train --dataset-version <version-id> \
  --model Qwen/Qwen2.5-1.5B --adaptation lora \
  --budget-mode passes --corpus-passes 1
```

## Execution Surfaces

Desktop, local browser, remote browser, and CLI use the same source catalog,
scenario revisions, immutable Dataset Lab versions, and work queue. A remote
browser does not make its own filesystem visible to the Halo Forge workstation;
upload a supported file or enter a path that exists on the workstation. See
[Workstation Surfaces](/docs/reference/workstation-surfaces/) for the platform
and distribution matrix.
