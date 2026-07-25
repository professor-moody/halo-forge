---
title: "Command Index"
description: "Complete index of all halo-forge commands and flags"
weight: 1
---

# Command Index

Complete reference for all halo forge commands, subcommands, and flags.

---

## Command Hierarchy

```
halo-forge
├── config
│   └── validate
├── data
│   ├── add / list / show / build / versions / preview / export / materialize
│   ├── render / compare / mine / jobs / extract / corpus-profile
│   ├── repair
│   │   └── inspect / preview / apply / rebase
│   ├── scenarios
│   │   └── list / show / template / advise
│   ├── prepare
│   ├── generate
│   ├── validate
│   ├── synthesize       # generate completions via teacher + verifier filter
│   ├── dedup            # exact + fuzzy MinHash dedup
│   └── score            # heuristic quality scoring + threshold / top-K filter
├── review
│   ├── capabilities
│   ├── schema
│   │   └── list / show / validate / create / revise
│   ├── acquire
│   │   └── list / create / show / cancel / retry
│   ├── queue
│   │   └── list / show / create / clone / pause / resume / archive / start-second-pass
│   ├── items / item / submit / correct / exclude / flag / adjudicate
│   ├── suggestions
│   │   └── generate / show
│   ├── stats
│   └── label-set
│       └── list / show / publish / verify / preview / build-dataset
├── verifier
│   ├── catalog
│   ├── profile
│   │   └── list / show / validate / create / revise
│   ├── protocol
│   │   └── list / show / create / revise
│   ├── qualification-profile
│   │   └── list / show / create / revise
│   ├── calibration
│   │   └── list / create / show / samples / cancel / retry / compare
│   └── qualify / promote / usage
├── reward
│   ├── capabilities
│   ├── system
│   │   └── list / show / validate / create / revise
│   ├── protocol
│   │   └── list / show / create / revise
│   ├── integrity-profile
│   │   └── list / show / create / revise
│   ├── trace
│   │   └── list / show / verify
│   └── audit
│       └── list / create / show / samples / metrics / compare / verify / cancel / retry / review
├── sft
│   ├── train            # supervised finetuning (LoRA / DoRA / rsLoRA / PiSSA / QLoRA)
│   └── datasets
├── cpt
│   └── train            # causal continued pretraining on packed corpus blocks
├── dpo
│   ├── train            # preference optimization
│   └── datasets
├── grpo
│   └── train            # verifier-grounded RL (group-relative advantages)
├── rm
│   └── train            # Bradley-Terry reward model
├── raft
│   └── train            # rejection-sampling RLVR
├── benchmark
│   ├── run
│   └── full
├── eval                 # direct eval plus suite / run / jobs / compare / history / drift
├── checkpoint-policy
│   └── list / show / create / validate
├── sweep                # create/list/show/checkpoints/analyze/decide/report/operate
├── jobs
│   └── worker           # headless durable workstation worker
├── setup
│   └── check / fix      # workstation readiness and safe local remediation
├── support
│   └── bundle
│       └── preview / create / verify
├── release
│   └── qualify          # checksummed platform-package evidence
├── artifact
│   ├── list / show / import / lineage / verify
│   ├── merge / convert / qualify / compare
│   ├── pin / unpin / tag / promote
│   └── serve / export / cleanup
├── storage
│   ├── status
│   └── cleanup
├── probe                # mid-training small-benchmark + baseline diff
├── dashboard            # user-facing dashboard app (alias: app)
├── serve-public         # dashboard API only, for frontend development
├── serve                # OpenAI-compatible /v1/chat/completions endpoint
├── convert              # HF ↔ MLX ↔ GGUF with normalized quant vocabulary
├── merge                # adapter bake / multi-adapter combine
├── replay               # show or relaunch a captured run from replay.json
├── token
│   ├── create           # bearer token for non-loopback API access
│   ├── list
│   └── revoke
├── inference          [EXPERIMENTAL]
│   ├── optimize
│   ├── export
│   └── benchmark
├── vlm                [EXPERIMENTAL]
│   ├── sft
│   ├── train
│   ├── benchmark
│   └── datasets
├── audio              [EXPERIMENTAL]
│   ├── sft
│   ├── train
│   ├── benchmark
│   └── datasets
├── reasoning          [EXPERIMENTAL]
│   ├── sft
│   ├── train
│   ├── benchmark
│   └── datasets
├── agentic            [EXPERIMENTAL]
│   ├── sft
│   ├── train
│   ├── benchmark
│   └── datasets
├── info
└── test
```

---

## Global Flags

These flags work with all commands:

| Flag | Short | Type | Description |
|------|-------|------|-------------|
| `--quiet` | `-q` | flag | Suppress terminal output (logs still written to file) |

### Auto-Logging

All training and benchmark commands automatically log output to `logs/` with timestamped filenames:

```
logs/
├── raft_train_20260110_143052.log
├── sft_train_20260110_121500.log
└── benchmark_run_20260110_160000.log
```

No need for manual `tee` or `PYTHONUNBUFFERED`. Use `--quiet` to suppress terminal output while still capturing logs.

### UI Relaunch Context

The web UI now writes durable launch metadata for each training and benchmark run:

- File: `<output_dir>/launch_context.json`
- Purpose: enables Run detail actions such as `Rerun`, `Clone in Train`, and
  training `Resume Latest`
- Resume scope: `raft`, `vlm`, `audio`, `reasoning`, `agentic` (checkpoint-based); benchmark supports rerun/clone only

---

## Core Commands (Production Ready)

### Guided document corpus and continued pretraining

```bash
halo-forge data scenarios advise \
  --goal "Adapt a model to our manuals" --source-layout pdf
halo-forge data extract --path ./manuals
halo-forge data corpus-profile <version-id>
halo-forge data render <version-id> --trainer cpt \
  --model Qwen/Qwen2.5-1.5B
halo-forge cpt train --dataset-version <version-id> \
  --model Qwen/Qwen2.5-1.5B \
  --adaptation lora \
  --budget-mode passes --corpus-passes 1
```

Document extraction supports text, Markdown, visible HTML, text-layer PDF,
DOCX, and existing structured text columns. Encrypted/image-only PDFs are
quarantined; OCR is not implied. CPT requires an explicit LoRA/full choice and
never receives test or canary records. See
[Adapt a Model to Documents](/docs/data/corpus-adaptation/).

### Artifact Studio and workstation storage

Artifact Studio manages immutable occurrences and content-addressed model
blobs. Transformations and qualification are queued on the durable workstation
worker and return both an operation ID and a work-item ID.

```bash
halo-forge artifact list
halo-forge artifact show <occurrence-id-or-alias>
halo-forge artifact import ./run/final_model --kind final --format hf --managed
halo-forge artifact lineage <occurrence-id>
halo-forge artifact verify <occurrence-id>

halo-forge artifact merge <adapter-a> <adapter-b> \
  --base-model Qwen/Qwen2.5-7B --method dare_ties
halo-forge artifact convert <occurrence-id> --format gguf --quantization q4
halo-forge artifact qualify <occurrence-id> --profile <profile-revision-id>
halo-forge artifact compare <parent-id> <candidate-id>
halo-forge artifact promote <occurrence-id> candidate

halo-forge artifact serve <occurrence-id>
halo-forge artifact export <occurrence-id> ./portable-model
halo-forge storage status
halo-forge storage cleanup
```

q4/q8 are post-training quantization, not QAT. Cleanup previews first, requires
a review note to apply, and keeps removed content in trash for seven days.

See [Artifact Studio](/docs/artifact-studio/) for the dashboard workflow,
qualification gates, serving, and retention protections.

### Adaptive training and research evidence

Immutable checkpoint policies and matched-seed evidence use the same catalog
and scheduler as the dashboard:

```bash
halo-forge checkpoint-policy validate --spec ./policy.yaml
halo-forge checkpoint-policy create --spec ./policy.yaml
halo-forge checkpoint-policy list --trainer sft
halo-forge checkpoint-policy show <policy-or-revision-id>

halo-forge sweep create --spec ./repeat.yaml \
  --checkpoint-policy <revision-id> --max-steps 1000
halo-forge sweep checkpoints <group-id>
halo-forge sweep analyze <group-id>
halo-forge sweep decide <group-id> --select <trial-id> \
  --rationale "Matched-seed evidence supports this branch"
halo-forge sweep report <group-id> --decision <decision-id>
halo-forge sweep resume <group-id> --reason "Reviewed retry"

halo-forge eval history --suite-revision <revision-id>
halo-forge eval drift --base <evaluation-id> --candidate <evaluation-id>
```

Hugging Face adaptive policies use step budgets; RAFT, VLM, audio,
reasoning, and agentic policies use cycle budgets. Unsupported resume
contracts remain final-only. Cohort analysis defaults to matched seeds, a 95%
percentile-bootstrap interval, 10,000 resamples, and bootstrap seed 42.

See [Repeats and hyperparameter sweeps](/docs/sweep/) and the local
`docs/ADAPTIVE_EVIDENCE.md` guide for policy schemas, evidence classifications,
and reviewed-decision behavior.

### Reward Integrity and Training Signal Studio

The `reward` family pins an optimization verifier and independent sentinel,
captures the exact retained outputs used during verifier-guided training, and
audits those same outputs without regeneration.

```bash
halo-forge reward capabilities

halo-forge reward system validate --spec ./reward-system.json
halo-forge reward system create --name "Code reward + sentinel" \
  --spec ./reward-system.json
halo-forge reward system list
halo-forge reward system show <system-or-revision-id>

halo-forge reward protocol list
halo-forge reward protocol create --name balanced-custom --spec ./protocol.json
halo-forge reward integrity-profile list
halo-forge reward integrity-profile create \
  --name "Human aligned" --template human_aligned_integrity
```

Compatible verifier-guided training commands accept one managed audit binding:

| Flag | Repeatable | Description |
|---|---:|---|
| `--reward-system-revision` | no | Immutable optimizer, sentinel, mapping, and auditor identity |
| `--reward-audit-protocol-revision` | no | Immutable snapshot-retention protocol |
| `--reward-integrity-profile-revision` | no | Immutable pass/warn/fail policy |
| `--reward-development-suite-revision` | no | Optional pinned development/unspecified suite; evaluates each published checkpoint as durable completion evidence before its reward audit |
| `--reward-audit-boundary` | yes | Selected step/cycle value or `final`, as supported by the capability |
| `--wait` | no | Wait for managed training/audit work where the command supports it |

The three revision flags must be supplied together. They conflict with a
separate verifier-profile revision because the reward system already pins the
training verifier. Raw `--verifier` usage remains compatible but is unmonitored.

```bash
halo-forge grpo train \
  --max-steps 400 \
  --reward-system-revision <system-revision-id> \
  --reward-audit-protocol-revision <protocol-revision-id> \
  --reward-integrity-profile-revision <profile-revision-id> \
  --reward-development-suite-revision <optional-development-suite-revision-id> \
  --reward-audit-boundary 100 --reward-audit-boundary 400

halo-forge reward trace list --run-id <run-id>
halo-forge reward trace show <trace-id>
halo-forge reward trace verify <trace-id>

halo-forge reward audit list --run-id <run-id>
halo-forge reward audit show <audit-id>
halo-forge reward audit samples <audit-id> --population uniform_core
halo-forge reward audit metrics <audit-id>
halo-forge reward audit compare <base-id> <candidate-id>
halo-forge reward audit verify <audit-id>
halo-forge reward audit review <audit-id> \
  --action continue --reason "Reviewed paired sentinel evidence"
```

Standalone audit creation accepts `--trace`, `--reward-system-revision`,
`--protocol-revision`, and `--integrity-profile-revision`; add `--wait` for
foreground execution. Review actions are `continue`, `stop`, `fork`, and
`create_review_proposal`, and every action requires `--reason`.

Built-in capture is final-only for Hugging Face RAFT, cycle/final and resumable
for MLX RAFT, step/final and resumable for Hugging Face GRPO when `max_steps` is
configured, and final-only for MLX GRPO. Reasoning, agentic/tool, VLM, and audio
remain cycle/final and resumable. A pinned development-suite revision is
evidence identity only in this release: no independent suite evaluation runs at
the boundary and no suite result is combined with the checkpoint gate.
`balanced_256` is the default retention protocol and
`human_aligned_integrity` is the normal guided policy. No result automatically
tunes a verifier, creates data, launches a fork, or promotes an artifact.

Replay manifest v4 checks the complete reward identity. Intentional drift
requires both `--allow-reward-drift` and `--reward-drift-reason`; the reason and
differences are appended beside the manifest.

See [Reward Integrity](/docs/reward-integrity/) for capture fidelity,
same-output guarantees, evidence thresholds, and pause behavior.

### Human Feedback and Active Data Studio

The `review` family creates deterministic acquisition proposals, records human
review as append-only events, publishes immutable label sets, and hands reviewed
records to Dataset Lab. It never creates a queue, publishes labels, builds a
dataset, or starts training without a separate operator action.

```bash
# Discover capabilities and manage immutable annotation definitions.
halo-forge review capabilities
halo-forge review schema list
halo-forge review schema create \
  --name "SFT correction" --modality text --task-type text_correction \
  --definition ./annotation-schema.yaml
halo-forge review schema revise <schema-id> \
  --definition ./annotation-schema-v2.yaml

# Create and inspect a deterministic proposal.
halo-forge review acquire create \
  --source-kind evaluation_comparison \
  --base-evaluation <base-evaluation-id> \
  --candidate-evaluation <candidate-evaluation-id> \
  --strategy regression --quota 100 --seed 42
halo-forge review acquire show <batch-id> --candidates

# Explicitly create a queue and record review events.
halo-forge review queue create \
  --batch <batch-id> --schema <schema-revision-id>
halo-forge review items <queue-id>
halo-forge review item <item-id>
halo-forge review submit <item-id> --label '{"response": "corrected"}'
halo-forge review correct <item-id> \
  --label '{"response": "revised"}' --reason "Fixed an omission" \
  --supersedes-event <event-id>
halo-forge review stats <queue-id>

# Suggestions are optional, on demand, and do not become human labels.
halo-forge review suggestions generate <item-id> \
  --provider ollama --model qwen2.5:7b

# Explicitly publish, verify, preview, and build immutable outputs.
halo-forge review label-set publish --queue <queue-id>
halo-forge review label-set verify <revision-id>
halo-forge review label-set preview <revision-id> --dataset <dataset-id>
halo-forge review label-set build-dataset <revision-id> \
  --dataset <dataset-id> --target-split train
```

Use `review acquire create --source-kind ...` and repeatable `--strategy` flags
for common proposals. Use `--spec` for multiple ordered sources or acquisition
strata. Review commands support `--database`, `--root`, and `--json`; paginated
commands also accept `--limit` and `--offset`.

Operational, holdout, test, and canary sources are ineligible for acquisition.
Low-margin selection requires compatible scores, and diversity requires a
pinned available embedding revision. Development exposure is recorded and
propagated into the label set and descendant dataset version.

See [Review Studio](/docs/review-studio/) for dashboard routes, two-pass review,
protected evidence, output modes, and the full reviewed workflow.

### halo-forge config validate

Validate a configuration file.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `config` | - | path | Yes | - | Path to config file |
| `--type` | `-t` | string | No | `auto` | Config type: `raft`, `sft`, `auto` |
| `--verbose` | `-v` | flag | No | false | Show config contents |

```bash
halo-forge config validate configs/raft_windows.yaml
halo-forge config validate configs/sft.yaml --type sft --verbose
```

---

### halo-forge data prepare

Download and prepare public datasets.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--dataset` | `-d` | string | No | - | Dataset name |
| `--output` | `-o` | path | No | - | Output file path |
| `--template` | - | string | No | `qwen` | Chat template |
| `--system-prompt` | - | string | No | - | Override system prompt |
| `--list` | - | flag | No | false | List available datasets |

```bash
halo-forge data prepare --list
halo-forge data prepare --dataset humaneval --output data/humaneval.jsonl
halo-forge data prepare --dataset mbpp --template qwen --system-prompt "You are a Python expert."
```

---

### halo-forge data generate

Generate training data using LLM.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--topic` | `-t` | string | No | - | Topic name |
| `--backend` | `-b` | string | No | `deepseek` | LLM backend |
| `--model` | - | string | No | - | Model name for backend |
| `--output` | `-o` | path | No | - | Output file path |
| `--template` | - | string | No | `qwen` | Chat template |
| `--list` | - | flag | No | false | List available topics |

```bash
halo-forge data generate --list
halo-forge data generate --topic windows_api --backend deepseek --output data/windows.jsonl
```

---

### halo-forge data validate

Validate dataset format.

This command is dependency-light by design: it validates local JSONL structure
without requiring public dataset download dependencies.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `file` | - | path | Yes | - | Path to JSONL file |
| `--preview` | `-p` | flag | No | false | Show preview of examples |

```bash
halo-forge data validate data/training.jsonl
halo-forge data validate data/training.jsonl --preview
```

---

### halo-forge sft train

Run supervised fine-tuning.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--config` | `-c` | path | No | - | Config file path |
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-Coder-7B` | Base model |
| `--dataset` | `-d` | string | No | - | HuggingFace dataset ID or short name |
| `--data` | - | path | No | - | Local training data file (JSONL) |
| `--max-samples` | - | int | No | - | Limit number of training samples |
| `--output` | `-o` | path | No | `models/sft` | Output directory |
| `--epochs` | - | int | No | 3 | Number of epochs |
| `--resume` | - | path | No | - | Resume from checkpoint |
| `--dry-run` | - | flag | No | false | Validate config without training |

**Dataset short names:** `codealpaca`, `metamath`, `gsm8k_sft`, `llava`, `librispeech_sft`, `xlam_sft`, `glaive_sft`

```bash
# Using HuggingFace dataset
halo-forge sft train --dataset codealpaca --model Qwen/Qwen2.5-Coder-3B --output models/sft_3b

# Using local data
halo-forge sft train --data data/sft.jsonl --model Qwen/Qwen2.5-Coder-3B --output models/sft_3b

# With sample limit
halo-forge sft train --dataset metamath --max-samples 50000 --epochs 2

# Resume from checkpoint
halo-forge sft train --config configs/sft.yaml --resume models/sft/checkpoint-500
```

---

### halo-forge sft datasets

List available SFT datasets.

```bash
halo-forge sft datasets
```

Output shows datasets organized by domain (Code, Reasoning, VLM, Audio, Agentic) with HuggingFace IDs and sizes.

---

### halo-forge raft train

Run RAFT (Reward-Ranked Fine-Tuning).

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--config` | `-c` | path | No | - | Config file path |
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-Coder-3B` | Base model |
| `--checkpoint` | - | path | No | - | SFT checkpoint path |
| `--prompts` | `-p` | path | No | - | Prompts file |
| `--output` | `-o` | path | No | `models/raft` | Output directory |
| `--cycles` | - | int | No | 6 | Number of RAFT cycles |
| `--verifier` | - | string | No | `gcc` | Verifier type (see below) |
| `--samples-per-prompt` | - | int | No | 8 | Samples to generate per prompt |
| `--temperature` | - | float | No | 0.7 | Generation temperature |
| `--max-new-tokens` | - | int | No | 1024 | Max tokens to generate |
| `--keep-percent` | - | float | No | 0.5 | Keep top X% of passing samples |
| `--reward-threshold` | - | float | No | 0.5 | Minimum reward to pass |
| `--min-samples` | - | int | No | - | Auto-adjust threshold if fewer pass |
| `--curriculum` | - | string | No | `none` | Curriculum strategy |
| `--curriculum-stats` | - | path | No | - | Historical stats file (for `historical` curriculum) |
| `--curriculum-start` | - | float | No | 0.2 | Start fraction (for `progressive` curriculum) |
| `--curriculum-increment` | - | float | No | 0.2 | Increment per cycle (for `progressive` curriculum) |
| `--reward-shaping` | - | string | No | `fixed` | Reward shaping strategy |
| `--lr-decay` | - | float | No | 0.85 | LR decay per cycle |
| `--min-lr` | - | float | No | 1e-6 | Minimum learning rate |
| `--experimental-attention` | - | flag | No | false | Enable experimental ROCm attention |
| `--system-prompt` | - | string | No | (Windows prompt) | System prompt |
| `--host` | - | string | No | - | MSVC verifier host |
| `--user` | - | string | No | - | MSVC verifier user |
| `--ssh-key` | - | path | No | - | MSVC verifier SSH key |

**Verifier choices:** `gcc`, `mingw`, `msvc`, `rust`, `go`, `dotnet`, `powershell`, `humaneval`, `mbpp`, `python`, `auto`

**Curriculum choices:** `none`, `complexity`, `progressive`, `adaptive`, `historical`

**Reward shaping choices:** `fixed`, `annealing`, `adaptive`, `warmup`

```bash
# Basic RAFT training
halo-forge raft train \
  --model Qwen/Qwen2.5-Coder-3B \
  --prompts data/prompts.jsonl \
  --verifier mingw \
  --cycles 6 \
  --output models/raft_3b

# With SFT checkpoint and LR decay
halo-forge raft train \
  --checkpoint models/sft_3b/final \
  --prompts data/prompts.jsonl \
  --verifier auto \
  --lr-decay 0.85 \
  --cycles 6

# With MSVC verifier
halo-forge raft train \
  --prompts data/windows.jsonl \
  --verifier msvc \
  --host 10.0.0.152 \
  --user keys \
  --ssh-key ~/.ssh/win
```

---

### halo-forge benchmark run

Run pass@k benchmark.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | Model path |
| `--prompts` | `-p` | path | Yes | - | Prompts file |
| `--output` | `-o` | path | No | - | Output file path |
| `--samples` | - | int | No | 10 | Samples per prompt |
| `--k` | - | string | No | `1,5,10` | k values (comma-separated) |
| `--max-prompts` | - | int | No | - | Max prompts to evaluate |
| `--verifier` | - | string | No | `gcc` | Verifier type (gcc, humaneval, mbpp, etc.) |
| `--base-model` | - | string | No | `Qwen/Qwen2.5-Coder-7B` | Base model |
| `--system-prompt` | - | string | No | (Windows prompt) | System prompt |
| `--host` | - | string | No | - | MSVC host |
| `--user` | - | string | No | - | MSVC user |
| `--ssh-key` | - | path | No | - | MSVC SSH key |
| `--cross-compile` | - | flag | No | false | Windows cross-compile (rust/go) |
| `--run-after-compile` | - | flag | No | false | Run after compile |
| `--experimental-attention` | - | flag | No | false | Enable experimental ROCm attention |

```bash
halo-forge benchmark run \
  --model models/raft_3b/cycle_6 \
  --prompts data/test.jsonl \
  --verifier mingw \
  --samples 10 \
  --output results/benchmark.json
```

---

### halo-forge benchmark full

Run comprehensive RAFT benchmark.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No* | - | Model to benchmark |
| `--suite` | `-s` | string | No* | - | Predefined suite |
| `--cycles` | `-c` | int | No | 2 | RAFT cycles |
| `--output` | `-o` | path | No | `results/benchmarks` | Output directory |
| `--quiet` | `-q` | flag | No | false | Minimal output |

*Either `--model` or `--suite` is required.

**Suite choices:** `all` (0.5B, 1.5B, 3B), `small` (0.5B), `medium` (0.5B, 1.5B)

```bash
halo-forge benchmark full --model Qwen/Qwen2.5-Coder-0.5B --cycles 2
halo-forge benchmark full --suite all --output results/full_benchmark
```

---

### halo-forge benchmark eval

Evaluate a model on standard benchmarks (HumanEval, MBPP, LiveCodeBench, etc.).

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | Yes | - | Model name or path |
| `--benchmark` | `-b` | string | No | `humaneval` | Benchmark name |
| `--limit` | - | int | No | - | Max samples to evaluate |
| `--output` | `-o` | path | No | - | Output file path |
| `--samples-per-prompt` | - | int | No | 5 | Samples per prompt for pass@k |
| `--run-after-compile` | - | flag | No | false | Run compiled code |
| `--language` | - | string | No | - | Language (cpp, rust, go) |
| `--verifier` | - | string | No | - | Verifier type |

**Benchmark choices:** `humaneval`, `mbpp`, `livecodebench`, `cpp`, `rust`, `go`

```bash
halo-forge benchmark eval --model models/raft/final --benchmark humaneval --limit 164
halo-forge benchmark eval --model models/raft/final --benchmark cpp --language cpp --run-after-compile
```

---

### halo-forge info

Show hardware and system information.

```bash
halo-forge info
```

---

### halo-forge ui

Launch the web UI.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--host` | - | string | No | `127.0.0.1` | Host to bind |
| `--port` | `-p` | int | No | `8080` | Port to bind |
| `--reload` | - | flag | No | false | Enable hot reload |
| `--open-browser` | - | flag | No | false | Auto-open browser after startup |
| `--no-browser` | - | flag | No | false | Disable browser auto-open (this is the default behavior) |

```bash
halo-forge ui --no-browser
halo-forge ui --open-browser
halo-forge ui --host 0.0.0.0 --port 8080
```

Common deep-link routes for operator workflows:

```bash
http://127.0.0.1:8080/training?mode=audio&ui_mode=quickstart&preset=audio_whisper_tiny
http://127.0.0.1:8080/benchmark?view=non_code&ui_mode=quickstart&preset=non_code_smoke
http://127.0.0.1:8080/inference?mode=optimize&ui_mode=quickstart&preset=optimize_int4_smoke
http://127.0.0.1:8080/ops-console?module=data&execution_mode=live
```

---

### halo-forge test

Run pipeline validation tests.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--level` | `-l` | string | No | `standard` | Test level |
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-Coder-0.5B` | Model for testing |
| `--verbose` | `-v` | flag | No | false | Verbose output |
| `--baseline-file` | - | path | No | `tests/baselines/modality_runtime_baseline.v1.json` | Baseline JSON path (`modality` / `ops-burnin` / `all-module-qualification`) |
| `--write-baseline` | - | flag | No | false | Write/overwrite baseline (`modality` / `ops-burnin` / `all-module-qualification`) |
| `--compare-baseline` | - | flag | No | false | Compare run to baseline and fail on hard drift |
| `--report-file` | - | path | No | `results/readiness/ops_e2e_launch_reliability.v1.json` | Report output path (`ops-e2e` / `ops-burnin` / `all-modules` / `walkthroughs` / `all-module-qualification` / `all-module-bootstrap` / `all-module-live`) |
| `--strict` | - | flag | No | false | Fail on `status=fail` modules (`ops-e2e` / `ops-burnin` / `all-modules` / `walkthroughs` / `all-module-qualification` / `all-module-bootstrap` / `all-module-live`) |
| `--seed` | - | int | No | `42` | Deterministic seed (`ops-e2e` / `ops-burnin` / `all-modules` / `walkthroughs` / `all-module-qualification` / `all-module-bootstrap` / `all-module-live`) |
| `--fixture-pack` | - | string | No | - | Fixture pack (`v1`) or custom path (`ops-e2e` / `all-modules` / `all-module-qualification` level) |
| `--burnin-profile` | - | string | No | `tiny-v1` | Dataset-backed burn-in profile (`ops-burnin` level) |
| `--profile` | - | string | No | `bounded-v1` | Readiness profile (`all-modules` level) or walkthrough profile (`contract-v1`/`live-local`) |
| `--qualification-profile` | - | string | No | `contract-v1` | Qualification profile (`contract-v1` / `fixture-v1` / `live-local`) for `all-module-qualification` |
| `--bootstrap-profile` | - | string | No | `contract-v1` | Bootstrap profile (`contract-v1` / `live-local`) for `all-module-bootstrap` |
| `--live-profile` | - | string | No | `live-smoke-v1` | Live execution profile (`live-smoke-v1` / `live-local`) for `all-module-live` |
| `--output-root` | - | path | No | `results/bootstrap` | Evidence output root for `all-module-bootstrap` or `all-module-live` |
| `--module` | - | string (repeatable) | No | - | Filter module(s) for `all-modules`, `walkthroughs`, `all-module-qualification`, `all-module-bootstrap`, or `all-module-live` |
| `--execute` | - | flag | No | false | Execute bounded probes for `walkthroughs` when using `--profile live-local` |
| `--show-fix-commands` | - | flag | No | false | Emit `ALL_QUAL_FIX` remediation lines for `all-module-qualification` |

**Level choices:** `smoke` (no GPU), `standard` (with GPU), `full` (with training), `modality` (deterministic modality fixture + smoke suite), `ops-e2e` (non-code launch lifecycle reliability), `ops-burnin` (bounded dataset-backed non-code burn-in), `all-modules` (coding + non-coding readiness checks), `walkthroughs` (internal/operator E2E walkthrough contract validation), `all-module-qualification` (explicit bounded lifecycle qualification orchestration), `all-module-bootstrap` (bounded evidence generation/remediation for all-module readiness), `all-module-live` (bounded live execution closure probes across all modules)

Baseline drift checks validate runtime contract stability, not model-quality promotion thresholds.

```bash
halo-forge test --level smoke
halo-forge test --level standard --verbose
halo-forge test --level full --model Qwen/Qwen2.5-Coder-1.5B
halo-forge test --level modality
halo-forge test --level modality --compare-baseline
halo-forge test --level modality --write-baseline
halo-forge test --level ops-e2e --fixture-pack v1 --report-file results/readiness/ops_e2e_launch_reliability.v1.json
halo-forge test --level ops-e2e --fixture-pack v1 --strict
halo-forge test --level ops-burnin --burnin-profile tiny-v1 --report-file results/readiness/ops_dataset_burnin.v1.json
halo-forge test --level ops-burnin --burnin-profile tiny-v1 --compare-baseline --strict
halo-forge test --level all-modules --fixture-pack v1 --report-file results/readiness/all_modules_readiness.v1.json
halo-forge test --level all-modules --fixture-pack v1 --strict
halo-forge test --level all-modules --module sft --module raft --strict
halo-forge test --level walkthroughs --profile contract-v1 --report-file .internal_docs/research_testing/walkthroughs/reports/all_module_e2e_walkthrough_report.v1.json
halo-forge test --level walkthroughs --module sft --module raft --profile live-local --execute
halo-forge test --level all-module-qualification --qualification-profile contract-v1 --report-file results/readiness/all_module_qualification.v1.json
halo-forge test --level all-module-qualification --qualification-profile fixture-v1 --fixture-pack v1 --compare-baseline --baseline-file tests/baselines/all_module_qualification_baseline.v1.json --strict
halo-forge test --level all-module-qualification --show-fix-commands
halo-forge test --level all-module-bootstrap --bootstrap-profile contract-v1 --report-file results/readiness/all_module_bootstrap.v1.json
halo-forge test --level all-module-bootstrap --bootstrap-profile live-local --module inference --strict
halo-forge test --level all-module-live --live-profile live-smoke-v1 --report-file results/readiness/all_module_live_execution.v1.json
halo-forge test --level all-module-live --live-profile live-local --module inference --strict
```

Equivalent script entrypoint:

```bash
python3 scripts/run_all_module_qualification.py \
  --qualification-profile fixture-v1 \
  --fixture-pack v1 \
  --write-report \
  --show-fix-commands \
  --report-file results/readiness/all_module_qualification.v1.json

python3 scripts/run_all_module_bootstrap.py \
  --bootstrap-profile contract-v1 \
  --write-report \
  --report-file results/readiness/all_module_bootstrap.v1.json

python3 scripts/run_all_module_live_matrix.py \
  --live-profile live-smoke-v1 \
  --write-report \
  --report-file results/readiness/all_module_live_execution.v1.json
```

Non-code modality UI readiness reports (contract-only) can be generated with:

```bash
python3 scripts/run_non_code_modality_matrix.py \
  --validate-training vlm=models/phase7d/vlm_phase7d \
  --validate-training audio=models/phase7d/audio_phase7d \
  --validate-training reasoning=models/phase7d/reasoning_phase7d \
  --validate-training agentic=models/phase7d/agentic_phase7d \
  --write-readiness-report \
  --readiness-from-validation \
  --readiness-report-file results/readiness/non_code_modalities_readiness.v1.json
```

`status=pass|warn|fail` is contract-based runtime readiness, not model-quality promotion.

Cross-module ops readiness reports (non-coding scope) can be generated with:

```bash
python3 scripts/run_ops_module_matrix.py \
  --fixture-pack v1 \
  --write-report \
  --report-file results/readiness/ops_modules_readiness.v1.json
```

Strict fixture-backed gate (used in nightly CI):

```bash
python3 scripts/run_ops_module_matrix.py --fixture-pack v1 --strict
```

Ops E2E launch lifecycle reliability (non-coding scope):

```bash
python3 scripts/run_ops_e2e_reliability.py \
  --fixture-pack v1 \
  --write-report \
  --report-file results/readiness/ops_e2e_launch_reliability.v1.json
```

Strict nightly E2E gate:

```bash
python3 scripts/run_ops_e2e_reliability.py --fixture-pack v1 --strict
```

Dataset-backed non-code burn-in report:

```bash
python3 scripts/run_ops_dataset_burnin.py \
  --burnin-profile tiny-v1 \
  --write-report \
  --report-file results/readiness/ops_dataset_burnin.v1.json
```

Strict nightly burn-in + baseline drift gate:

```bash
python3 scripts/run_ops_dataset_burnin.py \
  --burnin-profile tiny-v1 \
  --strict \
  --compare-baseline \
  --baseline-file tests/baselines/ops_dataset_burnin_baseline.v1.json
```

All-module parity readiness (coding + non-coding):

```bash
python3 scripts/run_all_module_matrix.py \
  --fixture-pack v1 \
  --write-report \
  --report-file results/readiness/all_modules_readiness.v1.json
```

Strict nightly all-module gate:

```bash
python3 scripts/run_all_module_matrix.py --fixture-pack v1 --strict
```

CI policy:
- PR/push CI uses non-strict report generation (informational for readiness status).
- Nightly CI uses strict mode and fails on module `status=fail` plus hard contract drift (readiness + qualification + bootstrap + live execution + E2E + dataset burn-in reports).

Readiness interpretation:
- `WARN` commonly indicates missing historical evidence (for example prior `training_summary.json` or benchmark outputs) and remains non-blocking for UI launches.
- `FAIL` indicates a contract/preflight issue; check `launch_blocked`, `issue_code`, `severity`, `what_is_missing`, and `fix_now` in readiness/qualification payload entries.

---

## Experimental Commands

These commands are in active development. APIs may change.

### halo-forge inference optimize

Optimize model for deployment.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | Model path |
| `--target-precision` | - | string | No | `int4` | Target precision |
| `--target-latency` | - | float | No | 50.0 | Target latency (ms) |
| `--calibration-data` | - | path | No | - | Calibration data JSONL |
| `--output` | `-o` | path | No | `models/optimized` | Output directory |

**Precision choices:** `int4`, `int8`, `fp16`

```bash
halo-forge inference optimize \
  --model models/raft_7b/cycle_6 \
  --target-precision int4 \
  --output models/optimized
```

---

### halo-forge inference export

Export model to deployment format.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | Model path |
| `--format` | `-f` | string | Yes | - | Export format |
| `--quantization` | `-q` | string | No | `Q4_K_M` | GGUF quantization |
| `--output` | `-o` | path | Yes | - | Output path |

**Format choices:** `gguf`, `onnx`

**GGUF quantization types:** `Q4_K_M`, `Q4_K_S`, `Q8_0`, `F16`

```bash
halo-forge inference export \
  --model models/trained \
  --format gguf \
  --quantization Q4_K_M \
  --output models/model.gguf
```

---

### halo-forge inference benchmark

Benchmark inference latency.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | Model path |
| `--prompts` | `-p` | path | No | - | Test prompts JSONL |
| `--num-prompts` | - | int | No | 10 | Number of prompts |
| `--max-tokens` | - | int | No | 100 | Max tokens to generate |
| `--warmup` | - | int | No | 3 | Warmup iterations |
| `--measure-memory` | - | flag | No | false | Measure memory usage |

```bash
halo-forge inference benchmark \
  --model models/optimized \
  --num-prompts 50 \
  --measure-memory
```

---

### halo-forge vlm sft

SFT training for VLM.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2-VL-2B-Instruct` | VLM model |
| `--dataset` | `-d` | string | No | `llava` | SFT dataset name |
| `--max-samples` | - | int | No | - | Limit training samples |
| `--output` | `-o` | path | No | `models/vlm_sft` | Output directory |
| `--epochs` | - | int | No | 2 | Number of epochs |
| `--dry-run` | - | flag | No | false | Validate config only |

```bash
halo-forge vlm sft --dataset llava --model Qwen/Qwen2-VL-2B-Instruct --output models/vlm_sft
```

---

### halo-forge vlm train

Train VLM with RAFT.
Real-training enabled. Use `--allow-prototype-train` only if a modality is temporarily re-gated.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2-VL-7B-Instruct` | VLM model |
| `--dataset` | `-d` | string | Yes | - | Dataset name or JSONL |
| `--output` | `-o` | path | No | `models/vlm_raft` | Output directory |
| `--cycles` | - | int | No | 6 | RAFT cycles |
| `--samples-per-prompt` | - | int | No | 4 | Samples per prompt |
| `--perception-weight` | - | float | No | 0.3 | Perception weight |
| `--reasoning-weight` | - | float | No | 0.4 | Reasoning weight |
| `--output-weight` | - | float | No | 0.3 | Output weight |
| `--lr-decay` | - | float | No | 0.85 | LR decay per cycle |
| `--temperature` | - | float | No | 0.7 | Generation temperature |
| `--limit` | - | int | No | - | Limit dataset samples |
| `--allow-prototype-train` | - | flag | No | false | Compatibility override for temporary prototype gating |
| `--resume-from-cycle` | - | int | No | 0 | Resume from a previously saved cycle index |
| `--seed` | - | int | No | 42 | Deterministic runtime seed |

**Dataset choices:** `textvqa`, `docvqa`, `chartqa`, `realworldqa`, `mathvista`
**Supported model families:** `qwen2-vl`, `qwen-vl`, `llava`

```bash
halo-forge vlm train \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset textvqa \
  --cycles 6 \
  --output models/vlm_textvqa \
  --seed 42
```

---

### halo-forge vlm benchmark

Benchmark VLM on dataset.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | VLM model path |
| `--dataset` | `-d` | string | No | `textvqa` | Dataset name |
| `--split` | - | string | No | `validation` | Dataset split |
| `--limit` | - | int | No | 100 | Limit samples |
| `--output` | `-o` | path | No | - | Output file |

```bash
halo-forge vlm benchmark \
  --model models/vlm_raft/cycle_6 \
  --dataset docvqa \
  --limit 200 \
  --output results/vlm_benchmark.json
```

---

### halo-forge vlm datasets

List available VLM datasets.

```bash
halo-forge vlm datasets
```

---

### halo-forge audio sft

SFT training for audio models.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `openai/whisper-small` | Audio model |
| `--dataset` | `-d` | string | No | `librispeech_sft` | SFT dataset name |
| `--max-samples` | - | int | No | - | Limit training samples |
| `--output` | `-o` | path | No | `models/audio_sft` | Output directory |
| `--epochs` | - | int | No | 3 | Number of epochs |
| `--dry-run` | - | flag | No | false | Validate config only |

```bash
halo-forge audio sft --dataset librispeech_sft --model openai/whisper-small --output models/audio_sft
```

---

### halo-forge audio train

Train audio model with RAFT.
Real-training enabled. Use `--allow-prototype-train` only if a modality is temporarily re-gated.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `openai/whisper-small` | Audio model |
| `--dataset` | `-d` | string | Yes | - | Dataset name |
| `--task` | - | string | No | `asr` | Task: `asr`, `tts`, `classification` |
| `--output` | `-o` | path | No | `models/audio_raft` | Output directory |
| `--cycles` | - | int | No | 4 | RAFT cycles |
| `--lr` | - | float | No | 1e-5 | Learning rate |
| `--lr-decay` | - | float | No | 0.85 | LR decay per cycle |
| `--limit` | - | int | No | - | Limit dataset samples |
| `--allow-prototype-train` | - | flag | No | false | Compatibility override for temporary prototype gating |
| `--resume-from-cycle` | - | int | No | 0 | Resume from a previously saved cycle index |
| `--seed` | - | int | No | 42 | Deterministic runtime seed |

**Dataset choices:** `librispeech`, `common_voice`, `audioset`, `speech_commands`
**Supported model families:** `whisper`

```bash
halo-forge audio train \
  --model openai/whisper-small \
  --dataset librispeech \
  --task asr \
  --cycles 4 \
  --output models/audio_asr \
  --seed 42
```

---

### halo-forge audio benchmark

Benchmark audio model on dataset.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | Audio model path |
| `--dataset` | `-d` | string | No | `librispeech` | Dataset name |
| `--task` | - | string | No | `asr` | Task type |
| `--limit` | - | int | No | 100 | Limit samples |
| `--output` | `-o` | path | No | - | Output file |

```bash
halo-forge audio benchmark \
  --model openai/whisper-small \
  --dataset librispeech \
  --limit 50 \
  --output results/audio_benchmark.json
```

---

### halo-forge audio datasets

List available audio datasets.

```bash
halo-forge audio datasets
```

---

### halo-forge reasoning sft

SFT training for reasoning models.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-3B-Instruct` | Base model |
| `--dataset` | `-d` | string | No | `metamath` | SFT dataset name |
| `--max-samples` | - | int | No | - | Limit training samples |
| `--output` | `-o` | path | No | `models/reasoning_sft` | Output directory |
| `--epochs` | - | int | No | 2 | Number of epochs |
| `--dry-run` | - | flag | No | false | Validate config only |

**SFT Dataset choices:** `metamath`, `gsm8k_sft`

```bash
halo-forge reasoning sft --dataset metamath --model Qwen/Qwen2.5-3B-Instruct --output models/reasoning_sft
```

---

### halo-forge reasoning train

Train on math/reasoning datasets with RAFT.
Real-training enabled. Use `--allow-prototype-train` only if a modality is temporarily re-gated.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-7B-Instruct` | Base model |
| `--dataset` | `-d` | string | Yes | - | Dataset name |
| `--output` | `-o` | path | No | `models/reasoning_raft` | Output directory |
| `--cycles` | - | int | No | 4 | RAFT cycles |
| `--lr` | - | float | No | 1e-5 | Learning rate |
| `--lr-decay` | - | float | No | 0.85 | LR decay per cycle |
| `--limit` | - | int | No | - | Limit dataset samples |
| `--allow-prototype-train` | - | flag | No | false | Compatibility override for temporary prototype gating |
| `--resume-from-cycle` | - | int | No | 0 | Resume from a previously saved cycle index |
| `--seed` | - | int | No | 42 | Deterministic runtime seed |
**Supported model families:** `qwen2.5`, `qwen2`, `qwen`, `llama-3`, `llama3`, `mistral`

**RAFT Dataset choices:** `gsm8k`, `math`, `aime`

```bash
halo-forge reasoning train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset gsm8k \
  --cycles 4 \
  --output models/reasoning_gsm8k \
  --seed 42
```

---

### halo-forge reasoning benchmark

Benchmark on math/reasoning dataset.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | path | Yes | - | Model path |
| `--dataset` | `-d` | string | No | `gsm8k` | Dataset name |
| `--limit` | - | int | No | 100 | Limit samples |
| `--output` | `-o` | path | No | - | Output file |

```bash
halo-forge reasoning benchmark \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset gsm8k \
  --limit 100 \
  --output results/reasoning_benchmark.json
```

---

### halo-forge reasoning datasets

List available math/reasoning datasets.

```bash
halo-forge reasoning datasets
```

---

## Agentic Commands (Experimental)

### halo-forge agentic sft

SFT training for tool calling models.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-7B-Instruct` | Base model |
| `--dataset` | `-d` | string | No | `xlam_sft` | SFT dataset name |
| `--max-samples` | - | int | No | - | Limit training samples |
| `--output` | `-o` | path | No | `models/agentic_sft` | Output directory |
| `--epochs` | - | int | No | 2 | Number of epochs |
| `--dry-run` | - | flag | No | false | Validate config only |

**SFT Dataset choices:** `xlam_sft`, `glaive_sft`

```bash
halo-forge agentic sft --dataset xlam_sft --model Qwen/Qwen2.5-7B-Instruct --output models/agentic_sft
```

---

### halo-forge agentic train

Train on tool calling datasets with RAFT.
Real-training enabled. Use `--allow-prototype-train` only if a modality is temporarily re-gated.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-7B-Instruct` | Base model |
| `--dataset` | `-d` | string | No | `xlam` | RAFT Dataset: xlam, glaive |
| `--output` | `-o` | path | No | `models/agentic_raft` | Output directory |
| `--cycles` | - | int | No | 5 | RAFT cycles |
| `--lr` | - | float | No | 5e-5 | Learning rate |
| `--lr-decay` | - | float | No | 0.85 | LR decay per cycle |
| `--limit` | - | int | No | - | Limit dataset samples |
| `--dry-run` | - | flag | No | false | Validate config only |
| `--allow-prototype-train` | - | flag | No | false | Compatibility override for temporary prototype gating |
| `--resume-from-cycle` | - | int | No | 0 | Resume from a previously saved cycle index |
| `--seed` | - | int | No | 42 | Deterministic runtime seed |
**Supported model families:** `qwen2.5`, `qwen2`, `qwen`, `llama-3`, `llama3`, `mistral`

```bash
halo-forge agentic train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset xlam \
  --cycles 5 \
  --output models/agentic_raft \
  --seed 42
```

---

### halo-forge agentic benchmark

Benchmark tool calling accuracy.

| Flag | Short | Type | Required | Default | Description |
|------|-------|------|----------|---------|-------------|
| `--model` | `-m` | string | No | `Qwen/Qwen2.5-7B-Instruct` | Model to benchmark |
| `--dataset` | `-d` | string | No | `xlam` | Dataset: xlam, glaive |
| `--limit` | - | int | No | 100 | Limit samples |
| `--output` | `-o` | path | No | - | Output file |

```bash
halo-forge agentic benchmark \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset xlam \
  --limit 100 \
  --output results/agentic_benchmark.json
```

---

### halo-forge agentic datasets

List available tool calling datasets.

```bash
halo-forge agentic datasets
```

Output:
```
Available Agentic / Tool Calling Datasets
============================================================
  xlam         [Tool Calling] - 60k verified, 3,673 APIs
  glaive       [Tool Calling] - 113k samples, irrelevance
  toolbench    [Tool Calling] - 188k samples, 16k APIs
  hermes       [Tool Calling] - Format reference
```

---

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Configuration error |
| 4 | GPU not available |
| 5 | Verification failed |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PYTORCH_HIP_ALLOC_CONF` | ROCm memory configuration |
| `HF_HOME` | HuggingFace cache directory |
| `CUDA_VISIBLE_DEVICES` | GPU selection |
| `HIP_VISIBLE_DEVICES` | AMD GPU selection |

---

## Quick Reference

### Most Common Commands

```bash
# Test installation
halo-forge test --level smoke

# Train with RAFT
halo-forge raft train --prompts data/prompts.jsonl --verifier mingw --cycles 6

# Benchmark model
halo-forge benchmark run --model models/raft/cycle_6 --prompts data/test.jsonl

# Show info
halo-forge info
```

### Verifier Quick Reference

| Verifier | Language | Cross-compile | Requires |
|----------|----------|---------------|----------|
| `gcc` | C/C++ | No | gcc installed |
| `mingw` | C/C++ | Yes (Windows PE) | mingw-w64 |
| `msvc` | C/C++ | Yes (Windows) | SSH to Windows |
| `rust` | Rust | Yes (Windows) | rustc, cargo |
| `go` | Go | Yes (Windows) | go installed |
| `dotnet` | C# | Yes (Windows PE) | dotnet-sdk |
| `powershell` | PowerShell | No | pwsh |
| `auto` | Multi-lang | Varies | Depends on detected |
