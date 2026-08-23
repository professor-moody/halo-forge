<p align="center">
  <img src="halo-forge.png" alt="halo forge logo" width="350">
</p>

<h1 align="center">halo forge</h1>

<p align="center">
  Cross-vendor local finetuning workstation.<br>
  SFT · CPT · DPO · GRPO · RAFT, with verifier-grounded rewards and runtime-qualified accelerator paths.
</p>

<p align="center">
  <a href="https://halo-forge.io/docs/getting-started/hardware/"><img src="https://img.shields.io/badge/Backend_matrix-✓-0A66C2?style=for-the-badge" alt="Backend matrix"></a>
  <a href="https://halo-forge.io/docs/training-pipeline/methods/"><img src="https://img.shields.io/badge/Trainers-✓-0A192F?style=for-the-badge" alt="Trainers"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Apache_2.0-License-blue?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AMD_Strix_Halo-Optimized-ED1C24?style=flat-square&logo=amd&logoColor=white" alt="Strix Halo">
  <img src="https://img.shields.io/badge/Apple_Silicon-MPS_+_MLX_native-000000?style=flat-square&logo=apple&logoColor=white" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/CUDA-vLLM_+_TRL-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/Python-3.10--3.13-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
</p>

---

## What halo-forge is

A workstation tool that takes a base model and turns it into a finetuned,
evaluated, served artifact without leaving the machine you control. The command
shape is consistent across ROCm, CUDA, Apple MPS, and Apple MLX; guided ROCm
and CUDA execution additionally requires a pinned managed runtime to pass real
hardware qualification. The runtime
capability check remains authoritative because algorithms, optional packages,
model adapters, and precision support are not identical on every backend.

For your own data, Halo Forge recommends one compatible model and training
plan, measures whether it fits the workstation, and asks for confirmation
before a proof run. Experienced operators can still choose the algorithm,
verifier, backend, and low-level settings directly.

```bash
# Strix Halo / RTX 4090 / Apple M-series — same commands.
halo-forge sft train   --dataset codealpaca --model Qwen/Qwen2.5-Coder-3B
halo-forge cpt train   --dataset-version <version-id> --model Qwen/Qwen2.5-1.5B --adaptation lora
halo-forge dpo train   --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct
halo-forge grpo train  --data prompts.jsonl --verifier execution --num-generations 8

halo-forge eval        --model ./models/sft/final_model --tasks core
halo-forge merge       --base Qwen/Qwen2.5-3B-Instruct --adapter ./my-lora --output ./shipped
halo-forge convert     --source ./shipped --format hf --quant bf16 --output ./shipped-bf16 --verify
halo-forge serve       --model ./shipped
```

## Capabilities

### Trainers
- **SFT** — supervised finetuning with QLoRA / LoRA / DoRA / rsLoRA / PiSSA. PyTorch on every torch backend; MLX-native on Apple Silicon.
- **CPT** — causal next-token continued pretraining over immutable document
  corpora, with explicit LoRA or full adaptation, exact tokenizer-aware
  packing, and PyTorch or native MLX execution where the active runtime
  verifies it.
- **DPO** — preference optimization (sigmoid / IPO / hinge / KTO-pair / RPO / cDPO). PyTorch via TRL; MLX-native reference-free DPO.
- **GRPO** — verifier-grounded policy gradient (DeepSeek-R1 / Tülu 3 family). PyTorch via TRL; MLX-native reference-free GRPO.
- **RAFT** — rejection-sampling RLVR with curriculum + reward shaping. PyTorch + native MLX.
- **Optimizers**: AdamW (default), AdamW8bit, Lion, paged variants (bnb-backed where the platform supports it).

### Verifiers
Pluggable registry — drop a `.py` in `~/.halo-forge/verifiers/` or use `@register_verifier` decoration. 24 short names ship out of the box.

**18 work directly as `--verifier <name>`** (they construct with no arguments):
- **Execution & compile**: `gcc`, `clang`, `mingw`, `execution`, `gcc_execution`, `mingw_execution`, `clang_execution`, `pytest`, `unittest`, `humaneval`, `mbpp`, `rust`, `cargo`, `go`, `custom`
- **Schema** (V3): `json_structure`, `json_schema`
- **LLM-as-judge** (V2): `llm_judge` — rubric-graded with any local or hosted judge model

`humaneval` and `mbpp` read their bundled task files from `data/rlvr/` **relative to the working
directory**, so run them from the repo root or bind an explicit dataset path through a verifier
profile; everywhere else they fail at construction.

**Need constructor arguments**, so a bare `--verifier <name>` fails — bind them through an immutable verifier profile instead:
- **Reference metrics** (V4): `bleu`, `rouge`, `chrf` — require reference text
- **Format** (V3): `regex_format` — requires a pattern
- **Test/dataset runners**: `rlvr_pytest` — requires a dataset path; `subprocess` — requires a command

```bash
halo-forge verifier profile create --name strict-json \
  --implementation regex_format \
  --configuration '{"pattern": "^\\{.*\\}$"}'
halo-forge grpo train --data prompts.jsonl --verifier-profile-revision <revision-id>
```

### Data pipeline
- **Dataset Lab**: immutable local/Hugging Face dataset versions for text, VLM,
  and audio, with ordered recipes, previews, jobs, and direct training handoff.
  In the dashboard choose **Train on your data** to select a verified scenario,
  map fields, preview validation, and publish a version without moving source
  files. The CLI equivalent starts with `halo-forge data inspect`, followed by
  scenario-backed `data add`, `data build`, and `data versions`.
- **Corpus adaptation**: extract visible text from local text, Markdown, HTML,
  text-layer PDF, DOCX, or structured sources; review failures and provenance;
  publish an immutable corpus version; then render and launch CPT without
  hand-converting documents. Start in **Train on your data** or use
  `halo-forge data extract`.
- **Dedup** (D2): `halo-forge data dedup --method exact|fuzzy --threshold 0.85`
- **Quality scoring** (D3): `halo-forge data score --threshold 0.5` or `--top-k-pct 0.5`
- **Format converters + previewers**: see [`halo_forge/data/`](halo_forge/data/)
- **Synthetic generation**: `halo-forge data synthesize`, or a Dataset Lab
  synthesis step using Ollama or an OpenAI-compatible teacher.
- **Human Feedback and Active Data Studio**: deterministic development-data
  proposals, text/preference/tool/VLM/audio review, optional blinded second
  passes, immutable label sets, and reviewed child Dataset Lab versions. Start
  with `halo-forge review acquire create` or open **Data -> Review Queues**.
- **Guided training plan and capacity coach**: one explainable plan from the
  immutable dataset and active workstation, explicit model preparation, a
  disposable capacity check, and one **Start proof run** action. Use the
  dashboard or begin with `halo-forge train-plan recommend`.

### Inference + serving
- **OpenAI-compatible serving** (I1): `halo-forge serve --model X` — `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- **Unified convert** (I5): `halo-forge convert --format mlx|gguf|hf --quant q4|q8|fp16|bf16|fp32`.
  `--format hf` is a dtype recast only, so it accepts `bf16|fp16|fp32`; `--format gguf` needs a
  local llama.cpp checkout (see [Installation profiles](#installation-profiles)).
- **Round-trip verify** (I4): `halo-forge convert --verify` — catches silently-broken exports and
  **exits nonzero** when the export diverges from the source. Supported for `mlx` and `hf` targets;
  GGUF has no loading adapter yet and reports that explicitly.
- **Artifact Studio**: one content-addressed library for trained checkpoints,
  adapters, merged/converted variants, qualification evidence, promotion,
  managed serving, portable export, and reviewed seven-day-trash cleanup.
  Start with `halo-forge artifact list` or open **Models** in the dashboard.
- **vLLM rollout** (I6): `halo-forge raft train --rollout-engine vllm` — continuous-batched generation on CUDA/ROCm
- **MLX rollout** (I6.1): `halo-forge raft train --rollout-engine mlx` — Apple Silicon equivalent

### Evaluation
- **lm-evaluation-harness** (V8): `halo-forge eval --tasks core` — MMLU, GSM8K, HumanEval, IFEval, ARC, …
- Curated task groups: `core`, `reasoning`, `code`, `instruction_following`, `knowledge`
- **Reward Integrity and Training Signal Studio**: capture the exact retained
  outputs used by verifier-guided training, rescore those same outputs with an
  independent qualified sentinel, inspect paired evidence, and review a
  continue/stop/fork gate. Start with `halo-forge reward capabilities`.
- **Adaptive Training and Evidence Studio**: immutable checkpoint policies,
  trainer-compatible step/cycle gates, matched-seed bootstrap analysis,
  reviewed research decisions, longitudinal drift, and reproducible evidence
  bundles. Start with `halo-forge checkpoint-policy create`, then
  `sweep checkpoints`, `sweep analyze`, and `sweep report`.

### Reproducibility
- **Replay manifests v12** (T15): `halo-forge replay <run_dir>` — capture run,
  dataset, trainer-artifact, verifier, reward-system, signal-trace, audit, seed,
  corpus/tokenizer/packing identity, training-plan and capacity evidence,
  configuration, and runtime identity, then reconstruct the launch command.
- **Evidence bundles**: atomic Markdown, HTML, JSON, CSV, and SVG reports with
  dataset, suite, run, checkpoint, runtime, assumption, and missing-evidence
  identity under `~/.halo-forge/research/evidence/`.

### Run management
- **SQLite run database** (F-G) — search/filter/sort/paginate runs by modality, model, status, eval-presence, time
- **Multi-run comparison** (F) — pin runs, overlay loss + reward curves, side-by-side config diff
- **Cost rollup** (P2/F-R) — per-run kWh + $ estimate from wall-clock × backend nominal power
- **Live telemetry strip** — SSE-streamed GPU util / VRAM / power / throughput
- **Run cancellation** — graceful SIGTERM with checkpoint save

### Adapter merging
- **Bake** (T13): `halo-forge merge --mode bake --base X --adapter Y` — single LoRA into base
- **Combine** (T12): `halo-forge merge --mode combine --adapters a,b,c --method dare_ties` — N adapters into one (linear / ties / dare_linear / dare_ties / magnitude_prune)

## Backend matrix

Authoritative coverage is in the [Hardware Notes](https://halo-forge.io/docs/getting-started/hardware/).
This table is deliberately conservative; a green host does not make every
model, trainer, quantizer, or evaluator compatible.

| Runtime | Training truth | Important limits |
|---|---|---|
| CUDA + PyTorch | Primary torch path; capability-checked trainers and vLLM where installed | QLoRA/bitsandbytes and vLLM require their optional packages and supported GPUs |
| ROCm + PyTorch | Torch trainers; Strix Halo has explicit unified-memory defaults | vLLM and bitsandbytes-on-ROCm remain environment-dependent and are never assumed by preflight |
| Apple MPS + PyTorch | Torch training for methods whose model adapter supports MPS | No bitsandbytes QLoRA; some operators or third-party models can fall back or fail preflight |
| Apple MLX | Native SFT, CPT, RAFT, DPO, and GRPO paths | MLX is not a blanket backend for VLM, audio, reasoning, or agentic trainers; advanced PEFT flags are not silently emulated |
| CPU | Functional checks, metadata work, and tiny smoke runs | Not presented as a practical heavy-training target |

Missing telemetry stays **unavailable**, never zero. The selected trainer,
model, installed extras, memory/disk preflight, and backend capability
descriptor decide whether a managed launch is ready.

## Quick start

```bash
git clone https://github.com/professor-moody/halo-forge.git
cd halo-forge
pip install -e ".[dev]"

# Minimal SFT smoke test
halo-forge sft train \
  --dataset codealpaca \
  --model Qwen/Qwen2.5-Coder-0.5B \
  --max-samples 100 \
  --epochs 1

# DPO with the LLM-judge verifier
halo-forge dpo train \
  --dataset ultrafeedback \
  --model Qwen/Qwen2.5-3B-Instruct \
  --beta 0.1 --loss-type sigmoid

# GRPO with code execution verifier + 8-generation groups
halo-forge grpo train \
  --data prompts.jsonl \
  --model Qwen/Qwen2.5-Coder-3B \
  --verifier execution --num-generations 8 \
  --rollout-engine vllm   # or 'mlx' on Apple Silicon

# Score + filter your dataset
halo-forge data dedup --input raw.jsonl --output deduped.jsonl --method fuzzy
halo-forge data score --input deduped.jsonl --output clean.jsonl --top-k-pct 0.5

# Adapt a causal model to a reviewed document corpus
halo-forge data inspect --path ./manuals --scenario corpus-adaptation
halo-forge data build <dataset-id> --recommended-recipe
halo-forge cpt train --dataset-version <version-id> \
  --model Qwen/Qwen2.5-1.5B --adaptation lora \
  --budget-mode passes --corpus-passes 1
```

## Installation profiles

```bash
pip install -e .                # core
pip install -e ".[mlx]"         # Apple Silicon native
pip install -e ".[inference]"   # bitsandbytes quantization + tokenizer/model deps
pip install -e ".[dev]"         # tests + linting
pip install -e ".[all]"         # everything
```

Optional dev integrations are lazy-imported — install only what you use:

```bash
pip install datasketch          # for fuzzy dedup
pip install jsonschema          # for the json_schema verifier
pip install sacrebleu rouge_score  # for BLEU/ROUGE/chrF verifiers
pip install lm-eval             # for halo-forge eval
pip install vllm                # for --rollout-engine vllm (CUDA/ROCm)
```

GGUF export is not pip-installable from here: `halo-forge convert --format gguf`
shells out to llama.cpp's `convert_hf_to_gguf.py`. Clone llama.cpp to
`~/llama.cpp`, `/opt/llama.cpp`, or `./llama.cpp` first; without it the command
fails immediately with install instructions rather than after loading weights.

## Workstation surfaces

Halo Forge exposes the same managed datasets, launch preflight, durable work,
runs, evaluations, and artifacts through four equal operator surfaces. They
share one SQLite catalog and service layer; choosing a surface does not create
a second product or data format.

| Surface | Best for | File selection and access |
|---|---|---|
| Desktop shell | Local interactive work on a supported desktop build | Native file/folder chooser; paths refer to the workstation running Halo Forge |
| Local browser | The normal source/CLI installation on the same workstation | Browser upload and explicit workstation paths |
| Remote browser | Operating a trusted workstation from another machine | Same workflows over token-authenticated API; client-local paths are not silently treated as host paths |
| CLI | Automation, headless work, and reproducible scripts | Explicit paths and flags; managed dataset/version commands use the same catalog |

The React app lives in [`public_app/`](public_app/). Desktop is a thin Tauri
shell around that same app, not a separate implementation.

Start the app:

```bash
halo-forge dashboard
```

Open `http://127.0.0.1:8000`. The `dashboard` command serves the FastAPI public API and the built React dashboard from one origin. If `public_app/dist` is missing in a source checkout, it builds the dashboard assets first. `halo-forge app` is the same command.

For a no-bind startup check:

```bash
halo-forge dashboard --check
```

For trusted-network remote access, run the app on the workstation and open the same port from the other machine:

```bash
halo-forge token create dashboard
halo-forge dashboard --host 0.0.0.0 --port 8000
```

Open `http://<workstation-host>:8000` and paste the `hfk_...` token in **Connection**.

For frontend development, run the API and Vite separately:

```bash
# Terminal 1: dashboard API
halo-forge serve-public

# Terminal 2: React app
cd public_app
npm install
npm run dev
```

Open `http://127.0.0.1:3000`. Vite proxies `/api/*` to the dashboard API at `http://127.0.0.1:8000`.

For repeatable screenshot QA:

```bash
cd public_app
npm run qa:visual
```

Serve a local model from the dashboard:

1. Open **Models -> Catalog** and click **Serve** on a small model.
2. Open **Models -> Serve & Test** and wait for **Local serving** to show `ready`.
3. Chat with the model, then click **Stop** before serving a different one.

Desktop app development starts from [`apps/desktop-tauri/`](apps/desktop-tauri/).
Browser and CLI installs are supported on macOS, Linux, and Windows. The desktop
build matrix produces macOS arm64 DMG, Linux x86-64 AppImage/deb, and Windows
x86-64 NSIS candidates with the same bundled-runtime health and proof-run smoke
contract. The application and download documentation read the verified release
manifest before recommending an installer. Unsigned candidates remain preview
artifacts: macOS prereleases may expose an unmistakably named, checksummed
developer preview, while stable releases and the normal download path never
offer an unsigned DMG. Normal users are never instructed to bypass Gatekeeper
or Windows SmartScreen. Removing the desktop shell does not remove
`~/.halo-forge` data.

The public app surfaces:

- **Telemetry strip** — live backend metrics where the host exposes them;
  unavailable utilization, memory, power, or throughput remains unavailable
- **Train** — guided and advanced launch modes with safe catalog defaults and preflight
- **Data** — immutable multimodal versions, recipes, quality review, and trainer handoff
- **Review Queues** — acquisition proposals, accessible multimodal labeling,
  adjudication, label-set publication, and explicit child-version handoff
- **Training audits** — same-output optimizer/sentinel evidence, boundary
  trends, exact decision reasons, and reviewed pause actions
- **Experiments** — deterministic repeats/sweeps with objective and budget review
- **Advanced training launcher** — direct SFT / RAFT configurator
- **Live run view** — cycle-by-cycle loss + reward charts, scrubber, log tail, sample inspector, cancel button
- **Multi-run comparison** — pin up to 6 runs, overlay loss/reward, side-by-side config diff
- **Run search** — DB-backed filter chips for modality / status / model / has-eval
- **Models / Artifact Studio** — transform, qualify, compare, promote, serve/test,
  export, and inspect artifact lineage without entering raw IDs
- **Activity Center** — the global queue, current resource owner, blockers,
  progress, telemetry, logs, retries, and disk forecast
- **Energy & spend card** — kWh + $/kWh per run
- **Remote workstation connection** — paste a `halo-forge token create dashboard` token when accessing a non-loopback host

## Documentation

The canonical user documentation lives at **[halo-forge.io/docs](https://halo-forge.io/docs/)**. The repo-local `docs/` directory is kept for release, engineering, and offline reference artifacts.

| | |
|---|---|
| [Quick Start](https://halo-forge.io/docs/getting-started/quickstart/) | Install, launch the dashboard, and run the first safe training job |
| [Choose a Model](https://halo-forge.io/docs/getting-started/choose-a-model/) | Catalog guidance by goal, backend, memory tier, and first-run risk |
| [Training Methods](https://halo-forge.io/docs/training-pipeline/methods/) | SFT / RAFT / DPO / ORPO / RM / GRPO / VLM / audio / reasoning / agentic |
| [Dashboard Training](https://halo-forge.io/docs/reference/dashboard-training/) | Goal-first dashboard launch workflow |
| [Hardware Notes](https://halo-forge.io/docs/getting-started/hardware/) | Per-backend recommendations + feature matrix |
| [Verifiers](https://halo-forge.io/docs/verifiers/) | Execution, compile, schema, metric, judge, and custom reward plugins |
| [Serving](https://halo-forge.io/docs/serving/) | Dashboard-managed serving, OpenAI-compatible endpoint, conversion, export |
| [Review Studio](https://halo-forge.io/docs/review-studio/) | Deterministic proposals, human review, immutable labels, and explicit Dataset Lab handoff |
| [Reward Integrity](https://halo-forge.io/docs/reward-integrity/) | Exact training-signal capture, same-output sentinel audits, and reviewed gates |
| [Guided Operational Completion](docs/guided-operational-completion.md) | Plain-language proof outcomes, guided studies, grounded examples, specialized tasks, and local environments |
| [Guided Training Plans](docs/GUIDED_TRAINING_PLAN.md) | Explainable recommendations, model preparation, disposable capacity checks, and proof-run handoff |
| [Command Index](https://halo-forge.io/docs/reference/command-index/) | CLI commands, flags, token operations, and test profiles |
| [Troubleshooting](https://halo-forge.io/docs/reference/troubleshooting/) | Backend setup, launch failures, auth, model access, and training issues |

## License

Copyright 2025 Halo Forge Labs LLC. Licensed under Apache 2.0. See [LICENSE](LICENSE).

## Acknowledgments

- AMD for Strix Halo hardware
- Apple for MLX
- HuggingFace for `transformers` / `peft` / `trl` / `datasets`
- EleutherAI for `lm-evaluation-harness`
- The DeepSeek / Tülu / Open-R1 / RAFT authors for foundational RLVR recipes
- [kyuz0](https://github.com/kyuz0/amd-strix-halo-llm-finetuning) for the original Strix Halo finetuning toolbox
- TheRock for ROCm nightlies
