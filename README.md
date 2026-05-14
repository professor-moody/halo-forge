<p align="center">
  <img src="halo-forge.png" alt="halo forge logo" width="350">
</p>

<h1 align="center">halo forge</h1>

<p align="center">
  Cross-vendor local finetuning workstation.<br>
  SFT · DPO · GRPO · RAFT, with verifier-grounded rewards, on ROCm · CUDA · Apple MLX · Apple MPS.
</p>

<p align="center">
  <a href="docs/HARDWARE_NOTES.md"><img src="https://img.shields.io/badge/Backend_matrix-✓-0A66C2?style=for-the-badge" alt="Backend matrix"></a>
  <a href="docs/TRAINERS.md"><img src="https://img.shields.io/badge/Trainers-✓-0A192F?style=for-the-badge" alt="Trainers"></a>
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

A workstation tool that takes a base model and turns it into a finetuned, evaluated, served artifact — without leaving the local machine. The single thing that makes it different from every adjacent project (axolotl, llama-factory, unsloth, mlx-lm-lora, torchtune): **it runs natively on every modern accelerator**, not just CUDA.

Pick a model. Pick an algorithm. Pick a verifier. Pick a backend. Train. Evaluate. Serve.

```bash
# Strix Halo / RTX 4090 / Apple M-series — same commands.
halo-forge sft train   --dataset codealpaca --model Qwen/Qwen2.5-Coder-3B
halo-forge dpo train   --dataset ultrafeedback --model Qwen/Qwen2.5-3B-Instruct
halo-forge grpo train  --data prompts.jsonl --verifier execution --num-generations 8

halo-forge eval        --model ./models/sft/final_model --tasks core
halo-forge merge       --base Qwen/Qwen2.5-3B-Instruct --adapter ./my-lora --output ./shipped
halo-forge convert     --source ./shipped --format gguf --quant q4 --output ./out.gguf --verify
halo-forge serve       --model ./shipped
```

## Capabilities

### Trainers
- **SFT** — supervised finetuning with QLoRA / LoRA / DoRA / rsLoRA / PiSSA. PyTorch on every torch backend; MLX-native on Apple Silicon.
- **DPO** — preference optimization (sigmoid / IPO / hinge / KTO-pair / RPO / cDPO). PyTorch via TRL; MLX-native reference-free DPO.
- **GRPO** — verifier-grounded policy gradient (DeepSeek-R1 / Tülu 3 family). PyTorch via TRL; MLX-native reference-free GRPO.
- **RAFT** — rejection-sampling RLVR with curriculum + reward shaping. PyTorch + native MLX.
- **Optimizers**: AdamW (default), AdamW8bit, Lion, paged variants (bnb-backed where the platform supports it).

### Verifiers
Pluggable registry — drop a `.py` in `~/.halo-forge/verifiers/` or use `@register_verifier` decoration. 18 short names ship out of the box:
- **Execution & compile**: `gcc`, `clang`, `mingw`, `execution`, `gcc_execution`, `mingw_execution`, `clang_execution`, `pytest`, `unittest`, `rlvr_pytest`, `humaneval`, `mbpp`, `rust`, `cargo`, `go`, `custom`, `subprocess`
- **Schema & format** (V3): `json_structure`, `json_schema`, `regex_format`
- **Reference metrics** (V4): `bleu`, `rouge`, `chrf`
- **LLM-as-judge** (V2): `llm_judge` — rubric-graded with any local or hosted judge model

### Data pipeline
- **Dedup** (D2): `halo-forge data dedup --method exact|fuzzy --threshold 0.85`
- **Quality scoring** (D3): `halo-forge data score --threshold 0.5` or `--top-k-pct 0.5`
- **Format converters + previewers**: see [`halo_forge/data/`](halo_forge/data/)
- **Synthetic generation**: roadmap (D1)

### Inference + serving
- **OpenAI-compatible serving** (I1): `halo-forge serve --model X` — `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- **Unified convert** (I5): `halo-forge convert --format mlx|gguf|hf --quant q4|q8|fp16|bf16|fp32`
- **Round-trip verify** (I4): `halo-forge convert --verify` — catches silently-broken exports
- **vLLM rollout** (I6): `halo-forge raft train --rollout-engine vllm` — continuous-batched generation on CUDA/ROCm
- **MLX rollout** (I6.1): `halo-forge raft train --rollout-engine mlx` — Apple Silicon equivalent

### Evaluation
- **lm-evaluation-harness** (V8): `halo-forge eval --tasks core` — MMLU, GSM8K, HumanEval, IFEval, ARC, …
- Curated task groups: `core`, `reasoning`, `code`, `instruction_following`, `knowledge`

### Reproducibility
- **Replay manifests** (T15): `halo-forge replay <run_dir>` — capture every input (seed, dataset hash, env fingerprint, full config) at run launch, regenerate the launch command from any output directory.

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

Authoritative coverage at [`docs/HARDWARE_NOTES.md`](docs/HARDWARE_NOTES.md). High-level: every shipped feature works on every shipped backend, with these clearly-flagged exceptions:

| | rocm_gfx1151 | cuda | mps | mlx | cpu |
|---|---|---|---|---|---|
| **Trainers** | ✅ all | ✅ all | ✅ all | ✅ SFT/RAFT/DPO/GRPO¹ | ✅ tiny only |
| **vLLM rollout** | ⚠️ experimental | ✅ | ❌ typed err | ❌ typed err | ❌ typed err |
| **MLX rollout** | ❌ | ❌ | ⚠️ unusual | ✅ | ❌ |
| **QLoRA training** | ⚠️ bnb-rocm | ✅ | ❌ | ❌ | ❌ |
| **DoRA / PiSSA** | ✅ | ✅ | ✅ | ❌² | ✅ |
| **bnb optimizers** | ⚠️ bnb-rocm | ✅ | ❌ | ❌ | ❌ |
| **OpenAI serve** | ✅ | ✅ | ✅ | ✅ | ✅ |

1. MLX DPO supports sigmoid, IPO, hinge, and KTO-pair in reference-free and reference-model modes. MLX GRPO supports reference-free and reference-model eager single-cycle updates.
2. PEFT additions (DoRA / rsLoRA / PiSSA) are peft-only; mlx-lm.tuner ships LoRA. Setting these flags on MLX prints a loud warning at trainer init.

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
```

## Installation profiles

```bash
pip install -e .                # core
pip install -e ".[mlx]"         # Apple Silicon native
pip install -e ".[inference]"   # quantization + GGUF export tooling
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

## Web UI

A Vite + React 19 + Tanstack Router frontend lives in [`public_app/`](public_app/). It is the user-facing local or remote workstation surface for guided launch, monitoring, results, model selection, docs, and token-based remote access.

Run locally with two terminals:

```bash
# Terminal 1: dashboard API
halo-forge serve-public

# Terminal 2: React app
cd public_app
npm install
npm run dev
```

Open `http://127.0.0.1:3000`. Vite proxies `/api/*` to the dashboard API at `http://127.0.0.1:8000`.

For a no-bind startup check:

```bash
halo-forge serve-public --check
```

For trusted-network remote access, run both processes on the workstation and open the app port from the other machine:

```bash
halo-forge token create dashboard
halo-forge serve-public --host 0.0.0.0 --port 8000

cd public_app
npm install
npm run dev -- --host 0.0.0.0
```

Open `http://<workstation-host>:3000` and paste the `hfk_...` token in **Connection**.

For repeatable screenshot QA:

```bash
cd public_app
npm run qa:visual
```

The public app surfaces:

- **Telemetry strip** — live GPU util / VRAM / power / throughput across MPS / MLX / ROCm / CUDA
- **Start flow** — guided first run with safe catalog defaults and preflight
- **Advanced training launcher** — direct SFT / RAFT configurator
- **Live run view** — cycle-by-cycle loss + reward charts, scrubber, log tail, sample inspector, cancel button
- **Multi-run comparison** — pin up to 6 runs, overlay loss/reward, side-by-side config diff
- **Run search** — DB-backed filter chips for modality / status / model / has-eval
- **Energy & spend card** — kWh + $/kWh per run
- **Remote workstation connection** — paste a `halo-forge token create dashboard` token when accessing a non-loopback host

## Documentation

| | |
|---|---|
| [`docs/HARDWARE_NOTES.md`](docs/HARDWARE_NOTES.md) | Per-backend recommendations + feature × backend matrix |
| [`docs/TRAINERS.md`](docs/TRAINERS.md) | SFT / DPO / GRPO / RAFT — choosing, configuring, comparing |
| [`docs/VERIFIERS.md`](docs/VERIFIERS.md) | Verifier ecosystem + plugin authoring |
| [`docs/DATA.md`](docs/DATA.md) | Dedup, quality scoring, format conventions |
| [`docs/EVAL.md`](docs/EVAL.md) | lm-eval integration, curated task groups |
| [`docs/SERVING.md`](docs/SERVING.md) | OpenAI-compatible endpoint, conversion, round-trip verify |
| [`docs/REPLAY.md`](docs/REPLAY.md) | Deterministic replay manifests |
| [`docs/MLX.md`](docs/MLX.md) | Apple Silicon specifics |
| [`docs/MODELS.md`](docs/MODELS.md) | Curated model recommendations per task |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Internal architecture |

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
