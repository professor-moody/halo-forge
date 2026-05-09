---
title: "Documentation"
description: "Cross-vendor local finetuning workstation — SFT, DPO, GRPO, RAFT, RM with verifier-grounded rewards on ROCm, CUDA, Apple MLX, Apple MPS."
---

## What halo-forge is

A workstation tool that takes a base model and turns it into a finetuned, evaluated, served artifact — without leaving the local machine.

The single thing that makes it different from every adjacent project (axolotl, llama-factory, unsloth, mlx-lm-lora, torchtune): **it runs natively on every modern accelerator**, not just CUDA.

Pick a goal. Choose a catalog model. Pick an algorithm and verifier. Train, evaluate, serve, and save the run into a bundle when it is worth comparing.

## Start By Intent

| I want to... | Start here |
|---|---|
| Train my first local model | [Quick Start](/docs/getting-started/quickstart/) |
| Pick the right base model | [Choose a Model](/docs/getting-started/choose-a-model/) |
| See runnable examples | [Usage Scenarios](/docs/getting-started/scenarios/) |
| Run on Apple Silicon | [Hardware Notes](/docs/getting-started/hardware/) and [MLX notes](/docs/serving/) |
| Serve or export a trained artifact | [Serve / convert / merge](/docs/serving/) |

## Capabilities

### Trainers

- **SFT** — supervised finetuning with QLoRA / LoRA / DoRA / rsLoRA / PiSSA. PyTorch on every torch backend; MLX-native on Apple Silicon.
- **DPO** — preference optimization (sigmoid / IPO / hinge / KTO-pair / RPO / cDPO). PyTorch via TRL; MLX-native reference-free DPO.
- **GRPO** — verifier-grounded policy gradient (DeepSeek-R1 / Tülu 3 family). PyTorch via TRL; MLX-native reference-free GRPO.
- **RAFT** — rejection-sampling RLVR with curriculum + reward shaping. PyTorch + native MLX.
- **Reward Model** — Bradley-Terry RM from preference pairs. Becomes a learned verifier for any other modality.

### Verifiers

Pluggable registry — drop a `.py` in `~/.halo-forge/verifiers/` or use `@register_verifier`. Out of the box:

- **Execution & compile**: `gcc`, `clang`, `mingw`, `execution`, `pytest`, `humaneval`, `mbpp`, `rust`, `cargo`, `go`, `custom`, `subprocess`
- **Schema & format**: `json_structure`, `json_schema`, `regex_format`
- **Reference metrics**: `bleu`, `rouge`, `chrf`
- **LLM-as-judge**: `llm_judge` — rubric-graded with any local or hosted judge model

### Data pipeline

- **Synthesize** — generate completions from seed prompts via a teacher model + verifier filter.
- **Dedup** — exact (SHA-256) + fuzzy (MinHash + LSH).
- **Score** — heuristic quality scoring + threshold / top-K filter.
- **Compose** — `synthesize → dedup → score → filter` is the four-command pre-finetune sequence.

### Inference + serving

- **OpenAI-compatible serving** — `halo-forge serve --model X` exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`.
- **Unified convert** — `halo-forge convert --format mlx|gguf|hf --quant q4|q8|fp16|bf16|fp32`
- **Round-trip verify** — `halo-forge convert --verify` catches silently-broken exports.
- **vLLM rollout** — continuous-batched generation on CUDA/ROCm.
- **MLX rollout** — Apple Silicon equivalent via `mlx_lm.generate`.

### Evaluation

- **lm-evaluation-harness** — `halo-forge eval --tasks core` runs MMLU / GSM8K / HumanEval / IFEval / ARC etc.
- **Mid-training probe** — `halo-forge probe` runs a small held-out benchmark and diffs against a baseline; catches catastrophic forgetting in single-digit minutes.

### Reproducibility

- **Replay manifests** — `halo-forge replay <run_dir>` regenerates the exact launch command.
- **Sweep infrastructure** — Optuna-style hyperparameter search with random / TPE / grid samplers.

### Run management

- **SQLite run database** — search / filter / sort / paginate runs.
- **Multi-run comparison** — pin runs, overlay loss + reward curves, side-by-side config diff.
- **Cohort eval dashboard** — runs × tasks grid; best-per-task highlighted.
- **Cost rollup** — per-run kWh + $ from wall-clock × backend nominal power.
- **Live telemetry strip** — SSE-streamed GPU util / VRAM / power / throughput.

### Adapter merging

- **Bake** — single LoRA into base, output is a standard HF checkpoint.
- **Combine** — N adapters via `linear` / `ties` / `dare_linear` / `dare_ties` / `magnitude_prune`.

### Auth + multi-user

- **API tokens** — bearer-token auth, automatic when bound to non-loopback. Local-first stays zero-config.

## Quick navigation

### Getting started
- [Quick Start](/docs/getting-started/quickstart/) — Install + first run
- [Choose a Model](/docs/getting-started/choose-a-model/) — Model catalog, Liquid AI caveats, and first picks
- [Usage Scenarios](/docs/getting-started/scenarios/) — Code, preference, reasoning, VLM, audio, agentic, serve/export
- [Hardware Notes](/docs/getting-started/hardware/) — Per-backend recommendations + feature matrix

### Trainers
- [Overview](/docs/trainers/) — Choosing between SFT / DPO / GRPO / RAFT / RM

### Verifiers
- [Plugin registry + ecosystem](/docs/verifiers/)
- [Execution + compile](/docs/verifiers/execution/)
- [Schema + format](/docs/verifiers/schema/)
- [Reference metrics](/docs/verifiers/metrics/)
- [LLM-as-judge](/docs/verifiers/llm-judge/)
- [Multi-language](/docs/verifiers/multi-language/)
- [Custom verifiers](/docs/verifiers/custom/)

### Data pipeline
- [Overview](/docs/data/)

### Evaluation
- [lm-eval + mid-training probe](/docs/eval/)

### Inference + serving
- [Serve / convert / merge](/docs/serving/)

### Reproducibility
- [Replay manifests](/docs/replay/)
- [Hyperparameter sweeps](/docs/sweep/)

### Reference
- [Command Index](/docs/reference/command-index/)
- [Auth + tokens](/docs/auth/)
- [Configuration](/docs/reference/configuration/)
- [Web UI](/docs/reference/web-ui/)
- [Troubleshooting](/docs/reference/troubleshooting/)

### Background
- [Theory & Research](/docs/background/theory/) — RLVR foundations
- [Graduated Rewards](/docs/background/graduated-rewards/) — Partial credit
- [Learning Rate Strategies](/docs/background/learning-rates/) — LR per algorithm

### Meta
- [Changelog](/docs/changelog/) — Version history
- [Contributing](/docs/contributing/) — How to contribute
