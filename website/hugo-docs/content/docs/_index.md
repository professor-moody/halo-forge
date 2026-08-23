---
title: "Documentation"
description: "Cross-vendor local model-training workstation for corpus adaptation, supervised and preference tuning, verifier-guided learning, evaluation, and serving."
---

The latest Lab capabilities cover
[outcome validation, controlled studies, grounded data, specialized task
models, and deterministic environments](labs-v11-v15/).

## What halo-forge is

A workstation tool that takes a base model and turns it into a finetuned, evaluated, served artifact — without leaving the local machine.

Halo Forge keeps one guided data-to-training workflow across supported accelerator backends. ROCm and CUDA guided paths are exposed only after their pinned managed runtime passes real hardware qualification.

Pick a goal. Choose a catalog model. Pick an algorithm and verifier. Train, evaluate, serve, and save the run into a bundle when it is worth comparing.

## Start By Intent

| I want to... | Start here |
|---|---|
| Train my first local model | [Quick Start](/docs/getting-started/quickstart/) |
| Adapt a model to local documents | [Corpus Adaptation](/docs/data/corpus-adaptation/) |
| Control a training workstation remotely | [Public Frontend: remote workstation](/docs/reference/public-frontend/#remote-workstation) |
| Pick the right base model | [Choose a Model](/docs/getting-started/choose-a-model/) |
| See runnable examples | [Usage Scenarios](/docs/getting-started/scenarios/) |
| Run on Apple Silicon | [Hardware Notes](/docs/getting-started/hardware/) and [Apple Silicon MLX Quickstart](/docs/getting-started/apple-silicon-mlx/) |
| Serve or export a trained artifact | [Serve / convert / merge](/docs/serving/) |
| Review failures or unlabeled data | [Review Studio](/docs/review-studio/) |
| Check whether a training reward remains trustworthy | [Reward Integrity](/docs/reward-integrity/) |

## Capabilities

### Trainers

- **CPT** — continued causal pretraining over reviewed document corpora with
  deterministic tokenizer packing and explicit LoRA/full adaptation.
- **SFT** — supervised finetuning with QLoRA / LoRA / DoRA / rsLoRA / PiSSA. PyTorch on every torch backend; MLX-native on Apple Silicon.
- **DPO** — preference optimization (sigmoid / IPO / hinge / KTO-pair / RPO / cDPO). PyTorch via TRL; MLX-native DPO supports sigmoid, IPO, hinge, and KTO-pair in reference-free and reference-model modes.
- **GRPO** — verifier-grounded policy gradient (DeepSeek-R1 / Tülu 3 family). PyTorch via TRL; MLX-native reference-free and reference-model GRPO.
- **RAFT** — rejection-sampling RLVR with curriculum + reward shaping. PyTorch + native MLX.
- **Reward Model** — Bradley-Terry RM from preference pairs. Becomes a learned verifier for any other modality.

### Verifiers

Pluggable registry — drop a `.py` in `~/.halo-forge/verifiers/` or use `@register_verifier`. Out of the box:

- **Execution & compile**: `gcc`, `clang`, `mingw`, `execution`, `pytest`, `humaneval`, `mbpp`, `rust`, `cargo`, `go`, `custom`, `subprocess`
- **Schema & format**: `json_structure`, `json_schema`, `regex_format`
- **Reference metrics**: `bleu`, `rouge`, `chrf`
- **LLM-as-judge**: `llm_judge` — rubric-graded with any local or hosted judge model

### Data pipeline

- **Dataset Lab** — immutable local and pinned Hugging Face versions for text,
  preference, tool-use, VLM, audio, and extracted document-corpus training
  data.
- **Review Studio** — deterministic development-data proposals, one- or
  blinded two-pass human review, immutable label sets, and reviewed child
  dataset versions. Protected holdout, operational, test, and canary evidence
  cannot enter acquisition.
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
- **Reward Integrity** — deterministically retains the exact outputs scored by
  verifier-guided training and rescores those same outputs with an independent
  qualified sentinel. Unsupported capture stays report-only; Hugging Face RAFT
  and MLX GRPO are truthfully final-boundary-only, while Hugging Face GRPO is
  step-resumable when `max_steps` is configured.

### Reproducibility

- **Replay manifests** — `halo-forge replay <run_dir>` regenerates the exact launch command.
- **Sweep infrastructure** — Optuna-style hyperparameter search with random / TPE / grid samplers.

### Run management

- **SQLite run database** — search / filter / sort / paginate runs.
- **Multi-run comparison** — pin runs, overlay loss + reward curves, side-by-side config diff.
- **Cohort eval dashboard** — runs × tasks grid; best-per-task highlighted.
- **Cost rollup** — per-run kWh + $ from wall-clock × backend nominal power.
- **Live telemetry strip** — SSE-streamed GPU util / VRAM / power / throughput.
- **Remote workstation** — non-loopback access uses bearer tokens and controls one Halo Forge host.

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
- [Remote Workstation](/docs/reference/public-frontend/#remote-workstation) — Token-authenticated browser access to one training host

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
- [Review Studio](/docs/review-studio/) — acquisition, annotation, adjudication, and immutable label sets

### Evaluation
- [lm-eval + mid-training probe](/docs/eval/)
- [Reward Integrity](/docs/reward-integrity/) — training-signal capture, same-output sentinel audits, and reviewed gates

### Inference + serving
- [Serve / convert / merge](/docs/serving/)

### Reproducibility
- [Replay manifests](/docs/replay/)
- [Hyperparameter sweeps](/docs/sweep/)

### Reference
- [Command Index](/docs/reference/command-index/)
- [Auth + tokens](/docs/auth/)
- [Public Frontend](/docs/reference/public-frontend/)
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
