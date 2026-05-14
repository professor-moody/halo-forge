---
title: "Trainers"
description: "Halo-forge ships four post-training algorithms. They share a common config / dispatch / output shape so the public API and frontend treat every run the same way regardless of which algorithm produced it."
weight: 10
---


Halo-forge ships four post-training algorithms. They share a common config / dispatch / output shape so the public API and frontend treat every run the same way regardless of which algorithm produced it.

| Algorithm | What it does | When to use | CLI |
|---|---|---|---|
| **SFT** | Supervised finetuning on instruction/response pairs. | Adapting a base to a domain or task; first step in every recipe. | `halo-forge sft train` |
| **DPO** | Preference optimization from `(prompt, chosen, rejected)` triples. | Alignment without RL. The default published format on HF. | `halo-forge dpo train` |
| **GRPO** | Verifier-grounded policy gradient with group-relative advantages. | RLVR — code execution, math, tool calling, anything with a programmatic reward. | `halo-forge grpo train` |
| **RAFT** | Rejection-sampling RL: sample N, verify, SFT on the kept ones. | The simplest RLVR; works without KL terms or ratio clipping. | `halo-forge raft train` |

The post-training triad is **SFT → DPO → GRPO** (plus optional RAFT). Run them in order; each builds on the previous artifact.

## Backend dispatch

Every trainer routes through `halo_forge.<modality>._dispatch.get_<modality>_trainer(config)`. The dispatcher:

1. Reads the active backend (`halo_forge.backend.get_backend()`), or honors `--accelerator` if set.
2. Picks the right trainer class for that backend.
3. Returns the constructed trainer; the trainer's `__init__` validates the config against what *that* backend can honor — see [`halo_forge.utils.backend_config.warn_unsupported_for_mlx`](../halo_forge/utils/backend_config.py).

So setting `--use-dora` on an MLX host doesn't silently fall back to vanilla LoRA — the trainer prints a warning at init pointing at the limitation.

## SFT

```bash
halo-forge sft train \
  --dataset codealpaca \
  --model Qwen/Qwen2.5-Coder-3B \
  --epochs 3 --batch-size 2 --gradient-accumulation 16 \
  --lora-rank 16 --lora-alpha 32 --use-dora \
  --learning-rate 2e-4 --optim adamw_torch
```

**LoRA / PEFT options.** All four advanced PEFT methods are available on PyTorch backends:
- `--use-dora` — DoRA decomposition (magnitude + direction)
- `--use-rslora` — rank-stabilized LoRA scaling (alpha/√r)
- `--init-lora-weights pissa` — PiSSA initialization (faster convergence)
- `--init-lora-weights loftq|olora|gaussian|true|false` — other init strategies

MLX SFT supports vanilla LoRA only; the other PEFT flags warn at init.

**Optimizer choice.** `--optim` is forwarded to `transformers.TrainingArguments.optim`:
- `adamw_torch` (default) — works everywhere
- `adamw_bnb_8bit` — bitsandbytes 8-bit AdamW; halves optimizer memory. CUDA/ROCm-with-bnb only.
- `lion_8bit`, `paged_adamw_8bit`, `paged_adamw_32bit` — bitsandbytes variants

**Datasets.** Built-in short names (run `halo-forge sft datasets`): `codealpaca`, `metamath`, `gsm8k_sft`, `llava`, `xlam_sft`, `glaive_sft`, …. Or pass any HuggingFace dataset id, or `--data path/to/file.jsonl` for a local file.

## DPO

```bash
halo-forge dpo train \
  --dataset ultrafeedback \
  --model Qwen/Qwen2.5-3B-Instruct \
  --beta 0.1 --loss-type sigmoid \
  --learning-rate 5e-6 --epochs 1
```

**Algorithm knobs.**
- `--beta` — KL-regularization strength against the reference model (0.1 canonical; 0.05 gentler; 0.5 sharper)
- `--loss-type` — `sigmoid` (canonical), `ipo`, `hinge`, `kto_pair`, `rpo`
- `--label-smoothing` — cDPO smoothing (0.0 disables)
- `--reference-free` — skip loading the reference model; use a frozen policy at step 0 instead. Saves memory and is still the lowest-risk MLX mode.

**Datasets.** Built-in short names: `ultrafeedback`, `orca_dpo`, `hh_rlhf`, `py_dpo`. JSONL files need `prompt`, `chosen`, `rejected` columns; UltraFeedback's chat-list format is auto-collapsed.

**MLX DPO scope.** MLX supports `sigmoid`, `ipo`, `hinge`, and `kto_pair` in reference-free and reference-model modes. `rpo` remains on the PyTorch/TRL path.

## GRPO

```bash
halo-forge grpo train \
  --data prompts.jsonl \
  --model Qwen/Qwen2.5-Coder-3B \
  --verifier execution --num-generations 8 \
  --beta 0.04 --epsilon 0.2 \
  --rollout-engine vllm   # or 'mlx' on Apple Silicon
```

**Algorithm knobs.**
- `--num-generations` — group size; how many completions per prompt. 4-8 is typical.
- `--beta` — KL strength (0.04 = DeepSeek-R1 default; 0 disables KL).
- `--epsilon` — PPO ratio clip (0.2 standard).
- `--temperature` — rollout sampling temperature (0.9 default; higher = more diverse groups).
- `--no-scale-rewards` — flip from canonical GRPO (advantage / std) to RLOO-flavored (mean baseline only).
- `--reference-free` — skip the reference model; saves memory. MLX supports both reference-free and reference-model GRPO.

**Verifier integration.** `--verifier <short_name>` resolves through the [V1 plugin registry](VERIFIERS.md). Pass `execution`, `pytest`, `humaneval`, `llm_judge`, `json_schema`, or any `@register_verifier`-decorated class. The trainer instantiates and bridges to TRL's `reward_funcs` (PyTorch) or the manual scoring loop (MLX).

**Rollout engine** (`--rollout-engine`):
- `auto` (default) — torch fallback (HF generate)
- `torch` — same as auto; explicit
- `vllm` — continuous-batched on CUDA/ROCm. 5-10× faster generation. gfx1151 is experimental.
- `mlx` — Apple Silicon native via `mlx_lm.generate`

## RAFT

The original RLVR algorithm in halo-forge. Sample N, verify, SFT on the kept samples — no KL term, no ratio clipping, just iterated rejection sampling.

```bash
halo-forge raft train \
  --prompts data/prompts.jsonl \
  --model Qwen/Qwen2.5-Coder-3B \
  --verifier execution \
  --cycles 5 --samples-per-prompt 8 --keep-percent 0.3 \
  --rollout-engine vllm
```

**MLX RAFT.** Two paths:
- `--accelerator mlx --rollout-only` — hybrid: MLX rollouts + PyTorch policy update.
- `--accelerator mlx` (no `--rollout-only`) — full MLX-native RAFT (Phase 5b).

## Comparing trainers

| | SFT | DPO | GRPO | RAFT |
|---|---|---|---|---|
| Data shape | `(prompt, completion)` | `(prompt, chosen, rejected)` | `prompt` only | `prompt` only |
| Reward source | label likelihood | preference labels | verifiers | verifiers |
| Wall-clock per step | low | medium (2× forward) | high (N forwards + verify) | high (N forwards + verify) |
| Sample efficiency | high (every token contributes) | medium | high (advantages reweight all samples) | low (drops 70%+ via threshold) |
| KL term | no | yes | yes | no |
| MLX native | ✅ | ✅ reference-free/reference-model | ✅ reference-free/reference-model | ✅ |

**Recommended pipeline for a new task.**
1. **SFT** on a domain dataset to align format + style. 1-3 epochs, LR 2e-4.
2. **DPO** on a preference-pair dataset to align quality. 1 epoch, LR 5e-6, β=0.1.
3. **GRPO** with a programmatic verifier to optimize the actual objective. 1 cycle, LR 1e-6, β=0.04, num_generations=8.

## Output shape

Every trainer writes a `training_summary.json` to `--output` with the same schema:

```json
{
  "modality": "dpo",
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "run_id": "dpo-1730000000000",
  "seed": 42,
  "cycles": [{"cycle": 0, "train_loss": ..., "cycle_duration_seconds": ..., ...}],
  "total_train_steps_executed": 200,
  "final_train_loss": 0.42,
  "weights_updated": true,
  "effectiveness": {"verdict": "passed", "reasons": [...]},
  "yield_diagnostics": {...},
  "recovery": {...},
  "extra": {"beta": 0.1, "loss_type": "sigmoid", "reward_accuracy_history": [...]}
}
```

The same shape powers the run database (`/runs/search`), the run-detail UI, and the deterministic replay manifest. See [`docs/REPLAY.md`](REPLAY.md) for the replay contract.
