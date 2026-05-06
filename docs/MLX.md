# MLX backend

halo-forge supports Apple's [MLX](https://github.com/ml-explore/mlx) framework as a first-class backend on Apple Silicon. MLX is faster than PyTorch MPS for many workloads because it's purpose-built for Apple Silicon — but it's a fundamentally different runtime, with its own model format, tokenizer wrappers, and inference loop.

## Phase status

The roadmap stages MLX support across multiple phases. Today (Phase 3):

| Capability | Status | Notes |
|---|---|---|
| Inference (text generation) | ✅ shipped | via `--accelerator mlx` CLI flag |
| LoRA SFT | 🟡 Phase 4 | will use `mlx_lm.tuner` |
| RAFT / RLVR | 🟡 Phase 5 | staged: rollout-only first, then full policy update |

For training on Apple Silicon today, use `--backend mps` (PyTorch MPS) — it works for SFT and RLVR with the same trainer code as ROCm/CUDA.

## Install

```bash
pip install -e '.[mlx]'
```

This pulls `mlx>=0.18.0` and `mlx-lm>=0.20.0`. Wheels exist only for arm64 macOS — Intel Macs and Linux will fail at install time, which is correct (MLX is Apple Silicon only).

Verify:

```bash
python -c "from halo_forge.backend import get_backend; b = get_backend('mlx'); print(b.name, b.capabilities)"
```

## Use

The `--accelerator mlx` flag is global (distinct from the per-subcommand `--backend` flag on `data generate`, which selects the LLM API). Until Phase 4 ships training, only inference paths route to MLX:

```bash
halo-forge --accelerator mlx inference optimize \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --target-precision fp16 \
  --target-latency 100 \
  --output ./out
```

The flag also sets `HALOFORGE_BACKEND=mlx`, so any subprocess (training jobs spawned by the public API, evaluation runners) sees the same choice.

## Weight format

MLX cannot load HuggingFace safetensors directly — it needs MLX-format weights. There are two paths:

### Option 1: Use a pre-converted model

The `mlx-community` HuggingFace org publishes MLX-converted variants of common models, often with quantization baked in:

- `mlx-community/Qwen2.5-7B-Instruct-4bit` — 4-bit quantized
- `mlx-community/Qwen2.5-7B-Instruct-8bit` — 8-bit quantized
- `mlx-community/Qwen2.5-7B-Instruct-bf16` — full precision

Browse the namespace at https://huggingface.co/mlx-community.

### Option 2: Convert your own

`mlx-lm` ships a `convert` entrypoint that takes a HuggingFace repo or local path and produces an MLX-format directory:

```bash
python -m mlx_lm convert --hf-path Qwen/Qwen2.5-7B-Instruct -q --upload-repo your-org/qwen2.5-7b-mlx-4bit
```

Drop `-q` for unquantized. Drop `--upload-repo` to keep it local.

## Quantization differs from bitsandbytes

bitsandbytes-style runtime quantization (`load_in_4bit=True`) does **not** work on MLX. MLX's quantization is group-wise and applied at conversion time — quantized weights are baked into the model artifact. The MLX backend rejects `LoadSpec.quantization="4bit"` with a clear error directing you to use a pre-quantized model or run `mlx_lm.convert -q`.

## What can talk to the MLX backend

The `BackendStrategy` ABC at [halo_forge/backend/base.py](../halo_forge/backend/base.py) is the contract every backend speaks. The MLX backend at [halo_forge/backend/mlx.py](../halo_forge/backend/mlx.py) implements it for `load_causal_lm`, `load_tokenizer`, `resolve_dtype`, `resolve_quantization`, `empty_cache`, and `seed_all`.

`MLXInferenceAdapter` (same file) is a higher-level convenience that holds a (model, tokenizer) pair and exposes `load()` / `generate()` / `cleanup()` — it's what the inference CLI path uses.

```python
from halo_forge.backend.mlx import MLXInferenceAdapter

adapter = MLXInferenceAdapter("mlx-community/Qwen2.5-7B-Instruct-4bit")
adapter.load()
print(adapter.generate("Write a binary search.", max_tokens=128))
adapter.cleanup()
```

## Inspecting the active backend

The public API exposes `GET /api/public/backend` which returns the active backend name, capabilities, and recommended training/inference defaults. The frontend uses this to render the "Running on MLX / MPS / ROCm" badge and to gate UI affordances (e.g. hide 4-bit toggles on backends that don't support runtime quantization).

```bash
curl localhost:8000/api/public/backend | jq
```

## Roadmap reminders

- **Phase 4** (next major MLX work): MLX LoRA SFT via `mlx_lm.tuner`. Will land as `halo_forge.sft.mlx_trainer.MLXSFTTrainer`, sibling of the PyTorch `SFTTrainer`. Same `SFTConfig`, same dataset loaders.
- **Phase 5**: MLX RAFT loops. Stages: (5a) MLX rollout + PyTorch policy update, (5b) full MLX-native RAFT, (5c) curriculum/recovery.

The verifier sandbox at [halo_forge/rlvr/verifiers/execution_runner.py](../halo_forge/rlvr/verifiers/execution_runner.py) is already cross-platform via macOS-native `sandbox-exec`, so MLX RLVR doesn't need any sandbox work — only the policy/optimizer side changes.
