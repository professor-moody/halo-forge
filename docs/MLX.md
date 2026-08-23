# MLX backend

halo-forge supports Apple's [MLX](https://github.com/ml-explore/mlx) framework as a first-class backend on Apple Silicon. MLX is faster than PyTorch MPS for many workloads because it's purpose-built for Apple Silicon — but it's a fundamentally different runtime, with its own model format, tokenizer wrappers, and inference loop.

## Current status

MLX support has moved past the original staged port. Current status:

| Capability | Status | Notes |
|---|---|---|
| Inference (text generation) | ✅ shipped | via `--accelerator mlx` CLI flag |
| LoRA SFT | ✅ shipped | MLX-native trainer path |
| RAFT / RLVR | ✅ shipped | MLX-native RAFT path |
| DPO | ✅ shipped | sigmoid, IPO, hinge, and KTO-pair; reference-free and reference-model paths |
| GRPO | ✅ shipped | reference-free and reference-model eager paths |
| MPS fallback telemetry | ✅ shipped | dashboard warning-line counter |
| Chip-tier surfacing | ✅ shipped | telemetry/backend metadata only, no auto-tuning |

For Apple Silicon training, use MLX when a trainer explicitly supports it and MPS
for PyTorch trainer paths. The dashboard and `halo-forge info` expose the active
backend and chip metadata.

## Install

```bash
pip install -e '.[mlx]'
```

This pulls `mlx>=0.31.0,<0.32.0` and `mlx-lm>=0.31.0,<0.32.0`. Wheels exist only for arm64 macOS — Intel Macs and Linux will fail at install time, which is correct (MLX is Apple Silicon only). The upper bound is intentional: MLX compile/cache behavior has moved quickly across minor releases, so halo-forge keeps the MLX extra inside the tested minor range and bumps it deliberately. The `0.31.x` floor also avoids a Metal-initialization abort observed with `0.29.x` wheels on macOS 26.x.

Verify:

```bash
halo-forge doctor mlx
halo-forge doctor mlx --json
```

The doctor command checks package versions and executes a tiny MLX array in a
subprocess. This matters because headless/sandboxed macOS processes can import
`mlx` but fail with `No Metal device available`; Halo Forge treats that as
`unavailable`, not a crash. If doctor is ready, use MLX explicitly:

```bash
halo-forge --accelerator mlx sft train \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --dataset codealpaca \
  --output models/sft_mlx_quickstart \
  --epochs 1 \
  --batch-size 1 \
  --max-samples 200
```

## Use

The `--accelerator mlx` flag is global (distinct from the per-subcommand `--backend` flag on `data generate`, which selects the LLM API). Use it with MLX-format models on serving, inference, and trainer paths that explicitly support MLX:

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

On Apple Silicon, telemetry and backend responses also include parsed chip metadata when the host exposes it, for example `M3 Max` plus the system-reported GPU core count. Halo-forge surfaces this information only; it does not auto-tune batch sizes, LoRA ranks, or memory settings from the chip tier.

The backend response also includes `mlx_readiness`, the same stable schema used
by `halo-forge doctor mlx`. **Train → Guided** recommends an MLX-format
model only when that executable probe passes; otherwise it falls back to the
MPS-safe first-run path and shows the readiness error.

### Expected doctor shapes

Ready terminal sessions report an executable GPU probe:

```json
{
  "status": "ready",
  "executable": true,
  "probe": {"default_device": "Device(gpu, 0)", "value": 6.0}
}
```

Headless or sandboxed sessions can still import MLX but fail at array
execution. Halo Forge reports that as `unavailable`, not as a crash:

```json
{
  "status": "unavailable",
  "executable": false,
  "errors": ["No Metal device available"]
}
```

## Live MLX acceptance

For a release-confidence pass from a normal Apple Silicon Terminal:

```bash
python scripts/run_mlx_smoke.py --output-dir runs/mlx-smoke
```

The script writes `runs/mlx-smoke/mlx_smoke_summary.json`. A healthy MLX host
should pass the SFT/RAFT live smoke, DPO reference-free sigmoid live smoke, DPO
reference-model sigmoid live smoke, DPO non-sigmoid variant smoke, GRPO
reference-free live smoke, and GRPO reference-model live smoke. No DPO variant
skip is expected on a healthy MLX host.

For a focused dual-model GRPO memory check, run:

```bash
python scripts/measure_mlx_grpo_reference_model.py --json
```

Headless/Codex sessions may report `status=unavailable`; use a normal Terminal
for the real Metal-backed feasibility result.

## MLX training support matrix

| Trainer | MLX status | Notes |
|---|---|---|
| SFT | ✅ supported | MLX LoRA path via `mlx_lm.tuner` |
| RAFT / RLVR | ✅ supported | MLX rollout + verifier filtering + MLX SFT update |
| DPO sigmoid | ✅ supported | reference-free and reference-model |
| DPO IPO / hinge / KTO | ✅ supported | eager MLX path; compiled execution remains measurement-only |
| GRPO | ✅ supported | eager reference-free and reference-model single-cycle policy update |
| Reward model | ❌ roadmap | use PyTorch MPS/CUDA/ROCm today |

## Experimental M5 Neural Accelerator flag

`supports_neural_accelerators` is reported only when the active MLX backend sees an Apple M5-generation chip on macOS 26.2 or newer. Trainer configs expose `enable_neural_accelerators=False` as an explicit opt-in and validate it at trainer initialization.

This is claim-tracking only in this pass. The flag records intent and catches unsupported hosts, but it does not route kernels, alter Metal TensorOps behavior, or promise a speedup yet.

## MPS CPU fallback visibility

PyTorch MPS can silently fall back to CPU for unsupported operations when `PYTORCH_ENABLE_MPS_FALLBACK=1`. Dashboard-launched training now preserves that permissive default and counts fallback warning lines in the public telemetry stream. If the frontend shows the amber `MPS FALLBACK` chip, training is still running, but throughput may be much lower than expected.

## Roadmap reminders

- **MLX compile measurement**: `mx.compile` is measurement-only. Current DPO sigmoid
  numbers are recorded in [MLX_COMPILE_MEASUREMENT.md](MLX_COMPILE_MEASUREMENT.md);
  no compiled trainer path is enabled by default.
- **MLX DPO completion**: sigmoid, IPO, hinge, and KTO-pair support reference-free
  and reference-model paths. RPO remains on PyTorch backends.
- **MLX GRPO completion**: reference-free and reference-model GRPO both run eager
  on MLX. Reference-model mode loads a frozen second copy and records KL summary
  metadata; no compiled path or memory auto-tuning is enabled.
- **Serving follow-up**: speculative decoding and native token streaming remain
  opt-in future work after the shipped OpenAI-shaped streaming surface.

The verifier sandbox at [halo_forge/rlvr/verifiers/execution_runner.py](../halo_forge/rlvr/verifiers/execution_runner.py) is already cross-platform via macOS-native `sandbox-exec`, so MLX RLVR doesn't need any sandbox work — only the policy/optimizer side changes.
