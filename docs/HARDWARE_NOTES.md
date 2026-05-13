# Hardware Notes

Performance findings and per-backend recommendations.

- [Feature × backend matrix](#feature--backend-matrix) — what works where, today
- [AMD Strix Halo (gfx1151)](#amd-strix-halo-gfx1151) — first-class, original target
- [Apple Silicon (PyTorch MPS)](#apple-silicon-pytorch-mps) — Mac primary; supported now
- [Apple Silicon (MLX)](#apple-silicon-mlx) — first-class for supported native paths
- [NVIDIA CUDA](#nvidia-cuda) — falls out of the ROCm code path
- [CPU only](#cpu-only) — last-resort fallback

---

## Feature × backend matrix

Authoritative coverage map across the features halo-forge ships. The
trainer / serving / convert / verifier / cost paths each go through a
backend-aware dispatch; this table records what each dispatch actually
honors.

Legend: ✅ supported · ⚠️ supported with caveats (see footnotes) · ❌ not supported

| Feature | rocm_gfx1151 | cuda | mps | mlx | cpu |
|---|---|---|---|---|---|
| **Trainers** ||||||
| SFT (`halo-forge sft train`) | ✅ | ✅ | ✅¹ | ✅ | ✅² |
| RAFT (`halo-forge raft train`) | ✅ | ✅ | ✅¹ | ✅ | ✅² |
| DPO (`halo-forge dpo train`) | ✅ | ✅ | ✅ | ⚠️³ | ✅² |
| GRPO (`halo-forge grpo train`) | ✅ | ✅ | ✅ | ⚠️¹⁰ | ✅² |
| Reward Model (`halo-forge rm train`) | ✅ | ✅ | ✅ | ❌¹¹ | ✅² |
| **PEFT methods** ||||||
| LoRA | ✅ | ✅ | ✅ | ✅ | ✅ |
| QLoRA (4-bit base) | ⚠️⁴ | ✅ | ❌⁵ | ❌⁵ | ❌⁵ |
| DoRA / rsLoRA / PiSSA / LoftQ | ✅ | ✅ | ✅ | ❌⁶ | ✅ |
| **Optimizers** ||||||
| `adamw_torch` (default) | ✅ | ✅ | ✅ | ✅ via mlx.optimizers | ✅ |
| `adamw_bnb_8bit` / `lion_8bit` | ⚠️⁴ | ✅ | ❌ | ❌ | ❌ |
| **Rollout engines** (`--rollout-engine`) ||||||
| `torch` (HF generate) | ✅ | ✅ | ✅ | ❌⁷ | ✅ |
| `vllm` (continuous batching) | ⚠️⁸ | ✅ | ❌ typed err | ❌ typed err | ❌ typed err |
| `mlx` (mlx_lm.generate) | ❌ | ❌ | ⚠️⁹ | ✅ | ❌ |
| **Verifiers** ||||||
| Code execution (sandboxed where `VerifierExecutionRunner` is used) | ✅ bwrap | ✅ bwrap | ✅ sandbox-exec | ✅ sandbox-exec | ✅ |
| Plugin registry / LLM-as-judge | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Inference + serving** ||||||
| OpenAI-compatible serve (`halo-forge serve`) | ✅ via torch | ✅ via torch | ✅ via torch | ✅ via MLX | ✅ via torch |
| Convert → MLX (`halo-forge convert -f mlx`) | requires mlx-lm | requires mlx-lm | requires mlx-lm | ✅ | requires mlx-lm |
| Convert → GGUF | ✅ | ✅ | ✅ | ✅ | ✅ |
| Convert → HF dtype recast | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Cost & observability** ||||||
| Run-cost rollup (energy + $/kWh) | ✅ 120W nominal | ✅ 350W nominal | ✅ 30W nominal | ✅ 30W nominal | ✅ 25W nominal |
| Run database + filter UI | ✅ | ✅ | ✅ | ✅ | ✅ |

**Footnotes**

1. **MPS** training works for 1B-3B models; 7B+ may hit dtype / memory edges (see MPS section).
2. **CPU** training "works" for tiny models or smoke tests — production runs are infeasible.
3. **MLX DPO** supports sigmoid DPO in both reference-free and reference-model modes. Other loss types (IPO / hinge / kto_pair) raise typed `NotImplementedError` until the MLX measurement track justifies their memory and latency behavior.
4. **QLoRA / bnb-optim on Strix Halo** depends on community-built bitsandbytes-rocm wheels. They exist but aren't always current; if a load fails, set `allow_quantization_fallback=True` to drop to bf16.
5. **QLoRA on MPS / MLX / CPU**: bitsandbytes has no Apple Silicon kernels. The trainer warns and falls back to bf16 (or fails loudly if you set `allow_quantization_fallback=False`).
6. **PEFT on MLX**: `mlx_lm.tuner` only ships LoRA. Setting `--use-dora`, `--use-rslora`, or `--init-lora-weights pissa` on the MLX backend now prints a loud warning at trainer init so you know the flag was ignored. PyTorch backends honor all four.
7. **`--rollout-engine torch` on MLX** is a category error: the trainer is MLX-native, the rollout would have to round-trip through PyTorch on a different backend. Use `--rollout-engine mlx` (default for MLX) or run on a PyTorch backend.
8. **vLLM on Strix Halo (gfx1151)** is experimental; community ROCm support targets MI300X / 7900 XTX class. Halo-forge prints an experimental-status warning at vLLM init and recommends `--rollout-engine torch` or `--rollout-engine mlx` if you hit issues.
9. **`--rollout-engine mlx` on MPS** technically works (uses `mlx_lm.generate` which doesn't care that the host trainer is on MPS) but is unusual; prefer `--accelerator mlx` end-to-end for an MLX-native trainer.
10. **MLX GRPO** v1 ships **reference-free** sigmoid-equivalent GRPO only (`--reference-free`). Pass that flag and the MLX-native trainer runs end-to-end on Apple Silicon: rollouts via `MLXRolloutGenerator`, scoring via the V1 verifier registry, group-relative advantages, single-cycle policy update. Multi-cycle GRPO + reference-model support are roadmap follow-ups.
11. **MLX reward-model training** raises `NotImplementedError`. The PyTorch path on MPS works; native MLX RM training is roadmap. The trained RM artifact is portable (HF `AutoModelForSequenceClassification`), so a model trained on MPS / CUDA can still be loaded as a verifier on MLX hosts at scoring time.

---

## Apple Silicon (PyTorch MPS)

Mac with an M-series chip is the second-supported backend. The training and inference code routes through PyTorch's MPS backend automatically when CUDA/ROCm aren't available.

| Capability | Status | Notes |
|---|---|---|
| Inference | ✅ | 1B–7B models work; 4-bit quantization is unavailable (no Apple Silicon bitsandbytes wheels). |
| LoRA SFT | ✅ small models | 1B–3B fine-tunes complete; expect MPS-specific dtype quirks on 7B+. |
| RLVR/RAFT | ✅ via MPS path | Verifier sandbox uses macOS-native `sandbox-exec` (no extra setup). |
| `bfloat16` | ⚠️ patchy | Defaults to `float16` on macOS <14. Override with `--dtype bf16` if your macOS version is recent. |
| `device_map="auto"` | ⚠️ avoid | accelerate's MPS auto-placer was unreliable pre-`transformers` 4.45. We pin to explicit `{"": "mps:0"}`. |
| Dataloader workers | Use 0 | macOS multiprocessing fork issues with PyTorch tensors. |

### Setup

```bash
xcode-select --install                  # clang + git
brew install rust go dotnet mingw-w64    # verifier toolchains (mingw optional)
pip install -e .
```

The validator script handles macOS specifics:

```bash
bash scripts/validate_environment.sh
```

### Known limitations

- `bitsandbytes` quantization is unavailable. The trainer logs a warning and loads unquantized; gate explicitly with `SFTConfig.allow_quantization_fallback = False` if you want to fail loudly instead.
- ROCm-specific env vars (`HSA_*`, `PYTORCH_HIP_ALLOC_CONF`, `PYTORCH_ROCM_ARCH`) are skipped on macOS hosts — `setup_strix_halo_env()` no-ops cleanly.

---

## Apple Silicon (MLX)

MLX (Apple's native ML framework) is now a first-class Apple Silicon backend for supported inference and trainer paths.

Use MLX-format weights, typically from `mlx-community`, when running with
`--accelerator mlx`. Use MPS for PyTorch trainer paths that do not yet have an
MLX-native implementation.

---

## NVIDIA CUDA

Untested as a primary target but the ROCm code path uses standard PyTorch CUDA APIs throughout, so a CUDA host should work after `pip install` with stock PyTorch CUDA wheels. The Strix-specific env vars and tunings are skipped on non-ROCm hosts.

CUDA-specific tunings (FlashAttention 2, multi-GPU device maps, etc.) aren't yet plumbed; PRs welcome.

---

## CPU only

Tests and tiny models only. No realistic training. The trainer will warn and continue if no accelerator is detected.

---

## AMD Strix Halo (gfx1151)

Performance findings and recommendations for training on AMD Strix Halo — the original first-class target.

## Hardware Specifications

| Component | Specification |
|-----------|---------------|
| GPU | AMD Strix Halo (RDNA 3.5) |
| GPU ID | gfx1151 |
| Compute Units | 40 CUs (2560 shaders) |
| Memory | 128GB unified LPDDR5X |
| Memory Bandwidth | 273 GB/s |
| TDP | 120W |

## Key Insight: Unified Memory Architecture

Strix Halo uses **unified memory** - the GPU and CPU share the same memory pool. This is fundamentally different from discrete GPUs with dedicated VRAM.

- **VRAM pool**: ~2GB (small dedicated cache)
- **GTT (Graphics Translation Table)**: ~128GB (system memory accessible by GPU)
- **Total available**: 128GB+

This architecture has important implications for training optimization.

---

## Performance Findings

### BF16 is Optimal ✅

We tested various precision modes and found **BF16 is optimal** for Strix Halo:

| Precision | Speed | Notes |
|-----------|-------|-------|
| BF16 | Baseline | Optimal for Strix Halo |
| FP16 | Similar | Less numerically stable |
| 4-bit (NF4) | **2x slower** | Dequantization overhead |

### Why 4-bit Quantization is Slower

1. **Dequantization overhead**: 4-bit requires converting weights back to BF16/FP16 during each forward pass, adding compute cycles.

2. **Compute-bound workload**: Strix Halo runs at 96-99% compute utilization with BF16. Since compute is the bottleneck (not memory), 4-bit's memory savings don't help.

3. **Unified memory negates VRAM savings**: The main benefit of 4-bit is fitting models in limited VRAM. With 128GB unified memory, you're never VRAM-limited.

---

## Optimal Configuration

### Training Settings

```yaml
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  gradient_checkpointing: true
  bf16: true
  optim: "adamw_torch"
  
  # CRITICAL for unified memory - prevents GPU hangs
  dataloader_num_workers: 0
  dataloader_pin_memory: false
```

### Generation Settings

```yaml
generation:
  batch_size: 8
  samples_per_prompt: 8
  max_new_tokens: 1024
```

### Environment Variables

```bash
# Memory management for unified memory
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"

# Use hipBLASLt for optimized BLAS (already set in toolbox)
export ROCBLAS_USE_HIPBLASLT=1

# Disable SDMA (prevents hangs on some systems)
export HSA_ENABLE_SDMA=0
```

---

## What Works Well

| Feature | Status | Notes |
|---------|--------|-------|
| BF16 training | ✅ | Optimal precision |
| Flash Attention | ✅ | Works via ROCm fork (main_perf branch) |
| LoRA | ✅ | Full support |
| Gradient checkpointing | ✅ | Essential for 7B+ models |
| Large batch sizes | ✅ | 128GB allows generous batching |

## What Doesn't Work Well

| Feature | Status | Notes |
|---------|--------|-------|
| 4-bit quantization | ❌ | 2x slower than BF16 |
| FP16 | ⚠️ | Less stable than BF16 on AMD |
| SDPA attention | ⚠️ | 30-40% slower than eager |
| dataloader_num_workers > 0 | ❌ | Causes GPU hangs |
| dataloader_pin_memory | ❌ | Causes GPU hangs |

---

## Performance Expectations

### RAFT Training (7B model, ~500 prompts, 8 samples/prompt)

| Phase | Time | Notes |
|-------|------|-------|
| Generation | ~7-8 hours | ~6s per sample |
| Verification | ~30-60 min | Parallelized on CPU |
| Training | ~30 min | Per cycle |
| **Total per cycle** | ~8-9 hours | |

### SFT Training

| Dataset Size | Epochs | Estimated Time |
|--------------|--------|----------------|
| 10K samples | 3 | ~8 hours |
| 50K samples | 3 | ~2 days |
| 100K samples | 3 | ~4 days |

---

## Monitoring

### Check GPU utilization

```bash
# Real-time GPU stats
watch -n 1 rocm-smi

# Or use radeontop
radeontop
```

### Expected metrics during training

- **Command Processor - Compute**: 95-99% (compute-bound)
- **VRAM**: 500-800 MB (small cache)
- **GTT**: 20-30 GB (actual GPU memory usage)
- **GFX Activity**: 95-99%

---

## Troubleshooting

### GPU hang during training

1. Ensure `dataloader_num_workers: 0`
2. Ensure `dataloader_pin_memory: false`
3. Add `export HSA_ENABLE_SDMA=0`

### Out of memory

Unlikely with 128GB, but if it happens:
1. Reduce batch size
2. Enable gradient checkpointing
3. Reduce max_seq_length

### Slow generation

1. Verify using BF16 (not 4-bit)
2. Check GPU is at 95%+ utilization
3. Ensure only one training process running

---

## ROCm Stack

The halo-forge toolbox uses:

| Component | Version | Source |
|-----------|---------|--------|
| ROCm | 7.x nightly | TheRock S3 bucket |
| PyTorch | Nightly | AMD nightlies for gfx1151 |
| bitsandbytes | Upstream | Built with gfx1151 target |
| Flash Attention | ROCm fork | main_perf branch |

## Acknowledgments

- AMD for Strix Halo hardware
- kyuz0 for the original fine-tuning toolbox
- TheRock project for ROCm nightlies
- The Strix Halo community for testing and feedback
