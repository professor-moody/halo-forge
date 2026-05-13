---
title: "Apple Silicon MLX Quickstart"
weight: 24
---

# Apple Silicon MLX Quickstart

MLX is the recommended Apple Silicon path when it is actually executable on
your machine. Halo Forge checks that before the dashboard recommends it.

## 1. Install MLX support

```bash
pip install -e '.[mlx]'
```

MLX wheels are arm64 macOS only. Linux, Windows, and Intel Macs should use the
CPU/CUDA/ROCm/MPS paths instead.

## 2. Run the doctor

```bash
halo-forge doctor mlx
halo-forge doctor mlx --json
```

`ready` means MLX imported and executed a tiny array on Metal. `unavailable`
usually means the process cannot see a Metal device, which can happen in
headless or sandboxed shells even when a normal Terminal works.

Ready JSON includes the executable probe:

```json
{"status": "ready", "executable": true, "probe": {"default_device": "Device(gpu, 0)"}}
```

Headless sessions should fail softly:

```json
{"status": "unavailable", "executable": false, "errors": ["No Metal device available"]}
```

## 3. Start with SFT

Use a small MLX-format model first:

```bash
halo-forge --accelerator mlx sft train \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --dataset codealpaca \
  --output models/sft_mlx_quickstart \
  --epochs 1 \
  --batch-size 1 \
  --max-samples 200
```

The dashboard `/start` flow uses the same readiness check. If MLX is ready, it
prefills an MLX first-run model and launches with `accelerator=mlx`. If MLX is
installed but cannot execute, it shows the readiness error and falls back to a
safer MPS choice.

## 4. Try verifier and preference tracks

Code RAFT:

```bash
halo-forge --accelerator mlx raft train \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --prompts humaneval \
  --verifier execution \
  --output models/raft_mlx_code \
  --cycles 1 \
  --samples-per-prompt 2
```

DPO sigmoid:

```bash
halo-forge --accelerator mlx dpo train \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --dataset ultrafeedback-binarized \
  --output models/dpo_mlx_sigmoid \
  --loss-type sigmoid \
  --batch-size 1
```

MLX supports sigmoid DPO in both reference-free and reference-model modes. Keep
IPO, hinge, and KTO on the PyTorch backends until the MLX memory measurements
justify those variants.

GRPO reasoning:

```bash
halo-forge --accelerator mlx grpo train \
  --model mlx-community/Qwen2.5-0.5B-Instruct-bf16 \
  --dataset gsm8k \
  --verifier json_schema \
  --output models/grpo_mlx_reasoning \
  --group-size 4
```

## 5. Smoke test from Terminal

For a local acceptance pass:

```bash
python scripts/run_mlx_smoke.py --output-dir runs/mlx-smoke
```

The script writes `mlx_smoke_summary.json` and leaves repo fixtures untouched.
Expected passing checks on a healthy Apple Silicon Terminal:

- `mlx_sft_raft_live_smoke`
- `mlx_dpo_reference_free_live_smoke`
- `mlx_dpo_reference_model_live_smoke`
- `mlx_grpo_reference_free_live_smoke`
- `mlx_dpo_loss_unit`
- `mlx_dpo_reference_model_terminal`
- `mlx_grpo_terminal`

The only expected skip is `mlx_dpo_non_sigmoid_variants`.

## Support matrix

| Trainer | MLX status |
|---|---|
| SFT | supported |
| RAFT / RLVR | supported |
| DPO sigmoid | reference-free and reference-model supported |
| DPO IPO / hinge / KTO | measurement-gated |
| GRPO | reference-free supported |
| Reward model | roadmap |

## Troubleshooting

- `No Metal device available`: rerun from normal Terminal and confirm the app has
  GPU access. This is common in headless/sandboxed processes.
- Missing `mlx-lm`: reinstall with `pip install -e '.[mlx]'`.
- Hugging Face model fails to load: choose an `mlx-community/...` MLX-format
  model or convert with `mlx_lm.convert`.
- Slow MPS run: if you are not on MLX, watch the dashboard telemetry strip for
  `MPS FALLBACK`, which means PyTorch moved an unsupported operation to CPU.

`mx.compile` remains measurement-only in Halo Forge. No trainer path auto-enables
compiled MLX kernels yet.
