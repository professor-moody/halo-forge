# Halo Forge Roadmap

This document is the repo-tracked status source for follow-on work. Older
scratch plans may still describe already-shipped tracks as future work.

## Shipped

- **Apple Silicon polish**: MLX seeding, MPS environment defaults, replay
  manifest enrichment, dashboard-launched `caffeinate`, spawn-not-fork verifier
  subprocesses on macOS, MLX version bounds, MPS fallback telemetry, chip-tier
  telemetry, and opt-in Neural Accelerator capability metadata.
- **Model catalog v1**: static curated upstream/base-model catalog shared by CLI,
  public API, dashboard quick-picks, and docs.
- **First-run UI polish**: training presets, model decision details, launch
  summaries, model intent filters, and dashboard docs alignment.
- **First Run Experience v2**: guided `/start` backend detection, MLX readiness,
  safe model recommendation, preflight, launch, and route-to-run flow.
- **Model Catalog v2**: first-run ranking, memory estimates, license/download
  notes, risk levels, and MLX catalog/template alignment.
- **Serving streaming**: `/v1/chat/completions` and `/v1/completions` support
  OpenAI-shaped `text/event-stream` responses ending in `data: [DONE]`.
- **MLX compile measurement**: terminal harness and recorded DPO sigmoid and
  non-sigmoid variant measurements; production compiled trainer paths remain
  disabled.
- **MLX DPO**: reference-free and reference-model DPO run on MLX for sigmoid,
  IPO, hinge, and KTO-pair; RPO remains a PyTorch path.
- **MLX GRPO**: reference-free and reference-model GRPO run on MLX with eager
  single-cycle updates and recorded KL metadata.

## Next Tracks

1. **Serving I4**: keep OpenAI streaming compatibility tested and documented;
   native token streaming is a later optimization.
2. **MLX compile production gate**: use the recorded DPO/GRPO measurements to
   decide whether a disabled-by-default `compile_loss` opt-in is worth adding.
3. **GRPO scale-out**: measure multi-cycle and larger dual-model GRPO before
   enabling broader defaults.
4. **Release hardening**: keep all-module qualification, MLX smoke validation,
   and release-confidence checks green across nightly and push workflows.

## Deferred

- Speculative RAFT rollouts.
- `mlx-lm-lora` wrapper as an alternate backend.
- `mlx.distributed`.
- NVFP4.
- MoE per-expert LoRA.
- Batch-size, LoRA-rank, or chip-tier auto-tuning.

## Policy

- Surface hardware and model facts; do not auto-tune from chip tier.
- Keep experimental features opt-in with explicit metadata.
- Preserve direct CLI workflows while improving dashboard onboarding.
- Keep the static catalog as the v2 source of truth; no live Hugging Face search.
