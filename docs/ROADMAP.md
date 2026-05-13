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
- **MLX compile measurement**: terminal harness and recorded DPO sigmoid
  measurement; production compiled trainer paths remain disabled.
- **MLX sigmoid DPO**: reference-free and reference-model sigmoid DPO run on
  MLX; non-sigmoid losses remain gated.

## Next Tracks

1. **MLX productization**: live terminal acceptance for SFT, RAFT, DPO
   reference-free/reference-model sigmoid, and GRPO reference-free paths.
2. **Serving QA**: keep OpenAI streaming compatibility tested and documented;
   native token streaming is a later optimization.
3. **MLX compile expansion**: measure larger DPO/GRPO shapes before enabling any
   compiled default.
4. **MLX DPO variants**: IPO / hinge / KTO only after measurement establishes
   acceptable memory and latency behavior.
5. **Reference-model GRPO on MLX**: implement only after dual-model memory is
   measured on real Apple Silicon hosts.

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
