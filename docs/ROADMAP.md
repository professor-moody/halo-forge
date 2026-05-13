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

## Next Tracks

1. **First Run Experience v2**: guided `/start` flow that detects the backend,
   chooses safe catalog defaults, preflights, launches, and routes to the run.
2. **Model Catalog v2**: decision metadata, memory estimates, first-run ranking,
   license/download warnings, and clearer risk levels.
3. **Serving I3**: OpenAI-compatible streaming responses; speculative decoding is
   a later opt-in flag after streaming lands.
4. **MLX compile measurement**: measure `mx.compile` candidates for DPO/GRPO
   loss paths before enabling any compiled default.
5. **MLX DPO completion**: non-sigmoid DPO losses after the measurement track
   establishes memory and latency behavior.

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
