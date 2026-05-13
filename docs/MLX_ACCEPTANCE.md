# MLX Acceptance Evidence

This document records Terminal acceptance runs that prove the Apple Silicon MLX
path works end to end outside headless/sandboxed automation.

## 2026-05-13 — Apple M4 Max

Command:

```bash
python scripts/run_mlx_smoke.py --output-dir runs/mlx-smoke
```

Summary path:

```text
runs/mlx-smoke/mlx_smoke_summary.json
```

Environment:

| Field | Value |
|---|---|
| Status | `passed` |
| macOS | `26.3.1` |
| Chip | `M4 Max` |
| GPU cores | `32` |
| Metal supported | `true` |
| MLX device | `Device(gpu, 0)` |
| MLX | `0.31.2` |
| mlx-lm | `0.31.3` |

Passed checks:

- `mlx_sft_raft_live_smoke`
- `mlx_dpo_reference_free_live_smoke`
- `mlx_dpo_reference_model_live_smoke`
- `mlx_grpo_reference_free_live_smoke`
- `mlx_dpo_loss_unit`
- `mlx_dpo_reference_model_terminal`
- `mlx_grpo_terminal`

Intentional skip:

- `mlx_dpo_non_sigmoid_variants` — IPO, hinge, and KTO remain disabled until
  larger MLX memory measurements justify them.

Validate a fresh run with:

```bash
python scripts/validate_mlx_smoke_summary.py runs/mlx-smoke/mlx_smoke_summary.json
```

Generated smoke summaries are local artifacts and should not be committed.
