# MLX Compile Measurement Track

`mx.compile` is a measurement track, not a default runtime behavior. Halo Forge
does not enable compiled DPO/GRPO loss paths until measurements show stable
speed or memory wins across realistic shapes.

## Candidate Scope

- MLX DPO reference-free sigmoid loss.
- MLX GRPO advantage/loss reduction.
- Reference-model DPO only after the dual-model memory footprint is measured.

## Measurement Protocol

Run the local harness:

```bash
python scripts/measure_mlx_compile.py --json
```

Record each run with:

- model id and parameter count
- chip metadata from `halo-forge info`
- macOS and MLX / `mlx-lm` versions
- sequence length, batch size, and group size where applicable
- first-step compile time
- steady-state step time
- peak memory if available
- whether shapes stayed cache-stable

Compare each candidate against the existing eager implementation. A compiled
path should not ship as default unless it improves steady-state throughput
without making first-step latency, memory, or shape-cache behavior operationally
worse.

## Current Decision

- No default `mx.compile` integration.
- No chip-tier auto-tuning.
- No speculative claims in docs or UI.
- Reference-model sigmoid DPO is implemented for MLX, but `mx.compile`
  remains measurement-only.
- Keep typed `NotImplementedError` paths for MLX IPO / hinge / KTO variants
  until the measurement track justifies implementation.

## Latest Local Measurement

Not recorded in this workspace: the active `.venv` does not have `mlx`
installed. Run the harness above on an Apple Silicon `[mlx]` environment and
paste the JSON output here before enabling any compiled production path.
