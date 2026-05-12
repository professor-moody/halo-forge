# MLX Compile Measurement Track

`mx.compile` is a measurement track, not a default runtime behavior. Halo Forge
does not enable compiled DPO/GRPO loss paths until measurements show stable
speed or memory wins across realistic shapes.

## Candidate Scope

- MLX DPO reference-free sigmoid loss.
- MLX GRPO advantage/loss reduction.
- Reference-model DPO only after the dual-model memory footprint is measured.

## Measurement Protocol

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
- Keep typed `NotImplementedError` paths for unsupported MLX DPO variants until
  the measurement track justifies implementation.
