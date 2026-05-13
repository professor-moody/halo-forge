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

Attempted from the Codex workspace on an Apple M4 Max host:

- macOS: `26.3.1`
- GPU: Apple M4 Max, 32 GPU cores, Metal supported according to
  `system_profiler SPDisplaysDataType -json`
- Python: `3.13`
- MLX install from the project extra: `mlx==0.31.2`, `mlx-lm==0.31.3`
- Candidate: synthetic reference-model DPO sigmoid loss
- Batch sizes attempted: `32`, `128`, `512`

Result: **measurement unavailable in this execution context**.

`mlx==0.29.4` / `mlx-lm==0.29.1` aborted during import-time Metal
initialization on macOS 26.3.1. After bumping the project extra to the
`0.31.x` compatibility line, MLX imports cleanly, but array execution from
this Codex runner still fails with:

```text
[metal::load_device] No Metal device available. This typically occurs in
headless, sandboxed, or virtualized macOS sessions where the GPU is not
accessible.
```

The harness now reports that state as structured JSON instead of surfacing a
Python traceback. No compiled production path is enabled from this attempt.
Run the same harness in a normal terminal session with GPU access before
implementing IPO / hinge / KTO MLX DPO variants or enabling any compiled loss
path.
