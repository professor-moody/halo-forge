# MLX Compile Measurement Track

`mx.compile` is a measurement track, not a default runtime behavior. Halo Forge
does not enable compiled DPO/GRPO loss paths until measurements show stable
speed or memory wins across realistic shapes.

## Candidate Scope

- MLX DPO reference-free sigmoid, IPO, hinge, and KTO-pair loss reductions.
- MLX DPO reference-model sigmoid, IPO, hinge, and KTO-pair loss reductions.
- MLX GRPO advantage/loss reduction.
- Larger synthetic batch shapes before enabling any compiled path.

## Measurement Protocol

Run the local harness:

```bash
python scripts/measure_mlx_compile.py --json
```

By default this measures:

- `dpo_reference_free_sigmoid`
- `dpo_reference_model_sigmoid`
- `dpo_reference_free_ipo`
- `dpo_reference_model_ipo`
- `dpo_reference_free_hinge`
- `dpo_reference_model_hinge`
- `dpo_reference_free_kto_pair`
- `dpo_reference_model_kto_pair`
- `grpo_advantage_loss`

Across batch sizes:

- `32`
- `128`
- `512`

To narrow a run:

```bash
python scripts/measure_mlx_compile.py \
  --candidate dpo_reference_model_hinge \
  --batch-sizes 32,128 \
  --json
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
- Reference-free and reference-model sigmoid DPO are candidates for a future
  explicit opt-in compiled loss path, but `mx.compile` remains measurement-only.
- GRPO stays eager: the synthetic advantage-loss results are mixed and do not
  justify a production path.
- Keep typed `NotImplementedError` paths for MLX IPO / hinge / KTO variants
  until Terminal measurements show stable behavior and the live trainer path is
  implemented deliberately.
- Keep reference-model GRPO on MLX disabled until dual-model memory is measured.

If a compiled path is implemented later, it should start as MLX DPO sigmoid only
with `compile_loss=False` by default, a CLI opt-in such as `--compile-loss`, and
summary metadata recording `compiled_loss_enabled=true`.

## Earlier Terminal Measurement

Recorded from a normal Terminal session on the same Apple M4 Max host:

- macOS: `26.3.1`
- MLX: `0.31.2`
- Candidate: synthetic reference-model DPO sigmoid loss
- Shape: `batch_size=32`
- Steps: `100`, warmup: `10`
- Memory: active `540` bytes, peak `1080` bytes, cache `680` bytes

| Mode | First Step | Steady Mean | Steady P50 |
| --- | ---: | ---: | ---: |
| eager | `0.095464s` | `0.000240s` | `0.000217s` |
| compiled | `0.234942s` | `0.000177s` | `0.000172s` |

Interpretation: `mx.compile` improves this tiny synthetic loss by about 26%
at steady state, but first-step latency is about 2.5x eager. This is useful
signal, but not enough to enable a production compiled path by default:
larger shapes, label smoothing, GRPO loss reductions, and model-adjacent
memory pressure still need measurement.

## Expanded Terminal Measurement

Recorded from a normal Terminal session on the Apple M4 Max host:

- macOS: `26.3.1`
- MLX: `0.31.2`
- mlx-lm: `0.31.3`
- Candidates: `dpo_reference_free_sigmoid`,
  `dpo_reference_model_sigmoid`, `grpo_advantage_loss`
- Batch sizes: `32`, `128`, `512`
- Steps: `100`, warmup: `10`

| Candidate | Batch | Eager Mean | Compiled Mean | Steady Delta | First-Step Note |
|---|---:|---:|---:|---:|---|
| DPO reference-free sigmoid | `32` | `0.000203s` | `0.000155s` | `+23.8%` | compiled first step `0.221509s` |
| DPO reference-free sigmoid | `128` | `0.000169s` | `0.000123s` | `+27.3%` | compile cache appears warm |
| DPO reference-free sigmoid | `512` | `0.000138s` | `0.000118s` | `+14.7%` | compile cache appears warm |
| DPO reference-model sigmoid | `32` | `0.000140s` | `0.000107s` | `+23.4%` | compiled first step `0.002828s` |
| DPO reference-model sigmoid | `128` | `0.000147s` | `0.000111s` | `+24.8%` | compile cache appears warm |
| DPO reference-model sigmoid | `512` | `0.000133s` | `0.000109s` | `+18.1%` | compile cache appears warm |
| GRPO advantage loss | `32` | `0.000099s` | `0.000142s` | `-43.9%` | compiled slower |
| GRPO advantage loss | `128` | `0.000147s` | `0.000133s` | `+9.3%` | first step near `0.047s` |
| GRPO advantage loss | `512` | `0.000133s` | `0.000096s` | `+28.0%` | first step near `0.049s` |

Interpretation:

- DPO sigmoid reductions show consistent steady-state wins across measured
  shapes. This is enough to justify a future opt-in experiment, not a default.
- GRPO is mixed: batch `32` regresses, larger batches improve, and first-step
  compile cost is high relative to these tiny reductions.
- The measurement is synthetic and reduction-only. It does not include full
  trainer/model-forward overhead, padding behavior, or real sequence shapes.
- IPO / hinge / KTO are still gated because this run did not measure their
  variant-specific loss behavior.

## Codex Runner Measurement

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

## Expanded Measurement Protocol

The harness now emits a top-level `results` list. Each item carries a candidate
name, batch shape, status, eager/compiled timing data when measured, and
best-effort memory telemetry. If a larger shape fails due to memory pressure,
that item should be recorded as `status="error"` while smaller shapes remain
usable evidence.

Next Terminal run:

```bash
python scripts/measure_mlx_compile.py --json
```

Use the expanded results to decide:

- whether compiled DPO/GRPO reductions are worth a production opt-in;
- whether IPO / hinge / KTO have acceptable memory behavior on MLX;
- whether reference-model GRPO is feasible on the M4 Max baseline host.

## DPO Variant Gate Protocol

The harness now includes non-sigmoid DPO reductions for IPO, hinge, and
KTO-pair in both reference-free and reference-model forms. These candidates are
measurement-only. The MLX trainer still raises typed `NotImplementedError` for
those loss types until the Terminal results are reviewed and live smoke coverage
is added for any implemented variant.

Run the variant gate from a normal Apple Silicon Terminal:

```bash
python scripts/measure_mlx_compile.py --json
```

If the full run is too broad while iterating, narrow by candidate:

```bash
python scripts/measure_mlx_compile.py \
  --candidate dpo_reference_model_kto_pair \
  --batch-sizes 32,128,512 \
  --json
```

Promotion criteria for any non-sigmoid DPO variant:

- measured eager and compiled reductions complete at `32`, `128`, and `512`;
- memory telemetry does not show a variant-specific jump relative to sigmoid;
- the eager path is implemented first and covered by loss math tests;
- live MLX smoke is added before the variant leaves typed-unsupported status;
- compiled execution remains opt-in even if the reduction benchmark is strong.
