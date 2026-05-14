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
- Reference-free and reference-model DPO losses are eager in production.
  `mx.compile` remains measurement-only.
- GRPO stays eager: the synthetic advantage-loss results are mixed and do not
  justify a compiled production path.
- IPO, hinge, and KTO-pair DPO reductions have complete synthetic Terminal
  measurements across `32`, `128`, and `512`, and now run through eager MLX
  trainer paths. Compiled execution remains disabled.
- Reference-model GRPO on MLX has a separate dual-model feasibility harness
  (`scripts/measure_mlx_grpo_reference_model.py`) and stays eager.

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
- whether reference-model GRPO needs further dual-model memory runs before
  multi-cycle expansion.

## DPO Variant Gate Protocol

The harness includes non-sigmoid DPO reductions for IPO, hinge, and KTO-pair in
both reference-free and reference-model forms. These candidates are still
compile-measurement-only: the eager trainer paths are supported, but no compiled
trainer path is enabled.

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
- live MLX smoke passes before the variant is treated as release-supported;
- compiled execution remains opt-in even if the reduction benchmark is strong.

## DPO Variant Terminal Measurement

Recorded from a normal Terminal session on the Apple M4 Max host and saved
outside the repo history as `runs/mlx-compile-variants.json`:

- macOS: `26.3.1`
- MLX: `0.31.2`
- mlx-lm: `0.31.3`
- Candidates: sigmoid, IPO, hinge, KTO-pair, and GRPO advantage reduction
- Batch sizes: `32`, `128`, `512`
- Steps: `100`, warmup: `10`

| Candidate | Batch | Eager Mean | Compiled Mean | Steady Delta | Compiled First Step | Peak Bytes |
|---|---:|---:|---:|---:|---:|---:|
| DPO reference-free sigmoid | `32` | `0.000184s` | `0.000148s` | `+19.2%` | `0.002786s` | `560` |
| DPO reference-free sigmoid | `128` | `0.000195s` | `0.000131s` | `+32.6%` | `0.000174s` | `2104` |
| DPO reference-free sigmoid | `512` | `0.000269s` | `0.000139s` | `+48.3%` | `0.000195s` | `8248` |
| DPO reference-model sigmoid | `32` | `0.000147s` | `0.000099s` | `+32.9%` | `0.001217s` | `8248` |
| DPO reference-model sigmoid | `128` | `0.000120s` | `0.000105s` | `+12.1%` | `0.000147s` | `8248` |
| DPO reference-model sigmoid | `512` | `0.000133s` | `0.000108s` | `+18.5%` | `0.000155s` | `16440` |
| DPO reference-free IPO | `32` | `0.000116s` | `0.000105s` | `+9.1%` | `0.002173s` | `16440` |
| DPO reference-free IPO | `128` | `0.000120s` | `0.000107s` | `+10.5%` | `0.000135s` | `16440` |
| DPO reference-free IPO | `512` | `0.000118s` | `0.000109s` | `+8.4%` | `0.000134s` | `16440` |
| DPO reference-model IPO | `32` | `0.000127s` | `0.000105s` | `+17.0%` | `0.000778s` | `16440` |
| DPO reference-model IPO | `128` | `0.000121s` | `0.000109s` | `+10.5%` | `0.000134s` | `16440` |
| DPO reference-model IPO | `512` | `0.000132s` | `0.000109s` | `+17.4%` | `0.000141s` | `16456` |
| DPO reference-free hinge | `32` | `0.000130s` | `0.000104s` | `+20.1%` | `0.001769s` | `16456` |
| DPO reference-free hinge | `128` | `0.000125s` | `0.000117s` | `+6.9%` | `0.000134s` | `16456` |
| DPO reference-free hinge | `512` | `0.000128s` | `0.000111s` | `+13.4%` | `0.000144s` | `16456` |
| DPO reference-model hinge | `32` | `0.000132s` | `0.000108s` | `+17.8%` | `0.002004s` | `16456` |
| DPO reference-model hinge | `128` | `0.000131s` | `0.000104s` | `+20.8%` | `0.000148s` | `16456` |
| DPO reference-model hinge | `512` | `0.000137s` | `0.000111s` | `+19.4%` | `0.000175s` | `16456` |
| DPO reference-free KTO-pair | `32` | `0.000179s` | `0.000138s` | `+23.0%` | `0.001783s` | `16456` |
| DPO reference-free KTO-pair | `128` | `0.000173s` | `0.000122s` | `+29.2%` | `0.000209s` | `16456` |
| DPO reference-free KTO-pair | `512` | `0.000179s` | `0.000126s` | `+29.3%` | `0.000201s` | `16456` |
| DPO reference-model KTO-pair | `32` | `0.000170s` | `0.000126s` | `+25.9%` | `0.000309s` | `16456` |
| DPO reference-model KTO-pair | `128` | `0.000199s` | `0.000131s` | `+34.0%` | `0.000232s` | `16456` |
| DPO reference-model KTO-pair | `512` | `0.000191s` | `0.000136s` | `+28.7%` | `0.000245s` | `24708` |
| GRPO advantage loss | `32` | `0.000110s` | `0.000109s` | `+1.4%` | `0.002057s` | `24708` |
| GRPO advantage loss | `128` | `0.000113s` | `0.000097s` | `+13.8%` | `0.001078s` | `24708` |
| GRPO advantage loss | `512` | `0.000108s` | `0.000102s` | `+5.8%` | `0.000953s` | `24708` |

Interpretation:

- All non-sigmoid DPO variants completed synthetic reduction measurement at
  every planned batch size. That clears the first gate for an eager MLX DPO
  implementation pass.
- KTO-pair has the strongest compiled steady-state signal, but the
  reference-model `512` case also reached the highest observed synthetic peak
  memory. Treat this as a reason to implement eager first and require live
  smoke before exposing it as supported.
- IPO and hinge are also feasible for eager implementation. Their compiled
  gains are smaller than KTO-pair and do not justify a default compiled path.
- GRPO remains eager. The reduction-only compile signal is modest; dual-model
  reference GRPO is measured by `scripts/measure_mlx_grpo_reference_model.py`,
  not by the compile-reduction harness.
