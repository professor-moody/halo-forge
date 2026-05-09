# Deterministic replay

`halo-forge replay <run_dir>` regenerates the exact launch command for a captured run, optionally relaunching it. Every shipped trainer writes a `replay.json` manifest next to the `training_summary.json`, capturing every input that influenced the run.

```bash
halo-forge replay ./models/dpo
# Reproducible launch command:
#   halo-forge dpo train --model Qwen/Qwen2.5-3B --dataset ultrafeedback ...

halo-forge replay ./models/dpo --launch         # relaunch in a subprocess
halo-forge replay ./models/dpo --launch --force # relaunch even on env drift
```

## What the manifest captures

```json
{
  "manifest_version": 1,
  "run_id": "dpo-1730000000000",
  "modality": "dpo",
  "timestamp": "2026-05-07T11:42:00+0000",
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "seed": 42,
  "pythonhashseed": "42",
  "config": { /* full DPOConfig as dict */ },
  "dataset": {
    "kind": "huggingface",
    "id": "HuggingFaceH4/ultrafeedback_binarized",
    "revision": null
  },
  "environment": {
    "python": "3.13.12",
    "platform": "Darwin arm64",
    "backend": "mlx",
    "packages": {
      "halo_forge": "1.3.0",
      "torch": "2.5.0",
      "mlx": "0.21.0",
      "transformers": "4.49.0",
      "peft": "0.14.0",
      "trl": "0.29.1",
      ...
    }
  },
  "cli_args": ["dpo", "train", "--model", "Qwen/Qwen2.5-3B-Instruct", ...]
}
```

### Captured

- **Run identity**: `run_id`, `modality`, `timestamp`, `model_name`.
- **Seed bundle**: training seed + `PYTHONHASHSEED` + accelerator seed. Re-applied at replay time via `set_global_seed`.
- **Dataset identity**:
  - Local files → SHA-256 + size, streamed so multi-GB JSONLs don't OOM.
  - HuggingFace → id + revision, so a renamed/updated dataset is detected.
- **Full config snapshot** (dataclass → dict).
- **Environment fingerprint**: Python version, platform, active backend, and pinned versions of `halo_forge`, `torch`, `mlx`, `mlx_lm`, `transformers`, `peft`, `trl`, `accelerate`, `datasets`, `bitsandbytes`, `vllm`, `numpy`.
- **Literal argv** that produced the run.

### Deliberately not captured

- Wall-clock timing (non-deterministic across hosts).
- GPU compute precision (handled by `torch.use_deterministic_algorithms` at replay time).
- Full dataset contents — we hash instead. A divergent dataset surfaces as a hash mismatch at replay.

## Environment diff

`halo-forge replay <run_dir>` automatically diffs the captured environment fingerprint against the active host. Sample output:

```
Environment differs from the captured run:
            python: '3.13.12' -> '3.12.5'
          platform: 'Darwin arm64' -> 'Linux x86_64'
           backend: 'mlx' -> 'cuda'
     packages.torch: '2.5.0' -> '2.4.0'
       packages.trl: '0.29.1' -> '0.31.0'
```

Environment drift refuses `--launch` unless `--force` is provided (sometimes re-running on a different host is the point — comparing reproducibility across MLX vs CUDA, or validating that a torch upgrade didn't change behavior). The warning gives you the information to interpret divergence.

`--launch --force` skips the env-match gate.

## Why dataset hashing matters

The classic reproducibility failure: a training run completes, six months pass, the dataset gets quietly rebuilt with new examples, and the "replay" silently trains on different data. The hash catches this:

- Local-file datasets: SHA-256 over the file. Change one byte → hash differs.
- HF datasets: id + revision (when set). Newer revision → mismatch.

A `dataset.sha256` mismatch at replay is the single most informative signal for "this isn't actually a replay." Local-file hash mismatches refuse `--launch` unless `--allow-dataset-drift` is provided.

## Programmatic API

```python
from halo_forge.replay import (
    capture_manifest, save_manifest, load_manifest,
    compare_environments, EnvironmentFingerprint,
)

manifest = capture_manifest(
    run_id=run.run_id,
    modality="dpo",
    model_name=cfg.model_name,
    seed=cfg.seed,
    config=cfg,
    dataset_id="HuggingFaceH4/ultrafeedback_binarized",
    cli_args=sys.argv[1:],
)
save_manifest(manifest, output_dir)

# At replay time:
loaded = load_manifest(output_dir)
diff = compare_environments(loaded.environment, EnvironmentFingerprint.capture().to_dict())
if not diff["matched"]:
    print("Environment drift:", diff["differences"])
```

## Manifest versioning

`MANIFEST_VERSION = 1`. A future-version manifest still loads; the loader warns and serves the seed + config (the load-bearing fields for replay) without refusing. Older clients can replay newer-format manifests as long as the field names they care about are stable.

## Storage

Manifests live next to `training_summary.json`:

```
models/dpo/
  ├── adapter_config.json
  ├── adapters.safetensors
  ├── final_model/
  ├── replay.json                ← here
  └── training_summary.json
```

About 5-10 KB per manifest. The dataset SHA-256 is the dominant fixed cost; the env fingerprint is small.

## What replay catches that other approaches miss

- **Dataset swap** — caught by the hash mismatch.
- **Code drift** — the `halo_forge` package version is captured; a non-trivial version bump surfaces as env diff.
- **Backend-detection drift** — `environment.backend` captures what halo-forge actually selected; useful for debugging "it ran on MLX last time, why is it picking MPS now?"
- **Seed regression** — random seeds are explicit in the manifest, not implicit in `--seed 42` getting silently dropped.

## Roadmap

- **Compute-precision capture** — record `torch.use_deterministic_algorithms` state, cudnn determinism flags, MLX deterministic mode if/when MLX exposes one.
- **Run forking** (F-Q) — replay manifests will be the foundation; "fork from cycle 3 with one knob changed" needs a place to declare what was forked from what.
- **Cross-machine replay verification** — score divergence across hosts; surface "your reproduction differs from the captured run's checkpoint by X%".
