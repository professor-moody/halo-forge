# Release Checklist

Use this checklist for a 2.0.0-alpha-1 release-candidate pass. Generated outputs under
`runs/*`, local datasets under `examples/datasets/*`, and `uv.lock` are local
artifacts unless a release task explicitly says otherwise.

## Required Local Checks

```bash
git diff --check
.venv/bin/python -m compileall -q halo_forge ui tests scripts
.venv/bin/python -m pytest tests/test_serving.py -q
.venv/bin/python -m pytest tests/test_mlx_readiness.py tests/test_mlx_smoke_summary_validator.py tests/test_mlx_grpo_reference_model_measurement.py -q
.venv/bin/python -m pytest tests/test_grpo.py tests/test_mlx_terminal_smoke.py -q
.venv/bin/python -m pytest tests/test_public_api_pivot.py tests/test_model_catalog.py tests/test_huggingface_access.py tests/test_serving.py tests/test_playground_proxy.py -q
```

Or run the bundled harness:

```bash
.venv/bin/python scripts/run_release_confidence.py
```

## Frontend Checks

Run these when `public_app/node_modules` is installed:

```bash
cd public_app
npm run lint
npm run build
```

## Alpha Dashboard/Desktop Acceptance

For the 2.0.0-alpha-1 app-first pass:

```bash
cd apps/desktop-tauri
npm run build:runtime
npm run build
npm run smoke:runtime
```

Install the unsigned macOS DMG and smoke these dashboard routes:

- `/` Overview opens with current backend/telemetry status and no stale runs on a fresh profile.
- `/start` first-run goal chooser and writable output defaults.
- `/runs/$runId` live monitor after a tiny SFT launch: timestamp says now, stage rail advances, log tail is available, optimizer steps are real, and final artifact state becomes available.
- `/train` method/goal workspace for SFT, RAFT, DPO, ORPO, RM, GRPO, VLM, audio, reasoning, and agentic templates.
- `/models` Hugging Face access checks, open-model defaults, and managed Serve action.
- `/playground` managed local serving, chat, gated-model recovery, and Stop.
- `/results` completed-run actions, Results-to-run links, and local artifact paths.
- `/diagnostics` app-run roots, app logs, failed launches, completed launches, and log tails under `~/.halo-forge`.
- `/connect` Halo Forge API token and workstation-scoped Hugging Face token flows.

Expected desktop behavior: the app reports `2.0.0-alpha-1`, starts the local dashboard on `127.0.0.1:8765`, keeps logs under `~/.halo-forge/desktop/runtime.log`, and quits without killing unrelated Halo Forge processes.

Expected training monitor behavior: stages read Prepare, Data, Model, Trainer,
Train, Save, Finalize, Done/Failed; live loss appears only after optimizer
steps; dataset `Map:` progress is not counted as training; success shows Open
Results, Serve model when available, output path, final loss, and duration;
failure shows the classified cause, last useful log lines, retry action, and
Diagnostics link.

The release-confidence harness runs these automatically when dependencies are
present. Use `--skip-frontend` only when documenting an unavailable frontend
dependency state in release notes.

## MLX Terminal Acceptance

From a normal Apple Silicon Terminal with Metal access:

```bash
halo-forge doctor mlx --json
python scripts/measure_mlx_grpo_reference_model.py --json > runs/mlx-grpo-reference-model.json
python scripts/run_mlx_smoke.py --output-dir runs/mlx-smoke
python scripts/validate_mlx_smoke_summary.py runs/mlx-smoke/mlx_smoke_summary.json
```

To include the existing generated smoke summary in the release-confidence
harness without creating new artifacts:

```bash
.venv/bin/python scripts/run_release_confidence.py --include-live-mlx
```

Expected MLX smoke result: overall `passed`, with SFT/RAFT, DPO
reference-free/reference-model, DPO IPO/hinge/KTO-pair, GRPO reference-free,
GRPO reference-model, and terminal math checks all passing.

## CI And Release Metadata

```bash
git status --short
git log --oneline -5
gh run list --branch main --limit 5
```

Before tagging or publishing, confirm:

- `pyproject.toml` and `halo_forge.__version__` report package version `2.0.0a1`.
- `halo_forge.version.DISPLAY_VERSION` and `/api/public/version` report `2.0.0-alpha-1`.
- `public_app/package.json`, Tauri config, and desktop package metadata report `2.0.0-alpha-1`.
- `docs/MLX_ACCEPTANCE.md` records the latest Terminal MLX acceptance evidence.
- GitHub Actions for the latest pushed commit are green.
- No generated `runs/*`, local `examples/datasets/*`, or `uv.lock` artifacts are staged.

## Tagging

Only tag after the latest push CI and nightly qualification proof are green:

```bash
git tag -a v2.0.0-alpha-1 -m "Halo Forge 2.0.0-alpha-1"
git push origin v2.0.0-alpha-1
```

Do not create a GitHub Release from this checklist unless the release task
explicitly asks for one.
