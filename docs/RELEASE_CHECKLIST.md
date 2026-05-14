# Release Checklist

Use this checklist for a 1.4.0 release-candidate pass. Generated outputs under
`runs/*`, local datasets under `examples/datasets/*`, and `uv.lock` are local
artifacts unless a release task explicitly says otherwise.

## Required Local Checks

```bash
git diff --check
.venv/bin/python -m compileall -q halo_forge ui tests scripts
.venv/bin/python -m pytest tests/test_serving.py -q
.venv/bin/python -m pytest tests/test_mlx_readiness.py tests/test_mlx_smoke_summary_validator.py tests/test_mlx_grpo_reference_model_measurement.py -q
.venv/bin/python -m pytest tests/test_grpo.py tests/test_mlx_terminal_smoke.py -q
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

- `pyproject.toml` and `halo_forge.__version__` report `1.4.0`.
- `docs/MLX_ACCEPTANCE.md` records the latest Terminal MLX acceptance evidence.
- GitHub Actions for the latest pushed commit are green.
- No generated `runs/*`, local `examples/datasets/*`, or `uv.lock` artifacts are staged.

## Tagging

Only tag after the latest push CI and nightly qualification proof are green:

```bash
git tag -a v1.4.0 -m "Halo Forge 1.4.0"
git push origin v1.4.0
```

Do not create a GitHub Release from this checklist unless the release task
explicitly asks for one.
