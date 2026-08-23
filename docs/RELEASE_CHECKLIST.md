# Release Checklist

Use this checklist for a 2.0.0-alpha-2 release-candidate pass. Generated outputs under
`runs/*` and local datasets under `examples/datasets/*` are local artifacts.
`uv.lock` and the two files under `constraints/` are reviewed release inputs and
must be committed together whenever dependency metadata changes.

## Required Local Checks

```bash
git diff --check
uv lock --check
.venv/bin/python scripts/check_dependency_contract.py
.venv/bin/python -m ruff check --select E9,F63,F7,F82 --exclude .venv,node_modules,target .
.venv/bin/python -m compileall -q halo_forge ui tests scripts
.venv/bin/python scripts/check_release_interfaces.py
.venv/bin/python scripts/generate_modality_baseline.py --check
.venv/bin/python scripts/run_ops_module_matrix.py --fixture-pack v1 --strict
.venv/bin/python -m pytest tests -q
```

Regenerate the reviewed pip constraints after changing `pyproject.toml`:

```bash
uv lock
uv export --extra dev --no-emit-project --no-hashes --no-annotate --no-header --frozen --output-file constraints/release.txt
uv export --extra dev --extra mlx --no-emit-project --no-hashes --no-annotate --no-header --frozen --output-file constraints/release-mlx.txt
python scripts/check_dependency_contract.py
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
npm run test:desktop-bridge
npm run test:own-data-studio

cd ../website/hugo-docs
hugo --minify
```

## Cross-Platform Beta Dashboard/Desktop Acceptance

For the 2.0.0-alpha-2 app-first pass:

```bash
cd apps/desktop-tauri
npm run build:runtime
npm run build
npm run smoke:runtime
```

The release workflow builds macOS arm64 DMG, Linux x86-64 AppImage/deb, and
Windows x86-64 NSIS candidates. A candidate is a normal installer only when the
release manifest records a passing packaged-runtime smoke and the required
signature state. Install each qualified artifact and smoke these routes:

- `/` Overview opens with current backend/telemetry status and no stale runs on a fresh profile.
- `/train` Guided mode, own-data selection, capability-aware defaults, and writable output paths.
- `/runs/$runId` live monitor after a tiny SFT launch: timestamp says now, stage rail advances, log tail is available, optimizer steps are real, and final artifact state becomes available.
- `/train` method/goal workspace for SFT, RAFT, DPO, ORPO, RM, GRPO, VLM, audio, reasoning, and agentic templates.
- `/models` catalog, trained artifacts, Hugging Face access checks, and **Serve
  & Test** managed local serving, chat, gated-model recovery, and Stop.
- `/runs` completed-run actions, run detail links, and local artifact paths.
- `/diagnostics` app-run roots, app logs, failed launches, completed launches, and log tails under `~/.halo-forge`.
- `/connect` Halo Forge API token and workstation-scoped Hugging Face token flows.
- `/setup` runtime, roots, disk/RAM, backend, model access, and safe remediation.
- `/datasets/repair` exact issue scan, reviewed plan, immutable publication, and source-drift refusal.

Expected desktop behavior: the app reports `2.0.0-alpha-2`, starts the local dashboard on `127.0.0.1:8765`, keeps logs under `~/.halo-forge/desktop/runtime.log`, and quits without killing unrelated Halo Forge processes. Windows shutdown covers only the owned process tree. Uninstall preserves `~/.halo-forge` datasets, runs, models, and review history.

Expected training monitor behavior: stages read Prepare, Data, Model, Trainer,
Train, Save, Finalize, Done/Failed; live loss appears only after optimizer
steps; dataset `Map:` progress is not counted as training; success shows Open
Run, artifact/serve actions when available, output path, final loss, and duration;
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

## V21 Real-Path And Workstation Evidence

Before describing a workstation as beta-qualified:

- core runtime qualification is current;
- the recommended instruction-SFT path has ten passed, checksummed steps;
- the own-data proof is bound to a ready capacity check and immutable Dataset Version;
- its trainable-parameter SHA-256 changed and its saved artifact is reload-verified;
- base and proof evaluations use the same development suite revision and have a completed outcome assessment;
- scheduler restart recovery and external-workload wait→idle transitions exist as database events;
- the control-plane event window spans at least twelve hours with no more than three sequential bounded proofs; and
- `halo-forge release workstation-report CERTIFICATION_ID --json` reports `beta_qualified`, while its integrity endpoint is valid.

Boolean claims or generic tensor diagnostics are not release evidence. CUDA
remains hardware-unqualified until the same ladder passes on a real NVIDIA host.

## CI And Release Metadata

```bash
git status --short
git log --oneline -5
gh run list --branch main --limit 5
gh release list --limit 5
```

Before tagging or publishing, confirm:

- `pyproject.toml` and `halo_forge.__version__` report package version `2.0.0a2`.
- `halo_forge.version.DISPLAY_VERSION` and `/api/public/version` report `2.0.0-alpha-2`.
- `public_app/package.json`, Tauri config, and desktop package metadata report `2.0.0-alpha-2`.
- `docs/MLX_ACCEPTANCE.md` records the latest Terminal MLX acceptance evidence.
- GitHub Actions for the latest pushed commit are green.
- `.github/workflows/release.yml` is present and configured to create/update a
  prerelease when `v2.0.0-alpha-2` is pushed.
- When Apple credentials are available, GitHub Actions secrets are configured for macOS signing and notarization:
  `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`,
  `APPLE_SIGNING_IDENTITY`, `APPLE_ID`, `APPLE_PASSWORD`, and
  `APPLE_TEAM_ID`.
- No generated `runs/*` or local `examples/datasets/*` artifacts are staged.
- `uv.lock`, `constraints/release.txt`, and `constraints/release-mlx.txt` match
  the committed dependency metadata and are staged when that metadata changes.

## Tagging

Only tag after the latest push CI and nightly qualification proof are green.
Pushing the tag starts the release workflow. For alpha, beta, and release
candidate tags, an unsigned macOS DMG is published only as a clearly named,
checksummed developer-preview asset. Stable tags never publish an unsigned
macOS DMG. Unsigned Linux and Windows candidates remain workflow artifacts:

```bash
git tag -a v2.0.0-alpha-2 -m "Halo Forge 2.0.0-alpha-2"
git push origin v2.0.0-alpha-2
gh run list --workflow Release --limit 3
gh release view v2.0.0-alpha-2
```

The release workflow can also be re-run manually from GitHub Actions with the
existing tag name. When signing credentials exist, the macOS release job must
pass these gates before its DMG is represented as supported:

```bash
codesign --verify --deep --strict --verbose=2 "Halo Forge.app"
xcrun stapler validate "Halo Forge.app"
xcrun stapler validate "Halo-Forge_2.0.0-alpha-2_aarch64.dmg"
spctl -a -vvv -t execute "Halo Forge.app"
spctl -a -vvv -t open --context context:primary-signature "Halo-Forge_2.0.0-alpha-2_aarch64.dmg"
```

Without Apple credentials, verify the prerelease contains all three preview
records and that their names and contents agree:

```bash
Halo-Forge_2.0.0-alpha-2_aarch64-unsigned-preview.dmg
Halo-Forge_2.0.0-alpha-2_aarch64-unsigned-preview.dmg.sha256
halo-forge-release-manifest.json
```

The checksum file is in `shasum --check` format. The release manifest must say
`signature_state: unsigned`, `notarization_status: not_submitted`,
`distribution_status: preview`, and `supported: false`. Confirm the generated
GitHub release is a prerelease and its warning links to the preview install
instructions. These checks close the no-cost publication gate; they do not
qualify the DMG as a trusted normal installer.

GitHub Release assets have a per-file size limit, so any oversized package is
skipped and recorded in `OVERSIZE_RELEASE_ASSETS.txt`. Unsigned Linux and
Windows candidates remain workflow preview artifacts. Do not publish a
stable/non-prerelease artifact from this beta checklist.

Current alpha release surface before the alpha-2 tag and workflow succeed:

- Published GitHub prerelease: `https://github.com/professor-moody/halo-forge/releases/tag/v2.0.0-alpha-1` (unsigned developer-test build).
- Alpha-2 release target: a checksummed macOS arm64 developer-preview DMG and
  truthful `halo-forge-release-manifest.json`. If signing credentials are
  available, the same workflow may promote it to a signed/notarized supported
  artifact after all validation gates pass.
- Linux packages: AppImage/deb candidates are built and qualified; unsigned candidates remain preview artifacts.
- Windows packages: the NSIS candidate includes the Windows runtime sidecar and packaged proof smoke; unsigned candidates remain preview artifacts.
- Canonical user docs: `https://halo-forge.io/docs/`; root `docs/` is for release and engineering artifacts.

The `v2.0.0-alpha-1` DMG was unsigned and can be rejected by macOS Gatekeeper
as damaged. Do not promote it as a normal website download. Alpha-2's preview
label and checksum make the trust state explicit; they do not remove that OS
limitation.
