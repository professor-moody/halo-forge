# Halo Forge Desktop

Tauri v2 shell for the workstation-first Halo Forge dashboard.

The desktop app is intentionally thin:

- it loads the same React dashboard built in `public_app`
- it owns one local Halo Forge service sidecar
- it waits for `/api/public/health` before showing the dashboard
- it targets macOS and Linux first

## Dev runtime sidecar

The tracked sidecar scripts are dev-runtime entrypoints. In a source checkout they prefer the repo-local `.venv` and run:

```bash
python -m halo_forge.cli serve-public --host 127.0.0.1 --port 8000
```

Tauri v2 expects target-suffixed sidecars for real builds, so macOS arm64 and Linux x86_64 scripts are tracked next to the generic `halo-forge-runtime` contract.

Release packaging should replace these scripts with a bundled Python/Halo Forge runtime binary named `halo-forge-runtime` for each target platform.

## Local smoke

```bash
cd public_app
npm ci
npm run build

cd ../apps/desktop-tauri
npm ci
npm run build
```

macOS arm64 and Linux are the v1 desktop targets. Windows is intentionally out of scope for this branch.
