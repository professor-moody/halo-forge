# Halo Forge Desktop

Tauri v2 shell for the workstation-first Halo Forge dashboard.

The desktop app is intentionally thin:

- it starts with a native startup/status screen
- it loads the same React dashboard built in `public_app`
- it owns one local Halo Forge service sidecar
- it waits for `/api/public/health` before showing the dashboard
- it uses `127.0.0.1:8765` so it does not collide with the CLI default dashboard port
- it targets macOS and Linux first

## Dev runtime sidecar

The tracked sidecar scripts are dev-runtime entrypoints. In a source checkout they prefer the repo-local `.venv` and run:

```bash
python -m halo_forge.cli dashboard --no-build --host 127.0.0.1 --port 8765
```

This is not yet a fully self-contained Python runtime. If `.venv` is missing, uses an unsupported Python version, or cannot import `halo_forge.cli`, the startup screen reports the error and points at the desktop runtime log.

Tauri v2 expects target-suffixed sidecars for real builds, so macOS arm64 and Linux x86_64 scripts are tracked next to the generic `halo-forge-runtime` contract.

Release packaging should replace these scripts with a bundled Python/Halo Forge runtime binary named `halo-forge-runtime` for each target platform.

Runtime logs for the dev desktop app are written to:

```bash
~/.halo-forge/desktop/runtime.log
```

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

## Remote workstation checklist

Remote v1 means one Halo Forge workstation controlled over the network:

1. On the workstation, create a token with `halo-forge token create dashboard`.
2. Start the dashboard on a trusted interface: `halo-forge dashboard --host 0.0.0.0 --port 8000`.
3. Open `http://<workstation-host>:8000` from another machine on the same trusted network.
4. Paste the `hfk_...` token in Connection.
5. Confirm Overview, Runs, Run monitor, and Playground are reachable.

The desktop dev build itself remains loopback-local in this pass.
