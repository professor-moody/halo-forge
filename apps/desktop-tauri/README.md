# Halo Forge Desktop

Tauri v2 shell for the workstation-first Halo Forge dashboard.

The desktop app is intentionally thin:

- it starts with a native startup/status screen
- it loads the same React dashboard built in `public_app`
- it owns one local Halo Forge service sidecar
- it waits for `/api/public/health` before showing the dashboard
- it uses `127.0.0.1:8765` so it does not collide with the CLI default dashboard port
- it targets macOS and Linux first

## Runtime paths

Halo Forge Desktop supports three runtime paths:

- source development: the sidecar falls back to the repo-local `.venv`
- bundled unsigned app: Tauri launches the PyInstaller `onedir` runtime resource
- future distribution: the same bundled runtime will be signed/notarized with the app

The tracked sidecar scripts are launchers. In a source checkout they can still run:

```bash
python -m halo_forge.cli dashboard --no-build --host 127.0.0.1 --port 8765
```

For bundled unsigned builds, generate the PyInstaller runtime first:

```bash
cd apps/desktop-tauri
python3 scripts/build_runtime.py
```

This creates:

```bash
apps/desktop-tauri/runtime/dist/halo-forge-runtime/halo-forge-runtime
```

The desktop shell passes `HALO_FORGE_FRONTEND_DIST` and validates the runtime with `--desktop-self-check` before starting the dashboard. If bundled runtime validation fails, the startup screen reports the self-check error and points at the desktop runtime log.

Tauri v2 expects target-suffixed sidecars for real builds, so macOS arm64 and Linux x86_64 scripts are tracked next to the generic `halo-forge-runtime` contract.

The app targets macOS arm64 + Linux unsigned builds in this branch. Signing, notarization, Windows, and auto-update remain later release work.

Runtime logs for the dev desktop app are written to:

```bash
~/.halo-forge/desktop/runtime.log
```

## Unsigned artifact status

The current DMG/app bundle is a developer-test artifact, not a finished public installer. A successful smoke test means:

- the app starts its own loopback dashboard on `127.0.0.1:8765`
- `/api/public/health` returns ok
- Start, Train, Models, Playground, Results, and Docs load without the source dev server
- quitting the app stops only the desktop-owned service

Known release gaps remain deliberate: the app is unsigned/not notarized, the bundled runtime is large, and Linux is still a smoke/contract target. Gated or private Hugging Face repos can be connected from **Connection → Hugging Face access**; `HF_TOKEN` in the desktop runtime environment still takes precedence for ops workflows. Use open Qwen/MLX models for first serving tests.

## Local smoke

```bash
cd public_app
npm ci
npm run build

cd ../apps/desktop-tauri
npm ci
npm run build:runtime
npm run build
```

macOS arm64 and Linux are the v1 desktop targets. Windows is intentionally out of scope for this branch.

## Remote workstation checklist

Remote v1 means one Halo Forge workstation controlled over the network:

1. On the workstation, create a token with `halo-forge token create dashboard`.
2. Start the dashboard on a trusted interface: `halo-forge dashboard --host 0.0.0.0 --port 8000`.
3. Open `http://<workstation-host>:8000` from another machine on the same trusted network.
4. Paste the `hfk_...` token in Connection.
5. If the workstation will download gated/private Hugging Face models, connect the `hf_...` token in **Connection → Hugging Face access**. This is separate from the `hfk_...` API token and is stored server-side.
6. Confirm Overview, Runs, Run monitor, and Playground are reachable.

The desktop dev build itself remains loopback-local in this pass.
