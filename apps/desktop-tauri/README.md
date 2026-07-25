# Halo Forge Desktop

Tauri v2 shell for the workstation-first Halo Forge dashboard.

The desktop app is intentionally thin:

- it starts with a native startup/status screen
- it loads the same React dashboard built in `public_app`
- it owns one local Halo Forge service sidecar
- it waits for `/api/public/health` before showing the dashboard
- it uses `127.0.0.1:8765` so it does not collide with the CLI default dashboard port
- it has package contracts for macOS arm64, Linux x86-64, and Windows x86-64

## Runtime paths

Halo Forge Desktop supports three runtime paths:

- source development: the sidecar falls back to the repo-local `.venv`
- packaged candidate: Tauri launches the platform's PyInstaller `onedir` runtime resource
- qualified release: the download path is selected from signature, runtime-smoke, and platform evidence

The tracked sidecar scripts are launchers. In a source checkout they can still run:

```bash
python -m halo_forge.cli dashboard --no-build --host 127.0.0.1 --port 8765
```

For bundled unsigned builds, generate the PyInstaller runtime first:

```bash
cd apps/desktop-tauri
python3 scripts/build_runtime.py
```

This creates a platform executable under:

```bash
apps/desktop-tauri/runtime/dist/halo-forge-runtime/
```

The desktop shell passes `HALO_FORGE_FRONTEND_DIST` and validates the runtime with `--desktop-self-check` before starting the dashboard. If bundled runtime validation fails, the startup screen reports the self-check error and points at the desktop runtime log.

The shell starts that bundled executable directly, including
`halo-forge-runtime.exe` on Windows. It records the child process identity,
owns only that process tree, and performs runtime self-check before navigation.

Runtime logs for the dev desktop app are written to:

```bash
~/.halo-forge/desktop/runtime.log
```

## Distribution status

The build matrix produces macOS arm64 DMG, Linux x86-64 AppImage/deb, and
Windows x86-64 NSIS candidates. Unsigned candidates are preview artifacts, not
trusted public installers. The website recommends only artifacts represented
as supported by the checksummed release manifest. A successful smoke test means:

- the app starts its own loopback dashboard on `127.0.0.1:8765`
- `/api/public/health` returns ok
- Data, Train, Experiments, Runs, Evaluate, Models, and Docs load without the source dev server
- quitting the app stops only the desktop-owned service process tree
- the packaged fixture reaches a real proof-run optimizer update

macOS signing/notarization is a credential-backed publication gate. Linux and
Windows signatures are reported truthfully; unsigned candidates stay preview
only. No normal-user documentation directs users around Gatekeeper or
SmartScreen. Gated or private Hugging Face repos can be connected from
**Connection → Hugging Face access**; `HF_TOKEN` in the desktop runtime
environment still takes precedence for ops workflows.

Repository metadata or a version tag does not prove a package is trusted. The
distribution-capability record pins the platform, architecture, package type,
runtime version, signature state, smoke result, and supported backends.

## Native dataset chooser

The dashboard's stable bridge is exported from
`public_app/src/lib/desktop-bridge.ts`:

```ts
declare function pickDatasetSource(request: {
  kind: "file" | "folder";
  multiple?: boolean;
}): Promise<{ paths: string[] } | null>
```

Inside Tauri it is also available as
`window.haloForgeDesktop?.pickDatasetSource(...)`. Paths always belong to the
desktop workstation. Cancel and non-desktop use return `null`; invocation
failures reject. The capability grants only `dialog:allow-open` to the `main`
window and the desktop-owned `http://127.0.0.1:8765/*` origin. Browser flows
must keep upload and explicit workstation-path fallbacks.

## Local smoke

```bash
cd public_app
npm ci
npm run build

cd ../apps/desktop-tauri
npm ci
npm run build:runtime
npm run build
npm run smoke:runtime
```

`npm run smoke:runtime` runs the packaged runtime when available. It performs
the beta training gate with a tiny open SFT model,
confirms optimizer steps execute, checks that `training_summary.json` and
`final_model/` are written, starts the packaged dashboard service, and verifies
the normal operator routes return dashboard HTML.

CI and Release build all three desktop targets. Browser and CLI remain the
supported cross-platform fallback when a desktop candidate is preview-only.

## What users should see during training

After a Train launch, the run monitor should feel active immediately:

- **Prepare**: the dashboard has accepted the launch and is creating run state.
- **Data**: datasets or prompt files are being loaded and checked.
- **Model**: model files are being resolved or downloaded.
- **Trainer**: the local trainer is being constructed.
- **Train**: optimizer steps are running; step count and loss should update from real training events.
- **Save / Finalize**: checkpoints, summaries, and final artifacts are being written.
- **Done or Failed**: the page should show Run, Artifact/Serve, Compare, retry, or Diagnostics actions instead of leaving the user with raw logs.

If no loss is visible yet, the copy should explain the current stage, for example
“Loading dataset” or “Waiting for the first optimizer step.” Dataset preprocessing
progress must not be shown as optimizer steps.

## Remote workstation checklist

Remote v1 means one Halo Forge workstation controlled over the network:

1. On the workstation, create a token with `halo-forge token create dashboard`.
2. Start the dashboard on a trusted interface: `halo-forge dashboard --host 0.0.0.0 --port 8000`.
3. Open `http://<workstation-host>:8000` from another machine on the same trusted network.
4. Paste the `hfk_...` token in Connection.
5. If the workstation will download gated/private Hugging Face models, connect the `hf_...` token in **Connection → Hugging Face access**. This is separate from the `hfk_...` API token and is stored server-side.
6. Confirm Overview, Runs, Run monitor, and Models → Serve & Test are reachable.

The desktop dev build itself remains loopback-local in this pass.
