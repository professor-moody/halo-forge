# Workstation Surfaces And Desktop Bridge

Halo Forge has one service layer and four equal operator surfaces. Dataset
versions, work items, runs, evaluations, and model artifacts remain the same
objects regardless of where an operation is launched.

| Surface | Availability | Operator notes |
|---|---|---|
| Desktop shell | macOS arm64, Linux x86-64, and Windows x86-64 build contracts; see distribution status below | Thin Tauri wrapper around the same dashboard and local API |
| Local browser | macOS, Linux, or Windows Halo Forge host | Default source-install workflow at `127.0.0.1:8000` |
| Remote browser | Any modern browser that can reach a supported host | Non-loopback API access requires a Halo Forge bearer token |
| CLI | Python 3.10–3.13 on macOS, Linux, or Windows | Automation/headless surface over the same managed catalog |

“Equal” means workflow and object parity. It does not mean a remote browser can
read files from the workstation or that every accelerator supports every
trainer. File paths are always interpreted on the Halo Forge host.

## Desktop distribution truth

| Platform | Current state |
|---|---|
| macOS arm64 | DMG build, bundled runtime, ownership/shutdown, health, and packaged proof-run contracts. A normal installer requires a signed and notarized release-manifest entry. |
| Linux x86-64 | AppImage and Debian package build, bundled-runtime smoke, upgrade, and uninstall-preservation contracts. Unsigned packages remain preview artifacts. |
| Windows x86-64 | NSIS build with a Windows runtime sidecar, native picker, Windows paths, process-tree shutdown, health check, and packaged proof-run smoke. Unsigned packages remain preview artifacts. |

Browser and CLI are supported on all three operating systems. Desktop
availability is determined from the verified release manifest, not from a
version string in source. Do not bypass Gatekeeper or SmartScreen for normal
use. Uninstalling the desktop shell preserves datasets, runs, models, reviews,
and other state under the platform's `~/.halo-forge` logical root.

## Native dataset chooser contract

The dashboard exports this stable helper from
`public_app/src/lib/desktop-bridge.ts`:

```ts
declare function pickDatasetSource(request: {
  kind: "file" | "folder",
  multiple?: boolean,
}): Promise<{ paths: string[] } | null>
```

The same callable is available as
`window.haloForgeDesktop?.pickDatasetSource(...)` after the bridge module loads
inside Tauri. A selection returns absolute workstation paths. Cancel,
non-desktop use, or an unavailable desktop runtime returns `null`; an actual
desktop invocation failure rejects so the UI can explain it.

The Tauri capability grants `dialog:allow-open` only to the `main` window and
the desktop-owned `http://127.0.0.1:8765/*` origin. Save dialogs and arbitrary
remote origins are not granted. Browser workflows must retain upload and
explicit-host-path fallbacks.

## Capability truth

The UI and CLI must use runtime capability/preflight results, not platform
marketing shorthand:

- CUDA and ROCm are PyTorch paths; vLLM, bitsandbytes, and particular model
  adapters remain optional and environment-dependent.
- Apple MPS is the PyTorch path on Apple Silicon. Apple MLX has dedicated SFT,
  RAFT, DPO, and GRPO implementations; it is not a universal replacement for
  every trainer.
- CPU is for metadata work, validation, and tiny smoke runs, not a promised
  heavy-training target.
- Missing GPU telemetry is shown as unavailable, never fabricated as zero.
- Disk, RAM, model, tokenizer, trainer adapter, and optional-package checks can
  still refuse a launch on an otherwise supported host.
