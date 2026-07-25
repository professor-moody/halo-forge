---
title: "Download Halo Forge"
description: "Verified desktop beta candidates and cross-platform browser and CLI installs."
---

## Recommended Install

The recommended installer is generated from Halo Forge's checksummed release
manifest. A desktop entry is shown as supported only when its platform,
architecture, package type, bundled runtime, signature state, packaged proof-run
smoke, and supported backends have been qualified together.
The manifest records the exact runtime version used by that package.

If no trusted desktop entry matches your workstation, use the [Quick Start](/docs/getting-started/quickstart/).
The browser dashboard and CLI are supported on macOS, Linux, and Windows and use
the same managed data, training, evaluation, and model services as desktop.

## Desktop Package Matrix

| Platform | Engineering contract | Normal-user availability |
|---|---|---|
| macOS Apple Silicon | DMG plus bundled runtime and packaged proof smoke | Requires a signed and notarized qualified manifest entry |
| Linux x86-64 | AppImage and Debian package plus bundled-runtime smoke | Only entries represented as supported by the manifest |
| Windows x86-64 | NSIS plus Windows sidecar, picker, process, path, health, and proof-smoke contracts | Only entries represented as supported by the manifest |

Unsigned packages are preview artifacts. Halo Forge never asks normal users to
bypass Gatekeeper or SmartScreen.

## Data Location And Uninstall

Halo Forge keeps application data under the platform's `~/.halo-forge` logical
root. Removing the desktop shell does not remove datasets, runs, models, review
history, or support bundles.

## Browser And CLI

Use the [Quick Start](/docs/getting-started/quickstart/) for a macOS, Linux, or
Windows browser/CLI install. On first launch, Halo Forge checks runtime health,
writable roots, disk, RAM, backend availability, model access, and verified
training capabilities before recommending one next action.
