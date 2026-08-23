---
title: "Install Halo Forge Desktop"
description: "Choose a verified desktop beta candidate or the cross-platform browser and CLI path."
weight: 7
---

Halo Forge uses one local service layer. Desktop, browser, and CLI launches open
the same datasets, runs, evaluations, reviews, and models. Browser and CLI are
supported on macOS, Linux, and Windows. The download page recommends a desktop
installer only when that exact artifact appears in the checksummed release
manifest with a passing packaged-runtime smoke result.

## Release Types

| Type | Use it for | Trust state |
|---|---|---|
| Signed/notarized macOS DMG | Normal macOS install after its release qualification passes | Verified by macOS and the Halo Forge manifest |
| Linux AppImage or Debian package | Desktop beta when listed as supported by the release manifest | Package signature and smoke state are shown before download |
| Windows NSIS installer | Desktop beta when listed as supported by the release manifest | Signature and smoke state are shown before download |
| Unsigned candidate | Engineering and preview QA only | Not presented as a trusted normal installer |

Do not treat a repository tag or filename as proof that an installer is trusted.
Halo Forge uses the release manifest's platform, architecture, package type,
runtime version, signature state, smoke result, and supported-backend record.

## Current Recommended Path

Open the [Download page](/download/) and use its recommended path. If no trusted
desktop package matches your platform, use the [Quick Start](/docs/getting-started/quickstart/)
for the browser dashboard and CLI. It provides the same managed workflow.

## Desktop Install

For a release-manifest-qualified package:

1. Download the package recommended for your platform at [halo-forge.io/download](/download/).
2. macOS: open the DMG and drag **Halo Forge** to **Applications**.
3. Linux: install the `.deb` or make the AppImage executable, according to the displayed release instructions.
4. Windows: run the qualified NSIS installer.
5. Open Halo Forge and follow the single workstation-readiness action, if one is shown.

The desktop app starts a local Halo Forge dashboard at `127.0.0.1:8765` and
stops the backend process tree it owns when the app quits. Uninstalling the
desktop shell does not delete datasets, runs, models, or review history.

The shell uses the same dashboard/API as browser and CLI installs. Its only
Dataset Lab-specific convenience is a native open dialog for choosing files or
folders on the workstation. See [Workstation Surfaces](/docs/reference/workstation-surfaces/)
for the callable contract and browser fallback.

## Unsigned macOS Developer Preview

Alpha, beta, and release-candidate GitHub prereleases may include a DMG whose
filename ends in `-unsigned-preview.dmg`. This optional path is for informed
technical testing. It is not notarized, is not supported as a normal installer,
and may be blocked by Gatekeeper. Stable releases never include it.

1. Download the DMG and matching `.sha256` file only from the official Halo
   Forge GitHub prerelease.
2. In Terminal, change to the download directory and run
   `shasum -a 256 -c Halo-Forge_*_aarch64-unsigned-preview.dmg.sha256`.
   Continue only if it reports `OK`.
3. Open the DMG. If macOS blocks the known preview and you intentionally accept
   its unsigned status, follow Apple's current
   [Open Anyway instructions](https://support.apple.com/en-us/102445) in
   **System Settings → Privacy & Security**.
4. If the checksum fails, the download source is not the official prerelease,
   or the warning differs from Apple's documented flow, stop and use the
   browser/CLI Quick Start instead.

Do not disable Gatekeeper and do not remove quarantine attributes globally.
The checksum proves that the download matches the project's artifact; it does
not provide Apple Developer ID identity or notarization.

## If The Operating System Blocks The Installer

Do not bypass Gatekeeper or Windows SmartScreen for normal use. Remove the
candidate and use the browser/CLI Quick Start path. From a startup failure or
Diagnostics, **Create support bundle** produces a previewable, privacy-safe ZIP
that you may choose to share; Halo Forge never uploads it automatically.

For source-development builds, use `apps/desktop-tauri/README.md` instead.
