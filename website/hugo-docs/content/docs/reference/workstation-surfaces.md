---
title: "Workstation Surfaces"
description: "Desktop, local browser, remote browser, and CLI parity with honest file and platform semantics"
weight: 19
---

Halo Forge has one catalog and service layer with four equal operator surfaces:

| Surface | Best for | File semantics |
|---|---|---|
| Desktop shell | Local interactive operation | Native chooser returns paths on the Halo Forge workstation |
| Local browser | Normal source/CLI install | Browser upload and explicit workstation paths |
| Remote browser | Operating a trusted host from another machine | Token-authenticated workflows; client-local paths are never treated as host paths |
| CLI | Automation and headless work | Explicit paths/flags over the same managed catalog |

Dataset versions, jobs, runs, evaluations, and artifacts keep the same IDs and
provenance regardless of the launching surface. Desktop is a thin Tauri shell
around the same React dashboard and API.

## Native Dataset Chooser

The public dashboard exports:

```ts
declare function pickDatasetSource(request: {
  kind: "file" | "folder",
  multiple?: boolean,
}): Promise<{ paths: string[] } | null>
```

from `@/lib/desktop-bridge`. Inside Tauri, the same callable is installed at
`window.haloForgeDesktop?.pickDatasetSource`. Cancel and non-desktop use return
`null`. Invocation failures reject so the UI can show the problem.

The desktop grants only `dialog:allow-open`, only to its main window and its
owned `http://127.0.0.1:8765/*` dashboard origin. It does not grant save dialogs
or arbitrary remote origins.

## Remote Access

```bash
halo-forge token create dashboard
halo-forge dashboard --host 0.0.0.0 --port 8000
```

Open `http://<workstation-host>:8000` and paste the token in **Connection**.
Treat this as trusted-network access to the workstation. Compute, paths, data,
and artifacts remain on the host.

See [Hardware and Capability Notes](/docs/getting-started/hardware/) for the
backend and desktop-distribution matrix.
