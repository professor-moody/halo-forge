---
title: "Public Frontend"
description: "One managed workstation interface across desktop, local browser, remote browser, and CLI"
weight: 6
---

The public Halo Forge frontend is the default product surface for normal training work. It is a Vite + React + TanStack Router app in `public_app/`, backed by the FastAPI public API under `/api/public/*`.

Use it for own-data preparation, guided or advanced Train launches, live run
monitoring, completed-run inspection, model selection, evaluation, verifier
work, serving tests, and remote workstation access.

Desktop, local-browser, remote-browser, and CLI operations use the same
catalog and services. See [Workstation Surfaces](/docs/reference/workstation-surfaces/)
for host-path and platform semantics.

## Run locally

```bash
halo-forge dashboard
```

Open `http://127.0.0.1:8000`. The `dashboard` command serves the public API and built React dashboard from one origin. If `public_app/dist` is missing in a source checkout, it builds the dashboard assets first. `halo-forge app` is an alias.

For a no-bind check:

```bash
halo-forge dashboard --check
```

For frontend development, run the API and Vite separately:

```bash
# Terminal 1: dashboard API
halo-forge serve-public

# Optional no-bind check
halo-forge serve-public --check

# Terminal 2: React app
cd public_app
npm install
npm run dev
```

Open `http://127.0.0.1:3000`. In development, Vite proxies `/api/*` to `http://127.0.0.1:8000`.

Run repeatable desktop screenshot QA with:

```bash
cd public_app
npm run qa:visual
```

## Product flow

- **Overview**: workstation health, current work, recent runs, and next actions.
- **Data**: add your own file/folder data, build immutable versions, and review labels.
- **Train**: guided and advanced launches for every capability-declared method.
- **Experiments**: deterministic repeats, sweeps, boundaries, and evidence.
- **Runs**: active and completed work, metrics, data bindings, evaluations, artifacts, logs, recovery, and comparison.
- **Evaluate**: suites, launches, immutable evidence, comparisons, failure review, verifiers, and training audits.
- **Models**: catalog, trained artifacts, cached models, and **Serve & Test**.
- **Activity**: queue, resource owner, blockers, progress, telemetry, retry history, and direct actions.
- **System / Connection**: diagnostics, tokens, remote connection, and workstation configuration.

Legacy deep links may redirect, but current documentation and controls use these
names exclusively.

## Serve from the dashboard

Serving v1 manages one local `halo-forge serve` process at a time on `127.0.0.1:8001`.

1. Open **Models**.
2. Pick a small model such as `mlx-community/Qwen2.5-0.5B-Instruct-bf16`.
3. Click **Serve**.
4. Open **Models → Serve & Test** and wait for **Local serving** to show
   `ready`.
5. Send a message.
6. Click **Stop** before serving a different model.

Completed runs expose their artifacts from **Runs → Artifacts** and the shared
**Models** library. If another model is already serving, the dashboard blocks
the second launch and tells you to stop the current server first.

## Desktop app development

The desktop app shell lives in `apps/desktop-tauri`. It is a Tauri v2 wrapper around the same dashboard, with a dev sidecar that starts:

```bash
halo-forge serve-public --host 127.0.0.1 --port 8000
```

Local smoke:

```bash
cd public_app
npm ci
npm run build

cd ../apps/desktop-tauri
npm ci
npm run build
```

The desktop build matrix covers macOS arm64 DMG, Linux x86-64 AppImage/deb, and
Windows x86-64 NSIS candidates. Browser and CLI installs are supported on all
three operating systems. The verified release manifest—not a source version—
determines whether a desktop candidate is supported or preview-only. Unsigned
candidates are never presented as trusted normal installers.

The desktop shell additionally exposes a narrowly scoped native file/folder
chooser for Dataset Lab. Browser surfaces keep upload and explicit host-path
fallbacks; they do not lose workflow parity.

## Remote workstation

Remote v1 means one Halo Forge machine with the accelerator is exposed to a trusted network, and another browser controls that same workstation. It is not a worker registry, cloud queue, or distributed scheduler.

### Workstation setup

```bash
# On the training workstation
halo-forge token create dashboard
halo-forge dashboard --host 0.0.0.0 --port 8000
```

Save the token when it is printed. Halo Forge stores only a hash in `~/.halo-forge/tokens.json`; the bearer secret is shown once.

For frontend development against a remote workstation, use the two-process dev setup instead:

```bash
halo-forge serve-public --host 0.0.0.0 --port 8000

cd public_app
npm install
npm run dev -- --host 0.0.0.0
```

### Browser setup

1. Open `http://<workstation-host>:8000` from the remote device.
2. Go to **Connection**.
3. Paste the `hfk_...` token.
4. Click **Save and test**.
5. Launch or monitor runs against that workstation.

Loopback requests stay zero-config. Non-loopback requests require `Authorization: Bearer <token>` for every `/api/public/*` endpoint. The frontend stores the token in `localStorage["halo-forge:api-token"]` and attaches it automatically.

## Internal console

The retired NiceGUI product surface is no longer the primary user workflow. Internal modules under `ui/services/` still provide service-layer behavior consumed by the public API, and any staff-only diagnostics should remain behind advanced/internal docs.

Use the public frontend for default product workflows. Use internal tools only for raw traces, development diagnostics, or low-level remediation.

## Product rules

- One primary action per state.
- Plain-language status first, research detail second.
- Train Guided is the first-run path; Train Advanced exposes direct configuration.
- Remote v1 controls one workstation and uses bearer tokens.
- Capability copy should follow backend/readiness truth, not hand-written claims.
