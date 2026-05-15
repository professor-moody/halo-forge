# Sidecars

The files in this directory are launcher sidecars. They first look for the
bundled PyInstaller runtime path passed by the Tauri shell:

`HALO_FORGE_BUNDLED_RUNTIME/halo-forge-runtime`

If no bundled runtime is available, they are dev-runtime entrypoints for source checkouts. When `HALO_FORGE_REPO_ROOT` is set by the desktop app, they require the repo-local `.venv` and fail loudly if it is missing or unsupported. Outside the desktop app they still prefer `.venv`, then fall back to `halo-forge`, and then to `python3 -m halo_forge.cli`.

Tauri v2 resolves target-specific sidecars during build, so CI tracks:

- `halo-forge-runtime-aarch64-apple-darwin`
- `halo-forge-runtime-x86_64-unknown-linux-gnu`

Each executable owns or locates the Halo Forge runtime and starts:

`halo-forge dashboard --no-build --host 127.0.0.1 --port 8765`

Build the bundled runtime with:

`cd apps/desktop-tauri && python3 scripts/build_runtime.py`
