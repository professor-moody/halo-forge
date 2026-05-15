# Sidecars

Release packaging should place a platform-specific executable here:

`halo-forge-runtime`

The tracked scripts are dev-runtime entrypoints for source checkouts. They prefer the repo-local `.venv`, fall back to `halo-forge`, and then to `python3 -m halo_forge.cli`.

Tauri v2 resolves target-specific sidecars during build, so CI tracks:

- `halo-forge-runtime-aarch64-apple-darwin`
- `halo-forge-runtime-x86_64-unknown-linux-gnu`

Each executable owns or locates the Halo Forge runtime and starts:

`halo-forge serve-public --host 127.0.0.1 --port 8000`
