# Halo Forge Desktop

Tauri v2 shell for the workstation-first Halo Forge dashboard.

The desktop app is intentionally thin:

- it loads the same React dashboard built in `public_app`
- it owns one local Halo Forge service sidecar
- it waits for `/api/public/health` before showing the dashboard
- it targets macOS and Linux first

The tracked sidecar placeholder documents the contract. Release packaging should replace it with a bundled Python/Halo Forge runtime binary named `halo-forge-runtime` for each target platform.
