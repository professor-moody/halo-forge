# Sidecars

Release packaging should place a platform-specific executable here:

`halo-forge-runtime`

That executable owns the bundled Python runtime and starts:

`halo-forge serve-public --host 127.0.0.1 --port 8000`
