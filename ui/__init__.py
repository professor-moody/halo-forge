"""halo-forge service layer.

The historical NiceGUI web UI was retired in favor of the Vite + React
frontend at `public_app/`. This package keeps only the *service* layer
(`ui.services.*`) and the in-memory job state (`ui.state`) that
`halo_forge.public_api` consumes — neither has any UI dependency, both
just predate the rename to a more honest module path.

If you're looking for the GUI, see `public_app/`.
"""

__version__ = "1.4.0"
