"""Optional trainer-to-managed-sink boundary lifecycle bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def complete_signal_boundary(
    sink: Any,
    *,
    boundary_value: Any,
    checkpoint_path: str | Path,
) -> Optional[Any]:
    """Hash a published checkpoint and notify a managed multi-boundary sink.

    Plain ``TrainingSignalSink`` instances intentionally do not implement the
    lifecycle hook; their caller seals them explicitly.  Managed execution can
    supply a coordinator exposing ``boundary_complete``.  Hashing only occurs
    when that hook exists, so historical launches pay no extra checkpoint I/O.
    """

    hook = getattr(sink, "boundary_complete", None)
    if not callable(hook):
        return None
    from halo_forge.artifact_lab.hashing import hash_path

    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint_hash = hash_path(checkpoint).content_hash
    return hook(
        boundary_value=boundary_value,
        checkpoint_hash=checkpoint_hash,
    )


__all__ = ["complete_signal_boundary"]
