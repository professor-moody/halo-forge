"""Replay manifest capture + load + diff (Track T15)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


MANIFEST_VERSION = 1
MANIFEST_FILENAME = "replay.json"


# ---------- environment fingerprint ----------------------------------------


_TRACKED_PACKAGES = (
    "halo_forge",
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "mlx",
    "mlx_lm",
    "datasets",
    "bitsandbytes",
    "vllm",
    "numpy",
)


@dataclass
class EnvironmentFingerprint:
    """Snapshot of the host + package state at run time.

    Compared at replay time to surface divergence (different torch,
    different MLX, different OS) without refusing the replay outright.
    """

    python: str
    platform: str
    backend: str
    packages: Dict[str, Optional[str]] = field(default_factory=dict)

    @classmethod
    def capture(cls) -> "EnvironmentFingerprint":
        return cls(
            python=sys.version.split()[0],
            platform=f"{platform.system()} {platform.machine()}",
            backend=_detect_backend_name_safe(),
            packages={name: _get_pkg_version(name) for name in _TRACKED_PACKAGES},
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _detect_backend_name_safe() -> str:
    try:
        from halo_forge.backend import get_backend

        return get_backend().name
    except Exception:
        return "unknown"


def _get_pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(name)
    except Exception:
        return None


# ---------- dataset hashing ------------------------------------------------


def hash_dataset_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """SHA-256 over a local dataset file. Returns the hex digest.

    Streams the file so multi-GB JSONLs don't load into memory. The
    digest is stable across hosts; a hash mismatch at replay time
    means the file changed (rebuilt, mistakenly overwritten, swapped).
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------- manifest -------------------------------------------------------


@dataclass
class ReplayManifest:
    """Everything we need to re-launch a run identically."""

    manifest_version: int
    run_id: str
    modality: str
    timestamp: str
    model_name: str
    seed: int
    pythonhashseed: Optional[str]
    config: Dict[str, Any]
    dataset: Dict[str, Any]
    environment: Dict[str, Any]
    cli_args: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def capture_manifest(
    *,
    run_id: str,
    modality: str,
    model_name: str,
    seed: int,
    config: Any,
    dataset_file: Optional[Path] = None,
    dataset_id: Optional[str] = None,
    dataset_revision: Optional[str] = None,
    cli_args: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> ReplayManifest:
    """Build a `ReplayManifest` for the active run."""
    import os

    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        config_dict = dataclasses.asdict(config)
    elif isinstance(config, dict):
        config_dict = dict(config)
    else:
        config_dict = {"value": str(config)}

    dataset: Dict[str, Any] = {}
    if dataset_file is not None:
        path = Path(dataset_file)
        if path.exists() and path.is_file():
            dataset = {
                "kind": "local_file",
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": hash_dataset_file(path),
            }
        else:
            dataset = {"kind": "local_file", "path": str(path), "missing": True}
    elif dataset_id:
        dataset = {
            "kind": "huggingface",
            "id": dataset_id,
            "revision": dataset_revision,
        }

    return ReplayManifest(
        manifest_version=MANIFEST_VERSION,
        run_id=run_id,
        modality=modality,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z") or "",
        model_name=model_name,
        seed=int(seed),
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        config=config_dict,
        dataset=dataset,
        environment=EnvironmentFingerprint.capture().to_dict(),
        cli_args=list(cli_args or []),
        notes=notes,
    )


def save_manifest(manifest: ReplayManifest, output_dir: Path) -> Path:
    """Write the manifest to `<output_dir>/replay.json` and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest.to_dict(), indent=2, default=str))
    return path


def load_manifest(source: Path) -> ReplayManifest:
    """Load a manifest from a path that's either the manifest file
    itself or the run's output directory.

    Tolerates older / future manifest_version fields by warning instead
    of refusing — the captured config + seed are forward/backward
    compatible enough that an older client can usually still replay.
    """
    p = Path(source)
    if p.is_dir():
        p = p / MANIFEST_FILENAME
    if not p.exists():
        raise FileNotFoundError(f"Replay manifest not found at {p}")

    data = json.loads(p.read_text())
    version = data.get("manifest_version", 0)
    if version != MANIFEST_VERSION:
        logger.warning(
            "Replay manifest version %s does not match current %s — "
            "config + seed should still replay; environment diff may be skewed.",
            version, MANIFEST_VERSION,
        )

    # Construct via field-by-field assignment so missing fields don't
    # explode on older manifests. We treat absent fields as None / [].
    return ReplayManifest(
        manifest_version=int(data.get("manifest_version", 0)),
        run_id=str(data.get("run_id") or ""),
        modality=str(data.get("modality") or ""),
        timestamp=str(data.get("timestamp") or ""),
        model_name=str(data.get("model_name") or ""),
        seed=int(data.get("seed") or 0),
        pythonhashseed=data.get("pythonhashseed"),
        config=dict(data.get("config") or {}),
        dataset=dict(data.get("dataset") or {}),
        environment=dict(data.get("environment") or {}),
        cli_args=list(data.get("cli_args") or []),
        notes=data.get("notes"),
    )


# ---------- environment diff -----------------------------------------------


def compare_environments(
    captured: Dict[str, Any], current: Dict[str, Any]
) -> Dict[str, Any]:
    """Produce a structured diff between two environment fingerprints.

    Returns:
        ``{
            "matched": bool,
            "differences": [
                {"key": "torch.version", "captured": "2.4.0", "current": "2.5.1"},
                {"key": "platform", "captured": "Darwin arm64", "current": "Linux x86_64"},
                ...
            ]
        }``
    """
    differences: List[Dict[str, Any]] = []

    for top_key in ("python", "platform", "backend"):
        if captured.get(top_key) != current.get(top_key):
            differences.append({
                "key": top_key,
                "captured": captured.get(top_key),
                "current": current.get(top_key),
            })

    captured_pkgs = captured.get("packages") or {}
    current_pkgs = current.get("packages") or {}
    all_keys = set(captured_pkgs) | set(current_pkgs)
    for k in sorted(all_keys):
        if captured_pkgs.get(k) != current_pkgs.get(k):
            differences.append({
                "key": f"packages.{k}",
                "captured": captured_pkgs.get(k),
                "current": current_pkgs.get(k),
            })

    return {
        "matched": not differences,
        "differences": differences,
    }


__all__ = [
    "MANIFEST_VERSION",
    "MANIFEST_FILENAME",
    "EnvironmentFingerprint",
    "ReplayManifest",
    "capture_manifest",
    "save_manifest",
    "load_manifest",
    "compare_environments",
    "hash_dataset_file",
]
