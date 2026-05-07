"""Deterministic replay (Track T15).

Given a run id (or its output directory), regenerate the exact training
trajectory. The contract is: capture every input that influenced the
run at launch time, write it to a `replay.json` manifest next to the
training_summary, and provide a CLI that materializes those inputs
back into a launch command (or directly invokes one).

What the manifest captures:
  - run_id, modality, timestamp, model_name
  - seed bundle (training seed + PYTHONHASHSEED + accelerator seed)
  - dataset identity: content hash for local files; HF id+revision for
    HF datasets so a renamed/updated dataset is detected
  - full training config snapshot (dataclass → dict)
  - environment fingerprint: Python version, halo-forge version,
    key package versions (torch / mlx / transformers / peft / trl)
  - host fingerprint: backend name + accelerator-kind detection result

What the manifest deliberately does *not* capture:
  - Wall-clock timing (non-deterministic)
  - GPU compute precision (handled by torch.use_deterministic_algorithms)
  - The full dataset contents (would bloat the manifest;
    we hash instead so a divergent dataset surfaces as a hash mismatch)

Reproducibility caveats are surfaced at replay time: if the active
host's environment fingerprint differs from the captured one,
`halo-forge replay` warns loudly. We don't refuse — sometimes
re-running on a different host is the *point* — but the user has the
information to interpret divergence.
"""

from halo_forge.replay.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    EnvironmentFingerprint,
    ReplayManifest,
    capture_manifest,
    compare_environments,
    hash_dataset_file,
    load_manifest,
    save_manifest,
)

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "EnvironmentFingerprint",
    "ReplayManifest",
    "capture_manifest",
    "compare_environments",
    "hash_dataset_file",
    "load_manifest",
    "save_manifest",
]
