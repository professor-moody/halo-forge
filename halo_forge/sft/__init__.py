"""SFT (Supervised Fine-Tuning) training module.

Two trainer backends share the same `SFTConfig`:
  - `SFTTrainer` — PyTorch (CUDA / ROCm / MPS / CPU)
  - `MLXSFTTrainer` — Apple MLX (LoRA only, Apple Silicon only)

Most callers should use `get_sft_trainer(config)` to pick the right one
based on the active accelerator. Direct imports of `SFTTrainer` /
`MLXSFTTrainer` are lazy via PEP 562 `__getattr__` so importing this
package doesn't pull torch on MLX-only hosts (or mlx_lm on torch hosts).
"""

from halo_forge.sft.config import SFTConfig
from halo_forge.sft._dispatch import get_sft_trainer


# Lazy attribute resolution. Importing `halo_forge.sft` no longer eagerly
# pulls torch (via SFTTrainer) or HF datasets (via the dataset registry) —
# they load on first access. Useful on MLX-only Mac installs and on CI
# images that probe the package without the full training stack.
_LAZY_TRAINERS = {
    "SFTTrainer": ("halo_forge.sft.trainer", "SFTTrainer"),
    "MLXSFTTrainer": ("halo_forge.sft.mlx_trainer", "MLXSFTTrainer"),
}
_LAZY_DATASETS = {
    "SFT_DATASETS": ("halo_forge.sft.datasets", "SFT_DATASETS"),
    "SFTDatasetSpec": ("halo_forge.sft.datasets", "SFTDatasetSpec"),
    "get_default_sft_dataset": ("halo_forge.sft.datasets", "get_default_sft_dataset"),
    "get_sft_dataset_spec": ("halo_forge.sft.datasets", "get_sft_dataset_spec"),
    "list_sft_datasets": ("halo_forge.sft.datasets", "list_sft_datasets"),
    "load_sft_dataset": ("halo_forge.sft.datasets", "load_sft_dataset"),
}


def __getattr__(name: str):
    target = _LAZY_TRAINERS.get(name) or _LAZY_DATASETS.get(name)
    if target is not None:
        import importlib

        module = importlib.import_module(target[0])
        return getattr(module, target[1])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SFTConfig",
    "SFTTrainer",
    "MLXSFTTrainer",
    "get_sft_trainer",
    "load_sft_dataset",
    "list_sft_datasets",
    "get_sft_dataset_spec",
    "get_default_sft_dataset",
    "SFTDatasetSpec",
    "SFT_DATASETS",
]
