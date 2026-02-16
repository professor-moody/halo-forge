"""Data generation and preparation modules.

Keep optional dependency imports lazy so validation/formatting paths work in
minimal environments that do not install full dataset download stacks.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from halo_forge.data.formatters import format_for_training

if TYPE_CHECKING:
    from halo_forge.data.llm_generate import TopicSpec, TrainingDataGenerator
    from halo_forge.data.public_datasets import DatasetPreparer, DatasetSpec

__all__ = [
    "DatasetPreparer",
    "DatasetSpec",
    "TrainingDataGenerator",
    "TopicSpec",
    "format_for_training",
]

_LAZY_EXPORTS = {
    "DatasetPreparer": ("halo_forge.data.public_datasets", "DatasetPreparer"),
    "DatasetSpec": ("halo_forge.data.public_datasets", "DatasetSpec"),
    "TrainingDataGenerator": ("halo_forge.data.llm_generate", "TrainingDataGenerator"),
    "TopicSpec": ("halo_forge.data.llm_generate", "TopicSpec"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
