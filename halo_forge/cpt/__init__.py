"""Continued causal pretraining (CPT).

Both backends use the same explicit ``CPTConfig`` and deterministic corpus
packing contract:

* ``CPTTrainer`` — Hugging Face/PyTorch, checkpoint-resumable.
* ``MLXCPTTrainer`` — native MLX, final/full-trial execution.
"""

from __future__ import annotations

from halo_forge.cpt._dispatch import get_cpt_trainer
from halo_forge.cpt.config import CPTConfig
from halo_forge.cpt.packing import (
    PACKING_ALGORITHM,
    CorpusPackingPlan,
    PackedCorpusSequence,
    PackedCorpusSplit,
    build_corpus_packing_plan,
    model_identity_hash,
    pack_corpus_records,
    packing_plan_hash,
    split_paragraphs,
    tokenizer_identity_hash,
)

_LAZY_TRAINERS = {
    "CPTTrainer": ("halo_forge.cpt.trainer", "CPTTrainer"),
    "MLXCPTTrainer": ("halo_forge.cpt.mlx_trainer", "MLXCPTTrainer"),
}


def __getattr__(name: str):
    target = _LAZY_TRAINERS.get(name)
    if target is not None:
        import importlib

        module = importlib.import_module(target[0])
        return getattr(module, target[1])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PACKING_ALGORITHM",
    "CPTConfig",
    "CPTTrainer",
    "CorpusPackingPlan",
    "MLXCPTTrainer",
    "PackedCorpusSequence",
    "PackedCorpusSplit",
    "build_corpus_packing_plan",
    "get_cpt_trainer",
    "model_identity_hash",
    "pack_corpus_records",
    "packing_plan_hash",
    "split_paragraphs",
    "tokenizer_identity_hash",
]
