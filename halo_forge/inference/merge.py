"""Adapter / model merging (Tracks T12 + T13).

Two operations users need after training:

1. **Bake (T13):** "I have a LoRA adapter that worked, give me a single
   merged checkpoint I can ship." Standard merge_and_unload through
   peft. Saves the merged base+adapter to a new directory.

2. **Combine (T12):** "I have N LoRA adapters, give me one merged
   adapter blending their behaviors." Supports four combination
   strategies via peft's `add_weighted_adapter`:

       linear         — straight weighted sum (a₁·w₁ + a₂·w₂ + …)
       ties           — Tang et al. 2023; resolves sign conflicts +
                         keeps top-k magnitudes per parameter
       dare_linear    — DARE pruning + linear; trims redundant deltas
       dare_ties      — DARE + TIES; current best general-purpose

We expose normalized names so the CLI vocabulary doesn't leak peft
version detail; new combination types in upstream peft can be added by
extending the table.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# Normalized merge method → (peft combination_type, optional kwargs).
# Adding a new method = a new row here; the CLI vocabulary stays stable.
_MERGE_METHOD_TABLE: Dict[str, Dict[str, Any]] = {
    "linear": {"combination_type": "linear"},
    "ties": {"combination_type": "ties_svd"},  # SVD variant is the default-better
    "dare_linear": {"combination_type": "dare_linear", "density": 0.5},
    "dare_ties": {"combination_type": "dare_ties", "density": 0.5},
    "magnitude_prune": {"combination_type": "magnitude_prune", "density": 0.5},
}


@dataclass
class MergeResult:
    """Outcome of a merge operation, returned to the CLI for printing."""

    operation: str  # "bake" or "combine"
    output_path: str
    method: str  # "bake" or one of _MERGE_METHOD_TABLE keys
    base_model: str
    adapters: List[str]
    weights: List[float]
    bytes_written: Optional[int] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def list_supported_methods() -> List[str]:
    """The merge methods the `combine` operation accepts."""
    return list(_MERGE_METHOD_TABLE)


def _resolve_method_kwargs(method: str) -> Dict[str, Any]:
    canonical = method.strip().lower()
    if canonical not in _MERGE_METHOD_TABLE:
        raise ValueError(
            f"Unknown merge method {method!r}. Choose from: {', '.join(_MERGE_METHOD_TABLE)}"
        )
    return dict(_MERGE_METHOD_TABLE[canonical])


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def bake_adapter(
    *,
    base_model: str,
    adapter_path: str,
    output_path: str,
    trust_remote_code: bool = False,
) -> MergeResult:
    """Merge a single LoRA adapter into its base model and save the
    full merged checkpoint (Track T13).

    The output is a standard HuggingFace checkpoint — load it back with
    `AutoModelForCausalLM.from_pretrained(output_path)` and you get the
    behavior of base+adapter, with no LoRA infrastructure required.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Bake: %s + %s → %s", base_model, adapter_path, output_path)

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        device_map="cpu",  # Bake is a CPU-bound merge; no need to load on accelerator
    )
    peft_model = PeftModel.from_pretrained(base, adapter_path)
    merged = peft_model.merge_and_unload()

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out))

    # Save the tokenizer alongside so the merged checkpoint is a
    # complete drop-in. We pull it from the adapter dir first (it may
    # have been retrained with custom tokens), falling back to the base.
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, trust_remote_code=trust_remote_code
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=trust_remote_code
        )
    tokenizer.save_pretrained(str(out))

    return MergeResult(
        operation="bake",
        output_path=str(out),
        method="bake",
        base_model=base_model,
        adapters=[adapter_path],
        weights=[1.0],
        bytes_written=_dir_size(out),
        notes="Merged checkpoint is loadable via AutoModelForCausalLM.",
    )


def combine_adapters(
    *,
    base_model: str,
    adapter_paths: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    method: str = "dare_ties",
    output_path: str,
    bake_after_merge: bool = False,
    trust_remote_code: bool = False,
    svd_rank: Optional[int] = None,
) -> MergeResult:
    """Combine multiple LoRA adapters into one (Track T12).

    Args:
        base_model: HF id / local path for the base the adapters trained against.
        adapter_paths: Two or more LoRA adapter directories. They must
            target the same base model (peft enforces this).
        weights: Per-adapter weights. Defaults to uniform if omitted.
        method: Normalized merge method (linear / ties / dare_ties / ...).
        output_path: Directory the combined adapter is saved to.
        bake_after_merge: If True, also merge the combined adapter into
            the base and save the full checkpoint alongside (T12 + T13
            in one go).
        svd_rank: Override the rank of the SVD reconstruction for
            ties / dare_ties. None lets peft pick.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if len(adapter_paths) < 2:
        raise ValueError("combine_adapters needs at least two adapters")
    if weights is None:
        weights = [1.0 / len(adapter_paths)] * len(adapter_paths)
    if len(weights) != len(adapter_paths):
        raise ValueError(
            f"weights length {len(weights)} != adapters length {len(adapter_paths)}"
        )

    method_kwargs = _resolve_method_kwargs(method)
    if svd_rank is not None and "svd" in method_kwargs.get("combination_type", ""):
        method_kwargs["svd_rank"] = svd_rank

    logger.info(
        "Combine: %d adapters via %s onto %s → %s",
        len(adapter_paths), method, base_model, output_path,
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        device_map="cpu",
    )

    # Load the first adapter, then add the rest by name. peft's
    # `add_weighted_adapter` operates on adapters that are already loaded
    # into the same PeftModel, so we walk the list and load each.
    peft_model = PeftModel.from_pretrained(
        base, adapter_paths[0], adapter_name="adapter_0"
    )
    for i, p in enumerate(adapter_paths[1:], start=1):
        peft_model.load_adapter(p, adapter_name=f"adapter_{i}")

    # Hand all loaded names to add_weighted_adapter. It writes a new
    # adapter to the model under the requested combined name.
    combined_name = "halo_forge_combined"
    adapter_names = [f"adapter_{i}" for i in range(len(adapter_paths))]
    peft_model.add_weighted_adapter(
        adapters=adapter_names,
        weights=list(weights),
        adapter_name=combined_name,
        **method_kwargs,
    )
    peft_model.set_adapter(combined_name)

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    if bake_after_merge:
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(str(out))
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                adapter_paths[0], trust_remote_code=trust_remote_code
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=trust_remote_code
            )
        tokenizer.save_pretrained(str(out))
        notes = "Combined adapters then baked into base; output is a merged checkpoint."
    else:
        # Save just the combined adapter, not the base. Keeps the output
        # small (~LoRA size) and reusable on top of any compatible base.
        peft_model.save_pretrained(
            str(out), selected_adapters=[combined_name]
        )
        notes = "Combined adapter saved; load it on top of the base model."

    return MergeResult(
        operation="combine",
        output_path=str(out),
        method=method,
        base_model=base_model,
        adapters=list(adapter_paths),
        weights=list(weights),
        bytes_written=_dir_size(out),
        notes=notes,
    )


def merge(
    *,
    operation: str,
    base_model: str,
    output_path: str,
    adapter_path: Optional[str] = None,
    adapter_paths: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
    method: str = "dare_ties",
    bake_after_merge: bool = False,
    trust_remote_code: bool = False,
    svd_rank: Optional[int] = None,
) -> MergeResult:
    """Dispatch entry point — picks the right merger for ``operation``."""
    if operation == "bake":
        if not adapter_path:
            raise ValueError("--mode bake requires --adapter")
        return bake_adapter(
            base_model=base_model,
            adapter_path=adapter_path,
            output_path=output_path,
            trust_remote_code=trust_remote_code,
        )
    if operation == "combine":
        if not adapter_paths or len(adapter_paths) < 2:
            raise ValueError("--mode combine requires at least two --adapters")
        return combine_adapters(
            base_model=base_model,
            adapter_paths=adapter_paths,
            weights=weights,
            method=method,
            output_path=output_path,
            bake_after_merge=bake_after_merge,
            trust_remote_code=trust_remote_code,
            svd_rank=svd_rank,
        )
    raise ValueError(
        f"Unknown merge operation {operation!r}. Choose from: bake, combine"
    )


__all__ = [
    "MergeResult",
    "merge",
    "bake_adapter",
    "combine_adapters",
    "list_supported_methods",
]
