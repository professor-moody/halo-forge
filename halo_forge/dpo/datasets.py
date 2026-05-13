"""Preference dataset loading for DPO.

DPO consumes pairs: each row carries a `prompt`, a `chosen` continuation
(human-preferred), and a `rejected` continuation. TRL's `DPOTrainer`
expects exactly those three columns.

This module provides:
    - load_preference_dataset(path_or_id) — autodetects local JSONL vs HF id,
      normalizes columns, splits train/val.
    - A small registry of canonical preference datasets we tested against
      so the CLI / public-API can offer a short-name shortcut.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class PreferenceDatasetSpec:
    """A known preference dataset we ship a short-name alias for."""

    name: str
    huggingface_id: str
    description: str
    size_hint: str
    prompt_column: str = "prompt"
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    default_split: str = "train"


PREFERENCE_DATASETS: dict[str, PreferenceDatasetSpec] = {
    "ultrafeedback": PreferenceDatasetSpec(
        name="ultrafeedback",
        huggingface_id="HuggingFaceH4/ultrafeedback_binarized",
        description="64K binarized GPT-4-judged preference pairs",
        size_hint="64K",
        default_split="train_prefs",
    ),
    "orca_dpo": PreferenceDatasetSpec(
        name="orca_dpo",
        huggingface_id="Intel/orca_dpo_pairs",
        description="13K Orca-style preference pairs",
        size_hint="13K",
    ),
    "hh_rlhf": PreferenceDatasetSpec(
        name="hh_rlhf",
        huggingface_id="Anthropic/hh-rlhf",
        description="Anthropic helpful+harmless preference pairs",
        size_hint="161K",
    ),
    "py_dpo": PreferenceDatasetSpec(
        name="py_dpo",
        huggingface_id="jondurbin/py-dpo-v0.1",
        description="Python-coding preference pairs",
        size_hint="18K",
    ),
}


def list_preference_datasets() -> list[PreferenceDatasetSpec]:
    """Return the registry as an ordered list (CLI + public-API consumer)."""
    return list(PREFERENCE_DATASETS.values())


def _normalize_messages(value, *, field_name: str) -> str:
    """Collapse a chat-style messages list into a single string.

    UltraFeedback and similar datasets ship `chosen` / `rejected` as a
    list of `{"role": ..., "content": ...}` dicts. TRL's DPOTrainer
    accepts a string column; we collapse here rather than push the
    canonicalization into the trainer so all our datasets look the same.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        chunks = []
        for msg in value:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if content is None:
                    raise ValueError(f"{field_name} message content cannot be null")
                if not isinstance(content, str):
                    raise ValueError(f"{field_name} message content must be a string")
                if role:
                    chunks.append(f"{role}: {content}")
                else:
                    chunks.append(content)
            else:
                if msg is None:
                    raise ValueError(f"{field_name} message cannot be null")
                if not isinstance(msg, str):
                    raise ValueError(f"{field_name} message must be a string")
                chunks.append(msg)
        return "\n".join(chunks).strip()
    if value is None:
        raise ValueError(f"{field_name} cannot be null")
    raise ValueError(f"{field_name} must be a string or chat-message list")


def _normalize_row(row: dict, spec: Optional[PreferenceDatasetSpec]) -> dict:
    prompt_col = spec.prompt_column if spec else "prompt"
    chosen_col = spec.chosen_column if spec else "chosen"
    rejected_col = spec.rejected_column if spec else "rejected"
    prompt = row.get(prompt_col)
    if prompt is None and "question" in row:
        prompt = row["question"]
    if prompt is None and "instruction" in row:
        prompt = row["instruction"]
    return {
        "prompt": _normalize_messages(prompt, field_name="prompt"),
        "chosen": _normalize_messages(row.get(chosen_col), field_name="chosen"),
        "rejected": _normalize_messages(row.get(rejected_col), field_name="rejected"),
    }


def load_preference_dataset(
    *,
    train_file: Optional[str] = None,
    dataset: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    validation_split: float = 0.05,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Load a preference dataset and return (train, validation) splits.

    Exactly one of `train_file` or `dataset` must be provided. Local files
    are JSONL with `prompt` / `chosen` / `rejected` keys (extra keys ignored).
    Dataset names resolve through `PREFERENCE_DATASETS` first; otherwise
    the string is forwarded to `datasets.load_dataset` directly.
    """
    if not train_file and not dataset:
        raise ValueError("load_preference_dataset requires train_file or dataset")
    if train_file and dataset:
        raise ValueError("Pass either train_file or dataset, not both")

    spec: Optional[PreferenceDatasetSpec] = None
    if dataset:
        spec = PREFERENCE_DATASETS.get(dataset)
        hf_id = spec.huggingface_id if spec else dataset
        hf_split = spec.default_split if spec and split == "train" else split
        logger.info("Loading preference dataset %s (split=%s)", hf_id, hf_split)
        ds = load_dataset(hf_id, split=hf_split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        def _normalize_or_mark(row):
            try:
                normalized = _normalize_row(row, spec)
            except ValueError as exc:
                return {"prompt": "", "chosen": "", "rejected": "", "_halo_invalid": str(exc)}
            normalized["_halo_invalid"] = ""
            return normalized

        ds = ds.map(
            _normalize_or_mark,
            remove_columns=[
                c for c in ds.column_names if c not in ("prompt", "chosen", "rejected")
            ],
            desc="Normalizing preference rows",
        )
    else:
        # Local JSONL path. Normalize in pure-Python *before* building the
        # Dataset — `Dataset.from_list` uses pyarrow which can't infer a
        # single schema across mixed-type rows (e.g. chosen as string in
        # one row, list-of-dicts in the next).
        path = Path(train_file)  # type: ignore[arg-type]
        if not path.exists():
            raise FileNotFoundError(f"Preference data file not found: {path}")
        logger.info("Loading preference dataset from local JSONL %s", path)
        normalized_rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                try:
                    normalized_rows.append(_normalize_row(row, spec))
                except ValueError as exc:
                    normalized_rows.append({
                        "prompt": "",
                        "chosen": "",
                        "rejected": "",
                        "_halo_invalid": str(exc),
                    })
        if max_samples:
            normalized_rows = normalized_rows[:max_samples]
        ds = Dataset.from_list(normalized_rows)

    before_filter = len(ds)
    if before_filter == 0:
        raise ValueError("Preference dataset has no rows")
    invalid_count = 0
    if "_halo_invalid" in ds.column_names:
        invalid_count = sum(1 for r in ds if r.get("_halo_invalid"))

    # Drop empty rows defensively. Some upstream datasets sneak in blanks.
    ds = ds.filter(lambda r: bool(r["prompt"]) and bool(r["chosen"]) and bool(r["rejected"]))
    empty_count = before_filter - invalid_count - len(ds)
    if "_halo_invalid" in ds.column_names:
        ds = ds.remove_columns(["_halo_invalid"])
    if len(ds) == 0:
        raise ValueError(
            "Preference dataset has no valid rows after validation; invalid rows were rejected "
            f"(invalid={invalid_count}, empty={max(0, empty_count)}, total={before_filter})"
        )
    if invalid_count:
        raise ValueError(
            "Preference dataset contains invalid rows "
            f"(invalid={invalid_count}, empty={max(0, empty_count)}, valid={len(ds)})"
        )

    if validation_split <= 0 or len(ds) < 10:
        return ds, ds.select(range(min(2, len(ds))))

    split_ds = ds.train_test_split(test_size=validation_split, seed=seed)
    return split_ds["train"], split_ds["test"]


__all__ = [
    "PreferenceDatasetSpec",
    "PREFERENCE_DATASETS",
    "list_preference_datasets",
    "load_preference_dataset",
]
