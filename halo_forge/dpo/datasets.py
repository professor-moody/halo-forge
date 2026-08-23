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
from typing import Any, Optional, Tuple

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


def _coerce_messages(value: Any, *, field_name: str) -> Optional[list[dict[str, str]]]:
    """Return `value` as a list of `{"role", "content"}` dicts, or None for plain text.

    UltraFeedback and friends ship `chosen` / `rejected` as chat-message
    lists; py-dpo and orca ship flat strings. Malformed shapes raise so the
    caller can quarantine the row instead of silently training on garbage.
    """
    if isinstance(value, str):
        return None
    if value is None:
        raise ValueError(f"{field_name} cannot be null")
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a string or chat-message list")
    messages: list[dict[str, str]] = []
    for msg in value:
        if isinstance(msg, dict):
            content = msg.get("content", "")
            if content is None:
                raise ValueError(f"{field_name} message content cannot be null")
            if not isinstance(content, str):
                raise ValueError(f"{field_name} message content must be a string")
            role = msg.get("role") or ""
            if not isinstance(role, str):
                raise ValueError(f"{field_name} message role must be a string")
            messages.append({"role": role, "content": content})
        elif isinstance(msg, str):
            messages.append({"role": "", "content": msg})
        elif msg is None:
            raise ValueError(f"{field_name} message cannot be null")
        else:
            raise ValueError(f"{field_name} message must be a string")
    return messages


def _split_completion(
    messages: list[dict[str, str]],
) -> Tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split a conversational completion into (context turns, response turns).

    UltraFeedback repeats the prompt inside `chosen` / `rejected`:
    `[{"role": "user", ...}, {"role": "assistant", ...}]`. Only the trailing
    assistant turn(s) are the completion DPO scores — everything before them
    is prompt context and must not be duplicated into the response.
    """
    if not messages:
        return [], []
    roles = [m["role"].lower() for m in messages]
    if "assistant" in roles:
        cut = len(messages)
        while cut > 0 and roles[cut - 1] == "assistant":
            cut -= 1
        return messages[:cut], messages[cut:]
    # Unlabelled turns (bare-string message lists): treat the last as the
    # response and anything before it as context.
    return messages[:-1], messages[-1:]


def _has_chat_template(tokenizer: Any) -> bool:
    """True when `tokenizer` can render messages with a real chat template."""
    if tokenizer is None:
        return False
    if not getattr(tokenizer, "chat_template", None):
        return False
    return callable(getattr(tokenizer, "apply_chat_template", None))


def _all_roled(messages: list[dict[str, str]]) -> bool:
    """True when every message carries a role a chat template can render.

    Bare-string message lists coerce to `role == ""`; real chat templates
    raise a jinja error on those, and that error is not a `ValueError`, so it
    would escape the per-row quarantine and abort the whole load.
    """
    return all(bool(m["role"]) for m in messages)


def _join_contents(messages: list[dict[str, str]]) -> str:
    """Template-free rendering: message contents only, no invented role markers.

    Used when no tokenizer (or no chat template) is available. It is lossy for
    multi-turn prompts — speaker boundaries are not marked — but it never
    teaches the model a `role: text` format the tokenizer will not emit at
    inference time. Pass a tokenizer to get proper templating.
    """
    return "\n\n".join(m["content"] for m in messages if m["content"]).strip()


def _normalize_row(
    row: dict,
    spec: Optional[PreferenceDatasetSpec],
    tokenizer: Any = None,
) -> dict:
    """Normalize one preference row into prompt / chosen / rejected strings.

    `chosen` and `rejected` come out as the RESPONSE ONLY: for conversational
    rows the prompt-restating leading turns are dropped, and the remainder is
    rendered with `tokenizer`'s chat template when it has one (same prefix
    slicing TRL uses, so the strings match what the model is asked to produce
    at inference). Flat string rows pass through verbatim — they carry no role
    structure to template, and re-wrapping them would corrupt datasets that
    already ship pre-formatted text.
    """
    prompt_col = spec.prompt_column if spec else "prompt"
    chosen_col = spec.chosen_column if spec else "chosen"
    rejected_col = spec.rejected_column if spec else "rejected"
    prompt = row.get(prompt_col)
    if prompt is None and "question" in row:
        prompt = row["question"]
    if prompt is None and "instruction" in row:
        prompt = row["instruction"]

    chosen_value = row.get(chosen_col)
    rejected_value = row.get(rejected_col)
    chosen_messages = _coerce_messages(chosen_value, field_name="chosen")
    rejected_messages = _coerce_messages(rejected_value, field_name="rejected")
    conversational = chosen_messages is not None or rejected_messages is not None

    if not conversational and not isinstance(prompt, list):
        # Flat string row: already response-only text, nothing to split.
        if prompt is None:
            raise ValueError("prompt cannot be null")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string or chat-message list")
        return {
            "prompt": prompt.strip(),
            "chosen": chosen_value.strip(),  # type: ignore[union-attr]
            "rejected": rejected_value.strip(),  # type: ignore[union-attr]
        }

    chosen_context, chosen_response = _split_completion(chosen_messages or [])
    rejected_context, rejected_response = _split_completion(rejected_messages or [])
    context_messages = chosen_context or rejected_context

    prompt_messages = _coerce_messages(prompt, field_name="prompt") if prompt is not None else None
    if prompt_messages is not None:
        prompt_plain = _join_contents(prompt_messages)
    else:
        prompt_plain = prompt.strip() if isinstance(prompt, str) else ""
        if prompt is None and not context_messages:
            raise ValueError("prompt cannot be null")
        # The completion's own leading turns are the authoritative conversation
        # (they carry roles and any earlier turns); fall back to wrapping a
        # bare string prompt as a single user turn.
        prompt_messages = context_messages or [{"role": "user", "content": prompt_plain}]
        if not prompt_plain:
            # No prompt column at all: the completion's own context turns are
            # the prompt. Without this the template-free path emits an empty
            # prompt and the row is silently dropped by the empty-row filter.
            prompt_plain = _join_contents(prompt_messages)

    use_template = (
        _has_chat_template(tokenizer) and bool(prompt_messages) and _all_roled(prompt_messages)
    )
    if use_template:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = prompt_plain

    def _completion_text(messages, response, raw) -> str:
        if messages is None:
            return raw.strip()
        if not (use_template and response and _all_roled(response)):
            return _join_contents(response)
        rendered = tokenizer.apply_chat_template(
            list(prompt_messages) + list(response), tokenize=False
        )
        if rendered.startswith(prompt_text):
            return rendered[len(prompt_text):]
        # Template did not emit the prompt as a literal prefix (some templates
        # reorder or inject a system turn). Prefer a bare response over risking
        # the prompt leaking back into the completion.
        return _join_contents(response)

    return {
        "prompt": prompt_text,
        "chosen": _completion_text(chosen_messages, chosen_response, chosen_value),
        "rejected": _completion_text(rejected_messages, rejected_response, rejected_value),
    }


def load_preference_dataset(
    *,
    train_file: Optional[str] = None,
    validation_file: Optional[str] = None,
    dataset: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    validation_split: float = 0.05,
    seed: int = 42,
    tokenizer: Any = None,
) -> Tuple[Dataset, Dataset]:
    """Load a preference dataset and return (train, validation) splits.

    Exactly one of `train_file` or `dataset` must be provided. Local files
    are JSONL with `prompt` / `chosen` / `rejected` keys (extra keys ignored).
    Dataset names resolve through `PREFERENCE_DATASETS` first; otherwise
    the string is forwarded to `datasets.load_dataset` directly.

    `chosen` / `rejected` are always normalized to the response only. Pass
    `tokenizer` (the model's own) so conversational rows are rendered with its
    chat template; without one they fall back to plain message text (see
    `_join_contents`).
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
                normalized = _normalize_row(row, spec, tokenizer)
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
                    normalized_rows.append(_normalize_row(row, spec, tokenizer))
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

    validation_ds = None
    if validation_file:
        validation_path = Path(validation_file)
        if not validation_path.exists():
            raise FileNotFoundError(
                f"Preference validation file not found: {validation_path}"
            )
        validation_rows = []
        with validation_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Preference validation row {line_number} must be an object"
                    )
                validation_rows.append(_normalize_row(row, None, tokenizer))
        if not validation_rows:
            raise ValueError("Preference validation file has no rows")
        validation_ds = Dataset.from_list(validation_rows)

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

    if validation_ds is not None:
        return ds, validation_ds
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
