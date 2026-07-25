"""Preference-pair normalization contract.

DPO / ORPO / RM all consume `load_preference_dataset`, and all three expect
`chosen` / `rejected` to hold the RESPONSE ONLY — never the prompt again, and
never hand-rolled ``role: text`` markers the tokenizer will not emit at
inference time. These tests pin that contract for the two shapes we actually
ship: the conversational (UltraFeedback) shape and the flat string shape.

No real tokenizer is downloaded; `_ChatMLTokenizer` stands in for one and
implements the only surface the loader touches (`chat_template` +
`apply_chat_template`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

datasets = pytest.importorskip("datasets", reason="datasets is required for preference loading")


class _ChatMLTokenizer:
    """Minimal stand-in for a HF tokenizer that carries a ChatML template."""

    chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}"
        "<|im_end|>\n{% endfor %}"
    )

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **_: Any,
    ) -> str:
        assert tokenize is False, "loader must render text, not token ids"
        rendered = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


class _TemplatelessTokenizer:
    """A tokenizer without a chat template (base models ship like this)."""

    chat_template: Optional[str] = None


def _write_jsonl(tmp_path: Path, rows: list[dict]) -> str:
    path = tmp_path / "pairs.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return str(path)


def _ultrafeedback_row() -> dict:
    """The shape `HuggingFaceH4/ultrafeedback_binarized` actually ships."""
    return {
        "prompt": "What is 2+2?",
        "chosen": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "rejected": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "5"},
        ],
    }


def _load_one(tmp_path: Path, row: dict, tokenizer: Any = None) -> dict:
    from halo_forge.dpo.datasets import load_preference_dataset

    train, _ = load_preference_dataset(
        train_file=_write_jsonl(tmp_path, [row]),
        validation_split=0.0,
        tokenizer=tokenizer,
    )
    assert set(train.column_names) == {"prompt", "chosen", "rejected"}
    return train[0]


def test_conversational_row_does_not_duplicate_prompt(tmp_path: Path):
    """UltraFeedback rows must normalize to response-only completions."""
    normalized = _load_one(tmp_path, _ultrafeedback_row())

    assert normalized["prompt"] == "What is 2+2?"
    assert normalized["chosen"] == "4"
    assert normalized["rejected"] == "5"
    for field in ("chosen", "rejected"):
        assert "What is 2+2?" not in normalized[field]
        assert "user:" not in normalized[field]
        assert "assistant:" not in normalized[field]


def test_conversational_row_uses_chat_template_when_available(tmp_path: Path):
    """With a chat template the prompt is templated and the response is the
    templated remainder — no hand-rolled ``role: text`` prefixes anywhere."""
    normalized = _load_one(tmp_path, _ultrafeedback_row(), tokenizer=_ChatMLTokenizer())

    assert normalized["prompt"] == (
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    )
    assert normalized["chosen"] == "4<|im_end|>\n"
    assert normalized["rejected"] == "5<|im_end|>\n"
    for field in ("chosen", "rejected"):
        assert "What is 2+2?" not in normalized[field]
        assert "user:" not in normalized[field]
        assert "assistant:" not in normalized[field]


def test_multi_turn_conversation_keeps_context_out_of_the_response(tmp_path: Path):
    """Only the trailing assistant turn is the completion; earlier turns are
    prompt context, and are recovered as the prompt when none was supplied."""
    row = {
        "chosen": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "rejected": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "banana"},
        ],
    }
    normalized = _load_one(tmp_path, row, tokenizer=_ChatMLTokenizer())

    assert normalized["chosen"] == "4<|im_end|>\n"
    assert normalized["rejected"] == "banana<|im_end|>\n"
    assert "Hello!" in normalized["prompt"]
    assert "What is 2+2?" in normalized["prompt"]
    assert normalized["prompt"].endswith("<|im_start|>assistant\n")
    for field in ("chosen", "rejected"):
        assert "Hello!" not in normalized[field]
        assert "What is 2+2?" not in normalized[field]


def test_plain_string_row_passes_through_verbatim(tmp_path: Path):
    """Flat string rows (py_dpo / orca_dpo shape) are already response-only
    text; we must not re-template or otherwise rewrite them."""
    row = {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}

    for tokenizer in (None, _ChatMLTokenizer(), _TemplatelessTokenizer()):
        normalized = _load_one(tmp_path, row, tokenizer=tokenizer)
        assert normalized == {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}


def test_templateless_tokenizer_falls_back_without_role_prefixes(tmp_path: Path):
    """Base models have no chat template. The fallback is plain text — the
    prompt still must not leak into the completion."""
    normalized = _load_one(
        tmp_path, _ultrafeedback_row(), tokenizer=_TemplatelessTokenizer()
    )

    assert normalized["prompt"] == "What is 2+2?"
    assert normalized["chosen"] == "4"
    assert normalized["rejected"] == "5"
    assert "user:" not in normalized["chosen"]


def test_prompt_recovered_from_context_without_a_tokenizer(tmp_path: Path):
    """DPO / ORPO / MLX pass no tokenizer today, so the template-free path is
    the live one. A row whose prompt lives only inside the completion turns
    must still produce a non-empty prompt — an empty one is silently dropped
    by the empty-row filter, losing the sample without any error."""
    row = {
        "chosen": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "rejected": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "banana"},
        ],
    }
    normalized = _load_one(tmp_path, row, tokenizer=None)

    assert normalized["prompt"] != ""
    assert "What is 2+2?" in normalized["prompt"]
    assert "Hello!" in normalized["prompt"]
    assert normalized["chosen"] == "4"
    assert normalized["rejected"] == "banana"


def test_roleless_message_list_does_not_crash_a_strict_chat_template(tmp_path: Path):
    """Bare-string message lists coerce to empty roles. Real chat templates
    raise a jinja error on those, and that error is not a ValueError, so it
    escapes the per-row quarantine and aborts the entire load."""

    class _StrictTokenizer:
        chat_template = "strict"

        def apply_chat_template(self, messages, *, tokenize=True, **_: Any) -> str:
            for message in messages:
                if message["role"] not in ("system", "user", "assistant"):
                    raise Exception(f"unknown role {message['role']!r}")
            return "".join(f"<{m['role']}>{m['content']}" for m in messages)

    row = {"prompt": "hi", "chosen": ["ctx", "good"], "rejected": ["ctx", "bad"]}
    normalized = _load_one(tmp_path, row, tokenizer=_StrictTokenizer())

    assert normalized["prompt"] == "hi"
    assert normalized["chosen"] == "good"
    assert normalized["rejected"] == "bad"


def test_null_message_content_is_still_rejected(tmp_path: Path):
    """Validation behaviour from the old collapse path must survive the fix."""
    from halo_forge.dpo.datasets import load_preference_dataset

    path = _write_jsonl(
        tmp_path,
        [{"prompt": "p", "chosen": [{"role": "assistant", "content": None}], "rejected": "x"}],
    )
    with pytest.raises(ValueError, match="invalid rows"):
        load_preference_dataset(train_file=path, validation_split=0.0)


def test_reward_pair_text_separates_prompt_from_response():
    """The RM trainer glued prompt+response with no separator at all."""
    from halo_forge.rm.trainer import _pair_text

    # Untemplated prompt: needs an explicit separator.
    joined = _pair_text("What is 2+2?", "4")
    assert joined != "What is 2+2?4"
    assert joined == "What is 2+2?\n4"

    # Templated prompt already ends in the assistant generation prompt.
    templated = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    assert _pair_text(templated, "4<|im_end|>\n") == templated + "4<|im_end|>\n"

    # Degenerate inputs stay safe.
    assert _pair_text("", "4") == "4"
    assert _pair_text(None, None) == ""
