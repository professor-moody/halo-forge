"""Chat-template identity must survive conversion and must never verify on nulls.

Exercises `halo_forge.chat_template_identity` **directly**. An earlier version of
this file re-implemented the production expression inline, which meant production
could regress while these tests stayed green — the defect the tests exist to
prevent, reproduced in the tests themselves.

Background: the field had two disagreeing producers (`content_hash`, which
JSON-encodes before hashing, versus a raw `sha256`) and no consumer at all.
Neither producer defect had a symptom precisely because nothing ever compared
two values. Adding the obvious comparison first would have fired on the producer
disagreement and *passed* for converted artifacts by comparing `None` to `None`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from halo_forge.chat_template_identity import (
    SCHEME,
    ChatTemplateIdentity,
    ChatTemplateState,
    ComparisonOutcome,
    compare,
    digest_template,
    identify_from_path,
    identify_from_tokenizer,
)

TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
)


class _Tokenizer:
    """Minimal stand-in for a loaded tokenizer, resolving either storage layout.

    `transformers` performs this resolution itself; reproducing it here keeps the
    layout tests free of a model download. The layouts are captured from real
    MLX and `save_pretrained` output, not invented.
    """

    def __init__(self, path: Path) -> None:
        cfg = json.loads((path / "tokenizer_config.json").read_text())
        template = cfg.get("chat_template")
        if template is None:
            sidecar = path / "chat_template.jinja"
            template = sidecar.read_text() if sidecar.exists() else None
        self.chat_template = template


def _source_layout(root: Path) -> Path:
    """Pre-conversion: template inside tokenizer_config.json."""
    d = root / "source"
    d.mkdir(parents=True)
    (d / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "chat_template": TEMPLATE})
    )
    return d


def _converted_layout(root: Path) -> Path:
    """Post-conversion: template relocated to chat_template.jinja, key dropped."""
    d = root / "converted"
    d.mkdir(parents=True)
    (d / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))
    (d / "chat_template.jinja").write_text(TEMPLATE)
    return d


# --- the layout move ------------------------------------------------------

def test_identity_is_stable_across_the_conversion_layout_move(tmp_path: Path) -> None:
    src = identify_from_tokenizer(_Tokenizer(_source_layout(tmp_path)))
    conv = identify_from_tokenizer(_Tokenizer(_converted_layout(tmp_path)))

    assert src.state is ChatTemplateState.PRESENT
    assert conv.state is ChatTemplateState.PRESENT
    outcome, reason = compare(src, conv)
    assert outcome is ComparisonOutcome.MATCH, reason


def test_converted_fixture_really_lacks_the_json_key(tmp_path: Path) -> None:
    """Guards the premise so the test above cannot pass vacuously."""
    cfg = json.loads((_converted_layout(tmp_path) / "tokenizer_config.json").read_text())
    assert "chat_template" not in cfg, (
        "Fixture drifted: the converted layout now keeps the template in "
        "tokenizer_config.json, so the layout-move test no longer covers the "
        "relocation. Re-capture from a real conversion."
    )


def test_raw_json_read_would_miss_the_converted_template(tmp_path: Path) -> None:
    """The regression being pinned, kept executable rather than described."""
    conv = _converted_layout(tmp_path)
    raw = json.loads((conv / "tokenizer_config.json").read_text()).get("chat_template")
    assert raw is None
    assert identify_from_tokenizer(_Tokenizer(conv)).state is ChatTemplateState.PRESENT


# --- nulls must never verify ----------------------------------------------

@pytest.mark.parametrize(
    "state",
    [ChatTemplateState.ABSENT, ChatTemplateState.UNREADABLE, ChatTemplateState.UNSUPPORTED],
)
def test_two_non_present_identities_never_match(state: ChatTemplateState) -> None:
    """`None == None` used to read as agreement. It must now be indeterminate."""
    a = ChatTemplateIdentity(state)
    b = ChatTemplateIdentity(state)
    outcome, reason = compare(a, b)
    assert outcome is ComparisonOutcome.INDETERMINATE, reason


def test_present_against_absent_is_indeterminate_not_mismatch() -> None:
    present = ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE))
    absent = ChatTemplateIdentity(ChatTemplateState.ABSENT)
    outcome, _ = compare(present, absent)
    assert outcome is ComparisonOutcome.INDETERMINATE


def test_differing_digests_are_a_real_mismatch() -> None:
    a = ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE))
    b = ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE + " "))
    outcome, _ = compare(a, b)
    assert outcome is ComparisonOutcome.MISMATCH


# --- scheme versioning ----------------------------------------------------

def test_digest_is_scheme_bound() -> None:
    """Two schemes over the same text must not collide."""
    import hashlib

    naked = hashlib.sha256(TEMPLATE.encode()).hexdigest()
    assert digest_template(TEMPLATE) != naked


def test_cross_scheme_comparison_is_indeterminate_not_mismatch() -> None:
    """A scheme change must not look like tampering."""
    current = ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE))
    other = ChatTemplateIdentity(
        ChatTemplateState.PRESENT, scheme="cth/999", digest=digest_template(TEMPLATE)
    )
    outcome, reason = compare(current, other)
    assert outcome is ComparisonOutcome.INDETERMINATE, reason


def test_legacy_hashes_never_falsely_match_or_mismatch() -> None:
    """Pre-contract rows were written under one of two disagreeing producers."""
    legacy = ChatTemplateIdentity.from_legacy_hash("deadbeef")
    current = ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE))
    outcome, _ = compare(legacy, current)
    assert outcome is ComparisonOutcome.INDETERMINATE
    assert legacy.scheme != SCHEME


# --- state discipline -----------------------------------------------------

def test_missing_path_is_unreadable_not_absent(tmp_path: Path) -> None:
    identity = identify_from_path(tmp_path / "does-not-exist")
    assert identity.state is ChatTemplateState.UNREADABLE


def test_directory_without_tokenizer_is_unsupported(tmp_path: Path) -> None:
    d = tmp_path / "weights-only"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"\x00")
    assert identify_from_path(d).state is ChatTemplateState.UNSUPPORTED


def test_present_requires_a_digest_and_others_forbid_one() -> None:
    with pytest.raises(ValueError):
        ChatTemplateIdentity(ChatTemplateState.PRESENT)
    with pytest.raises(ValueError):
        ChatTemplateIdentity(ChatTemplateState.ABSENT, digest="x")


def test_round_trips_through_serialisation() -> None:
    original = ChatTemplateIdentity(ChatTemplateState.PRESENT, digest=digest_template(TEMPLATE))
    restored = ChatTemplateIdentity.from_dict(original.to_dict())
    assert compare(original, restored)[0] is ComparisonOutcome.MATCH
