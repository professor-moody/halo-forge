"""One definition of what `chat_template_hash` means.

Before this module the field had three separate problems that only looked like
one, because the metadata contract was never written down:

**Two producers that disagreed.** ``data_lab/training_artifacts.py`` hashed with
``content_hash``, which JSON-encodes first, so it saw ``"{% for ...%}"`` *with*
surrounding quotes and escapes. ``training_plan/service.py`` hashed the raw
string. The same template therefore produced two different digests depending on
which path recorded it.

**A producer that could not see the template.** ``training_plan`` read
``tokenizer_config.json`` directly. Conversion relocates the template out of that
file into a standalone ``chat_template.jinja`` — measured across MLX, HF recast
and GGUF — so every converted artifact recorded ``None`` while the template sat
on disk beside it.

**No consumer.** Nothing in the codebase ever compared two values of this field.
That is why neither producer defect had a symptom: an integrity field that is
written, inherited and displayed but never checked cannot be observed to be
wrong.

Fixing any one of those alone would have been unsafe. Adding the obvious
comparison to the old field would have fired on the producer disagreement, and
would have silently *passed* for converted artifacts by comparing ``None`` to
``None``.

## The contract

- **Identity is derived from the loaded tokenizer**, never from raw storage.
  ``transformers`` resolves both the in-JSON and the sidecar layouts; a file read
  resolves only one, and which one depends on whether the artifact has been
  through a converter.
- **Absence is a state, not a null.** ``present`` / ``absent`` / ``unreadable`` /
  ``unsupported`` are distinguished, so "we looked and there is no template"
  never compares equal to "we could not look".
- **The scheme is versioned.** Changing how the digest is computed without
  changing ``SCHEME`` would silently make new rows incomparable with old ones
  while both look like hex strings.
- **Comparison is a three-valued outcome.** Only two ``present`` identities with
  the same scheme can produce ``match`` or ``mismatch``; everything else is
  ``indeterminate`` and must not be read as a pass.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

# Bump when the digest computation changes in any way that alters output for an
# unchanged template. Old rows keep their recorded scheme and compare
# `indeterminate` against new ones rather than falsely mismatching.
LEGACY_SCHEME = "legacy/unknown"
"""Marks a digest written before the contract existed.

Two such values are **never** comparable, even to each other: they may have
come from either of the two disagreeing producers, so equal strings do not
imply equal templates and unequal strings do not imply different ones.
"""

SCHEME = "cth/1"
"""Definition settled 2026-08-17, before any value was persisted.

Covers the *whole* named-template set in canonical JSON, not the
runtime-selected template and not `str(mapping)`. See
`canonical_template_set` for why both alternatives were rejected.
"""

DEFAULT_TEMPLATE_NAME = "default"


class ChatTemplateState(str, Enum):
    """Why a digest is or is not available. Never collapse these to a null."""

    PRESENT = "present"
    """A template was found and hashed."""

    ABSENT = "absent"
    """The tokenizer loaded and genuinely carries no chat template."""

    UNREADABLE = "unreadable"
    """Something prevented us from looking. Not evidence of absence."""

    UNSUPPORTED = "unsupported"
    """The artifact kind has no tokenizer to interrogate (e.g. a raw adapter)."""


class ComparisonOutcome(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class ChatTemplateIdentity:
    """A chat template's identity, or an explicit account of why there isn't one."""

    state: ChatTemplateState
    scheme: str = SCHEME
    digest: Optional[str] = None
    detail: Optional[str] = None

    def __post_init__(self) -> None:
        if self.state is ChatTemplateState.PRESENT and not self.digest:
            raise ValueError("PRESENT identity requires a digest")
        if self.state is not ChatTemplateState.PRESENT and self.digest:
            raise ValueError(f"{self.state.value} identity must not carry a digest")

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "scheme": self.scheme,
            "digest": self.digest,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, value: Optional[Mapping[str, Any]]) -> "ChatTemplateIdentity":
        if not value:
            return cls(ChatTemplateState.UNREADABLE, detail="no identity recorded")
        try:
            state = ChatTemplateState(str(value.get("state")))
        except ValueError:
            return cls(ChatTemplateState.UNREADABLE, detail="unrecognised state")
        # A missing scheme must NOT default to the current one. A record
        # without a scheme was written by something that did not know about
        # schemes, which is precisely the legacy case; defaulting to `SCHEME`
        # would assert comparability that was never established.
        scheme = value.get("scheme")
        return cls(
            state=state,
            scheme=str(scheme) if scheme else LEGACY_SCHEME,
            digest=value.get("digest"),
            detail=value.get("detail"),
        )

    @classmethod
    def from_legacy_hash(cls, value: Optional[str]) -> "ChatTemplateIdentity":
        """Adapt a pre-contract bare hash.

        Recorded under an unknown scheme by definition — the two old producers
        disagreed — so it is deliberately *not* given `SCHEME`. It will compare
        `indeterminate` against anything current rather than falsely matching or
        falsely mismatching.
        """
        if not value:
            return cls(ChatTemplateState.UNREADABLE, detail="legacy null hash")
        return cls(ChatTemplateState.PRESENT, scheme=LEGACY_SCHEME, digest=value)


def canonical_template_set(template: Any) -> dict[str, str]:
    """Normalise any template form into an order-independent name -> text map.

    `transformers` exposes `chat_template` as **either** a string or a dict of
    named templates (`default`, `tool_use`, ...), and a round-trip through
    `save_pretrained` preserves the dict form. `str(mapping)` is therefore
    unusable as a digest input: it is order-sensitive, so two artifacts with
    identical templates inserted in different orders hash differently.

    Two decisions are frozen here, because leaving them implicit is what made
    the old field meaningless:

    **Identity covers the whole set, not the runtime-selected template.** A
    change to `tool_use` is a change to behaviour-bearing content even when
    `default` is untouched, and a scheme that hashed only the selected template
    would call that artifact unchanged.

    **A bare string is identical to a single-entry set under `default`.** The
    two forms produce the same runtime behaviour, and treating them as different
    identities would fire on a pure serialisation change.
    """
    if isinstance(template, str):
        return {DEFAULT_TEMPLATE_NAME: template}
    if isinstance(template, Mapping):
        return {str(k): str(v) for k, v in template.items()}
    return {DEFAULT_TEMPLATE_NAME: str(template)}


def digest_template(template: Any) -> str:
    """The one hashing rule.

    Canonical JSON of the name -> text map, sorted, scheme-prefixed. Sorting is
    what makes it order-independent; the scheme prefix is what stops a future
    change to this function from silently producing comparable-looking digests.
    """
    canonical = json.dumps(
        canonical_template_set(template), sort_keys=True, ensure_ascii=False
    )
    payload = f"{SCHEME}\n{canonical}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def identify_from_tokenizer(tokenizer: Any) -> ChatTemplateIdentity:
    """Derive identity from a *loaded* tokenizer.

    Uses the object attribute rather than any file, so it is unaffected by
    conversion relocating the template between `tokenizer_config.json` and
    `chat_template.jinja`.
    """
    if tokenizer is None:
        return ChatTemplateIdentity(
            ChatTemplateState.UNSUPPORTED, detail="no tokenizer available"
        )
    try:
        template = getattr(tokenizer, "chat_template", None)
    except Exception as exc:  # defensive: exotic tokenizers may raise on access
        return ChatTemplateIdentity(
            ChatTemplateState.UNREADABLE, detail=f"{type(exc).__name__} reading attribute"
        )
    if template is None:
        return ChatTemplateIdentity(ChatTemplateState.ABSENT)
    # `str(template)` here would defeat `canonical_template_set` entirely: a
    # mapping would be stringified with its insertion order baked in and then
    # wrapped as a single `default` entry, so two artifacts with identical named
    # templates in different orders would compare `mismatch`. Pass the mapping.
    #
    # `is None` rather than falsiness, so an empty template is PRESENT with a
    # digest and remains distinguishable from a tokenizer that has none.
    return ChatTemplateIdentity(
        ChatTemplateState.PRESENT, digest=digest_template(template)
    )


def identify_from_path(path: str | Path, *, trust_remote_code: bool = False) -> ChatTemplateIdentity:
    """Load the tokenizer at `path` and derive identity from it.

    A load failure is `unreadable`, never `absent` — the distinction is the
    whole point of the state enum.
    """
    target = Path(path)
    if not target.exists():
        return ChatTemplateIdentity(ChatTemplateState.UNREADABLE, detail="path does not exist")
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        return ChatTemplateIdentity(
            ChatTemplateState.UNREADABLE, detail=f"transformers unavailable: {type(exc).__name__}"
        )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(target), trust_remote_code=trust_remote_code
        )
    except Exception as exc:
        # A model directory with no tokenizer at all is `unsupported`; anything
        # else that goes wrong while looking is `unreadable`.
        if not _looks_like_tokenizer_dir(target):
            return ChatTemplateIdentity(
                ChatTemplateState.UNSUPPORTED, detail="no tokenizer files present"
            )
        return ChatTemplateIdentity(
            ChatTemplateState.UNREADABLE, detail=f"{type(exc).__name__} loading tokenizer"
        )
    return identify_from_tokenizer(tokenizer)


def _looks_like_tokenizer_dir(path: Path) -> bool:
    return any(
        (path / name).exists()
        for name in ("tokenizer_config.json", "tokenizer.json", "tokenizer.model")
    )


def compare(
    recorded: ChatTemplateIdentity, observed: ChatTemplateIdentity
) -> tuple[ComparisonOutcome, str]:
    """Three-valued comparison. Only `MATCH` may be treated as verification.

    Critically, two non-`PRESENT` identities never match. "Both absent" and
    "both unreadable" are consistent, not verified — which is what makes the old
    `None == None` pass impossible to reproduce.
    """
    if recorded.state is not ChatTemplateState.PRESENT or observed.state is not ChatTemplateState.PRESENT:
        return (
            ComparisonOutcome.INDETERMINATE,
            f"recorded={recorded.state.value}, observed={observed.state.value}: "
            "at least one side has no digest to compare",
        )
    if LEGACY_SCHEME in (recorded.scheme, observed.scheme):
        # Never comparable, including legacy-to-legacy. The two pre-contract
        # producers disagreed, so two equal legacy strings may describe
        # different templates and two different ones may describe the same.
        return (
            ComparisonOutcome.INDETERMINATE,
            "at least one digest predates the identity contract and was written "
            "under an unknown scheme",
        )
    if recorded.scheme != observed.scheme:
        return (
            ComparisonOutcome.INDETERMINATE,
            f"scheme differs ({recorded.scheme} vs {observed.scheme}); "
            "digests are not comparable across schemes",
        )
    if recorded.digest == observed.digest:
        return ComparisonOutcome.MATCH, "digests identical"
    return (
        ComparisonOutcome.MISMATCH,
        f"digest differs under {recorded.scheme}: {recorded.digest} vs {observed.digest}",
    )


class DerivationMode(str, Enum):
    """How an identity came to be attached to an artifact.

    A digest alone cannot say whether it describes *this* artifact or was copied
    from the thing it was made from. Recording only the digest is what let the
    old field claim source provenance for transformed outputs.
    """

    DERIVED = "derived"
    """Computed from the artifact this identity is attached to."""

    INHERITED = "inherited"
    """Copied from an input because the output could not be interrogated.

    The GGUF path necessarily lands here: a `.gguf` container has no
    `transformers`-loadable tokenizer directory, so identity cannot be derived
    from the output even though the template is present inside the container.
    """

    UNSUPPORTED = "unsupported"
    """No identity is applicable to this artifact kind at all."""


@dataclass(frozen=True)
class ChatTemplateRecord:
    """The full persisted envelope: identity plus how it was obtained.

    `chat_template_hash` is retained downstream only as a **compatibility
    projection** of `identity.digest`. Anything making a trust decision must read
    this record, because the bare column cannot distinguish a current digest from
    a legacy one, nor a derived identity from an inherited one.
    """

    identity: ChatTemplateIdentity
    mode: DerivationMode
    source_occurrence_id: Optional[str] = None

    def __post_init__(self) -> None:
        # `derived` asserts the identity was computed from this artifact, which
        # is impossible when the artifact could not be read. Allowing the pair
        # to be constructed leaves `describes_this_artifact()` guarding a state
        # that should never have existed.
        if self.mode is DerivationMode.DERIVED and self.identity.state in (
            ChatTemplateState.UNREADABLE,
            ChatTemplateState.UNSUPPORTED,
        ):
            raise ValueError(
                f"mode=derived is incompatible with state={self.identity.state.value}: "
                "an identity cannot be derived from an artifact that could not be read"
            )
        if self.mode is DerivationMode.INHERITED and not self.source_occurrence_id:
            # Previously this assigned None over None -- a no-op that read like
            # an invariant. An inherited identity that cannot name where it came
            # from is untraceable, and claiming inheritance without provenance
            # is worse than admitting the identity is unsupported.
            raise ValueError(
                "mode=inherited requires source_occurrence_id; an identity that "
                "cannot name its source must be recorded as unsupported"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity.to_dict()
        payload["mode"] = self.mode.value
        if self.source_occurrence_id:
            payload["source_occurrence_id"] = self.source_occurrence_id
        return payload

    @classmethod
    def from_dict(cls, value: Optional[Mapping[str, Any]]) -> "ChatTemplateRecord":
        if not value:
            return cls(
                ChatTemplateIdentity(ChatTemplateState.UNREADABLE, detail="no record"),
                DerivationMode.UNSUPPORTED,
            )
        try:
            mode = DerivationMode(str(value.get("mode")))
        except ValueError:
            mode = DerivationMode.UNSUPPORTED
        return cls(
            identity=ChatTemplateIdentity.from_dict(value),
            mode=mode,
            source_occurrence_id=value.get("source_occurrence_id"),
        )

    @property
    def projected_hash(self) -> Optional[str]:
        """The legacy `chat_template_hash` column value."""
        return self.identity.digest

    def describes_this_artifact(self) -> bool:
        """True only when the identity was computed from the artifact itself.

        Callers enforcing an invariant must gate on this. An inherited identity
        may well be correct -- measured 2026-08-17, conversion preserves the
        template across MLX, HF recast and GGUF -- but it is an assumption about
        the converter, not an observation of the output.
        """
        return (
            self.mode is DerivationMode.DERIVED
            and self.identity.state is ChatTemplateState.PRESENT
            # A legacy-scheme digest is not comparable to anything, including
            # a freshly computed one, so it cannot support a claim that this
            # artifact's template was verified. Records reloaded without a
            # scheme land here, and previously returned true.
            and self.identity.scheme == SCHEME
        )


def record_for_derivation(identity: ChatTemplateIdentity) -> "ChatTemplateRecord":
    """Build a record whose mode is consistent with what was actually observed.

    A producer that computed identity from the artifact in front of it is
    `derived` -- but only when it managed to read something. When the read
    failed there is nothing to have derived, so the mode is `unsupported`.
    Asserting `derived` regardless is what the constructor now rejects.
    """
    if identity.state in (ChatTemplateState.PRESENT, ChatTemplateState.ABSENT):
        return ChatTemplateRecord(identity, DerivationMode.DERIVED)
    return ChatTemplateRecord(identity, DerivationMode.UNSUPPORTED)


def record_for_inheritance(
    identity: ChatTemplateIdentity, source_occurrence_id: Optional[str]
) -> "ChatTemplateRecord":
    """Inherit an identity, or decline to when the source cannot be named."""
    if identity.state is not ChatTemplateState.PRESENT or not source_occurrence_id:
        return ChatTemplateRecord(
            ChatTemplateIdentity(
                ChatTemplateState.UNSUPPORTED,
                detail="no usable, traceable source identity",
            ),
            DerivationMode.UNSUPPORTED,
        )
    return ChatTemplateRecord(
        identity, DerivationMode.INHERITED, source_occurrence_id=source_occurrence_id
    )
