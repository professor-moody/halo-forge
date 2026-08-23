"""Deterministic corpus packing for continued causal pretraining.

The packer treats paragraphs as the smallest preferred packing unit, appends
one EOS token per document, and never creates sliding-window overlap.  A
paragraph that fits in an empty block is never split; paragraphs longer than
the configured sequence length are partitioned into adjacent, non-overlapping
chunks.

A packing unit carries the whitespace that separates it from the next
paragraph, so concatenating the packed blocks of a document reproduces its
source text exactly modulo newline normalization and the injected EOS.  Tokens
that cannot be packed into a trainable block are reported in ``dropped_tokens``
and clear the ``exact`` flag rather than being lost silently.

The public ``CorpusPackingPlan`` type lives in ``halo_forge.own_data.models``
because it is also the transport contract used by guided own-data workflows.
This module supplies the executable implementation that produces that plan.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from halo_forge.own_data.models import CorpusPackingPlan

PACKING_ALGORITHM = "paragraph_eos_non_overlap_v1"
PACKING_SEPARATOR = "eos"
_PARAGRAPH_BREAK = re.compile(r"(?:\r?\n[ \t]*){2,}")
_PARAGRAPH_BREAK_CAPTURED = re.compile(r"((?:\r?\n[ \t]*){2,})")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        return _content_hash(str(path))
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def model_identity_hash(
    model: Optional[str],
    *,
    revision: Optional[str] = None,
    explicit_hash: Optional[str] = None,
) -> Optional[str]:
    """Return a stable model identity.

    Local files/directories are hashed by content.  Remote identifiers are
    hashed together with their pinned revision.  Callers that already resolved
    an immutable model artifact may pass its content hash directly.
    """

    if explicit_hash is not None:
        value = str(explicit_hash).strip()
        if not value:
            raise ValueError("model_hash cannot be empty")
        return value
    if not model:
        return None
    path = Path(str(model)).expanduser()
    if path.exists():
        return _hash_path(path.resolve())
    return _content_hash({"model": str(model), "revision": revision})


def tokenizer_identity_hash(
    tokenizer: Any,
    *,
    tokenizer_id: Optional[str] = None,
    revision: Optional[str] = None,
    explicit_hash: Optional[str] = None,
) -> str:
    """Hash the tokenizer state that can change corpus tokenization."""

    if explicit_hash is not None:
        value = str(explicit_hash).strip()
        if not value:
            raise ValueError("tokenizer_hash cannot be empty")
        return value
    if tokenizer is None:
        raise ValueError("an exact tokenizer is required for corpus packing")

    vocabulary: Any = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocabulary = sorted(
                (str(token), int(identifier)) for token, identifier in dict(get_vocab()).items()
            )
        except Exception:
            vocabulary = None
    added_tokens = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(added_tokens, Mapping):
        added_tokens = {
            str(key): str(value)
            for key, value in sorted(added_tokens.items(), key=lambda item: str(item[0]))
        }
    special_tokens = getattr(tokenizer, "special_tokens_map", None)
    if not isinstance(special_tokens, Mapping):
        special_tokens = {
            name: getattr(tokenizer, name, None)
            for name in (
                "bos_token",
                "eos_token",
                "pad_token",
                "unk_token",
                "sep_token",
                "cls_token",
                "mask_token",
            )
        }
    identity = {
        "tokenizer_id": tokenizer_id
        or getattr(tokenizer, "name_or_path", None)
        or tokenizer.__class__.__qualname__,
        "revision": revision,
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "vocabulary": vocabulary,
        "added_tokens": added_tokens,
        "special_tokens": dict(special_tokens),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "init_kwargs": getattr(tokenizer, "init_kwargs", None),
    }
    return _content_hash(identity)


def _normalize_newlines(text: str) -> str:
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def split_paragraphs(text: str) -> tuple[str, ...]:
    """Return non-empty, trimmed paragraphs for display and inspection.

    This view discards the whitespace that separates paragraphs, so it is not
    suitable for packing.  Use :func:`paragraph_units` when the concatenation
    has to reproduce the source text.
    """

    normalized = _normalize_newlines(text).strip()
    if not normalized:
        return ()
    return tuple(value.strip() for value in _PARAGRAPH_BREAK.split(normalized) if value.strip())


def paragraph_units(text: str) -> tuple[str, ...]:
    """Return packing units whose concatenation reproduces the source text.

    The paragraph separator stays attached to the paragraph that precedes it
    and intra-paragraph indentation is never trimmed, so packing a document and
    concatenating its units is a lossless round trip modulo newline
    normalization.  Continued pretraining on code depends on this: trimming
    would de-indent the corpus and destroy paragraph boundaries.
    """

    normalized = _normalize_newlines(text)
    if not normalized.strip():
        return ()
    pieces = _PARAGRAPH_BREAK_CAPTURED.split(normalized)
    units: list[str] = []
    for index in range(0, len(pieces), 2):
        separator = pieces[index + 1] if index + 1 < len(pieces) else ""
        unit = pieces[index] + separator
        if unit:
            units.append(unit)
    return tuple(units)


def _encode(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            values = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            values = tokenizer.encode(text)
    else:
        values = tokenizer(text, add_special_tokens=False)
        if isinstance(values, Mapping):
            values = values.get("input_ids")
    if values is None:
        raise ValueError("tokenizer did not return input_ids")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("corpus tokenizer returned an unexpected batch")
        values = values[0]
    return [int(value) for value in values]


def _record_text(record: Mapping[str, Any] | str) -> str:
    if isinstance(record, str):
        value = record
    elif isinstance(record, Mapping):
        value = record.get("text")
    else:
        raise TypeError("corpus records must be strings or mappings with a text field")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("corpus records require non-empty text")
    return value


def _record_identity(record: Mapping[str, Any] | str, index: int) -> Dict[str, Any]:
    if isinstance(record, Mapping):
        return {
            key: record.get(key)
            for key in ("document_id", "document_hash", "source_ref")
            if record.get(key) is not None
        } or {"source_index": index}
    return {"source_index": index}


@dataclass(frozen=True)
class PackedCorpusSequence:
    """One variable-length causal-LM sequence and its source spans."""

    input_ids: tuple[int, ...]
    source_spans: tuple[Dict[str, Any], ...]

    @property
    def attention_mask(self) -> tuple[int, ...]:
        return (1,) * len(self.input_ids)

    @property
    def labels(self) -> tuple[int, ...]:
        return self.input_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "labels": list(self.labels),
            "source_spans": [dict(value) for value in self.source_spans],
        }


@dataclass(frozen=True)
class PackedCorpusSplit:
    """Packed sequences plus exact statistics for one immutable split."""

    sequences: tuple[PackedCorpusSequence, ...]
    statistics: Dict[str, Any]
    source_hash: str


def pack_corpus_records(
    records: Iterable[Mapping[str, Any] | str],
    tokenizer: Any,
    *,
    max_sequence_length: int,
    eos_token_id: Optional[int] = None,
) -> PackedCorpusSplit:
    """Pack one corpus split deterministically.

    Blocks are variable length.  A trainer collator may pad them within a
    batch, but padding never becomes a target token.  Every emitted block has
    at least two tokens so causal next-token loss always has a target.
    """

    block_size = int(max_sequence_length)
    if block_size < 2:
        raise ValueError("max_sequence_length must be at least 2")
    eos = eos_token_id
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError("corpus packing requires tokenizer.eos_token_id")
    eos = int(eos)

    sequences: list[PackedCorpusSequence] = []
    current_tokens: list[int] = []
    current_spans: list[Dict[str, Any]] = []
    source_hasher = hashlib.sha256()
    document_count = 0
    paragraph_count = 0
    content_tokens = 0
    eos_tokens = 0
    long_paragraph_splits = 0
    dropped_tokens = 0

    def flush() -> bool:
        """Emit the pending block.  Return ``True`` when it was drained."""

        nonlocal current_tokens, current_spans
        if not current_tokens:
            return True
        if len(current_tokens) == 1:
            # A single EOS-only block has no causal target.  Attach it to the
            # previous block when that block still has room.  Otherwise report
            # that no progress was made so the caller keeps filling instead of
            # re-entering with identical state.
            if sequences and len(sequences[-1].input_ids) < block_size:
                previous = sequences.pop()
                sequences.append(
                    PackedCorpusSequence(
                        previous.input_ids + tuple(current_tokens),
                        previous.source_spans + tuple(current_spans),
                    )
                )
                current_tokens = []
                current_spans = []
                return True
            return False
        sequences.append(PackedCorpusSequence(tuple(current_tokens), tuple(current_spans)))
        current_tokens = []
        current_spans = []
        return True

    def append_unit(tokens: Sequence[int], span: Dict[str, Any]) -> None:
        nonlocal current_tokens, current_spans, long_paragraph_splits
        offset = 0
        values = [int(value) for value in tokens]
        while offset < len(values):
            remaining = block_size - len(current_tokens)
            available = len(values) - offset
            if current_tokens and available > remaining:
                # Preserve the paragraph boundary when the paragraph would fit
                # in a fresh block.  Only an intrinsically oversized paragraph
                # is split.  When the pending block cannot be drained (a lone
                # token with nowhere to attach) fall through and fill it from
                # this unit: retrying the flush would never make progress.
                if available <= block_size and flush():
                    continue
            take = min(block_size - len(current_tokens), available)
            start = offset
            end = offset + take
            current_tokens.extend(values[start:end])
            current_spans.append({**span, "token_start": start, "token_end": end})
            offset = end
            if len(current_tokens) == block_size:
                if offset < len(values):
                    long_paragraph_splits += 1
                flush()

    for document_index, record in enumerate(records):
        text = _record_text(record)
        identity = _record_identity(record, document_index)
        source_hasher.update(_canonical_json(record).encode("utf-8"))
        source_hasher.update(b"\n")
        paragraphs = paragraph_units(text)
        if not paragraphs:
            continue
        document_count += 1
        for paragraph_index, paragraph in enumerate(paragraphs):
            paragraph_tokens = _encode(tokenizer, paragraph)
            paragraph_count += 1
            content_tokens += len(paragraph_tokens)
            if paragraph_tokens:
                append_unit(
                    paragraph_tokens,
                    {
                        **identity,
                        "document_index": document_index,
                        "paragraph_index": paragraph_index,
                        "kind": "paragraph",
                    },
                )
        append_unit(
            [eos],
            {
                **identity,
                "document_index": document_index,
                "paragraph_index": len(paragraphs) - 1,
                "kind": "eos",
            },
        )
        eos_tokens += 1
    flush()

    # If the final document ended exactly at a block boundary, EOS is the only
    # token in the pending block. Rebalance one token from the preceding full
    # block so both sequences retain a causal target without overlap.  The
    # donor must keep two tokens of its own, which only a block size of two
    # can prevent.
    if len(current_tokens) == 1 and sequences and len(sequences[-1].input_ids) >= 3:
        previous = sequences.pop()
        previous_spans = previous.source_spans
        tail_span = dict(previous_spans[-1]) if previous_spans else {"kind": "paragraph"}
        sequences.append(PackedCorpusSequence(previous.input_ids[:-1], previous_spans))
        current_tokens.insert(0, previous.input_ids[-1])
        current_spans.insert(0, tail_span)

    if current_tokens:
        if len(current_tokens) < 2:
            # A single stranded token cannot be trained on and cannot be
            # rebalanced against a two-token donor.  Report the loss instead of
            # emitting a block with no causal next-token target.
            dropped_tokens += len(current_tokens)
            current_tokens = []
            current_spans = []
        else:
            flush()
    if not sequences:
        raise ValueError("corpus packing produced no trainable sequences")

    lengths = sorted(len(value.input_ids) for value in sequences)

    def percentile(fraction: float) -> int:
        return lengths[min(len(lengths) - 1, math.ceil(len(lengths) * fraction) - 1)]

    packed_tokens = sum(lengths)
    capacity_tokens = len(lengths) * block_size
    statistics = {
        "exact": dropped_tokens == 0,
        "algorithm": PACKING_ALGORITHM,
        "document_count": document_count,
        "paragraph_count": paragraph_count,
        "content_tokens": content_tokens,
        "eos_tokens": eos_tokens,
        "packed_tokens": packed_tokens,
        "sequence_count": len(lengths),
        "block_count": len(lengths),
        "min_sequence_tokens": lengths[0],
        "max_sequence_tokens": lengths[-1],
        "mean_sequence_tokens": packed_tokens / len(lengths),
        "p50_sequence_tokens": percentile(0.50),
        "p95_sequence_tokens": percentile(0.95),
        "capacity_tokens": capacity_tokens,
        "padding_tokens": capacity_tokens - packed_tokens,
        "utilization": packed_tokens / capacity_tokens,
        "overlap_tokens": 0,
        "dropped_tokens": dropped_tokens,
        "long_paragraph_splits": long_paragraph_splits,
    }
    if packed_tokens + dropped_tokens != content_tokens + eos_tokens:
        raise ValueError("corpus packing lost tokens without accounting for them")
    return PackedCorpusSplit(
        sequences=tuple(sequences),
        statistics=statistics,
        source_hash=source_hasher.hexdigest(),
    )


def packing_plan_hash(plan: CorpusPackingPlan | Mapping[str, Any]) -> str:
    value = plan.to_dict() if isinstance(plan, CorpusPackingPlan) else dict(plan)
    # ``artifact_hash`` links a published artifact back to the plan and is not
    # part of the plan's own deterministic identity.
    value["artifact_hash"] = None
    return _content_hash(value)


def build_corpus_packing_plan(
    *,
    train: PackedCorpusSplit,
    validation: Optional[PackedCorpusSplit],
    tokenizer_id: str,
    tokenizer_revision: Optional[str],
    tokenizer_hash: str,
    max_sequence_length: int,
    budget_mode: str,
    target_tokens: Optional[int],
    corpus_passes: Optional[float],
    effective_batch_size: int,
    artifact_hash: Optional[str] = None,
) -> CorpusPackingPlan:
    """Build the transport-level plan from exact packed split statistics."""

    mode = str(budget_mode).strip().lower()
    if mode not in {"tokens", "passes"}:
        raise ValueError("budget_mode must be 'tokens' or 'passes'")
    if mode == "tokens":
        if target_tokens is None or int(target_tokens) <= 0:
            raise ValueError("target_tokens is required for token-budget training")
        resolved_target_tokens = int(target_tokens)
        resolved_passes = None
    else:
        if corpus_passes is None or float(corpus_passes) <= 0:
            raise ValueError("corpus_passes is required for pass-budget training")
        resolved_target_tokens = None
        resolved_passes = float(corpus_passes)
    batch = int(effective_batch_size)
    if batch <= 0:
        raise ValueError("effective_batch_size must be positive")

    train_tokens = int(train.statistics["packed_tokens"])
    validation_tokens = int(validation.statistics["packed_tokens"]) if validation is not None else 0
    train_blocks = int(train.statistics["block_count"])
    validation_blocks = int(validation.statistics["block_count"]) if validation is not None else 0
    padding_tokens = int(train.statistics["padding_tokens"]) + (
        int(validation.statistics["padding_tokens"]) if validation is not None else 0
    )
    capacity = (train_blocks + validation_blocks) * int(max_sequence_length)
    utilization = (train_tokens + validation_tokens) / capacity if capacity else 0.0
    scheduled_tokens = (
        resolved_target_tokens
        if mode == "tokens"
        else max(1, math.ceil(train_tokens * float(resolved_passes)))
    )
    mean_train_block_tokens = train_tokens / max(1, train_blocks)
    tokens_per_step = max(1.0, mean_train_block_tokens * batch)
    estimated_steps = max(1, math.ceil(scheduled_tokens / tokens_per_step))

    return CorpusPackingPlan(
        tokenizer_id=str(tokenizer_id),
        tokenizer_revision=tokenizer_revision,
        tokenizer_hash=str(tokenizer_hash),
        max_sequence_length=int(max_sequence_length),
        separator=PACKING_SEPARATOR,
        packing=PACKING_ALGORITHM,
        budget_mode=mode,
        target_tokens=resolved_target_tokens,
        corpus_passes=resolved_passes,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        train_blocks=train_blocks,
        validation_blocks=validation_blocks,
        padding_tokens=padding_tokens,
        utilization=utilization,
        estimated_steps=estimated_steps,
        effective_batch_size=batch,
        artifact_hash=artifact_hash,
    )


def plan_with_hash(
    plan: CorpusPackingPlan,
    *,
    artifact_hash: str,
) -> CorpusPackingPlan:
    value = asdict(plan)
    value["artifact_hash"] = str(artifact_hash)
    return CorpusPackingPlan(**value)


__all__ = [
    "PACKING_ALGORITHM",
    "PACKING_SEPARATOR",
    "CorpusPackingPlan",
    "PackedCorpusSequence",
    "PackedCorpusSplit",
    "build_corpus_packing_plan",
    "model_identity_hash",
    "pack_corpus_records",
    "packing_plan_hash",
    "paragraph_units",
    "plan_with_hash",
    "split_paragraphs",
    "tokenizer_identity_hash",
]
