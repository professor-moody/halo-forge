"""Canonical record schemas and adapters for Dataset Lab.

Canonical records intentionally remain JSON-compatible dictionaries.  Trainers and
exporters can therefore consume them without importing these dataclasses, while this
module supplies the validation and field-mapping contract used by every front end.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .errors import SchemaError
from .identity import INTERNAL_LINEAGE_KEY


class CanonicalKind(str, Enum):
    CORPUS = "corpus"
    SFT = "sft"
    CHAT = "chat"
    PREFERENCE = "preference"
    PROMPT = "prompt"
    RLVR = "rlvr"  # accepted alias; normalized to prompt shape
    TOOL = "tool"
    VLM = "vlm"
    AUDIO = "audio"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"
    RERANKING = "reranking"


@dataclass(frozen=True)
class CanonicalSchema:
    kind: CanonicalKind
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    modality: str = "text"


SCHEMAS: Dict[CanonicalKind, CanonicalSchema] = {
    CanonicalKind.CORPUS: CanonicalSchema(
        CanonicalKind.CORPUS,
        ("document_id", "document_hash", "text", "source_ref"),
        ("title", "source_spans", "timestamp", "metadata"),
    ),
    CanonicalKind.SFT: CanonicalSchema(
        CanonicalKind.SFT, ("prompt", "response"), ("system", "metadata")
    ),
    CanonicalKind.CHAT: CanonicalSchema(CanonicalKind.CHAT, ("messages",), ("metadata",)),
    CanonicalKind.PREFERENCE: CanonicalSchema(
        CanonicalKind.PREFERENCE,
        ("prompt", "chosen", "rejected"),
        ("system", "metadata"),
    ),
    CanonicalKind.PROMPT: CanonicalSchema(
        CanonicalKind.PROMPT, ("prompt",), ("reference_answer", "metadata")
    ),
    CanonicalKind.RLVR: CanonicalSchema(
        CanonicalKind.RLVR, ("prompt",), ("reference_answer", "metadata")
    ),
    CanonicalKind.TOOL: CanonicalSchema(
        CanonicalKind.TOOL,
        ("messages", "tools"),
        ("expected_calls", "expected_results", "metadata"),
    ),
    CanonicalKind.VLM: CanonicalSchema(
        CanonicalKind.VLM,
        ("image", "prompt"),
        ("response", "ground_truth", "alternatives", "metadata"),
        "image",
    ),
    CanonicalKind.AUDIO: CanonicalSchema(
        CanonicalKind.AUDIO,
        ("audio", "task"),
        ("transcript", "label", "metadata"),
        "audio",
    ),
    CanonicalKind.CLASSIFICATION: CanonicalSchema(
        CanonicalKind.CLASSIFICATION,
        (),
        ("input", "media", "label", "labels", "metadata"),
    ),
    CanonicalKind.EMBEDDING: CanonicalSchema(
        CanonicalKind.EMBEDDING,
        ("anchor", "positive"),
        ("negatives", "metadata"),
    ),
    CanonicalKind.RERANKING: CanonicalSchema(
        CanonicalKind.RERANKING,
        ("query",),
        ("document", "candidates", "relevance", "ordered_preference", "metadata"),
    ),
}


@dataclass
class AdaptedRecord:
    kind: CanonicalKind
    record: Dict[str, Any]
    source_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "record": copy.deepcopy(self.record),
            "source_index": self.source_index,
        }


_ALIASES: Dict[CanonicalKind, Dict[str, tuple[str, ...]]] = {
    CanonicalKind.CORPUS: {
        "document_id": ("document_id", "id", "doc_id"),
        "document_hash": ("document_hash", "content_hash", "hash"),
        "text": ("text", "content", "document", "body"),
        "title": ("title", "name", "heading"),
        "source_ref": ("source_ref", "source", "path", "filename", "url"),
        "source_spans": ("source_spans", "spans", "provenance"),
        "timestamp": ("timestamp", "date", "created_at", "published_at"),
    },
    CanonicalKind.SFT: {
        "prompt": ("prompt", "instruction", "question", "input"),
        "response": ("response", "completion", "output", "answer"),
        "system": ("system", "system_prompt"),
    },
    CanonicalKind.CHAT: {"messages": ("messages", "conversations")},
    CanonicalKind.PREFERENCE: {
        "prompt": ("prompt", "instruction", "question"),
        "chosen": ("chosen", "preferred", "winner"),
        "rejected": ("rejected", "dispreferred", "loser"),
    },
    CanonicalKind.PROMPT: {
        "prompt": ("prompt", "instruction", "question", "problem"),
        "reference_answer": ("reference_answer", "answer", "solution", "reference"),
    },
    CanonicalKind.RLVR: {
        "prompt": ("prompt", "instruction", "question", "problem"),
        "reference_answer": ("reference_answer", "answer", "solution", "reference"),
    },
    CanonicalKind.TOOL: {
        "messages": ("messages", "conversations"),
        "tools": ("tools", "tool_definitions", "functions"),
        "expected_calls": ("expected_calls", "tool_calls"),
        "expected_results": ("expected_results", "tool_results"),
    },
    CanonicalKind.VLM: {
        "image": ("image", "image_path", "image_url", "images"),
        "prompt": ("prompt", "question", "instruction"),
        "response": ("response", "answer", "completion"),
        "ground_truth": ("ground_truth", "label", "target"),
        "alternatives": ("alternatives", "answers", "choices"),
    },
    CanonicalKind.AUDIO: {
        "audio": ("audio", "audio_path", "file", "path"),
        "task": ("task", "instruction", "prompt"),
        "transcript": ("transcript", "text", "sentence"),
        "label": ("label", "target", "class"),
    },
    CanonicalKind.CLASSIFICATION: {
        "input": ("input", "text", "content", "sentence"),
        "media": ("media", "image", "audio", "file", "path"),
        "label": ("label", "class", "target", "category"),
        "labels": ("labels", "classes", "targets", "categories"),
    },
    CanonicalKind.EMBEDDING: {
        "anchor": ("anchor", "query", "question", "input"),
        "positive": ("positive", "document", "passage", "answer"),
        "negatives": ("negatives", "negative", "hard_negatives"),
    },
    CanonicalKind.RERANKING: {
        "query": ("query", "question", "prompt"),
        "document": ("document", "passage", "candidate"),
        "candidates": ("candidates", "documents", "passages"),
        "relevance": ("relevance", "score", "label"),
        "ordered_preference": ("ordered_preference", "ranking", "order"),
    },
}


def normalize_kind(kind: CanonicalKind | str) -> CanonicalKind:
    aliases = {
        "reasoning": "prompt",
        "rlvr/reasoning": "prompt",
        "tool_use": "tool",
        "tool-use": "tool",
        "agentic": "tool",
        "classify": "classification",
        "multilabel": "classification",
        "rerank": "reranking",
    }
    raw = str(kind.value if isinstance(kind, CanonicalKind) else kind).lower()
    raw = aliases.get(raw, raw)
    try:
        return CanonicalKind(raw)
    except ValueError as exc:
        raise SchemaError(
            f"Unknown canonical schema {kind!r}; choose: {', '.join(k.value for k in CanonicalKind)}"
        ) from exc


def get_field(record: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Read a dotted field path without evaluating user code."""
    value: Any = record
    for part in str(path).split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return default
    return value


def set_field(record: Dict[str, Any], path: str, value: Any) -> None:
    target = record
    parts = str(path).split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _first(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    marker = object()
    for name in names:
        value = get_field(record, name, marker)
        if value is not marker:
            return value
    return None


def _normalize_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    normalized = []
    for message in messages:
        if not isinstance(message, Mapping):
            normalized.append(message)
            continue
        normalized.append(
            {
                "role": message.get("role", message.get("from")),
                "content": message.get("content", message.get("value")),
                **({"tool_calls": message["tool_calls"]} if "tool_calls" in message else {}),
                **(
                    {"function_call": message["function_call"]}
                    if "function_call" in message
                    else {}
                ),
                **({"name": message["name"]} if "name" in message else {}),
                **(
                    {"tool_call_id": message["tool_call_id"]}
                    if "tool_call_id" in message
                    else {}
                ),
            }
        )
    return normalized


def _mapping_expression_value(record: Mapping[str, Any], expression: Any) -> Any:
    """Evaluate the safe mapping-v2 wire shape while retaining legacy strings."""
    if isinstance(expression, str):
        return get_field(record, expression)
    if not isinstance(expression, Mapping):
        raise SchemaError("field mapping values must be source paths or mapping-v2 objects")
    kind = str(expression.get("kind") or expression.get("type") or "direct").lower()
    source = expression.get("source")
    if kind == "direct":
        if not source:
            raise SchemaError("direct mapping requires source")
        return copy.deepcopy(get_field(record, str(source)))
    if kind == "nested_path":
        if not source:
            raise SchemaError("nested_path mapping requires source")
        base = get_field(record, str(source))
        path = expression.get("path")
        if path:
            return copy.deepcopy(get_field(base, str(path))) if isinstance(base, Mapping) else None
        return copy.deepcopy(base)
    if kind == "constant":
        return copy.deepcopy(expression.get("value"))
    if kind == "concat":
        sources = expression.get("sources") or []
        if not isinstance(sources, list) or not sources:
            raise SchemaError("concat mapping requires a non-empty sources list")
        separator = str(expression.get("separator", "\n"))
        return separator.join(
            str(value)
            for value in (get_field(record, str(name)) for name in sources)
            if value is not None
        )
    if kind in {"conversation", "role_normalize"}:
        if not source:
            raise SchemaError("conversation mapping requires source")
        messages = get_field(record, str(source))
        if not isinstance(messages, list):
            return messages
        role_field = str(expression.get("role_field", "role"))
        content_field = str(expression.get("content_field", "content"))
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "assistant": "assistant",
            "system": "system",
            "function": "tool",
            "tool": "tool",
            **{str(k): str(v) for k, v in dict(expression.get("role_map") or {}).items()},
        }
        output = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise SchemaError("conversation messages must be objects")
            raw_role = str(
                message.get(role_field, message.get("role", message.get("from", "")))
            ).lower()
            normalized = {
                "role": role_map.get(raw_role, raw_role),
                "content": message.get(
                    content_field, message.get("content", message.get("value"))
                ),
            }
            for key in ("tool_calls", "function_call", "name", "tool_call_id"):
                if key in message:
                    normalized[key] = copy.deepcopy(message[key])
            output.append(normalized)
        return output
    if kind in {"media_root", "media_path"}:
        if not source:
            raise SchemaError("media_root mapping requires source")
        value = get_field(record, str(source))
        root_value = expression.get("root", expression.get("media_root"))
        if not root_value or not isinstance(value, str):
            return copy.deepcopy(value)
        selected_root = Path(str(root_value)).expanduser()
        if selected_root.is_symlink():
            raise SchemaError("symbolic-link media roots are not accepted")
        root = selected_root.resolve()
        path = Path(value).expanduser()
        candidate = path if path.is_absolute() else root / path
        try:
            relative_candidate = candidate.relative_to(root)
            cursor = root
            for part in relative_candidate.parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise SchemaError("symbolic-link media paths are not accepted")
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except ValueError as exc:
            raise SchemaError("media path escapes the selected media root") from exc
        if not resolved.is_file():
            raise SchemaError(f"media asset does not exist: {value}")
        return str(resolved)
    raise SchemaError(f"unsupported mapping-v2 expression kind: {kind}")


def adapt_record(
    record: Mapping[str, Any],
    kind: CanonicalKind | str,
    *,
    mapping: Optional[Mapping[str, Any]] = None,
    preserve_unmapped_metadata: bool = True,
) -> Dict[str, Any]:
    """Map a source record to a canonical, JSON-compatible record.

    ``mapping`` is target-to-source (for example ``{"response": "answer.text"}``).
    Missing explicit fields are not silently filled from aliases, making recipes
    predictable.  Without an explicit map, common source aliases are recognized.
    """
    if not isinstance(record, Mapping):
        raise SchemaError(f"Expected an object record, got {type(record).__name__}")
    resolved = normalize_kind(kind)
    aliases = _ALIASES[resolved]
    output: Dict[str, Any] = {}
    consumed: set[str] = set()
    for target in (*SCHEMAS[resolved].required, *SCHEMAS[resolved].optional):
        if target == "metadata":
            continue
        if mapping is not None and target in mapping:
            expression = mapping[target]
            value = _mapping_expression_value(record, expression)
            if isinstance(expression, str):
                consumed.add(expression.split(".", 1)[0])
            elif isinstance(expression, Mapping):
                names = list(expression.get("sources") or [])
                if expression.get("source"):
                    names.append(expression["source"])
                consumed.update(str(name).split(".", 1)[0] for name in names)
        elif mapping is not None:
            value = None
        else:
            choices = aliases.get(target, (target,))
            value = _first(record, choices)
            consumed.update(
                name.split(".", 1)[0] for name in choices if get_field(record, name) is not None
            )
        if value is not None:
            output[target] = copy.deepcopy(value)

    if "messages" in output:
        output["messages"] = _normalize_messages(output["messages"])
    # Dataset Lab attaches this private envelope only while a recipe is
    # running.  A map step replaces the row, so explicitly carry the envelope
    # forward; VersionStore strips it before writing canonical data.
    if isinstance(record.get(INTERNAL_LINEAGE_KEY), Mapping):
        output[INTERNAL_LINEAGE_KEY] = copy.deepcopy(record[INTERNAL_LINEAGE_KEY])
    existing_metadata = record.get("metadata")
    metadata: Dict[str, Any] = (
        copy.deepcopy(existing_metadata) if isinstance(existing_metadata, Mapping) else {}
    )
    if preserve_unmapped_metadata:
        for key, value in record.items():
            if (
                key not in consumed
                and key not in {"metadata", INTERNAL_LINEAGE_KEY}
                and key not in output
            ):
                metadata[key] = copy.deepcopy(value)
    if metadata:
        output["metadata"] = metadata
    validate_record(output, resolved)
    return output


def _nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def validate_record(record: Mapping[str, Any], kind: CanonicalKind | str) -> None:
    resolved = normalize_kind(kind)
    schema = SCHEMAS[resolved]
    missing = [field for field in schema.required if not _nonempty(record.get(field))]
    if missing:
        raise SchemaError(
            f"{resolved.value} record is missing required field(s): {', '.join(missing)}"
        )
    if resolved in {CanonicalKind.CHAT, CanonicalKind.TOOL}:
        messages = record.get("messages")

        def valid_message(message: Any) -> bool:
            if not isinstance(message, Mapping) or not _nonempty(message.get("role")):
                return False
            if _nonempty(message.get("content")):
                return True
            # OpenAI-compatible tool traces intentionally use an empty or null
            # assistant ``content`` while the actual assistant action lives in
            # ``tool_calls`` (or the legacy singular ``function_call``).  These
            # are semantically complete turns and must survive canonicalization.
            return str(message.get("role")).lower() == "assistant" and (
                _nonempty(message.get("tool_calls"))
                or _nonempty(message.get("function_call"))
            )

        if not isinstance(messages, list) or any(
            not valid_message(message) for message in messages
        ):
            raise SchemaError(
                "messages must be a non-empty list of role/content objects; "
                "assistant tool-call turns may use empty content"
            )
    if resolved == CanonicalKind.VLM and not (
        _nonempty(record.get("response")) or _nonempty(record.get("ground_truth"))
    ):
        raise SchemaError("vlm record requires response or ground_truth")
    if resolved == CanonicalKind.AUDIO and not (
        _nonempty(record.get("transcript")) or _nonempty(record.get("label"))
    ):
        raise SchemaError("audio record requires transcript or label")
    if resolved == CanonicalKind.CLASSIFICATION and not (
        _nonempty(record.get("input")) or _nonempty(record.get("media"))
    ):
        raise SchemaError("classification record requires input or media")
    if resolved == CanonicalKind.CLASSIFICATION and not (
        _nonempty(record.get("label")) or _nonempty(record.get("labels"))
    ):
        raise SchemaError("classification record requires label or labels")
    if resolved == CanonicalKind.RERANKING and not (
        _nonempty(record.get("document")) or _nonempty(record.get("candidates"))
    ):
        raise SchemaError("reranking record requires document or candidates")
    if resolved == CanonicalKind.RERANKING and not (
        _nonempty(record.get("relevance"))
        or _nonempty(record.get("ordered_preference"))
    ):
        raise SchemaError("reranking record requires relevance or ordered_preference")


def infer_schema(record: Mapping[str, Any]) -> CanonicalKind:
    """Infer the most specific canonical schema from a representative row."""
    keys = set(record)
    if keys & {"anchor", "positive", "negatives", "hard_negatives"} and (
        "anchor" in keys or "positive" in keys
    ):
        return CanonicalKind.EMBEDDING
    if keys & {"query"} and keys & {
        "document",
        "candidates",
        "relevance",
        "ranking",
        "ordered_preference",
    }:
        return CanonicalKind.RERANKING
    if keys & {"label", "labels", "class", "classes", "category"} and keys & {
        "input",
        "text",
        "content",
        "sentence",
    }:
        return CanonicalKind.CLASSIFICATION
    if keys & {"audio", "audio_path"}:
        return CanonicalKind.AUDIO
    if keys & {"image", "image_path", "image_url", "images"}:
        return CanonicalKind.VLM
    if "messages" in keys or "conversations" in keys:
        if keys & {"tools", "tool_definitions", "functions", "expected_calls", "tool_calls"}:
            return CanonicalKind.TOOL
        return CanonicalKind.CHAT
    if (keys & {"chosen", "preferred", "winner"}) and (
        keys & {"rejected", "dispreferred", "loser"}
    ):
        return CanonicalKind.PREFERENCE
    if keys & {"response", "completion", "output"}:
        return CanonicalKind.SFT
    if (
        keys & {"text", "content", "document", "body"}
        and not keys & {"prompt", "instruction", "question"}
    ):
        return CanonicalKind.CORPUS
    return CanonicalKind.PROMPT


def schema_info(kind: CanonicalKind | str) -> Dict[str, Any]:
    schema = SCHEMAS[normalize_kind(kind)]
    result = asdict(schema)
    result["kind"] = schema.kind.value
    return result


__all__ = [
    "AdaptedRecord",
    "CanonicalKind",
    "CanonicalSchema",
    "SCHEMAS",
    "adapt_record",
    "get_field",
    "infer_schema",
    "normalize_kind",
    "schema_info",
    "set_field",
    "validate_record",
]
