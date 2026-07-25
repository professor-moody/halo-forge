"""Canonical identities and transport-neutral contracts for corpus extraction."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

CORPUS_EXTRACTOR_VERSION = "corpus-extraction-v2"
CORPUS_BUNDLE_FORMAT = "halo-forge-corpus-extraction"
CORPUS_BUNDLE_FORMAT_VERSION = 1
CORPUS_IDENTITY_VERSION = 1

SUPPORTED_EXTENSIONS = (
    ".csv",
    ".docx",
    ".htm",
    ".html",
    ".jl",
    ".json",
    ".jsonl",
    ".markdown",
    ".md",
    ".mdown",
    ".mkd",
    ".parquet",
    ".pdf",
    ".text",
    ".tsv",
    ".txt",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(value: str) -> str:
    """Normalize extracted text without destroying paragraph boundaries."""

    text = unicodedata.normalize("NFC", str(value or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines).strip()
    return re.sub(r"\n{4,}", "\n\n\n", text)


def _normalized_extensions(values: Sequence[str]) -> tuple[str, ...]:
    extensions: set[str] = set()
    for value in values:
        suffix = str(value or "").strip().lower()
        if not suffix:
            continue
        extensions.add(suffix if suffix.startswith(".") else f".{suffix}")
    return tuple(sorted(extensions))


@dataclass(frozen=True)
class CorpusExtractionConfig:
    """Deterministic extraction choices that participate in reuse identity."""

    text_columns: tuple[str, ...] = ()
    title_column: Optional[str] = None
    id_column: Optional[str] = None
    metadata_columns: tuple[str, ...] = ()
    join_separator: str = "\n\n"
    min_text_chars: int = 1
    include_extensions: tuple[str, ...] = SUPPORTED_EXTENSIONS
    include_hidden: bool = False
    pdf_page_documents: bool = True

    @classmethod
    def from_value(
        cls, value: "CorpusExtractionConfig | Mapping[str, Any] | None"
    ) -> "CorpusExtractionConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("corpus extraction config must be an object")

        def columns(name: str) -> tuple[str, ...]:
            raw = value.get(name) or ()
            if isinstance(raw, str):
                raw = (raw,)
            return tuple(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))

        include_extensions = value.get("include_extensions", SUPPORTED_EXTENSIONS)
        if isinstance(include_extensions, str):
            include_extensions = (include_extensions,)
        min_text_chars = int(value.get("min_text_chars", 1))
        if min_text_chars < 1:
            raise ValueError("min_text_chars must be at least 1")
        separator = str(value.get("join_separator", "\n\n"))
        if not separator:
            raise ValueError("join_separator cannot be empty")
        return cls(
            text_columns=columns("text_columns"),
            title_column=(
                str(value["title_column"]).strip()
                if value.get("title_column") is not None
                else None
            ),
            id_column=(
                str(value["id_column"]).strip() if value.get("id_column") is not None else None
            ),
            metadata_columns=columns("metadata_columns"),
            join_separator=separator,
            min_text_chars=min_text_chars,
            include_extensions=_normalized_extensions(include_extensions),
            include_hidden=bool(value.get("include_hidden", False)),
            pdf_page_documents=bool(value.get("pdf_page_documents", True)),
        )

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json(self.to_dict()).encode("utf-8"))

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        for key in ("text_columns", "metadata_columns", "include_extensions"):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True)
class CorpusDocument:
    """One canonical text document with immutable content/origin identity."""

    id: str
    text: str
    source_uri: str
    source_kind: str
    media_type: str
    source_fingerprint: str
    content_hash: str
    ordinal: int
    relative_path: str = ""
    title: Optional[str] = None
    locator: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        text: str,
        source_uri: str,
        source_kind: str,
        media_type: str,
        source_fingerprint: str,
        ordinal: int,
        relative_path: str = "",
        title: Optional[str] = None,
        locator: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        extractor_version: str = CORPUS_EXTRACTOR_VERSION,
    ) -> "CorpusDocument":
        normalized = normalize_text(text)
        content_hash = sha256_bytes(normalized.encode("utf-8"))
        normalized_locator = copy.deepcopy(dict(locator or {}))
        identity = sha256_bytes(
            canonical_json(
                {
                    "identity_version": CORPUS_IDENTITY_VERSION,
                    "source_fingerprint": source_fingerprint,
                    "relative_path": str(relative_path),
                    "locator": normalized_locator,
                    "content_hash": content_hash,
                    "extractor_version": extractor_version,
                }
            ).encode("utf-8")
        )
        evidence = {
            "identity_version": CORPUS_IDENTITY_VERSION,
            "extractor_version": extractor_version,
            "source_fingerprint": source_fingerprint,
            "source_uri": source_uri,
            "source_kind": source_kind,
            "media_type": media_type,
            "relative_path": str(relative_path),
            "locator": copy.deepcopy(normalized_locator),
            **copy.deepcopy(dict(provenance or {})),
        }
        return cls(
            id=f"doc-{identity}",
            text=normalized,
            source_uri=str(source_uri),
            source_kind=str(source_kind),
            media_type=str(media_type),
            source_fingerprint=str(source_fingerprint),
            content_hash=content_hash,
            ordinal=int(ordinal),
            relative_path=str(relative_path),
            title=normalize_text(title) if title else None,
            locator=normalized_locator,
            provenance=evidence,
            metadata=copy.deepcopy(dict(metadata or {})),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorpusDocument":
        return cls(
            id=str(value["id"]),
            text=str(value["text"]),
            source_uri=str(value["source_uri"]),
            source_kind=str(value["source_kind"]),
            media_type=str(value["media_type"]),
            source_fingerprint=str(value["source_fingerprint"]),
            content_hash=str(value.get("content_hash") or value.get("text_sha256")),
            ordinal=int(value.get("ordinal", 0)),
            relative_path=str(value.get("relative_path") or ""),
            title=(str(value["title"]) if value.get("title") is not None else None),
            locator=copy.deepcopy(dict(value.get("locator") or {})),
            provenance=copy.deepcopy(dict(value.get("provenance") or {})),
            metadata=copy.deepcopy(dict(value.get("metadata") or {})),
        )

    @property
    def document_id(self) -> str:
        return self.id

    @property
    def document_hash(self) -> str:
        return self.content_hash

    @property
    def text_sha256(self) -> str:
        return self.content_hash

    @property
    def source_ref(self) -> str:
        explicit = str(self.metadata.get("source_ref") or "").strip()
        if explicit:
            return explicit
        # A structured file is a container of independent logical documents.
        # Give each row its own stable reference so grouped corpus splits do
        # not collapse an entire JSONL/CSV/Parquet source into one group.
        if self.provenance.get("structured_format") is not None:
            row = self.locator.get("row")
            if row is not None:
                base = self.relative_path or self.source_uri
                return f"{base}#row={int(row)}"
        # Page-level PDF extraction deliberately keeps the file-level
        # reference. Grouping by source_ref therefore keeps every page from
        # one PDF on the same side of the train/validation boundary.
        return self.relative_path or self.source_uri

    @property
    def source_spans(self) -> list[Dict[str, Any]]:
        explicit = self.metadata.get("source_spans")
        if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes, bytearray)):
            return [copy.deepcopy(dict(value)) for value in explicit if isinstance(value, Mapping)]
        return [{"source_ref": self.source_ref, **copy.deepcopy(self.locator)}]

    @property
    def timestamp(self) -> Optional[str]:
        for key in ("timestamp", "created_at", "published_at", "date"):
            value = self.metadata.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        selected = self.metadata.get("selected_metadata")
        if isinstance(selected, Mapping):
            for key in ("timestamp", "created_at", "published_at", "date"):
                value = selected.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        return None

    @property
    def text_char_count(self) -> int:
        return len(self.text)

    @property
    def text_byte_count(self) -> int:
        return len(self.text.encode("utf-8"))

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["document_id"] = self.id
        value["document_hash"] = self.content_hash
        value["text_sha256"] = self.content_hash
        value["source_ref"] = self.source_ref
        value["source_spans"] = self.source_spans
        value["timestamp"] = self.timestamp
        value["text_char_count"] = self.text_char_count
        value["text_byte_count"] = self.text_byte_count
        return value

    def to_index_item(self, *, bundle_ordinal: int) -> Dict[str, Any]:
        return {
            "document_id": self.id,
            "status": "extracted",
            "source_uri": self.source_uri,
            "relative_path": self.relative_path,
            "source_kind": self.source_kind,
            "media_type": self.media_type,
            "title": self.title,
            "content_hash": self.content_hash,
            "text_char_count": self.text_char_count,
            "text_byte_count": self.text_byte_count,
            "bundle_member": "documents.jsonl",
            "bundle_ordinal": int(bundle_ordinal),
            "locator": copy.deepcopy(self.locator),
            "provenance": copy.deepcopy(self.provenance),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class ExtractionFailure:
    """A deterministic, persisted quarantine entry for one failed input item."""

    id: str
    source_uri: str
    source_kind: str
    media_type: str
    source_fingerprint: str
    relative_path: str
    ordinal: int
    error_code: str
    error: str
    locator: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        source_uri: str,
        source_kind: str,
        media_type: str,
        source_fingerprint: str,
        relative_path: str,
        ordinal: int,
        error_code: str,
        error: str,
        locator: Optional[Mapping[str, Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ExtractionFailure":
        normalized_locator = copy.deepcopy(dict(locator or {}))
        identity = sha256_bytes(
            canonical_json(
                {
                    "identity_version": CORPUS_IDENTITY_VERSION,
                    "source_fingerprint": source_fingerprint,
                    "relative_path": str(relative_path),
                    "locator": normalized_locator,
                    "error_code": str(error_code),
                }
            ).encode("utf-8")
        )
        evidence = {
            "identity_version": CORPUS_IDENTITY_VERSION,
            "extractor_version": CORPUS_EXTRACTOR_VERSION,
            "source_fingerprint": source_fingerprint,
            "source_uri": source_uri,
            "source_kind": source_kind,
            "media_type": media_type,
            "relative_path": str(relative_path),
            "locator": copy.deepcopy(normalized_locator),
            **copy.deepcopy(dict(provenance or {})),
        }
        return cls(
            id=f"quarantine-{identity}",
            source_uri=str(source_uri),
            source_kind=str(source_kind),
            media_type=str(media_type),
            source_fingerprint=str(source_fingerprint),
            relative_path=str(relative_path),
            ordinal=int(ordinal),
            error_code=str(error_code),
            error=normalize_text(error) or str(error_code),
            locator=normalized_locator,
            provenance=evidence,
            metadata=copy.deepcopy(dict(metadata or {})),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExtractionFailure":
        return cls(
            id=str(value.get("id") or value.get("document_id")),
            source_uri=str(value["source_uri"]),
            source_kind=str(value["source_kind"]),
            media_type=str(value["media_type"]),
            source_fingerprint=str(value["source_fingerprint"]),
            relative_path=str(value.get("relative_path") or ""),
            ordinal=int(value.get("ordinal", 0)),
            error_code=str(value["error_code"]),
            error=str(value["error"]),
            locator=copy.deepcopy(dict(value.get("locator") or {})),
            provenance=copy.deepcopy(dict(value.get("provenance") or {})),
            metadata=copy.deepcopy(dict(value.get("metadata") or {})),
        )

    @property
    def document_id(self) -> str:
        return self.id

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["document_id"] = self.id
        value["status"] = "quarantined"
        return value

    def to_index_item(self, *, bundle_ordinal: int) -> Dict[str, Any]:
        return {
            "document_id": self.id,
            "status": "quarantined",
            "source_uri": self.source_uri,
            "relative_path": self.relative_path,
            "source_kind": self.source_kind,
            "media_type": self.media_type,
            "bundle_member": "quarantine.jsonl",
            "bundle_ordinal": int(bundle_ordinal),
            "locator": copy.deepcopy(self.locator),
            "provenance": copy.deepcopy(self.provenance),
            "metadata": copy.deepcopy(self.metadata),
            "error_code": self.error_code,
            "error": self.error,
        }


@dataclass(frozen=True)
class CorpusBundle:
    extraction_id: str
    content_hash: str
    path: str
    manifest_hash: str
    document_count: int
    quarantined_count: int
    checksums: Dict[str, str]
    created_at: str
    reused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusBundleVerification:
    content_hash: str
    valid: bool
    path: str
    checksums: Dict[str, str] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["errors"] = list(self.errors)
        return value


@dataclass(frozen=True)
class CorpusExtractionResult:
    extraction_id: str
    source_uri: str
    source_kind: str
    source_fingerprint: str
    extractor_version: str
    config_hash: str
    documents: tuple[CorpusDocument, ...]
    quarantine: tuple[ExtractionFailure, ...]
    bundle: CorpusBundle
    statistics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_records(self) -> tuple[Dict[str, Any], ...]:
        return tuple(document.to_dict() for document in self.documents)

    @property
    def extraction(self) -> Dict[str, Any]:
        return {
            "id": self.extraction_id,
            "status": "completed",
            "source_uri": self.source_uri,
            "source_kind": self.source_kind,
            "source_fingerprint": self.source_fingerprint,
            "extractor_version": self.extractor_version,
            "config_hash": self.config_hash,
            "content_hash": self.bundle.content_hash,
            "bundle_path": self.bundle.path,
            "manifest_hash": self.bundle.manifest_hash,
            "document_count": len(self.documents),
            "item_count": len(self.documents) + len(self.quarantine),
            "quarantined_count": len(self.quarantine),
            "statistics": copy.deepcopy(self.statistics),
            "provenance": copy.deepcopy(self.provenance),
            "reused": self.bundle.reused,
        }

    def to_dict(self) -> Dict[str, Any]:
        records = [document.to_dict() for document in self.documents]
        return {
            "extraction": self.extraction,
            "records": records,
            "documents": copy.deepcopy(records),
            "quarantine": [failure.to_dict() for failure in self.quarantine],
            "bundle": self.bundle.to_dict(),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


def default_extraction_id(
    source_fingerprint: str,
    config_hash: str,
    *,
    extractor_version: str = CORPUS_EXTRACTOR_VERSION,
) -> str:
    identity = sha256_bytes(
        canonical_json(
            {
                "source_fingerprint": source_fingerprint,
                "extractor_version": extractor_version,
                "config_hash": config_hash,
            }
        ).encode("utf-8")
    )
    return f"dex-{identity[:24]}"


def source_display_uri(path: Path | str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


__all__ = [
    "CORPUS_BUNDLE_FORMAT",
    "CORPUS_BUNDLE_FORMAT_VERSION",
    "CORPUS_EXTRACTOR_VERSION",
    "CORPUS_IDENTITY_VERSION",
    "SUPPORTED_EXTENSIONS",
    "CorpusBundle",
    "CorpusBundleVerification",
    "CorpusDocument",
    "CorpusExtractionConfig",
    "CorpusExtractionResult",
    "ExtractionFailure",
    "canonical_json",
    "default_extraction_id",
    "normalize_text",
    "sha256_bytes",
    "source_display_uri",
]
