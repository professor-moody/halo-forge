"""Deterministic extractors for document and structured-text corpus sources."""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence
from xml.etree import ElementTree

from halo_forge.data_lab.models import get_field
from halo_forge.own_data.inspection import fingerprint_path, iter_file_records

from .models import (
    CORPUS_EXTRACTOR_VERSION,
    CorpusDocument,
    CorpusExtractionConfig,
    ExtractionFailure,
    canonical_json,
    normalize_text,
)

ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], None]

_PLAIN_TEXT_EXTENSIONS = {".txt", ".text"}
_MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd"}
_HTML_EXTENSIONS = {".html", ".htm"}
_STRUCTURED_EXTENSIONS = {".json", ".jsonl", ".jl", ".csv", ".tsv", ".parquet"}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
_SUPPRESSED_HTML_TAGS = {
    "canvas",
    "head",
    "iframe",
    "noscript",
    "script",
    "style",
    "svg",
    "template",
}
_VOID_HTML_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_AUTO_TEXT_COLUMNS = (
    "text",
    "content",
    "body",
    "document",
    "article",
    "markdown",
    "transcript",
    "caption",
    "description",
    "prompt",
    "instruction",
    "question",
    "problem",
    "response",
    "completion",
    "output",
    "answer",
    "solution",
    "messages",
    "conversations",
)
_NON_TEXT_COLUMN_NAMES = {
    "id",
    "uuid",
    "hash",
    "sha256",
    "path",
    "file",
    "filename",
    "image",
    "audio",
    "metadata",
    "label",
    "score",
}


class CorpusExtractionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


class CorpusExtractionCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class _RawDocument:
    text: str
    title: Optional[str] = None
    locator: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _RawFailure:
    code: str
    error: str
    locator: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionCollection:
    source_fingerprint: str
    source_size_bytes: int
    source_file_count: int
    documents: tuple[CorpusDocument, ...]
    quarantine: tuple[ExtractionFailure, ...]
    statistics: Dict[str, Any]
    provenance: Dict[str, Any]


def _is_hidden(relative_path: Path) -> bool:
    return any(part.startswith(".") for part in relative_path.parts)


def _source_files(source: Path, config: CorpusExtractionConfig) -> tuple[list[Path], int]:
    if source.is_file():
        return [source], 0
    supported: list[Path] = []
    skipped = 0
    for path in sorted(source.rglob("*"), key=lambda value: value.relative_to(source).as_posix()):
        if path.is_symlink():
            raise CorpusExtractionError(
                "unsafe_symbolic_link",
                f"symbolic-link source entries are not accepted: {path.relative_to(source)}",
            )
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if not config.include_hidden and _is_hidden(relative):
            skipped += 1
            continue
        if path.suffix.lower() in config.include_extensions:
            supported.append(path)
        else:
            skipped += 1
    return supported, skipped


def _decode_text(path: Path) -> tuple[str, str]:
    payload = path.read_bytes()
    if payload.startswith(b"\xef\xbb\xbf"):
        return payload.decode("utf-8-sig"), "utf-8-sig"
    if payload.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return payload.decode("utf-32"), "utf-32"
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16"), "utf-16"
    try:
        return payload.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        # Windows-1252 remains a common encoding for workstation text exports.
        # It is deterministic and reversible for all byte values.
        text = payload.decode("cp1252")
        if text.count("\x00") > max(1, len(text) // 100):
            raise CorpusExtractionError(
                "binary_text_source",
                "the text source contains too many NUL bytes to be treated as text",
            )
        return text, "cp1252"


class _VisibleHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._stack: list[tuple[str, bool]] = []
        self._suppressed_depth = 0
        self._title_depth = 0

    @staticmethod
    def _hidden(tag: str, attrs: Mapping[str, Optional[str]]) -> bool:
        if tag in _SUPPRESSED_HTML_TAGS:
            return True
        if "hidden" in attrs:
            return True
        if str(attrs.get("aria-hidden") or "").strip().lower() == "true":
            return True
        style = re.sub(r"\s+", "", str(attrs.get("style") or "").lower())
        return "display:none" in style or "visibility:hidden" in style

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        normalized = tag.lower()
        attributes = {str(key).lower(): value for key, value in attrs}
        if normalized == "title":
            self._title_depth += 1
        hidden = self._suppressed_depth > 0 or self._hidden(normalized, attributes)
        if normalized not in _VOID_HTML_TAGS:
            self._stack.append((normalized, hidden))
            if hidden:
                self._suppressed_depth += 1
        if not hidden and normalized in _BLOCK_TAGS:
            self.parts.append("\n")
        if not hidden and normalized == "img":
            alt = str(attributes.get("alt") or "").strip()
            if alt:
                self.parts.append(alt)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in _VOID_HTML_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.lower()
        if normalized == "title" and self._title_depth:
            self._title_depth -= 1
        if normalized in _BLOCK_TAGS and self._suppressed_depth == 0:
            self.parts.append("\n")
        if not self._stack:
            return
        stack_index = next(
            (
                index
                for index in range(len(self._stack) - 1, -1, -1)
                if self._stack[index][0] == normalized
            ),
            len(self._stack) - 1,
        )
        removed = self._stack[stack_index:]
        del self._stack[stack_index:]
        self._suppressed_depth = max(
            0,
            self._suppressed_depth - sum(1 for _, hidden in removed if hidden),
        )

    def handle_data(self, data: str) -> None:
        if self._title_depth:
            self.title_parts.append(data)
        if self._suppressed_depth == 0 and not self._title_depth:
            self.parts.append(data)

    def visible_text(self) -> str:
        raw = "".join(self.parts)
        lines = []
        for line in raw.splitlines():
            normalized = re.sub(r"[^\S\n]+", " ", line).strip()
            if normalized:
                lines.append(normalized)
            elif lines and lines[-1] != "":
                lines.append("")
        return normalize_text("\n".join(lines))

    def title(self) -> Optional[str]:
        value = normalize_text("".join(self.title_parts))
        return value or None


def _extract_html(path: Path) -> list[_RawDocument | _RawFailure]:
    text, encoding = _decode_text(path)
    parser = _VisibleHTMLParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:
        return [
            _RawFailure(
                "invalid_html",
                f"HTML parser failed: {type(exc).__name__}: {exc}",
                metadata={"encoding": encoding},
            )
        ]
    visible = parser.visible_text()
    if not visible:
        return [
            _RawFailure(
                "html_no_visible_text",
                "the HTML document contains no visible text",
                metadata={"encoding": encoding},
            )
        ]
    return [
        _RawDocument(
            visible,
            title=parser.title(),
            metadata={"encoding": encoding},
            provenance={"html_policy": "visible-text-v1"},
        )
    ]


def _docx_paragraph_text(element: ElementTree.Element, namespace: str) -> str:
    values: list[str] = []
    for node in element.iter():
        local = node.tag.rsplit("}", 1)[-1]
        if local == "t" and node.text:
            values.append(node.text)
        elif local == "tab":
            values.append("\t")
        elif local in {"br", "cr"}:
            values.append("\n")
    return "".join(values)


def _extract_docx(path: Path) -> list[_RawDocument | _RawFailure]:
    try:
        with zipfile.ZipFile(path) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > 64 * 1024 * 1024:
                raise CorpusExtractionError(
                    "docx_document_too_large",
                    "word/document.xml exceeds the 64 MiB extraction limit",
                )
            document_root = ElementTree.fromstring(archive.read(info))
            title: Optional[str] = None
            try:
                core_root = ElementTree.fromstring(archive.read("docProps/core.xml"))
                for node in core_root.iter():
                    if node.tag.rsplit("}", 1)[-1] == "title" and node.text:
                        title = normalize_text(node.text) or None
                        break
            except (KeyError, ElementTree.ParseError):
                pass
    except CorpusExtractionError:
        raise
    except (OSError, KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        return [
            _RawFailure(
                "invalid_docx",
                f"DOCX package is unreadable: {type(exc).__name__}: {exc}",
            )
        ]

    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    body = document_root.find(f"{namespace}body")
    if body is None:
        return [_RawFailure("invalid_docx", "DOCX document.xml has no document body")]
    blocks: list[str] = []
    for child in body:
        local = child.tag.rsplit("}", 1)[-1]
        if local == "p":
            paragraph = normalize_text(_docx_paragraph_text(child, namespace))
            if paragraph:
                blocks.append(paragraph)
        elif local == "tbl":
            for row in child.findall(f".//{namespace}tr"):
                cells = [
                    normalize_text(_docx_paragraph_text(cell, namespace))
                    for cell in row.findall(f"./{namespace}tc")
                ]
                if any(cells):
                    blocks.append("\t".join(cells))
    text = normalize_text("\n\n".join(blocks))
    if not text:
        return [_RawFailure("docx_no_text", "the DOCX document contains no text")]
    return [
        _RawDocument(
            text,
            title=title,
            provenance={"docx_parts": ["word/document.xml"]},
        )
    ]


def _pypdf_pages(path: Path) -> Optional[tuple[list[str], Optional[str]]]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-not-found,no-redef]
        except ImportError:
            return None
    try:
        reader = PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            try:
                unlocked = reader.decrypt("")
            except Exception as exc:
                raise CorpusExtractionError(
                    "pdf_encrypted", f"PDF decryption failed: {type(exc).__name__}: {exc}"
                ) from exc
            if not unlocked:
                raise CorpusExtractionError("pdf_encrypted", "the PDF requires a password")
        pages = [normalize_text(page.extract_text() or "") for page in reader.pages]
        metadata = getattr(reader, "metadata", None)
        title = normalize_text(str(getattr(metadata, "title", "") or "")) or None
        return pages, title
    except CorpusExtractionError:
        raise
    except Exception as exc:
        raise CorpusExtractionError(
            "invalid_pdf", f"PDF parser failed: {type(exc).__name__}: {exc}"
        ) from exc


def _pdftotext_pages(path: Path) -> tuple[list[str], Optional[str]]:
    executable = shutil.which("pdftotext")
    if not executable:
        raise CorpusExtractionError(
            "pdf_extractor_unavailable",
            "text-layer PDF extraction requires pypdf or the pdftotext executable",
        )
    try:
        completed = subprocess.run(
            [executable, "-enc", "UTF-8", str(path), "-"],
            check=False,
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CorpusExtractionError(
            "pdf_extraction_failed",
            f"pdftotext failed: {type(exc).__name__}: {exc}",
        ) from exc
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise CorpusExtractionError(
            "invalid_pdf", f"pdftotext rejected the PDF: {message or completed.returncode}"
        )
    text = completed.stdout.decode("utf-8", errors="strict")
    pages = [normalize_text(value) for value in text.split("\f")]
    while pages and not pages[-1]:
        pages.pop()
    return pages, None


def _extract_pdf(path: Path, config: CorpusExtractionConfig) -> list[_RawDocument | _RawFailure]:
    loaded = _pypdf_pages(path)
    pages, title = loaded if loaded is not None else _pdftotext_pages(path)
    if not pages:
        return [
            _RawFailure(
                "pdf_no_text_layer",
                "the PDF exposes no extractable text layer",
                locator={"page_count": 0},
            )
        ]
    if not config.pdf_page_documents:
        text_pages = [text for text in pages if text]
        if not text_pages:
            return [
                _RawFailure(
                    "pdf_no_text_layer",
                    "the PDF exposes no extractable text layer",
                    locator={"page_count": len(pages)},
                )
            ]
        return [
            _RawDocument(
                "\n\n".join(text_pages),
                title=title,
                locator={"page_start": 1, "page_end": len(pages)},
                provenance={"pdf_page_count": len(pages), "page_documents": False},
            )
        ]
    values: list[_RawDocument | _RawFailure] = []
    for index, text in enumerate(pages, start=1):
        locator = {"page": index, "page_count": len(pages)}
        if text:
            values.append(
                _RawDocument(
                    text,
                    title=title,
                    locator=locator,
                    provenance={"page_documents": True},
                )
            )
        else:
            values.append(
                _RawFailure(
                    "pdf_page_no_text",
                    f"PDF page {index} exposes no extractable text layer",
                    locator=locator,
                )
            )
    return values


def _value_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        messages: list[str] = []
        is_conversation = True
        for item in value:
            if not isinstance(item, Mapping):
                is_conversation = False
                break
            role = item.get("role", item.get("from"))
            content = item.get("content", item.get("value"))
            if role is None or content is None:
                is_conversation = False
                break
            messages.append(f"{role}: {_value_text(content)}")
        if is_conversation:
            return normalize_text("\n".join(messages))
        return normalize_text("\n".join(_value_text(item) for item in value))
    if isinstance(value, Mapping):
        for key in ("text", "content", "value"):
            if key in value:
                return _value_text(value[key])
        return canonical_json(value)
    return normalize_text(str(value))


def _text_columns(record: Mapping[str, Any], config: CorpusExtractionConfig) -> tuple[str, ...]:
    if config.text_columns:
        return config.text_columns
    lower_to_key = {str(key).lower(): str(key) for key in record}
    preferred = [lower_to_key[name] for name in _AUTO_TEXT_COLUMNS if name in lower_to_key]
    if preferred:
        return tuple(dict.fromkeys(preferred))
    inferred = [
        str(key)
        for key, value in sorted(record.items(), key=lambda item: str(item[0]))
        if str(key).lower() not in _NON_TEXT_COLUMN_NAMES
        and isinstance(value, (str, list, tuple))
        and _value_text(value)
    ]
    return tuple(inferred)


def _structured_records(
    path: Path, config: CorpusExtractionConfig
) -> Iterator[_RawDocument | _RawFailure]:
    row_index = 0
    try:
        iterator = iter_file_records(path)
        for record, issue in iterator:
            if issue is not None:
                locator = {
                    key: issue[key] for key in ("line", "index") if issue.get(key) is not None
                }
                yield _RawFailure(
                    str(issue.get("code") or "structured_parse_error"),
                    str(issue.get("message") or "structured source row could not be parsed"),
                    locator=locator,
                    metadata={"source_issue": dict(issue)},
                )
                continue
            assert record is not None
            locator: Dict[str, Any] = {"row": row_index}
            external_id = None
            if config.id_column:
                external_id = get_field(record, config.id_column)
                if external_id is not None:
                    locator["external_id"] = str(external_id)
            columns = _text_columns(record, config)
            text_values = [_value_text(get_field(record, column)) for column in columns]
            text_values = [value for value in text_values if value]
            if not text_values:
                yield _RawFailure(
                    "structured_row_no_text",
                    "the structured row has no non-empty selected text column",
                    locator=locator,
                    metadata={"text_columns": list(columns)},
                )
                row_index += 1
                continue
            title_value = get_field(record, config.title_column) if config.title_column else None
            selected_metadata = {
                column: get_field(record, column)
                for column in config.metadata_columns
                if get_field(record, column) is not None
            }
            timestamp = next(
                (
                    record[key]
                    for key in ("timestamp", "created_at", "published_at", "date")
                    if key in record and record[key] is not None
                ),
                None,
            )
            document_metadata: Dict[str, Any] = {
                "text_columns": list(columns),
                "selected_metadata": selected_metadata,
            }
            if external_id is not None:
                document_metadata["source_ref"] = str(external_id)
            if timestamp is not None:
                document_metadata["timestamp"] = str(timestamp)
            yield _RawDocument(
                config.join_separator.join(text_values),
                title=_value_text(title_value) or None,
                locator=locator,
                metadata=document_metadata,
                provenance={"structured_format": path.suffix.lower().lstrip(".")},
            )
            row_index += 1
    except CorpusExtractionError:
        raise
    except Exception as exc:
        yield _RawFailure(
            "structured_source_error",
            f"structured source could not be read: {type(exc).__name__}: {exc}",
            locator={"row": row_index},
        )


def _extract_file(
    path: Path,
    config: CorpusExtractionConfig,
) -> tuple[str, str, Iterable[_RawDocument | _RawFailure]]:
    suffix = path.suffix.lower()
    if suffix in _PLAIN_TEXT_EXTENSIONS | _MARKDOWN_EXTENSIONS:
        text, encoding = _decode_text(path)
        source_kind = "markdown" if suffix in _MARKDOWN_EXTENSIONS else "text"
        media_type = "text/markdown" if source_kind == "markdown" else "text/plain"
        title = None
        if source_kind == "markdown":
            match = re.search(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", text)
            if match:
                title = normalize_text(match.group(1)) or None
        return (
            source_kind,
            media_type,
            [
                _RawDocument(
                    text,
                    title=title or path.stem,
                    metadata={"encoding": encoding, "markdown_mode": "source"},
                )
            ],
        )
    if suffix in _HTML_EXTENSIONS:
        return "html", "text/html", _extract_html(path)
    if suffix == ".pdf":
        return "pdf", "application/pdf", _extract_pdf(path, config)
    if suffix == ".docx":
        return (
            "docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            _extract_docx(path),
        )
    if suffix in _STRUCTURED_EXTENSIONS:
        media_type = {
            ".csv": "text/csv",
            ".tsv": "text/tab-separated-values",
            ".json": "application/json",
            ".jsonl": "application/x-ndjson",
            ".jl": "application/x-ndjson",
            ".parquet": "application/vnd.apache.parquet",
        }[suffix]
        return "structured", media_type, _structured_records(path, config)
    return (
        "unsupported",
        "application/octet-stream",
        [
            _RawFailure(
                "unsupported_document_format",
                f"unsupported corpus document format: {suffix or '<none>'}",
            )
        ],
    )


def collect_documents(
    path: Path | str,
    *,
    config: CorpusExtractionConfig | Mapping[str, Any] | None = None,
    expected_source_fingerprint: Optional[str] = None,
    source_uri: Optional[str] = None,
    check_cancelled: Optional[CancelCallback] = None,
    progress: Optional[ProgressCallback] = None,
) -> ExtractionCollection:
    """Extract canonical documents while quarantining individual failures."""

    cancel = check_cancelled or (lambda: None)
    report = progress or (lambda _processed, _total: None)
    resolved_config = CorpusExtractionConfig.from_value(config)
    selected = Path(path).expanduser()
    if selected.is_symlink():
        raise CorpusExtractionError(
            "unsafe_symbolic_link", "symbolic-link corpus sources are not accepted"
        )
    source = selected.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not source.is_file() and not source.is_dir():
        raise CorpusExtractionError(
            "unsupported_source_type", "corpus source must be a file or directory"
        )
    cancel()
    fingerprint, size_bytes, file_count = fingerprint_path(source)
    if expected_source_fingerprint is not None and fingerprint != expected_source_fingerprint:
        raise CorpusExtractionError(
            "source_changed",
            "corpus source content changed after the extraction was scheduled",
        )
    files, skipped_files = _source_files(source, resolved_config)
    root_uri = str(source_uri or source)
    documents: list[CorpusDocument] = []
    failures: list[ExtractionFailure] = []
    format_statistics: Dict[str, Dict[str, int]] = {}
    total = max(1, len(files))
    processed = 0
    if not files:
        failures.append(
            ExtractionFailure.build(
                source_uri=root_uri,
                source_kind="source",
                media_type="application/octet-stream",
                source_fingerprint=fingerprint,
                relative_path="",
                ordinal=0,
                error_code="no_supported_documents",
                error="the corpus source contains no supported document files",
                locator={"root": True},
            )
        )
        report(total, total)
    for file_path in files:
        cancel()
        relative_path = (
            file_path.name if source.is_file() else file_path.relative_to(source).as_posix()
        )
        file_uri = str(file_path)
        try:
            source_kind, media_type, raw_values = _extract_file(file_path, resolved_config)
            for raw in raw_values:
                cancel()
                bucket = format_statistics.setdefault(
                    source_kind, {"documents": 0, "quarantined": 0}
                )
                if isinstance(raw, _RawFailure):
                    failure = ExtractionFailure.build(
                        source_uri=file_uri,
                        source_kind=source_kind,
                        media_type=media_type,
                        source_fingerprint=fingerprint,
                        relative_path=relative_path,
                        ordinal=len(failures),
                        error_code=raw.code,
                        error=raw.error.replace(str(source), "<source>"),
                        locator=raw.locator,
                        provenance=raw.provenance,
                        metadata=raw.metadata,
                    )
                    failures.append(failure)
                    bucket["quarantined"] += 1
                    continue
                normalized = normalize_text(raw.text)
                if len(normalized) < resolved_config.min_text_chars:
                    failure = ExtractionFailure.build(
                        source_uri=file_uri,
                        source_kind=source_kind,
                        media_type=media_type,
                        source_fingerprint=fingerprint,
                        relative_path=relative_path,
                        ordinal=len(failures),
                        error_code="document_text_too_short",
                        error=(
                            "extracted text is shorter than "
                            f"{resolved_config.min_text_chars} character(s)"
                        ),
                        locator=raw.locator,
                        provenance=raw.provenance,
                        metadata=raw.metadata,
                    )
                    failures.append(failure)
                    bucket["quarantined"] += 1
                    continue
                document = CorpusDocument.build(
                    text=normalized,
                    source_uri=file_uri,
                    source_kind=source_kind,
                    media_type=media_type,
                    source_fingerprint=fingerprint,
                    ordinal=len(documents),
                    relative_path=relative_path,
                    title=raw.title,
                    locator=raw.locator,
                    provenance=raw.provenance,
                    metadata=raw.metadata,
                )
                documents.append(document)
                bucket["documents"] += 1
        except CorpusExtractionCancelled:
            raise
        except CorpusExtractionError as exc:
            source_kind = file_path.suffix.lower().lstrip(".") or "document"
            media_type = "application/octet-stream"
            failures.append(
                ExtractionFailure.build(
                    source_uri=file_uri,
                    source_kind=source_kind,
                    media_type=media_type,
                    source_fingerprint=fingerprint,
                    relative_path=relative_path,
                    ordinal=len(failures),
                    error_code=exc.code,
                    error=str(exc).replace(str(source), "<source>"),
                )
            )
            bucket = format_statistics.setdefault(source_kind, {"documents": 0, "quarantined": 0})
            bucket["quarantined"] += 1
        except Exception as exc:
            source_kind = file_path.suffix.lower().lstrip(".") or "document"
            failures.append(
                ExtractionFailure.build(
                    source_uri=file_uri,
                    source_kind=source_kind,
                    media_type="application/octet-stream",
                    source_fingerprint=fingerprint,
                    relative_path=relative_path,
                    ordinal=len(failures),
                    error_code="document_extraction_failed",
                    error=(f"{type(exc).__name__}: {exc}").replace(str(source), "<source>"),
                )
            )
            bucket = format_statistics.setdefault(source_kind, {"documents": 0, "quarantined": 0})
            bucket["quarantined"] += 1
        processed += 1
        report(processed, total)
    cancel()
    statistics = {
        "document_count": len(documents),
        "quarantined_count": len(failures),
        "item_count": len(documents) + len(failures),
        "source_size_bytes": int(size_bytes),
        "source_file_count": int(file_count),
        "processed_file_count": len(files),
        "skipped_file_count": skipped_files,
        "extracted_text_bytes": sum(document.text_byte_count for document in documents),
        "formats": format_statistics,
    }
    provenance = {
        "extractor_version": CORPUS_EXTRACTOR_VERSION,
        "source_uri": root_uri,
        "source_path": str(source),
        "source_fingerprint": fingerprint,
        "config_hash": resolved_config.fingerprint,
        "config": resolved_config.to_dict(),
    }
    return ExtractionCollection(
        source_fingerprint=fingerprint,
        source_size_bytes=int(size_bytes),
        source_file_count=int(file_count),
        documents=tuple(documents),
        quarantine=tuple(failures),
        statistics=statistics,
        provenance=provenance,
    )


__all__ = [
    "CorpusExtractionCancelled",
    "CorpusExtractionError",
    "ExtractionCollection",
    "collect_documents",
]
