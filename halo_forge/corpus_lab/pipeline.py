"""Synchronous corpus extraction entrypoint shared by services and workers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .extractors import ExtractionCollection, collect_documents
from .models import (
    CORPUS_EXTRACTOR_VERSION,
    CorpusExtractionConfig,
    CorpusExtractionResult,
    default_extraction_id,
)
from .storage import CorpusBundleStore


def default_corpus_root() -> Path:
    return Path.home() / ".halo-forge" / "corpus"


def extract_source(
    path: Path | str,
    *,
    root: Path | str | None = None,
    config: CorpusExtractionConfig | Mapping[str, Any] | None = None,
    extraction_id: Optional[str] = None,
    source_kind: Optional[str] = None,
    source_uri: Optional[str] = None,
    source_fingerprint: Optional[str] = None,
    check_cancelled: Optional[Callable[[], None]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> CorpusExtractionResult:
    """Extract one source and atomically publish its canonical records.

    The function has no database dependency. Callers receive both immutable
    extraction metadata and the canonical ``CorpusDocument`` records, making
    it suitable for v9 inspection, direct tests, and claimed worker execution.
    """

    selected = Path(path).expanduser().resolve()
    resolved_config = CorpusExtractionConfig.from_value(config)
    collection: ExtractionCollection = collect_documents(
        selected,
        config=resolved_config,
        expected_source_fingerprint=source_fingerprint,
        source_uri=source_uri,
        check_cancelled=check_cancelled,
        progress=progress,
    )
    identifier = extraction_id or default_extraction_id(
        collection.source_fingerprint,
        resolved_config.fingerprint,
    )
    normalized_source_kind = (
        str(source_kind or ("file" if selected.is_file() else "directory")).strip().lower()
    )
    resolved_source_uri = str(source_uri or selected)
    store = CorpusBundleStore(root or default_corpus_root())
    bundle = store.publish(
        extraction_id=identifier,
        source_uri=resolved_source_uri,
        source_kind=normalized_source_kind,
        source_fingerprint=collection.source_fingerprint,
        extractor_version=CORPUS_EXTRACTOR_VERSION,
        config=resolved_config,
        documents=collection.documents,
        quarantine=collection.quarantine,
        statistics=collection.statistics,
        provenance=collection.provenance,
        check_cancelled=check_cancelled,
    )
    return CorpusExtractionResult(
        extraction_id=identifier,
        source_uri=resolved_source_uri,
        source_kind=normalized_source_kind,
        source_fingerprint=collection.source_fingerprint,
        extractor_version=CORPUS_EXTRACTOR_VERSION,
        config_hash=resolved_config.fingerprint,
        documents=collection.documents,
        quarantine=collection.quarantine,
        bundle=bundle,
        statistics=collection.statistics,
        provenance=collection.provenance,
    )


__all__ = ["default_corpus_root", "extract_source"]
