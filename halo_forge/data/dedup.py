"""Dataset deduplication (Track D2).

Two methods ship in v1:

  - **exact** — SHA-256 over a normalized version of the record's text
    field. Removes literal duplicates (identical content modulo
    whitespace + case). O(n).
  - **fuzzy** — MinHash + LSH over k-shingles. Removes near-duplicates
    above a configurable Jaccard threshold (default 0.85). Scales to
    millions of records via the LSH index. Requires the
    ``datasketch`` package, lazy-imported.

Order is preserved: when collisions are found, the first occurrence is
kept and subsequent duplicates are dropped. This matches the
``deduplicate`` semantics of NeMo Curator and the published recipes.

Semantic dedup (SemDeDup via sentence-embedding clustering) is roadmap
follow-up — it adds a heavyweight embedding-model dependency that
deserves its own opt-in story.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

from halo_forge.data.primitives import normalize_text as _normalize_text
from halo_forge.data.primitives import word_shingles as _shared_shingles


# ---------- normalization ----------------------------------------------------


def _shingles(text: str, *, n: int = 5) -> List[str]:
    """Word n-gram shingles for MinHash. n=5 is the LLM dedup standard
    (matches the FineWeb-Edu / NeMo Curator pipelines)."""
    return _shared_shingles(text, n=n)


# ---------- result shape ----------------------------------------------------


@dataclass
class DedupResult:
    """Outcome of a dedup run, returned to CLI / API consumers.

    Index lists are absolute positions in the input sequence, useful
    when the caller wants to reconstruct provenance. Both lists
    together account for every input record exactly once.
    """

    method: str
    threshold: Optional[float]  # None for exact
    n_input: int
    n_output: int
    n_removed: int
    kept_indices: List[int] = field(default_factory=list)
    removed_indices: List[int] = field(default_factory=list)
    duration_seconds: float = 0.0
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- core methods ----------------------------------------------------


def _extract_text(record: Any, key: str) -> str:
    """Get the text field from a record. Tolerates plain strings (treats
    them as the text), dict-shaped records (looks up `key`), and falls
    back to str(record) when neither shape matches."""
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        v = record.get(key)
        if isinstance(v, str):
            return v
        # Some preference datasets use list-of-messages; flatten to text.
        if isinstance(v, list):
            return " ".join(
                m.get("content", "") if isinstance(m, dict) else str(m) for m in v
            )
    return str(record)


def exact_dedup(
    records: Sequence[Any],
    *,
    key: str = "text",
    case_sensitive: bool = False,
) -> DedupResult:
    """SHA-256-keyed exact deduplication.

    Args:
        records: Iterable of strings or dict-shaped records.
        key: When records are dicts, the field to dedup on.
        case_sensitive: When False (default), case + whitespace
            collapse before hashing.
    """
    import time

    t0 = time.time()
    seen: dict[str, int] = {}
    kept: List[int] = []
    removed: List[int] = []

    for idx, record in enumerate(records):
        text = _extract_text(record, key)
        normalized = _normalize_text(text, case_sensitive=case_sensitive)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            removed.append(idx)
        else:
            seen[digest] = idx
            kept.append(idx)

    return DedupResult(
        method="exact",
        threshold=None,
        n_input=len(records),
        n_output=len(kept),
        n_removed=len(removed),
        kept_indices=kept,
        removed_indices=removed,
        duration_seconds=time.time() - t0,
    )


def fuzzy_dedup(
    records: Sequence[Any],
    *,
    threshold: float = 0.85,
    num_perm: int = 128,
    shingle_n: int = 5,
    key: str = "text",
    case_sensitive: bool = False,
) -> DedupResult:
    """MinHash-LSH near-duplicate deduplication.

    Two records are considered duplicates if their MinHash-estimated
    Jaccard similarity over word n-gram shingles exceeds ``threshold``.
    The LSH index gives sub-quadratic scaling — well-suited to
    million-record datasets.

    Args:
        records: Iterable of strings or dict-shaped records.
        threshold: Jaccard similarity above which records are merged.
            0.85 is the published default for finetune-grade dedup;
            0.7 is more aggressive (closer to topic-clustering).
        num_perm: MinHash permutations. Higher = more precise estimate
            at higher memory cost. 128 is the standard.
        shingle_n: Word n-gram size. 5 matches FineWeb / NeMo Curator.
        key / case_sensitive: as in `exact_dedup`.

    Raises:
        ImportError if ``datasketch`` isn't installed.
    """
    import time

    # Validate args first so a stupid threshold doesn't make the user
    # install an unrelated dep before they see the real error.
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:
        raise ImportError(
            "fuzzy_dedup requires `datasketch`. Install with `pip install datasketch`."
        ) from exc

    t0 = time.time()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    # Signatures are deliberately not accumulated: the LSH index keeps the
    # band data it needs for kept rows, and holding every MinHash alive costs
    # ~1 KB/record — a gigabyte at the million-record scale advertised above.
    kept: List[int] = []
    removed: List[int] = []

    for idx, record in enumerate(records):
        text = _extract_text(record, key)
        normalized = _normalize_text(text, case_sensitive=case_sensitive)
        m = MinHash(num_perm=num_perm)
        for shingle in _shingles(normalized, n=shingle_n):
            m.update(shingle.encode("utf-8"))

        # Query first; if any candidate is above threshold, drop this row.
        if lsh.query(m):
            removed.append(idx)
        else:
            lsh.insert(str(idx), m)
            kept.append(idx)

    return DedupResult(
        method="fuzzy",
        threshold=threshold,
        n_input=len(records),
        n_output=len(kept),
        n_removed=len(removed),
        kept_indices=kept,
        removed_indices=removed,
        duration_seconds=time.time() - t0,
    )


def dedup(
    records: Sequence[Any],
    *,
    method: str = "exact",
    threshold: float = 0.85,
    key: str = "text",
    case_sensitive: bool = False,
    num_perm: int = 128,
    shingle_n: int = 5,
) -> DedupResult:
    """Dispatch entry point. ``method`` ∈ {"exact", "fuzzy"}."""
    method = method.strip().lower()
    if method == "exact":
        return exact_dedup(records, key=key, case_sensitive=case_sensitive)
    if method == "fuzzy":
        return fuzzy_dedup(
            records,
            threshold=threshold,
            num_perm=num_perm,
            shingle_n=shingle_n,
            key=key,
            case_sensitive=case_sensitive,
        )
    raise ValueError(
        f"Unknown dedup method {method!r}. Choose: exact, fuzzy."
    )


# ---------- file IO helpers --------------------------------------------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    records: List[Dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    """Write records to JSONL; returns count written. Strings are written
    as ``{"text": s}`` so the output schema stays uniform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for record in records:
            if isinstance(record, str):
                row = {"text": record}
            elif isinstance(record, dict):
                row = record
            else:
                row = {"text": str(record)}
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def dedup_file(
    *,
    input_path: Path,
    output_path: Path,
    method: str = "exact",
    threshold: float = 0.85,
    key: str = "text",
    case_sensitive: bool = False,
) -> DedupResult:
    """End-to-end: load JSONL → dedup → write surviving rows."""
    records = load_jsonl(input_path)
    result = dedup(
        records,
        method=method,
        threshold=threshold,
        key=key,
        case_sensitive=case_sensitive,
    )
    survivors = [records[i] for i in result.kept_indices]
    write_jsonl(output_path, survivors)
    return result


__all__ = [
    "DedupResult",
    "dedup",
    "exact_dedup",
    "fuzzy_dedup",
    "dedup_file",
    "load_jsonl",
    "write_jsonl",
]
