"""Small, dependency-free transformation primitives shared with Dataset Lab.

The legacy file commands keep their existing result shapes and flags while
the catalog recipe runner composes the same normalization and similarity
operations into immutable builds.
"""

from __future__ import annotations

import re
from typing import List, Set

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str, *, case_sensitive: bool = False) -> str:
    value = _WS_RE.sub(" ", text or "").strip()
    return value if case_sensitive else value.lower()


def word_shingles(text: str, *, n: int = 5) -> List[str]:
    tokens = text.split()
    if len(tokens) <= n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def jaccard_text(left: str, right: str, *, n: int = 3) -> float:
    a: Set[str] = set(word_shingles(normalize_text(left), n=n))
    b: Set[str] = set(word_shingles(normalize_text(right), n=n))
    return len(a & b) / len(a | b) if a or b else 1.0


__all__ = ["jaccard_text", "normalize_text", "word_shingles"]
