"""Canonical JSON identities for adaptive research records."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Dict, Tuple


def _freeze(value: Any) -> Any:
    if isinstance(value, FrozenJsonMap):
        return value
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical values must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        return FrozenJsonMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    raise TypeError(f"{type(value).__name__} is not a supported JSON value")


def thaw_json(value: Any) -> Any:
    if isinstance(value, FrozenJsonMap):
        return value.to_dict()
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


class FrozenJsonMap(Mapping[str, Any]):
    """A recursively immutable JSON mapping with deterministic key order."""

    __slots__ = ("_items", "_lookup")

    def __init__(self, value: Mapping[str, Any] | None = None) -> None:
        source = value or {}
        if not isinstance(source, Mapping):
            raise TypeError("value must be a mapping")
        normalized: Dict[str, Any] = {}
        for key, item in source.items():
            if not isinstance(key, str):
                raise TypeError("canonical mapping keys must be strings")
            normalized[key] = _freeze(item)
        self._items: Tuple[Tuple[str, Any], ...] = tuple(sorted(normalized.items()))
        self._lookup = dict(self._items)

    def __getitem__(self, key: str) -> Any:
        return self._lookup[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def to_dict(self) -> Dict[str, Any]:
        return {key: thaw_json(value) for key, value in self._items}


def canonical_json(value: Any) -> str:
    return json.dumps(
        thaw_json(_freeze(value)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


__all__ = ["FrozenJsonMap", "canonical_json", "content_fingerprint", "thaw_json"]
