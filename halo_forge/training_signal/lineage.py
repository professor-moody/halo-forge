"""Bounded, row-aligned Dataset Lab lineage resolution for signal capture."""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any, Mapping, Optional

from .models import TrainingRecordRef


class TrainingArtifactRecordResolver:
    """Resolve a v3 artifact row to its immutable Dataset Lab identity.

    The resolver uses sparse byte offsets rather than materializing the full
    lineage sidecar.  This keeps managed runs bounded in memory even when a
    trainer revisits rows out of order.  A one-row cache makes the common
    multiple-candidates-per-source pattern constant time.
    """

    def __init__(
        self,
        artifact: Any,
        *,
        role: str = "train",
        offset_stride: int = 1024,
    ) -> None:
        normalized = str(role).strip().lower()
        format_version = int(getattr(artifact, "format_version", 0) or 0)
        if format_version < 3:
            raise ValueError("row-aligned signal identity requires a format v3 artifact")
        paths = dict(getattr(artifact, "lineage_paths", {}) or {})
        raw_path = paths.get(normalized)
        if not raw_path:
            raise ValueError(
                f"format v3 training artifact has no lineage index for role {normalized!r}"
            )
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"training artifact lineage index is missing: {path}")

        counts = dict(getattr(artifact, "row_counts", {}) or {})
        self.expected_count = int(counts.get(normalized, 0) or 0)
        self.path = path
        self.role = normalized
        self.artifact_id = str(getattr(artifact, "artifact_id", "") or "")
        self.artifact_hash = str(getattr(artifact, "artifact_hash", "") or "")
        self.offset_stride = max(1, int(offset_stride))
        self._offsets: dict[int, int] = {0: 0}
        self._cached_index: Optional[int] = None
        self._cached_value: Optional[TrainingRecordRef] = None
        self._lock = threading.RLock()

    def __call__(self, source_index: int) -> TrainingRecordRef:
        index = int(source_index)
        if index < 0 or index >= self.expected_count:
            raise IndexError(
                f"training artifact {self.artifact_id or self.path.parent.name} "
                f"role {self.role!r} has {self.expected_count} rows; index {index} is invalid"
            )
        with self._lock:
            if self._cached_index == index and self._cached_value is not None:
                return self._cached_value
            start = max(value for value in self._offsets if value <= index)
            with self.path.open("rb") as handle:
                handle.seek(self._offsets[start])
                raw: Optional[Mapping[str, Any]] = None
                for position in range(start, index + 1):
                    if position % self.offset_stride == 0:
                        self._offsets.setdefault(position, handle.tell())
                    line = handle.readline()
                    if not line:
                        raise ValueError(
                            f"training artifact lineage ended before row {index}"
                        )
                    if position == index:
                        try:
                            decoded = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"invalid training artifact lineage row {index}: {exc}"
                            ) from exc
                        if not isinstance(decoded, Mapping):
                            raise ValueError(
                                f"training artifact lineage row {index} must be an object"
                            )
                        raw = decoded
            assert raw is not None
            missing = [
                name
                for name in ("record_id", "record_hash", "instance_id")
                if not raw.get(name)
            ]
            if missing:
                raise ValueError(
                    "training artifact lineage row is missing identity fields: "
                    + ", ".join(missing)
                )
            lineage = copy.deepcopy(dict(raw))
            canonical_record_index = lineage.get("record_index")
            source = {
                "kind": "training_dataset_artifact",
                "artifact_id": self.artifact_id,
                "artifact_hash": self.artifact_hash,
                "role": self.role,
                "record_index": canonical_record_index,
                "lineage": {
                    key: copy.deepcopy(value)
                    for key, value in lineage.items()
                    if key not in {"record_id", "record_hash", "instance_id"}
                },
            }
            # ``source_index`` is the trainer-visible row ordinal. Preserve the
            # canonical version's record_index separately in source lineage.
            value = {
                **lineage,
                "source_index": index,
                "source": source,
            }
            resolved = TrainingRecordRef.from_value(value, source_index=index, source=source)
            self._cached_index = index
            self._cached_value = resolved
            return resolved


def record_resolver_from_training_artifact(
    artifact: Any,
    *,
    role: str = "train",
) -> Optional[TrainingArtifactRecordResolver]:
    """Return a real-identity resolver for v3, or ``None`` for legacy data.

    Legacy artifacts and manual paths intentionally use the signal model's
    explicit deterministic virtual identities.  A malformed v3 artifact is an
    error rather than a silent downgrade to virtual identity.
    """

    if int(getattr(artifact, "format_version", 0) or 0) < 3:
        return None
    return TrainingArtifactRecordResolver(artifact, role=role)


__all__ = [
    "TrainingArtifactRecordResolver",
    "record_resolver_from_training_artifact",
]
