"""Boundary-aware facade over append-only :class:`TrainingSignalSink` shards."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .registry import TrainingSignalCapabilityDescriptor
from .sink import (
    TrainingSignalShard,
    TrainingSignalSink,
    build_training_runtime_identity,
)


def default_audit_boundaries(total: int, *, maximum: int = 4) -> list[str]:
    """Return deterministic, evenly spaced boundaries including first/final."""

    count = max(1, int(total))
    width = max(1, min(int(maximum), count))
    if width == 1:
        return ["final"]
    values = {
        1 + round(index * (count - 1) / (width - 1))
        for index in range(width)
    }
    ordered = [str(value) for value in sorted(values) if value != count]
    ordered.append("final")
    return ordered


class BoundarySignalSession:
    """Route observations into selected boundary shards without regeneration.

    Trainers continue to call the small ``capture`` sink protocol. The session
    reads the trainer-provided cycle identity, retains only selected boundary
    populations, and seals each shard after checkpoint publication when the
    trainer calls ``boundary_complete``. ``finalize`` safely seals any final
    tail for older third-party trainer integrations.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        trainer: str,
        capability: TrainingSignalCapabilityDescriptor,
        total_boundaries: int,
        boundaries: Sequence[str | int],
        protocol: str,
        reward_threshold: float,
        attempt_id: str = "attempt-1",
        record_resolver: Optional[Callable[[int], Any]] = None,
        producer_model_hash: Optional[str] = None,
        producer_model_identity: Optional[Mapping[str, Any]] = None,
        runtime_identity: Optional[Mapping[str, Any]] = None,
        on_sealed: Optional[Callable[[TrainingSignalShard, int], None]] = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.run_id = str(run_id)
        self.trainer = str(trainer)
        self.capability = capability
        self.total_boundaries = max(1, int(total_boundaries))
        self.protocol = str(protocol)
        self.reward_threshold = float(reward_threshold)
        self.attempt_id = str(attempt_id)
        self.record_resolver = record_resolver
        self.producer_model_hash = producer_model_hash
        self.producer_model_identity = dict(producer_model_identity or {})
        self.runtime_identity = dict(
            runtime_identity
            if runtime_identity is not None
            else build_training_runtime_identity(capability, trainer=self.trainer)
        )
        self.on_sealed = on_sealed
        self.selected = {
            self._boundary_number(value) for value in boundaries
        }
        self.selected.add(self.total_boundaries)
        self._sinks: dict[int, TrainingSignalSink] = {}
        self._sealed: dict[int, TrainingSignalShard] = {}

    def _boundary_number(self, value: str | int) -> int:
        raw = str(value).strip().lower()
        if raw in {"final", "last"}:
            return self.total_boundaries
        if ":" in raw:
            raw = raw.rsplit(":", 1)[-1]
        number = int(raw)
        if not 1 <= number <= self.total_boundaries:
            raise ValueError(
                f"audit boundary {value!r} is outside 1..{self.total_boundaries}"
            )
        return number

    def _source_boundary(self, values: Mapping[str, Any]) -> int:
        source = values.get("source")
        cycle = source.get("cycle") if isinstance(source, Mapping) else None
        if cycle is None:
            return self.total_boundaries
        number = int(cycle)
        # The general RAFT trainer numbers cycles from one; modality trainers
        # expose zero-based loop indices in their capture metadata.
        if self.trainer in {"vlm", "audio", "reasoning", "agentic"}:
            number += 1
        return max(1, min(self.total_boundaries, number))

    def _sink(self, boundary: int) -> TrainingSignalSink:
        sink = self._sinks.get(boundary)
        if sink is None:
            segment_id = f"boundary-{boundary}"
            sink = TrainingSignalSink(
                self.root / self.run_id / segment_id,
                run_id=self.run_id,
                segment_id=segment_id,
                boundary=str(boundary),
                capability=self.capability,
                protocol=self.protocol,
                reward_threshold=self.reward_threshold,
                attempt_id=self.attempt_id,
                record_resolver=self.record_resolver,
                producer_model_hash=self.producer_model_hash,
                producer_model_identity=self.producer_model_identity,
                runtime_identity=self.runtime_identity,
            )
            self._sinks[boundary] = sink
        return sink

    def capture(self, **values: Any) -> Any:
        boundary = self._source_boundary(values)
        if boundary not in self.selected:
            return None
        return self._sink(boundary).capture(**values)

    def boundary_complete(
        self, *, boundary_value: str | int, checkpoint_hash: Optional[str] = None
    ) -> Optional[TrainingSignalShard]:
        raw = str(boundary_value).strip().lower()
        if (
            raw not in {"final", "last"}
            and self.trainer in {"vlm", "audio", "reasoning", "agentic"}
        ):
            # These trainers publish checkpoints with their zero-based loop
            # index, while audit schedules and capture shards are one-based.
            boundary_value = int(raw) + 1
        boundary = self._boundary_number(boundary_value)
        if boundary not in self.selected:
            return None
        # An audited boundary with zero produced candidates is still evidence:
        # seal an empty shard so the integrity decision becomes
        # ``incomplete_evidence`` instead of silently omitting the gate.
        sink = self._sink(boundary)
        shard = sink.seal(checkpoint_hash=checkpoint_hash)
        self._sealed[boundary] = shard
        if self.on_sealed is not None:
            self.on_sealed(shard, boundary)
        return shard

    def finalize(self, *, checkpoint_hash: Optional[str] = None) -> list[TrainingSignalShard]:
        # Shipped trainers call ``boundary_complete`` after publication.  This
        # fallback also makes third-party integrations truthful: missed or
        # empty selected boundaries publish zero-count shards rather than
        # disappearing from the audit schedule.
        for boundary in sorted(self.selected):
            self._sink(boundary)
        for boundary in sorted(self._sinks):
            if boundary in self._sealed:
                continue
            shard = self._sinks[boundary].seal(checkpoint_hash=checkpoint_hash)
            self._sealed[boundary] = shard
            if self.on_sealed is not None:
                self.on_sealed(shard, boundary)
        return [self._sealed[key] for key in sorted(self._sealed)]

    @property
    def sealed_shards(self) -> tuple[TrainingSignalShard, ...]:
        return tuple(self._sealed[key] for key in sorted(self._sealed))


__all__ = ["BoundarySignalSession", "default_audit_boundaries"]
