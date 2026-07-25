"""Disk-backed, deterministic training-signal selection and shard sealing."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from .models import TrainingSignalSnapshot, canonical_json, content_hash
from .registry import CaptureFidelity, TrainingSignalCapabilityDescriptor


PROTOCOLS: Dict[str, Dict[str, Optional[int]]] = {
    "balanced_256": {"limit": 256, "core": 192, "diagnostic_each": 16},
    "broad_512": {"limit": 512, "core": 384, "diagnostic_each": 32},
    "exhaustive": {"limit": None, "core": None, "diagnostic_each": None},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_training_runtime_identity(
    capability: TrainingSignalCapabilityDescriptor,
    *,
    trainer: Optional[str] = None,
) -> Dict[str, Any]:
    """Capture one sanitized runtime fingerprint for a signal sink/session."""

    from halo_forge.replay import EnvironmentFingerprint
    from halo_forge.verifier_lab.store import scrub_secrets

    value = EnvironmentFingerprint.capture().to_dict()
    value["trainer"] = str(trainer or capability.trainer)
    value["training_signal_capability"] = {
        "id": capability.id,
        "version": capability.version,
        "backend": capability.backend,
    }
    clean = scrub_secrets(value)
    clean["fingerprint"] = content_hash(clean)
    return clean


class _JsonSequenceHash:
    """Incrementally hash a canonical JSON array without retaining its values."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(b"[")
        self._first = True

    def add(self, value: Any) -> None:
        if not self._first:
            self._digest.update(b",")
        self._digest.update(canonical_json(value).encode("utf-8"))
        self._first = False

    def hexdigest(self) -> str:
        digest = self._digest.copy()
        digest.update(b"]")
        return digest.hexdigest()


@dataclass(frozen=True)
class TrainingSignalShard:
    shard_id: str
    trace_hash: str
    path: str
    run_id: str
    segment_id: str
    boundary: str
    capability_id: str
    capability_version: str
    capture_fidelity: str
    protocol: str
    observed_count: int
    retained_count: int
    core_count: int
    diagnostic_count: int
    aggregate: Dict[str, Any]
    checkpoint_hash: Optional[str]
    created_at: str
    reused: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainingSignalSink:
    """Attempt-scoped append sink with bounded-memory deterministic sealing.

    The temporary SQLite spool is intentionally not a published artifact.  It
    provides idempotent event insertion and lets selection operate over very
    large rollouts without retaining every model output in process memory.
    Successful sealing writes only the selected snapshot set and aggregate
    telemetry, then removes the spool.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        run_id: str,
        segment_id: str,
        boundary: str,
        capability: TrainingSignalCapabilityDescriptor,
        protocol: str = "balanced_256",
        reward_threshold: float = 0.5,
        attempt_id: str = "attempt-1",
        record_resolver: Optional[Callable[[int], Any]] = None,
        producer_model_hash: Optional[str] = None,
        producer_model_identity: Optional[Mapping[str, Any]] = None,
        runtime_identity: Optional[Mapping[str, Any]] = None,
        commit_interval: int = 128,
    ) -> None:
        if protocol not in PROTOCOLS:
            raise ValueError(f"unknown training signal protocol: {protocol}")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.segment_id = str(segment_id)
        self.boundary = str(boundary)
        self.capability = capability
        self.protocol = protocol
        self.reward_threshold = float(reward_threshold)
        self.attempt_id = str(attempt_id)
        self.record_resolver = record_resolver
        self.producer_model_hash = producer_model_hash
        self.producer_model_identity = copy.deepcopy(
            dict(producer_model_identity or {})
        )
        self.runtime_identity = copy.deepcopy(
            dict(runtime_identity)
            if runtime_identity is not None
            else build_training_runtime_identity(capability)
        )
        self.commit_interval = max(1, int(commit_interval))
        self._pending_writes = 0
        self._sealed: Optional[TrainingSignalShard] = None
        spool_name = "." + content_hash(
            {
                "run_id": self.run_id,
                "segment_id": self.segment_id,
                "boundary": self.boundary,
                "attempt_id": self.attempt_id,
            }
        )[:24] + ".signal-spool.sqlite3"
        self._spool_path = self.root / spool_name
        self._connection = sqlite3.connect(self._spool_path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                snapshot_id TEXT PRIMARY KEY,
                record_id TEXT NOT NULL,
                hash_rank TEXT NOT NULL,
                reward REAL,
                passed INTEGER,
                has_error INTEGER NOT NULL,
                threshold_distance REAL,
                component_disagreement REAL NOT NULL,
                payload TEXT
            )
            """
        )
        self._connection.commit()

    @property
    def enabled(self) -> bool:
        return self.capability.fidelity != CaptureFidelity.UNAVAILABLE

    def observe(self, snapshot: TrainingSignalSnapshot) -> bool:
        if self._sealed is not None:
            raise RuntimeError("cannot append to a sealed training signal shard")
        if not self.enabled:
            return False
        if (
            snapshot.run_id != self.run_id
            or snapshot.segment_id != self.segment_id
            or snapshot.boundary != self.boundary
        ):
            raise ValueError("training signal snapshot does not belong to this sink")
        observation = snapshot.training_observation
        component_rewards = [
            float(item["reward"])
            for item in observation.component_trace
            if item.get("reward") is not None
        ]
        disagreement = (
            max(component_rewards) - min(component_rewards) if len(component_rewards) >= 2 else 0.0
        )
        rank = content_hash(
            {
                "protocol": self.protocol,
                "run_id": self.run_id,
                "segment_id": self.segment_id,
                "boundary": self.boundary,
                "snapshot_id": snapshot.snapshot_id,
            }
        )
        payload = None
        if self.capability.fidelity not in {
            CaptureFidelity.AGGREGATE_ONLY,
            CaptureFidelity.UNAVAILABLE,
        }:
            payload = canonical_json(snapshot.to_dict())
        cursor = self._connection.execute(
            """
            INSERT OR IGNORE INTO observations
            (snapshot_id,record_id,hash_rank,reward,passed,has_error,
             threshold_distance,component_disagreement,payload)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                snapshot.snapshot_id,
                snapshot.record.record_id,
                rank,
                observation.reward,
                None if observation.passed is None else int(observation.passed),
                int(bool(observation.error)),
                (
                    abs(float(observation.reward) - self.reward_threshold)
                    if observation.reward is not None
                    else None
                ),
                disagreement,
                payload,
            ),
        )
        self._pending_writes += int(cursor.rowcount == 1)
        if self._pending_writes >= self.commit_interval:
            self._connection.commit()
            self._pending_writes = 0
        return cursor.rowcount == 1

    def capture(self, **values: Any) -> TrainingSignalSnapshot:
        values.setdefault("run_id", self.run_id)
        values.setdefault("segment_id", self.segment_id)
        values.setdefault("boundary", self.boundary)
        if values.get("record") is None and self.record_resolver is not None:
            values["record"] = self.record_resolver(int(values.get("source_index", 0)))
        if self.producer_model_hash is not None:
            # A session-level identity was resolved before training began and
            # is more trustworthy than trainer-local legacy name/path values.
            values["producer_model_hash"] = self.producer_model_hash
            values["producer_model_identity"] = self.producer_model_identity
        values.setdefault("runtime_identity", self.runtime_identity)
        snapshot = TrainingSignalSnapshot.create(**values)
        self.observe(snapshot)
        return snapshot

    def _aggregate(self) -> Dict[str, Any]:
        self._connection.commit()
        self._pending_writes = 0
        row = self._connection.execute(
            """
            SELECT COUNT(*), COUNT(reward), SUM(reward), MIN(reward), MAX(reward),
                   SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN passed = 0 THEN 1 ELSE 0 END),
                   SUM(has_error)
            FROM observations
            """
        ).fetchone()
        assert row is not None
        observed, reward_count, reward_sum, minimum, maximum, passed, failed, errors = row
        distribution = {}
        for name, lower, upper in (
            ("lt_0_2", None, 0.2),
            ("0_2_to_0_5", 0.2, 0.5),
            ("0_5_to_0_9", 0.5, 0.9),
            ("gte_0_9", 0.9, None),
        ):
            clauses = ["reward IS NOT NULL"]
            params = []
            if lower is not None:
                clauses.append("reward >= ?")
                params.append(lower)
            if upper is not None:
                clauses.append("reward < ?")
                params.append(upper)
            count = self._connection.execute(
                "SELECT COUNT(*) FROM observations WHERE " + " AND ".join(clauses), params
            ).fetchone()[0]
            distribution[name] = int(count)
        return {
            "observed_count": int(observed or 0),
            "reward_count": int(reward_count or 0),
            "reward_mean": (
                float(reward_sum) / int(reward_count) if reward_count else None
            ),
            "reward_min": float(minimum) if minimum is not None else None,
            "reward_max": float(maximum) if maximum is not None else None,
            "passed_count": int(passed or 0),
            "failed_count": int(failed or 0),
            "error_count": int(errors or 0),
            "reward_distribution": distribution,
        }

    def _selected_rows(self) -> Iterator[tuple[str, str]]:
        total = self._connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        spec = PROTOCOLS[self.protocol]
        limit = spec["limit"]
        if limit is None or total <= limit:
            for snapshot_id, payload in self._connection.execute(
                "SELECT snapshot_id,payload FROM observations ORDER BY hash_rank,snapshot_id"
            ):
                if payload is not None:
                    yield "exact", payload
            return
        if self.capability.fidelity == CaptureFidelity.AGGREGATE_ONLY:
            return

        selected: set[str] = set()

        def take(stratum: str, query: str, count: int) -> Iterator[tuple[str, str]]:
            emitted = 0
            for snapshot_id, payload in self._connection.execute(query):
                if emitted >= count:
                    break
                if snapshot_id in selected or payload is None:
                    continue
                selected.add(snapshot_id)
                emitted += 1
                yield stratum, payload

        core_count = int(spec["core"] or 0)
        diagnostic_each = int(spec["diagnostic_each"] or 0)
        # Select diagnostic strata first so their observations do not leak into
        # population-rate estimates through the uniform core.
        yield from take(
            "verifier_error",
            "SELECT snapshot_id,payload FROM observations WHERE has_error=1 "
            "ORDER BY hash_rank,snapshot_id",
            diagnostic_each,
        )
        yield from take(
            "threshold_adjacent",
            "SELECT snapshot_id,payload FROM observations WHERE threshold_distance IS NOT NULL "
            "ORDER BY threshold_distance,hash_rank,snapshot_id",
            diagnostic_each,
        )
        yield from take(
            "highest_reward",
            "SELECT snapshot_id,payload FROM observations WHERE reward IS NOT NULL "
            "ORDER BY reward DESC,hash_rank,snapshot_id",
            diagnostic_each,
        )
        yield from take(
            "component_disagreement",
            "SELECT snapshot_id,payload FROM observations WHERE component_disagreement>0 "
            "ORDER BY component_disagreement DESC,hash_rank,snapshot_id",
            diagnostic_each,
        )
        yield from take(
            "uniform_core",
            "SELECT snapshot_id,payload FROM observations ORDER BY hash_rank,snapshot_id",
            core_count,
        )
        remaining = int(limit) - len(selected)
        if remaining > 0:
            # Sparse diagnostic strata should not shrink the promised bounded
            # sample. Fill deterministically from the uniform population.
            query = (
                "SELECT snapshot_id,payload FROM observations ORDER BY hash_rank,snapshot_id"
            )
            for snapshot_id, payload in self._connection.execute(query):
                if remaining <= 0:
                    break
                if snapshot_id in selected or payload is None:
                    continue
                selected.add(snapshot_id)
                remaining -= 1
                yield "uniform_fill", payload

    def seal(self, *, checkpoint_hash: Optional[str] = None) -> TrainingSignalShard:
        if self._sealed is not None:
            return self._sealed
        aggregate = self._aggregate()
        if self.protocol == "exhaustive":
            return self._seal_exhaustive(aggregate, checkpoint_hash=checkpoint_hash)
        selected = list(self._selected_rows())
        core_count = sum(1 for stratum, _ in selected if stratum in {"exact", "uniform_core", "uniform_fill"})
        diagnostic_count = len(selected) - core_count
        observed = aggregate["observed_count"]
        if self.capability.fidelity == CaptureFidelity.UNAVAILABLE:
            fidelity = CaptureFidelity.UNAVAILABLE.value
        elif self.capability.fidelity == CaptureFidelity.AGGREGATE_ONLY:
            fidelity = CaptureFidelity.AGGREGATE_ONLY.value
        elif observed == len(selected):
            fidelity = CaptureFidelity.EXACT.value
        else:
            fidelity = CaptureFidelity.SAMPLED.value
        identity = {
            "format_version": 1,
            "run_id": self.run_id,
            "segment_id": self.segment_id,
            "boundary": self.boundary,
            "capability_id": self.capability.id,
            "capability_version": self.capability.version,
            "capture_fidelity": fidelity,
            "protocol": self.protocol,
            "reward_threshold": self.reward_threshold,
            "observed_count": observed,
            "retained_ids": [json.loads(payload)["snapshot_id"] for _, payload in selected],
            "aggregate": aggregate,
            "checkpoint_hash": checkpoint_hash,
        }
        trace_hash = content_hash(identity)
        shard_id = "trace_" + trace_hash[:24]
        final_path = self.root / shard_id
        if final_path.exists():
            verification = verify_training_signal_shard(final_path)
            if not verification["valid"]:
                raise RuntimeError(
                    "existing training signal shard failed integrity verification: "
                    + "; ".join(verification["problems"])
                )
            manifest = json.loads((final_path / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("trace_hash") != trace_hash:
                raise RuntimeError(f"training signal content-address collision: {final_path}")
            shard = self._shard_from_manifest(final_path, manifest, reused=True)
            self._finish_spool()
            self._sealed = shard
            return shard

        stage = Path(tempfile.mkdtemp(prefix=f".{shard_id}.tmp-", dir=self.root))
        try:
            samples_path = stage / "samples.jsonl"
            with samples_path.open("w", encoding="utf-8") as handle:
                for stratum, payload in selected:
                    value = json.loads(payload)
                    value["selection_stratum"] = stratum
                    value["checkpoint_hash"] = checkpoint_hash
                    handle.write(canonical_json(value) + "\n")
            manifest = {
                **identity,
                "status": "complete",
                "shard_id": shard_id,
                "trace_hash": trace_hash,
                "retained_count": len(selected),
                "core_count": core_count,
                "diagnostic_count": diagnostic_count,
                "created_at": _now(),
                "files": {"samples.jsonl": _file_hash(samples_path)},
            }
            _write_json(stage / "manifest.json", manifest)
            os.replace(stage, final_path)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        shard = self._shard_from_manifest(final_path, manifest)
        self._finish_spool()
        self._sealed = shard
        return shard

    def _seal_exhaustive(
        self,
        aggregate: Mapping[str, Any],
        *,
        checkpoint_hash: Optional[str],
    ) -> TrainingSignalShard:
        """Seal an exhaustive trace without materializing snapshots or IDs.

        Trace manifest v2 replaces the potentially enormous ``retained_ids``
        array with the hash of the same canonical ordered ID sequence.  V1
        remains the format for the bounded protocols and remains readable.
        """

        stage = Path(tempfile.mkdtemp(prefix=".trace.tmp-", dir=self.root))
        samples_path = stage / "samples.jsonl"
        retained_ids = _JsonSequenceHash()
        retained_count = 0
        core_count = 0
        diagnostic_count = 0
        try:
            with samples_path.open("w", encoding="utf-8") as handle:
                for stratum, payload in self._selected_rows():
                    value = json.loads(payload)
                    value["selection_stratum"] = stratum
                    value["checkpoint_hash"] = checkpoint_hash
                    retained_ids.add(value["snapshot_id"])
                    retained_count += 1
                    if stratum in {"exact", "uniform_core", "uniform_fill"}:
                        core_count += 1
                    else:
                        diagnostic_count += 1
                    handle.write(canonical_json(value) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            observed = int(aggregate["observed_count"])
            if self.capability.fidelity == CaptureFidelity.UNAVAILABLE:
                fidelity = CaptureFidelity.UNAVAILABLE.value
            elif self.capability.fidelity == CaptureFidelity.AGGREGATE_ONLY:
                fidelity = CaptureFidelity.AGGREGATE_ONLY.value
            elif observed == retained_count:
                fidelity = CaptureFidelity.EXACT.value
            else:
                fidelity = CaptureFidelity.SAMPLED.value
            identity = {
                "format_version": 2,
                "run_id": self.run_id,
                "segment_id": self.segment_id,
                "boundary": self.boundary,
                "capability_id": self.capability.id,
                "capability_version": self.capability.version,
                "capture_fidelity": fidelity,
                "protocol": self.protocol,
                "reward_threshold": self.reward_threshold,
                "observed_count": observed,
                "retained_count": retained_count,
                "retained_ids_hash": retained_ids.hexdigest(),
                "aggregate": dict(aggregate),
                "checkpoint_hash": checkpoint_hash,
            }
            trace_hash = content_hash(identity)
            shard_id = "trace_" + trace_hash[:24]
            final_path = self.root / shard_id
            manifest = {
                **identity,
                "status": "complete",
                "shard_id": shard_id,
                "trace_hash": trace_hash,
                "core_count": core_count,
                "diagnostic_count": diagnostic_count,
                "created_at": _now(),
                "files": {"samples.jsonl": _file_hash(samples_path)},
            }
            _write_json(stage / "manifest.json", manifest)
            if final_path.exists():
                verification = verify_training_signal_shard(final_path)
                if not verification["valid"]:
                    raise RuntimeError(
                        "existing training signal shard failed integrity verification: "
                        + "; ".join(verification["problems"])
                    )
                existing = json.loads(
                    (final_path / "manifest.json").read_text(encoding="utf-8")
                )
                if existing.get("trace_hash") != trace_hash:
                    raise RuntimeError(
                        f"training signal content-address collision: {final_path}"
                    )
                shutil.rmtree(stage, ignore_errors=True)
                shard = self._shard_from_manifest(final_path, existing, reused=True)
            else:
                os.replace(stage, final_path)
                shard = self._shard_from_manifest(final_path, manifest)
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        self._finish_spool()
        self._sealed = shard
        return shard

    @staticmethod
    def _shard_from_manifest(
        path: Path, manifest: Mapping[str, Any], *, reused: bool = False
    ) -> TrainingSignalShard:
        return TrainingSignalShard(
            shard_id=str(manifest["shard_id"]),
            trace_hash=str(manifest["trace_hash"]),
            path=str(path),
            run_id=str(manifest["run_id"]),
            segment_id=str(manifest["segment_id"]),
            boundary=str(manifest["boundary"]),
            capability_id=str(manifest["capability_id"]),
            capability_version=str(manifest["capability_version"]),
            capture_fidelity=str(manifest["capture_fidelity"]),
            protocol=str(manifest["protocol"]),
            observed_count=int(manifest["observed_count"]),
            retained_count=int(manifest["retained_count"]),
            core_count=int(manifest["core_count"]),
            diagnostic_count=int(manifest["diagnostic_count"]),
            aggregate=copy.deepcopy(dict(manifest.get("aggregate") or {})),
            checkpoint_hash=manifest.get("checkpoint_hash"),
            created_at=str(manifest["created_at"]),
            reused=reused,
        )

    def _finish_spool(self) -> None:
        try:
            self._connection.commit()
            self._connection.close()
        finally:
            self._spool_path.unlink(missing_ok=True)
            Path(str(self._spool_path) + "-wal").unlink(missing_ok=True)
            Path(str(self._spool_path) + "-shm").unlink(missing_ok=True)

    def close(self) -> None:
        if self._sealed is None:
            self._connection.commit()
            self._pending_writes = 0
            self._connection.close()

    def __enter__(self) -> "TrainingSignalSink":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def verify_training_signal_shard(path: str | Path) -> Dict[str, Any]:
    """Verify a sealed shard's content address, inventory, and sample IDs."""

    root = Path(path).expanduser().resolve()
    problems = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "path": str(root), "problems": [str(exc)]}
    format_version = manifest.get("format_version")
    if manifest.get("status") != "complete" or format_version not in {1, 2}:
        problems.append("training signal shard is incomplete or unsupported")
    expected_files = manifest.get("files")
    if not isinstance(expected_files, Mapping) or set(expected_files) != {"samples.jsonl"}:
        problems.append("training signal shard has an invalid file inventory")
        expected_files = {}
    samples_path = root / "samples.jsonl"
    row_count = 0
    duplicate_ids = False
    ids_hash = _JsonSequenceHash()
    retained_ids = manifest.get("retained_ids") or [] if format_version == 1 else []
    if not samples_path.is_file():
        problems.append("training signal samples are missing")
    else:
        if expected_files and _file_hash(samples_path) != expected_files.get("samples.jsonl"):
            problems.append("training signal samples checksum mismatch")
        seen = sqlite3.connect("")
        try:
            seen.execute("CREATE TABLE ids (snapshot_id TEXT PRIMARY KEY)")
            with samples_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    snapshot_id = row.get("snapshot_id") if isinstance(row, Mapping) else None
                    ids_hash.add(snapshot_id)
                    if format_version == 1 and (
                        row_count >= len(retained_ids)
                        or snapshot_id != retained_ids[row_count]
                    ):
                        problems.append(
                            "training signal retained identities do not match the manifest"
                        )
                    try:
                        seen.execute("INSERT INTO ids VALUES (?)", (snapshot_id,))
                    except sqlite3.IntegrityError:
                        duplicate_ids = True
                    row_count += 1
        except (json.JSONDecodeError, TypeError):
            problems.append("training signal samples are invalid JSONL")
        finally:
            seen.close()
    if duplicate_ids:
        problems.append("training signal retained identities do not match the manifest")
    if format_version == 1 and row_count != len(retained_ids):
        problems.append("training signal retained identities do not match the manifest")
    if format_version == 2 and ids_hash.hexdigest() != manifest.get("retained_ids_hash"):
        problems.append("training signal retained identities do not match the manifest")
    if row_count != int(manifest.get("retained_count", -1)):
        problems.append("training signal retained count does not match its samples")
    identity_keys = [
        "format_version", "run_id", "segment_id", "boundary", "capability_id",
        "capability_version", "capture_fidelity", "protocol", "reward_threshold",
        "observed_count",
    ]
    if format_version == 1:
        identity_keys.append("retained_ids")
    else:
        identity_keys.extend(("retained_count", "retained_ids_hash"))
    identity_keys.extend(("aggregate", "checkpoint_hash"))
    computed = content_hash({key: manifest.get(key) for key in identity_keys})
    if computed != manifest.get("trace_hash"):
        problems.append("training signal manifest identity changed")
    if manifest.get("shard_id") != "trace_" + computed[:24] or root.name != manifest.get(
        "shard_id"
    ):
        problems.append("training signal content-addressed path is inconsistent")
    return {
        "valid": not problems,
        "path": str(root),
        "shard_id": manifest.get("shard_id"),
        "trace_hash": manifest.get("trace_hash"),
        "problems": problems,
    }


def load_training_signal_shard(path: str | Path) -> TrainingSignalShard:
    verification = verify_training_signal_shard(path)
    if not verification["valid"]:
        raise ValueError("invalid training signal shard: " + "; ".join(verification["problems"]))
    root = Path(path).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return TrainingSignalSink._shard_from_manifest(root, manifest)


__all__ = [
    "PROTOCOLS",
    "TrainingSignalShard",
    "TrainingSignalSink",
    "build_training_runtime_identity",
    "load_training_signal_shard",
    "verify_training_signal_shard",
]
