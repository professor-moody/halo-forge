"""Trainer adapter registry and immutable training-dataset artifacts.

The Dataset Lab canonical schemas are intentionally trainer-neutral.  This
module is the single conversion boundary from an immutable version to a
trainer-ready bundle.  Bundles are content addressed, published atomically,
and stored outside immutable version directories.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sqlite3
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    TextIO,
)

from .errors import VersionError
from .identity import RecordIdentity
from .models import infer_schema, normalize_kind, validate_record
from .sources import content_hash, hash_file, stable_json
from .storage import DatasetVersion, VersionStore

# V4 adds exact corpus/model/tokenizer/packing identity. Existing non-CPT
# renderers continue to mint V3 bundles so their established content addresses
# remain stable.
TRAINING_ARTIFACT_FORMAT_VERSION = 4
SUPPORTED_TRAINING_ARTIFACT_FORMAT_VERSIONS = frozenset({2, 3, 4})
DATASET_BINDING_ROLES = frozenset({"train", "validation", "test", "canary"})
HELD_OUT_SPLIT_NAMES = frozenset({"test", "canary"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl_value(handle: TextIO, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str) + "\n")


class _StableListHasher:
    """Incrementally hash a sequence exactly like ``content_hash(list(rows))``."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(b"[")
        self.count = 0

    def add(self, value: Any) -> None:
        if self.count:
            self._digest.update(b",")
        self._digest.update(stable_json(value).encode("utf-8"))
        self.count += 1

    def hexdigest(self) -> str:
        digest = self._digest.copy()
        digest.update(b"]")
        return digest.hexdigest()


def _hash_jsonl_list(path: Path) -> tuple[str, int]:
    hasher = _StableListHasher()
    for value in _iter_jsonl(path):
        hasher.add(value)
    return hasher.hexdigest(), hasher.count


@dataclass(frozen=True)
class DatasetBinding:
    role: str
    dataset_version_id: str
    split: str
    dataset_id: Optional[str] = None

    def __post_init__(self) -> None:
        role = self.role.strip().lower()
        if role not in DATASET_BINDING_ROLES:
            raise VersionError(
                f"Unknown dataset binding role {self.role!r}; choose: "
                + ", ".join(sorted(DATASET_BINDING_ROLES))
            )
        if not self.dataset_version_id.strip() or not self.split.strip():
            raise VersionError("Dataset bindings require dataset_version_id and split")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "dataset_version_id", self.dataset_version_id.strip())
        object.__setattr__(self, "split", self.split.strip())

    @classmethod
    def from_value(cls, value: "DatasetBinding | Mapping[str, Any] | str") -> "DatasetBinding":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                role, target = value.split("=", 1)
                version_id, split = target.rsplit(":", 1)
            except ValueError as exc:
                raise VersionError("String dataset bindings use role=version-id:split") from exc
            return cls(role, version_id, split)
        return cls(**dict(value))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _select(record: Mapping[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    return {name: copy.deepcopy(record[name]) for name in names if name in record}


def _fallback_chat_text(messages: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        f"<|{message.get('role', 'unknown')}|>\n{message.get('content', '')}"
        for message in messages
    )


@dataclass(frozen=True)
class TrainerDatasetAdapter:
    """Versioned conversion contract for one canonical trainer format."""

    id: str
    version: str
    canonical_schemas: tuple[str, ...]
    trainer_modes: tuple[str, ...]
    asset_fields: tuple[str, ...] = ()
    output_kind: Optional[str] = None

    def supports(self, schema: str, trainer_mode: Optional[str] = None) -> bool:
        canonical = normalize_kind(schema).value
        return canonical in self.canonical_schemas and (
            trainer_mode is None or trainer_mode.strip().lower() in self.trainer_modes
        )

    def render_record(self, record: Mapping[str, Any], *, tokenizer: Any = None) -> Dict[str, Any]:
        kind = self.output_kind or self.id
        if kind == "corpus":
            return _select(
                record,
                (
                    "document_id",
                    "document_hash",
                    "text",
                    "title",
                    "source_ref",
                    "source_spans",
                    "timestamp",
                    "metadata",
                ),
            )
        if kind == "sft":
            output = _select(record, ("prompt", "response", "system", "metadata"))
            chunks = [
                str(output.get("system") or "").strip(),
                str(output["prompt"]).strip(),
                str(output["response"]).strip(),
            ]
            output["text"] = "\n".join(chunk for chunk in chunks if chunk)
            return output
        if kind in {"chat", "tool", "agentic"}:
            fields = (
                "messages",
                "tools",
                "expected_calls",
                "expected_results",
                "metadata",
            )
            output = _select(record, fields)
            messages = output.get("messages") or []
            if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
                try:
                    output["text"] = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    output["text"] = _fallback_chat_text(messages)
            else:
                output["text"] = _fallback_chat_text(messages)
            return output
        if kind == "preference":
            return _select(record, ("prompt", "chosen", "rejected", "system", "metadata"))
        if kind == "prompt":
            return _select(record, ("prompt", "reference_answer", "metadata"))
        if kind == "vlm":
            output = _select(
                record,
                ("image", "prompt", "response", "ground_truth", "alternatives", "metadata"),
            )
            if "ground_truth" not in output and "response" in output:
                output["ground_truth"] = copy.deepcopy(output["response"])
            return output
        if kind == "audio":
            output = _select(record, ("audio", "task", "transcript", "label", "metadata"))
            output["audio_path"] = copy.deepcopy(output["audio"])
            output["text"] = str(output.get("transcript", output.get("label", "")))
            return output
        if kind == "classification":
            return _select(record, ("input", "media", "label", "labels", "metadata"))
        if kind == "embedding":
            return _select(record, ("anchor", "positive", "negatives", "metadata"))
        if kind == "reranking":
            return _select(
                record,
                (
                    "query",
                    "document",
                    "candidates",
                    "relevance",
                    "ordered_preference",
                    "metadata",
                ),
            )
        raise VersionError(f"Trainer dataset adapter {self.id!r} has no renderer")

    def record_text(self, record: Mapping[str, Any]) -> str:
        if isinstance(record.get("text"), str):
            return str(record["text"])
        values: List[str] = []
        for field_name in (
            "system",
            "prompt",
            "response",
            "chosen",
            "rejected",
            "reference_answer",
            "ground_truth",
            "transcript",
            "label",
            "task",
            "input",
            "anchor",
            "positive",
            "query",
            "document",
        ):
            if record.get(field_name) is not None:
                values.append(str(record[field_name]))
        if record.get("messages"):
            values.append(_fallback_chat_text(record["messages"]))
        return "\n".join(values)


class TrainerDatasetAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: Dict[str, TrainerDatasetAdapter] = {}

    def register(
        self, adapter: TrainerDatasetAdapter, *, replace: bool = False
    ) -> TrainerDatasetAdapter:
        identifier = adapter.id.strip().lower()
        if identifier in self._adapters and not replace:
            raise ValueError(f"Trainer dataset adapter {identifier!r} is already registered")
        self._adapters[identifier] = adapter
        return adapter

    def get(self, adapter_id: str) -> TrainerDatasetAdapter:
        try:
            return self._adapters[adapter_id.strip().lower()]
        except KeyError as exc:
            raise VersionError(f"Unknown trainer dataset adapter: {adapter_id}") from exc

    def list(
        self, *, schema: Optional[str] = None, trainer_mode: Optional[str] = None
    ) -> List[TrainerDatasetAdapter]:
        values = sorted(self._adapters.values(), key=lambda adapter: adapter.id)
        if schema is not None:
            values = [adapter for adapter in values if adapter.supports(schema, trainer_mode)]
        elif trainer_mode is not None:
            mode = trainer_mode.strip().lower()
            values = [adapter for adapter in values if mode in adapter.trainer_modes]
        return values

    def resolve(
        self,
        *,
        schema: str,
        trainer_mode: Optional[str] = None,
        adapter_id: Optional[str] = None,
    ) -> TrainerDatasetAdapter:
        if adapter_id:
            adapter = self.get(adapter_id)
            if not adapter.supports(schema, trainer_mode):
                raise VersionError(
                    f"Adapter {adapter.id!r} does not support schema {schema!r}"
                    + (f" for trainer {trainer_mode!r}" if trainer_mode else "")
                )
            return adapter
        normalized_schema = normalize_kind(schema).value
        if trainer_mode is None and normalized_schema in self._adapters:
            direct = self._adapters[normalized_schema]
            if direct.supports(schema):
                return direct
        matches = self.list(schema=schema, trainer_mode=trainer_mode)
        if not matches:
            raise VersionError(
                f"No trainer dataset adapter supports schema {schema!r}"
                + (f" for trainer {trainer_mode!r}" if trainer_mode else "")
            )
        return matches[0]


def default_trainer_dataset_adapters() -> TrainerDatasetAdapterRegistry:
    registry = TrainerDatasetAdapterRegistry()
    for adapter in (
        TrainerDatasetAdapter(
            "corpus",
            "1",
            ("corpus",),
            ("cpt",),
            output_kind="corpus",
        ),
        TrainerDatasetAdapter("sft", "1", ("sft",), ("sft",), output_kind="sft"),
        TrainerDatasetAdapter("chat", "1", ("chat",), ("sft", "chat"), output_kind="chat"),
        TrainerDatasetAdapter("tool", "1", ("tool",), ("sft", "tool"), output_kind="tool"),
        TrainerDatasetAdapter(
            "preference",
            "1",
            ("preference",),
            ("dpo", "orpo", "rm"),
            output_kind="preference",
        ),
        TrainerDatasetAdapter(
            "prompt",
            "1",
            ("prompt", "rlvr"),
            ("raft", "grpo", "reasoning", "rlvr"),
            output_kind="prompt",
        ),
        TrainerDatasetAdapter(
            "vlm", "1", ("vlm",), ("vlm",), asset_fields=("image",), output_kind="vlm"
        ),
        TrainerDatasetAdapter(
            "audio",
            "1",
            ("audio",),
            ("audio",),
            asset_fields=("audio",),
            output_kind="audio",
        ),
        TrainerDatasetAdapter("agentic", "1", ("tool",), ("agentic",), output_kind="agentic"),
        TrainerDatasetAdapter(
            "classification",
            "1",
            ("classification",),
            ("classify",),
            asset_fields=("media",),
            output_kind="classification",
        ),
        TrainerDatasetAdapter(
            "embedding",
            "1",
            ("embedding",),
            ("embed",),
            output_kind="embedding",
        ),
        TrainerDatasetAdapter(
            "reranking",
            "1",
            ("reranking",),
            ("rerank",),
            output_kind="reranking",
        ),
    ):
        registry.register(adapter)
    return registry


TRAINER_DATASET_ADAPTERS = default_trainer_dataset_adapters()


@dataclass(frozen=True)
class TrainingDatasetArtifact:
    artifact_id: str
    artifact_hash: str
    path: str
    adapter_id: str
    adapter_version: str
    trainer_mode: str
    schema: str
    bindings: tuple[DatasetBinding, ...]
    resolved_bindings: tuple[Dict[str, Any], ...]
    split_paths: Dict[str, str]
    row_counts: Dict[str, int]
    token_statistics: Dict[str, Any]
    model: Optional[str]
    tokenizer_revision: Optional[str]
    chat_template_hash: Optional[str]
    asset_roots: tuple[str, ...]
    validation_policy: Dict[str, Any]
    created_at: str
    model_revision: Optional[str] = None
    model_hash: Optional[str] = None
    tokenizer_hash: Optional[str] = None
    packing_plan: Optional[Dict[str, Any]] = None
    packing_plan_hash: Optional[str] = None
    split_fidelity: Dict[str, Any] = field(default_factory=dict)
    format_version: int = 3
    lineage_paths: Dict[str, str] = field(default_factory=dict)
    reused: bool = False

    @property
    def asset_root(self) -> Optional[str]:
        return self.asset_roots[0] if len(self.asset_roots) == 1 else None

    def to_dict(self) -> Dict[str, Any]:
        output = asdict(self)
        output["bindings"] = [binding.to_dict() for binding in self.bindings]
        output["resolved_bindings"] = [copy.deepcopy(value) for value in self.resolved_bindings]
        output["asset_roots"] = list(self.asset_roots)
        output["asset_root"] = self.asset_root
        return output

    def iter_lineage(self, role: str) -> Iterator[Dict[str, Any]]:
        """Yield row-aligned identity without materializing the full artifact.

        Format v3 exposes one JSONL index per trainer-visible split.  Format
        v2 bundles remain readable through their historical canonical JSON
        object, which is necessarily materialized as a compatibility path.
        """

        normalized = str(role).strip().lower()
        if normalized not in self.split_paths:
            raise VersionError(f"Training artifact has no split role {role!r}")
        lineage_path = self.lineage_paths.get(normalized)
        if lineage_path:
            yield from _iter_jsonl(Path(lineage_path))
            return
        legacy_path = Path(self.path) / "lineage.json"
        try:
            value = json.loads(legacy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VersionError(f"Invalid legacy training artifact lineage: {exc}") from exc
        rows = value.get(normalized) if isinstance(value, Mapping) else None
        if not isinstance(rows, list):
            raise VersionError(f"Training artifact lineage has no role {normalized!r}")
        for row in rows:
            if not isinstance(row, Mapping):
                raise VersionError("Training artifact lineage row must be an object")
            yield copy.deepcopy(dict(row))


def _summary(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "total": 0, "min": 0, "max": 0, "mean": 0.0, "p50": 0, "p95": 0}
    ordered = sorted(int(value) for value in values)
    percentile = lambda fraction: ordered[
        min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    ]
    return {
        "count": len(ordered),
        "total": sum(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
    }


class TrainingArtifactRenderer:
    """Render content-addressed trainer bundles from immutable versions."""

    def __init__(
        self,
        store: VersionStore,
        *,
        root: Optional[Path | str] = None,
        registry: Optional[TrainerDatasetAdapterRegistry] = None,
    ):
        self.store = store
        self.root = (
            Path(root).expanduser().resolve()
            if root is not None
            else store.root / ".artifacts" / "training"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry = registry or TRAINER_DATASET_ADAPTERS

    @staticmethod
    def _version_manifest(version: DatasetVersion) -> Dict[str, Any]:
        path = Path(version.path) / "manifest.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VersionError(f"Invalid dataset version manifest at {path}: {exc}") from exc

    @staticmethod
    def _schema(version: DatasetVersion, rows: Sequence[Mapping[str, Any]]) -> str:
        if version.schema:
            return normalize_kind(version.schema).value
        if not rows:
            raise VersionError(
                f"Cannot infer schema for empty dataset version {version.version_id}"
            )
        return infer_schema(rows[0]).value

    @staticmethod
    def _resolve_asset_value(
        value: Any,
        *,
        version_path: Path,
        manifest: Mapping[str, Any],
        asset_roots: set[str],
    ) -> Any:
        if isinstance(value, list):
            return [
                TrainingArtifactRenderer._resolve_asset_value(
                    item,
                    version_path=version_path,
                    manifest=manifest,
                    asset_roots=asset_roots,
                )
                for item in value
            ]
        if isinstance(value, Mapping):
            output = copy.deepcopy(dict(value))
            for path_field in ("path", "filename"):
                if path_field in output:
                    output[path_field] = TrainingArtifactRenderer._resolve_asset_value(
                        output[path_field],
                        version_path=version_path,
                        manifest=manifest,
                        asset_roots=asset_roots,
                    )
                    break
            return output
        if not isinstance(value, str) or value.startswith(("http://", "https://", "data:")):
            return copy.deepcopy(value)

        assets = {
            str(item.get("reference")): item
            for item in manifest.get("asset_fingerprints") or ()
            if isinstance(item, Mapping) and item.get("reference")
        }
        source_assets = {
            str(item.get("reference")): item
            for item in (manifest.get("source") or {}).get("assets") or ()
            if isinstance(item, Mapping) and item.get("reference")
        }
        mapping = dict(manifest.get("asset_mapping") or {})
        candidate: Optional[Path] = None
        expected: Optional[str] = None
        if value in mapping:
            candidate = (version_path / str(mapping[value])).resolve()
            expected = str((assets.get(value) or {}).get("fingerprint") or "") or None
        else:
            direct = Path(value).expanduser()
            if direct.is_absolute():
                candidate = direct.resolve()
            elif (version_path / direct).is_file():
                candidate = (version_path / direct).resolve()
            source_asset = source_assets.get(value) or assets.get(value)
            if candidate is None and source_asset and source_asset.get("resolved_path"):
                candidate = Path(str(source_asset["resolved_path"])).expanduser().resolve()
            if source_asset:
                expected = str(source_asset.get("fingerprint") or "") or None
            if candidate is not None and expected is None:
                original_reference = next(
                    (
                        reference
                        for reference, relative in mapping.items()
                        if str(relative) == value
                    ),
                    None,
                )
                if original_reference is not None:
                    expected = (
                        str((assets.get(str(original_reference)) or {}).get("fingerprint") or "")
                        or None
                    )
        if candidate is None or not candidate.is_file():
            raise VersionError(f"Dataset asset is missing: {value}")
        if expected and hash_file(candidate) != expected:
            raise VersionError(f"Dataset asset changed after version publication: {value}")
        asset_roots.add(str(candidate.parent))
        return str(candidate)

    def _render_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        adapter: TrainerDatasetAdapter,
        version: DatasetVersion,
        manifest: Mapping[str, Any],
        schema: str,
        tokenizer: Any,
        asset_roots: set[str],
    ) -> List[Dict[str, Any]]:
        return [
            self._render_row(
                row,
                index=index,
                adapter=adapter,
                version=version,
                manifest=manifest,
                schema=schema,
                tokenizer=tokenizer,
                asset_roots=asset_roots,
            )
            for index, row in enumerate(rows)
        ]

    def _render_row(
        self,
        row: Mapping[str, Any],
        *,
        index: int,
        adapter: TrainerDatasetAdapter,
        version: DatasetVersion,
        manifest: Mapping[str, Any],
        schema: str,
        tokenizer: Any,
        asset_roots: set[str],
    ) -> Dict[str, Any]:
        try:
            validate_record(row, schema)
            rendered = adapter.render_record(row, tokenizer=tokenizer)
        except Exception as exc:
            raise VersionError(
                f"Record {index} is incompatible with adapter {adapter.id!r}: {exc}"
            ) from exc
        for field_name in adapter.asset_fields:
            if field_name in rendered:
                rendered[field_name] = self._resolve_asset_value(
                    rendered[field_name],
                    version_path=Path(version.path),
                    manifest=manifest,
                    asset_roots=asset_roots,
                )
        if adapter.output_kind == "audio" and "audio" in rendered:
            rendered["audio_path"] = copy.deepcopy(rendered["audio"])
        return rendered

    @staticmethod
    def _load_tokenizer(
        tokenizer: Any, model: Optional[str], tokenizer_revision: Optional[str]
    ) -> tuple[Any, Optional[str]]:
        if tokenizer is not None:
            return tokenizer, None
        if not model:
            return None, "No tokenizer or model was supplied"
        try:
            from transformers import AutoTokenizer

            return (
                AutoTokenizer.from_pretrained(
                    model,
                    revision=tokenizer_revision,
                    local_files_only=True,
                    trust_remote_code=True,
                ),
                None,
            )
        except Exception as exc:
            return None, f"Tokenizer unavailable locally: {exc}"

    @staticmethod
    def _token_count(tokenizer: Any, adapter: TrainerDatasetAdapter, row: Mapping[str, Any]) -> int:
        if row.get("messages") and hasattr(tokenizer, "apply_chat_template"):
            try:
                tokens = tokenizer.apply_chat_template(
                    row["messages"], tokenize=True, add_generation_prompt=False
                )
                return len(tokens)
            except Exception:
                pass
        text = adapter.record_text(row)
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text, add_special_tokens=True))
        encoded = tokenizer(text)
        ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else encoded
        return len(ids)

    def _token_statistics(
        self,
        rows: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        adapter: TrainerDatasetAdapter,
        tokenizer: Any,
        fallback_reason: Optional[str],
    ) -> Dict[str, Any]:
        exact = tokenizer is not None
        split_stats: Dict[str, Any] = {}
        all_counts: List[int] = []
        for split_name, split_rows in rows.items():
            if exact:
                try:
                    counts = [self._token_count(tokenizer, adapter, row) for row in split_rows]
                except Exception as exc:
                    exact = False
                    fallback_reason = f"Tokenizer failed during counting: {exc}"
                    break
            else:
                counts = [
                    max(0, math.ceil(len(adapter.record_text(row)) / 4)) for row in split_rows
                ]
            split_stats[split_name] = _summary(counts)
            all_counts.extend(counts)
        if not exact:
            split_stats = {}
            all_counts = []
            for split_name, split_rows in rows.items():
                counts = [
                    max(0, math.ceil(len(adapter.record_text(row)) / 4)) for row in split_rows
                ]
                split_stats[split_name] = _summary(counts)
                all_counts.extend(counts)
        return {
            "exact": exact,
            "method": "tokenizer" if exact else "character_estimate",
            "fallback_reason": None if exact else fallback_reason,
            "overall": _summary(all_counts),
            "splits": split_stats,
        }

    @staticmethod
    def _disk_count_summary(
        connection: sqlite3.Connection, role: Optional[str] = None
    ) -> Dict[str, Any]:
        where = " WHERE role = ?" if role is not None else ""
        parameters: tuple[Any, ...] = (role,) if role is not None else ()
        count, total, minimum, maximum = connection.execute(
            f"SELECT COUNT(*), COALESCE(SUM(token_count), 0), "
            f"MIN(token_count), MAX(token_count) FROM token_counts{where}",
            parameters,
        ).fetchone()
        count = int(count)
        if count == 0:
            return {
                "count": 0,
                "total": 0,
                "min": 0,
                "max": 0,
                "mean": 0.0,
                "p50": 0,
                "p95": 0,
            }

        def percentile(fraction: float) -> int:
            offset = min(count - 1, math.ceil(count * fraction) - 1)
            row = connection.execute(
                f"SELECT token_count FROM token_counts{where} "
                "ORDER BY token_count LIMIT 1 OFFSET ?",
                (*parameters, offset),
            ).fetchone()
            return int(row[0])

        total = int(total)
        return {
            "count": count,
            "total": total,
            "min": int(minimum),
            "max": int(maximum),
            "mean": total / count,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
        }

    def _token_statistics_from_paths(
        self,
        rows: Mapping[str, Path],
        *,
        adapter: TrainerDatasetAdapter,
        tokenizer: Any,
        fallback_reason: Optional[str],
        database_path: Path,
    ) -> Dict[str, Any]:
        """Compute exact statistics with counts spooled to an on-disk index."""

        connection = sqlite3.connect(database_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute(
                "CREATE TABLE token_counts (role TEXT NOT NULL, token_count INTEGER NOT NULL)"
            )

            def populate(exact: bool) -> None:
                connection.execute("DELETE FROM token_counts")
                pending: List[tuple[str, int]] = []
                for split_name, path in rows.items():
                    for row in _iter_jsonl(path):
                        count = (
                            self._token_count(tokenizer, adapter, row)
                            if exact
                            else max(0, math.ceil(len(adapter.record_text(row)) / 4))
                        )
                        pending.append((split_name, int(count)))
                        if len(pending) >= 1000:
                            connection.executemany(
                                "INSERT INTO token_counts VALUES (?, ?)", pending
                            )
                            pending.clear()
                if pending:
                    connection.executemany("INSERT INTO token_counts VALUES (?, ?)", pending)
                connection.commit()

            exact = tokenizer is not None
            if exact:
                try:
                    populate(True)
                except Exception as exc:
                    exact = False
                    fallback_reason = f"Tokenizer failed during counting: {exc}"
                    connection.rollback()
            if not exact:
                populate(False)
            connection.execute(
                "CREATE INDEX token_counts_by_role " "ON token_counts (role, token_count)"
            )
            connection.execute("CREATE INDEX token_counts_overall ON token_counts (token_count)")
            connection.commit()
            split_stats = {
                split_name: self._disk_count_summary(connection, split_name) for split_name in rows
            }
            return {
                "exact": exact,
                "method": "tokenizer" if exact else "character_estimate",
                "fallback_reason": None if exact else fallback_reason,
                "overall": self._disk_count_summary(connection),
                "splits": split_stats,
            }
        finally:
            connection.close()

    @staticmethod
    def _write_canonical_lineage(path: Path, role_paths: Mapping[str, Path]) -> str:
        """Write lineage as canonical JSON while retaining JSON object format."""

        digest = hashlib.sha256()
        with path.open("w", encoding="utf-8") as handle:

            def emit(value: str) -> None:
                handle.write(value)
                digest.update(value.encode("utf-8"))

            emit("{")
            for role_index, role in enumerate(sorted(role_paths)):
                if role_index:
                    emit(",")
                emit(stable_json(role))
                emit(":[")
                for item_index, value in enumerate(_iter_jsonl(role_paths[role])):
                    if item_index:
                        emit(",")
                    emit(stable_json(value))
                emit("]")
            emit("}")
        return digest.hexdigest()

    def _artifact_from_manifest(
        self, path: Path, manifest: Mapping[str, Any], *, reused: bool = False
    ) -> TrainingDatasetArtifact:
        split_paths = {
            str(role): str((path / str(relative)).resolve())
            for role, relative in dict(manifest.get("split_paths") or {}).items()
        }
        lineage_paths = {
            str(role): str((path / str(relative)).resolve())
            for role, relative in dict(manifest.get("lineage_paths") or {}).items()
        }
        return TrainingDatasetArtifact(
            artifact_id=str(manifest["artifact_id"]),
            artifact_hash=str(manifest["artifact_hash"]),
            path=str(path),
            adapter_id=str(manifest["adapter_id"]),
            adapter_version=str(manifest["adapter_version"]),
            trainer_mode=str(manifest["trainer_mode"]),
            schema=str(manifest["schema"]),
            bindings=tuple(
                DatasetBinding.from_value(value) for value in manifest.get("bindings") or ()
            ),
            resolved_bindings=tuple(
                copy.deepcopy(dict(value)) for value in manifest.get("resolved_bindings") or ()
            ),
            split_paths=split_paths,
            row_counts={str(k): int(v) for k, v in dict(manifest.get("row_counts") or {}).items()},
            token_statistics=copy.deepcopy(dict(manifest.get("token_statistics") or {})),
            model=manifest.get("model"),
            tokenizer_revision=manifest.get("tokenizer_revision"),
            chat_template_hash=manifest.get("chat_template_hash"),
            asset_roots=tuple(str(value) for value in manifest.get("asset_roots") or ()),
            validation_policy=copy.deepcopy(dict(manifest.get("validation_policy") or {})),
            created_at=str(manifest["created_at"]),
            model_revision=manifest.get("model_revision"),
            model_hash=manifest.get("model_hash"),
            tokenizer_hash=manifest.get("tokenizer_hash"),
            packing_plan=(
                copy.deepcopy(dict(manifest.get("packing_plan") or {}))
                if manifest.get("packing_plan")
                else None
            ),
            packing_plan_hash=manifest.get("packing_plan_hash"),
            split_fidelity=copy.deepcopy(dict(manifest.get("split_fidelity") or {})),
            format_version=int(manifest.get("format_version") or 2),
            lineage_paths=lineage_paths,
            reused=reused,
        )

    @staticmethod
    def _verify_manifest_files(path: Path, manifest: Mapping[str, Any]) -> None:
        if manifest.get("status") != "complete":
            raise VersionError(f"Training artifact is incomplete: {path}")
        try:
            format_version = int(manifest.get("format_version"))
        except (TypeError, ValueError) as exc:
            raise VersionError(f"Training artifact has no verifiable integrity manifest: {path}") from exc
        if format_version not in SUPPORTED_TRAINING_ARTIFACT_FORMAT_VERSIONS:
            raise VersionError(f"Training artifact has no verifiable integrity manifest: {path}")

        artifact_hash = str(manifest.get("artifact_hash") or "")
        artifact_id = str(manifest.get("artifact_id") or "")
        identity = {
            "format_version": manifest.get("format_version"),
            "adapter_id": manifest.get("adapter_id"),
            "adapter_version": manifest.get("adapter_version"),
            "trainer_mode": manifest.get("trainer_mode"),
            "schema": manifest.get("schema"),
            "canonical_schemas": manifest.get("canonical_schemas"),
            # ``bindings`` is the public, compact binding list.  The content
            # identity was intentionally minted from the fully resolved list.
            "bindings": manifest.get("resolved_bindings"),
            "model": manifest.get("model"),
            "tokenizer_revision": manifest.get("tokenizer_revision"),
            "chat_template_hash": manifest.get("chat_template_hash"),
            "token_statistics": manifest.get("token_statistics"),
            "validation_policy": manifest.get("validation_policy"),
            "trainer_content": manifest.get("trainer_content"),
            "held_out_content": manifest.get("held_out_content"),
            "row_counts": manifest.get("row_counts"),
            "split_paths": manifest.get("split_paths"),
            "asset_roots": manifest.get("asset_roots"),
            "lineage_content": manifest.get("lineage_content"),
        }
        if format_version >= 3:
            identity["lineage_paths"] = manifest.get("lineage_paths")
            identity["lineage_index_content"] = manifest.get("lineage_index_content")
        if format_version >= 4:
            plan_identity = copy.deepcopy(dict(manifest.get("packing_plan") or {}))
            plan_identity["artifact_hash"] = None
            identity["model_revision"] = manifest.get("model_revision")
            identity["model_hash"] = manifest.get("model_hash")
            identity["tokenizer_hash"] = manifest.get("tokenizer_hash")
            identity["packing_plan"] = plan_identity
            identity["packing_plan_hash"] = manifest.get("packing_plan_hash")
            identity["split_fidelity"] = manifest.get("split_fidelity")
        computed_hash = content_hash(identity)
        if computed_hash != artifact_hash:
            raise VersionError(f"Training artifact manifest identity changed: {path}")
        if artifact_id != artifact_hash[:24] or path.name != artifact_id:
            raise VersionError(f"Training artifact content-address identity mismatch: {path}")

        hashes_value = manifest.get("artifact_hashes") or {}
        if not isinstance(hashes_value, Mapping):
            raise VersionError("Training artifact file hashes must be an object")
        recorded_hashes = {
            str(relative): str(expected) for relative, expected in hashes_value.items()
        }
        actual_files: set[str] = set()
        for artifact in path.rglob("*"):
            if artifact.is_symlink():
                raise VersionError(f"Training artifact contains a symbolic link: {artifact}")
            if artifact.is_file() and artifact.name != "manifest.json":
                actual_files.add(artifact.relative_to(path).as_posix())
        split_values = manifest.get("split_paths") or {}
        if not isinstance(split_values, Mapping):
            raise VersionError("Training artifact split paths must be an object")
        lineage_values = manifest.get("lineage_paths") or {}
        if not isinstance(lineage_values, Mapping):
            raise VersionError("Training artifact lineage paths must be an object")
        expected_files = {
            "lineage.json",
            *(str(value) for value in split_values.values()),
            *(str(value) for value in lineage_values.values()),
        }
        if actual_files != expected_files or set(recorded_hashes) != expected_files:
            missing = sorted(expected_files - actual_files)
            untracked = sorted(actual_files - expected_files)
            unrecorded = sorted(expected_files - set(recorded_hashes))
            unexpected_hashes = sorted(set(recorded_hashes) - expected_files)
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if untracked:
                details.append("untracked: " + ", ".join(untracked))
            if unrecorded:
                details.append("unrecorded: " + ", ".join(unrecorded))
            if unexpected_hashes:
                details.append("unexpected hashes: " + ", ".join(unexpected_hashes))
            raise VersionError("Training artifact file inventory changed: " + "; ".join(details))
        for relative, expected in recorded_hashes.items():
            artifact = (path / relative).resolve()
            try:
                artifact.relative_to(path.resolve())
            except ValueError as exc:
                raise VersionError(
                    f"Training artifact path escapes its bundle: {relative}"
                ) from exc
            if not artifact.is_file():
                raise VersionError(f"Training artifact file is missing: {relative}")
            if hash_file(artifact) != expected:
                raise VersionError(f"Training artifact file changed after publication: {relative}")

        split_paths_value = manifest.get("split_paths") or {}
        trainer_content_value = manifest.get("trainer_content") or {}
        row_counts_value = manifest.get("row_counts") or {}
        if not all(
            isinstance(value, Mapping)
            for value in (split_paths_value, trainer_content_value, row_counts_value)
        ):
            raise VersionError("Training artifact split metadata must be objects")
        split_paths = {str(role): str(relative) for role, relative in split_paths_value.items()}
        trainer_content = dict(trainer_content_value)
        row_counts = dict(row_counts_value)
        if set(split_paths) != set(trainer_content):
            raise VersionError("Training artifact split bindings do not match its content identity")
        for role, relative in split_paths.items():
            if role not in {"train", "validation"} or relative not in recorded_hashes:
                raise VersionError(f"Invalid trainer-visible split path: {role}={relative}")
            try:
                split_hash, split_count = _hash_jsonl_list(path / relative)
            except (OSError, json.JSONDecodeError) as exc:
                raise VersionError(f"Invalid training artifact split {relative}: {exc}") from exc
            if split_hash != str(trainer_content[role]):
                raise VersionError(f"Training artifact split content changed: {relative}")
            try:
                expected_count = int(row_counts.get(role, -1))
            except (TypeError, ValueError) as exc:
                raise VersionError(f"Invalid training artifact row count: {role}") from exc
            if split_count != expected_count:
                raise VersionError(f"Training artifact split row count changed: {relative}")

        lineage_path = path / "lineage.json"
        expected_lineage_hash = str(manifest.get("lineage_content") or "")
        if manifest.get("lineage_encoding") == "canonical-json":
            if hash_file(lineage_path) != expected_lineage_hash:
                raise VersionError("Training artifact lineage changed after publication")
        else:
            try:
                lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise VersionError(f"Invalid training artifact lineage: {exc}") from exc
            if not isinstance(lineage, Mapping):
                raise VersionError("Training artifact lineage must be an object")
            if content_hash(lineage) != expected_lineage_hash:
                raise VersionError("Training artifact lineage changed after publication")
            if set(lineage) != set(split_paths):
                raise VersionError("Training artifact lineage does not match trainer-visible splits")
            for role in split_paths:
                if not isinstance(lineage[role], list):
                    raise VersionError(f"Training artifact lineage split must be a list: {role}")
                try:
                    expected_count = int(row_counts.get(role, -1))
                except (TypeError, ValueError) as exc:
                    raise VersionError(f"Invalid training artifact row count: {role}") from exc
                if len(lineage[role]) != expected_count:
                    raise VersionError(f"Training artifact lineage row count changed: {role}")

        if format_version >= 3:
            lineage_paths = {str(role): str(relative) for role, relative in lineage_values.items()}
            lineage_hashes_value = manifest.get("lineage_index_content") or {}
            if not isinstance(lineage_hashes_value, Mapping):
                raise VersionError("Training artifact lineage index hashes must be an object")
            lineage_hashes = {str(role): str(value) for role, value in lineage_hashes_value.items()}
            if set(lineage_paths) != set(split_paths) or set(lineage_hashes) != set(split_paths):
                raise VersionError("Training artifact row-lineage index does not match its splits")
            for role, relative in lineage_paths.items():
                if relative not in recorded_hashes:
                    raise VersionError(f"Untracked training artifact lineage index: {relative}")
                try:
                    lineage_hash, lineage_count = _hash_jsonl_list(path / relative)
                except (OSError, json.JSONDecodeError) as exc:
                    raise VersionError(f"Invalid training artifact lineage index {relative}: {exc}") from exc
                if lineage_hash != lineage_hashes[role]:
                    raise VersionError(f"Training artifact lineage index changed: {role}")
                if lineage_count != int(row_counts.get(role, -1)):
                    raise VersionError(f"Training artifact lineage index row count changed: {role}")
        if format_version >= 4:
            if str(manifest.get("trainer_mode") or "").lower() != "cpt":
                raise VersionError("Training artifact format v4 is reserved for CPT corpus bundles")
            model_hash = str(manifest.get("model_hash") or "")
            tokenizer_hash = str(manifest.get("tokenizer_hash") or "")
            plan_hash = str(manifest.get("packing_plan_hash") or "")
            plan = manifest.get("packing_plan")
            fidelity = manifest.get("split_fidelity")
            if not model_hash or not tokenizer_hash or not plan_hash:
                raise VersionError("CPT artifact is missing model, tokenizer, or packing identity")
            if not isinstance(plan, Mapping) or not isinstance(fidelity, Mapping):
                raise VersionError("CPT artifact packing plan and split fidelity must be objects")
            if str(plan.get("artifact_hash") or "") != artifact_hash:
                raise VersionError("CPT packing plan does not link its published artifact")
            from halo_forge.cpt.packing import packing_plan_hash

            if packing_plan_hash(plan) != plan_hash:
                raise VersionError("CPT artifact packing plan identity changed")
            if set(fidelity) != set(split_paths):
                raise VersionError("CPT artifact split fidelity does not match trainer-visible splits")
            document_identities: Dict[str, set[str]] = {}
            for role, value in fidelity.items():
                if not isinstance(value, Mapping):
                    raise VersionError(f"CPT split fidelity must be an object: {role}")
                if str(value.get("source_content_hash") or "") != str(
                    trainer_content.get(role) or ""
                ):
                    raise VersionError(f"CPT split fidelity source changed: {role}")
                if int(value.get("overlap_tokens", -1)) != 0:
                    raise VersionError(f"CPT split packing overlaps source tokens: {role}")
                if int(value.get("dropped_tokens", -1)) != 0:
                    raise VersionError(f"CPT split packing dropped source tokens: {role}")
                identities = {
                    str(row.get("document_hash") or row.get("document_id"))
                    for row in _iter_jsonl(path / split_paths[role])
                }
                document_identities[role] = identities
                if str(value.get("document_identity_hash") or "") != content_hash(
                    sorted(identities)
                ):
                    raise VersionError(f"CPT split document identity changed: {role}")
            if document_identities.get("train", set()) & document_identities.get(
                "validation", set()
            ):
                raise VersionError("CPT train and validation document identity overlaps")

    def get(self, artifact_id: str) -> TrainingDatasetArtifact:
        matches = [path for path in self.root.glob(f"{artifact_id}*") if path.is_dir()]
        if len(matches) != 1:
            raise VersionError(f"Unknown or ambiguous training dataset artifact: {artifact_id}")
        path = matches[0]
        try:
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VersionError(f"Invalid training artifact manifest at {path}: {exc}") from exc
        self._verify_manifest_files(path, manifest)
        return self._artifact_from_manifest(path, manifest)

    def list(self) -> List[TrainingDatasetArtifact]:
        artifacts: List[TrainingDatasetArtifact] = []
        for path in sorted(self.root.iterdir()) if self.root.is_dir() else ():
            if path.name.startswith(".") or not (path / "manifest.json").is_file():
                continue
            try:
                manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                self._verify_manifest_files(path, manifest)
                artifacts.append(self._artifact_from_manifest(path, manifest))
            except (OSError, json.JSONDecodeError, VersionError):
                continue
        return sorted(artifacts, key=lambda artifact: artifact.created_at, reverse=True)

    def verify(self, artifact_id: str) -> Dict[str, Any]:
        try:
            artifact = self.get(artifact_id)
        except VersionError as exc:
            return {"valid": False, "artifact_id": artifact_id, "problems": [str(exc)]}
        return {"valid": True, "artifact_id": artifact.artifact_id, "problems": []}

    def clean_temporary(self) -> int:
        removed = 0
        for path in self.root.glob(".*.tmp-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        return removed

    def render(
        self,
        bindings: Sequence[DatasetBinding | Mapping[str, Any] | str],
        *,
        trainer_mode: str,
        adapter_id: Optional[str] = None,
        model: Optional[str] = None,
        model_revision: Optional[str] = None,
        model_hash: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        tokenizer_hash: Optional[str] = None,
        chat_template: Optional[str] = None,
        tokenizer: Any = None,
        max_sequence_length: int = 2048,
        packing: str = "paragraph_eos_non_overlap_v1",
        budget_mode: str = "passes",
        target_tokens: Optional[int] = None,
        corpus_passes: Optional[float] = 1.0,
        effective_batch_size: int = 1,
        validation_fraction: float = 0.05,
        seed: int = 0,
    ) -> TrainingDatasetArtifact:
        normalized_trainer_mode = trainer_mode.strip().lower()
        resolved_bindings = [DatasetBinding.from_value(value) for value in bindings]
        if not resolved_bindings or not any(
            binding.role == "train" for binding in resolved_bindings
        ):
            raise VersionError("A training artifact requires at least one train binding")
        if not 0 <= float(validation_fraction) < 1:
            raise VersionError("validation_fraction must be in [0, 1)")
        for binding in resolved_bindings:
            if (
                binding.role in {"train", "validation"}
                and binding.split.lower() in HELD_OUT_SPLIT_NAMES
            ):
                raise VersionError(
                    f"Held-out split {binding.split!r} cannot be exposed as {binding.role}"
                )

        effective_tokenizer_revision = tokenizer_revision or (
            model_revision if normalized_trainer_mode == "cpt" else None
        )
        tokenizer, fallback_reason = self._load_tokenizer(
            tokenizer, model, effective_tokenizer_revision
        )
        resolved_model_hash: Optional[str] = None
        resolved_tokenizer_hash: Optional[str] = None
        artifact_format_version = (
            TRAINING_ARTIFACT_FORMAT_VERSION if normalized_trainer_mode == "cpt" else 3
        )
        if normalized_trainer_mode == "cpt":
            if not model:
                raise VersionError("CPT training artifacts require an explicit model identity")
            if tokenizer is None:
                raise VersionError(
                    "CPT training artifacts require an exact tokenizer; "
                    + str(fallback_reason or "tokenizer loading failed")
                )
            from halo_forge.cpt.packing import (
                PACKING_ALGORITHM,
                model_identity_hash,
                tokenizer_identity_hash,
            )

            if str(packing).strip().lower() not in {
                PACKING_ALGORITHM,
                "paragraph_eos_non_overlap",
            }:
                raise VersionError(
                    "CPT supports only deterministic paragraph-aware non-overlap EOS packing"
                )
            packing = PACKING_ALGORITHM
            try:
                resolved_model_hash = model_identity_hash(
                    model,
                    revision=model_revision,
                    explicit_hash=model_hash,
                )
                resolved_tokenizer_hash = tokenizer_identity_hash(
                    tokenizer,
                    tokenizer_id=getattr(tokenizer, "name_or_path", None) or model,
                    revision=effective_tokenizer_revision,
                    explicit_hash=tokenizer_hash,
                )
            except (TypeError, ValueError) as exc:
                raise VersionError(str(exc)) from exc
        effective_template = chat_template or (
            str(getattr(tokenizer, "chat_template"))
            if tokenizer is not None and getattr(tokenizer, "chat_template", None)
            else None
        )
        chat_template_hash = content_hash(effective_template) if effective_template else None

        versions: Dict[tuple[Optional[str], str], DatasetVersion] = {}
        manifests: Dict[tuple[Optional[str], str], Dict[str, Any]] = {}
        schemas: List[str] = []
        version_schemas: Dict[tuple[Optional[str], str], str] = {}
        for binding in resolved_bindings:
            key = (binding.dataset_id, binding.dataset_version_id)
            version = versions.setdefault(
                key,
                self.store.get_any(binding.dataset_version_id, binding.dataset_id),
            )
            verification = self.store.verify(version.version_id, dataset_id=version.dataset_id)
            if not verification["valid"]:
                raise VersionError(
                    f"Dataset version {version.version_id} is invalid: "
                    + "; ".join(verification["problems"])
                )
            manifests.setdefault(key, self._version_manifest(version))
            if version.schema:
                resolved_schema = normalize_kind(version.schema).value
            else:
                first = next(
                    self.store.iter_records(
                        binding.dataset_version_id,
                        dataset_id=binding.dataset_id,
                        split=binding.split,
                    ),
                    None,
                )
                if first is None:
                    raise VersionError(
                        f"Cannot infer schema for empty dataset version {version.version_id}"
                    )
                resolved_schema = infer_schema(first).value
            version_schemas[key] = resolved_schema
            if resolved_schema not in schemas:
                schemas.append(resolved_schema)
        schema = schemas[0]
        adapter = self.registry.resolve(
            schema=schema, trainer_mode=normalized_trainer_mode, adapter_id=adapter_id
        )
        if any(not adapter.supports(value, normalized_trainer_mode) for value in schemas):
            raise VersionError(
                f"Adapter {adapter.id!r} cannot combine canonical schemas: " + ", ".join(schemas)
            )
        artifact_schema = schema if len(schemas) == 1 else "+".join(sorted(schemas))

        # If callers bind only train, honor a sibling validation split already
        # present in that immutable version before deriving one.  Existing test
        # and canary siblings are always bound as held-out identity, never as
        # trainer-readable files.
        additions: List[DatasetBinding] = []
        existing = {
            (binding.role, binding.dataset_id, binding.dataset_version_id, binding.split)
            for binding in resolved_bindings
        }
        sibling_roles = [
            role
            for role in ("validation", "test", "canary")
            if not any(binding.role == role for binding in resolved_bindings)
        ]
        for role in sibling_roles:
            for binding in tuple(resolved_bindings):
                if binding.role != "train":
                    continue
                version = versions[(binding.dataset_id, binding.dataset_version_id)]
                candidates = ("validation", "val") if role == "validation" else (role,)
                sibling = next((name for name in candidates if name in version.split_counts), None)
                if sibling:
                    candidate = DatasetBinding(
                        role,
                        binding.dataset_version_id,
                        sibling,
                        binding.dataset_id,
                    )
                    key = (
                        candidate.role,
                        candidate.dataset_id,
                        candidate.dataset_version_id,
                        candidate.split,
                    )
                    if key not in existing:
                        additions.append(candidate)
                        existing.add(key)
        resolved_bindings.extend(additions)

        temporary = Path(tempfile.mkdtemp(prefix=".render.tmp-", dir=self.root))
        try:
            splits_dir = temporary / "splits"
            splits_dir.mkdir()
            spool_dir = temporary / ".spool"
            spool_dir.mkdir()
            candidate_records = {
                role: spool_dir / f"{role}.records.jsonl" for role in ("train", "validation")
            }
            candidate_lineage = {
                role: spool_dir / f"{role}.lineage.jsonl" for role in ("train", "validation")
            }
            for path in (*candidate_records.values(), *candidate_lineage.values()):
                path.touch()

            asset_roots: set[str] = set()
            prepared_counts = {role: 0 for role in ("train", "validation", "test", "canary")}
            held_out_hashers = {role: _StableListHasher() for role in ("test", "canary")}
            binding_manifest: List[Dict[str, Any]] = []
            missing_reference = 0
            for binding in resolved_bindings:
                key = (binding.dataset_id, binding.dataset_version_id)
                version = versions[key]
                binding_hasher = _StableListHasher()
                visible = binding.role in {"train", "validation"}
                record_handle = (
                    candidate_records[binding.role].open("a", encoding="utf-8") if visible else None
                )
                lineage_handle = (
                    candidate_lineage[binding.role].open("a", encoding="utf-8") if visible else None
                )
                try:
                    for index, (row, identity) in enumerate(
                        self.store.iter_records_with_lineage(
                            binding.dataset_version_id,
                            dataset_id=binding.dataset_id,
                            split=binding.split,
                        )
                    ):
                        binding_hasher.add(row)
                        prepared_counts[binding.role] += 1
                        if row.get("reference_answer") is None:
                            missing_reference += 1
                        if visible:
                            rendered = self._render_row(
                                row,
                                index=index,
                                adapter=adapter,
                                version=version,
                                manifest=manifests[key],
                                schema=version_schemas[key],
                                tokenizer=tokenizer,
                                asset_roots=asset_roots,
                            )
                            assert record_handle is not None and lineage_handle is not None
                            _write_jsonl_value(record_handle, rendered)
                            _write_jsonl_value(lineage_handle, identity.to_dict())
                        else:
                            held_out_hashers[binding.role].add(row)
                finally:
                    if record_handle is not None:
                        record_handle.close()
                    if lineage_handle is not None:
                        lineage_handle.close()
                binding_manifest.append(
                    {
                        **binding.to_dict(),
                        "dataset_id": version.dataset_id,
                        "content_hash": version.content_hash,
                        "row_count": binding_hasher.count,
                        "content_fingerprint": binding_hasher.hexdigest(),
                        "exposed_to_trainer": visible,
                    }
                )

            if normalized_trainer_mode == "reasoning" and missing_reference:
                raise VersionError(
                    "Reasoning training requires reference_answer on every record; "
                    f"{missing_reference} record(s) are missing it"
                )

            validation_supplied = any(binding.role == "validation" for binding in resolved_bindings)
            final_counts = dict(prepared_counts)
            final_record_paths: Dict[str, Path] = {}
            final_lineage_paths: Dict[str, Path] = {}
            if validation_supplied:
                validation_policy: Dict[str, Any] = {
                    "kind": "supplied",
                    "preserved": True,
                    "row_count": prepared_counts["validation"],
                }
                for role in ("train", "validation"):
                    if not prepared_counts[role]:
                        continue
                    final_record_paths[role] = splits_dir / f"{role}.jsonl"
                    final_lineage_paths[role] = spool_dir / f"final-{role}.lineage.jsonl"
                    shutil.copyfile(candidate_records[role], final_record_paths[role])
                    shutil.copyfile(candidate_lineage[role], final_lineage_paths[role])
            else:
                total_train = prepared_counts["train"]
                validation_count = 0
                if total_train > 1 and validation_fraction > 0:
                    validation_count = min(
                        total_train - 1,
                        max(1, round(total_train * validation_fraction)),
                    )
                final_counts["train"] = total_train - validation_count
                final_counts["validation"] = validation_count
                validation_policy = {
                    "kind": "derived",
                    "preserved": False,
                    "seed": int(seed),
                    "fraction": float(validation_fraction),
                    "row_count": validation_count,
                    "selection": "sha256(seed,instance_id)",
                }

                selection_path = spool_dir / "validation-selection.sqlite3"
                selection = sqlite3.connect(selection_path)
                try:
                    selection.execute("PRAGMA journal_mode=OFF")
                    selection.execute("PRAGMA synchronous=OFF")
                    selection.execute(
                        "CREATE TABLE ranks (position INTEGER PRIMARY KEY, rank_hash TEXT NOT NULL)"
                    )
                    pending: List[tuple[int, str]] = []
                    for position, identity in enumerate(_iter_jsonl(candidate_lineage["train"])):
                        pending.append(
                            (
                                position,
                                content_hash(
                                    {
                                        "seed": int(seed),
                                        "instance_id": identity["instance_id"],
                                    }
                                ),
                            )
                        )
                        if len(pending) >= 1000:
                            selection.executemany("INSERT INTO ranks VALUES (?, ?)", pending)
                            pending.clear()
                    if pending:
                        selection.executemany("INSERT INTO ranks VALUES (?, ?)", pending)
                    selection.execute("CREATE INDEX ranks_by_hash ON ranks (rank_hash, position)")
                    selection.execute("CREATE TABLE selected (position INTEGER PRIMARY KEY)")
                    if validation_count:
                        selection.execute(
                            "INSERT INTO selected SELECT position FROM ranks "
                            "ORDER BY rank_hash, position LIMIT ?",
                            (validation_count,),
                        )
                    selection.commit()

                    for role in ("train", "validation"):
                        if final_counts[role]:
                            final_record_paths[role] = splits_dir / f"{role}.jsonl"
                            final_lineage_paths[role] = spool_dir / f"final-{role}.lineage.jsonl"
                    record_outputs = {
                        role: path.open("w", encoding="utf-8")
                        for role, path in final_record_paths.items()
                    }
                    lineage_outputs = {
                        role: path.open("w", encoding="utf-8")
                        for role, path in final_lineage_paths.items()
                    }
                    try:
                        selected_cursor = iter(
                            selection.execute("SELECT position FROM selected ORDER BY position")
                        )
                        selected_row = next(selected_cursor, None)
                        with (
                            candidate_records["train"].open(encoding="utf-8") as records_handle,
                            candidate_lineage["train"].open(encoding="utf-8") as lineage_handle,
                        ):
                            record_lines = (line for line in records_handle if line.strip())
                            lineage_lines = (line for line in lineage_handle if line.strip())
                            sentinel = object()
                            for position, pair in enumerate(
                                zip_longest(record_lines, lineage_lines, fillvalue=sentinel)
                            ):
                                record_line, lineage_line = pair
                                if record_line is sentinel or lineage_line is sentinel:
                                    raise VersionError(
                                        "Rendered training rows and lineage are inconsistent"
                                    )
                                selected = (
                                    selected_row is not None and int(selected_row[0]) == position
                                )
                                role = "validation" if selected else "train"
                                record_outputs[role].write(record_line)
                                lineage_outputs[role].write(lineage_line)
                                if selected:
                                    selected_row = next(selected_cursor, None)
                    finally:
                        for handle in (*record_outputs.values(), *lineage_outputs.values()):
                            handle.close()
                finally:
                    selection.close()

            relative_paths = {role: f"splits/{role}.jsonl" for role in final_record_paths}
            trainer_content = {
                role: _hash_jsonl_list(path)[0] for role, path in final_record_paths.items()
            }
            packing_plan: Optional[Dict[str, Any]] = None
            resolved_packing_plan_hash: Optional[str] = None
            split_fidelity: Dict[str, Any] = {}
            if normalized_trainer_mode == "cpt":
                from halo_forge.cpt.packing import (
                    build_corpus_packing_plan,
                    pack_corpus_records,
                    packing_plan_hash,
                )

                assert tokenizer is not None and resolved_tokenizer_hash is not None
                try:
                    document_identities = {
                        role: {
                            str(row.get("document_hash") or row.get("document_id"))
                            for row in _iter_jsonl(path)
                        }
                        for role, path in final_record_paths.items()
                    }
                    overlap = document_identities.get("train", set()) & document_identities.get(
                        "validation", set()
                    )
                    if overlap:
                        raise VersionError(
                            "CPT train and validation splits share document identity: "
                            + ", ".join(sorted(overlap)[:5])
                        )
                    packed_train = pack_corpus_records(
                        _iter_jsonl(final_record_paths["train"]),
                        tokenizer,
                        max_sequence_length=int(max_sequence_length),
                    )
                    packed_validation = (
                        pack_corpus_records(
                            _iter_jsonl(final_record_paths["validation"]),
                            tokenizer,
                            max_sequence_length=int(max_sequence_length),
                        )
                        if "validation" in final_record_paths
                        else None
                    )
                    plan = build_corpus_packing_plan(
                        train=packed_train,
                        validation=packed_validation,
                        tokenizer_id=getattr(tokenizer, "name_or_path", None) or str(model),
                        tokenizer_revision=effective_tokenizer_revision,
                        tokenizer_hash=resolved_tokenizer_hash,
                        max_sequence_length=int(max_sequence_length),
                        budget_mode=budget_mode,
                        target_tokens=target_tokens,
                        corpus_passes=corpus_passes,
                        effective_batch_size=int(effective_batch_size),
                    )
                except (TypeError, ValueError) as exc:
                    raise VersionError(f"CPT corpus packing failed: {exc}") from exc
                packing_plan = plan.to_dict()
                resolved_packing_plan_hash = packing_plan_hash(plan)
                packed_by_role = {"train": packed_train}
                if packed_validation is not None:
                    packed_by_role["validation"] = packed_validation
                split_fidelity = {
                    role: {
                        "source_content_hash": trainer_content[role],
                        "packing_source_hash": packed.source_hash,
                        "document_identity_hash": content_hash(
                            sorted(document_identities[role])
                        ),
                        "document_count": int(packed.statistics["document_count"]),
                        "paragraph_count": int(packed.statistics["paragraph_count"]),
                        "packed_tokens": int(packed.statistics["packed_tokens"]),
                        "block_count": int(packed.statistics["block_count"]),
                        "overlap_tokens": int(packed.statistics["overlap_tokens"]),
                        "dropped_tokens": int(packed.statistics["dropped_tokens"]),
                    }
                    for role, packed in packed_by_role.items()
                }
                all_lengths = sorted(
                    len(sequence.input_ids)
                    for packed in packed_by_role.values()
                    for sequence in packed.sequences
                )

                def packed_percentile(fraction: float) -> int:
                    return all_lengths[
                        min(
                            len(all_lengths) - 1,
                            math.ceil(len(all_lengths) * fraction) - 1,
                        )
                    ]

                total_packed_tokens = sum(all_lengths)
                token_statistics = {
                    "exact": True,
                    "method": "paragraph_eos_non_overlap",
                    "fallback_reason": None,
                    "overall": {
                        "count": len(all_lengths),
                        "total": total_packed_tokens,
                        "min": all_lengths[0],
                        "max": all_lengths[-1],
                        "mean": total_packed_tokens / len(all_lengths),
                        "p50": packed_percentile(0.50),
                        "p95": packed_percentile(0.95),
                    },
                    "splits": {
                        role: {
                            "count": int(packed.statistics["block_count"]),
                            "total": int(packed.statistics["packed_tokens"]),
                            "min": int(packed.statistics["min_sequence_tokens"]),
                            "max": int(packed.statistics["max_sequence_tokens"]),
                            "mean": float(packed.statistics["mean_sequence_tokens"]),
                            "p50": int(packed.statistics["p50_sequence_tokens"]),
                            "p95": int(packed.statistics["p95_sequence_tokens"]),
                            "documents": int(packed.statistics["document_count"]),
                            "paragraphs": int(packed.statistics["paragraph_count"]),
                            "content_tokens": int(packed.statistics["content_tokens"]),
                            "eos_tokens": int(packed.statistics["eos_tokens"]),
                            "padding_tokens": int(packed.statistics["padding_tokens"]),
                            "utilization": float(packed.statistics["utilization"]),
                            "overlap_tokens": 0,
                            "dropped_tokens": 0,
                        }
                        for role, packed in packed_by_role.items()
                    },
                }
            else:
                token_statistics = self._token_statistics_from_paths(
                    final_record_paths,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    fallback_reason=fallback_reason,
                    database_path=spool_dir / "token-counts.sqlite3",
                )
            lineage_content = self._write_canonical_lineage(
                temporary / "lineage.json", final_lineage_paths
            )
            lineage_dir = temporary / "lineage"
            lineage_dir.mkdir()
            lineage_relative_paths: Dict[str, str] = {}
            lineage_index_content: Dict[str, str] = {}
            for role, source_path in final_lineage_paths.items():
                relative = f"lineage/{role}.jsonl"
                destination = temporary / relative
                shutil.copyfile(source_path, destination)
                lineage_relative_paths[role] = relative
                lineage_index_content[role] = _hash_jsonl_list(destination)[0]
            row_counts = {
                role: int(final_counts[role]) for role in ("train", "validation", "test", "canary")
            }
            artifact_identity = {
                "format_version": artifact_format_version,
                "adapter_id": adapter.id,
                "adapter_version": adapter.version,
                "trainer_mode": normalized_trainer_mode,
                "schema": artifact_schema,
                "canonical_schemas": sorted(schemas),
                "bindings": binding_manifest,
                "model": model,
                "tokenizer_revision": effective_tokenizer_revision,
                "chat_template_hash": chat_template_hash,
                "token_statistics": token_statistics,
                "validation_policy": validation_policy,
                "trainer_content": trainer_content,
                "held_out_content": {
                    role: held_out_hashers[role].hexdigest() for role in ("test", "canary")
                },
                "row_counts": row_counts,
                "split_paths": relative_paths,
                "asset_roots": sorted(asset_roots),
                "lineage_content": lineage_content,
                "lineage_paths": lineage_relative_paths,
                "lineage_index_content": lineage_index_content,
            }
            if artifact_format_version >= 4:
                artifact_identity.update(
                    {
                        "model_revision": model_revision,
                        "model_hash": resolved_model_hash,
                        "tokenizer_hash": resolved_tokenizer_hash,
                        "packing_plan": packing_plan,
                        "packing_plan_hash": resolved_packing_plan_hash,
                        "split_fidelity": split_fidelity,
                    }
                )
            artifact_hash = content_hash(artifact_identity)
            artifact_id = artifact_hash[:24]
            final_path = self.root / artifact_id
            if final_path.exists():
                manifest = json.loads((final_path / "manifest.json").read_text(encoding="utf-8"))
                if manifest.get("artifact_hash") != artifact_hash:
                    raise VersionError(
                        f"Training artifact content-address collision at {final_path}"
                    )
                self._verify_manifest_files(final_path, manifest)
                shutil.rmtree(temporary, ignore_errors=True)
                return self._artifact_from_manifest(final_path, manifest, reused=True)

            shutil.rmtree(spool_dir)
            artifact_hashes = {
                path.relative_to(temporary).as_posix(): hash_file(path)
                for path in sorted(temporary.rglob("*"))
                if path.is_file()
            }
            manifest = {
                **artifact_identity,
                "status": "complete",
                "artifact_id": artifact_id,
                "artifact_hash": artifact_hash,
                "created_at": _now(),
                "bindings": [binding.to_dict() for binding in resolved_bindings],
                "resolved_bindings": binding_manifest,
                "split_paths": relative_paths,
                "row_counts": row_counts,
                "token_statistics": token_statistics,
                "asset_roots": sorted(asset_roots),
                "artifact_hashes": artifact_hashes,
                "lineage_encoding": "canonical-json",
            }
            if artifact_format_version >= 4 and packing_plan is not None:
                manifest["packing_plan"] = {
                    **packing_plan,
                    "artifact_hash": artifact_hash,
                }
            _write_json(temporary / "manifest.json", manifest)
            os.replace(temporary, final_path)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self._artifact_from_manifest(final_path, manifest)


__all__ = [
    "DATASET_BINDING_ROLES",
    "DatasetBinding",
    "TRAINER_DATASET_ADAPTERS",
    "TRAINING_ARTIFACT_FORMAT_VERSION",
    "SUPPORTED_TRAINING_ARTIFACT_FORMAT_VERSIONS",
    "TrainerDatasetAdapter",
    "TrainerDatasetAdapterRegistry",
    "TrainingArtifactRenderer",
    "TrainingDatasetArtifact",
    "default_trainer_dataset_adapters",
]
