"""Evaluation adapter registry and adapters over existing evaluators."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from halo_forge.run_db import BenchmarkSuiteRevisionRecord

from .models import (
    EvaluationAdapterResult,
    EvaluationLabError,
    EvaluationMetric,
    EvaluationSample,
    ResolvedSubject,
)

ADAPTER_ALIASES = {
    "lm_eval": "lm-eval",
    "lm-evaluation-harness": "lm-eval",
    "lm_evaluation_harness": "lm-eval",
    "dataset_fixture": "dataset",
    "fixture": "dataset",
    "halo_forge": "benchmark",
    "native": "benchmark",
    "pass_at_k": "code",
    "code_pass_at_k": "code",
    "tool_use": "tool",
    "tool-use": "tool",
    "vlmevalkit": "vlm",
    "vision": "vlm",
    "speech": "audio",
    "composite": "suite",
    "inference-performance": "performance",
    "operational": "performance",
}


def canonical_adapter_id(adapter_id: str) -> str:
    value = str(adapter_id or "").strip().lower()
    return ADAPTER_ALIASES.get(value, value)


def adapter_for_item(item: Mapping[str, Any]) -> str:
    selected = item.get("adapter_id") or item.get("adapter")
    if selected:
        return canonical_adapter_id(str(selected))
    if item.get("verifier"):
        return "verifier"
    if item.get("benchmark"):
        return "benchmark"
    if item.get("task"):
        return "lm-eval"
    return "dataset"


def infer_suite_adapter(revision: BenchmarkSuiteRevisionRecord) -> str:
    adapters = {adapter_for_item(item) for item in revision.items}
    if not adapters:
        raise EvaluationLabError("benchmark suite revision has no items")
    return next(iter(adapters)) if len(adapters) == 1 else "suite"


class EvaluationContext:
    """Progress/cancellation bridge supplied to an adapter by the manager."""

    def __init__(self, manager: Any, evaluation_id: str, work_dir: Path):
        self._manager = manager
        self.evaluation_id = evaluation_id
        self.work_dir = work_dir

    def check_cancelled(self) -> None:
        self._manager.check_cancelled(self.evaluation_id)

    def progress(
        self,
        *,
        processed: Optional[int] = None,
        total: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> None:
        self._manager.update_progress(
            self.evaluation_id, processed=processed, total=total, stage=stage
        )

    def log(self, message: str) -> None:
        self._manager.log(self.evaluation_id, message)


class EvaluationAdapter(ABC):
    adapter_id: str
    adapter_version: str

    @abstractmethod
    def evaluate(
        self,
        context: EvaluationContext,
        revision: BenchmarkSuiteRevisionRecord,
        subject: ResolvedSubject,
        request: Mapping[str, Any],
    ) -> EvaluationAdapterResult:
        raise NotImplementedError


class EvaluationAdapterRegistry:
    def __init__(self, adapters: Iterable[EvaluationAdapter] = ()):
        self._adapters: Dict[str, EvaluationAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: EvaluationAdapter, *, replace: bool = False) -> None:
        adapter_id = str(getattr(adapter, "adapter_id", "")).strip()
        version = str(getattr(adapter, "adapter_version", "")).strip()
        if not adapter_id or not version:
            raise EvaluationLabError("evaluation adapters need adapter_id and adapter_version")
        if adapter_id in self._adapters and not replace:
            raise EvaluationLabError(f"evaluation adapter already registered: {adapter_id}")
        self._adapters[adapter_id] = adapter

    def get(self, adapter_id: str) -> EvaluationAdapter:
        adapter_id = canonical_adapter_id(adapter_id)
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            raise EvaluationLabError(f"unknown evaluation adapter: {adapter_id}") from exc

    def list(self) -> list[dict[str, str]]:
        return [
            {"id": adapter.adapter_id, "version": adapter.adapter_version}
            for adapter in sorted(self._adapters.values(), key=lambda value: value.adapter_id)
        ]


_MISSING = object()


def _mapped_value(values: Mapping[str, Any], item: Mapping[str, Any], item_id: str) -> Any:
    """Find request evidence by occurrence ID first, then logical record ID."""

    for key in (item_id, item.get("record_id")):
        if key is not None and str(key) in values:
            return values[str(key)]
    return _MISSING


def _last_assistant_turn(messages: Any) -> tuple[Any, Any]:
    if not isinstance(messages, list):
        return messages, None
    copied = [dict(value) if isinstance(value, Mapping) else value for value in messages]
    if copied and isinstance(copied[-1], Mapping):
        role = str(copied[-1].get("role") or copied[-1].get("from") or "").lower()
        if role in {"assistant", "model", "gpt"}:
            final = copied[-1]
            return copied[:-1], final.get("content", final.get("value"))
    return copied, None


def _infer_canonical_schema(record: Mapping[str, Any]) -> str:
    if "image" in record:
        return "vlm"
    if "audio" in record:
        return "audio"
    if "tools" in record:
        return "tool"
    if "messages" in record:
        return "chat"
    if "chosen" in record and "rejected" in record:
        return "preference"
    if "reference_answer" in record:
        return "prompt"
    return "sft"


def _canonical_input_expected(
    record: Mapping[str, Any], schema: Optional[str]
) -> tuple[Any, Any, Dict[str, Any]]:
    """Project one canonical Dataset Lab row into evaluation evidence."""

    normalized = str(schema or _infer_canonical_schema(record)).lower()
    if normalized in {"reasoning", "rlvr", "rlvr/reasoning"}:
        normalized = "prompt"
    if normalized in {"tool-use", "tool_use", "agentic"}:
        normalized = "tool"
    metadata: Dict[str, Any] = {"canonical_schema": normalized}
    if record.get("system") is not None:
        metadata["system"] = record.get("system")
    if normalized == "sft":
        return record.get("prompt"), record.get("response"), metadata
    if normalized == "preference":
        return (
            record.get("prompt"),
            record.get("chosen"),
            {
                **metadata,
                "rejected": record.get("rejected"),
            },
        )
    if normalized == "prompt":
        return record.get("prompt"), record.get("reference_answer"), metadata
    if normalized == "chat":
        prompt_messages, expected = _last_assistant_turn(record.get("messages"))
        return prompt_messages, expected, metadata
    if normalized == "tool":
        prompt_messages, assistant_expected = _last_assistant_turn(record.get("messages"))
        expected = record.get("expected_calls")
        if expected is None:
            expected = record.get("expected_results")
        if expected is None:
            expected = assistant_expected
        return (
            {
                "messages": prompt_messages,
                "tools": record.get("tools"),
            },
            expected,
            metadata,
        )
    if normalized == "vlm":
        expected = record.get("ground_truth")
        if expected is None:
            expected = record.get("response")
        return (
            {
                "image": record.get("image"),
                "prompt": record.get("prompt"),
            },
            expected,
            metadata,
        )
    if normalized == "audio":
        expected = record.get("transcript")
        if expected is None:
            expected = record.get("label")
        return (
            {
                "audio": record.get("audio"),
                "task": record.get("task"),
            },
            expected,
            metadata,
        )
    return record.get("prompt", record), record.get("response"), metadata


def _dataset_root(item: Mapping[str, Any], request: Mapping[str, Any]) -> Path:
    configured = (
        item.get("dataset_root")
        or request.get("dataset_root")
        or os.environ.get("HALOFORGE_DATASET_ROOT")
    )
    return Path(configured or (Path.home() / ".halo-forge" / "datasets")).expanduser()


class _ExpandedDatasetItems:
    """A sized, re-iterable stream over suite items and Dataset Lab splits."""

    def __init__(self, items: Sequence[Mapping[str, Any]], request: Mapping[str, Any]):
        from halo_forge.data_lab import VersionStore
        from halo_forge.data_lab.errors import VersionError

        self._descriptors: list[dict[str, Any]] = []
        self._total = 0
        stores: Dict[str, VersionStore] = {}
        for item_index, raw_item in enumerate(items):
            item = dict(raw_item)
            version_id = item.get("dataset_version_id")
            if not version_id:
                self._descriptors.append({"kind": "item", "item": item})
                self._total += 1
                continue
            root = _dataset_root(item, request).resolve()
            store = stores.setdefault(str(root), VersionStore(root))
            dataset_id = str(item["dataset_id"]) if item.get("dataset_id") else None
            split = str(item.get("split") or "test")
            try:
                version = store.get_any(str(version_id), dataset_id=dataset_id)
            except VersionError as exc:
                raise EvaluationLabError(
                    f"cannot load Dataset Lab split {version_id}:{split}: {exc}"
                ) from exc
            if split not in version.split_counts:
                raise EvaluationLabError(
                    f"cannot load Dataset Lab split {version_id}:{split}: "
                    f"Dataset version has no split {split!r}"
                )
            limit_value = item.get("limit")
            limit = None if limit_value is None else max(0, int(limit_value))
            available = int(version.split_counts[split])
            selected = available if limit is None else min(available, limit)
            self._total += selected
            self._descriptors.append(
                {
                    "kind": "dataset",
                    "item": item,
                    "item_index": item_index,
                    "store": store,
                    "version": version,
                    "split": split,
                    "limit": limit,
                }
            )

    def __len__(self) -> int:
        return self._total

    def __iter__(self):
        from halo_forge.data_lab.errors import VersionError

        for descriptor in self._descriptors:
            item = descriptor["item"]
            if descriptor["kind"] == "item":
                yield dict(item)
                continue
            store = descriptor["store"]
            version = descriptor["version"]
            split = descriptor["split"]
            limit = descriptor["limit"]
            parent_item_id = str(
                item.get("id") or item.get("suite_item_id") or descriptor["item_index"]
            )
            if limit == 0:
                continue
            try:
                identified = store.iter_records_with_lineage(
                    version.version_id,
                    dataset_id=version.dataset_id,
                    split=split,
                )
                for ordinal, (raw_record, raw_identity) in enumerate(identified):
                    if limit is not None and ordinal >= limit:
                        break
                    record = dict(raw_record)
                    identity = raw_identity.to_dict()
                    schema = str(item.get("canonical_schema") or version.schema or "") or None
                    input_value, expected, canonical_metadata = _canonical_input_expected(
                        record, schema
                    )
                    instance_id = str(identity.get("instance_id") or ordinal)
                    occurrence_id = f"{parent_item_id}:{instance_id}"
                    metadata = {
                        **dict(item.get("metadata") or {}),
                        **canonical_metadata,
                        "canonical_record": record,
                        "dataset_id": version.dataset_id,
                        "dataset_version_id": version.version_id,
                        "dataset_split": split,
                        "dataset_version_path": version.path,
                        "dataset_assets_materialized": version.materialized_assets,
                        "record_hash": identity.get("record_hash"),
                        "instance_id": identity.get("instance_id"),
                        "dataset_suite_item_id": parent_item_id,
                        "dataset_split_ordinal": ordinal,
                    }
                    yield {
                        **item,
                        "id": occurrence_id,
                        "record_id": str(identity.get("record_id") or occurrence_id),
                        "input": input_value,
                        "expected": expected,
                        "metadata": metadata,
                        "_dataset_expanded": True,
                    }
            except VersionError as exc:
                raise EvaluationLabError(
                    f"cannot load Dataset Lab split " f"{version.version_id}:{split}: {exc}"
                ) from exc


def _expand_dataset_items(
    items: Sequence[Mapping[str, Any]], request: Mapping[str, Any]
) -> _ExpandedDatasetItems:
    """Return lazy Dataset Lab split expansion with an exact total."""

    return _ExpandedDatasetItems(items, request)


def _chat_prompt(messages: Sequence[Any]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, Mapping):
            role = message.get("role", message.get("from", "unknown"))
            content = message.get("content", message.get("value", ""))
            lines.append(f"<|{role}|>\n{content}")
        else:
            lines.append(str(message))
    lines.append("<|assistant|>\n")
    return "\n".join(lines)


def _generation_prompt(item: Mapping[str, Any]) -> str:
    metadata = dict(item.get("metadata") or {})
    schema = str(metadata.get("canonical_schema") or "").lower()
    if schema in {"vlm", "audio"}:
        raise EvaluationLabError(
            f"the local text serving adapter cannot evaluate {schema} inputs; "
            f"select a {schema} benchmark/evaluator or supply explicit evidence"
        )
    value = item.get("input")
    if value is None:
        raise EvaluationLabError("suite item has no input for model generation")
    if isinstance(value, Mapping) and ("image" in value or "audio" in value):
        modality = "image" if "image" in value else "audio"
        raise EvaluationLabError(
            f"the local text serving adapter cannot evaluate {modality} inputs; "
            "select a modality evaluator or supply explicit evidence"
        )
    if isinstance(value, list):
        prompt = _chat_prompt(value)
    elif isinstance(value, Mapping) and "messages" in value:
        tools = value.get("tools")
        prefix = ""
        if tools is not None:
            prefix = (
                "Available tools:\n"
                + json.dumps(tools, sort_keys=True, ensure_ascii=False, default=str)
                + "\n\n"
            )
        prompt = prefix + _chat_prompt(value.get("messages") or [])
    elif isinstance(value, str):
        prompt = value
    else:
        prompt = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    system = metadata.get("system")
    return f"{system}\n\n{prompt}" if system else prompt


def _subject_model_path(subject: ResolvedSubject, request: Mapping[str, Any]) -> str:
    """Resolve a subject to a local, revision-pinned model path when needed."""

    resolved = subject.payload.get("resolved_path")
    if resolved:
        return str(resolved)
    revision = subject.payload.get("revision")
    if not revision:
        return subject.subject_ref
    try:
        from huggingface_hub import snapshot_download

        return str(
            snapshot_download(
                repo_id=subject.subject_ref,
                revision=str(revision),
                local_files_only=bool(request.get("local_files_only", False)),
            )
        )
    except Exception as exc:
        raise EvaluationLabError(
            f"could not resolve pinned model {subject.subject_ref}@{revision}: {exc}"
        ) from exc


def _asset_path_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("path") or value.get("filename")
    return value


def _resolve_item_asset(
    item: Mapping[str, Any], request: Mapping[str, Any], field_name: str
) -> str:
    """Resolve and verify one local Dataset Lab media reference."""

    input_value = item.get("input")
    value = input_value.get(field_name) if isinstance(input_value, Mapping) else None
    if value is None:
        value = item.get(field_name)
    if value is None:
        raise EvaluationLabError(f"suite item has no {field_name} asset")

    metadata = dict(item.get("metadata") or {})
    version_path_value = metadata.get("dataset_version_path")
    if version_path_value:
        version_path = Path(str(version_path_value)).expanduser().resolve()
        try:
            manifest = json.loads((version_path / "manifest.json").read_text(encoding="utf-8"))
            from halo_forge.data_lab.training_artifacts import TrainingArtifactRenderer

            value = TrainingArtifactRenderer._resolve_asset_value(
                value,
                version_path=version_path,
                manifest=manifest,
                asset_roots=set(),
            )
        except Exception as exc:
            raise EvaluationLabError(
                f"cannot resolve Dataset Lab {field_name} asset: {exc}"
            ) from exc

    value = _asset_path_value(value)
    if not isinstance(value, (str, os.PathLike)):
        raise EvaluationLabError(f"suite item {field_name} asset is not a path")
    text_value = os.fspath(value)
    if text_value.startswith(("http://", "https://", "data:")):
        if field_name == "audio":
            raise EvaluationLabError("remote audio assets are not supported by the local adapter")
        return text_value

    candidate = Path(text_value).expanduser()
    if not candidate.is_absolute():
        asset_root = (
            item.get("asset_root") or metadata.get("asset_root") or request.get("asset_root")
        )
        if asset_root:
            candidate = Path(str(asset_root)).expanduser() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise EvaluationLabError(f"{field_name} asset does not exist: {candidate}")
    return str(candidate)


class _TextSubjectGenerator:
    """Lazily load one pinned text subject through Halo Forge serving."""

    def __init__(self, subject: ResolvedSubject, request: Mapping[str, Any]):
        self.subject = subject
        self.request = request
        self._adapter: Any = None
        self._load_error: Optional[str] = None

    def _load(self) -> Any:
        if self._adapter is not None:
            return self._adapter
        backend = self.request.get("backend")
        supported_backends = {
            None,
            "",
            "auto",
            "hf",
            "transformers",
            "torch",
            "cpu",
            "cuda",
            "rocm",
            "mps",
            "mlx",
        }
        if backend not in supported_backends:
            raise EvaluationLabError(
                f"the Dataset Lab text evaluator does not support backend {backend!r}"
            )
        model_name = _subject_model_path(self.subject, self.request)
        from halo_forge.serving.adapter import build_serving_adapter

        backend_name = None if backend in {None, "", "auto", "hf", "transformers"} else str(backend)
        self._adapter = build_serving_adapter(
            model_name,
            backend_name=backend_name,
            trust_remote_code=bool(self.request.get("trust_remote_code", False)),
        )
        return self._adapter

    def generate(self, item: Mapping[str, Any]) -> str:
        # Reject unsupported modalities before doing any heavyweight model load.
        prompt = _generation_prompt(item)
        if self._load_error is not None:
            raise EvaluationLabError(self._load_error)
        try:
            adapter = self._load()
        except Exception as exc:
            self._load_error = f"model subject could not be loaded: {type(exc).__name__}: {exc}"
            raise
        stop_value = self.request.get("stop")
        if isinstance(stop_value, str):
            stop = [stop_value]
        elif stop_value is None:
            stop = None
        else:
            stop = [str(value) for value in stop_value]
        output = adapter.generate(
            prompt,
            max_tokens=int(self.request.get("max_tokens", self.request.get("max_new_tokens", 256))),
            temperature=float(self.request.get("temperature", 0.0)),
            top_p=float(self.request.get("top_p", 1.0)),
            stop=stop,
        )
        if stop:
            cut = len(output)
            for marker in stop:
                position = output.find(marker)
                if position >= 0:
                    cut = min(cut, position)
            output = output[:cut]
        return output


def _resolve_score(
    *,
    item: Mapping[str, Any],
    item_id: str,
    subject: ResolvedSubject,
    scores: Mapping[str, Any],
    output: Any,
    expected: Any,
) -> Optional[float]:
    supplied = _mapped_value(scores, item, item_id)
    if supplied is not _MISSING:
        return float(supplied)
    score_by_subject = item.get("score_by_subject") or {}
    if subject.subject_ref in score_by_subject:
        return float(score_by_subject[subject.subject_ref])
    if item.get("score") is not None:
        return float(item["score"])
    if output is not _MISSING and expected is not None:
        return 1.0 if output == expected else 0.0
    return None


def _resolve_passed(
    item: Mapping[str, Any], score: Optional[float], direction: str, threshold: float
) -> Optional[bool]:
    if "passed" in item:
        return bool(item.get("passed"))
    if score is None:
        return None
    return score <= threshold if direction == "minimize" else score >= threshold


def _optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _evidence_fields(
    item: Mapping[str, Any],
    request: Mapping[str, Any],
    revision: BenchmarkSuiteRevisionRecord,
    *,
    evidence_kind: str,
    valid: bool,
    mineable: bool,
    threshold: Optional[float] = None,
    direction: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve common evidence provenance without fabricating measurements."""

    metadata = dict(item.get("metadata") or {})
    settings = dict(revision.generation_settings or {})
    usage = item.get("usage") or metadata.get("usage") or {}
    if not isinstance(usage, Mapping):
        usage = {}

    def first(*values: Any) -> Any:
        return next((value for value in values if value is not None), None)

    seed = _optional_int(
        first(
            item.get("generation_seed"),
            metadata.get("generation_seed"),
            request.get("generation_seed"),
            request.get("seed"),
            settings.get("seed"),
        )
    )
    runtime_versions: Dict[str, Any] = {}
    for values in (
        revision.evaluator_versions,
        request.get("runtime_versions"),
        metadata.get("runtime_versions"),
        item.get("runtime_versions"),
    ):
        if isinstance(values, Mapping):
            runtime_versions.update({str(key): value for key, value in values.items()})
    coverage_value = first(item.get("coverage"), metadata.get("coverage"))
    try:
        coverage = float(coverage_value) if coverage_value is not None else (1.0 if valid else 0.0)
    except (TypeError, ValueError):
        coverage = 1.0 if valid else 0.0
    if not 0.0 <= coverage <= 1.0:
        coverage = 1.0 if valid else 0.0
    return {
        "evidence_kind": evidence_kind,
        "valid": bool(valid),
        "mineable": bool(valid and mineable),
        "generation_seed": seed,
        "input_tokens": _optional_int(
            first(
                item.get("input_tokens"),
                item.get("prompt_tokens"),
                metadata.get("input_tokens"),
                metadata.get("prompt_tokens"),
                usage.get("input_tokens"),
                usage.get("prompt_tokens"),
            )
        ),
        "output_tokens": _optional_int(
            first(
                item.get("output_tokens"),
                item.get("completion_tokens"),
                metadata.get("output_tokens"),
                metadata.get("completion_tokens"),
                usage.get("output_tokens"),
                usage.get("completion_tokens"),
            )
        ),
        "finish_reason": (
            str(value)
            if (value := first(item.get("finish_reason"), metadata.get("finish_reason")))
            is not None
            else None
        ),
        "runtime_versions": runtime_versions,
        "score_direction": direction or revision.direction,
        "score_threshold": threshold,
        "coverage": coverage,
        "template_hash": (
            str(value)
            if (
                value := first(
                    item.get("template_hash"),
                    item.get("chat_template_hash"),
                    metadata.get("template_hash"),
                    metadata.get("chat_template_hash"),
                    request.get("template_hash"),
                    request.get("chat_template_hash"),
                )
            )
            is not None
            else None
        ),
    }


class DeterministicDatasetAdapter(EvaluationAdapter):
    """Per-record evaluator for Dataset Lab splits and deterministic fixtures.

    Each suite item may contain ``input``, ``expected``, ``record_id`` and an
    optional ``score_by_subject`` mapping. Callers can provide request-level
    ``outputs`` or ``scores`` mappings keyed by suite item/record ID. Items with
    ``dataset_version_id`` and ``split`` are expanded using immutable lineage.
    Missing evidence is generated against the selected text subject. Expected
    answers are never copied into model output.
    """

    adapter_id = "dataset"
    adapter_version = "3"

    def evaluate(
        self,
        context: EvaluationContext,
        revision: BenchmarkSuiteRevisionRecord,
        subject: ResolvedSubject,
        request: Mapping[str, Any],
    ) -> EvaluationAdapterResult:
        outputs = dict(request.get("outputs") or {})
        scores = dict(request.get("scores") or {})
        threshold = float(request.get("pass_threshold", 0.5))
        delay_ms = max(0.0, float(request.get("delay_ms", 0.0)))
        items = _expand_dataset_items(revision.items, request)
        samples: list[EvaluationSample] = []
        generator = _TextSubjectGenerator(subject, request)
        context.progress(stage="evaluating", processed=0, total=len(items))
        for index, item in enumerate(items):
            context.check_cancelled()
            item_id = str(item.get("id") or item.get("suite_item_id") or index)
            expected = item.get("expected")
            output = _mapped_value(outputs, item, item_id)
            if output is _MISSING and "output" in item:
                output = item.get("output")
            score = _resolve_score(
                item=item,
                item_id=item_id,
                subject=subject,
                scores=scores,
                output=output,
                expected=expected,
            )
            evidence_kind = "fixture"
            error: Optional[str] = None
            latency_ms = delay_ms
            # Explicit fixture scores remain useful for deterministic tests and
            # imported evidence; they do not require loading a model.
            if output is _MISSING and score is None:
                evidence_kind = "generated"
                started = time.perf_counter()
                try:
                    output = generator.generate(item)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    output = _MISSING
                latency_ms += (time.perf_counter() - started) * 1000.0
                if error is None:
                    score = _resolve_score(
                        item=item,
                        item_id=item_id,
                        subject=subject,
                        scores=scores,
                        output=output,
                        expected=expected,
                    )
            passed = False if error else _resolve_passed(item, score, revision.direction, threshold)
            valid = error is None and (output is not _MISSING or score is not None)
            if delay_ms:
                time.sleep(delay_ms / 1000.0)
            samples.append(
                EvaluationSample(
                    suite_item_id=item_id,
                    record_id=str(item.get("record_id") or item_id),
                    input=item.get("input"),
                    expected=expected,
                    output=None if output is _MISSING else output,
                    score=score,
                    passed=passed,
                    latency_ms=latency_ms,
                    error=error,
                    verifier_trace=item.get("verifier_trace"),
                    **_evidence_fields(
                        item,
                        request,
                        revision,
                        evidence_kind=evidence_kind,
                        valid=valid,
                        mineable=valid,
                        threshold=threshold,
                    ),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
            context.progress(processed=index + 1, total=len(items))
        scored = [sample.score for sample in samples if sample.score is not None]
        average = sum(scored) / len(scored) if scored else 0.0
        return EvaluationAdapterResult(
            metrics=[
                EvaluationMetric(
                    name=revision.primary_metric,
                    value=average,
                    direction=revision.direction,
                )
            ],
            samples=samples,
            summary={
                "passed": sum(sample.passed is True for sample in samples),
                "failed": sum(sample.passed is False for sample in samples),
                "unscored": sum(sample.passed is None for sample in samples),
                "errors": sum(bool(sample.error) for sample in samples),
                "valid_evidence": sum(sample.valid for sample in samples),
                "mineable_evidence": sum(sample.mineable for sample in samples),
                "total": len(samples),
            },
        )


class LegacyLMEvalAdapter(EvaluationAdapter):
    """Adapter over Halo Forge's existing lm-evaluation-harness wrapper.

    ``legacy_summary_path`` imports a historical ``lm_eval_summary.json`` as
    read-only evidence. Without it, this invokes the existing ``run_lm_eval``
    implementation and leaves its legacy files beneath the staging directory.
    """

    adapter_id = "lm-eval"
    adapter_version = "2"

    def _from_summary(
        self,
        summary: Mapping[str, Any],
        revision: BenchmarkSuiteRevisionRecord,
    ) -> EvaluationAdapterResult:
        metrics: list[EvaluationMetric] = []
        samples: list[EvaluationSample] = []
        for index, task in enumerate(summary.get("task_results") or []):
            task_id = str(task.get("task") or index)
            if task.get("error") is None:
                metrics.append(
                    EvaluationMetric(
                        name=str(task.get("primary_metric") or revision.primary_metric),
                        value=float(task.get("value") or 0.0),
                        direction="maximize" if task.get("higher_is_better", True) else "minimize",
                        suite_item_id=task_id,
                        metadata={"all_metrics": dict(task.get("all_metrics") or {})},
                    )
                )
            samples.append(
                EvaluationSample(
                    suite_item_id=task_id,
                    record_id=task_id,
                    score=float(task.get("value") or 0.0),
                    passed=None,
                    error=task.get("error"),
                    **_evidence_fields(
                        task,
                        {},
                        revision,
                        evidence_kind="legacy_aggregate",
                        valid=task.get("error") is None,
                        mineable=False,
                    ),
                    metadata={"aggregate_task_result": True},
                )
            )
        successful = [metric.value for metric in metrics]
        if successful:
            metrics.insert(
                0,
                EvaluationMetric(
                    name=revision.primary_metric,
                    value=sum(successful) / len(successful),
                    direction=revision.direction,
                ),
            )
        return EvaluationAdapterResult(metrics=metrics, samples=samples, summary=dict(summary))

    def evaluate(
        self,
        context: EvaluationContext,
        revision: BenchmarkSuiteRevisionRecord,
        subject: ResolvedSubject,
        request: Mapping[str, Any],
    ) -> EvaluationAdapterResult:
        legacy_path = request.get("legacy_summary_path")
        if legacy_path:
            path = Path(str(legacy_path)).expanduser().resolve()
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise EvaluationLabError(f"invalid legacy lm-eval summary {path}: {exc}") from exc
            context.progress(stage="importing", processed=1, total=1)
            return self._from_summary(summary, revision)

        from halo_forge.eval import run_lm_eval

        tasks = [
            str(item.get("task") or item.get("id") or item.get("suite_item_id") or "")
            for item in revision.items
        ]
        tasks = [task for task in tasks if task]
        context.check_cancelled()
        context.progress(stage="lm-eval", processed=0, total=len(tasks))
        result = run_lm_eval(
            model_name=_subject_model_path(subject, request),
            tasks=tasks,
            limit=request.get("limit"),
            batch_size=request.get("batch_size"),
            backend=str(request.get("backend") or "hf"),
            output_dir=context.work_dir / "legacy",
        )
        context.check_cancelled()
        context.progress(processed=len(tasks), total=len(tasks))
        return self._from_summary(result.to_dict(), revision)


class HaloForgeBenchmarkAdapter(EvaluationAdapter):
    """Adapter over Halo Forge's native code and VLMEvalKit benchmark router."""

    adapter_id = "benchmark"
    adapter_version = "2"

    def evaluate(
        self,
        context: EvaluationContext,
        revision: BenchmarkSuiteRevisionRecord,
        subject: ResolvedSubject,
        request: Mapping[str, Any],
    ) -> EvaluationAdapterResult:
        from halo_forge.benchmark import BenchmarkBackend, run_benchmark

        backend_name = str(request.get("backend") or "auto").lower()
        try:
            backend = BenchmarkBackend(backend_name)
        except ValueError as exc:
            raise EvaluationLabError(f"invalid benchmark backend: {backend_name}") from exc
        kwargs = dict(request.get("backend_args") or {})
        model = _subject_model_path(subject, request)
        metrics: list[EvaluationMetric] = []
        samples: list[EvaluationSample] = []
        primary_values: list[float] = []
        context.progress(stage="benchmarking", processed=0, total=len(revision.items))
        for index, item in enumerate(revision.items):
            context.check_cancelled()
            item_id = str(item.get("id") or item.get("suite_item_id") or index)
            benchmark_name = str(item.get("benchmark") or item.get("task") or item_id)
            item_backend_name = str(item.get("backend") or backend.value).lower()
            try:
                item_backend = BenchmarkBackend(item_backend_name)
            except ValueError as exc:
                raise EvaluationLabError(
                    f"invalid benchmark backend for {item_id}: {item_backend_name}"
                ) from exc
            result = run_benchmark(
                model=model,
                benchmark=benchmark_name,
                backend=item_backend,
                limit=item.get("limit", request.get("limit")),
                output=context.work_dir / f"benchmark-{index}.json",
                **{**kwargs, **dict(item.get("backend_args") or {})},
            )
            error = result.get("error")
            numeric: dict[str, float] = {}
            for name, value in dict(result.get("metrics") or {}).items():
                try:
                    numeric[str(name)] = float(value)
                except (TypeError, ValueError):
                    continue
            preferred = str(item.get("primary_metric") or revision.primary_metric)
            if preferred in numeric:
                primary = numeric[preferred]
            elif numeric:
                primary = next(iter(numeric.values()))
            else:
                primary = 0.0
            if not error:
                primary_values.append(primary)
            for name, value in numeric.items():
                metrics.append(
                    EvaluationMetric(
                        name=name,
                        value=value,
                        direction=str(item.get("direction") or revision.direction),
                        suite_item_id=item_id,
                        metadata={"benchmark": benchmark_name, "backend": result.get("backend")},
                    )
                )
            detailed_samples = result.get("sample_results")
            if isinstance(detailed_samples, list) and detailed_samples:
                for sample_index, raw_sample in enumerate(detailed_samples):
                    if not isinstance(raw_sample, Mapping):
                        continue
                    sample_metadata = dict(raw_sample.get("metadata") or {})
                    record_id_value = (
                        raw_sample.get("record_id")
                        or raw_sample.get("id")
                        or raw_sample.get("prompt_id")
                        or sample_metadata.get("record_id")
                        or sample_metadata.get("id")
                        or sample_metadata.get("task_id")
                    )
                    if record_id_value is None:
                        identity_payload = json.dumps(
                            {
                                "benchmark": benchmark_name,
                                "prompt": raw_sample.get("prompt", raw_sample.get("input")),
                                "metadata": sample_metadata,
                                "ordinal": sample_index,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            default=str,
                        )
                        record_id_value = (
                            "benchmark_"
                            + hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:24]
                        )
                    record_id = str(record_id_value)
                    sample_score_value = raw_sample.get("score")
                    if sample_score_value is None:
                        sample_score_value = raw_sample.get("reward")
                    sample_passed_value = raw_sample.get("passed")
                    if sample_passed_value is None and "success" in raw_sample:
                        sample_passed_value = raw_sample.get("success")
                    if sample_score_value is None and sample_passed_value is not None:
                        sample_score_value = 1.0 if sample_passed_value else 0.0
                    sample_score = (
                        float(sample_score_value) if sample_score_value is not None else None
                    )
                    sample_input = (
                        raw_sample.get("input")
                        if "input" in raw_sample
                        else raw_sample.get("prompt")
                    )
                    sample_expected = next(
                        (
                            raw_sample[name]
                            for name in ("expected", "ground_truth", "reference")
                            if name in raw_sample
                        ),
                        None,
                    )
                    sample_output = next(
                        (
                            raw_sample[name]
                            for name in ("output", "completion", "code")
                            if name in raw_sample
                        ),
                        None,
                    )
                    sample_error = str(raw_sample["error"]) if raw_sample.get("error") else None
                    valid_sample = sample_error is None and any(
                        value is not None
                        for value in (sample_score, sample_passed_value, sample_output)
                    )
                    threshold_value = item.get("pass_threshold", request.get("pass_threshold"))
                    threshold = float(threshold_value) if threshold_value is not None else None
                    evidence_source = {
                        **dict(item),
                        **dict(raw_sample),
                        "metadata": {
                            **dict(item.get("metadata") or {}),
                            **sample_metadata,
                        },
                    }
                    samples.append(
                        EvaluationSample(
                            suite_item_id=f"{item_id}:{record_id}",
                            record_id=record_id,
                            input=sample_input,
                            expected=sample_expected,
                            output=sample_output,
                            score=sample_score,
                            passed=(
                                bool(sample_passed_value)
                                if sample_passed_value is not None
                                else None
                            ),
                            latency_ms=(
                                float(raw_sample["latency_ms"])
                                if raw_sample.get("latency_ms") is not None
                                else None
                            ),
                            error=sample_error,
                            verifier_trace=raw_sample.get(
                                "verifier_trace",
                                raw_sample.get("verification_results"),
                            ),
                            **_evidence_fields(
                                evidence_source,
                                request,
                                revision,
                                evidence_kind="per_example",
                                valid=valid_sample,
                                mineable=valid_sample,
                                threshold=threshold,
                                direction=str(item.get("direction") or revision.direction),
                            ),
                            metadata={
                                **sample_metadata,
                                "benchmark": benchmark_name,
                                "backend": result.get("backend"),
                                "dataset_suite_item_id": item_id,
                                **(
                                    {"correct_count": raw_sample["correct_count"]}
                                    if "correct_count" in raw_sample
                                    else {}
                                ),
                            },
                        )
                    )
            else:
                aggregate_source = {
                    **dict(item),
                    **dict(result),
                    "metadata": {
                        **dict(item.get("metadata") or {}),
                        "benchmark": benchmark_name,
                    },
                }
                samples.append(
                    EvaluationSample(
                        suite_item_id=item_id,
                        record_id=item_id,
                        input={"benchmark": benchmark_name},
                        output=result,
                        score=primary,
                        passed=None,
                        error=str(error) if error else None,
                        **_evidence_fields(
                            aggregate_source,
                            request,
                            revision,
                            evidence_kind="aggregate_benchmark",
                            valid=not bool(error),
                            mineable=False,
                            direction=str(item.get("direction") or revision.direction),
                        ),
                        metadata={"aggregate_benchmark_result": True},
                    )
                )
            context.progress(processed=index + 1, total=len(revision.items))
        if primary_values:
            metrics.insert(
                0,
                EvaluationMetric(
                    name=revision.primary_metric,
                    value=sum(primary_values) / len(primary_values),
                    direction=revision.direction,
                ),
            )
        return EvaluationAdapterResult(
            metrics=metrics,
            samples=samples,
            summary={
                "benchmarks_completed": len(primary_values),
                "benchmarks_failed": len(revision.items) - len(primary_values),
                "evidence_samples": len(samples),
                "mineable_samples": sum(sample.mineable for sample in samples),
            },
        )


class VerifierEvaluationAdapter(EvaluationAdapter):
    """Score supplied model outputs through the existing verifier registry.

    This is the bridge for reasoning, structured tool-use and annotation
    evaluations whose suite items already contain (or request supplies) model
    outputs. When evidence is absent for a text item, generation uses the same
    pinned serving subject as the Dataset Lab evaluator.
    """

    adapter_id = "verifier"
    adapter_version = "3"

    def evaluate(
        self,
        context: EvaluationContext,
        revision: BenchmarkSuiteRevisionRecord,
        subject: ResolvedSubject,
        request: Mapping[str, Any],
    ) -> EvaluationAdapterResult:
        from halo_forge.rlvr.verifiers import get_verifier

        outputs = dict(request.get("outputs") or {})
        profile_revision_id = str(
            request.get("verifier_profile_revision_id") or ""
        ).strip()
        verifier_lab = None
        resolved_binding: Optional[Mapping[str, Any]] = None
        if profile_revision_id:
            contradictory = [
                key
                for key in ("verifier", "verifier_config", "pass_threshold", "reward_threshold")
                if request.get(key) not in (None, "", {})
            ]
            if contradictory:
                raise EvaluationLabError(
                    "verifier_profile_revision_id cannot be combined with raw fields: "
                    + ", ".join(contradictory)
                )
            from halo_forge.verifier_lab import VerifierLabService

            verifier_lab = VerifierLabService(context._manager.db)
            resolved_binding = verifier_lab.resolve_binding(profile_revision_id)
        default_name = request.get("verifier")
        default_config = dict(request.get("verifier_config") or {})
        items = _expand_dataset_items(revision.items, request)
        samples: list[EvaluationSample] = []
        generator = _TextSubjectGenerator(subject, request)
        context.progress(stage="verifying", processed=0, total=len(items))
        for index, item in enumerate(items):
            context.check_cancelled()
            item_id = str(item.get("id") or item.get("suite_item_id") or index)
            if profile_revision_id and any(
                item.get(key) not in (None, "", {})
                for key in ("verifier", "verifier_config", "threshold", "reward_threshold")
            ):
                raise EvaluationLabError(
                    f"suite item {item_id} contains raw verifier fields that conflict "
                    "with verifier_profile_revision_id"
                )
            verifier_name = str(
                (resolved_binding or {}).get("implementation_ref")
                or item.get("verifier")
                or default_name
                or ""
            )
            if not verifier_name and not profile_revision_id:
                raise EvaluationLabError(f"suite item {item_id} does not select a verifier")
            config = {**default_config, **dict(item.get("verifier_config") or {})}
            expected = item.get("expected")
            if verifier_name in {"bleu", "chrf"} and "references" not in config:
                config["references"] = expected
            elif verifier_name == "rouge" and "reference" not in config:
                config["reference"] = expected
            output = _mapped_value(outputs, item, item_id)
            if output is _MISSING and "output" in item:
                output = item.get("output")
            evidence_kind = "verified_fixture"
            generation_error: Optional[str] = None
            latency_ms: Optional[float] = None
            if output is _MISSING:
                evidence_kind = "verified_generation"
                started = time.perf_counter()
                try:
                    output = generator.generate(item)
                except Exception as exc:
                    generation_error = f"{type(exc).__name__}: {exc}"
                latency_ms = (time.perf_counter() - started) * 1000.0
            if generation_error is not None or output is _MISSING:
                samples.append(
                    EvaluationSample(
                        suite_item_id=item_id,
                        record_id=str(item.get("record_id") or item_id),
                        input=item.get("input"),
                        expected=expected,
                        output=None,
                        score=None,
                        passed=False,
                        latency_ms=latency_ms,
                        error=generation_error or "no output to verify",
                        **_evidence_fields(
                            item,
                            request,
                            revision,
                            evidence_kind=evidence_kind,
                            valid=False,
                            mineable=False,
                        ),
                        metadata={"verifier": verifier_name},
                    )
                )
                context.progress(processed=index + 1, total=len(items))
                continue
            try:
                if verifier_lab is not None:
                    observation = verifier_lab.invoke_revision(
                        profile_revision_id,
                        {
                            **dict(item),
                            "candidate": output,
                            "output": output,
                            "expected": expected,
                        },
                    )
                    verified = None
                else:
                    verifier_class = get_verifier(verifier_name)
                    verifier = verifier_class(**config)
                    verified = verifier.verify(str(output))
            except Exception as exc:
                samples.append(
                    EvaluationSample(
                        suite_item_id=item_id,
                        record_id=str(item.get("record_id") or item_id),
                        input=item.get("input"),
                        expected=expected,
                        output=output,
                        score=0.0,
                        passed=False,
                        latency_ms=latency_ms,
                        error=f"{type(exc).__name__}: {exc}",
                        **_evidence_fields(
                            item,
                            request,
                            revision,
                            evidence_kind=evidence_kind,
                            valid=False,
                            mineable=False,
                        ),
                        metadata={"verifier": verifier_name},
                    )
                )
            else:
                score = (
                    observation.reward
                    if verifier_lab is not None
                    else float(verified.reward)
                )
                passed = (
                    observation.passed
                    if verifier_lab is not None
                    else bool(verified.success)
                )
                verifier_error = (
                    observation.error if verifier_lab is not None else verified.error
                )
                trace = (
                    {
                        **observation.to_dict(),
                        "resolved_binding": dict(resolved_binding or {}),
                    }
                    if verifier_lab is not None
                    else {
                        "details": verified.details,
                        "reward": verified.reward,
                        "legacy_unqualified": True,
                        "warning": "Raw verifier configuration has no reliability qualification",
                    }
                )
                samples.append(
                    EvaluationSample(
                        suite_item_id=item_id,
                        record_id=str(item.get("record_id") or item_id),
                        input=item.get("input"),
                        expected=expected,
                        output=output,
                        score=score,
                        passed=passed,
                        latency_ms=latency_ms,
                        error=verifier_error,
                        verifier_trace=trace,
                        **_evidence_fields(
                            item,
                            request,
                            revision,
                            evidence_kind=evidence_kind,
                            valid=verifier_error is None,
                            mineable=verifier_error is None,
                        ),
                        metadata={"verifier": verifier_name},
                    )
                )
            context.progress(processed=index + 1, total=len(items))
        if verifier_lab is not None:
            verifier_lab.bind_revision(
                profile_revision_id,
                domain_kind="evaluation",
                domain_id=context.evaluation_id,
                role="scorer",
                context={"suite_revision_id": revision.id, "adapter_id": self.adapter_id},
            )
        average = sum(sample.score or 0.0 for sample in samples) / len(samples) if samples else 0.0
        return EvaluationAdapterResult(
            metrics=[
                EvaluationMetric(
                    name=revision.primary_metric,
                    value=average,
                    direction=revision.direction,
                )
            ],
            samples=samples,
            summary={
                "passed": sum(bool(sample.passed) for sample in samples),
                "total": len(samples),
            },
        )


class _VLMSubjectGenerator:
    def __init__(self, subject: ResolvedSubject, request: Mapping[str, Any]):
        self.subject = subject
        self.request = request
        self._adapter: Any = None

    def _load(self) -> Any:
        if self._adapter is None:
            from halo_forge.vlm.models import get_vlm_adapter

            kwargs: Dict[str, Any] = {
                "trust_remote_code": bool(self.request.get("trust_remote_code", False))
            }
            if self.request.get("device") is not None:
                kwargs["device"] = str(self.request["device"])
            self._adapter = get_vlm_adapter(
                _subject_model_path(self.subject, self.request),
                adapter_type=self.request.get("vlm_adapter_type")
                or self.request.get("model_adapter_type"),
                **kwargs,
            )
        return self._adapter

    def generate(self, item: Mapping[str, Any]) -> tuple[str, Dict[str, Any]]:
        input_value = item.get("input")
        prompt = (
            input_value.get("prompt") if isinstance(input_value, Mapping) else item.get("prompt")
        )
        if prompt is None:
            raise EvaluationLabError("VLM suite item has no prompt")
        image = _resolve_item_asset(item, self.request, "image")
        output = self._load().generate(
            image=image,
            prompt=str(prompt),
            max_new_tokens=int(
                self.request.get("max_new_tokens", self.request.get("max_tokens", 256))
            ),
            temperature=float(self.request.get("temperature", 0.0)),
            do_sample=bool(self.request.get("do_sample", False)),
        )
        text = output.text if hasattr(output, "text") else str(output)
        metadata = dict(getattr(output, "metadata", None) or {})
        metadata["resolved_image"] = image
        return str(text), metadata

    def cleanup(self) -> None:
        if self._adapter is not None:
            cleanup = getattr(self._adapter, "cleanup", None)
            if callable(cleanup):
                cleanup()


class _AudioSubjectGenerator:
    _SUPPORTED_TASKS = {
        "",
        "asr",
        "transcribe",
        "transcription",
        "speech-recognition",
        "speech_recognition",
        "speech-to-text",
        "speech_to_text",
    }

    def __init__(self, subject: ResolvedSubject, request: Mapping[str, Any]):
        self.subject = subject
        self.request = request
        self._adapter: Any = None

    def _load(self) -> Any:
        if self._adapter is None:
            from halo_forge.audio.models.adapters import get_audio_adapter

            self._adapter = get_audio_adapter(
                _subject_model_path(self.subject, self.request),
                device=(
                    str(self.request["device"]) if self.request.get("device") is not None else None
                ),
            )
        return self._adapter

    def generate(self, item: Mapping[str, Any]) -> tuple[str, Dict[str, Any]]:
        input_value = item.get("input")
        task = input_value.get("task") if isinstance(input_value, Mapping) else item.get("task")
        normalized_task = str(task or "").strip().lower()
        if normalized_task not in self._SUPPORTED_TASKS:
            raise EvaluationLabError(
                f"the local audio subject adapter does not support task {task!r}"
            )
        audio_path = _resolve_item_asset(item, self.request, "audio")
        adapter = self._load()
        target_rate = int(self.request.get("sample_rate", getattr(adapter, "sample_rate", 16000)))
        from halo_forge.audio.data.loaders import decode_audio

        waveform, sample_rate = decode_audio({"path": audio_path}, target_sr=target_rate)
        language = item.get("language", self.request.get("language"))
        result = adapter.transcribe(waveform, language=language)
        text = result.text if hasattr(result, "text") else str(result)
        metadata = {
            "resolved_audio": audio_path,
            "sample_rate": sample_rate,
        }
        for name in ("language", "confidence", "segments"):
            value = getattr(result, name, None)
            if value is not None:
                metadata[name] = value
        return str(text), metadata

    def cleanup(self) -> None:
        if self._adapter is not None:
            cleanup = getattr(self._adapter, "cleanup", None)
            if callable(cleanup):
                cleanup()


class SubjectModalityEvaluationAdapter(EvaluationAdapter):
    """Generate truthful VLM/audio evidence against the selected subject."""

    adapter_version = "4"

    def __init__(self, adapter_id: str):
        if adapter_id not in {"vlm", "audio"}:
            raise ValueError("subject modality adapter supports only vlm or audio")
        self.adapter_id = adapter_id

    def _score_generated(
        self,
        output: str,
        expected: Any,
        item: Mapping[str, Any],
        request: Mapping[str, Any],
        threshold: float,
        direction: str,
    ) -> tuple[Optional[float], Optional[bool], Any]:
        if expected is None:
            return None, None, None
        if self.adapter_id == "audio":
            from halo_forge.audio.verifiers.asr import ASRChecker

            checker = ASRChecker(
                wer_threshold=float(request.get("wer_threshold", 0.3)),
                use_cer=bool(request.get("use_cer", False)),
                normalize_text=bool(request.get("normalize_text", True)),
            )
            verified = checker.verify(output, str(expected))
            return (
                float(verified.reward),
                bool(verified.success),
                {"verifier": "asr", **dict(verified.details)},
            )

        from halo_forge.vlm.verifiers.output import OutputChecker

        checker = OutputChecker(
            fuzzy_threshold=float(request.get("fuzzy_threshold", 0.8)),
            use_semantic=bool(request.get("use_semantic", False)),
        )
        canonical = dict((item.get("metadata") or {}).get("canonical_record") or {})
        alternatives = canonical.get("alternatives") or item.get("alternatives")
        accepted = [str(expected), *[str(value) for value in alternatives or ()]]
        if any(checker.exact_match(output, value) for value in accepted):
            return (
                1.0,
                _resolve_passed(item, 1.0, direction, threshold),
                {
                    "verifier": "vlm_output",
                    "exact_match": True,
                    "fuzzy_score": 1.0,
                    "semantic_score": 0.0,
                    "format_score": 1.0,
                    "details": {"direct_normalized_exact_match": True},
                },
            )
        if alternatives:
            verified = checker.verify_with_alternatives(output, accepted)
        else:
            verified = checker.verify(
                output,
                str(expected),
                expected_format=item.get("expected_format"),
            )
        score = float(verified.overall_score)
        return (
            score,
            _resolve_passed(item, score, direction, threshold),
            {
                "verifier": "vlm_output",
                "exact_match": bool(verified.exact_match),
                "fuzzy_score": float(verified.fuzzy_score),
                "semantic_score": float(verified.semantic_score),
                "format_score": float(verified.format_score),
                "details": dict(verified.details),
            },
        )

    def evaluate(
        self,
        context: EvaluationContext,
        revision: BenchmarkSuiteRevisionRecord,
        subject: ResolvedSubject,
        request: Mapping[str, Any],
    ) -> EvaluationAdapterResult:
        if self.adapter_id == "vlm" and all(
            (item.get("benchmark") or item.get("task"))
            and not item.get("dataset_version_id")
            and not (
                isinstance(item.get("input"), Mapping) and item["input"].get("image") is not None
            )
            for item in revision.items
        ):
            routed_request = dict(request)
            routed_request.setdefault("backend", "vlmevalkit")
            return HaloForgeBenchmarkAdapter().evaluate(context, revision, subject, routed_request)
        outputs = dict(request.get("outputs") or {})
        scores = dict(request.get("scores") or {})
        threshold = float(request.get("pass_threshold", 0.5))
        items = _expand_dataset_items(revision.items, request)
        generator: Any = (
            _VLMSubjectGenerator(subject, request)
            if self.adapter_id == "vlm"
            else _AudioSubjectGenerator(subject, request)
        )
        samples: list[EvaluationSample] = []
        context.progress(stage=f"{self.adapter_id}:evaluating", processed=0, total=len(items))
        try:
            for index, item in enumerate(items):
                context.check_cancelled()
                item_id = str(item.get("id") or item.get("suite_item_id") or index)
                expected = item.get("expected", item.get("reference"))
                output = _mapped_value(outputs, item, item_id)
                if output is _MISSING and "output" in item:
                    output = item.get("output")
                score = _resolve_score(
                    item=item,
                    item_id=item_id,
                    subject=subject,
                    scores=scores,
                    output=output,
                    expected=expected,
                )
                error = str(item["error"]) if item.get("error") else None
                latency_ms: Optional[float] = None
                generated_metadata: Dict[str, Any] = {}
                verifier_trace = item.get("verifier_trace")
                generated_passed: Optional[bool] = None
                evidence_kind = "modality_fixture"
                if output is _MISSING and score is None and error is None:
                    evidence_kind = "modality_generation"
                    started = time.perf_counter()
                    try:
                        output, generated_metadata = generator.generate(item)
                        score, generated_passed, verifier_trace = self._score_generated(
                            str(output),
                            expected,
                            item,
                            request,
                            threshold,
                            revision.direction,
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        output = _MISSING
                    latency_ms = (time.perf_counter() - started) * 1000.0
                passed = (
                    False
                    if error
                    else (
                        generated_passed
                        if generated_passed is not None
                        else _resolve_passed(item, score, revision.direction, threshold)
                    )
                )
                valid = error is None and (output is not _MISSING or score is not None)
                evidence_source = {
                    **dict(item),
                    "metadata": {
                        **dict(item.get("metadata") or {}),
                        **generated_metadata,
                    },
                }
                samples.append(
                    EvaluationSample(
                        suite_item_id=item_id,
                        record_id=str(item.get("record_id") or item_id),
                        input=item.get("input", item.get("prompt")),
                        expected=expected,
                        output=None if output is _MISSING else output,
                        score=score,
                        passed=passed,
                        latency_ms=latency_ms,
                        error=error,
                        verifier_trace=verifier_trace,
                        **_evidence_fields(
                            evidence_source,
                            request,
                            revision,
                            evidence_kind=evidence_kind,
                            valid=valid,
                            mineable=valid,
                            threshold=threshold,
                        ),
                        metadata={
                            "modality": self.adapter_id,
                            **dict(item.get("metadata") or {}),
                            **generated_metadata,
                        },
                    )
                )
                context.progress(processed=index + 1, total=len(items))
        finally:
            try:
                generator.cleanup()
            except Exception as exc:
                context.log(
                    f"{self.adapter_id} subject cleanup failed: " f"{type(exc).__name__}: {exc}"
                )

        scored = [sample.score for sample in samples if sample.score is not None]
        average = sum(scored) / len(scored) if scored else 0.0
        return EvaluationAdapterResult(
            metrics=[EvaluationMetric(revision.primary_metric, average, revision.direction)],
            samples=samples,
            summary={
                "adapter": self.adapter_id,
                "total": len(samples),
                "errors": sum(bool(sample.error) for sample in samples),
                "scored": len(scored),
                "passed": sum(sample.passed is True for sample in samples),
            },
        )


class EvidenceEvaluationAdapter(EvaluationAdapter):
    """Standardize already-generated modality evidence without new generation."""

    adapter_id = "evidence"
    adapter_version = "3"

    def __init__(self, adapter_id: str = "evidence"):
        self.adapter_id = adapter_id

    def evaluate(
        self,
        context: EvaluationContext,
        revision: BenchmarkSuiteRevisionRecord,
        subject: ResolvedSubject,
        request: Mapping[str, Any],
    ) -> EvaluationAdapterResult:
        outputs = dict(request.get("outputs") or {})
        scores = dict(request.get("scores") or {})
        threshold = float(request.get("pass_threshold", 0.5))
        items = _expand_dataset_items(revision.items, request)
        samples: list[EvaluationSample] = []
        context.progress(stage=f"{self.adapter_id}:evidence", processed=0, total=len(items))
        for index, item in enumerate(items):
            context.check_cancelled()
            item_id = str(item.get("id") or item.get("suite_item_id") or index)
            expected = item.get("expected", item.get("reference"))
            output = _mapped_value(outputs, item, item_id)
            if output is _MISSING and "output" in item:
                output = item.get("output")
            score = _resolve_score(
                item=item,
                item_id=item_id,
                subject=subject,
                scores=scores,
                output=output,
                expected=expected,
            )
            error = item.get("error")
            if output is _MISSING and score is None and not error:
                error = (
                    f"no {self.adapter_id} evaluator output or score was supplied; "
                    "this adapter cannot generate modality evidence"
                )
            passed = False if error else _resolve_passed(item, score, revision.direction, threshold)
            valid = not error and (output is not _MISSING or score is not None)
            samples.append(
                EvaluationSample(
                    suite_item_id=item_id,
                    record_id=str(item.get("record_id") or item_id),
                    input=item.get("input", item.get("prompt")),
                    expected=expected,
                    output=None if output is _MISSING else output,
                    score=score,
                    passed=passed,
                    error=str(error) if error else None,
                    verifier_trace=item.get("verifier_trace"),
                    **_evidence_fields(
                        item,
                        request,
                        revision,
                        evidence_kind="imported_per_example",
                        valid=valid,
                        mineable=valid,
                        threshold=threshold,
                    ),
                    metadata={"modality": self.adapter_id, **dict(item.get("metadata") or {})},
                )
            )
            context.progress(processed=index + 1, total=len(items))
        scored = [sample.score for sample in samples if sample.score is not None]
        average = sum(scored) / len(scored) if scored else 0.0
        return EvaluationAdapterResult(
            metrics=[EvaluationMetric(revision.primary_metric, average, revision.direction)],
            samples=samples,
            summary={
                "adapter": self.adapter_id,
                "total": len(samples),
                "errors": sum(bool(sample.error) for sample in samples),
                "scored": len(scored),
            },
        )


class ModalityEvaluationAdapter(EvaluationAdapter):
    """Route an advertised modality to existing evaluators when available."""

    adapter_version = "3"

    def __init__(self, adapter_id: str):
        self.adapter_id = adapter_id
        self._evidence = EvidenceEvaluationAdapter(adapter_id)

    def evaluate(
        self,
        context: EvaluationContext,
        revision: BenchmarkSuiteRevisionRecord,
        subject: ResolvedSubject,
        request: Mapping[str, Any],
    ) -> EvaluationAdapterResult:
        if self.adapter_id in {"code", "vlm"} and all(
            item.get("benchmark") or item.get("task") for item in revision.items
        ):
            routed_request = dict(request)
            if self.adapter_id == "vlm":
                routed_request.setdefault("backend", "vlmevalkit")
            return HaloForgeBenchmarkAdapter().evaluate(context, revision, subject, routed_request)
        if self.adapter_id == "reasoning" and all(item.get("task") for item in revision.items):
            return LegacyLMEvalAdapter().evaluate(context, revision, subject, request)
        if self.adapter_id == "tool" and all(item.get("verifier") for item in revision.items):
            return VerifierEvaluationAdapter().evaluate(context, revision, subject, request)
        if self.adapter_id in {"reasoning", "tool"}:
            return DeterministicDatasetAdapter().evaluate(context, revision, subject, request)
        return self._evidence.evaluate(context, revision, subject, request)


class CompositeSuiteAdapter(EvaluationAdapter):
    """Execute heterogeneous ordered suite items through their selected adapters."""

    adapter_id = "suite"
    adapter_version = "2"

    def __init__(self, registry: EvaluationAdapterRegistry):
        self.registry = registry

    def version_for(self, revision: BenchmarkSuiteRevisionRecord) -> str:
        identities = []
        for item in revision.items:
            adapter_id = adapter_for_item(item)
            if adapter_id == self.adapter_id:
                raise EvaluationLabError("suite items cannot recursively select the suite adapter")
            adapter = self.registry.get(adapter_id)
            identities.append((str(item.get("id")), adapter_id, adapter.adapter_version))
        payload = json.dumps(identities, sort_keys=True, separators=(",", ":"))
        return f"2+{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"

    def evaluate(
        self,
        context: EvaluationContext,
        revision: BenchmarkSuiteRevisionRecord,
        subject: ResolvedSubject,
        request: Mapping[str, Any],
    ) -> EvaluationAdapterResult:
        grouped: Dict[str, list[dict[str, Any]]] = {}
        ordered_ids: list[str] = []
        for index, raw_item in enumerate(revision.items):
            item = dict(raw_item)
            item_id = str(item.get("id") or item.get("suite_item_id") or index)
            item["id"] = item_id
            ordered_ids.append(item_id)
            adapter_id = adapter_for_item(item)
            if adapter_id == self.adapter_id:
                raise EvaluationLabError("suite items cannot recursively select the suite adapter")
            grouped.setdefault(adapter_id, []).append(item)

        adapter_requests = dict(request.get("adapter_requests") or {})
        common_request = {key: value for key, value in request.items() if key != "adapter_requests"}
        group_requests: Dict[str, Dict[str, Any]] = {}
        group_totals: Dict[str, int] = {}
        for adapter_id, items in grouped.items():
            specific = adapter_requests.get(adapter_id) or adapter_requests.get(
                next(
                    (alias for alias, target in ADAPTER_ALIASES.items() if target == adapter_id),
                    "",
                )
            )
            adapter_request = {**common_request, **dict(specific or {})}
            group_requests[adapter_id] = adapter_request
            group_totals[adapter_id] = (
                len(_expand_dataset_items(items, adapter_request))
                if any(item.get("dataset_version_id") for item in items)
                else len(items)
            )
        overall_total = sum(group_totals.values())

        class _GroupContext:
            def __init__(self, parent: EvaluationContext, offset: int):
                self._parent = parent
                self._offset = offset
                self.evaluation_id = parent.evaluation_id
                self.work_dir = parent.work_dir

            def check_cancelled(self) -> None:
                self._parent.check_cancelled()

            def log(self, message: str) -> None:
                self._parent.log(message)

            def progress(
                self,
                *,
                processed: Optional[int] = None,
                total: Optional[int] = None,
                stage: Optional[str] = None,
            ) -> None:
                self._parent.progress(
                    processed=(self._offset + processed if processed is not None else None),
                    total=overall_total,
                    stage=stage,
                )

        all_metrics: list[EvaluationMetric] = []
        samples_by_item: Dict[str, list[EvaluationSample]] = {}
        summaries: Dict[str, Any] = {}
        processed = 0
        context.progress(stage="suite", processed=0, total=overall_total)
        for adapter_id, items in grouped.items():
            adapter = self.registry.get(adapter_id)
            context.log(f"suite group {adapter_id}@{adapter.adapter_version}: {len(items)} item(s)")
            subset = BenchmarkSuiteRevisionRecord(
                id=revision.id,
                suite_id=revision.suite_id,
                revision_number=revision.revision_number,
                content_hash=revision.content_hash,
                items_json=json.dumps(items, sort_keys=True),
                generation_settings_json=revision.generation_settings_json,
                evaluator_versions_json=revision.evaluator_versions_json,
                primary_metric=revision.primary_metric,
                direction=revision.direction,
                created_at=revision.created_at,
            )
            adapter_request = group_requests[adapter_id]
            group_context = _GroupContext(context, processed)
            result = adapter.evaluate(group_context, subset, subject, adapter_request)
            summaries[adapter_id] = result.summary
            for metric in result.metrics:
                if metric.suite_item_id:
                    all_metrics.append(metric)
            for sample in result.samples:
                parent_item_id = str(
                    sample.metadata.get("dataset_suite_item_id") or sample.suite_item_id
                )
                samples_by_item.setdefault(parent_item_id, []).append(sample)
            actual_group_total = len(result.samples)
            if actual_group_total != group_totals[adapter_id]:
                overall_total += actual_group_total - group_totals[adapter_id]
                group_totals[adapter_id] = actual_group_total
            processed += actual_group_total
            context.progress(
                stage=f"suite:{adapter_id}",
                processed=processed,
                total=overall_total,
            )

        samples: list[EvaluationSample] = []
        for item_id in ordered_ids:
            selected = samples_by_item.get(item_id) or []
            if selected:
                samples.extend(selected)
            else:
                samples.append(
                    EvaluationSample(
                        suite_item_id=item_id,
                        record_id=item_id,
                        passed=False,
                        error="adapter_did_not_return_sample",
                        evidence_kind="missing",
                        valid=False,
                        mineable=False,
                        score_direction=revision.direction,
                        coverage=0.0,
                    )
                )
        scored = [sample.score for sample in samples if sample.score is not None]
        primary = sum(scored) / len(scored) if scored else 0.0
        all_metrics.insert(
            0,
            EvaluationMetric(
                name=revision.primary_metric,
                value=primary,
                direction=revision.direction,
            ),
        )
        return EvaluationAdapterResult(
            metrics=all_metrics,
            samples=samples,
            summary={"adapters": summaries, "total": len(samples)},
        )


class OperationalPerformanceEvaluationAdapter(EvaluationAdapter):
    """Fixed-policy local inference performance measurements for qualification."""

    adapter_id = "performance"
    adapter_version = "1"
    _DIRECTIONS = {
        "load_time_ms": "minimize",
        "time_to_first_token_ms": "minimize",
        "total_latency_ms": "minimize",
        "output_tokens_per_second": "maximize",
        "peak_process_memory_bytes": "minimize",
        "peak_system_memory_bytes": "minimize",
        "peak_device_memory_bytes": "minimize",
        "artifact_size_bytes": "minimize",
        "error_rate": "minimize",
    }

    @staticmethod
    def _artifact_size(path: Path) -> int:
        if path.is_file():
            return int(path.stat().st_size)
        return sum(int(value.stat().st_size) for value in path.rglob("*") if value.is_file())

    def evaluate(
        self,
        context: EvaluationContext,
        revision: BenchmarkSuiteRevisionRecord,
        subject: ResolvedSubject,
        request: Mapping[str, Any],
    ) -> EvaluationAdapterResult:
        from halo_forge.qualification_lab import (
            InferencePerformanceAdapter,
            PerformanceSettings,
        )
        from halo_forge.serving.adapter import build_serving_adapter
        from halo_forge.workstation_jobs.resources import sample_workstation_capacity

        subject_path = Path(
            str(subject.payload.get("resolved_path") or subject.subject_ref)
        ).expanduser()
        if not subject_path.exists():
            raise EvaluationLabError(
                f"performance evaluation artifact does not exist: {subject_path}"
            )
        item = dict(revision.items[0]) if revision.items else {}
        prompt_value = item.get("prompt", item.get("input", "Hello"))
        if isinstance(prompt_value, Mapping):
            prompt_value = prompt_value.get("prompt") or json.dumps(prompt_value, sort_keys=True)
        prompt = str(prompt_value)
        generation = dict(request.get("generation_settings") or {})
        backend_value = request.get("backend") or subject.payload.get("backend")
        backend = str(backend_value) if backend_value else None
        settings = PerformanceSettings.from_dict(request.get("performance_settings") or {})
        artifact_size = self._artifact_size(subject_path)
        loaded: Dict[str, Any] = {}
        completed_runs = 0

        def runner(run_request: Any) -> Mapping[str, Any]:
            nonlocal completed_runs
            context.check_cancelled()
            if "adapter" not in loaded:
                load_started = time.perf_counter()
                loaded["adapter"] = build_serving_adapter(str(subject_path), backend_name=backend)
                # Model load is a property of the evaluated subject/runtime, not
                # just the first warmup sample. Carry it into measured samples so
                # the immutable aggregate can report it without counting a
                # warmup as a measured repetition.
                loaded["load_time_ms"] = (time.perf_counter() - load_started) * 1000.0
            adapter = loaded["adapter"]
            random.seed(run_request.generation_seed)
            try:
                import torch

                torch.manual_seed(run_request.generation_seed)
            except Exception:
                pass
            before = sample_workstation_capacity(
                subject_path, pid=os.getpid(), include_accelerator=True
            )
            started = time.perf_counter()
            output = adapter.generate(
                prompt,
                max_tokens=int(generation.get("max_tokens", 128)),
                temperature=float(generation.get("temperature", 0.0)),
                top_p=float(generation.get("top_p", 1.0)),
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            after = sample_workstation_capacity(
                subject_path, pid=os.getpid(), include_accelerator=True
            )
            output_tokens = None
            tokenizer = getattr(adapter, "_tokenizer", None)
            if tokenizer is not None:
                try:
                    output_tokens = len(tokenizer.encode(output, add_special_tokens=False))
                except Exception:
                    output_tokens = None
            throughput = (
                output_tokens / (latency_ms / 1000.0)
                if output_tokens is not None and latency_ms > 0
                else None
            )
            process_values = [
                sample.process.rss_bytes
                for sample in (before, after)
                if sample.process and sample.process.rss_bytes is not None
            ]
            system_values = [
                sample.memory.used_bytes
                for sample in (before, after)
                if sample.memory and sample.memory.used_bytes is not None
            ]
            device_values = [
                sample.accelerator.device_memory_used_bytes
                for sample in (before, after)
                if sample.accelerator and sample.accelerator.device_memory_used_bytes is not None
            ]
            completed_runs += 1
            context.progress(
                stage="performance:measuring",
                processed=completed_runs,
                total=settings.warmup_runs + settings.measured_repeats,
            )
            return {
                "load_time_ms": loaded.get("load_time_ms"),
                "time_to_first_token_ms": None,
                "total_latency_ms": latency_ms,
                "output_tokens": output_tokens,
                "output_tokens_per_second": throughput,
                "peak_process_memory_bytes": max(process_values) if process_values else None,
                "peak_system_memory_bytes": max(system_values) if system_values else None,
                "peak_device_memory_bytes": max(device_values) if device_values else None,
                "artifact_size_bytes": artifact_size,
                "runtime_versions": {
                    "python": platform.python_version(),
                    "adapter": type(adapter).__name__,
                },
                "hardware_identity": {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "accelerator": (after.accelerator.to_dict() if after.accelerator else None),
                },
            }

        total = settings.warmup_runs + settings.measured_repeats
        context.progress(stage="performance:measuring", processed=0, total=total)
        aggregate = InferencePerformanceAdapter(runner, settings=settings).run(
            artifact_ref=str(subject_path),
            backend=backend or "auto",
            prompt=prompt,
            generation_settings=generation,
            artifact_size_bytes=artifact_size,
        )
        values = aggregate.metric_values()
        values["error_rate"] = (
            aggregate.failed_count / aggregate.measured_count if aggregate.measured_count else 1.0
        )
        metrics = [
            EvaluationMetric(
                name=name,
                value=float(value),
                direction=self._DIRECTIONS[name],
            )
            for name, value in values.items()
            if value is not None and name in self._DIRECTIONS
        ]
        if revision.primary_metric not in {metric.name for metric in metrics}:
            raise EvaluationLabError(
                "operational suite primary metric is unavailable under this runtime: "
                f"{revision.primary_metric}"
            )
        samples = [
            EvaluationSample(
                suite_item_id=f"performance-{sample.phase}-{sample.iteration}",
                record_id=f"performance-{sample.phase}-{sample.iteration}",
                input={
                    "prompt": prompt,
                    "phase": sample.phase,
                    "generation_seed": sample.generation_seed,
                },
                output=sample.to_dict(),
                latency_ms=sample.total_latency_ms,
                error=sample.error,
                evidence_kind="operational_measurement",
                valid=sample.successful,
                mineable=False,
                generation_seed=sample.generation_seed,
                output_tokens=sample.output_tokens,
                runtime_versions=dict(sample.runtime_versions),
                metadata={"hardware_identity": dict(sample.hardware_identity)},
            )
            for sample in aggregate.samples
        ]
        context.progress(stage="performance:complete", processed=total, total=total)
        return EvaluationAdapterResult(
            metrics=metrics,
            samples=samples,
            summary={"adapter": self.adapter_id, "performance": aggregate.to_dict()},
        )


def default_adapter_registry() -> EvaluationAdapterRegistry:
    registry = EvaluationAdapterRegistry(
        (
            DeterministicDatasetAdapter(),
            LegacyLMEvalAdapter(),
            HaloForgeBenchmarkAdapter(),
            VerifierEvaluationAdapter(),
            ModalityEvaluationAdapter("code"),
            ModalityEvaluationAdapter("reasoning"),
            ModalityEvaluationAdapter("tool"),
            SubjectModalityEvaluationAdapter("vlm"),
            SubjectModalityEvaluationAdapter("audio"),
            OperationalPerformanceEvaluationAdapter(),
        )
    )
    registry.register(CompositeSuiteAdapter(registry))
    return registry


__all__ = [
    "DeterministicDatasetAdapter",
    "ADAPTER_ALIASES",
    "CompositeSuiteAdapter",
    "EvaluationAdapter",
    "EvaluationAdapterRegistry",
    "EvaluationContext",
    "EvidenceEvaluationAdapter",
    "HaloForgeBenchmarkAdapter",
    "LegacyLMEvalAdapter",
    "ModalityEvaluationAdapter",
    "OperationalPerformanceEvaluationAdapter",
    "SubjectModalityEvaluationAdapter",
    "VerifierEvaluationAdapter",
    "adapter_for_item",
    "canonical_adapter_id",
    "default_adapter_registry",
    "infer_suite_adapter",
]
