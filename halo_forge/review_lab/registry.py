"""Versioned annotation-task and Dataset Lab output adapter registries."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .errors import ReviewValidationError

MODALITY_TASKS: Dict[str, tuple[str, ...]] = {
    "text": (
        "binary",
        "categorical",
        "multi_label",
        "scalar",
        "text_correction",
        "pairwise",
        "ranking",
    ),
    "preference": ("binary", "categorical", "scalar", "pairwise", "ranking"),
    "tool": ("binary", "categorical", "scalar", "structured_correction"),
    "vlm": (
        "binary",
        "categorical",
        "multi_label",
        "scalar",
        "text_correction",
        "pairwise",
        "ranking",
    ),
    "audio": ("binary", "categorical", "multi_label", "scalar", "text_correction"),
}


def normalize_modality(value: str) -> str:
    modality = str(value or "").strip().lower().replace("-", "_")
    aliases = {"image": "vlm", "vision": "vlm", "tool_use": "tool"}
    modality = aliases.get(modality, modality)
    if modality not in MODALITY_TASKS:
        raise ReviewValidationError(
            f"unsupported review modality {value!r}; choose from {sorted(MODALITY_TASKS)}"
        )
    return modality


def normalize_task_type(value: str) -> str:
    task = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "accept_reject": "binary",
        "classification": "categorical",
        "multilabel": "multi_label",
        "score": "scalar",
        "correction": "text_correction",
        "tool_correction": "structured_correction",
        "preference": "pairwise",
    }
    return aliases.get(task, task)


def validate_schema_definition(
    modality: str, task_type: str, definition: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    resolved_modality = normalize_modality(modality)
    resolved_task = normalize_task_type(task_type)
    if resolved_task not in MODALITY_TASKS[resolved_modality]:
        raise ReviewValidationError(
            f"task {resolved_task!r} is not supported for {resolved_modality}; "
            f"choose from {list(MODALITY_TASKS[resolved_modality])}"
        )
    result = copy.deepcopy(dict(definition or {}))
    result["modality"] = resolved_modality
    result["task_type"] = resolved_task
    if resolved_task == "categorical":
        labels = result.get("labels")
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)) or not labels:
            raise ReviewValidationError("categorical schemas require a non-empty labels list")
        normalized = [str(label) for label in labels]
        if len(set(normalized)) != len(normalized):
            raise ReviewValidationError("categorical schema labels must be unique")
        result["labels"] = normalized
    if resolved_task == "multi_label" and result.get("labels") is not None:
        labels = result["labels"]
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise ReviewValidationError("multi-label schema labels must be a list")
        result["labels"] = [str(label) for label in labels]
    if resolved_task == "scalar":
        lower = float(result.get("minimum", 0.0))
        upper = float(result.get("maximum", 1.0))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ReviewValidationError("scalar schema minimum must be below maximum")
        result.update(minimum=lower, maximum=upper)
    adapter_id = str(result.get("output_adapter_id") or "").strip()
    if not adapter_id:
        adapter_id = default_output_adapter_id(resolved_modality, resolved_task)
    result["output_adapter_id"] = adapter_id
    return result


def normalize_annotation(
    definition: Mapping[str, Any], payload: Mapping[str, Any]
) -> Dict[str, Any]:
    task = normalize_task_type(str(definition.get("task_type") or ""))
    raw = copy.deepcopy(dict(payload or {}))
    annotation = raw.get("annotation", raw)
    if not isinstance(annotation, Mapping):
        if task == "binary":
            annotation = {"accepted": annotation}
        elif task == "categorical":
            annotation = {"label": annotation}
        elif task == "scalar":
            annotation = {"score": annotation}
        elif task == "text_correction":
            annotation = {"corrected_text": annotation}
        else:
            raise ReviewValidationError("annotation must be an object")
    value = copy.deepcopy(dict(annotation))

    if task == "binary":
        accepted = value.get("accepted", value.get("value", value.get("label")))
        if isinstance(accepted, str):
            normalized = accepted.strip().lower()
            if normalized in {"accept", "accepted", "yes", "true", "1", "pass"}:
                accepted = True
            elif normalized in {"reject", "rejected", "no", "false", "0", "fail"}:
                accepted = False
        if not isinstance(accepted, bool):
            raise ReviewValidationError("binary annotations require accepted=true or false")
        value = {**value, "accepted": accepted}
    elif task == "categorical":
        label = str(value.get("label", value.get("value", "")))
        labels = [str(item) for item in definition.get("labels") or []]
        if label not in labels:
            raise ReviewValidationError(f"categorical label must be one of {labels}")
        value = {**value, "label": label}
    elif task == "multi_label":
        labels = value.get("labels", value.get("value"))
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise ReviewValidationError("multi-label annotations require a labels list")
        normalized_labels = [str(item) for item in labels]
        allowed = [str(item) for item in definition.get("labels") or []]
        if allowed and any(item not in allowed for item in normalized_labels):
            raise ReviewValidationError(f"multi-label values must be drawn from {allowed}")
        value = {**value, "labels": list(dict.fromkeys(normalized_labels))}
    elif task == "scalar":
        try:
            score = float(value.get("score", value.get("value")))
        except (TypeError, ValueError) as exc:
            raise ReviewValidationError("scalar annotations require a numeric score") from exc
        lower = float(definition.get("minimum", 0.0))
        upper = float(definition.get("maximum", 1.0))
        if not math.isfinite(score) or not lower <= score <= upper:
            raise ReviewValidationError(f"scalar score must be between {lower} and {upper}")
        value = {**value, "score": score}
    elif task == "text_correction":
        corrected = value.get(
            "corrected_text",
            value.get("response", value.get("transcript", value.get("text"))),
        )
        if not isinstance(corrected, str) or not corrected.strip():
            raise ReviewValidationError("text corrections require non-empty corrected_text")
        value = {**value, "corrected_text": corrected}
    elif task == "structured_correction":
        correction = value.get("correction", value.get("value", value))
        if not isinstance(correction, Mapping) or not correction:
            raise ReviewValidationError("structured corrections require a correction object")
        value = {**value, "correction": copy.deepcopy(dict(correction))}
    elif task == "pairwise":
        chosen = value.get("chosen", value.get("choice"))
        if chosen is None:
            raise ReviewValidationError("pairwise annotations require chosen or choice")
        value = {**value, "chosen": copy.deepcopy(chosen)}
        if value.get("rejected") == value["chosen"]:
            raise ReviewValidationError("chosen and rejected cannot be identical")
    elif task == "ranking":
        ranking = value.get("ranking", value.get("value"))
        if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
            raise ReviewValidationError("ranking annotations require an ordered ranking list")
        ranking_values = list(ranking)
        if len(ranking_values) < 2 or len({str(item) for item in ranking_values}) != len(
            ranking_values
        ):
            raise ReviewValidationError("ranking must contain at least two unique entries")
        value = {**value, "ranking": copy.deepcopy(ranking_values)}
    else:
        raise ReviewValidationError(f"unsupported annotation task {task!r}")
    return value


@dataclass(frozen=True)
class ReviewOutputAdapter:
    id: str
    version: int
    modalities: tuple[str, ...]
    task_types: tuple[str, ...]
    build_modes: tuple[str, ...]
    default_build_mode: str

    def descriptor(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "modalities": list(self.modalities),
            "task_types": list(self.task_types),
            "build_modes": list(self.build_modes),
            "default_build_mode": self.default_build_mode,
        }

    def compatible(self, modality: str, task_type: str) -> bool:
        return modality in self.modalities and task_type in self.task_types

    def render(
        self,
        record: Mapping[str, Any],
        annotation: Mapping[str, Any],
        *,
        build_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        mode = str(build_mode or self.default_build_mode)
        if mode not in self.build_modes:
            raise ReviewValidationError(
                f"adapter {self.id} does not support build mode {mode!r}; "
                f"choose from {list(self.build_modes)}"
            )
        return self._render(copy.deepcopy(dict(record)), copy.deepcopy(dict(annotation)), mode)

    def _render(
        self, record: Dict[str, Any], annotation: Dict[str, Any], build_mode: str
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


@dataclass(frozen=True)
class FilterAdapter(ReviewOutputAdapter):
    def _render(
        self, record: Dict[str, Any], annotation: Dict[str, Any], build_mode: str
    ) -> List[Dict[str, Any]]:
        return [record] if bool(annotation.get("accepted")) else []


@dataclass(frozen=True)
class MetadataAdapter(ReviewOutputAdapter):
    def _render(
        self, record: Dict[str, Any], annotation: Dict[str, Any], build_mode: str
    ) -> List[Dict[str, Any]]:
        metadata = record.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = record["metadata"] = {"source_metadata": metadata}
        metadata["review"] = annotation
        return [record]


@dataclass(frozen=True)
class SFTCorrectionAdapter(ReviewOutputAdapter):
    def _render(
        self, record: Dict[str, Any], annotation: Dict[str, Any], build_mode: str
    ) -> List[Dict[str, Any]]:
        response = str(annotation["corrected_text"])
        if isinstance(record.get("messages"), list):
            output = copy.deepcopy(record)
            messages = list(output["messages"])
            if (
                messages
                and isinstance(messages[-1], Mapping)
                and messages[-1].get("role") == "assistant"
            ):
                messages[-1] = {**dict(messages[-1]), "content": response}
            else:
                messages.append({"role": "assistant", "content": response})
            output["messages"] = messages
            return [output]
        prompt = record.get("prompt", record.get("question", record.get("instruction")))
        if prompt is None:
            raise ReviewValidationError("SFT correction source requires prompt/messages")
        output = {"prompt": prompt, "response": response}
        if record.get("system") is not None:
            output["system"] = record["system"]
        if isinstance(record.get("metadata"), Mapping):
            output["metadata"] = copy.deepcopy(record["metadata"])
        return [output]


def _alternatives(record: Mapping[str, Any]) -> List[Any]:
    raw = record.get("alternatives", record.get("candidates", record.get("responses", [])))
    return list(raw) if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else []


def _resolve_choice(value: Any, alternatives: Sequence[Any]) -> Any:
    if isinstance(value, int):
        if value < 0 or value >= len(alternatives):
            raise ReviewValidationError("preference choice index is out of range")
        return copy.deepcopy(alternatives[value])
    return copy.deepcopy(value)


@dataclass(frozen=True)
class PreferenceAdapter(ReviewOutputAdapter):
    def _render(
        self, record: Dict[str, Any], annotation: Dict[str, Any], build_mode: str
    ) -> List[Dict[str, Any]]:
        prompt = record.get("prompt", record.get("question", record.get("instruction")))
        if prompt is None:
            raise ReviewValidationError("preference source requires prompt")
        alternatives = _alternatives(record)

        def output(chosen: Any, rejected: Any) -> Dict[str, Any]:
            if "image" not in record:
                return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

            # A VLM preference remains a VLM record.  Keep the media and its
            # canonical context while retaining the winner/loser evidence for
            # preference-aware exporters.  ``response`` makes the selected
            # winner a valid VLM training target instead of silently turning
            # the reviewed row into text-only preference data.
            rendered = copy.deepcopy(record)
            rendered.update(
                prompt=copy.deepcopy(prompt),
                response=copy.deepcopy(chosen),
                chosen=copy.deepcopy(chosen),
                rejected=copy.deepcopy(rejected),
            )
            if not _alternatives(rendered):
                rendered["alternatives"] = [
                    copy.deepcopy(chosen),
                    copy.deepcopy(rejected),
                ]
            return rendered

        if "ranking" in annotation:
            ranking = [_resolve_choice(value, alternatives) for value in annotation["ranking"]]
            winner = ranking[0]
            return [output(winner, loser) for loser in ranking[1:]]
        chosen = _resolve_choice(annotation["chosen"], alternatives)
        # A reviewed tie is a valid human outcome but cannot truthfully be
        # rendered as a winner/loser preference. Retain it in label-set
        # lineage and emit no fabricated training pair.
        if chosen == "tie":
            return []
        rejected = annotation.get("rejected")
        if rejected is None and alternatives:
            remaining = [value for value in alternatives if value != chosen]
            if len(remaining) == 1:
                rejected = remaining[0]
        if rejected is None:
            raise ReviewValidationError(
                "pairwise output requires rejected or exactly two alternatives"
            )
        return [output(chosen, _resolve_choice(rejected, alternatives))]


@dataclass(frozen=True)
class ToolCorrectionAdapter(ReviewOutputAdapter):
    def _render(
        self, record: Dict[str, Any], annotation: Dict[str, Any], build_mode: str
    ) -> List[Dict[str, Any]]:
        correction = dict(annotation["correction"])
        output = copy.deepcopy(record)
        for key in ("messages", "tools", "expected_calls", "expected_results"):
            if key in correction:
                output[key] = copy.deepcopy(correction[key])
        metadata = output.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["review_correction"] = {
                key: value for key, value in correction.items() if key not in output
            }
        return [output]


@dataclass(frozen=True)
class VLMAnnotationAdapter(ReviewOutputAdapter):
    def _render(
        self, record: Dict[str, Any], annotation: Dict[str, Any], build_mode: str
    ) -> List[Dict[str, Any]]:
        if "image" not in record:
            raise ReviewValidationError("VLM annotation source requires an image reference")
        if "corrected_text" in annotation:
            record["response"] = annotation["corrected_text"]
        else:
            metadata = record.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["review"] = annotation
        return [record]


@dataclass(frozen=True)
class AudioAnnotationAdapter(ReviewOutputAdapter):
    def _render(
        self, record: Dict[str, Any], annotation: Dict[str, Any], build_mode: str
    ) -> List[Dict[str, Any]]:
        if "audio" not in record:
            raise ReviewValidationError("audio annotation source requires an audio reference")
        if "corrected_text" in annotation:
            record["transcript"] = annotation["corrected_text"]
        elif "label" in annotation:
            record["label"] = annotation["label"]
        else:
            metadata = record.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["review"] = annotation
        return [record]


class OutputAdapterRegistry:
    def __init__(self, adapters: Optional[Iterable[ReviewOutputAdapter]] = None):
        self._adapters: Dict[str, ReviewOutputAdapter] = {}
        for adapter in adapters or default_output_adapters():
            self.register(adapter)

    def register(self, adapter: ReviewOutputAdapter) -> None:
        if adapter.id in self._adapters:
            raise ReviewValidationError(f"duplicate review output adapter: {adapter.id}")
        self._adapters[adapter.id] = adapter

    def get(self, adapter_id: str) -> ReviewOutputAdapter:
        requested = str(adapter_id)
        if requested not in self._adapters and f"{requested}.v1" in self._adapters:
            requested = f"{requested}.v1"
        try:
            return self._adapters[requested]
        except KeyError as exc:
            raise ReviewValidationError(f"unknown review output adapter: {adapter_id}") from exc

    def compatible(self, modality: str, task_type: str) -> List[ReviewOutputAdapter]:
        return [value for value in self._adapters.values() if value.compatible(modality, task_type)]

    def descriptors(self) -> List[Dict[str, Any]]:
        return [self._adapters[key].descriptor() for key in sorted(self._adapters)]


def default_output_adapter_id(modality: str, task_type: str) -> str:
    if task_type == "binary":
        return "filter.v1"
    if task_type in {"pairwise", "ranking"}:
        return "preference.v1"
    if task_type == "structured_correction":
        return "tool_trace.v1"
    if modality == "vlm":
        return "vlm_annotation.v1"
    if modality == "audio":
        return "audio_annotation.v1"
    if task_type == "text_correction":
        return "sft_correction.v1"
    return "metadata.v1"


def default_output_adapters() -> List[ReviewOutputAdapter]:
    all_modalities = tuple(MODALITY_TASKS)
    non_binary_tasks = tuple(
        sorted({task for tasks in MODALITY_TASKS.values() for task in tasks if task != "binary"})
    )
    return [
        FilterAdapter(
            id="filter.v1",
            version=1,
            modalities=all_modalities,
            task_types=("binary",),
            build_modes=("filter",),
            default_build_mode="filter",
        ),
        MetadataAdapter(
            id="metadata.v1",
            version=1,
            modalities=all_modalities,
            task_types=non_binary_tasks,
            build_modes=("annotate", "append", "replace_by_record_id"),
            default_build_mode="annotate",
        ),
        SFTCorrectionAdapter(
            id="sft_correction.v1",
            version=1,
            modalities=("text",),
            task_types=("text_correction",),
            build_modes=("replace_by_record_id", "append"),
            default_build_mode="replace_by_record_id",
        ),
        PreferenceAdapter(
            id="preference.v1",
            version=1,
            modalities=("text", "preference", "vlm"),
            task_types=("pairwise", "ranking"),
            build_modes=("append", "replace_by_record_id"),
            default_build_mode="append",
        ),
        ToolCorrectionAdapter(
            id="tool_trace.v1",
            version=1,
            modalities=("tool",),
            task_types=("structured_correction",),
            build_modes=("replace_by_record_id", "append"),
            default_build_mode="replace_by_record_id",
        ),
        VLMAnnotationAdapter(
            id="vlm_annotation.v1",
            version=1,
            modalities=("vlm",),
            task_types=("categorical", "multi_label", "scalar", "text_correction"),
            build_modes=("annotate", "replace_by_record_id", "append"),
            default_build_mode="replace_by_record_id",
        ),
        AudioAnnotationAdapter(
            id="audio_annotation.v1",
            version=1,
            modalities=("audio",),
            task_types=("categorical", "multi_label", "scalar", "text_correction"),
            build_modes=("annotate", "replace_by_record_id", "append"),
            default_build_mode="replace_by_record_id",
        ),
    ]


__all__ = [
    "MODALITY_TASKS",
    "OutputAdapterRegistry",
    "ReviewOutputAdapter",
    "default_output_adapter_id",
    "normalize_annotation",
    "normalize_modality",
    "normalize_task_type",
    "validate_schema_definition",
]
