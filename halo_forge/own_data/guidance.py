"""Backend-driven guidance and semantic previews for own-data workflows.

The functions in this module are deliberately transport neutral.  Dashboard,
desktop, and CLI clients receive the same ranked advice, preview semantics,
and remediation identifiers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .models import (
    DatasetReadinessReport,
    GuidedExampleDescriptor,
    RemediationAction,
    ScenarioAdviceRequest,
    ScenarioAdviceResult,
    SemanticRecordPreview,
)
from .registry import TRAINING_SCENARIOS, TrainingScenarioRegistry


_GOAL_TERMS: Dict[str, tuple[str, ...]] = {
    "instruction-sft": (
        "answer",
        "assistant",
        "code",
        "completion",
        "instruction",
        "question",
        "respond",
        "response",
    ),
    "chat-sft": ("chat", "conversation", "dialogue", "multi turn", "messages"),
    "preference-pairs": (
        "chosen",
        "compare answers",
        "dpo",
        "orpo",
        "preference",
        "rank responses",
        "reward model",
        "winner",
        "loser",
    ),
    "prompt-reward": (
        "grpo",
        "prompt only",
        "raft",
        "reward",
        "verifier",
        "generate candidates",
    ),
    "reasoning-sft": (
        "chain of thought",
        "reasoning",
        "solution steps",
        "worked answer",
        "worked solution",
    ),
    "tool-agentic": (
        "agent",
        "function call",
        "tool call",
        "tool result",
        "tools",
    ),
    "vlm-captioning": ("caption", "describe image", "image description", "photos"),
    "vlm-qa": (
        "document extraction",
        "image question",
        "invoice",
        "ocr",
        "visual question",
    ),
    "audio-asr": ("asr", "audio", "speech", "transcribe", "transcript"),
    "corpus-adaptation": (
        "adapt",
        "continued pretraining",
        "corpus",
        "documents",
        "domain language",
        "markdown",
        "pdf",
        "pretrain",
        "prose",
        "text collection",
    ),
}

_OUTCOME_TEXT: Dict[str, str] = {
    "instruction-sft": "A model that follows the style and answers demonstrated in the source.",
    "chat-sft": "A model that continues multi-turn conversations in the demonstrated roles and style.",
    "preference-pairs": "A model optimized toward the reviewed response preferences.",
    "prompt-reward": "A verifier-guided model trained from prompts and scored generated candidates.",
    "reasoning-sft": "A model supervised on the reviewed worked-solution format.",
    "tool-agentic": "A model that emits the demonstrated tool-call and result traces.",
    "vlm-captioning": "A vision-language model adapted to produce the reviewed image descriptions.",
    "vlm-qa": "A vision-language model adapted to answer questions about existing images.",
    "audio-asr": "A Whisper-style model adapted to the reviewed audio transcripts.",
    "corpus-adaptation": "A causal language model adapted to the language and structure of the document corpus.",
}


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_+-]*", value.lower())
        if len(token) > 1
    }


def _source_field_score(scenario: Any, source_fields: Sequence[str]) -> tuple[float, list[str]]:
    normalized = {str(field).strip().lower() for field in source_fields if str(field).strip()}
    if not normalized:
        return 0.0, []
    hits: list[str] = []
    required_scores: list[float] = []
    for target in scenario.required_fields:
        aliases = {
            str(value).lower()
            for value in scenario.field_aliases.get(target, (target,))
        }
        matched = sorted(normalized & aliases)
        if matched:
            required_scores.append(1.0)
            hits.append(f"{target} matches {matched[0]}")
        elif target in scenario.safe_constants:
            required_scores.append(1.0)
            hits.append(f"{target} can use the reviewed scenario default")
        else:
            required_scores.append(0.0)
    if not required_scores:
        return 0.0, hits
    return sum(required_scores) / len(required_scores), hits


def advise_scenarios(
    request: ScenarioAdviceRequest | Mapping[str, Any],
    *,
    registry: TrainingScenarioRegistry = TRAINING_SCENARIOS,
    runtime_values: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> ScenarioAdviceResult:
    """Rank scenarios with explicit evidence and no automatic selection."""

    if not isinstance(request, ScenarioAdviceRequest):
        request = ScenarioAdviceRequest.from_value(request)
    goal = request.goal.lower()
    goal_tokens = _tokens(goal)
    recommendations: list[Dict[str, Any]] = []
    unavailable: list[Dict[str, Any]] = []

    for scenario in registry.list(include_unavailable=True):
        runtime = dict((runtime_values or {}).get(scenario.revision_id) or {})
        available = bool(runtime.get("available", scenario.available))
        unavailable_reason = runtime.get("unavailable_reason") or scenario.unavailable_reason
        reasons: list[str] = []
        cautions: list[str] = []
        score = 0.0

        if request.modality:
            if request.modality == scenario.modality:
                score += 0.32
                reasons.append(f"It matches the selected {request.modality} modality.")
            else:
                score -= 0.45
                cautions.append(
                    f"It expects {scenario.modality} data, not {request.modality}."
                )

        terms = _GOAL_TERMS.get(scenario.id, ())
        matched_terms = [
            term
            for term in terms
            if term in goal or (_tokens(term) and _tokens(term).issubset(goal_tokens))
        ]
        if matched_terms:
            term_score = min(0.44, 0.16 + 0.07 * len(matched_terms))
            score += term_score
            reasons.append(
                "Your goal mentions "
                + ", ".join(f"“{term}”" for term in matched_terms[:3])
                + "."
            )

        field_score, field_reasons = _source_field_score(
            scenario, request.source_fields
        )
        if request.source_fields:
            score += 0.34 * field_score
            reasons.extend(field_reasons[:4])
            if field_score < 1.0:
                cautions.append(
                    "Some required canonical fields are not evident in the supplied field names."
                )

        if request.source_layout:
            if request.source_layout in scenario.source_layouts:
                score += 0.1
                reasons.append(
                    f"The {request.source_layout} source layout is supported."
                )
            else:
                cautions.append(
                    f"The {request.source_layout} layout needs conversion or extraction first."
                )

        if not goal and not request.source_fields and not request.modality:
            score = 0.0

        value = {
            "scenario_id": scenario.id,
            "scenario_revision_id": scenario.revision_id,
            "label": scenario.label,
            "score": round(max(0.0, min(1.0, score)), 4),
            "confidence": (
                "high" if score >= 0.72 else "medium" if score >= 0.46 else "low"
            ),
            "why_fit": reasons
            or ["No direct evidence was supplied; inspect the expected source shape."],
            "cautions": cautions,
            "required_fields": list(scenario.required_fields),
            "optional_fields": list(scenario.optional_fields),
            "trainer_modes": list(
                runtime.get("trainer_modes", scenario.trainer_modes)
            ),
            "expected_outcome": _OUTCOME_TEXT.get(
                scenario.id, scenario.description
            ),
            "available": available,
            "unavailable_reason": unavailable_reason,
            "requires_confirmation": True,
        }
        if available:
            recommendations.append(value)
        elif request.include_unavailable:
            unavailable.append(value)

    recommendations.sort(
        key=lambda item: (-float(item["score"]), str(item["label"]).lower())
    )
    unavailable.sort(key=lambda item: str(item["label"]).lower())
    meaningful = [item for item in recommendations if item["score"] > 0]
    if meaningful:
        recommendations = meaningful + [
            item for item in recommendations if item["score"] == 0
        ]

    questions: list[Dict[str, Any]] = []
    if not request.goal:
        questions.append(
            {
                "id": "goal",
                "label": "What should the model learn?",
                "help": "Describe the behavior, content, or modality you want to adapt.",
            }
        )
    if not request.modality:
        questions.append(
            {
                "id": "modality",
                "label": "What kind of source do you have?",
                "options": ["text", "image", "audio"],
            }
        )
    if not request.source_fields and request.source_layout not in {
        "txt",
        "markdown",
        "html",
        "pdf",
        "docx",
        "document_directory",
    }:
        questions.append(
            {
                "id": "source_fields",
                "label": "Which fields or columns are present?",
                "help": "Inspection can answer this automatically after a source is selected.",
            }
        )

    return ScenarioAdviceResult(
        registry_revision=registry.revision,
        recommendations=tuple(recommendations),
        questions=tuple(questions),
        unavailable=tuple(unavailable),
        explanation=(
            "Recommendations use only the stated goal, modality, source layout, "
            "and field names. Halo Forge never selects or launches a scenario automatically."
        ),
    )


def guided_example_descriptors(
    *,
    registry: TrainingScenarioRegistry = TRAINING_SCENARIOS,
    runtime_values: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> list[GuidedExampleDescriptor]:
    values: list[GuidedExampleDescriptor] = []
    for scenario in registry.list(include_unavailable=False):
        runtime = dict((runtime_values or {}).get(scenario.revision_id) or {})
        if runtime and not runtime.get("available", False):
            continue
        source_shape = ", ".join(scenario.required_fields)
        hardware = (
            "Use a workstation with a verified vision trainer and enough memory for the selected model."
            if scenario.modality == "image"
            else "Use a workstation with the verified Whisper training runtime."
            if scenario.modality == "audio"
            else "Halo Forge will recommend the smallest verified model that fits the active workstation."
        )
        if scenario.id == "corpus-adaptation":
            hardware = (
                "PyTorch and native MLX paths are supported where the active runtime verifies them; "
                "sequence length and adaptation method determine memory use."
            )
        for example in scenario.examples:
            values.append(
                GuidedExampleDescriptor(
                    id=example.id,
                    scenario_id=scenario.id,
                    scenario_revision_id=scenario.revision_id,
                    label=example.name,
                    description=example.description,
                    expected_source_shape=source_shape,
                    expected_outcome=_OUTCOME_TEXT.get(
                        scenario.id, scenario.description
                    ),
                    hardware_guidance=hardware,
                    fixture_format=example.format,
                    fixture_filename=example.filename,
                    record_count=len(example.records),
                    modality=scenario.modality,
                    trainer_modes=tuple(
                        runtime.get("trainer_modes", scenario.trainer_modes)
                    ),
                    documentation_anchor=scenario.documentation_anchor,
                )
            )
    return values


def _issue_values(raw: Any) -> tuple[Dict[str, Any], ...]:
    if not raw:
        return ()
    values = raw if isinstance(raw, list) else [raw]
    output = []
    for value in values:
        if isinstance(value, Mapping):
            output.append(copy.deepcopy(dict(value)))
        else:
            output.append(
                {"code": "preview_issue", "message": str(value), "severity": "error"}
            )
    return tuple(output)


def _semantic_presentation(kind: str, canonical: Mapping[str, Any]) -> tuple[str, str, Dict[str, Any]]:
    if kind in {"chat", "tool"}:
        turns = []
        for index, message in enumerate(canonical.get("messages") or []):
            if not isinstance(message, Mapping):
                continue
            turns.append(
                {
                    "index": index,
                    "role": str(message.get("role") or message.get("from") or "unknown"),
                    "content": str(message.get("content") or message.get("value") or ""),
                    "tool_calls": copy.deepcopy(message.get("tool_calls") or []),
                }
            )
        presentation: Dict[str, Any] = {"turns": turns}
        if kind == "tool":
            presentation.update(
                tools=copy.deepcopy(canonical.get("tools") or []),
                expected_calls=copy.deepcopy(canonical.get("expected_calls") or []),
                expected_results=copy.deepcopy(
                    canonical.get("expected_results") or []
                ),
            )
        return (
            "Tool trace" if kind == "tool" else "Conversation",
            f"{len(turns)} ordered turn{'s' if len(turns) != 1 else ''}",
            presentation,
        )
    if kind == "preference":
        prompt = str(canonical.get("prompt") or "")
        return (
            prompt[:80] or "Preference pair",
            "Compare the reviewed chosen and rejected responses.",
            {
                "prompt": prompt,
                "chosen": canonical.get("chosen"),
                "rejected": canonical.get("rejected"),
                "system": canonical.get("system"),
            },
        )
    if kind == "vlm":
        prompt = str(canonical.get("prompt") or "")
        return (
            prompt[:80] or "Image record",
            str(canonical.get("response") or canonical.get("ground_truth") or "")[:160],
            {
                "image": copy.deepcopy(canonical.get("image")),
                "prompt": prompt,
                "response": canonical.get("response"),
                "ground_truth": canonical.get("ground_truth"),
                "alternatives": copy.deepcopy(canonical.get("alternatives") or []),
            },
        )
    if kind == "audio":
        transcript = str(
            canonical.get("transcript") or canonical.get("label") or ""
        )
        return (
            transcript[:80] or "Audio record",
            str(canonical.get("task") or "audio"),
            {
                "audio": copy.deepcopy(canonical.get("audio")),
                "task": canonical.get("task"),
                "transcript": canonical.get("transcript"),
                "label": canonical.get("label"),
                "metadata": copy.deepcopy(canonical.get("metadata") or {}),
            },
        )
    if kind == "corpus":
        text = str(canonical.get("text") or "")
        title = str(canonical.get("title") or "").strip()
        return (
            title or str(canonical.get("source_ref") or "Corpus document"),
            f"{len(text):,} characters · {max(1, text.count(chr(10)) + 1) if text else 0} lines",
            {
                "title": title or None,
                "text": text,
                "source_ref": canonical.get("source_ref"),
                "source_spans": copy.deepcopy(canonical.get("source_spans") or []),
                "metadata": copy.deepcopy(canonical.get("metadata") or {}),
            },
        )
    prompt = str(canonical.get("prompt") or "")
    response = str(
        canonical.get("response")
        or canonical.get("reference_answer")
        or canonical.get("text")
        or ""
    )
    return (
        prompt[:80] or "Training record",
        response[:160],
        {
            key: copy.deepcopy(canonical.get(key))
            for key in (
                "system",
                "prompt",
                "response",
                "reference_answer",
                "text",
                "metadata",
            )
            if canonical.get(key) is not None
        },
    )


def semantic_previews(
    mapping_preview: Mapping[str, Any],
    *,
    canonical_schema: str,
    limit: int = 50,
) -> Dict[str, Any]:
    items: list[Dict[str, Any]] = []
    for index, row in enumerate(mapping_preview.get("items") or []):
        if index >= max(1, min(int(limit), 200)):
            break
        source = copy.deepcopy(dict(row.get("source") or {}))
        canonical = copy.deepcopy(dict(row.get("canonical") or {}))
        title, summary, presentation = _semantic_presentation(
            canonical_schema, canonical
        )
        items.append(
            SemanticRecordPreview(
                kind=canonical_schema,
                ordinal=int(row.get("ordinal", index)),
                title=title,
                summary=summary,
                source=source,
                canonical=canonical,
                presentation=presentation,
                issues=_issue_values(row.get("issues")),
                provenance={
                    "inspection_id": mapping_preview.get("inspection_id"),
                    "scenario_revision_id": mapping_preview.get(
                        "scenario_revision_id"
                    ),
                },
            ).to_dict()
        )
    return {
        "items": items,
        "total": len(mapping_preview.get("items") or []),
        "limit": max(1, min(int(limit), 200)),
        "offset": 0,
        "canonical_schema": canonical_schema,
        "sampled": True,
    }


def _summary(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p50": 0,
            "p95": 0,
        }
    ordered = sorted(int(value) for value in values)

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
    }


def _stable_record_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


def build_readiness_report(
    inspection: Mapping[str, Any],
    mapping_preview: Mapping[str, Any],
    preparation: Mapping[str, Any],
    *,
    canonical_schema: str,
    scenario_revision_id: Optional[str],
) -> DatasetReadinessReport:
    """Summarize sampled quality with a direct action for every blocker."""

    items = list(mapping_preview.get("items") or [])
    valid = [
        item
        for item in items
        if item.get("canonical") and not item.get("issues")
    ]
    invalid = [item for item in items if item not in valid]
    hashes = Counter(
        _stable_record_hash(dict(item.get("canonical") or {})) for item in valid
    )
    duplicates = sum(count - 1 for count in hashes.values() if count > 1)
    character_lengths = [
        sum(
            len(str(value))
            for value in dict(item.get("canonical") or {}).values()
            if isinstance(value, (str, int, float))
        )
        for item in valid
    ]
    token_estimates = [math.ceil(length / 4) for length in character_lengths]
    total_records = int(
        inspection.get("row_count")
        or inspection.get("total_records")
        or len(items)
    )
    ratio = (
        len(valid) / max(1, len(items))
        if items
        else 0.0
    )
    estimated_accepted = round(total_records * ratio)
    split_policy = copy.deepcopy(preparation.get("split_policy") or {})
    ratios = dict(
        split_policy.get("ratios")
        or next(
            (
                step.get("ratios")
                for step in (preparation.get("recipe") or {}).get("steps", [])
                if isinstance(step, Mapping)
                and str(step.get("kind") or "").lower() == "split"
            ),
            {},
        )
        or {}
    )
    split_balance = {
        split: {
            "ratio": float(value),
            "estimated_records": round(estimated_accepted * float(value)),
        }
        for split, value in ratios.items()
    }

    blockers: list[Dict[str, Any]] = []
    warnings: list[Dict[str, Any]] = []
    actions: list[RemediationAction] = []
    if not items or not valid:
        blockers.append(
            {
                "code": "no_valid_records",
                "message": "No retained preview record satisfies the confirmed mapping.",
                "severity": "error",
                "action_id": "review_mapping",
            }
        )
        actions.append(
            RemediationAction(
                id="review_mapping",
                label="Review field mapping",
                action="open_mapping",
                description="Return to Map and connect every required field.",
                target="map",
            )
        )
    if invalid:
        blockers.append(
            {
                "code": "sample_mapping_errors",
                "message": (
                    f"{len(invalid)} of {len(items)} retained preview records have "
                    "mapping or validation errors."
                ),
                "severity": "error",
                "action_id": "inspect_rejected",
            }
        )
        actions.append(
            RemediationAction(
                id="inspect_rejected",
                label="Inspect rejected examples",
                action="open_rejected_records",
                description="Review representative failures and adjust mapping or quarantine rules.",
                target="format",
            )
        )
    if duplicates:
        warnings.append(
            {
                "code": "sample_exact_duplicates",
                "message": f"{duplicates} exact duplicate preview records were detected.",
                "severity": "warning",
                "action_id": "review_dedup",
            }
        )
        actions.append(
            RemediationAction(
                id="review_dedup",
                label="Review deduplication",
                action="open_preparation_control",
                description="Keep exact deduplication enabled or inspect the repeated examples.",
                target="deduplication",
            )
        )

    minimum = 2 if canonical_schema == "corpus" else 10
    if estimated_accepted < minimum:
        blockers.append(
            {
                "code": "insufficient_records_for_split",
                "message": (
                    f"About {estimated_accepted} accepted records are expected; "
                    f"at least {minimum} are needed for the default reviewed split."
                ),
                "severity": "error",
                "action_id": "add_source_records",
            }
        )
        actions.append(
            RemediationAction(
                id="add_source_records",
                label="Add more source data",
                action="open_source",
                description="Select a larger source or add files before publishing a version.",
                target="source",
            )
        )

    media = copy.deepcopy(
        inspection.get("media_summary")
        or (inspection.get("statistics") or {}).get("media_summary")
        or {}
    )
    missing_media = int(
        media.get("missing")
        or media.get("unresolved")
        or media.get("invalid")
        or 0
    )
    if missing_media:
        blockers.append(
            {
                "code": "media_resolution_failed",
                "message": f"{missing_media} referenced media assets could not be resolved.",
                "severity": "error",
                "action_id": "set_media_root",
            }
        )
        actions.append(
            RemediationAction(
                id="set_media_root",
                label="Set media folder",
                action="open_media_root",
                description="Choose the folder that relative image or audio paths are resolved from.",
                target="map",
            )
        )

    extraction = copy.deepcopy(
        inspection.get("extraction_summary")
        or (inspection.get("statistics") or {}).get("extraction_summary")
        or {}
    )
    extraction_failures = int(
        extraction.get("failed")
        or extraction.get("quarantined")
        or 0
    )
    if extraction_failures:
        warnings.append(
            {
                "code": "document_extraction_failures",
                "message": f"{extraction_failures} documents could not be extracted and will be quarantined.",
                "severity": "warning",
                "action_id": "inspect_extraction_failures",
            }
        )
        actions.append(
            RemediationAction(
                id="inspect_extraction_failures",
                label="Inspect extraction failures",
                action="open_extraction_failures",
                description="See encrypted, image-only, empty, or unsupported documents before publishing.",
                target="format",
            )
        )

    class_counts: Counter[str] = Counter()
    for item in valid:
        record = dict(item.get("canonical") or {})
        value = (
            record.get("label")
            or record.get("task")
            or (record.get("metadata") or {}).get("category")
            if isinstance(record.get("metadata") or {}, Mapping)
            else None
        )
        if value is not None:
            class_counts[str(value)] += 1

    return DatasetReadinessReport(
        scope="inspection",
        subject_id=str(
            inspection.get("id")
            or mapping_preview.get("inspection_id")
            or ""
        ),
        ready=not blockers,
        scenario_revision_id=scenario_revision_id,
        sampled=True,
        summary={
            "source_records": total_records,
            "preview_records": len(items),
            "valid_preview_records": len(valid),
            "invalid_preview_records": len(invalid),
            "estimated_accepted_records": estimated_accepted,
            "estimated_quarantined_records": max(
                0, total_records - estimated_accepted
            ),
            "exact_duplicate_preview_records": duplicates,
            "token_count_is_estimated": True,
        },
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        actions=tuple(actions),
        rejected_examples=tuple(
            {
                "ordinal": item.get("ordinal"),
                "source": copy.deepcopy(item.get("source") or {}),
                "issues": copy.deepcopy(item.get("issues") or []),
            }
            for item in invalid[:10]
        ),
        distributions={
            "characters_per_record": _summary(character_lengths),
            "estimated_tokens_per_record": _summary(token_estimates),
            "class_balance": dict(sorted(class_counts.items())),
        },
        split_balance=split_balance,
        media=media,
        extraction=extraction,
        minimum_data={
            "required_for_default_split": minimum,
            "estimated_available": estimated_accepted,
            "satisfied": estimated_accepted >= minimum,
            "scientific_quality_threshold": None,
            "note": (
                "This is an operational split minimum, not a claim that the dataset "
                "is sufficient for a useful model."
            ),
        },
    )


__all__ = [
    "advise_scenarios",
    "build_readiness_report",
    "guided_example_descriptors",
    "semantic_previews",
]
