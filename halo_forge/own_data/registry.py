"""Versioned, truthful scenarios and execution-surface capabilities."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .models import (
    InterfaceCapabilityDescriptor,
    TrainingScenarioDescriptor,
    TrainingScenarioExample,
)


_TABULAR_SOURCE_LAYOUTS = (
    "json",
    "jsonl",
    "csv",
    "tsv",
    "parquet",
    "huggingface",
)
_MEDIA_SOURCE_LAYOUTS = _TABULAR_SOURCE_LAYOUTS + (
    "media_directory_manifest",
    "media_directory_sidecar",
    "paired_media_text",
)
_DOCUMENT_SOURCE_LAYOUTS = _TABULAR_SOURCE_LAYOUTS + (
    "txt",
    "markdown",
    "html",
    "pdf",
    "docx",
    "document_directory",
)

_EXAMPLE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _example_wav() -> bytes:
    sample_rate = 8_000
    samples = b"\x00\x00" * 800
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(samples))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(samples))
        + samples
    )


def _fixture_asset(path: str) -> Optional[bytes]:
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        return _EXAMPLE_PNG
    if suffix == ".wav":
        return _example_wav()
    return None


def _recipe(schema: str, *, grouped_asset_field: Optional[str] = None) -> Dict[str, Any]:
    split: Dict[str, Any] = {
        "kind": "split",
        "method": "random",
        "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "seed": 42,
    }
    if grouped_asset_field:
        # Group by the canonical media reference. Dataset Lab validates and
        # fingerprints its content before splitting, so the same image/audio
        # bytes cannot cross a held-out boundary even when a source uses two
        # different relative filenames for that asset.
        split.update(
            method="grouped",
            group_field=grouped_asset_field,
            group_by_asset_hash=True,
        )
    return {
        "name": "guided-own-data",
        "schema": schema,
        "seed": 42,
        "steps": [
            {"kind": "map", "schema": schema, "fields": {}},
            {"kind": "normalize", "trim": True, "collapse_whitespace": True},
            {"kind": "validate", "schema": schema, "on_error": "quarantine"},
            {"kind": "dedup", "method": "exact"},
            split,
            {"kind": "contamination", "action": "report"},
        ],
    }


def _corpus_recipe() -> Dict[str, Any]:
    """Reviewed, document-preserving default for continued pretraining."""

    return {
        "name": "guided-corpus-adaptation",
        "schema": "corpus",
        "seed": 42,
        "steps": [
            {
                "kind": "map",
                "schema": "corpus",
                "fields": {
                    "document_id": "document_id",
                    "document_hash": "document_hash",
                    "text": "text",
                    "title": "title",
                    "source_ref": "source_ref",
                    "source_spans": "source_spans",
                    "timestamp": "timestamp",
                    "metadata": "metadata",
                },
            },
            {
                "kind": "normalize",
                "fields": ["text", "title"],
                "trim": True,
                "collapse_whitespace": False,
            },
            {
                "kind": "document_clean",
                "strip_boilerplate": True,
                "preserve_headings": True,
                "preserve_code_blocks": True,
            },
            {
                "kind": "document_filter",
                "quarantine_extraction_failures": True,
                "require_visible_text": True,
            },
            {"kind": "validate", "schema": "corpus", "on_error": "quarantine"},
            {"kind": "dedup", "method": "exact", "fields": ["text"]},
            {
                "kind": "dedup",
                "method": "fuzzy",
                "fields": ["text"],
                "threshold": 0.92,
            },
            {
                "kind": "split",
                "method": "grouped",
                "group_field": "source_ref",
                "ratios": {"train": 0.9, "validation": 0.1},
                "seed": 42,
            },
            {"kind": "contamination", "action": "report"},
        ],
    }


def _example(
    identifier: str,
    name: str,
    description: str,
    records: Iterable[Dict[str, Any]],
    *,
    filename: Optional[str] = None,
    format: str = "jsonl",
) -> TrainingScenarioExample:
    return TrainingScenarioExample(
        id=identifier,
        name=name,
        description=description,
        format=format,
        filename=filename or f"{identifier}.jsonl",
        records=tuple(copy.deepcopy(list(records))),
    )


def _scenario(
    identifier: str,
    *,
    label: str,
    description: str,
    modality: str,
    schema: str,
    task: str,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    aliases: Dict[str, tuple[str, ...]],
    trainers: tuple[str, ...],
    models: tuple[str, ...],
    examples: tuple[TrainingScenarioExample, ...],
    constants: Optional[Dict[str, Any]] = None,
    failures: tuple[str, ...] = (),
    detection: Optional[Dict[str, Any]] = None,
    source_layouts: Optional[tuple[str, ...]] = None,
    available: bool = True,
    unavailable_reason: Optional[str] = None,
    default_recipe: Optional[Dict[str, Any]] = None,
    proof_budget: Optional[Dict[str, Any]] = None,
) -> TrainingScenarioDescriptor:
    version = 1
    return TrainingScenarioDescriptor(
        id=identifier,
        revision_id=f"{identifier}@{version}",
        version=version,
        label=label,
        description=description,
        modality=modality,
        canonical_schema=schema,
        task=task,
        available=available,
        unavailable_reason=unavailable_reason,
        required_fields=required,
        optional_fields=optional,
        field_aliases=aliases,
        safe_constants=copy.deepcopy(constants or {}),
        source_layouts=tuple(
            source_layouts
            or (_MEDIA_SOURCE_LAYOUTS if modality in {"image", "audio"} else _TABULAR_SOURCE_LAYOUTS)
        ),
        trainer_modes=trainers,
        model_families=models,
        default_recipe=copy.deepcopy(
            default_recipe
            or _recipe(
                schema,
                grouped_asset_field=(
                    "image" if modality == "image" else "audio" if modality == "audio" else None
                ),
            )
        ),
        proof_budget=copy.deepcopy(
            proof_budget
            or {
                "max_samples": 50 if modality in {"image", "audio"} else 200,
                "epochs": 1 if trainers and trainers[0] not in {"raft", "grpo"} else None,
                "cycles": 1 if any(item in {"raft", "grpo"} for item in trainers) else None,
                "seed": 42,
            }
        ),
        common_failures=failures,
        documentation_anchor=f"own-data/{identifier}",
        examples=examples,
        detection_hints=copy.deepcopy(detection or {}),
    )


_SCENARIOS = (
    _scenario(
        "instruction-sft",
        label="Instruction and response",
        description="Teach a model to answer instructions, questions, or code tasks.",
        modality="text",
        schema="sft",
        task="supervised_fine_tuning",
        required=("prompt", "response"),
        optional=("system",),
        aliases={
            "prompt": ("prompt", "instruction", "question", "input"),
            "response": ("response", "completion", "output", "answer", "code"),
            "system": ("system", "system_prompt"),
        },
        trainers=("sft",),
        models=("qwen2.5", "qwen2", "llama-3", "mistral"),
        examples=(
            _example(
                "instruction-text",
                "Question and answer",
                "A compact instruction-response source.",
                [
                    {
                        "instruction": "Summarize the water cycle.",
                        "answer": "Water evaporates, condenses, and returns as precipitation.",
                    },
                    {"instruction": "Name the capital of Japan.", "answer": "Tokyo."},
                ],
            ),
            _example(
                "instruction-code",
                "Code instruction",
                "Natural-language programming tasks with reviewed answers.",
                [
                    {
                        "prompt": "Write a Python function that squares a number.",
                        "response": "def square(value):\n    return value * value",
                    },
                    {
                        "prompt": "Explain what a stable sort preserves.",
                        "response": "It preserves the relative order of records with equal keys.",
                    },
                ],
            ),
        ),
        failures=(
            "Prompt or response is empty",
            "Answers are stored in several incompatible columns",
        ),
        detection={"exclude_fields": ("chosen", "rejected", "messages", "image", "audio", "tools")},
    ),
    _scenario(
        "chat-sft",
        label="Multi-turn conversation",
        description="Teach ordered system, user, assistant, and tool conversations.",
        modality="text",
        schema="chat",
        task="chat_supervised_fine_tuning",
        required=("messages",),
        optional=(),
        aliases={"messages": ("messages", "conversations", "dialogue")},
        trainers=("sft",),
        models=("qwen2.5", "qwen2", "llama-3", "mistral"),
        examples=(
            _example(
                "sharegpt-chat",
                "ShareGPT-style chat",
                "The mapper normalizes from/value to role/content.",
                [
                    {
                        "conversations": [
                            {"from": "human", "value": "Hello"},
                            {"from": "gpt", "value": "Hi! How can I help?"},
                        ]
                    }
                ],
            ),
        ),
        failures=("Conversation entries lack roles", "No assistant response is present"),
        detection={"exclude_fields": ("tools", "tool_definitions", "functions")},
    ),
    _scenario(
        "preference-pairs",
        label="Preference pairs",
        description="Teach which of two responses is preferred.",
        modality="text",
        schema="preference",
        task="preference_optimization",
        required=("prompt", "chosen", "rejected"),
        optional=("system",),
        aliases={
            "prompt": ("prompt", "instruction", "question"),
            "chosen": ("chosen", "preferred", "winner"),
            "rejected": ("rejected", "dispreferred", "loser"),
            "system": ("system", "system_prompt"),
        },
        trainers=("dpo", "orpo", "rm"),
        models=("qwen2.5", "qwen2", "llama-3", "mistral"),
        examples=(
            _example(
                "preference-basic",
                "Chosen and rejected responses",
                "One reviewed winner and loser for each prompt.",
                [
                    {
                        "prompt": "Explain gravity simply.",
                        "chosen": "Gravity pulls masses toward one another.",
                        "rejected": "Gravity is when things always fall down.",
                    }
                ],
            ),
        ),
        failures=("Chosen and rejected are identical", "A preference side is missing"),
    ),
    _scenario(
        "prompt-reward",
        label="Prompt-only reward training",
        description="Generate candidates from prompts for RAFT or GRPO and score them with a verifier.",
        modality="text",
        schema="prompt",
        task="verifier_guided_optimization",
        required=("prompt",),
        optional=("reference_answer",),
        aliases={
            "prompt": ("prompt", "instruction", "question", "problem"),
            "reference_answer": ("reference_answer", "solution", "reference"),
        },
        trainers=("raft", "grpo"),
        models=("qwen2.5", "qwen2", "llama-3", "mistral"),
        examples=(
            _example(
                "reward-prompts",
                "Verified prompts",
                "Prompts with optional reviewed reference answers.",
                [
                    {"problem": "What is 12 * 8?", "reference_answer": "96"},
                    {"problem": "Return the first prime after 10.", "reference_answer": "11"},
                ],
            ),
        ),
        failures=("No qualified compatible verifier is selected",),
        detection={
            "exclude_fields": (
                "response",
                "answer",
                "completion",
                "output",
                "code",
                "reasoning",
                "worked_solution",
                "chain_of_thought",
                "solution_steps",
                "chosen",
                "rejected",
                "messages",
                "image",
                "audio",
            )
        },
    ),
    _scenario(
        "reasoning-sft",
        label="Worked reasoning traces",
        description="Supervise reviewed worked solutions rather than scoring generated candidates.",
        modality="text",
        schema="sft",
        task="reasoning_trace_supervised_fine_tuning",
        required=("prompt", "response"),
        optional=("system",),
        aliases={
            "prompt": ("problem", "question", "prompt"),
            "response": ("reasoning", "worked_solution", "chain_of_thought", "solution_steps"),
            "system": ("system", "system_prompt"),
        },
        # This scenario is worked-trace supervision. The separate reasoning
        # trainer consumes prompt-only verifier-guided data, not SFT traces.
        trainers=("sft",),
        models=("qwen2.5", "qwen2", "llama-3", "mistral"),
        examples=(
            _example(
                "reasoning-worked",
                "Worked arithmetic",
                "A reviewed problem and worked solution.",
                [
                    {
                        "problem": "A box has 3 rows of 4 items. How many?",
                        "worked_solution": "There are 3 groups of 4, so 3 × 4 = 12.",
                    }
                ],
            ),
        ),
        detection={
            "require_any_fields": (
                "reasoning",
                "worked_solution",
                "chain_of_thought",
                "solution_steps",
            )
        },
    ),
    _scenario(
        "tool-agentic",
        label="Tool calls and agent traces",
        description="Teach structured tool definitions, calls, results, and conversational traces.",
        modality="text",
        schema="tool",
        task="tool_and_agentic_training",
        required=("messages", "tools"),
        optional=("expected_calls", "expected_results"),
        aliases={
            "messages": ("messages", "conversations", "trace"),
            "tools": ("tools", "tool_definitions", "functions"),
            "expected_calls": ("expected_calls", "tool_calls"),
            "expected_results": ("expected_results", "tool_results"),
        },
        trainers=("sft", "agentic"),
        models=("qwen2.5", "qwen2", "llama-3", "mistral"),
        examples=(
            _example(
                "tool-trace",
                "Weather tool trace",
                "A message sequence with the exact available tool schema.",
                [
                    {
                        "messages": [
                            {"role": "user", "content": "Weather in Austin?"},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {"name": "weather", "arguments": {"city": "Austin"}}
                                ],
                            },
                        ],
                        "tools": [
                            {
                                "name": "weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                },
                            }
                        ],
                    }
                ],
            ),
        ),
        failures=(
            "Tool calls reference undeclared tools",
            "Structured arguments are stored as invalid JSON strings",
        ),
    ),
    _scenario(
        "vlm-captioning",
        label="Image captioning",
        description="Teach captions or descriptions for existing images.",
        modality="image",
        schema="vlm",
        task="image_captioning",
        required=("image", "prompt", "response"),
        optional=(),
        aliases={
            "image": (
                "image", "image_path", "file", "path", "filename",
                "relative_path", "media", "media_path",
            ),
            "prompt": ("prompt", "instruction"),
            "response": ("caption", "response", "description", "text"),
        },
        constants={"prompt": "Describe this image."},
        trainers=("vlm",),
        models=("qwen2-vl", "qwen-vl", "llava"),
        examples=(
            _example(
                "vlm-captions",
                "Image manifest",
                "Relative image paths keep media beside the manifest.",
                [{"image": "images/sample.png", "caption": "A red bicycle beside a brick wall."}],
            ),
        ),
        failures=("Image path is missing", "Image assets appear in both train and held-out splits"),
        detection={"exclude_fields": ("question",), "preferred_fields": ("caption", "description")},
    ),
    _scenario(
        "vlm-qa",
        label="Visual question answering or extraction",
        description="Answer questions about images and document pages.",
        modality="image",
        schema="vlm",
        task="visual_question_answering",
        required=("image", "prompt", "response"),
        optional=("ground_truth",),
        aliases={
            "image": (
                "image", "image_path", "file", "path", "filename",
                "relative_path", "media", "media_path",
            ),
            "prompt": ("prompt", "question", "instruction"),
            "response": ("response", "answer", "completion", "text"),
            "ground_truth": ("ground_truth", "label", "target"),
        },
        trainers=("vlm",),
        models=("qwen2-vl", "qwen-vl", "llava"),
        examples=(
            _example(
                "vlm-question-answer",
                "Document question answering",
                "A relative page image with its reviewed question and answer.",
                [
                    {
                        "image_path": "pages/invoice-001.png",
                        "question": "What is the invoice total?",
                        "answer": "$42.50",
                    }
                ],
            ),
        ),
        failures=("Question or answer is missing", "Document page paths cannot be resolved"),
        detection={"preferred_fields": ("question", "answer")},
    ),
    _scenario(
        "audio-asr",
        label="Audio transcription (ASR)",
        description="Fine-tune Whisper-family models on existing audio and reviewed transcripts.",
        modality="audio",
        schema="audio",
        task="automatic_speech_recognition",
        required=("audio", "task", "transcript"),
        optional=(),
        aliases={
            "audio": (
                "audio", "audio_path", "file", "path", "filename",
                "relative_path", "media", "media_path",
            ),
            "task": ("task", "instruction"),
            "transcript": ("transcript", "text", "sentence"),
        },
        constants={"task": "transcribe"},
        trainers=("audio",),
        models=("whisper",),
        examples=(
            _example(
                "audio-transcripts",
                "Audio transcript manifest",
                "Relative audio paths plus reviewed transcripts.",
                [{"audio": "clips/hello.wav", "transcript": "Hello from Halo Forge."}],
            ),
        ),
        failures=(
            "Audio cannot be decoded",
            "Transcript is empty",
            "Shared audio crosses held-out splits",
        ),
    ),
    _scenario(
        "corpus-adaptation",
        label="Adapt a model to documents",
        description=(
            "Continue causal language-model training on a reviewed collection of "
            "documents while preserving document provenance."
        ),
        modality="text",
        schema="corpus",
        task="continued_pretraining",
        required=("document_id", "document_hash", "text", "source_ref"),
        optional=("title", "source_spans", "timestamp", "metadata"),
        aliases={
            "document_id": ("document_id", "id", "doc_id"),
            "document_hash": ("document_hash", "content_hash", "hash"),
            "text": ("text", "content", "document", "body"),
            "title": ("title", "name", "heading"),
            "source_ref": ("source_ref", "source", "path", "filename", "url"),
            "source_spans": ("source_spans", "spans", "provenance"),
            "timestamp": ("timestamp", "date", "created_at", "published_at"),
            "metadata": ("metadata", "meta"),
        },
        trainers=("cpt",),
        models=("qwen2.5", "qwen2", "llama-3", "mistral"),
        source_layouts=_DOCUMENT_SOURCE_LAYOUTS,
        examples=(
            _example(
                "corpus-markdown",
                "Small document collection",
                "Two short Markdown documents with headings, prose, and code.",
                [
                    {
                        "title": "Halo Forge operating notes",
                        "text": (
                            "# Halo Forge operating notes\n\n"
                            "Dataset versions are immutable. A new source revision is "
                            "created when referenced files change.\n\n"
                            "```text\ninspect → prepare → version → train\n```\n"
                        ),
                    },
                    {
                        "title": "Evaluation vocabulary",
                        "text": (
                            "# Evaluation vocabulary\n\n"
                            "Development evidence may guide iteration. Holdout evidence "
                            "is reserved for final confirmation.\n"
                        ),
                    },
                ],
                filename="corpus-notes.md",
                format="markdown",
            ),
        ),
        failures=(
            "A PDF is encrypted, empty, or image-only",
            "Document text is empty after visible-content extraction",
            "Near-duplicate documents dominate the corpus",
            "A document appears in both training and validation",
        ),
        detection={
            "source_layouts": _DOCUMENT_SOURCE_LAYOUTS,
            "preferred_fields": ("text", "content", "body", "document"),
        },
        default_recipe=_corpus_recipe(),
        proof_budget={
            "proof_run": False,
            "budget_mode": "passes",
            "corpus_passes": 1.0,
            "seed": 42,
        },
    ),
    _scenario(
        "text-classification",
        label="Text classification",
        description="Train a compact model to assign one reviewed class to text.",
        modality="text",
        schema="classification",
        task="text_classification",
        required=("input", "label"),
        optional=(),
        aliases={
            "input": ("input", "text", "content", "sentence"),
            "label": ("label", "class", "target", "category"),
        },
        trainers=("classify",),
        models=("bert", "roberta", "deberta", "modernbert"),
        examples=(
            _example(
                "text-classification",
                "Support request labels",
                "Reviewed text and one class per row.",
                [
                    {"text": "The invoice total is wrong.", "label": "billing"},
                    {"text": "I cannot sign in.", "label": "account_access"},
                ],
            ),
        ),
        failures=("A class is missing", "A label appears only in held-out data"),
    ),
    _scenario(
        "text-multilabel",
        label="Multi-label text classification",
        description="Assign zero or more reviewed labels to each text record.",
        modality="text",
        schema="classification",
        task="text_multilabel_classification",
        required=("input", "labels"),
        optional=(),
        aliases={
            "input": ("input", "text", "content", "sentence"),
            "labels": ("labels", "classes", "targets", "categories"),
        },
        trainers=("classify",),
        models=("bert", "roberta", "deberta", "modernbert"),
        examples=(
            _example(
                "text-multilabel",
                "Document topics",
                "A reviewed list of topics for each document.",
                [{"text": "Quarterly cloud security review.", "labels": ["cloud", "security"]}],
            ),
        ),
        failures=("Labels are encoded inconsistently", "Unknown labels appear after the split"),
    ),
    _scenario(
        "embedding-pairs",
        label="Embedding pairs",
        description="Train a bi-encoder to place reviewed matching texts close together.",
        modality="text",
        schema="embedding",
        task="embedding_training",
        required=("anchor", "positive"),
        optional=("negatives",),
        aliases={
            "anchor": ("anchor", "query", "question", "input"),
            # Keep answer out of automatic detection so ordinary question /
            # answer SFT data is not ambiguously preselected as retrieval
            # training. Operators can still map an answer field explicitly.
            "positive": ("positive", "document", "passage"),
            "negatives": ("negatives", "negative", "hard_negatives"),
        },
        trainers=("embed",),
        models=("sentence-transformers", "bert", "roberta"),
        examples=(
            _example(
                "embedding-pairs",
                "Query and relevant passage",
                "A query, its positive passage, and optional reviewed negatives.",
                [{"query": "reset password", "document": "Open Settings, then choose Reset password."}],
            ),
        ),
        failures=("Anchor and positive are identical", "A negative duplicates the positive"),
    ),
    _scenario(
        "reranking",
        label="Search reranking",
        description="Train a cross-encoder to order candidate documents for a query.",
        modality="text",
        schema="reranking",
        task="reranker_training",
        required=("query", "document", "relevance"),
        optional=("candidates", "ordered_preference"),
        aliases={
            "query": ("query", "question", "prompt"),
            "document": ("document", "passage", "candidate"),
            "candidates": ("candidates", "documents", "passages"),
            "relevance": ("relevance", "score", "label"),
            "ordered_preference": ("ordered_preference", "ranking", "order"),
        },
        trainers=("rerank",),
        models=("cross-encoder", "bert", "roberta"),
        examples=(
            _example(
                "reranking",
                "Query and relevance",
                "One query-document pair with a reviewed relevance score.",
                [{"query": "refund policy", "document": "Refunds are available for 30 days.", "relevance": 1}],
            ),
        ),
        failures=("A query has no candidate", "Relevance labels use incompatible scales"),
    ),
    _scenario(
        "image-classification",
        label="Image classification",
        description="Classify existing images with a verified image-classification head.",
        modality="image",
        schema="classification",
        task="image_classification",
        required=("media", "label"),
        optional=(),
        aliases={
            "media": ("media", "image", "image_path", "file", "path", "relative_path"),
            "label": ("label", "class", "target", "category"),
        },
        trainers=("classify",),
        models=("vit", "convnext", "swin"),
        examples=(
            _example(
                "image-classification",
                "Image manifest",
                "Relative image paths and reviewed classes.",
                [{"image": "images/sample.png", "label": "bicycle"}],
            ),
        ),
        failures=("Image cannot be decoded", "An identical image crosses protected splits"),
        default_recipe=_recipe("classification", grouped_asset_field="media"),
    ),
    _scenario(
        "audio-classification",
        label="Audio classification",
        description="Classify existing audio clips with a verified audio-classification head.",
        modality="audio",
        schema="classification",
        task="audio_classification",
        required=("media", "label"),
        optional=(),
        aliases={
            "media": ("media", "audio", "audio_path", "file", "path", "relative_path"),
            "label": ("label", "class", "target", "category"),
        },
        trainers=("classify",),
        models=("wav2vec2", "hubert", "audio-spectrogram-transformer"),
        examples=(
            _example(
                "audio-classification",
                "Audio manifest",
                "Relative audio paths and reviewed classes.",
                [{"audio": "clips/hello.wav", "label": "speech"}],
            ),
        ),
        failures=("Audio cannot be decoded", "An identical clip crosses protected splits"),
        default_recipe=_recipe("classification", grouped_asset_field="media"),
    ),
    _scenario(
        "audio-tts",
        label="Text to speech",
        description="Generate speech from text.",
        modality="audio",
        schema="audio",
        task="text_to_speech",
        required=("audio", "task", "transcript"),
        optional=(),
        aliases={"audio": ("audio", "audio_path", "file"), "transcript": ("transcript", "text")},
        constants={"task": "synthesize"},
        trainers=(),
        models=(),
        examples=(),
        available=False,
        unavailable_reason="Halo Forge does not have a verified TTS trainer contract.",
    ),
)


class TrainingScenarioRegistry:
    def __init__(self, scenarios: Iterable[TrainingScenarioDescriptor] = _SCENARIOS):
        self._by_id: Dict[str, TrainingScenarioDescriptor] = {}
        self._by_revision: Dict[str, TrainingScenarioDescriptor] = {}
        for scenario in scenarios:
            if scenario.id in self._by_id or scenario.revision_id in self._by_revision:
                raise ValueError(f"duplicate training scenario: {scenario.id}")
            self._by_id[scenario.id] = scenario
            self._by_revision[scenario.revision_id] = scenario
        canonical = [
            item.to_dict(include_examples=True) for item in self.list(include_unavailable=True)
        ]
        self.revision = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def list(self, *, include_unavailable: bool = True) -> list[TrainingScenarioDescriptor]:
        values = sorted(self._by_id.values(), key=lambda item: (item.modality, item.label, item.id))
        return values if include_unavailable else [item for item in values if item.available]

    def get(self, identifier: str) -> TrainingScenarioDescriptor:
        value = self._by_revision.get(identifier) or self._by_id.get(identifier)
        if value is None:
            raise KeyError(identifier)
        return value

    def template_bytes(
        self, identifier: str, example_id: Optional[str] = None
    ) -> tuple[str, bytes]:
        filename, files = self.template_files(identifier, example_id)
        return filename, files[filename]

    def template_files(
        self, identifier: str, example_id: Optional[str] = None
    ) -> tuple[str, Dict[str, bytes]]:
        """Return one complete fixture, including tiny existing media assets."""

        scenario = self.get(identifier)
        if not scenario.examples:
            raise ValueError(scenario.unavailable_reason or "scenario has no template")
        example = next(
            (item for item in scenario.examples if example_id is None or item.id == example_id),
            None,
        )
        if example is None:
            raise KeyError(example_id)
        payload = "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in example.records
        )
        if example.format in {"txt", "text", "markdown", "md"}:
            parts = []
            for record in example.records:
                title = str(record.get("title") or "").strip()
                text = str(record.get("text") or "")
                if title and not text.lstrip().startswith("#"):
                    parts.append(f"# {title}\n\n{text}".rstrip())
                else:
                    parts.append(text.rstrip())
            payload = "\n\n---\n\n".join(parts).rstrip() + "\n"
        files: Dict[str, bytes] = {example.filename: payload.encode("utf-8")}
        for record in example.records:
            for value in record.values():
                if not isinstance(value, str) or value in files:
                    continue
                content = _fixture_asset(value)
                if content is not None:
                    files[value] = content
        return example.filename, files


TRAINING_SCENARIOS = TrainingScenarioRegistry()


def interface_capabilities(*, backend_name: str = "unknown") -> list[InterfaceCapabilityDescriptor]:
    backend = str(backend_name or "unknown").lower()
    mac_signed = str(os.environ.get("HALOFORGE_SIGNATURE_STATE") or "unsigned") in {
        "signed",
        "signed_notarized",
    }
    descriptors = [
        InterfaceCapabilityDescriptor(
            id="desktop-macos-arm64",
            kind="execution_surface",
            label="macOS desktop",
            status="supported" if mac_signed else "preview",
            available=True,
            requirements=("macOS arm64",),
            reason=None if mac_signed else "Unsigned desktop candidates are preview artifacts; browser and CLI remain supported.",
            metadata={
                "execution_surface": "desktop",
                "presentation": "desktop",
                "local": True,
                "platform": "macos-arm64",
                "distribution_status": "signed" if mac_signed else "unsigned_preview",
                "signed_public_artifact": mac_signed,
                "package_type": "dmg",
            },
        ),
        InterfaceCapabilityDescriptor(
            id="desktop-linux",
            kind="execution_surface",
            label="Linux desktop",
            status="preview",
            available=True,
            reason="AppImage and Debian candidates are unsigned preview artifacts.",
            metadata={"execution_surface": "desktop", "platform": "linux-x86_64", "package_type": "appimage/deb", "distribution_status": "unsigned_preview"},
        ),
        InterfaceCapabilityDescriptor(
            id="desktop-windows",
            kind="execution_surface",
            label="Windows desktop",
            status="preview",
            available=True,
            reason="The NSIS candidate is an unsigned preview artifact.",
            metadata={"execution_surface": "desktop", "platform": "windows-x86_64", "package_type": "nsis", "distribution_status": "unsigned_preview"},
        ),
        InterfaceCapabilityDescriptor(
            id="browser-local",
            kind="execution_surface",
            label="Local browser dashboard",
            status="supported",
            available=True,
            metadata={
                "execution_surface": "local_browser",
                "presentation": "browser",
                "local": True,
            },
        ),
        InterfaceCapabilityDescriptor(
            id="browser-remote",
            kind="execution_surface",
            label="Remote browser dashboard",
            status="supported",
            available=True,
            requirements=("Reachable Halo Forge workstation",),
            metadata={
                "execution_surface": "remote_browser",
                "presentation": "browser",
                "local": False,
            },
        ),
        InterfaceCapabilityDescriptor(
            id="cli",
            kind="execution_surface",
            label="Command line",
            status="supported",
            available=True,
            metadata={"execution_surface": "cli", "platforms": ["macos", "linux", "windows"]},
        ),
    ]
    for modality, label in (
        ("text", "Text datasets"),
        ("image", "Image and document datasets"),
        ("audio", "Audio datasets"),
    ):
        descriptors.append(
            InterfaceCapabilityDescriptor(
                id=f"modality:{modality}",
                kind="modality",
                label=label,
                status="verified",
                available=True,
                metadata={"modality": modality},
            )
        )
    for shape in (
        "sft",
        "chat",
        "preference",
        "prompt",
        "tool",
        "vlm",
        "audio",
        "corpus",
        "classification",
        "embedding",
        "reranking",
    ):
        descriptors.append(
            InterfaceCapabilityDescriptor(
                id=f"canonical-shape:{shape}",
                kind="canonical_shape",
                label=shape.upper() if shape in {"sft", "vlm"} else shape.replace("_", " ").title(),
                status="verified",
                available=True,
                metadata={"canonical_shape": shape},
            )
        )
    for source_kind, label in (
        ("desktop_reference", "Desktop file or folder"),
        ("workstation_path", "Workstation path"),
        ("upload", "Browser upload"),
        ("huggingface", "Pinned Hugging Face dataset"),
    ):
        descriptors.append(
            InterfaceCapabilityDescriptor(
                id=f"source:{source_kind}",
                kind="source",
                label=label,
                status="supported",
                available=True,
                metadata={"source_kind": source_kind},
            )
        )
    for scenario in TRAINING_SCENARIOS.list(include_unavailable=True):
        descriptors.append(
            InterfaceCapabilityDescriptor(
                id=f"scenario:{scenario.id}",
                kind="training_scenario",
                label=scenario.label,
                status="verified" if scenario.available else "unavailable",
                available=scenario.available,
                reason=scenario.unavailable_reason,
                metadata={
                    "scenario_id": scenario.id,
                    "scenario_revision_id": scenario.revision_id,
                    "modality": scenario.modality,
                    "canonical_schema": scenario.canonical_schema,
                    "canonical_shape": scenario.canonical_schema,
                    "trainer_modes": list(scenario.trainer_modes),
                    "model_families": list(scenario.model_families),
                    "active_backend": backend,
                },
            )
        )
    return descriptors


__all__ = ["TRAINING_SCENARIOS", "TrainingScenarioRegistry", "interface_capabilities"]
