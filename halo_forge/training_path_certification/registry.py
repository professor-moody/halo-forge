"""Versioned built-in paths and small, reviewable certification fixtures."""

from __future__ import annotations

import hashlib
import json
from typing import Any


PROFILE_VERSION = "1"
TRAINER_ADAPTER_VERSION = "v21-real-entrypoint-1"
CAPACITY_ADAPTER_VERSION = "v21-disposable-step-1"
TOKENIZER_PROCESSOR_CONTRACT = "resolved-content-hash-at-certification"

SFT_FIXTURE: tuple[dict[str, str], ...] = tuple(
    {
        "prompt": prompt,
        "response": response,
    }
    for prompt, response in (
        ("Say hello politely.", "Hello! It is nice to meet you."),
        ("Name the largest ocean.", "The Pacific Ocean."),
        ("Add 2 and 3.", "5"),
        ("What color is a clear daytime sky?", "Blue."),
        ("Finish: water freezes at", "zero degrees Celsius."),
        ("Give one benefit of exercise.", "It can improve cardiovascular health."),
        ("Translate yes to Spanish.", "Sí."),
        ("Write a short greeting.", "Good morning!"),
        ("Which planet is closest to the Sun?", "Mercury."),
        ("Opposite of cold?", "Hot."),
        ("Name a primary color.", "Red."),
        ("What do bees make?", "Honey."),
    )
)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


FIXTURES: dict[str, tuple[dict[str, Any], ...]] = {
    "instruction-sft-v1": SFT_FIXTURE,
}


PATH_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "profile_id": "instruction-sft-qwen-0.5b",
        "label": "Instruction training",
        "scenario_revision_id": "instruction-sft-v1",
        "trainer_mode": "sft",
        "model_id": "Qwen/Qwen2.5-Coder-0.5B",
        "fixture_id": "instruction-sft-v1",
        "expected_artifacts": ("training_summary.json", "final_model/adapter_config.json"),
        "recommended": True,
    },
    {"profile_id": "chat-sft-qwen-0.5b", "label": "Multi-turn chat training", "scenario_revision_id": "chat-sft-v1", "trainer_mode": "sft", "model_id": "Qwen/Qwen2.5-0.5B", "fixture_id": "chat-sft-v1"},
    {"profile_id": "tool-sft-qwen-0.5b", "label": "Tool-call training", "scenario_revision_id": "tool-agentic-v1", "trainer_mode": "agentic", "model_id": "Qwen/Qwen2.5-0.5B", "fixture_id": "tool-agentic-v1"},
    {"profile_id": "preference-dpo-qwen-0.5b", "label": "Preference training (DPO)", "scenario_revision_id": "preference-pairs-v1", "trainer_mode": "dpo", "model_id": "Qwen/Qwen2.5-0.5B", "fixture_id": "preference-v1"},
    {"profile_id": "preference-orpo-qwen-0.5b", "label": "Preference training (ORPO)", "scenario_revision_id": "preference-pairs-v1", "trainer_mode": "orpo", "model_id": "Qwen/Qwen2.5-0.5B", "fixture_id": "preference-v1"},
    {"profile_id": "reward-model-qwen-0.5b", "label": "Reward model training", "scenario_revision_id": "preference-pairs-v1", "trainer_mode": "rm", "model_id": "Qwen/Qwen2.5-0.5B", "fixture_id": "preference-v1"},
    {"profile_id": "cpt-qwen-0.5b", "label": "Continued pretraining", "scenario_revision_id": "corpus-cpt-v1", "trainer_mode": "cpt", "model_id": "Qwen/Qwen2.5-0.5B", "fixture_id": "corpus-v1"},
    {"profile_id": "raft-qwen-0.5b", "label": "Verifier-guided RAFT", "scenario_revision_id": "prompt-reward-v1", "trainer_mode": "raft", "model_id": "Qwen/Qwen2.5-0.5B", "fixture_id": "prompt-verifier-v1", "certification_only_verifier": True},
    {"profile_id": "grpo-qwen-0.5b", "label": "Verifier-guided GRPO", "scenario_revision_id": "prompt-reward-v1", "trainer_mode": "grpo", "model_id": "Qwen/Qwen2.5-0.5B", "fixture_id": "prompt-verifier-v1", "certification_only_verifier": True},
    {"profile_id": "reasoning-qwen-0.5b", "label": "Reasoning training", "scenario_revision_id": "reasoning-trace-v1", "trainer_mode": "reasoning", "model_id": "Qwen/Qwen2.5-0.5B", "fixture_id": "reasoning-v1"},
    {"profile_id": "vlm-qwen2-vl-2b", "label": "Vision-language training", "scenario_revision_id": "vlm-caption-v1", "trainer_mode": "vlm", "model_id": "Qwen/Qwen2-VL-2B-Instruct", "fixture_id": "vlm-v1"},
    {"profile_id": "asr-whisper-tiny", "label": "Speech recognition training", "scenario_revision_id": "audio-asr-v1", "trainer_mode": "audio", "model_id": "openai/whisper-tiny", "fixture_id": "asr-v1"},
    {"profile_id": "text-classification-distilbert", "label": "Text classification", "scenario_revision_id": "text-classification-v1", "trainer_mode": "classify", "model_id": "distilbert/distilbert-base-uncased", "fixture_id": "classification-v1"},
    {"profile_id": "embedding-minilm-l6", "label": "Embedding training", "scenario_revision_id": "embedding-pairs-v1", "trainer_mode": "embed", "model_id": "sentence-transformers/all-MiniLM-L6-v2", "fixture_id": "embedding-v1"},
    {"profile_id": "reranking-msmarco-minilm", "label": "Reranker training", "scenario_revision_id": "reranking-v1", "trainer_mode": "rerank", "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2", "fixture_id": "reranking-v1"},
    {"profile_id": "image-classification-vit", "label": "Image classification", "scenario_revision_id": "image-classification-v1", "trainer_mode": "classify", "model_id": "google/vit-base-patch16-224-in21k", "fixture_id": "image-classification-v1"},
    {"profile_id": "audio-classification-wav2vec2", "label": "Audio classification", "scenario_revision_id": "audio-classification-v1", "trainer_mode": "classify", "model_id": "facebook/wav2vec2-base", "fixture_id": "audio-classification-v1"},
)


def normalized_definition(raw: dict[str, Any], runtime_family: str) -> dict[str, Any]:
    fixture_id = str(raw["fixture_id"])
    fixture = FIXTURES.get(fixture_id)
    fixture_hash = _hash(fixture) if fixture is not None else _hash({"fixture_id": fixture_id})
    return {
        **raw,
        "runtime_family": runtime_family,
        "backend": "pytorch",
        "model_revision": "resolve-to-immutable-commit",
        "tokenizer_processor_hash": TOKENIZER_PROCESSOR_CONTRACT,
        "fixture_hash": fixture_hash,
        "trainer_adapter_version": TRAINER_ADAPTER_VERSION,
        "capacity_adapter_version": CAPACITY_ADAPTER_VERSION,
        "expected_artifacts": tuple(raw.get("expected_artifacts") or ("training_summary.json",)),
        "configuration": {
            "proof_steps": 1,
            "seed": 42,
            "dataset_renderer": "training-artifact-v3",
            "real_entrypoint_required": True,
            "parameter_delta_required": True,
            "artifact_reload_required": True,
            "certification_only_verifier": bool(raw.get("certification_only_verifier")),
            "executor_available": fixture_id in FIXTURES,
        },
    }


__all__ = ["FIXTURES", "PATH_DEFINITIONS", "PROFILE_VERSION", "normalized_definition"]
