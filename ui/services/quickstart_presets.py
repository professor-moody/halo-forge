"""Canonical quickstart preset registry for training, benchmark, and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Optional


WorkflowType = Literal["training", "benchmark", "inference"]
RuntimeSize = Literal["quick", "medium"]


@dataclass(frozen=True)
class QuickstartFieldSet:
    """Quickstart form field grouping."""

    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuickstartRecommendation:
    """Plain-language recommendation metadata."""

    when_to_use: str
    expected_runtime: RuntimeSize


@dataclass(frozen=True)
class QuickstartPreset:
    """Single quickstart preset definition."""

    key: str
    workflow: WorkflowType
    target: str
    label: str
    description: str
    field_set: QuickstartFieldSet
    recommendation: QuickstartRecommendation
    values: Dict[str, Any] = field(default_factory=dict)


_PRESETS: Dict[str, QuickstartPreset] = {
    # Training
    "sft_fast_local": QuickstartPreset(
        key="sft_fast_local",
        workflow="training",
        target="sft",
        label="SFT Fast Local",
        description="Single-epoch SFT starter for first successful local launch.",
        field_set=QuickstartFieldSet(
            required_fields=("model", "dataset", "output_dir"),
            optional_fields=("epochs", "max_samples"),
        ),
        recommendation=QuickstartRecommendation(
            when_to_use="First local SFT run and smoke validation.",
            expected_runtime="quick",
        ),
        values={
            "model": "Qwen/Qwen2.5-Coder-1.5B",
            "dataset": "codealpaca",
            "output_dir": "models/sft_run",
            "epochs": 1,
            "max_samples": 200,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
        },
    ),
    "raft_safe_default": QuickstartPreset(
        key="raft_safe_default",
        workflow="training",
        target="raft",
        label="RAFT Safe Default",
        description="Conservative RAFT loop for first-run reward pipeline verification.",
        field_set=QuickstartFieldSet(
            required_fields=("model", "prompts", "output_dir"),
            optional_fields=("cycles", "samples_per_prompt", "verifier"),
        ),
        recommendation=QuickstartRecommendation(
            when_to_use="Initial RAFT pipeline validation with low risk defaults.",
            expected_runtime="quick",
        ),
        values={
            "model": "Qwen/Qwen2.5-Coder-3B",
            "prompts": "data/rlvr/humaneval_prompts.jsonl",
            "output_dir": "models/raft_run",
            "cycles": 2,
            "samples_per_prompt": 6,
            "temperature": 0.7,
            "keep_percent": 0.6,
            "verifier": "humaneval",
        },
    ),
    "vlm_tiny": QuickstartPreset(
        key="vlm_tiny",
        workflow="training",
        target="vlm",
        label="VLM Tiny",
        description="Quick VLM RAFT run with compact defaults.",
        field_set=QuickstartFieldSet(
            required_fields=("model", "dataset", "output_dir"),
            optional_fields=("cycles", "samples_per_prompt"),
        ),
        recommendation=QuickstartRecommendation(
            when_to_use="Sanity-check VLM training loop quickly.",
            expected_runtime="quick",
        ),
        values={
            "model": "Qwen/Qwen2-VL-2B-Instruct",
            "dataset": "textvqa",
            "output_dir": "models/vlm_raft",
            "cycles": 2,
            "samples_per_prompt": 3,
            "learning_rate": 5e-5,
        },
    ),
    "audio_whisper_tiny": QuickstartPreset(
        key="audio_whisper_tiny",
        workflow="training",
        target="audio",
        label="Audio Whisper Tiny",
        description="Fast ASR training loop verification on Whisper family.",
        field_set=QuickstartFieldSet(
            required_fields=("model", "dataset", "output_dir"),
            optional_fields=("cycles", "task"),
        ),
        recommendation=QuickstartRecommendation(
            when_to_use="Validate audio training path in minimum time.",
            expected_runtime="quick",
        ),
        values={
            "model": "openai/whisper-tiny",
            "dataset": "librispeech",
            "output_dir": "models/audio_raft",
            "task": "asr",
            "cycles": 2,
            "samples_per_prompt": 3,
            "learning_rate": 5e-5,
        },
    ),
    "reasoning_small": QuickstartPreset(
        key="reasoning_small",
        workflow="training",
        target="reasoning",
        label="Reasoning Small",
        description="Reasoning module quickstart with bounded cycles.",
        field_set=QuickstartFieldSet(
            required_fields=("model", "dataset", "output_dir"),
            optional_fields=("cycles", "limit"),
        ),
        recommendation=QuickstartRecommendation(
            when_to_use="Check reasoning training orchestration and artifacts.",
            expected_runtime="quick",
        ),
        values={
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "dataset": "gsm8k",
            "output_dir": "models/reasoning_raft",
            "cycles": 2,
            "limit": 64,
            "learning_rate": 1e-5,
        },
    ),
    "agentic_small": QuickstartPreset(
        key="agentic_small",
        workflow="training",
        target="agentic",
        label="Agentic Small",
        description="Agentic quickstart for function-calling training flow checks.",
        field_set=QuickstartFieldSet(
            required_fields=("model", "dataset", "output_dir"),
            optional_fields=("cycles", "limit"),
        ),
        recommendation=QuickstartRecommendation(
            when_to_use="Validate agentic loop, resume metadata, and summaries.",
            expected_runtime="quick",
        ),
        values={
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "dataset": "xlam",
            "output_dir": "models/agentic_raft",
            "cycles": 2,
            "limit": 64,
            "learning_rate": 5e-5,
        },
    ),
    # Benchmark
    "code_smoke": QuickstartPreset(
        key="code_smoke",
        workflow="benchmark",
        target="code",
        label="Code Smoke",
        description="Small code benchmark run for first result generation.",
        field_set=QuickstartFieldSet(
            required_fields=("model", "benchmark", "output_dir"),
            optional_fields=("limit", "samples_per_prompt"),
        ),
        recommendation=QuickstartRecommendation(
            when_to_use="Generate first benchmark.json quickly for code models.",
            expected_runtime="quick",
        ),
        values={
            "benchmark_dataset": "humaneval",
            "limit": 25,
            "samples_per_prompt": 3,
            "verifier": "humaneval",
            "run_after_compile": True,
        },
    ),
    "non_code_smoke": QuickstartPreset(
        key="non_code_smoke",
        workflow="benchmark",
        target="non_code",
        label="Non-Code Smoke",
        description="Tiny non-code benchmark run for first result artifact.",
        field_set=QuickstartFieldSet(
            required_fields=("model", "benchmark", "output_dir"),
            optional_fields=("limit",),
        ),
        recommendation=QuickstartRecommendation(
            when_to_use="Validate VLM/audio/reasoning/agentic benchmark flow quickly.",
            expected_runtime="quick",
        ),
        values={
            "benchmark_type": "vlm",
            "benchmark_dataset": "textvqa",
            "limit": 20,
        },
    ),
    # Inference
    "optimize_int4_smoke": QuickstartPreset(
        key="optimize_int4_smoke",
        workflow="inference",
        target="optimize",
        label="Optimize INT4 Smoke",
        description="Quick optimization profile for first inference artifact.",
        field_set=QuickstartFieldSet(
            required_fields=("model", "output_dir"),
            optional_fields=("target_precision", "target_latency", "dry_run"),
        ),
        recommendation=QuickstartRecommendation(
            when_to_use="First optimization pass before benchmarking.",
            expected_runtime="quick",
        ),
        values={
            "target_precision": "int4",
            "target_latency": 50.0,
            "output_dir": "models/optimized",
            "dry_run": True,
        },
    ),
    "benchmark_latency_smoke": QuickstartPreset(
        key="benchmark_latency_smoke",
        workflow="inference",
        target="benchmark",
        label="Benchmark Latency Smoke",
        description="Short inference benchmark for latency/memory sanity checks.",
        field_set=QuickstartFieldSet(
            required_fields=("model", "output_dir"),
            optional_fields=("num_prompts", "max_tokens", "warmup"),
        ),
        recommendation=QuickstartRecommendation(
            when_to_use="Quick runtime latency check on an optimized or base model.",
            expected_runtime="quick",
        ),
        values={
            "output_dir": "results/inference_benchmarks",
            "num_prompts": 10,
            "max_tokens": 100,
            "warmup": 3,
            "measure_memory": True,
        },
    ),
}


def list_quickstart_presets(
    workflow: WorkflowType,
    *,
    target: Optional[str] = None,
) -> list[QuickstartPreset]:
    presets = [p for p in _PRESETS.values() if p.workflow == workflow]
    if target:
        target_key = str(target).strip().lower()
        presets = [p for p in presets if p.target == target_key]
    return sorted(presets, key=lambda p: p.key)


def get_quickstart_preset(
    workflow: WorkflowType,
    preset_key: str,
    *,
    target: Optional[str] = None,
) -> Optional[QuickstartPreset]:
    key = str(preset_key or "").strip().lower()
    preset = _PRESETS.get(key)
    if not preset:
        return None
    if preset.workflow != workflow:
        return None
    if target and preset.target != str(target).strip().lower():
        return None
    return preset


def default_preset_key(workflow: WorkflowType, target: str) -> Optional[str]:
    presets = list_quickstart_presets(workflow, target=target)
    if not presets:
        return None
    return presets[0].key


def apply_preset_values(target_obj: Any, values: Dict[str, Any], *, allowed_fields: Optional[Iterable[str]] = None) -> None:
    """Assign preset values to object attributes (optionally whitelisted)."""
    allowed = set(allowed_fields or [])
    use_filter = bool(allowed_fields)
    for key, value in values.items():
        if use_filter and key not in allowed:
            continue
        if hasattr(target_obj, key):
            setattr(target_obj, key, value)
