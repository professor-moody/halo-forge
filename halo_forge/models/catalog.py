"""Static curated model catalog for training, serving, and docs surfaces.

This is deliberately not a live Hugging Face search. The catalog is the
small, opinionated list Halo Forge can explain, filter, and keep consistent
across CLI, public API, dashboard quick-picks, and documentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional


CATALOG_VERSION = "2026.05"


def _estimate_memory_gb(parameter_count: str, memory_tier: str) -> Optional[float]:
    """Conservative first-run memory estimate for LoRA-style local workflows."""
    try:
        normalized = parameter_count.strip().upper()
        if normalized.endswith("M"):
            params_b = float(normalized[:-1]) / 1000
        elif normalized.endswith("B"):
            params_b = float(normalized[:-1])
        else:
            return {"tiny": 2.0, "small": 8.0, "medium": 18.0, "large": 40.0}.get(memory_tier)
        # Rough bf16 model footprint plus optimizer/activation headroom for LoRA training.
        return round(max(1.5, params_b * 3.2 + 1.0), 1)
    except Exception:
        return {"tiny": 2.0, "small": 8.0, "medium": 18.0, "large": 40.0}.get(memory_tier)


@dataclass(frozen=True)
class ModelCatalogEntry:
    id: str
    label: str
    provider: str
    family: str
    parameter_count: str
    modalities: tuple[str, ...]
    tasks: tuple[str, ...]
    trainer_support: tuple[str, ...]
    backend_support: tuple[str, ...]
    memory_tier: str
    recommended_use: str
    known_caveats: tuple[str, ...] = ()
    trust_remote_code_required: bool = False
    mlx_variant: Optional[str] = None
    status: str = "recommended"
    recommended_first_run: bool = False
    estimated_memory_gb: Optional[float] = None
    license_note: Optional[str] = None
    download_note: Optional[str] = None
    fit_notes: tuple[str, ...] = ()
    risk_level: str = "safe"
    last_verified: str = "2026-05-09"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in (
            "modalities",
            "tasks",
            "trainer_support",
            "backend_support",
            "known_caveats",
            "fit_notes",
        ):
            data[key] = list(data[key])
        data["catalog_version"] = CATALOG_VERSION
        return data


def _entry(
    id: str,
    label: str,
    provider: str,
    family: str,
    parameter_count: str,
    modalities: Iterable[str],
    tasks: Iterable[str],
    trainer_support: Iterable[str],
    backend_support: Iterable[str],
    memory_tier: str,
    recommended_use: str,
    *,
    known_caveats: Iterable[str] = (),
    trust_remote_code_required: bool = False,
    mlx_variant: Optional[str] = None,
    status: str = "recommended",
    recommended_first_run: bool = False,
    estimated_memory_gb: Optional[float] = None,
    license_note: Optional[str] = None,
    download_note: Optional[str] = None,
    fit_notes: Iterable[str] = (),
    risk_level: Optional[str] = None,
) -> ModelCatalogEntry:
    risk = risk_level
    if risk is None:
        risk = "experimental" if status == "experimental" else "caveated" if known_caveats or trust_remote_code_required else "safe"
    if estimated_memory_gb is None:
        estimated_memory_gb = _estimate_memory_gb(parameter_count, memory_tier)
    if license_note is None and any("license" in str(c).lower() for c in known_caveats):
        license_note = "Accept the upstream model license before download."
    if download_note is None and trust_remote_code_required:
        download_note = "Requires explicit trust-remote-code opt-in where supported."
    return ModelCatalogEntry(
        id=id,
        label=label,
        provider=provider,
        family=family,
        parameter_count=parameter_count,
        modalities=tuple(modalities),
        tasks=tuple(tasks),
        trainer_support=tuple(trainer_support),
        backend_support=tuple(backend_support),
        memory_tier=memory_tier,
        recommended_use=recommended_use,
        known_caveats=tuple(known_caveats),
        trust_remote_code_required=trust_remote_code_required,
        mlx_variant=mlx_variant,
        status=status,
        recommended_first_run=recommended_first_run,
        estimated_memory_gb=estimated_memory_gb,
        license_note=license_note,
        download_note=download_note,
        fit_notes=tuple(fit_notes),
        risk_level=risk,
    )


_MODELS: tuple[ModelCatalogEntry, ...] = (
    _entry(
        "Qwen/Qwen2.5-Coder-0.5B",
        "Qwen2.5 Coder 0.5B",
        "Qwen",
        "Qwen2.5-Coder",
        "0.5B",
        ("text", "code"),
        ("code", "quickstart", "sft", "raft"),
        ("sft", "raft", "grpo"),
        ("cpu", "cuda", "rocm_gfx1151", "rocm", "mps"),
        "tiny",
        "Fastest code smoke tests and CI-friendly trainer checks.",
        mlx_variant="mlx-community/Qwen2.5-0.5B-Instruct-bf16",
        recommended_first_run=True,
        fit_notes=("Best when the operator needs to validate install, dataset shape, and launch plumbing quickly.",),
    ),
    _entry(
        "Qwen/Qwen2.5-Coder-1.5B",
        "Qwen2.5 Coder 1.5B",
        "Qwen",
        "Qwen2.5-Coder",
        "1.5B",
        ("text", "code"),
        ("code", "sft", "raft"),
        ("sft", "raft", "grpo"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Low-memory code fine-tuning with enough capacity to show meaningful gains.",
        recommended_first_run=True,
        fit_notes=("Default first real code-training pick on PyTorch backends.",),
    ),
    _entry(
        "Qwen/Qwen2.5-Coder-3B",
        "Qwen2.5 Coder 3B",
        "Qwen",
        "Qwen2.5-Coder",
        "3B",
        ("text", "code"),
        ("code", "sft", "raft", "benchmark"),
        ("sft", "raft", "grpo"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Recommended starting point for code SFT and verifier-ranked RAFT.",
    ),
    _entry(
        "Qwen/Qwen2.5-Coder-7B",
        "Qwen2.5 Coder 7B",
        "Qwen",
        "Qwen2.5-Coder",
        "7B",
        ("text", "code"),
        ("code", "sft", "raft", "production"),
        ("sft", "raft", "grpo"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "medium",
        "Higher-quality code runs when you can afford the memory and wall-clock.",
    ),
    _entry(
        "Qwen/Qwen2.5-0.5B",
        "Qwen2.5 0.5B",
        "Qwen",
        "Qwen2.5",
        "0.5B",
        ("text",),
        ("reward-model", "quickstart", "sft"),
        ("sft", "rm"),
        ("cpu", "cuda", "rocm_gfx1151", "rocm", "mps"),
        "tiny",
        "Tiny base model for reward-model smoke tests and low-cost local experiments.",
    ),
    _entry(
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen2.5 Instruct 1.5B",
        "Qwen",
        "Qwen2.5-Instruct",
        "1.5B",
        ("text",),
        ("chat", "reasoning", "agentic", "quickstart"),
        ("sft", "dpo", "orpo", "grpo", "reasoning", "agentic"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Small general-purpose instruct model for reasoning and agentic quickstarts.",
        recommended_first_run=True,
        fit_notes=("Good first instruct model when the task is not code-specific.",),
    ),
    _entry(
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5 Instruct 3B",
        "Qwen",
        "Qwen2.5-Instruct",
        "3B",
        ("text",),
        ("chat", "preference", "agentic", "serving"),
        ("sft", "dpo", "orpo", "grpo", "rm", "agentic"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Default preference-tuning and local chat refinement model.",
        mlx_variant="mlx-community/Qwen2.5-3B-Instruct-bf16",
    ),
    _entry(
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5 Instruct 7B",
        "Qwen",
        "Qwen2.5-Instruct",
        "7B",
        ("text",),
        ("chat", "reasoning", "agentic", "preference"),
        ("sft", "dpo", "orpo", "grpo", "rm", "reasoning", "agentic"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "medium",
        "General-purpose 7B base for higher-quality preference, reasoning, and tool-use runs.",
        mlx_variant="mlx-community/Qwen2.5-7B-Instruct-bf16",
    ),
    _entry(
        "Qwen/Qwen2.5-Math-1.5B",
        "Qwen2.5 Math 1.5B",
        "Qwen",
        "Qwen2.5-Math",
        "1.5B",
        ("text",),
        ("math", "reasoning"),
        ("reasoning", "grpo", "sft"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Math/reasoning templates where symbolic answer checks are available.",
    ),
    _entry(
        "meta-llama/Llama-3.2-3B-Instruct",
        "Llama 3.2 3B Instruct",
        "Meta",
        "Llama 3.2",
        "3B",
        ("text",),
        ("chat", "preference", "serving"),
        ("sft", "dpo", "orpo", "rm"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Llama-family baseline for chat, preference tuning, and judge-model experiments.",
        known_caveats=("Requires accepting Meta license terms before download.",),
    ),
    _entry(
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Mistral 7B Instruct v0.3",
        "Mistral AI",
        "Mistral",
        "7B",
        ("text",),
        ("chat", "preference", "serving"),
        ("sft", "dpo", "orpo", "rm"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "medium",
        "Strong general instruct baseline when a Mistral-style tokenizer/template is desired.",
    ),
    _entry(
        "google/gemma-2-2b-it",
        "Gemma 2 2B IT",
        "Google",
        "Gemma 2",
        "2B",
        ("text",),
        ("chat", "sft", "serving"),
        ("sft", "dpo", "orpo"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Compact general instruct model for low-memory experiments.",
        known_caveats=("Requires accepting Google Gemma license terms before download.",),
    ),
    _entry(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        "DeepSeek Coder 1.3B Instruct",
        "DeepSeek",
        "DeepSeek-Coder",
        "1.3B",
        ("text", "code"),
        ("code", "sft", "raft"),
        ("sft", "raft", "grpo"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Alternative small code model for compiler-verifier experiments.",
        trust_remote_code_required=True,
    ),
    _entry(
        "bigcode/starcoder2-3b",
        "StarCoder2 3B",
        "BigCode",
        "StarCoder2",
        "3B",
        ("text", "code"),
        ("code", "sft", "raft"),
        ("sft", "raft", "grpo"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Multi-language code baseline from BigCode.",
    ),
    _entry(
        "codellama/CodeLlama-7b-hf",
        "CodeLlama 7B",
        "Meta",
        "CodeLlama",
        "7B",
        ("text", "code"),
        ("code", "sft", "raft"),
        ("sft", "raft"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "medium",
        "Older but still useful Llama-family code baseline.",
        known_caveats=("Larger CodeLlama variants need much more memory.",),
    ),
    _entry(
        "microsoft/Phi-3.5-mini-instruct",
        "Phi 3.5 Mini Instruct",
        "Microsoft",
        "Phi",
        "3.8B",
        ("text",),
        ("chat", "sft", "serving"),
        ("sft", "dpo", "orpo"),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Compact instruct alternative; verify chat template behavior before production runs.",
        trust_remote_code_required=True,
        status="compatible",
    ),
    _entry(
        "openai/whisper-tiny",
        "Whisper Tiny",
        "OpenAI",
        "Whisper",
        "39M",
        ("audio",),
        ("asr", "audio", "quickstart"),
        ("audio",),
        ("cuda", "rocm_gfx1151", "rocm", "mps", "cpu"),
        "tiny",
        "Fast audio/ASR smoke tests.",
        recommended_first_run=True,
        fit_notes=("Use before Whisper Small to verify audio dataset formatting.",),
    ),
    _entry(
        "openai/whisper-small",
        "Whisper Small",
        "OpenAI",
        "Whisper",
        "244M",
        ("audio",),
        ("asr", "audio"),
        ("audio",),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "tiny",
        "Default ASR fine-tuning model for audio workflows.",
    ),
    _entry(
        "Qwen/Qwen2-VL-2B-Instruct",
        "Qwen2-VL 2B Instruct",
        "Qwen",
        "Qwen2-VL",
        "2B",
        ("vision", "text"),
        ("vlm", "vqa", "document-extraction"),
        ("vlm",),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Fast VLM qualification and visual question answering.",
        trust_remote_code_required=True,
    ),
    _entry(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL 3B Instruct",
        "Qwen",
        "Qwen2.5-VL",
        "3B",
        ("vision", "text"),
        ("vlm", "vqa", "document-extraction"),
        ("vlm",),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Recommended compact VLM target for document extraction and VQA scenarios.",
        trust_remote_code_required=True,
    ),
    _entry(
        "LiquidAI/LFM2.5-350M",
        "LFM2.5 350M",
        "Liquid AI",
        "LFM2.5",
        "350M",
        ("text",),
        ("structured-output", "tool-use", "edge", "serving"),
        ("sft", "dpo", "grpo"),
        ("cpu", "cuda", "rocm_gfx1151", "rocm", "mps", "mlx"),
        "tiny",
        "Interesting tiny model for structured output, tool use, extraction, and edge experiments.",
        known_caveats=("Liquid notes it is not recommended for knowledge-intensive tasks or programming.",),
        mlx_variant="LiquidAI/LFM2.5-350M-MLX",
        status="experimental",
    ),
    _entry(
        "LiquidAI/LFM2.5-1.2B-Instruct",
        "LFM2.5 1.2B Instruct",
        "Liquid AI",
        "LFM2.5",
        "1.2B",
        ("text",),
        ("chat", "tool-use", "structured-output", "agentic"),
        ("sft", "dpo", "grpo", "agentic"),
        ("cuda", "rocm_gfx1151", "rocm", "mps", "mlx"),
        "small",
        "Liquid's recommended LFM2.5 chat/instruction model; promising for small local agentic workflows.",
        mlx_variant="LiquidAI/LFM2.5-1.2B-Instruct-MLX",
        status="experimental",
    ),
    _entry(
        "LiquidAI/LFM2.5-1.2B-Thinking",
        "LFM2.5 1.2B Thinking",
        "Liquid AI",
        "LFM2.5",
        "1.2B",
        ("text",),
        ("reasoning", "math"),
        ("reasoning", "grpo", "sft"),
        ("cuda", "rocm_gfx1151", "rocm", "mps", "mlx"),
        "small",
        "Liquid's reasoning-optimized small model for math and logic experiments.",
        mlx_variant="LiquidAI/LFM2.5-1.2B-Thinking-MLX",
        status="experimental",
    ),
    _entry(
        "LiquidAI/LFM2.5-1.2B-Base",
        "LFM2.5 1.2B Base",
        "Liquid AI",
        "LFM2.5",
        "1.2B",
        ("text",),
        ("sft", "continued-pretraining"),
        ("sft",),
        ("cuda", "rocm_gfx1151", "rocm", "mps", "mlx"),
        "small",
        "Base Liquid checkpoint for custom fine-tuning experiments.",
        mlx_variant="LiquidAI/LFM2.5-1.2B-Base-MLX",
        status="experimental",
    ),
    _entry(
        "LiquidAI/LFM2.5-VL-450M",
        "LFM2.5-VL 450M",
        "Liquid AI",
        "LFM2.5-VL",
        "450M",
        ("vision", "text"),
        ("vlm", "ocr", "visual-extraction"),
        ("vlm",),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "tiny",
        "Tiny Liquid vision-language candidate for edge visual extraction.",
        known_caveats=("Halo Forge VLM training uses adapter-specific paths; verify Liquid VL generation before relying on results.",),
        trust_remote_code_required=True,
        status="experimental",
    ),
    _entry(
        "LiquidAI/LFM2.5-VL-1.6B",
        "LFM2.5-VL 1.6B",
        "Liquid AI",
        "LFM2.5-VL",
        "1.6B",
        ("vision", "text"),
        ("vlm", "vqa", "document-extraction"),
        ("vlm",),
        ("cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Liquid's recommended vision-language model; keep as experimental until Halo Forge adapters are verified end-to-end.",
        known_caveats=("Adapter-dependent in Halo Forge; use Qwen-VL for the safest VLM path today.",),
        trust_remote_code_required=True,
        status="experimental",
    ),
    _entry(
        "LiquidAI/LFM2.5-Audio-1.5B",
        "LFM2.5-Audio 1.5B",
        "Liquid AI",
        "LFM2.5-Audio",
        "1.5B",
        ("audio", "text"),
        ("asr", "tts", "voice-chat"),
        ("audio",),
        ("cpu", "cuda", "rocm_gfx1151", "rocm", "mps"),
        "small",
        "Compact Liquid audio/text model for ASR/TTS experiments.",
        known_caveats=("Halo Forge audio path is Whisper-oriented today; Liquid audio needs adapter validation before first-class training.",),
        trust_remote_code_required=True,
        status="experimental",
    ),
    _entry(
        "mlx-community/Qwen2.5-0.5B-Instruct-bf16",
        "Qwen2.5 0.5B Instruct MLX",
        "mlx-community",
        "Qwen2.5-Instruct",
        "0.5B",
        ("text", "code"),
        ("mlx", "chat", "serving", "code"),
        ("sft", "dpo", "grpo", "raft"),
        ("mlx",),
        "tiny",
        "Smallest safe Apple Silicon MLX-format first-run model for proving training, verifier, and output paths.",
        recommended_first_run=True,
        download_note="MLX-format Hugging Face artifact; use on Apple Silicon with `--accelerator mlx`.",
        fit_notes=("Best first MLX pick for laptops and dashboard smoke runs.",),
    ),
    _entry(
        "mlx-community/Qwen2.5-3B-Instruct-bf16",
        "Qwen2.5 3B Instruct MLX",
        "mlx-community",
        "Qwen2.5-Instruct",
        "3B",
        ("text",),
        ("mlx", "chat", "serving"),
        ("sft", "dpo", "grpo", "raft"),
        ("mlx",),
        "small",
        "Apple Silicon MLX-format quickstart model for local inference and MLX-native trainer paths.",
        recommended_first_run=True,
        fit_notes=("Higher-quality MLX first-run pick when memory headroom is comfortable.",),
    ),
    _entry(
        "mlx-community/Qwen2.5-7B-Instruct-bf16",
        "Qwen2.5 7B Instruct MLX",
        "mlx-community",
        "Qwen2.5-Instruct",
        "7B",
        ("text",),
        ("mlx", "chat", "serving"),
        ("sft", "dpo", "grpo", "raft"),
        ("mlx",),
        "medium",
        "Higher-quality Apple Silicon MLX-format instruct model.",
    ),
    _entry(
        "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "Llama 3.2 3B Instruct MLX 4-bit",
        "mlx-community",
        "Llama 3.2",
        "3B",
        ("text",),
        ("mlx", "chat", "serving"),
        ("serve", "raft"),
        ("mlx",),
        "small",
        "Pre-quantized Apple Silicon serving and rollout candidate.",
        known_caveats=("Quantization is baked into the artifact; not bitsandbytes runtime quantization.",),
    ),
)


_BY_ID = {entry.id: entry for entry in _MODELS}
_CATALOG_ORDER = {entry.id: index for index, entry in enumerate(_MODELS)}


def _matches(value: str, candidates: Iterable[str]) -> bool:
    wanted = value.strip().lower()
    if not wanted:
        return True
    return wanted in {str(candidate).strip().lower() for candidate in candidates}


def _mode_matches(mode: str, entry: ModelCatalogEntry) -> bool:
    wanted = mode.strip().lower()
    if not wanted:
        return True
    aliases = {
        "code": ("sft", "raft", "grpo"),
        "preference": ("dpo", "orpo", "rm"),
        "chat": ("sft", "dpo", "orpo"),
        "serve": ("serve", "inference"),
    }
    modes = set(entry.trainer_support) | set(entry.tasks)
    for alias in aliases.get(wanted, (wanted,)):
        if alias in modes:
            return True
    return False


def list_models(filters: Optional[Mapping[str, Any]] = None) -> list[dict[str, Any]]:
    filters = filters or {}
    entries = list(_MODELS)
    mode = str(filters.get("mode") or "").strip()
    backend = str(filters.get("backend") or "").strip()
    modality = str(filters.get("modality") or "").strip()
    provider = str(filters.get("provider") or "").strip()
    status = str(filters.get("status") or "").strip()
    memory_tier = str(filters.get("memory_tier") or "").strip()

    if mode:
        entries = [entry for entry in entries if _mode_matches(mode, entry)]
    if backend:
        entries = [entry for entry in entries if _matches(backend, entry.backend_support)]
    if modality:
        entries = [entry for entry in entries if _matches(modality, entry.modalities)]
    if provider:
        entries = [entry for entry in entries if entry.provider.lower() == provider.lower()]
    if status:
        entries = [entry for entry in entries if entry.status.lower() == status.lower()]
    if memory_tier:
        entries = [entry for entry in entries if entry.memory_tier.lower() == memory_tier.lower()]

    rank = {"recommended": 0, "compatible": 1, "experimental": 2, "deprecated": 3}
    entries.sort(key=lambda e: (rank.get(e.status, 99), _CATALOG_ORDER.get(e.id, 9999)))
    return [entry.to_dict() for entry in entries]


def get_model(model_id: str) -> Optional[dict[str, Any]]:
    entry = _BY_ID.get(str(model_id or "").strip())
    return entry.to_dict() if entry else None


def recommended_models(
    mode: Optional[str] = None,
    backend: Optional[str] = None,
    modality: Optional[str] = None,
) -> list[dict[str, Any]]:
    filters = {
        "mode": mode or "",
        "backend": backend or "",
        "modality": modality or "",
    }
    items = list_models(filters)
    preferred = [item for item in items if item["status"] in {"recommended", "compatible"}]
    candidates = preferred if preferred else items
    risk_rank = {"safe": 0, "caveated": 1, "experimental": 2}
    status_rank = {"recommended": 0, "compatible": 1, "experimental": 2, "deprecated": 3}
    candidates.sort(
        key=lambda item: (
            not bool(item.get("recommended_first_run")),
            risk_rank.get(str(item.get("risk_level", "")), 9),
            status_rank.get(str(item.get("status", "")), 9),
            float(item.get("estimated_memory_gb") or 999),
            _CATALOG_ORDER.get(str(item.get("id")), 9999),
        )
    )
    return candidates[:8]


def catalog_facets(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    def collect(key: str) -> list[str]:
        values: set[str] = set()
        for item in items:
            raw = item.get(key)
            if isinstance(raw, list):
                values.update(str(v) for v in raw if v)
            elif raw:
                values.add(str(raw))
        return sorted(values, key=str.lower)

    return {
        "providers": collect("provider"),
        "families": collect("family"),
        "modalities": collect("modalities"),
        "trainer_support": collect("trainer_support"),
        "backend_support": collect("backend_support"),
        "memory_tiers": collect("memory_tier"),
        "statuses": collect("status"),
        "risk_levels": collect("risk_level"),
    }
