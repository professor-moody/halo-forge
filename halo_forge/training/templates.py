"""Training templates — intent-first starting points for `/train`.

The /train page is a knob-by-knob configurator: choose a mode, then a
model, then a dataset, then hyperparams. That's the right surface for
people who already know what they want, but it's a poor introduction
to "what can I actually train this thing to do?"

Templates flip the question. Each template is a single, opinionated
recipe tied to a goal a person walks in with — "teach a model to
write Python that compiles", "fine-tune for math reasoning",
"adapt a vision model for invoice extraction". Picking a template
fills in the modality, model hint, dataset hint, verifier, and
hyperparams in one step; the user can still tweak any field
afterward.

The set is deliberately small — six categories × 1-2 templates
each — because a 60-template gallery is overwhelming and most users
will end up customizing anyway. Adding new templates is cheap (one
dict in the registry below); deleting them is freer than editing
the configurator UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# Categories the UI groups templates under. Order matters — this is
# the rendering order on the gallery page.
CATEGORIES = (
    ("code",        "Code",        "Programming languages, code-completion, code-review."),
    ("reasoning",   "Reasoning",   "Math, logic, multi-step problem solving."),
    ("vision",      "Vision",      "Image-Q&A, captioning, document understanding."),
    ("audio",       "Audio",       "Speech-to-text, audio-Q&A, transcription."),
    ("preference",  "Preferences", "Preference learning, RLAIF, reward modeling."),
    ("agentic",     "Agentic",     "Tool use, function calling, multi-turn agents."),
)


@dataclass(frozen=True)
class TrainingTemplate:
    id: str
    name: str
    category: str          # one of CATEGORIES keys
    intent: str            # one-line "what this teaches the model to do"
    modality: str          # canonical trainer key: sft / raft / dpo / grpo / rm / vlm / audio / reasoning / agentic
    model_hint: str        # suggested base model id
    dataset_hint: str      # suggested dataset (HF id, short name, or @custom)
    verifier: Optional[str] = None  # RAFT/GRPO only
    hyperparams: dict[str, Any] = field(default_factory=dict)
    expected_runtime: str = ""  # human string, e.g. "2-3h on Strix Halo"
    learn_more: Optional[str] = None  # docs path
    cli_hint: Optional[str] = None    # custom CLI invocation override


# Curated registry. Keep it tight — six categories, 1-2 templates each.
# Hyperparam overrides should be the *delta* from the trainer default,
# not a full config dump.
TEMPLATES: tuple[TrainingTemplate, ...] = (
    # ---- Code -----------------------------------------------------------
    TrainingTemplate(
        id="code-python-sft",
        name="Python coding (SFT)",
        category="code",
        intent="Teach a small model to write idiomatic, compilable Python from natural-language prompts.",
        modality="sft",
        model_hint="Qwen/Qwen2.5-Coder-0.5B",
        dataset_hint="codealpaca",
        hyperparams={"epochs": 3, "batch_size": 2, "learning_rate": 2e-4, "lora_rank": 16},
        expected_runtime="20-40 min on Strix Halo / Apple Silicon",
        learn_more="docs/TRAINERS.md#sft",
    ),
    TrainingTemplate(
        id="code-multi-language-raft",
        name="Multi-language coding (RAFT)",
        category="code",
        intent="Reject-sample with a real compiler — keep only generations that build, then SFT on the survivors.",
        modality="raft",
        model_hint="Qwen/Qwen2.5-Coder-3B",
        dataset_hint="humaneval",
        verifier="gcc",
        hyperparams={"cycles": 3, "samples_per_prompt": 8, "batch_size": 2},
        expected_runtime="2-4h on Strix Halo",
        learn_more="docs/TRAINERS.md#raft",
    ),

    # ---- Reasoning ------------------------------------------------------
    TrainingTemplate(
        id="reasoning-math-sft",
        name="Math reasoning (SFT)",
        category="reasoning",
        intent="Fine-tune for chain-of-thought math: GSM8K-style word problems with explicit reasoning traces.",
        modality="reasoning",
        model_hint="Qwen/Qwen2.5-Math-1.5B",
        dataset_hint="gsm8k",
        hyperparams={"epochs": 2, "batch_size": 4, "learning_rate": 1e-4},
        expected_runtime="1-2h on Strix Halo",
        learn_more="docs/TRAINERS.md#reasoning",
    ),
    TrainingTemplate(
        id="reasoning-grpo",
        name="Reasoning + GRPO",
        category="reasoning",
        intent="Group Relative Policy Optimization on verifiable reasoning tasks — reward = symbolic answer match.",
        modality="grpo",
        model_hint="Qwen/Qwen2.5-Math-1.5B",
        dataset_hint="gsm8k",
        verifier="json_schema",
        hyperparams={"group_size": 8, "kl_coef": 0.05},
        expected_runtime="3-6h on Strix Halo",
        learn_more="docs/TRAINERS.md#grpo",
    ),

    # ---- Vision ---------------------------------------------------------
    TrainingTemplate(
        id="vision-vqa",
        name="Visual Q&A (VLM)",
        category="vision",
        intent="Adapt a small vision-language model for image-based question answering — captions, charts, screenshots.",
        modality="vlm",
        model_hint="Qwen/Qwen2.5-VL-3B-Instruct",
        dataset_hint="vqa-rad",
        hyperparams={"epochs": 2, "batch_size": 1, "learning_rate": 5e-5},
        expected_runtime="2-4h on Strix Halo",
        learn_more="docs/HARDWARE_NOTES.md#vlm",
    ),
    TrainingTemplate(
        id="vision-document-extraction",
        name="Document extraction (VLM)",
        category="vision",
        intent="Train a VLM to extract structured fields from invoices, receipts, forms — JSON-schema-validated outputs.",
        modality="vlm",
        model_hint="Qwen/Qwen2.5-VL-3B-Instruct",
        dataset_hint="@custom",
        verifier="json_schema",
        hyperparams={"epochs": 3, "batch_size": 1, "learning_rate": 5e-5},
        expected_runtime="2-4h on Strix Halo",
        learn_more="docs/HARDWARE_NOTES.md#vlm",
    ),

    # ---- Audio ----------------------------------------------------------
    TrainingTemplate(
        id="audio-asr-finetune",
        name="Speech-to-text fine-tune",
        category="audio",
        intent="Fine-tune Whisper-class ASR on domain-specific audio — call center, podcasts, dialect.",
        modality="audio",
        model_hint="openai/whisper-small",
        dataset_hint="librispeech-clean",
        hyperparams={"epochs": 2, "batch_size": 8, "learning_rate": 1e-5},
        expected_runtime="1-3h on Strix Halo",
        learn_more="docs/HARDWARE_NOTES.md#audio",
    ),

    # ---- Preference learning -------------------------------------------
    TrainingTemplate(
        id="pref-dpo-chat",
        name="Chat refinement (DPO)",
        category="preference",
        intent="Direct Preference Optimization on a chosen/rejected pair set — sharpen tone, refusal behavior, formatting.",
        modality="dpo",
        model_hint="Qwen/Qwen2.5-3B-Instruct",
        dataset_hint="ultrafeedback-binarized",
        hyperparams={"beta": 0.1, "epochs": 1, "batch_size": 2, "learning_rate": 5e-7},
        expected_runtime="1-2h on Strix Halo",
        learn_more="docs/TRAINERS.md#dpo",
    ),
    TrainingTemplate(
        id="pref-orpo-chat",
        name="Chat refinement (ORPO)",
        category="preference",
        intent="Odds-Ratio Preference Optimization — same chosen/rejected pairs as DPO, half the wall-time, no reference model.",
        modality="orpo",
        model_hint="Qwen/Qwen2.5-3B-Instruct",
        dataset_hint="ultrafeedback",
        hyperparams={"beta": 0.1, "epochs": 1, "batch_size": 1, "learning_rate": 8e-6},
        expected_runtime="30-60 min on Strix Halo",
        learn_more="docs/TRAINERS.md#orpo",
    ),
    TrainingTemplate(
        id="pref-rm",
        name="Reward model (Bradley-Terry)",
        category="preference",
        intent="Train a small classifier on chosen/rejected pairs to score future generations — building block for RLHF.",
        modality="rm",
        model_hint="Qwen/Qwen2.5-0.5B",
        dataset_hint="ultrafeedback-binarized",
        hyperparams={"epochs": 1, "batch_size": 4, "learning_rate": 5e-6},
        expected_runtime="30-60 min on Strix Halo",
        learn_more="docs/TRAINERS.md#rm",
    ),

    # ---- Agentic -------------------------------------------------------
    TrainingTemplate(
        id="agentic-tool-use",
        name="Tool-use / function calling",
        category="agentic",
        intent="Teach a model to call tools and follow multi-turn agent traces — JSON-formatted function calls.",
        modality="agentic",
        model_hint="Qwen/Qwen2.5-3B-Instruct",
        dataset_hint="@custom",
        verifier="json_schema",
        hyperparams={"epochs": 2, "batch_size": 1, "learning_rate": 1e-4},
        expected_runtime="1-3h on Strix Halo",
        learn_more="docs/TRAINERS.md#agentic",
    ),
)


def list_templates() -> list[dict[str, Any]]:
    """Return all templates as plain dicts, ordered by category then by id."""
    by_cat = {key: idx for idx, (key, _label, _desc) in enumerate(CATEGORIES)}
    items = sorted(
        TEMPLATES,
        key=lambda t: (by_cat.get(t.category, 999), t.id),
    )
    return [asdict(t) for t in items]


def list_categories() -> list[dict[str, str]]:
    """Return the canonical category list with display labels."""
    return [{"id": k, "label": label, "description": desc} for k, label, desc in CATEGORIES]


def get_template(template_id: str) -> Optional[dict[str, Any]]:
    """Return a single template by id, or None if not found."""
    for t in TEMPLATES:
        if t.id == template_id:
            return asdict(t)
    return None


def cli_invocation(template_id: str) -> Optional[str]:
    """Render the `halo-forge` CLI invocation for a template.

    Useful for the gallery's "Show CLI" affordance on modalities the
    /train form doesn't yet cover. Returns None if the template id
    isn't recognized.
    """
    t = next((tt for tt in TEMPLATES if tt.id == template_id), None)
    if t is None:
        return None
    if t.cli_hint:
        return t.cli_hint
    parts: list[str] = ["halo-forge", t.modality, "train"]
    parts.extend(["--model", t.model_hint])
    if t.dataset_hint and t.dataset_hint != "@custom":
        parts.extend(["--dataset", t.dataset_hint])
    if t.verifier and t.modality in ("raft", "grpo"):
        parts.extend(["--verifier", t.verifier])
    for key, value in sorted(t.hyperparams.items()):
        flag = "--" + key.replace("_", "-")
        parts.extend([flag, str(value)])
    parts.extend(["--output", f"models/{t.id}"])
    return " ".join(parts)


__all__ = [
    "CATEGORIES",
    "TrainingTemplate",
    "TEMPLATES",
    "list_templates",
    "list_categories",
    "get_template",
    "cli_invocation",
]
