"""Round-trip verification for exported / converted model artifacts (Track I4).

After ``halo-forge convert`` produces an MLX or GGUF artifact, an
independent question is "does the exported file actually produce the
same answers as the source?" This module answers it: run a fixed prompt
set through both source and exported, score similarity, and flag drift
that exceeds a threshold.

Why this matters: GGUF / MLX conversion silently breaks more often
than anyone admits. The model file looks valid, the loader returns,
the first token looks plausible, and your finetune is silently
serving garbage. Running a prompt-pair sanity check after every
export catches this in seconds.

Three drift metrics ship in v1:
  - **exact_match_rate** — fraction of prompts whose source/exported
    completions match character-for-character. The cleanest signal
    when both backends use deterministic decoding.
  - **avg_char_overlap** — character-level Jaccard similarity, averaged
    across the prompt set. Robust to the small numerical-precision
    drifts that legitimate quantization introduces.
  - **avg_first_token_match** — fraction of prompts whose first
    generated token matches. Catches the most common "exported
    model immediately diverges" failure mode.

The verifier is pluggable: pass any callable
``(prompt) -> str`` for source / exported. The ``halo-forge convert
--verify`` integration wires the right callable based on the source
and target formats.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)


GenerateFn = Callable[[str], str]


# Default fixed prompt set used when the caller doesn't supply one. Kept
# small + deterministic so a verification pass is sub-minute on any
# backend; users can pass their own list for task-specific coverage.
DEFAULT_VERIFICATION_PROMPTS: tuple[str, ...] = (
    "Write a single-line Python function that returns the square of its input.",
    "What is 17 * 23?",
    "List three primary colors.",
    "Translate 'hello' to French.",
    "Complete: The quick brown fox jumps over the lazy",
    "Define monotonic in one short sentence.",
    "What is the capital of Japan?",
    "Write a haiku about autumn.",
)


@dataclass
class PromptComparison:
    prompt: str
    source_completion: str
    exported_completion: str
    char_overlap: float
    first_token_match: bool
    exact: bool


@dataclass
class VerificationReport:
    """Aggregate stats over a verification run."""

    n_prompts: int
    exact_match_rate: float
    avg_char_overlap: float
    avg_first_token_match: float
    passed: bool  # all metrics above their thresholds
    failures: List[PromptComparison] = field(default_factory=list)
    samples: List[PromptComparison] = field(default_factory=list)
    duration_seconds: float = 0.0
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Trim sample lists for the JSON wire — full lists stay on the
        # in-memory dataclass for callers that want them.
        d["samples"] = [asdict(s) for s in self.samples[:5]]
        d["failures"] = [asdict(s) for s in self.failures[:10]]
        return d


# ---------- similarity helpers ---------------------------------------------


def _char_overlap(a: str, b: str) -> float:
    """Character-level Jaccard. 1.0 when identical; 0.0 when disjoint.

    Cheaper than computing token-level edit distance and good enough
    for "did this completion drift" — small numerical-quantization
    differences usually preserve >95% character overlap, while
    silently-broken exports cluster well below 0.5.
    """
    if not a and not b:
        return 1.0
    set_a = set(a)
    set_b = set(b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _first_token(completion: str) -> str:
    """Crude tokenizer-agnostic 'first token': everything up to the first
    whitespace. Mirrors what most chat-format completions look like at
    position 0 closely enough for a drift signal."""
    text = completion.lstrip()
    if not text:
        return ""
    for i, ch in enumerate(text):
        if ch.isspace():
            return text[:i]
    return text


# ---------- core comparison -------------------------------------------------


def compare_generation(
    *,
    source_generate: GenerateFn,
    exported_generate: GenerateFn,
    prompts: Optional[Sequence[str]] = None,
    char_overlap_threshold: float = 0.7,
    exact_match_threshold: float = 0.0,
    first_token_threshold: float = 0.5,
) -> VerificationReport:
    """Run both generators on the same prompt set; score drift.

    Args:
        source_generate / exported_generate: ``(prompt) -> completion``
            callables for the source model and the exported artifact.
        prompts: Custom prompt set. Defaults to
            `DEFAULT_VERIFICATION_PROMPTS`.
        char_overlap_threshold: Average char-overlap below which the
            verification fails. 0.7 is the published rule-of-thumb for
            "different but recognizable"; lower means accepting more
            quantization drift.
        exact_match_threshold: Exact-match rate below which the
            verification fails. 0.0 (default) treats exact match as
            informational only — most legit quantization drops it
            slightly without semantic damage.
        first_token_threshold: First-token-match rate below which the
            verification fails. The single best signal for "the
            exported model immediately disagrees."

    Returns:
        `VerificationReport` with aggregate stats + per-prompt detail.
    """
    import time

    # `prompts is None` → use the defaults; `prompts == []` → user
    # explicitly cleared the list, which is a category error worth
    # surfacing instead of silently re-injecting the defaults.
    if prompts is None:
        prompt_list = list(DEFAULT_VERIFICATION_PROMPTS)
    else:
        prompt_list = list(prompts)
    if not prompt_list:
        raise ValueError("compare_generation needs at least one prompt")

    t0 = time.time()
    samples: List[PromptComparison] = []
    n_exact = 0
    sum_overlap = 0.0
    n_first_token_match = 0

    for prompt in prompt_list:
        try:
            src = source_generate(prompt)
        except Exception as exc:
            logger.warning("source_generate failed on prompt %r: %s", prompt[:40], exc)
            src = ""
        try:
            exp = exported_generate(prompt)
        except Exception as exc:
            logger.warning("exported_generate failed on prompt %r: %s", prompt[:40], exc)
            exp = ""

        overlap = _char_overlap(src, exp)
        first_match = _first_token(src) == _first_token(exp) and bool(_first_token(src))
        exact = src == exp and bool(src)
        if exact:
            n_exact += 1
        if first_match:
            n_first_token_match += 1
        sum_overlap += overlap

        samples.append(
            PromptComparison(
                prompt=prompt,
                source_completion=src,
                exported_completion=exp,
                char_overlap=overlap,
                first_token_match=first_match,
                exact=exact,
            )
        )

    n = len(prompt_list)
    exact_rate = n_exact / n
    avg_overlap = sum_overlap / n
    first_token_rate = n_first_token_match / n

    failures = [
        s for s in samples
        if s.char_overlap < char_overlap_threshold
        or (exact_match_threshold > 0 and not s.exact)
    ]

    passed = (
        avg_overlap >= char_overlap_threshold
        and exact_rate >= exact_match_threshold
        and first_token_rate >= first_token_threshold
    )

    return VerificationReport(
        n_prompts=n,
        exact_match_rate=exact_rate,
        avg_char_overlap=avg_overlap,
        avg_first_token_match=first_token_rate,
        passed=passed,
        failures=failures,
        samples=samples,
        duration_seconds=time.time() - t0,
    )


# ---------- ergonomic wrapper ----------------------------------------------


def verify_export(
    *,
    source_model: str,
    exported_path: str,
    target_format: str,
    prompts: Optional[Sequence[str]] = None,
    max_tokens: int = 64,
    char_overlap_threshold: float = 0.7,
    first_token_threshold: float = 0.5,
) -> VerificationReport:
    """High-level: load source + exported via halo-forge's serving
    adapters, run the prompt set, return the report.

    ``target_format`` ∈ {"mlx", "gguf", "hf"}; routes the exported-side
    loader. Source model loads through the active backend.

    GGUF is special: halo-forge's serving adapter doesn't yet load
    GGUF directly (that's roadmap), so v1 surfaces a typed error
    pointing the user at llama.cpp / Ollama for the exported side.
    """
    from halo_forge.serving.adapter import build_serving_adapter

    target_format = target_format.strip().lower()
    if target_format == "gguf":
        raise NotImplementedError(
            "GGUF round-trip verification is not implemented: halo-forge's serving "
            "adapters cannot load a .gguf file, so there is no exported-side "
            "generator to compare against. Nothing you install fixes this today. "
            "To verify a GGUF export now:\n"
            "  1. pip install llama-cpp-python   (or run `ollama create` on the file)\n"
            "  2. wrap the loaded model in a `(prompt) -> str` callable\n"
            "  3. call halo_forge.inference.verify_export.compare_generation("
            "source_generate=..., exported_generate=...)\n"
            "For an in-CLI check, convert with --format mlx or --format hf and "
            "verify that artifact instead."
        )

    src_adapter = build_serving_adapter(source_model)

    if target_format == "mlx":
        exp_adapter = build_serving_adapter(exported_path, backend_name="mlx")
    else:  # hf
        exp_adapter = build_serving_adapter(exported_path)

    def src_gen(p: str) -> str:
        return src_adapter.generate(p, max_tokens=max_tokens, temperature=0.0)

    def exp_gen(p: str) -> str:
        return exp_adapter.generate(p, max_tokens=max_tokens, temperature=0.0)

    return compare_generation(
        source_generate=src_gen,
        exported_generate=exp_gen,
        prompts=prompts,
        char_overlap_threshold=char_overlap_threshold,
        first_token_threshold=first_token_threshold,
    )


__all__ = [
    "DEFAULT_VERIFICATION_PROMPTS",
    "PromptComparison",
    "VerificationReport",
    "compare_generation",
    "verify_export",
]
