"""Per-row quality scoring (Track D3).

After dedup (D2) the highest-leverage dataset cleanup is filtering on
quality — keeping rows that look like actual instruction / preference
data and dropping the cosmetic noise. DEITA et al. show that filtering
to the top-K% by quality consistently outperforms training on the
full dataset *at any size*.

This module ships a lightweight, dependency-free heuristic scorer (so
it runs anywhere, including in the dedup → score → filter pipeline on
a CPU-only host) and a pluggable `score_with_judge` path for users
who want a real LLM-as-judge quality signal (V2 territory).

Heuristic scorer composes five signals into a single 0-1 score:

  - **length_score**         — penalizes too-short and runaway-long
                                completions; sweet spot 50-1500 chars.
  - **whitespace_score**     — fraction non-whitespace; catches blank
                                or near-blank rows.
  - **alpha_ratio_score**    — fraction alphabetic; catches rows that
                                are mostly punctuation / repeating chars.
  - **repetition_score**     — penalizes high n-gram repetition;
                                catches stuck-loop generations.
  - **format_score**         — for chat-shaped rows, expects keys to
                                exist and content to be non-empty.

Weights default to uniform; override per task. The result carries
both the composite score and the per-component breakdown so users
can debug "why was this row filtered" without re-running.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ---------- result shape ---------------------------------------------------


@dataclass
class QualityScore:
    """Per-row scoring result."""

    score: float  # composite, in [0, 1]
    components: Dict[str, float] = field(default_factory=dict)
    rejected: bool = False
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreResult:
    """Aggregate stats over a batch."""

    n_input: int
    n_kept: int
    n_rejected: int
    score_threshold: float
    duration_seconds: float = 0.0
    kept_indices: List[int] = field(default_factory=list)
    rejected_indices: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- heuristic components -------------------------------------------


_WS_RE = re.compile(r"\s+")
_ALPHA_RE = re.compile(r"[A-Za-z]")
_NON_WS_RE = re.compile(r"\S")


def length_score(text: str, *, target_min: int = 50, target_max: int = 1500) -> float:
    """Triangle-shaped score: 1.0 inside [target_min, target_max], decays
    smoothly outside the band. Empty text scores 0."""
    n = len(text or "")
    if n == 0:
        return 0.0
    if n < target_min:
        return n / target_min
    if n > target_max:
        # Decay slowly past the cap; runaway-long completions trend toward 0.
        excess = n - target_max
        return max(0.0, 1.0 - excess / (target_max * 4))
    return 1.0


def whitespace_score(text: str) -> float:
    """Fraction of non-whitespace characters. Catches near-blank rows."""
    if not text:
        return 0.0
    non_ws = len(_NON_WS_RE.findall(text))
    return non_ws / len(text)


def alpha_ratio_score(text: str) -> float:
    """Fraction of alphabetic characters. Catches rows that are mostly
    punctuation, code-like noise, or repeated characters."""
    if not text:
        return 0.0
    alpha = len(_ALPHA_RE.findall(text))
    return alpha / len(text)


def repetition_score(text: str, *, n: int = 3) -> float:
    """1.0 = no repetition; 0.0 = the same n-gram fills the entire row.

    Computed as `unique_ngrams / total_ngrams` over word n-grams. The
    classic stuck-loop generation ("the the the the …") drops below
    0.2 immediately.
    """
    tokens = (text or "").split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def format_score(record: Any) -> float:
    """For dict-shaped chat records, check that the canonical keys are
    present and non-empty. For plain strings, returns 1.0 (the other
    signals already cover content quality)."""
    if isinstance(record, str):
        return 1.0
    if not isinstance(record, dict):
        return 0.5  # unknown shape — neither penalize nor reward strongly
    # Common shapes: {prompt, chosen, rejected} (preference),
    # {prompt, completion} (SFT), {messages: [...]} (chat).
    if "messages" in record:
        msgs = record.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return 0.0
        valid = sum(
            1
            for m in msgs
            if isinstance(m, dict) and isinstance(m.get("content"), str) and m["content"].strip()
        )
        return valid / max(1, len(msgs))
    expected_present = sum(
        1
        for k in ("prompt", "completion", "chosen", "rejected", "text")
        if isinstance(record.get(k), str) and record[k].strip()
    )
    return min(1.0, expected_present / 2)  # any 2 of those = full credit


# ---------- composite scorer -----------------------------------------------


_DEFAULT_WEIGHTS: Dict[str, float] = {
    "length": 0.25,
    "whitespace": 0.15,
    "alpha_ratio": 0.20,
    "repetition": 0.25,
    "format": 0.15,
}


def _extract_text(record: Any) -> str:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        for k in ("text", "completion", "chosen", "prompt"):
            v = record.get(k)
            if isinstance(v, str) and v:
                return v
        # Chat-shape fallback.
        msgs = record.get("messages")
        if isinstance(msgs, list):
            return " ".join(m.get("content", "") if isinstance(m, dict) else str(m) for m in msgs)
    return str(record)


def heuristic_score(
    record: Any,
    *,
    weights: Optional[Dict[str, float]] = None,
    target_min_length: int = 50,
    target_max_length: int = 1500,
    repetition_n: int = 3,
) -> QualityScore:
    """Score a single record using the heuristic stack."""
    text = _extract_text(record)
    weights = weights or _DEFAULT_WEIGHTS

    components = {
        "length": length_score(text, target_min=target_min_length, target_max=target_max_length),
        "whitespace": whitespace_score(text),
        "alpha_ratio": alpha_ratio_score(text),
        "repetition": repetition_score(text, n=repetition_n),
        "format": format_score(record),
    }
    total_w = sum(weights.get(k, 0) for k in components) or 1.0
    composite = sum(components[k] * weights.get(k, 0) for k in components) / total_w

    # Multiplicative penalty: a row that fully fails on any one axis
    # shouldn't survive on the strength of its other axes alone.
    # Without this, a blank string scores 0.4 (repetition + format
    # default to 1.0 with no content to repeat against), and a
    # punctuation-only row scores 0.8 (alpha_ratio is the only failing
    # signal). The penalty lets the weakest component dominate when
    # it's catastrophic.
    weakest = min(components.values())
    if weakest < 0.1:
        composite *= 0.3

    return QualityScore(score=composite, components=components)


# ---------- pluggable LLM-as-judge scorer ----------------------------------


def score_with_judge(
    record: Any,
    *,
    judge: Callable[[str], float],
) -> QualityScore:
    """Hand the record's text to a user-supplied judge callable.

    The callable returns a float in [0, 1]; values outside that range
    are clipped. Useful when the user wants a real LLM-as-judge signal
    plugged through the V2 verifier path or a custom regression model.
    """
    text = _extract_text(record)
    try:
        raw = float(judge(text))
    except Exception as exc:
        logger.warning("Judge raised on record: %s", exc)
        return QualityScore(
            score=0.0,
            components={"judge": 0.0},
            rejected=True,
            reason="judge_error",
        )
    clipped = max(0.0, min(1.0, raw))
    return QualityScore(score=clipped, components={"judge": clipped})


# ---------- batch + filter -------------------------------------------------


def score_records(
    records: Sequence[Any],
    *,
    threshold: float = 0.5,
    scorer: Callable[[Any], QualityScore] = heuristic_score,
) -> ScoreResult:
    """Score every record; tag each as kept (score >= threshold) or
    rejected. Returns aggregate stats + per-row scores so callers can
    materialize either the surviving subset or the full annotation."""
    import time

    t0 = time.time()
    kept: List[int] = []
    rejected: List[int] = []
    scores: List[float] = []
    reasons: Dict[str, int] = {}

    for idx, record in enumerate(records):
        result = scorer(record)
        scores.append(result.score)
        if result.score >= threshold:
            kept.append(idx)
        else:
            rejected.append(idx)
            # Bucket by the lowest-scoring component so the reasons
            # dict tells the user *why* rejection happened in aggregate.
            if result.components:
                weakest = min(result.components, key=result.components.get)
                reasons[weakest] = reasons.get(weakest, 0) + 1

    return ScoreResult(
        n_input=len(records),
        n_kept=len(kept),
        n_rejected=len(rejected),
        score_threshold=threshold,
        duration_seconds=time.time() - t0,
        kept_indices=kept,
        rejected_indices=rejected,
        scores=scores,
        reasons=reasons,
    )


def score_file(
    *,
    input_path,
    output_path,
    threshold: float = 0.5,
    keep_top_k_pct: Optional[float] = None,
    scorer: Callable[[Any], QualityScore] = heuristic_score,
) -> ScoreResult:
    """End-to-end: load JSONL → score → write surviving rows."""
    from pathlib import Path

    from halo_forge.data.dedup import load_jsonl, write_jsonl

    records = load_jsonl(Path(input_path))
    result = score_records(records, threshold=threshold, scorer=scorer)

    if keep_top_k_pct is not None:
        if not 0.0 < keep_top_k_pct <= 1.0:
            raise ValueError(f"keep_top_k_pct must be in (0, 1], got {keep_top_k_pct}")
        # Override threshold-based filter with top-K-pct selection by score.
        cutoff = max(1, int(len(records) * keep_top_k_pct))
        ranked = sorted(
            range(len(records)),
            key=lambda i: result.scores[i],
            reverse=True,
        )
        result.kept_indices = sorted(ranked[:cutoff])
        result.rejected_indices = sorted(ranked[cutoff:])
        result.n_kept = len(result.kept_indices)
        result.n_rejected = len(result.rejected_indices)

    survivors = [records[i] for i in result.kept_indices]
    write_jsonl(Path(output_path), survivors)
    return result


__all__ = [
    "QualityScore",
    "ScoreResult",
    "heuristic_score",
    "score_with_judge",
    "score_records",
    "score_file",
    "length_score",
    "whitespace_score",
    "alpha_ratio_score",
    "repetition_score",
    "format_score",
]
