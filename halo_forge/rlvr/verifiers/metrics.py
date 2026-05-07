"""Reference-metric verifiers (Track V4).

Wraps the standard MT / summarization scoring metrics behind the
halo-forge `Verifier` interface so a trainer can score candidates
against a reference string without writing the bridge per-task.

Three verifiers ship:

  - **bleu**  — BLEU score via sacrebleu (the standard SacreBLEU
                tokenization; comparable across papers).
  - **rouge** — ROUGE-L F-measure via rouge_score.
  - **chrf**  — character n-gram F-score via sacrebleu.

All three normalize their score into reward ∈ [0.0, 1.0] so the RAFT
/ GRPO trainers can mix them with execution-based rewards in a chained
verifier without scaling math.

Each metric's underlying package is *lazy-imported* so this module
loads on installs without it; the verifier's first call surfaces a
clean ImportError pointing at the right `pip install` command.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult
from halo_forge.rlvr.verifiers.registry import register_verifier

logger = logging.getLogger(__name__)


def _ensure_iterable_references(refs) -> List[str]:
    """Normalize the references arg: accept a single string or a list.

    Multiple references are useful for translation tasks where several
    valid translations exist. BLEU and chrF both accept this shape;
    ROUGE typically gets a single reference.
    """
    if isinstance(refs, str):
        return [refs]
    return list(refs or [])


@register_verifier("bleu")
class BLEUVerifier(Verifier):
    """Score the candidate against ``references`` using SacreBLEU.

    BLEU is in [0, 100]; we divide by 100 so the verifier's reward
    matches the rest of halo-forge's [0, 1] convention.

    Args:
        references: Single reference string or list of references.
            Required at construction time so RAFT can score every
            candidate against the same target.
        success_threshold: BLEU/100 score above which the verifier
            reports success=True. 0.3 is generous; 0.5 is strict.
    """

    def __init__(
        self,
        *,
        references=None,
        success_threshold: float = 0.3,
        max_workers: int = 8,
    ):
        super().__init__(max_workers=max_workers)
        self.references = _ensure_iterable_references(references)
        if not self.references:
            raise ValueError("BLEUVerifier requires at least one reference")
        self.success_threshold = success_threshold
        self._sacrebleu = None

    def _get_sacrebleu(self):
        if self._sacrebleu is not None:
            return self._sacrebleu
        try:
            import sacrebleu
        except ImportError as exc:
            raise ImportError(
                "BLEUVerifier requires `sacrebleu`. Install with "
                "`pip install sacrebleu`."
            ) from exc
        self._sacrebleu = sacrebleu
        return sacrebleu

    def verify(self, code: str) -> VerifyResult:
        if not code:
            return VerifyResult(
                success=False, reward=0.0,
                details="Empty candidate", error="empty_response",
            )
        sb = self._get_sacrebleu()
        # corpus_bleu([candidate], [[ref1], [ref2], ...]) — sacrebleu
        # expects references in column-major shape (one list per ref slot).
        refs_columned = [[r] for r in self.references]
        bleu = sb.corpus_bleu([code], refs_columned).score / 100.0
        bleu = max(0.0, min(1.0, bleu))
        return VerifyResult(
            success=bleu >= self.success_threshold,
            reward=bleu,
            details=f"BLEU = {bleu:.4f} (threshold {self.success_threshold:.2f})",
        )


@register_verifier("rouge")
class ROUGEVerifier(Verifier):
    """Score against a reference via ROUGE-L F-measure.

    ROUGE is in [0, 1] natively, so the score is the reward directly.
    Uses google-research's `rouge_score` package; lazy-imported.

    Args:
        reference: Single reference string. ROUGE typically operates
            against one reference; multi-reference support is roadmap.
        rouge_type: "rouge1", "rouge2", or "rougeL" (default).
        success_threshold: F-measure above which success=True.
    """

    def __init__(
        self,
        *,
        reference: Optional[str] = None,
        rouge_type: str = "rougeL",
        success_threshold: float = 0.3,
        max_workers: int = 8,
    ):
        super().__init__(max_workers=max_workers)
        if not reference:
            raise ValueError("ROUGEVerifier requires a reference string")
        if rouge_type not in {"rouge1", "rouge2", "rougeL"}:
            raise ValueError(
                f"rouge_type must be one of rouge1/rouge2/rougeL; got {rouge_type!r}"
            )
        self.reference = reference
        self.rouge_type = rouge_type
        self.success_threshold = success_threshold
        self._scorer = None

    def _get_scorer(self):
        if self._scorer is not None:
            return self._scorer
        try:
            from rouge_score import rouge_scorer
        except ImportError as exc:
            raise ImportError(
                "ROUGEVerifier requires `rouge_score`. Install with "
                "`pip install rouge_score`."
            ) from exc
        self._scorer = rouge_scorer.RougeScorer([self.rouge_type], use_stemmer=True)
        return self._scorer

    def verify(self, code: str) -> VerifyResult:
        if not code:
            return VerifyResult(
                success=False, reward=0.0,
                details="Empty candidate", error="empty_response",
            )
        scorer = self._get_scorer()
        scores = scorer.score(self.reference, code)
        f_measure = float(scores[self.rouge_type].fmeasure)
        f_measure = max(0.0, min(1.0, f_measure))
        return VerifyResult(
            success=f_measure >= self.success_threshold,
            reward=f_measure,
            details=f"{self.rouge_type} F = {f_measure:.4f}",
        )


@register_verifier("chrf")
class ChrFVerifier(Verifier):
    """Character n-gram F-score (chrF) via SacreBLEU.

    chrF is more robust than BLEU on morphologically rich languages and
    is increasingly the recommended baseline for MT evaluation. Score
    is in [0, 100]; divided by 100 here.

    Args:
        references: Single reference or list (multi-ref supported).
        word_order: chrF++ word-order n-gram size; 0 disables (chrF),
            2 enables chrF++ (sacrebleu's recommended default).
        success_threshold: chrF/100 score above which success=True.
    """

    def __init__(
        self,
        *,
        references=None,
        word_order: int = 2,
        success_threshold: float = 0.3,
        max_workers: int = 8,
    ):
        super().__init__(max_workers=max_workers)
        self.references = _ensure_iterable_references(references)
        if not self.references:
            raise ValueError("ChrFVerifier requires at least one reference")
        self.word_order = word_order
        self.success_threshold = success_threshold
        self._sacrebleu = None

    def _get_sacrebleu(self):
        if self._sacrebleu is not None:
            return self._sacrebleu
        try:
            import sacrebleu
        except ImportError as exc:
            raise ImportError(
                "ChrFVerifier requires `sacrebleu`. Install with "
                "`pip install sacrebleu`."
            ) from exc
        self._sacrebleu = sacrebleu
        return sacrebleu

    def verify(self, code: str) -> VerifyResult:
        if not code:
            return VerifyResult(
                success=False, reward=0.0,
                details="Empty candidate", error="empty_response",
            )
        sb = self._get_sacrebleu()
        refs_columned = [[r] for r in self.references]
        chrf = sb.corpus_chrf(
            [code], refs_columned, word_order=self.word_order
        ).score / 100.0
        chrf = max(0.0, min(1.0, chrf))
        return VerifyResult(
            success=chrf >= self.success_threshold,
            reward=chrf,
            details=f"chrF{'++' if self.word_order > 0 else ''} = {chrf:.4f}",
        )


__all__ = ["BLEUVerifier", "ROUGEVerifier", "ChrFVerifier"]
