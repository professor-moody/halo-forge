"""LLM-as-judge verifier (Track V2).

Calls a judge model with a rubric prompt and parses a numeric score
from the response. Lets users grade outputs that don't have a
programmatic ground truth (creative writing, summarization, dialog
quality) without writing custom code per task.

Three knobs the operator owns:

  * **rubric** — a multi-line description of what "good" looks like.
    Embedded in the judge prompt under a ``Rubric`` heading.
  * **scoring_scale** — typically ``5`` for a 1-5 Likert scale, or
    ``10``. The judge prompt instructs the model to return a single
    integer in ``[1, scale]``; the verifier maps that to a reward in
    ``[0.0, 1.0]``.
  * **judge_callable** — any function ``(prompt: str) -> str``. The
    default uses an OpenAI-compatible HTTP client and reads the model
    name + base URL from constructor args (or env vars). Pass your own
    callable to wire vLLM / llama.cpp-server / a local mlx-served
    model — anything that can answer a chat prompt with text.

Why pluggable judge: serving infra is moving fast (vLLM, SGLang,
llama.cpp, MLX) and the right answer for this user is "use the judge
*they* are running". The default is an OpenAI-compatible call so an
operator who points it at any local server gets a working judge for
free.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable, Optional

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult
from halo_forge.rlvr.verifiers.registry import register_verifier

logger = logging.getLogger(__name__)


JudgeCallable = Callable[[str], str]


_DEFAULT_RUBRIC = (
    "Score the candidate response on overall quality, taking into "
    "account correctness, helpfulness, and clarity."
)


_PROMPT_TEMPLATE = """You are an expert evaluator. Grade the candidate response on a {scale_min}-{scale_max} integer scale where {scale_max} is best.

Rubric:
{rubric}

Prompt:
{prompt}

Candidate response:
{response}

Respond with ONLY a single integer between {scale_min} and {scale_max}. Do not explain.
Score:"""


def _default_openai_judge(
    *,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    timeout_s: float,
) -> JudgeCallable:
    """Build a JudgeCallable that hits any OpenAI-compatible chat endpoint.

    No httpx import at module load — the verifier should remain importable
    on stripped-down installs that don't include http clients. We import
    inside the closure so the cost is only paid when the default judge
    actually runs.
    """

    resolved_base = base_url or os.environ.get("HALOFORGE_JUDGE_BASE_URL") or "http://127.0.0.1:8001/v1"
    resolved_key = api_key or os.environ.get("HALOFORGE_JUDGE_API_KEY") or "EMPTY"

    def _call(prompt: str) -> str:
        import httpx  # local import — see module docstring

        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(
                f"{resolved_base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {resolved_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8,
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])

    return _call


def _parse_score(raw: str, scale_min: int, scale_max: int) -> Optional[int]:
    """Extract an integer score in [scale_min, scale_max] from judge output.

    Judges are noisy — some return ``"4"``, some ``"Score: 4"``, some
    ``"4/5"``, some a sentence. We scan for the first integer in range.
    Returns None on hopeless input so the verifier can flag the failure
    instead of fabricating a score.
    """
    if not raw:
        return None
    for match in re.finditer(r"-?\d+", raw):
        try:
            value = int(match.group(0))
        except ValueError:
            continue
        if scale_min <= value <= scale_max:
            return value
    return None


@register_verifier("llm_judge")
class LLMJudgeVerifier(Verifier):
    """Rubric-grading verifier that calls an LLM judge.

    Args:
        rubric: Free-text rubric the judge follows.
        scoring_scale: Top of the integer scale; min is always 1.
        judge_model: Model name passed to the judge endpoint. Ignored
            when ``judge_callable`` is provided.
        prompt: The user prompt the candidate response is answering.
            Stored on the verifier so ``verify(response)`` can be called
            with just the candidate. Override per-call by passing a
            ``prompt=`` kwarg through ``verify_with_prompt``.
        judge_callable: Optional ``(prompt: str) -> str``. Defaults to an
            OpenAI-compatible client.
        base_url / api_key / timeout_s: Forwarded to the default judge
            callable. Ignored when ``judge_callable`` is provided.

    Example:
        v = LLMJudgeVerifier(
            rubric="Concise and factually correct.",
            scoring_scale=5,
            judge_model="meta-llama/Llama-3.2-3B-Instruct",
            prompt="Explain attention.",
        )
        result = v.verify("Attention is ...")
    """

    def __init__(
        self,
        *,
        rubric: str = _DEFAULT_RUBRIC,
        scoring_scale: int = 5,
        judge_model: str = "default",
        prompt: str = "",
        judge_callable: Optional[JudgeCallable] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_s: float = 30.0,
        max_workers: int = 4,
    ):
        super().__init__(max_workers=max_workers)
        if scoring_scale < 2:
            raise ValueError("scoring_scale must be >= 2")
        self.rubric = rubric or _DEFAULT_RUBRIC
        self.scoring_scale = int(scoring_scale)
        self.scale_min = 1
        self.scale_max = self.scoring_scale
        self.judge_model = judge_model
        self.prompt = prompt
        self._judge: JudgeCallable = judge_callable or _default_openai_judge(
            model=judge_model,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    def _build_prompt(self, response: str, *, prompt: Optional[str] = None) -> str:
        return _PROMPT_TEMPLATE.format(
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            rubric=self.rubric.strip(),
            prompt=(prompt or self.prompt or "").strip() or "(no prompt provided)",
            response=response.strip(),
        )

    def verify(self, code: str) -> VerifyResult:
        """Grade ``code`` (the candidate response) against the rubric."""
        return self.verify_with_prompt(code, prompt=self.prompt)

    def verify_with_prompt(self, response: str, *, prompt: str) -> VerifyResult:
        """Variant that takes the prompt explicitly.

        RAFT and the agentic trainer hand each candidate alongside the
        prompt that produced it, so this is the path they use.
        """
        if not response:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Empty candidate response",
                error="empty_response",
            )

        prompt_text = self._build_prompt(response, prompt=prompt)
        try:
            raw = self._judge(prompt_text)
        except Exception as exc:
            logger.warning("LLM judge call failed: %s", exc)
            return VerifyResult(
                success=False,
                reward=0.0,
                details=f"Judge error: {exc}",
                error="judge_failure",
            )

        score = _parse_score(raw, self.scale_min, self.scale_max)
        if score is None:
            return VerifyResult(
                success=False,
                reward=0.0,
                details=f"Could not parse score from judge: {raw[:120]!r}",
                error="unparseable_score",
            )

        # Map [1, scale_max] to [0.0, 1.0]. The lowest score (1) earns 0.0
        # so a clearly-bad response signals "do not include in next cycle"
        # under RAFT's reject-by-threshold path.
        reward = (score - self.scale_min) / max(1, (self.scale_max - self.scale_min))
        # Standard RAFT convention: success is "above the median".
        midpoint = (self.scale_min + self.scale_max) / 2.0
        success = score >= midpoint

        return VerifyResult(
            success=success,
            reward=float(reward),
            details=f"Judge scored {score}/{self.scale_max}",
        )


__all__ = ["LLMJudgeVerifier"]
