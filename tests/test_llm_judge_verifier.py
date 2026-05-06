"""LLM-as-judge verifier tests (Track V2).

These tests use a stub judge_callable so we can validate scoring math,
prompt construction, and error handling without depending on a real
LLM endpoint or network. The default OpenAI-compatible judge has its
own integration test path (gated on a running server) that's not part
of CI.
"""

from __future__ import annotations

import pytest


def test_v2_registered_in_plugin_registry():
    """Importing the verifiers package registers ``llm_judge`` automatically."""
    from halo_forge.rlvr.verifiers import get_verifier, list_registered_verifiers

    assert "llm_judge" in list_registered_verifiers()
    cls = get_verifier("llm_judge")
    assert cls.__name__ == "LLMJudgeVerifier"


def test_judge_score_maps_to_reward_unit_interval():
    """A 5/5 score yields reward=1.0; 1/5 yields 0.0; 3/5 yields 0.5."""
    from halo_forge.rlvr.verifiers import LLMJudgeVerifier

    seen_prompts: list[str] = []

    def stub_judge(p: str) -> str:
        seen_prompts.append(p)
        # Cycle through scores so each call returns a different one.
        return ["5", "3", "1"][len(seen_prompts) - 1]

    v = LLMJudgeVerifier(
        rubric="Quality and correctness.",
        scoring_scale=5,
        prompt="Explain X.",
        judge_callable=stub_judge,
    )
    r5 = v.verify("Excellent answer")
    r3 = v.verify("Decent answer")
    r1 = v.verify("Bad answer")
    assert r5.reward == 1.0 and r5.success is True
    assert r3.reward == 0.5 and r3.success is True  # 3 ≥ midpoint=3
    assert r1.reward == 0.0 and r1.success is False
    # Prompt should embed the rubric and the candidate response text.
    assert "Quality and correctness" in seen_prompts[0]
    assert "Excellent answer" in seen_prompts[0]


def test_score_parsing_tolerates_noisy_judge_output():
    """Real judges return ``"Score: 4"`` / ``"4/5"`` / a sentence — pull
    the first in-range integer."""
    from halo_forge.rlvr.verifiers.llm_judge import _parse_score

    assert _parse_score("4", 1, 5) == 4
    assert _parse_score("Score: 4", 1, 5) == 4
    assert _parse_score("4/5", 1, 5) == 4
    assert _parse_score("I would rate this 3 out of 5.", 1, 5) == 3
    # Out-of-range integers are skipped in favor of an in-range one.
    assert _parse_score("100 is too high; my pick is 4", 1, 5) == 4
    # Hopeless input returns None.
    assert _parse_score("dunno", 1, 5) is None
    assert _parse_score("", 1, 5) is None


def test_unparseable_score_yields_failure_not_fabrication():
    from halo_forge.rlvr.verifiers import LLMJudgeVerifier

    v = LLMJudgeVerifier(
        rubric="r",
        scoring_scale=5,
        prompt="p",
        judge_callable=lambda p: "I cannot evaluate this.",
    )
    result = v.verify("anything")
    assert result.success is False
    assert result.reward == 0.0
    assert result.error == "unparseable_score"


def test_judge_exception_is_caught_and_reported():
    from halo_forge.rlvr.verifiers import LLMJudgeVerifier

    def boom(p: str) -> str:
        raise ConnectionError("judge offline")

    v = LLMJudgeVerifier(rubric="r", scoring_scale=5, prompt="p", judge_callable=boom)
    result = v.verify("anything")
    assert result.success is False
    assert result.reward == 0.0
    assert result.error == "judge_failure"
    assert "judge offline" in result.details


def test_empty_response_short_circuits_without_calling_judge():
    """An empty candidate is grade-able without bothering the judge —
    saves a round trip and avoids the parser tripping over empty input."""
    from halo_forge.rlvr.verifiers import LLMJudgeVerifier

    called = []

    def judge(p: str) -> str:
        called.append(p)
        return "5"

    v = LLMJudgeVerifier(rubric="r", scoring_scale=5, prompt="p", judge_callable=judge)
    result = v.verify("")
    assert called == []
    assert result.success is False
    assert result.error == "empty_response"


def test_verify_with_prompt_overrides_constructor_prompt():
    from halo_forge.rlvr.verifiers import LLMJudgeVerifier

    seen: list[str] = []

    def judge(p: str) -> str:
        seen.append(p)
        return "5"

    v = LLMJudgeVerifier(
        rubric="r",
        scoring_scale=5,
        prompt="constructor-prompt",
        judge_callable=judge,
    )
    v.verify_with_prompt("response", prompt="per-call prompt")
    assert "per-call prompt" in seen[0]
    assert "constructor-prompt" not in seen[0]


def test_scoring_scale_validation():
    from halo_forge.rlvr.verifiers import LLMJudgeVerifier

    # Scale of 1 makes no sense (no spread).
    with pytest.raises(ValueError):
        LLMJudgeVerifier(scoring_scale=1, judge_callable=lambda p: "1")


def test_alternative_scoring_scales():
    """1-10 scale should map proportionally."""
    from halo_forge.rlvr.verifiers import LLMJudgeVerifier

    v = LLMJudgeVerifier(scoring_scale=10, judge_callable=lambda p: "8")
    result = v.verify("response")
    # (8 - 1) / (10 - 1) = 7/9 ≈ 0.7778
    assert result.reward == pytest.approx(7 / 9, rel=1e-9)
    assert result.success is True
