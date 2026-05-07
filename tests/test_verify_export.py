"""Round-trip export verification tests (Track I4)."""

from __future__ import annotations

import pytest


def test_char_overlap_identical_strings_is_one():
    from halo_forge.inference.verify_export import _char_overlap

    assert _char_overlap("hello", "hello") == 1.0
    assert _char_overlap("", "") == 1.0


def test_char_overlap_disjoint_strings_is_zero():
    from halo_forge.inference.verify_export import _char_overlap

    assert _char_overlap("abc", "xyz") == 0.0


def test_char_overlap_partial_is_jaccard():
    from halo_forge.inference.verify_export import _char_overlap

    # {a,b,c} ∩ {b,c,d} / {a,b,c,d} = 2/4 = 0.5
    assert _char_overlap("abc", "bcd") == pytest.approx(0.5, rel=1e-9)


def test_first_token_extraction():
    from halo_forge.inference.verify_export import _first_token

    assert _first_token("hello world") == "hello"
    assert _first_token("  leading whitespace") == "leading"
    assert _first_token("singleword") == "singleword"
    assert _first_token("") == ""
    assert _first_token("   ") == ""


def test_compare_generation_matched_outputs_pass():
    from halo_forge.inference.verify_export import compare_generation

    def gen(prompt: str) -> str:
        return f"answer: {prompt[:8]}"

    report = compare_generation(
        source_generate=gen,
        exported_generate=gen,  # identical
        prompts=["alpha", "bravo", "charlie", "delta"],
    )
    assert report.n_prompts == 4
    assert report.exact_match_rate == 1.0
    assert report.avg_char_overlap == 1.0
    assert report.avg_first_token_match == 1.0
    assert report.passed is True
    assert report.failures == []


def test_compare_generation_drift_fails_under_threshold():
    from halo_forge.inference.verify_export import compare_generation

    def src(p: str) -> str:
        return "the answer is 42"

    def exp(p: str) -> str:
        return "garbage random output"

    report = compare_generation(
        source_generate=src,
        exported_generate=exp,
        prompts=["q1", "q2", "q3"],
        char_overlap_threshold=0.7,
    )
    assert report.passed is False
    assert report.exact_match_rate == 0.0
    assert report.avg_char_overlap < 0.7
    assert len(report.failures) == 3


def test_compare_generation_first_token_signal():
    """A model that produces wildly different first tokens should fail
    even if the rest of the completion happens to share characters."""
    from halo_forge.inference.verify_export import compare_generation

    def src(p: str) -> str:
        return "Yes the answer is 42"

    def exp(p: str) -> str:
        return "No  the answer is 42"

    report = compare_generation(
        source_generate=src,
        exported_generate=exp,
        prompts=["q1", "q2", "q3"],
        first_token_threshold=0.5,
    )
    # Char overlap might be high, but first-token match is 0%.
    assert report.avg_first_token_match == 0.0
    assert report.passed is False


def test_compare_generation_partial_drift():
    """One prompt drifts, others match — failure list isolates the bad rows."""
    from halo_forge.inference.verify_export import compare_generation

    def src(p: str) -> str:
        return f"src-output-for-{p}"

    def exp(p: str) -> str:
        if p == "broken":
            return "totally different garbage"
        return f"src-output-for-{p}"

    report = compare_generation(
        source_generate=src,
        exported_generate=exp,
        prompts=["a", "b", "broken", "c"],
        char_overlap_threshold=0.7,
        first_token_threshold=0.5,
    )
    assert report.exact_match_rate == 0.75  # 3/4
    assert len(report.failures) == 1
    assert report.failures[0].prompt == "broken"


def test_compare_generation_handles_generator_exceptions():
    """A generator that raises shouldn't take down the whole run."""
    from halo_forge.inference.verify_export import compare_generation

    def src(p: str) -> str:
        return "fine"

    def exp(p: str) -> str:
        if p == "boom":
            raise RuntimeError("loader exploded")
        return "fine"

    report = compare_generation(
        source_generate=src,
        exported_generate=exp,
        prompts=["a", "boom", "b"],
    )
    # The exploded prompt logs and contributes empty exported output.
    assert report.n_prompts == 3
    boom_sample = next(s for s in report.samples if s.prompt == "boom")
    assert boom_sample.exported_completion == ""


def test_compare_generation_requires_prompts():
    from halo_forge.inference.verify_export import compare_generation

    with pytest.raises(ValueError, match="at least one prompt"):
        compare_generation(
            source_generate=lambda p: "x",
            exported_generate=lambda p: "x",
            prompts=[],
        )


def test_verify_export_gguf_raises_typed_error():
    """GGUF loading isn't wired in halo-forge serving yet — error should
    be informative, not cryptic."""
    from halo_forge.inference.verify_export import verify_export

    with pytest.raises(NotImplementedError, match="GGUF"):
        verify_export(
            source_model="x", exported_path="/tmp/y", target_format="gguf"
        )


def test_report_to_dict_trims_long_sample_lists():
    """Wire shape: report.to_dict caps sample arrays so 5000-prompt
    runs don't bloat the response payload."""
    from halo_forge.inference.verify_export import compare_generation

    report = compare_generation(
        source_generate=lambda p: "x",
        exported_generate=lambda p: "x",
        prompts=[f"p{i}" for i in range(50)],
    )
    d = report.to_dict()
    assert len(d["samples"]) == 5
    assert len(d["failures"]) == 0


def test_default_prompts_are_diverse():
    """The default prompt set should cover several capabilities so a
    silent-export-broken model fails on at least one."""
    from halo_forge.inference.verify_export import DEFAULT_VERIFICATION_PROMPTS

    assert len(DEFAULT_VERIFICATION_PROMPTS) >= 6
    # Spot-check we have at least one math, one code, one factual, one creative.
    flat = " ".join(DEFAULT_VERIFICATION_PROMPTS).lower()
    assert any(t in flat for t in ("python", "function"))  # code
    assert any(t in flat for t in ("17", "23", "*"))  # math
    assert any(t in flat for t in ("haiku", "complete"))  # creative
    assert any(t in flat for t in ("capital", "translate", "primary"))  # factual


def test_cli_convert_help_includes_verify(monkeypatch, capsys):
    import sys
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "convert", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "--verify" in out
