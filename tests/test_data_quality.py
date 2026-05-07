"""Quality-scoring tests (Track D3)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# ---------- component scorers ----------


def test_length_score_in_band_is_one():
    from halo_forge.data.quality import length_score

    # Default band 50-1500.
    assert length_score("x" * 200) == 1.0


def test_length_score_too_short_decays_linearly():
    from halo_forge.data.quality import length_score

    assert length_score("") == 0.0
    assert length_score("a" * 25) == pytest.approx(0.5, rel=1e-6)


def test_length_score_runaway_long_decays():
    from halo_forge.data.quality import length_score

    band_max = 1500
    short_runaway = length_score("x" * (band_max + 100))
    big_runaway = length_score("x" * (band_max * 5))
    assert short_runaway < 1.0
    assert big_runaway < short_runaway
    assert big_runaway >= 0.0


def test_whitespace_score_blank_is_zero():
    from halo_forge.data.quality import whitespace_score

    assert whitespace_score("") == 0.0
    assert whitespace_score("    ") == 0.0


def test_alpha_ratio_score_penalizes_punctuation_only():
    from halo_forge.data.quality import alpha_ratio_score

    assert alpha_ratio_score("hello world") > 0.8
    assert alpha_ratio_score("###### !!!! ?????") < 0.05


def test_repetition_score_perfect_for_unique_text():
    from halo_forge.data.quality import repetition_score

    text = "the quick brown fox jumps over the lazy dog"
    assert repetition_score(text, n=3) == pytest.approx(1.0, rel=1e-6)


def test_repetition_score_low_for_stuck_loop():
    from halo_forge.data.quality import repetition_score

    text = "the the the the the the the the the the"
    assert repetition_score(text, n=2) < 0.2


def test_repetition_score_short_text_is_one():
    """Short text can't repeat; default to 1.0 instead of NaN."""
    from halo_forge.data.quality import repetition_score

    assert repetition_score("hi", n=3) == 1.0


def test_format_score_string_records_pass():
    from halo_forge.data.quality import format_score

    assert format_score("plain string") == 1.0


def test_format_score_chat_messages_shape():
    from halo_forge.data.quality import format_score

    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert format_score({"messages": msgs}) == 1.0
    # Empty content makes the message invalid.
    bad = [{"role": "user", "content": ""}, {"role": "assistant", "content": "hi"}]
    assert format_score({"messages": bad}) == 0.5


def test_format_score_preference_shape():
    """Preference rows: any 2 of {prompt, completion, chosen, rejected, text} = full credit."""
    from halo_forge.data.quality import format_score

    full = {"prompt": "p", "chosen": "c", "rejected": "r"}
    assert format_score(full) == 1.0


# ---------- heuristic_score composite ----------


def test_heuristic_score_high_for_clean_completion():
    from halo_forge.data.quality import heuristic_score

    text = (
        "The Pythagorean theorem states that in a right triangle, the square "
        "of the hypotenuse equals the sum of the squares of the other two sides. "
        "This relationship has been known for thousands of years."
    )
    score = heuristic_score(text)
    assert score.score > 0.7
    assert "length" in score.components
    assert all(0.0 <= v <= 1.0 for v in score.components.values())


def test_heuristic_score_low_for_blank():
    from halo_forge.data.quality import heuristic_score

    score = heuristic_score("")
    assert score.score < 0.3


def test_heuristic_score_low_for_stuck_loop():
    from halo_forge.data.quality import heuristic_score

    text = "the the the the the the the the the the the the"
    score = heuristic_score(text)
    # Repetition pulls the composite down even though length is fine.
    assert score.score < 0.7


def test_heuristic_score_low_for_punctuation_noise():
    from halo_forge.data.quality import heuristic_score

    text = "###############################################################"
    score = heuristic_score(text)
    assert score.components["alpha_ratio"] < 0.1
    assert score.score < 0.5


def test_heuristic_score_dict_records_extract_text():
    """Dict-shaped records pull the right field for scoring."""
    from halo_forge.data.quality import heuristic_score

    record = {
        "prompt": "Explain gravity.",
        "completion": (
            "Gravity is the force that attracts two bodies toward each other. "
            "It is one of the four fundamental forces of nature."
        ),
    }
    score = heuristic_score(record)
    assert score.score > 0.6


# ---------- score_with_judge ----------


def test_score_with_judge_clips_to_unit_interval():
    from halo_forge.data.quality import score_with_judge

    score = score_with_judge("any text", judge=lambda t: 1.5)
    assert score.score == 1.0
    score = score_with_judge("any text", judge=lambda t: -0.5)
    assert score.score == 0.0


def test_score_with_judge_handles_exceptions():
    from halo_forge.data.quality import score_with_judge

    def boom(t):
        raise RuntimeError("judge crashed")

    score = score_with_judge("text", judge=boom)
    assert score.score == 0.0
    assert score.rejected is True
    assert score.reason == "judge_error"


# ---------- batch + filter ----------


def test_score_records_threshold_filter():
    from halo_forge.data.quality import score_records

    records = [
        # Long, clean → score high
        "This is a thoughtful, well-formed sentence about astronomy and physics.",
        # Stuck loop → score low
        "the the the the the the the the the the",
        # Blank → score very low
        "",
        # Another clean
        "Compact, valid Python: def square(x): return x * x",
    ]
    result = score_records(records, threshold=0.5)
    assert result.n_input == 4
    # The two clean rows should survive; the loop / blank should drop.
    assert 0 in result.kept_indices
    assert 3 in result.kept_indices
    assert 2 in result.rejected_indices  # blank
    # `reasons` is bucketed by weakest component.
    assert sum(result.reasons.values()) == result.n_rejected


def test_score_file_threshold(tmp_path: Path):
    from halo_forge.data.quality import score_file

    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    src.write_text(
        "\n".join(
            json.dumps({"text": t})
            for t in [
                "Pristine sentence with proper structure and several words.",
                "",  # blank — should drop
                "qwerty " * 10,  # repetitive
                "Another well-formed sentence about programming languages.",
            ]
        )
    )
    result = score_file(input_path=src, output_path=dst, threshold=0.5)
    survivors = [json.loads(line) for line in dst.read_text().splitlines() if line]
    assert result.n_input == 4
    assert result.n_kept >= 2
    assert len(survivors) == result.n_kept


def test_score_file_top_k_pct_overrides_threshold(tmp_path: Path):
    from halo_forge.data.quality import score_file

    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    rows = [
        {"text": f"sentence number {i} about " + "topic " * 5} for i in range(10)
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows))

    result = score_file(input_path=src, output_path=dst, keep_top_k_pct=0.3)
    assert result.n_kept == 3  # 10 × 0.3 = 3
    survivors = [json.loads(line) for line in dst.read_text().splitlines() if line]
    assert len(survivors) == 3


def test_score_file_top_k_pct_validates_range(tmp_path: Path):
    from halo_forge.data.quality import score_file

    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"text": "x"}))
    with pytest.raises(ValueError, match="top_k_pct"):
        score_file(input_path=src, output_path=tmp_path / "y", keep_top_k_pct=2.0)


# ---------- CLI ----------


def test_cli_score_help_registers(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "data", "score", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for token in ("--threshold", "--top-k-pct"):
        assert token in out


def test_cli_score_end_to_end(tmp_path: Path, monkeypatch, capsys):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    src.write_text(
        "\n".join(
            json.dumps({"text": t})
            for t in [
                "Coherent sentence one with several real words.",
                "the the the the the the the the the the the",
                "Coherent sentence two with several real words.",
                "",
            ]
        )
    )

    import halo_forge.cli as cli_mod

    monkeypatch.setattr(
        sys, "argv",
        [
            "halo-forge", "data", "score",
            "--input", str(src),
            "--output", str(dst),
            "--threshold", "0.5",
        ],
    )
    cli_mod.main()
    out = capsys.readouterr().out
    assert "Done" in out
    assert "rejected:" in out
    survivors = [json.loads(line) for line in dst.read_text().splitlines() if line]
    assert len(survivors) >= 2
