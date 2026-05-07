"""Synthetic data generation tests (Track D1).

Inject a stub teacher callable so tests don't hit any HTTP endpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# ---------- seed loading ----------------------------------------------------


def test_load_seeds_from_list():
    from halo_forge.data.synthesize import load_seeds

    assert load_seeds(["a", "b", "c"]) == ["a", "b", "c"]
    # Empty / falsy entries get filtered out.
    assert load_seeds(["a", "", None, "b"]) == ["a", "b"]


def test_load_seeds_from_text_file(tmp_path: Path):
    from halo_forge.data.synthesize import load_seeds

    p = tmp_path / "seeds.txt"
    p.write_text("alpha\nbeta\n\ngamma\n")
    assert load_seeds(p) == ["alpha", "beta", "gamma"]


def test_load_seeds_from_jsonl_uses_known_keys(tmp_path: Path):
    from halo_forge.data.synthesize import load_seeds

    p = tmp_path / "seeds.jsonl"
    p.write_text(
        "\n".join([
            json.dumps({"prompt": "p1"}),
            json.dumps({"text": "p2"}),
            json.dumps({"question": "p3"}),
            json.dumps({"instruction": "p4"}),
            json.dumps({"unrelated": "ignored"}),
        ])
    )
    assert load_seeds(p) == ["p1", "p2", "p3", "p4"]


def test_load_seeds_missing_file(tmp_path: Path):
    from halo_forge.data.synthesize import load_seeds

    with pytest.raises(FileNotFoundError):
        load_seeds(tmp_path / "nope.jsonl")


# ---------- synthesize_dataset (with stub teacher + stub verifier) ----------


@pytest.fixture(autouse=True)
def stub_verifier_registry(monkeypatch):
    """Inject a stub verifier into the V1 registry so tests don't depend
    on the real `execution`/`llm_judge` deps."""
    from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult

    class _StubByLength(Verifier):
        """Score = min(1, len(text) / 50). Deterministic + dep-free."""

        def __init__(self, **_):
            super().__init__(max_workers=1)

        def verify(self, code: str) -> VerifyResult:
            r = min(1.0, len(code or "") / 50.0)
            return VerifyResult(success=r >= 0.5, reward=r, details=f"len={len(code)}")

    from halo_forge.rlvr.verifiers.registry import register_verifier

    # Register under a name unlikely to collide with real verifiers.
    register_verifier("__test_length__")(_StubByLength)
    yield


def test_synthesize_sft_filters_below_threshold(tmp_path: Path):
    from halo_forge.data.synthesize import synthesize_dataset

    teacher_responses = iter([
        "x" * 60,   # reward = 1.0  (above 0.5) — kept
        "x" * 10,   # reward = 0.2  (below 0.5) — dropped
        "x" * 50,   # reward = 1.0  — kept
    ])

    result = synthesize_dataset(
        seeds=["s1", "s2", "s3"],
        output_path=tmp_path / "out.jsonl",
        teacher=lambda p: next(teacher_responses),
        verifier_name="__test_length__",
        n_per_prompt=1,
        reward_threshold=0.5,
    )

    assert result.n_seeds == 3
    assert result.n_generated == 3
    assert result.n_accepted == 2
    assert result.n_rejected == 1
    rows = [
        json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines() if l
    ]
    assert len(rows) == 2
    for row in rows:
        assert "prompt" in row and "completion" in row


def test_synthesize_preference_kind_writes_chosen_rejected(tmp_path: Path):
    from halo_forge.data.synthesize import synthesize_dataset

    # n=2 per prompt: best becomes chosen, worst becomes rejected.
    pairs = iter([
        ("x" * 60, "x" * 5),  # group 1: best=60, worst=5 → keep
        ("x" * 30, "x" * 30), # group 2: tied → drop
        ("x" * 4, "x" * 6),   # group 3: best=6 below threshold → drop
    ])
    flat = []
    for a, b in list(pairs):
        flat.append(a)
        flat.append(b)
    teacher_responses = iter(flat)

    result = synthesize_dataset(
        seeds=["s1", "s2", "s3"],
        output_path=tmp_path / "pref.jsonl",
        teacher=lambda p: next(teacher_responses),
        verifier_name="__test_length__",
        n_per_prompt=2,
        reward_threshold=0.5,
        output_kind="preference",
    )

    rows = [
        json.loads(l) for l in (tmp_path / "pref.jsonl").read_text().splitlines() if l
    ]
    assert len(rows) == 1
    assert rows[0]["chosen"] == "x" * 60
    assert rows[0]["rejected"] == "x" * 5
    assert rows[0]["chosen_reward"] > rows[0]["rejected_reward"]


def test_synthesize_preference_requires_n_per_prompt_ge_2():
    from halo_forge.data.synthesize import synthesize_dataset

    with pytest.raises(ValueError, match="n_per_prompt"):
        synthesize_dataset(
            seeds=["s1"],
            output_path="/tmp/x.jsonl",
            teacher=lambda p: "x",
            verifier_name="__test_length__",
            output_kind="preference",
            n_per_prompt=1,
        )


def test_synthesize_invalid_output_kind_raises():
    from halo_forge.data.synthesize import synthesize_dataset

    with pytest.raises(ValueError, match="output_kind"):
        synthesize_dataset(
            seeds=["s1"],
            output_path="/tmp/x.jsonl",
            teacher=lambda p: "x",
            verifier_name="__test_length__",
            output_kind="franken",
        )


def test_synthesize_no_seeds_raises(tmp_path: Path):
    from halo_forge.data.synthesize import synthesize_dataset

    with pytest.raises(ValueError, match="No seed prompts"):
        synthesize_dataset(
            seeds=[],
            output_path=tmp_path / "x.jsonl",
            teacher=lambda p: "x",
            verifier_name="__test_length__",
        )


def test_synthesize_handles_teacher_exceptions(tmp_path: Path):
    """A teacher that raises on one prompt shouldn't crash the run."""
    from halo_forge.data.synthesize import synthesize_dataset

    counter = {"n": 0}

    def flaky_teacher(p: str) -> str:
        counter["n"] += 1
        if counter["n"] == 2:
            raise ConnectionError("teacher offline")
        return "x" * 60

    result = synthesize_dataset(
        seeds=["s1", "s2", "s3"],
        output_path=tmp_path / "out.jsonl",
        teacher=flaky_teacher,
        verifier_name="__test_length__",
    )
    # 3 attempted; 2 succeeded; 1 errored.
    assert result.n_generated == 3
    assert result.n_accepted == 2
    rows = result.rows
    assert any(r.rejected_reason == "teacher_error" for r in rows)


def test_synthesize_loads_seeds_from_jsonl_path(tmp_path: Path):
    from halo_forge.data.synthesize import synthesize_dataset

    src = tmp_path / "seeds.jsonl"
    src.write_text(
        "\n".join([json.dumps({"prompt": f"p{i}"}) for i in range(5)])
    )

    result = synthesize_dataset(
        seeds=src,
        output_path=tmp_path / "out.jsonl",
        teacher=lambda p: "x" * 60,
        verifier_name="__test_length__",
    )
    assert result.n_seeds == 5
    assert result.n_accepted == 5


def test_synthesize_result_to_dict_trims_rows():
    """Wire shape: large rows array gets capped for the dict export."""
    from halo_forge.data.synthesize import SynthesisResult, SynthesisRow

    r = SynthesisResult(
        n_seeds=100, n_generated=100, n_accepted=80, n_rejected=20,
        avg_reward=0.5, threshold=0.5, output_path="/tmp/x.jsonl",
        duration_seconds=10.0, teacher_model="m", verifier_name="v",
        rows=[
            SynthesisRow(prompt=f"p{i}", completion="c", reward=0.8, accepted=True)
            for i in range(50)
        ],
    )
    d = r.to_dict()
    # Rows trimmed to 10 in the dict; full list still on the dataclass.
    assert len(d["rows"]) == 10
    assert len(r.rows) == 50


# ---------- CLI ------------------------------------------------------------


def test_cli_synthesize_help_registers(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "data", "synthesize", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for token in (
        "--seeds", "--teacher-model", "--verifier",
        "--n-per-prompt", "--kind", "--threshold",
    ):
        assert token in out
