"""Schema + reference-metric verifier tests (Tracks V3 + V4)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


# ----- V3: schema -----------------------------------------------------------


def test_json_structure_accepts_valid_json():
    from halo_forge.rlvr.verifiers import JSONStructureVerifier

    v = JSONStructureVerifier()
    result = v.verify('{"a": 1, "b": [1, 2, 3]}')
    assert result.success is True
    assert result.reward == 1.0


def test_json_structure_rejects_garbage():
    from halo_forge.rlvr.verifiers import JSONStructureVerifier

    v = JSONStructureVerifier()
    result = v.verify("{not valid json")
    assert result.success is False
    assert result.reward == 0.0
    assert result.error == "invalid_json"


def test_json_structure_strips_code_fence():
    """A model that wraps JSON in ```json ... ``` shouldn't be penalized."""
    from halo_forge.rlvr.verifiers import JSONStructureVerifier

    v = JSONStructureVerifier()
    fenced = '```json\n{"answer": 42}\n```'
    result = v.verify(fenced)
    assert result.success is True


def test_json_structure_empty_response():
    from halo_forge.rlvr.verifiers import JSONStructureVerifier

    v = JSONStructureVerifier()
    result = v.verify("")
    assert result.success is False
    assert result.error == "empty_response"


def test_json_schema_no_schema_falls_back_to_structure_check():
    from halo_forge.rlvr.verifiers import JSONSchemaVerifier

    v = JSONSchemaVerifier()
    result = v.verify('{"x": 1}')
    assert result.success is True


def test_json_schema_validates_against_schema():
    pytest.importorskip("jsonschema")
    from halo_forge.rlvr.verifiers import JSONSchemaVerifier

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name"],
    }
    v = JSONSchemaVerifier(schema=schema)

    ok = v.verify('{"name": "Alice", "age": 30}')
    assert ok.success is True
    assert ok.reward == 1.0

    missing = v.verify('{"age": 30}')
    assert missing.success is False
    assert missing.reward == 0.5  # partial credit (valid JSON, schema fail)


def test_json_schema_no_partial_credit():
    pytest.importorskip("jsonschema")
    from halo_forge.rlvr.verifiers import JSONSchemaVerifier

    schema = {"type": "object", "required": ["x"]}
    v = JSONSchemaVerifier(schema=schema, partial_credit=False)
    fail = v.verify('{"y": 1}')
    assert fail.reward == 0.0


def test_regex_format_search_semantics():
    from halo_forge.rlvr.verifiers import RegexFormatVerifier

    v = RegexFormatVerifier(pattern=r"Final answer:\s*(\d+)")
    ok = v.verify("Some reasoning... Final answer: 42")
    assert ok.success is True
    assert ok.reward == 1.0


def test_regex_format_full_match():
    from halo_forge.rlvr.verifiers import RegexFormatVerifier

    v = RegexFormatVerifier(pattern=r"\d+", full_match=True)
    full = v.verify("12345")
    assert full.success is True
    partial = v.verify("answer is 42")
    assert partial.success is False


def test_regex_format_full_match_partial_credit():
    from halo_forge.rlvr.verifiers import RegexFormatVerifier

    v = RegexFormatVerifier(pattern=r"\d+", full_match=True, partial_credit=True)
    near_miss = v.verify("answer is 42")
    assert near_miss.success is False
    assert near_miss.reward == 0.5
    assert near_miss.error == "partial_match"


def test_regex_format_requires_pattern():
    from halo_forge.rlvr.verifiers import RegexFormatVerifier

    with pytest.raises(ValueError):
        RegexFormatVerifier(pattern="")


# ----- registry membership --------------------------------------------------


def test_v3_v4_short_names_in_registry():
    """All six new verifiers register on package import."""
    from halo_forge.rlvr.verifiers import list_registered_verifiers

    names = set(list_registered_verifiers())
    assert {
        "json_structure",
        "json_schema",
        "regex_format",
        "bleu",
        "rouge",
        "chrf",
    }.issubset(names)


# ----- V4: reference metrics ------------------------------------------------


def _stub_sacrebleu(score: float):
    """Inject a fake sacrebleu module with a configurable score."""
    fake = ModuleType("sacrebleu")
    fake.corpus_bleu = lambda hyps, refs: SimpleNamespace(score=score * 100)  # type: ignore[attr-defined]
    fake.corpus_chrf = lambda hyps, refs, word_order=2: SimpleNamespace(score=score * 100)  # type: ignore[attr-defined]
    return fake


def test_bleu_verifier_requires_reference():
    from halo_forge.rlvr.verifiers import BLEUVerifier

    with pytest.raises(ValueError):
        BLEUVerifier()


def test_bleu_verifier_score_to_reward(monkeypatch):
    """BLEU is in [0, 100]; verifier divides by 100 to land in [0, 1]."""
    from halo_forge.rlvr.verifiers import BLEUVerifier

    monkeypatch.setitem(sys.modules, "sacrebleu", _stub_sacrebleu(0.42))
    v = BLEUVerifier(references="reference text")
    result = v.verify("candidate text")
    assert result.reward == pytest.approx(0.42, rel=1e-9)


def test_bleu_verifier_threshold_drives_success(monkeypatch):
    from halo_forge.rlvr.verifiers import BLEUVerifier

    monkeypatch.setitem(sys.modules, "sacrebleu", _stub_sacrebleu(0.45))
    high = BLEUVerifier(references="ref", success_threshold=0.3)
    assert high.verify("c").success is True

    monkeypatch.setitem(sys.modules, "sacrebleu", _stub_sacrebleu(0.20))
    low = BLEUVerifier(references="ref", success_threshold=0.3)
    assert low.verify("c").success is False


def test_chrf_verifier_score_to_reward(monkeypatch):
    from halo_forge.rlvr.verifiers import ChrFVerifier

    monkeypatch.setitem(sys.modules, "sacrebleu", _stub_sacrebleu(0.55))
    v = ChrFVerifier(references=["ref one", "ref two"])
    result = v.verify("candidate")
    assert result.reward == pytest.approx(0.55, rel=1e-9)


def test_rouge_verifier_uses_pluggable_scorer(monkeypatch):
    from halo_forge.rlvr.verifiers import ROUGEVerifier

    fake_module = ModuleType("rouge_score")
    rouge_scorer_module = ModuleType("rouge_score.rouge_scorer")

    class _FakeRougeScorer:
        def __init__(self, types, use_stemmer=True):
            self.types = types

        def score(self, ref, cand):
            return {self.types[0]: SimpleNamespace(fmeasure=0.78)}

    rouge_scorer_module.RougeScorer = _FakeRougeScorer  # type: ignore[attr-defined]
    fake_module.rouge_scorer = rouge_scorer_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rouge_score", fake_module)
    monkeypatch.setitem(sys.modules, "rouge_score.rouge_scorer", rouge_scorer_module)

    v = ROUGEVerifier(reference="hello world", rouge_type="rougeL")
    result = v.verify("hello there")
    assert result.reward == pytest.approx(0.78, rel=1e-9)


def test_rouge_verifier_validates_rouge_type():
    from halo_forge.rlvr.verifiers import ROUGEVerifier

    with pytest.raises(ValueError):
        ROUGEVerifier(reference="x", rouge_type="rouge9000")


def test_metric_verifiers_handle_empty_candidate():
    """All three metric verifiers short-circuit on empty input
    instead of asking the underlying lib to score an empty string."""
    from halo_forge.rlvr.verifiers import BLEUVerifier, ChrFVerifier, ROUGEVerifier

    # No need to inject sacrebleu/rouge — empty path runs before lazy import.
    bleu_v = BLEUVerifier(references="x")
    chrf_v = ChrFVerifier(references="x")
    rouge_v = ROUGEVerifier(reference="x")
    for v in (bleu_v, chrf_v, rouge_v):
        result = v.verify("")
        assert result.success is False
        assert result.error == "empty_response"
