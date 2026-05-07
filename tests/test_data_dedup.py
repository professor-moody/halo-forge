"""Dataset dedup tests (Track D2)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


# ----- exact dedup ----------------------------------------------------------


def test_exact_dedup_strings_removes_literal_duplicates():
    from halo_forge.data.dedup import exact_dedup

    records = ["hello", "world", "hello", "WORLD", "  hello  "]
    result = exact_dedup(records)
    # case-insensitive + whitespace-collapsing default catches all three
    # "hello" variants and both "WORLD" forms.
    assert result.n_input == 5
    assert result.n_output == 2
    assert result.kept_indices == [0, 1]


def test_exact_dedup_case_sensitive():
    from halo_forge.data.dedup import exact_dedup

    records = ["hello", "Hello", "HELLO"]
    insensitive = exact_dedup(records)
    assert insensitive.n_output == 1

    sensitive = exact_dedup(records, case_sensitive=True)
    assert sensitive.n_output == 3


def test_exact_dedup_dict_records_use_key():
    from halo_forge.data.dedup import exact_dedup

    records = [
        {"text": "alpha", "id": "a"},
        {"text": "alpha", "id": "b"},
        {"text": "beta", "id": "c"},
    ]
    result = exact_dedup(records, key="text")
    assert result.kept_indices == [0, 2]
    assert result.removed_indices == [1]


def test_exact_dedup_falls_back_to_str_for_unknown_records():
    from halo_forge.data.dedup import exact_dedup

    records = [42, 42, 99]
    result = exact_dedup(records)
    assert result.n_output == 2


def test_exact_dedup_empty_input_is_clean():
    from halo_forge.data.dedup import exact_dedup

    result = exact_dedup([])
    assert result.n_input == 0
    assert result.n_output == 0
    assert result.kept_indices == []


def test_exact_dedup_preserves_first_occurrence_order():
    """Order matters: when shuffling input changes which row survives,
    results aren't reproducible across runs."""
    from halo_forge.data.dedup import exact_dedup

    records = ["a", "b", "a", "c", "b"]
    result = exact_dedup(records)
    assert result.kept_indices == [0, 1, 3]
    assert result.removed_indices == [2, 4]


# ----- fuzzy dedup (with stubbed datasketch) --------------------------------


class _FakeMinHash:
    """Captures the shingles fed in so tests can assert on what we hashed."""

    def __init__(self, *, num_perm=128):
        self.num_perm = num_perm
        self.shingles: list[bytes] = []

    def update(self, b: bytes):
        self.shingles.append(b)


class _FakeLSH:
    """Threshold-based duplicate detector backed by exact-shingle equality.

    Treats two records as duplicates iff they share at least one shingle.
    Crude but enough to validate the fuzzy_dedup flow without datasketch."""

    def __init__(self, *, threshold=0.85, num_perm=128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.indexed: dict[str, set[bytes]] = {}

    def query(self, m):
        wanted = set(m.shingles)
        return [
            key for key, shingles in self.indexed.items()
            if wanted & shingles
        ]

    def insert(self, key, m):
        self.indexed[key] = set(m.shingles)


@pytest.fixture
def stub_datasketch(monkeypatch):
    fake = ModuleType("datasketch")
    fake.MinHash = _FakeMinHash  # type: ignore[attr-defined]
    fake.MinHashLSH = _FakeLSH  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasketch", fake)
    yield fake


def test_fuzzy_dedup_collapses_near_duplicates(stub_datasketch):
    from halo_forge.data.dedup import fuzzy_dedup

    # Both rows share enough word shingles to overlap under the fake LSH.
    records = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog!",  # near-duplicate
        "Completely different sentence about astronomy stars.",
    ]
    result = fuzzy_dedup(records, threshold=0.5, shingle_n=3)
    assert result.method == "fuzzy"
    assert result.threshold == 0.5
    assert result.n_input == 3
    # Row 1 deduped against row 0.
    assert result.kept_indices == [0, 2]
    assert result.removed_indices == [1]


def test_fuzzy_dedup_threshold_validation():
    from halo_forge.data.dedup import fuzzy_dedup

    # threshold must be in (0, 1) — values outside that range are invalid.
    with pytest.raises(ValueError):
        fuzzy_dedup(["a"], threshold=0.0)
    with pytest.raises(ValueError):
        fuzzy_dedup(["a"], threshold=1.0)


def test_fuzzy_dedup_requires_datasketch_when_unavailable(monkeypatch):
    """If datasketch is *not* present, the lazy import surfaces a clean
    error rather than a cryptic ImportError mid-loop."""
    import halo_forge.data.dedup as dedup_mod

    # Force the import to fail by removing any cached datasketch entry.
    monkeypatch.setitem(sys.modules, "datasketch", None)
    with pytest.raises(ImportError, match="datasketch"):
        dedup_mod.fuzzy_dedup(["a", "b"])


# ----- dispatch + file IO ---------------------------------------------------


def test_dispatch_unknown_method_raises():
    from halo_forge.data.dedup import dedup

    with pytest.raises(ValueError, match="Unknown dedup method"):
        dedup(["a", "b"], method="franken")


def test_dedup_file_roundtrip(tmp_path: Path):
    from halo_forge.data.dedup import dedup_file

    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    src.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"text": "hello world"},
                {"text": "hello world"},  # exact dup
                {"text": "different"},
            ]
        )
    )
    result = dedup_file(input_path=src, output_path=dst, method="exact")
    assert result.n_input == 3
    assert result.n_output == 2
    survivors = [json.loads(line) for line in dst.read_text().splitlines() if line]
    assert len(survivors) == 2


def test_write_jsonl_normalizes_strings_to_text_dict(tmp_path: Path):
    from halo_forge.data.dedup import write_jsonl

    out = tmp_path / "x.jsonl"
    write_jsonl(out, ["a", "b", "c"])
    rows = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert rows == [{"text": "a"}, {"text": "b"}, {"text": "c"}]


# ----- CLI ------------------------------------------------------------------


def test_cli_dedup_help_registers(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "data", "dedup", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for token in ("--method", "exact", "fuzzy", "--threshold"):
        assert token in out


def test_cli_dedup_end_to_end(tmp_path: Path, monkeypatch, capsys):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    src.write_text(
        "\n".join(
            json.dumps({"text": t}) for t in ["a", "a", "b", "c", "b"]
        )
    )

    import halo_forge.cli as cli_mod

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge", "data", "dedup",
            "--input", str(src),
            "--output", str(dst),
            "--method", "exact",
        ],
    )
    cli_mod.main()
    out = capsys.readouterr().out
    assert "Done" in out
    assert "kept:" in out
    survivors = [json.loads(line) for line in dst.read_text().splitlines() if line]
    # 5 inputs, 3 unique under exact dedup.
    assert len(survivors) == 3
