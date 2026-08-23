"""Termination and fidelity contracts for the continued-pretraining packer.

The packer is the only component that decides what text the model actually
sees, so these tests pin two properties the rest of the CPT stack assumes:

* ``append_unit`` always terminates and every source token is either packed or
  reported in the drop accounting;
* concatenating the packed blocks reconstructs the source corpus exactly,
  modulo the deliberately injected EOS tokens.

The termination cases run the packer in a subprocess with a timeout because a
regression there is an unkillable in-process spin, not an exception.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pytest

from halo_forge.cpt.packing import pack_corpus_records

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent

EOS_ID = 1


class CharTokenizer:
    """Character-level tokenizer with an exact decode inverse."""

    name_or_path = "test/char-tokenizer"
    eos_token = "<eos>"
    eos_token_id = EOS_ID
    pad_token = "<pad>"
    pad_token_id = 0
    vocab_size = 1 << 20
    special_tokens_map = {"eos_token": "<eos>", "pad_token": "<pad>"}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [ord(value) + 2 for value in str(text)]

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(int(value) - 2) for value in ids if int(value) > 1)


_CHILD_PROGRAM = """
import json
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])

from test_cpt_packing_contract import CharTokenizer
from halo_forge.cpt.packing import pack_corpus_records

payload = json.loads(sys.stdin.read())
results = []
for case in payload:
    split = pack_corpus_records(
        case["records"],
        CharTokenizer(),
        max_sequence_length=case["block_size"],
    )
    results.append(
        {
            "block_size": case["block_size"],
            "sequences": [list(value.input_ids) for value in split.sequences],
            "statistics": split.statistics,
        }
    )
json.dump(results, sys.stdout)
"""


def _pack_out_of_process(
    cases: List[Dict[str, Any]],
    *,
    timeout: float = 30.0,
) -> List[Dict[str, Any]]:
    """Pack corpora in a child process so a spin fails instead of hanging pytest."""

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _CHILD_PROGRAM, str(_TESTS_DIR), str(_REPO_ROOT)],
            input=json.dumps(cases),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_REPO_ROOT),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"pack_corpus_records did not terminate within {timeout}s: {exc}")
    if completed.returncode != 0:
        pytest.fail(f"packing child failed ({completed.returncode}):\n{completed.stderr}")
    return json.loads(completed.stdout)


def _assert_block_invariants(result: Mapping[str, Any]) -> None:
    block_size = int(result["block_size"])
    statistics = result["statistics"]
    for sequence in result["sequences"]:
        assert 2 <= len(sequence) <= block_size, (
            f"block_size={block_size} produced an unusable sequence of "
            f"{len(sequence)} tokens"
        )
    emitted = sum(len(sequence) for sequence in result["sequences"])
    assert emitted == int(statistics["packed_tokens"])
    accounted = int(statistics["packed_tokens"]) + int(statistics["dropped_tokens"])
    produced = int(statistics["content_tokens"]) + int(statistics["eos_tokens"])
    assert accounted == produced, (
        f"block_size={block_size}: {produced} tokens produced but "
        f"{accounted} packed/dropped"
    )
    assert bool(statistics["exact"]) == (int(statistics["dropped_tokens"]) == 0)


def test_packer_terminates_when_documents_end_on_a_block_boundary():
    """Two block-aligned documents must not spin forever in ``append_unit``."""

    cases = [
        {
            "records": [
                {"document_id": "one", "text": "abcdefgh"},
                {"document_id": "two", "text": "ijklmnop"},
            ],
            "block_size": 8,
        }
    ]
    result = _pack_out_of_process(cases)[0]
    _assert_block_invariants(result)
    flat = [value for sequence in result["sequences"] for value in sequence]
    assert flat.count(EOS_ID) == 2
    assert CharTokenizer().decode(flat) == "abcdefghijklmnop"


def test_every_token_is_packed_or_accounted_across_block_sizes():
    """Sweep pathological block alignments; each must terminate and balance."""

    corpora = [
        [
            {"document_id": "aligned-a", "text": "abcdefgh"},
            {"document_id": "aligned-b", "text": "ijklmnop"},
        ],
        [
            {"document_id": "single", "text": "a"},
            {"document_id": "long", "text": "bcdefghijklmnop"},
        ],
        [{"document_id": "paras", "text": "abc\n\ndefghij\n\nk"}],
        [{"document_id": "one-char-docs", "text": "a"}, {"document_id": "b", "text": "b"}],
        [{"document_id": "wide", "text": "abcdefghijklmnopqrstuvwxyz"}],
    ]
    cases = [
        {"records": records, "block_size": block_size}
        for records in corpora
        for block_size in (2, 3, 4, 5, 8, 9, 16)
    ]
    for result in _pack_out_of_process(cases, timeout=60.0):
        _assert_block_invariants(result)


def test_stranded_token_is_reported_instead_of_emitted_untrainable():
    """A token that cannot be rebalanced is dropped truthfully, never emitted."""

    split = pack_corpus_records(
        [{"document_id": "tiny", "text": "ab"}],
        CharTokenizer(),
        max_sequence_length=2,
    )
    assert [list(value.input_ids) for value in split.sequences] == [[99, 100]]
    assert split.statistics["dropped_tokens"] == 1
    assert split.statistics["exact"] is False


def test_packing_round_trips_code_paragraph_structure():
    """Packed blocks must decode back to the source, EOS tokens aside."""

    source = "def f():\n    a = 1\n\n    b = 2\n"
    tokenizer = CharTokenizer()
    split = pack_corpus_records(
        [{"document_id": "code", "text": source}],
        tokenizer,
        max_sequence_length=512,
    )
    flat = [value for sequence in split.sequences for value in sequence.input_ids]
    assert flat.count(EOS_ID) == 1
    assert flat[-1] == EOS_ID
    assert tokenizer.decode(flat) == source
    assert split.statistics["exact"] is True
    assert split.statistics["dropped_tokens"] == 0


def test_round_trip_survives_multi_document_and_block_splits():
    """Reconstruction stays exact when paragraphs are split across blocks."""

    sources = [
        "class A:\n    x = 1\n\n    def m(self):\n        return self.x\n",
        "# header\n\n- item one\n- item two\n",
    ]
    tokenizer = CharTokenizer()
    split = pack_corpus_records(
        [{"document_id": f"doc-{index}", "text": text} for index, text in enumerate(sources)],
        tokenizer,
        max_sequence_length=12,
    )
    flat = [value for sequence in split.sequences for value in sequence.input_ids]
    assert flat.count(EOS_ID) == 2
    assert tokenizer.decode(flat) == "".join(sources)
    assert split.statistics["exact"] is True
    assert split.statistics["dropped_tokens"] == 0
