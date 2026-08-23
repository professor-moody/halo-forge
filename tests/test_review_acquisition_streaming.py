"""Bounded-memory acquisition planning over durable one-pass sources."""

from __future__ import annotations

import gc
import tracemalloc

import pytest

from halo_forge.review_lab.acquisition import plan_acquisition


def _records(count: int = 24):
    for index in range(count):
        yield {
            "record_id": f"record-{index:05d}",
            "record": {
                "prompt": f"prompt {index}",
                "bucket": "rare" if index < 4 else "common",
            },
            "evidence": {
                "passed": index % 3 != 0,
                "outcome": (
                    "regression"
                    if index % 4 == 0
                    else "improvement" if index % 4 == 1 else "unchanged_pass"
                ),
                "verifier_disagreement": index % 5 == 0,
                "score": index / max(1, count - 1),
                "score_direction": "maximize",
                "score_metric": "quality",
                "margin": abs(index - (count // 2)) / max(1, count),
                "margin_metric": "gap",
                "category": "rare" if index < 4 else "common",
                "embedding": [float(index), float(index % 3)],
                "embedding_revision": "embedding@test",
            },
            "source": {"kind": "jsonl", "ref": "stream.jsonl"},
        }


@pytest.mark.parametrize(
    "strategies",
    [
        [{"kind": "explicit", "quota": 7}],
        [{"kind": "candidate_failure", "quota": 7}],
        [{"kind": "regression", "quota": 7}],
        [{"kind": "improvement", "quota": 7}],
        [{"kind": "verifier_disagreement", "quota": 7}],
        [{"kind": "low_score", "quota": 7}],
        [{"kind": "low_margin", "quota": 7}],
        [{"kind": "coverage_gap", "quota": 7}],
        [
            {
                "kind": "diversity",
                "quota": 7,
                "embedding_revision": "embedding@test",
            }
        ],
        [{"kind": "random", "quota": 7}],
        [
            {"kind": "candidate_failure", "quota": 3},
            {"kind": "random", "quota": 5},
        ],
    ],
)
def test_one_pass_disk_planner_matches_small_sequence_identity(strategies):
    rows = list(_records())
    direct = plan_acquisition(rows, strategies=strategies, seed=917)
    streamed = plan_acquisition(iter(rows), strategies=strategies, seed=917)

    assert [value.record_id for value in streamed.selected] == [
        value.record_id for value in direct.selected
    ]
    assert streamed.eligibility == direct.eligibility
    assert streamed.request == direct.request
    assert streamed.source_hash == direct.source_hash
    assert streamed.source_pins == direct.source_pins
    assert streamed.content_hash == direct.content_hash
    assert streamed.default_batch_id == direct.default_batch_id


class _OnePassStressSource:
    def __init__(self, count: int):
        self.count = count
        self.iterations = 0
        self.yielded = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations != 1:
            raise AssertionError("durable source was iterated more than once")
        for index in range(self.count):
            self.yielded += 1
            yield {
                "record_id": f"stress-{index:07d}",
                "record": {"prompt": "x" * 128, "index": index},
                "evidence": {
                    "score": float(index),
                    "score_direction": "maximize",
                    "score_metric": "quality",
                },
                "source": {"kind": "jsonl", "ref": "large.jsonl"},
            }


def test_one_pass_disk_planner_does_not_retain_the_candidate_population():
    source = _OnePassStressSource(5_000)
    gc.collect()
    tracemalloc.start()
    try:
        plan = plan_acquisition(
            source,
            strategies=[{"kind": "low_score", "quota": 32}],
            seed=71,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert source.iterations == 1
    assert source.yielded == 5_000
    assert plan.eligibility["supplied"] == 5_000
    assert len(plan.selected) == 32
    assert not isinstance(plan.selected, (list, tuple))
    assert [value.record_id for value in plan.selected][:3] == [
        "stress-0000000",
        "stress-0000001",
        "stress-0000002",
    ]
    # Replaying the selected result reads the small, durable SQLite selection;
    # it never asks the source for a second pass.
    assert len(list(plan.selected)) == 32
    # The historical list planner retains roughly 10 MiB for this fixture.
    # Leave ample interpreter variance while guarding against a regression to
    # retaining all 5,000 normalized records in Python objects.
    assert peak < 3 * 1024 * 1024
