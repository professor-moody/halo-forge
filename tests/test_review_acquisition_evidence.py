from __future__ import annotations

import pytest

from halo_forge.review_lab.acquisition import plan_acquisition
from halo_forge.review_lab.embeddings import PinnedEmbeddingEngine
from halo_forge.review_lab.errors import ReviewValidationError


def _candidate(
    record_id: str,
    *,
    evidence=None,
    source=None,
    difficulty: int = 1,
):
    return {
        "record_id": record_id,
        "record": {"prompt": record_id, "metadata": {"difficulty": difficulty}},
        "evidence": dict(evidence or {}),
        "source": {
            "kind": "evaluation",
            "ref": "eval-1",
            "purpose": "development",
            **dict(source or {}),
        },
    }


def test_low_score_requires_compatible_finite_pinned_scores():
    with pytest.raises(ReviewValidationError, match="finite numeric score evidence"):
        plan_acquisition(
            [_candidate("a", evidence={"score": None})],
            strategies=[{"kind": "low_score"}],
        )

    with pytest.raises(ReviewValidationError, match="incompatible score directions"):
        plan_acquisition(
            [
                _candidate(
                    "a",
                    evidence={
                        "score": 0.1,
                        "score_direction": "maximize",
                        "score_metric": "accuracy",
                    },
                ),
                _candidate(
                    "b",
                    evidence={
                        "score": 0.2,
                        "score_direction": "minimize",
                        "score_metric": "accuracy",
                    },
                ),
            ],
            strategies=[{"kind": "low_score"}],
        )

    with pytest.raises(ReviewValidationError, match="incompatible score metrics"):
        plan_acquisition(
            [
                _candidate("a", evidence={"score": 0.1, "score_metric": "accuracy"}),
                _candidate("b", evidence={"score": 0.2, "score_metric": "reward"}),
            ],
            strategies=[{"kind": "low_score"}],
        )

    records = [
        _candidate(
            "b",
            evidence={
                "score": 0.2,
                "score_direction": "maximize",
                "score_metric": "accuracy",
            },
        ),
        _candidate(
            "a",
            evidence={
                "score": 0.1,
                "score_direction": "maximize",
                "score_metric": "accuracy",
            },
        ),
    ]
    first = plan_acquisition(records, strategies=[{"kind": "low_score", "quota": 1}])
    second = plan_acquisition(reversed(records), strategies=[{"kind": "low-score", "quota": 1}])
    options = first.request["strategies"][0]["options"]
    assert [value.record_id for value in first.selected] == ["a"]
    assert options["metric"] == "accuracy"
    assert options["direction"] == "maximize"
    assert options["score_count"] == 2
    assert options["score_evidence_hash"]
    assert first.content_hash == second.content_hash
    assert first.default_batch_id == second.default_batch_id
    assert first.source_pins == second.source_pins


def test_low_margin_requires_real_margin_or_paired_score_evidence():
    with pytest.raises(ReviewValidationError, match="margin evidence or compatible paired scores"):
        plan_acquisition(
            [_candidate("a", evidence={"score": 0.1})],
            strategies=[{"kind": "low_margin"}],
        )

    plan = plan_acquisition(
        [
            _candidate(
                "a",
                evidence={
                    "candidate_score": 0.7,
                    "base_score": 0.5,
                    "score_direction": "maximize",
                    "score_metric": "reward",
                },
            ),
            _candidate(
                "b",
                evidence={
                    "margin": 0.1,
                    "score_direction": "maximize",
                    "score_metric": "reward",
                },
            ),
        ],
        strategies=[{"kind": "low_margin"}],
    )
    assert [value.record_id for value in plan.selected] == ["b", "a"]
    options = plan.request["strategies"][0]["options"]
    assert options["metric"] == "reward"
    assert options["direction"] == "maximize"
    assert options["evidence_fields"] == ["margin", "paired_scores"]
    assert options["margin_semantics"] == "absolute_distance"


def test_diversity_requires_matching_revision_and_valid_fixed_dimension_embeddings():
    with pytest.raises(ReviewValidationError, match="lacks pinned revision"):
        plan_acquisition(
            [_candidate("a", evidence={"embedding": [0.0, 0.0]})],
            strategies=[{"kind": "diversity", "embedding_revision": "embed@sha"}],
        )
    with pytest.raises(ReviewValidationError, match="does not match pinned revision"):
        plan_acquisition(
            [
                _candidate(
                    "a",
                    evidence={
                        "embedding": [0.0, 0.0],
                        "embedding_revision": "other@sha",
                    },
                )
            ],
            strategies=[{"kind": "diversity", "embedding_revision": "embed@sha"}],
        )
    with pytest.raises(ReviewValidationError, match="one pinned vector dimension"):
        plan_acquisition(
            [
                _candidate(
                    "a",
                    evidence={"embedding": [0.0], "embedding_revision": "embed@sha"},
                ),
                _candidate(
                    "b",
                    evidence={
                        "embedding": [1.0, 1.0],
                        "embedding_revision": "embed@sha",
                    },
                ),
            ],
            strategies=[{"kind": "diversity", "quota": 1, "embedding_revision": "embed@sha"}],
        )

    plan = plan_acquisition(
        [
            _candidate(
                "b",
                evidence={"embedding": [1.0, 0.0], "embedding_revision": "embed@sha"},
            ),
            _candidate(
                "c",
                evidence={"embedding": [3.0, 0.0], "embedding_revision": "embed@sha"},
            ),
            _candidate(
                "a",
                evidence={"embedding": [0.0, 0.0], "embedding_revision": "embed@sha"},
            ),
        ],
        strategies=[{"kind": "diversity", "quota": 2, "embedding_revision": "embed@sha"}],
    )
    assert [value.record_id for value in plan.selected] == ["a", "c"]
    options = plan.request["strategies"][0]["options"]
    assert options["embedding_revision"] == "embed@sha"
    assert options["embedding_dimension"] == 2
    assert options["embedding_count"] == 3
    assert options["embedding_evidence_hash"]


def test_embedding_generation_refuses_unpinned_model_identity_before_loading_backend():
    with pytest.raises(ReviewValidationError, match="model@revision"):
        PinnedEmbeddingEngine().embed_envelopes(
            [{"record": {"prompt": "candidate"}, "source": {}}],
            embedding_revision="sentence-transformers/all-MiniLM-L6-v2",
        )


def test_safe_filters_are_applied_and_pinned_while_unsupported_filters_fail():
    records = [
        _candidate("a", evidence={"state": "ready"}, difficulty=2),
        _candidate("b", evidence={"state": "ready"}, difficulty=8),
        _candidate("c", evidence={"state": "blocked"}, difficulty=3),
    ]
    filters = [
        {"field": "evidence.state", "op": "eq", "value": "ready"},
        {"scope": "source", "field": "purpose", "op": "in", "value": ["development"]},
        {"field": "record.metadata.difficulty", "op": "range", "min": 1, "max": 5},
    ]
    plan = plan_acquisition(records, filters=filters)
    assert [value.record_id for value in plan.selected] == ["a"]
    assert plan.eligibility["filtered_out"] == 2
    assert plan.request["filters"] == [
        {
            "scope": "record",
            "field": "metadata.difficulty",
            "op": "range",
            "min": 1.0,
            "max": 5.0,
        },
        {
            "scope": "source",
            "field": "purpose",
            "op": "in",
            "value": ["development"],
        },
        {"scope": "evidence", "field": "state", "op": "eq", "value": "ready"},
    ]

    metadata_plan = plan_acquisition(records, metadata={"filters": filters})
    assert (
        metadata_plan.content_hash
        == plan_acquisition(records, filters=filters, metadata={"filters": filters}).content_hash
    )
    assert [value.record_id for value in metadata_plan.selected] == ["a"]

    with pytest.raises(ReviewValidationError, match="operator must be eq, in, or range"):
        plan_acquisition(records, filters=[{"field": "prompt", "op": "regex", "value": ".*"}])
    with pytest.raises(ReviewValidationError, match="unsupported fields"):
        plan_acquisition(
            records,
            filters=[{"field": "prompt", "op": "eq", "value": "a", "expression": "x"}],
        )
