"""Focused contracts for database-backed evaluation sample pagination."""

from __future__ import annotations


def _completed_evaluation(db, *, sample_count: int = 5):
    suite = db.create_benchmark_suite(name="pagination-suite")
    revision = db.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="pagination-suite-v1",
        items=[{"id": f"item-{index}"} for index in range(sample_count)],
        primary_metric="score",
        direction="maximize",
    )
    evaluation = db.create_evaluation(
        suite_revision_id=revision.id,
        adapter_id="dataset",
        adapter_version="1",
        subject_type="model",
        subject_ref="fixture-model",
        subject_hash="fixture-model-hash",
        reuse_key="pagination-evaluation",
        request={},
    )
    return db.complete_evaluation(
        evaluation.id,
        metrics=[{"name": "score", "value": 1.0, "direction": "maximize"}],
        samples=[
            {
                "suite_item_id": f"item-{index}",
                "record_id": f"record-{index}",
                "input": index,
                "expected": index,
                "output": index,
                "score": 1.0,
                "passed": True,
            }
            for index in range(sample_count)
        ],
        result={"primary_metric": "score"},
        artifact_path="/tmp/pagination-evaluation",
    )


def test_run_database_counts_and_pages_evaluation_samples():
    from halo_forge.run_db import RunDatabase

    db = RunDatabase(":memory:")
    try:
        evaluation = _completed_evaluation(db)

        assert db.count_evaluation_samples(evaluation.id) == 5
        page = db.list_evaluation_samples(evaluation.id, offset=2, limit=2)
        assert [sample.record_id for sample in page] == ["record-2", "record-3"]
        assert all(sample.evidence_kind == "legacy" for sample in page)
        assert all(sample.valid is False and sample.mineable is False for sample in page)
    finally:
        db.close()


def test_public_api_requests_only_the_selected_sample_page(monkeypatch, tmp_path):
    from halo_forge.public_api.service import PublicApiService
    from halo_forge.run_db import RunDatabase

    db = RunDatabase(":memory:")
    try:
        evaluation = _completed_evaluation(db)
        original = db.list_evaluation_samples
        calls = []

        def tracked(evaluation_id, *, limit=None, offset=0):
            calls.append({"evaluation_id": evaluation_id, "limit": limit, "offset": offset})
            return original(evaluation_id, limit=limit, offset=offset)

        monkeypatch.setattr(db, "list_evaluation_samples", tracked)
        service = PublicApiService(database=db, base_path=tmp_path)

        response = service.get_evaluation_samples(evaluation.id, offset=1, limit=2)

        assert response == {
            "items": [sample.to_dict() for sample in original(evaluation.id, offset=1, limit=2)],
            "total": 5,
            "offset": 1,
            "limit": 2,
        }
        assert calls == [{"evaluation_id": evaluation.id, "limit": 2, "offset": 1}]
    finally:
        db.close()
