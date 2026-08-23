from __future__ import annotations

from types import SimpleNamespace


def test_operational_performance_adapter_records_fixed_policy_evidence(tmp_path, monkeypatch):
    from halo_forge.evaluation_lab import EvaluationLabService
    from halo_forge.run_db import RunDatabase

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")

    class Tokenizer:
        @staticmethod
        def encode(value, add_special_tokens=False):
            assert add_special_tokens is False
            return str(value).split()

    class Adapter:
        _tokenizer = Tokenizer()

        @staticmethod
        def generate(prompt, **settings):
            assert prompt == "measure me"
            assert settings["temperature"] == 0.0
            return "three output tokens"

    capacity = SimpleNamespace(
        process=SimpleNamespace(rss_bytes=1_024),
        memory=SimpleNamespace(used_bytes=2_048),
        accelerator=None,
    )
    monkeypatch.setattr(
        "halo_forge.serving.adapter.build_serving_adapter",
        lambda *args, **kwargs: Adapter(),
    )
    monkeypatch.setattr(
        "halo_forge.workstation_jobs.resources.sample_workstation_capacity",
        lambda *args, **kwargs: capacity,
    )

    db = RunDatabase(":memory:")
    lab = EvaluationLabService(db, tmp_path / "evaluations")
    try:
        _suite, revision = lab.create_suite(
            name="Local performance",
            purpose="operational",
            items=[{"id": "prompt", "prompt": "measure me"}],
            primary_metric="total_latency_ms",
            direction="minimize",
            generation_settings={"temperature": 0.0, "max_tokens": 8},
        )
        assert revision is not None
        launched = lab.launch_evaluation(
            suite_revision_id=revision.id,
            adapter_id="performance",
            subject={"type": "model", "ref": str(model)},
        )
        completed = lab.jobs.wait(launched.evaluation.id, timeout=10)
        assert completed.status == "completed"

        metrics = {
            metric.name: metric.value
            for metric in db.list_evaluation_metrics(completed.id)
            if not metric.suite_item_id
        }
        assert metrics["load_time_ms"] >= 0
        assert metrics["total_latency_ms"] >= 0
        assert metrics["output_tokens_per_second"] > 0
        assert metrics["peak_process_memory_bytes"] == 1_024
        assert "time_to_first_token_ms" not in metrics

        samples = db.list_evaluation_samples(completed.id, limit=20)
        assert len(samples) == 7
        assert all(sample.mineable is False for sample in samples)
        measured = [sample for sample in samples if sample.input["phase"] == "measure"]
        assert len(measured) == 5
        assert all(sample.output["generation_seed"] == 42 for sample in measured)
        assert all(sample.output["load_time_ms"] is not None for sample in measured)
    finally:
        lab.shutdown()
        db.close()
