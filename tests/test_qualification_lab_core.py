from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from halo_forge.qualification_lab import (
    DEFAULT_CONCURRENCY,
    DEFAULT_GENERATION_SEED,
    DEFAULT_MEASURED_REPEATS,
    DEFAULT_WARMUP_RUNS,
    InferencePerformanceAdapter,
    PERFORMANCE_METRIC_POLICIES,
    ParetoPoint,
    PerformanceSample,
    PerformanceSettings,
    QualificationEvidence,
    QualificationMetricRule,
    QualificationProfileRevision,
    ServingProfileRevision,
    compare_qualification_evidence,
    evaluate_qualification,
    pareto_front,
    promotion_eligibility,
)


def _profile(*, holdout: bool = True) -> QualificationProfileRevision:
    return QualificationProfileRevision(
        profile_id="qual-local",
        revision_number=1,
        name="Local GGUF qualification",
        development_suite_revision_id="suite-development-r3",
        operational_suite_revision_id="suite-operational-r2",
        holdout_suite_revision_id="suite-holdout-r1" if holdout else None,
        development_rules=(
            QualificationMetricRule(
                "accuracy",
                "maximize",
                pass_threshold=0.80,
                warn_threshold=0.75,
                maximum_regression=0.03,
            ),
        ),
        operational_rules=(
            QualificationMetricRule(
                "total_latency_ms",
                "minimize",
                pass_threshold=100,
                warn_threshold=120,
            ),
            QualificationMetricRule(
                "peak_device_memory_bytes",
                "minimize",
                pass_threshold=8_000_000_000,
                required=False,
            ),
        ),
        holdout_rules=(
            (QualificationMetricRule("accuracy", "maximize", pass_threshold=0.78),)
            if holdout
            else ()
        ),
        target_backend="llama.cpp",
        generation_settings={"temperature": 0, "stop": ["</s>"]},
    )


def _evidence(
    profile: QualificationProfileRevision,
    artifact: str,
    *,
    accuracy: float = 0.82,
    latency: float = 90,
    memory: int | None = 7_000_000_000,
    holdout_accuracy: float | None = None,
) -> QualificationEvidence:
    return QualificationEvidence(
        artifact_hash=artifact,
        profile_content_hash=profile.content_hash,
        development_metrics={"accuracy": accuracy},
        operational_metrics={
            "total_latency_ms": latency,
            "peak_device_memory_bytes": memory,
        },
        holdout_metrics={"accuracy": holdout_accuracy} if holdout_accuracy is not None else {},
        holdout_complete=holdout_accuracy is not None,
    )


def test_profile_revision_is_immutable_and_has_definition_hash() -> None:
    settings = {"temperature": 0, "stop": ["</s>"]}
    profile = _profile()
    copied = profile.to_dict()
    copied.update(
        profile_id="another-logical-profile",
        revision_number=7,
        name="Renamed",
    )
    same_definition = QualificationProfileRevision.from_dict(copied)

    assert profile.content_hash == same_definition.content_hash
    assert len(profile.content_hash) == 64
    assert profile.performance_settings == PerformanceSettings()
    assert profile.to_dict()["generation_settings"] == settings
    with pytest.raises(FrozenInstanceError):
        profile.name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        profile.generation_settings["temperature"] = 1  # type: ignore[index]


def test_profile_round_trip_validates_stored_content_hash() -> None:
    profile = _profile()
    assert QualificationProfileRevision.from_dict(profile.to_dict()) == profile
    tampered = profile.to_dict()
    tampered["target_backend"] = "mlx"
    with pytest.raises(ValueError, match="content_hash"):
        QualificationProfileRevision.from_dict(tampered)


@pytest.mark.parametrize(
    "rule, message",
    [
        (
            {"metric": "quality", "direction": "sideways", "pass_threshold": 1},
            "direction",
        ),
        ({"metric": "quality", "direction": "maximize"}, "requires"),
        (
            {
                "metric": "quality",
                "direction": "maximize",
                "pass_threshold": 0.8,
                "warn_threshold": 0.9,
            },
            "warn_threshold",
        ),
        (
            {
                "metric": "quality",
                "direction": "minimize",
                "pass_threshold": 10,
                "maximum_regression": -1,
            },
            "maximum_regression",
        ),
    ],
)
def test_metric_rule_validation(rule: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        QualificationMetricRule.from_dict(rule)


def test_profile_requires_distinct_suites_and_matching_holdout_rules() -> None:
    profile = _profile(holdout=False)
    values = profile.to_dict()
    values.pop("content_hash")
    values["operational_suite_revision_id"] = values["development_suite_revision_id"]
    with pytest.raises(ValueError, match="distinct"):
        QualificationProfileRevision.from_dict(values)

    values = profile.to_dict()
    values.pop("content_hash")
    values["holdout_suite_revision_id"] = "suite-holdout"
    with pytest.raises(ValueError, match="configured together"):
        QualificationProfileRevision.from_dict(values)


def test_serving_profile_revision_hash_and_round_trip() -> None:
    profile = ServingProfileRevision(
        profile_id="serve-local",
        revision_number=1,
        name="Approved local endpoint",
        artifact_id="artifact-occurrence-1",
        artifact_hash="abc123",
        backend="llama.cpp",
        endpoint_settings={"host": "127.0.0.1", "port": 8080},
        chat_template="{{ messages }}",
        generation_defaults={"temperature": 0.2},
        resource_expectations={"device_memory_bytes": 4_000_000_000},
    )

    assert ServingProfileRevision.from_dict(profile.to_dict()) == profile
    copied = profile.to_dict()
    copied.update(profile_id="serve-copy", revision_number=2, name="Copy")
    assert profile.content_hash == ServingProfileRevision.from_dict(copied).content_hash


def test_default_performance_settings_are_fixed_and_sequential() -> None:
    settings = PerformanceSettings()
    assert settings.to_dict() == {
        "warmup_runs": DEFAULT_WARMUP_RUNS,
        "measured_repeats": DEFAULT_MEASURED_REPEATS,
        "concurrency": DEFAULT_CONCURRENCY,
        "generation_seed": DEFAULT_GENERATION_SEED,
    }
    with pytest.raises(ValueError, match="concurrency=1"):
        PerformanceSettings(concurrency=2)
    with pytest.raises(TypeError, match="integer"):
        PerformanceSettings(measured_repeats=5.0)  # type: ignore[arg-type]
    assert PERFORMANCE_METRIC_POLICIES["total_latency_ms"] == "median"


def test_performance_adapter_uses_injected_runner_and_preserves_missing_metrics() -> None:
    requests = []

    def runner(request):
        requests.append(request)
        if request.phase == "measure" and request.iteration == 1:
            raise RuntimeError("backend unavailable")
        return {
            "load_time_ms": 100 + request.iteration,
            "ttft_ms": 10 + request.iteration,
            "total_latency_ms": 50 + request.iteration,
            "output_tokens": 25,
            "output_tokens_per_second": 50,
            "peak_process_memory_bytes": 1024 + request.iteration,
            # Device memory is intentionally unavailable on this runner.
            "runtime_versions": {"runtime": "test-1"},
            "hardware_identity": {"cpu": "test"},
        }

    aggregate = InferencePerformanceAdapter(runner).run(
        artifact_ref="artifact-1",
        backend="test",
        prompt="hello",
        generation_settings={"max_new_tokens": 25},
        artifact_size_bytes=4096,
    )

    assert len(requests) == 7
    assert [request.phase for request in requests[:2]] == ["warmup", "warmup"]
    assert all(request.generation_seed == DEFAULT_GENERATION_SEED for request in requests)
    assert aggregate.warmup_count == 2
    assert aggregate.measured_count == 5
    assert aggregate.successful_count == 4
    assert aggregate.failed_count == 1
    assert aggregate.metric_values()["peak_device_memory_bytes"] is None
    assert aggregate.metric_values()["artifact_size_bytes"] == 4096
    assert aggregate.metric_values()["total_latency_ms"] == 52.5
    assert "RuntimeError: backend unavailable" in aggregate.samples[3].error
    assert type(aggregate).from_dict(aggregate.to_dict()) == aggregate


def test_performance_sample_rejects_invalid_readings_without_filling_zeros() -> None:
    sample = PerformanceSample("measure", 0, 42)
    assert sample.time_to_first_token_ms is None
    assert sample.peak_device_memory_bytes is None
    with pytest.raises(ValueError, match="non-negative"):
        PerformanceSample("measure", 0, 42, total_latency_ms=-1)


def test_qualification_pass_warn_fail_and_direction_aware_regression() -> None:
    profile = _profile()
    parent = _evidence(profile, "parent", accuracy=0.82, latency=95)

    passing = evaluate_qualification(profile, _evidence(profile, "pass"), parent=parent)
    assert passing.development.status == "pass"
    assert passing.operational.status == "pass"
    assert passing.holdout.status == "warn"
    assert passing.overall_status == "warn"

    warning = evaluate_qualification(
        profile,
        _evidence(profile, "warn", accuracy=0.79, latency=110),
        parent=parent,
    )
    assert warning.development.status == "warn"
    assert warning.operational.status == "warn"

    failed = evaluate_qualification(
        profile,
        _evidence(profile, "fail", accuracy=0.76, latency=130),
        parent=parent,
    )
    assert failed.development.status == "fail"
    assert failed.operational.status == "fail"
    accuracy = failed.development.metrics[0]
    assert accuracy.raw_delta == pytest.approx(-0.06)
    assert accuracy.favorable_delta == pytest.approx(-0.06)
    assert any("exceeded allowed" in reason for reason in accuracy.reasons)


def test_missing_metrics_are_explicit_failures_or_warnings() -> None:
    profile = _profile(holdout=False)
    candidate = QualificationEvidence(
        artifact_hash="candidate",
        profile_content_hash=profile.content_hash,
        development_metrics={},
        operational_metrics={"total_latency_ms": 90},
    )
    decision = evaluate_qualification(profile, candidate)
    assert decision.development.status == "fail"
    assert decision.operational.status == "warn"
    assert "required metric accuracy is missing" in decision.reasons
    assert "optional metric peak_device_memory_bytes is unavailable" in decision.reasons


def test_comparison_rejects_mixed_profiles_and_reports_favorable_deltas() -> None:
    profile = _profile(holdout=False)
    parent = _evidence(profile, "parent", accuracy=0.80, latency=100)
    candidate = _evidence(profile, "candidate", accuracy=0.82, latency=90)
    comparison = compare_qualification_evidence(profile, parent, candidate)
    deltas = {delta.metric: delta for delta in comparison.deltas}
    assert deltas["accuracy"].favorable_delta == pytest.approx(0.02)
    assert deltas["total_latency_ms"].raw_delta == -10
    assert deltas["total_latency_ms"].favorable_delta == 10

    other_profile = QualificationEvidence(
        artifact_hash="other",
        profile_content_hash="different-profile",
    )
    with pytest.raises(ValueError, match="same profile"):
        compare_qualification_evidence(profile, parent, other_profile)


def test_promotion_rules_distinguish_candidate_and_approved() -> None:
    profile = _profile()
    parent = _evidence(profile, "parent")
    pending_holdout = evaluate_qualification(
        profile,
        _evidence(profile, "candidate"),
        parent=parent,
    )
    assert promotion_eligibility(pending_holdout, "candidate").eligible
    approved = promotion_eligibility(pending_holdout, "approved")
    assert not approved.eligible
    assert approved.requires_override

    override = promotion_eligibility(
        pending_holdout,
        "approved",
        override_note="Reviewed operationally; accepting pending holdout.",
    )
    assert override.eligible and override.overridden
    assert override.override_note

    confirmed = evaluate_qualification(
        profile,
        _evidence(profile, "confirmed", holdout_accuracy=0.80),
        parent=parent,
    )
    assert confirmed.overall_status == "pass"
    assert promotion_eligibility(confirmed, "approved").eligible


def test_pareto_front_handles_directions_ties_and_incomplete_evidence() -> None:
    points = [
        ParetoPoint("small", {"quality": 0.80, "latency": 50, "size": 4}),
        ParetoPoint("quality", {"quality": 0.90, "latency": 80, "size": 8}),
        ParetoPoint("dominated", {"quality": 0.70, "latency": 90, "size": 10}),
        ParetoPoint("unknown-memory", {"quality": 0.95, "latency": 40, "size": None}),
        ParetoPoint("tie", {"quality": 0.80, "latency": 50, "size": 4}),
    ]
    result = pareto_front(
        points,
        {"quality": "maximize", "latency": "minimize", "size": "minimize"},
    )
    assert [point.identity for point in result.frontier] == ["quality", "small", "tie"]
    assert [point.identity for point in result.dominated] == ["dominated"]
    assert [point.identity for point in result.incomplete] == ["unknown-memory"]


def test_content_hashes_are_order_independent_but_scientific_settings_matter() -> None:
    profile = _profile(holdout=False)
    reordered = QualificationProfileRevision.from_dict(
        {
            **profile.to_dict(),
            "generation_settings": {"stop": ["</s>"], "temperature": 0},
        }
    )
    assert reordered.content_hash == profile.content_hash

    changed = profile.to_dict()
    changed.pop("content_hash")
    changed["performance_settings"]["generation_seed"] = 99
    assert QualificationProfileRevision.from_dict(changed).content_hash != profile.content_hash
