from __future__ import annotations

import math

import pytest

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult
from halo_forge.verifier_lab.adapters import (
    ArtifactRewardModelReliabilityAdapter,
    CallableReliabilityAdapter,
    RegistryVerifierReliabilityAdapter,
    VerifierReliabilityAdapterRegistry,
)
from halo_forge.verifier_lab.chains import aggregate_chain_observations, validate_chain_graph
from halo_forge.verifier_lab.comparison import (
    compare_calibration_samples,
    compare_calibrations,
)
from halo_forge.verifier_lab.fingerprints import (
    configuration_hash,
    fingerprint_verifier_class,
    sanitize_configuration,
)
from halo_forge.verifier_lab.metrics import (
    CalibrationEvidence,
    compute_calibration_metrics,
    compute_task_metrics,
    grouped_percentile_bootstrap,
)
from halo_forge.verifier_lab.observation import (
    RewardContract,
    RewardContractError,
    VerifierObservation,
    normalize_reward_contract,
    normalize_verifier_result,
)
from halo_forge.verifier_lab.protocol import (
    CalibrationProtocol,
    dtype_score_tolerance,
    expand_calibration_protocol,
    grouped_calibration_confirmation_partition,
    iter_calibration_protocol,
    reward_model_batch_consistent,
)
from halo_forge.verifier_lab.qualification import (
    binary_threshold_curve,
    promotion_eligibility,
    qualify_calibration,
    runtime_compatibility,
)


def test_reward_contract_rejects_non_finite_and_out_of_range_rewards():
    contract = RewardContract(tie_policy="fail")
    with pytest.raises(RewardContractError, match="finite"):
        contract.validate_reward(math.nan)
    with pytest.raises(RewardContractError, match="outside"):
        contract.validate_reward(1.01)
    with pytest.raises(RewardContractError, match="tie threshold"):
        RewardContract().classify(0.5)


def test_public_reward_contract_defaults_and_continuous_threshold_normalize():
    from halo_forge.verifier_lab.models import VerifierRewardContract

    public = VerifierRewardContract(threshold=None)
    contract = normalize_reward_contract(public)
    assert contract.tie_policy == "error"
    assert contract.error_behavior == "fail_closed"
    assert contract.classify(0.75) is None


def test_legacy_result_normalization_uses_pinned_contract_not_legacy_success():
    result = VerifyResult(success=True, reward=0.4, details="legacy said pass")
    observation = normalize_verifier_result(
        result,
        contract=RewardContract(threshold=0.5, tie_policy="fail"),
    )
    assert observation.passed is False
    assert observation.details["legacy_success"] is True


def test_configuration_scrubbing_is_recursive_and_hashes_sanitized_values():
    value = {
        "endpoint": "https://alice:password@example.test/v1?api_key=top-secret&mode=judge",
        "nested": {"authorization": "Bearer secret", "model": "judge-v2"},
    }
    scrubbed = sanitize_configuration(value)
    assert "password" not in scrubbed["endpoint"]
    assert "top-secret" not in scrubbed["endpoint"]
    assert scrubbed["nested"]["authorization"] == "<redacted>"
    assert configuration_hash(value) == configuration_hash(
        {
            "endpoint": "https://bob:different@example.test/v1?api_key=changed&mode=judge",
            "nested": {"authorization": "different", "model": "judge-v2"},
        }
    )


def test_fingerprint_builtin_includes_source_and_opaque_entry_point_cannot_qualify():
    from halo_forge.rlvr.verifiers.schema import RegexFormatVerifier

    builtin = fingerprint_verifier_class("regex_format", RegexFormatVerifier, origin="builtin")
    assert builtin.qualifiable is True
    assert builtin.source_hash
    assert builtin.fingerprint

    class OpaqueVerifier(Verifier):
        def verify(self, code: str) -> VerifyResult:
            return VerifyResult(success=True, reward=1.0, details="ok")

    external = fingerprint_verifier_class("opaque", OpaqueVerifier, origin="entry_point")
    assert external.qualifiable is False
    assert external.reason


def test_registry_adapter_wraps_existing_verifier_without_changing_base_class():
    adapter = RegistryVerifierReliabilityAdapter(
        "regex_format", configuration={"pattern": r"^ok$", "full_match": True}
    )
    capability = adapter.capability()
    assert capability.family == "deterministic"
    assert capability.qualifiable is True
    observation = adapter.invoke(
        {"candidate": "ok"},
        contract=RewardContract(threshold=0.5, tie_policy="fail"),
    )
    assert observation.passed is True
    assert observation.reward == 1.0
    assert observation.runtime_identity["identity_hash"]

    registry = VerifierReliabilityAdapterRegistry()
    registry.register(adapter)
    assert registry.get("regex_format") is adapter
    assert registry.capabilities()[0]["adapter_id"] == "registered_verifier"

    judge_capability = (
        VerifierReliabilityAdapterRegistry.from_existing_registry()
        .get("llm_judge")
        .capability()
    )
    assert set(judge_capability.modalities) == {"text", "tool", "vlm", "audio"}
    assert set(judge_capability.tasks) == {"scalar", "pairwise", "ranking"}


def test_callable_adapter_keeps_invocation_errors_visible_and_fail_closed():
    def broken(_item):
        raise RuntimeError("provider unavailable")

    adapter = CallableReliabilityAdapter(
        "judge-v1",
        broken,
        family="llm_judge",
        implementation_fingerprint="sha256:pinned",
    )
    observation = adapter.invoke({}, contract=RewardContract())
    assert observation.reward is None
    assert observation.passed is False
    assert "provider unavailable" in observation.error


def test_artifact_reward_adapter_scores_scalar_pairwise_and_batch_requests(tmp_path):
    class FixtureRewardAdapter(ArtifactRewardModelReliabilityAdapter):
        def _score_many(self, texts, *, batch_size):
            assert batch_size in {1, 8}
            return [float(len(value)) / 10.0 for value in texts]

    scalar = FixtureRewardAdapter(
        key="rm-scalar",
        model_path=tmp_path,
        content_hash="sha256:reward",
        task_type="scalar",
    )
    scored = scalar.invoke(
        {"candidate": "answer", "batch_size": 8},
        contract=RewardContract(minimum=0.0, maximum=2.0, threshold=0.5),
    )
    assert scored.reward == pytest.approx(0.6)
    assert scored.passed is True
    assert scored.details["batch_size"] == 8

    pairwise = FixtureRewardAdapter(
        key="rm-pair",
        model_path=tmp_path,
        content_hash="sha256:reward",
        task_type="pairwise",
    )
    compared = pairwise.invoke(
        {"candidates": ["long answer", "short"]},
        contract=RewardContract(threshold=0.5, tie_policy="fail"),
    )
    assert compared.reward > 0.5
    assert compared.parsed_value == "long answer"


def test_scored_negative_verdict_is_evidence_not_runtime_error():
    observation = normalize_verifier_result(
        VerifyResult(success=False, reward=0.0, details="invalid", error="invalid_json"),
        contract=RewardContract(tie_policy="fail"),
    )
    assert observation.passed is False
    assert observation.reward == 0.0
    assert observation.error is None
    assert observation.details["legacy_verifier_error"] == "invalid_json"


def test_llm_judge_provider_and_parser_failures_remain_operational_errors():
    def unavailable(_prompt):
        raise RuntimeError("provider offline")

    provider = RegistryVerifierReliabilityAdapter(
        "llm_judge", configuration={"judge_callable": unavailable}
    ).invoke(
        {"prompt": "Question", "candidate": "Answer"},
        contract=RewardContract(tie_policy="fail"),
    )
    assert provider.reward is None
    assert provider.passed is False
    assert provider.error == "provider_error:judge_failure"

    parser = RegistryVerifierReliabilityAdapter(
        "llm_judge", configuration={"judge_callable": lambda _prompt: "not a score"}
    ).invoke(
        {"prompt": "Question", "candidate": "Answer"},
        contract=RewardContract(tie_policy="fail"),
    )
    assert parser.reward is None
    assert parser.error == "parse_error:unparseable_score"


def test_llm_judge_adapter_uses_existing_scorer_for_pairwise_and_ranking():
    scalar = RegistryVerifierReliabilityAdapter(
        "llm_judge",
        configuration={"judge_callable": lambda _prompt: "5"},
        family="llm_judge",
        tasks=("scalar",),
    ).invoke(
        {"prompt": "Score", "candidate": "answer"},
        contract=RewardContract(
            minimum=1.0,
            maximum=5.0,
            threshold=3.0,
            tie_policy="fail",
        ),
    )
    assert scalar.reward == 5.0
    assert scalar.parsed_value == 5

    judge = RegistryVerifierReliabilityAdapter(
        "llm_judge",
        configuration={
            "judge_callable": lambda prompt: "5" if "better" in prompt else "1"
        },
        family="llm_judge",
        tasks=("pairwise",),
    )
    pair = judge.invoke(
        {"prompt": "Choose", "candidates": ["better response", "worse response"]},
        contract=RewardContract(threshold=0.5, tie_policy="fail"),
    )
    assert pair.error is None
    assert pair.parsed_value == "better response"
    assert len(pair.component_trace) == 2


def test_chain_validation_rejects_cycles_and_unqualified_children():
    components = {
        "chain-a": [{"revision_id": "chain-b", "order_index": 0}],
        "chain-b": [{"revision_id": "chain-a", "order_index": 0}],
    }
    with pytest.raises(ValueError, match="cycle"):
        validate_chain_graph(
            "chain-a",
            components,
            qualification_by_revision={"chain-a": "candidate", "chain-b": "approved"},
        )
    with pytest.raises(ValueError, match="candidate-qualified"):
        validate_chain_graph(
            "chain-a",
            {"chain-a": [{"revision_id": "leaf", "order_index": 0}]},
            qualification_by_revision={"leaf": "warn"},
        )


def test_chain_aggregation_retains_component_error_and_veto_trace():
    components = [
        {"revision_id": "one", "order_index": 0, "weight": 2.0},
        {"revision_id": "two", "order_index": 1, "veto": True},
    ]
    failed = aggregate_chain_observations(
        components,
        {
            "one": VerifierObservation(reward=1.0, passed=True),
            "two": VerifierObservation(reward=None, passed=False, error="parser failed"),
        },
        aggregation="weighted_mean",
        contract=RewardContract(tie_policy="fail"),
    )
    assert failed.reward is None
    assert "parser failed" in failed.error
    assert len(failed.component_trace) == 2

    vetoed = aggregate_chain_observations(
        components,
        {
            "one": VerifierObservation(reward=1.0, passed=True),
            "two": VerifierObservation(reward=0.0, passed=False),
        },
        aggregation="weighted_mean",
        contract=RewardContract(threshold=0.2, tie_policy="fail"),
    )
    assert vetoed.reward == pytest.approx(2 / 3)
    assert vetoed.passed is False
    assert vetoed.details["veto_triggered"] is True


def test_grouped_partition_is_deterministic_and_prevents_identity_leakage():
    records = [
        {"record_id": "a", "content_hash": "same"},
        {"record_id": "b", "content_hash": "same"},
        {"record_id": "c", "media_hash": "media-1"},
        {"record_id": "d", "media_hashes": ["media-1"]},
        {"record_id": "e", "group_id": "g2"},
        {"record_id": "f", "related_group_id": "g2"},
        {"record_id": "g"},
        {"record_id": "h"},
    ]
    first = grouped_calibration_confirmation_partition(records)
    second = grouped_calibration_confirmation_partition(records)
    assert first == second
    assert not any(first.leakage.values())
    assert set(first.calibration_record_ids).isdisjoint(first.confirmation_record_ids)
    for linked in ({"a", "b"}, {"c", "d"}, {"e", "f"}):
        assert linked.issubset(first.calibration_record_ids) or linked.issubset(
            first.confirmation_record_ids
        )


def test_protocol_expands_required_repeats_orientations_and_reviewed_probes():
    pair = {
        "record_id": "pair-1",
        "candidates": ["A", "B"],
        "reviewed_variants": [
            {"kind": "style", "reviewed": True, "payload": {"prompt": "style probe"}},
            {"kind": "length", "reviewed": False, "payload": {}},
        ],
    }
    protocol = CalibrationProtocol(
        family="llm_judge",
        task_type="pairwise",
        deterministic=False,
        reviewed_probe_kinds=("style", "length"),
    )
    expanded = expand_calibration_protocol([pair], protocol)
    assert len(expanded) == 9  # three seeds x (A/B, B/A, one reviewed probe)
    assert {item.seed for item in expanded} == {17, 42, 101}
    assert all(item.fresh_process for item in expanded)
    assert all(item.generation_settings["concurrency"] == 1 for item in expanded)

    ranking = expand_calibration_protocol(
        [{"record_id": "rank", "candidates": ["a", "b", "c", "d", "e"]}],
        CalibrationProtocol(family="deterministic", task_type="ranking", deterministic=True),
    )
    assert len(ranking) == 8  # two fresh-process repeats x four orientations


def test_protocol_streaming_iterator_preserves_materialized_public_order():
    records = [
        {"record_id": f"pair-{index:05d}", "candidates": ["A", "B"]}
        for index in range(2_000, -1, -1)
    ]
    protocol = CalibrationProtocol(
        family="llm_judge", task_type="pairwise", deterministic=False
    )
    streamed = iter_calibration_protocol(records, protocol)
    assert iter(streamed) is streamed
    first = next(streamed)
    assert first.record_id == "pair-00000"

    # The compatibility API still provides the same payload/order to callers
    # that explicitly request a materialized tuple.
    small = records[-3:]
    assert tuple(iter_calibration_protocol(small, protocol)) == expand_calibration_protocol(
        small, protocol
    )


def test_reward_model_batch_protocol_and_dtype_tolerances():
    records = [{"record_id": f"r-{index}"} for index in range(40)]
    expanded = expand_calibration_protocol(
        records,
        CalibrationProtocol(
            family="reward_model",
            task_type="scalar",
            deterministic=True,
            production_batch_size=8,
            reward_model_dtype="bf16",
        ),
    )
    parity = [item for item in expanded if item.perturbation.startswith("batch_size") or item.perturbation == "production_batch_size"]
    assert len(parity) == 64
    assert dtype_score_tolerance("fp32") == 1e-6
    assert dtype_score_tolerance("bf16") == 1e-4
    assert dtype_score_tolerance("q4") == 1e-3
    assert reward_model_batch_consistent(0.5, 0.50009, dtype="bf16")
    assert not reward_model_batch_consistent(0.5, 0.5002, dtype="bf16")


def _perfect_binary_evidence(count: int = 100):
    rows = []
    for index in range(count):
        expected = index % 2 == 0
        reward = 1.0 if expected else 0.0
        for repetition in (0, 1):
            rows.append(
                CalibrationEvidence(
                    record_id=f"record-{index:03d}",
                    expected=expected,
                    predicted=expected,
                    reward=reward,
                    passed=expected,
                    repetition_index=repetition,
                    subgroup={"domain": "even" if index < count // 2 else "odd"},
                    latency_ms=10 + index % 3,
                )
            )
    return rows


def test_binary_metrics_repeats_bootstrap_subgroups_and_probability_semantics():
    rows = _perfect_binary_evidence()
    rows = [
        CalibrationEvidence(**{**row.to_dict(), "probability": row.reward}) for row in rows
    ]
    metrics = compute_calibration_metrics(
        rows,
        task_type="binary",
        reward_contract=RewardContract(tie_policy="fail", probability_semantics=True),
        bootstrap_resamples=200,
    )
    assert metrics["task"]["balanced_accuracy"] == 1.0
    assert metrics["task"]["mcc"] == 1.0
    assert metrics["universal"]["exact_repeat_agreement"] == 1.0
    assert metrics["primary_metric_interval"]["replicate_unit"] == "stable_record"
    assert metrics["probability"]["brier_score"] == 0.0
    assert metrics["subgroups"]["domain=even"]["available"] is True


def test_large_binary_bootstrap_compresses_stable_records_into_exact_categories():
    rows = [
        CalibrationEvidence(
            record_id=f"record-{index}",
            expected=bool(index % 2),
            predicted=bool((index + (index % 17 == 0)) % 2),
            reward=float(bool((index + (index % 17 == 0)) % 2)),
        )
        for index in range(50_000)
    ]
    metrics = compute_calibration_metrics(
        rows,
        task_type="binary",
        reward_contract=RewardContract(tie_policy="fail"),
        bootstrap_resamples=10_000,
        bootstrap_seed=42,
    )
    interval = metrics["primary_metric_interval"]
    assert interval["method"] == "compressed_multinomial_percentile"
    assert interval["exact"] is True
    assert interval["category_count"] <= 4
    assert interval["resamples"] == 10_000
    assert 0.0 <= interval["lower"] <= interval["upper"] <= 1.0


def test_large_continuous_scalar_bootstrap_is_bounded_and_truthfully_approximate():
    rows = [
        CalibrationEvidence(
            record_id=f"record-{index}",
            expected=float(index),
            predicted=float(index) + ((index % 13) - 6) * 0.25,
            reward=0.5,
        )
        for index in range(12_000)
    ]
    first = compute_calibration_metrics(
        rows,
        task_type="scalar",
        reward_contract=RewardContract(minimum=0.0, maximum=12_000.0, threshold=None),
        bootstrap_resamples=1_000,
        bootstrap_seed=42,
    )
    second = compute_calibration_metrics(
        rows,
        task_type="scalar",
        reward_contract=RewardContract(minimum=0.0, maximum=12_000.0, threshold=None),
        bootstrap_resamples=1_000,
        bootstrap_seed=42,
    )
    interval = first["primary_metric_interval"]
    assert interval == second["primary_metric_interval"]
    assert interval["method"] == "bag_of_little_bootstraps_percentile"
    assert interval["exact"] is False
    assert interval["replicate_unit"] == "stable_record"
    assert interval["subsample_size"] < len(rows)
    assert interval["resample_size"] == len(rows)
    assert first["task"]["record_count"] == len(rows)
    assert first["task"]["spearman"] > 0.99


def test_high_cardinality_subgroups_are_bounded_without_losing_eligible_group():
    rows = [
        CalibrationEvidence(
            record_id=f"record-{index}",
            expected=bool(index % 2),
            predicted=bool(index % 2),
            reward=float(bool(index % 2)),
            subgroup={
                "unique": f"value-{index}",
                **({"cohort": "reviewed"} if index < 25 else {}),
            },
        )
        for index in range(25_000)
    ]
    metrics = compute_calibration_metrics(
        rows,
        task_type="binary",
        reward_contract=RewardContract(tie_policy="fail"),
        bootstrap_resamples=32,
    )
    assert list(metrics["subgroups"]) == ["cohort=reviewed"]
    assert metrics["subgroups"]["cohort=reviewed"]["record_count"] == 25
    analysis = metrics["subgroup_analysis"]
    assert analysis["high_cardinality"] is True
    assert analysis["ineligible_groups_omitted"] is True
    assert analysis["tracked_candidate_count"] <= 10_000
    assert analysis["eligible_group_detection_exhaustive"] is True


def test_exact_repeat_agreement_includes_parsed_outputs_and_errors():
    rows = [
        CalibrationEvidence(
            record_id="parsed-drift",
            expected=True,
            predicted={"label": "accepted", "reason": "first"},
            reward=1.0,
            passed=True,
            repetition_index=0,
        ),
        CalibrationEvidence(
            record_id="parsed-drift",
            expected=True,
            predicted={"label": "accepted", "reason": "second"},
            reward=1.0,
            passed=True,
            repetition_index=1,
        ),
        CalibrationEvidence(
            record_id="error-drift",
            expected=False,
            reward=0.0,
            passed=False,
            error="parser returned no value",
            error_kind="parse",
            repetition_index=0,
        ),
        CalibrationEvidence(
            record_id="error-drift",
            expected=False,
            reward=0.0,
            passed=False,
            error="endpoint timed out",
            error_kind="timeout",
            timeout=True,
            repetition_index=1,
        ),
    ]
    metrics = compute_calibration_metrics(
        rows,
        task_type="binary",
        reward_contract=RewardContract(),
        bootstrap_resamples=16,
    )
    assert metrics["universal"]["repeat_agreement"] == 1.0
    assert metrics["universal"]["exact_repeat_agreement"] == 0.0


def test_all_task_specific_metric_families_report_without_fabrication():
    contract = RewardContract(threshold=None)
    categorical = compute_task_metrics(
        [
            {"record_id": "1", "expected": "a", "predicted": "a"},
            {"record_id": "2", "expected": "b", "predicted": "a"},
        ],
        task_type="categorical",
        reward_contract=contract,
    )
    assert categorical["accuracy"] == 0.5
    assert categorical["confusion_matrix"]["b"]["a"] == 1

    multi = compute_task_metrics(
        [
            {"record_id": "1", "expected": ["a", "b"], "predicted": ["a", "b"]},
            {"record_id": "2", "expected": ["b"], "predicted": ["a"]},
        ],
        task_type="multi_label",
        reward_contract=contract,
    )
    assert multi["exact_match"] == 0.5
    assert multi["hamming_loss"] == 0.5

    scalar = compute_task_metrics(
        [
            {"record_id": str(index), "expected": index / 3, "predicted": index / 3}
            for index in range(4)
        ],
        task_type="scalar",
        reward_contract=contract,
    )
    assert scalar["normalized_mae"] == 0.0
    assert scalar["spearman"] == pytest.approx(1.0)

    pairwise = compute_task_metrics(
        [
            {"record_id": "p", "expected": "winner-a", "predicted": "winner-a", "orientation": "a_b"},
            {"record_id": "p", "expected": "winner-a", "predicted": "winner-a", "orientation": "b_a"},
        ],
        task_type="pairwise",
        reward_contract=contract,
    )
    assert pairwise["tie_aware_accuracy"] == 1.0
    assert pairwise["order_consistency"] == 1.0

    ranking = compute_task_metrics(
        [{"record_id": "r", "expected": ["a", "b", "c"], "predicted": ["a", "b", "c"]}],
        task_type="ranking",
        reward_contract=contract,
    )
    assert ranking["kendall_tau"] == 1.0
    assert ranking["ndcg"] == 1.0
    assert ranking["implied_comparisons"] == 3


def test_grouped_bootstrap_uses_records_not_repeat_level_pseudoreplicates():
    rows = [
        CalibrationEvidence(record_id="zero", expected=False, predicted=False, reward=0.0, repetition_index=i)
        for i in range(5)
    ] + [
        CalibrationEvidence(record_id="one", expected=True, predicted=True, reward=1.0, repetition_index=i)
        for i in range(2)
    ]
    interval = grouped_percentile_bootstrap(
        rows,
        lambda sampled: sum(row.reward for row in sampled if row.primary) / len(
            [row for row in sampled if row.primary]
        ),
        resamples=400,
    )
    assert interval["replicate_unit"] == "stable_record"
    assert interval["lower"] == 0.0
    assert interval["upper"] == 1.0


def test_qualification_templates_minimum_evidence_and_promotion_rules():
    metrics = compute_calibration_metrics(
        _perfect_binary_evidence(),
        task_type="binary",
        reward_contract=RewardContract(tie_policy="fail"),
        bootstrap_resamples=100,
    )
    strict = qualify_calibration(
        metrics, task_type="binary", template="strict_oracle", scope="development"
    )
    assert strict.decision == "pass"
    assert strict.promotable is True

    stricter_profile = qualify_calibration(
        metrics,
        task_type="binary",
        template="strict_oracle",
        scope="development",
        requirements={"primary_agreement": {"pass": 1.01, "warn": 1.0}},
    )
    assert stricter_profile.decision == "warn"
    assert any("warn threshold" in reason for reason in stricter_profile.reasons)

    insufficient_metrics = {**metrics, "task": {**metrics["task"], "record_count": 50}}
    insufficient = qualify_calibration(
        insufficient_metrics,
        task_type="binary",
        template="human_aligned",
        scope="development",
    )
    assert insufficient.decision == "warn"
    assert any("100 records" in warning for warning in insufficient.warnings)

    exploratory = qualify_calibration(
        metrics, task_type="binary", template="exploratory", scope="development"
    )
    assert exploratory.decision == "warn"
    assert exploratory.promotable is False

    eligibility = promotion_eligibility(
        [
            strict,
            {**strict.to_dict(), "scope": "operational"},
            {**strict.to_dict(), "scope": "confirmation"},
        ],
        confirmation_required=True,
    )
    assert eligibility["candidate"] is True
    assert eligibility["approved"] is True
    without_confirmation = promotion_eligibility(
        [strict, {**strict.to_dict(), "scope": "operational"}],
        confirmation_required=True,
    )
    assert without_confirmation["candidate"] is True
    assert without_confirmation["approved"] is False
    assert without_confirmation["approved_missing_scopes"] == ["confirmation"]
    overridden = promotion_eligibility(
        [{**strict.to_dict(), "scope": "development", "override": True}],
        confirmation_required=False,
    )
    assert overridden["candidate"] is False


def test_threshold_curves_are_read_only_and_runtime_drift_is_explicit():
    curve = binary_threshold_curve(
        _perfect_binary_evidence(20),
        reward_contract=RewardContract(tie_policy="fail"),
        thresholds=[0.25, 0.75],
    )
    assert all(item["applied"] is False for item in curve)
    assert all(item["accuracy"] == 1.0 for item in curve)

    compatible = runtime_compatibility(
        {"python": "3.12", "model": {"revision": "abc"}},
        {"python": "3.12", "model": {"revision": "abc"}, "extra": "ignored"},
    )
    assert compatible["status"] == "compatible"
    stale = runtime_compatibility(
        {"python": "3.12", "tokenizer": "rev-a"},
        {"python": "3.13", "tokenizer": "rev-b"},
    )
    assert stale["status"] == "stale_runtime"
    assert {item["field"] for item in stale["mismatches"]} == {"python", "tokenizer"}


def test_calibration_comparison_is_direction_aware_and_sample_join_is_bounded():
    base = {
        "id": "base",
        "source_hash": "source",
        "protocol_hash": "protocol",
        "qualification_hash": "qualification",
        "task_type": "binary",
        "reward_contract_hash": "reward",
        "decision": "warn",
        "metrics": [
            {"name": "accuracy", "value": 0.8, "direction": "maximize"},
            {"name": "error_rate", "value": 0.1, "direction": "minimize"},
        ],
    }
    candidate = {
        **base,
        "id": "candidate",
        "decision": "pass",
        "metrics": [
            {"name": "accuracy", "value": 0.9, "direction": "maximize"},
            {"name": "error_rate", "value": 0.05, "direction": "minimize"},
        ],
    }
    comparison = compare_calibrations(base, candidate)
    assert comparison.compatible is True
    assert {item["classification"] for item in comparison.metric_deltas} == {"improved"}
    assert comparison.decision_delta["classification"] == "improved"

    samples = compare_calibration_samples(
        [{"record_id": "one", "observation": {"passed": False}}],
        [{"record_id": "one", "observation": {"passed": True}}],
        limit=10,
    )
    assert samples["total"] == 1
    assert samples["items"][0]["classification"] == "improved"

    incompatible = compare_calibrations(base, {**candidate, "source_hash": "other"})
    assert incompatible.compatible is False
    assert incompatible.metric_deltas == ()


def test_comparison_keeps_partitions_scopes_and_structured_sample_changes_distinct():
    shared = {
        "source_hash": "source",
        "protocol_hash": "protocol",
        "qualification_hash": "qualification",
        "task_type": "categorical",
        "reward_contract_hash": "reward",
    }
    base = {
        **shared,
        "id": "base",
        "metrics": [
            {
                "partition": "calibration",
                "name": "latency_ms_p95",
                "value": 12.0,
                "direction": "minimize",
            },
            {
                "partition": "confirmation",
                "name": "record_count",
                "value": 100,
                "direction": None,
            },
        ],
        "decisions": [
            {"scope": "development", "decision": "pass"},
            {"scope": "confirmation", "decision": "warn"},
        ],
    }
    candidate = {
        **shared,
        "id": "candidate",
        "metrics": [
            {
                "partition": "calibration",
                "name": "latency_ms_p95",
                "value": 9.0,
                "direction": "minimize",
            },
            {
                "partition": "confirmation",
                "name": "record_count",
                "value": 120,
                "direction": None,
            },
        ],
        "decisions": [
            {"scope": "development", "decision": "fail"},
            {"scope": "confirmation", "decision": "pass"},
        ],
    }
    result = compare_calibrations(base, candidate)
    by_name = {(item["partition"], item["name"]): item for item in result.metric_deltas}
    assert by_name[("calibration", "latency_ms_p95")]["classification"] == "improved"
    assert by_name[("confirmation", "record_count")]["classification"] == "descriptive_change"
    assert result.decision_delta == {
        "base": "warn",
        "candidate": "pass",
        "scope": "confirmation",
        "classification": "improved",
    }

    samples = compare_calibration_samples(
        [
            {
                "partition": "calibration",
                "record_id": "one",
                "reference": {"expected": {"label": "safe"}},
                "observation": {
                    "parsed_value": {"label": "safe"},
                    "reward": 0.8,
                    "passed": True,
                },
            }
        ],
        [
            {
                "partition": "calibration",
                "record_id": "one",
                "reference": {"expected": {"label": "safe"}},
                "observation": {
                    "parsed_value": {"label": "unsafe"},
                    "reward": 0.9,
                    "passed": True,
                },
            }
        ],
    )
    assert samples["items"][0]["classification"] == "regressed"
    assert samples["items"][0]["parsed_changed"] is True
    assert samples["items"][0]["reward_delta"] == pytest.approx(0.1)
