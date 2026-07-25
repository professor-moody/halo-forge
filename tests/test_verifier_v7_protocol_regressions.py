"""Focused regressions for v7 protocol fidelity and evidence preservation."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest

from halo_forge.run_db import RunDatabase
from halo_forge.verifier_lab import VerifierLabService, VerifierObservation
from halo_forge.verifier_lab.adapters import (
    ArtifactRewardModelReliabilityAdapter,
    CallableReliabilityAdapter,
    RegistryVerifierReliabilityAdapter,
    VerifierReliabilityAdapterRegistry,
)
from halo_forge.verifier_lab.models import VerifierRewardContract
from halo_forge.verifier_lab.observation import RewardContract
from halo_forge.verifier_lab.process_runner import IsolatedVerifierPass
from halo_forge.verifier_lab.protocol import ProtocolInvocation


def _request(candidate: str) -> dict:
    return {
        "implementation_ref": "regex_format",
        "configuration": {"pattern": r"^ok$", "full_match": True},
        "modality": "text",
        "task_type": "binary",
        "reward_contract": {
            "minimum": 0.0,
            "maximum": 1.0,
            "threshold": 0.5,
            "tie_policy": "fail",
            "error_behavior": "fail_closed",
        },
        "item": {"candidate": candidate},
    }


def test_isolated_verifier_pass_reuses_one_pid_and_two_passes_are_distinct():
    with IsolatedVerifierPass(timeout_seconds=10) as first, IsolatedVerifierPass(
        timeout_seconds=10
    ) as second:
        first_one = first.invoke(_request("ok"))
        first_two = first.invoke(_request("no"))
        second_one = second.invoke(_request("ok"))

        assert first_one["runtime_identity"]["process_id"] == first.process_id
        assert first_two["runtime_identity"]["process_id"] == first.process_id
        assert second_one["runtime_identity"]["process_id"] == second.process_id
        assert first.process_id != second.process_id
        assert first_one["passed"] is True
        assert first_two["passed"] is False


def test_reward_model_batch_probe_uses_distinct_records_in_one_tensor_batch(tmp_path):
    calls: list[tuple[list[str], int]] = []

    class FixtureRewardModel(ArtifactRewardModelReliabilityAdapter):
        def _score_many(self, texts, *, batch_size):
            calls.append((list(texts), int(batch_size)))
            return [float(index + 1) for index, _ in enumerate(texts)]

    adapter = FixtureRewardModel(
        key="fixture-rm",
        model_path=tmp_path,
        content_hash="sha256:fixture-rm",
        task_type="scalar",
    )
    observations = adapter.invoke_batch(
        [{"record_id": "one", "candidate": "alpha"}, {"record_id": "two", "candidate": "bravo"}],
        contract=RewardContract(minimum=0.0, maximum=3.0, threshold=0.5),
        batch_size=8,
    )

    assert calls == [(["alpha", "bravo"], 8)]
    assert [value.reward for value in observations] == [1.0, 2.0]
    assert all(value.details["true_batch_record_count"] == 2 for value in observations)


def test_unsupported_reward_adapter_produces_no_batch_evidence_coverage(tmp_path):
    db = RunDatabase(str(tmp_path / "unsupported-batch.db"))
    registry = VerifierReliabilityAdapterRegistry()
    registry.register(
        CallableReliabilityAdapter(
            "unsupported-rm",
            lambda item: {"reward": 0.5, "passed": True},
            family="reward_model",
            implementation_fingerprint="sha256:unsupported-rm",
            tasks=("scalar",),
        )
    )
    service = VerifierLabService(
        db,
        root=tmp_path / "calibrations",
        adapter_registry=registry,
    )
    try:
        revision = SimpleNamespace(
            definition={"implementation": {"kind": "runtime"}},
            implementation_ref="unsupported-rm",
            reliability_adapter_version="1",
            reward_contract=VerifierRewardContract(tie_policy="fail"),
        )
        invocations = [
            ProtocolInvocation(
                invocation_id=f"invocation-{index}",
                record_id=f"record-{index}",
                repetition_index=0,
                seed=None,
                orientation="canonical",
                perturbation="production_batch_size",
                payload={"candidate": f"response-{index}"},
            )
            for index in range(2)
        ]
        results = service._reward_model_production_observations(
            revision,
            invocations,
            batch_size=8,
            runtime={},
        )
        samples = []
        for index, invocation in enumerate(invocations):
            samples.extend(
                [
                    SimpleNamespace(
                        record_id=invocation.record_id,
                        probe_kind="batch_size_one",
                        observation=SimpleNamespace(reward=0.5),
                    ),
                    SimpleNamespace(
                        record_id=invocation.record_id,
                        probe_kind="production_batch_size",
                        observation=results[invocation.invocation_id],
                    ),
                ]
            )

        assert all(value.reward is None and value.error for value in results.values())
        metrics = service._reward_model_batch_metrics(samples, dtype="fp32")
        assert metrics["reward_model_batch_evidence_coverage"] == 0.0
        assert metrics["reward_model_batch_record_count"] == 0.0
    finally:
        db.close()


def _llm_profile_definition(*, model_revision: str | None) -> dict:
    definition = {
        "family": "llm_judge",
        "implementation": {"ref": "llm_judge"},
        "modality": "text",
        "task_type": "scalar",
        "input_mapping": {"candidate": "output", "prompt": "input"},
        "reward_contract": {
            "minimum": 0.0,
            "maximum": 1.0,
            "threshold": 0.5,
            "tie_policy": "fail",
            "error_behavior": "fail_closed",
        },
        "runtime_contract": {},
    }
    if model_revision is not None:
        definition["model_revision"] = model_revision
    return definition


def test_llm_profiles_require_pinned_model_and_observation_keeps_judge_evidence(tmp_path):
    db = RunDatabase(str(tmp_path / "llm-profile.db"))
    service = VerifierLabService(db, root=tmp_path / "calibrations")
    try:
        with pytest.raises(ValueError, match="pinned model"):
            service.validate_profile_definition(_llm_profile_definition(model_revision=None))

        resolved = service.validate_profile_definition(
            _llm_profile_definition(model_revision="judge/model@revision-abc")
        )["resolved_definition"]
        assert resolved["model_revision"] == "judge/model@revision-abc"
        assert resolved["configuration"]["judge_model"] == "judge/model@revision-abc"

        observation = RegistryVerifierReliabilityAdapter(
            "llm_judge",
            configuration={
                "judge_callable": lambda _prompt: "Score: 4",
                "judge_model": "judge/model@revision-abc",
            },
            family="llm_judge",
            tasks=("scalar",),
        ).invoke(
            {"prompt": "Question", "candidate": "Answer"},
            contract=RewardContract(threshold=0.5, tie_policy="fail"),
        )
        assert observation.error is None
        assert observation.raw_output == "Score: 4"
        assert observation.parsed_value == 4
        assert observation.reward == pytest.approx(0.75)
    finally:
        db.close()


def test_calibration_sample_retains_scrubbed_multimodal_source_and_string_details(tmp_path):
    db = RunDatabase(str(tmp_path / "source-evidence.db"))
    service = VerifierLabService(db, root=tmp_path / "calibrations")
    try:
        profile = service.create_profile(
            name="Pinned evidence judge",
            description=None,
            definition=_llm_profile_definition(model_revision="judge/model@revision-abc"),
        )
        protocol = service.create_protocol(
            name="Replicated judge protocol",
            description=None,
            definition={"bootstrap_resamples": 8},
        )
        qualification = service.create_qualification_profile(
            name="Exploratory evidence policy",
            description=None,
            template_kind="exploratory",
        )
        suite = db.create_benchmark_suite(name="Multimodal evidence", purpose="development")
        source = db.create_benchmark_suite_revision(
            suite_id=suite.id,
            content_hash="multimodal-evidence-v1",
            items=[
                {
                    "record_id": f"record-{index}",
                    "input": f"Question {index}",
                    "output": f"Answer {index}",
                    "expected": 0.75,
                    "candidates": [f"Answer {index}", f"Alternative {index}"],
                    "image": {"path": f"image-{index}.png", "width": 64},
                    "audio": {"path": f"audio-{index}.wav", "duration": 1.5},
                    "provider": {
                        "api_key": "must-not-persist",
                        "endpoint": "https://user:password@example.test/v1?api_key=hidden&mode=judge",
                    },
                }
                for index in range(2)
            ],
            primary_metric="spearman",
            direction="maximize",
        )

        def invoke(self, revision, item, *, runtime=None):
            return VerifierObservation(
                reward=0.75,
                passed=True,
                parsed_value=0.75,
                raw_output="Score: 4",
                details="provider returned a scalar score",  # historical shape
                runtime_identity=dict(runtime or {}),
            )

        service._invoke_revision = MethodType(invoke, service)
        launched = service.launch_calibration(
            verifier_revision_id=profile["revision"]["id"],
            source_kind="benchmark_suite",
            source_revision_id=source.id,
            protocol_revision_id=protocol["revision"]["id"],
            qualification_profile_revision_id=qualification["revision"]["id"],
        )
        completed = service.run_calibration(launched["id"])
        assert completed.status == "completed"

        sample = service.store.list_samples(completed.id, limit=1)[0]
        retained = sample.reference["input"]
        assert retained["input"] == "Question 0"
        assert retained["candidates"] == ["Answer 0", "Alternative 0"]
        assert retained["image"]["path"] == "image-0.png"
        assert retained["audio"]["path"] == "audio-0.wav"
        assert "api_key" not in retained["provider"]
        assert "password" not in retained["provider"]["endpoint"]
        assert "hidden" not in retained["provider"]["endpoint"]
        assert sample.observation.details == {
            "message": "provider returned a scalar score"
        }
        assert sample.metadata["seed_honored"] is None
    finally:
        db.close()
