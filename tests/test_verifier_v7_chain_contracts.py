"""Regressions for chain isolation and immutable scalar references."""

from __future__ import annotations

import math

import pytest

from halo_forge.run_db import RunDatabase
from halo_forge.verifier_lab import VerifierLabService
from halo_forge.verifier_lab.process_runner import IsolatedVerifierPass


def _deterministic_profile(
    service: VerifierLabService, *, name: str, pattern: str
) -> dict:
    return service.create_profile(
        name=name,
        description=None,
        definition={
            "family": "deterministic",
            "implementation": {"ref": "regex_format"},
            "configuration": {"pattern": pattern, "full_match": True},
            "modality": "text",
            "task_type": "binary",
            "input_mapping": {"candidate": "output"},
            "reward_contract": {
                "minimum": 0.0,
                "maximum": 1.0,
                "threshold": 0.5,
                "tie_policy": "fail",
                "error_behavior": "fail_closed",
            },
            "runtime_contract": {"timeout_seconds": 10},
        },
    )


def _chain_revision(service: VerifierLabService, *children: str):
    profile = service.store.create_profile(name="Replicated deterministic chain")
    return service.store.create_profile_revision(
        profile.id,
        {
            "family": "chain",
            "implementation": {
                "kind": "chain",
                "ref": "ordered_chain",
                "fingerprint": "sha256:ordered-chain-v1",
                "pinned": True,
            },
            "reliability_adapter": {"id": "chain", "version": "1"},
            "modality": "text",
            "task_type": "binary",
            "input_mapping": {"candidate": "output"},
            "reward_contract": {
                "minimum": 0.0,
                "maximum": 1.0,
                "threshold": 0.5,
                "tie_policy": "fail",
                "error_behavior": "fail_closed",
            },
            "aggregation": "weighted_mean",
        },
        components=[{"child_revision_id": child} for child in children],
    )


def test_deterministic_chain_reuses_one_fresh_process_per_child_and_pass(tmp_path):
    db = RunDatabase(str(tmp_path / "chain-isolation.db"))
    service = VerifierLabService(db, root=tmp_path / "calibrations")
    try:
        first = _deterministic_profile(service, name="First regex", pattern=r"^ok$")
        second = _deterministic_profile(
            service, name="Second regex", pattern=r"^(ok|fine)$"
        )
        child_ids = {
            first["revision"]["id"],
            second["revision"]["id"],
        }
        chain = _chain_revision(service, *sorted(child_ids))
        protocol = service.create_protocol(
            name="Two fresh deterministic passes",
            description=None,
            # Chain determinism is inferred recursively from both children.
            definition={"bootstrap_resamples": 8},
        )
        qualification = service.create_qualification_profile(
            name="Exploratory chain evidence",
            description=None,
            template_kind="exploratory",
        )
        suite = db.create_benchmark_suite(
            name="Chain process reference", purpose="development"
        )
        source = db.create_benchmark_suite_revision(
            suite_id=suite.id,
            content_hash="chain-process-reference-v1",
            items=[
                {
                    "record_id": f"record-{index}",
                    "content_hash": f"record-hash-{index}",
                    "input": f"Prompt {index}",
                    "output": "ok",
                    "expected": True,
                }
                for index in range(3)
            ],
            primary_metric="balanced_accuracy",
            direction="maximize",
        )
        launched = service.launch_calibration(
            verifier_revision_id=chain.id,
            source_kind="benchmark_suite",
            source_revision_id=source.id,
            protocol_revision_id=protocol["revision"]["id"],
            qualification_profile_revision_id=qualification["revision"]["id"],
        )

        completed = service.run_calibration(launched["id"])
        samples = service.store.list_samples(completed.id, limit=100)

        assert completed.status == "completed"
        assert len(samples) == 6
        process_ids: dict[tuple[int, str], set[int]] = {}
        for sample in samples:
            assert sample.observation.error is None
            assert len(sample.observation.component_trace) == 2
            for component in sample.observation.component_trace:
                child_id = str(component["revision_id"])
                observation = component["observation"]
                process_ids.setdefault((sample.repeat_index, child_id), set()).add(
                    int(observation["runtime_identity"]["process_id"])
                )

        assert set(child for _, child in process_ids) == child_ids
        assert set(repeat for repeat, _ in process_ids) == {0, 1}
        assert all(len(values) == 1 for values in process_ids.values())
        # All four workers overlap for the duration of the calibration, so a
        # child cannot silently share process state with a sibling or pass.
        assert len({next(iter(values)) for values in process_ids.values()}) == 4
    finally:
        db.close()


def test_chain_isolation_keeps_component_error_evidence(tmp_path):
    db = RunDatabase(str(tmp_path / "chain-errors.db"))
    service = VerifierLabService(db, root=tmp_path / "calibrations")
    try:
        valid = _deterministic_profile(service, name="Valid regex", pattern=r"^ok$")
        invalid = _deterministic_profile(service, name="Invalid regex", pattern="[")
        valid_id = valid["revision"]["id"]
        invalid_id = invalid["revision"]["id"]
        chain = _chain_revision(service, valid_id, invalid_id)

        with IsolatedVerifierPass(timeout_seconds=10) as valid_pass, IsolatedVerifierPass(
            timeout_seconds=10
        ) as invalid_pass:
            observation = service._invoke_revision(
                chain,
                {"output": "ok"},
                runtime={"fresh_process_requested": True},
                isolated_pass={valid_id: valid_pass, invalid_id: invalid_pass},
            )

        assert observation.error is not None
        assert invalid_id in observation.error
        trace = {
            value["revision_id"]: value["observation"]
            for value in observation.component_trace
        }
        assert trace[valid_id]["error"] is None
        assert trace[valid_id]["runtime_identity"]["process_id"] == valid_pass.process_id
        assert trace[invalid_id]["error"]
        assert "unterminated character set" in trace[invalid_id]["error"].lower()
    finally:
        db.close()


def test_chain_protocol_inference_keeps_mixed_judge_chains_stochastic(tmp_path):
    db = RunDatabase(str(tmp_path / "chain-protocol.db"))
    service = VerifierLabService(db, root=tmp_path / "calibrations")
    try:
        deterministic = _deterministic_profile(
            service, name="Deterministic leaf", pattern=r"^ok$"
        )["revision"]["id"]
        judge_profile = service.store.create_profile(name="Stochastic judge leaf")
        judge = service.store.create_profile_revision(
            judge_profile.id,
            {
                "family": "llm_judge",
                "implementation": {
                    "kind": "builtin",
                    "ref": "llm_judge",
                    "fingerprint": "sha256:pinned-judge",
                    "pinned": True,
                },
                "reliability_adapter": {"id": "registered_verifier", "version": "1"},
                "modality": "text",
                "task_type": "binary",
                "reward_contract": {"minimum": 0.0, "maximum": 1.0},
            },
        )
        mixed = _chain_revision(service, deterministic, judge.id)

        protocol = service._protocol_for(mixed, {}, confirmation=False)

        assert protocol.deterministic is False
        assert protocol.stochastic_seeds == (17, 42, 101)
    finally:
        db.close()


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        (math.nan, "must be finite"),
        (math.inf, "must be finite"),
        (5.01, "outside the immutable reward contract"),
        (True, "must be numeric"),
    ],
)
def test_scalar_human_reference_must_satisfy_immutable_contract(
    tmp_path, expected, message
):
    db = RunDatabase(str(tmp_path / "scalar-contract.db"))
    service = VerifierLabService(db, root=tmp_path / "calibrations")
    try:
        profile = service.create_profile(
            name="Pinned scalar judge",
            description=None,
            definition={
                "family": "llm_judge",
                "implementation": {"ref": "llm_judge"},
                "model_revision": "fixture/judge@revision-1",
                "configuration": {"judge_model": "fixture/judge@revision-1"},
                "modality": "text",
                "task_type": "scalar",
                "input_mapping": {"candidate": "output", "prompt": "input"},
                "reward_contract": {
                    "minimum": 1.0,
                    "maximum": 5.0,
                    "threshold": 3.0,
                    "tie_policy": "fail",
                    "error_behavior": "fail_closed",
                },
            },
        )
        protocol = service.create_protocol(
            name="Scalar protocol",
            description=None,
            definition={"bootstrap_resamples": 8},
        )
        qualification = service.create_qualification_profile(
            name="Exploratory scalar evidence",
            description=None,
            template_kind="exploratory",
        )
        suite = db.create_benchmark_suite(
            name="Invalid scalar reference", purpose="development"
        )
        source = db.create_benchmark_suite_revision(
            suite_id=suite.id,
            content_hash=f"invalid-scalar-{message}",
            items=[
                {
                    "record_id": "scalar-record",
                    # Avoid deriving an identity from the deliberately invalid
                    # non-finite reference before the contract boundary sees it.
                    "content_hash": "scalar-record-content",
                    "input": "Rate this response",
                    "output": "A response",
                    "expected": expected,
                }
            ],
            primary_metric="spearman",
            direction="maximize",
        )
        launched = service.launch_calibration(
            verifier_revision_id=profile["revision"]["id"],
            source_kind="benchmark_suite",
            source_revision_id=source.id,
            protocol_revision_id=protocol["revision"]["id"],
            qualification_profile_revision_id=qualification["revision"]["id"],
        )

        with pytest.raises(ValueError, match=message):
            service.run_calibration(launched["id"])

        failed = service.store.get_calibration(launched["id"])
        assert failed.status == "failed"
        assert failed.sample_count == 0
    finally:
        db.close()
