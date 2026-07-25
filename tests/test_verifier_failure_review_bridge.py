"""Reviewed V7 verifier failures feed V6 acquisition without protected evidence."""

from __future__ import annotations

from pathlib import Path
from types import MethodType

import pytest

from halo_forge.public_api.service import PublicApiService
from halo_forge.review_lab import ReviewEligibilityError, ReviewLabService
from halo_forge.review_lab.acquisition import plan_acquisition
from halo_forge.run_db import RunDatabase
from halo_forge.verifier_lab import VerifierLabService, VerifierObservation
from halo_forge.verifier_lab.failure_mining import (
    normalize_failure_selector,
    select_calibration_failure,
)
from halo_forge.workstation_jobs import WorkstationScheduler

CONTRACT = {
    "minimum": 0.0,
    "maximum": 1.0,
    "threshold": 0.5,
    "direction": "maximize",
}


def _sample(
    *,
    expected=False,
    reward=1.0,
    passed=True,
    parsed_value=None,
    repeat=0,
    orientation="canonical",
    error=None,
    details=None,
    component_trace=None,
    subgroup=None,
):
    return {
        "ordinal": repeat,
        "record_id": "record-1",
        "partition": "calibration",
        "repeat_index": repeat,
        "orientation": orientation,
        "probe_kind": "canonical",
        "reference": {"expected": expected},
        "observation": {
            "reward": reward,
            "passed": passed,
            "parsed_value": parsed_value,
            "error": error,
            "details": dict(details or {}),
            "component_trace": list(component_trace or []),
        },
        "metadata": {"subgroup": dict(subgroup or {})},
    }


@pytest.mark.parametrize(
    ("selector", "task_type", "family", "samples", "options"),
    [
        ("false_accept", "binary", "deterministic", [_sample()], {}),
        (
            "false_reject",
            "binary",
            "deterministic",
            [_sample(expected=True, reward=0.0, passed=False)],
            {},
        ),
        (
            "high_confidence_disagreement",
            "binary",
            "llm_judge",
            [_sample(reward=0.95, passed=True)],
            {"margin": 0.4},
        ),
        (
            "repeat_instability",
            "binary",
            "llm_judge",
            [_sample(repeat=0), _sample(repeat=1, reward=0.0, passed=False)],
            {},
        ),
        (
            "order_flip",
            "pairwise",
            "llm_judge",
            [
                _sample(expected="a", parsed_value="a", orientation="a_b"),
                _sample(expected="a", parsed_value="b", orientation="b_a"),
            ],
            {},
        ),
        (
            "ranking_inversion",
            "ranking",
            "reward_model",
            [
                _sample(
                    expected=["a", "b", "c"],
                    parsed_value=["b", "a", "c"],
                )
            ],
            {},
        ),
        (
            "threshold_adjacent",
            "binary",
            "deterministic",
            [_sample(reward=0.51)],
            {"margin": 0.02},
        ),
        (
            "parser_runtime",
            "binary",
            "llm_judge",
            [_sample(reward=None, passed=None, error="parser failed")],
            {},
        ),
        (
            "subgroup",
            "binary",
            "deterministic",
            [_sample(subgroup={"language": "fr"})],
            {"subgroup": {"language": "fr"}},
        ),
        (
            "chain_component",
            "binary",
            "chain",
            [
                _sample(
                    component_trace=[
                        {
                            "child_revision_id": "first",
                            "observation": {"reward": 1.0, "passed": True},
                        },
                        {
                            "child_revision_id": "second",
                            "observation": {"reward": 0.0, "passed": False},
                        },
                    ]
                )
            ],
            {},
        ),
    ],
)
def test_all_verifier_failure_selectors_return_explicit_evidence(
    selector, task_type, family, samples, options
):
    selected = select_calibration_failure(
        selector=selector,
        options=options,
        samples=samples,
        task_type=task_type,
        verifier_family=family,
        reward_contract=CONTRACT,
    )
    assert selected is not None
    assert selected["selector"] == selector
    assert selected["selector_version"] == 1
    assert selected["observation_count"] == len(samples)


def test_failure_selector_rejects_nested_credentials():
    with pytest.raises(ValueError, match="cannot contain credentials"):
        normalize_failure_selector(
            {
                "kind": "threshold_adjacent",
                "options": {"provider": [{"access-token": "do-not-store"}]},
            }
        )


def _profile_definition():
    return {
        "family": "deterministic",
        "implementation": {"ref": "json_structure"},
        "modality": "text",
        "task_type": "binary",
        "input_mapping": {"candidate": "output", "reference": "expected"},
        "reward_contract": CONTRACT,
        "runtime_contract": {},
    }


def _always_accept(self, revision, item, *, runtime=None):
    return VerifierObservation(
        reward=1.0,
        passed=True,
        parsed_value=True,
        details={"fixture": "always-accept"},
        runtime_identity=dict(runtime or {}),
    )


def _completed_calibration(tmp_path: Path, *, protected_split: bool = False):
    db = RunDatabase(str(tmp_path / "runs.db"))
    scheduler = WorkstationScheduler(db)
    verifier = VerifierLabService(
        db,
        root=tmp_path / "calibrations",
        scheduler=scheduler,
    )
    profile = verifier.create_profile(
        name="Always accept reference",
        description=None,
        definition=_profile_definition(),
    )
    protocol = verifier.create_protocol(
        name="Failure mining protocol",
        description=None,
        definition={"bootstrap_resamples": 8},
    )
    qualification = verifier.create_qualification_profile(
        name="Failure mining policy",
        description=None,
        template_kind="exploratory",
    )
    suite = db.create_benchmark_suite(name="Development reference", purpose="development")
    rows = [
        {
            "id": f"record-{index:03d}",
            "record_id": f"record-{index:03d}",
            "input": f"Prompt {index}",
            "output": "rejected",
            "expected": False,
            "group_id": f"group-{index:03d}",
            **({"split": "test"} if protected_split and index == 0 else {}),
        }
        for index in range(10)
    ]
    source = db.create_benchmark_suite_revision(
        suite_id=suite.id,
        content_hash="review-source-protected" if protected_split else "review-source",
        items=rows,
        primary_metric="balanced_accuracy",
        direction="maximize",
    )
    verifier._invoke_revision = MethodType(_always_accept, verifier)
    launched = verifier.launch_calibration(
        verifier_revision_id=profile["revision"]["id"],
        source_kind="benchmark_suite",
        source_revision_id=source.id,
        protocol_revision_id=protocol["revision"]["id"],
        qualification_profile_revision_id=qualification["revision"]["id"],
        confirmation=not protected_split,
    )
    completed = verifier.run_calibration(launched["id"])
    return db, verifier, completed


def test_development_calibration_failures_create_review_candidates_without_confirmation(
    tmp_path: Path,
):
    db, verifier, calibration = _completed_calibration(tmp_path)
    try:
        api = PublicApiService(
            database=db,
            verifier_lab=verifier,
            workstation_scheduler=WorkstationScheduler(db),
            review_storage_root=tmp_path / "reviews",
        )
        payload = {
            "sources": [
                {
                    "kind": "verifier_calibration",
                    "ref": calibration.id,
                    "selector": "false_accept",
                }
            ]
        }
        records = list(api._review_acquisition_records(payload))
        confirmation_ids = set(calibration.partition["confirmation_record_ids"])
        assert records
        assert not confirmation_ids.intersection(value["record_id"] for value in records)
        assert all(value["source"]["partition"] == "calibration" for value in records)
        assert all(value["evidence"]["selector"] == "false_accept" for value in records)
        assert all(
            value["source"]["protocol_hash"] == calibration.protocol_hash for value in records
        )

        batch = ReviewLabService(db, tmp_path / "reviews").create_acquisition(records)
        assert batch.row_count == len(records)
        candidate = ReviewLabService(db, tmp_path / "reviews").list_acquisition_candidates(
            batch.id
        )[0]
        assert candidate.source_kind == "verifier_calibration"
        assert candidate.source["calibration_manifest_hash"] == calibration.manifest_hash

        with db._lock:
            db._conn.execute(
                "UPDATE verifier_calibrations SET source_purpose='operational' WHERE id=?",
                (calibration.id,),
            )
            db._conn.commit()
        with pytest.raises(ValueError, match="cannot guide review acquisition"):
            list(api._review_acquisition_records(payload))
    finally:
        db.close()


def test_test_split_verifier_evidence_is_refused(tmp_path: Path):
    db, verifier, calibration = _completed_calibration(tmp_path, protected_split=True)
    try:
        api = PublicApiService(database=db, verifier_lab=verifier)
        with pytest.raises(ValueError, match="protected verifier evidence"):
            list(
                api._review_acquisition_records(
                    {
                        "sources": [
                            {
                                "kind": "verifier_calibration",
                                "ref": calibration.id,
                                "selector": "false_accept",
                            }
                        ]
                    }
                )
            )
    finally:
        db.close()


def test_raw_confirmation_calibration_evidence_cannot_bypass_the_bridge():
    with pytest.raises(ReviewEligibilityError, match="protected_verifier_partition"):
        plan_acquisition(
            [
                {
                    "record_id": "confirmation-only",
                    "record": {"input": "held evidence"},
                    "source": {
                        "kind": "verifier_calibration",
                        "ref": "calibration-id",
                        "purpose": "development",
                        "partition": "confirmation",
                    },
                }
            ]
        )
