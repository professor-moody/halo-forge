from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import pytest

from halo_forge.run_db import get_database
from halo_forge.verifier_lab import VerifierLabService


def test_verifier_qualify_cli_advertises_only_supported_scopes():
    from halo_forge.verifier_cli import add_verifier_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_verifier_parser(subparsers)
    for scope in ("development", "operational", "confirmation"):
        parsed = parser.parse_args(
            ["verifier", "qualify", "calibration-1", "--scope", scope]
        )
        assert parsed.scope == scope
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["verifier", "qualify", "calibration-1", "--scope", "overall"]
        )


def _profile(tmp_path, monkeypatch, *, threshold=0.5, direction="maximize"):
    database = tmp_path / "runs.db"
    monkeypatch.setenv("HALOFORGE_RUN_DB_PATH", str(database))
    service = VerifierLabService(get_database(str(database)))
    created = service.create_profile(
        name="Pinned JSON verifier",
        description=None,
        definition={
            "family": "deterministic",
            "implementation": {"kind": "builtin", "ref": "json_structure"},
            "modality": "text",
            "task_type": "binary",
            "input_mapping": {},
            "reward_contract": {
                "minimum": 0.0,
                "maximum": 1.0,
                "direction": direction,
                "threshold": threshold,
                "tie_policy": "fail",
                "error_behavior": "fail_closed",
            },
            "runtime_requirements": {},
        },
    )
    return service, created["revision"]["id"]


def test_profile_runtime_bridge_preserves_exact_revision(tmp_path, monkeypatch):
    _service, revision_id = _profile(tmp_path, monkeypatch)
    from halo_forge.verifier_lab.runtime import ProfileRevisionVerifier

    verifier = ProfileRevisionVerifier(revision_id)
    accepted = verifier.verify(candidate='{"answer": 42}', prompt="Return JSON")
    rejected = verifier.verify(candidate="not json", prompt="Return JSON")

    assert accepted.success is True
    assert accepted.metadata["verifier_profile_revision_id"] == revision_id
    assert accepted.metadata["revision_hash"]
    assert rejected.success is False


def test_profile_runtime_can_be_registered_for_grpo(tmp_path, monkeypatch):
    _service, revision_id = _profile(tmp_path, monkeypatch)
    from halo_forge.rlvr.verifiers import get_verifier
    from halo_forge.verifier_lab.runtime import register_profile_verifier

    name = register_profile_verifier(revision_id)
    verifier = get_verifier(name)()
    result = verifier.verify_with_prompt('{"ok": true}', "Return JSON")

    assert name.startswith("profile_revision_")
    assert result.success is True
    assert result.metadata["verifier_profile_revision_id"] == revision_id


def test_cli_profile_rejects_contradictory_raw_threshold(tmp_path, monkeypatch):
    _service, revision_id = _profile(tmp_path, monkeypatch, threshold=0.75)
    from halo_forge.cli import _prepare_profile_verifier

    args = SimpleNamespace(
        verifier_profile_revision=revision_id,
        verifier="execution",
        reward_threshold=0.25,
        database=None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "grpo",
            "train",
            "--verifier-profile-revision",
            revision_id,
            "--reward-threshold",
            "0.25",
        ],
    )
    with pytest.raises(ValueError, match="reward contract controls"):
        _prepare_profile_verifier(args, consumer="registry", modality="text", training=True)


def test_profile_backed_score_file_uses_injected_scorer(tmp_path):
    from halo_forge.data.quality import QualityScore, score_file

    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    source.write_text('{"text":"keep"}\n{"text":"drop"}\n', encoding="utf-8")

    result = score_file(
        input_path=source,
        output_path=output,
        threshold=0.5,
        scorer=lambda row: QualityScore(
            score=1.0 if row["text"] == "keep" else 0.0,
            components={"verifier": 1.0 if row["text"] == "keep" else 0.0},
        ),
    )

    assert result.n_kept == 1
    assert json.loads(output.read_text(encoding="utf-8"))["text"] == "keep"


def test_profile_backed_legacy_data_output_records_exact_binding(tmp_path, monkeypatch):
    service, revision_id = _profile(tmp_path, monkeypatch)
    from halo_forge.cli import _bind_profile_verifier

    args = SimpleNamespace(
        verifier_profile_revision=revision_id,
        database=str(tmp_path / "runs.db"),
    )
    output = tmp_path / "scored.jsonl"
    _bind_profile_verifier(
        args,
        domain_kind="dataset_output",
        domain_id=str(output),
        role="quality_scoring",
    )

    bindings = service.store.list_bindings(
        revision_id=revision_id,
        domain_kind="dataset_output",
        domain_id=str(output),
    )
    assert len(bindings) == 1
    assert bindings[0].role == "quality_scoring"


def test_synthesis_accepts_exact_runtime_and_records_revision(tmp_path):
    from halo_forge.data.synthesize import synthesize_dataset
    from halo_forge.rlvr.verifiers.base import VerifyResult

    class Runtime:
        def verify(self, value=None, **_kwargs):
            return VerifyResult(True, 1.0, "accepted")

    output = tmp_path / "synthetic.jsonl"
    result = synthesize_dataset(
        seeds=["prompt"],
        output_path=output,
        teacher=lambda _prompt: "completion",
        verifier=Runtime(),
        verifier_name="profile",
        verifier_profile_revision_id="revision-1",
    )

    assert result.verifier_profile_revision_id == "revision-1"
    assert result.n_accepted == 1


def test_replay_requires_recorded_verifier_drift_override(tmp_path, monkeypatch):
    service, revision_id = _profile(tmp_path, monkeypatch)
    from halo_forge.cli import cmd_replay
    from halo_forge.replay import capture_manifest, save_manifest

    binding = service.resolve_binding(revision_id)
    binding["revision_hash"] = "drifted-captured-hash"
    manifest = capture_manifest(
        run_id="run-verifier",
        modality="grpo",
        model_name="model",
        seed=42,
        config={},
        verifier_binding=binding,
    )
    save_manifest(manifest, tmp_path)
    base = dict(
        source=str(tmp_path),
        launch=True,
        force=True,
        allow_dataset_drift=True,
        allow_verifier_drift=False,
        verifier_drift_reason=None,
    )
    with pytest.raises(SystemExit) as refused:
        cmd_replay(SimpleNamespace(**base))
    assert refused.value.code == 2

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    base.update(
        allow_verifier_drift=True,
        verifier_drift_reason="Reproduce historical result after reviewed runtime update",
    )
    with pytest.raises(SystemExit) as launched:
        cmd_replay(SimpleNamespace(**base))
    assert launched.value.code == 0
    event = json.loads(
        (tmp_path / "replay_overrides.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert event["event_type"] == "allow_verifier_drift"
    assert event["reason"].startswith("Reproduce historical")
