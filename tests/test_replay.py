"""Replay manifest tests (Track T15)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------- environment fingerprint ----------------------------------------


def test_environment_fingerprint_captures_python_and_platform():
    from halo_forge.replay import EnvironmentFingerprint

    fp = EnvironmentFingerprint.capture()
    assert fp.python  # e.g. "3.13.12"
    assert fp.platform  # e.g. "Darwin arm64"
    assert isinstance(fp.packages, dict)
    # halo_forge itself should always be in the tracked set.
    assert "halo_forge" in fp.packages


# ---------- dataset hashing ------------------------------------------------


def test_hash_dataset_file_is_stable_and_content_addressed(tmp_path: Path):
    from halo_forge.replay import hash_dataset_file

    p = tmp_path / "data.jsonl"
    p.write_text('{"text": "hello"}\n{"text": "world"}\n')
    h1 = hash_dataset_file(p)
    h2 = hash_dataset_file(p)
    assert h1 == h2

    p.write_text('{"text": "different"}\n')
    h3 = hash_dataset_file(p)
    assert h3 != h1


# ---------- capture + save + load ------------------------------------------


@dataclass
class _StubConfig:
    model_name: str = "Qwen/Qwen2.5-3B"
    learning_rate: float = 5e-6
    num_epochs: int = 1


def test_capture_manifest_with_dataclass_config():
    from halo_forge.replay import capture_manifest

    m = capture_manifest(
        run_id="run_a",
        modality="dpo",
        model_name="Qwen/Qwen2.5-3B",
        seed=42,
        config=_StubConfig(),
        dataset_id="HuggingFaceH4/ultrafeedback_binarized",
        cli_args=["--beta", "0.1"],
    )
    assert m.run_id == "run_a"
    assert m.seed == 42
    assert m.config["learning_rate"] == 5e-6
    assert m.dataset["kind"] == "huggingface"
    assert m.dataset["id"] == "HuggingFaceH4/ultrafeedback_binarized"
    assert m.environment["python"]


def test_capture_manifest_with_local_dataset_hashes_it(tmp_path: Path):
    from halo_forge.replay import capture_manifest

    src = tmp_path / "train.jsonl"
    src.write_text('{"text": "hello"}\n')
    m = capture_manifest(
        run_id="r",
        modality="sft",
        model_name="x",
        seed=42,
        config={"learning_rate": 0.001},
        dataset_file=src,
    )
    assert m.dataset["kind"] == "local_file"
    assert m.dataset["sha256"]
    assert m.dataset["size_bytes"] == src.stat().st_size


def test_capture_manifest_with_managed_dataset_version():
    from halo_forge.replay import MANIFEST_VERSION, capture_manifest

    m = capture_manifest(
        run_id="managed",
        modality="vlm",
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        seed=42,
        config={"cycles": 1},
        dataset_version={
            "version_id": "dsv_abc123",
            "dataset_id": "ds_images",
            "content_hash": "abc123",
            "recipe_hash": "def456",
            "split": "train",
            "source_fingerprints": [{"source_id": "src_1", "sha256": "123"}],
            "assets_materialized": False,
        },
    )

    assert m.manifest_version == MANIFEST_VERSION == 14
    assert m.dataset["kind"] == "managed_version"
    assert m.dataset["version_id"] == "dsv_abc123"
    assert m.dataset["split"] == "train"
    assert m.dataset["assets_materialized"] is False


def test_capture_manifest_managed_version_requires_identity():
    from halo_forge.replay import capture_manifest

    with pytest.raises(ValueError, match="version_id"):
        capture_manifest(
            run_id="managed",
            modality="sft",
            model_name="x",
            seed=42,
            config={},
            dataset_version={"content_hash": "abc123"},
        )


def test_capture_manifest_with_role_aware_managed_bindings():
    from halo_forge.replay import capture_manifest

    artifact = {
        "artifact_id": "artifact-1",
        "artifact_hash": "artifact-hash",
        "adapter_id": "preference",
        "adapter_version": "2",
        "tokenizer_revision": "tok-rev",
        "chat_template_hash": "chat-hash",
        "validation_policy": {"kind": "supplied", "preserved": True},
    }
    manifest = capture_manifest(
        run_id="managed-run",
        modality="dpo",
        model_name="model",
        seed=7,
        config={},
        dataset_bindings=[
            {
                "role": "train",
                "dataset_version_id": "version-a",
                "split": "train",
                "content_hash": "content-a",
            },
            {
                "role": "validation",
                "dataset_version_id": "version-b",
                "split": "validation",
                "content_hash": "content-b",
            },
        ],
        training_artifact=artifact,
    )

    assert manifest.dataset["kind"] == "managed_versions"
    assert [item["role"] for item in manifest.dataset["bindings"]] == [
        "train",
        "validation",
    ]
    assert manifest.dataset["training_artifact"] == artifact


def test_capture_manifest_v4_reward_integrity_identity():
    from halo_forge.replay import capture_manifest, compare_reward_identities

    binding = {
        "reward_system_revision_id": "reward-system-r1",
        "reward_system_hash": "system-hash",
        "optimizer_verifier_revision_id": "optimizer-r1",
        "auditors": [
            {
                "role": "primary_sentinel",
                "verifier_revision_id": "sentinel-r2",
                "revision_hash": "sentinel-hash",
            }
        ],
        "reward_mapping_hash": "mapping-hash",
        "protocol_revision_id": "balanced-256-r1",
        "protocol_hash": "protocol-hash",
        "integrity_profile_revision_id": "human-aligned-r1",
        "integrity_profile_hash": "profile-hash",
        "boundaries": [100, 200],
        "signal_capability": {"id": "hf-grpo-step-v1", "fidelity": "sampled"},
        "trace_manifests": [{"boundary": 100, "content_hash": "trace-hash"}],
        "runtime_compatibility": {"state": "compatible"},
    }
    manifest = capture_manifest(
        run_id="reward-run",
        modality="grpo",
        model_name="model",
        seed=42,
        config={},
        reward_integrity_binding=binding,
    )

    assert manifest.manifest_version == 14
    assert manifest.reward_integrity["reward_system_revision_id"] == "reward-system-r1"
    assert manifest.reward_integrity["auditors"][0]["verifier_revision_id"] == "sentinel-r2"
    assert compare_reward_identities(manifest.reward_integrity, manifest.reward_integrity)[
        "matched"
    ]


@pytest.mark.parametrize(
    ("blockers", "expected_state"),
    [
        (["primary_sentinel:stale_runtime"], "stale_runtime"),
        (["optimizer:verifier_not_candidate_or_approved"], "ineligible"),
    ],
)
def test_current_reward_identity_surfaces_runtime_and_eligibility_blockers(
    monkeypatch, blockers, expected_state
):
    import halo_forge.reward_integrity as reward_integrity
    from halo_forge.cli import _resolve_current_reward_identity
    from halo_forge.replay import compare_reward_identities

    captured = {
        "reward_system_revision_id": "reward-system-r1",
        "reward_system_hash": "system-hash",
        "optimizer_verifier_revision_id": "optimizer-r1",
        "auditors": [],
        "reward_mapping_hash": "system-hash",
        "protocol_revision_id": "protocol-r1",
        "protocol_hash": "protocol-hash",
        "integrity_profile_revision_id": "profile-r1",
        "integrity_profile_hash": "profile-hash",
        "boundaries": [],
        "signal_capability": {},
        "trace_manifests": [],
        "audit_decisions": [],
        "runtime_compatibility": {"state": "compatible"},
    }

    class FakeResolved:
        @staticmethod
        def to_dict():
            return {
                "reward_system_revision": {
                    "id": "reward-system-r1",
                    "content_hash": "system-hash",
                    "optimizer_verifier_revision_id": "optimizer-r1",
                    "auditors": [],
                },
                "protocol_revision": {
                    "id": "protocol-r1",
                    "content_hash": "protocol-hash",
                },
                "integrity_profile_revision": {
                    "id": "profile-r1",
                    "content_hash": "profile-hash",
                },
                "gating_eligible": False,
                "blockers": blockers,
            }

    class FakeService:
        @staticmethod
        def resolve_binding(*_args, **_kwargs):
            return FakeResolved()

    monkeypatch.setattr(
        reward_integrity, "RewardIntegrityService", lambda _database: FakeService()
    )
    current = _resolve_current_reward_identity(captured)
    assert current["runtime_compatibility"] == {
        "state": expected_state,
        "compatible": False,
        "blockers": blockers,
    }
    comparison = compare_reward_identities(captured, current)
    assert comparison["matched"] is False
    assert any(
        item["key"] == "runtime_compatibility"
        for item in comparison["differences"]
    )


def test_current_reward_identity_resolves_signal_capability_from_registry(monkeypatch):
    import halo_forge.reward_integrity as reward_integrity
    from halo_forge.cli import _resolve_current_reward_identity
    from halo_forge.replay import compare_reward_identities
    from halo_forge.training_signal import TRAINING_SIGNAL_CAPABILITIES

    registered = TRAINING_SIGNAL_CAPABILITIES.get("grpo:hf").to_dict()
    captured_capability = {**registered, "version": "stale-version"}
    captured = {
        "reward_system_revision_id": "reward-system-r1",
        "reward_system_hash": "system-hash",
        "optimizer_verifier_revision_id": "optimizer-r1",
        "auditors": [],
        "reward_mapping_hash": "system-hash",
        "protocol_revision_id": "protocol-r1",
        "protocol_hash": "protocol-hash",
        "integrity_profile_revision_id": "profile-r1",
        "integrity_profile_hash": "profile-hash",
        "boundaries": [],
        "signal_capability": captured_capability,
        "trace_manifests": [],
        "audit_decisions": [],
        "runtime_compatibility": {"state": "compatible"},
    }

    class FakeResolved:
        @staticmethod
        def to_dict():
            return {
                "reward_system_revision": {
                    "id": "reward-system-r1",
                    "content_hash": "system-hash",
                    "optimizer_verifier_revision_id": "optimizer-r1",
                    "auditors": [],
                },
                "protocol_revision": {
                    "id": "protocol-r1",
                    "content_hash": "protocol-hash",
                },
                "integrity_profile_revision": {
                    "id": "profile-r1",
                    "content_hash": "profile-hash",
                },
                "gating_eligible": True,
                "blockers": [],
            }

    class FakeService:
        @staticmethod
        def resolve_binding(*_args, **_kwargs):
            return FakeResolved()

    monkeypatch.setattr(
        reward_integrity, "RewardIntegrityService", lambda _database: FakeService()
    )
    current = _resolve_current_reward_identity(captured)
    assert current["signal_capability"] == registered
    assert current["signal_capability"] != captured_capability
    comparison = compare_reward_identities(captured, current)
    assert comparison["matched"] is False
    assert any(item["key"] == "signal_capability" for item in comparison["differences"])


def test_load_v3_manifest_defaults_reward_integrity_to_empty(tmp_path: Path):
    from halo_forge.replay import load_manifest

    path = tmp_path / "replay.json"
    path.write_text(
        json.dumps(
            {
                "manifest_version": 3,
                "run_id": "legacy-v3",
                "modality": "raft",
                "seed": 42,
                "config": {},
                "dataset": {},
                "environment": {},
                "verifier": {"verifier_profile_revision_id": "verifier-r1"},
            }
        )
    )

    manifest = load_manifest(path)
    assert manifest.reward_integrity == {}


def test_direct_cli_managed_replay_captures_complete_identity(tmp_path: Path):
    from halo_forge.cli import _finalize_managed_training_replay, _managed_replay_identity
    from halo_forge.replay import load_manifest

    version_dir = tmp_path / "datasets" / "dataset-a" / "version-a"
    version_dir.mkdir(parents=True)
    (version_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_fingerprint": "source-hash",
                "asset_fingerprints": [{"reference": "image.png", "fingerprint": "asset-hash"}],
                "materialized_assets": False,
            }
        )
    )
    version = SimpleNamespace(
        dataset_id="dataset-a",
        version_id="version-a",
        path=str(version_dir),
        content_hash="content-hash",
        recipe_hash="recipe-hash",
        source_fingerprint="source-hash",
        materialized_assets=False,
    )
    store = SimpleNamespace(get_any=lambda *_args, **_kwargs: version)
    lab = SimpleNamespace(store=store)
    db = SimpleNamespace(get_dataset_version=lambda _version_id: None)
    binding = SimpleNamespace(
        role="train",
        dataset_id="dataset-a",
        dataset_version_id="version-a",
        split="train",
    )
    artifact = SimpleNamespace(
        artifact_id="artifact-1",
        artifact_hash="artifact-hash",
        adapter_id="sft",
        adapter_version="3",
        trainer_mode="sft",
        model="model-a",
        tokenizer_revision="tokenizer-rev",
        chat_template_hash="chat-template-hash",
        validation_policy={
            "kind": "derived",
            "seed": 13,
            "selection": "sha256(seed,instance_id)",
        },
        token_statistics={"exact": True, "overall": {"total": 123}},
        row_counts={"train": 9, "validation": 1},
        split_paths={"train": "/artifact/train.jsonl"},
        asset_roots=(),
        bindings=(binding,),
        resolved_bindings=(
            {
                "role": "train",
                "dataset_version_id": "version-a",
                "split": "train",
                "row_count": 10,
                "content_fingerprint": "split-hash",
                "exposed_to_trainer": True,
            },
        ),
    )
    managed = _managed_replay_identity(lab, artifact, db)
    managed["run_id"] = "canonical-run"
    args = SimpleNamespace(
        model="model-a",
        seed=13,
        _managed_dataset_replay=managed,
    )
    config = {"model_name": "model-a", "output_dir": str(tmp_path / "run")}

    path = _finalize_managed_training_replay(
        args,
        "sft",
        tmp_path / "run",
        config,
        {"run_id": "canonical-run", "seed": 13},
    )
    assert path == tmp_path / "run" / "replay.json"
    replay = load_manifest(path)
    assert replay.run_id == "canonical-run"
    replay_binding = replay.dataset["bindings"][0]
    assert replay_binding == {
        "role": "train",
        "dataset_id": "dataset-a",
        "dataset_version_id": "version-a",
        "split": "train",
        "content_hash": "content-hash",
        "recipe_hash": "recipe-hash",
        "source_fingerprints": {
            "source": "source-hash",
            "assets": [{"reference": "image.png", "fingerprint": "asset-hash"}],
        },
        "assets_materialized": False,
        "asset_materialization_state": "referenced",
        "row_count": 10,
        "split_content_hash": "split-hash",
        "exposed_to_trainer": True,
    }
    replay_artifact = replay.dataset["training_artifact"]
    assert replay_artifact["artifact_hash"] == "artifact-hash"
    assert replay_artifact["adapter_id"] == "sft"
    assert replay_artifact["adapter_version"] == "3"
    assert replay_artifact["tokenizer_revision"] == "tokenizer-rev"
    assert replay_artifact["chat_template_hash"] == "chat-template-hash"
    assert replay_artifact["validation_policy"]["kind"] == "derived"


def test_direct_cli_manual_dataset_does_not_create_managed_replay(tmp_path: Path):
    from halo_forge.cli import _finalize_managed_training_replay

    result = _finalize_managed_training_replay(
        SimpleNamespace(model="model", seed=42),
        "sft",
        tmp_path,
        {"model_name": "model"},
        {"run_id": "manual-run"},
    )
    assert result is None
    assert not (tmp_path / "replay.json").exists()


def test_capture_manifest_with_dict_config():
    from halo_forge.replay import capture_manifest

    m = capture_manifest(
        run_id="r",
        modality="grpo",
        model_name="x",
        seed=42,
        config={"beta": 0.04, "num_generations": 4},
    )
    assert m.config == {"beta": 0.04, "num_generations": 4}


def test_save_and_load_roundtrip(tmp_path: Path):
    from halo_forge.replay import capture_manifest, load_manifest, save_manifest

    m = capture_manifest(
        run_id="r",
        modality="sft",
        model_name="x",
        seed=99,
        config={"learning_rate": 0.001, "num_epochs": 2},
    )
    save_manifest(m, tmp_path)

    loaded = load_manifest(tmp_path)
    assert loaded.run_id == m.run_id
    assert loaded.seed == 99
    assert loaded.config == {"learning_rate": 0.001, "num_epochs": 2}
    assert loaded.environment["python"] == m.environment["python"]


def test_load_manifest_accepts_file_or_dir(tmp_path: Path):
    from halo_forge.replay import (
        capture_manifest,
        load_manifest,
        MANIFEST_FILENAME,
        save_manifest,
    )

    m = capture_manifest(
        run_id="r",
        modality="sft",
        model_name="x",
        seed=1,
        config={},
    )
    save_manifest(m, tmp_path)

    via_dir = load_manifest(tmp_path)
    via_file = load_manifest(tmp_path / MANIFEST_FILENAME)
    assert via_dir.run_id == via_file.run_id


def test_load_manifest_missing_file(tmp_path: Path):
    from halo_forge.replay import load_manifest

    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "nope")


def test_load_manifest_tolerates_unknown_version(tmp_path: Path, caplog):
    """Forward-compat: a future-version manifest still loads with a warning."""
    from halo_forge.replay import MANIFEST_FILENAME, load_manifest

    payload = {
        "manifest_version": 99,
        "run_id": "r",
        "modality": "sft",
        "model_name": "x",
        "seed": 5,
        "config": {},
        "dataset": {},
        "environment": {},
        "cli_args": [],
        "timestamp": "",
    }
    (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(payload))
    with caplog.at_level("WARNING"):
        m = load_manifest(tmp_path)
    assert m.seed == 5
    assert any("manifest version" in rec.message.lower() for rec in caplog.records)


# ---------- environment diff ------------------------------------------------


def test_compare_environments_match():
    from halo_forge.replay import compare_environments

    fp = {
        "python": "3.13.0",
        "platform": "Darwin arm64",
        "backend": "mlx",
        "packages": {"torch": "2.5.0"},
    }
    diff = compare_environments(fp, fp)
    assert diff["matched"] is True
    assert diff["differences"] == []


def test_compare_environments_top_level_diff():
    from halo_forge.replay import compare_environments

    captured = {
        "python": "3.13.0",
        "platform": "Darwin arm64",
        "backend": "mlx",
        "packages": {},
    }
    current = {
        "python": "3.13.0",
        "platform": "Linux x86_64",  # different
        "backend": "cuda",  # different
        "packages": {},
    }
    diff = compare_environments(captured, current)
    assert diff["matched"] is False
    keys = {d["key"] for d in diff["differences"]}
    assert keys == {"platform", "backend"}


def test_compare_environments_package_diff():
    from halo_forge.replay import compare_environments

    captured = {
        "python": "3.13.0",
        "platform": "x",
        "backend": "cuda",
        "packages": {"torch": "2.4.0", "trl": "0.20.0"},
    }
    current = {
        "python": "3.13.0",
        "platform": "x",
        "backend": "cuda",
        "packages": {"torch": "2.5.0", "trl": "0.20.0", "vllm": "0.6.0"},
    }
    diff = compare_environments(captured, current)
    keys = {d["key"] for d in diff["differences"]}
    assert "packages.torch" in keys
    assert "packages.vllm" in keys
    # trl unchanged → not in diff
    assert "packages.trl" not in keys


# ---------- CLI ------------------------------------------------------------


@pytest.fixture
def manifest_dir(tmp_path: Path):
    """Write a real manifest to disk so the CLI tests are end-to-end."""
    from halo_forge.replay import capture_manifest, save_manifest

    src = tmp_path / "train.jsonl"
    src.write_text('{"text": "x"}\n')
    m = capture_manifest(
        run_id="run-test",
        modality="sft",
        model_name="Qwen/Qwen2.5-3B",
        seed=42,
        config={
            "model_name": "Qwen/Qwen2.5-3B",
            "dataset": "codealpaca",
            "output_dir": "models/sft",
            "num_epochs": 3,
            "max_samples": 1000,
        },
        dataset_file=src,
    )
    save_manifest(m, tmp_path)
    return tmp_path


def test_cli_replay_help_registers(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "replay", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for token in ("--launch", "--force", "replay"):
        assert token in out


def test_cli_replay_prints_command(manifest_dir, monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "replay", str(manifest_dir)])
    cli_mod.main()
    out = capsys.readouterr().out
    assert "run-test" in out
    assert "Reproducible launch command" in out
    # The command should include the captured subcommand + key flags.
    assert "halo-forge sft train" in out
    assert "--model Qwen/Qwen2.5-3B" in out
    assert "--dataset codealpaca" in out


def test_cli_replay_missing_source_exits_clean(tmp_path: Path, monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "replay", str(tmp_path / "nope")])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "not found" in out.lower()


def test_reconstruct_launch_command_dispatches_per_modality():
    from halo_forge.cli import _reconstruct_launch_command
    from halo_forge.replay import ReplayManifest

    def make(modality, **cfg):
        return ReplayManifest(
            manifest_version=1,
            run_id="r",
            modality=modality,
            timestamp="",
            model_name=cfg.get("model_name", ""),
            seed=42,
            pythonhashseed=None,
            config=cfg,
            dataset={},
            environment={},
            cli_args=[],
        )

    sft = _reconstruct_launch_command(make("sft", model_name="m", dataset="d"))
    assert sft[:3] == ["halo-forge", "sft", "train"]

    dpo = _reconstruct_launch_command(make("dpo", model_name="m"))
    assert dpo[:3] == ["halo-forge", "dpo", "train"]

    grpo = _reconstruct_launch_command(make("grpo", model_name="m"))
    assert grpo[:3] == ["halo-forge", "grpo", "train"]

    # MLX-flavored modalities (e.g. "dpo_mlx-...") still route to dpo.
    mlx_dpo = _reconstruct_launch_command(make("dpo_mlx-1234567"))
    assert mlx_dpo[:3] == ["halo-forge", "dpo", "train"]


def test_reconstruct_managed_launch_uses_immutable_role_bindings():
    from halo_forge.cli import _reconstruct_launch_command
    from halo_forge.replay import ReplayManifest

    manifest = ReplayManifest(
        manifest_version=2,
        run_id="run",
        modality="dpo",
        timestamp="",
        model_name="model",
        seed=42,
        pythonhashseed=None,
        config={
            "model_name": "model",
            "train_file": "/tmp/rendered-artifact/train.jsonl",
            "output_dir": "models/dpo",
        },
        dataset={
            "kind": "managed_versions",
            "bindings": [
                {
                    "role": "train",
                    "dataset_version_id": "version-train",
                    "split": "train",
                },
                {
                    "role": "validation",
                    "dataset_version_id": "version-validation",
                    "split": "validation",
                },
            ],
        },
        environment={},
        cli_args=[],
    )

    command = _reconstruct_launch_command(manifest)
    assert command.count("--dataset-binding") == 2
    assert "train=version-train:train" in command
    assert "validation=version-validation:validation" in command
    assert "--data" not in command
