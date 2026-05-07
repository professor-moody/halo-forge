"""Replay manifest tests (Track T15)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

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
        run_id="r", modality="sft", model_name="x", seed=42,
        config={"learning_rate": 0.001},
        dataset_file=src,
    )
    assert m.dataset["kind"] == "local_file"
    assert m.dataset["sha256"]
    assert m.dataset["size_bytes"] == src.stat().st_size


def test_capture_manifest_with_dict_config():
    from halo_forge.replay import capture_manifest

    m = capture_manifest(
        run_id="r", modality="grpo", model_name="x", seed=42,
        config={"beta": 0.04, "num_generations": 4},
    )
    assert m.config == {"beta": 0.04, "num_generations": 4}


def test_save_and_load_roundtrip(tmp_path: Path):
    from halo_forge.replay import capture_manifest, load_manifest, save_manifest

    m = capture_manifest(
        run_id="r", modality="sft", model_name="x", seed=99,
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
        run_id="r", modality="sft", model_name="x", seed=1, config={},
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
        "backend": "cuda",            # different
        "packages": {},
    }
    diff = compare_environments(captured, current)
    assert diff["matched"] is False
    keys = {d["key"] for d in diff["differences"]}
    assert keys == {"platform", "backend"}


def test_compare_environments_package_diff():
    from halo_forge.replay import compare_environments

    captured = {
        "python": "3.13.0", "platform": "x", "backend": "cuda",
        "packages": {"torch": "2.4.0", "trl": "0.20.0"},
    }
    current = {
        "python": "3.13.0", "platform": "x", "backend": "cuda",
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

    monkeypatch.setattr(
        sys, "argv", ["halo-forge", "replay", str(manifest_dir)]
    )
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

    monkeypatch.setattr(
        sys, "argv", ["halo-forge", "replay", str(tmp_path / "nope")]
    )
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
            manifest_version=1, run_id="r", modality=modality,
            timestamp="", model_name=cfg.get("model_name", ""),
            seed=42, pythonhashseed=None,
            config=cfg, dataset={}, environment={}, cli_args=[],
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
