"""Reward-model trainer tests (Track T3)."""

from __future__ import annotations

import sys

import pytest


def test_rm_module_imports():
    from halo_forge.rm import RMConfig, get_rm_trainer

    cfg = RMConfig()
    assert callable(get_rm_trainer)
    # RM defaults: small LoRA, smaller LR than SFT, narrow training run.
    assert cfg.lora_r <= 16
    assert cfg.learning_rate <= 5e-5
    assert cfg.num_epochs == 1
    assert cfg.center_rewards_coefficient is not None


def test_rm_config_target_modules_default():
    """Same Qwen-shaped default as SFTConfig / DPOConfig — base RMs are
    typically the same family of base models."""
    from halo_forge.rm.config import RMConfig

    cfg = RMConfig()
    assert "q_proj" in cfg.target_modules
    assert "down_proj" in cfg.target_modules


def test_dispatch_mlx_path_raises_typed_error():
    from halo_forge.rm._dispatch import get_rm_trainer

    class _FakeMLXBackend:
        name = "mlx"

    with pytest.raises(NotImplementedError) as ei:
        get_rm_trainer(backend=_FakeMLXBackend())  # type: ignore[arg-type]
    msg = str(ei.value)
    assert "MLX" in msg
    assert "roadmap" in msg.lower() or "mps" in msg.lower()


def test_format_pair_produces_required_columns():
    """The pre-tokenize pass must produce the exact columns TRL's
    RewardTrainer iterates."""
    from halo_forge.rm.trainer import _format_pair

    class _StubTokenizer:
        def __call__(self, text, *, truncation, max_length, padding):
            ids = list(range(min(len(text), max_length)))
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    row = {"prompt": "Q: ", "chosen": "good", "rejected": "bad"}
    out = _format_pair(row, _StubTokenizer(), max_length=128)
    assert set(out) == {
        "input_ids_chosen", "attention_mask_chosen",
        "input_ids_rejected", "attention_mask_rejected",
    }
    # Chosen and rejected concatenate the prompt — same prefix.
    chosen_len = len(out["input_ids_chosen"])
    rejected_len = len(out["input_ids_rejected"])
    assert chosen_len == len("Q: good")
    assert rejected_len == len("Q: bad")


def test_cli_rm_help_registers(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "rm", "train", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    for token in (
        "--model", "--dataset", "--center-rewards-coefficient",
        "--lora-rank", "--epochs",
    ):
        assert token in out


def test_cli_rm_no_args_exits_with_hint(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "rm", "train"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "--dataset" in out or "--data" in out


def test_cli_rm_dry_run_routes_through_config(monkeypatch, capsys):
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(
        sys, "argv",
        [
            "halo-forge", "rm", "train",
            "--dataset", "ultrafeedback",
            "--model", "Qwen/Qwen2.5-3B-Instruct",
            "--lora-rank", "16",
            "--learning-rate", "5e-6",
            "--dry-run",
        ],
    )
    cli_mod.main()
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert "lora_rank=16" in out
    assert "lr=5e-06" in out or "lr=5e-06" in out.lower()
