"""GRPO trainer tests (Track T2)."""

from __future__ import annotations

import math

import pytest


def test_grpo_module_imports():
    from halo_forge.grpo import GRPOConfig, get_grpo_trainer

    cfg = GRPOConfig()
    assert callable(get_grpo_trainer)
    assert cfg.num_generations == 4
    assert cfg.beta == 0.04  # DeepSeek-R1 default
    # GRPO LR is much smaller than DPO/SFT
    assert cfg.learning_rate < 1e-5
    # default verifier resolves to a registered name
    assert cfg.verifier_name


def test_group_advantages_zeroes_singleton_groups():
    """A group of 1 has no within-group signal — advantage must be 0."""
    from halo_forge.grpo.mlx_trainer import _group_advantages

    assert _group_advantages([0.5]) == [0.0]
    assert _group_advantages([]) == []


def test_group_advantages_canonical_grpo_normalization():
    """Group [0, 1] (mean 0.5, std 0.5) → advantages [-1, +1]."""
    from halo_forge.grpo.mlx_trainer import _group_advantages

    result = _group_advantages([0.0, 1.0])
    assert result[0] == pytest.approx(-1.0, rel=1e-6)
    assert result[1] == pytest.approx(1.0, rel=1e-6)


def test_group_advantages_constant_group_yields_zero():
    """If every completion has the same reward, no signal — all 0s.
    Avoids division-by-zero when std collapses."""
    from halo_forge.grpo.mlx_trainer import _group_advantages

    assert _group_advantages([0.7, 0.7, 0.7, 0.7]) == [0.0, 0.0, 0.0, 0.0]


def test_group_advantages_rloo_flavor_skips_std():
    """scale_by_std=False → mean-only baseline (RLOO-flavored)."""
    from halo_forge.grpo.mlx_trainer import _group_advantages

    result = _group_advantages([0.0, 1.0], scale_by_std=False)
    assert result == [-0.5, 0.5]


def test_group_advantages_three_member_group():
    """Sanity: a 3-member group preserves zero-mean advantages."""
    from halo_forge.grpo.mlx_trainer import _group_advantages

    result = _group_advantages([0.0, 0.5, 1.0])
    assert sum(result) == pytest.approx(0.0, abs=1e-6)
    # And ordering: lowest reward → most negative advantage.
    assert result[0] < result[1] < result[2]


def test_dispatch_mlx_path_returns_mlx_trainer_when_reference_free():
    from halo_forge.grpo import GRPOConfig
    from halo_forge.grpo._dispatch import get_grpo_trainer

    class _FakeMLXBackend:
        name = "mlx"

    # Default reference_free=False → typed error.
    cfg = GRPOConfig()
    with pytest.raises(NotImplementedError, match=r"reference-free|reference_free"):
        get_grpo_trainer(cfg, backend=_FakeMLXBackend())  # type: ignore[arg-type]

    # reference_free=True → instantiates.
    cfg_rf = GRPOConfig(reference_free=True)
    trainer = get_grpo_trainer(cfg_rf, backend=_FakeMLXBackend())  # type: ignore[arg-type]
    assert trainer.__class__.__name__ == "MLXGRPOTrainer"


def test_mlx_grpo_trainer_module_imports_without_mlx():
    """Module must load on non-MLX hosts so dispatcher can import it."""
    import halo_forge.grpo.mlx_trainer as mod

    assert hasattr(mod, "MLXGRPOTrainer")
    assert hasattr(mod, "_group_advantages")


def test_mlx_grpo_warns_on_unsupported_config(capsys):
    """DoRA/optim warnings fire at MLX trainer init."""
    from halo_forge.grpo import GRPOConfig
    from halo_forge.grpo.mlx_trainer import MLXGRPOTrainer

    cfg = GRPOConfig(reference_free=True, use_dora=True, optim="adamw_bnb_8bit")
    MLXGRPOTrainer(cfg)
    out = capsys.readouterr().out
    assert "use_dora" in out
    assert "optim" in out


def test_cli_help_includes_grpo_subcommand(monkeypatch, capsys):
    """`halo-forge grpo train --help` must register."""
    import sys
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "grpo", "train", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    # All the GRPO knobs we expose must surface in help.
    for token in ("num-generations", "beta", "verifier", "reference-free", "rollout-engine"):
        assert token in out


def test_cli_grpo_train_routes_dry_run(monkeypatch, capsys):
    """End-to-end CLI: `--dry-run` validates without training, prints config echo."""
    import sys
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(
        sys, "argv",
        [
            "halo-forge", "grpo", "train",
            "--data", "/tmp/_does_not_exist.jsonl",
            "--model", "Qwen/Qwen2.5-3B-Instruct",
            "--num-generations", "8",
            "--beta", "0.05",
            "--verifier", "execution",
            "--reference-free",
            "--dry-run",
        ],
    )
    cli_mod.main()
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert "num_generations=8" in out
    assert "beta=0.05" in out
    assert "verifier=execution" in out
    assert "reference_free=True" in out


def test_grpo_no_data_or_dataset_exits_with_hint(monkeypatch, capsys):
    import sys
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "grpo", "train"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "--dataset" in out or "--data" in out
