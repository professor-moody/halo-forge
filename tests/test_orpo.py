"""ORPO trainer wiring tests (Track T17b).

Don't actually run training — that needs a GPU + tens of MB of weights.
Cover the parts the rest of the system depends on:

  - ORPOConfig defaults are sane.
  - get_orpo_trainer() picks the right path per backend.
  - The CLI parser wires the orpo subcommand.
  - The recovery / summary contract uses modality="orpo".
"""

from __future__ import annotations

import pytest


def test_orpo_config_defaults():
    from halo_forge.orpo import ORPOConfig

    cfg = ORPOConfig()
    # Sensible defaults — same shape as DPO, slightly higher LR (no
    # ref-model regularization keeping it small).
    assert cfg.beta == 0.1
    assert cfg.learning_rate > 0 and cfg.learning_rate < 1e-4
    assert cfg.lora_r == 16
    assert cfg.target_modules and "q_proj" in cfg.target_modules


def test_orpo_config_post_init_fills_target_modules():
    from halo_forge.orpo import ORPOConfig

    cfg = ORPOConfig(target_modules=None)
    assert cfg.target_modules == [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]


def test_get_orpo_trainer_raises_clearly_on_mlx(monkeypatch):
    """MLX path is roadmap; the dispatcher should raise with a pointer
    rather than silently routing to the PyTorch trainer."""
    from halo_forge.orpo import get_orpo_trainer

    class FakeBackend:
        name = "mlx"

    monkeypatch.setattr(
        "halo_forge.orpo._dispatch.get_backend",
        lambda **kwargs: FakeBackend(),
    )
    with pytest.raises(NotImplementedError, match="ORPO on MLX"):
        get_orpo_trainer()


def test_orpo_module_exports():
    """Both the public symbols downstream code imports."""
    import halo_forge.orpo as mod

    assert hasattr(mod, "ORPOConfig")
    assert hasattr(mod, "get_orpo_trainer")


def test_orpo_template_exists():
    """Track T17b: at least one ORPO template lives in the gallery so
    the modality is discoverable from /train/templates."""
    from halo_forge.training import TEMPLATES

    orpo_templates = [t for t in TEMPLATES if t.modality == "orpo"]
    assert len(orpo_templates) >= 1, "Expected at least one ORPO template"


def test_cli_handler_exists():
    """The CLI dispatch handler is wired and importable. The argparse
    parser is built inside `main()`, so we verify the handler symbol
    rather than the parser tree (which is awkward to introspect
    without invoking sys.argv parsing)."""
    from halo_forge import cli

    assert callable(getattr(cli, "cmd_orpo_train", None)), (
        "halo_forge.cli must expose cmd_orpo_train for the orpo subcommand"
    )


def test_cli_help_includes_orpo(tmp_path, capsys):
    """`halo-forge --help` should list orpo as a top-level subcommand."""
    import sys

    from halo_forge import cli

    argv_backup = sys.argv
    sys.argv = ["halo-forge", "--help"]
    with pytest.raises(SystemExit) as exc:
        cli.main()
    sys.argv = argv_backup
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "orpo" in out
