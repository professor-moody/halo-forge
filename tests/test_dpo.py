"""DPO trainer smoke tests (Track T1 / phase Q1).

These tests exercise the import graph and config / dispatch / dataset
surfaces without actually running training — DPO needs torch + a real
model to do anything meaningful, and we already gate end-to-end training
behind hardware-specific markers elsewhere. The fast tests here catch the
common failure modes:

  - import-time errors (missing trl, broken module wiring)
  - dispatch logic on each backend (PyTorch path vs MLX-stub error)
  - dataset normalization (UltraFeedback's chat-list `chosen` column,
    local JSONL with raw strings)
  - config defaults that align with TRL's expectations
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_dpo_module_imports():
    """The top-level `halo_forge.dpo` import must succeed without torch."""
    from halo_forge.dpo import DPOConfig, get_dpo_trainer, load_preference_dataset

    assert callable(get_dpo_trainer)
    assert callable(load_preference_dataset)
    cfg = DPOConfig()
    assert cfg.beta > 0
    assert cfg.loss_type == "sigmoid"
    assert cfg.learning_rate < 1e-4, "DPO LR must default smaller than SFT"


def test_dpo_config_defaults_are_dpo_shaped():
    """Spot-check the values that distinguish DPO from SFT defaults."""
    from halo_forge.dpo.config import DPOConfig

    cfg = DPOConfig()
    # DPO defaults: small LR, short epochs, larger warmup, no weight decay.
    assert cfg.num_epochs == 1
    assert cfg.learning_rate == 5e-6
    assert cfg.warmup_ratio >= 0.05
    assert cfg.max_grad_norm == 1.0
    # DPO-specific knobs.
    assert cfg.max_prompt_length <= cfg.max_seq_length
    assert cfg.label_smoothing == 0.0
    assert cfg.reference_free is False


def test_preference_registry_short_names_resolve():
    from halo_forge.dpo.datasets import PREFERENCE_DATASETS, list_preference_datasets

    items = list_preference_datasets()
    names = {i.name for i in items}
    # The four datasets we ship short names for.
    assert {"ultrafeedback", "orca_dpo", "hh_rlhf", "py_dpo"}.issubset(names)
    for spec in items:
        assert spec.huggingface_id and "/" in spec.huggingface_id
        assert PREFERENCE_DATASETS[spec.name] is spec


def test_local_jsonl_normalization_round_trip(tmp_path: Path):
    """A local JSONL file with mixed string / messages columns normalizes
    into the prompt/chosen/rejected schema TRL expects."""
    from halo_forge.dpo.datasets import load_preference_dataset

    rows = [
        # Pure-string row (jondurbin / py_dpo style).
        {
            "prompt": "What is 2+2?",
            "chosen": "4",
            "rejected": "5",
        },
        # ChatML-list row (UltraFeedback style chosen/rejected).
        {
            "prompt": "Explain gravity",
            "chosen": [
                {"role": "user", "content": "Explain gravity"},
                {"role": "assistant", "content": "Mass attracts mass."},
            ],
            "rejected": [
                {"role": "user", "content": "Explain gravity"},
                {"role": "assistant", "content": "Idk lol"},
            ],
        },
        # Empty row that should be filtered out.
        {"prompt": "", "chosen": "x", "rejected": "y"},
    ]
    path = tmp_path / "pairs.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    train, val = load_preference_dataset(
        train_file=str(path),
        validation_split=0.0,
        max_samples=None,
    )
    # Empty-prompt row should be filtered.
    assert len(train) == 2
    cols = set(train.column_names)
    assert cols == {"prompt", "chosen", "rejected"}
    # Strings stay strings; chat lists collapse to "role: content" lines.
    assert "Mass attracts mass" in train[1]["chosen"]
    assert train[0]["chosen"] == "4"


def test_dispatch_mlx_path_returns_mlx_trainer_when_reference_free():
    """T17 v1: MLX dispatch returns MLXDPOTrainer when reference_free=True
    is set on the config; otherwise raises NotImplementedError pointing
    at the knob to flip. This replaces the old "always raise" stub."""
    from halo_forge.dpo import DPOConfig
    from halo_forge.dpo._dispatch import get_dpo_trainer

    class _FakeMLXBackend:
        name = "mlx"

    # Without reference_free: trainer construction raises.
    cfg_default = DPOConfig()  # reference_free defaults to False
    with pytest.raises(NotImplementedError) as ei:
        get_dpo_trainer(cfg_default, backend=_FakeMLXBackend())  # type: ignore[arg-type]
    assert "reference-free" in str(ei.value).lower() or "reference_free" in str(ei.value)

    # With reference_free=True: trainer instantiates.
    cfg_ref_free = DPOConfig(reference_free=True)
    trainer = get_dpo_trainer(cfg_ref_free, backend=_FakeMLXBackend())  # type: ignore[arg-type]
    assert trainer.__class__.__name__ == "MLXDPOTrainer"


def test_mlx_dpo_rejects_non_sigmoid_loss():
    """v1 only supports loss_type='sigmoid' — IPO / hinge / kto_pair
    require the reference model."""
    from halo_forge.dpo import DPOConfig
    from halo_forge.dpo._dispatch import get_dpo_trainer

    class _FakeMLXBackend:
        name = "mlx"

    cfg = DPOConfig(reference_free=True, loss_type="ipo")
    with pytest.raises(NotImplementedError) as ei:
        get_dpo_trainer(cfg, backend=_FakeMLXBackend())  # type: ignore[arg-type]
    assert "ipo" in str(ei.value).lower() or "sigmoid" in str(ei.value).lower()


@pytest.mark.requires_cuda
def test_dispatch_returns_pytorch_trainer_on_cuda():
    """On a CUDA host the dispatcher returns the PyTorch trainer class."""
    from halo_forge.dpo import DPOConfig, get_dpo_trainer

    trainer = get_dpo_trainer(DPOConfig())
    assert trainer.__class__.__name__ == "DPOTrainer"


def test_cli_help_includes_dpo_subcommand(monkeypatch):
    """`halo-forge dpo train --help` must register so the CLI knows the new
    subparser. We exercise argparse by feeding sys.argv and catching the
    SystemExit argparse raises after printing help.
    """
    import sys
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "dpo", "train", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    # argparse's --help exits 0.
    assert ei.value.code == 0


def test_cli_dpo_train_routes_without_args(monkeypatch, capsys):
    """`halo-forge dpo train` with no dataset/data should hint and exit 1,
    not blow up importing the module. Catches dispatch + handler wiring."""
    import sys
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "dpo", "train"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 1
    out = capsys.readouterr().out
    assert "ultrafeedback" in out  # the help text we printed
