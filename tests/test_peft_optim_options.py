"""PEFT and optimizer option tests (Tracks T4 + T5).

Validates that the DoRA / rsLoRA / PiSSA / optimizer-choice options on
``SFTConfig`` and ``DPOConfig`` propagate through to the LoraConfig kwargs
and TrainingArguments the trainers build, without actually running training.

We exercise:
  - the ``_parse_init_lora_weights`` helper handles "true" / "false" /
    PEFT-name strings (pissa / loftq / olora / gaussian) correctly
  - default config values keep behavior identical to pre-T4/T5 (no
    silent regression for users who don't opt in)
  - new CLI flags parse and route to the config
"""

from __future__ import annotations

import pytest


def test_parse_init_lora_weights_string_to_value():
    from halo_forge.sft.trainer import _parse_init_lora_weights

    # bool-like strings collapse to actual booleans (PEFT distinguishes)
    assert _parse_init_lora_weights("true") is True
    assert _parse_init_lora_weights("True") is True
    assert _parse_init_lora_weights("FALSE") is False
    # PEFT-named init schemes pass through unchanged
    assert _parse_init_lora_weights("pissa") == "pissa"
    assert _parse_init_lora_weights("pissa_niter_4") == "pissa_niter_4"
    assert _parse_init_lora_weights("loftq") == "loftq"
    assert _parse_init_lora_weights("olora") == "olora"
    assert _parse_init_lora_weights("gaussian") == "gaussian"
    # None falls back to True (the LoraConfig default)
    assert _parse_init_lora_weights(None) is True


def test_sft_config_peft_optim_defaults_unchanged():
    """Pre-existing SFT users see no behavior change."""
    from halo_forge.sft.config import SFTConfig

    cfg = SFTConfig()
    assert cfg.use_dora is False
    assert cfg.use_rslora is False
    assert cfg.init_lora_weights == "true"
    assert cfg.optim == "adamw_torch"


def test_dpo_config_peft_optim_defaults_unchanged():
    from halo_forge.dpo.config import DPOConfig

    cfg = DPOConfig()
    assert cfg.use_dora is False
    assert cfg.use_rslora is False
    assert cfg.init_lora_weights == "true"
    assert cfg.optim == "adamw_torch"


def test_sft_cli_routes_peft_and_optim_flags(monkeypatch, capsys):
    """`halo-forge sft train --use-dora --init-lora-weights pissa --optim adamw_bnb_8bit`
    must route those flags into SFTConfig without erroring at parse time."""
    import sys
    import halo_forge.cli as cli_mod

    captured = {}

    def fake_get_sft_trainer(config):
        captured["config"] = config

        class _Stub:
            def train(self, **kwargs):
                return {"summary": "stub"}

        return _Stub()

    # The handler imports `get_sft_trainer` lazily inside the function body;
    # patch the module path it pulls from.
    monkeypatch.setattr("halo_forge.sft._dispatch.get_sft_trainer", fake_get_sft_trainer)
    # Skip the real summary print which expects fields we didn't populate.
    monkeypatch.setattr(cli_mod, "_print_completed_training_summary", lambda *a, **k: None)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "sft",
            "train",
            "--dataset",
            "codealpaca",
            "--model",
            "tiny/model",
            "--use-dora",
            "--use-rslora",
            "--init-lora-weights",
            "pissa",
            "--optim",
            "adamw_bnb_8bit",
            "--max-samples",
            "1",
        ],
    )
    cli_mod.main()
    cfg = captured["config"]
    assert cfg.use_dora is True
    assert cfg.use_rslora is True
    assert cfg.init_lora_weights == "pissa"
    assert cfg.optim == "adamw_bnb_8bit"


def test_dpo_cli_routes_peft_and_optim_flags(monkeypatch):
    import sys
    import halo_forge.cli as cli_mod

    captured = {}

    def fake_get_dpo_trainer(config):
        captured["config"] = config

        class _Stub:
            def train(self, **kwargs):
                return {"summary": "stub"}

        return _Stub()

    # Patch both the source and the package-level re-export — the CLI
    # handler imports `get_dpo_trainer` from `halo_forge.dpo` (which
    # binds the name at __init__ time), so patching only the underlying
    # _dispatch module would miss it.
    monkeypatch.setattr("halo_forge.dpo._dispatch.get_dpo_trainer", fake_get_dpo_trainer)
    monkeypatch.setattr("halo_forge.dpo.get_dpo_trainer", fake_get_dpo_trainer)
    monkeypatch.setattr(cli_mod, "_print_completed_training_summary", lambda *a, **k: None)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "halo-forge",
            "dpo",
            "train",
            "--dataset",
            "ultrafeedback",
            "--model",
            "tiny/model",
            "--use-dora",
            "--init-lora-weights",
            "pissa_niter_8",
            "--optim",
            "lion_8bit",
        ],
    )
    cli_mod.main()
    cfg = captured["config"]
    assert cfg.use_dora is True
    assert cfg.use_rslora is False
    assert cfg.init_lora_weights == "pissa_niter_8"
    assert cfg.optim == "lion_8bit"


def test_lora_config_accepts_dora_rslora_kwargs():
    """Sanity check: the LoraConfig kwargs we forward are real PEFT params."""
    import dataclasses
    from peft import LoraConfig

    fields = {f.name for f in dataclasses.fields(LoraConfig)}
    assert {"use_dora", "use_rslora", "init_lora_weights"}.issubset(fields)
