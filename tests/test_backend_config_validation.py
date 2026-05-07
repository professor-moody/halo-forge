"""Backend-specific config validation tests.

Exercises `warn_unsupported_for_mlx` and `warn_experimental_vllm_backend`
so silent-failure paths from earlier in Q1 (PEFT/optimizer flags
silently dropped on MLX, gfx1151 vLLM allowed without note) stay
covered.
"""

from __future__ import annotations

import logging

import pytest


def test_default_config_emits_no_warnings():
    """If a user runs `halo-forge sft train --accelerator mlx` with
    nothing else, no warnings fire — the defaults are MLX-compatible."""
    from halo_forge.sft.config import SFTConfig
    from halo_forge.utils.backend_config import warn_unsupported_for_mlx

    cfg = SFTConfig()
    emitted = warn_unsupported_for_mlx(cfg)
    assert emitted == []


def test_use_dora_warns_on_mlx():
    from halo_forge.sft.config import SFTConfig
    from halo_forge.utils.backend_config import warn_unsupported_for_mlx

    cfg = SFTConfig(use_dora=True)
    emitted = warn_unsupported_for_mlx(cfg)
    assert any("use_dora" in line for line in emitted)
    assert any("DoRA" in line for line in emitted)


def test_pissa_init_warns_on_mlx():
    from halo_forge.sft.config import SFTConfig
    from halo_forge.utils.backend_config import warn_unsupported_for_mlx

    cfg = SFTConfig(init_lora_weights="pissa")
    emitted = warn_unsupported_for_mlx(cfg)
    assert any("init_lora_weights" in line for line in emitted)
    assert any("pissa" in line.lower() or "peft-only" in line for line in emitted)


def test_bnb_optim_warns_on_mlx():
    from halo_forge.sft.config import SFTConfig
    from halo_forge.utils.backend_config import warn_unsupported_for_mlx

    cfg = SFTConfig(optim="adamw_bnb_8bit")
    emitted = warn_unsupported_for_mlx(cfg)
    assert any("optim" in line for line in emitted)
    assert any("bitsandbytes" in line.lower() or "mlx.optimizers" in line for line in emitted)


def test_multiple_unsupported_fields_each_get_warning():
    from halo_forge.sft.config import SFTConfig
    from halo_forge.utils.backend_config import warn_unsupported_for_mlx

    cfg = SFTConfig(
        use_dora=True,
        use_rslora=True,
        init_lora_weights="loftq",
        optim="lion_8bit",
    )
    emitted = warn_unsupported_for_mlx(cfg)
    assert len(emitted) == 4


def test_warnings_logged_via_logging_module(caplog):
    """User-facing print is good for interactive sessions; the logging
    path is what log files capture."""
    from halo_forge.sft.config import SFTConfig
    from halo_forge.utils.backend_config import warn_unsupported_for_mlx

    cfg = SFTConfig(use_dora=True)
    with caplog.at_level(logging.WARNING):
        warn_unsupported_for_mlx(cfg)
    assert any("use_dora" in rec.message for rec in caplog.records)


def test_warnings_printed_to_stdout(capsys):
    from halo_forge.sft.config import SFTConfig
    from halo_forge.utils.backend_config import warn_unsupported_for_mlx

    cfg = SFTConfig(use_dora=True)
    warn_unsupported_for_mlx(cfg)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "use_dora" in out


def test_dpo_config_unsupported_fields_also_warn():
    """DPOConfig shares the PEFT / optim fields with SFTConfig; the
    same validator should catch them."""
    from halo_forge.dpo.config import DPOConfig
    from halo_forge.utils.backend_config import warn_unsupported_for_mlx

    cfg = DPOConfig(use_dora=True, optim="adamw_bnb_8bit")
    emitted = warn_unsupported_for_mlx(cfg, trainer_label="MLX DPO")
    assert len(emitted) == 2
    assert all("[MLX DPO]" in line for line in emitted)


def test_vllm_gfx1151_warning():
    from halo_forge.utils.backend_config import warn_experimental_vllm_backend

    assert warn_experimental_vllm_backend("rocm_gfx1151") is True
    assert warn_experimental_vllm_backend("cuda") is False
    assert warn_experimental_vllm_backend("rocm_mi300x") is False


def test_vllm_gfx1151_warning_has_fallback_advice(capsys):
    from halo_forge.utils.backend_config import warn_experimental_vllm_backend

    warn_experimental_vllm_backend("rocm_gfx1151")
    out = capsys.readouterr().out
    # Mention both fallbacks so a stuck operator can pivot.
    assert "torch" in out
    assert "mlx" in out


def test_mlx_sft_trainer_init_emits_warnings(monkeypatch, capsys):
    """End-to-end: instantiating MLXSFTTrainer with a non-MLX-compatible
    config should print the warnings without requiring a real model."""
    pytest.importorskip("mlx_lm")  # Test only meaningful if MLX is around
    from halo_forge.sft.config import SFTConfig
    from halo_forge.sft.mlx_trainer import MLXSFTTrainer

    cfg = SFTConfig(use_dora=True, optim="adamw_bnb_8bit")
    MLXSFTTrainer(cfg)
    out = capsys.readouterr().out
    assert "use_dora" in out
    assert "optim" in out
