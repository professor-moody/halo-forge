"""Unified-conversion tests (Track I5).

These exercise the dispatch + quant-table logic without actually loading
or quantizing a model — that's a slow integration test gated on the
right backend. The fast tests here catch:
  - dispatch picks the right backend converter
  - quant translation is correct per format
  - unsupported (format, quant) combos raise typed errors
  - HF re-export rejects true quantization terms cleanly
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_supported_format_and_quant_listings_are_stable():
    from halo_forge.inference.convert import (
        list_supported_formats,
        list_supported_quants,
    )

    assert "mlx" in list_supported_formats()
    assert "gguf" in list_supported_formats()
    assert "hf" in list_supported_formats()
    assert {"q4", "q8", "fp16", "bf16", "fp32"}.issubset(set(list_supported_quants()))


def test_unknown_format_raises():
    from halo_forge.inference.convert import convert

    with pytest.raises(ValueError, match="Unknown target format"):
        convert(
            source="x", output_path="/tmp/out", target_format="bogus", quantization="q4"
        )


def test_unknown_quant_raises():
    from halo_forge.inference.convert import _resolve_quant

    with pytest.raises(ValueError, match="Unknown quantization"):
        _resolve_quant("q3", "mlx")


def test_unsupported_quant_for_format_raises_with_hint():
    """Some quants don't make sense for some formats — error must
    say which quants ARE supported instead of leaving the user guessing."""
    from halo_forge.inference.convert import _resolve_quant

    # Make sure the table currently exposes mlx for q4. We tamper with
    # the table copy via a patched table to force the negative case.
    with patch.dict(
        "halo_forge.inference.convert._QUANT_TABLE",
        {"q4": {"gguf": {"quantization": "q4_k_m"}}},
        clear=True,
    ):
        with pytest.raises(ValueError, match="not supported for format"):
            _resolve_quant("q4", "mlx")


def test_quant_table_translates_q4_to_mlx_args():
    from halo_forge.inference.convert import _resolve_quant

    args = _resolve_quant("q4", "mlx")
    assert args["quantize"] is True
    assert args["q_bits"] == 4
    assert args["q_group_size"] == 64


def test_quant_table_translates_q8_to_gguf_args():
    from halo_forge.inference.convert import _resolve_quant

    args = _resolve_quant("q8", "gguf")
    assert args["quantization"] == "q8_0"


def test_hf_reexport_rejects_quant_terms():
    """`--format hf` is dtype recast, not quantization. Asking for q4 is
    a category error and must produce a precise message rather than
    silently doing the wrong thing."""
    from halo_forge.inference.convert import convert_to_hf

    with pytest.raises(ValueError, match="dtype changes"):
        convert_to_hf(source="x", output_path="/tmp/y", quantization="q4")


def test_convert_dispatches_to_mlx(monkeypatch, tmp_path):
    from halo_forge.inference import convert as convert_mod

    captured = {}

    def fake_mlx_convert(*, source, output_path, quantization, trust_remote_code):
        captured.update(
            source=source,
            output_path=output_path,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
        )
        return SimpleNamespace(  # ConvertResult-shaped enough for the dispatch test
            source=source,
            output_path=output_path,
            target_format="mlx",
            quantization=quantization,
            bytes_written=42,
        )

    monkeypatch.setattr(convert_mod, "convert_to_mlx", fake_mlx_convert)
    out = convert_mod.convert(
        source="hf-id",
        output_path=str(tmp_path / "out"),
        target_format="mlx",
        quantization="q4",
    )
    assert captured["source"] == "hf-id"
    assert captured["quantization"] == "q4"
    assert captured["trust_remote_code"] is True
    assert out.target_format == "mlx"


def test_convert_dispatches_to_gguf(monkeypatch, tmp_path):
    from halo_forge.inference import convert as convert_mod

    captured = {}

    def fake_gguf_convert(*, source, output_path, quantization, trust_remote_code):
        captured.update(
            source=source,
            output_path=output_path,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
        )
        return SimpleNamespace(
            source=source,
            output_path=output_path,
            target_format="gguf",
            quantization=quantization,
            bytes_written=99,
        )

    monkeypatch.setattr(convert_mod, "convert_to_gguf", fake_gguf_convert)
    out = convert_mod.convert(
        source="src",
        output_path=str(tmp_path / "out.gguf"),
        target_format="gguf",
        quantization="q8",
    )
    assert captured["quantization"] == "q8"
    assert out.target_format == "gguf"


def test_cli_convert_help_registers(monkeypatch, capsys):
    """`halo-forge convert --help` must show the new subcommand without
    blowing up imports."""
    import sys
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "convert", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "format" in out.lower() or "quant" in out.lower()


def test_cli_convert_list_prints_supported(monkeypatch, capsys):
    """`halo-forge convert --list` short-circuits without requiring source."""
    import sys
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "convert", "--list"])
    cli_mod.main()
    out = capsys.readouterr().out
    assert "mlx" in out and "gguf" in out
    assert "q4" in out and "fp16" in out
