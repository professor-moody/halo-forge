"""Adapter / model merging tests (Tracks T12 + T13).

Validates dispatch, method-table translation, and CLI wiring without
loading a real model. Real-merge integration tests are gated behind
running on a host with a checkpoint to merge — that's a slow path
intentionally not in the default suite.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def test_supported_methods_listing_stable():
    from halo_forge.inference.merge import list_supported_methods

    methods = list_supported_methods()
    assert {"linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"}.issubset(
        set(methods)
    )


def test_unknown_method_raises_with_listing():
    from halo_forge.inference.merge import _resolve_method_kwargs

    with pytest.raises(ValueError, match="Unknown merge method"):
        _resolve_method_kwargs("does_not_exist")


def test_method_table_maps_to_peft_combination_types():
    """The CLI vocabulary translates to peft's actual `combination_type`
    arg names so callers don't have to know peft internals."""
    from halo_forge.inference.merge import _resolve_method_kwargs

    assert _resolve_method_kwargs("linear")["combination_type"] == "linear"
    assert "ties" in _resolve_method_kwargs("ties")["combination_type"]
    # DARE methods carry a default density.
    dare = _resolve_method_kwargs("dare_ties")
    assert "density" in dare
    assert "ties" in dare["combination_type"]


def test_merge_dispatch_to_bake(monkeypatch):
    from halo_forge.inference import merge as merge_mod

    captured = {}

    def fake_bake(*, base_model, adapter_path, output_path, trust_remote_code):
        captured.update(
            base_model=base_model,
            adapter_path=adapter_path,
            output_path=output_path,
        )
        return SimpleNamespace(
            operation="bake",
            output_path=output_path,
            method="bake",
            base_model=base_model,
            adapters=[adapter_path],
            weights=[1.0],
            bytes_written=0,
        )

    monkeypatch.setattr(merge_mod, "bake_adapter", fake_bake)
    out = merge_mod.merge(
        operation="bake",
        base_model="qwen-3b",
        adapter_path="/tmp/lora",
        output_path="/tmp/merged",
    )
    assert out.operation == "bake"
    assert captured["base_model"] == "qwen-3b"


def test_merge_dispatch_to_combine(monkeypatch):
    from halo_forge.inference import merge as merge_mod

    captured = {}

    def fake_combine(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            operation="combine",
            output_path=kwargs["output_path"],
            method=kwargs["method"],
            base_model=kwargs["base_model"],
            adapters=list(kwargs["adapter_paths"]),
            weights=kwargs.get("weights") or [1.0, 1.0],
            bytes_written=0,
        )

    monkeypatch.setattr(merge_mod, "combine_adapters", fake_combine)
    out = merge_mod.merge(
        operation="combine",
        base_model="qwen-3b",
        adapter_paths=["/tmp/a", "/tmp/b"],
        weights=[0.7, 0.3],
        method="dare_ties",
        output_path="/tmp/out",
    )
    assert out.operation == "combine"
    assert captured["adapter_paths"] == ["/tmp/a", "/tmp/b"]
    assert captured["weights"] == [0.7, 0.3]
    assert captured["method"] == "dare_ties"


def test_combine_requires_two_adapters():
    """Single adapter into combine is a category error — should be bake."""
    from halo_forge.inference.merge import combine_adapters

    with pytest.raises(ValueError, match="at least two"):
        combine_adapters(
            base_model="x",
            adapter_paths=["/tmp/only_one"],
            output_path="/tmp/out",
        )


def test_combine_weights_length_must_match():
    from halo_forge.inference.merge import combine_adapters

    with pytest.raises(ValueError, match="weights length"):
        combine_adapters(
            base_model="x",
            adapter_paths=["/tmp/a", "/tmp/b", "/tmp/c"],
            weights=[0.5, 0.5],
            output_path="/tmp/out",
        )


def test_bake_requires_adapter():
    from halo_forge.inference.merge import merge

    with pytest.raises(ValueError, match="adapter"):
        merge(operation="bake", base_model="x", output_path="/tmp/y")


def test_unknown_operation_raises():
    from halo_forge.inference.merge import merge

    with pytest.raises(ValueError, match="Unknown merge operation"):
        merge(
            operation="franken",
            base_model="x",
            adapter_paths=["/tmp/a", "/tmp/b"],
            output_path="/tmp/y",
        )


def test_cli_help_includes_merge_subcommand(monkeypatch, capsys):
    import sys
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "merge", "--help"])
    with pytest.raises(SystemExit) as ei:
        cli_mod.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "bake" in out
    assert "combine" in out
    assert "method" in out


def test_cli_merge_list_short_circuits(monkeypatch, capsys):
    import sys
    import halo_forge.cli as cli_mod

    monkeypatch.setattr(sys, "argv", ["halo-forge", "merge", "--list"])
    cli_mod.main()
    out = capsys.readouterr().out
    assert "bake" in out
    assert "combine" in out
    for method in ("linear", "ties", "dare_ties"):
        assert method in out
