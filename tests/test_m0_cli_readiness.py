"""Command readiness follows the shipped parser and dispatcher."""

from __future__ import annotations

import argparse
from unittest.mock import Mock

import pytest

from halo_forge import cli
from halo_forge.cli_readiness import has_cli_command


@pytest.mark.parametrize(
    "path",
    [
        ("serve-public",),
        ("config", "validate"),
        ("data", "prepare"),
        ("data", "generate"),
        ("data", "validate"),
        ("info",),
        ("sft", "train"),
        ("raft", "train"),
        ("plot", "training"),
        ("plot", "benchmarks"),
        ("benchmark", "run"),
        ("benchmark", "full"),
        ("benchmark", "eval"),
    ],
)
def test_registered_commands_have_working_help(path):
    assert has_cli_command(*path)
    with pytest.raises(SystemExit) as caught:
        cli.build_parser().parse_args([*path, "--help"])
    assert caught.value.code == 0


def test_missing_or_misnested_command_fails_without_source_text_fallback(monkeypatch):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers()
    plot = commands.add_parser("plot")
    plot.add_subparsers().add_parser("training")
    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    assert has_cli_command("plot", "training")
    assert not has_cli_command("training")
    assert not has_cli_command("plot", "benchmarks")
    assert not has_cli_command("serve-public")
    assert not has_cli_command()


@pytest.mark.parametrize(
    ("namespace", "handler"),
    [
        ({"command": "serve-public"}, "cmd_serve_public"),
        ({"command": "plot", "plot_command": "training"}, "cmd_plot_training"),
        ({"command": "plot", "plot_command": "benchmarks"}, "cmd_plot_benchmarks"),
        ({"command": "benchmark", "bench_command": "run"}, "cmd_benchmark"),
        (
            {"command": "benchmark", "bench_command": "full", "suite": "all"},
            "cmd_benchmark_full",
        ),
        ({"command": "benchmark", "bench_command": "eval"}, "cmd_benchmark_eval"),
    ],
)
def test_affected_commands_dispatch_to_their_handler(monkeypatch, namespace, handler):
    mocked = Mock()
    monkeypatch.setattr(cli, handler, mocked)
    monkeypatch.setattr(cli, "setup_auto_logging", Mock(return_value="unused-test-log"))
    args = argparse.Namespace(model=None, quiet=True, **namespace)
    cli._dispatch_commands(args)
    mocked.assert_called_once_with(args)
