"""Inspect registered CLI commands without launching their workloads."""

from __future__ import annotations

import argparse


def has_cli_command(*command_path: str) -> bool:
    """Check the actual parser hierarchy, independently of source formatting.

    This is registration evidence, not evidence that a workload ran. Readiness
    callers must still retain their separate execution and artifact checks.
    """
    from halo_forge.cli import build_parser

    if not command_path:
        return False
    parser = build_parser()
    for command in command_path:
        child = next(
            (
                action.choices[command]
                for action in parser._actions
                if isinstance(action, argparse._SubParsersAction)
                and command in action.choices
            ),
            None,
        )
        if child is None:
            return False
        parser = child
    return True
