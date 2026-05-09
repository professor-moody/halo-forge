"""macOS runtime helpers for long-running training jobs."""

from __future__ import annotations

import shutil
import subprocess
import sys


def caffeinate_command(command: list[str], enabled: bool = True) -> list[str]:
    if not enabled or sys.platform != "darwin" or not shutil.which("caffeinate"):
        return list(command)
    return ["caffeinate", "-i", "-m", "-s", "--", *command]


def caffeinate_subprocess(command: list[str]) -> subprocess.Popen:
    return subprocess.Popen(caffeinate_command(command))

