"""Shared execution runner for verifier subprocesses."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional


ExecutionPolicy = Literal["sandbox", "unsafe_host"]


class SandboxUnavailableError(RuntimeError):
    """Raised when sandboxed verifier execution is unavailable."""


@dataclass(frozen=True)
class VerifierExecutionRunner:
    """Run verifier subprocesses with explicit execution policy."""

    execution_policy: ExecutionPolicy = "sandbox"
    workspace_root: Path = Path.cwd()

    def __post_init__(self) -> None:
        policy = str(self.execution_policy or "").strip().lower()
        if policy not in {"sandbox", "unsafe_host"}:
            raise ValueError("execution_policy must be 'sandbox' or 'unsafe_host'")
        object.__setattr__(self, "execution_policy", policy)
        object.__setattr__(self, "workspace_root", Path(self.workspace_root).resolve())

    def run(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        timeout: int,
        input_text: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        preexec_fn: Optional[Callable[[], None]] = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a verifier subprocess."""
        working_dir = Path(cwd).resolve()
        resolved_env = self._build_env(working_dir, env)

        if self.execution_policy == "unsafe_host":
            return self._run_unsafe_host(
                cmd,
                working_dir=working_dir,
                timeout=timeout,
                input_text=input_text,
                env=resolved_env,
                preexec_fn=preexec_fn,
            )

        # macOS: sandbox-exec (native, ships with the OS).
        if sys.platform == "darwin":
            return self._run_macos_sandbox(
                cmd,
                working_dir=working_dir,
                timeout=timeout,
                input_text=input_text,
                env=resolved_env,
                preexec_fn=preexec_fn,
            )

        # Linux: bubblewrap (bwrap) — read-only system roots, read-write
        # working dir, network/IPC/PID namespaces dropped. Phase 6 of the
        # multi-backend roadmap: previously Linux contributors could only
        # use execution_policy='unsafe_host'.
        if sys.platform.startswith("linux"):
            return self._run_linux_sandbox(
                cmd,
                working_dir=working_dir,
                timeout=timeout,
                input_text=input_text,
                env=resolved_env,
                preexec_fn=preexec_fn,
            )

        raise SandboxUnavailableError(
            f"No sandbox backend available for platform {sys.platform!r}. "
            "Install a supported sandbox backend or opt into execution_policy='unsafe_host'."
        )

    def _run_macos_sandbox(
        self,
        cmd: list[str],
        *,
        working_dir: Path,
        timeout: int,
        input_text: Optional[str],
        env: dict[str, str],
        preexec_fn: Optional[Callable[[], None]],
    ) -> subprocess.CompletedProcess[str]:
        """sandbox-exec branch (macOS native)."""
        sandbox_exec = shutil.which("sandbox-exec")
        if not sandbox_exec:
            raise SandboxUnavailableError(
                "Local sandboxed verifier execution is unavailable on this host "
                "(sandbox-exec not found). Install a supported sandbox backend "
                "or opt into execution_policy='unsafe_host'."
            )

        profile_path = self._write_sandbox_profile(working_dir)
        sandbox_cmd = [sandbox_exec, "-f", str(profile_path), *cmd]
        try:
            result = subprocess.run(
                sandbox_cmd,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(working_dir),
                env=env,
                preexec_fn=preexec_fn,
            )
        except FileNotFoundError as exc:
            raise SandboxUnavailableError(
                "sandbox-exec could not be started. "
                "Install a supported sandbox backend or opt into execution_policy='unsafe_host'."
            ) from exc
        stderr_text = result.stderr or ""
        if result.returncode == 71 and "sandbox_apply" in stderr_text:
            raise SandboxUnavailableError(
                "sandbox-exec is present but could not be applied in this environment. "
                "Use a supported local sandbox or opt into execution_policy='unsafe_host'."
            )
        return result

    def _run_linux_sandbox(
        self,
        cmd: list[str],
        *,
        working_dir: Path,
        timeout: int,
        input_text: Optional[str],
        env: dict[str, str],
        preexec_fn: Optional[Callable[[], None]],
    ) -> subprocess.CompletedProcess[str]:
        """bubblewrap (bwrap) branch for Linux contributors.

        Mirrors the spirit of the macOS sandbox-exec profile:
          - System dirs (/usr, /lib*, /etc, /bin, /sbin) bound read-only so
            the verifier can exec compilers/runtimes but not modify them.
          - The working directory is bound read-write at its real path so
            the verifier writes its outputs where callers expect them.
          - The wider workspace_root is *not* mounted: the verifier can't
            walk up to read other users' or other runs' files.
          - Network, IPC, PID, UTS, and cgroup namespaces dropped.
          - /tmp + /dev provided fresh per invocation.

        Requires the `bubblewrap` package (Debian/Ubuntu/Fedora all package it).
        """
        bwrap = shutil.which("bwrap")
        if not bwrap:
            raise SandboxUnavailableError(
                "bubblewrap (`bwrap`) is not on PATH. Install it (`apt install bubblewrap` "
                "or `dnf install bubblewrap`) or opt into execution_policy='unsafe_host'."
            )

        ro_binds: list[str] = []
        for system_path in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"):
            if os.path.isdir(system_path) or os.path.islink(system_path):
                ro_binds.extend(["--ro-bind", system_path, system_path])

        bwrap_cmd = [
            bwrap,
            *ro_binds,
            "--bind", str(working_dir), str(working_dir),
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--unshare-all",  # net/ipc/pid/uts/cgroup/user namespaces dropped
            "--die-with-parent",
            "--new-session",
            "--chdir", str(working_dir),
            *cmd,
        ]

        try:
            return subprocess.run(
                bwrap_cmd,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(working_dir),
                env=env,
                preexec_fn=preexec_fn,
            )
        except FileNotFoundError as exc:
            raise SandboxUnavailableError(
                "bwrap could not be started. Install bubblewrap or opt into "
                "execution_policy='unsafe_host'."
            ) from exc

    @staticmethod
    def _run_unsafe_host(
        cmd: list[str],
        *,
        working_dir: Path,
        timeout: int,
        input_text: Optional[str],
        env: dict[str, str],
        preexec_fn: Optional[Callable[[], None]],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(working_dir),
            env=env,
            preexec_fn=preexec_fn,
        )

    def _build_env(
        self,
        working_dir: Path,
        env: Optional[dict[str, str]],
    ) -> dict[str, str]:
        source = dict(env or os.environ)
        resolved = {
            "PATH": source.get("PATH", os.defpath),
            "HOME": str(working_dir),
            "TMPDIR": source.get("TMPDIR", tempfile.gettempdir()),
            "LANG": source.get("LANG", "C.UTF-8"),
            "LC_ALL": source.get("LC_ALL", source.get("LANG", "C.UTF-8")),
        }
        if "SYSTEMROOT" in source:
            resolved["SYSTEMROOT"] = source["SYSTEMROOT"]
        return resolved

    def _write_sandbox_profile(self, working_dir: Path) -> Path:
        profile = "\n".join(
            [
                "(version 1)",
                '(import "system.sb")',
                "(deny network*)",
                f'(deny file-read* (subpath "{self._escape_path(self.workspace_root)}"))',
                f'(deny file-write* (subpath "{self._escape_path(self.workspace_root)}"))',
                f'(allow file-read* (subpath "{self._escape_path(working_dir)}"))',
                f'(allow file-write* (subpath "{self._escape_path(working_dir)}"))',
            ]
        )
        profile_path = working_dir / ".halo_forge_sandbox.sb"
        profile_path.write_text(profile, encoding="utf-8")
        return profile_path

    @staticmethod
    def _escape_path(path: Path) -> str:
        return str(path).replace("\\", "\\\\").replace('"', '\\"')
