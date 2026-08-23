"""Shared execution runner for verifier subprocesses."""

from __future__ import annotations

import os
import multiprocessing
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Optional


ExecutionPolicy = Literal["sandbox", "unsafe_host"]

_UNSAFE_HOST_HINT = (
    "Use a supported local sandbox or opt into execution_policy='unsafe_host'."
)


def _prefer_spawn_on_darwin() -> None:
    if sys.platform != "darwin":
        return
    try:
        if multiprocessing.get_start_method(allow_none=True) == "fork":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


_prefer_spawn_on_darwin()


class SandboxUnavailableError(RuntimeError):
    """Raised when sandboxed verifier execution is unavailable."""


def _escape_sandbox_path(path: Path) -> str:
    """Escape a filesystem path for embedding in a sandbox profile literal."""
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def render_sandbox_profile(workspace_root: Path, working_dir: Path) -> str:
    """Render the macOS sandbox profile used for verifier subprocesses.

    The profile starts from ``(allow default)`` and then subtracts capabilities:
    all networking is denied, the caller's workspace is unreadable/unwritable,
    and only the per-run working directory is readable/writable.  Rules are
    evaluated last-match-wins, so the working-directory allows re-open the hole
    punched by the workspace denies when the two overlap.

    ``(import "system.sb")`` was used here previously.  On macOS 26 that bundled
    profile no longer grants ``process-exec``, so every compile and every
    execution failed with ``execvp() ... Operation not permitted`` and scored
    reward 0.0.
    """
    return "\n".join(
        [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            f'(deny file-read* (subpath "{_escape_sandbox_path(workspace_root)}"))',
            f'(deny file-write* (subpath "{_escape_sandbox_path(workspace_root)}"))',
            f'(allow file-read* (subpath "{_escape_sandbox_path(working_dir)}"))',
            f'(allow file-write* (subpath "{_escape_sandbox_path(working_dir)}"))',
        ]
    )


@lru_cache(maxsize=1)
def macos_sdk_path() -> Optional[str]:
    """Resolve the macOS SDK root via ``xcrun``, or ``None`` when unavailable.

    ``_build_env`` deliberately strips the environment, so an inherited
    ``SDKROOT`` never reaches the compiler.  Command Line Tools clang/gcc then
    fail with ``'stdio.h' file not found``.  Resolved once per process; callers
    must tolerate ``None`` (Command Line Tools not installed).
    """
    if sys.platform != "darwin":
        return None
    xcrun = shutil.which("xcrun")
    if not xcrun:
        return None
    try:
        result = subprocess.run(
            [xcrun, "--show-sdk-path"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sdk_path = (result.stdout or "").strip()
    return sdk_path if sdk_path and os.path.isdir(sdk_path) else None


@lru_cache(maxsize=1)
def _macos_sandbox_probe_failure() -> Optional[str]:
    """Run the real profile against a trivial binary; return the failure detail.

    Returns ``None`` when the sandbox can exec normally (or when no probe binary
    exists, in which case we do not block the caller).  Cached for the life of
    the process; call ``_macos_sandbox_probe_failure.cache_clear()`` in tests.
    """
    sandbox_exec = shutil.which("sandbox-exec")
    if not sandbox_exec:
        return "sandbox-exec was not found on PATH."
    probe_binary = "/usr/bin/true" if os.path.exists("/usr/bin/true") else shutil.which("true")
    if not probe_binary:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="halo_forge_sandbox_probe_") as tmp_dir:
            workspace_root = Path(tmp_dir).resolve()
            working_dir = workspace_root / "run"
            working_dir.mkdir()
            profile_path = working_dir / ".halo_forge_sandbox.sb"
            profile_path.write_text(
                render_sandbox_profile(workspace_root, working_dir), encoding="utf-8"
            )
            result = subprocess.run(
                [sandbox_exec, "-f", str(profile_path), probe_binary],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(working_dir),
                check=False,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"sandbox-exec probe could not be started ({exc})."
    if result.returncode == 0:
        return None
    detail = (result.stderr or "").strip() or f"exit code {result.returncode}"
    return f"sandbox-exec could not run {probe_binary}: {detail}"


def _ensure_macos_sandbox_usable() -> None:
    """Fail loudly if the sandbox profile cannot exec anything on this host.

    Without this probe a broken sandbox looks exactly like a model writing
    uncompilable code: every verifier returns reward 0.0 with a plausible
    "Compilation failed" message.
    """
    failure = _macos_sandbox_probe_failure()
    if failure:
        raise SandboxUnavailableError(
            "Local sandboxed verifier execution is unavailable on this host. "
            f"{failure} All verifier rewards would be silently 0.0. "
            f"{_UNSAFE_HOST_HINT}"
        )


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
            "Local sandboxed verifier execution is unavailable on this host "
            f"(unsupported platform {sys.platform!r}). Install a supported "
            "sandbox backend or opt into execution_policy='unsafe_host'."
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

        _ensure_macos_sandbox_usable()

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
        # Exit 71 (EX_OSERR) always comes from sandbox-exec itself, never from the
        # verified program: a failed `sandbox_apply`, a denied `execvp`, or a
        # malformed profile.  Any of those is sandbox unavailability, not a
        # verifier failure, and must never be scored as reward 0.0.
        if result.returncode == 71 and (
            "sandbox_apply" in stderr_text or stderr_text.lstrip().startswith("sandbox-exec:")
        ):
            raise SandboxUnavailableError(
                "sandbox-exec is present but could not run the verifier command "
                f"({stderr_text.strip() or 'no stderr'}). {_UNSAFE_HOST_HINT}"
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
                "Local sandboxed verifier execution is unavailable on this host "
                "(bubblewrap `bwrap` not found). Install a supported sandbox backend "
                "(`apt install bubblewrap` or `dnf install bubblewrap`) or opt into "
                "execution_policy='unsafe_host'."
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
        # Rustup and Cargo commonly live below the user's real home directory.
        # Verifiers intentionally replace HOME with the isolated working
        # directory, so retain only these explicit toolchain locations when a
        # caller supplies them.  Without them, a rustup-backed ``cargo`` shim
        # sees an empty home and reports that no default toolchain exists.
        for variable in ("CARGO_HOME", "RUSTUP_HOME"):
            value = source.get(variable)
            if value:
                resolved[variable] = value
        if "SYSTEMROOT" in source:
            resolved["SYSTEMROOT"] = source["SYSTEMROOT"]
        # macOS Command Line Tools compilers cannot find system headers without an
        # SDK root.  The stripped environment above drops any inherited SDKROOT,
        # so re-resolve it here rather than leaving every C/C++ compile broken.
        if sys.platform == "darwin":
            sdk_root = source.get("SDKROOT") or macos_sdk_path()
            if sdk_root:
                resolved["SDKROOT"] = sdk_root
        return resolved

    def _write_sandbox_profile(self, working_dir: Path) -> Path:
        """Write the per-run sandbox profile into the working directory."""
        profile = render_sandbox_profile(self.workspace_root, working_dir)
        profile_path = working_dir / ".halo_forge_sandbox.sb"
        profile_path.write_text(profile, encoding="utf-8")
        return profile_path

    @staticmethod
    def _escape_path(path: Path) -> str:
        return _escape_sandbox_path(path)
