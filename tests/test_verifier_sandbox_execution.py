"""End-to-end sandbox execution tests for the shipped compile verifiers.

These exercise the *real* sandbox backend under the *default* execution policy
(``"sandbox"``) rather than monkeypatching it away.  ``tests/
test_verifier_sandbox_policy.py`` replaces ``VerifierExecutionRunner.run``
with a stub that always raises, so it stays green even when the sandbox cannot
launch a single process; the tests here are the ones that notice.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional

import pytest

from halo_forge.rlvr.verifiers import execution_runner as execution_runner_module
from halo_forge.rlvr.verifiers.compile import CompileVerifier
from halo_forge.rlvr.verifiers.execution_runner import (
    SandboxUnavailableError,
    VerifierExecutionRunner,
)


HELLO_C = """
#include <stdio.h>

int main(void) {
    printf("hello from the sandbox\\n");
    return 0;
}
"""

# Reads a TCP port on stdin and tries to connect to it on loopback.  Loopback
# keeps the test deterministic offline while still exercising `(deny network*)`.
NETWORK_C = """
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

int main(void) {
    int port = 0;
    if (scanf("%d", &port) != 1) {
        printf("NO_PORT\\n");
        return 0;
    }
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        printf("NETWORK_BLOCKED socket\\n");
        return 0;
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
        printf("NETWORK_OK\\n");
    } else {
        printf("NETWORK_BLOCKED connect\\n");
    }
    return 0;
}
"""


def _sandbox_backend() -> Optional[str]:
    """Return the sandbox backend only when it can execute a real process."""
    if sys.platform == "darwin":
        backend = shutil.which("sandbox-exec")
    elif sys.platform.startswith("linux"):
        backend = shutil.which("bwrap")
    else:
        backend = None
    if backend is None:
        return None

    trivial = "/usr/bin/true" if Path("/usr/bin/true").exists() else shutil.which("true")
    if trivial is None:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="halo-forge-test-sandbox-") as temp_dir:
            runner = VerifierExecutionRunner(
                execution_policy="sandbox", workspace_root=Path(temp_dir)
            )
            result = runner.run([trivial], cwd=Path(temp_dir), timeout=30)
    except (OSError, subprocess.SubprocessError, SandboxUnavailableError):
        return None
    return backend if result.returncode == 0 else None


def _c_compiler() -> Optional[str]:
    """Return a usable system C compiler command, or None."""
    for candidate in ("cc", "clang", "gcc"):
        found = shutil.which(candidate)
        if found:
            return candidate
    return None


requires_sandbox = pytest.mark.skipif(
    _sandbox_backend() is None,
    reason="no executable local sandbox backend (sandbox-exec / bwrap) on this host",
)
requires_compiler = pytest.mark.skipif(
    _c_compiler() is None,
    reason="no system C compiler on this host",
)


@pytest.fixture(autouse=True)
def _clear_sandbox_probe_cache():
    """Keep the cached sandbox probe from leaking between tests."""
    probe = getattr(execution_runner_module, "_macos_sandbox_probe_failure", None)
    if probe is not None:
        probe.cache_clear()
    yield
    if probe is not None:
        probe.cache_clear()


@requires_sandbox
def test_sandbox_profile_allows_process_execution(tmp_path: Path) -> None:
    """The shipped profile must let the sandbox exec a trivial binary at all."""
    trivial = "/usr/bin/true" if Path("/usr/bin/true").exists() else shutil.which("true")
    if not trivial:
        pytest.skip("no trivial binary available for the exec probe")

    runner = VerifierExecutionRunner(execution_policy="sandbox", workspace_root=tmp_path)
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    result = runner.run([trivial], cwd=working_dir, timeout=30)

    assert result.returncode == 0, f"sandbox refused to exec {trivial}: {result.stderr!r}"


@requires_sandbox
def test_workspace_root_is_unreadable_and_unwritable(tmp_path: Path) -> None:
    """Verified code must not reach the caller's workspace outside its own run dir.

    ``(allow default)`` makes the workspace deny rules the only thing protecting
    the caller's tree, so they need direct coverage: a profile that merely fails
    to match (see the resolved-path test below) silently grants full access while
    every other test here still passes.
    """
    secret = tmp_path / "secret.txt"
    secret.write_text("TOPSECRET\n", encoding="utf-8")
    working_dir = tmp_path / "work"
    working_dir.mkdir()
    runner = VerifierExecutionRunner(execution_policy="sandbox", workspace_root=tmp_path)

    read_attempt = runner.run(["/bin/cat", str(secret)], cwd=working_dir, timeout=30)
    assert read_attempt.returncode != 0, "sandbox leaked workspace contents"
    assert "TOPSECRET" not in read_attempt.stdout

    write_attempt = runner.run(
        ["/usr/bin/touch", str(tmp_path / "pwned.txt")], cwd=working_dir, timeout=30
    )
    assert write_attempt.returncode != 0, "sandbox allowed a write into the workspace root"
    assert not (tmp_path / "pwned.txt").exists()

    # The per-run working directory must stay writable, or verifiers cannot build.
    allowed = runner.run(["/usr/bin/touch", str(working_dir / "ok.txt")], cwd=working_dir, timeout=30)
    assert allowed.returncode == 0, f"working dir not writable: {allowed.stderr!r}"


@pytest.mark.skipif(sys.platform != "darwin", reason="path resolution trap is darwin-only")
@requires_sandbox
def test_workspace_deny_survives_an_unresolved_symlinked_root(tmp_path: Path) -> None:
    """A workspace root given via a symlinked path must still be denied.

    ``/tmp`` and ``/var`` are symlinks into ``/private`` on macOS, and sandbox
    profile ``subpath`` rules match the *physical* path only.  If the runner ever
    stops calling ``resolve()``, the deny rules quietly match nothing and the
    workspace becomes fully readable while the sandbox still looks healthy.
    """
    symlinked_tmp = Path("/tmp")
    if not symlinked_tmp.is_symlink():
        pytest.skip("/tmp is not a symlink on this host; the trap does not apply")

    workspace_root = symlinked_tmp / f"halo_forge_unresolved_{os.getpid()}"
    working_dir = workspace_root / "work"
    working_dir.mkdir(parents=True)
    secret = workspace_root / "secret.txt"
    secret.write_text("TOPSECRET\n", encoding="utf-8")
    try:
        runner = VerifierExecutionRunner(
            execution_policy="sandbox", workspace_root=workspace_root
        )

        result = runner.run(["/bin/cat", str(secret)], cwd=working_dir, timeout=30)

        assert result.returncode != 0, "deny rule did not match the physical workspace path"
        assert "TOPSECRET" not in result.stdout
    finally:
        shutil.rmtree(workspace_root, ignore_errors=True)


@requires_sandbox
@requires_compiler
def test_default_policy_compiles_and_runs_hello_world(tmp_path: Path) -> None:
    """The real compile+run verifier succeeds under the default sandbox policy."""
    verifier = CompileVerifier(
        compiler=_c_compiler(),
        flags=["-w", "-O2"],
        run_after_compile=True,
        expected_output="hello from the sandbox",
        binary_cache_dir=str(tmp_path / "cache"),
    )
    assert verifier.execution_policy == "sandbox"

    result = verifier.verify(HELLO_C)

    assert result.success is True, f"{result.details}: {result.error}"
    assert result.reward == 1.0
    assert result.metadata["output"] == "hello from the sandbox"


@requires_sandbox
@requires_compiler
def test_default_policy_captures_program_output_without_expected_output(tmp_path: Path) -> None:
    """Program stdout is captured and surfaced in metadata."""
    verifier = CompileVerifier(
        compiler=_c_compiler(),
        flags=["-w", "-O2"],
        run_after_compile=True,
    )

    result = verifier.verify(HELLO_C)

    assert result.success is True, f"{result.details}: {result.error}"
    assert "hello from the sandbox" in result.metadata["stdout"]


@pytest.fixture
def loopback_listener():
    """An accepting TCP listener on loopback; yields its port."""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(8)

    def _accept_loop() -> None:
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            conn.close()

    threading.Thread(target=_accept_loop, daemon=True).start()
    try:
        yield server.getsockname()[1]
    finally:
        server.close()


def _connect_attempt_output(port: int, execution_policy: str) -> str:
    """Compile+run the connect probe under one execution policy; return its stdout."""
    verifier = CompileVerifier(
        compiler=_c_compiler(),
        flags=["-w"],
        run_after_compile=True,
        stdin_input=f"{port}\n",
        run_timeout=10,
        execution_policy=execution_policy,
    )
    result = verifier.verify(NETWORK_C)
    return f"{result.metadata.get('stdout', '')}{result.error or ''}"


@requires_sandbox
@requires_compiler
def test_verified_code_cannot_reach_the_network(loopback_listener) -> None:
    """Outbound network from verified code must not succeed under the sandbox.

    The ``unsafe_host`` leg is the control: it proves the probe really can
    connect on this host, so the sandboxed leg's refusal is the sandbox and not
    a dead listener.
    """
    control = _connect_attempt_output(loopback_listener, "unsafe_host")
    assert "NETWORK_OK" in control, f"probe cannot connect even unsandboxed: {control!r}"

    sandboxed = _connect_attempt_output(loopback_listener, "sandbox")

    assert "NETWORK_OK" not in sandboxed, f"sandbox allowed an outbound connection: {sandboxed!r}"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS SDK resolution is darwin-only")
@requires_sandbox
def test_macos_sdk_root_is_supplied_to_the_compiler(tmp_path: Path) -> None:
    """Command Line Tools clang needs an SDK root; the verifier must supply it.

    ``_build_env`` strips the environment, so an inherited ``SDKROOT`` never
    survives.  Without an explicit sysroot this compile dies on ``'stdio.h'
    file not found``.
    """
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is unavailable; Command Line Tools are not installed")
    probe = subprocess.run(
        ["xcrun", "-f", "clang"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        pytest.skip("xcrun cannot resolve clang on this host")
    toolchain_clang = probe.stdout.strip()

    verifier = CompileVerifier(
        compiler=toolchain_clang,
        flags=["-w"],
        run_after_compile=True,
        expected_output="hello from the sandbox",
    )

    result = verifier.verify(HELLO_C)

    assert result.success is True, f"{result.details}: {result.error}"
    assert result.reward == 1.0


@pytest.mark.skipif(sys.platform != "darwin", reason="SDKROOT propagation is darwin-only")
def test_build_env_propagates_sdkroot_on_macos(tmp_path: Path) -> None:
    """Stripped verifier environments still carry a resolvable SDKROOT."""
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is unavailable; Command Line Tools are not installed")

    runner = VerifierExecutionRunner(execution_policy="sandbox", workspace_root=tmp_path)
    env = runner._build_env(tmp_path, None)

    assert "SDKROOT" in env
    assert Path(env["SDKROOT"]).is_dir()


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec exit codes are darwin-only")
def test_execvp_denial_is_reported_as_sandbox_unavailable(monkeypatch, tmp_path: Path) -> None:
    """Exit 71 from sandbox-exec must never look like an ordinary build failure.

    A denied ``execvp()`` used to fall through the ``sandbox_apply`` check and
    be reported as a compile failure with reward 0.0 -- indistinguishable from
    the model emitting broken code.
    """
    monkeypatch.setattr(execution_runner_module, "_ensure_macos_sandbox_usable", lambda: None)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=71,
            stdout="",
            stderr="sandbox-exec: execvp() of '/bin/echo' failed: Operation not permitted\n",
        )

    monkeypatch.setattr(execution_runner_module.subprocess, "run", fake_run)

    runner = VerifierExecutionRunner(execution_policy="sandbox", workspace_root=tmp_path)

    with pytest.raises(SandboxUnavailableError):
        runner.run(["/bin/echo", "hi"], cwd=tmp_path, timeout=5)


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox probe is darwin-only")
def test_broken_sandbox_raises_instead_of_scoring_zero(monkeypatch, tmp_path: Path) -> None:
    """A host whose sandbox cannot exec must fail loudly, not silently score 0."""
    monkeypatch.setattr(
        execution_runner_module,
        "_macos_sandbox_probe_failure",
        lambda: "execvp() of '/usr/bin/true' failed: Operation not permitted",
    )

    runner = VerifierExecutionRunner(execution_policy="sandbox", workspace_root=tmp_path)

    with pytest.raises(SandboxUnavailableError) as excinfo:
        runner.run(["/usr/bin/true"], cwd=tmp_path, timeout=5)

    assert "unsafe_host" in str(excinfo.value)


def test_unsupported_platform_still_fails_closed(monkeypatch, tmp_path: Path) -> None:
    """Non-darwin/linux hosts keep raising rather than silently running unsandboxed."""
    monkeypatch.setattr(execution_runner_module.sys, "platform", "win32")

    runner = VerifierExecutionRunner(execution_policy="sandbox", workspace_root=tmp_path)

    with pytest.raises(SandboxUnavailableError):
        runner.run(["cmd"], cwd=tmp_path, timeout=5)


def test_unsafe_host_policy_still_runs_directly(tmp_path: Path) -> None:
    """The escape hatch keeps working (no sandbox binary involved)."""
    runner = VerifierExecutionRunner(execution_policy="unsafe_host", workspace_root=tmp_path)

    result = runner.run([sys.executable, "-c", "print('direct')"], cwd=tmp_path, timeout=60)

    assert result.returncode == 0
    assert "direct" in result.stdout
