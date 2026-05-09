from halo_forge.rlvr.verifiers import (
    DotNetVerifier,
    GoVerifier,
    PytestVerifier,
    RustVerifier,
    SubprocessVerifier,
)
from halo_forge.rlvr.verifiers.execution_runner import SandboxUnavailableError, VerifierExecutionRunner


def _force_sandbox_unavailable(monkeypatch):
    def fail_run(self, cmd, *, cwd, timeout, input_text=None, env=None, preexec_fn=None):
        raise SandboxUnavailableError("test sandbox unavailable")

    monkeypatch.setattr(VerifierExecutionRunner, "run", fail_run)


def test_pytest_verifier_fails_closed_when_sandbox_unavailable(monkeypatch):
    _force_sandbox_unavailable(monkeypatch)

    result = PytestVerifier(timeout=1).verify("def test_ok():\n    assert True\n")

    assert result.success is False
    assert result.reward == 0.0
    assert result.details == "Sandbox unavailable"
    assert result.metadata["execution_policy"] == "sandbox"


def test_go_verifier_fails_closed_when_sandbox_unavailable(monkeypatch):
    _force_sandbox_unavailable(monkeypatch)

    result = GoVerifier(timeout=1).verify("package main\nfunc main() {}\n")

    assert result.success is False
    assert result.reward == 0.0
    assert result.details == "Sandbox unavailable"
    assert result.metadata["execution_policy"] == "sandbox"


def test_rust_verifier_fails_closed_when_sandbox_unavailable(monkeypatch):
    _force_sandbox_unavailable(monkeypatch)

    result = RustVerifier(timeout=1).verify("fn main() {}\n")

    assert result.success is False
    assert result.reward == 0.0
    assert result.details == "Sandbox unavailable"
    assert result.metadata["execution_policy"] == "sandbox"


def test_dotnet_verifier_fails_closed_when_sandbox_unavailable(monkeypatch):
    _force_sandbox_unavailable(monkeypatch)

    result = DotNetVerifier(timeout=1).verify("using System; class Program { static void Main() {} }\n")

    assert result.success is False
    assert result.reward == 0.0
    assert result.details == "Sandbox unavailable"
    assert result.metadata["execution_policy"] == "sandbox"


def test_subprocess_verifier_fails_closed_when_sandbox_unavailable(monkeypatch):
    _force_sandbox_unavailable(monkeypatch)

    result = SubprocessVerifier(["python", "{file}"], timeout=1).verify("print('ok')\n")

    assert result.success is False
    assert result.reward == 0.0
    assert result.details == "Sandbox unavailable"
    assert result.metadata["execution_policy"] == "sandbox"
