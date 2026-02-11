#!/usr/bin/env python3
"""Phase 1 security and runtime hardening regression tests."""

import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from halo_forge.rlvr.verifiers.base import VerifyResult
from halo_forge.rlvr.verifiers.custom import SubprocessVerifier
from halo_forge.rlvr.verifiers.execution import ExecutionVerifier
from halo_forge.rlvr.verifiers.multi_language import LanguageConfig, MultiLanguageVerifier


def test_subprocess_verifier_rejects_string_command():
    """String commands are no longer accepted."""
    with pytest.raises(TypeError):
        SubprocessVerifier(command_args="echo {file}")


def test_subprocess_verifier_requires_placeholder_when_enabled():
    """Placeholder validation should fail fast when required."""
    with pytest.raises(ValueError):
        SubprocessVerifier(command_args=["echo", "ok"], require_placeholder=True)


def test_subprocess_verifier_uses_argv_shell_false_and_placeholder_substitution(monkeypatch):
    """Subprocess verifier must execute argv without a shell."""
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    verifier = SubprocessVerifier(
        command_args=["cat", "{file}"],
        file_extension=".py",
    )
    result = verifier.verify("print('hello')")

    assert result.success is True
    assert captured["cmd"][0] == "cat"
    assert "{file}" not in captured["cmd"][1]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["cwd"]


def test_execution_verifier_test_case_run_passes_resource_limit_hook(monkeypatch):
    """ExecutionVerifier must pass the limit hook for per-test execution."""
    verifier = ExecutionVerifier(test_cases=[])
    sentinel_preexec = lambda: None

    def fake_limit(timeout_seconds):
        assert timeout_seconds == 3
        return sentinel_preexec

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="42\n", stderr="")

    monkeypatch.setattr(verifier, "_build_limit_preexec", fake_limit)
    monkeypatch.setattr("subprocess.run", fake_run)

    tc = SimpleNamespace(input="", expected="42", name="t1", timeout=3)
    outcome = verifier._run_test_case("/tmp/fake_binary", tc)

    assert outcome["passed"] is True
    assert captured["cmd"] == ["/tmp/fake_binary"]
    assert captured["kwargs"]["preexec_fn"] is sentinel_preexec
    assert captured["kwargs"]["timeout"] == 3


def test_mingw_benchmark_factory_does_not_pass_run_after_compile(monkeypatch):
    """MinGW verifier construction should remain compile-only."""
    try:
        import halo_forge.benchmark as benchmark_module
        from halo_forge.rlvr import verifiers as verifiers_module
    except ModuleNotFoundError as e:
        if e.name == "torch":
            pytest.skip("torch not installed; skipping benchmark factory test")
        raise

    captured = {}

    class FakeMinGWVerifier:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(verifiers_module, "MinGWVerifier", FakeMinGWVerifier, raising=True)

    benchmark_module._get_verifier_for_language(
        language="cpp",
        verifier_type="mingw",
        run_after_compile=True,
    )

    assert "run_after_compile" not in captured["kwargs"]
    assert captured["kwargs"]["timeout"] == 30


def test_multi_language_optional_kwarg_propagation_by_signature(monkeypatch):
    """Optional kwargs should only be passed when constructor supports them."""
    from halo_forge.rlvr import verifiers as verifiers_module

    captured = {}

    class SupportsVerifier:
        def __init__(self, max_workers=8, run_after_compile=False, binary_cache_dir=None):
            captured["supports"] = {
                "max_workers": max_workers,
                "run_after_compile": run_after_compile,
                "binary_cache_dir": binary_cache_dir,
            }

        def verify(self, code):
            return VerifyResult(success=True, reward=1.0, details="ok")

    class PlainVerifier:
        def __init__(self, max_workers=8):
            captured["plain"] = {"max_workers": max_workers}

        def verify(self, code):
            return VerifyResult(success=True, reward=1.0, details="ok")

    monkeypatch.setattr(verifiers_module, "SupportsVerifier", SupportsVerifier, raising=False)
    monkeypatch.setattr(verifiers_module, "PlainVerifier", PlainVerifier, raising=False)

    configs = {
        "supported": LanguageConfig(
            name="supported",
            patterns=[r"^support"],
            verifier_class="SupportsVerifier",
            priority=10,
        ),
        "plain": LanguageConfig(
            name="plain",
            patterns=[r"^plain"],
            verifier_class="PlainVerifier",
            priority=9,
        ),
    }

    verifier = MultiLanguageVerifier(
        language_configs=configs,
        default_language="plain",
        run_after_compile=True,
        binary_cache_dir="cache/bin",
    )

    verifier._get_verifier("supported")
    verifier._get_verifier("plain")

    assert captured["supports"]["run_after_compile"] is True
    assert captured["supports"]["binary_cache_dir"] == "cache/bin"
    assert "run_after_compile" not in captured["plain"]
    assert "binary_cache_dir" not in captured["plain"]

    result = verifier.verify("plain sample", language="plain")
    init_metadata = result.metadata["verifier_init"]
    assert "run_after_compile" in init_metadata["skipped_optional_kwargs"]
    assert "binary_cache_dir" in init_metadata["skipped_optional_kwargs"]


@pytest.mark.parametrize(
    "ui_file",
    [
        Path("ui/pages/datasets.py"),
        Path("ui/pages/verifiers.py"),
    ],
)
def test_ui_pages_do_not_use_unsanitized_html(ui_file):
    """Dynamic UI pages should not use sanitize=False rendering."""
    content = ui_file.read_text(encoding="utf-8")
    assert "sanitize=False" not in content


def test_planning_docs_policy_guards():
    """Tracked docs should not include internal planning artifacts."""
    if not shutil.which("git"):
        pytest.skip("git not available")

    tracked_docs = subprocess.run(
        ["git", "ls-files", "docs"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked_docs.returncode == 0

    blocked_patterns = re.compile(r"(plan|sub[_-]?plan|roadmap)", re.IGNORECASE)
    violations = [
        path for path in tracked_docs.stdout.splitlines()
        if blocked_patterns.search(Path(path).name)
    ]
    assert violations == []

    ignored_internal_docs = subprocess.run(
        ["git", "check-ignore", ".internal_docs"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ignored_internal_docs.returncode == 0

    master_plan_tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "docs/MASTER_PLAN.md"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert master_plan_tracked.returncode != 0
