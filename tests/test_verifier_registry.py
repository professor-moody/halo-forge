"""Verifier plugin registry tests (Track V1)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_registry(monkeypatch, tmp_path):
    """Each test gets a clean registry + an isolated plugin directory.

    `reset_registry_for_tests` clears the dict; `HALOFORGE_VERIFIERS_DIR`
    redirects discovery away from the user's real `~/.halo-forge/verifiers`.

    Teardown re-seeds the built-in verifier set. Without this, any test
    file that runs after this module sees an empty registry — the
    `__init__` seed only fires on first module import, which already
    happened before the fixture cleared the dict.
    """
    from halo_forge.rlvr.verifiers import registry as reg

    reg.reset_registry_for_tests()
    plugin_dir = tmp_path / "verifiers"
    plugin_dir.mkdir()
    monkeypatch.setenv("HALOFORGE_VERIFIERS_DIR", str(plugin_dir))
    yield plugin_dir
    reg.reset_registry_for_tests()
    reg._seed_builtin_registrations()


def test_decorator_registers_class():
    from halo_forge.rlvr.verifiers import (
        Verifier,
        VerifyResult,
        get_verifier,
        register_verifier,
    )

    @register_verifier("dummy")
    class _Dummy(Verifier):
        def verify(self, code: str) -> VerifyResult:  # pragma: no cover
            return VerifyResult(success=True, reward=1.0, details="ok")

    assert get_verifier("dummy") is _Dummy
    # Case-insensitive lookup.
    assert get_verifier("DUMMY") is _Dummy
    assert get_verifier(" dummy ") is _Dummy


def test_unknown_name_raises_with_hint():
    from halo_forge.rlvr.verifiers import (
        Verifier,
        VerifyResult,
        get_verifier,
        register_verifier,
    )

    @register_verifier("known_one")
    class _Known(Verifier):
        def verify(self, code: str) -> VerifyResult:  # pragma: no cover
            return VerifyResult(success=True, reward=1.0, details="")

    with pytest.raises(KeyError) as ei:
        get_verifier("does_not_exist")
    msg = str(ei.value)
    assert "does_not_exist" in msg
    assert "known_one" in msg  # the listing of registered names
    # The hint mentions both registration paths so a confused operator
    # can find the right knob without grepping the codebase.
    assert "Drop a .py" in msg or "register_verifier" in msg


def test_register_rejects_non_verifier_subclass():
    from halo_forge.rlvr.verifiers import register_verifier

    with pytest.raises(TypeError):

        @register_verifier("bogus")
        class _NotAVerifier:  # noqa: not a Verifier subclass
            pass


def test_register_rejects_empty_name():
    from halo_forge.rlvr.verifiers import register_verifier

    with pytest.raises(ValueError):
        register_verifier("")


def test_collision_overwrites_with_warning(caplog):
    from halo_forge.rlvr.verifiers import (
        Verifier,
        VerifyResult,
        get_verifier,
        register_verifier,
    )

    @register_verifier("shared_name")
    class _A(Verifier):
        def verify(self, code: str) -> VerifyResult:  # pragma: no cover
            return VerifyResult(success=True, reward=1.0, details="")

    with caplog.at_level("WARNING"):

        @register_verifier("shared_name")
        class _B(Verifier):
            def verify(self, code: str) -> VerifyResult:  # pragma: no cover
                return VerifyResult(success=True, reward=1.0, details="")

    assert get_verifier("shared_name") is _B
    assert any("collision" in rec.message.lower() for rec in caplog.records)


def test_plugin_directory_loads_user_py(isolate_registry):
    """A bare `.py` in the plugin dir gets imported on first lookup; its
    `@register_verifier` decoration fires and the class becomes discoverable."""
    from halo_forge.rlvr.verifiers import get_verifier, list_registered_verifiers

    plugin_dir: Path = isolate_registry
    plugin = plugin_dir / "user_check.py"
    plugin.write_text(
        textwrap.dedent(
            """
            from halo_forge.rlvr.verifiers import (
                Verifier,
                VerifyResult,
                register_verifier,
            )

            @register_verifier("from_plugin_dir")
            class FromPluginDir(Verifier):
                def verify(self, code: str) -> VerifyResult:
                    return VerifyResult(success=True, reward=1.0, details="ok")
            """
        )
    )

    assert "from_plugin_dir" in list_registered_verifiers()
    cls = get_verifier("from_plugin_dir")
    assert cls.__name__ == "FromPluginDir"


def test_underscore_files_are_skipped(isolate_registry):
    """Files like `_helpers.py` are imported by Python convention as private
    modules; the registry should not load them as plugins."""
    from halo_forge.rlvr.verifiers import list_registered_verifiers

    plugin_dir: Path = isolate_registry
    (plugin_dir / "_helpers.py").write_text(
        "raise RuntimeError('this should not load')"
    )
    # No exception means the underscore file was skipped.
    list_registered_verifiers()


def test_broken_plugin_does_not_take_down_discovery(isolate_registry, caplog):
    """A user plugin with a syntax / import error logs a warning and is
    skipped — the rest of discovery must continue."""
    from halo_forge.rlvr.verifiers import list_registered_verifiers

    plugin_dir: Path = isolate_registry
    (plugin_dir / "broken.py").write_text("import some_module_that_does_not_exist")
    (plugin_dir / "good.py").write_text(
        textwrap.dedent(
            """
            from halo_forge.rlvr.verifiers import (
                Verifier,
                VerifyResult,
                register_verifier,
            )

            @register_verifier("good_one")
            class GoodOne(Verifier):
                def verify(self, code: str) -> VerifyResult:
                    return VerifyResult(success=True, reward=1.0, details="ok")
            """
        )
    )

    with caplog.at_level("WARNING"):
        names = list_registered_verifiers()

    assert "good_one" in names
    assert any("broken" in rec.message for rec in caplog.records)


def test_seed_builtins_populates_canonical_names():
    """After importing the package init, the canonical short names for the
    halo-forge built-ins must be discoverable without a separate import."""
    from halo_forge.rlvr.verifiers import list_registered_verifiers
    from halo_forge.rlvr.verifiers.registry import _seed_builtin_registrations

    _seed_builtin_registrations()
    names = list_registered_verifiers()
    assert {"gcc", "execution", "pytest", "humaneval", "custom"}.issubset(set(names))


def test_list_returns_sorted_for_stable_ui_render():
    from halo_forge.rlvr.verifiers import (
        Verifier,
        VerifyResult,
        list_registered_verifiers,
        register_verifier,
    )

    @register_verifier("zzz_last")
    class _Z(Verifier):
        def verify(self, code: str) -> VerifyResult:  # pragma: no cover
            return VerifyResult(success=True, reward=1.0, details="")

    @register_verifier("aaa_first")
    class _A(Verifier):
        def verify(self, code: str) -> VerifyResult:  # pragma: no cover
            return VerifyResult(success=True, reward=1.0, details="")

    names = list_registered_verifiers()
    # Sort order is stable and alphabetical so the eventual UI list is
    # deterministic.
    assert names == sorted(names)
    assert "aaa_first" in names and "zzz_last" in names
