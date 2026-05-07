"""Verifier plugin registry (Track V1).

Today: users subclass `Verifier` and manually import in their train script.
Tomorrow: `@register_verifier("my_check")` + drop a `.py` into
`~/.halo-forge/verifiers/` and the trainer / CLI / public API see it
automatically.

Three registration paths, all funnel into the same dict:

1. **Decorator** (programmatic):
       @register_verifier("regex_format")
       class MyVerifier(Verifier): ...

2. **Plugin directory** (no edits required):
       cp my_check.py ~/.halo-forge/verifiers/
   Files there are imported on first registry access; any class
   subclassing `Verifier` and decorated with `@register_verifier(...)`
   is picked up.

3. **Entry points** (pip-installable plugin packages):
       [project.entry-points."halo_forge.verifiers"]
       my_check = "my_pkg.verifiers:MyVerifier"
   Discovered via `importlib.metadata.entry_points()`.

The CLI / RAFT trainer / public-API consumers should call
`get_verifier(name)` instead of importing concrete classes directly.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Type

from halo_forge.rlvr.verifiers.base import Verifier

logger = logging.getLogger(__name__)


# Internal registry. Maps name -> verifier class. Names are lowercase
# strings, conventionally snake_case.
_REGISTRY: Dict[str, Type[Verifier]] = {}

# Track whether we've done the lazy load of plugin directory + entry
# points. Cheap to redo on cache invalidation, but noisy if every lookup
# hits the filesystem.
_DISCOVERED = False


def _plugin_dir() -> Path:
    """Where user-dropped plugin .py files live.

    Override with `HALOFORGE_VERIFIERS_DIR` for tests / non-default homes.
    """
    override = os.environ.get("HALOFORGE_VERIFIERS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".halo-forge" / "verifiers"


def register_verifier(name: str) -> Callable[[Type[Verifier]], Type[Verifier]]:
    """Decorator that registers a verifier class under ``name``.

    Usage:
        @register_verifier("regex_format")
        class RegexFormatVerifier(Verifier):
            ...

    Re-registration with the same name overwrites — useful when
    iterating in a notebook; loud if the same plugin is loaded twice
    via different paths so we log a warning.
    """
    if not name or not isinstance(name, str):
        raise ValueError("verifier name must be a non-empty string")
    canonical = name.strip().lower()

    def _decorator(cls: Type[Verifier]) -> Type[Verifier]:
        if not isinstance(cls, type) or not issubclass(cls, Verifier):
            raise TypeError(
                f"@register_verifier({name!r}) target must subclass Verifier; "
                f"got {cls!r}"
            )
        if canonical in _REGISTRY and _REGISTRY[canonical] is not cls:
            logger.warning(
                "Verifier name collision: %r was %s, replacing with %s",
                canonical,
                _REGISTRY[canonical].__name__,
                cls.__name__,
            )
        _REGISTRY[canonical] = cls
        return cls

    return _decorator


def get_verifier(name: str) -> Type[Verifier]:
    """Return the registered verifier class for ``name``.

    Triggers lazy plugin-directory + entry-point discovery on first call.
    Raises KeyError with the list of known names on miss so the operator
    can correct a typo without grepping the codebase.
    """
    canonical = (name or "").strip().lower()
    _ensure_discovered()
    if canonical not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(
            f"Unknown verifier {name!r}. Registered verifiers: {known}. "
            f"Drop a .py into {_plugin_dir()} or use @register_verifier."
        )
    return _REGISTRY[canonical]


def list_registered_verifiers() -> List[str]:
    """Return the sorted list of registered verifier names."""
    _ensure_discovered()
    return sorted(_REGISTRY)


def reset_registry_for_tests() -> None:
    """Clear the registry. Test-only — production code should never call this."""
    global _DISCOVERED
    _REGISTRY.clear()
    _DISCOVERED = False


def _ensure_discovered() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True  # set first so re-entry from importlib doesn't recurse
    _load_plugin_directory(_plugin_dir())
    _load_entry_points()


def _load_plugin_directory(path: Path) -> None:
    """Import every .py in `path` so its `@register_verifier` calls fire.

    Errors loading individual files are logged and skipped — a broken
    user plugin should not take down the whole training launch.
    """
    if not path.exists() or not path.is_dir():
        return
    for entry in sorted(path.glob("*.py")):
        if entry.name.startswith("_"):
            continue
        module_name = f"halo_forge_user_verifier_{entry.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, entry)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - logged at runtime
            logger.warning(
                "Skipping verifier plugin %s: %s",
                entry,
                exc,
            )


def _load_entry_points() -> None:
    """Discover verifier classes published via `halo_forge.verifiers` entry points.

    Each entry point should resolve to a `Verifier` subclass; the entry
    point's name becomes the registry key (with `register_verifier`
    semantics). This lets pip-installed plugin packages register without
    touching the filesystem.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - py<3.8, not supported anyway
        return
    try:
        eps = entry_points(group="halo_forge.verifiers")
    except TypeError:
        # py<3.10 returns a dict-like; pluck the group manually.
        all_eps = entry_points()
        eps = all_eps.get("halo_forge.verifiers", [])  # type: ignore[union-attr]
    for ep in eps:
        try:
            cls = ep.load()
            if not isinstance(cls, type) or not issubclass(cls, Verifier):
                logger.warning(
                    "Entry point %r resolved to non-Verifier %r — skipping",
                    ep.name,
                    cls,
                )
                continue
            _REGISTRY[ep.name.strip().lower()] = cls
        except Exception as exc:  # pragma: no cover - logged at runtime
            logger.warning("Failed to load verifier entry point %r: %s", ep.name, exc)


def _seed_builtin_registrations() -> None:
    """Register the verifiers halo-forge ships under canonical short names.

    Called from `halo_forge.rlvr.verifiers` package init so the registry
    is non-empty out of the box. Idempotent — safe to call multiple times.
    """
    # Lazy imports break the import cycle with `verifiers/__init__.py`.
    from halo_forge.rlvr.verifiers.compile import (
        ClangVerifier,
        GCCVerifier,
        MinGWVerifier,
    )
    from halo_forge.rlvr.verifiers.custom import CustomVerifier, SubprocessVerifier
    from halo_forge.rlvr.verifiers.execution import (
        ClangExecutionVerifier,
        ExecutionVerifier,
        GCCExecutionVerifier,
        MinGWExecutionVerifier,
    )
    from halo_forge.rlvr.verifiers.go_verifier import GoVerifier
    from halo_forge.rlvr.verifiers.llm_judge import LLMJudgeVerifier
    from halo_forge.rlvr.verifiers.metrics import (
        BLEUVerifier,
        ChrFVerifier,
        ROUGEVerifier,
    )
    from halo_forge.rlvr.verifiers.pytest_verifier import (
        HumanEvalVerifier,
        MBPPVerifier,
        RLVRPytestVerifier,
    )
    from halo_forge.rlvr.verifiers.rust_verifier import CargoVerifier, RustVerifier
    from halo_forge.rlvr.verifiers.schema import (
        JSONSchemaVerifier,
        JSONStructureVerifier,
        RegexFormatVerifier,
    )
    from halo_forge.rlvr.verifiers.test_runner import PytestVerifier, UnittestVerifier

    builtins: Iterable[tuple[str, Type[Verifier]]] = (
        # Compile
        ("gcc", GCCVerifier),
        ("clang", ClangVerifier),
        ("mingw", MinGWVerifier),
        # Execution
        ("execution", ExecutionVerifier),
        ("gcc_execution", GCCExecutionVerifier),
        ("clang_execution", ClangExecutionVerifier),
        ("mingw_execution", MinGWExecutionVerifier),
        # Test runners
        ("pytest", PytestVerifier),
        ("unittest", UnittestVerifier),
        ("rlvr_pytest", RLVRPytestVerifier),
        ("humaneval", HumanEvalVerifier),
        ("mbpp", MBPPVerifier),
        # Custom / subprocess
        ("custom", CustomVerifier),
        ("subprocess", SubprocessVerifier),
        # Other languages
        ("rust", RustVerifier),
        ("cargo", CargoVerifier),
        ("go", GoVerifier),
        # LLM-as-judge (Track V2)
        ("llm_judge", LLMJudgeVerifier),
        # Schema verifiers (Track V3)
        ("json_structure", JSONStructureVerifier),
        ("json_schema", JSONSchemaVerifier),
        ("regex_format", RegexFormatVerifier),
        # Reference-metric verifiers (Track V4)
        ("bleu", BLEUVerifier),
        ("rouge", ROUGEVerifier),
        ("chrf", ChrFVerifier),
    )
    for name, cls in builtins:
        if name not in _REGISTRY:
            _REGISTRY[name] = cls


__all__ = [
    "register_verifier",
    "get_verifier",
    "list_registered_verifiers",
    "reset_registry_for_tests",
]
