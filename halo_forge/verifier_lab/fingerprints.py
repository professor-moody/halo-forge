"""Verifier implementation and sanitized-configuration fingerprints."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Type
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from halo_forge.rlvr.verifiers.base import Verifier


_SECRET_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|token|password|passwd|secret|authorization|credential|cookie)($|[_-])",
    re.IGNORECASE,
)
_REDACTED = "<redacted>"


_EXECUTABLE_VERSION_ARGS = {
    "cargo": ("--version",),
    "clang": ("--version",),
    "gcc": ("--version",),
    "g++": ("--version",),
    "go": ("version",),
    "rustc": ("--version",),
    "x86_64-w64-mingw32-g++": ("--version",),
}


@dataclass(frozen=True)
class ImplementationFingerprint:
    verifier_name: str
    class_path: str
    origin: str
    fingerprint: Optional[str]
    source_hash: Optional[str]
    distribution: Optional[str] = None
    distribution_version: Optional[str] = None
    qualifiable: bool = False
    reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _scrub_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc:
        return value
    hostname = parts.hostname or ""
    if parts.port is not None:
        hostname = f"{hostname}:{parts.port}"
    if parts.username is not None or parts.password is not None:
        hostname = f"{_REDACTED}@{hostname}"
    query = []
    for key, item in parse_qsl(parts.query, keep_blank_values=True):
        query.append((key, _REDACTED if _SECRET_KEY.search(key) else item))
    return urlunsplit((parts.scheme, hostname, parts.path, urlencode(query), parts.fragment))


def sanitize_configuration(value: Any, *, _key: str = "") -> Any:
    """Recursively remove credentials while retaining reproducibility fields."""

    if _key and _SECRET_KEY.search(_key):
        return _REDACTED
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        result: MutableMapping[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            result[key_text] = sanitize_configuration(item, _key=key_text)
        return dict(result)
    if isinstance(value, (list, tuple)):
        return [sanitize_configuration(item, _key=_key) for item in value]
    if isinstance(value, set):
        return sorted(sanitize_configuration(item, _key=_key) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _scrub_url(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return repr(value)


def configuration_hash(value: Any) -> str:
    """Hash the sanitized form; credentials never influence persisted identity."""

    return _sha256_bytes(_canonical_json(sanitize_configuration(value)).encode("utf-8"))


def runtime_identity(extra: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Return observable runtime facts without fabricating accelerator data."""

    # Measured fields are applied after caller context so callers cannot spoof
    # Python, platform, machine, executable, or the derived identity hash.
    context = sanitize_configuration(extra or {})
    result: dict[str, Any] = dict(context) if isinstance(context, Mapping) else {}
    result.update({
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": str(Path(sys.executable).resolve()),
    })
    result["identity_hash"] = configuration_hash(result)
    return result


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            # An unavailable package is not represented as an invented
            # version. If a previously pinned package disappears, recursive
            # contract comparison reports the missing key as runtime drift.
            continue
    return result


def _executable_identities(names: tuple[str, ...]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name in names:
        path = shutil.which(name)
        if not path:
            continue
        resolved = str(Path(path).resolve())
        command = [resolved, *_EXECUTABLE_VERSION_ARGS.get(name, ("--version",))]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        lines = [
            line.strip()
            for line in (completed.stdout or completed.stderr or "").splitlines()
            if line.strip()
        ]
        if not lines:
            continue
        result[name] = {"path": resolved, "version": lines[0][:500]}
    return result


def _hardware_identity(*, include_accelerator: bool) -> dict[str, Any]:
    result: dict[str, Any] = {"machine": platform.machine()}
    processor = platform.processor().strip()
    if processor:
        result["processor"] = processor
    if not include_accelerator:
        return result

    # The backend is an observed capability. Device details are included only
    # when the active runtime exposes them; unavailable GPU metrics are never
    # inferred from the host architecture.
    try:
        from halo_forge.utils.accelerator import detect_gpu_kind

        backend = detect_gpu_kind()
    except Exception:
        return result
    accelerator: dict[str, Any] = {"backend": backend}
    if backend in {"cuda", "rocm_gfx1151"}:
        try:
            import torch

            name = str(torch.cuda.get_device_name(0) or "").strip()
            if name:
                accelerator["device_name"] = name
            cuda_version = str(getattr(torch.version, "cuda", None) or "").strip()
            hip_version = str(getattr(torch.version, "hip", None) or "").strip()
            if cuda_version:
                accelerator["cuda_version"] = cuda_version
            if hip_version:
                accelerator["hip_version"] = hip_version
        except Exception:
            pass
    result["accelerator"] = accelerator
    return result


def runtime_contract_snapshot(
    *,
    package_names: tuple[str, ...] = ("halo-forge",),
    executable_names: tuple[str, ...] = (),
    include_accelerator: bool = False,
) -> dict[str, Any]:
    """Capture the stable, observable portion of the current verifier runtime.

    This is intentionally narrower than a telemetry snapshot: changing Python,
    the platform, a pinned package/tool executable, or relevant accelerator
    identity should make qualification stale, while transient memory and load
    measurements must not.
    """

    return {
        "schema_version": 1,
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "toolchain": {
            "packages": _package_versions(tuple(sorted(set(package_names)))),
            "executables": _executable_identities(
                tuple(sorted(set(executable_names)))
            ),
        },
        "hardware": _hardware_identity(include_accelerator=include_accelerator),
    }


def _merge_runtime_override(target: dict[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(target.get(str(key)), dict):
            _merge_runtime_override(target[str(key)], value)
        else:
            target[str(key)] = sanitize_configuration(value, _key=str(key))


def runtime_identity_for_contract(
    expected: Mapping[str, Any],
    override: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Observe exactly the fields selected by a stored runtime contract."""

    if not (
        int(expected.get("schema_version") or 0) == 1
        and isinstance(expected.get("runtime"), Mapping)
        and isinstance(expected.get("toolchain"), Mapping)
        and isinstance(expected.get("hardware"), Mapping)
    ):
        return runtime_identity(override or {})
    toolchain = expected.get("toolchain") or {}
    package_names = tuple(str(key) for key in (toolchain.get("packages") or {}))
    executable_names = tuple(
        str(key) for key in (toolchain.get("executables") or {})
    )
    hardware = expected.get("hardware") or {}
    observed = runtime_contract_snapshot(
        package_names=package_names,
        executable_names=executable_names,
        include_accelerator="accelerator" in hardware,
    )
    if override:
        _merge_runtime_override(observed, override)
    return observed


def _distribution_for_module(module_name: str) -> tuple[Optional[str], Optional[str]]:
    top_level = module_name.split(".", 1)[0]
    try:
        candidates = metadata.packages_distributions().get(top_level, [])
    except Exception:
        candidates = []
    if not candidates and top_level == "halo_forge":
        candidates = ["halo-forge"]
    for candidate in sorted(candidates):
        try:
            return candidate, metadata.version(candidate)
        except metadata.PackageNotFoundError:
            continue
    return None, None


def _source_bytes(cls: type[Any]) -> tuple[Optional[bytes], Optional[str]]:
    try:
        path = inspect.getsourcefile(cls) or inspect.getfile(cls)
    except (OSError, TypeError):
        path = None
    if path:
        try:
            return Path(path).read_bytes(), str(Path(path).resolve())
        except OSError:
            pass
    try:
        return inspect.getsource(cls).encode("utf-8"), None
    except (OSError, TypeError):
        return None, None


def fingerprint_verifier_class(
    verifier_name: str,
    cls: type[Any],
    *,
    origin: Optional[str] = None,
) -> ImplementationFingerprint:
    """Fingerprint a built-in, file plugin, or installed entry point.

    User plugins are pinned by the complete file content.  Installed entry
    points include distribution/version and implementation source where it is
    inspectable.  An opaque implementation remains visible but cannot satisfy
    normal qualification policy.
    """

    # Several shipped pre-V1 verifiers intentionally predate the ``Verifier``
    # ABC but expose the same ``verify`` contract. Reliability wrapping must
    # not break those plugins, so fingerprint the legacy duck-typed class while
    # still refusing objects that are not executable verifiers.
    if not isinstance(cls, type) or not callable(getattr(cls, "verify", None)):
        raise TypeError("verifier implementation must expose verify()")
    name = str(verifier_name).strip().lower()
    module_name = str(getattr(cls, "__module__", ""))
    class_path = f"{module_name}.{cls.__qualname__}" if module_name else cls.__qualname__
    resolved_origin = origin
    if resolved_origin is None:
        if module_name.startswith("halo_forge_user_verifier_"):
            resolved_origin = "user_plugin"
        elif module_name.startswith("halo_forge."):
            resolved_origin = "builtin"
        else:
            resolved_origin = "entry_point"

    source, source_path = _source_bytes(cls)
    source_hash = _sha256_bytes(source) if source is not None else None
    distribution, distribution_version = _distribution_for_module(module_name)
    payload = {
        "name": name,
        "class_path": class_path,
        "origin": resolved_origin,
        "source_hash": source_hash,
        "source_path_name": Path(source_path).name if source_path else None,
        "distribution": distribution,
        "distribution_version": distribution_version,
    }
    qualifiable = source_hash is not None
    if resolved_origin == "entry_point":
        qualifiable = qualifiable and distribution is not None and distribution_version is not None
    reason = None if qualifiable else "implementation_source_or_distribution_unavailable"
    fingerprint = configuration_hash(payload) if qualifiable else None
    return ImplementationFingerprint(
        verifier_name=name,
        class_path=class_path,
        origin=resolved_origin,
        fingerprint=fingerprint,
        source_hash=source_hash,
        distribution=distribution,
        distribution_version=distribution_version,
        qualifiable=qualifiable,
        reason=reason,
    )


def fingerprint_registered_verifier(verifier_name: str) -> ImplementationFingerprint:
    from halo_forge.rlvr.verifiers.registry import get_verifier, inventory

    canonical = str(verifier_name).strip().lower()
    origin_by_name = {str(item["name"]): str(item["origin"]) for item in inventory()}
    return fingerprint_verifier_class(
        canonical,
        get_verifier(canonical),
        origin=origin_by_name.get(canonical),
    )


__all__ = [
    "ImplementationFingerprint",
    "configuration_hash",
    "fingerprint_registered_verifier",
    "fingerprint_verifier_class",
    "runtime_contract_snapshot",
    "runtime_identity",
    "runtime_identity_for_contract",
    "sanitize_configuration",
]
