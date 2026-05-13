"""MLX runtime readiness probe.

The important distinction is "MLX imports" vs. "MLX can execute on this
process host."  The latter can fail in headless/sandboxed macOS sessions even
when packages are installed, so the executable probe runs in a subprocess to
protect API/CLI callers from MLX/Metal aborts.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ReadinessStatus = Literal["ready", "unavailable", "error"]


@dataclass(frozen=True)
class MLXReadiness:
    status: ReadinessStatus
    executable: bool
    package_versions: dict[str, str | None]
    chip: dict[str, Any] | None
    macos_version: str | None
    metal_device: dict[str, Any] | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggested_fixes: list[str] = field(default_factory=list)
    probe: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _chip_info() -> dict[str, Any] | None:
    try:
        from halo_forge.telemetry.apple_silicon import AppleSiliconTelemetry
        from halo_forge.utils.apple_chip import parse_chip_brand

        raw = AppleSiliconTelemetry._detect_device_name()
        parsed = parse_chip_brand(raw)
        if parsed is None:
            return {"brand": raw} if raw else None
        data = parsed.to_dict()
        data["raw_brand"] = raw
        return data
    except Exception:
        return None


def _metal_device() -> dict[str, Any] | None:
    if platform.system() != "Darwin":
        return None
    try:
        proc = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None
    items = payload.get("SPDisplaysDataType")
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        model = item.get("sppci_model") or item.get("_name")
        if model:
            return {
                "model": model,
                "gpu_cores": _optional_int(item.get("sppci_cores")),
                "metal_supported": _metal_supported(item.get("spdisplays_metal")),
            }
    return None


def _metal_supported(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if "unsupported" in text or "not supported" in text:
        return False
    if "supported" in text or "metal" in text:
        return True
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group(0))
    try:
        return int(value)
    except Exception:
        return None


def _is_metal_unavailable(text: str) -> bool:
    lowered = text.lower()
    return any(
        needle in lowered
        for needle in (
            "no metal device available",
            "metal::load_device",
            "nsrangeexception",
            "metalallocator",
        )
    )


def _probe_mlx(timeout_seconds: float) -> tuple[int, str, str]:
    code = """
import json
import mlx.core as mx
x = mx.array([1.0, 2.0])
y = (x * 2).sum()
mx.eval(y)
print(json.dumps({"default_device": str(mx.default_device()), "value": float(y.item())}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def check_mlx_readiness(*, timeout_seconds: float = 10.0) -> MLXReadiness:
    versions = {"mlx": _package_version("mlx"), "mlx-lm": _package_version("mlx-lm")}
    chip = _chip_info()
    macos_version = platform.mac_ver()[0] or None
    metal = _metal_device()
    should_enrich_chip = bool(
        metal
        and (
            not chip
            or str(chip.get("brand") or "").lower() in {"arm", "apple silicon"}
            or chip.get("gpu_cores") is None
        )
    )
    if should_enrich_chip:
        try:
            from halo_forge.utils.apple_chip import parse_chip_brand, with_gpu_cores

            gpu_cores = _optional_int(metal.get("gpu_cores"))
            parsed = parse_chip_brand(str(metal.get("model") or ""))
            if parsed is not None:
                enriched = with_gpu_cores(parsed, gpu_cores)
                if enriched is not None:
                    chip = enriched.to_dict()
                    chip["raw_brand"] = metal.get("model")
            elif chip and gpu_cores is not None:
                chip = dict(chip)
                chip["gpu_cores"] = gpu_cores
        except Exception:
            pass

    missing = [name for name, version in versions.items() if version is None]
    if missing:
        return MLXReadiness(
            status="unavailable",
            executable=False,
            package_versions=versions,
            chip=chip,
            macos_version=macos_version,
            metal_device=metal,
            errors=[f"Missing package(s): {', '.join(missing)}"],
            suggested_fixes=["Install MLX support with `pip install -e '.[mlx]'`."],
        )

    try:
        returncode, stdout, stderr = _probe_mlx(timeout_seconds)
    except subprocess.TimeoutExpired:
        return MLXReadiness(
            status="error",
            executable=False,
            package_versions=versions,
            chip=chip,
            macos_version=macos_version,
            metal_device=metal,
            errors=["MLX executable probe timed out."],
            suggested_fixes=["Retry from a normal Terminal session and check Activity Monitor for stuck Python processes."],
        )
    except Exception as exc:
        return MLXReadiness(
            status="error",
            executable=False,
            package_versions=versions,
            chip=chip,
            macos_version=macos_version,
            metal_device=metal,
            errors=[f"MLX executable probe failed: {exc}"],
        )

    combined = "\n".join(part for part in (stdout, stderr) if part)
    if returncode == 0:
        probe: dict[str, Any] = {}
        try:
            probe = json.loads(stdout.strip().splitlines()[-1])
        except Exception:
            probe = {"stdout": stdout.strip()}
        default_device = str(probe.get("default_device") or "").lower()
        if metal and metal.get("metal_supported") is None and "gpu" in default_device:
            metal = dict(metal)
            metal["metal_supported"] = True
        return MLXReadiness(
            status="ready",
            executable=True,
            package_versions=versions,
            chip=chip,
            macos_version=macos_version,
            metal_device=metal,
            warnings=[],
            suggested_fixes=["Use `halo-forge --accelerator mlx ...` for MLX-native training and serving."],
            probe=probe,
        )

    status: ReadinessStatus = "unavailable" if _is_metal_unavailable(combined) else "error"
    fixes = [
        "Run from a normal Terminal session with GPU access.",
        "Verify MLX with `.venv/bin/python -c \"import mlx.core as mx; x = mx.array([1.0]); mx.eval(x)\"`.",
    ]
    if status == "error":
        fixes.append("Reinstall MLX support with `pip install -e '.[mlx]'`.")
    return MLXReadiness(
        status=status,
        executable=False,
        package_versions=versions,
        chip=chip,
        macos_version=macos_version,
        metal_device=metal,
        errors=[combined.strip() or f"MLX probe exited with code {returncode}"],
        warnings=["MLX packages are installed, but this process cannot execute MLX arrays."],
        suggested_fixes=fixes,
        probe={"returncode": returncode},
    )


__all__ = ["MLXReadiness", "check_mlx_readiness"]
