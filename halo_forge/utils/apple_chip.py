"""Apple Silicon chip parsing helpers."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, replace
from typing import Literal, Optional


ChipVariant = Literal["base", "Pro", "Max", "Ultra"]


@dataclass(frozen=True)
class ChipInfo:
    generation: int
    variant: ChipVariant | None
    gpu_cores: int | None = None
    nominal_memory_bandwidth_gbps: int | None = None

    @property
    def brand(self) -> str:
        suffix = "" if self.variant in (None, "base") else f" {self.variant}"
        return f"M{self.generation}{suffix}"

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "brand": self.brand,
            "generation": self.generation,
            "variant": self.variant,
            "gpu_cores": self.gpu_cores,
            "nominal_memory_bandwidth_gbps": self.nominal_memory_bandwidth_gbps,
        }


_BANDWIDTH_GBPS: dict[tuple[int, ChipVariant], int] = {
    (1, "base"): 68,
    (1, "Pro"): 200,
    (1, "Max"): 400,
    (1, "Ultra"): 800,
    (2, "base"): 100,
    (2, "Pro"): 200,
    (2, "Max"): 400,
    (2, "Ultra"): 800,
    (3, "base"): 100,
    (3, "Pro"): 150,
    (3, "Max"): 300,
    (4, "base"): 120,
    (4, "Pro"): 273,
    (4, "Max"): 546,
    (4, "Ultra"): 819,
    (5, "base"): 120,
    (5, "Pro"): 273,
    (5, "Max"): 546,
}


def parse_chip_brand(brand: str | None) -> ChipInfo | None:
    """Parse Apple chip names like ``Apple M3 Max`` into structured fields."""
    if not brand:
        return None
    match = re.search(r"\bApple\s+M([1-5])(?:\s+(Pro|Max|Ultra))?\b", brand.strip())
    if not match:
        return None
    generation = int(match.group(1))
    variant = (match.group(2) or "base")
    key = (generation, variant)
    if key not in _BANDWIDTH_GBPS:
        return None
    return ChipInfo(
        generation=generation,
        variant=variant,  # type: ignore[arg-type]
        nominal_memory_bandwidth_gbps=_BANDWIDTH_GBPS[key],
    )


def with_gpu_cores(chip: ChipInfo | None, gpu_cores: int | None) -> ChipInfo | None:
    if chip is None or gpu_cores is None:
        return chip
    return replace(chip, gpu_cores=gpu_cores)


def detect_apple_gpu_cores() -> int | None:
    info = _system_profiler_display_json()
    if not info:
        return None
    candidates: list[str] = []
    for section in info.get("SPDisplaysDataType", []) or []:
        if isinstance(section, dict):
            for key in ("sppci_cores", "sppci_gpu_cores", "spdisplays_cores"):
                value = section.get(key)
                if value is not None:
                    candidates.append(str(value))
    for candidate in candidates:
        match = re.search(r"\d+", candidate)
        if match:
            return int(match.group(0))
    return None


def detect_metal_version() -> str | None:
    info = _system_profiler_display_json()
    if not info:
        return None
    for section in info.get("SPDisplaysDataType", []) or []:
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if "metal" in str(key).lower() and value:
                return str(value)
    return None


def _system_profiler_display_json() -> dict | None:
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
